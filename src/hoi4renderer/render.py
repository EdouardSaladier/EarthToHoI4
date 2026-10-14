"""Software re-implementation of the HOI4 political map look (zoomed out)."""
import colorsys
import math
import os
import time

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from . import geometry
from .data import MapData, load_indexed, load_rgb, log
from .post import CubeMap, PostProcess, load_defines

# --- constants taken from gfx/FX/constants.fxh and common/defines/00_graphics.lua
TERRAIN_TILE_FREQ = 128.0
MAP_NUM_TILES = 4
COLORMAP_OVERLAY_STRENGTH = 0.75
BORDER_LIGHT_REMOVAL_FACTOR = 0.8
GB_OPACITY_FAR = 0.85
# GB_OPACITY_NEAR + (1 - camdist) * (FAR - NEAR) with camdist clamped to 0.2 when fully zoomed out
GB_FAR_MIX = 1.0 + (1.0 - 0.2) * (GB_OPACITY_FAR - 1.0)
WATER_LEVEL = 95.7  # heightmap value of the water plane (WATER_HEIGHT 9.5 * 10 + ~0.2)
BORDER_WIDTH = 1.5  # map pixels
GB_THICK_LOW, GB_THICK_HIGH = 5.0, 25.0
GB_FIELD_LOW, GB_FIELD_HIGH = 300.0, 9000.0

COUNTRY_BORDER_COLOR = np.array([24, 26, 43], np.float32) / 255.0
WATER_COLOR_LIGHTNESS = 0.5
# DayAmbientMap{Pos,Neg}{X,Y,Z} from gfx/FX/constants.fxh (X east, Y up, Z north)
DAY_AMBIENT_POS = [(0.1, 0.1, 0.05), (0.03, 0.03, 0.06), (0.0502, 0.05023, 0.1023)]
DAY_AMBIENT_NEG = [(0.15, 0.15, 0.15), (0.0, 0.0, 0.0), (0.03, 0.033, 0.033)]
COAST_GRADIENT = 1.0   # relative strength of the colour gradient along coastlines (same as land borders in game)


class Options:
    scale = 2.0
    fill = 0.4               # country colour opacity in the interior (0.4 = fully zoomed out, as in the shader)
    gradient = 1.0           # multiplier on gradient border thickness
    relief = 1.0             # terrain lighting strength
    lut = "gfx/world/colorcorrection_farrange.tga"  # post-effect LUT of the "max_distance" volume
    avg_luminance = 0.45     # scene average luminance for the auto exposure (MiddleGrey / avg)
    sky_strength = 1.0       # multiplier of the fresnel weight of the sky reflection on the sea
    sky_contrast = 0.1       # contrast of the reflected clouds around the sky's mean colour
    sky_period = 1400.0      # map pixels per 360 degrees of sky swept in the sea reflection
    # river ribbon width (map px) per rivers.bmp palette index (3 = narrowest .. 11 = widest), as in vanilla
    # where the colour index sets the river size; measured on in-game screenshots of Map_Tool_Asia
    river_width = 1.6        # width of index 3
    river_width_step = 0.3   # extra width per index above 3
    river_widths = None      # optional explicit {index: width} table, overrides the two values above
    river_colormap = "mod"   # water colour under rivers: 'mod' (as the game would) or 'vanilla'
    river_lod = 1            # RiverSurface_diffuse_<lod>.dds: 0 = close-up banks, 1/2 = far (dark banks)
    state_borders = True
    province_borders = False
    sea_borders = True
    country_borders = True
    rivers = True
    names = True
    name_case = "upper"
    name_color = (236, 236, 236)
    name_opacity = 0.9
    political = True
    font = None
    strip = 256


# ----------------------------------------------------------------- helpers
def _gamma(x):
    return np.power(np.clip(x, 0.0, 1.0), 0.45)


def _linear(x):
    return np.power(np.clip(x, 0.0, 1.0), 2.2)


def overlay(color, over, pct):
    """GetOverlay() from standardfuncsgfx.fxh."""
    cg = _gamma(color)
    og = _gamma(over)
    res = np.where(og < 0.5, 2.0 * og * cg, 1.0 - 2.0 * (1.0 - og) * (1.0 - cg))
    res = _linear(res)
    return color + (res - color) * pct


class Sampler:
    """Bilinear sampling of a texture covering the whole map, for output strips."""

    def __init__(self, arr, map_w, map_h, out_w, scale):
        self.arr = arr
        th, tw = arr.shape[:2]
        # output pixel centre -> map uv -> texel coordinate
        ox = (np.arange(out_w) + 0.5) / scale
        tx = ox / map_w * tw - 0.5
        x0 = np.floor(tx).astype(np.int64)
        self.wx = (tx - x0).astype(np.float32)
        self.x0 = np.clip(x0, 0, tw - 1)
        self.x1 = np.clip(x0 + 1, 0, tw - 1)
        self.th = th
        self.map_h = map_h
        self.scale = scale

    def rows(self, r0, r1, dtype=np.float32):
        oy = (np.arange(r0, r1) + 0.5) / self.scale
        ty = oy / self.map_h * self.th - 0.5
        y0 = np.floor(ty).astype(np.int64)
        wy = (ty - y0).astype(np.float32)
        y0c = np.clip(y0, 0, self.th - 1)
        y1c = np.clip(y0 + 1, 0, self.th - 1)
        a = self.arr
        top = a[y0c]
        bot = a[y1c]
        wx = self.wx
        if a.ndim == 3:
            wx = wx[:, None]
            wyb = wy[:, None, None]
        else:
            wyb = wy[:, None]
        t = top[:, self.x0].astype(dtype)
        t += (top[:, self.x1].astype(dtype) - t) * wx
        b = bot[:, self.x0].astype(dtype)
        b += (bot[:, self.x1].astype(dtype) - b) * wx
        return t + (b - t) * wyb


def load_texture(vfs, rel, mode="RGBA"):
    p = vfs.find(rel)
    if p is None:
        return None
    im = Image.open(p)
    if im.mode != mode:
        im = im.convert(mode)
    return im


def build_tiles(atlas_img, period_px):
    """Split a 4x4 terrain atlas into 16 tiles resized to the on-screen tile period."""
    w, h = atlas_img.size
    ts = w // MAP_NUM_TILES
    tiles = []
    for i in range(MAP_NUM_TILES * MAP_NUM_TILES):
        u, v = i % MAP_NUM_TILES, i // MAP_NUM_TILES
        t = atlas_img.crop((u * ts, v * ts, (u + 1) * ts, (v + 1) * ts))
        t = t.resize((period_px, period_px), Image.BOX if period_px < ts else Image.BICUBIC)
        tiles.append(np.asarray(t, dtype=np.float32) / 255.0)
    return np.stack(tiles)


# ----------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, md: MapData, opts: Options):
        self.md = md
        self.o = opts
        self.W = md.width_px
        self.H = md.height_px
        self.s = float(opts.scale)
        self.OW = max(1, int(round(self.W * self.s)))
        self.OH = max(1, int(round(self.H * self.s)))
        self.sx = self.OW / self.W
        self.sy = self.OH / self.H

    # ------------------------------------------------------------ prepare
    def prepare(self):
        md, vfs = self.md, self.md.vfs
        t0 = time.time()
        log("Preparing map layers...")
        self.height = load_indexed(md._map_file("heightmap", "heightmap.bmp"))
        terrain_idx = load_indexed(md._map_file("terrain", "terrain.bmp"))
        self.tex_index = md.terrain_texture_lut()[terrain_idx]
        del terrain_idx

        # terrain atlas tiles at the on-screen tile period (equivalent to mip selection)
        period = max(4, int(round(self.H / TERRAIN_TILE_FREQ * self.s)))
        self.period = period
        atlas = load_texture(vfs, "map/terrain/atlas0.dds", "RGBA")
        atlas_n = load_texture(vfs, "map/terrain/atlas_normal0.dds", "RGBA")
        # colour textures are sampled as sRGB -> linear by the GPU
        self.tiles = np.power(build_tiles(atlas, period)[..., :3], 2.2)
        self.tiles_n = build_tiles(atlas_n, period)
        cmap = load_texture(vfs, "map/terrain/colormap_rgb_cityemissivemask_a.dds", "RGB")
        self.colormap = np.asarray(cmap, dtype=np.uint8)
        wmap = load_texture(vfs, "map/terrain/colormap_water_0.dds", "RGB")
        self.watermap = np.asarray(wmap, dtype=np.uint8)
        npath = md.vfs.find("map/world_normal.bmp")
        if npath:
            self.normalmap = load_rgb(npath)
        else:
            self.normalmap = self._normals_from_height()

        # political layers
        owner = md.owner_map()
        # lake provinces whose terrain is above the water plane are drawn as land by the game;
        # give them to the surrounding country so they don't show as uncoloured discs
        lake_lut = md.province_type == 2
        dry_lake = (owner == 0) & lake_lut[md.province_map] & (self.height >= WATER_LEVEL - 1.0)
        if dry_lake.any() and (owner > 0).any():
            _, (iy, ix) = ndimage.distance_transform_edt(owner == 0, return_indices=True)
            owner[dry_lake] = owner[iy[dry_lake], ix[dry_lake]]
            del iy, ix
            log(f"  {int(dry_lake.sum())} px of dry lake provinces given to the surrounding country")
        self.owner = owner
        ncol = len(md.tags) + 1
        pal = np.zeros((ncol, 3), np.uint8)
        defs = load_defines(vfs)
        hue_m = defs.get("COUNTRY_COLOR_HUE_MODIFIER", 0.0)
        sat_m = defs.get("COUNTRY_COLOR_SATURATION_MODIFIER", 1.0)
        val_m = defs.get("COUNTRY_COLOR_BRIGHTNESS_MODIFIER", 1.0)
        for t, i in md.tag_index.items():
            # map colour = country colour with the COUNTRY_COLOR_*_MODIFIER defines applied in HSV
            r, g, b = (c / 255.0 for c in md.country_color[t])
            h, sat, v = colorsys.rgb_to_hsv(r, g, b)
            r, g, b = colorsys.hsv_to_rgb((h + hue_m) % 1.0, min(1.0, sat * sat_m), min(1.0, v * val_m))
            pal[i] = (int(r * 255 + 0.5), int(g * 255 + 0.5), int(b * 255 + 0.5))
        owned = owner > 0
        if not self.o.political:
            owned = np.zeros_like(owned)
        # slightly soften ownership so coasts are not pixel-stepped (premultiplied colour + alpha)
        own_f = owned.astype(np.float32)
        a_soft = cv2.GaussianBlur(own_f, (0, 0), 0.65)
        rgb_soft = cv2.GaussianBlur(pal[owner].astype(np.float32) * own_f[..., None], (0, 0), 0.65)
        self.country_rgb = np.clip(rgb_soft + 0.5, 0, 255).astype(np.uint8)
        self.country_a = np.clip(a_soft * 255 + 0.5, 0, 255).astype(np.uint8)
        del own_f, a_soft, rgb_soft
        # smoothed heightmap for the water line
        self.height_smooth = cv2.GaussianBlur(self.height.astype(np.float32), (0, 0), 0.8)

        # gradient borders: distance to the edge of the country, thickness by country size.
        # Land borders get the full gradient, coasts / unowned edges a lighter one.
        log("  gradient borders...")
        # lakes inside a country do not produce a coloured edge in game: fill them with the surrounding owner
        edge_owner = owner
        lake_px = lake_lut[md.province_map] & (owner == 0)
        if lake_px.any() and owned.any():
            _, (iy, ix) = ndimage.distance_transform_edt(owner == 0, return_indices=True)
            edge_owner = owner.copy()
            edge_owner[lake_px] = owner[iy[lake_px], ix[lake_px]]
            del iy, ix
        land_edge = np.zeros_like(owned)
        other_edge = np.zeros_like(owned)
        for a_sl, b_sl in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                           ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
            oa, ob = edge_owner[a_sl], edge_owner[b_sl]
            diff = oa != ob
            both = diff & (oa > 0) & (ob > 0)
            land_edge[a_sl] |= both
            land_edge[b_sl] |= both
            other_edge[a_sl] |= diff & ~both
            other_edge[b_sl] |= diff & ~both
        area = np.bincount(owner.ravel(), minlength=ncol).astype(np.float64)
        # GRADIENT_BORDERS_FIELD_COUNTRY_LOW/HIGH compared with the country area in pixels
        k = np.clip((area - GB_FIELD_LOW) / (GB_FIELD_HIGH - GB_FIELD_LOW), 0, 1)
        thick = np.maximum(1e-3, (GB_THICK_LOW + k * (GB_THICK_HIGH - GB_THICK_LOW)) * self.o.gradient)
        thick_px = thick[owner].astype(np.float32)
        g = np.zeros(owner.shape, np.float32)
        for edge, weight in ((land_edge, 1.0), (other_edge, COAST_GRADIENT)):
            if edge.any():
                dist = ndimage.distance_transform_edt(~edge).astype(np.float32) + 0.5
                np.maximum(g, np.clip(1.0 - dist / thick_px, 0.0, 1.0) * weight, out=g)
                del dist
        g[~owned] = 0
        self.grad = (g * 255 + 0.5).astype(np.uint8)
        del g, land_edge, other_edge, thick_px
        log(f"  done in {time.time() - t0:.1f}s")

    def _normals_from_height(self):
        h = self.height.astype(np.float32) * 0.1
        gy, gx = np.gradient(h)
        n = np.stack([-gx, np.ones_like(h), gy], axis=-1)  # (east, up, north)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        # pack like world_normal.bmp: r=x, g=z(north), b=y(up)
        packed = np.stack([n[..., 0], n[..., 2], n[..., 1]], axis=-1) * 0.5 + 0.5
        return (packed * 255).astype(np.uint8)

    # -------------------------------------------------------------- vector
    def _scale_pts(self, pts):
        # map continuous coords -> cv2 pixel-centre coords, fixed point (shift 4)
        out = np.empty_like(pts)
        out[:, 0] = pts[:, 0] * self.sx - 0.5
        out[:, 1] = pts[:, 1] * self.sy - 0.5
        return np.round(out * 16).astype(np.int32)

    def _draw_lines(self, mask, chains, thickness, value):
        if not chains:
            return
        polys = [self._scale_pts(c) for c in chains if len(c) >= 2]
        t = max(1, int(round(thickness)))
        val = int(round(value * min(1.0, thickness if thickness < 1 else 1.0)))
        for i in range(0, len(polys), 5000):
            cv2.polylines(mask, polys[i:i + 5000], False, val, t, cv2.LINE_AA, 4)

    def build_vector_layers(self):
        md, o = self.md, self.o
        s = self.s
        self.masks = {}
        t0 = time.time()
        owner = self.owner

        if o.state_borders or o.province_borders:
            states = md.state_map()
            st_owner = np.zeros(max(md.states) + 1 if md.states else 1, np.int32)
            for sid, st in md.states.items():
                st_owner[sid] = md.tag_index.get(st["owner"], 0)

            if o.province_borders:
                log("Tracing province borders...")
                prov = md.province_map.copy()
                prov[states == 0] = 0
                pst = md.prov_state

                def same_state(a, b):
                    return (a > 0) & (b > 0) & (pst[a] == pst[b])

                chains = [geometry.smooth_boundary(c) for c in geometry.label_boundaries(prov, same_state)]
                m = np.zeros((self.OH, self.OW), np.uint8)
                self._draw_lines(m, chains, max(1.0, BORDER_WIDTH * 0.25 * s), 60)
                self.masks["province"] = m
                del prov

            if o.state_borders:
                log("Tracing state borders...")

                def same_owner(a, b):
                    return (a > 0) & (b > 0) & (st_owner[a] == st_owner[b])

                chains = [geometry.smooth_boundary(c) for c in geometry.label_boundaries(states, same_owner)]
                m = np.zeros((self.OH, self.OW), np.uint8)
                self._draw_lines(m, chains, max(1.0, BORDER_WIDTH * 0.34 * s), 114)
                self.masks["state"] = m
            del states

        if o.sea_borders:
            log("Tracing sea borders...")
            sea_lut = md.province_type == 1
            sea = np.where(sea_lut[md.province_map], md.province_map, 0).astype(np.int32)

            def both_sea(a, b):
                return (a > 0) & (b > 0)

            m = np.zeros((self.OH, self.OW), np.uint8)
            if o.province_borders:  # sea province borders fade out when zoomed out, like land ones
                chains = [geometry.smooth_boundary(c) for c in geometry.label_boundaries(sea, both_sea)]
                # border_sea_0.dds: black, peak alpha 114, ~1/3 of BORDER_WIDTH
                self._draw_lines(m, chains, max(1.0, BORDER_WIDTH * 0.3 * s), 114)
            reg = md.sea_region_lut()
            if reg is not None:
                rmap = np.where(sea_lut[md.province_map], reg[md.province_map], 0).astype(np.int32)
                chains = [geometry.smooth_boundary(c) for c in geometry.label_boundaries(rmap, both_sea)]
                core = np.zeros_like(m)
                # border_sea_region_0.dds: black, peak alpha ~219, ~0.6 of BORDER_WIDTH
                self._draw_lines(core, chains, max(1.0, BORDER_WIDTH * 0.6 * s), 200)
                np.maximum(m, core, out=m)
                del core, rmap
            self.masks["sea"] = m
            del sea

        if o.country_borders:
            log("Tracing country borders...")

            def both_owned(a, b):
                return (a > 0) & (b > 0)

            chains = [geometry.smooth_boundary(c) for c in geometry.label_boundaries(owner, both_owned)]
            m = np.zeros((self.OH, self.OW), np.uint8)
            # soft shoulder + solid core (profile of border_country_0.dds)
            self._draw_lines(m, chains, BORDER_WIDTH * s, 115)
            core = np.zeros_like(m)
            self._draw_lines(core, chains, max(1.0, BORDER_WIDTH * 0.6 * s), 207)
            np.maximum(m, core, out=m)
            del core
            self.masks["country"] = m

        if o.rivers:
            log("Tracing rivers...")
            rv = load_indexed(md._map_file("rivers", "rivers.bmp"))
            lines = geometry.pixel_lines(rv < 254)
            # width class from the rivers.bmp palette index: 3 (narrowest) .. 11 (widest); 0-2 are markers
            widx = np.where((rv >= 3) & (rv < 254), rv, 3).astype(np.float64)
            centre = np.zeros((self.OH, self.OW), np.uint8)
            groups = {}
            for ys, xs in lines:
                if len(ys) < 2:
                    continue
                pts = np.stack([xs + 0.5, ys + 0.5, widx[ys, xs]], axis=1)
                pts = geometry.chaikin(pts, 2)
                cls = np.clip(np.round(pts[:, 2]), 3, 11).astype(int)
                cut = np.nonzero(np.diff(cls))[0] + 1
                k0 = 0
                for k1 in list(cut) + [len(pts)]:
                    seg = pts[max(0, k0 - 1):k1, :2]
                    if len(seg) >= 2:
                        groups.setdefault(int(cls[k0]), []).append(seg)
                    k0 = k1
            for c in sorted(groups):  # wider rivers drawn last win at confluences
                polys = [self._scale_pts(seg) for seg in groups[c]]
                for i0 in range(0, len(polys), 5000):
                    cv2.polylines(centre, polys[i0:i0 + 5000], False, c, 1, cv2.LINE_8, 4)
            self.river_centre = centre
            del rv
        log(f"  vector layers in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------- raster
    def _setup_shading(self):
        md, vfs = self.md, self.md.vfs
        d = load_defines(vfs)
        self.defines = d
        self.sun_height = d.get("SUN_HEIGHT", 600.0)
        self.sun_lat = d.get("SUN_LATITUDE", 848.0)
        self.sun_height_water = d.get("SUN_HEIGHT_WATER", 5000.0)
        self.hdr_range = d.get("LIGHT_HDR_RANGE", 1.0)
        self.cubemap_intensity = d.get("CUBEMAP_INTENSITY", 1.0)
        self.post = PostProcess(vfs, self.o.lut, avg_luminance=self.o.avg_luminance)
        cp = vfs.find("map/terrain/reflection.dds")
        self.cube = CubeMap(cp) if cp else None
        uw = load_texture(vfs, "map/terrain/underwater_terrain_0.dds", "RGB")
        self.seabed = np.power(np.asarray(uw, np.float32) / 255.0, 2.2).reshape(-1, 3).mean(0) if uw else \
            np.array([0.03, 0.04, 0.05], np.float32)
        # river ribbon profiles across the bank (RiverSurface_masks: b = side, r = alpha; diffuse: bank colour)
        rm = load_texture(vfs, "map/terrain/RiverSurface_masks.dds", "RGBA")
        rd = load_texture(vfs, f"map/terrain/RiverSurface_diffuse_{self.o.river_lod}.dds", "RGBA")
        rm = np.asarray(rm, np.float32) / 255.0 if rm else None
        rd = np.asarray(rd, np.float32) / 255.0 if rd else None
        self.river_watermap = None
        if self.o.river_colormap == "vanilla" and vfs.game:
            vp = os.path.join(vfs.game, "map", "terrain", "colormap_water_0.dds")
            if os.path.isfile(vp):
                self.river_watermap = np.asarray(Image.open(vp).convert("RGB"), dtype=np.uint8)
        self.river_side = rm[..., 2].mean(1) if rm is not None else np.linspace(0, 1, 16)
        self.river_alpha = rm[..., 0].mean(1) if rm is not None else np.linspace(1, 0, 16)
        self.river_bank = np.power(rd[..., :3], 2.2).mean(1) if rd is not None else np.tile([[0.13, 0.07, 0.03]], (64, 1))

    def _sky(self, rows, cols):
        """Reflection of the sky cubemap (reflection.dds) as seen on water.

        In game the view direction of the tilted camera is reflected into the sky; for a full-map
        orthographic image the reflected direction wanders smoothly over the upper part of the
        cubemap (mirrored tiling) so the clouds spread over the sea like on screen.
        Returns linear sky colour and the fresnel weight of pdxwater.shader (~0.5 for that camera).
        """
        per = self.o.sky_period * self.s

        def tri(x):  # mirrored repeat in [-1, 1]
            f = np.mod(x, 2.0)
            return np.where(f < 1.0, f, 2.0 - f) * 2.0 - 1.0

        u = 0.85 * tri(cols[None, :] / per + 0.37 * np.sin(rows[:, None] / per * 2.1))
        v = 0.85 * tri(rows[:, None] / (per * 0.8) + 0.29 * np.sin(cols[None, :] / per * 1.7))
        u, v = np.broadcast_arrays(u, v)
        d = np.stack([u, np.ones_like(u), v], -1)
        sky = self.cube.sample(d) if self.cube is not None else np.full(d.shape, 0.35, np.float32)
        if self.cube is not None:
            mean = self.cube.faces[2].reshape(-1, 3).mean(0)
            sky = mean + (sky - mean) * self.o.sky_contrast
        return np.power(sky, 2.2), np.full(u.shape, 0.5016, np.float32)

    def render(self):
        o = self.o
        self._setup_shading()
        post = self.post
        W, H, OW, OH = self.W, self.H, self.OW, self.OH
        base = np.empty((OH, OW, 3), np.uint8)
        colors = np.zeros((OH, OW, 4), np.uint8)
        reflection = np.zeros((OH, OW, 4), np.uint8)
        has_rivers = getattr(self, "river_centre", None) is not None
        rivers = np.zeros((OH, OW, 4), np.uint8) if has_rivers else None
        P = self.period

        smp_height = Sampler(self.height_smooth, W, H, OW, self.sx)
        smp_cmap = Sampler(self.colormap, W, H, OW, self.sx)
        smp_water = Sampler(self.watermap, W, H, OW, self.sx)
        smp_norm = Sampler(self.normalmap, W, H, OW, self.sx)
        smp_crgb = Sampler(self.country_rgb, W, H, OW, self.sx)
        smp_ca = Sampler(self.country_a, W, H, OW, self.sx)
        smp_grad = Sampler(self.grad, W, H, OW, self.sx)
        smp_rwater = None
        if getattr(self, "river_watermap", None) is not None:
            smp_rwater = Sampler(self.river_watermap, W, H, OW, self.sx)

        mx = (np.arange(OW) + 0.5) / self.sx - 0.5
        x0 = np.floor(mx).astype(np.int64)
        wx = (mx - x0).astype(np.float32)[None, :, None]
        x0c = np.clip(x0, 0, W - 1)
        x1c = np.clip(x0 + 1, 0, W - 1)
        tx = (np.arange(OW) % P)[None, :]
        cols = np.arange(OW, dtype=np.float32)

        amb_pos = np.array(DAY_AMBIENT_POS, np.float32)  # +X +Y +Z
        amb_neg = np.array(DAY_AMBIENT_NEG, np.float32)
        river_half = None
        if has_rivers:
            # ribbon half width in output pixels per width class
            river_half = np.zeros(256, np.float32)
            for c in range(3, 12):
                wmap = o.river_width + (c - 3) * o.river_width_step
                if o.river_widths and c in o.river_widths:
                    wmap = o.river_widths[c]
                river_half[c] = 0.5 * wmap * self.s
            margin = int(math.ceil(river_half.max())) + 2

        t0 = time.time()
        for r0 in range(0, OH, o.strip):
            r1 = min(OH, r0 + o.strip)
            rows = np.arange(r0, r1, dtype=np.float32)
            # --- terrain diffuse: bilinear blend of the 4 neighbouring terrain tiles (pdxmap.shader)
            my = (rows + 0.5) / self.sy - 0.5
            y0 = np.floor(my).astype(np.int64)
            wy = (my - y0).astype(np.float32)[:, None, None]
            y0c = np.clip(y0, 0, H - 1)
            y1c = np.clip(y0 + 1, 0, H - 1)
            ty = (np.arange(r0, r1) % P)[:, None]
            ti = self.tex_index
            top = ti[y0c]
            bot = ti[y1c]
            idx = (top[:, x0c], top[:, x1c], bot[:, x0c], bot[:, x1c])

            def gather(tiles):
                c00 = tiles[idx[0], ty, tx]
                c01 = tiles[idx[1], ty, tx]
                c10 = tiles[idx[2], ty, tx]
                c11 = tiles[idx[3], ty, tx]
                a = c00 + (c01 - c00) * wx
                b = c10 + (c11 - c10) * wx
                return a + (b - a) * wy

            diffuse = gather(self.tiles)
            tn_raw = gather(self.tiles_n)
            cmap = np.power(smp_cmap.rows(r0, r1) / 255.0, 2.2)
            diffuse = overlay(diffuse, cmap, COLORMAP_OVERLAY_STRENGTH)
            del cmap

            # --- normal: world normal rotated by the detail normal (RotateVectorByVector)
            wn = smp_norm.rows(r0, r1) / 255.0 - 0.5
            n = np.stack([wn[..., 0], wn[..., 2], wn[..., 1]], -1)  # (x east, y up, z north)
            n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
            t = np.stack([tn_raw[..., 0], tn_raw[..., 2], tn_raw[..., 1]], -1) - 0.5
            t /= np.linalg.norm(t, axis=-1, keepdims=True) + 1e-6
            del wn, tn_raw
            xa = np.stack([n[..., 1], -n[..., 0], np.zeros_like(n[..., 0])], -1)
            xa /= np.linalg.norm(xa, axis=-1, keepdims=True) + 1e-6
            ya = np.stack([xa[..., 1] * n[..., 2], -xa[..., 0] * n[..., 2],
                           xa[..., 0] * n[..., 1] - xa[..., 1] * n[..., 0]], -1)
            ya /= np.linalg.norm(ya, axis=-1, keepdims=True) + 1e-6
            nn = xa * t[..., 0:1] + n * t[..., 1:2] + ya * t[..., 2:3]
            nn /= np.linalg.norm(nn, axis=-1, keepdims=True) + 1e-6
            # relief option scales the slope (1 = as in game)
            if o.relief != 1.0:
                nn[..., 0] *= o.relief
                nn[..., 2] *= o.relief
                nn /= np.linalg.norm(nn, axis=-1, keepdims=True)
            del xa, ya, t, n

            # --- sun (CalculateSunDirection: sun at SUN_HEIGHT above SUN_LATITUDE, same x)
            h = smp_height.rows(r0, r1)
            zworld = (H - (rows + 0.5) / self.sy)[:, None]
            Ly = self.sun_height - h * 0.1
            Lz = self.sun_lat - zworld
            inv = 1.0 / np.sqrt(Ly * Ly + Lz * Lz)
            ndl = np.clip((nn[..., 1] * Ly + nn[..., 2] * Lz) * inv, 0.0, 1.0)
            sq = nn * nn
            amb = (sq[..., 0:1] * np.where(nn[..., 0:1] >= 0, amb_pos[0], amb_neg[0])
                   + sq[..., 1:2] * np.where(nn[..., 1:2] >= 0, amb_pos[1], amb_neg[1])
                   + sq[..., 2:3] * np.where(nn[..., 2:3] >= 0, amb_pos[2], amb_neg[2]))
            light = (amb + ndl[..., None]) * self.hdr_range
            land = diffuse * light
            # light factor of flat ground on this row, used for the political colour layer
            flat_light = (amb_pos[1] + np.clip(Ly * inv, 0, 1)[..., None]) * self.hdr_range
            del nn, sq, amb, ndl, diffuse, light

            # --- water (pdxwater.shader + underwater pass of pdxmap.shader)
            shore = np.clip((h / WATER_LEVEL - 0.954) * 25.0, 0.0, 1.0)  # 1 = land
            tint = np.power(smp_water.rows(r0, r1) / 255.0, 2.2)
            wdepth = np.clip((h * 0.1 - 5.3) / (13.0 - 5.3), 0.0, 1.0)[..., None]
            seabed = self.seabed * (amb_pos[1] + 1.0)
            refr = tint * WATER_COLOR_LIGHTNESS + (seabed - tint * WATER_COLOR_LIGHTNESS) * wdepth
            sky, fres = self._sky(rows, cols)
            fres = fres[..., None]
            # 0.8: "darkness offset" of pdxwater.shader; sun ~ straight above (SUN_HEIGHT_WATER)
            wlight = 0.8 * (amb_pos[1] + 1.0) * self.hdr_range
            fres = fres * self.o.sky_strength
            water = (refr * (1.0 - fres) + sky * self.cubemap_intensity * fres) * wlight
            water_plain = refr * wlight  # same water without the sky reflection
            p_plain = post(water_plain)
            p_full = post(water)
            p_land = post(land)
            img = p_plain + (p_land - p_plain) * shore[..., None]
            base[r0:r1] = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
            # reflection layer: alpha = fresnel weight on water; colour chosen so that
            # base + reflection reproduces the shader's result exactly (up to clipping)
            ra = (fres * (1.0 - shore[..., None]))
            rcol = p_plain + (p_full - p_plain) / np.maximum(fres, 1e-3)
            reflection[r0:r1, :, :3] = np.clip(rcol * 255.0 + 0.5, 0, 255).astype(np.uint8)
            reflection[r0:r1, :, 3] = np.clip(ra[..., 0] * 255.0 + 0.5, 0, 255).astype(np.uint8)
            del water_plain, p_plain, p_full, p_land, ra, rcol
            del refr, img, land, wdepth

            # --- rivers (river.shader): ribbon = water core + textured banks fading out
            if has_rivers:
                lo = max(0, r0 - margin)
                hi = min(OH, r1 + margin)
                cen = self.river_centre[lo:hi]
                if cen.any():
                    dist, (iy, ix) = ndimage.distance_transform_edt(cen == 0, return_indices=True)
                    s0, s1 = r0 - lo, r0 - lo + (r1 - r0)
                    dist = dist[s0:s1].astype(np.float32)
                    half = river_half[cen[iy[s0:s1], ix[s0:s1]]]
                    del iy, ix
                    across = np.where(half > 0, dist / np.maximum(half, 1e-3), 2.0)  # 0 centre .. 1 edge
                    v = np.clip(across * 2.0 - 1.0, 0.0, 1.0)   # vNewUV.y = saturate(y*4-3)
                    inside = across <= 1.0
                    nb = len(self.river_side)
                    side = np.interp(v, np.linspace(0, 1, nb), self.river_side)
                    alpha = np.interp(v, np.linspace(0, 1, nb), self.river_alpha)
                    side = np.where(across <= 0.5, 0.0, side)
                    alpha = np.where(inside, np.where(across <= 0.5, 1.0, alpha), 0.0)
                    nbk = len(self.river_bank)
                    bk = np.stack([np.interp(1.0 - v, np.linspace(0, 1, nbk), self.river_bank[:, c])
                                   for c in range(3)], -1)
                    rt = np.power(smp_rwater.rows(r0, r1) / 255.0, 2.2) if smp_rwater is not None else tint
                    rwater = rt * (1.0 - fres) + sky * 1.3 * fres
                    rcol = rwater + (bk - rwater) * side[..., None]
                    rcol = rcol * (amb_pos[1] + 1.0) * self.hdr_range
                    rcol = post(rcol)
                    alpha = alpha * shore  # rivers end at the coast
                    rivers[r0:r1, :, :3] = np.clip(rcol * 255 + 0.5, 0, 255).astype(np.uint8)
                    rivers[r0:r1, :, 3] = np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)
                    del dist, half, across, v, side, alpha, bk, rwater, rcol
            del tint, sky, fres, water, h, shore

            # --- political colour (gradient borders, zoomed-out fill)
            ca = smp_ca.rows(r0, r1) / 255.0
            crgb = smp_crgb.rows(r0, r1) / 255.0
            crgb /= np.maximum(ca, 1e-4)[..., None]
            crgb = np.power(np.clip(crgb, 0, 1), 2.2)  # sRGB -> linear like the other colour textures
            g = smp_grad.rows(r0, r1) / 255.0
            gopa = 1.0 - (1.0 - o.fill) * (1.0 - g)
            a_pol = gopa * ca * GB_FAR_MIX
            # lighting is partly removed under the borders (BORDER_LIGHT_REMOVAL_FACTOR)
            k = flat_light * (1.0 - BORDER_LIGHT_REMOVAL_FACTOR * a_pol[..., None]) \
                + BORDER_LIGHT_REMOVAL_FACTOR * a_pol[..., None]
            ccol = post(crgb * k)
            colors[r0:r1, :, :3] = np.clip(ccol * 255.0 + 0.5, 0, 255).astype(np.uint8)
            colors[r0:r1, :, 3] = np.clip(a_pol * 255.0 + 0.5, 0, 255).astype(np.uint8)
            del ca, crgb, g, gopa, a_pol, k, ccol

            if (r0 // o.strip) % 4 == 0:
                log(f"  rows {r1}/{OH} ({time.time() - t0:.0f}s)")
        self.base = base
        self.reflection = reflection
        self.colors = colors
        self.rivers_rgba = rivers
        # border mesh colours go through the same post-processing
        self.country_border_color = tuple(int(c * 255 + 0.5) for c in post(np.power(COUNTRY_BORDER_COLOR, 2.2)[None])[0])

    # ------------------------------------------------------------- layers
    def layers(self, names_mask=None):
        """Ordered list of (name, kind, data): kind 'rgb' (opaque), 'rgba', or ('mask', color, opacity)."""
        out = [("base", "rgb", self.base)]
        if getattr(self, "reflection", None) is not None:
            out.append(("water_reflection", "rgba", self.reflection))
        if getattr(self, "rivers_rgba", None) is not None:
            out.append(("rivers", "rgba", self.rivers_rgba))
        if self.o.political:
            out.append(("country_colors", "rgba", self.colors))
        if "sea" in self.masks:
            out.append(("sea_borders", ("mask", (0, 0, 0), 1.0), self.masks["sea"]))
        if "province" in self.masks:
            out.append(("province_borders", ("mask", (0, 0, 0), 0.9), self.masks["province"]))
        if "state" in self.masks:
            out.append(("state_borders", ("mask", (0, 0, 0), 0.9), self.masks["state"]))
        if "country" in self.masks:
            out.append(("country_borders", ("mask", self.country_border_color, 1.0),
                        self.masks["country"]))
        if names_mask is not None:
            out.append(("country_names", "rgba", names_rgba(names_mask, self.o.name_color, self.o.name_opacity, self.s)))
        return out


def composite(layers, strip=512):
    """Alpha-composite the layer stack into one RGB image."""
    base = layers[0][2]
    OH, OW = base.shape[:2]
    out = np.empty((OH, OW, 3), np.uint8)
    for r0 in range(0, OH, strip):
        r1 = min(OH, r0 + strip)
        img = base[r0:r1].astype(np.float32)
        for _, kind, data in layers[1:]:
            if kind == "rgba":
                a = data[r0:r1, :, 3:4].astype(np.float32) / 255.0
                img += (data[r0:r1, :, :3].astype(np.float32) - img) * a
            else:
                _, color, op = kind
                a = data[r0:r1, :, None].astype(np.float32) * (op / 255.0)
                img += (np.array(color, np.float32) - img) * a
        out[r0:r1] = np.clip(img + 0.5, 0, 255).astype(np.uint8)
    return out


def names_rgba(mask, color, opacity, scale, strip=512):
    """White map names with the soft dark edge they have in game."""
    OH, OW = mask.shape
    out = np.zeros((OH, OW, 4), np.uint8)
    rad = max(1.0, 0.9 * scale)
    pad = int(math.ceil(rad * 3)) + 2
    col = np.array(color, np.float32) / 255.0
    for r0 in range(0, OH, strip):
        r1 = min(OH, r0 + strip)
        lo, hi = max(0, r0 - pad), min(OH, r1 + pad)
        m = mask[lo:hi]
        if not m.any():
            continue
        t = m.astype(np.float32) / 255.0
        sh = cv2.GaussianBlur(cv2.dilate(t, np.ones((3, 3), np.uint8)), (0, 0), rad)
        t = t[r0 - lo:r1 - lo] * opacity
        sh = np.clip(sh[r0 - lo:r1 - lo] * 0.55, 0, 1)
        a = t + sh * (1.0 - t)
        rgb = (col * t[..., None]) / np.maximum(a, 1e-4)[..., None]  # dark edge is black
        out[r0:r1, :, :3] = np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)
        out[r0:r1, :, 3] = np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8)
    return out


def layer_rgba(kind, data, r0, r1):
    """RGBA rows of a layer, for saving as a transparent PNG."""
    if kind == "rgba":
        return data[r0:r1]
    _, color, op = kind
    m = data[r0:r1]
    rgba = np.empty(m.shape + (4,), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
    rgba[..., 3] = (m.astype(np.float32) * op + 0.5).astype(np.uint8) if op != 1.0 else m
    return rgba
