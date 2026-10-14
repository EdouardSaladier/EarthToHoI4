"""Permanent snow layer (ApplySnow() of standardfuncsgfx.fxh), rendered on its own over an existing render.

Usage: python -m hoi4renderer.snow <mod folder> <render folder> [--scale 1] [--strength 1]

In game the snow amount comes from the SnowMudData texture, filled at runtime from the weather. Terrain types
with `perm_snow = yes` in common/terrain always have snow, so they are used as the snow mask. The layer is
saved as `0b_snow.png`, to be stacked right above `0_base.png`: base + layer reproduces the shaded snowy terrain.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

from . import pdx
from .__main__ import find_game_dir
from .data import MapData, load_indexed, load_rgb, log
from .post import PostProcess, load_defines
from .render import (COLORMAP_OVERLAY_STRENGTH, DAY_AMBIENT_NEG, DAY_AMBIENT_POS,
                     TERRAIN_TILE_FREQ, WATER_LEVEL, Options, Sampler, build_tiles, load_texture, overlay)

# --- constants of gfx/FX/constants.fxh
SNOW_OPACITY_MAX = 0.2      # cam_distance() is 1 when fully zoomed out
SNOW_START_HEIGHT = 3.0
SNOW_RIDGE_START_HEIGHT = 11.0
SNOW_NORMAL_START = 0.7
SNOW_COLOR = np.array([0.46, 0.48, 0.69], np.float32)
SNOW_CLIFFS = 5.0
SNOW_TILING = 0.05
SNOW_NOISE_TILING = 0.06
SNOW_FROST_MIN_EFFECT = 0.4
SNOW_MASK_BLUR = 2.0        # map px: SnowMudData is a low resolution, bilinear filtered texture


def perm_snow_lut(vfs):
    """terrain.bmp palette index -> 1 if the graphical terrain has perm_snow = yes."""
    lut = np.zeros(256, np.float32)
    for _, path in vfs.list_dir("common/terrain", (".txt",)).items():
        terr = pdx.parse_file(path).get("terrain")
        if not isinstance(terr, pdx.Block):
            continue
        for _, _, v in terr:
            if not isinstance(v, pdx.Block) or not isinstance(v.get("color"), pdx.Block):
                continue
            try:
                idx = int(v.get("color").values()[0])
            except (TypeError, ValueError, IndexError):
                continue
            if 0 <= idx < 256 and v.get("perm_snow") == "yes":
                lut[idx] = 1.0
    return lut


def tiled_texture(img, period_px):
    """Texture repeating every period_px output pixels, at the matching mip size."""
    size = max(4, int(round(period_px)))
    w, h = img.size
    img = img.resize((size, size), Image.BOX if size < w else Image.BICUBIC)
    return np.asarray(img, np.float32) / 255.0


def sample_wrap(tex, u, v):
    """Bilinear sample of a repeating texture at uv (in repeats)."""
    n = tex.shape[0]
    x = u * n - 0.5
    y = v * n - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    x0 %= n
    y0 %= n
    x1 = (x0 + 1) % n
    y1 = (y0 + 1) % n
    a = tex[y0, x0] + (tex[y0, x1] - tex[y0, x0]) * fx
    b = tex[y1, x0] + (tex[y1, x1] - tex[y1, x0]) * fx
    return a + (b - a) * fy


def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)


def rotate_by(t, n):
    """RotateVectorByVector(t, n): detail normal t expressed around the surface normal n."""
    xa = normalize(np.stack([n[..., 1], -n[..., 0], np.zeros_like(n[..., 0])], -1))
    ya = normalize(np.stack([xa[..., 1] * n[..., 2], -xa[..., 0] * n[..., 2],
                             xa[..., 0] * n[..., 1] - xa[..., 1] * n[..., 0]], -1))
    return normalize(xa * t[..., 0:1] + n * t[..., 1:2] + ya * t[..., 2:3])


def render_snow(md, scale, avg_luminance, lut, strength=1.0, base=None, strip=256):
    vfs = md.vfs
    W, H = md.width_px, md.height_px
    OW, OH = int(round(W * scale)), int(round(H * scale))
    s = OW / W

    terrain_idx = load_indexed(md._map_file("terrain", "terrain.bmp"))
    snow_mask = perm_snow_lut(vfs)[terrain_idx]
    if not snow_mask.any():
        log("No perm_snow terrain on this map")
    tex_index = md.terrain_texture_lut()[terrain_idx]
    del terrain_idx
    snow_mask = cv2.GaussianBlur(snow_mask, (0, 0), SNOW_MASK_BLUR)

    height = load_indexed(md._map_file("heightmap", "heightmap.bmp"))
    height = cv2.GaussianBlur(height.astype(np.float32), (0, 0), 0.8)
    npath = vfs.find("map/world_normal.bmp")
    normalmap = load_rgb(npath)
    colormap = np.asarray(load_texture(vfs, "map/terrain/colormap_rgb_cityemissivemask_a.dds", "RGB"), np.uint8)

    P = max(4, int(round(H / TERRAIN_TILE_FREQ * s)))
    tiles = np.power(build_tiles(load_texture(vfs, "map/terrain/atlas0.dds", "RGBA"), P)[..., :3], 2.2)
    tiles_n = build_tiles(load_texture(vfs, "map/terrain/atlas_normal0.dds", "RGBA"), P)
    snow_tex = tiled_texture(load_texture(vfs, "map/terrain/snow_normal_rgb_diffuse_a.dds", "RGBA"),
                             s / SNOW_TILING)
    noise_tex = tiled_texture(load_texture(vfs, "map/terrain/citylights_rgb_snowmask_a_0.dds", "RGBA"),
                              s / SNOW_NOISE_TILING)[..., 3:4]

    d = load_defines(vfs)
    sun_height = d.get("SUN_HEIGHT", 600.0)
    sun_lat = d.get("SUN_LATITUDE", 848.0)
    hdr = d.get("LIGHT_HDR_RANGE", 1.0)
    post = PostProcess(vfs, lut, avg_luminance=avg_luminance)
    amb_pos = np.array(DAY_AMBIENT_POS, np.float32)
    amb_neg = np.array(DAY_AMBIENT_NEG, np.float32)

    smp_height = Sampler(height, W, H, OW, s)
    smp_cmap = Sampler(colormap, W, H, OW, s)
    smp_norm = Sampler(normalmap, W, H, OW, s)
    smp_snow = Sampler(snow_mask, W, H, OW, s)

    mx = (np.arange(OW) + 0.5) / s - 0.5
    x0 = np.floor(mx).astype(np.int64)
    wx = (mx - x0).astype(np.float32)[None, :, None]
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x0 + 1, 0, W - 1)
    tx = (np.arange(OW) % P)[None, :]
    xworld = ((np.arange(OW) + 0.5) / s)[None, :]

    def lit(diffuse, nn, h, rows, extra=0.0):
        zworld = (H - (rows + 0.5) / s)[:, None]
        Ly = sun_height - h * 0.1
        Lz = sun_lat - zworld
        inv = 1.0 / np.sqrt(Ly * Ly + Lz * Lz)
        ndl = np.clip((nn[..., 1] * Ly + nn[..., 2] * Lz) * inv, 0.0, 1.0)
        sq = nn * nn
        amb = (sq[..., 0:1] * np.where(nn[..., 0:1] >= 0, amb_pos[0], amb_neg[0])
               + sq[..., 1:2] * np.where(nn[..., 1:2] >= 0, amb_pos[1], amb_neg[1])
               + sq[..., 2:3] * np.where(nn[..., 2:3] >= 0, amb_pos[2], amb_neg[2]))
        return diffuse * (amb + ndl[..., None] + extra) * hdr

    out = np.zeros((OH, OW, 4), np.uint8)
    t0 = time.time()
    for r0 in range(0, OH, strip):
        r1 = min(OH, r0 + strip)
        isnow = smp_snow.rows(r0, r1)
        if isnow.max() <= 1e-3:
            continue
        rows = np.arange(r0, r1, dtype=np.float32)
        # --- terrain as in Renderer.render()
        my = (rows + 0.5) / s - 0.5
        y0 = np.floor(my).astype(np.int64)
        wy = (my - y0).astype(np.float32)[:, None, None]
        y0c = np.clip(y0, 0, H - 1)
        y1c = np.clip(y0 + 1, 0, H - 1)
        ty = (np.arange(r0, r1) % P)[:, None]
        top, bot = tex_index[y0c], tex_index[y1c]
        idx = (top[:, x0c], top[:, x1c], bot[:, x0c], bot[:, x1c])

        def gather(t):
            a = t[idx[0], ty, tx] + (t[idx[1], ty, tx] - t[idx[0], ty, tx]) * wx
            b = t[idx[2], ty, tx] + (t[idx[3], ty, tx] - t[idx[2], ty, tx]) * wx
            return a + (b - a) * wy

        diffuse = overlay(gather(tiles), np.power(smp_cmap.rows(r0, r1) / 255.0, 2.2), COLORMAP_OVERLAY_STRENGTH)
        tn_raw = gather(tiles_n)
        wn = smp_norm.rows(r0, r1) / 255.0 - 0.5
        n = normalize(np.stack([wn[..., 0], wn[..., 2], wn[..., 1]], -1))
        nn = rotate_by(normalize(np.stack([tn_raw[..., 0], tn_raw[..., 2], tn_raw[..., 1]], -1) - 0.5), n)
        spec = tn_raw[..., 3]
        h = smp_height.rows(r0, r1)
        shore = np.clip((h / WATER_LEVEL - 0.954) * 25.0, 0.0, 1.0)
        plain = post(lit(diffuse, nn, h, rows))

        # --- ApplySnow()
        zworld = (H - (rows + 0.5) / s)[:, None]
        u, v = np.broadcast_arrays(xworld * SNOW_TILING, zworld * SNOW_TILING)
        stex = sample_wrap(snow_tex, u, v)
        u, v = np.broadcast_arrays(xworld * SNOW_NOISE_TILING, zworld * SNOW_NOISE_TILING)
        noise = sample_wrap(noise_tex, u, v)[..., 0]
        y = h * 0.1
        snow_fade = np.clip(y - SNOW_START_HEIGHT, 0, 1)
        normal_fade = np.clip(np.clip(nn[..., 1] - SNOW_NORMAL_START, 0, 1) * SNOW_CLIFFS, 0, 1)
        transp = noise + np.clip(y - SNOW_RIDGE_START_HEIGHT, 0, 1) * np.clip((nn[..., 1] - 0.9) * 1000, 0, 1) * isnow
        transp = np.clip(transp, 0, 1)
        snow = np.clip(np.clip(transp - (1 - isnow), 0, 1) * 5, 0, 1)
        frost = np.clip(np.clip(transp + 0.5, 0, 1) - (1 - isnow), 0, 1)
        salpha = np.clip(np.clip(snow + frost, 0, 1) * snow_fade * normal_fade
                         * np.clip(isnow * 2.25, 0, 1) * SNOW_OPACITY_MAX, 0, 1)
        min_snow = isnow * isnow * (3 - 2 * isnow)
        w = np.clip((1 - spec) * np.clip(salpha + SNOW_FROST_MIN_EFFECT * min_snow, 0, 1) * strength, 0, 1)
        sdiffuse = diffuse + (stex[..., 3:4] * SNOW_COLOR - diffuse) * w[..., None]
        snormal = rotate_by(normalize(np.stack([stex[..., 0], stex[..., 2], stex[..., 1]], -1) - 0.5), nn)
        sn = normalize(nn + (snormal - nn) * salpha[..., None])
        k = np.clip(salpha / 0.1, 0, 1)
        snow_amb = np.array([0.2, 0.7, 1.0], np.float32) * 0.07 * (k * k * (3 - 2 * k))[..., None]
        snowy = post(lit(sdiffuse, sn, h, rows, snow_amb))

        # --- layer: alpha = snow weight; colour such that base + layer = snowy terrain
        a = w * shore
        b = base[r0:r1].astype(np.float32) / 255.0 if base is not None else plain
        col = b + (snowy - plain) / np.maximum(w, 0.05)[..., None]
        out[r0:r1, :, :3] = np.clip(col * 255 + 0.5, 0, 255).astype(np.uint8)
        out[r0:r1, :, 3] = np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8)
        log(f"  rows {r1}/{OH} ({time.time() - t0:.0f}s)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hoi4renderer.snow", description=__doc__.splitlines()[0])
    ap.add_argument("mod")
    ap.add_argument("render", help="existing render folder (0_base.png is read from it, 0b_snow.png written)")
    ap.add_argument("--game")
    ap.add_argument("-s", "--scale", type=float, default=Options.scale)
    ap.add_argument("--avg-luminance", type=float, default=Options.avg_luminance)
    ap.add_argument("--lut", default=Options.lut)
    ap.add_argument("--strength", type=float, default=1.0,
                    help="multiplier of the snow cover (1 = as in game fully zoomed out)")
    args = ap.parse_args(argv)

    game = args.game or find_game_dir()
    vfs = pdx.VFS.from_mod(game, os.path.abspath(args.mod))
    md = MapData(vfs)
    base = None
    bpath = os.path.join(args.render, "0_base.png")
    if os.path.isfile(bpath):
        Image.MAX_IMAGE_PIXELS = None
        base = np.asarray(Image.open(bpath).convert("RGB"))
        if base.shape[:2] != (int(round(md.height_px * args.scale)), int(round(md.width_px * args.scale))):
            ap.error(f"0_base.png is {base.shape[1]}x{base.shape[0]}, does not match --scale {args.scale}")
    log("Rendering snow...")
    layer = render_snow(md, args.scale, args.avg_luminance, args.lut, args.strength, base)
    path = os.path.join(args.render, "0b_snow.png")
    log(f"Writing {path}...")
    Image.fromarray(layer, "RGBA").save(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
