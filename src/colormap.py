"""HOI4 colormap_rgb_cityemissivemask_a.dds for the Terrain tab.

The game's file (vanilla in the project root as a reference): uncompressed
32-bit BGRA DDS, no mipmaps, half the heightmap's size (2816 x 1024 for
5632 x 2048) -- RGB = the map's colour (a muted satellite image, water a
flat 79,102,142), alpha = city lights at night.

Here:
  * RGB: NASA Blue Marble (August, data/blue_marble/, 8 tiles of 90 x 90
    deg at 15", columns A-D from 180 W, row 1 north) reprojected onto the
    render's frame at half resolution (bilinear, tiles decoded at the
    coarsest JPEG draft scale still as fine as the output pixel), then
    per-channel curves towards the vanilla file's land colours
    (COLOR_KNOTS). Water (sea + lakes, from the country map, never from
    the satellite image) = WATER_RGB, blended by each output pixel's water
    share, so coasts are soft like the vanilla file's.
  * Alpha: the terrain map's urban pixels (land cover cities + the City
    tab's discs): urban share per output pixel, Gaussian blur of
    `light_glow_px`, times `light_gain`. Vanilla, measured against its
    terrain.bmp: median 175 on urban pixels, mean ~40 one pixel out, ~20
    two out. Defaults (gain 2.5, glow 0.7) matched on a world render:
    median ~175 on urban pixels.
"""
import struct
import threading
import time
from collections import OrderedDict

import numpy as np
from PIL import Image
from scipy import ndimage

import core

BLUE_MARBLE_DIR = core.DATA_DIR / "blue_marble"
BM_TILE_PX = 21600
BM_TILE_DEG = 90.0
BM_MEMO_BYTES = 600_000_000
WATER_RGB = (79, 102, 142)    # the vanilla file's sea (and lakes): flat
COLORMAP_DEFAULTS = {
    "light_glow_px": 0.7,     # blur of the city lights (colormap px)
    "light_gain": 2.5,        # city lights brightness (alpha = 255 x gain x blurred urban share)
}
COLORMAP_LIMITS = {"light_glow_px": (0, 10), "light_gain": (0, 10)}
# Blue Marble -> vanilla land colour, per channel (input knots -> output):
# quantile matching of inland land, world Miller render vs the vanilla file
# (_dev/colormap_calib.py, percentiles 0-99; the 100th dropped). Vanilla is
# darker and muted, with no white: Greenland's ice (Blue Marble's top 5%)
# ends up a pale grey as in the game.
COLOR_KNOTS = [
    ([0.0, 18.4, 19.9, 22.4, 25.6, 32.4, 40.1, 49.4, 61.6, 81.5, 110.3, 147.0, 198.7, 251.3, 253.4, 255.0],
     [0.0, 18.0, 24.0, 29.0, 33.0, 41.0, 49.0, 52.0, 60.0, 71.0, 82.0, 99.0, 120.0, 139.0, 148.0, 153.0]),
    ([4.4, 28.7, 30.9, 34.2, 37.5, 43.2, 48.4, 54.3, 61.9, 72.3, 91.3, 125.8, 172.1, 253.0, 254.9, 255.0],
     [8.0, 32.0, 36.0, 40.0, 44.0, 50.0, 55.0, 59.0, 63.0, 68.0, 77.0, 89.0, 101.0, 113.0, 123.0, 129.0]),
    ([1.8, 7.5, 8.3, 9.8, 11.5, 14.7, 18.5, 23.6, 31.3, 43.0, 59.8, 93.2, 139.2, 253.0, 254.6, 255.0],
     [0.0, 8.0, 8.0, 8.0, 13.0, 16.0, 24.0, 29.0, 33.0, 41.0, 49.0, 57.0, 74.0, 90.0, 123.0, 129.0]),
]

SNOW_FROM, SNOW_FULL = 170.0, 230.0   # Blue Marble min channel: snow turns grey from .. to
_lock = threading.Lock()
_memo = OrderedDict()


def _tile_path(col: int, row: int):
    return BLUE_MARBLE_DIR / f"world.200408.3x21600x21600.{'ABCD'[col]}{row + 1}.jpg"


def available() -> bool:
    return all(_tile_path(c, r).exists() for c in range(4) for r in range(2))


def normalize_settings(raw):
    out = dict(COLORMAP_DEFAULTS)
    for k, v in (raw or {}).items():
        if k not in COLORMAP_DEFAULTS or v is None or v == "":
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise core.RenderError(f"Colour map setting '{k}' must be a number")
        lo, hi = COLORMAP_LIMITS[k]
        if not lo <= v <= hi:
            raise core.RenderError(f"Colour map setting '{k}' must be between {lo:g} and {hi:g}")
        out[k] = v
    return out


def _tile(col: int, row: int, reduce: int) -> np.ndarray:
    """Tile decoded at 1/reduce (1, 2, 4, 8: JPEG draft mode), LRU-memoized."""
    key = (col, row, reduce)
    with _lock:
        if key in _memo:
            _memo.move_to_end(key)
            return _memo[key]
    Image.MAX_IMAGE_PIXELS = None
    n = BM_TILE_PX // reduce
    im = Image.open(_tile_path(col, row))
    im.draft("RGB", (n, n))
    arr = np.asarray(im.convert("RGB"))
    if arr.shape[:2] != (n, n):
        arr = np.asarray(Image.fromarray(arr).resize((n, n), Image.BILINEAR))
    with _lock:
        _memo[key] = arr
        while sum(a.nbytes for a in _memo.values()) > BM_MEMO_BYTES and len(_memo) > 1:
            _memo.popitem(last=False)
    return arr


def satellite(view, w: int, h: int, chunk_px: int = 1_000_000):
    """Blue Marble RGB (h x w x 3 float32, 0-255) on the view's frame
    resampled to w x h pixels, and the 'inside the world outline' mask."""
    scale = view["scale"] * w / view["width"]
    cols_x = view["win_xmin"] + (np.arange(w) + 0.5) / scale
    rows_y = view["win_ymin"] + (h - (np.arange(h) + 0.5)) * (view["height"] / h) / view["scale"]
    # finest output pixel in degrees -> source reduction
    gy, gx = np.meshgrid(rows_y[:: max(h // 32, 1)], cols_x[:: max(w // 32, 1)], indexing="ij")
    lon0, lat0, ok0 = core._unproject(view["projection"], view["center"], gx, gy)
    lon1, _l, ok1 = core._unproject(view["projection"], view["center"], gx + 1.0 / scale, gy)
    _l, lat2, ok2 = core._unproject(view["projection"], view["center"], gx, gy - 1.0 / scale)
    ok = ok0 & ok1 & ok2
    step = np.minimum(np.abs((lon1 - lon0 + 180.0) % 360.0 - 180.0), np.abs(lat2 - lat0))[ok]
    deg_px = float(step.min()) if len(step) else 1.0
    reduce = 1
    while reduce < 8 and BM_TILE_DEG / BM_TILE_PX * reduce * 2 <= deg_px:
        reduce *= 2
    n = BM_TILE_PX // reduce
    d = BM_TILE_DEG / n
    n_cols, n_rows = 4 * n, 2 * n
    out = np.zeros((h, w, 3), dtype=np.float32)
    inside_all = np.zeros((h, w), dtype=bool)
    rows_per = max(1, chunk_px // max(w, 1))
    for r0 in range(0, h, rows_per):
        r1 = min(h, r0 + rows_per)
        yy, xx = np.meshgrid(rows_y[r0:r1], cols_x, indexing="ij")
        lon, lat, inside = core._unproject(view["projection"], view["center"], xx, yy)
        u = (lon + 180.0) / d - 0.5
        q = (90.0 - lat) / d - 0.5
        j0 = np.floor(u).astype(np.int64)
        i0 = np.floor(q).astype(np.int64)
        fu, fq = (u - j0)[..., None], (q - i0)[..., None]
        acc = np.zeros(u.shape + (3,), dtype=np.float32)
        for di, dj, wgt in ((0, 0, (1 - fq) * (1 - fu)), (0, 1, (1 - fq) * fu),
                            (1, 0, fq * (1 - fu)), (1, 1, fq * fu)):
            ii = np.clip(i0 + di, 0, n_rows - 1)
            jj = (j0 + dj) % n_cols
            tile = (ii // n) * 4 + jj // n
            vals = np.zeros(u.shape + (3,), dtype=np.float32)
            for t in np.unique(tile[inside]).tolist():
                m = (tile == t) & inside
                arr = _tile(t % 4, t // 4, reduce)
                vals[m] = arr[ii[m] - (t // 4) * n, jj[m] - (t % 4) * n]
            acc += wgt.astype(np.float32) * vals
        out[r0:r1] = acc
        inside_all[r0:r1] = inside
    return out, inside_all, reduce


def _box(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    """Share of each output pixel covered by a full-resolution mask."""
    return np.asarray(Image.fromarray(mask.astype(np.float32), mode="F").resize((w, h), Image.BOX))


def render_colormap(countries: Image.Image, debug, terrain_img: Image.Image, settings=None):
    """(RGBA image: RGB = colour map, A = city lights, report) at half the
    render's size (rounded up, like world_normal.bmp)."""
    s = normalize_settings(settings)
    t0 = time.time()
    view = debug["view"]
    W, H = view["width"], view["height"]
    w, h = (W + 1) // 2, (H + 1) // 2
    c = np.asarray(countries.convert("RGB"))
    water = (c == np.asarray(core.BACKGROUND_COLOR, dtype=c.dtype)).all(axis=2)
    if debug.get("hidden_land") is not None:
        water &= ~debug["hidden_land"]
    if debug.get("lakes") is not None:
        water |= debug["lakes"]
    report = {"settings": s, "width": w, "height": h, "available": available()}
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    if report["available"]:
        sat, _inside, reduce = satellite(view, w, h)
        report["source_arcsec"] = round(BM_TILE_DEG / (BM_TILE_PX // reduce) * 3600.0, 1)
        # Blue Marble's sea / lakes where GADM has land (the shorelines
        # differ): a dark rim along coasts -> the nearest land colour.
        r_, g_, b_ = sat[..., 0], sat[..., 1], sat[..., 2]
        bm_water = (b_ > r_ + 10) & (b_ >= g_) & (r_ < 60)
        land_px = _box(~water, w, h) > 0
        bad = bm_water & land_px
        good = ~bm_water & land_px
        if bad.any() and good.any():
            _d, (iy, ix) = ndimage.distance_transform_edt(~good, return_indices=True)
            sat[bad] = sat[iy[bad], ix[bad]]
        report["satellite_water_on_land_px"] = int(bad.sum())
        lo = sat.min(axis=2)
        for k in range(3):
            sat[..., k] = np.interp(sat[..., k], *COLOR_KNOTS[k])
        # Snow / ice (near white in Blue Marble) -> neutral pale grey: the
        # per-channel curves alone tinted Greenland pink.
        wgt = np.clip((lo - SNOW_FROM) / (SNOW_FULL - SNOW_FROM), 0.0, 1.0)[..., None]
        rgb = sat * (1.0 - wgt) + sat.mean(axis=2, keepdims=True) * wgt
    wf = _box(water, w, h)[..., None]
    rgb = rgb * (1.0 - wf) + np.asarray(WATER_RGB, dtype=np.float32) * wf
    # City lights from the terrain's urban pixels.
    urban = np.asarray(terrain_img) == 13
    u = _box(urban, w, h)
    if s["light_glow_px"] > 0:
        u = ndimage.gaussian_filter(u, s["light_glow_px"])
    alpha = np.clip(np.rint(255.0 * s["light_gain"] * u), 0, 255).astype(np.uint8)
    out = np.dstack([np.clip(np.rint(rgb), 0, 255).astype(np.uint8), alpha])
    report["lit_share"] = round(float((alpha > 0).mean()), 4)
    report["seconds"] = round(time.time() - t0, 2)
    return Image.fromarray(out, mode="RGBA"), report


def dds_bytes(img: Image.Image) -> bytes:
    """Uncompressed 32-bit DDS (A8R8G8B8: bytes B, G, R, A), no mipmaps --
    the vanilla file's exact header layout."""
    a = np.asarray(img.convert("RGBA"))
    h, w = a.shape[:2]
    # reserved words: "NVT3" at word 9, as NVIDIA's texture tools wrote the vanilla file
    header = struct.pack("<4s7I36x4s4x", b"DDS ", 124, 0x100F, h, w, w * 4, 1, 1, b"NVT3")
    pf = struct.pack("<2I4s5I", 32, 0x41, b"\0\0\0\0", 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    caps = struct.pack("<5I", 0x1000, 0, 0, 0, 0)
    return header + pf + caps + a[..., [2, 1, 0, 3]].tobytes()
