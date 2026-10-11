"""Population density map (Provinces tab: bigger provinces where few people live).

Source: JRC GHS-POP R2023A, epoch 2025, 30 arc-second (~1 km) grid in
EPSG:4326 (Download tab -> "Population density (GHS-POP 1 km)"): a BigTIFF,
43202 x 21384 float64 people per cell, LZW, 256 x 256 tiles, origin
180.0079 W / 89.0996 N, cells of 1/120 deg. No GeoTIFF library: each tile's
LZW data is wrapped in a minimal in-memory TIFF (8-bit, 2048 bytes wide =
256 float64) that Pillow decodes, then read back as float64.

Built once (`build`, ~1 min) into a REDUCE-times coarser grid of people
per cell (sums), data/density/pop_4min.npy (5400 x 2673 float32, 58 MB);
`density_on_view` resamples it onto a render's frame as people / km2.
"""
import io
import struct
import threading
import time
import zipfile

import numpy as np
from PIL import Image

import core

DENSITY_DIR = core.DATA_DIR / "density"
ZIP_NAME = "GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.zip"
TIF_PATH = DENSITY_DIR / "GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif"
GRID_PATH = DENSITY_DIR / "pop_4min.npy"
REDUCE = 8                       # 30" cells per grid cell (8 -> 4 arc-minutes)
CELL_DEG = 1.0 / 120.0
_lock = threading.Lock()
_grid = None
_geo = None


def available() -> bool:
    return TIF_PATH.exists() or GRID_PATH.exists()


def _tags(f):
    """BigTIFF header -> {tag: (type, count, value or offset)}."""
    f.seek(0)
    hdr = f.read(16)
    if hdr[2:4] != b"\x2b\x00":
        raise core.RenderError("GHS-POP: expected a BigTIFF")
    f.seek(struct.unpack("<Q", hdr[8:16])[0])
    n = struct.unpack("<Q", f.read(8))[0]
    tags = {}
    for _ in range(n):
        tag, typ, cnt, val = struct.unpack("<HHQQ", f.read(20))
        tags[tag] = (typ, cnt, val)
    return tags


def _array(f, typ, cnt, off):
    fmt = {3: "<u2", 4: "<u4", 16: "<u8", 12: "<f8"}[typ]
    size = np.dtype(fmt).itemsize * cnt
    if size <= 8:
        return np.frombuffer(struct.pack("<Q", off)[:size], dtype=fmt)
    f.seek(off)
    return np.frombuffer(f.read(size), dtype=fmt)


def _decode_tile(data: bytes, tw: int, th: int) -> np.ndarray:
    """One LZW tile of float64 -> (th, tw) array, via a minimal classic TIFF
    (8-bit grey, tw * 8 bytes wide, one strip) that Pillow decodes."""
    width = tw * 8
    n_tags = 9
    data_off = 8 + 2 + 12 * n_tags + 4
    entries = [(256, 4, 1, width), (257, 4, 1, th), (258, 3, 1, 8), (259, 3, 1, 5), (262, 3, 1, 1),
               (273, 4, 1, data_off), (277, 3, 1, 1), (278, 4, 1, th), (279, 4, 1, len(data))]
    b = bytearray(b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", n_tags))
    for tag, typ, cnt, value in entries:
        b += struct.pack("<HHIHH", tag, typ, cnt, value, 0) if typ == 3 else struct.pack("<HHII", tag, typ, cnt, value)
    b += struct.pack("<I", 0) + data
    raw = np.ascontiguousarray(np.asarray(Image.open(io.BytesIO(bytes(b)))))
    return raw.view("<f8").reshape(th, tw)


def build(progress=None):
    """The reduced grid from the GeoTIFF (extracting it from the zip if
    needed); saved to GRID_PATH."""
    if not TIF_PATH.exists():
        with zipfile.ZipFile(DENSITY_DIR / ZIP_NAME) as z:
            z.extract(TIF_PATH.name, DENSITY_DIR)
    t0 = time.time()
    with open(TIF_PATH, "rb") as f:
        tags = _tags(f)
        W, H = int(tags[256][2]), int(tags[257][2])
        tw, th = int(tags[322][2]), int(tags[323][2])
        offs = _array(f, *tags[324]).astype(np.int64)
        cnts = _array(f, *tags[325]).astype(np.int64)
        scale = _array(f, *tags[33550])
        tie = _array(f, *tags[33922])
        tx, ty = (W + tw - 1) // tw, (H + th - 1) // th
        gw, gh = (W + REDUCE - 1) // REDUCE, (H + REDUCE - 1) // REDUCE
        grid = np.zeros((gh, gw), dtype=np.float64)
        for r in range(ty):
            for c in range(tx):
                k = r * tx + c
                f.seek(int(offs[k]))
                tile = _decode_tile(f.read(int(cnts[k])), tw, th)
                y0, x0 = r * th, c * tw
                hh, ww = min(th, H - y0), min(tw, W - x0)
                tile = np.nan_to_num(np.maximum(tile[:hh, :ww], 0.0))
                # sum into REDUCE x REDUCE cells (tiles are multiples of REDUCE)
                ph, pw = (-hh) % REDUCE, (-ww) % REDUCE
                tile = np.pad(tile, ((0, ph), (0, pw)))
                s = tile.reshape(tile.shape[0] // REDUCE, REDUCE, tile.shape[1] // REDUCE, REDUCE).sum((1, 3))
                gy, gx = y0 // REDUCE, x0 // REDUCE
                grid[gy:gy + s.shape[0], gx:gx + s.shape[1]] += s
            if progress:
                progress((r + 1) / ty)
    np.save(GRID_PATH, grid.astype(np.float32))
    np.save(GRID_PATH.with_suffix(".geo.npy"), np.array([tie[3], tie[4], scale[0] * REDUCE, scale[1] * REDUCE]))
    return time.time() - t0


def load():
    """(grid of people per cell, (lon0, lat0, dlon, dlat)), built if missing."""
    global _grid, _geo
    with _lock:
        if _grid is None:
            if not GRID_PATH.exists():
                build()
            _grid = np.load(GRID_PATH, mmap_mode="r")
            _geo = tuple(np.load(GRID_PATH.with_suffix(".geo.npy")).tolist())
        return _grid, _geo


def density_on_view(view, w: int, h: int) -> np.ndarray:
    """People per km2 at every pixel of the frame (nearest grid cell)."""
    grid, (lon0, lat0, dlon, dlat) = load()
    gh, gw = grid.shape
    cols_x = view["win_xmin"] + (np.arange(w) + 0.5) / view["scale"]
    out = np.zeros((h, w), dtype=np.float32)
    step = max(1, 1_000_000 // max(w, 1))
    for r0 in range(0, h, step):
        r1 = min(h, r0 + step)
        rows_y = view["win_ymin"] + (h - (np.arange(r0, r1) + 0.5)) / view["scale"]
        yy, xx = np.meshgrid(rows_y, cols_x, indexing="ij")
        lon, lat, inside = core._unproject(view["projection"], view["center"], xx, yy)
        gi = np.clip(((lat0 - lat) / dlat).astype(np.int64), 0, gh - 1)
        gj = np.clip(((lon - lon0) / dlon).astype(np.int64) % gw, 0, gw - 1)
        cell_km2 = (dlat * 110.57) * (dlon * 111.32 * np.cos(np.radians(np.clip(lat, -89.9, 89.9))))
        out[r0:r1] = np.where(inside, grid[gi, gj] / np.maximum(cell_km2, 1e-6), 0.0)
    return out
