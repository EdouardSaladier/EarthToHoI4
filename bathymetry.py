"""NASA Blue Marble bathymetry (GEBCO 08): ocean depth on a 30" grid, 8 JPEG
tiles of 90 x 90 degrees like the topography (columns A-D from 180 W, row 1
north), in data/bathymetry/ (Download tab card "NASA Bathymetry"). NASA:
"depths between -8000 m and 0 m (surface)" -- grey 255 = surface (land is
255 too), 0 = 8000 m or deeper, linear; "shading indicates changes in
slope or depth", so single pixels carry some hill-shade noise (checked:
Gotland Deep 248 = 220 m, Black Sea 185 = 2,200 m, Norwegian Sea 181 =
2,300 m, mid-Atlantic 147 = 3,400 m, Mariana Trench 0): average over areas.
Used for the strategic regions' naval terrain."""
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
BATHY_DIR = ROOT / "data" / "bathymetry"
TILE_PX = 10800
TILE_DEG = 90.0
MAX_DEPTH_M = 8000.0
MEMO_BYTES = 400_000_000
_lock = threading.Lock()
_memo = OrderedDict()


def tile_path(col: int, row: int) -> Path:
    return BATHY_DIR / f"gebco_08_rev_bath_{'ABCD'[col]}{row + 1}_grey_geo.jpg"


def available() -> bool:
    return all(tile_path(c, r).exists() for c in range(4) for r in range(2))


def _tile(col: int, row: int, reduce: int) -> np.ndarray:
    """Tile decoded at 1/`reduce` resolution (JPEG draft mode), memoized."""
    key = (col, row, reduce)
    with _lock:
        if key in _memo:
            _memo.move_to_end(key)
            return _memo[key]
    Image.MAX_IMAGE_PIXELS = None
    n = TILE_PX // reduce
    im = Image.open(tile_path(col, row))
    im.draft("L", (n, n))
    arr = np.asarray(im.convert("L"))
    if arr.shape != (n, n):
        arr = np.asarray(Image.fromarray(arr).resize((n, n), Image.BILINEAR))
    with _lock:
        _memo[key] = arr
        while sum(a.nbytes for a in _memo.values()) > MEMO_BYTES and len(_memo) > 1:
            _memo.popitem(last=False)
    return arr


def depth_m(lat, lon, reduce: int = 4) -> np.ndarray:
    """Depth below the surface in metres (>= 0; 0 on land) at points, from
    the tiles decoded at 1/`reduce` (4 = ~3.7 km cells, plenty for region
    averages)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = (np.asarray(lon, dtype=np.float64) + 180.0) % 360.0 - 180.0
    n = TILE_PX // reduce
    col = np.clip(((lon + 180.0) // TILE_DEG).astype(np.int64), 0, 3)
    row = (lat < 0).astype(np.int64)
    x = np.clip((((lon + 180.0) - col * TILE_DEG) / TILE_DEG * n).astype(np.int64), 0, n - 1)
    y = np.clip((((90.0 - lat) - row * TILE_DEG) / TILE_DEG * n).astype(np.int64), 0, n - 1)
    out = np.zeros(lat.shape, dtype=np.float64)
    for c in range(4):
        for r in range(2):
            m = (col == c) & (row == r)
            if m.any():
                g = _tile(c, r, reduce)[y[m], x[m]].astype(np.float64)
                out[m] = (255.0 - g) / 255.0 * MAX_DEPTH_M
    return out
