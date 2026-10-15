"""GMTED2010 land elevations in metres (USGS), this tool's precise heightmap.

The NASA Blue Marble topography the tool started with is 8-bit over 0-6400 m,
about `core.NASA_M_PER_UNIT` (26.5 m) per step at 30 arcsec. That is coarser
than the game's own heightmap (one HOI4 level is ~38 m near sea level, see
`core.HOI4_HEIGHT_KNOTS`), so it costs nothing on the exported
`heightmap.bmp` -- but it decides two things that are not 8-bit at all:

  * **the flooded map**: everything under ~13 m is one grey step and drowns
    at once. Measured against GMTED on the user's own tiles, a 1 m rise on
    the 8-bit data drowns 86% of the Camargue, 71% of the Medoc and 98% of
    coastal Flanders;
  * **river mouths**: the delta a mouth has to cross is a single value (the
    Po delta: one value against GMTED's 20, from -6 to +16 m), so there is
    no ground for `core._elevation_route` to follow.

GMTED is signed 16-bit **metres**, nodata -32768, land only (the sea reads
exactly 0, and land below sea level is real -- the Dutch polders come out at
-14 m), covering 84N to 56S in 30 x 20 degree tiles at 7.5, 15 or 30 arcsec.
The Download tab fetches the 15 arcsec set (`mea150`, 7200 x 4800 px, 69 MB
per tile), which matches a country-sized frame: a France map at 3840 px is
11.4 arcsec per pixel.

Tiles are read from `data/gmted/`, plus any folder listed in
`MAP_TOOL_GMTED_DIRS` (`;`-separated) -- that is for a drive that already
holds some, the user's `E:\\GIS Data\\heightmaps` 7.5 arcsec set among them.
The finest resolution on disk for each tile wins, and a view is sampled on
the finest grid its own tiles offer. **Where no tile covers a pixel** (the
file isn't downloaded, or it is outside GMTED's coverage: Antarctica, the
far south), `heights_m` says so and `core.elevation_m` falls back to the
NASA tiles for exactly those pixels.

No GDAL: the tiles are plain uncompressed strip TIFFs, read here through
their own header (`ModelPixelScale` / `ModelTiepoint` are checked against
the grid the file name implies).
"""
import os
import re
import struct
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GMTED_DIR = ROOT / "data" / "gmted"

TILE_LAT_DEG = 20.0
TILE_LON_DEG = 30.0
LAT_SOUTH = (70, 50, 30, 10, -10, -30, -50, -70)      # north to south: the grid's rows
LON_WEST = tuple(range(-180, 180, 30))                # west to east: its columns
NORTH = LAT_SOUTH[0] + TILE_LAT_DEG                   # 90 N: the top of the virtual grid
# USGS statistic / resolution, finest first. "mea" = the mean elevation of
# the cells averaged into each pixel (the smooth one; "mx" would spike every
# summit, "mi" every valley).
RESOLUTIONS = (("075", 7.5), ("150", 15.0), ("300", 30.0))
DEFAULT_RES = "150"
NODATA = -32768
MEMO_BYTES = 600_000_000                              # tiles kept in memory (LRU)
URL = ("https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/topo/downloads/GMTED/"
       "Global_tiles_GMTED/{res}darcsec/mea/{folder}/{name}_20101117_gmted_mea{res}.tif")

_lock = threading.Lock()
_memo = OrderedDict()        # (lat, lon, arcsec) -> int16 array
_broken = {}                 # (lat, lon) -> why that file could not be read
_index_cache = (None, None)  # (signature, {(lat, lon): (path, arcsec)})


def tile_name(lat: int, lon: int) -> str:
    """The USGS name of the tile whose south-west corner is (lat, lon)."""
    return f"{abs(lat):02d}{'N' if lat >= 0 else 'S'}{abs(lon):03d}{'E' if lon >= 0 else 'W'}"


def tile_filename(lat: int, lon: int, res: str = DEFAULT_RES) -> str:
    return f"{tile_name(lat, lon)}_20101117_gmted_mea{res}.tif"


def tile_url(lat: int, lon: int, res: str = DEFAULT_RES) -> str:
    return URL.format(res=res, folder=f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}",
                      name=tile_name(lat, lon))


def tiles():
    """Every tile of the grid, north to south and west to east (96)."""
    return [(lat, lon) for lat in LAT_SOUTH for lon in LON_WEST]


def extra_dirs():
    return [Path(p) for p in os.environ.get("MAP_TOOL_GMTED_DIRS", "").split(";") if p.strip()]


_NAME_RE = re.compile(r"^(\d{2})([NS])(\d{3})([EW])_\d+_gmted_mea(075|150|300)\.tif$", re.I)


def index(refresh: bool = False):
    """{(lat, lon): (path, arcsec)} for the tiles on disk, the finest
    resolution of each. Cached until a folder's listing changes."""
    global _index_cache
    dirs = [GMTED_DIR] + extra_dirs()
    sig = tuple(sorted((str(d), f.name, f.stat().st_size) for d in dirs if d.is_dir()
                       for f in d.iterdir() if f.suffix.lower() == ".tif"))
    if not refresh and _index_cache[0] == sig:
        return _index_cache[1]
    by_res = {r: a for r, a in RESOLUTIONS}
    out = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            m = _NAME_RE.match(f.name)
            if not m:
                continue
            lat = int(m.group(1)) * (1 if m.group(2).upper() == "N" else -1)
            lon = int(m.group(3)) * (1 if m.group(4).upper() == "E" else -1)
            arc = by_res[m.group(5)]
            key = (lat, lon)
            if key not in out or arc < out[key][1]:
                out[key] = (f, arc)
    _index_cache = (sig, out)
    return out


def available() -> bool:
    return bool(index())


def missing_message() -> str:
    return ("The precise elevations aren't downloaded (Download tab -> GMTED2010 land elevations); "
            "the NASA topographic heightmap is used instead, in steps of ~26.5 m.")


# ------------------------------------------------------------------ tiles
_TAGS = {256: "width", 257: "height", 258: "bits", 259: "compression", 273: "strip_offsets",
         277: "samples", 278: "rows_per_strip", 279: "strip_bytes", 339: "sample_format",
         33550: "pixel_scale", 33922: "tiepoint"}
_TYPE = {1: "B", 2: "s", 3: "H", 4: "I", 5: "2I", 6: "b", 7: "s", 8: "h", 9: "i", 11: "f", 12: "d"}
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 11: 4, 12: 8}


def _header(f):
    """The tags of a TIFF's first image, by the names in _TAGS."""
    head = f.read(8)
    if head[:2] not in (b"II", b"MM"):
        raise ValueError("not a TIFF")
    en = "<" if head[:2] == b"II" else ">"
    f.seek(struct.unpack(en + "I", head[4:8])[0])
    out = {}
    for _ in range(struct.unpack(en + "H", f.read(2))[0]):
        tag, typ, cnt, raw = struct.unpack(en + "HHI4s", f.read(12))
        if typ not in _TYPE:
            continue
        size = _TYPE_SIZE[typ] * cnt
        if size > 4:
            here = f.tell()
            f.seek(struct.unpack(en + "I", raw)[0])
            raw = f.read(size)
            f.seek(here)
        out[_TAGS.get(tag, tag)] = struct.unpack(en + _TYPE[typ] * cnt, raw[:size])
    return en, out


def tile(lat: int, lon: int, arcsec: float):
    """Tile (lat, lon) on a grid of `arcsec` per pixel: an int16 array of
    (20 / arcsec, 30 / arcsec) degrees' worth of pixels, metres, NODATA
    where GMTED has no value. A finer tile on disk is subsampled (every
    k-th pixel, as the NASA path decodes 1/k), a coarser one repeated.
    Memoized with a byte budget. None if no tile covers it."""
    rows = int(round(TILE_LAT_DEG * 3600.0 / arcsec))
    cols = int(round(TILE_LON_DEG * 3600.0 / arcsec))
    key = (lat, lon, arcsec)
    with _lock:
        if key in _memo:
            _memo.move_to_end(key)
            return _memo[key]
    found = index().get((lat, lon))
    if found is None or (lat, lon) in _broken:
        return None
    path, arc = found
    try:
        return _read(path, arc, key, rows, cols, arcsec)
    except (OSError, ValueError, IndexError) as e:     # half a download, or not the file it claims
        _broken[(lat, lon)] = f"{path.name}: {e}"
        return None


def problems():
    """{(lat, lon): why} for the tiles that could not be read (a file left
    half-downloaded, say): they are skipped and their pixels fall back to the
    other source, and `core.elevation_m` passes this on in its report."""
    return dict(_broken)


def _read(path, arc, key, rows, cols, arcsec):
    with open(path, "rb") as f:
        en, t = _header(f)
        w, h = t["width"][0], t["height"][0]
        if t["compression"][0] != 1 or t["bits"][0] != 16 or t["samples"][0] != 1:
            raise ValueError(f"{path.name}: expected an uncompressed 16-bit tile")
        scale = t["pixel_scale"][1] * 3600.0 if "pixel_scale" in t else arc
        if abs(scale - arc) > 0.01 or abs(w * scale / 3600.0 - TILE_LON_DEG) > 0.01:
            raise ValueError(f"{path.name}: {scale:.2f} arcsec and {w}x{h} px don't match its name")
        # finer on disk than the grid asked for: every step-th pixel (as the
        # NASA path decodes 1/k); coarser: each pixel repeated rep times
        step = max(1, int(round(arcsec / arc)))
        rep = max(1, int(round(arc / arcsec)))
        n_r, n_c = (rows, cols) if rep == 1 else (rows // rep, cols // rep)
        if n_r * step > h or n_c * step > w:
            raise ValueError(f"{path.name}: {w}x{h} px is short of the {n_c * step}x{n_r * step} "
                             f"a {arc:g} arcsec tile has (half a download?)")
        per = t["rows_per_strip"][0]
        offs = t["strip_offsets"]
        out = np.empty((n_r, n_c), dtype=en + "i2")
        for i, r in enumerate(range(0, n_r * step, step)):
            strip, within = divmod(r, per)
            f.seek(offs[strip] + within * w * 2)
            row = np.frombuffer(f.read(n_c * step * 2), dtype=en + "i2")
            out[i] = row[::step] if step > 1 else row
    if rep > 1:
        out = np.repeat(np.repeat(out, rep, 0), rep, 1)[:rows, :cols]
    out = np.ascontiguousarray(out, dtype=np.int16)
    with _lock:
        _memo[key] = out
        while sum(a.nbytes for a in _memo.values()) > MEMO_BYTES and len(_memo) > 1:
            _memo.popitem(last=False)
    return out


# ------------------------------------------------------------------ sampling
def grid_arcsec(lon, lat, deg_px=None):
    """(arcsec per pixel to sample on, the tiles it will come from) for a
    set of points: the finest resolution their own tiles have, coarsened
    while it stays at least twice as fine as the view's pixel (`deg_px`),
    so a world map doesn't read 15 arcsec it cannot show. None if no tile
    covers any of the points."""
    idx = index()
    if not idx:
        return None, []
    rows = np.clip(((NORTH - np.asarray(lat)) // TILE_LAT_DEG).astype(np.int64), 0, len(LAT_SOUTH) - 1)
    cols = np.clip(((np.asarray(lon) + 180.0) // TILE_LON_DEG).astype(np.int64), 0, len(LON_WEST) - 1)
    keys = {(int(LAT_SOUTH[r]), int(LON_WEST[c])) for r, c in
            set(zip(rows.ravel().tolist(), cols.ravel().tolist()))}
    have = [k for k in keys if k in idx]
    if not have:
        return None, []
    arc = min(idx[k][1] for k in have)
    if deg_px:
        while arc * 2 <= deg_px * 3600.0 and arc < 30.0:
            arc *= 2
    return arc, sorted(have)


def heights_m(lon, lat, deg_px=None, arcsec=None):
    """Elevations in metres for arrays of lon / lat, bilinear over the tiles
    on disk: (metres float32, covered bool). Pixels no tile covers, or where
    GMTED has nodata, come back False -- their caller fills them in from
    another source. (None, None) when no tile covers anything."""
    if arcsec is None:
        arcsec, _used = grid_arcsec(lon, lat, deg_px)
    if arcsec is None:
        return None, None
    d = arcsec / 3600.0
    n_lat = int(round(TILE_LAT_DEG / d))
    n_lon = int(round(TILE_LON_DEG / d))
    n_rows, n_cols = len(LAT_SOUTH) * n_lat, len(LON_WEST) * n_lon
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    q = (NORTH - lat) / d - 0.5          # virtual-grid row / column, centres at integers
    u = (lon + 180.0) / d - 0.5
    i0 = np.floor(q).astype(np.int64)
    j0 = np.floor(u).astype(np.int64)
    fq, fu = q - i0, u - j0
    acc = np.zeros(q.shape, dtype=np.float64)
    ok = np.ones(q.shape, dtype=bool)
    for di, dj, wgt in ((0, 0, (1 - fq) * (1 - fu)), (0, 1, (1 - fq) * fu),
                        (1, 0, fq * (1 - fu)), (1, 1, fq * fu)):
        ii = i0 + di
        jj = (j0 + dj) % n_cols
        vals = np.zeros(q.shape, dtype=np.float64)
        good = (ii >= 0) & (ii < n_rows)
        tr = np.clip(ii, 0, n_rows - 1) // n_lat
        tc = jj // n_lon
        for t in set(zip(tr[good].ravel().tolist(), tc[good].ravel().tolist())):
            m = good & (tr == t[0]) & (tc == t[1])
            arr = tile(int(LAT_SOUTH[t[0]]), int(LON_WEST[t[1]]), arcsec)
            if arr is None:
                ok &= ~m
                continue
            v = arr[ii[m] - t[0] * n_lat, jj[m] - t[1] * n_lon]
            vals[m] = v
            if (v == NODATA).any():
                miss = np.zeros(q.shape, dtype=bool)
                miss[m] = v == NODATA
                ok &= ~miss
        ok &= good
        acc += wgt * vals
    return acc.astype(np.float32), ok


if __name__ == "__main__":
    idx = index()
    print(f"{GMTED_DIR}" + (f" + {[str(d) for d in extra_dirs()]}" if extra_dirs() else ""))
    print(f"{len(idx)} of {len(tiles())} tiles on disk")
    for res, arc in RESOLUTIONS:
        n = sum(1 for _p, a in idx.values() if a == arc)
        if n:
            print(f"  {n} at {arc:g} arcsec (mea{res})")
    for (lat, lon), (path, arc) in sorted(idx.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        print(f"  {tile_name(lat, lon)}  {arc:>4g}\"  {path}")
