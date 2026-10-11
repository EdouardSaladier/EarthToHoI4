"""Copernicus Global Land Cover (CGLS-LC100, epoch 2015, discrete
classification) access: read the global GeoTIFF, keep a pyramid of reduced
copies, and sample it onto a render's frame.

The source (Download tab -> Copernicus Land Cover) is one 2 GB BigTIFF,
362880 x 141120 px (1/1008 degree, ~100 m), 80 N .. 60 S, one LZW strip per
row, 8-bit class codes (CLASS_NAMES). Nothing here reads GeoTIFFs in
general: Pillow decodes the LZW strips once they're wrapped in a minimal
in-memory TIFF (`_decode_rows`, ~1 ms per full row).

Reading a continent from the source would mean decompressing ~10 GB, so a
pyramid of reduced copies is built once (`build_pyramid`, ~5-8 min, started
by the app at start-up when missing): levels 2..6 = 4x .. 64x reduced, each
pixel the *most common* class of its block (exact counts, not a mode of
modes), stored as zlib-compressed 512 x 512 tiles in
data/landcover/pyramid/level<k>.bin + .idx.npy (+ done.json when complete).
`sample` picks the coarsest level still at least as fine as the view's
finest pixel; views finer than level 2 (~440 m) read the source rows
directly (a country, a few seconds).
"""
import io
import json
import struct
import threading
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

import core

SOURCE_NAME = "ProbaV_LC100_epoch2015_global_v2.0.2_discrete-classification_EPSG-4326.tif"
SOURCE_PATH = core.DATA_DIR / "landcover" / SOURCE_NAME
PYRAMID_DIR = core.DATA_DIR / "landcover" / "pyramid"
PYRAMID_VERSION = 1
LEVELS = (2, 3, 4, 5, 6)      # reduction factor 2**k
TILE = 512
BAND_ROWS = 64                # source rows per build step (= 1 row of level 6)
LON0, LAT0 = -180.0, 80.0     # top-left corner of the source (pixel edges)
DEG_PER_PX = 1.0 / 1008.0     # source pixel size

# Copernicus LC100 discrete classes.
CLASS_NAMES = {
    0: "Unknown", 20: "Shrubs", 30: "Herbaceous vegetation", 40: "Cropland", 50: "Urban / built up",
    60: "Bare / sparse vegetation", 70: "Snow and ice", 80: "Permanent water", 90: "Herbaceous wetland",
    100: "Moss and lichen",
    111: "Closed forest, evergreen needle leaf", 112: "Closed forest, evergreen broad leaf",
    113: "Closed forest, deciduous needle leaf", 114: "Closed forest, deciduous broad leaf",
    115: "Closed forest, mixed", 116: "Closed forest, unknown",
    121: "Open forest, evergreen needle leaf", 122: "Open forest, evergreen broad leaf",
    123: "Open forest, deciduous needle leaf", 124: "Open forest, deciduous broad leaf",
    125: "Open forest, mixed", 126: "Open forest, unknown", 200: "Open sea", 255: "No data",
}
_CODES = np.array(sorted(CLASS_NAMES), dtype=np.uint8)          # compact index -> code
_TO_IDX = np.zeros(256, dtype=np.uint8)
_TO_IDX[_CODES] = np.arange(len(_CODES), dtype=np.uint8)       # code -> compact index
NCLS = len(_CODES)


SOURCE_DIR = core.DATA_DIR / "landcover"
SOURCE_SHAPE = (362880, 141120)   # width, height of the global discrete-classification map

_inspected = {}                   # (path, size, mtime_ns) -> None (usable) or the problem


def _inspect(path: Path):
    """None if `path` is the complete global LC100 discrete-classification
    BigTIFF, else what is wrong with it (reads the header and the last
    strip's offset only)."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key in _inspected:
        return _inspected[key]
    try:
        with open(path, "rb") as f:
            head = f.read(16)
            ifd = struct.unpack("<Q", head[8:])[0] if head[:4] == b"II+\x00" and len(head) == 16 else None
            if ifd is None:
                problem = "not a BigTIFF"
            elif ifd >= st.st_size:  # the header is at the end of this file
                problem = f"incomplete ({st.st_size / 1e9:.2f} GB of ~2.07 GB): download it again"
            else:
                f.seek(ifd)
                tags = {}
                for _ in range(struct.unpack("<Q", f.read(8))[0]):
                    tag, typ, cnt = struct.unpack("<HHQ", f.read(12))
                    tags[tag] = (typ, cnt, f.read(8))
                shape = (_Source._scalar(tags[256]), _Source._scalar(tags[257]))
                if shape != SOURCE_SHAPE:
                    problem = (f"{shape[0]} x {shape[1]} px, not the global map "
                               f"({SOURCE_SHAPE[0]} x {SOURCE_SHAPE[1]})")
                elif 320 not in tags:  # the cover-fraction / probability layers have the same size, no colour table
                    problem = "not the discrete-classification map (another Copernicus layer?)"
                else:
                    # End of the last strip vs the file size: an interrupted download.
                    ends = []
                    for t in (273, 279):
                        typ, cnt, val = tags[t]
                        size = {3: 2, 4: 4, 16: 8}[typ]
                        f.seek(struct.unpack("<Q", val)[0] + size * (cnt - 1))
                        ends.append(int.from_bytes(f.read(size), "little"))
                    need = ends[0] + ends[1]
                    problem = (None if need <= st.st_size else
                               f"incomplete ({st.st_size / 1e9:.2f} of {need / 1e9:.2f} GB): download it again")
    except (OSError, KeyError, struct.error, ValueError) as e:
        problem = f"unreadable ({type(e).__name__})"
    _inspected[key] = problem
    return problem


def _candidates():
    """The Download tab's file first, then any other .tif / .tiff under
    data/landcover/ (a manual download saved under another name -- "(1)",
    a sub-folder), largest first."""
    out = [SOURCE_PATH] if SOURCE_PATH.is_file() else []
    if SOURCE_DIR.is_dir():
        others = [p for p in SOURCE_DIR.rglob("*") if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
                  and p != SOURCE_PATH and PYRAMID_DIR not in p.parents]
        out += sorted(others, key=lambda p: -p.stat().st_size)
    return out


def find_source():
    """Path of the usable land cover GeoTIFF, or None."""
    for p in _candidates():
        if _inspect(p) is None:
            return p
    return None


def available() -> bool:
    return find_source() is not None


def missing_message() -> str:
    """Why the land cover can't be used, for the UI."""
    found = [(p, _inspect(p)) for p in _candidates()]
    if not found:
        return (f"The Copernicus land cover isn't downloaded (Download tab -> Copernicus Land Cover, "
                f"or put {SOURCE_NAME} in {SOURCE_DIR})")
    return ("The Copernicus land cover file can't be used: "
            + "; ".join(f"{p.relative_to(SOURCE_DIR)}: {why}" for p, why in found)
            + f". Expected the global discrete-classification map {SOURCE_NAME} (2.07 GB) in {SOURCE_DIR}")


# ---- source reader ---------------------------------------------------------

class _Source:
    """Strip offsets / counts of the BigTIFF + an open file (thread-safe reads)."""

    def __init__(self, path: Path):
        self.f = open(path, "rb")
        self.lock = threading.Lock()
        f = self.f
        if f.read(4) != b"II+\x00":
            raise ValueError(f"{path.name}: not a little-endian BigTIFF")
        f.seek(8)
        f.seek(struct.unpack("<Q", f.read(8))[0])
        tags = {}
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            tag, typ, cnt = struct.unpack("<HHQ", f.read(12))
            tags[tag] = (typ, cnt, f.read(8))
        self.width = self._scalar(tags[256])
        self.height = self._scalar(tags[257])
        if self._scalar(tags[259]) != 5 or self._scalar(tags.get(278, (4, 1, struct.pack("<Q", 1)))) != 1:
            raise ValueError(f"{path.name}: expected LZW, one row per strip")
        self.offsets = self._array(tags[273]).astype(np.int64)
        self.counts = self._array(tags[279]).astype(np.int64)
        # ColorMap (Copernicus' own class colours): 3 x 256 16-bit values.
        cmap = self._array(tags[320]).astype(np.int64) if 320 in tags else None
        self.palette = ((cmap.reshape(3, 256).T >> 8).astype(np.uint8)
                        if cmap is not None and len(cmap) == 768 else None)

    @staticmethod
    def _scalar(t):
        typ, _cnt, val = t
        return int(struct.unpack({3: "<H", 4: "<I", 16: "<Q"}[typ], val[:{3: 2, 4: 4, 16: 8}[typ]])[0])

    def _array(self, t):
        typ, cnt, val = t
        dt = {3: "<u2", 4: "<u4", 16: "<u8"}[typ]
        size = np.dtype(dt).itemsize * cnt
        if size <= 8:
            return np.frombuffer(val[:size], dtype=dt)
        self.f.seek(struct.unpack("<Q", val)[0])
        return np.frombuffer(self.f.read(size), dtype=dt)

    def rows(self, r0: int, r1: int) -> np.ndarray:
        """Source rows r0..r1-1 (uint8, full width)."""
        with self.lock:
            self.f.seek(int(self.offsets[r0]))
            data = self.f.read(int(self.offsets[r1 - 1] + self.counts[r1 - 1] - self.offsets[r0]))
        return _decode_rows(data, self.offsets[r0:r1] - self.offsets[r0], self.counts[r0:r1], self.width)


def _decode_rows(data: bytes, rel_offsets, counts, width: int) -> np.ndarray:
    """LZW strips (one row each) -> uint8 (n, width), via a minimal classic
    TIFF around them that Pillow's libtiff decoder reads."""
    n = len(counts)
    n_tags = 9
    arr_off = 8 + 2 + 12 * n_tags + 4
    data_off = arr_off + 8 * n
    so = (np.asarray(rel_offsets, dtype=np.int64) + data_off).astype("<u4")
    sc = np.asarray(counts, dtype="<u4")
    entries = [(256, 4, 1, width), (257, 4, 1, n), (258, 3, 1, 8), (259, 3, 1, 5), (262, 3, 1, 1),
               (273, 4, n, arr_off if n > 1 else int(so[0])), (277, 3, 1, 1), (278, 4, 1, 1),
               (279, 4, n, arr_off + 4 * n if n > 1 else int(sc[0]))]
    b = bytearray(b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", n_tags))
    for tag, typ, cnt, value in entries:
        b += struct.pack("<HHIHH", tag, typ, cnt, value, 0) if typ == 3 else struct.pack("<HHII", tag, typ, cnt, value)
    b += struct.pack("<I", 0) + so.tobytes() + sc.tobytes()
    b += data
    return np.asarray(Image.open(io.BytesIO(bytes(b))))


_source = None
_source_lock = threading.Lock()


def _get_source() -> _Source:
    global _source
    with _source_lock:
        if _source is None:
            path = find_source()
            if path is None:
                raise core.DataNotFoundError(missing_message())
            _source = _Source(path)
        return _source


# ---- pyramid ---------------------------------------------------------------

_build = {"status": "idle", "done_rows": 0, "total_rows": 0, "error": None, "started": None}
_build_lock = threading.Lock()


_pyramid_checked = {}   # file stats -> ready?


def pyramid_ready() -> bool:
    """done.json matches this version and source, and every level's tiles
    are all there (a .bin cut short -- a build interrupted after an old
    done.json, a copy that stopped -- would fail on reading)."""
    p = PYRAMID_DIR / "done.json"
    src = find_source()
    if not p.exists() or src is None:
        return False
    files = [p] + [PYRAMID_DIR / f"level{k}{ext}" for k in LEVELS for ext in (".bin", ".idx.npy")]
    try:
        key = tuple((st.st_size, st.st_mtime_ns) for st in map(Path.stat, files)) + (src.stat().st_size,)
    except OSError:
        return False
    if key in _pyramid_checked:
        return _pyramid_checked[key]
    try:
        info = json.loads(p.read_text())
        ok = info.get("version") == PYRAMID_VERSION and info.get("source_size") == src.stat().st_size
        for k in LEVELS if ok else ():
            idx = np.load(PYRAMID_DIR / f"level{k}.idx.npy")
            ok = ok and int((idx[..., 0] + idx[..., 1]).max()) <= (PYRAMID_DIR / f"level{k}.bin").stat().st_size
    except (OSError, ValueError):
        ok = False
    _pyramid_checked[key] = ok
    return ok


def build_status() -> dict:
    with _build_lock:
        st = dict(_build)
    st["ready"] = available() and pyramid_ready()
    st["available"] = available()
    if st["total_rows"]:
        st["progress"] = round(st["done_rows"] / st["total_rows"], 4)
    return st


def start_build():
    """Build the pyramid in a background thread (no-op if ready or running)."""
    with _build_lock:
        if _build["status"] == "running" or not available() or pyramid_ready():
            return
        _build.update(status="running", done_rows=0, error=None, started=time.time())
    threading.Thread(target=_build_thread, daemon=True).start()


def _build_thread():
    print("Preparing the land cover for the Terrain tab (one-time, about 10-20 min).\n"
          "  Keep EarthToHoI4 open until it's done: closing it restarts this step from zero.", flush=True)
    last = [-1]

    def progress(_msg):
        with _build_lock:
            pct = int(100 * _build["done_rows"] / max(1, _build["total_rows"]))
        if pct // 10 != last[0] // 10:
            last[0] = pct
            print(f"  Land cover: {pct}%", flush=True)
    try:
        build_pyramid(progress=progress)
        with _build_lock:
            _build["status"] = "done"
        print("Land cover ready.", flush=True)
    except Exception as e:  # reported to the UI
        with _build_lock:
            _build.update(status="error", error=f"{type(e).__name__}: {e}")
        print(f"Preparing the land cover failed: {type(e).__name__}: {e}", flush=True)


def build_pyramid(progress=None):
    src = _get_source()
    W, H = src.width, src.height
    PYRAMID_DIR.mkdir(parents=True, exist_ok=True)
    (PYRAMID_DIR / "done.json").unlink(missing_ok=True)
    with _build_lock:
        _build["total_rows"] = H
    writers = {k: _LevelWriter(k, W >> k, -(-H // (1 << k))) for k in LEVELS}
    base = LEVELS[0]
    f0 = 1 << base
    cols0 = W // f0
    br = BAND_ROWS // f0
    # Level-`base` block of each pixel of a band, times NCLS (same every band).
    block_key = ((np.arange(BAND_ROWS)[:, None] // f0) * cols0
                 + (np.arange(cols0 * f0)[None, :] // f0)).astype(np.int64) * NCLS
    t0 = time.time()
    for r0 in range(0, H, BAND_ROWS):
        r1 = min(H, r0 + BAND_ROWS)
        rows = src.rows(r0, r1)
        n = r1 - r0
        pad = (-n) % BAND_ROWS
        if pad:  # last band: pad with "no data"
            rows = np.vstack([rows, np.full((pad, W), 255, np.uint8)])
        idx = _TO_IDX[rows[:, : cols0 * f0]]
        # Exact class counts per level-`base` block: bincount of block * NCLS + class.
        counts = np.bincount((block_key + idx).ravel(), minlength=br * cols0 * NCLS)
        counts = counts.reshape(br, cols0, NCLS).astype(np.uint16)
        for k in LEVELS:
            writers[k].add(_CODES[counts.argmax(axis=2)])
            if k != LEVELS[-1]:
                counts = counts.reshape(counts.shape[0] // 2, 2, counts.shape[1] // 2, 2, NCLS).sum(axis=(1, 3))
        with _build_lock:
            _build["done_rows"] = r1
        if progress and (r0 // BAND_ROWS) % 100 == 0:
            progress(f"  land cover pyramid: {r1:,}/{H:,} rows ({time.time() - t0:.0f}s)")
    for w in writers.values():
        w.close()
    (PYRAMID_DIR / "done.json").write_text(json.dumps({"version": PYRAMID_VERSION, "levels": list(LEVELS),
                                                         "source_size": find_source().stat().st_size,
                                                         "seconds": round(time.time() - t0)}))


class _LevelWriter:
    """Accumulates rows of one level, writes zlib tiles one tile-row at a time."""

    def __init__(self, k, width, height):
        self.k, self.width, self.height = k, width, height
        self.nx, self.ny = -(-width // TILE), -(-height // TILE)
        self.index = np.zeros((self.ny, self.nx, 2), dtype=np.int64)  # offset, length
        self.f = open(PYRAMID_DIR / f"level{k}.bin", "wb")
        self.buf = []
        self.rows = 0
        self.ty = 0

    def add(self, rows):
        self.buf.append(rows[:, : self.width])
        self.rows += len(rows)
        if self.rows >= TILE:
            self._flush()

    def _flush(self):
        if not self.buf or self.ty >= self.ny:
            return
        band = np.vstack(self.buf)
        take, rest = band[:TILE], band[TILE:]
        if len(take) < TILE:
            take = np.vstack([take, np.full((TILE - len(take), take.shape[1]), 255, np.uint8)])
        if take.shape[1] < self.nx * TILE:
            take = np.hstack([take, np.full((TILE, self.nx * TILE - take.shape[1]), 255, np.uint8)])
        for tx in range(self.nx):
            z = zlib.compress(np.ascontiguousarray(take[:, tx * TILE:(tx + 1) * TILE]).tobytes(), 6)
            self.index[self.ty, tx] = (self.f.tell(), len(z))
            self.f.write(z)
        self.ty += 1
        self.buf = [rest] if len(rest) else []
        self.rows = len(rest)

    def close(self):
        while self.rows > 0 and self.ty < self.ny:
            self._flush()
        self.f.close()
        np.save(PYRAMID_DIR / f"level{self.k}.idx.npy", self.index)


class _Level:
    def __init__(self, k):
        self.k = k
        self.f = open(PYRAMID_DIR / f"level{k}.bin", "rb")
        self.index = np.load(PYRAMID_DIR / f"level{k}.idx.npy")
        self.lock = threading.Lock()

    def tile(self, ty, tx):
        off, n = self.index[ty, tx]
        with self.lock:
            self.f.seek(int(off))
            z = self.f.read(int(n))
        return np.frombuffer(zlib.decompress(z), dtype=np.uint8).reshape(TILE, TILE)


_levels = {}
_levels_lock = threading.Lock()


def _get_level(k) -> _Level:
    with _levels_lock:
        if k not in _levels:
            _levels[k] = _Level(k)
        return _levels[k]


def palette():
    """Copernicus' class colours (256 x 3 uint8, index = class code), from
    the source file's colour table (fallback: greys)."""
    src = _get_source()
    if src.palette is not None:
        return src.palette
    return np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)


def _gather(get_block, rows, cols, n_rows, n_cols):
    """Values at integer (rows, cols) of a raster read by blocks of
    `get_block(r0, r1, c0, c1)`, reading only the bbox of the points."""
    out = np.full(rows.shape, 255, dtype=np.uint8)
    ok = (rows >= 0) & (rows < n_rows)
    if not ok.any():
        return out
    r0, r1 = int(rows[ok].min()), int(rows[ok].max()) + 1
    c0, c1 = int(cols[ok].min()), int(cols[ok].max()) + 1
    block = get_block(r0, r1, c0, c1)
    out[ok] = block[rows[ok] - r0, cols[ok] - c0]
    return out


def sample(lon, lat, deg_per_px: float):
    """Land cover codes (uint8) at lon/lat arrays, from the coarsest level
    at least as fine as `deg_per_px` (the view's finest pixel, in degrees).
    Longitudes wrap. Returns (codes, level used; 0 = source)."""
    lon = (np.asarray(lon, dtype=np.float64) + 180.0) % 360.0 - 180.0
    lat = np.asarray(lat, dtype=np.float64)
    level = 0
    if pyramid_ready():
        for k in LEVELS:
            if DEG_PER_PX * (1 << k) <= deg_per_px:
                level = k
        if level == 0 and (lat.max() - lat.min()) > 25.0:
            level = LEVELS[0]  # too big to read from the source: finest cached level
    f = 1 << level
    d = DEG_PER_PX * f
    rows = np.floor((LAT0 - lat) / d).astype(np.int64)
    cols = np.floor((lon - LON0) / d).astype(np.int64)
    src = _get_source()
    n_rows, n_cols = -(-src.height // f), src.width // f
    cols = np.clip(cols, 0, n_cols - 1)
    if level == 0:
        def get_block(r0, r1, c0, c1):
            out = np.empty((r1 - r0, c1 - c0), dtype=np.uint8)
            for a in range(r0, r1, 256):
                b = min(r1, a + 256)
                out[a - r0:b - r0] = src.rows(a, b)[:, c0:c1]
            return out
    else:
        lv = _get_level(level)

        def get_block(r0, r1, c0, c1):
            ty0, ty1, tx0, tx1 = r0 // TILE, (r1 - 1) // TILE, c0 // TILE, (c1 - 1) // TILE
            big = np.empty(((ty1 - ty0 + 1) * TILE, (tx1 - tx0 + 1) * TILE), dtype=np.uint8)
            for ty in range(ty0, ty1 + 1):
                for tx in range(tx0, tx1 + 1):
                    big[(ty - ty0) * TILE:(ty - ty0 + 1) * TILE, (tx - tx0) * TILE:(tx - tx0 + 1) * TILE] = lv.tile(ty, tx)
            return big[r0 - ty0 * TILE:r1 - ty0 * TILE, c0 - tx0 * TILE:c1 - tx0 * TILE]
    # A view across the antimeridian would make the column bbox span the
    # world: read the two sides separately.
    west = cols < n_cols // 2
    if (cols[west].max(initial=-1) - cols[west].min(initial=0) < n_cols // 4
            and cols[~west].max(initial=-1) - cols[~west].min(initial=0) < n_cols // 4
            and west.any() and (~west).any() and cols.max() - cols.min() > n_cols // 2):
        out = np.empty(rows.shape, dtype=np.uint8)
        out[west] = _gather(get_block, rows[west], cols[west], n_rows, n_cols)
        out[~west] = _gather(get_block, rows[~west], cols[~west], n_rows, n_cols)
        return out, level
    return _gather(get_block, rows, cols, n_rows, n_cols), level


if __name__ == "__main__":
    t = time.time()
    build_pyramid(progress=print)
    print(f"done in {time.time() - t:.0f}s")
