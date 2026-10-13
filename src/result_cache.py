"""Disk cache of the app's results (user: "not regenerate everything each
time I open / change the tool"): every stage's output -- the country render,
rivers, states, terrain, provinces, strategic regions, ... -- is kept in
cache/results/<stage>/<key>.pkl.z and reused across server restarts.

Keys are *content* keys: a hash of the stage name, the source code of the
modules the stage depends on (STAGE_CODE), the stage's own settings and the
content keys of the results it was made from. So a result is reused only
when nothing that went into it changed, and changing a module's code (or a
setting) makes that stage -- and every stage built on it, through their
keys -- compute afresh, while the stages before it stay cached.

Files: pickle (protocol 5) + zlib level 1 (the render's 230 MB of masks ->
1.8 MB in 0.4 s). Pickled in the request thread (objects returned to the
caller may be modified later), compressed and written by one background
thread, atomically (temp file + rename). The cache is capped at
MAX_BYTES: least recently used files go first (a hit touches its file)."""
import hashlib
import json
import os
import pickle
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # src/: the modules whose code keys the stages
CACHE_DIR = ROOT.parent / "cache" / "results"
CACHE_VERSION = 1            # bump to drop every cached result
MAX_BYTES = 4_000_000_000    # the whole results cache on disk
ENABLED = os.environ.get("MAP_TOOL_NO_RESULT_CACHE", "") == ""

# stage -> the modules its result depends on (its own code only: upstream
# stages are covered by their content keys inside this stage's key)
STAGE_CODE = {
    "render": ("core.py", "palettes.py"),
    "recolor": ("core.py", "palettes.py"),   # the render's colours (memory only, see app._render_colored_from_args)
    "eez_zones": ("eez.py", "states.py"),
    "eez_overlay": ("eez.py", "states.py"),
    "rivers": ("core.py",),
    "cities": ("cities.py", "city_names.py"),
    "heightmap": ("core.py",),
    "height_levels": ("core.py",),
    "density": ("density.py",),
    "states": ("states.py", "population.py", "density.py"),
    "terrain": ("terrain.py", "landcover.py", "core.py"),
    "colormap": ("colormap.py",),
    "province_eez": ("eez.py", "states.py"),
    "provinces": ("provinces.py", "states.py", "terrain.py", "core.py"),
    "strategic": ("strategic.py", "climate.py", "bathymetry.py"),
}

_code_hash = {}
_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="result-cache")
_lock = threading.Lock()
stats = {"hits": 0, "misses": 0, "writes": 0, "errors": 0}


def _file_hash(name):
    if name not in _code_hash:
        try:
            _code_hash[name] = hashlib.sha1((ROOT / name).read_bytes()).hexdigest()
        except OSError:
            _code_hash[name] = "missing"
    return _code_hash[name]


def digest(obj) -> str:
    """Stable hash of JSON-able data (tuples as lists; not Python's hash(),
    which is salted per process for strings)."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def key(stage: str, *parts) -> str:
    """Content key of a stage's result made from `parts` (settings and the
    content keys of the upstream results)."""
    code = [_file_hash(f) for f in STAGE_CODE.get(stage, ())]
    return f"{stage}-" + digest([CACHE_VERSION, stage, code, list(parts)])[:32]


def _path(stage, k):
    return CACHE_DIR / stage / f"{k}.pkl.z"


def exists(stage, k) -> bool:
    return ENABLED and _path(stage, k).exists()


def load(stage, k):
    """The cached result, or None."""
    if not ENABLED:
        return None
    p = _path(stage, k)
    try:
        data = p.read_bytes()
    except OSError:
        stats["misses"] += 1
        return None
    try:
        obj = pickle.loads(zlib.decompress(data))
    except Exception:  # corrupt / written by an incompatible version
        stats["errors"] += 1
        try:
            p.unlink()
        except OSError:
            pass
        return None
    try:
        os.utime(p)  # least recently used = oldest mtime
    except OSError:
        pass
    stats["hits"] += 1
    return obj


def save(stage, k, obj):
    """Pickle now, compress + write in the background."""
    if not ENABLED:
        return
    try:
        raw = pickle.dumps(obj, protocol=5)
    except Exception as e:
        stats["errors"] += 1
        print(f"result cache: {stage} not cacheable ({e})")
        return
    _writer.submit(_write, stage, k, raw)


def _write(stage, k, raw):
    try:
        p = _path(stage, k)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".tmp{threading.get_ident()}")
        tmp.write_bytes(zlib.compress(raw, 1))
        os.replace(tmp, p)
        stats["writes"] += 1
        _prune()
    except Exception as e:
        stats["errors"] += 1
        print(f"result cache: writing {stage} failed ({e})")


def _prune():
    files = []
    total = 0
    for p in CACHE_DIR.glob("*/*.pkl.z"):
        try:
            st = p.stat()
        except OSError:
            continue
        files.append((st.st_mtime, st.st_size, p))
        total += st.st_size
    if total <= MAX_BYTES:
        return
    for _m, size, p in sorted(files):
        try:
            p.unlink()
            total -= size
        except OSError:
            pass
        if total <= MAX_BYTES * 0.9:
            break


def usage():
    """{stage: (files, bytes)} and the total, for the UI."""
    out, total = {}, 0
    for p in CACHE_DIR.glob("*/*.pkl.z"):
        try:
            size = p.stat().st_size
        except OSError:
            continue
        n, b = out.get(p.parent.name, (0, 0))
        out[p.parent.name] = (n + 1, b + size)
        total += size
    return {"stages": out, "bytes": total, "max_bytes": MAX_BYTES, "enabled": ENABLED, **stats}


def clear():
    n = 0
    for p in CACHE_DIR.glob("*/*.pkl.z"):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


def flush(timeout=60.0):
    """Wait for pending writes (tests, shutdown)."""
    done = threading.Event()
    _writer.submit(done.set)
    done.wait(timeout)
