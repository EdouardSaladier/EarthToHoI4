"""
Pre-build everything in `cache/` (plus the saved built-in colour presets and
river_borders.json), so
the folder can ship with the tool instead of each user rebuilding it on
first start.

Usage:
    python build_cache.py           # build whatever is missing
    python build_cache.py --clean   # delete stale cache entries first, then build

Cache entries are keyed on a *content* fingerprint of the GADM shapefiles
(core._cache_key), not their path or date, so a user who downloads the same
GADM release gets a matching cache.
"""

import shutil
import sys
import time

import core
import palettes


def main():
    q = lambda *a: None
    t_all = time.time()
    geometries = core.load_geometries(progress=q)
    tags = [t for t in geometries if t not in core.WATER_TAGS]

    if "--clean" in sys.argv:
        keep = {core._coast_cache_dir().name, core._overseas_cache_file().name,
                f"adjacency_{core._cache_key(core.SHAPEFILE_PATH)}_{core._COAST_GRID_DEG:g}_v2.json"}
        for entry in core.CACHE_DIR.iterdir():
            if entry.name not in keep:
                print(f"removing stale {entry.name}")
                shutil.rmtree(entry) if entry.is_dir() else entry.unlink()

    t = time.time()
    core.country_adjacency(geometries)
    print(f"country adjacency: {time.time() - t:.1f}s")

    t = time.time()
    for i, tag in enumerate(tags, 1):
        core.ring_topology(geometries, tag)
        print(f"\rring topology {i}/{len(tags)}", end="", flush=True)
    print(f"\rring topology: {len(tags)} countries in {time.time() - t:.1f}s")

    t = time.time()
    failed = []
    for i, tag in enumerate(tags, 1):
        try:
            core.mainland_ring_mask(geometries, tag)
        except core.DataNotFoundError:
            failed.append(tag)
        print(f"\roverseas territories {i}/{len(tags)}", end="", flush=True)
    note = f" (no GADM level 1/2 data: {failed})" if failed else ""
    print(f"\roverseas territories: {len(tags) - len(failed)} countries in {time.time() - t:.1f}s{note}")

    t = time.time()
    index = core.load_river_index()
    if index is None:
        print("river borders: skipped (HydroRIVERS not downloaded)")
    else:
        rb = core.river_borders(geometries, index, progress=q)
        print(f"river borders: {len(rb['stretches'])} stretches ({time.time() - t:.1f}s) in {core.RIVER_BORDERS_PATH.name}")

    t = time.time()
    presets = palettes.builtin_presets()
    print(f"built-in colour presets: {len(presets)} ({time.time() - t:.1f}s) in {palettes.BUILTIN_PRESETS_DIR}")

    size = sum(f.stat().st_size for f in core.CACHE_DIR.rglob("*") if f.is_file())
    print(f"done in {time.time() - t_all:.0f}s -- cache/ is {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
