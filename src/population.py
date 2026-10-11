"""Population of GADM 3.6 subdivisions (States tab).

Source: JRC GHS-DUC R2023A V2-0 (Download tab -> "Population by GADM
subdivision"): GHS-POP R2023A summed per GADM *4.1* unit, one CSV per level
and epoch; we read the 2025 `Tot_Pop` column of levels 0-5 once into
data/population/ghs_pop_2025.pkl (the zip stays zipped).

This tool uses GADM 3.6, so ids are matched (measured: 96.6% of level-1
ids, 78.6% of level 2, 90.1% of level 3, 97.5% of level 4, ~100% of level
5 exist in 4.1; the misses are countries GADM renumbered: Brazil and Mexico
level 2, Colombia, Uruguay, Ghana, Austria level 3, Cambodia, half the
Dutch municipalities). Per country and level, siblings (same parent) are
checked against their parent's resolved population:
  * all found but summing more than MISMATCH off the parent, or the found
    ones alone above the parent by more than MISMATCH -> the ids were
    renumbered: dropped;
  * the parent's remainder is split among the unmatched siblings by the
    GeoNames places (>= 1,000 people, own figure) inside each (point in
    polygon, shapely; places >= BIG_PLACE spread over their urban disc,
    radius sqrt(pop / (pi * URBAN_DENSITY)), by area -- as a point, Mexico
    City's 12 M all went to one borough), with AREA_WEIGHT of the split by
    area so a unit without such places still gets some; parent unknown ->
    one level up (down to the country total).
resolve(level, tag) -> {GID (3.6): (population, source)}; source "table"
(4.1 figure), "split" (from the parent), "places" (no parent figure: the
GeoNames places alone).
"""
import csv
import io
import pickle
import threading
import zipfile

import numpy as np

import core

ZIP_PATH = core.DATA_DIR / "population" / "GHS_DUC_MT_GLOBE_R2023A_V2_0.zip"
TABLE_PATH = core.DATA_DIR / "population" / "ghs_pop_2025.pkl"
EPOCH = 2025
MISMATCH = 0.2        # relative difference to the parent that marks renumbered ids
AREA_WEIGHT = 0.15    # share of a split made by area (the rest by GeoNames places)
BIG_PLACE = 100_000   # places this big are spread over their urban disc, not one point
URBAN_DENSITY = 7000.0  # people / km2 for that disc's radius (a city core: Sao Paulo's disc ~ its municipality)
SOURCE = "GHS-POP 2025 (JRC GHS-DUC R2023A, GADM 4.1 ids matched to 3.6)"

_lock = threading.RLock()
_table = None
_memo = {}


def available() -> bool:
    return ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 0


def _stamp():
    st = ZIP_PATH.stat()
    return (st.st_size, int(st.st_mtime))


def table():
    """{level: {GID (4.1): population}} for EPOCH, built from the zip once."""
    global _table
    with _lock:
        if _table is not None:
            return _table
        if TABLE_PATH.exists():
            try:
                with open(TABLE_PATH, "rb") as f:
                    data = pickle.load(f)
                if data.get("stamp") == _stamp():
                    _table = data["levels"]
                    return _table
            except Exception:
                pass
        levels = {}
        with zipfile.ZipFile(ZIP_PATH) as z:
            names = z.namelist()
            for lv in range(6):
                name = next((n for n in names if n.endswith(f"_{EPOCH}_level{lv}.csv")), None)
                if name is None:
                    continue
                with z.open(name) as fh:
                    rd = csv.reader(io.TextIOWrapper(fh, encoding="utf-8-sig"))
                    head = next(rd)
                    gi, pi = head.index(f"GID_{lv}"), head.index("Tot_Pop")
                    levels[lv] = {r[gi]: float(r[pi] or 0) for r in rd if len(r) > pi}
        with open(TABLE_PATH, "wb") as f:
            pickle.dump({"stamp": _stamp(), "levels": levels}, f)
        _table = levels
        return _table


def _parent(gid: str, level: int, tag: str) -> str:
    """GADM 3.6 parent id: "BRA.1.2_1" -> "BRA.1_1"; level 1 -> the country."""
    if level <= 1:
        return tag
    parts = gid.split(".")
    ver = parts[-1].split("_")[1] if "_" in parts[-1] else "1"
    return ".".join(parts[:-1]) + "_" + ver


def _split_weights(level: int, gids, tag: str) -> np.ndarray:
    """Weights of some level-`level` units of `tag`: GeoNames places inside
    (1 - AREA_WEIGHT) + area (AREA_WEIGHT)."""
    import shapely
    import states
    idx = states.level_index(level)
    pos = {g: i for g, i in zip(idx["gid"][idx["by_tag"][tag]].tolist(), idx["by_tag"][tag].tolist())}
    polys, areas = [], []
    for g in gids:
        rings = states._record_rings(idx, pos[g]) if g in pos else []
        ps = [shapely.Polygon(r) for _h, r in rings if len(r) >= 4]
        ps = [p if p.is_valid else p.buffer(0) for p in ps]
        geom = shapely.union_all(ps) if ps else shapely.Polygon()
        polys.append(geom)
        c = geom.centroid
        areas.append(geom.area * (np.cos(np.radians(c.y)) if not geom.is_empty else 0.0))
    areas = np.asarray(areas, dtype=np.float64)
    place = np.zeros(len(gids))
    try:
        import cities
        if cities.available() and len(polys):
            ci = cities.load_city_index()
            b = np.array([p.bounds for p in polys if not p.is_empty] or [(0, 0, 0, 0)])
            sel = ((ci["lon"] >= b[:, 0].min()) & (ci["lon"] <= b[:, 2].max()) &
                   (ci["lat"] >= b[:, 1].min()) & (ci["lat"] <= b[:, 3].max()))
            if sel.any():
                lon, lat = ci["lon"][sel], ci["lat"][sel]
                pop = ci["own_population"][sel].astype(np.float64)
                tree = shapely.STRtree(polys)
                small = pop < BIG_PLACE
                # small places: the unit they lie in
                pi, gi = tree.query(shapely.points(lon[small], lat[small]), predicate="within")
                first = np.unique(pi, return_index=True)[1]  # a place counts once
                np.add.at(place, gi[first], pop[small][pi[first]])
                # big ones: spread over their urban disc by area (one point
                # held all of Mexico City in one borough)
                for x, y, p_ in zip(lon[~small], lat[~small], pop[~small]):
                    r_km = np.sqrt(p_ / (np.pi * URBAN_DENSITY))
                    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
                    disc = shapely.Polygon(np.column_stack([
                        x + r_km / (111.32 * max(np.cos(np.radians(y)), 0.05)) * np.cos(ang),
                        y + r_km / 110.57 * np.sin(ang)]))
                    hit = tree.query(disc, predicate="intersects")
                    if not len(hit):
                        continue
                    a = np.array([polys[k].intersection(disc).area for k in hit])
                    if a.sum() > 0:
                        place[hit] += p_ * a / a.sum()
    except Exception:
        pass
    w = np.zeros(len(gids))
    if place.sum() > 0:
        w += (1.0 - AREA_WEIGHT) * place / place.sum()
    if areas.sum() > 0:
        w += (AREA_WEIGHT if place.sum() > 0 else 1.0) * areas / areas.sum()
    if w.sum() <= 0:
        w[:] = 1.0
    return w / w.sum()


def resolve(level: int, tag: str) -> dict:
    """{GID (3.6): (population or None, source)} for every level-`level`
    unit of `tag` (see the module docstring)."""
    key = (level, tag)
    with _lock:
        if key in _memo:
            return _memo[key]
    T = table()
    if level == 0:
        p = T.get(0, {}).get(tag)
        if p is None:
            s = [v for g, v in T.get(1, {}).items() if g.split(".")[0] == tag]
            p = float(sum(s)) if s else None
        out = {tag: (p, "table")}
    else:
        import states
        idx = states.level_index(level)
        out = {}
        if idx is not None and tag in idx["by_tag"]:
            gids = idx["gid"][idx["by_tag"][tag]].tolist()
            parents = resolve(level - 1, tag)
            groups = {}
            for g in gids:
                groups.setdefault(_parent(g, level, tag), []).append(g)
            Tl = T.get(level, {})
            for par, kids in groups.items():
                P = parents.get(par, (None, None))[0]
                found = {g: Tl[g] for g in kids if g in Tl}
                S = sum(found.values())
                if P is not None and found:
                    if len(found) == len(kids) and abs(S - P) > MISMATCH * max(P, 1.0):
                        found = {}
                    elif S > (1.0 + MISMATCH) * max(P, 1.0):
                        found = {}
                    S = sum(found.values())
                for g, v in found.items():
                    out[g] = (v, "table")
                rest = [g for g in kids if g not in found]
                if not rest:
                    continue
                w = _split_weights(level, rest, tag)
                if P is not None:
                    R = max(P - S, 0.0)
                    for g, wi in zip(rest, w):
                        out[g] = (R * wi, "split")
                else:
                    for g in rest:
                        out[g] = (None, "places")
    with _lock:
        _memo[key] = out
    return out


def population_of(level: int, tag: str, gid) -> tuple:
    """(population or None, source) of one GADM 3.6 unit."""
    if not available():
        return None, None
    r = resolve(level, tag)
    return r.get(tag if level == 0 else str(gid), (None, None))
