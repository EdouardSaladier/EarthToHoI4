"""States tab: HOI4-like states from GADM subdivisions.

Per visible country, a GADM level (0 = the country itself .. 5) sized for
HOI4 states (`target_px`, default the vanilla mean: Blank_state_map.png,
5632 x 2048, 1,134 states >= 50 px, mean 3,413 px, median 1,278) -- size =
the country's pixels in view / its subdivisions whose bbox centre is in
view. User: rather aim lower and merge than aim bigger -> the coarsest
level with a mean size <= target (none: the finest); at least level 1
unless the country is under the merge threshold (then level 0).
Per-country overrides (`levels`).

Rules (user):
  1. States follow the country borders exactly: every pixel of a country on
     the country map (the Rivers tab's fitted one when rivers are used)
     belongs to a subdivision of that country -- pixels a neighbour's
     subdivision covers (GADM levels don't share the rasterized level-0
     borders; the river fitting moves them) go to the nearest one of its own.
  2. Where a state border comes within RIVER_SNAP_PX (3) of a river, it
     follows the river, within rule 1: pixels within 3 px of a river (not the
     river itself) are re-assigned from the state areas outside that band,
     by a flood fill that can't cross river pixels or leave the country --
     each side of a river gets the state that holds that side just beyond
     the band. River pixels then join a state beside them (their own when
     it is one). States wholly inside the band keep their pixels.
Colours: the country's colour, jittered a little in HSV per state (seeded
by the GADM id: stable across renders), unique over all states and
different from the country colour; neighbouring states at least
MIN_SIBLING_DIST apart (RGB) when possible.

GADM levels are read with numpy (no pyshp): the .dbf columns GID_0 / GID_n /
NAME_n and every record's bbox from its .shp header (via the .shx offsets),
memory-mapped; a record's points are read only when drawn.
"""
import colorsys
import math
import struct
import threading
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

import core

TARGET_PX_DEFAULT = 3400      # vanilla mean state size (Blank_state_map.png)
RIVER_SNAP_PX = 3
MIN_SIBLING_DIST = 10.0
MAX_LEVEL = 5
CROSS4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
WHOLE_UNIT_FACTOR = 1.5  # a GADM unit up to this x target_px is one state, whatever the density
EXCLAVE_MIN_PX = 20   # a state piece the GADM raster also has apart is kept (a real exclave) from this size
MERGE_PX_DEFAULT = 2000       # states under this are merged into a neighbour (0 = off)
CLIP_SHARE = 0.9             # a unit showing less than this share of its area (frame edge) gets that share of its people
POP_SNAP_PX = 3               # a place off the state pixels (coast) is counted in the nearest state this close
STATE_DEFAULTS = {"target_px": TARGET_PX_DEFAULT, "rivers": True, "merge_px": MERGE_PX_DEFAULT,
                  "density_factor": 4, "density_dense_factor": 2, "density_ref": 50}
# Density (like the Provinces tab, user): a country's / state's size limits
# x clip(sqrt(density_ref / density), 1 / density_dense_factor,
# density_factor) x the user's country factor.
DENSITY_LIMITS = {"density_factor": (1, 20), "density_dense_factor": (1, 20), "density_ref": (1, 100_000)}
COUNTRY_FACTOR_LIMITS = (0.1, 10.0)

_lock = threading.Lock()
_levels = {}


def _dbf_columns(path: Path, names):
    """{field: list of str} for some .dbf character fields (numpy, memory-mapped)."""
    with open(path, "rb") as f:
        head = f.read(32)
        n, hlen, rlen = struct.unpack("<IHH", head[4:12])
        fields, off = {}, 1  # byte 0 of a record = deletion flag
        while True:
            d = f.read(32)
            if not d or d[0] == 0x0D:
                break
            name = d[:11].split(b"\0")[0].decode("ascii")
            length = d[16]
            fields[name] = (off, length)
            off += length
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=hlen, shape=(n, rlen))
    out = {}
    for name in names:
        if name not in fields:
            out[name] = [""] * n
            continue
        a, ln = fields[name]
        col = np.ascontiguousarray(raw[:, a:a + ln]).view(f"S{ln}").ravel()
        out[name] = [v.decode("utf-8", "replace").strip() for v in col]
    return out


def _shp_path(level: int):
    if core.SHAPEFILE_PATH is None:
        return None
    p = core.SHAPEFILE_PATH.parent / f"gadm36_{level}.shp"
    return p if p.exists() else None


def level_index(level: int):
    """{"gid0", "gid", "name" (arrays), "cx", "cy" (bbox centre lon/lat),
    "off" (record content byte offsets), "shp" (memmap), "by_tag" {GID_0:
    record indices}} for one GADM level, or None without its file."""
    with _lock:
        if level in _levels:
            return _levels[level]
    path = _shp_path(level)
    if path is None:
        idx = None
    else:
        cols = _dbf_columns(path.with_suffix(".dbf"), ["GID_0", f"GID_{level}" if level else "GID_0",
                                                       f"NAME_{level}" if level else "NAME_0", f"ENGTYPE_{level}"])
        gid0 = np.array(cols["GID_0"], dtype=object)
        gid = np.array(cols[f"GID_{level}" if level else "GID_0"], dtype=object)
        name = np.array(cols[f"NAME_{level}" if level else "NAME_0"], dtype=object)
        shx = np.fromfile(path.with_suffix(".shx"), dtype=">i4", offset=100).reshape(-1, 2)
        off = shx[:, 0].astype(np.int64) * 2 + 8
        shp = np.memmap(path, dtype=np.uint8, mode="r")
        box = shp[(off[:, None] + 4 + np.arange(32)[None, :]).ravel()].view("<f8").reshape(-1, 4)
        by_tag = {}
        for i, t in enumerate(gid0):
            by_tag.setdefault(t, []).append(i)
        etype = np.array(cols[f"ENGTYPE_{level}"], dtype=object)
        idx = {"gid0": gid0, "gid": gid, "name": name, "type": etype, "cx": (box[:, 0] + box[:, 2]) / 2,
               "cy": (box[:, 1] + box[:, 3]) / 2, "off": off, "shp": shp,
               "by_tag": {t: np.array(v) for t, v in by_tag.items()}}
    with _lock:
        _levels[level] = idx
    return idx


def _record_rings(idx, i: int):
    """[(False, lon/lat array)] of one record (even-odd fill: holes need no flag)."""
    shp, off = idx["shp"], int(idx["off"][i])
    if int(shp[off:off + 4].view("<i4")[0]) == 0:
        return []
    n_parts, n_pts = (int(v) for v in shp[off + 36:off + 44].view("<i4"))
    parts = np.asarray(shp[off + 44:off + 44 + 4 * n_parts].view("<i4")).astype(np.int64).tolist() + [n_pts]
    p0 = off + 44 + 4 * n_parts
    pts = np.asarray(shp[p0:p0 + 16 * n_pts].view("<f8")).reshape(n_pts, 2)
    return [(False, pts[a:b]) for a, b in zip(parts[:-1], parts[1:]) if b - a >= 3]


def normalize_settings(raw):
    s = dict(STATE_DEFAULTS)
    if raw.get("target_px") not in (None, ""):
        try:
            s["target_px"] = int(round(float(raw["target_px"])))
        except ValueError:
            raise core.RenderError("The target state size must be a number")
        if not 10 <= s["target_px"] <= 10_000_000:
            raise core.RenderError("The target state size must be between 10 and 10,000,000 px")
    if raw.get("merge_px") not in (None, ""):
        try:
            s["merge_px"] = int(round(float(raw["merge_px"])))
        except ValueError:
            raise core.RenderError("The small-state size must be a number")
        if not 0 <= s["merge_px"] <= 10_000_000:
            raise core.RenderError("The small-state size must be between 0 (no merging) and 10,000,000 px")
    if raw.get("rivers") not in (None, ""):
        s["rivers"] = str(raw["rivers"]).lower() in ("1", "true", "on", "yes")
    for k, (lo, hi) in DENSITY_LIMITS.items():
        v = raw.get(k)
        if v in (None, ""):
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise core.RenderError(f"State setting '{k}' must be a number")
        if not lo <= v <= hi:
            raise core.RenderError(f"State setting '{k}' must be between {lo:g} and {hi:g}")
        s[k] = v
    cf = {}
    raw_cf = raw.get("country_factors") or ""
    if isinstance(raw_cf, dict):
        raw_cf = ",".join(f"{k}:{v}" for k, v in raw_cf.items())
    for item in filter(None, raw_cf.split(",")):
        try:
            tag, v = item.split(":")
            v = float(v)
        except ValueError:
            raise core.RenderError(f"Bad country factor '{item}' (TAG:factor)")
        if not COUNTRY_FACTOR_LIMITS[0] <= v <= COUNTRY_FACTOR_LIMITS[1]:
            raise core.RenderError(f"{tag}: the country factor must be between {COUNTRY_FACTOR_LIMITS[0]:g} and "
                                   f"{COUNTRY_FACTOR_LIMITS[1]:g}")
        if abs(v - 1.0) > 1e-9:
            cf[tag.strip().upper()] = v
    s["country_factors"] = cf
    levels = {}
    raw_levels = raw.get("levels") or ""
    if isinstance(raw_levels, dict):  # already normalized
        raw_levels = ",".join(f"{k}:{v}" for k, v in raw_levels.items())
    for item in filter(None, raw_levels.split(",")):
        try:
            tag, lv = item.split(":")
            lv = int(lv)
        except ValueError:
            raise core.RenderError(f"Bad level override '{item}' (TAG:level)")
        if not 0 <= lv <= MAX_LEVEL:
            raise core.RenderError(f"{tag}: the GADM level must be between 0 and {MAX_LEVEL}")
        levels[tag.strip().upper()] = lv
    s["levels"] = levels
    return s


def _to_px(view, rings, func, with_area=False):
    """Projected pixel rings (seam copies included) of lon/lat rings, or [];
    with_area: (rings, full area in px -- all of the unit, in or out of
    the frame; shapefile holes wind the other way, so the signed sum nets
    them out)."""
    if not rings:
        return ([], 0.0) if with_area else []
    out, bbox, _rb, wraps, _wb = core._project_rings(rings, func, view["center"])
    sc, h = view["scale"], view["height"]
    if (bbox[2] - view["win_xmin"]) * sc < 0 or (bbox[0] - view["win_xmin"]) * sc > view["width"] \
            or h - (bbox[1] - view["win_ymin"]) * sc < 0 or h - (bbox[3] - view["win_ymin"]) * sc > h:
        return ([], 0.0) if with_area else []
    xy = [r for _h, r in out] + [r for _k, _h, r in wraps]
    px = [np.column_stack([(r[:, 0] - view["win_xmin"]) * sc, h - (r[:, 1] - view["win_ymin"]) * sc]) for r in xy]
    if not with_area:
        return px
    area = abs(sum(0.5 * float(np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1]))
                   for _h, r in out)) * sc * sc
    return px, area


def _state_color(base, seed: int, used: set, neighbours: list):
    """A colour near `base` (HSV jitter from `seed`), not in `used`, not the
    base itself, >= MIN_SIBLING_DIST from its already coloured neighbouring
    states of the same country when possible (else the farthest)."""
    rng = np.random.default_rng(seed)
    h0, s0, v0 = colorsys.rgb_to_hsv(*(c / 255.0 for c in base))
    n = 64
    for spread in (1.0, 1.5, 2.5, 4.0):
        dh, ds, dv = rng.uniform(-1, 1, (3, n)) * spread
        hh = (h0 + 0.012 * dh) % 1.0
        ss = np.clip(s0 + 0.10 * ds, 0.05, 1.0)
        vv = np.clip(v0 + 0.12 * dv, 0.12, 1.0)
        cand = [tuple(int(round(x * 255)) for x in colorsys.hsv_to_rgb(a, b, c)) for a, b, c in zip(hh, ss, vv)]
        ok = np.array([c not in used and c != tuple(base) and c not in (core.BACKGROUND_COLOR, core.LAKE_COLOR, core.EDGE_LAND_COLOR)
                       for c in cand])
        if not ok.any():
            continue
        if neighbours:
            d = np.sqrt(((np.array(cand, float)[:, None, :] - np.array(neighbours, float)[None]) ** 2).sum(2)).min(1)
        else:
            d = np.full(n, 1e9)
        d[~ok] = -1
        good = np.flatnonzero(d >= MIN_SIBLING_DIST)
        if len(good):
            return cand[good[0]]
        if spread == 4.0:
            return cand[int(d.argmax())]
    k = 0
    while True:  # last resort: any unused colour near base
        k += 1
        c = (base[0], base[1], (base[2] + k) % 256)
        if c not in used:
            return c


def _nearest_from_outside(state, gid_img, todo, river):
    """`state` with every `todo` pixel re-assigned the state of the nearest
    (4-connected path length) pixel outside `todo`, the path staying in one
    country and never crossing a river pixel -- one multi-source Dijkstra
    over the todo pixels and the anchors next to them. Unreached pixels keep
    their state; river pixels are left as they are (set by the caller)."""
    h, w = state.shape
    node_ok = (gid_img >= 0) & ~river
    near = ndimage.binary_dilation(todo, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
    nodes = near & node_ok
    anchor = nodes & ~todo
    ids = np.full((h, w), -1, dtype=np.int64)
    flat = np.flatnonzero(nodes)
    ids.ravel()[flat] = np.arange(len(flat))
    rows, cols = [], []
    for dy, dx in ((0, 1), (1, 0)):
        a = ids[: h - dy, : w - dx]
        b = ids[dy:, dx:]
        m = (a >= 0) & (b >= 0) & (gid_img[: h - dy, : w - dx] == gid_img[dy:, dx:])
        m &= todo[: h - dy, : w - dx] | todo[dy:, dx:]  # anchor-anchor edges aren't needed
        rows.append(a[m]); cols.append(b[m])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    out = state.copy()
    src = np.flatnonzero(anchor.ravel()[flat])
    if not len(src) or not len(rows):
        return out
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(flat), len(flat))).tocsr()
    _d, _p, sources = dijkstra(g, directed=False, indices=src, min_only=True, return_predecessors=True)
    todo_nodes = todo.ravel()[flat]
    reached = todo_nodes & (sources >= 0)
    pix = flat[reached]
    out.ravel()[pix] = state.ravel()[flat[sources[reached]]]
    return out


def _gid_path(st) -> tuple:
    """(country, level-1 key, ..., own key) of a GADM subdivision: GID
    "FRA.1.2_1" -> ("FRA", "FRA.1", "FRA.1.2")."""
    if st["level"] == 0:
        return (st["tag"],)
    parts = str(st["gid"]).split(".")
    parts[-1] = parts[-1].split("_")[0]
    return (st["tag"],) + tuple(".".join(parts[:k + 1]) for k in range(1, len(parts)))


def _common_prefix(a: tuple, b: tuple) -> tuple:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return a[:n]


def _state_populations(state, n_labels: int, view):
    """Population per state label: GeoNames places (>= 1,000 people, each
    place's own figure -- a city's districts aren't counted twice) inside
    it; a place just off the state pixels (coast) goes to the nearest state
    within POP_SNAP_PX. None without the city dataset. Rural people outside
    such places aren't counted."""
    try:
        import cities
        if not cities.available():
            return None
        idx = cities.load_city_index()
    except Exception:
        return None
    h, w = state.shape
    func = core.PROJECTIONS[view["projection"]]["func"]
    x, y = func(core._relative_lon(idx["lon"], view["center"]), idx["lat"])
    ix = np.floor((x - view["win_xmin"]) * view["scale"]).astype(np.int64)
    iy = np.floor(h - (y - view["win_ymin"]) * view["scale"]).astype(np.int64)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix, iy, pop = ix[ok], iy[ok], idx["own_population"][ok].astype(np.float64)
    lab = state[iy, ix]
    off = lab == 0
    if off.any() and (state > 0).any():
        dist, (ny, nx) = ndimage.distance_transform_edt(state == 0, return_indices=True)
        near = off & (dist[iy, ix] <= POP_SNAP_PX)
        lab[near] = state[ny[iy[near], ix[near]], nx[iy[near], ix[near]]]
    return np.bincount(lab, weights=pop, minlength=n_labels)


def _populations(state, states, counts, view):
    """(population per label or None, report). GHS-POP per GADM unit when
    downloaded (population.py: 4.1 table matched to 3.6, gaps split from
    the parent) -- a unit cut by the frame gets its visible share
    (pixels / full projected area, when under CLIP_SHARE) -- else, and for
    units it can't place, the GeoNames places inside the state."""
    import population
    geo = _state_populations(state, len(states), view)
    rep = {"source": None, "table": 0, "split": 0, "places": 0, "clipped": 0}
    if not population.available():
        rep["source"] = "GeoNames places >= 1,000 people (download GHS-POP for full figures)" if geo is not None else None
        return geo, rep
    # Units reaching the frame (or the sea band along it: the fully water
    # lines at each edge) -- only those are scaled; a unit short of pixels
    # elsewhere (erased islets, lakes) keeps its full population.
    land = state > 0
    h, w = state.shape
    at_edge = np.zeros(len(states), dtype=bool)
    edge_zone = np.zeros((h, w), dtype=bool)
    for axis_rev, sl in ((False, 0), (True, 0), (False, 1), (True, 1)):
        m = land if sl == 0 else land.T
        m = m[::-1] if axis_rev else m
        rows = np.flatnonzero(m.any(axis=1))
        k = (int(rows[0]) if len(rows) else 0) + 3
        z = np.zeros(m.shape, dtype=bool)
        z[:k] = True
        z = z[::-1] if axis_rev else z
        edge_zone |= z if sl == 0 else z.T
    at_edge[np.unique(state[edge_zone & land])] = True
    rep["frame_units"] = int(at_edge[1:].sum())
    pops = np.zeros(len(states))
    for k in range(1, len(states)):
        st = states[k]
        if counts[k] == 0:
            continue
        p, src = population.population_of(st["level"], st["tag"], st["gid"])
        if p is None:
            pops[k] = geo[k] if geo is not None else 0.0
            rep["places"] += 1
            continue
        full = st.get("full_px") or 0.0
        if at_edge[k] and full > 0 and counts[k] < CLIP_SHARE * full:
            p *= counts[k] / full
            rep["clipped"] += 1
        pops[k] = p
        rep[src if src in ("table", "split") else "places"] += 1
    rep["source"] = population.SOURCE
    return pops, rep


def density_factor(mean_density, s):
    """sqrt(ref / density) clipped to [1 / dense factor, sparse factor] (1 without density)."""
    if mean_density is None:
        return 1.0
    return float(np.clip(np.sqrt(s["density_ref"] / max(mean_density, 1e-3)),
                         1.0 / max(s["density_dense_factor"], 1.0), s["density_factor"]))


def _merge_small_states(state, states, counts, pops, threshold, target=None):
    """Merge every state under `threshold` px into a neighbour (user's
    rules): smallest first; only with a neighbour in the same enclosing
    GADM subdivision -- the deepest one that holds a neighbour (a district
    merges within its oblast; if the oblast has no other district beside
    it, within the country), never across countries; among those, a
    neighbour also under the threshold that the merge lifts above it
    first, then the longest shared border (4-connected pixel contacts).
    The merged state keeps the more populated one's label (name, GADM id,
    colour seed; by area without population data); pixels and population are added; still small -> back in the
    queue. Returns (label map old -> final, merges done)."""
    import heapq
    n = len(states)
    final = np.arange(n)
    thr = (np.full(n, float(threshold)) if np.ndim(threshold) == 0
           else np.asarray(threshold, dtype=np.float64).copy())  # per state (density / country scaled)
    if not (thr > 0).any() or n <= 2:
        return final, 0, 0
    h, w = state.shape
    border = {}
    for dy, dx in ((0, 1), (1, 0)):
        a = state[: h - dy, : w - dx]
        b = state[dy:, dx:]
        m = (a != b) & (a > 0) & (b > 0)
        lo, hi = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])
        pairs, cnt = np.unique(np.stack([lo, hi], 1), axis=0, return_counts=True)
        for (x, y), c in zip(pairs.tolist(), cnt.tolist()):
            border.setdefault(x, {})
            border.setdefault(y, {})
            border[x][y] = border[x].get(y, 0) + c
            border[y][x] = border[y].get(x, 0) + c
    size = counts.astype(np.int64).copy()
    pop = pops.copy() if pops is not None else np.zeros(n)
    chain = [None] + [_gid_path(st)[:-1] for st in states[1:]]  # enclosing subdivisions
    members = [[k] for k in range(n)]
    rank = (lambda x: (pop[x], size[x], -x)) if pops is not None else (lambda x: (size[x], -x))

    def join(a, b):
        """Merge a and b; returns the survivor."""
        # the label of the more populated one (user: Moscow was named after a
        # town beside it, Berlin after Barnim -- by area); no population: area
        keep, gone = (a, b) if rank(a) > rank(b) else (b, a)
        # merge `gone` into `keep`
        # merged threshold = the size-weighted mean (the max snowballed: one
        # sparse municipality's 3-8k px limit made North Macedonia one state)
        thr[keep] = (thr[keep] * size[keep] + thr[gone] * size[gone]) / max(size[keep] + size[gone], 1)
        size[keep] += size[gone]
        pop[keep] += pop[gone]
        size[gone] = 0
        members[keep] += members[gone]
        chain[keep] = _common_prefix(chain[keep], chain[gone])
        for x, c in border.pop(gone, {}).items():
            if x == keep:
                continue
            border[keep][x] = border[keep].get(x, 0) + c
            border[x][keep] = border[x].get(keep, 0) + c
            border[x].pop(gone, None)
        border[keep].pop(gone, None)
        final[gone] = keep
        return keep
    heap = [(int(size[k]), k) for k in range(1, n) if 0 < size[k] < thr[k]]
    heapq.heapify(heap)
    merges = whole = 0
    while heap:
        sz, a = heapq.heappop(heap)
        if final[a] != a or sz != size[a] or sz >= thr[a]:
            continue
        nbrs = [x for x in border.get(a, {}) if final[x] == x and size[x] > 0]
        depth = {x: len(_common_prefix(chain[a], chain[x])) for x in nbrs}
        nbrs = [x for x in nbrs if depth[x] >= 1 and states[x]["tag"] == states[a]["tag"]]
        if not nbrs:
            continue  # an island, or alone in its country: stays small
        deepest = max(depth[x] for x in nbrs)
        allowed = [x for x in nbrs if depth[x] == deepest]
        pair_up = [x for x in allowed if size[x] < thr[x] and size[a] + size[x] >= thr[a]]
        # else a neighbour still under its threshold first: with only
        # "longest border", the first merged blob had the longest border with
        # everything and swallowed them all (North Macedonia: 85
        # municipalities of ~100 px -> one 8,300 px state)
        still_small = [x for x in allowed if size[x] < thr[x]]
        pool = pair_up or still_small or allowed
        b = max(pool, key=lambda x: (border[a][x], -size[x], -x))
        keep = join(a, b)
        merges += 1
        if size[keep] < thr[keep]:
            heapq.heappush(heap, (int(size[keep]), keep))
    # A GADM unit whose states together fit under the target is one state
    # (user: Seine-Maritime, 2,695 px vs a 3,400 px target, came out as two
    # states -- the merge above stops once each piece passes the merge size)
    if target is not None:
        tgt0 = np.asarray(target, dtype=np.float64)
        cnt0 = np.asarray(counts, dtype=np.float64)

        def tgt_of(k):  # the members' targets, weighted by their pixels
            w_ = sum(cnt0[m] for m in members[k])
            return sum(tgt0[m] * cnt0[m] for m in members[k]) / max(w_, 1.0)
        for _round in range(4):
            alive = [k for k in range(1, n) if final[k] == k and size[k] > 0]
            parents = sorted({chain[k][:d] for k in alive for d in range(2, len(chain[k]) + 1)},
                             key=lambda c: -len(c))
            done = False
            for par in parents:
                grp = [k for k in range(1, n) if final[k] == k and size[k] > 0 and chain[k][:len(par)] == par]
                if len(grp) < 2 or len({states[k]["tag"] for k in grp}) > 1:
                    continue
                total = float(sum(size[k] for k in grp))
                if total > sum(tgt_of(k) * size[k] for k in grp) / total:
                    continue
                left = set(grp)  # merged along shared borders (a unit in pieces stays in pieces)
                cur = max(grp, key=rank)
                left.discard(cur)
                grew = True
                while left and grew:
                    grew = False
                    for x in sorted(left, key=lambda x: -border.get(cur, {}).get(x, 0)):
                        if border.get(cur, {}).get(x, 0) > 0:
                            left.discard(x)
                            cur = join(cur, x)
                            merges += 1
                            whole += 1
                            grew = done = True
                            break
            if not done:
                break
    # path compression: every label -> its surviving state
    for k in range(n):
        r = k
        while final[r] != r:
            r = final[r]
        final[k] = r
    for k in range(1, n):
        if final[k] == k and len(members[k]) > 1:
            states[k]["merged"] = [states[m]["gid"] for m in members[k]]
    if pops is not None:
        pops[:] = pop
    return final, merges, whole


def _water_unit(st):
    """A GADM unit that is a water body (NLD's IJsselmeer, Zeeuwse meren)."""
    lv = st.get("level") or 0
    if lv < 1:
        return False
    idx = level_index(lv)
    if idx is None:
        return False
    if "_gid_pos" not in idx:
        idx["_gid_pos"] = {g: i for i, g in enumerate(idx["gid"].tolist())}
    i = idx["_gid_pos"].get(st["gid"])
    return i is not None and "water" in str(idx["type"][i]).lower()


def choose_levels(countries: Image.Image, debug, settings=None, density_img=None):
    """The level choice alone (no rasterizing): {"countries": per-country
    report with the candidate levels, "chosen": {tag: level}, plus the
    colour-group structures render_states goes on with}. With a density map
    (people / km2 per pixel), each country's target and merge size are
    scaled by its density factor and the user's country factor."""
    s = normalize_settings(settings or {})
    view = debug["view"]
    h, w = view["height"], view["width"]
    func = core.PROJECTIONS[view["projection"]]["func"]
    arr = np.asarray(countries.convert("RGB"))
    code = (arr[..., 0].astype(np.int64) << 16) | (arr[..., 1].astype(np.int64) << 8) | arr[..., 2]
    # Colour -> tags (merged autonomous territories share their parent's colour).
    groups = {}
    for t, rgb in debug["country_rgb"].items():
        if t in core.WATER_TAGS:
            continue
        groups.setdefault((rgb[0] << 16) | (rgb[1] << 8) | rgb[2], []).append(t)
    present = np.unique(code)
    gcodes = np.array([c for c in present.tolist() if c in groups], dtype=np.int64)
    gid_img = np.full((h, w), -1, dtype=np.int32)
    if len(gcodes):
        pos = np.searchsorted(gcodes, code)
        pos = np.clip(pos, 0, len(gcodes) - 1)
        hit = gcodes[pos] == code
        gid_img[hit] = pos[hit]
    group_tags = [groups[c] for c in gcodes.tolist()]
    group_px = np.bincount(gid_img[gid_img >= 0], minlength=len(gcodes))
    geoms = core.load_geometries()

    # Pixels of each tag inside its colour group (only groups of several tags
    # need their members' level-0 outlines).
    tag_px = {}
    for g, tags in enumerate(group_tags):
        if len(tags) == 1:
            tag_px[tags[0]] = int(group_px[g])
            continue
        # the parent: the member that isn't a merged autonomous territory
        import palettes
        owner = next((t for t in tags if t not in palettes.AUTONOMOUS_PARENT), tags[0])
        rest = int(group_px[g])
        for t in tags:
            if t == owner or t not in geoms:
                continue
            m = np.zeros((h, w), dtype=bool)
            rp = _to_px(view, geoms[t]["rings"], func)
            if rp:
                core._scanline_fill(m, rp, True)
            n = int((m & (gid_img == g)).sum())
            tag_px[t] = n
            rest -= n
        tag_px[owner] = max(rest, 0)
    tag_px = {t: n for t, n in tag_px.items() if n > 0}

    # Density factor per colour group (mean people / km2 over its pixels).
    group_dens = None
    if density_img is not None and len(gcodes):
        m_ = gid_img >= 0
        tot = np.bincount(gid_img[m_], weights=density_img[m_].astype(np.float64), minlength=len(gcodes))
        group_dens = tot / np.maximum(group_px, 1)
    tag_group_ = {t: g for g, tags in enumerate(group_tags) for t in tags}
    # Level per tag: subdivisions whose bbox centre is in the frame.
    report_countries = []
    chosen = {}
    no_merge = set()
    for t, px in sorted(tag_px.items()):
        dens_t = None if group_dens is None else float(group_dens[tag_group_[t]])
        f_t = density_factor(dens_t, s) * s["country_factors"].get(t, 1.0)
        cands = [{"level": 0, "n": 1, "mean_px": px, "type": "Country"}]
        for lv in range(1, MAX_LEVEL + 1):
            idx = level_index(lv)
            if idx is None or t not in idx["by_tag"]:
                continue
            recs = idx["by_tag"][t]
            x, y = func(core._relative_lon(idx["cx"][recs], view["center"]), idx["cy"][recs])
            cx = (x - view["win_xmin"]) * view["scale"]
            cy = h - (y - view["win_ymin"]) * view["scale"]
            n = int(((cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)).sum())
            n = max(n, 1)
            if n == cands[-1]["n"] and lv > 1:
                continue  # same subdivisions as the level above (GADM repeats some)
            types, tc = np.unique(idx["type"][recs], return_counts=True)
            cands.append({"level": lv, "n": n, "mean_px": round(px / n), "type": str(types[tc.argmax()])})
        # User: rather aim lower and merge than aim bigger; at least level 1
        # unless the country itself is under the merge threshold.
        small = (s["merge_px"] if s["merge_px"] > 0 else s["target_px"]) * f_t
        subs = [c for c in cands if c["level"] >= 1]
        # User: a level whose units are big enough on average to need no
        # merging (mean >= the merge size, and not over WHOLE_UNIT_FACTOR x the
        # target) is used as it is, the coarsest such -- France's departements;
        # else the old way, merged afterwards (Slovenia's small L1 regions)
        # (the merge size as set, not density-scaled: China's whole-country
        # density factor let its ~850 px counties pass -- 2,377 states)
        merge_t = s["merge_px"]
        top = WHOLE_UNIT_FACTOR * max(s["target_px"] * f_t, s["target_px"])
        as_is = [c for c in subs if s["merge_px"] > 0 and merge_t <= c["mean_px"] <= top]
        if px < small or not subs:
            best = cands[0]
        elif as_is:  # the one closest to the target (Belgium: provinces, not its 3 regions)
            best = min(as_is, key=lambda c: abs(math.log(c["mean_px"] / (s["target_px"] * f_t))))
        else:
            under = [c for c in subs if c["mean_px"] <= s["target_px"] * f_t]
            best = max(under, key=lambda c: c["mean_px"]) if under else min(subs, key=lambda c: c["mean_px"])
        lv = s["levels"].get(t, best["level"])
        if all(c["level"] != lv for c in cands):
            lv = best["level"]
        chosen[t] = lv
        lv_mean = next(c["mean_px"] for c in cands if c["level"] == lv)
        if lv >= 1 and s["merge_px"] > 0 and lv_mean >= merge_t:
            no_merge.add(t)  # the chosen level needs no merging: none for this country
        report_countries.append({"tag": t, "name": geoms[t]["name"] if t in geoms else t, "px": px,
                                 "auto_level": best["level"], "level": lv,
                                 "override": t in s["levels"] and s["levels"][t] == lv, "candidates": cands,
                                 "factor": s["country_factors"].get(t, 1.0),
                                 "density": None if dens_t is None else round(dens_t, 1), "size_factor": round(f_t, 2),
                                 "merge": t not in no_merge})

    return {"countries": report_countries, "chosen": chosen, "no_merge": no_merge, "gid_img": gid_img, "gcodes": gcodes,
            "group_tags": group_tags, "arr": arr, "geoms": geoms}


def _fix_state_exclaves(state, orig, label_group, max_rounds=5, raw=None):
    """Every state one piece (user: minimise exclaves / enclaves): a piece
    of a state other than its largest goes to the neighbouring state of the
    same country it borders most -- unless the rasterized GADM units
    (`orig`, same labels) already had it apart from the main piece (a real
    exclave, an island) or no state of its country borders it. With `raw`
    (the GADM unit of each pixel, before merging) a piece is a real exclave
    only if one of its units also lies in the main piece: a piece made of
    whole other units is a merge gone wrong (user: state 1459, Slantsy joined
    to a block 250 px away through a raster sliver) and is moved too. In
    place; returns pixels moved."""
    h, w = state.shape
    moved = 0
    for _ in range(max_rounds):
        changed = 0
        for k, sl in enumerate(ndimage.find_objects(state), start=1):
            if sl is None:
                continue
            sl = tuple(slice(max(a.start - 1, 0), a.stop + 1) for a in sl)
            m = state[sl] == k
            comp, n = ndimage.label(m, structure=CROSS4)
            if n < 2:
                continue
            sizes = np.bincount(comp.ravel())[1:]
            main = int(np.argmax(sizes)) + 1
            oc, _on = ndimage.label(orig[sl] == k, structure=CROSS4)
            osz = np.bincount(oc.ravel())
            main_orig = set(np.unique(oc[(comp == main) & (oc > 0)]).tolist())
            for c in range(1, n + 1):
                if c == main:
                    continue
                piece = comp == c
                own_orig = set(np.unique(oc[piece & (oc > 0)]).tolist())
                # apart in GADM too, at a real size: a real exclave / island
                # (rasterizing leaves 1-2 px slivers apart along borders)
                if any(osz[o] >= EXCLAVE_MIN_PX for o in own_orig - main_orig):
                    if raw is None:
                        continue
                    rs = raw[sl]
                    pu, pc = np.unique(rs[piece], return_counts=True)
                    mu, mc = np.unique(rs[comp == main], return_counts=True)
                    in_main = dict(zip(mu.tolist(), mc.tolist()))
                    au, ac = np.unique(rs[m], return_counts=True)
                    in_state = dict(zip(au.tolist(), ac.tolist()))
                    # most of the piece is units whose body (most of their pixels
                    # in this state) is in the main piece: their exclave. Not: raster
                    # slivers of a unit whose body is this piece (bad merge), shreds
                    # of a unit cut up by a lake (Onega's shore: 2,000 px in 60 bits,
                    # 820 in main), nor another unit's piece with a corner of one
                    # (Cherepovets: 1,961 px of Belozersk + 191 of a split rayon)
                    ex = sum(c_ for u, c_ in zip(pu.tolist(), pc.tolist())
                             if u and in_main.get(u, 0) >= EXCLAVE_MIN_PX and 2 * in_main.get(u, 0) >= in_state[u])
                    if 2 * ex >= int(piece.sum()):
                        continue  # a unit split in GADM itself: a real exclave / island
                ring = ndimage.binary_dilation(piece, structure=CROSS4) & ~piece
                nb = state[sl][ring]
                nb = [v for v in nb.tolist() if v > 0 and v != k and label_group[v] == label_group[k]]
                if not nb:
                    continue
                state[sl][piece] = max(set(nb), key=nb.count)
                changed += int(piece.sum())
        moved += changed
        if not changed:
            break
    return moved


def render_states(countries: Image.Image, debug, rivers: Image.Image | None = None, settings=None,
                  density_img=None):
    """(state map RGB image, report). `countries` = the country map to follow
    (the river-fitted one with rivers), `rivers` = the rivers.bmp image (P,
    indices 0-11 = river) or None."""
    s = normalize_settings(settings or {})
    t0 = time.time()
    lv_info = choose_levels(countries, debug, s, density_img)
    view = debug["view"]
    h, w = view["height"], view["width"]
    func = core.PROJECTIONS[view["projection"]]["func"]
    report_countries, chosen = lv_info["countries"], lv_info["chosen"]
    gid_img, gcodes, group_tags = lv_info["gid_img"], lv_info["gcodes"], lv_info["group_tags"]
    arr, geoms = lv_info["arr"], lv_info["geoms"]
    report_t = {"levels": round(time.time() - t0, 2)}
    # Rasterize the chosen subdivisions: label image (0 = none).
    lab = np.zeros((h, w), dtype=np.int32)
    states = [None]  # label -> {"gid", "name", "tag", "level"}
    tag_group = {t: g for g, tags in enumerate(group_tags) for t in tags}
    for t, lv in chosen.items():
        if lv == 0:
            rp, full = _to_px(view, geoms[t]["rings"], func, with_area=True)
            if rp:
                states.append({"gid": t, "name": geoms[t]["name"], "tag": t, "level": 0, "full_px": full})
                core._scanline_fill(lab, rp, len(states) - 1)
            continue
        idx = level_index(lv)
        for i in idx["by_tag"][t].tolist():
            rp, full = _to_px(view, _record_rings(idx, i), func, with_area=True)
            if not rp:
                continue
            states.append({"gid": idx["gid"][i], "name": idx["name"][i], "tag": t, "level": lv, "full_px": full})
            core._scanline_fill(lab, rp, len(states) - 1)
    label_group = np.array([-1] + [tag_group.get(st["tag"], -1) for st in states[1:]], dtype=np.int32)

    report_t["raster"] = round(time.time() - t0, 2)
    # Rule 1: every pixel of a country gets one of its own subdivisions.
    state = np.zeros((h, w), dtype=np.int32)
    valid = (gid_img >= 0) & (label_group[lab] == gid_img)
    state[valid] = lab[valid]
    fixed = 0
    boxes = ndimage.find_objects(gid_img + 1)
    for g, sl in enumerate(boxes):
        if sl is None:
            continue
        mg = gid_img[sl] == g
        bg = mg & ~valid[sl]
        if not bg.any():
            continue
        (y0, y1), (x0, x1) = (sl[0].start, sl[0].stop), (sl[1].start, sl[1].stop)
        vg = valid[sl] & mg
        if not vg.any():  # nothing drawn for it: the whole country is one state
            t = group_tags[g][0]
            states.append({"gid": t, "name": geoms[t]["name"] if t in geoms else t, "tag": t, "level": 0})
            state[y0:y1, x0:x1][bg] = len(states) - 1
            fixed += int(bg.sum())
            continue
        _d, (iy, ix) = ndimage.distance_transform_edt(~vg, return_indices=True)
        sub = state[y0:y1, x0:x1]
        sub[bg] = sub[iy[bg], ix[bg]]
        fixed += int(bg.sum())
    report = {"settings": {k: v for k, v in s.items()}, "reassigned_px": fixed}

    report_t["rule1"] = round(time.time() - t0, 2)
    # Rule 2: state borders along rivers follow them.
    river_changed = 0
    if s["rivers"] and rivers is not None:
        river = np.asarray(rivers) <= 11
        river &= gid_img >= 0
        if river.any():
            band = (ndimage.distance_transform_edt(~river) <= RIVER_SNAP_PX) & ~river & (gid_img >= 0)
            # states with no pixel outside the band keep theirs
            outside = np.bincount(state[(gid_img >= 0) & ~band & ~river].ravel(), minlength=len(states))
            keep = band & (outside[state] == 0)
            todo = band & ~keep
            before = state.copy()
            new = _nearest_from_outside(state, gid_img, todo, river)
            # River pixels: their own state if it is beside them, else the most common beside them.
            ry, rx = np.nonzero(river)
            nb = []
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                yy, xx = np.clip(ry + dy, 0, h - 1), np.clip(rx + dx, 0, w - 1)
                v = np.where((gid_img[yy, xx] == gid_img[ry, rx]) & ~river[yy, xx], new[yy, xx], 0)
                nb.append(v)
            nb = np.stack(nb, axis=1)
            own = state[ry, rx]
            votes = ((nb[:, :, None] == nb[:, None, :]) & (nb[:, None, :] > 0)).sum(axis=2) * (nb > 0)
            mode = nb[np.arange(len(ry)), votes.argmax(axis=1)]
            has_own = (nb == own[:, None]).any(axis=1)
            new[ry, rx] = np.where(has_own | (mode == 0), own, mode)
            state = new
            river_changed = int((state != before).sum())
            report["river_band_px"] = int(band.sum())
    report["river_moved_px"] = river_changed

    report_t["rule2"] = round(time.time() - t0, 2)

    # Small states merged into a neighbour.
    counts0 = np.bincount(state.ravel(), minlength=len(states))
    pops, report["population"] = _populations(state, states, counts0, view)
    report["states_before_merge"] = int((counts0[1:] > 0).sum())
    # per-state merge size: x its density factor x its country's factor
    thr = np.full(len(states), float(s["merge_px"]))
    if density_img is not None or s["country_factors"]:
        if density_img is not None:
            dsum = np.bincount(state.ravel(), weights=density_img.ravel().astype(np.float64), minlength=len(states))
            dmean = dsum / np.maximum(counts0, 1)
        for k in range(1, len(states)):
            f_ = density_factor(float(dmean[k]) if density_img is not None else None, s)
            thr[k] *= f_ * s["country_factors"].get(states[k]["tag"], 1.0)
    # a GADM unit whose states fit under the target together is one state
    # (density never lowers that limit below WHOLE_UNIT_FACTOR x the target --
    # user: Seine-Maritime and 14 other departements, 2,300-3,700 px, were
    # split because dense areas aim at half-size states)
    tgt = (np.maximum(thr * (s["target_px"] / s["merge_px"]), WHOLE_UNIT_FACTOR * s["target_px"])
           if s["merge_px"] else None)
    no_merge = lv_info.get("no_merge", set())
    if no_merge:  # countries whose level is big enough on average: no merging at all
        # (GADM water-body units -- the IJsselmeer -- are never states of their own: still merged)
        nm = np.array([False] + [st["tag"] in no_merge and not _water_unit(st) for st in states[1:]])
        thr[nm] = 0.0
        if tgt is not None:
            tgt[nm] = 0.0
    report["countries_unmerged"] = sorted(no_merge)
    state0 = state.copy()
    unit_pops = pops.copy() if pops is not None else None
    final, merges, report["whole_unit_merges"] = _merge_small_states(state, states, counts0, pops, thr, tgt)
    if merges:
        state = final[state].astype(np.int32)
    report["merges"] = merges
    # stray pieces of states (not GADM exclaves) join their neighbour
    orig = final[lab].astype(np.int32)
    lg = np.array([-1] + [tag_group.get(st["tag"], -1) for st in states[1:]], dtype=np.int32)
    report["exclave_px_moved"] = _fix_state_exclaves(state, orig, lg, raw=lab)
    if unit_pops is not None and report["exclave_px_moved"]:
        # pieces moved between states: each unit's population follows its pixels
        n_ = len(states)
        keys, cnt = np.unique(state0.ravel().astype(np.int64) * n_ + state.ravel(), return_counts=True)
        u_, k_ = np.divmod(keys, n_)
        upx = np.bincount(u_, weights=cnt, minlength=n_)
        pops[:] = np.bincount(k_, weights=unit_pops[u_] * cnt / np.maximum(upx[u_], 1), minlength=n_)
    report["population_source"] = report["population"]["source"]
    report_t["merge"] = round(time.time() - t0, 2)
    report["timings"] = report_t
    # Colours.
    counts = np.bincount(state.ravel(), minlength=len(states))
    # neighbouring states (side contact)
    adj = {}
    for dy, dx in ((0, 1), (1, 0)):
        a = state[: h - dy, : w - dx]
        b = state[dy:, dx:]
        m = (a != b) & (a > 0) & (b > 0)
        for x, y in np.unique(np.stack([a[m], b[m]], 1), axis=0).tolist():
            adj.setdefault(x, set()).add(y)
            adj.setdefault(y, set()).add(x)
    used = set()
    colors = [None] * len(states)
    order = sorted(range(1, len(states)), key=lambda k: (states[k]["tag"], str(states[k]["gid"])))
    for k in order:
        if counts[k] == 0:
            continue
        st = states[k]
        base = debug["country_rgb"][st["tag"]]
        nbs = [colors[j] for j in adj.get(k, ()) if colors[j] is not None]
        c = _state_color(base, zlib.crc32(str(st["gid"]).encode()), used, nbs)
        used.add(c)
        colors[k] = c
    lut = np.zeros((len(states), 3), dtype=np.uint8)
    for k, c in enumerate(colors):
        if c is not None:
            lut[k] = c
    out = arr.copy()
    m = state > 0
    out[m] = lut[state[m]]

    # Report.
    ids = [k for k in range(1, len(states)) if counts[k] > 0]
    com = ndimage.center_of_mass(np.ones((h, w)), state, ids) if ids else []
    report["states"] = [{"id": i + 1, "gid": states[k]["gid"], "name": states[k]["name"], "tag": states[k]["tag"],
                         "level": states[k]["level"], "px": int(counts[k]), "color": core.rgb_to_hex(colors[k]),
                         "x": int(round(c_[1])), "y": int(round(c_[0])),
                         "population": None if pops is None else int(round(pops[k])),
                         "merged_from": "|".join(map(str, states[k].get("merged", [])))}
                        for i, (k, c_) in enumerate(zip(ids, com))]
    report["small_states_left"] = int(sum(1 for st in report["states"] if st["px"] < s["merge_px"]))
    per_tag = {}
    for st in report["states"]:
        per_tag[st["tag"]] = per_tag.get(st["tag"], 0) + 1
    per_pop = {}
    for st in report["states"]:
        per_pop[st["tag"]] = per_pop.get(st["tag"], 0) + (st["population"] or 0)
    for c in report_countries:
        c["population"] = per_pop.get(c["tag"], 0) if pops is not None else None
        c["states"] = per_tag.get(c["tag"], 0)
        c["color"] = core.rgb_to_hex(debug["country_rgb"][c["tag"]])
        c["mean_state_px"] = round(c["px"] / c["states"]) if c["states"] else None
    report["countries"] = sorted(report_countries, key=lambda c: c["name"])
    sizes = np.array([st["px"] for st in report["states"]])
    report["count"] = len(sizes)
    report["mean_px"] = round(float(sizes.mean())) if len(sizes) else None
    report["median_px"] = round(float(np.median(sizes))) if len(sizes) else None
    report["seconds"] = round(time.time() - t0, 2)
    # Per-pixel state ids (= report["states"][i]["id"]; 0 = none) for the
    # Provinces tab, which cuts each state into provinces.
    remap = np.zeros(len(states), dtype=np.int32)
    remap[ids] = np.arange(1, len(ids) + 1)
    report["_labels"] = remap[state]
    return Image.fromarray(out, mode="RGB"), report
