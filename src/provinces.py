"""Provinces tab: HOI4 provinces.bmp + definition.csv.

Land provinces are cut out of the States tab's states (HOI4: every province
belongs to exactly one state), with the states' rules:
  * GADM subdivisions as the base (states.choose_levels with the land
    maximum as the target and the land minimum as the merge size: the
    coarsest level whose subdivisions average at most `land_max`, at least
    level 1 unless the country itself is under `land_min`; never coarser
    than the country's state level); a pixel whose subdivision isn't one of
    its state's country takes the nearest one in the same state;
  * borders within states.RIVER_SNAP_PX of a river follow it (the states'
    river rule, inside each state);
  * provinces over `land_max` are split into compact pieces (k-means on
    their pixels), provinces under `land_min` merged into a neighbour of
    the same state -- smallest first, a neighbour also too small that the
    merge lifts above the minimum first, else the longest shared border.
Sea: each sea area (4-connected) cut into compact cells between `sea_min`
and `sea_max`; each lake = one lake province.
Topology (user), repeated until stable:
  * every province is one 4-connected piece (stray parts of a few pixels
    join a neighbour of the same type, larger ones become provinces);
  * no X-crossings: no 2x2 block with a diagonal pair of one province and
    two other provinces across it, nor four different provinces -- fixed
    by moving one pixel (same land/sea type and a locally connected donor
    first);
  * two provinces touch along one stretch only: when they touch in several
    places (typically around a lake on their border, or a small province
    enclosed between them), the larger of the two is split so that each
    piece holds one of the contacts (geodesic nearest contact, inside the
    province).
definition.csv: id;R;G;B;land|sea|lake;coastal;terrain;continent, like the
game's; terrain = each land province's most common Terrain-tab type,
continent = vanilla numbering (1 europe, 2 north_america, 3 south_america,
4 australia, 5 africa, 6 asia, 7 middle_east) from GeoNames' country
continents; coastal = a land province touching the sea / a sea province
touching land.
"""
import functools
import heapq
import math
import time

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import core
import states

PROVINCE_DEFAULTS = {"land_min": 100, "land_max": 900, "sea_min": 1200, "sea_max": 3600,
                     "density_factor": 4,     # sparse land: provinces up to this many times bigger (1 = off)
                     "density_dense_factor": 2,  # dense land: provinces down to 1 / this (1 = off)
                     "density_ref": 50,       # people / km2 at and above which provinces keep their normal size
                     "sea_far_factor": 3,     # sea provinces grow up to this many times bigger far from the coast
                     "river_split_px": 200,   # a land province this big that a river cuts in two is split (0 = off)
                     "target_count": 0,       # total provinces aimed at: land sizes scaled to reach it (0 = off)
                     "city_provinces": 0,     # 1 = the City tab's cities get provinces following the urban terrain
                     "city_classes": 3,       # ... cities of classes 1..this
                     "city_min_px": 20}       # ... grown to at least this (within their state)
# The game's province border limit (hardcoded in hoi4.exe): it draws at most
# 65,535 border segments -- one per pair of neighbouring provinces --, and
# past that province borders stop being drawn over large parts of the map
# (reported by the user on a 35,188-province map: 102,999 pairs). Maps made
# here have ~2.93 pairs per province, so the ceiling is ~22,000 provinces,
# whatever the map's size: the limit is a count, not a resolution. Over it,
# the smallest land pairs are merged until under (`_cap_borders`, the
# target-count trim's rules), and the target number can't be set above it.
GAME_MAX_BORDERS = 65_536
BORDER_LIMIT = 65_000             # aimed at: the steps after the cap may add a few borders back
BORDER_CAP_ROUNDS = 10
BORDERS_PER_PROVINCE = 2.93       # measured on every map made here, and on the user's other mods
MAX_PROVINCES = int(GAME_MAX_BORDERS / BORDERS_PER_PROVINCE)   # ~22,367
PROVINCE_LIMITS = {"land_min": (4, 1_000_000), "land_max": (4, 1_000_000), "sea_min": (4, 1_000_000),
                   "sea_max": (4, 1_000_000), "density_factor": (1, 20), "density_dense_factor": (1, 20),
                   "density_ref": (1, 100_000),
                   "sea_far_factor": (1, 20), "river_split_px": (0, 1_000_000),
                   "target_count": (0, MAX_PROVINCES), "city_provinces": (0, 1), "city_classes": (1, 8),
                   "city_min_px": (20, 100_000)}
SETTING_UNITS = {"target_count": "", "city_provinces": "", "city_classes": "", "density_factor": "",
                 "density_dense_factor": "", "density_ref": "", "sea_far_factor": ""}
URBAN_INDEX = 13                  # terrain.bmp index of urban (Terrain tab)
TARGET_PASSES = 3                 # land sizing passes to reach target_count
TARGET_TOLERANCE = 0.02           # ... stop within this share of the land provinces aimed at
TARGET_MODEL_RATIO = 1.2          # land provinces made / land px over the aimed size (first guess)
TARGET_BIAS = 0.02                # land aimed this much above the target: the final trim merges down to it
SEA_CENTRE_MAX_PARTS = 12         # ... into at most this many parts (estuary networks need several)
SEA_CENTRE_IN_LOOP = True         # the rule also in every main topology round (else: sea step, endgame, last pass, sweep)
SEA_CENTRE_ROUNDS = 10            # sea provinces whose centre lies outside them: split, up to this many rounds
SEA_BASE_RGB = (0x19, 0x1F, 0x61)   # user: base sea colour 191F61
LAKE_BASE_RGB = (0x3A, 0x6E, 0xC8)
TINT = 0.35                       # share of the EEZ / state colour in a sea / lake province's colour
LAKE_SPLIT_MIN_PX = 100           # a lake bordered by several countries is divided between them from this size
LAKE_PART_MIN_PX = 40             # ... as long as every part keeps at least this much
RIVER_PIECE_MIN_PX = 40           # a river split keeps pieces of at least this (and RIVER_PIECE_MIN_SHARE)
RIVER_PIECE_MIN_SHARE = 0.15
RIVER_SOURCE_DIR_PX = 6           # a source extension's direction: the river's own, over its last pixels
MIN_PROVINCE_PX = 20
GAME_MIN_PX = 12                  # HOI4: "Province N has only 8 pixels around ... Should have at least 8" (8 px fail) -- never kept below
THIN_PX, THIN_MAX_BOX_PX = 8, 128  # water strips at most this wide: cut into pieces at most this long
MAX_BOX_PX = 256                  # HOI4 "TOO LARGE BOX" (edge-band strips 4 x 430-590 px); vanilla's largest 173 x 280
MERGE_TOWARD = 0.5                # land provinces merge until at least this share of their target size
COUNTRY_FACTOR_LIMITS = (0.1, 10.0)
# Organic land splits (user: k-means cells looked too geometric): seeds from
# k-means, then each piece grows from its seed by least cost; crossing a
# finer GADM border, a river or a terrain change costs extra, so borders
# settle on them; smooth noise elsewhere makes the lines wander.
SPLIT_NOISE_W = 1.5
SPLIT_ADMIN_W = 6.0
SPLIT_RIVER_W = 4.0
SPLIT_TERRAIN_W = 2.0              # user: provinces under this are merged out (microstates / islets excepted)
LAND, SEA, LAKE = 0, 1, 2
TYPE_NAMES = ("land", "sea", "lake")
STRAY_PART_PX = 8          # a disconnected part of a province under this joins a neighbour of the same type
TOPOLOGY_ROUNDS = 12
HANDOVER_ROUNDS = 2        # the main topology loop hands over to the endgame once this many rounds in a row
HANDOVER_CHANGES = 12      # ... each changed at most this many things (or changed exactly the same)
SEED = 1939
CONTINENT_IDS = {"EU": 1, "NA": 2, "SA": 3, "OC": 4, "AF": 5, "AS": 6}
MIDDLE_EAST_ID = 7
UNIT_PX_VIEWS = 2                 # views whose projected GADM units are kept (_UNIT_PX_CACHE)
_UNIT_PX_CACHE = {}
MEMO_MIN_PX = 1_000_000           # _memo_last: whole-image calls only (windows are cheap)
KDTREE_LEAF = 16                  # scipy cKDTree's default leafsize (see _nearest_centre)
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def normalize_settings(raw):
    s = dict(PROVINCE_DEFAULTS)
    for k in PROVINCE_DEFAULTS:
        v = (raw or {}).get(k)
        if v in (None, ""):
            continue
        try:
            v = int(round(float(v)))
        except (TypeError, ValueError):
            raise core.RenderError(f"Province setting '{k}' must be a number")
        lo, hi = PROVINCE_LIMITS[k]
        if not lo <= v <= hi:
            unit = SETTING_UNITS.get(k, " px")
            raise core.RenderError(f"Province setting '{k}' must be between {lo:,} and {hi:,}{unit}")
        s[k] = v
    cf = {}
    raw_cf = (raw or {}).get("country_factors") or ""
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
    if s["land_min"] >= s["land_max"]:
        raise core.RenderError("The land province minimum must be below the maximum")
    if s["sea_min"] >= s["sea_max"]:
        raise core.RenderError("The sea province minimum must be below the maximum")
    return s


# ---------------------------------------------------------------- helpers

def _relabel(P, return_old=False):
    """Ids made dense 1..n (0 stays 0); returns (P, n) (+ old id of each new id)."""
    u, inv = np.unique(P, return_inverse=True)
    inv = inv.reshape(P.shape).astype(np.int32)
    if u[0] != 0:
        inv += 1
        u = np.concatenate([[0], u])
    if return_old:
        return inv, int(inv.max()), u
    return inv, int(inv.max())


def _majority(P, values, n):
    """Most common value of `values` (ints >= 0) per province id 1..n (0 if none)."""
    m = P > 0
    vmax = int(values[m].max()) + 1 if m.any() else 1
    key = P[m].astype(np.int64) * vmax + values[m]
    u, cnt = np.unique(key, return_counts=True)
    pid, val = u // vmax, u % vmax
    order = np.lexsort((-cnt, pid))
    firsts = np.r_[True, pid[order][1:] != pid[order][:-1]]
    out = np.zeros(n + 1, dtype=np.int64)
    out[pid[order][firsts]] = val[order][firsts]
    return out


def _nearest_centre(C, pts):
    """Index of the nearest centre for each point = cKDTree(C).query(pts)[1].
    Up to KDTREE_LEAF centres the tree is a single leaf scanned in index
    order with a strict '<' -- the lowest index wins a tie, like argmin on
    the same squared distances -- so brute force gives identical labels,
    without building a tree 13 times per k-means (the land split's cost)."""
    if len(C) > KDTREE_LEAF:
        return cKDTree(C).query(pts)[1]
    q0, q1 = pts[:, 0], pts[:, 1]
    best = (q0 - C[0, 0]) ** 2 + (q1 - C[0, 1]) ** 2
    lab = np.zeros(len(pts), dtype=np.intp)
    for j in range(1, len(C)):   # running minimum, strict '<': the lowest index keeps a tie
        d = (q0 - C[j, 0]) ** 2 + (q1 - C[j, 1]) ** 2
        m = d < best
        best = np.where(m, d, best)
        lab[m] = j
    return lab


def _kmeans_split(ys, xs, k, rng, iters=12, weights=None):
    """Labels 0..k-1 for pixels (ys, xs): k-means (k-means++ init) on a
    sample, every pixel to its nearest centre. With `weights`, the sample is
    drawn in proportion to them, so centres are denser where weights are
    high and cells smaller there (sea: small near the coast)."""
    pts = np.column_stack([ys, xs]).astype(np.float64)
    n = len(pts)
    if weights is not None:
        pr = weights / weights.sum()
        sample = pts[rng.choice(n, min(20000, max(n, 1) * 2), replace=True, p=pr)]
    else:
        sample = pts if n <= 20000 else pts[rng.choice(n, 20000, replace=False)]
    centres = [sample[rng.integers(len(sample))]]
    d2 = ((sample - centres[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = d2 / d2.sum() if d2.sum() > 0 else None
        c = sample[rng.choice(len(sample), p=p)]
        centres.append(c)
        d2 = np.minimum(d2, ((sample - c) ** 2).sum(1))
    C = np.array(centres)
    for _ in range(iters):
        lab = _nearest_centre(C, sample)
        cnt = np.bincount(lab, minlength=k)
        sy = np.bincount(lab, weights=sample[:, 0], minlength=k)
        sx = np.bincount(lab, weights=sample[:, 1], minlength=k)
        ok = cnt > 0
        prev = C.copy()
        C[ok, 0], C[ok, 1] = sy[ok] / cnt[ok], sx[ok] / cnt[ok]
        if np.array_equal(C, prev):
            break   # converged: the remaining iterations would repeat this one exactly
    return _nearest_centre(C, pts)


def _pair_borders(P, valid=None):
    """{(a, b): contact edge count} for side-adjacent pixels of different ids (a < b, both > 0)."""
    h, w = P.shape
    out = {}
    for dy, dx in ((0, 1), (1, 0)):
        a = P[: h - dy, : w - dx]
        b = P[dy:, dx:]
        m = (a != b) & (a > 0) & (b > 0)
        lo, hi = np.minimum(a[m], b[m]).astype(np.int64), np.maximum(a[m], b[m]).astype(np.int64)
        big = int(hi.max()) + 1 if len(hi) else 1
        # one int64 key per pair: same (lo, hi) order as unique(axis=0), much faster
        keys, cnt = np.unique(lo * big + hi, return_counts=True)
        for x, y, c in zip((keys // big).tolist(), (keys % big).tolist(), cnt.tolist()):
            out[(x, y)] = out.get((x, y), 0) + c
    return out


def border_count(P):
    """Neighbouring province pairs (side contact, ids > 0), the map's wrap
    (last column / first column) included = the border segments the game
    draws, see GAME_MAX_BORDERS."""
    n = int(P.max()) + 1
    keys = []
    for a, b in ((P[:, :-1], P[:, 1:]), (P[:-1, :], P[1:, :]), (P[:, -1:], P[:, :1])):
        m = (a != b) & (a > 0) & (b > 0)
        lo, hi = np.minimum(a[m], b[m]).astype(np.int64), np.maximum(a[m], b[m]).astype(np.int64)
        keys.append(lo * n + hi)
    return int(len(np.unique(np.concatenate(keys))))


def _cap_borders(P, ptype, group, protect, river, limit=BORDER_LIMIT):
    """Merge the smallest land pairs until the map is under the game's border
    limit (see GAME_MAX_BORDERS). Returns (merges, borders before, borders
    now). A merge removes `BORDERS_PER_PROVINCE` borders on average (the
    province's own: its pair with the kept one, and the ones they shared) --
    measured 2.98 on a map merged from 87,655 to 53,906 borders --, and the
    rounds re-measure, so the count lands just under the limit instead of
    well below it (every merge past it is a province lost for nothing)."""
    before = now = border_count(P)
    merges = 0
    for _ in range(BORDER_CAP_ROUNDS):
        if now <= limit:
            break
        got = _trim_to_count(P, ptype, group, max(1, int((now - limit) / BORDERS_PER_PROVINCE)), protect, river)
        if not got:
            break
        merges += got
        now = border_count(P)
    return merges, before, now


def _organic_labels(m, ys, xs, lab, parts, cost):
    """k-means labels -> least-cost growth from each cell's pixel nearest
    its centre (multi-source Dijkstra over the province's pixels, 4-neighbour
    edges weighing the mean of their pixels' costs): cells as big as the
    k-means ones, borders on the costly lines (finer admin borders, rivers,
    terrain changes, noise ridges) instead of straight Voronoi edges."""
    n = len(ys)
    idx = np.full(m.shape, -1, dtype=np.int64)
    idx[ys, xs] = np.arange(n)
    cy = np.bincount(lab, weights=ys, minlength=parts) / np.maximum(np.bincount(lab, minlength=parts), 1)
    cx = np.bincount(lab, weights=xs, minlength=parts) / np.maximum(np.bincount(lab, minlength=parts), 1)
    src = []
    for j in range(parts):
        sel = np.flatnonzero(lab == j)
        if not len(sel):
            continue
        d = (ys[sel] - cy[j]) ** 2 + (xs[sel] - cx[j]) ** 2
        src.append(int(sel[np.argmin(d)]))
    if len(src) < 2:
        return lab
    rows, cols, wts = [], [], []
    c = cost[ys, xs]
    for dy, dx in ((0, 1), (1, 0)):
        yy, xx = ys + dy, xs + dx
        ok = (yy < m.shape[0]) & (xx < m.shape[1])
        nb = np.full(n, -1, dtype=np.int64)
        nb[ok] = idx[yy[ok], xx[ok]]
        ok = nb >= 0
        rows.append(np.flatnonzero(ok))
        cols.append(nb[ok])
        wts.append((c[ok] + c[nb[ok]]) / 2.0)
    g = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    from scipy.sparse.csgraph import dijkstra
    _d, _p, sources = dijkstra(g, directed=False, indices=src, min_only=True, return_predecessors=True)
    seed_lab = np.full(n, -1, dtype=lab.dtype)
    seed_lab[src] = lab[src]
    out = lab.copy()
    reached = sources >= 0
    out[reached] = seed_lab[sources[reached]]
    return out


def _weighted_nearest(mask, seeds, cost):
    """Label of the least-cost seed for every pixel of `mask` (seeds: labels
    > 0 on some mask pixels; 4-neighbour edges weigh their pixels' mean cost)."""
    ys, xs = np.nonzero(mask)
    n = len(ys)
    idx = np.full(mask.shape, -1, dtype=np.int64)
    idx[ys, xs] = np.arange(n)
    rows, cols, wts = [], [], []
    c = cost[ys, xs]
    for dy, dx in ((0, 1), (1, 0)):
        yy, xx = ys + dy, xs + dx
        ok = (yy < mask.shape[0]) & (xx < mask.shape[1])
        nb = np.full(n, -1, dtype=np.int64)
        nb[ok] = idx[yy[ok], xx[ok]]
        ok = nb >= 0
        rows.append(np.flatnonzero(ok))
        cols.append(nb[ok])
        wts.append((c[ok] + c[nb[ok]]) / 2.0)
    out = np.zeros(mask.shape, dtype=np.int32)
    src = np.flatnonzero(seeds[ys, xs] > 0)
    if not len(src):
        return out
    g = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    from scipy.sparse.csgraph import dijkstra
    _d, _p, sources = dijkstra(g, directed=False, indices=src, min_only=True, return_predecessors=True)
    reached = sources >= 0
    lab = np.zeros(n, dtype=np.int32)
    lab[reached] = seeds[ys[sources[reached]], xs[sources[reached]]]
    out[ys, xs] = lab
    return out


def split_cost(land, rivers=None, terrain_img=None, finer_units=None, seed=SEED):
    """Per-pixel cost of the organic land split (see the SPLIT_* weights)."""
    h, w = land.shape
    rng = np.random.default_rng(seed + 7)
    noise = rng.random((h, w), dtype=np.float32)
    n1 = ndimage.gaussian_filter(noise, 2.0)
    n2 = ndimage.gaussian_filter(noise, 6.0)
    def norm(a):
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    cost = 1.0 + SPLIT_NOISE_W * (0.5 * norm(n1) + 0.5 * norm(n2))
    def edges(lab):
        e = np.zeros((h, w), dtype=bool)
        e[:, :-1] |= lab[:, :-1] != lab[:, 1:]
        e[:, 1:] |= lab[:, :-1] != lab[:, 1:]
        e[:-1, :] |= lab[:-1, :] != lab[1:, :]
        e[1:, :] |= lab[:-1, :] != lab[1:, :]
        return e
    if finer_units is not None:
        cost += SPLIT_ADMIN_W * (edges(finer_units) & (finer_units > 0))
    if terrain_img is not None:
        import terrain as terrain_mod
        idx_type = {i: t_ for i, _n, t_ in terrain_mod.TERRAIN_CLASSES}
        names = sorted(set(idx_type.values()))
        lut = np.zeros(256, dtype=np.int32)
        for i, t_ in idx_type.items():
            lut[i] = names.index(t_) + 1
        cost += SPLIT_TERRAIN_W * edges(lut[np.asarray(terrain_img)])
    if rivers is not None:
        rv = np.asarray(rivers)
        cost += SPLIT_RIVER_W * (rv if rv.dtype == bool else (rv <= 11))
    return cost.astype(np.float32)


def _merge_small(P, ptype, group, threshold, which, protect=frozenset()):
    """Merge provinces of type `which` under `threshold` px (a number or a
    per-id array: density-scaled) into a neighbour of the same type and
    group (state / sea area): smallest first; a neighbour also too small
    that the merge lifts above it first, else the longest shared border.
    Provinces in `protect` (city provinces) neither merge nor take others
    in. In place; returns merges done."""
    n = int(P.max())
    size = np.bincount(P.ravel(), minlength=n + 1).astype(np.int64)
    thr = (np.full(n + 1, float(threshold)) if np.ndim(threshold) == 0
           else np.asarray(threshold, dtype=np.float64)[: n + 1].copy())
    border = {}
    for (a, b), c in _pair_borders(P).items():
        border.setdefault(a, {})[b] = c
        border.setdefault(b, {})[a] = c
    final = np.arange(n + 1)
    heap = [(int(size[k]), k) for k in range(1, n + 1)
            if ptype[k] == which and 0 < size[k] < thr[k] and k not in protect]
    heapq.heapify(heap)
    merges = 0
    while heap:
        sz, a = heapq.heappop(heap)
        if final[a] != a or sz != size[a] or sz >= thr[a]:
            continue
        nb = [x for x in border.get(a, {}) if final[x] == x and ptype[x] == which and group[x] == group[a]
              and x not in protect]
        if not nb:
            continue
        pair_up = [x for x in nb if size[x] < thr[x] and size[a] + size[x] >= thr[a]]
        still_small = [x for x in nb if size[x] < thr[x]]  # no snowballing into one blob
        b = max(pair_up or still_small or nb, key=lambda x: (border[a][x], -size[x], -x))
        keep, gone = (a, b) if (size[a], -a) > (size[b], -b) else (b, a)
        # merged threshold = the size-weighted mean (the max snowballed: one
        # sparse municipality's 3-8k px limit made North Macedonia one state)
        thr[keep] = (thr[keep] * size[keep] + thr[gone] * size[gone]) / max(size[keep] + size[gone], 1)
        size[keep] += size[gone]
        size[gone] = 0
        for x, c in border.pop(gone, {}).items():
            if x == keep:
                continue
            border[keep][x] = border[keep].get(x, 0) + c
            border[x][keep] = border[x].get(keep, 0) + c
            border[x].pop(gone, None)
        border[keep].pop(gone, None)
        final[gone] = keep
        merges += 1
        if size[keep] < thr[keep]:
            heapq.heappush(heap, (int(size[keep]), keep))
    for k in range(n + 1):
        r = k
        while final[r] != r:
            r = final[r]
        final[k] = r
    if merges:
        P[:] = final[P]
    return merges


def _split_big(P, ptype, group, which, max_px, target_px, rng, cost=None):
    """Split provinces of type `which` over max_px into compact pieces of
    about target_px (at least enough to fit under max_px); max_px /
    target_px: numbers or per-id arrays (density-scaled). Returns the
    number of provinces split; new ids are appended (ptype / group grow)."""
    n = int(P.max())
    size = np.bincount(P.ravel(), minlength=n + 1)
    mx = np.broadcast_to(np.asarray(max_px, dtype=np.float64), (n + 1,)) if np.ndim(max_px) == 0 else max_px
    tg = np.broadcast_to(np.asarray(target_px, dtype=np.float64), (n + 1,)) if np.ndim(target_px) == 0 else target_px
    big = [k for k in range(1, n + 1) if ptype[k] == which and size[k] > mx[k]]
    if not big:
        return 0
    boxes = ndimage.find_objects(P)
    nxt = len(ptype)  # ids of merged provinces are gone from P but keep their table rows
    for k in big:
        sl = boxes[k - 1]
        m = P[sl] == k
        ys, xs = np.nonzero(m)
        parts = max(int(math.ceil(size[k] / mx[k])), int(round(size[k] / tg[k])), 2)
        lab = _kmeans_split(ys, xs, parts, rng)
        if cost is not None:
            lab = _organic_labels(m, ys, xs, lab, parts, cost[sl])
        P[sl][ys, xs] = np.where(lab == 0, k, nxt + lab - 1)
        ptype.extend([which] * (parts - 1))
        group.extend([group[k]] * (parts - 1))
        nxt += parts - 1
    return len(big)


def _components_fix(P, ptype, group, only=None):
    """Every province one 4-connected piece: the largest part keeps the id;
    parts under STRAY_PART_PX join the neighbour of the same type they
    touch most (else become provinces), larger ones become provinces.
    Returns (parts reassigned, new provinces)."""
    n = int(P.max())
    boxes = ndimage.find_objects(P)
    nxt = len(ptype)  # ids of merged provinces are gone from P but keep their table rows
    merged = created = 0
    for k in (range(1, n + 1) if only is None else sorted(i for i in only if 0 < i <= n)):
        sl = boxes[k - 1] if k - 1 < len(boxes) else None
        if sl is None:
            continue
        m = P[sl] == k
        comp, nc = ndimage.label(m, structure=CROSS)   # = the default, without rebuilding it per call
        if nc <= 1:
            continue
        sizes = np.bincount(comp.ravel())[1:]
        keep = int(np.argmax(sizes)) + 1
        big_sl = tuple(slice(max(a.start - 1, 0), a.stop + 1) for a in sl)
        oy, ox = sl[0].start - big_sl[0].start, sl[1].start - big_sl[1].start
        for c in range(1, nc + 1):
            if c == keep:
                continue
            part = np.zeros(P[big_sl].shape, dtype=bool)
            part[oy:oy + m.shape[0], ox:ox + m.shape[1]] = comp == c
            if sizes[c - 1] < STRAY_PART_PX:
                ring = ndimage.binary_dilation(part, structure=CROSS) & ~part
                nb = P[big_sl][ring]
                nb = nb[(nb > 0) & (nb != k)]
                nb = np.array([x for x in nb.tolist() if ptype[x] == ptype[k]], dtype=np.int64)
                same_g = np.array([x for x in nb.tolist() if group[x] == group[k]], dtype=np.int64)
                if len(same_g):
                    nb = same_g
                if len(nb):
                    P[big_sl][part] = np.bincount(nb).argmax()
                    merged += 1
                    continue
            P[big_sl][part] = nxt
            ptype.append(ptype[k])
            group.append(group[k])
            nxt += 1
            created += 1
    return merged, created


def _memo_last(fn):
    """Full-image calls of a function of P alone: the result of the previous
    call is returned again when P is unchanged since (a compare costs ~15 ms,
    the scans 0.1-0.4 s; ~1 in 3 full calls repeat one: the endgame's check
    then the last pass, the report after the sweep). Callers only read the
    results."""
    last = {}

    @functools.wraps(fn)
    def inner(P):
        if P.size < MEMO_MIN_PX:
            return fn(P)
        prev = last.get("P")
        if prev is not None and prev.shape == P.shape and prev.dtype == P.dtype and np.array_equal(prev, P):
            return last["r"]
        r = fn(P)
        last["P"] = P.copy()
        last["r"] = r
        return r
    inner.cache_clear = last.clear
    return inner


@_memo_last
def _x_blocks(P):
    """Top-left corners of 2x2 blocks with an X-crossing: a diagonal pair of
    one province with other provinces on both other corners, or four
    different provinces."""
    a, b, c, d = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]
    bad = ((a == d) & (b != a) & (c != a)) | ((b == c) & (a != b) & (d != b))
    bad |= (a != b) & (a != c) & (a != d) & (b != c) & (b != d) & (c != d)
    return np.nonzero(bad)


def _locally_simple(P, y, x):
    """Can pixel (y, x) leave its province without splitting it locally?
    (its province's other pixels in the 3x3 around stay 4-connected)."""
    h, w = P.shape
    v = P[y, x]
    y0, y1, x0, x1 = max(y - 1, 0), min(y + 2, h), max(x - 1, 0), min(x + 2, w)
    win = P[y0:y1, x0:x1] == v
    win[y - y0, x - x0] = False
    if not win.any():
        return False  # its last pixel here: would vanish or detach
    return ndimage.label(win, structure=CROSS)[1] == 1


def _block_ok(P, y, x):
    a, b, c, d = P[y, x], P[y, x + 1], P[y + 1, x], P[y + 1, x + 1]
    if a == d and b != a and c != a:
        return False
    if b == c and a != b and d != b:
        return False
    return len({a, b, c, d}) < 4


def _bad_around(P, y, x):
    """X-crossings among the (up to 4) 2x2 blocks containing pixel (y, x)."""
    h, w = P.shape
    return sum(1 for by in (y - 1, y) for bx in (x - 1, x)
               if 0 <= by < h - 1 and 0 <= bx < w - 1 and not _block_ok(P, by, bx))


def _fix_x(P, ptype, group=None):
    """Remove X-crossings by moving single pixels; a move is taken only if the
    2x2 blocks around the pixel end up with strictly fewer X-crossings (a
    move that fixed one block and broke its neighbour made the loop
    oscillate). Returns (fixed, type changes, left)."""
    fixed = type_changes = 0
    touched = set()
    for _round in range(20):
        ys, xs = _x_blocks(P)
        if not len(ys):
            break
        progress = 0
        for y, x in zip(ys.tolist(), xs.tolist()):
            if _block_ok(P, y, x):
                continue
            corners = [(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)]
            vals = [P[p] for p in corners]
            # candidate moves: an off-diagonal corner takes a diagonal pair's
            # province, or a corner takes a side neighbour's
            cands = []
            for i, p in enumerate(corners):
                for j in (1, 2) if i in (0, 3) else (0, 3):
                    if vals[j] != vals[i]:
                        cands.append((p, vals[j]))
            best = None
            for p, new in cands:
                old = P[p]
                before = _bad_around(P, *p)
                P[p] = new
                ok = _block_ok(P, y, x) and _bad_around(P, *p) < before  # strictly fewer X-crossings around
                P[p] = old
                if not ok or not _locally_simple(P, *p):
                    continue
                same = ptype[old] == ptype[new]
                # a type change should widen water, not cut it (a thin lake
                # cut by one land pixel fell into several lake provinces);
                # then prefer a move leaving no 1 px bump
                P[p] = new
                blocky = any(0 <= by < P.shape[0] - 1 and 0 <= bx < P.shape[1] - 1 and
                             (P[by:by + 2, bx:bx + 2] == new).all() for by in (p[0] - 1, p[0]) for bx in (p[1] - 1, p[1]))
                P[p] = old
                score = (same, not (ptype[old] != LAND and ptype[new] == LAND), blocky,
                         group is None or group[old] == group[new])
                if best is None or score > best[0]:
                    best = (score, p, new, same)
            if best is None:
                continue
            _s, p, new, same = best
            touched.update((int(P[p]), int(new)))
            P[p] = new
            fixed += 1
            progress += 1
            type_changes += 0 if same else 1
        if not progress:
            break
    return fixed, type_changes, int(len(_x_blocks(P)[0])), touched


@_memo_last
def _contact_segments(P):
    """Pairs of provinces touching along more than one separate stretch:
    {(a, b): [edge arrays per stretch]} with each stretch's edges as
    (pixel on a's side, pixel on b's side) flat indices."""
    h, w = P.shape
    W = w + 1
    recs = []
    # horizontal neighbours (y, x) | (y, x+1): lattice edge (y, x+1)-(y+1, x+1)
    a, b = P[:, :-1], P[:, 1:]
    m = (a != b) & (a > 0) & (b > 0)
    yy, xx = np.nonzero(m)
    recs.append((a[m], b[m], yy * w + xx, yy * w + xx + 1, yy * W + xx + 1, (yy + 1) * W + xx + 1))
    # vertical neighbours (y, x) / (y+1, x): lattice edge (y+1, x)-(y+1, x+1)
    a, b = P[:-1, :], P[1:, :]
    m = (a != b) & (a > 0) & (b > 0)
    yy, xx = np.nonzero(m)
    recs.append((a[m], b[m], yy * w + xx, (yy + 1) * w + xx, (yy + 1) * W + xx, (yy + 1) * W + xx + 1))
    A = np.concatenate([r[0] for r in recs]).astype(np.int64)
    B = np.concatenate([r[1] for r in recs]).astype(np.int64)
    pa = np.concatenate([r[2] for r in recs])
    pb = np.concatenate([r[3] for r in recs])
    e1 = np.concatenate([r[4] for r in recs])
    e2 = np.concatenate([r[5] for r in recs])
    if not len(A):
        return {}  # no contact at all (a window of one province)
    lo, hi = np.minimum(A, B), np.maximum(A, B)
    # which pixel is on lo's side
    plo = np.where(A == lo, pa, pb)
    phi = np.where(A == lo, pb, pa)
    pair = lo * (int(P.max()) + 1) + hi
    # graph nodes: (pair, lattice point)
    k1 = pair * ((h + 1) * W) + e1
    k2 = pair * ((h + 1) * W) + e2
    nodes, inv = np.unique(np.concatenate([k1, k2]), return_inverse=True)
    ne = len(k1)
    g = coo_matrix((np.ones(ne), (inv[:ne], inv[ne:])), shape=(len(nodes), len(nodes)))
    _n, comp = connected_components(g, directed=False)
    edge_comp = comp[inv[:ne]]
    # stretches per pair
    order = np.lexsort((edge_comp, pair))
    pair_s, comp_s = pair[order], edge_comp[order]
    out = {}
    starts = np.flatnonzero(np.r_[True, pair_s[1:] != pair_s[:-1]])
    ends = np.r_[starts[1:], len(pair_s)]
    for s0, s1 in zip(starts, ends):
        cs = comp_s[s0:s1]
        if cs[0] == cs[-1]:
            continue  # one stretch
        idx = order[s0:s1]
        key = (int(lo[idx[0]]), int(hi[idx[0]]))
        out[key] = [(plo[idx[cs == c]], phi[idx[cs == c]]) for c in np.unique(cs)]
    return out


def _fix_double(P, ptype, group, allow_tiny=False, cost=None):
    """Split one province of each pair touching in several places so each
    piece holds one contact. Returns (pairs fixed, pairs left)."""
    segs = _contact_segments(P)
    if not segs:
        return 0, 0, set()
    h, w = P.shape
    size = np.bincount(P.ravel())
    boxes = ndimage.find_objects(P)
    touched = set()
    fixed = 0
    nxt = len(ptype)  # ids of merged provinces are gone from P but keep their table rows
    for (a, b), stretches in sorted(segs.items(), key=lambda kv: -len(kv[1])):
        if a in touched or b in touched:
            continue
        done = False
        # never split a lake (Ladoga / Onega came out in rectangles): split
        # the shore province instead
        # nor, between land and sea, the sea cell: the sea's count stays the
        # sea settings' (user), whatever the land provinces' size
        for x, side in sorted(((a, 0), (b, 1)), key=lambda t: (ptype[t[0]] == LAKE, ptype[t[0]] == SEA, -size[t[0]])):
            if ptype[x] == LAKE:
                continue
            seeds = [np.unique(s[side]) for s in stretches]
            if sum(len(s_) for s_ in seeds) >= size[x]:
                continue
            box = boxes[x - 1]
            win = P[box]
            mx = win == x
            lab = np.zeros(win.shape, dtype=np.int32)
            for i, s_ in enumerate(seeds, start=1):
                lab[s_ // w - box[0].start, s_ % w - box[1].start] = i
            grp = np.where(mx, 0, -1).astype(np.int32)
            todo = mx & (lab == 0)
            if cost is not None and ptype[x] == LAND:
                part = _weighted_nearest(mx, lab, cost[box])  # organic, like the land splits
            else:
                part = states._nearest_from_outside(lab, grp, todo, np.zeros(win.shape, dtype=bool))
            if (mx & (part == 0)).any():
                continue  # not all reached: leave it
            psz = np.bincount(part[mx].ravel(), minlength=len(seeds) + 1)[1:]
            if not allow_tiny and (psz < MIN_PROVINCE_PX).any():  # tiny merges would undo it
                continue  # would leave a tiny piece: try the other province
            for i in range(2, len(seeds) + 1):
                sel = mx & (part == i)
                if not sel.any():
                    continue
                touched.add(nxt)
                win[sel] = nxt
                ptype.append(ptype[x])
                group.append(group[x])
                nxt += 1
            touched.update((a, b))
            fixed += 1
            done = True
            break
        if done:
            continue
        # a split would leave a tiny piece (one stretch is a few pixels):
        # trim the minor stretches instead -- the non-lake province's pixels
        # along them go to another neighbour of the same type and group
        longest = int(np.argmax([len(s_[0]) for s_ in stretches]))
        # same group (state) first; then any neighbour of the same type (a
        # state border moves by the stretch's few pixels -- the states are
        # made of their provinces anyway)
        # last: the lake's own pixels on the minor stretches to a third land
        # province (the lake loses a pixel or two, stays connected)
        order = sorted(((a, 0), (b, 1)), key=lambda t: (ptype[t[0]] == LAKE, size[t[0]]))
        tries = [(True, xs_, False) for xs_ in order] + [(False, xs_, False) for xs_ in order] +                 [(False, xs_, True) for xs_ in order if ptype[xs_[0]] == LAKE]
        for strict, (x, side), lake_px in tries:
            if ptype[x] == LAKE and not lake_px:
                continue
            other = b if x == a else a
            pix = np.unique(np.concatenate([s_[side] for j, s_ in enumerate(stretches) if j != longest]))
            moves = []
            saved = []
            for f_ in pix.tolist():
                y_, x_ = divmod(f_, w)
                nb = [int(P[yy, xx]) for yy, xx in ((y_ - 1, x_), (y_ + 1, x_), (y_, x_ - 1), (y_, x_ + 1))
                      if 0 <= yy < h and 0 <= xx < w]
                nb = [v for v in set(nb) if v not in (x, other, 0)
                      and (ptype[v] == ptype[x] or (lake_px and ptype[v] == LAND))
                      and (group[v] == group[x] or not strict)]
                if not nb or (lake_px and not _locally_simple(P, y_, x_)):
                    moves = None
                    break
                # the neighbour leaving the fewest X-crossings around (a
                # trim that made one got undone by the X fix, forever)
                old = int(P[y_, x_])
                best = None
                for v in nb:
                    P[y_, x_] = v
                    bad = _bad_around(P, y_, x_)
                    if best is None or bad < best[0]:
                        best = (bad, v)
                P[y_, x_] = best[1]
                saved.append((y_, x_, old))
                moves.append((y_, x_, best[1]))
            if not moves:
                for y_, x_, old in reversed(saved):
                    P[y_, x_] = old
                continue
            if any(_bad_around(P, y_, x_) for y_, x_, _v in moves):
                for y_, x_, old in reversed(saved):
                    P[y_, x_] = old
                continue
            touched.update((a, b) + tuple(v for _y, _x, v in moves))
            fixed += 1
            done = True
            break
        if done:
            continue
        # two water provinces of the same kind (lake parts of two states whose
        # dividing line interleaves; two sea cells a 2-edge sliver apart that
        # a split + the tiny merge kept rebuilding): the smaller joins the larger
        if ptype[a] != LAND and ptype[a] == ptype[b] and size[a] and size[b]:
            small, big = (a, b) if size[a] < size[b] else (b, a)
            sl = boxes[small - 1]
            if sl is not None:
                win = P[sl]
                win[win == small] = big
                size[big] += size[small]
                size[small] = 0
                touched.update((a, b))
                fixed += 1
                continue
        # last resort, small provinces only (merging big ones snowballed: a
        # shore province swallowed its neighbours round after round): merge
        # the smaller non-lake one into a same-type, same-group neighbour
        # (its partner first) -- the pair's contacts go with it
        for x in sorted((a, b), key=lambda v: (ptype[v] == LAKE, size[v])):
            if ptype[x] == LAKE or size[x] >= 5 * MIN_PROVINCE_PX:
                continue
            other = b if x == a else a
            box = boxes[x - 1]
            if box is None:
                continue
            box = tuple(slice(max(s_.start - 1, 0), s_.stop + 1) for s_ in box)
            win = P[box]
            mx = win == x
            ring = ndimage.binary_dilation(mx, structure=CROSS) & ~mx
            nb = win[ring]
            nb = [v for v in nb[nb > 0].tolist() if v != x and ptype[v] == ptype[x] and group[v] == group[x]]
            if not nb:
                continue
            target = other if other in nb else int(np.bincount(nb).argmax())
            win[mx] = target
            touched.update((a, b, target))
            fixed += 1
            break
    return fixed, len(segs) - fixed, touched


def _sea_centres_outside(P, ptype):
    """Sea provinces whose centre (mean of their pixels) is not one of their
    own pixels (user rule: a sea province's centre lies within it)."""
    pt = np.asarray(ptype, dtype=np.int8)
    n = int(P.max())
    if n >= len(pt):
        pt = np.concatenate([pt, np.full(n + 1 - len(pt), -1, dtype=np.int8)])
    flat = P.ravel()
    idx = np.flatnonzero(pt[flat] == SEA)
    if not len(idx):
        return []
    ids = flat[idx]
    w = P.shape[1]
    ys = idx // w
    xs = idx - ys * w
    size = np.bincount(ids, minlength=n + 1)
    has = size > 0
    cy = np.rint(np.bincount(ids, weights=ys, minlength=n + 1)[has] / size[has]).astype(np.int64)
    cx = np.rint(np.bincount(ids, weights=xs, minlength=n + 1)[has] / size[has]).astype(np.int64)
    k = np.flatnonzero(has)
    return k[P[cy, cx] != k].tolist()


def _part_mask(shape, ys, xs, sel):
    out = np.zeros(shape, dtype=bool)
    out[ys[sel], xs[sel]] = True
    return out


def _local_doubles(P, sl, pad=3):
    """Pairs touching in several places within the window sl (+ pad)."""
    h, w = P.shape
    win = P[max(sl[0].start - pad, 0):min(sl[0].stop + pad, h), max(sl[1].start - pad, 0):min(sl[1].stop + pad, w)]
    return set(_contact_segments(win))


def _split_sea_centres(P, ptype, group, rounds=SEA_CENTRE_ROUNDS, tries=4, strict=False):
    """Split every sea province whose centre lies outside it in two, until
    none is left or the rounds run out. Cuts into 2, 3 .. parts of >=
    MIN_PROVINCE_PX (k-means seeds, parts grown through the province, so a
    thin estuary is cut along its length), `tries` per count, best first:
    every part's centre inside > no new double contact nearby > parts in
    one piece; fewest parts that put the centres inside (a cut the
    double-contact fix merged back came round again: USA, 12 per round;
    refusing cuts that add a double contact left big provinces unsplit).
    Only a cut putting every part's centre inside is made. One with no such
    cut and under 5 x MIN_PROVINCE_PX (thin channels: two parallel 1 px arms
    of an estuary, a V-shaped lagoon) joins a sea neighbour -- one whose
    centre stays inside, then no new double contact, same sea area, longest
    border -- and the merged province is cut again next round if needed.
    Left: such shapes with no sea neighbour (a lagoon enclosed by land).
    strict (the final sweep, nothing runs after it): only cuts / joins
    adding no double contact. Returns (splits + joins, left)."""
    rng = np.random.default_rng(SEED + 11)
    split = 0
    for _ in range(rounds):
        bad = _sea_centres_outside(P, ptype)
        if not bad:
            return split, 0
        boxes = ndimage.find_objects(P)
        first_new = len(ptype)
        done = 0
        grown = set()  # took a small one in this round: its box is stale, next round
        for k in bad:
            if k in grown:
                continue
            sl = boxes[k - 1]
            m = P[sl] == k
            if not m.any():
                continue  # joined another this round
            ys, xs = np.nonzero(m)
            before = _local_doubles(P, sl)
            chosen = None
            if len(ys) >= 2 * MIN_PROVINCE_PX:
                # 2, 3, ... parts (>= MIN_PROVINCE_PX each), grown through the
                # province from k-means seeds (thin estuaries / fjord channels:
                # plain k-means halves were bent or in pieces, centres outside)
                # fewest parts putting every centre inside; else the fewest
                # parts with some inside -- the parts still outside are cut
                # again next round (an estuary network needed ~40 parts at
                # once, 4 then more on the bad parts only)
                ones = np.ones(m.shape, dtype=np.float32)
                chosen, n_parts, fallback = None, 0, None
                for parts in range(2, min(SEA_CENTRE_MAX_PARTS, len(ys) // MIN_PROVINCE_PX) + 1):
                    best = None
                    for _t in range(tries):
                        lab = _kmeans_split(ys, xs, parts, rng)
                        lab = _organic_labels(m, ys, xs, lab, parts, ones)
                        cnt = np.bincount(lab, minlength=parts)
                        if (cnt < MIN_PROVINCE_PX).any():
                            continue
                        tmp = len(ptype)  # the new parts' ids-to-be
                        P[sl][ys, xs] = np.where(lab == 0, k, tmp + lab - 1)
                        no_dbl = not (_local_doubles(P, sl) - before)
                        bad_px = sum(int(cnt[j]) for j in range(parts)
                                     if P[sl][int(round(ys[lab == j].mean())), int(round(xs[lab == j].mean()))]
                                     != (k if j == 0 else tmp + j - 1))
                        pieces = all(ndimage.label(_part_mask(m.shape, ys, xs, lab == j), structure=CROSS)[1] == 1
                                     for j in range(parts))
                        P[sl][ys, xs] = k
                        key_ = (bad_px == 0, no_dbl, -bad_px, pieces)
                        if best is None or key_ > best[0]:
                            best = (key_, lab)
                        if bad_px == 0 and no_dbl and pieces:
                            break
                    if best is None or (strict and not best[0][1]):
                        continue
                    if best[0][0]:
                        chosen, n_parts = best[1], parts
                        break
                    if fallback is None and not strict and -best[0][2] < len(ys):
                        fallback = (best[1], parts)
                if chosen is None and fallback is not None:
                    chosen, n_parts = fallback
            if chosen is not None:
                P[sl][ys, xs] = np.where(chosen == 0, k, len(ptype) + chosen - 1)
                ptype.extend([SEA] * (n_parts - 1))
                group.extend([group[k]] * (n_parts - 1))
                done += 1
                continue
            box = tuple(slice(max(s_.start - 1, 0), s_.stop + 1) for s_ in sl)
            win = P[box]
            mk = win == k
            nb = win[ndimage.binary_dilation(mk, structure=CROSS) & ~mk]
            nb = [v for v in nb.tolist() if 0 < v < len(ptype) and ptype[v] == SEA
                  and v <= len(boxes) and boxes[v - 1] is not None]   # not one made this round
            same = [v for v in nb if group[v] == group[k]]
            if len(ys) >= 5 * MIN_PROVINCE_PX:
                continue  # no cut works and too big to join away: left, reported
            # receivers ranked: centre still inside, no new double contact,
            # same sea area, longest border; if none keeps its centre inside,
            # the merged province is cut again next round (more room for a
            # valid cut than the thin piece alone)
            ranked = []
            for to in set(nb):
                win[mk] = to
                bt = boxes[to - 1]
                ub = (slice(min(bt[0].start, box[0].start), max(bt[0].stop, box[0].stop)),
                      slice(min(bt[1].start, box[1].start), max(bt[1].stop, box[1].stop)))
                ty, tx = np.nonzero(P[ub] == to)
                inside_ = P[int(round(ty.mean())) + ub[0].start, int(round(tx.mean())) + ub[1].start] == to
                no_dbl_ = not (_local_doubles(P, sl) - before)
                win[mk] = k
                ranked.append((not inside_, not no_dbl_, to not in same, -nb.count(to), to))
            if strict:
                ranked = [r_ for r_ in ranked if not r_[1]]
            if ranked:
                to = min(ranked)[-1]
                win[mk] = to
                grown.add(to)
                done += 1
        if not done:
            return split, len(bad)
        split += done
        _components_fix(P, ptype, group, only=set(bad) | set(range(first_new, len(ptype))))
    return split, len(_sea_centres_outside(P, ptype))


def _trim_to_count(P, ptype, group, excess, protect=frozenset(), river=None, rounds=12):
    """Target count (option): merge `excess` pairs of land provinces, the
    smallest pairs first (same state, neither a city province, not two
    sides of a river -- most of their border on river pixels); one merge
    per province per round, and a merge that makes a double contact is
    undone. Merging can't make an X-crossing where there is none. Returns
    the merges done."""
    h, w = P.shape
    done = 0
    failed = set()  # pairs whose merge made a double contact: not retried
    for _ in range(rounds):
        if done >= excess:
            break
        n = int(P.max())
        size = np.bincount(P.ravel(), minlength=n + 1)
        riv_edges = {}
        if river is not None:
            for dy, dx in ((0, 1), (1, 0)):
                a_ = P[: h - dy, : w - dx]
                b_ = P[dy:, dx:]
                m_ = (a_ != b_) & (river[: h - dy, : w - dx] | river[dy:, dx:])
                lo, hi = np.minimum(a_[m_], b_[m_]), np.maximum(a_[m_], b_[m_])
                for (x_, y_), c_ in zip(*np.unique(np.stack([lo, hi], 1), axis=0, return_counts=True)):
                    riv_edges[(int(x_), int(y_))] = riv_edges.get((int(x_), int(y_)), 0) + int(c_)
        cand = []
        for (x_, y_), c_ in _pair_borders(P).items():
            if (x_ >= len(ptype) or y_ >= len(ptype) or ptype[x_] != LAND or ptype[y_] != LAND
                    or group[x_] != group[y_] or x_ in protect or y_ in protect
                    or riv_edges.get((x_, y_), 0) * 2 > c_ or (x_, y_) in failed):
                continue
            cand.append((int(size[x_] + size[y_]), -c_, x_, y_))
        cand.sort()
        used = set()
        merged = []
        for _s, _c, x_, y_ in cand:
            if done + len(merged) >= excess:
                break
            if x_ in used or y_ in used:
                continue
            used.update((x_, y_))
            keep, gone = (x_, y_) if size[x_] >= size[y_] else (y_, x_)
            merged.append((keep, gone))
        if not merged:
            break
        # apply (remember the pixels to undo)
        boxes = ndimage.find_objects(P)
        undo = []
        for keep, gone in merged:
            sl = boxes[gone - 1]
            m = P[sl] == gone
            P[sl][m] = keep
            undo.append((keep, gone, sl, m))
        bad = set()
        for (x_, y_) in _contact_segments(P):
            bad.update((x_, y_))
        ok = 0
        for keep, gone, sl, m in undo:
            if keep in bad:
                P[sl][m] = gone
                failed.add((min(keep, gone), max(keep, gone)))
            else:
                ok += 1
        done += ok
    return done


# Sea areas = the sea strategic regions to be (user's rules, 2026-10-11):
# (1) they follow the EEZ lines, even when that makes small regions
# (Belgium); (2) a large EEZ part is split into a coastal band, its border
# parallel to the coast (the smoothed distance to land), cut across by
# straight lines along long coasts, and offshore water beyond; (3) water
# outside every EEZ gets straight-edged cells. Sea provinces are then cut
# inside the areas, so the regions' borders are exactly those lines.
SEA_AREA_KEEP_PX = 60_000     # an EEZ part up to this size is one area, whatever its shape
COAST_BAND_PX = 60            # coastal band: water within this distance of land (smoothed)
COAST_BAND_AREA_PX = 25_000   # a band is cut across into pieces of about this size
OFFSHORE_AREA_PX = 80_000     # offshore / high-seas cells of about this size (Voronoi: straight edges)
BAND_LAND_MIN_PX = 8_000      # land masses smaller than this make no band of their own (a ring around an islet
                              # has its centre outside: the Kurils came out as slivers; Shetland a "keyhole")
SEA_AREA_MIN_PX = 2_000       # smaller pieces join a neighbouring area of the same EEZ part
STRAIT_CUT_PX = 6             # bands / offshore water are separated where narrower than 2x this (straits)
SEA_AREA_TINY_PX = 500        # a smaller area with no such neighbour joins any (slivers between EEZ polygons)
AREA_CENTRE_ROUNDS = 4        # an area whose centre lies outside it is split in two, at most this often


def _area_centre_inside(A, aid, sl):
    m = A[sl] == aid
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean(), xs.mean()
    for y in {int(math.floor(cy)), int(math.ceil(cy))}:
        for x in {int(math.floor(cx)), int(math.ceil(cx))}:
            if not (0 <= y < m.shape[0] and 0 <= x < m.shape[1]) or not m[y, x]:
                return False
    return True


def _sea_areas(sea, zone, have_eez, rng):
    """(area id image A: 1.. on the sea, 0 elsewhere; zone of each area
    (index 0 unused), stats) -- see SEA_AREA_* above. `zone`: EEZ zone per
    pixel (0 = none)."""
    h, w = sea.shape
    # distance to land masses that count (islets make no band), smoothed so
    # the band's edge runs parallel to the coast, not along every cove
    landm = ~sea
    lab, _nl = ndimage.label(landm, structure=CROSS)
    big = landm & (np.bincount(lab.ravel())[lab] >= BAND_LAND_MIN_PX)
    d = ndimage.distance_transform_edt(~big) if big.any() else np.full((h, w), 1e9)
    band_all = ndimage.gaussian_filter(np.minimum(d, 4 * COAST_BAND_PX), COAST_BAND_PX / 3.0) <= COAST_BAND_PX
    A = np.zeros((h, w), dtype=np.int32)
    area_zone = [0]
    part_of = [0]   # EEZ part (connected piece of a zone) of each area
    key = np.where(sea, zone + 1, 0)
    boxes = ndimage.find_objects(key)
    stats = {"eez_parts": 0, "kept_whole": 0, "banded": 0, "cells": 0}
    n_part = 0

    def cut(mask_box, sl, target, z, part, straits=False):
        """Areas for one mask (in box sl): its pieces, each cut into about
        size / target Voronoi cells (k-means centres: straight edges). With
        `straits`, first separated where the water narrows to under
        2 x STRAIT_CUT_PX (user: Morocco's band ran from the Atlantic through
        Gibraltar into the Mediterranean in one region): the wider parts
        (the mask eroded) are the seeds, every pixel goes to the nearest."""
        if straits:
            core_ = ndimage.binary_erosion(mask_box, structure=CROSS, iterations=STRAIT_CUT_PX)
            cc, ncc = ndimage.label(core_, structure=CROSS)
            if ncc > 1:
                _d, (iy, ix) = ndimage.distance_transform_edt(cc == 0, return_indices=True)
                pc = np.where(mask_box, cc[iy, ix], 0)
                # a pixel cut off from its seed by the narrowing: by mask components
                pc2, npc2 = ndimage.label(mask_box, structure=CROSS)
                lab_ = pc * (npc2 + 1) + pc2
                _u, pc = np.unique(np.where(mask_box, lab_, 0), return_inverse=True)
                pc = pc.reshape(mask_box.shape)
                return _cut_pieces(pc, int(pc.max()), sl, target, z, part)  # small bits merged below
        pc, npc = ndimage.label(mask_box, structure=CROSS)
        return _cut_pieces(pc, npc, sl, target, z, part)

    def _cut_pieces(pc, npc, sl, target, z, part):
        for c, pb in enumerate(ndimage.find_objects(pc), start=1):
            if pb is None:
                continue
            ys, xs = np.nonzero(pc[pb] == c)
            ys, xs = ys + pb[0].start, xs + pb[1].start
            k = max(1, int(round(len(ys) / target)))
            labs = _kmeans_split(ys, xs, k, rng) if k > 1 else np.zeros(len(ys), dtype=np.int64)
            for j in range(k):
                sel = labs == j
                if not sel.any():
                    continue
                aid = len(area_zone)
                A[sl][ys[sel], xs[sel]] = aid
                area_zone.append(z)
                part_of.append(part)
            stats["cells"] += k
    for z in np.unique(key[sea]).tolist():
        zb = boxes[z - 1]
        cz, nz = ndimage.label(key[zb] == z, structure=CROSS)
        for c in range(1, nz + 1):
            cm = cz == c
            n_part += 1
            stats["eez_parts"] += 1
            size = int(cm.sum())
            zid = z - 1
            if zid == 0 and have_eez:  # outside every EEZ: straight-edged cells
                cut(cm, zb, OFFSHORE_AREA_PX, zid, n_part)
                continue
            if size <= SEA_AREA_KEEP_PX:
                aid = len(area_zone)
                A[zb][cm] = aid
                area_zone.append(zid)
                part_of.append(n_part)
                stats["kept_whole"] += 1
                continue
            bm = cm & band_all[zb]
            om = cm & ~bm
            # far water is always its own area (user: region 685 held Morocco's
            # offshore water with both its coasts); only a scrap joins the band
            if om.sum() < 2 * SEA_AREA_MIN_PX:
                bm, om = cm, None
            elif bm.sum() < 2 * SEA_AREA_MIN_PX:
                bm, om = None, cm   # (almost) no coast: all offshore
            if bm is not None:
                cut(bm, zb, COAST_BAND_AREA_PX, zid, n_part, straits=True)
            if om is not None:
                cut(om, zb, OFFSHORE_AREA_PX, zid, n_part, straits=True)
            stats["banded"] += 1
    # pieces: every area one 4-connected piece; pieces under SEA_AREA_MIN_PX
    # (and Voronoi cells cut apart by the coast) join the area of the same
    # EEZ part they touch most, larger detached pieces become areas
    part_of = np.array(part_of)
    for _round in range(3):
        changed = 0
        abox = ndimage.find_objects(A)
        for aid in range(1, len(abox) + 1):
            sl = abox[aid - 1]
            if sl is None:
                continue
            sl = tuple(slice(max(a_.start - 1, 0), a_.stop + 1) for a_ in sl)
            m = A[sl] == aid
            pc, npc = ndimage.label(m, structure=CROSS)
            sizes = np.bincount(pc.ravel())[1:]
            small_whole = npc == 1 and sizes[0] < SEA_AREA_MIN_PX
            if npc < 2 and not small_whole:
                continue
            keep = int(np.argmax(sizes)) + 1
            for c in range(1, npc + 1):
                if c == keep and not small_whole:
                    continue
                piece = pc == c
                ring = ndimage.binary_dilation(piece, structure=CROSS) & ~piece
                nb_all = A[sl][ring]
                nb_all = nb_all[(nb_all > 0) & (nb_all != aid)]
                nb = nb_all[part_of[nb_all] == part_of[aid]]
                if not len(nb) and piece.sum() < SEA_AREA_TINY_PX:
                    nb = nb_all  # a sliver between EEZ polygons, not a real zone: any neighbour
                if len(nb):
                    A[sl][piece] = np.bincount(nb).argmax()
                    changed += 1
                elif c != keep:
                    na = len(area_zone)
                    A[sl][piece] = na
                    area_zone.append(area_zone[aid])
                    part_of = np.append(part_of, part_of[aid])
                    changed += 1
        if not changed:
            break
    # an area's centre must lie inside it (the strategic regions' rule): split in two
    for _round in range(AREA_CENTRE_ROUNDS):
        changed = 0
        abox = ndimage.find_objects(A)
        for aid in range(1, len(abox) + 1):
            sl = abox[aid - 1]
            if sl is None or _area_centre_inside(A, aid, sl):
                continue
            ys, xs = np.nonzero(A[sl] == aid)
            if len(ys) < 2 * SEA_AREA_MIN_PX:
                continue
            labs = _kmeans_split(ys, xs, 2, rng)
            na = len(area_zone)
            A[sl][ys[labs == 1], xs[labs == 1]] = na
            area_zone.append(area_zone[aid])
            part_of = np.append(part_of, part_of[aid])
            changed += 1
        stats["centre_splits"] = stats.get("centre_splits", 0) + changed
        if not changed:
            break
    A, n_a, old = _relabel(A, return_old=True)
    area_zone = [0] + [area_zone[int(o)] for o in old[1:]]
    stats["areas"] = int(n_a)
    return A, area_zone, stats


def _make_sea(sea, eez_zones, s):
    """Sea provinces on their own id image S (1..; 0 elsewhere): each
    4-connected part of an EEZ is one sea area, cut by weighted k-means into
    cells growing with the distance from land (A(d) = A0 * min(1 + d / G,
    far factor)), merged below sea_min within the area, and cells whose
    centre lies outside them split. Independent of the land provinces, so
    it runs before them (the target count needs the sea's). Returns (S,
    ptype, group, {group: EEZ owner tag}, stats)."""
    h, w = sea.shape
    rng = np.random.default_rng(SEED + 3)
    zone = eez_zones[0] if eez_zones is not None else np.zeros((h, w), dtype=np.int32)
    zone_tags = eez_zones[1] if eez_zones is not None else []
    # sea areas (the future sea strategic regions): see _sea_areas
    A, area_zone, area_stats = _sea_areas(sea, zone, eez_zones is not None, np.random.default_rng(SEED + 5))
    S = A.copy()
    ptype, group = [LAND], [0]   # row 0: not sea
    owner = {}
    for aid in range(1, len(area_zone)):
        g = -(10_000_000 + aid)
        ptype.append(SEA)
        group.append(g)
        z = area_zone[aid]
        owner[g] = zone_tags[z - 1] if z >= 1 and z - 1 < len(zone_tags) else None
    n_areas = len(area_zone) - 1
    a0 = (s["sea_min"] + s["sea_max"]) / 2
    d_land = ndimage.distance_transform_edt(sea)
    grow = np.minimum(1.0 + d_land / (2.0 * math.sqrt(a0)), float(s["sea_far_factor"]))
    split = 0
    size = np.bincount(S.ravel(), minlength=len(ptype))
    boxes = ndimage.find_objects(S)
    for k in range(len(boxes)):
        pid = k + 1
        if boxes[k] is None:
            continue
        sl = boxes[k]
        m = S[sl] == pid
        ys, xs = np.nonzero(m)
        wts = 1.0 / (a0 * grow[sl][m])
        parts = int(round(wts.sum()))
        if parts < 2 and size[pid] <= s["sea_max"] * s["sea_far_factor"]:
            continue
        parts = max(parts, 2)
        lab = _kmeans_split(ys, xs, parts, rng, weights=wts)
        nid = len(ptype)
        S[sl][ys, xs] = np.where(lab == 0, pid, nid + lab - 1)
        ptype.extend([SEA] * (parts - 1))
        group.extend([group[pid]] * (parts - 1))
        split += 1
    _components_fix(S, ptype, group)
    merged = _merge_small(S, np.array(ptype), np.array(group), s["sea_min"], SEA)
    # user rule: a sea province's centre lies within it, else it is split
    centre_split, _left = _split_sea_centres(S, ptype, group)
    count = int(len(np.unique(S[S > 0])))
    return S, ptype, group, owner, {"areas": n_areas, "area_design": area_stats, "split": split, "merged": merged,
                                    "centre_split": centre_split, "count": count}


def _city_provinces(P, ptype, group, land, stl, terrain_img, points, max_class, min_px):
    """Carve a province out of the land for every City-tab city of classes
    1..max_class (points: (x, y, class, name), most important first): the
    urban area (Terrain tab's urban, 4-connected, within the city's state)
    it lies on or within 3 px of. An urban area holding several cities is
    divided between them, each taking the part nearest to it through the
    area (a city whose part would be under min_px leaves it to the others:
    least important first). Parts under min_px -- and cities with no urban
    area -- grow by the nearest land pixels of their state. Returns (mask
    of the city provinces' pixels, [(id, name, class, px, y, x)])."""
    h, w = P.shape
    urban = (np.asarray(terrain_img) == URBAN_INDEX) & land & (stl > 0)
    comp, _nc = ndimage.label(urban, structure=CROSS)
    # urban areas split by state: dense ids on the urban pixels, pixel lists by id
    uy, ux = np.nonzero(urban)
    _u, inv = np.unique(comp[uy, ux].astype(np.int64) * (int(stl.max()) + 1) + stl[uy, ux], return_inverse=True)
    code = np.zeros((h, w), dtype=np.int32)
    code[uy, ux] = inv + 1
    order_ = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order_], np.arange(len(_u) + 1))

    # 1. each city's urban area (0: none) and the pixel it grows from
    cities = []   # [rank, cls, name, y, x, sid, area]
    seen = set()
    for rank, (x, y, cls, name) in enumerate(points):
        x, y = int(round(x)), int(round(y))
        if cls > max_class or not (0 <= y < h and 0 <= x < w) or not land[y, x] or stl[y, x] <= 0:
            continue
        sid = int(stl[y, x])
        y0, y1, x0, x1 = max(y - 3, 0), min(y + 4, h), max(x - 3, 0), min(x + 4, w)
        win = code[y0:y1, x0:x1]
        cand = (win > 0) & (stl[y0:y1, x0:x1] == sid)
        c = 0
        if cand.any():
            yy, xx = np.nonzero(cand)
            j = int(np.argmin((yy + y0 - y) ** 2 + (xx + x0 - x) ** 2))
            c = int(win[yy[j], xx[j]])
            y, x = int(yy[j] + y0), int(xx[j] + x0)   # its seed: the nearest urban pixel
        if (y, x) in seen:
            continue  # two cities on one pixel: the first
        seen.add((y, x))
        cities.append([rank, int(cls), name, y, x, sid, c])
    cities.sort(key=lambda c_: (c_[1], c_[0]))

    # 2. urban areas shared by several cities: divided by nearest city
    pieces = {}   # rank -> (ys, xs)
    by_area = {}
    for c_ in cities:
        if c_[6]:
            by_area.setdefault(c_[6], []).append(c_)
    for area, members in by_area.items():
        sel = order_[starts[area - 1]:starts[area]]
        ys, xs = uy[sel], ux[sel]
        if len(members) == 1:
            pieces[members[0][0]] = (ys, xs)
            continue
        sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        mask = np.zeros((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start), dtype=bool)
        mask[ys - sl[0].start, xs - sl[1].start] = True
        ones = np.ones(mask.shape, dtype=np.float32)
        members = list(members)
        while True:
            seeds = np.zeros(mask.shape, dtype=np.int32)
            for j, c_ in enumerate(members, start=1):
                seeds[c_[3] - sl[0].start, c_[4] - sl[1].start] = j
            lab = _weighted_nearest(mask, seeds, ones)
            sz = np.bincount(lab[mask], minlength=len(members) + 1)[1:]
            small = [j for j in range(len(members)) if sz[j] < min_px]
            if not small or len(members) == 1:
                break
            members.pop(max(small, key=lambda j: (members[j][1], members[j][0])))  # least important
        for j, c_ in enumerate(members, start=1):
            iy, ix = np.nonzero(lab == j)
            pieces[c_[0]] = (iy + sl[0].start, ix + sl[1].start)

    # 3. carve, most important first; small ones grow within their state
    city_mask = np.zeros((h, w), dtype=bool)
    made = []
    for rank, cls, name, y, x, sid, area in cities:
        if area and rank not in pieces:
            continue  # its part of a shared urban area went to the others
        ys, xs = pieces.get(rank, (np.array([y]), np.array([x])))
        keep = ~city_mask[ys, xs]
        ys, xs = ys[keep], xs[keep]
        if not len(ys):
            continue  # inside a more important city's province
        if len(ys) < min_px:
            r = int(math.ceil(math.sqrt(min_px / math.pi))) + 3
            sl = (slice(max(ys.min() - r, 0), min(ys.max() + r + 1, h)),
                  slice(max(xs.min() - r, 0), min(xs.max() + r + 1, w)))
            blob = np.zeros(P[sl].shape, dtype=bool)
            blob[ys - sl[0].start, xs - sl[1].start] = True
            ok = (stl[sl] == sid) & land[sl] & ~city_mask[sl] & ~blob
            d = np.where(ok, ndimage.distance_transform_edt(~blob), np.inf).ravel()
            grow = np.argsort(d, kind="stable")[: min_px - len(ys)]
            grow = grow[np.isfinite(d[grow])]
            iy, ix = np.unravel_index(grow, blob.shape)
            ys = np.concatenate([ys, iy + sl[0].start])
            xs = np.concatenate([xs, ix + sl[1].start])
        pid = len(ptype)
        P[ys, xs] = pid
        ptype.append(LAND)
        group.append(sid)
        city_mask[ys, xs] = True
        made.append((pid, name, cls, int(len(ys)), y, x))
    return city_mask, made


def _protected(P, mask, n=None):
    """Ids with most of their pixels in `mask` (the city provinces, whatever
    ids splits gave them)."""
    n = int(P.max()) if n is None else n
    tot = np.bincount(P.ravel(), minlength=n + 1)
    inn = np.bincount(P[mask].ravel(), minlength=n + 1)
    return frozenset(np.flatnonzero((inn * 2 > tot) & (tot > 0)).tolist()) - {0}


# ---------------------------------------------------------------- main

def _fix_x_force(P, ptype, group, rounds=50):
    """X-crossings left by the careful pass (user: suppressed no matter
    what): any single-pixel move that makes the block valid (same type /
    state first), and as a last resort the whole 2x2 block to its most
    common province. Returns (moves, blocks painted)."""
    moves = painted = 0
    for _ in range(rounds):
        ys, xs = _x_blocks(P)
        if not len(ys):
            break
        for y, x in zip(ys.tolist(), xs.tolist()):
            if _block_ok(P, y, x):
                continue
            corners = [(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)]
            vals = [int(P[c]) for c in corners]
            done = False
            for pref in (2, 1, 0):
                for i, p in enumerate(corners):
                    for j in range(4):
                        new = vals[j]
                        if new == vals[i]:
                            continue
                        old = int(P[p])
                        score = (ptype[old] == ptype[new]) + (group[old] == group[new])
                        if score < pref or (pref > 0 and ptype[old] != LAND and ptype[new] == LAND):
                            continue
                        P[p] = new
                        if _block_ok(P, y, x):
                            moves += 1
                            done = True
                            break
                        P[p] = old
                    if done:
                        break
                if done:
                    break
            if not done:
                u, c = np.unique(vals, return_counts=True)
                P[y:y + 2, x:x + 2] = u[c.argmax()]
                painted += 1
    return moves, painted


def _force_double(P, ptype, group, rounds=3):
    """Double contacts the loop couldn't settle (a trim and an X fix undoing
    each other): water / land pairs pixel by pixel, then a split allowing
    tiny pieces (same-kind water pairs merge in it), then any X-crossing is
    forced out; repeated. Returns (fixed, left)."""
    fixed = 0
    for _ in range(rounds):
        segs = _contact_segments(P)
        if not segs:
            break
        for (a, b) in segs:
            if (ptype[a] == LAND) != (ptype[b] == LAND) and _resolve_lake_pair(P, a, b, ptype):
                fixed += 1
        f_, _l, _t = _fix_double(P, ptype, group, allow_tiny=True)
        fixed += f_
        _fix_x_force(P, ptype, group)
    return fixed, len(_contact_segments(P))


def _resolve_lake_pair(P, a, b, ptype, limit=40):
    """A lake / land pair still touching along several stretches: pixel by
    pixel (<= limit), on the minor stretches, a land pixel goes to another
    land neighbour or a lake pixel to a land neighbour other than the pair
    (then, if neither works, the lake pixel to the pair's land province) --
    each move keeping its province locally connected and adding no
    X-crossing -- until the pair touches along one stretch. Returns True if
    it does."""
    h, w = P.shape
    lake, land_ = (a, b) if ptype[a] != LAND else (b, a)  # the water province (lake or sea) and the land one
    for _ in range(limit):
        m = (P == lake) | (P == land_)
        yy, xx = np.nonzero(m)
        y0, y1 = max(yy.min() - 2, 0), min(yy.max() + 3, h)
        x0, x1 = max(xx.min() - 2, 0), min(xx.max() + 3, w)
        win = P[y0:y1, x0:x1]
        segs = _contact_segments(win)
        key = (min(lake, land_), max(lake, land_))
        if key not in segs:
            return True
        stretches = segs[key]
        longest = int(np.argmax([len(s_[0]) for s_ in stretches]))
        ww = x1 - x0
        cand = []
        for j, (plo, phi) in enumerate(stretches):
            if j == longest:
                continue
            for f_ in np.concatenate([plo, phi]).tolist():
                cand.append((f_ // ww + y0, f_ % ww + x0))
        moved = False
        for pref in (0, 1, 2):
            for y_, x_ in cand:
                cur = int(P[y_, x_])
                nb = {int(P[p_]) for p_ in ((y_ - 1, x_), (y_ + 1, x_), (y_, x_ - 1), (y_, x_ + 1))
                      if 0 <= p_[0] < h and 0 <= p_[1] < w}
                if pref == 0 and cur == land_:
                    opts = [v for v in nb if v not in (lake, land_, 0) and ptype[v] == LAND]
                elif pref == 1 and cur == lake:
                    opts = [v for v in nb if v not in (lake, land_, 0) and ptype[v] in (LAND, ptype[lake])]
                elif pref == 2 and cur == lake:
                    opts = [land_] if land_ in nb else []
                else:
                    continue
                if not opts or not _locally_simple(P, y_, x_):
                    continue
                before = _bad_around(P, y_, x_)
                for v in opts:
                    P[y_, x_] = v
                    if _bad_around(P, y_, x_) <= before:
                        moved = True
                        break
                    P[y_, x_] = cur
                if moved:
                    break
            if moved:
                break
        if not moved:
            return False
    return False


def _brute_pair(P, a, b, radius=2):
    """Last resort for a pair still touching twice (every single fix made an
    X-crossing beside it): in a (2r+1)^2 window around its minor stretch,
    try every one- and two-pixel reassignment to a province present there;
    keep the first that ends the pair's second contact with no X-crossing,
    no new double contact and no province split locally. True if found."""
    import itertools
    h, w = P.shape
    ys, xs = np.nonzero((P == a) | (P == b))
    Y0, Y1 = max(ys.min() - 2, 0), min(ys.max() + 3, h)
    X0, X1 = max(xs.min() - 2, 0), min(xs.max() + 3, w)
    area = P[Y0:Y1, X0:X1]
    segs = _contact_segments(area)
    key = (min(a, b), max(a, b))
    if key not in segs:
        return True
    st = segs[key]
    longest = int(np.argmax([len(s_[0]) for s_ in st]))
    aw = X1 - X0
    minor = np.concatenate([np.concatenate([s_[0], s_[1]]) for j, s_ in enumerate(st) if j != longest])
    cy, cx = int((minor // aw).mean()) + Y0, int((minor % aw).mean()) + X0
    y0, y1 = max(cy - radius, 0), min(cy + radius + 1, h)
    x0, x1 = max(cx - radius, 0), min(cx + radius + 1, w)
    base_dbl = len(segs)

    def ok():
        win = P[max(y0 - 1, 0):y1 + 1, max(x0 - 1, 0):x1 + 1]
        if len(_x_blocks(win)[0]):
            return False
        s2 = _contact_segments(P[Y0:Y1, X0:X1])
        return key not in s2 and len(s2) < base_dbl
    cells = [(y, x) for y in range(y0, y1) for x in range(x0, x1)]
    moves = []
    for (y, x) in cells:
        vals = {int(P[yy, xx]) for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                if 0 <= yy < h and 0 <= xx < w} - {int(P[y, x])}
        moves += [((y, x), v) for v in vals]
    for n_ in (1, 2):
        for combo in itertools.combinations(moves, n_):
            if len({c[0] for c in combo}) < n_:
                continue
            old = [(c[0], int(P[c[0]])) for c in combo]
            if not all(_locally_simple(P, *c[0]) for c in combo):
                continue
            for c in combo:
                P[c[0]] = c[1]
            if ok():
                return True
            for pos, v in old:
                P[pos] = v
    return False


def _merge_tiny(P, ptype, group, country_of, min_px=MIN_PROVINCE_PX):
    """Provinces under min_px merged into the neighbour of the same type they
    touch most -- land: same state first, then same country; sea: same EEZ
    area first, then any sea; a tiny lake becomes land of the land province
    it touches most (user: every province under 20 px merged out). A land
    province with no neighbour of its country (a microstate, an islet) is
    kept. Returns (merged, kept)."""
    n = int(P.max())
    size = np.bincount(P.ravel(), minlength=n + 1)
    boxes = ndimage.find_objects(P)
    merged = kept = 0
    for k in np.argsort(size[1:], kind="stable") + 1:
        k = int(k)
        if size[k] == 0 or size[k] >= min_px or k - 1 >= len(boxes) or boxes[k - 1] is None:
            continue
        box = tuple(slice(max(s_.start - 1, 0), s_.stop + 1) for s_ in boxes[k - 1])
        win = P[box]
        m = win == k
        if not m.any():
            continue
        ring = ndimage.binary_dilation(m, structure=CROSS) & ~m
        nb = win[ring]
        nb = nb[nb > 0]
        tk = ptype[k]
        if tk == LAND:
            prefs = [[v for v in nb.tolist() if ptype[v] == LAND and group[v] == group[k]],
                     [v for v in nb.tolist() if ptype[v] == LAND and country_of(v) is not None
                      and country_of(v) == country_of(k)]]
        elif tk == SEA:  # a sea pocket enclosed by land becomes that land
            prefs = [[v for v in nb.tolist() if ptype[v] == SEA and group[v] == group[k]],
                     [v for v in nb.tolist() if ptype[v] == SEA], [v for v in nb.tolist() if ptype[v] == LAND]]
        else:
            prefs = [[v for v in nb.tolist() if ptype[v] == LAKE], [v for v in nb.tolist() if ptype[v] == LAND]]
        if tk == LAND and size[k] < GAME_MIN_PX:  # never below the game's minimum: islets join the water around
            prefs.append([v for v in nb.tolist() if ptype[v] != LAND])
            prefs.append([v for v in nb.tolist() if ptype[v] == LAND])
        target = None
        for cand in prefs:
            if cand:
                target = int(np.bincount(cand).argmax())
                break
        if target is None:
            kept += 1
            continue
        win[m] = target
        size[target] += size[k]
        size[k] = 0
        merged += 1
    return merged, kept


def _split_tall(P, ptype, group, pstate, max_box=MAX_BOX_PX):
    """Provinces whose bounding box exceeds max_box (the edge bands' 4 px
    wide strips -- the sea ones and the unowned land one: "Province 6334 has
    TOO LARGE BOX", 370 px; land ports "too far away from center") cut across
    their long side into equal bands. Returns the number of provinces cut."""
    boxes = ndimage.find_objects(P)
    cut = 0
    for k, sl in enumerate(boxes):
        pid = k + 1
        if sl is None:
            continue
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        # thin strips (the edge bands): shorter still -- the game wants a coastal
        # sea province's port stack within ~100 px of its centre ("Ship in port is
        # too far away from center (dist 0 105)" on 256 px strips)
        limit = THIN_MAX_BOX_PX if min(hh, ww) <= THIN_PX else max_box
        if max(hh, ww) <= limit:
            continue
        sub = P[sl]
        ys, xs = np.nonzero(sub == pid)
        along = ys if hh >= ww else xs
        parts = int(math.ceil(max(hh, ww) / limit))
        edges = np.quantile(along, np.linspace(0, 1, parts + 1)[1:-1])
        band = np.searchsorted(edges, along, side="right")
        for b in range(1, parts):
            sel = band == b
            if sel.any():
                nid = len(ptype)
                ptype.append(ptype[pid])
                group.append(group[pid])
                pstate.append(pstate[pid] if pid < len(pstate) else 0)
                sub[ys[sel], xs[sel]] = nid
        cut += 1
    return cut


def _join_lake_fragments(lakes, land, P, lake_ids, gap=2):
    """A lake the map draws as a string of fragments 1-2 px apart (a thin
    reservoir: Kama, Volga) becomes one piece: per HydroLAKES lake (never
    two different lakes), a land pixel with two of its fragments in its 3x3
    is made lake, repeated `gap` times. Returns pixels changed."""
    changed = 0
    if lake_ids is None:
        return 0
    boxes = ndimage.find_objects(lake_ids)
    for lid, sl in enumerate(boxes, start=1):
        if sl is None:
            continue
        sl = tuple(slice(max(s_.start - gap, 0), s_.stop + gap) for s_ in sl)
        for _ in range(gap):
            mine = (lake_ids[sl] == lid) & lakes[sl]
            comp, n = ndimage.label(mine, structure=CROSS)
            if n < 2:
                break
            hi = ndimage.maximum_filter(comp, size=3)
            lo = -ndimage.maximum_filter(np.where(comp > 0, -comp, -(n + 1)), size=3)
            join = land[sl] & (hi > 0) & (lo <= n) & (hi != lo)
            if not join.any():
                break
            lakes[sl] |= join
            land[sl] &= ~join
            P[sl][join] = 0
            lake_ids[sl][join] = lid
            changed += int(join.sum())
    return changed


def _bridge_lakes(lakes, land, P):
    """A lake joined to itself only corner to corner (thin winding
    reservoirs: Kama, Volga) becomes one piece: one of the two land pixels
    of such a 2x2 becomes lake. Returns pixels changed (masks / P in place)."""
    changed = 0
    for _ in range(50):
        a, b, c, d = lakes[:-1, :-1], lakes[:-1, 1:], lakes[1:, :-1], lakes[1:, 1:]
        diag = (a & d & ~b & ~c) | (b & c & ~a & ~d)
        ys, xs = np.nonzero(diag)
        if not len(ys):
            break
        for y, x in zip(ys.tolist(), xs.tolist()):
            if lakes[y, x] and lakes[y + 1, x + 1] and not lakes[y, x + 1] and not lakes[y + 1, x]:
                py, px = (y, x + 1) if land[y, x + 1] else (y + 1, x)
            elif lakes[y, x + 1] and lakes[y + 1, x] and not lakes[y, x] and not lakes[y + 1, x + 1]:
                py, px = (y, x) if land[y, x] else (y + 1, x + 1)
            else:
                continue
            if not land[py, px]:
                continue
            lakes[py, px] = True
            land[py, px] = False
            P[py, px] = 0
            changed += 1
    return changed


def _province_factor(P, dens, ref, cap, dense_cap=1.0, country_mult=None):
    """Per province id: sqrt(ref / mean density) clipped to [1 / dense_cap,
    cap] -- bigger provinces where few people live, smaller where many do
    (user: sizes from density, not from how fine a country's admin units
    are) -- times the country's user factor (country_mult per id)."""
    n = int(P.max())
    if dens is None or (cap <= 1 and dense_cap <= 1):
        f = np.ones(n + 1)
    else:
        cnt = np.bincount(P.ravel(), minlength=n + 1).astype(np.float64)
        tot = np.bincount(P.ravel(), weights=dens.ravel().astype(np.float64), minlength=n + 1)
        mean = tot / np.maximum(cnt, 1)
        f = np.clip(np.sqrt(ref / np.maximum(mean, 1e-3)), 1.0 / max(dense_cap, 1.0), cap)
    if country_mult is not None:
        f = f * country_mult(n)
    return f


def _river_back(river, y, x, steps):
    """`steps` pixels back along the river from one of its ends (the river is
    1 px wide and 4-connected, so the walk is unambiguous): the pixel the
    direction of flow is read from."""
    h, w = river.shape
    seen = {(y, x)}
    cur = (y, x)
    for _ in range(steps):
        nxt = next((p for p in ((cur[0] - 1, cur[1]), (cur[0] + 1, cur[1]),
                                (cur[0], cur[1] - 1), (cur[0], cur[1] + 1))
                    if p not in seen and 0 <= p[0] < h and 0 <= p[1] < w and river[p]), None)
        if nxt is None:
            break
        seen.add(nxt)
        cur = nxt
    return cur


def _source_extensions(P, river):
    """A river's **source** is a dead end: the river stops inside a province,
    so the province is not cut in two and stays one (user, 2026-10-15:
    "extend the river at its source to properly cut the province where the
    source of a river is located").

    Every dead end that lies *inside* a province -- all 8 of its neighbours
    in the same one, so not an end sitting on the border, where the river
    simply carries on into the next province -- is continued in a straight
    line, in the direction the river was running (`_river_back`), until it
    leaves that province or meets another river. River + extension then runs
    from border to border and `_river_splits` can cut the province along it.

    **For cutting only**: these pixels are plain land, never river pixels in
    rivers.bmp, and they are not in `split_cost` either -- they are not a
    river on the ground, only the line the one that is there would take.
    Returns the added mask."""
    h, w = river.shape
    p = np.pad(river, 1)
    deg = (p[:-2, 1:-1].astype(np.uint8) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) * river
    ends = river & (deg <= 1)
    out = np.zeros_like(river)
    ey, ex = np.nonzero(ends)
    if not len(ey):
        return out
    own = P[ey, ex]
    inside = own > 0
    for dy in (-1, 0, 1):                       # inside its province, border excluded
        for dx in (-1, 0, 1):
            yy, xx = ey + dy, ex + dx
            ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            inside &= ok & (P[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)] == own)
    for i in np.flatnonzero(inside):
        y, x, k = int(ey[i]), int(ex[i]), int(own[i])
        by, bx = _river_back(river, y, x, RIVER_SOURCE_DIR_PX)
        dy, dx = y - by, x - bx
        if not dy and not dx:                   # a lone river pixel: no direction to follow
            continue
        n = max(abs(dy), abs(dx))
        sy, sx = dy / n, dx / n
        cy, cx = y + 0.5, x + 0.5
        for _ in range(h + w):                  # it always leaves the province; a plain safety net
            cy, cx = cy + sy, cx + sx
            iy, ix = int(cy), int(cx)
            if not (0 <= iy < h and 0 <= ix < w) or P[iy, ix] != k or river[iy, ix]:
                break
            out[iy, ix] = True
    return out


def _river_splits(P, ptype, group, river, min_px, given=None, country_of=None):
    """Land provinces of >= min_px that rivers cut into pieces are split
    along them (user: gameplay): every piece >= RIVER_PIECE_MIN_PX and
    RIVER_PIECE_MIN_SHARE of the province becomes a province (two rivers ->
    three provinces); smaller pieces across a river go to the neighbouring
    province on their side -- same state first, else another state of the
    same country (user: from the province or the state), never another
    country -- else to the nearest kept piece; each river pixel goes to the
    kept piece it borders most (all to the largest left 1 px strips where a
    river ran on past the other piece).

    A river that ends inside a province -- its source -- is continued to that
    province's border first (`_source_extensions`), so a source cuts its
    province like any other stretch. Returns splits done."""
    given = given if given is not None else [0]
    # a source inside a province cuts it too (_source_extensions); the
    # provinces move between rounds, so the extensions are taken afresh
    river = river | _source_extensions(P, river)
    n = int(P.max())
    size = np.bincount(P.ravel(), minlength=n + 1)
    on_river = np.bincount(P[river].ravel(), minlength=n + 1)
    boxes = ndimage.find_objects(P)
    nxt = len(ptype)
    splits = 0
    for k in range(1, n + 1):
        if ptype[k] != LAND or not on_river[k] or k - 1 >= len(boxes) or boxes[k - 1] is None:
            continue
        sl = boxes[k - 1]
        m = P[sl] == k
        comp, nc = ndimage.label(m & ~river[sl], structure=CROSS)
        if nc < 2:
            continue
        sizes = np.bincount(comp.ravel())[1:]
        need = max(RIVER_PIECE_MIN_PX, RIVER_PIECE_MIN_SHARE * size[k])
        keep_ids = [c + 1 for c in np.argsort(-sizes) if c == int(np.argmax(sizes))
                    or (size[k] >= min_px and sizes[c] >= need)]
        wide = tuple(slice(max(s_.start - 1, 0), s_.stop + 1) for s_ in sl)
        oy, ox = sl[0].start - wide[0].start, sl[1].start - wide[1].start
        for c in range(1, nc + 1):
            if c in keep_ids:
                continue
            bit = np.zeros(P[wide].shape, dtype=bool)
            bit[oy:oy + m.shape[0], ox:ox + m.shape[1]] = comp == c
            ring = ndimage.binary_dilation(bit, structure=CROSS) & ~bit & ~river[wide]
            nb = [v for v in P[wide][ring].tolist() if v > 0 and v != k and ptype[v] == LAND]
            same_state = [v for v in nb if group[v] == group[k]]
            same_country = [v for v in nb if country_of is not None and country_of(v) is not None
                            and country_of(v) == country_of(k)]
            cand = same_state or same_country
            if cand:
                P[wide][bit] = int(np.bincount(cand).argmax())
                given[0] += 1
        m = P[sl] == k
        if len(keep_ids) < 2:
            continue
        keep = m & np.isin(comp, keep_ids)
        _d, (iy, ix) = ndimage.distance_transform_edt(~keep, return_indices=True)
        lab = np.where(keep, comp, comp[iy, ix])
        # river pixels: the kept piece bordering them most
        rv = m & river[sl]
        if rv.any():
            ry, rx = np.nonzero(rv)
            votes = np.zeros((len(ry), nc + 1), dtype=np.int32)
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                yy, xx = np.clip(ry + dy, 0, m.shape[0] - 1), np.clip(rx + dx, 0, m.shape[1] - 1)
                v = np.where(keep[yy, xx], comp[yy, xx], 0)
                votes[np.arange(len(ry)), v] += 1
            votes[:, 0] = 0
            best = votes.argmax(1)
            lab[ry, rx] = np.where(votes.max(1) > 0, best, lab[ry, rx])
        sub = P[sl]
        for c in keep_ids[1:]:
            sub[m & (lab == c)] = nxt
            ptype.append(LAND)
            group.append(group[k])
            nxt += 1
            splits += 1
    return splits


def _block_ok_v(a, b, c, d):
    """_block_ok on four values (None: outside the image -> no block)."""
    if a == d and b != a and c != a:
        return False
    if b == c and a != b and d != b:
        return False
    return len({a, b, c, d}) < 4


def _thin_decide(P, y, x, h, w, ptype, group, country_of):
    """_clean_thin's decision for pixel (y, x), on a plain-Python copy of
    its 3x3 window (numpy scalar reads were the cost): (the province it
    moves to or None, the window, its complete 2x2 blocks' corners). Same
    rules and order as the original loop."""
    y0, x0 = max(y - 1, 0), max(x - 1, 0)
    sub = P[y0:y + 2, x0:x + 2].tolist()
    win = [[None] * 3 for _ in range(3)]
    for i, row in enumerate(sub):
        for j, v in enumerate(row):
            win[y0 - y + 1 + i][x0 - x + 1 + j] = v
    k = win[1][1]
    nb = [win[0][1], win[2][1], win[1][0], win[1][2]]   # up, down, left, right (the original's order)
    nb = [v for v in nb if v is not None and v != k and v > 0 and ptype[v] == LAND]
    cand = [v for v in nb if group[v] == group[k]] or \
           [v for v in nb if country_of(v) is not None and country_of(v) == country_of(k)]
    if not cand:
        return None, win, None
    corners = [(by, bx) for by in (0, 1) for bx in (0, 1)
               if win[by][bx] is not None and win[by][bx + 1] is not None
               and win[by + 1][bx] is not None and win[by + 1][bx + 1] is not None]

    def bad():
        return sum(1 for by, bx in corners
                   if not _block_ok_v(win[by][bx], win[by][bx + 1], win[by + 1][bx], win[by + 1][bx + 1]))
    before = bad()
    for v in sorted(set(cand), key=lambda v: -cand.count(v)):
        win[1][1] = v
        if bad() <= before and any(win[by][bx] == v and win[by][bx + 1] == v and win[by + 1][bx] == v
                                   and win[by + 1][bx + 1] == v for by, bx in corners):
            win[1][1] = k
            return v, win, corners
    win[1][1] = k
    return None, win, corners


def _thin_mask(P, pt, body, ys=None, xs=None):
    """Thin pixels (land, in no complete 2x2 block of their province, the
    province having such a block somewhere): the whole image, or only the
    pixels (ys, xs) -> their values."""
    h, w = P.shape
    if ys is None:
        full = (P[:-1, :-1] == P[:-1, 1:]) & (P[:-1, :-1] == P[1:, :-1]) & (P[:-1, :-1] == P[1:, 1:])
        inblk = np.zeros((h, w), dtype=bool)
        inblk[:-1, :-1] |= full
        inblk[:-1, 1:] |= full
        inblk[1:, :-1] |= full
        inblk[1:, 1:] |= full
        return (P > 0) & ~inblk & (pt[np.clip(P, 0, len(pt) - 1)] == LAND) & (body[np.clip(P, 0, len(body) - 1)] > 0)
    vals = P[ys, xs]
    inb = np.zeros(len(ys), dtype=bool)
    for dy, dx in ((-1, -1), (-1, 0), (0, -1), (0, 0)):
        by, bx = ys + dy, xs + dx
        okb = (by >= 0) & (bx >= 0) & (by < h - 1) & (bx < w - 1)
        byc, bxc = np.clip(by, 0, h - 2), np.clip(bx, 0, w - 2)
        okb &= (P[byc, bxc] == vals) & (P[byc, bxc + 1] == vals) & (P[byc + 1, bxc] == vals) & (P[byc + 1, bxc + 1] == vals)
        inb |= okb
    return (vals > 0) & ~inb & (pt[np.clip(vals, 0, len(pt) - 1)] == LAND) & (body[np.clip(vals, 0, len(body) - 1)] > 0)


def _clean_thin(P, ptype, group, country_of, rounds=30, touched=None):
    """1 px wide protrusions of land provinces (user): a pixel in no 2x2
    block of its own province (a spur, a strip, a staircase) goes to the
    neighbouring land province it touches most -- same state first, then
    same country, never another country -- repeated until none is left.
    A province made only of such pixels is left to the tiny merge. Returns
    pixels moved.

    Speed (same result as the plain pixel loop, checked on captured calls
    with _dev/province_replay.py): a pixel's decision only reads its 3x3
    window, so each round a vectorized test first finds the thin pixels
    that *could* move (a 2x2 block of one neighbouring land province of the
    same state or country around them); only those, and thin pixels next
    to one that moved earlier in the round, run through the loop, in the
    same (row-major) order. Between rounds the thin set is re-tested only
    around the moves, each province's count of complete 2x2 blocks kept up
    to date move by move (a count crossing zero: the whole set again).
    touched: a set that receives the ids of the provinces pixels left or
    joined."""
    import heapq
    h, w = P.shape
    moved = 0
    pt = np.asarray(ptype)
    grp = np.asarray(group, dtype=np.int64)
    codes = {}
    ctry = np.array([codes.setdefault(c_, len(codes)) if c_ is not None else -1
                     for c_ in (country_of(v) for v in range(len(ptype)))], dtype=np.int64)
    full = (P[:-1, :-1] == P[:-1, 1:]) & (P[:-1, :-1] == P[1:, :-1]) & (P[:-1, :-1] == P[1:, 1:])
    body = np.bincount(P[:-1, :-1][full].ravel(), minlength=int(P.max()) + 1).astype(np.int64)
    del full
    thin = _thin_mask(P, pt, body)
    sel3 = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    for _ in range(rounds):
        ys, xs = np.nonzero(thin)
        if not len(ys):
            break
        k_ = P[ys, xs]
        kc = np.clip(k_, 0, len(pt) - 1)
        possible = np.zeros(len(ys), dtype=bool)
        for dy, dx in ((-1, -1), (-1, 0), (0, -1), (0, 0)):   # block's top-left corner
            by, bx = ys + dy, xs + dx
            ok = (by >= 0) & (bx >= 0) & (by < h - 1) & (bx < w - 1)
            byc, bxc = np.clip(by, 0, h - 2), np.clip(bx, 0, w - 2)
            blk = np.stack([P[byc, bxc], P[byc, bxc + 1], P[byc + 1, bxc], P[byc + 1, bxc + 1]], 1)
            me = (ys - byc) * 2 + (xs - bxc)
            others = blk[np.arange(len(ys))[:, None], sel3[me]]
            v = others[:, 0]
            vc = np.clip(v, 0, len(pt) - 1)
            ok &= (others[:, 1] == v) & (others[:, 2] == v) & (v != k_) & (v > 0) & (v < len(pt)) & (pt[vc] == LAND)
            ok &= (grp[vc] == grp[kc]) | ((ctry[vc] >= 0) & (ctry[vc] == ctry[kc]))
            possible |= ok
        heap = (ys * w + xs)[possible].tolist()
        heapq.heapify(heap)
        done_ = set()
        moved_px = []
        body_before = {}
        while heap:
            f_ = heapq.heappop(heap)
            if f_ in done_:
                continue
            done_.add(f_)
            y, x = divmod(f_, w)
            best, win, corners = _thin_decide(P, y, x, h, w, ptype, group, country_of)
            if best is None:
                continue
            k = win[1][1]
            # complete blocks through this pixel: k's go, best's come
            for by, bx in corners:
                oth = [win[by + i_][bx + j_] for i_ in (0, 1) for j_ in (0, 1) if (by + i_, bx + j_) != (1, 1)]
                if oth[0] == oth[1] == oth[2]:
                    if oth[0] == k:
                        body_before.setdefault(k, int(body[k]))
                        body[k] -= 1
                    elif oth[0] == best:
                        body_before.setdefault(best, int(body[best]))
                        body[best] += 1
            P[y, x] = best
            moved_px.append(f_)
            if touched is not None:
                touched.add(k)
                touched.add(best)
            # the thin pixels after it whose 3x3 window it is in
            for yy in (y, y + 1):
                for xx in (x - 1, x, x + 1):
                    if 0 <= yy < h and 0 <= xx < w and (yy > y or xx > x) and thin[yy, xx]:
                        g_ = yy * w + xx
                        if g_ not in done_:
                            heapq.heappush(heap, g_)
        moved += len(moved_px)
        if not moved_px:
            break
        if any((b0 > 0) != (body[p_] > 0) for p_, b0 in body_before.items()):
            thin = _thin_mask(P, pt, body)   # a province lost its last / got its first block
            continue
        my, mx = np.divmod(np.array(moved_px), w)
        cy = (my[:, None] + np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])[None, :]).ravel()
        cx = (mx[:, None] + np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])[None, :]).ravel()
        okc = (cy >= 0) & (cy < h) & (cx >= 0) & (cx < w)
        near = np.unique(cy[okc] * w + cx[okc])
        ny, nx = np.divmod(near, w)
        thin[ny, nx] = _thin_mask(P, pt, body, ny, nx)
    return moved


def _vary(base, n, rng, dh, ds, dv):
    """n colours around `base` (HSV jitter), as an (n, 3) uint8 array."""
    import colorsys
    h0, s0, v0 = colorsys.rgb_to_hsv(*(c / 255.0 for c in base))
    hh = (h0 + rng.uniform(-dh, dh, n)) % 1.0
    ss = np.clip(s0 + rng.uniform(-ds, ds, n), 0.05, 1.0)
    vv = np.clip(v0 + rng.uniform(-dv, dv, n), 0.08, 1.0)
    out = np.array([colorsys.hsv_to_rgb(a, b, c) for a, b, c in zip(hh, ss, vv)]).reshape(-1, 3)
    return np.clip(np.rint(out * 255), 0, 255).astype(np.uint8)


def _colours(P, n, base, spread, rng, min_nb=10.0):
    """Unique province colours: each a variation of its base colour
    (base[i] rgb, spread[i] = (dh, ds, dv)), never black, never repeated,
    and >= min_nb (RGB) from each neighbouring province when possible."""
    rgb = np.zeros((n + 1, 3), dtype=np.uint8)
    for i in range(1, n + 1):
        rgb[i] = _vary(base[i], 1, rng, *spread[i])[0]
    pairs = np.array(list(_pair_borders(P)), dtype=np.int64).reshape(-1, 2)
    for _round in range(12):
        code = (rgb[:, 0].astype(np.int64) << 16) | (rgb[:, 1].astype(np.int64) << 8) | rgb[:, 2]
        code[0] = -1
        bad = set()
        u, first, cnt = np.unique(code[1:], return_index=True, return_counts=True)
        dup = np.flatnonzero(np.isin(code[1:], u[cnt > 1])) + 1
        firsts = set((first[cnt > 1] + 1).tolist())
        bad.update(i for i in dup.tolist() if i not in firsts)
        bad.update(np.flatnonzero(code == 0).tolist())
        if len(pairs):
            d = np.sqrt(((rgb[pairs[:, 0]].astype(np.float64) - rgb[pairs[:, 1]]) ** 2).sum(1))
            bad.update(pairs[d < min_nb, 1].tolist())
        bad.discard(0)
        if not bad:
            break
        grow = 1.0 + _round / 3.0
        for i in bad:
            dh, ds, dv = spread[i]
            rgb[i] = _vary(base[i], 1, rng, dh * grow, ds * grow, dv * grow)[0]
    # last resort for exact duplicates: step the blue channel
    code = (rgb[:, 0].astype(np.int64) << 16) | (rgb[:, 1].astype(np.int64) << 8) | rgb[:, 2]
    seen = {0}
    for i in range(1, n + 1):
        c = int(code[i])
        while c in seen or c == 0:
            c = (c & ~255) | ((c + 1) & 255)
            if (c & 255) == 0:
                c = (c + (1 << 8)) & 0xFFFFFF
        seen.add(c)
        rgb[i] = ((c >> 16) & 255, (c >> 8) & 255, c & 255)
    return rgb


def _lakes(P, lakes, land, stl, st_code, unit, unit_code, ptype, group, pstate, report, unit_state=None):
    """Lake provinces and their owners (user's rules):
      * a lake bordered by one country -> that country, in the neighbouring
        state it shares the longest shore with;
      * bordered by several and under LAKE_SPLIT_MIN_PX -> the country with
        the longest shore (its state with the longest shore);
      * bordered by several and bigger -> divided along the GADM country
        border (the rasterized subdivisions cover the lakes), as long as
        every part keeps >= LAKE_PART_MIN_PX (smaller parts join the part
        they touch most); each part in its country's state with the longest
        shore along it.
    New lake provinces are appended (ptype / group / pstate)."""
    comp, n_l = ndimage.label(lakes, structure=CROSS)
    nxt = len(ptype)
    divided = shared = split_states = 0
    stl_ok = np.where(land, stl, 0)
    for k, sl in enumerate(ndimage.find_objects(comp), start=1):
        if sl is None:
            continue
        sl = tuple(slice(max(a.start - 1, 0), a.stop + 1) for a in sl)
        L = comp[sl] == k
        ring = ndimage.binary_dilation(L, structure=CROSS) & ~L & (stl_ok[sl] > 0)
        rs = stl_ok[sl][ring]
        rc = st_code[rs] if len(rs) else np.zeros(0, dtype=np.int64)

        def best_state(mask_ring, country=None):
            v = stl_ok[sl][mask_ring & ring]
            if country is not None:
                v = v[st_code[v] == country]
            return int(np.bincount(v).argmax()) if len(v) else 0

        parts = [(L, None)]
        countries = np.unique(rc)
        if len(countries) >= 2:
            shared += 1
            if L.sum() >= LAKE_SPLIT_MIN_PX:
                uc = np.where(unit[sl] > 0, unit_code[unit[sl]], -1)
                lc = np.where(L, uc, -1)
                has = L & np.isin(lc, countries)
                if has.any():
                    _d, (iy, ix) = ndimage.distance_transform_edt(~has, return_indices=True)
                    lc = np.where(L, lc[iy, ix], -1)
                    # parts per country, the small ones join the part they touch most
                    for _ in range(20):
                        plab = np.zeros(L.shape, dtype=np.int32)
                        pc = []
                        for c in np.unique(lc[L]).tolist():
                            cc, nn = ndimage.label(L & (lc == c), structure=CROSS)
                            for j in range(1, nn + 1):
                                plab[cc == j] = len(pc) + 1
                                pc.append(c)
                        sizes = np.bincount(plab.ravel())[1:]
                        small = [j for j in range(len(sizes)) if sizes[j] < LAKE_PART_MIN_PX]
                        if not small or len(sizes) == 1:
                            break
                        j = min(small, key=lambda j: sizes[j])
                        pm = plab == j + 1
                        rr = ndimage.binary_dilation(pm, structure=CROSS) & ~pm & L
                        nbp = plab[rr]
                        nbp = nbp[nbp > 0]
                        if not len(nbp):
                            break
                        lc[pm] = pc[np.bincount(nbp).argmax() - 1]
                    if len(sizes) > 1:
                        parts = [(plab == j + 1, pc[j]) for j in range(len(sizes))]
                        divided += 1
            if len(parts) == 1:  # the whole lake to the country with the longest shore
                cc_, cnt = np.unique(rc, return_counts=True)
                parts = [(L, int(cc_[cnt.argmax()]))]
        elif len(countries) == 1:
            parts = [(L, int(countries[0]))]
        # each country's part divided between the states around it (user:
        # lakes fit the adjacent states; one state per lake made states snake
        # around the Karelian lakes and along the Volga reservoirs)
        state_parts = []
        for pm, country in parts:
            pring = ndimage.binary_dilation(pm, structure=CROSS) & ~pm
            sid = best_state(pring, country) if country is not None else best_state(pring)
            if not sid and country is not None:
                sid = best_state(np.ones_like(pm), country)
            shore = stl_ok[sl][pring & ring]
            if country is not None:
                shore = shore[st_code[shore] == country]
            shore_states = np.unique(shore)
            if pm.sum() >= LAKE_SPLIT_MIN_PX and len(shore_states) >= 2 and unit_state is not None:
                us = np.where(pm & (unit[sl] > 0), unit_state[np.clip(unit[sl], 0, len(unit_state) - 1)], 0)
                us = np.where(np.isin(us, shore_states), us, 0)
                if not (us > 0).any():  # no GADM unit: the nearest shore state
                    src = np.where(pring & ring & np.isin(stl_ok[sl], shore_states), stl_ok[sl], 0)
                    us = src
                _d, (iy, ix) = ndimage.distance_transform_edt(~(us > 0), return_indices=True)
                ls = np.where(pm, us[iy, ix], 0)
                for _ in range(20):  # parts under LAKE_PART_MIN_PX join the part they touch most
                    plab = np.zeros(pm.shape, dtype=np.int32)
                    pst = []
                    for st_ in np.unique(ls[pm]).tolist():
                        cc, nn = ndimage.label(pm & (ls == st_), structure=CROSS)
                        for j in range(1, nn + 1):
                            plab[cc == j] = len(pst) + 1
                            pst.append(st_)
                    sz = np.bincount(plab.ravel())[1:]
                    small = [j for j in range(len(sz)) if sz[j] < LAKE_PART_MIN_PX]
                    if not small or len(sz) == 1:
                        break
                    j = min(small, key=lambda j: sz[j])
                    pmj = plab == j + 1
                    rr = ndimage.binary_dilation(pmj, structure=CROSS) & ~pmj & pm
                    nbp = plab[rr]
                    nbp = nbp[nbp > 0]
                    if not len(nbp):
                        break
                    ls[pmj] = pst[np.bincount(nbp).argmax() - 1]
                if len(sz) > 1:
                    state_parts += [(plab == j + 1, int(pst[j])) for j in range(len(sz))]
                    split_states += 1
                    continue
            state_parts.append((pm, sid))
        sub = P[sl]
        for pm, sid in state_parts:
            sub[pm] = nxt
            ptype.append(LAKE)
            group.append(-(20_000_000 + nxt))
            pstate.append(sid)
            nxt += 1
    report["lakes_shared"] = shared
    report["lakes_divided"] = divided
    report["lakes_split_by_state"] = split_states


def _wrap_x_bad(a, b, c, d):
    """A 2x2 block [a b / c d] the game calls an X crossing."""
    return (a == d and b == c and a != b) or len({a, b, c, d}) == 4


def _fix_wrap_x(P, ptype, rounds=4):
    """The game wraps the map: the last column touches the first, and it
    checks those 2x2 blocks for X crossings too ("Map invalid X crossing ...
    coords: 3839,391" -- two edge-band pieces ending on the same row on both
    sides). Each bad block: the column-0 pixel of its lower row takes the
    province above it (else the last column's does), when that is the same
    type and leaves no X inside the map. Returns pixels changed."""
    h, w = P.shape
    changed = 0
    tp = np.asarray(ptype)
    for _ in range(rounds):
        n = 0
        for y in range(h - 1):
            a, b, c, d = int(P[y, w - 1]), int(P[y, 0]), int(P[y + 1, w - 1]), int(P[y + 1, 0])
            if not _wrap_x_bad(a, b, c, d):
                continue
            for (yy, xx, new) in ((y + 1, 0, b), (y + 1, w - 1, a), (y, 0, d), (y, w - 1, c)):
                old = int(P[yy, xx])
                if old == new or tp[old] != tp[new]:
                    continue
                P[yy, xx] = new
                x0 = 0 if xx == 0 else w - 2
                ok = not _wrap_x_bad(int(P[y, w - 1]), int(P[y, 0]), int(P[y + 1, w - 1]), int(P[y + 1, 0]))
                for y0 in (yy - 1, yy):  # the blocks inside the map around the changed pixel
                    if 0 <= y0 < h - 1 and ok:
                        ok = not _wrap_x_bad(int(P[y0, x0]), int(P[y0, x0 + 1]), int(P[y0 + 1, x0]), int(P[y0 + 1, x0 + 1]))
                if ok:
                    n += 1
                    break
                P[yy, xx] = old
        changed += n
        if not n:
            break
    return changed


def _groups_of(provs, border):
    """Provinces grouped by side contact (border: {id: {neighbour: contact}})."""
    left, out = set(provs), []
    while left:
        a = left.pop()
        comp, st = {a}, [a]
        while st:
            q = st.pop()
            for r in border.get(q, {}):
                if r in left:
                    left.discard(r)
                    comp.add(r)
                    st.append(r)
        out.append(comp)
    return out


def _state_exclaves(P, st_of, ptype_new, stl, st_code_of, rounds=12):
    """Every state one piece at province level (user: minimise state
    exclaves / enclaves, unless GADM has them): a state's provinces are
    grouped by adjacency; a group other than its largest moves to the
    neighbouring state of the same country it borders most -- unless the
    States tab's map (`stl`, from the GADM units) already had that part
    apart from the main one (a real exclave, an island) or no state of its
    country borders it. `st_of` changed in place; returns provinces moved."""
    n = len(st_of) - 1
    size = np.bincount(P.ravel(), minlength=n + 1)
    border = {}
    for (a, b), c in _pair_borders(P).items():
        border.setdefault(a, {})[b] = c
        border.setdefault(b, {})[a] = c
    stl_boxes = ndimage.find_objects(np.maximum(stl, 0).astype(np.int32))
    moved = 0
    for _ in range(rounds):
        changed = 0
        members = {}
        for i in range(1, n + 1):
            if st_of[i] > 0 and ptype_new[i] != SEA:
                members.setdefault(int(st_of[i]), []).append(i)
        for s_, provs in members.items():
            if len(provs) < 2:
                continue
            # groups of this state's provinces by adjacency (union-find)
            parent = {i: i for i in provs}

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i
            pset = set(provs)
            for i in provs:
                for j in border.get(i, {}):
                    if j in pset:
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[ri] = rj
            groups = {}
            for i in provs:
                groups.setdefault(find(i), []).append(i)
            if len(groups) < 2:
                continue
            # a lake touching two pieces of this state joins it (user: Varkaus,
            # split by a lake given to its neighbour), if the lake's own state
            # doesn't fall apart without it
            grp_of = {i: find(i) for i in provs}
            lake_done = False
            for i in provs:
                for j in border.get(i, {}):
                    if ptype_new[j] != LAKE or st_of[j] == s_ or j in grp_of:
                        continue
                    t_ = int(st_of[j])
                    if t_ > 0 and st_code_of[t_] != st_code_of[s_]:
                        continue
                    touch = {grp_of[q] for q in border.get(j, {}) if q in grp_of}
                    if len(touch) < 2:
                        continue
                    if t_ > 0:
                        rest = [q for q in members.get(t_, []) if q != j]
                        if len(_groups_of(rest, border)) > len(_groups_of(members.get(t_, []), border)):
                            continue
                    st_of[j] = s_
                    lake_done = True
                    break
                if lake_done:
                    break
            if lake_done:
                changed += 1
                continue
            gl = sorted(groups.values(), key=lambda g: -sum(size[i] for i in g))
            main = gl[0]
            # the States tab's pieces of this state: which ones the main group holds
            sl = stl_boxes[s_ - 1] if s_ - 1 < len(stl_boxes) else None
            main_pieces = set()
            if sl is not None:
                oc, _on = ndimage.label(stl[sl] == s_, structure=CROSS)
                osz = np.bincount(oc.ravel())
                pw = P[sl]
                main_pieces = set(np.unique(oc[np.isin(pw, main) & (oc > 0)]).tolist())
            for g in gl[1:]:
                if sl is not None:
                    own = set(np.unique(oc[np.isin(pw, g) & (oc > 0)]).tolist())
                    if any(osz[o] >= states.EXCLAVE_MIN_PX for o in own - main_pieces):
                        continue  # apart in the source map too, at a real size: an exclave / island
                votes = {}
                for i in g:
                    for j, c in border.get(i, {}).items():
                        t_ = int(st_of[j])
                        if t_ > 0 and t_ != s_ and st_code_of[t_] == st_code_of[s_] and ptype_new[j] != SEA:
                            votes[t_] = votes.get(t_, 0) + c
                if not votes:
                    continue
                target = max(votes, key=votes.get)
                for i in g:
                    st_of[i] = target
                changed += len(g)
        moved += changed
        if not changed:
            break
    return moved


def _continents():
    """GADM tag (ISO3) -> vanilla continent id, from GeoNames' countryInfo.txt."""
    out = {}
    path = core.DATA_DIR / "cities" / "countryInfo.txt"
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) > 8 and f[1]:
                out[f[1]] = CONTINENT_IDS.get(f[8], 0)
    try:
        import palettes
        for t in palettes.MIDDLE_EAST:
            out[t] = MIDDLE_EAST_ID
    except Exception:
        pass
    return out


def render_provinces(countries: Image.Image, debug, state_labels, states_report, rivers=None, terrain_img=None,
                     settings=None, eez_zones=None, density_img=None, lake_ids=None, city_points=None,
                     river_cut_px=None):
    """(provinces image RGB = provinces.bmp, report, definition.csv text);
    report["_states_csv"] = province -> state table (popped by the app).
    city_points: the City tab's cities [(x, y, class, name)] (city provinces).
    river_cut_px: the Rivers tab's `network_cut_px` -- the pixels the branch
      cap deleted to keep each river network small enough for the game to
      load (core._split_big_networks). They are river on the ground, so they
      are put back into the mask every river rule here reads (user): a
      province is still split along a river where one of those 1 px gaps
      falls, and a province border still follows it."""
    s = normalize_settings(settings)
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    view = debug["view"]
    h, w = view["height"], view["width"]
    func = core.PROJECTIONS[view["projection"]]["func"]
    arr = np.asarray(countries.convert("RGB"))
    lakes = debug.get("lakes") if debug.get("lakes") is not None else np.zeros((h, w), bool)
    sea = (arr == np.asarray(core.BACKGROUND_COLOR, dtype=arr.dtype)).all(axis=2)
    if debug.get("hidden_land") is not None:
        sea &= ~debug["hidden_land"]
    sea &= ~lakes
    # an enclosed body of sea water too small to be a sea region (a fjord's
    # inner end cut off at this scale) is a lake: no ship reaches it, and as
    # sea it made a tiny region with its centre outside or a "naval" land region
    sc, nsc = ndimage.label(sea, structure=CROSS)
    if nsc:
        pocket = (np.bincount(sc.ravel(), minlength=nsc + 1) < SEA_AREA_MIN_PX)[sc] & sea
        if pocket.any():
            lakes = lakes | pocket
            sea &= ~pocket
    land = ~sea & ~lakes
    timings = {}
    # The rivers as a mask, the branch cap's gaps healed (river_cut_px): the
    # game wants the gap in rivers.bmp, the map does not want a province to
    # stop being cut because of it.
    river_all = None
    if rivers is not None:
        river_all = np.asarray(rivers) <= 11
        for cy_, cx_ in river_cut_px or ():
            if 0 <= cy_ < h and 0 <= cx_ < w:
                river_all[cy_, cx_] = True

    # Target province count (option): the land sizes are scaled until the
    # land provinces the sizing makes (+ the sea and lakes, whose settings
    # stay as they are) reach it -- a few passes of steps 1-3, count ~ scale^-a.
    S, s_ptype, s_group, sea_group_owner, sea_rep = _make_sea(sea, eez_zones, s)
    sea_centre_split = sea_rep["centre_split"]
    timings["sea"] = round(time.time() - t0, 2)
    land_scale = 1.0
    aim_land = None
    passes = []
    sea_est = lake_est = model1 = None
    want_count = s["target_count"]   # not `target`: that is the province size below
    if want_count:
        sea_est = sea_rep["count"]
        # lakes: one province each; under MIN_PROVINCE_PX they become land
        lab_, nl_ = ndimage.label(lakes, structure=CROSS)
        lake_est = int((np.bincount(lab_.ravel(), minlength=nl_ + 1)[1:] >= MIN_PROVINCE_PX).sum())
        del lab_
        aim_land = max(int(round((want_count - sea_est - lake_est) * (1 + TARGET_BIAS))), 1)
        # first guess: one province per target size (density-scaled per pixel)
        if density_img is not None:
            fpx = np.clip(np.sqrt(s["density_ref"] / np.maximum(density_img[land], 1e-3)),
                          1.0 / max(s["density_dense_factor"], 1), s["density_factor"])
        else:
            fpx = np.ones(int(land.sum()))
        model1 = float((1.0 / (math.sqrt(s["land_min"] * s["land_max"]) * fpx)).sum())
        land_scale = min(max(TARGET_MODEL_RATIO * model1 / aim_land, 0.02), 50.0)
    units_cache = {}
    # projected GADM units per (country, level): kept for the last views, so
    # re-rendering a view with other province settings skips the projection
    vkey = tuple(view.get(k_) for k_ in ("projection", "center", "scale", "win_xmin", "win_ymin", "width", "height"))
    rings_px = _UNIT_PX_CACHE.pop(vkey, None) or {}
    _UNIT_PX_CACHE[vkey] = rings_px
    while len(_UNIT_PX_CACHE) > UNIT_PX_VIEWS:
        _UNIT_PX_CACHE.pop(next(iter(_UNIT_PX_CACHE)))
    for land_pass in range(TARGET_PASSES if aim_land else 1):
        rng = np.random.default_rng(SEED)
        lmin, lmax = s["land_min"] * land_scale, s["land_max"] * land_scale
        rsplit = s["river_split_px"] * land_scale
        # 1. Land: GADM units at the province level, cut by the states.
        info = states.choose_levels(countries, debug, {"target_px": max(4, round(lmax)), "merge_px": max(4, round(lmin))})
        state_level = {c["tag"]: c["level"] for c in states_report.get("countries", [])}
        chosen = {}
        for c in info["countries"]:
            lv = c["level"]
            st_lv = state_level.get(c["tag"], 0)
            if lv < st_lv and any(k["level"] == st_lv for k in c["candidates"]):
                lv = st_lv
            chosen[c["tag"]] = lv
        geoms = info["geoms"]
        units_key = tuple(sorted(chosen.items()))
        if units_key in units_cache:   # same levels as an earlier pass: same raster
            unit, unit_tag = units_cache[units_key]
        else:
            unit = np.zeros((h, w), dtype=np.int32)
            unit_tag = [None]
            for t, lv in chosen.items():
                if (t, lv) not in rings_px:   # projected once per (country, level) across passes
                    if lv == 0:
                        rings_px[(t, lv)] = [states._to_px(view, geoms[t]["rings"], func)]
                    else:
                        idx = states.level_index(lv)
                        rings_px[(t, lv)] = [states._to_px(view, states._record_rings(idx, i), func)
                                             for i in idx["by_tag"][t].tolist()]
                for rp in rings_px[(t, lv)]:
                    if rp:
                        unit_tag.append(t)
                        core._scanline_fill(unit, rp, len(unit_tag) - 1)
            units_cache.clear()
            units_cache[units_key] = (unit, unit_tag)
        timings["units"] = round(time.time() - t0, 2)
        st_tag = [None] + [st["tag"] for st in states_report.get("states", [])]
        # colour group of a tag (merged autonomous territories share the parent's)
        rgbs = debug["country_rgb"]
        same_country = lambda t1, t2: t1 is not None and t2 is not None and rgbs.get(t1) == rgbs.get(t2)
        unit_code = np.array([-1] + [hash(rgbs.get(t)) & 0x7FFFFFFF for t in unit_tag[1:]], dtype=np.int64)
        st_code = np.array([-1] + [hash(rgbs.get(t)) & 0x7FFFFFFF for t in st_tag[1:]], dtype=np.int64)
        stl = np.where(land, state_labels, 0).astype(np.int64)
        valid = land & (stl > 0) & (unit > 0) & (unit_code[unit] == st_code[stl])
        base = np.where(valid, stl * (len(unit_tag) + 1) + unit, 0).astype(np.int64)
        # pixels of a state without a unit of its own: nearest valid pixel in the same state
        reassigned = 0
        boxes = ndimage.find_objects(stl.astype(np.int32))
        for sid, sl in enumerate(boxes, start=1):
            if sl is None:
                continue
            ms = stl[sl] == sid
            bad = ms & ~valid[sl]
            if not bad.any():
                continue
            good = ms & valid[sl]
            sub = base[sl]
            if not good.any():
                sub[bad] = sid * (len(unit_tag) + 1)  # the whole state one unit
            else:
                _d, (iy, ix) = ndimage.distance_transform_edt(~good, return_indices=True)
                sub[bad] = sub[iy[bad], ix[bad]]
            reassigned += int(bad.sum())
        # land outside any state (hidden countries): one unit per land area, split later
        orphan = land & (stl == 0)
        if orphan.any():
            comp, _n = ndimage.label(orphan, structure=CROSS)
            base[orphan] = -comp[orphan]
        P, n = _relabel(np.where(land, base, 0))
        # province tables (index = id)
        first = _majority(P, np.maximum(stl, 0), n)[1:]
        ptype = [LAND] + [LAND] * n
        group = [0] + [int(v) if v > 0 else -1 - i for i, v in enumerate(first)]  # orphan land: its own group
        timings["base"] = round(time.time() - t0, 2)

        # 2. Rivers: borders within the band follow them, inside each state.
        river_moved = 0
        if rivers is not None:
            river = river_all & land & (stl > 0)
            if river.any():
                band = (ndimage.distance_transform_edt(~river) <= states.RIVER_SNAP_PX) & ~river & land & (stl > 0)
                outside = np.bincount(P[land & ~band & ~river].ravel(), minlength=int(P.max()) + 1)
                todo = band & (outside[P] > 0)
                before = P.copy()
                gid = np.where(land, stl, -1).astype(np.int32)
                P = states._nearest_from_outside(P, gid, todo, river)
                ry, rx = np.nonzero(river)
                nb = []
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    yy, xx = np.clip(ry + dy, 0, h - 1), np.clip(rx + dx, 0, w - 1)
                    nb.append(np.where((stl[yy, xx] == stl[ry, rx]) & ~river[yy, xx], P[yy, xx], 0))
                nb = np.stack(nb, 1)
                own = before[ry, rx]
                votes = ((nb[:, :, None] == nb[:, None, :]) & (nb[:, None, :] > 0)).sum(2) * (nb > 0)
                mode = nb[np.arange(len(ry)), votes.argmax(1)]
                has_own = (nb == own[:, None]).any(1)
                P[ry, rx] = np.where(has_own | (mode == 0), own, mode)
                river_moved = int((P != before).sum())
        timings["rivers"] = round(time.time() - t0, 2)

        # 2b. City provinces (option): the City tab's largest cities, on the
        #     Terrain tab's urban areas; never merged away below.
        city_mask, city_made = None, []
        if s["city_provinces"] and terrain_img is not None and city_points:
            city_mask, city_made = _city_provinces(P, ptype, group, land, stl, terrain_img, city_points,
                                                   s["city_classes"], s["city_min_px"])
        timings["cities"] = round(time.time() - t0, 2)

        # 3. Land sizes (density-scaled: sparse areas get bigger provinces),
        #    then rivers cutting a big province in two split it.
        pstate = [0] * len(ptype)          # lake provinces' states (land: majority of the state labels)
        dens = density_img if density_img is not None else None
        st_tag_of = [None] + [st["tag"] for st in states_report.get("states", [])]
        country_of = lambda v: (rgbs.get(st_tag_of[group[v]]) if 0 < group[v] < len(st_tag_of) else None)

        def country_mult(n_):
            m_ = np.ones(n_ + 1)
            if s["country_factors"]:
                for k_ in range(1, min(n_ + 1, len(group))):
                    g_ = group[k_]
                    if 0 < g_ < len(st_tag_of):
                        m_[k_] = s["country_factors"].get(st_tag_of[g_], 1.0)
            return m_
        fac = lambda: _province_factor(P, np.where(land, dens, 0) if dens is not None else None,
                                       s["density_ref"], s["density_factor"], s["density_dense_factor"], country_mult)
        # sizes aim at the geometric mean of min and max, scaled by density /
        # country (a country's GADM level put it anywhere in [min, max] before:
        # North Algeria near the min, Belgium near the max)
        target = math.sqrt(lmin * lmax)
        merge_px = max(lmin, MERGE_TOWARD * target)
        _components_fix(P, ptype, group)
        f = fac()
        # the next finer GADM level of the countries with provinces to split:
        # its borders are where the organic split prefers to cut
        size0 = np.bincount(P.ravel(), minlength=len(ptype))
        big_tags = set()
        for k in range(1, len(size0)):
            if k < len(ptype) and ptype[k] == LAND and size0[k] > lmax * f[min(k, len(f) - 1)]:
                g_ = group[k]
                if 0 < g_ < len(st_tag):
                    big_tags.add(st_tag[g_])
        finer = np.zeros((h, w), dtype=np.int32)
        n_f = 0
        for t_ in sorted(big_tags):
            lv = chosen.get(t_, 0) + 1
            idx = states.level_index(lv) if lv <= states.MAX_LEVEL else None
            while idx is not None and t_ not in idx["by_tag"] and lv < states.MAX_LEVEL:
                lv += 1
                idx = states.level_index(lv)
            if idx is None or t_ not in idx["by_tag"]:
                continue
            if (t_, lv) not in rings_px:
                rings_px[(t_, lv)] = [states._to_px(view, states._record_rings(idx, i), func)
                                      for i in idx["by_tag"][t_].tolist()]
            for rp in rings_px[(t_, lv)]:
                if rp:
                    n_f += 1
                    core._scanline_fill(finer, rp, n_f)
        split_cost_img = split_cost(land, river_all, terrain_img, finer if n_f else None)
        f_split = f
        if city_mask is not None:  # a city isn't cut smaller for being dense: plain max (Paris came in 7 pieces)
            f_split = f.copy()
            prot_ = [k_ for k_ in _protected(P, city_mask) if k_ < len(f_split)]
            f_split[prot_] = np.maximum(f_split[prot_], 1.0)
        split_land = _split_big(P, ptype, group, LAND, lmax * f_split, target * f_split, rng, cost=split_cost_img)
        _components_fix(P, ptype, group)
        f = fac()
        merged_land = _merge_small(P, np.array(ptype), np.array(group), merge_px * f, LAND,
                                   protect=_protected(P, city_mask) if city_mask is not None else frozenset())
        river_splits = 0
        river_bits = [0]
        if rivers is not None and rsplit > 0:
            river_land = river_all & land
            for _ in range(3):
                sp = _river_splits(P, ptype, group, river_land, rsplit, river_bits, country_of)
                river_splits += sp
                if not sp:
                    break
        thin_moved = _clean_thin(P, ptype, group, country_of)
        _components_fix(P, ptype, group)
        pstate += [0] * (len(ptype) - len(pstate))
        timings["land"] = round(time.time() - t0, 2)
        n_land_now = int(len(np.unique(P[land & (P > 0)])))
        passes.append((round(land_scale, 4), n_land_now))
        if not aim_land or abs(n_land_now - aim_land) <= TARGET_TOLERANCE * aim_land:
            break
        # out of reach: provinces are parts of the states (at least one each,
        # + the city provinces), and under MIN_PROVINCE_PX they're merged
        if len(passes) >= 2:
            (k1, n1), (k2, n2) = passes[-2:]
            if abs(k2 / k1 - 1) > 0.2 and abs(n2 - n1) <= TARGET_TOLERANCE * max(n1, 1):
                break
        alpha = 1.0
        if len(passes) >= 2:
            (k1, n1), (k2, n2) = passes[-2:]
            if n1 > 0 and n2 > 0 and n1 != n2 and k1 != k2:
                alpha = min(max(-math.log(n2 / n1) / math.log(k2 / k1), 0.3), 3.0)
        if land_pass + 1 < TARGET_PASSES:
            land_scale = min(max(land_scale * (n_land_now / aim_land) ** (1.0 / alpha), 0.02), 50.0)

    # 4. Lakes (owners: user's rules) and the sea: split within each EEZ,
    #    cells growing with the distance from land.
    lake_rep = {}
    lakes = lakes.copy()
    land = land.copy()
    lake_ids = lake_ids.copy() if lake_ids is not None else None
    lake_rep["joined_px"] = _join_lake_fragments(lakes, land, P, lake_ids)
    lake_rep["bridged_px"] = _bridge_lakes(lakes, land, P)
    stl = np.where(land, stl, 0)
    # each GADM unit's state (the majority of its land pixels): lakes are
    # divided between states along the unit borders that cross them
    unit_state = _majority(np.where(land, unit, 0).astype(np.int32), np.maximum(stl, 0).astype(np.int64),
                           int(unit.max()))
    _lakes(P, lakes, land, stl, st_code, unit, unit_code, ptype, group, pstate, lake_rep, unit_state)
    # the sea (made before the land, see above): its ids after the lakes'
    nxt = len(ptype)
    m_ = S > 0
    P[m_] = S[m_] + (nxt - 1)
    ptype += s_ptype[1:]
    group += s_group[1:]
    pstate += [0] * (len(s_ptype) - 1)
    n_lake = int(sum(1 for t_ in ptype if t_ == LAKE))
    timings["sea_joined"] = round(time.time() - t0, 2)

    # 5. Topology, until stable; then any X-crossing left is forced out.
    topo = {"x_fixed": 0, "x_type_changes": 0, "double_fixed": 0, "parts_merged": 0, "parts_split": 0, "rounds": 0}
    topo["tiny_merged"] = topo["tiny_kept"] = 0
    topo["thin_moved"] = thin_moved
    changed = None  # first round: every province; then only those the last round changed
    for rnd in range(TOPOLOGY_ROUNDS):
        thin_ids = set()
        th = _clean_thin(P, ptype, group, country_of, touched=thin_ids)
        topo["thin_moved"] += th
        # only provinces whose pixels changed can be in pieces (a full check
        # after every thin move cost ~0.5 s a round)
        pm, pc = _components_fix(P, ptype, group, only=None if changed is None else changed | thin_ids)
        tm, tk = _merge_tiny(P, ptype, group, country_of)
        topo["tiny_merged"] += tm
        pm += tm
        xf, xt, _x_left, t1 = _fix_x(P, ptype, group)
        df, _dbl_left, t2 = _fix_double(P, ptype, group, cost=split_cost_img)
        sc = _split_sea_centres(P, ptype, group)[0] if SEA_CENTRE_IN_LOOP else 0
        sea_centre_split += sc
        pc += sc
        changed = t1 | t2
        topo["parts_merged"] += pm
        topo["parts_split"] += pc
        topo["x_fixed"] += xf
        topo["x_type_changes"] += xt
        topo["double_fixed"] += df
        topo["rounds"] = rnd + 1
        topo.setdefault("history", []).append((pm, pc, xf, df, th))
        if not (pm or pc or xf or df or th):
            break
        # a case or two nudged back and forth: the endgame below settles those
        hist = topo["history"]
        last_ = hist[-HANDOVER_ROUNDS:]
        if len(hist) >= HANDOVER_ROUNDS and (all(sum(r_) <= HANDOVER_CHANGES for r_ in last_)
                                             or all(r_ == last_[0] for r_ in last_)):
            break
    timings["topology_loop"] = round(time.time() - t0, 2)
    fm, fp = _fix_x_force(P, ptype, group)
    topo["x_forced"], topo["x_blocks_painted"] = fm, fp
    topo["double_forced"] = 0
    # endgame: until nothing is left (forcing may leave tiny pieces, merging
    # them may bring a contact back)
    for _end in range(3):
        topo["double_forced"] += _force_double(P, ptype, group, rounds=3)[0]
        th = _clean_thin(P, ptype, group, country_of)
        topo["thin_moved"] += th
        pm, pc = _components_fix(P, ptype, group)
        tm, tk = _merge_tiny(P, ptype, group, country_of)
        sc, _left = _split_sea_centres(P, ptype, group)
        sea_centre_split += sc
        fm, fp = _fix_x_force(P, ptype, group)
        topo["parts_merged"] += pm + tm
        topo["parts_split"] += pc
        topo["tiny_merged"] += tm
        topo["x_forced"] += fm
        topo["x_blocks_painted"] += fp
        topo.setdefault("endgame_history", []).append((int(topo["double_forced"]), th, pm, pc, tm, sc, fm, fp,
                                                       round(time.time() - t0, 1)))
        if not (pm or pc or tm or fm or fp or th or sc) and not _contact_segments(P):
            break
    timings["endgame"] = round(time.time() - t0, 2)
    topo["tiny_kept"] = tk  # tiny provinces with no neighbour of their own country (microstates, islets)
    # last word: only fixes nothing afterwards can undo (the endgame's splits /
    # tiny merges rebuilt a 2-sea-cell contact round after round)
    for _last in range(6):
        sc, _left = _split_sea_centres(P, ptype, group)
        sea_centre_split += sc
        segs = _contact_segments(P)
        if not segs and not sc:
            break
        size_ = np.bincount(P.ravel())
        for (a, b) in segs:
            if ptype[a] == SEA and ptype[b] == SEA and _brute_pair(P, a, b):
                topo["double_forced"] += 1
            elif ptype[a] != LAND and ptype[a] == ptype[b]:
                small, big = (a, b) if size_[a] < size_[b] else (b, a)
                if size_[small] and size_[big]:
                    P[P == small] = big
                    size_[big] += size_[small]
                    size_[small] = 0
            elif (ptype[a] == LAND) != (ptype[b] == LAND):
                _resolve_lake_pair(P, a, b, ptype)
        for (a, b) in _contact_segments(P):
            if _brute_pair(P, a, b):
                topo["double_forced"] += 1
        fm, fp = _fix_x_force(P, ptype, group)
        topo["x_forced"] += fm
        topo["x_blocks_painted"] += fp
    timings["last_pass"] = round(time.time() - t0, 2)
    pstate += [0] * (len(ptype) - len(pstate))
    # target count: the topology adds provinces (sea +5-7%, land +2-5% away
    # from Europe): merge small land pairs down to the target
    trimmed = 0
    if want_count:
        total_now = int(len(np.unique(P[P > 0])))
        if total_now > want_count:
            trimmed = _trim_to_count(P, ptype, group, total_now - want_count,
                                     _protected(P, city_mask) if city_mask is not None else frozenset(),
                                     river_all)
    # the game's border limit (hardcoded): merge the smallest land pairs
    # until under it, else province borders stop being drawn on the map
    border_merges, borders_before, _b = _cap_borders(
        P, ptype, group,
        _protected(P, city_mask) if city_mask is not None else frozenset(),
        river_all)
    timings["trim"] = round(time.time() - t0, 2)
    pstate += [0] * (len(ptype) - len(pstate))
    topo["split_tall"] = _split_tall(P, ptype, group, pstate)
    _components_fix(P, ptype, group)
    pstate += [0] * (len(ptype) - len(pstate))
    # last sweep of the sea centre rule (the last pass's water merges can
    # leave one), with whatever it breaks mended pixel by pixel
    for _sweep in range(3):
        sc, _left = _split_sea_centres(P, ptype, group, strict=True)
        sea_centre_split += sc
        segs = _contact_segments(P)
        for (a, b) in segs:
            if _brute_pair(P, a, b):
                topo["double_forced"] += 1
        fm, fp = _fix_x_force(P, ptype, group)
        topo["x_forced"] += fm
        topo["x_blocks_painted"] += fp
        if not sc and not segs and not (fm or fp):
            break
    timings["sweep"] = round(time.time() - t0, 2)
    topo["wrap_x_fixed"] = _fix_wrap_x(P, ptype)
    topo["x_left"] = int(len(_x_blocks(P)[0]))
    topo["double_left"] = len(_contact_segments(P))
    topo["sea_centre_split"] = sea_centre_split
    timings["topology"] = round(time.time() - t0, 2)

    # 6. Dense ids, tables, colours, definition.csv.
    # Type per province: the majority of its pixels' land / sea / lake (X
    # fixes may have moved a single pixel across the coast).
    P, n, old = _relabel(P, return_old=True)
    typ_img = np.where(sea, SEA, np.where(lakes, LAKE, LAND))
    ptype_new = _majority(P, typ_img, n)
    size = np.bincount(P.ravel(), minlength=n + 1)
    # state per province: land = the majority of the state labels (so the
    # states are exactly their provinces -- where an X fix moved a pixel
    # across a state border, the state follows); lakes = their owner.
    st_of = _majority(P, np.maximum(stl, 0), n)
    for i in range(1, n + 1):
        if ptype_new[i] == LAKE:
            st_of[i] = pstate[old[i]] if old[i] < len(pstate) else 0
        elif ptype_new[i] == SEA:
            st_of[i] = 0
    st_code_of = np.array([-1] + [hash(rgbs.get(st["tag"])) & 0x7FFFFFFF for st in states_report.get("states", [])],
                          dtype=np.int64)
    exclaves_moved = _state_exclaves(P, st_of, ptype_new, stl, st_code_of)
    m_land = (P > 0) & (stl > 0)
    if m_land.any():
        u = np.unique(P[m_land].astype(np.int64) * (int(stl.max()) + 1) + stl[m_land])
        report_spans = int((np.bincount(u // (int(stl.max()) + 1)) > 1).sum())
    else:
        report_spans = 0
    # coastal: land touching sea / sea touching land (side contact).
    # **The map wraps**: the game builds its own coastality from the bitmap,
    # last column against the first, and prefers it over this column
    # ("Bitmap and province definition disagree on whether or not province N
    # is coastal"). A land province it calls coastal with no port building
    # then crashes the game on start ("is setup as coastal but has no port
    # building in the nudger"; the user's France map, edge bands off: 96 of
    # them). So the wrap pair counts here exactly as the game counts it --
    # and `hoi4mod` writes those provinces' ports from the same contacts.
    coastal = np.zeros(n + 1, dtype=bool)
    pairs = [(P[: h - dy, : w - dx], P[dy:, dx:]) for dy, dx in ((0, 1), (1, 0))]
    pairs.append((P[:, w - 1], P[:, 0]))  # the map's wrap
    for a_, b_ in pairs:
        ta, tb = ptype_new[a_], ptype_new[b_]
        m = (a_ != b_) & (((ta == LAND) & (tb == SEA)) | ((ta == SEA) & (tb == LAND)))
        coastal[a_[m]] = True
        coastal[b_[m]] = True
    # terrain: most common Terrain-tab type of a land province
    import terrain as terrain_mod
    if terrain_img is not None:
        idx_type = {i: t_ for i, _n, t_ in terrain_mod.TERRAIN_CLASSES}
        type_list = sorted(set(idx_type.values()))
        lut = np.full(256, -1, dtype=np.int64)
        for i, t_ in idx_type.items():
            lut[i] = type_list.index(t_)
        tt = lut[np.asarray(terrain_img)]
        mm = (tt >= 0) & (ptype_new[P] == LAND)
        tc = np.zeros((n + 1, len(type_list)), dtype=np.int64)
        np.add.at(tc, (P[mm], tt[mm]), 1)
        if "ocean" in type_list:
            tc[:, type_list.index("ocean")] = 0
        best = tc.argmax(1)
        terr = [type_list[b_] if tc[i].sum() else "plains" for i, b_ in enumerate(best)]
    else:
        terr = ["plains"] * (n + 1)
    # city provinces: urban terrain; named after their largest city
    city_name = {}
    if city_mask is not None:
        city_ids = _protected(P, city_mask, n)
        for pid_, nm_, _cls, _px, cy_, cx_ in city_made:
            k_ = int(P[cy_, cx_])
            if k_ in city_ids and k_ not in city_name:
                city_name[k_] = nm_
        for k_ in city_ids:
            if ptype_new[k_] == LAND:
                terr[k_] = "urban"
                city_name.setdefault(k_, "")
    cont = _continents()
    st_rows = {st["id"]: st for st in states_report.get("states", [])}
    st_tags = [None] + [st["tag"] for st in states_report.get("states", [])]
    # Colours (user): land = variations of its state's colour; sea = the
    # base sea colour 191F61 tinted by its EEZ country's colour; lakes = a
    # lake blue tinted by its state's colour; all unique.
    def hex_rgb(hx):
        return tuple(int(hx[i:i + 2], 16) for i in (1, 3, 5))
    rgbs = debug["country_rgb"]
    base = [(0, 0, 0)] * (n + 1)
    spread = [(0.02, 0.12, 0.15)] * (n + 1)
    for i in range(1, n + 1):
        tp = int(ptype_new[i])
        st = st_rows.get(int(st_of[i]))
        if tp == LAND:
            base[i] = hex_rgb(st["color"]) if st else (128, 128, 128)
        elif tp == SEA:
            o_ = sea_group_owner.get(group[old[i]]) if old[i] < len(group) else None
            c_ = rgbs.get(o_) if o_ else None
            base[i] = tuple(int(round(SEA_BASE_RGB[j] * (1 - TINT) + c_[j] * TINT)) for j in range(3)) if c_ else SEA_BASE_RGB
            spread[i] = (0.015, 0.08, 0.07)
        else:
            c_ = hex_rgb(st["color"]) if st else None
            base[i] = tuple(int(round(LAKE_BASE_RGB[j] * (1 - TINT) + c_[j] * TINT)) for j in range(3)) if c_ else LAKE_BASE_RGB
            spread[i] = (0.015, 0.08, 0.08)
    rgb = _colours(P, n, base, spread, rng)
    img = Image.fromarray(rgb[P], mode="RGB")
    lines = ["0;0;0;0;land;false;unknown;0"]
    counts = {"land": 0, "sea": 0, "lake": 0}
    for i in range(1, n + 1):
        tp = TYPE_NAMES[int(ptype_new[i])]
        counts[tp] += 1
        if tp == "land":
            tag = st_tags[st_of[i]] if st_of[i] < len(st_tags) else None
            tr, co = terr[i], cont.get(tag, 0) or 1
            cst = "true" if coastal[i] else "false"
        elif tp == "sea":
            tr, co, cst = "ocean", 0, "true" if coastal[i] else "false"
        else:
            tr, co, cst = "lakes", 0, "false"
        lines.append(f"{i};{rgb[i, 0]};{rgb[i, 1]};{rgb[i, 2]};{tp};{cst};{tr};{co}")
    definition = "\n".join(lines) + "\n"
    # Province -> state, for the game's state files (lakes included).
    # sea provinces: their sea area (= their strategic region to be, see
    # _sea_areas) and its EEZ owner ("" outside every EEZ)
    srows = ["province;type;state;state_gadm;state_name;country;city;sea_area;sea_zone"]
    for i in range(1, n + 1):
        sid = int(st_of[i])
        st = st_rows.get(sid)
        area, zone_ = "", ""
        if ptype_new[i] == SEA and old[i] < len(group) and group[old[i]] <= -10_000_000:
            area = str(-group[old[i]] - 10_000_000)
            zone_ = sea_group_owner.get(group[old[i]]) or ""
        srows.append(f"{i};{TYPE_NAMES[int(ptype_new[i])]};{sid if st else ''};{st['gid'] if st else ''};"
                     f"{(st['name'] if st else '').replace(';', ',')};{st['tag'] if st else ''};"
                     f"{(city_name.get(i) or '').replace(';', ',')};{area};{zone_}")
    states_csv = "\n".join(srows) + "\n"
    # States as made of the provinces (+ owned lakes), for the downloads and
    # the border overlay.
    state_px = st_of[P].astype(np.int32)
    st_rgb = np.zeros((len(st_tags), 3), dtype=np.uint8)
    for sid, st in st_rows.items():
        if sid < len(st_tags):
            st_rgb[sid] = hex_rgb(st["color"])
    states_img = np.where((state_px > 0)[..., None], st_rgb[np.clip(state_px, 0, len(st_tags) - 1)],
                          np.asarray(core.BACKGROUND_COLOR, dtype=np.uint8))
    code_of = np.array([-1] + [hash(rgbs.get(tg)) & 0x7FFFFFFF for tg in st_tags[1:]], dtype=np.int64)
    country_px = np.where(state_px > 0, code_of[np.clip(state_px, 0, len(st_tags) - 1)], -1)

    def stats(tp):
        v = size[1:][ptype_new[1:] == tp]
        if not len(v):
            return None
        return {"count": int(len(v)), "mean_px": round(float(v.mean())), "median_px": round(float(np.median(v))),
                "min_px": int(v.min()), "max_px": int(v.max())}
    terr_counts = {}
    for i in range(1, n + 1):
        if ptype_new[i] == LAND:
            terr_counts[terr[i]] = terr_counts.get(terr[i], 0) + 1
    land_sizes = size[1:][ptype_new[1:] == LAND]
    sea_sizes = size[1:][ptype_new[1:] == SEA]
    report = {
        "settings": s, "count": n, "counts": counts,
        "land": stats(LAND), "sea": stats(SEA), "lake": stats(LAKE),
        "land_below_min": int((land_sizes < lmin).sum()), "land_above_max": int((land_sizes > lmax).sum()),
        "sea_below_min": int((sea_sizes < s["sea_min"]).sum()), "sea_above_max": int((sea_sizes > s["sea_max"]).sum()),
        "levels": {t: lv for t, lv in sorted(chosen.items())},
        "reassigned_px": reassigned, "river_moved_px": river_moved,
        "land_split": split_land, "land_merged": merged_land, "sea_split": sea_rep["split"], "sea_merged": sea_rep["merged"],
        "sea_areas": int(sea_rep["areas"]), "lakes": int(n_lake), "river_splits": river_splits,
        "river_bits_given": river_bits[0], "lakes_bridged_px": lake_rep.get("bridged_px", 0),
        "lakes_joined_px": lake_rep.get("joined_px", 0), "lakes_split_by_state": lake_rep.get("lakes_split_by_state", 0),
        "state_exclaves_moved": exclaves_moved,
        "tiny_left": int(((size[1:] < MIN_PROVINCE_PX) & (size[1:] > 0)).sum()),
        "lakes_shared": lake_rep.get("lakes_shared", 0), "lakes_divided": lake_rep.get("lakes_divided", 0),
        "density": "GHS-POP 2025" if density_img is not None else None,
        "split_finer_units": int(n_f), "split_finer_tags": sorted(big_tags),
        "eez": eez_zones is not None,
        "topology": topo, "provinces_spanning_states": report_spans,
        "states": int(len(np.unique(st_of[1:][ptype_new[1:] == LAND]))),
        "coastal": int(coastal[1:].sum()), "terrain": dict(sorted(terr_counts.items(), key=lambda kv: -kv[1])),
        "terrain_source": "Terrain tab" if terrain_img is not None else "none (Terrain tab unavailable): plains",
        "timings": timings,
    }
    per = {}
    for i in range(1, n + 1):
        if ptype_new[i] != LAND:
            continue
        tag = st_tags[st_of[i]] if 0 < st_of[i] < len(st_tags) else None
        if tag is None:
            continue
        e = per.setdefault(tag, [0, 0])
        e[0] += 1
        e[1] += int(size[i])
    report["countries"] = sorted(
        [{"tag": tg, "name": geoms[tg]["name"] if tg in geoms else tg,
          "color": core.rgb_to_hex(rgbs[tg]) if tg in rgbs else "#888888",
          "provinces": c_, "mean_px": round(px_ / c_), "factor": s["country_factors"].get(tg, 1.0)}
         for tg, (c_, px_) in per.items()], key=lambda c: c["name"])
    report["settings"] = {k: v for k, v in s.items()}
    report["target_px"] = round(target)
    target_note = None
    if want_count and abs(n_land_now - aim_land) > 0.05 * aim_land:
        n_states_ = int(len(np.unique(stl[stl > 0])))
        if n_land_now > aim_land:
            target_note = (f"too few: land provinces are parts of the states ({n_states_:,} in view, one province "
                           f"at least each{', + the city provinces' if city_made else ''}) -- fewer, larger states "
                           f"(States tab) allow fewer provinces")
        else:
            target_note = f"too many: provinces under {MIN_PROVINCE_PX} px are merged into their neighbours"
    report["target_count"] = ({"aim": want_count, "land_scale": round(land_scale, 3), "passes": passes,
                               "land_aimed": aim_land, "sea_estimate": sea_est, "lake_estimate": lake_est, "model1": round(model1), "trim_merges": trimmed,
                               "land_after_sizing": n_land_now,
                               "note": target_note,
                               "land_min": round(lmin), "land_max": round(lmax)}
                              if want_count else None)
    borders_now = border_count(P)
    report["borders"] = {
        "count": borders_now, "limit": GAME_MAX_BORDERS, "before_cap": borders_before, "merges": border_merges,
        "max_provinces": MAX_PROVINCES,
        "note": (f"over the game's limit of {GAME_MAX_BORDERS:,} province borders: it stops drawing them in "
                 "large parts of the map -- aim at fewer provinces (target number, or larger sizes)"
                 if borders_now > GAME_MAX_BORDERS else
                 f"{border_merges:,} pairs of small land provinces merged to stay under the game's limit of "
                 f"{GAME_MAX_BORDERS:,} borders ({borders_before:,} before)" if border_merges else None)}
    report["sea_centre_outside"] = len(_sea_centres_outside(P, ptype_new))
    report["sea_centre_split"] = sea_centre_split
    if s["city_provinces"]:
        report["city_provinces"] = {
            "count": len(city_name), "named": sum(1 for v in city_name.values() if v), "cities_used": len(city_made),
            "available": terrain_img is not None and bool(city_points),
            "names": sorted({nm_ for nm_ in city_name.values() if nm_})[:400],
            "px": int(sum(size[k_] for k_ in city_name))}
    else:
        report["city_provinces"] = None
    report["seconds"] = round(time.time() - t0, 2)
    _x_blocks.cache_clear()
    _contact_segments.cache_clear()
    report["_states_csv"] = states_csv
    report["_extras"] = {"state_px": state_px, "country_px": country_px,
                         "states_img": Image.fromarray(states_img.astype(np.uint8), mode="RGB")}
    return img, report, definition
