"""Strategic regions tab: HOI4 map/strategicregions/*.txt from the Provinces
tab's output (provinces.bmp + definition.csv + province_states.csv) -- the
boundaries only, no weather (a placeholder weather block is written).

Rules (user + game):
  * land regions are made of whole states (the game: every province of a
    state in one strategic region, else a map error); sea regions of sea
    provinces;
  * a sea region may hold islands, as long as the whole island (every
    state on it) is in it: land masses -- states joined by land or lakes --
    under `island_px` go to the sea region they border most;
  * every province belongs to exactly one region (the game crashes on a
    province without one); no region is empty;
  * every region's centre lies within itself (the forum-reported crash,
    "division by zero ... only when a centre is outside the region, but not
    every time"). Centre = the mean of the region's pixels, as the bug
    report has it; checked strictly: the 2x2 pixels around it (floor / ceil
    of x and y, so the rounding or a flipped y doesn't matter) all in the
    region, and the mean of its provinces' centres (the other plausible
    definition) inside too;
  * each region is one connected group (land regions over land / lakes,
    sea regions over the sea, an island through the sea it touches).
Land without a state (the edge bands' unowned strip, hidden countries) is
never given to the sea: it forms land regions of its own.

Sizes (px): `land_min` / `land_max`, `sea_min` / `sea_max` (defaults =
vanilla's 5th / 95th percentiles). Method: agglomerative, like the
provinces' merges. Units = states, stateless land groups, sea provinces.
Land: within each land mass, the smallest region under the minimum or
under `MERGE_SHARE` x sqrt(min x max) merges into the neighbour that keeps
the merged centre inside, preferring a merged size <= `FIT_FACTOR` x
sqrt(min x max), then <= the maximum, a
neighbour of the same country (`by_country`), one also under the threshold,
then the longest border relative to the neighbour's size. Sea likewise per
connected sea, up to `MERGE_SHARE` x the maximum with provinces near a
coast weighing up to `COAST_WEIGHT` x their size (vanilla's coastal and
enclosed seas are small regions, its oceans big ones), then the islands
are attached. Regions whose centre is still
outside merge with a neighbour that fixes it, or give a unit to a neighbour.
"""
import colorsys
import heapq
import io
import math
import re
import time
import zipfile
import zlib

import numpy as np
from PIL import Image
from scipy import ndimage

LAND, SEA, LAKE = 0, 1, 2
TYPES = {"land": LAND, "sea": SEA, "lake": LAKE}

# Vanilla (map/strategicregions, measured with _dev/sr_vanilla.py), px:
# land regions p5 3,821 / median 13,403 / p95 52,920 (4 states); sea
# regions (islands included) p5 7,872 / median 67,055 / p95 189,878 --
# the defaults are those minima / maxima, rounded (user: "same approximate
# minimum & maximum sizes as vanilla"). Islands in sea regions up to 3,492
# px (most under 900); islands with land regions of their own from ~500 px.
SR_DEFAULTS = {"land_min": 4000, "land_max": 50000, "sea_min": 8000, "sea_max": 190000,
               "island_px": 2500, "by_country": 1}
SR_LIMITS = {"land_min": (100, 2_000_000), "land_max": (200, 5_000_000), "sea_min": (100, 2_000_000),
             "sea_max": (200, 10_000_000), "island_px": (0, 2_000_000)}
SEA_AREA_TINY_PX = 500  # a sea area (region) smaller than this joins a neighbour (topology slivers)
MERGE_SHARE = 0.6     # land: merged below this x sqrt(min x max); sea: below this x max (coast-weighted)
FIT_FACTOR = 1.2      # land merges staying under this x sqrt(min x max) come first
COAST_WEIGHT = 3.0    # a sea province at the coast counts this x its size
FIX_ROUNDS = 6


def normalize_settings(raw):
    s = dict(SR_DEFAULTS)
    for k, v in (raw or {}).items():
        if k in s and v not in (None, ""):
            try:
                s[k] = int(float(v))
            except (TypeError, ValueError):
                raise ValueError(f"'{k}' must be a number")
    for k, (lo, hi) in SR_LIMITS.items():
        if not lo <= s[k] <= hi:
            raise ValueError(f"'{k}' must be between {lo:,} and {hi:,}")
    s["by_country"] = 1 if s["by_country"] else 0
    for k in ("land", "sea"):
        if s[f"{k}_min"] >= s[f"{k}_max"]:
            raise ValueError(f"The {k} regions' minimum must be below their maximum")
    return s


# ---------------------------------------------------------------- inputs
def parse_inputs(pimg: Image.Image, definition: str, states_csv: str | None):
    """(P id image, ptype per id (-1 = none), pstate per id (0 = none),
    states {id: (name, country)})."""
    a = np.asarray(pimg.convert("RGB")).astype(np.int64)
    code = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
    rows = []
    for line in definition.splitlines():
        f = line.split(";")
        if len(f) < 5:
            continue
        try:
            rows.append((int(f[0]), (int(f[1]) << 16) | (int(f[2]) << 8) | int(f[3]), f[4].strip().lower()))
        except ValueError:
            continue
    n = max(i for i, _c, _t in rows)
    ptype = np.full(n + 1, -1, dtype=np.int8)
    c2id = {}
    for i, c, t in rows:
        if i == 0:
            continue
        ptype[i] = TYPES.get(t, LAND)
        c2id[c] = i
    u, inv = np.unique(code, return_inverse=True)
    ids = np.array([c2id.get(int(c), 0) for c in u.tolist()], dtype=np.int32)
    P = ids[inv].reshape(code.shape)
    if (P == 0).any():
        raise ValueError(f"{int((P == 0).sum()):,} pixels of provinces.bmp have no definition.csv row")
    pstate = np.zeros(n + 1, dtype=np.int64)
    states = {}
    if states_csv:
        lines = states_csv.splitlines()
        head = lines[0].split(";")
        col = {k: head.index(k) for k in ("province", "state", "state_name", "country") if k in head}
        for line in lines[1:]:
            f = line.split(";")
            if len(f) < len(head):
                continue
            try:
                p = int(f[col["province"]])
            except ValueError:
                continue
            st = f[col["state"]].strip()
            if st and 0 < p <= n:
                pstate[p] = int(st)
                states.setdefault(int(st), (f[col["state_name"]] if "state_name" in col else "",
                                            f[col["country"]] if "country" in col else ""))
    return P, ptype, pstate, states


def _pairs(P):
    """Province pairs touching side by side -> (a, b, contact length), a < b."""
    n1 = int(P.max()) + 1
    keys = []
    for x, y in ((P[:, :-1], P[:, 1:]), (P[:-1, :], P[1:, :])):
        m = x != y
        lo = np.minimum(x[m], y[m]).astype(np.int64)
        hi = np.maximum(x[m], y[m]).astype(np.int64)
        keys.append(lo * n1 + hi)
    k, c = np.unique(np.concatenate(keys), return_counts=True)
    return k // n1, k % n1, c


# ---------------------------------------------------------------- regions
class _Regions:
    """Regions as groups of units, with pixel sums for the centre test."""

    def __init__(self, U, upx, usy, usx, upcy, upcx, upn, ucountry, unbr):
        self.U = U
        self.h, self.w = U.shape
        nu = len(upx)
        self.reg = np.arange(nu)                      # unit -> region (initially itself)
        self.units = {r: [r] for r in range(nu)}
        self.px = upx.astype(np.float64).copy()
        self.wt = self.px.copy()                     # size the merges aim at (sea: coast-weighted)
        self.sy, self.sx = usy.copy(), usx.copy()
        self.pcy, self.pcx, self.pn = upcy.copy(), upcx.copy(), upn.astype(np.float64).copy()
        self.country = [dict(c) for c in ucountry]   # region -> {country: px}
        self.nbr = [dict(d) for d in unbr]           # region -> {region: contact}
        self.alive = np.ones(nu, dtype=bool)

    def _in(self, cy, cx, ok):
        ys = {min(max(int(math.floor(cy)), 0), self.h - 1), min(max(int(math.ceil(cy)), 0), self.h - 1)}
        xs = {min(max(int(math.floor(cx)), 0), self.w - 1), min(max(int(math.ceil(cx)), 0), self.w - 1)}
        return all(int(self.reg[self.U[y, x]]) in ok for y in ys for x in xs)

    def inside(self, rs):
        """Is the centre of the union of regions `rs` within it (both tests)?"""
        rs = tuple(rs)
        px = sum(self.px[r] for r in rs)
        pn = sum(self.pn[r] for r in rs)
        if px <= 0 or pn <= 0:
            return False
        ok = set(rs)
        if not self._in(sum(self.sy[r] for r in rs) / px, sum(self.sx[r] for r in rs) / px, ok):
            return False
        return self._in(sum(self.pcy[r] for r in rs) / pn, sum(self.pcx[r] for r in rs) / pn, ok)

    def centre(self, r):
        return self.sy[r] / self.px[r], self.sx[r] / self.px[r]

    def top_country(self, r):
        c = self.country[r]
        return max(c, key=c.get) if c else None

    def merge(self, a, b):
        """b into a."""
        for u in self.units[b]:
            self.reg[u] = a
        self.units[a].extend(self.units.pop(b))
        for arr in (self.px, self.wt, self.sy, self.sx, self.pcy, self.pcx, self.pn):
            arr[a] += arr[b]
        for c, v in self.country[b].items():
            self.country[a][c] = self.country[a].get(c, 0) + v
        self.country[b] = {}
        for m, ln in self.nbr[b].items():
            if m == a:
                continue
            self.nbr[a][m] = self.nbr[a].get(m, 0) + ln
            self.nbr[m][a] = self.nbr[m].get(a, 0) + ln
            self.nbr[m].pop(b, None)
        self.nbr[a].pop(b, None)
        self.nbr[b] = {}
        self.alive[b] = False


def _agglomerate(R, members, allowed, thr, fit, cap, min_px, by_country):
    """Merge the regions in `members` (allowed(r, n) says which neighbours
    may join), smallest first, while a region weighs under `thr` (R.wt) or
    has under `min_px` pixels; merges keeping the centre inside first, then
    a merged size <= `fit`, then <= `cap` (px)."""
    heap = [(R.wt[r], r) for r in members if R.wt[r] < thr or R.px[r] < min_px]
    heapq.heapify(heap)
    merges = 0
    while heap:
        wt, r = heapq.heappop(heap)
        if not R.alive[r] or wt != R.wt[r] or (wt >= thr and R.px[r] >= min_px):
            continue
        cands = [n for n in R.nbr[r] if R.alive[n] and allowed(r, n)]
        if not cands:
            continue
        own_in = R.inside((r,))
        tc = R.top_country(r)
        best, best_key = None, None
        for n in cands:
            valid = R.inside((r, n))
            if not valid and own_in:
                continue
            merged = R.px[r] + R.px[n]
            key = (valid, merged <= fit, merged <= cap, bool(by_country) and R.top_country(n) == tc,
                   R.wt[n] < thr or R.px[n] < min_px, R.nbr[r][n] / math.sqrt(R.px[n]))
            if best_key is None or key > best_key:
                best, best_key = n, key
        if best is None:
            continue
        a, b = (best, r) if R.px[best] >= R.px[r] else (r, best)
        R.merge(a, b)
        merges += 1
        if R.wt[a] < thr or R.px[a] < min_px:
            heapq.heappush(heap, (R.wt[a], a))
    return merges


def _fix_centres(R, regions, allowed, unit_ok_move):
    """Regions whose centre is outside: merge with a neighbour that fixes it
    (smallest result first), else hand one of their units to a neighbour
    when both regions then have their centre inside. Returns (merged, moved)."""
    merged = moved = 0
    for _round in range(FIX_ROUNDS):
        bad = [r for r in regions() if not R.inside((r,))]
        if not bad:
            break
        changed = False
        for r in bad:
            if not R.alive[r] or R.inside((r,)):
                continue
            opts = sorted((R.px[r] + R.px[n], n) for n in R.nbr[r] if R.alive[n] and allowed(r, n)
                          and R.inside((r, n)))
            if opts:
                n = opts[0][1]
                a, b = (n, r) if R.px[n] >= R.px[r] else (r, n)
                R.merge(a, b)
                merged += 1
                changed = True
                continue
            if _move_unit(R, r, allowed, unit_ok_move):
                moved += 1
                changed = True
        if not changed:
            break
    return merged, moved


def _move_unit(R, r, allowed, unit_ok_move):
    """Give one unit of region r (on its border) to a neighbouring region so
    that both have their centre inside and r stays one piece."""
    units = R.units[r]
    if len(units) < 2:
        return False
    for u in sorted(units, key=lambda u: R.unit_px[u]):
        if not unit_ok_move(u):
            continue
        for n in R.unit_nbr[u]:
            t = int(R.reg[n])
            if t == r or not R.alive[t] or not allowed(r, t):
                continue
            # try: u from r to t
            _transfer(R, u, r, t)
            ok = R.inside((r,)) and R.inside((t,)) and _connected(R, r)
            if ok:
                return True
            _transfer(R, u, t, r)
    return False


def _new_region(R, units, r):
    """A new region (a merged-away id revived) holding `units` taken from r;
    None if no id is free."""
    d = next((u for u in units if u != r and not R.alive[u] and u not in R.units), None)
    if d is None:
        return None
    for arr in (R.px, R.wt, R.sy, R.sx, R.pcy, R.pcx, R.pn):
        arr[d] = 0.0
    R.units[d] = []
    R.country[d] = {}
    R.nbr[d] = {}
    R.alive[d] = True
    for u in units:
        _transfer(R, u, r, d)
    return d


def _kill_region(R, d, r):
    """Undo _new_region: d's units back to r."""
    for u in list(R.units[d]):
        _transfer(R, u, d, r)
    R.units.pop(d, None)
    R.alive[d] = False


def _split_region(R, r, rng, tries=8, max_parts=6):
    """Region r cut into 2, then 3 ... max_parts parts along its units
    (k-means on the unit centres: straight lines between the parts) so that
    every part is one piece and has its centre inside. Returns the new
    regions (list) or None."""
    us = list(R.units[r])
    if len(us) < 2:
        return None
    cy = np.array([R.unit_sy[u] / max(R.unit_px[u], 1) for u in us])
    cx = np.array([R.unit_sx[u] / max(R.unit_px[u], 1) for u in us])
    wt = np.array([R.unit_px[u] for u in us])
    for k in range(2, min(max_parts, len(us)) + 1):
        seen = set()
        for _t in range(tries):
            c = np.column_stack([cy, cx])[rng.choice(len(us), k, replace=False)].astype(float)
            for _it in range(12):
                lab = ((cy[:, None] - c[None, :, 0]) ** 2 + (cx[:, None] - c[None, :, 1]) ** 2).argmin(1)
                if len(set(lab.tolist())) < k:
                    break
                c = np.array([[np.average(cy[lab == j], weights=wt[lab == j]),
                               np.average(cx[lab == j], weights=wt[lab == j])] for j in range(k)])
            if len(set(lab.tolist())) < k:
                continue
            key = tuple(lab.tolist())
            if key in seen:
                continue
            seen.add(key)
            made = []
            for j in range(1, k):
                d = _new_region(R, [u for u, l_ in zip(us, lab) if l_ == j], r)
                if d is None:
                    break
                made.append(d)
            total = R.px[r] + sum(R.px[x] for x in made)
            min_part = max(300.0, min(1000.0, total / (2 * k)))  # no slivers (Kuril rings: 195 px bits)
            if len(made) == k - 1 and all(_connected(R, x) and R.inside((x,)) and R.px[x] >= min_part
                                          for x in [r] + made):
                return made
            for d in made:
                _kill_region(R, d, r)
    return None


def _province_pieces(provs, pnbr):
    """Groups of `provs` joined by side contact (province adjacency)."""
    left = set(provs)
    out = []
    while left:
        a = left.pop()
        comp, stack = {a}, [a]
        while stack:
            q = stack.pop()
            for r in pnbr.get(q, ()):
                if r in left:
                    left.discard(r)
                    comp.add(r)
                    stack.append(r)
        out.append(comp)
    out.sort(key=len, reverse=True)
    return out


def _connect_sea_regions(R, uprov, unit, ukind, pnbr, is_sea=None, rounds=6, uzone=None):
    """The game: "Naval strategic region X is fractioned! The following
    provinces are separated from the rest" -- every province of a region
    with sea must touch the rest of it (an island state's islets included,
    though the state counts as one unit here). A cut-off piece gets the
    sea province of a neighbouring region it touches most, when both
    regions keep their centre inside and the donor stays in one piece.
    Returns (sea provinces moved, regions still in pieces)."""
    moved = 0

    def provs_of(r):
        return [p for u in R.units[r] for p in uprov[u]]

    def connected(r):
        return len(_province_pieces(provs_of(r), pnbr)) <= 1
    for _round in range(rounds):
        changed = False
        for r in [int(r) for r in np.flatnonzero(R.alive)]:
            if not R.alive[r] or not any(ukind[u] == SEA for u in R.units[r]):
                continue
            if is_sea is not None and not is_sea(r):
                continue  # never pull sea into a land region (user: region 110, a third sea)
            pieces = _province_pieces(provs_of(r), pnbr)
            for piece in pieces[1:]:
                touch = {}
                for p in piece:
                    for q in pnbr.get(p, ()):
                        u = int(unit[q])
                        if u >= 0 and ukind[u] == SEA and int(R.reg[u]) != r:
                            touch[u] = touch.get(u, 0) + 1
                rz = {uzone[x] for x in R.units[r] if ukind[x] == SEA} if uzone is not None else set()
                for u in sorted(touch, key=lambda u: -touch[u]):
                    t = int(R.reg[u])
                    if len(R.units[t]) < 2:
                        continue
                    if uzone is not None and uzone[u] is not None and rz and uzone[u] not in rz:
                        continue  # never across an EEZ line
                    _transfer(R, u, t, r)
                    if R.inside((r,)) and R.inside((t,)) and connected(t) and (is_sea is None or is_sea(t)):
                        moved += 1
                        changed = True
                        break
                    _transfer(R, u, r, t)
        if not changed:
            break
    left = sum(1 for r in np.flatnonzero(R.alive).tolist()
               if any(ukind[u] == SEA for u in R.units[int(r)]) and not connected(int(r)))
    return moved, left


def _transfer(R, u, a, b):
    """Unit u from region a to region b (sums and neighbour contacts)."""
    R.reg[u] = b
    R.units[a].remove(u)
    R.units[b].append(u)
    for arr, uarr in ((R.px, R.unit_px), (R.wt, R.unit_wt), (R.sy, R.unit_sy), (R.sx, R.unit_sx), (R.pcy, R.unit_pcy),
                      (R.pcx, R.unit_pcx), (R.pn, R.unit_pn)):
        arr[a] -= uarr[u]
        arr[b] += uarr[u]
    for c, v in R.unit_country[u].items():
        R.country[a][c] = R.country[a].get(c, 0) - v
        R.country[b][c] = R.country[b].get(c, 0) + v
    _rebuild_nbr(R, a)
    _rebuild_nbr(R, b)


def _rebuild_nbr(R, r):
    for m in list(R.nbr[r]):
        R.nbr[m].pop(r, None)
    d = {}
    for u in R.units[r]:
        for n, ln in R.unit_nbr[u].items():
            t = int(R.reg[n])
            if t != r:
                d[t] = d.get(t, 0) + ln
    R.nbr[r] = d
    for t, ln in d.items():
        R.nbr[t][r] = ln


def _connected(R, r):
    us = R.units[r]
    if len(us) <= 1:
        return True
    s = set(us)
    seen = {us[0]}
    stack = [us[0]]
    while stack:
        u = stack.pop()
        for n in R.unit_nbr[u]:
            if n in s and n not in seen and R.unit_link(u, n):
                seen.add(n)
                stack.append(n)
    return len(seen) == len(s)


# ---------------------------------------------------------------- main
def parse_sea_areas(states_csv: str | None, n: int):
    """(sea area per province id (0 = none), EEZ owner per province id ("" =
    none / outside every EEZ)) from province_states.csv's sea_area / sea_zone
    columns (the Provinces tab's sea areas = the sea strategic regions);
    None without them."""
    if not states_csv:
        return None
    lines = states_csv.splitlines()
    head = lines[0].split(";")
    if "sea_area" not in head:
        return None
    ip, ia = head.index("province"), head.index("sea_area")
    iz = head.index("sea_zone") if "sea_zone" in head else -1
    area = np.zeros(n + 1, dtype=np.int64)
    zone = [""] * (n + 1)
    for line in lines[1:]:
        f = line.split(";")
        if len(f) < len(head):
            continue
        try:
            pr = int(f[ip])
        except ValueError:
            continue
        if 0 < pr <= n and f[ia].strip():
            area[pr] = int(f[ia])
            zone[pr] = f[iz].strip() if iz >= 0 else ""
    return (area, zone) if area.any() else None


def build_regions(pimg: Image.Image, definition: str, states_csv: str | None, settings=None, view=None):
    """(regions list, report, region id per province (array), P) -- see the
    module docstring. Each region: id, name, kind (land / sea), provinces,
    states, px, centre (x, y), and with `view` (the render's debug["view"])
    and the climate data, "climate" (12 monthly means per variable) and
    "weather" (12 HOI4 weather periods, climate.weather_month)."""
    t0 = time.time()
    s = normalize_settings(settings)
    P, ptype, pstate, states = parse_inputs(pimg, definition, states_csv)
    h, w = P.shape
    n = len(ptype) - 1
    sea_areas = parse_sea_areas(states_csv, n)
    present = np.bincount(P.ravel(), minlength=n + 1)
    pa, pb, plen = _pairs(P)

    # Units: states (land + lake provinces), stateless land/lake groups, sea provinces.
    is_sea = ptype == SEA
    unit = np.full(n + 1, -1, dtype=np.int64)
    sid_list = sorted(set(pstate[(ptype != SEA) & (pstate > 0) & (present > 0)].tolist()))
    sid_unit = {sid: k for k, sid in enumerate(sid_list)}
    for p in np.flatnonzero((ptype != SEA) & (pstate > 0) & (present > 0)).tolist():
        unit[p] = sid_unit[int(pstate[p])]
    nu = len(sid_list)
    stateless = (ptype != SEA) & (ptype >= 0) & (pstate == 0) & (present > 0)
    stateless[0] = False
    if stateless.any():
        m = stateless[pa] & stateless[pb]
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components
        g = sp.coo_matrix((np.ones(int(m.sum())), (pa[m], pb[m])), shape=(n + 1, n + 1))
        _nc, lab = connected_components(g, directed=False)
        idx = np.flatnonzero(stateless)
        _u, comp = np.unique(lab[idx], return_inverse=True)
        unit[idx] = nu + comp
        nu += int(comp.max()) + 1
    first_sea_unit = nu
    sea_ids = np.flatnonzero(is_sea & (present > 0))
    unit[sea_ids] = nu + np.arange(len(sea_ids))
    nu += len(sea_ids)
    ukind = np.array([LAND] * first_sea_unit + [SEA] * (nu - first_sea_unit), dtype=np.int8)
    ustate = np.zeros(nu, dtype=np.int64)
    ustate[:len(sid_list)] = sid_list

    U = unit[P]
    ys, xs = np.mgrid[0:h, 0:w]
    upx = np.bincount(U.ravel(), minlength=nu).astype(np.float64)
    usy = np.bincount(U.ravel(), weights=ys.ravel(), minlength=nu)
    usx = np.bincount(U.ravel(), weights=xs.ravel(), minlength=nu)
    pcy = np.bincount(P.ravel(), weights=ys.ravel(), minlength=n + 1) / np.maximum(present, 1)
    pcx = np.bincount(P.ravel(), weights=xs.ravel(), minlength=n + 1) / np.maximum(present, 1)
    have = np.flatnonzero(present > 0)
    have = have[have > 0]
    upcy = np.bincount(unit[have], weights=pcy[have], minlength=nu)
    upcx = np.bincount(unit[have], weights=pcx[have], minlength=nu)
    upn = np.bincount(unit[have], minlength=nu)
    ucountry = [{} for _ in range(nu)]
    for sid, k in sid_unit.items():
        c = states.get(sid, ("", ""))[1]
        if c:
            ucountry[k][c] = upx[k]
    ua, ub = unit[pa], unit[pb]
    keep = ua != ub
    unbr = [dict() for _ in range(nu)]
    for a, b, ln in zip(ua[keep].tolist(), ub[keep].tolist(), plen[keep].tolist()):
        unbr[a][b] = unbr[a].get(b, 0) + ln
        unbr[b][a] = unbr[b].get(a, 0) + ln

    R = _Regions(U, upx, usy, usx, upcy, upcx, upn, ucountry, unbr)
    # Sea units near a coast count up to COAST_WEIGHT x their size (vanilla's
    # coastal / enclosed seas are regions of ~10 provinces, open ocean ~35+):
    # weight = 1 + (COAST_WEIGHT - 1) x clip(1 - d / sqrt(sea_max), 0, 1),
    # d = distance from the province's centre to land.
    land_px = ptype[P] != SEA
    if land_px.any():
        dist = ndimage.distance_transform_edt(~land_px)
        uy = np.clip(np.rint(usy / np.maximum(upx, 1)).astype(np.int64), 0, h - 1)
        ux = np.clip(np.rint(usx / np.maximum(upx, 1)).astype(np.int64), 0, w - 1)
        d = dist[uy, ux]
        f = 1 + (COAST_WEIGHT - 1) * np.clip(1 - d / math.sqrt(s["sea_max"]), 0, 1)
        R.wt = np.where(ukind == SEA, upx * f, upx)
    R.unit_wt = R.wt.copy()
    R.unit_px, R.unit_sy, R.unit_sx = upx, usy, usx
    R.unit_pcy, R.unit_pcx, R.unit_pn = upcy, upcx, upn.astype(np.float64)
    R.unit_country = ucountry
    R.unit_nbr = unbr
    R.unit_link = lambda a, b: True  # any side contact joins units

    # Land masses (land units joined over land / lakes) and seas (sea units joined).
    def comps(kind):
        idx = np.flatnonzero(ukind == kind)
        parent = {int(u): int(u) for u in idx}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for u in idx.tolist():
            for v in unbr[u]:
                if ukind[v] == kind:
                    a_, b_ = find(u), find(v)
                    if a_ != b_:
                        parent[a_] = b_
        out = {}
        for u in idx.tolist():
            out.setdefault(find(u), []).append(u)
        return list(out.values())

    masses = comps(LAND)
    seas = comps(SEA)
    mass_of = np.full(nu, -1, dtype=np.int64)
    for k, us in enumerate(masses):
        mass_of[us] = k
    sea_of = np.full(nu, -1, dtype=np.int64)
    for k, us in enumerate(seas):
        sea_of[us] = k
    touches_sea = np.zeros(nu, dtype=bool)
    for u in range(first_sea_unit):
        touches_sea[u] = any(ukind[v] == SEA for v in unbr[u])
    islands = []
    for k, us in enumerate(masses):
        has_state = any(ustate[u] > 0 for u in us)
        if (has_state and all(ustate[u] > 0 for u in us) and sum(upx[u] for u in us) < s["island_px"]
                and any(touches_sea[u] for u in us)):
            islands.append(k)
    island_set = set(islands)
    report = {"settings": s, "units": {"states": len(sid_list), "stateless_land": first_sea_unit - len(sid_list),
                                       "sea_provinces": nu - first_sea_unit},
              "land_masses": len(masses), "seas": len(seas), "islands": len(islands)}

    # Land regions, per mass (islands wait for the sea).
    land_allowed = lambda r, nn: ukind[nn] == LAND and mass_of[R.units[nn][0]] == mass_of[R.units[r][0]]
    merges = 0
    aim = math.sqrt(s["land_min"] * s["land_max"])
    for k, us in enumerate(masses):
        if k in island_set:
            continue
        merges += _agglomerate(R, us, land_allowed, MERGE_SHARE * aim, FIT_FACTOR * aim, s["land_max"],
                               s["land_min"], s["by_country"])
    report["land_merges"] = merges

    # Sea regions. With the Provinces tab's sea areas (user: regions follow
    # the EEZ lines, banded / straight-edged inside -- provinces.SEA_AREA_*):
    # each area is one region, as it is. Without them: agglomerated per
    # connected sea.
    uzone = [None] * nu  # EEZ owner of a sea unit ("" outside every EEZ); None: no areas
    merges = 0
    if sea_areas is not None:
        parea, pzone = sea_areas
        for p_ in sea_ids.tolist():
            uzone[int(unit[p_])] = pzone[p_]
        by_area = {}
        for p_ in sea_ids.tolist():
            if parea[p_] > 0:
                by_area.setdefault(int(parea[p_]), []).append(int(unit[p_]))
        for us in by_area.values():
            uset, seen_ = set(us), set()
            for u0 in us:  # one region per connected group of the area's provinces
                if u0 in seen_:
                    continue
                seen_.add(u0)
                stack, r0 = [u0], int(R.reg[u0])
                while stack:
                    u = stack.pop()
                    for v in unbr[u]:
                        if v in uset and v not in seen_:
                            seen_.add(v)
                            stack.append(v)
                            if int(R.reg[v]) != r0:
                                R.merge(r0, int(R.reg[v]))
                                merges += 1
        report["sea_areas"] = len(by_area)
        # an area the Provinces tab's topology clean-up left as a sliver (a
        # 21 px province): into the sea region of the same EEZ it touches most
        tiny = 0
        for r in sorted({int(R.reg[u]) for us in by_area.values() for u in us}, key=lambda r: R.px[r]):
            if not R.alive[r] or R.px[r] >= SEA_AREA_TINY_PX:
                continue
            z = uzone[R.units[r][0]]
            opts = [(ln, t) for t, ln in R.nbr[r].items() if R.alive[t] and ukind[R.units[t][0]] == SEA]
            same = [o for o in opts if uzone[R.units[o[1]][0]] == z]
            pick = max(same or opts, default=None)
            if pick:
                R.merge(pick[1], r)
                tiny += 1
        report["sea_slivers_merged"] = tiny
    else:
        sea_allowed = lambda r, nn: (ukind[R.units[nn][0]] == SEA and ukind[R.units[r][0]] == SEA
                                     and sea_of[R.units[nn][0]] == sea_of[R.units[r][0]])
        for us in seas:
            merges += _agglomerate(R, us, sea_allowed, MERGE_SHARE * s["sea_max"], s["sea_max"], s["sea_max"],
                                   s["sea_min"], s["by_country"])
    report["sea_merges"] = merges

    def is_sea(r):
        """A sea region: more sea than land (islands in it, or a small
        enclosed sea in a land region, don't change its kind)."""
        sea = sum(upx[u] for u in R.units[r] if ukind[u] == SEA)
        return 2 * sea > R.px[r]

    def key_unit(r):
        k = SEA if is_sea(r) else LAND
        return next(u for u in R.units[r] if ukind[u] == k)

    # Islands: the whole mass into the sea region it borders most that keeps
    # the centre inside; else a land region of its own (with sea areas: the
    # region it borders most, its centre fixed by splitting below).
    attached, own = 0, []
    island_joined = []
    for k in islands:
        us = masses[k]
        contact = {}
        for u in us:
            for v, ln in unbr[u].items():
                if ukind[v] == SEA:
                    t = int(R.reg[v])
                    contact[t] = contact.get(t, 0) + ln
        # gather the island into one region first
        a = int(R.reg[us[0]])
        for u in us[1:]:
            b = int(R.reg[u])
            if b != a:
                R.merge(a, b)
        done = False
        for t, _ln in sorted(contact.items(), key=lambda kv: -kv[1]):
            if R.inside((t, a)):
                R.merge(t, a)
                attached += 1
                done = True
                break
        if not done and sea_areas is not None and contact:
            t = max(contact, key=contact.get)
            R.merge(t, a)
            attached += 1
            island_joined.append((t, list(us)))
            done = True
        if not done:
            own.append(a)
    report["islands_in_sea_regions"] = attached
    report["islands_alone"] = len(own)


    # Centres still outside: merge / move.
    def same_group(r, nn):
        sr, sn = is_sea(r), is_sea(nn)
        if sr or sn:
            return (sr and sn and sea_of[key_unit(r)] == sea_of[key_unit(nn)]
                    and uzone[key_unit(r)] == uzone[key_unit(nn)])  # never across an EEZ line
        return mass_of[key_unit(r)] == mass_of[key_unit(nn)]

    def unit_ok_move(u):
        # an island's units move only together: never one by one
        return not (ukind[u] == LAND and mass_of[u] in island_set)
    fm, mv = _fix_centres(R, lambda: [int(r) for r in np.flatnonzero(R.alive)], same_group, unit_ok_move)
    report["centre_fix_merges"] = fm
    report["centre_fix_moves"] = mv
    # Second pass: regions left small because every merge would have put a
    # centre outside at the time (their neighbours' centres were fixed since).
    alive_now = lambda: [int(r) for r in np.flatnonzero(R.alive)]
    m2 = _agglomerate(R, [r for r in alive_now() if not is_sea(r)], lambda r, nn: not is_sea(nn) and same_group(r, nn),
                      MERGE_SHARE * aim, FIT_FACTOR * aim, s["land_max"], s["land_min"], s["by_country"])
    if sea_areas is None:  # with areas, sea regions are the areas
        m2 += _agglomerate(R, [r for r in alive_now() if is_sea(r)], lambda r, nn: is_sea(nn) and same_group(r, nn),
                           MERGE_SHARE * s["sea_max"], s["sea_max"], s["sea_max"], s["sea_min"], s["by_country"])
    report["second_pass_merges"] = m2
    # Small enclosed seas (a whole connected sea under the sea minimum: fjord
    # pockets, pieces of the edge bands' sea) join the region they border
    # most (land, or an island's sea region) when its centre stays inside --
    # the mirror of islands in sea regions; they can't reach any other sea.
    # (only into a region that stays one piece by province contact: a region
    # with sea is "naval" for the game, which wants it unfractioned -- a fjord
    # pocket merged into a land region with islets once made the connection
    # fix below pull a 7,700 px North Sea province into a Norwegian land region)
    pnbr = {}
    for a_, b_ in zip(pa.tolist(), pb.tolist()):
        if a_ and b_:
            pnbr.setdefault(a_, set()).add(b_)
            pnbr.setdefault(b_, set()).add(a_)
    uprov = {}
    for p_ in np.flatnonzero(unit >= 0).tolist():
        uprov.setdefault(int(unit[p_]), []).append(p_)

    def one_piece(*rs):
        return len(_province_pieces([p_ for r_ in rs for u in R.units[r_] for p_ in uprov.get(u, ())], pnbr)) <= 1
    small_seas = 0
    for us in seas:
        if sum(upx[u] for u in us) >= s["sea_min"]:
            continue
        for r in sorted({int(R.reg[u]) for u in us}):
            if not R.alive[r] or not is_sea(r):
                continue
            opts = sorted((-ln, t) for t, ln in R.nbr[r].items() if R.alive[t] and R.inside((t, r))
                          and one_piece(t, r))
            if opts:
                R.merge(opts[0][1], r)
                small_seas += 1
    report["small_seas_in_land_regions"] = small_seas
    fm2, mv2 = _fix_centres(R, alive_now, same_group, unit_ok_move)
    report["centre_fix_merges"] += fm2
    report["centre_fix_moves"] += mv2
    # An island (whole land masses) still with its centre outside joins the
    # sea region around it that puts it inside, whatever its size.
    joined = 0
    for r in [int(r) for r in np.flatnonzero(R.alive)]:
        if not R.alive[r] or is_sea(r) or R.inside((r,)):
            continue
        if any(ukind[u] == SEA for u in R.units[r]):
            continue
        ms = {int(mass_of[u]) for u in R.units[r]}
        if sum(len(masses[k]) for k in ms) != len(R.units[r]):
            continue  # not whole land masses
        opts = sorted(((-ln, t) for t, ln in R.nbr[r].items() if R.alive[t] and is_sea(t)
                       and R.inside((t, r))))
        if opts:
            R.merge(opts[0][1], r)
            joined += 1
    report["islands_joined_to_fix_centre"] = joined
    # Sea regions still with their centre outside (thin coastal bands, an
    # island added): cut in two along their provinces, again until inside --
    # a single sea province always has its centre inside (Provinces tab rule).
    # An island whose region can't be fixed so gets a land region of its own.
    split_n = 0
    if sea_areas is not None:
        rng_ = np.random.default_rng(1936)
        for _round in range(12):
            bad = [r for r in alive_now() if is_sea(r) and not R.inside((r,))]
            if not bad:
                break
            for r in bad:
                if not R.alive[r] or R.inside((r,)):
                    continue
                if _split_region(R, r, rng_) is not None:
                    split_n += 1
                    continue
                # no split into real regions: join the same-EEZ sea neighbour that
                # puts the centre inside (most contact first)
                z = uzone[key_unit(r)]
                for _ln, t in sorted((-ln, t) for t, ln in R.nbr[r].items()
                                     if R.alive[t] and is_sea(t) and uzone[key_unit(t)] == z):
                    if R.inside((t, r)):
                        R.merge(t, r)
                        split_n += 1
                        break
        for t, isl in island_joined:
            r = int(R.reg[isl[0]])
            if R.inside((r,)) or len(R.units[r]) <= len(isl):
                continue
            d = _new_region(R, isl, r)  # out again; then any neighbouring sea region taking it
            if d is None:
                continue
            for _ln, t2 in sorted(((-ln, t2) for t2, ln in R.nbr[d].items() if R.alive[t2] and is_sea(t2))):
                if R.inside((t2, d)):
                    R.merge(t2, d)
                    d = None
                    break
            if d is not None:
                report["islands_alone"] += 1
    report["sea_centre_splits"] = split_n
    # every province of a sea region touching the rest of it (the game's
    # "fractioned" map error)
    report["sea_provinces_moved_to_connect"], report["sea_regions_in_pieces"] = \
        _connect_sea_regions(R, uprov, unit, ukind, pnbr, is_sea, uzone=uzone)

    # Output: ids 1..N, land regions first (by position), then sea.
    alive = [int(r) for r in np.flatnonzero(R.alive) if R.units.get(int(r))]
    def sort_key(r):
        cy, cx = R.centre(r)
        return (is_sea(r), round(cy / 256), cx)
    alive.sort(key=sort_key)
    rid_of_reg = {r: i + 1 for i, r in enumerate(alive)}
    prov_region = np.zeros(n + 1, dtype=np.int64)
    ok = unit >= 0
    prov_region[ok] = [rid_of_reg.get(int(R.reg[u]), 0) for u in unit[ok].tolist()]
    regions = []
    for r in alive:
        provs = np.flatnonzero(prov_region == rid_of_reg[r]).tolist()
        sts = sorted({int(ustate[u]) for u in R.units[r] if ustate[u] > 0})
        cy, cx = R.centre(r)
        kind = "sea" if is_sea(r) else "land"
        regions.append({"id": rid_of_reg[r], "kind": kind, "provinces": provs, "states": sts,
                        "px": int(R.px[r]), "centre": (round(cx, 2), round(cy, 2)),
                        "countries": sorted(R.country[r], key=lambda c: -R.country[r][c])[:4]})
    _name_regions(regions, states, R, rid_of_reg, unbr, ukind, ustate)
    report["check"] = check(P, ptype, pstate, prov_region, regions)
    report["count"] = len(regions)
    for kind in ("land", "sea"):
        v = np.array([g["px"] for g in regions if g["kind"] == kind])
        report[kind] = {"count": int(len(v)), "mean_px": int(v.mean()) if len(v) else 0,
                        "median_px": int(np.median(v)) if len(v) else 0,
                        "min_px": int(v.min()) if len(v) else 0, "max_px": int(v.max()) if len(v) else 0,
                        "below_min": int((v < s[f"{kind}_min"]).sum()), "above_max": int((v > s[f"{kind}_max"]).sum())}
    report["sea_with_islands"] = sum(1 for g in regions if g["kind"] == "sea" and g["states"])
    report["weather"] = _add_weather(regions, prov_region, P, view)
    report["naval_terrain"] = _add_naval_terrain(regions, prov_region, P, ptype, view)
    report["seconds"] = round(time.time() - t0, 2)
    return regions, report, prov_region, P


def _name_regions(regions, states, R, rid_of_reg, unbr, ukind, ustate):
    """Land: its largest state's name ("Unowned land" without states); sea:
    "<largest island state> waters" with islands, else "<state on its
    longest coast> coast", else "Open sea"; made unique with a number."""
    reg_of_rid = {i: r for r, i in rid_of_reg.items()}

    def largest_state(r):
        us = [u for u in R.units[r] if ustate[u] > 0]
        if not us:
            return ""
        u = max(us, key=lambda u: R.unit_px[u])
        return states.get(int(ustate[u]), ("",))[0]

    def coast(r):
        contact = {}
        for u in R.units[r]:
            if ukind[u] != SEA:
                continue
            for v, ln in unbr[u].items():
                if ukind[v] != SEA and ustate[v] > 0 and int(R.reg[v]) != r:
                    contact[v] = contact.get(v, 0) + ln
        if not contact:
            return ""
        return states.get(int(ustate[max(contact, key=contact.get)]), ("",))[0]

    seen = {}
    for g in regions:
        r = reg_of_rid[g["id"]]
        if g["kind"] == "land":
            name = largest_state(r) or "Unowned land"
        elif g["states"]:
            name = f"{largest_state(r)} waters"
        else:
            c = coast(r)
            name = f"{c} coast" if c else "Open sea"
        base = name.strip() or "Region"
        k = seen.get(base, 0) + 1
        seen[base] = k
        g["name"] = base if k == 1 else f"{base} {k}"


def _add_weather(regions, prov_region, P, view):
    """Each region's climate (ERA5 1991-2020 normals over its pixels) and
    HOI4 weather; the report says where it came from."""
    import climate
    if view is None:
        return {"source": None, "note": "no view"}
    if not climate.available():
        return {"source": None, "note": "climate normals not downloaded (Download tab): placeholder weather"}
    t0 = time.time()
    RI = prov_region[P]
    clim = climate.region_climate(view, RI, len(regions))
    for g in regions:
        r = g["id"]
        g["climate"] = {v: [round(float(x), 2) for x in clim[v][r]] for v in climate.VARIABLES}
        g["weather"] = climate.region_weather(clim, r, g["kind"] == "sea")
    return {"source": "ERA5 1991-2020 monthly normals (World Bank CCKP)", "seconds": round(time.time() - t0, 2)}


# Naval terrain of sea regions (game: ocean / water_shallow_sea /
# water_deep_ocean / water_fjords, common/terrain/00_terrain.txt). Rules in
# km, set from vanilla's sea regions (_dev/naval_vanilla.py: fjord regions
# have a median 10.6 islands per 10,000 px of sea vs 4.5 for shallow seas
# and 1.8 for plain ocean; Danish Belts / Aegean / Baltic have >= 25% of
# their water within ~2 px of land; deep-ocean regions lie far from land)
# and the bathymetry (vanilla's shallow seas are shelf seas: North Sea,
# Channel, Persian Gulf, Yellow Sea, ...). Order: fjords, shallow, deep, ocean.
NAVAL_ISLAND_KM2 = (100.0, 150_000.0)   # land masses counted as islands
# The vanilla figures in px don't carry over (finer maps show more small
# islands): set on the default view by geography -- fjords / archipelagos
# at 5.5-22 islands per 100,000 km2 (Norwegian coast, Hebrides, Danish
# belts, Estonian islands, Gotland, Aegean, Ionian / Adriatic), plain seas
# at 0-3.4 (Tyrrhenian, Ligurian, west of Ireland, Persian Gulf, Azov).
NAVAL_FJORD_ISLANDS = 5.5               # islands per 100,000 km2 of sea
NAVAL_FJORD_NEAR = (15.0, 0.3)          # or this share of the water within 15 km of land
NAVAL_SHALLOW = (200.0, 0.6)            # share of the water shallower than 200 m
NAVAL_DEEP = (3000.0, 100.0, 0.3)       # median depth (m), and under 30% within 100 km of land
NAVAL_TERRAINS = ("water_fjords", "water_shallow_sea", "water_deep_ocean", "ocean")


def _km_per_px(view, x, y):
    """(km per px east, km per px north) at pixel (x, y) of a render's frame."""
    import core
    h, scale = view["height"], view["scale"]
    gx = view["win_xmin"] + (np.array([x, x + 1, x]) + 0.5) / scale
    gy = view["win_ymin"] + (h - (np.array([y, y, y + 1]) + 0.5)) / scale
    lon, lat, _ok = core._unproject(view["projection"], view["center"], gx, gy)
    lat_r = np.radians(lat)

    def hav(i, j):
        dlat, dlon = lat_r[j] - lat_r[i], np.radians(((lon[j] - lon[i]) + 180.0) % 360.0 - 180.0)
        a = np.sin(dlat / 2) ** 2 + np.cos(lat_r[i]) * np.cos(lat_r[j]) * np.sin(dlon / 2) ** 2
        return 2 * 6371.0 * np.arcsin(np.sqrt(min(1.0, a)))
    return max(hav(0, 1), 1e-6), max(hav(0, 2), 1e-6)


def _add_naval_terrain(regions, prov_region, P, ptype, view):
    """g["naval_terrain"] and g["naval"] (the measures) for every sea
    region; the report counts them."""
    import bathymetry
    import core
    sea_regions = [g for g in regions if g["kind"] == "sea"]
    if not sea_regions:
        return {"counts": {}}
    if view is None:
        for g in sea_regions:
            g["naval_terrain"] = "ocean"
        return {"counts": {"ocean": len(sea_regions)}, "note": "no view: every sea region ocean"}
    t0 = time.time()
    h, w = P.shape
    sea = ptype[P] == SEA
    RI = prov_region[P]
    land = ~sea
    lab, _nl = ndimage.label(land, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(lab.ravel()).astype(np.float64)
    # the edge bands' unowned strip is neither an island nor a coast
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is not None and (sl[1].stop - sl[1].start) <= 12 and (sl[0].stop - sl[0].start) >= h // 2:
            sizes[k] = 0
            land[sl] &= lab[sl] != k
    dist = ndimage.distance_transform_edt(~land)
    have_depth = bathymetry.available()
    counts = {}
    objs = ndimage.find_objects(np.where(sea, RI, 0))
    for g in sea_regions:
        sl = objs[g["id"] - 1] if g["id"] - 1 < len(objs) else None
        if sl is None:
            g["naval_terrain"] = "ocean"
            counts["ocean"] = counts.get("ocean", 0) + 1
            continue
        r0, r1 = max(sl[0].start - 1, 0), min(sl[0].stop + 1, h)
        c0, c1 = max(sl[1].start - 1, 0), min(sl[1].stop + 1, w)
        m = (RI[r0:r1, c0:c1] == g["id"]) & sea[r0:r1, c0:c1]
        cx, cy = g["centre"]
        kx, ky = _km_per_px(view, int(cx), int(cy))
        km_px, km2_px = math.sqrt(kx * ky), kx * ky
        area_km2 = m.sum() * km2_px
        d_km = dist[r0:r1, c0:c1][m] * km_px
        near15 = float((d_km <= NAVAL_FJORD_NEAR[0]).mean())
        near100 = float((d_km <= NAVAL_DEEP[1]).mean())
        # islands: land masses of island size touching this region's water
        ring = ndimage.binary_dilation(m, structure=np.ones((3, 3), dtype=bool)) & land[r0:r1, c0:c1]
        ids = np.unique(lab[r0:r1, c0:c1][ring])
        ids = ids[ids > 0]
        isl_km2 = sizes[ids] * km2_px
        n_isl = int(((isl_km2 >= NAVAL_ISLAND_KM2[0]) & (isl_km2 <= NAVAL_ISLAND_KM2[1])).sum())
        isl_rate = n_isl / max(area_km2, 1.0) * 100_000.0
        meas = {"area_km2": int(area_km2), "islands": n_isl, "islands_per_100k_km2": round(isl_rate, 2),
                "near15km": round(near15, 3), "near100km": round(near100, 3)}
        if have_depth:
            ys, xs = np.nonzero(m)
            step = max(1, len(ys) // 2000)
            ys, xs = ys[::step] + r0, xs[::step] + c0
            gx = view["win_xmin"] + (xs + 0.5) / view["scale"]
            gy = view["win_ymin"] + (h - (ys + 0.5)) / view["scale"]
            lon, lat, _ok = core._unproject(view["projection"], view["center"], gx, gy)
            dep = bathymetry.depth_m(lat, lon)
            meas["median_depth_m"] = int(np.median(dep))
            meas["shallow_share"] = round(float((dep < NAVAL_SHALLOW[0]).mean()), 3)
        # water cut thin by the frame's left / right edge (the edge bands' sea,
        # slivers of a sea crossing the edge) is near land only because of
        # the frame: no near-land test
        edge_strip = (c1 - c0) <= max(14, 0.02 * w) and (c0 == 0 or c1 == w)
        meas["edge_strip"] = edge_strip
        if isl_rate >= NAVAL_FJORD_ISLANDS or (near15 >= NAVAL_FJORD_NEAR[1] and not edge_strip):
            t = "water_fjords"
        elif have_depth and meas["shallow_share"] >= NAVAL_SHALLOW[1]:
            t = "water_shallow_sea"
        elif have_depth and meas["median_depth_m"] >= NAVAL_DEEP[0] and near100 < NAVAL_DEEP[2]:
            t = "water_deep_ocean"
        else:
            t = "ocean"
        g["naval_terrain"] = t
        g["naval"] = meas
        counts[t] = counts.get(t, 0) + 1
    rep = {"counts": counts, "seconds": round(time.time() - t0, 2)}
    if not have_depth:
        rep["note"] = "no bathymetry (Download tab): shallow seas and deep ocean can't be told apart, they are ocean"
    return rep


# ---------------------------------------------------------------- checks
def check(P, ptype, pstate, prov_region, regions):
    """Independent check of the game's rules (see the module docstring)."""
    n = len(ptype) - 1
    present = np.bincount(P.ravel(), minlength=n + 1) > 0
    present[0] = False
    out = {}
    out["provinces_without_region"] = int((present & (prov_region == 0)).sum())
    listed = np.concatenate([np.asarray(g["provinces"], dtype=np.int64) for g in regions]) if regions else np.zeros(0)
    out["provinces_in_two_regions"] = int(len(listed) - len(np.unique(listed)))
    out["empty_regions"] = sum(1 for g in regions if not g["provinces"])
    st = {}
    for p in np.flatnonzero(pstate > 0).tolist():
        if present[p]:
            st.setdefault(int(pstate[p]), set()).add(int(prov_region[p]))
    out["states_in_two_regions"] = sum(1 for v in st.values() if len(v) > 1)
    RI = prov_region[P]
    h, w = P.shape
    ys, xs = np.mgrid[0:h, 0:w]
    nr = len(regions) + 1
    cnt = np.bincount(RI.ravel(), minlength=nr)
    sy = np.bincount(RI.ravel(), weights=ys.ravel(), minlength=nr)
    sx = np.bincount(RI.ravel(), weights=xs.ravel(), minlength=nr)
    pc = np.bincount(P.ravel(), minlength=n + 1)
    pcy = np.bincount(P.ravel(), weights=ys.ravel(), minlength=n + 1) / np.maximum(pc, 1)
    pcx = np.bincount(P.ravel(), weights=xs.ravel(), minlength=n + 1) / np.maximum(pc, 1)
    outside, outside_p, pieces, mixed_islands = [], [], [], []

    def inside(rid, cy, cx):
        for y in {int(math.floor(cy)), int(math.ceil(cy))}:
            for x in {int(math.floor(cx)), int(math.ceil(cx))}:
                if not (0 <= y < h and 0 <= x < w) or RI[y, x] != rid:
                    return False
        return True
    for g in regions:
        rid = g["id"]
        if not cnt[rid]:
            continue
        if not inside(rid, sy[rid] / cnt[rid], sx[rid] / cnt[rid]):
            outside.append(rid)
        pv = np.asarray(g["provinces"], dtype=np.int64)
        pv = pv[pc[pv] > 0]
        if len(pv) and not inside(rid, pcy[pv].mean(), pcx[pv].mean()):
            outside_p.append(rid)
    # connectivity: provinces joined by side contact within the region, and
    # all provinces of a state joined (a state's own islands are fine)
    pa, pb, _c = _pairs(P)
    same = (prov_region[pa] == prov_region[pb]) & (prov_region[pa] > 0)
    stp = np.flatnonzero((pstate > 0) & present)
    first = {}
    ea, eb = [pa[same]], [pb[same]]
    for p in stp.tolist():
        q = first.setdefault(int(pstate[p]), p)
        if q != p:
            ea.append(np.array([q]))
            eb.append(np.array([p]))
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    ea, eb = np.concatenate(ea), np.concatenate(eb)
    g_ = sp.coo_matrix((np.ones(len(ea)), (ea, eb)), shape=(n + 1, n + 1))
    _k, lab = connected_components(g_, directed=False)
    for g in regions:
        k = len(np.unique(lab[np.asarray(g["provinces"], dtype=np.int64)])) if g["provinces"] else 0
        if k > 1:
            pieces.append((g["id"], int(k)))
    out["centre_outside"] = outside
    out["province_centre_outside"] = outside_p
    out["regions_in_pieces"] = pieces
    # the game's own rule for regions with sea ("Naval strategic region X is
    # fractioned"): every province touching the rest, islets of a state too
    pnbr = {}
    for a_, b_ in zip(pa[same].tolist(), pb[same].tolist()):
        pnbr.setdefault(a_, set()).add(b_)
        pnbr.setdefault(b_, set()).add(a_)
    fractioned = []
    for g in regions:
        pv = [q for q in g["provinces"] if present[q]]
        if any(ptype[q] == SEA for q in pv):
            k = len(_province_pieces(pv, pnbr))
            if k > 1:
                fractioned.append((g["id"], k))
    out["sea_regions_fractioned"] = fractioned
    # islands in sea regions: every state of the island in that region
    by_state = {sid: next(iter(v)) for sid, v in st.items() if len(v) == 1}
    out["sea_regions_with_partial_states"] = sum(
        1 for g in regions if g["kind"] == "sea" for sid in g["states"] if by_state.get(sid) != g["id"])
    out["ok"] = not (out["provinces_without_region"] or out["provinces_in_two_regions"] or out["empty_regions"]
                     or out["states_in_two_regions"] or outside or outside_p
                     or out["sea_regions_with_partial_states"] or pieces or fractioned)
    return out


# ---------------------------------------------------------------- outputs
_DAYS = (30, 27, 30, 29, 30, 29, 30, 30, 29, 30, 29, 30)  # vanilla's "between" ends (0-based day.month)


def _weather_block(weather=None):
    """12 monthly periods as vanilla writes them: the region's weather
    (climate.weather_month) or, without climate data, a neutral placeholder
    (mild, no phenomena but light rain)."""
    out = ["\tweather={\n"]
    for mth, end in enumerate(_DAYS):
        w = weather[mth] if weather else {"tmin": 5.0, "tmax": 20.0, "no_phenomenon": 1.0, "rain_light": 0.2}
        out.append("\t\tperiod={\n"
                   f"\t\t\tbetween={{ 0.{mth} {end}.{mth} }}\n"
                   f"\t\t\ttemperature={{ {w['tmin']:.1f} {w['tmax']:.1f} }}\n")
        for k in ("no_phenomenon", "rain_light", "rain_heavy", "snow", "blizzard", "arctic_water", "mud",
                  "sandstorm", "min_snow_level"):
            out.append(f"\t\t\t{k}={w.get(k, 0.0):.3f}\n")
        out.append("\t\t}\n")
    out.append("\t}\n")
    return "".join(out)


def _safe_file_name(s):
    try:
        from unidecode import unidecode
        s = unidecode(s)
    except ImportError:
        import unicodedata
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", s).strip().rstrip(".")
    return s or "region"


def region_file(g):
    provs = " ".join(str(p) for p in g["provinces"])
    return ("strategic_region={\n"
            f"\tid={g['id']}\n"
            f"\tname=\"STRATEGICREGION_{g['id']}\"\n"
            "\tprovinces={\n"
            f"\t\t{provs} \n"
            "\t}\n"
            + (f"\tnaval_terrain={g['naval_terrain']}\n" if g.get("naval_terrain") else "")
            + _weather_block(g.get("weather")) +
            "}\n")


def weather_positions(regions, height):
    """map/weatherpositions.txt: one big and one small weather object per
    region at its centre (id;x;y;z;size, z = map height - row, as vanilla;
    the game wants at least one of each size)."""
    lines = []
    for g in regions:
        x, y = g["centre"]
        z = height - 1 - y  # the game reads row = map height - 1 - z (see hoi4mod.Geo.world)
        lines.append(f"{g['id']};{x:.2f};10.00;{z:.2f};big")
        lines.append(f"{g['id']};{x:.2f};10.00;{z:.2f};small")
    return "\n".join(lines) + "\n"


def localisation(regions):
    out = ["﻿l_english:"]
    for g in regions:
        nm = g["name"].replace('"', "'")
        out.append(f' STRATEGICREGION_{g["id"]}:0 "{nm}"')
    return "\n".join(out) + "\n"


def regions_csv(regions):
    rows = ["id;name;kind;provinces;states;px;centre_x;centre_y;countries;naval_terrain"]
    for g in regions:
        rows.append(f"{g['id']};{g['name'].replace(';', ',')};{g['kind']};{len(g['provinces'])};"
                    f"{' '.join(map(str, g['states']))};{g['px']};{g['centre'][0]};{g['centre'][1]};"
                    f"{' '.join(g['countries'])};{g.get('naval_terrain', '')}")
    return "\n".join(rows) + "\n"


def zip_bytes(regions, height):
    """map/strategicregions/<id>-<name>.txt + map/weatherpositions.txt +
    localisation/english/strategic_region_names_l_english.yml + the CSV."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for g in regions:
            z.writestr(f"map/strategicregions/{g['id']}-{_safe_file_name(g['name'])}.txt", region_file(g))
        z.writestr("map/weatherpositions.txt", weather_positions(regions, height))
        z.writestr("localisation/english/strategic_region_names_l_english.yml",
                   localisation(regions).encode("utf-8"))
        z.writestr("strategic_regions.csv", regions_csv(regions))
    buf.seek(0)
    return buf


def render_image(regions, prov_region, P, ptype, borders="regions", centres=True):
    """Regions coloured (land: varied hues; sea: blues; islands in a sea
    region in its colour, lighter), black region borders, (optional) grey
    province borders, red crosses on the centres."""
    nr = len(regions) + 1
    RI = prov_region[P]
    # neighbouring regions never share a colour: greedy colouring over a
    # fixed palette per kind (land: 12 hues x 2 shades, sea: 4 blues x 3)
    nb = [set() for _ in range(nr)]
    for x, y in ((RI[:, :-1], RI[:, 1:]), (RI[:-1, :], RI[1:, :])):
        m = x != y
        for a_, b_ in set(zip(x[m].tolist(), y[m].tolist())):
            nb[a_].add(b_)
            nb[b_].add(a_)
    hues = [h_ / 360 for h_ in (0, 22, 40, 55, 75, 100, 125, 150, 170, 280, 305, 330)]  # no blues: those are the sea
    land_pal = [colorsys.hsv_to_rgb(hh, sat, val) for sat, val in ((0.45, 0.9), (0.6, 0.72)) for hh in hues]
    sea_pal = [colorsys.hsv_to_rgb(hh, sat, val) for hh in (0.55, 0.58, 0.62, 0.66)
               for sat, val in ((0.55, 0.75), (0.7, 0.55), (0.4, 0.85))]
    col = np.zeros((nr, 3), dtype=np.uint8)
    used = {}
    for g in sorted(regions, key=lambda g: -len(nb[g["id"]])):
        pal = sea_pal if g["kind"] == "sea" else land_pal
        taken = {used.get(m) for m in nb[g["id"]]}
        start = zlib.crc32(f"sr{g['id']}".encode()) % len(pal)
        k = next((j % len(pal) for j in range(start, start + len(pal)) if (g["kind"], j % len(pal)) not in taken),
                 start)
        used[g["id"]] = (g["kind"], k)
        col[g["id"]] = [int(round(255 * c)) for c in pal[k]]
    a = col[RI]
    land_in_sea = np.zeros(nr, dtype=bool)
    for g in regions:
        land_in_sea[g["id"]] = g["kind"] == "sea"
    isl = land_in_sea[RI] & (ptype[P] != SEA)
    a[isl] = (a[isl].astype(np.float32) * 0.45 + 255 * 0.55).astype(np.uint8)
    if borders == "provinces":
        e = np.zeros(P.shape, dtype=bool)
        e[:, :-1] |= P[:, :-1] != P[:, 1:]
        e[:-1, :] |= P[:-1, :] != P[1:, :]
        a[e] = (a[e].astype(np.float32) * 0.6).astype(np.uint8)
    e = np.zeros(P.shape, dtype=bool)
    e[:, :-1] |= RI[:, :-1] != RI[:, 1:]
    e[:-1, :] |= RI[:-1, :] != RI[1:, :]
    a[e] = (0, 0, 0)
    if centres:
        h, w = P.shape
        for g in regions:
            x, y = int(round(g["centre"][0])), int(round(g["centre"][1]))
            for d in range(-3, 4):
                for yy, xx in ((y + d, x), (y, x + d)):
                    if 0 <= yy < h and 0 <= xx < w:
                        a[yy, xx] = (230, 0, 0)
    return Image.fromarray(a, mode="RGB")
