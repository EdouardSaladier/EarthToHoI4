"""City tab: GeoNames cities (population >= 1,000) on a blank map of the
Countries tab's view, as stars coloured by population class.

Pipeline (render_cities), in the spirit of the Rivers tab:
  1. Cities in view: every GeoNames populated place (sections of a city,
     historical / abandoned / destroyed places left out, see
     EXCLUDED_FEATURES) whose projected position is on the drawn land of a
     visible country -- a coastal city a pixel or two off GADM's coastline
     is moved to the nearest land pixel (up to SNAP_PX), hidden countries'
     cities are left out -- and with at least `min_population` people.
  2. Number of cities (user's design): one `target` for the whole map,
     shared out between the visible countries by a blend of their land and
     their population in view -- weight = (1 - p) * land share + p *
     population share, p = `pop_weight` % (_apportion: largest remainders,
     so the shares add up to the target exactly). A country's population in
     view = the sum of its GeoNames places in view (>= min_population,
     merged parts included): no extra dataset, and a country half in the
     frame (Russia) only counts its visible half. With `by_region` (default
     on), each country's number is split the same way between its regions
     (_region_weights / _split_regions; GeoNames admin1 of each city: England / Scotland /
     Wales / N. Ireland, oblasts, states ...) -- user: "England is full,
     Scotland and Wales nearly empty". The data wasn't the cause (GeoNames
     is proportional there: 948 / 94 / 65 places >= 10k for 57 / 5.4 / 3.1
     M people); filling the whole UK's number with its largest cities first
     was. Urban areas (`urban_areas`, default on; user: "France's cities
     concentrated in Ile-de-France close to each other"): a town within a
     larger city's urban area (same country; radius sqrt(P / (pi *
     3000/km2)), P growing with the towns it absorbs -- Paris ~30 km; towns
     >= 1/3 of the city stay cities) is part of that city and never
     shown on its own (_urban_areas) -- Sevran, Palaiseau, Chatou, Domont
     no longer take Ile-de-France's slots; they go to Meaux, Melun,
     Fontainebleau ... The shares use each place's *own* GeoNames figure
     (merged parts added would count Paris twice) and every place in view,
     urban-area towns included (their people are there).
     `counts` = the user's per-country overrides (the
     City tab's "Visible countries" panel), which replace a country's share
     and leave the others' unchanged. Each country then takes its most
     populated cities one by one until its number is reached (fewer if it
     has fewer in view). Countries = the render's colours
     (_country_groups): a merged autonomous territory counts with its
     parent, hidden countries not at all. A city closer than `min_dist_px`
     to a larger picked one (any country) is skipped and the country's
     next one takes its place (_spaced_pick): suburbs would otherwise
     stack their stars on one spot. Isolated cities (user): a city with
     no more populated city in view within `isolated_px` (50) is shown even
     when its country's number is full -- an extra, beyond the target (while
     there's room it just uses the number: a region's largest city is
     always "isolated"); extras don't set class limits (villages would
     stretch the log range) and are classed by them (_isolated).
     Exception (user): the cities of the top
     `dist_exempt_classes` classes only keep EXEMPT_FLOOR_PX from others,
     so dense clusters of large cities (Rhine-Ruhr) all show.
     Replaced designs (history): one city per spacing^2 px of land per
     continent (Finland too empty, the Netherlands too full), then per
     country (+ a "no larger city within 200 px" rule for sparse areas);
     before that a target average nearest-city distance -- erratic, since
     GeoNames lists city parts as separate places (Brooklyn next to New
     York stopped North America at 11 cities on a world map).
  2b. Small islands: a land area (8-connected) under `island_px` px shows
     its largest city only, whatever the density selection -- every
     inhabited small island gets one star, on top of the target (not
     counted in its country's land or number). The minimum distance only
     applies between cities on the same land area (user: Jersey's Saint
     Helier must not keep Cherbourg off): across the sea -- any island,
     small or not; Copenhagen / Malmoe -- cities never block each other,
     and the isolated-city test only looks for larger cities on the same
     land (Saint Helier, 28k, 42 px away, made Cherbourg, 26.7k, "not
     isolated" in the default view) -- unless a larger shown city is within
     `min_dist_px` (a speck off Shanghai must not hide Shanghai: islands
     are placed after the density selection, never block it).
  3. Classes per continent: `classes` population classes, class 1 = the
     largest cities; "log" = equal steps in log(population) between the
     continent's smallest and largest selected city, "count" = equal numbers
     of cities per class (limits from the density selection; small-island
     cities are classed by the same limits, below the last -> last class).
  5. Names (user): each drawn city labelled with its endonym or its exonym
     in `name_lang` (city_names.py), ASCII only or with accents; placed
     largest city first, larger text for higher classes, dark text on a
     light halo, at the first of 8 spots around its star (right, left,
     above, below, then the corners) that overlaps no star, no earlier
     label and stays in the frame -- else left out (report "unplaced").
  4. Output: land light grey, sea and lakes dark blue, every city a star of
     `star_px` px coloured by its class (CLASS_COLORS), drawn smallest
     population first, so a larger city's star is never covered by a
     smaller one's.
"""
from __future__ import annotations

import csv
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree

import city_names
import core

CITIES_CSV_PATH = core.DATA_DIR / "cities" / "cities1000.csv"

# Continents (codes shared with the Rivers tab, but political here, from
# each city's country as GeoNames assigns them: Turkey, the Caucasus and the
# Middle East are Asia, Cyprus Europe; Russia is split at the Urals).
CONTINENTS = [
    ("af", "Africa"),
    ("eu", "Europe"),
    ("as", "Asia"),
    ("oc", "Oceania"),
    ("sa", "South America"),
    ("na", "North America"),
]
_CONTINENT_COUNTRIES = {
    "af": "DZ AO BJ BW BF BI CM CV CF TD KM CD CG CI DJ EG GQ ER ET GA GM GH GN GW KE LS LR LY MG MW ML MR MU "
          "YT MA MZ NA NE NG RE RW SH ST SN SC SL SO ZA SS SD SZ TZ TG TN UG EH ZM ZW",
    "eu": "AX AL AD AT BY BE BA BG HR CY CZ DK EE FO FI FR DE GI GR GG HU IS IE IM IT JE XK LV LI LT LU MK MT "
          "MD MC ME NL NO PL PT RO RU SM RS SK SI ES SJ SE CH UA GB VA",
    "as": "AF AM AZ BH BD BT BN KH CN GE HK IN ID IR IQ IL JP JO KZ KW KG LA LB MO MY MV MN MM NP KP OM PK PS "
          "PH QA SA SG KR LK SY TW TJ TH TL TR TM AE UZ VN YE CX CC IO",
    "oc": "AS AU CK FJ PF GU KI MH FM NR NC NZ NU NF MP PW PG PN WS SB TK TO TV UM VU WF",
    "sa": "AR BO BR CL CO EC FK GF GY PY PE SR UY VE GS",
    "na": "AI AG AW BS BB BZ BM BQ VG CA KY CR CU CW DM DO SV GL GD GP GT HT HN JM MQ MX MS NI PA PR BL KN LC "
          "MF PM VC SX TT TC US VI",
}
COUNTRY_CONTINENT = {cc: code for code, ccs in _CONTINENT_COUNTRIES.items() for cc in ccs.split()}
URALS_LON = 60.0  # Russian cities east of this are Asia

# GeoNames feature codes left out: PPLX = section of a populated place (city
# districts, which would crowd every big city), PPLH historical, PPLQ
# abandoned, PPLW destroyed, PPLCH historical capital.
EXCLUDED_FEATURES = {"PPLX", "PPLH", "PPLQ", "PPLW", "PPLCH"}
SNAP_PX = 3  # a city this close to drawn land (off GADM's coastline) is moved onto it
EXEMPT_FLOOR_PX = 3  # cities exempt from the minimum distance still keep this far apart

CLASS_COLORS = [(220, 20, 60), (255, 140, 0), (255, 215, 0), (50, 205, 50), (0, 191, 255),
                (65, 105, 225), (148, 0, 211), (255, 0, 255)]  # class 1 (largest) ... 8
LAND_COLOR = (235, 235, 235)
WATER_COLOR = (38, 50, 72)
STAR_OUTLINE = (25, 25, 25)

CITY_DEFAULTS = {
    "target": 500,            # cities on the whole map, shared out by land
    "counts": {},             # per-country overrides {tag: n}
    "min_population": 1000,   # never a city below this (the dataset's own floor)
    "classes": 5,
    "class_method": "log",
    "star_px": 9,
    "min_dist_px": 10,        # no two cities closer than this (0 = off)
    "island_px": 40,          # islands smaller than this show their largest city only (0 = off)
    "dist_exempt_classes": 2,  # classes 1..N ignore min_dist_px (0 = none)
    "pop_weight": 50,         # % of each country's share from its population in view (rest: land)
    "by_region": True,        # split each country's number between its regions the same way
    "urban_areas": True,      # a town inside a larger city's urban radius isn't shown on its own
    "labels": True,           # city names on the map
    "name_set": "endonym",    # "endonym" (local name) or "exonym" (name in name_lang)
    "name_lang": "en",
    "ascii_only": False,      # fold accents / special letters
    "label_px": 12,           # text size of the lowest class (higher classes larger)
    "isolated_px": 50,        # a city with no larger one this close is shown beyond the limit (0 = off)
}
CITY_LIMITS = {"target": (0, 100_000), "min_population": (0, 100_000_000), "classes": (1, len(CLASS_COLORS)),
               "star_px": (3, 41), "min_dist_px": (0, 500),
               "island_px": (0, 100_000), "dist_exempt_classes": (0, len(CLASS_COLORS)),
               "pop_weight": (0, 100), "isolated_px": (0, 5000), "label_px": (6, 60)}
MAX_COUNT_PER_COUNTRY = 100_000

_index = None
_index_lock = threading.Lock()


def available() -> bool:
    return CITIES_CSV_PATH.exists()


# Merging a city's parts (Paris's arrondissements and quartiers, Warsaw's
# districts, New York's boroughs: separate GeoNames places) into it. GeoNames
# has no "part of" field; the admin codes are the signal: a part shares the
# city's code on every admin level the city has (Paris 75056 = its commune,
# Warsaw 146501 = its city county, New York only NY: it spans five counties),
# and lies within the city's radius at CITY_DENSITY_PER_KM2 (sqrt(pop / (pi
# density)): Paris 15 km, New York 31 km, Lyon 7 km). Names made of the
# city's name + a suffix ("Paris 15 Vaugirard", "Lyon 03") also merge, up to
# twice that. The parts' populations are added to the city's (see
# load_city_index). Villeurbanne stays apart from Lyon (other commune);
# known limit: a separate town in the same unit as a city that has only a
# state/region code (Yonkers vs New York) merges too if within the radius.
CITY_DENSITY_PER_KM2 = 3000.0


def _merge_city_parts(names, admin, country, pop, lon, lat):
    """Index of the city each place is merged into (-1 = kept on its own)."""
    n = len(pop)
    radius_km = np.sqrt(np.maximum(pop, 1) / (np.pi * CITY_DENSITY_PER_KM2))
    lon_r, lat_r = np.radians(lon), np.radians(lat)
    xyz = np.column_stack([np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)])
    tree = cKDTree(xyz)
    parent = np.full(n, -1, dtype=np.int64)
    # Only cities whose radius can reach another place are worth querying.
    big = np.flatnonzero(radius_km >= 0.5)
    big = big[np.argsort(-pop[big], kind="stable")]
    reach = 2.0 * radius_km[big] / core.EARTH_RADIUS
    for i, near in zip(big.tolist(), tree.query_ball_point(xyz[big], reach).tolist() if len(big) else []):
        if parent[i] >= 0:
            continue
        codes = admin[i]
        levels = [k for k, c in enumerate(codes) if c]
        prefix = names[i] + " "
        r2 = radius_km[i]
        for j in near:
            if j == i or parent[j] >= 0 or pop[j] > pop[i] or country[j] != country[i]:
                continue
            d = core.EARTH_RADIUS * float(np.linalg.norm(xyz[i] - xyz[j]))
            if names[j].startswith(prefix) or (d <= r2 and levels and all(admin[j][k] == codes[k] for k in levels)):
                parent[j] = i
    return parent


URBAN_MAX_SHARE = 1.0 / 3.0  # a place at least this share of a city's size is a city of its own, never absorbed
URBAN_MAX_POP = 1_000_000    # nor one of a million or more (Yokohama, Kawasaki, Saitama next to Tokyo)
URBAN_ROUNDS = 6


def _urban_areas(pop, country, lon, lat):
    """For places sorted largest first: the index of the larger city of the
    same country whose urban area holds it, or -1. The area grows: radius
    sqrt(P / (pi * CITY_DENSITY_PER_KM2)) with P = the city's population
    plus every town it has absorbed so far, recomputed until nothing new
    joins (Paris: 4.4 M -> 21.6 km, then with its suburbs ~30 km, like the
    real 10.8 M urban area). Only towns under URBAN_MAX_SHARE of the city
    join (Duisburg 500k and Gelsenkirchen 260k stay apart from Essen 593k;
    Villeurbanne 150k joins Lyon 537k), nor places of URBAN_MAX_POP or more.
    A place inside another's area
    doesn't grow one itself."""
    n = len(pop)
    lon_r, lat_r = np.radians(lon), np.radians(lat)
    xyz = np.column_stack([np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)])
    tree = cKDTree(xyz)
    urban_of = np.full(n, -1, dtype=np.int64)
    radius = lambda p: np.sqrt(np.maximum(p, 1) / (np.pi * CITY_DENSITY_PER_KM2)) / core.EARTH_RADIUS
    big = np.flatnonzero(radius(pop) * core.EARTH_RADIUS >= 1.0)  # >= ~9,400 people
    near_all = tree.query_ball_point(xyz[big], radius(pop[big])) if len(big) else []
    for i, near in zip(big.tolist(), near_all):
        if urban_of[i] >= 0:
            continue
        total = float(pop[i])
        limit = pop[i] * URBAN_MAX_SHARE
        for _round in range(URBAN_ROUNDS):
            new = [j for j in near if j > i and urban_of[j] < 0 and country[j] == country[i]
                   and pop[j] <= limit and pop[j] < URBAN_MAX_POP]
            if not new:
                break
            urban_of[new] = i
            total += float(pop[new].sum())
            near = tree.query_ball_point(xyz[i], radius(total))
    return urban_of


def load_city_index(path: Path = CITIES_CSV_PATH):
    """The GeoNames CSV (Opendatasoft export, `;`-separated) as arrays,
    memoized (~2 s), sorted most populated first: name, alternate names,
    feature code, ISO2 country,
    continent code, population, lon, lat, and "parts" (how many places were
    merged into each, _merge_city_parts). Excluded feature codes, places
    without a continent (Antarctic islands) and merged parts are dropped."""
    global _index
    with _index_lock:
        if _index is not None:
            return _index
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = csv.reader(f, delimiter=";")
            head = next(rows)
            col = {k: head.index(k) for k in ("geoname_id", "name", "alternate_names", "feature_code",
                                              "country_code", "population", "coordinates", "admin1_code",
                                              "admin2_code", "admin3_code", "admin4_code")}
            recs = []
            for r in rows:
                if r[col["feature_code"]] in EXCLUDED_FEATURES:
                    continue
                cc = r[col["country_code"]]
                try:
                    lat, lon = (float(v) for v in r[col["coordinates"]].split(","))
                    pop = int(r[col["population"]] or 0)
                except ValueError:
                    continue
                cont = COUNTRY_CONTINENT.get(cc)
                if cc == "RU" and lon >= URALS_LON:
                    cont = "as"
                if cont is None:
                    continue
                recs.append((r[col["geoname_id"]], r[col["name"]], r[col["alternate_names"]],
                             r[col["feature_code"]], cc, cont, pop, lon, lat,
                             tuple(r[col[f"admin{k}_code"]] for k in range(1, 5))))
        cols = list(zip(*recs))
        pop = np.array(cols[6], dtype=np.int64)
        lon, lat = np.array(cols[7], dtype=np.float64), np.array(cols[8], dtype=np.float64)
        parent = _merge_city_parts(cols[1], cols[9], cols[4], pop, lon, lat)
        keep = parent < 0
        parts = np.bincount(parent[~keep], minlength=len(pop))
        # A merged city's population = its own + all its parts' (user's
        # choice, knowingly: GeoNames' own figure usually already covers its
        # districts, so Paris becomes 4.43 M, New York 17.97 M, Warsaw
        # 3.42 M). The own figure is kept as "own_population".
        total = pop + np.bincount(parent[~keep], weights=pop[~keep], minlength=len(pop)).astype(np.int64)
        _index = {
            "id": np.array(cols[0], dtype=object)[keep],
            "name": np.array(cols[1], dtype=object)[keep],
            "alt": np.array(cols[2], dtype=object)[keep],
            "feature": np.array(cols[3], dtype=object)[keep],
            "country": np.array(cols[4], dtype=object)[keep],
            "continent": np.array(cols[5])[keep],
            "population": total[keep],
            "own_population": pop[keep],
            "lon": lon[keep],
            "lat": lat[keep],
            "parts": parts[keep],
            "admin1": np.array([a[0] for a in cols[9]], dtype=object)[keep],
            "merged": int((~keep).sum()),
        }
        # Most populated first (user): every later step -- selection, the
        # isolated-city test, ties, drawing, the CSV -- sees one order.
        order = np.argsort(-_index["population"], kind="stable")
        for k, v in _index.items():
            if isinstance(v, np.ndarray):
                _index[k] = v[order]
        _index["urban_of"] = _urban_areas(_index["population"], _index["country"], _index["lon"], _index["lat"])
        return _index


def normalize_settings(raw: dict) -> dict:
    """Settings from the app's query params (strings or None) or a dict,
    validated; missing -> CITY_DEFAULTS. `counts` may be "FIN:30,NLD:5".
    Raises core.RenderError."""
    s = {k: (dict(v) if isinstance(v, dict) else v) for k, v in CITY_DEFAULTS.items()}
    try:
        for k, cast in (("target", int), ("min_population", int), ("classes", int), ("star_px", int),
                         ("min_dist_px", int), ("island_px", int), ("dist_exempt_classes", int),
                         ("pop_weight", float), ("isolated_px", int), ("label_px", int)):
            if raw.get(k) not in (None, ""):
                s[k] = cast(float(raw[k]))
        for k in ("by_region", "urban_areas", "labels", "ascii_only"):
            if raw.get(k) not in (None, ""):
                s[k] = str(raw[k]).lower() in ("1", "true")
        counts = raw.get("counts")
        if isinstance(counts, str):
            counts = dict(item.split(":") for item in counts.split(",") if item)
        s["counts"] = {str(t).upper(): int(float(v)) for t, v in (counts or {}).items()}
    except (ValueError, TypeError):
        raise core.RenderError("City settings must be numbers")
    if raw.get("class_method"):
        s["class_method"] = raw["class_method"]
    if s["class_method"] not in ("log", "count"):
        raise core.RenderError("Unknown class method (expected 'log' or 'count')")
    for k in ("name_set", "name_lang"):
        if raw.get(k):
            s[k] = str(raw[k]).strip().lower()
    if s["name_set"] not in ("endonym", "exonym"):
        raise core.RenderError("Unknown name set (expected 'endonym' or 'exonym')")
    if not (s["name_lang"].isalpha() and 2 <= len(s["name_lang"]) <= 3):
        raise core.RenderError("The exonym language must be a 2-3 letter code (en, fr, de, ...)")
    for k, (lo, hi) in CITY_LIMITS.items():
        if not lo <= s[k] <= hi:
            raise core.RenderError(f"'{k}' must be between {lo:g} and {hi:g}")
    for t, v in s["counts"].items():
        if not 0 <= v <= MAX_COUNT_PER_COUNTRY:
            raise core.RenderError(f"{t}: the number of cities must be between 0 and {MAX_COUNT_PER_COUNTRY}")
    return s


def _mean_nn(pts) -> float:
    """Average distance from each point to its nearest other point (inf for
    fewer than 2)."""
    if len(pts) < 2:
        return float("inf")
    return float(cKDTree(pts).query(pts, k=2)[0][:, 1].mean())


def _apportion(total: int, weights) -> np.ndarray:
    """Split `total` into whole numbers proportional to `weights` (largest
    remainders: the parts add up to `total` exactly)."""
    w = np.asarray(weights, dtype=np.float64)
    if total <= 0 or w.sum() <= 0:
        return np.zeros(len(w), dtype=np.int64)
    exact = total * w / w.sum()
    out = np.floor(exact).astype(np.int64)
    rest = int(total - out.sum())
    if rest > 0:
        out[np.argsort(-(exact - out), kind="stable")[:rest]] += 1
    return out


def _isolated(pts, radius: float, comp=None):
    """_isolated_one per land area (`comp`): a city across the sea doesn't
    count as a larger neighbour (Saint Helier on Jersey vs Cherbourg)."""
    if comp is None:
        return _isolated_one(pts, radius)
    out = np.zeros(len(pts), dtype=bool)
    for c in np.unique(comp).tolist():
        k = np.flatnonzero(comp == c)  # keeps the largest-first order
        out[k] = _isolated_one(pts[k], radius)
    return out


def _isolated_one(pts, radius: float):
    """For points sorted largest city first: True where no earlier (more
    populated) point lies within `radius`. Divide and conquer: each half's
    points are checked against the earlier half with a KD-tree, so every
    pair is tested once, vectorized (a grid-hash scan in Python took 3.5 s
    on a Europe view; this ~0.2 s)."""
    n = len(pts)
    out = np.ones(n, dtype=bool)
    if radius <= 0 or n == 0:
        return np.zeros(n, dtype=bool)
    r2 = radius * radius

    def rec(lo, hi):
        if hi - lo <= 64:
            p = pts[lo:hi]
            d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
            near = np.tril(d2 < r2, k=-1)  # [i, j]: j earlier than i
            out[lo:hi] &= ~near.any(axis=1)
            return
        mid = (lo + hi) // 2
        rec(lo, mid)
        rec(mid, hi)
        d, _i = cKDTree(pts[lo:mid]).query(pts[mid:hi], distance_upper_bound=radius)
        out[mid:hi] &= ~(d < radius)

    rec(0, n)
    return out


def _apportion_capped(total: int, weights, cap) -> np.ndarray:
    """_apportion, but no part above its `cap` (cities available): what a
    capped part can't take goes to the others, by their weights."""
    w = np.asarray(weights, dtype=np.float64).copy()
    cap = np.asarray(cap, dtype=np.int64)
    out = np.zeros(len(w), dtype=np.int64)
    left = int(total)
    free = np.ones(len(w), dtype=bool)
    while left > 0 and free.any():
        if w[free].sum() <= 0:
            w[free] = 1.0  # no weight left: share evenly between the rest
        part = np.zeros(len(w), dtype=np.int64)
        part[free] = _apportion(left, w[free])
        over = free & (out + part > cap)
        if not over.any():
            out += part
            break
        left -= int((cap[over] - out[over]).sum())
        out[over] = cap[over]
        free &= ~over
    return out


def _region_weights(groups_n, rid_cand, cand_pop, ix, iy, m_rid, m_group, gid_img, land_mask, a, step=4):
    """Regions (GeoNames admin1 of each city in view): the country each
    belongs to (most of its cities) and its weight within that country --
    the same land / population blend as between countries. A region's land
    = the country's land pixels whose nearest city in view (any size) is in
    that region (on a 1/`step` grid); its population = the sum of its
    places in view. Returns (region -> country, weight)."""
    n_r = int(rid_cand.max()) + 1 if len(rid_cand) else 0
    if not n_r:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    pair = np.bincount(m_rid * groups_n + m_group, minlength=n_r * groups_n).reshape(n_r, groups_n)
    r_group = np.where(pair.sum(axis=1) > 0, pair.argmax(axis=1), -1)
    h, w = gid_img.shape
    lab = np.zeros(((h + step - 1) // step, (w + step - 1) // step), dtype=np.int32)
    lab[iy // step, ix // step] = rid_cand + 1
    _d, (ny, nx) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    near = lab[ny, nx] - 1
    land_c = land_mask[::step, ::step] & (near >= 0)
    rr, gg = near[land_c], gid_img[::step, ::step][land_c]
    match = r_group[rr] == gg
    r_land = np.bincount(rr[match], minlength=n_r).astype(np.float64)
    r_pop = np.bincount(rid_cand, weights=cand_pop.astype(np.float64), minlength=n_r)
    r_w = np.zeros(n_r)
    for g in np.unique(r_group[r_group >= 0]).tolist():
        R = np.flatnonzero(r_group == g)
        lf = r_land[R] / r_land[R].sum() if r_land[R].sum() > 0 else np.full(len(R), 1.0 / len(R))
        pf = r_pop[R] / r_pop[R].sum() if r_pop[R].sum() > 0 else lf
        r_w[R] = (1.0 - a) * lf + a * pf
    return r_group, r_w


def _split_regions(quota, r_group, r_w, cap):
    """Each country's number split between its regions by weight, none
    above its cap (cities it can supply); the rest goes to the others."""
    r_quota = np.zeros(len(r_group), dtype=np.int64)
    for g in np.flatnonzero(quota > 0).tolist():
        R = np.flatnonzero(r_group == g)
        if len(R):
            r_quota[R] = _apportion_capped(int(quota[g]), r_w[R], cap[R])
    return r_quota


def _spaced_pick(pts, group, quota, min_dist: float, exempt=None, force=None, comp=None):
    """Indices of the points (in order, largest city first) picked while
    their group (country) is under its `quota` (array per group) and each
    is at least `min_dist` from every point picked before (any group; grid
    hashing, cells of min_dist) -- `exempt` points only need
    EXEMPT_FLOOR_PX; `force` points are picked even when their group's
    quota is full (then as extras, not counted in it; while there's room
    they use the quota like any city). With `comp` (land area of each
    point), the distance only counts between points on the same land area.
    Returns (picked indices, extra mask over them)."""
    n = len(pts)
    exempt = np.zeros(n, dtype=bool) if exempt is None else exempt
    force = np.zeros(n, dtype=bool) if force is None else force
    comp = np.zeros(n, dtype=np.int64) if comp is None else comp
    # Nothing left to pick once every quota is full and no forced point follows.
    force_after = np.r_[np.logical_or.accumulate(force[::-1])[::-1], False]
    taken = np.zeros(len(quota), dtype=np.int64)
    left = int(np.sum(quota))
    cells = {}
    d2 = min_dist * min_dist
    floor2 = min(EXEMPT_FLOOR_PX, min_dist) ** 2
    picked, extra = [], []
    comp_l = comp.tolist()
    for i, ((x, y), g) in enumerate(zip(pts.tolist(), group.tolist())):
        if left <= 0 and not force_after[i]:
            break
        if taken[g] >= quota[g] and not force[i]:
            continue
        if min_dist > 0:
            gx, gy = int(x // min_dist), int(y // min_dist)
            lim = floor2 if exempt[i] else d2
            ci = comp_l[i]
            if any(qc == ci and (x - qx) ** 2 + (y - qy) ** 2 < lim
                   for cx in (gx - 1, gx, gx + 1) for cy in (gy - 1, gy, gy + 1)
                   for qx, qy, qc in cells.get((cx, cy), ())):
                continue
            cells.setdefault((gx, gy), []).append((x, y, ci))
        picked.append(i)
        if taken[g] < quota[g]:
            taken[g] += 1
            left -= 1
            extra.append(False)
        else:
            extra.append(True)
    return np.array(picked, dtype=np.int64), np.array(extra, dtype=bool)


def _country_groups(arr, mask, country_rgb):
    """Per-pixel country group of the render (one per colour: a merged
    autonomous territory shares its parent's) -> (group index image over
    `mask` (-1 elsewhere), [tags per group], land px per group)."""
    import palettes  # local import, as in core
    by_rgb = {}
    for tag, rgb in country_rgb.items():
        by_rgb.setdefault(tuple(rgb), []).append(tag)
    groups = sorted(by_rgb.values(), key=lambda ts: sorted(ts))
    # The parent (not an autonomous territory) first: it names the group.
    groups = [sorted(ts, key=lambda t: (t in palettes.AUTONOMOUS_PARENT, t)) for ts in groups]
    lut = np.full(1 << 24, -1, dtype=np.int16)
    for g, ts in enumerate(groups):
        r, gg, b = country_rgb[ts[0]]
        lut[(r << 16) | (gg << 8) | b] = g
    key = (arr[..., 0].astype(np.int32) << 16) | (arr[..., 1].astype(np.int32) << 8) | arr[..., 2]
    gid = np.where(mask, lut[key], -1)
    land = np.bincount(gid[gid >= 0].ravel(), minlength=len(groups))
    return gid, groups, land


def _class_limits(pop, k: int, method: str):
    """Lower population limits of classes 1..k-1 (descending) from a
    continent's selected cities: equal steps in log(population) between its
    smallest and largest ("log"), or every n/k-th city ("count")."""
    if not len(pop) or k <= 1:
        return np.zeros(0)
    if method == "count":
        srt = np.sort(pop)[::-1]
        return np.array([srt[min(len(srt) - 1, max(0, -(-j * len(srt) // k) - 1))] for j in range(1, k)],
                        dtype=float)
    lo, hi = np.log(max(int(pop.min()), 1)), np.log(max(int(pop.max()), 1))
    if hi <= lo:
        return np.full(k - 1, np.inf)  # a single population value: one class (see _classes)
    return np.exp(hi - np.arange(1, k) * (hi - lo) / k)


def _classes(pop, limits, k: int):
    """Class per city (1 = largest: at least limits[0]) and the per-class
    counts / population ranges."""
    if len(limits) and np.isinf(limits).all():
        cls = np.ones(len(pop), dtype=np.int64)
    else:
        cls = 1 + (pop[:, None] < limits[None, :] * (1 - 1e-9)).sum(axis=1).astype(np.int64)
    bounds = []
    for c in range(1, k + 1):
        p = pop[cls == c]
        bounds.append({"class": c, "count": int(len(p)),
                       "min_pop": int(p.min()) if len(p) else None, "max_pop": int(p.max()) if len(p) else None})
    return cls, bounds


def _star_sprite(size: int, color) -> Image.Image:
    """A five-pointed star, `size` px across, drawn at 4x and downsampled
    (antialiased), with a thin dark outline."""
    ss = 4
    big = size * ss
    im = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    c = big / 2.0
    r_out, r_in = big / 2.0 - 0.5, big / 2.0 * 0.45
    pts = []
    for i in range(10):
        r = r_out if i % 2 == 0 else r_in
        a = -np.pi / 2 + i * np.pi / 5
        pts.append((c + r * np.cos(a), c + r * np.sin(a) + big * 0.04))
    ImageDraw.Draw(im).polygon(pts, fill=tuple(color) + (255,), outline=STAR_OUTLINE + (255,), width=ss)
    return im.resize((size, size), Image.LANCZOS)


def _country_name(tag: str) -> str:
    g = core.load_geometries(progress=lambda *a: None)
    return g[tag]["name"] if tag in g else tag


def _region_report(g, r_quota, r_group, r_names, m_rid, m_cand, picked, is_extra, idx):
    """A country's regions with their number, cities shown and largest city
    in view (GeoNames has no admin1 names in this export: the largest city
    names the region), largest number first."""
    if r_quota is None:
        return []
    out = []
    for r in np.flatnonzero(r_group == g).tolist():
        in_r = m_rid == r
        if not in_r.any():
            continue
        n_sel, n_iso = int((picked & in_r & ~is_extra).sum()), int((picked & in_r & is_extra).sum())
        if not (r_quota[r] or n_sel or n_iso):
            continue  # nothing for it (Isle of Man / Channel Islands parishes under the UK ...)
        first = int(np.flatnonzero(in_r)[0])  # m_* are sorted largest city first
        out.append({"code": str(r_names[r]).split(".", 1)[1], "city": idx["name"][m_cand[first]],
                    "quota": int(r_quota[r]), "selected": n_sel, "isolated": n_iso})
    out.sort(key=lambda d: (-d["quota"], -d["selected"]))
    return out


LABEL_TEXT = (25, 25, 25)
LABEL_HALO = (245, 245, 245)
_FONT_FILES = ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf")
_fonts = {}


def _font(size: int):
    """A TrueType font with Latin Extended (Windows' Arial, else fallbacks),
    memoized per size."""
    if size not in _fonts:
        font = None
        for name in _FONT_FILES:
            try:
                font = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        _fonts[size] = font if font is not None else ImageFont.load_default(size)
    return _fonts[size]


def _place_labels(img, cities_px, star_px: int):
    """Draw labels [(x, y, text, size)] (already in priority order) at the
    first free spot around each star; returns (placed, unplaced)."""
    h, w = img.height, img.width
    occ = np.zeros((h, w), dtype=bool)
    half = star_px // 2
    for x, y, _t, _s in cities_px:
        occ[max(0, y - half):y + half + 1, max(0, x - half):x + half + 1] = True
    draw = ImageDraw.Draw(img)
    placed = unplaced = 0
    gap = half + 2
    for x, y, text, size in cities_px:
        font = _font(size)
        halo = max(1, size // 7)
        l, t, r, b = draw.textbbox((0, 0), text, font=font, stroke_width=halo)
        tw, th = r - l, b - t
        spots = [(x + gap, y - th // 2), (x - gap - tw, y - th // 2), (x - tw // 2, y - gap - th),
                 (x - tw // 2, y + gap), (x + gap, y - gap - th), (x + gap, y + gap), (x - gap - tw, y - gap - th),
                 (x - gap - tw, y + gap)]
        for sx, sy in spots:
            if sx < 0 or sy < 0 or sx + tw > w or sy + th > h:
                continue
            if occ[sy:sy + th, sx:sx + tw].any():
                continue
            draw.text((sx - l, sy - t), text, font=font, fill=LABEL_TEXT, stroke_width=halo, stroke_fill=LABEL_HALO)
            occ[max(0, sy - 1):sy + th + 1, max(0, sx - 1):sx + tw + 1] = True
            placed += 1
            break
        else:
            unplaced += 1
    return placed, unplaced


def render_cities(image: Image.Image, debug, settings=None):
    """(RGB map, report) for a render (core.render_map_detailed's image and
    debug) -- see the module docstring. report["cities"] lists the drawn
    cities (for the CSV export), largest first."""
    s = normalize_settings(settings or {})
    arr = np.asarray(image)
    bg = (arr == np.asarray(core.BACKGROUND_COLOR, dtype=arr.dtype)).all(axis=2)
    hidden = debug.get("hidden_land")
    lakes = debug.get("lakes")
    water = bg.copy()
    if hidden is not None:
        water &= ~hidden
    if lakes is not None:
        water |= lakes
    land = ~water
    visible = land & ~bg  # drawn land of visible countries (lakes are LAKE_COLOR, not bg)
    if lakes is not None:
        visible &= ~lakes
    out = np.empty_like(arr)
    out[:] = LAND_COLOR
    out[water] = WATER_COLOR
    report = {"settings": s, "available": available(),
              "continents": [{"code": c, "name": n, "in_view": 0, "selected": 0} for c, n in CONTINENTS],
              "class_colors": [core.rgb_to_hex(CLASS_COLORS[i]) for i in range(s["classes"])],
              "land_color": core.rgb_to_hex(LAND_COLOR), "water_color": core.rgb_to_hex(WATER_COLOR), "cities": []}
    if not report["available"]:
        return Image.fromarray(out, mode="RGB"), report
    idx = load_city_index()

    v = debug["view"]
    h, w, scale = v["height"], v["width"], v["scale"]
    func = core.PROJECTIONS[v["projection"]]["func"]
    x, y = func(core._relative_lon(idx["lon"], v["center"]), idx["lat"])
    px, py = (x - v["win_xmin"]) * scale, h - (y - v["win_ymin"]) * scale
    ix, iy = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
    cand = np.flatnonzero((ix >= 0) & (ix < w) & (iy >= 0) & (iy < h) & (idx["population"] >= s["min_population"]))
    ix, iy = ix[cand], iy[cand]
    # On a visible country's land, or moved onto it from up to SNAP_PX away.
    on = visible[iy, ix]
    snapped = 0
    if (~on).any() and visible.any():
        dist, (ny, nx) = ndimage.distance_transform_edt(~visible, return_indices=True)
        off = np.flatnonzero(~on)
        near = off[dist[iy[off], ix[off]] <= SNAP_PX]
        iy[near], ix[near] = ny[iy[near], ix[near]], nx[iy[near], ix[near]]
        on[near] = True
        snapped = int(len(near))
    cand, ix, iy = cand[on], ix[on], iy[on]
    report["in_view"], report["snapped"] = int(len(cand)), snapped

    # Small islands: only their largest city, set aside from the density
    # selection (their land isn't counted in their country's either).
    island_of = np.zeros(len(cand), dtype=bool)
    island_pick = np.zeros(len(cand), dtype=bool)
    small_land = np.zeros_like(visible)
    comp = np.zeros(len(cand), dtype=np.int64)  # land area (8-connected) of each city
    if len(cand):
        lab, n_lab = ndimage.label(land, structure=np.ones((3, 3), dtype=bool))
        comp = lab[iy, ix].astype(np.int64)
    if s["island_px"] > 0 and len(cand):
        small = np.bincount(lab.ravel(), minlength=n_lab + 1) < s["island_px"]
        small[0] = False
        small_land = small[lab]
        island_of = small[comp]
        ii = np.flatnonzero(island_of)
        if len(ii):
            order = ii[np.lexsort((-idx["population"][cand[ii]], comp[ii]))]  # by island, largest city first
            first = np.r_[True, comp[order][1:] != comp[order][:-1]]
            island_pick[order[first]] = True
    report["island_cities"] = int(island_pick.sum())
    report["island_cities_hidden"] = int(island_of.sum() - island_pick.sum())

    chosen, main_all = [], []
    main = np.flatnonzero(~island_of)
    main = main[np.argsort(-idx["population"][cand[main]], kind="stable")]  # largest first
    m_cand, m_pop = cand[main], idx["population"][cand[main]]
    m_pts = np.column_stack([ix[main], iy[main]]).astype(np.float64)
    m_comp = comp[main]
    m_cont = idx["continent"][m_cand]
    gid_img, groups, group_land = _country_groups(arr, visible & ~small_land, debug["country_rgb"])
    m_group = gid_img[iy[main], ix[main]]
    ok = m_group >= 0  # a city snapped onto a hidden country's edge etc.
    m_cand, m_pop, m_pts, m_cont, m_group, m_comp = (m_cand[ok], m_pop[ok], m_pts[ok], m_cont[ok], m_group[ok],
                                                     m_comp[ok])
    # Region (GeoNames admin1) of every city in view.
    rkeys = np.array([f"{c}.{r}" for c, r in zip(idx["country"][cand], idx["admin1"][cand])], dtype=object)
    r_names, rid_cand = np.unique(rkeys, return_inverse=True) if len(cand) else (np.zeros(0, dtype=object),
                                                                                np.zeros(0, dtype=np.int64))
    rid_cand = rid_cand.astype(np.int64)
    m_rid = rid_cand[main][ok]
    # The target shared out by land; the user's overrides replace a share.
    # Population in view per country: its places' populations (the land's
    # small islands included -- only their largest city is drawn, but their
    # people count).
    isl_group = gid_img[iy[island_of], ix[island_of]] if island_of.any() else np.zeros(0, dtype=np.int64)
    isl_pop = idx["own_population"][cand[island_of]]
    group_pop = (np.bincount(m_group, weights=idx["own_population"][m_cand].astype(np.float64), minlength=len(groups))
                 + np.bincount(isl_group[isl_group >= 0], weights=isl_pop[isl_group >= 0].astype(np.float64),
                               minlength=len(groups)))
    group_pop[group_land <= 0] = 0.0
    a = s["pop_weight"] / 100.0
    land_frac = group_land / max(group_land.sum(), 1)
    pop_frac = group_pop / group_pop.sum() if group_pop.sum() > 0 else land_frac
    share = _apportion(s["target"], (1.0 - a) * land_frac + a * pop_frac)
    quota = share.copy()
    overridden = np.zeros(len(groups), dtype=bool)
    tag_of = [ts[0] for ts in groups]
    for g, ts in enumerate(groups):
        for t in ts:  # an override may name the parent or a merged territory
            if t in s["counts"] and group_land[g] > 0:
                quota[g] = s["counts"][t]
                overridden[g] = True
                break

    # Distance exception for the top classes. Their (per continent) limits
    # depend on the selection, which depends on the exception: start
    # without it, re-pick with the picked cities' limits until the exempt
    # set is stable (1-3 rounds).
    n_ex = min(s["dist_exempt_classes"], s["classes"] - 1)  # the lowest class is never exempt
    exempt = None
    # Towns inside a larger city's urban area aren't picked (they still
    # counted in the weights above).
    report["in_urban_area"] = 0
    if s["urban_areas"] and len(m_cand):
        own = idx["urban_of"][m_cand] < 0
        report["in_urban_area"] = int((~own).sum())
        m_cand, m_pop, m_pts, m_cont, m_group, m_rid, m_comp = (m_cand[own], m_pop[own], m_pts[own], m_cont[own],
                                                                 m_group[own], m_rid[own], m_comp[own])
    # Within each country, its number split between its regions.
    if s["by_region"] and len(m_cand):
        r_group, r_w = _region_weights(len(groups), rid_cand, idx["own_population"][cand], ix, iy, m_rid, m_group,
                                       gid_img, visible & ~small_land, a)
        r_cap = np.bincount(m_rid, minlength=len(r_group))
        r_quota = _split_regions(quota, r_group, r_w, r_cap)
        pick_group, pick_quota = m_rid, r_quota
    else:
        r_quota = r_group = None
        pick_group, pick_quota = m_group, quota
    iso = _isolated(m_pts, s["isolated_px"], m_comp)
    is_extra = np.zeros(len(m_cand), dtype=bool)  # isolated cities beyond their country's number
    keep, ext = _spaced_pick(m_pts, pick_group, pick_quota, s["min_dist_px"], force=iso, comp=m_comp)
    is_extra[keep[ext]] = True
    for _round in range(5):
        if not (n_ex and len(keep)) or s["min_dist_px"] <= 0:
            break
        new = np.zeros(len(m_cand), dtype=bool)
        for code, _n in CONTINENTS:
            kc = keep[(m_cont[keep] == code) & ~is_extra[keep]]
            if not len(kc):
                continue
            lim = _class_limits(m_pop[kc], s["classes"], s["class_method"])
            floor_pop = lim[n_ex - 1] if n_ex <= len(lim) else 0
            if np.isfinite(floor_pop):
                new |= (m_cont == code) & (m_pop >= floor_pop * (1 - 1e-9))
        if exempt is not None and (new == exempt).all():
            break
        exempt = new
        keep, ext = _spaced_pick(m_pts, pick_group, pick_quota, s["min_dist_px"], exempt=exempt, force=iso,
                                 comp=m_comp)
        is_extra[:] = False
        is_extra[keep[ext]] = True
    # A region that couldn't fill its number (its other cities all too
    # close to picked ones -- Moscow oblast around Moscow) passes the rest to
    # its country's other regions: cap it at what it took, re-split, re-pick.
    for _round in range(3):
        if r_quota is None:
            break
        taken = np.bincount(m_rid[keep[~ext]], minlength=len(r_quota))
        short_r = taken < r_quota
        if not short_r.any():
            break
        r_cap = np.where(short_r, taken, r_cap)
        new_quota = _split_regions(quota, r_group, r_w, r_cap)
        if (new_quota == r_quota).all():
            break
        r_quota = pick_quota = new_quota
        keep, ext = _spaced_pick(m_pts, pick_group, pick_quota, s["min_dist_px"], exempt=exempt, force=iso,
                                 comp=m_comp)
        is_extra[:] = False
        is_extra[keep[ext]] = True
    picked = np.zeros(len(m_cand), dtype=bool)
    picked[keep] = True

    # Per country (every visible one, for the tab's "Visible countries" list).
    report["countries"] = []
    for g, ts in enumerate(groups):
        if group_land[g] <= 0:
            continue
        in_g = m_group == g
        shown = picked & in_g
        report["countries"].append({
            "tag": tag_of[g], "name": _country_name(tag_of[g]),
            "color": core.rgb_to_hex(debug["country_rgb"][tag_of[g]]), "land_px": int(group_land[g]),
            "share": int(share[g]), "quota": int(quota[g]), "override": bool(overridden[g]),
            "population": int(group_pop[g]),
            "in_view": int(in_g.sum()), "selected": int(shown.sum()),
            "isolated": int((shown & is_extra).sum()),
            "regions": _region_report(g, r_quota, r_group, r_names, m_rid, m_cand, picked, is_extra, idx),
            "min_pop": int(m_pop[shown & ~is_extra].min()) if (shown & ~is_extra).any() else None})
    report["countries"].sort(key=lambda c: c["name"])
    report["target"] = s["target"]
    report["assigned"] = int(sum(c["quota"] for c in report["countries"]))
    report["short"] = int(sum(max(0, c["quota"] - (c["selected"] - c["isolated"])) for c in report["countries"]))
    report["isolated"] = int(is_extra.sum())

    for info in report["continents"]:
        in_c = idx["continent"][cand] == info["code"]
        info["in_view"] = int(in_c.sum())
        isl = in_c & island_pick
        info["island_cities"] = int(isl.sum())
        if not in_c.any():
            continue
        mc = m_cont == info["code"]
        main_sel, main_pts = m_cand[picked & mc], m_pts[picked & mc]
        info["distance_exempt"] = int((exempt & picked & mc).sum()) if exempt is not None else 0
        info["isolated"] = int((is_extra & picked & mc).sum())
        main_all.append((main_pts, idx["population"][main_sel]))
        # Class limits from the picked cities (else the island cities),
        # applied to the island cities too.
        q_pop = m_pop[picked & mc & ~is_extra]
        base = q_pop if len(q_pop) else (idx["population"][main_sel] if len(main_sel) else idx["population"][cand[isl]])
        limits = _class_limits(base, s["classes"], s["class_method"])
        sel = np.concatenate([main_sel, cand[isl]])
        pts = np.concatenate([main_pts, np.column_stack([ix[isl], iy[isl]]).astype(np.float64)])
        if not len(sel):
            continue
        pop = idx["population"][sel]
        cls, bounds = _classes(pop, limits, s["classes"])
        nn = _mean_nn(main_pts)
        # min_pop: of the picked cities (island cities may be far smaller)
        info.update(selected=int(len(sel)), min_pop=int(base.min()),
                    max_pop=int(pop.max()), nearest_px=None if not np.isfinite(nn) else round(nn, 1), classes=bounds)
        chosen.append((sel, pts, cls, info["code"]))
    report["island_cities_near_city"] = 0  # small-island cities are never dropped (other land area)
    chosen = [c for c in chosen if len(c[0])]
    report["selected"] = int(sum(len(c[0]) for c in chosen))

    # Stars, most populated first (user): each star only fills what the
    # larger ones drawn before it left uncovered (front-to-back "under"
    # compositing, antialiased edges included), so a smaller city can never
    # cover any part of a larger one's star.
    size = s["star_px"] | 1  # odd: centred on the city's pixel
    sprites = [np.asarray(_star_sprite(size, CLASS_COLORS[i]), dtype=np.float32) / 255.0
               for i in range(s["classes"])]
    canvas = out.copy()
    cover = np.zeros(out.shape[:2], dtype=np.float32)  # star coverage so far, 0..1 (blended per star window)
    if chosen:
        sel = np.concatenate([c[0] for c in chosen])
        pts = np.concatenate([c[1] for c in chosen]).astype(np.int64)
        cls = np.concatenate([c[2] for c in chosen])
        cont = np.concatenate([np.full(len(c[0]), c[3]) for c in chosen])
        pop = idx["population"][sel]
        half = size // 2
        for i in np.argsort(-pop, kind="stable").tolist():
            x0, y0 = int(pts[i, 0]) - half, int(pts[i, 1]) - half
            sx0, sy0 = max(0, -x0), max(0, -y0)
            sx1, sy1 = min(size, w - x0), min(size, h - y0)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            sp = sprites[cls[i] - 1][sy0:sy1, sx0:sx1]
            ys, xs = slice(y0 + sy0, y0 + sy1), slice(x0 + sx0, x0 + sx1)
            a = sp[..., 3] * (1.0 - cover[ys, xs])  # only the part still uncovered
            win = canvas[ys, xs].astype(np.float32) / 255.0
            canvas[ys, xs] = np.clip((win * (1.0 - a[..., None]) + sp[..., :3] * a[..., None]) * 255.0 + 0.5,
                                     0, 255).astype(np.uint8)
            cover[ys, xs] += a
    img = Image.fromarray(canvas, mode="RGB")
    # Names: of every drawn city (the CSV gets them all), on the map if asked.
    report["names"] = dict(city_names.build_status(), labels=s["labels"], placed=0, unplaced=0)
    labels = {}
    if chosen:
        names = city_names.load_names(lambda: idx["id"].tolist(), CITIES_CSV_PATH)
        report["names"].update(city_names.build_status())
        for i in range(len(sel)):
            j = int(sel[i])
            gid, cc, main_name = idx["id"][j], idx["country"][j], idx["name"][j]
            endo = city_names.label_for(names, gid, cc, main_name, "endonym", "", False)
            exo = city_names.label_for(names, gid, cc, main_name, "exonym", s["name_lang"], False)
            shown_name = endo if s["name_set"] == "endonym" else exo
            labels[i] = (city_names.to_ascii(shown_name) if s["ascii_only"] else shown_name, endo, exo)
        if s["labels"]:
            k = s["classes"]
            todo = [(int(pts[i, 0]), int(pts[i, 1]), labels[i][0],
                     int(round(s["label_px"] * (1 + 0.15 * (k - int(cls[i]))))))
                    for i in np.argsort(-pop, kind="stable").tolist()]
            report["names"]["placed"], report["names"]["unplaced"] = _place_labels(img, todo, size)
    if chosen:
        for i in np.lexsort((-pop, cls)).tolist():
            j = int(sel[i])
            report["cities"].append({"geoname_id": idx["id"][j], "name": idx["name"][j], "country": idx["country"][j],
                                     "continent": str(cont[i]), "class": int(cls[i]), "population": int(pop[i]),
                                     "feature_code": idx["feature"][j], "lon": float(idx["lon"][j]),
                                     "lat": float(idx["lat"][j]), "x": int(pts[i, 0]), "y": int(pts[i, 1]),
                                     "merged_parts": int(idx["parts"][j]),
                                     "own_population": int(idx["own_population"][j]),
                                     "label": labels[i][0], "endonym": labels[i][1], "exonym": labels[i][2],
                                     "alternate_names": idx["alt"][j]})
    return img, report
