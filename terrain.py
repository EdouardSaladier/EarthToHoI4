"""Hearts of Iron IV terrain.bmp from the Copernicus land cover
(landcover.py) + the NASA heightmap, on the Countries tab's frame.

Palette indices and terrain types follow the game's terrain.bmp /
00_terrain.txt (TERRAIN_CLASSES). The look follows the user's example map
(example.png, a Europe + North Africa map they made earlier from the same
data), calibrated on it: example.png lines up with our Miller render of
ISL,PRT,FIN,TUR,TUN,UKR,GEO at scale 1.0348, offset (-512, -156) px, and
its classes were cross-tabulated against the land cover and local relief
(_dev/terrain_align.py, _dev/terrain_calib_*.py; numbers below). The game tells desert and temperate
relief apart only by texture: both are hills / mountain types.

Pipeline (render_terrain):
  1. Water = the country map's sea + lakes -> ocean (15). Everything else is
     land (hidden countries included), whatever the land cover says there:
     land-cover water / sea / no data on land takes the nearest land class.
  2. Local shares (window MIX_WINDOW_PX) of forest and cropland in the land
     cover. The example isn't a per-class mapping: where cropland is 100%,
     only ~30% of its pixels are farmland (the rest plains); dense forest is
     ~72% forest, ~10% open forest, ~14% plains; open forest peaks (~22%)
     at half forest. So each pixel draws its class from those measured odds
     (FOREST_ODDS, FARMLAND_ODDS; a fixed-seed noise, same result every
     render) -> the example's speckle. Farmland draws from patchy noise
     (farmland_patch_px; odds unchanged) -> larger farmland / plains patches. Forest borders are then smoothed
     (median filter, forest_px).
  3. Other land cover: wetland -> marsh (9) -- plus, with small_lakes_marsh,
     every lake the Land/Sea step drew but didn't keep (too small) -- tropical evergreen broad-leaf
     forest -> jungle (22, dense core 21), snow -> snowy plains (19), bare:
       - cold (|lat| >= cold_lat: arctic barrens, Iceland, Svalbard) ->
         snowy plains (19), since HOI4 has no cold desert;
       - hot (|lat| < cold_lat): hot desert, per pixel (user: the land
         cover's bare / sparse vegetation *is* the desert) -> sand (7) with
         ~35% rocky desert (3) in clumps of ~desert_smooth_px (as in the
         example, independent of height or latitude). A 40-degree limit
         was tried and dropped: the Kyzylkum, Ustyurt, Aral Karakum and the
         Gobi (40-48 N) came out as plain rocky desert without transition;
       - grass <-> desert transition (FF00F0, index 12, user): grass /
         shrubs / moss where hot bare land covers part of the surroundings
         (share within transition_km), speckled, odds rising with it up to
         TRANSITION_MAX_ODDS right at the edge;
  4. Relief = elevation range (max - min) within relief_km (default 1 km), circular:
     Paint.NET's Outline effect at 100% intensity, which the user ran over
     the heightmap for example.png (then brightness / contrast = the
     thresholds). Never the elevation itself. Tested on example.png (1.42
     km/px): circle r 4-5 px at 100% agrees with 79% of its relief classes,
     r 2 px 77% (lower Outline intensities worse), but the larger radius
     dilates peaks into round blobs; the user wants granular, detailed
     relief, so 2.5 km (~2 px there), thresholds quantile-matched to the
     example's class shares at that radius. Heights = the Heightmap tab's,
     lower ranges exaggerated (user; core.exaggerated_height, default x2):
     thresholds re-matched on them (_dev/terrain_exag_calib.py: 182 / 414 /
     846 m at x2, 132 / 265 / 530 at x1, same 76% agreement). Temperate:
     hills >= hill_m (default 180 m; 2, outlined in 17 where they meet other land),
     "mountains (grass)" >= low_mountain_m (350 m; 20), mountains >=
     mountain_m (600 m; 11; 16 on snow / ice). Desert relief (user): the
     same tiers and thresholds everywhere; every relief pixel lying on hot
     bare land gets the desert texture of its tier (user's indices: hills
     493B0F = 8, their outline AE00FF = 10, mountains (grass) and mountains
     both F3C793 = 31) -- where bare land covers
     >= DESERT_RELIEF_SHARE of the land within DESERT_RELIEF_KM, so bare
     rock above the tree line isn't desert (measured bare share within 10
     km: Alps 0.13-0.15, Caucasus 0.19, Pyrenees 0.04; Karakum 0.27,
     Kyzylkum 0.43, Zagros 0.95, Gobi / Taklamakan / Sahara 1.0) -- with no smoothing, so desert ranges follow the bare
     ground pixel by pixel. A high plateau is flat locally (Tibet's interior,
     Anatolia, the Meseta, the Altiplano), so it isn't mountainous; its
     escarpments are. Kept granular (user: detailed mountains / hills,
     not flat blobs): no smoothing by default (relief_smooth_px = median
     filter, 1 = off), only specks under RELIEF_MIN_PX merged -- except
     mountains: smoothing / merging never lowers a mountain or mountain
     (grass) pixel (user: mountains always on top of hills). Relief
     overrides the land cover except cities (and forest on hills with
     forest_over_hills).
  5. Cities: land-cover urban smoothed into blobs (urban share within
     city_px >= CITY_SHARE; city_px 1 = no smoothing), then grown outwards
     by city_grow_px -> urban (13). Plus the City tab's cities of classes
     1-3 (its current selection and classes): a disc of city_disc_r1/2/3 px
     radius around each (0 = none for that class), on land -> urban.
"""
import time

import numpy as np
from PIL import Image
from scipy import ndimage, special

import core
import landcover

# (index, name, HOI4 terrain type) -- indices and colours from the game's terrain.bmp.
TERRAIN_CLASSES = [
    (0, "Plains", "plains"), (1, "Forest", "forest"), (2, "Hills", "hills"), (3, "Desert (rocky)", "desert"),
    (4, "Forest (open)", "forest"), (5, "Farmland", "plains"), (7, "Desert (sand)", "desert"),
    (8, "Desert hills", "desert"), (9, "Marsh", "marsh"), (10, "Desert hills (blend)", "mountain"),
    (11, "Mountains", "mountain"), (12, "Grass / desert transition", "desert"), (13, "Urban", "urban"),
    (15, "Ocean / lakes", "ocean"), (16, "Snowy mountains", "mountain"), (17, "Hills (blend)", "hills"),
    (19, "Snowy plains", "plains"), (20, "Mountains (grass)", "mountain"), (21, "Jungle", "jungle"),
    (22, "Jungle (blend)", "jungle"), (31, "Desert mountains", "mountain"),
]
_PAL = {0: (86, 124, 27), 1: (0, 86, 6), 2: (112, 74, 31), 3: (206, 169, 99), 4: (6, 200, 11), 5: (255, 0, 24),
        6: (134, 84, 30), 7: (252, 255, 0), 8: (73, 59, 15), 9: (75, 147, 174), 10: (174, 0, 255),
        11: (92, 83, 76), 12: (255, 0, 240), 13: (240, 255, 0), 14: (55, 90, 220), 15: (8, 31, 130),
        16: (255, 255, 255), 17: (132, 255, 0), 18: (255, 126, 0), 19: (114, 137, 105), 20: (58, 131, 82),
        21: (255, 0, 127), 22: (0, 82, 82), 31: (243, 199, 147), 35: (255, 0, 252), 36: (23, 23, 23)}
TERRAIN_PALETTE = [_PAL.get(i, (i, i, i)) for i in range(256)]  # the game's palette (greys elsewhere)

# trees.bmp: the vanilla file's palette (entries 0-30; greys elsewhere) and
# size -- terrain.bmp / 3.4133 (5632x2048 -> 1650x600).
_TREE_PAL = {0: (0, 0, 0), 1: (255, 0, 0), 2: (30, 139, 109), 3: (18, 100, 78), 4: (8, 58, 44), 5: (76, 156, 51),
             6: (47, 120, 24), 7: (20, 85, 0), 8: (154, 156, 51), 9: (118, 120, 24), 10: (83, 85, 0),
             11: (255, 255, 0), 12: (213, 160, 0), 13: (0, 183, 0), 14: (0, 128, 0), 15: (0, 60, 0),
             18: (11, 207, 172), 19: (8, 155, 129), 20: (4, 75, 62), 27: (169, 93, 255), 28: (150, 0, 255),
             29: (88, 0, 138), 30: (252, 0, 255)}
TREES_PALETTE = [_TREE_PAL.get(i, (i, i, i)) for i in range(256)]
TREES_SCALE = (1650 / 5632, 600 / 2048)
# Tree indices (user's names): 2 sparse palm, 3 dense palm, 5 sparse
# forest, 6 dense forest, 11 impassable jungle, 28 sparse jungle, 29 dense
# jungle; 0 = none. Vanilla trees.bmp vs its terrain.bmp (cross-tabulated):
# 6 on forest (92% of its pixels), 5 on open forest, 11 on the jungle core
# (97%), 28 / 29 on jungle and tropical forest. Here straight from the land
# cover: closed / open forest, tropical evergreen broad-leaf closed / open,
# the jungle core as in the terrain. Palms aren't drawn (no land cover
# class for them).
TREE_NONE, TREE_SPARSE, TREE_DENSE, TREE_IMPASSABLE, TREE_SPARSE_JUNGLE, TREE_DENSE_JUNGLE = 0, 5, 6, 11, 28, 29
TREE_CLASSES = [(TREE_DENSE, "Dense forest"), (TREE_SPARSE, "Sparse forest"), (TREE_IMPASSABLE, "Impassable jungle"),
                (TREE_DENSE_JUNGLE, "Dense jungle"), (TREE_SPARSE_JUNGLE, "Sparse jungle")]
TREE_MIN_SHARE = 0.5          # a trees.bmp pixel has trees where forest covers this share of its terrain pixels ...
TREE_SHARE_RANGE = (0.1, 1.0)  # ... moved within this range to meet tree_density
JUNGLE_HOLE_MIN_PX = 8        # trees.bmp: gaps in a jungle smaller than this don't get a sparse jungle edge
CLOSED_FOREST_CODES = (111, 112, 113, 114, 115, 116)

PLAINS, FOREST, HILLS, DESERT_ROCK, FOREST_OPEN, FARMLAND = 0, 1, 2, 3, 4, 5
DESERT_SAND, DESERT_HILLS, MARSH, DESERT_MOUNTAIN, MOUNTAIN, DESERT_STEPPE = 7, 8, 9, 10, 11, 12
URBAN, OCEAN, SNOW_MOUNTAIN, HILLS_BLEND, SNOW_PLAINS, MOUNTAIN_GRASS = 13, 15, 16, 17, 19, 20
JUNGLE, JUNGLE_BLEND, DESERT_TOPS = 21, 22, 31

TERRAIN_DEFAULTS = {         # the user's own settings (their Terrain tab, 2026-09-27)
    "hill_m": 180.0,          # local relief (m) for hills -- on the Heightmap tab's exaggerated heights (x4)
    "low_mountain_m": 350.0,  # ... for "mountains (grass)"
    "mountain_m": 600.0,      # ... for mountains
    "relief_km": 1.0,         # radius of the relief window
    "relief_smooth_px": 1,    # median filter on relief class borders (1 = off: granular, detailed relief)
    "cold_lat": 60.0,         # bare land at or beyond this latitude is cold (snowy plains)
    "city_px": 2,             # city smoothing window (cities not calibrated: the example took them from another source)
    "city_grow_px": 0,        # cities grown outwards by this many px (no smoothing involved)
    "city_discs": True,       # urban discs at the City tab's cities of classes 1-3
    "city_disc_r1": 3.0,      # disc radius (px) for class 1 cities (0 = none)
    "city_disc_r2": 2.0,      # ... class 2
    "city_disc_r3": 1.0,      # ... class 3
    "farmland_patch_px": 2,   # size of the farmland / plains patches (0 = per-pixel speckle)
    "small_lakes_marsh": True,  # lakes the Land/Sea step drew but didn't keep (too small) -> marsh
    "forest_px": 2,           # median filter on forest borders
    "forest_over_hills": False,  # forest kept where the relief says hills (mountains still win)
    "jungle_blend_px": 2,     # trees.bmp: jungle within this many px of its edge -> sparse jungle (0 = off)
    "tree_density": 75,       # trees.bmp: tree area = this % of the land cover's forest area (user: 100 too dense)
    "desert_smooth_px": 1,    # size of the sand / rocky desert clumps (0 = per-pixel speckle)
    "transition_km": 40.0,    # grass <-> desert transition (FF00F0, index 12) near hot desert; 0 = off
}
TERRAIN_LIMITS = {"hill_m": (20, 3000), "low_mountain_m": (30, 4000), "mountain_m": (50, 5000),
                  "relief_km": (0.5, 50), "relief_smooth_px": (1, 25), "cold_lat": (30, 90),
                  "city_px": (1, 25), "city_grow_px": (0, 25),
                  "city_disc_r1": (0, 50), "city_disc_r2": (0, 50), "city_disc_r3": (0, 50),
                  "forest_px": (1, 25), "desert_smooth_px": (0, 25), "transition_km": (0, 500),
                  "farmland_patch_px": (0, 25), "jungle_blend_px": (0, 25), "tree_density": (5, 400)}
DESERT_ROCKY_SHARE = 0.35     # share of rocky desert (3) in sand desert, in clumps of ~desert_smooth_px
TRANSITION_MAX_ODDS = 0.8     # share of open land that is transition (12) where hot desert dominates
TRANSITION_FULL_SHARE = 0.5   # hot bare share (within transition_km) giving the full odds
DESERT_RELIEF_KM = 10.0       # relief on bare land takes desert textures where bare land covers
DESERT_RELIEF_SHARE = 0.25    # at least this share of the land within DESERT_RELIEF_KM
JUNGLE_CORE_KM = 25.0         # dense jungle: closed tropical forest share >= JUNGLE_CORE_SHARE around
JUNGLE_CORE_SHARE = 0.9
TROPICS_LAT = 25.0            # evergreen broad-leaf forest within this latitude is jungle
CITY_SHARE = 0.15
RELIEF_MIN_PX = 3             # relief class specks smaller than this are merged into their surroundings
MIX_WINDOW_PX = 5             # window of the forest / cropland shares
NOISE_SEED = 1939
# Odds measured on example.png (flat land, by local share in a 5 px window):
# share ->  P(forest), P(open forest); P(farmland) by cropland share.
_SHARE_PTS = [0.0, 0.05, 0.2, 0.4, 0.6, 0.78, 0.9, 0.98, 1.0]
FOREST_ODDS = [0.0, 0.01, 0.03, 0.10, 0.23, 0.40, 0.55, 0.72, 0.75]
OPEN_FOREST_ODDS = [0.0, 0.03, 0.11, 0.19, 0.22, 0.20, 0.16, 0.10, 0.09]
FARMLAND_ODDS = [0.0, 0.02, 0.09, 0.15, 0.20, 0.25, 0.29, 0.31, 0.32]

FOREST_CODES = (111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126)
LAND_CODES = (20, 30, 40, 50, 60, 70, 90, 100) + FOREST_CODES


def normalize_settings(s):
    out = dict(TERRAIN_DEFAULTS)
    for k, v in (s or {}).items():
        if k not in TERRAIN_DEFAULTS or v is None or v == "":
            continue
        if isinstance(TERRAIN_DEFAULTS[k], bool):
            out[k] = v is True or str(v).strip().lower() in ("1", "true", "on", "yes")
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise core.RenderError(f"Terrain setting '{k}' must be a number")
        lo, hi = TERRAIN_LIMITS[k]
        if not lo <= v <= hi:
            raise core.RenderError(f"Terrain setting '{k}' must be between {lo:g} and {hi:g}")
        out[k] = int(round(v)) if isinstance(TERRAIN_DEFAULTS[k], int) else v
    if not out["hill_m"] < out["low_mountain_m"] < out["mountain_m"]:
        raise core.RenderError("The relief thresholds must rise: hills < mountains (grass) < mountains")
    return out


def _water_mask(countries: Image.Image, debug):
    c = np.asarray(countries.convert("RGB"))
    water = (c == np.asarray(core.BACKGROUND_COLOR, dtype=c.dtype)).all(axis=2)
    if debug.get("hidden_land") is not None:
        water &= ~debug["hidden_land"]
    if debug.get("lakes") is not None:
        water |= debug["lakes"]
    water |= (c == np.asarray(core.LAKE_COLOR, dtype=c.dtype)).all(axis=2)
    return water


def _pixel_lonlat(view, rows=None):
    """lon, lat, inside of every pixel centre (or of the given rows)."""
    h, w, scale = view["height"], view["width"], view["scale"]
    rows = np.arange(h) if rows is None else np.asarray(rows)
    xs = view["win_xmin"] + (np.arange(w) + 0.5) / scale
    ys = view["win_ymin"] + (h - (rows + 0.5)) / scale
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return core._unproject(view["projection"], view["center"], xx, yy)


def _finest_deg(view):
    """The view's finest pixel, in degrees (lon step * cos is not used: the
    land cover grid is in plain degrees)."""
    h, w, scale = view["height"], view["width"], view["scale"]
    gx = view["win_xmin"] + (np.linspace(0, w - 1, 33) + 0.5) / scale
    gy = view["win_ymin"] + (h - (np.linspace(0, h - 1, 33) + 0.5)) / scale
    yy, xx = np.meshgrid(gy, gx, indexing="ij")
    lon0, lat0, ok0 = core._unproject(view["projection"], view["center"], xx, yy)
    lon1, _l, ok1 = core._unproject(view["projection"], view["center"], xx + 1.0 / scale, yy)
    _l, lat2, ok2 = core._unproject(view["projection"], view["center"], xx, yy - 1.0 / scale)
    ok = ok0 & ok1 & ok2
    step = np.minimum(np.abs((lon1 - lon0 + 180.0) % 360.0 - 180.0), np.abs(lat2 - lat0))[ok]
    return float(step.min()) if len(step) else 1.0


def _row_km_per_px(view, lat_rows):
    """East-west ground size (km) of one pixel on each row (by its latitude)."""
    func = core.PROJECTIONS[view["projection"]]["func"]
    lat = np.clip(lat_rows, -85.0, 85.0)
    dx_deg = func(np.ones_like(lat), lat)[0]  # projected x per degree of longitude
    return core._KM_PER_DEG * np.cos(np.radians(lat)) / np.maximum(dx_deg * view["scale"], 1e-12)


def _majority(mask, size):
    """Median filter of a boolean mask (size x size window, majority)."""
    size = int(size)
    if size <= 1:
        return mask.copy()
    return ndimage.uniform_filter(mask.astype(np.float32), size=size, mode="nearest") > 0.5


def _share(mask, sigma_px, weight=None):
    """Gaussian-weighted share of `mask` (among `weight` pixels), computed on
    a reduced grid when the radius is large."""
    m = mask.astype(np.float64)
    wgt = np.ones_like(m) if weight is None else weight.astype(np.float64)
    f = max(1, int(sigma_px // 4))
    H, W = m.shape
    ph, pw = (-H) % f, (-W) % f
    mp = np.pad(m * wgt, ((0, ph), (0, pw)), mode="edge")
    wp = np.pad(wgt, ((0, ph), (0, pw)), mode="edge")
    cells = lambda a: a.reshape(a.shape[0] // f, f, a.shape[1] // f, f).sum(axis=(1, 3))
    s = sigma_px / f
    num = ndimage.gaussian_filter(cells(mp), s, mode="nearest")
    den = ndimage.gaussian_filter(cells(wp), s, mode="nearest")
    share = num / np.maximum(den, 1e-9)
    zoom = (mp.shape[0] / share.shape[0], mp.shape[1] / share.shape[1])
    return ndimage.zoom(share, zoom, order=1, mode="nearest", grid_mode=True)[:H, :W]


def _patch_noise(noise, sigma):
    """Uniform [0, 1) noise in patches of ~sigma px: Gaussian-smoothed white
    noise mapped back to uniform through its (known) normal distribution, so
    P(u < p) is still p -- the calibrated odds hold, only the grain changes."""
    if sigma <= 0:
        return noise
    g = ndimage.gaussian_filter(noise.astype(np.float64), sigma, mode="wrap")
    sd = np.sqrt(1.0 / 12.0 / (4.0 * np.pi * sigma * sigma))  # std of a 2-D Gaussian mean of U(0,1)
    return special.ndtr((g - 0.5) / sd).astype(np.float32)


def _drop_small(mask, min_px):
    lab, n = ndimage.label(mask)
    if not n:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def _relief(elev_m, land, lat_rows, view, s):
    """Local relief (m): elevation range (highest minus lowest point) within
    relief_km -- a difference, never the elevation itself -- per band of rows
    (window from that band's km per pixel)."""
    h = elev_m.shape[0]
    km_px = _row_km_per_px(view, lat_rows)
    relief = np.zeros_like(elev_m)
    n_bands = 12
    edges = np.linspace(0, h, n_bands + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        r_px = max(1, int(round(s["relief_km"] / max(float(np.median(km_px[a:b])), 1e-6))))
        pad = r_px
        a0, b0 = max(0, a - pad), min(h, b + pad)
        sub = elev_m[a0:b0]
        # Circular window, like Paint.NET's Outline effect at 100% intensity
        # (the user's method for example.png: outline over the heightmap).
        fp = np.hypot(*np.mgrid[-r_px:r_px + 1, -r_px:r_px + 1]) <= r_px + 0.5
        rng = ndimage.maximum_filter(sub, footprint=fp, mode="nearest") - ndimage.minimum_filter(sub, footprint=fp, mode="nearest")
        relief[a:b] = rng[a - a0:b - a0]
    return relief


def _mode_filter(cls, mask, size):
    """Median-style filter of a class map inside `mask`: each pixel takes
    the most common class in its size x size window (ties: lowest index).
    Only for the few classes present (one uniform filter per class)."""
    if size <= 1:
        return cls
    vals = np.unique(cls[mask])
    best = np.full(cls.shape, -1.0, dtype=np.float32)
    out = cls.copy()
    for v in vals:
        f = ndimage.uniform_filter(((cls == v) & mask).astype(np.float32), size=size, mode="nearest")
        take = mask & (f > best + 1e-6)
        out[take] = v
        best = np.maximum(best, np.where(mask, f, best))
    return out


def _merge_small(cls, mask, min_px):
    """Relief class patches under min_px take the most common class around them."""
    out = cls.copy()
    for v in np.unique(cls[mask]):
        lab, n = ndimage.label((cls == v) & mask)
        if not n:
            continue
        sizes = np.bincount(lab.ravel())
        small = (sizes < min_px)
        small[0] = False
        m = small[lab]
        if not m.any():
            continue
        other = mask & ~m & (cls != v)
        if not other.any():
            continue
        _d, (iy, ix) = ndimage.distance_transform_edt(~other, return_indices=True)
        out[m] = cls[iy[m], ix[m]]
    return out


def _city_discs(points, radii, land):
    """Urban discs around the City tab's cities: points = [(x, y, class)],
    radii = {class: radius px}; on land only."""
    h, w = land.shape
    mask = np.zeros((h, w), dtype=bool)
    for x, y, c in points:
        r = radii.get(int(c), 0.0)
        if r <= 0:
            continue
        k = int(np.floor(r))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        dy, dx = yy[np.hypot(yy, xx) <= r], xx[np.hypot(yy, xx) <= r]
        py, px = y + dy, x + dx
        ok = (py >= 0) & (py < h) & (px >= 0) & (px < w)
        mask[py[ok], px[ok]] = True
    return mask & land


def _trees_map(masks, w, h, blend_px=0, target_frac=None):
    """trees.bmp (P image, the game's palette) from full-resolution masks
    [(tree index, bool mask)]: each tree pixel covers ~3.4x3.4 terrain
    pixels; trees where the masks together cover >= TREE_MIN_SHARE of it,
    of the index covering most. Then sparse jungle as the jungles' blend
    (user): dense / impassable jungle within blend_px (tree px, Euclidean)
    of the jungle's edge -> 28 -- as in the vanilla file (28 median depth
    1.4 px inside the jungle, 29 2, 11 3.2). target_frac: share of the
    tree map to cover with trees (user: the plain 50% majority -- as much
    tree area as land-cover forest -- was a bit too dense); the cutoff
    moves to meet it, so the thinnest forest goes first."""
    tw, th = max(1, round(w * TREES_SCALE[0])), max(1, round(h * TREES_SCALE[1]))
    shares = np.stack([np.asarray(Image.fromarray(m.astype(np.float32), mode="F").resize((tw, th), Image.BOX))
                       for _i, m in masks])
    idx = np.array([i for i, _m in masks], dtype=np.uint8)
    total = shares.sum(axis=0)
    cut = TREE_MIN_SHARE
    if target_frac is not None:
        n = int(round(target_frac * total.size))
        if n <= 0:
            cut = np.inf
        elif n < total.size:
            cut = float(np.partition(total.ravel(), total.size - n)[total.size - n])
        cut = float(np.clip(cut, *TREE_SHARE_RANGE)) if np.isfinite(cut) else cut
    out = np.where(total >= cut, idx[shares.argmax(axis=0)], TREE_NONE).astype(np.uint8)
    jungle = np.isin(out, (TREE_IMPASSABLE, TREE_DENSE_JUNGLE, TREE_SPARSE_JUNGLE))
    blend_n = 0
    if blend_px > 0 and jungle.any():
        # Tiny gaps inside a jungle (a river, a clearing) aren't edges: a
        # sparse ring around every one of them speckled the Amazon.
        lab, n_lab = ndimage.label(~jungle)
        small = np.bincount(lab.ravel(), minlength=n_lab + 1) < JUNGLE_HOLE_MIN_PX
        small[0] = False
        solid = jungle | small[lab]
        depth = ndimage.distance_transform_edt(np.pad(solid, 1))[1:-1, 1:-1]  # frame edge = not jungle
        blend = jungle & (depth <= blend_px) & (out != TREE_SPARSE_JUNGLE)
        out[blend] = TREE_SPARSE_JUNGLE
        blend_n = int(blend.sum())
    img = Image.fromarray(out, mode="P")
    img.putpalette([v for rgb in TREES_PALETTE for v in rgb])
    counts = np.bincount(out.ravel(), minlength=256)
    classes = [{"index": i, "name": n, "color": core.rgb_to_hex(_TREE_PAL[i]), "px": int(counts[i]),
                "share": round(float(counts[i]) / out.size, 4)} for i, n in TREE_CLASSES]
    return img, {"width": tw, "height": th, "classes": classes, "jungle_blend_px": blend_n,
                 "cutoff": None if not np.isfinite(cut) else round(cut, 3)}


def render_terrain(countries: Image.Image, debug, height: Image.Image, settings=None, city_points=None):
    """(terrain P image with the game's palette, report, raw land cover P
    image -- Copernicus codes as indices, Copernicus colours). `height` is
    the raw NASA heightmap on the same frame (core.render_heightmap).
    city_points: the City tab's cities [(x, y, class)] (None = not
    available) for the urban discs."""
    s = normalize_settings(settings)
    t0 = time.time()
    view = debug["view"]
    h, w = view["height"], view["width"]
    water = _water_mask(countries, debug)
    land = ~water
    report = {"settings": s, "width": w, "height": h}
    rng = np.random.default_rng(NOISE_SEED)
    noise = rng.random((h, w), dtype=np.float32)
    noise2 = rng.random((h, w), dtype=np.float32)

    # 1. Land cover on the frame (land pixels only), non-land codes filled
    #    from the nearest land class.
    lon, lat, inside = _pixel_lonlat(view)
    # Sampled everywhere in the world outline (land and water), for the raw
    # land cover view / export; the terrain itself only uses land pixels.
    lc_raw = np.full((h, w), 255, dtype=np.uint8)
    sel = inside & (lat < 80.0) & (lat > -60.0)
    level = None
    if sel.any():
        codes, level = landcover.sample(lon[sel], lat[sel], _finest_deg(view))
        lc_raw[sel] = codes
    lc = np.where(land, lc_raw, 255).astype(np.uint8)
    report["landcover_level"] = level
    t_lc = time.time()
    urban_raw = (lc == 50) & land
    lc_land = lc.copy()   # before the fill below: the tree map uses only real forest pixels
    valid = land & np.isin(lc, LAND_CODES)
    if valid.any() and (land & ~valid).any():
        _d, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
        fill = land & ~valid
        lc[fill] = lc[iy[fill], ix[fill]]
        report["filled_px"] = int(fill.sum())
    km_rows = _row_km_per_px(view, lat[:, w // 2])
    km_mid = float(np.median(km_rows))
    report["km_per_px"] = round(km_mid, 3)
    alat = np.abs(lat)
    cold = alat >= s["cold_lat"]
    tropics = alat < TROPICS_LAT
    bare = lc == 60
    hot_bare = bare & ~cold & land        # hot desert, per pixel

    # 2. Plains / farmland / forest from the local shares (example's odds).
    out = np.full((h, w), PLAINS, dtype=np.uint8)
    evb = np.isin(lc, (112, 122)) & tropics
    forest_lc = np.isin(lc, FOREST_CODES) & ~evb
    fshare = ndimage.uniform_filter(forest_lc.astype(np.float32), MIX_WINDOW_PX, mode="nearest")
    cshare = ndimage.uniform_filter((lc == 40).astype(np.float32), MIX_WINDOW_PX, mode="nearest")
    p1 = np.interp(fshare, _SHARE_PTS, FOREST_ODDS)
    p4 = np.interp(fshare, _SHARE_PTS, OPEN_FOREST_ODDS)
    p5 = np.interp(cshare, _SHARE_PTS, FARMLAND_ODDS)
    forest = land & (noise < p1)
    open_f = land & ~forest & (noise < p1 + p4)
    out[forest] = FOREST
    out[open_f] = FOREST_OPEN
    # Farmland drawn from patchy noise: larger patches of farmland and of
    # plains, the same share of each (farmland_patch_px, 0 = per pixel).
    farm_u = _patch_noise(noise2, s["farmland_patch_px"])
    farm = land & ~forest & ~open_f & (farm_u < p5)
    out[farm] = FARMLAND
    # Forest borders: median filter on the forest mask.
    fm = forest | open_f
    fs = _majority(fm, s["forest_px"]) & land
    if (fs != fm).any():
        grow, shrink = fs & ~fm, fm & ~fs
        if fm.any() and grow.any():
            _d, (iy, ix) = ndimage.distance_transform_edt(~fm, return_indices=True)
            out[grow] = out[iy[grow], ix[grow]]
        out[shrink] = np.where(farm_u[shrink] < p5[shrink], FARMLAND, PLAINS)

    # 3. Other land cover classes.
    out[(lc == 90) & land] = MARSH
    out[evb] = JUNGLE_BLEND
    if evb.any():
        core_ = _share((lc == 112) & tropics, JUNGLE_CORE_KM / km_mid, land) >= JUNGLE_CORE_SHARE
        out[evb & core_] = JUNGLE
    # Tree map (trees.bmp) straight from the land cover's forest.
    jcore = np.zeros((h, w), dtype=bool)
    if evb.any():
        jcore = evb & core_ & (lc_land == 112)
    trop = np.isin(lc_land, (112, 122)) & tropics
    closed = np.isin(lc_land, CLOSED_FOREST_CODES)
    forest_any = np.isin(lc_land, FOREST_CODES)
    tree_masks = [
        (TREE_DENSE, closed & ~trop), (TREE_SPARSE, forest_any & ~closed & ~trop),
        (TREE_IMPASSABLE, jcore), (TREE_DENSE_JUNGLE, trop & closed & ~jcore),
        (TREE_SPARSE_JUNGLE, trop & ~closed)]
    lc_forest = float(sum(m.mean() for _i, m in tree_masks))
    trees_img, report["trees"] = _trees_map(tree_masks, w, h, s["jungle_blend_px"],
                                            s["tree_density"] / 100.0 * lc_forest)
    report["trees"]["landcover_forest"] = round(lc_forest, 4)
    report["_trees_image"] = trees_img
    out[lc == 70] = SNOW_PLAINS
    out[bare & cold] = SNOW_PLAINS
    clump = ndimage.gaussian_filter(noise, s["desert_smooth_px"]) if s["desert_smooth_px"] > 0 else noise
    rocky = clump >= np.quantile(clump, 1.0 - DESERT_ROCKY_SHARE)
    out[hot_bare] = np.where(rocky[hot_bare], DESERT_ROCK, DESERT_SAND)
    # Grass <-> desert transition (FF00F0, index 12): grass, shrubs and moss
    # where hot desert covers part of the surroundings (share within
    # transition_km), speckled, odds rising with that share -- dense at the
    # desert's edge, nothing around an isolated bare speck.
    report["transition_px"] = 0
    if s["transition_km"] > 0 and hot_bare.any():
        share = _share(hot_bare, s["transition_km"] / km_mid, land)
        odds = TRANSITION_MAX_ODDS * np.clip(share / TRANSITION_FULL_SHARE, 0.0, 1.0)
        trans = land & np.isin(lc, (20, 30, 100)) & (out == PLAINS) & (noise2 < odds)
        out[trans] = DESERT_STEPPE
        report["transition_px"] = int(trans.sum())

    # 4. Relief.
    # Height in NASA grey units -> metres: the Heightmap tab's exaggerated
    # heights (core.exaggerated_height, float array), or a plain image.
    h_units = np.asarray(height.convert("L") if isinstance(height, Image.Image) else height, dtype=np.float32)
    elev = h_units * core.NASA_M_PER_UNIT
    relief = _relief(elev, land, lat[:, w // 2], view, s)
    rcls = np.zeros((h, w), dtype=np.uint8)   # 0 flat, 1 hills, 2 low mountains, 3 mountains
    rcls[relief >= s["hill_m"]] = 1
    rcls[relief >= s["low_mountain_m"]] = 2
    rcls[relief >= s["mountain_m"]] = 3
    rcls[~land] = 0
    raw_rcls = rcls
    rcls = _mode_filter(rcls, land, s["relief_smooth_px"])
    rcls = _merge_small(rcls, land, RELIEF_MIN_PX)
    # Mountains always stay on top of hills (user): the clean-up above may
    # raise a pixel's tier or flatten hill specks, but never turn a
    # mountain / mountain (grass) pixel into hills or flat land.
    mtn_raw = raw_rcls >= 2
    report["mountain_px_restored"] = int((mtn_raw & (rcls < raw_rcls)).sum())
    rcls = np.where(mtn_raw, np.maximum(rcls, raw_rcls), rcls).astype(np.uint8)
    snow = lc == 70
    # Relief on hot bare land -> desert textures, unless the bare land is
    # just rock among mountain meadows / forest (Alps, Caucasus).
    desert = hot_bare & (_share(hot_bare, DESERT_RELIEF_KM / km_mid, land) >= DESERT_RELIEF_SHARE)
    report["desert_relief_px"] = int((desert & (rcls > 0)).sum())
    temperate = land & ~desert
    hill, low, mtn = rcls == 1, rcls == 2, rcls == 3
    if s["forest_over_hills"]:
        # Forest drawn above hills: hill pixels that are forest stay forest.
        hill = hill & ~np.isin(out, (FOREST, FOREST_OPEN, JUNGLE, JUNGLE_BLEND))
    out[hill & temperate] = HILLS
    ring = hill & temperate & ndimage.binary_dilation(land & ~hill, structure=np.ones((3, 3), bool))
    out[ring] = HILLS_BLEND
    out[low & temperate] = MOUNTAIN_GRASS
    out[mtn & temperate] = MOUNTAIN
    out[(low | mtn) & temperate & snow] = SNOW_MOUNTAIN
    # Desert relief (user's indices): hills 493B0F (8), outlined in AE00FF
    # (10) where they meet other land, and both mountain tiers F3C793 (31).
    out[hill & desert] = DESERT_HILLS
    d_ring = hill & desert & ndimage.binary_dilation(land & ~hill, structure=np.ones((3, 3), bool))
    out[d_ring] = DESERT_MOUNTAIN
    out[(low | mtn) & desert] = DESERT_TOPS

    # Wetlands also include the lakes the Land/Sea step drew but didn't keep
    # (under its minimum lake size, or shrunk below it later): now land. After the relief:
    # a lake is flat, whatever the hills around it.
    report["small_lake_marsh_px"] = 0
    if s["small_lakes_marsh"] and debug.get("lakes_drawn") is not None:
        small_lakes = debug["lakes_drawn"] & land
        out[small_lakes] = MARSH
        report["small_lake_marsh_px"] = int(small_lakes.sum())

    # 5. Cities.
    urban = land & (ndimage.uniform_filter(urban_raw.astype(np.float32), size=s["city_px"], mode="nearest")
                    >= CITY_SHARE)
    urban = _drop_small(urban, max(2, s["city_px"]))
    if s["city_grow_px"] > 0 and urban.any():
        # Grown outwards by a disc (no smoothing): every city keeps its
        # shape, just bigger.
        g = s["city_grow_px"]
        disc = np.hypot(*np.mgrid[-g:g + 1, -g:g + 1]) <= g + 0.5
        urban = ndimage.binary_dilation(urban, structure=disc) & land
    out[urban] = URBAN
    # The City tab's largest cities (classes 1-3) always urban: a disc each.
    report["city_discs"] = None
    if s["city_discs"] and city_points is not None:
        radii = {1: s["city_disc_r1"], 2: s["city_disc_r2"], 3: s["city_disc_r3"]}
        pts = [p for p in city_points if radii.get(int(p[2]), 0) > 0]
        discs = _city_discs(pts, radii, land)
        report["city_discs"] = {"cities": len(pts), "px": int(discs.sum()), "added_px": int((discs & ~urban).sum())}
        out[discs] = URBAN

    out[water] = OCEAN
    img = Image.fromarray(out, mode="P")
    img.putpalette([v for rgb in TERRAIN_PALETTE for v in rgb])
    counts = np.bincount(out.ravel(), minlength=256)
    total_land = max(int(land.sum()), 1)
    report["classes"] = [{"index": i, "name": n, "type": t, "color": core.rgb_to_hex(_PAL[i]),
                          "px": int(counts[i]),
                          "share_of_land": round(float(counts[i]) / total_land, 4) if i != OCEAN else None}
                         for i, n, t in TERRAIN_CLASSES]
    # Raw land cover: the Copernicus codes as palette indices, in
    # Copernicus' own colours (an indexed PNG keeps the codes).
    lc_img = Image.fromarray(lc_raw, mode="P")
    lc_img.putpalette(landcover.palette().ravel().tolist())
    lc_counts = np.bincount(lc_raw.ravel(), minlength=256)
    pal = landcover.palette()
    n_all = max(int(lc_counts.sum() - lc_counts[255]), 1)
    report["landcover_classes"] = [
        {"code": int(c), "name": n, "color": core.rgb_to_hex(tuple(int(v) for v in pal[c])),
         "px": int(lc_counts[c]), "share": round(float(lc_counts[c]) / n_all, 4)}
        for c, n in sorted(landcover.CLASS_NAMES.items()) if lc_counts[c] and c != 255]
    report["seconds"] = round(time.time() - t0, 2)
    report["landcover_seconds"] = round(t_lc - t0, 2)
    return img, report, lc_img
