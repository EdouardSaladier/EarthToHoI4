"""
Core geometry / projection / rendering logic for the map tool.

Pipeline:
  1. Load the GADM level-0 shapefile once, at full (unsimplified) resolution,
     and cache the parsed rings to disk (rings are classified as
     exterior/hole purely from their signed area, matching the Shapefile
     winding convention -- this avoids pyshp's expensive polygon/hole
     grouping, which is not needed since holes are punched via an even-odd
     raster fill instead of true polygon topology).
  2. Project every country's rings into a chosen 2D projection (cached
     in-memory per projection so repeated renders are instant).
  3. Given a focus country, compute the smallest of the allowed output image
     dimensions whose aspect ratio best matches the country's (margin-padded)
     bounding box, then rasterize every country that falls inside that
     viewport as a flat, unique, non-anti-aliased, borderless color fill
     with an exact pixel-centre scanline fill (`_scanline_fill`, numpy),
     which handles the full-resolution multi-million-point coastlines in
     well under a second.
  4. Optionally reconnect straits that rasterization closed and widen them
     to a minimum width (`fix_straits`, configured from the Land/Sea tab).
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import math
import pickle
import threading
import time
import zlib
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import shapefile
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

CODE_DIR = Path(__file__).resolve().parent   # src/: the code and the page
BASE_DIR = CODE_DIR.parent                    # the project: data/, cache/, settings
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# All Download-tab files (see app.py) live here, one subfolder per resource.
DATA_DIR = BASE_DIR / "data"

# The Download tab fetches GADM's global "levels" archive and extracts it
# into DATA_DIR/gadm/, where gadm36_0.shp (country boundaries) ends up at the
# top level alongside the other admin levels. Some setups may instead already
# have gadm36_0.shp sitting in the project root (how this tool originally
# shipped, before the Download tab existed) -- that still works as a
# fallback. Resolved once at import time: if you download GADM data while
# the server is already running, restart it to pick the file up.
GADM_DOWNLOAD_PATH = DATA_DIR / "gadm" / "gadm36_0.shp"
GADM_FALLBACK_PATH = BASE_DIR / "gadm36_0.shp"


def resolve_shapefile_path() -> Path | None:
    if GADM_DOWNLOAD_PATH.exists():
        return GADM_DOWNLOAD_PATH
    if GADM_FALLBACK_PATH.exists():
        return GADM_FALLBACK_PATH
    return None


SHAPEFILE_PATH = resolve_shapefile_path()

GEOMETRY_CACHE_VERSION = "raw-v1"  # bump to invalidate the on-disk geometry cache
EARTH_RADIUS = 6371.0088  # km, mean Earth radius

TAG_FIELD = "GID_0"
# GADM 3.6 models the Caspian Sea as a level-0 "country" (GID_0 XCA). It is
# water: never drawn, not selectable, and not a land neighbour -- so the
# shores of Azerbaijan, Iran, Kazakhstan, Russia and Turkmenistan on it
# count as coastline.
WATER_TAGS = {"XCA"}
NAME_FIELD = "NAME_0"


class DataNotFoundError(Exception):
    """Raised when no GADM level-0 shapefile can be found (neither
    downloaded via the Download tab nor present in the project root)."""
    pass

# Allowed output dimensions (either orientation is allowed).
# `double_size` on render_map multiplies both sides by BIG_MAP_FACTOR (11264 x
# 4096 = 2x vanilla): maps over ~13.1M px only load in a hoi4.exe patched with
# the map size unlocker (hoi4_map_size_patcher.py). Every size here, doubled,
# stays a multiple of 512 (the patched game's tile size) and <= 16384 per side.
DIMENSIONS = [
    (3584, 3584),
    (3840, 3328),
    (4096, 3072),
    (4608, 2816),
    (5120, 2560),
    (5632, 2304),
    (6400, 2048),
]
BIG_MAP_FACTOR = 2

DEFAULT_MARGIN_PERCENT = 10.0
BACKGROUND_COLOR = (232, 232, 232)  # light neutral grey for unassigned pixels
LAKE_COLOR = (158, 196, 232)  # lakes (HydroLAKES) in the output; reserved, never a country colour
EDGE_LAND_COLOR = (64, 64, 64)  # the unowned land strip of the edge bands; reserved, never a country colour

# HydroLAKES v1.0 polygons (Download tab -> HydroSHEDS -> Lakes), see cut_lakes.
LAKES_SHAPEFILE_PATH = DATA_DIR / "hydrosheds" / "HydroLAKES_polys_v10_shp" / "HydroLAKES_polys_v10.shp"
# HydroRIVERS v1.0 polylines (Download tab -> HydroSHEDS -> Rivers), see render_rivers.
RIVERS_SHAPEFILE_PATH = DATA_DIR / "hydrosheds" / "HydroRIVERS_v10_shp" / "HydroRIVERS_v10.shp"
# NASA Blue Marble topography (GEBCO_08 based), 8 tiles of 90 x 90 degrees
# at 30 arc-seconds: columns A-D from 180 W eastward, row 1 north / 2 south.
TOPOGRAPHY_DIR = DATA_DIR / "topography"


# --------------------------------------------------------------------------
# Step 1: load + cache raw (lon, lat) ring geometry, at full shapefile
# resolution (no simplification -- cv2's rasterizer is fast enough that
# there is no need to trade detail for speed, so renders match GADM's true
# resolution, the same one QGIS shows).
# --------------------------------------------------------------------------

def _signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


_fingerprint_memo = {}


def _cache_key(shapefile_path: Path) -> str:
    """Content fingerprint of a shapefile, used to key every derived cache.

    Deliberately *not* the path or modification time: `cache/` is small and
    ships with the tool, and a user who downloads the same GADM release gets
    byte-identical files at a different path/mtime -- the shipped cache must
    still match. Hashes the .shp size, the whole .shx (record offsets and
    lengths) and .dbf (attributes), and the first and last MB of the .shp
    (~0.1 s for level 2's 50 MB .dbf, memoized per path+mtime in-process)."""
    shapefile_path = Path(shapefile_path)
    stat = shapefile_path.stat()
    memo_key = (str(shapefile_path.resolve()), stat.st_mtime_ns, stat.st_size)
    if memo_key not in _fingerprint_memo:
        h = hashlib.sha1(f"{stat.st_size}|{GEOMETRY_CACHE_VERSION}".encode())
        for sibling in (shapefile_path.with_suffix(".shx"), shapefile_path.with_suffix(".dbf")):
            if sibling.exists():
                h.update(sibling.read_bytes())
        with open(shapefile_path, "rb") as f:
            h.update(f.read(1 << 20))
            f.seek(max(0, stat.st_size - (1 << 20)))
            h.update(f.read(1 << 20))
        _fingerprint_memo[memo_key] = h.hexdigest()[:16]
    return _fingerprint_memo[memo_key]


def _read_polygon_shapefile(path: Path):
    """Read every polygon record of a .shp straight into one float64 array,
    via the .shx offsets and numpy -- ~10x faster than pyshp's per-point
    Python tuples, which is what made an on-disk geometry cache (a 529 MB
    pickle duplicating the 525 MB shapefile) worth having.

    Returns (points (N, 2) lon/lat, ring_start (R+1,), record_ring_start
    (n_records+1,)): ring r is points[ring_start[r]:ring_start[r+1]], and
    record i owns rings record_ring_start[i]:record_ring_start[i+1]."""
    shx = np.fromfile(path.with_suffix(".shx"), dtype=">i4", offset=100).reshape(-1, 2)
    raw = np.fromfile(path, dtype=np.uint8)
    offsets = shx[:, 0].astype(np.int64) * 2 + 8  # 16-bit words -> bytes, skip record header
    chunks, ring_starts, record_rings = [], [], [0]
    n_points = 0
    for off in offsets:
        shape_type = int(raw[off:off + 4].view("<i4")[0])
        if shape_type == 0:  # null shape
            record_rings.append(record_rings[-1])
            continue
        n_parts, n_pts = (int(v) for v in raw[off + 36:off + 44].view("<i4"))
        parts = raw[off + 44:off + 44 + 4 * n_parts].view("<i4").astype(np.int64)
        p0 = off + 44 + 4 * n_parts
        chunks.append(raw[p0:p0 + 16 * n_pts].view("<f8").reshape(n_pts, 2))
        ring_starts.append(parts + n_points)
        n_points += n_pts
        record_rings.append(record_rings[-1] + n_parts)
    points = np.concatenate(chunks)
    ring_start = np.concatenate(ring_starts + [np.array([n_points])])
    return points, ring_start, np.array(record_rings)


_geometries_memo = {}  # shapefile cache key -> loaded dict (kept for the process lifetime)
_world_memo = {}       # id(geometries) -> world-level ring arrays (see world_rings)
_geometries_lock = threading.Lock()


def load_geometries(shapefile_path: Path | None = SHAPEFILE_PATH, progress=print) -> dict:
    """Return {tag: {"name": str, "rings": [(is_hole, ndarray[N,2] lon/lat), ...],
    "lonlat_bbox": (lon0, lat0, lon1, lat1), "wide_span": bool, "ring_range": (r0, r1)}}.

    Read straight from the shapefile (~1 s, see _read_polygon_shapefile) and
    memoized in memory -- there is no on-disk geometry cache any more (it was
    a 529 MB pickle duplicating the shapefile, and slower to build)."""
    if shapefile_path is None:
        raise DataNotFoundError(
            "No GADM level-0 shapefile found. Use the Download tab to fetch GADM data "
            f"(expected at {GADM_DOWNLOAD_PATH} after extraction), or place gadm36_0.shp "
            f"(+ .dbf/.shx/.prj/.cpg) at {GADM_FALLBACK_PATH}."
        )
    key = _cache_key(shapefile_path)
    with _geometries_lock:
        if key not in _geometries_memo:
            _geometries_memo[key] = _build_geometries(shapefile_path, progress)
        return _geometries_memo[key]


def _build_geometries(shapefile_path: Path, progress) -> dict:
    t0 = time.time()
    points, ring_start, record_rings = _read_polygon_shapefile(shapefile_path)
    records = shapefile.Reader(str(shapefile_path)).records(fields=[TAG_FIELD, NAME_FIELD])

    # Exterior vs hole from each ring's signed area (Shapefile winding:
    # exteriors clockwise), all rings at once.
    x, y = points[:, 0], points[:, 1]
    succ = np.arange(1, len(points) + 1)
    succ[ring_start[1:] - 1] = ring_start[:-1]
    ring_area = 0.5 * np.add.reduceat(x * y[succ] - x[succ] * y, ring_start[:-1])
    ring_bbox = np.column_stack([np.minimum.reduceat(x, ring_start[:-1]), np.minimum.reduceat(y, ring_start[:-1]),
                                 np.maximum.reduceat(x, ring_start[:-1]), np.maximum.reduceat(y, ring_start[:-1])])

    result = {}
    ring_country = np.empty(len(ring_start) - 1, dtype=np.int32)
    for i, rec in enumerate(records):
        tag = rec[0]
        r0, r1 = int(record_rings[i]), int(record_rings[i + 1])
        if r1 == r0:
            continue
        ring_country[r0:r1] = len(result)
        rings = [(bool(ring_area[k] > 0), points[ring_start[k]:ring_start[k + 1]]) for k in range(r0, r1)]
        bb = ring_bbox[r0:r1]
        lon_min, lon_max = float(bb[:, 0].min()), float(bb[:, 2].max())
        lonlat_bbox = (lon_min, float(bb[:, 1].min()), lon_max, float(bb[:, 3].max()))
        # A country whose raw longitude naively spans most of the globe either
        # crosses the +/-180 antimeridian (Russia, Fiji, the USA via the
        # Aleutians, ...) or sits at a pole (Antarctica): its true extent is
        # NOT the naive min/max corners, so the cheap bbox estimate used to
        # cull far-away countries during rendering can't be trusted for it --
        # such countries are always projected exactly instead (see render_map).
        wide_span = (lon_max - lon_min) > 300.0
        result[tag] = {"name": rec[1], "rings": rings, "lonlat_bbox": lonlat_bbox,
                       "wide_span": wide_span, "ring_range": (r0, r1)}
    _world_memo[id(result)] = {"tags": list(result), "ring_bbox": ring_bbox, "ring_country": ring_country,
                               "ring_area": ring_area, "ring_start": ring_start, "points": points}
    progress(f"  read {len(result)} countries ({len(points):,} points) in {time.time() - t0:.1f}s")
    return result


def world_rings(geometries: dict) -> dict:
    """World-level arrays over every ring of `geometries`: "ring_bbox"
    (R, 4) lon/lat, "ring_country" (index into "tags"), "ring_start"."""
    return _world_memo[id(geometries)]


# --------------------------------------------------------------------------
# Step 2: projections
# --------------------------------------------------------------------------

_ROBINSON_LAT = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                           50, 55, 60, 65, 70, 75, 80, 85, 90], dtype=np.float64)
_ROBINSON_X = np.array([1.0000, 0.9986, 0.9954, 0.9900, 0.9822, 0.9730, 0.9600,
                         0.9427, 0.9216, 0.8962, 0.8679, 0.8350, 0.7986, 0.7597,
                         0.7186, 0.6732, 0.6213, 0.5722, 0.5322], dtype=np.float64)
_ROBINSON_Y = np.array([0.0000, 0.0620, 0.1240, 0.1860, 0.2480, 0.3100, 0.3720,
                         0.4340, 0.4958, 0.5571, 0.6176, 0.6769, 0.7346, 0.7903,
                         0.8435, 0.8936, 0.9394, 0.9761, 1.0000], dtype=np.float64)


def _relative_lon(lon, center):
    """Longitude relative to `center`, wrapped to (-180, 180]. Used point-wise
    (e.g. for the cheap bbox estimate); for a sequence of vertices along a
    ring, use `_continuous_relative_lon` instead -- see its docstring."""
    return ((np.asarray(lon, dtype=np.float64) - center + 180.0) % 360.0) - 180.0


def _continuous_relative_lon(lon, center):
    """Longitude relative to `center`, as a *continuous* sequence (no +/-360
    jumps) rather than wrapped to (-180, 180].

    A ring can legitimately cross the seam partway through -- e.g. Russia's
    mainland is one single ~766,000-point boundary trace running from the
    Baltic all the way to the Bering Strait and back, which cannot avoid
    crossing +/-180 somewhere no matter where the seam is placed. Wrapping
    each vertex independently to (-180, 180] would put the vertices right
    after that crossing on the opposite end of the range, producing a
    spurious edge-to-edge jump when the ring is filled. `np.unwrap` removes
    exactly that kind of jump by adding/subtracting full turns as needed, so
    the whole ring -- including the vast majority of it that has nothing to
    do with the antimeridian -- projects correctly instead of being
    discarded."""
    return np.unwrap(_relative_lon(lon, center), period=360.0)


def _circular_dist(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _in_forward_arc(x, start, end):
    """True if x lies on the arc going forward (increasing, wrapping at 360)
    from start to end."""
    x, start, end = x % 360.0, start % 360.0, end % 360.0
    if start <= end:
        return start <= x <= end
    return x >= start or x <= end


def compute_central_meridian(lons) -> float:
    """Return the seam longitude the world should be cut at so that `lons`
    (typically one country's vertices) never wraps across it. The standard
    seam at +/-180 degrees is kept whenever possible (it runs mostly through
    open ocean, so every other country stays intact); it is only nudged when
    the input itself straddles +/-180 (e.g. the USA via the Aleutians,
    Russia, Fiji), and then only just far enough to clear it, so the new seam
    still falls in that country's own "empty" longitude gap rather than
    slicing through unrelated landmasses on the far side of the world."""
    lons = np.mod(np.asarray(lons, dtype=np.float64), 360.0)
    lons_sorted = np.sort(np.unique(lons))
    if len(lons_sorted) < 2:
        return 0.0

    gaps = np.diff(lons_sorted)
    wrap_gap = lons_sorted[0] + 360.0 - lons_sorted[-1]
    all_gaps = np.append(gaps, wrap_gap)
    max_idx = int(np.argmax(all_gaps))
    if max_idx == len(gaps):
        gap_start, gap_end = lons_sorted[-1], lons_sorted[0] + 360.0
    else:
        gap_start, gap_end = lons_sorted[max_idx], lons_sorted[max_idx + 1]

    if _in_forward_arc(180.0, gap_start, gap_end):
        return 0.0  # standard seam already falls in empty space: no need to move it

    gap_len = gap_end - gap_start
    nudge = min(2.0, gap_len / 4.0)
    if _circular_dist(180.0, gap_start % 360.0) <= _circular_dist(180.0, gap_end % 360.0):
        seam = gap_start + nudge
    else:
        seam = gap_end - nudge

    center = (seam - 180.0) % 360.0
    if center > 180.0:
        center -= 360.0
    return center


# Every _proj_* function below takes `rel_lon` -- longitude already made
# relative to the render's central meridian (and, for a whole ring, already
# unwrapped into a continuous sequence -- see `_continuous_relative_lon`) --
# rather than raw longitude plus a center to re-derive it from, so a caller
# controls once and for all whether wrapping or continuity is wanted.

def _proj_equirectangular(rel_lon, lat):
    return EARTH_RADIUS * np.radians(rel_lon), EARTH_RADIUS * np.radians(lat)


def _proj_mercator(rel_lon, lat):
    lat_c = np.clip(lat, -85.05, 85.05)
    x = EARTH_RADIUS * np.radians(rel_lon)
    y = EARTH_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat_c) / 2))
    return x, y


def _proj_miller(rel_lon, lat):
    x = EARTH_RADIUS * np.radians(rel_lon)
    y = EARTH_RADIUS * 1.25 * np.log(np.tan(np.pi / 4 + 0.4 * np.radians(lat)))
    return x, y


def _proj_robinson(rel_lon, lat):
    lat = np.asarray(lat, dtype=np.float64)
    abslat = np.abs(lat)
    xf = np.interp(abslat, _ROBINSON_LAT, _ROBINSON_X)
    yf = np.interp(abslat, _ROBINSON_LAT, _ROBINSON_Y)
    x = 0.8487 * EARTH_RADIUS * xf * np.radians(rel_lon)
    y = 1.3523 * EARTH_RADIUS * yf * np.sign(lat)
    return x, y


def _mollweide_theta_exact(lat_rad):
    theta = lat_rad.copy()
    for _ in range(12):
        denom = 2 + 2 * np.cos(2 * theta)
        denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        delta = -(2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat_rad)) / denom
        theta = theta + delta
    return theta


# Mollweide's auxiliary angle depends on latitude only: solve it once on a
# 0.0001-degree grid and interpolate, instead of 12 Newton steps per vertex
# (projecting the whole world went from ~23 s to a few seconds).
_MOLLWEIDE_LAT = np.linspace(-90.0, 90.0, 1_800_001)
_MOLLWEIDE_THETA = _mollweide_theta_exact(np.radians(_MOLLWEIDE_LAT))


def _proj_mollweide(rel_lon, lat):
    lat = np.asarray(lat, dtype=np.float64)
    # Uniform grid: direct index arithmetic (np.interp's binary search was
    # barely faster than Newton). Within 0.5 deg of a pole theta's slope
    # blows up and linear interpolation drifts, so those (rare) vertices
    # get the exact solve instead.
    pos = np.clip((lat + 90.0) * 10000.0, 0.0, 1_800_000.0)
    i = np.minimum(pos.astype(np.int64), 1_799_999)
    f = pos - i
    theta = _MOLLWEIDE_THETA[i] * (1.0 - f) + _MOLLWEIDE_THETA[i + 1] * f
    polar = np.abs(lat) > 89.5
    if polar.any():
        theta[polar] = _mollweide_theta_exact(np.radians(lat[polar]))
    lon_rad = np.radians(rel_lon)
    x = EARTH_RADIUS * (2 * math.sqrt(2) / math.pi) * lon_rad * np.cos(theta)
    y = EARTH_RADIUS * math.sqrt(2) * np.sin(theta)
    return x, y


PROJECTIONS = {
    "miller": {"label": "Miller Cylindrical", "func": _proj_miller},
    "mercator": {"label": "Mercator", "func": _proj_mercator},
    "equirectangular": {"label": "Equirectangular (Plate Carree)", "func": _proj_equirectangular},
    "robinson": {"label": "Robinson", "func": _proj_robinson},
    "mollweide": {"label": "Mollweide", "func": _proj_mollweide},
}

_projected_country_cache: dict[tuple, dict] = {}

# Central meridians are snapped to this grid so repeated renders of nearby
# countries reuse the same cached projected geometry instead of recomputing.
_MERIDIAN_SNAP_DEG = 5.0


def _bbox_spans_seam(lonlat_bbox, center: float) -> bool:
    """True if this render's seam (where `_relative_lon` is discontinuous)
    falls inside the country's raw lon bbox -- e.g. small Pacific nations
    (Fiji, Vanuatu, the Marshall Islands, ...) sit close enough to +/-180
    that a focus country elsewhere can push the seam right through their own
    (otherwise unremarkable, non-wide) bbox. The cheap corner-sampled
    estimate silently misses the resulting jump, so such countries must be
    projected exactly instead, just like wide-span ones."""
    lon_min, _lat_min, lon_max, _lat_max = lonlat_bbox
    seam = ((center + 360.0) % 360.0) - 180.0
    return lon_min - 1e-6 <= seam <= lon_max + 1e-6


def _project_rings(rings, func, center: float):
    """Project rings; returns (rings, bbox, ring_bboxes, wraps, wrap_bboxes).

    A ring whose unwrapped longitude runs past the seam (+/-180 from the
    central meridian) also gets a copy shifted by a full turn -- `wraps`,
    as (source ring index, is_hole, xy) -- so the part beyond the seam shows
    up on the other side of the map instead of off the edge (China, Vietnam,
    ... vanished when the seam fell through East Asia). Original and copy
    are filled together; whatever lands outside the projection's world
    outline is cleared afterwards (see _domain_mask), which is exact for the
    pseudo-cylindrical projections too, where a plain x shift wouldn't be."""
    out_rings, wraps = [], []
    ring_bboxes = np.empty((len(rings), 4))
    wrap_boxes = []
    for k, (is_hole, lonlat) in enumerate(rings):
        # Unwrap rather than wrap-and-hope: a ring that crosses the seam
        # partway through (Russia's mainland is one continuous boundary
        # trace from the Baltic to the Bering Strait) still projects
        # correctly this way instead of needing to be discarded -- see
        # `_continuous_relative_lon`.
        rel_lon = _continuous_relative_lon(lonlat[:, 0], center)
        x, y = func(rel_lon, lonlat[:, 1])
        out_rings.append((is_hole, np.column_stack([x, y])))
        ring_bboxes[k] = (x.min(), y.min(), x.max(), y.max())
        lo, hi = float(rel_lon.min()), float(rel_lon.max())
        for shift in ((-360.0,) if hi > 180.0 else ()) + ((360.0,) if lo < -180.0 else ()):
            wx, wy = func(rel_lon + shift, lonlat[:, 1])
            wraps.append((k, is_hole, np.column_stack([wx, wy])))
            wrap_boxes.append((wx.min(), wy.min(), wx.max(), wy.max()))
    wrap_bboxes = np.array(wrap_boxes).reshape(-1, 4)
    all_boxes = np.concatenate([ring_bboxes, wrap_bboxes])
    bbox = (float(all_boxes[:, 0].min()), float(all_boxes[:, 1].min()),
            float(all_boxes[:, 2].max()), float(all_boxes[:, 3].max()))
    return out_rings, bbox, ring_bboxes, wraps, wrap_bboxes


def _domain_mask(func, win_xmin: float, win_ymin: float, scale: float, out_w: int, out_h: int):
    """Bool image: True where the pixel centre is inside the projection's
    world outline (relative longitude within +/-180, latitude within the
    projection's range)."""
    lats = np.linspace(-90.0, 90.0, 3601)
    xr, yr = func(np.full_like(lats, 180.0), lats)
    xl, _ = func(np.full_like(lats, -180.0), lats)
    rows_y = win_ymin + (out_h - (np.arange(out_h) + 0.5)) / scale
    cols_x = win_xmin + (np.arange(out_w) + 0.5) / scale
    xr_row = np.interp(rows_y, yr, xr)
    xl_row = np.interp(rows_y, yr, xl)
    row_ok = (rows_y >= yr.min()) & (rows_y <= yr.max())
    return (cols_x[None, :] >= xl_row[:, None]) & (cols_x[None, :] <= xr_row[:, None]) & row_ok[:, None]


def get_projected_country(geometries: dict, tag: str, projection_id: str, center: float = 0.0) -> dict:
    """Project one country's rings (lazily; cached per tag/projection/center).

    Rendering a view only ever needs the focus country plus whichever other
    countries actually fall inside the viewport, so projecting all 256
    countries' full-resolution geometry up front (as an earlier version of
    this module did) wastes most of that work -- at full GADM resolution
    that alone took several seconds per render. Projecting lazily, one
    country at a time, keeps renders fast regardless of dataset size."""
    key = (tag, projection_id, center)
    cached = _projected_country_cache.get(key)
    if cached is not None:
        return cached
    func = PROJECTIONS[projection_id]["func"]
    rings, bbox, ring_bboxes, wraps, wrap_bboxes = _project_rings(geometries[tag]["rings"], func, center)
    result = {"name": geometries[tag]["name"], "rings": rings, "bbox": bbox, "ring_bboxes": ring_bboxes,
              "wraps": wraps, "wrap_bboxes": wrap_bboxes}
    _projected_country_cache[key] = result
    return result


def _simplified_runs(px, py, mask, k: int):
    """Simplify a closed pixel-space ring for coastline/border extraction:
    keep a vertex only where it enters a new 1/k-pixel cell or where `mask`
    (per edge: coastline or not) changes, and join kept vertices by chords.
    Returns (start idx, end idx, mask of each chord). Chords deviate from
    the true line by < 1/k px -- the strait fixer's sub-pixel resolution --
    while run end points (where coastline meets a land border) are kept
    exactly, so neighbouring countries' runs still meet and every ring
    stays closed (crossing parity intact). On a world map this drops ~95%
    of the tens of millions of sub-pixel segments."""
    cx = np.floor(px * k).astype(np.int64)
    cy = np.floor(py * k).astype(np.int64)
    keep = (cx != np.roll(cx, 1)) | (cy != np.roll(cy, 1)) | (mask != np.roll(mask, 1))
    idx = np.flatnonzero(keep)
    if len(idx) == 0:
        idx = np.array([0])
    return idx, np.roll(idx, -1), mask[idx]


def _projected_area(proj: dict) -> float:
    """A projected country's land area (projection units^2: exteriors minus
    holes), computed once per projected country, vectorized over all its
    rings at once (per-ring Python loops cost ~1.6 s on a world map)."""
    if "area" not in proj:
        rings = proj["rings"]
        sizes = np.array([len(xy) for _h, xy in rings])
        pts = np.concatenate([xy for _h, xy in rings])
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        succ = np.arange(1, len(pts) + 1)
        succ[starts + sizes - 1] = starts
        x, y = pts[:, 0], pts[:, 1]
        signed = 0.5 * np.add.reduceat(x * y[succ] - x[succ] * y, starts)
        holes = np.array([h for h, _xy in rings])
        proj["area"] = float(abs(signed[~holes].sum()) - abs(signed[holes].sum()))
    return proj["area"]


def _estimate_projected_bbox(lonlat_bbox, func, center: float):
    """Cheap bounding-box estimate for a country from just its raw lon/lat
    bbox corners (plus the equator, if crossed), without projecting every
    vertex. Exact for all 5 projections implemented here: y depends only on
    latitude and is monotonic in it for each of them, and x is either
    independent of latitude (equirectangular/mercator/miller) or scales a
    lon-only term by a factor that peaks at the equator and falls off
    monotonically towards the poles (Robinson/Mollweide) -- so the true
    x/y extrema always occur at one of these sampled points."""
    # Only called for countries already known not to straddle the seam (see
    # wide_span / _bbox_spans_seam in render_map), so a plain point-wise
    # relative longitude -- no unwrapping needed -- is safe here.
    lon_min, lat_min, lon_max, lat_max = lonlat_bbox
    lats = {lat_min, lat_max}
    if lat_min <= 0.0 <= lat_max:
        lats.add(0.0)
    xs, ys = [], []
    for la in lats:
        for lo in (lon_min, lon_max):
            rel_lo = _relative_lon(np.array([lo]), center)
            x, y = func(rel_lo, np.array([la]))
            xs.append(float(x[0]))
            ys.append(float(y[0]))
    return min(xs), min(ys), max(xs), max(ys)


# World map default: the edge runs through the Bering Strait. 168.4 W is
# land-free from 62 to 72 N -- east of St Lawrence Island and both Diomedes,
# west of Alaska's Cape Prince of Wales (checked against GADM); no meridian
# there avoids the Aleutians further south, which wrap to both edges.
WORLD_SEAM_LON = -168.4
WORLD_CENTRAL_MERIDIAN = WORLD_SEAM_LON + 180.0  # 11.6 E

_land_hist_memo = {}


def _land_per_meridian(geometries: dict) -> np.ndarray:
    """360 one-degree bins (bin i = lon -180+i .. -179+i): how much land a
    seam there would cut, as the summed latitude extent of every exterior
    ring whose lon range covers it (rings spanning > 180 deg cross the
    antimeridian: they cover the outside of their naive range)."""
    key = id(geometries)
    if key not in _land_hist_memo:
        world = world_rings(geometries)
        bb, rc = world["ring_bbox"], world["ring_country"]
        exterior = world["ring_area"] <= 0
        water_idx = [world["tags"].index(t) for t in WATER_TAGS if t in world["tags"]]
        keep = exterior & ~np.isin(rc, water_idx)
        lon0, lat0, lon1, lat1 = bb[keep].T
        weight = lat1 - lat0
        diff = np.zeros(361)
        normal = (lon1 - lon0) <= 180.0
        a = np.clip(np.floor(lon0 + 180.0).astype(int), 0, 359)
        b = np.clip(np.floor(lon1 + 180.0).astype(int), 0, 359) + 1
        np.add.at(diff, a[normal], weight[normal])
        np.add.at(diff, b[normal], -weight[normal])
        wrap = ~normal  # covers [lon1, 180] and [-180, lon0]
        np.add.at(diff, b[wrap] - 1, weight[wrap])
        diff[360] -= weight[wrap].sum()
        diff[0] += weight[wrap].sum()
        np.add.at(diff, a[wrap] + 1, -weight[wrap])
        _land_hist_memo[key] = np.cumsum(diff)[:360]
    return _land_hist_memo[key]


PSEUDO_CYLINDRICAL = {"robinson", "mollweide"}  # projections whose shape depends on the central meridian
CENTERED_SEAM_VISIBLE_GAP_DEG = 120.0  # a seam in a gap this narrow may show in the view: keep it off land
CENTERED_MAX_SHIFT_DEG = 30.0          # how far the centre may move from the map's middle for that


def _centered_meridian_for(geometries: dict, tags, ring_masks=None, lon_override=None) -> float:
    """Central meridian for the pseudo-cylindrical projections (Robinson,
    Mollweide), where distortion grows away from it: the middle of the
    focus countries' longitude range (the arc left by the largest gap
    between their longitudes; mainland rings only when `ring_masks`), so the
    projection is centred on the map rather than near Greenwich. Its seam
    (the opposite meridian, the projection's curved edge) falls in the
    middle of that gap, off the focus countries. When the gap is narrow
    enough (<= CENTERED_SEAM_VISIBLE_GAP_DEG) for the seam to show in the
    view, the seam goes, over every gap between the focus countries, to the
    meridian cutting the least land within CENTERED_MAX_SHIFT_DEG of that
    gap's middle (_land_per_meridian, 1-degree bins; the Bering Strait
    meridian WORLD_SEAM_LON counts as open sea) -- a Russia + USA map cut
    Alaska and showed its western part as an "island" beyond Chukotka, and
    the Atlantic (the largest gap there) can't avoid Greenland/Europe."""
    lon_override = lon_override or {}
    lons = np.concatenate(list(lon_override.values()) + [
        ring[:, 0]
        for t in tags if t not in lon_override
        for k, (_is_hole, ring) in enumerate(geometries[t]["rings"])
        if ring_masks is None or ring_masks[t][k] or not ring_masks[t].any()
    ])
    pts = np.sort(np.unique(np.mod(lons, 360.0)))
    if len(pts) < 2:
        return float(((pts[0] if len(pts) else 0.0) + 180.0) % 360.0 - 180.0)
    ends = np.append(pts[1:], pts[0] + 360.0)
    widths = ends - pts
    i = int(np.argmax(widths))
    seam = (float(pts[i]) + float(ends[i])) / 2.0  # middle of the largest gap
    if widths[i] <= CENTERED_SEAM_VISIBLE_GAP_DEG:
        # Seam likely visible: over every gap between the focus countries,
        # the meridian cutting the least land within CENTERED_MAX_SHIFT_DEG
        # of that gap's middle (ties: the smallest shift). The seam and the
        # centre are tied (opposite meridians), so a shifted seam moves the
        # centre off the map's middle by as much -- bounded by the cap.
        hist = _land_per_meridian(geometries)
        floor_cost = float(hist.min())  # land every meridian crosses (Antarctica)
        bering = WORLD_SEAM_LON % 360.0
        cands = []
        for g0, g1 in zip(pts[widths >= 4.0], ends[widths >= 4.0]):
            mid = (g0 + g1) / 2.0
            lo, hi = max(g0 + 1.0, mid - CENTERED_MAX_SHIFT_DEG), min(g1 - 1.0, mid + CENTERED_MAX_SHIFT_DEG)
            cands += [(float(hist[int(np.floor((m + 180.0) % 360.0))]), abs(m - mid), m)
                      for m in np.arange(math.ceil(lo), hi + 1e-9, 1.0)]
            for b_ in (bering, bering + 360.0):
                if g0 + 0.3 <= b_ <= g1 - 0.3 and abs(b_ - mid) <= CENTERED_MAX_SHIFT_DEG:
                    cands.append((floor_cost, abs(b_ - mid), b_))
        if cands:
            seam = min(cands)[2]
    center = ((seam - 180.0) + 180.0) % 360.0 - 180.0
    return round(float(center), 1)


def _central_meridian_for(geometries: dict, tags, ring_masks=None, lon_override=None) -> float:
    """Pick one seam that keeps every focus country's own geometry
    contiguous (only the rings flagged in `ring_masks[tag]`, when given) and,
    among those, cuts through as little land as possible: candidates are the
    5-degree meridians at least 2 degrees inside a gap between the focus
    countries' longitudes, scored by _land_per_meridian (ties -> nearest to
    +/-180). The old rule -- keep +/-180 if it's in the largest gap, else
    nudge just past the gap's edge -- could put the seam through China for a
    Finland..Argentina..Japan selection."""
    lon_override = lon_override or {}
    lons = np.concatenate(list(lon_override.values()) + [
        ring[:, 0]
        for t in tags if t not in lon_override
        for k, (_is_hole, ring) in enumerate(geometries[t]["rings"])
        if ring_masks is None or ring_masks[t][k] or not ring_masks[t].any()
    ])
    pts = np.sort(np.unique(np.mod(lons, 360.0)))
    hist = _land_per_meridian(geometries)
    best = None
    if len(pts) >= 2:
        starts = pts
        ends = np.append(pts[1:], pts[0] + 360.0)
        for g0, g1 in zip(starts[ends - starts > 4.0], ends[ends - starts > 4.0]):
            first = math.ceil((g0 + 2.0) / _MERIDIAN_SNAP_DEG) * _MERIDIAN_SNAP_DEG
            for m in np.arange(first, g1 - 2.0 + 1e-9, _MERIDIAN_SNAP_DEG):
                seam = ((m + 180.0) % 360.0) - 180.0  # -> [-180, 180)
                cost = hist[int(np.floor(seam + 180.0)) % 360]
                score = (cost, _circular_dist(seam, 180.0))
                if best is None or score < best[0]:
                    best = (score, seam)
    if best is None:
        center = compute_central_meridian(lons)
        return round(center / _MERIDIAN_SNAP_DEG) * _MERIDIAN_SNAP_DEG
    center = ((best[1] - 180.0 + 180.0) % 360.0) - 180.0
    return float(round(center / _MERIDIAN_SNAP_DEG) * _MERIDIAN_SNAP_DEG)


def country_meridian(geometries: dict, tag: str) -> float:
    """Central meridian that puts `tag` in the middle of the map: circular
    mean of its main landmass's longitudes, on the 5-degree grid."""
    ring = geometries[tag]["rings"][largest_ring(geometries, tag)][1]
    ang = np.radians(ring[:, 0])
    lon = math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean()))
    center = round(lon / _MERIDIAN_SNAP_DEG) * _MERIDIAN_SNAP_DEG
    return float(((center + 180.0) % 360.0) - 180.0)


# --------------------------------------------------------------------------
# Step 2b: overseas territories (from GADM level 2, else level 1)
# --------------------------------------------------------------------------
# A country's subdivisions -- GADM level 2 where available (municipalities,
# counties, provinces, ...), else level 1 -- are grouped into clusters lying
# within OVERSEAS_DISTANCE_KM of each other (touching subdivisions are at
# distance 0). The cluster with the largest total area is the "mainland";
# every other cluster is overseas (Portugal: Azores and Madeira; USA: Alaska
# and Hawaii; Spain: the Canaries; Chile: Easter Island, which level 1 alone
# misses since it belongs to the mainland Valparaiso region). Each level-0
# ring is then assigned to its nearest subdivision, so framing can skip
# overseas rings while rendering still uses the full-res level-0 geometry.
#
# Limitations: granularity is the subdivision (one containing a far-flung
# island counts as a whole), and island chains with hops under the threshold
# stay connected (Japan's Ogasawara is 233 km from the Izu chain). GADM
# level 3 doesn't help either known case and multiplies region counts, so it
# is deliberately not used.

ADMIN_SHAPEFILE_NAMES = {2: "gadm36_2.shp", 1: "gadm36_1.shp"}
OVERSEAS_DISTANCE_KM = 500.0
# Vertices kept per country for clustering (spread over its subdivisions);
# clustering only -- rendering stays full-res.
_ADMIN_POINT_BUDGET = 300_000
_ADMIN_MIN_POINTS_PER_REGION = 200
OVERSEAS_CACHE_VERSION = "v2"

_admin_readers = {}  # level -> (shapefile.Reader, {GID_0: [record index, ...]}) or None
_admin_lock = threading.Lock()
_mainland_cache = None  # tag -> {"ring_mask": ndarray[bool], "overseas_regions": [names], "level": int}


def _admin_shapefile_path(level: int) -> Path | None:
    if SHAPEFILE_PATH is None:
        return None
    path = SHAPEFILE_PATH.parent / ADMIN_SHAPEFILE_NAMES[level]
    return path if path.exists() else None


def _admin_reader(level: int):
    """(Reader, {GID_0: [record index]}) for one admin level, or None."""
    if level not in _admin_readers:
        path = _admin_shapefile_path(level)
        if path is None:
            _admin_readers[level] = None
        else:
            reader = shapefile.Reader(str(path))
            index = {}
            for i, rec in enumerate(reader.records(fields=[TAG_FIELD])):
                index.setdefault(rec[0], []).append(i)
            _admin_readers[level] = (reader, index)
    return _admin_readers[level]


def _overseas_cache_file() -> Path:
    parts = [_cache_key(SHAPEFILE_PATH), OVERSEAS_CACHE_VERSION, str(OVERSEAS_DISTANCE_KM)]
    for level in sorted(ADMIN_SHAPEFILE_NAMES):
        path = _admin_shapefile_path(level)
        parts.append(_cache_key(path) if path else "-")
    return CACHE_DIR / f"overseas_{hashlib.sha1('|'.join(parts).encode()).hexdigest()[:16]}.pkl"


def _lonlat_to_xyz(lonlat: np.ndarray) -> np.ndarray:
    lon = np.radians(lonlat[:, 0])
    lat = np.radians(lonlat[:, 1])
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _read_subdivisions(tag: str):
    """Return (level, [(name_1, name_2 | None, decimated lon/lat points,
    approx area), ...]) for one country's finest available subdivisions,
    read via the .shx index (no full scan). Caller holds _admin_lock."""
    for level in sorted(ADMIN_SHAPEFILE_NAMES, reverse=True):
        entry = _admin_reader(level)
        if entry is None or tag not in entry[1]:
            continue
        reader, index = entry
        indices = index[tag]
        budget = max(_ADMIN_MIN_POINTS_PER_REGION, _ADMIN_POINT_BUDGET // len(indices))
        regions = []
        for i in indices:
            shp = reader.shape(i)
            pts_all = np.asarray(shp.points, dtype=np.float64)
            if len(pts_all) == 0:
                continue
            rec = reader.record(i)
            parts = list(shp.parts) + [len(pts_all)]
            step = max(1, len(pts_all) // budget)
            kept, area = [], 0.0
            for j in range(len(parts) - 1):
                ring = pts_all[parts[j]:parts[j + 1]]
                kept.append(ring[::step] if len(ring) > 4 * step else ring[:4])
                # Sinusoidal-style equal-area approximation; lon unwrapped so
                # rings crossing +/-180 (Chukotka, Alaska's Aleutians) work.
                lon = np.unwrap(ring[:, 0], period=360.0)
                lat = ring[:, 1]
                x = lon * np.cos(np.radians(lat))
                area -= 0.5 * float(np.sum(x * np.roll(lat, -1) - np.roll(x, -1) * lat))
            regions.append((rec["NAME_1"], rec["NAME_2"] if level >= 2 else None,
                            np.concatenate(kept), abs(area)))
        return level, regions
    if any(_admin_reader(level) is not None for level in ADMIN_SHAPEFILE_NAMES):
        # Files exist, this country just has no subdivisions in GADM (Monaco,
        # Vatican, Gibraltar, Kiribati, ...): one region, nothing overseas.
        return None, []
    raise DataNotFoundError(
        "Ignoring overseas territories needs a GADM level-2 or level-1 shapefile "
        f"({' / '.join(ADMIN_SHAPEFILE_NAMES.values())}) next to {SHAPEFILE_PATH}. Use the Download tab.")


def _cluster_regions(xyz, areas, radius: float) -> np.ndarray:
    """Union-find over regions within `radius` (chord) of each other; returns
    a bool array marking the largest-area cluster."""
    n = len(xyz)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Bounding spheres prune clearly distant pairs in one vectorized pass;
    # candidates are checked closest-first so neighbours merge early and
    # most later pairs are skipped as already connected (matters for the
    # thousands of counties/municipalities at level 2).
    centers = np.array([p.mean(axis=0) for p in xyz])
    spans = np.array([np.linalg.norm(p - c, axis=1).max() for p, c in zip(xyz, centers)])
    gap = cdist(centers, centers) - spans[:, None] - spans[None, :]
    ia, ib = np.nonzero(np.triu(gap <= radius, k=1))
    order = np.argsort(gap[ia, ib])
    trees = [None] * n
    for a, b in zip(ia[order].tolist(), ib[order].tolist()):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        for k in (a, b):
            if trees[k] is None:
                trees[k] = cKDTree(xyz[k])
        if trees[a].count_neighbors(trees[b], radius) > 0:
            parent[ra] = rb

    roots = np.array([find(i) for i in range(n)])
    cluster_area = np.bincount(roots, weights=areas, minlength=n)
    return roots == int(np.argmax(cluster_area))


def _overseas_names(regions, is_mainland) -> list:
    """Name overseas areas by level-1 region when all of it is overseas
    (e.g. "Azores"), else as "<level 2> (<level 1>)" (e.g. "Isla de Pascua
    (Valparaiso)")."""
    by_region = {}
    for (name_1, name_2, _pts, _area), main in zip(regions, is_mainland):
        by_region.setdefault(name_1, []).append((name_2, main))
    names = []
    for name_1, subs in by_region.items():
        overseas_subs = [n2 for n2, main in subs if not main]
        if not overseas_subs:
            continue
        if len(overseas_subs) == len(subs) or overseas_subs == [None]:
            names.append(name_1)
        else:
            names.extend(f"{n2} ({name_1})" for n2 in overseas_subs)
    return sorted(names)


_subdiv_parts_memo = {}   # (tag, level) -> {GID_n: [part lon/lat arrays]} (full resolution)
_subdiv_ring_memo = {}    # (tag, level) -> (GID_n list, level-0 ring -> subdivision index)
_subdiv_frame_memo = {}   # (tag, excluded, only, projection, center, level) -> projected bbox
_subdiv_parts_frame_memo = {}   # the same key -> one (area, bbox) per part

# GADM subdivisions are handled at level 1 (regions) and level 2 (counties /
# departments / council areas -- what the Shetland Islands are): an id says
# which level it is, so one set of ids can mix them and `subx=` / `subi=`
# need no second parameter. Level 2 is only read when an id asks for it.
MAX_SUBDIV_LEVEL = max(ADMIN_SHAPEFILE_NAMES)


def gid_level(gid: str) -> int:
    """GADM level of an id: "GBR" 0, "GBR.3_1" 1, "GBR.3.27_1" 2."""
    return str(gid).split("_")[0].count(".")


def _gid_path(gid: str) -> tuple:
    """An id as a path, comparable across levels: "GBR.3.27_1" ->
    ("GBR", "3", "27") -- note "GBR.3_1" is *not* a string prefix of it."""
    return tuple(str(gid).split("_")[0].split("."))


def _subdiv_level(*id_sets) -> int:
    """The finest level the given ids need (1 when there are none)."""
    return max([gid_level(g) for ids in id_sets for g in (ids or ())] + [1])


def _covers(gid: str, ids) -> bool:
    """True when `gid` is one of `ids`, or lies inside one of them: an id of
    a coarser level covers its children, so excluding Scotland (GBR.3_1)
    excludes the Shetland Islands (GBR.3.27_1) with it."""
    path = _gid_path(gid)
    return any(path[:len(q)] == q for q in (_gid_path(i) for i in ids))


def list_subdivisions(tag: str, level: int = 1) -> list:
    """A country's GADM subdivisions at `level`: [{"id", "name", "type"}]
    ([] without that level's file or without subdivisions). At level 2 each
    one also carries its level-1 "parent" (Shetland Islands -> Scotland)."""
    if level not in ADMIN_SHAPEFILE_NAMES:
        return []
    entry = _admin_reader(level)
    if entry is None or tag not in entry[1]:
        return []
    reader, index = entry
    with _admin_lock:
        recs = [reader.record(i) for i in index[tag]]
    out = [{"id": r[f"GID_{level}"], "name": r[f"NAME_{level}"], "type": r[f"ENGTYPE_{level}"]} for r in recs]
    if level >= 2:
        for item, r in zip(out, recs):
            item["parent"] = r[f"NAME_{level - 1}"]
    return out


_subdiv_all_memo = {}   # level -> [record tuples], built on the first search


def _fold(text: str) -> str:
    """Lower-case, accents removed (for searching names)."""
    import unicodedata
    return "".join(ch for ch in unicodedata.normalize("NFKD", text or "") if not unicodedata.combining(ch)).lower()


def search_subdivisions(query: str, limit: int = 30, level: int = 1) -> list:
    """GADM subdivisions of every country at `level` matching `query`
    (accent- and case-insensitive, on the name, its alternative names --
    VARNAME_n, e.g. "Republic of Karelia" -- the parent subdivision at level
    2, and the country name / code): names starting with it first.
    [{"id", "name", "type", "tag", "country", "parent" at level 2}].

    The whole level is folded into memory on the first search (level 2 is
    45,962 records, ~3 s and a few tens of MB), then kept."""
    if level not in ADMIN_SHAPEFILE_NAMES:
        return []
    entry = _admin_reader(level)
    if entry is None:
        return []
    if level not in _subdiv_all_memo:
        fields = ["GID_0", "NAME_0", f"GID_{level}", f"NAME_{level}", f"VARNAME_{level}", f"ENGTYPE_{level}"]
        if level >= 2:
            fields.append(f"NAME_{level - 1}")
        with _admin_lock:
            recs = entry[0].records(fields=fields)
        rows = []
        for r in recs:
            name = r[f"NAME_{level}"]
            parent = r[f"NAME_{level - 1}"] if level >= 2 else ""
            rows.append((r[f"GID_{level}"], name, r[f"ENGTYPE_{level}"], r["GID_0"], r["NAME_0"], parent,
                         _fold(name), [_fold(v) for v in (r[f"VARNAME_{level}"] or "").split("|") if v],
                         _fold(r["NAME_0"]), r["GID_0"].lower(), _fold(parent)))
        _subdiv_all_memo[level] = rows
    q = _fold(query.strip())
    if not q:
        return []
    hits = []
    for gid, name, typ, tag, country, parent, fname, fvars, fcountry, ftag, fparent in _subdiv_all_memo[level]:
        if fname.startswith(q):
            rank = 0
        elif q in fname:
            rank = 1
        elif any(q in v for v in fvars):
            rank = 2
        elif q in fcountry or q == ftag or (fparent and q in fparent):
            rank = 3
        else:
            continue
        item = {"id": gid, "name": name, "type": typ, "tag": tag, "country": country}
        if level >= 2:
            item["parent"] = parent
        hits.append((rank, fname, item))
    hits.sort(key=lambda x: (x[0], x[1]))
    return [h[2] for h in hits[:limit]]


def _subdivision_parts(tag: str, level: int = 1) -> dict:
    """{GID_n: [part (lon, lat) arrays]} for one country at `level`, full
    resolution (read via the .shx index; memoized)."""
    if (tag, level) not in _subdiv_parts_memo:
        entry = _admin_reader(level) if level in ADMIN_SHAPEFILE_NAMES else None
        out = {}
        if entry is not None and tag in entry[1]:
            reader, index = entry
            with _admin_lock:
                for i in index[tag]:
                    shp = reader.shape(i)
                    pts = np.asarray(shp.points, dtype=np.float64)
                    parts = list(shp.parts) + [len(pts)]
                    out[reader.record(i)[f"GID_{level}"]] = [pts[a_:b_] for a_, b_ in zip(parts[:-1], parts[1:]) if b_ - a_ >= 3]
        _subdiv_parts_memo[(tag, level)] = out
    return _subdiv_parts_memo[(tag, level)]


# Points per subdivision for the ring -> subdivision KD-tree. Level 1 keeps
# the original 20,000 (results must not move); at level 2 a country has far
# more units (GBR 183, FRA 341), so the budget is shared out instead -- a few
# thousand points per unit is ample to tell which one an island's ring sits on.
_RING_MASK_POINTS = 20_000
_RING_MASK_TOTAL = 400_000


def subdivision_ring_mask(geometries: dict, tag: str, excluded) -> np.ndarray:
    """Bool per level-0 ring of `tag`: False where the ring lies in an
    excluded subdivision (nearest subdivision by majority vote over up to 9
    sample vertices -- they share GADM's vertices). A ring spanning several
    subdivisions (the contiguous USA is one ring) follows the majority, so
    only islands / exclaves (Hawaii, Corsica, the Shetland Islands) can
    really be left out of the drawing. `excluded` may mix levels: the level
    used is the finest one in it, and a coarser id excludes its children."""
    level = _subdiv_level(excluded)
    if (tag, level) not in _subdiv_ring_memo:
        parts = _subdivision_parts(tag, level)
        gids = list(parts)
        rings = geometries[tag]["rings"]
        if len(gids) < 2:
            _subdiv_ring_memo[(tag, level)] = (gids, np.zeros(len(rings), dtype=np.int64))
        else:
            budget = _RING_MASK_POINTS if level <= 1 else max(300, min(_RING_MASK_POINTS,
                                                                       _RING_MASK_TOTAL // len(gids)))
            pts, lab = [], []
            for gi, g in enumerate(gids):
                allp = np.concatenate(parts[g])
                step = max(1, len(allp) // budget)
                pts.append(allp[::step])
                lab.append(np.full(len(pts[-1]), gi))
            tree = cKDTree(_lonlat_to_xyz(np.concatenate(pts)))
            lab = np.concatenate(lab)
            owner_of = np.empty(len(rings), dtype=np.int64)
            for k, (_h, ring) in enumerate(rings):
                sample = ring[np.linspace(0, len(ring) - 1, min(len(ring), 9)).astype(int)]
                _d, nearest = tree.query(_lonlat_to_xyz(sample))
                owner_of[k] = np.bincount(lab[nearest]).argmax()
            _subdiv_ring_memo[(tag, level)] = (gids, owner_of)
    gids, owner_of = _subdiv_ring_memo[(tag, level)]
    if not gids:
        return np.ones(len(geometries[tag]["rings"]), dtype=bool)
    keep = np.array([not _covers(g, excluded) for g in gids])
    return keep[owner_of]


def _subdivision_frame(tag: str, excluded, projection_id: str, center: float, only=None):
    """Projected bbox of a country's kept subdivisions (all of their points:
    exact whatever their shape) -- or of just the `only` ones -- memoized
    per view. The level is the finest one `excluded` / `only` name."""
    level = _subdiv_level(excluded, only)
    key = (tag, tuple(sorted(excluded)), tuple(sorted(only or ())), projection_id, center, level)
    if key not in _subdiv_frame_memo:
        func = PROJECTIONS[projection_id]["func"]
        xs0, ys0, xs1, ys1 = [], [], [], []
        for g, parts in _subdivision_parts(tag, level).items():
            if _covers(g, excluded) or (only is not None and not _covers(g, only)):
                continue
            for part in parts:
                x, y = func(_continuous_relative_lon(part[:, 0], center), part[:, 1])
                xs0.append(x.min()); ys0.append(y.min()); xs1.append(x.max()); ys1.append(y.max())
        _subdiv_frame_memo[key] = ((min(xs0), min(ys0), max(xs1), max(ys1)) if xs0 else None)
    return _subdiv_frame_memo[key]


def _subdivision_frame_parts(tag: str, excluded, projection_id: str, center: float, only=None):
    """The same pieces as _subdivision_frame, but one entry per part:
    [(projected area, x0, y0, x1, y1)] -- so the caller can leave out the
    ones that paint no land (see _pieces_with_no_land). Memoized per view."""
    level = _subdiv_level(excluded, only)
    key = (tag, tuple(sorted(excluded)), tuple(sorted(only or ())), projection_id, center, level)
    if key not in _subdiv_parts_frame_memo:
        func = PROJECTIONS[projection_id]["func"]
        out = []
        for g, parts in _subdivision_parts(tag, level).items():
            if _covers(g, excluded) or (only is not None and not _covers(g, only)):
                continue
            for part in parts:
                x, y = func(_continuous_relative_lon(part[:, 0], center), part[:, 1])
                xy = np.column_stack([x, y])
                out.append((abs(_signed_area(xy)), float(x.min()), float(y.min()),
                            float(x.max()), float(y.max())))
        _subdiv_parts_frame_memo[key] = out
    return _subdiv_parts_frame_memo[key]


def _ring_frame_parts(proj: dict, ring_mask=None):
    """A projected country's rings as framing pieces [(area, x0, y0, x1, y1)]
    (every ring when `ring_mask` is None)."""
    out = []
    for k, (_is_hole, xy) in enumerate(proj["rings"]):
        if ring_mask is not None and not ring_mask[k]:
            continue
        out.append((abs(_signed_area(xy)), float(xy[:, 0].min()), float(xy[:, 1].min()),
                    float(xy[:, 0].max()), float(xy[:, 1].max())))
    return out


def _pieces_with_no_land(frame_groups, biggest, land, win_xmin, win_ymin, scale, h, w):
    """Which framing pieces ended up with no land at all on the finished map
    -- so they should not have set the frame (user: "cleaned up pixels do
    not count for the map centering & auto size").

    This is measured on the rendered map rather than guessed from the
    polygons, because the guess cannot work: Rockall is a 39-point ring that
    paints nothing, while four sub-pixel islets off Sogn rasterize into one
    26 px island that survives. Only the raster knows. A group's largest
    piece is never dropped, so a microstate -- or a country that is one
    small island -- always keeps its frame."""
    out = {}
    for i, g in enumerate(frame_groups):
        gone = set()
        for j, (_area, x0, y0, x1, y1) in enumerate(g):
            if j == biggest[i]:
                continue
            c0 = int(math.floor((x0 - win_xmin) * scale)) - 1
            c1 = int(math.ceil((x1 - win_xmin) * scale)) + 1
            r0 = int(math.floor(h - (y1 - win_ymin) * scale)) - 1
            r1 = int(math.ceil(h - (y0 - win_ymin) * scale)) + 1
            c0, c1 = max(c0, 0), min(c1, w - 1)
            r0, r1 = max(r0, 0), min(r1, h - 1)
            if c0 > c1 or r0 > r1 or not land[r0:r1 + 1, c0:c1 + 1].any():
                gone.add(j)
        out[i] = gone
    return out


def _group_bbox(parts, dropped):
    """Union bbox of a framing group's pieces, minus `dropped` indices."""
    keep = [p for i, p in enumerate(parts) if i not in dropped]
    if not keep:
        return None
    return (min(p[1] for p in keep), min(p[2] for p in keep),
            max(p[3] for p in keep), max(p[4] for p in keep))


def _subdivision_lons(tag: str, excluded, only=None) -> np.ndarray:
    """Longitudes of a country's kept subdivisions -- or of just the `only`
    ones -- (decimated), for choosing the central meridian / seam."""
    level = _subdiv_level(excluded, only)
    lons = [part[::max(1, len(part) // 400), 0] for g, parts in _subdivision_parts(tag, level).items()
            if not _covers(g, excluded) and (only is None or _covers(g, only)) for part in parts]
    return np.concatenate(lons) if lons else np.zeros(0)


def mainland_ring_mask(geometries: dict, tag: str) -> dict:
    """For one country: which of its level-0 rings are mainland, and the
    names of the subdivisions treated as overseas. Results are small and
    cached on disk (keyed on the source shapefiles), so each country's
    first lookup -- up to several seconds for level-2-heavy countries like
    the USA -- is paid once, not once per server run."""
    global _mainland_cache
    with _admin_lock:
        cache_file = _overseas_cache_file()
        if _mainland_cache is None:
            _mainland_cache = {}
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    _mainland_cache = pickle.load(f)
        cached = _mainland_cache.get(tag)
        if cached is not None:
            return cached

        n_rings = len(geometries[tag]["rings"])
        level, regions = _read_subdivisions(tag)
        if len(regions) < 2:
            result = {"ring_mask": np.ones(n_rings, dtype=bool), "overseas_regions": [], "level": level}
        else:
            xyz = [_lonlat_to_xyz(pts) for _n1, _n2, pts, _area in regions]
            areas = np.array([area for _n1, _n2, _pts, area in regions])
            radius = 2.0 * math.sin(OVERSEAS_DISTANCE_KM / (2.0 * EARTH_RADIUS))  # chord length
            is_mainland = _cluster_regions(xyz, areas, radius)

            # Assign each level-0 ring to its nearest subdivision (they share
            # GADM's coastline vertices), by majority vote over a few samples.
            labels = np.concatenate([np.full(len(p), i) for i, p in enumerate(xyz)])
            tree = cKDTree(np.concatenate(xyz))
            mask = np.empty(n_rings, dtype=bool)
            for k, (_is_hole, ring) in enumerate(geometries[tag]["rings"]):
                sample = ring[np.linspace(0, len(ring) - 1, min(len(ring), 9)).astype(int)]
                _d, nearest = tree.query(_lonlat_to_xyz(sample))
                votes = is_mainland[labels[nearest]]
                mask[k] = votes.sum() * 2 >= len(votes)
            result = {"ring_mask": mask, "overseas_regions": _overseas_names(regions, is_mainland),
                      "level": level}

        _mainland_cache[tag] = result
        with open(cache_file, "wb") as f:
            pickle.dump(_mainland_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        return result


# --------------------------------------------------------------------------
# Step 2c: coastline vs land-border edges
# --------------------------------------------------------------------------
# Strait fixing (see fix_straits) needs the coastline -- the boundary of the
# union of all countries -- rather than every ring edge: a land border
# between two countries has land on both sides and must never be treated
# as a potential water passage. GADM neighbours usually share exact border
# vertices, but not always (e.g. France/Monaco), so an edge is classified
# as a land border when its midpoint lies within COAST_BORDER_TOL_KM of
# another country's (densified) boundary, and as coastline otherwise.
# View-independent, so computed once per country and cached on disk.

COAST_BORDER_TOL_KM = 0.1
COAST_CACHE_VERSION = "v7"  # v6: joined own rings -> tolerance test along the join  # v2: + ring neighbours, hole parents; v3: lon/lat-linear edges; v4: packed .bin
_COAST_GRID_DEG = 1.0

_coast_memo = {}  # tag -> [bool ndarray per ring: edge k = vertex k -> k+1 is coastline]
_coast_lock = threading.Lock()


def _coast_cache_dir() -> Path:
    raw = f"{_cache_key(SHAPEFILE_PATH)}|{COAST_CACHE_VERSION}|{COAST_BORDER_TOL_KM}"
    return CACHE_DIR / f"coast_{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def _ring_edges(rings):
    """(start, end) lon/lat arrays of every edge of `rings`, concatenated."""
    p0 = np.concatenate(rings)
    sizes = np.array([len(r) for r in rings])
    succ = np.arange(1, len(p0) + 1)
    ends = np.cumsum(sizes)
    succ[ends - 1] = ends - sizes
    return p0, p0[succ]


def _densify_xyz(a_xyz, b_xyz, spacing: float, return_source: bool = False):
    """Points every <= `spacing` (chord units) along each a->b segment
    (plus, optionally, the index of the segment each point came from)."""
    seg = np.linalg.norm(b_xyz - a_xyz, axis=1)
    n = np.maximum(1, np.ceil(seg / spacing).astype(np.int64))
    idx = np.repeat(np.arange(len(seg)), n)
    t = (np.arange(len(idx)) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
    pts = np.concatenate([a_xyz[idx] + (b_xyz[idx] - a_xyz[idx]) * t[:, None], b_xyz])
    if return_source:
        return pts, np.concatenate([idx, np.arange(len(seg))])
    return pts


def _densify_lonlat_xyz(a, b, spacing_deg: float):
    """Unit vectors of points every <= `spacing_deg` along each lon/lat
    edge a -> b, interpolated *linearly in lon/lat* -- how GADM edges are
    meant (and drawn). Great-circle interpolation instead drifts km off a
    long edge along a parallel (Morocco/Western Sahara, Algeria/Libya), which
    made those borders look like coastline. Also returns each point's edge."""
    d = b - a
    d[:, 0] = (d[:, 0] + 180.0) % 360.0 - 180.0  # shortest way round
    n = np.maximum(1, np.ceil(np.abs(d).max(axis=1) / spacing_deg).astype(np.int64))
    idx = np.repeat(np.arange(len(a)), n)
    t = (np.arange(len(idx)) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
    pts = np.concatenate([a[idx] + d[idx] * t[:, None], b])
    return _lonlat_to_xyz(pts), np.concatenate([idx, np.arange(len(a))])


def _lonlat_midpoints(a, b):
    d = b - a
    d[:, 0] = (d[:, 0] + 180.0) % 360.0 - 180.0
    return a + d / 2.0


def _grid_cells(lonlat: np.ndarray) -> np.ndarray:
    cols = np.floor((lonlat[:, 0] + 180.0) / _COAST_GRID_DEG).astype(np.int64) % int(360 / _COAST_GRID_DEG)
    rows = np.clip(np.floor((lonlat[:, 1] + 90.0) / _COAST_GRID_DEG).astype(np.int64), 0, int(180 / _COAST_GRID_DEG) - 1)
    return rows, cols


def _densify_lonlat_for_grid(a, b):
    """Endpoints plus points every <= half a grid cell along long edges."""
    step = _COAST_GRID_DEG / 2.0
    dlon = np.abs(b[:, 0] - a[:, 0])
    dlon = np.minimum(dlon, 360.0 - dlon)
    seg = np.maximum(dlon, np.abs(b[:, 1] - a[:, 1]))
    long_edges = np.flatnonzero(seg > step)
    extra = [a, b]
    for i in long_edges:  # rare (straight desert/parallel borders), so a loop is fine
        n = int(np.ceil(seg[i] / step))
        t = np.linspace(0.0, 1.0, n + 1)[:, None]
        extra.append(_lonlat_interp(a[i], b[i], t))
    return np.concatenate(extra)


def _lonlat_interp(a, b, t):
    d = b - a
    d[0] = (d[0] + 180.0) % 360.0 - 180.0  # shortest way round
    out = a + d * t
    out[:, 0] = (out[:, 0] + 180.0) % 360.0 - 180.0
    return out


def _point_in_ring(pt, ring) -> bool:
    """Even-odd ray cast (lon/lat, no wrap handling -- used for holes vs
    the exterior rings around them, which never straddle +/-180 in GADM)."""
    x, y = pt
    x0, y0 = ring[:, 0], ring[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crosses = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return bool(np.count_nonzero(crosses & (xs > x)) % 2)


def ring_topology(geometries: dict, tag: str) -> dict:
    """View-independent topology of one country's rings, cached on disk:

    "coastal":     per ring, bool array -- edge k (vertex k -> k+1) is
                   coastline, not a land border / shared edge.
    "neighbours":  set of (ring, other_tag, other_ring): ring pairs joined by
                   land (a shared border with another country, or two of
                   this country's own rings sharing vertices). Island
                   separation never cuts between these.
    "hole_parent": {hole ring: exterior ring containing it}.
    """
    with _coast_lock:
        cached = _coast_memo.get(tag)
        if cached is not None:
            return cached
        cache_dir = _coast_cache_dir()
        cache_file = cache_dir / f"{tag}.bin"
        if cache_file.exists():
            result = _unpack_topology(cache_file.read_bytes())
            _coast_memo[tag] = result
            return result

        world = world_rings(geometries)
        rings = [r for _is_hole, r in geometries[tag]["rings"]]
        is_hole = [h for h, _r in geometries[tag]["rings"]]
        sizes = np.array([len(r) for r in rings])
        ring_of_edge = np.repeat(np.arange(len(rings)), sizes)
        a, b = _ring_edges(rings)
        a_xyz, b_xyz = _lonlat_to_xyz(a), _lonlat_to_xyz(b)

        # Grid cells (dilated by one) touched by this country's boundary;
        # only other countries' edges passing through them can be borders.
        grid = np.zeros((int(180 / _COAST_GRID_DEG), int(360 / _COAST_GRID_DEG)), dtype=bool)
        r_, c_ = _grid_cells(_densify_lonlat_for_grid(a, b))
        grid[r_, c_] = True
        grid = ndimage.binary_dilation(grid, iterations=1)
        grid[:, 0] |= grid[:, -1]  # wrap the dilation across +/-180
        grid[:, -1] |= grid[:, 1]

        tol = 2.0 * math.sin(COAST_BORDER_TOL_KM / (2.0 * EARTH_RADIUS))
        spacing_deg = (COAST_BORDER_TOL_KM / 2.0) / 111.32  # <= tol/2 on the ground everywhere
        pts, pt_tag, pt_ring, other_tags = [], [], [], []
        # Ring-level prefilter: only other countries' rings whose bbox meets
        # a marked cell (O(1) each via a summed-area table) have their edges
        # examined. Previously every edge of every wide-span country (Russia,
        # the USA, Fiji, ...) was scanned for every country -- ~0.3 s each
        # even for Andorra, ~2-3 min for the full warm-up.
        wb, wc = world["ring_bbox"], world["ring_country"]
        n_rows, n_cols = grid.shape
        sat = np.zeros((n_rows + 1, n_cols + 1), dtype=np.int32)
        sat[1:, 1:] = grid.cumsum(axis=0).cumsum(axis=1)
        cells = lambda v, lo, n: np.clip(np.floor((v + lo) / _COAST_GRID_DEG).astype(np.int64), 0, n - 1)
        br0, br1 = cells(wb[:, 1], 90.0, n_rows), cells(wb[:, 3], 90.0, n_rows)
        bc0, bc1 = cells(wb[:, 0], 180.0, n_cols), cells(wb[:, 2], 180.0, n_cols)
        hits = sat[br1 + 1, bc1 + 1] - sat[br0, bc1 + 1] - sat[br1 + 1, bc0] + sat[br0, bc0]
        own_idx = world["tags"].index(tag)
        # Rings naively spanning > 180 deg cross the antimeridian: their bbox
        # is meaningless, so always examine them (only a handful exist).
        adjacent = [world["tags"].index(t) for t in country_adjacency(geometries).get(tag, [])]
        cand = ((hits > 0) | (wb[:, 2] - wb[:, 0] > 180.0)) & np.isin(wc, adjacent)
        cand_rings = np.flatnonzero(cand)
        for ci in np.unique(wc[cand_rings]):
            other = world["tags"][ci]
            data = geometries[other]
            local = cand_rings[wc[cand_rings] == ci] - data["ring_range"][0]
            o_rings = [data["rings"][k][1] for k in local]
            oa, ob = _ring_edges(o_rings)
            ra, ca = _grid_cells(oa)
            rb, cb = _grid_cells(ob)
            near = np.flatnonzero(grid[ra, ca] | grid[rb, cb])
            if not len(near):
                continue
            o_ring_of_edge = np.repeat(local, [len(r) for r in o_rings])[near]
            p, src = _densify_lonlat_xyz(oa[near], ob[near], spacing_deg)
            pts.append(p)
            pt_ring.append(o_ring_of_edge[src])
            pt_tag.append(np.full(len(p), len(other_tags)))
            other_tags.append(other)

        coastal = np.ones(len(a), dtype=bool)
        neighbours = set()
        if pts:
            tree = cKDTree(np.concatenate(pts))
            pt_ring, pt_tag = np.concatenate(pt_ring), np.concatenate(pt_tag)
            mid = _lonlat_to_xyz(_lonlat_midpoints(a, b))
            d, idx = tree.query(mid, distance_upper_bound=tol)
            border = np.isfinite(d)
            coastal &= ~border
            for ring, t_i, o_ring in set(zip(ring_of_edge[border].tolist(), pt_tag[idx[border]].tolist(),
                                             pt_ring[idx[border]].tolist())):
                neighbours.add((ring, other_tags[t_i], o_ring))

        # This country's own rings touching each other (shared vertices, e.g.
        # polygons split along a line): land-joined, and edges between two
        # shared vertices aren't coastline.
        if len(rings) > 1:
            shared_eps = 2.0 * math.sin(0.001 / (2.0 * EARTH_RADIUS))  # 1 m
            pairs = cKDTree(a_xyz).query_pairs(shared_eps, output_type="ndarray")
            if len(pairs):
                ri, rj = ring_of_edge[pairs[:, 0]], ring_of_edge[pairs[:, 1]]
                cross_ring = ri != rj
                pairs = pairs[cross_ring]
                for i, j in set(zip(ri[cross_ring].tolist(), rj[cross_ring].tolist())):
                    neighbours.add((i, tag, j))
                    neighbours.add((j, tag, i))
                shared = np.zeros(len(a), dtype=bool)
                shared[pairs.ravel()] = True
                succ_shared = np.empty(len(a), dtype=bool)  # edge k ends at vertex succ(k)
                ends = np.cumsum(sizes)
                succ = np.arange(1, len(a) + 1)
                succ[ends - 1] = ends - sizes
                succ_shared[:] = shared[succ]
                coastal &= ~(shared & succ_shared)
                # Joined rings don't always split their common line into the
                # same vertices (Russia's mainland and eastern Chukotka meet
                # along 180 deg with 5 vs 3 vertices there), so edges between
                # non-shared vertices would stay "coastline" and island
                # separation would cut a 1 px line along the join. Apply the
                # same midpoint-within-tolerance test as between countries,
                # against each joined ring's densified edges near this ring.
                starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
                joined = {}
                for i, j in set(zip(ri[cross_ring].tolist(), rj[cross_ring].tolist())):
                    joined.setdefault(i, set()).add(j)
                    joined.setdefault(j, set()).add(i)
                for i, js in joined.items():
                    ei = np.arange(starts[i], starts[i] + sizes[i])
                    lo_i = np.minimum(a[ei], b[ei]).min(axis=0) - 0.01
                    hi_i = np.maximum(a[ei], b[ei]).max(axis=0) + 0.01
                    pts = []
                    for j in js:
                        ej = np.arange(starts[j], starts[j] + sizes[j])
                        e_lo, e_hi = np.minimum(a[ej], b[ej]), np.maximum(a[ej], b[ej])
                        lat_ok = (e_hi[:, 1] >= lo_i[1]) & (e_lo[:, 1] <= hi_i[1])
                        # +/-360 too: the same meridian can be written as
                        # 180 in one ring and -180 in the other.
                        lon_ok = np.zeros(len(ej), dtype=bool)
                        for shift in (-360.0, 0.0, 360.0):
                            lon_ok |= (e_hi[:, 0] + shift >= lo_i[0]) & (e_lo[:, 0] + shift <= hi_i[0])
                        near = ej[lat_ok & lon_ok]
                        if len(near):
                            pts.append(_densify_lonlat_xyz(a[near], b[near], spacing_deg)[0])
                    if not pts:
                        continue
                    d_, _i = cKDTree(np.concatenate(pts)).query(
                        _lonlat_to_xyz(_lonlat_midpoints(a[ei], b[ei])), distance_upper_bound=tol)
                    coastal[ei[np.isfinite(d_)]] = False

        # Each hole's parent: the smallest exterior containing its first
        # vertex (bbox prefilter vectorized over the world ring arrays --
        # Canada has ~24k exteriors, which per-ring Python loops made slow).
        hole_parent = {}
        r_lo, r_hi = geometries[tag]["ring_range"]
        bb = world["ring_bbox"][r_lo:r_hi]
        area = np.abs(world["ring_area"][r_lo:r_hi])
        exteriors = np.flatnonzero(~np.array(is_hole))
        for k in np.flatnonzero(is_hole):
            x, y = rings[k][0]
            eb = bb[exteriors]
            cand_e = exteriors[(eb[:, 0] <= x) & (x <= eb[:, 2]) & (eb[:, 1] <= y) & (y <= eb[:, 3])]
            containing = [int(e) for e in cand_e[np.argsort(area[cand_e])] if _point_in_ring((x, y), rings[e])]
            if containing:
                hole_parent[int(k)] = containing[0]

        result = {
            "coastal": np.split(coastal, np.cumsum(sizes)[:-1]),
            "neighbours": neighbours,
            "hole_parent": hole_parent,
        }
        cache_dir.mkdir(exist_ok=True)
        cache_file.write_bytes(_pack_topology(result, sizes))
        _coast_memo[tag] = result
        return result


def _pack_topology(result: dict, sizes) -> bytes:
    """Coastline flags as packed bits (1 bit/edge, not a 1-byte numpy bool),
    zlib-compressed with the rest -- ~7x smaller than pickling the arrays."""
    payload = {
        "sizes": np.asarray(sizes, dtype=np.int64),
        "coastal_bits": np.packbits(np.concatenate(result["coastal"])),
        "neighbours": sorted(result["neighbours"]),
        "hole_parent": result["hole_parent"],
    }
    return zlib.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), 6)


def _unpack_topology(blob: bytes) -> dict:
    payload = pickle.loads(zlib.decompress(blob))
    sizes = payload["sizes"]
    flags = np.unpackbits(payload["coastal_bits"], count=int(sizes.sum())).astype(bool)
    return {
        "coastal": np.split(flags, np.cumsum(sizes)[:-1]),
        "neighbours": set(map(tuple, payload["neighbours"])),
        "hole_parent": payload["hole_parent"],
    }


_adjacency_memo = {}


def country_adjacency(geometries: dict) -> dict:
    """{tag: [tags whose boundary comes within ~1 grid cell (1 deg) of it]},
    built in one pass over the world and saved as a small JSON file
    (cache/adjacency_<hash>.json, a few KB), so each country's topology only
    examines its actual neighbours instead of every other country."""
    key = id(geometries)
    if key in _adjacency_memo:
        return _adjacency_memo[key]
    path = CACHE_DIR / f"adjacency_{_cache_key(SHAPEFILE_PATH)}_{_COAST_GRID_DEG:g}_v2.json"  # v2: no water tags
    if path.exists():
        with open(path, encoding="utf-8") as f:
            adjacency = json.load(f)
    else:
        n_rows, n_cols = int(180 / _COAST_GRID_DEG), int(360 / _COAST_GRID_DEG)
        cell_owner = {}
        for tag, data in geometries.items():
            if tag in WATER_TAGS:
                continue
            a, b = _ring_edges([r for _h, r in data["rings"]])
            r_, c_ = _grid_cells(_densify_lonlat_for_grid(a, b))
            # Dilate by one cell (wrapping across +/-180) so countries that
            # meet on a cell edge, or face each other across a strait, pair up.
            base = np.unique(r_ * n_cols + c_)
            rr, cc = base // n_cols, base % n_cols
            grown = [np.clip(rr + dr, 0, n_rows - 1) * n_cols + (cc + dc) % n_cols
                     for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
            for cell in np.unique(np.concatenate(grown)).tolist():
                cell_owner.setdefault(cell, []).append(tag)
        pairs = {t: set() for t in geometries if t not in WATER_TAGS}
        for owners in cell_owner.values():
            if len(owners) > 1:
                for t in owners:
                    pairs[t].update(owners)
        adjacency = {t: sorted(n - {t}) for t, n in pairs.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(adjacency, f, indent=0, sort_keys=True)
    _adjacency_memo[key] = adjacency
    return adjacency


def coastal_edge_masks(geometries: dict, tag: str) -> list:
    """Per ring of `tag`: bool array, True where edge k (vertex k -> k+1,
    wrapping) is coastline rather than a land border (see ring_topology)."""
    return ring_topology(geometries, tag)["coastal"]


# --------------------------------------------------------------------------
# Step 3: deterministic unique colors
# --------------------------------------------------------------------------

_GOLDEN_ANGLE = 137.508

_color_cache: dict | None = None


def assign_colors(geometries: dict) -> dict:
    global _color_cache
    if _color_cache is not None:
        return _color_cache

    tags = sorted(geometries.keys())
    used = {BACKGROUND_COLOR, LAKE_COLOR, EDGE_LAND_COLOR}
    colors = {}
    for i, tag in enumerate(tags):
        hue = (i * _GOLDEN_ANGLE) % 360.0
        sat = 0.55 + 0.35 * (((i * 928371) % 1000) / 1000.0)  # 0.55-0.90
        val = 0.65 + 0.30 * (((i * 5931) % 1000) / 1000.0)    # 0.65-0.95
        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, sat, val)
        rgb = (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
        nudge = 0
        while rgb in used:
            nudge += 1
            rgb = ((rgb[0] + nudge) % 256, rgb[1], rgb[2])
        used.add(rgb)
        colors[tag] = rgb

    _color_cache = colors
    return colors


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# --------------------------------------------------------------------------
# Step 4: dimension fitting
# --------------------------------------------------------------------------

def allowed_output_sizes():
    """Every (width, height) the renderer accepts: each DIMENSIONS entry in
    both orientations (square ones only once), landscape first."""
    sizes = []
    for w, h in DIMENSIONS:
        for wh in ((w, h), (h, w)):
            if wh not in sizes:
                sizes.append(wh)
    return sizes


def parse_output_size(value):
    """Parse a "WIDTHxHEIGHT" string (or None/""/"auto") into a (w, h) tuple,
    or None meaning "pick automatically". Only sizes from
    `allowed_output_sizes()` are accepted."""
    if value is None or str(value).strip().lower() in ("", "auto"):
        return None
    try:
        w, h = (int(p) for p in str(value).lower().split("x"))
    except ValueError:
        raise RenderError(f"Invalid output size '{value}' (expected WIDTHxHEIGHT or 'auto')")
    if (w, h) not in allowed_output_sizes():
        raise RenderError(f"Unsupported output size {w}x{h}")
    return w, h


CROP_SIDES = ("left", "right", "top", "bottom")
CROP_AXES = (("left", "right"), ("top", "bottom"))
CROP_STEP_PX = 256       # HOI4 wants map sides in whole 256 px steps -- so are DIMENSIONS
CROP_MIN_SIDE_PX = 256   # what a crop must leave of each side


def normalize_crop(crop):
    """A forced crop as {left, right, top, bottom} pixels: whole,
    non-negative, and otherwise free -- balance_crop is what makes the map's
    sides come out right. `crop` may be None, a dict of some or all sides, or
    a (left, right, top, bottom) sequence."""
    if crop is None:
        values = (0, 0, 0, 0)
    elif isinstance(crop, dict):
        unknown = sorted(set(crop) - set(CROP_SIDES))
        if unknown:
            raise RenderError(f"Unknown crop side(s) {unknown}: expected {list(CROP_SIDES)}")
        values = tuple(crop.get(side) or 0 for side in CROP_SIDES)
    else:
        values = tuple(crop)
        if len(values) > len(CROP_SIDES):
            raise RenderError(f"A crop takes at most {len(CROP_SIDES)} values (left, right, top, bottom)")
        values += (0,) * (len(CROP_SIDES) - len(values))
    out = {}
    for side, value in zip(CROP_SIDES, values):
        try:
            n = int(value)
        except (TypeError, ValueError):
            raise RenderError(f"The {side} crop must be a whole number of pixels")
        if n != float(value):
            raise RenderError(f"The {side} crop must be a whole number of pixels")
        if n < 0:
            raise RenderError(f"The {side} crop can't be negative")
        out[side] = n
    return out


def balance_crop(crop):
    """The crop actually taken, for a crop asked in any whole number of
    pixels (user). A map's sides have to stay in whole CROP_STEP_PX steps, so
    what an axis loses is rounded *up* to the next step and **the opposite
    side makes up the difference**: 156 px off the north takes 100 px off the
    south with it. Each axis on its own, and nothing is ever given back --
    the side the user asked for always loses at least what they asked. With
    both sides of an axis asked for, the make-up goes to the smaller of the
    two (the right / bottom one when they are equal), which is the same rule:
    with one side at 0, 0 is the smaller."""
    crop = dict(crop)
    for a, b in CROP_AXES:
        asked = crop[a] + crop[b]
        pad = -asked % CROP_STEP_PX
        if pad:
            crop[b if crop[b] <= crop[a] else a] += pad
    return crop


def choose_output_size(bbox_w: float, bbox_h: float, margin_percent: float, output_size=None, size_factor=1):
    """Pick the output (width, height) -- or use `output_size` if given --
    then contain-fit the margin-padded bbox into it. `size_factor` (1 or
    `BIG_MAP_FACTOR`) multiplies both sides of the chosen size."""
    margin_frac = margin_percent / 100.0
    target_w = bbox_w * (1 + 2 * margin_frac)
    target_h = bbox_h * (1 + 2 * margin_frac)
    aspect = target_w / target_h

    if output_size is not None:
        out_w, out_h = output_size
    else:
        out_w, out_h = min(allowed_output_sizes(),
                           key=lambda wh: abs(math.log(wh[0] / wh[1]) - math.log(aspect)))
    out_w, out_h = out_w * size_factor, out_h * size_factor

    scale = min(out_w / target_w, out_h / target_h) * 0.999  # epsilon safety margin
    shown_w = out_w / scale
    shown_h = out_h / scale
    return out_w, out_h, scale, shown_w, shown_h


# --------------------------------------------------------------------------
# Step 5: rendering
# --------------------------------------------------------------------------

class RenderError(Exception):
    pass


def _scanline_fill(target: np.ndarray, rings_px, value, labels=None, ring_labels=None) -> bool:
    """Even-odd fill of `rings_px` (float pixel coords, pixel (r, c) spans
    [c, c+1) x [r, r+1)) into `target`, setting exactly the pixels whose
    *centre* is inside. Returns True if any pixel was set.

    With `labels` (int32 image) and `ring_labels` (one int per ring), also
    writes which land polygon each pixel belongs to: a span is owned by the
    ring whose crossing opens it -- an exterior ring, or for a span leaving
    a hole, whatever label the caller mapped that hole to (its parent).

    Replaces cv2.fillPoly, which also sets every pixel an edge merely passes
    through: each polygon grew ~0.5 px on every side (a 20x20 square filled
    21x21 = +10%), so any strait or channel under ~1-2 px wide was closed
    and islands merged with the mainland. Edges are crossed with each row's
    centre line using a half-open [ymin, ymax) rule (so every row gets an
    even number of crossings per closed ring), crossings are paired
    left-to-right per row, and the resulting spans are painted through a
    difference array over just the rings' pixel bbox."""
    h, w = target.shape[:2]
    p0 = np.concatenate(rings_px)
    # Each vertex's successor within its own ring (last wraps to first).
    sizes = np.array([len(r) for r in rings_px])
    succ = np.arange(1, len(p0) + 1)
    ends = np.cumsum(sizes)
    succ[ends - 1] = ends - sizes
    x0, y0 = p0[:, 0], p0[:, 1]
    x1, y1 = x0[succ], y0[succ]

    # Rows whose centre (r + 0.5) lies in [min(y), max(y)) of each edge.
    r_lo = np.ceil(np.minimum(y0, y1) - 0.5).astype(np.int64)
    r_hi = np.ceil(np.maximum(y0, y1) - 0.5).astype(np.int64)
    np.clip(r_lo, 0, h, out=r_lo)
    np.clip(r_hi, 0, h, out=r_hi)
    counts = r_hi - r_lo
    edges = np.flatnonzero(counts > 0)
    if len(edges) == 0:
        return False
    counts = counts[edges]
    total = int(counts.sum())
    starts = np.repeat(r_lo[edges], counts)
    offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    rows = starts + offsets
    e = np.repeat(edges, counts)
    xs = x0[e] + (rows + 0.5 - y0[e]) * (x1[e] - x0[e]) / (y1[e] - y0[e])

    order = np.lexsort((xs, rows))
    rows, xs = rows[order], xs[order]
    span_rows = rows[0::2]
    # Columns whose centre (c + 0.5) lies in [x_left, x_right).
    c_lo = np.clip(np.ceil(xs[0::2] - 0.5).astype(np.int64), 0, w)
    c_hi = np.clip(np.ceil(xs[1::2] - 0.5).astype(np.int64), 0, w)
    keep = c_hi > c_lo
    if not keep.any():
        return False
    span_rows, c_lo, c_hi = span_rows[keep], c_lo[keep], c_hi[keep]
    span_label = None
    if labels is not None:
        ring_of_vertex = np.repeat(np.arange(len(rings_px)), sizes)
        span_label = np.asarray(ring_labels, dtype=np.int64)[ring_of_vertex[e[order][0::2]]][keep]

    r0, r1 = int(span_rows.min()), int(span_rows.max()) + 1
    q0, q1 = int(c_lo.min()), int(c_hi.max())
    bw = q1 - q0 + 1
    local = (span_rows - r0) * bw
    # Spans within one row are disjoint, so the running sum is only ever 0/1.
    diff = np.zeros((r1 - r0) * bw, dtype=np.int8)
    np.add.at(diff, local + (c_lo - q0), 1)
    np.add.at(diff, local + (c_hi - q0), -1)
    mask = np.cumsum(diff.reshape(r1 - r0, bw), axis=1, dtype=np.int8)[:, :-1].view(bool)
    view = target[r0:r1, q0:q1]
    np.copyto(view, np.asarray(value, dtype=target.dtype), where=mask[..., None] if view.ndim == 3 else mask)
    if span_label is not None:
        # Same difference-array trick with the span's label as the step.
        ldiff = np.zeros((r1 - r0) * bw, dtype=np.int64)
        np.add.at(ldiff, local + (c_lo - q0), span_label)
        np.add.at(ldiff, local + (c_hi - q0), -span_label)
        lab = np.cumsum(ldiff.reshape(r1 - r0, bw), axis=1)[:, :-1]
        np.copyto(labels[r0:r1, q0:q1], lab.astype(labels.dtype), where=mask)
    return True


# --------------------------------------------------------------------------
# Lakes (HydroLAKES v1.0): cut out of the land right after the fill
# --------------------------------------------------------------------------

_lake_index_memo = {}
_lake_proj_memo = {}  # (projection_id, center) -> projected lake bboxes + area factor (last 2 kept)
_lake_lock = threading.Lock()
_KM_PER_DEG = EARTH_RADIUS * math.pi / 180.0
CASPIAN_LAKE_ID = 1  # HydroLAKES Hylak_id


def _dbf_numeric_column(path: Path, field: str) -> np.ndarray:
    """One numeric column of a .dbf, parsed with numpy over the fixed-width
    records (pyshp's per-record parsing takes ~20 s for HydroLAKES' 1.4M)."""
    dbf = np.memmap(path, dtype=np.uint8, mode="r")
    n = int(dbf[4:8].view("<u4")[0])
    hlen, rlen = int(dbf[8:10].view("<u2")[0]), int(dbf[10:12].view("<u2")[0])
    pos, o = 1, 32  # byte 0 of each record is the deletion flag
    while dbf[o] != 0x0D:
        name, size = bytes(dbf[o:o + 11]).split(b"\0")[0].decode(), int(dbf[o + 16])
        if name == field:
            recs = dbf[hlen:hlen + n * rlen].reshape(n, rlen)
            col = recs[:, pos:pos + size].copy().view(f"S{size}").ravel()
            return np.char.strip(col).astype(np.float64)
        pos += size
        o += 32
    raise KeyError(f"{path.name}: no field {field}")


def load_lake_index(path: Path = LAKES_SHAPEFILE_PATH):
    """Per-lake index of the HydroLAKES shapefile, or None if it hasn't been
    downloaded. The 1.1 GB .shp is memory-mapped, not loaded: only each
    record's header is read up front (bbox, part/point counts: ~0.5 s for
    1.43M lakes, from the .shx offsets) plus Lake_area from the .dbf; a
    render then reads just the points of the lakes it draws."""
    if not path.exists():
        return None
    with _lake_lock:
        if path not in _lake_index_memo:
            shx = np.fromfile(path.with_suffix(".shx"), dtype=">i4", offset=100).reshape(-1, 2)
            raw = np.memmap(path, dtype=np.uint8, mode="r")
            off = shx[:, 0].astype(np.int64) * 2 + 8  # 16-bit words -> bytes, skip record header
            hdr = raw[off[:, None] + np.arange(44)]
            _lake_index_memo[path] = {
                "raw": raw, "off": off,
                "shape_type": hdr[:, 0:4].copy().view("<i4").ravel(),
                "bbox": hdr[:, 4:36].copy().view("<f8"),  # lon0, lat0, lon1, lat1
                "n_parts": hdr[:, 36:40].copy().view("<i4").ravel().astype(np.int64),
                "n_points": hdr[:, 40:44].copy().view("<i4").ravel().astype(np.int64),
                "area_km2": _dbf_numeric_column(path.with_suffix(".dbf"), "Lake_area"),
                "lake_id": _dbf_numeric_column(path.with_suffix(".dbf"), "Hylak_id").astype(np.int64),
            }
        return _lake_index_memo[path]


def _lake_projection(index, projection_id: str, center: float):
    """Projected bbox (x0, y0, x1, y1) of every lake, and the projection's
    area factor (units^2 per km^2) at each lake's centre. Exact bbox for the
    5 projections here (see _estimate_projected_bbox): x is evaluated at the
    lake's lowest and highest |lat|, y at its lat range ends. Longitude is
    continuous from lon0, so a lake straddling the seam runs past +/-180
    (drawn at that edge only; no wrapped copy -- lakes there are few and
    tiny)."""
    key = (projection_id, center)
    with _lake_lock:
        if key in _lake_proj_memo:
            return _lake_proj_memo[key]
    func = PROJECTIONS[projection_id]["func"]
    lon0, lat0, lon1, lat1 = index["bbox"].T
    rel0 = _relative_lon(lon0, center)
    rel1 = rel0 + (lon1 - lon0)
    lat_lo = np.where(lat0 > 0, lat0, np.where(lat1 < 0, lat1, 0.0))  # lowest |lat|
    lat_hi = np.where(np.abs(lat0) > np.abs(lat1), lat0, lat1)       # highest |lat|
    xs = [func(r, la)[0] for r in (rel0, rel1) for la in (lat_lo, lat_hi)]
    _, y0 = func(rel0, lat0)
    _, y1 = func(rel0, lat1)
    latc = np.clip((lat0 + lat1) / 2.0, -89.0, 89.0)
    # Every projection here has y depending on latitude only, so the
    # Jacobian is (dx/dlon) * (dy/dlat).
    dxdlon = np.abs(func(rel0 + 0.5, latc)[0] - func(rel0 - 0.5, latc)[0])
    dydlat = np.abs(func(rel0, latc + 0.01)[1] - func(rel0, latc - 0.01)[1]) / 0.02
    result = {
        "bbox": np.column_stack([np.minimum.reduce(xs), np.minimum(y0, y1), np.maximum.reduce(xs), np.maximum(y0, y1)]),
        "area_factor": dxdlon * dydlat / (_KM_PER_DEG ** 2 * np.cos(np.radians(latc))),
    }
    with _lake_lock:
        _lake_proj_memo[key] = result
        while len(_lake_proj_memo) > 2:
            _lake_proj_memo.pop(next(iter(_lake_proj_memo)))
    return result


def cut_lakes(canvas, projection_id: str, center: float, win_xmin: float, win_ymin: float, scale: float,
              min_px: int):
    """Rasterize every HydroLAKES lake in view and cut it out of the drawn
    land (canvas pixels that aren't background). Lakes are kept where the
    resulting lake area (4-connected, so touching lakes count together) is
    at least `min_px` px; they become background (water) in `canvas`, so
    the later land/sea steps treat them as water. Lakes over sea or hidden
    countries are ignored. The Caspian (HydroLAKES' lake #1) is skipped:
    it is sea here (GADM's XCA), and its HydroLAKES shoreline differs from
    GADM's, so cutting it would paint lake along the GADM coast.

    Lakes too small to reach `min_px` / 4 px of area (from Lake_area and the
    projection's local scale) are skipped before any point is read, which
    leaves a few thousand lakes to draw in most views.
    Returns (report dict, lake mask, every lake drawn on land incl. the
    ones too small to keep -- the Terrain tab's marsh; both None if
    HydroLAKES isn't downloaded)."""
    index = load_lake_index()
    if index is None:
        return {"enabled": True, "available": False}, None, None
    h, w = canvas.shape[:2]
    lp = _lake_projection(index, projection_id, center)
    bx0, by0, bx1, by1 = lp["bbox"].T
    win_xmax, win_ymax = win_xmin + w / scale, win_ymin + h / scale
    area_px = index["area_km2"] * lp["area_factor"] * scale * scale
    sel = np.flatnonzero((bx1 >= win_xmin) & (bx0 <= win_xmax) & (by1 >= win_ymin) & (by0 <= win_ymax)
                         & (area_px >= min_px / 4.0) & (index["shape_type"] == 5)
                         & (index["lake_id"] != CASPIAN_LAKE_ID))
    lakes = np.zeros((h, w), dtype=bool)
    if len(sel):
        func = PROJECTIONS[projection_id]["func"]
        raw, off, n_parts, n_points = index["raw"], index["off"], index["n_parts"], index["n_points"]
        rings_px = []
        for i in sel:
            p0 = off[i] + 44 + 4 * n_parts[i]
            parts = np.append(raw[off[i] + 44:p0].view("<i4"), n_points[i])
            pts = np.frombuffer(raw[p0:p0 + 16 * n_points[i]], dtype="<f8").reshape(-1, 2)
            for a, b in zip(parts[:-1], parts[1:]):
                ring = pts[a:b]
                x, y = func(_continuous_relative_lon(ring[:, 0], center), ring[:, 1])
                rings_px.append(np.column_stack([(x - win_xmin) * scale, h - (y - win_ymin) * scale]))
        _scanline_fill(lakes, rings_px, True)
    lakes &= ~(canvas == np.asarray(BACKGROUND_COLOR, dtype=canvas.dtype)).all(axis=2)
    drawn = lakes.copy()
    lab, n = ndimage.label(lakes, structure=_CONNECTIVITY_4)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    big = sizes >= min_px
    big[0] = False
    lakes = big[lab]
    canvas[lakes] = BACKGROUND_COLOR
    return {"enabled": True, "available": True, "lakes": int(big.sum()), "lake_px": int(lakes.sum()),
            "drawn_polygons": int(len(sel))}, lakes, drawn


def lake_id_image(view, lake_mask, min_px: int = 8):
    """HydroLAKES lake index (+1; 0 = none) of every pixel of `lake_mask`,
    re-rasterizing the lakes in view like cut_lakes (same selection) with
    the fill's per-ring labels -- the Provinces tab joins fragments of the
    same lake only (a thin reservoir drawn as a dashed line: Kama)."""
    index = load_lake_index()
    h, w = view["height"], view["width"]
    out = np.zeros((h, w), dtype=np.int32)
    if index is None or lake_mask is None or not lake_mask.any():
        return out
    projection_id, center, scale = view["projection"], view["center"], view["scale"]
    win_xmin, win_ymin = view["win_xmin"], view["win_ymin"]
    lp = _lake_projection(index, projection_id, center)
    bx0, by0, bx1, by1 = lp["bbox"].T
    win_xmax, win_ymax = win_xmin + w / scale, win_ymin + h / scale
    area_px = index["area_km2"] * lp["area_factor"] * scale * scale
    sel = np.flatnonzero((bx1 >= win_xmin) & (bx0 <= win_xmax) & (by1 >= win_ymin) & (by0 <= win_ymax)
                         & (area_px >= min_px / 4.0) & (index["shape_type"] == 5)
                         & (index["lake_id"] != CASPIAN_LAKE_ID))
    if not len(sel):
        return out
    func = PROJECTIONS[projection_id]["func"]
    raw, off, n_parts, n_points = index["raw"], index["off"], index["n_parts"], index["n_points"]
    rings_px, ring_labels = [], []
    for k, i in enumerate(sel, start=1):
        p0 = off[i] + 44 + 4 * n_parts[i]
        parts = np.append(raw[off[i] + 44:p0].view("<i4"), n_points[i])
        pts = np.frombuffer(raw[p0:p0 + 16 * n_points[i]], dtype="<f8").reshape(-1, 2)
        for a, b in zip(parts[:-1], parts[1:]):
            ring = pts[a:b]
            x, y = func(_continuous_relative_lon(ring[:, 0], center), ring[:, 1])
            rings_px.append(np.column_stack([(x - win_xmin) * scale, h - (y - win_ymin) * scale]))
            ring_labels.append(k)
    filled = np.zeros((h, w), dtype=bool)
    _scanline_fill(filled, rings_px, True, labels=out, ring_labels=np.array(ring_labels, dtype=np.int32))
    out[~lake_mask] = 0
    return out


def _classify_lake_water(water, lakes, sea_before):
    """Final lake pixels: water that was lake, plus water the land/sea steps
    created next to a lake that isn't connected to the original sea (e.g. an
    erased islet or a separation gap inside a lake). `sea_before` is the
    water (without lakes) right after the lakes were cut."""
    lake_now = water & lakes
    other = water & ~lakes
    lab, n = ndimage.label(other, structure=_CONNECTIVITY_4)
    if n == 0:
        return lake_now
    has_sea = np.zeros(n + 1, dtype=bool)
    has_sea[np.unique(lab[sea_before & other])] = True
    near_lake = ndimage.binary_dilation(lake_now, structure=_CONNECTIVITY_4)
    touches = np.zeros(n + 1, dtype=bool)
    touches[np.unique(lab[near_lake & other])] = True
    joins = touches & ~has_sea
    joins[0] = False
    return lake_now | joins[lab]


def _merge_sea_lakes(lakes, sea):
    """Lakes side by side with the sea become sea (a lagoon open to it:
    IJsselmeer, Sivash, ...). Returns (remaining lakes, lakes merged)."""
    lab, n = ndimage.label(lakes, structure=_CONNECTIVITY_4)
    if n == 0:
        return lakes, 0
    touching = np.unique(lab[ndimage.binary_dilation(sea, structure=_CONNECTIVITY_4) & lakes])
    touching = touching[touching > 0]
    if not len(touching):
        return lakes, 0
    return lakes & ~np.isin(lab, touching), int(len(touching))


def _separate_diagonal_water(canvas, lakes, water, hidden_land):
    """No two water bodies may touch only corner to corner when one is a
    lake (lake / lake or lake / sea): at each such contact the lake pixel
    -- the smaller lake's if both are lakes -- becomes land, in the colour
    of a land pixel beside it (not a hidden country's). Repeated until none
    is left. In place (canvas: water pixels are background here; `water`
    and `lakes` are updated); returns the number of pixels turned to land."""
    bg = np.asarray(BACKGROUND_COLOR, dtype=canvas.dtype)
    h, w = water.shape
    changed = 0
    for _pass in range(10):
        lab, n = ndimage.label(water, structure=_CONNECTIVITY_4)
        size = np.bincount(lab.ravel(), minlength=n + 1)
        is_lake = np.zeros(n + 1, dtype=bool)
        is_lake[np.unique(lab[lakes])] = True
        is_lake[0] = False
        cands = []
        for (A, D, B, C) in (((0, 0), (1, 1), (0, 1), (1, 0)), ((0, 1), (1, 0), (0, 0), (1, 1))):
            la = lab[A[0]:h - 1 + A[0], A[1]:w - 1 + A[1]]
            ld = lab[D[0]:h - 1 + D[0], D[1]:w - 1 + D[1]]
            lb = lab[B[0]:h - 1 + B[0], B[1]:w - 1 + B[1]]
            lc = lab[C[0]:h - 1 + C[0], C[1]:w - 1 + C[1]]
            m = (la > 0) & (ld > 0) & (la != ld) & (lb == 0) & (lc == 0) & (is_lake[la] | is_lake[ld])
            for r, c in zip(*np.nonzero(m)):
                cands.append(((r + A[0], c + A[1]), (r + D[0], c + D[1]), (r + B[0], c + B[1]), (r + C[0], c + C[1])))
        if not cands:
            break
        n_pass = 0
        for pa, pd, pb, pc in cands:
            if not (water[pa] and water[pd]) or water[pb] or water[pc]:
                continue  # already resolved by an earlier change this pass
            la_, ld_ = lab[pa], lab[pd]
            if is_lake[la_] and is_lake[ld_]:
                q = pa if size[la_] <= size[ld_] else pd
            else:
                q = pa if is_lake[la_] else pd
            land = [x for x in (pb, pc) if not hidden_land[x] and not (canvas[x] == bg).all()]
            if not land:
                continue
            canvas[q] = canvas[land[0]]
            water[q] = False
            lakes[q] = False
            n_pass += 1
        changed += n_pass
        if not n_pass:
            break
    return changed


def _drop_small_lakes(canvas, lakes, sea, min_px: int):
    """Lake areas (4-connected) that the land/sea steps shrank under
    `min_px` (diagonal bridges run after the cut and may take a lake pixel)
    are removed: merged into the sea if they touch it (a lagoon remnant),
    else filled with the most common land colour around them. Returns the
    remaining lake mask."""
    lab, n = ndimage.label(lakes, structure=_CONNECTIVITY_4)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    small = sizes < min_px
    small[0] = False
    if not small.any():
        return lakes
    h, w = lakes.shape
    bg = np.asarray(BACKGROUND_COLOR, dtype=canvas.dtype)
    lakes = lakes.copy()
    for i, sl in enumerate(ndimage.find_objects(np.where(small[lab], lab, 0)), start=1):
        if sl is None:
            continue
        r0, r1 = max(0, sl[0].start - 1), min(h, sl[0].stop + 1)
        c0, c1 = max(0, sl[1].start - 1), min(w, sl[1].stop + 1)
        comp = lab[r0:r1, c0:c1] == i
        ring = ndimage.binary_dilation(comp, structure=_CONNECTIVITY_4) & ~comp
        px = canvas[r0:r1, c0:c1]
        lakes[r0:r1, c0:c1][comp] = False
        land_ring = ring & ~(px == bg).all(axis=2) & ~lakes[r0:r1, c0:c1]
        if sea[r0:r1, c0:c1][ring].any() or not land_ring.any():
            px[comp] = BACKGROUND_COLOR
            continue
        cols = px[land_ring].astype(np.int32)
        codes, counts = np.unique((cols[:, 0] << 16) | (cols[:, 1] << 8) | cols[:, 2], return_counts=True)
        code = int(codes[np.argmax(counts)])
        px[comp] = ((code >> 16) & 255, (code >> 8) & 255, code & 255)
    return lakes


# Edge bands (HOI4): left edge = sea, unowned land, sea; right edge = sea.
# The land strip is impassable in the game (no movement across the map's
# wrap), and no land touches the left / right edge (game errors). Nothing
# along the top / bottom.
EDGE_BAND_PX_DEFAULT = 4  # width of each band (sea, land, sea / sea)
EDGE_BAND_PX_MIN = 2
EDGE_BAND_PX_MAX = 10

# A world map has no margin -- it fills the full 360 degrees edge to edge --
# so an island sitting on the seam paints right up against the border and the
# game, which builds its province adjacency across the wrap, joins it to
# whatever is on the other side. One column of water on each border is always
# forced there (user), whatever the edge bands are set to: enough for the
# wrap's 2x2 check to see water against water, and it costs a single pixel of
# longitude. The edge bands, where they are on, already contain it.
WORLD_WRAP_BAND_PX = 1


# --------------------------------------------------------------------------
# Sea level rise
# --------------------------------------------------------------------------

# Raising the sea level turns every land pixel at or below the new level
# into sea, *before* any of the land/sea steps, so lakes, island
# separation, strait fixing, speck filling and the sieve all clean up the
# new coastline (user: "the cleanup must happen afterwards").
#
# The elevation is the **real ground** (elevation_m), never the exaggerated
# HOI4 heights: the exaggeration is cosmetic (it makes the Scandes look like
# the Alps) and would drown the wrong land. With GMTED's tiles downloaded
# that is metres, so a 1 m rise means 1 m; on the 8-bit NASA fallback the
# grey is ~linear at NASA_M_PER_UNIT (~26.5 m) per step and clipped to whole
# steps, so the rise lands on the nearest one -- anything under ~13 m goes at
# the first step (measured against GMTED: a 1 m rise then drowns 86% of the
# Camargue and 98% of coastal Flanders), then in jumps of ~26.5 m. The report
# carries which source answered. The depth of the water now standing over the
# drowned land is kept per pixel (debug["drowned_depth"], uint16 metres) --
# the bathymetry has no soundings there, it is land to GEBCO (drowned_depth_m).
SEA_LEVEL_MAX_M = 8000


def drown_land(canvas, hidden_land, poly_labels, view, rise_m):
    """Drown every land pixel of `canvas` whose real elevation is at or
    below `rise_m` metres. Modifies canvas / hidden_land / poly_labels in
    place; returns (report, drowned mask, depth in metres as uint16)."""
    if not elevation_available():
        raise RenderError("Raising the sea level needs the elevations: download GMTED2010 (for metre "
                          "precision) or the NASA topography from the Download tab.")
    t0 = time.time()
    rise_m = float(rise_m)
    z, zrep = elevation_m({"view": view})
    land = ~(canvas == np.asarray(BACKGROUND_COLOR, dtype=canvas.dtype)).all(axis=2)
    if hidden_land is not None:
        land |= hidden_land
    land_px = int(land.sum())
    drowned = land & (z <= rise_m)
    depth = np.zeros(drowned.shape, dtype=np.uint16)
    depth[drowned] = np.clip(np.rint(rise_m - z[drowned]), 0, 65535).astype(np.uint16)
    canvas[drowned] = BACKGROUND_COLOR
    if hidden_land is not None:
        hidden_land &= ~drowned
    if poly_labels is not None:
        poly_labels[drowned] = 0
    n = int(drowned.sum())
    report = {"enabled": True, "sea_level_m": round(rise_m, 1), "drowned_px": n, "land_px_before": land_px,
              "drowned_share": round(n / land_px, 4) if land_px else 0.0,
              "elevation": zrep.get("source"), "gmted_share": zrep.get("gmted_share"),
              "gmted_arcsec": zrep.get("gmted_arcsec"),
              "step_m": 1.0 if zrep.get("gmted_share") == 1.0 else round(NASA_M_PER_UNIT, 1),
              "max_depth_m": int(depth.max()) if n else 0,
              "seconds": round(time.time() - t0, 2)}
    return report, drowned, (depth if n else None)


def drowned_depth_m(debug):
    """Water depth in metres over the land a sea level rise drowned, 0
    everywhere else (None when nothing drowned). GEBCO reads that land as
    land -- 0 m deep --, so every depth reading of this frame has to be
    taken as the larger of the two (see hoi4mod.write_water_colormaps,
    strategic._add_naval_terrain)."""
    d = (debug or {}).get("drowned_depth")
    return None if d is None else np.asarray(d, dtype=np.float32)


def _edge_band_masks(h: int, w: int, on: bool, px: int = EDGE_BAND_PX_DEFAULT, world: bool = False):
    """(band, land) bool masks of the edge bands, or (None, None) when there
    is nothing to paint. `band` = every column painted sea over whatever the
    frame shows (the output size doesn't change) before the land/sea steps,
    and enforced again after; `land` = the unowned land strip inside the left
    band, painted EDGE_LAND_COLOR at the very end (so no step joins it to
    real land: sea always separates the two), None when the bands are off.
    `world` adds WORLD_WRAP_BAND_PX of water to both borders on its own, so
    a world map never has land against the wrap even with the bands off."""
    if not on and not world:
        return None, None
    band = np.zeros((h, w), dtype=bool)
    land = None
    if on:
        band[:, :3 * px] = True
        band[:, -px:] = True
        land = np.zeros((h, w), dtype=bool)
        land[:, px:2 * px] = True
    if world:
        band[:, :WORLD_WRAP_BAND_PX] = True
        band[:, -WORLD_WRAP_BAND_PX:] = True
    return band, land


def render_map(tags, projection_id: str, margin_percent: float = DEFAULT_MARGIN_PERCENT,
                excluded_tags=None, output_size=None, ignore_overseas=False, land_sea=None, progress=print,
                color_config=None, merge_autonomous=False, center_tag=None, world=False,
                include_antarctica=False, edge_bands=False, edge_band_px=EDGE_BAND_PX_DEFAULT, excluded_subdivisions=None,
                included_subdivisions=None, double_size=False, sea_level_m=0, removed_tags=None,
                crop=None):  # HISTORICAL (shelved, see historical.py): add historical=None
    """Render the map; returns (PIL image, meta dict). `land_sea` is a
    Land/Sea settings dict (see LAND_SEA_DEFAULTS); None loads the saved
    settings (land_sea_settings.json). `color_config` is a colour
    configuration (see palettes.load_config); None loads the active one.
    `merge_autonomous` draws autonomous territories (palettes.AUTONOMOUS_PARENT:
    Aland, Faroe, Greenland, Hong Kong, ...) as part of their country.
    `center_tag` centres the map on that country (central meridian + a
    symmetric frame); `world` frames the whole world (tags may be empty),
    with no margin, the edge in the Bering Strait unless `center_tag` is
    given, and Antarctica left out unless `include_antarctica`.
    `edge_bands` adds the HOI4 edge bands, each `edge_band_px` wide
    (left: sea, unowned land, sea; right: sea) -- see _edge_band_masks. `excluded_subdivisions` lists GADM
    level-1 or level-2 ids ("USA.12_1" = Hawaii, "GBR.3.27_1" = the Shetland
    Islands; the level is read off the id): a selected country is
    then framed on its other subdivisions only, and its parts lying in the
    excluded ones are left out of the map -- sea (see subdivision_ring_mask).
    `included_subdivisions` (GID_1 of countries *not* selected, e.g.
    "RUS.26_1" = Karelia) are added to the framing (Finland + Karelia); their
    country is drawn as any neighbour. Tags may be empty when some are given.
    `double_size` doubles both sides of the output (BIG_MAP_FACTOR): a map
    for a hoi4.exe patched with the map size unlocker. `sea_level_m` raises
    the sea by that many metres before the land/sea steps (0 = off, see
    drown_land). `removed_tags` are countries taken off the map entirely --
    not drawn, not framed and *not* land either, so their area is plain sea
    on every tab (`excluded_tags` only hides them: see render_map_detailed).
    `crop` forces whole 256 px bands off the borders (see normalize_crop),
    world maps included."""
    # HISTORICAL (shelved, see historical.py): `historical` = id of a
    # historical border map (historical_maps/<id>/): its focus country (1939
    # Poland) gets its historical extent; today's land of that country outside
    # it goes to the nearest other country. Coastlines + other countries: GADM.
    image, meta, _debug = render_map_detailed(tags, projection_id, margin_percent, excluded_tags,
                                              output_size, ignore_overseas, land_sea, progress=progress,
                                              removed_tags=removed_tags,
                                              color_config=color_config, merge_autonomous=merge_autonomous,
                                              center_tag=center_tag, world=world,
                                              include_antarctica=include_antarctica, edge_bands=edge_bands, edge_band_px=edge_band_px,
                                              excluded_subdivisions=excluded_subdivisions,
                                              included_subdivisions=included_subdivisions,
                                              double_size=double_size, sea_level_m=sea_level_m,
                                              crop=crop)
                                              # HISTORICAL (shelved, see historical.py): historical=historical
    return image, meta


class _ToPx:
    """lon/lat -> pixel (x, y) of a render's frame. A class, not a closure,
    so a render's debug dict can be pickled (the app's disk cache)."""

    def __init__(self, projection_id, central_meridian, win_xmin, win_ymin, scale, out_h):
        self.projection_id, self.central_meridian = projection_id, central_meridian
        self.win_xmin, self.win_ymin, self.scale, self.out_h = win_xmin, win_ymin, scale, out_h

    def __call__(self, lon, lat):
        func = PROJECTIONS[self.projection_id]["func"]
        x, y = func(_relative_lon(np.array([lon]), self.central_meridian), np.array([lat]))
        return (float(x[0]) - self.win_xmin) * self.scale, self.out_h - (float(y[0]) - self.win_ymin) * self.scale


def render_map_detailed(tags, projection_id: str, margin_percent: float = DEFAULT_MARGIN_PERCENT,
                        excluded_tags=None, output_size=None, ignore_overseas=False, land_sea=None,
                        progress=print, color_config=None, merge_autonomous=False, center_tag=None,
                        world=False, include_antarctica=False, edge_bands=False, edge_band_px=EDGE_BAND_PX_DEFAULT,
                        excluded_subdivisions=None, included_subdivisions=None, double_size=False,
                        sea_level_m=0, removed_tags=None, crop=None, _frame_drop=None):  # HISTORICAL (shelved, see historical.py): historical=None
    """render_map, plus the Land/Sea debug masks (None when strait fixing
    is off) as a third return value.

    The two ways to take a country off the map differ in what is left behind:
    `excluded_tags` **hides** it -- it keeps its land (unpainted on the canvas
    but in `debug["hidden_land"]`, so the land/sea steps, the heightmap, the
    terrain and the provinces all still see land there, and the export makes
    it impassable wasteland), which is what keeps a strait a strait.
    `removed_tags` **removes** it -- nothing of it is drawn, framed or even
    collected as coastline, so its area is plain sea on every tab, exactly
    like an excluded subdivision. Known limit, shared with `subx`: a
    neighbour's border with a removed country is a land border in the GADM
    topology, so that newly exposed shore is not seen as coastline by strait
    fixing or island separation.

    `crop` ({left, right, top, bottom} px, any whole number: balance_crop
    rounds what each axis loses up to a whole CROP_STEP_PX and takes the
    difference off the opposite side, so 156 px off the top takes 100 px off
    the bottom with it) forces those bands off the borders, as the render's
    last step -- after every land/sea step and after the framing, so the crop can
    never feed back into the frame or into the size it is subtracted from.
    The whole render is cut with the canvas (the masks in `debug`, the
    viewport, `to_px`, the legend, the preview markers) and the edge bands
    are re-laid on the new borders, since a crop can take one clean off and
    land against the left or right edge crashes the game. It works on a
    world map too, where a horizontal crop cuts real longitude away: the map
    then covers less than the full 360 degrees."""
    import palettes  # local import: palettes imports this module
    land_sea = load_land_sea_settings() if land_sea is None else normalize_land_sea(land_sea)
    fix = land_sea["fix_straits"]
    separate = land_sea["separate_islands"]
    island_ops = land_sea["suppress_islands"] or land_sea["grow_islands"]
    connect = land_sea["connect_diagonals"]
    specks = land_sea["fill_sea_specks"]
    micro_on = land_sea["grow_microstates"]
    sieve = land_sea["sieve_regions"]
    # HISTORICAL (shelved, see historical.py):
    # hmap = None
    # if historical:
    #     import historical as _historical  # local import: historical imports this module
    #     try:
    #         hmap = _historical.load_map(historical)
    #     except KeyError:
    #         raise RenderError(f"Unknown historical border map '{historical}'")
    # # Historical borders repaint land incl. hidden countries' (a hidden
    # # modern country may be another country's land in the map).
    # need_hidden = fix or island_ops or connect or specks or micro_on or sieve
    need_hidden = fix or island_ops or connect or specks or micro_on or sieve  # HISTORICAL: ... or hmap is not None
    if isinstance(tags, str):
        tags = [tags]
    tags = [t.upper() for t in tags]
    center_tag = center_tag.upper() if center_tag else None
    included_subdivisions = [str(g).strip() for g in (included_subdivisions or ()) if str(g).strip()]
    if not tags and not world and not included_subdivisions:
        raise RenderError("At least one country or subdivision must be selected (or use the world map)")
    if projection_id not in PROJECTIONS:
        raise RenderError(f"Unknown projection '{projection_id}'")
    edge_bands = bool(edge_bands)
    edge_band_px = int(edge_band_px)
    try:
        sea_level_m = int(sea_level_m or 0)
    except (TypeError, ValueError):
        raise RenderError("The sea level rise must be a whole number of metres")
    if not 0 <= sea_level_m <= SEA_LEVEL_MAX_M:
        raise RenderError(f"The sea level rise must be 0 to {SEA_LEVEL_MAX_M} m")
    if not EDGE_BAND_PX_MIN <= edge_band_px <= EDGE_BAND_PX_MAX:
        raise RenderError(f"The edge bands must be {EDGE_BAND_PX_MIN} to {EDGE_BAND_PX_MAX} px wide")
    crop_asked = normalize_crop(crop)
    crop = balance_crop(crop_asked)

    geometries = load_geometries(progress=progress)
    missing = [t for t in tags if t not in geometries]
    if missing:
        raise RenderError(f"No country found with {TAG_FIELD} in {missing}")
    if center_tag and (center_tag not in geometries or center_tag in WATER_TAGS):
        raise RenderError(f"Can't centre on '{center_tag}': not a country")
    water = [t for t in tags if t in WATER_TAGS]
    if water:
        raise RenderError(f"{water} is a water body in GADM (e.g. the Caspian Sea), not a country")

    excluded = {t.upper() for t in (excluded_tags or ())}
    # Removed countries are hidden ones that don't even leave land behind, so
    # every "not drawn / not framed" rule below applies to them as well; the
    # drawing loop then skips them outright. A removed country can't frame the
    # map either, so it leaves the selection here.
    removed = {t.upper() for t in (removed_tags or ())}
    excluded |= removed
    tags = [t for t in tags if t not in removed]
    if world:
        margin_percent = 0.0  # the world map is framed edge to edge
        if not include_antarctica:
            excluded.add("ATA")

    # With ignore_overseas, framing (bbox + seam choice) only considers each
    # focus country's mainland rings; overseas parts are still drawn if they
    # happen to fall inside the resulting viewport.
    ring_masks = None
    overseas = {}
    if ignore_overseas:
        ring_masks = {}
        for t in tags:
            info = mainland_ring_mask(geometries, t)
            ring_masks[t] = info["ring_mask"]
            if info["overseas_regions"]:
                overseas[t] = info["overseas_regions"]

    # Excluded GADM subdivisions, grouped by country (an id starts with its
    # country's GID_0: "USA.12_1" level 1, "GBR.3.27_1" level 2 -- the two
    # may be mixed, each id says which level it is).
    _known_memo = {}

    def known_ids(t, lv):
        if (t, lv) not in _known_memo:
            _known_memo[(t, lv)] = {sd["id"] for sd in list_subdivisions(t, lv)}
        return _known_memo[(t, lv)]

    sub_excluded = {}
    for gid in excluded_subdivisions or ():
        gid = str(gid).strip()
        if gid:
            sub_excluded.setdefault(gid.split(".")[0].upper(), set()).add(gid)
    for t, ex in sub_excluded.items():
        if t not in geometries:
            raise RenderError(f"Unknown country for subdivision(s) {sorted(ex)}")
        unknown = sorted(g for g in ex if g not in known_ids(t, gid_level(g)))
        if unknown:
            raise RenderError(f"Unknown subdivision(s) {unknown}")
        if t in tags:
            known = known_ids(t, _subdiv_level(ex))
            if known and not [g for g in known if not _covers(g, ex)]:
                raise RenderError(f"Every subdivision of {t} is excluded: keep at least one (or unselect the country)")
    sub_focus = {t: ex for t, ex in sub_excluded.items() if t in tags}
    lon_override = {t: _subdivision_lons(t, ex) for t, ex in sub_focus.items()}
    # Included subdivisions of countries that aren't selected: extra framing.
    sub_included = {}
    for gid in included_subdivisions:
        t = gid.split(".")[0].upper()
        if t not in geometries:
            raise RenderError(f"Unknown country for subdivision '{gid}'")
        if gid not in known_ids(t, gid_level(gid)):
            raise RenderError(f"Unknown subdivision '{gid}'")
        if t not in tags:  # a selected country already frames all of its subdivisions
            sub_included.setdefault(t, set()).add(gid)
    for t, only in sub_included.items():
        lon_override["+" + t] = _subdivision_lons(t, (), only)

    if center_tag:
        central_meridian = country_meridian(geometries, center_tag)
    elif world or (not tags and not sub_included):
        central_meridian = WORLD_CENTRAL_MERIDIAN
    elif projection_id in PSEUDO_CYLINDRICAL:
        central_meridian = _centered_meridian_for(geometries, tags, ring_masks, lon_override)
    else:
        central_meridian = _central_meridian_for(geometries, tags, ring_masks, lon_override)
    func = PROJECTIONS[projection_id]["func"]
    world_half_w = float(func(np.array([180.0]), np.array([0.0]))[0][0])  # widest point of the world
    # Drawn in the fixed colours (assign_colors), whatever the colour
    # configuration: the configured ones are applied afterwards (recolor_render).
    colors = assign_colors(geometries)
    # Merged autonomous territories are drawn in, and counted as, their
    # country; hiding the country hides them too.
    parent_of = {c: p for c, p in palettes.AUTONOMOUS_PARENT.items()
                 if merge_autonomous and c in geometries and p in geometries}
    owner = lambda t: parent_of.get(t, t)
    removed |= {c for c, p in parent_of.items() if p in removed}
    excluded |= {c for c, p in parent_of.items() if p in excluded}

    # The viewport is framed around the union of every selected country's
    # bbox, so picking several countries widens (or repositions) the view to
    # fit all of them rather than just the first one.
    focus_projs = {t: get_projected_country(geometries, t, projection_id, central_meridian) for t in tags}
    # The frame is built from *groups* of pieces (one group per country or
    # per subdivision selection), each piece carrying its projected area, so
    # the pieces the land/sea steps would erase can be left out below.
    frame_groups = []
    if world:
        # Every country that will be drawn (hidden ones excluded, so hiding
        # Antarctica tightens the frame), clipped to the world outline below.
        for t in geometries:
            if t not in WATER_TAGS and t not in excluded:
                frame_groups.append(_ring_frame_parts(
                    get_projected_country(geometries, t, projection_id, central_meridian)))
    for t, only in ([] if world else sub_included.items()):
        g = _subdivision_frame_parts(t, (), projection_id, central_meridian, only)
        if g:
            frame_groups.append(g)
    for t, p in ([] if world else focus_projs.items()):
        # HISTORICAL (shelved, see historical.py):
        # if hmap is not None and t == hmap["focus_tag"]:
        #     # Framed on its historical extent (1939 Poland), not today's.
        #     frame_bboxes.append(_estimate_projected_bbox(hmap["focus_lonlat_bbox"], func, central_meridian))
        #     continue
        if t in sub_focus:
            # Subdivision choice overrides "ignore overseas" for this country.
            g = _subdivision_frame_parts(t, sub_focus[t], projection_id, central_meridian)
            if g:
                frame_groups.append(g)
                continue
        g = _ring_frame_parts(p, ring_masks[t] if ring_masks is not None else None)
        frame_groups.append(g or _ring_frame_parts(p))
    frame_groups = [g for g in frame_groups if g]
    if not frame_groups:
        raise RenderError("Nothing to frame: every country is hidden or removed")

    # Land the cleanup erases must not set the frame or the centring (user).
    # The catch is circular -- which pieces survive depends on the scale, and
    # the scale on the frame -- so this iterates: the first pass frames
    # frame on everything; the second pass (below) reframes on what survived.
    biggest = [max(range(len(g)), key=lambda i: g[i][0]) for g in frame_groups]

    def _frame_for(drop):
        """(fxmin, fymin, fxmax, fymax) + choose_output_size, for one choice
        of dropped pieces."""
        bbs = [b for b in (_group_bbox(g, drop[i]) for i, g in enumerate(frame_groups)) if b is not None]
        # A focus country cut by a user-chosen seam extends past the world edge
        # once unwrapped; frame only what's inside the world outline.
        x0 = max(min(b[0] for b in bbs), -world_half_w)
        y0 = min(b[1] for b in bbs)
        x1 = min(max(b[2] for b in bbs), world_half_w)
        y1 = max(b[3] for b in bbs)
        if center_tag and not world:
            # Centre on the chosen country (x = 0 at the central meridian).
            half = min(max(-x0, x1), world_half_w)
            x0, x1 = -half, half
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            raise RenderError(f"Degenerate geometry for {tags}")
        return (x0, y0, x1, y1) + choose_output_size(x1 - x0, y1 - y0, margin_percent, output_size,
                                                     BIG_MAP_FACTOR if double_size else 1)

    frame_drop = _frame_drop if _frame_drop is not None else {i: set() for i in range(len(frame_groups))}
    fxmin, fymin, fxmax, fymax, out_w, out_h, scale, shown_w, shown_h = _frame_for(frame_drop)
    bbox_w, bbox_h = fxmax - fxmin, fymax - fymin
    frame_report = {"pass": 2 if _frame_drop is not None else 1,
                    "dropped_pieces": sum(len(d) for d in frame_drop.values()),
                    "pieces": sum(len(g) for g in frame_groups)}

    cx = (fxmin + fxmax) / 2.0
    cy = (fymin + fymax) / 2.0
    if world:
        # Edge to edge: the full 360 degrees fill the width exactly (the
        # seam -- by default the Bering Strait -- on both borders), centred
        # vertically on the land. "Contain" fitting instead left bands of
        # sea beyond the seam whenever the height was the tighter axis.
        scale = out_w / (2.0 * world_half_w)
        shown_w, shown_h = 2.0 * world_half_w, out_h / scale
        cx = 0.0
    win_xmin, win_xmax = cx - shown_w / 2.0, cx + shown_w / 2.0
    win_ymin, win_ymax = cy - shown_h / 2.0, cy + shown_h / 2.0

    canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
    canvas[:, :] = BACKGROUND_COLOR

    visible = []
    view_proj = {}  # tag -> projected country, for every non-hidden country with rings in view
    # Strait fixing needs every country in view as land -- including hidden
    # ones (drawn as background, but not water) -- plus their coastlines.
    hidden_land = np.zeros((out_h, out_w), dtype=bool) if need_hidden else None
    coast_seg, coast_lonlat = [], []
    # Island separation: one label per land polygon (holes map to the
    # polygon around them), plus land-border pixels and joined pairs.
    poly_labels = np.zeros((out_h, out_w), dtype=np.int32) if separate else None
    label_base = {}
    border_seg, neighbour_rings = [], []
    next_label = 1
    for other_tag, data in geometries.items():
        if other_tag in WATER_TAGS or other_tag in removed:
            continue
        if other_tag in excluded and not need_hidden:
            continue

        if other_tag in focus_projs:
            proj = focus_projs[other_tag]
        elif not data["wide_span"] and not _bbox_spans_seam(data["lonlat_bbox"], central_meridian):
            # Cheap pre-filter: estimate this country's projected bbox from
            # just its raw lon/lat bbox (a handful of point evaluations)
            # before paying to project every one of its vertices.
            ex0, ey0, ex1, ey1 = _estimate_projected_bbox(data["lonlat_bbox"], func, central_meridian)
            pad_x = (ex1 - ex0) * 0.02 + 1.0
            pad_y = (ey1 - ey0) * 0.02 + 1.0
            if ex1 + pad_x < win_xmin or ex0 - pad_x > win_xmax or \
               ey1 + pad_y < win_ymin or ey0 - pad_y > win_ymax:
                continue
            proj = get_projected_country(geometries, other_tag, projection_id, central_meridian)
        else:
            # Wide-span country (crosses the antimeridian, or a pole): the
            # cheap estimate isn't reliable, so always project it exactly.
            proj = get_projected_country(geometries, other_tag, projection_id, central_meridian)

        bxmin, bymin, bxmax, bymax = proj["bbox"]
        if bxmax < win_xmin or bxmin > win_xmax or bymax < win_ymin or bymin > win_ymax:
            continue

        # Per-ring cull: a wide country's overall bbox often overlaps the
        # view while none of its rings do (Russia/USA around a Portugal view
        # once projected with unwrapped longitudes); skipping those rings
        # saves the pixel transform + fill, and keeps them out of the legend.
        # Culling a hole is safe: it lies inside its exterior's bbox, and a
        # hole outside the view can't affect any pixel inside it.
        rb = proj["ring_bboxes"]
        in_view = (rb[:, 2] >= win_xmin) & (rb[:, 0] <= win_xmax) & (rb[:, 3] >= win_ymin) & (rb[:, 1] <= win_ymax)
        rings_px = []
        topo = ring_topology(geometries, other_tag) if (fix or separate) else None
        coast_masks = topo["coastal"] if topo else None
        ring_labels = []
        paint_labels = separate and other_tag not in excluded
        if paint_labels:
            label_base[other_tag] = next_label
            next_label += len(data["rings"])
        wb_ = proj["wrap_bboxes"]
        wrap_in_view = (wb_[:, 2] >= win_xmin) & (wb_[:, 0] <= win_xmax) & (wb_[:, 3] >= win_ymin) & (wb_[:, 1] <= win_ymax)
        ring_items = [(k, proj["rings"][k][1]) for k in np.flatnonzero(in_view)]
        ring_items += [(proj["wraps"][j][0], proj["wraps"][j][2]) for j in np.flatnonzero(wrap_in_view)]
        ring_ks = []
        for k, xy in ring_items:
            if len(xy) < 3:
                continue
            px = (xy[:, 0] - win_xmin) * scale
            py = out_h - (xy[:, 1] - win_ymin) * scale
            rings_px.append(np.column_stack([px, py]))
            ring_ks.append(k)
            if paint_labels:
                ring_labels.append(label_base[other_tag] + topo["hole_parent"].get(k, k))
                b_m = ~coast_masks[k]
                if b_m.any():
                    cur, nxt, run_b = _simplified_runs(px, py, b_m, 2)
                    cur, nxt = cur[run_b], nxt[run_b]  # the land-border chords
                    border_seg.append(np.column_stack([px[cur], py[cur], px[nxt], py[nxt]]))
            if fix:
                m = coast_masks[k]
                if m.any():
                    cur, nxt, run_m = _simplified_runs(px, py, m, land_sea["subpixel_rows"])
                    cur, nxt = cur[run_m], nxt[run_m]
                    coast_seg.append(np.column_stack([px[cur], py[cur], px[nxt], py[nxt]]))
                    raw = data["rings"][k][1]
                    coast_lonlat.append((raw[cur] + raw[nxt]) / 2.0)

        if not rings_px:
            continue
        if other_tag in sub_excluded:
            # Parts in excluded subdivisions (Hawaii) are left out of the
            # map entirely -- sea, on every tab (a hidden country instead
            # stays land, e.g. white on the river map).
            keep_r = subdivision_ring_mask(geometries, other_tag, sub_excluded[other_tag])
            hide_i = [i for i, k in enumerate(ring_ks) if not keep_r[k]]
            if hide_i:
                keep_i = [i for i in range(len(ring_ks)) if keep_r[ring_ks[i]]]
                rings_px = [rings_px[i] for i in keep_i]
                if paint_labels:
                    ring_labels = [ring_labels[i] for i in keep_i]
                if not rings_px:
                    continue

        # Even-odd fill: exterior + hole rings passed together punch the
        # holes (enclaves like Lesotho, San Marino) in one call.
        if other_tag not in excluded and other_tag not in parent_of:
            view_proj[other_tag] = proj
        if other_tag in excluded:
            if need_hidden:
                _scanline_fill(hidden_land, rings_px, True)
        elif _scanline_fill(canvas, rings_px, colors[owner(other_tag)],
                            labels=poly_labels, ring_labels=ring_labels if paint_labels else None):
            if owner(other_tag) not in visible:
                visible.append(owner(other_tag))
            if paint_labels:
                neighbour_rings.append((other_tag, topo))

    # HISTORICAL (shelved, see historical.py):
    # historical_report = None
    # if hmap is not None:
    #     historical_report = _historical.apply(
    #         hmap, canvas, hidden_land, colors, excluded,
    #         {"projection": projection_id, "center": central_meridian, "win_xmin": win_xmin,
    #          "win_ymin": win_ymin, "scale": scale}, BACKGROUND_COLOR)
    #     cand = list(dict.fromkeys(list(visible) + [hmap["focus_tag"]]))
    #     counts = np.bincount(_country_index(canvas, [colors[t] for t in cand]).ravel(), minlength=len(cand) + 1)
    #     visible = [t for i, t in enumerate(cand) if counts[i + 1] > 0]
    domain = _domain_mask(func, win_xmin, win_ymin, scale, out_w, out_h)
    if not domain.all():
        canvas[~domain] = BACKGROUND_COLOR
        if hidden_land is not None:
            hidden_land &= domain
        if poly_labels is not None:
            poly_labels[~domain] = 0
    # Edge bands: all sea before the land/sea steps, so they treat them as
    # sea (coasts, islands, straits); enforced again at the end, then the
    # unowned land strip is painted in.
    band, edge_land = _edge_band_masks(out_h, out_w, edge_bands, edge_band_px, world)
    if band is not None:
        canvas[band] = BACKGROUND_COLOR
        if hidden_land is not None:
            hidden_land &= ~band
        if poly_labels is not None:
            poly_labels[band] = 0

    # Sea level rise before every land/sea step, so they clean up the new
    # coastline (user: "the cleanup must happen afterwards").
    sea_level_report = {"enabled": False}
    drowned = drowned_depth = None
    if sea_level_m > 0:
        sea_level_report, drowned, drowned_depth = drown_land(
            canvas, hidden_land, poly_labels,
            {"projection": projection_id, "center": central_meridian, "win_xmin": win_xmin,
             "win_ymin": win_ymin, "scale": scale, "width": out_w, "height": out_h},
            sea_level_m)
        progress(f"  sea level +{sea_level_m} m: {sea_level_report['drowned_px']:,} px drowned "
                 f"({sea_level_report['drowned_share'] * 100:.1f}% of the land) "
                 f"in {sea_level_report['seconds']:.1f}s")

    # Lakes first: every later step then sees them as water (islands in a
    # lake are islands, specks/diagonals/sieve respect lake shores), except
    # strait fixing, which must not join lakes to the sea (see fix_straits).
    lakes_report = {"enabled": land_sea["lakes"]}
    lake_mask = sea_before = lakes_drawn = None
    if land_sea["lakes"]:
        lakes_report, lake_mask, lakes_drawn = cut_lakes(canvas, projection_id, central_meridian, win_xmin,
                                                         win_ymin, scale, land_sea["lake_min_px"])
        if lake_mask is not None:
            if poly_labels is not None:
                poly_labels[lake_mask] = 0
            sea_before = (canvas == np.asarray(BACKGROUND_COLOR, dtype=canvas.dtype)).all(axis=2) & ~lake_mask
            if hidden_land is not None:
                sea_before &= ~hidden_land

    land_sea_report = {"enabled": False}
    debug = None
    separated = None
    separation_report = {"enabled": separate}
    if separate:
        keys = []
        for t, topo in neighbour_rings:
            hp = topo["hole_parent"]
            for ring, o_tag, o_ring in topo["neighbours"]:
                if o_tag in label_base:
                    o_hp = ring_topology(geometries, o_tag)["hole_parent"] if o_tag != t else hp
                    a_ = label_base[t] + hp.get(ring, ring)
                    b_ = label_base[o_tag] + o_hp.get(o_ring, o_ring)
                    keys.extend((a_ * (1 << 32) + b_, b_ * (1 << 32) + a_))
        neighbour_keys = np.unique(np.array(keys, dtype=np.int64))
        border_px = (_segment_pixels(np.concatenate(border_seg).T, out_h, out_w) if border_seg
                     else np.zeros((out_h, out_w), dtype=bool))
        mainland_labels = np.array([base + largest_ring(geometries, t) for t, base in label_base.items()],
                                   dtype=np.int64)
        separated, n_pairs = separate_islands(canvas, poly_labels, neighbour_keys, border_px,
                                              land_sea["island_gap_px"], land_sea["min_island_px"],
                                              mainland_labels)
        separation_report.update({"separated_px": int(separated.sum()), "separated_pairs": n_pairs})
    to_px = _ToPx(projection_id, central_meridian, win_xmin, win_ymin, scale, out_h)

    if fix:
        channels_px = [(to_px(*c["from"]), to_px(*c["to"])) for c in land_sea["channels"]]
        blocked_px = [to_px(b["lon"], b["lat"]) for b in land_sea["blocked"]]
        if coast_seg:
            seg = np.concatenate(coast_seg)
            land_sea_report, debug = fix_straits(canvas, hidden_land, seg.T, np.concatenate(coast_lonlat),
                                                 land_sea, channels_px, blocked_px, lakes=lake_mask)
        else:
            land_sea_report = {"enabled": True, "straits": [], "carved_px": 0, "widened_px": 0, "channel_px": 0}
        land_sea_report["settings"] = {k: land_sea[k] for k in _LAND_SEA_LIMITS}
        land_sea_report["blocked_px"] = [[round(x, 1), round(y, 1)] for x, y in blocked_px]
    islands_report = {"suppress": land_sea["suppress_islands"], "grow": land_sea["grow_islands"]}
    suppressed = grown = None
    if island_ops:
        keep = max(land_sea["island_gap_px"] if separate else 1, land_sea["min_strait_width_px"] if fix else 1)
        rep_, suppressed, grown = adjust_islands(canvas, hidden_land, dict(land_sea, keep_px=keep))
        islands_report.update(rep_)
    # Microstates: whole projected area under microstate_px (not just what's
    # visible -- a sliver of Russia at the frame edge isn't a microstate).
    view_tags = list(view_proj) + [t for t in visible if t not in view_proj]
    view_colors = [colors[t] for t in view_tags]
    micro_report = {"enabled": micro_on}
    micro_mask = None
    protected = set()
    if micro_on:
        target = land_sea["microstate_px"]
        micro = []
        for i, t in enumerate(view_tags):
            if t not in view_proj:
                continue
            if _projected_area(view_proj[t]) * scale * scale >= target:
                continue
            rings = view_proj[t]["rings"]
            areas = np.array([_signed_area(xy) for _h, xy in rings]) * scale * scale
            holes = np.array([h_ for h_, _xy in rings])
            ext = [k for k, h_ in enumerate(holes) if not h_]
            wts = np.abs(areas[ext]) + 1e-12
            cxs = [rings[k][1][:, 0].mean() for k in ext]
            cys = [rings[k][1][:, 1].mean() for k in ext]
            mx, my = np.average(cxs, weights=wts), np.average(cys, weights=wts)
            px_, py_ = (mx - win_xmin) * scale, out_h - (my - win_ymin) * scale
            if 0 <= px_ < out_w and 0 <= py_ < out_h:
                micro.append((colors[t], px_, py_))
                protected.add(i)
                micro_report.setdefault("countries", []).append(t)
        rep_, micro_mask, _taken = grow_microstates(canvas, hidden_land, micro, target)
        micro_report.update(rep_)
    specks_report = {"enabled": specks}
    speck_mask = None
    if specks:
        rep_, speck_mask = fill_sea_specks(canvas, hidden_land, land_sea["sea_speck_below_px"])
        specks_report.update(rep_)
    # Last, so it also fixes diagonal-only pixels left by separation,
    # carving, widening, suppression, growth and speck filling.
    diagonal_report = {"enabled": connect}
    bridged = None
    if connect:
        rep_, bridged = connect_diagonals(canvas, hidden_land, view_colors)
        diagonal_report.update(rep_)
    # Region sieve after the diagonal step: pieces that step joins must not
    # be merged away first (Croatian staircase islands are 1-3 px pieces).
    sieve_report = {"enabled": sieve}
    sieved = None
    if sieve:
        rep_, sieved = sieve_regions(canvas, hidden_land, view_colors, land_sea["min_region_px"], protected)
        sieve_report.update(rep_)
    if band is not None:
        # No step may put land back into the band (microstates, diagonal
        # bridges, island growth, ...).
        canvas[band] = BACKGROUND_COLOR
        if hidden_land is not None:
            hidden_land &= ~band
        if lake_mask is not None:
            lake_mask &= ~band
    if (micro_on or sieve or island_ops or specks or connect or band is not None
            or drowned is not None):  # HISTORICAL: ... or hmap is not None
        # Steps above can add (microstates) or remove countries: recount.
        counts = np.bincount(_country_index(canvas, view_colors).ravel(), minlength=len(view_tags) + 1)
        visible = [t for i, t in enumerate(view_tags) if counts[i + 1] > 0]
    if separated is not None:
        debug = dict(debug or {}, separated=separated)
    if island_ops:
        debug = dict(debug or {}, suppressed=suppressed, grown=grown)
    land_sea_report["islands"] = islands_report
    land_sea_report["diagonals"] = diagonal_report
    land_sea_report["sea_specks"] = specks_report
    land_sea_report["microstates"] = micro_report
    land_sea_report["sieve"] = sieve_report
    if micro_mask is not None:
        debug = dict(debug or {}, micro=micro_mask)
    if sieved is not None:
        debug = dict(debug or {}, sieved=sieved)
    if speck_mask is not None:
        debug = dict(debug or {}, speck_filled=speck_mask)
    if bridged is not None:
        debug = dict(debug or {}, bridged=bridged)
    lake_final = None
    if lake_mask is not None:
        # Lakes get their own colour in the output, including water the
        # steps above created inside a lake (an erased islet, ...).
        water_now = (canvas == np.asarray(BACKGROUND_COLOR, dtype=canvas.dtype)).all(axis=2)
        if hidden_land is not None:
            water_now &= ~hidden_land
        lake_final = _classify_lake_water(water_now, lake_mask, sea_before)
        # Lakes open to the sea become sea; no lake touches another water
        # body only by a corner.
        lake_final, lakes_report["merged_into_sea"] = _merge_sea_lakes(lake_final, water_now & ~lake_final)
        lakes_report["diagonal_px"] = _separate_diagonal_water(
            canvas, lake_final, water_now, hidden_land if hidden_land is not None else np.zeros_like(water_now))
        lake_final = _drop_small_lakes(canvas, lake_final, water_now & ~lake_final, land_sea["lake_min_px"])
        canvas[lake_final] = LAKE_COLOR
        lakes_report["lake_px"] = int(lake_final.sum())
        lakes_report["lakes"] = int(ndimage.label(lake_final, structure=_CONNECTIVITY_4)[1])
    if edge_land is not None:
        canvas[edge_land] = EDGE_LAND_COLOR
    if drowned is not None:
        # The cleanup runs on the new coastline and may hand a few of the
        # drowned pixels back to the land (a speck of sea filled in, an
        # islet grown, a diagonal bridged). Those are land again, so they
        # leave the drowned mask: their people are not lost (states.py) and
        # the bathymetry gets no depth over them (drowned_depth_m).
        back = drowned & ~(canvas == np.asarray(BACKGROUND_COLOR, dtype=canvas.dtype)).all(axis=2)
        if lake_final is not None:
            back &= ~lake_final
        if hidden_land is not None:
            back |= drowned & hidden_land
        n_back = int(back.sum())
        if n_back:
            drowned &= ~back
            drowned_depth[back] = 0
            sea_level_report["reclaimed_px"] = n_back
            sea_level_report["drowned_px"] -= n_back
        if not drowned.any():
            drowned = drowned_depth = None
    # Second framing pass: anything the land/sea steps rubbed out (or that
    # never painted a pixel) must not have counted for the frame or the
    # centring, so reframe on what is actually on the map and draw it again.
    # Only pieces with no land at all are dropped, so nothing visible is ever
    # framed out; one extra pass at most, and none at all when the frame does
    # not move (the measurement itself is a few bbox tests).
    if _frame_drop is None and not world:
        land_now = ~(canvas == np.asarray(BACKGROUND_COLOR, dtype=canvas.dtype)).all(axis=2)
        if hidden_land is not None:
            land_now |= hidden_land
        if lake_final is not None:
            land_now &= ~lake_final
        if edge_land is not None:
            land_now &= ~edge_land
        drop2 = _pieces_with_no_land(frame_groups, biggest, land_now, win_xmin, win_ymin, scale, out_h, out_w)
        again = _frame_for(drop2) if any(drop2.values()) else None
        # Worth drawing again only for a different output size or a real gain
        # in scale: dropping one sub-pixel islet often moves the frame by a
        # hair, which would cost a second render for nothing.
        if again is not None and (again[4:6] != (out_w, out_h) or again[6] > scale * (1.0 + FRAME_MIN_GAIN)):
            return render_map_detailed(
                tags, projection_id, margin_percent, excluded_tags, output_size, ignore_overseas, land_sea,
                progress=progress, color_config=color_config, merge_autonomous=merge_autonomous,
                center_tag=center_tag, world=world, include_antarctica=include_antarctica,
                edge_bands=edge_bands, edge_band_px=edge_band_px,
                excluded_subdivisions=excluded_subdivisions, included_subdivisions=included_subdivisions,
                double_size=double_size, sea_level_m=sea_level_m, removed_tags=removed_tags,
                crop=crop_asked, _frame_drop=drop2)  # the ask, not the balanced crop: it is reported

    # Forced crop (user): cut whole 256 px bands off the chosen borders, as
    # the last thing the render does. Taken here, after every land/sea step
    # and after the framing has had its say, a cropped-away piece can never
    # move the frame or change the size the crop is subtracted from, so the
    # output is simply the map above minus those bands -- `scale` and the
    # position of everything left never change. The whole render is cut with
    # the canvas (every mask in `debug`, the viewport, the pixel transform,
    # the legend, the preview markers), so each tab sees the cropped map and
    # nothing downstream knows the difference.
    base_w, base_h = out_w, out_h
    if any(crop.values()):
        out_w -= crop["left"] + crop["right"]
        out_h -= crop["top"] + crop["bottom"]
        if out_w < CROP_MIN_SIDE_PX or out_h < CROP_MIN_SIDE_PX:
            raise RenderError(
                f"The crop leaves too little of this {base_w} x {base_h} map "
                f"({crop['left']} + {crop['right']} px across, {crop['top']} + {crop['bottom']} px down"
                + (", once rounded up to whole 256 px steps" if crop != crop_asked else "")
                + f"): keep at least {CROP_MIN_SIDE_PX} px each way")
        r0, c0 = crop["top"], crop["left"]
        r1, c1 = base_h - crop["bottom"], base_w - crop["right"]

        def _cut(a):
            """Anything shaped like the canvas follows it; everything else
            (counts, reports, lon/lat segments) is left as it is."""
            if isinstance(a, np.ndarray) and a.shape[:2] == (base_h, base_w):
                return np.ascontiguousarray(a[r0:r1, c0:c1])
            return a

        canvas = _cut(canvas)
        hidden_land, lake_final, lakes_drawn = _cut(hidden_land), _cut(lake_final), _cut(lakes_drawn)
        drowned, drowned_depth = _cut(drowned), _cut(drowned_depth)
        debug = {k: _cut(v) for k, v in (debug or {}).items()}
        win_xmin += crop["left"] / scale
        win_xmax -= crop["right"] / scale
        win_ymin += crop["bottom"] / scale
        win_ymax -= crop["top"] / scale
        shown_w, shown_h = out_w / scale, out_h / scale
        to_px = _ToPx(projection_id, central_meridian, win_xmin, win_ymin, scale, out_h)
        # The edge bands have to end up on the *new* borders: a crop can take
        # one clean off, and land against the left or right edge crashes the
        # game (see the HOI4 mod tab). They are painted over whatever is
        # there, as they always are; the one thing lost is that the strip they
        # now cover did not go through the land/sea steps as a border.
        band, edge_land = _edge_band_masks(out_h, out_w, edge_bands, edge_band_px, world)
        if band is not None:
            canvas[band] = BACKGROUND_COLOR
            if hidden_land is not None:
                hidden_land &= ~band
            if lake_final is not None:
                lake_final &= ~band
            if drowned is not None:
                drowned &= ~band
                drowned_depth[band] = 0
            if edge_land is not None:
                canvas[edge_land] = EDGE_LAND_COLOR
        # A country, or drowned land, left only in the cut-away bands is no
        # longer on the map.
        counts = np.bincount(_country_index(canvas, view_colors).ravel(), minlength=len(view_tags) + 1)
        visible = [t for i, t in enumerate(view_tags) if counts[i + 1] > 0]
        if drowned is not None:
            sea_level_report["drowned_px"] = int(drowned.sum())
            if not sea_level_report["drowned_px"]:
                drowned = drowned_depth = None
        # The Land/Sea preview's markers are pixel coordinates of the map the
        # steps ran on: move them with the crop, and drop the ones it cut off.
        land_sea_report["straits"] = [
            dict(st, x=st["x"] - c0, y=st["y"] - r0) for st in land_sea_report.get("straits", [])
            if c0 <= st["x"] < c1 and r0 <= st["y"] < r1]
        if "blocked_px" in land_sea_report:
            land_sea_report["blocked_px"] = [
                [round(x - c0, 1), round(y - r0, 1)] for x, y in land_sea_report["blocked_px"]
                if c0 <= x < c1 and r0 <= y < r1]

    # to_px: lon/lat -> pixel; view: the pixel frame, for overlays (render_rivers)
    # lakes_drawn: every HydroLAKES lake rasterized on land, incl. those too
    # small to keep (or dropped later) -- the Terrain tab turns the ones that
    # ended up as land into marsh.
    # edge_land: the unowned land strip of the edge bands (None when off).
    # drowned / drowned_depth: the land a sea level rise turned into sea and
    # the metres of water now over it (None when the rise is off) -- the
    # population there is lost (states.py) and the bathymetry has no
    # soundings for it (drowned_depth_m).
    debug = dict(debug or {}, to_px=to_px, lakes=lake_final, lakes_drawn=lakes_drawn, hidden_land=hidden_land,
                 edge_land=edge_land, drowned=drowned, drowned_depth=drowned_depth,
                 sea_level_m=sea_level_report.get("sea_level_m"),
                 country_rgb={t: tuple(int(v) for v in colors[owner(t)]) for t in geometries if t in colors},
                 view={"projection": projection_id, "center": central_meridian, "win_xmin": win_xmin,
                       "win_ymin": win_ymin, "scale": scale, "width": out_w, "height": out_h})
    land_sea_report["separation"] = separation_report
    land_sea_report["lakes"] = lakes_report
    land_sea_report["sea_level"] = sea_level_report

    image = Image.fromarray(canvas, mode="RGB")

    meta = {
        "tags": tags,
        "countries": [{"tag": t, "name": geometries[t]["name"]} for t in tags],
        "projection": projection_id,
        "projection_label": PROJECTIONS[projection_id]["label"],
        "width": out_w,
        "height": out_h,
        "size_mode": "auto" if output_size is None else "manual",
        "double_size": bool(double_size),
        "margin_percent": margin_percent,
        "ignore_overseas": bool(ignore_overseas),
        "overseas_regions": [
            {"tag": t, "name": geometries[t]["name"], "regions": names} for t, names in overseas.items()
        ],
        "central_meridian": central_meridian,
        "excluded_countries": sorted(excluded - removed),
        "removed_countries": sorted(removed),
        "crop": dict(crop),
        "crop_asked": dict(crop_asked),
        "uncropped_size": [base_w, base_h],
        "merge_autonomous": bool(merge_autonomous),
        "center_country": center_tag,
        "world": bool(world),
        "include_antarctica": bool(include_antarctica),
        "edge_bands": edge_bands,
        "edge_band_px": edge_band_px,
        "sea_level_m": sea_level_m,
        "framing": frame_report,
        "excluded_subdivisions": sorted(g for ex in sub_excluded.values() for g in ex),
        "included_subdivisions": sorted(g for inc in sub_included.values() for g in inc),
        "autonomous_territories": {c: p for c, p in palettes.AUTONOMOUS_PARENT.items() if c in geometries},
        "visible_countries": [
            {"tag": t, "name": geometries[t]["name"], "color": rgb_to_hex(colors[t])}
            for t in sorted(visible)
        ],
        "land_sea": land_sea_report,
        # "historical": historical_report,  # HISTORICAL (shelved)
    }
    if color_config is False:  # the render in assign_colors' fixed colours (the app recolours it)
        return image, meta, debug
    return recolor_render(image, meta, debug, color_config)[:3]


def recolor_pairs(debug, color_config=None):
    """{fixed colour: configured colour} for a render drawn in assign_colors'
    colours (render_map_detailed(color_config=False)), + {tag: configured
    colour} (debug["country_rgb"] of the recoloured render) and the
    effective colours. `color_config` None = the active configuration."""
    import palettes  # local import: palettes imports this module
    geometries = load_geometries()
    eff, _nudged = palettes.effective_colors(geometries, color_config)
    tag_of = {tuple(c): t for t, c in assign_colors(geometries).items()}
    country_rgb = {t: tuple(int(v) for v in eff[tag_of[tuple(c)]]) for t, c in debug["country_rgb"].items()}
    pairs = {tuple(c): country_rgb[t] for t, c in debug["country_rgb"].items()}
    return pairs, country_rgb, eff


def recolor_image(image, pairs):
    """`image` with every colour in `pairs` replaced by its value (one pass,
    so a colour that is both a source and a target can't be mapped twice)."""
    src = [(k, v) for k, v in pairs.items() if k != v]
    if not src:
        return image.copy()
    arr = np.asarray(image)
    key = (arr[..., 0].astype(np.uint32) << 16) | (arr[..., 1].astype(np.uint32) << 8) | arr[..., 2]
    lut = np.full(1 << 24, -1, dtype=np.int16)
    lut[[(r << 16) | (g << 8) | b for (r, g, b), _v in src]] = np.arange(len(src), dtype=np.int16)
    ix = lut[key]
    hit = ix >= 0
    out = arr.copy()
    out[hit] = np.array([v for _k, v in src], dtype=np.uint8)[ix[hit]]
    return Image.fromarray(out, mode="RGB")


def recolor_render(image, meta, debug, color_config=None):
    """Country colours as post-processing: a render made in the fixed colours
    (color_config=False) -> the colour configuration's (image, legend,
    debug["country_rgb"]), so a colour change never re-renders the map.
    Returns (image, meta, debug, pairs); `pairs` recolours other images
    made from that render (recolor_image)."""
    pairs, country_rgb, eff = recolor_pairs(debug, color_config)
    meta = dict(meta, visible_countries=[dict(c, color=rgb_to_hex(eff[c["tag"]]))
                                         for c in meta["visible_countries"]])
    return recolor_image(image, pairs), meta, dict(debug, country_rgb=country_rgb), pairs


def list_countries():
    geometries = load_geometries()
    return sorted(
        [{"tag": tag, "name": data["name"]} for tag, data in geometries.items() if tag not in WATER_TAGS],
        key=lambda d: d["name"],
    )


# Step 5b: land/sea -- strait fixing
# --------------------------------------------------------------------------
# Pixel-centre rasterization is exact, but a strait narrower than a pixel
# (the Bosphorus is ~0.7 km wide; a Europe-wide view is ~1-3 km/px) still
# vanishes, splitting one sea into disconnected pieces. fix_straits restores
# the true water connectivity:
#
#   1. Coast pixels: land pixels that a *coastline* edge (see
#      coastal_edge_masks -- never a land border) passes through.
#   2. For each coast pixel, the exact fraction of it that is water, sampled
#      on `subpixel_rows` horizontal lines per pixel (exact along each line,
#      from the coastline crossings), plus whether water crosses each of its
#      4 sides -- so water in two neighbouring pixels only links them when
#      it actually flows across their shared side (a strait), not when a
#      sub-pixel isthmus separates it.
#   3. A graph over coast pixels + small water specks, with every water body
#      of >= min_water_body_px as a source; one multi-source Dijkstra, then
#      a minimum spanning tree over the cheapest meeting points picks the
#      single best (most-water, shortest) channel between each pair of
#      bodies that are connected in the true geometry. Those channels are
#      carved to water.
#   4. Every water passage narrower than min_strait_width_px that links two
#      distinct "thick" water areas (carved or pre-existing, e.g. 1-px or
#      diagonal-only links) is widened to that width. Dead-end 1-px inlets
#      are left alone.
#
# User adjustments (Land/Sea tab, persisted in land_sea_settings.json):
# `blocked` points stop a false channel from being carved near them, and
# `channels` force a straight water channel between two lon/lat points
# (canals, or straits GADM itself closes).

LAND_SEA_SETTINGS_PATH = BASE_DIR / "land_sea_settings.json"
LAND_SEA_DEFAULTS = {
    "lakes": True,
    "lake_min_px": 8,
    "separate_islands": True,
    "island_gap_px": 1,
    "min_island_px": 1,
    "suppress_islands": True,
    "suppress_below_px": 8,
    "grow_islands": True,
    "grow_min_px": 8,
    "grow_max_px": 20,
    "grow_by_px": 1,
    "grow_min_distance_px": 10,
    "continent_min_px": 1000,
    "fill_sea_specks": True,
    "sea_speck_below_px": 8,
    "grow_microstates": True,
    "microstate_px": 12,
    "sieve_regions": True,
    "min_region_px": 8,
    "connect_diagonals": True,
    "fix_straits": True,
    "min_strait_width_px": 2,
    "min_water_body_px": 16,
    "max_strait_length_px": 64,
    "subpixel_rows": 8,
    "blocked": [],   # [{"lon": float, "lat": float, "label": str}]
    "channels": [],  # [{"from": [lon, lat], "to": [lon, lat], "label": str}]
}
_LAND_SEA_LIMITS = {
    "lake_min_px": (1, 100000),
    "island_gap_px": (1, 8),
    "min_island_px": (1, 100000),
    "suppress_below_px": (1, 100000),
    "grow_min_px": (1, 100000),
    "grow_max_px": (1, 100000),
    "grow_by_px": (1, 5),
    "grow_min_distance_px": (0, 500),
    "continent_min_px": (1, 10000000),
    "sea_speck_below_px": (1, 100000),
    "microstate_px": (1, 10000),
    "min_region_px": (1, 100000),
    "min_strait_width_px": (1, 8),
    "min_water_body_px": (1, 100000),
    "max_strait_length_px": (1, 1000),
    "subpixel_rows": (2, 32),
}
# Smallest gain in scale worth a second framing pass (see render_map_detailed).
FRAME_MIN_GAIN = 0.005
BLOCK_RADIUS_PX = 4
CHANNEL_SNAP_PX = 12  # manual channel ends extend to water this close


def normalize_land_sea(settings) -> dict:
    """Merge `settings` over the defaults and validate every field."""
    out = json.loads(json.dumps(LAND_SEA_DEFAULTS))
    settings = settings or {}
    for key in ("lakes", "fix_straits", "separate_islands", "suppress_islands", "grow_islands", "connect_diagonals",
                "fill_sea_specks", "grow_microstates", "sieve_regions"):
        if key in settings:
            out[key] = bool(settings[key])
    for key, (lo, hi) in _LAND_SEA_LIMITS.items():
        if key in settings:
            try:
                value = int(settings[key])
            except (TypeError, ValueError):
                raise RenderError(f"Land/Sea setting '{key}' must be an integer")
            if not lo <= value <= hi:
                raise RenderError(f"Land/Sea setting '{key}' must be between {lo} and {hi}")
            out[key] = value

    if out["grow_min_px"] >= out["grow_max_px"]:
        raise RenderError("Island growth: the minimum size must be below the maximum size")

    def lonlat(p, what):
        try:
            lon, lat = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError, KeyError):
            raise RenderError(f"Invalid {what} coordinates: {p!r}")
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise RenderError(f"{what} coordinates out of range: {p!r}")
        return [lon, lat]

    out["blocked"] = [
        {"lon": ll[0], "lat": ll[1], "label": str(b.get("label", ""))[:80]}
        for b in settings.get("blocked", [])
        for ll in [lonlat([b.get("lon"), b.get("lat")], "blocked point")]
    ]
    out["channels"] = [
        {"from": lonlat(c.get("from"), "channel start"), "to": lonlat(c.get("to"), "channel end"),
         "label": str(c.get("label", ""))[:80]}
        for c in settings.get("channels", [])
    ]
    return out


def load_land_sea_settings() -> dict:
    if LAND_SEA_SETTINGS_PATH.exists():
        with open(LAND_SEA_SETTINGS_PATH, encoding="utf-8") as f:
            return normalize_land_sea(json.load(f))
    return normalize_land_sea({})


def save_land_sea_settings(settings) -> dict:
    settings = normalize_land_sea(settings)
    tmp = LAND_SEA_SETTINGS_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    tmp.replace(LAND_SEA_SETTINGS_PATH)
    return settings


class _CoastCrossings:
    """Coastline crossings of every sub-row (k per pixel row), sorted, with a
    running "water length so far" per row, for O(log n) point queries."""

    def __init__(self, seg, h: int, w: int, k: int):
        x0, y0, x1, y1 = seg
        self.k, self.span = k, float(w + 4)
        r_lo = np.clip(np.ceil(np.minimum(y0, y1) * k - 0.5), 0, h * k).astype(np.int64)
        r_hi = np.clip(np.ceil(np.maximum(y0, y1) * k - 0.5), 0, h * k).astype(np.int64)
        counts = r_hi - r_lo
        e = np.flatnonzero(counts > 0)
        counts = counts[e]
        rows = np.repeat(r_lo[e], counts) + (np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts))
        e = np.repeat(e, counts)
        ys = (rows + 0.5) / k
        xs = np.clip(x0[e] + (ys - y0[e]) * (x1[e] - x0[e]) / (y1[e] - y0[e]), -1.0, w + 1.0)
        keys = rows * self.span + (xs + 2.0)
        order = np.argsort(keys, kind="stable")
        self.keys, self.xs, rows = keys[order], xs[order], rows[order]
        n = len(self.keys)
        row_start = np.searchsorted(self.keys, rows * self.span, side="left")
        local = np.arange(n) - row_start
        prev_x = np.where(local > 0, self.xs[np.maximum(np.arange(n) - 1, 0)], -1.0)
        seg_out = (self.xs - prev_x) * (local % 2 == 0)  # before an even count of crossings = water
        csum = np.cumsum(seg_out)
        base = np.where(row_start > 0, csum[np.maximum(row_start - 1, 0)], 0.0)
        self.G = csum - base

    def query(self, sub_rows, x):
        """(water length along the sub-row from x=-1 to x, is x in water)."""
        if len(self.keys) == 0:
            return x + 1.0, np.ones(len(x), dtype=bool)
        j = np.searchsorted(self.keys, sub_rows * self.span + (x + 2.0), side="right")
        m = j - np.searchsorted(self.keys, sub_rows * self.span, side="left")
        jj = np.maximum(j - 1, 0)
        in_water = m % 2 == 0
        F = np.where(m > 0, self.G[jj] + (x - self.xs[jj]) * in_water, x + 1.0)
        return F, in_water


def _widen_thin_passages(water, must_widen, width: int):
    """Pixels to turn into water so every passage narrower than `width`
    that links two distinct thick water areas (or is in `must_widen`)
    becomes `width` px wide in both axes."""
    if width <= 1:
        return np.zeros_like(water)
    struct = np.ones((width, width), dtype=bool)
    # Opening by a width x width square, via separable min/max filters
    # (several times faster than binary_opening; identical result -- the
    # dilation's origin compensates the even-size window's offset).
    # Edge padding: the image border doesn't count as land.
    padded = np.pad(water, width, mode="edge").view(np.uint8)
    eroded = ndimage.minimum_filter(padded, size=width, mode="constant", cval=0)
    opened = ndimage.maximum_filter(eroded, size=width, origin=-1 if width % 2 == 0 else 0,
                                    mode="constant", cval=0).view(bool)[width:-width, width:-width]
    thin = water & ~opened
    if not thin.any():
        return np.zeros_like(water)
    thick_lab, _ = ndimage.label(opened, structure=_CONNECTIVITY_4)
    thin_lab, n_thin = ndimage.label(thin, structure=np.ones((3, 3)))
    # Distinct thick areas touching each thin component (8-neighbourhood),
    # gathered at the (few) thin pixels only.
    tr, tc = np.nonzero(thin)
    h, w = water.shape
    pairs = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = tr + dr, tc + dc
            ok = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
            b = np.zeros(len(tr), dtype=thick_lab.dtype)
            b[ok] = thick_lab[nr[ok], nc[ok]]
            sel = b > 0
            pairs.append(np.stack([thin_lab[tr[sel], tc[sel]], b[sel]], axis=1))
    pairs = np.unique(np.concatenate(pairs), axis=0)
    links = np.bincount(pairs[:, 0], minlength=n_thin + 1)
    relevant = np.zeros_like(water)
    keep = (links[thin_lab[tr, tc]] >= 2) | must_widen[tr, tc]
    relevant[tr[keep], tc[keep]] = True
    grown = ndimage.maximum_filter(relevant.view(np.uint8), size=width, mode="constant", cval=0).view(bool)
    return grown & ~water


def _segment_pixels(seg, h: int, w: int) -> np.ndarray:
    """Bool mask of the pixels that pixel-space segments (x0, y0, x1, y1)
    pass through. Almost all GADM segments are sub-pixel at render scale, so
    their end points already cover every pixel they touch; only long ones
    are stepped at <= 0.5 px."""
    x0, y0, x1, y1 = seg
    px_x, px_y = [x0, x1], [y0, y1]
    long_seg = np.flatnonzero(np.hypot(x1 - x0, y1 - y0) > 0.5)
    if len(long_seg):
        lx0, ly0, lx1, ly1 = x0[long_seg], y0[long_seg], x1[long_seg], y1[long_seg]
        n_steps = np.ceil(np.hypot(lx1 - lx0, ly1 - ly0) / 0.5).astype(np.int64)
        idx = np.repeat(np.arange(len(long_seg)), n_steps)
        t = (np.arange(len(idx)) - np.repeat(np.cumsum(n_steps) - n_steps, n_steps)) / np.repeat(n_steps, n_steps)
        px_x.append(lx0[idx] + (lx1[idx] - lx0[idx]) * t)
        px_y.append(ly0[idx] + (ly1[idx] - ly0[idx]) * t)
    cx = np.floor(np.concatenate(px_x)).astype(np.int64)
    cy = np.floor(np.concatenate(px_y)).astype(np.int64)
    ok = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
    mask = np.zeros((h, w), dtype=bool)
    mask.ravel()[cy[ok] * w + cx[ok]] = True
    return mask


_largest_ring_memo = {}


def largest_ring(geometries: dict, tag: str) -> int:
    """Index of a country's main landmass: its largest exterior ring."""
    if tag not in _largest_ring_memo:
        rings = geometries[tag]["rings"]
        _largest_ring_memo[tag] = max((k for k, (h, _r) in enumerate(rings) if not h),
                                      key=lambda k: abs(_signed_area(rings[k][1])))
    return _largest_ring_memo[tag]


def separate_islands(canvas, labels, neighbour_keys, border_px, gap: int, min_island: int, mainland_labels):
    """Force a `gap`-px water gap wherever two land polygons that are
    separated by sea in the data (an island and the land next to it --
    Funen and Jutland across the 573 m Little Belt, say) touch in the
    raster. Carves in place; returns (carved mask, number of polygon pairs).

    labels: int32 image, one id per land polygon (0 = none), from the fill.
    neighbour_keys: sorted int64 keys a * 2**32 + b of polygon pairs joined
      by land (ring_topology "neighbours"); their contacts are left alone
      where they meet along a land border (`border_px`, the pixels crossed
      by non-coastline edges).
    The larger polygon (by pixel area) gives up the pixels, so small islands
    survive; pairs whose smaller polygon is under `min_island` px are left
    merged. Two countries' main landmasses (`mainland_labels`) are never cut
    apart -- only contacts involving an island are: GADM leaves real gaps
    along some border rivers (Estonia/Russia on the Narva), and a 1 px line
    down a river border isn't what this is for (sea straits between two
    mainlands are fix_straits' job)."""
    h, w = labels.shape
    size = 2 * gap + 1
    big_int = np.iinfo(np.int32).max
    mx = ndimage.maximum_filter(labels, size=size, mode="constant", cval=0)
    mn = ndimage.minimum_filter(np.where(labels > 0, labels, big_int), size=size, mode="constant", cval=big_int)
    cr, cc = np.nonzero((labels > 0) & ((mx != labels) | (mn != labels)))
    carved = np.zeros((h, w), dtype=bool)
    if not len(cr):
        return carved, 0
    pl = labels[cr, cc].astype(np.int64)
    area = np.bincount(labels.ravel())
    border_near = ndimage.maximum_filter(border_px.view(np.uint8), size=3).view(bool)
    pairs = []
    for dr in range(-gap, gap + 1):
        for dc in range(-gap, gap + 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = cr + dr, cc + dc
            ok = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
            ql = np.zeros(len(cr), dtype=np.int64)
            ql[ok] = labels[nr[ok], nc[ok]]
            sel = (ql > 0) & (ql != pl)
            if not sel.any():
                continue
            keys = pl[sel] * (1 << 32) + ql[sel]
            pos = np.clip(np.searchsorted(neighbour_keys, keys), 0, max(len(neighbour_keys) - 1, 0))
            joined = (neighbour_keys[pos] == keys) if len(neighbour_keys) else np.zeros(len(keys), dtype=bool)
            i = np.flatnonzero(sel)
            at_border = border_near[cr[i], cc[i]] | border_near[nr[i], nc[i]]
            a_p, a_q = area[pl[i]], area[ql[i]]
            bigger = (a_p > a_q) | ((a_p == a_q) & (pl[i] > ql[i]))
            both_main = np.isin(pl[i], mainland_labels) & np.isin(ql[i], mainland_labels)
            cut = ~(joined & at_border) & ~both_main & bigger & (np.minimum(a_p, a_q) >= min_island)
            carved[cr[i[cut]], cc[i[cut]]] = True
            pairs.append(keys[cut])
    n_pairs = 0
    if pairs:
        k = np.concatenate(pairs)
        n_pairs = len(np.unique(np.minimum(k >> 32, k & 0xFFFFFFFF) * (1 << 32)
                                + np.maximum(k >> 32, k & 0xFFFFFFFF)))
    canvas[carved] = BACKGROUND_COLOR
    labels[carved] = 0
    return carved, n_pairs


def adjust_islands(canvas, hidden_land, settings):
    """Island suppression and growth, on the rendered image (in place).

    An island is a connected land area (8-connectivity, all countries
    together; hidden countries count as land so nothing next to them is
    mistaken for an island).
      * Suppression: islands under `suppress_below_px` px become sea.
      * Growth: islands of `grow_min_px` <= size < `grow_max_px` px lying at
        least `grow_min_distance_px` (Chebyshev) from every continental mass
        (an island of >= `continent_min_px` px) are dilated by `grow_by_px`,
        one pixel ring at a time, into sea only -- never closer than
        `keep_px` (the island gap / minimum strait width) to any other land,
        including islands grown before it -- in their own colour(s).
    Runs after strait fixing, so suppression also removes specks that
    carving/widening chipped off, and growth can't narrow a strait.
    Returns (report dict, suppressed mask, grown mask)."""
    h, w = canvas.shape[:2]
    bg = np.asarray(BACKGROUND_COLOR, dtype=canvas.dtype)
    land = ~(canvas == bg).all(axis=2) | hidden_land
    eight = np.ones((3, 3), dtype=bool)
    report = {}
    suppressed = np.zeros((h, w), dtype=bool)
    grown = np.zeros((h, w), dtype=bool)

    lab, n = ndimage.label(land, structure=eight)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    sizes[0] = 0
    if settings["suppress_islands"]:
        small = sizes < settings["suppress_below_px"]
        small[0] = False
        suppressed = small[lab] & ~hidden_land
        canvas[suppressed] = BACKGROUND_COLOR
        land &= ~suppressed
        lab[suppressed] = 0
        report.update(suppressed_islands=int(np.count_nonzero(small & (sizes > 0))),
                      suppressed_px=int(suppressed.sum()))

    if settings["grow_islands"]:
        # Continental masses, and an integral image of them so "is any
        # continent pixel within d px of this island's bbox" is O(1).
        big = sizes >= settings["continent_min_px"]
        big[0] = False
        sat = np.zeros((h + 1, w + 1), dtype=np.int32)
        sat[1:, 1:] = big[lab].cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
        d = settings["grow_min_distance_px"]
        keep = settings["keep_px"]
        n_grown = 0
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None or not settings["grow_min_px"] <= sizes[i] < settings["grow_max_px"]:
                continue
            r0, r1 = max(0, sl[0].start - d), min(h, sl[0].stop + d)
            c0, c1 = max(0, sl[1].start - d), min(w, sl[1].stop + d)
            if sat[r1, c1] - sat[r0, c1] - sat[r1, c0] + sat[r0, c0] > 0:
                continue  # a continental mass is too close
            # Grow in a small window; islands are processed one at a time,
            # so earlier growth counts as "other land" and keeps its gap too.
            m = settings["grow_by_px"] + keep + 2
            wr0, wr1 = max(0, sl[0].start - m), min(h, sl[0].stop + m)
            wc0, wc1 = max(0, sl[1].start - m), min(w, sl[1].stop + m)
            own = lab[wr0:wr1, wc0:wc1] == i
            land_w = land[wr0:wr1, wc0:wc1]
            other = land_w & ~own
            forbidden = ndimage.maximum_filter(other.view(np.uint8), size=2 * keep + 1).view(bool)
            forbidden |= hidden_land[wr0:wr1, wc0:wc1]
            px = canvas[wr0:wr1, wc0:wc1].astype(np.int32)
            col = np.where(own, (px[..., 0] << 16) | (px[..., 1] << 8) | px[..., 2], -1)
            added = np.zeros_like(own)
            for _ in range(settings["grow_by_px"]):
                ring = ndimage.binary_dilation(own, structure=eight) & ~land_w & ~forbidden & ~added
                if not ring.any():
                    break
                col[ring] = ndimage.maximum_filter(col, size=3)[ring]
                own |= ring
                added |= ring
            if added.any():
                n_grown += 1
                land[wr0:wr1, wc0:wc1] |= added
                lab[wr0:wr1, wc0:wc1][added] = i
                grown[wr0:wr1, wc0:wc1] |= added
                c = col[added]
                canvas[wr0:wr1, wc0:wc1][added] = np.stack([(c >> 16) & 255, (c >> 8) & 255, c & 255],
                                                           axis=1).astype(canvas.dtype)
        report.update(grown_islands=n_grown, grown_px=int(grown.sum()))
    return report, suppressed, grown


def _country_index(canvas, country_colors) -> np.ndarray:
    """Per pixel: 1 + index into `country_colors`, 0 elsewhere -- through a
    2^24 colour lookup table (32 MB, ~10x faster than searchsorted)."""
    enc = (canvas[..., 0].astype(np.int32) << 16) | (canvas[..., 1].astype(np.int32) << 8) | canvas[..., 2]
    lut = np.zeros(1 << 24, dtype=np.uint16)
    for i, (r, g, b) in enumerate(country_colors, start=1):
        lut[(r << 16) | (g << 8) | b] = i
    return lut[enc].astype(np.int32)


def grow_microstates(canvas, hidden_land, micro, target: int):
    """Give each microstate (a country whose *whole* projected area is under
    `target` px -- Monaco, the Vatican, ...) a compact `target`-px blob
    around its centroid, so it's visible instead of 0-2 px. `micro` is a
    list of (color, cx, cy) in pixel coords. Pixels are taken nearest-first
    from the centroid, preferring land over sea (sea costs +3 px of
    distance) and never from another microstate or a hidden country.
    Returns (report dict, grown mask)."""
    h, w = canvas.shape[:2]
    bg = np.asarray(BACKGROUND_COLOR, dtype=canvas.dtype)
    taken = np.zeros((h, w), dtype=bool)
    grown = np.zeros((h, w), dtype=bool)
    radius = int(math.ceil(math.sqrt(target / math.pi))) + 4
    n = 0
    for color, cx, cy in micro:
        r0, r1 = max(0, int(cy) - radius), min(h, int(cy) + radius + 1)
        c0, c1 = max(0, int(cx) - radius), min(w, int(cx) + radius + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        win = canvas[r0:r1, c0:c1]
        rr, cc = np.mgrid[r0:r1, c0:c1]
        score = np.hypot(cc + 0.5 - cx, rr + 0.5 - cy)
        score = score + 3.0 * (win == bg).all(axis=2)
        own = (win == np.asarray(color, dtype=canvas.dtype)).all(axis=2)
        score[own] = -1.0
        score[taken[r0:r1, c0:c1] | hidden_land[r0:r1, c0:c1]] = np.inf
        pick = np.argsort(score, axis=None, kind="stable")[:target]
        pick = pick[np.isfinite(score.ravel()[pick])]
        pr, pc = np.unravel_index(pick, score.shape)
        new = ~own[pr, pc]
        win[pr[new], pc[new]] = color
        grown[pr[new] + r0, pc[new] + c0] = True
        taken[pr + r0, pc + c0] = True
        n += 1
    return {"grown_microstates": n, "grown_px": int(grown.sum())}, grown, taken


def sieve_regions(canvas, hidden_land, country_colors, min_px: int, protected):
    """Merge every same-colour land area (4-connected) under `min_px` into
    its most common land neighbour -- or into the sea if it only touches
    sea (an islet). Removes rogue specks such as 1-3 px shards along a
    border (Northern Cyprus) and tiny exclaves (Llivia, Spain's 3 px pocket
    inside France). Areas touching the image border and `protected`
    countries (grown microstates, by index into country_colors) are kept.
    Returns (report dict, changed mask)."""
    h, w = canvas.shape[:2]
    idx = _country_index(canvas, country_colors)
    water = (canvas == np.asarray(BACKGROUND_COLOR, dtype=canvas.dtype)).all(axis=2) & ~hidden_land
    changed = np.zeros((h, w), dtype=bool)
    merged = 0
    for k, sl in enumerate(ndimage.find_objects(idx), start=1):
        if sl is None or (k - 1) in protected:
            continue
        lab, n = ndimage.label(idx[sl] == k, structure=_CONNECTIVITY_4)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        small = np.flatnonzero(sizes[1:] < min_px) + 1
        if not len(small):
            continue
        r_off, c_off = sl[0].start, sl[1].start
        for i, piece in enumerate(ndimage.find_objects(np.where(np.isin(lab, small), lab, 0)), start=1):
            if piece is None:
                continue
            pr0, pr1 = piece[0].start + r_off, piece[0].stop + r_off
            pc0, pc1 = piece[1].start + c_off, piece[1].stop + c_off
            if pr0 == 0 or pc0 == 0 or pr1 == h or pc1 == w:
                continue  # touches the frame: may be part of something bigger
            wr0, wr1, wc0, wc1 = pr0 - 1, pr1 + 1, pc0 - 1, pc1 + 1
            comp = np.zeros((wr1 - wr0, wc1 - wc0), dtype=bool)
            comp[1:-1, 1:-1] = lab[piece] == i
            ring = ndimage.binary_dilation(comp, structure=_CONNECTIVITY_4) & ~comp
            nb = idx[wr0:wr1, wc0:wc1][ring]
            land_nb = nb[(nb > 0) & (nb != k)]
            if len(land_nb):
                j = int(np.bincount(land_nb).argmax())
                canvas[wr0:wr1, wc0:wc1][comp] = country_colors[j - 1]
                idx[wr0:wr1, wc0:wc1][comp] = j
            elif water[wr0:wr1, wc0:wc1][ring].any():
                canvas[wr0:wr1, wc0:wc1][comp] = BACKGROUND_COLOR
                idx[wr0:wr1, wc0:wc1][comp] = 0
            else:
                continue
            changed[wr0:wr1, wc0:wc1] |= comp
            merged += 1
    return {"merged_regions": merged, "merged_px": int(changed.sum())}, changed


def fill_sea_specks(canvas, hidden_land, below: int):
    """Fill isolated bits of sea -- water areas under `below` px that aren't
    4-connected to any other water -- with the most common land colour
    around each one, including those touching the image border (user: no
    stray pixels below 8 px, edge ones too; they used to be kept as possibly
    part of a bigger sea beyond the frame -- 1-4 px pockets in land at the
    edge). Hidden countries' pixels aren't sea. Returns (report dict,
    filled mask)."""
    h, w = canvas.shape[:2]
    bg = np.asarray(BACKGROUND_COLOR, dtype=canvas.dtype)
    water = (canvas == bg).all(axis=2) & ~hidden_land
    lab, n = ndimage.label(water, structure=_CONNECTIVITY_4)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    small = sizes < below
    small[0] = False
    filled = np.zeros((h, w), dtype=bool)
    n_filled = 0
    objects = ndimage.find_objects(np.where(small[lab], lab, 0))
    for i, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        r0, r1 = max(0, sl[0].start - 1), min(h, sl[0].stop + 1)
        c0, c1 = max(0, sl[1].start - 1), min(w, sl[1].stop + 1)
        comp = lab[r0:r1, c0:c1] == i
        ring = ndimage.binary_dilation(comp, structure=_CONNECTIVITY_4) & ~comp
        px = canvas[r0:r1, c0:c1]
        land_ring = ring & ~(px == bg).all(axis=2) & ~hidden_land[r0:r1, c0:c1]
        if not land_ring.any():
            continue
        cols = px[land_ring].astype(np.int32)
        codes, counts = np.unique((cols[:, 0] << 16) | (cols[:, 1] << 8) | cols[:, 2], return_counts=True)
        code = int(codes[np.argmax(counts)])
        px[comp] = ((code >> 16) & 255, (code >> 8) & 255, code & 255)
        filled[r0:r1, c0:c1] |= comp
        n_filled += 1
    return {"filled_specks": n_filled, "filled_px": int(filled.sum())}, filled


def _bridge_ok(water, r, c, h, w) -> bool:
    """May sea pixel (r, c) become land without hurting the water? Its water
    4-neighbours must stay 4-connected to each other within the 3x3 ring
    (so no local split), and every water pixel around it that sat in a 2x2
    all-water block must still do so (so no strait is narrowed below the
    2 px strait fixing guarantees)."""
    r0, r1, c0, c1 = max(0, r - 2), min(h, r + 3), max(0, c - 2), min(w, c + 3)
    win = water[r0:r1, c0:c1].copy()
    lr, lc = r - r0, c - c0

    def blocks(a):  # a 2x2 all-water block starts at each True of the result
        return a[:-1, :-1] & a[1:, :-1] & a[:-1, 1:] & a[1:, 1:]

    def covered(a):
        b = blocks(a)
        out = np.zeros_like(a)
        out[:-1, :-1] |= b
        out[1:, :-1] |= b
        out[:-1, 1:] |= b
        out[1:, 1:] |= b
        return out

    before = covered(win)
    win[lr, lc] = False
    after = covered(win)
    if (before & win & ~after)[max(0, lr - 1):lr + 2, max(0, lc - 1):lc + 2].any():
        return False
    ring = win[max(0, lr - 1):lr + 2, max(0, lc - 1):lc + 2]
    lab, n = ndimage.label(ring, structure=_CONNECTIVITY_4)
    cr, cc = lr - max(0, lr - 1), lc - max(0, lc - 1)
    touching = {lab[cr + dr, cc + dc] for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= cr + dr < ring.shape[0] and 0 <= cc + dc < ring.shape[1] and lab[cr + dr, cc + dc] > 0}
    return len(touching) <= 1


def _simple_for(mask, r, c) -> bool:
    """Can pixel (r, c) leave `mask` without splitting it locally? Its
    4-neighbours in the mask must stay 4-connected within the 3x3 ring."""
    h, w = mask.shape
    ring = mask[max(0, r - 1):r + 2, max(0, c - 1):c + 2].copy()
    cr, cc = r - max(0, r - 1), c - max(0, c - 1)
    ring[cr, cc] = False
    lab, _n = ndimage.label(ring, structure=_CONNECTIVITY_4)
    touching = {lab[cr + dr, cc + dc] for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= cr + dr < ring.shape[0] and 0 <= cc + dc < ring.shape[1] and lab[cr + dr, cc + dc] > 0}
    return len(touching) == 1  # 0 would mean it's that country's last pixel here


def connect_diagonals(canvas, hidden_land, country_colors, passes: int = 3):
    """Make every country's land 4-connected where it is only diagonally
    connected: for each pair of same-colour pixels touching corner to
    corner only (and belonging to different 4-connected pieces), one of the
    two side pixels between them is painted in that colour -- a sea pixel if
    _bridge_ok (no water split, no strait narrowed), else, on inland borders
    (a checkerboard pinch between two countries), a neighbouring country's
    pixel if that doesn't split the neighbour locally (_simple_for). One
    bridge per pair of pieces (union-find); a few passes, since taking a
    neighbour's pixel can leave that neighbour with a new diagonal pinch.
    Returns (report dict, bridge mask)."""
    h, w = canvas.shape[:2]
    bg = np.asarray(BACKGROUND_COLOR, dtype=canvas.dtype)
    water = (canvas == bg).all(axis=2) & ~hidden_land
    idx = _country_index(canvas, country_colors)
    bridges = np.zeros((h, w), dtype=bool)
    unresolved = 0
    # Only countries with a corner-only pinch somewhere need labelling.
    a, d_ = idx[:-1, :-1], idx[1:, 1:]
    b, c = idx[:-1, 1:], idx[1:, :-1]
    pinch = np.concatenate([a[(a > 0) & (a == d_) & (b != a) & (c != a)],
                            b[(b > 0) & (b == c) & (a != b) & (d_ != b)]])
    todo = set(np.unique(pinch).tolist())
    if not todo:
        return {"bridged_px": 0, "unresolved": 0}, bridges
    for _pass in range(passes):
        changed = set()
        unresolved = _connect_pass(canvas, water, idx, bridges, h, w, todo, changed)
        if not changed:
            break
        todo = changed
    return {"bridged_px": int(bridges.sum()), "unresolved": unresolved}, bridges


def _connect_pass(canvas, water, idx, bridges, h, w, todo, changed) -> int:
    unresolved = 0
    for k, sl in enumerate(ndimage.find_objects(idx), start=1):
        if sl is None or (todo is not None and k not in todo):
            continue
        r_off, c_off = sl[0].start, sl[1].start
        mask = idx[sl] == k
        lab, n = ndimage.label(mask, structure=_CONNECTIVITY_4)
        if n < 2:
            continue
        cand = []
        a, b = lab[:-1, :-1], lab[1:, 1:]      # "\" diagonal: bridge via (r, c+1) or (r+1, c)
        rr, cc = np.nonzero((a > 0) & (b > 0) & (a != b))
        cand.append((a[rr, cc], b[rr, cc], rr, cc + 1, rr + 1, cc))
        a, b = lab[:-1, 1:], lab[1:, :-1]      # "/" diagonal: bridge via (r, c) or (r+1, c+1)
        rr, cc = np.nonzero((a > 0) & (b > 0) & (a != b))
        cand.append((a[rr, cc], b[rr, cc], rr, cc, rr + 1, cc + 1))
        parent = list(range(n + 1))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        color = canvas[sl][mask][0].copy()
        for la, lb, r1, c1, r2, c2 in cand:
            for i in range(len(la)):
                x, y = find(int(la[i])), find(int(lb[i]))
                if x == y:
                    continue
                options = [(int(r1[i]) + r_off, int(c1[i]) + c_off), (int(r2[i]) + r_off, int(c2[i]) + c_off)]
                # Prefer the side pixel with more land around it (fills a notch).
                options.sort(key=lambda rc: -int((~water[max(0, rc[0] - 1):rc[0] + 2,
                                                         max(0, rc[1] - 1):rc[1] + 2]).sum()))
                done = False
                for br, bc in options:  # sea first
                    if water[br, bc] and _bridge_ok(water, br, bc, h, w):
                        water[br, bc] = False
                        done = True
                        break
                else:
                    for br, bc in options:  # then a neighbour's pixel, if it can spare it
                        j = idx[br, bc]
                        if j > 0 and j != k:
                            wr0, wr1 = max(0, br - 1), min(h, br + 2)
                            wc0, wc1 = max(0, bc - 1), min(w, bc + 2)
                            local = idx[wr0:wr1, wc0:wc1] == j
                            if _simple_for(local, br - wr0, bc - wc0):
                                done = True
                                break
                if done:
                    changed.add(k)
                    if idx[br, bc] > 0:
                        changed.add(int(idx[br, bc]))
                    canvas[br, bc] = color
                    idx[br, bc] = k
                    bridges[br, bc] = True
                    parent[x] = y
                else:
                    unresolved += 1
    return unresolved


def fix_straits(canvas, hidden_land, seg, seg_lonlat, settings, channels_px, blocked_px, lakes=None):
    """Carve and widen straits in `canvas` in place (see the section notes).

    hidden_land: bool mask of excluded countries' land (drawn as background
      but still land for water-connectivity purposes).
    seg: (x0, y0, x1, y1) pixel-space coastline segments in view.
    seg_lonlat: (n, 2) lon/lat of each segment's midpoint (for reporting).
    lakes: bool mask of lake pixels (cut_lakes). Treated like hidden land
      here -- neither a water body to connect nor pixels to carve or widen
      into -- so no channel ever joins a lake to the sea or another lake.
    Returns (report dict, debug masks dict)."""
    h, w = canvas.shape[:2]
    k = settings["subpixel_rows"]
    width = settings["min_strait_width_px"]
    min_body = settings["min_water_body_px"]
    max_len = settings["max_strait_length_px"]
    solid = hidden_land if lakes is None else hidden_land | lakes

    water = (canvas == np.asarray(BACKGROUND_COLOR, dtype=canvas.dtype)).all(axis=2) & ~solid

    forced = np.zeros((h, w), dtype=np.uint8)

    def snap_to_water(x, y):
        """Nearest water pixel within CHANNEL_SNAP_PX of (x, y), so a
        channel end typed slightly inland still reaches the sea."""
        c, r = int(round(x)), int(round(y))
        r0, r1 = max(0, r - CHANNEL_SNAP_PX), min(h, r + CHANNEL_SNAP_PX + 1)
        c0, c1 = max(0, c - CHANNEL_SNAP_PX), min(w, c + CHANNEL_SNAP_PX + 1)
        wr, wc = np.nonzero(water[r0:r1, c0:c1])
        if len(wr) == 0:
            return c, r
        i = int(np.argmin((wr + r0 - r) ** 2 + (wc + c0 - c) ** 2))
        return int(wc[i] + c0), int(wr[i] + r0)

    for (ax, ay), (bx, by) in channels_px:
        a, b = (int(round(ax)), int(round(ay))), (int(round(bx)), int(round(by)))
        for p, q in ((snap_to_water(ax, ay), a), (a, b), (b, snap_to_water(bx, by))):
            cv2.line(forced, p, q, 1, thickness=max(1, width), lineType=cv2.LINE_8)
    forced = forced.view(bool) & ~water
    water |= forced

    # 1. Coast pixels: land pixels crossed by a coastline edge.
    x0, y0, x1, y1 = seg
    coast = _segment_pixels(seg, h, w)
    candidate = coast & ~water & ~solid

    # 2. Sub-pixel water fraction of each candidate pixel.
    cross = _CoastCrossings(seg, h, w, k)
    pr, pc = np.nonzero(candidate)
    sub = (pr[:, None] * k + np.arange(k)).ravel()
    xs = np.repeat(pc, k).astype(np.float64)
    f0, _ = cross.query(sub, xs)
    f1, _ = cross.query(sub, xs + 1.0)
    frac = np.clip((f1 - f0).reshape(-1, k).mean(axis=1), 0.0, 1.0)
    wet = frac > 0.01
    pr, pc, frac = pr[wet], pc[wet], frac[wet]
    if len(blocked_px) and len(pr):
        bt = cKDTree(np.asarray(blocked_px))
        d, _i = bt.query(np.column_stack([pc + 0.5, pr + 0.5]))
        keep = d > BLOCK_RADIUS_PX
        pr, pc, frac = pr[keep], pc[keep], frac[keep]

    # 3. Graph: wet coast pixels, small water specks, big bodies as sources.
    lab, n_lab = ndimage.label(water, structure=_CONNECTIVITY_4)
    sizes = np.bincount(lab.ravel(), minlength=n_lab + 1)
    big = sizes >= min_body
    big[0] = False
    node = np.full((h, w), -1, dtype=np.int64)
    cost = []
    node[pr, pc] = np.arange(len(pr))
    cost.append(1.0 - 0.9 * frac)
    sr, sc = np.nonzero((lab > 0) & ~big[lab])
    node[sr, sc] = len(pr) + np.arange(len(sr))
    cost.append(np.full(len(sr), 0.1))
    base = len(pr) + len(sr)
    big_ids = np.flatnonzero(big)
    super_of = np.full(n_lab + 1, -1, dtype=np.int64)
    super_of[big_ids] = base + np.arange(len(big_ids))
    br, bc = np.nonzero(big[lab])
    node[br, bc] = super_of[lab[br, bc]]
    cost.append(np.zeros(len(big_ids)))
    cost = np.concatenate(cost)
    is_wet = np.zeros(len(cost), dtype=bool)
    is_wet[:len(pr)] = True
    is_super = np.zeros(len(cost), dtype=bool)
    is_super[base:] = True

    edges_u, edges_v = [], []
    # Horizontal neighbours share the vertical side x = c + 1.
    u, v = node[:, :-1], node[:, 1:]
    sel = (u >= 0) & (v >= 0) & (u != v)
    er, ec = np.nonzero(sel)
    uu, vv = u[sel], v[sel]
    test = is_wet[uu] | is_wet[vv]
    if test.any():
        tr, tc = er[test], ec[test]
        _f, inw = cross.query((tr[:, None] * k + np.arange(k)).ravel(), np.repeat(tc + 1.0, k))
        passable = np.ones(len(uu), dtype=bool)
        passable[test] = inw.reshape(-1, k).any(axis=1)
        uu, vv = uu[passable], vv[passable]
    edges_u.append(uu)
    edges_v.append(vv)
    # Vertical neighbours share the horizontal side y = r + 1: water must be
    # present on the sub-rows just above and just below it.
    u, v = node[:-1, :], node[1:, :]
    sel = (u >= 0) & (v >= 0) & (u != v)
    er, ec = np.nonzero(sel)
    uu, vv = u[sel], v[sel]
    test = is_wet[uu] | is_wet[vv]
    if test.any():
        tr, tc = er[test], ec[test].astype(np.float64)
        passable = np.ones(len(uu), dtype=bool)
        ok_side = np.ones(len(tr), dtype=bool)
        for srow in (tr * k + k - 1, (tr + 1) * k):
            a, _ = cross.query(srow, tc)
            b, _ = cross.query(srow, tc + 1.0)
            ok_side &= (b - a) > 1e-3
        passable[test] = ok_side
        uu, vv = uu[passable], vv[passable]
    edges_u.append(uu)
    edges_v.append(vv)
    eu, ev = np.concatenate(edges_u), np.concatenate(edges_v)

    carved = np.zeros((h, w), dtype=bool)
    straits = []
    sources = np.unique(np.concatenate([eu[is_super[eu]], ev[is_super[ev]]]))
    if len(sources) >= 2:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra

        wgt = np.maximum((cost[eu] + cost[ev]) / 2.0, 0.05)
        a_, b_ = np.minimum(eu, ev), np.maximum(eu, ev)
        order = np.lexsort((wgt, b_, a_))
        a_, b_, wgt = a_[order], b_[order], wgt[order]
        first = np.ones(len(a_), dtype=bool)
        first[1:] = (a_[1:] != a_[:-1]) | (b_[1:] != b_[:-1])
        a_, b_, wgt = a_[first], b_[first], wgt[first]
        n_nodes = len(cost)
        graph = coo_matrix((wgt, (a_, b_)), shape=(n_nodes, n_nodes)).tocsr()
        dist, pred, src = dijkstra(graph, directed=False, indices=sources, min_only=True,
                                   return_predecessors=True, limit=float(max_len))
        su, sv = src[a_], src[b_]
        meet = (su >= 0) & (sv >= 0) & (su != sv)
        total = dist[a_[meet]] + dist[b_[meet]] + wgt[meet]
        ma, mb, msu, msv = a_[meet], b_[meet], su[meet], sv[meet]
        within = total <= max_len
        ma, mb, msu, msv, total = ma[within], mb[within], msu[within], msv[within], total[within]
        # Cheapest meeting point per pair of bodies, before the Python loop.
        lo, hi = np.minimum(msu, msv), np.maximum(msu, msv)
        order = np.lexsort((total, hi, lo))
        lo, hi = lo[order], hi[order]
        first = np.ones(len(lo), dtype=bool)
        first[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
        pick = order[first]
        ma, mb, msu, msv, total = ma[pick], mb[pick], msu[pick], msv[pick], total[pick]
        label_of_super = np.full(len(cost), -1, dtype=np.int64)
        label_of_super[super_of[big_ids]] = big_ids

        # Kruskal over source bodies: one channel per MST edge.
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        accepted = []
        for i in np.argsort(total, kind="stable"):
            ra, rb = find(int(msu[i])), find(int(msv[i]))
            if ra == rb:
                continue
            parent[ra] = rb
            path = []
            for start in (int(ma[i]), int(mb[i])):
                nd = start
                while nd >= 0 and not is_super[nd]:
                    path.append(nd)
                    nd = pred[nd]
            path_px = [(int(pr[p]), int(pc[p])) if p < len(pr) else (int(sr[p - len(pr)]), int(sc[p - len(pr)]))
                       for p in path]
            wet_px = [rc for p, rc in zip(path, path_px) if p < len(pr)]
            if not wet_px:
                continue
            rr, cc = np.array(wet_px).T
            carved[rr, cc] = True
            mr, mc = path_px[len(path_px) // 2]
            accepted.append((mr, mc, {
                "carved_px": len(wet_px),
                "joins_px": sorted([int(sizes[label_of_super[int(msu[i])]]),
                                    int(sizes[label_of_super[int(msv[i])]])]),
                "x": int(mc), "y": int(mr),
            }))

        if accepted:
            # Report each channel at its nearest coastline point; the KD-tree
            # only holds segments next to carved pixels (all ~4M coastline
            # midpoints would take ~1 s to index).
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            near_carved = ndimage.maximum_filter(carved.view(np.uint8), size=7).view(bool)
            ix = np.clip(np.floor(mx).astype(np.int64), 0, w - 1)
            iy = np.clip(np.floor(my).astype(np.int64), 0, h - 1)
            cand = np.flatnonzero(near_carved.ravel()[iy * w + ix])
            if len(cand):
                tree = cKDTree(np.column_stack([mx[cand], my[cand]]))
                _d, near = tree.query([[mc + 0.5, mr + 0.5] for mr, mc, _info in accepted])
                for (mr, mc, info), j in zip(accepted, cand[near]):
                    straits.append({"lon": round(float(seg_lonlat[j, 0]), 4),
                                    "lat": round(float(seg_lonlat[j, 1]), 4), **info})

    water |= carved
    widened = _widen_thin_passages(water, carved | forced, width)
    water |= widened
    changed = (carved | widened | forced) & ~solid
    canvas[changed] = BACKGROUND_COLOR
    report = {
        "enabled": True,
        "straits": straits,
        "carved_px": int(carved.sum()),
        "widened_px": int(widened.sum()),
        "channel_px": int(forced.sum()),
    }
    return report, {"carved": carved, "widened": widened, "forced": forced, "water": water,
                    "hidden": hidden_land}


# --------------------------------------------------------------------------
PREVIEW_COLORS = {
    "sea": (52, 99, 153),
    "drowned": (21, 101, 92),
    "lake": (129, 199, 245),
    "land": (214, 209, 196),
    "hidden": (160, 156, 146),
    "separated": (0, 229, 255),
    "suppressed": (156, 39, 176),
    "grown": (118, 255, 3),
    "bridged": (255, 64, 129),
    "speck_filled": (141, 110, 99),
    "micro": (255, 214, 0),
    "sieved": (0, 150, 136),
    "carved": (229, 57, 53),
    "widened": (255, 152, 0),
    "channel": (213, 0, 249),
    "blocked": (255, 235, 59),
}


MASK_COLORS = {"land": (255, 255, 255), "sea": (0, 0, 0), "lake": (0, 0, 255)}


def land_sea_mask(image: Image.Image, debug) -> Image.Image:
    """Plain land/sea/lake image of a render: land white (hidden countries
    included -- they are land), sea black, lakes blue."""
    arr = np.asarray(image)
    water = (arr == np.asarray(BACKGROUND_COLOR, dtype=arr.dtype)).all(axis=2)
    if debug is not None and debug.get("hidden_land") is not None:
        water &= ~debug["hidden_land"]
    out = np.empty_like(arr)
    out[:] = MASK_COLORS["land"]
    out[water] = MASK_COLORS["sea"]
    if debug is not None and debug.get("lakes") is not None:
        out[debug["lakes"]] = MASK_COLORS["lake"]
    return Image.fromarray(out, mode="RGB")


def land_sea_preview(image: Image.Image, meta: dict, debug) -> Image.Image:
    """Land/sea view of a render for the Land/Sea tab: sea vs land (and
    lakes), with carved channels, widened pixels, manual channels and
    blocked points highlighted (and a ring around each carved channel so
    they can be found in a large image)."""
    arr = np.asarray(image)
    water = (arr == np.asarray(BACKGROUND_COLOR, dtype=arr.dtype)).all(axis=2)
    out = np.empty_like(arr)
    out[:] = PREVIEW_COLORS["land"]
    out[water] = PREVIEW_COLORS["sea"]
    if debug is not None and debug.get("lakes") is not None:
        out[debug["lakes"]] = PREVIEW_COLORS["lake"]
    if debug is not None and debug.get("drowned") is not None:
        out[debug["drowned"]] = PREVIEW_COLORS["drowned"]
    if debug is not None:
        if "hidden" in debug:
            out[debug["hidden"] & ~debug["water"]] = PREVIEW_COLORS["hidden"]
            out[debug["widened"]] = PREVIEW_COLORS["widened"]
            out[debug["carved"]] = PREVIEW_COLORS["carved"]
            out[debug["forced"]] = PREVIEW_COLORS["channel"]
        if "separated" in debug:
            out[debug["separated"]] = PREVIEW_COLORS["separated"]
        if "suppressed" in debug:
            out[debug["suppressed"]] = PREVIEW_COLORS["suppressed"]
            out[debug["grown"]] = PREVIEW_COLORS["grown"]
        if "speck_filled" in debug:
            out[debug["speck_filled"]] = PREVIEW_COLORS["speck_filled"]
        if "bridged" in debug:
            out[debug["bridged"]] = PREVIEW_COLORS["bridged"]
        if "sieved" in debug:
            out[debug["sieved"]] = PREVIEW_COLORS["sieved"]
        if "micro" in debug:
            out[debug["micro"]] = PREVIEW_COLORS["micro"]
    report = meta.get("land_sea", {})
    for s in report.get("straits", []):
        cv2.circle(out, (s["x"], s["y"]), 9, PREVIEW_COLORS["carved"], thickness=2, lineType=cv2.LINE_AA)
    for x, y in report.get("blocked_px", []):
        cv2.circle(out, (int(round(x)), int(round(y))), BLOCK_RADIUS_PX, PREVIEW_COLORS["blocked"],
                   thickness=2, lineType=cv2.LINE_AA)
    return Image.fromarray(out, mode="RGB")


# --------------------------------------------------------------------------
# Rivers (HydroRIVERS v1.0): the Rivers tab's map
# --------------------------------------------------------------------------

# The Rivers tab renders Hearts of Iron IV's rivers.bmp: an 8-bit indexed
# image with this exact palette (copied from the game's rivers.bmp; entries
# 17-253 are filler). 0 = source (green: the start of a river network's main
# river), 1 = flow-in (red: a tributary's pixel where it joins the river),
# 2 = flow-out (yellow: a river splitting off -- HydroRIVERS has no splits,
# so unused), 3-11 = river width from narrowest (light cyan) to widest
# (dark navy), 254 = water (sea and lakes), 255 = land. The game requires
# exactly one source per contiguous river network (red junctions included).
RIVER_PALETTE = ([(0, 255, 0), (255, 0, 0), (255, 252, 0), (0, 225, 255), (0, 200, 255), (0, 150, 255),
                  (0, 100, 255), (0, 0, 255), (0, 0, 225), (0, 0, 200), (0, 0, 150), (0, 0, 100),
                  (0, 85, 0), (0, 125, 0), (0, 158, 0), (24, 206, 0), (0, 0, 0)]
                 + [(2, 0, 1)] * 237 + [(122, 122, 122), (255, 255, 255)])
RIVER_SOURCE, RIVER_FLOW_IN, RIVER_FLOW_OUT = 0, 1, 2
RIVER_WIDTHS = list(range(3, 12))  # palette indices, narrow -> wide
RIVER_WATER, RIVER_LAND = 254, 255

# How big one river network (4-connected, tributaries included) may get.
# The game walks each network when it loads the map, at a cost that grows
# far faster than the network does, and a map of ours is 20-30x finer than
# vanilla's, so a whole basin drawn at full resolution fuses into one river
# the size of the map. Vanilla's biggest: 4,683 px, 12% of the map diagonal,
# 26 branch pixels. Measured on the user's Yugoslavia map (the Danube basin,
# 115,654 px / 91% / 623 branches, 20,053 provinces), as the time from
# launching the game to the first dated log line, against 60 s for the same
# map with no rivers at all:
#     branch cap   cuts   biggest network        load
#     none            0   115,654 px  623 br     never finished (>6 min)
#     200             4    43,994 px  198 br     never finished (>4 min)
#     100             9    22,830 px   99 br     never finished (>5 min)
#     50             20    16,319 px   50 br     140 s
#     24             43     7,160 px   24 br      65 s
# So the networks are cut down to vanilla's number of branch points, which
# on real rivers brings the size down with it; RIVER_MAX_NETWORK_PX is only
# a guard for a long river with few tributaries (it binds on no map tested).
RIVER_MAX_BRANCHES_DEFAULT = 25   # branch pixels (degree >= 3) per network; 0 = no limit
RIVER_MAX_NETWORK_PX = 16000      # pixels per network: the backstop for a long river with
                                  # few tributaries (16,319 px / 50 branches loaded in 140 s)


def _river_image(idx_img):
    """Indexed (mode P) image with the HOI4 rivers palette."""
    im = Image.fromarray(idx_img.astype(np.uint8), mode="P")
    im.putpalette([v for rgb in RIVER_PALETTE for v in rgb])
    return im
RIVER_TOP_PERCENT_DEFAULT = 10.0
# The user's own per-continent percentages (their Rivers tab, 2026-09-27)
# = the defaults for a continent not given explicitly.
RIVER_TOP_PERCENT_BY_CONTINENT = {"af": 2.0, "eu": 5.0, "as": 5.0, "oc": 4.0, "sa": 4.0, "na": 5.0}
RIVER_AUTO_DEFAULT = True  # the Rivers tab starts with automatic percentages (user's setting)
RIVER_MIN_LENGTH_DEFAULT = 10  # px; shorter river branches are removed (0/1 = keep all)
# Automatic percentages: target average spacing between rivers, in px
# (_auto_river_count). 70 ~ what 5% gives for Europe on a Europe-wide map
# (measured 72 px for an ESP/FIN/GRC/GBR view, 61 for PRT/AZE/FIN/TUN);
# 5% on a world map is only 12-23 px apart, South America 39 px on its own map.
RIVER_AUTO_SPACING_DEFAULT = 50.0  # user's setting (was 70)
# Continents for per-continent discharge filtering, as groups of HydroSHEDS
# regions (the first digit of HYRIV_ID). These are drainage-basin regions,
# so a river and all its tributaries always share one continent (no river
# changes percentage at a border) -- but their edges aren't political:
# "eu" also covers Turkey, the Caucasus, Iran, Arabia and the whole Caspian
# basin (Volga, Ural); "oc" includes Indonesia and the Philippines; "as" is
# HydroSHEDS' Siberia + Asia, "na" its North America + Arctic + Greenland.
RIVER_CONTINENTS = [
    ("af", "Africa", (1,)),
    ("eu", "Europe & Middle East", (2,)),
    ("as", "Asia", (3, 4)),
    ("oc", "Australia & Oceania", (5,)),
    ("sa", "South America", (6,)),
    ("na", "North America", (7, 8, 9)),
]
_river_index_memo = {}
_river_lock = threading.Lock()
_RIVER_SAMPLE_PX = 0.25  # densify step: < 1/sqrt(2) px, so consecutive samples are 8-neighbours at most


def load_river_index(path: Path = RIVERS_SHAPEFILE_PATH):
    """Compact per-segment index of HydroRIVERS (8.48M polylines, 48M
    points, 1.25 GB .shp -- memory-mapped, never loaded): bbox centre and
    extent (float32), record offset, point count and average discharge
    (DIS_AV_CMS, m3/s), downstream segment (NEXT_DOWN as a record index).
    ~5-6 s once per process (headers gathered in chunks via the .shx
    offsets, .dbf columns parsed with numpy; warmed at server start),
    plus LENGTH_KM / CATCH_SKM for the auto percentages; ~310 MB resident.
    None if HydroRIVERS isn't downloaded."""
    if not path.exists():
        return None
    with _river_lock:
        if path not in _river_index_memo:
            shx = np.fromfile(path.with_suffix(".shx"), dtype=">i4", offset=100).reshape(-1, 2)
            raw = np.memmap(path, dtype=np.uint8, mode="r")
            off = shx[:, 0].astype(np.int64) * 2 + 8
            n = len(off)
            cx, cy, ext = (np.empty(n, np.float32) for _ in range(3))
            n_points = np.empty(n, np.int32)
            chunk = 1_000_000
            for a in range(0, n, chunk):
                o = off[a:a + chunk]
                hdr = raw[o[:, None] + np.arange(44)]
                bb = hdr[:, 4:36].copy().view("<f8")
                cx[a:a + len(o)] = (bb[:, 0] + bb[:, 2]) / 2.0
                cy[a:a + len(o)] = (bb[:, 1] + bb[:, 3]) / 2.0
                ext[a:a + len(o)] = np.maximum(bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1]) / 2.0
                # Every HydroRIVERS record is a single-part polyline.
                n_points[a:a + len(o)] = hdr[:, 40:44].copy().view("<i4").ravel()
            dbf = path.with_suffix(".dbf")
            ids = _dbf_numeric_column(dbf, "HYRIV_ID").astype(np.int64)  # sorted ascending
            next_id = _dbf_numeric_column(dbf, "NEXT_DOWN").astype(np.int64)  # 0 = outlet
            nxt = np.minimum(np.searchsorted(ids, next_id), n - 1)
            nxt = np.where((next_id > 0) & (ids[nxt] == next_id), nxt, -1).astype(np.int32)
            _river_index_memo[path] = {
                "ids": ids.astype(np.int32),  # HYRIV_ID per record (river_borders.json refers to these)
                "raw": raw, "off": off.astype(np.uint32), "cx": cx, "cy": cy, "ext": ext, "n_points": n_points,
                "discharge": _dbf_numeric_column(dbf, "DIS_AV_CMS").astype(np.float32),
                "next": nxt,  # record index of the downstream segment, -1 at an outlet
                "region": (ids // 10_000_000).astype(np.uint8),  # HydroSHEDS region: HYRIV_ID's first digit
                # For density-based auto percentages (_auto_river_percent):
                "length_km": _dbf_numeric_column(dbf, "LENGTH_KM").astype(np.float32),
                "catch_km2": _dbf_numeric_column(dbf, "CATCH_SKM").astype(np.float32),  # own catchment
                "endorheic": _dbf_numeric_column(dbf, "ENDORHEIC").astype(np.int8),  # 1 = inland basin
                "order": _dbf_numeric_column(dbf, "ORD_CLAS").astype(np.int8),  # 1 = main stem, 2+ = tributary
            }
        return _river_index_memo[path]


_river_dead_memo = {}
RIVER_LAKE_TOL_DEG = 0.02  # an inland river end within ~2 km of a HydroLAKES lake flows into it
# A river mouth this close to the sea is extended to it: GADM's coast runs past
# HydroSHEDS' where GADM counts estuaries/reclaimed land as land -- 25-100 px
# for the Amazon/Para estuary, Sao Marcos bay, Rhine-Meuse delta, Flevoland,
# Szczecin lagoon (checked); a few px elsewhere.
RIVER_MOUTH_MAX_GAP_PX = 100
# Where that extension goes is a question about the ground, not about pixels,
# so it follows the elevation the way the user's own river generator does
# (Documents/Ring World/River Work, river_editor.py Terrain._dijkstra): the
# cheapest route, where a step costs its length, plus RIVER_SEA_CLIMB_PX_PER_M
# px per metre it climbs (UPHILL_COST there), plus up to
# RIVER_SEA_LOW_GROUND_COST for being high above the lowest ground around
# (that editor has a drainage raster and charges up to VALLEY_COST for leaving
# the drainage lines; here the heights are all there is, so the valley floor
# is "the lowest ground in the window"), plus, where the ground is flat and
# the cost would otherwise draw a ruler-straight line, smooth noise worth
# RIVER_SEA_MEANDER_COST so the river wanders a little (its FLAT_NOISE).
# Heights are the **raw NASA topography** in metres, like the sea level rise:
# the HOI4 exaggeration is cosmetic and would invent slopes. They are coarse
# -- NASA_M_PER_UNIT (~26.5 m) per step -- so a coastal plain reads as
# perfectly flat and only the noise term is left to shape the route, which is
# the honest answer: at that resolution there is no relief there to follow.
RIVER_SEA_CLIMB_PX_PER_M = 1.0
RIVER_SEA_LOW_GROUND_COST = 3.0
RIVER_SEA_MEANDER_COST = 2.5
RIVER_SEA_MEANDER_SCALES = (1.5, 4.0)   # px: the sizes of the noise features
RIVER_SEA_MEANDER_SEED = 7
# ... and how far round it may go to get there: route pixels per pixel of
# straight line to the water it reaches. On the user's France map every
# sound extension is within x1.53 (x1.41 = a diagonal staircase) and the one
# that looped back alongside its own river was x6.20 -- 149 px of route for
# 24 px of straight line, up a valley and back down another.
RIVER_MOUTH_MAX_DETOUR = 2.5
RIVER_MOUTH_DETOUR_SLACK_PX = 6   # a couple of pixels of gap are never a detour


def _segment_last_points(index, segs):
    """Last vertex (lon, lat) of each segment in `segs`, from the memory map."""
    last = index["off"][segs].astype(np.int64) + 48 + 16 * (index["n_points"][segs].astype(np.int64) - 1)
    raw = index["raw"]
    rows16 = np.lib.stride_tricks.as_strided(raw, shape=(len(raw) - 15, 16), strides=(1, 1))
    return np.ascontiguousarray(rows16[last]).view("<f8").reshape(-1, 2)


_river_terminal_memo = {}


def river_terminals(index):
    """Per HydroRIVERS segment, the outlet segment it finally drains to
    (pointer doubling along NEXT_DOWN, ~1 s once per process)."""
    key = id(index)
    with _river_lock:
        if key in _river_terminal_memo:
            return _river_terminal_memo[key]
    n = len(index["next"])
    jump = np.where(index["next"] >= 0, index["next"], np.arange(n, dtype=np.int32)).astype(np.int32)
    for _ in range(40):
        nxt = jump[jump]
        if np.array_equal(nxt, jump):
            break
        jump = nxt
    with _river_lock:
        _river_terminal_memo[key] = jump
    return jump


def river_dead_ends(index):
    """Bool per HydroRIVERS segment: its river ends inland somewhere other
    than a lake -- the endorheic outlet it drains to (NEXT_DOWN = 0,
    ENDORHEIC = 1) is not within RIVER_LAKE_TOL_DEG of a HydroLAKES lake.
    Such rivers are dropped from the river map: desert systems whose
    modelled discharge is an artefact (Wadi ad-Dawasir, 292 m3/s, ends in
    the Arabian sand), the Tarim (Lop Nur), the Okavango delta, ... Rivers
    into the Caspian (HydroLAKES' lake #1), the Aral, the Dead Sea or Lake
    Chad end in a lake and are kept; ocean outlets are always kept.
    ~5-10 s once per process (STRtree of the 32k inland ends against all
    lake bboxes, shapely distance test for the candidates, terminal outlet
    of every segment by pointer doubling). Without HydroLAKES nothing is
    dropped."""
    key = id(index)
    with _river_lock:
        if key in _river_dead_memo:
            return _river_dead_memo[key]
    import shapely
    n = len(index["next"])
    outlets = np.flatnonzero((index["next"] < 0) & (index["endorheic"] == 1))
    dead_outlet = np.zeros(n, dtype=bool)
    lakes = load_lake_index()
    if lakes is not None and len(outlets):
        ends = _segment_last_points(index, outlets)
        pts = shapely.points(ends)
        tol = RIVER_LAKE_TOL_DEG
        lb = lakes["bbox"]
        tree = shapely.STRtree(pts)
        lake_i, pt_i = tree.query(shapely.box(lb[:, 0] - tol, lb[:, 1] - tol, lb[:, 2] + tol, lb[:, 3] + tol),
                                  predicate="intersects")
        in_lake = np.zeros(len(outlets), dtype=bool)
        if len(lake_i):
            cand_lakes, inv = np.unique(lake_i, return_inverse=True)
            raw, off = lakes["raw"], lakes["off"]
            polys = []
            for li in cand_lakes:  # exterior ring (first part) of each candidate lake
                p0 = off[li] + 44 + 4 * lakes["n_parts"][li]
                parts = np.append(raw[off[li] + 44:p0].view("<i4"), lakes["n_points"][li])
                ring = np.frombuffer(raw[p0:p0 + 16 * parts[1]], dtype="<f8").reshape(-1, 2)
                polys.append(shapely.polygons(ring))
            near = shapely.dwithin(np.array(polys, dtype=object)[inv], pts[pt_i], tol)
            in_lake[pt_i[near]] = True
        dead_outlet[outlets[~in_lake]] = True
    dead = dead_outlet[river_terminals(index)]
    with _river_lock:
        _river_dead_memo[key] = dead
    return dead


def _river_points(index, segs):
    """Vertices (lon/lat) of HydroRIVERS segments `segs`, gathered straight
    from the memory map, and each segment's point count."""
    n_pts = index["n_points"][segs].astype(np.int64)
    p0 = index["off"][segs].astype(np.int64) + 48  # 44-byte polyline header + 1 part index
    seg_start = np.concatenate([[0], np.cumsum(n_pts)])
    pt_byte = np.repeat(p0, n_pts) + 16 * (np.arange(int(n_pts.sum())) - np.repeat(seg_start[:-1], n_pts))
    raw = index["raw"]
    rows16 = np.lib.stride_tricks.as_strided(raw, shape=(len(raw) - 15, 16), strides=(1, 1))  # 16 bytes per offset
    return np.ascontiguousarray(rows16[pt_byte]).view("<f8").reshape(-1, 2), n_pts


# --------------------------------------------------------------------------
# River borders: land borders that run along a river (river_borders.json)
# --------------------------------------------------------------------------

RIVER_BORDERS_PATH = BASE_DIR / "river_borders.json"
RIVER_BORDERS_VERSION = 1
RB_MIN_DISCHARGE = 5.0  # m3/s: smaller HydroRIVERS segments are never considered
RB_STEP_KM = 0.2        # border sampled every ~200 m
RB_TOL_KM = 1.5         # a border point this close to a river is on it ...
RB_WEAK_KM = 3.0        # ... and this close continues a stretch (HydroSHEDS' 15" lines drift)
RB_GAP_KM = 1.0         # longer stretches beyond RB_WEAK_KM end a stretch
RB_MIN_KM = 15.0        # shortest river border kept
RB_STRONG_FRAC = 0.6    # share of a stretch within RB_TOL_KM
RB_SNAP_KM = 3.0        # river vertices this close to their stretch are moved onto the border
# Names for the main river borders (not in either dataset): (tag a, tag b,
# lon, lat, radius deg, name) -- a stretch of that pair whose midpoint is
# within the radius gets the name.
RIVER_BORDER_NAMES = [
    # Europe
    ("DEU", "FRA", 7.7, 48.3, 1.5, "Rhine"), ("CHE", "DEU", 8.5, 47.6, 1.2, "Rhine"),
    ("AUT", "CHE", 9.6, 47.4, 0.5, "Rhine"), ("CHE", "LIE", 9.5, 47.1, 0.5, "Rhine"),
    ("DEU", "LUX", 6.3, 49.8, 1.0, "Moselle / Sauer / Our"), ("BEL", "NLD", 5.7, 51.0, 0.6, "Meuse"),
    ("DEU", "POL", 14.8, 51.6, 2.5, "Oder / Neisse"), ("AUT", "DEU", 13.0, 48.2, 1.0, "Inn / Salzach"),
    ("AUT", "SVK", 16.8, 48.4, 0.6, "Morava / Danube"), ("CZE", "SVK", 17.0, 48.8, 0.5, "Morava"),
    ("AUT", "CZE", 16.6, 48.7, 0.8, "Thaya / Morava"), ("HUN", "SVK", 18.8, 47.9, 1.2, "Danube / Ipel"),
    ("HRV", "SRB", 19.0, 45.5, 0.8, "Danube"), ("ROU", "SRB", 21.6, 44.7, 1.5, "Danube"),
    ("BGR", "SRB", 22.6, 44.2, 0.3, "Timok"), ("BGR", "ROU", 24.7, 43.7, 3.0, "Danube"),
    ("ROU", "UKR", 28.8, 45.3, 0.8, "Danube (Kiliya arm)"), ("ROU", "UKR", 24.0, 47.9, 0.8, "Tisza"),
    ("MDA", "ROU", 28.1, 46.9, 3.0, "Prut"), ("MDA", "UKR", 28.2, 48.2, 0.8, "Dniester"),
    ("MDA", "UKR", 30.0, 46.6, 0.5, "Dniester"),
    ("HRV", "HUN", 17.2, 46.1, 1.0, "Drava / Mura"), ("HRV", "SVN", 15.5, 45.8, 1.2, "Kupa / Sutla / Drava"),
    ("AUT", "SVN", 15.9, 46.7, 0.3, "Mura"), ("BIH", "HRV", 17.6, 45.1, 1.5, "Sava / Una"),
    ("BIH", "SRB", 19.1, 44.4, 1.0, "Drina"), ("BIH", "MNE", 19.0, 43.3, 0.3, "Tara"),
    ("ALB", "MNE", 19.4, 42.0, 0.4, "Buna / Drin"), ("GRC", "TUR", 26.3, 41.2, 1.0, "Evros / Maritsa"),
    ("BGR", "GRC", 26.2, 41.7, 0.3, "Maritsa"), ("BGR", "TUR", 27.9, 42.0, 0.3, "Rezovo"),
    ("HUN", "UKR", 22.5, 48.2, 0.5, "Tisza"), ("HUN", "ROU", 20.6, 46.1, 0.3, "Mures"),
    ("POL", "UKR", 23.9, 51.1, 1.5, "Bug"), ("BLR", "POL", 23.6, 51.9, 1.0, "Bug"),
    ("BLR", "UKR", 30.6, 51.7, 0.8, "Dnieper"), ("BLR", "LTU", 23.8, 53.9, 0.3, "Neman"),
    ("LTU", "RUS", 22.0, 55.1, 0.8, "Neman"), ("EST", "RUS", 28.1, 59.3, 0.5, "Narva"),
    ("EST", "RUS", 27.5, 58.2, 0.5, "Pededze / Lake Peipus"), ("BLR", "LVA", 27.5, 55.8, 0.3, "Daugava"),
    ("FIN", "SWE", 23.5, 67.5, 2.5, "Torne / Muonio"), ("FIN", "NOR", 26.4, 69.9, 1.2, "Tana"),
    ("NOR", "RUS", 29.4, 69.3, 0.6, "Pasvik"), ("ESP", "PRT", -6.6, 41.2, 0.8, "Douro"),
    ("ESP", "PRT", -7.0, 39.8, 0.6, "Tagus"), ("ESP", "PRT", -7.4, 38.2, 1.0, "Guadiana"),
    ("ESP", "PRT", -8.4, 42.1, 0.4, "Minho"), ("CHE", "FRA", 6.4, 46.4, 0.4, "Lake Geneva / Rhone"),
    ("CHE", "FRA", 6.9, 47.2, 0.4, "Doubs"), ("GBR", "IRL", -7.3, 54.5, 1.0, "Foyle / Erne"),
    ("RUS", "UKR", 39.7, 48.7, 0.5, "Seversky Donets"),
    # Caucasus, Middle East, Central Asia
    ("AZE", "IRN", 46.5, 39.2, 1.5, "Aras"), ("ARM", "TUR", 43.8, 40.1, 1.0, "Aras / Akhurian"),
    ("ARM", "IRN", 46.4, 38.9, 0.3, "Aras"), ("AZE", "GEO", 46.5, 41.3, 0.5, "Alazani"),
    ("AZE", "RUS", 48.3, 41.6, 0.4, "Samur"), ("IRN", "IRQ", 48.4, 30.2, 0.6, "Shatt al-Arab"),
    ("IRQ", "TUR", 42.6, 37.2, 0.4, "Khabur"), ("SYR", "TUR", 42.3, 37.3, 0.3, "Tigris"),
    ("SYR", "TUR", 36.4, 36.1, 0.3, "Orontes"), ("LBN", "SYR", 36.1, 34.6, 0.5, "Nahr al-Kabir"),
    ("ISR", "JOR", 35.6, 32.6, 0.4, "Jordan / Yarmouk"), ("JOR", "PSE", 35.6, 32.1, 0.5, "Jordan"),
    ("AFG", "TJK", 71.0, 37.8, 2.5, "Panj / Amu Darya"), ("AFG", "UZB", 67.2, 37.3, 0.8, "Amu Darya"),
    ("AFG", "TKM", 66.0, 37.4, 0.6, "Amu Darya"), ("AFG", "TKM", 62.8, 35.5, 0.8, "Murghab / Hari"),
    ("TKM", "UZB", 62.0, 40.8, 1.5, "Amu Darya"), ("AFG", "IRN", 61.1, 35.3, 0.8, "Hari (Tejen)"),
    ("IRN", "TKM", 61.2, 35.9, 0.8, "Hari (Tejen)"), ("IRN", "TKM", 54.8, 37.7, 0.8, "Atrek"),
    ("KAZ", "RUS", 53.0, 51.5, 1.5, "Ural"), ("KAZ", "RUS", 63.8, 54.3, 1.0, "Tobol"),
    ("KAZ", "UZB", 68.6, 41.0, 0.3, "Syr Darya"), ("CHN", "KAZ", 80.4, 43.5, 1.0, "Ili / Khorgos"),
    ("IND", "PAK", 74.7, 31.9, 0.8, "Ravi"), ("IND", "PAK", 73.9, 30.4, 0.3, "Sutlej"),
    # East and South Asia
    ("CHN", "RUS", 120.6, 53.1, 3.0, "Argun / Amur"), ("CHN", "RUS", 127.5, 50.3, 3.0, "Amur"),
    ("CHN", "RUS", 133.9, 46.5, 2.5, "Ussuri"), ("CHN", "RUS", 119.2, 50.3, 1.2, "Argun"),
    ("CHN", "PRK", 126.8, 41.3, 2.0, "Yalu"), ("CHN", "PRK", 129.6, 42.4, 1.2, "Tumen"),
    ("PRK", "RUS", 130.7, 42.4, 0.3, "Tumen"), ("LAO", "THA", 104.2, 17.6, 3.0, "Mekong"),
    ("LAO", "THA", 100.4, 20.4, 0.5, "Mekong"), ("LAO", "MMR", 100.5, 20.9, 0.8, "Mekong"),
    ("MMR", "THA", 98.3, 17.1, 1.5, "Moei / Salween"), ("CHN", "VNM", 103.8, 22.7, 0.8, "Red River / Nanxi"),
    ("MYS", "THA", 102.0, 6.0, 0.4, "Golok"), ("IND", "NPL", 80.4, 29.6, 0.8, "Mahakali (Sarda)"),
    ("IDN", "PNG", 140.9, -6.7, 0.8, "Fly"), ("MNG", "RUS", 107.4, 50.0, 0.5, "Selenga"),
    # Africa
    ("MRT", "SEN", -14.1, 16.5, 2.5, "Senegal"), ("MLI", "SEN", -11.4, 13.5, 1.5, "Faleme"),
    ("MLI", "MRT", -12.1, 14.8, 0.3, "Senegal"), ("NAM", "ZAF", 18.1, -28.9, 2.5, "Orange"),
    ("BWA", "ZAF", 27.7, -23.2, 2.0, "Limpopo"), ("BWA", "ZAF", 21.5, -26.3, 2.0, "Molopo / Nossob"),
    ("ZAF", "ZWE", 30.5, -22.3, 1.0, "Limpopo"), ("BWA", "ZWE", 28.5, -21.6, 0.6, "Shashe"),
    ("LSO", "ZAF", 27.7, -29.2, 1.5, "Caledon"), ("ZMB", "ZWE", 28.0, -17.0, 3.0, "Zambezi / Lake Kariba"),
    ("NAM", "ZMB", 25.0, -17.6, 0.6, "Zambezi"), ("BWA", "NAM", 24.2, -18.0, 1.0, "Chobe (Linyanti)"),
    ("MOZ", "ZMB", 30.3, -15.3, 0.4, "Zambezi / Luangwa"), ("AGO", "NAM", 19.8, -17.9, 2.0, "Okavango"),
    ("AGO", "NAM", 12.8, -17.1, 1.0, "Kunene"), ("MOZ", "TZA", 38.1, -11.3, 2.5, "Ruvuma"),
    ("MWI", "TZA", 33.3, -9.5, 0.5, "Songwe"), ("MOZ", "MWI", 35.3, -16.6, 0.6, "Shire / Ruo"),
    ("CAF", "COD", 22.0, 4.5, 4.5, "Ubangi / Mbomou"), ("COD", "COG", 16.1, -3.5, 1.5, "Congo"),
    ("COD", "COG", 18.0, 1.5, 1.5, "Ubangi"), ("CMR", "COG", 16.0, 1.9, 1.0, "Sangha"),
    ("CMR", "TCD", 15.1, 11.5, 1.5, "Logone / Chari"), ("CAF", "TCD", 19.4, 9.0, 2.0, "Aouk"),
    ("AGO", "COD", 21.9, -9.0, 1.5, "Kasai"), ("AGO", "COD", 17.3, -7.8, 0.6, "Kwango"),
    ("COD", "ZMB", 28.4, -11.8, 1.5, "Luapula"), ("BDI", "RWA", 29.9, -2.7, 0.6, "Akanyaru / Kagera"),
    ("RWA", "TZA", 30.8, -1.9, 1.0, "Kagera"), ("TZA", "UGA", 30.6, -1.1, 0.3, "Kagera"),
    ("RWA", "UGA", 30.4, -1.1, 0.3, "Kagera"), ("BDI", "COD", 29.2, -3.2, 0.6, "Ruzizi"),
    ("COD", "RWA", 29.0, -2.5, 0.5, "Ruzizi"), ("BDI", "TZA", 30.3, -3.5, 1.0, "Malagarasi / Ruvubu"),
    ("COD", "UGA", 30.2, 1.0, 0.5, "Semliki"), ("ETH", "SSD", 33.1, 8.1, 1.0, "Baro / Akobo"),
    ("ETH", "SSD", 34.4, 6.9, 1.0, "Akobo"), ("ERI", "ETH", 37.5, 14.4, 1.5, "Mareb / Tekeze"),
    ("BEN", "NER", 2.9, 12.4, 1.0, "Niger / Mekrou"), ("GHA", "TGO", 0.3, 9.8, 1.2, "Oti"),
    ("BFA", "GHA", -2.8, 10.4, 1.0, "Black Volta"), ("BFA", "CIV", -5.0, 10.0, 1.2, "Leraba / Black Volta"),
    ("CIV", "GHA", -2.8, 9.1, 0.6, "Black Volta"), ("CIV", "LBR", -7.4, 5.8, 1.5, "Cavalla"),
    ("LBR", "SLE", -11.0, 7.4, 1.0, "Mano"), ("GIN", "SLE", -10.6, 8.5, 0.5, "Moa / Makona"),
    ("GIN", "LBR", -9.2, 7.5, 1.0, "Makona / Mano"), ("CMR", "NGA", 14.6, 12.2, 0.8, "El Beid"),
    ("NER", "NGA", 12.8, 13.4, 1.0, "Komadugu Yobe"), ("DZA", "MAR", -8.3, 29.0, 0.6, "Draa"),
    ("COG", "GAB", 13.2, 1.3, 1.2, "Ivindo / Djoua"), ("CMR", "GAB", 12.3, 2.3, 1.0, "Ntem / Kom"),
    ("CMR", "GNQ", 10.0, 2.2, 0.4, "Ntem"), ("GAB", "GNQ", 10.0, 0.9, 0.4, "Muni / Utamboni"),
    ("SDN", "TCD", 22.5, 13.5, 2.0, "Wadi Kaja / Azum"), ("MOZ", "ZWE", 33.0, -17.7, 0.5, "Pungwe / Musapa"),
    # Americas
    ("MEX", "USA", -102.9, 29.2, 5.0, "Rio Grande"), ("MEX", "USA", -97.9, 26.1, 1.5, "Rio Grande"),
    ("MEX", "USA", -114.8, 32.6, 0.3, "Colorado"), ("CAN", "USA", -92.3, 48.2, 2.0, "Rainy River / boundary waters"),
    ("CAN", "USA", -79.0, 43.1, 0.4, "Niagara"), ("CAN", "USA", -75.0, 44.8, 1.0, "St. Lawrence"),
    ("CAN", "USA", -68.4, 47.3, 0.8, "Saint John"), ("CAN", "USA", -67.5, 45.5, 0.5, "St. Croix"),
    ("CAN", "USA", -83.0, 42.5, 0.6, "Detroit / St. Clair"), ("CAN", "USA", -84.3, 46.4, 0.5, "St. Marys"),
    ("GTM", "MEX", -90.5, 16.5, 1.2, "Usumacinta"), ("GTM", "MEX", -92.2, 14.8, 0.4, "Suchiate"),
    ("BLZ", "MEX", -88.7, 18.2, 0.6, "Hondo"), ("BLZ", "GTM", -89.1, 15.9, 0.3, "Sarstoon"),
    ("GTM", "SLV", -89.9, 14.0, 0.5, "Paz"), ("HND", "SLV", -88.2, 13.8, 0.8, "Lempa / Goascoran"),
    ("HND", "NIC", -84.5, 14.7, 1.5, "Coco (Segovia)"), ("CRI", "NIC", -84.1, 10.8, 0.8, "San Juan"),
    ("CRI", "PAN", -82.7, 9.6, 0.3, "Sixaola"), ("DOM", "HTI", -71.7, 19.1, 0.6, "Artibonite / Massacre"),
    ("COL", "VEN", -67.8, 5.0, 2.5, "Orinoco / Meta / Atabapo"), ("COL", "VEN", -70.5, 7.0, 0.6, "Arauca"),
    ("BRA", "COL", -69.5, 0.5, 1.5, "Rio Negro / Apaporis / Taraira"),
    ("COL", "PER", -73.0, -1.8, 3.0, "Putumayo"), ("COL", "PER", -70.2, -4.0, 0.5, "Amazon"),
    ("BRA", "PER", -73.0, -5.5, 1.5, "Javari (Yavari)"), ("BRA", "PER", -70.7, -4.2, 0.7, "Javari / Amazon"),
    ("COL", "ECU", -76.5, 0.3, 1.2, "San Miguel / Putumayo"), ("ECU", "PER", -75.3, -0.5, 0.6, "Putumayo / Napo"),
    ("BOL", "BRA", -65.9, -9.8, 2.0, "Mamore / Madeira"), ("BOL", "BRA", -62.5, -13.1, 2.5, "Guapore (Itenez)"),
    ("BOL", "BRA", -68.8, -11.0, 0.5, "Abuna / Acre"), ("BOL", "PER", -68.9, -13.1, 1.0, "Madre de Dios / Heath"),
    ("ARG", "PRY", -57.8, -25.8, 3.0, "Paraguay / Parana"), ("ARG", "PRY", -55.0, -26.8, 1.2, "Parana"),
    ("ARG", "PRY", -62.2, -22.7, 1.5, "Pilcomayo"), ("ARG", "BOL", -64.3, -22.8, 1.0, "Bermejo / Tarija"),
    ("BRA", "PRY", -57.9, -21.3, 1.0, "Paraguay / Apa"), ("BRA", "PRY", -54.5, -24.8, 0.8, "Parana"),
    ("ARG", "BRA", -54.8, -27.6, 2.5, "Uruguay"), ("ARG", "BRA", -53.9, -25.6, 0.6, "Iguazu / Pepiri-Guazu"),
    ("ARG", "URY", -58.0, -31.5, 1.5, "Uruguay"), ("BRA", "URY", -56.6, -30.3, 1.0, "Quarai (Cuareim)"),
    ("BRA", "URY", -53.7, -32.3, 0.6, "Jaguarao"), ("GUY", "VEN", -61.2, 6.5, 1.5, "Cuyuni"),
    ("GUY", "SUR", -57.2, 2.8, 1.5, "Courantyne"), ("GUF", "SUR", -54.1, 3.7, 2.0, "Maroni (Marowijne)"),
    ("BRA", "GUF", -52.5, 2.8, 1.5, "Oyapock"), ("BRA", "GUY", -59.6, 3.8, 1.5, "Takutu / Ireng"),
]

_river_borders_lock = threading.Lock()
_river_borders_memo = {}


def _river_borders_key(index) -> dict:
    return {"version": RIVER_BORDERS_VERSION, "gadm": _cache_key(SHAPEFILE_PATH),
            "hydrorivers_records": int(len(index["next"]))}


def build_river_borders(geometries: dict, index, progress=print) -> dict:
    """Find the GADM land borders that follow a HydroRIVERS river.

    Every land-border edge (ring_topology) is sampled every RB_STEP_KM and
    given its neighbour (nearest other country's border vertex, <= 1 km);
    samples within RB_TOL_KM of a river segment with >= RB_MIN_DISCHARGE
    m3/s are "on a river". A stretch = consecutive samples of one ring with
    one neighbour, within RB_WEAK_KM (gaps <= RB_GAP_KM bridged), trimmed to
    its first/last sample within RB_TOL_KM, >= RB_MIN_KM long, >=
    RB_STRONG_FRAC of it within RB_TOL_KM, and >= 80% of it next to one
    river system (same final outlet) -- so a border crossing rivers, or
    running through a dense network of small ones, isn't one. Each stretch
    keeps the vertex range of its ring in the alphabetically first country
    and the river segments running along it (>= half their vertices within
    RB_WEAK_KM, one within RB_TOL_KM; each segment in one stretch only).
    ~1 min."""
    t0 = time.time()
    tags = [t for t in geometries if t not in WATER_TAGS]
    bv, bt, bring, bidx, bcum = [], [], [], [], []
    for ti, tag in enumerate(tags):
        coast = ring_topology(geometries, tag)["coastal"]
        for ri, ((_h, ring), c) in enumerate(zip(geometries[tag]["rings"], coast)):
            if c.all():
                continue
            # Samples every ~RB_STEP_KM along the land-border edges (long
            # straight edges interpolated, dense vertices thinned): edge k
            # runs from vertex k to k + 1.
            nxt_v = np.roll(ring, -1, axis=0)
            step = np.linalg.norm(_lonlat_to_xyz(nxt_v) - _lonlat_to_xyz(ring), axis=1) * EARTH_RADIUS
            cum = np.concatenate([[0.0], np.cumsum(step)[:-1]])
            k = np.flatnonzero(~c)
            n = np.maximum(np.ceil(step[k] / RB_STEP_KM).astype(np.int64), 1)
            ei = np.repeat(k, n)
            t = (np.arange(len(ei)) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
            s_cum = cum[ei] + t * step[ei]
            cell = np.floor(s_cum / RB_STEP_KM).astype(np.int64)
            keep = np.r_[True, cell[1:] != cell[:-1]]
            ei, t, s_cum = ei[keep], t[keep], s_cum[keep]
            d_ = nxt_v[ei] - ring[ei]
            d_[:, 0] = (d_[:, 0] + 180.0) % 360.0 - 180.0
            bv.append(_lonlat_to_xyz(ring[ei] + d_ * t[:, None]))
            bidx.append(ei)
            bcum.append(s_cum)
            bt.append(np.full(len(ei), ti, np.int32))
            bring.append(np.full(len(ei), ri, np.int32))
    bv, bt, bring, bidx, bcum = map(np.concatenate, (bv, bt, bring, bidx, bcum))
    d, nn = cKDTree(bv).query(bv, k=12, workers=-1)
    other = bt[nn]
    ok = other != bt[:, None]
    first = np.argmax(ok, axis=1)
    nb = np.where(ok.any(axis=1) & (d[np.arange(len(bv)), first] * EARTH_RADIUS <= 1.0),
                  other[np.arange(len(bv)), first], -1)
    progress(f"  {len(bv):,} border samples ({time.time() - t0:.0f}s)")

    segs = np.flatnonzero(index["discharge"] >= RB_MIN_DISCHARGE)
    r_lonlat, r_n = _river_points(index, segs)
    r_start = np.concatenate([[0], np.cumsum(r_n)])
    r_pos = np.repeat(np.arange(len(segs)), r_n)
    tree_r = cKDTree(_lonlat_to_xyz(r_lonlat))
    dr, nr = tree_r.query(bv, workers=-1)
    dr *= EARTH_RADIUS
    term = river_terminals(index)
    progress(f"  {len(segs):,} river segments >= {RB_MIN_DISCHARGE:g} m3/s ({time.time() - t0:.0f}s)")

    order = np.lexsort((bcum, bring, bt))
    bv, bt, bring, bidx, bcum, nb, dr, nr = (a[order] for a in (bv, bt, bring, bidx, bcum, nb, dr, nr))
    brk = np.flatnonzero((np.diff(bt) != 0) | (np.diff(bring) != 0) | (np.diff(nb) != 0)) + 1
    stretches, seg_best = [], {}
    for grp in np.split(np.arange(len(bt)), brk):
        ta, tb = int(bt[grp[0]]), int(nb[grp[0]])
        if tb < 0 or tags[ta] > tags[tb]:
            continue
        weak = np.flatnonzero(dr[grp] <= RB_WEAK_KM)
        if not len(weak):
            continue
        cuts = np.flatnonzero(np.diff(bcum[grp][weak]) > RB_GAP_KM + RB_STEP_KM) + 1
        for run in np.split(weak, cuts):
            strong = run[dr[grp][run] <= RB_TOL_KM]
            if not len(strong):
                continue
            run = run[(run >= strong[0]) & (run <= strong[-1])]
            g = grp[run[0]:run[-1] + 1]
            length = float(bcum[g[-1]] - bcum[g[0]])
            if length < RB_MIN_KM or (dr[g] <= RB_TOL_KM).mean() < RB_STRONG_FRAC or (dr[g] <= RB_WEAK_KM).mean() < 0.8:
                continue
            t_, cnt = np.unique(term[segs[r_pos[nr[g]]]], return_counts=True)
            if cnt.max() < 0.8 * len(g):
                continue
            # River segments along the stretch.
            ring = geometries[tags[ta]]["rings"][int(bring[g[0]])][1]
            v0, v1 = int(bidx[g[0]]), min(int(bidx[g[-1]]) + 1, len(ring) - 1)  # edges -> vertices
            tree_p = cKDTree(_lonlat_to_xyz(ring[v0:v1 + 1]))
            near_pts = np.unique(np.concatenate(tree_r.query_ball_point(bv[g], RB_WEAK_KM / EARTH_RADIUS)).astype(np.int64))
            cand = np.unique(r_pos[near_pts])
            pts = np.concatenate([np.arange(r_start[s], r_start[s + 1]) for s in cand])
            dp = tree_p.query(_lonlat_to_xyz(r_lonlat[pts]))[0] * EARTH_RADIUS
            owner = np.searchsorted(cand, r_pos[pts])
            n_all = np.bincount(owner, minlength=len(cand))
            frac = np.bincount(owner, weights=dp <= RB_WEAK_KM, minlength=len(cand)) / n_all
            has_strong = np.bincount(owner, weights=dp <= RB_TOL_KM, minlength=len(cand)) > 0
            sid = len(stretches)
            for s, f in zip(cand[(frac >= 0.5) & has_strong].tolist(), frac[(frac >= 0.5) & has_strong].tolist()):
                rec = int(segs[s])
                if rec not in seg_best or f > seg_best[rec][1]:
                    seg_best[rec] = (sid, f)
            mid = ring[(v0 + v1) // 2]
            stretches.append({"a": tags[ta], "b": tags[tb], "ring": int(bring[g[0]]), "v0": v0, "v1": v1,
                              "length_km": round(length, 1), "lon": round(float(mid[0]), 3),
                              "lat": round(float(mid[1]), 3),
                              "max_cms": round(float(index["discharge"][segs[r_pos[nr[g]]]].max()), 1)})
    for st in stretches:
        st["segments"] = []
    for rec, (sid, _f) in sorted(seg_best.items()):
        stretches[sid]["segments"].append(int(index["ids"][rec]))
    for st in stretches:
        st["name"] = ""
        for a, b, lon, lat, rad, name in RIVER_BORDER_NAMES:
            if (st["a"], st["b"]) == (a, b) and math.hypot(st["lon"] - lon, st["lat"] - lat) <= rad:
                st["name"] = name
                break
    progress(f"  {len(stretches)} river border stretches, {sum(s['length_km'] for s in stretches):,.0f} km, "
             f"{len(seg_best):,} river segments ({time.time() - t0:.0f}s)")
    return {**_river_borders_key(index), "stretches": stretches}


def _write_river_borders(data: dict, path: Path):
    """One stretch per line: small (~100 KB) and diffable."""
    head = {k: v for k, v in data.items() if k != "stretches"}
    lines = [json.dumps(s, separators=(",", ":")) for s in data["stretches"]]
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(head)[:-1] + ', "stretches": [\n' + ",\n".join(lines) + "\n]}\n")


def river_borders(geometries: dict, index, progress=print):
    """river_borders.json (built and saved when missing, or made for other
    GADM / HydroRIVERS data), plus lookup arrays: "seg_rec" (HydroRIVERS
    record indices along a river border, sorted) and "seg_stretch" (its
    stretch). None without the river index."""
    if index is None:
        return None
    key = _river_borders_key(index)
    with _river_borders_lock:
        memo = _river_borders_memo.get(id(index))
        if memo is not None:
            return memo
        data = None
        if RIVER_BORDERS_PATH.exists():
            with open(RIVER_BORDERS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if any(data.get(k) != v for k, v in key.items()):
                data = None
        if data is None:
            progress("Finding land borders along rivers (river_borders.json)...")
            data = build_river_borders(geometries, index, progress)
            _write_river_borders(data, RIVER_BORDERS_PATH)
        ids = [(i, s) for s, st in enumerate(data["stretches"]) for i in st["segments"]]
        rec = np.searchsorted(index["ids"], np.array([i for i, _s in ids], dtype=np.int64))
        o = np.argsort(rec)
        data["seg_rec"] = rec[o]
        data["seg_stretch"] = np.array([s for _i, s in ids], dtype=np.int64)[o]
        data["_lines"] = {}
        _river_borders_memo[id(index)] = data
        return data


def _border_line(geometries: dict, rb: dict, sid: int):
    """Stretch `sid`'s border polyline: (lon/lat vertices, KD-tree of their
    unit vectors, cumulative length km), memoized."""
    lines = rb["_lines"]
    if sid not in lines:
        st = rb["stretches"][sid]
        line = geometries[st["a"]]["rings"][st["ring"]][1][st["v0"]:st["v1"] + 1]
        xyz = _lonlat_to_xyz(line)
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]) * EARTH_RADIUS
        lines[sid] = (line, cKDTree(xyz), cum)
    return lines[sid]


def _snap_rivers_to_borders(geometries: dict, rb, sel, lonlat, n_pts):
    """Move the selected river segments that run along a river border
    (river_borders.json) onto the GADM border: each vertex within
    RB_SNAP_KM of its stretch goes to the nearest border vertex, and the
    border's own vertices between two snapped ones are inserted, so the
    river follows every bend of the border. A tributary's confluence
    vertex (shared with the snapped segment) moves with it. Returns
    (lonlat, n_pts, stretch per point: -1 where not snapped)."""
    none = np.full(len(lonlat), -1, dtype=np.int32)
    if rb is None or not len(rb["seg_rec"]) or not len(sel):
        return lonlat, n_pts, none
    p = np.minimum(np.searchsorted(rb["seg_rec"], sel), len(rb["seg_rec"]) - 1)
    on = rb["seg_rec"][p] == sel
    if not on.any():
        return lonlat, n_pts, none
    seg_start = np.concatenate([[0], np.cumsum(n_pts)])
    chunks, flags, new_n, moved = [], [], n_pts.copy(), {}
    for i in range(len(sel)):
        s0, s1 = int(seg_start[i]), int(seg_start[i + 1])
        pts = lonlat[s0:s1]
        if not on[i]:
            chunks.append(pts)
            flags.append(np.full(len(pts), -1, dtype=np.int32))
            continue
        sid = int(rb["seg_stretch"][p[i]])
        line, tree, cum = _border_line(geometries, rb, sid)
        xyz = _lonlat_to_xyz(pts)
        d, j = tree.query(xyz)
        snap = d * EARTH_RADIUS <= RB_SNAP_KM
        out, fl, prev = [], [], -1
        for q in range(len(pts)):
            if not snap[q]:
                out.append(pts[q:q + 1])
                fl.append(-1)
                prev = -1
                continue
            jq = int(j[q])
            if prev >= 0 and abs(jq - prev) > 1:
                lo, hi = min(prev, jq), max(prev, jq)
                straight = float(np.linalg.norm(tree.data[jq] - tree.data[prev])) * EARTH_RADIUS
                if cum[hi] - cum[lo] <= 3.0 * straight + 1.0:
                    mid = line[lo + 1:hi] if jq > prev else line[lo + 1:hi][::-1]
                    out.append(mid)
                    fl.extend([sid] * len(mid))
            out.append(line[jq:jq + 1])
            fl.append(sid)
            moved[(float(pts[q, 0]), float(pts[q, 1]))] = line[jq]
            prev = jq
        seg = np.concatenate(out)
        chunks.append(seg)
        flags.append(np.array(fl, dtype=np.int32))
        new_n[i] = len(seg)
    # Confluences: a segment's end vertex shared with a snapped one moves too.
    if moved:
        for i in np.flatnonzero(~on).tolist():
            c = chunks[i]
            for q in (0, len(c) - 1):
                m = moved.get((float(c[q, 0]), float(c[q, 1])))
                if m is not None:
                    if c.base is not None:
                        c = chunks[i] = c.copy()
                    c[q] = m
    return np.concatenate(chunks), new_n, np.concatenate(flags)


RB_FIT_MAX_PX = 12   # a pocket of the wrong country farther than this from its river is left alone
RB_FIT_GAP_PX = 30   # path pixels between two pieces of one stretch (detours, joins) belong to it


def _fit_borders_to_rivers(canvas, painted, px_str, rb, country_rgb, min_px: int = 8, dry_run: bool = False, marks=None):
    """Make every river border of the country map (`canvas`, modified in
    place) follow its drawn river, which never moves (the HOI4 river rules
    come first): per stretch, the river keeps to one side -- the country
    most of its pixels already lie in (S), the other one being O. Along
    each piece of the river (consecutive path pixels drawn from the
    stretch's snapped vertices, px_str), every pixel within RB_FIT_MAX_PX is
    classified left/right of the river by the local direction of the path;
    each 4-connected pocket of the wrong country (O on S's side, S on O's
    side) that touches the river and lies wholly within that band is given
    to the other country, i.e. the border moves onto the river where
    detours / gaps / other rivers pushed the river off it. River pixels
    beside both sides are then painted S. Left-over pieces of S or O under
    `min_px` without a river pixel are merged into their most common
    neighbour. Only pixels of the two countries change. Returns a report:
    river_px / on_side_px / on_border_px = river pixels along stretches on
    either country's land (ends excluded) / of them in S / of them beside O
    (8-neighbours);
    dry_run only measures."""
    h, w = px_str.shape
    code = (canvas[:, :, 0].astype(np.int32) << 16) | (canvas[:, :, 1].astype(np.int32) << 8) | canvas[:, :, 2]
    runs = {}
    for k, path in painted.items():
        p = np.asarray(path, dtype=np.int64)
        if len(p) < 5:
            continue
        s = px_str[p[:, 0], p[:, 1]].copy()
        known = np.flatnonzero(s >= 0)
        if not len(known):
            continue
        # Fill short unknown stretches between two pixels of one stretch.
        gaps = np.flatnonzero(np.diff(known) > 1)
        for g in gaps.tolist():
            a, b = known[g], known[g + 1]
            if s[a] == s[b] and b - a - 1 <= RB_FIT_GAP_PX:
                s[a + 1:b] = s[a]
        brk = np.flatnonzero(np.diff(s) != 0) + 1
        for seg in np.split(np.arange(len(p)), brk):
            if s[seg[0]] >= 0 and len(seg) >= 5:
                runs.setdefault(int(s[seg[0]]), []).append(p[seg])
    rep = {"stretches": 0, "moved_px": 0, "cleaned_px": 0, "river_px": 0, "on_side_px": 0, "on_border_px": 0,
           "skipped": 0}
    touched = np.zeros((h, w), dtype=bool)
    for sid, rs in sorted(runs.items()):
        st = rb["stretches"][sid]
        ca, cb = (int((c[0] << 16) | (c[1] << 8) | c[2]) for c in (country_rgb[st["a"]], country_rgb[st["b"]]))
        if ca == cb:
            continue
        allp = np.concatenate(rs)
        cols = code[allp[:, 0], allp[:, 1]]
        na, nb = int((cols == ca).sum()), int((cols == cb).sum())
        if na + nb < len(allp) // 2:
            rep["skipped"] += 1
            continue
        S, O = (ca, cb) if na >= nb else (cb, ca)
        rep["stretches"] += 1
        for run in rs:
            n = len(run)
            r0, c0 = max(int(run[:, 0].min()) - RB_FIT_MAX_PX - 1, 0), max(int(run[:, 1].min()) - RB_FIT_MAX_PX - 1, 0)
            r1, c1 = min(int(run[:, 0].max()) + RB_FIT_MAX_PX + 2, h), min(int(run[:, 1].max()) + RB_FIT_MAX_PX + 2, w)
            sub = code[r0:r1, c0:c1]
            loc = run - [r0, c0]
            on_run = np.zeros(sub.shape, dtype=bool)
            on_run[loc[:, 0], loc[:, 1]] = True
            band = ndimage.distance_transform_edt(~on_run) <= RB_FIT_MAX_PX
            ab = ((sub == S) | (sub == O)) & ~on_run
            qs = np.argwhere(band & ab)
            if not len(qs):
                continue
            _d, i = cKDTree(loc).query(qs)
            valid = (i >= 2) & (i <= n - 3)
            tan = loc[np.minimum(i + 3, n - 1)] - loc[np.maximum(i - 3, 0)]
            rel = qs - loc[i]
            cross = tan[:, 0] * rel[:, 1] - tan[:, 1] * rel[:, 0]
            side = np.zeros(sub.shape, dtype=np.int8)  # +1 / -1 = left / right of the river, 0 unknown
            side[qs[:, 0], qs[:, 1]] = np.where(valid, np.sign(cross), 0).astype(np.int8)
            is_o = sub == O
            o_left, o_right = int((is_o & (side > 0)).sum()), int((is_o & (side < 0)).sum())
            s_left, s_right = int(((sub == S) & (side > 0)).sum()), int(((sub == S) & (side < 0)).sum())
            o_side = 1 if o_left * (s_right + 1) >= o_right * (s_left + 1) else -1
            near_run = ndimage.binary_dilation(on_run, structure=_CONNECTIVITY_4)
            for wrong_col, wrong_side, new_col in (() if dry_run else ((O, -o_side, S), (S, o_side, O))):
                wrong = (sub == wrong_col) & ~on_run
                lab, nl = ndimage.label(wrong, structure=_CONNECTIVITY_4)
                if not nl:
                    continue
                ids = np.arange(1, nl + 1)
                good = ndimage.sum_labels(side == wrong_side, lab, ids) == ndimage.sum_labels(wrong, lab, ids)
                touch = ndimage.maximum(near_run, lab, ids) > 0
                flip = ids[good & touch]  # "good" = wholly classified, i.e. within the band
                if len(flip):
                    m = np.isin(lab, flip)
                    sub[m] = new_col
                    touched[r0:r1, c0:c1] |= m
                    rep["moved_px"] += int(m.sum())
            # River pixels beside both countries (on the border): S.
            rr, cc = loc[:, 0], loc[:, 1]
            pad = np.pad(sub, 1, constant_values=-1)
            nb8 = np.stack([pad[1 + dr + rr, 1 + dc + cc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                            if dr or dc], axis=1)
            beside_o = (nb8 == O).any(axis=1)
            # S only where it joins S by a side (else a 1 px island of S):
            # repeated, since each one painted can join the next.
            while not dry_run:
                nb4 = np.stack([pad[1 + dr + rr, 1 + dc + cc] for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))], axis=1)
                fix = (sub[rr, cc] == O) & beside_o & (nb4 == S).any(axis=1)
                if not fix.any():
                    break
                sub[rr[fix], cc[fix]] = S
                pad[1 + rr[fix], 1 + cc[fix]] = S
                touched[rr[fix] + r0, cc[fix] + c0] = True
                rep["moved_px"] += int(fix.sum())
            inner_px = np.arange(2, n - 2)
            inner_px = inner_px[(sub[rr[inner_px], cc[inner_px]] == S) | (sub[rr[inner_px], cc[inner_px]] == O)]
            rep["river_px"] += len(inner_px)  # on either country's land (not across a lake)
            rep["on_side_px"] += int((sub[rr[inner_px], cc[inner_px]] == S).sum())
            if marks is not None:
                bad = sub[rr[inner_px], cc[inner_px]] != S
                marks.setdefault("off_side", []).append(run[inner_px][bad])
            pad = np.pad(sub, 1, constant_values=-1)
            beside = np.stack([pad[1 + dr + rr[inner_px], 1 + dc + cc[inner_px]] == O
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc], axis=1).any(axis=1)
            rep["on_border_px"] += int(beside.sum())
            if marks is not None:
                marks.setdefault("off_border", []).append(run[inner_px][~beside])
    # Clean-up. The flips can cut pieces off a country (a sliver between a
    # lake and the river, a strip past the river's end) or leave specks:
    # every piece of a changed country that is under `min_px`, or that
    # was split off a larger area by the flips (all but the largest piece
    # of it) and holds no river pixel of a stretch, goes to its most common
    # land neighbour -- unless it reaches the window edge.
    if touched.any():
        orig = (canvas[:, :, 0].astype(np.int32) << 16) | (canvas[:, :, 1].astype(np.int32) << 8) | canvas[:, :, 2]
        river = np.zeros((h, w), dtype=bool)
        for rs in runs.values():
            for run in rs:
                river[run[:, 0], run[:, 1]] = True
        water = {int((c[0] << 16) | (c[1] << 8) | c[2]) for c in (BACKGROUND_COLOR, LAKE_COLOR)}
        margin = 2 * RB_FIT_MAX_PX
        lab_t, _nt = ndimage.label(ndimage.binary_dilation(touched, iterations=2))
        for sl in ndimage.find_objects(lab_t):
            r0, r1 = max(sl[0].start - margin, 0), min(sl[0].stop + margin, h)
            c0, c1 = max(sl[1].start - margin, 0), min(sl[1].stop + margin, w)
            sub, osub, riv = code[r0:r1, c0:c1], orig[r0:r1, c0:c1], river[r0:r1, c0:c1]
            t_ = touched[r0:r1, c0:c1]
            edge = np.zeros(sub.shape, dtype=bool)
            edge[[0, -1], :] = True
            edge[:, [0, -1]] = True
            for col in np.union1d(sub[t_], osub[t_]).tolist():
                if col in water:
                    continue
                la, na = ndimage.label(sub == col, structure=_CONNECTIVITY_4)
                if not na:
                    continue
                ids = np.arange(1, na + 1)
                lb, _nb = ndimage.label(osub == col, structure=_CONNECTIVITY_4)
                size = ndimage.sum_labels(np.ones(sub.shape), la, ids)
                at_edge = ndimage.maximum(edge, la, ids) > 0
                keep = (ndimage.maximum(riv, la, ids) > 0) | at_edge
                src = ndimage.maximum(lb, la, ids).astype(np.int64)  # the area it was part of before
                drop = (size < min_px) & ~at_edge  # specks go even with a river pixel in them
                for b in np.unique(src[src > 0]).tolist():
                    parts = np.flatnonzero(src == b)
                    if len(parts) > 1:
                        biggest = parts[np.argmax(size[parts])]
                        drop[parts[(parts != biggest) & ~keep[parts]]] = True
                for li in ids[drop].tolist():
                    comp = la == li
                    ring = ndimage.binary_dilation(comp, structure=_CONNECTIVITY_4) & ~comp
                    vals, cnt = np.unique(sub[ring], return_counts=True)
                    land = np.array([v not in water and v != col for v in vals.tolist()], dtype=bool)
                    if not land.any():
                        continue
                    sub[comp] = vals[land][np.argmax(cnt[land])]
                    rep["cleaned_px"] += int(comp.sum())
    changed = code != ((canvas[:, :, 0].astype(np.int32) << 16) | (canvas[:, :, 1].astype(np.int32) << 8) | canvas[:, :, 2])
    canvas[changed] = np.stack([(code[changed] >> 16) & 255, (code[changed] >> 8) & 255, code[changed] & 255],
                               axis=1).astype(np.uint8)
    return rep


def _selection_strokes(index, sel):
    """Group the drawn selection of segments into rivers: at every
    confluence, the drawn upstream branch with the most discharge is the
    river's continuation (ties: lower ORD_CLAS), the others are tributaries.
    Based on what is drawn, not on HydroSHEDS' ORD_CLAS alone, whose main
    stem follows the *longest* branch -- the Sakarya's upper course came out
    a tributary (52 m3/s branch vs a 40 m3/s "main stem" at 32.0E 39.7N) --
    and whose main stem may not be drawn at all upstream of a confluence.
    Returns (root segment, hops to it) per selected segment."""
    sel = np.asarray(sel)
    n = len(sel)
    order_s = np.argsort(sel, kind="stable")
    ss = sel[order_s]
    nxt = index["next"][sel]
    p_ = np.clip(np.searchsorted(ss, nxt), 0, n - 1)
    npos = np.where((nxt >= 0) & (ss[p_] == nxt), order_s[p_], -1)
    ii = np.flatnonzero(npos >= 0)
    o = np.lexsort((index["order"][sel][ii], -index["discharge"][sel][ii], npos[ii]))
    ii = ii[o]
    best = ii[np.r_[True, npos[ii][1:] != npos[ii][:-1]]] if len(ii) else ii
    cont = np.zeros(n, dtype=bool)
    cont[best] = True
    jump = np.where(cont, npos, np.arange(n)).astype(np.int64)
    hops = cont.astype(np.int64)
    for _ in range(40):
        nj = jump[jump]
        if np.array_equal(nj, jump):
            break
        hops = hops + hops[jump]
        jump = nj
    return sel[jump], hops


def _bfs_path(passable, start, goal):
    """Shortest 4-connected path in a window from `start` (row, col) to the
    nearest `goal` pixel, moving through `passable` pixels only (the goal
    itself needn't be passable). Returns the pixels after `start`, goal
    included, or None."""
    from collections import deque
    h, w = passable.shape
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur != start and goal[cur]:
            path = []
            while cur != start:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]
        r, c = cur
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in prev and (passable[nb] or goal[nb]):
                prev[nb] = cur
                q.append(nb)
    return None


def _elevation_route(passable, goal, start, z):
    """Cheapest 4-connected path in a window from `start` (row, col) to the
    cheapest `goal` pixel, over `passable` pixels (a goal pixel needn't be
    passable, and no path goes *through* one -- a route may not cross the
    water to reach water further out). `z` = the window's heights in metres.

    Stepping into a pixel costs
        1 (the step's length)
      + RIVER_SEA_CLIMB_PX_PER_M x the metres it climbs
      + RIVER_SEA_LOW_GROUND_COST x how high it lies between the lowest and
        the highest ground in the window (0 where the window is flat)
      + RIVER_SEA_MEANDER_COST x smooth noise, only where the ground around
        it has no relief at all.
    So the route runs down the lowest ground it can find, goes round a ridge
    unless the detour costs more than the climb, and, where the heights say
    nothing, wanders instead of drawing a straight line (see
    RIVER_SEA_CLIMB_PX_PER_M). Same shape of answer as `_bfs_path`: the
    pixels after `start`, the goal included, or None.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    h, w = passable.shape
    node = passable | goal
    node[start] = True              # the river's own last pixel is where this starts from
    if not goal.any():
        return None
    z = np.asarray(z, dtype=np.float32)
    lo, hi = float(z[node].min()), float(z[node].max())
    extra = np.zeros((h, w), dtype=np.float32)
    if hi > lo:
        extra += RIVER_SEA_LOW_GROUND_COST * (z - lo) / (hi - lo)
    # the noise only where the ground is flat: with relief to follow, the
    # heights decide, and a wander over a ridge would be nonsense
    flat = ndimage.maximum_filter(z, size=5) - ndimage.minimum_filter(z, size=5) <= 0.0
    if flat.any():
        rng = np.random.default_rng(RIVER_SEA_MEANDER_SEED)
        noise = np.zeros((h, w), dtype=np.float32)
        for scale in RIVER_SEA_MEANDER_SCALES:
            f = ndimage.gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), scale)
            noise += f / (float(f.std()) + 1e-9)
        noise -= noise.min()
        extra += RIVER_SEA_MEANDER_COST * flat * (noise / (float(noise.max()) + 1e-9))
    rows, cols, wts = [], [], []
    idx = np.arange(h * w).reshape(h, w)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        sr, sc = slice(max(0, -dr), h - max(0, dr)), slice(max(0, -dc), w - max(0, dc))
        tr, tc = slice(max(0, dr), h + min(0, dr)), slice(max(0, dc), w + min(0, dc))
        ok = node[sr, sc] & node[tr, tc] & ~goal[sr, sc]      # never on through water
        rows.append(idx[sr, sc][ok])
        cols.append(idx[tr, tc][ok])
        wts.append((1.0 + RIVER_SEA_CLIMB_PX_PER_M * np.maximum(z[tr, tc] - z[sr, sc], 0.0)
                    + extra[tr, tc])[ok])
    g = csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(h * w, h * w))
    dist, pred = dijkstra(g, indices=start[0] * w + start[1], return_predecessors=True)
    cand = np.flatnonzero(goal.ravel() & np.isfinite(dist))
    if not len(cand):
        return None
    out, i = [], int(cand[np.argmin(dist[cand])])
    while i >= 0:
        out.append((i // w, i % w))
        i = int(pred[i])
    return out[-2::-1]        # after `start`, the goal last


def _near_others(owner, k, r0, r1, c0, c1):
    """Window mask: pixels within 1 px (8-neighbourhood) of a river other than k."""
    win = owner[r0:r1, c0:c1]
    return ndimage.maximum_filter(((win >= 0) & (win != k)).view(np.uint8), size=3).view(bool)


def _own_near(path, keep_last: int, r0: int, r1: int, c0: int, c1: int):
    """Window mask: pixels within 1 px (8-neighbourhood) of a river's own
    path, ignoring its last `keep_last` pixels (a path may continue from its
    end, touching them as consecutive steps do)."""
    m = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    pts = path[:-keep_last] if keep_last else path
    for pr, pc in pts:
        if r0 - 1 <= pr <= r1 and c0 - 1 <= pc <= c1:
            m[max(0, pr - 1 - r0):max(0, pr + 2 - r0), max(0, pc - 1 - c0):max(0, pc + 2 - c0)] = True
    return m


def _self_avoiding_pass(path, corner_ok):
    """Make a river's pixel path never touch itself: wherever it comes back
    within 1 px (8-neighbourhood) of an earlier pixel -- a meander brushing
    itself, a line doubling back -- the loop is cut short, jumping from that
    pixel straight to the farthest later pixel beside it (a diagonal brush is
    crossed through a side pixel, if `corner_ok(pixel)`). The only diagonal
    neighbours left are consecutive staircase steps (i and i+2)."""
    pos = {}
    for i, q in enumerate(path):
        pos[q] = i
    out = []
    i, n = 0, len(path)
    while i < n:
        pr, pc = path[i]
        out.append((pr, pc))
        best = -1
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                j = pos.get((pr + dr, pc + dc), -1)
                if j > best and (j > i + 2 or (dr == 0 and dc == 0 and j > i)):
                    best = j
        if best < 0:
            i += 1
            continue
        qr, qc = path[best]
        if (qr, qc) == (pr, pc):
            i = best + 1                       # revisited this very pixel
        elif abs(qr - pr) + abs(qc - pc) == 1:
            i = best                           # side by side: jump
        else:
            for corner in ((pr, qc), (qr, pc)):  # diagonal: through a side pixel
                if corner_ok(corner):
                    out.append(corner)
                    i = best
                    break
            else:
                i += 1
    return out


def _self_avoiding(path, corner_ok):
    """_self_avoiding_pass until the path doesn't touch itself anywhere (a
    side pixel inserted for a diagonal brush, or the jump after it, can land
    beside a pixel already kept). Each pass only shortens the path; the
    last pixel is always kept."""
    for _ in range(50):
        new = _self_avoiding_pass(path, corner_ok)
        if len(new) == len(path) and new == path:
            break
        path = new
    return path


def _draw_strokes(rows, cols, sidx, parent, allowed, h: int, w: int):
    """Draw rivers one at a time, in the order they come (parents before
    their tributaries, biggest first), enforcing the river rule: a river is
    1 px wide (a side-connected path), never touches itself (_self_avoiding)
    and never comes within 1 px -- diagonals included -- of another river,
    except at a tributary's confluence: its last pixel sits side by side
    with the parent (the one before may touch it diagonally, where the
    parent runs diagonally and no other approach exists).
      * A tributary is cut where it first comes near its parent for good
        (the rest of its path is <= 10 px or mostly beside the parent), then
        joined (_join_path, <= 10 px, else <= 45 px within 30 px) -- when
        its confluence is inside the frame on land; one outside the frame
        (or in the sea band) just flows out of view. A tributary that can't
        join -- or whose parent isn't drawn at a confluence inside the
        frame -- isn't drawn, nor anything flowing into it: a tributary
        always touches its parent.
      * Stretches that would come too close to a river already drawn are
        rerouted (shortest detour in a small window), else left out (a
        gap); a blocked tail stops the river.
      * Sea pixels (outside `allowed`) are skipped.
    Painted pixels are never changed afterwards. Returns (owner image: river
    index or -1, painted paths {river: [(row, col), ...]}, stats)."""
    owner = np.full((h, w), -1, dtype=np.int32)
    near = np.zeros((h, w), dtype=bool)  # within 1 px of any drawn river
    stats = {"rerouted": 0, "gaps": 0, "truncated": 0, "joined": 0, "unjoined": 0, "early_joins": 0}
    painted = {}
    dropped = set()  # tributaries that couldn't join: not drawn, nor anything flowing into them
    starts = np.flatnonzero(np.r_[True, sidx[1:] != sidx[:-1]])
    ends = np.r_[starts[1:], len(sidx)]
    D8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]

    def free(q):
        return 0 <= q[0] < h and 0 <= q[1] < w and allowed[q] and owner[q] == -1 and not near[q]

    for s0, s1 in zip(starts.tolist(), ends.tolist()):
        k = int(sidx[s0])
        r, c = rows[s0:s1], cols[s0:s1]
        par = int(parent[k])
        if par in dropped:
            dropped.add(k)
            continue
        # The confluence (the river's last point) inside the frame, on land:
        # only then must a tributary join its parent here. Outside the frame
        # (or in the sea band) it simply flows out of view -- the Sava on an
        # Italy map joins the Danube beyond the edge.
        cr_, cc_ = int(rows[s1 - 1]), int(cols[s1 - 1])
        conf_in_frame = 0 <= cr_ < h and 0 <= cc_ < w
        conf_inside = conf_in_frame and bool(allowed[cr_, cc_])  # on land: it must join
        joins = par >= 0 and par in painted and conf_in_frame
        if joins:
            nb = np.stack([owner[np.clip(r + dr, 0, h - 1), np.clip(c + dc, 0, w - 1)] for dr, dc in D8])
            nb[:, ~((r >= 0) & (r < h) & (c >= 0) & (c < w))] = -1
            near_par = (nb == par).any(axis=0)
            for i in np.flatnonzero(near_par).tolist():
                if len(r) - i <= 10 or near_par[i:].mean() >= 0.5:
                    r, c = r[:i], c[:i]
                    stats["early_joins"] += 1
                    break
        path = []
        if len(r):
            inb = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            rc, cc = np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)
            valid = inb & allowed[rc, cc]
            # A river that reaches the sea and comes back onto land within
            # 30 px (HydroSHEDS' line through a GADM bay/estuary, onto an
            # estuary island) ends at the sea instead.
            first_ok = np.flatnonzero(valid)
            if len(first_ok):
                wet = np.flatnonzero(inb & ~allowed[rc, cc])
                wet = wet[wet > first_ok[0]]
                if len(wet) and len(r) - wet[0] <= 30:
                    r, c, inb, rc, cc, valid = (r[:wet[0]], c[:wet[0]], inb[:wet[0]], rc[:wet[0]], cc[:wet[0]],
                                                valid[:wet[0]])
            good = valid & (owner[rc, cc] == -1) & ~near[rc, cc]
            idx = np.flatnonzero(good)
            if len(idx):
                path = [(int(r[idx[0]]), int(c[idx[0]]))]
                for i, j in zip(idx[:-1].tolist(), idx[1:].tolist()):
                    # Blocked by another river, or a short stretch of sea
                    # (a small inlet the line crosses): detour over land.
                    if j > i + 1 and inb[i + 1:j].all() and (valid[i + 1:j].all() or j - i <= 8):
                        a_, b_ = path[-1], (int(r[j]), int(c[j]))
                        r0, r1 = max(0, min(a_[0], b_[0]) - 6), min(h, max(a_[0], b_[0]) + 7)
                        c0, c1 = max(0, min(a_[1], b_[1]) - 6), min(w, max(a_[1], b_[1]) + 7)
                        passable = (allowed[r0:r1, c0:c1] & (owner[r0:r1, c0:c1] == -1) & ~near[r0:r1, c0:c1]
                                    & ~_own_near(path, 2, r0, r1, c0, c1))
                        goal = np.zeros_like(passable)
                        goal[b_[0] - r0, b_[1] - c0] = True
                        det = _bfs_path(passable, (a_[0] - r0, a_[1] - c0), goal)
                        if det is not None and len(det) <= 3 * (j - i) + 12:
                            path.extend((pr + r0, pc + c0) for pr, pc in det[:-1])
                            stats["rerouted"] += 1
                        else:
                            stats["gaps"] += 1
                    path.append((int(r[j]), int(c[j])))
                if idx[-1] < len(r) - 1 and valid[idx[-1] + 1:].any():
                    stats["truncated"] += 1
                path = _self_avoiding(path, free)
        if joins and path:
            # Join the parent; if even the wide search fails, the tributary
            # isn't drawn (a tributary always touches its parent).
            det = None
            for radius, max_len in ((11, 10), (30, 45)):
                det = _join_path(path, par, owner, near, allowed, radius, max_len)
                if det is not None:
                    break
            if det is None and conf_inside:
                stats["unjoined"] += 1
                dropped.add(k)
                continue
            if det is not None:
                path = _self_avoiding(path + det, free)
                stats["joined"] += 1
            # else: the confluence is in the sea (next to the parent's mouth):
            # the tributary reaches the water there on its own.
        elif par >= 0 and path and conf_inside:
            # Parent not drawn at a confluence inside the frame: it would dangle.
            stats["unjoined"] += 1
            dropped.add(k)
            continue
        if not path:
            continue
        pr_, pc_ = np.array(path).T
        owner[pr_, pc_] = k
        for dr, dc in D8:
            near[np.clip(pr_ + dr, 0, h - 1), np.clip(pc_ + dc, 0, w - 1)] = True
        painted[k] = path
    return owner, painted, stats


def _join_path(path, par, owner, near, allowed, radius: int, max_len: int):
    """Pixels joining a tributary's path end to its parent (drawn), or None:
    the shortest path to a pixel side by side with exactly one parent pixel
    (in the inner corner of the parent's staircase it would make a 2x2
    blob), clear of other rivers and of the tributary's own path, whose
    other pixels are not near any river. Where the parent runs diagonally,
    every pixel beside it has its other neighbours diagonally next to the
    parent, so the second-to-last pixel may touch the parent diagonally."""
    h, w = owner.shape
    lr, lc = path[-1]
    r0, r1, c0, c1 = max(0, lr - radius), min(h, lr + radius + 1), max(0, lc - radius), min(w, lc + radius + 1)
    win = owner[r0:r1, c0:c1]
    par_px = win == par
    if not par_px.any():
        return None
    other = (win >= 0) & (win != par)
    clear_other = ~ndimage.maximum_filter(other.view(np.uint8), size=3).view(bool)
    own = _own_near(path, 2, r0, r1, c0, c1)
    n_side = ndimage.convolve(par_px.view(np.uint8), _CONNECTIVITY_4.astype(np.uint8), mode="constant")
    side = (n_side >= 1) & ~par_px
    ok_px = (win == -1) & allowed[r0:r1, c0:c1] & clear_other & ~own
    goal = (n_side == 1) & ~par_px & ok_px
    diag_only = ndimage.maximum_filter(par_px.view(np.uint8), size=3).view(bool) & ~side & ~par_px
    goal2 = diag_only & ok_px & ndimage.binary_dilation(goal, structure=_CONNECTIVITY_4)
    if near is None:  # computed for this window only (1 px of context around it)
        q0, q1, p0_, p1_ = max(0, r0 - 1), min(h, r1 + 1), max(0, c0 - 1), min(w, c1 + 1)
        nw = ndimage.maximum_filter((owner[q0:q1, p0_:p1_] >= 0).view(np.uint8), size=3).view(bool)
        near_w = nw[r0 - q0:r0 - q0 + (r1 - r0), c0 - p0_:c0 - p0_ + (c1 - c0)]
    else:
        near_w = near[r0:r1, c0:c1]
    passable = (win == -1) & allowed[r0:r1, c0:c1] & ~near_w & ~own
    det = _bfs_path(passable, (lr - r0, lc - c0), goal | goal2)
    if det is not None and goal2[det[-1]] and not goal[det[-1]]:
        er, ec = det[-1]
        nxt_ = [(er + a, ec + b) for a, b in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= er + a < r1 - r0 and 0 <= ec + b < c1 - c0 and goal[er + a, ec + b]]
        det = det + [nxt_[0]] if nxt_ else None
    if det is None or len(det) > max_len:
        return None
    return [(pr + r0, pc + c0) for pr, pc in det]


LAKE_HUG_MIN_PX = 3  # a river with this many pixels running along a lake's shore is pulled into the lake


def _lake_route(path, path_arr, t, j, L, k, owner, lake_lab, depth, allowed, h, w, stats):
    """Least-cost route replacing path[s_i..e_i] (the stretch t..j plus one
    anchor pixel each side) through lake L: lake pixels cost 1 + 4 / depth,
    land 8; clear of other rivers and of the rest of the river's own path
    (but the 2 pixels each side of the stretch). Returns (route, s_i, e_i)
    or (None, ...)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    s_i, e_i = max(0, t - 1), min(len(path) - 1, j + 1)
    s_q, e_q = path[s_i], path[e_i]
    seg = path[s_i:e_i + 1]
    ys, xs = path_arr[s_i:e_i + 1, 0], path_arr[s_i:e_i + 1, 1]
    r0, r1 = max(0, int(ys.min()) - 8), min(h, int(ys.max()) + 9)
    c0, c1 = max(0, int(xs.min()) - 8), min(w, int(xs.max()) + 9)
    win_own = owner[r0:r1, c0:c1]
    other = (win_own >= 0) & (win_own != k)
    clear = ~ndimage.maximum_filter(other.view(np.uint8), size=3).view(bool)
    own_block = np.zeros_like(clear)
    sel_ = np.ones(len(path_arr), dtype=bool)
    sel_[max(0, s_i - 2):e_i + 3] = False  # the route may step next to the path just outside
    rp = path_arr[sel_]
    rp = rp[(rp[:, 0] >= r0 - 1) & (rp[:, 0] <= r1) & (rp[:, 1] >= c0 - 1) & (rp[:, 1] <= c1)]
    if len(rp):
        m_ = np.zeros((r1 - r0 + 2, c1 - c0 + 2), dtype=bool)
        m_[rp[:, 0] - r0 + 1, rp[:, 1] - c0 + 1] = True
        own_block = ndimage.maximum_filter(m_.view(np.uint8), size=3).view(bool)[1:-1, 1:-1]
    in_lake = lake_lab[r0:r1, c0:c1] == L
    passable = in_lake & clear & ~own_block & allowed[r0:r1, c0:c1]
    for (y, x) in seg:  # the old stretch stays usable (to leave / reach the shore)
        if clear[y - r0, x - c0]:
            passable[y - r0, x - c0] = True
    # Never onto the river's own pixels outside the stretch (only next to
    # the 2 each side): the new path must not revisit them.
    out_px = path_arr[np.r_[max(0, s_i - 2):s_i, e_i + 1:min(len(path_arr), e_i + 3)]]
    for y, x in out_px:
        if r0 <= y < r1 and c0 <= x < c1:
            passable[y - r0, x - c0] = False
    passable[s_q[0] - r0, s_q[1] - c0] = passable[e_q[0] - r0, e_q[1] - c0] = True
    # Lake pixels cheap (cheaper away from the shore), land dear: even a
    # 1-2 px wide lake beats the shore (1 + 4/depth vs 8 kept thin lakes out).
    cost = np.where(in_lake, 1.0 + 2.0 / np.maximum(depth[r0:r1, c0:c1], 1.0), 10.0)
    hh, ww = passable.shape
    idx = -np.ones((hh, ww), dtype=np.int64)
    pr_, pc_ = np.nonzero(passable)
    idx[pr_, pc_] = np.arange(len(pr_))
    us, vs, ws = [], [], []
    for dr, dc in ((0, 1), (1, 0)):
        a_, b_ = idx[:hh - dr, :ww - dc], idx[dr:, dc:]
        m = (a_ >= 0) & (b_ >= 0)
        us.append(a_[m]); vs.append(b_[m])
        ws.append((cost[:hh - dr, :ww - dc][m] + cost[dr:, dc:][m]) / 2.0)
    u, v, wgt = np.concatenate(us), np.concatenate(vs), np.concatenate(ws)
    si, ei = idx[s_q[0] - r0, s_q[1] - c0], idx[e_q[0] - r0, e_q[1] - c0]
    route = None
    if len(u) and si >= 0 and ei >= 0 and si != ei:
        g = coo_matrix((wgt, (u, v)), shape=(len(pr_), len(pr_))).tocsr()
        dist, pred = dijkstra(g, directed=False, indices=si, return_predecessors=True)
        if np.isfinite(dist[ei]):
            route, nd = [], ei
            while nd != si and nd >= 0:
                route.append((int(pr_[nd]) + r0, int(pc_[nd]) + c0))
                nd = pred[nd]
            route.append(s_q)
            route = route[::-1]
    if route is None or len(route) > 3 * len(seg) + 20 or not any(lake_lab[q] > 0 for q in route):
        stats["no_route" if route is None else "too_long_or_dry"] += 1
        return None, s_i, e_i
    return route, s_i, e_i


def _pull_rivers_into_lakes(owner, painted, parent, lakes, allowed):
    """Rivers go through lakes, but HydroRIVERS and the (GADM-cut) HydroLAKES
    lakes don't always line up: a river often runs just beside a lake,
    along its shore, instead of inside it. For each stretch of a river that
    stays in or right next to one lake and hugs its shore for >=
    LAKE_HUG_MIN_PX pixels, redraw it as the least-cost path from the
    stretch's start to its end where lake pixels are cheap (the further from
    the shore the cheaper) and land expensive -- so it crosses the lake's
    interior. The new stretch keeps the river rule: clear of every other
    river and of the rest of its own path. Tributaries that joined the old
    stretch are re-joined (_join_path); if one can't be, that stretch keeps
    its course (a whole-river revert threw away every fix of long rivers).
    Parents first (drawing order). Route search: _lake_route. In place;
    returns (stretches rerouted, pixels pulled in)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    h, w = owner.shape
    lake_lab, _n = ndimage.label(lakes, structure=_CONNECTIVITY_4)
    near_lab = ndimage.maximum_filter(lake_lab, size=3)  # a lake next to (or at) each pixel
    depth = ndimage.distance_transform_edt(lakes)        # inside a lake: distance to its shore
    D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
    n_done = pulled = 0
    stats = {"no_route": 0, "too_long_or_dry": 0, "reverted": 0}
    tribs_of = {}
    for x in painted:
        if x < len(parent) and parent[x] >= 0:
            tribs_of.setdefault(int(parent[x]), []).append(x)
    for k in sorted(painted):
        t = 0
        while k in painted and t < len(painted[k]):
            path = painted[k]
            if len(path) < 4:
                break
            path_arr = np.array(path)
            lab_at = near_lab[path_arr[:, 0], path_arr[:, 1]]
            hug = ~lakes[path_arr[:, 0], path_arr[:, 1]] & (lab_at > 0)
            # Next stretch (from t) that stays by one lake and hugs its shore.
            found = None
            while t < len(path):
                L = lab_at[t]
                if L == 0:
                    t += 1
                    continue
                j = t
                while j + 1 < len(path) and lab_at[j + 1] == L:
                    j += 1
                if int(hug[t:j + 1].sum()) >= LAKE_HUG_MIN_PX:
                    found = (t, j, L)
                    break
                t = j + 1
            if found is None:
                break
            t, j, L = found
            route, s_i, e_i = _lake_route(path, path_arr, t, j, L, k, owner, lake_lab, depth, allowed, h, w, stats)
            if route is None:
                t = j + 1
                continue
            new_path = path[:s_i] + route + path[e_i + 1:]
            removed = set(path[s_i:e_i + 1]) - set(route)
            for q in removed:
                owner[q] = -1
            for q in route:
                owner[q] = k
            painted[k] = new_path
            # Tributaries that joined a removed pixel: re-join, else undo this stretch.
            undo, ok = [], True
            for tt in tribs_of.get(k, []):
                if tt not in painted:
                    continue
                ty, tx = painted[tt][-1]
                nb = [(ty + a2, tx + b2) for a2, b2 in D4 if 0 <= ty + a2 < h and 0 <= tx + b2 < w]
                if any(owner[q] == k for q in nb) or not any(q in removed for q in nb):
                    continue
                det = _join_path(painted[tt], k, owner, None, allowed, 11, 10)
                if det is None:
                    ok = False
                    break
                undo.append((tt, len(painted[tt])))
                for q in det:
                    owner[q] = tt
                painted[tt] = painted[tt] + det
            if not ok:
                for tt, n_old in undo:
                    for q in painted[tt][n_old:]:
                        owner[q] = -1
                    painted[tt] = painted[tt][:n_old]
                for q in route:
                    owner[q] = -1
                for q in path:  # the whole original path (the route may share pixels with it)
                    owner[q] = k
                painted[k] = path
                stats["reverted"] += 1
                t = j + 1
                continue
            n_done += 1
            pulled += sum(1 for q in route if lakes[q])
            t = s_i + len(route)
    _pull_rivers_into_lakes.last_stats = stats
    return n_done, pulled


def _extend_strokes_to_sea(owner, painted, mouths, sea, lakes, allowed, heights=None):
    """Continue every river that should reach the sea (ocean outlets) but
    whose last drawn pixel doesn't touch it -- GADM's coast (the map) runs
    past HydroSHEDS' (the rivers) in deltas, estuaries and polders -- to the
    sea within RIVER_MOUTH_MAX_GAP_PX, keeping the river rule: never within
    1 px of another river, nor of the river itself (so neighbouring mouths
    don't merge). Rivers ending next to a lake (a lagoon) count as having
    reached water.

    **`heights`** (the frame's raw NASA topography, 0-255 as
    `render_heightmap` gives it, or metres as a float array): the route then
    follows the ground -- downhill along the lowest way to the water, round a
    ridge rather than over it, wandering where the ground is flat
    (`_elevation_route`, the cost model of the user's own river generator;
    see RIVER_SEA_CLIMB_PX_PER_M). Without it the route is the one with the
    fewest pixels (`_bfs_path`), which draws a straight line across whatever
    lies in the way.

    A mouth whose only route is more than `RIVER_MOUTH_MAX_DETOUR` times the
    straight line to the water is left alone -- it ends inland.

    In place; returns (rivers extended, pixels added, mouths left for the
    water being too far round)."""
    h, w = owner.shape
    gap = RIVER_MOUTH_MAX_GAP_PX
    z = None
    if heights is not None:
        raw = np.asarray(heights)
        if raw.shape != owner.shape:
            raise RenderError(f"the heights are {raw.shape[1]}x{raw.shape[0]}, the map {w}x{h}")
        z = raw.astype(np.float32)
        if raw.dtype == np.uint8:                   # NASA grey -> metres
            z *= NASA_M_PER_UNIT
    n_ext = added = n_far = 0
    for k in mouths:
        if k not in painted:
            continue
        path = painted[k]
        r, c = path[-1]
        if any(0 <= r + dr < h and 0 <= c + dc < w and sea[r + dr, c + dc]
               for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
            continue
        if lakes[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any():
            continue
        r0, r1, c0, c1 = max(0, r - gap), min(h, r + gap + 1), max(0, c - gap), min(w, c + gap + 1)
        sea_w = sea[r0:r1, c0:c1]
        if not sea_w.any():
            continue
        passable = (allowed[r0:r1, c0:c1] & (owner[r0:r1, c0:c1] == -1)
                    & ~_near_others(owner, k, r0, r1, c0, c1) & ~_own_near(path, 2, r0, r1, c0, c1))
        det = (_elevation_route(passable, sea_w, (r - r0, c - c0), z[r0:r1, c0:c1]) if z is not None
               else _bfs_path(passable, (r - r0, c - c0), sea_w))
        if det is None:
            continue
        # A mouth that only reaches water the long way round is not a mouth:
        # the river really does end inland, and drawing it there sends it
        # **snaking back alongside its own course** -- the loop the user
        # found on the France map, 149 px of route for 24 px of straight
        # line. Every other extension on that map is within x1.53 of the
        # straight line (x1.41 is a plain diagonal staircase), so the cap
        # separates them cleanly. The shortest-path router needs it just as
        # much: it drew the same snake, which _remove_straight_rivers then
        # threw away *with the whole river*.
        far = float(np.hypot(det[-1][0] - (r - r0), det[-1][1] - (c - c0)))
        if len(det) > max(RIVER_MOUTH_DETOUR_SLACK_PX, RIVER_MOUTH_MAX_DETOUR * far):
            n_far += 1
            continue
        # The last step is the sea pixel itself. The search keeps clear of
        # the river's path but its last 2 pixels, so only that tail + the
        # extension are made self-avoiding (re-running it on the whole path
        # could cut out the pixel a tributary joins at).
        det = [(pr + r0, pc + c0) for pr, pc in det[:-1]]
        tail = _self_avoiding(path[-2:] + det, lambda q: allowed[q] and owner[q] in (-1, k))
        for q in set(path[-2:]) - set(tail):
            owner[q] = -1
        for q in tail:
            owner[q] = k
        painted[k] = path[:-2] + tail
        det = tail[2:] if len(tail) >= 2 else det
        if det:
            n_ext += 1
            added += len(det)
    return n_ext, added, n_far


def _remove_short_strokes(owner, painted, parent_of, path_dis, min_len: int):
    """Remove rivers (visible sections, see _split_sections) shorter than
    `min_len` px that have no tributary drawn -- a whole short river, a
    short tributary, or a small corner of a river clipping the frame (each
    section is a river of its own, so the frame edge doesn't exempt it).
    One pass: a river whose short tributaries go isn't re-measured. In
    place; returns (rivers, px)."""
    if min_len <= 1:
        return 0, 0
    has_child = {int(p_) for k, p_ in parent_of.items() if p_ >= 0 and k in painted}
    n = px = 0
    for k in list(painted):
        path = painted[k]
        if k in has_child or len(path) >= min_len:
            continue
        pr_, pc_ = np.array(path).T
        owner[pr_, pc_] = -1
        del painted[k]
        path_dis.pop(k, None)
        parent_of.pop(k, None)
        n += 1
        px += len(path)
    return n, px


RIVER_STRAIGHT_RUN_PX = 8          # a run of this many same-direction steps is a straight edge
RIVER_MAX_STRAIGHT_FRACTION = 0.5  # rivers with more of their pixels in straight edges are removed


def _straight_fraction(path):
    """Share of a pixel path lying in straight edges: runs of at least
    RIVER_STRAIGHT_RUN_PX consecutive steps in the same direction. The
    pixel searches (mouth extensions to the sea, detours around other
    rivers, joins) draw such horizontal / vertical runs; a river rasterized
    from HydroRIVERS rarely keeps one direction that long."""
    if len(path) < 2:
        return 0.0
    a = np.asarray(path)
    step = np.diff(a, axis=0)
    code = step[:, 0] * 3 + step[:, 1]  # one value per direction
    change = np.flatnonzero(np.diff(code) != 0) + 1
    starts = np.r_[0, change]
    lengths = np.diff(np.r_[starts, len(code)])
    straight_steps = int(lengths[lengths >= RIVER_STRAIGHT_RUN_PX].sum())
    return straight_steps / len(path)


def _remove_straight_rivers(owner, painted, path_dis, parent_of):
    """Remove rivers that are mostly straight edges (more than
    RIVER_MAX_STRAIGHT_FRACTION of their pixels, see _straight_fraction) --
    they look artificial: typically a short river whose mouth extension or
    detours are most of it -- together with everything flowing into them
    (their tributaries would otherwise end in empty land). In place;
    returns (rivers removed as straight, rivers removed with them, px)."""
    straight = {k for k, p in painted.items() if _straight_fraction(p) > RIVER_MAX_STRAIGHT_FRACTION}
    if not straight:
        return 0, 0, 0
    children = {}
    for k, par in parent_of.items():
        if par >= 0 and k in painted:
            children.setdefault(par, []).append(k)
    gone, stack = set(), list(straight)
    while stack:
        k = stack.pop()
        if k in gone or k not in painted:
            continue
        gone.add(k)
        stack.extend(children.get(k, []))
    px = 0
    for k in gone:
        pr_, pc_ = np.array(painted[k]).T
        owner[pr_, pc_] = -1
        px += len(painted[k])
        del painted[k]
        path_dis.pop(k, None)
        parent_of.pop(k, None)
    return len(straight), len(gone) - len(straight), px


def _touches_itself(path) -> bool:
    pos = {q: i for i, q in enumerate(path)}
    for i, (y, x) in enumerate(path):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                j = pos.get((y + dy, x + dx))
                if j is not None and j > i + 2:
                    return True
    return False


def _repair_self_contacts(owner, painted, path_dis, parent_of, allowed):
    """Final safety pass: a river that still touches itself (a pixel added
    after its self-avoidance -- a join, a mouth extension -- closing a tight
    bend, seen near the frame edge in Robinson/Mollweide views) gets
    _self_avoiding again, unless that would drop a pixel a tributary joins
    at; an inserted side pixel must be free and clear of other rivers. In
    place; returns the number of rivers repaired."""
    h, w = owner.shape
    D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
    repaired = 0
    for k in list(painted):
        path = painted[k]
        if not _touches_itself(path):
            continue
        joins = set()
        for t, par in parent_of.items():
            if par == k and t in painted:
                ty, tx = painted[t][-1]
                joins.update((ty + a, tx + b) for a, b in D4)

        def ok(q):
            if not (0 <= q[0] < h and 0 <= q[1] < w) or not allowed[q] or owner[q] != -1:
                return False
            nb = owner[max(0, q[0] - 1):q[0] + 2, max(0, q[1] - 1):q[1] + 2]
            return bool(((nb == -1) | (nb == k)).all())
        new = _self_avoiding(path, ok)
        dropped = set(path) - set(new)
        if dropped & joins:
            continue
        dmap = dict(zip(path, path_dis[k].tolist()))
        for q in dropped:
            owner[q] = -1
        for q in new:
            owner[q] = k
        painted[k] = new
        path_dis[k] = np.maximum.accumulate(np.array([dmap.get(q, np.nan) for q in new], dtype=np.float64))
        if np.isnan(path_dis[k]).any():  # inserted side pixels: their neighbour's discharge
            v = path_dis[k]
            ix = np.where(~np.isnan(v), np.arange(len(v)), 0)
            np.maximum.accumulate(ix, out=ix)
            path_dis[k] = v[ix]
        repaired += 1
    return repaired


def _merge_head_tributaries(owner, painted, path_dis, parent_of):
    """A river section that starts right where a tributary joins it (its
    upstream part lies beyond a gap or outside the frame) has no end to put
    its source on; the tributary is simply its upstream continuation. Merge
    them into one river: the tributary's path followed by the section's (the
    biggest such tributary if several). In place; returns merges done."""
    h, w = owner.shape
    D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
    merged = 0
    for s_ in list(painted):
        if s_ not in painted:
            continue
        head = painted[s_][0]
        cands = [t for t, par in parent_of.items() if par == s_ and t in painted
                 and abs(painted[t][-1][0] - head[0]) + abs(painted[t][-1][1] - head[1]) == 1]
        if not cands:
            continue
        t = max(cands, key=lambda k2: path_dis[k2][-1])
        painted[t] = painted[t] + painted[s_]
        path_dis[t] = np.maximum.accumulate(np.concatenate([path_dis[t], path_dis[s_]]))
        parent_of[t] = parent_of.get(s_, -1)
        for q in painted[s_]:
            owner[q] = t
        for k2, par in list(parent_of.items()):
            if par == s_:
                parent_of[k2] = t
        del painted[s_], path_dis[s_]
        parent_of.pop(s_, None)
        merged += 1
    return merged


def _split_sections(owner, painted, path_dis, parent_of, mouths):
    """Split every river into its visible sections: wherever its path jumps
    (it left the frame and came back -- the Yellow River on a Russia map --
    crossed the sea, or has a gap no detour could avoid), each section
    becomes a river of its own, with its own source and end. Only the last
    section keeps the river's end: its confluence (so it alone joins the
    parent) and its mouth. Tributaries follow the section they touch. In
    place; returns the number of sections added."""
    h, w = owner.shape
    D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
    next_id = max(painted) + 1 if painted else 0
    added = 0
    new_of = {}
    for k in list(painted):
        path = painted[k]
        if len(path) < 2:
            continue
        a = np.array(path)
        jump = np.flatnonzero(np.abs(np.diff(a, axis=0)).max(axis=1) > 1)
        if not len(jump):
            continue
        cuts = [0] + (jump + 1).tolist() + [len(path)]
        dis = path_dis[k]
        ids = []
        for t, (a0, a1) in enumerate(zip(cuts[:-1], cuts[1:])):
            kk = k if t == len(cuts) - 2 else next_id   # the last section keeps the id (end, parent)
            if kk != k:
                next_id += 1
                added += 1
                parent_of[kk] = -1
            painted[kk] = path[a0:a1]
            path_dis[kk] = dis[a0:a1]
            for q in painted[kk]:
                owner[q] = kk
            ids.append(kk)
        new_of[k] = ids
    if new_of:
        for t, par in list(parent_of.items()):
            if par in new_of and t in painted:
                lr, lc = painted[t][-1]
                hit = [kk for kk in new_of[par] if any(0 <= lr + a2 < h and 0 <= lc + b2 < w
                                                       and owner[lr + a2, lc + b2] == kk for a2, b2 in D4)]
                if hit:
                    parent_of[t] = hit[0]
    return added


def _river_pixels(pts, seg_start, h: int, w: int):
    """4-connected pixel path of every polyline: pixel-space points `pts`
    (N, 2), polyline i = pts[seg_start[i]:seg_start[i+1]]. Each edge is
    sampled every _RIVER_SAMPLE_PX, consecutive repeats dropped, and where
    two consecutive pixels touch only by a corner, the side pixel the line
    actually crosses is inserted -- so every path is side-connected.
    Returns (rows, cols, polyline index, point index -- the edge's start
    vertex) of all path pixels, in path order (may lie outside the image)."""
    n_seg = len(seg_start) - 1
    seg_of_pt = np.repeat(np.arange(n_seg), np.diff(seg_start))
    last = np.zeros(len(pts), dtype=bool)
    last[seg_start[1:] - 1] = True
    e = np.flatnonzero(~last)  # edge e runs from point e to e + 1
    d = pts[e + 1] - pts[e]
    steps = np.maximum(np.ceil(np.hypot(d[:, 0], d[:, 1]) / _RIVER_SAMPLE_PX).astype(np.int64), 1)
    ei = np.repeat(np.arange(len(e)), steps)
    k = np.arange(len(ei)) - np.repeat(np.cumsum(steps) - steps, steps)
    t = (k / steps[ei])[:, None]
    samples = np.concatenate([pts[e[ei]] + d[ei] * t, pts[last]])
    s_seg = np.concatenate([seg_of_pt[e[ei]], seg_of_pt[last]])
    s_pt = np.concatenate([e[ei], np.flatnonzero(last)])
    # Order: by point index, the samples of an edge before the next point.
    key = np.concatenate([e[ei] * 2.0 + t[:, 0], np.flatnonzero(last) * 2.0])
    order = np.argsort(key, kind="stable")
    samples, s_seg, s_pt = samples[order], s_seg[order], s_pt[order]
    px = np.floor(samples).astype(np.int64)
    new = np.ones(len(px), dtype=bool)
    new[1:] = (s_seg[1:] != s_seg[:-1]) | (px[1:] != px[:-1]).any(axis=1)
    kept = np.flatnonzero(new)
    cols, rows = px[kept, 0], px[kept, 1]
    # Corner-only steps within a polyline: insert the crossed side pixel.
    j = np.flatnonzero((s_seg[kept[1:]] == s_seg[kept[:-1]])
                       & (np.abs(cols[1:] - cols[:-1]) == 1) & (np.abs(rows[1:] - rows[:-1]) == 1))
    a, b = samples[kept[j + 1] - 1], samples[kept[j + 1]]  # last sample in pixel j, first in pixel j+1
    gx = np.maximum(cols[j], cols[j + 1]).astype(np.float64)  # the vertical grid line crossed
    tt = np.clip((gx - a[:, 0]) / np.where(b[:, 0] != a[:, 0], b[:, 0] - a[:, 0], 1e-12), 0.0, 1.0)
    y_at = a[:, 1] + tt * (b[:, 1] - a[:, 1])
    x_first = np.floor(y_at).astype(np.int64) == rows[j]  # crosses x first: the side pixel is (next col, this row)
    corner_c = np.where(x_first, cols[j + 1], cols[j])
    corner_r = np.where(x_first, rows[j], rows[j + 1])
    order = np.argsort(np.concatenate([np.arange(len(rows)) * 2, j * 2 + 1]), kind="stable")
    return (np.concatenate([rows, corner_r])[order], np.concatenate([cols, corner_c])[order],
            np.concatenate([s_seg[kept], s_seg[kept[j]]])[order], np.concatenate([s_pt[kept], s_pt[kept[j]]])[order])


def _line_4(r0: int, c0: int, r1: int, c1: int):
    """Ordered 4-connected pixel path from (r0, c0) to (r1, c1) along the
    straight line between the pixel centres (at a corner-only step, the side
    pixel the line crosses first)."""
    n = max(1, int(np.ceil(4 * max(abs(r1 - r0), abs(c1 - c0)))))
    path = [(r0, c0)]
    for k in range(1, n + 1):
        t = k / n
        rr = int(np.floor(r0 + 0.5 + (r1 - r0) * t))
        cc = int(np.floor(c0 + 0.5 + (c1 - c0) * t))
        pr, pc = path[-1]
        if (rr, cc) == (pr, pc):
            continue
        if rr != pr and cc != pc:
            # Crossing the vertical grid line first -> the side pixel is (pr, cc).
            gx = max(pc, cc)
            tx = (gx - (c0 + 0.5)) / (c1 - c0) if c1 != c0 else 1.0
            y_at = r0 + 0.5 + (r1 - r0) * tx
            path.append((pr, cc) if int(np.floor(y_at)) == pr else (rr, pc))
        path.append((rr, cc))
    return path


def _auto_river_count(index, c_cand, area_px_per_km2, spacing_px: float):
    """How many of one continent's in-view segments (highest discharge
    first) to draw so its rivers are about `spacing_px` apart on average:
    river length (px) >= land area (px^2) / spacing. The land area is the
    sum of the segments' own catchments (HydroRIVERS catchments tile the
    land) and each segment's length is LENGTH_KM, both through the local
    projection scale (length by the square root of the area factor). In
    pixels, so the same spacing thins a world map far more than a Europe
    map, and a river-dense continent (South America) more than a sparse
    one. Returns (count, land px, achieved spacing px)."""
    land_px = float((index["catch_km2"][c_cand] * area_px_per_km2).sum())
    len_px = index["length_km"][c_cand] * np.sqrt(area_px_per_km2)
    order = np.argsort(-index["discharge"][c_cand], kind="stable")
    cum = np.cumsum(len_px[order])
    k = int(min(np.searchsorted(cum, land_px / spacing_px) + 1, len(c_cand)))
    return k, round(land_px), round(land_px / max(float(cum[k - 1]), 1e-9), 1)






def _path_discharges(painted, rr, cc, sidx, dis_of_px):
    """Discharge along each painted path: each pixel takes the discharge of
    the segment it was drawn from (pixels added by detours / joins / mouth
    extensions take their neighbour's), never decreasing downstream."""
    starts = np.flatnonzero(np.r_[True, sidx[1:] != sidx[:-1]])
    ends = np.r_[starts[1:], len(sidx)]
    raster = {}
    for s0, s1 in zip(starts.tolist(), ends.tolist()):
        k = int(sidx[s0])
        if k in painted:
            raster[k] = dict(zip(zip(rr[s0:s1].tolist(), cc[s0:s1].tolist()), dis_of_px[s0:s1].tolist()))
    path_dis = {}
    for k, path in painted.items():
        vals = np.array([raster.get(k, {}).get(q, np.nan) for q in path], dtype=np.float64)
        ok = ~np.isnan(vals)
        if not ok.any():
            vals[:] = 1.0
        else:
            ix = np.where(ok, np.arange(len(vals)), 0)
            np.maximum.accumulate(ix, out=ix)
            vals = vals[ix]
            first = int(np.argmax(ok))
            vals[:first] = vals[first]
        path_dis[k] = np.maximum.accumulate(np.maximum(vals, 1e-3))
    return path_dis


def _split_big_networks(out, owner, water=None, max_branches=RIVER_MAX_BRANCHES_DEFAULT,
                        max_px=RIVER_MAX_NETWORK_PX):
    """Cut the drawn rivers so no 4-connected network is bigger than the game
    can load quickly (RIVER_MAX_BRANCHES / RIVER_MAX_NETWORK_PX), with as few
    cuts as possible.

    The river pixels form a tree, so it is contracted to a skeleton -- nodes
    are the branch pixels (degree >= 3) and the ends (degree <= 1), edges the
    stretches of degree-2 pixels between them -- and partitioned by the
    classic greedy: post-order from the source, a child's subtree is kept
    with its parent while both limits still hold, and the edge is cut when
    they don't. That is the smallest number of cuts for one additive weight;
    with two it cuts as soon as either would be passed (measured on the
    Yugoslavia map: 43 cuts, 0.02% of the river pixels, against 2,943 for
    cutting every river at a fixed length).

    A cut deletes one pixel of a stretch, taken **beside the junction** --
    the tributary's last pixel before it meets its parent -- so the gap is
    hidden at the confluence and the junction loses a branch with it. Called
    before the one-source-per-network rule, which then re-sources every piece
    the cuts make (see `_main_stem_source`).

    Returns (cuts, [(y, x) of every deleted pixel]). The pixel list is what
    `report["network_cut_px"]` carries on to the Provinces tab: the gap is
    there for the game's sake, but the river is still a river on the ground,
    so a province must go on being split along it (user).
    """
    h, w = owner.shape
    river = owner >= 0
    if not river.any() or not max_branches:   # 0 = the user turned the limit off
        return 0, []

    def degrees(mask):
        p = np.pad(mask, 1).view(np.uint8)
        return (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) * mask

    deg = degrees(river)
    lab, n_net = ndimage.label(river, structure=_CONNECTIVITY_4)
    # nothing to do when every network already fits
    sizes = np.bincount(lab.ravel(), minlength=n_net + 1)
    branches = np.bincount(lab[river & (deg >= 3)].ravel(), minlength=n_net + 1)
    if not ((branches[1:] > max_branches) | (sizes[1:] > max_px)).any():
        return 0, []

    nid = {}
    for y, x in zip(*np.nonzero(river & (deg != 2))):
        nid[(int(y), int(x))] = len(nid)
    edges = []            # (node, node, [the degree-2 pixels between them])
    walked = set()
    for (y, x), i in list(nid.items()):
        for dy, dx in _D4:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w and river[ny, nx]) or ((y, x), (ny, nx)) in walked:
                continue
            path = []
            py, pxx, cy, cx = y, x, ny, nx
            while (cy, cx) not in nid:
                path.append((cy, cx))
                nxt = None
                for ey, ex in _D4:
                    ty, tx = cy + ey, cx + ex
                    if (ty, tx) != (py, pxx) and 0 <= ty < h and 0 <= tx < w and river[ty, tx]:
                        nxt = (ty, tx)
                        break
                if nxt is None:
                    break
                py, pxx, cy, cx = cy, cx, *nxt
            walked.add(((y, x), (ny, nx)))
            if (cy, cx) in nid:
                walked.add(((cy, cx), path[-1] if path else (y, x)))
                if nid[(cy, cx)] >= i:        # each stretch once
                    edges.append((i, nid[(cy, cx)], path))

    adj = {}
    for ei, (u, v, _path) in enumerate(edges):
        adj.setdefault(u, []).append((v, ei))
        adj.setdefault(v, []).append((u, ei))
    pos = {i: yx for yx, i in nid.items()}
    is_branch = {i: int(deg[pos[i]] >= 3) for i in nid.values()}
    by_net = {}
    for yx, i in nid.items():
        by_net.setdefault(int(lab[yx]), []).append(i)

    cuts, cut_px = 0, []
    for net, members in by_net.items():
        if branches[net] <= max_branches and sizes[net] <= max_px:
            continue
        root = next((i for i in members if out[pos[i]] == RIVER_SOURCE), members[0])
        parent = {root: (None, None)}
        order = []
        stack = [root]
        while stack:                          # depth first, iterative
            u = stack.pop()
            order.append(u)
            for v, ei in adj.get(u, []):
                if v not in parent:
                    parent[v] = (u, ei)
                    stack.append(v)
        nb = {u: is_branch[u] for u in order}
        npx = {u: 1 for u in order}
        for u in reversed(order):             # children before their parent
            par, ei = parent[u]
            if par is None:
                continue
            path = edges[ei][2]
            if nb[par] + nb[u] > max_branches or npx[par] + npx[u] + len(path) > max_px:
                if not path:                  # two junctions side by side: nothing to take out
                    continue
                # the pixel beside the junction -- the tributary's last one,
                # the red flow-in marker itself -- so the gap hides at the
                # confluence, nothing is left pointing at a river that has
                # gone, and the junction loses a branch with it
                y, x = path[0] if is_branch[edges[ei][0]] else path[-1]
                out[y, x] = RIVER_WATER if water is not None and water[y, x] else RIVER_LAND
                owner[y, x] = -1
                cuts += 1
                cut_px.append((int(y), int(x)))
            else:
                nb[par] += nb[u]
                npx[par] += npx[u] + len(path)
    return cuts, cut_px


def _main_stem_source(out, owner, painted, lab, comp, dis_img, pixels):
    """Where a river network's green source belongs: the first pixel of its
    **main stem**, plus the flow-in markers that have to go with it.

    A piece a cut split off keeps no source, and which pixel deserves one
    cannot be read off the picture: the water comes in at a pixel that looks
    like any other. The drawn strokes know -- they run head -> mouth, and at
    each confluence the branch with the most discharge is the one that
    continues the river (`_selection_strokes`) -- so the piece's main river
    is the stroke owning its biggest discharge, and its start is that
    stroke's head. (Walking the picture by discharge instead cannot tell
    upstream from downstream where a stroke's discharge is flat, which is
    most of it: on the user's France map it walked *down* a 3,887 px piece
    and cleared two flow-ins on the way.)

    Where the cut took the stem above that head, the stem simply stops inside
    the piece, and the biggest branch still feeding it *is* the river here:
    the walk carries on into that branch (and so on), and the branch's red
    flow-in marker is reported back to be painted as plain river -- it no
    longer flows into anything, there is nothing left above it. A marker left
    there would say a tributary flows into a river with no source, which is
    what broke the cut-off half of the Seine on the user's France map (its
    green had gone to a tributary's head 200 px away, that tributary's own
    flow-in still in place).

    Returns (y, x, [flow-in pixels to clear]).
    """
    h, w = out.shape
    y, x = max(pixels, key=lambda q: dis_img[q])   # the piece's outlet: its biggest discharge
    reds = []
    walked = set()
    while True:
        k = int(owner[y, x])
        path = painted.get(k)
        j = None if path is None else next((i for i, q in enumerate(path) if tuple(q) == (y, x)), None)
        if j is None or k in walked:                   # nothing to follow: the river starts here
            return int(y), int(x), reds
        walked.add(k)
        while j > 0:                                   # up the stroke, as far as the piece goes
            py, pxx = (int(v) for v in path[j - 1])
            if owner[py, pxx] < 0 or lab[py, pxx] != comp:
                break                                  # the cut took the rest of it
            j -= 1
            y, x = py, pxx
        if j == 0:                                     # the stroke's own head
            return int(y), int(x), reds
        feed = max(((dis_img[y + a, x + b], y + a, x + b) for a, b in _D4
                    if 0 <= y + a < h and 0 <= x + b < w and out[y + a, x + b] == RIVER_FLOW_IN
                    and lab[y + a, x + b] == comp), default=None)
        if feed is None:                               # nothing feeds the stem's first pixel
            return int(y), int(x), reds
        y, x = int(feed[1]), int(feed[2])              # into the biggest branch left, its marker gone
        reds.append((y, x))


def _paint_hoi4_rivers(out, owner, painted, parent_of, path_dis, water=None,
                       max_branches=RIVER_MAX_BRANCHES_DEFAULT):
    """Paint the drawn rivers into `out` (palette indices) the way HOI4's
    rivers.bmp wants them:
      * width 3-11 by discharge (`path_dis`, see _path_discharges), on a log
        scale from the 5th percentile of river pixels to the largest drawn --
        every map uses the whole range;
      * a red flow-in pixel: a tributary's last pixel, beside its parent;
      * a green source pixel: the first pixel of every river that isn't a
        joined tributary (main rivers, and tributaries whose confluence is
        outside the frame);
      * then HOI4's rule -- exactly one source per contiguous river network
        (4-connected, red junctions included) -- is enforced: a network
        without one (a piece split off by a gap) is re-sourced at the start
        of its own main stem (`_main_stem_source`); extra ones are removed
        (the biggest river's is kept).
    Returns report fields."""
    h, w = owner.shape
    if not painted:
        return {"river_px": 0, "width_classes_cms": []}
    # Scale from the 5th percentile of river pixels (a few near-zero rivers
    # would otherwise squeeze everything else into the wide classes) to the
    # biggest river.
    all_dis = np.concatenate(list(path_dis.values()))
    lo = float(np.percentile(all_dis, 5))
    hi = float(all_dis.max())
    span = max(np.log(hi) - np.log(lo), 1e-9)
    n_w = len(RIVER_WIDTHS)
    for k, path in painted.items():
        cls = np.clip(np.floor((np.log(path_dis[k]) - np.log(lo)) / span * n_w).astype(np.int64), 0, n_w - 1)
        pr_, pc_ = np.array(path).T
        out[pr_, pc_] = np.array(RIVER_WIDTHS, dtype=np.uint8)[cls]
    # Markers.
    D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
    joined = set()
    for k, path in painted.items():
        par = int(parent_of.get(k, -1))
        lr, lc = path[-1]
        if par >= 0 and par in painted and any(0 <= lr + a < h and 0 <= lc + b < w and owner[lr + a, lc + b] == par
                                               for a, b in D4):
            out[lr, lc] = RIVER_FLOW_IN
            joined.add(k)
    for k, path in painted.items():
        if k not in joined:
            out[path[0]] = RIVER_SOURCE
    # No network bigger than the game loads quickly (before the source rule,
    # which gives every piece that comes out of it its own source).
    n_cut, cut_px = _split_big_networks(out, owner, water, max_branches)
    # Exactly one source per contiguous network.
    river = owner >= 0
    lab, n = ndimage.label(river, structure=_CONNECTIVITY_4)
    src = out == RIVER_SOURCE
    n_src = np.bincount(lab[src], minlength=n + 1)
    added = removed = cleared = 0
    if (n_src[1:] != 1).any():
        dis_img = np.zeros((h, w), dtype=np.float64)
        for k, path in painted.items():
            pr_, pc_ = np.array(path).T
            dis_img[pr_, pc_] = path_dis[k]

        def width_of(d):                     # the width class a marker pixel would have had
            return RIVER_WIDTHS[int(np.clip(int((np.log(max(float(d), 1e-3)) - np.log(lo)) / span * n_w),
                                            0, n_w - 1))]

        for comp in np.flatnonzero(n_src[1:] != 1) + 1:
            cr, cc_ = np.nonzero(lab == comp)
            if n_src[comp] > 1:              # keep the one on the biggest river
                srcs = [(dis_img[y, x], y, x) for y, x in zip(cr.tolist(), cc_.tolist()) if src[y, x]]
                srcs.sort(reverse=True)
                for _d, y, x in srcs[1:]:
                    out[y, x] = width_of(dis_img[y, x])
                    removed += 1
            else:                            # none: the start of this piece's own main stem
                y, x, reds = _main_stem_source(out, owner, painted, lab, int(comp), dis_img,
                                               zip(cr.tolist(), cc_.tolist()))
                for fy, fx in reds:          # a tributary of a river that is no longer there
                    out[fy, fx] = width_of(dis_img[fy, fx])
                    cleared += 1
                out[y, x] = RIVER_SOURCE
                added += 1
    bounds = [float(np.exp(np.log(lo) + span * i / n_w)) for i in range(n_w + 1)]
    return {"river_px": int(river.sum()), "sources": int((out == RIVER_SOURCE).sum()),
            "flow_ins": int((out == RIVER_FLOW_IN).sum()),
            "sources_added": added, "sources_removed": removed, "flow_ins_cleared": cleared,
            "networks": int(n), "network_cuts": n_cut, "network_cut_px": cut_px,
            "width_classes_cms": [round(v, 2) for v in bounds]}


def render_rivers(image: Image.Image, debug, top_percent=RIVER_TOP_PERCENT_DEFAULT,
                  min_length_px: int = RIVER_MIN_LENGTH_DEFAULT, auto_spacing_px=None,
                  through_lakes: bool = True, reach_sea: bool = True, drop_dead_ends: bool = True,
                  fit_borders: bool = True, max_branches: int = RIVER_MAX_BRANCHES_DEFAULT,
                  heights=None):
    """The Rivers tab's map of a render: land white, sea *and lakes* black,
    rivers blue -- per continent (RIVER_CONTINENTS), the `top_percent` % of
    that continent's HydroRIVERS segments in view with the highest average
    discharge, drawn as 1 px wide, side-connected (4-connected) lines on land
    (and across lakes, see through_lakes), never on the sea; river branches shorter than `min_length_px` are then removed
    (_remove_short_strokes). `top_percent` is one number for every continent
    or a {continent code: percent} dict (missing codes: the default).
    With `auto_spacing_px`, each continent's percentage is instead chosen
    from river density (_auto_river_count) and reported back.
      through_lakes: rivers are drawn across lakes (not the sea), so the
        Nile isn't cut by Lake Nasser nor the Dnieper by its reservoirs.
      reach_sea: an ocean outlet (NEXT_DOWN = 0, not endorheic) whose end
        pixel doesn't touch the sea -- GADM's coastline lies further out
        than HydroSHEDS' -- is continued to the sea, up to
        RIVER_MOUTH_MAX_GAP_PX away, **following the ground**: downhill
        along the lowest way there, round a ridge rather than over it
        (_extend_strokes_to_sea, _elevation_route).
      heights: the frame's raw topography for that routing. Left out, it is
        read for the frame when a mouth needs it (render_heightmap); without
        the NASA topography downloaded at all there is nothing to follow and
        the route falls back to the one with the fewest pixels -- a straight
        line -- which report["mouth_routing"] says.
      drop_dead_ends: rivers ending inland outside any lake (river_dead_ends)
        are left out before the percentages are computed.
      max_branches: the most branch pixels one river network (4-connected,
        its tributaries included) may have before it is cut into several
        (_split_big_networks); 0 = no limit, at the cost of the game taking
        minutes to load the map.
      fit_borders: rivers along a river border (river_borders.json) are
        moved onto the GADM border (_snap_rivers_to_borders), and the
        country map is then fitted to the drawn rivers
        (_fit_borders_to_rivers): each keeps to one side of its border and
        the border follows it where the river rules kept it off.
    Returns (image, report); report["countries_image"] is the country map
    with its borders fitted to the rivers (the input image when off), and
    report["network_cut_px"] the pixels the branch cap deleted -- the
    Provinces tab puts them back, so a province is still split along a river
    where one of those gaps falls (provinces.render_provinces)."""
    if not isinstance(top_percent, dict):
        top_percent = {code: float(top_percent) for code, _n, _r in RIVER_CONTINENTS}
    top_percent = {code: float(top_percent.get(code, RIVER_TOP_PERCENT_BY_CONTINENT.get(code, RIVER_TOP_PERCENT_DEFAULT)))
                   for code, _n, _r in RIVER_CONTINENTS}
    arr = np.asarray(image)
    water = (arr == np.asarray(BACKGROUND_COLOR, dtype=arr.dtype)).all(axis=2)
    if debug.get("hidden_land") is not None:
        water &= ~debug["hidden_land"]
    if debug.get("lakes") is not None:
        water |= debug["lakes"]
    out = np.full(water.shape, RIVER_LAND, dtype=np.uint8)
    out[water] = RIVER_WATER
    lakes_px = debug.get("lakes") if debug.get("lakes") is not None else np.zeros(water.shape, dtype=bool)
    sea = water & ~lakes_px
    allowed = ~sea if through_lakes else ~water  # where river pixels may go
    if debug.get("edge_land") is not None:
        allowed &= ~debug["edge_land"]  # never on the unowned edge strip
    report = {"top_percent": top_percent, "min_length_px": int(min_length_px),
              "continents": [{"code": c, "name": n, "top_percent": top_percent[c], "in_view": 0, "selected": 0}
                             for c, n, _r in RIVER_CONTINENTS]}
    index = load_river_index()
    if index is None:
        report["available"] = False
        return _river_image(out), report
    report["available"] = True

    v = debug["view"]
    h, w, scale = v["height"], v["width"], v["scale"]
    func = PROJECTIONS[v["projection"]]["func"]
    # Coarse cull in lon/lat: the view's lon/lat range from a 0.25-degree
    # grid of projected points, padded by a cell and each segment's extent.
    glon, glat = np.meshgrid(np.arange(-180.0, 180.01, 0.25), np.arange(-89.75, 89.76, 0.25))
    gx, gy = func(glon.ravel(), glat.ravel())
    gpx, gpy = (gx - v["win_xmin"]) * scale, h - (gy - v["win_ymin"]) * scale
    inside = (gpx >= 0) & (gpx <= w) & (gpy >= 0) & (gpy <= h)
    if not inside.any():
        return _river_image(out), dict(report, in_view=0, drawn=0, river_px=0)
    lon_lo, lon_hi = glon.ravel()[inside].min() - 0.5, glon.ravel()[inside].max() + 0.5
    lat_lo, lat_hi = glat.ravel()[inside].min() - 0.5, glat.ravel()[inside].max() + 0.5
    rel = _relative_lon(index["cx"], v["center"])
    cand = np.flatnonzero((rel + index["ext"] >= lon_lo) & (rel - index["ext"] <= lon_hi)
                          & (index["cy"] + index["ext"] >= lat_lo) & (index["cy"] - index["ext"] <= lat_hi))
    # In view = the segment's bbox centre inside the frame.
    x, y = func(rel[cand].astype(np.float64), index["cy"][cand].astype(np.float64))
    cpx, cpy = (x - v["win_xmin"]) * scale, h - (y - v["win_ymin"]) * scale
    cand = cand[(cpx >= 0) & (cpx < w) & (cpy >= 0) & (cpy < h)]
    report["in_view"] = int(len(cand))
    if drop_dead_ends:
        dead = river_dead_ends(index)[cand]
        report["dead_end_segments"] = int(dead.sum())
        cand = cand[~dead]
    if not len(cand):
        return _river_image(out), dict(report, drawn=0, river_px=0)
    region = index["region"][cand]
    # Projection scale at each segment (px^2 per km^2), for the density
    # estimate (_auto_river_count, and the spacing shown in manual mode):
    # every projection here has y depending on latitude only, so the area
    # factor is (dx/dlon) * (dy/dlat).
    rel_c = rel[cand].astype(np.float64)
    lat_c = np.clip(index["cy"][cand].astype(np.float64), -89.0, 89.0)
    dxdlon = np.abs(func(rel_c + 0.5, lat_c)[0] - func(rel_c - 0.5, lat_c)[0])
    dydlat = np.abs(func(rel_c, lat_c + 0.01)[1] - func(rel_c, lat_c - 0.01)[1]) / 0.02
    area_px_per_km2 = dxdlon * dydlat / (_KM_PER_DEG ** 2 * np.cos(np.radians(lat_c))) * scale * scale
    report["auto_spacing_px"] = None if auto_spacing_px is None else float(auto_spacing_px)
    sel = []
    for info, (code, _name, regions) in zip(report["continents"], RIVER_CONTINENTS):
        in_c = np.isin(region, regions)
        c_cand = cand[in_c]
        info["in_view"] = int(len(c_cand))
        if not len(c_cand):
            continue
        dis = index["discharge"][c_cand]
        if auto_spacing_px is not None:
            k, info["land_px"], info["spacing_px"] = _auto_river_count(index, c_cand, area_px_per_km2[in_c],
                                                                     auto_spacing_px)
            top = np.argsort(-dis, kind="stable")[:k]
            sel.append(c_cand[top])
            threshold = float(dis[top].min())
            info["top_percent"] = round(100.0 * k / len(c_cand), 3)
            report["top_percent"][code] = info["top_percent"]
        else:
            pct = top_percent[code]
            threshold = float(np.quantile(dis, 1.0 - pct / 100.0)) if pct < 100 else float(dis.min())
            chosen_c = dis >= threshold
            sel.append(c_cand[chosen_c])
            a = area_px_per_km2[in_c]
            info["land_px"] = round(float((index["catch_km2"][c_cand] * a).sum()))
            river_len = float((index["length_km"][c_cand][chosen_c] * np.sqrt(a[chosen_c])).sum())
            info["spacing_px"] = round(info["land_px"] / max(river_len, 1e-9), 1)
        info.update(threshold_cms=round(threshold, 3), selected=int(len(sel[-1])))
    sel = np.concatenate(sel) if sel else np.zeros(0, dtype=np.int64)
    report["selected"] = int(len(sel))
    # Follow every selected segment downstream to its outlet: discharge
    # sometimes dips downstream (evaporation, delta branches), which would
    # otherwise stop a river mid-course.
    chosen = np.zeros(len(index["next"]), dtype=bool)
    chosen[sel] = True
    frontier = index["next"][sel]
    while True:
        frontier = np.unique(frontier[frontier >= 0])
        frontier = frontier[~chosen[frontier]]
        if not len(frontier):
            break
        chosen[frontier] = True
        frontier = index["next"][frontier]
    sel = np.flatnonzero(chosen)
    report["drawn"] = int(len(sel))

    # Group the selection into rivers (strokes) and order them for drawing:
    # biggest discharge first, but every river after the one it flows into.
    s_root, hops = _selection_strokes(index, sel)
    roots, s_of_seg = np.unique(s_root, return_inverse=True)
    n_str = len(roots)
    str_dis = index["discharge"][roots]
    root_pos = {int(rt): i for i, rt in enumerate(roots.tolist())}
    root_of_seg = dict(zip(sel.tolist(), s_root.tolist()))
    nxt_root = index["next"][roots]
    parent = np.array([root_pos.get(root_of_seg.get(int(nr), -1), -1) if nr >= 0 else -1
                       for nr in nxt_root.tolist()], dtype=np.int64)
    rank_order = []
    emitted = np.zeros(n_str, dtype=bool)
    for k0 in np.argsort(-str_dis, kind="stable").tolist():
        chain = []
        k = k0
        while k >= 0 and not emitted[k] and k not in chain:
            chain.append(k)
            k = int(parent[k])
        for k in reversed(chain):
            emitted[k] = True
            rank_order.append(k)
    draw_rank = np.empty(n_str, dtype=np.int64)
    draw_rank[np.array(rank_order, dtype=np.int64)] = np.arange(n_str)
    # Segments in drawing order, each river from its source downstream.
    seg_order = np.lexsort((-hops, draw_rank[s_of_seg]))
    sel = sel[seg_order]
    s_of_seg = draw_rank[s_of_seg[seg_order]]  # river index = drawing rank from here on
    parent = np.where(parent >= 0, draw_rank[np.maximum(parent, 0)], -1)[np.argsort(draw_rank)]
    is_main = (nxt_root < 0)[np.argsort(draw_rank)]  # reaches the sea / its lake: a main river
    str_dis = str_dis[np.argsort(draw_rank)]
    is_mouth = ((nxt_root < 0) & (index["endorheic"][roots] == 0))[np.argsort(draw_rank)]

    # Gather the selected segments' points straight from the memory map.
    lonlat, n_pts = _river_points(index, sel)
    rb = river_borders(load_geometries(progress=lambda *a: None), index) if fit_borders else None
    lonlat, n_pts, str_of_pt = _snap_rivers_to_borders(load_geometries(progress=lambda *a: None), rb, sel,
                                                       lonlat, n_pts)
    seg_start = np.concatenate([[0], np.cumsum(n_pts)])
    # Segments span < 3.3 degrees: relative longitude from each one's own
    # first vertex keeps it continuous across the seam.
    first = np.repeat(lonlat[seg_start[:-1], 0], n_pts)
    rel_pts = _relative_lon(first, v["center"]) + ((lonlat[:, 0] - first + 180.0) % 360.0 - 180.0)
    px, py = func(rel_pts, lonlat[:, 1])
    pts = np.column_stack([(px - v["win_xmin"]) * scale, h - (py - v["win_ymin"]) * scale])
    # One polyline per river: its segments' points, source to mouth.
    str_start = np.concatenate([[0], np.cumsum(np.bincount(s_of_seg, weights=n_pts, minlength=n_str).astype(np.int64))])
    rr, cc, sidx, pt_of_px = _river_pixels(pts, str_start, h, w)
    px_str = np.full((h, w), -1, dtype=np.int32)  # river border stretch each pixel was drawn along
    s_px = str_of_pt[pt_of_px]
    m = (s_px >= 0) & (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    px_str[rr[m], cc[m]] = s_px[m]
    owner, painted, draw_stats = _draw_strokes(rr, cc, sidx, parent, allowed, h, w)
    dis_of_px = index["discharge"][sel][np.repeat(np.arange(len(sel)), n_pts)[pt_of_px]]
    report.update({f"{k}_rivers": v_ for k, v_ in draw_stats.items()})
    report["rivers"] = int(len(painted))
    if reach_sea:
        to_extend = [k for k in range(n_str) if is_mouth[k]]
        # the heights only where a mouth may need them (a frame's worth of
        # topography costs a few seconds, a cache hit costs nothing)
        if heights is None and to_extend and elevation_available():
            heights, zrep = elevation_m(debug)
            report["mouth_elevation"] = {k: zrep.get(k) for k in ("source", "gmted_arcsec", "gmted_share")}
        report["mouth_routing"] = "elevation" if heights is not None else "shortest"
        (report["extended_mouths"], report["extended_px"],
         report["mouths_too_far_round"]) = _extend_strokes_to_sea(
            owner, painted, to_extend, sea, lakes_px, allowed, heights)
    if through_lakes and lakes_px.any():
        report["lake_rerouted"], report["lake_pulled_px"] = _pull_rivers_into_lakes(owner, painted, parent,
                                                                                   lakes_px, allowed)
    path_dis = _path_discharges(painted, rr, cc, sidx, dis_of_px)
    parent_of = {k: int(parent[k]) for k in painted}
    report["sections_added"] = _split_sections(owner, painted, path_dis, parent_of,
                                               {k for k in painted if is_mouth[k]})
    report["short_removed"], report["short_removed_px"] = _remove_short_strokes(owner, painted, parent_of,
                                                                                path_dis, min_length_px)
    report["head_merges"] = _merge_head_tributaries(owner, painted, path_dis, parent_of)
    report["straight_removed"], report["straight_removed_tributaries"], report["straight_removed_px"] =         _remove_straight_rivers(owner, painted, path_dis, parent_of)
    report["self_repaired"] = _repair_self_contacts(owner, painted, path_dis, parent_of, allowed)
    report.update(_paint_hoi4_rivers(out, owner, painted, parent_of, path_dis, water, max_branches))
    countries = image
    if rb is not None:
        report["snapped_points"] = int((str_of_pt >= 0).sum())
        canvas = arr.copy()
        report["border_fit"] = _fit_borders_to_rivers(canvas, painted, px_str, rb, debug["country_rgb"])
        countries = Image.fromarray(canvas, mode="RGB")
    report["countries_image"] = countries
    return _river_image(out), report


_CONNECTIVITY_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


# --------------------------------------------------------------------------
# Heightmap: the NASA topography reprojected onto a render's frame
# --------------------------------------------------------------------------

TOPO_TILE_PX = 10800          # per tile, at full resolution
TOPO_TILE_DEG = 90.0
TOPO_MEMO_BYTES = 700_000_000  # decoded tiles kept in memory (LRU)
_topo_lock = threading.Lock()
_topo_memo = OrderedDict()  # (col, row, reduce) -> uint8 array


def _topo_tile_path(col: int, row: int) -> Path:
    return TOPOGRAPHY_DIR / f"gebco_08_rev_elev_{'ABCD'[col]}{row + 1}_grey_geo.jpg"


def topography_available() -> bool:
    return all(_topo_tile_path(c, r).exists() for c in range(4) for r in range(2))


def _topo_tile(col: int, row: int, reduce: int) -> np.ndarray:
    """Tile (col 0-3 = A-D, row 0-1) decoded at 1/`reduce` resolution (JPEG
    draft mode decodes 1/2, 1/4, 1/8 directly -- a full tile is ~0.4 s,
    117 MB), memoized with an LRU byte budget."""
    key = (col, row, reduce)
    with _topo_lock:
        if key in _topo_memo:
            _topo_memo.move_to_end(key)
            return _topo_memo[key]
    Image.MAX_IMAGE_PIXELS = None  # 116.6 MP per tile trips PIL's bomb check
    n = TOPO_TILE_PX // reduce
    im = Image.open(_topo_tile_path(col, row))
    im.draft("L", (n, n))
    arr = np.asarray(im.convert("L"))
    if arr.shape != (n, n):
        arr = np.asarray(Image.fromarray(arr).resize((n, n), Image.BILINEAR))
    with _topo_lock:
        _topo_memo[key] = arr
        while sum(a.nbytes for a in _topo_memo.values()) > TOPO_MEMO_BYTES and len(_topo_memo) > 1:
            _topo_memo.popitem(last=False)
    return arr


_inverse_memo = {}


def _inverse_lat_table(projection_id: str):
    """(y, lat) with y strictly increasing, for inverting a projection's y(lat)
    by interpolation -- every projection here has y depending on latitude
    only (and x linear in relative longitude, see _unproject)."""
    if projection_id not in _inverse_memo:
        lat = np.linspace(-90.0, 90.0, 180_001)
        _x, y = PROJECTIONS[projection_id]["func"](np.zeros_like(lat), lat)
        y_u, idx = np.unique(y, return_index=True)  # drops Mercator's clipped poles
        _inverse_memo[projection_id] = (y_u, lat[idx])
    return _inverse_memo[projection_id]


def _unproject(projection_id: str, center: float, x, y):
    """Projected (x, y) -> (lon, lat, inside the world outline). x is
    rel_lon * g(lat) for all five projections (g = x per degree at that
    latitude), so relative longitude = x / g(lat)."""
    y_tab, lat_tab = _inverse_lat_table(projection_id)
    inside = (y >= y_tab[0]) & (y <= y_tab[-1])
    lat = np.interp(y, y_tab, lat_tab)
    g = PROJECTIONS[projection_id]["func"](np.ones_like(lat), lat)[0]
    rel = x / np.where(np.abs(g) > 1e-12, g, 1e-12)
    inside &= np.abs(rel) <= 180.0
    lon = (center + rel + 180.0) % 360.0 - 180.0
    return lon, lat, inside


def _view_sample(v, n: int = 32):
    """(degrees per pixel at the view's finest, lon, lat over a coarse grid
    of it): how fine the frame is -- from a sample of its pixels and their
    neighbours one pixel right / down -- and which ground it covers, for
    picking a source resolution (see render_heightmap, elevation_m)."""
    h, w, scale = v["height"], v["width"], v["scale"]
    cols_x = v["win_xmin"] + (np.arange(w) + 0.5) / scale
    rows_y = v["win_ymin"] + (h - (np.arange(h) + 0.5)) / scale
    gy, gx = np.meshgrid(rows_y[:: max(h // n, 1)], cols_x[:: max(w // n, 1)], indexing="ij")
    lon0, lat0, ok0 = _unproject(v["projection"], v["center"], gx, gy)
    lon1, _l, ok1 = _unproject(v["projection"], v["center"], gx + 1.0 / scale, gy)
    _l, lat2, ok2 = _unproject(v["projection"], v["center"], gx, gy - 1.0 / scale)
    ok = ok0 & ok1 & ok2
    step = np.minimum(np.abs((lon1 - lon0 + 180.0) % 360.0 - 180.0), np.abs(lat2 - lat0))[ok]
    return (float(step.min()) if len(step) else 1.0), lon0[ok0], lat0[ok0]


def _view_lonlat_chunks(v, chunk_px: int = 2_000_000):
    """The frame's pixel centres un-projected to lon / lat, a band of rows at
    a time: (first row, last row + 1, lon, lat, inside the world outline)."""
    h, w, scale = v["height"], v["width"], v["scale"]
    cols_x = v["win_xmin"] + (np.arange(w) + 0.5) / scale
    rows_y = v["win_ymin"] + (h - (np.arange(h) + 0.5)) / scale
    rows_per = max(1, chunk_px // max(w, 1))
    for r0 in range(0, h, rows_per):
        r1 = min(h, r0 + rows_per)
        yy, xx = np.meshgrid(rows_y[r0:r1], cols_x, indexing="ij")
        lon, lat, inside = _unproject(v["projection"], v["center"], xx, yy)
        yield r0, r1, lon, lat, inside


def render_heightmap(debug, chunk_px: int = 2_000_000):
    """The topography (8-bit grey, as NASA's tiles: 0 = sea level or below,
    255 = highest peaks) reprojected onto the render's frame (debug["view"]):
    every output pixel centre is un-projected to lon/lat and the tiles are
    sampled bilinearly, decoded at the coarsest 1/1, 1/2, 1/4 or 1/8
    resolution still at least as fine as the view's finest pixel. Pixels
    outside the projection's world outline are 0. Returns (image, report).

    This is the 8-bit source, ~26.5 m per step: `elevation_m` prefers
    GMTED's metres and falls back to this."""
    v = debug["view"]
    h, w = v["height"], v["width"]
    if not topography_available():
        return Image.new("L", (w, h), 0), {"available": False}
    t0 = time.time()
    deg_px, _slon, _slat = _view_sample(v)
    src_deg = TOPO_TILE_DEG / TOPO_TILE_PX
    reduce = 1
    while reduce < 8 and src_deg * reduce * 2 <= deg_px:
        reduce *= 2
    n = TOPO_TILE_PX // reduce
    d = TOPO_TILE_DEG / n  # degrees per source pixel
    n_cols, n_rows = 4 * n, 2 * n
    out = np.zeros((h, w), dtype=np.uint8)
    used = set()
    for r0, r1, lon, lat, inside in _view_lonlat_chunks(v, chunk_px):
        u = (lon + 180.0) / d - 0.5  # source column (pixel centres at integers)
        q = (90.0 - lat) / d - 0.5
        j0 = np.floor(u).astype(np.int64)
        i0 = np.floor(q).astype(np.int64)
        fu, fq = u - j0, q - i0
        acc = np.zeros(u.shape, dtype=np.float64)
        for di, dj, wgt in ((0, 0, (1 - fq) * (1 - fu)), (0, 1, (1 - fq) * fu),
                            (1, 0, fq * (1 - fu)), (1, 1, fq * fu)):
            ii = np.clip(i0 + di, 0, n_rows - 1)
            jj = (j0 + dj) % n_cols
            tile = (ii // n) * 4 + jj // n
            vals = np.zeros(u.shape, dtype=np.float64)
            for t in np.unique(tile[inside]).tolist():
                m = (tile == t) & inside
                arr = _topo_tile(t % 4, t // 4, reduce)
                vals[m] = arr[ii[m] - (t // 4) * n, jj[m] - (t % 4) * n]
                used.add(t)
            acc += wgt * vals
        out[r0:r1] = np.where(inside, np.clip(np.rint(acc), 0, 255), 0).astype(np.uint8)
    return Image.fromarray(out, mode="L"), {
        "available": True, "reduce": reduce, "source_arcsec": round(d * 3600.0, 1),
        "view_deg_per_px": round(deg_px, 5),
        "tiles": sorted("ABCD"[t % 4] + str(t // 4 + 1) for t in used), "seconds": round(time.time() - t0, 2)}


def elevation_available() -> bool:
    """Whether any elevation source is on disk (see elevation_m)."""
    import gmted
    return gmted.available() or topography_available()


def elevation_missing_message() -> str:
    return ("No elevations are downloaded: get GMTED2010 land elevations (metres, the tiles your map "
            "covers) or the NASA Topographic Heightmap from the Download tab.")


def elevation_m(debug, chunk_px: int = 2_000_000):
    """**The frame's land elevations in metres** -- the one source every step
    that needs real heights reads: the sea level rise (`drown_land`), the
    river mouths (`_elevation_route`), the HOI4 heightmap's levels
    (`hoi4_heightmap`) and the Terrain tab's relief.

    GMTED2010 (16-bit metres, 15 or 7.5 arcsec, `gmted.py`) wherever its
    tiles are on disk, sampled on the finest grid the view's own tiles offer;
    the NASA 8-bit topography x NASA_M_PER_UNIT (~26.5 m per step) for the
    rest of the frame, so a view outside GMTED's coverage (Antarctica, below
    56S) or with tiles not downloaded still works, just coarsely. Pixels no
    source covers are 0 m and counted in the report.

    Returns (float32 metres, report) -- (None, report) when neither source is
    there."""
    import gmted
    v = debug["view"]
    h, w = v["height"], v["width"]
    t0 = time.time()
    deg_px, slon, slat = _view_sample(v)
    arcsec, want = gmted.grid_arcsec(slon, slat, deg_px) if gmted.available() else (None, [])
    out = np.zeros((h, w), dtype=np.float32)
    have = np.zeros((h, w), dtype=bool)
    if arcsec is not None:
        for r0, r1, lon, lat, inside in _view_lonlat_chunks(v, chunk_px):
            z, ok = gmted.heights_m(lon, lat, arcsec=arcsec)
            if z is None:
                break
            ok &= inside
            out[r0:r1] = np.where(ok, z, 0.0)
            have[r0:r1] = ok
    report = {"available": True, "source": "gmted" if have.all() else "gmted+nasa" if have.any() else "nasa",
              "gmted_arcsec": arcsec, "gmted_share": round(float(have.mean()), 4),
              "gmted_tiles": [gmted.tile_name(*k) for k in want],
              "view_deg_per_px": round(deg_px, 5), "view_arcsec_per_px": round(deg_px * 3600.0, 2)}
    if gmted.problems():
        report["gmted_unreadable"] = sorted(f"{gmted.tile_name(*k)}: {why}"
                                            for k, why in gmted.problems().items())
    if not have.all():
        if topography_available():
            grey, grep = render_heightmap(debug, chunk_px)
            out = np.where(have, out, np.asarray(grey, dtype=np.float32) * NASA_M_PER_UNIT)
            report["nasa_m_per_step"] = NASA_M_PER_UNIT
            report["nasa_reduce"] = grep.get("reduce")
        elif not have.any():
            return None, {"available": False, "source": None}
        else:
            report["uncovered_px"] = int((~have).sum())   # no second source: those pixels stay 0 m
    report["seconds"] = round(time.time() - t0, 2)
    return out, report


def heightmap_overlay(height: Image.Image, countries: Image.Image) -> Image.Image:
    """The heightmap in colour with the country map's coastlines and borders
    (pixels whose right or lower neighbour has another colour) drawn over it
    in red -- to check that the two line up."""
    c = np.asarray(countries.convert("RGB"))
    code = (c[:, :, 0].astype(np.int32) << 16) | (c[:, :, 1].astype(np.int32) << 8) | c[:, :, 2]
    edge = np.zeros(code.shape, dtype=bool)
    edge[:, :-1] |= code[:, :-1] != code[:, 1:]
    edge[:-1, :] |= code[:-1, :] != code[1:, :]
    out = np.repeat(np.asarray(height.convert("L"))[:, :, None], 3, axis=2).copy()
    out[edge] = (255, 0, 0)
    return Image.fromarray(out, mode="RGB")


# --------------------------------------------------------------------------
# HOI4 heightmap.bmp
# --------------------------------------------------------------------------

# Levels of the game's heightmap.bmp (measured on the vanilla 5632 x 2048
# file): all water -- sea and lakes -- is a flat 89; the land pixel on the
# coast ~94, rising with the terrain behind it; land elsewhere >= ~96, up to
# 239 in the Himalaya.
HOI4_WATER_LEVEL = 89
HOI4_COAST_LEVEL = 94
HOI4_LAND_MIN = 96
# NASA grey (0-255) -> HOI4 height: quantile matching of inland land (>= 8
# px from water) on a Miller world render against the vanilla heightmap
# (_dev/hm_lut.py), sampled at these knots and interpolated linearly.
HOI4_HEIGHT_KNOTS = ((0, 96.3), (1, 97.0), (2, 97.5), (5, 99.5), (10, 104.0), (20, 110.5), (30, 117.2),
                     (40, 123.1), (60, 135.0), (80, 144.8), (100, 155.5), (130, 187.0), (160, 204.0),
                     (190, 219.0), (220, 231.1), (255, 239.0))
# Coast ramp: at taxicab distance d from water, height = coast level +
# (terrain - coast level) * weight[d] (1 beyond), as the vanilla map rises
# from its beaches (its d = 1 pixel follows ~8% of the terrain behind it, d
# = 2 about 40%).
HOI4_COAST_RAMP = (0.0, 0.08, 0.4, 0.65, 0.9)


NASA_M_PER_UNIT = 26.5  # NASA grey is ~linear: Mont Blanc 181, Snezka 60, Ben Nevis 48 (saturates at 255)
# Relief exaggeration (exaggerate_relief): scales in km so the look doesn't
# depend on the map's zoom.
RELIEF_BASE_KM = 80.0     # regional base level: land-only Gaussian average at this scale
RELIEF_AMP_KM = 80.0      # local relief amplitude: RMS of (height - base) at this scale
RELIEF_REF_M = 1000.0     # amplitude that is left as is (Alps / Himalaya / Zagros are above)
RELIEF_FLAT_M = 60.0      # below this amplitude (plains) nothing is exaggerated ...
RELIEF_FULL_M = 150.0     # ... from this one on the full gain applies (hills and mountains)
RELIEF_EXPONENT = 0.6     # gain = (REF / amplitude) ** exponent, capped by the strength
# The exaggeration may not invent ground outside what is really there: not
# above the top of the HOI4 curve (255 NASA steps, where hoi4_heightmap
# saturates) and not below sea level, which is what the old 0-255 grey clip
# did and what the Terrain tab's relief tiers were calibrated against. Real
# ground outside that range keeps its height, now that the elevations are
# metres: Everest is 8,849 m, the Dead Sea shore -430 m, and GMTED reads the
# Dutch polders at -14 m.
HEIGHT_CEIL_M = 255.0 * 26.5      # 255 x NASA_M_PER_UNIT, defined just below
RELIEF_STRENGTH_DEFAULT = 4.0  # max gain; 1 = off (user's setting, 2026-09-27; was 2)
RELIEF_STRENGTH_MAX = 4.0

# The Heightmap tab's five relief settings, sent by every tab that builds on
# the heights (Heightmap, Terrain, and the provinces / export requests that
# carry them along). **The cap alone decides nothing**: the gain is
# (ref / amplitude) ** exponent faded out between flat and full, whose own
# maximum over all amplitudes is ~3.2 with the defaults, so the cap only
# binds once the shape is pushed (ref 3000 + exponent 1.0 asks for 20 at an
# amplitude of 150 m). Measured, and why each knob exists, in NOTES.md.
RELIEF_DEFAULTS = {
    "relief": RELIEF_STRENGTH_DEFAULT,   # cap on the gain; 1 = off
    "relief_ref_m": RELIEF_REF_M,        # the amplitude already big enough to leave alone
    "relief_exponent": RELIEF_EXPONENT,  # how fast the gain falls off as a range grows
    "relief_flat_m": RELIEF_FLAT_M,      # below this amplitude nothing is exaggerated ...
    "relief_full_m": RELIEF_FULL_M,      # ... from this one on the full gain applies
}
RELIEF_LIMITS = {"relief": (1.0, RELIEF_STRENGTH_MAX), "relief_ref_m": (50.0, 20000.0),
                 "relief_exponent": (0.0, 3.0), "relief_flat_m": (0.0, 5000.0),
                 "relief_full_m": (1.0, 5000.0)}


def normalize_relief(values=None):
    """The relief settings of a request (strings fine, missing ones take
    their default), checked against RELIEF_LIMITS. `flat` must stay under
    `full`, or the fade between them has no width."""
    out = dict(RELIEF_DEFAULTS)
    for key, default in RELIEF_DEFAULTS.items():
        raw = (values or {}).get(key)
        if raw in (None, ""):
            continue
        try:
            out[key] = float(raw)
        except (TypeError, ValueError):
            raise RenderError(f"{key} must be a number")
        lo, hi = RELIEF_LIMITS[key]
        if not lo <= out[key] <= hi:
            raise RenderError(f"{key} must be between {lo:g} and {hi:g}")
    if out["relief_flat_m"] >= out["relief_full_m"]:
        raise RenderError("relief_flat_m must be under relief_full_m (the fade needs room between them)")
    return out


def _km_per_px(debug) -> float:
    """Ground distance of one pixel at the frame's centre (east-west)."""
    v = debug["view"]
    lon, lat, ok = _unproject(v["projection"], v["center"], np.array([v["win_xmin"] + v["width"] / 2 / v["scale"]]),
                              np.array([v["win_ymin"] + v["height"] / 2 / v["scale"]]))
    lat = float(np.clip(lat[0], -85.0, 85.0))
    dx_deg = float(PROJECTIONS[v["projection"]]["func"](np.array([1.0]), np.array([lat]))[0][0])
    return _KM_PER_DEG * math.cos(math.radians(lat)) / max(dx_deg * v["scale"], 1e-12)


def exaggerate_relief(height: np.ndarray, land: np.ndarray, km_per_px: float,
                      strength: float = RELIEF_STRENGTH_DEFAULT, ref_m: float = RELIEF_REF_M,
                      exponent: float = RELIEF_EXPONENT, flat_m: float = RELIEF_FLAT_M,
                      full_m: float = RELIEF_FULL_M):
    """Exaggerate the relief of lower mountain ranges relative to their
    surroundings, so the Scandes, Sudetes, Appalachians, Scottish Highlands
    or Urals don't look flat next to the Himalaya or the Iranian ranges.
    On land (metres, float): base = land-only Gaussian average at
    RELIEF_BASE_KM; amplitude A = RMS of (height - base) at RELIEF_AMP_KM;
    height' = base + (height - base) * gain, with gain = (ref_m / A) **
    exponent capped to [1, strength] -- ranges already rising ~ref_m above
    their base keep their shape -- and faded out below full_m down to
    nothing at flat_m, so plains stay flat (their noise isn't amplified).
    The four shape settings default to the RELIEF_* constants and come from
    the Heightmap tab (RELIEF_DEFAULTS, normalize_relief). Computed on a reduced grid (~4 cells per
    sigma), then interpolated back. Returns (height', gain)."""
    h = height.astype(np.float64)
    if strength <= 1.0 or not land.any():
        return h, np.ones(h.shape)
    sig_px = RELIEF_BASE_KM / max(km_per_px, 1e-9)
    f = max(1, int(sig_px // 4))  # reduced grid: ~4 cells per sigma
    H, W = h.shape
    ph, pw = (-H) % f, (-W) % f
    hp = np.pad(h * land, ((0, ph), (0, pw)), mode="edge")
    lp = np.pad(land.astype(np.float64), ((0, ph), (0, pw)), mode="edge")
    cells = lambda a: a.reshape(a.shape[0] // f, f, a.shape[1] // f, f).sum(axis=(1, 3))
    hs, ls, h2s = cells(hp), cells(lp), cells(hp * hp)  # per reduced cell: sums of h, land, h^2 (land only)
    sb, sa = RELIEF_BASE_KM / km_per_px / f, RELIEF_AMP_KM / km_per_px / f
    wl = ndimage.gaussian_filter(ls, sb, mode="nearest")
    base_s = ndimage.gaussian_filter(hs, sb, mode="nearest") / np.maximum(wl, 1e-9)
    dev2 = np.maximum(h2s - 2.0 * base_s * hs + base_s ** 2 * ls, 0.0)  # sum over the cell of (h - base)^2
    amp_s = np.sqrt(ndimage.gaussian_filter(dev2, sa, mode="nearest")
                    / np.maximum(ndimage.gaussian_filter(ls, sa, mode="nearest"), 1e-9))
    zoom = (hp.shape[0] / base_s.shape[0], hp.shape[1] / base_s.shape[1])
    base = ndimage.zoom(base_s, zoom, order=1, mode="nearest", grid_mode=True)[:H, :W]
    amp = ndimage.zoom(amp_s, zoom, order=1, mode="nearest", grid_mode=True)[:H, :W]
    gain = np.clip((ref_m / np.maximum(amp, 1e-6)) ** exponent, 1.0, strength)
    t = np.clip((amp - flat_m) / max(full_m - flat_m, 1e-6), 0.0, 1.0)
    gain = 1.0 + (gain - 1.0) * t * t * (3 - 2 * t)
    gain = np.where(land, gain, 1.0)
    out = np.where(land, base + (h - base) * gain, h)
    return np.clip(out, np.minimum(h, 0.0), np.maximum(h, HEIGHT_CEIL_M)), gain


def country_map_water(countries: Image.Image, debug) -> np.ndarray:
    """Water (sea + lakes) of a render's country map: background minus
    hidden countries, plus the lakes -- the land/sea every derived map
    (heightmap, terrain) uses."""
    c = np.asarray(countries.convert("RGB"))
    water = (c == np.asarray(BACKGROUND_COLOR, dtype=c.dtype)).all(axis=2)
    if debug.get("hidden_land") is not None:
        water &= ~debug["hidden_land"]
    lakes = debug.get("lakes")
    if lakes is not None:
        water |= lakes
    water |= (c == np.asarray(LAKE_COLOR, dtype=c.dtype)).all(axis=2)
    return water


def exaggerated_height(height, countries: Image.Image, debug,
                       relief_strength: float = RELIEF_STRENGTH_DEFAULT, relief=None):
    """The frame's elevations (`elevation_m`, **metres**) with lower ranges
    exaggerated (exaggerate_relief), as the Heightmap tab builds its HOI4
    heightmap from it -- also the Terrain tab's relief source. An 8-bit
    image is taken as NASA grey and converted. `relief` = the tab's shape
    settings (RELIEF_DEFAULTS / normalize_relief), `relief_strength` the cap
    among them. Returns (metres, gain, km per px)."""
    water = country_map_water(countries, debug)
    kpp = _km_per_px(debug)
    if isinstance(height, Image.Image):
        height = np.asarray(height.convert("L"), dtype=np.float32) * NASA_M_PER_UNIT
    r = dict(RELIEF_DEFAULTS, **(relief or {}))
    metres, gain = exaggerate_relief(np.asarray(height, dtype=np.float32), ~water, kpp, relief_strength,
                                     r["relief_ref_m"], r["relief_exponent"],
                                     r["relief_flat_m"], r["relief_full_m"])
    return metres, gain, kpp


def hoi4_heightmap(height, countries: Image.Image, debug,
                   relief_strength: float = RELIEF_STRENGTH_DEFAULT, report=None, relief=None):
    """The frame's elevations (`elevation_m`, metres) turned into HOI4's
    heightmap.bmp levels. Land and water come from the country map and its
    Land/Sea steps, not from the elevations -- whose coastline doesn't match
    GADM's (land at 0 along low coasts, sea above 0 near cliffs): water (sea,
    lakes) is HOI4_WATER_LEVEL; land (incl. hidden countries) is the height
    through HOI4_HEIGHT_KNOTS, at least HOI4_LAND_MIN, ramped down to
    HOI4_COAST_LEVEL over the pixels next to the water (HOI4_COAST_RAMP).
    The river map isn't involved (its rivers may cross lakes). Lower
    mountain ranges are first exaggerated (exaggerate_relief, max gain
    `relief_strength`; 1 = off). Returns an 8-bit grey image; `report`
    (dict) gets the relief stats."""
    water = country_map_water(countries, debug)
    kx, ky = zip(*HOI4_HEIGHT_KNOTS)
    kx = np.asarray(kx, dtype=np.float64) * NASA_M_PER_UNIT   # the knots were measured in NASA steps
    grey, gain, kpp = exaggerated_height(height, countries, debug, relief_strength, relief)
    if report is not None:
        g = gain[~water] if (~water).any() else np.ones(1)
        report.update({k: float(v) for k, v in dict(RELIEF_DEFAULTS, **(relief or {})).items()})
        report.update(relief_strength=float(relief_strength), km_per_px=round(kpp, 3),
                      relief_gain_p50=round(float(np.median(g)), 2), relief_gain_max=round(float(g.max()), 2),
                      relief_boosted_share=round(float((g > 1.05).mean()), 3))
    terrain = np.maximum(np.interp(grey, kx, ky), HOI4_LAND_MIN)
    d = ndimage.distance_transform_cdt(~water, metric="taxicab")
    ramp = np.asarray(HOI4_COAST_RAMP + (1.0,))
    wgt = ramp[np.minimum(d, len(ramp) - 1)]
    out = HOI4_COAST_LEVEL + (terrain - HOI4_COAST_LEVEL) * wgt
    out = np.where(d >= 2, np.maximum(out, HOI4_LAND_MIN), out)
    out = np.where(water, HOI4_WATER_LEVEL, out)
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), mode="L")


# HOI4 world_normal.bmp: 24-bit RGB, half the heightmap's size in the
# vanilla game (2816 x 1024 for 5632 x 2048; the wiki says "same size as
# provinces.bmp", the game's own file is half). Per pixel a unit normal:
# R = X (0 west .. 255 east), G = Y (0 south .. 255 north), B = Z (128
# edge-on .. 255 facing the viewer); flat = (128, 128, 255). Fitted on the
# vanilla files (_dev/nm_ref.py, nm_grid.py): n ~ normalize(-s * dh/dx,
# -s * dh/dy, 1) with the gradient of the heightmap (HOI4 levels)
# Gaussian-blurred by 0.5 px, per full-resolution pixel, averaged 2 x 2 --
# s = 0.11: rms error ~11 levels on R/G vs vanilla made from its own
# heightmap, and the same contrast (|R - 128| p90/p99 26/60 vs 27.5/58.5).
NORMAL_MAP_SCALE = 0.11
NORMAL_MAP_BLUR_PX = 0.5


def hoi4_normal_map(height: Image.Image, half: bool = True, scale: float = NORMAL_MAP_SCALE) -> Image.Image:
    """world_normal.bmp from a HOI4-level heightmap (hoi4_heightmap): half
    its size (rounded up) like the vanilla file, or the same size. The
    scale is per heightmap pixel, so the normals match the terrain the game
    builds from that heightmap whatever the map's real-world zoom."""
    h = ndimage.gaussian_filter(np.asarray(height.convert("L")).astype(np.float64), NORMAL_MAP_BLUR_PX)
    gx = np.gradient(h, axis=1)   # rising eastwards
    gy = -np.gradient(h, axis=0)  # rising northwards (rows run south)
    if half:
        hh, ww = gx.shape
        pad = ((0, hh % 2), (0, ww % 2))
        gx, gy = (np.pad(g, pad, mode="edge") for g in (gx, gy))
        gx, gy = (g.reshape(g.shape[0] // 2, 2, g.shape[1] // 2, 2).mean(axis=(1, 3)) for g in (gx, gy))
    nx, ny = -scale * gx, -scale * gy
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)
    rgb = np.stack([127.5 + 127.5 * nx * inv, 127.5 + 127.5 * ny * inv, 127.5 + 127.5 * inv], axis=2)
    return Image.fromarray(np.clip(np.rint(rgb), 0, 255).astype(np.uint8), mode="RGB")
