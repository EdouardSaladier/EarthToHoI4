"""SHELVED (not used by the tool): nothing imports this module; the hooks in
core.render_map_detailed are commented out ("# HISTORICAL (shelved" markers),
and app.py / the UI never got a `hist=` parameter. To revive: uncomment
those hooks, re-run the import below (historical_maps/poland_1939/map.json
is still in the older multi-country format this version doesn't read), then
add `hist` to app._RENDER_PARAMS + a selector in the Countries tab.

Historical borders from raster maps (e.g. XrysD's administrative maps on
Wikimedia Commons): georeference the image automatically, find the map's
focus country (the coloured land -- 1939 Poland), and give that country its
historical extent on the tool's country map.

Only the focus country changes. Coastlines, lakes and every land/sea step
stay GADM's, and so do all other countries: inside the focus country's
historical area, land becomes that country; land of today's focus country
outside it goes to the nearest other country (1939: Pomerania, Silesia and
Masuria -> Germany / Russia's Kaliningrad, Gdansk area -> nearest).

Import (once per map; needs GADM + HydroLAKES):
    python historical.py import historical_maps/poland_1939/config.json

  1. Water mask of the raster: the maps' light-blue water fill + its blue
     outline (lakes drawn as outline + fill, holes filled).
  2. Reference water on a lon/lat grid: GADM land cut by HydroLAKES lakes.
  3. Coarse fit: similarity transform (scale, rotation, shift) of the
     reference water edges -- in a Lambert conformal conic -- by FFT
     correlation against the raster's water edges.
  4. Fine fit: cubic polynomial (LCC km -> raster px) minimising the
     distance from reference water edges to raster water edges (chamfer,
     Cauchy loss: lakes the map leaves out or modern reservoirs are outliers).
  5. Labels: 1 = focus country (coloured land, areas >= 1000 px), 2 = other
     land (light grey); red borders, white subdivision lines, text and water
     within FILL_MAX_PX take the nearest label; open water further out and
     the ignored boxes (title, insets) are 0 = no data.
  Output: historical_maps/<id>/labels.png (8-bit) + map.json.

Render (`apply`, called by core.render_map_detailed right after the fill):
every land pixel inside the raster is un-projected to lon/lat and mapped
through the fit into the raster.
"""
import json
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, optimize, signal

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent
MAPS_DIR = ROOT / "historical_maps"   # config.json + built labels.png / map.json per map (ship with the tool)
R_EARTH_KM = 6371.0
REF_STEP_DEG = 0.004                  # reference water grid (~0.3-0.45 km)
REF_LAKE_MIN_KM2 = 0.3
COARSE_DOWNSAMPLE = 8
POLY_DEGREE = 3
USER_AGENT = "map-tool/1.0 (historical borders importer)"
FILL_MAX_PX = 20                      # raster px a region label spreads over water / lines / labels


# ---- projection + fit model ------------------------------------------------

def lcc(lon, lat, p):
    """Spherical Lambert conformal conic, km (v = north). p: lon0, lat0, p1, p2."""
    f = np.radians
    phi1, phi2, phi0 = f(p["p1"]), f(p["p2"]), f(p["lat0"])
    n = np.log(np.cos(phi1) / np.cos(phi2)) / np.log(np.tan(np.pi / 4 + phi2 / 2) / np.tan(np.pi / 4 + phi1 / 2))
    F = np.cos(phi1) * np.tan(np.pi / 4 + phi1 / 2) ** n / n
    rho = R_EARTH_KM * F / np.tan(np.pi / 4 + f(np.asarray(lat, dtype=np.float64)) / 2) ** n
    rho0 = R_EARTH_KM * F / np.tan(np.pi / 4 + phi0 / 2) ** n
    th = n * f((np.asarray(lon, dtype=np.float64) - p["lon0"] + 180.0) % 360.0 - 180.0)
    return rho * np.sin(th), rho0 - rho * np.cos(th)


def _terms(u, v, deg):
    return np.stack([u ** i * v ** (d - i) for d in range(deg + 1) for i in range(d, -1, -1)])


def lonlat_to_raster(model, lon, lat):
    """lon/lat -> raster pixel (x, y) through the fitted model."""
    U, V = lcc(lon, lat, model["lcc"])
    T = _terms((U - model["u0"]) / model["k"], (V - model["v0"]) / model["k"], model["deg"])
    return np.asarray(model["cx"]) @ T, np.asarray(model["cy"]) @ T


def _edges(mask):
    return mask & ~ndimage.binary_erosion(mask, border_value=1)


# ---- import -----------------------------------------------------------------

def _download_commons(file_name: str, dest: Path):
    if dest.exists():
        return
    api = ("https://commons.wikimedia.org/w/api.php?action=query&prop=imageinfo&iiprop=url&format=json&titles="
           + urllib.parse.quote("File:" + file_name))
    with urllib.request.urlopen(urllib.request.Request(api, headers={"User-Agent": USER_AGENT}), timeout=30) as r:
        page = next(iter(json.load(r)["query"]["pages"].values()))
    url = page["imageinfo"][0]["url"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=120) as r:
        tmp.write_bytes(r.read())
    tmp.replace(dest)


def _raster_classes(rgb, cfg):
    """Water, red border, grey land, coloured land masks of the raster."""
    a = rgb.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    wr, wg, wb = cfg.get("water_fill", [222, 235, 247])
    fill = (abs(r - wr) <= 10) & (abs(g - wg) <= 10) & (abs(b - wb) <= 10) & (b - g >= 6) & (g - r >= 6)
    outline = (b - r >= 50) & (b >= 150) & (g - r >= 30)
    water = ndimage.binary_fill_holes(fill | outline)
    red = (r - g >= 50) & (r - b >= 50)
    # Light grey only: mid-grey labels / inset connector lines (127) take the
    # nearest region instead (the connectors cross Poland).
    grey = (abs(r - g) <= 12) & (abs(g - b) <= 12) & (r >= 185) & (r <= 215) & ~water
    spread = a.max(axis=2) - a.min(axis=2)
    # Blue-dominant pixels (sea gradients, antialiased lake outlines) are
    # water-ish, never the focus country.
    bluish = (b >= g) & (b >= r)
    coloured = (spread >= 20) & ~red & ~water & ~grey & ~bluish
    return water, red, grey, coloured


def _ignore_mask(shape, cfg):
    m = np.zeros(shape, dtype=bool)
    for x0, y0, x1, y1 in cfg.get("ignore", []):
        m[y0:y1, x0:x1] = True
    return m


def _reference_water(bounds, progress):
    """Water (sea + HydroLAKES lakes) on a lon/lat grid over `bounds`."""
    import core
    lon0, lat0, lon1, lat1 = bounds
    gw, gh = int((lon1 - lon0) / REF_STEP_DEG), int((lat1 - lat0) / REF_STEP_DEG)
    to_grid = lambda pts: np.column_stack([(pts[:, 0] - lon0) / REF_STEP_DEG, (lat1 - pts[:, 1]) / REF_STEP_DEG])
    geoms = core.load_geometries(progress=progress)
    rings = []
    for tag, gm in geoms.items():
        x0, y0, x1, y1 = gm["lonlat_bbox"]
        if tag in core.WATER_TAGS or x1 < lon0 or x0 > lon1 or y1 < lat0 or y0 > lat1:
            continue
        for _is_hole, pts in gm["rings"]:
            if pts[:, 0].max() >= lon0 and pts[:, 0].min() <= lon1 and pts[:, 1].max() >= lat0 and pts[:, 1].min() <= lat1:
                rings.append(to_grid(pts))
    land = np.zeros((gh, gw), dtype=bool)
    core._scanline_fill(land, rings, True)
    idx = core.load_lake_index()
    if idx is None:
        raise RuntimeError("HydroLAKES is needed to georeference a map (Download tab -> HydroSHEDS Lakes)")
    bb = idx["bbox"]
    sel = np.flatnonzero((bb[:, 2] >= lon0) & (bb[:, 0] <= lon1) & (bb[:, 3] >= lat0) & (bb[:, 1] <= lat1)
                         & (idx["area_km2"] >= REF_LAKE_MIN_KM2) & (idx["shape_type"] == 5)
                         & (idx["lake_id"] != core.CASPIAN_LAKE_ID))
    raw, off, n_parts, n_points = idx["raw"], idx["off"], idx["n_parts"], idx["n_points"]
    lake_rings = []
    for i in sel:
        p0 = off[i] + 44 + 4 * n_parts[i]
        parts = np.append(raw[off[i] + 44:p0].view("<i4"), n_points[i])
        pts = np.frombuffer(raw[p0:p0 + 16 * n_points[i]], dtype="<f8").reshape(-1, 2)
        lake_rings += [to_grid(pts[a:b]) for a, b in zip(parts[:-1], parts[1:])]
    lakes = np.zeros((gh, gw), dtype=bool)
    core._scanline_fill(lakes, lake_rings, True)
    return ~land | lakes


def _coarse_fit(ref_U, ref_V, water, px_per_km_guess):
    """Best similarity (scale, rotation, shift): LCC km -> raster px, by FFT
    correlation of edge maps at 1/COARSE_DOWNSAMPLE. Returns (2x3 affine, score)."""
    D = COARSE_DOWNSAMPLE
    H, W = water.shape
    small = water[: H // D * D, : W // D * D].reshape(H // D, D, W // D, D).mean(axis=(1, 3)) > 0.5
    e_r = ndimage.binary_dilation(_edges(small)).astype(np.float32)
    kernel = e_r[::-1, ::-1]
    best = (-1.0, None)
    for s_full in np.geomspace(px_per_km_guess / 2.0, px_per_km_guess * 2.0, 70):
        s = s_full / D
        for rot in np.radians(np.arange(-10.0, 10.1, 2.0)):
            c, sn = np.cos(rot), np.sin(rot)
            X = s * (c * ref_U - sn * ref_V)
            Y = -s * (sn * ref_U + c * ref_V)
            xmin, ymin = X.min(), Y.min()
            cw, ch = int(X.max() - xmin) + 1, int(Y.max() - ymin) + 1
            if cw < e_r.shape[1] or ch < e_r.shape[0] or cw * ch > 25_000_000:
                continue
            img = np.zeros((ch, cw), dtype=np.float32)
            img[(Y - ymin).astype(int), (X - xmin).astype(int)] = 1.0
            corr = signal.fftconvolve(img, kernel, mode="valid")
            k = np.unravel_index(np.argmax(corr), corr.shape)
            score = float(corr[k] / e_r.sum())
            if score > best[0]:
                tx, ty = -D * (xmin + k[1]), -D * (ymin + k[0])
                best = (score, np.array([[D * s * c, -D * s * sn, tx], [-D * s * sn, -D * s * c, ty]]))
    return best[1], best[0]


def _poly_fit(U, V, x_init, y_init, dt, deg, f_scale):
    u0, v0, k = float(U.mean()), float(V.mean()), float(max(U.std(), V.std()))
    T = _terms((U - u0) / k, (V - v0) / k, deg)
    n = T.shape[0]
    p0 = np.concatenate([np.linalg.lstsq(T.T, x_init, rcond=None)[0], np.linalg.lstsq(T.T, y_init, rcond=None)[0]])
    gy, gx = np.gradient(dt)

    def res(p):
        return ndimage.map_coordinates(dt, [p[n:] @ T, p[:n] @ T], order=1, mode="nearest")

    def jac(p):
        X, Y = p[:n] @ T, p[n:] @ T
        dx = ndimage.map_coordinates(gx, [Y, X], order=1, mode="nearest")
        dy = ndimage.map_coordinates(gy, [Y, X], order=1, mode="nearest")
        return np.hstack([(dx * T).T, (dy * T).T])

    r = optimize.least_squares(res, p0, jac=jac, loss="cauchy", f_scale=f_scale, max_nfev=300)
    return {"deg": deg, "u0": u0, "v0": v0, "k": k, "cx": r.x[:n].tolist(), "cy": r.x[n:].tolist()}


def georeference(rgb, cfg, progress=print):
    """Fit model (dict) + stats for a raster map, from its water."""
    t0 = time.time()
    water, _red, _grey, _col = _raster_classes(rgb, cfg)
    ignore = _ignore_mask(water.shape, cfg)
    water &= ~ignore
    H, W = water.shape
    lon0, lat0, lon1, lat1 = cfg["approx_bounds"]
    p = {"lon0": (lon0 + lon1) / 2.0, "lat0": (lat0 + lat1) / 2.0,
         "p1": lat0 + (lat1 - lat0) / 6.0, "p2": lat1 - (lat1 - lat0) / 6.0}
    margin = max(lon1 - lon0, lat1 - lat0) * 0.1  # wider pulls in unrelated lake districts (Finland)
    ref_bounds = (lon0 - margin, lat0 - margin, lon1 + margin, lat1 + margin)
    ref = _reference_water(ref_bounds, progress)
    ry, rx = np.nonzero(_edges(ref))
    U, V = lcc(ref_bounds[0] + (rx + 0.5) * REF_STEP_DEG, ref_bounds[3] - (ry + 0.5) * REF_STEP_DEG, p)
    progress(f"  reference: {len(U):,} water edge points ({time.time() - t0:.1f}s)")
    r_edges = _edges(water)
    bx, by = lcc(np.array([lon0, lon1]), np.array([(lat0 + lat1) / 2] * 2), p)
    guess = W / abs(bx[1] - bx[0])
    A, score = _coarse_fit(U, V, water, guess)
    progress(f"  coarse fit: {np.hypot(A[0, 0], A[1, 0]):.3f} px/km, "
             f"rotation {np.degrees(np.arctan2(-A[1, 0], A[0, 0])):+.1f} deg, score {score:.2f} "
             f"({time.time() - t0:.1f}s)")
    X, Y = A[0, 0] * U + A[0, 1] * V + A[0, 2], A[1, 0] * U + A[1, 1] * V + A[1, 2]
    inside = (X >= 5) & (X < W - 5) & (Y >= 5) & (Y < H - 5)
    inside[inside] &= ~ignore[Y[inside].astype(int), X[inside].astype(int)]
    U, V, X, Y = U[inside], V[inside], X[inside], Y[inside]
    dt = np.minimum(ndimage.distance_transform_edt(~r_edges), 40.0)
    model = None
    for deg, fs in [(1, 8.0), (2, 4.0), (POLY_DEGREE, 2.0)]:
        model = _poly_fit(U, V, X, Y, dt, deg, fs)
        model["lcc"] = p
        T = _terms((U - model["u0"]) / model["k"], (V - model["v0"]) / model["k"], deg)
        X, Y = np.asarray(model["cx"]) @ T, np.asarray(model["cy"]) @ T
    d = ndimage.map_coordinates(dt, [Y, X], order=1, mode="nearest")
    inl = d <= 6
    stats = {"coarse_score": round(score, 3), "edge_points": int(len(d)), "matched_share": round(float(inl.mean()), 3),
             "median_error_px": round(float(np.median(d[inl])), 2) if inl.any() else None,
             "px_per_km": round(float(np.hypot(A[0, 0], A[1, 0])), 4)}
    progress(f"  fine fit: {stats['matched_share']:.0%} of reference edges matched, "
             f"median error {stats['median_error_px']} px ({time.time() - t0:.1f}s)")
    return model, stats


FOCUS, OTHER = 1, 2  # label values


def segment(rgb, cfg):
    """Label image (uint8): FOCUS / OTHER land, 0 = no data."""
    _water, _red, grey, coloured = _raster_classes(rgb, cfg)
    ignore = _ignore_mask(grey.shape, cfg)
    labels = np.zeros(grey.shape, dtype=np.uint8)
    # The focus country: coloured areas of at least 1000 px (antialiased
    # fringes of the red lines are "coloured" specks too).
    col_lab, _nc = ndimage.label(coloured & ~ignore, structure=np.ones((3, 3)))
    col_keep = np.bincount(col_lab.ravel()) >= 1000
    col_keep[0] = False
    labels[grey & ~ignore] = OTHER
    labels[col_keep[col_lab]] = FOCUS
    # Borders, subdivision lines, labels and water near land take the
    # nearest label; open water further out is no data, so islands the map
    # leaves out (Bornholm) keep their country.
    dist, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    labels = labels[iy, ix]
    labels[dist > FILL_MAX_PX] = 0
    # The ignored boxes (title, insets) hide what's under them: no data.
    labels[ignore] = 0
    return labels


def build(config_path, progress=print):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    out_dir = config_path.parent
    import core
    src = core.DATA_DIR / "historical_maps" / cfg["commons_file"]
    progress(f"{cfg['title']}: source {src}")
    _download_commons(cfg["commons_file"], src)
    rgb = np.array(Image.open(src).convert("RGB"))
    model, stats = georeference(rgb, cfg, progress)
    labels = segment(rgb, cfg)
    # Lon/lat extent of the focus country, for framing (a 0.02-degree grid
    # mapped into the raster).
    lon0, lat0, lon1, lat1 = cfg["approx_bounds"]
    m = 3.0
    glon, glat = np.meshgrid(np.arange(lon0 - m, lon1 + m, 0.02), np.arange(lat0 - m, lat1 + m, 0.02))
    glon, glat = glon.ravel(), glat.ravel()
    X, Y = lonlat_to_raster(model, glon, glat)
    H, W = labels.shape
    ok = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
    lab_at = np.zeros(len(X), dtype=np.uint8)
    lab_at[ok] = labels[Y[ok].astype(int), X[ok].astype(int)]
    sel = lab_at == FOCUS
    focus_bbox = [round(float(glon[sel].min()), 3), round(float(glat[sel].min()), 3),
                  round(float(glon[sel].max()), 3), round(float(glat[sel].max()), 3)]
    Image.fromarray(labels, mode="L").save(out_dir / "labels.png", optimize=True)
    meta = {"id": out_dir.name, "title": cfg["title"], "source": cfg.get("source"), "author": cfg.get("author"),
            "license": cfg.get("license"), "size": [W, H], "model": model, "fit": stats,
            "focus_tag": cfg["focus_tag"], "focus_lonlat_bbox": focus_bbox,
            "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "map.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    progress(f"  {cfg['focus_tag']}: {int((labels == FOCUS).sum()):,} raster px, lon/lat extent {focus_bbox}")
    progress(f"  saved {out_dir / 'labels.png'} + map.json")
    return meta


# ---- runtime ------------------------------------------------------------------

_memo = {}
_lock = threading.Lock()


def list_maps():
    out = []
    for p in sorted(MAPS_DIR.glob("*/map.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        out.append({"id": m["id"], "title": m["title"], "source": m.get("source"), "author": m.get("author"),
                    "license": m.get("license"), "fit": m.get("fit")})
    return out


def load_map(map_id: str):
    with _lock:
        if map_id not in _memo:
            d = MAPS_DIR / map_id
            if not map_id or "/" in map_id or "\\" in map_id or not (d / "map.json").exists():
                raise KeyError(map_id)
            meta = json.loads((d / "map.json").read_text(encoding="utf-8"))
            meta["labels"] = np.array(Image.open(d / "labels.png"))
            _memo[map_id] = meta
        return _memo[map_id]


def apply(hmap, canvas, hidden_land, colors, excluded, view, background, chunk_px=2_000_000):
    """Give the map's focus country its historical extent on `canvas`: land
    the raster calls the focus country becomes it; land of today's focus
    country anywhere else goes to the nearest other country's land. Other
    countries are untouched. A hidden focus country becomes hidden land.
    Returns a report dict."""
    import core
    t0 = time.time()
    h, w = canvas.shape[:2]
    scale = view["scale"]
    bg = np.asarray(background, dtype=canvas.dtype)
    labels = hmap["labels"]
    lh, lw = labels.shape
    tag = hmap["focus_tag"]
    rgb = np.asarray(colors[tag], dtype=canvas.dtype)
    hidden = tag in excluded
    cols_x = view["win_xmin"] + (np.arange(w) + 0.5) / scale
    rows_y = view["win_ymin"] + (h - (np.arange(h) + 0.5)) / scale
    focus = np.zeros((h, w), dtype=bool)
    rows_per = max(1, chunk_px // max(w, 1))
    for r0 in range(0, h, rows_per):
        r1 = min(h, r0 + rows_per)
        land = ~(canvas[r0:r1] == bg).all(axis=2)
        if hidden_land is not None:
            land |= hidden_land[r0:r1]
        ys, xs = np.nonzero(land)
        if not len(ys):
            continue
        lon, lat, inside = core._unproject(view["projection"], view["center"], cols_x[xs], rows_y[r0 + ys])
        X, Y = lonlat_to_raster(hmap["model"], lon, lat)
        ok = inside & (X >= 0) & (X < lw) & (Y >= 0) & (Y < lh)
        if ok.any():
            f = labels[Y[ok].astype(np.int64), X[ok].astype(np.int64)] == FOCUS
            focus[ys[ok][f] + r0, xs[ok][f]] = True
    # Today's focus country outside its historical area -> the nearest
    # other country's land. (A hidden focus country's land is hidden land,
    # indistinguishable from other hidden countries': it stays hidden.)
    today = (canvas == rgb).all(axis=2)
    lost = today & ~focus
    other_land = ~(canvas == bg).all(axis=2) & ~today & ~focus
    reassigned = 0
    if lost.any() and other_land.any():
        _d, (iy, ix) = ndimage.distance_transform_edt(~other_land, return_indices=True)
        canvas[lost] = canvas[iy[lost], ix[lost]]
        reassigned = int(lost.sum())
    if hidden:
        canvas[focus] = bg
        if hidden_land is not None:
            hidden_land[focus] = True
    else:
        canvas[focus] = rgb
        if hidden_land is not None:
            hidden_land[focus] = False
    return {"map": hmap["id"], "title": hmap["title"], "tag": tag, "focus_px": int(focus.sum()),
            "reassigned_px": reassigned, "seconds": round(time.time() - t0, 2)}


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "import":
        for c in sys.argv[2:]:
            build(c)
    else:
        print(__doc__)
