"""Climate normals for the Strategic regions tab's weather: ERA5 1991-2020
monthly climatologies (0.25 deg, land and sea) from the World Bank Climate
Change Knowledge Portal's open bucket (CC BY 4.0), one NetCDF3 file per
variable in data/climate/ (Download tab card "Climate normals (ERA5)").
Read with scipy.io.netcdf_file (memory-mapped; no HDF5 needed: these files
are classic NetCDF, unlike the portal's CRU ones).

Variables (each 12 x 721 x 1440, lat -90..90, lon -180..179.75):
  tas / tasmin / tasmax  mean / daily min / daily max near-surface air
                         temperature (deg C)
  pr                     precipitation (mm per month)
  fd / id                frost days (daily min < 0 C) / ice days (daily max < 0 C)
  r20mm                  days with >= 20 mm of precipitation
  sfcwind                mean 10 m wind speed (m/s)
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CLIMATE_DIR = ROOT / "data" / "climate"
VARIABLES = ("tas", "tasmin", "tasmax", "pr", "fd", "id", "r20mm", "sfcwind")
URL = ("https://wbg-cckp.s3.amazonaws.com/data/era5-x0.25/{v}/era5-x0.25-historical/"
       "climatology-{v}-monthly-mean_era5-x0.25_era5-x0.25-historical_climatology_mean_1991-2020.nc")
FILL = 1e20

_LOADED = None


def file_name(v):
    return URL.format(v=v).rsplit("/", 1)[-1]


def available():
    return all((CLIMATE_DIR / file_name(v)).exists() for v in VARIABLES)


def load():
    """{var: (12, 721, 1440) float32 memmap, 'lat': (721,), 'lon': (1440,)};
    loaded once per process."""
    global _LOADED
    if _LOADED is None:
        import warnings
        from scipy.io import netcdf_file
        warnings.filterwarnings("ignore", message="Cannot close a netcdf_file opened with mmap=True")
        out = {}
        for v in VARIABLES:
            f = netcdf_file(str(CLIMATE_DIR / file_name(v)), "r", mmap=True)
            name = next(k for k in f.variables if k.startswith("climatology-"))
            out[v] = f.variables[name].data            # memmap; f stays open with it
            if "lat" not in out:
                out["lat"] = np.array(f.variables["lat"].data, dtype=np.float64)
                out["lon"] = np.array(f.variables["lon"].data, dtype=np.float64)
        _LOADED = out
    return _LOADED


def _idx(C, lat, lon):
    """Grid indices of points (nearest cell centre; lon wraps)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = (np.asarray(lon, dtype=np.float64) + 180.0) % 360.0 - 180.0
    i = np.clip(np.rint((lat + 90.0) * 4).astype(np.int64), 0, 720)
    j = np.rint((lon + 180.0) * 4).astype(np.int64) % 1440
    return i, j


def box_mean(C, lat0, lat1, lon0, lon1):
    """{var: 12 monthly means} over a lat/lon box (cells weighted by cos(lat))."""
    lat, lon = C["lat"], C["lon"]
    li = np.flatnonzero((lat >= lat0) & (lat <= lat1))
    lj = np.flatnonzero(((lon - lon0) % 360 <= (lon1 - lon0) % 360))
    w = np.cos(np.radians(lat[li]))[:, None] * np.ones(len(lj))[None, :]
    out = {}
    for v in VARIABLES:
        a = np.asarray(C[v][:, li[0]:li[-1] + 1, :][:, :, lj], dtype=np.float64)
        ok = np.abs(a) < FILL / 10
        out[v] = (np.where(ok, a, 0) * w).sum(axis=(1, 2)) / np.maximum((ok * w).sum(axis=(1, 2)), 1e-9)
    return out


def region_means(C, labels, lat, lon, n):
    """{var: (n + 1, 12)} means per label (0..n) of points (lat, lon) --
    e.g. every pixel of a map, labelled by strategic region; each point
    samples its grid cell."""
    i, j = _idx(C, lat, lon)
    flat = i * 1440 + j
    cnt = np.bincount(labels, minlength=n + 1).astype(np.float64)
    out = {}
    for v in VARIABLES:
        a = np.asarray(C[v], dtype=np.float32).reshape(12, -1)
        res = np.zeros((n + 1, 12))
        for m in range(12):
            x = a[m][flat].astype(np.float64)
            ok = np.abs(x) < FILL / 10
            s = np.bincount(labels[ok], weights=x[ok], minlength=n + 1)
            k = np.bincount(labels[ok], minlength=n + 1)
            res[:, m] = np.where(k > 0, s / np.maximum(k, 1), np.nan)
        out[v] = res
    out["count"] = cnt
    return out


# ---------------------------------------------------------------- HOI4 weather
# Each field of a strategic region's weather period as a formula of the
# month's climate; parameters fitted on vanilla regions with a known place
# (_dev/weather_calib.py -> _dev/weather_fit.py: least squares against the
# game's own values, land and sea apart). Vanilla's values are hand-made
# (discrete steps, regional quirks): the fit follows their trends, not
# every region.
# Fit quality (_dev/weather_fit.py, 85 vanilla regions x 12 months): the
# trends by temperature / precipitation band are reproduced (e.g. land
# below -20 C: snow 0.37 vanilla / 0.35 here, blizzard 0.24 / 0.23; land
# with >= 250 mm: heavy rain 0.39 / 0.34), but single region-months scatter
# (rms ~ the spread of vanilla's own values for rain and calm weather,
# temperatures +-6-7 C): vanilla's numbers are hand-tuned.
WEATHER_PARAMS = {
    # temperature = { min max }: a + b tas + c (tasmax - tasmin); vanilla's
    # are extremes, wider than the mean daily min / max
    "tmin": (-12.978, 0.9547, 0.4898), "tmax": (6.94, 0.7987, 0.3979),
    # peak, pr scale (mm), drop at high pr, drop scale, rain share in frost
    "rain_light": (0.4144, 81.62, 0.4852, 90.13, 0.558),
    "rain_heavy": (0.0637, 0.0009, 0.0),             # base, per mm, per day >= 20 mm
    "no_phenomenon": (0.8419, 0.0655),               # a - b log1p(pr)
    "snow": (0.0101, 0.0028, 0.45),                  # per frost day, per degree below -10, cap
    "blizzard": (0.0069, 6.946, 0.3),                # slope, tas where it starts, cap
    "min_snow_level": (0.1341, 2.284, 38.85),        # level, tas where it starts, ramp
    "mud": (0.4814, -32.08, 54.02),                  # floor, tas of the floor, ramp to 1
    "sandstorm": (0.7038, 24.0, 13.53, 100.0),       # max, tas start, tas ramp, pr scale
    "arctic_water": (0.7427, 8.87, 3.202),           # max, tas midpoint, width (logistic)
    # sea: rain grows with precipitation up to a level, flat above (most of
    # vanilla's seas are constant all year; the Mediterranean dries out in
    # summer); calm weather rises as it dries
    "sea_rain_light": (0.2893, 13.53), "sea_rain_heavy": (0.1583, 13.65),   # level, pr (mm) where it saturates
    "sea_no_phenomenon": (0.5097, 0.2458, 30.12),                       # level, extra when dry, pr scale
}


def weather_month(c, sea, P=None):
    """HOI4 weather period values for one month's climate `c` (dict of
    scalars: tas, tasmin, tasmax, pr, fd, id, r20mm, sfcwind) of a land or
    sea region."""
    P = P or WEATHER_PARAMS
    tas, tmn, tmx, pr = c["tas"], c["tasmin"], c["tasmax"], max(c["pr"], 0.0)
    rng_ = tmx - tmn
    out = {}
    a, b, cc = P["tmin"]
    out["tmin"] = a + b * tas + cc * rng_
    a, b, cc = P["tmax"]
    out["tmax"] = a + b * tas + cc * rng_
    if out["tmax"] < out["tmin"] + 1:
        out["tmax"] = out["tmin"] + 1
    if sea:
        a, sc = P["sea_rain_light"]
        out["rain_light"] = a * min(1.0, pr / sc)
        a, sc = P["sea_rain_heavy"]
        out["rain_heavy"] = a * min(1.0, pr / sc)
        a, extra, sc = P["sea_no_phenomenon"]
        out["no_phenomenon"] = a + extra * max(0.0, 1.0 - pr / sc)
        out["snow"] = out["blizzard"] = out["min_snow_level"] = out["sandstorm"] = 0.0
        out["mud"] = 1.0
        mx, mid, wd = P["arctic_water"]
        out["arctic_water"] = mx / (1.0 + np.exp((tas - mid) / wd))
    else:
        peak, sc, drop, dsc, f0 = P["rain_light"]
        liquid = f0 + (1.0 - f0) * float(np.clip((tas + 2.0) / 8.0, 0.0, 1.0))
        out["rain_light"] = liquid * peak * min(1.0, pr / sc) * (1.0 - drop * float(np.clip((pr - 100.0) / dsc, 0, 1)))
        base, per_mm, per_day = P["rain_heavy"]
        out["rain_heavy"] = liquid * (base + per_mm * pr + per_day * c["r20mm"])
        a, b = P["no_phenomenon"]
        out["no_phenomenon"] = a - b * np.log1p(pr)
        per_fd, per_deg, cap = P["snow"]
        out["snow"] = min(cap, per_fd * c["fd"] + per_deg * max(0.0, -10.0 - tas))
        slope, start, cap = P["blizzard"]
        out["blizzard"] = float(np.clip(slope * (start - tas), 0.0, cap))
        lvl, start, ramp = P["min_snow_level"]
        out["min_snow_level"] = lvl * float(np.clip((start - tas) / ramp, 0.0, 1.0))
        floor, t0, ramp = P["mud"]
        out["mud"] = float(np.clip(floor + (1.0 - floor) * (tas - t0) / ramp, floor, 1.0))
        mx, t0, ramp, psc = P["sandstorm"]
        out["sandstorm"] = mx * float(np.clip((tas - t0) / ramp, 0.0, 1.0)) * float(np.exp(-pr / psc))
        out["arctic_water"] = 0.0
    for k, v in out.items():
        if k not in ("tmin", "tmax"):
            out[k] = float(np.clip(v, 0.0, 1.0)) if k != "no_phenomenon" else float(np.clip(v, 0.05, 1.5))
    return out


def pixel_lonlat(view, step=1):
    """(lat, lon, inside) of the centres of every `step`-th pixel of a
    render's frame (debug["view"]), as 2-D arrays."""
    import core
    h, w, scale = view["height"], view["width"], view["scale"]
    cols = np.arange(0, w, step)
    rows = np.arange(0, h, step)
    xx = view["win_xmin"] + (cols + 0.5) / scale
    yy = view["win_ymin"] + (h - (rows + 0.5)) / scale
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    lon, lat, inside = core._unproject(view["projection"], view["center"], X, Y)
    return lat, lon, inside, rows, cols


def region_climate(view, RI, n, step=4):
    """{var: (n + 1, 12)} climate per region label (RI: region id per
    pixel of the frame), sampling every `step`-th pixel; a region with no
    sample takes the cell at one of its pixels."""
    C = load()
    lat, lon, inside, rows, cols = pixel_lonlat(view, step)
    lab = RI[np.ix_(rows, cols)]
    ok = inside & (lab > 0)
    res = region_means(C, lab[ok].astype(np.int64), lat[ok], lon[ok], n)
    missing = np.flatnonzero(res["count"][1:] == 0) + 1
    if len(missing):
        ys, xs = np.nonzero(np.isin(RI, missing))
        first = {}
        for y_, x_ in zip(ys.tolist(), xs.tolist()):
            first.setdefault(int(RI[y_, x_]), (y_, x_))
        h, scale = view["height"], view["scale"]
        import core
        for r, (y_, x_) in first.items():
            lo, la, _in = core._unproject(view["projection"], view["center"],
                                          np.array([view["win_xmin"] + (x_ + 0.5) / scale]),
                                          np.array([view["win_ymin"] + (h - (y_ + 0.5)) / scale]))
            one = region_means(C, np.array([1]), la, lo, 1)
            for v in VARIABLES:
                res[v][r] = one[v][1]
    return res


def region_weather(clim, r, sea):
    """12 monthly weather dicts for region r of region_climate's result."""
    out = []
    for m in range(12):
        c = {v: float(clim[v][r, m]) if np.isfinite(clim[v][r, m]) else 0.0 for v in VARIABLES}
        out.append(weather_month(c, sea))
    return out
