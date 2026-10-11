"""
Web UI: pick a country + projection, preview the rendered map.

Usage:
    python app.py
    (then open http://127.0.0.1:5000)
"""

import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, send_file, Response

import cities
import density
import eez
import hoi4mod
import city_names
import climate
import colormap
import core
import landcover
import palettes
import provinces
import result_cache
import states
import strategic
import terrain

app = Flask(__name__)

INDEX_HTML = (core.BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")

# All Download-tab files are saved here, one subfolder per resource, rather
# than left to the browser's own download folder, so this tool's data lives
# alongside the tool itself. core.py looks for the extracted GADM shapefile
# at DATA_DIR/gadm/gadm36_0.shp (see core.resolve_shapefile_path).
DATA_DIR = core.DATA_DIR

# Full-resolution tiles: the globe is split into an A-D (west-to-east) by 1-2
# (north/south) grid of 8 tiles.
_TILES = ["A1", "B1", "C1", "D1", "A2", "B2", "C2", "D2"]


def _tile_files(url_for_tile):
    return [{"label": t, "url": url_for_tile(t)} for t in _TILES]


# External datasets offered on the Download tab. GADM's own global bundle is
# offered as an actual .shp archive (version 3.6, the exact dataset this
# tool's gadm36_0.shp comes from) rather than the current 4.1 site's
# GeoPackage/geodatabase-only world bundle, since the user asked for
# shapefiles specifically. The Blue Marble image is the August "Blue Marble:
# Next Generation" satellite composite; the topographic map is NASA's
# separate grayscale elevation-only dataset (GEBCO-derived, land elevation
# 0-6400m encoded as 8-bit brightness, no color/satellite imagery, and not
# month-specific since terrain doesn't change monthly) -- NOT the shaded
# "Topography and Bathymetry" color composite, which looks like a satellite
# photo rather than a heightmap. Both are offered as their 8 full-resolution
# tiles (A1-D2) so the whole globe can be downloaded at full detail.
DOWNLOAD_RESOURCES = [
    {
        "id": "gadm",
        "title": "GADM Shapefiles",
        "description": "Global Administrative Areas boundaries for every country, all levels (the dataset this tool itself is built on). Automatically extracted after download.",
        "source_url": "https://gadm.org/",
        "files": [{"label": None, "url": "https://geodata.ucdavis.edu/gadm/gadm3.6/gadm36_levels_shp.zip"}],
    },
    {
        "id": "blue_marble",
        "title": "NASA Blue Marble (August)",
        "description": "Blue Marble: Next Generation cloud-free satellite composite for August, full resolution (500 m/pixel), as 8 tiles (A1-D2) covering the whole globe.",
        "source_url": "https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/base-map",
        "files": _tile_files(lambda t:
            f"https://assets.science.nasa.gov/content/dam/science/esd/eo/images/bmng/bmng-base/august/world.200408.3x21600x21600.{t}.jpg"
        ),
    },
    {
        "id": "topography",
        "title": "NASA Topographic Heightmap",
        "description": "Grayscale land-elevation heightmap (0-6400 m, black = sea level), full resolution (1 km/pixel), as 8 tiles (A1-D2) covering the whole globe. Not a satellite image.",
        "source_url": "https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/topography-bathymetry-maps",
        "files": _tile_files(lambda t:
            f"https://assets.science.nasa.gov/content/dam/science/esd/eo/images/bmng/topography/gebco_08_rev_elev_{t}_grey_geo.jpg"
        ),
    },
    # The same NASA Blue Marble set's ocean depths (GEBCO 08), same 8-tile
    # layout as the topography -- the Strategic regions' naval terrain
    # (bathymetry.py), and meant for other uses later.
    {
        "id": "bathymetry",
        "title": "NASA Bathymetry (ocean depth)",
        "description": "Grayscale ocean-depth map (GEBCO 08 via NASA Blue Marble), full resolution (1 km/pixel), as 8 tiles (A1-D2) covering the whole globe, like the topography heightmap. Used for the strategic regions' naval terrain (shallow sea / ocean / deep ocean).",
        "source_url": "https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/topography-bathymetry-maps",
        "files": _tile_files(lambda t:
            f"https://assets.science.nasa.gov/content/dam/science/esd/eo/images/bmng/bathymetry/gebco_08_rev_bath_{t}_grey_geo.jpg"
        ),
    },
    # HydroSHEDS v1 derived vector products. HydroRIVERS: global river
    # network as polylines, each segment carrying its long-term average
    # discharge (DIS_AV_CMS, m3/s) -- for rivers coloured by discharge.
    # HydroLAKES: every lake/reservoir >= 10 ha as polygons (~90 m shoreline
    # precision), used to cut lakes out of the land (core.load_lakes). The
    # HydroSHEDS Core rasters (DEM, flow direction/accumulation, ...) aren't
    # offered: nothing here uses them.
    {
        "id": "hydrosheds",
        "title": "HydroSHEDS (v1)",
        "description": "HydroRIVERS: global river network (~8.5M segments) with long-term average discharge per segment (DIS_AV_CMS, m³/s). HydroLAKES: ~1.4M lakes and reservoirs of at least 10 ha, as polygons. Shapefiles, automatically extracted after download.",
        "source_url": "https://www.hydrosheds.org/",
        "files": [
            {"label": "Rivers", "url": "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_shp.zip"},
            {"label": "Lakes", "url": "https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip"},
        ],
    },
    # Copernicus Global Land Service Land Cover 100 m (collection 2, epoch
    # 2015, v2.0.2, CC BY 4.0): the discrete classification only -- one
    # GeoTIFF, 8-bit class codes (20 shrubs, 30 herbaceous, 40 cropland,
    # 50 urban, 60 bare/sparse, 70 snow/ice, 80 permanent water, 90 herbaceous
    # wetland, 100 moss/lichen, 111-126 forest types, 200 open sea), EPSG:4326
    # at ~100 m. Not used by the renderer yet. Zenodo's URL ends in /content,
    # hence "filename".
    {
        "id": "landcover",
        "title": "Copernicus Land Cover (100 m)",
        "description": "Copernicus Global Land Service land cover map, 2015, 100 m resolution: one discrete class per pixel (forest types, shrubs, grassland, cropland, urban, bare, snow/ice, water, wetland, moss/lichen), single global GeoTIFF (EPSG:4326).",
        "source_url": "https://zenodo.org/records/3243509",
        "files": [
            {"label": None,
             "url": "https://zenodo.org/api/records/3243509/files/ProbaV_LC100_epoch2015_global_v2.0.2_discrete-classification_EPSG-4326.tif/content",
             "filename": "ProbaV_LC100_epoch2015_global_v2.0.2_discrete-classification_EPSG-4326.tif"},
        ],
    },
    # GeoNames "cities1000" (every populated place with >= 1,000 people,
    # ~171k, CC BY 4.0) via Opendatasoft's export API, full CSV (35 MB) --
    # all columns kept for the alternate names (display in other languages);
    # Parquet would be 19 MB but needs pyarrow (not installed). Semicolon-
    # separated, UTF-8 with BOM; alternate_names comma-separated; coordinates
    # = "lat, lon"; feature_code PPLC = capital, PPLA..PPLA4 = admin seats.
    # The export is streamed without Content-Length, hence "size_bytes" as a
    # fallback for the listed size. Not used by the renderer yet.
    {
        "id": "cities",
        "title": "GeoNames Cities (population >= 1,000)",
        "description": "Cities: every city, town and village with at least 1,000 inhabitants (~171,000 places): name, alternate names, feature code (capital, admin seat, ...), country, admin codes, population, coordinates (CSV). Names: every place's names tagged by language (all of GeoNames, kept zipped; the City tab keeps only its cities' names, a few MB) -- for city labels as endonyms or exonyms. Country info: each country's languages (which names are local).",
        "source_url": "https://www.geonames.org/",
        "files": [
            {"label": "Cities",
             "url": "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/geonames-all-cities-with-a-population-1000/exports/csv",
             "filename": "cities1000.csv",
             "size_bytes": 34_844_703},
            # All GeoNames alternate names with their language (alternateNameId,
            # geonameid, isolanguage, name, isPreferredName, isShortName,
            # isColloquial, isHistoric, from, to); not extracted (~700 MB of
            # text): cities.build_names streams it from the zip once.
            {"label": "Names", "url": "https://download.geonames.org/export/dump/alternateNamesV2.zip",
             "extract": False},
            {"label": "Country info", "url": "https://download.geonames.org/export/dump/countryInfo.txt"},
        ],
    },
    # Marine Regions (VLIZ) World EEZ v12, CC BY 4.0, straight from their
    # GeoServer WFS as shapefiles (the downloads page needs a form): the
    # 200 NM zones + overlapping claims / joint regimes, and the internal /
    # archipelagic waters they leave out (behind baselines). Streamed without
    # a Content-Length, and a HEAD would make the server build the export:
    # sizes given (size_bytes skips the remote size check).
    {
        "id": "eez",
        "title": "Exclusive Economic Zones (Marine Regions)",
        "description": "World EEZ v12 (Flanders Marine Institute, CC BY 4.0): every country's 200 NM exclusive economic zone, overlapping claims and joint regimes, plus internal waters (behind baselines: fjords, bays) and archipelagic waters. Shapefiles, extracted after download. Shown on the Countries tab's map as a light tint of each country's colour over its sea.",
        "source_url": "https://www.marineregions.org/downloads.php",
        "files": [
            {"label": "EEZ", "url": "https://geo.vliz.be/geoserver/MarineRegions/wfs?service=WFS&version=1.0.0&request=GetFeature&outputFormat=SHAPE-ZIP&typeName=MarineRegions:eez", "filename": "eez.zip", "size_bytes": 118_171_225},
            {"label": "Internal waters", "url": "https://geo.vliz.be/geoserver/MarineRegions/wfs?service=WFS&version=1.0.0&request=GetFeature&outputFormat=SHAPE-ZIP&typeName=MarineRegions:eez_internal_waters",
             "filename": "eez_internal_waters.zip", "size_bytes": 57_238_457},
            {"label": "Archipelagic waters", "url": "https://geo.vliz.be/geoserver/MarineRegions/wfs?service=WFS&version=1.0.0&request=GetFeature&outputFormat=SHAPE-ZIP&typeName=MarineRegions:eez_archipelagic_waters",
             "filename": "eez_archipelagic_waters.zip", "size_bytes": 29_249_379},
        ],
    },
    # JRC GHS-POP R2023A, epoch 2025, 30 arc-seconds (~1 km), EPSG:4326: people
    # per cell -- the Provinces tab's density map (density.py reduces it to a
    # 4 arc-minute grid once; bigger provinces where few people live).
    {
        "id": "density",
        "title": "Population density (GHS-POP 1 km)",
        "description": "JRC Global Human Settlement Layer GHS-POP R2023A: estimated population in 2025 on a 30 arc-second (~1 km) world grid (BigTIFF, extracted after download). The Provinces tab uses it as a density map: provinces get bigger where few people live (deserts, taiga, tundra).",
        "source_url": "https://human-settlement.emergency.copernicus.eu/download.php?ds=pop",
        "files": [{"label": None,
                   "url": "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_4326_30ss/V1-0/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.zip"}],
    },
    # JRC GHS-DUC R2023A V2-0: GHS-POP (R2023A) summed per GADM subdivision,
    # one CSV per GADM 4.1 level and epoch (1975-2030) -- the States tab's
    # population (population.py matches the 4.1 ids to our GADM 3.6 ones).
    # Kept zipped: only the 2025 total-population columns are read, once.
    {
        "id": "population",
        "title": "Population by GADM subdivision (GHS-POP)",
        "description": "JRC Global Human Settlement Layer: GHS-POP population (2025 estimate) summed for every GADM subdivision, all levels, as tables (GHS-DUC R2023A, keyed to GADM 4.1; matched to this tool's GADM 3.6, gaps split from the parent level by GeoNames places). Kept zipped; used for the States tab's state populations.",
        "source_url": "https://human-settlement.emergency.copernicus.eu/download.php?ds=DUC",
        "files": [{"label": None, "extract": False,
                   "url": "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_DUC_GLOBE_R2023A/V2-0/GHS_DUC_MT_GLOBE_R2023A_V2_0.zip"}],
    },
    # ERA5 1991-2020 monthly climate normals (0.25 deg, land and sea) from the
    # World Bank Climate Change Knowledge Portal's open S3 bucket (CC BY 4.0),
    # classic NetCDF3 (scipy reads them) -- the Strategic regions tab's
    # weather (climate.py). Sizes listed: no HEAD per file.
    {
        "id": "climate",
        "title": "Climate normals (ERA5, monthly)",
        "description": "ERA5 1991-2020 monthly climate normals on a 0.25-degree world grid (land and sea), from the World Bank Climate Change Knowledge Portal (CC BY 4.0): mean, daily minimum and daily maximum temperature, precipitation, frost days, ice days, days with 20 mm of rain or more, wind speed. One NetCDF file per variable. The Strategic regions tab averages them over each region to write its weather (temperatures, rain, snow, blizzards, mud, sandstorms, arctic water), calibrated on the game's own regions.",
        "source_url": "https://climateknowledgeportal.worldbank.org/download-data",
        "files": [{"label": label, "url": climate.URL.format(v=v), "size_bytes": 49_900_000}
                  for v, label in (("tas", "Mean temperature"), ("tasmin", "Daily minimum"),
                                   ("tasmax", "Daily maximum"), ("pr", "Precipitation"), ("fd", "Frost days"),
                                   ("id", "Ice days"), ("r20mm", "Days >= 20 mm"), ("sfcwind", "Wind speed"))],
    },
    # Real national flags for the HOI4 mod export (user: no flags from the game).
    # PNG per ISO 3166-1 alpha-2 code; sizes listed: no HEAD per file.
    {
        "id": "flags",
        "title": "Country flags",
        "description": "The real national flag of every country, 320 px wide PNG, from flagcdn.com (Wikimedia Commons' flags, mostly public domain). The HOI4 mod tab resizes them to the game's flag sizes for each exported country.",
        "source_url": "https://flagcdn.com",
        "files": [{"label": cc, "url": f"https://flagcdn.com/w320/{cc.lower()}.png", "size_bytes": 4000}
                  for cc in sorted(set(cities.COUNTRY_CONTINENT) | {"XK"})],
    },
    # SHELVED (user request): historical borders taken off the Download tab;
    # uncomment the block below to offer them again.
    # # Historical country borders (WW1 / WW2 era), high-detail only. Not
    # # used by the renderer yet -- downloaded for reference / future use.
    # # * Stanford Spatial History "European Borders WWII" (Holocaust
    # #   Geographies): 88 monthly shapefiles, Feb 1938 - May 1945, ~367k
    # #   vertices each, annexations / occupation zones / governments in exile
    # #   included. Projected (Europe Albers Equal Area, ED50), not lon/lat.
    # # * CHGIS V6 (Harvard / Fudan), 1911 layer = the Qing empire's last year,
    # #   the closest CHGIS gets to WW1: provinces (~185k vertices) and
    # #   prefectures, UTF-8 names. Served by Harvard Dataverse (the old
    # #   fas.harvard.edu/~chgis download pages only survive on the Wayback
    # #   Machine); its /api/access URLs end in a numeric id, hence "filename".
    # # Dropped as too coarse for this tool: CShapes 2.0 (ETH ICR, world
    # # 1886-2019 with exact dates, but ~1,000-1,500 vertices per country vs
    # # GADM's 100k+), aourednik/historical-basemaps (~1.5 MB GeoJSON for the
    # # whole world), Cliopatria. Not offered either: MPIDR / Census Mosaic
    # # (fine European detail, 1900/1930, but needs a registered login),
    # # HistoGIS (its "Europe Stateborders 1914" source holds just Denmark),
    # # Euratlas / GeaCron (commercial).
    # {
    #     "id": "historical",
    #     "title": "Historical Borders (WW1 / WW2)",
    #     "description": "High-detail country borders of the world wars era. Europe WWII: Stanford Spatial History, 88 monthly shapefiles Feb 1938 - May 1945 incl. annexations and occupation zones (Europe Albers projection). China 1911: CHGIS V6 provinces and prefectures at the end of the Qing (shapefiles).",
    #     "source_url": "https://web.archive.org/web/20210731221637/https://web.stanford.edu/group/spatialhistory/cgi-bin/site/pub.php?id=51",
    #     "files": [
    #         {"label": "Europe WWII", "url": "https://web.stanford.edu/group/spatialhistory/Publications/HolocaustGeographies/EuropeanBorders_WWII.zip"},
    #         {"label": "China 1911 provinces", "url": "https://dataverse.harvard.edu/api/access/datafile/2966688", "filename": "v6_1911_prov_pgn_utf.zip"},
    #         {"label": "China 1911 prefectures", "url": "https://dataverse.harvard.edu/api/access/datafile/2966689", "filename": "v6_1911_pref_pgn_utf.zip"},
    #     ],
    # },
]

_size_cache: dict[str, int | None] = {}
_NO_EXTRACT = {f.get("filename") or os.path.basename(urllib.parse.urlparse(f["url"]).path)
               for r in DOWNLOAD_RESOURCES for f in r["files"] if f.get("extract") is False}


def _remote_size(url: str):
    if url in _size_cache:
        return _size_cache[url]
    try:
        # Explicit User-Agent: HydroSHEDS' Cloudflare front rejects urllib's default one.
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "map-tool/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            size = int(resp.headers.get("Content-Length", 0)) or None
    except Exception:
        size = None
    if size is None:
        # Some hosts refuse HEAD (Dataverse redirects to S3 URLs signed for
        # GET only): ask for the first byte and read the total from
        # Content-Range.
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "map-tool/1.0", "Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                total = resp.headers.get("Content-Range", "").rpartition("/")[2]
                size = int(total) if total.isdigit() else None
        except Exception:
            size = None
    _size_cache[url] = size
    return size


def _filename_for(f: dict) -> str:
    """Local file name of a DOWNLOAD_RESOURCES file entry: its explicit
    "filename" if given (URLs like Dataverse's .../datafile/<id>), else the
    URL's last path component."""
    return f.get("filename") or os.path.basename(urllib.parse.urlparse(f["url"]).path)


def _local_path(resource_id: str, filename: str):
    return DATA_DIR / resource_id / filename


def _local_size(resource_id: str, filename: str):
    p = _local_path(resource_id, filename)
    return p.stat().st_size if p.exists() else None


def _extracts(filename: str) -> bool:
    """Whether a downloaded file is unzipped in place (every .zip except the
    DOWNLOAD_RESOURCES entries marked "extract": False)."""
    return filename.lower().endswith(".zip") and filename not in _NO_EXTRACT


def _extract_marker(dest):
    return dest.parent / f".{dest.name}.extracted"


def _is_present(resource_id: str, filename: str) -> bool:
    """Whether this file's data is already usable in the workspace, without
    any network call. A downloaded .zip only counts once it has actually
    been extracted (see `_extract_zip`) -- that's the point of extracting:
    the tool should find its data already sitting there, ready to use."""
    dest = _local_path(resource_id, filename)
    if _extracts(filename):
        return _extract_marker(dest).exists()
    return dest.exists() and dest.stat().st_size > 0


# ---- background download manager -----------------------------------------
# Downloads run in a small worker pool and write straight into DATA_DIR, so
# the Download tab's buttons trigger a server-side fetch (with progress the
# UI can poll) instead of a plain browser download.

_dl_state: dict[str, dict] = {}   # "<resource_id>/<filename>" -> {status, bytes, total, error}
_dl_lock = threading.Lock()
_dl_queued: set[str] = set()
_dl_pool = ThreadPoolExecutor(max_workers=4)


def _set_state(key: str, **kwargs):
    with _dl_lock:
        _dl_state.setdefault(key, {"status": "pending", "bytes": 0, "total": None, "error": None}).update(kwargs)


def _get_state(key: str) -> dict:
    with _dl_lock:
        return dict(_dl_state.get(key, {"status": "idle", "bytes": 0, "total": None, "error": None}))


def _extract_zip(dest, key: str):
    """Extract an archive in place (alongside itself) so its contents are
    directly usable, tracking per-entry progress the same way download bytes
    are tracked."""
    with zipfile.ZipFile(dest) as zf:
        infos = zf.infolist()
        total = sum(i.file_size for i in infos) or None
        _set_state(key, status="extracting", bytes=0, total=total)
        done = 0
        for info in infos:
            zf.extract(info, dest.parent)
            done += info.file_size
            _set_state(key, bytes=done)
    _extract_marker(dest).touch()


def _download_worker(resource_id: str, filename: str, url: str):
    key = f"{resource_id}/{filename}"
    dest = _local_path(resource_id, filename)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        expected = _remote_size(url)
        if dest.exists() and expected is not None and dest.stat().st_size == expected:
            # Already fully fetched (e.g. a zip downloaded but not yet
            # extracted) -- skip straight to using it.
            downloaded = dest.stat().st_size
            _set_state(key, status="downloading", bytes=downloaded, total=downloaded, error=None)
        else:
            _set_state(key, status="downloading", bytes=0, total=None, error=None)
            req = urllib.request.Request(url, headers={"User-Agent": "map-tool/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                total = int(resp.headers.get("Content-Length") or 0) or None
                _set_state(key, total=total)
                downloaded = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(262144)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        _set_state(key, bytes=downloaded)
            os.replace(tmp, dest)

        if _extracts(filename):
            _extract_zip(dest, key)

        _set_state(key, status="done", bytes=downloaded, total=downloaded)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        _set_state(key, status="error", error=str(e))
    finally:
        with _dl_lock:
            _dl_queued.discard(key)


def _queue_download(resource_id: str, filename: str, url: str):
    key = f"{resource_id}/{filename}"
    if _is_present(resource_id, filename):
        size = _local_size(resource_id, filename)
        _set_state(key, status="done", bytes=size or 0, total=size)
        return key
    with _dl_lock:
        if key in _dl_queued:
            return key
        _dl_queued.add(key)
    _dl_pool.submit(_download_worker, resource_id, filename, url)
    return key


@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.get("/api/downloads")
def api_downloads():
    # Files with a listed size aren't asked (a WFS export would be built
    # just to answer a HEAD).
    all_urls = [f["url"] for r in DOWNLOAD_RESOURCES for f in r["files"] if not f.get("size_bytes")]
    with ThreadPoolExecutor(max_workers=min(16, len(all_urls))) as ex:
        sizes = dict(zip(all_urls, ex.map(_remote_size, all_urls)))

    result = []
    for r in DOWNLOAD_RESOURCES:
        files = []
        for f in r["files"]:
            filename = _filename_for(f)
            size = sizes.get(f["url"]) or f.get("size_bytes")
            key = f'{r["id"]}/{filename}'
            state = _get_state(key)
            downloaded = state["status"] == "done" or _is_present(r["id"], filename)
            files.append({**f, "filename": filename, "size_bytes": size, "key": key,
                          "downloaded": downloaded, "state": state})
        sized = [f["size_bytes"] for f in files if f["size_bytes"] is not None]
        total = sum(sized) if len(sized) == len(files) else None
        result.append({**r, "files": files, "total_size_bytes": total})
    return jsonify(result)


@app.post("/api/downloads/start")
def api_downloads_start():
    body = request.get_json(silent=True) or {}
    resource_id = body.get("resource_id", "all")
    label = body.get("label")  # optional: restrict to one file within a resource

    keys = []
    for r in DOWNLOAD_RESOURCES:
        if resource_id != "all" and r["id"] != resource_id:
            continue
        for f in r["files"]:
            if label is not None and f.get("label") != label:
                continue
            filename = _filename_for(f)
            keys.append(_queue_download(r["id"], filename, f["url"]))
    return jsonify({"queued": keys})


@app.get("/api/downloads/status")
def api_downloads_status():
    with _dl_lock:
        return jsonify(dict(_dl_state))


# ---------- version / restart ----------
# The tool's code as it was when this server started (name / size / mtime of
# the .py files and the page): the page compares it with what's on disk now
# (code changed -> offers to restart the server) and with what it was loaded
# with (server restarted with new code -> offers to reload the page).
_ROOT = Path(__file__).resolve().parent
_BOOT_ID = f"{os.getpid()}-{time.time():.0f}"


def _code_files():
    out = {}
    for path in sorted(list(_ROOT.glob("*.py")) + list((_ROOT / "templates").glob("*.html"))):
        try:
            st = path.stat()
            out[path.relative_to(_ROOT).as_posix()] = (st.st_size, st.st_mtime_ns)
        except OSError:
            pass
    return out


def _version_of(files):
    return result_cache.digest(sorted(files.items()))[:12]


_RUNNING_FILES = _code_files()
_RUNNING_VERSION = _version_of(_RUNNING_FILES)


@app.get("/api/version")
def api_version():
    now = _code_files()
    changed = sorted(n for n in set(now) | set(_RUNNING_FILES) if now.get(n) != _RUNNING_FILES.get(n))
    return jsonify({"boot": _BOOT_ID, "running": _RUNNING_VERSION, "on_disk": _version_of(now),
                    "changed_files": changed})


@app.post("/api/restart")
def api_restart():
    """Starts a fresh server (it waits for the port, see __main__) and stops
    this one: picks up the tool's current code. The new process gets its own
    console window when this one has one (started from start_webui.bat),
    else appends to flask.log."""
    import subprocess
    import sys
    env = dict(os.environ, MAP_TOOL_WAIT_PORT="1")
    kwargs = {"cwd": str(_ROOT), "env": env}
    console = False
    try:
        console = sys.stdout is not None and sys.stdout.isatty()
    except (AttributeError, ValueError):
        pass
    if os.name == "nt":
        if console:
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
        else:
            kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    if not console:
        log = open(_ROOT / "flask.log", "ab")
        kwargs.update(stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    subprocess.Popen([sys.executable, str(_ROOT / "app.py")], **kwargs)

    def _die():
        time.sleep(0.5)
        os._exit(0)
    threading.Thread(target=_die, daemon=True).start()
    return jsonify({"status": "restarting", "boot": _BOOT_ID})


@app.post("/api/shutdown")
def api_shutdown():
    def _die():
        time.sleep(0.3)
        os._exit(0)
    threading.Thread(target=_die, daemon=True).start()
    return jsonify({"status": "stopping"})


@app.get("/api/countries")
def api_countries():
    try:
        return jsonify(core.list_countries())
    except core.DataNotFoundError as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/subdivisions")
def api_subdivisions():
    """GADM level-1 subdivisions of one country: ?tag=USA."""
    tag = request.args.get("tag", "").upper()
    try:
        if tag not in core.load_geometries():
            return jsonify({"error": f"Unknown country '{tag}'"}), 400
        return jsonify({"tag": tag, "subdivisions": core.list_subdivisions(tag)})
    except core.DataNotFoundError as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/subdivisions/search")
def api_subdivisions_search():
    """Subdivisions of every country matching ?q= (name, alternative names, country)."""
    return jsonify({"results": core.search_subdivisions(request.args.get("q", ""))})


@app.get("/api/projections")
def api_projections():
    return jsonify([{"id": pid, "label": p["label"]} for pid, p in core.PROJECTIONS.items()])


@app.get("/api/dimensions")
def api_dimensions():
    return jsonify(core.allowed_output_sizes())


def _parse_list(param):
    raw = request.args.get(param, "")
    return [t for t in raw.split(",") if t]


# The UI fetches a view's meta, then its image (and the Land/Sea tab its
# preview) as separate requests; caching the last few renders means one
# render per view instead of one per request. Keyed on the render params +
# the saved Land/Sea settings, so saving new settings re-renders.
_RENDER_PARAMS = ("tags", "projection", "margin", "excluded", "size", "mainland", "merge", "world", "center",
                  "antarctica", "edges", "edge_px", "subx", "subi")
_RENDER_CACHE_SIZE = 3
_render_cache = OrderedDict()
_render_lock = threading.Lock()


# ---- results cache (memory + disk, see result_cache.py). Every stage's
# result is found by a content key: its settings, the content keys of the
# results it is made from (`_ck`: stamped on the result images), its code
# and the data files it reads -- so results survive restarts and a change
# anywhere recomputes exactly what depends on it.
def _ck(obj):
    """Content key of a result image (None for None)."""
    if obj is None:
        return None
    k = getattr(obj, "_ck", None)
    if k is None:
        raise RuntimeError("a result image without a content key")
    return k


def _stamp(obj, k):
    if obj is not None:
        obj._ck = k  # PIL images take attributes


_data_sig_memo = {}


def _data_sig(*dirs):
    """Name / size / mtime of the files directly in data/<dir> (and the
    given extra files): results are recomputed when a dataset changes.
    Memoized for a few seconds (a request asks for it several times)."""
    key = dirs
    now = time.time()
    if key in _data_sig_memo and now - _data_sig_memo[key][0] < 5:
        return _data_sig_memo[key][1]
    out = []
    for d in dirs:
        path = DATA_DIR / d if not str(d).endswith((".json", ".shp")) else Path(core.__file__).resolve().parent / d
        try:
            if path.is_dir():
                out.append([d, sorted((e.name, e.stat().st_size, e.stat().st_mtime_ns)
                                      for e in os.scandir(path) if e.is_file())])
            else:
                st = path.stat()
                out.append([d, st.st_size, st.st_mtime_ns])
        except OSError:
            out.append([d, None])
    _data_sig_memo[key] = (now, out)
    return out


def _stage_get(mem, size, stage, k, compute):
    """A stage's result: memory (LRU of `size`) -> disk -> compute() (then
    saved). Call with _render_lock held."""
    if k in mem:
        mem.move_to_end(k)
        return mem[k]
    entry = result_cache.load(stage, k)
    if entry is None:
        entry = compute()
        result_cache.save(stage, k, entry)
    mem[k] = entry
    while len(mem) > size:
        mem.popitem(last=False)
    return entry


def _int_arg(name, default):
    try:
        return int(request.args.get(name, default) or default)
    except ValueError:
        raise core.RenderError(f"'{name}' must be a whole number")


def _render_detailed_from_args():
    """(image, meta, land/sea debug) for the view described by the request's
    query params (shared by the render, rivers and Land/Sea endpoints)."""
    land_sea = core.load_land_sea_settings()
    color_config = palettes.load_config()
    k = result_cache.key("render", [request.args.get(p, "") for p in _RENDER_PARAMS], land_sea, color_config,
                         _data_sig("gadm", "hydrosheds"))
    with _render_lock:
        result = _stage_get(_render_cache, _RENDER_CACHE_SIZE, "render", k, lambda: core.render_map_detailed(
            _parse_list("tags"),
            request.args.get("projection", "miller"),
            margin_percent=float(request.args.get("margin", core.DEFAULT_MARGIN_PERCENT)),
            excluded_tags=_parse_list("excluded"),
            output_size=core.parse_output_size(request.args.get("size", "auto")),
            ignore_overseas=request.args.get("mainland", "0") in ("1", "true"),
            land_sea=land_sea,
            color_config=color_config,
            merge_autonomous=request.args.get("merge", "0") in ("1", "true"),
            world=request.args.get("world", "0") in ("1", "true"),
            center_tag=request.args.get("center") or None,
            include_antarctica=request.args.get("antarctica", "0") in ("1", "true"),
            edge_bands=request.args.get("edges", "0") in ("1", "true"),
            edge_band_px=_int_arg("edge_px", core.EDGE_BAND_PX_DEFAULT),
            excluded_subdivisions=_parse_list("subx"),
            included_subdivisions=_parse_list("subi"),
        ))
        _stamp(result[0], k)
        return result


def _render_from_args():
    img, meta, _debug = _render_detailed_from_args()
    return img, meta


def _png_response(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.get("/api/render/meta")
def api_render_meta():
    try:
        img, meta, debug = _render_detailed_from_args()
        if request.args.get("eez") in ("1", "true"):
            _eimg, rep = _eez_overlay(img, debug)
            meta = dict(meta, eez=rep)
    except (core.RenderError, core.DataNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(meta)


# EEZ overlay (eez=1&eez_opacity=0.35): the last 4, keyed on the country map
# (held by the entry) -- the render itself is left untouched, other tabs find
# the sea by its colour.
_eez_cache = OrderedDict()


def _eez_overlay(img, debug):
    """(country map with the EEZ tint on its sea, report); unchanged map +
    {"available": False} without the data."""
    if not eez.available():
        return img, {"available": False}
    try:
        opacity = float(request.args.get("eez_opacity", eez.OPACITY_DEFAULT))
    except ValueError:
        raise core.RenderError("The EEZ opacity must be a number")
    if not 0.0 < opacity <= 1.0:
        raise core.RenderError("The EEZ opacity must be above 0 and at most 1")
    excluded = tuple(sorted(_parse_list("excluded")))
    min_region = core.load_land_sea_settings().get("min_region_px", 8)
    k = result_cache.key("eez_overlay", _ck(img), round(opacity, 3), excluded, min_region, _data_sig("eez"))

    def compute():
        out, rep = eez.overlay(img, debug, excluded, opacity, min_region)
        rep["available"] = True
        return out, rep
    with _render_lock:
        entry = _stage_get(_eez_cache, 4, "eez_overlay", k, compute)
        _stamp(entry[0], k)
        return entry


@app.get("/api/render/image")
def api_render_image():
    """The country map; eez=1 -> with the EEZ tint on the sea; download=1 ->
    as a file."""
    try:
        img, _meta, debug = _render_detailed_from_args()
        if request.args.get("eez") in ("1", "true"):
            img, _rep = _eez_overlay(img, debug)
    except (core.RenderError, core.DataNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    if request.args.get("download") in ("1", "true"):
        return send_file(_image_buffer(img, "PNG"), mimetype="image/png", as_attachment=True,
                         download_name="countries_eez.png" if request.args.get("eez") in ("1", "true") else "countries.png")
    return _png_response(img)


# ---------- country colours: presets + per-country overrides ----------

def _colors_state():
    cfg = palettes.load_config()
    colors, nudged = palettes.effective_colors(core.load_geometries(), cfg)
    return {
        "active": cfg,
        "presets": palettes.list_presets(),
        "nudged": nudged,
        "autonomous": palettes.AUTONOMOUS_PARENT,
    }


@app.get("/api/colors")
def api_colors():
    return jsonify(_colors_state())


@app.post("/api/colors/set")
def api_colors_set():
    """Body {"tag": "FRA", "rgb": [r, g, b]} sets a country's colour;
    "rgb": null removes the override (back to the preset / tool colour)."""
    body = request.get_json(force=True) or {}
    tag = str(body.get("tag", "")).upper()
    if tag not in core.load_geometries():
        return jsonify({"error": f"Unknown country '{tag}'"}), 400
    cfg = palettes.load_config()
    try:
        if body.get("rgb") is None:
            cfg["overrides"].pop(tag, None)
        else:
            cfg["overrides"][tag] = palettes.valid_rgb(body["rgb"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    palettes.save_config(cfg)
    colors, nudged = palettes.effective_colors(core.load_geometries(), cfg)
    return jsonify({**_colors_state(), "tag": tag, "color": core.rgb_to_hex(colors[tag]),
                    "was_nudged": tag in nudged})


@app.post("/api/colors/load")
def api_colors_load():
    name = str((request.get_json(force=True) or {}).get("name", ""))
    try:
        palettes.get_preset(name)
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    palettes.save_config({"preset": name, "overrides": {}})
    return jsonify(_colors_state())


@app.post("/api/colors/save")
def api_colors_save():
    """Save the current effective colours of every country as a preset."""
    name = str((request.get_json(force=True) or {}).get("name", "")).strip()
    cfg = palettes.load_config()
    colors, _nudged = palettes.effective_colors(core.load_geometries(), cfg)
    try:
        palettes.save_user_preset(name, {t: list(c) for t, c in colors.items()})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    palettes.save_config({"preset": name, "overrides": {}})
    return jsonify(_colors_state())


@app.delete("/api/colors/presets/<path:name>")
def api_colors_delete(name):
    try:
        palettes.delete_user_preset(name)
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    cfg = palettes.load_config()
    if cfg.get("preset") == name:
        palettes.save_config({"preset": palettes.DEFAULT_PRESET, "overrides": cfg.get("overrides", {})})
    return jsonify(_colors_state())


@app.get("/api/landsea/settings")
def api_landsea_settings():
    return jsonify({
        "settings": core.load_land_sea_settings(),
        "defaults": core.normalize_land_sea({}),
        "limits": core._LAND_SEA_LIMITS,
        "colors": {k: core.rgb_to_hex(v) for k, v in core.PREVIEW_COLORS.items()},
        "mask_colors": {k: core.rgb_to_hex(v) for k, v in core.MASK_COLORS.items()},
        "lakes_available": core.LAKES_SHAPEFILE_PATH.exists(),
    })


@app.post("/api/landsea/settings")
def api_landsea_save():
    try:
        settings = core.save_land_sea_settings(request.get_json(force=True) or {})
    except core.RenderError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"settings": settings})


@app.get("/api/landsea/preview")
def api_landsea_preview():
    """mode=steps (default): land/sea with every step highlighted;
    mode=mask: plain land white / sea black / lakes blue."""
    try:
        img, meta, debug = _render_detailed_from_args()
    except (core.RenderError, core.DataNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    if request.args.get("mode") == "mask":
        return _png_response(core.land_sea_mask(img, debug))
    return _png_response(core.land_sea_preview(img, meta, debug))


def _warm_coast_cache():
    """Classify every country's coastline edges in the background, so the
    first strait-fixed render of a wide view doesn't pay for dozens of
    countries at once (~0.3-2 s each, cached on disk afterwards)."""
    try:
        geometries = core.load_geometries()
        for tag in geometries:
            core.coastal_edge_masks(geometries, tag)
        print("Coastline cache ready.")
    except core.DataNotFoundError:
        pass


def _warm_hydro_indexes():
    """Build the HydroLAKES / HydroRIVERS indexes in the background (~1 s +
    ~6 s), so the first lake cut or river map doesn't wait for them."""
    core.load_lake_index()
    index = core.load_river_index()
    if index is not None:
        core.river_dead_ends(index)
        core.river_borders(core.load_geometries(progress=lambda *a: None), index)
        print("River index ready.")


# ---------- rivers tab ----------
# The tab fetches meta, then the image, for one view + percentage: keep the
# last couple of river maps so that's one river pass, not two.
_river_cache = OrderedDict()


def _rivers_from_args():
    """Query params: pct=af:10,eu:5,as:1,... (per continent, see
    core.RIVER_CONTINENTS; missing ones use top=, default 10), minlen=
    (px, rivers shorter than that are removed; 0 keeps all), and auto=1 +
    spacing= (px) to pick each continent's percentage from river density
    instead of pct; lakes=0 / sea=0 / deadends=0 / borders=0 turn off
    drawing rivers across lakes, extending mouths to the sea, dropping
    rivers that end inland outside a lake, and fitting river borders.
    Returns (river map, report, meta, country map with borders fitted to
    the rivers)."""
    names = {code: name for code, name, _r in core.RIVER_CONTINENTS}
    try:
        if request.args.get("top"):
            pct = {code: float(request.args["top"]) for code in names}
        else:  # the user's per-continent defaults
            pct = {code: core.RIVER_TOP_PERCENT_BY_CONTINENT.get(code, core.RIVER_TOP_PERCENT_DEFAULT) for code in names}
        for item in filter(None, request.args.get("pct", "").split(",")):
            code, value = item.split(":")
            if code not in names:
                raise core.RenderError(f"Unknown continent '{code}'")
            pct[code] = float(value)
        min_len = int(request.args.get("minlen", core.RIVER_MIN_LENGTH_DEFAULT))
        spacing = None
        if request.args.get("auto", "0") in ("1", "true"):
            spacing = float(request.args.get("spacing", core.RIVER_AUTO_SPACING_DEFAULT))
            if not 2 <= spacing <= 2000:
                raise core.RenderError("The target river spacing must be between 2 and 2000 px")
    except ValueError:
        raise core.RenderError("River percentages and the minimum length must be numbers")
    for code, value in pct.items():
        if not 0 < value <= 100:
            raise core.RenderError(f"{names[code]}: the river percentage must be above 0 and at most 100")
    if not 0 <= min_len <= 10000:
        raise core.RenderError("The minimum river length must be between 0 and 10000 px")
    flags = {name: request.args.get(param, "1") not in ("0", "false")
             for name, param in (("through_lakes", "lakes"), ("reach_sea", "sea"), ("drop_dead_ends", "deadends"),
                                 ("fit_borders", "borders"))}
    img, meta, debug = _render_detailed_from_args()
    k = result_cache.key("rivers", _ck(img), sorted(pct.items()), min_len, spacing, sorted(flags.items()),
                         _data_sig("hydrosheds", "river_borders.json"))

    def compute():
        rivers, report = core.render_rivers(img, debug, pct, min_len, spacing, **flags)
        countries = report.pop("countries_image", None)
        return rivers, report, (None if countries is img else countries)
    with _render_lock:
        rivers, report, countries = _stage_get(_river_cache, 2, "rivers", k, compute)
        _stamp(rivers, k + ":rivers")
        _stamp(countries, k + ":countries")
        return rivers, report, meta, (countries if countries is not None else img)


@app.get("/api/rivers/settings")
def api_rivers_settings():
    return jsonify({"continents": [{"code": c, "name": n} for c, n, _r in core.RIVER_CONTINENTS],
                    "default_percent": core.RIVER_TOP_PERCENT_DEFAULT,
                    "default_percents": core.RIVER_TOP_PERCENT_BY_CONTINENT,
                    "default_auto": core.RIVER_AUTO_DEFAULT,
                    "default_min_length": core.RIVER_MIN_LENGTH_DEFAULT,
                    "default_spacing": core.RIVER_AUTO_SPACING_DEFAULT})


@app.get("/api/rivers/meta")
def api_rivers_meta():
    try:
        _img, report, meta, _countries = _rivers_from_args()
    except (core.RenderError, core.DataNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"rivers": report, "countries": meta["countries"], "width": meta["width"],
                    "height": meta["height"],
                    "colors": {"land": core.rgb_to_hex(core.RIVER_PALETTE[core.RIVER_LAND]),
                               "water": core.rgb_to_hex(core.RIVER_PALETTE[core.RIVER_WATER]),
                               "source": core.rgb_to_hex(core.RIVER_PALETTE[core.RIVER_SOURCE]),
                               "flow_in": core.rgb_to_hex(core.RIVER_PALETTE[core.RIVER_FLOW_IN]),
                               "widths": [core.rgb_to_hex(core.RIVER_PALETTE[i]) for i in core.RIVER_WIDTHS]}})


@app.get("/api/rivers/bmp")
def api_rivers_bmp():
    """The river map as HOI4's rivers.bmp: 8-bit indexed BMP, game palette."""
    try:
        img, _report, _meta, _countries = _rivers_from_args()
    except (core.RenderError, core.DataNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    buf.seek(0)
    return send_file(buf, mimetype="image/bmp", as_attachment=True, download_name="rivers.bmp")


@app.get("/api/rivers/countries")
def api_rivers_countries():
    """The country map of the river map's view with its river borders fitted
    to the drawn rivers (core._fit_borders_to_rivers) -- the one to use
    together with rivers.bmp. ?download=1 sends it as a file."""
    try:
        _img, _report, _meta, countries = _rivers_from_args()
        if request.args.get("eez") in ("1", "true"):
            _i, _m, debug = _render_detailed_from_args()
            countries, _rep = _eez_overlay(countries, debug)
    except (core.RenderError, core.DataNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    if request.args.get("download") in ("1", "true"):
        buf = io.BytesIO()
        countries.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png", as_attachment=True, download_name="countries_rivers.png")
    return _png_response(countries)


@app.get("/api/rivers/image")
def api_rivers_image():
    try:
        img, _report, _meta, _countries = _rivers_from_args()
    except (core.RenderError, core.DataNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    return _png_response(img)


# ---------- city tab ----------
# Last 2 city maps, keyed on the render image (held by the entry, so its id
# stays unique) + the city settings.
_city_cache = OrderedDict()
_CITY_PARAMS = ("target", "counts", "min_population", "classes", "class_method", "star_px", "min_dist_px",
                "island_px", "dist_exempt_classes", "pop_weight", "isolated_px", "by_region",
                "urban_areas", "labels", "name_set", "name_lang", "ascii_only", "label_px")


def _cities_from_args():
    """(city map, report, meta) for the request's view + city settings
    (query params named as cities.CITY_DEFAULTS; counts=FIN:30,NLD:5 = per-country overrides)."""
    settings = cities.normalize_settings({k: request.args.get(k) for k in _CITY_PARAMS})
    if not cities.available():
        raise core.RenderError("The city dataset isn't downloaded (Download tab -> GeoNames Cities)")
    img, meta, debug = _render_detailed_from_args()
    return _city_render(img, meta, debug, settings)


def _city_key(img, settings):
    return result_cache.key("cities", _ck(img), settings, _data_sig("cities"))


def _city_render(img, meta, debug, settings):
    k = _city_key(img, settings)
    with _render_lock:
        cimg, report = _stage_get(_city_cache, 2, "cities", k, lambda: cities.render_cities(img, debug, settings))
        _stamp(cimg, k)
        return cimg, report, meta


def _city_points_for_terrain(img, meta, debug, names=False):
    """The City tab's cities [(x, y, class)] for the request's city
    settings (sent along by the Terrain tab), or None without the dataset.
    Reuses the City tab's render if it's cached; else renders without
    names (they don't change the selection, and cost seconds).
    names=True: (x, y, class, name) -- GeoNames' main name."""
    if not cities.available():
        return None
    settings = cities.normalize_settings({k: request.args.get(k) for k in _CITY_PARAMS})
    k = _city_key(img, settings)
    with _render_lock:
        cached = k in _city_cache or result_cache.exists("cities", k)
    if not cached:
        settings = dict(settings, labels=False)
    _cimg, report, _meta = _city_render(img, meta, debug, settings)
    if names:
        return [(c["x"], c["y"], c["class"], c.get("label") or c["name"]) for c in report["cities"]]
    return [(c["x"], c["y"], c["class"]) for c in report["cities"]]


@app.get("/api/cities/settings")
def api_cities_settings():
    return jsonify({"continents": [{"code": c, "name": n} for c, n in cities.CONTINENTS],
                    "defaults": cities.CITY_DEFAULTS, "limits": cities.CITY_LIMITS,
                    "class_colors": [core.rgb_to_hex(c) for c in cities.CLASS_COLORS],
                    "exonym_langs": [{"code": c, "name": n} for c, n in city_names.EXONYM_LANGS],
                    "names": city_names.build_status()})


@app.get("/api/cities/meta")
def api_cities_meta():
    try:
        _cimg, report, meta = _cities_from_args()
    except (core.RenderError, core.DataNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    rep = {k: v for k, v in report.items() if k != "cities"}
    rep["merged_places"] = cities.load_city_index()["merged"]
    rep["top"] = [{k: c[k] for k in ("name", "country", "population", "class", "continent")} for c in report["cities"][:15]]
    return jsonify({"cities": rep, "countries": meta["countries"], "width": meta["width"], "height": meta["height"]})


@app.get("/api/cities/image")
def api_cities_image():
    """PNG of the city map; download=1 -> cities.png."""
    try:
        cimg, _report, _meta = _cities_from_args()
    except (core.RenderError, core.DataNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    if request.args.get("download") in ("1", "true"):
        return send_file(_image_buffer(cimg, "PNG"), mimetype="image/png", as_attachment=True,
                         download_name="cities.png")
    return _png_response(cimg)


@app.get("/api/cities/csv")
def api_cities_csv():
    """The drawn cities as CSV (largest class first): name, country,
    continent, class, population, feature code, lon/lat, pixel x/y,
    GeoNames id, alternate names."""
    try:
        _cimg, report, _meta = _cities_from_args()
    except (core.RenderError, core.DataNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    import csv as _csv
    buf = io.StringIO()
    cols = ["name", "country", "continent", "class", "population", "feature_code", "lon", "lat", "x", "y",
            "geoname_id", "merged_parts", "own_population", "label", "endonym", "exonym", "alternate_names"]
    wr = _csv.DictWriter(buf, fieldnames=cols, delimiter=";", extrasaction="ignore")
    wr.writeheader()
    wr.writerows(report["cities"])
    return send_file(io.BytesIO(buf.getvalue().encode("utf-8-sig")), mimetype="text/csv", as_attachment=True,
                     download_name="cities.csv")


# ---------- heightmap tab ----------
_height_cache = OrderedDict()


_height_levels_cache = OrderedDict()


def _heightmap_from_args():
    """(raw heightmap, report, meta, country image, HOI4 heightmap, normal
    map) for the request's view: the NASA topography reprojected onto the
    Countries tab's frame (the last 2 renders), then HOI4 levels + normal
    map per relief= strength (core.exaggerate_relief; 1 = off, last 4 kept)."""
    try:
        relief = float(request.args.get("relief", core.RELIEF_STRENGTH_DEFAULT))
    except ValueError:
        raise core.RenderError("The relief exaggeration must be a number")
    if not 1.0 <= relief <= core.RELIEF_STRENGTH_MAX:
        raise core.RenderError(f"The relief exaggeration must be between 1 (off) and {core.RELIEF_STRENGTH_MAX:g}")
    img, meta, debug = _render_detailed_from_args()
    hm, report = _raw_heightmap(img, debug)
    k = result_cache.key("height_levels", _ck(hm), round(relief, 3))

    def compute():
        rep = dict(report)
        hoi4 = core.hoi4_heightmap(hm, img, debug, relief_strength=relief, report=rep) \
            if report.get("available") else hm
        normal = core.hoi4_normal_map(hoi4) if report.get("available") else None
        return rep, (None if hoi4 is hm else hoi4), normal
    with _render_lock:
        rep, hoi4, normal = _stage_get(_height_levels_cache, 4, "height_levels", k, compute)
        return hm, rep, meta, img, (hoi4 if hoi4 is not None else hm), normal


@app.get("/api/heightmap/meta")
def api_heightmap_meta():
    try:
        _hm, report, meta, _img, _hoi4, normal = _heightmap_from_args()
    except (core.RenderError, core.DataNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"heightmap": report, "normal_size": list(normal.size) if normal else None,
                    "countries": meta["countries"], "world": meta["world"],
                    "projection_label": meta["projection_label"], "width": meta["width"], "height": meta["height"]})


@app.get("/api/heightmap/image")
def api_heightmap_image():
    """PNG of the heightmap (8-bit grey): mode=hoi4 (default) in HOI4's
    heightmap.bmp levels (core.hoi4_heightmap), mode=raw the reprojected
    NASA values, mode=normal the normal map (world_normal.bmp, half size);
    overlay=1 draws the country map's coastlines and borders on it in red
    (not on the normal map); download=1 sends it as a file."""
    try:
        raw, _report, _meta, img, hoi4, normal = _heightmap_from_args()
    except (core.RenderError, core.DataNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    mode = request.args.get("mode")
    if mode == "normal":
        if normal is None:
            return jsonify({"error": "The topography is not downloaded (Download tab -> Topography)"}), 400
        return _png_response(normal) if request.args.get("download") not in ("1", "true") else             send_file(_image_buffer(normal, "PNG"), mimetype="image/png", as_attachment=True,
                      download_name="world_normal.png")
    hm = raw if mode == "raw" else hoi4
    if request.args.get("overlay") in ("1", "true"):
        hm = core.heightmap_overlay(hm, img)
    if request.args.get("download") in ("1", "true"):
        buf = io.BytesIO()
        hm.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png", as_attachment=True, download_name="heightmap.png")
    return _png_response(hm)


@app.get("/api/heightmap/bmp")
def api_heightmap_bmp():
    """The HOI4 heightmap as the game's heightmap.bmp: 8-bit greyscale BMP."""
    try:
        _raw, report, _meta, _img, hoi4, _normal = _heightmap_from_args()
    except (core.RenderError, core.DataNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    if not report.get("available"):
        return jsonify({"error": "The topography is not downloaded (Download tab -> Topography)"}), 400
    buf = io.BytesIO()
    hoi4.save(buf, format="BMP")
    buf.seek(0)
    return send_file(buf, mimetype="image/bmp", as_attachment=True, download_name="heightmap.bmp")


def _image_buffer(img, fmt):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


@app.get("/api/heightmap/normal_bmp")
def api_heightmap_normal_bmp():
    """The normal map as the game's world_normal.bmp: 24-bit RGB BMP, half the
    heightmap's size like the vanilla file."""
    try:
        _raw, _report, _meta, _img, _hoi4, normal = _heightmap_from_args()
    except (core.RenderError, core.DataNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    if normal is None:
        return jsonify({"error": "The topography is not downloaded (Download tab -> Topography)"}), 400
    return send_file(_image_buffer(normal, "BMP"), mimetype="image/bmp", as_attachment=True,
                     download_name="world_normal.bmp")


# ---------- states tab ----------
_state_cache = OrderedDict()


_STATE_PARAMS = ("target_px", "levels", "rivers", "merge_px", "density_factor", "density_dense_factor",
                 "density_ref", "country_factors")


def _states_from_args(full=False):
    """(state map, report, meta) for the request's view + state settings
    (target_px=, levels=FIN:2,..., rivers=1 -> the Rivers tab's params come
    along: its fitted country map and its rivers are followed); last 2 kept.
    full=True: the whole cache entry (+ image, fitted country map, rivers,
    per-pixel state ids, render debug) for the Provinces tab."""
    settings = states.normalize_settings({k: request.args.get(k) for k in _STATE_PARAMS})
    if states.level_index(1) is None:
        raise core.RenderError("The GADM subdivisions aren't downloaded (Download tab -> GADM)")
    img, meta, debug = _render_detailed_from_args()
    rivers = None
    countries = img
    if settings["rivers"]:
        rivers, _rrep, _rmeta, countries = _rivers_from_args()
    # before taking the lock: _province_density takes it too (not re-entrant --
    # calling it inside deadlocked the States render)
    dens = _province_density(img, debug)
    k = result_cache.key("states", _ck(img), _ck(countries), _ck(rivers), settings,
                         _data_sig("gadm", "population", "density"))

    def compute():
        simg, report = states.render_states(countries, debug, rivers, settings, density_img=dens)
        labels = report.pop("_labels")
        return simg, report, labels
    with _render_lock:
        simg, report, labels = _stage_get(_state_cache, 2, "states", k, compute)
        _stamp(simg, k)
    if full:
        return simg, report, meta, img, countries, rivers, labels, debug
    return simg, report, meta


# ---------------------------------------------------------------- provinces tab
_province_cache = OrderedDict()


def _provinces_from_args(full=False):
    """(provinces image, report, definition.csv text, meta) for the request:
    view + Rivers / States / Terrain tabs' params + land_min / land_max /
    sea_min / sea_max; last 2 kept. The terrain comes from the Terrain tab's
    render when it can be made (else provinces are "plains", reported)."""
    settings = provinces.normalize_settings({k: request.args.get(k) for k in list(provinces.PROVINCE_DEFAULTS) + ["country_factors"]})
    simg, sreport, meta, img, countries, rivers, labels, debug = _states_from_args(full=True)
    # City provinces: the City tab's cities (its settings come along)
    city_points = _city_points_for_terrain(img, meta, debug, names=True) if settings["city_provinces"] else None
    timg = None
    terrain_note = None
    try:
        timg = _terrain_from_args()[0]
    except _Preparing:
        terrain_note = "the land cover is still being prepared (Terrain tab)"
    except (core.RenderError, core.DataNotFoundError) as e:
        terrain_note = str(e)
    k = result_cache.key("provinces", _ck(simg), _ck(timg), settings, terrain_note,
                         None if city_points is None else result_cache.digest(city_points),
                         _data_sig("density", "eez", "hydrosheds"))
    with _render_lock:
        cached = k in _province_cache or result_cache.exists("provinces", k)
    if not cached:
        zones = _province_eez(countries, debug)
        dens = _province_density(img, debug)

    def compute():
        lake_ids = core.lake_id_image(debug["view"], debug.get("lakes"),
                                      core.load_land_sea_settings().get("lake_min_px", 8))
        pimg, report, definition = provinces.render_provinces(countries, debug, labels, sreport, rivers, timg,
                                                              settings, eez_zones=zones, density_img=dens,
                                                              lake_ids=lake_ids, city_points=city_points)
        if terrain_note:
            report["terrain_source"] = "none (" + terrain_note + "): plains"
        states_csv = report.pop("_states_csv")
        extras = report.pop("_extras")
        return pimg, report, definition, states_csv, extras
    with _render_lock:
        pimg, report, definition, states_csv, extras = _stage_get(_province_cache, 2, "provinces", k, compute)
        _stamp(pimg, k)
    if full:
        return pimg, report, definition, meta, states_csv, extras, rivers
    return pimg, report, definition, meta


_prov_aux_cache = OrderedDict()


def _province_eez(countries, debug):
    """EEZ zone labels on the (fitted) country map for the sea provinces, or None."""
    if not eez.available():
        return None
    excluded = tuple(sorted(_parse_list("excluded")))
    min_region = core.load_land_sea_settings().get("min_region_px", 8)
    k = result_cache.key("province_eez", _ck(countries), excluded, min_region, _data_sig("eez"))

    def compute():
        lab, tags, _rep = eez.zones(countries, debug, excluded, min_region)
        return lab, tags
    with _render_lock:
        return _stage_get(_prov_aux_cache, 4, "province_eez", k, compute)


def _province_density(img, debug):
    """People / km2 on the frame (GHS-POP), or None without the data."""
    if not density.available():
        return None
    k = result_cache.key("density", _ck(img), _data_sig("density"))
    v = debug["view"]
    with _render_lock:
        return _stage_get(_prov_aux_cache, 4, "density", k,
                          lambda: density.density_on_view(v, v["width"], v["height"]))


def _province_view_image(pimg, extras, rivers):
    """The province map with the request's overlays: borders=country|state
    (black lines where the country / state -- as made of the provinces --
    changes, coasts included) and show_rivers=1 (the river map's rivers)."""
    mode = request.args.get("borders", "none")
    show_rivers = request.args.get("show_rivers") in ("1", "true")
    if mode not in ("country", "state") and not show_rivers:
        return pimg
    a = np.asarray(pimg).copy()
    if mode in ("country", "state"):
        ids = extras["country_px"] if mode == "country" else extras["state_px"]
        edge = np.zeros(ids.shape, dtype=bool)
        edge[:, :-1] |= ids[:, :-1] != ids[:, 1:]
        edge[:-1, :] |= ids[:-1, :] != ids[1:, :]
        a[edge] = (0, 0, 0)
    if show_rivers and rivers is not None:
        a[np.asarray(rivers) <= 11] = (0, 170, 255)
    return Image.fromarray(a, mode="RGB")


def _province_error(e):
    if isinstance(e, _Preparing):
        return _terrain_error(e)
    return jsonify({"error": str(e)}), 400


@app.get("/api/provinces/settings")
def api_provinces_settings():
    return jsonify({"defaults": provinces.PROVINCE_DEFAULTS, "limits": provinces.PROVINCE_LIMITS})


@app.get("/api/provinces/meta")
def api_provinces_meta():
    try:
        _pimg, report, _definition, meta = _provinces_from_args()
    except (core.RenderError, core.DataNotFoundError, _Preparing) as e:
        return _province_error(e)
    return jsonify({"provinces": report, "countries": meta["countries"], "width": meta["width"],
                    "height": meta["height"], "world": meta.get("world")})


@app.get("/api/provinces/image")
def api_provinces_image():
    """The province map (PNG) with the overlays asked (borders=country|state,
    show_rivers=1); mode=states -> the states as made of the provinces;
    download=1 -> as a file."""
    try:
        pimg, _report, _definition, _meta, _states_csv, extras, rivers = _provinces_from_args(full=True)
    except (core.RenderError, core.DataNotFoundError, _Preparing) as e:
        return _province_error(e)
    if request.args.get("mode") == "states":
        simg = extras["states_img"]
        if request.args.get("download") in ("1", "true"):
            return send_file(_image_buffer(simg, "PNG"), mimetype="image/png", as_attachment=True,
                             download_name="states_from_provinces.png")
        return _png_response(simg)
    pimg = _province_view_image(pimg, extras, rivers)
    if request.args.get("download") in ("1", "true"):
        return send_file(_image_buffer(pimg, "PNG"), mimetype="image/png", as_attachment=True,
                         download_name="provinces.png")
    return _png_response(pimg)


@app.get("/api/provinces/bmp")
def api_provinces_bmp():
    """The game's provinces.bmp (24-bit)."""
    try:
        pimg, _report, _definition, _meta = _provinces_from_args()
    except (core.RenderError, core.DataNotFoundError, _Preparing) as e:
        return _province_error(e)
    return send_file(_image_buffer(pimg, "BMP"), mimetype="image/bmp", as_attachment=True,
                     download_name="provinces.bmp")


@app.get("/api/provinces/states_csv")
def api_provinces_states_csv():
    """Province -> state table (province;type;state;state_gadm;state_name;country)."""
    try:
        _pimg, _report, _definition, _meta, states_csv, _extras, _rivers = _provinces_from_args(full=True)
    except (core.RenderError, core.DataNotFoundError, _Preparing) as e:
        return _province_error(e)
    return send_file(io.BytesIO(states_csv.encode("utf-8")), mimetype="text/csv", as_attachment=True,
                     download_name="province_states.csv")


@app.get("/api/provinces/definition")
def api_provinces_definition():
    """The game's definition.csv (id;R;G;B;type;coastal;terrain;continent)."""
    try:
        _pimg, _report, definition, _meta = _provinces_from_args()
    except (core.RenderError, core.DataNotFoundError, _Preparing) as e:
        return _province_error(e)
    return send_file(io.BytesIO(definition.encode("utf-8")), mimetype="text/csv", as_attachment=True,
                     download_name="definition.csv")


# ---------- strategic regions (from the Provinces tab's output) ----------
_strategic_cache = OrderedDict()


def _strategic_from_args():
    """(regions, report, region id per province, P, province types, map
    height) for the request: the Provinces tab's params + the strategic
    settings (sr_* params); last 2 kept."""
    try:
        settings = strategic.normalize_settings({k: request.args.get("sr_" + k) for k in strategic.SR_DEFAULTS})
    except ValueError as e:
        raise core.RenderError(str(e))
    pimg, _report, definition, _meta, states_csv, _extras, _rivers = _provinces_from_args(full=True)
    view = _render_detailed_from_args()[2]["view"]  # the frame, for the climate (weather)
    k = result_cache.key("strategic", _ck(pimg), settings, _data_sig("climate", "bathymetry"))

    def compute():
        regions, report, prov_region, P = strategic.build_regions(pimg, definition, states_csv, settings,
                                                                  view=view)
        ptype = strategic.parse_inputs(pimg, definition, None)[1]
        return regions, report, prov_region, P, ptype
    with _render_lock:
        regions, report, prov_region, P, ptype = _stage_get(_strategic_cache, 2, "strategic", k, compute)
    return regions, report, prov_region, P, ptype, P.shape[0]


@app.get("/api/cache")
def api_cache():
    """The results cache's use on disk (result_cache.usage)."""
    return jsonify(result_cache.usage())


@app.post("/api/cache/clear")
def api_cache_clear():
    """Deletes every cached result on disk and in memory."""
    with _render_lock:
        for c in (_render_cache, _eez_cache, _river_cache, _city_cache, _height_cache, _height_levels_cache,
                  _state_cache, _province_cache, _prov_aux_cache, _strategic_cache, _terrain_cache, _colormap_cache):
            c.clear()
    return jsonify({"deleted": result_cache.clear()})


@app.get("/api/strategic/settings")
def api_strategic_settings():
    return jsonify({"defaults": strategic.SR_DEFAULTS, "limits": strategic.SR_LIMITS})


@app.get("/api/strategic/meta")
def api_strategic_meta():
    try:
        regions, report, _pr, P, _pt, _h = _strategic_from_args()
    except (core.RenderError, core.DataNotFoundError, _Preparing) as e:
        return _province_error(e)
    rows = [{"id": g["id"], "name": g["name"], "kind": g["kind"], "provinces": len(g["provinces"]),
             "states": len(g["states"]), "px": g["px"], "countries": g["countries"],
             "naval_terrain": g.get("naval_terrain")} for g in regions]
    return jsonify({"strategic": report, "regions": rows, "width": P.shape[1], "height": P.shape[0]})


@app.get("/api/strategic/image")
def api_strategic_image():
    """Regions map (PNG): borders=regions|provinces, centres=1; download=1."""
    try:
        regions, _report, prov_region, P, ptype, _h = _strategic_from_args()
    except (core.RenderError, core.DataNotFoundError, _Preparing) as e:
        return _province_error(e)
    img = strategic.render_image(regions, prov_region, P, ptype, borders=request.args.get("borders", "regions"),
                                 centres=request.args.get("centres", "1") in ("1", "true"))
    if request.args.get("download") in ("1", "true"):
        return send_file(_image_buffer(img, "PNG"), mimetype="image/png", as_attachment=True,
                         download_name="strategic_regions.png")
    return _png_response(img)


@app.get("/api/strategic/zip")
def api_strategic_zip():
    """map/strategicregions/*.txt + map/weatherpositions.txt + localisation + CSV."""
    try:
        regions, _report, _pr, _P, _pt, height = _strategic_from_args()
    except (core.RenderError, core.DataNotFoundError, _Preparing) as e:
        return _province_error(e)
    return send_file(strategic.zip_bytes(regions, height), mimetype="application/zip", as_attachment=True,
                     download_name="strategic_regions.zip")


@app.get("/api/states/levels")
def api_states_levels():
    """Each visible country's GADM level choice + candidates, without drawing
    the states (on the plain country map: quick; the fitted one moves only a
    few border pixels) -- for the States tab's country list."""
    try:
        settings = states.normalize_settings({k: request.args.get(k) for k in _STATE_PARAMS})
        if states.level_index(1) is None:
            raise core.RenderError("The GADM subdivisions aren't downloaded (Download tab -> GADM)")
        img, meta, debug = _render_detailed_from_args()
        info = states.choose_levels(img, debug, settings, _province_density(img, debug))
    except (core.RenderError, core.DataNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    out = []
    for c in info["countries"]:
        c = dict(c, color=core.rgb_to_hex(debug["country_rgb"][c["tag"]]))
        c["mean_state_px"] = next(k["mean_px"] for k in c["candidates"] if k["level"] == c["level"])
        c["states"] = next(k["n"] for k in c["candidates"] if k["level"] == c["level"])
        out.append(c)
    return jsonify({"countries": sorted(out, key=lambda c: c["name"]), "target_px": settings["target_px"]})


@app.get("/api/states/meta")
def api_states_meta():
    try:
        _simg, report, meta = _states_from_args()
    except (core.RenderError, core.DataNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    rep = {k: v for k, v in report.items() if k != "states"}
    return jsonify({"states": rep, "countries": meta["countries"], "width": meta["width"], "height": meta["height"],
                    "world": meta.get("world"), "projection_label": meta.get("projection_label")})


@app.get("/api/states/image")
def api_states_image():
    """PNG of the state map; download=1 -> states.png."""
    try:
        simg, _report, _meta = _states_from_args()
    except (core.RenderError, core.DataNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    if request.args.get("download") in ("1", "true"):
        return send_file(_image_buffer(simg, "PNG"), mimetype="image/png", as_attachment=True,
                         download_name="states.png")
    return _png_response(simg)


@app.get("/api/states/csv")
def api_states_csv():
    """Every state: id, GADM id, name, country, level, pixels, population
    (GeoNames places), colour, centre, GADM ids merged into it."""
    try:
        _simg, report, _meta = _states_from_args()
    except (core.RenderError, core.DataNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    import csv
    buf = io.StringIO()
    fields = ["id", "gid", "name", "tag", "level", "px", "population", "color", "x", "y", "merged_from"]
    wr = csv.DictWriter(buf, fieldnames=fields)
    wr.writeheader()
    wr.writerows(report["states"])
    return send_file(io.BytesIO(buf.getvalue().encode("utf-8-sig")), mimetype="text/csv", as_attachment=True,
                     download_name="states.csv")


# ---------- terrain tab ----------
_terrain_cache = OrderedDict()


def _raw_heightmap(img, debug):
    """The NASA topography reprojected onto a render (shared by the
    Heightmap and Terrain tabs)."""
    k = result_cache.key("heightmap", _ck(img), _data_sig("topography"))
    with _render_lock:
        hm, report = _stage_get(_height_cache, 2, "heightmap", k, lambda: core.render_heightmap(debug))
        _stamp(hm, k)
        return hm, report


class _Preparing(Exception):
    pass


def _terrain_from_args():
    """(terrain P image, report, meta, raw land cover P image) for the
    request's view + terrain settings (query params named as
    terrain.TERRAIN_DEFAULTS); last 2 kept."""
    settings = terrain.normalize_settings({k: request.args.get(k) for k in terrain.TERRAIN_DEFAULTS})
    # Relief source: the Heightmap tab's heights, lower ranges exaggerated
    # by its current factor (relief=, as on the Heightmap tab).
    try:
        relief = float(request.args.get("relief", core.RELIEF_STRENGTH_DEFAULT))
    except ValueError:
        raise core.RenderError("The relief exaggeration must be a number")
    if not 1.0 <= relief <= core.RELIEF_STRENGTH_MAX:
        raise core.RenderError(f"The relief exaggeration must be between 1 (off) and {core.RELIEF_STRENGTH_MAX:g}")
    if not landcover.available():
        raise core.RenderError("The Copernicus land cover isn't downloaded (Download tab -> Copernicus Land Cover)")
    if not landcover.pyramid_ready():
        landcover.start_build()
        raise _Preparing()
    if not core.topography_available():
        raise core.RenderError("The topography isn't downloaded (Download tab -> NASA Topographic Heightmap)")
    img, meta, debug = _render_detailed_from_args()
    hm, _hm_report = _raw_heightmap(img, debug)
    # Urban discs at the City tab's largest cities (its settings come along).
    points = _city_points_for_terrain(img, meta, debug) if settings["city_discs"] else None
    k = result_cache.key("terrain", _ck(img), _ck(hm), settings, round(relief, 3),
                         None if points is None else result_cache.digest(points),
                         _data_sig("landcover", "landcover/pyramid", "cities"))

    def compute():
        height, _gain, _kpp = core.exaggerated_height(hm, img, debug, relief)
        timg, report, lcimg = terrain.render_terrain(img, debug, height, settings, city_points=points)
        report["cities_available"] = cities.available()
        report["trees_image"] = report.pop("_trees_image")
        report["relief_strength"] = relief
        return timg, report, lcimg
    with _render_lock:
        timg, report, lcimg = _stage_get(_terrain_cache, 2, "terrain", k, compute)
        _stamp(timg, k)
        return timg, report, meta, lcimg


_colormap_cache = OrderedDict()


def _colormap_from_args():
    """(RGBA colour map: RGB = satellite colours, A = city lights; report)
    for the request's terrain (its urban pixels light up) + colour map
    settings (colormap.COLORMAP_DEFAULTS names); last 2 kept, keyed on the
    terrain image (held by the entry)."""
    settings = colormap.normalize_settings({k: request.args.get(k) for k in colormap.COLORMAP_DEFAULTS})
    if not colormap.available():
        raise core.RenderError("The NASA Blue Marble isn't downloaded (Download tab -> NASA Blue Marble)")
    timg, _report, _meta, _lc = _terrain_from_args()
    img, _meta2, debug = _render_detailed_from_args()
    k = result_cache.key("colormap", _ck(timg), _ck(img), settings, _data_sig("blue_marble"))
    with _render_lock:
        return _stage_get(_colormap_cache, 2, "colormap", k,
                          lambda: colormap.render_colormap(img, debug, timg, settings))


def _terrain_error(e):
    if isinstance(e, _Preparing):
        st = landcover.build_status()
        return jsonify({"error": "Preparing the land cover (one-time, ~15 min)...", "preparing": True,
                        "status": st}), 503
    return jsonify({"error": str(e)}), 400


@app.get("/api/terrain/status")
def api_terrain_status():
    """Land cover pyramid state (see landcover.build_status); starts the
    build if it's missing and the source is there."""
    if landcover.available() and not landcover.pyramid_ready():
        landcover.start_build()
    return jsonify(landcover.build_status())


@app.get("/api/terrain/settings")
def api_terrain_settings():
    return jsonify({"defaults": {**terrain.TERRAIN_DEFAULTS, **colormap.COLORMAP_DEFAULTS},
                    "limits": {**terrain.TERRAIN_LIMITS, **colormap.COLORMAP_LIMITS},
                    "classes": [{"index": i, "name": n, "type": t, "color": core.rgb_to_hex(terrain._PAL[i])}
                                for i, n, t in terrain.TERRAIN_CLASSES]})


@app.get("/api/terrain/meta")
def api_terrain_meta():
    try:
        _timg, report, meta, _lc = _terrain_from_args()
    except (core.RenderError, core.DataNotFoundError, _Preparing) as e:
        return _terrain_error(e)
    report = {k: v for k, v in report.items() if k != "trees_image"}
    return jsonify({"terrain": report, "countries": meta["countries"], "world": meta["world"],
                    "projection_label": meta["projection_label"], "width": meta["width"], "height": meta["height"]})


@app.get("/api/terrain/image")
def api_terrain_image():
    """PNG of the terrain map (the game's palette), or with mode=landcover
    the raw Copernicus land cover on the same frame (indexed PNG: index =
    Copernicus class code, Copernicus' colours); download=1 -> terrain.png /
    landcover.png."""
    try:
        timg, _report, _meta, lcimg = _terrain_from_args()
    except (core.RenderError, core.DataNotFoundError, _Preparing) as e:
        return _terrain_error(e)
    mode = request.args.get("mode", "terrain")
    if mode in ("colormap", "lights"):
        try:
            cm, _rep = _colormap_from_args()
        except (core.RenderError, core.DataNotFoundError, _Preparing) as e:
            return _terrain_error(e)
        out = cm.convert("RGB") if mode == "colormap" else cm.getchannel("A")
        name = "colormap.png" if mode == "colormap" else "city_lights.png"
        if request.args.get("download") in ("1", "true"):
            return send_file(_image_buffer(out, "PNG"), mimetype="image/png", as_attachment=True, download_name=name)
        return _png_response(out)
    out, name = {"landcover": (lcimg, "landcover.png"),
                 "trees": (_report["trees_image"], "trees.png")}.get(mode, (timg, "terrain.png"))
    if request.args.get("download") in ("1", "true"):
        return send_file(_image_buffer(out, "PNG"), mimetype="image/png", as_attachment=True, download_name=name)
    return _png_response(out)


@app.get("/api/terrain/bmp")
def api_terrain_bmp():
    """The game's terrain.bmp: 8-bit indexed BMP with its palette."""
    try:
        timg, _report, _meta, _lc = _terrain_from_args()
    except (core.RenderError, core.DataNotFoundError, _Preparing) as e:
        return _terrain_error(e)
    return send_file(_image_buffer(timg, "BMP"), mimetype="image/bmp", as_attachment=True,
                     download_name="terrain.bmp")


@app.get("/api/terrain/colormap_dds")
def api_terrain_colormap_dds():
    """The game's colormap_rgb_cityemissivemask_a.dds (uncompressed BGRA,
    RGB = colour map, alpha = city lights)."""
    try:
        cm, _rep = _colormap_from_args()
    except (core.RenderError, core.DataNotFoundError, _Preparing) as e:
        return _terrain_error(e)
    return send_file(io.BytesIO(colormap.dds_bytes(cm)), mimetype="image/vnd-ms.dds", as_attachment=True,
                     download_name="colormap_rgb_cityemissivemask_a.dds")


@app.get("/api/terrain/trees_bmp")
def api_terrain_trees_bmp():
    """The game's trees.bmp (terrain.bmp / 3.41): 8-bit indexed BMP with its palette."""
    try:
        _timg, report, _meta, _lc = _terrain_from_args()
    except (core.RenderError, core.DataNotFoundError, _Preparing) as e:
        return _terrain_error(e)
    return send_file(_image_buffer(report["trees_image"], "BMP"), mimetype="image/bmp", as_attachment=True,
                     download_name="trees.bmp")

# ---------- HOI4 mod export (hoi4mod.py) ----------
_export_lock = threading.Lock()
_export_job = {"state": "idle"}


def _hoi4_sources():
    """Every tab's result for the request's view (the Strategic regions
    request = the Provinces tab's, which carries every other tab's params),
    as hoi4mod.build_mod wants them."""
    img, meta, debug = _render_detailed_from_args()
    rivers, _rrep, _rmeta, countries = _rivers_from_args()
    _hm, hrep, _m, _i, hoi4h, normal = _heightmap_from_args()
    if not hrep.get("available"):
        raise core.RenderError("The topography isn't downloaded (Download tab -> NASA Topographic Heightmap)")
    timg, trep, _tm, _lc = _terrain_from_args()
    try:
        cm, _crep = _colormap_from_args()
    except (core.RenderError, core.DataNotFoundError):
        cm = None
    pimg, _prep, definition, _pmeta, states_csv, _extras, _r = _provinces_from_args(full=True)
    _simg, srep, _sm = _states_from_args()
    regions, _strep, _pr, _P, _pt, _h = _strategic_from_args()
    city_list = None
    if cities.available():
        _ci, crep, _cm = _cities_from_args()
        city_list = crep["cities"]
    return {"meta": meta, "view": debug["view"], "rivers": rivers, "heightmap": hoi4h, "normal": normal,
            "terrain": timg, "trees": trep["trees_image"], "colormap": cm, "provinces_img": pimg,
            "definition": definition, "states_csv": states_csv, "states": srep["states"], "regions": regions,
            "cities": city_list, "thumbnail": countries}


def _hoi4_paths_from_args():
    """The folders chosen in the HOI4 mod tab (game_dir= / mod_dir=; empty =
    detect) -> hoi4mod.PATHS; returns a note per invalid choice."""
    notes = {}
    g = (request.args.get("game_dir") or "").strip().strip('"')
    m = (request.args.get("mod_dir") or "").strip().strip('"')
    if g and not hoi4mod.is_game_dir(g):
        notes["game_dir"] = "Not a Hearts of Iron IV folder (no map/default.map or hoi4.exe in it)"
        g = ""
    if m and not Path(m).is_dir():
        notes["mod_dir"] = "This folder doesn't exist"
        m = ""
    hoi4mod.PATHS["game"] = g or None
    hoi4mod.PATHS["mod"] = m or None
    return notes


@app.get("/api/hoi4/info")
def api_hoi4_info():
    """Game / mod folders (the chosen ones if valid, else detected), version,
    defaults, the export job."""
    notes = _hoi4_paths_from_args()
    game = hoi4mod.find_game_dir()
    detected = hoi4mod.detect_game_dir()
    return jsonify({"game_dir": str(game) if game else None,
                    "game_detected": str(detected) if detected else None,
                    "game_version": hoi4mod.game_version(game) if game else None,
                    "mod_dir": str(hoi4mod.mod_dir()), "mod_detected": str(hoi4mod.detect_user_dir() / "mod"),
                    "mod_dir_exists": hoi4mod.mod_dir().is_dir(), "notes": notes,
                    "defaults": hoi4mod.MOD_DEFAULTS, "limits": hoi4mod.MOD_LIMITS, "job": _export_job})


@app.post("/api/hoi4/browse")
def api_hoi4_browse():
    """A native folder picker on this machine (the server runs locally):
    {"path": chosen or null}. kind=game|mod picks the dialog's title."""
    kind = request.args.get("kind", "game")
    title = ("Select the Hearts of Iron IV folder (the one holding hoi4.exe)" if kind == "game"
             else "Select the Hearts of Iron IV mod folder (Documents/Paradox Interactive/Hearts of Iron IV/mod)")
    start = request.args.get("start") or ""
    code = ("import sys, tkinter as tk; from tkinter import filedialog; r = tk.Tk(); r.withdraw(); "
            "r.attributes('-topmost', True); p = filedialog.askdirectory(title=sys.argv[1], initialdir=sys.argv[2] or None, "
            "mustexist=True); sys.stdout.write(p or '')")
    try:
        out = subprocess.run([sys.executable, "-c", code, title, start], capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        return jsonify({"error": f"The folder picker failed: {e}"}), 500
    path = out.stdout.strip()
    return jsonify({"path": str(Path(path)) if path else None})


@app.post("/api/hoi4/export")
def api_hoi4_export():
    """Starts writing the mod (background): the request's view + every tab's
    params + mod_<setting>=; GET /api/hoi4/info follows it (job)."""
    try:
        settings = hoi4mod.normalize_settings({k: request.args.get("mod_" + k) for k in hoi4mod.MOD_DEFAULTS})
    except hoi4mod.ExportError as e:
        return jsonify({"error": str(e)}), 400
    notes = _hoi4_paths_from_args()
    if notes:
        return jsonify({"error": "; ".join(notes.values())}), 400
    game = hoi4mod.find_game_dir()
    if game is None:
        return jsonify({"error": "Hearts of Iron IV wasn't found: choose its folder (HOI4 mod tab)"}), 400
    with _export_lock:
        if _export_job.get("state") == "running":
            return jsonify({"error": "An export is already running"}), 409
        _export_job.clear()
        _export_job.update(state="running", step="Collecting every tab's results", started=time.time())
    query = request.query_string.decode()
    mod_dir = hoi4mod.mod_dir()
    if not mod_dir.is_dir():
        return jsonify({"error": f"The mod folder doesn't exist: {mod_dir} (choose it in the HOI4 mod tab)"}), 400
    out = mod_dir / hoi4mod.ascii_name(settings["name"])

    def run():
        try:
            with app.test_request_context("/?" + query):
                src = _hoi4_sources()
            rep = hoi4mod.build_mod(src, settings, out, game,
                                    progress=lambda msg: _export_job.update(step=msg))
            hoi4mod.install_descriptor(out, mod_dir)
            _export_job.update(state="done", step="Done", path=str(out), report={
                k: rep.get(k) for k in ("states", "impassable_states", "provinces", "strategic_regions",
                                        "victory_points", "railways", "straits", "problems", "warnings")},
                countries=len(rep.get("countries", [])), seconds=round(time.time() - _export_job["started"], 1))
        except _Preparing:
            _export_job.update(state="error", error="The land cover is still being prepared (Terrain tab)")
        except Exception as e:  # report anything to the page
            import traceback
            traceback.print_exc()
            _export_job.update(state="error", error=f"{type(e).__name__}: {e}")
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"started": True, "path": str(out)})


if __name__ == "__main__":
    try:
        print("Warming geometry cache...")
        core.load_geometries()
        threading.Thread(target=_warm_coast_cache, daemon=True).start()
        threading.Thread(target=_warm_hydro_indexes, daemon=True).start()
        if cities.available():  # City tab: CSV parse + merging city parts, ~1.5 s
            def _warm_cities():
                idx = cities.load_city_index()
                # Per-city names for the labels (from the cache, or built once
                # from alternateNamesV2.zip, ~10 s).
                city_names.load_names(lambda: idx["id"].tolist(), cities.CITIES_CSV_PATH, wait=True)
            threading.Thread(target=_warm_cities, daemon=True).start()
        landcover.start_build()  # one-time land cover pyramid for the Terrain tab (no-op if ready / not downloaded)
    except core.DataNotFoundError as e:
        print(f"WARNING: {e}")
        print("The Countries/Land-Sea/Rivers tabs won't work until GADM data is available; "
              "the Download tab still will.")
    if os.environ.pop("MAP_TOOL_WAIT_PORT", ""):
        # started by /api/restart: the old server is still exiting
        import socket
        for _ in range(150):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(("127.0.0.1", 5000))
                    break
                except OSError:
                    time.sleep(0.2)
    app.run(debug=False, port=5000)
