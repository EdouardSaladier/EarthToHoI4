# Development notes

How EarthToHoI4 works inside, and the rules that took real bugs to learn.

## Layout

All the code is in `src/` (the Python modules and `templates/index.html`);
the project root holds the docs, the launcher and what the tool reads and
writes: `data/` (datasets), `cache/`, `color_presets/`, `runs/`, the settings
files (`land_sea_settings.json`, `color_config.json`), `river_borders.json`.
`core.CODE_DIR` is `src/`, `core.BASE_DIR` the project root. Run with
`python src/app.py` (or `start_webui.bat`).


| File | Role |
|---|---|
| `app.py` | Flask server: the web UI, its JSON API, the download manager, the per-stage results cache, the HOI4 export job |
| `templates/index.html` | The whole web UI (one file, inline CSS/JS) |
| `core.py` | GADM reading, projections, rasterizing, the land/sea steps, rivers (`rivers.bmp`), heightmap, normal map |
| `eez.py` | Exclusive economic zones (Marine Regions) on the frame |
| `states.py` | States from GADM subdivisions; `population.py` (GHS-POP per subdivision) |
| `cities.py`, `city_names.py` | Cities (GeoNames), their selection, classes and names |
| `terrain.py`, `landcover.py`, `colormap.py` | `terrain.bmp` / `trees.bmp` from Copernicus land cover + relief; the colour map from Blue Marble |
| `density.py` | Population density grid (GHS-POP 1 km) |
| `provinces.py` | Provinces, sea areas, topology rules, `definition.csv`, `province_states.csv` |
| `strategic.py`, `bathymetry.py`, `climate.py` | Strategic regions, naval terrain, weather |
| `hoi4mod.py`, `hoi4clean.py`, `pdx.py` | The mod export; the cleaner for the base game's scripts; a Paradox script parser / writer |
| `modthumbnail.py`, `hoi4renderer/` | The mod's `thumbnail.png`; a copy of the standalone Hoi4renderer (the game's map shaders in software) |
| `palettes.py`, `color_presets/` | Country colours and presets |
| `result_cache.py` | Disk cache of every stage's results |
| `river_borders.json`, `cache/` | Small derived data shipped with the tool (rebuilt automatically when missing or stale; `python src/build_cache.py --clean` rebuilds it all) |

Datasets go to `data/<resource>/` (Download tab); nothing there is tracked.

## Pipeline

Every tab is a stage working on the previous ones' cached results:
render (country map) → rivers → states → heightmap → cities → terrain /
colour map → provinces → strategic regions → mod. The **results cache**
keys each stage on a hash of its settings, the source of the modules it
depends on, its inputs' keys and the datasets' file signatures, so a change
recomputes exactly the stages that depend on it and results survive
restarts. `templates/index.html` is read once at start: restart the server
after editing it (the header's refresh button does that).

## Rendering (core.py)

- No geometry simplification; a custom numpy scanline fill sets exactly the
  pixels whose centre is inside (OpenCV / PIL fills grew polygons and closed
  straits).
- Antimeridian: per-ring `np.unwrap`, the seam chosen in open sea, rings cut
  by the seam drawn on both sides.
- Land/sea steps, in order: lakes → island separation → strait fixing
  (Dijkstra channels between water bodies, widened to a minimum width) →
  island suppression / growth → microstates → sea specks → diagonal
  connections → region sieve → lake classification.
- Rivers: 1 px wide, never touching themselves or another river except a
  tributary at its confluence — enforced while drawing, never fixed after.
  Where a border follows a river, the border is moved to the river.
- No river network (a whole basin, 4-connected) bigger than the game loads
  quickly: the game walks each one at a cost that grows much faster than the
  network, and these maps are far finer than vanilla's, so a basin can end up
  one river the size of the map — measured, a 115,654-px network (623
  branches) left the game loading for over six minutes where the same map
  without rivers took 60 s. Over the Rivers tab's branch limit (default 25),
  the network is cut at its smallest confluences, with the fewest cuts that
  do it (one pixel each, hidden where a tributary meets its parent).

## States

- Level per country: a level whose units are big enough on average (mean >=
  the merge size, <= 1.5 x the target) is used as it is, with no merging;
  else the coarsest level averaging under the (density-scaled) target, merged
  smallest-first within the same parent subdivision. A merged state keeps
  the most populated member's label.
- A GADM unit whose states fit under 1.5 x the target becomes one state.
- Every state one piece: a separate piece is kept only if most of it belongs
  to units whose body is in the main piece (real exclaves, islands).
- Rivers cut provinces even where the branch cap deleted a pixel, and a
  river's source is extended to its province border so the province it ends
  in is cut too.
- State names: GADM's own name for the division the state came from, used
  verbatim -- that field is the romanized endonym (Bayern, Toscana, Fejer),
  so nothing is cleaned, shortened or disambiguated and repeated names are
  kept. The one exception: a division GADM leaves unnamed ("n.a.") takes its
  parent's name.

## Provinces

- Land provinces are parts of the states; sizes scaled by population
  density; organic (least-cost) splits along admin lines, rivers and terrain
  edges; river splits; optional city provinces.
- The sea is designed first, as **sea areas** = the future sea strategic
  regions: never across an EEZ line; large EEZ parts split into a coastal
  band (smoothed distance to land >= 8,000 px land masses) cut across into
  pieces, and offshore Voronoi cells; separated at straits; outside every EEZ,
  Voronoi cells. Enclosed sea bodies under 2,000 px become lakes.
- Topology, all enforced: every province one piece; no X crossings — also
  across the map's wrap (the game joins the last column to the first); no two
  provinces touching along two stretches; > 8 px; bounding box under the
  game's limit (thin strips cut to <= 128 px); a sea province's centre inside it.

## Strategic regions

Land regions are whole states, agglomerated per land mass towards vanilla's
sizes; sea regions are the sea areas. Every region's centre must lie inside
it (the 2x2 pixels around the pixel mean and the mean of its provinces'
centres): islands join a sea region only if that holds, otherwise regions are
cut along their provinces. Weather comes from ERA5 normals through formulas
fitted on vanilla; naval terrain from bathymetry, islands and coast distance.

## Mod export (hoi4mod.py)

- File names ASCII only; UTF-8 **with BOM only** for localisation `.yml`.
- `common/country_tags/00_countries.txt` replaces the base game's tags (its
  countries existing with empty histories crashed the game at start);
  `zz_dynamic_countries.txt` is kept.
- Positions (buildings, unit stacks): z = map height - 1 - row; a position
  needs its whole 3x3 at heightmap >= 96 (islets are raised); ports within
  ~100 px of the province centre (the coast pixel nearest it).
- `adjacencies.csv`: a sea crossing's "through" province must touch at least
  one of the two shores.
- Ambient objects: only the generic ones (wind, water, frame, logo).
- Map scale and day/night, computed per export from the map's frame:
  `zz_map_tool_scale.lua` sets `MAP_SCALE_PIXEL_TO_KM` to the map's km per
  pixel (at the frame's centre) and scales every define the game measures in
  pixels (radar ranges, railway gun ranges, interception distance, influence,
  AI invasion distances: x 7.114 / km-per-px; the air-superiority area scale
  / that squared). `zz_map_tool_daynight.lua` + patched copies of the
  player's own `gfx/FX/standardfuncsgfx.fxh` (Greenwich's x, the px of a
  full 360 degrees of longitude instead of the map width, the pole offsets =
  the frame's top / bottom latitudes) and `constants.fxh` (the tree-season
  band at vanilla's latitudes). A shader whose expected lines aren't found
  (another game version) is left out rather than written half-patched.
- Lighting: the game lights the terrain from a virtual sun at map row
  `SUN_LATITUDE` (from the bottom), `SUN_HEIGHT` above the map — vanilla's
  values put it over ~12.4°N, 600 px high. On another frame the far end of the
  map was lit at a grazing angle (the Sahara dark on an Africa map), so the
  export writes `common/defines/zz_map_tool_lighting.lua` placing the sun /
  moon over the same latitude, heights scaled by the map's px per degree
  (`map_lighting`). Lua comments are `--`: a `#` broke every define (crash).
- Neutralising: `replace_path` on the content folders, their generic files
  copied back cleaned; every other base-game script file either reduced to
  the definitions generic files use (country-specific files) or cleaned by
  `hoi4clean` (dead tags / focuses / decisions / events → triggers false,
  effects removed, gated definitions dropped). Exclusion lists that empty out
  lose only the list — dropping the generic AI division templates crashed the
  game mid-game.
- `thumbnail.png` (512x512): the mod that was just written, rendered by the
  vendored `hoi4renderer` the way the game draws the map zoomed out, cut to a
  centred square, with the mod's name across it in its own map font (white
  over a blurred black copy of the letters, as the game draws map names).
  Written last, since it reads the mod's finished map files and its
  `descriptor.mod`; about 20 s, and it falls back to the Countries tab's map
  rather than failing the export.
- `ABOUT_EarthToHoI4.txt` in every mod: made by the tool (link, date, code
  version, frame), the datasets used with links and licences (`DATA_CREDITS`),
  flags / fonts credits, the Paradox disclaimer, the parameters file.
- `check_mod` re-checks the written files; every 8-bit BMP is re-read and
  compared (Pillow once wrote a 255-colour palette short).

## Generation parameters (`run_params.py`)

Everything a run depends on, in one JSON file: the full request (the view +
every tab's settings + the mod's, as the export's query string), the
server-side settings (Land/Sea, colours, chosen game / mod folders), the page's
localStorage (`mapTool.*`, including the Countries view, `mapTool.view`), the
code version (content hash + git commit / local changes), Python, packages,
OS, game version, dataset files (name, size, date), the last server errors
(tracebacks, recorded through Flask's `got_request_exception` and the export
thread) and the run's result. Written by every export before it starts
(`runs/`, last 30) and into the mod folder, then completed with the result
(error + traceback + the step it failed at). `POST /api/params/export` /
`POST /api/params/import` back the header's Save / Load buttons (loading
applies the server settings, replaces the page's `mapTool.*` keys and
reloads). `python src/run_params.py replay <file> [--export NAME]` reruns a file
with its settings, restoring the local ones afterwards.

## Testing

There is no automated test suite yet. The checks used during development:
each tab reports its own rule checks (rivers, provinces topology, strategic
regions); the export runs `check_mod`. Launching the game with `-debug
-crash_data_log` and reading `logs/error.log` remains the final test.
