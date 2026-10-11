# Development notes

How EarthToHoI4 works inside, and the rules that took real bugs to learn.

## Layout

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
| `palettes.py`, `color_presets/` | Country colours and presets |
| `result_cache.py` | Disk cache of every stage's results |
| `river_borders.json`, `cache/` | Small derived data shipped with the tool (rebuilt automatically when missing or stale; `python build_cache.py --clean` rebuilds it all) |

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

## States

- Level per country: a level whose units are big enough on average (mean >=
  the merge size, <= 1.5 x the target) is used as it is, with no merging;
  else the coarsest level averaging under the (density-scaled) target, merged
  smallest-first within the same parent subdivision. A merged state keeps
  the most populated member's label.
- A GADM unit whose states fit under 1.5 x the target becomes one state.
- Every state one piece: a separate piece is kept only if most of it belongs
  to units whose body is in the main piece (real exclaves, islands).
- State names: the GADM division (departements for France), cleaned
  ("n.a." -> parent, Russian "rayon" dropped, Ukrainian raion adjectives ->
  their town); repeated names told apart by parent or compass direction;
  capital states that are the city itself take the capital's name.

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
- Neutralising: `replace_path` on the content folders, their generic files
  copied back cleaned; every other base-game script file either reduced to
  the definitions generic files use (country-specific files) or cleaned by
  `hoi4clean` (dead tags / focuses / decisions / events → triggers false,
  effects removed, gated definitions dropped). Exclusion lists that empty out
  lose only the list — dropping the generic AI division templates crashed the
  game mid-game.
- `check_mod` re-checks the written files; every 8-bit BMP is re-read and
  compared (Pillow once wrote a 255-colour palette short).

## Testing

There is no automated test suite yet. The checks used during development:
each tab reports its own rule checks (rivers, provinces topology, strategic
regions); the export runs `check_mod`. Launching the game with `-debug
-crash_data_log` and reading `logs/error.log` remains the final test.
