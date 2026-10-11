# EarthToHoI4

A tool using real-life geodata to automatically generate Hearts of Iron IV map mods.

Pick some countries, a projection and a frame; EarthToHoI4 builds every map
layer from real shapefiles and rasters — country borders (GADM), rivers and
lakes (HydroSHEDS), relief and sea depth (NASA), land cover (Copernicus),
cities (GeoNames), maritime zones (Marine Regions), population (JRC GHS),
climate (ERA5) — and writes a complete, launchable HOI4 mod: provinces,
states, strategic regions with weather, terrain, rivers, heightmap, textures,
countries with their real names and flags, and the base game's
country-specific content neutralised.

## What it makes

The web UI has one tab per step, each previewable and downloadable on its own:

| Tab | Output |
|---|---|
| Download | Fetches the datasets below into `data/` (once) |
| Land/Sea | Lakes, islands, straits and channels for the chosen frame |
| Countries | Flat-colour country map in Miller, Mercator, equirectangular, Robinson or Mollweide; optional EEZ overlay |
| Rivers | HOI4 `rivers.bmp` (1 px rivers, flow-ins, sources, width by discharge); borders that follow rivers are fitted to them |
| States | States from GADM subdivisions (levels 0–5), sized by area and population density, merged where too small |
| Heightmap | HOI4 `heightmap.bmp` and `world_normal.bmp`, with lower mountain ranges exaggerated |
| City | Cities by population class, with names (endonyms / exonyms) |
| Terrain | HOI4 `terrain.bmp`, `trees.bmp` and the colour map, from land cover and relief |
| Provinces | HOI4 `provinces.bmp` + `definition.csv`: provinces inside the states, sized by density, following rivers; sea provinces inside the future sea regions |
| Strategic regions | Land regions from whole states; sea regions following EEZ lines (coastal bands, straight-edged offshore cells); naval terrain from bathymetry; weather from climate normals |
| HOI4 mod | Everything above written as a mod into your HOI4 `mod` folder |

The exported mod contains its own country tags, GADM names and real flags,
non-aligned governments, one infantry division per country (content comes
later: no factories or resources yet), victory points from the cities,
buildings / unit positions, railways and supply nodes, sea crossings over
straits, a 1936 bookmark and localisation in every game language. The base
game's countries are not loaded and its country-specific files are neutralised
(`replace_path` + a dead-reference cleaner), so the mod launches on its own.

## Requirements

- Python 3.11 or newer
- Windows (tested), Linux or macOS
- ~20 GB of free disk space for the datasets
- Hearts of Iron IV (tested with 1.19) for the mod export — the tool reads a
  few of the game's files to neutralise and complete the mod

```
pip install -r requirements.txt
python src/app.py
```

Then open http://127.0.0.1:5000 (on Windows, `start_webui.bat` does both).
Start with the **Download** tab ("Download all"), then go through the tabs
left to right. The HOI4 mod tab finds the game and your mod folder by itself
(Steam, GOG, Microsoft Store); if it can't, choose them there.

The first terrain render prepares the land cover once, which takes about
15 minutes. Keep the tool open until it's done. If you download the land
cover yourself
([Zenodo](https://zenodo.org/records/3243509): the 2 GB
`..._discrete-classification_EPSG-4326.tif`), put it in `data/landcover/`.
Any file name works.

A command-line renderer for country maps also exists:
`python src/render_map.py FRA DEU ITA --projection mercator --margin 10`.

## Reporting a problem

Every export saves its **generation parameters** — every setting of every tab,
plus the tool's version, Python, OS, game version and datasets — as
`earthtohoi4_params.json` in the exported mod's folder, and in the tool's
`runs/` folder (the last 30 runs; a run that failed records its error and
traceback there). The **Save parameters** button (top right) downloads the
current settings at any time; **Load parameters** restores them.

If the tool or an exported mod crashes, open an issue and attach that file:
it lets the run be reproduced exactly
(`python src/run_params.py replay <file> [--export "Mod name"]`).

## Data sources

The tool downloads these datasets itself (Download tab). They keep their own
licences, which also apply to the maps and mods you make with them:

| Dataset | Used for | Licence |
|---|---|---|
| [GADM 3.6](https://gadm.org) | Country and subdivision borders | Free for academic and other non-commercial use; no redistribution of the data |
| [NASA Blue Marble](https://visibleearth.nasa.gov/collection/1484/blue-marble): topography, bathymetry, imagery | Heightmap, sea depth, colour map | NASA imagery, free to use with credit |
| [HydroSHEDS](https://www.hydrosheds.org) HydroRIVERS / HydroLAKES | Rivers, lakes | CC BY 4.0 |
| [Copernicus Global Land Cover](https://land.copernicus.eu/global/products/lc) (CGLS-LC100) | Terrain, forests | Free and open, with attribution |
| [GeoNames](https://www.geonames.org) | Cities, names, country info | CC BY 4.0 |
| [Marine Regions World EEZ v12](https://www.marineregions.org) | Exclusive economic zones | CC BY 4.0 |
| [JRC GHS-POP / GHS-DUC](https://ghsl.jrc.ec.europa.eu) | Population, density | CC BY 4.0 |
| ERA5 monthly climate normals, via the [World Bank Climate Change Knowledge Portal](https://climateknowledgeportal.worldbank.org) | Weather of strategic regions | CC BY 4.0 |
| [flagcdn.com](https://flagcdn.com) | Country flags | Free to use |

Built-in colour presets are taken from the `colors.txt` files of the base game
and of the mods Kaiserreich, Millennium Dawn, The New Order and Magna Europa.

## Licence

The code is under the MIT licence (see `LICENSE`). Hearts of Iron IV is a
trademark of Paradox Interactive; this project is not affiliated with or
endorsed by Paradox Interactive. It ships none of the game's files; the code
only embeds the colour palettes of the game's map bitmaps (rivers, terrain,
trees), which the output files must use. The mod export reads the rest from
your own installation.

See `DEVELOPMENT.md` for how the tool works inside.
