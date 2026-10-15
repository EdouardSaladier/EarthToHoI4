# EarthToHoI4 — manual

Every setting of the web UI, tab by tab, with the reasoning the interface
itself leaves out. The UI keeps one short line per control; this is the long
version. Each tab has a **Manual** link that opens the matching section here.

Nothing is ever rendered on its own: a changed setting is saved and applied at
the next **Render** / **Generate**, and that button is the only thing that
draws.

- [Download](#download)
- [Land/Sea](#landsea)
- [Countries](#countries)
- [Rivers](#rivers)
- [States](#states)
- [Heightmap](#heightmap)
- [City](#city)
- [Terrain](#terrain)
- [Provinces](#provinces)
- [Strategic regions](#strategic-regions)
- [HOI4 mod](#hoi4-mod)

---

## Download

Files are fetched by the tool itself, into `data/<resource>/` next to the
code — not through the browser. **Download All** skips whatever is already
present, without asking the servers for it. A `.zip` is extracted in place and
only counts as present once it has been extracted whole.

### What each dataset is for

| Card | Dataset | Needed by |
|---|---|---|
| GADM Shapefiles | GADM 3.6, levels 0–5 | everything — the tool does not start a map without it |
| NASA Blue Marble | August satellite composite, 500 m | the colour map |
| Land elevations | see below | heightmap, sea level rise, river mouths, terrain relief |
| NASA Bathymetry | GEBCO 08 ocean depths, 1 km | naval terrain, water colours |
| HydroSHEDS | HydroRIVERS (discharge per segment), HydroLAKES (≥ 10 ha) | rivers, lakes |
| Copernicus Land Cover | one class per 100 m pixel, 2015 | terrain, forests, deserts, urban areas |
| GeoNames | places ≥ 1,000 people, their names per language, country info | cities, labels, victory points, continents |
| Marine Regions EEZ | World EEZ v12 + internal and archipelagic waters | sea provinces, sea regions, the EEZ overlay |
| Population density | GHS-POP, ~1 km cells, 2025 | state and province sizes |
| Population by subdivision | GHS-DUC, per GADM unit | state populations, manpower in the mod |
| Climate normals | ERA5 1991–2020 monthly | the strategic regions' weather |
| Country flags | flagcdn.com, 320 px | the exported countries' flags |

Licences are in `README.md` and in every export's `ABOUT_EarthToHoI4.txt`;
they apply to the maps and mods you make too.

### Land elevations: one of the two is enough

The elevation card holds two alternatives on sub-tabs. Only one is needed.

| | NASA Topographic Heightmap | GMTED2010 |
|---|---|---|
| What | 8-bit grey, 0–6400 m in ~26.5 m steps | 16-bit, real metres |
| Resolution | 1 km (30 arcsec) | ~450 m (15 arcsec) |
| Coverage | the whole globe, 8 tiles | land only, 84°N–56°S, 96 tiles |
| Size | ~1 GB | 69 MB a tile, 6.6 GB for all |
| In "Download All" | yes | no — pick the tiles your maps cover |

The tool reads GMTED wherever its tiles are on disk and falls back to the NASA
heightmap everywhere else, so having both only adds precision where GMTED is
missing. GMTED tiles are named after their south-west corner: a map of France
needs `30N000E` and `50N000E`.

GMTED matters wherever real metres decide something. On the NASA heightmap a
1 m sea level rise drowns everything under the first grey step, so +1 m, +5 m
and +20 m all flood the same land (5.57% of France); with GMTED they flood
0.98%, 1.93% and 4.54%. A river delta, one flat value on the NASA tiles, has
ground to follow on GMTED.

`MAP_TOOL_GMTED_DIRS` (`;`-separated) adds folders to read, for a drive that
already holds some tiles. The finest tile on disk wins.

### Results cache

Every tab's results — country map, rivers, states, terrain, provinces,
strategic regions — are kept in `cache/results/`, so reopening the tool or
coming back to a view doesn't compute them again. A result is reused only while
nothing that went into it changed: its settings, the tool's code, the
downloaded data. The oldest go first beyond the size limit. **Clearing is
never needed for correctness** — only to free disk space.

---

## Land/Sea

These settings clean up the coastline of *every* render (the Countries and
Rivers tabs and the CLI), in this order: fill → lakes → island separation →
strait fixing → island growth → microstates → sea specks → diagonal
connections → the small-area sieve. They are saved in
`land_sea_settings.json`.

**Lakes** (minimum size, px). Lakes and reservoirs covering at least that many
pixels become water before every other step, in their own colour. Islands in a
lake count as islands. Strait fixing never joins a lake to the sea. The
Caspian is sea here, not a lake.

**Island separation** (gap px). Wherever two land masses that are separate in
the data touch in the image — Funen and Jutland across the 573 m Little
Belt — a water gap is cut on the larger one's side. Land borders between
countries are left alone, and so are two countries' main landmasses (GADM
leaves real gaps along some border rivers).

**Minimum island size.** Islands smaller than this stay merged instead of
cutting into the neighbouring coast.

**Island growth** (min/max size). Islands from the minimum size (inclusive) up
to the maximum (exclusive) that lie far enough from every continental mass are
widened into the sea, keeping at least the island gap / minimum strait width
from any other land.

**Microstates** (px). A country whose whole projected area would be smaller
than this gets a compact blob of that size around its centre — taken from the
land around it first, then from the sea — and is protected from the sieve. The
area measured is the whole country, so a sliver of a big country at the frame
edge never qualifies.

**Sea specks** (px). Sea areas not side-connected to other water and below
this size become land, in the most common colour around them. Areas touching
the image border are kept.

**Diagonal connections.** Where a country's pixels touch only corner to
corner, one side pixel between them is filled in that country's colour: a sea
pixel when that splits no water and narrows no strait, else — on land
borders — a neighbour's pixel it can spare.

**Small-area sieve** (px). Every connected area of one colour below this size
(rogue border specks, tiny exclaves such as Llívia) joins its most common land
neighbour, or the sea if it only touches sea. Areas touching the image border
and microstates are kept. Runs last. The size is an *area*, not a width: an
8 px minimum width would erase every thin peninsula.

**Strait fixing.** Water bodies that are connected in the real vector
coastline but got split by rasterization — the Bosphorus in a Europe-wide
view — are re-joined by carving the shortest, most water-rich channel between
them.

- *Minimum strait width* (px): every water passage linking two water areas is
  widened to at least that many pixels on both axes, so the connection
  survives the game's 4-connectivity. Dead-end inlets are left alone.
- *Minimum water body* (px): smaller specks of water aren't reconnected. They
  can still serve as stepping stones inside a strait.

**Manual channels.** Force a straight water channel — canals, or straits the
data itself closes. Coordinates as `lon, lat`; the ends snap to water within
12 px.

**Blocked points.** No channel is carved within a few pixels of a blocked
point. Use it to reject a false strait: press **Block** in the detected
channels list, or type `lon, lat`.

**Save & Preview** writes the settings and renders the Countries tab's current
view as a land/sea image: drowned land dark teal, separation gaps cyan, carved
channels red, widened passages orange, manual channels magenta, blocked points
yellow, with a clickable list of the carved channels. The **Display** switch
shows the plain mask instead (land white, sea black, lakes blue).

---

## Countries

Pick countries, a projection and a margin; the tool frames the selection and
draws a flat-colour map. This view is what every other tab works on.

### Subdivisions

**Level 1** (regions, states, provinces). Untick a selected country's
subdivisions to leave them out: the view is framed on the ticked ones only,
and parts of the country lying entirely inside unticked ones — islands,
exclaves: Hawaii, Corsica — are left out of the map, becoming sea on every
tab. This overrides "Ignore overseas territories" for that country, so the USA
can be drawn with Alaska but without Hawaii.

**Level 2** (counties, districts, council areas) does the same one level
finer, for the parts a level-1 region does not isolate: untick Shetland
Islands (Scotland) or Isles of Scilly (England) to drop them from a map of the
British Isles. The two levels work together — an unticked level-1 region takes
all of its level-2 units with it, and the finer list is only read when you use
it.

The search filters both lists and also lists matching subdivisions of
*unselected* countries: adding one frames it together with your selection
(Finland + Karelia), and its country is then drawn like any neighbour.

> **Limit, both levels:** only land that GADM draws as its own shape (islands,
> exclaves) can be left out of the *drawing*. A subdivision joined to the rest
> by land can only be left out of the *framing*, because GADM draws the
> country as one shape and a shape spanning several subdivisions follows the
> majority. Shetland, Scilly and the Channel Islands qualify; mainland
> Scotland does not.

The Channel Islands are not British subdivisions: Jersey, Guernsey and the
Isle of Man are GADM countries of their own, with no level 2 at all. While
**Autonomous territories** is ticked they are drawn as part of the UK, and the
level-1 list then carries a block for each of them — untick all of a block's
parishes and that island goes.

### Size and crop

**Output size.** Auto picks the resolution whose aspect ratio best matches the
selection. A manual size keeps the whole selection visible and adds extra
surrounding area on the looser axis.

**Crop.** Cuts bands of pixels off the chosen borders, on top of whatever the
size gives — one border or several, 0 to leave one alone.

Any whole number of pixels works, but a map's sides have to stay in whole
256 px steps, so what an *axis* loses is rounded up to the next step and the
**opposite side makes up the difference**: ask for 156 px off the top and
100 px come off the bottom with it. The side you asked for always loses
exactly what you asked; the box shows what that ends up taking.

The crop is the last thing the render does, after the Land/Sea steps *and*
after the framing, so it can only remove map: zoom and position never change,
and every other tab — rivers, states, terrain, provinces — sees the cropped
map. The HOI4 edge bands are re-laid on the new borders, so no land is left
touching them. On a world map, cropping left or right cuts real longitude
away: the map then covers less than the full 360°, so keep the seam in mind.

### Frame and projection

**World map.** Shows every country (hidden ones excepted) edge to edge, with
no margin; the country selection is then optional. With **Centre on** left on
Auto the map's edge runs through the Bering Strait, the only land-free
meridian band. One column of water is always kept on each side, whatever the
edge bands are set to: the game joins the two borders — the map wraps — so an
island painted against one of them would end up next to whatever sits on the
other.

**Centre on.** Auto places the map's edge where it cuts through the least
land. Picking a country puts that country in the middle, and the edge on the
opposite side of the globe. Robinson and Mollweide are always centred on the
map itself, with the seam moved over the emptiest meridian.

**Ignore overseas territories.** Frames the view on each selected country's
mainland only, worked out from GADM level-2 subdivisions clustered within
500 km: Portugal without the Azores and Madeira. Overseas parts are still
*drawn* if they fall inside the view.

**HOI4 edge bands.** Left edge: a band of sea, a band of unowned land (dark
grey, impassable in the game — no movement across the map's wrap), a band of
sea. Right edge: a band of sea. Each band 2–10 px wide; nothing along the top
and bottom. They are painted over whatever the view shows there, so no land
touches the left or right edge and rivers leaving the map end at the sea. The
unowned strip belongs to no country or state, gets provinces of its own and
never carries rivers.

> **Always export with the edge bands on.** The game builds its whole province
> adjacency across the wrap, including its own idea of which provinces are
> coastal, and a coastal province with no port in the nudger crashes the game
> on start. A France map exported with the bands off had 96 of them.

### Sea level rise

Every piece of land at or below that height becomes sea **before** the
Land/Sea steps, so lakes, island separation, strait fixing and the sieve clean
up the new coastline.

- The people who lived there are lost: each state keeps the share of its
  population still above water, weighted by the population raster where it is
  downloaded. A subdivision drowned whole leaves no state.
- No city — and so no victory point — is left in the water. A city whose pixel
  drowned is dropped rather than moved ashore.
- The drowned ground keeps its depth, which the sea colours and the naval
  terrain use. The bathymetry reads drowned land as 0 m, so the deeper of the
  two is taken.

Heights are the **real ground**, never the exaggerated HOI4 ones. With GMTED
downloaded the figure is in true metres; on the NASA heightmap the rise lands
on the nearest ~26.5 m step, so anything under ~13 m goes at the first one. The
Land/Sea tab's preview shows the drowned land in its own colour.

### Hide and Remove

**Hide** keeps the country's land. It is unpainted on the canvas but still
land to every later step: the coastline, the heightmap, the terrain and the
provinces all see land there, which is what keeps a strait a strait. The mod
turns it into impassable wasteland owned by the country it borders most. Try
this first.

**Remove** takes the country off the map altogether: its land becomes sea on
every tab, as if it had never been there. Use it when a neighbour is simply in
the way — mainland Europe on a British Isles map. A removed country is dropped
from the selection and no longer counts for the frame.

> **One limit, shared by Remove and unticked subdivisions:** a neighbour's
> border with removed land is a land border in GADM, so the newly exposed
> shore is not read as coastline by strait fixing or island separation.

### Colours and EEZ

Click a colour square in the legend to change that country's colour. Colours
are forced unique over the map, because the Land/Sea steps identify countries
by colour.

**Exclusive economic zones** (Marine Regions World EEZ v12) are tinted flat in
the country's colour — a disputed or joint zone in its first-listed party's.
They follow the map's countries: hidden ones get none, merged territories
their parent's colour, and fragments under the Land/Sea minimum region size
are merged away. Where the EEZ coastline and GADM's differ:

- uncovered sea near land (up to 25 km) takes the neighbouring zone;
- enclosed bays no zone reaches (Garabogazköl) take the zone beside them;
- sea within 10 km of a coast belongs to that coast's country, so zone borders
  meet the land borders;
- microstates' tiny zones (under 1,000 km²: Monaco, Singapore, Slovenia,
  Neum, Aqaba) are erased and shared among their neighbours;
- every piece of a zone touches its country's land on the map or the map edge;
  stray slivers join the zone beside them, and the zones of islands the map
  doesn't show (Kiribati, Micronesia, Tonga at world scale) are left as plain
  sea.

The overlay is on the displayed and downloaded image only. Every other tab
still sees plain sea there.

---

## Rivers

Uses the Countries tab's view and the Land/Sea settings. The output is Hearts
of Iron IV's `rivers.bmp`: 8-bit indexed in the game's palette — land white,
sea and lakes grey, rivers 1 px wide coloured by discharge from light cyan
(smallest) to dark navy (largest). Each river network starts with one green
source pixel on its main river, and every tributary ends with a red flow-in
pixel where it joins. Continuity is side-adjacency only; a corner-to-corner
touch does not count.

> **The river rule:** every river is 1 px wide at all times and never touches
> itself or another river — sides or corners — except a tributary at its
> confluence. It is enforced while drawing; painted pixels are never moved
> afterwards.

### How many rivers

**Automatic spacing** (px). Rivers are added, highest discharge first, until a
continent's rivers are that many pixels apart on average — its land area in
view divided by its river length. Being in pixels, a world map automatically
gets far fewer rivers than a Europe map, and a river-dense continent (South
America) a lower percentage than Europe. 70 px is about what 5% gives for
Europe on a Europe-wide map. It is an average over the whole continent, so its
river-rich parts stay denser than its dry ones.

**Percentage per continent.** Only that share of a continent's river segments
in view, by highest average discharge, is drawn — plus everything downstream
of them, so every river runs on to its mouth. Lower it for wide views.
Continents here are drainage regions, not political ones, so a whole river
basin shares one value: "Europe" includes Turkey, the Caucasus, Iran, Arabia
and the Caspian basin; Oceania includes Indonesia and the Philippines.

**Shortest river** (px). A river is a branch from its source to where it joins
another river, or a whole river on its own. Branches reaching the image edge
and the lower course of a river at its mouth are kept whatever their length.
0 keeps everything.

### Network fixes

**Across lakes.** The Nile isn't cut by Lake Nasser, nor the Dnieper by its
reservoirs.

**Mouths reach the sea.** Where the map's coastline lies beyond the river
data's — deltas, estuaries, polders — the river is continued to the nearest
sea, **following the ground**: the cheapest route over the frame's real
elevations, which costs a step its length, plus a penalty per metre it climbs,
plus a penalty for standing high above the lowest ground around, plus smooth
noise where the ground is flat so the route wanders instead of ruling a
straight line. A mouth is only extended if the water is actually near: a route
more than 2.5× the straight-line gap is rejected, which is what keeps a river
from snaking up one valley and down another. Without any elevations downloaded
it falls back to the shortest pixel path.

**Dead ends.** Rivers ending in the desert or any inland sink other than a
lake — Arabian and Saharan wadis, the Tarim — are left out before the
percentages are computed. The Volga, Jordan, Chari, Syr Darya, Helmand and
Okavango are kept.

**River borders.** For the rivers listed in `river_borders.json` (the Rhine,
Danube, Rio Grande, Mekong …) the river is moved onto the border, and where
the river rules keep it off, **the border is moved to the river instead** —
the HOI4 river rules always take precedence. The river always stays on one
side of the border. The fitted country map is the one to use with
`rivers.bmp`, and it is what the download button gives.

### Most branches in one river

A river network is a whole basin: every tributary that touches it, drawn as one
connected set of pixels. The game walks each network when it loads the map, at
a cost that grows far faster than the network does, and these maps are 20–30×
finer than the game's own — so a basin like the Danube fuses into one river the
size of the map. Measured on a 4096×3072 Yugoslavia map with 623 branches in
one river, the game never finished loading in 6 minutes; the same map with no
rivers at all loads in 60 s. Vanilla's biggest river has 26 branches and spans
an eighth of the map.

Over this many branch points the network is cut into several at its smallest
confluences — the fewest cuts that do it, one pixel each, the gap hidden where
a tributary meets its parent, and each piece given its own green source.

| Cap | Cuts | Biggest network | Load |
|---|---|---|---|
| none | 0 | 115,654 px, 623 branches | never finished (>6 min) |
| 100 | 9 | 22,830 px, 99 branches | never finished (>5 min) |
| 50 | 20 | 16,319 px, 50 branches | 140 s |
| 24 | 43 | 7,160 px, 24 branches | 65 s |

0 turns the limit off, and the map may then take minutes to load or never
finish. The cut pixels are a concession to the game's loader, not a statement
about the ground: the Provinces tab puts them back, so a province is still cut
along a river where one of those gaps falls.

---

## States

States are made from GADM subdivisions and follow the country borders exactly.

**Level per country.** Each country gets the GADM level (0 = the whole
country, 1 = regions, 2 = departments / districts, up to 5) whose subdivisions
in view are at most the target size on average — the *coarsest* such level, so
states come out a little small and get merged rather than too big. If every
level is bigger, the finest is used. Always at least level 1, unless the
country itself is smaller than the merge size. The default target is the
vanilla HOI4 map's average state: 3,400 px on its 5632×2048 map (median
1,300).

A level whose mean is at least the merge size and at most 1.5× the target is
used as it is, and that country then gets **no merging at all** — France's 96
departements, Italy's 110 provinces, Spain's 50, Belgium's 11, Finland's 80
sub-regions.

Per-country overrides live in the **Countries** column on the right, which
also shows each country's level, state count and average size in view.

**Merge small states** (px). Smallest first, each small state joins a
neighbour in the same larger GADM subdivision — a district within its region;
if it has no neighbour there, one level up; never across a country. It prefers
a neighbour that is also too small and that the merge lifts above the size,
else the one it shares the longest border with. The merged state keeps the
**more populated** one's name, id and colour, and the populations are added.
Islands with no neighbour stay as they are. 0 = no merging.

Populations come from GHS-POP per GADM subdivision when downloaded, otherwise
from GeoNames places. A unit clipped by the frame is scaled by its visible
share.

**Density.** Like the provinces: each country's target and merge size are
multiplied by √(reference ÷ density), from ÷ the dense factor to × the sparse
one, and each state's merge size by its own density — then by the country's
factor in the list on the right. 1 = off.

**Follow rivers.** States always follow the country borders exactly; with this
on, those of the Rivers tab's country map fitted to the rivers. Where a state
border comes within 3 px of a river it is moved onto the river, never across a
country border.

**Colours.** Each state is a slight variation of its country's colour, unique
over the whole map, different from the country's own, and kept visibly apart
from its *neighbouring* states.

**Stray pieces.** A piece away from a state's main piece is a real exclave
only if one of its GADM units has most of its pixels in the main piece and the
piece itself is at least 20 px; otherwise it joins the same-country neighbour
it borders most, and its population goes with the pixels.

---

## Heightmap

Uses the Countries tab's view: the land elevations reprojected pixel by pixel
onto the same frame and projection, bilinearly interpolated — GMTED2010 in
metres wherever its tiles are on disk, the NASA topography × ~26.5 m
elsewhere. The report says which source answered.

**HOI4 levels** (the export's `heightmap.bmp`). Sea and lakes come from the
**country map** and its Land/Sea steps, never from the elevation data, whose
coastline doesn't match GADM's, and never from the river map, where rivers
cross lakes. Water is a flat 89, the coast pixel 94 rising into the terrain
over about 5 px, land at least 96, and the real heights are mapped onto the
vanilla map's range (up to 239 for the Himalaya) by a curve measured against
the game's own file.

**Raw** shows the metres themselves, stretched over the frame's range.

**Overlay** draws the country map's coastlines and borders in red, to check
that the heightmap lines up with it. The download never includes them.

### Exaggerate lower mountain ranges

Next to the Himalaya or the Iranian ranges, lower ones — Scandes,
Appalachians, Sudetes, Scottish Highlands, Urals — look flat. Each range's
relief above and below its surroundings (~80 km around) is multiplied by up to
the chosen factor: the lower the range, the more. Plains are left flat. 1 =
off. It applies to the HOI4 heightmap, the normal map and the Terrain tab's
relief, and it never invents ground outside what is really there.

The shape of the curve:

    factor = (range left alone ÷ the range's own relief) ^ exponent

capped above by the maximum, and faded to 1 between "left flat below" and
"full factor from" so plains keep their noise flat.

With the defaults the formula itself never asks for more than about ×3.2, so
the cap only bites once these are pushed. Raising the first number lifts every
range, the Alps included; the exponent favours the low ones (exponent 1 asks
×6.7 of a 150 m range); lowering the last two lets smaller hills count as
ranges.

**Normal map** (`world_normal.bmp`). The surface direction per pixel, computed
from the HOI4 heightmap as in the vanilla file: 24-bit RGB at half the
heightmap's size, red = east–west, green = north–south, blue = facing up, flat
= 128,128,255. It adds the terrain shading in game.

---

## City

Uses the Countries tab's view and the Land/Sea settings. Cities come from
GeoNames — every place with at least 1,000 inhabitants; city districts and
historical or abandoned places are left out. Parts of a city listed as
separate places (Paris's arrondissements, Warsaw's districts, New York's
boroughs) are merged into it and their populations added to its own; the CSV
also lists the city's own GeoNames figure.

Only cities on a shown country's land are drawn, and a coastal city just off
the map's coastline is moved onto it. Stars are drawn from the most to the
least populated city, each one only where the larger ones left room, so **a
smaller city never covers a larger one**. The CSV lists the drawn cities with
their class, pixel position and alternate names.

**Number of cities.** One number for the whole map, shared out between the
visible countries, then between each country's regions.

- *Land / population split*: 0 = by land in view only, 100 = by population in
  view only (the sum of a country's cities and towns in view, so a country half
  in the frame counts only that half), 50 = half of each.
- Each country then shows its most populated cities one by one until it reaches
  its number, fewer if it has fewer. Any country's number can be set in the
  **Visible countries** list on the right; the others keep theirs.
- *Split inside each country*: the same land / population split between a
  country's first-level regions — England, Scotland, Wales, Northern Ireland;
  Russian oblasts; US states — so a country's most populous part doesn't take
  all of its cities. Hover a country on the right to see its regions.

**Urban areas.** A town within a larger city's urban radius (same country;
about 15–20 km for Paris, 7 km for Lyon, derived from its population) is part
of that city — Sevran, Palaiseau and Chatou are Paris — so it doesn't take one
of its region's places. A town further out does. A town is never swallowed if
it is at least a third of the city's population or over a million people, so
Duisburg stays apart from Essen and Yokohama from Tokyo.

**Isolated cities** (px). Regardless of the numbers, a city with no more
populated city in view that close, on the same land, is always shown — on top
of its country's number once that is full — so sparse regions (Lapland,
Siberia, deserts) keep their local centres. 0 = off.

**Minimum distance** (px). A city closer than this to a larger shown city on
the same land is skipped and its country's next one takes its place, so suburbs
don't pile up on one spot. Cities across the sea never block each other: Saint
Helier on Jersey and Cherbourg both show. 0 = off.

**Exempt classes.** The largest cities are always shown however close together
— only a 3 px floor, and the lowest class never is — so dense clusters of big
cities like the Rhine-Ruhr (Köln, Düsseldorf, Essen, Dortmund, Duisburg) keep
all their stars. 0 = no exception.

**Small islands** (px). A land area smaller than this shows its largest city
alone, on top of the number above; it isn't counted in its country's land or
number. 0 = off.

**Classes.** Per continent, among its shown cities; class 1 = the largest.
*Log steps*: between the continent's smallest and largest shown city each
class spans the same population ratio, so the top classes hold few cities.
*Equal numbers*: each class holds the same number of cities. Continents follow
each city's country as GeoNames assigns it — Turkey, the Caucasus and the
Middle East are Asia, Cyprus is Europe — and Russia is split at the Urals
(60°E). The classes drive the victory point values in the export.

**Labels.** Names from GeoNames (Cities → Names + Country info). **Always in
Latin script**: names in other scripts are romanized (Москва → Moskva,
القاهرة → El-Qâhira), and exonyms fall back to the endonym where the language
has no name of its own, which is most towns. Labels are placed from the
largest city down, each beside its star where it overlaps no star and no
earlier name, else left out.

---

## Terrain

Uses the Countries tab's view. Land types come from the Copernicus land cover
(100 m, 2015), relief from the elevations. The output is Hearts of Iron IV's
`terrain.bmp`, 8-bit indexed with the game's palette, plus `trees.bmp` and the
colour map. The raw land cover export is an indexed PNG whose pixel values are
the Copernicus class codes.

The first terrain render prepares the land cover once, which takes about
15 minutes. Keep the tool open until it is done.

**Relief.** Heights come from the Heightmap tab, with its "exaggerate lower
mountain ranges" factor applied (the default thresholds were set with ×4).
What counts is **height differences, never the height itself**: highest minus
lowest point within the chosen distance, like Paint.NET's Outline effect. A
high flat plateau therefore stays flat. A small distance gives granular,
detailed relief rather than blobs. Smoothing 1 = off. Mountains are always
drawn on top of hills: the clean-up may raise a tier but never lowers a
mountain pixel.

**Forest and farmland** are mixed with plains in the proportions of the
example map — not as a per-class mapping, but by per-pixel odds against the
local share, then a median filter on forest borders. Patch size 0 = pixel
speckle. Tropical evergreen forest within 25° of the equator is jungle. In
`trees.bmp`, jungle within the edge width of the jungle's border becomes
sparse jungle (0 = off). **Tree density**: `trees.bmp` covers that percentage
of the land cover's forest area (100 = all of it), and the thinnest forest is
left out first.

**Desert and cold.** The land cover's bare / sparse vegetation is hot desert
below the cold latitude, pixel by pixel: sand with ~35% rocky patches
(smoothing = patch size). Hills standing on it become desert hills and
mountains desert mountains, where bare ground makes up at least a quarter of
the land within 10 km — so bare rock in the Alps or the Caucasus stays normal
mountains. Beyond the cold latitude it is snowy plains, since HOI4 has no cold
desert. **Transition**: grass and shrubs near hot desert get speckled, denser
where desert covers more of the surroundings within that distance (0 = off).
Much of Central Asia (Betpak-Dala, Moyynkum) is shrubland in the land cover,
so it stays plains with the transition only near bare ground.

**Water and wetlands.** Sea and lakes come from the country map. Lakes too
small for the Land/Sea step become marsh, together with the land cover's
wetlands.

**Cities.** Smoothing turns urban areas into round blobs (1 = off: every urban
pixel). Growing enlarges each city by that many pixels, keeping its shape.
**Urban discs** use the City tab's current settings: every city of classes 1–3
gets a round urban area of the given radius on top of the land cover's, 0 =
none for that class.

**Colour map** (`colormap_rgb_cityemissivemask_a.dds`, half the terrain's
size). RGB = NASA Blue Marble (August) with its colours quantile-matched to
the vanilla file and water a flat blue taken from the country map; near-white
snow is neutralised, or the curves alone turn Greenland pink. The alpha
channel is the city lights at night, from the terrain's urban pixels — land
cover cities plus the City tab's discs. **Brightness** multiplies it, **glow**
blurs it, in colour map pixels.

---

## Provinces

Land provinces are cut out of the States tab's states, with the same settings,
rules and rivers; a province always belongs to exactly one state.

**Land size** (min / max, px). GADM subdivisions at the coarsest level
averaging at most the maximum, split into compact pieces above it, merged
within their state below the minimum. Vanilla: mean 400 px, median 160.

Splits are **organic**, not geometric: k-means seeds for balance, then each
piece grown by least cost, where crossing the next finer GADM border, a river
or a terrain change costs extra. Borders therefore settle on admin lines,
rivers and terrain edges, and wander elsewhere.

**Sea size** (min / max, px). Each sea is cut into compact cells between the
two sizes, within one country's EEZ, so province borders follow the future sea
region lines exactly. Every lake is one lake province, or one per country — see
below. Vanilla: mean 2,370 px.

**Density.** Land province sizes follow the population density, not how fine a
country's administrative units happen to be: size × √(reference ÷ density),
from ÷ the dense factor (cities, Belgium, the Netherlands) up to × the sparse
one (deserts, taiga, tundra). Provinces aim at √(min × max) scaled that way,
are split above the maximum and merged below half the aim or the minimum. Sea
provinces grow with the distance from the coast, up to their factor. 1 = off.
Per-country size factors live in the **Countries** column on the right: 2 =
about twice as big, 0.5 = half, applied at the next render.

**River splits** (px). A land province at least this big that the rivers cut in
two is split along the river, each piece at least 40 px and 15% of the
province. 0 = off. Bits left across a river, too small to be a province, go to
the province on their side of the river when it is in the same state — never
to another country. River pixels themselves go to the kept piece they border
most.

- The branch cap's gaps (see [Rivers](#rivers)) are healed first, so a
  province is still split where one of those single deleted pixels falls.
- A river that **ends inside a province** — its source — is continued in a
  straight line, in the direction it was running, until it leaves that
  province. Without that the river would be a slit and the province would stay
  in one piece. Those pixels are land, used only to decide the cut; they are
  never river in `rivers.bmp`.

Provinces under 20 px are merged into a neighbour, islets and microstates with
no neighbour of their own country excepted.

**Lakes.** Bordered by one country → that country's state with the longest
shore. By several → the country with the longest shore, or, from 100 px,
divided along the GADM border when every part keeps 40 px.

**Target count.** 0 = off. Otherwise the land provinces' sizes — min, max and
the river split — are scaled together so that land + sea + lakes come out at
about this total, typically within 1%. The sea and lakes keep their own
settings, so their count doesn't change. The scale used is shown after the
render.

> **At most 22,367 provinces.** Hearts of Iron IV draws at most 65,535
> province borders — one per pair of neighbouring provinces, about 2.93 per
> province — and stops drawing them over large parts of the map past that. The
> limit is hardcoded in the game and is **a count, not a resolution**: it is
> the same at any map size. Whatever the settings, pairs of small land
> provinces are merged at the end until the map is under it, and the border
> count is shown after the render.

**City provinces.** Each City-tab city of the chosen classes gets a province
shaped on the Terrain tab's urban area it lies on, within its state; several
cities on one urban area share it, divided by nearest. It is grown to the
minimum size with the nearest land of its state, never merged with its
neighbours, its terrain is forced to urban, and `province_states.csv` names it
in a `city` column. Uses the City and Terrain tabs' current settings.

### Topology and output

Every province is one 4-connected piece; no two provinces touch only by a
corner (an X-crossing) nor along more than one stretch. A sea province's
pixel-mean centre must be one of its own pixels, else it is split.

`definition.csv`: `id;R;G;B;land/sea/lake;coastal;terrain;continent` — terrain
is the province's most common Terrain-tab type, continent is the game's
numbering (1 Europe … 7 Middle East), and *coastal* counts the map's wrap,
because the game does.

`province_states.csv`: each land and lake province's state, which is what the
state files use — the states are exactly their provinces. Where an X-crossing
fix moved a pixel across a state border the state follows it;
`states_from_provinces.png` shows the result.

Colours: land = variations of the state's colour, sea = variations of #191F61
tinted by the EEZ, lakes = a lake blue tinted by the state. All unique, never
black.

---

## Strategic regions

Uses the Provinces tab's current settings, and so every other tab's. Reuses
the cached province map.

**Land regions** are made of whole states — the game wants every province of a
state in one region. They are merged below the minimum, aiming at about
√(min × max), and only go above the maximum when a single state is bigger.
Vanilla: 5% under 3,800 px, median 13,400, 5% over 52,900.

**Sea regions** are made of the Provinces tab's sea areas, so province borders
follow the region lines exactly. They are small along the coasts and in
enclosed seas and large in the open ocean, up to the maximum, and they never
cross an EEZ line. A big EEZ part is cut into a smoothed coastal band sliced
into pieces, plus offshore cells; outside every EEZ, cells only. Bands and
offshore water are separated at straits, or Gibraltar would join Morocco's
Atlantic and Mediterranean coasts in one region. Vanilla: 5% under 7,900 px,
median 67,000, 5% over 190,000.

**Islands** (px). An island — states joined by land — smaller than this goes,
whole, to the sea region it borders most; larger ones get land regions. An
island whose own region would have its centre outside it joins the sea around
it whatever its size. Vanilla: islands up to ~3,500 px sit in sea regions.
0 = never.

### The rules the game needs

Every province gets exactly one region, no region is empty, every state is in
one region, an island in a sea region is there whole, each region is one
connected piece, and **each region's centre lies within it** — the mean of its
pixels (all four of the 2×2 pixels around it, so the rounding can't matter)
*and* the mean of its provinces' centres. Land without a state — the edge
strip, hidden countries — never goes to the sea.

The strict centre test exists because of a "division by 0" crash that happens
only when a centre falls outside its region, though not every time: vanilla
itself has 18 such regions and 3 empty ones without crashing, so "outside" is
necessary but not sufficient, and the exact trigger is unknown. The tool makes
it impossible instead.

### Output

The zip holds:

- `map/strategicregions/` — one file per region. A sea region carries its
  **naval terrain**: fjords / archipelago where islands are many or the water
  is narrow, shallow sea on the continental shelf (from the bathymetry), deep
  ocean far from land and deeper than 3,000 m, else ocean.
- its **weather** — 12 monthly periods from the climate normals (ERA5
  1991–2020, averaged over the region), converted to the game's temperatures
  and odds of rain, snow, blizzards, mud, sandstorms and arctic water by
  formulas fitted on the game's own regions. A neutral placeholder without the
  data.
- `map/weatherpositions.txt` — one big and one small weather object per
  region, at its centre. The game crashes without them.
- the region names (`localisation/english/strategic_region_names_l_english.yml`)
  and a CSV.

---

## HOI4 mod

Uses every tab's current settings — the Strategic regions tab's request, which
carries the Provinces tab's and so all the others'. The export takes about 40 s
with every stage cached.

The mod folder is the name's ASCII form, in your Hearts of Iron IV mod folder
next to your other mods. A mod of the same name is replaced; no other mod is
touched.

**Game and mod folders.** Leave empty to use the folders found automatically:
Steam libraries, GOG, Microsoft Store, and
`Documents/Paradox Interactive/Hearts of Iron IV/mod`. Set them if the tool
can't find them — the game folder is the one holding `hoi4.exe`.

**What the export writes.** The whole map (provinces, heightmap, terrain,
rivers, trees, normal map, colour map, water and fog-of-war textures, cities),
the states (owner, cores, provinces, manpower, victory points from the City
tab), the strategic regions and their weather, buildings and unit positions,
railways and supply nodes, sea crossings over straits, one country per map
country — own tags, its GADM name, its real flag, non-aligned, one infantry
division each — a 1936 bookmark and localisation in every game language.

The base game's countries are not loaded, its country-specific files are
emptied and every remaining reference to them is cleaned out, so the mod
launches on its own. There are no factories or resources yet.

**State names** are GADM's own name for the division the state came from, used
verbatim: that field is the romanized endonym (Bayern, Toscana, Fejér), so
nothing is cleaned, shortened or disambiguated, and repeated names are kept.
A division GADM leaves unnamed ("n.a.") takes its parent's name.

**Thumbnail.** `thumbnail.png`, 512×512, is what the launcher and the Workshop
show: the mod's own map rendered the way the game draws it zoomed out —
terrain, water, rivers, country colours and borders — cut to a square around
the middle of the map, with the mod's name across it in the map font, white
with the game's soft dark edge. It adds about 20 s to the export. Unticked, the
Countries tab's map is used instead, with the name written on it all the same.

**After exporting.** Enable the mod in the launcher's playset. In game, the
political colours need the post effects on (`draw_postfx=yes` in your
`settings.txt`).

**Map scale, lighting and day/night** are computed on every export from the
frame itself. The shaders are copied from your own installation, so **a game
update needs a re-export**.

**Camera and map labels.** `common/defines/zz_map_tool_view.lua` raises the
camera's zoom range and pushes back the distances at which province names,
victory points and capital icons fade out — vanilla's are tuned for its own
5632×2048 map, and these maps are much finer. It also raises the flag texture
caps, which would otherwise scale the real national flags back down.

---

## Reporting a problem

Every export saves its generation parameters — every setting of every tab,
plus the tool's version, Python, OS, game version and datasets — as
`earthtohoi4_params.json` in the exported mod's folder and in `runs/` (the
last 30 runs; a failed run records its error and traceback there). **Save
parameters** downloads the current settings at any time, **Load parameters**
restores them.

If the tool or an exported mod crashes, open an issue and attach that file: it
lets the run be reproduced exactly with
`python src/run_params.py replay <file> [--export "Mod name"]`.
