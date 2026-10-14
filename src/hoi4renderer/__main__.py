"""Hoi4renderer command line interface.

Usage: python -m hoi4renderer <mod folder> [-o out.png] [--scale 2] ...
"""
import argparse
import os
import sys
import time

from PIL import Image

from . import pdx
from .data import MapData, log
from .render import Options, Renderer, composite, layer_rgba

DEFAULT_GAME_DIRS = [
    r"C:\Program Files (x86)\Steam\steamapps\common\Hearts of Iron IV",
    r"C:\Program Files\Steam\steamapps\common\Hearts of Iron IV",
    os.path.expanduser("~/.steam/steam/steamapps/common/Hearts of Iron IV"),
    os.path.expanduser("~/Library/Application Support/Steam/steamapps/common/Hearts of Iron IV"),
]


def find_game_dir():
    for d in DEFAULT_GAME_DIRS:
        if os.path.isfile(os.path.join(d, "map", "default.map")):
            return d
    return None


def parse_color(s):
    s = s.strip().lstrip("#")
    if "," in s:
        return tuple(int(x) for x in s.split(","))
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="hoi4renderer",
        description="Render a Hearts of Iron IV mod's full map to a game-like PNG.")
    ap.add_argument("mod", help="mod folder (the one containing map/, common/, history/...). "
                               "Use the game folder itself to render the vanilla map.")
    ap.add_argument("-o", "--output", help="output folder (default: <mod name>_render in the current folder)")
    ap.add_argument("--no-layers", action="store_true",
                    help="only write the flattened map.png, not the separate transparent layers")
    ap.add_argument("--no-composite", action="store_true", help="only write the separate layers, not map.png")
    ap.add_argument("--game", help="HOI4 install folder (auto-detected on Steam default paths)")
    ap.add_argument("-s", "--scale", type=float, default=2.0,
                    help="output resolution as a multiple of the map size (default: 2)")
    ap.add_argument("--width", type=int, help="output width in pixels (overrides --scale)")
    ap.add_argument("--date", default="1936.1.1", help="history date used for state owners (default: 1936.1.1)")
    ap.add_argument("--language", default="english", help="localisation language for names (default: english)")
    ap.add_argument("--fill", type=float, default=Options.fill,
                    help="country colour opacity inside countries, 0..1 (default 0.4 = game fully zoomed out; "
                         "0 = only coloured gradient borders like when zoomed in)")
    ap.add_argument("--gradient", type=float, default=Options.gradient,
                    help="multiplier for the width of the coloured gradient along borders (default 1)")
    ap.add_argument("--relief", type=float, default=Options.relief,
                    help="terrain slope exaggeration for the lighting (default 1 = as in game)")
    ap.add_argument("--avg-luminance", type=float, default=Options.avg_luminance,
                    help="scene average luminance used by the game's auto exposure; lower = brighter "
                         f"(default {Options.avg_luminance})")
    ap.add_argument("--lut", default=Options.lut,
                    help="colour-correction LUT (game path) applied after tonemapping "
                         "(default: the zoomed-out one, gfx/world/colorcorrection_farrange.tga; '' = none)")
    ap.add_argument("--sky-period", type=float, default=Options.sky_period,
                    help="size (map px) of the sky reflection pattern on water (default 1400)")
    ap.add_argument("--river-width", type=float, default=Options.river_width,
                    help=f"width (map px) of index-3 rivers, the narrowest in rivers.bmp (default {Options.river_width})")
    ap.add_argument("--river-width-step", type=float, default=Options.river_width_step,
                    help=f"extra width (map px) per rivers.bmp colour index above 3 (default {Options.river_width_step})")
    ap.add_argument("--river-widths", help="explicit width per rivers.bmp index, e.g. '3:1.2,4:1.5,11:3.5' "
                                           "(unlisted indices use --river-width/--river-width-step)")
    ap.add_argument("--river-colormap", choices=["mod", "vanilla"], default="mod",
                    help="water colour texture used under rivers: the mod's (as in game, default) or vanilla's "
                         "(darker slate rivers if the mod's water colormap is bright over land)")
    ap.add_argument("--no-names", action="store_true", help="do not draw country names")
    ap.add_argument("--no-rivers", action="store_true", help="do not draw rivers")
    ap.add_argument("--no-state-borders", action="store_true", help="do not draw state borders")
    ap.add_argument("--no-country-borders", action="store_true", help="do not draw country border lines")
    ap.add_argument("--no-sea-borders", action="store_true", help="do not draw sea province / sea region borders")
    ap.add_argument("--province-borders", action="store_true", help="also draw province borders")
    ap.add_argument("--terrain-only", action="store_true",
                    help="no political layer at all (no colours, borders or names)")
    ap.add_argument("--name-case", choices=["upper", "normal"], default=Options.name_case,
                    help="country name letter case (default: upper)")
    ap.add_argument("--name-color", default="ececec", help="country name colour, hex RRGGBB or r,g,b")
    ap.add_argument("--name-opacity", type=float, default=Options.name_opacity)
    ap.add_argument("--font", help="BMFont .fnt file for names (default: the mod's/game's gfx/fonts/hoi_mapfont4.fnt)")
    ap.add_argument("--compress", type=int, default=6, help="PNG compression level 0-9 (default 6)")
    args = ap.parse_args(argv)

    mod = os.path.abspath(args.mod)
    if not os.path.isdir(mod):
        ap.error(f"mod folder not found: {mod}")
    game = args.game or find_game_dir()
    if not game or not os.path.isdir(game):
        ap.error("HOI4 install folder not found, pass it with --game")
    if os.path.normcase(os.path.abspath(game)) == os.path.normcase(mod):
        vfs = pdx.VFS(game, None)
    else:
        vfs = pdx.VFS.from_mod(game, mod)

    t0 = time.time()
    md = MapData(vfs, date=args.date, language=args.language)
    o = Options()
    o.scale = args.scale if not args.width else args.width / md.width_px
    o.fill = args.fill
    o.gradient = args.gradient
    o.relief = args.relief
    o.avg_luminance = args.avg_luminance
    o.lut = args.lut
    o.sky_period = args.sky_period
    o.river_width = args.river_width
    o.river_width_step = args.river_width_step
    if args.river_widths:
        o.river_widths = {int(k): float(v) for k, v in (kv.split(":") for kv in args.river_widths.split(","))}
    o.river_colormap = args.river_colormap
    o.names = not args.no_names
    o.rivers = not args.no_rivers
    o.state_borders = not args.no_state_borders
    o.country_borders = not args.no_country_borders
    o.province_borders = args.province_borders
    o.sea_borders = not args.no_sea_borders
    o.name_case = args.name_case
    o.name_color = parse_color(args.name_color)
    o.name_opacity = args.name_opacity
    o.font = args.font
    if args.terrain_only:
        o.fill = 0.0
        o.gradient = 0.0
        o.names = o.state_borders = o.country_borders = o.province_borders = False
        o.political = False

    r = Renderer(md, o)
    log(f"Map {md.width_px}x{md.height_px} -> output {r.OW}x{r.OH}")
    r.prepare()
    r.build_vector_layers()
    log("Rendering...")
    r.render()
    names_mask = None
    if o.names:
        from .names import draw_names
        log("Placing country names...")
        names_mask = draw_names(r)
    layers = r.layers(names_mask)

    out_dir = args.output or (os.path.basename(mod.rstrip("\/")).replace(" ", "_") + "_render")
    os.makedirs(out_dir, exist_ok=True)
    level = max(0, min(9, args.compress))
    if not args.no_layers:
        for i, (name, kind, data) in enumerate(layers):
            path = os.path.join(out_dir, f"{i}_{name}.png")
            log(f"Writing {path}...")
            if kind == "rgb":
                Image.fromarray(data).save(path, compress_level=level)
            else:
                Image.fromarray(layer_rgba(kind, data, 0, r.OH), "RGBA").save(path, compress_level=level)
    if not args.no_composite:
        path = os.path.join(out_dir, "map.png")
        log(f"Writing {path}...")
        Image.fromarray(composite(layers)).save(path, compress_level=level)
    log(f"Done in {time.time() - t0:.0f}s -> {os.path.abspath(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
