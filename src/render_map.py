"""
CLI: render one or more countries as a flat-color map (with visible
neighbors) in one of the fixed output resolutions. When several countries
are given, the viewport is framed to fit all of them.

Usage:
    python render_map.py FRA
    python render_map.py FRA DEU ITA --projection mercator --margin 10 --output europe.png
    python render_map.py CHL --size 2048x6400
    python render_map.py PRT --ignore-overseas
"""

import argparse

import core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="*", help="Country tag(s) to render, e.g. FRA USA JPN (GID_0 field)")
    parser.add_argument("--projection", default="miller", choices=sorted(core.PROJECTIONS.keys()))
    parser.add_argument("--margin", type=float, default=core.DEFAULT_MARGIN_PERCENT,
                         help="Margin percent added to each side of the selection's bbox")
    parser.add_argument("--size", default="auto",
                         help="Output size as WIDTHxHEIGHT (one of core.allowed_output_sizes()), or 'auto' (default)")
    parser.add_argument("--sea-level", type=int, default=0, metavar="METRES",
                         help="Raise the sea level by this many metres before the land/sea steps (0 = off)")
    parser.add_argument("--double-size", action="store_true",
                         help="Double both sides of the output (for a hoi4.exe patched with the map size unlocker)")
    parser.add_argument("--ignore-overseas", action="store_true",
                         help="Frame the view on each country's mainland only (uses GADM level 2, else level 1)")
    parser.add_argument("--merge-autonomous", action="store_true",
                         help="Draw autonomous territories (Aland, Faroe, Greenland, Hong Kong, ...) as part of their country")
    parser.add_argument("--world", action="store_true", help="Frame the whole world (tags optional)")
    parser.add_argument("--antarctica", action="store_true", help="World map: include Antarctica")
    parser.add_argument("--center", default=None, help="Country tag to centre the map on (default: auto)")
    parser.add_argument("--edge-bands", action="store_true",
                        help="HOI4 edge bands: left sea + unowned land + sea, right sea")
    parser.add_argument("--edge-band-px", type=int, default=core.EDGE_BAND_PX_DEFAULT,
                        help="Width of each edge band, 2-10 px (default 4)")
    parser.add_argument("--exclude-subdivisions", nargs="*", default=None,
                        help="GADM level-1 ids (GID_1) to leave out of their country, e.g. USA.12_1 (Hawaii)")
    parser.add_argument("--include-subdivisions", nargs="*", default=None,
                        help="GADM level-1 ids of other countries to add to the framing, e.g. RUS.26_1 (Karelia)")
    parser.add_argument("--no-straits", action="store_true",
                         help="Skip strait fixing (otherwise the Land/Sea tab's saved settings apply)")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="Country tag(s) to hide: not drawn, but still land (wasteland in the mod)")
    parser.add_argument("--remove", nargs="*", default=None,
                        help="Country tag(s) to take off the map entirely: their area becomes sea")
    parser.add_argument("--crop", nargs=4, type=int, default=None,
                        metavar=("LEFT", "RIGHT", "TOP", "BOTTOM"),
                        help="Force a crop off each border, any whole number of px (world maps included); "
                             "the opposite side makes each axis up to a multiple of 256")
    parser.add_argument("--output", default=None, help="Output PNG path (default: <tags>_<projection>.png)")
    args = parser.parse_args()

    tags = [t.upper() for t in args.tags]
    output_path = args.output or f"{'_'.join(tags) or 'world'}_{args.projection}.png"

    image, meta = core.render_map(tags, args.projection, margin_percent=args.margin,
                                   excluded_tags=args.exclude, removed_tags=args.remove,
                                   output_size=core.parse_output_size(args.size),
                                   ignore_overseas=args.ignore_overseas,
                                   land_sea={"fix_straits": False} if args.no_straits else None,
                                   merge_autonomous=args.merge_autonomous,
                                   world=args.world, center_tag=args.center,
                                   include_antarctica=args.antarctica, edge_bands=args.edge_bands, edge_band_px=args.edge_band_px,
                                   excluded_subdivisions=args.exclude_subdivisions,
                                   included_subdivisions=args.include_subdivisions,
                                   double_size=args.double_size, sea_level_m=args.sea_level,
                                   crop=args.crop)
    image.save(output_path)
    print(f"Saved: {output_path} ({meta['width']}x{meta['height']}, {len(meta['visible_countries'])} countries visible)")


if __name__ == "__main__":
    main()
