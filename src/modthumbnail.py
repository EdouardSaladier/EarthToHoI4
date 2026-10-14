"""The mod's thumbnail.png: the map as the game draws it, with the mod's name.

The picture the launcher (and the Workshop) shows is rendered from the mod the
export has just written, by the vendored Hoi4renderer (``hoi4renderer/``, a
software re-implementation of the game's map shaders): terrain, water, rivers,
country colours and borders exactly as the player sees them zoomed out. It is
then cut to a centred square, scaled to THUMB_PX, and the mod's name is
written across it in the map font the mod itself ships (user), white with the
soft black edge the game gives map names.

``render_thumbnail`` is the whole public interface; ``hoi4mod.write_thumbnail``
calls it and falls back to the Countries tab's image if anything here fails --
a thumbnail is never worth failing an export over.
"""
import os
import time

import cv2
import numpy as np
from PIL import Image

THUMB_PX = 512            # the launcher and the Workshop show a square picture
SUPERSAMPLE = 2           # rendered this much bigger, then averaged down
MAX_SCALE = 2.0           # never render more than this multiple of the map itself

# The title, in fractions of the thumbnail's side (so any size looks the same).
TITLE_SIDE_MARGIN = 0.07  # free space left and right of the text
TITLE_MAX_CAP = 0.15      # cap height of a short name
TITLE_MAX_LINES = 3
TITLE_LINE_GAIN = 1.03    # a line more has to buy at least this much letter size
TITLE_LINE_GAP = 1.45     # line spacing, in cap heights
TITLE_TRACKING = 0.05     # extra space between letters, in cap heights
TITLE_MAX_HEIGHT = 0.62   # the lines together never take more of the picture

# The dark edge, in cap heights: the game's map names are white over a dilated,
# blurred copy of themselves (render.names_rgba), which is what keeps them
# readable over bright terrain. A title is many times bigger than a map name,
# so both the spread and the blur follow the cap height instead of the scale.
OUTLINE_GROW = 0.030
OUTLINE_BLUR = 0.060
OUTLINE_ALPHA = 0.85

# Country names are left out: at thumbnail size they are illegible smudges and
# the mod's own name is what the picture is for.
RENDER_OPTIONS = {"names": False}


class ThumbnailError(Exception):
    pass


# ------------------------------------------------------------------ the map
def render_map(mod_dir, game_dir, size=THUMB_PX, options=None):
    """The mod's map, rendered the way the game draws it, as a PIL RGB image
    of `size` x `size` (the centre of the map: a mod's frame is already built
    around what it is about)."""
    try:
        from hoi4renderer import pdx
        from hoi4renderer.data import MapData
        from hoi4renderer.render import Options, Renderer, composite
    except ImportError as e:  # numpy / scipy / cv2 / Pillow, or a broken copy
        raise ThumbnailError(f"the map renderer isn't available: {e}")
    if not os.path.isdir(game_dir or ""):
        raise ThumbnailError("the Hearts of Iron IV folder is needed for the terrain textures")

    vfs = pdx.VFS.from_mod(game_dir, mod_dir)
    md = MapData(vfs)
    o = Options()
    # the square cut keeps the shorter side whole, so that is the side which
    # has to come out at SUPERSAMPLE x the thumbnail
    o.scale = min(MAX_SCALE, SUPERSAMPLE * size / min(md.width_px, md.height_px))
    for k, v in dict(RENDER_OPTIONS, **(options or {})).items():
        setattr(o, k, v)
    r = Renderer(md, o)
    r.prepare()
    r.build_vector_layers()
    r.render()
    names = None
    if o.names:
        from hoi4renderer.names import draw_names
        names = draw_names(r)
    return square(Image.fromarray(composite(r.layers(names))), size)


def square(img, size=THUMB_PX):
    """The centre of an image, as a `size` x `size` RGB picture."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    x, y = (w - s) // 2, (h - s) // 2
    return img.crop((x, y, x + s, y + s)).resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------- the title
def find_font(mod_dir, game_dir):
    """The map font of the mod, else the game's (hoi_mapfont4: the export
    copies Magna Europa: Reloaded's in, which has the most characters)."""
    from hoi4renderer import pdx
    from hoi4renderer.names import find_font as _find
    return _find(pdx.VFS(game_dir, mod_dir))


def _line_units(font, words):
    """Width of a line of words in font units, the tracking included."""
    text = " ".join(words)
    if not text:
        return 0.0
    return sum(font.advances(text)) + TITLE_TRACKING * font.cap * (len(text) - 1)


def _split_lines(font, words, n):
    """`words` cut into exactly n lines, with the narrowest widest line."""
    if len(words) < n:
        return None
    if n == 1:
        return [list(words)]
    best = None
    for i in range(1, len(words) - n + 2):
        rest = _split_lines(font, words[i:], n - 1)
        if rest is None:
            continue
        lines = [list(words[:i])] + rest
        widest = max(_line_units(font, line) for line in lines)
        if best is None or widest < best[0]:
            best = (widest, lines)
    return best[1] if best else None


def _layout(font, text, size):
    """(cap height in px, [(line, width in px)]) for the name, wrapped over up
    to TITLE_MAX_LINES lines -- whichever wrapping gives the biggest letters."""
    words = text.split()
    if not words:
        return None
    box_w = size * (1 - 2 * TITLE_SIDE_MARGIN)
    box_h = size * TITLE_MAX_HEIGHT
    best = None
    for n in range(1, TITLE_MAX_LINES + 1):
        lines = _split_lines(font, words, n)
        if lines is None:
            break
        units = [_line_units(font, line) for line in lines]
        # the cap height this wrapping allows, never bigger than TITLE_MAX_CAP
        # -- so a name that already fits stays on one line instead of being
        # broken for letters it wouldn't be allowed to grow into
        cap = min(box_w / max(units) * font.cap, box_h / ((n - 1) * TITLE_LINE_GAP + 1),
                  size * TITLE_MAX_CAP)
        if best is None or cap > best[0] * TITLE_LINE_GAIN:
            best = (cap, lines, units)
    cap, lines, units = best
    f = cap / font.cap
    return cap, [(" ".join(line), u * f) for line, u in zip(lines, units)]


def _stamp(mask, font, ch, f, x_centre, y_centre):
    """One glyph into the alpha mask, centred on its advance box horizontally
    and on the middle of the capitals vertically (the pivot names.py uses)."""
    g = font.glyph(ch)
    if g is None:
        return
    gx, gy, gw, gh, xo, yo, xa = g
    if gw <= 0 or gh <= 0:
        return
    w = max(1, int(round(gw * f)))
    h = max(1, int(round(gh * f)))
    glyph = font.page.crop((gx, gy, gx + gw, gy + gh)).resize((w, h), Image.LANCZOS)
    x0 = int(round(x_centre + (xo - xa / 2.0) * f))
    y0 = int(round(y_centre + (yo - font.cap_top - font.cap / 2.0) * f))
    H, W = mask.shape
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(W, x0 + w), min(H, y0 + h)
    if xs0 >= xs1 or ys0 >= ys1:
        return
    a = np.asarray(glyph, dtype=np.uint8)[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    np.maximum(mask[ys0:ys1, xs0:xs1], a, out=mask[ys0:ys1, xs0:xs1])


def draw_title(img, text, font_path):
    """The mod's name across the middle of `img` (a square RGB image), in the
    map font, white over its own blurred black copy."""
    from hoi4renderer.names import BMFont
    font = BMFont(font_path)
    W, H = img.size
    lay = _layout(font, text, W)
    if lay is None:
        return img
    cap, lines = lay
    f = cap / font.cap
    mask = np.zeros((H, W), np.uint8)
    total = cap * ((len(lines) - 1) * TITLE_LINE_GAP + 1)
    y = (H - total) / 2 + cap / 2  # middle of the first line's capitals
    for line, width in lines:
        x = (W - width) / 2
        for ch, adv in zip(line, font.advances(line)):
            _stamp(mask, font, ch, f, x + adv * f / 2, y)
            x += adv * f + TITLE_TRACKING * cap
        y += TITLE_LINE_GAP * cap

    t = mask.astype(np.float32) / 255.0
    grow = max(1, int(round(cap * OUTLINE_GROW)))
    sh = cv2.dilate(t, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    sh = np.clip(cv2.GaussianBlur(sh, (0, 0), max(0.6, cap * OUTLINE_BLUR)) * OUTLINE_ALPHA, 0, 1)
    base = np.asarray(img, dtype=np.float32)
    # composed as the game composes map names: the dark edge behind, the white
    # letters over it (render.names_rgba, resolved onto the picture)
    out = base * ((1.0 - t) * (1.0 - sh))[..., None] + 255.0 * t[..., None]
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


# ------------------------------------------------------------------ the lot
def render_thumbnail(mod_dir, game_dir, name, size=THUMB_PX, fallback=None, render=True, options=None):
    """The mod's thumbnail: its map, square, with its name across it.

    `fallback` is the picture to use when the map isn't rendered (any PIL
    image -- the Countries tab's render); the name is written on it all the
    same. Returns (image, report): which picture it is, and how long it took.
    """
    t0 = time.time()
    report = {"source": "map render", "size": size}
    img = None
    if not render:
        report["note"] = "rendering the map for the thumbnail is turned off"
    else:
        try:
            img = render_map(mod_dir, game_dir, size, options)
        except Exception as e:  # a thumbnail is never worth failing an export over
            if fallback is None:
                raise
            report["note"] = f"the map couldn't be rendered ({type(e).__name__}: {e})"
    if img is None:
        report["source"] = "country map"
        img = square(fallback, size)
    try:
        font = find_font(mod_dir, game_dir)
        if font and name.strip():
            img = draw_title(img, name, font)
            report["font"] = os.path.basename(font)
        elif name.strip():
            report["title_error"] = "no map font found: the name isn't written on the thumbnail"
    except Exception as e:
        report["title_error"] = f"the name couldn't be written ({type(e).__name__}: {e})"
    report["seconds"] = round(time.time() - t0, 1)
    return img, report
