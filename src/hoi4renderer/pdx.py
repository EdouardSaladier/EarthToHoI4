"""Minimal parser for Paradox (Clausewitz) script files and localisation."""
import os
import re

_TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"|[{}]|[<>!]?=|[<>]|[^\s{}=<>"#]+|#[^\n]*')


def read_text(path):
    with open(path, "rb") as f:
        data = f.read()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1")


def _tokens(text):
    for m in _TOKEN_RE.finditer(text):
        t = m.group(0)
        if t[0] == "#":
            continue
        yield t


class Block(list):
    """Ordered list of (key, op, value) entries. Bare values use key=None."""

    def get(self, key, default=None):
        for k, _, v in self:
            if k == key:
                return v
        return default

    def get_all(self, key):
        return [v for k, _, v in self if k == key]

    def values(self):
        return [v if k is None else k for k, _, v in self]


def _unquote(t):
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        return t[1:-1]
    return t


def parse(text):
    toks = list(_tokens(text))
    pos = 0
    n = len(toks)

    def parse_block():
        nonlocal pos
        blk = Block()
        while pos < n:
            t = toks[pos]
            if t == "}":
                pos += 1
                return blk
            if t == "{":
                pos += 1
                blk.append((None, None, parse_block()))
                continue
            pos += 1
            if pos < n and toks[pos] in ("=", "<", ">", "<=", ">=", "!=", "?="):
                op = toks[pos]
                pos += 1
                if pos >= n:
                    break
                v = toks[pos]
                pos += 1
                if v == "{":
                    v = parse_block()
                elif pos < n and toks[pos] == "{" and v in ("rgb", "hsv", "RGB", "HSV", "hsv360"):
                    # color = rgb { r g b }
                    pos += 1
                    inner = parse_block()
                    inner.insert(0, ("__colorspace__", "=", v.lower()))
                    v = inner
                else:
                    v = _unquote(v)
                blk.append((_unquote(t), op, v))
            else:
                blk.append((None, None, _unquote(t)))
        return blk

    root = Block()
    while pos < n:
        b = parse_block()
        root.extend(b)
        if pos < n and toks[pos] == "}":  # stray closing brace
            pos += 1
    return root


def parse_file(path):
    return parse(read_text(path))


def parse_color(v):
    """Return (r, g, b) 0-255 from a parsed color value, or None."""
    if not isinstance(v, Block):
        return None
    space = v.get("__colorspace__", "rgb")
    nums = []
    for k, _, x in v:
        if k is None:
            try:
                nums.append(float(x))
            except (TypeError, ValueError):
                pass
    if len(nums) < 3:
        return None
    if space.startswith("hsv"):
        import colorsys
        h, s, val = nums[:3]
        if space == "hsv360":
            h, s, val = h / 360.0, s / 100.0, val / 100.0
        r, g, b = colorsys.hsv_to_rgb(h, s, val)
        return int(r * 255 + 0.5), int(g * 255 + 0.5), int(b * 255 + 0.5)
    r, g, b = nums[:3]
    if max(r, g, b) <= 1.0 and any(isinstance(x, float) and not float(x).is_integer() for x in (r, g, b)):
        r, g, b = r * 255, g * 255, b * 255
    return int(r), int(g), int(b)


_LOC_RE = re.compile(r'^\s*([^\s:#]+):\d*\s*"(.*)"\s*(?:#.*)?$')


def parse_localisation(path, into):
    try:
        text = read_text(path)
    except OSError:
        return
    for line in text.splitlines():
        m = _LOC_RE.match(line)
        if m:
            into[m.group(1)] = m.group(2)


def date_key(s):
    """'1936.1.1' -> (1936, 1, 1) or None."""
    parts = str(s).split(".")
    if len(parts) < 2 or not all(p.lstrip("-").isdigit() for p in parts if p):
        return None
    vals = [int(p) for p in parts if p] + [1, 1]
    return tuple(vals[:3])


class VFS:
    """Merges the game directory with a mod directory, honouring replace_path."""

    def __init__(self, game_dir, mod_dir, replace_paths=()):
        self.game = game_dir
        self.mod = mod_dir
        self.replace = {p.replace("\\", "/").strip("/").lower() for p in replace_paths}

    @staticmethod
    def from_mod(game_dir, mod_dir):
        replace = []
        desc = os.path.join(mod_dir, "descriptor.mod")
        if os.path.isfile(desc):
            for k, _, v in parse_file(desc):
                if k == "replace_path":
                    replace.append(v)
        return VFS(game_dir, mod_dir, replace)

    def find(self, rel):
        """Path of a single file, mod first."""
        rel = rel.replace("\\", "/")
        for base in (self.mod, self.game):
            if base:
                p = os.path.join(base, rel)
                if os.path.isfile(p):
                    return p
        return None

    def list_dir(self, rel, exts=None):
        """Merged listing of a folder: {filename: path}, sorted by filename."""
        rel = rel.replace("\\", "/").strip("/")
        out = {}
        bases = [self.mod] if rel.lower() in self.replace else [self.game, self.mod]
        for base in bases:
            if not base:
                continue
            d = os.path.join(base, rel)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                p = os.path.join(d, fn)
                if os.path.isfile(p) and (exts is None or fn.lower().endswith(exts)):
                    out[fn] = p
        return dict(sorted(out.items()))
