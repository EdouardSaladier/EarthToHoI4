"""
Country colour palettes: HOI4 colour presets read from the `colors/`
folder, user presets, and the active colour configuration.

The `colors/` folder holds one sub-folder per game/mod (Base Game,
Kaiserreich, Millenium Dawn, The New Order, Magna Europa), each laid out
like HOI4's `common/`: `countries/colors.txt` gives each HOI4 tag's map
colour (`TAG = { color = rgb { r g b } ... }`, sometimes `hsv { h s v }`),
`countries/<Name>.txt` a fallback `color = { r g b }`, and
`country_tags/*.txt` maps tags to those files (`GER = "countries/Germany.txt"`).

HOI4 tags aren't ISO codes (GER, ENG, SOV, SPR, HOL, PER, ...), so each mod's
tags are resolved to GADM GID_0 codes by, in priority order:
  1. the tag's country file name matched to a GADM country name (with
     HOI4_NAME_ALIASES for historical/misspelt names: "Soviet Union",
     "Holland", "Persia", "Guatemla", ...);
  2. HOI4_TAG_TO_ISO, the standard vanilla tags (mods keep most of them --
     this is what resolves Kaiserreich, whose country files are regional
     like "WestEur.txt", and The New Order);
  3. the majority mapping of that tag across the other mods (step 1).
When several tags land on one country, the best-priority one wins, then the
first listed in country_tags. Tags with no GADM counterpart (Bavaria,
Reichskommissariats, cosmetic tags, ...) are simply not used.

Built-in presets are saved to `color_presets/builtin/<name>.json` the first
time they're built (or with `python palettes.py --rebuild`), and loaded from
there afterwards: `colors/` is only needed to (re)build them.

Colours must stay unique per country -- the land/sea pipeline in core.py
identifies countries by colour -- so effective_colors() nudges duplicates
(and anything equal to the background) by a unit or two.
"""

from __future__ import annotations

import colorsys
import json
import os
import re
import stat
import threading
import unicodedata
from collections import Counter
from pathlib import Path

import core

COLORS_DIR = core.BASE_DIR / "colors"
USER_PRESETS_DIR = core.BASE_DIR / "color_presets"
# Built-in presets are saved here as JSON (one file per preset), so the app
# doesn't need `colors/` at runtime -- it can be deleted from an install.
BUILTIN_PRESETS_DIR = USER_PRESETS_DIR / "builtin"
CONFIG_PATH = core.BASE_DIR / "color_config.json"

MODS = ["Base Game", "Kaiserreich", "Millenium Dawn", "The New Order", "Magna Europa"]
DEFAULT_PRESET = "Default"

# Historical / misspelt HOI4 country-file names -> GADM GID_0.
HOI4_NAME_ALIASES = {
    "soviet union": "RUS", "luxemburg": "LUX", "holland": "NLD", "czechoslovakia": "CZE",
    "yugoslavia": "SRB", "guatemla": "GTM", "korea": "KOR", "persia": "IRN", "siam": "THA",
    "venezula": "VEN", "usa": "USA", "yeman": "YEM", "british raj": "IND", "palestine": "PSE",
    "burma": "MMR", "ivory coast": "CIV", "dahomey": "BEN", "upper volta": "BFA", "bosnia": "BIH",
    "cabo verde": "CPV", "solomon": "SLB", "cayenne": "GUF", "french guyana": "GUF",
    "isle of mann": "IMN", "holy see": "VAT", "vatican": "VAT", "macau": "MAC", "east timor": "TLS",
    "st.kitts and nevis": "KNA", "st.lucia": "LCA", "st.vincent and the grenadines": "VCT",
    "trinidad": "TTO", "sao tome": "STP", "sahrawi": "ESH", "formosa": "TWN", "tahiti": "PYF",
    "micronesian federation": "FSM", "outer mongolia": "MNG",
}

# Standard (vanilla) HOI4 tags -> GADM GID_0; used when a mod's file names
# don't identify the country (Kaiserreich, The New Order).
HOI4_TAG_TO_ISO = {
    "GER": "DEU", "ENG": "GBR", "SOV": "RUS", "FRA": "FRA", "ITA": "ITA", "JAP": "JPN", "USA": "USA",
    "CAN": "CAN", "AST": "AUS", "NZL": "NZL", "SAF": "ZAF", "RAJ": "IND", "CHI": "CHN", "SPR": "ESP",
    "POR": "PRT", "HOL": "NLD", "BEL": "BEL", "LUX": "LUX", "SWI": "CHE", "AUS": "AUT", "CZE": "CZE",
    "SLO": "SVK", "HUN": "HUN", "POL": "POL", "ROM": "ROU", "YUG": "SRB", "SER": "SRB", "CRO": "HRV",
    "SLV": "SVN", "BOS": "BIH", "MNT": "MNE", "BUL": "BGR", "GRE": "GRC", "ALB": "ALB", "TUR": "TUR",
    "DEN": "DNK", "NOR": "NOR", "SWE": "SWE", "FIN": "FIN", "EST": "EST", "LAT": "LVA", "LIT": "LTU",
    "IRE": "IRL", "ICE": "ISL", "UKR": "UKR", "BLR": "BLR", "MOL": "MDA", "GEO": "GEO", "ARM": "ARM",
    "AZR": "AZE", "KAZ": "KAZ", "UZB": "UZB", "TMS": "TKM", "KYR": "KGZ", "TAJ": "TJK", "MON": "MNG",
    "SIA": "THA", "INS": "IDN", "PHI": "PHL", "MAL": "MYS", "VIN": "VNM", "LAO": "LAO", "CAM": "KHM",
    "BRM": "MMR", "NEP": "NPL", "BHU": "BTN", "AFG": "AFG", "PER": "IRN", "IRQ": "IRQ", "SYR": "SYR",
    "LEB": "LBN", "JOR": "JOR", "SAU": "SAU", "YEM": "YEM", "OMA": "OMN", "ISR": "ISR", "PAL": "PSE",
    "EGY": "EGY", "LBA": "LBY", "ETH": "ETH", "LIB": "LBR", "MEX": "MEX", "GUA": "GTM", "HON": "HND",
    "ELS": "SLV", "NIC": "NIC", "COS": "CRI", "PAN": "PAN", "CUB": "CUB", "HAI": "HTI", "DOM": "DOM",
    "COL": "COL", "VEN": "VEN", "ECU": "ECU", "PRU": "PER", "BOL": "BOL", "CHL": "CHL", "ARG": "ARG",
    "URG": "URY", "PAR": "PRY", "BRA": "BRA", "KOR": "KOR", "FOR": "TWN", "ALG": "DZA", "MOR": "MAR",
    "TUN": "TUN", "RUS": "RUS",
}

# Per-mod corrections, applied before everything else (None = not a GADM
# country): tags whose meaning in that mod differs from what the cross-mod
# vote would infer.
MOD_TAG_OVERRIDES = {
    "The New Order": {"IND": "IND", "BUR": "MMR", "SAM": None, "JER": None},
    "Millenium Dawn": {"CAS": None, "GNA": None},
}

# Default preset: Magna Europa for Europe, Millenium Dawn for Africa and the
# Middle East, Kaiserreich elsewhere -- each falling back to the other
# presets, in order.
EUROPE = set("""ALA ALB AND AUT BEL BGR BIH BLR CHE CYP CZE DEU DNK ESP EST FIN FRA FRO GBR GGY GIB GRC HRV
HUN IMN IRL ISL ITA JEY LIE LTU LUX LVA MCO MDA MKD MLT MNE NLD NOR POL PRT ROU RUS SJM SMR SRB SVK SVN
SWE UKR VAT XKO XNC XAD ARM AZE GEO TUR""".split())  # + Caucasus/Turkey: Magna Europa covers them
AFRICA = set("""AGO BDI BEN BFA BWA CAF CIV CMR COD COG COM CPV DJI DZA EGY ERI ESH ETH GAB GHA GIN GMB GNB
GNQ KEN LBR LBY LSO MAR MDG MLI MOZ MRT MUS MWI MYT NAM NER NGA REU RWA SDN SEN SHN SLE SOM SSD STP SWZ
SYC TCD TGO TUN TZA UGA ZAF ZMB ZWE""".split())
MIDDLE_EAST = set("ARE BHR IRN IRQ ISR JOR KWT LBN OMN PSE QAT SAU SYR YEM".split())
DEFAULT_ORDER = {
    "europe": ["Magna Europa", "Base Game", "Millenium Dawn", "Kaiserreich", "The New Order"],
    "africa": ["Millenium Dawn", "Magna Europa", "Base Game", "Kaiserreich", "The New Order"],
    "middle_east": ["Millenium Dawn", "Magna Europa", "Base Game", "Kaiserreich", "The New Order"],
    "world": ["Kaiserreich", "Base Game", "Millenium Dawn", "Magna Europa", "The New Order"],
}

# Autonomous territories GADM keeps as separate "countries", drawn as part
# of their country when merge_autonomous is on (see core.render_map).
AUTONOMOUS_PARENT = {
    "ALA": "FIN",  # Aland
    "FRO": "DNK", "GRL": "DNK",  # Faroe Islands, Greenland
    "HKG": "CHN", "MAC": "CHN",  # Hong Kong, Macao
    "JEY": "GBR", "GGY": "GBR", "IMN": "GBR",  # Jersey, Guernsey, Isle of Man
    "ABW": "NLD", "CUW": "NLD", "SXM": "NLD",  # Aruba, Curacao, Sint Maarten
}

_BLOCK = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*\{(.*?)^\s*\}", re.S | re.M)
_COLOR = re.compile(r"(?<![_a-z])color\s*=\s*(rgb|hsv)?\s*\{\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\}", re.I)
_TAGLINE = re.compile(r'^\s*([A-Z][A-Z0-9]{2})\s*=\s*"([^"]+)"', re.M)
_lock = threading.Lock()
_builtin_cache = None


def _read(path: Path) -> str:
    return re.sub(r"#[^\n]*", "", path.read_text(encoding="utf-8", errors="replace"))


def _parse_color(m) -> tuple:
    kind = (m.group(1) or "").lower()
    vals = [float(m.group(i)) for i in (2, 3, 4)]
    if kind == "hsv":
        return tuple(int(round(v * 255)) for v in colorsys.hsv_to_rgb(*vals))
    if max(vals) <= 1.0 and any("." in m.group(i) for i in (2, 3, 4)):
        return tuple(int(round(v * 255)) for v in vals)
    return tuple(max(0, min(255, int(round(v)))) for v in vals)


def _norm(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().strip()


def _key(name: str) -> str:
    return re.sub(r"[^a-z]", "", _norm(name))


def _load_mod(mod: str):
    """(tag -> country file name, tag -> rgb) for one mod folder."""
    root = COLORS_DIR / mod
    files = {}
    for f in sorted((root / "country_tags").glob("*.txt")):
        for tag, path in _TAGLINE.findall(_read(f)):
            files.setdefault(tag, path)
    colors = {}
    cf = root / "countries" / "colors.txt"
    if cf.exists():
        for tag, body in _BLOCK.findall(_read(cf)):
            m = _COLOR.search(body)
            if m and re.fullmatch(r"[A-Z][A-Z0-9]{2}", tag):
                colors[tag] = _parse_color(m)
    for tag, path in files.items():  # fallback: the country file's own colour
        if tag not in colors:
            f = root / path
            if f.exists():
                m = _COLOR.search(_read(f))
                if m:
                    colors[tag] = _parse_color(m)
    return files, colors


def build_builtin_presets(geometries) -> dict:
    """{preset name: {GID_0: [r, g, b]}} for every mod folder + "Default"."""
    by_name = {_key(d["name"]): t for t, d in geometries.items() if t not in core.WATER_TAGS}
    mods = {m: _load_mod(m) for m in MODS if (COLORS_DIR / m).is_dir()}

    def name_match(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return by_name.get(_key(stem)) or HOI4_NAME_ALIASES.get(_norm(stem))

    votes = {}
    for files, _colors in mods.values():
        for tag, path in files.items():
            iso = name_match(path)
            if iso:
                votes.setdefault(tag, Counter())[iso] += 1

    presets = {}
    for mod, (files, colors) in mods.items():
        order = {t: i for i, t in enumerate(files)}
        best = {}  # iso -> (priority, order, rgb)
        fixes = MOD_TAG_OVERRIDES.get(mod, {})
        for tag, rgb in colors.items():
            cands = []
            if tag in fixes:
                if fixes[tag] is None:
                    continue
                cands.append((-1, fixes[tag]))
            if tag in files and name_match(files[tag]):
                cands.append((0, name_match(files[tag])))
            if tag in HOI4_TAG_TO_ISO:
                cands.append((1, HOI4_TAG_TO_ISO[tag]))
            if tag in votes:
                cands.append((2, votes[tag].most_common(1)[0][0]))
            if not cands:
                continue
            prio, iso = min(cands)
            if iso not in geometries:
                continue
            rank = (prio, order.get(tag, 10 ** 6))
            if iso not in best or rank < best[iso][0]:
                best[iso] = (rank, list(rgb))
        presets[mod] = {iso: v[1] for iso, v in sorted(best.items())}

    default = {}
    for iso in sorted(geometries):
        region = ("europe" if iso in EUROPE else "africa" if iso in AFRICA
                  else "middle_east" if iso in MIDDLE_EAST else "world")
        for mod in DEFAULT_ORDER[region]:
            if iso in presets.get(mod, {}):
                default[iso] = presets[mod][iso]
                break
    presets[DEFAULT_PRESET] = default
    return presets


def _builtin_names() -> list:
    return [DEFAULT_PRESET] + MODS


def save_builtin_presets(presets: dict):
    """Write the built-in presets, read-only on disk (a safety net against
    deleting them by hand: the app never deletes them, and without colors/
    they can't be rebuilt)."""
    BUILTIN_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    for name, colors in presets.items():
        payload = {"name": name, "builtin": True,
                   "source": "Default mix" if name == DEFAULT_PRESET else f"colors/{name}",
                   "colors": {k: list(v) for k, v in sorted(colors.items())}}
        path = BUILTIN_PRESETS_DIR / f"{name}.json"
        if path.exists():
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE)  # lift our own read-only flag to rewrite it
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.chmod(path, stat.S_IREAD)


def is_builtin_name(name: str) -> bool:
    """A built-in preset's name, ignoring case and outer spaces -- so no
    user preset can be saved as a look-alike ("default", "KAISERREICH")
    that sits next to the real one in the list."""
    key = name.strip().casefold()
    return any(key == n.casefold() for n in _builtin_names())


def rebuild_builtin_presets() -> dict:
    """Re-parse `colors/` and overwrite the saved built-in presets."""
    if not COLORS_DIR.is_dir():
        raise FileNotFoundError(f"{COLORS_DIR} not found: can't rebuild the built-in presets")
    presets = build_builtin_presets(core.load_geometries(progress=lambda *a: None))
    save_builtin_presets(presets)
    return presets


def builtin_presets() -> dict:
    """The built-in presets, loaded from BUILTIN_PRESETS_DIR. They're only
    (re)built from `colors/` when a saved file is missing and that folder
    still exists; without either, the missing presets are simply absent."""
    global _builtin_cache
    with _lock:
        if _builtin_cache is None:
            files = {n: BUILTIN_PRESETS_DIR / f"{n}.json" for n in _builtin_names()}
            if not all(f.exists() for f in files.values()) and COLORS_DIR.is_dir():
                save_builtin_presets(build_builtin_presets(core.load_geometries(progress=lambda *a: None)))
            presets = {}
            for name, f in files.items():
                if f.exists():
                    presets[name] = {k: list(v) for k, v in json.loads(f.read_text(encoding="utf-8"))["colors"].items()}
            _builtin_cache = presets
        return _builtin_cache


_SAFE_NAME = re.compile(r"^[\w .()\-]{1,60}$")


def _user_preset_path(name: str) -> Path:
    if not _SAFE_NAME.fullmatch(name) or name.strip() != name or name in (".", ".."):
        raise ValueError("Preset names may use letters, digits, spaces and . _ - ( ), up to 60 characters")
    return USER_PRESETS_DIR / f"{name}.json"


def list_presets() -> list:
    out = [{"name": n, "builtin": True, "count": len(c)} for n, c in builtin_presets().items()]
    out.sort(key=lambda p: (p["name"] != DEFAULT_PRESET, p["name"]))
    if USER_PRESETS_DIR.is_dir():
        for f in sorted(USER_PRESETS_DIR.glob("*.json")):
            try:
                colors = json.loads(f.read_text(encoding="utf-8")).get("colors", {})
            except (OSError, ValueError):
                continue
            out.append({"name": f.stem, "builtin": False, "count": len(colors)})
    return out


def get_preset(name: str) -> dict:
    if name in builtin_presets():
        return builtin_presets()[name]
    path = _user_preset_path(name)
    if not path.exists():
        raise KeyError(f"No colour preset named '{name}'")
    return {k: list(v) for k, v in json.loads(path.read_text(encoding="utf-8")).get("colors", {}).items()}


def save_user_preset(name: str, colors: dict):
    if is_builtin_name(name):
        raise ValueError(f"'{name}' is the name of a built-in preset; pick another name")
    path = _user_preset_path(name)
    USER_PRESETS_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps({"name": name, "colors": colors}, indent=1, sort_keys=True), encoding="utf-8")


def delete_user_preset(name: str):
    """Delete a user preset. Built-in presets can never be deleted: their
    names are refused (ignoring case), and a user preset's path is always
    outside color_presets/builtin/."""
    if is_builtin_name(name):
        raise ValueError("Built-in presets can't be deleted")
    path = _user_preset_path(name)
    if path.resolve().parent != USER_PRESETS_DIR.resolve():
        raise ValueError("Built-in presets can't be deleted")
    if not path.exists():
        raise KeyError(f"No colour preset named '{name}'")
    path.unlink()


def load_config() -> dict:
    """Active colour configuration: {"preset": name | None, "overrides": {GID_0: [r, g, b]}}."""
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return {"preset": cfg.get("preset"), "overrides": cfg.get("overrides", {})}
        except (OSError, ValueError):
            pass
    return {"preset": DEFAULT_PRESET, "overrides": {}}


def save_config(cfg: dict):
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(CONFIG_PATH)


def valid_rgb(rgb) -> list:
    try:
        vals = [int(v) for v in rgb]
    except (TypeError, ValueError):
        raise ValueError(f"Invalid RGB value {rgb!r}")
    if len(vals) != 3 or not all(0 <= v <= 255 for v in vals):
        raise ValueError(f"RGB components must be three integers 0-255, got {rgb!r}")
    return vals


def effective_colors(geometries, cfg=None) -> tuple:
    """(colours {GID_0: (r, g, b)}, nudged {GID_0: requested rgb}) for the
    active configuration: tool colours <- preset <- per-country overrides,
    then made unique (duplicates and the background get nudged)."""
    cfg = load_config() if cfg is None else cfg
    base = dict(core.assign_colors(geometries))
    preset = {}
    if cfg.get("preset"):
        try:
            preset = get_preset(cfg["preset"])
        except (KeyError, ValueError):
            preset = {}
    wanted = {t: tuple(preset.get(t, base[t])) for t in base}
    for t, rgb in cfg.get("overrides", {}).items():
        if t in wanted:
            wanted[t] = tuple(rgb)
    # Overrides first, then preset colours, then the rest: a colour the user
    # picked is the one that stays exact if something must be nudged.
    order = sorted(wanted, key=lambda t: (t not in cfg.get("overrides", {}), t not in preset, t))
    used = {tuple(core.BACKGROUND_COLOR), tuple(core.LAKE_COLOR), tuple(core.EDGE_LAND_COLOR)}
    out, nudged = {}, {}
    for t in order:
        rgb = wanted[t]
        cand, step = rgb, 0
        while cand in used:
            step += 1
            d = (step + 1) // 2 * (1 if step % 2 else -1)
            channel = (step - 1) // 16 % 3
            cand = list(rgb)
            cand[channel] = max(0, min(255, cand[channel] + d))
            cand = tuple(cand)
        if cand != rgb:
            nudged[t] = list(rgb)
        used.add(cand)
        out[t] = cand
    return out, nudged


if __name__ == "__main__":
    import sys
    if "--rebuild" in sys.argv:
        built = rebuild_builtin_presets()
        for n, c in built.items():
            print(f"{n}: {len(c)} countries -> {BUILTIN_PRESETS_DIR / (n + '.json')}")
    else:
        print("Usage: python palettes.py --rebuild   (re-parse colors/ into color_presets/builtin/)")
