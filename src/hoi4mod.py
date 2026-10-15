"""HOI4 mod export: everything the tool generates for one view, written as a
complete Hearts of Iron IV mod (map + states + countries + strategic regions)
with the base game's country-specific content neutralised.

User's scope (2026-10-11): "a working map that can be launched, content will
come later"; no industry / resources for now; one infantry division per
country (the AI breaks without an OOB); file names ASCII only -- every real
name goes in the localisation.

Countries get their own tags, never one of the game's (vanilla content is
keyed on tags: a "GER" here would get Germany's events / decisions). A
country with a vanilla equivalent (palettes.HOI4_TAG_TO_ISO) borrows its
flags, names and graphical culture.

Neutralising: replace_path on the content folders (focus trees, events,
decisions, on_actions, history, AI strategies, bookmarks, strategic regions,
supply areas); the generic vanilla files of those folders are copied back
with every event / decision / effect / focus that names a specific country
or state number removed (the base game's state and province ids mean other
places here: its on_startup builds the Kiel canal in "state 58").

build_mod(src, settings, out_dir, game_dir) writes the mod folder; app.py
collects `src` from the tabs' cached results."""
import json
import math
import os
import re
import shutil
import struct
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import pdx

Image.MAX_IMAGE_PIXELS = None

LANGUAGES = ("english", "french", "german", "spanish", "polish", "russian", "braz_por", "japanese", "korean",
             "simp_chinese")
IDEOLOGIES = ("democratic", "fascism", "communism", "neutrality")
VANILLA_W, VANILLA_H = 5632, 2048

MOD_DEFAULTS = {
    "name": "Map Tool World",       # shown in the launcher; the folder is its ASCII form
    "manpower_pct": 100,            # state manpower = population x this %
    "infrastructure": 2,            # every owned state
    "thumbnail": 1,                 # render the map for thumbnail.png (adds ~20 s)
}
MOD_LIMITS = {"manpower_pct": (1, 1000), "infrastructure": (0, 5), "thumbnail": (0, 1)}

# Folders whose base-game content is unloaded (replace_path); KEEP lists the
# generic files copied back (sanitised, see _sanitise).
REPLACE_PATHS = [
    "common/national_focus", "events", "common/decisions", "common/on_actions", "common/ai_strategy",
    "common/ai_areas", "common/bookmarks", "history/countries", "history/states", "history/units",
    "history/general", "map/strategicregions", "map/supplyareas",
]
KEEP = {
    "common/national_focus": ("generic.txt", "00_titlebar_styles.txt"),
    "events": ("Generic.txt", "ElectionEvents.txt", "CapitulationEvents.txt", "WarJustification.txt",
               "AcePilots.txt", "NukeEvents.txt", "Decisions_Events.txt", "stability_events.txt",
               "LaR_agent_events.txt", "LaR_espionage_operations.txt", "Special_Project_Events.txt",
               "GOT_Generic_Events.txt", "PoliticalEvents.txt", "MTG_generic.txt", "AAT_Generic_Events.txt",
               "GOE_generic_events.txt", "SEA_generic_events.txt", "TOA_Generic_Events.txt"),
    "common/decisions": ("_generic_decisions.txt", "_exiled_governments_decisions.txt", "foreign_influence.txt",
                         "lar_agent_recruitment_decisions.txt", "political_decisions.txt",
                         "stability_war_support.txt"),
    "common/on_actions": ("00_on_actions.txt",),
    "common/ai_strategy": ("default.txt", "doctrines.txt", "generic_operation_strats.txt",
                           "operatives_default.txt"),
}
# the unit each kept file is filtered by: dropped whole when it names a country / state
EVENT_KEYS = {"country_event", "news_event", "state_event", "unit_leader_event", "operative_leader_event"}
NOT_TAGS = {"AND", "NOT", "NUL", "ALL", "ANY", "AUX", "PRN", "CON"}
RESERVED_TAGS = NOT_TAGS | {"NAN", "INF", "NUM", "LPT", "COM"}

DEFAULT_TECH = ("infantry_weapons", "infantry_weapons1", "tech_support", "tech_engineers", "tech_trucks",
                "gw_artillery", "basic_train")
CONTINENT_GFX = {  # graphical culture for countries with no vanilla equivalent, by ISO continent
    "eu": ("western_european_gfx", "western_european_2d"),
    "as": ("asian_gfx", "asian_2d"),
    "af": ("african_gfx", "african_2d"),
    "me": ("middle_eastern_gfx", "middle_eastern_2d"),
    "sa": ("southamerican_gfx", "southamerican_2d"),
    "na": ("western_european_gfx", "western_european_2d"),
    "oc": ("commonwealth_gfx", "commonwealth_2d"),
}
# vanilla map georeference (_dev/vanilla_georef.py: 886 victory points matched to
# GeoNames, median error 5 px): column = X0 + XL * lon, row = poly(lat)
GEO_X = (2805.792, 15.6099)
GEO_Y = (-3.39626749e-07, 3.02025325e-05, -1.65003656e-04, -1.09411967e-01, -1.26069263e+01, 1.37136543e+03)

# state categories by manpower (people), smallest first
CATEGORIES = ((50_000, "pastoral"), (300_000, "rural"), (800_000, "town"), (1_600_000, "large_town"),
              (3_000_000, "city"), (6_000_000, "large_city"), (12_000_000, "metropolis"))
VP_BY_CLASS = {1: 50, 2: 20, 3: 10, 4: 5, 5: 3}  # City tab class -> VPs (user), else 1

LAND, SEA, LAKE = 0, 1, 2
LAND_LEVEL = 96   # heightmap grey: positions on land need at least this (sea level 9.5 = 95; coast pixels 94)
MIN_MANPOWER = 1000  # every state (vanilla's impassable ones have thousands)
GOOD_SCORE = 18   # Geo.score of a pixel on land and >= 2 px inside its province


class ExportError(Exception):
    pass


# ------------------------------------------------------------------ helpers
def ascii_name(s, maxlen=60):
    """File-name-safe ASCII (user: no special characters in file names)."""
    try:
        from unidecode import unidecode
        s = unidecode(s)
    except ImportError:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9_\- ]+", "", s).strip()
    s = re.sub(r"\s+", "_", s)
    return s[:maxlen] or "unnamed"


_GLYPHS = [None]  # characters every font of the mod has (set by write_fonts); None = no folding


def _fold_char(ch):
    """A character the fonts lack -> the closest they have: its base letter
    (NFKD: H with dot below -> H), else unidecode's transliteration."""
    base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
    if base and all(ord(c) in _GLYPHS[0] for c in base):
        return base
    try:
        from unidecode import unidecode
        return unidecode(ch)
    except ImportError:
        return base.encode("ascii", "ignore").decode()


def loc_text(s):
    s = str(s).replace('"', "'").replace("\n", " ")
    if _GLYPHS[0] is not None:
        s = "".join(c if ord(c) in _GLYPHS[0] else _fold_char(c) for c in s)
    return s


def _write(path, text, bom=False, crlf=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if crlf:
        text = text.replace("\r\n", "\n").replace("\n", "\r\n")
    with open(path, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as f:
        f.write(text)


def px_lonlat(view, xs, ys):
    """Pixel centres of the render's frame -> (lon, lat) arrays."""
    import core
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    lon, lat, _in = core._unproject(view["projection"], view["center"],
                                    view["win_xmin"] + (xs + 0.5) / view["scale"],
                                    view["win_ymin"] + (view["height"] - (ys + 0.5)) / view["scale"])
    return lon, lat


def _km(lon1, lat1, lon2, lat2):
    la1, la2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


# ------------------------------------------------------------------ state names
# user (2026-10-15): "remove the state naming rules, use directly the proper
# GADM administrative names without changes, endonyms if possible".
# GADM's NAME_<level> *is* the endonym, romanized (Bayern, Toscana, Fejer,
# Moskovskaya) -- its NL_NAME_<level> native-script field covers only 15% of
# the units, is almost never Latin script (so the mod's fonts have no glyphs
# for it) and carries junk rows, so it is not used. Everything the export used
# to do to a GADM name is gone: the per-country naming level (France's
# departements), the merged-state "whole unit" rule, the suffix / "Unorganized
# in" / trailing-number clean-up, Russia's dropped "rayon", Ukraine's
# adjectival raions resolved to a town, the parent / compass disambiguation of
# repeated names, and the capital state taking its victory point's name.
# A division's name is now whatever GADM calls it, repeats included -- each
# state has its own STATE_<id> localisation key, so duplicates are legal.
_NO_NAME = re.compile(r"^\s*(n\.\s*a\.?|unknown)?\s*$", re.I)


def _gadm_names(levels):
    """{level: {GID without its _version suffix: GADM's NAME_<level>}}."""
    import states as states_mod
    out = {}
    for lv in levels:
        idx = states_mod.level_index(lv)
        if idx is not None:
            out[lv] = {g.rsplit("_", 1)[0]: nm for g, nm in zip(idx["gid"].tolist(), idx["name"].tolist())}
    return out


def _gid_at(gid, level):
    parts = gid.rsplit("_", 1)[0].split(".")
    return ".".join(parts[:level + 1]) if len(parts) > level else None


def name_states(states):
    """Sets each owned state's "name" to the GADM name of the division it was
    built from, spelled exactly as GADM spells it. A merged state takes the
    unit its own `gid` names. The only fallback: a unit GADM leaves unnamed
    ("n.a." or blank -- 5 of its 45,962 level-2 units, mostly the Emirates'
    sectors and the Dutch lake "provinces", and 2,838 of its level-3 ones)
    takes its parent division's name, since a state with no name at all
    cannot be localised.
    """
    owned = [s for s in states if s.get("gid") and not s["impassable"]]
    levels = sorted({lv for s in owned for lv in range(0, int(s.get("level") or 0) + 1)})
    tables = _gadm_names(levels)
    for s in owned:
        lv = int(s.get("level") or 0)
        while lv >= 0:
            g = _gid_at(s["gid"], lv)
            nm = str(tables.get(lv, {}).get(g) or "").strip() if g else ""
            if nm and not _NO_NAME.match(nm):
                s["name"] = nm
                break
            lv -= 1



def name_regions(regions, state_of, geo):
    """Strategic region names from the final state names (strategic.py's
    rule): land = its largest state; sea = "<largest island state> waters",
    else "<state on its longest coast> coast", else "Open sea"; numbered when
    repeated."""
    seen = defaultdict(int)
    for g in regions:
        provs = [p for p in g["provinces"] if 0 < p < geo.n]
        area = defaultdict(int)
        for p in provs:
            s = state_of.get(p)
            if s is not None and s.get("owner"):
                area[id(s)] += int(geo.area[p])
        by_id = {id(state_of[p]): state_of[p] for p in provs if p in state_of}
        if area:
            nm = by_id[max(area, key=area.get)]["name"]
            name = nm if g["kind"] == "land" else f"{nm} waters"
        elif g["kind"] == "land":
            name = "Unowned land"
        else:
            contact = defaultdict(int)
            inreg = set(provs)
            for p in provs:
                for q in geo.nbrs.get(p, []):
                    s = state_of.get(q)
                    if q not in inreg and s is not None and s.get("owner"):
                        contact[id(s)] += geo.contact_n.get(p * geo.n + q, 0)
                        by_id[id(s)] = s
            name = f"{by_id[max(contact, key=contact.get)]['name']} coast" if contact else "Open sea"
        seen[name] += 1
        g["name"] = name if seen[name] == 1 else f"{name} {seen[name]}"


def normalize_settings(raw):
    s = dict(MOD_DEFAULTS)
    for k, v in (raw or {}).items():
        if v is None or v == "" or k not in MOD_DEFAULTS:
            continue
        if k == "name":
            s[k] = str(v).strip()[:80] or MOD_DEFAULTS["name"]
            continue
        try:
            val = type(MOD_DEFAULTS[k])(float(v))
        except ValueError:
            raise ExportError(f"'{k}' must be a number")
        lo, hi = MOD_LIMITS[k]
        if not lo <= val <= hi:
            raise ExportError(f"'{k}' must be between {lo} and {hi}")
        s[k] = val
    if not ascii_name(s["name"]).strip("_"):
        raise ExportError("The mod name needs at least one letter or digit")
    return s


# ------------------------------------------------------------------ game
# user-chosen folders (HOI4 mod tab, when detection fails): None = detect
PATHS = {"game": None, "mod": None}


def is_game_dir(p):
    """A Hearts of Iron IV install (its map files, or its executable)."""
    p = Path(p)
    return (p / "map" / "default.map").exists() or (p / "hoi4.exe").exists() or (p / "hoi4").exists()


def _steam_roots():
    home = Path.home()
    return [Path(r"C:\Program Files (x86)\Steam"), Path(r"C:\Program Files\Steam"),
            home / ".steam" / "steam", home / ".local" / "share" / "Steam",
            home / ".var" / "app" / "com.valvesoftware.Steam" / ".local" / "share" / "Steam",
            home / "Library" / "Application Support" / "Steam"]


def detect_game_dir():
    """The HOI4 install: the Steam libraries (Windows, Linux, macOS; every
    library listed in libraryfolders.vdf), then GOG / Microsoft Store folders."""
    cands = []
    for steam in _steam_roots():
        cands.append(steam / "steamapps" / "common" / "Hearts of Iron IV")
        vdf = steam / "steamapps" / "libraryfolders.vdf"
        if vdf.exists():
            for m in re.finditer(r'"path"\s+"([^"]+)"', vdf.read_text(encoding="utf-8", errors="replace")):
                cands.append(Path(m.group(1).replace("\\\\", "\\")) / "steamapps" / "common" / "Hearts of Iron IV")
    for drive in "CDEFG":
        cands += [Path(f"{drive}:/GOG Games/Hearts of Iron IV"), Path(f"{drive}:/XboxGames/Hearts of Iron IV/Content"),
                  Path(f"{drive}:/Program Files (x86)/GOG Galaxy/Games/Hearts of Iron IV")]
    for c in cands:
        try:
            if is_game_dir(c):
                return c
        except OSError:
            continue
    return None


def find_game_dir():
    """The folder chosen in the HOI4 mod tab if valid, else the detected one."""
    if PATHS["game"] and is_game_dir(PATHS["game"]):
        return Path(PATHS["game"])
    return detect_game_dir()


def detect_user_dir():
    """Documents/Paradox Interactive/Hearts of Iron IV (Windows' real
    Documents folder, may be redirected to OneDrive; Linux: ~/.local/share)."""
    docs = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Documents"
    try:  # the real Documents folder (may be redirected, e.g. to OneDrive)
        import ctypes.wintypes
        buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
        ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf)
        if buf.value:
            docs = Path(buf.value)
    except Exception:
        pass
    cands = [docs / "Paradox Interactive" / "Hearts of Iron IV",
             Path.home() / ".local" / "share" / "Paradox Interactive" / "Hearts of Iron IV"]
    return next((c for c in cands if c.exists()), cands[0])


def user_dir():
    """The HOI4 user folder (holds mod/, logs/ ...): the parent of the mod
    folder chosen in the HOI4 mod tab, else the detected one."""
    if PATHS["mod"]:
        return Path(PATHS["mod"]).parent
    return detect_user_dir()


def mod_dir():
    return Path(PATHS["mod"]) if PATHS["mod"] else detect_user_dir() / "mod"


def game_version(game):
    try:
        return json.loads((game / "launcher-settings.json").read_text(encoding="utf-8"))["rawVersion"]
    except Exception:
        return None


class Vanilla:
    """What the export reads from the installed game."""

    def __init__(self, game):
        self.game = Path(game)
        self.tags = {}  # tag -> countries/<file>
        for f in sorted((self.game / "common/country_tags").glob("*.txt")):
            for k, _o, v in pdx.read(f):
                if k and re.fullmatch(r"[A-Z][A-Z0-9]{2}", k) and isinstance(v, str):
                    self.tags[k] = v.strip('"')
        self.dynamic_tags = []
        dyn = self.game / "common/country_tags/zz_dynamic_countries.txt"
        if dyn.exists():
            self.dynamic_tags = [k for k, _o, _v in pdx.read(dyn) if k and re.fullmatch(r"[A-Z][A-Z0-9]{2}", k)]
        self.owners_1936 = set()
        for f in (self.game / "history/states").glob("*.txt"):
            for m in re.finditer(r"\bowner\s*=\s*([A-Z][A-Z0-9]{2})", f.read_text(encoding="utf-8-sig",
                                                                                     errors="replace")):
                self.owners_1936.add(m.group(1))
        self._loc = {}

    def history_file(self, tag):
        hits = sorted((self.game / "history/countries").glob(f"{tag} - *.txt")) + \
            sorted((self.game / "history/countries").glob(f"{tag}-*.txt"))
        return hits[0] if hits else None

    def country_file(self, tag):
        rel = self.tags.get(tag)
        p = self.game / "common" / rel if rel else None
        return p if p and p.exists() else None

    def loc(self, lang):
        """{key: text} of the country name files in one language."""
        if lang not in self._loc:
            out = {}
            d = self.game / "localisation" / lang
            for name in (f"countries_l_{lang}.yml", f"countries_cosmetic_l_{lang}.yml"):
                p = d / name
                if p.exists():
                    for m in re.finditer(r'^\s*([A-Za-z0-9_]+):\d*\s*"(.*)"\s*$',
                                         p.read_text(encoding="utf-8-sig", errors="replace"), re.M):
                        out.setdefault(m.group(1), m.group(2))
            self._loc[lang] = out
        return self._loc[lang]


# ------------------------------------------------------------------ neutralising
STATE_KEYS = {"state", "province", "capital", "set_capital", "owns_state", "controls_state",
              "has_full_control_of_state", "transfer_state", "add_state_core", "remove_state_core",
              "add_state_claim", "remove_state_claim", "target_state", "state_id", "province_id",
              "controls_province", "is_capital_of", "capital_scope", "set_state_owner", "set_state_controller",
              "set_province_controller", "has_state_flag", "is_core_of_state", "location", "naval_base",
              "states", "provinces", "prioritize", "set_state_category", "add_state_modifier"}


def _refs(entries, vtags=None):
    """True if a block names a state or a province by its number -- the base
    game's numbers are other places on this map (its on_startup builds the
    Kiel canal in "state 58"). Country tags are left alone: no country here
    has a vanilla tag, so tag-gated vanilla content can't apply."""
    for k, _o, v in pdx.walk(entries):
        if k is None:
            continue
        if k.isdigit() and isinstance(v, list):  # "58 = { ... }" state scope (a scalar = random_events weight)
            return True
        if k.lower() in STATE_KEYS:
            if isinstance(v, str) and v.isdigit():
                return True
            if isinstance(v, list) and any(e[0] is None and isinstance(e[2], str) and e[2].isdigit() for e in v):
                return True
    return False


def _sanitise(folder, entries, vtags, stats):
    """Drops the units of a kept vanilla file that name a country or a state:
    events, decisions, on_action effects, focuses, AI strategies."""
    def drop(e):
        stats["dropped"] += 1
        return False

    def keep(e):
        stats["kept"] += 1
        return True
    out = []
    if folder == "events":
        for e in entries:
            if e[0] in EVENT_KEYS and isinstance(e[2], list):
                if _refs(e[2], vtags):
                    drop(e)
                    continue
                keep(e)
            out.append(e)
    elif folder == "common/decisions":
        for k, o, v in entries:  # category = { decision = {...} ... }
            if isinstance(v, list):
                v = [d for d in v if not isinstance(d[2], list) or (keep(d) if not _refs(d[2], vtags) else drop(d))]
            out.append((k, o, v))
    elif folder == "common/on_actions":
        for k, o, v in entries:  # on_actions = { on_x = { effect = {..} random_events = {..} } }
            if isinstance(v, list):
                acts = []
                for ak, ao, av in v:
                    if isinstance(av, list):
                        av = [b for b in av if not isinstance(b[2], list)
                              or (keep(b) if not _refs(b[2], vtags) else drop(b))]
                    acts.append((ak, ao, av))
                v = acts
            out.append((k, o, v))
    elif folder == "common/national_focus":
        for k, o, v in entries:
            if k == "focus_tree" and isinstance(v, list):
                focs = {pdx.get(f[2], "id"): f[2] for f in v if f[0] == "focus" and isinstance(f[2], list)}
                gone = {i for i, f in focs.items() if _refs(f, vtags)}
                while True:  # and every focus that needs a removed one
                    more = {i for i, f in focs.items() if i not in gone and any(
                        all(x[2] in gone for x in pre if x[0] == "focus") and any(x[0] == "focus" for x in pre)
                        for pre in pdx.get_all(f, "prerequisite") if isinstance(pre, list))}
                    if not more:
                        break
                    gone |= more
                for f in focs.values():  # references to removed focuses
                    for k2, _o2, me in f:
                        if k2 in ("mutually_exclusive", "prerequisite") and isinstance(me, list):
                            me[:] = [x for x in me if x[2] not in gone]
                    f[:] = [e for e in f if not (e[0] in ("prerequisite", "mutually_exclusive")
                                                 and isinstance(e[2], list) and not e[2])
                            and not (e[0] == "relative_position_id" and e[2] in gone)]
                stats["dropped"] += len(gone)
                v = [f for f in v if not (f[0] == "focus" and pdx.get(f[2], "id") in gone)
                     and not (f[0] == "initial_show_position" and isinstance(f[2], list)
                              and pdx.get(f[2], "focus") in gone)]
                stats["kept"] += sum(1 for f in v if f[0] == "focus")
            out.append((k, o, v))
    else:  # ai_strategy and the like: top-level units
        for e in entries:
            if isinstance(e[2], list) and _refs(e[2], vtags):
                drop(e)
                continue
            keep(e)
            out.append(e)
    return out


def _drop_missing_events(entries, kept_ids, namespaces):
    """on_actions: random_events entries / effects firing an event that isn't
    kept (its file was dropped) are removed."""
    def missing(v):
        return isinstance(v, str) and "." in v and v.split(".")[0] in namespaces and v not in kept_ids
    out = []
    for k, o, v in entries:
        if isinstance(v, list):
            if k == "random_events":
                v = [e for e in v if not missing(e[2])]
                if not any(e[0] and not e[0].isdigit() or not missing(e[2]) for e in v):
                    pass
            elif any(missing(e[2]) for e in pdx.walk(v)):
                continue
            v = _drop_missing_events(v, kept_ids, namespaces)
        elif missing(v):
            continue
        out.append((k, o, v))
    return out


def write_neutralised(out, van, report, our_tags=()):
    vtags = set(van.tags)
    stats = {}
    namespaces, kept_ids = set(), set()
    for f in (van.game / "events").glob("*.txt"):
        for m in re.finditer(r"^\s*add_namespace\s*=\s*([\w-]+)", f.read_text(encoding="utf-8-sig",
                                                                              errors="replace"), re.M):
            namespaces.add(m.group(1))
    for folder, names in KEEP.items():
        for name in names:
            src = van.game / folder / name
            if not src.exists():
                continue
            st = {"kept": 0, "dropped": 0}
            entries = _sanitise(folder, pdx.read(src), vtags, st)
            if folder == "events":
                kept_ids |= {pdx.get(v, "id") for k, _o, v in entries if k in EVENT_KEYS and isinstance(v, list)}
            elif folder == "common/on_actions":
                entries = _drop_missing_events(entries, kept_ids, namespaces)
            if folder == "events":  # namespaces declared in dropped files (BBA_Ethiopia.txt declares
                # generic_purchase_ships_events): every namespace the kept events use
                have = {v for k, _o, v in entries if k == "add_namespace"}
                used = []
                for k, _o, v in entries:
                    if k in EVENT_KEYS and isinstance(v, list):
                        i = pdx.get(v, "id")
                        if i and "." in i and i.split(".")[0] not in have and i.split(".")[0] not in used:
                            used.append(i.split(".")[0])
                entries = [("add_namespace", "=", u) for u in used] + entries
            _write(out / folder / name, pdx.dump(entries))
            stats[f"{folder}/{name}"] = st
    # AI areas: only the continent-based ones (the others list vanilla strategic regions)
    src = van.game / "common/ai_areas/default.txt"
    areas = []
    if src.exists():
        for k, o, v in pdx.read(src):
            if k == "areas" and isinstance(v, list):
                v = [a for a in v if isinstance(a[2], list) and pdx.get(a[2], "strategic_regions") is None]
                areas = [(k, o, v)]
    _write(out / "common/ai_areas/default.txt", pdx.dump(areas or [("areas", "=", [])]))
    _write(out / "common/on_actions/zz_map_tool_on_actions.txt", "on_actions = {\n}\n")
    report["neutralised"] = stats
    report["cleaned"] = clean_base_game(out, van, our_tags)


# base-game folders the cleaner leaves alone: replaced (REPLACE_PATHS), written
# by the export, or not script
CLEAN_SKIP = {"common/country_tags", "common/defines", "common/bookmarks"}
# folders whose files are never emptied as country-specific (the definitions
# of the base game's countries: unused once their tags are gone, harmless)
COUNTRY_FILE_KEEP = {"common/countries"}


def _country_file_names(van):
    names = {re.sub(r"[^a-z0-9]+", "_", f.stem.lower()).strip("_") for f in (van.game / "common/countries").glob("*.txt")}
    return {n for n in names if len(n) > 3}


def is_country_file(rel, tags, names):
    """A base-game file about one country (user: "clean up every
    country-specific file"): named after a tag (GER.txt, GER_organization.txt,
    SWI_scripted_effects.txt) or a country (italy.txt, china_shared.txt,
    british_raj_goe.txt)."""
    stem = Path(rel).stem.strip()
    toks = re.split(r"[_ \-.]", stem)
    if any(t in tags for t in toks):  # GER.txt, SEA_JAP_scripted_loc.txt, templates_GER.txt, 00_GER_names.txt
        return True
    if rel.startswith("common/focus_inlay_windows/") and toks[0].upper() in tags:  # jap_imperial_...
        return True
    low = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return any(low == n or low.startswith(n + "_") or f"_{n}_" in f"_{low}_" for n in names)  # BBA_Switzerland_...


def _event_broken(ev):
    """An event the cleaning left without an option or a description (a
    hidden event needs neither)."""
    if str(pdx.get(ev, "hidden", "no")).lower() == "yes" or str(pdx.get(ev, "is_triggered_only", "")) == "":
        if str(pdx.get(ev, "hidden", "no")).lower() == "yes":
            return False
    return not pdx.get_all(ev, "option") or pdx.get(ev, "desc") is None


# depth of the named definitions in a folder's files (ideas = { category = { NAME = { } } })
DEF_DEPTH = {"common/ideas": 3, "common/characters": 2, "common/decisions": 2}


def _definitions(rel, entries):
    """Names a file defines (top-level blocks; deeper for ideas / characters)."""
    depth = DEF_DEPTH.get(rel.rsplit("/", 1)[0], 1)
    level = [entries]
    for _d in range(depth - 1):
        level = [v for blk in level for _k, _o, v in blk if isinstance(v, list)]
    names = {k for blk in level for k, _o, v in blk
             if k and isinstance(v, list) and not k.isdigit() and k.lower() not in ("if", "else", "else_if")}
    # ... and their ids (autonomy_state = { id = autonomy_puppet })
    names |= {pdx.get(v, "id").strip('"') for blk in level for k, _o, v in blk
              if isinstance(v, list) and isinstance(pdx.get(v, "id"), str) and not pdx.get(v, "id").isdigit()}
    return names


def _keep_definitions(rel, entries, refs):
    """A country-specific file reduced to the definitions (at the folder's
    definition depth) whose name or id is in refs."""
    depth = DEF_DEPTH.get(rel.rsplit("/", 1)[0], 1)

    def walk(blk, level):
        out = []
        for k, o, v in blk:
            if not (k and isinstance(v, list)):
                continue
            if level < depth:
                sub_ = walk(v, level + 1)
                if sub_:
                    out.append((k, o, sub_))
            else:
                ident = pdx.get(v, "id")
                if k in refs or (isinstance(ident, str) and ident.strip('"') in refs):
                    out.append((k, o, v))
        return out
    return walk(entries, 1)


def clean_base_game(out, van, our_tags):
    """Every base-game script file still loaded (common/, the kept events)
    that names a removed country / focus / decision / event, rewritten
    without those references under the same name (hoi4clean)."""
    import hoi4clean
    kept_focus, kept_decisions, kept_events = set(), set(), set()
    for f in (out / "common/national_focus").glob("*.txt"):
        for k, _o, v in pdx.walk(pdx.read(f)):
            if k in ("focus", "shared_focus") and isinstance(v, list) and pdx.get(v, "id"):
                kept_focus.add(pdx.get(v, "id"))
    for f in (out / "common/decisions").glob("*.txt"):
        for _k, _o, v in pdx.read(f):
            if isinstance(v, list):
                kept_decisions.update(dk for dk, _do, dv in v if dk and isinstance(dv, list))
    for f in (out / "events").glob("*.txt"):
        for k, _o, v in pdx.read(f):
            if k in EVENT_KEYS and isinstance(v, list) and pdx.get(v, "id"):
                kept_events.add(pdx.get(v, "id"))
    dead = hoi4clean.dead_tokens(van, kept_focus, kept_decisions, kept_events, set(our_tags))
    # AI areas that aren't kept (they listed the base game's strategic regions)
    def area_names(path):
        return {k for blk in (pdx.get(pdx.read(path), "areas") or []) for k in [blk[0]] if k} if path.exists() else set()
    dead |= area_names(van.game / "common/ai_areas/default.txt") - area_names(out / "common/ai_areas/default.txt")
    replaced = set(REPLACE_PATHS) | CLEAN_SKIP
    stats = {"files_changed": 0, "files_read": 0, "dead_tokens": len(dead), "country_files_emptied": 0}
    vtags = set(van.tags) - set(van.dynamic_tags)
    cnames = _country_file_names(van)
    sources = []
    # the game and every DLC folder (DLC music / icon databases keep the
    # same relative paths, so an override of the same name replaces them)
    bases = [van.game] + sorted(d for sub_ in ("dlc", "integrated_dlc") for d in (van.game / sub_).glob("*")
                                if d.is_dir())
    files = []
    for base in bases:
        for sub_ in ("common", "music", "gfx/interface/equipmentdesigner/graphic_db", "gfx/train_gfx_database",
                     "gfx/army_icons", "country_metadata"):
            if (base / sub_).is_dir():
                files += [(base, x) for x in (base / sub_).rglob("*.txt")]
    seen = set()
    for base, f in sorted(files, key=lambda e: str(e[1])):
        rel = f.relative_to(base).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        folder = rel.rsplit("/", 1)[0]
        if folder in replaced or any(rel.startswith(r + "/") and r == folder for r in replaced):
            continue
        if (out / rel).exists():
            continue  # written by the export (colors.txt, ...)
        sources.append((f, rel))
    # our own kept copies of the replaced folders, cleaned in place
    for folder in KEEP:
        for f in sorted((out / folder).glob("*.txt")):
            sources.append((f, f.relative_to(out).as_posix()))
    originals = []
    for f, rel in sources:
        stats["files_read"] += 1
        try:
            originals.append((rel, pdx.read(f), is_country_file(rel, vtags, cnames)
                              and rel.rsplit("/", 1)[0] not in COUNTRY_FILE_KEEP))
        except Exception:
            continue
    # passes until stable: definitions a pass removed (and defined nowhere
    # else) are dead names for the next (has_idea = totaler_krieg_economy,
    # SWI_some_effect = yes ...)
    # a country-named file can hold generic definitions (has_air_academy lives in
    # ETH_scripted_triggers.txt, and every generic air spirit uses it): such a
    # file keeps the definitions generic files name, nothing else
    generic_refs = set()
    for rel, entries, country in originals:
        if not country:
            for k, _o, v in pdx.walk(entries):
                for tok in (k, v):
                    if isinstance(tok, str):
                        generic_refs.add(tok.strip('"'))
                        generic_refs.update(re.split(r"[:.|]", tok.strip('"')))
    defined_before = {rel: _definitions(rel, entries) for rel, entries, _c in originals}  # unfiltered
    originals = [(rel, _keep_definitions(rel, entries, generic_refs) if country else entries, country)
                 for rel, entries, country in originals]
    stats["country_definitions_kept"] = sum(len(_definitions(rel, e)) for rel, e, c in originals if c)
    results = {}
    for _pass in range(4):
        before, after = set(), set()
        for rel, entries, country in originals:
            before |= defined_before[rel]  # what the base game defines, incl. what the filter dropped
            if country and not entries:
                results[rel] = None
                continue
            new, changed = hoi4clean.clean_file(entries, dead, keep_gated=rel.startswith("gfx/army_icons/"))
            if country:
                new, changed = (new or None), True
            results[rel] = new if changed else False
            if new:
                after |= _definitions(rel, new)
        gone = {n for n in before - after if n not in dead}
        stats.setdefault("dead_definitions_per_pass", []).append(len(gone))
        if not gone:
            break
        dead |= gone
    stats["dead_tokens"] = len(dead)
    for rel, entries, country in originals:
        r = results.get(rel)
        if rel.startswith("events/") and r:
            r = [e for e in r if not (e[0] in EVENT_KEYS and isinstance(e[2], list) and _event_broken(e[2]))]
        if r is None:
            _write(out / rel, "# EarthToHoI4: the base game's country-specific content, neutralised\n")
            stats["country_files_emptied"] += 1
        elif r is not False:
            _write(out / rel, pdx.dump(r))
            stats["files_changed"] += 1
    return stats


# ------------------------------------------------------------------ tags
def assign_tags(gadm_tags, van):
    """{GADM tag: (new tag, vanilla equivalent or None)}. New tags never equal a
    vanilla one; the GADM code itself when free."""
    import palettes
    iso_to_vanilla = defaultdict(list)
    for t, iso in palettes.HOI4_TAG_TO_ISO.items():
        if van is None or t in van.tags:
            iso_to_vanilla[iso].append(t)
    taken = set(van.tags) if van else set()
    taken |= RESERVED_TAGS
    if van:  # nor a base-game localisation key ("MAX": core_l_english.yml's "Max")
        for f in (van.game / "localisation" / "english").rglob("*.yml"):
            taken.update(re.findall(r"^\s*([A-Z][A-Z0-9]{2}):", f.read_text(encoding="utf-8-sig", errors="replace"),
                                    re.M))
    if van:  # nor one the base game has a flag for (releasables, cosmetic tags ...)
        for f in (van.game / "gfx/flags").glob("*.tga"):
            taken.add(f.stem.split("_")[0].upper())
    out = {}
    letters = "XZYQJKVWUHGFEDCBAIOLMNPRST"
    for g in sorted(gadm_tags):
        cands = iso_to_vanilla.get(g, [])
        src = None
        if cands:
            in36 = [t for t in cands if van and t in van.owners_1936]
            src = (in36 or cands)[0]
        base = re.sub(r"[^A-Z0-9]", "", g.upper())[:3].ljust(3, "X")
        options = [base] + [base[:2] + c for c in letters] + [base[0] + c + base[2] for c in letters] + \
            [base[0] + a + b for a in letters for b in letters]
        for t in options:
            if t not in taken and not re.fullmatch(r"D\d\d", t) and not t[0].isdigit():
                taken.add(t)
                out[g] = (t, src)
                break
        else:
            raise ExportError(f"No free country tag for {g}")
    return out


# ------------------------------------------------------------------ geometry
class Geo:
    """Province geometry on the province map: adjacency, interior points,
    coast contacts."""

    def __init__(self, P, ptype, hm):
        self.P = P
        self.h, self.w = P.shape
        self.n = int(P.max()) + 1
        self.ptype = ptype
        flat = P.ravel()
        self.area = np.bincount(flat, minlength=self.n)
        # the game finds land by the heightmap: a position on land below
        # LAND_LEVEL counts as "not over the land" (coast pixels are 94). A land
        # province with no pixel that high (islets) is raised to it.
        # The height is interpolated around a position: its whole 3x3 must be
        # that high. A province with no such pixel gets one: the 3x3 around its
        # most interior pixel raised (islets, 1 px spits).
        hm = hm.copy()
        landpx = (ptype[P] == LAND)
        b0 = np.zeros(P.shape, bool)
        b0[:, 1:] |= P[:, 1:] != P[:, :-1]
        b0[:, :-1] |= P[:, :-1] != P[:, 1:]
        b0[1:, :] |= P[1:, :] != P[:-1, :]
        b0[:-1, :] |= P[:-1, :] != P[1:, :]
        d0 = ndimage.distance_transform_edt(~b0).ravel()
        firm = (ndimage.minimum_filter(hm, 3) >= LAND_LEVEL) & landpx
        has = np.bincount(flat, weights=firm.ravel(), minlength=self.n) > 0
        low = ~has & (ptype[:self.n] == LAND) & (self.area > 0)
        self.raised = int(low.sum())
        if low.any():
            sel = np.flatnonzero(low[flat])
            o = sel[np.lexsort((-d0[sel], flat[sel]))]
            firsts = o[np.r_[True, flat[o][1:] != flat[o][:-1]]]
            for i in firsts.tolist():
                y0, x0 = divmod(i, self.w)
                win = hm[max(0, y0 - 1):y0 + 2, max(0, x0 - 1):x0 + 2]
                np.maximum(win, LAND_LEVEL, out=win)
        self.hm = hm
        # distance to the province's border -> interior points
        b = np.zeros(P.shape, bool)
        b[:, 1:] |= P[:, 1:] != P[:, :-1]
        b[:, :-1] |= P[:, :-1] != P[:, 1:]
        b[1:, :] |= P[1:, :] != P[:-1, :]
        b[:-1, :] |= P[:-1, :] != P[1:, :]
        b[0, :] = b[-1, :] = True  # the map's frame is a border too (the edge bands'
        b[:, 0] = b[:, -1] = True  # strips put their centre on column 0: "Ship in port too far")
        dist = ndimage.distance_transform_edt(~b).ravel()
        valid = ((ptype[P] != LAND) | (ndimage.minimum_filter(hm, 3) >= LAND_LEVEL)).ravel()
        # score: on land (or water) first, then away from the province border
        # (2 px in: the game may round a position to the next pixel)
        self.score = (valid * 16 + np.minimum(dist, 15)).astype(np.int16)
        order = np.lexsort((-self.score, flat))
        self.order = order.astype(np.int64)
        self.starts = np.searchsorted(flat[order], np.arange(self.n + 1))
        self.centre = np.full(self.n, -1, dtype=np.int64)
        has = self.area > 0
        self.centre[has] = order[self.starts[:-1][has]]
        # contacts (both directions), with the border pixel on each side
        keys, xs, ys = [], [], []
        for sa, sb in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                       ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
            a, c = P[sa], P[sb]
            m = a != c
            yy, xx = np.nonzero(m)
            ya, xa = yy + (sa[0].start or 0), xx + (sa[1].start or 0)
            yb, xb = yy + (sb[0].start or 0), xx + (sb[1].start or 0)
            pa, pc = a[m].astype(np.int64), c[m].astype(np.int64)
            keys += [pa * self.n + pc, pc * self.n + pa]
            xs += [xa, xb]
            ys += [ya, yb]
        keys = np.concatenate(keys)
        xs = np.concatenate(xs).astype(np.float64)
        ys = np.concatenate(ys).astype(np.float64)
        uk, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
        self.contact_n = dict(zip(uk.tolist(), cnt.tolist()))
        sx = np.bincount(inv, xs)
        sy = np.bincount(inv, ys)
        self.contact_xy = {int(k): (sx[i] / cnt[i], sy[i] / cnt[i]) for i, k in enumerate(uk.tolist())}
        self.nbrs = defaultdict(list)
        for k, c in zip(uk.tolist(), cnt.tolist()):
            a, b2 = divmod(k, self.n)
            self.nbrs[a].append((c, b2))
        for a in self.nbrs:
            self.nbrs[a].sort(reverse=True)
            self.nbrs[a] = [b2 for _c, b2 in self.nbrs[a]]
        # contacts across the map's wrap (last column against the first).
        # Kept apart from `nbrs`, so railways / state connectivity / straits /
        # unit directions stay geometric -- but the game builds its own
        # adjacency from the bitmap *with* the wrap, so a land province the
        # wrap puts against the sea is coastal to it and needs a port building
        # or the game crashes on start (see `write_map_text`). Only reached
        # when land runs to the left / right edge, i.e. with the edge bands off.
        self.wrap = defaultdict(list)
        self.wrap_xy = {}
        ca, cb = P[:, -1], P[:, 0]
        mw = ca != cb
        if mw.any():
            yy = np.nonzero(mw)[0].astype(np.float64)
            pa, pb = ca[mw].astype(np.int64), cb[mw].astype(np.int64)
            wk = np.concatenate([pa * self.n + pb, pb * self.n + pa])
            wx = np.concatenate([np.full(len(pa), self.w - 1.0), np.zeros(len(pb))])
            wy = np.concatenate([yy, yy])
            uw, iw, cw = np.unique(wk, return_inverse=True, return_counts=True)
            sxw, syw = np.bincount(iw, wx), np.bincount(iw, wy)
            for i, k in enumerate(uw.tolist()):
                self.wrap_xy[int(k)] = (sxw[i] / cw[i], syw[i] / cw[i])
                a, b2 = divmod(int(k), self.n)
                self.wrap[a].append((int(cw[i]), b2))
            for a in self.wrap:
                self.wrap[a].sort(reverse=True)
                self.wrap[a] = [b2 for _c, b2 in self.wrap[a]]

    def xy(self, flat_idx):
        return int(flat_idx % self.w), int(flat_idx // self.w)

    def centre_xy(self, p):
        return self.xy(self.centre[p])

    def pool(self, p):
        """Province p's best position pixels: on land, >= 2 px inside if any."""
        s, e = self.starts[p], self.starts[p + 1]
        idx = self.order[s:e]
        sc = self.score[idx]
        need = min(int(sc[0]), GOOD_SCORE)
        return idx[:max(1, int(np.searchsorted(-sc, -need, side="right")))]

    def interior(self, p, k, rng):
        """k position pixels of province p (repeats if few)."""
        pool = self.pool(p)
        pick = rng.choice(pool, size=k, replace=len(pool) < k)
        return [self.xy(i) for i in pick]

    def near(self, p, x, y):
        """The position pixel of province p nearest to (x, y)."""
        pool = self.pool(p)
        px, py = pool % self.w, pool // self.w
        i = int(np.argmin((px - x) ** 2 + (py - y) ** 2))
        return int(px[i]), int(py[i])

    def ok(self, p, x, y):
        i = int(round(y)) * self.w + int(round(x))
        return 0 <= i < self.P.size and self.P.flat[i] == p and self.score[i] >= GOOD_SCORE

    def world(self, x, y, water=False):
        """(x, height, z) in HOI4 map coordinates: z = map height - 1 - row
        (the game's own reading: every "not within specified state" error
        pointed one row off with map height - row)."""
        h = 9.5 if water else max(9.5, float(self.hm[min(y, self.h - 1), min(x, self.w - 1)]) / 10.0)
        return float(x), h, float(self.h - 1 - y)


def save_bmp8(img, path):
    """8-bit BMP with the game's header values: a full 256-entry palette,
    colours used / important 0, palette reserved bytes 0 (Pillow writes 256 /
    256, copies the grey level into the reserved byte -- "Palette in
    rivers.bmp is probably not correct" -- and writes only as many entries
    as the palette has: vanilla's cities.bmp has 255, which shifted the
    pixel data). Verified afterwards (read_bmp)."""
    if img.mode == "L":
        pal = [v for i in range(256) for v in (i, i, i)]
        arr = np.asarray(img)
    elif img.mode == "P":
        pal = list(img.getpalette() or [])[:768]
        pal += [0] * (768 - len(pal))
        arr = np.asarray(img)
    else:
        raise ExportError(f"{Path(path).name}: an 8-bit image is needed, not {img.mode}")
    out = Image.fromarray(arr, "P")
    out.putpalette(pal)
    out.save(path, format="BMP")
    with open(path, "r+b") as f:
        head = bytearray(f.read(1078))
        off = struct.unpack("<I", head[10:14])[0]
        if off != 1078 or struct.unpack("<I", head[14:18])[0] != 40:
            raise ExportError(f"{Path(path).name}: unexpected BMP layout (pixel data at {off})")
        head[46:54] = bytes(8)
        for i in range(256):
            head[54 + 4 * i + 3] = 0
        f.seek(0)
        f.write(head)
    verify_bmp(path, arr, pal)


def read_bmp(path):
    """Independent BMP reader (no Pillow): (pixels array, palette list or
    None, header dict). Checks the header's own consistency."""
    data = Path(path).read_bytes()
    if data[:2] != b"BM":
        raise ExportError(f"{Path(path).name}: not a BMP")
    size, off = struct.unpack("<I", data[2:6])[0], struct.unpack("<I", data[10:14])[0]
    hs, w, h, planes, bpp, comp, isz, _xr, _yr, clr, imp = struct.unpack("<IiiHHIIiiII", data[14:54])
    hdr = {"file_size": size, "offset": off, "header": hs, "width": w, "height": h, "bpp": bpp,
           "compression": comp, "colours_used": clr}
    if size != len(data):
        raise ExportError(f"{Path(path).name}: header says {size} bytes, file has {len(data)}")
    if comp != 0 or planes != 1 or bpp not in (8, 24):
        raise ExportError(f"{Path(path).name}: unsupported BMP (bpp {bpp}, compression {comp})")
    pal = None
    if bpp == 8:
        n = clr or 256
        if off != 14 + hs + 4 * n:
            raise ExportError(f"{Path(path).name}: palette of {n} entries but pixel data at {off}")
        raw = np.frombuffer(data, np.uint8, 4 * n, 14 + hs).reshape(n, 4)
        pal = raw[:, [2, 1, 0]].ravel().tolist()
    row = ((w * bpp // 8) + 3) // 4 * 4
    if off + row * abs(h) > len(data):  # trailing bytes are fine (vanilla's heightmap.bmp has 2)
        raise ExportError(f"{Path(path).name}: pixel data size doesn't match {w}x{h}")
    px = np.frombuffer(data, np.uint8, row * abs(h), off).reshape(abs(h), row)[:, :w * bpp // 8]
    if h > 0:
        px = px[::-1]
    if bpp == 24:
        px = px.reshape(abs(h), w, 3)[..., ::-1]
    return px, pal, hdr


def verify_bmp(path, arr, pal=None):
    px, ppal, _hdr = read_bmp(path)
    if px.shape != arr.shape or not np.array_equal(px, arr):
        raise ExportError(f"{Path(path).name}: written pixels differ from the image")
    if pal is not None and ppal[:len(pal)] != list(pal):
        raise ExportError(f"{Path(path).name}: written palette differs")


def _fmt(v):
    return f"{v:.2f}"


# ------------------------------------------------------------------ main
# ---- map size: the unpatched game takes ~13.1M px (a 50 MiB terrain buffer
# at 4 bytes / px), texture files <= 16 MiB (heightmap.bmp) and <= 256 map
# tiles of 256 px. The map size unlocker (MapSizeUnlocker/hoi4_map_size_patcher.py
# in the game folder) patches those in hoi4.exe; its patterns are read here
# (read only) to tell whether a big map (Countries tab "Double size") loads.
EXE_LIMITS = [  # (name, pattern, groups, vanilla value, struct format)
    ("buffer", rb"\xC7[\x80-\x87]\x50\x03\x00\x00(?P<a>....)\xB9(?P<b>....)\xE8", ("a", "b"), 0x3200000, "<I"),
    ("texture", rb"\xE8....\x3D(?P<a>....)\x0F\x86....\x80\x3D", ("a",), 0x1000000, "<I"),
    ("tile", rb"\x44\x0F\xB6\xE2\x44\x03\xE0\x41\x8B\x40\x44\x99\x41\xC1\xFC(?P<a>.)"
             rb"\x0F\xB6\xCA\x03\xC8\x41\x8B\x40\x40\x99\xC1\xF9(?P<b>.)", ("a", "b"), 8, "B"),
]
MAX_MAP_TILES = 256


def exe_map_limits(game):
    """{"buffer": bytes, "texture": bytes, "tile": shift} read from the game's
    hoi4.exe (vanilla values unless patched), or None if it can't be read."""
    try:
        data = (Path(game) / "hoi4.exe").read_bytes()
    except OSError:
        return None
    out = {}
    for name, pat, groups, vanilla, fmt in EXE_LIMITS:
        vals = []
        for m in re.finditer(pat, data, re.DOTALL):
            v = {struct.unpack(fmt, m.group(g))[0] for g in groups}
            if len(v) == 1 and min(v) >= vanilla:
                vals.append(v.pop())
        if len(vals) != 1:
            return None  # another game version: patterns not found
        out[name] = vals[0]
    return out


def map_size_problems(game, W, H):
    """What keeps a W x H map from loading in this game's hoi4.exe ([] = fine)."""
    need = {"buffer": W * H * 4, "texture": W * H + 2048}  # heightmap.bmp: 8 bit + palette
    if need["buffer"] <= EXE_LIMITS[0][3] and need["texture"] <= EXE_LIMITS[1][3]:
        return []
    fix = (f"python hoi4_map_size_patcher.py --width {W} --height {H} in the game's MapSizeUnlocker folder "
           "(again after every game update / file check)")
    lim = exe_map_limits(game) if game else None
    if lim is None:
        return [f"the map is {W} x {H} ({W * H:,} px), over the unpatched game's ~13.1M px, and hoi4.exe "
                f"couldn't be checked: make sure it is patched -- {fix}"]
    probs = []
    if lim["buffer"] < need["buffer"]:
        probs.append(f"terrain buffer {lim['buffer'] // 2**20} MiB < {-(-need['buffer'] // 2**20)} MiB needed")
    if lim["texture"] < need["texture"]:
        probs.append(f"MAX_TEXTURE_SIZE {lim['texture'] // 2**20} MiB < {-(-need['texture'] // 2**20)} MiB needed")
    tiles = (W >> lim["tile"]) * (H >> lim["tile"])
    if tiles > MAX_MAP_TILES:
        probs.append(f"{1 << lim['tile']} px tiles: {tiles} map tiles > {MAX_MAP_TILES}")
    if probs:
        return [f"hoi4.exe isn't patched for a {W} x {H} map ({'; '.join(probs)}): the game will crash. Run {fix}"]
    return []


def build_mod(src, settings, out, game=None, progress=lambda msg: None):
    """Writes the mod into `out` (a folder; emptied first). src: dict from
    app._hoi4_sources. Returns a report."""
    settings = normalize_settings(settings)
    import copy
    src = dict(src, regions=copy.deepcopy(src["regions"]))  # edited below; the app's cache keeps its own
    out = Path(out)
    report = {"warnings": []}
    van = Vanilla(game) if game else None
    if van is None:
        raise ExportError("Hearts of Iron IV wasn't found: the export copies and neutralises its files")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    progress("Reading the provinces")
    import strategic
    pimg = src["provinces_img"]
    P, ptype_arr, pstate, states_named = strategic.parse_inputs(pimg, src["definition"], src["states_csv"])
    H, W = P.shape
    report["map_size"] = [W, H]
    report["warnings"] += map_size_problems(van.game, W, H)
    # the game's province border limit (the Provinces tab keeps maps under it;
    # a map made before that was added, or by hand, may not be)
    import provinces as _provinces
    report["province_borders"] = nb = _provinces.border_count(P)
    if nb > _provinces.GAME_MAX_BORDERS:
        report["warnings"].append(
            f"{nb:,} province borders, over the {_provinces.GAME_MAX_BORDERS:,} the game draws: it will stop "
            f"drawing province borders over large parts of the map -- make at most ~{_provinces.MAX_PROVINCES:,} "
            "provinces (Provinces tab, target number)")
    n = int(P.max()) + 1
    ptype = np.full(n, -1, dtype=np.int8)
    ptype[:len(ptype_arr)] = ptype_arr[:n]
    # land at the left / right edge: the game wraps the map there, so that land
    # is adjacent to whatever is on the other edge -- sea (it becomes coastal
    # and gets a port here, else the game crashes on start) or land (armies
    # walk across the seam). The Countries tab's HOI4 edge bands prevent it.
    edge = ptype[np.unique(np.concatenate([P[:, 0], P[:, -1]]))]
    if (edge == LAND).any():
        report["warnings"].append(
            f"{int((edge == LAND).sum())} land provinces touch the left / right edge of the map, which the "
            "game wraps: they are adjacent to the opposite edge (coastal over the wrap, or land armies can "
            'cross the seam). Tick "HOI4 edge bands" on the Countries tab to keep land off the edges.')
    hm = np.asarray(src["heightmap"].convert("L"))
    geo = Geo(P, ptype, hm)
    report["provinces_raised"] = geo.raised
    rng = np.random.default_rng(1936)

    # ---- states: land provinces, + their lakes (user: lakes are part of the
    # states as the state map has them, for gameplay); the sea never
    progress("States")
    meta = src["meta"]
    parent = meta.get("autonomous_territories", {}) if meta.get("merge_autonomous") else {}
    srep = {int(s["id"]): s for s in src["states"]}
    by_state = defaultdict(list)
    lakes_of = defaultdict(list)
    stateless = []
    for p in range(1, n):
        if ptype[p] not in (LAND, LAKE) or geo.area[p] == 0:
            continue
        s = int(pstate[p]) if p < len(pstate) else 0
        if ptype[p] == LAKE:
            if s > 0:
                lakes_of[s].append(p)
            continue
        if s > 0:
            by_state[s].append(p)
        else:
            stateless.append(p)
    region_of = {}
    for g in src["regions"]:
        for p in g["provinces"]:
            region_of[int(p)] = g["id"]
    report["islets_moved"] = _fix_fractioned(src["regions"], region_of, geo)
    states = []
    for sid in sorted(by_state):
        name, country = states_named.get(sid, ("", ""))
        info = srep.get(sid, {})
        country = info.get("tag") or country
        owner = parent.get(country, country) or None
        # a state split between strategic regions is a map error: split it (shouldn't happen)
        parts = defaultdict(list)
        for p in by_state[sid]:
            parts[region_of.get(p)].append(p)
        # each lake goes with the part (strategic region) it borders most, and
        # into that part's region (a state split between regions is a map error)
        lake_part = {}
        for lk in lakes_of.get(sid, []):
            votes = defaultdict(int)
            for q in geo.nbrs.get(lk, []):
                if q < len(pstate) and pstate[q] == sid and ptype[q] == LAND:
                    votes[region_of.get(q)] += geo.contact_n.get(lk * geo.n + q, 0)
            votes = {r: v for r, v in votes.items() if r in parts}
            lake_part[lk] = max(votes, key=votes.get) if votes else next(iter(parts))
        for i, (reg, provs) in enumerate(parts.items()):
            lakes = sorted(lk for lk, r in lake_part.items() if r == reg)
            for lk in lakes:
                if region_of.get(lk) != reg:
                    _move_province(src["regions"], region_of, lk, reg)
            states.append({"lakes": lakes, "src": sid, "name": info.get("name") or name or f"State {sid}", "gadm": country,
                           "gid": info.get("gid"), "level": info.get("level"),
                           "merged": [g for g in str(info.get("merged_from") or "").split("|") if g],
                           "owner": owner, "provinces": sorted(provs), "impassable": False,
                           "population": int(info.get("population") or 0) * (1 if i == 0 else 0)})
    # land with no state (the edge bands' strip, hidden countries): impassable,
    # unowned states, one per connected group and strategic region
    sl = set(stateless)
    seen = set()
    for p in stateless:
        if p in seen:
            continue
        comp, stack = [], [p]
        seen.add(p)
        while stack:
            q = stack.pop()
            comp.append(q)
            for r in geo.nbrs.get(q, []):
                if r in sl and r not in seen and region_of.get(r) == region_of.get(p):
                    seen.add(r)
                    stack.append(r)
        edge = all(geo.centre_xy(q)[0] < 0.05 * W for q in comp)
        states.append({"src": None, "name": "Map edge" if edge else "Wasteland", "gadm": None, "owner": None,
                       "provinces": sorted(comp), "impassable": True, "population": 0, "edge": edge})
    report["state_lakes_moved"], report["state_provinces_moved"] = connect_states(states, geo, region_of)
    for i, s in enumerate(states, 1):
        s["id"] = i
    state_of = {}
    for s in states:
        for p in s["provinces"]:
            state_of[p] = s
    # impassable land other than the map edge (hidden countries) goes to the
    # country it borders most, else the nearest (wiki: an unowned state can
    # crash mid-game); the map edge's strip stays unowned (user)
    owned_centres = [(geo.centre_xy(t["provinces"][0]), t["owner"]) for t in states if t["owner"]]
    for s in states:
        if s["owner"] or s.get("edge"):
            continue
        votes = defaultdict(int)
        for p in s["provinces"]:
            for q in geo.nbrs.get(p, []):
                t = state_of.get(q)
                if t is not None and t["owner"] and not t["impassable"]:
                    votes[t["owner"]] += geo.contact_n.get(p * geo.n + q, 0)
        if votes:
            s["owner"] = max(votes, key=votes.get)
        elif owned_centres:
            x0, y0 = geo.centre_xy(s["provinces"][0])
            s["owner"] = min(owned_centres, key=lambda c: (c[0][0] - x0) ** 2 + (c[0][1] - y0) ** 2)[1]
    report["state_lakes"] = sum(len(s.get("lakes", ())) for s in states)
    report["states"] = len(states)
    report["impassable_states"] = sum(s["impassable"] for s in states)

    # ---- countries
    progress("Countries")
    gadm_owners = sorted({s["owner"] for s in states if s["owner"] and not s["impassable"]})
    tagmap = assign_tags(gadm_owners, van)
    names = {c["tag"]: c["name"] for c in meta.get("visible_countries", [])}
    colours = {c["tag"]: c["color"] for c in meta.get("visible_countries", [])}
    countries = {}
    for g in gadm_owners:
        tag, vsrc = tagmap[g]
        countries[g] = {"tag": tag, "vanilla": vsrc, "gadm": g, "name": names.get(g, g),
                        "color": colours.get(g, "#808080"), "states": []}
    for s in states:
        if s.get("edge"):
            s["owner"] = s["tag"] = None
            continue
        if s["owner"] not in countries:  # impassable land of a country with no other state here
            s["owner"] = gadm_owners[0]
        s["tag"] = countries[s["owner"]]["tag"]
        if not s["impassable"]:
            countries[s["owner"]]["states"].append(s)
    name_states(states)
    name_regions(src["regions"], state_of, geo)

    # ---- victory points from the City tab's cities
    vps = {}  # province -> (value, name)
    rank = {}  # province -> (country capital, population): the state's top city
    capital_city = {}
    for c in src.get("cities") or []:
        x, y = int(c["x"]), int(c["y"])
        if not (0 <= x < W and 0 <= y < H):
            continue
        p = int(P[y, x])
        s = state_of.get(p)
        if s is None or s["impassable"]:
            continue
        val = VP_BY_CLASS.get(int(c.get("class") or 9), 1)
        if c.get("feature_code") == "PPLC":
            capital_city.setdefault(s["owner"], (int(c.get("population") or 0), s, p))
        nm = c.get("label") or c.get("endonym") or c.get("name")
        rank[p] = max(rank.get(p, (0, 0)), (int(c.get("feature_code") == "PPLC"), int(c.get("population") or 0)))
        if p not in vps or val > vps[p][0]:
            vps[p] = (val, nm)
    # one top city per state (user: Giza and Cairo both 50 in one state made
    # Giza the state's capital): the country capital, else the most populated
    # city, keeps its value; every other one goes below it (next lower value)
    steps = sorted(set(VP_BY_CLASS.values()) | {1}, reverse=True)
    demoted = 0
    for s in states:
        vp = sorted((q for q in s["provinces"] if q in vps), key=lambda q: rank.get(q, (0, 0)), reverse=True)
        if len(vp) < 2:
            continue
        top = vps[vp[0]][0]
        for q in vp[1:]:
            if vps[q][0] >= top:
                lower = [v for v in steps if v < top]
                vps[q] = ((lower[0] if lower else max(top - 1, 1)), vps[q][1])
                demoted += 1
    report["victory_points"] = len(vps)
    report["victory_points_demoted"] = demoted

    def main_province(s):
        vp = [p for p in s["provinces"] if p in vps]
        if vp:
            return max(vp, key=lambda p: (vps[p][0], rank.get(p, (0, 0))))
        return max(s["provinces"], key=lambda p: geo.area[p])
    for s in states:
        s["main"] = main_province(s)
    for g, c in countries.items():
        cap = capital_city.get(g)
        cs = cap[1] if cap else max(c["states"], key=lambda s: (s["population"], len(s["provinces"])))
        c["capital"] = cs
        c["capital_province"] = cap[2] if cap else cs["main"]

    progress("Map images")
    write_map_images(out, src, geo, states, state_of, van, report)
    progress("Map positions (buildings, units, weather)")
    write_map_text(out, src, geo, states, state_of, vps, van, report)
    progress("States and countries")
    write_states(out, states, vps, settings)
    write_countries(out, countries, van, settings, report)
    progress("Localisation")
    write_fonts(out, van, report)
    write_localisation(out, states, vps, src["regions"], countries, van)
    progress("Neutralising the base game's country content")
    write_neutralised(out, van, report, our_tags=[c["tag"] for c in countries.values()])
    _write(out / "tutorial/tutorial.txt", "tutorial = {}\n")  # vanilla's links its own states
    write_lighting(out, src["view"], H, report)
    write_daynight(out, van, src["view"], W, H, report)
    write_map_scale(out, van, src["view"], W, H, report)
    write_view_defines(out, report)
    write_descriptor(out, settings, van, src)
    write_about(out, src, settings, countries, report)
    # last: the thumbnail renders the mod's own map files, so they all have to
    # be written (and descriptor.mod with them, for its replace_path)
    progress("Thumbnail")
    write_thumbnail(out, settings, van, src, report)
    report["countries"] = [{"gadm": g, "tag": c["tag"], "vanilla": c["vanilla"], "name": c["name"],
                            "states": len(c["states"]), "capital": c["capital"]["id"], "flag": c.get("flag")}
                           for g, c in countries.items()]
    report["provinces"] = int(n - 1)
    report["strategic_regions"] = len(src["regions"])
    problems = check_mod(out)
    report["problems"] = problems
    (out / "map_tool_export.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report


# ------------------------------------------------------------------ about
TOOL_URL = "https://github.com/EdouardSaladier/EarthToHoI4"
# (name, used for, link, licence) -- the datasets every export is made from
DATA_CREDITS = (
    ("GADM 3.6", "country and administrative borders, states, names", "https://gadm.org",
     "free for academic and other non-commercial use; redistribution of the data and commercial use not allowed"),
    ("NASA Blue Marble: topography, bathymetry, imagery", "heightmap, normal map, sea depth, colour map",
     "https://visibleearth.nasa.gov/collection/1484/blue-marble", "NASA imagery, free to use with credit"),
    ("GMTED2010 (USGS / NGA)", "land elevations in metres: heightmap, flooded coasts, river mouths",
     "https://topotools.cr.usgs.gov/gmted_viewer/", "public domain (USGS), credit requested"),
    ("HydroSHEDS: HydroRIVERS and HydroLAKES", "rivers, lakes", "https://www.hydrosheds.org", "CC BY 4.0"),
    ("Copernicus Global Land Cover (CGLS-LC100)", "terrain, forests", "https://land.copernicus.eu/global/products/lc",
     "free and open, with attribution (Copernicus Service information)"),
    ("GeoNames", "cities, victory points, city names", "https://www.geonames.org", "CC BY 4.0"),
    ("Marine Regions: World EEZ v12", "maritime zones, sea strategic regions", "https://www.marineregions.org",
     "CC BY 4.0"),
    ("JRC Global Human Settlement Layer (GHS-POP, GHS-DUC)", "population, manpower, density",
     "https://ghsl.jrc.ec.europa.eu", "CC BY 4.0"),
    ("ERA5 climate normals (World Bank Climate Change Knowledge Portal)", "weather of strategic regions",
     "https://climateknowledgeportal.worldbank.org", "CC BY 4.0"),
)


def write_about(out, src, settings, countries, report):
    """ABOUT_EarthToHoI4.txt: the mod was made by the tool, from which data
    (links and licences) -- user request."""
    import datetime
    import run_params
    view = src.get("view") or {}
    cv = run_params.code_version()
    report["tool_commit"] = cv.get("git_commit") or cv["code_hash"]
    lines = [
        f"{settings['name']}",
        "=" * max(len(settings["name"]), 10),
        "",
        "This Hearts of Iron IV mod was generated automatically by EarthToHoI4,",
        "a tool that builds HOI4 map mods from real-world geodata:",
        f"    {TOOL_URL}",
        "",
        f"Generated on {datetime.date.today().isoformat()}"
        + f" (tool version {report['tool_commit'][:10]}).",
        f"Map: {str(view.get('projection', '?')).capitalize()} projection, {view.get('width', '?')} x {view.get('height', '?')} px, "
        f"{len(countries)} countries.",
        "",
        "Data sources",
        "------------",
    ]
    credits = list(DATA_CREDITS)
    if any(c.get("flag") for c in countries.values()):
        credits.append(("Country flags from flagcdn.com", "country flags", "https://flagcdn.com", "free to use"))
    for name, use, url, lic in credits:
        lines += [f"- {name}", f"    used for: {use}", f"    {url}", f"    licence: {lic}"]
    if report.get("fonts"):
        lines += ["- Fonts from the mod Magna Europa: Reloaded (special characters in names)",
                  "    https://steamcommunity.com/sharedfiles/filedetails/?id=" + FONT_MOD_WORKSHOP_ID]
    lines += [
        "",
        "The data keep their own licences, which apply to this mod too -- notably",
        "GADM's: free for non-commercial use only.",
        "",
        "Hearts of Iron IV is a trademark of Paradox Interactive. This mod and the",
        "tool are not affiliated with or endorsed by Paradox Interactive.",
        "",
        "Reporting a problem: earthtohoi4_params.json in this folder holds every",
        "setting used to generate this mod. Attach it to an issue on the tool's",
        "page so the generation can be reproduced.",
        "",
    ]
    _write(out / "ABOUT_EarthToHoI4.txt", "\n".join(lines))


# ------------------------------------------------------------------ map
def write_map_images(out, src, geo, states, state_of, van, report):
    m = out / "map"
    m.mkdir(parents=True, exist_ok=True)
    pv = src["provinces_img"].convert("RGB")
    pv.save(m / "provinces.bmp")
    verify_bmp(m / "provinces.bmp", np.asarray(pv))
    save_bmp8(Image.fromarray(geo.hm, "L"), m / "heightmap.bmp")
    for key, name in (("terrain", "terrain.bmp"), ("rivers", "rivers.bmp"), ("trees", "trees.bmp")):
        img = src.get(key)
        if img is None:
            raise ExportError(f"{name} is missing (its tab can't render: check the Download tab)")
        save_bmp8(img, m / name)
    if src.get("normal") is not None:
        nm = src["normal"].convert("RGB")
        nm.save(m / "world_normal.bmp")
        verify_bmp(m / "world_normal.bmp", np.asarray(nm))
    import colormap
    t = m / "terrain"
    t.mkdir(parents=True, exist_ok=True)
    if src.get("colormap") is not None:
        (t / "colormap_rgb_cityemissivemask_a.dds").write_bytes(colormap.dds_bytes(src["colormap"]))
    else:
        report["warnings"].append("no colour map (Blue Marble missing): the game's own is used, stretched")
    write_water_colormaps(t, src["view"], geo, van, src.get("drowned_depth"))
    write_fow(t, geo, (geo.ptype[geo.P] != LAND).astype(np.float64))
    # cities.bmp: every pixel the "western" city group (index 15, like vanilla's
    # default), impassable land none (0)
    pal = None
    vc = van.game / "map/cities.bmp"
    if vc.exists():
        with Image.open(vc) as im:
            pal = im.getpalette()
    a = np.full(geo.P.shape, 15, dtype=np.uint8)
    imp = np.zeros(geo.n, bool)
    for s in states:
        if s["impassable"]:
            imp[s["provinces"]] = True
    a[imp[geo.P]] = 0
    ci = Image.fromarray(a, "P")
    ci.putpalette(pal or sum(([i, i, i] for i in range(256)), []))
    save_bmp8(ci, m / "cities.bmp")


def _vanilla_rowcol(lon, lat):
    col = (GEO_X[0] + GEO_X[1] * lon) % VANILLA_W
    row = np.clip(np.polyval(GEO_Y, np.clip(lat, -80, 85)), 0, VANILLA_H - 1)
    return col, row


def _dxt5_blocks(a):
    """DXT5 (BC3) blocks of an RGBA array (h, w multiples of 4): colour =
    bounding-box endpoints in RGB565 + 2-bit indices, alpha = min / max +
    3-bit indices (8-value mode)."""
    h, w = a.shape[:2]
    b = a.reshape(h // 4, 4, w // 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 4).astype(np.int32)
    # alpha
    amax, amin = b[..., 3].max(1), b[..., 3].min(1)
    steps = np.stack([amax, amin] + [((7 - i) * amax + i * amin) // 7 for i in range(1, 7)], 1)  # codes 0..7
    aidx = np.abs(b[..., 3][:, :, None] - steps[:, None, :]).argmin(2)
    aidx[amax == amin] = 0
    abits = np.zeros(len(b), dtype=np.uint64)
    for i in range(16):
        abits |= aidx[:, i].astype(np.uint64) << np.uint64(3 * i)
    # colour
    rgb = b[..., :3]
    cmax, cmin = rgb.max(1), rgb.min(1)

    def to565(c):
        return ((c[:, 0] >> 3) << 11) | ((c[:, 1] >> 2) << 5) | (c[:, 2] >> 3)

    def from565(v):
        r, g, bb = (v >> 11) & 31, (v >> 5) & 63, v & 31
        return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (bb << 3) | (bb >> 2)], 1)
    c0, c1 = to565(cmax), to565(cmin)
    swap = c0 < c1
    c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)
    e0, e1 = from565(c0), from565(c1)
    pal = np.stack([e0, e1, (2 * e0 + e1) // 3, (e0 + 2 * e1) // 3], 1)  # 4-colour mode (c0 > c1)
    cidx = ((rgb[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(3).argmin(2)
    cidx[c0 == c1] = 0
    cbits = np.zeros(len(b), dtype=np.uint32)
    for i in range(16):
        cbits |= cidx[:, i].astype(np.uint32) << np.uint32(2 * i)
    out = np.zeros((len(b), 16), dtype=np.uint8)
    out[:, 0] = amax
    out[:, 1] = amin
    for k in range(6):
        out[:, 2 + k] = ((abits >> np.uint64(8 * k)) & np.uint64(255)).astype(np.uint8)
    out[:, 8] = c0 & 255
    out[:, 9] = c0 >> 8
    out[:, 10] = c1 & 255
    out[:, 11] = c1 >> 8
    for k in range(4):
        out[:, 12 + k] = (cbits >> np.uint32(8 * k)) & 255
    return out.tobytes()


def dds_dxt5(img, mips=False):
    """DXT5 DDS like the game's colormap_water_*.dds (no mips) and
    fow_rgb_waterspec_a.dds (full mip chain): uncompressed, the latter was
    "larger than MAX_TEXTURE_SIZE" for the game."""
    img = img.convert("RGBA")
    w, h = img.size
    levels = [img]
    if mips:
        lw, lh = w, h
        while max(lw, lh) > 1:
            lw, lh = max(1, lw // 2), max(1, lh // 2)
            levels.append(levels[-1].resize((lw, lh), Image.BOX))
    body = []
    for lv in levels:
        a = np.asarray(lv)
        ph, pw = -(-a.shape[0] // 4) * 4, -(-a.shape[1] // 4) * 4
        if (ph, pw) != a.shape[:2]:
            a = np.pad(a, ((0, ph - a.shape[0]), (0, pw - a.shape[1]), (0, 0)), mode="edge")
        body.append(_dxt5_blocks(a))
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000 | (0x20000 if mips else 0)
    header = struct.pack("<4s7I44x", b"DDS ", 124, flags, h, w, len(body[0]), 0, len(levels) if mips else 0)
    pf = struct.pack("<2I4s5I", 32, 0x4, b"DXT5", 0, 0, 0, 0, 0)
    caps = struct.pack("<5I", 0x1000 | ((0x8 | 0x400000) if mips else 0), 0, 0, 0, 0)
    return header + pf + caps + b"".join(body)


def dds_mips(img):
    """Uncompressed A8R8G8B8 DDS with a full mip chain (as the game's
    fow_rgb_waterspec_a.dds has: 12 levels for 2816x1024)."""
    img = img.convert("RGBA")
    w, h = img.size
    levels = [img]
    while max(w, h) > 1:
        w, h = max(1, w // 2), max(1, h // 2)
        levels.append(levels[-1].resize((w, h), Image.BOX))
    w, h = img.size
    header = struct.pack("<4s7I44x", b"DDS ", 124, 0x1 | 0x2 | 0x4 | 0x8 | 0x1000 | 0x20000, h, w, w * 4, 0,
                         len(levels))
    pf = struct.pack("<2I4s5I", 32, 0x41, b"\0\0\0\0", 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    caps = struct.pack("<5I", 0x1000 | 0x8 | 0x400000, 0, 0, 0, 0)
    body = b"".join(np.asarray(l)[..., [2, 1, 0, 3]].tobytes() for l in levels)
    return header + pf + caps + body


def write_fow(t, geo, water):
    """fow_rgb_waterspec_a.dds (half the map, mips): the base game's is a
    grey picture of its own world (fog of war: land ~148 grey with texture,
    sea ~68; alpha = water specular, land ~21, sea 38 +- 12 mottled) --
    made the same way for this map."""
    rng = np.random.default_rng(7)
    h0, w0 = (geo.h + 1) // 2, (geo.w + 1) // 2
    wat = np.asarray(Image.fromarray((water * 255).astype(np.uint8)).resize((w0, h0), Image.BILINEAR)) / 255.0
    hm = np.asarray(Image.fromarray(geo.hm).resize((w0, h0), Image.BILINEAR)).astype(np.float64)

    def noise(sigma, amp):
        n = ndimage.gaussian_filter(rng.standard_normal((h0, w0)), sigma)
        return n / max(n.std(), 1e-9) * amp
    detail = hm - ndimage.gaussian_filter(hm, 6)
    land_g = 148 + np.clip(detail * 2.5, -25, 25) + noise(1.5, 8)
    sea_g = 68 + noise(25, 4) + noise(2, 1.5)
    g = np.clip(land_g * (1 - wat) + sea_g * wat, 0, 255)
    a = np.clip(21 * (1 - wat) + (38 + noise(3, 12)) * wat, 0, 255)
    rgba = np.dstack([g, g, g, a]).astype(np.uint8)
    (t / "fow_rgb_waterspec_a.dds").write_bytes(dds_dxt5(Image.fromarray(rgba, "RGBA"), mips=True))


# water colour model (user: "from the bathymetry, replicating the vanilla
# colour map's colours, latitude tints included"): vanilla's colormap_water_0
# colour on its sea pixels (georeferenced: GEO_X / GEO_Y) as a table over depth
# x latitude, fitted on water far from coasts, + an offset by distance to the
# coast (vanilla's lighter coastal rim). NASA's bathymetry steps 31.4 m.
WATER_DEP = np.array([0, 16, 47, 79, 141, 251, 502, 1004, 2008, 3012, 4016, 5020, 6024, 9000.0])
WATER_LAT = np.arange(-80, 90, 5.0)
WATER_COAST_KM = np.array([0, 15, 30, 60, 100, 160, 250, 400, 1e9])
WATER_CACHE = Path(__file__).resolve().parent.parent / "cache" / "hoi4_water_lut.npz"


def water_model(game):
    """(lut [depth bin, lat bin, rgb], coast offsets [coast bin, rgb]) fitted
    on the base game's colormap_water_0.dds; cached in cache/ (keyed on the
    game files' sizes and mtimes). None without bathymetry."""
    import bathymetry
    if not bathymetry.available():
        return None
    srcs = [game / "map/terrain/colormap_water_0.dds", game / "map/heightmap.bmp"]
    sig = np.array([v for f in srcs for v in (f.stat().st_size, int(f.stat().st_mtime))], np.int64)
    try:
        z = np.load(WATER_CACHE)
        if np.array_equal(z["sig"], sig):
            return z["lut"], z["coast"]
    except Exception:
        pass
    Wv = np.asarray(Image.open(srcs[0]).convert("RGB")).astype(np.float64)
    h, w = Wv.shape[:2]
    hm = np.asarray(Image.open(srcs[1]).convert("L"))
    sea = (hm[:h * 2, :w * 2] < 92).reshape(h, 2, w, 2).all(axis=(1, 3))
    rr, cc = np.nonzero(sea)
    lon = ((cc * 2 + 1.0 - GEO_X[0]) / GEO_X[1] + 180) % 360 - 180
    lt = np.linspace(-80, 85, 20001)
    rt = np.polyval(GEO_Y, lt)
    o = np.argsort(rt)
    row = rr * 2 + 1.0
    ok = (row >= rt.min()) & (row <= rt.max())
    lat = np.interp(row, rt[o], lt[o])
    dep = np.asarray(bathymetry.depth_m(lat, lon), float)
    # coast distance in km (vanilla: ~2 x 360 / 5632 degrees per half-res px)
    kmpx = 2 * 360.0 / VANILLA_W * 111.2 * np.cos(np.radians(lat))
    ckm = ndimage.distance_transform_edt(sea)[rr, cc] * kmpx
    rgb = Wv[rr, cc]
    di = np.clip(np.searchsorted(WATER_DEP, dep, side="right") - 1, 0, len(WATER_DEP) - 2)
    li = np.clip(np.searchsorted(WATER_LAT, lat, side="right") - 1, 0, len(WATER_LAT) - 2)
    # quantile matching per latitude band (+-5 degrees): the georeference is
    # ~35 km off, so colour medians per depth regress to the mean (shelves too
    # dark, deep water too light); deeper <-> darker by rank isn't blurred
    nd, nl = len(WATER_DEP) - 1, len(WATER_LAT) - 1
    lum = rgb @ np.array([0.3, 0.5, 0.2])
    dc = 0.5 * (WATER_DEP[:-1] + WATER_DEP[1:])
    lut = np.full((nd, nl, 3), np.nan)
    for b in range(nl):
        lo_, hi_ = WATER_LAT[b] - 5, WATER_LAT[b + 1] + 5
        m = ok & (lat >= lo_) & (lat < hi_)
        if m.sum() < 2000:
            continue
        d_sorted = np.sort(dep[m])
        o = np.argsort(-lum[m], kind="stable")  # lightest first = shallowest
        cols = rgb[m][o]
        nn = len(cols)
        for a in range(nd):
            if not ((dep[m] >= WATER_DEP[a]) & (dep[m] < WATER_DEP[a + 1])).any():
                continue
            q = np.searchsorted(d_sorted, dc[a]) / nn  # share of the band's water shallower
            k0, k1 = int(max(0, (q - 0.02) * nn)), int(min(nn, (q + 0.02) * nn + 1))
            lut[a, b] = cols[k0:k1].mean(0)
    have = ~np.isnan(lut[..., 0])
    if have.any():
        _d, (ia, ib) = ndimage.distance_transform_edt(~have, sampling=(3.0, 1.0), return_indices=True)
        lut = lut[ia, ib]
    pred = lut[di, li]
    res = rgb - pred
    ci = np.clip(np.searchsorted(WATER_COAST_KM, ckm, side="right") - 1, 0, len(WATER_COAST_KM) - 2)
    coast = np.zeros((len(WATER_COAST_KM) - 1, 3))
    for k in range(len(coast)):
        m = ok & (ci == k)
        if m.sum() > 100:
            coast[k] = np.median(res[m], axis=0)
    coast[-2:] = 0.0  # far from coasts: the table itself
    try:
        WATER_CACHE.parent.mkdir(exist_ok=True)
        np.savez(WATER_CACHE, lut=lut, coast=coast, sig=sig)
    except OSError:
        pass
    return lut, coast


def water_colours(lat, dep, coast_km, model):
    """RGB (float) for points: the table interpolated (depth on a log scale,
    latitude linearly between bin centres) + the coast offset."""
    from scipy.interpolate import RegularGridInterpolator
    lut, coast = model
    dc = np.log1p(0.5 * (WATER_DEP[:-1] + WATER_DEP[1:]) / 20.0)
    lc = 0.5 * (WATER_LAT[:-1] + WATER_LAT[1:])
    pts = np.column_stack([np.clip(np.log1p(dep / 20.0), dc[0], dc[-1]), np.clip(lat, lc[0], lc[-1])])
    out = np.stack([RegularGridInterpolator((dc, lc), lut[..., k])(pts) for k in range(3)], axis=-1)
    kc = 0.5 * (WATER_COAST_KM[:-2] + WATER_COAST_KM[1:-1])
    kc = np.r_[kc, 1e6]
    off = np.stack([np.interp(coast_km, kc, np.r_[coast[:-1, k], 0.0]) for k in range(3)], axis=-1)
    return np.clip(out + off, 0, 255)


def write_water_colormaps(t, view, geo, van, drowned_depth=None):
    """colormap_water_0/1/2.dds (half, quarter, eighth of the map): from the
    bathymetry with vanilla's colours (water_model); without the bathymetry,
    the base game's water colours sampled at the same place on Earth (its map
    georeferenced, GEO_X / GEO_Y). DXT5.

    `drowned_depth` (core.drowned_depth_m, metres per pixel of the render)
    is land a sea level rise turned into sea: GEBCO still reads it as land,
    0 m, which would paint a drowned plain in the palest shore colour, so
    the deeper of the two readings is used."""
    import core
    h0, w0 = (geo.h + 1) // 2, (geo.w + 1) // 2
    model = water_model(van.game)
    if model is not None:
        import bathymetry
        yy, xx = np.mgrid[0:h0, 0:w0]
        lon, lat = px_lonlat(view, (xx * 2 + 0.5).ravel(), (yy * 2 + 0.5).ravel())
        dep = np.asarray(bathymetry.depth_m(lat, lon, reduce=2), float).reshape(h0, w0)
        if drowned_depth is not None:
            dd = np.zeros((h0 * 2, w0 * 2), np.float32)
            dd[:geo.h, :geo.w] = drowned_depth[:geo.h, :geo.w]
            dep = np.maximum(dep, dd.reshape(h0, 2, w0, 2).max(axis=(1, 3)))
        water = np.ones((h0 * 2, w0 * 2), bool)
        water[:geo.h, :geo.w] = geo.ptype[geo.P] != LAND
        water = water.reshape(h0, 2, w0, 2).any(axis=(1, 3))
        dep = np.where(water, ndimage.gaussian_filter(dep, 1.0), 0.0)  # NASA's slope shading is noise
        lat2 = lat.reshape(h0, w0)
        ys = np.arange(h0) * 2 + 0.5  # km per half-res px, per row (Miller & co: scale varies with latitude)
        lo1, la1 = px_lonlat(view, np.full(h0, geo.w / 2), ys)
        lo2, la2 = px_lonlat(view, np.full(h0, geo.w / 2 + 2), ys)
        ckm = ndimage.distance_transform_edt(water) * _km(lo1, la1, lo2, la2)[:, None]
        rgb = water_colours(lat2.ravel(), dep.ravel(), ckm.ravel(), model).reshape(h0, w0, 3)
        alpha = np.full((h0, w0, 1), 255.0)
        img = Image.fromarray(np.concatenate([rgb, alpha], -1).round().astype(np.uint8), "RGBA")
        for i in range(3):
            w, h = max(1, w0 >> i), max(1, h0 >> i)
            (t / f"colormap_water_{i}.dds").write_bytes(dds_dxt5(img.resize((w, h), Image.BILINEAR)))
        return
    src = van.game / "map/terrain/colormap_water_0.dds"
    try:
        with Image.open(src) as im:
            vw = np.asarray(im.convert("RGBA"))
    except Exception:
        vw = None
    if vw is None:
        img = Image.new("RGBA", (w0, h0), (40, 70, 110, 255))
    else:
        yy, xx = np.mgrid[0:h0, 0:w0]
        px = (xx * 2 + 1.0)
        py = (yy * 2 + 1.0)
        gx = view["win_xmin"] + px / view["scale"]
        gy = view["win_ymin"] + (geo.h - py) / view["scale"]
        lon, lat, _ok = core._unproject(view["projection"], view["center"], gx.ravel(), gy.ravel())
        col, row = _vanilla_rowcol(lon, lat)
        sx = vw.shape[1] / VANILLA_W
        sy = vw.shape[0] / VANILLA_H
        ci = np.clip((col * sx).astype(int), 0, vw.shape[1] - 1)
        ri = np.clip((row * sy).astype(int), 0, vw.shape[0] - 1)
        img = Image.fromarray(vw[ri, ci].reshape(h0, w0, 4), "RGBA")
    for i in range(3):
        w, h = max(1, w0 >> i), max(1, h0 >> i)
        (t / f"colormap_water_{i}.dds").write_bytes(dds_dxt5(img.resize((w, h), Image.BILINEAR)))


def write_faction_theaters(out, van, region_ids, report):
    """The base game's AI faction theaters name its own strategic regions, which
    this map doesn't have ("western_europe: region list contains invalid entries
    (max id = 243!)" -- 18 of those in the user's France map). Each theater
    keeps only the ids this map has; one left without any goes. Written before
    the base-game clean-up, which leaves files the export wrote alone."""
    d = van.game / "common/ai_faction_theaters"
    if not d.is_dir():
        return
    kept = dropped = 0
    for f in sorted(d.glob("*.txt")):
        entries = []
        for k, o, v in pdx.read(f):
            if not isinstance(v, list) or pdx.get(v, "regions") is None:
                entries.append((k, o, v))
                continue
            new = []
            for k2, o2, v2 in v:
                if k2 == "regions" and isinstance(v2, list):
                    v2 = [e for e in v2 if str(e[2]).isdigit() and int(e[2]) in region_ids]
                new.append((k2, o2, v2))
            if pdx.get(new, "regions"):
                entries.append((k, o, new))
                kept += 1
            else:
                dropped += 1
        _write(out / "common/ai_faction_theaters" / f.name, pdx.dump(entries))
    report["faction_theaters"] = {"kept": kept, "dropped": dropped}


def write_map_text(out, src, geo, states, state_of, vps, van, report):
    import strategic
    m = out / "map"
    P, H, W = geo.P, geo.h, geo.w
    rng = np.random.default_rng(42)
    # definition.csv (CRLF, like the game wants)
    _write(m / "definition.csv", src["definition"], crlf=True)
    # strategic regions + weather positions
    for g in src["regions"]:
        _write(m / "strategicregions" / f"{g['id']}-{ascii_name(g['name'])}.txt", strategic.region_file(g))
    _write(m / "weatherpositions.txt", weather_positions(src["regions"], geo))
    write_faction_theaters(out, van, {int(g["id"]) for g in src["regions"]}, report)
    for name in ("default.map", "continent.txt", "seasons.txt", "cities.txt", "colors.txt"):
        sp = van.game / "map" / name
        if sp.exists():
            shutil.copyfile(sp, m / name)
    _write(m / "positions.txt", "")
    _write(m / "adjacency_rules.txt", "# no adjacency rules (the base game's name its own canals' provinces)\n")
    write_ambient(m, van, W, H, geo)

    # coast contacts: land province -> (best sea neighbour, contact point).
    # A land province whose only sea neighbour is across the map's wrap counts
    # too: the game reads coastality off the bitmap, wrap included, and a
    # province it calls coastal with no port building crashes the game on start
    # ("Province N is setup as coastal but has no port building in the
    # nudger"). definition.csv's coastal column is built the same way.
    coast = {}
    wrap_coast = []
    for p in range(1, geo.n):
        if geo.ptype[p] != LAND:
            continue
        seas = [q for q in geo.nbrs.get(p, []) if geo.ptype[q] == SEA]
        if seas:
            q = seas[0]  # most shared border
            coast[p] = (q, geo.contact_xy[p * geo.n + q], geo.contact_xy[q * geo.n + p])
            continue
        seas = [q for q in geo.wrap.get(p, []) if geo.ptype[q] == SEA]
        if seas:
            q = seas[0]
            coast[p] = (q, geo.wrap_xy[p * geo.n + q], geo.wrap_xy[q * geo.n + p])
            wrap_coast.append(p)
    report["coastal_over_wrap"] = len(wrap_coast)

    # ---- buildings.txt
    lines = []

    def add(sid, kind, x, y, rot=0.0, sea=0, water=False):
        X, Y, Z = geo.world(int(round(x)), int(round(y)), water)
        lines.append(f"{sid};{kind};{_fmt(X)};{_fmt(Y)};{_fmt(Z)};{_fmt(rot)};{sea}")
    state_level = (("arms_factory", 6), ("industrial_complex", 6), ("anti_air_building", 3), ("air_base", 1),
                   ("synthetic_refinery", 1), ("nuclear_reactor_spawn", 1), ("rocket_site_spawn", 1),
                   ("radar_station", 1), ("fuel_silo", 1), ("stronghold_network", 1))
    for s in states:
        sid = s["id"]
        main = s["main"]
        for kind, k in state_level:
            for x, y in geo.interior(main, k, rng):
                add(sid, kind, x, y, float(rng.uniform(0, 6.28)))
        for p in s["provinces"]:
            for kind in ("bunker", "supply_node", "special_project_facility_spawn"):
                x, y = geo.interior(p, 1, rng)[0]
                add(sid, kind, x, y, float(rng.uniform(0, 6.28)))
        coastal = [p for p in s["provinces"] if p in coast]
        for p in coastal:
            q, (cx, cy), (sx, sy) = coast[p]
            lx, ly = geo.near(p, cx, cy)
            hx, hy = geo.near(q, sx, sy)
            dx = sx - cx
            if abs(dx) > geo.w / 2:  # the sea is across the wrap: point that way
                dx -= math.copysign(geo.w, dx)
            rot = math.atan2(dx, -(sy - cy))
            add(sid, "naval_base_spawn", lx, ly, rot, q)
            add(sid, "coastal_bunker", lx, ly, rot)
            add(sid, "naval_supply_hub", lx, ly, rot)
            add(sid, "naval_headquarters", lx, ly, rot)
            add(sid, "floating_harbor", hx, hy, rot, q, water=True)
        if coastal:
            p = max(coastal, key=lambda p: geo.area[p])
            q, (cx, cy), _s = coast[p]
            lx, ly = geo.near(p, cx, cy)
            add(sid, "dockyard", lx, ly, 0.0)
    _write(m / "buildings.txt", "\n".join(lines))  # no empty last line (game: "invalid arguments count")

    # ports (stacks 19 / 20): the coast pixel nearest the province's centre --
    # the game wants ships in port within ~100 px of it ("Ship in port is too
    # far away from center (dist 1 176)" on large Asian provinces)
    seapx = geo.ptype[P] == SEA
    nsea = np.zeros_like(seapx)
    nsea[1:] |= seapx[:-1]
    nsea[:-1] |= seapx[1:]
    nsea[:, 1:] |= seapx[:, :-1]
    nsea[:, :-1] |= seapx[:, 1:]
    nsea[:, 0] |= seapx[:, -1]  # the map's wrap
    nsea[:, -1] |= seapx[:, 0]
    cy_, cx_ = np.nonzero(nsea & (geo.ptype[P] == LAND))
    cp = P[cy_, cx_]
    port_xy = {}
    if len(cp):
        cen = np.array([geo.centre_xy(int(q)) if geo.area[q] else (0, 0) for q in range(geo.n)], float)
        d2 = (cx_ - cen[cp, 0]) ** 2 + (cy_ - cen[cp, 1]) ** 2
        o = np.lexsort((d2, cp))
        first = o[np.r_[True, cp[o][1:] != cp[o][:-1]]]
        port_xy = {int(cp[i]): (int(cx_[i]), int(cy_[i])) for i in first}

    # ---- unitstacks.txt: id;type;x;y;z;rotation;offset
    us = []
    for p in range(1, geo.n):
        if geo.area[p] == 0:
            continue
        water = geo.ptype[p] != LAND
        cx, cy = geo.centre_xy(p)
        nb = geo.nbrs.get(p, [])[:8]

        def put(t, x, y, rot=0.0):
            X, Y, Z = geo.world(int(round(x)), int(round(y)), water)
            us.append(f"{p};{t};{_fmt(X)};{_fmt(Y)};{_fmt(Z)};{_fmt(rot)};{_fmt(float(rng.uniform(0.1, 1.3)))}")
        for t in (0, 9, 10, 21, 38):
            put(t, cx, cy)
        dirs = []
        for q in nb:
            bx, by = geo.contact_xy[p * geo.n + q]
            x, y = cx + 0.5 * (bx - cx), cy + 0.5 * (by - cy)
            if not geo.ok(p, x, y):
                x, y = cx, cy
            dirs.append((x, y, math.atan2(bx - cx, -(by - cy))))
        if not dirs:
            dirs = [(cx, cy, 0.0)]
        for i, (x, y, r) in enumerate(dirs):
            put(1 + i, x, y, r)
            put(22 + i, x, y, r)
            if water:
                put(11 + i, x, y, r)
                put(30 + i, x, y, r)
        if p in coast:
            lx, ly = port_xy.get(p) or coast[p][1]
            lx, ly = geo.near(p, lx, ly)
            put(19, lx, ly)
            put(20, lx, ly)

    _write(m / "unitstacks.txt", "\n".join(us) + "\n")

    # ---- railways + supply nodes: every owned state's main province is a
    # node; a minimum spanning tree over neighbouring states, each edge a
    # railway through the two states' provinces
    owned = [s for s in states if not s["impassable"]]
    sidx = {s["id"]: i for i, s in enumerate(owned)}
    edges = {}
    for s in owned:
        for p in s["provinces"]:
            for q in geo.nbrs.get(p, []):
                t = state_of.get(q)
                if t is not None and t["id"] in sidx and t["id"] != s["id"]:
                    a, b = sorted((s["id"], t["id"]))
                    if (a, b) not in edges:
                        (x1, y1), (x2, y2) = geo.centre_xy(state_of_id(states, a)["main"]), \
                            geo.centre_xy(state_of_id(states, b)["main"])
                        edges[(a, b)] = math.hypot(x2 - x1, y2 - y1)
    par = list(range(len(owned)))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    rails = []
    for (a, b), _d in sorted(edges.items(), key=lambda e: e[1]):
        ra, rb = find(sidx[a]), find(sidx[b])
        if ra == rb:
            continue
        sa, sb = state_of_id(states, a), state_of_id(states, b)
        path = _province_path(geo, set(sa["provinces"]) | set(sb["provinces"]), sa["main"], sb["main"])
        if path and len(path) >= 2:
            par[ra] = rb
            rails.append(path)
    _write(m / "railways.txt", "".join(f"1 {len(r)} {' '.join(map(str, r))}\n" for r in rails))
    nodes = sorted({s["main"] for s in owned})
    _write(m / "supply_nodes.txt", "".join(f"1 {p}\n" for p in nodes))
    report["railways"] = len(rails)
    report["supply_nodes"] = len(nodes)

    # ---- adjacencies.csv: sea crossings over narrow straits
    adj = _strait_crossings(geo, state_of, src.get("view"))
    rows = ["From;To;Type;Through;start_x;start_y;stop_x;stop_y;adjacency_rule_name;Comment"]
    for a, b, through in adj:
        rows.append(f"{a};{b};sea;{through};-1;-1;-1;-1;;strait")
    rows.append("-1;-1;-1;-1;-1;-1;-1;-1;-1")
    _write(m / "adjacencies.csv", "\n".join(rows) + "\n", crlf=True)
    report["straits"] = len(adj)


def _state_pieces(s, geo):
    """A state's provinces (land + its lakes) grouped by side contact, largest first."""
    allp = set(s["provinces"]) | set(s.get("lakes", ()))
    left, out = set(allp), []
    while left:
        a = left.pop()
        comp, st = {a}, [a]
        while st:
            q = st.pop()
            for r in geo.nbrs.get(q, []):
                if r in left:
                    left.discard(r)
                    comp.add(r)
                    st.append(r)
        out.append(comp)
    out.sort(key=lambda c: -sum(int(geo.area[p]) for p in c))
    return out


def connect_states(states, geo, region_of, rounds=4):
    """Every state one piece where the map allows it (user: state 1459 in two
    pieces; Varkaus split by a lake given to its neighbour): (1) a lake that
    touches two pieces of a state goes to it, if the lake's state stays as
    whole as it was; (2) a piece other than the largest that touches another
    state of the same owner over land, in the same strategic region, joins
    the one it borders most. Islands / pieces among water or foreign land stay.
    Returns (lakes moved, provinces moved)."""
    lake_state = {lk: s for s in states for lk in s.get("lakes", ())}
    land_state = {p: s for s in states for p in s["provinces"]}
    lakes_moved = provs_moved = 0
    for _round in range(rounds):
        changed = False
        for s in states:
            if not s.get("owner") or s["impassable"]:
                continue
            pieces = _state_pieces(s, geo)
            if len(pieces) < 2:
                continue
            piece_of = {p: i for i, c in enumerate(pieces) for p in c}
            for lk, t in list(lake_state.items()):
                if t is s or t.get("owner") != s["owner"]:
                    continue
                touch = {piece_of[q] for q in geo.nbrs.get(lk, []) if q in piece_of}
                if len(touch) < 2 or region_of.get(lk) != region_of.get(s["provinces"][0]):
                    continue
                before = len(_state_pieces(t, geo))
                t["lakes"].remove(lk)
                if len(_state_pieces(t, geo)) > before:
                    t["lakes"].append(lk)
                    continue
                s.setdefault("lakes", []).append(lk)
                lake_state[lk] = s
                lakes_moved += 1
                changed = True
                break
            if changed:
                continue
            for c in pieces[1:]:
                votes = defaultdict(int)
                for p in c:
                    for q in geo.nbrs.get(p, []):
                        t = land_state.get(q)
                        if (t is not None and t is not s and t.get("owner") == s["owner"] and not t["impassable"]
                                and region_of.get(q) == region_of.get(p)):
                            votes[id(t)] += geo.contact_n.get(p * geo.n + q, 0)
                if not votes or len({region_of.get(p) for p in c}) > 1:
                    continue
                t = next(x for x in states if id(x) == max(votes, key=votes.get))
                for p in c:
                    if p in land_state and land_state[p] is s:
                        s["provinces"].remove(p)
                        t["provinces"].append(p)
                        land_state[p] = t
                        provs_moved += 1
                    elif p in lake_state:
                        s["lakes"].remove(p)
                        t.setdefault("lakes", []).append(p)
                        lake_state[p] = t
                changed = True
        if not changed:
            break
    for s in states:
        s["provinces"].sort()
        if s.get("lakes"):
            s["lakes"].sort()
    return lakes_moved, provs_moved


def _move_province(regions, region_of, p, rid):
    """Moves province p into strategic region rid."""
    old = region_of.get(p)
    for g in regions:
        if g["id"] == old and p in g["provinces"]:
            g["provinces"].remove(p)
        if g["id"] == rid and p not in g["provinces"]:
            g["provinces"].append(p)
    region_of[p] = rid


def _fix_fractioned(regions, region_of, geo):
    """The game's "Naval strategic region X is fractioned": every province of
    a region with sea must touch the rest of it. A cut-off piece (islets of a
    coastal state) moves to the region it touches most, sea first; its
    provinces then make a state of their own there (user: islands may be
    split from their states) -- build_mod splits states by region. Returns
    the provinces moved."""
    by_id = {g["id"]: g for g in regions}
    moved = 0
    for _round in range(6):
        changed = False
        for g in regions:
            provs = [p for p in g["provinces"] if region_of.get(p) == g["id"]]
            if not any(geo.ptype[p] == SEA for p in provs):
                continue
            left = set(provs)
            pieces = []
            while left:
                a = left.pop()
                comp, stack = {a}, [a]
                while stack:
                    q = stack.pop()
                    for r in geo.nbrs.get(q, []):
                        if r in left:
                            left.discard(r)
                            comp.add(r)
                            stack.append(r)
                pieces.append(comp)
            pieces.sort(key=len, reverse=True)
            for piece in pieces[1:]:
                votes = defaultdict(int)
                for p in piece:
                    for q in geo.nbrs.get(p, []):
                        t = region_of.get(q)
                        if t is not None and t != g["id"] and q not in piece:
                            votes[t] += geo.contact_n.get(p * geo.n + q, 0) * (4 if geo.ptype[q] == SEA else 1)
                if not votes:
                    continue
                t = max(votes, key=votes.get)
                for p in piece:
                    region_of[p] = t
                moved += len(piece)
                changed = True
        for g in regions:  # province lists follow region_of
            g["provinces"] = [p for p in g["provinces"] if region_of.get(p) == g["id"]]
        extra = defaultdict(list)
        for p, t in region_of.items():
            if p not in set(by_id[t]["provinces"]):
                extra[t].append(p)
        for t, ps in extra.items():
            by_id[t]["provinces"] = sorted(set(by_id[t]["provinces"]) | set(ps))
        if not changed:
            break
    return moved


_state_by_id = {}


def state_of_id(states, sid):
    if _state_by_id.get("_list") is not states:
        _state_by_id.clear()
        _state_by_id["_list"] = states
        for s in states:
            _state_by_id[s["id"]] = s
    return _state_by_id[sid]


def _province_path(geo, allowed, a, b):
    """Shortest chain of neighbouring provinces from a to b inside `allowed`
    (centre-to-centre distance)."""
    import heapq
    dist = {a: 0.0}
    prev = {}
    heap = [(0.0, a)]
    while heap:
        d, p = heapq.heappop(heap)
        if p == b:
            break
        if d > dist.get(p, 1e18):
            continue
        px, py = geo.centre_xy(p)
        for q in geo.nbrs.get(p, []):
            if q not in allowed:
                continue
            qx, qy = geo.centre_xy(q)
            nd = d + math.hypot(qx - px, qy - py)
            if nd < dist.get(q, 1e18):
                dist[q] = nd
                prev[q] = p
                heapq.heappush(heap, (nd, q))
    if b not in dist:
        return None
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    return path[::-1]


STRAIT_MAX_PX = 6       # always a crossing at most this wide (any map scale) ...
STRAIT_MAX_KM = 25.0    # ... and up to this much water (Gibraltar 14 km, Great Belt 16, Bonifacio 11)
STRAIT_MAX_GAP_PX = 40
STRAIT_DETOUR_KM = 250.0  # one land mass: a crossing where going round by land is at least this long
STRAIT_DETOUR_RATIO = 10.0  # ... and this many times the crossing (Bosporus, Dardanelles, Kerch)
STRAIT_SPACING_PX = 25    # one crossing per strait of one land mass (Bosporus: not one per province pair)


def _strait_crossings(geo, state_of, view=None):
    """(land a, land b, sea through): land provinces (owned states, not
    impassable) facing each other across a strait -- at most STRAIT_MAX_KM
    (or STRAIT_MAX_PX) of sea in a row or column. Different land masses (an
    island and the mainland, two islands: up to 4 crossings per pair), or one
    land mass where the way round by land is a long detour (user: the
    Bosporus / Dardanelles, the Kerch strait, the Danish belts)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    P = geo.P
    H, W = P.shape
    landp = np.zeros(geo.n, bool)
    for p, s in state_of.items():
        if not s["impassable"]:
            landp[p] = True
    land = landp[P]
    sea = geo.ptype[P] == SEA
    max_gap = STRAIT_MAX_PX
    if view is not None:  # widest gap that can be STRAIT_MAX_KM anywhere on the frame
        ys = np.linspace(0, H - 1, 33)
        lon, lat = px_lonlat(view, np.full(33, W / 2), ys)
        lon2, lat2 = px_lonlat(view, np.full(33, W / 2 + 1), ys)
        kmpx = float(np.nanmin(_km(lon, lat, lon2, lat2)))
        max_gap = int(min(STRAIT_MAX_GAP_PX, max(max_gap, math.ceil(STRAIT_MAX_KM / max(kmpx, 1e-6)))))
    # land masses = connected land provinces (by province adjacency)
    comp = np.full(geo.n, -1, dtype=np.int64)
    c = 0
    for p in np.nonzero(landp)[0].tolist():
        if comp[p] >= 0:
            continue
        stack = [p]
        comp[p] = c
        while stack:
            q = stack.pop()
            for r in geo.nbrs.get(q, []):
                if landp[r] and comp[r] < 0:
                    comp[r] = c
                    stack.append(r)
        c += 1
    best = {}  # (a, b) -> (gap, through, xa, ya, xb, yb)
    for axis in (0, 1):
        A = P if axis == 1 else P.T
        L = land if axis == 1 else land.T
        S = sea if axis == 1 else sea.T
        rows, cols = A.shape
        mid = None
        for gap in range(1, max_gap + 1):
            # mid[:, j] = the gap pixels j+1 .. j+gap all sea
            mid = S[:, 1:cols - gap] if gap == 1 else (mid[:, :-1] & S[:, gap:cols - 1])
            m = L[:, :-gap - 1] & L[:, gap + 1:] & mid
            if not m.any():
                continue
            rr, jj = np.nonzero(m)
            pa, pb = A[rr, jj], A[rr, jj + gap + 1]
            through = A[rr, jj + 1 + gap // 2]
            keep = pa != pb
            for r_, j_, x, y, t in zip(rr[keep].tolist(), jj[keep].tolist(), pa[keep].tolist(),
                                       pb[keep].tolist(), through[keep].tolist()):
                k = (min(x, y), max(x, y))
                if k not in best or gap < best[k][0]:
                    ya_, xa_ = (r_, j_) if axis == 1 else (j_, r_)
                    yb_, xb_ = (r_, j_ + gap + 1) if axis == 1 else (j_ + gap + 1, r_)
                    best[k] = (gap, t, xa_, ya_, xb_, yb_)
    if not best:
        return []
    keys = list(best)
    vals = np.array([best[k][2:] for k in keys], float)
    gaps = np.array([best[k][0] for k in keys])
    if view is not None:
        lo1, la1 = px_lonlat(view, vals[:, 0], vals[:, 1])
        lo2, la2 = px_lonlat(view, vals[:, 2], vals[:, 3])
        km = _km(lo1, la1, lo2, la2) * (gaps / (gaps + 1.0))  # the water only
    else:
        km = gaps.astype(float)
    ok = (gaps <= STRAIT_MAX_PX) | (km <= STRAIT_MAX_KM)
    # one land mass: the way round by land (province centres) must be a long detour
    same = [i for i, (a, b) in enumerate(keys) if ok[i] and comp[a] == comp[b]]
    detour_ok = set()
    if same:
        lp = np.flatnonzero(landp)
        cx = np.array([geo.centre_xy(p) for p in lp], float)
        pos = {p: i for i, p in enumerate(lp.tolist())}
        r_, c_, w_ = [], [], []
        for p in lp.tolist():
            for q in geo.nbrs.get(p, []):
                if landp[q]:
                    r_.append(pos[p])
                    c_.append(pos[q])
                    w_.append(max(float(np.hypot(*(cx[pos[p]] - cx[pos[q]]))), 0.5))
        G = csr_matrix((w_, (r_, c_)), shape=(len(lp), len(lp)))
        kmpx_i = km / (gaps + 1.0)  # local km per px
        need_px = {i: max(STRAIT_DETOUR_KM, STRAIT_DETOUR_RATIO * km[i]) / max(kmpx_i[i], 1e-6) for i in same}
        srcs = sorted({pos[keys[i][0]] for i in same})
        lim = max(need_px.values()) * 1.01
        by_src = defaultdict(list)
        for i in same:
            by_src[pos[keys[i][0]]].append(i)
        for k0 in range(0, len(srcs), 400):
            chunk = srcs[k0:k0 + 400]
            D = dijkstra(G, indices=chunk, limit=lim)
            for row, sidx in enumerate(chunk):
                for i in by_src[sidx]:
                    if D[row, pos[keys[i][1]]] >= need_px[i]:
                        detour_ok.add(i)
    out = []
    per_pair = defaultdict(int)
    placed = []
    for i in sorted(range(len(keys)), key=lambda i: (km[i], gaps[i])):
        if not ok[i]:
            continue
        a, b = keys[i]
        gap, t, xa, ya, xb, yb = best[(a, b)]
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        if comp[a] == comp[b]:
            if i not in detour_ok or any((mx - px_) ** 2 + (my - py_) ** 2 < STRAIT_SPACING_PX ** 2
                                         for px_, py_ in placed):
                continue
        else:
            cp = (min(comp[a], comp[b]), max(comp[a], comp[b]))
            if per_pair[cp] >= 4:
                continue
        # the sea province crossed: one touching both shores if the line has one
        line = P[ya, min(xa, xb) + 1:max(xa, xb)] if ya == yb else P[min(ya, yb) + 1:max(ya, yb), xa]
        both = [q for q in dict.fromkeys(line.tolist()) if a in geo.nbrs.get(q, []) and b in geo.nbrs.get(q, [])]
        if not both:  # the game wants it next to one of them at least ("Adj between A and B is
            both = [q for q in geo.nbrs.get(a, []) if geo.ptype[q] == SEA and b in geo.nbrs.get(q, [])]
        if not both:  # not adjacent with THROUGH" only for one touching neither)
            both = [q for q in dict.fromkeys(line.tolist()) if a in geo.nbrs.get(q, []) or b in geo.nbrs.get(q, [])]
        if not both:
            continue
        t = both[0]
        if comp[a] != comp[b]:
            per_pair[(min(comp[a], comp[b]), max(comp[a], comp[b]))] += 1
        placed.append((mx, my))
        out.append((a, b, t))
    return out


def weather_positions(regions, geo):
    """map/weatherpositions.txt: two distinct positions per region like
    vanilla (it never puts two at one point): the region's centre and the
    centre of its province farthest from it; the first "big" for the larger
    half of the regions (the game wants both sizes)."""
    med = float(np.median([g["px"] for g in regions])) if regions else 0
    lines = []
    for g in regions:
        cx, cy = g["centre"]
        provs = [p for p in g["provinces"] if 0 < p < geo.n and geo.area[p] > 0]
        far = max(provs, key=lambda p: (geo.centre_xy(p)[0] - cx) ** 2 + (geo.centre_xy(p)[1] - cy) ** 2)
        fx, fy = geo.centre_xy(far)
        if abs(fx - cx) < 1 and abs(fy - cy) < 1:  # one-province region: a pixel beside the centre
            fx, fy = geo.near(far, cx + 3, cy + 3)
        for (x, y), size in (((cx, cy), "big" if g["px"] >= med else "small"), ((fx, fy), "small")):
            lines.append(f"{g['id']};{float(x):.2f};10.00;{geo.h - 1 - float(y):.2f};{size}")
    return "\n".join(lines) + "\n"


AMBIENT_KEEP = {"ambient_wind_entity", "ambient_water_entity", "frame_border_entity",
                "frame_border_bottom_entity", "frame_border_logo_entity"}  # never a landmark (the Great Wall)


def write_ambient(m, van, W, H, geo=None):
    """The base game's generic ambient objects (wind, the map frame, logo,
    water) moved to this map's size; landmark objects dropped (user: the
    Great Wall showed in the North Sea). Water objects go to this map's open
    sea (the points farthest from land), not vanilla's spots."""
    sp = van.game / "map/ambient_object.txt"
    if not sp.exists():
        return
    sx, sz = W / VANILLA_W, H / VANILLA_H
    spots = []
    if geo is not None:  # open-sea points, far from land and from each other
        water = geo.ptype[geo.P] == SEA
        d = ndimage.distance_transform_edt(water)
        step = 16
        dd = d[::step, ::step].copy()
        for _ in range(40):
            k = int(np.argmax(dd))
            if dd.flat[k] < 40:
                break
            y, x = divmod(k, dd.shape[1])
            spots.append((x * step, y * step))
            yy, xx = np.ogrid[:dd.shape[0], :dd.shape[1]]
            dd[(yy - y) ** 2 + (xx - x) ** 2 < (300 // step) ** 2] = 0

    def fix(entries, kind=None):
        res = []
        for k, o, v in entries:
            if k == "type" and isinstance(v, list):
                tname = str(pdx.get(v, "type") or "").strip('"')
                if tname not in AMBIENT_KEEP:
                    continue
                v = fix(v, tname)
                if tname == "ambient_water_entity" and geo is not None:
                    objs = [e for e in v if e[0] == "object"]
                    rest = [e for e in v if e[0] != "object"]
                    if objs:
                        v = rest + [_water_obj(objs[0], xy, H) for xy in spots]
            elif k == "position" and isinstance(v, list) and len(v) == 3:
                x, y, z = (float(e[2]) for e in v)
                if z > VANILLA_H:
                    z = z + (H - VANILLA_H)
                elif z > 0:
                    z = z * sz
                x = x * sx
                v = [(None, None, f"{x:.3f}"), (None, None, f"{y:.3f}"), (None, None, f"{z:.3f}")]
            elif isinstance(v, list):
                v = fix(v, kind)
            res.append((k, o, v))
        return res
    _write(m / "ambient_object.txt", pdx.dump(fix(pdx.read(sp))))


def _water_obj(template, xy, H):
    """A copy of an ambient water object at pixel xy (z = H - 1 - row)."""
    x, y = xy
    out = []
    for k, o, v in template[2]:
        if k == "position":
            v = [(None, None, f"{x:.3f}"), (None, None, "10.000"), (None, None, f"{H - 1 - y:.3f}")]
        out.append((k, o, v))
    return ("object", template[1], out)


# ------------------------------------------------------------------ states
def _category(pop):
    for lim, cat in CATEGORIES:
        if pop < lim:
            return cat
    return "megalopolis"


def write_states(out, states, vps, settings):
    d = out / "history/states"
    for s in states:
        pop = int(round(s["population"] * settings["manpower_pct"] / 100))
        s["manpower"] = pop
        hist = []
        if s["tag"]:
            hist.append(f"\t\towner = {s['tag']}")
        if not s["impassable"]:
            for p in s["provinces"]:
                if p in vps:
                    hist.append(f"\t\tvictory_points = {{ {p} {vps[p][0]} }}")
            hist.append(f"\t\tbuildings = {{\n\t\t\tinfrastructure = {settings['infrastructure']}\n\t\t}}")
            hist.append(f"\t\tadd_core_of = {s['tag']}")
        cat = "wasteland" if s["impassable"] else _category(pop)
        provs = " ".join(map(str, sorted(s["provinces"] + s.get("lakes", []))))
        text = ("state = {\n"
                f"\tid = {s['id']}\n"
                f"\tname = \"STATE_{s['id']}\"\n"
                + ("\timpassable = yes\n" if s["impassable"] else "")
                + f"\tmanpower = {max(pop, MIN_MANPOWER)}\n"
                f"\tstate_category = {cat}\n"
                "\thistory = {\n" + "\n".join(hist) + "\n\t}\n"
                f"\tprovinces = {{\n\t\t{provs}\n\t}}\n"
                "\tlocal_supplies = 0.0\n"
                "}\n")
        _write(d / f"{s['id']}-{ascii_name(s['name'], 40)}.txt", text)


# ------------------------------------------------------------------ countries
def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _vanilla_politics(van, vtag):
    """(set_politics block, set_popularities block) of a vanilla country, or None."""
    f = van.history_file(vtag) if vtag else None
    if not f:
        return None, None
    e = pdx.read(f)
    return pdx.get(e, "set_politics"), pdx.get(e, "set_popularities")


def write_countries(out, countries, van, settings, report):
    tags_lines = []
    colors = []
    vcol = van.game / "common/countries/colors.txt"
    base_colors = vcol.read_text(encoding="utf-8-sig", errors="replace") if vcol.exists() else ""
    import palettes
    iso2 = _iso3_to_iso2()
    for g, c in sorted(countries.items(), key=lambda e: e[1]["tag"]):
        tag, vtag = c["tag"], c["vanilla"]
        fname = f"{tag}_{ascii_name(c['name'], 40)}"
        c["file"] = fname
        tags_lines.append(f'{tag} = "countries/{fname}.txt"')
        r, gg, b = _hex_rgb(c["color"])
        gfx = None
        vf = van.country_file(vtag) if vtag else None
        if vf:
            e = pdx.read(vf)
            gc, g2 = pdx.get(e, "graphical_culture"), pdx.get(e, "graphical_culture_2d")
            if gc:
                gfx = (gc, g2 or gc.replace("_gfx", "_2d"))
        if gfx is None:
            cont = "eu" if g in palettes.EUROPE else "af" if g in palettes.AFRICA else \
                "me" if g in palettes.MIDDLE_EAST else "as"
            gfx = CONTINENT_GFX[cont]
        c["gfx"] = gfx
        _write(out / "common/countries" / f"{fname}.txt",
               f"graphical_culture = {gfx[0]}\ngraphical_culture_2d = {gfx[1]}\n\ncolor = {{ {r} {gg} {b} }}\n")
        colors.append(f"{tag} = {{\n\tcolor = rgb {{ {r} {gg} {b} }}\n\tcolor_ui = rgb {{ {r} {gg} {b} }}\n}}")
        # history: capital, one infantry division, basic tech, politics
        # every country non-aligned (user), no base-game politics
        ruling, politics, pops = "neutrality", None, None
        pol = pdx.dump([("set_politics", "=", politics)]) if politics else \
            ('set_politics = {\n\truling_party = neutrality\n\tlast_election = "1932.1.1"\n'
             '\telection_frequency = 48\n\telections_allowed = no\n}\n')
        if pops:  # rescaled to 100 (the base game's Iran sums to 99)
            vals = {k: float(v) for k, _o, v in pops if k in IDEOLOGIES and isinstance(v, str)
                    and re.fullmatch(r"-?\d+(\.\d+)?", v)}
            tot = sum(vals.values())
            if vals and tot > 0:
                scaled = {k: int(round(v * 100 / tot)) for k, v in vals.items()}
                big = max(scaled, key=scaled.get)
                scaled[big] += 100 - sum(scaled.values())
                pops = [(k, "=", str(v)) for k, v in scaled.items()]
            else:
                pops = None
        pop_block = pdx.dump([("set_popularities", "=", pops)]) if pops else \
            "set_popularities = {\n\tdemocratic = 0\n\tfascism = 0\n\tcommunism = 0\n\tneutrality = 100\n}\n"
        c["ruling"] = ruling
        tech = "\n".join(f"\t{t} = 1" for t in DEFAULT_TECH)
        _write(out / "history/countries" / f"{tag} - {ascii_name(c['name'], 40)}.txt",
               f"capital = {c['capital']['id']}\n"
               f"oob = \"{tag}_1936\"\n\n"
               "set_research_slots = 3\nset_stability = 0.6\nset_war_support = 0.2\n\n"
               f"set_technology = {{\n{tech}\n}}\n\n" + pol + pop_block)
        _write(out / "history/units" / f"{tag}_1936.txt",
               "division_template = {\n\tname = \"Infantry Division\"\n\tregiments = {\n"
               "\t\tinfantry = { x = 0 y = 0 }\n\t\tinfantry = { x = 0 y = 1 }\n"
               "\t\tinfantry = { x = 1 y = 0 }\n\t\tinfantry = { x = 1 y = 1 }\n\t}\n}\n"
               "units = {\n\tdivision = {\n\t\tname = \"1st Infantry Division\"\n"
               f"\t\tlocation = {c['capital_province']}\n"
               "\t\tdivision_template = \"Infantry Division\"\n"
               "\t\tstart_experience_factor = 0.3\n\t\tstart_equipment_factor = 1.0\n\t}\n}\n")
        write_flags(out, c, van, iso2)
    # the base game's countries aren't loaded at all (user): this file
    # replaces its 00_countries.txt; zz_dynamic_countries.txt (civil wars,
    # releases -- the wiki: too few dynamic countries crash) stays
    _write(out / "common/country_tags/00_countries.txt", "\n".join(tags_lines) + "\n")
    # colours: ours + the dynamic countries' (no entry for an undefined tag)
    dyn = set(van.dynamic_tags)
    base = [(k, o, v) for k, o, v in pdx.parse(base_colors) if k in dyn]
    _write(out / "common/countries/colors.txt", (pdx.dump(base) if base else "") + "\n# EarthToHoI4 countries\n"
           + "\n".join(colors) + "\n")
    write_bookmark(out, countries)
    write_names(out, countries, van)


# a base-game country whose name lists stand in for countries with no equivalent,
# by graphical culture
NAMES_BY_CULTURE = {"western_european_gfx": "FRA", "eastern_european_gfx": "POL", "middle_eastern_gfx": "TUR",
                    "african_gfx": "ETH", "asian_gfx": "CHI", "southamerican_gfx": "ARG", "commonwealth_gfx": "AST"}


def write_names(out, countries, van):
    """common/names: first names / surnames per tag (the game generates
    leaders and generals from them: "Failed to generate a name for a
    character"), copied from the base-game equivalent, else from a country of
    the same graphical culture."""
    src = {}
    for f in sorted((van.game / "common/names").glob("*.txt")):
        for k, _o, v in pdx.read(f):
            if k and isinstance(v, list):
                src.setdefault(k, v)
    entries = []
    for c in countries.values():
        block = src.get(c["vanilla"]) if c["vanilla"] else None
        if block is None:
            block = src.get(NAMES_BY_CULTURE.get(c.get("gfx", ("",))[0], "FRA")) or src.get("FRA")
        if block is not None:
            entries.append((c["tag"], "=", block))
    _write(out / "common/names/zz_map_tool_names.txt", pdx.dump(entries))


FLAGS_DIR = Path(__file__).resolve().parent.parent / "data" / "flags"
GADM_TO_ISO2_EXTRA = {"XKO": "XK"}  # GADM codes GeoNames writes otherwise


def _iso3_to_iso2():
    """ISO3 -> ISO2 from GeoNames' countryInfo.txt (City tab data)."""
    out = dict(GADM_TO_ISO2_EXTRA)
    ci = Path(__file__).resolve().parent.parent / "data" / "cities" / "countryInfo.txt"
    if ci.exists():
        for line in ci.read_text(encoding="utf-8", errors="replace").splitlines():
            f = line.split("\t")
            if len(f) > 2 and not line.startswith("#") and len(f[0]) == 2 and len(f[1]) == 3:
                out.setdefault(f[1], f[0])
    return out


def write_flags(out, c, van, iso2=None):
    """The country's real flag (user: "only use country real flags, suppress
    everything from Hoi4"): Download tab -> Country flags (flagcdn.com, PNG
    by ISO code) resized to the game's sizes (large 82x52, medium 41x26,
    small 10x7, 32-bit TGA). No ideology variants: the game falls back to
    TAG.tga. Without the file: a plain flag in the country's colour."""
    d = out / "gfx/flags"
    tag = c["tag"]
    code = (iso2 or {}).get(c["gadm"], "")
    src = FLAGS_DIR / f"{code.lower()}.png" if code else None
    img = None
    if src is not None and src.exists():
        try:
            with Image.open(src) as im:
                img = im.convert("RGBA")
        except Exception:
            img = None
    c["flag"] = "real" if img is not None else "generated"
    if img is None:
        r, g, b = _hex_rgb(c["color"])
        dark = tuple(int(v * 0.55) for v in (r, g, b))
        light = tuple(min(255, int(v + (255 - v) * 0.6)) for v in (r, g, b))
        a = np.zeros((52, 82, 4), dtype=np.uint8)
        a[:, :] = (*dark, 255)
        a[13:39, :] = (r, g, b, 255)
        a[21:31, :] = (*light, 255)
        img = Image.fromarray(a, "RGBA")
    for sub_, (w, h) in (("", (82, 52)), ("medium", (41, 26)), ("small", (10, 7))):
        (d / sub_).mkdir(parents=True, exist_ok=True)
        small = img.resize((w, h), Image.LANCZOS)
        for suffix in [""] + [f"_{i}" for i in IDEOLOGIES]:  # TAG_fascism.tga ... (a base-game
            small.save(d / sub_ / f"{tag}{suffix}.tga", compression=None)  # flag must never show)

def write_bookmark(out, countries):
    ranked = sorted(countries.values(), key=lambda c: -sum(s["population"] for s in c["states"]))
    majors = ranked[:7] if len(ranked) > 7 else ranked
    minors = ranked[7:14]
    blocks = []
    for c in majors:
        blocks.append(f'\t\t"{c["tag"]}" = {{\n\t\t\thistory = "MAP_TOOL_COUNTRY_DESC"\n'
                      f'\t\t\tideology = {c["ruling"]}\n\t\t\tideas = {{ }}\n\t\t\tfocuses = {{ }}\n\t\t}}')
    for c in minors:
        blocks.append(f'\t\t"{c["tag"]}" = {{\n\t\t\tminor = yes\n\t\t\thistory = "MAP_TOOL_COUNTRY_DESC"\n'
                      f'\t\t\tideology = {c["ruling"]}\n\t\t\tideas = {{ }}\n\t\t\tfocuses = {{ }}\n\t\t}}')
    text = ("bookmarks = {\n\tbookmark = {\n\t\tname = \"MAP_TOOL_BOOKMARK_NAME\"\n"
            "\t\tdesc = \"MAP_TOOL_BOOKMARK_DESC\"\n\t\tdate = 1936.1.1.12\n"
            "\t\tpicture = \"GFX_select_date_1936\"\n"
            f"\t\tdefault_country = \"{ranked[0]['tag']}\"\n\t\tdefault = yes\n"
            + "\n".join(blocks) +
            "\n\t\t\"---\" = {\n\t\t\thistory = \"MAP_TOOL_OTHER_DESC\"\n\t\t}\n"
            "\t\teffect = {\n\t\t\trandomize_weather = 22345 # <- Obligatory in every bookmark !\n\t\t}\n"
            "\t}\n}\n")
    _write(out / "common/bookmarks/map_tool.txt", text)


# ------------------------------------------------------------------ localisation
def write_localisation(out, states, vps, regions, countries, van):
    for lang in LANGUAGES:
        d = out / "localisation" / lang
        hdr = f"l_{lang}:\n"
        st = [f' STATE_{s["id"]}:0 "{loc_text(s["name"])}"' for s in states]
        vp = [f' VICTORY_POINTS_{p}:0 "{loc_text(nm)}"' for p, (_v, nm) in sorted(vps.items())]
        sr = [f' STRATEGICREGION_{g["id"]}:0 "{loc_text(g["name"])}"' for g in regions]
        _write(d / f"state_names_l_{lang}.yml", hdr + "\n".join(st) + "\n", bom=True)
        _write(d / f"victory_points_l_{lang}.yml", hdr + "\n".join(vp) + "\n", bom=True)
        _write(d / f"strategic_region_names_l_{lang}.yml", hdr + "\n".join(sr) + "\n", bom=True)
        cl = []
        for c in countries.values():  # the GADM0 name for every key (user: no base-game names)
            tag = c["tag"]
            for k in [""] + [f"_{i}" for i in IDEOLOGIES]:
                for suf in ("", "_DEF", "_ADJ"):
                    cl.append(f' {tag}{k}{suf}:0 "{loc_text(c["name"])}"')
        cl += [' MAP_TOOL_BOOKMARK_NAME:0 "Map Tool World"',
               ' MAP_TOOL_BOOKMARK_DESC:0 "A map made with EarthToHoI4. Country content comes later."',
               ' MAP_TOOL_COUNTRY_DESC:0 ""', ' MAP_TOOL_OTHER_DESC:0 ""']
        _write(d / f"map_tool_countries_l_{lang}.yml", hdr + "\n".join(cl) + "\n", bom=True)


# ------------------------------------------------------------------ fonts
# Magna Europa: Reloaded's fonts (by the user, its author): they hold the
# special characters the base game's fonts lack (state / city names)
FONT_MOD_NAME = "Magna Europa: Reloaded"
FONT_MOD_WORKSHOP_ID = "3150495839"


def find_font_mod(game=None):
    """Magna Europa: Reloaded's folder: a local mod (descriptor in the mod
    folder) or its Steam Workshop copy; None if neither has gfx/fonts."""
    cands = []
    for d in sorted(mod_dir().glob("*.mod")):
        try:
            t = d.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        name = re.search(r'^\s*name\s*=\s*"([^"]*)"', t, re.M)
        path = re.search(r'^\s*path\s*=\s*"([^"]*)"', t, re.M)
        if name and path and name[1].strip() == FONT_MOD_NAME:
            p = Path(path[1])
            cands.append(p if p.is_absolute() else user_dir() / p)
    if game is not None:  # steamapps/common/<game> -> steamapps/workshop/content/394360/<id>
        cands.append(Path(game).parent.parent / "workshop" / "content" / "394360" / FONT_MOD_WORKSHOP_ID)
    for c in cands:
        if (c / "gfx" / "fonts").is_dir() and any((c / "gfx" / "fonts").glob("*.fnt")):
            return c
    return None


def write_fonts(out, van, report):
    src = find_font_mod(van.game if van else None)
    if src is None:
        report["warnings"].append(f"{FONT_MOD_NAME} not found: the base game's fonts are used "
                                  "(some special characters won't show)")
        report["fonts"] = None
        _GLYPHS[0] = None
        return
    d = out / "gfx" / "fonts"
    d.mkdir(parents=True, exist_ok=True)
    files = [f for f in (src / "gfx" / "fonts").iterdir() if f.is_file()]
    glyphs = None
    for f in files:
        shutil.copyfile(f, d / f.name)
        if f.suffix.lower() == ".fnt":  # names are folded to what every font can show
            ids = {int(x) for x in re.findall(r"char id=(\d+)", f.read_text(encoding="utf-8", errors="replace"))}
            glyphs = ids if glyphs is None else glyphs & ids
    _GLYPHS[0] = glyphs
    report["fonts"] = {"from": str(src), "files": len(files)}


# -------------------------------------------------------------- view defines
# Camera, map-label and flag defines the user asked for (2026-10-15), written
# into every export **verbatim, their own comments included**: unlike the
# lighting / day-night / map-scale files these are fixed values, not computed
# from the frame, so the block below is the file. Edit it here to change them.
#
# Why they are wanted: vanilla's values are tuned for its own 5632x2048 map.
# These maps are far finer, so the camera stops too low to see a big frame at
# once and province names, victory points and capital icons fade out long
# before you are zoomed far enough back; and the export writes real national
# flags, which vanilla's texture caps would scale back down.
#
# **Lua comments are `--`**: a `#` in a defines file makes the game drop every
# define in it and crash.
VIEW_DEFINES = """\
NDefines.NFrontend.CAMERA_MIN_HEIGHT = 50.0
NDefines.NFrontend.CAMERA_MAX_HEIGHT = 3800.0
NDefines.NGraphics.PROVINCE_NAME_DRAW_DISTANCE = 1000.0             -- Remove province names beyond this distance
NDefines.NGraphics.COUNTRY_FLAG_LARGE_STRIPE_MAX_HEIGHT = 24000        -- VANILLA: 8192
NDefines.NGraphics.COUNTRY_FLAG_SMALL_TEX_MAX_SIZE = 512            -- VANILLA: 64
NDefines.NGraphics.COUNTRY_FLAG_STRIPE_TEX_MAX_HEIGHT = 8196        -- VANILLA: 2048
NDefines.NGraphics.COUNTRY_FLAG_TEX_MAX_SIZE = 2048                    -- VANILLA: 256
NDefines.NGraphics.VICTORY_POINT_MAP_ICON_TEXT_CUTOFF = {225, 375, 750}  -- At what camera distance the VP name text disappears.
NDefines.NGraphics.VICTORY_POINTS_DISTANCE_CUTOFF = {375,750,1500} -- At what distance VPs are hidden
NDefines.NGraphics.CAPITAL_ICON_CUTOFF = 1650    -- At what camera distance capital icons disappears
"""


def write_view_defines(out, report=None):
    """The camera / label / flag defines (VIEW_DEFINES), as their own file so
    the computed ones stay readable."""
    _write(out / "common/defines/zz_map_tool_view.lua",
           "-- Camera, map labels and flags (EarthToHoI4: hoi4mod.VIEW_DEFINES)\n" + VIEW_DEFINES)
    if report is not None:
        report["view_defines"] = sorted(re.findall(r"^NDefines\.\w+\.(\w+)", VIEW_DEFINES, re.M))


# ------------------------------------------------------------------ lighting
# The game lights the terrain from a virtual sun at map row SUN_LATITUDE
# (from the bottom), SUN_HEIGHT above the map: the light's angle falls with
# the distance from that row. The values are vanilla's map's (sun over
# ~13.6 N, 600 px high, ~12.6 px per degree there): on a map framed
# elsewhere the sun sat at the wrong place and the far end was lit at a
# grazing angle (user: the Sahara dark on an Africa map, 2,000+ px from it).
# Placed over the same latitude and scaled by this map's px per degree,
# every latitude gets vanilla's sun angle.
VANILLA_SUN = {"SUN_LATITUDE": 848.0, "SECOND_SUN_LATITUDE": 100.0, "SECOND_MOON_LATITUDE": 100.0,
               "SUN_HEIGHT": 600.0, "MOON_HEIGHT": 600.0, "SUN_HEIGHT_WATER": 5000.0, "MOON_HEIGHT_WATER": 550.0}


def _vanilla_row_lat(row):
    """Latitude at a vanilla map row (top-down), from the georeference fit."""
    lt = np.linspace(-80, 85, 33001)
    rt = np.polyval(GEO_Y, lt)
    o = np.argsort(rt)
    return float(np.interp(row, rt[o], lt[o]))


def map_lighting(view, H):
    """({define: value} placing the virtual sun / moon like vanilla's, info)."""
    import core
    func = core.PROJECTIONS[view["projection"]]["func"]

    def row_of(lat):  # pixel row (top-down) of a latitude on this map
        _x, y = func(np.array([0.0]), np.array([float(lat)]))
        return float(view["height"] - (y[0] - view["win_ymin"]) * view["scale"] - 0.5)
    sun_lat = _vanilla_row_lat(VANILLA_H - 1 - VANILLA_SUN["SUN_LATITUDE"])
    van_ppd = abs(np.polyval(GEO_Y, sun_lat + 0.5) - np.polyval(GEO_Y, sun_lat - 0.5))
    our_ppd = abs(row_of(sun_lat + 0.5) - row_of(sun_lat - 0.5))
    k = our_ppd / max(van_ppd, 1e-9)
    sun_z = H - 1 - row_of(sun_lat)
    vals = {"SUN_LATITUDE": sun_z}
    for key in ("SECOND_SUN_LATITUDE", "SECOND_MOON_LATITUDE"):  # same offset from the sun, scaled
        vals[key] = sun_z + (VANILLA_SUN[key] - VANILLA_SUN["SUN_LATITUDE"]) * k
    for key in ("SUN_HEIGHT", "MOON_HEIGHT", "SUN_HEIGHT_WATER", "MOON_HEIGHT_WATER"):
        vals[key] = VANILLA_SUN[key] * k
    return vals, {"sun_latitude_deg": round(sun_lat, 2), "scale": round(k, 3)}


def write_lighting(out, view, H, report):
    vals, info = map_lighting(view, H)
    lines = ["-- The map lighting placed for this map's frame (EarthToHoI4: hoi4mod.map_lighting)"]
    lines += [f"NDefines.NGraphics.{k} = {v:.1f}" for k, v in vals.items()]
    _write(out / "common/defines/zz_map_tool_lighting.lua", "\n".join(lines) + "\n")
    report["lighting"] = dict(info, **{k: round(v, 1) for k, v in vals.items()})


def frame_geo(view, W, H):
    """This map's geography for the shaders: latitude of the bottom / top rows
    and longitude of the left / right columns (centre column / row), and
    row_of(lat) (top-down)."""
    import core
    func = core.PROJECTIONS[view["projection"]]["func"]

    def row_of(lat):
        _x, y = func(np.array([0.0]), np.array([float(lat)]))
        return float(view["height"] - (y[0] - view["win_ymin"]) * view["scale"] - 0.5)
    lon, lat = px_lonlat(view, [W / 2, W / 2], [H - 1, 0])
    lon2, _lat2 = px_lonlat(view, [0, W - 1], [H / 2, H / 2])
    left, right = float(lon2[0]), float(lon2[1])
    while right <= left:
        right += 360.0
    return {"lat_bottom": float(lat[0]), "lat_top": float(lat[1]), "lon_left": left, "lon_right": right,
            "row_of": row_of}


# Day / night and seasons. The shaders turn map positions into a globe with
# vanilla's constants (standardfuncsgfx.fxh: Greenwich at x = 2793, the map
# = 360 degrees of longitude, rows = 59 S .. 77 N; constants.fxh: trees take
# season colours above half the map's height): on another frame the night
# band crossed the whole map in half a day and the latitudes were wrong. The
# export ships patched copies of the player's own two files.
def write_daynight(out, van, view, W, H, report):
    g = frame_geo(view, W, H)
    deg_px = (g["lon_right"] - g["lon_left"]) / max(W - 1, 1)  # degrees of longitude per pixel
    full_turn = 360.0 / max(deg_px, 1e-9)            # px for 360 degrees of longitude
    gmt = -g["lon_left"] / max(deg_px, 1e-9) + 0.5   # x of longitude 0 (may lie off the map)
    south = (g["lat_bottom"] + 90.0) / 180.0          # the shader: latitude = 180 * y - 90
    north = (g["lat_top"] + 90.0) / 180.0
    # trees' season colour: from vanilla's latitude at half its height, full at 0.9
    lat_a = _vanilla_row_lat(VANILLA_H / 2.0)
    lat_b = _vanilla_row_lat(VANILLA_H * 0.1)
    za = 1.0 - (g["row_of"](lat_a) + 0.5) / H
    zb = 1.0 - (g["row_of"](lat_b) + 0.5) / H
    tree_min, tree_tweak = za, 1.0 / max(zb - za, 1e-3)
    fx = van.game / "gfx" / "FX"
    done = []
    sf = fx / "standardfuncsgfx.fxh"
    if sf.exists():
        t = sf.read_text(encoding="utf-8-sig", errors="replace")
        rep = [(r"static const float GMT_OFFSET = [-0-9.]+f;",
                f"static const float GMT_OFFSET = {gmt:.1f}f; static const float FULL_TURN_X = {full_turn:.1f}f;"),
               (r"static const float SOUTH_POLE_OFFSET = [-0-9.]+f;", f"static const float SOUTH_POLE_OFFSET = {south:.4f}f;"),
               (r"static const float NORTH_POLE_OFFSET = [-0-9.]+f;", f"static const float NORTH_POLE_OFFSET = {north:.4f}f;"),
               (r"\( vWorldXZ\.x - GMT_OFFSET \) / MAP_SIZE_X", "( vWorldXZ.x - GMT_OFFSET ) / FULL_TURN_X")]
        ok = True
        for pat, new in rep:
            t, k = re.subn(pat, new, t, count=1)
            ok &= k == 1
        if ok:  # only a file every edit applied to (another game version: vanilla's stays)
            _write(out / "gfx/FX/standardfuncsgfx.fxh", t)
            done.append("standardfuncsgfx.fxh")
    cf = fx / "constants.fxh"
    if cf.exists():
        t = cf.read_text(encoding="utf-8-sig", errors="replace")
        t, k1 = re.subn(r"(TREE_SEASON_MIN\s*=\s*)[-0-9.]+f", lambda m: f"{m.group(1)}{tree_min:.4f}f", t, count=1)
        t, k2 = re.subn(r"(TREE_SEASON_FADE_TWEAK\s*=\s*)[-0-9.]+f", lambda m: f"{m.group(1)}{tree_tweak:.4f}f", t, count=1)
        if k1 and k2:
            _write(out / "gfx/FX/constants.fxh", t)
            done.append("constants.fxh")
    lines = ["-- Day / night placed for this map's frame (EarthToHoI4: hoi4mod.write_daynight)",
             f"NDefines.NGraphics.GMT_OFFSET = {gmt:.1f}",
             f"NDefines.NGraphics.SOUTH_POLE_OFFSET = {south:.4f}",
             f"NDefines.NGraphics.NORTH_POLE_OFFSET = {north:.4f}"]
    _write(out / "common/defines/zz_map_tool_daynight.lua", "\n".join(lines) + "\n")
    report["daynight"] = {"shaders": done, "gmt_offset": round(gmt, 1), "full_turn_px": round(full_turn, 1),
                          "south_pole_offset": round(south, 4), "north_pole_offset": round(north, 4),
                          "tree_season_min": round(tree_min, 4), "tree_season_fade": round(tree_tweak, 3)}


# Distances the base game measures in map pixels (00_defines.lua), for a
# 7.114 km / px map: scaled so they cover the same ground here; the map's
# km per pixel set too (aircraft ranges, movement ...). value x k, or / k
# where the pixels divide; area (pixels^2) x k^2.
VANILLA_KM_PER_PX = 7.114
PIXEL_DEFINES = {"MAP_SCALE_PIXEL_TO_KM": "km", "INFLUENCE_DISTANCE_DIVISOR": "x", "RADAR_RANGE_BASE": "x",
                 "RADAR_RANGE_MIN": "x", "RADAR_RANGE_MAX": "x", "AIR_REGION_SUPERIORITY_PIXEL_SCALE": "/area",
                 "INTERCEPTION_DISTANCE_SCALE": "x", "RAILWAY_GUN_POSSIBLE_RANGES": "x",
                 "INVASION_TARGET_DISTANCE_DENOMINATOR": "x", "DANGEROUS_DISTANCE_TO_CAPITAL": "x"}


def write_map_scale(out, van, view, W, H, report):
    import core
    kmpx = float(core._km_per_px({"view": view}))
    k = VANILLA_KM_PER_PX / max(kmpx, 1e-6)
    src = van.game / "common" / "defines" / "00_defines.lua"
    if not src.exists():
        return
    text = src.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    ns, found = None, {}
    for line in text:
        m = re.match(r"^\s*(N\w+)\s*=\s*\{", line)
        if m:
            ns = m.group(1)
            continue
        m = re.match(r"^\s*([A-Z0-9_]+)\s*=\s*(\{[^}]*\}|[-0-9.]+)", line)
        if m and m.group(1) in PIXEL_DEFINES and ns and m.group(1) not in found:
            found[m.group(1)] = (ns, m.group(2))
    lines = [f"-- Map scale: {kmpx:.3f} km per pixel ({k:.3f} x vanilla's pixels per km) "
             "(EarthToHoI4: hoi4mod.write_map_scale)"]
    vals = {}
    for key, (ns, raw) in found.items():
        kind = PIXEL_DEFINES[key]

        def conv(v):
            v = float(v)
            return kmpx if kind == "km" else v / (k * k) if kind == "/area" else v * k
        if raw.startswith("{"):
            nums = [conv(x) for x in re.findall(r"[-0-9.]+", raw)]
            val = "{ " + ", ".join(f"{x:.1f}" for x in nums) + " }"
        else:
            x = conv(raw)
            val = f"{x:.6g}"
        vals[key] = val
        lines.append(f"NDefines.{ns}.{key} = {val}")
    _write(out / "common/defines/zz_map_tool_scale.lua", "\n".join(lines) + "\n")
    report["map_scale"] = {"km_per_px": round(kmpx, 3), "factor": round(k, 3), "defines": vals,
                           "missing": sorted(set(PIXEL_DEFINES) - set(found))}


# ------------------------------------------------------------------ descriptor
def write_descriptor(out, settings, van, src):
    ver = game_version(van.game) or "1.19"
    major = ".".join(ver.split(".")[:2]) + ".*"
    lines = [f'name="{settings["name"]}"', 'version="1"', 'picture="thumbnail.png"',
             'tags={\n\t"Map"\n\t"Alternative History"\n}', f'supported_version="{major}"']
    lines += [f'replace_path="{p}"' for p in REPLACE_PATHS]
    _write(out / "descriptor.mod", "\n".join(lines) + "\n")


def write_thumbnail(out, settings, van, src, report):
    """thumbnail.png (the picture the launcher and the Workshop show): the mod
    we have just finished writing, rendered the way the game draws it
    (modthumbnail -> hoi4renderer), cut to a 512x512 square, with the mod's
    name across it in the map font (user). Needs the written map files, so it
    comes last. Never fails the export: without the render (or with it turned
    off) the Countries tab's map is used, and the name is written all the
    same."""
    out = Path(out)
    try:
        import modthumbnail
        img, rep = modthumbnail.render_thumbnail(
            str(out), str(van.game) if van else "", settings["name"], fallback=src.get("thumbnail"),
            render=bool(settings.get("thumbnail", 1)))
        img.save(out / "thumbnail.png")
    except Exception as e:
        report["thumbnail"] = {"error": f"{type(e).__name__}: {e}"}
        report["warnings"].append(f"No thumbnail: {type(e).__name__}: {e}")
        return
    report["thumbnail"] = rep
    for key in ("note", "title_error"):
        if rep.get(key) and settings.get("thumbnail", 1):
            report["warnings"].append(f"Thumbnail: {rep[key]}")


def install_descriptor(out, mod_dir):
    """<user dir>/mod/<folder>.mod pointing at the mod (the launcher's list)."""
    text = (out / "descriptor.mod").read_text(encoding="utf-8")
    text += f'path="{str(out).replace(chr(92), "/")}"\n'
    p = Path(mod_dir) / f"{out.name}.mod"
    _write(p, text)
    return p


# ------------------------------------------------------------------ checks
def check_mod(out):
    """Rules the game crashes / errors on, checked on the written files."""
    out = Path(out)
    problems = []
    defs = {}
    raw = (out / "map/definition.csv").read_bytes()
    if b"\n" in raw.replace(b"\r\n", b""):
        problems.append("definition.csv has LF line endings")
    for line in raw.decode("utf-8").splitlines():
        f = line.split(";")
        if len(f) >= 8 and f[0].isdigit():
            defs[int(f[0])] = f
    ids = sorted(defs)
    if ids != list(range(len(ids))):
        problems.append("definition.csv ids are not 0..N without gaps")
    land = {i for i, f in defs.items() if i and f[4] == "land"}
    sea = {i for i, f in defs.items() if f[4] == "sea"}
    for i in land:
        if not defs[i][7].isdigit() or int(defs[i][7]) < 1:
            problems.append(f"land province {i} has no continent")
            break
    in_state = {}
    state_ids = []
    for f in (out / "history/states").glob("*.txt"):
        e = pdx.read(f)
        st = pdx.get(e, "state")
        sid = int(pdx.get(st, "id"))
        state_ids.append(sid)
        for _k, _o, v in pdx.get(st, "provinces"):
            p = int(v)
            if p in in_state:
                problems.append(f"province {p} in states {in_state[p]} and {sid}")
            in_state[p] = sid
            if p in sea or p not in defs:  # land and lakes (user: lakes are part of the states)
                problems.append(f"state {sid} holds sea / unknown province {p}")
    if sorted(state_ids) != list(range(1, len(state_ids) + 1)):
        problems.append("state ids are not 1..N")
    missing = land - set(in_state)
    if missing:
        problems.append(f"{len(missing)} land provinces in no state (e.g. {sorted(missing)[:5]})")
    in_reg = {}
    for f in (out / "map/strategicregions").glob("*.txt"):
        e = pdx.get(pdx.read(f), "strategic_region")
        rid = int(pdx.get(e, "id"))
        for _k, _o, v in pdx.get(e, "provinces"):
            if int(v) in in_reg:
                problems.append(f"province {v} in two strategic regions")
            in_reg[int(v)] = rid
    nore = set(defs) - {0} - set(in_reg)
    if nore:
        problems.append(f"{len(nore)} provinces in no strategic region")
    # regions with sea: every province touching the rest ("... is fractioned")
    a = np.asarray(Image.open(out / "map/provinces.bmp").convert("RGB")).astype(np.int64)
    key = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
    col = {(int(f[1]) << 16) | (int(f[2]) << 8) | int(f[3]): i for i, f in defs.items() if i}
    u, inv = np.unique(key, return_inverse=True)
    P = np.array([col.get(int(k), 0) for k in u.tolist()])[inv].reshape(key.shape)
    # coastality as the game reads it off the bitmap -- **the map wraps**, so
    # the last column counts against the first. The game prefers this over
    # definition.csv ("Bitmap and province definition disagree ...") and
    # crashes on start on a coastal province with no port building in the
    # nudger. Land at the left / right edge (edge bands off) is the case.
    bit_coast = set()
    for A, B in ((P[:, :-1], P[:, 1:]), (P[:-1, :], P[1:, :]), (P[:, -1], P[:, 0])):
        m = A != B
        for x, y in set(zip(A[m].tolist(), B[m].tolist())):
            if (x in land and y in sea) or (x in sea and y in land):
                bit_coast |= {x, y}
    bad = sorted(i for i in defs if i and (i in bit_coast) != (defs[i][5] == "true"))
    if bad:
        problems.append(f"{len(bad)} provinces whose definition.csv coastal flag "
                        f"disagrees with the bitmap (wrap included): {bad[:10]}")
    ports = set()  # the province each naval base stands in (x = X, row = h-1-Z)
    for line in (out / "map/buildings.txt").read_text().splitlines():
        f = line.split(";")
        if len(f) >= 7 and f[1] == "naval_base_spawn":
            try:
                x, z = int(round(float(f[2]))), int(round(float(f[4])))
            except ValueError:
                continue
            y = P.shape[0] - 1 - z
            if 0 <= y < P.shape[0] and 0 <= x < P.shape[1]:
                ports.add(int(P[y, x]))
    noport = sorted((bit_coast & land) - ports)
    if noport:
        problems.append(f"{len(noport)} coastal land provinces with no port building "
                        f'(game: "setup as coastal but has no port building in the '
                        f'nudger. This will likely crash the game"): {noport[:10]}')
    nb = defaultdict(set)
    for A, B in ((P[:, :-1], P[:, 1:]), (P[:-1, :], P[1:, :])):
        m = A != B
        for x, y in set(zip(A[m].tolist(), B[m].tolist())):
            nb[x].add(y)
            nb[y].add(x)
    by_reg = defaultdict(list)
    for p, r in in_reg.items():
        by_reg[r].append(p)
    frac = []
    for r, ps in by_reg.items():
        if not any(p in sea for p in ps):
            continue
        left = set(ps)
        first = left.pop()
        stack = [first]
        while stack:
            q = stack.pop()
            for x in nb[q]:
                if x in left:
                    left.discard(x)
                    stack.append(x)
        if left:
            frac.append(r)
    if frac:
        problems.append(f"{len(frac)} strategic regions with sea in pieces (game: fractioned): {sorted(frac)[:10]}")
    for line in (out / "map/railways.txt").read_text().splitlines():
        f = list(map(int, line.split()))
        if len(f) != f[1] + 2 or any(p not in in_state for p in f[2:]):
            problems.append("railway with a bad province list: " + line[:60])
            break
    for line in (out / "map/supply_nodes.txt").read_text().splitlines():
        if int(line.split()[1]) not in in_state:
            problems.append("supply node outside a state: " + line)
            break
    wp = (out / "map/weatherpositions.txt").read_text()
    if ";big" not in wp or ";small" not in wp:
        problems.append("weatherpositions.txt lacks a big or small entry")
    if not (out / "map/buildings.txt").read_text().strip():
        problems.append("buildings.txt is empty")
    return problems
