"""City names for the City tab's labels: endonyms (the local name) or
exonyms (the name in a chosen language), always in Latin script, with or
without accents.

Sources (Download tab -> GeoNames Cities):
  * alternateNamesV2.zip -- every GeoNames name with its language and flags
    (preferred / short / colloquial / historic), ~16 M rows, 205 MB zipped,
    kept zipped. `build_names` streams it once (~30-60 s) and keeps only
    the City tab's places' names in `city_names.pkl` (a few MB), rebuilt
    when the zip or the city CSV changes.
  * countryInfo.txt -- each country's languages, most used first (which
    names are local).

Choice (`label_for`):
  * endonym: the name in the country's first language (countryInfo's
    order; English only when it's first -- Egypt lists it second). In the
    bilingual countries of MULTILINGUAL (Belgium, Switzerland, Canada,
    Finland, Luxembourg, Spain) the local form is the one most used across
    *all* languages' names of that place among the official ones (Geneve
    is also Dutch, Swedish, Danish ... while Genf is German only;
    Antwerpen over Anvers, Liege over Luik); ties -> the list order; same
    letters -> the exact spelling most languages use (Brussel, not the
    German Brüssel), then with diacritics (Montréal). That rule everywhere made minority
    languages win (Hungarian Bukarest, Belgrad, Kijev; French Le Caire).
    No entry in the first language: GeoNames' main name if it's Latin (it
    is the local name for such places: Bolzano, not the German Bozen).
    Within a language: the preferred entry, then short, then the first
    listed.
  * exonym: the name in `lang` (English by default); a place with none (most
    small towns) keeps its endonym -- that's what it's called in English
    too.
  * Non-Latin scripts (Москва, Αθήνα, القاهرة): among the place's
    Latin-script names that are untagged (GeoNames keeps romanizations
    untagged), in a local language or English -- other languages gave
    Icelandic "Aþena", Azerbaijani "Hələb" -- the one closest to a
    transliteration (unidecode; Greek the modern way: η -> i, β -> v) of
    the preferred local name: fewest edits, then the exact spelling most
    languages use (Karachi over the old Kurrachi), then English before
    untagged / local (Yerevan, not Erewan), then with diacritics. Abjads (Arabic, Hebrew: unidecode
    drops the vowels) compare consonants only (w, y count as vowels: Tunis,
    not Tiwnis). Too far: the transliteration itself for Cyrillic / Greek /
    Armenian / Georgian, else GeoNames' main name (unidecode is unusable for
    Thai, CJK, Indic scripts: Bangkok, Tokyo, Beijing). Pillow here has no
    complex text layout (raqm) and the labels are meant to be Latin anyway.
  * ascii_only: accents and special letters folded (Muenchen stays
    "Munchen", Lodz, Sao Paulo) via unidecode, or a built-in fold without it.
"""
from __future__ import annotations

import csv
import difflib
import io
import pickle
import threading
import unicodedata
import zipfile

import core

NAMES_DIR = core.DATA_DIR / "cities"
NAMES_ZIP = NAMES_DIR / "alternateNamesV2.zip"
COUNTRY_INFO = NAMES_DIR / "countryInfo.txt"
NAMES_CACHE = NAMES_DIR / "city_names.pkl"
NAMES_CACHE_VERSION = 1

# Exonym languages offered in the tab (Latin-script languages; the romanized
# ones work too but give transliterations).
EXONYM_LANGS = [("en", "English"), ("fr", "French"), ("de", "German"), ("es", "Spanish"), ("it", "Italian"),
                ("pt", "Portuguese"), ("nl", "Dutch"), ("pl", "Polish"), ("cs", "Czech"), ("sv", "Swedish"),
                ("tr", "Turkish"), ("ru", "Russian (romanized)")]
# GeoNames pseudo-languages that aren't names.
# Countries where several official languages each have their own city names:
# the endonym is the most used of them, not the first listed.
MULTILINGUAL = {"BE": ["nl", "fr", "de"], "CH": ["de", "fr", "it", "rm"], "CA": ["en", "fr"], "FI": ["fi", "sv"],
                "LU": ["lb", "fr", "de"], "ES": ["es", "ca", "eu", "gl"]}
_NOT_LANGS = {"link", "post", "iata", "icao", "faac", "abbr", "wkdt", "unlc", "tcid", "phon", "piny", "fr_1793"}

try:
    from unidecode import unidecode as _unidecode
except ImportError:  # optional: the built-in fold covers Latin letters
    _unidecode = None

_lock = threading.Lock()
_names = None      # {geoname_id: [(lang, name, preferred, short), ...]}
_langs = None      # {ISO2: [lang, ...]}
_build = {"running": False, "error": None}


def names_downloaded() -> bool:
    return NAMES_ZIP.exists() and COUNTRY_INFO.exists()


def _cache_key(csv_path) -> dict:
    zs, cs = NAMES_ZIP.stat(), csv_path.stat()
    return {"v": NAMES_CACHE_VERSION, "zip": [zs.st_size, int(zs.st_mtime)], "csv": [cs.st_size, int(cs.st_mtime)]}


def country_languages() -> dict:
    """{ISO2: [language codes, most used first]} from countryInfo.txt."""
    global _langs
    if _langs is None:
        out = {}
        if COUNTRY_INFO.exists():
            for line in COUNTRY_INFO.read_text(encoding="utf-8").splitlines():
                if not line or line.startswith("#"):
                    continue
                cols = line.split("\t")
                if len(cols) > 15:
                    codes = []
                    for c in cols[15].split(","):
                        c = c.strip().split("-")[0].lower()
                        if c and c not in codes:
                            codes.append(c)
                    out[cols[0]] = codes
        _langs = out
    return _langs


def build_names(ids, csv_path, progress=print) -> dict:
    """Stream alternateNamesV2.zip and keep the names of `ids` (GeoNames
    ids as strings); saved to NAMES_CACHE."""
    ids = set(ids)
    out = {}
    with zipfile.ZipFile(NAMES_ZIP) as zf:
        member = next(n for n in zf.namelist() if n.lower().startswith("alternatenames") and n.endswith(".txt"))
        with io.TextIOWrapper(zf.open(member), encoding="utf-8", newline="") as f:
            for n, line in enumerate(f):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4 or parts[1] not in ids:
                    continue
                lang = parts[2]
                if lang in _NOT_LANGS or (len(parts) > 7 and (parts[6] == "1" or parts[7] == "1")):
                    continue  # not a name, colloquial or historic
                out.setdefault(parts[1], []).append((lang, parts[3], len(parts) > 4 and parts[4] == "1",
                                                     len(parts) > 5 and parts[5] == "1"))
                if n % 4_000_000 == 0 and n:
                    progress(f"City names: {n // 1_000_000} M rows read")
    with open(NAMES_CACHE, "wb") as f:
        pickle.dump({"key": _cache_key(csv_path), "names": out}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def load_names(ids_fn, csv_path, wait: bool = False):
    """The per-city names, or None while they're being built (the build runs
    in a background thread; `wait` builds in the calling thread). `ids_fn`
    gives the city ids (called only when building)."""
    global _names
    if _names is not None:
        return _names
    if not names_downloaded():
        return None
    with _lock:
        if _names is not None:
            return _names
        if NAMES_CACHE.exists():
            try:
                with open(NAMES_CACHE, "rb") as f:
                    data = pickle.load(f)
                if data.get("key") == _cache_key(csv_path):
                    _names = data["names"]
                    return _names
            except Exception:
                pass
        if _build["running"]:
            return None
        _build["running"] = True

    def run():
        global _names
        try:
            names = build_names(ids_fn(), csv_path)
            with _lock:
                _names = names
        except Exception as e:  # reported to the tab
            _build["error"] = str(e)
        finally:
            _build["running"] = False

    if wait:
        run()
        return _names
    threading.Thread(target=run, daemon=True).start()
    return None


def build_status() -> dict:
    return {"downloaded": names_downloaded(), "ready": _names is not None, "running": _build["running"],
            "error": _build["error"]}


# ---- text helpers ------------------------------------------------------

_FOLD = str.maketrans({"ß": "ss", "æ": "ae", "Æ": "AE", "ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D",
                       "ð": "d", "Ð": "D", "þ": "th", "Þ": "Th", "œ": "oe", "Œ": "OE", "ı": "i", "ħ": "h", "Ħ": "H",
                       "ŧ": "t", "Ŧ": "T", "ŀ": "l", "ʻ": "'", "ʼ": "'", "’": "'", "‘": "'"})


def to_ascii(text: str) -> str:
    """Accents and special letters folded to plain ASCII."""
    if _unidecode is not None:
        return _unidecode(text)
    text = unicodedata.normalize("NFKD", text.translate(_FOLD))
    return "".join(c for c in text if not unicodedata.combining(c) and ord(c) < 128)


def _is_latin(text: str) -> bool:
    for c in text:
        if c.isalpha():
            try:
                if not unicodedata.name(c).startswith("LATIN"):
                    return False
            except ValueError:
                return False
    return True


def _is_cjk(text: str) -> bool:
    for c in text:
        if c.isalpha():
            n = unicodedata.name(c, "")
            if n.startswith(("CJK", "HIRAGANA", "KATAKANA", "HANGUL")):
                return True
    return False


def _is_abjad(text: str) -> bool:
    for c in text:
        if c.isalpha():
            n = unicodedata.name(c, "")
            if n.startswith(("ARABIC", "HEBREW", "SYRIAC", "THAANA")):
                return True
    return False


def _fold_key(text: str) -> str:
    return "".join(c for c in to_ascii(text).lower() if c.isalnum())


def _best(entries):
    """The name to use among one language's entries: preferred, then short,
    then the first listed."""
    return min(enumerate(entries), key=lambda ie: (not ie[1][2], not ie[1][3], ie[0]))[1][1]


def _lev(a: str, b: str) -> int:
    """Levenshtein distance."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _consonants(key: str) -> str:
    return "".join(c for c in key if c not in "aeiouwy")


_GREEK_MODERN = str.maketrans({"η": "ι", "ή": "ί", "Η": "Ι", "Ή": "Ί", "β": "v", "Β": "V"})
_TRANSLIT_OK = ("CYRILLIC", "GREEK", "ARMENIAN", "GEORGIAN")


def _script(text: str) -> str:
    for c in text:
        if c.isalpha() and ord(c) > 0x24F:
            return unicodedata.name(c, "").split(" ")[0]
    return "LATIN"


def _romanize(native: str, entries, main_name: str, local_langs=()) -> str:
    """A Latin-script form of a non-Latin name (see the module docstring)."""
    if _is_cjk(native) or _unidecode is None:
        return main_name
    script = _script(native)
    translit = to_ascii(native.translate(_GREEK_MODERN) if script == "GREEK" else native)
    abjad = _is_abjad(native)
    key = (lambda t: _consonants(_fold_key(t))) if abjad else _fold_key
    target = key(translit)
    if not target:
        return main_name
    exact_use = {}
    for e in entries:
        exact_use[e[1]] = exact_use.get(e[1], 0) + 1
    pool = {}  # name -> priority on ties (0 English / main name, 1 untagged / local: Yerevan, not Erewan)
    for lang, name, _p, _s in entries:
        base = lang.split("-")[0].lower()
        if not _is_latin(name) or not (base == "" or base in local_langs or base == "en"):
            continue
        prio = 0 if base == "en" else 1
        pool[name] = min(pool.get(name, 9), prio)
    if _is_latin(main_name):
        pool[main_name] = min(pool.get(main_name, 9), 0)
    if pool:
        best = min(pool, key=lambda n: (_lev(key(n), target), -exact_use.get(n, 0), pool[n], n == to_ascii(n)))
        if _lev(key(best), target) <= max(1, round(0.34 * len(target))):
            return best
    if script in _TRANSLIT_OK:
        return " ".join(w[:1].upper() + w[1:] for w in translit.split())
    return main_name


def _local_langs(country: str) -> list:
    langs = country_languages().get(country, [])
    return [l for i, l in enumerate(langs) if l != "en" or i == 0]


def endonym(entries, country: str, main_name: str) -> str:
    langs = _local_langs(country)
    by_lang = {}
    for e in entries:
        by_lang.setdefault(e[0].split("-")[0].lower(), []).append(e)
    name = None
    multi = [l for l in MULTILINGUAL.get(country, []) if l in by_lang]
    if multi:
        rank = {l: i for i, l in enumerate(multi)}
        use = {}
        for e in entries:
            k = _fold_key(e[1])
            use[k] = use.get(k, 0) + 1
        local = [e for l in multi for e in by_lang[l]]
        best = min(local, key=lambda e: (-use.get(_fold_key(e[1]), 0), rank[e[0].split("-")[0].lower()],
                                         not e[2], not e[3]))
        same = [e for e in local if _fold_key(e[1]) == _fold_key(best[1])]
        # Same letters: the exact spelling most languages use (Dutch Brussel,
        # not German Brüssel), then with diacritics (Montréal), then preferred.
        exact_use = {}
        for e in entries:
            exact_use[e[1]] = exact_use.get(e[1], 0) + 1
        name = min(same, key=lambda e: (-exact_use.get(e[1], 0), e[1] == to_ascii(e[1]), not e[2], not e[3]))[1]
    elif langs and langs[0] in by_lang:
        name = _best(by_lang[langs[0]])
    elif _is_latin(main_name):
        name = main_name  # no entry in the first language: the main name is the local one
    else:
        name = next((_best(by_lang[l]) for l in langs if l in by_lang), main_name)
    if not _is_latin(name):
        name = _romanize(name, entries, main_name, langs)
    return name


def exonym(entries, lang: str, country: str, main_name: str) -> str:
    own = [e for e in entries if e[0].split("-")[0].lower() == lang]
    if not own:
        return endonym(entries, country, main_name)
    name = _best(own)
    if not _is_latin(name):
        name = _romanize(name, entries, main_name, _local_langs(country) + [lang])
    return name


def label_for(names, gid: str, country: str, main_name: str, name_set: str, lang: str, ascii_only: bool) -> str:
    entries = names.get(gid, []) if names else []
    name = endonym(entries, country, main_name) if name_set == "endonym" else exonym(entries, lang, country, main_name)
    if not _is_latin(name):  # e.g. no names file and a non-Latin main name
        name = _romanize(name, entries, main_name, _local_langs(country))
    return to_ascii(name) if ascii_only else name
