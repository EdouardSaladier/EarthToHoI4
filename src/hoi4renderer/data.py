"""Loads map bitmaps and game data (provinces, states, owners, colors, names)."""
import os
import re

import numpy as np
from PIL import Image

from . import pdx

Image.MAX_IMAGE_PIXELS = None


def log(*a):
    print(*a, flush=True)


def load_indexed(path):
    """8-bit bitmap -> palette indices (H, W) uint8."""
    im = Image.open(path)
    if im.mode not in ("P", "L"):
        im = im.convert("L")
    return np.asarray(im, dtype=np.uint8)


def load_rgb(path):
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def load_rgba(path):
    im = Image.open(path)
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    return np.asarray(im, dtype=np.uint8)


class MapData:
    def __init__(self, vfs, date="1936.1.1", language="english"):
        self.vfs = vfs
        self.date = pdx.date_key(date) or (1936, 1, 1)
        self.language = language
        self._load_default_map()
        self._load_provinces()
        self._load_states()
        self._load_countries()
        self._load_localisation()

    # ------------------------------------------------------------------ map
    def _map_file(self, key, default):
        name = self.default_map.get(key, default)
        p = self.vfs.find("map/" + name)
        if p is None:
            raise FileNotFoundError("map/" + name)
        return p

    def _load_default_map(self):
        p = self.vfs.find("map/default.map")
        self.default_map = {}
        if p:
            for k, _, v in pdx.parse_file(p):
                if k and isinstance(v, str):
                    self.default_map[k] = v

    def _load_provinces(self):
        log("Reading provinces...")
        rgb = load_rgb(self._map_file("provinces", "provinces.bmp"))
        self.height_px, self.width_px = rgb.shape[:2]
        keys = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
        del rgb

        ids, cols, types = [], [], []
        with open(self._map_file("definitions", "definition.csv"), "r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                parts = line.strip().split(";")
                if len(parts) < 5:
                    continue
                try:
                    pid, r, g, b = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                except ValueError:
                    continue
                ids.append(pid)
                cols.append((r << 16) | (g << 8) | b)
                types.append(parts[4].strip().lower())
        ids = np.array(ids, dtype=np.int32)
        cols = np.array(cols, dtype=np.int32)
        order = np.argsort(cols, kind="stable")
        scols, sids = cols[order], ids[order]
        pos = np.searchsorted(scols, keys)
        pos = np.clip(pos, 0, len(scols) - 1)
        prov = np.where(scols[pos] == keys, sids[pos], 0).astype(np.int32)
        self.province_map = prov
        self.max_province = int(ids.max()) if len(ids) else 0
        self.province_type = np.zeros(self.max_province + 1, dtype=np.uint8)  # 0 land 1 sea 2 lake
        for pid, t in zip(ids, types):
            self.province_type[pid] = 1 if t == "sea" else (2 if t == "lake" else 0)

    # --------------------------------------------------------------- states
    def _apply_history(self, hist, state):
        """Apply owner changes from a state's history block up to self.date."""
        if not isinstance(hist, pdx.Block):
            return
        dated = []
        for k, _, v in hist:
            if k == "owner" and isinstance(v, str):
                state["owner"] = v
            dk = pdx.date_key(k) if k else None
            if dk and isinstance(v, pdx.Block):
                dated.append((dk, v))
        for dk, v in sorted(dated, key=lambda x: x[0]):
            if dk <= self.date:
                own = v.get("owner")
                if isinstance(own, str):
                    state["owner"] = own

    def _load_states(self):
        log("Reading states...")
        self.states = {}
        for fn, path in self.vfs.list_dir("history/states", (".txt",)).items():
            try:
                root = pdx.parse_file(path)
            except Exception as e:  # noqa: BLE001
                log(f"  warning: cannot parse {fn}: {e}")
                continue
            for blk in root.get_all("state"):
                if not isinstance(blk, pdx.Block):
                    continue
                try:
                    sid = int(blk.get("id"))
                except (TypeError, ValueError):
                    continue
                provs = blk.get("provinces")
                plist = []
                if isinstance(provs, pdx.Block):
                    for v in provs.values():
                        try:
                            plist.append(int(v))
                        except (TypeError, ValueError):
                            pass
                st = {"id": sid, "name": blk.get("name", ""), "provinces": plist, "owner": None,
                      "impassable": str(blk.get("impassable", "no")).lower() == "yes"}
                self._apply_history(blk.get("history"), st)
                self.states[sid] = st

        self.prov_state = np.zeros(self.max_province + 1, dtype=np.int32)
        for sid, st in self.states.items():
            for p in st["provinces"]:
                if 0 < p <= self.max_province:
                    self.prov_state[p] = sid

    # ------------------------------------------------------------ countries
    def _load_countries(self):
        log("Reading countries...")
        tag_files = {}
        for fn, path in self.vfs.list_dir("common/country_tags", (".txt",)).items():
            for k, _, v in pdx.parse_file(path):
                if k and isinstance(v, str) and len(k) == 3:
                    tag_files[k] = v
        owners = sorted({st["owner"] for st in self.states.values() if st["owner"]})
        self.tags = owners
        self.tag_index = {t: i + 1 for i, t in enumerate(owners)}  # 0 = none
        colors = {}
        for tag in owners:
            rel = tag_files.get(tag)
            c = None
            if rel:
                p = self.vfs.find("common/" + rel)
                if p:
                    c = pdx.parse_color(pdx.parse_file(p).get("color"))
            colors[tag] = c
        for fn, path in self.vfs.list_dir("common/countries", (".txt",)).items():
            if fn.lower() != "colors.txt":
                continue
            for k, _, v in pdx.parse_file(path):
                if k in colors and isinstance(v, pdx.Block):
                    c = pdx.parse_color(v.get("color"))
                    if c:
                        colors[k] = c
        rng = np.random.default_rng(1)
        for t in owners:
            if colors[t] is None:
                colors[t] = tuple(int(x) for x in rng.integers(60, 220, 3))
        self.country_color = colors

        # ruling party (for the displayed name)
        self.ruling_party = {}
        for fn, path in self.vfs.list_dir("history/countries", (".txt",)).items():
            tag = fn[:3].upper()
            if tag not in self.tag_index:
                continue
            try:
                root = pdx.parse_file(path)
            except Exception:  # noqa: BLE001
                continue
            pol = root.get("set_politics")
            if isinstance(pol, pdx.Block) and isinstance(pol.get("ruling_party"), str):
                self.ruling_party[tag] = pol.get("ruling_party")

    # ---------------------------------------------------------- localisation
    def _load_localisation(self):
        log("Reading localisation...")
        wanted = set()
        for t in self.tags:
            wanted.add(t)
            wanted.add(t + "_DEF")
            for ideo in ("democratic", "fascism", "communism", "neutrality", self.ruling_party.get(t, "")):
                wanted.add(f"{t}_{ideo}")
        loc = {}
        lang = self.language
        pat = re.compile(r"_l_" + re.escape(lang) + r"\.yml$", re.I)
        files = []
        for base in (self.vfs.game, self.vfs.mod):
            if not base:
                continue
            root = os.path.join(base, "localisation")
            for dp, _, fns in os.walk(root):
                for fn in fns:
                    if pat.search(fn):
                        files.append((0 if base == self.vfs.game else 1, "replace" in dp.lower(), os.path.join(dp, fn)))
        files.sort(key=lambda x: (x[0], x[1]))
        for _, _, path in files:
            tmp = {}
            pdx.parse_localisation(path, tmp)
            for k, v in tmp.items():
                if k in wanted:
                    loc[k] = v
        self.names = {}
        for t in self.tags:
            party = self.ruling_party.get(t)
            name = (loc.get(f"{t}_{party}") if party else None) or loc.get(t) or t
            self.names[t] = _clean_loc(name)

    # ------------------------------------------------------------- derived
    def owner_map(self):
        """Per-pixel country index (0 = none / sea), uint8 or uint16."""
        lut = np.zeros(self.max_province + 1, dtype=np.uint16)
        for sid, st in self.states.items():
            o = st["owner"]
            if not o:
                continue
            ci = self.tag_index[o]
            for p in st["provinces"]:
                if 0 < p <= self.max_province and self.province_type[p] == 0:
                    lut[p] = ci
        return lut[self.province_map]

    def state_map(self):
        """Per-pixel state id, 0 for sea / lake / stateless."""
        lut = self.prov_state.copy()
        lut[self.province_type != 0] = 0
        return lut[self.province_map]

    def impassable_map(self):
        lut = np.zeros(self.max_province + 1, dtype=bool)
        for st in self.states.values():
            if st["impassable"]:
                for p in st["provinces"]:
                    if 0 < p <= self.max_province:
                        lut[p] = True
        return lut[self.province_map]

    def sea_region_lut(self):
        """Province id -> strategic region id (map/strategicregions), or None."""
        files = self.vfs.list_dir("map/strategicregions", (".txt",))
        if not files:
            return None
        lut = np.zeros(self.max_province + 1, dtype=np.int32)
        for fn, path in files.items():
            try:
                root = pdx.parse_file(path)
            except Exception:  # noqa: BLE001
                continue
            for blk in root.get_all("strategic_region"):
                if not isinstance(blk, pdx.Block):
                    continue
                try:
                    rid = int(blk.get("id"))
                except (TypeError, ValueError):
                    continue
                provs = blk.get("provinces")
                if isinstance(provs, pdx.Block):
                    for v in provs.values():
                        try:
                            p = int(v)
                        except (TypeError, ValueError):
                            continue
                        if 0 < p <= self.max_province:
                            lut[p] = rid
        return lut

    def terrain_texture_lut(self):
        """terrain.bmp palette index -> atlas texture index (from common/terrain)."""
        lut = np.zeros(256, dtype=np.int16)
        lut[:] = -1
        for fn, path in self.vfs.list_dir("common/terrain", (".txt",)).items():
            root = pdx.parse_file(path)
            terr = root.get("terrain")
            if not isinstance(terr, pdx.Block):
                continue
            for _, _, v in terr:
                if not isinstance(v, pdx.Block):
                    continue
                col = v.get("color")
                tex = v.get("texture")
                if not isinstance(col, pdx.Block):
                    continue
                vals = col.values()
                try:
                    idx = int(vals[0])
                    t = int(tex)
                except (TypeError, ValueError, IndexError):
                    continue
                if 0 <= idx < 256 and lut[idx] < 0:
                    lut[idx] = t
        lut[(lut < 0) | (lut > 15)] = 0
        return lut.astype(np.uint8)


def _clean_loc(s):
    s = re.sub(r"§.", "", s)          # colour codes
    s = re.sub(r"\$[^$]*\$", "", s)   # variables
    s = re.sub(r"\[[^\]]*\]", "", s)  # scripted loc
    return s.replace("\\n", " ").strip()
