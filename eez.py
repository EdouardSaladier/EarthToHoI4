"""Exclusive Economic Zones on the country map (Countries tab overlay).

Data: Marine Regions (VLIZ) World EEZ v12 (CC BY 4.0), from their GeoServer
WFS as shapefiles (Download tab -> "Exclusive Economic Zones"), three layers
into data/eez/:
  * eez -- the 200 NM zones (229), overlapping claims (35) and joint
    regimes (21); they start at the baseline, so
  * eez_internal_waters (113: behind straight baselines -- Norway's
    fjords, bays, ...) and
  * eez_archipelagic_waters (23: Indonesia, the Philippines, ...) are
    drawn too, else those coasts would be hollow.
Owner = iso_ter1 (the territory: American Samoa, Svalbard, ...) else
iso_sov1 -- all of them are GADM tags; overlapping claims / joint regimes
also have a second party (iso_ter2 / iso_sov2), not used: they get the first
party's flat colour like any zone (user: flat colours only, no stripes).

On the render (`overlay`), sea pixels only (the country map's water: not
land, not lakes, not hidden countries' land):
  * same filters as the countries: hidden (excluded) countries' zones not
    drawn; merged autonomous territories in their parent's colour (the
    render's country colours); fragments under the Land/Sea
    `min_region_px` merged into the most common neighbouring zone, else
    left as plain sea;
  * GADM coast vs EEZ discrepancies: sea pixels no zone covers within
    GAP_KM of land (Marine Regions' coastline differs from GADM's; bays and
    river mouths GADM keeps as water) are filled from the adjacent zones,
    growing through water only (never across land), one pixel per step --
    the open ocean beyond every EEZ and the Caspian (no EEZ) stay plain;
  * microstates' zones erased (a country's zones < MICRO_EEZ_KM2 in all:
    Monaco, Singapore, Sint Maarten, Slovenia, Neum, Aqaba), their sea to
    the nearest other zone -- user: too small to be significant;
  * sea within COAST_KM of a coast belongs to that coast's country (zone
    boundaries meet the land border where the map draws it);
  * no exclaves: every piece of a country's zones touches its own land or
    the frame; slivers join the neighbouring zone they border most, larger
    pieces (zones of islands not on the map) become plain sea;
  * enclosed bays no zone reaches (Garabogazkol) -- unzoned water not
    reaching the frame, bordered mostly by land -- take the nearest zone;
  * tint = sea colour blended with the country colour at `opacity`.
"""
import math
import threading
import time

import numpy as np
from PIL import Image
from scipy import ndimage

import core

EEZ_DIR = core.DATA_DIR / "eez"
LAYERS = ("eez", "eez_internal_waters", "eez_archipelagic_waters")
OPACITY_DEFAULT = 0.35
GAP_KM = 25.0
HOLE_MIN_DEG2 = 0.001  # polygon holes at least this big are drawn (smaller ones: islets)
MICRO_EEZ_KM2 = 1000.0  # a country's zones smaller than this in all are erased (Monaco, Singapore, Neum, Aqaba)
COAST_KM = 10.0        # sea this close to a coast belongs to that coast's country (the zones' lateral
                       # boundaries meet the land border where the map draws it)
EXCLAVE_MERGE_KM2 = 5000.0  # an exclave piece smaller than this joins its neighbouring zone; larger ones (zones
                            # of islands the map doesn't show) become plain sea
BAY_LAND_SHARE = 0.5   # an unzoned water body not reaching the frame, bordered this much by land, is a bay
_lock = threading.Lock()
_data = None


def available() -> bool:
    return all((EEZ_DIR / f"{name}.shp").exists() for name in LAYERS)


def _dbf_text(path, names):
    """Some .dbf text fields (Marine Regions writes Latin-1)."""
    import struct
    with open(path, "rb") as f:
        n, hlen, rlen = struct.unpack("<IHH", f.read(32)[4:12])
        fields, off = {}, 1
        while True:
            d = f.read(32)
            if not d or d[0] == 0x0D:
                break
            fields[d[:11].split(b"\0")[0].decode("ascii")] = (off, d[16])
            off += d[16]
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=hlen, shape=(n, rlen))
    out = {}
    for name in names:
        a, ln = fields[name]
        col = np.ascontiguousarray(raw[:, a:a + ln]).view(f"S{ln}").ravel()
        out[name] = [v.decode("latin-1").strip() for v in col]
    return out


def load():
    """[(points, ring_start, record_rings, owner, second owner, name, type)] per layer (memoized)."""
    global _data
    with _lock:
        if _data is None:
            layers = []
            for name in LAYERS:
                path = EEZ_DIR / f"{name}.shp"
                pts, ring_start, rec_rings = core._read_polygon_shapefile(path)
                c = _dbf_text(path.with_suffix(".dbf"), ["geoname", "pol_type", "iso_sov1", "iso_ter1",
                                                         "iso_sov2", "iso_ter2"] if name == "eez"
                              else ["geoname", "pol_type", "iso_sov1", "iso_ter1"])
                own = [t or s for t, s in zip(c["iso_ter1"], c["iso_sov1"])]
                own2 = ([t or s for t, s in zip(c["iso_ter2"], c["iso_sov2"])] if "iso_sov2" in c
                        else [""] * len(own))
                # Rings drawn: the exteriors (clockwise in shapefiles) and the
                # holes of at least HOLE_MIN_DEG2. The ~400k small holes are
                # islets (land anyway -- only sea pixels are painted -- and
                # projecting them took 15 s), but big holes can be *other
                # zones*: the joint regimes Iceland / Jan Mayen and Iceland /
                # Faroe, the Egypt / Libya overlapping claim, Libya's
                # internal waters (Gulf of Sirte) are cut out of the 200 NM
                # zones -- dropping every hole painted those over.
                x, y = pts[:, 0], pts[:, 1]
                nxt = np.arange(1, len(pts) + 1)
                nxt[ring_start[1:] - 1] = ring_start[:-1]
                signed = 0.5 * np.add.reduceat(x * y[nxt] - x[nxt] * y, ring_start[:-1])
                keep = (signed < 0) | (np.abs(signed) >= HOLE_MIN_DEG2)
                # zone area in km2 (holes subtracted; the dbf's area_km2 is
                # wrong for some: Bouvet 6 km2)
                mid_lat = np.add.reduceat(y, ring_start[:-1]) / np.diff(ring_start)
                ring_km2 = -signed * np.cos(np.radians(mid_lat)) * 111.32 * 110.57
                area_km2 = np.array([ring_km2[rec_rings[i]:rec_rings[i + 1]].sum() for i in range(len(own))])
                # 200 NM zones first, then overlapping claims / joint regimes
                order = sorted(range(len(own)), key=lambda i: c["pol_type"][i] not in ("200NM", "Internal waters",
                                                                                     "Archipelagic waters"))
                layers.append((pts, ring_start, rec_rings, own, own2, c["geoname"], c["pol_type"], keep, order,
                               area_km2))
            _data = layers
        return _data


def _clean_strays(lab, id_code, water, min_px: int) -> int:
    """Patches of one sea colour (zone by colour, or plain sea = -2) under
    min_px -> the colour they border most (zones first); a patch with only
    land around it stays. Repeated until stable. Returns pixels changed."""
    h, w = lab.shape
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    changed = 0
    for _round in range(4):
        col = np.where(lab > 0, id_code[lab], -2)
        col[~water] = -1                                    # land: not a sea colour
        codes, dense = np.unique(col, return_inverse=True)
        dense = dense.reshape(col.shape).astype(np.int32) + 1
        land_idx = int(np.searchsorted(codes, -1)) + 1 if (codes == -1).any() else -1
        fixes = []
        for ci, box in enumerate(ndimage.find_objects(dense), start=1):
            if box is None or ci == land_idx:
                continue
            comp, _n = ndimage.label(dense[box] == ci)
            sizes = np.bincount(comp.ravel())
            for k in np.flatnonzero(sizes[1:] < min_px) + 1:
                ys, xs = np.nonzero(comp == k)
                ys, xs = ys + box[0].start, xs + box[1].start
                sl = (slice(max(ys.min() - 1, 0), min(ys.max() + 2, h)), slice(max(xs.min() - 1, 0), min(xs.max() + 2, w)))
                m = np.zeros((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start), dtype=bool)
                m[ys - sl[0].start, xs - sl[1].start] = True
                fixes.append((sl, m))
        n_fix = 0
        for sl, m in fixes:
            colw = np.where(lab[sl] > 0, id_code[lab[sl]], -2)
            colw[~water[sl]] = -1
            ring = ndimage.binary_dilation(m, structure=cross) & ~m & water[sl]
            nb = colw[ring]
            if not len(nb):
                continue  # only land around: its own little water body
            zones_nb = nb[nb >= 0]
            Lw = lab[sl]
            if len(zones_nb):
                vals, cnt = np.unique(zones_nb, return_counts=True)
                best = vals[cnt.argmax()]
                if (colw[m] == best).all():
                    continue
                Lw[m] = np.bincount(Lw[ring & (colw == best)]).argmax()
            else:
                if (Lw[m] == 0).all():
                    continue
                Lw[m] = 0
            n_fix += int(m.sum())
        changed += n_fix
        if not n_fix:
            break
    return changed


def _exclaves(lab, id_code, code, land):
    """Pieces of a country's zones (4-connected, by colour) touching neither
    that country's land nor the frame: [(window slice, piece mask, its
    4-neighbour ring, zone colours in the window)]."""
    h, w = lab.shape
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    zc = np.where(lab > 0, id_code[lab], -1)
    codes, dense = np.unique(zc, return_inverse=True)
    dense = dense.reshape(zc.shape).astype(np.int32)  # 0 = no zone (code -1 sorts first)
    if codes[0] != -1:
        dense += 1
    out = []
    for ci, box in enumerate(ndimage.find_objects(dense), start=1):
        if box is None:
            continue
        zcode = int(codes[ci]) if codes[0] == -1 else int(codes[ci - 1])
        comp, _n = ndimage.label(dense[box] == ci)  # this colour's pieces, in its bbox only
        oy, ox = box[0].start, box[1].start
        for k, sl0 in enumerate(ndimage.find_objects(comp), start=1):
            if sl0 is None:
                continue
            y0, y1 = sl0[0].start + oy, sl0[0].stop + oy
            x0, x1 = sl0[1].start + ox, sl0[1].stop + ox
            if y0 == 0 or x0 == 0 or y1 == h or x1 == w:
                continue  # reaches the frame: goes on outside the view
            sl = (slice(max(y0 - 1, 0), y1 + 1), slice(max(x0 - 1, 0), x1 + 1))
            m = np.zeros((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start), dtype=bool)
            cy0, cx0 = sl[0].start - oy, sl[1].start - ox
            sub = comp[max(cy0, 0):cy0 + m.shape[0], max(cx0, 0):cx0 + m.shape[1]] == k
            m[max(-cy0, 0):max(-cy0, 0) + sub.shape[0], max(-cx0, 0):max(-cx0, 0) + sub.shape[1]] = sub
            ring = ndimage.binary_dilation(m, structure=cross) & ~m
            if ((code[sl] == zcode) & land[sl] & ring).any():
                continue  # touches its own coast
            out.append((sl, m, ring, zc[sl], zcode))
    return out


def micro_tags() -> set:
    """Countries whose zones (200 NM + internal + archipelagic, not the
    overlapping / joint ones) add up to less than MICRO_EEZ_KM2."""
    tot = {}
    for layer in load():
        own, types, area = layer[3], layer[6], layer[9]
        for o, ty, a in zip(own, types, area):
            if ty in ("Overlapping claim", "Joint regime"):
                continue
            tot[o] = tot.get(o, 0.0) + float(a)
    return {o for o, a in tot.items() if a < MICRO_EEZ_KM2}


def overlay(image: Image.Image, debug, excluded=(), opacity: float = OPACITY_DEFAULT, min_region_px: int = 8):
    """(country map with the EEZ tint on its sea, report)."""
    lab, tags, report = zones(image, debug, excluded, min_region_px)
    t0 = time.time()
    rgb = debug["country_rgb"]
    arr = np.asarray(image.convert("RGB"))
    colors = np.zeros((len(tags) + 1, 3), dtype=np.float32)
    for k, t in enumerate(tags, start=1):
        colors[k] = rgb.get(t, core.BACKGROUND_COLOR)
    out = arr.astype(np.float32)
    zone = lab > 0
    c = colors[lab[zone]]
    sea = np.asarray(core.BACKGROUND_COLOR, dtype=np.float32)
    out[zone] = sea * (1.0 - opacity) + c * opacity
    report["seconds"] = round(report["seconds"] + time.time() - t0, 2)
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), mode="RGB"), report


def zones(image: Image.Image, debug, excluded=(), min_region_px: int = 8):
    """(zone label per pixel -- 0 = none, k = tags[k - 1] --, tags, report):
    the EEZ zones on the render's sea after every rule below (the overlay
    tints them; the Provinces tab colours its sea provinces by them)."""
    import states
    t0 = time.time()
    view = debug["view"]
    h, w = view["height"], view["width"]
    func = core.PROJECTIONS[view["projection"]]["func"]
    arr = np.asarray(image.convert("RGB"))
    water = (arr == np.asarray(core.BACKGROUND_COLOR, dtype=arr.dtype)).all(axis=2)
    if debug.get("hidden_land") is not None:
        water &= ~debug["hidden_land"]
    rgb = debug["country_rgb"]
    excluded = set(excluded)
    tags = sorted({t for layer in load() for t in layer[3] if t})
    tag_id = {t: i + 1 for i, t in enumerate(tags)}
    lab = np.zeros((h, w), dtype=np.int32)    # zone owner (tag id), 0 = none
    zones = 0
    for pts, ring_start, rec_rings, own, _own2, _names, _types, keep, order, _area in load():
        for i in order:
            r0, r1 = rec_rings[i], rec_rings[i + 1]
            if r1 <= r0:
                continue
            rings = [(False, pts[ring_start[r]:ring_start[r + 1]]) for r in range(r0, r1)
                     if keep[r] and ring_start[r + 1] - ring_start[r] >= 3]
            rp = states._to_px(view, rings, func)
            if not rp:
                continue
            if core._scanline_fill(lab, rp, tag_id[own[i]]):
                zones += 1
    lab[~water] = 0
    # same filters as the countries: hidden countries' zones out
    hidden_ids = [tag_id[t] for t in excluded if t in tag_id]
    lab[np.isin(lab, hidden_ids)] = 0
    report = {"zones": zones}

    # Microstates' zones (all of a country's zones < MICRO_EEZ_KM2) erased:
    # their sea goes to the nearest other zone.
    micro = micro_tags()
    micro_ids = [tag_id[t] for t in micro if t in tag_id and (lab == tag_id[t]).any()]
    report["microstates_erased"] = sorted(t for t in micro if t in tag_id and tag_id[t] in micro_ids)
    if micro_ids:
        gone = np.isin(lab, micro_ids)
        keep_z = (lab > 0) & ~gone
        if keep_z.any():
            _d, (iy, ix) = ndimage.distance_transform_edt(~keep_z, return_indices=True)
            lab[gone] = lab[iy[gone], ix[gone]]
        else:
            lab[gone] = 0

    # Coast discrepancies: unzoned sea near land takes the adjacent zone
    # (growth through water only).
    km_px = core._km_per_px(debug)
    gap_px = max(1, int(math.ceil(GAP_KM / max(km_px, 1e-6))))
    land = ~water & ~(debug["lakes"] if debug.get("lakes") is not None else np.zeros_like(water))
    near = ndimage.distance_transform_edt(~land) <= gap_px if land.any() else np.zeros_like(water)
    todo = water & (lab == 0) & near
    filled = 0
    if todo.any() and (lab > 0).any():
        ys, xs = np.nonzero(todo)
        y0, y1 = max(ys.min() - 1, 0), min(ys.max() + 2, h)
        x0, x1 = max(xs.min() - 1, 0), min(xs.max() + 2, w)
        L, T = lab[y0:y1, x0:x1], todo[y0:y1, x0:x1]
        for _ in range(gap_px):
            cur = L.copy()
            grew = False
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                src = np.roll(cur, (dy, dx), axis=(0, 1))
                ok = T & (L == 0) & (src > 0)
                if dy == 1:
                    ok[0, :] = False
                elif dy == -1:
                    ok[-1, :] = False
                if dx == 1:
                    ok[:, 0] = False
                elif dx == -1:
                    ok[:, -1] = False
                if ok.any():
                    L[ok] = src[ok]
                    grew = True
            if not grew:
                break
        filled = int((T & (L > 0)).sum())
    report["gap_px"], report["gap_km"] = filled, GAP_KM

    # Enclosed bays no zone reaches (Garabogazkol: a Caspian lagoon GADM
    # draws as sea, joined by its strait, ~70 km across): an unzoned water
    # body that doesn't reach the frame and is bordered mostly by land takes
    # the nearest zone. High-seas pockets between EEZs (bordered by zones)
    # stay plain.
    bays = 0
    unz = water & (lab == 0)
    if unz.any() and (lab > 0).any():
        comp, n = ndimage.label(unz)
        edge_ids = np.unique(np.concatenate([comp[0], comp[-1], comp[:, 0], comp[:, -1]]))
        cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
        for k, sl in enumerate(ndimage.find_objects(comp), start=1):
            if sl is None or k in edge_ids:
                continue
            sl = tuple(slice(max(a.start - 2, 0), a.stop + 2) for a in sl)
            m = comp[sl] == k
            ring = ndimage.binary_dilation(m, structure=cross) & ~m
            if not ring.any() or land[sl][ring].mean() < BAY_LAND_SHARE or not (lab[sl][ring] > 0).any():
                continue
            src = lab[sl] > 0
            _d, (iy, ix) = ndimage.distance_transform_edt(~src, return_indices=True)
            Lw = lab[sl]
            Lw[m] = Lw[iy[m], ix[m]]
            bays += int(m.sum())
    report["bay_px"] = bays

    # Coast anchoring: sea within COAST_KM of a coast takes that coast's
    # country's zone (when it has one in view) -- Marine Regions' land
    # borders are a little off GADM's, so a zone's lateral boundary could
    # start beside the neighbour's coast (Libya's zone ran ~20 px along
    # Egypt's). Compared by colour, so merged territories count as their parent.
    anchored = 0
    coast_px = max(1, int(round(COAST_KM / max(km_px, 1e-6))))
    code = (arr[..., 0].astype(np.int64) << 16) | (arr[..., 1].astype(np.int64) << 8) | arr[..., 2]
    present = np.unique(lab[lab > 0])
    code_to_id = {}
    for k in present.tolist():
        c_ = rgb.get(tags[k - 1])
        if c_ is not None:
            code_to_id.setdefault((c_[0] << 16) | (c_[1] << 8) | c_[2], k)
    if code_to_id:
        ccodes = np.array(sorted(code_to_id), dtype=np.int64)
        pos = np.clip(np.searchsorted(ccodes, code), 0, len(ccodes) - 1)
        coast_land = land & (ccodes[pos] == code)
        if debug.get("hidden_land") is not None:
            coast_land &= ~debug["hidden_land"]
        if coast_land.any():
            dist, (iy, ix) = ndimage.distance_transform_edt(~coast_land, return_indices=True)
            band = (lab > 0) & (dist <= coast_px)
            nearest_code = code[iy[band], ix[band]]
            id_of = np.array([code_to_id[int(v)] for v in ccodes], dtype=np.int64)
            cand = id_of[np.searchsorted(ccodes, nearest_code)]
            cur = lab[band]
            colour_of = np.array([[0, 0, 0]] + [list(rgb.get(tg, (0, 0, 0))) for tg in tags], dtype=np.int64)
            ccur = (colour_of[cur, 0] << 16) | (colour_of[cur, 1] << 8) | colour_of[cur, 2]
            differ = ccur != nearest_code
            by, bx = np.nonzero(band)
            lab[by[differ], bx[differ]] = cand[differ]
            anchored = int(differ.sum())
    report["coast_anchored_px"] = anchored

    # No exclaves (user): each piece of a country's zones (4-connected, by
    # colour: merged territories count as their parent) must touch that
    # country's own land or the frame (it goes on outside the view). Other
    # pieces: slivers (< EXCLAVE_MERGE_KM2 -- left by coast anchoring, a
    # microstate's erasing) join the neighbouring zone they share the
    # longest border with; larger ones are the zones of islands the map
    # doesn't show (erased by the Land/Sea steps at world scale: Micronesia,
    # Kiribati, St Helena, Tonga, ...) -> plain sea, not handed to a
    # neighbour. Repeated until stable.
    colour_of = np.array([[0, 0, 0]] + [list(rgb.get(tg, (0, 0, 0))) for tg in tags], dtype=np.int64)
    id_code = (colour_of[:, 0] << 16) | (colour_of[:, 1] << 8) | colour_of[:, 2]
    id_code[0] = -1
    exclave_px = exclaves = exclave_sea_px = 0
    biggest = []
    for _round in range(5):
        found = _exclaves(lab, id_code, code, land)
        for sl, m, ring, zc_w, zcode in found:
            biggest.append((int(m.sum()), zcode))
            if m.sum() * km_px * km_px >= EXCLAVE_MERGE_KM2 or not (ring & (zc_w >= 0)).any():
                exclave_sea_px += int(m.sum())
            Lw = lab[sl]
            nb = zc_w[ring & (zc_w >= 0)]
            if len(nb) and m.sum() * km_px * km_px < EXCLAVE_MERGE_KM2:
                vals, cnt = np.unique(nb, return_counts=True)
                Lw[m] = np.bincount(Lw[ring & (zc_w == vals[cnt.argmax()])]).argmax()
            else:
                Lw[m] = 0
            exclave_px += int(m.sum())
        exclaves += len(found)
        if not found:
            break
    report["exclaves"], report["exclave_px"], report["exclave_to_sea_px"] = exclaves, exclave_px, exclave_sea_px
    code_tag = {}
    for k, tg in enumerate(tags, start=1):
        code_tag.setdefault(int(id_code[k]), tg)
    report["exclave_largest"] = [(code_tag.get(zc_, "?"), n_) for n_, zc_ in sorted(biggest, reverse=True)[:12]]

    # Stray pixels (user): every one-colour patch of sea (4-connected) under
    # min_region_px -- a zone fragment, or plain sea left between two zones
    # that don't quite meet -- takes the colour it borders most, a zone
    # before plain sea. By colour, so merged territories count with their
    # parent. (The old sieve sized all of a zone's pieces in one connected
    # sea area together: a 3 px fragment beside its own zone survived.)
    sieved = 0
    if min_region_px > 1:
        sieved = _clean_strays(lab, id_code, water, min_region_px)
    report["sieved_px"] = sieved
    report["exclaves_left"] = len(_exclaves(lab, id_code, code, land))  # self-check: 0

    zone = lab > 0
    report["zone_px"] = int(zone.sum())
    report["countries"] = int(len(np.unique(lab[zone])))
    report["seconds"] = round(time.time() - t0, 2)
    return lab, tags, report
