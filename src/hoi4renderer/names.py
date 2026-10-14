"""Curved country names drawn with the game's bitmap map font."""
import math
import os
import re

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from .data import log


class BMFont:
    def __init__(self, fnt_path):
        self.chars = {}
        self.kern = {}
        page_file = None
        with open(fnt_path, "r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                tag, _, rest = line.strip().partition(" ")
                kv = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', rest))
                if tag == "common":
                    self.line_height = int(kv.get("lineHeight", 0))
                    self.base = int(kv.get("base", 0))
                elif tag == "page" and page_file is None:
                    page_file = kv.get("file", "").strip('"')
                elif tag == "char":
                    cid = int(kv["id"])
                    self.chars[cid] = tuple(int(kv[k]) for k in
                                            ("x", "y", "width", "height", "xoffset", "yoffset", "xadvance"))
                elif tag == "kerning":
                    self.kern[(int(kv["first"]), int(kv["second"]))] = int(kv["amount"])
        folder = os.path.dirname(fnt_path)
        candidates = [os.path.join(folder, page_file)] if page_file else []
        stem = os.path.splitext(fnt_path)[0]
        candidates += [stem + ext for ext in (".dds", ".tga", ".png")]
        img = None
        for c in candidates:
            if os.path.isfile(c):
                img = Image.open(c)
                break
        if img is None:
            raise FileNotFoundError("font page for " + fnt_path)
        img = img.convert("RGBA")
        a = np.asarray(img)
        alpha = a[..., 3]
        if alpha.max() == alpha.min():  # font stored in colour channels
            alpha = a[..., :3].max(axis=-1)
        if 0 < alpha.max() < 255:  # some fonts store a reduced alpha range
            alpha = (alpha.astype(np.float32) * (255.0 / alpha.max())).astype(np.uint8)
        self.page = Image.fromarray(alpha)
        h = self.chars.get(ord("H"))
        self.cap = h[3] if h else int(self.base * 0.72)
        self.cap_top = h[5] if h else self.base - self.cap  # yoffset of the cap top

    def glyph(self, ch):
        return self.chars.get(ord(ch))

    def advances(self, text):
        adv = []
        prev = None
        for ch in text:
            g = self.glyph(ch) or self.glyph("?") or (0, 0, 0, 0, 0, 0, self.line_height // 3)
            k = self.kern.get((prev, ord(ch)), 0) if prev is not None else 0
            adv.append(g[6] + k)
            prev = ord(ch)
        return adv


def find_font(vfs, override=None):
    if override:
        return override
    p = vfs.find("gfx/fonts/hoi_mapfont4.fnt")
    if p:
        return p
    return None


# ------------------------------------------------------------------ layout
def _longest_runs(R):
    """Per column of binary image R: (start, length) of the longest vertical run of ones."""
    H, W = R.shape
    run = np.zeros(W, np.int32)
    best = np.zeros(W, np.int32)
    best_end = np.zeros(W, np.int32)
    for y in range(H):
        row = R[y]
        run = np.where(row, run + 1, 0)
        better = run > best
        best = np.where(better, run, best)
        best_end = np.where(better, y, best_end)
    return best_end - best + 1, best


def _fit_label(mask, aspect, angles=range(-75, 76, 5), nb=48, work=360):
    """Find the best curved baseline for a label inside a binary mask (map pixels).

    aspect = text width / cap height at natural spacing.
    Returns a dict describing the curve in mask pixel coordinates, or None.
    """
    h, w = mask.shape
    if mask.sum() < 30:
        return None
    f = min(1.0, work / max(h, w))
    m = mask.astype(np.uint8)
    if f < 1.0:
        m = cv2.resize(m * 255, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA) > 127
        m = m.astype(np.uint8)
    hs, ws = m.shape
    ys, xs = np.nonzero(m)
    if len(xs) < 10:
        return None
    c = np.array([xs.mean() + 0.5, ys.mean() + 0.5])
    D = int(math.ceil(math.hypot(hs, ws))) + 4
    C = D / 2.0
    best = None
    for ang in angles:
        th = math.radians(ang)
        d = np.array([math.cos(th), math.sin(th)])
        n = np.array([-math.sin(th), math.cos(th)])
        M = np.array([[d[0], n[0], c[0] - d[0] * C - n[0] * C],
                      [d[1], n[1], c[1] - d[1] * C - n[1] * C]], np.float64)
        R = cv2.warpAffine(m, M, (D, D), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0)
        cols = np.nonzero(R.any(axis=0))[0]
        if len(cols) < 4:
            continue
        x0c, x1c = cols[0], cols[-1] + 1
        start, length = _longest_runs(R[:, x0c:x1c].astype(bool))
        ncol = x1c - x0c
        nbb = min(nb, ncol)
        edges = np.linspace(0, ncol, nbb + 1).astype(int)
        lo_e = np.zeros(nbb)
        hi_e = np.zeros(nbb)
        empty = np.zeros(nbb, bool)
        for b in range(nbb):
            sl = slice(edges[b], max(edges[b] + 1, edges[b + 1]))
            ln = length[sl]
            if ln.min() <= 0:
                empty[b] = True
                continue
            st = start[sl]
            lo_e[b] = st.max()                # conservative: intersection of the runs
            hi_e[b] = (st + ln).min()
            if hi_e[b] <= lo_e[b]:
                empty[b] = True
        cen = (lo_e + hi_e) * 0.5 - C
        lo_v = lo_e - C
        hi_v = hi_e - C
        tc = x0c + (edges[:-1] + edges[1:]) * 0.5 - C
        bw = ncol / nbb
        wgt = np.where(empty, 0.0, np.maximum(hi_e - lo_e, 1e-3))

        I, J = np.meshgrid(np.arange(nbb), np.arange(nbb + 1), indexing="ij")
        ok = J - I >= 3
        I = I[ok]
        J = J[ok]
        tt = tc - tc.mean()
        S = [np.concatenate([[0], np.cumsum(wgt * tt ** k)]) for k in range(5)]
        Cs = [np.concatenate([[0], np.cumsum(wgt * cen * tt ** k)]) for k in range(3)]
        s_ = [S[k][J] - S[k][I] for k in range(5)]
        c_ = [Cs[k][J] - Cs[k][I] for k in range(3)]
        A = np.stack([np.stack([s_[4], s_[3], s_[2]], -1),
                      np.stack([s_[3], s_[2], s_[1]], -1),
                      np.stack([s_[2], s_[1], s_[0]], -1)], -2)
        B = np.stack([c_[2], c_[1], c_[0]], -1)
        L = (J - I) * bw
        A = A + np.eye(3)[None] * (1e-6 * (1 + np.abs(A).max(axis=(1, 2))))[:, None, None]
        A[:, 0, 0] += s_[0] * (L ** 4) * 0.02  # keep the curve gentle
        try:
            coef = np.linalg.solve(A, B[..., None])[..., 0]
        except np.linalg.LinAlgError:
            continue
        q = coef[:, 0:1] * tt[None] ** 2 + coef[:, 1:2] * tt[None] + coef[:, 2:3]
        idx = np.arange(nbb)[None]
        inwin = (idx >= I[:, None]) & (idx < J[:, None])
        room = np.minimum(hi_v[None] - q, q - lo_v[None])
        room = np.where(empty[None], -1.0, room)
        room = np.where(inwin, room, np.inf)
        half = room.min(axis=1)
        cap = np.minimum(2 * half * 0.5, 0.92 * L / aspect)
        bend = np.abs(coef[:, 0]) * (L * 0.5) ** 2
        # prefer big letters but also long baselines: the game spreads names over most of the country
        score = cap * np.power(L, 0.5) * (1 - 0.008 * max(0, abs(ang) - 10)) * np.where(bend > 0.18 * L, 0.5, 1.0)
        k = int(np.argmax(score))
        if score[k] > 0 and (best is None or score[k] > best["score"]):
            best = dict(score=float(score[k]), cap=float(cap[k]) / f, center=c / f, d=d, n=n,
                        coef=coef[k], tmean=tc.mean(), t0=tc[I[k]] - bw / 2, t1=tc[J[k] - 1] + bw / 2,
                        L=float(L[k]) / f, f=f)
    return best


def _curve_points(fit, t):
    """t in working (downscaled) units -> points in mask pixel coordinates."""
    tt = t - fit["tmean"]
    a, b, c = fit["coef"]
    v = a * tt * tt + b * tt + c
    f = fit["f"]
    return fit["center"][None] + (t[:, None] * fit["d"][None] + v[:, None] * fit["n"][None]) / f


# ---------------------------------------------------------------- drawing
def draw_names(r):
    """Returns an (OH, OW) uint8 alpha mask with all country names."""
    md, o = r.md, r.o
    path = find_font(md.vfs, getattr(o, "font", None))
    if not path:
        log("  no map font found, skipping names")
        return None
    font = BMFont(path)
    OH, OW = r.OH, r.OW
    sx, sy = r.sx, r.sy
    layer = np.zeros((OH, OW), np.uint8)
    owner = r.owner
    min_cap_out = max(9.0, 6.0 * min(sx, 2.5))
    max_cap_map = 0.09 * md.height_px

    for tag, ci in md.tag_index.items():
        name = md.names.get(tag, tag)
        if o.name_case == "upper":
            name = name.upper()
        if not name.strip():
            continue
        m = owner == ci
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = m[y0:y1, x0:x1].astype(np.uint8)
        # close small gaps (rivers, lakes, narrow straits) before choosing the main landmass
        k = max(3, int(round(max(crop.shape) * 0.004)) | 1)
        joined = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        ncc, lab, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
        if ncc <= 1:
            continue
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        comp = ndimage.binary_fill_holes(lab == big)  # names may run over lakes

        adv = font.advances(name)
        natural = sum(adv) / font.cap  # width in cap heights
        fit = _fit_label(comp, natural)
        if fit is None:
            continue
        cap = min(fit["cap"], max_cap_map)
        cap_out = cap * (sx + sy) * 0.5
        if cap_out < min_cap_out:
            continue

        # spread letters along the window like the game does
        f = cap_out / font.cap  # font units -> output px
        nat_out = sum(adv) * f
        L_out = fit["L"] * (sx + sy) * 0.5
        n_gaps = max(1, len(name) - 1)
        span_out = min(L_out * 0.8, nat_out + n_gaps * cap_out * 1.0)
        span_out = max(span_out, nat_out)
        extra = (span_out - nat_out) / n_gaps if len(name) > 1 else 0.0

        # arc-length parametrisation of the baseline (in output pixels)
        tmid = 0.5 * (fit["t0"] + fit["t1"])
        ts = np.linspace(fit["t0"] - fit["L"] * fit["f"], fit["t1"] + fit["L"] * fit["f"], 3000)
        pts = _curve_points(fit, ts)
        pts[:, 0] = (pts[:, 0] + x0) * sx
        pts[:, 1] = (pts[:, 1] + y0) * sy
        seg = np.sqrt((np.diff(pts, axis=0) ** 2).sum(1))
        arc = np.concatenate([[0], np.cumsum(seg)])
        amid = np.interp(tmid, ts, arc)
        start = amid - span_out / 2

        pos = start
        for ch, a in zip(name, adv):
            w = a * f
            centre = pos + w / 2
            pos += w + extra
            g = font.glyph(ch)
            if ch == " " or g is None or g[2] == 0 or g[3] == 0:
                continue
            p = np.array([np.interp(centre, arc, pts[:, 0]), np.interp(centre, arc, pts[:, 1])])
            pa = np.array([np.interp(centre - 2, arc, pts[:, 0]), np.interp(centre - 2, arc, pts[:, 1])])
            pb = np.array([np.interp(centre + 2, arc, pts[:, 0]), np.interp(centre + 2, arc, pts[:, 1])])
            tang = pb - pa
            ang = math.degrees(math.atan2(tang[1], tang[0]))
            _stamp_glyph(layer, font, g, f, p, ang)

    return layer


def _stamp_glyph(layer, font, g, f, p, ang):
    gx, gy, gw, gh, xo, yo, xa = g
    glyph = font.page.crop((gx, gy, gx + gw, gy + gh))
    # pivot: horizontal centre of the advance, vertical centre of capitals
    piv_x = xa / 2.0
    piv_y = font.cap_top + font.cap / 2.0
    # glyph box relative to pivot (font units)
    left = xo - piv_x
    top = yo - piv_y
    R = math.ceil(max(abs(left), abs(left + gw), abs(top), abs(top + gh)) * f) + 3
    size = 2 * R
    gw_o = max(1, int(round(gw * f)))
    gh_o = max(1, int(round(gh * f)))
    gimg = glyph.resize((gw_o, gh_o), Image.LANCZOS)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(gimg, (int(round(R + left * f)), int(round(R + top * f))))
    canvas = canvas.rotate(-ang, resample=Image.BICUBIC, center=(R, R))
    arr = np.asarray(canvas)
    X0 = int(round(p[0] - R))
    Y0 = int(round(p[1] - R))
    H, W = layer.shape
    xs0, ys0 = max(0, X0), max(0, Y0)
    xs1, ys1 = min(W, X0 + size), min(H, Y0 + size)
    if xs0 >= xs1 or ys0 >= ys1:
        return
    sub = arr[ys0 - Y0:ys1 - Y0, xs0 - X0:xs1 - X0]
    np.maximum(layer[ys0:ys1, xs0:xs1], sub, out=layer[ys0:ys1, xs0:xs1])
