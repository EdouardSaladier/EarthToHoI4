"""Game post-processing (gfx/FX/restorescene.shader), defines and the sky cubemap."""
import os
import re
import struct

import numpy as np
from PIL import Image

from . import pdx

# ------------------------------------------------------------------ defines
_DEF_RE = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*(-?[0-9.]+)", re.M)
_OVR_RE = re.compile(r"NDefines\.(\w+)\.([A-Z0-9_]+)\s*=\s*(-?[0-9.]+)")


def load_defines(vfs):
    """Numeric defines: vanilla 00_*.lua values, overridden by any other file (mods)."""
    out = {}
    files = vfs.list_dir("common/defines", (".lua",))
    for fn, path in files.items():
        text = pdx.read_text(path)
        if fn.startswith("00_"):
            for k, v in _DEF_RE.findall(text):
                try:
                    out.setdefault(k, float(v))
                except ValueError:
                    pass
    for fn, path in files.items():
        text = pdx.read_text(path)
        for _, k, v in _OVR_RE.findall(text):
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


# ------------------------------------------------------------- tonemapping
def _uncharted2(x):
    A, B, C, D, E, F = 0.22, 0.30, 0.10, 0.20, 0.01, 0.30
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F


_WHITE = 1.0 / _uncharted2(np.float32(11.2))


class PostProcess:
    """HDR exposure -> Uncharted2 tonemap -> colour cube LUT -> Levels, as in restorescene.shader."""

    LEVELS = (0.04, 0.8)

    def __init__(self, vfs, lut_rel="gfx/world/colorcorrection_farrange.tga", middle_grey=0.55,
                 avg_luminance=0.25):
        self.exposure = middle_grey / avg_luminance
        self.lut = None
        p = vfs.find(lut_rel) if lut_rel else None
        if p:
            a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            n = a.shape[0]
            if a.shape[1] == n * n:
                # cube[b, g, r] = colour
                self.lut = a.reshape(n, n, n, 3).transpose(1, 0, 2, 3).copy()
                self.n = n

    def tonemap(self, x):
        c = _uncharted2(np.maximum(x * self.exposure, 0.0)) * _WHITE
        return np.power(np.clip(c, 0.0, 1.0), 1.0 / 2.2)

    def lut_apply(self, c):
        if self.lut is None:
            return c
        n = self.n
        scale = (n - 1.0) / n
        # equivalent to the shader's offsets: index = colour * (n - 1)
        f = np.clip(c, 0.0, 1.0) * (n - 1)
        r = f[..., 0]
        g = f[..., 1]
        b = f[..., 2]
        r0 = np.floor(r).astype(np.int32)
        g0 = np.floor(g).astype(np.int32)
        r1 = np.minimum(r0 + 1, n - 1)
        g1 = np.minimum(g0 + 1, n - 1)
        wr = (r - r0)[..., None]
        wg = (g - g0)[..., None]
        bz = np.clip(c[..., 2], 0, 1) * scale * n
        b0 = np.floor(bz).astype(np.int32)
        b0 = np.clip(b0, 0, n - 1)
        b1 = np.minimum(b0 + 1, n - 1)
        wb = (bz - b0)[..., None]
        del b, f
        L = self.lut

        def bilerp(bi):
            c00 = L[bi, g0, r0]
            c01 = L[bi, g0, r1]
            c10 = L[bi, g1, r0]
            c11 = L[bi, g1, r1]
            top = c00 + (c01 - c00) * wr
            bot = c10 + (c11 - c10) * wr
            return top + (bot - top) * wg

        lo = bilerp(b0)
        hi = bilerp(b1)
        return lo + (hi - lo) * wb

    def levels(self, c):
        lo, hi = self.LEVELS
        return np.clip((c - lo) / (hi - lo), 0.0, 1.0)

    def __call__(self, x):
        """Linear scene colour (as the shaders output it) -> final display colour 0..1."""
        return self.levels(self.lut_apply(self.tonemap(x)))


# ---------------------------------------------------------------- cubemap
class CubeMap:
    """Uncompressed DDS cubemap (map/terrain/reflection.dds), top mip of the six faces."""

    def __init__(self, path):
        b = open(path, "rb").read()
        h, w = struct.unpack("<2I", b[12:20])
        mips = max(1, struct.unpack("<I", b[28:32])[0])
        bits = struct.unpack("<I", b[88:92])[0]
        rmask = struct.unpack("<I", b[92:96])[0]
        if bits != 32:
            raise ValueError("unsupported cubemap format")
        off = 128
        faces = []
        face_bytes = sum(max(1, w >> i) * max(1, h >> i) * 4 for i in range(mips))
        for _ in range(6):
            a = np.frombuffer(b, np.uint8, w * h * 4, off).reshape(h, w, 4)
            rgb = a[..., [2, 1, 0]] if rmask == 0xFF0000 else a[..., :3]
            faces.append(rgb.astype(np.float32) / 255.0)
            off += face_bytes
        self.faces = np.stack(faces)  # +X -X +Y -Y +Z -Z
        self.size = w

    def sample(self, d):
        """d: (..., 3) directions (x, y up, z). Bilinear inside each face."""
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
        face = np.where((ax >= ay) & (ax >= az), np.where(x > 0, 0, 1),
                        np.where(ay >= az, np.where(y > 0, 2, 3), np.where(z > 0, 4, 5)))
        ma = np.maximum(np.maximum(ax, ay), az) + 1e-9
        sc = np.select([face == 0, face == 1, face == 2, face == 3, face == 4], [-z, z, x, x, x], -x)
        tc = np.select([face == 0, face == 1, face == 2, face == 3, face == 4], [-y, -y, z, -z, -y], -y)
        n = self.size
        u = (sc / ma + 1) * 0.5 * n - 0.5
        v = (tc / ma + 1) * 0.5 * n - 0.5
        u0 = np.clip(np.floor(u).astype(np.int32), 0, n - 1)
        v0 = np.clip(np.floor(v).astype(np.int32), 0, n - 1)
        u1 = np.minimum(u0 + 1, n - 1)
        v1 = np.minimum(v0 + 1, n - 1)
        wu = np.clip(u - u0, 0, 1)[..., None]
        wv = np.clip(v - v0, 0, 1)[..., None]
        F = self.faces
        top = F[face, v0, u0] + (F[face, v0, u1] - F[face, v0, u0]) * wu
        bot = F[face, v1, u0] + (F[face, v1, u1] - F[face, v1, u0]) * wu
        return top + (bot - top) * wv
