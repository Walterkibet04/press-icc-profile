"""Strict reader used by the verifier. Only the structures described in the task instruction are accepted."""
import struct
import numpy as np


class ICCError(Exception):
    pass


def _u32(b, o):
    if o < 0 or o + 4 > len(b):
        raise ICCError("read past end of file")
    return struct.unpack(">I", b[o:o + 4])[0]


def _curve(b, off):
    if b[off:off + 4] != b"curv":
        raise ICCError("curves must use the 'curv' type")
    n = _u32(b, off + 8)
    if n == 0:
        return ("id", None), 12
    if n == 1:
        if off + 14 > len(b):
            raise ICCError("truncated curve")
        return ("gamma", struct.unpack(">H", b[off + 12:off + 14])[0] / 256.0), 14
    if n > 4096 or off + 12 + 2 * n > len(b):
        raise ICCError("bad curve table")
    return ("table", np.frombuffer(b, ">u2", n, off + 12).astype(float) / 65535.0), 12 + 2 * n


def _apply_curve(c, x):
    kind, v = c
    x = np.clip(x, 0.0, 1.0)
    if kind == "id":
        return x
    if kind == "gamma":
        return x ** v
    return np.interp(x, np.linspace(0.0, 1.0, v.size), v)


def _curves(b, off, count):
    out = []
    for _ in range(count):
        c, size = _curve(b, off)
        out.append(c)
        off += (size + 3) & ~3
    return out


def _interp(grid, x):
    """n-linear interpolation. grid has shape (g0,..,g_{n-1}, out); x has shape (N, n) in [0,1]."""
    n = x.shape[1]
    dims = grid.shape[:n]
    pos = np.clip(x, 0.0, 1.0) * (np.array(dims) - 1)
    i0 = np.minimum(np.floor(pos).astype(int), np.array(dims) - 2)
    f = pos - i0
    out = np.zeros((x.shape[0], grid.shape[-1]))
    for corner in range(1 << n):
        w = np.ones(x.shape[0]); idx = []
        for d in range(n):
            bit = corner >> d & 1
            w = w * (f[:, d] if bit else 1 - f[:, d])
            idx.append(i0[:, d] + bit)
        out += w[:, None] * grid[tuple(idx)]
    return out


class Lut:
    def __init__(self, b, off, size, kind, nin, nout, max_grid):
        t = b[off:off + 4]
        if t != kind:
            raise ICCError("expected a %r tag, found %r" % (kind, t))
        if b[off + 8] != nin or b[off + 9] != nout:
            raise ICCError("%r must have %d inputs and %d outputs" % (kind, nin, nout))
        oB, oMat, oM, oC, oA = (_u32(b, off + 12 + 4 * i) for i in range(5))
        if oMat or oM:
            raise ICCError("matrix and M curves are not supported")
        if not oC or not oB or not oA:
            raise ICCError("A curves, CLUT and B curves are all required")
        self.curve_b = _curves(b, off + oB, nout if kind == b"mAB " else 3)
        self.curve_a = _curves(b, off + oA, nin if kind == b"mAB " else 4)
        c = off + oC
        grid = list(b[c:c + nin])
        if any(g < 2 or g > max_grid for g in grid) or any(b[c + nin:c + 16]):
            raise ICCError("CLUT grid sizes must be between 2 and %d" % max_grid)
        if b[c + 16] != 2:
            raise ICCError("CLUT precision must be 16 bit")
        n = int(np.prod(grid)) * nout
        if c + 20 + 2 * n > len(b):
            raise ICCError("CLUT data truncated")
        self.grid = np.frombuffer(b, ">u2", n, c + 20).astype(float).reshape(*grid, nout) / 65535.0
        self.kind = kind


class Profile:
    def __init__(self, data):
        b = bytes(data)
        if len(b) < 132 or len(b) > 8 * 1024 * 1024:
            raise ICCError("profile size out of range")
        if _u32(b, 0) != len(b):
            raise ICCError("header size field does not match the file size")
        if b[36:40] != b"acsp":
            raise ICCError("missing 'acsp' signature")
        if b[8] != 4:
            raise ICCError("profile version must be 4.x")
        if b[12:16] != b"prtr" or b[16:20] != b"CMYK" or b[20:24] != b"Lab ":
            raise ICCError("header must declare an output profile, CMYK device space, Lab PCS")
        ill = [struct.unpack(">i", b[68 + 4 * i:72 + 4 * i])[0] / 65536.0 for i in range(3)]
        if max(abs(ill[0] - 0.9642), abs(ill[1] - 1.0), abs(ill[2] - 0.8249)) > 0.002:
            raise ICCError("illuminant must be D50")
        n = _u32(b, 128)
        if n > 100:
            raise ICCError("too many tags")
        tags = {}
        for i in range(n):
            sig = b[132 + 12 * i:136 + 12 * i]; off = _u32(b, 136 + 12 * i); size = _u32(b, 140 + 12 * i)
            if off % 4 or off + size > len(b) or off < 132 + 12 * n:
                raise ICCError("bad tag table entry")
            tags[sig] = (off, size)
        for need in (b"A2B1", b"B2A1", b"wtpt", b"desc", b"cprt"):
            if need not in tags:
                raise ICCError("required tag %r is missing" % need)
        wo = tags[b"wtpt"][0]
        if b[wo:wo + 4] != b"XYZ ":
            raise ICCError("wtpt must be of type XYZ")
        for t in (b"desc", b"cprt"):
            if b[tags[t][0]:tags[t][0] + 4] != b"mluc":
                raise ICCError("%r must be of type mluc" % t)
        self.a2b_lut = Lut(b, tags[b"A2B1"][0], tags[b"A2B1"][1], b"mAB ", 4, 3, 17)
        self.b2a_lut = Lut(b, tags[b"B2A1"][0], tags[b"B2A1"][1], b"mBA ", 3, 4, 33)

    def a2b(self, cmyk):
        """cmyk fractions (N,4) -> Lab (N,3). A curves, 4-D n-linear CLUT, B curves, v4 Lab decoding."""
        x = np.stack([_apply_curve(c, v) for c, v in zip(self.a2b_lut.curve_a, np.asarray(cmyk, float).T)], 1)
        y = _interp(self.a2b_lut.grid, x)
        y = np.stack([_apply_curve(c, v) for c, v in zip(self.a2b_lut.curve_b, y.T)], 1)
        return np.stack([y[:, 0] * 100.0, y[:, 1] * 255.0 - 128.0, y[:, 2] * 255.0 - 128.0], 1)

    def b2a(self, lab):
        """Lab (N,3) -> cmyk fractions (N,4). v4 Lab encoding, B curves, trilinear CLUT, A curves."""
        lab = np.asarray(lab, float)
        x = np.stack([lab[:, 0] / 100.0, (lab[:, 1] + 128.0) / 255.0, (lab[:, 2] + 128.0) / 255.0], 1)
        x = np.stack([_apply_curve(c, v) for c, v in zip(self.b2a_lut.curve_b, x.T)], 1)
        y = _interp(self.b2a_lut.grid, x)
        return np.stack([_apply_curve(c, v) for c, v in zip(self.b2a_lut.curve_a, y.T)], 1)
