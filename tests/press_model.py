"""Hidden synthetic press: spectral Yule-Nielsen modified Neugebauer with dot gain that depends on the other inks, ink trapping and internal-reflection (Saunderson) flare.
Inputs are CMYK fractions in [0,1]; output is CIE Lab relative to the unprinted paper (paper = 100,0,0)."""
import numpy as np

WL = np.arange(400.0, 701.0, 10.0)
N_YN = 2.1
TRAP = np.array([1.0, 0.90, 0.82, 0.76])           # exponent applied to the 1st..4th ink laid down
ORDER = [3, 0, 1, 2]                                # print order: K, C, M, Y (indices into C,M,Y,K)
GAIN = np.array([0.17, 0.15, 0.13, 0.20])
SKEW = np.array([0.15, -0.10, 0.20, -0.15])
BETA = np.array([1.10, -0.90, 1.20, 0.80])         # dot gain of an ink changes with the coverage of the other inks
KI = 0.10                                           # internal reflectance of the paper/ink surface


def _g(l, mu, s1, s2):
    s = np.where(l < mu, s1, s2)
    return np.exp(-0.5 * ((l - mu) / s) ** 2)


def cmfs():
    x = 1.056 * _g(WL, 599.8, 37.9, 31.0) + 0.362 * _g(WL, 442.0, 16.0, 26.7) - 0.065 * _g(WL, 501.1, 20.4, 26.2)
    y = 0.821 * _g(WL, 568.8, 46.9, 40.5) + 0.286 * _g(WL, 530.9, 16.3, 31.1)
    z = 1.217 * _g(WL, 437.0, 11.8, 36.0) + 0.681 * _g(WL, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], 1)


def illuminant():
    l = WL * 1e-9; T = 5003.0
    return 1.0 / (l ** 5 * (np.exp(1.4388e-2 / (l * T)) - 1.0))


def _sig(x): return 1.0 / (1.0 + np.exp(-x))


def _build():
    paper = 0.88 - 0.07 * np.exp(-((WL - 420) / 45) ** 2) + 0.00012 * (WL - 550)
    dens = np.zeros((4, WL.size))
    dens[0] = 1.20 * _sig((WL - 572) / 17) + 0.12 + 0.16 * np.exp(-((WL - 440) / 55) ** 2)
    dens[1] = 1.30 * np.exp(-((WL - 545) / 44) ** 2) + 0.10 * np.exp(-((WL - 430) / 40) ** 2) + 0.05
    dens[2] = 1.10 * (1 - _sig((WL - 503) / 14)) + 0.04 * np.exp(-((WL - 550) / 50) ** 2) + 0.03
    dens[3] = 1.85 + 0.0004 * (700 - WL)
    prim = np.zeros((16, WL.size))
    for s in range(16):
        r = paper.copy(); rank = 0
        for ink in ORDER:
            if s >> ink & 1:
                r = r * 10.0 ** (-dens[ink] * TRAP[rank]); rank += 1
        prim[s] = r
    return paper, prim


PAPER, PRIM = _build()
_CM = cmfs(); _E = illuminant()
_K = 1.0 / np.sum(_E * _CM[:, 1])
_PW = _K * (_E * (PAPER / (1.0 - KI * PAPER))) @ _CM            # XYZ of the paper
_PRIM_YN = PRIM ** (1.0 / N_YN)


def _lab(xyz):
    t = xyz / _PW
    f = np.where(t > 216 / 24389, np.cbrt(t), (24389 / 27 * t + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def dot_area(c):
    c = np.clip(c, 0.0, 1.0)
    a0 = np.clip(c + 4 * GAIN * c * (1 - c) * (1 + SKEW * (1 - 2 * c)), 0.0, 1.0)
    others = (a0.sum(axis=-1, keepdims=True) - a0) / 3.0
    return np.clip(a0 + BETA * 2 * a0 * (1 - a0) * others, 0.0, 1.0)


def press(cmyk):
    """cmyk: (N,4) fractions -> Lab (N,3)."""
    cmyk = np.atleast_2d(np.asarray(cmyk, float))
    if not np.isfinite(cmyk).all():
        raise ValueError("non-finite ink values")
    a = dot_area(cmyk)
    w = np.ones((cmyk.shape[0], 16))
    for s in range(16):
        for ink in range(4):
            w[:, s] *= a[:, ink] if (s >> ink & 1) else (1 - a[:, ink])
    r = (w @ _PRIM_YN) ** N_YN
    r = r / (1.0 - KI * r)
    return _lab(_K * (r * _E) @ _CM)


def delta_e2000(l1, l2):
    l1 = np.atleast_2d(np.asarray(l1, float)); l2 = np.atleast_2d(np.asarray(l2, float))
    if not (np.isfinite(l1).all() and np.isfinite(l2).all()):
        raise ValueError("non-finite colour values")
    L1, a1, b1 = l1.T; L2, a2, b2 = l2.T
    C1 = np.hypot(a1, b1); C2 = np.hypot(a2, b2); Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p = (1 + G) * a1; a2p = (1 + G) * a2
    C1p = np.hypot(a1p, b1); C2p = np.hypot(a2p, b2)
    h1 = np.degrees(np.arctan2(b1, a1p)) % 360; h2 = np.degrees(np.arctan2(b2, a2p)) % 360
    dL = L2 - L1; dC = C2p - C1p
    dh = h2 - h1; dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh)); dh = np.where(C1p * C2p == 0, 0, dh)
    dH = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lb = (L1 + L2) / 2; Cbp = (C1p + C2p) / 2
    hs = h1 + h2
    hb = np.where(np.abs(h1 - h2) > 180, (hs + 360) / 2, hs / 2); hb = np.where(C1p * C2p == 0, hs, hb)
    T = 1 - 0.17 * np.cos(np.radians(hb - 30)) + 0.24 * np.cos(np.radians(2 * hb)) + 0.32 * np.cos(np.radians(3 * hb + 6)) - 0.20 * np.cos(np.radians(4 * hb - 63))
    dth = 30 * np.exp(-(((hb - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lb - 50) ** 2 / np.sqrt(20 + (Lb - 50) ** 2)
    Sc = 1 + 0.045 * Cbp; Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    out = np.sqrt((dL / Sl) ** 2 + (dC / Sc) ** 2 + (dH / Sh) ** 2 + Rt * (dC / Sc) * (dH / Sh))
    if not np.isfinite(out).all():
        raise ValueError("non-finite colour difference")
    return out
