#!/usr/bin/env python3
"""Reference solution: robust Yule-Nielsen Neugebauer fit with per-zone ink film factors plus a GP residual correction as the forward model (evaluated at the sheet-average film), black-generation-aware and gamut-clipping inversion, hand-written ICC v4 profile."""
import struct
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

TAC_LIMIT = 3.0
LG, LT = 0.55, 1025


# ---------------------------------------------------------------- colour helpers (Lab is relative to the paper)
def lab_to_xyz(lab):
    L, a, b = lab.T
    fy = (L + 16) / 116; fx = fy + a / 500; fz = fy - b / 200
    f = lambda t: np.where(t ** 3 > 216 / 24389, t ** 3, (116 * t - 16) / (24389 / 27))
    return np.stack([f(fx), f(fy), f(fz)], 1)


def xyz_to_lab(xyz):
    t = np.clip(xyz, 1e-9, None)
    f = np.where(t > 216 / 24389, np.cbrt(t), (24389 / 27 * t + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def de2000(l1, l2):
    L1, a1, b1 = l1.T; L2, a2, b2 = l2.T
    C1 = np.hypot(a1, b1); C2 = np.hypot(a2, b2); Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p = (1 + G) * a1; a2p = (1 + G) * a2
    C1p = np.hypot(a1p, b1); C2p = np.hypot(a2p, b2)
    h1 = np.degrees(np.arctan2(b1, a1p)) % 360; h2 = np.degrees(np.arctan2(b2, a2p)) % 360
    dL = L2 - L1; dC = C2p - C1p
    dh = h2 - h1; dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh)); dh = np.where(C1p * C2p == 0, 0, dh)
    dH = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lb = (L1 + L2) / 2; Cbp = (C1p + C2p) / 2; hs = h1 + h2
    hb = np.where(np.abs(h1 - h2) > 180, (hs + 360) / 2, hs / 2); hb = np.where(C1p * C2p == 0, hs, hb)
    T = 1 - 0.17 * np.cos(np.radians(hb - 30)) + 0.24 * np.cos(np.radians(2 * hb)) + 0.32 * np.cos(np.radians(3 * hb + 6)) - 0.20 * np.cos(np.radians(4 * hb - 63))
    dth = 30 * np.exp(-(((hb - 275) / 25) ** 2)); Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lb - 50) ** 2 / np.sqrt(20 + (Lb - 50) ** 2); Sc = 1 + 0.045 * Cbp; Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    return np.sqrt((dL / Sl) ** 2 + (dC / Sc) ** 2 + (dH / Sh) ** 2 + Rt * (dC / Sc) * (dH / Sh))


def de00_components(l1, l2):
    """Three residuals whose squared norm is exactly CIEDE2000(l1, l2)^2 (reference l1, N x 3), so a least-squares solver minimises the real colour difference."""
    L1, a1, b1 = l1.T; L2, a2, b2 = l2.T
    C1 = np.hypot(a1, b1); C2 = np.hypot(a2, b2); Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p = (1 + G) * a1; a2p = (1 + G) * a2
    C1p = np.hypot(a1p, b1); C2p = np.hypot(a2p, b2)
    h1 = np.degrees(np.arctan2(b1, a1p)) % 360; h2 = np.degrees(np.arctan2(b2, a2p)) % 360
    dh = h2 - h1; dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh)); dh = np.where(C1p * C2p == 0, 0, dh)
    dH = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lb = (L1 + L2) / 2; Cbp = (C1p + C2p) / 2; hs = h1 + h2
    hb = np.where(np.abs(h1 - h2) > 180, (hs + 360) / 2, hs / 2); hb = np.where(C1p * C2p == 0, hs, hb)
    T = 1 - 0.17 * np.cos(np.radians(hb - 30)) + 0.24 * np.cos(np.radians(2 * hb)) + 0.32 * np.cos(np.radians(3 * hb + 6)) - 0.20 * np.cos(np.radians(4 * hb - 63))
    Sl = 1 + 0.015 * (Lb - 50) ** 2 / np.sqrt(20 + (Lb - 50) ** 2); Sc = 1 + 0.045 * Cbp; Sh = 1 + 0.015 * Cbp * T
    dth = 30 * np.exp(-(((hb - 275) / 25) ** 2)); Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7)); Rt = -np.sin(np.radians(2 * dth)) * Rc
    y = (C2p - C1p) / Sc; z = dH / Sh      # dE^2 = x^2 + y^2 + z^2 + Rt*y*z = x^2 + (y + Rt*z/2)^2 + (1 - Rt^2/4)*z^2
    return np.stack([(L2 - L1) / Sl, y + Rt * z / 2, z * np.sqrt(np.maximum(1 - Rt ** 2 / 4, 1e-4))], 1)


# ---------------------------------------------------------------- forward model
# Colour-space Yule-Nielsen modified Neugebauer model.  Beyond the textbook model it carries the terms found from the
# residuals of the textbook fit: a dot gain that depends on how much of the other inks is printed (one coefficient
# per ink), an internal-reflection flare t/(1-k t) per XYZ channel, and the uneven ink keys: in each zone every ink
# has a film factor f (mean 1 over the zones), and since optical density is proportional to film thickness the
# transmittance of that ink in every overprint is raised to the power f.  Fitted with a robust loss; patches that
# still miss are dropped; a GP on the Lab residual absorbs what is left.  The profile is the model at f = 1.
BITS = np.array([[s >> i & 1 for i in range(4)] for s in range(16)])
NB = 61                                                     # parameters before the zone film factors


def dot_area(c, g, sk, be):
    c = np.clip(c, 0, 1)
    a0 = np.clip(c + 4 * g * c * (1 - c) * (1 + sk * (1 - 2 * c)), 0, 1)
    oth = (a0.sum(1, keepdims=True) - a0) / 3.0
    return np.clip(a0 + be * 2 * a0 * (1 - a0) * oth, 0, 1)


def films(u, nz):
    u = u.reshape(4, nz)
    return (1 + u - u.mean(1, keepdims=True)).T             # (zones, inks), each ink averages exactly 1


def yn_model(theta, X, zone=None, nz=1):
    prim = np.ones((16, 3)); prim[1:] = theta[:45].reshape(15, 3); prim = np.clip(prim, 1e-6, None)
    n = theta[45]; a = dot_area(X, theta[46:50], theta[50:54], theta[54:58]); k = theta[58:61]
    w = np.ones((len(X), 16))
    for s in range(16):
        for i in range(4):
            w[:, s] *= a[:, i] if BITS[s, i] else 1 - a[:, i]
    if zone is None:                                          # sheet-average press
        xyz = (w @ prim ** (1 / n)) ** n
    else:
        f = films(theta[NB:], nz)[zone]
        P = prim[None] * np.exp(np.einsum('si,ni,ic->nsc', BITS.astype(float), f - 1, np.log(prim[[1, 2, 4, 8]])))
        xyz = np.einsum('ns,nsc->nc', w, P ** (1 / n)) ** n
    return xyz_to_lab(xyz * (1 - k) / (1 - k * xyz))


def fit_yn(X, lab, zone, nz, th_init=None, scale=0.8):
    prim0 = np.full((16, 3), 0.5)
    for s in range(16):
        m = np.abs(X - BITS[s]).max(1) < 1e-9
        if m.any():
            prim0[s] = lab_to_xyz(lab[m][:1])[0]
    prim0 = np.clip(prim0, 1e-3, None)
    th0 = np.concatenate([prim0[1:].ravel(), [2.0], [0.15] * 4, [0.0] * 4, [0.0] * 4, [0.1] * 3, np.zeros(4 * nz)])
    lo = np.concatenate([np.full(45, 1e-4), [1.0], [0.0] * 4, [-1.0] * 4, [-3.0] * 4, [-0.5] * 3, np.full(4 * nz, -0.3)])
    hi = np.concatenate([np.full(45, 1.5), [6.0], [0.4] * 4, [1.0] * 4, [3.0] * 4, [0.9] * 3, np.full(4 * nz, 0.3)])
    if th_init is not None:
        th0 = np.clip(th_init, lo + 1e-9, hi - 1e-9)
    sol = least_squares(lambda th: (yn_model(th, X, zone, nz) - lab).ravel(), th0, bounds=(lo, hi), loss="soft_l1",
                        f_scale=scale, x_scale="jac", max_nfev=400)
    return sol.x


def fit_forward(cmyk, lab, zone, reject=1.6):
    nz = zone.max() + 1
    keep = np.ones(len(cmyk), bool); th = None
    for _ in range(8):
        th = fit_yn(cmyk[keep], lab[keep], zone[keep], nz, th)
        new = de2000(yn_model(th, cmyk, zone, nz), lab) < reject
        if (new == keep).all():
            break
        keep = new
    resid = lab[keep] - yn_model(th, cmyk[keep], zone[keep], nz)
    gp = GaussianProcessRegressor(ConstantKernel(0.1) * RBF([0.5] * 4) + WhiteKernel(0.05), n_restarts_optimizer=2,
                                  random_state=0).fit(cmyk[keep], resid)
    f = lambda Q: yn_model(th, np.clip(Q, 0, 1)) + gp.predict(np.clip(Q, 0, 1))
    return f, keep


# ---------------------------------------------------------------- inversion
def k_pref(lab, lk=65.0, p=1.2, cref=60.0):
    dark = np.clip((lk - lab[:, 0]) / lk, 0, 1) ** p
    return dark * (1 - np.clip(np.hypot(lab[:, 1], lab[:, 2]) / cref, 0, 1))


def lm_invert(f, target, kp, x0, wk=6.0, wt=300.0, iters=40):
    x = x0.copy(); n = len(x); eps = 1e-3

    def res(x):
        r = np.empty((len(x), 5))
        r[:, :3] = de00_components(target, f(x)); r[:, 3] = wk * (x[:, 3] - kp); r[:, 4] = wt * np.maximum(0.0, x.sum(1) - TAC_LIMIT)
        return r
    r = res(x); cost = (r ** 2).sum(1); lam = np.ones(n)
    for _ in range(iters):
        J = np.empty((n, 5, 4))
        for i in range(4):
            xp = x.copy(); step = np.where(x[:, i] + eps <= 1.0, eps, -eps); xp[:, i] += step
            J[:, :, i] = (res(xp) - r) / step[:, None]
        A = np.einsum('nri,nrj->nij', J, J) + lam[:, None, None] * np.eye(4)
        dx = -np.linalg.solve(A, np.einsum('nri,nr->ni', J, r)[:, :, None])[:, :, 0]
        xn = np.clip(x + dx, 0, 1); rn = res(xn); cn = (rn ** 2).sum(1); ok = cn < cost
        x[ok] = xn[ok]; r[ok] = rn[ok]; cost[ok] = cn[ok]
        lam = np.where(ok, np.maximum(lam * 0.4, 1e-6), np.minimum(lam * 4, 1e6))
    return x


def build_tables(f, ga=17, gb=33, nsamp=250000, seed=0):
    rng = np.random.default_rng(seed)
    ax = np.linspace(0, 1, ga)
    nodes = np.stack(np.meshgrid(ax, ax, ax, ax, indexing='ij'), -1).reshape(-1, 4)
    a2b = np.concatenate([f(nodes[i:i + 20000]) for i in range(0, len(nodes), 20000)]).reshape(ga, ga, ga, ga, 3)
    cand = rng.uniform(0, 1, (nsamp, 4)); cand = np.concatenate([cand[cand.sum(1) <= TAC_LIMIT], nodes[nodes.sum(1) <= TAC_LIMIT]])
    clab = np.concatenate([f(cand[i:i + 20000]) for i in range(0, len(cand), 20000)])
    tree = cKDTree(clab)
    xs = np.linspace(0, 1, LT); tab = xs ** LG                       # L B-curve: more grid nodes in the shadows
    L = 100 * np.interp(np.linspace(0, 1, gb), tab, xs); A = np.linspace(-128, 127, gb)
    grid = np.stack(np.meshgrid(L, A, A, indexing='ij'), -1).reshape(-1, 3)
    kp = k_pref(grid)
    k = 96
    _, idx = tree.query(grid, k=k)
    de = np.linalg.norm(de00_components(np.repeat(grid[:, None], k, 1).reshape(-1, 3), clab[idx].reshape(-1, 3)), axis=1).reshape(-1, k)
    score = de + 30.0 * np.abs(cand[idx][:, :, 3] - kp[:, None])
    x0 = cand[idx[np.arange(len(grid)), score.argmin(1)]]
    return a2b, lm_invert(f, grid, kp, x0).reshape(gb, gb, gb, 4)


# ---------------------------------------------------------------- ICC v4 writer
def s15(v): return struct.pack(">i", int(round(v * 65536)))
def pad4(b): return b + bytes(-len(b) % 4)
def mluc(t):
    s = t.encode("utf-16-be"); return b"mluc" + bytes(4) + struct.pack(">II", 1, 12) + b"enUS" + struct.pack(">II", len(s), 28) + s
def identity_curves(n): return b"".join(b"curv" + bytes(4) + struct.pack(">I", 0) for _ in range(n))   # 12 bytes each, already aligned


def table_curve(vals):
    d = np.round(np.clip(vals, 0, 1) * 65535).astype(">u2").tobytes()
    return pad4(b"curv" + bytes(4) + struct.pack(">I", len(vals)) + d)


def lut(sig, nin, nout, dims, data16):
    nb, na = (nout, nin) if sig == b"mAB " else (nin, nout)
    cb, ca = identity_curves(nb), identity_curves(na)
    if sig == b"mBA ":
        cb = table_curve(np.linspace(0, 1, LT) ** LG) + identity_curves(2)
    clut = pad4(bytes(dims) + bytes(16 - len(dims)) + bytes([2, 0, 0, 0]) + data16.astype(">u2").tobytes())
    o_b = 32; o_c = o_b + len(cb); o_a = o_c + len(clut)
    return sig + bytes(4) + bytes([nin, nout, 0, 0]) + struct.pack(">IIIII", o_b, 0, 0, o_c, o_a) + cb + clut + ca


def write_icc(a2b_lab, b2a_cmyk):
    enc = np.stack([a2b_lab[..., 0] / 100, (a2b_lab[..., 1] + 128) / 255, (a2b_lab[..., 2] + 128) / 255], -1)
    tags = [(b"A2B1", lut(b"mAB ", 4, 3, a2b_lab.shape[:4], np.round(np.clip(enc, 0, 1) * 65535))),
            (b"B2A1", lut(b"mBA ", 3, 4, b2a_cmyk.shape[:3], np.round(np.clip(b2a_cmyk, 0, 1) * 65535))),
            (b"wtpt", b"XYZ " + bytes(4) + s15(0.9642) + s15(1.0) + s15(0.8249)),
            (b"desc", mluc("offset press profile")), (b"cprt", mluc("no copyright"))]
    off = 128 + 4 + 12 * len(tags); table = struct.pack(">I", len(tags)); body = b""
    for sig, d in tags:
        d = pad4(d); table += sig + struct.pack(">II", off + len(body), len(d)); body += d
    h = bytearray(128)
    h[8:12] = bytes([4, 0x20, 0, 0]); h[12:16] = b"prtr"; h[16:20] = b"CMYK"; h[20:24] = b"Lab "; h[36:40] = b"acsp"
    h[64:68] = struct.pack(">I", 1); h[68:80] = s15(0.9642) + s15(1.0) + s15(0.8249)
    out = bytes(h) + table + body
    return struct.pack(">I", len(out)) + out[4:]


def main():
    d = np.genfromtxt("/app/chart.csv", delimiter=",", names=True)
    cmyk = np.stack([d["C"], d["M"], d["Y"], d["K"]], 1) / 100.0
    lab = np.stack([d["L"], d["a"], d["b"]], 1)
    zone = d["zone"].astype(int) - 1
    f, keep = fit_forward(cmyk, lab, zone)
    print("patches treated as misreads:", int((~keep).sum()))
    a2b, b2a = build_tables(f)
    open("/app/press.icc", "wb").write(write_icc(a2b, b2a))


if __name__ == "__main__":
    main()
