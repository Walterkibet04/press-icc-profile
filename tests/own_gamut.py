"""Closest colour the profile's own A2B1 can reach within the 300% ink limit, for out-of-gamut targets.
Dense ink cloud through A2B1, a CIEDE2000 scan of every target against the cloud, then a projected Gauss-Newton
refinement on the interpolated table. Deterministic (fixed seed)."""
import numpy as np

import press_model as P

TAC = 3.0


def project(y, tac=TAC):
    """Euclidean projection onto {0 <= x <= 1, sum(x) <= tac}, row-wise."""
    x = np.clip(y, 0.0, 1.0)
    over = x.sum(1) > tac
    if over.any():
        yo = y[over]
        lo = np.zeros(len(yo)); hi = np.full(len(yo), yo.max(1).clip(0) + 1.0)
        for _ in range(50):
            mid = (lo + hi) / 2
            s = np.clip(yo - mid[:, None], 0.0, 1.0).sum(1)
            lo = np.where(s > tac, mid, lo); hi = np.where(s > tac, hi, mid)
        x[over] = np.clip(yo - hi[:, None], 0.0, 1.0)
    return x


def de_components(ref, lab):
    """Three residuals whose squared norm is CIEDE2000(ref, lab)^2."""
    L1, a1, b1 = ref.T; L2, a2, b2 = lab.T
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
    y = (C2p - C1p) / Sc; z = dH / Sh
    return np.stack([(L2 - L1) / Sl, y + Rt * z / 2, z * np.sqrt(np.maximum(1 - Rt ** 2 / 4, 1e-4))], 1)


def _refine(f, target, x, iters=40, eps=2e-3):
    n = len(x)
    r = de_components(target, f(x)); cost = (r ** 2).sum(1); lam = np.full(n, 1e-2)
    for _ in range(iters):
        J = np.empty((n, 3, 4))
        for i in range(4):
            step = np.where(x[:, i] + eps <= 1.0, eps, -eps)
            xp = x.copy(); xp[:, i] += step
            J[:, :, i] = (de_components(target, f(xp)) - r) / step[:, None]
        A = np.einsum('nri,nrj->nij', J, J) + lam[:, None, None] * np.eye(4)
        dx = -np.linalg.solve(A, np.einsum('nri,nr->ni', J, r)[:, :, None])[:, :, 0]
        xn = project(x + dx); rn = de_components(target, f(xn)); cn = (rn ** 2).sum(1); ok = cn < cost
        x[ok] = xn[ok]; r[ok] = rn[ok]; cost[ok] = cn[ok]
        lam = np.where(ok, np.maximum(lam * 0.3, 1e-7), np.minimum(lam * 10, 1e7))
    return x, np.sqrt(cost)


def best_reachable(a2b, targets, n_cloud=200000, starts=4, seed=12345, chunk=25):
    """a2b: callable cmyk (N,4) -> Lab (N,3). Returns (best CIEDE2000 distance, ink) per target.
    Every target is scored by CIEDE2000 against the whole ink cloud (CIEDE2000 is far from Euclidean in Lab for
    colours well outside the gamut), then the best few starts are refined on the interpolated table."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(0, 1, (n_cloud, 4))
    c = np.concatenate([c[c.sum(1) <= TAC], project(c[c.sum(1) > TAC][: n_cloud // 3])])
    ax = np.linspace(0, 1, 9)
    g = np.stack(np.meshgrid(ax, ax, ax, ax, indexing='ij'), -1).reshape(-1, 4)
    c = np.concatenate([c, project(g)])
    lab = np.concatenate([a2b(c[i:i + 100000]) for i in range(0, len(c), 100000)])
    nt = len(targets); top = np.zeros((nt, starts), int)
    for i in range(0, nt, chunk):
        t = targets[i:i + chunk]
        de = P.delta_e2000(np.repeat(t, len(lab), 0), np.tile(lab, (len(t), 1))).reshape(len(t), -1)
        top[i:i + chunk] = np.argpartition(de, starts, 1)[:, :starts]
    best_d = np.full(nt, np.inf); best_x = np.zeros((nt, 4))
    for s in range(starts):
        x, d = _refine(a2b, targets, c[top[:, s]].copy())
        better = d < best_d
        best_d[better] = d[better]; best_x[better] = x[better]
    return best_d, best_x
