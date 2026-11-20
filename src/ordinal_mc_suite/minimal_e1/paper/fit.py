"""Estimators on a CellSet (training data only). Full batch, memory-bounded pair kernels.

RNC   RankNet with per-cell temperatures shrunk to the global one (same objective as
      robust_estimators.ranknet_cell); optional hard basis B and optional curvature-calibrated
      stability penalty lambda * P(theta).
SUR   smoothed Abrevaya OBJ1 (bounded sigmoid), tau_c = n_c^-0.6, annealed 1 -> 0.3 -> tau_c
      (r3_estimators.anneal_schedule); grid+refine when the (active) space is 2-d, sphere L-BFGS
      with the RNC warm start plus random starts otherwise.
MRC   exact maximum rank correlation by the streamed arc sweep (2-d only).
FE, NS, MM, ML  wrappers on the existing estimators in an (active) coordinate basis.
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import kendalltau

from . import kernels as kn
from .. import estimators as be
from .. import r3_estimators as r3e
from .. import robust_estimators as rbe
from ..estimators import unit, GRAD_OK


# ---------------------------------------------------------------------------
# RankNet with cell temperatures
# ---------------------------------------------------------------------------
def newton_global(cs, Z, max_iter=60):
    """Damped Newton for the single-temperature RankNet (convex), u in the coordinates of Z."""
    K, n, q = Z.shape
    ones = np.ones(K)

    def vg(u):
        v, _, gs = kn.rank_vg(cs, Z @ u, ones)
        return v, np.einsum('kia,ki->a', Z, gs)

    u = np.zeros(q)
    val, g = vg(u)
    for _ in range(max_iter):
        H = kn.rank_curv(cs, Z @ u, Z, ones)
        step = -np.linalg.solve(H + 1e-12 * np.eye(q), g)
        dec = -g @ step
        if np.max(np.abs(g)) < 1e-9 or dec / 2 < 1e-20:
            break
        t = 1.0
        while True:
            new = u + t * step
            nv, ng = vg(new)
            if nv <= val - 1e-4 * t * dec or t < 1e-10:
                break
            t *= 0.5
        if t < 1e-10 or nv >= val:
            break
        u, val, g = new, nv, ng
        if np.linalg.norm(u) > 1e6:
            break
    return u


def fit_rnc(cs, B=None, pen=None, lam=0.0, prior_sd=0.5, max_iter=1000):
    """Minimise  R(u, d) + sum_k pen_k d_k^2 + lam * P(B u / |B u|),  t_ij = exp(d_k) u'dz_ij.
    B: (p x q) orthonormal columns (hard restriction / active coordinates), default identity."""
    Z = cs.Zs if B is None else cs.Zs @ B
    K, n, q = Z.shape
    Bm = np.eye(q) if B is None else B
    u0 = newton_global(cs, Z)
    pen_d = np.full(K, 1.0 / K / (2 * prior_sd ** 2 * n))
    use_pen = lam > 0 and pen is not None

    def fun(x):
        u, d = x[:q], x[q:]
        M = np.exp(d)
        val, sgt, gs = kn.rank_vg(cs, Z @ u, M)
        g_u = np.einsum('kiq,ki->q', Z, gs)
        g_d = -sgt + 2 * pen_d * d
        val += float(pen_d @ (d * d))
        if use_pen:
            nu = np.linalg.norm(u)
            th = Bm @ u / nu
            P, gP = pen.value_grad(th)
            val += lam * P
            g_u = g_u + lam * (Bm.T @ (gP - th * (th @ gP))) / nu
        return val, np.concatenate([g_u, g_d])

    res = minimize(fun, np.concatenate([u0, np.zeros(K)]), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=max_iter, gtol=1e-10, ftol=1e-16, maxcor=30))
    u, d = res.x[:q], res.x[q:]
    gn = float(np.max(np.abs(fun(res.x)[1])))
    nu = float(np.linalg.norm(u))
    theta = unit(Bm @ u)
    return dict(theta=theta, M=np.exp(d) * nu, d=d, obj=float(res.fun), nit=int(res.nit), grad_norm=gn,
                ok=bool(res.success or gn < GRAD_OK), tau_median=float(np.median(1.0 / (nu * np.exp(d)))),
                tau_sd_log=float(np.std(-np.log(nu * np.exp(d)))))


# ---------------------------------------------------------------------------
# SUR
# ---------------------------------------------------------------------------
def _angdiff(a, b):
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def sur_value(cs, S, tau):
    return kn.sur_vg(cs, S, tau)[0]


def fit_sur(cs, tau_c, B, theta0=None, extra_starts=3, seed=0, grids=(36, 180), max_iter=2000):
    """Annealed SUR on span(B) (columns orthonormal). Returns dict(theta, ok, obj)."""
    Z = cs.Zs @ B
    K, n, q = Z.shape
    sched = r3e.anneal_schedule(np.full(K, float(tau_c)))
    if q == 1:
        vals = [sur_value(cs, s * Z[..., 0], sched[-1]) for s in (1.0, -1.0)]
        sgn = 1.0 if vals[0] <= vals[1] else -1.0
        return dict(theta=sgn * B[:, 0], ok=True, obj=float(min(vals)))
    if q == 2:
        prev, phi, val = None, 0.0, np.nan
        for li, tau in enumerate(sched):
            G = grids[-1] if li == len(sched) - 1 else grids[0]
            angles = np.linspace(-np.pi, np.pi, G, endpoint=False)
            step = 2 * np.pi / G
            vals = kn.sur_angles(cs, Z, tau, angles)
            best = float(angles[int(np.argmin(vals))])
            cands = [best] + ([prev] if prev is not None and _angdiff(prev, best) > step else [])
            fits = []
            for c in cands:
                r = minimize_scalar(lambda a: float(kn.sur_angles(cs, Z, tau, np.array([a]))[0]),
                                    bounds=(c - step, c + step), method='bounded', options=dict(xatol=1e-8))
                fits.append((float(r.fun), float(r.x)))
            val, phi = min(fits)
            prev = phi
        return dict(theta=B @ np.array([np.cos(phi), np.sin(phi)]), ok=True, obj=val)
    g = np.random.default_rng(seed)
    starts = []
    if theta0 is not None:
        starts.append(unit(B.T @ theta0))
    starts += [unit(g.normal(size=q)) for _ in range(extra_starts)]

    def fun_for(tau):
        def fun(u):
            nu = np.linalg.norm(u)
            th = u / nu
            v, gs = kn.sur_vg(cs, Z @ th, tau)
            gth = np.einsum('kiq,ki->q', Z, gs)
            return v, (gth - th * (th @ gth)) / nu
        return fun
    out = []
    for st in starts:
        th = st
        for tau in sched:
            r = minimize(fun_for(tau), th, jac=True, method='L-BFGS-B',
                         options=dict(maxiter=max_iter, gtol=1e-10, ftol=1e-15))
            th = unit(r.x)
        gn = float(np.max(np.abs(fun_for(sched[-1])(th)[1])))
        out.append((float(r.fun), th, bool(r.success or gn < GRAD_OK)))
    val, th, ok = min(out, key=lambda t: t[0])
    return dict(theta=B @ th, ok=ok, obj=val, spread=float(max(o[0] for o in out) - val))


def fit_mrc(cs, B, **kw):
    if B.shape[1] != 2:
        raise ValueError('exact MRC needs a 2-d (active) restricted space')
    phi, level, info = kn.mrc_exact_stream(cs, cs.Zs @ B, **kw)
    return dict(theta=B @ np.array([np.cos(phi), np.sin(phi)]), ok=True, obj=level, **info)


# ---------------------------------------------------------------------------
# within / likelihood estimators in an active basis
# ---------------------------------------------------------------------------
def fit_fe(z, W, B, scores='raw'):
    """Cell-FE OLS of W (or its within-cell normal scores) on z B; theta = unit(B delta)."""
    Y = rbe.normal_scores(W) if scores == 'normal' else W
    th_q, ex, info = be.fixed_effects(z @ B, Y)
    return dict(theta=unit(B @ th_q), delta=ex['b'] * (B @ th_q), ok=True)


def fit_mm(z, W, B):
    th_q, ex, info = be.mixed_model(z @ B, W)
    return dict(theta=unit(B @ th_q), delta=ex['b'] * (B @ th_q), ok=bool(info.get('success', True)),
                spec=ex.get('spec'))


def fit_ml(z, W, B, family, kappa, theta_start):
    zb = z @ B
    if family == 'a':
        th_q, ex, info = be.fgls(zb, W)
    else:
        th_q, ex, info = be.maximum_likelihood(zb, W, family, kappa, [unit(B.T @ theta_start)])
    return dict(theta=unit(B @ th_q), ok=bool(info.get('success', True)))


# ---------------------------------------------------------------------------
# native predictions of the practitioner arms
# ---------------------------------------------------------------------------
def native_fe(z_tr, W_tr, delta, z_te):
    """W_hat = cell intercept + x'delta (intercepts from the training cell means)."""
    a = W_tr.mean(1) - np.einsum('cntp,p->ct', z_tr, delta) / z_tr.shape[1]
    return a[:, None, :] + z_te @ delta


def native_mm(z_tr, W_tr, delta, z_te):
    """Fixed slope delta with empirical-Bayes shrunk cell-date slope on the index (random-slope
    BLUP analogue): W_hat = a_ct + b_ct * x'delta, b_ct shrunk towards 1."""
    s = z_tr @ delta
    sc, Wc = s - s.mean(1, keepdims=True), W_tr - W_tr.mean(1, keepdims=True)
    b = (sc * Wc).sum(1) / (sc * sc).sum(1)                                    # (C,2)
    res = Wc - b[:, None, :] * sc
    n = s.shape[1]
    v_noise = np.mean((res ** 2).sum(1) / (n - 2) / (sc * sc).sum(1))
    v_b = max(np.var(b) - v_noise, 0.0)
    shrink = v_b / (v_b + v_noise) if v_b + v_noise > 0 else 0.0
    bs = 1.0 + shrink * (b - 1.0)
    a = W_tr.mean(1) - bs * s.mean(1)
    return a[:, None, :] + bs[:, None, :] * (z_te @ delta)


def mean_kendall(s, W):
    """Mean within-cell Kendall tau (s, W: (C,m,2))."""
    return float(np.mean([kendalltau(s[c, :, t], W[c, :, t])[0]
                          for c in range(s.shape[0]) for t in range(2)]))
