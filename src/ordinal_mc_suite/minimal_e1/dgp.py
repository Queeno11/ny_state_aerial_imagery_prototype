"""Data-generating process: covariates, residuals, the three maps, links.

Everything the estimators may use is ``z`` (whitened observed features) and
``W`` (outcomes). ``f``, map parameters and the true coefficient ``theta_star``
are evaluation-only and never passed to a fitting routine.
"""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.linalg import solve_triangular

# Fixed nuisance loading: A = H B' + city shift + N(0, I_4), redrawn by date.
B_NUISANCE = np.array([[0.8, 0.4],
                       [0.0, -0.3],
                       [0.5, 0.0],
                       [-0.2, 0.6]])

STREAM = dict(ref_city=1, ref_tract=2, mix=3,
              src_city=10, src_tract=11, src_resid=12,
              tgt_city=20, tgt_test=21, cal_tract=23, cal_resid=24)


def rng(seed, *keys):
    return np.random.default_rng(np.random.SeedSequence([int(seed), *map(int, keys)]))


# --------------------------------------------------------------------------
# Covariates
# --------------------------------------------------------------------------
def mixing(seed, p):
    q, r = np.linalg.qr(rng(seed, STREAM['mix']).normal(size=(p, p)))
    return q * np.sign(np.diag(r))


def city_context(g, cfg):
    dc = cfg['city_sd'] * g.normal(size=cfg['p_h'])
    da = 0.5 * g.normal(size=cfg['p_a'])
    alpha = (0.5 * dc[0] + 0.25 * dc[1] + 0.5 * g.normal()
             + 0.2 * np.arange(2) + 0.1 * g.normal(size=2))
    logbeta = cfg['logbeta_sd'] * g.normal(size=2)
    lam = g.uniform(-cfg['lambda_range'], cfg['lambda_range'])
    return dict(dc=dc, da=da, alpha=alpha, beta=np.exp(logbeta), lam=float(lam))


def features(context, n, g, mix, cfg, n_draw=None):
    """Paired covariates for n tracts (prefix of an n_draw draw).

    Even-indexed tracts are structurally stable (Delta H = 0); the rest move by
    change_sd * N(0, I). H is skewed (centered Exp(1)) on purpose: with Gaussian
    X every single-index estimator would be direction-consistent for free.
    """
    n_draw = n_draw or n
    ph, pa = cfg['p_h'], cfg['p_a']
    v = g.exponential(size=(n_draw, ph)) - 1.0
    dz = g.normal(size=(n_draw, ph))
    noise = g.normal(size=(n_draw, 2, pa))
    stable = np.arange(n_draw) % 2 == 0
    h0 = context['dc'] + v
    dh = np.where(stable[:, None], 0.0, cfg['change_sd'] * dz)
    h = np.stack([h0, h0 + dh], axis=1)
    a = h @ B_NUISANCE.T + context['da'] + noise
    x = np.concatenate([h, a], axis=-1) @ mix.T
    return x[:n], h[:n], stable[:n]


@dataclass
class Reference:
    mean: np.ndarray
    chol: np.ndarray
    signal_mean: float
    signal_sd: float
    signal_coef: np.ndarray
    mix: np.ndarray
    theta_star: np.ndarray = field(repr=False)   # evaluation only

    def whiten(self, x):
        flat = x.reshape(-1, x.shape[-1]) - self.mean
        z = solve_triangular(self.chol, flat.T, lower=True).T
        return z.reshape(x.shape)

    def signal(self, h):
        return (h @ self.signal_coef - self.signal_mean) / self.signal_sd


@lru_cache(maxsize=4)
def _reference(seed, size, p_h, p_a, city_sd, change_sd, signal):
    cfg = dict(p_h=p_h, p_a=p_a, city_sd=city_sd, change_sd=change_sd,
               logbeta_sd=0.3, lambda_range=0.9)
    p = p_h + p_a
    mix = mixing(seed, p)
    xs, hs = [], []
    per_city, c, remaining = 500, 0, size // 2
    while remaining > 0:
        n = min(per_city, remaining)
        ctx = city_context(rng(seed, STREAM['ref_city'], c), cfg)
        x, h, _ = features(ctx, n, rng(seed, STREAM['ref_tract'], c), mix, cfg)
        xs.append(x.reshape(-1, p))
        hs.append(h.reshape(-1, p_h))
        remaining -= n
        c += 1
    x, h = np.concatenate(xs), np.concatenate(hs)
    mean = x.mean(0)
    cov = (x - mean).T @ (x - mean) / len(x)
    chol = np.linalg.cholesky(cov)
    coef = np.asarray(signal, float)
    g = h @ coef
    gm, gs = float(g.mean()), float(g.std())
    theta_star = chol.T @ (mix[:, :p_h] @ coef) / gs
    return Reference(mean, chol, gm, gs, coef, mix, theta_star)


def reference(cfg):
    return _reference(cfg['seed'], cfg['reference_size'], cfg['p_h'], cfg['p_a'],
                      cfg['city_sd'], cfg['change_sd'], tuple(cfg['signal']))


# --------------------------------------------------------------------------
# Maps
# --------------------------------------------------------------------------
def sinh_map(r, lam, kappa):
    return ((1 + lam) * np.exp(kappa * r) - (1 - lam) * np.exp(-kappa * r)) / (2 * kappa)


def sinh_deriv(r, lam, kappa):
    return ((1 + lam) * np.exp(kappa * r) + (1 - lam) * np.exp(-kappa * r)) / 2


def sinh_inverse(y, lam, kappa):
    """Solve S_lambda(r) = y exactly, without cancellation for very negative y."""
    ky = kappa * y
    root = np.sqrt(ky * ky + 1 - lam * lam)
    positive = (ky + root) / (1 + lam)
    negative = (1 - lam) / np.where(ky < 0, root - ky, 1.0)
    e = np.where(ky >= 0, positive, negative)
    return np.log(e) / kappa


def _cell(params, ndim):
    """Broadcast (C,2) cell parameters and (C,) city parameters over tracts."""
    alpha = params['alpha'][:, None, :]
    beta = params['beta'][:, None, :]
    lam = params['lam'][:, None, None]
    return alpha, beta, lam


def psi(r, family, params, kappa):
    alpha, beta, lam = _cell(params, r.ndim)
    if family == 'a':
        return alpha + r
    if family == 'b':
        return alpha + beta * r
    if family == 'c':
        return alpha + beta * sinh_map(r, lam, kappa)
    raise ValueError(family)


def psi_inverse(w, family, params, kappa):
    alpha, beta, lam = _cell(params, w.ndim)
    if family == 'a':
        return w - alpha
    if family == 'b':
        return (w - alpha) / beta
    if family == 'c':
        return sinh_inverse((w - alpha) / beta, lam, kappa)
    raise ValueError(family)


def conditional_mean(s, family, params, kappa, sigma=1.0):
    """E[psi(s + eta)] with eta ~ N(0, sigma^2) marginally (any rho)."""
    alpha, beta, lam = _cell(params, s.ndim)
    if family == 'a':
        return alpha + s
    if family == 'b':
        return alpha + beta * s
    if family == 'c':
        return alpha + beta * np.exp(kappa ** 2 * sigma ** 2 / 2) * sinh_map(s, lam, kappa)
    raise ValueError(family)


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------
def residuals(g, n, rho, n_draw=None):
    n_draw = n_draw or n
    u = g.normal(size=(n_draw, 1))
    eps = g.normal(size=(n_draw, 2))
    return (np.sqrt(rho) * u + np.sqrt(1 - rho) * eps)[:n]


@dataclass
class Panel:
    z: np.ndarray            # (C, n, 2, p) whitened features: estimator input
    W: dict                  # map -> (C, n, 2) outcomes: estimator input
    stable: np.ndarray       # (n,) declared-stable tracts (link supply)
    f: np.ndarray            # evaluation only
    params: dict             # evaluation only (alpha, beta, lam per city)


def _assemble(ctxs, xs, ref):
    z = ref.whiten(np.stack(xs))
    params = dict(alpha=np.stack([c['alpha'] for c in ctxs]),
                  beta=np.stack([c['beta'] for c in ctxs]),
                  lam=np.array([c['lam'] for c in ctxs]))
    return z, params


def source_panel(cfg, rho, cities, n, rep, block=0, maps=('a', 'b', 'c')):
    """Source data. Common random numbers: X and (u, eps) do not depend on rho
    and every (cities, n) panel is a prefix of the same (rep, block) draw."""
    ref = reference(cfg)
    seed, n_draw = cfg['seed'], cfg['n_draw']
    if n > n_draw:
        raise ValueError('n exceeds n_draw')
    ctxs, xs, hs, etas = [], [], [], []
    stable = None
    for c in range(cities):
        ctx = city_context(rng(seed, STREAM['src_city'], block, rep, c), cfg)
        x, h, stable = features(ctx, n, rng(seed, STREAM['src_tract'], block, rep, c),
                                ref.mix, cfg, n_draw)
        eta = residuals(rng(seed, STREAM['src_resid'], block, rep, c), n, rho, n_draw)
        ctxs.append(ctx); xs.append(x); hs.append(h); etas.append(eta)
    z, params = _assemble(ctxs, xs, ref)
    f = ref.signal(np.stack(hs))
    r = f + np.stack(etas)
    W = {m: psi(r, m, params, cfg['kappa']) for m in maps}
    return Panel(z, W, stable, f, params)


@dataclass
class Target:
    z: np.ndarray            # (T, n_test, 2, p)
    f: np.ndarray
    mu: dict                 # map -> true conditional mean at f
    params: dict
    stable: np.ndarray
    sigma_level: np.ndarray  # E[z z'] over the test panel (bias/variance forms)
    sigma_change: np.ndarray  # E[dz dz']


def target_environment(cfg, maps=('a', 'b', 'c')):
    """Fixed across replications: target cities, maps and test panel."""
    ref = reference(cfg)
    seed = cfg['seed']
    ctxs, xs, hs = [], [], []
    stable = None
    for c in range(cfg['target_cities']):
        ctx = city_context(rng(seed, STREAM['tgt_city'], c), cfg)
        x, h, stable = features(ctx, cfg['test_n'], rng(seed, STREAM['tgt_test'], c), ref.mix, cfg)
        ctxs.append(ctx); xs.append(x); hs.append(h)
    z, params = _assemble(ctxs, xs, ref)
    f = ref.signal(np.stack(hs))
    mu = {m: conditional_mean(f, m, params, cfg['kappa']) for m in maps}
    flat = z.reshape(-1, z.shape[-1])
    dz = (z[:, :, 1] - z[:, :, 0]).reshape(-1, z.shape[-1])
    return Target(z, f, mu, params, stable, flat.T @ flat / len(flat), dz.T @ dz / len(dz))


def calibration_sample(cfg, target, rho, rep, maps=('a', 'b', 'c')):
    """Fresh labelled target-city calibration tracts each replication,
    disjoint from (and independent of) the fixed test panel."""
    ref = reference(cfg)
    seed, n = cfg['seed'], cfg['cal_n']
    xs, hs, etas = [], [], []
    for c in range(cfg['target_cities']):
        # Re-create the same city characteristics as the test panel.
        ctx = city_context(rng(seed, STREAM['tgt_city'], c), cfg)
        x, h, _ = features(ctx, n, rng(seed, STREAM['cal_tract'], rep, c), ref.mix, cfg)
        eta = residuals(rng(seed, STREAM['cal_resid'], rep, c), n, rho)
        xs.append(x); hs.append(h); etas.append(eta)
    z = ref.whiten(np.stack(xs))
    f = ref.signal(np.stack(hs))
    r = f + np.stack(etas)
    W = {m: psi(r, m, target.params, cfg['kappa']) for m in maps}
    return z, W, f


# --------------------------------------------------------------------------
# Links
# --------------------------------------------------------------------------
def link_null_space(z, stable, tol=1e-9):
    """Exact temporal links: every declared-stable tract requires
    (z_i1 - z_i0)' theta = 0. Returns an orthonormal basis of the feasible set."""
    gaps = (z[:, stable, 1, :] - z[:, stable, 0, :]).reshape(-1, z.shape[-1])
    _, sv, vt = np.linalg.svd(gaps, full_matrices=True)
    rank = int(np.sum(sv > tol * sv[0]))
    basis = vt[rank:].T
    return basis, rank, sv
