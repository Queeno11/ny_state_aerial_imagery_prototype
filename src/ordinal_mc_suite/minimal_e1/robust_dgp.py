"""Round 2 DGPs: noise-robustness scenarios for the ranking scorer.

Same covariates, links, f and map parameters as round 1 (dgp.py); only the noise
changes. rho = 0 throughout. Scenarios (map c unless noted):
  A  inside Gaussian:             W = psi(f + eta), eta ~ N(0,1)
  B  city-varying scale:          eta = sigma_c N(0,1), sigma_c = exp(0.5 z_c)
  C  city-varying shape:          even cities Gaussian, odd cities skewed mixture
  D  inside + outside:            W = psi(f + sqrt(.5) eps) + sqrt(.5) 1.45 beta_ct nu,
                                  nu = standardized LogNormal(0, .75^2) - 1 (median 0, skewed)
  E  map a, X-dependent skew:     eta = exp(0.5 A1c - 0.25 v) * skewed mixture
                                  (A1c = centered raw nuisance coordinate A1; the
                                  boundary case where within-cell ranking is NOT valid)
  F  inside Gaussian + 5% gross errors: W replaced by cell mean + 3 cell SD * N(0,1)
  A8 (round 3) as A but eta persistent across dates, rho = 0.8 (same X and innovations as A)
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import dgp

SCENARIOS = ['A', 'B', 'C', 'D', 'E', 'F']
SCENARIO_MAP = dict(A='c', B='c', C='c', D='c', E='a', F='c', A8='c')
# Round 3: A8 = scenario A with tract-level persistence rho = 0.8 (same X, same innovations).
SCENARIOS_R3 = ['A', 'A8', 'B', 'C', 'D', 'E', 'F']
A8_RHO = 0.8
NOISE_STREAM = dict(scale=31, shape=32, outer=33, contam=34)
OUTER_SD = 1.45          # approx. conditional SD of W in map c at lambda = 0, sigma = 1
LOGNORMAL_S = 0.75


def skewed_mixture(g, shape):
    """Mean 0, variance 1, right-skewed (median < 0): 20% N(1,1), 80% N(-.25,.5^2)."""
    label = g.random(shape) < 0.2
    z = g.normal(size=shape)
    return np.where(label, 1.0 + z, -0.25 + 0.5 * z) / np.sqrt(0.65)


def lognormal_noise(g, shape, s=LOGNORMAL_S):
    """Median-zero, unit-variance, right-skewed outer noise."""
    sd = np.sqrt((np.exp(s * s) - 1) * np.exp(s * s))
    return (np.exp(s * g.normal(size=shape)) - 1.0) / sd


@lru_cache(maxsize=4)
def _a1_moments(seed, size, p_h, p_a, city_sd, change_sd):
    cfg = dict(p_h=p_h, p_a=p_a, city_sd=city_sd, change_sd=change_sd,
               logbeta_sd=0.3, lambda_range=0.9)
    mix = dgp.mixing(seed, p_h + p_a)
    vals, c, remaining = [], 0, size // 2
    while remaining > 0:
        n = min(500, remaining)
        ctx = dgp.city_context(dgp.rng(seed, dgp.STREAM['ref_city'], c), cfg)
        x, _, _ = dgp.features(ctx, n, dgp.rng(seed, dgp.STREAM['ref_tract'], c), mix, cfg)
        vals.append((x @ mix)[..., p_h].ravel())
        remaining -= n
        c += 1
    a1 = np.concatenate(vals)
    return float(a1.mean()), float(a1.var())


def a1_moments(cfg):
    return _a1_moments(cfg['seed'], cfg['reference_size'], cfg['p_h'], cfg['p_a'],
                       cfg['city_sd'], cfg['change_sd'])


@dataclass
class NoisyPanel:
    z: np.ndarray
    W: np.ndarray
    stable: np.ndarray
    f: np.ndarray
    map_name: str
    contaminated: float = 0.0


def noisy_panel(cfg, scenario, cities, n, rep, block, city_key=None):
    """city_key=(block, rep) (round 3): draw the city-level objects (context and the
    city scale of scenario B) from another (block, rep) so that fresh tract/noise draws
    can be taken in the SAME cities. Default None = unchanged behaviour."""
    ref = dgp.reference(cfg)
    seed, n_draw = cfg['seed'], cfg['n_draw']
    map_name = SCENARIO_MAP[scenario]
    p_h = cfg['p_h']
    ctxs, xs, hs, etas, outers = [], [], [], [], []
    stable = None
    for c in range(cities):
        keys = (block, rep, c)
        ckeys = keys if city_key is None else (city_key[0], city_key[1], c)
        ctx = dgp.city_context(dgp.rng(seed, dgp.STREAM['src_city'], *ckeys), cfg)
        x, h, stable = dgp.features(ctx, n, dgp.rng(seed, dgp.STREAM['src_tract'], *keys),
                                    ref.mix, cfg, n_draw)
        eps = dgp.residuals(dgp.rng(seed, dgp.STREAM['src_resid'], *keys), n, 0.0, n_draw)
        outer = np.zeros((n, 2))
        if scenario in ('A', 'F'):
            eta = eps
        elif scenario == 'A8':
            eta = dgp.residuals(dgp.rng(seed, dgp.STREAM['src_resid'], *keys), n, A8_RHO, n_draw)
        elif scenario == 'B':
            sigma_c = np.exp(0.5 * dgp.rng(seed, NOISE_STREAM['scale'], *ckeys).normal())
            eta = sigma_c * eps
        elif scenario == 'C':
            eta = eps if c % 2 == 0 else skewed_mixture(
                dgp.rng(seed, NOISE_STREAM['shape'], *keys), (n_draw, 2))[:n]
        elif scenario == 'D':
            eta = np.sqrt(0.5) * eps
            nu = lognormal_noise(dgp.rng(seed, NOISE_STREAM['outer'], *keys), (n_draw, 2))[:n]
            outer = np.sqrt(0.5) * OUTER_SD * ctx['beta'][None, :] * nu
        elif scenario == 'E':
            mean, var = a1_moments(cfg)
            a1 = (x @ ref.mix)[..., p_h] - mean
            eps_skew = skewed_mixture(dgp.rng(seed, NOISE_STREAM['shape'], *keys), (n_draw, 2))[:n]
            eta = np.exp(0.5 * a1 - 0.25 * var) * eps_skew
        else:
            raise ValueError(scenario)
        ctxs.append(ctx); xs.append(x); hs.append(h); etas.append(eta); outers.append(outer)
    z = ref.whiten(np.stack(xs))
    params = dict(alpha=np.stack([c['alpha'] for c in ctxs]),
                  beta=np.stack([c['beta'] for c in ctxs]),
                  lam=np.array([c['lam'] for c in ctxs]))
    f = ref.signal(np.stack(hs))
    W = dgp.psi(f + np.stack(etas), map_name, params, cfg['kappa']) + np.stack(outers)
    frac = 0.0
    if scenario == 'F':
        W, frac = contaminate(W, cities, seed, rep, block, n, n_draw)
    return NoisyPanel(z, W, stable, f, map_name, frac)


def contaminate(W, cities, seed, rep, block, n, n_draw, share=0.05):
    W = W.copy()
    mean, sd = W.mean(axis=1), W.std(axis=1)
    hit = 0
    for c in range(cities):
        g = dgp.rng(seed, NOISE_STREAM['contam'], block, rep, c)
        mask = (g.random((n_draw, 2)) < share)[:n]
        draw = g.normal(size=(n_draw, 2))[:n]
        W[c] = np.where(mask, mean[c] + 3.0 * sd[c] * draw, W[c])
        hit += mask.sum()
    return W, float(hit / W.size)


def evaluation_frame(cfg):
    """Fixed coordinates for signed-angle evaluation.

    The link null space is the same 2-d subspace in every dataset (the span of
    the H directions in whitened coordinates). e1 = theta*, e2 = the unit vector
    of that plane orthogonal to theta*, oriented toward the H2 direction.
    Also returns the unit direction of raw A1 orthogonal to the plane."""
    ref = dgp.reference(cfg)
    p_h = cfg['p_h']
    dirs = ref.chol.T @ ref.mix[:, :p_h]            # z-gradients of H1, H2
    q, _ = np.linalg.qr(dirs)
    e1 = ref.theta_star
    e2 = q @ (q.T @ dirs[:, 1])
    e2 = e2 - e1 * (e1 @ e2)
    e2 /= np.linalg.norm(e2)
    a1 = ref.chol.T @ ref.mix[:, p_h]
    a1 = a1 - q @ (q.T @ a1)
    a1 /= np.linalg.norm(a1)
    return e1, e2, q, a1
