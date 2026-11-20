"""Round 2 scorers: distribution-free variants of the ranking criterion.

  ranknet_cell     RankNet with a temperature per city-date cell, shrunk to the
                   global one (alternating: convex Newton in u, per-cell 1-d in log m)
  surrogate        bounded sigmoid loss on the unit-variance score, annealed over a
                   temperature schedule (non-convex; sphere L-BFGS with several
                   starts, or a global grid when the problem is one angle)
  mrc_exact_2d     exact maximum rank correlation for a 2-d feasible set (one
                   angle), by an O(N log N) arc sweep: global by construction
  normal_scores_fe FE OLS on within-cell normal scores of W (psi-invariant, closed form)
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, ndtri
from scipy.stats import rankdata

from .estimators import fixed_effects, unit, _tangent, GRAD_OK
from .losses import PairData, newton_ranknet

# ---------------------------------------------------------------------------
# RankNet with per-cell temperatures
# ---------------------------------------------------------------------------


def _cell_sum(pairs, values):
    return np.bincount(pairs.cell, weights=values, minlength=pairs.n_cells)


def cell_value_grad(pairs, x, penalty):
    """Per-cell-temperature RankNet objective in x = (u, d), d = log m per cell."""
    p = pairs.v.shape[1]
    u, d = x[:p], x[p:]
    c = pairs.v @ u
    m = np.exp(d)[pairs.cell]
    t = m * c
    sig = expit(-t)
    val = float(pairs.weight @ np.logaddexp(0.0, -t) + penalty @ (d * d))
    ws = pairs.weight * sig
    g_u = -(ws * m) @ pairs.v
    g_d = -_cell_sum(pairs, ws * t) + 2 * penalty * d
    return val, np.concatenate([g_u, g_d])


def ranknet_cell(pairs, n_per_cell, prior_sd=0.5, max_iter=3000):
    """RankNet with per-cell temperatures tau_ct = 1 / (|u| m_ct), m_ct = exp(d_ct).

    Joint smooth objective in (u, d), minimised by L-BFGS from the global-temperature
    RankNet solution (u, d = 0). Penalty per cell: w_cell * d^2 / (2 prior_sd^2 n),
    w_cell = 1/(number of cells), so the shrinkage weakens as tracts per cell grow."""
    K, p = pairs.n_cells, pairs.v.shape[1]
    penalty = np.full(K, 1.0 / K / (2 * prior_sd ** 2 * n_per_cell))
    u0, _ = newton_ranknet(pairs)

    fun = lambda x: cell_value_grad(pairs, x, penalty)

    res = minimize(fun, np.concatenate([u0, np.zeros(K)]), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=max_iter, gtol=1e-10, ftol=1e-16, maxcor=30))
    u, d = res.x[:p], res.x[p:]
    gn = float(np.max(np.abs(fun(res.x)[1])))
    tau = 1.0 / (np.linalg.norm(u) * np.exp(d))
    return unit(u), dict(tau_cell_sd_log=float(np.std(np.log(tau))), tau_median=float(np.median(tau))), \
        dict(success=bool(res.success or gn < GRAD_OK), objective=float(res.fun), iterations=int(res.nit),
             grad_norm=gn, n_starts=1, spread=0.0)


# ---------------------------------------------------------------------------
# Bounded sigmoid surrogate
# ---------------------------------------------------------------------------


def surrogate_value_grad(pairs, theta, tau):
    m = pairs.v @ theta / tau
    s = expit(-m)
    val = float(pairs.weight @ s)
    grad = -((pairs.weight * s * (1 - s)) / tau) @ pairs.v
    return val, grad


def _sphere_fun(pairs, tau):
    def fun(u):
        norm = np.linalg.norm(u)
        theta = u / norm
        val, g = surrogate_value_grad(pairs, theta, tau)
        return val, _tangent(theta, g) / norm
    return fun


def surrogate(pairs, theta0, taus=(1.0, 0.3, 0.1), extra_starts=3, seed=0, max_iter=2000):
    """Annealed bounded-sigmoid surrogate on the unit sphere (>= 3 dims).
    Returns {tau: (theta, info)} for every tau in the schedule."""
    p = pairs.v.shape[1]
    g = np.random.default_rng(seed)
    starts = [unit(theta0)] + [unit(g.normal(size=p)) for _ in range(extra_starts)]
    paths = []
    for start in starts:
        theta, path = start, {}
        for tau in taus:
            res = minimize(_sphere_fun(pairs, tau), theta, jac=True, method='L-BFGS-B',
                           options=dict(maxiter=max_iter, gtol=1e-10, ftol=1e-15))
            theta = unit(res.x)
            gn = float(np.max(np.abs(_sphere_fun(pairs, tau)(theta)[1])))
            path[tau] = (theta, float(res.fun), gn, bool(res.success or gn < GRAD_OK))
        paths.append(path)
    out = {}
    for tau in taus:
        vals = [pth[tau][1] for pth in paths]
        best = paths[int(np.argmin(vals))][tau]
        theta, val, gn, ok = best
        out[tau] = (theta, dict(success=ok, objective=val, grad_norm=gn, n_starts=len(starts),
                                spread=float(max(vals) - min(vals))))
    return out


def _angle_values(v2, weight, angles, tau, budget=2_000_000):
    out = np.empty(len(angles))
    chunk = max(1, budget // len(v2))
    for s in range(0, len(angles), chunk):
        a = angles[s:s + chunk]
        m = (v2[:, :1] * np.cos(a) + v2[:, 1:] * np.sin(a)) / tau
        out[s:s + chunk] = weight @ expit(-m)
    return out


def surrogate_2d(pairs2, taus=(1.0, 0.3, 0.1), grid=180):
    """One-angle surrogate (2-d feasible set): global grid + local refinement at
    every temperature, so no annealing dependence or local optima."""
    v2, w = pairs2.v, pairs2.weight
    angles = np.linspace(-np.pi, np.pi, grid, endpoint=False)
    step = 2 * np.pi / grid
    out = {}
    for tau in taus:
        vals = _angle_values(v2, w, angles, tau)
        best = angles[int(np.argmin(vals))]
        res = minimize_scalar(lambda a: _angle_values(v2, w, np.array([a]), tau)[0],
                              bounds=(best - step, best + step), method='bounded',
                              options=dict(xatol=1e-10))
        phi = float(res.x)
        out[tau] = (np.array([np.cos(phi), np.sin(phi)]),
                    dict(success=True, objective=float(res.fun), n_starts=grid, spread=0.0))
    return out


# ---------------------------------------------------------------------------
# Exact maximum rank correlation in two dimensions
# ---------------------------------------------------------------------------


def mrc_exact_2d(pairs2):
    """argmax_phi sum_k w_k 1{v_k'(cos phi, sin phi) > 0}: each pair is concordant
    on an open half-circle centred at its own angle, so sweep the arc endpoints."""
    v2, w = pairs2.v, pairs2.weight
    omega = np.arctan2(v2[:, 1], v2[:, 0])
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    start, end = wrap(omega - np.pi / 2), wrap(omega + np.pi / 2)
    initial = float(w[v2[:, 0] < 0].sum())            # value at phi = -pi
    events = np.concatenate([start, end])
    delta = np.concatenate([w, -w])
    order = np.argsort(events, kind='mergesort')
    events, delta = events[order], delta[order]
    level = initial + np.cumsum(delta)                 # value on (events[k], events[k+1])
    right = np.append(events[1:], events[0] + 2 * np.pi)
    length = right - events
    level = np.where(length > 1e-15, level, -np.inf)   # ignore zero-length intervals
    k = int(np.argmax(level))
    phi = wrap(0.5 * (events[k] + right[k]))
    return np.array([np.cos(phi), np.sin(phi)]), {}, dict(success=True, objective=float(level[k]),
                                                       n_starts=1, spread=0.0)


# ---------------------------------------------------------------------------
# Normal scores
# ---------------------------------------------------------------------------


def normal_scores(W):
    n = W.shape[1]
    ranks = rankdata(W, axis=1)
    return ndtri((ranks - 0.5) / n)


def normal_scores_fe(z, W):
    return fixed_effects(z, normal_scores(W))
