"""Round 3 ordinal-only scorers: smoothed OBJ1 (Abrevaya 2000, Eq. 9) with a per-cell
bandwidth schedule, and the same with a small RankNet anchor.

Notation. pairs.v = y_k * (z_j - z_i) (sign-oriented pair differences), pairs.weight = w_k
(sums to one), pairs.cell = cell index of pair k. theta is on the unit sphere.

  SUR(theta; tau_c) = sum_k w_k * sigmoid(-v_k'theta / tau_{c(k)})                 (bounded)

Equivalence with Abrevaya's smoothed OBJ1 with the logistic kernel K: K(-v) = 1 - K(v),
hence SUR = sum_k w_k (1 - K(v_k'theta/tau)) = const - sum_k w_k y_k K(Delta z_k'theta/tau)
(const = sum of w_k over pairs with y_k = +1, free of theta). It is consistent only if the
bandwidth tau -> 0 with sample size; at a fixed tau it is biased, hence the schedule
tau_c = n_c^(-0.6) (n_c = tracts in cell c).

  SURN   minimise SUR(theta; tau_c), annealed tau = 1.0 -> 0.3 -> tau_c (levels at or
         below the target are skipped). Full space: start from the RankNet-per-cell
         solution plus 3 seeded random starts, keep the best final objective. Two-d
         (link null space): global grid over the angle + bounded refinement at every level.
  SURH   minimise SUR(theta; tau_c) + eps_n * P_RN(theta / tau0), eps_n = 1/n,
         P_RN(u) = sum_k w_k log(1 + exp(-v_k'u)) and tau0 = median RNC temperature of the
         starting fit (fixed, not optimised). Same annealing and starts as SURN. The RankNet
         term is a small convex anchor that vanishes at rate 1/n and does not change the
         population maximiser of the smoothed OBJ1 in the limit.
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit

from .estimators import unit, _tangent, GRAD_OK

TAU_EXPONENT = 0.6
ANNEAL_LEVELS = (1.0, 0.3)


# ---------------------------------------------------------------------------
# Bandwidths and schedules
# ---------------------------------------------------------------------------
def bandwidth(n_c, exponent=TAU_EXPONENT):
    """tau_c = n_c^(-0.6): n=10 -> 0.251, 40 -> 0.109, 250 -> 0.036."""
    return np.asarray(n_c, float) ** (-exponent)


def tau_per_cell(pairs, n_c):
    """Target bandwidth vector over the pairs' cells (n_c scalar or one value per cell)."""
    return np.broadcast_to(bandwidth(n_c), (pairs.n_cells,)).copy()


def anneal_schedule(tau_target, levels=ANNEAL_LEVELS):
    """List of per-cell tau vectors, decreasing, ending at tau_target. A level is used
    only where it exceeds the target (levels at or below the target are skipped)."""
    target = np.broadcast_to(np.asarray(tau_target, float), np.shape(tau_target)).copy()
    sched = []
    for level in levels:
        vec = np.maximum(level, target)
        if np.any(vec > target * (1 + 1e-12)):
            sched.append(vec)
    sched.append(target)
    return sched


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
class SurObjective:
    """SUR(theta; tau_c) + eps * P_RN(theta / tau0) on pair data (any dimension)."""

    def __init__(self, pairs, tau_cell, eps=0.0, tau0=None):
        tau_cell = np.broadcast_to(np.asarray(tau_cell, float), (pairs.n_cells,))
        self.v, self.w, self.cell = pairs.v, pairs.weight, pairs.cell
        self.tau_cell = tau_cell
        self.inv_tau = (1.0 / tau_cell[0] if np.all(tau_cell == tau_cell[0])
                        else (1.0 / tau_cell)[pairs.cell])
        self.eps = float(eps)
        if self.eps > 0 and not tau0:
            raise ValueError('tau0 is required when eps > 0')
        self.inv_tau0 = 1.0 / tau0 if self.eps > 0 else 0.0

    def from_margin(self, m):
        val = float(self.w @ expit(-m * self.inv_tau))
        if self.eps > 0:
            val += self.eps * float(self.w @ np.logaddexp(0.0, -m * self.inv_tau0))
        return val

    def value(self, theta):
        return self.from_margin(self.v @ theta)

    def value_grad(self, theta):
        m = self.v @ theta
        s = expit(-m * self.inv_tau)
        val = float(self.w @ s)
        coef = -(self.w * s * (1.0 - s)) * self.inv_tau
        if self.eps > 0:
            b = m * self.inv_tau0
            val += self.eps * float(self.w @ np.logaddexp(0.0, -b))
            coef = coef - self.eps * self.inv_tau0 * self.w * expit(-b)
        return val, coef @ self.v

    def angle_values(self, angles, budget=2_000_000):
        """Value along theta = (cos a, sin a) (two-d problems only)."""
        out = np.empty(len(angles))
        chunk = max(1, budget // len(self.v))
        for s in range(0, len(angles), chunk):
            a = np.asarray(angles[s:s + chunk])
            m = self.v[:, :1] * np.cos(a) + self.v[:, 1:] * np.sin(a)
            val = self.w @ expit(-m * (self.inv_tau if np.ndim(self.inv_tau) == 0
                                       else self.inv_tau[:, None]))
            if self.eps > 0:
                val = val + self.eps * (self.w @ np.logaddexp(0.0, -m * self.inv_tau0))
            out[s:s + chunk] = val
        return out

    def subsample(self, max_pairs, seed=0):
        """Unbiased fixed subsample of the pairs (for the global grid scan only)."""
        n = len(self.w)
        if n <= max_pairs:
            return self
        idx = np.random.default_rng(seed).choice(n, size=max_pairs, replace=False)
        sub = SurObjective.__new__(SurObjective)
        sub.__dict__.update(self.__dict__)
        sub.v, sub.w = self.v[idx], self.w[idx] * (n / max_pairs)
        if np.ndim(self.inv_tau):
            sub.inv_tau = self.inv_tau[idx]
        return sub


def _sphere_fun(obj):
    def fun(u):
        norm = np.linalg.norm(u)
        theta = u / norm
        val, g = obj.value_grad(theta)
        return val, _tangent(theta, g) / norm
    return fun


# ---------------------------------------------------------------------------
# Optimisers
# ---------------------------------------------------------------------------
def sur_sphere(pairs, tau_target, theta0, eps=0.0, tau0=None, extra_starts=3, seed=0,
               max_iter=2000):
    """Annealed SUR (+ eps RankNet) on the unit sphere (>= 3 dims), multi-start."""
    sched = anneal_schedule(tau_target)
    objs = [SurObjective(pairs, t, eps, tau0) for t in sched]
    p = pairs.v.shape[1]
    g = np.random.default_rng(seed)
    starts = [unit(theta0)] + [unit(g.normal(size=p)) for _ in range(extra_starts)]
    results = []
    for start in starts:
        theta, res = start, None
        for obj in objs:
            res = minimize(_sphere_fun(obj), theta, jac=True, method='L-BFGS-B',
                           options=dict(maxiter=max_iter, gtol=1e-10, ftol=1e-15))
            theta = unit(res.x)
        gn = float(np.max(np.abs(_sphere_fun(objs[-1])(theta)[1])))
        results.append((float(res.fun), theta, gn, bool(res.success or gn < GRAD_OK)))
    vals = [r[0] for r in results]
    val, theta, gn, ok = results[int(np.argmin(vals))]
    return theta, dict(levels=len(sched)), dict(
        success=ok, objective=val, grad_norm=gn, n_starts=len(starts),
        spread=float(max(vals) - min(vals)))


def _angdiff(a, b):
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def sur_2d(pairs2, tau_target, eps=0.0, tau0=None, grid=180, grid_pairs=400_000, seed=0):
    """One-angle SUR (+ eps RankNet): at every annealing level a global grid scan
    (on a fixed unbiased subsample of at most grid_pairs pairs, exact otherwise) and a
    bounded refinement on the full data around the best grid point and around the
    previous level's solution (if it is in another basin)."""
    angles = np.linspace(-np.pi, np.pi, grid, endpoint=False)
    step = 2 * np.pi / grid
    prev, phi, val = None, 0.0, np.nan
    sched = anneal_schedule(tau_target)
    for tau in sched:
        obj = SurObjective(pairs2, tau, eps, tau0)
        vals = obj.subsample(grid_pairs, seed).angle_values(angles)
        best = float(angles[int(np.argmin(vals))])
        cands = [best] + ([prev] if prev is not None and _angdiff(prev, best) > step else [])
        fits = []
        for c in cands:
            res = minimize_scalar(lambda a: obj.angle_values(np.array([a]))[0],
                                  bounds=(c - step, c + step), method='bounded',
                                  options=dict(xatol=1e-9))
            fits.append((float(res.fun), float(res.x)))
        val, phi = min(fits)
        prev = phi
    return np.array([np.cos(phi), np.sin(phi)]), dict(levels=len(sched)), dict(
        success=True, objective=float(val), n_starts=grid, spread=0.0)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def _dispatch(pairs, n_c, start, eps, tau0, seed):
    tau_target = tau_per_cell(pairs, n_c)
    if pairs.v.shape[1] == 2:
        return sur_2d(pairs, tau_target, eps, tau0, seed=seed)
    return sur_sphere(pairs, tau_target, start, eps, tau0, seed=seed)


def surn(pairs, n_c, start=None, seed=0):
    """Smoothed OBJ1 with per-cell bandwidth n_c^(-0.6) (start unused in 2-d)."""
    return _dispatch(pairs, n_c, start, 0.0, None, seed)


def surh(pairs, n_c, start, tau0, seed=0):
    """Smoothed OBJ1 + (1/n) * RankNet(theta / tau0)."""
    return _dispatch(pairs, n_c, start, 1.0 / float(np.mean(n_c)), tau0, seed)
