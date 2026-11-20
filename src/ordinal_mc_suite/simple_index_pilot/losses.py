"""Profile the city heads and fit the normalized shared linear index."""
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.optimize import minimize


SLOPE_FLOOR = 1e-8


@njit(cache=False)
def smooth_rank(pairs, beta, tau):
    loss = 0.
    grad = np.zeros(3)
    for i in range(len(pairs)):
        x = pairs[i]
        margin = (x[0]*beta[0] + x[1]*beta[1] + x[2]*beta[2]) / tau
        if margin >= 0:
            exp = np.exp(-margin)
            p = exp / (1 + exp)
        else:
            p = 1 / (1 + np.exp(margin))
        loss += p
        derivative = -p * (1 - p) / tau
        for j in range(3):
            grad[j] += derivative * x[j]
    return loss / len(pairs), grad / len(pairs)


def signed_pairs(x, w):
    i, j = np.triu_indices(w.shape[1], 1)
    differences = x[:, j, :, :] - x[:, i, :, :]
    signs = np.sign(w[:, j, :] - w[:, i, :])
    if not np.all(signs):
        raise ValueError("Outcome ties are outside this continuous fixture")
    return np.ascontiguousarray((differences * signs[..., None]).reshape(-1, 3))


@dataclass
class Data:
    x: np.ndarray
    w: np.ndarray
    aux_scale: float
    case: str
    denom_w: float
    denom_d: float
    pairs: np.ndarray


def prepare(panel, case, aux, denoms=None, with_pairs=True):
    _, w = panel.outcomes(case)
    if denoms is None:
        denoms = (float(w.reshape(len(w), -1).var(axis=1, ddof=1).mean()),
                  float(np.diff(w, axis=2).var()))
    return Data(panel.x, w, float(np.mean(aux**2)), case, *denoms,
                signed_pairs(panel.x, w) if with_pairs else np.empty((0, 3)))


def city_center(v):
    return v - v.mean(axis=(1, 2), keepdims=True)


def city_mean(v):
    return np.mean(v, axis=(1, 2))


def fit_head(beta, data, level=1., change=0., gamma=0.):
    """Minimize the exact weighted level/change quadratic, with positive slope."""
    r = data.x @ beta
    w = data.w
    # A change-only head intercept is anchored to fitting level means.
    def inner(x, y):
        return (level / data.denom_w * city_mean(city_center(x) * city_center(y))
                + change / data.denom_d * np.mean(np.diff(x, axis=2) * np.diff(y, axis=2), axis=(1, 2)))
    a11, t1 = inner(r, r), inner(r, w)
    if data.case == "A":
        linear = np.full(len(w), max(float(t1.sum() / a11.sum()), SLOPE_FLOOR))
        cubic = np.zeros_like(linear)
    elif data.case == "B":
        linear = np.maximum(t1 / a11, SLOPE_FLOOR)
        cubic = np.zeros_like(linear)
    elif data.case == "C":
        r3 = r**3
        a12, a22, t2 = inner(r, r3), inner(r3, r3) + 36 * gamma / data.denom_w, inner(r3, w)
        determinant = a11 * a22 - a12**2
        if (determinant <= 0).any():
            raise ArithmeticError("Degenerate cubic quadratic")
        free_u, free_v = (t1*a22-t2*a12)/determinant, (t2*a11-t1*a12)/determinant
        candidates = np.stack((np.column_stack((free_u, free_v)),
                               np.column_stack((np.maximum(t1/a11, SLOPE_FLOOR), np.zeros_like(t1))),
                               np.column_stack((np.full_like(t1, SLOPE_FLOOR),
                                                np.maximum((t2-a12*SLOPE_FLOOR)/a22, 0.)))), axis=1)
        u, v = candidates[..., 0], candidates[..., 1]
        objective = (a11[:, None]*u**2 + 2*a12[:, None]*u*v + a22[:, None]*v**2
                     - 2*t1[:, None]*u - 2*t2[:, None]*v)
        objective[(u < SLOPE_FLOOR) | (v < 0)] = np.inf
        best = candidates[np.arange(len(w)), np.argmin(objective, axis=1)]
        linear, cubic = best.T
    else:
        raise ValueError(data.case)
    intercept = city_mean(w) - linear*city_mean(r) - cubic*city_mean(r**3)
    return np.column_stack((intercept, linear, cubic))


def head_prediction(beta, head, x):
    r = x @ beta
    a, b, c = (head[:, i, None, None] for i in range(3))
    return a + b*r + c*r**3


def cardinal(beta, head, data, gradients=True):
    r = data.x @ beta
    a, b, c = (head[:, i, None, None] for i in range(3))
    residual = a + b*r + c*r**3 - data.w
    difference = np.diff(residual, axis=2)
    lw = np.mean(residual**2) / data.denom_w
    ld = np.mean(difference**2) / data.denom_d
    j = 36 * np.mean(c**2) / data.denom_w
    if gradients:
        deriv = (b + 3*c*r**2)[..., None] * data.x
        axes = (0, 1, 2)
        gw = 2*np.mean(residual[..., None]*deriv, axis=axes) / data.denom_w
        gd = 2*np.mean(difference[..., None]*np.diff(deriv, axis=2), axis=axes) / data.denom_d
        return (lw, ld, j), (gw, gd)
    return lw, ld, j


def stable(beta, data):
    g = np.zeros(3)
    g[2] = 2*data.aux_scale*beta[2]
    return data.aux_scale*beta[2]**2, g


def unit(beta):
    norm = np.linalg.norm(beta)
    if norm < 1e-14:
        raise ArithmeticError("Cannot normalize zero score")
    return beta / norm


def sphere(angles):
    azimuth, elev = angles
    ca, sa, ce, se = np.cos(azimuth), np.sin(azimuth), np.cos(elev), np.sin(elev)
    return np.array([ca*ce, sa*ce, se]), np.array([[-sa*ce, ca*ce, 0], [-ca*se, -sa*se, ce]])


def angles(beta):
    x, y, z = unit(beta)
    return np.array([np.arctan2(y, x), np.arcsin(np.clip(z, -1., 1.))])


def spherical_grid(count):
    i = np.arange(count)
    z = (2*i + 1)/count - 1
    longitude = np.pi*(3 - np.sqrt(5))*i
    return np.column_stack((np.sqrt(1-z**2)*np.cos(longitude), np.sqrt(1-z**2)*np.sin(longitude), z))


def smooth_at(beta, data, weights, cfg, rank_cache=None):
    wr, ws, ww, wd, gamma = weights
    value = 0.
    grad = np.zeros(3)
    if wr:
        rank, gr = smooth_rank(data.pairs, beta, cfg["tau"]) if rank_cache is None else (rank_cache, np.zeros(3))
        value += wr * rank
        grad += wr * gr
    if ws:
        s, gs = stable(beta, data)
        value += ws * s
        grad += ws * gs
    if ww or wd:
        head = fit_head(beta, data, ww, wd, gamma)
        (lw, ld, j), (gw, gd) = cardinal(beta, head, data)
        value += ww*lw + wd*ld + gamma*j
        grad += ww*gw + wd*gd
    return float(value), grad


def minimize_sphere(objective, starts, grid, grid_values, maxiter):
    def on_angles(x):
        beta, jac = sphere(x)
        val, grad = objective(beta)
        return val, jac @ grad
    ordering = np.argsort(grid_values)
    spread = next((i for i in ordering[1:] if grid[ordering[0]] @ grid[i] < .55), ordering[1])
    candidates = [*starts, grid[ordering[0]], grid[spread]]
    solutions = [minimize(on_angles, angles(start), jac=True, method="BFGS",
                          options=dict(maxiter=maxiter, gtol=3e-8)) for start in candidates]
    best = min(solutions, key=lambda x: x.fun)
    beta, _ = sphere(best.x)
    obj, grad = objective(beta)
    tangent = grad - beta*np.dot(beta, grad)
    if obj > min(grid_values) + 1e-9 or np.linalg.norm(tangent) > 1e-4:
        raise ArithmeticError(f"Poor sphere solution: {best.message}, tangent {np.linalg.norm(tangent)}")
    return beta, dict(objective=obj, grid_min=float(min(grid_values)), evaluations=sum(x.nfev for x in solutions),
                      min_tangent_grad=float(np.linalg.norm(tangent)), any_success=any(x.success for x in solutions),
                      successful_starts=sum(x.success for x in solutions), starts=len(solutions),
                      start_best_spread=float(np.ptp([x.fun for x in solutions])))


def fit_index(data, weights, cfg, start, grid, rank_grid):
    if not (weights[0] or weights[2] or weights[3]):
        raise ValueError("No outcome or ordinal supervision")
    vals = [smooth_at(b, data, weights, cfg, rank_cache=v)[0] for b, v in zip(grid, rank_grid)]
    return minimize_sphere(lambda beta: smooth_at(beta, data, weights, cfg),
                           [start, -start], grid, vals, cfg["optimizer_maxiter"])


def validation(beta, prediction, data, cfg):
    beta = unit(beta)
    margin = data.pairs @ beta
    rank = np.mean(margin < 0) + .5*np.mean(margin == 0)
    s, _ = stable(beta, data)
    residual = prediction - data.w
    w = np.mean(residual**2) / data.denom_w
    d = np.mean(np.diff(residual, axis=2)**2) / data.denom_d
    return dict(rank=float(rank), stability=float(s), level=float(w), change=float(d),
                composite=float(rank + cfg["lambda_S"]*s + w + d))
