"""Fixed effects, standard mixed effects, and conditional Gaussian benchmarks."""
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import minimize
from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMParams

from losses import (city_center, city_mean, fit_head, head_prediction, minimize_sphere, stable, unit)


@dataclass
class Model:
    beta: np.ndarray
    head: np.ndarray
    city_linear: np.ndarray = None
    sigma: np.ndarray = None

    def predict(self, x):
        if self.city_linear is not None:
            return self.city_linear[:, 0, None, None] + np.einsum("cntr,cr->cnt", x, self.city_linear[:, 1:])
        return head_prediction(self.beta, self.head, x)


def fixed_effects(data):
    x = city_center(data.x)
    y = city_center(data.w)
    beta = np.linalg.lstsq(x.reshape(-1, 3), y.ravel(), rcond=None)[0]
    intercept = city_mean(data.w) - city_mean(data.x) @ beta
    return Model(beta, np.column_stack((intercept, np.ones_like(intercept), np.zeros_like(intercept))))


def city_slopes(data):
    nc = len(data.w)
    x = data.x.reshape(nc, -1, 3)
    exog = np.concatenate((np.ones((*x.shape[:2], 1)), x), axis=2)
    y = data.w.reshape(nc, -1)
    coef = np.stack([np.linalg.lstsq(xx, yy, rcond=None)[0] for xx, yy in zip(exog, y)])
    residual = y - np.einsum("cni,ci->cn", exog, coef)
    sigma = np.sqrt(np.sum(residual**2, axis=1) / (y.shape[1] - exog.shape[2]))
    return Model(coef[:, 1:].mean(axis=0), None, coef, sigma)


def mixed_model(data, city_model, specification="unstructured"):
    """Standard REML MixedLM on city-prewhitened y, fixed X, and random X."""
    nc = len(data.w)
    n = data.w.shape[1] * 2
    x = data.x.reshape(-1, 3)
    exog = np.column_stack((np.ones(len(x)), x))
    groups = np.repeat(np.arange(nc), n)
    weights = np.repeat(1 / city_model.sigma, n)
    exog_white = exog * weights[:, None]
    y_white = data.w.reshape(-1) * weights
    if specification == "index":
        direction = unit(fixed_effects(data).beta)
        transform = np.array([[1., 0., 0., 0.], [0., *direction]])
    else:
        transform = np.eye(4)
    re_white = exog_white @ transform.T
    dimension = re_white.shape[1]
    model = MixedLM(y_white, exog_white, groups=groups, exog_re=re_white, use_sqrt=True)
    if specification == "diagonal":
        free = MixedLMParams.from_components(np.ones(4), np.eye(dimension))
    elif specification in ("unstructured", "index"):
        free = None
    else:
        raise ValueError(specification)
    covariance_start = np.cov(city_model.city_linear @ transform.T, rowvar=False)
    if specification == "diagonal":
        covariance_start = np.diag(np.diag(covariance_start))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_start)
    covariance_start = (eigenvectors * np.maximum(eigenvalues, 1e-4)) @ eigenvectors.T
    start = MixedLMParams.from_components(city_model.city_linear.mean(axis=0), covariance_start)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        result = model.fit(start_params=start, reml=True, method=["lbfgs", "bfgs"],
                           maxiter=350, free=free, full_output=True)
    covariance = np.asarray(result.cov_re)
    eigenvalues = np.linalg.eigvalsh(covariance)
    # Equivalent BLUP formula, including a well-defined limit for singular fitted covariance.
    coefficient = np.asarray(result.fe_params)
    city = re_white.reshape(nc, n, dimension)
    residual = (y_white - exog_white @ coefficient).reshape(nc, n)
    effects = []
    for x_c, y_c in zip(city, residual):
        small = np.eye(dimension)*result.scale + covariance @ (x_c.T @ x_c)
        effects.append(np.linalg.solve(small, covariance @ (x_c.T @ y_c)))
    city_params = coefficient + np.stack(effects) @ transform
    if not np.isfinite(city_params).all():
        raise ArithmeticError("Mixed model gave nonfinite predictions")
    audit = dict(converged=bool(result.converged), specification=specification, llf=float(result.llf),
                 residual_scale=float(result.scale), covariance_eigen_min=float(eigenvalues[0]),
                 covariance_eigen_max=float(eigenvalues[-1]), covariance_rank=int(np.sum(eigenvalues > 1e-6)),
                 warnings=" | ".join(dict.fromkeys(str(item.message) for item in messages)),
                 random_effect_dimension=dimension,
                 random_index_direction=transform[1, 1:].tolist() if specification == "index" else None)
    try:
        np.testing.assert_allclose(np.stack(list(result.random_effects.values())), effects,
                                   atol=1e-6, rtol=1e-6)
        audit["blup_matches_package"] = True
    except ValueError:
        audit["blup_matches_package"] = "singular covariance, valid limiting BLUP"
    return Model(coefficient[1:], None, city_params, np.sqrt(result.scale)*city_model.sigma), audit


def gaussian(beta, data, head, stability_weight, raw_a=False):
    r = data.x @ beta
    a, b, c = (head[:, i, None, None] for i in range(3))
    residual = a + b*r + c*r**3 - data.w
    variance = city_mean(residual**2)
    if np.min(variance) < 1e-10:
        raise ArithmeticError("Degenerate city residual variance")
    value = .5*np.mean(np.log(variance) + 1 + np.log(2*np.pi))
    derivative = (b + 3*c*r**2)[..., None] * data.x
    gradient = np.mean(residual[..., None] * derivative / variance[:, None, None, None], axis=(0, 1, 2))
    if stability_weight:
        z = unit(beta) if raw_a else beta
        s, gs = stable(z, data)
        if raw_a:
            gs = (gs-z*np.dot(gs, z))/np.linalg.norm(beta)
        value += stability_weight*s
        gradient += stability_weight*gs
    return float(value), gradient, np.sqrt(variance)


def conditional_ml(data, stability_weight, start, cfg, grid):
    if data.case == "A":
        def heads(beta):
            a = city_mean(data.w - data.x @ beta)
            return np.column_stack((a, np.ones(len(a)), np.zeros(len(a))))
        def objective(beta):
            value, gradient, _ = gaussian(beta, data, heads(beta), stability_weight, raw_a=True)
            return value, gradient
        starts = (start, .5*start, 1.5*start, -start)
        runs = [minimize(objective, init, jac=True, method="BFGS",
                         options=dict(maxiter=cfg["optimizer_maxiter"], gtol=3e-8)) for init in starts]
        best = min(runs, key=lambda r: r.fun)
        beta = best.x
        value, gradient = objective(beta)
        if np.linalg.norm(gradient) > 1e-5:
            raise ArithmeticError("Poor raw conditional ML solution")
        head = heads(beta)
        _, _, sigma = gaussian(beta, data, head, stability_weight, raw_a=True)
        audit = dict(objective=value, min_tangent_grad=float(np.linalg.norm(gradient)),
                     starts=len(runs), successful_starts=sum(x.success for x in runs),
                     evaluations=sum(x.nfev for x in runs), start_best_spread=float(np.ptp([x.fun for x in runs])))
    else:
        def objective(beta):
            val, grad, _ = gaussian(beta, data, fit_head(beta, data), stability_weight)
            return val, grad
        grid_values = [objective(beta)[0] for beta in grid]
        beta, audit = minimize_sphere(objective, [start, -start], grid, grid_values, cfg["optimizer_maxiter"])
        head = fit_head(beta, data)
        _, _, sigma = gaussian(beta, data, head, stability_weight)
    return Model(beta, head, sigma=sigma), audit
