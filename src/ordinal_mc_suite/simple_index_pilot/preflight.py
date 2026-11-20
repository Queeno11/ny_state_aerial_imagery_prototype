"""Audit the simple panel and optimizer before coefficient calibration."""
import json
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
from statsmodels import __version__ as statsmodels_version
from threadpoolctl import threadpool_limits

from classical import city_slopes, conditional_ml, fixed_effects, mixed_model
from data import CASES, TRUE_BETA, external_views, training
from losses import (angles, fit_head, fit_index, head_prediction, prepare, signed_pairs,
                    smooth_at, smooth_rank, sphere, spherical_grid, unit)


def checks(cfg):
    tiny = {**cfg, "cities": 4, "paired_tracts": 32, "fit_tracts": 24}
    seeds, fit, valid = training(tiny, 99)
    assert np.intersect1d(fit.tract_id, valid.tract_id).size == 0
    links = external_views(fit, seeds["fit_links"], cfg["stability_views"])
    grid = spherical_grid(24)
    report = {}
    for case in CASES:
        data = prepare(fit, case, links)
        mu, w = fit.outcomes(case)
        standardized_noise = (w-mu) / fit.cities.sigma[:, None, None]
        np.testing.assert_allclose(standardized_noise, fit.epsilon, atol=5e-15)
        assert np.min(signed_pairs(fit.x, mu) @ TRUE_BETA) > 0
        fe = fixed_effects(data)
        ml, status = conditional_ml(data, 0., fe.beta, tiny, grid)
        assert np.all(ml.head[:, 1] > 0) and np.all(ml.head[:, 2] >= 0)
        np.testing.assert_allclose(ml.sigma, np.sqrt(np.mean((ml.predict(fit.x)-w)**2, axis=(1,2))), atol=1e-14)
        params = angles(fe.beta)
        direction = np.array([.6, .8])
        weights = (1., cfg["lambda_S"], .2, .05, .001 if case == "C" else 0.)
        def eval_objective(angle):
            beta, jac = sphere(angle)
            val, grad = smooth_at(beta, data, weights, cfg)
            return val, float((jac @ grad) @ direction)
        _, deriv = eval_objective(params)
        h = 1e-5
        finite = (eval_objective(params + h*direction)[0] - eval_objective(params - h*direction)[0])/(2*h)
        assert abs(deriv - finite) < 1e-7
        numeric = dict(derivative_error=abs(deriv - finite), conditional_ml=status)
        if case == "A":
            beta = unit(fe.beta)
            head = fit_head(beta, data)
            err = float(np.max(np.abs(head_prediction(beta, head, fit.x) - fe.predict(fit.x))))
            assert err < 1e-12
            numeric["fe_fitted_value_max_difference"] = err
        if case == "C":
            noiseless = replace(data, w=mu)
            head = fit_head(TRUE_BETA, noiseless)
            shape = fit.cities.k
            slope = fit.cities.b / np.sqrt(1 + 6*shape + 15*shape**2)
            target = np.column_stack((fit.cities.a, slope, slope*shape))
            error = float(np.max(np.abs(head - target)))
            assert error < 1e-12
            numeric["noiseless_cubic_head_coefficient_error"] = error
        report[case] = numeric
    return report


def benchmark(cfg):
    seeds, fit, _ = training(cfg, 99)
    links = external_views(fit, seeds["fit_links"], cfg["stability_views"])
    d_b = prepare(fit, "B", links)
    d_c = prepare(fit, "C", links)
    fe = fixed_effects(d_c)
    beta = unit(fe.beta)
    smooth_rank(d_c.pairs, beta, cfg["tau"])
    start = time.perf_counter()
    for _ in range(20):
        smooth_at(beta, d_c, (1., .3, .2, .05, .001), cfg)
    per_eval = (time.perf_counter() - start)/20
    grid = spherical_grid(cfg["sphere_grid"])
    start = time.perf_counter()
    rank_grid = [smooth_rank(d_c.pairs, x, cfg["tau"])[0] for x in grid]
    grid_seconds = time.perf_counter() - start
    start = time.perf_counter()
    _, opt = fit_index(d_c, (1., .3, .2, .05, .001), cfg, beta, grid, rank_grid)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    _, mixed = mixed_model(d_b, city_slopes(d_b), "index")
    mixed_seconds = time.perf_counter() - start
    return dict(fit_endpoints=int(d_c.w.size), unordered_pairs=len(d_c.pairs), mean_eval_seconds=per_eval,
                rank_grid_seconds=grid_seconds, four_start_profile_seconds=fit_seconds,
                profile=opt, mixed_b_seconds=mixed_seconds, mixed_b=mixed, statsmodels=statsmodels_version,
                baseline_f_variance=1., case_c_gaussian_variance_ratio=1.,
                case_c_linear_projection_factor_formula="(1+3k)/sqrt(1+6k+15k^2)",
                date_1_f_variance=1 + np.mean(np.array([-.4, -.15, .15, .4])**2)/2)


def main():
    root = Path(__file__).resolve().parent
    cfg = json.loads((root / "config.json").read_text())
    with threadpool_limits(limits=1):
        output = dict(numerical=checks(cfg), actual_size=benchmark(cfg))
    with (root / "preflight_final.json").open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
