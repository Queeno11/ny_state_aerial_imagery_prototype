"""Finite-difference, conditional-optimum, and actual-size profile-step checks."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from prior import ORIGINAL, BALANCED
from data import replication
from fitting import fit_data, new_scorer, normalize, scores
from heads import HeadSpec
from profile_objective import cardinal_objective, fitted_module, objective, solve


def load_parameters():
    cfg = json.loads((ORIGINAL / "config.json").read_text())
    selection = json.loads((BALANCED / "calibration/selected.json").read_text())
    pair = selection["best_positive_pair"]
    w, d, gamma = pair["lambda_W"], pair["lambda_D"], selection["best_full"]["gamma"]
    return cfg, dict(L=(w, 0., 0.), LC=(w, 0., gamma), LD=(w, d, 0.), LDC=(w, d, gamma))


def setup(cfg, device, dtype, replay):
    runtime, _, _, _, panel, valid, aux, aux_val = replication(cfg, 201)
    fit, val, _, denoms = fit_data(panel, valid, aux, aux_val, cfg, runtime, device, dtype)
    model = new_scorer(1674, cfg, device, dtype)
    if replay:
        saved = torch.load(BALANCED / "evaluation/confirmation/rep_201_start_0/checkpoints.pt",
                           weights_only=False, map_location="cpu")
        model.load_state_dict(saved["anchor"]["model"])
    z, _, _ = scores(model, fit, val, cfg)
    spec = HeadSpec(np.quantile(z[:fit.n].cpu().numpy(), cfg["knot_quantiles"]),
                    float(panel.w.mean()), np.sqrt(denoms["W"]), cfg["min_slope"])
    return model, fit, panel, spec, denoms


def finite_difference(cfg, options):
    tiny = {**cfg, "cities": 4, "tracts_per_city": 32}
    model, fit, panel, spec, denoms = setup(tiny, "cpu", torch.float64, False)
    params = tuple(model.parameters())
    vectors = tuple(torch.randn_like(p) for p in params)
    length = torch.sqrt(sum(v.square().sum() for v in vectors))
    directions = tuple(v / length for v in vectors)
    origin = tuple(p.detach().clone() for p in params)
    steps = [1e-4, 3e-5, 1e-5]
    results = {}
    for key, weights in options.items():
        z, _ = normalize(model(fit.inputs), fit, tiny["min_score_variance"])
        coefs, info = solve(z, panel, spec, denoms, tiny, weights)
        value, parts = objective(z, fit, fitted_module(spec, coefs, fit), denoms, tiny, weights)
        g = torch.autograd.grad(value, params)
        analytic = float(sum((a * b).sum() for a, b in zip(g, directions)))
        actual = cardinal_objective(z, panel, spec, denoms, weights, coefs)
        assert abs(actual - info["objective"] * weights[0]) < 1e-10
        torch_value = sum(weight * float(parts[name].detach())
                          for weight, name in zip(weights, ("level", "change", "curvature")))
        assert abs(actual - torch_value) < 1e-10
        estimates = []
        with torch.no_grad():
            for h in steps:
                endpoints = []
                for sign in (-1, 1):
                    for p, p0, d in zip(params, origin, directions):
                        p.copy_(p0 + h * sign * d)
                    shifted, _ = normalize(model(fit.inputs), fit, tiny["min_score_variance"])
                    shifted_coefs, _ = solve(shifted, panel, spec, denoms, tiny, weights, start=coefs)
                    trial, _ = objective(shifted, fit, fitted_module(spec, shifted_coefs, fit), denoms, tiny, weights)
                    endpoints.append(float(trial))
                estimates.append((endpoints[1] - endpoints[0]) / (2 * h))
            for p, p0 in zip(params, origin):
                p.copy_(p0)
        error = min(abs(x - analytic) for x in estimates)
        assert error < 2e-5, (key, analytic, estimates)
        results[key] = dict(analytic=analytic, finite_difference=estimates, steps=steps,
                            best_abs_error=error, weighted_head_loss=actual,
                            projected_head_gradient=info["projected_gradient"],
                            slope_min=float(coefs[:, 1:].min()))
    return results


def benchmark(cfg, options):
    results = {}
    for key, weights in options.items():
        model, fit, panel, spec, denoms = setup(cfg, "cuda", torch.float32, True)
        adam = torch.optim.Adam(model.parameters(), lr=cfg["joint_lr"])
        coefs = None
        times, head_times, iters, kkt, losses = [], [], [], [], []
        for update in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            adam.zero_grad(set_to_none=True)
            z, _ = normalize(model(fit.inputs), fit, cfg["min_score_variance"])
            head_start = time.perf_counter()
            coefs, info = solve(z, panel, spec, denoms, cfg, weights, start=coefs)
            head_time = time.perf_counter() - head_start
            value, parts = objective(z, fit, fitted_module(spec, coefs, fit), denoms, cfg, weights)
            torch_cardinal = sum(weight * float(parts[name].detach())
                                 for weight, name in zip(weights, ("level", "change", "curvature")))
            assert abs(torch_cardinal - info["objective"] * weights[0]) < 5e-7
            value.backward()
            adam.step()
            torch.cuda.synchronize()
            if update >= 4:
                times.append(time.perf_counter() - start)
                head_times.append(head_time)
                iters.append(info["iterations"])
                kkt.append(info["projected_gradient"])
            losses.append(float(value.detach()))
        results[key] = dict(fit_endpoints=fit.n, timed_updates=len(times), median_seconds=float(np.median(times)),
                            mean_seconds=float(np.mean(times)), mean_head_seconds=float(np.mean(head_times)),
                            mean_head_iterations=float(np.mean(iters)), max_projected_head_gradient=max(kkt),
                            objective_start=losses[0], objective_after_19=losses[-1],
                            increases=int((np.diff(losses) > 1e-7).sum()))
        print(key, results[key], flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg, weights = load_parameters()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        report = dict(weights=weights, finite_difference=finite_difference(cfg, weights),
                      timing=benchmark(cfg, weights))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
