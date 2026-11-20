"""Numerical and tiny-fit checks before the production gate."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ordinal_mc.core import seed_for
from ordinal_mc.design import inverse_measurement
from ordinal_mc.e6.data import measurement

from data import counterfactual, fit_moments, population_check, replication, test_panels
from fitting import cardinal, fit_data, known_prediction, loss_parts, new_scorer, normalize, pair_stats, scores
from heads import HeadSpec, MonotoneHead, curvature, fit_heads, slope_spacing


def invariance(fit, aux, support, dgp, cfg):
    from data import TRUTH
    f = support.f
    h = lambda u: u + 0.3 * np.tanh(u)
    original = fit_moments(fit.f, fit.n_cells)
    warped = fit_moments(h(fit.f), fit.n_cells)
    z0 = (f - original[0]) / np.sqrt(original[1])
    z1 = (h(f) - warped[0]) / np.sqrt(warped[1])
    recovered = inverse_measurement(warped[0] + np.sqrt(warped[1]) * z1,
                                    0., 1., 0.3, 1., "tanh")
    m0 = measurement(original[0] + np.sqrt(original[1]) * z0,
                     support.a, support.b, support.k, "nonlinear")
    m1 = measurement(recovered, support.a, support.b, support.k, "nonlinear")
    d = lambda x: x[support.m:] - x[:support.m]
    true_delta = d(f)
    transform_effect = d(z1) - d(z0)
    f_aux = dgp.score((aux @ dgp.mix.T)[:, :3], TRUTH)
    max_stable = float(np.max(np.abs(f_aux - np.repeat(fit.f, cfg["views"]))))
    assert max_stable < 1e-12
    ranks_match = np.array_equal(np.argsort(f.reshape(support.n_cells, -1), axis=1),
                                 np.argsort(z1.reshape(support.n_cells, -1), axis=1))
    assert ranks_match and np.max(abs(m1 - m0)) < 1e-11
    assert np.max(abs(d(m1) - d(m0))) < 1e-11
    assert np.array_equal(np.sign(d(z1)), np.sign(true_delta))
    # Equal latent quantile ranges in each coordinate. First/last secants include .05/.95 tails.
    def representation_j(transform, moments):
        q = np.quantile(fit.f, [0.05, *cfg["knot_quantiles"], 0.95])
        z = (transform(q) - moments[0]) / np.sqrt(moments[1])
        _, iw = fit_moments(fit.w, fit.n_cells)
        psi = (fit.b[:, None] * (q + fit.k[:, None] * np.tanh(q))) / np.sqrt(iw)
        slopes = np.diff(psi, axis=1) / np.diff(z)
        return float(np.mean(np.sum(np.diff(slopes, axis=1)**2 / slope_spacing(z[1:-1]), axis=1)))
    stable = ~support.changed
    return dict(original_fit_mean=original[0], original_fit_dispersion=original[1],
                warped_fit_mean=warped[0], warped_fit_dispersion=warped[1],
                level_prediction_max_diff=float(np.max(abs(m1 - m0))),
                change_prediction_max_diff=float(np.max(abs(d(m1) - d(m0)))),
                observed_prediction_max_diff=float(np.max(abs(m1 + support.errors - support.w))),
                ranks_match=ranks_match, exact_aux_latent_gap_max=max_stable,
                exact_aux_warped_gap_max=float(np.max(np.abs(h(f_aux) - h(np.repeat(fit.f, cfg["views"]))))),
                stable_movement_max=float(max(np.max(abs(d(z0)[stable])), np.max(abs(d(z1)[stable])))),
                changed_gap_difference_rms=float(np.sqrt(np.mean(transform_effect[~stable]**2))),
                changed_gap_difference_max=float(np.max(np.abs(transform_effect[~stable]))),
                normalization_only_change_rmse=float(np.sqrt(np.mean((d(z0)[~stable] - true_delta[~stable])**2))),
                secant_curvature_original=representation_j(lambda u: u, original),
                secant_curvature_warped=representation_j(h, warped))


def regime_checks(cfg, dgp, panels):
    output = {}
    for where, (support, fixture) in panels.items():
        x, f1 = counterfactual(fixture, 0.4 * np.where(np.arange(fixture.m) % 2, 1, -1), dgp)
        f0 = fixture.f[:fixture.m]
        p = fixture.m
        baseline = (fixture.a[:p], fixture.b[:p], fixture.k[:p])
        w0 = measurement(f0, *baseline, "nonlinear")
        w_structural = measurement(f1, *baseline, "nonlinear")
        w_opposing = w_structural - 2 * fixture.b[:p] * np.sign(f1 - f0)
        keep = abs(f1 - f0) > 0.05
        assert np.all(np.sign(w_opposing[keep] - w0[keep]) == -np.sign(f1[keep] - f0[keep]))
        _, f_small = counterfactual(fixture, 0.15 * np.where(np.arange(p) % 2, 1, -1), dgp)
        w_small = measurement(f_small, fixture.a[p:], fixture.b[p:], fixture.k[p:], "nonlinear")
        ds = f_small - f0
        true_opposition = (abs(ds) > 0.05) & ((w_small - w0) * ds < 0)
        output[where] = dict(max_abs_structural_step_gap=float(np.max(abs(f1 - f0))),
                             opposed_custom_panel_share=float(keep.mean()),
                             supported_opposition_share=float(true_opposition.mean()),
                             supported_opposition_count=int(true_opposition.sum()),
                             supported_opposition_positive=int((true_opposition & (ds > 0)).sum()),
                             observed_fit_map_level_sd=float(np.std(support.mean_w)))
    return output


def setup(cfg, device, dtype):
    runtime, dgp, seeds, contexts, fit, val, aux, val_aux = replication(cfg, 1)
    a, b, _, denoms = fit_data(fit, val, aux, val_aux, cfg, runtime, device, dtype)
    model = new_scorer(seed_for(cfg["seed"], "preflight_start"), cfg, device, dtype)
    return fit, val, a, b, model, denoms


def verify_tensor(cfg, fixture):
    fit, _, a, b, model, denoms = fixture
    raw = model(a.inputs)
    z, _ = normalize(raw, a, cfg["min_score_variance"])
    z_affine, _ = normalize(2.3 * raw + 0.2, a, cfg["min_score_variance"])
    torch.testing.assert_close(z, z_affine, atol=2e-12, rtol=2e-12)
    target_z = (a.target_f - a.mean_f) / a.sd_f
    means = known_prediction(target_z, a, a)
    torch.testing.assert_close(means, torch.tensor(fit.mean_w), rtol=1e-12, atol=1e-12)
    count = a.n // a.n_cells
    idx, jdx = np.triu_indices(count, 1)
    cells = z[:a.n].reshape(a.n_cells, -1)
    reference = torch.sigmoid(-a.signs[:, idx, jdx] * (cells[:, jdx] - cells[:, idx]) / 0.2).mean()
    rank = pair_stats(z, a, 0.2)
    torch.testing.assert_close(rank, reference, rtol=1e-12, atol=1e-12)
    params = tuple(model.parameters())
    g = torch.cat([x.ravel() for x in torch.autograd.grad(rank, params, retain_graph=True)])
    g_ref = torch.cat([x.ravel() for x in torch.autograd.grad(reference, params, retain_graph=True)])
    torch.testing.assert_close(g, g_ref, rtol=1e-11, atol=1e-12)
    affine_rank = pair_stats(z_affine, a, 0.2)
    g_affine = torch.cat([x.ravel() for x in torch.autograd.grad(affine_rank, params)])
    torch.testing.assert_close(g, g_affine, rtol=1e-11, atol=1e-12)
    z, _, _ = scores(model, a, b, cfg)
    z = z[:a.n].numpy()
    spec = HeadSpec(np.quantile(z, cfg["knot_quantiles"]), float(fit.w.mean()),
                    np.sqrt(denoms["W"]), cfg["min_slope"])
    coefficients, report = fit_heads(z, fit, spec, denoms, cfg["head_max_iter"], gamma=cfg["gamma"])
    joint = MonotoneHead(spec, coefficients, torch.float64, "cpu")
    prediction = joint(torch.tensor(z), a.cells).detach().numpy()
    direct = spec.direct(coefficients, z, fit.cells)
    max_discrepancy = float(np.max(abs(direct - prediction)))
    assert max_discrepancy < 2e-5 and min(joint.slopes().detach().ravel()) > 0
    assert abs(float(joint.penalty().detach()) - curvature(coefficients, spec)) < 1e-5
    residual = prediction - fit.w
    d = residual[fit.m:] - residual[:fit.m]
    identity_error = np.mean(d**2) - (np.mean(residual[fit.m:]**2)
        + np.mean(residual[:fit.m]**2) - 2 * np.mean(residual[fit.m:] * residual[:fit.m]))
    assert abs(identity_error) < 1e-10
    return dict(smooth_risk=float(rank.detach()), gradient_norm=float(g.norm()),
                head_prediction_discrepancy=max_discrepancy, residual_identity_error=float(identity_error),
                head_solver=report, start_score_variance=fit_moments(raw[:a.n].detach().numpy(), a.n_cells)[1])


def timed(cfg, device, dtype, updates, joint):
    fit, val, a, b, model, denoms = setup(cfg, device, dtype)
    model_parameters = [dict(params=model.parameters(), lr=cfg["baseline_lr"])]
    head = None
    if joint:
        z, _, _ = scores(model, a, b, cfg)
        z = z[:a.n].cpu().numpy()
        spec = HeadSpec(np.quantile(z, cfg["knot_quantiles"]), float(fit.w.mean()),
                        np.sqrt(denoms["W"]), cfg["min_slope"])
        start = time.perf_counter()
        coefs, info = fit_heads(z, fit, spec, denoms, cfg["head_max_iter"])
        refit_seconds = time.perf_counter() - start
        head = MonotoneHead(spec, coefs, dtype, device)
        model_parameters.append(dict(params=head.parameters(), lr=cfg["head_lr"]))
    else:
        info, refit_seconds = None, None
    opt = torch.optim.Adam(model_parameters)
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
    def metrics():
        with torch.no_grad():
            z, zv, (_, scale) = scores(model, a, b, cfg)
            return dict(fit_smooth=[float(pair_stats(z, a, tau)) for tau in cfg["taus"]],
                        fit_actual=float(pair_stats(z, a, 0.1, actual=True)),
                        val_actual=float(pair_stats(zv, b, 0.1, actual=True)), raw_v=scale**2)
    def gradients():
        norms = []
        for tau in cfg["taus"]:
            z, _ = normalize(model(a.inputs), a, cfg["min_score_variance"])
            rank = pair_stats(z, a, tau)
            grad = torch.autograd.grad(rank, tuple(model.parameters()))
            cells = z[:a.n].reshape(a.n_cells, -1)
            wide = abs(cells[:, None, :] - cells[:, :, None]) / tau > 8
            active = a.signs != 0
            norms.append(dict(tau=tau, norm=float(torch.sqrt(sum(x.square().sum() for x in grad))),
                              active_margin_abs_gt_8=float((wide & active).sum() / active.sum())))
        return norms
    start_metrics = metrics()
    start_gradients = gradients()
    intervals = []
    for step in range(updates + 2):
        sync()
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        z, _ = normalize(model(a.inputs), a, cfg["min_score_variance"])
        parts = loss_parts(z, a, cfg["taus"][0], denoms, head)
        obj = parts["rank"] + cfg["lambda_stability"] * parts["stability"]
        if joint:
            obj = obj + 0.1 * (parts["level"] + parts["change"]) + cfg["gamma"] * parts["curvature"]
        obj.backward()
        opt.step()
        if head is not None:
            head.project()
        sync()
        if step >= 2:
            intervals.append(time.perf_counter() - start)
    return dict(n_endpoints=a.n, n_pairs=int(a.signs.count_nonzero()),
                timed_updates=updates, warmup_updates=2,
                median_seconds=float(np.median(intervals)), mean_seconds=float(np.mean(intervals)),
                start=start_metrics, end=metrics(), gradients_start=start_gradients,
                gradients_end=gradients(), refit_seconds=refit_seconds, head_solver=info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    tiny = {**cfg, "cities": 4, "tracts_per_city": 32, "test_tracts_per_city": 32}
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        runtime, dgp, seeds, contexts, fit, val, aux, _ = replication(cfg, 1)
        cutoff, panels = test_panels(cfg, runtime, dgp, seeds, contexts)
        assert len({seeds["paired"], cutoff.seed, *(p.seed for pair in panels.values() for p in pair)}) == 6
        result = dict(population=population_check(cfg, dgp),
                      invariance=invariance(fit, aux, panels["seen"][0], dgp, cfg),
                      regimes=regime_checks(cfg, dgp, panels),
                      tensor=verify_tensor(tiny, setup(tiny, "cpu", torch.float64)),
                      sample=dict(fit_tracts=fit.m, val_tracts=val.m, fit_links=len(aux),
                                  fit_stable=int((~fit.changed).sum()),
                                  val_stable=int((~val.changed).sum()),
                                  noise_cross_date_corr=float(np.corrcoef(fit.errors[:fit.m] / fit.b[:fit.m],
                                                                         fit.errors[fit.m:] / fit.b[fit.m:])[0, 1])))
        result["timing"] = {}
        for device, dtype in [("cpu", torch.float64)] + ([("cuda", torch.float32)] if torch.cuda.is_available() else []):
            for name, settings, updates, joint in [("tiny", tiny, 30, False),
                                                   ("actual_ordinal", cfg, 6, False),
                                                   ("actual_joint", cfg, 6, True)]:
                tag = f"{device}_{name}"
                print("Preflight", tag, flush=True)
                result["timing"][tag] = timed(settings, device, dtype, updates, joint)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as file:
            json.dump(result, file, indent=2)


if __name__ == "__main__":
    main()
