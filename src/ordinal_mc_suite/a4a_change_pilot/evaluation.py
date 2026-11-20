"""Held-out ranks, tract changes, and outcome residual decompositions."""
import numpy as np
import torch

from ordinal_mc.e6.data import measurement

from data import counterfactual
from fitting import new_scorer


def delta(x):
    return np.diff(np.asarray(x).reshape(2, -1), axis=0).ravel()


def rms(x):
    return float(np.sqrt(np.mean(x**2)))


def ranking(s, target, n_cells):
    x = s.reshape(n_cells, -1)
    y = target.reshape(n_cells, -1)
    observed = np.sign(y[:, None, :] - y[:, :, None])
    predicted = np.sign(x[:, None, :] - x[:, :, None])
    errors = (observed * predicted < 0).sum() + 0.5 * ((predicted == 0) & (observed != 0)).sum()
    return float(errors / (x.size * (x.shape[1] - 1)))


def structural(true, pred, cutoff, indices, kind, fit_sd):
    f, z = true[indices], pred[indices]
    if not len(z):
        return dict(count=0)
    var = np.var(f)
    var_z = np.var(z)
    covariance = np.mean((f - f.mean()) * (z - z.mean()))
    changed = kind == "changed"
    error = z - f
    return dict(count=len(z), truth_mean=float(f.mean()), prediction_mean=float(z.mean()),
                bias=float(error.mean()), rmse=rms(error), rms_score_movement=rms(z),
                normalization_only_rmse=rms(f / fit_sd - f), fit_normalized_truth_rmse=rms(z - f / fit_sd),
                slope=float(covariance / var) if var > 0 else np.nan,
                correlation=float(covariance / np.sqrt(var * var_z)) if min(var, var_z) > 0 else np.nan,
                sign=float(np.mean(np.sign(z) == np.sign(f))) if changed else np.nan,
                sensitivity=float(np.mean(abs(z) > cutoff)) if changed else np.nan,
                false_positive=float(np.mean(abs(z) > cutoff)) if kind == "stable" else np.nan,
                cutoff=cutoff)


def outcome(pred, observed, mean, indices):
    n = len(observed) // 2
    endpoint = np.r_[indices, n + indices]
    error = pred[endpoint] - mean[endpoint]
    obs_residual = pred[endpoint] - observed[endpoint]
    noise = observed[endpoint] - mean[endpoint]
    d_error = delta(error)
    d_obs_residual = delta(obs_residual)
    d_noise = delta(noise)
    e0, e1 = error.reshape(2, -1)
    mean_square0, mean_square1 = np.mean(e0**2), np.mean(e1**2)
    cross = np.mean(e0 * e1)
    change_mean_mse = np.mean(d_error**2)
    if not np.isclose(change_mean_mse, mean_square0 + mean_square1 - 2 * cross, atol=1e-11):
        raise ArithmeticError("Temporal residual decomposition failed")
    return dict(count=len(indices), mean_level_mse=float(np.mean(error**2)),
                realized_level_mse=float(np.mean(obs_residual**2)), mean_change_mse=float(change_mean_mse),
                realized_change_mse=float(np.mean(d_obs_residual**2)),
                mean_level_noise_floor=float(np.mean(noise**2)), mean_change_noise_floor=float(np.mean(d_noise**2)),
                realized_change_cross_term=float(-2 * np.mean(d_error * d_noise)),
                zero_mean_change_mse=float(np.mean(delta(mean[endpoint])**2)),
                zero_realized_change_mse=float(np.mean(delta(observed[endpoint])**2)),
                mean_change_bias=float(d_error.mean()), mean_residual_date0_mse=float(mean_square0),
                mean_residual_date1_mse=float(mean_square1), mean_residual_cross_moment=float(cross),
                mean_residual_covariance=float(np.mean((e0 - e0.mean()) * (e1 - e1.mean()))),
                mean_change_sign=float(np.mean(np.sign(delta(pred[endpoint])) == np.sign(delta(mean[endpoint])))),
                realized_change_sign=float(np.mean(np.sign(delta(pred[endpoint])) == np.sign(delta(observed[endpoint])))))


@torch.no_grad()
def evaluator(record, model, features):
    model.load_state_dict(record["model"])
    parameter = next(model.parameters())
    @torch.no_grad()
    def predict(x):
        features_x = torch.as_tensor(features(x), dtype=parameter.dtype, device=parameter.device)
        return ((model(features_x) - record["score_mu"]) / record["score_sd"]).cpu().double().numpy()
    return predict


def append_structural(rows, true, pred, cutoff, masks, prefix, fit_sd):
    for stratum, kind, mask in masks:
        rows.append(dict(**prefix, stratum=stratum, kind=kind,
                         **structural(true, pred, cutoff, np.flatnonzero(mask), kind, fit_sd)))


def supported_masks(p, true, cfg):
    relevant = abs(true) > 0.05
    masks = [("all", "all", np.ones(p.m, bool)), ("assigned_stable", "stable", ~p.changed),
             ("actual_abs_gt_.05", "changed", relevant),
             ("increase", "changed", relevant & (true > 0)),
             ("decline", "changed", relevant & (true < 0))]
    bins = cfg["change_bins"][1:] + [np.inf]
    for lo, hi in zip(bins, bins[1:]):
        for direction, direction_mask in [("both", np.ones(p.m, bool)),
                                          ("increase", true > 0), ("decline", true < 0)]:
            name = f"abs_{lo:g}_to_{hi:g}_{direction}"
            masks.append((name, "changed", (abs(true) > lo) & (abs(true) <= hi) & direction_mask))
    tertile = np.searchsorted(np.quantile(p.f[:p.m], [1 / 3, 2 / 3]), p.f[:p.m])
    for k in range(3):
        masks.append((f"start_tertile_{k + 1}", "changed", relevant & (tertile == k)))
        for lo, hi in zip(bins, bins[1:]):
            masks.append((f"start_tertile_{k + 1}_abs_{lo:g}_to_{hi:g}", "changed",
                          (tertile == k) & (abs(true) > lo) & (abs(true) <= hi)))
    return masks


def evaluation(record, model, features, fit, cutoff_panel, panels, spec, dgp, cfg,
               prefix, tables, known_units=None, compact=False):
    predict = evaluator(record, model, features)
    cutoff = float(np.quantile(abs(delta(predict(cutoff_panel.x))), cfg["cutoff_quantile"], method="higher"))
    _, fit_var = (float(fit.f.mean()), float(fit.f.reshape(fit.n_cells, -1).var(axis=1, ddof=1).mean()))
    fit_sd = np.sqrt(fit_var)
    modes = [("common", record["calibrated"]), ("native", record["native"])]
    if known_units is not None:
        modes = [("known", None)]
    def head_predictions(mode, coefficients, z, panel):
        if mode == "known":
            return measurement(known_units[0] + known_units[1] * z, panel.a, panel.b, panel.k, "nonlinear")
        return spec.direct(coefficients, z, panel.cells)
    for where, (p, stable) in panels.items():
        group = dict(**prefix, where=where)
        z = predict(p.x)
        df, dz = delta(p.f), delta(z)
        noise = delta(p.errors)
        tables["ranks"].append(dict(**group, observed_misranking=ranking(z, p.w, p.n_cells),
                                    truth_misranking=ranking(z, p.f, p.n_cells),
                                    bayes_reference_realized_misranking=ranking(p.f, p.w, p.n_cells),
                                    fit_true_f_sd=fit_sd, cutoff=cutoff,
                                    stable_calibration_count=cutoff_panel.m,
                                    pair_error_noise_sd=float(np.std(noise))))
        masks = supported_masks(p, df, cfg)
        append_structural(tables["structural"], df, dz, cutoff, masks,
                          dict(**group, panel="supported_mixed"), fit_sd)
        if where == "seen":
            for mode, coefs in modes:
                if mode != "known" and coefs is None:
                    continue
                preds = head_predictions(mode, coefs, z, p)
                for name, _, mask in masks[:5]:
                    if mask.any():
                        tables["outcome"].append(dict(**group, panel="supported_mixed", head=mode, stratum=name,
                                                      **outcome(preds, p.w, p.mean_w, np.flatnonzero(mask))))
        if compact:
            continue
        n = stable.m
        base = (stable.a[:n], stable.b[:n], stable.k[:n])
        later = (stable.a[n:], stable.b[n:], stable.k[n:])
        f0 = stable.f[:n]
        z0, z_same = predict(stable.x).reshape(2, -1)
        signs = np.where(np.arange(n) % 2, 1, -1)
        x_large, f_large = counterfactual(stable, 0.4 * signs, dgp)
        z_large = predict(x_large)
        w0 = measurement(f0, *base, "nonlinear")
        w_map = measurement(f0, *later, "nonlinear")
        w_structure = measurement(f_large, *base, "nonlinear")
        w_opposed = w_structure - 2 * stable.b[:n] * np.sign(f_large - f0)
        panels_for_change = [("acquisition_only", f0, z_same, w0),
                             ("map_only", f0, z_same, w_map),
                             ("structural_only", f_large, z_large, w_structure),
                             ("opposing_shifted_map", f_large, z_large, w_opposed)]
        for panel_name, f1, z1, w1 in panels_for_change:
            truth, pred, mean_change = f1 - f0, z1 - z0, w1 - w0
            zero = abs(truth) <= 0.05
            if panel_name in ("acquisition_only", "map_only"):
                regimes = [("all", "stable", np.ones(n, bool))]
            else:
                regimes = [("all", "all", np.ones(n, bool)), ("actual_abs_gt_.05", "changed", ~zero),
                           ("increase", "changed", (truth > 0) & ~zero),
                           ("decline", "changed", (truth < 0) & ~zero)]
            append_structural(tables["structural"], truth, pred, cutoff, regimes,
                              dict(**group, panel=panel_name), fit_sd)
            tables["regimes"].append(dict(**group, panel=panel_name, count=n,
                                          mean_true_abs_change=float(np.mean(abs(truth))),
                                          mean_expected_w_abs_change=float(np.mean(abs(mean_change))),
                                          score_expected_w_direction_agreement=float(np.mean(np.sign(pred) == np.sign(mean_change))),
                                          true_opposition_rate=float(np.mean(truth * mean_change < 0))))
        x_small, f_small = counterfactual(stable, 0.15 * signs, dgp)
        z_small = predict(x_small)
        w_small = measurement(f_small, *later, "nonlinear")
        opposition = ((f_small - f0) * (w_small - w0) < 0) & (abs(f_small - f0) > 0.05)
        subsets = [("all", "all", np.ones(n, bool)), ("opposed", "changed", opposition)]
        append_structural(tables["structural"], f_small - f0, z_small - z0, cutoff, subsets,
                          dict(**group, panel="supported_small_mixed"), fit_sd)
        if where == "seen":
            # The map is supported by training even though these paired H movements are new.
            mean = np.r_[w0, w_small]
            realized = mean + stable.errors
            z_both = np.r_[z0, z_small]
            for mode, coefs in modes:
                if mode != "known" and coefs is None:
                    continue
                pred_w = head_predictions(mode, coefs, z_both, stable)
                for name, _, subset in subsets:
                    if subset.any():
                        tables["outcome"].append(dict(**group, panel="supported_small_mixed", head=mode, stratum=name,
                                                      **outcome(pred_w, realized, mean, np.flatnonzero(subset))))
        for step in cfg["dose_steps"]:
            x1, f1 = counterfactual(stable, step, dgp)
            pred, truth = predict(x1) - z0, f1 - f0
            tertile = np.searchsorted(np.quantile(f0, [1 / 3, 2 / 3]), f0)
            for label, mask in [("all", np.ones(n, bool)), *((str(k + 1), tertile == k) for k in range(3))]:
                selected = pred[mask]
                quantiles = np.quantile(selected, [0.1, 0.9])
                tables["dose"].append(dict(**group, nominal_step=step, start_tertile=label, count=len(selected),
                                            predicted_mean=float(selected.mean()), true_mean=float(truth[mask].mean()),
                                            predicted_p10=float(quantiles[0]), predicted_p90=float(quantiles[1]),
                                            mean_bias=float(np.mean((pred - truth)[mask])),
                                            rmse=rms((pred - truth)[mask])))
