"""Independent structural and outcome panels, including signed exact doses."""
import numpy as np
from scipy.special import ndtr

from data import TRUE_BETA
from losses import signed_pairs, unit


def mse(x):
    return float(np.mean(x**2))


def beta_row(model, case):
    b = unit(model.beta)
    row = dict(beta_h1=b[0], beta_h2=b[1], beta_nuisance=b[2], abs_nuisance=abs(b[2]),
               angular_error_deg=np.degrees(np.arccos(np.clip(b @ TRUE_BETA, -1., 1.))),
               response_slope=b[0]*np.sqrt(1.25))
    if case == "A":
        raw = model.beta if model.city_linear is not None else model.beta*model.head[0, 1]
        row.update(raw_beta_h1=raw[0], raw_beta_h2=raw[1], raw_beta_nuisance=raw[2],
                   raw_beta_error=np.linalg.norm(raw - TRUE_BETA))
    return row


def groups(panel):
    sigma = panel.cities.sigma
    below = sigma <= np.median(sigma)
    return [("all", np.ones(len(sigma), bool)), ("lower_sigma", below), ("upper_sigma", ~below)]


def structural_row(d, predicted, cutoff, stable_panel=False):
    valid = (d != 0)
    stable = ~valid
    err = predicted - d
    row = dict(count=len(d), cutoff=cutoff, true_mean=float(d.mean()), predicted_mean=float(predicted.mean()),
               bias=float(err.mean()), latent_rmse=np.sqrt(mse(err)), zero_change_rmse=np.sqrt(mse(d)))
    if valid.any():
        row.update(sign_accuracy=float(np.mean(np.sign(predicted[valid]) == np.sign(d[valid]))),
                   sensitivity=float(np.mean(abs(predicted[valid]) > cutoff)),
                   changed_rmse=np.sqrt(mse(err[valid])), changed_bias=float(err[valid].mean()))
    if stable.any():
        row.update(stable_rms=np.sqrt(mse(predicted[stable])),
                   fresh_fpr=float(np.mean(abs(predicted[stable]) > cutoff)))
    if np.var(d) > 1e-12:
        row["individual_response_slope"] = float(np.mean((d-d.mean())*(predicted-predicted.mean()))/np.var(d))
    return row


def outcome_row(pred, mean, observed):
    r_mean, r_obs = pred-mean, pred-observed
    delta = lambda a: a[..., 1]-a[..., 0]
    d_mean, d_obs = delta(r_mean), delta(r_obs)
    noise = observed - mean
    cross = float(np.mean(r_mean[..., 0]*r_mean[..., 1]))
    d_mse = mse(d_mean)
    assert abs(d_mse - (mse(r_mean[..., 0])+mse(r_mean[..., 1])-2*cross)) < 1e-11
    return dict(expected_W_level_mse=mse(r_mean), expected_W_change_mse=d_mse,
                realized_W_level_mse=mse(r_obs), realized_W_change_mse=mse(d_obs),
                level_noise_floor=mse(noise), change_noise_floor=mse(delta(noise)),
                zero_expected_W_change_mse=mse(delta(mean)), zero_realized_W_change_mse=mse(delta(observed)),
                mean_residual_date0_mse=mse(r_mean[..., 0]), mean_residual_date1_mse=mse(r_mean[..., 1]),
                mean_residual_cross_moment=cross, mean_change_bias=float(d_mean.mean()))


def rank_cache(panel, case):
    mean, observed = panel.outcomes(case)
    i, j = np.triu_indices(panel.x.shape[1], 1)
    gap = abs(mean[:, j] - mean[:, i]) / (np.sqrt(2.) * panel.cities.sigma[:, None, None])
    return (signed_pairs(panel.x, mean), signed_pairs(panel.x, observed), float(np.mean(ndtr(-gap))))


def evaluate(rep, case, models, variants, cutoff_panel, tests, cfg):
    structural, outcome, coefficients, ranks = [], [], [], []
    caches = {where: rank_cache(panel, case) for where, (panel, _) in tests.items()}
    cut_diffs = np.diff(cutoff_panel.x, axis=2)[..., 0, :]
    for method, model in models.items():
        beta = unit(model.beta)
        cutoff = float(np.quantile(abs(cut_diffs @ beta), cfg["cutoff_quantile"], method="higher"))
        common = dict(rep=rep, case=case, method=method)
        cr = beta_row(model, case)
        if model.sigma is not None:
            truth_sigma = tests["seen"][0].cities.sigma
            cr["sigma_rmse"] = np.sqrt(mse(model.sigma-truth_sigma))
        coefficients.append(dict(**common, **cr))
        for where, (mixed, base) in tests.items():
            latent_pairs, observed_pairs, bayes = caches[where]
            def error(pairs):
                signed = pairs @ beta
                return float(np.mean(signed < 0) + .5*np.mean(signed == 0))
            ranks.append(dict(**common, where=where, latent_misrank=error(latent_pairs),
                              observed_misrank=error(observed_pairs), expected_bayes_misrank=bayes))
            panels = [("mixed", mixed)] + [(f"dose_{d:+g}", base.dose(d)) for d in cfg["test_doses"]]
            for panel_name, panel in panels:
                pred_delta = np.diff(panel.x @ beta, axis=2)[..., 0]
                group_masks = groups(panel)
                for group, mask in group_masks:
                    d, est = panel.change[mask].ravel(), pred_delta[mask].ravel()
                    structural.append(dict(**common, where=where, panel=panel_name, sigma_group=group,
                                           **structural_row(d, est, cutoff)))
                if where == "seen":
                    mean, observed = panel.outcomes(case)
                    for head, estimator in (("own", model), ("equal_library", variants[method])):
                        pred = estimator.predict(panel.x)
                        for group, mask in group_masks:
                            outcome.append(dict(**common, where=where, panel=panel_name, sigma_group=group, head=head,
                                                **outcome_row(pred[mask], mean[mask], observed[mask])))
    return dict(structural=structural, outcomes=outcome, coefficients=coefficients, ranks=ranks)
