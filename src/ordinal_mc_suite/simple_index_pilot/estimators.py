"""Predeclared comparison arms and a separate observed-validation calibration draw."""
from itertools import product

import numpy as np

from classical import Model, city_slopes, conditional_ml, fixed_effects, mixed_model
from losses import (cardinal, fit_head, fit_index, head_prediction, smooth_rank, spherical_grid,
                    stable, unit, validation)


def library(data, cfg, tuned=None):
    ratios = list(cfg["level_change_ratios"])
    if tuned is not None:
        w, d = tuned["full"]
        ratios.extend([d / w, tuned["level_change_ratio"]])
    opts = {"L": (1., 0., 0.)}
    for r in dict.fromkeys(ratios):
        opts[f"LD_{r:.12g}"] = (1., r, 0.)
    if data.case == "C":
        gamma = cfg["curvature_per_level"]
        opts.update({key.replace("L", "LC", 1): (w, d, gamma) for key, (w, d, _) in list(opts.items())})
    return opts


def choose_head(beta, fit, valid, cfg, options):
    beta = unit(beta)
    records = []
    for key, weights in options.items():
        head = fit_head(beta, fit, *weights)
        values = validation(beta, head_prediction(beta, head, valid.x), valid, cfg)
        records.append((key, head, values))
    min_score = min(row[2]["composite"] for row in records)
    selected = next(row for row in records if row[2]["composite"] <= min_score + 1e-8)
    return Model(beta, selected[1]), selected[0], [(key, score) for key, _, score in records]


def optimize(fit, valid, cfg, weights, start, grid, rank_cache, options):
    beta, info = fit_index(fit, weights, cfg, start, grid, rank_cache)
    w, d, gamma = weights[2:]
    if w or d:
        model = Model(beta, fit_head(beta, fit, w, d, gamma))
        info["head_option"] = "own"
    else:
        model, name, _ = choose_head(beta, fit, valid, cfg, options)
        info["head_option"] = name
    return model, info, validation(beta, model.predict(valid.x), valid, cfg)


def ranked_grid(fit, cfg):
    grid = spherical_grid(cfg["sphere_grid"])
    return grid, [smooth_rank(fit.pairs, beta, cfg["tau"])[0] for beta in grid]


def projected(g, beta):
    return g - np.dot(g, beta) * beta


def tune(fit, valid, cfg):
    start = unit(fixed_effects(fit).beta)
    grid, cache = ranked_grid(fit, cfg)
    options = library(fit, cfg)
    s = cfg["lambda_S"]
    anchor, info, v = optimize(fit, valid, cfg, (1., s, 0., 0., 0.), start, grid, cache, options)
    b = anchor.beta
    _, gr = smooth_rank(fit.pairs, b, cfg["tau"])
    _, gs = stable(b, fit)
    head0 = fit_head(b, fit, 1., 1.)
    _, (gw, gd) = cardinal(b, head0, fit)
    norms = [float(np.linalg.norm(projected(g, b))) for g in (gr, s*gs, gw, gd)]
    g_ref = np.hypot(norms[0], norms[1])
    if min(*norms[2:], g_ref) < 1e-9:
        raise ArithmeticError(f"Degenerate gradient scaling on calibration draw: {norms}")
    scale_w, scale_d = float(g_ref/norms[2]), float(g_ref/norms[3])
    rows = [dict(candidate="rW0_rD0", rW=0., rD=0., rank_weight=1., level_weight=0., change_weight=0.,
                 **v, **info)]
    budgets = list(product(cfg["gradient_budgets"], repeat=2)) + cfg["additional_gradient_budgets"]
    for rw, rd in budgets:
        if rw == rd == 0:
            continue
        w, d = rw*scale_w, rd*scale_d
        _, info, v = optimize(fit, valid, cfg, (1., s, w, d, 0.), start, grid, cache, options)
        rows.append(dict(candidate=f"rW{rw:g}_rD{rd:g}", rW=rw, rD=rd, rank_weight=1.,
                         level_weight=w, change_weight=d, **v, **info))
    for r in cfg["level_change_ratios"]:
        _, info, v = optimize(fit, valid, cfg, (0., s, 1., r, 0.), start, grid, cache, options)
        rows.append(dict(candidate=f"L2_delta_{r:g}", rW=None, rD=None, rank_weight=0.,
                         level_weight=1., change_weight=r, **v, **info))
    def best(values, budget):
        minimum = min(row["composite"] for row in values)
        tied = [row for row in values if row["composite"] <= minimum + 1e-8]
        return min(tied, key=lambda row: (budget(row), row["candidate"]))
    grid_rows = [row for row in rows if row["rank_weight"]]
    global_choice = best(grid_rows, lambda x: x["rW"] + x["rD"])
    level = best([x for x in grid_rows if x["rW"] and x["rD"] == 0], lambda x: x["rW"])
    full = best([x for x in grid_rows if x["rW"] and x["rD"]], lambda x: x["rW"] + x["rD"])
    l2_delta = best([x for x in rows if x["rank_weight"] == 0], lambda x: x["change_weight"])
    selected = dict(level=level["level_weight"], full=[full["level_weight"], full["change_weight"]],
                    level_change_ratio=l2_delta["change_weight"], grid_winner=global_choice["candidate"],
                    grid_winner_score=global_choice["composite"], level_candidate=level["candidate"],
                    full_candidate=full["candidate"], l2_delta_candidate=l2_delta["candidate"],
                    gradient_reference=float(g_ref), gradient_scale_w=scale_w, gradient_scale_d=scale_d,
                    gradient_norms=dict(zip(("rank", "weighted_stability", "level", "change"), norms)))
    return selected, rows


def estimators(fit, valid, cfg, tuned):
    fe = fixed_effects(fit)
    cs = city_slopes(fit)
    if fit.case == "A":
        normalized = unit(fe.beta)
        np.testing.assert_allclose(head_prediction(normalized, fit_head(normalized, fit), fit.x),
                                   fe.predict(fit.x), atol=2e-12, rtol=2e-12)
    models = {"FE": fe, "City_slopes": cs}
    audits = {}
    for method, spec in (("Mixed_index", "index"), ("Mixed_diagonal", "diagonal")):
        models[method], audits[method] = mixed_model(fit, cs, spec)
    s = cfg["lambda_S"]
    grid, cache = ranked_grid(fit, cfg)
    for method, weight in (("Gaussian_ML", 0.), ("Gaussian_ML_S", s)):
        models[method], audits[method] = conditional_ml(fit, weight, fe.beta, cfg, grid)
    w, d = tuned["full"]
    wl = tuned["level"]
    arms = {"Ordinal_S": (1., s, 0., 0., 0.),
            "L2_S": (0., s, 1., 0., 0.),
            "L2_D_S": (0., s, 1., tuned["level_change_ratio"], 0.),
            "Ordinal_L2_S": (1., s, wl, 0., 0.),
            "Ordinal_L2_D_S": (1., s, w, d, 0.)}
    if fit.case == "C":
        g = cfg["curvature_per_level"]
        arms.update(Ordinal_L2_S_C=(1., s, wl, 0., g*wl),
                    Ordinal_L2_D_S_C=(1., s, w, d, g*w))
    opts = library(fit, cfg, tuned)
    for method, weights in arms.items():
        models[method], audits[method], _ = optimize(fit, valid, cfg, weights, unit(fe.beta), grid, cache, opts)
        audits[method]["training_weights"] = list(weights)
    variants, values = {}, {}
    for name, model in models.items():
        fixed_calibrated, chosen, scores = choose_head(model.beta, fit, valid, cfg, opts)
        variants[name] = fixed_calibrated
        values[name] = dict(own=validation(model.beta, model.predict(valid.x), valid, cfg),
                            common_library=validation(fixed_calibrated.beta, fixed_calibrated.predict(valid.x), valid, cfg),
                            library_choice=chosen, library_scores=dict(scores))
    return models, variants, values, audits
