"""Matched all-pair versus oracle changed-only outcome-difference supervision."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from calibrate_10pct import (array_sha, calibrate, city_rows, outcome_rows, panels, read,
                             sha, structural_rows, write)
from classical import Model, fixed_effects
from data import CASES, TRUE_BETA
from losses import (SLOPE_FLOOR, angles, cardinal, city_center, city_mean, fit_head, fit_index,
                    head_prediction, minimize_sphere, smooth_at, smooth_rank, sphere, spherical_grid,
                    stable, unit, validation)
from run_simple import case_data, create_data, json_numpy


ROOT = Path(__file__).resolve().parent
ALL = "Ordinal_L2_D_all_matched"
CHANGED = "Ordinal_L2_D_changed_matched"
ANCHOR = "Ordinal_L2_S"
TARGETS = ("f", "W_conditional_mean", "delta_f", "delta_W_conditional_mean",
           "W_realized", "delta_W_realized", "delta_f_stable", "delta_f_changed")
CONTRASTS = ((CHANGED, ALL), (ALL, ANCHOR), (CHANGED, ANCHOR),
             (ALL, "Ordinal_L2_D_S"), (CHANGED, "Ordinal_L2_D_S"))


def fit_masked_head(beta, data, level, change, mask):
    if not np.array_equal(mask, mask.astype(bool)):
        raise ValueError("Difference masks must be binary")
    mask = np.asarray(mask)[..., None]
    if mask.shape != data.w.shape[:2] + (1,):
        raise ValueError("Mask must index paired tracts")
    r, w = data.x @ beta, data.w
    def inner(x, y):
        return (level / data.denom_w * city_mean(city_center(x) * city_center(y))
                + change / data.denom_d * np.mean(mask*np.diff(x, axis=2)*np.diff(y, axis=2), axis=(1, 2)))
    a11, t1 = inner(r, r), inner(r, w)
    if data.case == "A":
        u = np.full(len(w), max(float(t1.sum()/a11.sum()), SLOPE_FLOOR))
        v = np.zeros_like(u)
    elif data.case == "B":
        u, v = np.maximum(t1/a11, SLOPE_FLOOR), np.zeros_like(t1)
    elif data.case == "C":
        r3 = r**3
        a12, a22, t2 = inner(r, r3), inner(r3, r3), inner(r3, w)
        determinant = a11*a22-a12**2
        if np.min(determinant) <= 0:
            raise ArithmeticError("Singular masked cubic head")
        uf, vf = (t1*a22-t2*a12)/determinant, (t2*a11-t1*a12)/determinant
        candidates = np.stack((np.column_stack((uf, vf)),
                               np.column_stack((np.maximum(t1/a11, SLOPE_FLOOR), np.zeros_like(t1))),
                               np.column_stack((np.full_like(t1, SLOPE_FLOOR),
                                                np.maximum((t2-a12*SLOPE_FLOOR)/a22, 0.)))), axis=1)
        uu, vv = candidates[..., 0], candidates[..., 1]
        objective = (a11[:, None]*uu**2 + 2*a12[:, None]*uu*vv + a22[:, None]*vv**2
                     -2*t1[:, None]*uu-2*t2[:, None]*vv)
        objective[(uu < SLOPE_FLOOR) | (vv < 0)] = np.inf
        u, v = candidates[np.arange(len(w)), np.argmin(objective, axis=1)].T
    else:
        raise ValueError(data.case)
    a = city_mean(w)-u*city_mean(r)-v*city_mean(r**3)
    return np.column_stack((a, u, v))


def masked_cardinal(beta, head, data, mask):
    r = data.x @ beta
    residual = head_prediction(beta, head, data.x)-data.w
    dr = np.diff(residual, axis=2)
    m = np.asarray(mask)[..., None]
    deriv = (head[:, 1, None, None] + 3*head[:, 2, None, None]*r**2)[..., None] * data.x
    dd = np.diff(deriv, axis=2)
    lw, ld = np.mean(residual**2)/data.denom_w, np.mean(m*dr**2)/data.denom_d
    gw = 2*np.mean(residual[..., None]*deriv, axis=(0, 1, 2))/data.denom_w
    gd = 2*np.mean(m[..., None]*dr[..., None]*dd, axis=(0, 1, 2))/data.denom_d
    return (float(lw), float(ld)), (gw, gd)


def masked_objective(beta, data, weights, cfg, mask, rank_cache=None):
    rank, gr = (smooth_rank(data.pairs, beta, cfg["tau"]) if rank_cache is None
                else (rank_cache, np.zeros(3)))
    stability, gs = stable(beta, data)
    head = fit_masked_head(beta, data, weights[2], weights[3], mask)
    (lw, ld), (gw, gd) = masked_cardinal(beta, head, data, mask)
    return float(weights[0]*rank + weights[1]*stability + weights[2]*lw + weights[3]*ld), (
        weights[0]*gr + weights[1]*gs + weights[2]*gw + weights[3]*gd)


def fit_changed(data, cfg, weights, start, grid, cache, mask):
    values = [masked_objective(beta, data, weights, cfg, mask, rank_cache=r)[0]
              for beta, r in zip(grid, cache)]
    return minimize_sphere(lambda beta: masked_objective(beta, data, weights, cfg, mask),
                           [start, -start], grid, values, cfg["optimizer_maxiter"])


def numerical_checks(data, cfg, weights, beta, mask):
    ones, zeros = np.ones_like(mask), np.zeros_like(mask)
    original_head = fit_head(beta, data, weights[2], weights[3], 0.)
    all_head = fit_masked_head(beta, data, weights[2], weights[3], ones)
    np.testing.assert_array_equal(all_head, original_head)
    old_value, old_grad = smooth_at(beta, data, weights, cfg)
    new_value, new_grad = masked_objective(beta, data, weights, cfg, ones)
    np.testing.assert_allclose(new_value, old_value, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(new_grad, old_grad, atol=1e-13, rtol=1e-13)
    zero_head = fit_masked_head(beta, data, weights[2], weights[3], zeros)
    np.testing.assert_array_equal(zero_head, fit_head(beta, data, weights[2], 0., 0.))
    (_, zero_change), (_, zero_gradient) = masked_cardinal(beta, zero_head, data, zeros)
    assert zero_change == 0 and np.array_equal(zero_gradient, np.zeros(3))
    direction = np.array([.6, .8])
    x = angles(beta)
    _, jac = sphere(x)
    _, grad = masked_objective(beta, data, weights, cfg, mask)
    analytic = float((jac@grad)@direction)
    h = 1e-5
    endpoints = [masked_objective(sphere(x+sign*h*direction)[0], data, weights, cfg, mask)[0] for sign in (-1, 1)]
    finite = (endpoints[1]-endpoints[0])/(2*h)
    error = abs(analytic-finite)
    assert error < 2e-7, (data.case, analytic, finite)
    return dict(all_head_exact=True, all_objective_error=abs(new_value-old_value),
                all_gradient_error=float(np.max(abs(new_grad-old_grad))),
                zero_mask_change_and_gradient_zero=True, directional_finite_difference_error=error,
                changed_fraction=float(mask.mean()))


def head_kkt(beta, head, data, weights, mask):
    r = data.x@beta
    residual = head_prediction(beta, head, data.x)-data.w
    dr = np.diff(residual, axis=2)
    def gradient(feature):
        return (2*weights[2]/data.denom_w*city_mean(residual*feature)
                + 2*weights[3]/data.denom_d*np.mean(mask[..., None]*dr*np.diff(feature, axis=2), axis=(1,2)))
    intercept = 2*weights[2]*city_mean(residual)/data.denom_w
    linear = gradient(r)
    slopes = head[:, 1]
    if data.case == "A":
        linear, slopes = linear.mean(keepdims=True), slopes[:1]
    linear = np.where(slopes <= SLOPE_FLOOR+1e-10, np.minimum(linear, 0.), linear)
    gradients = [intercept, linear]
    if data.case == "C":
        cubic = gradient(r**3)
        gradients.append(np.where(head[:, 2] <= 1e-10, np.minimum(cubic, 0.), cubic))
    maximum = float(np.max(abs(np.concatenate(gradients))))
    assert maximum < 2e-9
    contributions = mask*np.diff(residual, axis=2)[..., 0]**2
    assert np.all(contributions[~mask] == 0)
    return maximum


def summarize(city, old_calibration, out):
    base_city = pd.read_csv(old_calibration / "city_metrics.csv")
    combined = pd.concat((base_city, pd.DataFrame(city)), ignore_index=True)
    combined.to_csv(out / "city_metrics.csv", index=False)
    keys = ["rep", "case", "method", "panel", "target"]
    reps = combined.groupby(keys).mse.agg(city_count="size", city_mean_mse="mean", city_sd_mse="std",
                                         city_median_mse="median", city_max_mse="max").reset_index()
    assert reps.city_count.eq(20).all()
    reps.to_csv(out / "per_rep_city_summary.csv", index=False)
    keys = ["case", "method", "panel", "target"]
    summary = reps.groupby(keys).agg(
        replication_count=("rep", "size"), mean_city_mse=("city_mean_mse", "mean"),
        mean_within_rep_city_sd=("city_sd_mse", "mean"),
        mean_within_rep_city_median=("city_median_mse", "mean"),
        mean_within_rep_city_max=("city_max_mse", "mean"),
        min_rep_mean_mse=("city_mean_mse", "min"), max_rep_mean_mse=("city_mean_mse", "max")).reset_index()
    assert summary.replication_count.eq(5).all()
    summary.to_csv(out / "five_rep_summary.csv", index=False)
    paired = []
    for a, b in CONTRASTS:
        index = ["rep", "case", "panel", "target"]
        aa, bb = (reps[reps.method.eq(method)].set_index(index).city_mean_mse for method in (a,b))
        paired.append((aa-bb).dropna().rename("difference").reset_index().assign(method=a, reference=b))
    pair = pd.concat(paired, ignore_index=True)
    pair.to_csv(out / "paired_by_rep.csv", index=False)
    stats = pair.groupby(["case","method","reference","panel","target"]).difference.agg(
        replication_count="size", mean_difference="mean", minimum="min", maximum="max",
        lower=lambda x: (x<0).sum(), higher=lambda x: (x>0).sum(), ties=lambda x: (x==0).sum()).reset_index()
    assert stats.replication_count.eq(5).all()
    stats.to_csv(out / "paired_summary.csv", index=False)


def run(source, old_calibration, out):
    before = {path: sha(path.read_bytes()) for path in source.glob("rep_*/case_*_models.json")}
    manifest = read(source / "manifest.json")
    cfg, selection = manifest["config"], read(source / "frozen_selection.json")["weights"]
    cal_manifest = read(old_calibration / "manifest.json")
    assert sha((source / "manifest.json").read_bytes()) == cal_manifest["original_manifest_sha256"]
    for name in ("data.py","losses.py","classical.py","run_simple.py"):
        assert sha((ROOT/name).read_bytes()) == manifest["source_sha256"][name]
    all_weights = {case: [1., cfg["lambda_S"], selection[case]["level"], selection[case]["full"][1], 0.]
                   for case in CASES}
    write(out / "manifest.json", dict(
        config=cfg, training_weights=all_weights, new_scorer_fits=30,
        source_run_manifest_sha256=sha((source/"manifest.json").read_bytes()),
        shared_calibration_manifest_sha256=sha((old_calibration/"manifest.json").read_bytes()),
        script_sha256=sha(Path(__file__).read_bytes()), source_model_sha256={str(p):h for p,h in before.items()},
        changed_status="oracle simulation true_change != 0, never observed outcome threshold",
        masked_loss="mean over ALL fitting pairs of changed_mask*(predicted_delta_W-observed_delta_W)^2, "
                    "divided by original all-pair fitting delta_W variance",
        calibration="same saved seed namespace, 28 fitting pairs and 252 disjoint test pairs per city, "
                    "unpenalized least squares with same A/B/C families"))
    rows, heads, fit_audits, numerics, validation_rows = [], [], [], [], []
    old_heads = read(old_calibration/"calibrated_heads.json")
    initial_seeds = set(read(source/"frozen_selection.json")["fitting_and_validation_seeds"].values())
    for rep in cfg["replications"]:
        initial_seeds.update(read(source/f"rep_{rep}/design.json")["seeds"].values())
    used_seeds = initial_seeds.copy()
    started = time.perf_counter()
    for rep in cfg["replications"]:
        dest = out/f"rep_{rep}"
        dest.mkdir()
        seeds, panel, valid, links, val_links = create_data(cfg, rep)
        role_panels, design = panels(cfg, rep, panel, seeds["paired"], used_seeds)
        for tag in ("seen", "unseen"):
            for field in ("panel_seed","split_seed","calibration_ids_sha256","evaluation_ids_sha256",
                          "calibration_features_sha256","evaluation_features_sha256"):
                assert design[tag][field] == cal_manifest["panels"][str(rep)][tag][field]
        fitted = {}
        for case in CASES:
            fit, validation_data = case_data(case, panel, valid, links, val_links)
            weights = all_weights[case]
            mask = panel.change != 0
            assert mask.dtype == bool and 0 < mask.mean() < 1
            start = unit(fixed_effects(fit).beta)
            grid = spherical_grid(cfg["sphere_grid"])
            cache = [smooth_rank(fit.pairs, b, cfg["tau"])[0] for b in grid]
            check = numerical_checks(fit, cfg, weights, start, mask)
            numerics.append(dict(rep=rep,case=case,**check))
            original_anchor = read(source/f"rep_{rep}/case_{case}_models.json")[ANCHOR]["own"]
            anchor_beta = np.asarray(original_anchor["beta"])
            baseline_value, baseline_gradient = smooth_at(anchor_beta, fit, (1.,cfg["lambda_S"],weights[2],0.,0.),cfg)
            assert np.linalg.norm(baseline_gradient-anchor_beta*np.dot(baseline_gradient,anchor_beta)) < 1e-4
            for method in (ALL, CHANGED):
                if method == ALL:
                    beta, info = fit_index(fit, weights, cfg, start, grid, cache)
                    native = fit_head(beta, fit, weights[2], weights[3],0.)
                    active_mask = np.ones_like(mask)
                else:
                    beta, info = fit_changed(fit,cfg,weights,start,grid,cache,mask)
                    native = fit_masked_head(beta,fit,weights[2],weights[3],mask)
                    active_mask = mask
                kkt = head_kkt(beta,native,fit,weights,active_mask)
                model = Model(beta,native)
                val = validation(beta,model.predict(valid.x),validation_data,cfg)
                validation_rows.append(dict(rep=rep,case=case,method=method,**val))
                fit_audits.append(dict(rep=rep,case=case,method=method,changed_fraction=float(mask.mean()),
                                      delta_denominator=fit.denom_d,max_conditional_head_gradient=kkt,**info))
                fitted[(case,method)] = (model,unit(beta))
                structural_rows(rows,dict(rep=rep,case=case,method=method,panel="original_score_fit"),
                                panel,panel.x@unit(beta))
            print(f"Fitted matched pair {rep}/{case}",flush=True)
        for case in CASES:
            for role,(cal,test) in role_panels.items():
                observed_cal = cal.outcomes(case)[1]
                expected_sha = next(x for x in old_heads if x["rep"]==rep and x["case"]==case and x["panel"]==role)
                assert array_sha(cal.x)==expected_sha["calibration_features_sha256"]
                assert array_sha(observed_cal)==expected_sha["calibration_outcome_sha256"]
                fitted_heads = {}
                for method in (ALL,CHANGED):
                    _,beta = fitted[(case,method)]
                    before_score = test.x@beta
                    head,kkt=calibrate(beta,cal.x,observed_cal,case)
                    np.testing.assert_array_equal(test.x@beta,before_score)
                    fitted_heads[method] = head
                    heads.append(dict(rep=rep,case=case,method=method,panel=role,beta=beta.tolist(),head=head.tolist(),
                                      calibration_gradient=kkt,calibration_outcome_sha256=array_sha(observed_cal)))
                mean,observed = test.outcomes(case)
                for method in (ALL,CHANGED):
                    _,beta=fitted[(case,method)]
                    prefix=dict(rep=rep,case=case,method=method,panel=role)
                    structural_rows(rows,prefix,test,test.x@beta)
                    outcome_rows(rows,prefix,head_prediction(beta,fitted_heads[method],test.x),mean,observed)
        saved={f"{case}/{method}":asdict(value[0]) for (case,method),value in fitted.items()}
        write(dest/"native_models.json",json.loads(json.dumps(saved,default=json_numpy)))
    for path,digest in before.items():
        assert sha(path.read_bytes())==digest
    pd.DataFrame(fit_audits).to_csv(out/"optimizer.csv",index=False)
    pd.DataFrame(numerics).to_csv(out/"numerical_checks.csv",index=False)
    pd.DataFrame(validation_rows).to_csv(out/"validation.csv",index=False)
    write(out/"calibrated_heads.json",heads)
    pd.DataFrame(rows).to_csv(out/"new_city_metrics.csv",index=False)
    summarize(rows,old_calibration,out)
    assert len(fit_audits)==30 and len(heads)==60
    assert all(x["all_head_exact"] for x in numerics)
    write(out/"audit.json",dict(
        passed=True,new_scorer_fits=30,head_calibration_sets=60,score_freeze_checks=60,
        all_original_model_files_unchanged=True,shared_calibration_and_test_hashes_match=True,
        oracle_mask_matches_true_change=True,stable_delta_contributions_zero=True,
        all_pair_denominators_unchanged=True,
        max_head_gradient=max(x["max_conditional_head_gradient"] for x in fit_audits),
        max_tangent_gradient=max(x["min_tangent_grad"] for x in fit_audits),
        max_finite_difference_error=max(x["directional_finite_difference_error"] for x in numerics),
        calibration_features_and_outcomes_identical_to_prior=True,elapsed_seconds=time.perf_counter()-started))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--calibration",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:
        with threadpool_limits(limits=1):
            run(args.run,args.calibration,args.out)
    except Exception:
        (args.out/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    main()
