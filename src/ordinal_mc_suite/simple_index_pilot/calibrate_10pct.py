"""Recalibrate saved scores on matched city panels and score disjoint tracts."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from data import CASES, TRUE_BETA, draw_cities, make_panel, seed_for, training
from losses import Data, SLOPE_FLOOR, city_mean, fit_head, head_prediction, unit


ROOT = Path(__file__).resolve().parent
PANEL_TRACTS = 280
CALIBRATION_TRACTS = 28
PANEL_NAMESPACE = "simple-index-frozen-head-10pct-v1"
HOLDOUTS = ("heldout_seen", "heldout_unseen")
PRIMARY = ("f", "W_conditional_mean", "delta_f", "delta_W_conditional_mean")
SECONDARY = ("W_realized", "delta_W_realized", "delta_f_stable", "delta_f_changed")
CONTRASTS = (
    ("Ordinal_L2_S", "Ordinal_S"),
    ("Ordinal_L2_D_S", "Ordinal_L2_S"),
    ("Ordinal_L2_D_S", "Ordinal_S"),
    ("L2_D_S", "L2_S"),
    ("Ordinal_S", "Gaussian_ML_S"),
    ("Ordinal_L2_S", "Gaussian_ML_S"),
    ("Gaussian_ML_S", "Gaussian_ML"),
)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def array_sha(array):
    return sha(np.ascontiguousarray(array).tobytes())


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def city_split(panel, panel_seed, split_seed):
    rng = np.random.default_rng(split_seed)
    order = np.stack([rng.permutation(PANEL_TRACTS) for _ in panel.tract_id])
    calibration = panel.take(order[:, :CALIBRATION_TRACTS])
    evaluation = panel.take(order[:, CALIBRATION_TRACTS:])
    # Local tract IDs restart in a new panel. The seed namespaces them.
    calibration_keys = {(panel_seed, int(i)) for i in calibration.tract_id.flat}
    evaluation_keys = {(panel_seed, int(i)) for i in evaluation.tract_id.flat}
    assert not calibration_keys & evaluation_keys
    assert calibration_keys | evaluation_keys == {(panel_seed, int(i)) for i in panel.tract_id.flat}
    assert calibration.x.shape[2] == evaluation.x.shape[2] == 2
    assert 10 * calibration.x.shape[1] == panel.x.shape[1]
    return calibration, evaluation


def panels(cfg, rep, fit, seen_source_seed, used_seeds):
    base = seed_for(cfg["seed"], PANEL_NAMESPACE, rep)
    seeds = {tag: seed_for(base, tag) for tag in
             ("unseen_cities", "seen_panel", "unseen_panel", "seen_split", "unseen_split")}
    assert len(set(seeds.values())) == len(seeds)
    assert not set(seeds.values()) & used_seeds
    used_seeds.update(seeds.values())
    unseen = draw_cities(cfg["cities"], seeds["unseen_cities"])
    out, design = {}, {}
    for tag, cities in (("seen", fit.cities), ("unseen", unseen)):
        panel_seed = seeds[f"{tag}_panel"]
        full = make_panel(cities, PANEL_TRACTS, panel_seed)
        cal, test = city_split(full, panel_seed, seeds[f"{tag}_split"])
        assert panel_seed != seen_source_seed
        out[f"heldout_{tag}"] = (cal, test)
        design[tag] = dict(panel_seed=panel_seed, split_seed=seeds[f"{tag}_split"],
                           city_effect_seed=seeds["unseen_cities"] if tag == "unseen" else None,
                           calibration_ids_sha256=array_sha(cal.tract_id),
                           evaluation_ids_sha256=array_sha(test.tract_id),
                           calibration_features_sha256=array_sha(cal.x),
                           evaluation_features_sha256=array_sha(test.x),
                           calibration_tracts_per_city=cal.x.shape[1],
                           evaluation_tracts_per_city=test.x.shape[1],
                           city_count=cal.x.shape[0], calibration_dates_per_tract=cal.x.shape[2])
    return out, design


def calibrate(beta, features, observed_w, case):
    """The fitting input contains only the frozen score features and observed calibration W."""
    data = Data(x=features, w=observed_w, aux_scale=0., case=case, denom_w=1., denom_d=1.,
                pairs=np.empty((0, 3)))
    head = fit_head(beta, data, level=1., change=0., gamma=0.)
    score = features @ beta
    residual = head_prediction(beta, head, features) - observed_w
    g_intercept = 2 * city_mean(residual)
    g_linear = 2 * city_mean(residual * score)
    if case == "A":
        assert np.all(head[:, 1] == head[0, 1])
        g_linear = g_linear.mean(keepdims=True)
    slope = head[:len(g_linear), 1]
    g_linear = np.where(slope <= SLOPE_FLOOR + 1e-10, np.minimum(g_linear, 0.), g_linear)
    projected = [g_intercept, g_linear]
    if case == "C":
        g_cubic = 2 * city_mean(residual * score**3)
        projected.append(np.where(head[:, 2] <= 1e-10, np.minimum(g_cubic, 0.), g_cubic))
    else:
        assert np.all(head[:, 2] == 0.)
    assert np.all(head[:, 1] >= SLOPE_FLOOR) and np.all(head[:, 2] >= 0.)
    max_gradient = float(np.max(np.abs(np.concatenate(projected))))
    assert max_gradient < 1e-9, (case, max_gradient)
    return head, max_gradient


def city_rows(rows, prefix, target, estimate, truth, mask=None):
    for city, (p, t) in enumerate(zip(estimate, truth), start=1):
        if mask is not None:
            city_mask = mask[city - 1]
            p, t = p[city_mask], t[city_mask]
        assert p.size and p.shape == t.shape
        error = p - t
        rows.append(dict(**prefix, city=city, target=target, sample_count=p.size,
                         mean_prediction=float(np.mean(p)), mean_target=float(np.mean(t)),
                         mse=float(np.mean(error**2))))


def delta(value):
    return value[..., 1] - value[..., 0]


def structural_rows(rows, prefix, panel, score):
    city_rows(rows, prefix, "f", score, panel.x @ TRUE_BETA)
    change = delta(score)
    city_rows(rows, prefix, "delta_f", change, panel.change)
    city_rows(rows, prefix, "delta_f_stable", change, panel.change, panel.change == 0)
    city_rows(rows, prefix, "delta_f_changed", change, panel.change, panel.change != 0)


def outcome_rows(rows, prefix, prediction, mean, observed):
    city_rows(rows, prefix, "W_conditional_mean", prediction, mean)
    city_rows(rows, prefix, "delta_W_conditional_mean", delta(prediction), delta(mean))
    city_rows(rows, prefix, "W_realized", prediction, observed)
    city_rows(rows, prefix, "delta_W_realized", delta(prediction), delta(observed))


def summarize(by_city, out):
    keys = ["case", "method", "panel", "target"]
    per_rep = (by_city.groupby(["rep", *keys], sort=True).mse
               .agg(city_count="size", city_mean_mse="mean", city_sd_mse="std",
                    city_median_mse="median", city_max_mse="max").reset_index())
    assert per_rep.city_count.eq(20).all()
    summary = (per_rep.groupby(keys, sort=True)
               .agg(replication_count=("rep", "size"), mean_city_mse=("city_mean_mse", "mean"),
                    mean_within_rep_city_sd=("city_sd_mse", "mean"),
                    mean_within_rep_city_median=("city_median_mse", "mean"),
                    mean_within_rep_city_max=("city_max_mse", "mean"),
                    min_rep_mean_mse=("city_mean_mse", "min"), max_rep_mean_mse=("city_mean_mse", "max"))
               .reset_index())
    assert summary.replication_count.eq(5).all()
    by_city.to_csv(out / "city_metrics.csv", index=False)
    per_rep.to_csv(out / "per_rep_city_summary.csv", index=False)
    summary.to_csv(out / "five_rep_summary.csv", index=False)
    summary[summary.panel.eq("original_score_fit")].to_csv(out / "original_fit_structural.csv", index=False)
    for target in (*PRIMARY, *SECONDARY):
        selected = summary[summary.target.eq(target) & summary.panel.isin(HOLDOUTS)]
        metrics = ["mean_city_mse", "mean_within_rep_city_sd"]
        wide = selected.pivot(index=["case", "method"], columns="panel", values=metrics)
        wide = wide.reindex(columns=pd.MultiIndex.from_tuples([(m, p) for p in HOLDOUTS for m in metrics]))
        wide.columns = [f"{p}_{m}" for m, p in wide.columns]
        assert not wide.isna().any().any() and len(wide) == 35
        wide.to_csv(out / f"table_{target}.csv")
    paired_rows = []
    for a, b in CONTRASTS:
        index = ["rep", "case", "panel", "target"]
        left, right = (per_rep[per_rep.method.eq(m)].set_index(index).city_mean_mse for m in (a, b))
        paired_rows.append((left-right).dropna().rename("difference").reset_index().assign(method=a, reference=b))
    paired = pd.concat(paired_rows, ignore_index=True)
    paired.to_csv(out / "paired_by_rep.csv", index=False)
    counts = (paired.groupby(["case", "method", "reference", "panel", "target"]).difference
              .agg(replication_count="size", mean_difference="mean", minimum="min", maximum="max",
                   lower=lambda x: (x < 0).sum(), higher=lambda x: (x > 0).sum(), ties=lambda x: (x == 0).sum())
              .reset_index())
    assert counts.replication_count.eq(5).all()
    counts.to_csv(out / "paired_summary.csv", index=False)
    return summary


def run(source, out):
    manifest = read(source / "manifest.json")
    cfg = manifest["config"]
    reps = cfg["replications"]
    assert reps == [1, 2, 3, 4, 5] and cfg["cities"] == 20
    assert read(source / "completion.json")["fitted_estimators"] == 175
    used_seeds = set(read(source / "frozen_selection.json")["fitting_and_validation_seeds"].values())
    for rep in reps:
        old_seeds = set(read(source / f"rep_{rep}/design.json")["seeds"].values())
        assert not used_seeds & old_seeds
        used_seeds.update(old_seeds)
    old_seed_count = len(used_seeds)
    for name in ("data.py", "losses.py"):
        assert sha((ROOT / name).read_bytes()) == manifest["source_sha256"][name]
    city, oracle, heads, designs, sources = [], [], [], {}, {}
    for rep in reps:
        original = source / f"rep_{rep}"
        original_design = read(original / "design.json")
        original_seeds, fit, _ = training(cfg, rep)
        assert original_seeds == original_design["seeds"]
        assert array_sha(fit.tract_id) == original_design["fit_tract_id_sha256"]
        roles, design = panels(cfg, rep, fit, original_seeds["paired"], used_seeds)
        design["seen"]["city_effect_seed"] = original_seeds["cities"]
        designs[str(rep)] = design
        for case in CASES:
            file = original / f"case_{case}_models.json"
            sources[str(file.relative_to(source))] = sha(file.read_bytes())
            models = read(file)
            scores = {}
            for method, record in models.items():
                beta = unit(np.asarray(record["own"]["beta"]))
                np.testing.assert_array_equal(beta, np.asarray(record["equal_library"]["beta"]))
                beta.setflags(write=False)
                scores[method] = beta
                prefix = dict(rep=rep, case=case, method=method, panel="original_score_fit")
                structural_rows(city, prefix, fit, fit.x @ beta)
            for panel_name, (cal, test) in roles.items():
                cal_observed = cal.outcomes(case)[1]
                cal_label_sha = array_sha(cal_observed)
                score_before = {method: test.x @ beta for method, beta in scores.items()}
                fitted = {}
                for method, beta in scores.items():
                    coeffs, error = calibrate(beta, cal.x, cal_observed, case)
                    fitted[method] = coeffs
                    heads.append(dict(rep=rep, case=case, method=method, panel=panel_name,
                                      calibration_outcome_sha256=cal_label_sha,
                                      calibration_features_sha256=array_sha(cal.x), frozen_beta=beta.tolist(),
                                      frozen_beta_sha256=array_sha(beta), head=coeffs.tolist(),
                                      max_projected_gradient=error,
                                      linear_at_floor_city_count=int(np.sum(coeffs[:, 1] <= SLOPE_FLOOR + 1e-10)),
                                      cubic_zero_city_count=int(np.sum(coeffs[:, 2] == 0)) if case == "C" else None))
                # A separate diagnostic uses true f but the same observed calibration W and unknown head.
                oracle_head, _ = calibrate(TRUE_BETA, cal.x, cal_observed, case)
                # Open the evaluation outcomes only after every head for this panel and case is fitted.
                mean, observed = test.outcomes(case)
                outcome_rows(oracle, dict(rep=rep, case=case, method="true_f_diagnostic", panel=panel_name),
                             head_prediction(TRUE_BETA, oracle_head, test.x), mean, observed)
                for method, beta in scores.items():
                    before = score_before[method]
                    score_after = test.x @ beta
                    np.testing.assert_array_equal(before, score_after)
                    np.testing.assert_array_equal(delta(before), delta(score_after))
                    assert array_sha(cal_observed) == cal_label_sha
                    prefix = dict(rep=rep, case=case, method=method, panel=panel_name)
                    structural_rows(city, prefix, test, score_after)
                    outcome_rows(city, prefix, head_prediction(beta, fitted[method], test.x), mean, observed)
        print(f"Completed frozen-score recalibration for replication {rep}", flush=True)
    assert len(heads) == 350
    group_keys = ("rep", "case", "panel")
    inputs = pd.DataFrame(heads).groupby(list(group_keys))
    assert inputs.calibration_outcome_sha256.nunique().eq(1).all()
    assert inputs.calibration_features_sha256.nunique().eq(1).all()
    city_table = pd.DataFrame(city)
    assert city_table[city_table.panel.eq("original_score_fit")].target.str.endswith("f").sum() > 0
    assert not city_table[city_table.panel.eq("original_score_fit")].target.str.contains("W").any()
    summary = summarize(city_table, out)
    oracle_table = pd.DataFrame(oracle)
    oracle_table.to_csv(out / "true_f_calibration_city_diagnostic.csv", index=False)
    oracle_summary = (oracle_table.groupby(["rep", "case", "panel", "target"]).mse
                      .agg(city_mean_mse="mean", city_sd_mse="std").reset_index())
    (oracle_summary.groupby(["case", "panel", "target"])
     .agg(mean_city_mse=("city_mean_mse", "mean"), mean_within_rep_city_sd=("city_sd_mse", "mean"))
     .reset_index().to_csv(out / "true_f_calibration_summary.csv", index=False))
    write(out / "calibrated_heads.json", heads)
    cols = ("rep", "case", "method", "panel", "max_projected_gradient",
            "linear_at_floor_city_count", "cubic_zero_city_count")
    pd.DataFrame(heads).loc[:, list(cols)].to_csv(out / "head_diagnostics.csv", index=False)
    write(out / "manifest.json", dict(
        original_run=source.resolve().as_posix(), original_manifest_sha256=sha((source / "manifest.json").read_bytes()),
        saved_model_source_sha256=sources, code_sha256={name: sha((ROOT / name).read_bytes())
                                                       for name in (Path(__file__).name, "data.py", "losses.py")},
        config=cfg, seed_namespace=PANEL_NAMESPACE, panels=designs,
        separate_diagnostic="true f as a frozen score with the same observed calibration W and an estimated head",
        head_criterion="unpenalized unweighted constrained least squares on observed calibration W",
        head_inputs="frozen global unit-normalized score and city; common positive slope A, "
                    "city affine B, city monotone cubic C; same map both dates",
        city_aggregation="equal-city mean and sample SD of MSE within each replication, then mean over five replications",
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)))
    write(out / "audit.json", dict(
        passed=True, scorer_fits=0, original_scorers_verified=175, calibration_head_sets=len(heads),
        separate_true_f_head_sets=len(oracle_summary) // 4,
        exact_f_and_delta_f_score_freeze_checks=350, original_seed_count=old_seed_count,
        distinct_new_panel_seeds=len(used_seeds) - old_seed_count, original_fit_has_only_structural_metrics=True,
        calibration_test_tract_ids_disjoint=True, exact_calibration_fraction=.1,
        calibration_tracts_per_city=CALIBRATION_TRACTS, test_tracts_per_city=PANEL_TRACTS - CALIBRATION_TRACTS,
        same_calibration_data_across_methods=True, test_outcomes_used_for_fitting=False,
        city_metric_rows=len(city_table), summary_rows=len(summary),
        max_calibration_kkt_gradient=max(r["max_projected_gradient"] for r in heads)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="Original simple-index fit output")
    parser.add_argument("--out", type=Path, required=True, help="Fresh recalibration output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1):
        run(args.run, args.out)


if __name__ == "__main__":
    main()
