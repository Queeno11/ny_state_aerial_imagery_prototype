"""Verify the saved five-shot run and produce per-case comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from classical import Model
from data import CASES, training, TRUE_BETA
from losses import fit_head, head_prediction, unit
from run_simple import case_data, create_data, write

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def model_from(record):
    return Model(**{key: None if value is None else np.asarray(value) for key, value in record.items()})


def audit(run, tables):
    manifest = read(run / "manifest.json")
    cfg = manifest["config"]
    completed = read(run / "completion.json")
    assert completed["fitted_estimators"] == 175 and completed["tuning_scorer_fits"] == 39
    assert not (run / "failure.txt").exists()
    for name, hash_ in manifest["source_sha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == hash_
    assert hashlib.sha256((HERE / "config.json").read_bytes()).hexdigest() == manifest["config_sha256"]
    frozen = read(run / "frozen_selection.json")
    assert frozen["rep"] not in cfg["replications"] and not frozen["evaluated_tests"]
    assert tables["calibration_candidates"].groupby("case").size().eq(13).all()
    for case, row in frozen["weights"].items():
        candidates = tables["calibration_candidates"].query("case == @case and rank_weight == 1")
        best = candidates[candidates.composite <= candidates.composite.min()+1e-8].copy()
        best["budget"] = best.rW+best.rD
        assert best.sort_values(["budget", "candidate"]).iloc[0].candidate == row["grid_winner"]
    out = tables["outcomes"]
    assert set(out["where"]) == {"seen"}
    decomposition = out.mean_residual_date0_mse+out.mean_residual_date1_mse-2*out.mean_residual_cross_moment
    np.testing.assert_allclose(out.expected_W_change_mse, decomposition, atol=1e-11, rtol=1e-11)
    seeds = set(frozen["fitting_and_validation_seeds"].values())
    fe_match, sigma_match, head_match = [], [], []
    for rep in cfg["replications"]:
        folder = run / f"rep_{rep}"
        design = read(folder / "design.json")
        regenerated, panel, valid, fit_links, val_links = create_data(cfg, rep)
        assert regenerated == design["seeds"]
        assert not seeds & set(regenerated.values())
        seeds.update(regenerated.values())
        assert design["fit_tracts"] == 3200 and design["validation_tracts"] == 800
        assert design["auxiliary_fit_links"] == 12800
        for value, name in ((panel, "fit"), (valid, "validation")):
            assert hashlib.sha256(value.tract_id.tobytes()).hexdigest() == design[f"{name}_tract_id_sha256"]
        for case in CASES:
            data, _ = case_data(case, panel, valid, fit_links, val_links)
            models = read(folder / f"case_{case}_models.json")
            for name, params in models.items():
                model = model_from(params["own"])
                lib = model_from(params["equal_library"])
                assert np.all(lib.head[:, 1] > 0) and np.all(lib.head[:, 2] >= 0)
                if name == "Ordinal_S":
                    np.testing.assert_array_equal(model.beta, lib.beta)
                    np.testing.assert_array_equal(model.head, lib.head)
                if name.startswith("Gaussian_ML"):
                    sd = np.sqrt(np.mean((model.predict(data.x)-data.w)**2, axis=(1, 2)))
                    sigma_match.append(float(np.max(abs(model.sigma-sd))))
                if case == "A" and name == "FE":
                    beta = unit(model.beta)
                    pred = head_prediction(beta, fit_head(beta, data), data.x)
                    fe_match.append(float(np.max(abs(pred-model.predict(data.x)))))
                diag = tables["optimizer"].query("rep == @rep and case == @case and method == @name")
                if len(diag) and isinstance(diag.iloc[0].training_weights, str) and name != "Ordinal_S":
                    w, d, g = json.loads(diag.iloc[0].training_weights)[2:]
                    head_match.append(float(np.max(abs(model.head-fit_head(model.beta, data, w, d, g)))))
    assert max(fe_match) < 1e-12 and max(sigma_match) < 1e-13 and max(head_match) < 1e-12
    diagnostics = tables["optimizer"]
    numerical = diagnostics[diagnostics.min_tangent_grad.notna()]
    assert len(numerical) == 115 and numerical.min_tangent_grad.max() < 1e-4
    mix = diagnostics[diagnostics.method.str.startswith("Mixed_")]
    return dict(passed=True, replications=cfg["replications"], fits=175, calibration_fits=39,
                distinct_panel_seeds=len(seeds), max_fe_prediction_difference=max(fe_match),
                max_gaussian_sigma_difference=max(sigma_match), max_profile_head_difference=max(head_match),
                max_optimizer_tangent_gradient=float(numerical.min_tangent_grad.max()),
                mixed_converged={method: int(values.converged.astype(bool).sum())
                                 for method, values in mix.groupby("method")})


def tabulate(tables):
    key = ["rep", "case", "method"]
    beta = tables["coefficients"].set_index(key)
    structural = tables["structural"].query("where == 'unseen' and panel == 'mixed' and sigma_group == 'all'")
    selected = ["changed_rmse", "stable_rms", "fresh_fpr", "sensitivity", "sign_accuracy",
                "changed_bias", "individual_response_slope"]
    beta = beta.join(structural.set_index(key)[selected], validate="one_to_one")
    out = tables["outcomes"].query("panel == 'mixed' and sigma_group == 'all'")
    columns = ["expected_W_level_mse", "expected_W_change_mse", "realized_W_level_mse", "realized_W_change_mse",
               "zero_expected_W_change_mse", "change_noise_floor"]
    for head in ("own", "equal_library"):
        frame = out[out["head"] == head].set_index(key)[columns]
        beta = beta.join(frame.rename(columns={x: f"{head}_{x}" for x in columns}), validate="one_to_one")
    ranks = tables["ranks"].query("where == 'unseen'").set_index(key)
    beta = beta.join(ranks[["latent_misrank", "observed_misrank", "expected_bayes_misrank"]], validate="one_to_one")
    for dose in ("dose_-0.15", "dose_-0.05", "dose_+0.05", "dose_+0.15", "dose_+0"):
        struct = tables["structural"]
        subset = struct.query("where == 'unseen' and panel == @dose and sigma_group == 'all'").set_index(key)
        for field in ("sensitivity", "sign_accuracy", "fresh_fpr"):
            if not subset[field].isna().all():
                beta[f"{field}_{dose}"] = subset[field]
    return beta.reset_index()


def paired(combined):
    comparisons = []
    for case, block in combined.groupby("case"):
        arms = dict(tuple(block.groupby("method")))
        metrics = [k for k in block.select_dtypes(include="number") if k != "rep"]
        for arm, other in (("Ordinal_L2_S", "Ordinal_S"), ("Ordinal_L2_D_S", "Ordinal_L2_S"),
                           ("L2_D_S", "L2_S"), ("L2_S", "FE"), ("Gaussian_ML_S", "Gaussian_ML"),
                           ("Ordinal_L2_D_S_C", "Ordinal_L2_S_C"), ("Ordinal_L2_S_C", "Ordinal_L2_S"),
                           ("Gaussian_ML", "Mixed_index"), ("Gaussian_ML", "Ordinal_S"),
                           ("Gaussian_ML", "City_slopes")):
            if not {arm, other} <= set(arms):
                continue
            a, b = (arms[name].set_index("rep")[metrics] for name in (arm, other))
            delta = (a-b).stack().rename("difference").reset_index().rename(columns={"level_1": "metric"})
            comparisons.append(delta.assign(case=case, arm=arm, comparator=other))
    return pd.concat(comparisons, ignore_index=True)


def figure(main, out):
    keys = ["FE", "City_slopes", "Mixed_index", "Gaussian_ML", "Gaussian_ML_S",
            "Ordinal_S", "L2_S", "Ordinal_L2_S", "Ordinal_L2_D_S"]
    colors = plt.get_cmap("tab10")(np.arange(len(keys)))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    mean = main.groupby(["case", "method"]).mean(numeric_only=True)
    for i, case in enumerate(CASES):
        axis = axes[i]
        for method, color in zip(keys, colors):
            row = mean.loc[(case, method)]
            axis.scatter(row.changed_rmse, row.own_expected_W_change_mse, s=28, color=color, label=method)
        axis.set_yscale("log")
        axis.set_title(f"Map {case}")
        axis.set_xlabel("Unseen latent-change RMSE")
        axis.grid(alpha=.18)
    axes[0].set_ylabel("Seen expected outcome-change MSE (log scale)")
    axes[-1].legend(fontsize=7)
    fig.savefig(out / "structural_outcome_comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(args.run / "results.sqlite") as connection:
        tables = {name: pd.read_sql_query(f"SELECT * FROM {name}", connection) for name in
                  ("structural", "outcomes", "coefficients", "ranks", "optimizer", "validation", "calibration_candidates")}
    with threadpool_limits(limits=1):
        verified = audit(args.run, tables)
        data = tabulate(tables)
        data.to_csv(args.out / "by_rep.csv", index=False)
        data.groupby(["case", "method"]).mean(numeric_only=True).drop(columns="rep").to_csv(args.out / "means.csv")
        contrasts = paired(data)
        contrasts.to_csv(args.out / "paired_by_rep.csv", index=False)
        contrasts.groupby(["case", "arm", "comparator", "metric"]).difference.agg(["mean", "min", "max", "count"]).to_csv(
            args.out / "paired_ranges.csv")
        figure(data, args.out)
    write(args.out / "audit.json", verified)


if __name__ == "__main__":
    main()
