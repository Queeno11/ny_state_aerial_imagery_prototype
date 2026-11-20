"""Summarize paired replay and confirmation results without retuning."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from prior import ORIGINAL
from report import collect, plots


def primary(folder, doses):
    s = collect(folder, "structural.csv")
    s = s[(s["endpoint"] == "selected") & (s["panel"] == "supported_mixed") & (s["where"] == "unseen")]
    idx = ["rep", "arm"]
    columns = {
        "actual_abs_gt_.05": ["rmse", "sign", "slope", "fit_normalized_truth_rmse", "normalization_only_rmse"],
        "assigned_stable": ["rms_score_movement", "cutoff", "false_positive"],
        **{f"start_tertile_{i}": ["rmse", "slope", "bias"] for i in range(1, 4)},
    }
    parts = []
    for key, names in columns.items():
        x = s[s["stratum"] == key].set_index(idx)[names]
        if key.startswith("start_"):
            x = x.rename(columns={k: f"{k}_{key}" for k in names})
        parts.append(x)
    main = pd.concat(parts, axis=1)
    for dose in (-.15, -.05, 0., .05, .15):
        df = doses[(doses["where"] == "unseen") & (doses["dose"] == dose) & (doses["stratum"] == "all")]
        cols = ["flag_rate", "sign_accuracy", "correct_signed_flag_rate"]
        df = df.set_index(idx)[cols].rename(columns={col: f"{col}_{dose:+g}" for col in cols})
        main = main.join(df, validate="one_to_one")
    outcomes = collect(folder, "outcome.csv")
    outcomes = outcomes[(outcomes["endpoint"] == "selected") & (outcomes["panel"] == "supported_mixed")
                        & (outcomes["stratum"] == "all")]
    for kind in ("common", "native", "deployed"):
        cols = ["mean_level_mse", "mean_change_mse", "realized_level_mse", "realized_change_mse",
                "mean_residual_covariance", "mean_change_sign"]
        rows = outcomes[outcomes["head"] == kind].set_index(idx)[cols]
        rows.columns = [f"{kind}_{col}" for col in cols]
        main = main.join(rows, validate="one_to_one")
    return main.reset_index()


def paired(current, old, groups):
    result = []
    for arm, comp, frame in groups:
        a = current[current["arm"] == arm].set_index("rep").drop(columns="arm")
        b = frame[frame["arm"] == comp].set_index("rep").drop(columns="arm")
        if not len(a) or not len(b):
            continue
        assert set(a.index) == set(b.index)
        delta = a - b
        d = delta.stack().rename("difference").reset_index().rename(columns={"level_1": "metric"})
        d["arm"] = arm
        d["comparator"] = ("original_" if frame is old else "") + comp
        result.append(d)
    return pd.concat(result, ignore_index=True)


def known(folder):
    s = collect(folder, "structural.csv")
    s = s[(s["endpoint"] == "selected") & (s["panel"] == "supported_mixed") & (s["where"] == "unseen")]
    idx = ["noise", "start", "arm"]
    selected = collect(folder, "selections.csv")
    selected = selected[selected["endpoint"] == "selected"].set_index(idx)
    x = s[s["stratum"] == "actual_abs_gt_.05"].set_index(idx)[["rmse", "sign", "sensitivity"]]
    x = x.join(s[s["stratum"] == "assigned_stable"].set_index(idx)[["rms_score_movement", "false_positive"]])
    o = collect(folder, "outcome.csv")
    o = o[(o["endpoint"] == "selected") & (o["stratum"] == "all")].set_index(idx)
    cols = ["mean_level_mse", "mean_change_mse", "realized_level_mse", "realized_change_mse"]
    return x.join(o[cols]).join(selected[["update", "direct_f", "level", "change"]]).reset_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--confirmation", type=Path, required=True)
    p.add_argument("--known", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    old = primary(ORIGINAL / "main_pilot", pd.read_csv(ORIGINAL / "results/dose_detection_by_rep.csv"))
    for group in ("replay", "confirmation"):
        folder = getattr(args, group)
        dest = args.out / group
        dest.mkdir()
        doses = pd.read_csv(folder / "dose_detection_by_rep.csv")
        current = primary(folder, doses)
        current.to_csv(dest / "selected_by_rep.csv", index=False)
        current.groupby("arm").mean(numeric_only=True).drop(columns="rep").to_csv(dest / "selected_means.csv")
        controls = [(a, c, current) for a in ("L", "D", "LD", "LDC", "Global_selected")
                    for c in ("O_continue", "F_anchor")]
        if group == "replay":
            controls += [(a, a, old) for a in ("LD", "LDC")]
        else:
            controls += [(a, f"Old_{a}", current) for a in ("LD", "LDC")]
        differences = paired(current, old, controls)
        differences.to_csv(dest / "paired_differences.csv", index=False)
        differences.groupby(["arm", "comparator", "metric"]).difference.agg(["mean", "min", "max", "count"]).to_csv(
            dest / "paired_ranges.csv")
        tables = {name: collect(folder, name + ".csv") for name in ("dose", "structural")}
        plots(tables, dest)
        collect(folder, "head_choices.csv").to_csv(dest / "head_choices.csv", index=False)
    known(args.known).to_csv(args.out / "known_selected.csv", index=False)
    known(ORIGINAL / "known_control").to_csv(args.out / "known_original_selected.csv", index=False)


if __name__ == "__main__":
    main()
