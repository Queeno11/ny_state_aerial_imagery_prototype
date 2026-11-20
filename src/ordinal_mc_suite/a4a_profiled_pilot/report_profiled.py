"""Produce comparison tables and a signed-dose figure from the frozen pilot."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prior import ORIGINAL
from report import collect

KEYS = ["role", "rep", "arm", "endpoint"]


def main_table(run, doses):
    structural = collect(run, "structural.csv")
    s = structural.query("panel == 'supported_mixed' and where == 'unseen'")
    idx = ["role", "rep", "arm", "endpoint"]
    sections = []
    strata = {"actual_abs_gt_.05": ["rmse", "sign", "slope", "normalization_only_rmse",
                                    "fit_normalized_truth_rmse"],
              "assigned_stable": ["rms_score_movement", "false_positive", "cutoff"]}
    strata.update({f"start_tertile_{k}": ["rmse", "slope", "bias"] for k in range(1, 4)})
    for key, columns in strata.items():
        section = s[s["stratum"] == key].set_index(idx)[columns]
        if key.startswith("start_"):
            section = section.rename(columns={c: f"{c}_{key}" for c in columns})
        sections.append(section)
    primary = pd.concat(sections, axis=1)
    for head in ("own", "equal_library"):
        y = collect(run, "outcome.csv")
        y = y.query("panel == 'supported_mixed' and stratum == 'all' and head == @head").set_index(idx)
        columns = ["mean_level_mse", "mean_change_mse", "realized_level_mse", "realized_change_mse",
                   "mean_residual_covariance", "mean_change_noise_floor", "zero_mean_change_mse", "mean_change_sign"]
        primary = primary.join(y[columns].rename(columns={k: f"{head}_{k}" for k in columns}), validate="one_to_one")
    ranks = collect(run, "ranks.csv").query("where == 'unseen'").set_index(idx)
    primary = primary.join(ranks[["observed_misranking", "truth_misranking"]], validate="one_to_one")
    choices = collect(run, "selections.csv").set_index(idx)
    columns = ["source_path", "head_name", "update", "selector", "rank", "stability", "level", "change",
               "historical_selector"]
    primary = primary.join(choices[columns], validate="one_to_one").reset_index()
    doses = doses.query("where == 'unseen' and stratum == 'all'")
    for dose in (-.15, -.05, 0., .05, .15):
        d = doses[doses.dose == dose].set_index(["rep", "arm"])
        c = ["flag_rate", "correct_signed_flag_rate", "sign_accuracy"]
        x = d[c].rename(columns={key: f"{key}_{dose:+g}" for key in c})
        x = x.reset_index().assign(endpoint="selected")
        primary = primary.merge(x, how="left", on=["rep", "arm", "endpoint"], validate="one_to_one")
    return primary


def paired(main):
    contrasts = [
        ("Legacy_Adam_LDC", "selected", "Legacy_Adam_LDC", "selected_common", "own_vs_historical_selection"),
        ("Adam_LDC", "selected", "Legacy_Adam_LDC", "selected", "correct_init"),
        ("Profile_LDC", "selected", "Adam_LDC", "selected", "profile_vs_adam_with_curvature"),
        ("Profile_LD", "selected", "Adam_LD", "selected", "profile_vs_adam_without_curvature"),
        ("Profile_LDC", "selected", "Profile_LC", "selected", "difference_with_curvature"),
        ("Profile_LD", "selected", "Profile_L", "selected", "difference_without_curvature"),
        ("Profile_LC", "selected", "Profile_L", "selected", "curvature_without_difference"),
        ("Profile_LDC", "selected", "Profile_LD", "selected", "curvature_with_difference"),
        ("Profile_LDC", "selected", "O", "selected", "profile_ldc_vs_ordinal"),
        ("Profile_selected", "selected", "O", "selected", "profile_family_vs_ordinal"),
        ("Profile_selected", "selected", "F", "selected", "profile_family_vs_frozen"),
        ("Profile_LDC", "fixed", "Adam_LDC", "fixed", "fixed_profile_vs_adam"),
        ("Profile_LDC", "fixed", "Profile_LC", "fixed", "fixed_difference_with_curvature"),
        ("Profile_LDC", "fixed", "O", "fixed", "fixed_profile_ldc_vs_ordinal"),
        ("Profile_selected", "fixed", "O", "fixed", "fixed_profile_family_vs_ordinal"),
    ]
    combined = []
    metrics = main.select_dtypes(include="number").columns.difference(["rep"])
    for arm, ep, comparator, other_ep, question in contrasts:
        a = main[(main.arm == arm) & (main.endpoint == ep)].set_index(["role", "rep"])
        b = main[(main.arm == comparator) & (main.endpoint == other_ep)].set_index(["role", "rep"])
        diff = (a[metrics] - b[metrics]).stack().rename("difference").reset_index()
        diff = diff.rename(columns={diff.columns[-2]: "metric"})
        combined.append(diff.assign(contrast=question, arm=arm, endpoint=ep,
                                    comparator=comparator, comparison_endpoint=other_ep))
    return pd.concat(combined, ignore_index=True)


def optimization(run, primary):
    rows = []
    for folder in sorted(run.glob("rep_*")):
        rep = int(folder.name.split("_")[1])
        traj = pd.read_csv(folder / "fitting_trajectories.csv")
        sel = primary[(primary.rep == rep) & (primary.endpoint == "selected")].set_index("arm")
        for arm, t in traj.groupby("path"):
            row = sel.loc[arm]
            first, final = t.iloc[0], t.iloc[-1]
            chosen = t[t["update"] == row["update"]].iloc[0]
            rows.append(dict(rep=rep, role=row["role"], arm=arm, selected_update=chosen["update"],
                             fit_objective_initial=first.fit_objective, fit_objective_selected=chosen.fit_objective,
                             fit_objective_final=final.fit_objective,
                             head_gap_initial=first.head_gap, head_gap_selected=chosen.head_gap,
                             head_gap_final=final.head_gap, mean_head_gap=t.head_gap.mean(),
                             steps_with_objective_increase_gt_1e_7=int((np.diff(t.fit_objective) > 1e-7).sum()),
                             max_adjacent_objective_increase=float(np.diff(t.fit_objective).max()),
                             trajectory_points=len(t), max_head_kkt=t.projected_head_gradient.max(),
                             mean_head_iterations=t.head_iterations.mean(), mean_step_seconds=t.step_seconds.mean()))
    return pd.DataFrame(rows)


def figure(run, doses, out):
    d = doses.query("where == 'unseen' and stratum == 'all'").copy()
    raw = collect(run, "dose.csv").query("where == 'unseen' and start_tertile == 'all'")
    labels = {"F": "Frozen", "O": "Ordinal", "Adam_LDC": "Correct-init Adam LDC",
              "Profile_LC": "Profiled LC", "Profile_LDC": "Profiled LDC",
              "Profile_selected": "Profiled family selection"}
    colors = ["#525C67", "#21477C", "#A25E38", "#24837B", "#A33754", "#5A4B90"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), layout="constrained")
    for col, (role, title) in enumerate((("replay", "Replay draws"), ("confirmation", "Fresh draws"))):
        names = sorted(raw[raw.role == role].rep.unique())
        for (arm, label), color in zip(labels.items(), colors):
            subset = raw[(raw.role == role) & (raw.arm == arm)].groupby("nominal_step").mean(numeric_only=True)
            axes[0, col].plot(subset.index, subset.predicted_mean, marker=".", lw=1.15, label=label, color=color)
            if arm == "O":
                axes[0, col].plot(subset.index, subset.true_mean, color="#202020", ls="--", label="True mean after clipping")
                axes[0, col].fill_between(subset.index, subset.predicted_p10, subset.predicted_p90,
                                          color=color, alpha=.13, label="Ordinal tract percentiles")
            flags = d[(d.rep.isin(names)) & (d.arm == arm) & (d.dose.abs() <= .15)].groupby("dose").flag_rate.mean()
            axes[1, col].plot(flags.index, flags, marker=".", lw=1.15, color=color)
        axes[0, col].set_title(title)
        axes[0, col].set_xlabel("Nominal signed step")
        axes[1, col].set_xlabel("Nominal signed step")
        axes[1, col].set_ylim(0, 1)
        for axis in axes[:, col]:
            axis.grid(alpha=.2)
    axes[0, 0].set_ylabel("Mean normalized structural change")
    axes[1, 0].set_ylabel("Flag fraction with stable cutoff")
    axes[0, 1].legend(loc="upper left", fontsize=6.5, ncol=2)
    fig.savefig(out / "signed_dose_response.png", dpi=175)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--doses", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    assert (args.run / "completion.json").exists()
    args.out.mkdir(parents=True, exist_ok=False)
    doses = pd.read_csv(args.doses)
    primary = main_table(args.run, doses)
    primary.to_csv(args.out / "by_rep.csv", index=False)
    primary.groupby(["role", "arm", "endpoint"]).mean(numeric_only=True).drop(columns="rep").to_csv(
        args.out / "means.csv")
    differences = paired(primary)
    differences.to_csv(args.out / "paired_by_rep.csv", index=False)
    differences.groupby(["role", "contrast", "metric"]).difference.agg(["mean", "min", "max", "count"]).to_csv(
        args.out / "paired_ranges.csv")
    optimization(args.run, primary).to_csv(args.out / "optimization_by_rep.csv", index=False)
    figure(args.run, doses, args.out)


if __name__ == "__main__":
    main()
