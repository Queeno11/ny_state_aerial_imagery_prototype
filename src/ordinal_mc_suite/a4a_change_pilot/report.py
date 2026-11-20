"""Combine completed tables and render held-out dose and detection figures."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARMS = ["A_anchor", "F_anchor", "O_continue", "L", "D", "LD", "LDC"]
COLORS = {"O_continue": "#254B87", "F_anchor": "#627380", "L": "#0C7976",
          "D": "#6F568D", "LD": "#D67A28", "LDC": "#A8474C"}


def collect(run, filename):
    return pd.concat((pd.read_csv(path) for path in sorted(run.glob("*/" + filename))),
                     ignore_index=True)


def summarize(main, destination):
    changes = main["structural"]
    changes = changes[(changes["panel"] == "supported_mixed") & (changes["endpoint"] == "selected")]
    parts = []
    for stratum, columns in [
        ("assigned_stable", ["rms_score_movement", "cutoff", "false_positive"]),
        ("actual_abs_gt_.05", ["rmse", "fit_normalized_truth_rmse", "normalization_only_rmse",
                              "bias", "slope", "correlation", "sign", "sensitivity"]),
        ("abs_0.05_to_0.15_both", ["count", "sensitivity"]),
        ("abs_0.15_to_0.35_both", ["count", "sensitivity"]),
    ]:
        df = changes[changes["stratum"] == stratum].copy()
        if stratum.startswith("abs_"):
            df = df.rename(columns={key: key + "_" + stratum for key in columns})
            columns = [key + "_" + stratum for key in columns]
        parts.append(df.set_index(["rep", "start", "arm", "where"])[columns])
    stats = pd.concat(parts, axis=1).reset_index()
    ranks = main["ranks"]
    rank_columns = ["observed_misranking", "truth_misranking", "bayes_reference_realized_misranking"]
    stats = stats.merge(ranks[ranks["endpoint"] == "selected"][["rep", "start", "arm", "where", *rank_columns]],
                        on=["rep", "start", "arm", "where"], validate="one_to_one")
    stats.to_csv(destination / "selected_structural_by_rep.csv", index=False)
    stats.groupby(["where", "arm"]).mean(numeric_only=True).drop(columns=["rep", "start"]).to_csv(
        destination / "selected_structural_means.csv")
    outcome = main["outcome"]
    outcome = outcome[(outcome["panel"] == "supported_mixed") & (outcome["stratum"] == "all")
                      & (outcome["endpoint"] == "selected")]
    outcome.to_csv(destination / "selected_outcome_by_rep.csv", index=False)
    outcome.groupby(["head", "arm"]).mean(numeric_only=True).drop(columns=["rep", "start"]).to_csv(
        destination / "selected_outcome_means.csv")
    comparisons = []
    for kind, data, metrics, additional in [
        ("structural", stats, ["rmse", "rms_score_movement", "false_positive",
                              "sensitivity_abs_0.05_to_0.15_both",
                              "sensitivity_abs_0.15_to_0.35_both", *rank_columns], ["where"]),
        ("outcome", outcome[outcome["head"] == "common"],
         ["mean_level_mse", "mean_change_mse", "realized_level_mse", "realized_change_mse",
          "mean_residual_covariance"], ["where"]),
    ]:
        index = ["rep", "start", *additional]
        for comparator in ("F_anchor", "O_continue"):
            baseline = data[data["arm"] == comparator].set_index(index)[metrics]
            for arm in ("L", "D", "LD", "LDC"):
                current = data[data["arm"] == arm].set_index(index)[metrics]
                diff = (current - baseline).stack().rename("difference").reset_index()
                diff = diff.rename(columns={diff.columns[-2]: "metric"})
                diff["arm"], diff["comparator"], diff["kind"] = arm, comparator, kind
                comparisons.append(diff)
    differences = pd.concat(comparisons, ignore_index=True)
    differences.to_csv(destination / "paired_differences_by_rep.csv", index=False)
    differences.groupby(["kind", "where", "arm", "comparator", "metric"])["difference"].agg(
        ["mean", "min", "max", "count"]).to_csv(destination / "paired_difference_ranges.csv")


def plots(main, dest):
    dose = main["dose"]
    dose = dose[dose["where"] == "unseen"]
    label = ["All starting tracts", "Low starting score", "Middle starting score", "High starting score"]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True, layout="constrained")
    for ax, tertile, title in zip(axs.ravel(), ["all", "1", "2", "3"], label):
        rows = dose[dose["start_tertile"].astype(str) == tertile]
        for arm in ("F_anchor", "O_continue", "L", "D", "LD", "LDC"):
            mean = rows[rows["arm"] == arm].groupby("nominal_step").mean(numeric_only=True)
            ax.plot(mean.index, mean["predicted_mean"], color=COLORS[arm], label=arm, linewidth=1.6)
            if arm == "O_continue":
                ax.fill_between(mean.index, mean["predicted_p10"], mean["predicted_p90"],
                                color=COLORS[arm], alpha=.13, label="O: 10th–90th tract percentiles")
                ax.plot(mean.index, mean["true_mean"], color="#202020", linestyle="--",
                        label="True mean after clipping", linewidth=1.4)
        ax.axhline(0, color="#aaaaaa", linewidth=.65)
        ax.set_title(title)
        ax.grid(alpha=.17)
    fig.supxlabel("Nominal signed perturbation along the structural gradient")
    fig.supylabel("Mean latent change (normalized units)")
    axs[0, 0].legend(loc="upper left", fontsize=7, ncol=2)
    fig.suptitle("Matched-start dose response on unseen cities", fontsize=15)
    fig.savefig(dest / "unseen_dose_response.png", dpi=170)
    plt.close(fig)

    c = main["structural"]
    c = c[(c["panel"] == "supported_mixed") & (c["endpoint"] == "selected") & (c["where"] == "unseen")]
    keys = ["abs_0.05_to_0.15_both", "abs_0.15_to_0.35_both",
            "abs_0.35_to_0.65_both", "abs_0.65_to_inf_both"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained", sharey=True)
    for ax, direction in zip(axs, ("increase", "decline")):
        for arm in ("F_anchor", "O_continue", "L", "D", "LD", "LDC"):
            rows = c[c["arm"] == arm]
            values = [rows[rows["stratum"] == key.replace("both", direction)]["sensitivity"].mean()
                      for key in keys]
            ax.plot(np.arange(4), values, marker="o", markersize=3.5, linewidth=1.3,
                    color=COLORS[arm], label=arm)
        ax.set_xticks(np.arange(4), [".05–.15", ".15–.35", ".35–.65", ">.65"])
        ax.set_title("True increase" if direction == "increase" else "True decline")
        ax.set_ylim(0, 1.03)
        ax.grid(alpha=.2)
    axs[0].legend(loc="lower right", fontsize=7, ncol=2)
    fig.supxlabel("Absolute true change")
    fig.supylabel("Sensitivity at scorer-specific stable cutoff")
    fig.suptitle("Detection by realized structural change size on unseen cities", fontsize=13)
    fig.savefig(dest / "unseen_detection.png", dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", type=Path, required=True)
    p.add_argument("--known", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    assert (args.main / "completion.json").exists() and (args.known / "completion.json").exists()
    args.out.mkdir(parents=True, exist_ok=False)
    main_tables = {}
    for table in ["structural", "outcome", "ranks", "regimes", "dose", "selections"]:
        main_tables[table] = collect(args.main, table + ".csv")
        main_tables[table].to_csv(args.out / (table + ".csv"), index=False)
    for table in ["structural", "outcome", "ranks", "selections"]:
        collect(args.known, table + ".csv").to_csv(args.out / ("known_" + table + ".csv"), index=False)
    summarize(main_tables, args.out)
    plots(main_tables, args.out)


if __name__ == "__main__":
    main()
