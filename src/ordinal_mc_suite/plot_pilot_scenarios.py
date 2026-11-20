"""Render saved pilot results by measurement family; never trains models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter


ROOT = Path(__file__).resolve().parent
METHODS = ["A", "B", "C", "D", "O", "scratch_C", "scratch_D"]
COLORS = dict(zip(METHODS, ["#2771B5", "#29A6B8", "#178260", "#D77523",
                           "#7755AA", "#A66C51", "#CC628C"]))
LABELS = {"A": "A  Frozen / levels", "B": "B  Frozen / levels + changes",
          "C": "C  Joint / levels", "D": "D  Joint / levels + changes",
          "O": "O  Ordinal continuation", "scratch_C": "Scratch / levels",
          "scratch_D": "Scratch / levels + changes"}
METRICS = ["f_mse", "mu_mse", "delta_f_mse", "delta_mu_mse"]
PANEL_TITLES = [r"$f$  |  Latent signal", r"$\mu$  |  Expected outcome",
                r"$\Delta f$  |  Latent change", r"$\Delta\mu$  |  Expected outcome change"]
FAMILIES = {"M1": [1, 4], "M2": [2, 5], "M3": [3, 6, 7, 8]}
TITLES = {"M1": "Additive cell effects", "M2": "Cell-specific affine maps",
          "M3": "Nonlinear monotone maps"}
FORMULAS = {"M1": r"$\Psi_{ct}(r)=\alpha_{ct}+r$",
            "M2": r"$\Psi_{ct}(r)=\alpha_{ct}+\beta_{ct}r$",
            "M3": r"$\Psi_{ct}(r)=\alpha_{ct}+\beta_{ct}[r+k_{ct}\tanh(r)]$"}
SETTING_LABELS = {1: "Linear f\nSimple change", 2: "Linear f\nSimple change",
                  3: "Linear f\nSimple change", 4: "Quadratic f\nSimple change",
                  5: "Quadratic f\nSimple change", 6: "Quadratic f\nSimple change",
                  7: "Quadratic f\nNonlinear change", 8: "Quadratic f\nNonlinear change"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "joint_two_step_one_seed" / "metrics.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or args.input.parent / "scenario_figures"
    out.mkdir(parents=True, exist_ok=True)
    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream)
                if r["where"] == "target" and r["split"] == "test" and r["stratum"] == "all"]
    seeds = sorted({r["seed"] for r in rows})
    if not rows:
        raise ValueError("No target-city test rows")
    for r in rows:
        r["setting"] = int(r["setting"])
        for key in METRICS:
            r[key] = float(r[key])
            if not np.isfinite(r[key]) or r[key] < 0:
                raise ValueError(f"Invalid {key} in {r}")
    # A/B use the same frozen scorer. Catch a wrong input/filter before plotting.
    indexed = {(r["seed"], r["setting"], r["rank"], r["links"], r["arm"]): r for r in rows}
    for r in rows:
        if r["arm"] == "A":
            b = indexed[(r["seed"], r["setting"], r["rank"], r["links"], "B")]
            for key in ("f_mse", "delta_f_mse"):
                np.testing.assert_allclose(r[key], b[key], rtol=1e-12, atol=1e-14)
        if r["arm"] == "true_parameter_oracle":
            assert all(abs(r[key]) < 1e-14 for key in METRICS)

    limits = {}
    for key in METRICS:
        values = [r[key] for r in rows if r[key] > 0]
        limits[key] = (min(values) / 1.7, max(values) * 1.65)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": "#334155", "text.color": "#172B3A",
                         "xtick.color": "#475569", "ytick.color": "#475569",
                         "axes.edgecolor": "#C8D1D9", "axes.linewidth": .7,
                         "svg.fonttype": "none", "savefig.facecolor": "#FAFAF8"})
    export_rows = []
    png_paths = []
    notes = []
    for family, settings in FAMILIES.items():
        categories = [(setting, link) for setting in settings for link in ("0", "1")]
        relevant = [arm for arm in METHODS if any(r["arm"] == arm and r["setting"] in settings for r in rows)]
        offsets = dict(zip(relevant, np.linspace(-.33, .33, len(relevant))))
        fig = plt.figure(figsize=(18 if family == "M3" else 14.5, 11.5), facecolor="#FAFAF8")
        grid = fig.add_gridspec(2, 2, left=.075, right=.98, bottom=.29, top=.86,
                               wspace=.20, hspace=.50)
        fig.text(.075, .962, f"{family}  |  {TITLES[family]}", fontsize=25, weight="bold")
        fig.text(.075, .921, FORMULAS[family], fontsize=16)
        fig.text(.98, .947, "HELD-OUT CITIES  /  LOCALLY CALIBRATED HEADS", fontsize=10,
                 ha="right", color="#64748B")
        fig.text(.98, .917, f"{len(seeds)} seed{'s' if len(seeds)>1 else ''}  |  MSE, lower is better",
                 fontsize=11, ha="right", color="#475569")
        for panel, key in enumerate(METRICS):
            ax = fig.add_subplot(grid[panel // 2, panel % 2])
            ax.set_facecolor("white")
            for block in range(len(settings)):
                if block % 2 == 0:
                    ax.axvspan(2*block-.5, 2*block+1.5, color="#EFF3F7", alpha=.65, zorder=0)
                if block:
                    ax.axvline(2*block-.5, color="#CCD5DF", lw=.8, zorder=1)
            for x, (setting, link) in enumerate(categories):
                for arm in relevant:
                    for rank, marker, shift in (("ranknet", "o", -.022), ("sigmoid", "s", .022)):
                        group = [r for r in rows if r["setting"] == setting and r["links"] == link
                                 and r["arm"] == arm and r["rank"] == rank]
                        if not group:
                            if arm.startswith("scratch_") and setting not in (1, 8):
                                continue
                            raise ValueError(f"Missing {family}/{setting}/{link}/{rank}/{arm}")
                        value = float(np.mean([r[key] for r in group]))
                        if value <= 0:
                            raise ValueError("A fitted-method MSE is zero; log plot requires explicit handling")
                        ax.scatter(x + offsets[arm] + shift, value, marker=marker,
                                   s=37 if marker == "o" else 29, color=COLORS[arm],
                                   edgecolors="white", linewidths=.55, zorder=4)
                        export_rows.append(dict(family=family, setting=setting, links=link,
                            rank=rank, arm=arm, metric=key, mse=value, seeds=len(group),
                            benchmark_status="not_applicable"))
                mle = [r for r in rows if r["setting"] == setting and r["arm"] == "estimated_parametric"]
                if len(mle) != len(seeds):
                    raise ValueError(f"Missing or duplicated MLE for setting {setting}")
                value = float(np.mean([r[key] for r in mle]))
                source_ok = all(r["benchmark_convergence"] == "gradient tolerance met" for r in mle)
                target_ok = all(r["target_benchmark_convergence"].startswith("CONVERGENCE") for r in mle)
                certified = source_ok and (target_ok or key in ("f_mse", "delta_f_mse"))
                ax.plot([x-.43, x+.43], [value, value], color="#27313B", lw=.9,
                        linestyle=(0, (3, 3)), alpha=.85, zorder=2)
                ax.scatter(x, value, marker="D", s=44, edgecolor="#17212B",
                           facecolor="#17212B" if certified else "white", linewidth=1.25, zorder=5)
                export_rows.append(dict(family=family, setting=setting, links=link,
                    rank="shared", arm="estimated_parametric", metric=key, mse=value,
                    seeds=len(mle), benchmark_status="converged" if certified else "convergence_unconfirmed"))
                export_rows.append(dict(family=family, setting=setting, links=link,
                    rank="shared", arm="true_parameter_oracle", metric=key, mse=0.,
                    seeds=len(seeds), benchmark_status="known_parameters"))
            ax.set_yscale("log")
            ax.set_ylim(*limits[key])
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.,), numticks=6))
            ax.yaxis.set_major_formatter(LogFormatterMathtext())
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2., 5.), numticks=15))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(axis="y", which="major", color="#DAE1E7", lw=.7, zorder=0)
            ax.grid(axis="y", which="minor", color="#E8EDF1", lw=.5, zorder=0)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(-.55, len(categories)-.45)
            labels = []
            for setting, link in categories:
                rho = "0.9" if setting == 8 else "0"
                labels.append(SETTING_LABELS[setting].replace(" change", "") +
                              "\n" + r"$\rho=" + rho + "$" +
                              f"\nLinks {'ON' if link=='1' else 'OFF'}")
            ax.set_xticks(range(len(categories)), labels, fontsize=8.7 if family=="M3" else 9.5)
            ax.tick_params(axis="x", length=0, pad=8)
            ax.set_ylabel("Mean squared error (log scale)", fontsize=10)
            ax.set_title(f"{'ABCD'[panel]}   {PANEL_TITLES[panel]}", loc="left", fontsize=15, pad=13, weight="bold")
        method_handles = [Line2D([], [], color=COLORS[a], marker="o", linestyle="none",
                                 markersize=7, label=LABELS[a]) for a in relevant]
        fig.legend(handles=method_handles, loc="lower center", bbox_to_anchor=(.53, .155),
                   ncol=4 if len(relevant)>5 else 3, frameon=False, fontsize=10,
                   columnspacing=2.0, handletextpad=.6, labelspacing=.9)
        marker_handles = [Line2D([], [], color="#536574", marker="o", linestyle="none", label="RankNet"),
                          Line2D([], [], color="#536574", marker="s", linestyle="none", label="Bounded sigmoid"),
                          Line2D([], [], color="#17212B", marker="D", linestyle="--", label="Estimated maximum likelihood")]
        if family == "M3":
            marker_handles.append(Line2D([], [], color="#17212B", marker="D", markerfacecolor="white",
                                         linestyle="none", label="MLE target fit: iteration limit"))
        fig.legend(handles=marker_handles, loc="lower center", bbox_to_anchor=(.53, .107),
                   ncol=len(marker_handles), frameon=False, fontsize=10, columnspacing=2.3)
        foot = "No pooling across DGPs, ranking losses or city-link regimes. All tract types included; rho is residual correlation."
        fig.text(.075, .071, foot, fontsize=9.3, color="#536574")
        fig.text(.075, .048, "Common axis limits across M1 / M2 / M3 for each metric. True-parameter oracle: MSE = 0, outside the log axes.",
                 fontsize=9.3, color="#536574")
        extra = ("Scratch: 200 updates, shown only where tested; warm joint / continuation: 600 total. "
                 if any(a.startswith("scratch_") for a in relevant) else "")
        fig.text(.075, .025, extra + "One-seed comparisons have no Monte Carlo uncertainty intervals.",
                 fontsize=9.3, color="#536574")
        name = f"ordinal_pilot_{family}"
        fig.savefig(out / (name + ".png"), dpi=200)
        fig.savefig(out / (name + ".svg"))
        plt.close(fig)
        png_paths.append(str(out / (name + ".png")))
        notes.append(f"{family}: settings {settings}; {TITLES[family]}; the four panels show f, mu, delta f and delta mu MSE.")
    data_path = out / "scenario_plot_data.csv"
    with data_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(export_rows[0]))
        writer.writeheader(); writer.writerows(export_rows)
    caption = ("Ordinal joint-versus-two-step pilot: scenario figures\n\n" + "\n".join(notes) +
        "\n\nEvaluation: held-out target-city test tracts after separate local outcome-head calibration. "
        "Latent NN scores have a single positive source-fitting affine alignment to true f, used only for evaluation. "
        "All tract types are included. Symbols distinguish ranking losses, colors estimation procedures. "
        "All plotted values are mean squared errors, not RMSE. No averaging over distinct DGPs or link regimes. "
        "Maximum likelihood knows the structural basis, noise distribution/parameters and latent normalization. "
        "Its cell maps and structural coefficients are estimated. It is an information-advantaged reference, "
        "not a guaranteed finite-sample optimum. Open MLE diamonds flag target-calibration nonconvergence "
        "for outcome panels; source-only latent predictions do not depend on that target calibration. "
        "Oracle f and mu errors are zero and explicitly labeled outside the logarithmic axes. "
        "Scratch methods have 200 total updates; warm joint and ordinal continuation have 600, "
        "and A/B have 400 scorer updates. One seed supplies no Monte Carlo uncertainty estimate.\n")
    (out / "FIGURE_NOTES.txt").write_text(caption, encoding="utf-8")
    (out / "figure_provenance.json").write_text(json.dumps(dict(
        input_metrics=str(args.input.resolve()), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        seed_ids=seeds, metrics=METRICS, scenarios=FAMILIES, plotted_records=len(export_rows),
        means="Within setting/ranking/link/method over seed IDs; current data have one seed",
        axis_limits=limits), indent=2), encoding="utf-8")
    with zipfile.ZipFile(out / "ordinal_pilot_scenario_figures.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for pattern in ("ordinal_pilot_M*.png", "ordinal_pilot_M*.svg", "scenario_plot_data.csv",
                        "FIGURE_NOTES.txt", "figure_provenance.json"):
            for file in sorted(out.glob(pattern)):
                archive.write(file, file.name)
    print(json.dumps(dict(figures=png_paths, plotted_records=len(export_rows), seeds=seeds,
                          output=str(out.resolve())), indent=2))


if __name__ == "__main__":
    main()
