"""Publication-style mean-error and city-dispersion figures; no model fitting."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, LogLocator, NullLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "calibration_10pct" / "figures"
OUT.mkdir(exist_ok=True)
DATA = pd.read_csv(HERE / "calibration_10pct" / "five_rep_summary.csv")
METHODS = [
    ("Ordinal_S", "Ordinal", "#2166AC"),
    ("Ordinal_L2_S", "Ordinal + levels", "#15877B"),
    ("Ordinal_L2_D_S", "Full loss", "#DB6537"),
    ("Gaussian_ML_S", "Gaussian likelihood", "#8056A1"),
]
TARGETS = [
    ("f", "1  |  Structural levels", "Error in f", False),
    ("W_conditional_mean", "2  |  Outcome levels", "Error in W", True),
    ("delta_f", "3  |  Structural changes", "Error in changes in f", False),
    ("delta_W_conditional_mean", "4  |  Outcome changes", "Error in changes in W", True),
]
PANELS = [("heldout_seen", True), ("heldout_unseen", False)]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 15,
    "axes.labelsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 10,
    "text.color": "#263443", "axes.labelcolor": "#526171", "axes.edgecolor": "#CCD4DD",
    "xtick.color": "#526171", "ytick.color": "#526171", "pdf.fonttype": 42,
    "svg.fonttype": "none", "savefig.facecolor": "#FAFBFD",
})


def value(case, method, panel, target, column):
    cell = DATA[(DATA.case == case) & (DATA.method == method) &
                (DATA.panel == panel) & (DATA.target == target)]
    assert len(cell) == 1 and cell.replication_count.iloc[0] == 5
    return float(cell[column].iloc[0])


def draw(column, dispersion=False, mode="original", study_label=None, figure_title=None,
         figure_subtitle=None, panel_labels=None, footer_headline=None, extra_note=None):
    expanded = len(METHODS) > 4
    controlled = mode == "controlled"
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 10.2), facecolor="#FAFBFD")
    fig.subplots_adjust(left=.085, right=.98, bottom=.165, top=.695 if expanded else .735,
                        wspace=.22, hspace=.42)
    fig.text(.065, .958, study_label or "CITY CALIBRATION  /  FIVE-REPLICATION PILOT", fontsize=10,
             color="#60788B", weight="bold")
    fig.text(.065, .914,
             figure_title or ("How much does error vary across cities?" if dispersion else
             ("Should the change loss use only changed tracts?" if controlled else
              "Which learned score performs better?")),
             fontsize=25, weight="bold", color="#182D40")
    fig.text(.065, .877,
             figure_subtitle or ("Same level and stability weights. Compare tract selection and change-loss strength."
              if controlled else
              "Same 10% calibration for every method. Scores frozen; city mappings fitted from scratch."),
             fontsize=11.5, color="#526171")
    method_handles = [Line2D([], [], marker="o", linestyle="", markersize=8, color=c, label=l)
                      for _, l, c in METHODS]
    fig.legend(handles=method_handles, loc="upper left", bbox_to_anchor=(.058, .851),
               ncol=3 if expanded else len(METHODS), frameon=False,
               columnspacing=1.9, handletextpad=.35, fontsize=10.4)
    panel_labels = panel_labels or ("Held-out tracts in training cities", "Held-out tracts in new cities")
    panel_handles = [
        Line2D([], [], marker="o", linestyle="", color="#516577", markersize=7,
               label=panel_labels[0]),
        Line2D([], [], marker="o", linestyle="", color="#516577", markerfacecolor="white",
               markersize=7, label=panel_labels[1]),
    ]
    fig.legend(handles=panel_handles, loc="upper left", bbox_to_anchor=(.058, .775 if expanded else .815),
               ncol=2, frameon=False, fontsize=10, columnspacing=2.1, handletextpad=.35)

    for ax, (target, title, _, logarithmic) in zip(axes.flat, TARGETS):
        ax.set_facecolor("white")
        ax.set_title(title, loc="left", pad=13, weight="bold")
        for i in range(3):
            ax.axvspan(i-.40, i+.40, color="#F4F6F9", zorder=0)
        for i, case in enumerate("ABC"):
            for j, (method, _, color) in enumerate(METHODS):
                spacing = .62 / len(METHODS)
                x = i + (j-(len(METHODS)-1)/2)*spacing
                ys = [value(case, method, panel, target, column) for panel, _ in PANELS]
                if not logarithmic:
                    ys = [v*1e6 for v in ys]
                xs = [x-.024, x+.024]
                ax.plot(xs, ys, color=color, alpha=.55, linewidth=1.2, zorder=3)
                for xx, yy, (_, filled) in zip(xs, ys, PANELS):
                    ax.scatter(xx, yy, s=56, facecolors=color if filled else "white",
                               edgecolors=color, linewidths=1.65, zorder=5)
        label = "City SD of MSE" if dispersion else "Mean city MSE"
        if logarithmic:
            ax.set_yscale("log")
            subs = (1.,) if dispersion and target == "delta_W_conditional_mean" else (1., 2., 5.)
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs, numticks=12))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
            ax.set_ylabel(label + "  ·  log scale")
            ax.margins(y=.13)
        else:
            ax.set_ylim(bottom=0)
            ax.set_ylabel(label + "  (×10⁻⁶)")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.margins(y=.17)
        ax.set_xlim(-.5, 2.5)
        ax.set_xticks([0, 1, 2], ["A\nIntercept", "B\nAffine", "C\nCubic"])
        ax.tick_params(axis="both", which="both", length=0, pad=7)
        ax.grid(axis="y", which="major", color="#DFE5EB", linewidth=.7, zorder=1)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#E1E6ED")

    headline = ("City SD measures differences between cities; it is not uncertainty around the average."
                if dispersion else
                ("Changed-only uses true simulation status and preserves each retained pair's weight."
                 if mode != "original" else
                 "Full-loss change errors remain higher. Outcome differences narrow after equal calibration."))
    fig.text(.065, .083, footer_headline or headline, fontsize=12, weight="bold", color="#263F53")
    note = ("Dots average the within-replication SD of 20 city MSEs over five replications. These are not standard errors."
            if dispersion else
            "Dots average city MSE over 20 cities and five replications. Lower is better; city SD is shown separately.")
    fig.text(.065, .054, note, fontsize=9.4, color="#607181")
    fig.text(.065, .032,
             "28 paired calibration tracts + 252 test tracts per city. All methods use stable acquisition pairs. "
             "W targets are noise-free conditional means.", fontsize=9.1, color="#607181")
    if extra_note:
        fig.text(.065, .013, extra_note, fontsize=8.7, color="#607181")
    stem = "city_variability" if dispersion else "equal_calibration_comparison"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--changed-summary", type=Path)
    args = parser.parse_args()
    if args.changed_summary is None:
        draw("mean_city_mse")
        draw("mean_within_rep_city_sd", dispersion=True)
        print(OUT)
    else:
        new_data = pd.read_csv(args.changed_summary)
        keys = ["case", "method", "panel", "target"]
        DATA = pd.concat([DATA, new_data], ignore_index=True).drop_duplicates(keys, keep="last")
        extra = [
            ("Ordinal_L2_D_all_matched", "+ W changes: all tracts", "#B83C52"),
            ("Ordinal_L2_D_changed_matched", "+ W changes: changed only", "#A18316"),
        ]
        METHODS[2] = ("Ordinal_L2_D_S", "Full loss: original weights", "#DB6537")
        METHODS += extra
        OUT = args.changed_summary.parent / "figures" / "all_methods"
        OUT.mkdir(parents=True, exist_ok=True)
        draw("mean_city_mse", mode="expanded")
        draw("mean_within_rep_city_sd", dispersion=True, mode="expanded")
        METHODS = [
            ("Ordinal_L2_S", "No W-change loss", "#15877B"),
            ("Ordinal_L2_D_all_matched", "All tracts", "#B83C52"),
            ("Ordinal_L2_D_changed_matched", "Changed tracts only", "#A18316"),
        ]
        if "Ordinal_L2_D_all_massmatched" in DATA.method.values:
            METHODS.append(("Ordinal_L2_D_all_massmatched", "All tracts: lower weight", "#8056A1"))
        OUT = args.changed_summary.parent / "figures" / "matched_comparison"
        OUT.mkdir(parents=True, exist_ok=True)
        draw("mean_city_mse", mode="controlled")
        draw("mean_within_rep_city_sd", dispersion=True, mode="controlled")
        print(OUT.parent)
