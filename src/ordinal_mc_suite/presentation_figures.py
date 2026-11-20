"""Presentation figures for the Vives Lab deck, built from finished MC summaries.

Reads only the exported ``summary.csv`` / ``paired_summary.csv`` of completed runs;
it never refits anything. Intervals are the suite's own ``ci95_low``/``ci95_high``
(mean +/- 1.96 Monte Carlo SE across independent replications).

    python presentation_figures.py [--out <dir>]
    python presentation_figures.py --e1-additive [--e1-run <dir>] [--scale 10]

Default output: ``paper/Presentacion Vives Lab/figures/mc`` relative to the repo root.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

SUITE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUITE_DIR.parents[1]
DECK_DIR = REPO_ROOT / "paper" / "Presentacion Vives Lab"
DEFAULT_OUT = DECK_DIR / "figures" / "mc"

# Light-mode colours of beamerthememarginal.sty
INK = "#3B3F41"
MUTED = "#677176"
RULE = "#C9CDD0"
VIOLET = "#7459B4"  # mgAccent: the proposed estimator
BLUE = "#2F7899"
RED = "#C8253D"
YELLOW = "#8F6802"

RUNS = {
    "E1": SUITE_DIR / "overnight_measurement",
    "E3_nn": SUITE_DIR / "overnight_nn",
    "E3_linear": SUITE_DIR / "linear_extended",
    "E4": SUITE_DIR / "overnight_links",
}


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def load_csv(run_dir: Path, name: str) -> pd.DataFrame:
    """Read one exported CSV of a finished run."""
    original=Path(run_dir)/name
    augmented=original.with_name(original.stem+'_with_ml'+original.suffix)
    return pd.read_csv(augmented if augmented.exists() else original)


def select(df: pd.DataFrame, **filters) -> pd.DataFrame:
    """Rows matching every ``column=value`` filter (values compared as given)."""
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask &= df[col] == val
    return df[mask]


@dataclass
class Curve:
    """One plotted series: x, mean and 95% band, plus its legend style."""
    label: str
    x: list
    mean: list
    lo: list
    hi: list
    color: str = INK
    marker: str = "o"
    linestyle: str = "-"
    extra: dict = field(default_factory=dict)


def curve(df: pd.DataFrame, label: str, color: str, x: str = "cities",
          scale: float = 100.0, **style) -> Curve:
    """Build a Curve from summary rows (one row per x); values scaled to percent."""
    if df.empty:
        raise ValueError(f"no rows for curve '{label}'")
    if df[x].duplicated().any():
        raise ValueError(f"curve '{label}' has duplicated {x} values: filters too loose")
    d = df.sort_values(x)
    return Curve(label, d[x].tolist(), (d["mean"] * scale).tolist(),
                 (d["ci95_low"] * scale).tolist(), (d["ci95_high"] * scale).tolist(),
                 color=color, **style)


# ---------------------------------------------------------------------------
# Styling and drawing
# ---------------------------------------------------------------------------
def setup_style(deck_dir: Path = DECK_DIR) -> None:
    """Match the deck: iA Writer Quattro if available, thin ink axes, no top/right spines."""
    font = deck_dir / "fonts" / "iAWriterQuattroS-Regular.ttf"
    family = "DejaVu Sans"
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        bold = deck_dir / "fonts" / "iAWriterQuattroS-Bold.ttf"
        if bold.exists():
            font_manager.fontManager.addfont(str(bold))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({
        "font.family": family, "font.size": 9, "axes.titlesize": 9.5,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.transparent": True, "pdf.fonttype": 42,
    })


def draw_curves(ax, curves: list[Curve], xticks=(8, 16, 32, 64)) -> None:
    """Mean line with markers and a light 95% band for each curve."""
    for c in curves:
        ax.fill_between(c.x, c.lo, c.hi, color=c.color, alpha=0.13, lw=0)
        ax.plot(c.x, c.mean, color=c.color, marker=c.marker, ms=4, lw=1.6,
                ls=c.linestyle, label=c.label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(xticks))
    ax.set_xticklabels([str(t) for t in xticks])
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=RULE, lw=0.5)
    ax.set_axisbelow(True)


def with_ml(curves: list[Curve], rows: pd.DataFrame) -> list[Curve]:
    """Show the benchmark when reading an upgraded run; retain older figures."""
    ml=select(rows,method='ml_correct')
    if not ml.empty:
        curves.append(curve(ml,'correctly specified ML (oracle nuisance)',YELLOW,linestyle=':'))
    return curves


def finish_legend(fig, ax, ncol=2):
    """Keep the longer ML legend outside the data panels."""
    handles,labels=ax.get_legend_handles_labels()
    width,height=fig.get_size_inches()
    fig.set_size_inches(width,height+.45)
    fig.legend(handles,labels,loc='lower center',ncol=ncol,fontsize=7.5,
               bbox_to_anchor=(.5,0.),frameon=False)
    fig.tight_layout(rect=(0,.18,1,1))


def save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    if out.suffix.lower()=='.pdf':
        fig.savefig(out.with_suffix('.png'),dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
E1_METHODS = [  # (method, legend, colour, linestyle)
    ("raw_levels", "pooled $L_2$ on dollars", RED, "-"),
    ("log_cell_fe", "log + city-year FE", BLUE, "--"),
    ("rank_common", "ranking (no links)", VIOLET, "-"),
]
E1_PANELS = [
    ("multiplier", r"$\psi_{ct}$: multiplier (log + FE correct)"),
    ("shape", r"$\psi_{ct}$: level + shape (log + FE misspecified)"),
]


def fig_e1(summary: pd.DataFrame, composition: float, metric: str = "unseen_within_latent",
           ylabel: str = "misranked pairs, unseen cities (%)"):
    """E1: three estimators vs number of labeled cities, one panel per label map."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, (stage, title) in zip(axes, E1_PANELS):
        rows = select(summary, metric=metric, psi_stage=stage, composition=composition)
        curves = [curve(select(rows, method=m), lab, col, linestyle=ls)
                  for m, lab, col, ls in E1_METHODS]
        draw_curves(ax, with_ml(curves,rows))
        ax.set_title(title, loc="left")
        ax.set_xlabel("labeled training cities $C$")
    axes[0].set_ylabel(ylabel)
    finish_legend(fig,axes[1])
    return fig


def fig_links(panels: list[tuple[pd.DataFrame, dict, str]],
              metric: str = "unseen_within_latent",
              ylabel: str = "misranked pairs, unseen cities (%)"):
    """E3: no links vs temporal links. ``panels`` = [(summary, row filters, title)];
    a panel's filters may override ``metric``."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 2.7), squeeze=False)
    for ax, (summary, filt, title) in zip(axes[0], panels):
        rows = select(summary, **{"metric": metric, **filt})
        curves = [curve(select(rows, method="none"), "no links", MUTED, linestyle="--"),
                  curve(select(rows, method="temporal"), "temporal links", VIOLET)]
        draw_curves(ax, with_ml(curves,rows))
        ax.set_title(title, loc="left")
        ax.set_xlabel("labeled training cities $C$")
    axes[0][0].set_ylabel(ylabel)
    finish_legend(fig,axes[0][-1])
    return fig


E4_CASES = [("exact", "exact"), ("drift", "drift 0.15"),
            ("random_false", "20% random\nfalse links"),
            ("systematic_false", "20% one-sided\nfalse links")]
E4_METHODS = [("temporal_fixed", r"fixed $\lambda_t=0.3$", VIOLET),
              ("temporal_tuned", r"$\lambda_t$ tuned on validation", YELLOW)]


def fig_e4(paired: pd.DataFrame, metric: str = "unseen_within_latent"):
    """E4: paired reduction in misranking vs no links (pp, 95% CI), by link regime."""
    rows = select(paired, metric=metric)
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    methods=E4_METHODS+([('ml_correct','correctly specified ML',BLUE)] if 'ml_correct' in set(rows.method) else [])
    width = .72/len(methods)
    for k, (method, label, color) in enumerate(methods):
        for i, (case, _) in enumerate(E4_CASES):
            r = select(rows, method=method, link_case=case)
            if len(r) != 1:
                raise ValueError(f"expected one row for {method}/{case}, got {len(r)}")
            r = r.iloc[0]
            x = i + (k - (len(methods)-1)/2) * width
            m, lo, hi = 100 * r["mean"], 100 * r["ci95_low"], 100 * r["ci95_high"]
            ax.bar(x, m, width * 0.92, color=color, alpha=0.85, label=label if i == 0 else None)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=INK, lw=0.9, capsize=2.5)
            rel = 100 * r["relative_reduction_of_mean"]
            ax.annotate(f"{rel:+.0f}%", (x, max(hi, 0) + 0.02), ha="center",
                        va="bottom", fontsize=7, color=MUTED)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks(range(len(E4_CASES)))
    ax.set_xticklabels([lab for _, lab in E4_CASES])
    ax.set_ylabel("misranking reduction vs.\nno links (pp)")
    ax.grid(axis="y", color=RULE, lw=0.5)
    ax.set_axisbelow(True)
    finish_legend(fig,ax,ncol=3)
    return fig


def fig_cross_city(e1: pd.DataFrame, e3_lin: pd.DataFrame):
    """Cross-city misranking: no estimator here is designed to align city levels."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    metric = "unseen_cross_city_latent"
    rows = select(e1, metric=metric, psi_stage="shape", composition=0.0)
    draw_curves(axes[0], with_ml([curve(select(rows, method=m), lab, col, linestyle=ls)
                          for m, lab, col, ls in E1_METHODS],rows))
    axes[0].set_title("E1 estimators (shape map)", loc="left")
    rows = select(e3_lin, metric=metric)
    draw_curves(axes[1], with_ml([curve(select(rows, method="none"), "no links", MUTED, linestyle="--"),
                          curve(select(rows, method="temporal"), "temporal links", VIOLET)],rows))
    axes[1].set_title("E3 linear: links within city", loc="left")
    for ax in axes:
        ax.set_xlabel("labeled training cities $C$")
        ax.legend(loc="lower left")
    axes[0].set_ylabel("misranked cross-city pairs,\nunseen cities (%)")
    fig.tight_layout()
    return fig


E1_ADDITIVE_METHODS = [  # (method, legend, colour, linestyle)
    ("raw_levels", "pooled $L_2$ on dollars", RED, "-"),
    ("log_cell_fe", "log + city-year FE", BLUE, "--"),
    ("rank_common", "ranking", VIOLET, "-"),
]


def e1_additive_run(suite_dir: Path = SUITE_DIR) -> Path | None:
    """Finished additive-baseline E1: the trained run if present, else the exact diagnostic."""
    for name in ("e1_additive", "e1_additive_exact"):
        if (Path(suite_dir) / name / "summary.csv").exists():
            return Path(suite_dir) / name
    return None


def fig_e1_additive(summary: pd.DataFrame, scale: float = 10.0,
                    metric: str = "unseen_within_latent"):
    """E1 with an additive baseline: (a) multiplier map, (b) baseline of size ``scale``.

    Plots every E1 method present in the summary (pooled L2 only if it was run).
    """
    panels = [(dict(psi_stage="multiplier", psi_strength=1.0), r"(a) $W=b_{ct}\,e^{0.6R}$"),
              (dict(psi_stage="additive", psi_strength=float(scale)),
               r"(b) $W=a_{ct}+b_{ct}\,e^{0.6R}$")]
    rows = select(summary, metric=metric)
    if "composition" in rows:
        rows = rows[rows.composition == 0]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    top = 0.0
    for ax, (filt, title) in zip(axes, panels):
        panel = select(rows, **filt)
        if panel.empty:
            raise ValueError(f"no rows for {filt}: was this map run?")
        curves = [curve(select(panel, method=m), lab, col, linestyle=ls)
                  for m, lab, col, ls in E1_ADDITIVE_METHODS if m in set(panel.method)]
        curves=with_ml(curves,panel)
        draw_curves(ax, curves)
        top = max(top, *(max(c.hi) for c in curves))
        ax.set_title(title, loc="left")
        ax.set_xlabel("labeled training cities $C$")
    axes[0].set_ylim(0, 1.08 * top)  # shared axis: fit the higher panel's bands
    axes[0].set_ylabel("misranked pairs, unseen cities (%)")
    finish_legend(fig,axes[1])
    return fig


def build_e1_additive(run_dir: Path, out_dir: Path = DEFAULT_OUT, scale: float = 10.0) -> Path:
    """Write mc_e1_additive.pdf from a finished additive-baseline E1 run."""
    setup_style()
    return save(fig_e1_additive(load_csv(run_dir, "summary.csv"), scale),
                Path(out_dir) / "mc_e1_additive.pdf")


def invariance_shift(summary: pd.DataFrame, composition: float = 0.0,
                     metric: str = "unseen_within_latent") -> pd.DataFrame:
    """Change in error (pp) when the label map goes from multiplier to level+shape.

    E1 shares latent draws across the two maps, so an estimator that only uses
    within-cell orderings must show exactly zero shift. Returns method, cities, shift_pp.
    """
    rows = select(summary, metric=metric, composition=composition)
    wide = rows.pivot_table(index=["method", "cities"], columns="psi_stage", values="mean")
    missing = {"multiplier", "shape"} - set(wide.columns)
    if missing:
        raise ValueError(f"summary lacks psi_stage {sorted(missing)}")
    out = (100 * (wide["shape"] - wide["multiplier"])).rename("shift_pp").reset_index()
    return out


def fig_invariance(summary: pd.DataFrame, composition: float = 0.0):
    """E1: bars of the error shift caused by changing psi, by method and C."""
    shift = invariance_shift(summary, composition)
    cities = sorted(shift["cities"].unique())
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    methods=E1_METHODS+([('ml_correct','correctly specified ML',YELLOW,':')] if 'ml_correct' in set(shift.method) else [])
    width = 0.8 / len(methods)
    for k, (method, label, color, _) in enumerate(methods):
        d = select(shift, method=method).set_index("cities").reindex(cities)
        xs = np.arange(len(cities)) + (k - (len(methods) - 1) / 2) * width
        ax.bar(xs, d["shift_pp"], width * 0.9, color=color, alpha=0.85, label=label)
        if np.allclose(d["shift_pp"], 0, atol=1e-9):
            for x in xs:
                ax.annotate("0", (x, 0), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7, color=color)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks(range(len(cities)))
    ax.set_xticklabels([f"C = {c}" for c in cities])
    ax.set_ylabel("error change when $\\psi_{ct}$\nchanges shape (pp)")
    ax.grid(axis="y", color=RULE, lw=0.5)
    ax.set_axisbelow(True)
    finish_legend(fig,ax)
    return fig


def equivalent_cities(x: list, y: list, target: float) -> float | None:
    """C at which a decreasing error curve (x on a log2 axis) reaches ``target``.

    Linear interpolation in log2(C); None if ``target`` is outside the curve's range.
    """
    lx = np.log2(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float)
    for i in range(len(y) - 1):
        hi, lo = y[i], y[i + 1]
        if lo <= target <= hi and hi > lo:
            frac = (hi - target) / (hi - lo)
            return float(2 ** (lx[i] + frac * (lx[i + 1] - lx[i])))
    return None


def fig_doubling(summary: pd.DataFrame, metric: str = "unseen_within_latent",
                 title: str = "linear truth, linear estimator (50 reps)"):
    """E3: no links vs links, with arrows to the C the unlinked model needs to match."""
    rows = select(summary, metric=metric)
    none = curve(select(rows, method="none"), "no links", MUTED, linestyle="--")
    linked = curve(select(rows, method="temporal"), "temporal links", VIOLET)
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    draw_curves(ax, with_ml([none, linked],rows))
    for c, m in zip(linked.x, linked.mean):
        c_eq = equivalent_cities(none.x, none.mean, m)
        if c_eq is None:
            continue
        ax.annotate("", xy=(c_eq, m), xytext=(c, m),
                    arrowprops=dict(arrowstyle="->", color=VIOLET, lw=0.9, ls=":"))
        ax.annotate(f"$\\approx${c_eq / c:.1f}$\\times$", (c_eq, m), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=7.5,
                    color=VIOLET, bbox=dict(boxstyle="square,pad=0.15", fc="white",
                                            ec="none", alpha=0.9))
    ax.set_title(title, loc="left")
    ax.set_xlabel("labeled training cities $C$")
    ax.set_ylabel("misranked pairs, unseen cities (%)")
    finish_legend(fig,ax)
    return fig


def fig_paired_strip(panels: list[tuple[pd.DataFrame, dict, str]],
                     metric: str = "unseen_within_latent", seed: int = 0):
    """Per-replication paired gains (pp) of links over no links, one dot per replication.

    ``panels`` = [(paired_replications, row filters, title)]. Each column is annotated
    with the share of replications in which links lowered the error.
    """
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 2.6),
                             squeeze=False)
    for ax, (paired, filt, title) in zip(axes[0], panels):
        rows = select(paired, **{"metric": metric, "method": "temporal", **filt})
        if rows.empty:
            raise ValueError(f"no paired rows for panel '{title}'")
        cities = sorted(rows["cities"].unique())
        for i, c in enumerate(cities):
            g = 100 * select(rows, cities=c)["gain"].to_numpy()
            ax.scatter(i + rng.uniform(-0.18, 0.18, g.size), g, s=7, color=VIOLET,
                       alpha=0.45, lw=0)
            ax.plot([i - 0.26, i + 0.26], [g.mean()] * 2, color=INK, lw=1.4)
            ax.annotate(f"{(g > 0).mean():.0%}", (i, 1.0), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=7, color=MUTED)
        ax.axhline(0, color=INK, lw=0.7)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.15 * (hi - lo))  # headroom for the share labels
        ax.set_xticks(range(len(cities)))
        ax.set_xticklabels([str(c) for c in cities])
        ax.set_xlabel("labeled training cities $C$")
        ax.set_title(title, loc="left", pad=12)
        ax.grid(axis="y", color=RULE, lw=0.5)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("error reduction from\nlinks, per replication (pp)")
    fig.tight_layout()
    return fig


def build_all(out_dir: Path = DEFAULT_OUT, runs: dict = RUNS) -> list[Path]:
    """Write every deck figure; returns the paths written."""
    setup_style()
    e1 = load_csv(runs["E1"], "summary.csv")
    nn = load_csv(runs["E3_nn"], "summary.csv")
    lin = load_csv(runs["E3_linear"], "summary.csv")
    e4 = load_csv(runs["E4"], "paired_summary.csv")
    nn_reps = load_csv(runs["E3_nn"], "paired_replications.csv")
    lin_reps = load_csv(runs["E3_linear"], "paired_replications.csv")
    written = [
        save(fig_invariance(e1), out_dir / "mc_e1_invariance.pdf"),
        save(fig_doubling(lin), out_dir / "mc_e3_doubling.pdf"),
        save(fig_links([(nn, {"truth": "linear"}, "linear truth, neural net (30 reps)"),
                        (nn, {"truth": "quadratic"}, "quadratic truth, neural net (30 reps)")]),
             out_dir / "mc_e3_nn.pdf"),
        save(fig_paired_strip([(lin_reps, {}, "linear / linear"),
                               (nn_reps, {"truth": "linear"}, "linear / NN"),
                               (nn_reps, {"truth": "quadratic"}, "quadratic / NN")]),
             out_dir / "mc_e3_paired.pdf"),
        save(fig_e1(e1, 0.0), out_dir / "mc_e1_measurement.pdf"),
        save(fig_e1(e1, 0.7), out_dir / "mc_e1_composition.pdf"),
        save(fig_links([(lin, {}, "linear truth, linear estimator (50 reps)"),
                        (nn, {"truth": "quadratic"}, "quadratic truth, neural net (30 reps)")]),
             out_dir / "mc_e3_links.pdf"),
        save(fig_links([(nn, {"truth": "linear"}, "linear truth, neural net (30 reps)"),
                        (lin, {"metric": "unseen_across_time_latent"},
                         "linear/linear: across-date pairs")],
                       ylabel="misranked pairs, unseen cities (%)"),
             out_dir / "mc_e3_backup.pdf"),
        save(fig_e4(e4), out_dir / "mc_e4_links.pdf"),
        save(fig_cross_city(e1, lin), out_dir / "mc_cross_city.pdf"),
    ]
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--e1-additive", action="store_true",
                    help="only write mc_e1_additive.pdf (additive-baseline E1)")
    ap.add_argument("--e1-run", type=Path, default=None,
                    help="run folder for --e1-additive (default: e1_additive, else e1_additive_exact)")
    ap.add_argument("--scale", type=float, default=10.0, help="baseline size s for panel (b)")
    args = ap.parse_args()
    if args.e1_additive:
        run_dir = args.e1_run or e1_additive_run()
        if run_dir is None:
            raise SystemExit("no finished additive E1 run: run e1_additive_exact.py or configs/e1_additive.json")
        print(build_e1_additive(run_dir, args.out, args.scale))
    else:
        for p in build_all(args.out):
            print(p)
