"""Per-city rank autocorrelation of stable buildings, predictions vs labels (deck figure).

For every held-out (test) CBSA, the cross-year Spearman of stable buildings'
predictions is drawn next to the same statistic for their ACS labels — the
genuine-reshuffling benchmark (see ``utils.metrics.rank_autocorrelation_by_city``).

    python src/plot_autocorr_dumbbell.py [--run run_20260722] [--split test] [--out <pdf>]

Writes the per-city table next to the figure (``<out stem>.csv``).
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from src.utils.metrics import rank_autocorrelation_by_city  # noqa: E402
from src.utils.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, RESULTS_DIR  # noqa: E402

DECK_DIR = PROJECT_ROOT / "paper" / "Presentacion Vives Lab"
DEFAULT_OUT = DECK_DIR / "figures" / "US_autocorr_dumbbell.pdf"

# beamerthememarginal light-mode colours
INK, MUTED, RULE = "#3B3F41", "#677176", "#C9CDD0"
VIOLET, BLUE = "#7459B4", "#2F7899"


def city_autocorr(bld: pd.DataFrame, split: str = "test", names: dict | None = None,
                  min_common: int = 5) -> pd.DataFrame:
    """Per-city pred/label rank autocorrelation on one split, with a display name."""
    sub = bld[bld["type"] == split] if "type" in bld.columns else bld
    out = rank_autocorrelation_by_city(sub, min_common)
    names = names or {}
    out["name"] = [names.get(str(c), str(c)) for c in out["cbsa"]]
    return out.sort_values("rho_l").reset_index(drop=True)


def weighted_means(df: pd.DataFrame) -> tuple[float, float]:
    """n-weighted (pred, label) means, identical to utils.metrics.rank_autocorrelation."""
    w = df["n"].to_numpy(dtype=float)
    return float(np.average(df["rho_p"], weights=w)), float(np.average(df["rho_l"], weights=w))


def setup_style() -> None:
    """Deck font if available, thin ink axes (resets styles set by importing evaluation)."""
    plt.rcdefaults()
    font = DECK_DIR / "fonts" / "iAWriterQuattroS-Regular.ttf"
    family = "DejaVu Sans"
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({
        "font.family": family, "font.size": 8, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "text.color": INK, "axes.linewidth": 0.7, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "savefig.transparent": True, "pdf.fonttype": 42,
    })


def fig_dumbbell(df: pd.DataFrame):
    """One column per city: label autocorrelation (benchmark) and prediction autocorrelation."""
    if df.empty:
        raise ValueError("no cities to plot")
    fig, ax = plt.subplots(figsize=(0.17 * len(df) + 1.2, 3.1))
    x = np.arange(len(df))
    ax.vlines(x, df[["rho_p", "rho_l"]].min(axis=1), df[["rho_p", "rho_l"]].max(axis=1),
              color=RULE, lw=1.6, zorder=1)
    ax.scatter(x, df["rho_l"], s=14, color=BLUE, marker="s", zorder=2, label="ACS labels")
    ax.scatter(x, df["rho_p"], s=16, color=VIOLET, zorder=3, label="predictions")
    mp, ml = weighted_means(df)
    ax.axhline(ml, color=BLUE, lw=0.7, ls="--")
    ax.axhline(mp, color=VIOLET, lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(df["name"], fontsize=6, rotation=60, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.7, len(df) - 0.3)
    ax.set_ylabel("rank autocorrelation of\nstable buildings across years")
    ax.grid(axis="y", color=RULE, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7,
              title=f"dashed: weighted means, predictions {mp:.3f} · labels {ml:.3f}",
              title_fontsize=7)
    fig.tight_layout()
    return fig


def load_buildings(run_dir: Path, indicator: str | None = None):
    """Building predictions with cbsa/change/type attached, as the US evaluation loads them.

    ``indicator`` defaults to the evaluation default, as in ``run_evaluation``.
    """
    from src.data import indicators
    from src.evaluation import _USContext, _city_names  # heavy import
    ctx = _USContext(run_dir, PROCESSED_DATA_DIR, Path(tempfile.mkdtemp()),
                     indicator or indicators.DEFAULT_INDICATOR)
    return ctx.bld, _city_names(ctx.cbsa_meta)


def main(run: str, split: str, out: Path) -> Path:
    run_dir = RESULTS_DIR / run
    bld, names = load_buildings(run_dir)
    df = city_autocorr(bld, split, names)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = out.with_suffix(".csv")
    df.to_csv(table, index=False)
    mp, ml = weighted_means(df)
    print(f"{len(df)} cities; weighted pred={mp:.4f} label={ml:.4f}; table -> {table}")
    setup_style()
    fig = fig_dumbbell(df)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="run_20260722")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    print(main(a.run, a.split, a.out))
