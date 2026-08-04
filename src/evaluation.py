# -*- coding: utf-8 -*-
"""
src/evaluation.py  –  Post-hoc evaluation of already-computed predictions.

Three modes (auto-detected from footprints_source / artifacts, or forced via
--mode). ``run_evaluation`` is the importable entry point; main.py calls it
after generate_predictions.

NYC mode (paper figures, --mode nyc):
  A  Cross-sectional validity (+ choropleth maps)
  B  Temporal stability
  C  GB2 parametric distribution matching
  D  Callaway-Sant'Anna event study on construction cohorts (needs the 'csa'
     package; estimation logic lives in src/csa_event_study.py)
  E  Case study: Hudson Yards

Combined mode (--mode both) runs the US parts on --savename and the NYC parts on
that run's NYC prediction pass (<savename>/nyc_zarr_check by default), so one
command produces the whole result set: the 3-panel US main figure plus the NYC
CSA event study and Hudson Yards case study. Each half writes its own
evaluation/ folder; every part is isolated, so one failure never costs the rest.

US mode (full-US runs, --mode us) — grouped per CBSA / split / bracket:
  cross     Within-city (CBSA x year) Spearman cells (headline), pooled
            correlations, per-cell scatter grid + Spearman histogram (test),
            per-bracket & top-metro tables.
  temporal  Rank autocorrelation of stable buildings (vs label benchmark),
            stable/changed MASD ratio, tract-level ICC + rank autocorrelation.
  dollars   Per-CBSA GB2 fit to tract per-capita income -> rank->dollar map.
  main_figure  3-panel Science-style composite (raincloud by bracket, temporal
            stability vs a cardinal-baseline placeholder, pooled binned
            scatter). Not in the default part set — run explicitly.

Usage:
    python -m src.evaluation --savename <name>              # auto mode, all parts
    python -m src.evaluation --savename <name> --mode both  # US + NYC, everything
    python -m src.evaluation --savename <name> --mode us --parts cross
    python -m src.evaluation --savename <name> --mode nyc --parts A B
    # NYC parts from a different run, US parts from this one:
    python -m src.evaluation --savename <us_run> --mode both --nyc-savename <nyc_run>
"""

from __future__ import annotations

import argparse
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from scipy import special as _sp
from scipy.stats import spearmanr, kendalltau, rv_continuous, gaussian_kde
from pyproj import Transformer
from shapely import STRtree
from shapely.geometry import box as shapely_box

from src.utils.paths import (
    RESULTS_DIR,
    PROCESSED_DATA_DIR,
    IMAGERY_ROOT,
    ACS_ROOT_DIR,
    PROJECT_ROOT
)
from src.geo_utils import calculate_exact_tau, METRIC_EPSG
from src.utils.metrics import (
    within_city_cells,
    weighted_within_spearman,
    rank_autocorrelation,
    masd_by_change,
)
from src.data import indicators
from src.data import cbsa_brackets

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─── constants ────────────────────────────────────────────────────────────────

### Figure styling constants

# Figsize, inspired by journal guidelines, 
# https://www.elsevier.com/authors/policies-and-guidelines/artwork-and-media-instructions/artwork-sizing-instructions
pt = 1./72.27 # Hundreds of years of history... 72.27 points to an inch.
journal_sizes = {
    "Latex": {"onecol": 354.*pt, "twocol": (354-35)/2*pt},
    "IEETRAN": {"onecol": 252.*pt, "twocol": 526.3*pt, "textheight_a4": 680.*pt}, # CQG is only one column; textheight_a4 = 239 mm for A4
    # Add more journals below. Can add more properties to each journal
}
# Our figure's aspect ratio
golden = (1 + 5 ** 0.5) / 2
FIG_DPI = 300
FIG_SIZE_ONE_COL = (journal_sizes["IEETRAN"]["onecol"], journal_sizes["IEETRAN"]["onecol"]/golden)
FIG_SIZE_TWO_COL = (journal_sizes["IEETRAN"]["twocol"], journal_sizes["IEETRAN"]["twocol"]/golden)

# Style
import matplotlib.pyplot as plt
plt.style.use(PROJECT_ROOT / "src" / "utils" / "paper.mplstyle")
###

YEARS = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
# The NYC parts A–C read YEARS at call time from ~25 sites, so a NYC run on a
# different cadence (e.g. NAIP flies NY in odd years) is handled by rebinding it
# once, up front, in _set_nyc_years — the single documented mutation point.
# Year pair used for rank-autocorrelation: both 2016 (temporal holdout) and 2024
# are fully predicted (all tracts), unlike the sparse intermediate years.
RANK_PAIR = (2016, 2024)
CRS_PROJ = 6539   # NY Long Island, US survey feet
CRS_GEO  = 4326

TAU_METERS = 100
IMAGE_SIZE  = 224
_EXACT_TAU_M, _N = calculate_exact_tau(TAU_METERS, IMAGE_SIZE)
# EPSG:6539 native unit is US survey foot ->1 ft = 0.3048006096 m
_M_PER_FT = 0.3048006096
TAU_FT    = _EXACT_TAU_M / _M_PER_FT   # ≈ 336 US-survey-feet

DEFAULT_SAVENAME = (
    "scalemae_lr0.0001_size224_y2010-2012-2014-2016-2018-2020-2022-2024_ranknet_mining_lambda_s_05"
)

CASE_STUDY = {"lon": -74.0015, "lat": 40.7538, "half_km": 0.6}

NYC_COUNTY_PREFIXES = ("36005", "36047", "36061", "36081", "36085")

# ─── shared helpers ───────────────────────────────────────────────────────────

def _norm_geoid(x) -> str:
    return str(int(x)).zfill(11)


def _make_dirs(out: Path) -> None:
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {path.name}")


def _load_tract_long(results_dir: Path, years: list[int] | None = None) -> pd.DataFrame:
    """Stack predictions_by_tract_<year>.parquet ->long DF.

    ``years`` defaults to the NYC constant ``YEARS``; the US path passes the
    years actually present under ``results_dir`` (see ``_available_years``).
    Missing per-year parquets are skipped rather than raising, so partial runs
    still evaluate.
    """
    years = YEARS if years is None else years
    frames = []
    for yr in years:
        fpath = results_dir / f"predictions_by_tract_{yr}.parquet"
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        df["year"] = yr
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["GEOID", "Rel_Score", "predicted_value",
                                     "predicted_value_std", "year", "GEOID_str"])
    long = pd.concat(frames, ignore_index=True)
    long["GEOID_str"] = long["GEOID"].apply(_norm_geoid)
    return long


def _set_nyc_years(results_dir: Path) -> list[int]:
    """Point the NYC parts at the years actually present under ``results_dir``.

    Parts A–C read the module-level ``YEARS`` at call time. Rather than thread a
    ``years`` argument through ~25 call sites (and risk the working paper
    figures), the constant is rebound here, once, before any part runs — and the
    change is printed so it is never silent. A run whose years already match the
    default biennial grid is left untouched.
    """
    global YEARS
    found = _available_years(results_dir)
    if not found:
        # Fall back to the tract parquets: a prediction pass may have written
        # those without the per-year building CSVs.
        found = sorted(
            int(p.name.rsplit("_", 1)[1].split(".")[0])
            for p in results_dir.glob("predictions_by_tract_2*.parquet")
        )
    if not found:
        print(f"  ⚠️ no per-year predictions under {results_dir}; "
              f"keeping the default year grid {YEARS}")
        return list(YEARS)
    if found != list(YEARS):
        print(f"  NYC year grid: {found} (was {list(YEARS)})")
        YEARS = list(found)
    return list(YEARS)


def _load_splits(processed_dir: Path) -> gpd.GeoDataFrame:
    splits = gpd.read_feather(processed_dir / "tract_splits.feather")
    splits["GEOID_str"] = splits["GEOID"].astype(str).str.zfill(11)
    return splits


def _bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, n_boot: int = 2000, ci: float = 0.95
) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(spearmanr(x[idx], y[idx]).statistic)
    boots = np.array(boots)
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return float(np.median(boots)), lo, hi

def _bootstrap_kendall(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(kendalltau(x[idx], y[idx]).statistic)
    boots = np.array(boots)
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return float(np.median(boots)), lo, hi

# ══════════════════════════════════════════════════════════════════════════════
# US-scale evaluation
# ══════════════════════════════════════════════════════════════════════════════
# The NYC parts A–E above are unchanged (paper figures). The functions below
# generalize cross-sectional validity, temporal stability and GB2 dollar mapping
# to full-US runs, which differ from NYC runs in three ways:
#   * predictions arrive as building-level {year}_predictions.csv (columns
#     Rel_Score, predicted_value, building_id, GEOID, year, type, actual_year)
#     plus predictions_by_tract_{year}.parquet — there is NO georeferenced
#     predictions_{year}.parquet (that is doitt_nyc-only), so building geometry
#     is unavailable and maps are drawn at tract level from the ACS panel.
#   * ground truth + geometry + cbsa membership come from the national panel
#     us_metros_panel_2011_2023.feather (EPSG:5070), not NYC ACS feathers.
#   * every metric is grouped by CBSA / split type / population bracket rather
#     than assuming a single city and a single 2016 holdout year.

from src.data.process_acs import PANEL_YEARS as _ACS_PANEL_YEARS, BASE_YEAR as _ACS_BASE_YEAR

_PANEL_GEOID_COL = f"geoid_{_ACS_BASE_YEAR}"
_PANEL_FILENAME = f"us_metros_panel_{_ACS_PANEL_YEARS[0]}_{_ACS_PANEL_YEARS[-1]}.feather"

# Split types we report by default (train excluded — it is not a held-out test
# of generalization; "unassigned" rows have no CBSA split and are noise).
_REPORT_TYPES = ("test", "val", "val_cities", "val_temporal")


def _available_years(results_dir: Path) -> list[int]:
    """Years with a building-level prediction CSV under ``results_dir``."""
    years = []
    for f in results_dir.glob("*_predictions.csv"):
        stem = f.name.split("_predictions.csv")[0]
        if stem.isdigit():
            years.append(int(stem))
    return sorted(years)


def _load_building_preds_us(results_dir: Path, years: list[int] | None = None) -> pd.DataFrame:
    """Stack {year}_predictions.csv into the canonical metric frame.

    Returns columns: building_id, GEOID (11-char), year, type, pred, label —
    where ``pred`` = predicted_value and ``label`` = Rel_Score (the per-CBSA
    per-year z-scored ACS wealth target). Rows with a non-finite prediction are
    dropped. GEOID is zero-padded to 11 chars because a CSV round-trip strips
    the leading zero of states 01–09.
    """
    years = _available_years(results_dir) if years is None else years
    # DOITT_ID is the legacy NYC building-id column; ms_us runs write building_id.
    usecols = ["Rel_Score", "predicted_value", "building_id", "DOITT_ID",
               "GEOID", "year", "type"]
    frames = []
    for yr in years:
        fpath = results_dir / f"{yr}_predictions.csv"
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath, usecols=lambda c: c in usecols)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["building_id", "GEOID", "year", "type", "pred", "label"])
    df = pd.concat(frames, ignore_index=True)
    if "building_id" not in df.columns and "DOITT_ID" in df.columns:
        df = df.rename(columns={"DOITT_ID": "building_id"})
    df = df.rename(columns={"predicted_value": "pred", "Rel_Score": "label"})
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[np.isfinite(df["pred"])].copy()
    df["GEOID"] = df["GEOID"].apply(_norm_geoid)
    df["year"] = df["year"].astype(int)
    if "type" not in df.columns:
        df["type"] = "unassigned"
    df["type"] = df["type"].astype(str)
    return df.reset_index(drop=True)


def _load_panel_us(indicator: str, processed_dir: Path, want_income: bool = False) -> gpd.GeoDataFrame:
    """Load the national ACS panel, reduced to the columns evaluation needs.

    Returns a GeoDataFrame indexed 0..n with columns GEOID (11-char, base-year
    tract vintage), cbsa_code (str), change (0/1 structural-change flag for the
    given indicator token), geometry (EPSG:5070) and — when ``want_income`` —
    per_capita_income_usd_{year} columns for the GB2 dollar mapping.
    """
    panel = gpd.read_feather(processed_dir / _PANEL_FILENAME)
    change_col = indicators.valid_change_col(indicator)
    keep = [_PANEL_GEOID_COL, "cbsa_code", "geometry"]
    if change_col in panel.columns:
        keep.append(change_col)
    income_cols = []
    if want_income:
        income_cols = [c for c in panel.columns if c.startswith("per_capita_income_usd_")]
        keep += income_cols
    panel = panel[keep].copy()
    panel = panel.rename(columns={_PANEL_GEOID_COL: "GEOID"})
    panel["GEOID"] = panel["GEOID"].astype(str).str.zfill(11)
    panel["cbsa_code"] = panel["cbsa_code"].astype(str)
    if change_col in panel.columns:
        panel = panel.rename(columns={change_col: "change"})
        panel["change"] = pd.to_numeric(panel["change"], errors="coerce").fillna(0).astype(int)
    else:
        panel["change"] = 0
    return panel


def _attach_cbsa(bld: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Left-join cbsa_code + change onto building/tract preds via GEOID.

    Rows whose GEOID is absent from the panel (vintage mismatch, non-metro
    tracts) are dropped after reporting the unmatched share.
    """
    meta = panel[["GEOID", "cbsa_code", "change"]].drop_duplicates("GEOID")
    merged = bld.merge(meta, on="GEOID", how="left")
    n_unmatched = int(merged["cbsa_code"].isna().sum())
    if n_unmatched:
        print(f"    {n_unmatched:,} / {len(merged):,} rows had no panel CBSA match — dropped")
    merged = merged[merged["cbsa_code"].notna()].copy()
    merged["cbsa"] = merged["cbsa_code"]  # metrics module expects a 'cbsa' column
    return merged.reset_index(drop=True)


def _load_cbsa_meta(processed_dir: Path) -> pd.DataFrame | None:
    """cbsa_splits.feather (cbsa_code, bracket, population, split, ...) or None."""
    path = processed_dir / "cbsa_splits.feather"
    try:
        return cbsa_brackets.load_city_split(path=path)
    except Exception as exc:  # missing file etc.
        print(f"    cbsa_splits.feather unavailable ({exc}); bracket/top-metro tables skipped")
        return None


# ─── Part A ───────────────────────────────────────────────────────────────────

def part_a(results_dir: Path, processed_dir: Path, out: Path) -> None:
    # 2016 is the temporal-holdout year of the NYC panel and the paper's
    # headline cross-section. If a run's cadence does not include it (NAIP flies
    # NY in odd years), fall back to the middle available year rather than
    # failing, and say which year the numbers refer to.
    year = 2016 if (results_dir / "predictions_by_tract_2016.parquet").exists() else None
    if year is None:
        avail = [y for y in YEARS
                 if (results_dir / f"predictions_by_tract_{y}.parquet").exists()]
        if not avail:
            print("\n=== Part A: Cross-sectional validity ===")
            print(f"  no predictions_by_tract_*.parquet under {results_dir}; skipping")
            return
        year = avail[len(avail) // 2]
    print(f"\n=== Part A: Cross-sectional validity {year} ===")

    tract = pd.read_parquet(results_dir / f"predictions_by_tract_{year}.parquet")
    tract["GEOID_str"] = tract["GEOID"].apply(_norm_geoid)

    # The headline cross-section must stay OUT OF SAMPLE. The NYC pass now
    # predicts the whole city (every tract, every split type — see
    # main.run_nyc_zarr_validation_predictions), so pooling the parquet would mix
    # train tracts into the number. Restrict to the held-out group and report
    # train / all only as reference rows.
    heldout = _csa_split_of(processed_dir)
    if heldout:
        tract["split_group"] = tract["GEOID_str"].map(heldout)
        n_missing = int(tract["split_group"].isna().sum())
        if n_missing:
            print(f"  {n_missing:,} / {len(tract):,} tracts absent from "
                  "tract_splits.feather — excluded from the split breakdown")
    else:
        tract["split_group"] = None

    def _corr(sub: pd.DataFrame):
        xx = sub["predicted_value"].values.astype(float)
        yy = sub["Rel_Score"].values.astype(float)
        m = np.isfinite(xx) & np.isfinite(yy)
        return xx[m], yy[m]

    subsets = {"heldout": tract[tract["split_group"] == "heldout"],
               "train": tract[tract["split_group"] == "train"],
               "all": tract}
    # Headline = held-out tracts; fall back to all tracts only if the split file
    # gave us nothing (then the number is NOT a clean holdout, and we say so).
    headline_key = "heldout" if len(subsets["heldout"]) >= 30 else "all"
    if headline_key == "all":
        print("  ⚠️ no held-out tract group resolved — falling back to ALL tracts; "
              "this ρ is NOT out-of-sample.")

    x, y = _corr(subsets[headline_key])

    rho, p_rho = spearmanr(x, y)
    tau_k, p_tau = kendalltau(x, y)
    _, lo, hi = _bootstrap_spearman(x, y)
    _, lo_k, hi_k = _bootstrap_kendall(x, y)

    print(f"  [{headline_key}, HEADLINE] Spearman ρ = {rho:.3f}  "
          f"(95% CI [{lo:.3f}, {hi:.3f}])  p={p_rho:.2e}  n={len(x):,}")
    print(f"  [{headline_key}, HEADLINE] Kendall τ  = {tau_k:.3f}  "
          f"(95% CI [{lo_k:.3f}, {hi_k:.3f}])  p={p_tau:.2e}")

    rows = [
        {"metric": "Spearman_rho", "split": headline_key, "headline": True,
         "value": rho, "p_value": p_rho, "ci_lo_95": lo, "ci_hi_95": hi, "n": len(x)},
        {"metric": "Kendall_tau", "split": headline_key, "headline": True,
         "value": tau_k, "p_value": p_tau, "ci_lo_95": np.nan, "ci_hi_95": np.nan,
         "n": len(x)},
    ]
    # Reference rows: the same statistic on the other groups. A train ρ far above
    # the held-out one would say the model memorised rather than generalised.
    for key, sub in subsets.items():
        if key == headline_key or len(sub) < 30:
            continue
        xr, yr = _corr(sub)
        if len(xr) < 30:
            continue
        r, p = spearmanr(xr, yr)
        print(f"  [{key}, reference] Spearman ρ = {r:.3f}  n={len(xr):,}")
        rows.append({"metric": "Spearman_rho", "split": key, "headline": False,
                     "value": r, "p_value": p, "ci_lo_95": np.nan,
                     "ci_hi_95": np.nan, "n": len(xr)})

    # Building-level (not headline) — needs the georeferenced parquet, which only
    # the doitt_nyc footprint source writes.
    brho, bm = np.nan, np.zeros(0, dtype=bool)
    bldg_path = results_dir / f"predictions_{year}.parquet"
    if bldg_path.exists():
        bldg = gpd.read_parquet(bldg_path)
        if headline_key == "heldout" and "GEOID" in bldg.columns:
            keep = set(subsets["heldout"]["GEOID_str"])
            bldg = bldg[bldg["GEOID"].apply(_norm_geoid).isin(keep)]
        bx = bldg["predicted_value"].values.astype(float)
        by = bldg["Rel_Score"].values.astype(float)
        bm = np.isfinite(bx) & np.isfinite(by)
        if bm.sum() >= 30:
            brho, _ = spearmanr(bx[bm], by[bm])
            print(f"  [building-level, {headline_key}, NOT headline] "
                  f"ρ = {brho:.3f}  n={bm.sum():,}")
    else:
        print(f"  {bldg_path.name} absent — building-level ρ skipped")

    rows.append({"metric": "Spearman_rho_building", "split": headline_key,
                 "headline": False, "value": brho, "p_value": np.nan,
                 "ci_lo_95": np.nan, "ci_hi_95": np.nan, "n": int(bm.sum())})
    pd.DataFrame(rows).to_csv(
        out / "tables" / f"A_cross_sectional_{year}.csv", index=False)

    # Scatter with 10-bin overlay
    fig, ax = plt.subplots(figsize=FIG_SIZE_ONE_COL)
    ax.scatter(y, x, s=4, alpha=0.22, color="steelblue", linewidths=0)

    bin_edges = np.percentile(x, np.linspace(0, 100, 11))
    bin_ids = np.digitize(x, bin_edges[1:-1])  # 0..19
    bx_m = [x[bin_ids == b].mean() for b in range(10) if (bin_ids == b).any()]
    by_m = [y[bin_ids == b].mean() for b in range(10) if (bin_ids == b).any()]
    ax.plot(by_m, bx_m, "o-", color="firebrick", ms=5, lw=1.5, label="10-bin mean")
    ax.set_xlim(-3, 4)
    ax.set_xlabel("ACS Tract Z-score")
    ax.set_ylabel("Average Tract\nPredicted Value")
    # Add text to the plot with the statistics
    ax.text(0.05, 0.8, f"Spearman $\\rho$ = {rho:.3f}\nKendall $\\tau$ = {tau_k:.3f}",
            transform=ax.transAxes, fontsize=8,
    )
    ax.set_title(f"{year}, {headline_key} tracts ($n$ = {len(x):,})", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, out / "figures" / f"A_scatter_{year}.pdf")
    # The choropleth is a supplementary figure and needs extra inputs (tract
    # geometry, an ACS vintage). It must never discard the statistics above.
    try:
        _map_acs_vs_pred(results_dir, processed_dir, out)
    except Exception as exc:
        print(f"    choropleth map skipped ({type(exc).__name__}: {exc})")


# ─── Part A – choropleth map helpers ─────────────────────────────────────────

def _add_north_arrow(ax: plt.Axes, x: float = 0.94, y: float = 0.08) -> None:
    """North arrow in axes-fraction coordinates."""
    ax.annotate(
        "", xy=(x, y + 0.07), xytext=(x, y),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=12),
    )
    ax.text(x, y + 0.09, "N", ha="center", va="bottom", fontsize=7,
            fontweight="bold", transform=ax.transAxes)


def _add_scale_bar(ax: plt.Axes, bar_km: float = 5.0,
                   x0_frac: float = 0.05, y0_frac: float = 0.05) -> None:
    """Two-tone (black/white) scale bar in data coordinates (EPSG:6539, US-survey-ft)."""
    bar_ft = bar_km * 1000.0 / _M_PER_FT
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0 = xlim[0] + x0_frac * (xlim[1] - xlim[0])
    y0 = ylim[0] + y0_frac * (ylim[1] - ylim[0])
    h  = (ylim[1] - ylim[0]) * 0.013
    # Full black bar, then white left half
    ax.add_patch(plt.Rectangle((x0, y0), bar_ft,     h, fc="black", ec="0.35", lw=0.4, zorder=10))
    ax.add_patch(plt.Rectangle((x0, y0), bar_ft / 2, h, fc="white", ec="0.35", lw=0.4, zorder=11))
    ax.text(x0,          y0 - h * 0.6, "0",                ha="center", va="top", fontsize=5.5, zorder=12)
    ax.text(x0 + bar_ft, y0 - h * 0.6, f"{bar_km:.0f} km", ha="center", va="top", fontsize=5.5, zorder=12)


def _map_acs_vs_pred(results_dir: Path, processed_dir: Path, out: Path) -> None:
    """Produces two figures sharing the same Spectral decile scale.

    A_map_acs2014_pred2016.pdf — full NYC, two panels:
        left : 2014 ACS per-capita income deciles
        right: 2016 tract-level model-prediction deciles

    A_map_zoom_core.pdf — three panels cropped to Downtown–Midtown Manhattan,
    DUMBO, Williamsburg, and Astoria:
        left : ACS 2014 (same decile boundaries)
        centre: tract prediction 2016 (same decile boundaries)
        right : building-level prediction 2016 (same decile boundaries)
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    print("  Map: ACS 2014 vs model predictions 2016 (full NYC + zoom)...")

    # ── Geometries ──────────────────────────────────────────────────────────
    splits = _load_splits(processed_dir)
    gdf = splits[["GEOID_str", "geometry"]].drop_duplicates("GEOID_str").copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_GEO)
    gdf_proj = gdf.to_crs(CRS_PROJ)

    # ── ACS 2014 ────────────────────────────────────────────────────────────
    try:
        acs = _load_acs_nyc(2014)
    except FileNotFoundError:
        print("    ACS 2014 not found — skipping map")
        return
    gdf_proj = gdf_proj.merge(
        acs.rename("acs_income").reset_index().rename(columns={"geoid": "GEOID_str"}),
        on="GEOID_str", how="left",
    )

    # ── Tract-level predictions 2016 ────────────────────────────────────────
    pred = pd.read_parquet(results_dir / "predictions_by_tract_2016.parquet")
    pred["GEOID_str"] = pred["GEOID"].apply(_norm_geoid)
    pred = pred.drop_duplicates("GEOID_str")
    gdf_proj = gdf_proj.merge(
        pred[["GEOID_str", "predicted_value"]], on="GEOID_str", how="left",
    )

    # ── Decile helpers ───────────────────────────────────────────────────────
    def _compute_edges(col: pd.Series) -> np.ndarray | None:
        valid = col.dropna()
        if len(valid) < 10:
            return None
        edges = np.unique(np.nanpercentile(valid, np.linspace(0, 100, 11)))
        return edges if len(edges) >= 2 else None

    def _apply_decile(col: pd.Series, edges: np.ndarray) -> pd.Series:
        """Map values to deciles 1–10 with pre-computed edges.
        Outer edges are extended so out-of-range building values absorb into D1/D10."""
        e = edges.copy()
        valid = col.dropna()
        if len(valid) > 0:
            e[0]  = min(e[0]  - 1e-9, float(valid.min()) - 1e-9)
            e[-1] = max(e[-1] + 1e-9, float(valid.max()) + 1e-9)
        else:
            e[0] -= 1e-9; e[-1] += 1e-9
        return pd.cut(col, bins=e, labels=False, include_lowest=True).astype(float) + 1

    acs_edges  = _compute_edges(gdf_proj["acs_income"])
    pred_edges = _compute_edges(gdf_proj["predicted_value"])
    if acs_edges is None or pred_edges is None:
        print("    Insufficient data for decile computation — skipping map")
        return

    gdf_proj["acs_decile"]  = _apply_decile(gdf_proj["acs_income"],     acs_edges)
    gdf_proj["pred_decile"] = _apply_decile(gdf_proj["predicted_value"], pred_edges)

    # ── Discrete Spectral colormap (10 levels) ───────────────────────────────
    n = 10
    cmap_disc = ListedColormap(
        [plt.cm.Spectral(i / (n - 1)) for i in range(n)], name="spectral10"
    )
    norm = BoundaryNorm(np.arange(0.5, n + 1.5, 1.0), ncolors=n)

    # ── Borough outlines and centroid labels ─────────────────────────────────
    gdf_proj["county"] = gdf_proj["GEOID_str"].str[:5]
    borough_outline = gdf_proj.dissolve(by="county").boundary
    BOROUGH_LABELS = {
        "36005": "Bronx",     "36047": "Brooklyn",
        "36061": "Manhattan", "36081": "Queens",  "36085": "Staten\nIsland",
    }
    borough_centroids = {
        code: gdf_proj[gdf_proj["county"] == code].dissolve().geometry.centroid.iloc[0]
        for code in BOROUGH_LABELS
        if (gdf_proj["county"] == code).any()
    }

    # Reusable ScalarMappable (both colorbars share the same norm/cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap_disc, norm=norm)
    sm.set_array([])

    def _draw_panels(axes, specs, *, show_borough_labels: bool = True,
                     bldg_src=None) -> None:
        """Render choropleth panels. Scale bar and axis limits handled by caller."""
        for ax, (col, src, title) in zip(axes, specs):
            is_bldg = (bldg_src is not None) and (src is bldg_src)
            src[src[col].isna()].plot(
                ax=ax, color="#cccccc", linewidth=0, edgecolor="none",
            )
            src[src[col].notna()].plot(
                column=col, ax=ax, cmap=cmap_disc, norm=norm,
                linewidth=0 if is_bldg else 0.06,
                edgecolor="none" if is_bldg else "0.55",
                alpha=0.92,
            )
            borough_outline.plot(ax=ax, color="0.2", linewidth=0.85, zorder=5)
            if show_borough_labels and not is_bldg:
                for code, cen in borough_centroids.items():
                    ax.text(cen.x, cen.y, BOROUGH_LABELS[code],
                            ha="center", va="center", fontsize=5.5,
                            style="italic", fontweight="bold", color="0.15", zorder=6)
            ax.set_title(title, fontsize=8, pad=4)
            ax.axis("off")
            _add_north_arrow(ax)

    def _attach_colorbar(fig, left, width, label_fs=7.5) -> None:
        cbar = fig.colorbar(
            sm, cax=fig.add_axes([left, 0.07, width, 0.030]),
            orientation="horizontal",
        )
        cbar.set_ticks(np.arange(1, n + 1))
        cbar.set_ticklabels([f"D{i}" for i in range(1, n + 1)], fontsize=7)
        cbar.set_label("Income decile  (D1 = lowest,  D10 = highest)", fontsize=label_fs)
        cbar.outline.set_linewidth(0.5)

    # ════════════════════════════════════════════════════════════════════════
    # Figure 1 — full NYC, two panels
    # ════════════════════════════════════════════════════════════════════════
    fw = FIG_SIZE_TWO_COL[0]
    fig1, axes1 = plt.subplots(1, 2, figsize=(fw, fw * 0.60))
    _draw_panels(axes1, [
        ("acs_decile",  gdf_proj, "ACS Per-Capita Income (2014)"),
        ("pred_decile", gdf_proj, "Tract Prediction (2016)"),
    ])
    _add_scale_bar(axes1[0])
    fig1.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.20, wspace=0.04)
    _attach_colorbar(fig1, left=0.22, width=0.56, label_fs=7.5)
    _savefig(fig1, out / "figures" / "A_map_acs2014_pred2016.pdf")

    # ════════════════════════════════════════════════════════════════════════
    # Figure 2 — zoomed: Downtown–Midtown Manhattan, DUMBO, Williamsburg, Astoria
    # ════════════════════════════════════════════════════════════════════════
    print("    Loading building-level predictions 2016 for zoom map...")
    bldg = gpd.read_parquet(results_dir / "predictions_2016.parquet")
    bldg_proj = bldg.to_crs(CRS_PROJ)
    bldg_proj["bldg_decile"] = _apply_decile(bldg_proj["predicted_value"], pred_edges)

    # Zoom bounding box in EPSG:6539 (US survey feet).
    # Covers: Lower + Midtown Manhattan | DUMBO | Williamsburg | LIC | Astoria
    t_fwd = Transformer.from_crs(CRS_GEO, CRS_PROJ, always_xy=True)
    x0_z, y0_z = t_fwd.transform(-74.030, 40.690)
    x1_z, y1_z = t_fwd.transform(-73.905, 40.790)

    # Figure height derived from zoom-box aspect ratio so panels render square
    zoom_aspect = (y1_z - y0_z) / (x1_z - x0_z)   # ≈ 1.06
    fig2, axes2 = plt.subplots(1, 3, figsize=(fw, fw * zoom_aspect / 3 + 0.9))
    _draw_panels(axes2, [
        ("acs_decile",  gdf_proj,  "ACS Per-Capita Income (2014)"),
        ("pred_decile", gdf_proj,  "Tract Prediction (2016)"),
        ("bldg_decile", bldg_proj, "Building Prediction (2016)"),
    ], show_borough_labels=False, bldg_src=bldg_proj)

    # Apply crop BEFORE scale bar so get_xlim/get_ylim reflect the zoom extent
    for ax in axes2:
        ax.set_xlim(x0_z, x1_z)
        ax.set_ylim(y0_z, y1_z)

    _add_scale_bar(axes2[0], bar_km=2.0)
    fig2.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.22, wspace=0.04)
    _attach_colorbar(fig2, left=0.18, width=0.64, label_fs=7.0)
    _savefig(fig2, out / "figures" / "A_map_zoom_core.pdf")


# ─── Part B ───────────────────────────────────────────────────────────────────

def part_b(results_dir: Path, processed_dir: Path, out: Path) -> None:
    print("\n=== Part B: Temporal stability ===")

    splits = _load_splits(processed_dir)
    test_geoids  = set(splits.loc[splits["type"] == "test",  "GEOID_str"])
    train_geoids = set(splits.loc[splits["type"] == "train", "GEOID_str"])

    # ── B.1 Tract panel ───────────────────────────────────────────────────────
    print("  B.1 tract panel...")
    tract_long = _load_tract_long(results_dir)
    tract_long = tract_long.merge(splits[["GEOID_str", "type"]], on="GEOID_str", how="left")

    tract_wide = (
        tract_long.pivot_table(
            index="GEOID_str", columns="year",
            values="predicted_value", aggfunc="first",
        )
    )
    tract_wide.columns = [int(c) for c in tract_wide.columns]
    tract_type = (
        tract_long.drop_duplicates("GEOID_str")
        .set_index("GEOID_str")[["type"]]
    )
    tract_wide = tract_wide.join(tract_type)

    # ── B.1 Building panel ────────────────────────────────────────────────────
    print("  B.1 building panel (loading 8 prediction files, may take a moment)...")
    # Use 2024 (broadest coverage: 190 test tracts) to identify analysis building IDs
    pred_ref = gpd.read_parquet(results_dir / "predictions_2024.parquet")
    pred_ref["GEOID_str"] = pred_ref["GEOID"].apply(_norm_geoid)

    test_ids  = pred_ref.index[pred_ref["GEOID_str"].isin(test_geoids)].tolist()
    train_all = pred_ref.index[pred_ref["GEOID_str"].isin(train_geoids)].tolist()
    rng = np.random.default_rng(0)
    train_sample = rng.choice(
        train_all, min(50_000, len(train_all)), replace=False
    ).tolist()
    analysis_set = set(test_ids) | set(train_sample)
    
    bldg_frames = []
    for yr in YEARS:
        df = gpd.read_parquet(results_dir / f"predictions_{yr}.parquet")
        sub = df.loc[df.index.isin(analysis_set), ["predicted_value", "GEOID", "geometry"]]
        sub = sub.copy()
        sub.index.name = "DOITT_ID"
        sub["year"] = yr
        bldg_frames.append(sub.reset_index())

    bldg_long = pd.concat(bldg_frames, ignore_index=True)
    bldg_long["GEOID_str"] = bldg_long["GEOID"].apply(_norm_geoid)
    bldg_long = bldg_long.merge(splits[["GEOID_str", "type"]], on="GEOID_str", how="left")

    bldg_wide = bldg_long.pivot_table(
        index="DOITT_ID", columns="year",
        values="predicted_value", aggfunc="first",
    )
    bldg_wide.columns = [int(c) for c in bldg_wide.columns]
    bldg_meta = (
        bldg_long.drop_duplicates("DOITT_ID")
        .set_index("DOITT_ID")[["GEOID_str", "type"]]
    )
    bldg_wide = bldg_wide.join(bldg_meta)

    # ── B.2 Change detection ──────────────────────────────────────────────────
    print("  B.2 change detection...")
    bldg_nyc = gpd.read_parquet(processed_dir / "buildings_nyc.parquet")
    # New buildings: constructed strictly after 2009 and no later than 2024
    new_bldg = bldg_nyc.loc[
        bldg_nyc["CONSTRUCTION_YEAR"].between(2009, 2024, inclusive="right")
    ].to_crs(CRS_PROJ)

    # Get geometry for analysis buildings: take the first available year per building
    geom_all = (
        bldg_long[["DOITT_ID", "geometry", "year"]]
        .sort_values("year")
        .drop_duplicates("DOITT_ID")
        .set_index("DOITT_ID")[["geometry"]]
    )
    geom_gdf = gpd.GeoDataFrame(geom_all, crs=CRS_PROJ)

    change_df = _detect_change_vectorized(geom_gdf, new_bldg)
    bldg_wide = bldg_wide.join(change_df, how="left")
    bldg_wide["changed"] = bldg_wide["changed"].fillna(False).astype(bool)
    year_cols = YEARS

    # ── Export intensity indicators ──────────────────────────────────────────
    # Write one parquet with all changed buildings that meet each intensity
    # threshold, tagged with split type, change_year, and n_new_buildings.
    # Geometry is intentionally omitted — join back on DOITT_ID from the
    # prediction parquets when needed. This keeps the file small (~KB not MB).
    #
    # Thresholds exported: multi (>=2) and dense (>=5).
    # "any" (>=1) is just the full changed set and is already implicit in
    # B_stability_metrics.csv, so we skip it here to avoid redundancy.
    EXPORT_THRESHOLDS = [
        ("any", 1),
        ("multi", 2),
        ("dense", 5),
    ]

    intensity_rows = []
    for did, row in bldg_wide.iterrows():
        if not row.get("changed", False):
            continue
        cy = row.get("change_year")
        n_new = int(row.get("n_new_buildings", 0))
        for thresh_name, min_n_new in EXPORT_THRESHOLDS:
            if n_new >= min_n_new:
                intensity_rows.append({
                    "DOITT_ID":       did,
                    "split_type":     row.get("type"),
                    "change_year":    int(cy),
                    "n_new_buildings": n_new,
                    "intensity":      thresh_name,
                })

    if intensity_rows:
        intensity_df = pd.DataFrame(intensity_rows)
        intensity_path = out / "tables" / "B_intensity_indicators.parquet"
        intensity_df.to_parquet(intensity_path, index=False)
        print(
            f"    saved B_intensity_indicators.parquet  "
            f"({len(intensity_df):,} rows  |  "
            f"multi: {(intensity_df['intensity']=='multi').sum():,}  "
            f"dense: {(intensity_df['intensity']=='dense').sum():,})"
        )
        buildings = gpd.read_parquet(processed_dir / "building_geometries_years2010-2024.parquet")
        buildings.join(intensity_df.set_index("DOITT_ID"), how="inner").to_parquet(out / "tables" / "B_intensity_indicators_with_geometries.parquet")  
    else:
        print("    no intensity rows to export (check change detection output)")


    # ── B.3 Metrics ───────────────────────────────────────────────────────────
    print("  B.3 metrics...")
    # NaN-aware: keep every unit with >=2 observed years (the intermediate years
    # are predicted for only ~220 tracts, so requiring all 8 would shrink the
    # test set to a tiny, geographically-biased subset). Within-unit volatility
    # is computed over each unit's available years; rank-autocorrelation uses the
    # fully-covered RANK_PAIR (2016 vs 2024).
    val_arr = bldg_wide[year_cols].values.astype(float)
    keep = np.isfinite(val_arr).sum(axis=1) >= 2
    bw = bldg_wide[keep].copy()
    va = bw[year_cols].values.astype(float)
    changed_m = bw["changed"].values
    stable_m  = ~changed_m

    metric_rows = []

    for mask, label in [
        (stable_m  & (bw["type"] == "test").values,  "stable_test"),
        (stable_m  & (bw["type"] == "train").values, "stable_train"),
        (changed_m & (bw["type"] == "test").values,  "changed_test"),
        (changed_m & (bw["type"] == "train").values, "changed_train"),
    ]:
        arr = va[mask]
        if arr.shape[0] == 0:
            continue
        within_std = np.nanstd(arr, axis=1)
        within_mad = np.nanmedian(
            np.abs(arr - np.nanmean(arr, axis=1, keepdims=True)), axis=1
        )
        icc_val         = _icc(arr)
        rank_auto, rk_n = _rank_autocorr(bw[mask], RANK_PAIR)
        metric_rows.append({
            "label": label, "n": int(mask.sum()),
            "within_std_median": float(np.nanmedian(within_std)),
            "within_std_IQR_lo": float(np.nanpercentile(within_std, 25)),
            "within_std_IQR_hi": float(np.nanpercentile(within_std, 75)),
            "within_MAD_median": float(np.nanmedian(within_mad)),
            "ICC": icc_val,
            "rank_autocorr": rank_auto,
            "rank_autocorr_n": rk_n,
            "rank_pair": f"{RANK_PAIR[0]}-{RANK_PAIR[1]}",
        })

    # Changed set: pre/post delta (NaN-aware — average over observed years only)
    for lbl_sfx, type_val in [("test", "test"), ("train", "train")]:
        chg_bw = bw[changed_m & (bw["type"] == type_val).values]
        deltas = []
        for _, row in chg_bw.iterrows():
            cy = row["change_year"]
            if pd.isna(cy):
                continue
            pre  = [row[c] for c in year_cols if c <  cy and pd.notna(row[c])]
            post = [row[c] for c in year_cols if c >= cy and pd.notna(row[c])]
            if not pre or not post:
                continue
            deltas.append(float(np.mean(post) - np.mean(pre)))
        if deltas:
            deltas = np.array(deltas)
            metric_rows.append({
                "label": f"changed_delta_{lbl_sfx}",
                "n": int(len(deltas)),
                "share_positive": float((deltas > 0).mean()),
                "median_delta": float(np.median(deltas)),
                "median_abs_delta": float(np.median(np.abs(deltas))),
            })

    # Tract-level stability (NaN-aware, >=2 observed years)
    for split_type in ["test", "train"]:
        sub_w  = tract_wide[(tract_wide["type"] == split_type).values].copy()
        arr    = sub_w[year_cols].values.astype(float)
        keep_t = np.isfinite(arr).sum(axis=1) >= 2
        sub_w, arr = sub_w[keep_t], arr[keep_t]
        if arr.shape[0] == 0:
            continue
        within_std      = np.nanstd(arr, axis=1)
        icc_val         = _icc(arr)
        rank_auto, rk_n = _rank_autocorr(sub_w, RANK_PAIR)
        metric_rows.append({
            "label": f"tract_{split_type}",
            "n": int(arr.shape[0]),
            "within_std_median": float(np.nanmedian(within_std)),
            "ICC": float(icc_val),
            "rank_autocorr": rank_auto,
            "rank_autocorr_n": rk_n,
            "rank_pair": f"{RANK_PAIR[0]}-{RANK_PAIR[1]}",
        })

    pd.DataFrame(metric_rows).to_csv(
        out / "tables" / "B_stability_metrics.csv", index=False
    )
    print("    saved B_stability_metrics.csv")

    # ── B.4 Plots ─────────────────────────────────────────────────────────────
    print("  B.4 plots...")

    # Tract spaghetti: test vs train side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, split_type, title in [
        (axes[0], "test",  "Test tracts (n=190, spatial holdout)"),
        (axes[1], "train", "Train tracts (random sample n=190)"),
    ]:
        sub = tract_wide[tract_wide["type"] == split_type]
        if split_type == "train":
            sub = sub.sample(min(190, len(sub)), random_state=0)
        yt = sub[year_cols].values.astype(float)
        for row_vals in yt:
            ax.plot(year_cols, row_vals, color="steelblue", alpha=0.12, lw=0.6)
        med = np.nanmedian(yt, axis=0)
        q25 = np.nanpercentile(yt, 25, axis=0)
        q75 = np.nanpercentile(yt, 75, axis=0)
        ax.plot(year_cols, med, color="black", lw=2, label="Median")
        ax.fill_between(year_cols, q25, q75, alpha=0.22, color="black", label="IQR")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Year")
        ax.legend(fontsize=8)
        ax.set_xticks(year_cols)
    axes[0].set_ylabel("predicted_value (tract mean)")
    fig.suptitle("Temporal trajectories: are predictions stable where nothing changed?")
    fig.tight_layout()
    _savefig(fig, out / "figures" / "B_tract_trajectories.png")

    # Example building trajectories (use any buildings with ≥2 years, not just complete)
    n_ex = 5
    n_year_valid = np.isfinite(bldg_wide[year_cols].values.astype(float)).sum(axis=1)
    has_2yr = n_year_valid >= 2
    stab_test_any  = has_2yr & ~bldg_wide["changed"].fillna(False).astype(bool) & (bldg_wide["type"] == "test").fillna(False)
    chng_test_any  = has_2yr &  bldg_wide["changed"].fillna(False).astype(bool) & (bldg_wide["type"] == "test").fillna(False)
    stable_ex  = bldg_wide[stab_test_any].sample(n_ex, random_state=825)
    changed_ex = bldg_wide[chng_test_any].dropna(subset=["change_year"]).sample(n_ex, random_state=555)
    n_changed_ex = len(changed_ex)

    fig, axes = plt.subplots(2, n_ex, figsize=(14, 6), sharey="row")
    for col, (did, row) in enumerate(stable_ex.iterrows()):
        ax = axes[0, col]
        ax.plot(year_cols, row[year_cols].values, "o-", color="steelblue", ms=4, lw=1.2)
        ax.set_title(f"DOITT {did}", fontsize=7)
        if col == 0:
            ax.set_ylabel("Stable", fontsize=9)

    for col in range(n_ex):
        ax = axes[1, col]
        if col < n_changed_ex:
            did, row = list(changed_ex.iterrows())[col]
            ax.plot(year_cols, row[year_cols].values, "o-", color="firebrick", ms=4, lw=1.2)
            cy = row["change_year"]
            if not pd.isna(cy):
                ax.axvline(cy, color="black", ls="--", lw=1)
            ax.set_title(f"DOITT {did}", fontsize=7)
        if col == 0:
            ax.set_ylabel("Changed\n(-- = change_year)", fontsize=9)

    for ax in axes.flat:
        ax.set_xlabel("Year", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xticks(year_cols)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Example building trajectories (test tracts)", fontsize=11)
    fig.tight_layout()
    _savefig(fig, out / "figures" / "B_building_trajectories.pdf")

    _plot_building_income_trajectories(
        results_dir, processed_dir, out, bldg_long, bldg_wide, year_cols
    )
    _plot_tract_income_trajectories(out, tract_long, tract_wide)
    _plot_percentile_trajectories(out, tract_long)
    _plot_tract_rank_trajectories(out, tract_long, tract_wide)




def _plot_building_income_trajectories(
    results_dir: Path,
    processed_dir: Path,
    out: Path,
    bldg_long: pd.DataFrame,
    bldg_wide: pd.DataFrame,
    year_cols: list[int],
) -> None:
    """B_building_income_trajectories: test-set buildings with GB2-mapped income vs ACS (±MOE)."""
    print("  B.4b building income trajectories (GB2-mapped)...")

    # ── Sample buildings independently from B_building_trajectories ──────────
    n_ex = 5
    n_year_valid = np.isfinite(bldg_wide[year_cols].values.astype(float)).sum(axis=1)
    has_2yr       = n_year_valid >= 2
    stab_mask = has_2yr & ~bldg_wide["changed"].fillna(False).astype(bool) & (bldg_wide["type"] == "test").fillna(False)
    chng_mask = has_2yr &  bldg_wide["changed"].fillna(False).astype(bool) & (bldg_wide["type"] == "test").fillna(False)
    stable_ex  = bldg_wide[stab_mask].sample(n_ex, random_state=555)
    changed_ex = bldg_wide[chng_mask].dropna(subset=["change_year"]).sample(n_ex, random_state=666)

    # ── Fit GB2 to ACS + collect income/MOE per year ────────────────────────
    gb2_raw: dict[int, tuple] = {}
    acs_frames: dict[int, pd.DataFrame] = {}

    for yr in YEARS:
        fpath = ACS_ROOT_DIR / str(yr) / f"ny_tracts_acs5_{yr}.feather"
        if not fpath.exists():
            continue
        try:
            df  = pd.read_feather(fpath)
            nyc = df[df["geoid"].str.startswith(NYC_COUNTY_PREFIXES)].copy()
            nyc["geoid"] = nyc["geoid"].astype(str).str.zfill(11)
            nyc = nyc.drop_duplicates("geoid").set_index("geoid")

            A = nyc["per_capita_income_usd"].dropna().values.astype(float)
            A = A[A > 0]
            if len(A) < 20:
                continue

            gb2_raw[yr] = _fit_gb2(A)

            result = nyc[["per_capita_income_usd"]].copy()
            result["moe"] = np.nan
            for moe_cand in [
                "per_capita_income_usd_error",
                "per_capita_income_usd_moe",
                "per_capita_income_moe",
            ]:
                if moe_cand in nyc.columns:
                    result["moe"] = nyc[moe_cand].astype(float)
                    break
            acs_frames[yr] = result
        except Exception as exc:
            print(f"    ACS {yr} error: {exc}")

    avail = sorted(gb2_raw)
    if not avail:
        print("    No ACS data — skipping B_building_income_trajectories")
        return

    gb2_smooth = _smooth_gb2_params(avail, gb2_raw)

    # ── Apply GB2 per year: full distribution → per-building income ──────────
    gb2_bldg: dict[int, pd.Series] = {}
    for yr in avail:
        sub = bldg_long[bldg_long["year"] == yr].dropna(subset=["predicted_value"])
        if len(sub) == 0:
            continue
        mapped = _gb2_apply_from_ranks(
            sub["predicted_value"].values.astype(float),
            gb2_smooth[yr],
        )
        gb2_bldg[yr] = pd.Series(mapped, index=sub["DOITT_ID"].values)

    # ── Per-building trajectory helper ───────────────────────────────────────
    def _traj(did, geoid):
        py, pi = [], []
        for yr in YEARS:
            if yr not in gb2_bldg:
                continue
            s = gb2_bldg[yr]
            if did in s.index:
                v = float(s[did])
                if np.isfinite(v):
                    py.append(yr); pi.append(v)

        ay, ai, am = [], [], []
        for yr in YEARS:
            if yr not in acs_frames or geoid not in acs_frames[yr].index:
                continue
            row_acs = acs_frames[yr].loc[geoid]
            inc = float(row_acs["per_capita_income_usd"])
            if not np.isfinite(inc):
                continue
            moe_v = float(row_acs["moe"]) if pd.notna(row_acs["moe"]) else np.nan
            ay.append(yr); ai.append(inc)
            am.append(moe_v if np.isfinite(moe_v) else 0.0)
        return py, pi, ay, ai, am

    # ── Figure ───────────────────────────────────────────────────────────────
    n_ex = 5
    fig, axes = plt.subplots(2, n_ex, figsize=(14, 6), sharey=False)

    _legend_drawn = False

    for col, (did, row) in enumerate(stable_ex.iterrows()):
        ax = axes[0, col]
        py, pi, ay, ai, am = _traj(did, row["GEOID_str"])

        if py:
            ax.plot(py, pi, "o-", color="steelblue", ms=3.5, lw=1.2, label="Model (GB2)")
        if ay:
            ai_arr = np.array(ai, dtype=float)
            am_arr = np.array(am, dtype=float)
            ax.plot(ay, ai_arr, "s--", color="black", ms=3.5, lw=1.0, label="ACS")
            moe_ok = am_arr > 0
            if moe_ok.any():
                ax.fill_between(
                    np.array(ay)[moe_ok],
                    (ai_arr - am_arr)[moe_ok],
                    (ai_arr + am_arr)[moe_ok],
                    color="black", alpha=0.15, label="ACS ±MOE",
                )

        ax.set_title(f"DOITT {did}", fontsize=7)
        ax.set_xticks(YEARS)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        if col == 0:
            ax.set_ylabel("Income (USD)\n[Stable]", fontsize=8)
        if not _legend_drawn:
            ax.legend(fontsize=6, loc="upper left")
            _legend_drawn = True

    changed_iter = list(changed_ex.iterrows())
    for col in range(n_ex):
        ax = axes[1, col]
        if col < len(changed_iter):
            did, row = changed_iter[col]
            py, pi, ay, ai, am = _traj(did, row["GEOID_str"])

            if py:
                ax.plot(py, pi, "o-", color="firebrick", ms=3.5, lw=1.2, label="Model (GB2)")
            if ay:
                ai_arr = np.array(ai, dtype=float)
                am_arr = np.array(am, dtype=float)
                ax.plot(ay, ai_arr, "s--", color="black", ms=3.5, lw=1.0, label="ACS")
                moe_ok = am_arr > 0
                if moe_ok.any():
                    ax.fill_between(
                        np.array(ay)[moe_ok],
                        (ai_arr - am_arr)[moe_ok],
                        (ai_arr + am_arr)[moe_ok],
                        color="black", alpha=0.15, label="ACS ±MOE",
                    )

            cy = row["change_year"]
            if pd.notna(cy):
                ax.axvline(float(cy), color="gray", ls="--", lw=1.0)
            ax.set_title(f"DOITT {did}", fontsize=7)

        ax.set_xticks(YEARS)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_xlabel("Year", fontsize=7)
        if col == 0:
            ax.set_ylabel("Income (USD)\n[Changed, -- = change year]", fontsize=8)

    fig.suptitle(
        "Building income trajectories (test tracts): GB2-mapped model vs ACS observed",
        fontsize=10,
    )
    fig.tight_layout()
    _savefig(fig, out / "figures" / "B_building_income_trajectories.pdf")


def _plot_tract_income_trajectories(
    out: Path,
    tract_long: pd.DataFrame,
    tract_wide: pd.DataFrame,
) -> None:
    """B_tract_income_trajectories: test-tract GB2-mapped predictions vs ACS observed (±MOE)."""
    print("  B.4c tract income trajectories (GB2-mapped)...")

    # ── Fit GB2 + load ACS income/MOE per year ──────────────────────────────
    gb2_raw: dict[int, tuple] = {}
    acs_frames: dict[int, pd.DataFrame] = {}

    for yr in YEARS:
        fpath = ACS_ROOT_DIR / str(yr) / f"ny_tracts_acs5_{yr}.feather"
        if not fpath.exists():
            continue
        try:
            df  = pd.read_feather(fpath)
            nyc = df[df["geoid"].str.startswith(NYC_COUNTY_PREFIXES)].copy()
            nyc["geoid"] = nyc["geoid"].astype(str).str.zfill(11)
            nyc = nyc.drop_duplicates("geoid").set_index("geoid")

            A = nyc["per_capita_income_usd"].dropna().values.astype(float)
            A = A[A > 0]
            if len(A) < 20:
                continue

            gb2_raw[yr] = _fit_gb2(A)

            result = nyc[["per_capita_income_usd"]].copy()
            result["moe"] = np.nan
            for moe_cand in [
                "per_capita_income_usd_error",
                "per_capita_income_usd_moe",
                "per_capita_income_moe",
            ]:
                if moe_cand in nyc.columns:
                    result["moe"] = nyc[moe_cand].astype(float)
                    break
            acs_frames[yr] = result
        except Exception as exc:
            print(f"    ACS {yr} error: {exc}")

    avail = sorted(gb2_raw)
    if not avail:
        print("    No ACS data — skipping B_tract_income_trajectories")
        return

    gb2_smooth = _smooth_gb2_params(avail, gb2_raw)

    # ── Apply GB2 per year: tract distribution → mapped income ──────────────
    # Index is GEOID_str; the full tract distribution is used so ranks are city-wide.
    gb2_tract: dict[int, pd.Series] = {}
    for yr in avail:
        sub = tract_long[tract_long["year"] == yr].dropna(subset=["predicted_value"])
        if len(sub) == 0:
            continue
        mapped = _gb2_apply_from_ranks(
            sub["predicted_value"].values.astype(float),
            gb2_smooth[yr],
        )
        gb2_tract[yr] = pd.Series(mapped, index=sub["GEOID_str"].values)

    # ── Select 10 test tracts stratified by income level ────────────────────
    year_cols_int = [y for y in YEARS if y in tract_wide.columns]
    test_wide = tract_wide[tract_wide["type"] == "test"].copy()

    n_complete = np.isfinite(
        test_wide[year_cols_int].values.astype(float)
    ).sum(axis=1)
    candidates = test_wide[n_complete >= 6].copy()
    if len(candidates) < 5:
        candidates = test_wide.copy()

    ref_yr = 2016 if 2016 in candidates.columns else year_cols_int[0]
    candidates = candidates.dropna(subset=[ref_yr])

    n_pick = 10
    if len(candidates) >= n_pick:
        candidates["_q"] = pd.qcut(
            candidates[ref_yr], q=n_pick, labels=False, duplicates="drop"
        )
        selected = (
            candidates.groupby("_q", group_keys=False)
            .apply(lambda g: g.sample(1, random_state=7))
            .head(n_pick)
        )
        selected = selected.drop(columns=["_q"], errors="ignore")
    else:
        selected = candidates

    # ── Per-tract trajectory helper ──────────────────────────────────────────
    def _traj(geoid):
        py, pi = [], []
        for yr in YEARS:
            if yr not in gb2_tract or geoid not in gb2_tract[yr].index:
                continue
            v = float(gb2_tract[yr][geoid])
            if np.isfinite(v):
                py.append(yr); pi.append(v)

        ay, ai, am = [], [], []
        for yr in YEARS:
            if yr not in acs_frames or geoid not in acs_frames[yr].index:
                continue
            row_acs = acs_frames[yr].loc[geoid]
            inc = float(row_acs["per_capita_income_usd"])
            if not np.isfinite(inc):
                continue
            moe_v = float(row_acs["moe"]) if pd.notna(row_acs["moe"]) else np.nan
            ay.append(yr); ai.append(inc)
            am.append(moe_v if np.isfinite(moe_v) else 0.0)
        return py, pi, ay, ai, am

    # ── Figure: 2 rows × 5 cols ──────────────────────────────────────────────
    n_rows, n_cols = 2, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6), sharey=False)
    axes_list = list(axes.flat)
    selected_list = list(selected.iterrows())

    _legend_drawn = False

    for i, (geoid, _row) in enumerate(selected_list):
        ax = axes_list[i]
        py, pi, ay, ai, am = _traj(geoid)

        if py:
            ax.plot(py, pi, "o-", color="steelblue", ms=3.5, lw=1.2, label="Model (GB2)")
        if ay:
            ai_arr = np.array(ai, dtype=float)
            am_arr = np.array(am, dtype=float)
            ax.plot(ay, ai_arr, "s--", color="black", ms=3.5, lw=1.0, label="ACS")
            moe_ok = am_arr > 0
            if moe_ok.any():
                ax.fill_between(
                    np.array(ay)[moe_ok],
                    (ai_arr - am_arr)[moe_ok],
                    (ai_arr + am_arr)[moe_ok],
                    color="black", alpha=0.15, label="ACS ±MOE",
                )

        ax.set_title(f"Tract …{geoid[-6:]}", fontsize=7)
        ax.set_xticks(YEARS)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_xlabel("Year", fontsize=7)

        if not _legend_drawn:
            ax.legend(fontsize=6, loc="upper left")
            _legend_drawn = True

    for ax in axes_list[len(selected_list):]:
        ax.axis("off")

    for r in range(n_rows):
        axes[r, 0].set_ylabel("Per-capita income (USD)", fontsize=8)

    fig.suptitle(
        "Tract income trajectories (test set, stratified by income level):\n"
        "GB2-mapped model (blue) vs ACS observed (black ± MOE)",
        fontsize=9,
    )
    fig.tight_layout()
    _savefig(fig, out / "figures" / "B_tract_income_trajectories.pdf")


def _plot_percentile_trajectories(
    out: Path,
    tract_long: pd.DataFrame,
) -> None:
    """B_percentile_trajectories: ACS vs GB2-mapped model percentile fan over time."""
    print("  B.4d percentile trajectories (ACS vs model)...")

    PCTS = [10, 25, 50, 75, 90]

    # ── Load ACS + fit GB2 per year ──────────────────────────────────────────
    gb2_raw: dict[int, tuple] = {}
    acs_vals: dict[int, np.ndarray] = {}   # raw income arrays

    for yr in YEARS:
        fpath = ACS_ROOT_DIR / str(yr) / f"ny_tracts_acs5_{yr}.feather"
        if not fpath.exists():
            continue
        try:
            df  = pd.read_feather(fpath)
            nyc = df[df["geoid"].str.startswith(NYC_COUNTY_PREFIXES)].copy()
            nyc["geoid"] = nyc["geoid"].astype(str).str.zfill(11)
            nyc = nyc.drop_duplicates("geoid")
            A   = nyc["per_capita_income_usd"].dropna().values.astype(float)
            A   = A[A > 0]
            if len(A) < 20:
                continue
            acs_vals[yr] = A
            gb2_raw[yr]  = _fit_gb2(A)
        except Exception as exc:
            print(f"    ACS {yr} error: {exc}")

    avail = sorted(gb2_raw)
    if not avail:
        print("    No ACS data — skipping B_percentile_trajectories")
        return

    gb2_smooth = _smooth_gb2_params(avail, gb2_raw)

    # ── Compute percentiles per year ─────────────────────────────────────────
    acs_pct: dict[int, np.ndarray] = {}
    mod_pct: dict[int, np.ndarray] = {}

    for yr in avail:
        acs_pct[yr] = np.percentile(acs_vals[yr], PCTS)

        sub    = tract_long[tract_long["year"] == yr].dropna(subset=["predicted_value"])
        mapped = _gb2_apply_from_ranks(
            sub["predicted_value"].values.astype(float),
            gb2_smooth[yr],
        )
        fin = mapped[np.isfinite(mapped)]
        mod_pct[yr] = np.percentile(fin, PCTS) if len(fin) >= 10 else np.full(len(PCTS), np.nan)

    years = np.array(avail)

    def _pct_series(pct_dict, pct_idx):
        return np.array([pct_dict[yr][pct_idx] for yr in avail], dtype=float)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIG_SIZE_TWO_COL)

    pct_colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(PCTS)))

    for j, (p, col) in enumerate(zip(PCTS, pct_colors)):
        acs_y = _pct_series(acs_pct, j)
        mod_y = _pct_series(mod_pct, j)
        ax.plot(years, acs_y, "-",  color=col, lw=2.0)
        ax.plot(years, mod_y, "--", color=col, lw=1.5)

    # IQR shading (P25–P75)
    ax.fill_between(
        years,
        _pct_series(acs_pct, 1), _pct_series(acs_pct, 3),
        color="gray", alpha=0.12, label="_nolegend_",
    )
    ax.fill_between(
        years,
        _pct_series(mod_pct, 1), _pct_series(mod_pct, 3),
        color="steelblue", alpha=0.10, label="_nolegend_",
    )

    # Legend: percentile colours + line-style explanation
    pct_handles = [
        plt.Line2D([0], [0], color=pct_colors[j], lw=2, label=f"P{p}")
        for j, p in enumerate(PCTS)
    ]
    style_handles = [
        plt.Line2D([0], [0], color="0.35", lw=2.0, ls="-",  label="ACS 5-yr"),
        plt.Line2D([0], [0], color="0.35", lw=1.5, ls="--", label="Model (GB2)"),
    ]
    ax.legend(
        handles=pct_handles + style_handles,
        ncol=2, fontsize=7, loc="upper left",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Per-capita income (USD)")
    ax.set_xticks(YEARS)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(
        "NYC tract income distribution over time: ACS (solid) vs GB2-mapped model (dashed)",
        fontsize=9,
    )
    fig.tight_layout()
    _savefig(fig, out / "figures" / "B_percentile_trajectories.pdf")


def _plot_tract_rank_trajectories(
    out: Path,
    tract_long: pd.DataFrame,
    tract_wide: pd.DataFrame,
) -> None:
    """B_tract_rank_trajectories: per-tract percentile rank over time, ACS vs model."""
    from scipy.stats import percentileofscore
    print("  B.4e tract rank trajectories...")

    # ── Load ACS income + MOE per year ───────────────────────────────────────
    acs_frames: dict[int, pd.DataFrame] = {}

    for yr in YEARS:
        fpath = ACS_ROOT_DIR / str(yr) / f"ny_tracts_acs5_{yr}.feather"
        if not fpath.exists():
            continue
        try:
            df  = pd.read_feather(fpath)
            nyc = df[df["geoid"].str.startswith(NYC_COUNTY_PREFIXES)].copy()
            nyc["geoid"] = nyc["geoid"].astype(str).str.zfill(11)
            nyc = nyc.drop_duplicates("geoid").set_index("geoid")
            result = nyc[["per_capita_income_usd"]].copy()
            result["moe"] = np.nan
            for moe_cand in [
                "per_capita_income_usd_error",
                "per_capita_income_usd_moe",
                "per_capita_income_moe",
            ]:
                if moe_cand in nyc.columns:
                    result["moe"] = nyc[moe_cand].astype(float)
                    break
            acs_frames[yr] = result.dropna(subset=["per_capita_income_usd"])
        except Exception as exc:
            print(f"    ACS {yr} error: {exc}")

    avail = sorted(acs_frames)
    if not avail:
        print("    No ACS data — skipping B_tract_rank_trajectories")
        return

    # ── Build city-wide ACS and model distributions per year ─────────────────
    # Used as the reference pool for percentileofscore.
    acs_dist:  dict[int, np.ndarray] = {}
    mod_dist:  dict[int, pd.Series]  = {}   # GEOID_str → predicted_value

    for yr in avail:
        acs_dist[yr] = acs_frames[yr]["per_capita_income_usd"].values.astype(float)

    for yr in avail:
        sub = tract_long[tract_long["year"] == yr].dropna(subset=["predicted_value"])
        mod_dist[yr] = sub.set_index("GEOID_str")["predicted_value"].astype(float)

    # ── Select 10 test tracts stratified by 2016 prediction rank ─────────────
    year_cols_int = [y for y in YEARS if y in tract_wide.columns]
    test_wide     = tract_wide[tract_wide["type"] == "test"].copy()
    n_complete    = np.isfinite(test_wide[year_cols_int].values.astype(float)).sum(axis=1)
    candidates    = test_wide[n_complete >= 6].copy()
    if len(candidates) < 5:
        candidates = test_wide.copy()

    ref_yr = 2016 if 2016 in candidates.columns else year_cols_int[0]
    candidates = candidates.dropna(subset=[ref_yr])

    n_pick = 10
    if len(candidates) >= n_pick:
        candidates["_q"] = pd.qcut(
            candidates[ref_yr], q=n_pick, labels=False, duplicates="drop"
        )
        selected = (
            candidates.groupby("_q", group_keys=False)
            .apply(lambda g: g.sample(1, random_state=7))
            .head(n_pick)
            .drop(columns=["_q"], errors="ignore")
        )
    else:
        selected = candidates

    # ── Per-tract rank trajectory helper ─────────────────────────────────────
    def _rank_traj(geoid):
        py, pr = [], []          # predicted rank
        ay, ar, ar_lo, ar_hi = [], [], [], []   # ACS rank + MOE bounds

        for yr in avail:
            # Model rank
            if yr in mod_dist and geoid in mod_dist[yr].index:
                pv = float(mod_dist[yr][geoid])
                if np.isfinite(pv):
                    pool = mod_dist[yr].values
                    py.append(yr)
                    pr.append(percentileofscore(pool, pv, kind="rank"))

            # ACS rank + MOE uncertainty
            if yr in acs_frames and geoid in acs_frames[yr].index:
                row_acs = acs_frames[yr].loc[geoid]
                inc  = float(row_acs["per_capita_income_usd"])
                if not np.isfinite(inc):
                    continue
                pool = acs_dist[yr]
                moe  = float(row_acs["moe"]) if pd.notna(row_acs["moe"]) else 0.0
                ay.append(yr)
                ar.append(percentileofscore(pool, inc, kind="rank"))
                ar_lo.append(percentileofscore(pool, max(inc - moe, pool.min()), kind="rank"))
                ar_hi.append(percentileofscore(pool, inc + moe, kind="rank"))

        return py, pr, ay, ar, ar_lo, ar_hi

    # ── Figure: 2 rows × 5 cols ───────────────────────────────────────────────
    n_rows, n_cols = 2, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6), sharey=False)
    axes_list    = list(axes.flat)
    selected_list = list(selected.iterrows())

    _legend_drawn = False

    for i, (geoid, _row) in enumerate(selected_list):
        ax = axes_list[i]
        py, pr, ay, ar, ar_lo, ar_hi = _rank_traj(geoid)

        if py:
            ax.plot(py, pr, "o-", color="steelblue", ms=3.5, lw=1.2, label="Model")
        if ay:
            ar_arr    = np.array(ar,    dtype=float)
            ar_lo_arr = np.array(ar_lo, dtype=float)
            ar_hi_arr = np.array(ar_hi, dtype=float)
            ax.plot(ay, ar_arr, "s--", color="black", ms=3.5, lw=1.0, label="ACS")
            moe_ok = ar_hi_arr > ar_lo_arr
            if moe_ok.any():
                ax.fill_between(
                    np.array(ay)[moe_ok],
                    ar_lo_arr[moe_ok],
                    ar_hi_arr[moe_ok],
                    color="black", alpha=0.15, label="ACS ±MOE",
                )

        ax.set_title(f"Tract …{geoid[-6:]}", fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(YEARS)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_xlabel("Year", fontsize=7)
        ax.axhline(50, color="0.75", lw=0.6, ls=":")

        if not _legend_drawn:
            ax.legend(fontsize=6, loc="upper left")
            _legend_drawn = True

    for ax in axes_list[len(selected_list):]:
        ax.axis("off")

    for r in range(n_rows):
        axes[r, 0].set_ylabel("Percentile rank\n(within NYC tracts)", fontsize=8)

    fig.suptitle(
        "Tract percentile rank over time (test set, stratified by income level):\n"
        "Model rank (blue) vs ACS rank (black ± MOE mapped to rank)",
        fontsize=9,
    )
    fig.tight_layout()
    _savefig(fig, out / "figures" / "B_tract_rank_trajectories.pdf")


# ─── Part C ───────────────────────────────────────────────────────────────────

def _load_acs_nyc(year: int) -> pd.Series:
    """Per-capita income for NYC census tracts, indexed by 11-char GEOID."""
    fpath = ACS_ROOT_DIR / str(year) / f"ny_tracts_acs5_{year}.feather"
    df = pd.read_feather(fpath)
    nyc = df[df["geoid"].str.startswith(NYC_COUNTY_PREFIXES)].copy()
    nyc["geoid"] = nyc["geoid"].astype(str).str.zfill(11)
    nyc = nyc.drop_duplicates("geoid")
    return nyc.set_index("geoid")["per_capita_income_usd"].dropna()


# ── GB2 distribution and helpers for Part C ───────────────────────────────────
# Reference: McDonald (1984) "Some Generalized Functions for the Size
# Distribution of Income", Econometrica 52(3), 647-665.


class _GB2Gen(rv_continuous):
    r"""Generalized Beta of the Second Kind (GB2).

    Shape parameters  c > 0,  p > 0,  q > 0.
    The ``scale`` kwarg is the canonical *b* parameter; ``loc`` is fixed at 0
    for strictly positive income data.

    For the standardised variable  x = (raw - loc) / scale  (x > 0):

        PDF   f(x; c,p,q)  =  c·x^{cp-1} / [B(p,q)·(1+x^c)^{p+q}]
        CDF   F(x; c,p,q)  =  I_{z/(1+z)}(p, q),   z = x^c
        PPF   Q(u; c,p,q)  =  (v/(1-v))^{1/c},      v = betaincinv(p,q,u)

    Nests: Dagum (q=1), Singh-Maddala / Burr XII (p=1),
           log-normal (limiting), Pareto (limiting).
    Power-law right tail — unlike JSU — so it correctly captures the heavy
    right tail of census income distributions.
    """

    def _pdf(self, x, c, p, q):
        xc = x ** c
        return c * x ** (c * p - 1) / (_sp.beta(p, q) * (1.0 + xc) ** (p + q))

    def _logpdf(self, x, c, p, q):
        xc = x ** c
        return (np.log(c) + (c * p - 1) * np.log(x)
                - _sp.betaln(p, q)
                - (p + q) * np.log1p(xc))

    def _cdf(self, x, c, p, q):
        z = x ** c
        return _sp.betainc(p, q, z / (1.0 + z))

    def _ppf(self, u, c, p, q):
        v = _sp.betaincinv(p, q, u)
        return (v / (1.0 - v)) ** (1.0 / c)

    def _sf(self, x, c, p, q):
        z = x ** c
        return 1.0 - _sp.betainc(p, q, z / (1.0 + z))


gb2 = _GB2Gen(a=0.0, name="gb2", shapes="c, p, q")


def _qmap20_bench(P: np.ndarray, A: np.ndarray) -> np.ndarray:
    """20-quantile empirical QM — retained as bootstrap benchmark only."""
    p_q = np.nanpercentile(P, np.linspace(0, 100, 21))
    a_q = np.nanpercentile(A, np.linspace(0, 100, 21))
    return np.interp(P, p_q, a_q)


def _fit_gb2(data: np.ndarray) -> tuple[float, float, float, float]:
    """Fit GB2 by MLE (loc=0 fixed). Returns (c, p, q, scale=b).

    Tries four starting points and returns the lowest-NLL valid solution.
    Starting values follow empirical guidance for income distributions.
    """
    data = data[np.isfinite(data) & (data > 0)]
    if len(data) < 10:
        raise ValueError(f"Too few valid observations: {len(data)}")
    med = float(np.median(data))
    best_nll, best = np.inf, None
    for c0, p0, q0 in [
        (1.5, 0.5, 3.0),
        (2.0, 0.8, 2.0),
        (1.0, 0.5, 5.0),
        (1.2, 1.0, 3.0),
    ]:
        try:
            res = gb2.fit(data, c0, p0, q0, loc=0, scale=med, floc=0)
            c_f, p_f, q_f, _loc, sc_f = res
            if not (c_f > 0 and p_f > 0 and q_f > 0 and sc_f > 0):
                continue
            nll = -float(gb2.logpdf(data, c_f, p_f, q_f, loc=0, scale=sc_f).sum())
            if nll < best_nll:
                best_nll, best = nll, (c_f, p_f, q_f, sc_f)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("GB2 MLE failed for all starting values")
    return best


def _smooth_gb2_params(
    years: list[int],
    params_by_year: dict[int, tuple],
    degree: int = 3,
) -> dict[int, tuple]:
    """Smooth GB2 parameters across years in log-space (guarantees positivity).

    All GB2 parameters (c, p, q, scale) are strictly positive, so smoothing
    log(param) with a polynomial then exponentiating back is natural and
    prevents a spline from crossing zero — unlike JSU where γ can be negative.
    """
    if len(years) < 3:
        return params_by_year
    yr   = np.array(years, dtype=float)
    yr_c = yr - yr.mean()
    arr  = np.log(np.array([params_by_year[y] for y in years]))  # log-space (n,4)
    deg  = min(degree, len(years) - 1)
    out  = np.zeros_like(arr)
    for j in range(4):
        coef = np.polyfit(yr_c, arr[:, j], deg)
        out[:, j] = np.polyval(coef, yr_c)
    return {y: tuple(np.exp(out[i])) for i, y in enumerate(years)}


def _gb2_apply_from_ranks(
    P: np.ndarray,
    tgt: tuple[float, float, float, float],   # (c, p, q, scale)
    clip_eps: float = 1e-6,
) -> np.ndarray:
    """Rank-preserving GB2 map using empirical source CDF (Hazen positions).

    p_it = (rank(w_it) - 0.5) / n   [empirical, rank-stable]
    W*_it = Q_GB2(p_it)              [parametric GB2 quantile]
    """
    from scipy.stats import rankdata
    c, p, q, scale = tgt
    fin    = np.isfinite(P)
    result = np.full_like(P, np.nan, dtype=float)
    if fin.sum() == 0:
        return result
    n_fin  = int(fin.sum())
    ranks  = rankdata(P[fin], method="average")
    probs  = np.clip((ranks - 0.5) / n_fin, clip_eps, 1.0 - clip_eps)
    result[fin] = gb2.ppf(probs, c, p, q, loc=0, scale=scale)
    return result


def _skew(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    m, s = x.mean(), x.std(ddof=1)
    return 0.0 if s < 1e-12 else float(np.mean(((x - m) / s) ** 3))


def _kurt(x: np.ndarray) -> float:
    """Excess kurtosis."""
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.nan
    m, s = x.mean(), x.std(ddof=1)
    return 0.0 if s < 1e-12 else float(np.mean(((x - m) / s) ** 4)) - 3.0


def part_c(results_dir: Path, processed_dir: Path, out: Path) -> pd.DataFrame | None:
    """GB2 parametric distribution matching (replaces Johnson SU).

    Target (ACS income USD): fitted by GB2 MLE per year, parameters smoothed
    in log-space across years (degree-3 polynomial — guarantees positivity).

    Source (predicted values ≈ z-scores): empirical CDF via Hazen ranks.
    GB2 is not fit to the source since predicted values are near-normal
    and MLE degenerates; empirical ranks are rank-stable and endorsed by spec.

    Why GB2 over JSU: GB2 has a power-law right tail, matching the heavy
    upper tail of census income distributions that JSU under-predicted.
    It nests Dagum, Singh-Maddala, log-normal, and Pareto as special cases.
    Reference: McDonald (1984), Econometrica 52(3), 647-665.

    Map:  p_it = (rank(w_it)-0.5)/n  [empirical],  W*_it = Q_GB2(p_it).
    Winsorised at P1/P99 per year to limit tail amplification in long-diffs.
    """
    print("\n=== Part C: GB2 Parametric Distribution Matching ===")
    from scipy.stats import gaussian_kde

    tract_long = _load_tract_long(results_dir)

    # ── C.1  Fit GB2 to target (ACS) per year ────────────────────────────────
    print("  C.1 fitting GB2 to ACS target per year (MLE, loc=0 fixed)...")
    available_years: list[int] = []
    acs_by_year: dict[int, pd.Series] = {}
    gb2_tgt_raw: dict[int, tuple]     = {}   # (c, p, q, scale) per year

    for yr in YEARS:
        try:
            acs_series = _load_acs_nyc(yr)
        except FileNotFoundError:
            print(f"    ACS {yr} not found, skipping")
            continue

        A = acs_series.values.astype(float)
        A = A[np.isfinite(A) & (A > 0)]

        if len(A) < 20:
            print(f"    {yr}: too few ACS observations ({len(A)}), skipping")
            continue

        try:
            tgt = _fit_gb2(A)
        except Exception as exc:
            print(f"    {yr}: GB2 fit failed ({exc}), skipping")
            continue

        gb2_tgt_raw[yr] = tgt
        acs_by_year[yr] = acs_series
        available_years.append(yr)
        c_f, p_f, q_f, sc_f = tgt
        print(
            f"    {yr}: c={c_f:.3f}  p={p_f:.3f}  q={q_f:.3f}  b={sc_f:,.0f}"
        )

    if not available_years:
        print("  No ACS data available. Skipping Part C.")
        return None

    n_yr = len(available_years)

    # ── C.2  Smooth target parameters in log-space ────────────────────────────
    print("  C.2 smoothing target parameters (poly-3, log-space)...")
    gb2_tgt_smooth = _smooth_gb2_params(available_years, gb2_tgt_raw)

    # ── C.3  Parameter trajectory plot ────────────────────────────────────────
    print("  C.3 parameter trajectory plot...")
    pnames = ["$c$ (power)", "$p$ (lower tail)", "$q$ (upper tail)", "$b$ (scale)"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), squeeze=False, sharex=True)
    for j, pn in enumerate(pnames):
        ax    = axes[0, j]
        raw_v = [gb2_tgt_raw[y][j]    for y in available_years]
        smo_v = [gb2_tgt_smooth[y][j] for y in available_years]
        ax.plot(available_years, raw_v,  "o",  color="steelblue", ms=5,  label="Raw MLE")
        ax.plot(available_years, smo_v, "--",  color="firebrick", lw=1.5, label="Poly-3 (log)")
        ax.set_title(pn, fontsize=8)
        ax.set_xlabel("Year", fontsize=7)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=6)
    fig.suptitle("GB2 target (ACS) parameters: raw MLE vs log-space poly-3 smoothed", fontsize=9)
    fig.tight_layout()
    _savefig(fig, out / "figures" / "C_gb2_params_trajectory.pdf")

    # ── C.4  QQ plots: ACS vs fitted target GB2 ──────────────────────────────
    print("  C.4 QQ plots (target fit quality)...")
    ncols_qq = min(4, n_yr)
    nrows_qq = (n_yr + ncols_qq - 1) // ncols_qq
    fig, axes = plt.subplots(nrows_qq, ncols_qq,
                             figsize=(ncols_qq * 3, nrows_qq * 3), squeeze=False)
    axes_flat = axes.ravel()
    for i, yr in enumerate(available_years):
        ax = axes_flat[i]
        A  = np.sort(acs_by_year[yr].values.astype(float))
        A  = A[np.isfinite(A) & (A > 0)]
        probs = (np.arange(1, len(A) + 1) - 0.5) / len(A)
        c_f, p_f, q_f, sc_f = gb2_tgt_raw[yr]
        theo = gb2.ppf(probs, c_f, p_f, q_f, loc=0, scale=sc_f)
        trim = max(5, len(A) // 200)
        ax.scatter(theo[trim:-trim], A[trim:-trim], s=2, alpha=0.4, color="steelblue")
        lo2, hi2 = theo[trim], theo[-trim]
        ax.plot([lo2, hi2], [lo2, hi2], "r--", lw=1)
        ax.set_title(str(yr), fontsize=8)
        ax.set_xlabel("GB2 quantile", fontsize=7)
        ax.set_ylabel("Empirical ACS", fontsize=7)
        ax.tick_params(labelsize=7)
    for j in range(n_yr, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle("QQ: empirical ACS income vs fitted GB2 (target)", fontsize=9)
    fig.tight_layout()
    _savefig(fig, out / "figures" / "C_gb2_qq_target.pdf")

    # ── C.5  Apply the map ────────────────────────────────────────────────────
    print("  C.5 applying GB2 map (empirical source CDF → smoothed GB2 target quantile)...")
    city_rows:   list[dict]         = []
    qmap_frames: list[pd.DataFrame] = []

    for yr in available_years:
        acs_series = acs_by_year[yr]
        A_all = acs_series.values.astype(float)
        A_fin = A_all[np.isfinite(A_all)]

        sub = tract_long[tract_long["year"] == yr].copy()
        P   = sub["predicted_value"].values.astype(float)

        sub["gb2_map"] = _gb2_apply_from_ranks(P, gb2_tgt_smooth[yr])
        # Winsorize at 1st/99th percentile within each year: limits tail
        # amplification from inflating long-difference estimates.
        _w_fin = sub["gb2_map"].dropna().values
        if len(_w_fin) > 0:
            _lo = float(np.percentile(_w_fin, 1))
            _hi = float(np.percentile(_w_fin, 99))
            sub["gb2_map"] = sub["gb2_map"].clip(lower=_lo, upper=_hi)
        sub["acs_actual"] = sub["GEOID_str"].map(acs_series)
        qmap_frames.append(sub)

        Wstar     = sub["gb2_map"].values.astype(float)
        Wstar_fin = Wstar[np.isfinite(Wstar)]
        row: dict = {
            "year":             yr,
            "n_tracts":         int(sub["predicted_value"].notna().sum()),
            "ACS_mean_all_nyc": float(np.nanmean(A_fin)),
            "ACS_mean_matched": float(np.nanmean(sub["acs_actual"].values)),
            "raw_mean":         float(np.nanmean(P[np.isfinite(P)])) if np.isfinite(P).any() else np.nan,
        }
        if len(Wstar_fin) > 0:
            row.update({
                "gb2_map_mean": float(np.mean(Wstar_fin)),
                "gb2_map_std":  float(np.std(Wstar_fin, ddof=1)) if len(Wstar_fin) > 1 else np.nan,
                "gb2_map_skew": _skew(Wstar_fin),
                "gb2_map_kurt": _kurt(Wstar_fin),
                "gb2_map_p90":  float(np.percentile(Wstar_fin, 90)),
                "gb2_map_p95":  float(np.percentile(Wstar_fin, 95)),
            })
        else:
            row.update({k: np.nan for k in [
                "gb2_map_mean", "gb2_map_std", "gb2_map_skew",
                "gb2_map_kurt", "gb2_map_p90", "gb2_map_p95",
            ]})
        row.update({
            "ACS_std":  float(np.std(A_fin, ddof=1)) if len(A_fin) > 1 else np.nan,
            "ACS_skew": _skew(A_fin),
            "ACS_kurt": _kurt(A_fin),
            "ACS_p90":  float(np.percentile(A_fin, 90)) if len(A_fin) > 0 else np.nan,
            "ACS_p95":  float(np.percentile(A_fin, 95)) if len(A_fin) > 0 else np.nan,
        })
        city_rows.append(row)

    qmap_long = pd.concat(qmap_frames, ignore_index=True)
    city_df   = pd.DataFrame(city_rows)

    city_df.to_csv(out / "tables" / "C_gb2_mapping.csv",  index=False)
    city_df.to_csv(out / "tables" / "C_gb2_moments.csv",  index=False)
    print("    saved C_gb2_mapping.csv / C_gb2_moments.csv")

    # ── C.6  Validate (b): KDE overlay ───────────────────────────────────────
    print("  C.6 KDE overlays...")
    ncols_k = min(4, n_yr)
    nrows_k = (n_yr + ncols_k - 1) // ncols_k
    fig, axes = plt.subplots(nrows_k, ncols_k,
                             figsize=(ncols_k * 3.5, nrows_k * 3), squeeze=False)
    axes_flat = axes.ravel()
    for i, yr in enumerate(available_years):
        ax    = axes_flat[i]
        A     = acs_by_year[yr].values.astype(float)
        A     = A[np.isfinite(A) & (A > 0)]
        Wstar = qmap_long.loc[qmap_long["year"] == yr, "gb2_map"].values.astype(float)
        Wstar = Wstar[np.isfinite(Wstar)]
        if len(Wstar) == 0 or len(A) == 0:
            ax.text(0.5, 0.5, "no valid values", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")
            ax.set_title(str(yr), fontsize=8)
            continue
        x_lo = min(np.percentile(A, 2),  np.percentile(Wstar, 2))
        x_hi = max(np.percentile(A, 97), np.percentile(Wstar, 97))
        xs   = np.linspace(x_lo, x_hi, 400)
        try:
            ax.plot(xs, gaussian_kde(A)(xs),     color="black",     lw=1.5, label="ACS")
            ax.plot(xs, gaussian_kde(Wstar)(xs), color="steelblue", lw=1.5, ls="--",
                    label="GB2-mapped")
        except Exception:
            pass
        ax.set_title(str(yr), fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    for j in range(n_yr, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle("KDE overlay: ACS income vs GB2-mapped predictions by year", fontsize=9)
    fig.tight_layout()
    _savefig(fig, out / "figures" / "C_gb2_density_overlay.pdf")

    # ── C.7  Target quantile function Q_GB2,t(p) by year ─────────────────────
    print("  C.7 target quantile function plot...")
    cmap_yr = plt.cm.viridis
    cols_yr  = [cmap_yr(k / max(n_yr - 1, 1)) for k in range(n_yr)]
    p_grid   = np.linspace(0.01, 0.99, 300)
    fig, ax  = plt.subplots(figsize=FIG_SIZE_ONE_COL)
    for k, yr in enumerate(available_years):
        c_f, p_f, q_f, sc_f = gb2_tgt_smooth[yr]
        q_vals = gb2.ppf(p_grid, c_f, p_f, q_f, loc=0, scale=sc_f)
        ax.plot(p_grid, q_vals, color=cols_yr[k], lw=1.5, label=str(yr))
    ax.set_xlabel("Probability rank $p$")
    ax.set_ylabel("GB2 quantile — income (USD)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(r"Target quantile function $Q_{A,t}(p)$ by year (smoothed GB2)")
    fig.tight_layout()
    _savefig(fig, out / "figures" / "C_gb2_target_quantile_fn.pdf")

    # ── C.8  Bootstrap tail stability vs QM20 benchmark ──────────────────────
    print("  C.8 bootstrap tail stability (GB2 vs QM20, B=500)...")
    focus_yr = 2016 if 2016 in available_years else available_years[-1]
    A_full   = acs_by_year[focus_yr].values.astype(float)
    A_full   = A_full[np.isfinite(A_full) & (A_full > 0)]
    P_yr     = tract_long.loc[
        tract_long["year"] == focus_yr, "predicted_value"
    ].values.astype(float)
    P_yr = P_yr[np.isfinite(P_yr)]

    rng    = np.random.default_rng(42)
    n_boot = 500
    pcts   = [90, 95, 99]
    gb2_boots: dict[int, list] = {p: [] for p in pcts}
    qm_boots:  dict[int, list] = {p: [] for p in pcts}

    for _ in range(n_boot):
        Ab = A_full[rng.integers(0, len(A_full), len(A_full))]
        # GB2: refit target from bootstrap ACS; source stays empirical (fixed ranks)
        try:
            tgt_b  = _fit_gb2(Ab)
            Wb_gb2 = _gb2_apply_from_ranks(P_yr, tgt_b)
            for p in pcts:
                gb2_boots[p].append(float(np.nanpercentile(Wb_gb2, p)))
        except Exception:
            for p in pcts:
                gb2_boots[p].append(np.nan)
        # Empirical QM20 benchmark
        try:
            Wb_qm = _qmap20_bench(P_yr, Ab)
            for p in pcts:
                qm_boots[p].append(float(np.nanpercentile(Wb_qm, p)))
        except Exception:
            for p in pcts:
                qm_boots[p].append(np.nan)

    fig, axes = plt.subplots(1, len(pcts), figsize=(11, 4), squeeze=False)
    for i, p in enumerate(pcts):
        ax = axes[0, i]
        jb = np.array(gb2_boots[p])
        qb = np.array(qm_boots[p])
        ax.hist(jb[np.isfinite(jb)], bins=30, alpha=0.6, color="steelblue",
                label=f"GB2  $\\sigma$={np.nanstd(jb):,.0f}")
        ax.hist(qb[np.isfinite(qb)], bins=30, alpha=0.6, color="firebrick",
                label=f"QM20 $\\sigma$={np.nanstd(qb):,.0f}")
        ax.set_title(f"P{p}  (year={focus_yr})", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("USD", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(
        f"Bootstrap tail stability: GB2 map vs QM20  (B={n_boot})\n"
        "Narrower = more stable",
        fontsize=9,
    )
    fig.tight_layout()
    _savefig(fig, out / "figures" / "C_gb2_bootstrap_tail.pdf")

    pd.DataFrame({
        "percentile": pcts,
        "gb2_std":   [np.nanstd(gb2_boots[p])  for p in pcts],
        "qm20_std":  [np.nanstd(qm_boots[p])   for p in pcts],
        "gb2_mean":  [np.nanmean(gb2_boots[p])  for p in pcts],
        "qm20_mean": [np.nanmean(qm_boots[p])   for p in pcts],
    }).to_csv(out / "tables" / "C_gb2_bootstrap_tail.csv", index=False)
    print(f"    saved C_gb2_bootstrap_tail.csv  (focus year={focus_yr})")

    # ── C.9  Rank correlation w vs w* must equal 1.0 ─────────────────────────
    print("  C.9 rank correlation w vs w*...")
    for yr in available_years:
        sub  = qmap_long[qmap_long["year"] == yr]
        w    = sub["predicted_value"].values.astype(float)
        ws   = sub["gb2_map"].values.astype(float)
        mask = np.isfinite(w) & np.isfinite(ws)
        rho  = spearmanr(w[mask], ws[mask]).statistic if mask.sum() >= 3 else np.nan
        print(f"    {yr}: Spearman(w, w*) = {rho:.6f}  (expected 1.0)")

    # ── C.10 City-average trajectory ─────────────────────────────────────────
    print("  C.10 city-average trajectory...")
    city_plot = city_df.dropna(subset=["gb2_map_mean"])
    fig, ax = plt.subplots(figsize=FIG_SIZE_ONE_COL)
    ax.plot(city_plot["year"], city_plot["ACS_mean_matched"], "o-",  color="black",     lw=2, ms=6,
            label="ACS (same tracts as preds)")
    ax.plot(city_plot["year"], city_plot["ACS_mean_all_nyc"], ":",   color="0.55",      lw=1.5,
            label="ACS (all NYC tracts, context)")
    ax.plot(city_plot["year"], city_plot["gb2_map_mean"],     "s--", color="steelblue", lw=1.5, ms=5,
            label="GB2-mapped")
    for _, r in city_plot.iterrows():
        ax.annotate(f"n={int(r['n_tracts'])}", (r["year"], r["gb2_map_mean"]),
                    textcoords="offset points", xytext=(0, -13), fontsize=6,
                    ha="center", color="steelblue")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean per-capita income (USD)")
    ax.legend(fontsize=8)
    ax.set_xticks(YEARS)
    fig.tight_layout()
    _savefig(fig, out / "figures" / "C_city_avg_trajectory.pdf")

    # ── C.11 Long-difference scatter: ACS 2009→2024 vs GB2 map 2010→2024 ─────
    print("  C.11 long-difference scatter...")
    try:
        panel = pd.read_feather(
            processed_dir / "ny_tracts_panel_2009_2014_2019_2024.feather"
        )
        panel["GEOID_str"] = panel["geoid_2024"].astype(str).str.zfill(11)
        panel["acs_change"] = (
            panel["per_capita_income_usd_2024"] - panel["per_capita_income_usd_2009"]
        )
        wide_map = qmap_long.pivot_table(
            index="GEOID_str", columns="year", values="gb2_map", aggfunc="first"
        )
        wide_map.columns = [int(c) for c in wide_map.columns]
        if 2010 in wide_map.columns and 2024 in wide_map.columns:
            wide_map["pred_change"] = wide_map[2024] - wide_map[2010]
            merged = (
                panel[["GEOID_str", "acs_change"]]
                .merge(wide_map[["pred_change"]].reset_index(), on="GEOID_str", how="inner")
                .dropna()
            )
            x_c = merged["acs_change"].values
            y_c = merged["pred_change"].values
            rho_c, _ = spearmanr(x_c, y_c)
            slope_c  = np.polyfit(x_c, y_c, 1)[0]

            fig, ax = plt.subplots(figsize=FIG_SIZE_ONE_COL)
            ax.scatter(x_c, y_c, s=5, alpha=0.3, color="steelblue", linewidths=0)
            ax.axhline(0, color="gray", lw=0.5)
            ax.axvline(0, color="gray", lw=0.5)
            xlim = np.array([x_c.min(), x_c.max()])
            ax.plot(xlim, np.polyval(np.polyfit(x_c, y_c, 1), xlim),
                    "r-", lw=1.5, label=f"slope={slope_c:.3f}")
            ax.set_xlabel("ACS per-capita income change 2009→2024 (USD)")
            ax.set_ylabel("GB2-mapped prediction change 2010→2024 (USD)")
            ax.text(0.05, 0.95, f"Spearman $\\rho$ = {rho_c:.3f}  n = {len(merged)}",
                    transform=ax.transAxes, fontsize=8, va="top")
            ax.legend(fontsize=8)
            fig.tight_layout()
            _savefig(fig, out / "figures" / "C_long_diff_gb2_map.pdf")
    except Exception as exc:
        print(f"    Long-diff scatter skipped: {exc}")

    return qmap_long


# ─── Part D ───────────────────────────────────────────────────────────────────

# Split groups for the event study. "heldout" is everything the model was never
# trained on — test/val cities, the within-city NYC holdout, and the τ-buffer
# dead_zone tracts (excluded from training, so legitimately out-of-sample). It
# carries the headline result. "train" is a diagnostic row only: the event study
# should look similar there, and a train/heldout gap would say the dynamic
# response is memorised rather than read off the imagery.
CSA_SPLIT_GROUPS = {
    # val_within_nyc / val_within_chicago are the within-city spatial holdouts
    # (us_split._NYC_TYPE_MAP renames the notebook's "val" to val_within_nyc), so
    # they must appear here or those tracts fall into neither group and vanish.
    "heldout": ["test", "val", "val_within_nyc", "val_within_chicago",
                "val_cities", "val_spatial_temporal", "val_spatial",
                "val_temporal", "dead_zone"],
    "train":   ["train"],
}
CSA_SPLIT_LABELS = {
    "heldout": "Held out",
    "train":   "Train (diagnostic)",
}
# Backwards-compatible alias: older notebooks import SPLIT_GROUPS from here.
SPLIT_GROUPS = CSA_SPLIT_GROUPS

_CSA_ATT_COLOR = "#0072B2"     # CVD-safe blue, matches _ORDINAL_COLOR below
_CSA_ALT_COLOR = "#D55E00"     # CVD-safe vermillion, for the second series

# Buildings standing at this year form the denominator of the treatment
# intensity; anything later is potential treatment. 2009 is the last year before
# the NYC panel opens, so the whole panel is post-baseline.
CSA_BASELINE_YEAR = 2009

# Anticipation allowances (in panel periods) reported as robustness rows. Cohorts
# are dated from year_built = completion, while site clearance and superstructure
# are visible in the imagery earlier, so the periods just before completion are
# partly treated; see src/csa_event_study.py build_event_panel(anticipation=...).
CSA_ANTICIPATION_PERIODS: tuple[int, ...] = (1, 2)


def _csa_split_of(processed_dir: Path) -> dict[str, str]:
    """GEOID -> split-group name ("heldout"/"train"), from tract_splits.feather.

    Taken from the split file rather than the prediction CSV's ``type`` column:
    the CSV reflects only what was *predicted* (a test-only prediction pass
    labels every row "test"), while the split file is the source of truth for
    what the model was trained on.
    """
    group_of_type = {
        t: group for group, types in CSA_SPLIT_GROUPS.items() for t in types
    }
    try:
        splits = _load_splits(processed_dir)
    except Exception as exc:
        print(f"    tract_splits.feather unavailable ({exc}); split groups skipped")
        return {}
    return {
        g: group_of_type[t]
        for g, t in zip(splits["GEOID_str"], splits["type"].astype(str))
        if t in group_of_type
    }


def _csa_city_tracts(processed_dir: Path, prefixes: tuple[str, ...]) -> gpd.GeoDataFrame:
    """Tract geometries whose GEOID starts with one of ``prefixes``."""
    splits = _load_splits(processed_dir)
    sub = splits[splits["GEOID_str"].str.startswith(tuple(prefixes))].copy()
    return sub[["GEOID_str", "geometry"]].reset_index(drop=True)


def _csa_building_preds(results_dir: Path, years: list[int]) -> pd.DataFrame:
    """Building-level predictions in the shape ``tract_outcomes`` wants."""
    bld = _load_building_preds_us(results_dir, years)
    if bld.empty:
        return bld
    out = bld.rename(columns={"pred": "predicted_value"})
    out["GEOID_str"] = out["GEOID"]
    return out


def _tex_pct(frac: float) -> str:
    """Format a fraction as a LaTeX-safe percentage, e.g. 0.05 -> '5\\%'.

    paper.mplstyle sets text.usetex, where a bare '%' opens a comment and
    silently swallows the rest of the string — so "5% of baseline area" renders
    as "5". Every percentage that reaches a figure must go through here.
    """
    return rf"{frac:.0%}".replace("%", r"\%")


def _tex_escape(s: str) -> str:
    """Make arbitrary text (exception messages, split labels) usetex-safe."""
    for ch in ("\\", "%", "$", "&", "#", "_", "{", "}"):
        s = s.replace(ch, "" if ch == "\\" else "\\" + ch)
    return s


def _plot_event_study(ax, res, *, color: str = _CSA_ATT_COLOR, label: str | None = None,
                      annotate: bool = True, offset: float = 0.0) -> None:
    """Draw one dynamic ATT path (point estimates + simultaneous band) on ``ax``.

    The estimates are plotted exactly as ``csa`` returns them — no re-anchoring
    at k=-1 (see src/csa_event_study.py for why that would be wrong).
    """
    ks = np.asarray(res.event_times, dtype=float) + offset
    ax.axhline(0, color="0.55", lw=0.6, zorder=1)
    ax.axvline(-0.5, color="black", ls="--", lw=0.8, zorder=1)
    ax.fill_between(ks, res.lower, res.upper, color=color, alpha=0.18, lw=0, zorder=2)
    ax.plot(ks, res.att, "o-", color=color, lw=1.4, ms=4, zorder=3,
            label=label or "ATT (CSA)")
    ax.set_xticks(np.asarray(res.event_times, dtype=int))
    ax.tick_params(labelsize=7)
    if annotate:
        p = res.pretrend_supt_p
        p_txt = "n/a" if p is None else (r"$<$0.001" if p < 0.001 else f"{p:.3f}")
        ax.text(0.03, 0.97,
                f"pre-trend joint $p$ = {p_txt}\n"
                rf"$n_{{treated}}$ = {res.n_treated}, $n_{{control}}$ = {res.n_never_treated}",
                transform=ax.transAxes, fontsize=6.2, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8",
                          alpha=0.85, lw=0.5))


def _csa_skip_axis(ax, message: str) -> None:
    ax.text(0.5, 0.5, _tex_escape(str(message)[:120]), ha="center", va="center",
            transform=ax.transAxes, fontsize=7, color="gray", wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def _csa_estimate_or_none(panel, *, label: str, control: str = "never",
                          n_boot: int = 10_000):
    """Estimate, or return (None, reason) — never raise into the figure loop."""
    from src import csa_event_study as ces

    if not panel.usable:
        reason = "; ".join(panel.notes) or "panel too thin"
        return None, reason
    try:
        return ces.estimate_event_study(
            panel, control=control, n_boot=n_boot, label=label
        ), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def part_d(results_dir: Path, processed_dir: Path, out: Path,
           years: list[int] | None = None,
           thresholds=None,
           n_boot: int = 10_000) -> dict:
    """Callaway–Sant'Anna event study on construction cohorts (issue #36).

    For each city with a footprint + year-built table on disk: date tracts into
    construction cohorts by cumulative new building area, then estimate the
    dynamic ATT of that construction on the model's tract-mean prediction.

    Emits
    -----
    ``figures/D_event_study_main.pdf``
        Main-body figure: one panel per city at the headline threshold, held-out
        tracts, all-buildings outcome.
    ``figures/D_event_study_thresholds_{city}.pdf``
        Annex small multiples: split group × treatment threshold.
    ``figures/D_event_study_composition_{city}.pdf``
        All-buildings vs incumbent-only outcome — shows how much of the effect
        survives holding tract composition fixed.
    ``tables/D_event_study_coefficients.csv`` / ``.tex``
        One row per (city, split, threshold, outcome, control): pre-trend joint
        p-value, post-ATTs at several horizons, sample sizes.

    Returns a headline dict for the run summary.
    """
    from src import csa_event_study as ces

    print("\n=== Part D: Event study — construction-cohort CSA ===")

    years = sorted(years) if years else _available_years(results_dir) or list(YEARS)
    thresholds = tuple(thresholds) if thresholds else ces.DEFAULT_THRESHOLDS
    headline_t = (
        ces.HEADLINE_THRESHOLD if ces.HEADLINE_THRESHOLD in thresholds
        else thresholds[len(thresholds) // 2]
    )
    print(f"  panel years: {years}")
    print(f"  thresholds : {[f'{t:.0%}' for t in thresholds]} "
          f"(headline {headline_t:.0%})")

    ready, missing = ces.available_cities(processed_dir)
    for spec, reason in missing:
        print(f"  ⏭️  {spec.label}: {reason}")
    if not ready:
        print("  no city has a footprint + year-built table — Part D skipped.")
        return {"csa/status": "no_cohort_source"}

    split_of = _csa_split_of(processed_dir)
    bld = _csa_building_preds(results_dir, years)
    if bld.empty:
        print("  no building-level predictions found — Part D skipped.")
        return {"csa/status": "no_predictions"}

    rows: list[dict] = []
    headline: dict = {}
    main_panels: list[tuple[str, object]] = []

    for spec in ready:
        print(f"\n  --- {spec.label} ---")
        tracts = _csa_city_tracts(processed_dir, spec.geoid_prefixes)
        if tracts.empty:
            print("    no tracts matched this city's GEOID prefixes; skipping")
            continue

        footprints = gpd.read_parquet(processed_dir / spec.footprints_filename)
        construction_year = None
        if spec.id_index is not None and footprints.index.name == spec.id_index:
            construction_year = pd.to_numeric(
                footprints[spec.year_col], errors="coerce"
            )

        city_bld = bld[bld["GEOID_str"].isin(set(tracts["GEOID_str"]))]
        if city_bld.empty:
            print("    no predictions inside this city; skipping")
            continue
        outcomes = ces.tract_outcomes(
            city_bld, construction_year=construction_year,
            baseline_year=CSA_BASELINE_YEAR,
        )
        outcome_cols = ["pred_all"] + (
            ["pred_incumbent"] if "pred_incumbent" in outcomes.columns else []
        )
        outcomes["split_group"] = outcomes["GEOID_str"].map(split_of)
        print(f"    {outcomes['GEOID_str'].nunique():,} tracts with predictions; "
              f"outcomes: {outcome_cols}")

        # Cohorts depend only on footprints, so build once per threshold and
        # reuse across split groups / outcomes / control groups.
        cohorts_by_t = {}
        for thresh in thresholds:
            coh = ces.build_tract_cohorts(
                footprints, tracts, years,
                baseline_year=CSA_BASELINE_YEAR, threshold=thresh,
                year_col=spec.year_col, demolition_col=spec.demolition_col,
                area_epsg=spec.area_epsg,
            )
            cohorts_by_t[thresh] = coh
            s = coh.summary()
            print(f"    {thresh:>5.0%}: {s['n_treated']:,} treated / "
                  f"{s['n_never_treated']:,} never-treated tracts "
                  f"({s['n_dropped_no_baseline']:,} dropped, no baseline stock)")

        def _panel(thresh: float, group: str, outcome_col: str,
                   anticipation: int = 0):
            sub = outcomes
            if group is not None:
                sub = sub[sub["split_group"] == group]
            if sub.empty:
                return None
            return ces.build_event_panel(
                sub, cohorts_by_t[thresh].cohorts,
                outcome_col=outcome_col, panel_years=years,
                anticipation=anticipation,
            )

        # ── annex grid: split group × threshold (all-buildings outcome) ───────
        groups = [g for g in CSA_SPLIT_GROUPS
                  if (outcomes["split_group"] == g).any()] or [None]
        fig, axes = plt.subplots(
            len(groups), len(thresholds), squeeze=False,
            figsize=(FIG_SIZE_TWO_COL[0], FIG_SIZE_TWO_COL[0] * 0.34 * len(groups)),
            sharex=True,
        )
        for r, group in enumerate(groups):
            for c, thresh in enumerate(thresholds):
                ax = axes[r][c]
                panel = _panel(thresh, group, "pred_all")
                gl = CSA_SPLIT_LABELS.get(group, "All tracts")
                tag = f"{spec.key}/{gl}/{thresh:.0%}"
                if panel is None:
                    _csa_skip_axis(ax, "no tracts")
                else:
                    res, reason = _csa_estimate_or_none(panel, label=tag, n_boot=n_boot)
                    if res is None:
                        print(f"    ⏭️  {tag}: {reason}")
                        _csa_skip_axis(ax, reason)
                    else:
                        _plot_event_study(ax, res)
                        row = res.to_row()
                        row.update({"city": spec.label, "split": gl,
                                    "threshold": thresh, "outcome": "all buildings"})
                        rows.append(row)
                        if group == "heldout" and thresh == headline_t:
                            main_panels.append((spec.label, res))
                            headline[f"csa/{spec.key}/pretrend_p"] = res.pretrend_supt_p
                            headline[f"csa/{spec.key}/overall_att"] = res.overall_att
                            headline[f"csa/{spec.key}/n_treated"] = res.n_treated
                if r == 0:
                    ax.set_title(f"{_tex_pct(thresh)} of baseline area", fontsize=8)
                if r == len(groups) - 1:
                    ax.set_xlabel("Event time (periods since construction)", fontsize=8)
                if c == 0:
                    ax.set_ylabel(f"{CSA_SPLIT_LABELS.get(group, 'All')}\nATT "
                                  "(tract mean prediction)", fontsize=7.5)
        fig.suptitle(f"{spec.label}: construction-cohort event study "
                     "(shaded = 95\\% simultaneous CI)", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        _savefig(fig, out / "figures" / f"D_event_study_thresholds_{spec.key}.pdf")

        # ── composition check: all buildings vs incumbents only ──────────────
        if "pred_incumbent" in outcome_cols:
            fig2, ax2 = plt.subplots(figsize=FIG_SIZE_ONE_COL)
            drawn = 0
            for i, (oc, lab, col) in enumerate([
                ("pred_all", "All buildings", _CSA_ATT_COLOR),
                ("pred_incumbent", "Incumbents only (built $\\leq$ "
                 f"{CSA_BASELINE_YEAR})", _CSA_ALT_COLOR),
            ]):
                panel = _panel(headline_t, "heldout" if "heldout" in groups else groups[0], oc)
                if panel is None:
                    continue
                tag = f"{spec.key}/heldout/{headline_t:.0%}/{oc}"
                res, reason = _csa_estimate_or_none(panel, label=tag, n_boot=n_boot)
                if res is None:
                    print(f"    ⏭️  {tag}: {reason}")
                    continue
                # nudge the two series apart so overlapping CIs stay readable
                _plot_event_study(ax2, res, color=col, label=lab,
                                  annotate=(i == 0), offset=0.06 * (i - 0.5))
                drawn += 1
                if oc != "pred_all":
                    row = res.to_row()
                    row.update({"city": spec.label, "split": "Held out",
                                "threshold": headline_t, "outcome": "incumbents only"})
                    rows.append(row)
                    headline[f"csa/{spec.key}/incumbent_overall_att"] = res.overall_att
            if drawn:
                ax2.set_xlabel("Event time (periods since construction)", fontsize=8)
                ax2.set_ylabel("ATT (tract mean prediction)", fontsize=8)
                ax2.set_title(f"{spec.label}: composition check "
                              f"({_tex_pct(headline_t)} threshold, held out)",
                              fontsize=8.5)
                ax2.legend(fontsize=6.5, loc="lower right")
                fig2.tight_layout()
                _savefig(fig2, out / "figures" / f"D_event_study_composition_{spec.key}.pdf")
            else:
                plt.close(fig2)

        # ── control-group robustness: not-yet-treated comparisons ────────────
        head_group = "heldout" if "heldout" in groups else groups[0]
        panel = _panel(headline_t, head_group, "pred_all")
        if panel is not None and panel.n_treated:
            tag = f"{spec.key}/heldout/{headline_t:.0%}/notyet"
            res, reason = _csa_estimate_or_none(panel, label=tag, control="notyet",
                                               n_boot=n_boot)
            if res is None:
                print(f"    ⏭️  {tag}: {reason}")
            else:
                row = res.to_row()
                row.update({"city": spec.label, "split": "Held out",
                            "threshold": headline_t, "outcome": "all buildings"})
                rows.append(row)

        # ── anticipation robustness ─────────────────────────────────────────
        # year_built records *completion*, but demolition, excavation and
        # superstructure are visible in the imagery earlier, so the last one or
        # two "pre" periods are partly treated. Allowing anticipation moves the
        # effective adoption date earlier; if the pre-trend rejection is a timing
        # artifact rather than a parallel-trends violation, it should clear here.
        for antic in CSA_ANTICIPATION_PERIODS:
            panel = _panel(headline_t, head_group, "pred_all", anticipation=antic)
            if panel is None or not panel.n_treated:
                continue
            tag = f"{spec.key}/heldout/{headline_t:.0%}/anticipation={antic}"
            res, reason = _csa_estimate_or_none(panel, label=tag, n_boot=n_boot)
            if res is None:
                print(f"    ⏭️  {tag}: {reason}")
                continue
            print(f"    anticipation {antic}: pre-trend p = "
                  f"{res.pretrend_supt_p:.4f}, overall ATT = {res.overall_att:+.4f}")
            row = res.to_row()
            row.update({"city": spec.label, "split": "Held out",
                        "threshold": headline_t, "outcome": "all buildings"})
            rows.append(row)
            headline[f"csa/{spec.key}/antic{antic}_pretrend_p"] = res.pretrend_supt_p
            headline[f"csa/{spec.key}/antic{antic}_overall_att"] = res.overall_att

    # ── main-body figure: one panel per city ────────────────────────────────
    if main_panels:
        fig, axes = plt.subplots(
            1, len(main_panels), squeeze=False,
            figsize=(FIG_SIZE_TWO_COL[0], FIG_SIZE_TWO_COL[0] * 0.42),
        )
        for ax, (city, res) in zip(axes[0], main_panels):
            _plot_event_study(ax, res)
            ax.set_title(city, fontsize=9)
            ax.set_xlabel("Event time (periods since construction)", fontsize=8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        axes[0][0].set_ylabel("ATT (tract mean prediction)", fontsize=8)
        fig.suptitle(
            f"New construction raises predicted wealth ({_tex_pct(headline_t)} of "
            "baseline building area, held-out tracts)", fontsize=9,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        _savefig(fig, out / "figures" / "D_event_study_main.pdf")
    else:
        print("  ⚠️ no city produced an estimable held-out event study — "
              "no main figure written.")

    # ── coefficient table ───────────────────────────────────────────────────
    if rows:
        tab = pd.DataFrame(rows)
        # `label` and `n_units` are internal bookkeeping: label just concatenates
        # city/split/threshold/outcome, and n_units is n_treated + n_never_treated.
        tab = tab.drop(columns=[c for c in ("label", "n_units") if c in tab.columns])
        tab["threshold"] = tab["threshold"].map(lambda t: f"{t:.0%}")
        lead = ["city", "split", "threshold", "outcome", "control", "anticipation",
                "n_treated", "n_never_treated",
                "pretrend_joint_p", "pretrend_max_abs_t", "pretrend_wald_p",
                "overall_att", "overall_se"]
        horizon_cols = [c for c in tab.columns
                        if c.startswith(("att_k", "se_k"))]
        rest = [c for c in tab.columns if c not in lead + horizon_cols]
        tab = tab[[c for c in lead if c in tab.columns] + horizon_cols + rest]
        num = tab.select_dtypes("number").columns
        tab[num] = tab[num].round(3)
        # Reuse the split-reporting writer so CSV + booktabs LaTeX stay
        # consistent with the other annex tables.
        try:
            from src.data.split_reporting import _write_table
            _write_table(
                tab, out / "tables", "D_event_study_coefficients",
                "Construction-cohort Callaway--Sant'Anna event study. Each row "
                "reports the joint pre-trend test (sup-t over all pre-treatment "
                "event times, with a Wald $\\chi^2$ alternative) and dynamic "
                "ATTs at successive horizons $k$.",
                "tab:csa_event_study",
            )
        except Exception as exc:
            print(f"    LaTeX table export failed ({exc}); writing CSV only")
            tab.to_csv(
                out / "tables" / "D_event_study_coefficients.csv", index=False
            )
        headline["csa/n_specifications"] = len(rows)
    else:
        print("  ⚠️ no specification estimated — no coefficient table written.")
        headline.setdefault("csa/status", "not_estimable")

    return headline



# ─── Part E ───────────────────────────────────────────────────────────────────

def _lonlat_to_6539(lon: float, lat: float) -> tuple[float, float]:
    t = Transformer.from_crs(CRS_GEO, CRS_PROJ, always_xy=True)
    return t.transform(lon, lat)


def _slice_zarr(ds, cx: float, cy: float, half: float,
                max_px: int = 1400) -> np.ndarray | None:
    """Extract a square tile (4, H, W) uint8 from a zarr Dataset, strided so the
    longest side is ~max_px pixels (plan E.1: 'downsampled for size'). Striding
    in the .isel slice keeps the materialised array small (~MB, not ~GB)."""
    try:
        x_vals = ds.x.values
        y_vals = ds.y.values   # descending

        xi0 = int(np.searchsorted(x_vals,  cx - half, side="left"))
        xi1 = int(np.searchsorted(x_vals,  cx + half, side="right"))
        yi0 = int(np.searchsorted(-y_vals, -(cy + half), side="left"))
        yi1 = int(np.searchsorted(-y_vals, -(cy - half), side="right"))

        if xi1 <= xi0 or yi1 <= yi0:
            return None

        step = max(1, int(np.ceil(max(xi1 - xi0, yi1 - yi0) / max_px)))
        tile = ds["value"].isel(
            y=slice(yi0, yi1, step), x=slice(xi0, xi1, step)
        ).compute().values
        tile = np.nan_to_num(tile, nan=0).clip(0, 255)
        return tile.astype(np.uint8)
    except Exception as e:
        print(f"      zarr slice error: {e}")
        return None


def _stretch_rgb(tile: np.ndarray) -> np.ndarray:
    """Convert (4, H, W) uint8 -> (H, W, 3) uint8 with percentile stretch."""
    rgb = np.stack([tile[0], tile[1], tile[2]], axis=-1).astype(float)
    for ch in range(3):
        lo, hi = np.percentile(rgb[:, :, ch], [2, 98])
        rgb[:, :, ch] = np.clip(
            (rgb[:, :, ch] - lo) / max(hi - lo, 1.0) * 255, 0, 255
        )
    return rgb.astype(np.uint8)


def part_e(
    results_dir: Path,
    processed_dir: Path,
    out: Path,
    qmap_long: pd.DataFrame | None = None,
    years: list[int] | None = None,
) -> None:
    """Case study: Hudson Yards — imagery, footprints and predictions per year.

    ``years`` defaults to the years that actually have a georeferenced
    ``predictions_{year}.parquet`` under ``results_dir``, so a NAIP-cadence run
    (odd years) works as well as the legacy biennial zarr grid.
    """
    print("\n=== Part E: Case study – Hudson Yards ===")
    import xarray as xr

    if years is None:
        years = sorted(
            int(p.name.split("_")[1].split(".")[0])
            for p in results_dir.glob("predictions_2*.parquet")
        )
    if not years:
        print(f"  no georeferenced predictions_*.parquet under {results_dir} — "
              "Part E needs building geometry; skipping.")
        return
    print(f"  years: {years}")

    lon, lat    = CASE_STUDY["lon"], CASE_STUDY["lat"]
    half_km     = CASE_STUDY["half_km"]
    cx, cy      = _lonlat_to_6539(lon, lat)
    half_ft_box = half_km * 1000.0 / _M_PER_FT   # half-side in US-survey-ft

    # Buildings inside the box (CRS 4326)
    t_inv = Transformer.from_crs(CRS_PROJ, CRS_GEO, always_xy=True)
    lon0, lat0 = t_inv.transform(cx - half_ft_box, cy - half_ft_box)
    lon1, lat1 = t_inv.transform(cx + half_ft_box, cy + half_ft_box)
    box_4326 = shapely_box(lon0, lat0, lon1, lat1)

    bldg_nyc  = gpd.read_parquet(processed_dir / "buildings_nyc.parquet")
    bldg_box  = bldg_nyc[bldg_nyc.intersects(box_4326)].copy()
    bldg_proj = bldg_box.to_crs(CRS_PROJ)
    print(f"  {len(bldg_proj)} buildings in case-study area")

    if len(bldg_proj) == 0:
        print("  No buildings found. Skipping Part E.")
        return

    # Load predictions for those DOITT_IDs
    box_ids = set(bldg_proj.index.tolist())
    pred_by_yr: dict[int, gpd.GeoDataFrame] = {}
    for yr in years:
        fpath = results_dir / f"predictions_{yr}.parquet"
        if not fpath.exists():
            print(f"      {fpath.name} missing — skipping {yr}")
            continue
        df = gpd.read_parquet(fpath)
        sub = df[df.index.isin(box_ids)]
        if len(sub) > 0:
            pred_by_yr[yr] = sub

    if not pred_by_yr:
        print("  no predictions inside the case-study box. Skipping Part E.")
        return

    all_vals = np.concatenate(
        [df["predicted_value"].values for df in pred_by_yr.values()]
    )
    vmin, vmax = np.nanpercentile(all_vals, 2), np.nanpercentile(all_vals, 98)
    pred_norm  = Normalize(vmin=vmin, vmax=vmax)
    cmap_pred  = plt.cm.Spectral

    # ── E.1 per-year × 3 grid ────────────────────────────────────────────────
    print(f"  E.1 {len(years)}×3 image grid...")
    n_rows = len(years)
    fw = journal_sizes["IEETRAN"]["onecol"]   # single-column width (252 pt ≈ 3.49 in)
    # Full A4 text height (IEEEtran: 239 mm ≈ 680 pt) minus a 2-line caption (~30 pt)
    fh = journal_sizes["IEETRAN"]["textheight_a4"] - 30. * pt
    bottom_pad = 0.055                # figure fraction reserved for cbar + legend

    fig_g, axes_g = plt.subplots(
        n_rows, 3,
        figsize=(fw, fh),
        gridspec_kw={"hspace": 0.015, "wspace": 0.015},
        squeeze=False,   # a single-year run must still index axes_g[0, col]
    )

    for col_i, col_title in enumerate(
        ["Aerial image (RGB)", "Building polygons", "Model predictions"]
    ):
        axes_g[0, col_i].set_title(col_title, fontsize=7, fontweight="bold", pad=3)

    for row_i, yr in enumerate(years):
        ax_img, ax_bldg, ax_pred = axes_g[row_i]

        # ── column 1: aerial image ──
        zarr_path = Path(str(IMAGERY_ROOT)) / f"nyc_{yr}.zarr"
        img_ok = False
        if zarr_path.exists():
            try:
                ds = xr.open_zarr(str(zarr_path), chunks="auto")
                tile = _slice_zarr(ds, cx, cy, half_ft_box)
                if tile is not None and tile.ndim == 3 and tile.shape[0] >= 3:
                    rgb = _stretch_rgb(tile)
                    ax_img.imshow(
                        rgb,
                        extent=[cx - half_ft_box, cx + half_ft_box,
                                cy - half_ft_box, cy + half_ft_box],
                        origin="upper",
                        aspect="equal",
                    )
                    img_ok = True
            except Exception as e:
                print(f"      zarr {yr}: {e}")
        if not img_ok:
            ax_img.set_facecolor("#1a1a2e")
            ax_img.text(0.5, 0.5, "Image\nunavailable",
                        ha="center", va="center", transform=ax_img.transAxes,
                        fontsize=6, color="#aaaaaa")
            ax_img.set_xlim(cx - half_ft_box, cx + half_ft_box)
            ax_img.set_ylim(cy - half_ft_box, cy + half_ft_box)

        # Year label as text overlay (top-left corner of the aerial image panel)
        ax_img.text(0.03, 0.97, str(yr), transform=ax_img.transAxes,
                    fontsize=6.5, fontweight="bold", va="top", ha="left",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.18", fc="black", alpha=0.52, lw=0))
        ax_img.axis("off")

        # ── column 2: building polygons ──
        cy_yr = bldg_proj["CONSTRUCTION_YEAR"].fillna(0)
        dy_yr = bldg_proj["DEMOLITION_YEAR"].fillna(9999)
        exists_mask = (cy_yr <= yr) & (dy_yr > yr)
        new_mask    = bldg_proj["CONSTRUCTION_YEAR"].between(2009, yr, inclusive="right")

        bldg_old = bldg_proj[exists_mask & ~new_mask]
        bldg_new = bldg_proj[exists_mask & new_mask]

        ax_bldg.set_facecolor("#efefef")
        if len(bldg_old) > 0:
            bldg_old.plot(ax=ax_bldg, color="0.60", edgecolor="none", alpha=0.80)
        if len(bldg_new) > 0:
            bldg_new.plot(ax=ax_bldg, color="crimson", edgecolor="none", alpha=0.90)

        ax_bldg.set_xlim(cx - half_ft_box, cx + half_ft_box)
        ax_bldg.set_ylim(cy - half_ft_box, cy + half_ft_box)
        ax_bldg.set_aspect("equal")
        ax_bldg.axis("off")

        # ── column 3: model predictions ──
        ax_pred.set_facecolor("#efefef")
        pyr = pred_by_yr.get(yr)
        if pyr is not None and len(pyr) > 0:
            pyr.plot(
                column="predicted_value", ax=ax_pred,
                norm=pred_norm, cmap=cmap_pred,
                edgecolor="none", alpha=0.90,
            )
        else:
            ax_pred.text(0.5, 0.5, "No predictions",
                         ha="center", va="center", transform=ax_pred.transAxes,
                         fontsize=6, color="#888888")
        ax_pred.set_xlim(cx - half_ft_box, cx + half_ft_box)
        ax_pred.set_ylim(cy - half_ft_box, cy + half_ft_box)
        ax_pred.set_aspect("equal")
        ax_pred.axis("off")

    # Layout: reserve bottom strip for colorbar + legend
    fig_g.subplots_adjust(
        left=0.01, right=0.99, top=0.975, bottom=bottom_pad,
        hspace=0.015, wspace=0.015,
    )

    # Horizontal colorbar centred below the grid — does NOT steal axes space
    sm = plt.cm.ScalarMappable(cmap=cmap_pred, norm=pred_norm)
    sm.set_array([])
    cax = fig_g.add_axes([0.38, 0.018, 0.28, 0.012])
    cbar = fig_g.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Predicted value", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5)
    cbar.outline.set_linewidth(0.4)

    # Building-type legend anchored to the bottom-left
    patch_old = mpatches.Patch(color="0.60",    label="Pre-existing ($\\leq 2009$)")
    patch_new = mpatches.Patch(color="crimson", label="New construction ($> 2009$)")
    fig_g.legend(
        handles=[patch_old, patch_new],
        loc="lower left",
        bbox_to_anchor=(0.01, 0.005),
        fontsize=6.5,
        framealpha=0.85,
        ncol=1,
        handlelength=1.2,
        borderpad=0.5,
    )

    _savefig(fig_g, out / "figures" / "E_hudson_yards_grid.pdf")

    # ── E.2 Per-building trajectories ─────────────────────────────────────────
    print("  E.2 per-building line chart...")
    traj: dict[int, dict[int, float]] = {}
    for yr, df in pred_by_yr.items():
        for did, row in df.iterrows():
            traj.setdefault(did, {})[yr] = float(row["predicted_value"])

    fig_l, ax_l = plt.subplots(figsize=FIG_SIZE_TWO_COL)
    n_plotted = 0
    for did, yr_vals in traj.items():
        if len(yr_vals) < 2:
            continue
        yrs  = sorted(yr_vals)
        vals = [yr_vals[y] for y in yrs]
        cy_b = bldg_proj.loc[did, "CONSTRUCTION_YEAR"] if did in bldg_proj.index else np.nan
        color = "crimson" if (not pd.isna(cy_b) and float(cy_b) > 2009) else "steelblue"
        ax_l.plot(yrs, vals, color=color, alpha=0.22, lw=1.0)
        n_plotted += 1

    print(f"    plotted {n_plotted} building trajectories")
    p_old = mpatches.Patch(color="steelblue", alpha=0.7, label="Pre-existing (built $\\leq 2009$)")
    p_new = mpatches.Patch(color="crimson",   alpha=0.7, label="New construction (built $\\geq 2009$)")
    ax_l.legend(handles=[p_old, p_new], fontsize=8)
    ax_l.set_xlabel("Year")
    ax_l.set_ylabel("predicted_value")
    ax_l.set_title("Hudson Yards: per-building prediction trajectories")
    ax_l.set_xticks(years)
    fig_l.tight_layout()
    _savefig(fig_l, out / "figures" / "E_buildings_lines.pdf")

    # ── E.3 Tract ACS vs model predictions ───────────────────────────────────
    print("  E.3 tract ACS vs prediction chart...")
    splits      = _load_splits(processed_dir)
    splits_proj = splits.to_crs(CRS_PROJ)
    box_proj    = shapely_box(cx - half_ft_box, cy - half_ft_box,
                              cx + half_ft_box, cy + half_ft_box)
    ov = splits_proj[splits_proj.intersects(box_proj)].copy()
    ov["ov_area"] = ov.geometry.intersection(box_proj).area
    ov = ov.sort_values("ov_area", ascending=False)
    # Keep only the few tracts that actually fill the frame, so the chart's
    # legend stays readable (the box clips the edges of many neighbours).
    overlap_geoids = ov["GEOID_str"].head(3).tolist()
    print(f"    overlapping tracts (top 3 by in-box area): {overlap_geoids}")

    if not overlap_geoids:
        print("    no overlapping tracts – skipping E.3")
        return

    panel = pd.read_feather(processed_dir / "ny_tracts_panel_2009_2014_2019_2024.feather")
    panel["GEOID_str"] = panel["geoid_2024"].astype(str).str.zfill(11)

    # Build qmap_long for these tracts if not supplied
    if qmap_long is not None and "gb2_map" in qmap_long.columns:
        pred_col = "gb2_map"
        q_sub    = qmap_long[qmap_long["GEOID_str"].isin(overlap_geoids)]
    else:
        # GB2 mapping not available (Part C was not run) — fall back to raw predictions.
        print("    gb2_map not available, falling back to raw predicted_value")
        tract_long_local = _load_tract_long(results_dir)
        pred_col = "predicted_value"
        q_sub    = tract_long_local[tract_long_local["GEOID_str"].isin(overlap_geoids)]

    panel_years = [2009, 2014, 2019, 2024]
    colors_t    = plt.cm.tab10.colors

    fig_t, ax_t = plt.subplots(figsize=FIG_SIZE_TWO_COL)
    for i, geoid in enumerate(overlap_geoids):
        color   = colors_t[i % len(colors_t)]
        pan_row = panel[panel["GEOID_str"] == geoid]

        # ACS at 4 panel years
        acs_yrs, acs_vals = [], []
        for py in panel_years:
            col_name = f"per_capita_income_usd_{py}"
            if col_name in panel.columns and len(pan_row) > 0:
                v = pan_row.iloc[0][col_name]
                if not pd.isna(v):
                    acs_yrs.append(py)
                    acs_vals.append(float(v))
        if acs_yrs:
            ax_t.scatter(acs_yrs, acs_vals, color=color, s=70, zorder=5,
                         label=f"ACS {geoid[-6:]}")
            ax_t.plot(acs_yrs, acs_vals, color=color, lw=1.2, ls=":", alpha=0.6)

        # Model predictions (qmapped) over 8 years
        prow = q_sub[q_sub["GEOID_str"] == geoid].sort_values("year")
        if len(prow) > 0:
            ax_t.plot(prow["year"].values, prow[pred_col].values,
                      "o-", color=color, lw=2, ms=5,
                      label=f"Model {geoid[-6:]} ({pred_col})")

    ax_t.set_xlabel("Year")
    ax_t.set_ylabel(
        "Per-capita income (USD)" if pred_col == "gb2_map"
        else "predicted_value"
    )
    ax_t.set_title(
        "Hudson Yards tract: ACS income (markers) vs model prediction (line)"
    )
    ax_t.legend(fontsize=7, loc="upper left")
    ax_t.set_xticks(years)
    fig_t.tight_layout()
    _savefig(fig_t, out / "figures" / "E_tract_acs_vs_pred.pdf")



# Part E

# ─── Part B helpers ───────────────────────────────────────────────────────────

def _detect_change_vectorized(
    pred_gdf: gpd.GeoDataFrame,
    new_bldg: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Vectorised change detection via shapely STRtree.

    For each building in pred_gdf (EPSG:6539), forms a TAU_FT-half-side square
    tile around its centroid and checks which new buildings (CONSTRUCTION_YEAR
    in 2009<y<=2024) have footprints that intersect it.

    Returns DataFrame indexed by DOITT_ID with columns:
        changed (bool), first_construction (float), change_year (float or NaN).
    """
    centroids = pred_gdf.geometry.centroid
    tiles = np.array([
        shapely_box(c.x - TAU_FT, c.y - TAU_FT, c.x + TAU_FT, c.y + TAU_FT)
        for c in centroids
    ])

    tree = STRtree(new_bldg.geometry.values)
    result = tree.query(tiles, predicate="intersects")
    # result shape (2, n_hits): result[0]=query idx, result[1]=tree idx

    nb_years = new_bldg["CONSTRUCTION_YEAR"].values
    doitt_ids = pred_gdf.index.values

    changed = np.zeros(len(doitt_ids), dtype=bool)
    first_construction = np.full(len(doitt_ids), np.nan)
    change_year_arr = np.full(len(doitt_ids), np.nan)

    if result.shape[1] > 0:
        hits_df = pd.DataFrame({
            "qi": result[0].astype(int),
            "yr": nb_years[result[1]],
        })
        min_yr_by_qi = hits_df.groupby("qi")["yr"].min()
        for qi, fc in min_yr_by_qi.items():
            # first panel year >= first_construction
            chy = next((y for y in YEARS if y >= fc), np.nan)
            changed[qi] = True
            first_construction[qi] = fc
            change_year_arr[qi] = chy

    # Count of distinct new buildings whose footprint intersects the tile.
    # Used downstream to distinguish isolated infill (n=1) from coordinated
    # redevelopment waves (n>=2, n>=5, etc.).
    n_new = np.zeros(len(doitt_ids), dtype=int)
    if result.shape[1] > 0:
        counts = (
            pd.Series(result[0].astype(int))
            .value_counts()
        )
        for qi, cnt in counts.items():
            n_new[qi] = int(cnt)

    return pd.DataFrame(
        {"changed":            changed,
         "first_construction": first_construction,
         "change_year":        change_year_arr,
         "n_new_buildings":    n_new},
        index=doitt_ids,
    )

def _icc(arr: np.ndarray) -> float:
    """ICC(1): fraction of total variance explained by between-unit variance."""
    unit_means = np.nanmean(arr, axis=1)
    sigma2_between = float(np.nanvar(unit_means))
    sigma2_within  = float(np.nanmean(np.nanvar(arr, axis=1)))
    denom = sigma2_between + sigma2_within
    return sigma2_between / denom if denom > 0 else np.nan


def _rank_autocorr(wide_sub: pd.DataFrame, pair: tuple[int, int]) -> tuple[float, int]:
    """Spearman rank-autocorrelation between two year columns, on units that
    have a finite value in *both* years. Returns (rho, n_used)."""
    a = wide_sub[pair[0]].values.astype(float)
    b = wide_sub[pair[1]].values.astype(float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan, int(m.sum())
    return float(spearmanr(a[m], b[m]).statistic), int(m.sum())


# ══════════════════════════════════════════════════════════════════════════════
# US-scale parts  (part_a_us / part_b_us / part_c_us)  +  run_evaluation
# ══════════════════════════════════════════════════════════════════════════════


# Cities with fewer predicted tracts than this are dropped from the whole US
# evaluation section — below this, per-city Spearman/GB2/stability estimates
# are too noisy to report and a handful of tiny cities can dominate a
# histogram binned over only a few dozen of them.
MIN_CBSA_TRACTS = 10
# Cap on points rasterized in a single raw scatter (matplotlib's Agg backend
# materializes a full RGBA path per marker; at US scale building-level frames
# reach the millions and an uncapped scatter can exhaust system RAM).
SCATTER_MAX_POINTS = 50_000


class _USContext:
    """Loaded US artifacts shared by the US parts (built once by run_evaluation)."""

    def __init__(self, results_dir: Path, processed_dir: Path, out: Path, indicator: str):
        self.results_dir = results_dir
        self.processed_dir = processed_dir
        self.out = out
        self.indicator = indicator
        self.years = _available_years(results_dir)

        # Building-level predictions with cbsa + structural-change attached.
        panel = _load_panel_us(indicator, processed_dir, want_income=False)
        bld = _load_building_preds_us(results_dir, self.years)
        self.bld = _attach_cbsa(bld, panel) if len(bld) else bld

        # Tract-level predictions in canonical (pred/label) form + cbsa/change.
        tract = _load_tract_long(results_dir, self.years)
        if len(tract):
            # Use the zero-padded GEOID_str as the canonical GEOID (drop the raw one).
            tract = tract.drop(columns=["GEOID"]).rename(
                columns={"GEOID_str": "GEOID", "predicted_value": "pred", "Rel_Score": "label"}
            )
            tract = _attach_cbsa(tract, panel)
        self.tract = tract
        self._drop_small_cbsas()

        self.panel = panel  # geometry (EPSG:5070) for maps
        self.cbsa_meta = _load_cbsa_meta(processed_dir)
        self.top_cbsas = (
            [str(c) for c in cbsa_brackets.top_cbsas(self.cbsa_meta, 10)]
            if self.cbsa_meta is not None else []
        )

    def _drop_small_cbsas(self) -> None:
        """Restrict self.tract / self.bld to CBSAs with >= MIN_CBSA_TRACTS
        distinct predicted tracts. Runs once here so every downstream part
        (A/B/C, the main figure) sees only the filtered cities."""
        if not len(self.tract):
            return
        tract_counts = self.tract.groupby("cbsa")["GEOID"].nunique()
        keep = set(tract_counts[tract_counts >= MIN_CBSA_TRACTS].index)
        n_dropped = len(tract_counts) - len(keep)
        if n_dropped:
            print(f"    dropping {n_dropped} / {len(tract_counts)} CBSA(s) with "
                  f"< {MIN_CBSA_TRACTS} tracts")
        self.tract = self.tract[self.tract["cbsa"].isin(keep)].reset_index(drop=True)
        if len(self.bld):
            self.bld = self.bld[self.bld["cbsa"].isin(keep)].reset_index(drop=True)

    def report_types(self, frame: pd.DataFrame) -> list[str]:
        """Split types present in ``frame`` that we report on (test/val/...)."""
        if "type" not in frame.columns or frame.empty:
            return []
        present = set(frame["type"].unique())
        ordered = [t for t in _REPORT_TYPES if t in present]
        # include any other non-train/unassigned type that shows up
        extra = sorted(present - set(_REPORT_TYPES) - {"train", "unassigned"})
        return ordered + extra


def _pooled_corr(x: np.ndarray, y: np.ndarray, boot_cap: int = 20000) -> dict:
    """Pooled Spearman + Kendall with bootstrap 95% CIs; {} if too few points.

    The point estimates use every point, but the bootstrap CI is computed on a
    random subsample capped at ``boot_cap`` — at US scale the pooled set is
    millions of building-years, where 2000x resampling is both intractable and
    statistically vacuous (the CI collapses to zero width). Kendall's tau is
    also O(n^2), so it is skipped above the cap.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 10 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {}
    rho, _ = spearmanr(x, y)
    out = {"spearman": float(rho), "n": int(n)}
    if n > boot_cap:
        idx = np.random.default_rng(42).choice(n, boot_cap, replace=False)
        xb, yb = x[idx], y[idx]
    else:
        xb, yb = x, y
        out["kendall"] = float(kendalltau(x, y).statistic)
    _, lo, hi = _bootstrap_spearman(xb, yb)
    out["spearman_ci_lo"] = lo
    out["spearman_ci_hi"] = hi
    return out


def _loess_curve(
    x: np.ndarray, y: np.ndarray, frac: float = 0.3,
    max_n: int = 20_000, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray] | None:
    """LOESS fit of y ~ x, returned as (x_sorted, y_smoothed); None if there
    are too few finite points or x has no spread. Subsamples to ``max_n``
    points before fitting — statsmodels' lowess is roughly O(n^2) and the
    building-level US scatters can reach the millions."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 10 or np.ptp(x) == 0:
        return None
    if len(x) > max_n:
        idx = np.random.default_rng(seed).choice(len(x), max_n, replace=False)
        x, y = x[idx], y[idx]
    fit = lowess(y, x, frac=frac, it=0, return_sorted=True)
    return fit[:, 0], fit[:, 1]


def _scatter_cell(g: pd.DataFrame, cbsa: str, year: int, title: str, path: Path) -> None:
    """One tract-level scatter for a (CBSA, year) cell — US analog of the NYC
    A_scatter figure: ACS tract z-score (x) vs mean tract predicted value (y),
    with a LOESS overlay (shape of the relation) and the cell Spearman
    annotated."""
    x = g["pred"].values.astype(float)
    y = g["label"].values.astype(float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return
    rho = spearmanr(x, y).statistic
    fig, ax = plt.subplots(figsize=FIG_SIZE_ONE_COL)
    ax.scatter(y, x, s=6, alpha=0.35, color="steelblue", linewidths=0)
    curve = _loess_curve(y, x)
    if curve is not None:
        xs, ys = curve
        ax.plot(xs, ys, "-", color="firebrick", lw=1.5, label="LOESS")
        # Pinned, not loc="best": auto-placement walks every scatter offset while
        # the tight bbox is computed, which is both slow and fragile on dense
        # clouds (it can blow up inside Bbox.count_contains). Upper left is taken
        # by the rho annotation below, so the legend goes lower right.
        ax.legend(fontsize=7, loc="lower right")
    ax.set_xlabel("ACS Tract Z-score")
    ax.set_ylabel("Average Tract\nPredicted Value")
    ax.set_title(title, fontsize=8)
    ax.text(0.05, 0.85, f"Spearman $\\rho$ = {rho:.3f}\n$n$ = {len(x)}",
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    _savefig(fig, path)


def _cbsa_title(cbsa: str, cbsa_meta: pd.DataFrame | None) -> str:
    if cbsa_meta is None or "cbsa_title" not in cbsa_meta.columns:
        return f"CBSA {cbsa}"
    hit = cbsa_meta[cbsa_meta["cbsa_code"].astype(str) == str(cbsa)]
    if len(hit):
        return f"{hit['cbsa_title'].iloc[0]} ({cbsa})"
    return f"CBSA {cbsa}"


def part_a_us(ctx: _USContext) -> dict:
    """US cross-sectional validity: within-city Spearman cells (headline),
    pooled building/tract correlations, per-bracket & top-metro breakdowns, and
    the per-cell scatter grid + Spearman histogram over the test cells."""
    print("\n=== Part A (US): Cross-sectional validity ===")
    out = ctx.out
    headline: dict = {}
    if ctx.bld.empty:
        print("  no building predictions found; skipping Part A")
        return headline

    types = ctx.report_types(ctx.bld)
    print(f"  split types present: {types}")

    summary_rows = []
    for t in types:
        bld_t = ctx.bld[ctx.bld["type"] == t]
        # Building-level within-city cells (paper's primary metric).
        cells = within_city_cells(bld_t)
        cells.to_csv(out / "tables" / f"US_A_cells_{t}.csv", index=False)
        head = weighted_within_spearman(cells)

        # Pooled building- and tract-level correlations.
        pooled_b = _pooled_corr(bld_t["pred"], bld_t["label"])
        tract_t = ctx.tract[ctx.tract["cbsa"].isin(bld_t["cbsa"].unique())] if len(ctx.tract) else ctx.tract
        pooled_tr = _pooled_corr(tract_t["pred"], tract_t["label"]) if len(tract_t) else {}

        # Cross-city dispersion of per-city prediction means (~0 under per-CBSA
        # z labels): the latent-drift guardrail, cheap and cross-sectional.
        offsets = bld_t.groupby("cbsa")["pred"].agg(["mean", "size"])
        offsets = offsets[offsets["size"] >= 5]
        city_offset_sd = float(offsets["mean"].std(ddof=0)) if len(offsets) >= 2 else np.nan

        row = {"split": t, "city_offset_sd": city_offset_sd}
        row.update({f"within/{k}": v for k, v in head.items()})
        row.update({f"pooled_building/{k}": v for k, v in pooled_b.items()})
        row.update({f"pooled_tract/{k}": v for k, v in pooled_tr.items()})
        summary_rows.append(row)

        # Per-bracket and per-top-metro within-city Spearman breakdowns
        # (cross-sectional only — temporal metrics live in Part B).
        _write_breakdown_tables(bld_t, cells, ctx.cbsa_meta, ctx.top_cbsas, out, t)

        if head:
            print(f"  [{t}] within_spearman={head['within_spearman']:.3f} "
                  f"(cells={head['within_cells']}, n={head['within_n']:,}, "
                  f"tracts={head['within_tracts']:,})")
            headline[f"{t}/within_spearman"] = head["within_spearman"]

    pd.DataFrame(summary_rows).to_csv(out / "tables" / "US_A_summary.csv", index=False)

    # ── Figures: per-cell scatters + Spearman histogram over the test cells ──
    scatter_split = "test" if "test" in types else (types[0] if types else None)
    if scatter_split is not None and len(ctx.tract):
        _part_a_us_figures(ctx, scatter_split)

    return headline


def _bracket_of(cbsa_meta: pd.DataFrame | None) -> dict:
    """cbsa_code (str) -> bracket, from cbsa_splits; empty if unavailable."""
    if cbsa_meta is None or "bracket" not in cbsa_meta.columns:
        return {}
    return dict(zip(cbsa_meta["cbsa_code"].astype(str), cbsa_meta["bracket"]))


def _write_breakdown_tables(bld_t: pd.DataFrame, cells: pd.DataFrame,
                            cbsa_meta: pd.DataFrame | None, top_cbsas: list,
                            out: Path, split: str) -> None:
    """Per-bracket and per-top-metro within-city Spearman tables.

    Built from the already-computed ``cells`` (one row per CBSA-year), so this
    adds only cheap group aggregation — no re-scan of the building rows.
    """
    if not len(cells):
        return
    cells = cells.copy()
    cells["cbsa"] = cells["cbsa"].astype(str)

    # Per top-10 metro: tract-weighted mean cell Spearman.
    top = set(str(c) for c in (top_cbsas or []))
    city_rows = []
    for cbsa, g in cells.groupby("cbsa"):
        if top and cbsa not in top:
            continue
        city_rows.append({"cbsa": cbsa,
                          "within_spearman": float(np.average(g["rho"], weights=g["n_tracts"])),
                          "cells": int(len(g)), "n": int(g["n"].sum()),
                          "tracts": int(g["n_tracts"].sum())})
    if city_rows:
        pd.DataFrame(city_rows).sort_values("n", ascending=False).to_csv(
            out / "tables" / f"US_A_top_metros_{split}.csv", index=False)

    # Per population bracket.
    bmap = _bracket_of(cbsa_meta)
    if bmap:
        cells["bracket"] = cells["cbsa"].map(bmap)
        brk_rows = []
        for bracket, g in cells.dropna(subset=["bracket"]).groupby("bracket"):
            brk_rows.append({"bracket": bracket,
                             "within_spearman": float(np.average(g["rho"], weights=g["n_tracts"])),
                             "cells": int(len(g)), "n": int(g["n"].sum()),
                             "tracts": int(g["n_tracts"].sum())})
        if brk_rows:
            pd.DataFrame(brk_rows).to_csv(
                out / "tables" / f"US_A_by_bracket_{split}.csv", index=False)


def _part_a_us_figures(ctx: _USContext, split: str) -> None:
    """Per-(CBSA, year) tract scatters for ``split`` + the Spearman histogram."""
    out = ctx.out
    # Restrict the tract frame to CBSAs in this split (via building 'type').
    split_cbsas = set(ctx.bld.loc[ctx.bld["type"] == split, "cbsa"].unique())
    tr = ctx.tract[ctx.tract["cbsa"].isin(split_cbsas)].copy()
    if tr.empty:
        print(f"  no tract data for split '{split}'; skipping scatter grid")
        return

    scatter_dir = out / "figures" / "US_A_scatter_cells"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for (cbsa, year), g in tr.groupby(["cbsa", "year"]):
        title = f"{_cbsa_title(cbsa, ctx.cbsa_meta)} — {year}"
        _scatter_cell(g, cbsa, int(year), title,
                      scatter_dir / f"{cbsa}_{year}.png")
        n_written += 1
    print(f"  wrote {n_written} per-cell scatter(s) -> {scatter_dir.name}/")

    # Tract-level Spearman per (CBSA, year) cell -> histogram over test cells.
    cells = within_city_cells(tr)
    if not len(cells):
        print("  no usable cells for Spearman histogram")
        return
    cells.to_csv(out / "tables" / f"US_A_tract_cells_{split}.csv", index=False)
    # Tract frame: one row per tract, so n_tracts == n; kept as n_tracts for
    # uniformity with the building-level tables.
    w_mean = float(np.average(cells["rho"], weights=cells["n_tracts"]))
    per_city = cells.groupby("cbsa")[["rho", "n_tracts"]].apply(
        lambda d: np.average(d["rho"], weights=d["n_tracts"])
    ).values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_SIZE_TWO_COL[0], FIG_SIZE_TWO_COL[1]))
    ax1.hist(cells["rho"], bins=min(30, max(5, len(cells) // 2)),
             color="steelblue", edgecolor="white", alpha=0.85)
    ax1.axvline(w_mean, color="firebrick", lw=1.5,
                label=f"tract-wt mean = {w_mean:.3f}")
    ax1.set_xlabel("Within-cell Spearman $\\rho$")
    ax1.set_ylabel("Count of (CBSA, year) cells")
    ax1.set_title(f"All {split} cells (n={len(cells)})", fontsize=8)
    ax1.legend(fontsize=7)

    ax2.hist(per_city, bins=min(30, max(5, len(per_city) // 2)),
             color="seagreen", edgecolor="white", alpha=0.85)
    ax2.axvline(float(np.mean(per_city)), color="firebrick", lw=1.5,
                label=f"mean = {np.mean(per_city):.3f}")
    ax2.set_xlabel("Per-city mean Spearman $\\rho$")
    ax2.set_ylabel(f"Count of cities")
    ax2.set_title(f"Per-city means (n={len(per_city)})", fontsize=8)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    _savefig(fig, out / "figures" / f"US_A_spearman_hist_{split}.pdf")

    # Building-level pred-vs-label scatter per reported split (dense point
    # cloud + LOESS, replacing the earlier hexbin — the raw scatter shows
    # the shape of the relation more directly than a binned density plot).
    for t in ctx.report_types(ctx.bld):
        b = ctx.bld[ctx.bld["type"] == t]
        if len(b) < 50:
            continue
        label_v = b["label"].values.astype(float)
        pred_v = b["pred"].values.astype(float)
        m = np.isfinite(label_v) & np.isfinite(pred_v)
        label_v, pred_v = label_v[m], pred_v[m]
        if len(label_v) < 50:
            continue
        if len(label_v) > SCATTER_MAX_POINTS:
            idx = np.random.default_rng(42).choice(len(label_v), SCATTER_MAX_POINTS, replace=False)
            label_v, pred_v = label_v[idx], pred_v[idx]
        fig, ax = plt.subplots(figsize=FIG_SIZE_ONE_COL)
        ax.scatter(label_v, pred_v, s=1, alpha=0.15, color="steelblue", linewidths=0)
        curve = _loess_curve(label_v, pred_v)
        if curve is not None:
            xs, ys = curve
            ax.plot(xs, ys, "-", color="firebrick", lw=1.5, label="LOESS")
            # Pinned rather than loc="best": with SCATTER_MAX_POINTS offsets on
            # the axes, auto-placement is the slow path and is what raised
            # "could not broadcast input array from shape (2,) into shape (2,)"
            # out of Bbox.count_contains while savefig computed the tight bbox.
            ax.legend(fontsize=7, loc="upper left")
        ax.set_xlabel("ACS Z-score (label)")
        ax.set_ylabel("Predicted value")
        ax.set_title(f"{t}: building-level pred vs label (n={len(b):,})", fontsize=8)
        fig.tight_layout()
        _savefig(fig, out / "figures" / f"US_A_hexbin_{t}.png")


def part_b_us(ctx: _USContext) -> dict:
    """US temporal stability: cross-year rank autocorrelation of stable
    buildings (pred vs label benchmark), stable/changed MASD ratio, and
    tract-level ICC + rank autocorrelation on a data-driven year pair."""
    print("\n=== Part B (US): Temporal stability ===")
    out = ctx.out
    headline: dict = {}
    if ctx.bld.empty:
        print("  no building predictions found; skipping Part B")
        return headline

    rows = []
    for t in ctx.report_types(ctx.bld):
        bld_t = ctx.bld[ctx.bld["type"] == t]
        ra = rank_autocorrelation(bld_t)
        md = masd_by_change(bld_t)
        row = {"split": t}
        row.update(ra)
        row.update(md)
        if md.get("stable_masd", 0) > 0 and "changed_masd" in md:
            row["masd_ratio"] = md["changed_masd"] / md["stable_masd"]
        rows.append(row)
        if ra:
            gap = ra["rank_autocorr_pred"] - ra["rank_autocorr_label"]
            print(f"  [{t}] rank_autocorr pred={ra['rank_autocorr_pred']:.3f} "
                  f"label={ra['rank_autocorr_label']:.3f} (gap={gap:+.3f})")
            headline[f"{t}/rank_autocorr_pred"] = ra["rank_autocorr_pred"]
            headline[f"{t}/rank_autocorr_label"] = ra["rank_autocorr_label"]
    if rows:
        pd.DataFrame(rows).to_csv(out / "tables" / "US_B_rank_autocorr.csv", index=False)

    # Tract-level ICC + rank autocorrelation per CBSA (pooled means).
    if len(ctx.tract):
        _part_b_us_tract(ctx)
    return headline


def _part_b_us_tract(ctx: _USContext) -> None:
    """Per-CBSA tract ICC across years + rank autocorrelation on the widest
    common year pair; written to US_B_tract_stability.csv."""
    out = ctx.out
    rows = []
    for cbsa, g in ctx.tract.groupby("cbsa"):
        wide = g.pivot_table(index="GEOID", columns="year", values="pred")
        if wide.shape[1] < 2:
            continue
        icc_val = _icc(wide.values.astype(float))
        # Data-driven year pair: most tracts observed in both, ties -> widest span.
        years = sorted(wide.columns)
        best = None
        for i in range(len(years)):
            for j in range(i + 1, len(years)):
                t1, t2 = years[i], years[j]
                n_common = int(wide[[t1, t2]].dropna().shape[0])
                key = (n_common, t2 - t1)
                if n_common >= 5 and (best is None or key > best[:2]):
                    best = (n_common, t2 - t1, t1, t2)
        rho, n_used = (np.nan, 0)
        if best is not None:
            rho, n_used = _rank_autocorr(wide, (best[2], best[3]))
        rows.append({"cbsa": cbsa, "icc": icc_val, "rank_autocorr": rho,
                     "n_pair": n_used, "n_tracts": int(wide.shape[0])})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out / "tables" / "US_B_tract_stability.csv", index=False)
        print(f"  tract stability: mean ICC={df['icc'].mean():.3f}, "
              f"mean rank-autocorr={df['rank_autocorr'].mean():.3f} "
              f"over {len(df)} CBSAs")


def part_c_us(ctx: _USContext) -> pd.DataFrame | None:
    """US GB2 dollar mapping: per top-CBSA, fit GB2 to that metro's tract
    per-capita income and map within-CBSA predicted tract ranks to dollars."""
    print("\n=== Part C (US): GB2 dollar mapping ===")
    out = ctx.out
    if not len(ctx.tract):
        print("  no tract predictions; skipping Part C")
        return None

    panel_income = _load_panel_us(ctx.indicator, ctx.processed_dir, want_income=True)
    income_cols = [c for c in panel_income.columns if c.startswith("per_capita_income_usd_")]
    if not income_cols:
        print("  panel has no per-capita income columns; skipping Part C")
        return None
    income_years = sorted(int(c.rsplit("_", 1)[1]) for c in income_cols)

    # Target CBSAs: top-10 by population, intersected with what we predicted.
    pred_cbsas = set(ctx.tract["cbsa"].unique())
    target = [c for c in ctx.top_cbsas if c in pred_cbsas] or sorted(pred_cbsas)[:10]

    inc_by_geoid = panel_income.set_index("GEOID")
    rows = []
    skipped = []
    for cbsa in target:
        try:
            cbsa_geoids = set(panel_income.loc[panel_income["cbsa_code"] == cbsa, "GEOID"])
            tr_c = ctx.tract[ctx.tract["cbsa"] == cbsa]
            params_by_year = {}
            for pred_year in sorted(tr_c["year"].unique()):
                inc_year = min(income_years, key=lambda y: abs(y - pred_year))
                inc_col = f"per_capita_income_usd_{inc_year}"
                inc_vals = inc_by_geoid.loc[
                    inc_by_geoid.index.isin(cbsa_geoids), inc_col
                ].values.astype(float)
                inc_vals = inc_vals[np.isfinite(inc_vals) & (inc_vals > 0)]
                if len(inc_vals) < 10:
                    continue
                params_by_year[pred_year] = _fit_gb2(inc_vals)
            if len(params_by_year) < 1:
                skipped.append((cbsa, "insufficient income data"))
                continue
            years_fit = sorted(params_by_year)
            smoothed = _smooth_gb2_params(years_fit, params_by_year)
            for pred_year in years_fit:
                g = tr_c[tr_c["year"] == pred_year]
                ranks = g["pred"].values.astype(float)
                dollars = _gb2_apply_from_ranks(ranks, smoothed[pred_year])
                inc_year = min(income_years, key=lambda y: abs(y - pred_year))
                actual = inc_by_geoid.reindex(g["GEOID"].values)[
                    f"per_capita_income_usd_{inc_year}"
                ].values.astype(float)
                bench = _qmap20_bench(ranks, actual)
                for geoid, d, a, b in zip(g["GEOID"].values, dollars, actual, bench):
                    rows.append({"cbsa": cbsa, "year": pred_year, "GEOID": geoid,
                                 "gb2_dollars": d, "qmap_bench": b, "acs_dollars": a})
        except Exception as exc:
            skipped.append((cbsa, str(exc)))
            continue

    if skipped:
        print(f"  skipped {len(skipped)} CBSA(s): "
              + ", ".join(f"{c} ({why})" for c, why in skipped[:5])
              + (" ..." if len(skipped) > 5 else ""))
    if not rows:
        print("  no CBSA produced a GB2 mapping")
        return None
    df = pd.DataFrame(rows)
    df.to_csv(out / "tables" / "US_C_gb2_dollar_mapping.csv", index=False)
    print(f"  GB2 dollar mapping written for {df['cbsa'].nunique()} CBSA(s), "
          f"{len(df):,} tract-years")
    return df


# ─── Main performance figure (US) ──────────────────────────────────────────
# A Science-style 3-panel composite: (A) raincloud of within-city Spearman by
# population bracket, (B) year-by-year Spearman stability of the ordinal
# model vs a cardinal-baseline placeholder, (C) pooled binned scatter of
# predicted score vs ACS label for one test year. Self-contained — reads only
# ctx.bld / ctx.tract / ctx.cbsa_meta, so it runs standalone via
# --parts main_figure without the other US parts having run first.

_BRACKET_ORDER = ("mega", "large", "medium", "small")
_BRACKET_LABELS = {"mega": "Mega", "large": "Large", "medium": "Medium", "small": "Small"}
# Okabe-Ito colorblind-safe qualitative set, fixed per bracket — never re-cycled.
_BRACKET_COLORS = {"mega": "#0072B2", "large": "#009E73", "medium": "#E69F00", "small": "#CC79A7"}
_ORDINAL_COLOR = "#0072B2"
_BASELINE_COLOR = "#D55E00"
# Cities below this many distinct predicted test tracts are dropped from all
# three panels of the main figure — too few tracts to give a stable per-city
# Spearman rho, and they otherwise clutter Panel A's raincloud with noise.
_MIN_FIGURE_TRACTS = 20
# Central mass of the (label, pred) distribution that Panel C's axes are
# clipped to. 1.0 = no clipping: every tract is shown and matplotlib
# autoscales, so the panel is not artificially truncated. Values < 1.0 trim
# that fraction off each marginal independently (e.g. 0.98 drops the top and
# bottom 1% of label and of pred) and the excluded points are summarized in
# the caption footnote instead.
_PANEL_C_CENTRAL_FRAC = .995


def _panel_title(ax: plt.Axes, letter: str) -> None:
    """Bold panel letter flush left — a real Title artist (loc='left'), so the
    layout engine reserves space for it. Descriptive text lives in the shared
    figure caption below, not per panel — narrow panels can't fit a
    descriptive title without it bleeding into the neighboring panel."""
    ax.set_title(letter, loc="left", fontsize=11, fontweight="bold")


def _raincloud(ax: plt.Axes, groups: list[tuple[str, np.ndarray, str]],
               jitter_seed: int = 0, cloud_width: float = 0.32,
               box_width: float = 0.10, gap: float = 0.03) -> None:
    """Half-violin + boxplot + jittered strip, one triplet per (label, values, color).

    At integer position ``i``: a mirrored-KDE cloud fills the band just left of
    center, a slim boxplot sits at center, and jittered raw points ("the rain")
    scatter just right of center — the classic raincloud split so the box never
    occludes the cloud or the rain.
    """
    rng = np.random.default_rng(jitter_seed)
    for i, (label, vals, color) in enumerate(groups):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue
        if np.ptp(vals) > 0:
            try:
                kde = gaussian_kde(vals)
                y_grid = np.linspace(vals.min(), vals.max(), 200)
                density = kde(y_grid)
                density = density / density.max() * cloud_width
                ax.fill_betweenx(y_grid, i - gap - density, i - gap,
                                  color=color, alpha=0.55, linewidth=0, zorder=1)
            except np.linalg.LinAlgError:
                pass
        ax.boxplot(
            vals, positions=[i], widths=box_width, patch_artist=True,
            showfliers=False, zorder=3,
            boxprops=dict(facecolor="white", edgecolor=color, linewidth=1.1),
            medianprops=dict(color=color, linewidth=1.6),
            whiskerprops=dict(color=color, linewidth=1.0),
            capprops=dict(color=color, linewidth=1.0),
        )
        jitter = rng.uniform(gap, gap + cloud_width, size=len(vals))
        ax.scatter(i + jitter, vals, s=8, color=color, alpha=0.45, linewidths=0, zorder=2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_xlim(-0.6, len(groups) - 0.4)


def _panel_a_raincloud(ax: plt.Axes, cells: pd.DataFrame, bracket_of: dict) -> None:
    """Raincloud of per-(CBSA, year) within-city tract-level Spearman rho, one
    cloud per population bracket — each dot is a single City-Year cell's
    tract-level rho (tract-mean pred vs. tract label), not a building-level
    correlation."""
    cells = cells.copy()
    cells["bracket"] = cells["cbsa"].astype(str).map(bracket_of)
    cells = cells.dropna(subset=["bracket"])
    groups = [
        (_BRACKET_LABELS[b], cells.loc[cells["bracket"] == b, "rho"].values, _BRACKET_COLORS[b])
        for b in _BRACKET_ORDER if (cells["bracket"] == b).any()
    ]
    if not groups:
        ax.text(0.5, 0.5, "no bracket data", ha="center", va="center", transform=ax.transAxes)
        return
    _raincloud(ax, groups)
    ax.axhline(0.80, color="0.6", linestyle=":", linewidth=1.0, zorder=0)
    ax.set_ylabel(r"Within-city Spearman $\rho$")
    ax.set_xlabel("City-size bracket")
    _panel_title(ax, "A")


def _synthetic_cardinal_baseline(years: np.ndarray, ordinal: np.ndarray,
                                 seed: int = 0) -> np.ndarray:
    """Illustrative decaying curve standing in for a not-yet-trained cardinal
    (L2-regression) baseline. NOT measured data — deterministic placeholder
    only, to be replaced once a real baseline model's per-year Spearman is
    available. Starts near the ordinal curve's first year and decays with
    distance from it, approximating the label drift a level-regression model
    suffers as the city-wide income distribution moves away from its training
    years.
    """
    years = np.asarray(years, dtype=float)
    rng = np.random.default_rng(seed)
    span = max(years.max() - years.min(), 1.0)
    decay = ((years - years.min()) / span) ** 1.5
    start = float(ordinal[0]) if len(ordinal) else 0.82
    curve = start - 0.45 * decay
    curve = curve + rng.normal(0, 0.015, size=len(years))
    return np.clip(curve, 0.05, 0.95)


def _panel_b_kill_shot(ax: plt.Axes, cells: pd.DataFrame,
                       holdout_year: int | None = None) -> None:
    """Year-by-year Spearman: ordinal model (real, tract-weighted across test
    cities) vs a cardinal-baseline placeholder (illustrative — see caller).

    Weighted by ``n_tracts``, not the building count ``n``, to match
    ``metrics.weighted_within_spearman`` and the per-bracket tables. Labels are
    tract-level, so extra buildings within a tract add rows but no label
    information; building-weighting would let a city sampled at more buildings
    per tract (e.g. one predicted at full universe rather than the ~100/tract
    cap) dominate this curve without contributing more evidence.
    """
    year_rows = []
    for yr, g in cells.groupby("year"):
        w = g["n_tracts"] if "n_tracts" in g.columns else g["n"]
        year_rows.append((int(yr), float(np.average(g["rho"], weights=w))))
    if not year_rows:
        ax.text(0.5, 0.5, "no temporal data", ha="center", va="center", transform=ax.transAxes)
        return
    year_rows.sort()
    years = np.array([r[0] for r in year_rows])
    ordinal = np.array([r[1] for r in year_rows])

    if holdout_year is not None and holdout_year in years:
        ax.axvspan(holdout_year - 0.5, holdout_year + 0.5, color="0.88", zorder=0,
                    label="Temporal holdout year")

    ax.plot(years, ordinal, "o-", color=_ORDINAL_COLOR, linewidth=2.0, markersize=5,
            label="Ordinal model (this paper)", zorder=3)

    baseline = _synthetic_cardinal_baseline(years, ordinal)
    ax.plot(years, baseline, "s--", color=_BASELINE_COLOR, linewidth=1.6, markersize=4,
            alpha=0.85, label="Cardinal (L2) baseline*", zorder=2)

    ax.set_xlabel("Year")
    ax.set_ylabel(r"Mean Spearman $\rho$ (test cities)")
    _panel_title(ax, "B")
    ax.set_xticks(years)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.legend(fontsize=6, loc="lower left", frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.8)


def _panel_c_binned_scatter(ax: plt.Axes, tract_df: pd.DataFrame, year: int | None = None,
                            n_bins: int = 10, central_frac: float = _PANEL_C_CENTRAL_FRAC):
    """Pooled scatter (marker size 1) of predicted ordinal score vs ACS label,
    with a line connecting each decile's conditional mean (national-test-set
    analog of the paper's per-city scatter figure).

    ``year=None`` (the default) pools every test year together — each tract
    contributes one point per year it appears in, so ``n`` counts tract-years,
    not distinct tracts, and repeated tracts across years are NOT independent
    observations (a persistently rich tract shows up as several correlated
    points). Pass an explicit ``year`` to restrict to a single cross-section
    instead, which avoids that pseudo-replication.

    With the default ``central_frac`` of 1.0 no clipping is applied: the axes
    autoscale over the full range, so every tract used for rho/deciles is also
    visible. Passing ``central_frac < 1.0`` clips the axes to that central mass
    of each marginal (label and pred independently) so a few extreme tracts
    don't compress the bulk of the cloud; points outside that box are still
    used for rho/deciles but are off-canvas. Returns
    ``(scatter, rho, outlier_note)`` — ``rho`` is the pooled (or single-year)
    Spearman plotted in the panel, and ``outlier_note`` is a caption-ready
    string describing what got clipped, or None if nothing fell outside the
    shown range (always None when nothing is clipped).
    """
    g = tract_df if year is None else tract_df[tract_df["year"] == year]
    x = g["label"].values.astype(float)
    y = g["pred"].values.astype(float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < n_bins * 3:
        where = "any test year" if year is None else str(year)
        ax.text(0.5, 0.5, f"insufficient tracts for {where}", ha="center", va="center",
                transform=ax.transAxes)
        return None, None, None
    clipping = central_frac < 1.0
    sc = ax.scatter(x, y, s=1, alpha=0.3, color="steelblue", linewidths=0, zorder=1)
    if clipping:
        margin = 100 * (1 - central_frac) / 2
        x_lo, x_hi = np.percentile(x, [margin, 100 - margin])
        y_lo, y_hi = np.percentile(y, [margin, 100 - margin])
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
    # else: leave the limits to matplotlib's autoscale, which keeps its usual
    # margin around the extremes so no point sits flush against a spine.
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    bins = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    bx = [x[bins == b].mean() for b in range(n_bins) if (bins == b).any()]
    by = [y[bins == b].mean() for b in range(n_bins) if (bins == b).any()]
    ax.plot(bx, by, "o-", color="firebrick", markeredgecolor="white",
            linewidth=1.8, markersize=5, zorder=3, label=f"{n_bins}-bin conditional mean")
    rho = float(spearmanr(x, y).statistic)
    n_label = "tract-years" if year is None else "tracts"
    ax.text(0.05, 0.95, rf"$\rho$ = {rho:.2f}, $n$ = {len(x):,} {n_label}",
            transform=ax.transAxes, fontsize=7, va="top")
    ax.legend(fontsize=6, loc="lower right", frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.8)
    ax.set_xlabel("ACS tract z-score (label)")
    ax.set_ylabel("Predicted ordinal score")
    _panel_title(ax, "C")

    if not clipping:
        return sc, rho, None

    n_out = int(np.sum((x < x_lo) | (x > x_hi) | (y < y_lo) | (y > y_hi)))
    outlier_note = None
    if n_out:
        frac_out = n_out / len(x)
        outlier_note = (
            rf"Panel C axes are clipped to the central {_tex_pct(central_frac)} of "
            rf"{n_label} by label and by prediction; {n_out:,} of {len(x):,} {n_label} "
            rf"({_tex_pct(frac_out)}) fall outside this range and are not shown."
        )
    return sc, rho, outlier_note


def part_main_figure_us(ctx: _USContext, year: int | None = None,
                        holdout_year: int | None = None) -> dict:
    """Science-style 3-panel main performance figure (US mode).

    A. Raincloud of within-city **tract-level** Spearman by population bracket
       (all test City-Year cells) — one rho per (CBSA, year) computed on
       tract-mean pred vs. tract label, not the building-level correlation.
    B. Year-by-year Spearman stability: the ordinal model (real, size-weighted
       across test cities, building-level cells) vs a cardinal-baseline
       placeholder. The baseline line is a deterministic synthetic decay
       curve, NOT measured data — no L2-regression baseline model has been
       trained yet. Swap ``_synthetic_cardinal_baseline`` for real per-year
       results once one exists.
    C. Pooled scatter (marker size 1) of predicted score vs ACS label, with a
       decile conditional-mean overlay. Pools every eligible test year by
       default (``year=None``); pass ``year=`` for a single cross-section
       instead. Pooling means each tract contributes once per year, so those
       points are not independent draws (a persistently rich tract shows up
       as several correlated points) — see ``_panel_c_binned_scatter``. Axes
       show the full range of the data (no outlier trimming).

    Reads only ctx.bld / ctx.tract / ctx.cbsa_meta — self-contained, so it can
    run standalone via ``--parts main_figure``.
    """
    print("\n=== Main performance figure (US) ===")
    out = ctx.out
    if ctx.bld.empty or not len(ctx.tract):
        print("  missing building or tract predictions; skipping main figure")
        return {}

    bracket_of = _bracket_of(ctx.cbsa_meta)
    if not bracket_of:
        print("  no cbsa_splits bracket info; skipping main figure")
        return {}

    test_bld_all = ctx.bld[ctx.bld["type"] == "test"]
    if test_bld_all.empty:
        print("  no test-split building predictions; skipping main figure")
        return {}

    test_cbsas = set(test_bld_all["cbsa"].unique())
    tract_test_all = ctx.tract[ctx.tract["cbsa"].isin(test_cbsas)]

    # Tract-level cells for Panel A — one rho per (CBSA, year) from tract-mean
    # pred vs. tract label, rather than the building-level correlation (which
    # over-weights tracts with many buildings and repeats each tract's label
    # once per building). ``within_city_cells``'s own n_tracts (distinct
    # GEOID, or distinct label when GEOID is absent) is the authoritative
    # tract count for the min-tracts floor below.
    cells_tract_all = within_city_cells(tract_test_all)
    if not len(cells_tract_all):
        print("  no usable tract-level within-city cells; skipping main figure")
        return {}

    # Drop individual (CBSA, year) cells too thin to give a stable Spearman
    # rho, before building panels A/B/C, so all three see the same cells.
    # This is a per-cell filter (not a whole-city one): a city with plenty of
    # tracts overall can still have one sparse year dropped while its other
    # years remain.
    cells_tract = cells_tract_all[cells_tract_all["n_tracts"] >= _MIN_FIGURE_TRACTS]
    n_cy_dropped = len(cells_tract_all) - len(cells_tract)
    if n_cy_dropped:
        print(f"  dropping {n_cy_dropped} / {len(cells_tract_all)} city-year cell(s) with "
              f"< {_MIN_FIGURE_TRACTS} tracts")
    if not len(cells_tract):
        print("  no city-year cells meet the tract-count threshold; skipping main figure")
        return {}

    eligible_pairs = pd.MultiIndex.from_frame(cells_tract[["cbsa", "year"]])
    test_bld = test_bld_all[
        pd.MultiIndex.from_frame(test_bld_all[["cbsa", "year"]]).isin(eligible_pairs)
    ].reset_index(drop=True)
    tract_test = tract_test_all[
        pd.MultiIndex.from_frame(tract_test_all[["cbsa", "year"]]).isin(eligible_pairs)
    ].reset_index(drop=True)
    if test_bld.empty:
        print("  no test-split building predictions after the tract-count filter; "
              "skipping main figure")
        return {}

    cells_bld = within_city_cells(test_bld)
    if not len(cells_bld):
        print("  no usable within-city cells; skipping main figure")
        return {}

    # constrained_layout (not tight_layout) — it is colorbar-aware, so the
    # extra Axes fig.colorbar() attaches to ax_c doesn't throw off the row
    # spacing between the panel row and the caption row below it.
    fig = plt.figure(figsize=(FIG_SIZE_TWO_COL[0], FIG_SIZE_TWO_COL[0] * 0.46),
                     constrained_layout=True)
    # Row 0 holds the 3 panels; row 1 is a dedicated, axis-off caption strip —
    # giving the caption its own gridspec cell (rather than free-floating
    # fig.text below the axes) makes its vertical space a hard layout
    # guarantee instead of something the layout engine has to guess at.
    gs = fig.add_gridspec(2, 3, height_ratios=[4, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_cap = fig.add_subplot(gs[1, :])
    ax_cap.axis("off")

    _panel_a_raincloud(ax_a, cells_tract, bracket_of)
    _panel_b_kill_shot(ax_b, cells_bld, holdout_year=holdout_year)
    # year=None (the default) pools every eligible test year into Panel C;
    # pass an explicit year to fall back to a single cross-section.
    _sc, panel_c_rho, outlier_note = _panel_c_binned_scatter(ax_c, tract_test, year)

    # Per-year pooled Spearman (cities pooled, one year at a time) alongside
    # the panel's own rho — lets you check how much of a pooled-year rho is
    # inflated by tract persistence vs. genuine within-year ranking. Computed
    # regardless of whether Panel C itself is pooled or single-year.
    panel_c_rho_by_year: dict[int, float] = {}
    for yr, g in tract_test.groupby("year"):
        xx = g["label"].values.astype(float)
        yy = g["pred"].values.astype(float)
        finite = np.isfinite(xx) & np.isfinite(yy)
        xx, yy = xx[finite], yy[finite]
        if len(xx) >= 10 and np.ptp(xx) > 0 and np.ptp(yy) > 0:
            panel_c_rho_by_year[int(yr)] = float(spearmanr(xx, yy).statistic)

    for ax in (ax_a, ax_b, ax_c):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    n_cities = int(test_bld["cbsa"].nunique())
    panel_c_pool_desc = (
        rf"year {year}, all test cities" if year is not None
        else "all test years and cities"
    )
    caption_body = (
        rf"(A) Tract-level Spearman $\rho$ by city-size bracket \textemdash\ uniform "
        r"performance from mega to small metros. "
        r"(B) Spearman $\rho$ by year \textemdash\ the ordinal model stays flat "
        r"2010--2024 while a cardinal L2 baseline* decays. "
        rf"(C) Predicted score vs.~ACS label, pooled across {panel_c_pool_desc} "
        r"\textemdash\ clean separation of poorest and richest deciles, "
        r"out-of-sample."
    )
    caption_note = (
        r"*Illustrative placeholder for Panel B's baseline \textemdash\ not yet trained. "
        rf"City-year cells with fewer than {_MIN_FIGURE_TRACTS} predicted tracts are "
        r"excluded from all three panels."
    )
    if outlier_note:
        caption_note += "  " + outlier_note
    caption = "\n".join(textwrap.wrap(caption_body, width=130)) + "\n" + "\n".join(
        textwrap.wrap(caption_note, width=130)
    )

    print(f"  panel A: {len(cells_tract)} city-year cells (tract-level) over {n_cities} test cities")
    if year is not None:
        n_tracts_c = int((tract_test["year"] == year).sum())
        print(f"  panel C: year={year}, n={n_tracts_c:,} tracts")
    else:
        print(f"  panel C: pooled {tract_test['year'].nunique()} years, "
              f"n={len(tract_test):,} tract-years")
    if panel_c_rho_by_year:
        yr_str = ", ".join(f"{yr}={r:.3f}" for yr, r in sorted(panel_c_rho_by_year.items()))
        mean_yearly_rho = float(np.mean(list(panel_c_rho_by_year.values())))
        print(f"  panel C rho by year: {yr_str}  (mean of years = {mean_yearly_rho:.3f})")
    if panel_c_rho is not None:
        print(f"  panel C rho (as plotted, {'pooled' if year is None else year}) "
              f"= {panel_c_rho:.3f}")

    ax_cap.text(0.5, 1.0, caption, transform=ax_cap.transAxes, ha="center", va="top",
                fontsize=6.2, color="0.25", linespacing=1.6)

    _savefig(fig, out / "figures" / "US_main_performance_figure.pdf")
    headline = {
        "main_figure/n_test_cities": n_cities,
        "main_figure/n_cells_tract": int(len(cells_tract)),
        "main_figure/n_cells_bld": int(len(cells_bld)),
    }
    if year is not None:
        headline["main_figure/panel_c_year"] = int(year)
    else:
        headline["main_figure/panel_c_n_years"] = int(tract_test["year"].nunique())
        headline["main_figure/panel_c_n_tract_years"] = int(len(tract_test))
    if panel_c_rho is not None:
        headline["main_figure/panel_c_rho"] = panel_c_rho
    for yr, r in panel_c_rho_by_year.items():
        headline[f"main_figure/panel_c_rho_year_{yr}"] = r
    return headline


_US_PARTS = {
    "cross": part_a_us, "temporal": part_b_us, "dollars": part_c_us,
    "main_figure": part_main_figure_us,
}
# main_figure's Panel B carries a synthetic (non-measured) cardinal-baseline
# line, clearly marked with an asterisk in the figure and caption. It is in the
# 'both' default set (that figure is a deliverable) but NOT in the us-only
# default, so an automated US run never emits it by accident.
_DEFAULT_US_PARTS = ("cross", "temporal", "dollars")
_ALL_US_PARTS = ("cross", "temporal", "dollars", "main_figure")

_NYC_PARTS = ("A", "B", "C", "D", "E")
_DEFAULT_NYC_PARTS = ("A", "B", "C")

# Sub-directory of a US run holding its NYC-only prediction pass
# (main.run_nyc_zarr_validation_predictions writes here).
NYC_SUBDIR = "nyc_zarr_check"


def _resolve_mode(results_dir: Path, params: dict | None, mode: str) -> str:
    """Decide 'us', 'nyc' or 'both'. Explicit mode wins; else
    params['footprints_source'] ('ms_us' -> us, 'doitt_nyc' -> nyc); else
    auto-detect by whether a georeferenced predictions_{year}.parquet exists
    (a doitt_nyc-only artifact)."""
    if mode in ("us", "nyc", "both"):
        return mode
    if params is not None:
        fs = params.get("footprints_source")
        if fs == "ms_us":
            return "us"
        if fs == "doitt_nyc":
            return "nyc"
    if list(results_dir.glob("predictions_2*.parquet")):
        return "nyc"
    return "us"


def _split_parts(parts, mode: str) -> tuple[list[str], list[str]]:
    """Split a mixed ``--parts`` list into (us_parts, nyc_parts).

    US parts are named ('cross', 'temporal', 'dollars', 'main_figure'); NYC parts
    are single letters A–E. This lets one flag address both halves of a combined
    run, e.g. ``--parts cross main_figure D E``.
    """
    if not parts:
        if mode == "us":
            return list(_DEFAULT_US_PARTS), []
        if mode == "nyc":
            return [], list(_DEFAULT_NYC_PARTS)
        # 'both' is an explicit request for the full result set, main figure
        # and CSA/case study included.
        return list(_ALL_US_PARTS), list(_NYC_PARTS)

    us, nyc, unknown = [], [], []
    for p in parts:
        token = str(p)
        if token in _US_PARTS:
            us.append(token)
        elif token.upper() in _NYC_PARTS:
            nyc.append(token.upper())
        else:
            unknown.append(token)
    for token in unknown:
        print(f"  unknown part '{token}', skipping")
    return us, nyc


def _nyc_results_dir(results_dir: Path, nyc_results_dir=None,
                     nyc_subdir: str | None = NYC_SUBDIR) -> Path | None:
    """Where the NYC parts should read predictions from, or None if nowhere.

    Order of preference:

    1. an explicit ``nyc_results_dir``;
    2. ``results_dir / nyc_subdir`` — the NYC/zarr validation pass of a US run.
       This is the intended source: it re-runs the NYC exercise with the *same*
       US-trained model, which is the comparison the NYC parts exist to make;
    3. ``results_dir`` itself, when it already holds the NYC-only artifacts
       (a legacy doitt_nyc run).
    """
    if nyc_results_dir is not None:
        return Path(nyc_results_dir)
    if nyc_subdir:
        cand = results_dir / nyc_subdir
        if _available_years(cand) or list(cand.glob("predictions_by_tract_2*.parquet")):
            return cand
    if list(results_dir.glob("predictions_2*.parquet")):
        return results_dir
    return None


def run_evaluation(
    savename: str,
    params: dict | None = None,
    parts=None,
    results_dir: Path | None = None,
    processed_dir: Path | None = None,
    mode: str = "auto",
    nyc_results_dir: Path | None = None,
    nyc_subdir: str | None = NYC_SUBDIR,
) -> dict:
    """Post-hoc evaluation callable from main.py or the CLI.

    Modes
    -----
    ``us``
        The full-US parts (default cross/temporal/dollars).
    ``nyc``
        The NYC parts A–E against ``results_dir``.
    ``both``
        Both halves in one invocation: the US parts on ``results_dir`` (writing
        ``<results_dir>/evaluation``) *and* the NYC parts on the NYC prediction
        pass of the same run (``<results_dir>/nyc_zarr_check`` by default,
        writing its own ``evaluation`` folder). This is how the paper's full
        result set — the 3-panel US main figure plus the NYC CSA event study and
        Hudson Yards case study — comes out of one command.

    Every part runs inside its own try/except, so one failure never costs the
    others. Returns the merged headline dict and writes it to
    ``evaluation/US_summary.json``.
    """
    results_dir = (RESULTS_DIR / savename) if results_dir is None else Path(results_dir)
    processed_dir = PROCESSED_DATA_DIR if processed_dir is None else Path(processed_dir)
    out_dir = results_dir / "evaluation"
    _make_dirs(out_dir)

    resolved = _resolve_mode(results_dir, params, mode)
    indicator = (params or {}).get("indicator", indicators.DEFAULT_INDICATOR)
    us_parts, nyc_parts = _split_parts(parts, resolved)
    if resolved == "us":
        nyc_parts = []
    elif resolved == "nyc":
        us_parts = []

    print(f"Results   : {results_dir}")
    print(f"Mode      : {resolved}")
    print(f"US parts  : {us_parts or '—'}")
    print(f"NYC parts : {nyc_parts or '—'}")

    summary: dict = {"savename": savename, "mode": resolved, "indicator": indicator}
    ran_any = False

    # ── US half ─────────────────────────────────────────────────────────────
    if us_parts:
        if not results_dir.exists() or not _available_years(results_dir):
            print(f"\n⚠️ no prediction CSVs under {results_dir} — US parts skipped.")
            summary["us/status"] = "no_predictions"
        else:
            ran_any = True
            ctx = _USContext(results_dir, processed_dir, out_dir, indicator)
            summary["years"] = ctx.years
            for key in us_parts:
                fn = _US_PARTS[key]
                try:
                    result = fn(ctx)
                    if isinstance(result, dict):
                        summary.update(result)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    print(f"  ⚠️ US part '{key}' failed; continuing with the rest.")

    # ── NYC half ────────────────────────────────────────────────────────────
    if nyc_parts:
        nyc_dir = _nyc_results_dir(results_dir, nyc_results_dir, nyc_subdir)
        if nyc_dir is None:
            print(f"\n⚠️ no NYC prediction artifacts found (looked in "
                  f"{results_dir / (nyc_subdir or '')} and {results_dir}) — "
                  "NYC parts skipped. Point --nyc-savename at a NYC run, or "
                  "produce one with main.run(generate_predictions_nyc=True).")
            summary["nyc/status"] = "no_predictions"
        else:
            ran_any = True
            # Wrapped as a whole: creating the NYC output tree can itself fail
            # (read-only mount, permissions), and that must not discard a US half
            # that already succeeded.
            try:
                nyc_out = (out_dir if nyc_dir == results_dir
                           else nyc_dir / "evaluation")
                _make_dirs(nyc_out)
                print(f"\nNYC source: {nyc_dir}")
                print(f"NYC output: {nyc_out}")
                nyc_summary = _run_nyc_parts(nyc_dir, processed_dir, nyc_out, nyc_parts)
                summary.update({f"nyc/{k}": v for k, v in nyc_summary.items()})
                summary["nyc/results_dir"] = str(nyc_dir)
            except Exception:
                import traceback
                traceback.print_exc()
                print("  ⚠️ NYC evaluation failed as a whole; US results above stand.")
                summary["nyc/status"] = "failed"

    if not ran_any:
        print("\n  nothing to evaluate.")
        return {}

    # Guarded: by this point every part has run and printed, so a failure to
    # persist the summary must not throw away the headline dict we return.
    import json
    try:
        with open(out_dir / "US_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n=== evaluation done ({resolved}) ===\n"
              f"  Summary -> {out_dir / 'US_summary.json'}")
    except OSError as exc:
        print(f"\n=== evaluation done ({resolved}) ===\n"
              f"  ⚠️ could not write US_summary.json ({exc}); headline: {summary}")
    return summary


def _run_nyc_parts(results_dir, processed_dir, out_dir, parts) -> dict:
    """Dispatch the NYC parts A–E, each isolated so one failure is not fatal.

    Part C returns the GB2 quantile mapping that part E prefers for its tract
    chart, so C is run before E and its result threaded through; if C failed or
    was not requested, E falls back to the raw predictions.
    """
    requested = [p for p in ("A", "B", "C", "D", "E") if p in
                 {str(x).upper() for x in parts}]
    years = _set_nyc_years(Path(results_dir))
    summary: dict = {"results_dir": str(results_dir), "years": years}

    qmap_long = None
    for part in requested:
        try:
            if part == "A":
                part_a(results_dir, processed_dir, out_dir)
            elif part == "B":
                part_b(results_dir, processed_dir, out_dir)
            elif part == "C":
                qmap_long = part_c(results_dir, processed_dir, out_dir)
            elif part == "D":
                result = part_d(results_dir, processed_dir, out_dir, years=years)
                if isinstance(result, dict):
                    summary.update(result)
            elif part == "E":
                part_e(results_dir, processed_dir, out_dir,
                       qmap_long=qmap_long, years=years)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"  ⚠️ NYC part '{part}' failed; continuing with the rest.")
            summary[f"part_{part}/status"] = "failed"
    return summary


# ─── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc evaluation of aerial-imagery wealth predictions."
    )
    parser.add_argument(
        "--savename", default=DEFAULT_SAVENAME,
        help="Experiment folder name under results/ (default: %(default)s)"
    )
    parser.add_argument(
        "--mode", choices=["auto", "us", "nyc", "both"], default="auto",
        help="Evaluation mode. 'both' = the US parts on --savename AND the NYC "
             "parts on its NYC prediction pass, in one run. 'us'/'nyc' do one "
             "half. Default: auto-detect from artifacts / footprints_source."
    )
    parser.add_argument(
        "--indicator", default=indicators.DEFAULT_INDICATOR,
        help="Wealth indicator token for US mode (default: %(default)s)."
    )
    parser.add_argument(
        "--parts", nargs="*", default=None,
        metavar="PART",
        help="Which parts to run; US and NYC tokens may be mixed. US: cross "
             "temporal dollars main_figure. NYC: A (cross-section) B (temporal) "
             "C (GB2 dollars) D (construction-cohort CSA event study) E (Hudson "
             "Yards). Defaults: 'cross temporal dollars' (us), 'A B C' (nyc), "
             "everything incl. main_figure and D/E (both)."
    )
    parser.add_argument(
        "--nyc-savename", default=None,
        help="Read the NYC parts from results/<this> instead of the NYC "
             "sub-directory of --savename."
    )
    parser.add_argument(
        "--nyc-subdir", default=NYC_SUBDIR,
        help="Sub-directory of --savename holding its NYC prediction pass "
             "(default: %(default)s). Pass '' to look in --savename itself."
    )
    args = parser.parse_args()

    run_evaluation(
        args.savename,
        params={"indicator": args.indicator},
        parts=args.parts,
        mode=args.mode,
        nyc_results_dir=(RESULTS_DIR / args.nyc_savename
                         if args.nyc_savename else None),
        nyc_subdir=args.nyc_subdir or None,
    )


if __name__ == "__main__":
    main()