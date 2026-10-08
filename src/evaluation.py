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
import faulthandler
import re
import textwrap
import warnings
from pathlib import Path

# This module drives a lot of native code — GEOS via geopandas, PROJ via pyproj,
# BLAS via numpy/statsmodels — and a crash down there arrives as a bare
# "Segmentation fault" with no indication of which line of Python was running.
# faulthandler costs nothing until a fatal signal arrives and then prints the
# Python traceback, which is the difference between a reproducible bug report and
# guesswork. Harmless if the host process already installed it.
faulthandler.enable()

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
# Module level (part_d also imports it locally) so `CONTROL_THRESHOLD` can be
# part_d's default argument rather than a sentinel.
from src import csa_event_study as ces

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


# ─── nodata-hole retrofit ─────────────────────────────────────────────────────
# `CyclicCacheManager._reject_for_zero_pixels` (src/main.py) drops a crop that is
# more zero-fill than imagery, so a nodata hole never reaches the model. Any
# prediction pass written BEFORE that guard existed carries the holes as ordinary
# finite numbers, and nothing downstream can tell them apart: the balanced-panel
# check tests presence, not validity.
#
# The tell is that the model's response to an all-zero crop is deterministic —
# every black crop in a pass scores the same value to the bit. Measured in
# results/run_20260722: Harford County (MD) NAIP 2013 has 3,919 buildings at
# exactly 0.68359375 against 178-185 in its neighbouring years, 17 tracts
# ≥99% black, biasing those 2013 tract means +0.25 on a city mean of -0.16. In
# Baltimore's a=1 event study that lands squarely on k=-1 and produces a spurious
# +0.256 pre-trend coefficient.
#
# This filter retrofits the guard onto predictions already on disk so those runs
# do not have to be regenerated. It is keyed on the sentinel value rather than on
# a run name on purpose: a run-name check silently does nothing the moment a
# directory is renamed or copied, while the sentinel is the actual evidence of the
# defect and disappears by itself once a pass is regenerated with the guard
# active. Set `NODATA_PREDICTION_VALUE = None` to disable.
NODATA_PREDICTION_VALUE: float | None = 0.68359375

# Dropping the black buildings is necessary but not sufficient. The guard drops
# the crop, which drops the building, which EMPTIES the tract-year, which drops
# the tract at `build_event_panel`'s balance step. Removing only the rows would
# instead leave a 99%-black tract represented by the ~1% of buildings that
# happened to fall outside the hole — a real tract-year backed by a garbage
# sample. So a tract-year at or above this share is removed whole. Measured on
# Baltimore a=1: rows-only gets k=-1 from +0.256 to +0.131, rows + tract-years
# gets it to +0.086, which is what re-running with the guard produces.
NODATA_TRACT_YEAR_SHARE: float = 0.5


def _drop_nodata_predictions(df: pd.DataFrame, source: Path | None = None) -> pd.DataFrame:
    """Remove nodata-hole predictions from an already-loaded prediction frame.

    Expects the canonical columns (``GEOID``, ``year``, ``pred``). Returns the
    frame unchanged when the sentinel is absent, which is the normal case for any
    pass written with the guard active — so this costs one comparison and says
    nothing on a clean run.
    """
    if NODATA_PREDICTION_VALUE is None or df.empty:
        return df
    black = df["pred"].round(6) == round(float(NODATA_PREDICTION_VALUE), 6)
    n_black = int(black.sum())
    if not n_black:
        return df

    # Share of each tract-year that is nodata, broadcast back over its rows.
    share = black.groupby([df["GEOID"], df["year"]]).transform("mean")
    dead = share >= float(NODATA_TRACT_YEAR_SHARE)
    n_dead_cells = int(df.loc[dead, ["GEOID", "year"]].drop_duplicates().shape[0])
    # Only the non-black rows are *additional* removals; the black ones are
    # already counted above, and most of them sit inside these same tract-years.
    n_remnant = int((dead & ~black).sum())

    kept = df[~black & ~dead]
    where = f" in {source}" if source is not None else ""
    print(f"    ⚠️ nodata-hole filter{where}: dropped {n_black:,} buildings at the "
          f"black-crop value {NODATA_PREDICTION_VALUE}"
          + (f", plus the {n_remnant:,} surviving buildings of "
             f"{n_dead_cells:,} tract-years that were ≥{NODATA_TRACT_YEAR_SHARE:.0%} "
             f"nodata" if n_dead_cells else "")
          + f" ({len(df) - len(kept):,} of {len(df):,} rows). These passes predate "
            f"the CyclicCacheManager zero-pixel guard; regenerating them makes this "
            f"filter a no-op.")
    return kept.reset_index(drop=True)


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
    df = _drop_nodata_predictions(df, source=results_dir)
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
        # Above x = 1 use I_{z/(1+z)}(p,q) = 1 - I_{1/(1+z)}(q,p): with large c,
        # z = x^c makes z/(1+z) round to 1.0 and the CDF jump straight to 1.
        z = x ** c
        return np.where(z <= 1.0, _sp.betainc(p, q, z / (1.0 + z)),
                        1.0 - _sp.betainc(q, p, 1.0 / (1.0 + z)))

    def _ppf(self, u, c, p, q):
        # v/(1-v) with 1-v from the complementary inverse (1-V ~ Beta(q, p)):
        # for large c and tiny p, q (fits seen on W2 wealth, e.g. CBSA 30780)
        # betaincinv(p, q, u) rounds to exactly 1.0 and the naive 1-v
        # overflows the quantile to inf even at the median's neighbours.
        v = _sp.betaincinv(p, q, u)
        w = _sp.betaincinv(q, p, 1.0 - u)
        return (v / w) ** (1.0 / c)

    def _sf(self, x, c, p, q):
        z = x ** c
        return np.where(z <= 1.0, 1.0 - _sp.betainc(p, q, z / (1.0 + z)),
                        _sp.betainc(q, p, 1.0 / (1.0 + z)))


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

# Anticipation allowances, in panel PERIODS, each run as a full arm of Part D
# into its own output folder. Cohorts are dated from year_built = completion,
# while demolition, excavation and superstructure are visible in the imagery
# earlier, so the periods just before completion are partly treated; see
# src/csa_event_study.py build_event_panel(anticipation=...).
#
# These are periods, not years, and a period is not the same number of years in
# every city: 2 for NYC/Seattle/San Antonio, 2-3 for Tampa's irregular NAIP grid,
# and 1 for Cook County's annual ortho panel if Chicago ever unblocks. So a=1
# buys a 4-year baseline gap in NYC and a 2-year one in Chicago. Read the arms
# against `PeriodMap.event_time_years`, not as a fixed number of years.
#
# Which arm is defensible is a per-city question, and the diagnostic is
# ATT(k=-1) under a=0: Tampa +0.012 (t=1.0) and San Antonio +0.022 (t=0.9) show
# no pre-completion contamination, Seattle +0.038 (t=2.7) does, and NYC's whole
# pre-path is elevated rather than just its last period, which no fixed shift
# fixes. San Antonio's four-period panel also cannot support a=2: it leaves ~23
# treated tracts against MIN_TREATED_UNITS.
CSA_ANTICIPATION_ARMS: tuple[int, ...] = (0, 1, 2)

# Backwards-compatible alias for the pre-arm robustness rows.
CSA_ANTICIPATION_PERIODS: tuple[int, ...] = tuple(
    a for a in CSA_ANTICIPATION_ARMS if a
)


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


def _csa_city_tracts(processed_dir: Path, prefixes: tuple[str, ...],
                     spec=None, footprints=None) -> gpd.GeoDataFrame:
    """Tract geometries for a city.

    ``spec.tract_source == "prefix"`` (default) selects by county FIPS. With
    ``"footprints"`` the prefix set is further restricted to tracts the city's
    *dated footprints* actually cover — which is what a sub-county source needs.
    Chicago's municipal footprint layer stops at the city line well inside Cook
    County, so a bare FIPS prefix would admit ~500 suburban tracts with no
    year-built data. Those tracts have no datable construction, so every one of
    them would join the event study as a permanent never-treated control,
    stuffing the control group with places where the treatment variable cannot
    even be measured.
    """
    splits = _load_splits(processed_dir)
    sub = splits[splits["GEOID_str"].str.startswith(tuple(prefixes))].copy()
    sub = sub[["GEOID_str", "geometry"]].reset_index(drop=True)
    if spec is None or getattr(spec, "tract_source", "prefix") != "footprints":
        return sub
    if footprints is None or sub.empty:
        return sub

    cent = gpd.GeoDataFrame(geometry=footprints.geometry.centroid,
                            crs=footprints.crs)
    joined = gpd.sjoin(cent, sub.to_crs(footprints.crs), how="inner",
                       predicate="within")
    counts = joined.groupby("GEOID_str").size()
    covered = set(counts[counts >= int(spec.min_tract_footprints)].index)
    kept = sub[sub["GEOID_str"].isin(covered)].reset_index(drop=True)
    print(f"    footprint coverage: {len(kept):,}/{len(sub):,} tracts have "
          f">={spec.min_tract_footprints} dated footprints")
    return kept


# ─── Part D input cache ───────────────────────────────────────────────────────
# Part D is called once per (anticipation arm x undated screen) — up to six times
# per run — and every one of those calls used to re-read each city's footprint
# parquet, reproject ~1M polygons per threshold, and re-parse the multi-million
# row prediction CSVs. None of that work depends on the arm or the screen: the
# anticipation shift is applied in `build_event_panel`, and the screen is applied
# to already-assigned per-building areas. Doing it once and holding the small,
# geometry-free derivatives is what keeps peak memory flat across the arms
# instead of stepping up on each one.
#
# What is cached is deliberately bounded: per-building (GEOID, area, year_built,
# demolition_year) tables and tract-year outcome means — tens of MB for the five
# cities combined. The footprint GeoDataFrames and the building-level prediction
# frames, which are the actual gigabytes, are read inside these helpers and
# dropped before they return. `_csa_cache_clear` releases the rest once the arms
# are done.
_CSA_GEOM_CACHE: dict = {}
_CSA_OUTCOME_CACHE: dict = {}


def _csa_cache_clear() -> None:
    """Drop the Part D input cache and hand the pages back to the OS."""
    _CSA_GEOM_CACHE.clear()
    _CSA_OUTCOME_CACHE.clear()
    _csa_release_memory()


def _csa_release_memory() -> None:
    """Collect, then ask glibc to return freed arenas to the OS.

    Part D's geometry work frees almost everything it allocates, but glibc keeps
    the arenas: RSS ratchets up by a few hundred MB per arm and never comes back
    down, because a freed 1M-element object array leaves the arena too
    fragmented to trim on its own. `malloc_trim` is the documented way to force
    it, and it is a no-op anywhere it is not available.
    """
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _csa_city_geometry(processed_dir: Path, spec):
    """``(per_building, tracts, construction_year)`` for one city, cached.

    ``per_building`` is :func:`ces.footprint_tract_areas` output — the footprint
    to tract assignment with areas and years, and the single most expensive step
    in Part D. ``construction_year`` is the year-built series indexed by building
    id, used for the incumbent-only outcome; ``None`` when the city's footprint
    id cannot be joined to predictions.

    The footprint GeoDataFrame itself is released before returning: it is the
    largest object in the whole part (NYC is ~1.1M polygons) and nothing
    downstream of the tract assignment needs the geometry.
    """
    key = (str(processed_dir), spec.key)
    if key in _CSA_GEOM_CACHE:
        return _CSA_GEOM_CACHE[key]

    footprints = gpd.read_parquet(processed_dir / spec.footprints_filename)
    try:
        construction_year = None
        if spec.id_index is not None and footprints.index.name == spec.id_index:
            construction_year = pd.to_numeric(
                footprints[spec.year_col], errors="coerce"
            )
        tracts = _csa_city_tracts(processed_dir, spec.geoid_prefixes,
                                  spec=spec, footprints=footprints)
        per_building = (
            pd.DataFrame() if tracts.empty else
            ces.footprint_tract_areas(
                footprints, tracts, year_col=spec.year_col,
                demolition_col=spec.demolition_col, area_epsg=spec.area_epsg,
            )
        )
    finally:
        del footprints
    _csa_release_memory()

    _CSA_GEOM_CACHE[key] = (per_building, tracts, construction_year)
    return _CSA_GEOM_CACHE[key]


def _csa_city_outcomes(csa_root: Path, results_dir: Path, processed_dir: Path,
                       spec, city_years: list[int], baseline_year: int,
                       tracts, construction_year, shared_bld_fn):
    """Tract-year outcome means for one city, cached.

    Reads the city's dense per-city prediction pass (falling back to the shared
    whole-city pass via ``shared_bld_fn``, called only if needed), reduces it to
    tract-year means, and drops the building-level frame — which is millions of
    rows and the other half of Part D's peak.
    """
    # `processed_dir` belongs in the key even though it is not read here: the
    # outcomes depend on `tracts` and `construction_year`, which come from it.
    key = (str(csa_root), str(results_dir), str(processed_dir), spec.key,
           tuple(city_years), int(baseline_year))
    if key in _CSA_OUTCOME_CACHE:
        return _CSA_OUTCOME_CACHE[key]

    city_bld = _csa_city_preds(csa_root, spec, city_years)
    if city_bld.empty:
        city_bld = shared_bld_fn()
    if city_bld.empty:
        _CSA_OUTCOME_CACHE[key] = pd.DataFrame()
        return _CSA_OUTCOME_CACHE[key]
    city_bld = city_bld[city_bld["GEOID_str"].isin(set(tracts["GEOID_str"]))]
    if city_bld.empty:
        _CSA_OUTCOME_CACHE[key] = pd.DataFrame()
        return _CSA_OUTCOME_CACHE[key]
    outcomes = ces.tract_outcomes(
        city_bld, construction_year=construction_year,
        baseline_year=baseline_year,
    )
    del city_bld
    _csa_release_memory()

    _CSA_OUTCOME_CACHE[key] = outcomes
    return outcomes


def _csa_city_preds(results_dir: Path, spec, years: list[int]) -> pd.DataFrame:
    """Building-level predictions for one city.

    NYC keeps reading the whole-city zarr pass at the top of ``results_dir``;
    every other city reads the dense per-city pass that
    :class:`src.csa_predict.CSAPredictionRunner` writes to
    ``<results>/csa/<city>/``. Those are different sensors and different
    sampling designs, so they are deliberately different directories rather
    than one pooled pile of CSVs.
    """
    city_dir = Path(results_dir) / "csa" / spec.key
    if city_dir.exists():
        bld = _load_building_preds_us(city_dir, years)
        if not bld.empty:
            out = bld.rename(columns={"pred": "predicted_value"})
            out["GEOID_str"] = out["GEOID"]
            return out
        print(f"    {city_dir} exists but holds no predictions for {years}")
    return pd.DataFrame()


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


def _csa_cut_label(thresh: float, control_threshold: float | None) -> str:
    """Figure label naming BOTH sides of the contrast, not just the treated cut.

    "5% of baseline area" hides the thing the pinned control group fixes: which
    tracts the treated ones are being compared *to*. Once the comparison group
    is held at a different cut from the treated group, a column heading that
    names only the treated cut is no longer a full description of the estimate.
    """
    treated = rf"$>${_tex_pct(thresh)} of baseline area"
    if control_threshold is None:
        return treated
    return treated + rf" vs $\leq${_tex_pct(control_threshold)}"


def _csa_antic_note(anticipation: int, spec, city_years) -> str:
    """Figure annotation for the arm, in periods AND that city's years.

    A period is 2 years in NYC and 1 in an annual ortho panel, so "anticipation
    = 1" alone does not tell a reader how much baseline gap the estimate has.
    Empty for a=0, where there is nothing to declare.
    """
    if not anticipation:
        return ""
    yrs = ces.PeriodMap.from_years(city_years).event_time_years(anticipation)
    return (rf"(adoption {anticipation} period"
            rf"{'' if anticipation == 1 else 's'} $\approx$ {yrs:.0f} yr early) ")


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
           n_boot: int = 10_000,
           csa_results_dir: Path | None = None,
           max_undated_area_share: float | None = None,
           control_threshold: float | None = ces.CONTROL_THRESHOLD,
           anticipation: int = 0,
           out_tag: str = "") -> dict:
    """Callaway–Sant'Anna event study on construction cohorts (issue #36).

    For each city with a footprint + year-built table on disk: date tracts into
    construction cohorts by cumulative new building area, then estimate the
    dynamic ATT of that construction on the model's tract-mean prediction.

    Every output goes under ``<out>/csa_anticipation{anticipation}/``, so one
    call is one complete, self-contained set of results and the arms can be
    compared folder against folder rather than row against row.

    Emits (all relative to that arm folder)
    ---------------------------------------
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

    ``anticipation`` moves every treated tract's effective adoption date that
    many panel periods earlier, for the whole arm — not just for one robustness
    row. Cohorts are dated from ``year_built`` = completion while site clearance
    and superstructure are visible from the air earlier, so under ``a=0`` the
    last pre-treatment periods are partly treated. Treated tracts left without a
    pre-period by the shift are dropped, never reclassified as controls. See
    :data:`CSA_ANTICIPATION_ARMS` for which arm each city can support.

    ``results_dir`` holds NYC's whole-city zarr pass. ``csa_results_dir`` (the
    *run's* results dir) is where the other cities' dense per-city passes live,
    under ``csa/<city>/`` — a different directory because they are a different
    sensor and a different sampling design. Defaults to ``results_dir``.

    ``control_threshold`` pins the never-treated comparison group across the
    whole threshold grid: every column contrasts "built more than ``threshold``"
    against the same tracts — those that never crossed ``control_threshold`` —
    and the tracts in between are dropped rather than handed to the control
    group. ``None`` restores the old complement-of-treated control group, which
    changes with every column and makes the grid unreadable as a dose-response
    (see :data:`src.csa_event_study.CONTROL_THRESHOLD` for the measured size of
    that effect).

    ``max_undated_area_share`` enables the undated-area sample restriction (see
    :data:`src.csa_event_study.MAX_UNDATED_AREA_SHARE`); ``None`` is the
    unrestricted analysis. ``out_tag`` suffixes every figure, table and headline key
    this call writes, so the restricted and unrestricted arms can both be run
    against one output directory without overwriting each other — which is the
    point, since the pair *is* the robustness evidence for the restriction.

    Returns a headline dict for the run summary.
    """
    from src import csa_event_study as ces

    # Two orthogonal arm dimensions, kept in different namespaces on purpose:
    # anticipation gets a FOLDER (a whole set of results, read on its own), the
    # undated screen keeps its filename SUFFIX (a paired robustness comparison,
    # read side by side with its unrestricted twin). Combining them lets both
    # vary without either having to know about the other.
    anticipation = int(anticipation)
    arm = out / f"csa_anticipation{anticipation}"
    (arm / "figures").mkdir(parents=True, exist_ok=True)
    (arm / "tables").mkdir(parents=True, exist_ok=True)
    ns = f"csa{out_tag}/a{anticipation}"

    print(f"\n=== Part D: Event study — construction-cohort CSA "
          f"(anticipation = {anticipation} period"
          f"{'' if anticipation == 1 else 's'}) ===")
    print(f"  writing to {arm}")
    if anticipation:
        print(f"  every treated tract adopts {anticipation} period(s) before its "
              f"recorded completion year; treated tracts left without a "
              f"pre-period are dropped, not made controls")
    if control_threshold is not None:
        print(f"  control group pinned: never-treated means a tract never crossed "
              f"{control_threshold:.0%} of baseline area; tracts between that and "
              f"each column's threshold are dropped, not used as controls")
    else:
        print("  ⚠️ control group NOT pinned — the never-treated group is the "
              "complement of the treated group and changes with every threshold")
    if max_undated_area_share is not None:
        print(f"  undated-area screen: dropping tracts with more than "
              f"{max_undated_area_share:.0%} of baseline area undated")

    years = sorted(years) if years else _available_years(results_dir) or list(YEARS)
    thresholds = tuple(thresholds) if thresholds else ces.DEFAULT_THRESHOLDS
    headline_t = (
        ces.HEADLINE_THRESHOLD if ces.HEADLINE_THRESHOLD in thresholds
        else thresholds[len(thresholds) // 2]
    )
    if control_threshold is not None and control_threshold > min(thresholds):
        raise ValueError(
            f"control_threshold {control_threshold:.0%} exceeds the smallest "
            f"threshold {min(thresholds):.0%}; the pinned control group must sit "
            f"at or below every treated cut in the grid"
        )
    print(f"  panel years: {years}")
    print(f"  thresholds : {[f'{t:.0%}' for t in thresholds]} "
          f"(headline {headline_t:.0%})")

    ready, missing = ces.available_cities(processed_dir)
    for spec, reason in missing:
        print(f"  ⏭️  {spec.label}: {reason}")
    if not ready:
        print("  no city has a footprint + year-built table — Part D skipped.")
        return {f"{ns}/status": "no_cohort_source"}

    split_of = _csa_split_of(processed_dir)
    # NYC's whole-city pass sits at the top of results_dir; the other cities have
    # their own per-city directories. Loaded lazily per city, because each city
    # now has its own panel years and reading one pooled frame would force a
    # single year grid back on everyone.
    #
    # Lazy in a second sense too: this frame is millions of rows, and a run where
    # every city has its own dense pass never needs it at all. Reading it up
    # front paid that cost on every arm regardless.
    _shared_bld: list = []

    def shared_bld_fn() -> pd.DataFrame:
        if not _shared_bld:
            _shared_bld.append(_csa_building_preds(results_dir, years))
        return _shared_bld[0]

    csa_root = Path(csa_results_dir) if csa_results_dir is not None else Path(results_dir)

    rows: list[dict] = []
    headline: dict = {}
    panels_by_city: dict[str, tuple[str, object]] = {}

    for spec in ready:
        print(f"\n  --- {spec.label} ---")
        city_years = spec.years(years)
        baseline_year = spec.baseline(CSA_BASELINE_YEAR)
        print(f"    panel: {city_years} (baseline {baseline_year}, "
              f"sensor {spec.sensor})")

        per_building, tracts, construction_year = _csa_city_geometry(
            processed_dir, spec
        )
        if tracts.empty:
            print("    no tracts matched this city; skipping")
            continue

        outcomes = _csa_city_outcomes(
            csa_root, results_dir, processed_dir, spec, city_years,
            baseline_year, tracts, construction_year, shared_bld_fn,
        )
        if outcomes.empty:
            print("    no building-level predictions inside this city; skipping")
            continue
        outcome_cols = ["pred_all"] + (
            ["pred_incumbent"] if "pred_incumbent" in outcomes.columns else []
        )
        # Copy before stamping the split group: `outcomes` is the cached frame,
        # shared with every other arm, and must not be mutated in place.
        outcomes = outcomes.copy()
        outcomes["split_group"] = outcomes["GEOID_str"].map(split_of)
        print(f"    {outcomes['GEOID_str'].nunique():,} tracts with predictions; "
              f"outcomes: {outcome_cols}")

        # Cohorts depend only on footprints, so build once per threshold and
        # reuse across split groups / outcomes / control groups. The footprint
        # to tract assignment underneath does not depend on the threshold
        # either, so it is hoisted out of this loop entirely — see
        # `_csa_city_geometry` and `ces.footprint_tract_areas`.
        cohorts_by_t = {}
        for thresh in thresholds:
            coh = ces.build_tract_cohorts(
                None, tracts, city_years,
                baseline_year=baseline_year, threshold=thresh,
                year_col=spec.year_col, demolition_col=spec.demolition_col,
                area_epsg=spec.area_epsg,
                max_undated_area_share=max_undated_area_share,
                control_threshold=control_threshold,
                per_building=per_building,
            )
            cohorts_by_t[thresh] = coh
            s = coh.summary()
            print(f"    {thresh:>5.0%}: {s['n_treated']:,} treated / "
                  f"{s['n_never_treated']:,} never-treated tracts "
                  f"({s['n_dropped_no_baseline']:,} dropped, no baseline stock"
                  + (f"; {s['n_dropped_undated']:,} dropped by the undated screen"
                     if max_undated_area_share is not None else "")
                  + (f"; {s['n_dropped_ambiguous']:,} dropped as ambiguous, "
                     f"built between {control_threshold:.0%} and {thresh:.0%}"
                     if control_threshold is not None else "") + ")")
            headline[f"{ns}/{spec.key}/n_dropped_undated"] = s["n_dropped_undated"]
            headline[f"{ns}/{spec.key}/{thresh:.0%}/n_dropped_ambiguous"] = \
                s["n_dropped_ambiguous"]

        def _panel(thresh: float, group: str, outcome_col: str):
            """Every panel in this arm carries the arm's anticipation.

            It is not a parameter here on purpose: the whole point of the arm
            structure is that one folder is one consistent design, so a figure
            in it cannot silently be built at a different adoption date from the
            table beside it.
            """
            sub = outcomes
            if group is not None:
                sub = sub[sub["split_group"] == group]
            if sub.empty:
                return None
            return ces.build_event_panel(
                sub, cohorts_by_t[thresh].cohorts,
                outcome_col=outcome_col, panel_years=city_years,
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
                            panels_by_city[spec.key] = (spec.label, res)
                            headline[f"{ns}/{spec.key}/pretrend_p"] = res.pretrend_supt_p
                            headline[f"{ns}/{spec.key}/overall_att"] = res.overall_att
                            headline[f"{ns}/{spec.key}/n_treated"] = res.n_treated
                if r == 0:
                    ax.set_title(_csa_cut_label(thresh, control_threshold),
                                 fontsize=8)
                if r == len(groups) - 1:
                    ax.set_xlabel("Event time (periods since construction)", fontsize=8)
                if c == 0:
                    ax.set_ylabel(f"{CSA_SPLIT_LABELS.get(group, 'All')}\nATT "
                                  "(tract mean prediction)", fontsize=7.5)
        fig.suptitle(f"{spec.label}: construction-cohort event study "
                     f"{_csa_antic_note(anticipation, spec, city_years)}"
                     "(shaded = 95\\% simultaneous CI)", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        _savefig(fig, arm / "figures" / f"D_event_study_thresholds_{spec.key}{out_tag}.pdf")

        # ── composition check: all buildings vs incumbents only ──────────────
        if "pred_incumbent" in outcome_cols:
            fig2, ax2 = plt.subplots(figsize=FIG_SIZE_ONE_COL)
            drawn = 0
            for i, (oc, lab, col) in enumerate([
                ("pred_all", "All buildings", _CSA_ATT_COLOR),
                ("pred_incumbent", "Incumbents only (built $\\leq$ "
                 f"{baseline_year})", _CSA_ALT_COLOR),
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
                    headline[f"{ns}/{spec.key}/incumbent_overall_att"] = res.overall_att
            if drawn:
                ax2.set_xlabel("Event time (periods since construction)", fontsize=8)
                ax2.set_ylabel("ATT (tract mean prediction)", fontsize=8)
                ax2.set_title(f"{spec.label}: composition check "
                              f"({_csa_cut_label(headline_t, control_threshold)}, "
                              "held out)", fontsize=8.5)
                ax2.legend(fontsize=6.5, loc="lower right")
                fig2.tight_layout()
                _savefig(fig2, arm / "figures" / f"D_event_study_composition_{spec.key}{out_tag}.pdf")
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

        # The anticipation robustness rows that used to live here are gone: this
        # whole call IS one anticipation setting, so every figure and every row
        # above already carries it. Re-running a=1 and a=2 inside the a=0 arm
        # would estimate them twice and, worse, put three different adoption
        # dates in one table under one folder.
        #
        # `pretrend_supt_p` can be None in the higher arms — a shift that leaves
        # no pre-period to test is a real outcome on a four-period panel like San
        # Antonio's, not an error — so the summary below must not format it
        # unconditionally.
        if spec.key in panels_by_city:
            head = panels_by_city[spec.key][1]
            p = head.pretrend_supt_p
            print(f"    headline ({headline_t:.0%}, a={anticipation}): "
                  f"pre-trend p = {'n/a' if p is None else f'{p:.4f}'}, "
                  f"overall ATT = {head.overall_att:+.4f}, "
                  f"{head.n_treated} treated / {head.n_never_treated} control")

    # ── main-body figure: the deep-dive + clean zero-shot pair ──────────────
    # Issue #36 wants the headline to be NYC (rich, deepest panel) plus one clean
    # zero-shot city, and to stay shippable whether or not the optional cities
    # ran. Chicago was the intended second panel but its dated-footprint source
    # is a frozen 2015 snapshot, so MAIN_FIGURE_CITIES pins NYC + Tampa; Seattle
    # and Baltimore are annex holdouts. The main figure is pinned to that tuple
    # in that order, and everything else goes to the annex. Falling back to
    # whatever estimated would silently change which cities carry the paper's
    # central claim depending on which parquets happened to be on disk.
    main_panels = [panels_by_city[k] for k in ces.MAIN_FIGURE_CITIES
                   if k in panels_by_city]
    annex_only = [k for k in panels_by_city if k not in ces.MAIN_FIGURE_CITIES]
    if not main_panels and panels_by_city:
        print(f"  ⚠️ none of MAIN_FIGURE_CITIES {ces.MAIN_FIGURE_CITIES} "
              f"estimated; falling back to {sorted(panels_by_city)}")
        main_panels = list(panels_by_city.values())
    if annex_only:
        print(f"  annex-only cities: {', '.join(sorted(annex_only))}")

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
            "New construction raises predicted wealth "
            f"({_csa_cut_label(headline_t, control_threshold)}, held-out tracts"
            + (f", anticipation {anticipation})" if anticipation else ")"),
            fontsize=9,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        _savefig(fig, arm / "figures" / f"D_event_study_main{out_tag}.pdf")
    else:
        print("  ⚠️ no city produced an estimable held-out event study — "
              "no main figure written.")

    # ── annex: pooled curve over a common event-time window ─────────────────
    # Only meaningful with >=2 cities, and only over the window they share:
    # cadences differ (Chicago annual, NYC/Seattle/Nashville biennial), so
    # averaging raw event times would pool k=+3 meaning "3 years" in one city
    # and "6 years" in another. Note in the caption that pooling across cities
    # also pools across *sensors* — an implicit sensor-invariance claim.
    if len(panels_by_city) >= 2:
        results = {k: r for k, (_, r) in panels_by_city.items()}
        window = ces.pooled_common_window(results)
        if window is None:
            print("  pooled curve skipped: no common event-time window")
        else:
            lo, hi = window
            figp, axp = plt.subplots(figsize=FIG_SIZE_ONE_COL)
            for i, (key, (label, res)) in enumerate(sorted(panels_by_city.items())):
                ks = np.asarray(res.event_times, dtype=float)
                keep = (ks >= lo) & (ks <= hi)
                axp.plot(ks[keep], np.asarray(res.att)[keep], "o-", ms=3, lw=1.2,
                         label=label, alpha=0.85)
            axp.axhline(0, color="0.55", lw=0.6)
            axp.axvline(-0.5, color="black", ls="--", lw=0.8)
            axp.set_xlabel("Event time (periods since construction)", fontsize=8)
            axp.set_ylabel("ATT (tract mean prediction)", fontsize=8)
            axp.set_title(f"Held-out cities on the common window "
                          f"$k \\in [{lo}, {hi}]$", fontsize=8.5)
            axp.legend(fontsize=6.5)
            figp.tight_layout()
            _savefig(figp, arm / "figures" / f"D_event_study_pooled{out_tag}.pdf")
            headline[f"{ns}/pooled_window"] = f"[{lo},{hi}]"

    # ── coefficient table ───────────────────────────────────────────────────
    if rows:
        tab = pd.DataFrame(rows)
        # `label` and `n_units` are internal bookkeeping: label just concatenates
        # city/split/threshold/outcome, and n_units is n_treated + n_never_treated.
        tab = tab.drop(columns=[c for c in ("label", "n_units") if c in tab.columns])
        tab["threshold"] = tab["threshold"].map(lambda t: f"{t:.0%}")
        # Which tracts the treated ones were compared to. Without this a 5% row
        # from a pinned run and a 5% row from an unpinned one are different
        # estimands printed under the same label.
        tab["control_cut"] = ("complement" if control_threshold is None
                              else f"{control_threshold:.0%}")
        # Stamp the sample restriction on every row: the restricted and
        # unrestricted tables are read side by side, and a coefficient without
        # its sample definition attached is the easiest thing to misattribute.
        tab["undated_screen"] = ("none" if max_undated_area_share is None
                                 else f"{max_undated_area_share:.0%}")
        lead = ["city", "split", "threshold", "control_cut", "undated_screen",
                "outcome",
                "control", "anticipation",
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
                tab, arm / "tables", f"D_event_study_coefficients{out_tag}",
                "Construction-cohort Callaway--Sant'Anna event study. A tract is "
                "treated once cumulative new building area exceeds "
                "\\emph{threshold} of its baseline stock; the never-treated "
                "comparison group is held fixed across thresholds at tracts that "
                "never exceeded \\emph{control cut}, and tracts falling between "
                "the two are dropped rather than used as controls. Each row "
                "reports the joint pre-trend test (sup-t over all pre-treatment "
                "event times, with a Wald $\\chi^2$ alternative) and dynamic "
                "ATTs at successive horizons $k$.",
                "tab:csa_event_study",
            )
        except Exception as exc:
            print(f"    LaTeX table export failed ({exc}); writing CSV only")
            tab.to_csv(
                arm / "tables" / f"D_event_study_coefficients{out_tag}.csv", index=False
            )
        headline[f"{ns}/n_specifications"] = len(rows)
    else:
        print("  ⚠️ no specification estimated — no coefficient table written.")
        headline.setdefault(f"{ns}/status", "not_estimable")

    # An arm is a self-contained set of results, so nothing it drew should still
    # be open when the next one starts. `_savefig` closes what it writes, but a
    # figure whose city raised part-way through never reaches it, and matplotlib
    # holds every un-closed figure in a global registry for the life of the
    # process.
    plt.close("all")
    _csa_release_memory()
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


def _figure_test_sample(ctx: _USContext, what: str = "figure",
                        min_tracts: int = _MIN_FIGURE_TRACTS) -> dict | None:
    """The test-city sample shared by the main figure and the growth figure.

    Test-split CBSAs (from building-level ``type == 'test'``), restricted to
    (CBSA, year) cells with at least ``min_tracts`` predicted tracts. Returns
    ``{'test_bld', 'tract_test', 'cells_tract'}`` or None (with a printed
    reason) when nothing survives. One function so both figures provably see
    the same cells.
    """
    if ctx.bld.empty or not len(ctx.tract):
        print(f"  missing building or tract predictions; skipping {what}")
        return None

    test_bld_all = ctx.bld[ctx.bld["type"] == "test"]
    if test_bld_all.empty:
        print(f"  no test-split building predictions; skipping {what}")
        return None

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
        print(f"  no usable tract-level within-city cells; skipping {what}")
        return None

    # Drop individual (CBSA, year) cells too thin to give a stable Spearman
    # rho, before building any panel, so every panel sees the same cells.
    # This is a per-cell filter (not a whole-city one): a city with plenty of
    # tracts overall can still have one sparse year dropped while its other
    # years remain.
    cells_tract = cells_tract_all[cells_tract_all["n_tracts"] >= min_tracts]
    n_cy_dropped = len(cells_tract_all) - len(cells_tract)
    if n_cy_dropped:
        print(f"  dropping {n_cy_dropped} / {len(cells_tract_all)} city-year cell(s) with "
              f"< {min_tracts} tracts")
    if not len(cells_tract):
        print(f"  no city-year cells meet the tract-count threshold; skipping {what}")
        return None

    eligible_pairs = pd.MultiIndex.from_frame(cells_tract[["cbsa", "year"]])
    test_bld = test_bld_all[
        pd.MultiIndex.from_frame(test_bld_all[["cbsa", "year"]]).isin(eligible_pairs)
    ].reset_index(drop=True)
    tract_test = tract_test_all[
        pd.MultiIndex.from_frame(tract_test_all[["cbsa", "year"]]).isin(eligible_pairs)
    ].reset_index(drop=True)
    return {"test_bld": test_bld, "tract_test": tract_test, "cells_tract": cells_tract}


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

    sample = _figure_test_sample(ctx, what="main figure")
    if sample is None:
        return {}
    test_bld, tract_test, cells_tract = (
        sample["test_bld"], sample["tract_test"], sample["cells_tract"])
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


# ─── Long-difference growth figure (US) ───────────────────────────────────────
# Does the model predict *change*, not just levels? Analog of Khachiyan et al.
# (2022, AER: Insights) Fig. 2 bottom row, on the main figure's test sample.
# Each tract is paired (t0, t1): t1 = its latest observed year, t0 = the earlier
# year minimising |t - (t1 - _GROWTH_TARGET_GAP)|. Realised gaps are bounded by
# the imagery span of the run (2016-2023 for the first NAIP run → 5-7 years).
# Predicted and actual ranks are converted to wealth dollars through each
# city-year's GB2 (Part C machinery) before differencing.

_GROWTH_TARGET_GAP = 10
_GROWTH_N_BOOT = 1000
# Khachiyan et al. (2022) Table 2, Income, *without* initial conditions — the
# out-of-sample R^2 of predicted vs actual change in log total personal income
# on 2.4 km cells (plus the in-sample 2000-2010 change for reference).
_KHACHIYAN_INCOME_R2 = (
    ("2000-2010 (in-sample)", 0.4331),
    ("2007-2017 (out-of-sample)", -0.0999),
    ("2000-2017 (out-of-sample)", 0.3731),
)


def _zscore_within(df: pd.DataFrame, col: str, by=("cbsa", "year")) -> pd.Series:
    """``col`` standardised within each ``by`` group (mean 0, sd 1).

    Puts the model's arbitrary ordinal scale on the label's per-(CBSA, year)
    z-score scale using only the predictions themselves — no label information
    enters, so a prediction built this way stays out-of-sample. Groups with no
    spread return NaN rather than inf.
    """
    g = df.groupby(list(by))[col]
    sd = g.transform("std")
    return (df[col] - g.transform("mean")) / sd.where(sd > 0)


def _long_difference_pairs(tract_df: pd.DataFrame, target_gap: int = _GROWTH_TARGET_GAP,
                           value_cols=("label", "pred")) -> pd.DataFrame:
    """One (t0, t1) long-difference row per tract.

    t1 is the tract's latest year; t0 is the earlier year closest to
    ``t1 - target_gap`` (ties -> the earlier year, i.e. the longer horizon).
    Tracts observed in a single year are dropped. Returns GEOID, cbsa, t0, t1,
    gap, ``{c}_0``, ``{c}_1`` and ``d_{c} = {c}_1 - {c}_0`` for each value col.
    """
    cols = ["GEOID", "cbsa", "year", *value_cols]
    if tract_df.duplicated(["GEOID", "year"]).any():
        raise ValueError("tract_df has duplicate (GEOID, year) rows")
    obs = tract_df[cols]
    t1 = obs.groupby("GEOID")["year"].max().rename("t1")
    cand = obs[["GEOID", "year"]].merge(t1, on="GEOID")
    cand = cand[cand["year"] < cand["t1"]].copy()
    cand["dist"] = (cand["year"] - (cand["t1"] - target_gap)).abs()
    pick = (cand.sort_values(["GEOID", "dist", "year"])
                .drop_duplicates("GEOID")[["GEOID", "year", "t1"]]
                .rename(columns={"year": "t0"}))
    vals = obs.set_index(["GEOID", "year"])[list(value_cols)]
    v0 = vals.loc[list(zip(pick["GEOID"], pick["t0"]))].add_suffix("_0").reset_index(drop=True)
    v1 = vals.loc[list(zip(pick["GEOID"], pick["t1"]))].add_suffix("_1").reset_index(drop=True)
    cbsa = obs.drop_duplicates("GEOID").set_index("GEOID")["cbsa"]
    pairs = pick.reset_index(drop=True)
    pairs.insert(1, "cbsa", cbsa.loc[pairs["GEOID"]].values)
    pairs["gap"] = pairs["t1"] - pairs["t0"]
    pairs = pd.concat([pairs, v0, v1], axis=1)
    for c in value_cols:
        pairs[f"d_{c}"] = pairs[f"{c}_1"] - pairs[f"{c}_0"]
    return pairs


def _r2_oos(y: np.ndarray, yhat: np.ndarray) -> float:
    """1 - SSE/SST with the prediction taken as-is (no refit) — the
    out-of-sample R^2 Khachiyan et al. report; negative when the prediction
    does worse than the sample mean of ``y``."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - yhat) ** 2) / sst) if sst > 0 else float("nan")


def _r2_ols(y: np.ndarray, x: np.ndarray) -> float:
    """R^2 of y on x with an intercept (= Pearson r^2): the fit after an
    ex-post linear calibration of x — an upper bound on any out-of-sample R^2."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    if len(y) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def _growth_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    """n, out-of-sample R^2, OLS R^2, Pearson r and Spearman rho of yhat vs y."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    if len(y) < 3 or np.ptp(y) == 0 or np.ptp(yhat) == 0:
        return {"n": int(len(y)), "r2_oos": float("nan"), "r2_ols": float("nan"),
                "pearson": float("nan"), "spearman": float("nan")}
    return {"n": int(len(y)), "r2_oos": _r2_oos(y, yhat), "r2_ols": _r2_ols(y, yhat),
            "pearson": float(np.corrcoef(y, yhat)[0, 1]),
            "spearman": float(spearmanr(y, yhat).statistic)}


def _cluster_bootstrap_ci(y: np.ndarray, yhat: np.ndarray, clusters: np.ndarray,
                          stat_fn, n_boot: int = _GROWTH_N_BOOT, seed: int = 0,
                          alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of ``stat_fn(y, yhat)`` resampling whole clusters (CBSAs).

    Tracts within a city share the city's NAIP flight, label vintage and
    z-score reference set, so they are not independent; resampling cities is
    the honest unit.
    """
    y, yhat, clusters = np.asarray(y, float), np.asarray(yhat, float), np.asarray(clusters)
    idx_by_c = [np.flatnonzero(clusters == c) for c in pd.unique(clusters)]
    if len(idx_by_c) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        take = rng.integers(0, len(idx_by_c), len(idx_by_c))
        ii = np.concatenate([idx_by_c[k] for k in take])
        s = stat_fn(y[ii], yhat[ii])
        if np.isfinite(s):
            stats.append(s)
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _demean_within(values: pd.Series, groups: pd.Series) -> pd.Series:
    """``values`` minus its group mean — strips between-city level shifts."""
    return values - values.groupby(groups).transform("mean")


# BLS CPI-U, U.S. city average, all items (series CUUR0000SA0), annual
# averages. ACS 5-year dollar estimates are in the nominal dollars of the
# vintage's final year; deflating to one base year keeps ~25% inflation
# between 2016 and 2023 from masquerading as wealth growth proportional to
# each tract's level (which any levels-only model would "predict").
_CPI_U_ANNUAL = {
    2010: 218.056, 2011: 224.939, 2012: 229.594, 2013: 232.957, 2014: 236.736,
    2015: 237.017, 2016: 240.007, 2017: 245.120, 2018: 251.107, 2019: 255.657,
    2020: 258.811, 2021: 270.970, 2022: 292.655, 2023: 304.702,
}
_GROWTH_DOLLAR_BASE_YEAR = 2023


def _deflate(values, year: int, base: int = _GROWTH_DOLLAR_BASE_YEAR):
    """Nominal ``year`` dollars -> constant ``base``-year dollars (CPI-U)."""
    return values * (_CPI_U_ANNUAL[base] / _CPI_U_ANNUAL[int(year)])


def _wealth_dollar_column(indicator: str, year: int) -> str:
    """Panel column holding the dollar variable behind ``indicator``'s label
    (the label is the per-CBSA z-score of its log): W-family tokens -> the
    wealth index (e.g. 'W2_i_r5pct_2016'), 'inc' -> per-capita income."""
    var = indicators.token_to_var(indicator)
    return f"{var}_{year}" if var else f"per_capita_income_usd_{year}"


def _load_wealth_dollars_long(processed_dir: Path, indicator: str, years) -> pd.DataFrame:
    """Long [GEOID, cbsa, year, usd] of the indicator's dollar variable for
    every panel tract, in constant base-year dollars. Years whose column is
    absent from the panel are skipped."""
    import pyarrow as pa
    path = processed_dir / _PANEL_FILENAME
    present = set(pa.ipc.open_file(path).schema.names)
    years = [int(y) for y in sorted(set(years))
             if _wealth_dollar_column(indicator, y) in present and int(y) in _CPI_U_ANNUAL]
    cols = [_PANEL_GEOID_COL, "cbsa_code"] + [_wealth_dollar_column(indicator, y) for y in years]
    panel = pd.read_feather(path, columns=cols)
    panel = panel.rename(columns={_PANEL_GEOID_COL: "GEOID"})
    panel["GEOID"] = panel["GEOID"].astype(str).str.zfill(11)
    panel["cbsa_code"] = panel["cbsa_code"].astype(str)
    panel = panel.drop_duplicates("GEOID")
    frames = [
        pd.DataFrame({"GEOID": panel["GEOID"].values, "cbsa": panel["cbsa_code"].values,
                      "year": y,
                      "usd": _deflate(panel[_wealth_dollar_column(indicator, y)]
                                      .values.astype(float), y)})
        for y in years
    ]
    return pd.concat(frames, ignore_index=True)


def _fit_city_gb2(wealth_long: pd.DataFrame, cbsas, min_obs: int = 10,
                  smooth: bool = False) -> dict:
    """{cbsa: {year: (c, p, q, scale)}} — GB2 MLE on each city-year's tract
    dollar distribution (all panel tracts of the CBSA, positive values).
    City-years that fail to fit are dropped (with a note).

    ``smooth=True`` applies Part C's log-space polynomial smoothing of the
    parameters across years. Off by default: on W2 wealth several cities fit
    a near-degenerate shape (large c, tiny p and q) where the four parameters
    trade off almost perfectly, so smoothing each one independently yields
    combinations describing a different distribution — measured on
    run_20260722, CBSA 23420 (Fresno) 2016 smoothed P50 = $1.39M vs
    empirical $307k (raw fit: $314k). Raw per-year fits on 100+ tracts track
    the empirical P10/P50/P90 to within ~0.1 log points everywhere.
    """
    out, failed = {}, []
    sub = wealth_long[wealth_long["cbsa"].isin({str(c) for c in cbsas})]
    for cbsa, g_city in sub.groupby("cbsa"):
        raw = {}
        for yr, g in g_city.groupby("year"):
            v = g["usd"].values.astype(float)
            v = v[np.isfinite(v) & (v > 0)]
            if len(v) < min_obs:
                continue
            try:
                raw[int(yr)] = _fit_gb2(v)
            except (ValueError, RuntimeError):
                failed.append((cbsa, int(yr)))
        if raw:
            out[str(cbsa)] = _smooth_gb2_params(sorted(raw), raw) if smooth else raw
    if failed:
        print(f"    GB2 fit failed for {len(failed)} city-year(s): {failed[:5]}")
    return out


def _gb2_fit_error(wealth_long: pd.DataFrame, params_by_city: dict, cells,
                   qs=(0.1, 0.5, 0.9)) -> pd.DataFrame:
    """Per (cbsa, year) in ``cells``: max |log(GB2 quantile / empirical
    quantile)| over ``qs`` — the check that the dollar map is faithful to the
    city's actual ACS distribution in the years it is used."""
    rows = []
    for cbsa, yr in cells:
        params = params_by_city.get(str(cbsa), {}).get(int(yr))
        v = wealth_long.loc[(wealth_long["cbsa"] == str(cbsa))
                            & (wealth_long["year"] == int(yr)), "usd"].values.astype(float)
        v = v[np.isfinite(v) & (v > 0)]
        if params is None or len(v) == 0:
            continue
        err = np.abs(np.log(_gb2_quantile(qs, params) / np.quantile(v, qs)))
        rows.append({"cbsa": str(cbsa), "year": int(yr), "max_log_err": float(err.max())})
    return pd.DataFrame(rows, columns=["cbsa", "year", "max_log_err"])


def _hazen_within(df: pd.DataFrame, col: str, by=("cbsa", "year")) -> pd.Series:
    """Hazen plotting position (rank - 0.5) / n of ``col`` within each group —
    the empirical CDF ``_gb2_apply_from_ranks`` feeds into the GB2 quantile."""
    from scipy.stats import rankdata
    return df.groupby(list(by))[col].transform(
        lambda s: pd.Series((rankdata(s) - 0.5) / len(s), index=s.index))


def _gb2_quantile(probs, params: tuple, clip_eps: float = 1e-6) -> np.ndarray:
    """GB2 quantile at ``probs`` for (c, p, q, scale), clipped like
    ``_gb2_apply_from_ranks`` so the extreme Hazen positions stay finite."""
    c, p, q, scale = params
    probs = np.clip(np.asarray(probs, float), clip_eps, 1.0 - clip_eps)
    return gb2.ppf(probs, c, p, q, loc=0, scale=scale)


def _gb2_dollars(df: pd.DataFrame, prob_col: str, params_by_city: dict,
                 year_col: str = "year") -> np.ndarray:
    """Dollar value of each row's ``prob_col`` under its (cbsa, ``year_col``)
    GB2. Rows whose city-year has no fit get NaN."""
    out = np.full(len(df), np.nan)
    keys = df[["cbsa", year_col]].astype({year_col: int}).itertuples(index=False, name=None)
    probs = df[prob_col].values.astype(float)
    for i, (cbsa, yr) in enumerate(keys):
        params = params_by_city.get(str(cbsa), {}).get(yr)
        if params is not None and np.isfinite(probs[i]):
            out[i] = _gb2_quantile(probs[i], params)
    return out


def _central_limits(v: np.ndarray, frac: float) -> tuple[float, float]:
    """[lo, hi] covering the central ``frac`` of ``v`` (frac=1 -> full range)."""
    margin = 100 * (1 - frac) / 2
    lo, hi = np.percentile(v, [margin, 100 - margin])
    return float(lo), float(hi)


def _panel_growth_scatter(ax: plt.Axes, x: np.ndarray, y: np.ndarray, *, letter: str,
                          xlabel: str, ylabel: str, r2_text: str, n_bins: int = 10,
                          central_frac: float = _PANEL_C_CENTRAL_FRAC) -> int:
    """Khachiyan-style actual-vs-predicted change scatter: marker-size-1 cloud,
    decile conditional-mean line (as in main-figure Panel C) and a 45-degree
    reference line. Axes are clipped to the central ``central_frac`` of each
    marginal; returns the number of off-canvas points for the caption.
    Both axes share one range (as in Khachiyan et al.'s Fig. 2), so the 45-degree
    line is a true diagonal and a near-constant prediction reads as the flat
    band it is rather than being stretched to fill the panel.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < n_bins * 3:
        ax.text(0.5, 0.5, "insufficient tract pairs", ha="center", va="center",
                transform=ax.transAxes)
        _panel_title(ax, letter)
        return 0
    ax.scatter(x, y, s=1, alpha=0.3, color="steelblue", linewidths=0, zorder=1)
    x_lo, x_hi = _central_limits(x, central_frac)
    y_lo, y_hi = _central_limits(y, central_frac)
    lo, hi = min(x_lo, y_lo), max(x_hi, y_hi)
    pad = 0.04 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.plot([lo, hi], [lo, hi], color="0.35", linestyle="--", linewidth=1.0, zorder=2,
            label=r"45$^\circ$")
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    bins = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    bx = [x[bins == b].mean() for b in range(n_bins) if (bins == b).any()]
    by = [y[bins == b].mean() for b in range(n_bins) if (bins == b).any()]
    ax.plot(bx, by, "o-", color="firebrick", markeredgecolor="white", linewidth=1.8,
            markersize=5, zorder=3, label=f"{n_bins}-bin conditional mean")
    ax.axhline(0, color="0.8", linewidth=0.6, zorder=0)
    ax.axvline(0, color="0.8", linewidth=0.6, zorder=0)
    ax.text(0.04, 0.96, r2_text, transform=ax.transAxes, fontsize=6.5, va="top",
            linespacing=1.5, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
    ax.legend(fontsize=6, loc="lower right", frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _panel_title(ax, letter)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return int(np.sum((x < x_lo) | (x > x_hi) | (y < y_lo) | (y > y_hi)))


def _metric_row(actual: pd.Series, predicted: pd.Series, clusters: pd.Series, *,
                target: str, predictor: str, scope: str, period: str,
                n_boot: int = _GROWTH_N_BOOT) -> dict:
    """One metrics-table row: R^2_oos (with city-cluster CI), R^2_OLS, r, rho.

    ``scope='within-city'`` demeans actual and predicted by CBSA first, which
    strips each city's average growth — a component the GB2 maps take from
    that city's own ACS distribution, not from the imagery.
    """
    a, p = actual.astype(float), predicted.astype(float)
    m = np.isfinite(a.values) & np.isfinite(p.values)
    a, p, cl = a[m], p[m], clusters[m]
    if scope == "within-city":
        a, p = _demean_within(a, cl), _demean_within(p, cl)
    met = _growth_metrics(a.values, p.values)
    lo, hi = _cluster_bootstrap_ci(a.values, p.values, cl.values, _r2_oos, n_boot=n_boot)
    return {"source": "this paper", "target": target, "predictor": predictor,
            "scope": scope, "period": period, **met, "r2_oos_ci_lo": lo, "r2_oos_ci_hi": hi}


def _growth_annotation(model: dict, model_w: dict, frozen: dict, frozen_w: dict) -> str:
    """Panel text: model R^2_oos [CI] pooled and within-city, the rank-frozen
    benchmark's pair, then rho and n."""
    return "\n".join([
        rf"$R^2_{{\mathrm{{oos}}}}$ = {model['r2_oos']:.3f} "
        rf"[{model['r2_oos_ci_lo']:.3f}, {model['r2_oos_ci_hi']:.3f}]",
        rf"within-city $R^2_{{\mathrm{{oos}}}}$ = {model_w['r2_oos']:.3f}",
        rf"rank-frozen: {frozen['r2_oos']:.3f} (within {frozen_w['r2_oos']:.3f})",
        rf"Spearman $\rho$ = {model['spearman']:.3f}, $n$ = {model['n']:,}",
    ])


def _tex_year_range(text: str) -> str:
    """'2007-2017 (out-of-sample)' -> '2007--2017 (out-of-sample)': en-dash
    year ranges only, leaving hyphenated words alone."""
    return re.sub(r"(\d{4})-(\d{4})", r"\1--\2", text)


def _growth_metrics_table(rows: list[dict]) -> pd.DataFrame:
    """Our rows plus the Khachiyan et al. benchmark rows, one table."""
    bench = [{"source": "Khachiyan et al. (2022) Table 2", "target": "dlog total personal income",
              "predictor": "CNN, no initial conditions", "scope": "pooled (2.4km cells)",
              "period": p, "r2_oos": r2}
             for p, r2 in _KHACHIYAN_INCOME_R2]
    return pd.DataFrame(rows + bench)


# ACS change significance for a pair, mirroring training's stable/change gate
# (process_acs.test_significance + STRUCTURAL_CHANGE_P): z = |Δ Rel_Score| /
# sqrt(SE0^2 + SE1^2), two-tailed. p < 0.10 is the training gate; 0.05 and
# 0.01 (the ``significant_*`` reporting cut) tighten it.
_CHANGE_P_LEVELS = (0.10, 0.05, 0.01)


def _load_rel_se_long(processed_dir: Path, indicator: str, years) -> pd.DataFrame:
    """Long [GEOID, year, se] of the label's replicate SE (``Rel_SE_{var}``;
    ``Rel_SE`` for income) plus each tract's training flag ``valid_change``.
    Years without an SE column are skipped (W-family SEs start in 2014)."""
    import pyarrow as pa
    var = indicators.token_to_var(indicator)
    prefix = f"Rel_SE_{var}" if var else "Rel_SE"
    path = processed_dir / _PANEL_FILENAME
    present = set(pa.ipc.open_file(path).schema.names)
    years = [int(y) for y in sorted(set(years)) if f"{prefix}_{y}" in present]
    flag = indicators.valid_change_col(indicator)
    cols = [_PANEL_GEOID_COL] + [f"{prefix}_{y}" for y in years]
    cols += [flag] if flag in present else []
    panel = pd.read_feather(path, columns=cols)
    panel = panel.rename(columns={_PANEL_GEOID_COL: "GEOID"})
    panel["GEOID"] = panel["GEOID"].astype(str).str.zfill(11)
    panel = panel.drop_duplicates("GEOID")
    vc = (panel[flag].astype(float).values if flag in panel.columns
          else np.full(len(panel), np.nan))
    frames = [pd.DataFrame({"GEOID": panel["GEOID"].values, "year": y,
                            "se": panel[f"{prefix}_{y}"].values.astype(float),
                            "valid_change": vc})
              for y in years]
    return pd.concat(frames, ignore_index=True)


def _attach_change_pvalue(pairs: pd.DataFrame, se_long: pd.DataFrame) -> pd.DataFrame:
    """Add se_0, se_1, change_z, change_p over each pair's own (t0, t1), and
    the tract's training flag ``valid_change`` (NaN where unavailable).

    Each year takes the SE of the closest year that has one — the same
    convention as training's ``main.build_se_lookup``. W-family replicate SEs
    exist only for 2014 and 2023, so a 2016/17 -> 2022/23 pair is tested
    with the 2014 and 2023 SEs.
    """
    se = se_long.set_index(["GEOID", "year"])
    se_years = np.array(sorted(se_long["year"].unique()))
    nearest = lambda yrs: se_years[np.abs(se_years[None, :] - np.asarray(yrs, int)[:, None])
                                   .argmin(axis=1)]
    out = pairs.copy()
    for k, tcol in (("0", "t0"), ("1", "t1")):
        out[f"se_year_{k}"] = nearest(out[tcol].values)
        out[f"se_{k}"] = se["se"].reindex(
            pd.MultiIndex.from_arrays([out["GEOID"], out[f"se_year_{k}"]])).values
    out["valid_change"] = se_long.drop_duplicates("GEOID").set_index("GEOID")[
        "valid_change"].reindex(out["GEOID"]).values
    se_diff = np.sqrt(out["se_0"] ** 2 + out["se_1"] ** 2)
    out["change_z"] = np.abs(out["label_1"] - out["label_0"]) / se_diff.where(se_diff > 0)
    from scipy.stats import norm
    out["change_p"] = 2 * (1 - norm.cdf(out["change_z"]))
    return out


def _change_groups(pairs: pd.DataFrame, levels=_CHANGE_P_LEVELS) -> dict[str, pd.Series]:
    """Boolean masks: 'stable' (p >= the loosest level) and nested
    'change p<L' for each level, plus training's own panel flag when present.
    Pairs without an SE belong to no group."""
    p = pairs["change_p"]
    known = p.notna()
    groups = {f"stable (p>={max(levels):.2f})": known & (p >= max(levels))}
    for lv in sorted(levels, reverse=True):
        groups[f"change p<{lv:.2f}"] = known & (p < lv)
    vc = pairs.get("valid_change")
    if vc is not None and vc.notna().any():
        groups["training flag: stable"] = vc == 0
        groups["training flag: change"] = vc == 1
    return groups


def _change_detection_stats(sub: pd.DataFrame, stable: pd.DataFrame, actual: str,
                            predicted: str) -> dict:
    """Diagnostics that separate label noise from prediction failure:

    * ``label_reliability`` = 1 - mean(SE0^2 + SE1^2) / Var(Δ label): the
      share of the group's label-change variance that is real (can be < 0 in
      the stable group, which is selected on small |Δ|).
    * ``sign_agree``: share of pairs whose predicted and actual change agree
      in sign (0.5 = coin flip).
    * ``auc_abs_vs_stable``: AUC of |Δ predicted| separating this group from
      the stable group (0.5 = predicted change size ignores real change).
    """
    from sklearn.metrics import roc_auc_score
    out = {"label_reliability": float("nan"), "auc_abs_vs_stable": float("nan")}
    if {"se_0", "se_1"} <= set(sub.columns) and sub["d_label"].var() > 0:
        noise = (sub["se_0"] ** 2 + sub["se_1"] ** 2).mean()
        out["label_reliability"] = float(1 - noise / sub["d_label"].var())
    a, p = sub[actual].values, sub[predicted].values
    ok = np.isfinite(a) & np.isfinite(p)
    out["sign_agree"] = float((np.sign(a[ok]) == np.sign(p[ok])).mean()) if ok.any() else float("nan")
    s = np.abs(stable[predicted].values)
    c = np.abs(p[ok])
    s = s[np.isfinite(s)]
    if len(c) and len(s) and sub.index.isin(stable.index).sum() == 0:
        out["auc_abs_vs_stable"] = float(roc_auc_score(
            np.r_[np.ones(len(c)), np.zeros(len(s))], np.r_[c, s]))
    return out


def _growth_by_change_rows(pairs: pd.DataFrame, n_boot: int = _GROWTH_N_BOOT,
                           levels=_CHANGE_P_LEVELS) -> list[dict]:
    """Model and rank-frozen R^2 for Δ$, Δlog$ and Δz within each change group
    (pooled across cities). Each row carries the group's n and sd of the
    actual change, so a rising R^2 can be read against the signal it had —
    R^2_oos mechanically rises with SST even when tracking does not improve —
    plus ``_change_detection_stats``."""
    groups = _change_groups(pairs, levels)
    stable = pairs[groups[f"stable (p>={max(levels):.2f})"]]
    specs = (("d wealth USD (GB2, 2023$)", "d_label_usd", "d_pred_usd", "d_frozen_usd"),
             ("dlog wealth USD (GB2)", "dlog_label_usd", "dlog_pred_usd", "dlog_frozen_usd"),
             ("d W2_r5 within-city z (label)", "d_label", "d_pred_z", None))
    rows = []
    for gname, mask in groups.items():
        sub = pairs[mask]
        for target, actual, pred_col, frz_col in specs:
            for who, col in (("model", pred_col), ("rank-frozen", frz_col)):
                if col is None:
                    continue
                if len(sub) < 10:
                    rows.append({"group": gname, "target": target, "predictor": who,
                                 "n": int(len(sub))})
                    continue
                r = _metric_row(sub[actual], sub[col], sub["cbsa"], target=target,
                                predictor=who, scope="pooled", period="", n_boot=n_boot)
                r["group"] = gname
                r["sd_actual"] = float(sub[actual].std())
                r.update(_change_detection_stats(sub, stable, actual, col))
                rows.append(r)
    return rows


def _add_gb2_dollar_pairs(pairs: pd.DataFrame, params_by_city: dict) -> pd.DataFrame:
    """Rank-frozen benchmark and log differences on top of the dollar pairs.

    ``frozen_usd_1`` carries the tract's t0 *predicted* rank to t1's GB2 — what
    the mapping alone predicts with no t1 imagery. Its R^2 is the share of
    dollar growth explained by the city-year GB2 maps (ACS distribution shift
    by quantile) plus the t0 level; the model adds information about change
    only to the extent it beats this benchmark.
    """
    out = pairs.copy()
    out["frozen_usd_1"] = _gb2_dollars(out, "pred_prob_0", params_by_city, year_col="t1")
    out["d_frozen_usd"] = out["frozen_usd_1"] - out["pred_usd_0"]
    with np.errstate(divide="ignore", invalid="ignore"):
        for side, a, b in (("label", "label_usd_1", "label_usd_0"),
                           ("pred", "pred_usd_1", "pred_usd_0"),
                           ("frozen", "frozen_usd_1", "pred_usd_0"),
                           ("acs", "acs_usd_1", "acs_usd_0")):
            ratio = out[a] / out[b]
            out[f"dlog_{side}_usd"] = np.log(ratio.where(ratio > 0))
    return out


def part_growth_figure_us(ctx: _USContext, target_gap: int = _GROWTH_TARGET_GAP,
                          wealth_long: pd.DataFrame | None = None,
                          se_long: pd.DataFrame | None = None,
                          n_boot: int = _GROWTH_N_BOOT) -> dict:
    """Long-difference growth figure (US): actual vs predicted change in
    wealth *dollars*, both sides through the GB2 mapping.

    Same test sample as ``part_main_figure_us`` (``_figure_test_sample``).
    Each tract contributes one (t0, t1) pair (see ``_long_difference_pairs``).

    Mapping (per test CBSA): GB2 fitted by MLE to the CBSA's tract
    distribution of the label's dollar variable (W2_r5 per-capita wealth) in
    each panel year, in constant 2023 dollars (CPI-U), unsmoothed (see
    ``_fit_city_gb2``; fit quality vs the empirical quantiles is printed and
    returned as ``growth/gb2_fit_max_log_err``). Within each (CBSA, year) cell the predicted score and the
    label are converted to Hazen ranks and pushed through that year's GB2
    quantile -> predicted and actual wealth $ at t0 and t1. Both sides are
    then in dollars, so R^2_oos = 1 - SSE/SST needs no refit.

    A. Δ wealth $ (actual vs predicted).  B. Δ log wealth $ (Khachiyan's
    log-difference scale). Each panel also reports the within-city R^2 and
    the rank-frozen benchmark (``_add_gb2_dollar_pairs``). The GB2 map at t1
    uses that year's ACS distribution for the city, i.e. contemporaneous ACS
    marginals Khachiyan et al.'s no-initial-conditions model never sees.

    ``wealth_long`` ([GEOID, cbsa, year, usd], constant dollars) is loaded
    from the ACS panel when omitted. Writes figures/US_growth_figure.pdf,
    tables/US_growth_metrics.csv and tables/US_growth_pairs.csv.
    """
    print("\n=== Long-difference growth figure (US, GB2 wealth dollars) ===")
    out = ctx.out
    sample = _figure_test_sample(ctx, what="growth figure")
    if sample is None:
        return {}
    tract = sample["tract_test"].copy()
    tract["cbsa"] = tract["cbsa"].astype(str)

    if wealth_long is None:
        try:
            wealth_long = _load_wealth_dollars_long(
                ctx.processed_dir, getattr(ctx, "indicator", indicators.DEFAULT_INDICATOR),
                _ACS_PANEL_YEARS)
        except Exception as exc:
            print(f"  wealth panel unavailable ({exc}); skipping growth figure")
            return {}
    print(f"  fitting GB2 per test city-year ({tract['cbsa'].nunique()} cities)...")
    params = _fit_city_gb2(wealth_long, tract["cbsa"].unique())
    fit_err = _gb2_fit_error(
        wealth_long, params, tract[["cbsa", "year"]].drop_duplicates().itertuples(index=False))
    if len(fit_err):
        worst = fit_err.loc[fit_err["max_log_err"].idxmax()]
        print(f"  GB2 fit vs empirical P10/P50/P90: median max|log err| = "
              f"{fit_err['max_log_err'].median():.3f}, worst {worst['max_log_err']:.3f} "
              f"(CBSA {worst['cbsa']}, {worst['year']})")

    tract["pred_z"] = _zscore_within(tract, "pred")
    tract["pred_prob"] = _hazen_within(tract, "pred")
    tract["label_prob"] = _hazen_within(tract, "label")
    tract["pred_usd"] = _gb2_dollars(tract, "pred_prob", params)
    tract["label_usd"] = _gb2_dollars(tract, "label_prob", params)
    acs = wealth_long.set_index(["GEOID", "year"])["usd"]
    tract["acs_usd"] = acs.reindex(
        pd.MultiIndex.from_arrays([tract["GEOID"], tract["year"].astype(int)])).values

    pairs = _long_difference_pairs(
        tract, target_gap,
        value_cols=("label", "pred_z", "pred_prob", "label_usd", "pred_usd", "acs_usd"))
    pairs = _add_gb2_dollar_pairs(pairs, params)
    ok = np.isfinite(pairs[["d_label_usd", "d_pred_usd", "d_frozen_usd"]]).all(axis=1)
    if (~ok).any():
        print(f"  dropping {int((~ok).sum())} pair(s) without a finite GB2 dollar value")
    pairs = pairs[ok].reset_index(drop=True)
    if len(pairs) < 30:
        print(f"  only {len(pairs)} tract pairs; skipping growth figure")
        return {}

    gaps = pairs["gap"].value_counts().sort_index()
    gap_str = "; ".join(f"{int(g)}y: {n:,}" for g, n in gaps.items())
    print(f"  {len(pairs):,} tract pairs over {pairs['cbsa'].nunique()} test cities; gaps {gap_str}")

    cl = pairs["cbsa"]
    row = lambda a, p, **kw: _metric_row(pairs[a], pairs[p], cl, period=gap_str,
                                         n_boot=n_boot, **kw)
    res = {}
    for key, actual, target in (("usd", "d_label_usd", "d wealth USD (GB2, 2023$)"),
                                ("log", "dlog_label_usd", "dlog wealth USD (GB2)")):
        pred_col = "d_pred_usd" if key == "usd" else "dlog_pred_usd"
        frz_col = "d_frozen_usd" if key == "usd" else "dlog_frozen_usd"
        for who, col in (("model", pred_col), ("rank-frozen", frz_col)):
            for scope in ("pooled", "within-city"):
                res[(key, who, scope)] = row(actual, col, target=target, predictor=who, scope=scope)
    rows = list(res.values())
    # Diagnostics: the raw-rank change (no GB2) and raw ACS dollars as the actual.
    rows.append(row("d_label", "d_pred_z", target="d W2_r5 within-city z (label)",
                    predictor="model (pred z within city-year)", scope="pooled"))
    for who, col in (("model", "d_pred_usd"), ("rank-frozen", "d_frozen_usd")):
        rows.append(row("d_acs_usd", col, target="d wealth USD (raw ACS W2, 2023$)",
                        predictor=who, scope="pooled"))

    # ── figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_SIZE_TWO_COL[0], FIG_SIZE_TWO_COL[0] * 0.56),
                     constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.15])
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_cap = fig.add_subplot(gs[1, :])
    ax_cap.axis("off")
    ann = {k: _growth_annotation(res[(k, "model", "pooled")], res[(k, "model", "within-city")],
                                 res[(k, "rank-frozen", "pooled")],
                                 res[(k, "rank-frozen", "within-city")]) for k in ("usd", "log")}
    n_off = _panel_growth_scatter(
        ax_a, pairs["d_label_usd"] / 1e3, pairs["d_pred_usd"] / 1e3, letter="A",
        xlabel=r"$\Delta$ wealth per capita, actual (\$k, 2023)",
        ylabel=r"$\Delta$ wealth per capita, predicted (\$k, 2023)", r2_text=ann["usd"])
    n_off += _panel_growth_scatter(
        ax_b, pairs["dlog_label_usd"], pairs["dlog_pred_usd"], letter="B",
        xlabel=r"$\Delta$ log wealth per capita, actual",
        ylabel=r"$\Delta$ log wealth per capita, predicted", r2_text=ann["log"])

    bench = "; ".join(f"{_tex_year_range(p)}: {r2:.2f}" for p, r2 in _KHACHIYAN_INCOME_R2)
    caption_body = (
        rf"Actual vs.~predicted tract wealth change, test cities ({pairs['cbsa'].nunique()} "
        rf"CBSAs, {len(pairs):,} tracts). Each tract pairs its latest year $t_1$ with the "
        rf"earlier year closest to $t_1-{target_gap}$ (gaps: {gap_str}). In every city-year, "
        r"predicted and ACS label ranks are mapped to dollars through a GB2 fitted to that "
        r"city's ACS W2 tract wealth that year (2023 dollars, CPI-U), so "
        r"$R^2_{\mathrm{oos}}=1-\mathrm{SSE}/\mathrm{SST}$ needs no refit. (A) Dollar change; "
        r"(B) log change. Within-city: both sides demeaned by CBSA. Rank-frozen: the tract's "
        r"$t_0$ predicted rank carried to $t_1$'s GB2 \textemdash\ no $t_1$ imagery, so its "
        r"$R^2$ comes from the GB2 maps alone. Brackets: 95\% city-cluster bootstrap CIs."
    )
    caption_note = (
        rf"Benchmark \textemdash\ Khachiyan et al.~(2022), Table 2, income, no initial "
        rf"conditions, $R^2$ of $\Delta$ log total personal income: {bench}. "
        r"Unlike theirs, the GB2 map at $t_1$ uses that year's ACS city distribution."
    )
    if n_off:
        caption_note += (rf"  Axes show the central {100 * _PANEL_C_CENTRAL_FRAC:.1f}\% of each "
                         rf"marginal; {n_off:,} points fall outside and are not shown.")
    caption = "\n".join(textwrap.wrap(caption_body, width=125)) + "\n" + "\n".join(
        textwrap.wrap(caption_note, width=125))
    ax_cap.text(0.5, 1.0, caption, transform=ax_cap.transAxes, ha="center", va="top",
                fontsize=6.0, color="0.25", linespacing=1.6)
    _savefig(fig, out / "figures" / "US_growth_figure.pdf")

    table = _growth_metrics_table(rows)
    table.to_csv(out / "tables" / "US_growth_metrics.csv", index=False)

    # ── R^2 by ACS change significance (training's MOE gate, per pair) ──────
    by_change = None
    if se_long is None:
        try:
            se_long = _load_rel_se_long(
                ctx.processed_dir, getattr(ctx, "indicator", indicators.DEFAULT_INDICATOR),
                _ACS_PANEL_YEARS)
        except Exception as exc:
            print(f"  label SEs unavailable ({exc}); stable/change split skipped")
    if se_long is not None:
        pairs = _attach_change_pvalue(pairs, se_long)
        by_change = pd.DataFrame(_growth_by_change_rows(pairs, n_boot=n_boot))
        by_change.to_csv(out / "tables" / "US_growth_by_change.csv", index=False)
        print("  R2 by ACS change significance (pooled; model vs rank-frozen):")
        for r in by_change.itertuples(index=False):
            if pd.notna(getattr(r, "r2_oos", np.nan)):
                print(f"    {r.group:24s} {r.target[:30]:30s} {r.predictor:11s} n={r.n:5d} "
                      f"sd={r.sd_actual:10.3f} R2_oos={r.r2_oos:8.4f} "
                      f"[{r.r2_oos_ci_lo:.3f}, {r.r2_oos_ci_hi:.3f}] "
                      f"R2_ols={r.r2_ols:.4f} rho={r.spearman:.3f} "
                      f"rel={r.label_reliability:.3f} sign={r.sign_agree:.3f} "
                      f"auc={r.auc_abs_vs_stable:.3f}")
    pairs.to_csv(out / "tables" / "US_growth_pairs.csv", index=False)
    for r in rows:
        print(f"  {r['target'][:34]:34s} {r['predictor'][:22]:22s} {r['scope']:11s} "
              f"R2_oos={r['r2_oos']:8.4f} [{r['r2_oos_ci_lo']:.4f}, {r['r2_oos_ci_hi']:.4f}] "
              f"R2_ols={r['r2_ols']:.4f} rho={r['spearman']:.4f}")
    print("  Khachiyan et al. (2022) Table 2, income, no IC: "
          + ", ".join(f"{p}={r2:.4f}" for p, r2 in _KHACHIYAN_INCOME_R2))

    headline = {
        "growth/n_tract_pairs": int(len(pairs)),
        "growth/n_test_cities": int(pairs["cbsa"].nunique()),
        "growth/gap_median": float(pairs["gap"].median()),
        "growth/gb2_fit_max_log_err": float(fit_err["max_log_err"].max()) if len(fit_err)
        else float("nan"),
    }
    for (key, who, scope), r in res.items():
        tag = f"growth/{key}_{'model' if who == 'model' else 'frozen'}_" \
              f"{'pooled' if scope == 'pooled' else 'within'}"
        headline[f"{tag}_r2_oos"] = r["r2_oos"]
        headline[f"{tag}_r2_ols"] = r["r2_ols"]
    if by_change is not None and "r2_oos" in by_change.columns:
        model = by_change[by_change["predictor"] == "model"]
        for r in model.itertuples(index=False):
            key = re.sub(r"[^a-z0-9]+", "_", f"{r.group} {r.target}".lower()).strip("_")
            headline[f"growth/by_change/{key}/r2_oos"] = r.r2_oos
            headline[f"growth/by_change/{key}/n"] = r.n
    return headline


# ─── Part F: growth R^2 by construction cohort (NYC) ──────────────────────────
# The long-difference growth test (``part_growth_figure_us``) on the sample
# whose change status is known from the ground rather than from ACS: NYC tracts
# grouped by Part D's construction cohorts. "stable" is Part D's pinned control
# (never crossed CONTROL_THRESHOLD of baseline building area); "change h" is a
# tract that first crossed h inside the pair's window (t0, t1]. Tracts that
# crossed before t0 changed outside the window and belong to no group.


def _construction_window_groups(pairs: pd.DataFrame, cohorts_by_t: dict) -> dict[str, pd.Series]:
    """Boolean masks over ``pairs`` rows (needs GEOID, t0, t1).

    ``cohorts_by_t`` maps threshold -> ``CohortResult.cohorts`` (GEOID_str,
    cohort_year; 0 = never treated under the pinned control; tracts dropped as
    ambiguous or without baseline stock are absent). The stable group is the
    same tracts at every threshold because the control is pinned, so it is
    taken from the first threshold.
    """
    groups = {}
    for i, (thresh, coh) in enumerate(sorted(cohorts_by_t.items())):
        year = pairs["GEOID"].map(coh.set_index("GEOID_str")["cohort_year"])
        if i == 0:
            groups["stable"] = year == 0
        groups[f"change {thresh:.0%}"] = (year > pairs["t0"]) & (year <= pairs["t1"])
        groups[f"pre-window {thresh:.0%}"] = (year > 0) & (year <= pairs["t0"])
    return groups


def _construction_group_row(sub: pd.DataFrame, stable: pd.DataFrame, n_boot: int) -> dict:
    """Metrics for one (split, group): R^2 of Δlabel on the within-year z-scored
    Δpred, plus the two-period mean shifts vs the stable group in both the
    label and the raw prediction (the latter is a 2-period analogue of the
    CSA ATT, in the same tract-mean-prediction units)."""
    row = {"n": int(len(sub)),
           "mean_d_label": float(sub["d_label"].mean()),
           "mean_d_pred": float(sub["d_pred"].mean()),
           "d_label_vs_stable": float(sub["d_label"].mean() - stable["d_label"].mean()),
           "d_pred_vs_stable": float(sub["d_pred"].mean() - stable["d_pred"].mean())}
    if len(sub) < 10:
        return row
    met = _growth_metrics(sub["d_label"].values, sub["d_pred_z"].values)
    lo, hi = _cluster_bootstrap_ci(sub["d_label"].values, sub["d_pred_z"].values,
                                   sub["GEOID"].values, _r2_oos, n_boot=n_boot)
    row.update(met)
    row.update({"r2_oos_ci_lo": lo, "r2_oos_ci_hi": hi,
                "sd_d_label": float(sub["d_label"].std())})
    row.update(_change_detection_stats(sub, stable, "d_label", "d_pred_z"))
    return row


def part_f(results_dir: Path, processed_dir: Path, out: Path,
           years: list[int] | None = None, target_gap: int = _GROWTH_TARGET_GAP,
           thresholds=None, n_boot: int = _GROWTH_N_BOOT) -> pd.DataFrame | None:
    """Growth R^2 for NYC tracts by construction cohort, train vs held out.

    Pairs: each tract's latest year t1 and the earlier year closest to
    t1 - ``target_gap`` (2014 -> 2024 on the biennial NYC grid). Outcome is
    Part D's tract-mean prediction (all buildings); the label is the tract's
    W2 z-score from the NYC pass. The prediction is z-scored within year so
    R^2_oos needs no refit. Writes tables/F_growth_by_construction.csv.
    """
    from src import csa_event_study as ces

    print("\n=== Part F: growth R^2 by construction cohort (NYC) ===")
    spec = ces.CSA_CITIES["nyc"]
    if not (Path(processed_dir) / spec.footprints_filename).exists():
        print(f"  {spec.footprints_filename} not found — Part F skipped")
        return None
    years = sorted(years) if years else _available_years(results_dir) or list(YEARS)
    thresholds = tuple(thresholds) if thresholds else ces.DEFAULT_THRESHOLDS
    city_years = spec.years(years)
    baseline_year = spec.baseline(CSA_BASELINE_YEAR)

    per_building, tracts, construction_year = _csa_city_geometry(processed_dir, spec)
    outcomes = _csa_city_outcomes(
        results_dir, results_dir, processed_dir, spec, city_years, baseline_year,
        tracts, construction_year, lambda: _csa_building_preds(results_dir, city_years))
    labels = _load_tract_long(results_dir, city_years)
    if outcomes.empty or labels.empty:
        print("  no NYC predictions or tract labels — Part F skipped")
        return None
    tract = outcomes[["GEOID_str", "year", "pred_all"]].merge(
        labels[["GEOID_str", "year", "Rel_Score"]], on=["GEOID_str", "year"])
    tract = tract.rename(columns={"GEOID_str": "GEOID", "pred_all": "pred",
                                  "Rel_Score": "label"}).dropna(subset=["pred", "label"])
    tract["cbsa"] = "35620"
    tract["pred_z"] = _zscore_within(tract, "pred")
    pairs = _long_difference_pairs(tract, target_gap, value_cols=("label", "pred", "pred_z"))
    windows = pairs.groupby(["t0", "t1"]).size()
    print(f"  {len(pairs):,} tract pairs; windows "
          + ", ".join(f"{a}-{b}: {n:,}" for (a, b), n in windows.items()))

    cohorts_by_t = {
        t: ces.build_tract_cohorts(
            None, tracts, city_years, baseline_year=baseline_year, threshold=t,
            year_col=spec.year_col, demolition_col=spec.demolition_col,
            area_epsg=spec.area_epsg, control_threshold=ces.CONTROL_THRESHOLD,
            per_building=per_building).cohorts
        for t in thresholds}
    groups = _construction_window_groups(pairs, cohorts_by_t)
    split_of = _csa_split_of(processed_dir)
    pairs["split_group"] = pairs["GEOID"].map(split_of)

    rows = []
    for split in [g for g in CSA_SPLIT_GROUPS if (pairs["split_group"] == g).any()]:
        in_split = pairs["split_group"] == split
        stable = pairs[in_split & groups["stable"]]
        for gname, mask in groups.items():
            sub = pairs[in_split & mask]
            row = {"split": CSA_SPLIT_LABELS.get(split, split), "group": gname}
            row.update(_construction_group_row(sub, stable, n_boot))
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "tables" / "F_growth_by_construction.csv", index=False)
    for r in table.itertuples(index=False):
        line = (f"  {r.split:18s} {r.group:16s} n={r.n:5d} "
                f"mean dlabel={r.mean_d_label:+.3f} (vs stable {r.d_label_vs_stable:+.3f})  "
                f"mean dpred={r.mean_d_pred:+.3f} (vs stable {r.d_pred_vs_stable:+.3f})")
        if pd.notna(getattr(r, "r2_oos", np.nan)):
            line += (f"  R2_oos={r.r2_oos:+.3f} [{r.r2_oos_ci_lo:+.3f}, {r.r2_oos_ci_hi:+.3f}] "
                     f"R2_ols={r.r2_ols:.4f} rho={r.spearman:+.3f} "
                     f"sign={r.sign_agree:.3f} auc={r.auc_abs_vs_stable:.3f}")
        print(line)
    return table


# ─── ACS-window growth scatter (NYC, construction groups) ─────────────────────
# A 5-year ACS estimate labelled Y averages survey years Y-4..Y, so its change
# is compared with the change in the tract's prediction averaged over the
# imagery years inside the same two windows, not with two single flights.

_ACS_WINDOW_SPAN = 5
_GROUP_COLORS = {"stable": "#0072B2", "change": "#D55E00"}   # Okabe-Ito


def _acs_window_years(acs_year: int, imagery_years, span: int = _ACS_WINDOW_SPAN) -> list[int]:
    """Imagery years inside the survey window of the ``acs_year`` 5-year
    estimate, i.e. in [acs_year - span + 1, acs_year]."""
    return sorted(int(y) for y in imagery_years if acs_year - span < int(y) <= acs_year)


def _window_average(tract: pd.DataFrame, years: list[int], col: str = "pred") -> pd.DataFrame:
    """Per-tract mean of ``col`` over ``years`` -> [GEOID, mean, n_years]."""
    sub = tract[tract["year"].isin(years)]
    g = sub.groupby("GEOID")[col]
    return pd.DataFrame({"mean": g.mean(), "n_years": g.size()}).reset_index()


def _load_panel_scores(processed_dir: Path, indicator: str, years) -> pd.DataFrame:
    """Long [GEOID, year, label] of the indicator's per-CBSA z-score straight
    from the ACS panel — the 5-year vintage labelled ``year``."""
    cols = [_PANEL_GEOID_COL] + [indicators.score_col(indicator, y) for y in years]
    panel = pd.read_feather(processed_dir / _PANEL_FILENAME, columns=cols)
    panel = panel.rename(columns={_PANEL_GEOID_COL: "GEOID"})
    panel["GEOID"] = panel["GEOID"].astype(str).str.zfill(11)
    panel = panel.drop_duplicates("GEOID")
    return pd.concat([pd.DataFrame({"GEOID": panel["GEOID"].values, "year": int(y),
                                    "label": panel[indicators.score_col(indicator, y)]
                                    .values.astype(float)}) for y in years],
                     ignore_index=True)


def _acs_window_pairs(tract: pd.DataFrame, labels: pd.DataFrame, acs_pre: int, acs_post: int,
                      imagery_years, span: int = _ACS_WINDOW_SPAN) -> pd.DataFrame:
    """One row per tract: Δ label between the two ACS vintages and Δ of the
    window-averaged prediction (each window average z-scored across tracts,
    so both sides are in within-sample z units). Tracts need at least one
    flight in each window. Also carries the single-flight Δ (last flight of
    each window) for comparison."""
    w_pre = _acs_window_years(acs_pre, imagery_years, span)
    w_post = _acs_window_years(acs_post, imagery_years, span)
    if not w_pre or not w_post:
        raise ValueError(f"no imagery year inside an ACS window: {w_pre} / {w_post}")
    a = _window_average(tract, w_pre).rename(columns={"mean": "pred_pre", "n_years": "n_pre"})
    b = _window_average(tract, w_post).rename(columns={"mean": "pred_post", "n_years": "n_post"})
    lab = labels.pivot(index="GEOID", columns="year", values="label")
    out = a.merge(b, on="GEOID")
    out["label_pre"] = lab[acs_pre].reindex(out["GEOID"]).values
    out["label_post"] = lab[acs_post].reindex(out["GEOID"]).values
    single = tract.pivot(index="GEOID", columns="year", values="pred")
    out["pred_single_pre"] = single[max(w_pre)].reindex(out["GEOID"]).values
    out["pred_single_post"] = single[max(w_post)].reindex(out["GEOID"]).values
    out = out.dropna(subset=["label_pre", "label_post"]).reset_index(drop=True)
    z = lambda s: (s - s.mean()) / s.std()
    out["d_label"] = out["label_post"] - out["label_pre"]
    out["d_pred_z"] = z(out["pred_post"]) - z(out["pred_pre"])
    out["d_pred_single_z"] = z(out["pred_single_post"]) - z(out["pred_single_pre"])
    out["d_pred"] = out["pred_post"] - out["pred_pre"]
    out["t0"], out["t1"] = acs_pre, acs_post
    out.attrs.update({"window_pre": w_pre, "window_post": w_post})
    return out


def _panel_group_scatter(ax: plt.Axes, groups: dict[str, tuple[pd.DataFrame, str]],
                         x: str = "d_label", y: str = "d_pred_z") -> None:
    """Δ label vs Δ prediction, one colour per group, with a per-group OLS
    line, a shared 45-degree line and a per-group R^2 legend entry."""
    allv = pd.concat([g[[x, y]] for g, _ in groups.values()])
    lo, hi = np.nanpercentile(allv.values, [0.5, 99.5])
    pad = 0.05 * (hi - lo)
    ax.plot([lo, hi], [lo, hi], color="0.4", ls="--", lw=0.9, zorder=1, label=r"45$^\circ$")
    for name, (g, color) in groups.items():
        m = _growth_metrics(g[x].values, g[y].values)
        ax.scatter(g[x], g[y], s=7, alpha=0.6, color=color, linewidths=0, zorder=2,
                   label=(rf"{name} ($n$={m['n']}): $R^2_{{\mathrm{{oos}}}}$={m['r2_oos']:.2f}, "
                          rf"$R^2_{{\mathrm{{OLS}}}}$={m['r2_ols']:.3f}"))
        ok = np.isfinite(g[x]) & np.isfinite(g[y])
        if ok.sum() >= 3 and np.ptp(g.loc[ok, x]) > 0:
            b, a = np.polyfit(g.loc[ok, x], g.loc[ok, y], 1)
            xs = np.array([lo, hi])
            ax.plot(xs, a + b * xs, color=color, lw=1.6, zorder=3)
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=6, loc="upper left", frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.85)


def _winsorize(v: pd.Series | np.ndarray, frac: float) -> np.ndarray:
    """Clip to the [frac, 1-frac] percentiles of its own finite values — Part C's
    per-year P1/P99 rule for GB2 dollars."""
    v = np.asarray(v, float)
    fin = v[np.isfinite(v)]
    if not len(fin):
        return v
    lo, hi = np.percentile(fin, [100 * frac, 100 * (1 - frac)])
    return np.clip(v, lo, hi)


def _add_window_gb2_dollars(pairs: pd.DataFrame, params_by_year: dict, acs_pre: int,
                            acs_post: int, acs_usd: pd.DataFrame | None = None,
                            winsor: float | None = 0.01) -> pd.DataFrame:
    """Map each vintage's ranks to wealth dollars through that vintage's GB2.

    Within a vintage only the ordering of the label is comparable (the draft's
    Ψ_ct), so the label and the window-averaged prediction are converted to
    Hazen ranks across the tracts of ``pairs`` and pushed through the GB2
    fitted to that vintage's distribution: ``label_usd_*``, ``pred_usd_*``,
    the single-flight ``pred_single_usd_*``, and ``frozen_usd_post`` (pre-window
    predicted rank under the post vintage's GB2 — the change the maps alone
    imply). Adds Δ ($) and Δlog columns for each. ``acs_usd`` ([GEOID, year,
    usd]) adds the raw ACS dollars as an alternative actual.

    ``winsor`` clips every dollar column at its own P1/P99 within the vintage
    before differencing, as Part C does: GB2's power-law tail maps the top
    Hazen ranks to tens of millions per capita (NYC 2023: $30.8M at rank
    0.9998), and without the clip five Manhattan tracts swapping places near
    the top carried 85% of the sum of squares of Δ label $.
    """
    out = pairs.copy()
    out["_cell"] = 0
    for side, pre_col, post_col in (("label", "label_pre", "label_post"),
                                    ("pred", "pred_pre", "pred_post"),
                                    ("pred_single", "pred_single_pre", "pred_single_post")):
        for when, col, yr in (("pre", pre_col, acs_pre), ("post", post_col, acs_post)):
            ok = out[col].notna()
            prob = np.full(len(out), np.nan)
            prob[ok.values] = _hazen_within(out.loc[ok].assign(_x=out.loc[ok, col]), "_x",
                                            by=("_cell",)).values
            out[f"{side}_prob_{when}"] = prob
            out[f"{side}_usd_{when}"] = _gb2_quantile(prob, params_by_year[yr])
    out["frozen_usd_post"] = _gb2_quantile(out["pred_prob_pre"].values, params_by_year[acs_post])
    if acs_usd is not None:
        a = acs_usd.set_index(["GEOID", "year"])["usd"]
        for when, yr in (("pre", acs_pre), ("post", acs_post)):
            out[f"acs_usd_{when}"] = a.reindex(
                pd.MultiIndex.from_arrays([out["GEOID"], np.full(len(out), yr)])).values
    if winsor:
        usd_cols = [c for c in out.columns
                    if c.endswith(("_usd_pre", "_usd_post")) and not c.startswith("d")]
        for c in usd_cols:
            out[c] = _winsorize(out[c], winsor)
    pairs_cols = [("label", "label_usd_post", "label_usd_pre"),
                  ("pred", "pred_usd_post", "pred_usd_pre"),
                  ("pred_single", "pred_single_usd_post", "pred_single_usd_pre"),
                  ("frozen", "frozen_usd_post", "pred_usd_pre")]
    if acs_usd is not None:
        pairs_cols.append(("acs", "acs_usd_post", "acs_usd_pre"))
    with np.errstate(divide="ignore", invalid="ignore"):
        for side, b, a in pairs_cols:
            out[f"d_{side}_usd"] = out[b] - out[a]
            ratio = out[b] / out[a]
            out[f"dlog_{side}_usd"] = np.log(ratio.where(ratio > 0))
    return out.drop(columns=["_cell"])


def _direction_metrics(actual, predicted, n_boot: int = _GROWTH_N_BOOT, seed: int = 0) -> dict:
    """Does the predicted change have the sign of the actual change?

    ``accuracy`` is the share of tracts whose signs agree; ``majority`` is the
    accuracy of always predicting the more common actual sign (the bar a
    direction call must clear when most tracts move the same way);
    ``balanced`` averages the hit rates on actual risers and fallers (0.5 =
    chance whatever the base rate); ``kappa`` is Cohen's agreement beyond
    chance. Tracts with a zero change on either side are dropped. The CI
    resamples tracts.
    """
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    ok = np.isfinite(a) & np.isfinite(p) & (a != 0) & (p != 0)
    ua, up = a[ok] > 0, p[ok] > 0
    n = int(ok.sum())
    nan = float("nan")
    if n == 0:
        return {"n": 0, "accuracy": nan, "acc_ci_lo": nan, "acc_ci_hi": nan, "majority": nan,
                "balanced": nan, "kappa": nan, "share_actual_up": nan, "share_pred_up": nan}

    def stats(ua_, up_):
        acc = float(np.mean(ua_ == up_))
        tpr = float(np.mean(up_[ua_])) if ua_.any() else nan
        tnr = float(np.mean(~up_[~ua_])) if (~ua_).any() else nan
        bal = np.nanmean([tpr, tnr])
        pa, pp = ua_.mean(), up_.mean()
        pe = pa * pp + (1 - pa) * (1 - pp)
        kappa = (acc - pe) / (1 - pe) if pe < 1 else nan
        return acc, float(bal), float(kappa)

    acc, bal, kappa = stats(ua, up)
    rng = np.random.default_rng(seed)
    boots = [stats(ua[i], up[i])[0] for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": n, "accuracy": acc, "acc_ci_lo": float(lo), "acc_ci_hi": float(hi),
            "majority": float(max(ua.mean(), 1 - ua.mean())), "balanced": bal,
            "kappa": kappa, "share_actual_up": float(ua.mean()),
            "share_pred_up": float(up.mean())}


def _direction_table(pairs: pd.DataFrame, groups: dict[str, pd.Series], preds: dict[str, str],
                     actual: str = "d_label_usd", n_boot: int = _GROWTH_N_BOOT) -> pd.DataFrame:
    """Direction metrics per (split, group, prediction, definition).

    ``absolute``: sign of the change itself. ``relative to city``: sign of the
    change minus the city's median change over all tracts of ``pairs`` (the
    whole cell, not the group) — did the tract gain more than the typical
    tract? That strips the city-wide growth both sides share through the GB2
    maps, which otherwise lets "everyone went up" pass for skill.
    """
    rel = {c: pairs[c] - pairs[c].median() for c in [actual, *preds.values()]}
    rows = []
    for split in [None, *[g for g in CSA_SPLIT_GROUPS if (pairs["split_group"] == g).any()]]:
        in_split = (pairs["split_group"] == split) if split else pairs["split_group"].notna()
        for gname, mask in groups.items():
            m = in_split & mask
            for kind, col in preds.items():
                for definition, a, p in (("absolute", pairs[actual], pairs[col]),
                                         ("relative to city", rel[actual], rel[col])):
                    rows.append({"split": CSA_SPLIT_LABELS.get(split, "All (held out + train)"),
                                 "group": gname, "prediction": kind, "definition": definition,
                                 **_direction_metrics(a[m], p[m], n_boot=n_boot)})
    return pd.DataFrame(rows)


def _plot_direction(ax: plt.Axes, tbl: pd.DataFrame, groups: list[str], preds: list[str],
                    colors: dict[str, str], letter: str, title: str) -> None:
    """Grouped bars of direction accuracy (95% CI) per group x prediction, a
    black tick at each group's majority-class baseline and a 0.5 line."""
    width = 0.8 / len(preds)
    for gi, g in enumerate(groups):
        for pi, kind in enumerate(preds):
            r = tbl[(tbl["group"] == g) & (tbl["prediction"] == kind)]
            if r.empty:
                continue
            r = r.iloc[0]
            x = gi - 0.4 + width * (pi + 0.5)
            ax.bar(x, r["accuracy"], width * 0.92, color=colors[kind],
                   label=kind.replace("->", r"$\to$") if gi == 0 else None, zorder=2)
            ax.errorbar(x, r["accuracy"], yerr=[[r["accuracy"] - r["acc_ci_lo"]],
                                                [r["acc_ci_hi"] - r["accuracy"]]],
                        color="0.2", lw=0.8, capsize=2, zorder=3)
        maj = tbl[tbl["group"] == g]["majority"].iloc[0]
        ax.hlines(maj, gi - 0.42, gi + 0.42, color="black", lw=1.4, zorder=4,
                  label="majority-class baseline" if gi == 0 else None)
    ax.axhline(0.5, color="0.6", ls=":", lw=0.9, zorder=1)
    ax.set_xticks(range(len(groups)))
    tex = lambda g: g.replace(">=", r"$\geq$").replace("%", r"\%")   # usetex-safe
    ax.set_xticklabels([f"{tex(g)}\n($n$={int(tbl[tbl['group'] == g]['n'].iloc[0])})"
                        for g in groups], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of tracts with the right direction")
    ax.set_title(title, fontsize=7.5)
    _panel_title(ax, letter)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def part_f_acs_window(results_dir: Path, processed_dir: Path, out: Path,
                      years: list[int] | None = None, acs_pre: int = 2018,
                      acs_post: int = 2023, threshold: float = 0.05,
                      indicator: str = indicators.DEFAULT_INDICATOR,
                      n_boot: int = _GROWTH_N_BOOT,
                      wealth_long: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """NYC: 5-year change in GB2 wealth dollars between two ACS vintages vs
    the change implied by the tract prediction averaged over each vintage's
    survey window.

    Per vintage, a GB2 is fitted (unsmoothed, constant 2023 dollars) to the
    W2 per-capita wealth of the NYC tracts (the CSA city's county prefixes —
    the same population whose ranks are mapped), and both label and
    window-averaged prediction ranks are mapped through it
    (``_add_window_gb2_dollars``). Comparing dollars, not z-scores, is what the
    model licenses: a z-score is a cell-specific transform of the latent
    position, so its change mixes movement with changes in the cell's scale.

    Groups (Part D cohorts, control pinned at CONTROL_THRESHOLD): ``stable``
    never crossed the control cut; ``change`` first crossed ``threshold``
    after ``acs_pre`` and by ``acs_post``. Tracts that crossed earlier are
    excluded. Writes figures/F_acs_window_scatter.pdf, tables/F_acs_window.csv
    and tables/F_acs_window_pairs.csv.
    """
    from src import csa_event_study as ces

    print(f"\n=== Part F (ACS window, GB2 dollars): ACS {acs_pre} -> {acs_post}, "
          f"window-averaged predictions, stable vs {threshold:.0%} (NYC) ===")
    spec = ces.CSA_CITIES["nyc"]
    if not (Path(processed_dir) / spec.footprints_filename).exists():
        print(f"  {spec.footprints_filename} not found — skipped")
        return None
    years = sorted(years) if years else _available_years(results_dir) or list(YEARS)
    city_years = spec.years(years)
    baseline_year = spec.baseline(CSA_BASELINE_YEAR)
    per_building, tracts, construction_year = _csa_city_geometry(processed_dir, spec)
    outcomes = _csa_city_outcomes(
        results_dir, results_dir, processed_dir, spec, city_years, baseline_year,
        tracts, construction_year, lambda: _csa_building_preds(results_dir, city_years))
    if outcomes.empty:
        print("  no NYC predictions — skipped")
        return None
    tract = outcomes[["GEOID_str", "year", "pred_all"]].rename(
        columns={"GEOID_str": "GEOID", "pred_all": "pred"}).dropna()
    labels = _load_panel_scores(processed_dir, indicator, [acs_pre, acs_post])
    pairs = _acs_window_pairs(tract, labels, acs_pre, acs_post, city_years)
    w_pre, w_post = pairs.attrs["window_pre"], pairs.attrs["window_post"]
    print(f"  ACS {acs_pre} covers {acs_pre - _ACS_WINDOW_SPAN + 1}-{acs_pre}: flights {w_pre}; "
          f"ACS {acs_post} covers {acs_post - _ACS_WINDOW_SPAN + 1}-{acs_post}: flights {w_post}; "
          f"{len(pairs):,} tracts")

    # ── GB2 per vintage on the NYC tract distribution ────────────────────────
    if wealth_long is None:
        wealth_long = _load_wealth_dollars_long(processed_dir, indicator, [acs_pre, acs_post])
    city_w = wealth_long[wealth_long["GEOID"].str[:5].isin(spec.geoid_prefixes)].copy()
    city_w["cbsa"] = "nyc"
    params = _fit_city_gb2(city_w, ["nyc"]).get("nyc", {})
    if acs_pre not in params or acs_post not in params:
        print("  GB2 fit unavailable for a vintage — skipped")
        return None
    fit_err = _gb2_fit_error(city_w, {"nyc": params}, [("nyc", acs_pre), ("nyc", acs_post)])
    print("  GB2 fit vs empirical P10/P50/P90 max|log err|: "
          + ", ".join(f"{int(r.year)}={r.max_log_err:.3f}" for r in fit_err.itertuples()))
    pairs = _add_window_gb2_dollars(pairs, params, acs_pre, acs_post, acs_usd=city_w)

    thresholds = sorted({ces.CONTROL_THRESHOLD, threshold})
    cohorts_by_t = {
        t: ces.build_tract_cohorts(
            None, tracts, city_years, baseline_year=baseline_year, threshold=t,
            year_col=spec.year_col, demolition_col=spec.demolition_col,
            area_epsg=spec.area_epsg, control_threshold=ces.CONTROL_THRESHOLD,
            per_building=per_building).cohorts
        for t in thresholds}
    masks = _construction_window_groups(pairs, cohorts_by_t)
    chg = f"change {threshold:.0%}"
    pairs["split_group"] = pairs["GEOID"].map(_csa_split_of(processed_dir))

    single = f"single flight ({max(w_pre)}->{max(w_post)})"
    specs = (  # (target, actual col, [(prediction kind, predicted col), ...])
        ("d wealth USD (GB2)", "d_label_usd",
         [("window-averaged", "d_pred_usd"), (single, "d_pred_single_usd"),
          ("rank-frozen", "d_frozen_usd")]),
        ("dlog wealth USD (GB2)", "dlog_label_usd",
         [("window-averaged", "dlog_pred_usd"), (single, "dlog_pred_single_usd"),
          ("rank-frozen", "dlog_frozen_usd")]),
        ("d wealth USD (raw ACS W2)", "d_acs_usd",
         [("window-averaged", "d_pred_usd"), ("rank-frozen", "d_frozen_usd")]),
        ("d W2 z-score (diagnostic only)", "d_label", [("window-averaged", "d_pred_z")]),
    )
    rows = []
    for split in [None, *[g for g in CSA_SPLIT_GROUPS if (pairs["split_group"] == g).any()]]:
        in_split = (pairs["split_group"] == split) if split else pairs["split_group"].notna()
        for gname, mask in (("stable", masks["stable"]), (chg, masks[chg])):
            sub = pairs[in_split & mask]
            for target, actual, preds in specs:
                for kind, pcol in preds:
                    ok = np.isfinite(sub[actual]) & np.isfinite(sub[pcol])
                    s = sub[ok]
                    met = _growth_metrics(s[actual].values, s[pcol].values)
                    lo_, hi_ = _cluster_bootstrap_ci(s[actual].values, s[pcol].values,
                                                     s["GEOID"].values, _r2_oos, n_boot=n_boot)
                    rows.append({"split": CSA_SPLIT_LABELS.get(split, "All (held out + train)"),
                                 "group": gname, "target": target, "prediction": kind, **met,
                                 "r2_oos_ci_lo": lo_, "r2_oos_ci_hi": hi_,
                                 "mean_actual": float(s[actual].mean()),
                                 "mean_predicted": float(s[pcol].mean())})
    table = pd.DataFrame(rows)
    table.to_csv(out / "tables" / "F_acs_window.csv", index=False)
    pairs.to_csv(out / "tables" / "F_acs_window_pairs.csv", index=False)
    for c in ("d_label_usd", "d_pred_usd", "d_acs_usd"):
        v = pairs[c].dropna()
        ss = ((v - v.mean()) ** 2).sort_values(ascending=False)
        print(f"  leverage check {c}: top-5 tracts carry {ss.iloc[:5].sum() / ss.sum():.1%} "
              f"of the sum of squares (n={len(v):,})")

    # ── figure: Δ$ and Δlog$, stable vs built, one colour each ──────────────
    in_any = pairs["split_group"].notna()
    st, ch = pairs[in_any & masks["stable"]].copy(), pairs[in_any & masks[chg]].copy()
    for d in (st, ch):
        d["d_label_k"], d["d_pred_k"] = d["d_label_usd"] / 1e3, d["d_pred_usd"] / 1e3
    built = f"Built $\\geq${_tex_pct(threshold)}"
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_SIZE_TWO_COL[0], FIG_SIZE_TWO_COL[0] * 0.52),
                                     constrained_layout=True)
    _panel_group_scatter(ax_a, {"Stable": (st, _GROUP_COLORS["stable"]),
                                built: (ch, _GROUP_COLORS["change"])},
                         x="d_label_k", y="d_pred_k")
    ax_a.set_xlabel(rf"$\Delta$ wealth per capita, ACS {acs_pre}$\to${acs_post} (\$k, 2023)")
    ax_a.set_ylabel(r"$\Delta$ predicted wealth per capita (\$k, 2023)")
    _panel_title(ax_a, "A")
    _panel_group_scatter(ax_b, {"Stable": (st, _GROUP_COLORS["stable"]),
                                built: (ch, _GROUP_COLORS["change"])},
                         x="dlog_label_usd", y="dlog_pred_usd")
    ax_b.set_xlabel(rf"$\Delta$ log wealth per capita, ACS {acs_pre}$\to${acs_post}")
    ax_b.set_ylabel(r"$\Delta$ log predicted wealth per capita")
    _panel_title(ax_b, "B")
    fig.suptitle(rf"NYC, GB2 wealth dollars per ACS vintage; predictions averaged over each "
                 rf"survey window ({min(w_pre)}--{max(w_pre)} vs {min(w_post)}--{max(w_post)})",
                 fontsize=7.5)
    _savefig(fig, out / "figures" / "F_acs_window_scatter.pdf")

    # ── direction accuracy: does the predicted change have the right sign? ──
    preds = {"window-averaged": "d_pred_usd", single: "d_pred_single_usd",
             "rank-frozen": "d_frozen_usd"}
    dir_groups = {"Stable": masks["stable"], f"Built >={threshold:.0%}": masks[chg]}
    direction = _direction_table(pairs, dir_groups, preds, n_boot=n_boot)
    direction.to_csv(out / "tables" / "F_acs_window_direction.csv", index=False)
    pooled = direction[direction["split"] == "All (held out + train)"]
    colors = {"window-averaged": _GROUP_COLORS["change"], single: "#E69F00",
              "rank-frozen": "#999999"}
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_SIZE_TWO_COL[0], FIG_SIZE_TWO_COL[0] * 0.42),
                                     constrained_layout=True)
    for ax, definition, letter, ttl in (
            (ax_a, "absolute", "A", r"Sign of $\Delta$ wealth \$ (GB2)"),
            (ax_b, "relative to city", "B",
             r"Sign of $\Delta$ wealth \$ minus the city's median $\Delta$")):
        _plot_direction(ax, pooled[pooled["definition"] == definition], list(dir_groups),
                        list(preds), colors, letter, ttl)
    ax_a.legend(fontsize=6, loc="lower left", frameon=True, facecolor="white",
                edgecolor="none", framealpha=0.85)
    fig.suptitle(rf"NYC direction accuracy, ACS {acs_pre}$\to${acs_post} "
                 r"(held out + train; bars: 95\% CI; dotted: 0.5)", fontsize=7.5)
    _savefig(fig, out / "figures" / "F_acs_window_direction.pdf")
    for r in direction.itertuples(index=False):
        print(f"  dir {r.split:22s} {r.group:10s} {r.prediction:26s} {r.definition:16s} n={r.n:4d} "
              f"acc={r.accuracy:.3f} [{r.acc_ci_lo:.3f}, {r.acc_ci_hi:.3f}] "
              f"majority={r.majority:.3f} balanced={r.balanced:.3f} kappa={r.kappa:+.3f} "
              f"up actual/pred={r.share_actual_up:.2f}/{r.share_pred_up:.2f}")

    for r in table.itertuples(index=False):
        print(f"  {r.split:22s} {r.group:10s} {r.target[:28]:28s} {r.prediction:26s} n={r.n:4d} "
              f"R2_oos={r.r2_oos:+.3f} [{r.r2_oos_ci_lo:+.3f}, {r.r2_oos_ci_hi:+.3f}] "
              f"R2_ols={r.r2_ols:.4f} rho={r.spearman:+.3f} "
              f"mean actual={r.mean_actual:+.3f} predicted={r.mean_predicted:+.3f}")
    return table


# ─── Khachiyan et al. (2022) per-capita benchmark ─────────────────────────────
# Online appendix, Appendix Table 2: R^2 = 1 - SSR/TSS (appendix "Computing R2
# Values") of the predicted vs actual 2000->2010 change in log total personal
# income per person (log Y_2010 - log Y_2000, 2012 dollars), test urban areas,
# from a differences model trained end-to-end on that change.
_KHACHIYAN_PC_DIFF_R2 = (
    ("2.4 km, without initial conditions", 0.0461),
    ("2.4 km, with initial conditions", 0.0674),
    ("1.2 km, without initial conditions", 0.0306),
    ("1.2 km, with initial conditions", 0.0653),
)


def _window_pc_income_pairs(tract: pd.DataFrame, income_long: pd.DataFrame,
                            params_by_city: dict, acs_pre: int, acs_post: int,
                            imagery_years, winsor: float | None = 0.01,
                            span: int = _ACS_WINDOW_SPAN) -> pd.DataFrame:
    """Per-tract Δlog per-capita income, actual vs predicted, on ACS windows.

    ``tract`` [GEOID, cbsa, year, pred]: each tract's prediction is averaged
    over its flights inside each ACS vintage's survey window, ranked (Hazen)
    within its city, mapped through that city-vintage's GB2 (``params_by_city
    [cbsa][vintage]``) and clipped at P1/P99 within the city-vintage.
    ``income_long`` [GEOID, year, usd] gives the actual per-capita income in
    constant dollars. Returns GEOID, cbsa, pred/actual levels, ``dlog_actual``,
    ``dlog_pred`` and ``dlog_frozen`` (pre rank under the post GB2 — no
    post-window imagery). Tracts lacking a flight in either window, a GB2 fit
    or a positive income are dropped.
    """
    w_pre = _acs_window_years(acs_pre, imagery_years, span)
    w_post = _acs_window_years(acs_post, imagery_years, span)
    pre = _window_average(tract, w_pre).rename(columns={"mean": "pred_pre", "n_years": "n_pre"})
    post = _window_average(tract, w_post).rename(columns={"mean": "pred_post", "n_years": "n_post"})
    out = pre.merge(post, on="GEOID")
    out["cbsa"] = out["GEOID"].map(tract.drop_duplicates("GEOID").set_index("GEOID")["cbsa"]).astype(str)
    out = out[out["cbsa"].isin(params_by_city.keys())].copy()
    out = out[[acs_pre in params_by_city[c] and acs_post in params_by_city[c]
               for c in out["cbsa"]]].reset_index(drop=True)
    for when, col in (("pre", "pred_pre"), ("post", "pred_post")):
        out[f"prob_{when}"] = _hazen_within(out.assign(year=0), col, by=("cbsa",)).values
    usd = {"usd_pre": [], "usd_post": [], "usd_frozen": []}
    for r in out.itertuples(index=False):
        prm = params_by_city[r.cbsa]
        usd["usd_pre"].append(_gb2_quantile(r.prob_pre, prm[acs_pre]))
        usd["usd_post"].append(_gb2_quantile(r.prob_post, prm[acs_post]))
        usd["usd_frozen"].append(_gb2_quantile(r.prob_pre, prm[acs_post]))
    for k, v in usd.items():
        out[k] = np.asarray(v, float)
    if winsor:
        for k in usd:
            out[k] = out.groupby("cbsa")[k].transform(lambda s: _winsorize(s, winsor))
    inc = income_long.set_index(["GEOID", "year"])["usd"]
    for when, yr in (("pre", acs_pre), ("post", acs_post)):
        out[f"actual_{when}"] = inc.reindex(
            pd.MultiIndex.from_arrays([out["GEOID"], np.full(len(out), yr)])).values
    ok = (out["actual_pre"] > 0) & (out["actual_post"] > 0)
    out = out[ok].reset_index(drop=True)
    out["dlog_actual"] = np.log(out["actual_post"] / out["actual_pre"])
    out["dlog_pred"] = np.log(out["usd_post"] / out["usd_pre"])
    out["dlog_frozen"] = np.log(out["usd_frozen"] / out["usd_pre"])
    out.attrs.update({"window_pre": w_pre, "window_post": w_post})
    return out


def _cross_fitted_calibration(actual: pd.Series, pred: pd.Series, groups: pd.Series,
                              fit_mask: pd.Series | None = None) -> pd.Series:
    """Out-of-sample linear calibration of ``pred`` to ``actual``'s scale.

    Khachiyan et al.'s differences model is trained by MSE on the change, so
    its predictions arrive shrunk to the change's scale; ours are differences
    of two level predictions and are not. This gives ours the same single
    property without touching the scored rows: with ``fit_mask`` None, each
    group (city) is scored with intercept and slope fitted on all OTHER groups
    (leave-one-city-out); with ``fit_mask``, one fit on those rows is applied
    to every row (NYC: fit on train tracts, score held-out tracts).
    """
    a, p = actual.astype(float), pred.astype(float)
    ok = np.isfinite(a) & np.isfinite(p)
    out = pd.Series(np.nan, index=a.index)
    if fit_mask is not None:
        m = ok & fit_mask
        if m.sum() >= 3:
            b, c = np.polyfit(p[m], a[m], 1)
            out[ok] = c + b * p[ok]
        return out
    for g in pd.unique(groups[ok]):
        train = ok & (groups != g)
        if train.sum() >= 3:
            b, c = np.polyfit(p[train], a[train], 1)
            test = ok & (groups == g)
            out[test] = c + b * p[test]
    return out


def _khachiyan_pc_rows(pairs: pd.DataFrame, label: str, n_boot: int = _GROWTH_N_BOOT,
                       calibrated: pd.Series | None = None) -> list[dict]:
    """R^2 = 1 - SSE/SST of Δlog per-capita income (Khachiyan's statistic),
    pooled and within city, for the model and the rank-frozen benchmark, plus
    the out-of-sample calibrated model when ``calibrated`` is given (rows where
    it is NaN are left out of that row). CIs resample cities, or tracts when
    there is a single city."""
    clusters = pairs["cbsa"] if pairs["cbsa"].nunique() > 1 else pairs["GEOID"]
    preds = [("model", pairs["dlog_pred"]), ("rank-frozen", pairs["dlog_frozen"])]
    if calibrated is not None:
        preds.append(("model, out-of-sample calibrated", calibrated))
    rows = []
    for who, col in preds:
        for scope in ("pooled", "within-city"):
            m = col.notna()
            r = _metric_row(pairs.loc[m, "dlog_actual"], col[m], clusters[m],
                            target="dlog per-capita income", predictor=who, scope=scope,
                            period=label, n_boot=n_boot)
            rows.append(r)
    return rows


def part_f_khachiyan(results_dir: Path, processed_dir: Path, out: Path,
                     years: list[int] | None = None, acs_pre: int = 2018,
                     acs_post: int = 2023, split_of: dict | None = None,
                     n_boot: int = _GROWTH_N_BOOT, tag: str = "") -> pd.DataFrame | None:
    """NYC: Khachiyan et al.'s per-capita change R^2 on the city's prediction pass.

    Same construction as ``part_khachiyan_pc_us`` with one city: the GB2 is
    fitted to the per-capita income of the NYC 5-borough tracts (the CSA
    city's county prefixes) per ACS vintage. Rows for all tracts and for each
    split group in ``split_of`` (GEOID -> group; defaults to the US split's
    held-out / train grouping). Writes tables/F_khachiyan_pc{tag}.csv.
    """
    from src import csa_event_study as ces

    print(f"\n=== Part F (Khachiyan per-capita benchmark): NYC, ACS {acs_pre}->{acs_post} {tag} ===")
    spec = ces.CSA_CITIES["nyc"]
    years = sorted(years) if years else _available_years(results_dir) or list(YEARS)
    tl = _load_tract_long(results_dir, years)
    if tl.empty:
        print("  no NYC tract predictions — skipped")
        return None
    # _load_tract_long carries the raw GEOID beside the padded GEOID_str: drop the
    # raw one first so the rename does not leave two GEOID columns.
    tract = tl.drop(columns=["GEOID"], errors="ignore").rename(
        columns={"GEOID_str": "GEOID", "predicted_value": "pred"})[["GEOID", "year", "pred"]].dropna()
    tract = tract[tract["GEOID"].str[:5].isin(spec.geoid_prefixes)].assign(cbsa="nyc")
    inc = _load_wealth_dollars_long(processed_dir, "inc", [acs_pre, acs_post])
    inc = inc[inc["GEOID"].str[:5].isin(spec.geoid_prefixes)].assign(cbsa="nyc")
    params = _fit_city_gb2(inc, ["nyc"])
    pairs = _window_pc_income_pairs(tract, inc, params, acs_pre, acs_post, years)
    print(f"  windows {pairs.attrs['window_pre']} vs {pairs.attrs['window_post']}; {len(pairs):,} tracts")
    split_of = _csa_split_of(processed_dir) if split_of is None else split_of
    pairs["split_group"] = pairs["GEOID"].map(split_of)
    train_groups = [g for g in pairs["split_group"].dropna().unique() if str(g).startswith("train")]
    fit_mask = pairs["split_group"].isin(train_groups)
    cal = (_cross_fitted_calibration(pairs["dlog_actual"], pairs["dlog_pred"], pairs["cbsa"],
                                     fit_mask=fit_mask) if fit_mask.sum() >= 3 else None)
    rows = []
    for group in [None, *sorted(pairs["split_group"].dropna().unique())]:
        sub = pairs if group is None else pairs[pairs["split_group"] == group]
        # Calibrated row only where it is out of sample: tracts outside the fit.
        sub_cal = None if cal is None else cal[sub.index].where(~fit_mask[sub.index])
        horizon = f"ACS {acs_pre}->{acs_post} ({acs_post - acs_pre}y)"
        for r in _khachiyan_pc_rows(sub, horizon, n_boot, calibrated=sub_cal):
            if r["scope"] == "pooled":
                r["tracts"] = "all" if group is None else str(group)
                rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(out / "tables" / f"F_khachiyan_pc{tag}.csv", index=False)
    for r in rows:
        print(f"  {r['tracts']:12s} {r['predictor'][:32]:32s} n={r['n']:5d} R2={r['r2_oos']:+.4f} "
              f"[{r['r2_oos_ci_lo']:+.4f}, {r['r2_oos_ci_hi']:+.4f}] R2_ols={r['r2_ols']:.4f} "
              f"rho={r['spearman']:+.3f}")
    return table


def part_khachiyan_pc_us(ctx: _USContext, acs_pre: int = 2018, acs_post: int = 2023,
                         n_boot: int = _GROWTH_N_BOOT,
                         income_long: pd.DataFrame | None = None) -> dict:
    """Khachiyan et al.'s per-capita change R^2 on the whole-city test sample.

    5-year ACS vintages (``acs_pre`` vs ``acs_post``) against predictions
    averaged over each survey window; per-capita income GB2 per test city and
    vintage (unsmoothed). Writes tables/US_khachiyan_pc.csv with our rows and
    Appendix Table 2's benchmark rows.
    """
    print(f"\n=== Khachiyan per-capita benchmark (US test cities, ACS {acs_pre}->{acs_post}) ===")
    sample = _figure_test_sample(ctx, what="Khachiyan benchmark")
    if sample is None:
        return {}
    tract = sample["tract_test"][["GEOID", "cbsa", "year", "pred"]].copy()
    tract["cbsa"] = tract["cbsa"].astype(str)
    if income_long is None:
        income_long = _load_wealth_dollars_long(ctx.processed_dir, "inc", [acs_pre, acs_post])
    params = _fit_city_gb2(income_long, tract["cbsa"].unique())
    pairs = _window_pc_income_pairs(tract, income_long, params, acs_pre, acs_post,
                                    sorted(tract["year"].unique()))
    print(f"  windows {pairs.attrs['window_pre']} vs {pairs.attrs['window_post']}; "
          f"{len(pairs):,} tracts in {pairs['cbsa'].nunique()} cities")
    cal = _cross_fitted_calibration(pairs["dlog_actual"], pairs["dlog_pred"], pairs["cbsa"])
    rows = _khachiyan_pc_rows(pairs, f"ACS {acs_pre}->{acs_post} (5y)", n_boot, calibrated=cal)
    bench = [{"source": "Khachiyan et al. (2022) App. Table 2", "target": "dlog per-capita income",
              "predictor": spec, "scope": "pooled (test urban areas)",
              "period": "2000->2010 (10y)", "r2_oos": r2} for spec, r2 in _KHACHIYAN_PC_DIFF_R2]
    table = pd.DataFrame(rows + bench)
    table.to_csv(ctx.out / "tables" / "US_khachiyan_pc.csv", index=False)
    pairs.to_csv(ctx.out / "tables" / "US_khachiyan_pc_pairs.csv", index=False)
    for r in rows:
        print(f"  {r['predictor'][:32]:32s} {r['scope']:11s} n={r['n']:5d} R2={r['r2_oos']:+.4f} "
              f"[{r['r2_oos_ci_lo']:+.4f}, {r['r2_oos_ci_hi']:+.4f}] R2_ols={r['r2_ols']:.4f} "
              f"rho={r['spearman']:+.3f}")
    model = rows[0]
    return {"khachiyan_pc/r2": model["r2_oos"], "khachiyan_pc/r2_ols": model["r2_ols"],
            "khachiyan_pc/n": model["n"]}


_US_PARTS = {
    "cross": part_a_us, "temporal": part_b_us, "dollars": part_c_us,
    "main_figure": part_main_figure_us, "growth": part_growth_figure_us,
    "khachiyan": part_khachiyan_pc_us,
}
# main_figure's Panel B carries a synthetic (non-measured) cardinal-baseline
# line, clearly marked with an asterisk in the figure and caption. It is in the
# 'both' default set (that figure is a deliverable) but NOT in the us-only
# default, so an automated US run never emits it by accident.
_DEFAULT_US_PARTS = ("cross", "temporal", "dollars")
_ALL_US_PARTS = ("cross", "temporal", "dollars", "main_figure", "growth", "khachiyan")

_NYC_PARTS = ("A", "B", "C", "D", "E", "F")
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

    US parts are named ('cross', 'temporal', 'dollars', 'main_figure', 'growth'); NYC parts
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
    csa_screen: float | None = None,
    csa_anticipation=CSA_ANTICIPATION_ARMS,
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
                # The CSA cities' dense per-city predictions live under the RUN's
                # results dir (csa/<city>/), not under nyc_zarr_check — different
                # sensor, different sampling design, so a separate tree.
                nyc_summary = _run_nyc_parts(nyc_dir, processed_dir, nyc_out,
                                             nyc_parts,
                                             csa_results_dir=results_dir,
                                             csa_screen=csa_screen,
                                             csa_anticipation=csa_anticipation)
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


def _run_nyc_parts(results_dir, processed_dir, out_dir, parts,
                   csa_results_dir=None, csa_screen=None,
                   csa_anticipation=CSA_ANTICIPATION_ARMS) -> dict:
    """Dispatch the NYC parts A–F, each isolated so one failure is not fatal.

    Part C returns the GB2 quantile mapping that part E prefers for its tract
    chart, so C is run before E and its result threaded through; if C failed or
    was not requested, E falls back to the raw predictions.
    """
    requested = [p for p in _NYC_PARTS if p in
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
                # The unrestricted arm always runs and keeps the untagged output
                # paths. `csa_screen` adds a second, fully independent pass whose
                # figures/tables/keys are suffixed — the two together are the
                # robustness evidence for the sample restriction, so producing
                # one without the other would defeat the point.
                screens = [(None, "")]
                if csa_screen is not None:
                    screens.append((float(csa_screen),
                                    f"_screen{int(csa_screen*100)}"))
                # Anticipation is the outer loop so each folder is finished
                # before the next starts, and a run interrupted partway leaves
                # whole arms rather than half of each.
                # The per-city footprint/prediction inputs are identical across
                # every (arm, screen) pair, so they are computed on the first
                # pass and reused; the cache is dropped once the last one is
                # done, before part E starts allocating.
                try:
                    for antic in (csa_anticipation or (0,)):
                        for screen, tag in screens:
                            result = part_d(results_dir, processed_dir, out_dir,
                                            years=years,
                                            csa_results_dir=csa_results_dir,
                                            max_undated_area_share=screen,
                                            anticipation=int(antic), out_tag=tag)
                            if isinstance(result, dict):
                                summary.update(result)
                finally:
                    _csa_cache_clear()
            elif part == "E":
                part_e(results_dir, processed_dir, out_dir,
                       qmap_long=qmap_long, years=years)
            elif part == "F":
                part_f(results_dir, processed_dir, out_dir, years=years)
                part_f_acs_window(results_dir, processed_dir, out_dir, years=years)
                part_f_khachiyan(results_dir, processed_dir, out_dir, years=years)
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
             "temporal dollars main_figure growth. NYC: A (cross-section) B (temporal) "
             "C (GB2 dollars) D (construction-cohort CSA event study) E (Hudson "
             "Yards) F (growth R2 by construction cohort). Defaults: 'cross temporal dollars' (us), 'A B C' (nyc), "
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
    parser.add_argument(
        "--csa-screen", type=float, default=None, metavar="SHARE",
        help="Also run Part D a second time with the undated-area sample "
             "restriction at SHARE (e.g. 0.20), writing a parallel set of "
             "figures/tables suffixed _screen<NN>. The unrestricted arm always "
             "runs; this adds the restricted one so the pair can be compared."
    )
    parser.add_argument(
        "--csa-anticipation", type=int, nargs="*", default=None, metavar="PERIODS",
        help="Anticipation arms for Part D, in panel periods (default: "
             f"{' '.join(map(str, CSA_ANTICIPATION_ARMS))}). Each value runs a "
             "complete, independent Part D into its own "
             "<out>/csa_anticipation<N>/ folder. A period is not the same number "
             "of years in every city — 2 in NYC, 2-3 in Tampa — so read the arms "
             "against each city's cadence."
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
        csa_screen=args.csa_screen,
        csa_anticipation=(CSA_ANTICIPATION_ARMS if args.csa_anticipation is None
                          else tuple(args.csa_anticipation)),
    )


if __name__ == "__main__":
    main()