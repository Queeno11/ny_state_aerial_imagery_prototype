# -*- coding: utf-8 -*-
"""
src/evaluation.py  –  Post-hoc evaluation of already-computed predictions.

Parts:
  A  Cross-sectional validity in 2016
  B  Temporal stability
  C  Quantile-mapping baseline
  D  Case study: Hudson Yards

Usage:
    python -m src.evaluation                     # all parts
    python -m src.evaluation --parts A B         # subset
    python -m src.evaluation --savename <name>
"""

from __future__ import annotations

import argparse
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
from scipy.stats import spearmanr, kendalltau
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
from src.geo_utils import calculate_exact_tau

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─── constants ────────────────────────────────────────────────────────────────

### Figure styling constants

# Figsize, inspired by journal guidelines, 
# https://www.elsevier.com/authors/policies-and-guidelines/artwork-and-media-instructions/artwork-sizing-instructions
pt = 1./72.27 # Hundreds of years of history... 72.27 points to an inch.
journal_sizes = {
    "Latex": {"onecol": 354.*pt, "twocol": (354-35)/2*pt},
    "CQG": {"onecol": 374.*pt}, # CQG is only one column
    # Add more journals below. Can add more properties to each journal
}
my_width = journal_sizes["Latex"]["onecol"]
# Our figure's aspect ratio
golden = (1 + 5 ** 0.5) / 2
FIG_DPI = 300
FIG_SIZE_ONE_COL = (my_width, my_width/golden)
FIG_SIZE_TWO_COL = (my_width, my_width/golden)

# Style
import matplotlib.pyplot as plt
plt.style.use(PROJECT_ROOT / "src" / "utils" / "paper.mplstyle")
###

YEARS = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
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


def _load_tract_long(results_dir: Path) -> pd.DataFrame:
    """Stack predictions_by_tract_<year>.parquet for all YEARS ->long DF."""
    frames = []
    for yr in YEARS:
        df = pd.read_parquet(results_dir / f"predictions_by_tract_{yr}.parquet")
        df["year"] = yr
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    long["GEOID_str"] = long["GEOID"].apply(_norm_geoid)
    return long


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

# ─── Part A ───────────────────────────────────────────────────────────────────

def part_a(results_dir: Path, processed_dir: Path, out: Path) -> None:
    print("\n=== Part A: Cross-sectional validity 2016 ===")

    tract = pd.read_parquet(results_dir / "predictions_by_tract_2016.parquet")
    x = tract["predicted_value"].values.astype(float)
    y = tract["Rel_Score"].values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    rho, p_rho = spearmanr(x, y)
    tau_k, p_tau = kendalltau(x, y)
    _, lo, hi = _bootstrap_spearman(x, y)
    _, lo_k, hi_k = _bootstrap_kendall(x, y)


    print(f"  Spearman ρ = {rho:.3f}  (95% CI [{lo:.3f}, {hi:.3f}])  p={p_rho:.2e}")
    print(f"  Kendall τ  = {tau_k:.3f}  (95% CI [{lo_k:.3f}, {hi_k:.3f}])  p={p_tau:.2e}")

    # Building-level (not headline)
    bldg16 = gpd.read_parquet(results_dir / "predictions_2016.parquet")
    bx = bldg16["predicted_value"].values.astype(float)
    by = bldg16["Rel_Score"].values.astype(float)
    bm = np.isfinite(bx) & np.isfinite(by)
    brho, _ = spearmanr(bx[bm], by[bm])
    print(f"  [building-level, NOT headline] ρ = {brho:.3f}  n={bm.sum():,}")

    # Table
    pd.DataFrame([
        {"metric": "Spearman_rho", "value": rho, "p_value": p_rho,
         "ci_lo_95": lo, "ci_hi_95": hi, "n": int(mask.sum())},
        {"metric": "Kendall_tau", "value": tau_k, "p_value": p_tau,
         "ci_lo_95": np.nan, "ci_hi_95": np.nan, "n": int(mask.sum())},
        {"metric": "Spearman_rho_building", "value": brho, "p_value": np.nan,
         "ci_lo_95": np.nan, "ci_hi_95": np.nan, "n": int(bm.sum())},
    ]).to_csv(out / "tables" / "A_cross_sectional_2016.csv", index=False)

    # Scatter with 10-bin overlay
    fig, ax = plt.subplots(figsize=FIG_SIZE_TWO_COL)
    ax.scatter(y, x, s=4, alpha=0.22, color="steelblue", linewidths=0)

    bin_edges = np.percentile(x, np.linspace(0, 100, 11))
    bin_ids = np.digitize(x, bin_edges[1:-1])  # 0..19
    bx_m = [x[bin_ids == b].mean() for b in range(10) if (bin_ids == b).any()]
    by_m = [y[bin_ids == b].mean() for b in range(10) if (bin_ids == b).any()]
    ax.plot(by_m, bx_m, "o-", color="firebrick", ms=5, lw=1.5, label="10-bin mean")
    ax.set_xlim(-3, 4)
    ax.set_xlabel("ACS Tract Z-score")
    ax.set_ylabel("Average Tract Predicted Value")
    # Add text to the plot with the statistics
    ax.text(0.05, 0.87, f"Spearman $\\rho$ = {rho:.3f}\nKendall $\\tau$ = {tau_k:.3f}",
            transform=ax.transAxes, fontsize=8, 
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, out / "figures" / "A_scatter_2016.pdf")


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


# ─── Part B ───────────────────────────────────────────────────────────────────

SPLIT_GROUPS = {
    "test":  ["test", "val_spatial_temporal", "val_spatial", "dead_zone"],
    "train": ["train"],
}


def part_b(results_dir: Path, processed_dir: Path, out: Path) -> None:
    print("\n=== Part B: Temporal stability (Tract-Level CSA DiD) ===")

    import csa
    import polars as pl

    splits = _load_splits(processed_dir)

    print("  B.1 Tract panel...")
    tract_long = _load_tract_long(results_dir)
    tract_long = tract_long.merge(splits[["GEOID_str", "type"]], on="GEOID_str", how="left")

    print("  B.2 Building footprints (loaded once)...")
    bldg_nyc = gpd.read_parquet(processed_dir / "buildings_nyc.parquet").to_crs(CRS_PROJ)
    bldg_nyc["area"] = bldg_nyc.geometry.area
    bldg_centroids = bldg_nyc.copy()
    bldg_centroids.geometry = bldg_centroids.geometry.centroid

    splits_geom = gpd.read_feather(processed_dir / "tract_splits.feather").to_crs(CRS_PROJ)
    splits_geom["GEOID_str"] = splits_geom["GEOID"].astype(str).str.zfill(11)

    def get_change_year(group, threshold):
        treated = group[group["change_pct"] > threshold]
        return treated["year"].min() if len(treated) > 0 else 0

    pd.DataFrame(metric_rows).to_csv(
        out / "tables" / "B_stability_metrics.csv", index=False
    thresholds = [0.01, 0.05, 0.1]
    n_rows, n_cols = len(SPLIT_GROUPS), len(thresholds)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(my_width * 1.5, my_width),
        sharey=False,
    )
    print("    saved B_stability_metrics.csv")

    legend_handles = None

    for row_i, (split_name, split_types) in enumerate(SPLIT_GROUPS.items()):
        print(f"\n  --- Split group: {split_name} ---")

        group_tract_long = tract_long[tract_long["type"].isin(split_types)].copy()
        group_tracts = splits_geom[splits_geom["type"].isin(split_types)][["GEOID_str", "geometry"]]

        print(f"  B.2 Tract-level change detection (Area-based) [{split_name}]...")
        bldg_tract = gpd.sjoin(bldg_centroids, group_tracts, how="inner", predicate="intersects")

        bldg_tract["base"] = bldg_tract["CONSTRUCTION_YEAR"] <= 2009
        bldg_tract["new"] = bldg_tract["CONSTRUCTION_YEAR"].between(2010, 2024, inclusive="both")

        base_area = bldg_tract[bldg_tract["base"]].groupby("GEOID_str")["area"].sum()
        new_bldgs = bldg_tract[bldg_tract["new"]]
        yearly_new_area = new_bldgs.groupby(["GEOID_str", "CONSTRUCTION_YEAR"])["area"].sum().reset_index()

        tract_ids = group_tracts["GEOID_str"].unique()
        all_years = list(range(2010, 2025))
        grid_all = pd.MultiIndex.from_product(
            [tract_ids, all_years], names=["GEOID_str", "year"]
        ).to_frame(index=False)

        merged = grid_all.merge(
            yearly_new_area,
            left_on=["GEOID_str", "year"],
            right_on=["GEOID_str", "CONSTRUCTION_YEAR"],
            how="left",
        )
        merged["area"] = merged["area"].fillna(0)
        merged["cum_area"] = merged.groupby("GEOID_str")["area"].cumsum()
        merged = merged.merge(base_area.rename("base_area"), on="GEOID_str", how="left")
        merged["base_area"] = merged["base_area"].fillna(np.inf)
        merged["change_pct"] = merged["cum_area"] / merged["base_area"]
        merged_panel = merged[merged["year"].isin(YEARS)]

        print(f"  B.3 CSA DiD Estimation [{split_name}]...")
        for col_i, thresh in enumerate(thresholds):
            ax = axes[row_i, col_i]
            col_name = f"change_year_{int(thresh * 100)}pct"
            change_years = merged_panel.groupby("GEOID_str").apply(
                get_change_year, threshold=thresh
            ).rename(col_name)
            df_thresh = group_tract_long.merge(change_years, on="GEOID_str", how="left")

            df_csa = df_thresh.dropna(subset=[col_name, "predicted_value"]).copy()
            df_csa[col_name] = df_csa[col_name].astype(int)
            df_csa["year"] = df_csa["year"].astype(int)
            df_csa = df_csa[df_csa["year"] != 2010]
            df_csa["year"] = (df_csa["year"] - 2010) // 2  # 2012->1, ..., 2024->7
            df_csa[col_name] = np.where(
                df_csa[col_name] == 0, 0, (df_csa[col_name] - 2010) // 2
            )
            df_csa["unit_id"] = pd.factorize(df_csa["GEOID_str"])[0]

            if df_csa[col_name].shape[0] <= 0:
                raise ValueError(f"No data for selected threshold: {int(thresh * 100)}%")

            try:
                np.random.seed(42)
                res = csa.estimate(
                    data=pl.from_pandas(df_csa),
                    outcome="predicted_value",
                    unit="unit_id",
                    group=col_name,
                    time="year",
                    control="never",
                    method="reg",
                )
                agg = csa.agg_te(res, method="dynamic", boot=True, B=10_000, verbose=True)

                if getattr(agg, "boot", None) is None:
                    print(f"      Skipping {int(thresh * 100)}% [{split_name}]: bootstrap did not run")
                    ax.text(0.5, 0.5, "bootstrap\nfailed", ha="center", va="center",
                            transform=ax.transAxes, fontsize=10, color="gray")
                    continue

                est_df = agg.boot.estimates.to_pandas()
                e_col = "k" if "k" in est_df.columns else (
                    "e" if "e" in est_df.columns else est_df.columns[0]
                )
                ks    = est_df[e_col].to_numpy()
                means = est_df["att"].to_numpy()
                lower = est_df["lower"].to_numpy()
                upper = est_df["upper"].to_numpy()

                if len(ks) == 0:
                    print(f"      Skipping {int(thresh * 100)}% [{split_name}]: no valid event times")
                    continue

                # Normalize to k=-1 reference period: subtract the k=-1 ATT from all three
                # arrays (means, lower, upper) as a rigid vertical shift. This anchors the
                # last pre-treatment period at 0 without altering CI width or coverage.
                ref_mask = ks == -1
                if ref_mask.any():
                    ref_att = means[ref_mask][0]
                    means = means - ref_att
                    lower = lower - ref_att
                    upper = upper - ref_att

                n_treated = int(df_csa[df_csa[col_name] > 0]["unit_id"].nunique())
                n_control = int(df_csa[df_csa[col_name] == 0]["unit_id"].nunique())
                print(f"    [{split_name}] {int(thresh * 100)}%: n_treated={n_treated}, n_control={n_control}")

                h_att, = ax.plot(ks, means, "o-", color="firebrick", lw=1, ms=4,
                                 label="ATT (CSA)")
                h_ci = ax.fill_between(ks, lower, upper, alpha=0.18, color="firebrick",
                                       label=r"95\% simultaneous CI")
                h_vline = ax.axvline(-0.5, color="black", ls="--", lw=0.8,
                                     label="Treatment year")
                ax.axhline(0, color="gray", ls="-", lw=0.6)

                if legend_handles is None:
                    legend_handles = [h_att, h_ci, h_vline]

                ax.set_xticks(ks)
                ax.tick_params(labelsize=8)
                if row_i == n_rows - 1:
                    ax.set_xlabel("Event time (relative to treatment)", fontsize=10)
                if col_i == 0:
                    if split_name == "test":
                        split_label = "Test Set"
                    elif split_name == "train":
                        split_label = "Train Set"
                    ax.set_ylabel(f"{split_label} ATT \n (Tract Avg. Prediction)", fontsize=10)
                ax.set_title(f"{int(thresh * 100)}\%", fontsize=10)

                csa.agg_te(res, method="simple").summary()

            except Exception as exc:
                print(f"    Error [{split_name}] {int(thresh * 100)}%: {exc}")
                ax.text(0.5, 0.5, f"error\n{exc}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="red")

    # fig.suptitle(
    #     "Tract-Level DiD Event Study (CSA): cumulative new building area thresholds",
    #     fontsize=11, fontweight="bold",
    # )
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            fontsize=10,
            framealpha=0.9,
            bbox_to_anchor=(0.5, 0.0),
        )
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    _savefig(fig, out / "figures" / "B_event_study_grid.pdf")


# ─── Part C ───────────────────────────────────────────────────────────────────

def _load_acs_nyc(year: int) -> pd.Series:
    """Per-capita income for NYC census tracts, indexed by 11-char GEOID."""
    fpath = ACS_ROOT_DIR / str(year) / f"ny_tracts_acs5_{year}.feather"
    df = pd.read_feather(fpath)
    nyc = df[df["geoid"].str.startswith(NYC_COUNTY_PREFIXES)].copy()
    nyc["geoid"] = nyc["geoid"].astype(str).str.zfill(11)
    nyc = nyc.drop_duplicates("geoid")
    return nyc.set_index("geoid")["per_capita_income_usd"].dropna()


def _qmap20(P: np.ndarray, A: np.ndarray) -> np.ndarray:
    p_q = np.nanpercentile(P, np.linspace(0, 100, 21))
    a_q = np.nanpercentile(A, np.linspace(0, 100, 21))
    return np.interp(P, p_q, a_q)


def _qmap_logn(P: np.ndarray, A: np.ndarray) -> np.ndarray:
    log_a = np.log(A[A > 0])
    mu, sigma = log_a.mean(), log_a.std()
    z = (P - np.nanmean(P)) / max(np.nanstd(P), 1e-12)
    return np.exp(mu + sigma * z)


def part_c(results_dir: Path, processed_dir: Path, out: Path) -> pd.DataFrame | None:
    print("\n=== Part C: Quantile-mapping baseline ===")

    tract_long = _load_tract_long(results_dir)

    city_rows = []
    qmap_frames = []
    for yr in YEARS:
        try:
            acs_series = _load_acs_nyc(yr)
        except FileNotFoundError:
            print(f"  ACS {yr} not found, skipping year")
            continue

        A   = acs_series.values.astype(float)        # full NYC ACS marginal (qmap target)
        sub = tract_long[tract_long["year"] == yr].copy()
        P   = sub["predicted_value"].values.astype(float)

        sub["qmap20"]   = _qmap20(P, A)
        sub["qmaplogn"] = _qmap_logn(P, A)
        # Actual ACS income for the SAME tracts that have predictions this year,
        # so the city-average lines compare like-with-like (sparse years have
        # far fewer predicted tracts than the full NYC marginal).
        sub["acs_actual"] = sub["GEOID_str"].map(acs_series)
        qmap_frames.append(sub)

        city_rows.append({
            "year": yr,
            "n_tracts":         int(sub["predicted_value"].notna().sum()),
            "ACS_mean_all_nyc": float(np.nanmean(A)),
            "ACS_mean_matched": float(np.nanmean(sub["acs_actual"].values)),
            "qmap20_mean":      float(np.nanmean(sub["qmap20"].values)),
            "qmaplogn_mean":    float(np.nanmean(sub["qmaplogn"].values)),
            "raw_mean":         float(np.nanmean(sub["predicted_value"].values)),
        })

    if not qmap_frames:
        print("  No ACS files available. Skipping Part C.")
        return None

    qmap_long = pd.concat(qmap_frames, ignore_index=True)

    # City-average trajectory
    city_df = pd.DataFrame(city_rows)
    city_df.to_csv(out / "tables" / "C_quantile_mapping.csv", index=False)
    print("    saved C_quantile_mapping.csv")

    fig, ax = plt.subplots(figsize=FIG_SIZE_TWO_COL)
    ax.plot(city_df["year"], city_df["ACS_mean_matched"], "o-",  color="black",     lw=2,   ms=6,
            label="ACS (same tracts as preds)")
    ax.plot(city_df["year"], city_df["ACS_mean_all_nyc"], ":",   color="0.55",      lw=1.5,
            label="ACS (all NYC tracts, context)")
    ax.plot(city_df["year"], city_df["qmap20_mean"],      "s--", color="steelblue", lw=1.5, ms=5, label="qmap20")
    ax.plot(city_df["year"], city_df["qmaplogn_mean"],    "^--", color="firebrick", lw=1.5, ms=5, label="qmaplogn")
    for _, r in city_df.iterrows():
        ax.annotate(f"n={int(r['n_tracts'])}", (r["year"], r["qmap20_mean"]),
                    textcoords="offset points", xytext=(0, -13), fontsize=6,
                    ha="center", color="steelblue")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean per-capita income (USD)")
    ax.set_title("City-average income: ACS vs quantile-mapped predictions\n"
                 "(ACS matched to the tracts predicted each year; n = tracts/year)")
    ax.legend(fontsize=8)
    ax.set_xticks(YEARS)
    fig.tight_layout()
    _savefig(fig, out / "figures" / "C_city_avg_trajectory.pdf")

    # Long-difference scatter: ACS 2009->2024 vs mapped prediction 2010->2024
    panel = pd.read_feather(processed_dir / "ny_tracts_panel_2009_2014_2019_2024.feather")
    panel["GEOID_str"] = panel["geoid_2022"].astype(str).str.zfill(11)
    panel["acs_change"] = (
        panel["per_capita_income_usd_2022"] - panel["per_capita_income_usd_2009"]
    )

    for map_col in ["qmap20", "qmaplogn"]:
        wide_map = qmap_long.pivot_table(
            index="GEOID_str", columns="year", values=map_col, aggfunc="first"
        )
        wide_map.columns = [int(c) for c in wide_map.columns]
        if 2010 not in wide_map.columns or 2022 not in wide_map.columns:
            continue
        wide_map["pred_change"] = wide_map[2022] - wide_map[2010]
        merged = (
            panel[["GEOID_str", "acs_change"]]
            .merge(wide_map[["pred_change"]].reset_index(), on="GEOID_str", how="inner")
            .dropna()
        )
        x_c = merged["acs_change"].values
        y_c = merged["pred_change"].values

        rho_c, _ = spearmanr(x_c, y_c)
        slope_c  = np.polyfit(x_c, y_c, 1)[0]

        fig, ax = plt.subplots(figsize=FIG_SIZE_TWO_COL)
        ax.scatter(x_c, y_c, s=5, alpha=0.3, color="steelblue", linewidths=0)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        xlim = np.array([x_c.min(), x_c.max()])
        ax.plot(xlim, np.polyval(np.polyfit(x_c, y_c, 1), xlim),
                "r-", lw=1.5, label=f"slope={slope_c:.3f}")
        ax.set_xlabel("ACS per-capita income change 2009→2022 (USD)")
        ax.set_ylabel(f"{map_col} change 2010→2022 (USD)")
        ax.set_title(
            f"Long-difference scatter ({map_col})  —  tracts predicted in both 2010 \\& 2022\n"
            f"Spearman $\\rho$ = {rho_c:.3f}  n = {len(merged)}"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        _savefig(fig, out / "figures" / f"C_long_diff_{map_col}.pdf")

    return qmap_long


# ─── Part D ───────────────────────────────────────────────────────────────────

def _lonlat_to_6539(lon: float, lat: float) -> tuple[float, float]:
    t = Transformer.from_crs(CRS_GEO, CRS_PROJ, always_xy=True)
    return t.transform(lon, lat)


def _slice_zarr(ds, cx: float, cy: float, half: float,
                max_px: int = 1400) -> np.ndarray | None:
    """Extract a square tile (4, H, W) uint8 from a zarr Dataset, strided so the
    longest side is ~max_px pixels (plan D.1: 'downsampled for size'). Striding
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


def part_d(
    results_dir: Path,
    processed_dir: Path,
    out: Path,
    qmap_long: pd.DataFrame | None = None,
) -> None:
    print("\n=== Part D: Case study – Hudson Yards ===")
    import xarray as xr

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
        print("  No buildings found. Skipping Part D.")
        return

    # Load predictions for those DOITT_IDs
    box_ids = set(bldg_proj.index.tolist())
    pred_by_yr: dict[int, gpd.GeoDataFrame] = {}
    for yr in YEARS:
        df = gpd.read_parquet(results_dir / f"predictions_{yr}.parquet")
        sub = df[df.index.isin(box_ids)]
        if len(sub) > 0:
            pred_by_yr[yr] = sub

    all_vals = np.concatenate(
        [df["predicted_value"].values for df in pred_by_yr.values()]
    )
    vmin, vmax = np.nanpercentile(all_vals, 2), np.nanpercentile(all_vals, 98)
    pred_norm  = Normalize(vmin=vmin, vmax=vmax)
    cmap_pred  = plt.cm.Spectral

    # ── D.1 8×3 grid ─────────────────────────────────────────────────────────
    print("  D.1 8×3 image grid...")
    n_rows = len(YEARS)
    fig_g, axes_g = plt.subplots(n_rows, 3, figsize=(12, n_rows * 3.0))

    for col_i, col_title in enumerate(
        ["Aerial image (RGB)", "Building polygons", "Model predictions"]
    ):
        axes_g[0, col_i].set_title(col_title, fontsize=9, fontweight="bold")

    for row_i, yr in enumerate(YEARS):
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
            ax_img.text(0.5, 0.5, f"Image\nunavailable\n{yr}",
                        ha="center", va="center", transform=ax_img.transAxes,
                        fontsize=8, color="gray")
            ax_img.set_xlim(cx - half_ft_box, cx + half_ft_box)
            ax_img.set_ylim(cy - half_ft_box, cy + half_ft_box)

        ax_img.set_ylabel(str(yr), fontsize=8, rotation=0, labelpad=28, va="center")
        ax_img.axis("off")

        # ── column 2: building polygons ──
        cy_yr = bldg_proj["CONSTRUCTION_YEAR"].fillna(0)
        dy_yr = bldg_proj["DEMOLITION_YEAR"].fillna(9999)
        exists_mask = (cy_yr <= yr) & (dy_yr > yr)
        new_mask    = bldg_proj["CONSTRUCTION_YEAR"].between(2009, yr, inclusive="right")

        bldg_old = bldg_proj[exists_mask & ~new_mask]
        bldg_new = bldg_proj[exists_mask & new_mask]

        if len(bldg_old) > 0:
            bldg_old.plot(ax=ax_bldg, color="0.65", edgecolor="none", alpha=0.75)
        if len(bldg_new) > 0:
            bldg_new.plot(ax=ax_bldg, color="crimson", edgecolor="none", alpha=0.85)

        ax_bldg.set_xlim(cx - half_ft_box, cx + half_ft_box)
        ax_bldg.set_ylim(cy - half_ft_box, cy + half_ft_box)
        ax_bldg.set_aspect("equal")
        ax_bldg.axis("off")

        # ── column 3: model predictions ──
        pyr = pred_by_yr.get(yr)
        if pyr is not None and len(pyr) > 0:
            pyr.plot(
                column="predicted_value", ax=ax_pred,
                norm=pred_norm, cmap=cmap_pred,
                edgecolor="none", alpha=0.85,
            )
        else:
            # This area is only predicted in the full years (2016, 2024); label
            # empty cells so the gap reads as missing data, not a render glitch.
            ax_pred.text(0.5, 0.5, f"No predictions\n{yr}",
                         ha="center", va="center", transform=ax_pred.transAxes,
                         fontsize=8, color="gray")
        ax_pred.set_xlim(cx - half_ft_box, cx + half_ft_box)
        ax_pred.set_ylim(cy - half_ft_box, cy + half_ft_box)
        ax_pred.set_aspect("equal")
        ax_pred.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap_pred, norm=pred_norm)
    sm.set_array([])
    cbar = fig_g.colorbar(sm, ax=axes_g[:, 2], shrink=0.5, pad=0.2)
    cbar.set_label("predicted_value", fontsize=8)

    patch_old = mpatches.Patch(color="0.65",    label="Pre-existing (built $\\leq 2009$)")
    patch_new = mpatches.Patch(color="crimson", label="New construction (built $\\geq 2009$)")
    fig_g.legend(handles=[patch_old, patch_new], loc="lower left",
                 fontsize=7, framealpha=0.85, ncol=2)

    fig_g.suptitle("Hudson Yards: 2010–2024 aerial imagery, buildings, predictions",
                   fontsize=11, fontweight="bold")
    fig_g.tight_layout(rect=[0, 0.03, 1, 0.97])
    _savefig(fig_g, out / "figures" / "D_hudson_yards_grid.pdf")

    # ── D.2 Per-building trajectories ─────────────────────────────────────────
    print("  D.2 per-building line chart...")
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
    ax_l.set_xticks(YEARS)
    fig_l.tight_layout()
    _savefig(fig_l, out / "figures" / "D_buildings_lines.pdf")

    # ── D.3 Tract ACS vs model predictions ───────────────────────────────────
    print("  D.3 tract ACS vs prediction chart...")
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
        print("    no overlapping tracts – skipping D.3")
        return

    panel = pd.read_feather(processed_dir / "ny_tracts_panel_2009_2014_2019_2024.feather")
    panel["GEOID_str"] = panel["geoid_2024"].astype(str).str.zfill(11)

    # Build qmap_long for these tracts if not supplied
    if qmap_long is not None and "qmap20" in qmap_long.columns:
        pred_col = "qmap20"
        q_sub    = qmap_long[qmap_long["GEOID_str"].isin(overlap_geoids)]
    else:
        # Rebuild minimal qmap for just these tracts, or fall back to raw
        tract_long = _load_tract_long(results_dir)
        q_frames = []
        for yr in YEARS:
            try:
                A = _load_acs_nyc(yr).values.astype(float)
            except FileNotFoundError:
                continue
            sub = tract_long[tract_long["year"] == yr].copy()
            sub["qmap20"] = _qmap20(sub["predicted_value"].values.astype(float), A)
            q_frames.append(sub)
        if q_frames:
            qmap_long = pd.concat(q_frames, ignore_index=True)
            pred_col  = "qmap20"
            q_sub     = qmap_long[qmap_long["GEOID_str"].isin(overlap_geoids)]
        else:
            tract_long = _load_tract_long(results_dir)
            pred_col   = "predicted_value"
            q_sub      = tract_long[tract_long["GEOID_str"].isin(overlap_geoids)]

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
        "Per-capita income (USD)" if pred_col != "predicted_value"
        else "predicted_value"
    )
    ax_t.set_title(
        "Hudson Yards tract: ACS income (markers) vs model prediction (line)"
    )
    ax_t.legend(fontsize=7, loc="upper left")
    ax_t.set_xticks(YEARS)
    fig_t.tight_layout()
    _savefig(fig_t, out / "figures" / "D_tract_acs_vs_pred.pdf")


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
        "--parts", nargs="*", default=list("ABCD"),
        metavar="PART",
        help="Which parts to run: A B C D  (default: all)"
    )
    args = parser.parse_args()

    results_dir = RESULTS_DIR / args.savename
    out_dir     = results_dir / "evaluation"
    _make_dirs(out_dir)

    parts = {p.upper() for p in args.parts}
    print(f"Results   : {results_dir}")
    print(f"Output    : {out_dir}")
    print(f"Parts     : {sorted(parts)}")
    print(f"Tile τ    : {_EXACT_TAU_M:.2f} m  ({TAU_FT:.1f} US-survey-ft)")

    qmap_long: pd.DataFrame | None = None

    if "A" in parts:
        part_a(results_dir, PROCESSED_DATA_DIR, out_dir)

    if "B" in parts:
        part_b(results_dir, PROCESSED_DATA_DIR, out_dir)

    if "C" in parts:
        qmap_long = part_c(results_dir, PROCESSED_DATA_DIR, out_dir)

    if "D" in parts:
        part_d(results_dir, PROCESSED_DATA_DIR, out_dir, qmap_long=qmap_long)

    print("\n=== Done ===")
    print(f"  Tables  ->{out_dir / 'tables'}")
    print(f"  Figures ->{out_dir / 'figures'}")


if __name__ == "__main__":
    main()