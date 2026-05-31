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
)
from src.geo_utils import calculate_exact_tau

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─── constants ────────────────────────────────────────────────────────────────

YEARS = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
# Year pair used for rank-autocorrelation: both 2016 (temporal holdout) and 2024
# are fully predicted (all tracts), unlike the sparse intermediate years.
RANK_PAIR = (2016, 2024)
CRS_PROJ = 6539   # NY Long Island, US survey feet
CRS_GEO  = 4326

TAU_METERS = 100
IMAGE_SIZE  = 224
_EXACT_TAU_M, _N = calculate_exact_tau(TAU_METERS, IMAGE_SIZE)
# EPSG:6539 native unit is US survey foot → 1 ft = 0.3048006096 m
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


def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {path.name}")


def _load_tract_long(results_dir: Path) -> pd.DataFrame:
    """Stack predictions_by_tract_<year>.parquet for all YEARS → long DF."""
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

    print(f"  Spearman ρ = {rho:.3f}  (95% CI [{lo:.3f}, {hi:.3f}])  p={p_rho:.2e}")
    print(f"  Kendall τ  = {tau_k:.3f}  p={p_tau:.2e}")

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

    # Scatter with 20-bin overlay
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=4, alpha=0.22, color="steelblue", linewidths=0)

    bin_edges = np.percentile(x, np.linspace(0, 100, 21))
    bin_ids = np.digitize(x, bin_edges[1:-1])  # 0..19
    bx_m = [x[bin_ids == b].mean() for b in range(20) if (bin_ids == b).any()]
    by_m = [y[bin_ids == b].mean() for b in range(20) if (bin_ids == b).any()]
    ax.plot(bx_m, by_m, "o-", color="firebrick", ms=5, lw=1.5, label="20-bin mean")

    ax.set_xlabel("predicted_value (tract mean)")
    ax.set_ylabel("Rel_Score (ACS z-score)")
    ax.set_title(
        f"2016 held-out year  |  Spearman ρ = {rho:.3f}  "
        f"(95% CI [{lo:.3f}, {hi:.3f}])"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, out / "figures" / "A_scatter_2016.png")


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
    changed_ex = bldg_wide[chng_test_any].dropna(subset=["change_year"]).sample(n_ex, random_state=528)
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
    _savefig(fig, out / "figures" / "B_building_trajectories.png")

    # Event study: align changed buildings at change_year = 0
    # Rules:
    #   – test and train are plotted as separate charts so the two can be compared
    #   – buildings with change_year == 2010 (no pre-event observations) or
    #     change_year == 2024 (no post-event observations) are excluded
    print("  B.4 event study...")
    ev_rng = np.random.default_rng(1)
    EXCLUDE_CHANGE_YEARS = {2010, 2024}

    def _build_event_df(
        bldg_wide: pd.DataFrame,
        split_type: str,
        n_stable_sample: int = 5_000,
    ) -> pd.DataFrame:
        """
        Build a long event-study DataFrame for one split (test or train).

        Changed group  – all changed buildings in `split_type` tracts whose
                         change_year is not in EXCLUDE_CHANGE_YEARS.
        Stable group   – up to `n_stable_sample` stable buildings from the
                         same split, event-time anchored at 2016 (arbitrary
                         calendar midpoint; the line should be flat regardless).
        """
        type_mask    = (bldg_wide["type"] == split_type).fillna(False)
        chg_mask     = bldg_wide["changed"].fillna(False).astype(bool)
        stb_mask     = ~chg_mask

        rows = []

        # ── changed buildings ──
        chg_pool = bldg_wide[type_mask & chg_mask]
        for _, row in chg_pool.iterrows():
            cy = row["change_year"]
            if pd.isna(cy) or int(cy) in EXCLUDE_CHANGE_YEARS:
                continue
            for yr in year_cols:
                v = row[yr]
                if pd.isna(v):
                    continue
                rows.append({
                    "event_time": yr - int(cy),
                    "value":      float(v),
                    "group":      "changed",
                })

        # ── stable reference ──
        stb_pool = bldg_wide[type_mask & stb_mask]
        n_sample = min(n_stable_sample, len(stb_pool))
        stb_sample = stb_pool.sample(n_sample, random_state=2)
        for _, row in stb_sample.iterrows():
            for yr in year_cols:
                v = row[yr]
                if pd.isna(v):
                    continue
                rows.append({
                    "event_time": yr - 2016,
                    "value":      float(v),
                    "group":      "stable",
                })

        return pd.DataFrame(rows)

    def _plot_event_study(
        ev_df: pd.DataFrame,
        split_type: str,
        out_path: Path,
    ) -> None:
        """Plot and save one event-study figure for a single split."""
        n_chg = ev_df[ev_df["group"] == "changed"]["event_time"].notna().sum()
        fig, ax = plt.subplots(figsize=(8, 5))

        for grp, color, label in [
            ("changed", "firebrick", "Changed buildings"),
            ("stable",  "steelblue", "Stable (reference)"),
        ]:
            sub = ev_df[ev_df["group"] == grp]
            times = sorted(sub["event_time"].unique())
            means, los, his, valid_times = [], [], [], []
            for t in times:
                vals = sub.loc[sub["event_time"] == t, "value"].dropna().values
                if len(vals) < 5:
                    continue
                boot = [
                    vals[ev_rng.integers(0, len(vals), len(vals))].mean()
                    for _ in range(500)
                ]
                means.append(float(vals.mean()))
                los.append(float(np.percentile(boot, 2.5)))
                his.append(float(np.percentile(boot, 97.5)))
                valid_times.append(t)
            if valid_times:
                ax.plot(valid_times, means, "o-", color=color,
                        lw=2, ms=5, label=label)
                ax.fill_between(valid_times, los, his,
                                alpha=0.18, color=color)

        ax.axvline(0, color="black", ls="--", lw=1, label="t = 0 (change_year)")
        ax.set_xlabel("Event time (years relative to change_year)")
        ax.set_ylabel("predicted_value")
        ax.set_title(
            f"Event study: model response around new construction\n"
            f"({split_type} tracts  |  change_year ∉ {{2010, 2024}})"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        _savefig(fig, out_path)

    # ── pooled event study (one chart per split, all valid change years) ──
    for split_type in ("test", "train"):
        ev_df = _build_event_df(bldg_wide, split_type)
        fname = f"B_event_study_{split_type}.png"
        _plot_event_study(ev_df, split_type, out / "figures" / fname)

    # ── per-change-year event study ──
    # Only years with a genuine pre AND post window are plotted.
    # 2010 → no pre-event observations.
    # 2012 → only one pre-event year (2010); too thin.
    # 2022 → only one post-event year (2024); too thin.
    # 2024 → no post-event observations.
    # Valid set: 2014, 2016, 2018, 2020.
    PER_YEAR_CHANGE_YEARS = [2014, 2016, 2018, 2020]

    def _build_event_df_single_year(
        bldg_wide: pd.DataFrame,
        split_type: str,
        target_change_year: int,
        n_stable_sample: int = 5_000,
    ) -> pd.DataFrame:
        """
        Like _build_event_df but restricts the changed group to buildings
        whose change_year == target_change_year exactly.  The stable reference
        pool is drawn from the same split and anchored at target_change_year
        (so event_time=0 on the stable line is the same calendar year as the
        treatment, making the two lines directly comparable).
        """
        type_mask = (bldg_wide["type"] == split_type).fillna(False)
        chg_mask  = bldg_wide["changed"].fillna(False).astype(bool)
        stb_mask  = ~chg_mask

        rows = []

        # ── changed: only this change_year ──
        chg_pool = bldg_wide[type_mask & chg_mask]
        for _, row in chg_pool.iterrows():
            cy = row["change_year"]
            if pd.isna(cy) or int(cy) != target_change_year:
                continue
            for yr in year_cols:
                v = row[yr]
                if pd.isna(v):
                    continue
                rows.append({
                    "event_time": yr - target_change_year,
                    "value":      float(v),
                    "group":      "changed",
                })

        # ── stable reference: anchor at target_change_year ──
        stb_pool   = bldg_wide[type_mask & stb_mask]
        n_sample   = min(n_stable_sample, len(stb_pool))
        stb_sample = stb_pool.sample(n_sample, random_state=2)
        for _, row in stb_sample.iterrows():
            for yr in year_cols:
                v = row[yr]
                if pd.isna(v):
                    continue
                rows.append({
                    "event_time": yr - target_change_year,
                    "value":      float(v),
                    "group":      "stable",
                })

        return pd.DataFrame(rows)

    def _plot_event_study_single_year(
        ev_df: pd.DataFrame,
        split_type: str,
        target_change_year: int,
        out_path: Path,
    ) -> None:
        """
        Plot one event-study panel for a single change_year cohort.
        The x-axis label shows the actual calendar years so the reader can
        orient without having to do the arithmetic themselves.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        for grp, color, label in [
            ("changed", "firebrick", "Changed buildings"),
            ("stable",  "steelblue", "Stable (reference)"),
        ]:
            sub = ev_df[ev_df["group"] == grp]
            if sub.empty:
                continue
            times = sorted(sub["event_time"].unique())
            means, los, his, valid_times = [], [], [], []
            for t in times:
                vals = sub.loc[sub["event_time"] == t, "value"].dropna().values
                if len(vals) < 5:
                    continue
                boot = [
                    vals[ev_rng.integers(0, len(vals), len(vals))].mean()
                    for _ in range(500)
                ]
                means.append(float(vals.mean()))
                los.append(float(np.percentile(boot, 2.5)))
                his.append(float(np.percentile(boot, 97.5)))
                valid_times.append(t)
            if valid_times:
                ax.plot(valid_times, means, "o-", color=color,
                        lw=2, ms=5, label=label)
                ax.fill_between(valid_times, los, his,
                                alpha=0.18, color=color)

        # Mark t=0 and annotate with the actual calendar year
        ax.axvline(0, color="black", ls="--", lw=1,
                   label=f"t = 0  ({target_change_year})")

        # Secondary x-axis tick labels: show calendar year alongside event time
        event_times = [yr - target_change_year for yr in year_cols]
        ax.set_xticks(event_times)
        ax.set_xticklabels(
            [f"{et:+d}\n({target_change_year + et})" for et in event_times],
            fontsize=7,
        )

        # Count how many changed buildings fed this chart
        n_chg_buildings = (
            ev_df[ev_df["group"] == "changed"]
            .groupby("event_time")["value"]
            .count()
            .max()
        )
        n_chg_buildings = int(n_chg_buildings) if not pd.isna(n_chg_buildings) else 0

        ax.set_xlabel("Event time (years relative to change_year)")
        ax.set_ylabel("predicted_value")
        ax.set_title(
            f"Event study: change_year cohort = {target_change_year}  "
            f"({split_type} tracts)\n"
            f"n_changed ≈ {n_chg_buildings}  |  "
            f"change_year ∉ {{2010, 2012, 2022, 2024}}"
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        _savefig(fig, out_path)

    for split_type in ("test", "train"):
        for cy_target in PER_YEAR_CHANGE_YEARS:
            ev_df_yr = _build_event_df_single_year(
                bldg_wide, split_type, cy_target
            )
            fname = f"B_event_study_{split_type}_cy{cy_target}.png"
            _plot_event_study_single_year(
                ev_df_yr, split_type, cy_target,
                out / "figures" / fname,
            )
            
    # ── pooled event study, stratified by construction intensity ──
    # All valid change years (PER_YEAR_CHANGE_YEARS) are pooled together so
    # event_time = yr - change_year, matching the existing _build_event_df
    # convention.  One chart is produced per split × intensity threshold,
    # giving a high-power view of whether the model response scales with the
    # intensity of physical change.
    #
    # n_new_buildings == 1  → isolated infill
    # n_new_buildings >= 2  → multiple new footprints (coordinated infill)
    # n_new_buildings >= 5  → dense redevelopment wave
    INTENSITY_THRESHOLDS = [
        ("any",   1, "any new construction (n_new ≥ 1)"),
        ("multi", 2, "multiple new buildings (n_new ≥ 2)"),
        ("dense", 5, "dense redevelopment (n_new ≥ 5)"),
    ]

    def _build_event_df_intensity_pooled(
        bldg_wide: pd.DataFrame,
        split_type: str,
        min_n_new: int,
        n_stable_sample: int = 5_000,
    ) -> pd.DataFrame:
        """
        Build a pooled event-study DataFrame for one split × intensity cell.

        Changed group  – all changed buildings in `split_type` tracts whose
                         change_year is in PER_YEAR_CHANGE_YEARS and whose tile
                         contains at least `min_n_new` new constructions.
                         event_time = yr - change_year (each building anchors at
                         its own change_year, same as _build_event_df).
        Stable group   – up to `n_stable_sample` stable buildings from the same
                         split, anchored at 2016 (the calendar midpoint used by
                         the pooled baseline).
        """
        type_mask = (bldg_wide["type"] == split_type).fillna(False)
        chg_mask  = bldg_wide["changed"].fillna(False).astype(bool)
        stb_mask  = ~chg_mask

        if "n_new_buildings" in bldg_wide.columns:
            intensity_mask = bldg_wide["n_new_buildings"].fillna(0).astype(int) >= min_n_new
        else:
            intensity_mask = pd.Series(True, index=bldg_wide.index)

        rows = []

        # ── changed: all valid change years, intensity-filtered ──
        chg_pool = bldg_wide[type_mask & chg_mask & intensity_mask]
        for _, row in chg_pool.iterrows():
            cy = row["change_year"]
            if pd.isna(cy) or int(cy) not in PER_YEAR_CHANGE_YEARS:
                continue
            for yr in year_cols:
                v = row[yr]
                if pd.isna(v):
                    continue
                rows.append({
                    "event_time":        yr - int(cy),
                    "value":             float(v),
                    "group":             "changed",
                    "construction_year": int(cy),
                })

        # ── stable reference: anchored at 2016 ──
        stb_pool   = bldg_wide[type_mask & stb_mask]
        n_sample   = min(n_stable_sample, len(stb_pool))
        stb_sample = stb_pool.sample(n_sample, random_state=2)
        for _, row in stb_sample.iterrows():
            for yr in year_cols:
                v = row[yr]
                if pd.isna(v):
                    continue
                rows.append({
                    "event_time":        yr - 2016,
                    "value":             float(v),
                    "group":             "stable",
                    "construction_year": None,
                })

        return pd.DataFrame(rows)

    def _plot_event_study_intensity_pooled(
        ev_df: pd.DataFrame,
        split_type: str,
        intensity_label: str,
        min_n_new: int,
        out_path: Path,
    ) -> None:
        """
        Plot a pooled event-study chart for one split × intensity cell.
        The x-axis shows event time only (no calendar year annotation, since
        buildings from different cohorts are pooled).  A per-construction-year
        breakdown is overlaid as thin coloured lines so cohort heterogeneity
        is visible without cluttering the headline comparison.
        """
        fig, ax = plt.subplots(figsize=(9, 5))

        # ── headline changed vs stable ──
        for grp, color, label in [
            ("changed", "firebrick", "Changed buildings (pooled)"),
            ("stable",  "steelblue", "Stable (reference)"),
        ]:
            sub = ev_df[ev_df["group"] == grp]
            if sub.empty:
                continue
            times = sorted(sub["event_time"].unique())
            means, los, his, valid_times = [], [], [], []
            for t in times:
                vals = sub.loc[sub["event_time"] == t, "value"].dropna().values
                if len(vals) < 5:
                    continue
                boot = [
                    vals[ev_rng.integers(0, len(vals), len(vals))].mean()
                    for _ in range(500)
                ]
                means.append(float(vals.mean()))
                los.append(float(np.percentile(boot, 2.5)))
                his.append(float(np.percentile(boot, 97.5)))
                valid_times.append(t)
            if valid_times:
                ax.plot(valid_times, means, "o-", color=color,
                        lw=2.5, ms=6, label=label, zorder=3)
                ax.fill_between(valid_times, los, his,
                                alpha=0.15, color=color, zorder=2)

        # ── per-construction-year overlay (thin lines, changed only) ──
        chg_sub = ev_df[ev_df["group"] == "changed"]
        cy_palette = plt.cm.tab10.colors
        for ci, cy in enumerate(sorted(chg_sub["construction_year"].dropna().unique())):
            cy_rows = chg_sub[chg_sub["construction_year"] == cy]
            times = sorted(cy_rows["event_time"].unique())
            means_cy, valid_t_cy = [], []
            for t in times:
                vals = cy_rows.loc[cy_rows["event_time"] == t, "value"].dropna().values
                if len(vals) < 3:
                    continue
                means_cy.append(float(vals.mean()))
                valid_t_cy.append(t)
            if valid_t_cy:
                color_cy = cy_palette[ci % len(cy_palette)]
                ax.plot(valid_t_cy, means_cy, "s--", color=color_cy,
                        lw=1.0, ms=3, alpha=0.55, label=f"cy={int(cy)}", zorder=1)

        ax.axvline(0, color="black", ls="--", lw=1, label="t = 0 (change_year)")

        n_chg_buildings = (
            ev_df[ev_df["group"] == "changed"]
            .groupby("event_time")["value"]
            .count()
            .max()
        )
        n_chg_buildings = int(n_chg_buildings) if not pd.isna(n_chg_buildings) else 0

        ax.set_xlabel("Event time (years relative to change_year)")
        ax.set_ylabel("predicted_value")
        ax.set_title(
            f"Event study (pooled years): {intensity_label}\n"
            f"({split_type} tracts  |  change_years {PER_YEAR_CHANGE_YEARS}"
            f"  |  n_changed ≈ {n_chg_buildings})"
        )
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        _savefig(fig, out_path)

    for split_type in ("test", "train"):
        for thresh_name, min_n_new, thresh_label in INTENSITY_THRESHOLDS:
            ev_df_int = _build_event_df_intensity_pooled(
                bldg_wide, split_type, min_n_new
            )
            n_chg = (ev_df_int["group"] == "changed").sum()
            if n_chg < 10:
                print(
                    f"    skipping {split_type} {thresh_name} (pooled): "
                    f"only {n_chg} changed observations"
                )
                continue
            fname = f"B_event_study_{split_type}_intensity_{thresh_name}_pooled.png"
            _plot_event_study_intensity_pooled(
                ev_df=ev_df_int,
                split_type=split_type,
                intensity_label=thresh_label,
                min_n_new=min_n_new,
                out_path=out / "figures" / fname,
            )

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

    fig, ax = plt.subplots(figsize=(8, 5))
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
    _savefig(fig, out / "figures" / "C_city_avg_trajectory.png")

    # Long-difference scatter: ACS 2009→2024 vs mapped prediction 2010→2024
    panel = pd.read_feather(processed_dir / "ny_tracts_panel_2009_2014_2019_2024.feather")
    panel["GEOID_str"] = panel["geoid_2024"].astype(str).str.zfill(11)
    panel["acs_change"] = (
        panel["per_capita_income_usd_2024"] - panel["per_capita_income_usd_2009"]
    )

    for map_col in ["qmap20", "qmaplogn"]:
        wide_map = qmap_long.pivot_table(
            index="GEOID_str", columns="year", values=map_col, aggfunc="first"
        )
        wide_map.columns = [int(c) for c in wide_map.columns]
        if 2010 not in wide_map.columns or 2024 not in wide_map.columns:
            continue
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

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x_c, y_c, s=5, alpha=0.3, color="steelblue", linewidths=0)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        xlim = np.array([x_c.min(), x_c.max()])
        ax.plot(xlim, np.polyval(np.polyfit(x_c, y_c, 1), xlim),
                "r-", lw=1.5, label=f"slope={slope_c:.3f}")
        ax.set_xlabel("ACS per-capita income change 2009→2024 (USD)")
        ax.set_ylabel(f"{map_col} change 2010→2024 (USD)")
        ax.set_title(
            f"Long-difference scatter ({map_col})  —  tracts predicted in both 2010 & 2024\n"
            f"Spearman ρ = {rho_c:.3f}  n = {len(merged)}"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        _savefig(fig, out / "figures" / f"C_long_diff_{map_col}.png")

    return qmap_long


# ─── Part D ───────────────────────────────────────────────────────────────────

def _lonlat_to_6539(lon: float, lat: float) -> tuple[float, float]:
    t = Transformer.from_crs(CRS_GEO, CRS_PROJ, always_xy=True)
    return t.transform(lon, lat)


def _slice_zarr(ds, cx: float, cy: float, half: float,
                max_px: int = 1400) -> np.ndarray | None:
    """Extract a square tile (4, H, W) uint8 from a zarr Dataset, strided so the
    longest side is ~max_px pixels (plan §D.1: 'downsampled for size'). Striding
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
    """Convert (4, H, W) uint8 → (H, W, 3) uint8 with percentile stretch."""
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

    patch_old = mpatches.Patch(color="0.65",    label="Pre-existing (built ≤2009)")
    patch_new = mpatches.Patch(color="crimson", label="New construction (2010–year)")
    fig_g.legend(handles=[patch_old, patch_new], loc="lower left",
                 fontsize=7, framealpha=0.85, ncol=2)

    fig_g.suptitle("Hudson Yards: 2010–2024 aerial imagery, buildings, predictions",
                   fontsize=11, fontweight="bold")
    fig_g.tight_layout(rect=[0, 0.03, 1, 0.97])
    _savefig(fig_g, out / "figures" / "D_hudson_yards_grid.png", dpi=300)

    # ── D.2 Per-building trajectories ─────────────────────────────────────────
    print("  D.2 per-building line chart...")
    traj: dict[int, dict[int, float]] = {}
    for yr, df in pred_by_yr.items():
        for did, row in df.iterrows():
            traj.setdefault(did, {})[yr] = float(row["predicted_value"])

    fig_l, ax_l = plt.subplots(figsize=(10, 5))
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
    p_old = mpatches.Patch(color="steelblue", alpha=0.7, label="Pre-existing (built ≤2009)")
    p_new = mpatches.Patch(color="crimson",   alpha=0.7, label="New construction (built >2009)")
    ax_l.legend(handles=[p_old, p_new], fontsize=8)
    ax_l.set_xlabel("Year")
    ax_l.set_ylabel("predicted_value")
    ax_l.set_title("Hudson Yards: per-building prediction trajectories")
    ax_l.set_xticks(YEARS)
    fig_l.tight_layout()
    _savefig(fig_l, out / "figures" / "D_buildings_lines.png")

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

    fig_t, ax_t = plt.subplots(figsize=(9, 5))
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
    _savefig(fig_t, out / "figures" / "D_tract_acs_vs_pred.png")


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
    print(f"  Tables  → {out_dir / 'tables'}")
    print(f"  Figures → {out_dir / 'figures'}")


if __name__ == "__main__":
    main()