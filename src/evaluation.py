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
from scipy import special as _sp
from scipy.stats import spearmanr, kendalltau, rv_continuous
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
    "IEETRAN": {"onecol": 252.*pt, "twocol": 526.3*pt}, # CQG is only one column
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
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, out / "figures" / "A_scatter_2016.pdf")
    _map_acs_vs_pred(results_dir, processed_dir, out)


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

SPLIT_GROUPS = {
    "test":  ["test", "val_spatial_temporal", "val_spatial", "dead_zone"],
    "train": ["train"],
}


def part_d(results_dir: Path, processed_dir: Path, out: Path) -> None:
    print("\n=== Part D: Event-Study (Tract-Level CSA DiD) ===")

    import csa
    import polars as pl

    splits = _load_splits(processed_dir)

    print("  D.1 Tract panel...")
    tract_long = _load_tract_long(results_dir)
    tract_long = tract_long.merge(splits[["GEOID_str", "type"]], on="GEOID_str", how="left")

    print("  D.2 Building footprints (loaded once)...")
    bldg_nyc = gpd.read_parquet(processed_dir / "buildings_nyc.parquet").to_crs(CRS_PROJ)
    bldg_nyc["area"] = bldg_nyc.geometry.area
    bldg_centroids = bldg_nyc.copy()
    bldg_centroids.geometry = bldg_centroids.geometry.centroid

    splits_geom = gpd.read_feather(processed_dir / "tract_splits.feather").to_crs(CRS_PROJ)
    splits_geom["GEOID_str"] = splits_geom["GEOID"].astype(str).str.zfill(11)

    def get_change_year(group, threshold):
        treated = group[group["change_pct"] > threshold]
        return treated["year"].min() if len(treated) > 0 else 0

    thresholds = [0.01, 0.05, 0.1]
    n_rows, n_cols = len(SPLIT_GROUPS), len(thresholds)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(journal_sizes["IEETRAN"]["twocol"] * 1.5, journal_sizes["IEETRAN"]["twocol"]),
        sharey=False,
    )

    legend_handles = None

    for row_i, (split_name, split_types) in enumerate(SPLIT_GROUPS.items()):
        print(f"\n  --- Split group: {split_name} ---")

        group_tract_long = tract_long[tract_long["type"].isin(split_types)].copy()
        group_tracts = splits_geom[splits_geom["type"].isin(split_types)][["GEOID_str", "geometry"]]

        print(f"  D.2 Tract-level change detection (Area-based) [{split_name}]...")
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

        print(f"  D.3 CSA DiD Estimation [{split_name}]...")
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
        print("  No buildings found. Skipping Part E.")
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

    # ── E.1 8×3 grid ─────────────────────────────────────────────────────────
    print("  E.1 8×3 image grid...")
    n_rows = len(YEARS)
    fw = FIG_SIZE_TWO_COL[0]
    panel_h = fw / 3 * 0.90          # slightly sub-square panels
    bottom_pad = 0.055                # figure fraction reserved for cbar + legend

    fig_g, axes_g = plt.subplots(
        n_rows, 3,
        figsize=(fw, n_rows * panel_h),
        gridspec_kw={"hspace": 0.015, "wspace": 0.015},
    )

    for col_i, col_title in enumerate(
        ["Aerial image (RGB)", "Building polygons", "Model predictions"]
    ):
        axes_g[0, col_i].set_title(col_title, fontsize=7, fontweight="bold", pad=3)

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

    _savefig(fig_g, out / "figures" / "D_hudson_yards_grid.pdf")

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
    ax_l.set_xticks(YEARS)
    fig_l.tight_layout()
    _savefig(fig_l, out / "figures" / "D_buildings_lines.pdf")

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
    ax_t.set_xticks(YEARS)
    fig_t.tight_layout()
    _savefig(fig_t, out / "figures" / "D_tract_acs_vs_pred.pdf")



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
        part_d(results_dir, PROCESSED_DATA_DIR, out_dir)

    if "E" in parts:
        part_e(results_dir, PROCESSED_DATA_DIR, out_dir, qmap_long=qmap_long)

    print("\n=== Done ===")
    print(f"  Tables  ->{out_dir / 'tables'}")
    print(f"  Figures ->{out_dir / 'figures'}")


if __name__ == "__main__":
    main()