"""Detect + repair sampler_validation crops corrupted by sliver-tile selection.

Companion to ``sampler_validation.ipynb``. Before the fix in
``src/data/naip_fetcher.py``, item choice ignored footprint coverage, so a tract
centroid near a DOQQ edge could be served by a tile covering only a sliver of
the 300m window; rasterio clipped the window and stretched the sliver to a
square (the smeared columns in the pairs_*.png grids).

This script replicates the notebook's sampling (SEED=825), asks for each cached
tract whether the OLD fetcher (year-distance sort only) would have picked a
non-containing STAC item for either year, deletes those .npz entries, refetches
them with the fixed fetcher, and regenerates the per-bin PNG grids.

Run once after the fetcher fix; safe to re-run (cached pairs are kept).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.paths import PROCESSED_DATA_DIR
from src.data import indicators
from src.data.naip_fetcher import (
    fetch_naip, get_catalog, get_fetch_stats, item_contains_bbox, lonlat_bbox,
)

N_PER_BIN, SEED = 20, 825
INDICATOR = indicators.DEFAULT_INDICATOR
YEAR_EARLY, YEAR_LATE = 2013, 2024
CROP_METERS, OUT_PIXELS = 300, 256
CACHE_DIR = Path.home() / "outputs" / "sampler_validation"


def load_sampled():
    """Rebuild the notebook's stratified sample (same seed → same tracts)."""
    panel = gpd.read_feather(PROCESSED_DATA_DIR / "us_metros_panel_2011_2023.feather")
    geoid_col = sorted(c for c in panel.columns if c.startswith("geoid_"))[-1]
    var = indicators.token_to_var(INDICATOR)
    pval_candidates = ([c for c in panel.columns if c.startswith(f"pvalue_{var}_")]
                       if var else [c for c in panel.columns if c.startswith("pvalue_2")])
    pval_col = pval_candidates[0]
    bin_edges = [0, 0.01, 0.05, 0.10, 0.25, 1.0]
    bin_labels = ["p<0.01", "0.01-0.05", "0.05-0.10", "0.10-0.25", "p>=0.25"]
    tracts = panel[[geoid_col, "cbsa_code", pval_col, "geometry"]].dropna(subset=[pval_col]).copy()
    tracts["p_bin"] = pd.cut(tracts[pval_col], bins=bin_edges, labels=bin_labels, include_lowest=True)
    sampled = (tracts.groupby("p_bin", observed=True, group_keys=False)
               .apply(lambda g: g.sample(min(N_PER_BIN, len(g)), random_state=SEED)))
    cent = sampled.geometry.centroid.to_crs("EPSG:4326")
    sampled["lon"], sampled["lat"] = cent.x, cent.y
    return sampled, geoid_col, pval_col, bin_labels


def detect_corrupted(sampled, geoid_col):
    """Cached tracts where the old year-only sort picked a non-containing tile."""
    corrupted = []
    for _, row in sampled.iterrows():
        geoid, lon, lat = row[geoid_col], row["lon"], row["lat"]
        if not (CACHE_DIR / f"{geoid}.npz").exists():
            continue
        bbox = lonlat_bbox(lon, lat, CROP_METERS)
        try:
            items = list(get_catalog().search(collections=["naip"], bbox=bbox,
                                              max_items=50).items())
        except Exception as e:
            print(f"{geoid}: search failed ({e}); skipping detection")
            continue
        if not items:
            continue
        bad_years = []
        for hint in (YEAR_EARLY, YEAR_LATE):
            old_choice = min(items, key=lambda it: (abs(it.datetime.year - hint), it.id))
            if not item_contains_bbox(old_choice, bbox):
                bad_years.append((hint, old_choice.id))
        if bad_years:
            corrupted.append((geoid, row["p_bin"], bad_years))
    return corrupted


def fetch_pair(geoid, lon, lat):
    """Same disk-cached pair fetch as the notebook, with partial-coverage log."""
    cache = CACHE_DIR / f"{geoid}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["early"], d["late"], int(d["early_year"]), int(d["late_year"])
    r_early = fetch_naip(lon, lat, CROP_METERS, nbands=3, out_pixels=OUT_PIXELS, year_hint=YEAR_EARLY)
    r_late = fetch_naip(lon, lat, CROP_METERS, nbands=3, out_pixels=OUT_PIXELS, year_hint=YEAR_LATE)
    if r_early.crop is None or r_late.crop is None:
        return None
    for tag, r in (("early", r_early), ("late", r_late)):
        if r.partial_coverage:
            print(f"  {geoid} {tag}: no fully-covering item; boundless zero-fill used")
    np.savez_compressed(cache, early=r_early.crop, late=r_late.crop,
                        early_year=r_early.actual_year, late_year=r_late.actual_year)
    return r_early.crop, r_late.crop, r_early.actual_year, r_late.actual_year


def save_bin_grid(sampled, pairs, geoid_col, pval_col, bin_label, max_cols=10):
    """Mirror of the notebook's show_bin, minus plt.show()."""
    sub = sampled[sampled["p_bin"] == bin_label]
    sub = sub[sub[geoid_col].isin(pairs)]
    if sub.empty:
        print(f"{bin_label}: no fetched pairs")
        return
    n = min(len(sub), max_cols)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.8))
    axes = np.atleast_2d(axes)
    if n == 1:
        axes = axes.reshape(2, 1)
    for j, (_, row) in enumerate(sub.head(n).iterrows()):
        early, late, y0, y1 = pairs[row[geoid_col]]
        axes[0, j].imshow(np.moveaxis(early, 0, -1)); axes[0, j].set_title(f"{y0}", fontsize=8)
        axes[1, j].imshow(np.moveaxis(late, 0, -1)); axes[1, j].set_title(f"{y1}\np={row[pval_col]:.3f}", fontsize=8)
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{pval_col} bin: {bin_label} — top: ~{YEAR_EARLY}, bottom: ~{YEAR_LATE}")
    fig.tight_layout()
    safe = bin_label.replace("<", "lt").replace(">=", "ge")
    fig.savefig(CACHE_DIR / f"pairs_{safe}.png", dpi=120)
    plt.close(fig)


def main():
    sampled, geoid_col, pval_col, bin_labels = load_sampled()

    corrupted = detect_corrupted(sampled, geoid_col)
    print(f"\nCorrupted cache entries: {len(corrupted)} of {len(sampled)} sampled tracts")
    for geoid, pbin, bad in corrupted:
        print(f"  {geoid} [{pbin}]: " + "; ".join(f"~{y} old item {i}" for y, i in bad))

    for geoid, _, _ in corrupted:
        (CACHE_DIR / f"{geoid}.npz").unlink()

    pairs = {}
    for _, row in sampled.iterrows():
        res = fetch_pair(row[geoid_col], row["lon"], row["lat"])
        if res is not None:
            pairs[row[geoid_col]] = res
    print(f"\nPairs available: {len(pairs)}/{len(sampled)} | fetcher stats: {get_fetch_stats()}")

    for lbl in bin_labels:
        save_bin_grid(sampled, pairs, geoid_col, pval_col, lbl)
    print("Grids regenerated.")


if __name__ == "__main__":
    main()
