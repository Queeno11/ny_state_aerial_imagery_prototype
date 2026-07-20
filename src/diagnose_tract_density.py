"""How sparse are the test tracts, and what would a per-tract building cap save?

For every tract in a split (default: test) this script computes
  (1) n_buildings   — Microsoft-footprint buildings in the tract
                      (pair_buildings_ms_us parquet, the training universe),
  (2) bldg_density  — buildings per km^2 (tract polygons are EPSG:5070,
                      an equal-area CRS, so .area is exact m^2),
  (3) pop_density   — ACS total population per km^2 (us_metros_panel),
then prints summary statistics aimed at two download-budget questions:
  * are there very disperse (low-density / huge-area) tracts we could drop
    with little loss of signal, and
  * how many NAIP fetches would a cap of x buildings per tract save
    (the LazyPairTable already supports n_buildings_per_tract sampling).

Read-only diagnostic: prints to stdout, optionally dumps the per-tract table
with --out. Run it as
    python src/diagnose_tract_density.py [--split test] [--pop-year 2023]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.utils.paths import PROCESSED_DATA_DIR

BUILDINGS_PATH = PROCESSED_DATA_DIR / "pair_buildings_ms_us_epsg5070_all.parquet"
TRACT_SPLITS_PATH = PROCESSED_DATA_DIR / "tract_splits.feather"
PANEL_PATH = PROCESSED_DATA_DIR / "us_metros_panel_2011_2023.feather"

PERCENTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
DEFAULT_CAPS = (25, 50, 100, 200, 500, 1000)
# "disperse tract" screens: buildings per km^2 below each threshold
DENSITY_THRESHOLDS = (1.0, 5.0, 10.0, 25.0, 50.0)


# ── pure pieces (unit-tested on synthetic data) ──────────────────────────────

def count_buildings_per_tract(geoids: pd.Series) -> pd.Series:
    """Buildings per tract GEOID from the per-building GEOID column.

    Returns a Series indexed by GEOID (plain str) named ``n_buildings``.
    Categorical inputs only count categories actually present (observed).
    """
    counts = geoids.value_counts()
    counts = counts[counts > 0]  # drop unused categorical levels
    counts.index = counts.index.astype(str)
    return counts.rename("n_buildings").rename_axis("GEOID")


def build_tract_table(tracts: pd.DataFrame, building_counts: pd.Series,
                      population: pd.Series) -> pd.DataFrame:
    """One row per tract: counts, area, and the two densities.

    Parameters
    ----------
    tracts : GeoDataFrame-like with ``GEOID`` (str) and ``geometry`` in an
        equal-area metric CRS (EPSG:5070 in production) — area comes from
        ``geometry.area``. Every tract in this frame is kept (left join), so
        tracts with zero footprint buildings surface with n_buildings = 0.
    building_counts : output of :func:`count_buildings_per_tract`.
    population : Series of total population indexed by GEOID str; tracts
        missing from it get NaN population / pop_density.
    """
    out = tracts[["GEOID"]].copy()
    out["GEOID"] = out["GEOID"].astype(str)
    out["area_km2"] = tracts.geometry.area.to_numpy() / 1e6

    out = out.merge(building_counts, on="GEOID", how="left")
    out["n_buildings"] = out["n_buildings"].fillna(0).astype("int64")

    pop = population.copy()
    pop.index = pop.index.astype(str)
    out["population"] = out["GEOID"].map(pop)

    out["bldg_density"] = out["n_buildings"] / out["area_km2"]
    out["pop_density"] = out["population"] / out["area_km2"]
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile table for the four diagnostic columns."""
    cols = ["n_buildings", "area_km2", "bldg_density", "pop_density"]
    return df[cols].describe(percentiles=PERCENTILES)


def cap_savings(counts: np.ndarray | pd.Series,
                caps=DEFAULT_CAPS) -> pd.DataFrame:
    """Fetch budget under a per-tract cap of x buildings.

    For each cap: total building fetches with min(n, cap), share of the
    uncapped total saved, and how many tracts the cap binds on.
    """
    n = np.asarray(counts, dtype="int64")
    total = int(n.sum())
    rows = []
    for cap in caps:
        capped = np.minimum(n, cap).sum()
        rows.append({
            "cap": cap,
            "fetches": int(capped),
            "pct_of_uncapped": 100.0 * capped / total if total else np.nan,
            "pct_saved": 100.0 * (total - capped) / total if total else np.nan,
            "tracts_binding": int((n > cap).sum()),
        })
    return pd.DataFrame(rows).set_index("cap")


def population_bin_report(population: pd.Series) -> pd.DataFrame:
    """Tract counts in the near-empty population range.

    Bins: exactly 0, then (0, 10], (10, 20], (20, 30], (30, 40], (40, 50],
    plus a ``> 50`` remainder and a NaN row so every tract is accounted for.
    Flags the unpopulated special-use tracts (water/employment 98xx/99xx
    GEOIDs) that carry no ACS income signal.
    """
    pop = population
    rows = [("== 0", int((pop == 0).sum()))]
    for lo in range(0, 50, 10):
        hi = lo + 10
        rows.append((f"({lo}, {hi}]", int(((pop > lo) & (pop <= hi)).sum())))
    rows.append(("> 50", int((pop > 50).sum())))
    rows.append(("NaN", int(pop.isna().sum())))
    out = pd.DataFrame(rows, columns=["population", "tracts"]).set_index("population")
    out["pct_tracts"] = 100.0 * out["tracts"] / len(pop) if len(pop) else np.nan
    return out


def sparse_tract_report(df: pd.DataFrame,
                        thresholds=DENSITY_THRESHOLDS) -> pd.DataFrame:
    """What dropping tracts below each building-density threshold costs.

    Per threshold: tracts removed, share of tracts, share of total buildings
    (i.e. fetches saved), and share of total land area those tracts cover.
    """
    total_tracts = len(df)
    total_bldgs = df["n_buildings"].sum()
    total_area = df["area_km2"].sum()
    rows = []
    for thr in thresholds:
        sel = df["bldg_density"] < thr
        rows.append({
            "density_lt": thr,
            "tracts": int(sel.sum()),
            "pct_tracts": 100.0 * sel.mean() if total_tracts else np.nan,
            "pct_buildings": 100.0 * df.loc[sel, "n_buildings"].sum() / total_bldgs
                             if total_bldgs else np.nan,
            "pct_area": 100.0 * df.loc[sel, "area_km2"].sum() / total_area
                        if total_area else np.nan,
        })
    return pd.DataFrame(rows).set_index("density_lt")


# ── real-data loaders / driver ───────────────────────────────────────────────

def load_inputs(split: str, pop_year: int):
    """(tracts GeoDataFrame for the split, per-tract building counts,
    population Series) from the processed data files."""
    import geopandas as gpd
    import pyarrow.parquet as pq

    tracts = gpd.read_feather(TRACT_SPLITS_PATH)
    tracts = tracts[tracts["type"] == split].reset_index(drop=True)
    if tracts.empty:
        raise ValueError(f"No tracts with type == {split!r} in {TRACT_SPLITS_PATH}")

    # 71.8M rows but a single dictionary-encoded column — cheap to load.
    geoids = pq.read_table(BUILDINGS_PATH, columns=["GEOID"])["GEOID"].to_pandas()
    counts = count_buildings_per_tract(geoids)

    panel = pd.read_feather(
        PANEL_PATH, columns=[f"geoid_{pop_year}", f"total_population_{pop_year}"])
    population = (panel.dropna(subset=[f"geoid_{pop_year}"])
                       .set_index(f"geoid_{pop_year}")[f"total_population_{pop_year}"])
    return tracts, counts, population


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--pop-year", type=int, default=2023)
    ap.add_argument("--caps", type=int, nargs="+", default=list(DEFAULT_CAPS))
    ap.add_argument("--out", default=None,
                    help="optional CSV path for the per-tract table")
    args = ap.parse_args()

    tracts, counts, population = load_inputs(args.split, args.pop_year)
    df = build_tract_table(tracts, counts, population)

    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    print(f"\n=== {args.split} split: {len(df):,} tracts, "
          f"{df['n_buildings'].sum():,} buildings ===")
    print(f"tracts with 0 buildings: {(df['n_buildings'] == 0).sum():,}   "
          f"missing ACS population: {df['population'].isna().sum():,}")

    print("\n--- Summary statistics (per tract) ---")
    print(summarize(df).to_string())

    print("\n--- Near-empty tracts by ACS total population ---")
    print(population_bin_report(df["population"]).to_string())

    print("\n--- Disperse tracts: cost of dropping density < threshold "
          "(buildings/km^2) ---")
    print(sparse_tract_report(df).to_string())

    print("\n--- Per-tract building cap: fetch savings ---")
    print(cap_savings(df["n_buildings"], caps=args.caps).to_string())

    print("\n--- 15 most disperse tracts by building density ---")
    worst = df.nsmallest(15, "bldg_density")
    print(worst[["GEOID", "n_buildings", "area_km2",
                 "bldg_density", "pop_density"]].to_string(index=False))

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nPer-tract table written to {args.out}")


if __name__ == "__main__":
    main()
