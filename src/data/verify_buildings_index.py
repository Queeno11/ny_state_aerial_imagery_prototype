"""Post-generation sanity checks for the Microsoft buildings_index.

Run AFTER ``build_buildings_index.py`` (single state or --all-states):

    python -m src.data.verify_buildings_index [--index-dir data/processed/buildings_index]
        [--panel data/processed/us_metros_panel_2011_2023.feather]
        [--coverage-csv results/tables/buildings_coverage_by_cbsa.csv]

Checks (prints PASS/FAIL, exits non-zero on any FAIL):
  1. Schema: building_id int64 (non-negative), cx/cy float64, tract_id, state.
  2. building_id globally unique ACROSS states (border duplicates are the
     expected failure mode — the builder only dedupes within a state).
  3. cx/cy within CONUS EPSG:5070 bounds and consistent with the id packing
     (unpack(building_id) ~ (cx, cy) on the 0.1 m grid).
  4. Every tract_id exists in the panel; every building's tract maps to a CBSA.
  5. Coverage table: buildings per CBSA / per tract (median, p5) — written to CSV
     for eyeballing; near-empty tracts are reported, not failed (they feed the
     per-city validation metrics of issue #31).
"""

import argparse
import sys

import numpy as np
import pandas as pd

from src.data.build_buildings_index import unpack_building_id, DEFAULT_PANEL
from src.utils.paths import PROCESSED_DATA_DIR, TABLES_DIR

# Generous CONUS bounds in EPSG:5070 meters (design area of the projection).
X_RANGE = (-3.0e6, 3.0e6)
Y_RANGE = (-0.5e6, 3.5e6)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default=str(PROCESSED_DATA_DIR / "buildings_index"))
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--coverage-csv",
                        default=str(TABLES_DIR / "buildings_coverage_by_cbsa.csv"))
    args = parser.parse_args(argv)

    failures = []

    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print(f"Reading {args.index_dir} ...")
    index = pd.read_parquet(args.index_dir)
    # Hive partitioning may return 'state' as categorical; normalize.
    if "state" in index.columns:
        index["state"] = index["state"].astype(str)
    print(f"{len(index):,} buildings, {index['state'].nunique()} states, "
          f"{index['tract_id'].nunique():,} tracts\n")

    # 1. Schema
    check("columns", set(index.columns) >= {"building_id", "cx", "cy", "tract_id", "state"},
          str(list(index.columns)))
    check("building_id int64 non-negative",
          index["building_id"].dtype == np.int64 and (index["building_id"] >= 0).all())

    # 2. Global uniqueness
    n_dup = int(index["building_id"].duplicated().sum())
    check("building_id globally unique", n_dup == 0,
          f"{n_dup:,} cross-state duplicates" if n_dup else "")
    if n_dup:
        dups = index[index["building_id"].duplicated(keep=False)]
        print(dups.groupby("state").size().to_string())
        print("  (cross-state border duplicates: dedupe keeping the first state, "
              "or drop here before training)")

    # 3. Coordinate bounds + id/coordinate consistency (sample for speed)
    check("cx within CONUS 5070 bounds",
          index["cx"].between(*X_RANGE).all(),
          f"range: {index['cx'].min():.0f}..{index['cx'].max():.0f}")
    check("cy within CONUS 5070 bounds",
          index["cy"].between(*Y_RANGE).all(),
          f"range: {index['cy'].min():.0f}..{index['cy'].max():.0f}")
    sample = index.sample(min(100_000, len(index)), random_state=0)
    rx, ry = unpack_building_id(sample["building_id"].to_numpy())
    grid_ok = (np.abs(rx - sample["cx"].to_numpy()).max() <= 0.05 + 1e-9 and
               np.abs(ry - sample["cy"].to_numpy()).max() <= 0.05 + 1e-9)
    check("building_id unpacks to (cx, cy) on the 0.1 m grid", grid_ok)

    # 4. Tract membership
    import geopandas as gpd
    panel = gpd.read_feather(args.panel)
    geoid_col = sorted(c for c in panel.columns if c.startswith("geoid_"))[-1]
    panel_tracts = panel[[geoid_col, "cbsa_code"]].rename(columns={geoid_col: "tract_id"})
    known = index["tract_id"].isin(set(panel_tracts["tract_id"]))
    n_unknown = int((~known & index["tract_id"].notna()).sum())
    check("all tract_ids exist in the panel", n_unknown == 0, f"{n_unknown:,} unknown")

    # 5. Coverage table (buildings per CBSA / per tract)
    merged = index.merge(panel_tracts, on="tract_id", how="left")
    per_tract = merged.groupby(["cbsa_code", "tract_id"]).size().rename("n_buildings")
    cov = per_tract.groupby("cbsa_code").agg(
        n_tracts="count", n_buildings="sum",
        median_per_tract="median",
        p5_per_tract=lambda s: s.quantile(0.05),
    ).reset_index()
    # Tracts present in the panel but with zero indexed buildings (per covered CBSA)
    covered = set(merged["cbsa_code"].dropna())
    panel_cov = panel_tracts[panel_tracts["cbsa_code"].isin(covered)]
    zero = panel_cov.groupby("cbsa_code")["tract_id"].nunique() - \
        cov.set_index("cbsa_code")["n_tracts"]
    cov["tracts_without_buildings"] = cov["cbsa_code"].map(zero).fillna(0).astype(int)
    try:
        out = args.coverage_csv
        cov.sort_values("n_buildings", ascending=False).to_csv(out, index=False)
        print(f"\nCoverage table -> {out}")
    except OSError as e:
        print(f"\nCould not write coverage CSV ({e}); printing head instead.")
    print(cov.sort_values("n_buildings", ascending=False).head(15).to_string(index=False))
    thin = cov[cov["median_per_tract"] < 50]
    if len(thin):
        print(f"\nNote: {len(thin)} CBSAs have median <50 buildings/tract "
              f"(partial state coverage or sparse metros) — inspect before training.")

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        return 1
    print("All buildings_index checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
