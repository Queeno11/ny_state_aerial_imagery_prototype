"""
export_building_changes.py
--------------------------
Extracts new buildings (DOITT_ID + CONSTRUCTION_YEAR) from buildings_nyc.parquet
and exports a lean CSV for tracking where development occurred (2010–2024).

Usage:
    python export_building_changes.py
    python export_building_changes.py --processed_dir /path/to/processed_data
    python export_building_changes.py --out building_changes.csv
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def export_building_changes(
    processed_dir: Path,
    out_path: Path,
) -> pd.DataFrame:
    print(f"Loading buildings_nyc.parquet from {processed_dir} ...")
    bldg_nyc = gpd.read_parquet(processed_dir / "buildings_nyc.parquet")

    # Mirror the exact filter used in evaluation.py Part B:
    # new buildings constructed strictly after 2009 and no later than 2024.
    new_bldg = bldg_nyc.loc[
        bldg_nyc["CONSTRUCTION_YEAR"].between(2009, 2024, inclusive="right")
    ].copy()

    # Keep only DOITT_ID (index) and CONSTRUCTION_YEAR — drop geometry and all
    # other columns to produce a small, flat tracking dataset.
    df = (
        new_bldg[["CONSTRUCTION_YEAR", "geometry"]]
        .reset_index()                          # brings DOITT_ID from index to column
        .rename(columns={"index": "DOITT_ID"})  # make the column name explicit
        [["DOITT_ID", "CONSTRUCTION_YEAR", "geometry"]]     # enforce column order
        .sort_values("CONSTRUCTION_YEAR")
        .reset_index(drop=True)
    )

    # Coerce to clean int types where possible
    df["DOITT_ID"] = df["DOITT_ID"].astype(int)
    df["CONSTRUCTION_YEAR"] = df["CONSTRUCTION_YEAR"].astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"Exported {len(df):,} new buildings → {out_path}")
    print(f"  Year range : {df['CONSTRUCTION_YEAR'].min()} – {df['CONSTRUCTION_YEAR'].max()}")
    print(f"  Rows/year  :")
    for yr, cnt in df["CONSTRUCTION_YEAR"].value_counts().sort_index().items():
        print(f"    {yr}: {cnt:,}")

    return df


def main() -> None:
    # Try to import project paths; fall back to a sensible default so the
    # script also works when run outside the src package.
    try:
        from src.utils.paths import PROCESSED_DATA_DIR
        default_processed = str(PROCESSED_DATA_DIR)
    except ImportError:
        default_processed = "data/processed"

    parser = argparse.ArgumentParser(
        description="Export new-building tracking dataset (DOITT_ID + CONSTRUCTION_YEAR)."
    )
    parser.add_argument(
        "--processed_dir",
        default=default_processed,
        help="Directory containing buildings_nyc.parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default="building_changes.parquet",
        help="Output Parquet path (default: %(default)s)",
    )
    args = parser.parse_args()

    export_building_changes(
        processed_dir=Path(args.processed_dir),
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
