#!/usr/bin/env python3
# /mnt/c/Working Papers/NY State Aerial Imagery Prototype/ny_state_aerial_imagery_prototype/src/download_acs.py
"""
ACS 5-Year Estimates — New York State, Census Tract Level  (2009 → 2024)
=========================================================================
Loops over every ACS 5-year vintage from START_YEAR to END_YEAR and saves
one Feather file per vintage inside a user-defined folder tree:

  OUTPUT_ROOT/
  ├── 2009/   ← 2005-2009 estimates  →  ny_tracts_acs5_2009.feather
  ├── 2010/   ← 2006-2010 estimates  →  ny_tracts_acs5_2010.feather
  │   …
  └── 2024/   ← 2020-2024 estimates  →  ny_tracts_acs5_2024.feather

Variables collected
───────────────────
  DIRECT
    • Median Home Value           B25077_001E
    • Median Gross Rent           B25064_001E
    • Median Household Income     B19013_001E
    • Per Capita Income           B19301_001E

  DERIVED  (ratio / weighted-mean from raw counts)
    • % HH with No Vehicle        B08201
    • % Overcrowded Housing       B25014   (> 1.00 occ / room)
    • Mean Commute Time (min)     B08136 / (B08301 - WFH workers)
    • % Below Poverty Line        B17001
    • Education shares            B15003   (< HS / HS+GED / Some col / Bach+)
    • Weighted Mean Age           B01001

Resilience features
───────────────────
  • Skip / resume   — if a Feather file already exists for a year it is skipped
  • Per-year errors — a single bad year is logged and skipped; run continues
  • Variable guard  — missing columns produce NaN rather than crashing
  • Year-compat     — tables unavailable in a given vintage noted & handled
  • Retry logic     — transient HTTP errors are retried up to 3x with back-off
  • Per-variable probe — on HTTP 400, bad codes are isolated and set to NaN

Requirements: Python >= 3.9  |  pip install requests pandas numpy tqdm
Census API key (free): https://api.census.gov/data/key_signup.html
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from itertools import islice
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from pygris import tracts

# ──────────────────────────────────────────────────────────────────────────────
# optional progress bar (degrades gracefully if tqdm not installed)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm as _tqdm
    def progress(iterable, **kw):
        return _tqdm(iterable, **kw)
except ImportError:
    def progress(iterable, **kw):
        return iterable

# ══════════════════════════════════════════════════════════════════════════════
# ① USER CONFIGURATION  ←  all tuneable knobs live here
# ══════════════════════════════════════════════════════════════════════════════
API_KEY     = os.environ.get("CENSUS_API_KEY", "364fb9378fe05cc52fda7a590e1cf616cf18e9b6")
STATE       = "36"           # FIPS — 36 = New York
START_YEAR  = 2009           # first ACS 5-yr release
END_YEAR    = 2024           # inclusive; update as new vintages drop
MAX_VARS    = 45             # variables per API call (Census hard-cap ~50)
RETRY_MAX   = 1              # max retries on transient HTTP errors
RETRY_SLEEP = 1              # seconds between retries (doubles each attempt)
CALL_SLEEP  = 0.4            # polite pause between chunk calls

# Default output root — override with --outdir or OUTPUT_ROOT env var
DEFAULT_OUT_ROOT = os.environ.get("OUTPUT_ROOT", r"/mnt/e/Datasets/US ACS 5-year Census Tract Estimates")

# ══════════════════════════════════════════════════════════════════════════════
# ② VARIABLE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Age midpoints for B01001 weighted-average ─────────────────────────────
_AGE_MIDPOINTS = [
    2.5, 7.0, 12.0, 16.0, 18.5, 20.0, 21.0, 23.0,
    27.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0,
    60.5, 63.0, 65.5, 68.0, 72.0, 77.0, 82.0, 90.0,
]
AGE_MALE_VARS   = {f"B01001_{i+3:03d}E":  mp for i, mp in enumerate(_AGE_MIDPOINTS)}
AGE_FEMALE_VARS = {f"B01001_{i+27:03d}E": mp for i, mp in enumerate(_AGE_MIDPOINTS)}

# ── Education bands (B15003, population 25+) ─────────────────────────────
EDU_LT_HS  = [f"B15003_{i:03d}E" for i in range(2, 17)]     # no schooling -> 12th no diploma
EDU_HS_GED = ["B15003_017E", "B15003_018E"]                   # HS diploma, GED
EDU_SOME   = ["B15003_019E", "B15003_020E", "B15003_021E"]    # some college, associate's
EDU_BACH   = ["B15003_022E", "B15003_023E",
              "B15003_024E", "B15003_025E"]                    # bach, master's, prof., doctorate

# ── Master variable manifest  (code -> output label; None = calc only) ────
RAW_VARS: dict[str, Optional[str]] = {
    # Housing
    "B25077_001E": "median_home_value_usd",
    "B25064_001E": "median_gross_rent_usd",
    # Income
    "B19013_001E": "median_hh_income_usd",
    "B19301_001E": "per_capita_income_usd",
    # Vehicle availability (B08201)
    "B08201_001E": None,   # total households
    "B08201_002E": None,   # no vehicle
    # Overcrowding (B25014)
    "B25014_001E": None,   # total occupied units
    "B25014_005E": None,   # owner: 1.01-1.50 occ/room
    "B25014_006E": None,   # owner: 1.51-2.00
    "B25014_007E": None,   # owner: 2.01+
    "B25014_011E": None,   # renter: 1.01-1.50
    "B25014_012E": None,   # renter: 1.51-2.00
    "B25014_013E": None,   # renter: 2.01+
    # Commute (B08136 / B08301)
    "B08135_001E": None,   # aggregate travel time (min), excl. WFH
    "B08301_001E": None,   # total workers 16+
    "B08301_021E": None,   # worked from home (available 2009+)
    # Poverty (B17001)
    "B17001_001E": None,   # total w/ poverty status determined
    "B17001_002E": None,   # below poverty level
    # Education (B15003)
    "B15003_001E": None,   # total pop 25+ (denominator)
    **{v: None for v in EDU_LT_HS + EDU_HS_GED + EDU_SOME + EDU_BACH},
    # Age (B01001)
    "B01001_001E": None,   # total population (denominator)
    **{code: None for code in AGE_MALE_VARS},
    **{code: None for code in AGE_FEMALE_VARS},
}
 
# Final column selection/rename map (raw or derived -> output name)
FINAL_COLS: dict[str, str] = {
    "GEOID":               "geoid",
    "NAME":                "name",
    "acs_year":            "acs_year",
    "acs_span":            "acs_span",
    # Direct
    "B25077_001E":         "median_home_value_usd",
    "B25064_001E":         "median_gross_rent_usd",
    "B19013_001E":         "median_hh_income_usd",
    "B19301_001E":         "per_capita_income_usd",
    "B08135_001E":         "mean_commute_unadjusted_min",
    # Derived
    "pct_no_vehicle":      "pct_hh_no_vehicle",
    "pct_overcrowded":     "pct_overcrowded_housing",
    "mean_commute_min":    "mean_commute_min",
    "pct_below_poverty":   "pct_below_poverty",
    "pct_edu_lt_hs":       "pct_edu_lt_hs",
    "pct_edu_hs_ged":      "pct_edu_hs_ged",
    "pct_edu_some_col":    "pct_edu_some_college",
    "pct_edu_bach_plus":   "pct_edu_bach_plus",
    "mean_age":            "mean_age_years",
}
 

# ══════════════════════════════════════════════════════════════════════════════
# ③ API HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def chunked(iterable, size: int):
    it = iter(iterable)
    while chunk := list(islice(it, size)):
        yield chunk


def fetch_chunk_with_retry(
    year: int,
    var_codes: list[str],
    base_url: str,
) -> pd.DataFrame:
    """Fetch one chunk with exponential-backoff retries on transient errors."""
    params = {
        "get": "NAME," + ",".join(var_codes),
        "for": "tract:*",
        "in":  f"state:{STATE} county:*",
        "key": API_KEY,
    }
    sleep = RETRY_SLEEP
    for attempt in range(1, RETRY_MAX + 1):
        try:
            resp = requests.get(base_url, params=params, timeout=90)
            if resp.status_code == 400:
                raise ValueError(
                    f"HTTP 400 for year {year} — variable(s) likely absent in this vintage."
                )
            resp.raise_for_status()
            data = resp.json()
            return pd.DataFrame(data[1:], columns=data[0])
        except (requests.RequestException, ValueError) as exc:
            if attempt == RETRY_MAX:
                raise
            print(f"    ! Attempt {attempt} failed: {exc}  — retrying in {sleep}s …")
            time.sleep(sleep)
            sleep *= 2


def _safe_fetch_chunk(
    year: int,
    chunk: list[str],
    base_url: str,
    chunk_idx: int,
    total_chunks: int,
) -> pd.DataFrame | None:
    """
    Fetch one chunk. On HTTP 400, probe each variable individually to find
    the culprit(s), drop them, fill with NaN, then fetch the rest normally.
    """
    print(f"    Chunk {chunk_idx:>2}/{total_chunks}  ({len(chunk)} vars) … ", end="", flush=True)
    try:
        df = fetch_chunk_with_retry(year, chunk, base_url)
        print("ok")
        return df
    except ValueError:
        print("400 — probing individually …")
        good: list[str] = []
        bad:  list[str] = []
        for code in chunk:
            try:
                fetch_chunk_with_retry(year, [code], base_url)
                good.append(code)
            except Exception:
                bad.append(code)
        if bad:
            print(f"    ! Codes absent in {year} (NaN): {bad}")
        if not good:
            return None
        df = fetch_chunk_with_retry(year, good, base_url)
        for code in bad:
            df[code] = np.nan
        print(f"    ok  ({len(good)} fetched, {len(bad)} set to NaN)")
        return df


def fetch_all_for_year(year: int) -> pd.DataFrame:
    """
    Pull every variable for *year* in MAX_VARS-sized chunks and merge into
    one wide DataFrame with numeric dtypes; Census sentinels become NaN.
    """
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    codes    = list(RAW_VARS.keys())
    chunks   = list(chunked(codes, MAX_VARS))
    frames:  list[pd.DataFrame] = []

    for idx, chunk in enumerate(chunks, 1):
        df = _safe_fetch_chunk(year, chunk, base_url, idx, len(chunks))
        if df is None:
            continue

        df["GEOID"] = (
            df["state"].str.zfill(2)
            + df["county"].str.zfill(3)
            + df["tract"].str.zfill(6)
        )
        keep = (["GEOID", "NAME"] if idx == 1 else ["GEOID"]) + [
            c for c in chunk if c in df.columns
        ]
        frames.append(df[keep])
        time.sleep(CALL_SLEEP)

    if not frames:
        raise RuntimeError(f"No data retrieved for {year}")

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="GEOID", how="outer")

    # Guarantee every expected code column exists (may have been fully absent)
    for code in codes:
        if code not in merged.columns:
            merged[code] = np.nan

    # Numeric coerce + sentinel removal
    for code in codes:
        merged[code] = pd.to_numeric(merged[code], errors="coerce")
        merged[code] = merged[code].where(merged[code] != -666666666, other=np.nan)

    return merged

def fetch_geometries(year: int) -> gpd.GeoDataFrame:
    """Fetch Census tract geometries for a specific year using pygris."""
    print(f"  |   Fetching TIGER geometries for {year} ... ", end="", flush=True)
    
    # pygris handles the API/FTP routing under the hood
    geo_df = tracts(state=STATE, year=year, cb=True, cache=True)
    
    # Census shapefiles change their ID column names (e.g., GEOID, GEOID10, GEOID20)
    # We find whatever column starts with "GEOID" and standardize it to "geoid"
    id_col = next((col for col in geo_df.columns if col.startswith("GEOID")), None)
    if not id_col:
        raise ValueError(f"Could not locate a GEOID column in the {year} spatial data.")

    # Clear IDS (from 2023 they have a "1400000US" prefix which we don't need) and ensure it's a string
    geo_df[id_col] = geo_df[id_col].str.replace(r"1400000US", "")
    
    print("ok")
    return geo_df[[id_col, "geometry"]].rename(columns={id_col: "geoid"})

# ══════════════════════════════════════════════════════════════════════════════
# ④ DERIVED INDICATOR CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_derived(df: pd.DataFrame, year: int) -> pd.DataFrame:
    d = df.copy()

    # Vintage metadata columns
    d["acs_year"] = year
    d["acs_span"] = f"{year - 4}-{year}"

    # --- % HH no vehicle ---------------------------------------------------
    d["pct_no_vehicle"] = (
        d["B08201_002E"] / d["B08201_001E"].replace(0, np.nan) * 100
    )

    # --- % Overcrowded housing (> 1.00 occ / room) -------------------------
    oc_cols = [
        "B25014_005E", "B25014_006E", "B25014_007E",
        "B25014_011E", "B25014_012E", "B25014_013E",
    ]
    present_oc = [c for c in oc_cols if c in d.columns]
    if present_oc:
        d["_n_overcrowded"] = d[present_oc].sum(axis=1)
        d["pct_overcrowded"] = d["_n_overcrowded"] / d["B25014_001E"].replace(0, np.nan) * 100
    else:
        d["pct_overcrowded"] = np.nan

    # # --- Mean commute time (min) -------------------------------------------
    #   aggregate_minutes / (total_workers - work_from_home_workers)
    wfh = d["B08301_021E"].fillna(0) if "B08301_021E" in d.columns else 0
    commuter_base = (d["B08301_001E"] - wfh).replace(0, np.nan)
    d["mean_commute_min"] = d["B08135_001E"] / commuter_base

    # --- % Below poverty ---------------------------------------------------
    d["pct_below_poverty"] = (
        d["B17001_002E"] / d["B17001_001E"].replace(0, np.nan) * 100
    )

    # --- Education shares --------------------------------------------------
    edu_denom = d["B15003_001E"].replace(0, np.nan)
    for derived_col, raw_list in [
        ("pct_edu_lt_hs",     EDU_LT_HS),
        ("pct_edu_hs_ged",    EDU_HS_GED),
        ("pct_edu_some_col",  EDU_SOME),
        ("pct_edu_bach_plus", EDU_BACH),
    ]:
        present = [c for c in raw_list if c in d.columns]
        d[derived_col] = (d[present].sum(axis=1) / edu_denom * 100) if present else np.nan

    # --- Weighted mean age -------------------------------------------------
    all_age  = {**AGE_MALE_VARS, **AGE_FEMALE_VARS}
    age_denom = d["B01001_001E"].replace(0, np.nan)
    w_sum = sum(
        d[code].fillna(0) * mp
        for code, mp in all_age.items()
        if code in d.columns
    )
    d["mean_age"] = w_sum / age_denom

    return d


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ PER-YEAR PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_year(year: int, out_root: Path, skip_existing: bool = True) -> dict:
    """Full pipeline for one ACS vintage. Returns a status dict."""
    out_dir  = out_root / str(year)
    out_path = out_dir / f"ny_tracts_acs5_{year}.feather"

    # --- Skip if already done ----------------------------------------------
    if skip_existing and out_path.exists():
        size_kb = out_path.stat().st_size // 1024
        print(f"  ↷  {year}  already exists ({size_kb:,} KB) — skipping")
        return {"year": year, "status": "skipped", "rows": None,
                "path": out_path, "error": None}

    print(f"\n  +-- {year}  ({year-4}-{year} ACS 5-year) " + "-" * 32)
    try:
        raw_df = fetch_all_for_year(year)
        print(f"  |   Raw: {len(raw_df):,} tracts  {raw_df.shape[1]} columns")

        derived_df = compute_derived(raw_df, year)
        present = {k: v for k, v in FINAL_COLS.items() if k in derived_df.columns}
        not_present = [k for k, v in FINAL_COLS.items() if k not in derived_df.columns]
        # Report not-present columns (should be rare, but good to know)
        if len(not_present) > 0:
            print(f"  |   Warning: {len(not_present)} expected columns not present in {year}: {not_present}")
            print(f"  |   Available columns: {list(derived_df.columns)}")
        out_df  = derived_df[list(present.keys())].rename(columns=present)

        # --- Spatial Merge -----------------------------------------------------
        # Fetch the shapefiles for this specific year
        geo_df = fetch_geometries(year)
        print(f"  |   Geometries: {len(geo_df):,} tracts")
        # Merge the tabular ACS data into the geometries (right join ensures we keep all tabular rows)
        final_gdf = geo_df.merge(out_df, on="geoid", how="right")
        assert final_gdf["geoid"].is_unique, "GEOID should be unique after merge"
        assert final_gdf["geometry"].notna().any(), "At least one row should have a geometry after merge. Check if GEOID values match between data and geometries."

        # --- Save to Disk ------------------------------------------------------
        out_dir.mkdir(parents=True, exist_ok=True)
        geo_df.to_csv(out_path.with_suffix(".csv"), index=False)
        final_gdf.to_feather(out_path, index=False)

        size_kb = out_path.stat().st_size // 1024
        print(f"  +-- OK  {len(out_df):,} rows x {len(out_df.columns)} cols"
              f"  ->  {out_path}  ({size_kb:,} KB)")

        return {"year": year, "status": "ok", "rows": len(out_df),
                "path": out_path, "error": None}

    except Exception as exc:
        print(f"  +-- FAILED  {year}: {exc}")
        traceback.print_exc()
        return {"year": year, "status": "error", "rows": None,
                "path": None, "error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════════
# ⑥ CLI & MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download ACS 5-year tract estimates for NY State (2009-2024).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Use all defaults  ->  ./acs_ny_tracts/2009/ ... ./acs_ny_tracts/2024/
  python nyc_acs_tract_collector.py

  # Custom output folder
  python nyc_acs_tract_collector.py --outdir /data/acs

  # Only 2015-2022
  python nyc_acs_tract_collector.py --start 2015 --end 2022

  # Re-download even if Feather files already exist
  python nyc_acs_tract_collector.py --force

  # Inline API key (overrides env var)
  python nyc_acs_tract_collector.py --key abc123def456
        """,
    )
    p.add_argument("--outdir",  default=DEFAULT_OUT_ROOT, metavar="PATH",
                   help=f"Root output folder (default: {DEFAULT_OUT_ROOT})")
    p.add_argument("--start",   default=START_YEAR, type=int,
                   help=f"First vintage year inclusive (default: {START_YEAR})")
    p.add_argument("--end",     default=END_YEAR,   type=int,
                   help=f"Last  vintage year inclusive (default: {END_YEAR})")
    p.add_argument("--key",     default=None,
                   help="Census API key (overrides CENSUS_API_KEY env var)")
    p.add_argument("--force",   action="store_true",
                   help="Re-download years that already have a Feather file on disk")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # --- Resolve API key ---------------------------------------------------
    global API_KEY
    if args.key:
        API_KEY = args.key
    if API_KEY == "YOUR_API_KEY_HERE":
        sys.exit(
            "\nERROR: No Census API key found.\n"
            "  Three options:\n"
            "    1. export CENSUS_API_KEY='your_key'\n"
            "    2. python script.py --key your_key\n"
            "    3. Edit API_KEY at the top of this script.\n"
            "  Free key: https://api.census.gov/data/key_signup.html\n"
        )

    if args.start < 2009:
        sys.exit("ERROR: ACS 5-year estimates start in 2009. Use --start 2009 or later.")
    if args.end > 2024:
        sys.exit("ERROR: 2024 is the latest available vintage. Use --end 2024 or earlier.")
    if args.start > args.end:
        sys.exit("ERROR: --start must be <= --end")

    out_root  = Path(args.outdir).expanduser().resolve()
    years     = list(range(args.start, args.end + 1))
    skip_flag = not args.force

    # --- Header ------------------------------------------------------------
    W = 66
    print(f"\n{'='*W}")
    print(f"  ACS 5-Year Estimates — NY State Census Tracts")
    print(f"  Vintages : {args.start} to {args.end}  ({len(years)} years)")
    print(f"  Output   : {out_root}")
    print(f"  Mode     : {'resume (skip existing)' if skip_flag else 'force (overwrite all)'}")
    print(f"{'='*W}\n")

    # --- Year loop ---------------------------------------------------------
    results: list[dict] = []
    t_start = time.time()

    for year in progress(years, desc="Vintages", unit="yr", leave=False):
        result = process_year(year, out_root, skip_existing=skip_flag)
        results.append(result)
        time.sleep(1.0)   # brief cooldown between vintages

    # --- Run summary -------------------------------------------------------
    elapsed = time.time() - t_start
    ok      = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors  = [r for r in results if r["status"] == "error"]

    print(f"\n{'='*W}")
    print(f"  Completed in {elapsed/60:.1f} min")
    print(f"  Downloaded : {len(ok)}")
    print(f"  Skipped    : {len(skipped)}")
    print(f"  Errors     : {len(errors)}")

    if errors:
        print(f"\n  Failed vintages:")
        for r in errors:
            print(f"    {r['year']}  ->  {r['error']}")

    print(f"\n  {out_root}/")
    icons = {"ok": "v", "skipped": "~", "error": "x"}
    for r in results:
        icon     = icons[r["status"]]
        rows_str = f"{r['rows']:,} rows" if r["rows"] else r["status"]
        span     = f"({r['year']-4}-{r['year']})"
        print(f"    [{icon}] {r['year']}/ {span:>13}   {rows_str}")
    print(f"{'='*W}\n")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()