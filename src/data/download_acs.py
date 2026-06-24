#!/usr/bin/env python3
# /mnt/c/Working Papers/NY State Aerial Imagery Prototype/ny_state_aerial_imagery_prototype/src/data/download_acs.py
"""
ACS 5-Year Estimates — United States, Census Tract Level  (2009 → 2024)
=========================================================================
Loops over every ACS 5-year vintage from START_YEAR to END_YEAR and, for
each vintage, over every U.S. state (50 states + DC). Saves one Feather file
per vintage — containing all states — inside a user-defined folder tree:

  OUTPUT_ROOT/
  ├── 2009/   ← 2005-2009 estimates  →  us_tracts_acs5_2009.feather
  ├── 2010/   ← 2006-2010 estimates  →  us_tracts_acs5_2010.feather
  │   …
  └── 2024/   ← 2020-2024 estimates  →  us_tracts_acs5_2024.feather

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

  OCCUPANT-WEALTH INDEX  (for the W_i target diagnostic)
    • V_i  = MEAN owner-occupied home value = B25082 / owner-occupied units.
        The mean is the correct aggregator for a stock quantity: the median
        (B25077) understates the value stock non-uniformly, compressing exactly
        the high-value tracts we want to discriminate.
    • W_i(r) = (1/r)*PerCapInc + per-capita net housing equity. Both terms are
        now dollar stocks PER CAPITA. The per-capita equity term collapses
        algebraically (the owner-unit and occupied-unit counts cancel) to
          (B25082 / total tract population) * [freeclear_i + mortgaged_i*(1-LTV)]
        aggregate owner value       B25082 (Aggregate Value by Mortgage Status)
        owner-occupied units        B25003 (Tenure)  -> alpha_i and V_i denom
        mortgage status shares      B25081 (Mortgage Status) -> leverage bracket
        total tract population      B01001_001E -> shared per-capita denominator
      Total tract population (not B25010 household population) is used so the
      equity term and the per-capita income term share a denominator definition.
      Tracts with no B25082 (suppressed / no owner units) get V_i=0, equity=0.
      One column per discount rate in DISCOUNT_RATES (robustness sweep).

Resilience features
───────────────────
  • Skip / resume   — if a Feather file already exists for a year it is skipped
  • Per-year errors — a single bad year is logged and skipped; run continues
  • Per-state errors — a single bad state is logged and skipped; year continues
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
from dotenv import load_dotenv
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
# Load env so CENSUS_API_KEY is available when this is run as a standalone
# script (it doesn't import src.utils.paths). Non-secret config lives in .env;
# API keys live in .env.secrets, which may be read-denied inside a sandbox, so
# don't crash if it's missing.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")
try:
    load_dotenv(_PROJECT_ROOT / ".env.secrets")
except OSError:
    pass

API_KEY     = os.environ.get("CENSUS_API_KEY")

# FIPS codes for all 50 states + DC (the Census tract API requires querying one
# state at a time, so we loop over these). Add "72" for Puerto Rico if you also
# want PR tracts. Threaded through every fetch so one vintage spans the country.
STATES = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
    "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
    "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
    "34", "35", "36", "37", "38", "39", "40", "41", "42", "44",
    "45", "46", "47", "48", "49", "50", "51", "53", "54", "55",
    "56",
]

START_YEAR  = 2011           # first ACS 5-yr release
END_YEAR    = 2024           # inclusive; update as new vintages drop
MAX_VARS    = 45             # variables per API call (Census hard-cap ~50)
RETRY_MAX   = 1              # max retries on transient HTTP errors
RETRY_SLEEP = 1              # seconds between retries (doubles each attempt)
CALL_SLEEP  = 0.4            # polite pause between chunk calls

# Default output root — override with --outdir or OUTPUT_ROOT env var
DEFAULT_OUT_ROOT = os.environ.get("OUTPUT_ROOT", r"/mnt/e/Datasets/US ACS 5-year Census Tract Estimates")

# ── Occupant-wealth index (W_i) parameters ───────────────────────────────────
# W_i = (1/r)*Y_i  +  per-capita net housing equity. The equity term used to be
# per-household (alpha_i * V_i * leverage / avg_hh_size); it is now per-capita to
# match the per-person income term. Using V_i = mean value = B25082/owner_units,
# the per-household counts cancel and the per-capita equity term reduces to
#   (B25082 / total_tract_population) * [freeclear_i + mortgaged_i*(1 - LTV_MACRO)].
# LTV_MACRO is the macro loan-to-value for mortgaged owners (SCF/AHS ~0.50).
LTV_MACRO = 0.50
# Discount/cap-rate grid for the robustness sweep. One W_i column is produced
# per rate; includes the Circular A-4 baseline (0.02) and spans 1%–7%.
DISCOUNT_RATES = (0.01, 0.02, 0.03, 0.05, 0.07)


def w_index_colname(r: float) -> str:
    """Output column name for the W_i index at discount rate ``r`` (e.g. 0.02 -> 'W_i_r2pct')."""
    return f"W_i_r{round(r * 100, 2):g}pct"

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
    "B19301_001M": "per_capita_income_usd_error",
    # Occupant-wealth index components (W_i / V_i)
    "B25082_001E": None,   # aggregate owner-occupied value ($) -> mean V_i & equity
    "B25003_001E": None,   # tenure: total occupied housing units
    "B25003_002E": None,   # tenure: owner-occupied units  -> alpha_i, V_i denom
    "B25081_001E": None,   # mortgage status: total owner-occupied units
    "B25081_002E": None,   # mortgage status: units with a mortgage
    #   free & clear is derived as (001 - 002); the "without a mortgage" code
    #   moved across vintages (008/009), so we never download it directly.
    "B25010_001E": None,   # average household size (diagnostic; not in W_i now)
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
    "B19301_001M":         "per_capita_income_usd_error",
    "B08135_001E":         "mean_commute_unadjusted_min",
    "B01001_001E":         "total_population",
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

# Occupant-wealth outputs. V_i is now the MEAN owner-occupied value
# (B25082 / owner-occupied units); each W_i column is the capitalized-income +
# per-capita-equity index at one discount rate (see DISCOUNT_RATES). Added here
# so the grid lives in one place. aggregate_owner_value_usd (B25082) is exported
# as a diagnostic since it is the numerator of both V_i and the equity term.
FINAL_COLS["B25082_001E"] = "aggregate_owner_value_usd"
for _col in ("V_i", "homeownership_rate", "pct_owner_with_mortgage",
             "pct_owner_free_clear", "avg_household_size"):
    FINAL_COLS[_col] = _col
for _r in DISCOUNT_RATES:
    FINAL_COLS[w_index_colname(_r)] = w_index_colname(_r)


# ══════════════════════════════════════════════════════════════════════════════
# ③ API HELPERS
# ══════════════════════════════════════════════════════════════════════════════

class CensusAPIError(RuntimeError):
    """Fatal Census API problem (e.g. missing/invalid key).

    Deliberately *not* a subclass of ValueError so it is never mistaken for the
    'variable absent in this vintage' case, which is signalled with ValueError
    and handled by per-variable probing. A CensusAPIError affects every request,
    so it propagates up and aborts the run instead of being probed or skipped.
    """


def chunked(iterable, size: int):
    it = iter(iterable)
    while chunk := list(islice(it, size)):
        yield chunk


def fetch_chunk_with_retry(
    year: int,
    var_codes: list[str],
    base_url: str,
    state: str,
) -> pd.DataFrame:
    """Fetch one chunk with exponential-backoff retries on transient errors."""
    params = {
        "get": "NAME," + ",".join(var_codes),
        "for": "tract:*",
        # Only 2009 requires the list-based 'in' parameter formatting
        "in":  [f"state:{state}", "county:*"] if year == 2009 else f"state:{state} county:*",
        "key": API_KEY,
    }
    sleep = RETRY_SLEEP
    for attempt in range(1, RETRY_MAX + 1):
        try:
            resp = requests.get(base_url, params=params, timeout=90)

            # The Census API serves an HTML error page (usually HTTP 200) when
            # the key is missing or invalid — e.g. <title>Missing Key</title> /
            # <title>Invalid Key</title>. Detect those explicitly so an auth
            # failure is reported clearly instead of being misread, after a
            # failed resp.json(), as every variable being "absent" in the vintage.
            head = resp.text[:512]
            if "Missing Key" in head or "Invalid Key" in head:
                kind = "missing" if "Missing Key" in head else "invalid"
                raise CensusAPIError(
                    f"Census API rejected the request — API key is {kind}. "
                    f"Set CENSUS_API_KEY (in .env.secrets or your environment) "
                    f"or pass --key. The API returned an HTML '{kind.title()} Key' page."
                )

            if resp.status_code == 400:
                raise ValueError(
                    f"HTTP 400 for year {year} — variable(s) likely absent in this vintage."
                )
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError as exc:
                raise CensusAPIError(
                    f"Census API returned non-JSON (HTTP {resp.status_code}, "
                    f"Content-Type {resp.headers.get('Content-Type')!r}). This "
                    f"usually means the key is missing/invalid or the service is "
                    f"down. Body began: {head!r}"
                ) from exc
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
    state: str,
) -> pd.DataFrame | None:
    """
    Fetch one chunk. On HTTP 400, probe each variable individually to find
    the culprit(s), drop them, fill with NaN, then fetch the rest normally.
    """
    try:
        df = fetch_chunk_with_retry(year, chunk, base_url, state)
        return df
    except ValueError:
        good: list[str] = []
        bad:  list[str] = []
        for code in chunk:
            try:
                fetch_chunk_with_retry(year, [code], base_url, state)
                good.append(code)
            except Exception:
                bad.append(code)
        if bad:
            print(f"    ! Codes absent in {year} for state {state} (NaN): {bad}")
        if not good:
            return None
        df = fetch_chunk_with_retry(year, good, base_url, state)
        for code in bad:
            df[code] = np.nan
        return df


def fetch_state_for_year(year: int, state: str) -> pd.DataFrame:
    """
    Pull every variable for one *state* in *year* in MAX_VARS-sized chunks and
    merge into one wide (still string-typed) DataFrame keyed on GEOID.
    """
    if year <= 2009:
        # The API endpoint structure changed after 2009; older vintages use a different URL pattern
        base_url = f"https://api.census.gov/data/{year}/acs5"
    else:
        base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    codes    = list(RAW_VARS.keys())
    chunks   = list(chunked(codes, MAX_VARS))
    frames:  list[pd.DataFrame] = []

    for idx, chunk in enumerate(chunks, 1):
        df = _safe_fetch_chunk(year, chunk, base_url, idx, len(chunks), state)
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
        raise RuntimeError(f"No data retrieved for state {state} in {year}")

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="GEOID", how="outer")

    return merged


def fetch_all_for_year(year: int, states: list[str]) -> pd.DataFrame:
    """
    Pull every variable for every state in *states* and stack them into one
    nationwide DataFrame with numeric dtypes; Census sentinels become NaN.

    A state that fails entirely is logged and skipped so one bad state does not
    abort the whole vintage.
    """
    codes = list(RAW_VARS.keys())
    state_frames: list[pd.DataFrame] = []

    for state in progress(states, desc=f"{year} states", unit="st", leave=False):
        print(f"  |   State {state} … ", end="", flush=True)
        try:
            sdf = fetch_state_for_year(year, state)
        except CensusAPIError:
            # Affects every request — don't mask it as a per-state skip.
            print("FAILED (Census API key error)")
            raise
        except Exception as exc:
            print(f"FAILED ({exc}) — skipping state")
            continue
        print(f"{len(sdf):,} tracts")
        state_frames.append(sdf)

    if not state_frames:
        raise RuntimeError(f"No data retrieved for any state in {year}")

    # Stack states; pandas aligns columns by name and NaN-fills any gaps.
    merged = pd.concat(state_frames, ignore_index=True)

    # Guarantee every expected code column exists (may have been fully absent)
    for code in codes:
        if code not in merged.columns:
            merged[code] = np.nan

    # Numeric coerce + sentinel removal
    for code in codes:
        merged[code] = pd.to_numeric(merged[code], errors="coerce")
        merged[code] = merged[code].where(merged[code] != -666666666, other=np.nan)

    return merged


def _fetch_state_geometry(year: int, state: str) -> gpd.GeoDataFrame:
    """Fetch Census tract geometries for a single state/year using pygris."""
    try:
        # Attempt to get the Cartographic Boundary file (water clipped out)
        geo_df = tracts(state=state, year=year, cb=True, cache=True)
    except Exception:
        # Fallback: if the cb=True file is missing (2009, 2011) or the cache is
        # corrupted (2012), fall back to raw TIGER/Line boundaries (cb=False)
        geo_df = tracts(state=state, year=year, cb=False, cache=True)

    # Census shapefiles change their ID column names (e.g., GEOID, GEOID10, GEOID20)
    # We find whatever column starts with "GEOID" and standardize it to "geoid"
    id_col = next((col for col in geo_df.columns if col.replace("_", "").startswith("GEOID")), None)

    # Fallback for older formats where it might be named CTIDFP00
    if not id_col:
        if "CTIDFP00" in geo_df.columns:
            id_col = "CTIDFP00"
        else:
            raise ValueError(f"Could not locate a GEOID column in the {year} spatial data. Available: {list(geo_df.columns)}")

    # Clear IDS (in 2010-2023 they have a "1400000US", and in 2009 they have a "14000US"
    #   prefix which we don't need)
    geo_df[id_col] = geo_df[id_col].astype(str).str.replace(r"^14000(00)?US", "", regex=True)
    return geo_df[[id_col, "geometry"]].rename(columns={id_col: "geoid"})


def fetch_geometries(year: int, states: list[str]) -> gpd.GeoDataFrame:
    """Fetch and stack Census tract geometries for every state in *states*."""
    print(f"  |   Fetching TIGER geometries for {year} ({len(states)} states) ... ", end="", flush=True)

    geo_frames: list[gpd.GeoDataFrame] = []
    for state in progress(states, desc=f"{year} geoms", unit="st", leave=False):
        try:
            geo_frames.append(_fetch_state_geometry(year, state))
        except Exception as exc:
            print(f"\n  |   ! geometry for state {state} {year} failed ({exc}) — skipping")

    if not geo_frames:
        raise RuntimeError(f"No geometries retrieved for any state in {year}")

    # All states share the same CRS within a vintage, so the concat is safe.
    crs = geo_frames[0].crs
    combined = gpd.GeoDataFrame(
        pd.concat(geo_frames, ignore_index=True), geometry="geometry", crs=crs
    )
    print(f"ok ({len(combined):,} tracts)")
    return combined

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

    # --- Occupant-wealth index (V_i, W_i) ----------------------------------
    # V_i: MEAN owner-occupied home value = aggregate value (B25082) / number of
    # owner-occupied units (B25003_002E). The mean (not the median B25077) is the
    # right aggregator for a stock quantity. Tracts with no aggregate value
    # published (suppressed / no owner units) get V_i = 0.
    agg_owner_value = d["B25082_001E"]
    owner_occupied  = d["B25003_002E"]
    d["V_i"] = (agg_owner_value / owner_occupied.replace(0, np.nan)).fillna(0.0)

    # alpha_i: homeownership rate = owner-occupied / total occupied households.
    d["homeownership_rate"] = d["B25003_002E"] / d["B25003_001E"].replace(0, np.nan)

    # Mortgage-status shares among owner-occupied units (B25081). Free-and-clear
    # is derived as (total - with_mortgage) so it is robust to the across-vintage
    # renumbering of the "without a mortgage" line (008 vs 009).
    owner_units = d["B25081_001E"].replace(0, np.nan)
    d["pct_owner_with_mortgage"] = d["B25081_002E"] / owner_units
    d["pct_owner_free_clear"]    = (d["B25081_001E"] - d["B25081_002E"]) / owner_units

    # Average household size (persons/household). No longer enters W_i — kept as a
    # diagnostic. The per-capita equity term uses total tract population instead,
    # so its denominator matches per-capita income (B19301) exactly.
    d["avg_household_size"] = d["B25010_001E"].replace(0, np.nan)

    # Leverage bracket (net-equity fraction of home value): free-and-clear owners
    # hold 100% equity; mortgaged owners hold (1 - LTV_MACRO). Unchanged.
    equity_fraction = (
        d["pct_owner_free_clear"] * 1.0
        + d["pct_owner_with_mortgage"] * (1.0 - LTV_MACRO)
    )
    # Per-capita net housing equity. The per-household form
    #   alpha_i * V_bar_i * equity_fraction / n_bar_i
    # collapses algebraically (owner-unit and occupied-unit counts cancel) to
    #   (B25082 / total_tract_population) * equity_fraction.
    # Total tract population (B01001_001E) is used so this term and the per-capita
    # income term share a denominator. The renter-zeroing of alpha_i is preserved
    # implicitly: B25082 = 0 where there are no owner units -> equity = 0.
    total_population = d["B01001_001E"].replace(0, np.nan)
    equity_per_capita = (agg_owner_value / total_population) * equity_fraction
    # Tracts with no aggregate owner value published (suppressed / fully renter)
    # contribute zero housing equity rather than NaN, so W_i stays defined
    # (= capitalized income) there.
    equity_per_capita = equity_per_capita.where(agg_owner_value > 0, 0.0)

    # W_i(r) = (1/r) * per-capita income + per-capita net housing equity.
    # One column per discount rate for the robustness sweep.
    y_pc = d["B19301_001E"]
    for r in DISCOUNT_RATES:
        d[w_index_colname(r)] = y_pc / r + equity_per_capita

    return d


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ PER-YEAR PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_year(year: int, out_root: Path, states: list[str], skip_existing: bool = True) -> dict:
    """Full pipeline for one ACS vintage (all states). Returns a status dict."""
    out_dir  = out_root / str(year)
    out_path = out_dir / f"us_tracts_acs5_{year}.feather"

    # --- Skip if already done ----------------------------------------------
    if skip_existing and out_path.exists():
        size_kb = out_path.stat().st_size // 1024
        print(f"  ↷  {year}  already exists ({size_kb:,} KB) — skipping")
        return {"year": year, "status": "skipped", "rows": None,
                "path": out_path, "error": None}

    print(f"\n  +-- {year}  ({year-4}-{year} ACS 5-year) " + "-" * 32)
    try:
        raw_df = fetch_all_for_year(year, states)
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
        # Fetch the shapefiles for this specific year (all states)
        geo_df = fetch_geometries(year, states)
        print(f"  |   Geometries: {len(geo_df):,} tracts")
        # Merge the tabular ACS data into the geometries (right join ensures we keep all tabular rows)
        final_gdf = geo_df.merge(out_df, on="geoid", how="right")
        assert final_gdf["geoid"].is_unique, "GEOID should be unique after merge"
        assert final_gdf["geometry"].notna().any(), "At least one row should have a geometry after merge. Check if GEOID values match between data and geometries."

        # --- Save to Disk ------------------------------------------------------
        out_dir.mkdir(parents=True, exist_ok=True)
        final_gdf.to_feather(out_path, index=False)

        size_kb = out_path.stat().st_size // 1024
        print(f"  +-- OK  {len(out_df):,} rows x {len(out_df.columns)} cols"
              f"  ->  {out_path}  ({size_kb:,} KB)")

        return {"year": year, "status": "ok", "rows": len(out_df),
                "path": out_path, "error": None}

    except CensusAPIError:
        # Fatal and run-wide — bubble up to main() for a clean exit.
        raise
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
        description="Download ACS 5-year tract estimates for the U.S. (all states + DC, 2009-2024).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Use all defaults (every state + DC, 2009-2024)
  python download_acs.py

  # Custom output folder
  python download_acs.py --outdir /data/acs

  # Only 2015-2022
  python download_acs.py --start 2015 --end 2022

  # Only a few states (e.g. NY, NJ, CT) -- handy for testing
  python download_acs.py --states 36,34,09

  # Re-download even if Feather files already exist
  python download_acs.py --force

  # Inline API key (overrides env var)
  python download_acs.py --key abc123def456
        """,
    )
    p.add_argument("--outdir",  default=DEFAULT_OUT_ROOT, metavar="PATH",
                   help=f"Root output folder (default: {DEFAULT_OUT_ROOT})")
    p.add_argument("--start",   default=START_YEAR, type=int,
                   help=f"First vintage year inclusive (default: {START_YEAR})")
    p.add_argument("--end",     default=END_YEAR,   type=int,
                   help=f"Last  vintage year inclusive (default: {END_YEAR})")
    p.add_argument("--states",  default=None, metavar="FIPS",
                   help="Comma-separated state FIPS codes (e.g. 36,34,09). "
                        "Default: all 50 states + DC.")
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
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        sys.exit(
            "\nERROR: No Census API key found.\n"
            "  Looked for CENSUS_API_KEY in the environment and in .env.secrets.\n"
            "  Three options:\n"
            "    1. Add CENSUS_API_KEY='your_key' to .env.secrets\n"
            "    2. export CENSUS_API_KEY='your_key'\n"
            "    3. python download_acs.py --key your_key\n"
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

    # --- Resolve states ----------------------------------------------------
    if args.states:
        states = [s.strip().zfill(2) for s in args.states.split(",") if s.strip()]
    else:
        states = STATES

    # --- Header ------------------------------------------------------------
    W = 66
    print(f"\n{'='*W}")
    print(f"  ACS 5-Year Estimates — U.S. Census Tracts")
    print(f"  Vintages : {args.start} to {args.end}  ({len(years)} years)")
    print(f"  States   : {len(states)} (50 states + DC by default)")
    print(f"  Output   : {out_root}")
    print(f"  Mode     : {'resume (skip existing)' if skip_flag else 'force (overwrite all)'}")
    print(f"{'='*W}\n")

    # --- Year loop ---------------------------------------------------------
    results: list[dict] = []
    t_start = time.time()

    for year in progress(years, desc="Vintages", unit="yr", leave=False):
        try:
            result = process_year(year, out_root, states, skip_existing=skip_flag)
        except CensusAPIError as exc:
            sys.exit(f"\nERROR: {exc}\n")
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
