"""
Build the ACS per-capita-income panel for every large US metro (CBSA).

Generalizes the former NYC-only pipeline to all Core-Based Statistical Areas
(Metropolitan Statistical Areas) with total population > ``MIN_METRO_POP``, over
the 5-year ACS vintages ``PANEL_YEARS`` (2011-2023 by default).

Key design choices (see the plan / CLAUDE.md):
  * City unit = CBSA. Counties nest exactly into CBSAs, so each tract is mapped to
    its metro by a county->CBSA crosswalk lookup on ``geoid[:5]`` (no spatial join).
  * Population for the 500k threshold = sum of tract total population
    (``total_population`` = ACS B01001_001E) within each CBSA, for ``BASE_YEAR``.
  * Relative wealth scores (``Rel_Score``) are z-scores of log per-capita income
    computed *within each CBSA* per year, to strip out city-wide macro drift.
  * ``Valid_Structural_Change`` is a single first-vs-last (2011 vs 2023)
    significance test (p < 0.05) -- no consecutive-pair / yo-yo macro gate.
    The wealth-index analogues (``Valid_Structural_Change_W*``) run the SAME
    first-vs-last test but start at ``VRE_FIRST_YEAR`` (2014 vs 2023), because the
    Census replicate-estimate SEs those indexes need are only published from 2014.
  * Robustness wealth variables (V_i, W_i_r{2,3,5,7}pct) are carried into the
    panel, and per-metro / global Spearman correlations between per-capita income
    and each of them are written to CSV (one file per year, raw + aligned).
"""

import warnings
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import scipy.stats as stats
from src.utils.paths import (
    EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR, TABLES_DIR, ACS_ROOT_DIR,
)

# ── Tunable globals ────────────────────────────────────────────────────────────
BASE_YEAR = 2023                          # all years aligned to this tract vintage
MIN_METRO_POP = 500_000                   # CBSA total-population threshold

# Connecticut switched from 8 traditional counties (09001–09015) to 9 Planning Regions
# (09110–09190) for ACS tract reporting starting with the 2022 vintage. The OMB crosswalk
# (list1_2023.xlsx) uses the new codes, so pre-2022 CT tracts would get no CBSA assignment
# without the recode below. fix_ct_county_fips() patches county FIPS on the fly.
_CT_FIPS_CUTOVER_YEAR = 2022
_CT_FIPS_RECODE = {
    # old county FIPS  →  dominant 2022+ Planning Region FIPS
    "09001": "09120",  # Fairfield    → Greater Bridgeport PR  (CBSA 14860)
    "09003": "09110",  # Hartford     → Capitol PR             (CBSA 25540)
    "09005": "09160",  # Litchfield   → Northwest Hills PR     (no large CBSA)
    "09007": "09130",  # Middlesex    → Lower CT River Valley  (CBSA 25540)
    # 09009 New Haven County intentionally omitted: it straddles two Planning Regions that
    # map to DIFFERENT CBSAs (09170 → New Haven 35300; 09140 → Waterbury 47930). Without a
    # spatial join we cannot tell which region each pre-2022 tract falls in, so all old
    # 09009 tracts are left unrecoded → cbsa_code = NaN → dropped from the panel.
    "09011": "09180",  # New London   → Southeastern CT PR     (no large CBSA)
    "09013": "09110",  # Tolland      → Capitol PR             (CBSA 25540)
    "09015": "09150",  # Windham      → Northeastern CT PR     (no large CBSA)
    # Fairfield (09001) also spans two Planning Regions (09120 / 09190) but both map to the
    # same CBSA 14860, so the recode is unambiguous for CBSA assignment purposes.
}
PANEL_YEARS = list(range(2011, 2024))     # 2011 … 2023 inclusive
ALIGN_CRS = "EPSG:3857"                   # metric CRS for max-overlap area matching

# Robustness wealth variables carried into the panel and correlated against PCI.
# Three index families share the same discount-rate grid (DISCOUNT_RATES):
#   * V_i      -- MEAN owner-occupied home value (diagnostic level).
#   * W_i  (W1)-- baseline: finite-horizon (HUMAN_CAPITAL_YEARS) annuity of per-capita
#                 LABOR earnings + per-capita net housing equity at the flat LTV_MACRO.
#                 LEFT UNCHANGED as the comparison baseline.
#   * W2_i     -- W1 with (a) a tract-varying human-capital horizon N_i from the age
#                 distribution, (b) a year-matched Fed-sourced aggregate LTV, and
#                 (c) a capitalized capital-income term (B19064, perpetuity 1/r_k).
#   * W3_i     -- as W2 but the equity LTV is imputed per tract from owner tenure x
#                 local FHFA house-price appreciation.
# All built by compute_acs_indicators / wealth_from_inputs below.
WEALTH_VARS = [
    "V_i",
    "W_i_r2pct",  "W_i_r3pct",  "W_i_r5pct",  "W_i_r7pct",
    "W2_i_r2pct", "W2_i_r3pct", "W2_i_r5pct", "W2_i_r7pct",
    "W3_i_r2pct", "W3_i_r3pct", "W3_i_r5pct", "W3_i_r7pct",
]
# The W index families only (exclude the V_i level): these get relative scores, replicate-
# variance SEs, and structural-change indicators, mirroring income (Y).
WEALTH_INDEX_VARS = [v for v in WEALTH_VARS if v != "V_i"]

# Per-tract columns pulled from each annual feather (besides geoid/geometry).
PCI_COL = "per_capita_income_usd"
PCI_ERR_COL = "per_capita_income_usd_error"
POP_COL = "total_population"
KEEP_RAW_COLS = [PCI_COL, PCI_ERR_COL, POP_COL] + WEALTH_VARS

# ── Occupant-wealth index (V_i, W_i) parameters ────────────────────────────────
# Moved here from download_acs.py (now strictly fetch+persist). W_i is built by
# compute_acs_indicators from the raw ACS columns persisted in each annual feather.
#
# W_i(r) = finite-horizon human-capital flow  +  per-capita net housing equity
#   * Human capital: per-capita LABOR earnings (B20003 aggregate earnings / total
#     population) capitalized as a FINITE annuity over HUMAN_CAPITAL_YEARS working
#     years -- NOT an infinite 1/r perpetuity. Earnings (wages + self-employment)
#     replace per-capita income (B19301) so capital income (interest, dividends,
#     rent) is not double-counted against the housing-equity / home-value terms.
#   * Equity: (B25082 / total population) * leverage, where
#     leverage = freeclear*1 + mortgaged*(1 - LTV_MACRO).
LTV_MACRO = 0.50               # macro loan-to-value for mortgaged owners (SCF/AHS ~0.50)
HUMAN_CAPITAL_YEARS = 20       # finite horizon over which labor earnings are capitalized
DISCOUNT_RATES = (0.02, 0.03, 0.05, 0.07)   # one W_i column per rate; matches WEALTH_VARS

# ── W2 / W3 extra parameters ────────────────────────────────────────────────────
# Capital-income term (W2/W3): per-capita interest+dividend+net-rental income (B19064)
# capitalized as a PERPETUITY 1/r_k, where r_k is a GROSS asset yield, distinct from the
# net-of-growth labor rate rho (= DISCOUNT_RATES). Financial assets don't expire at
# retirement, hence a perpetuity, not the finite human-capital annuity. CAPITAL_YIELD is
# the headline r_k; CAPITAL_YIELD_GRID drives the r_k sensitivity report.
CAPITAL_YIELD = 0.045
CAPITAL_YIELD_GRID = (0.040, 0.045, 0.050)
WORKING_EXIT_AGE = 65          # age at which the human-capital flow is assumed to stop

# Successive-difference replication (SDR) for the Census Variance Replicate Estimate (VRE)
# tables: 80 replicates per estimate -> Var(X) = (4/80) * sum_r (X_r - X_0)^2. Recomputing an
# index on each replicate and taking this variance captures the cross-table covariance that
# delta-method / independent-Monte-Carlo SEs ignore (the component tables are correlated).
N_REPLICATES = 80
SDR_SCALE = 4.0 / N_REPLICATES

# W2 equity term: aggregate loan-to-value anchored to the Federal Reserve Financial
# Accounts (Z.1) household "owners' equity in real estate as a % of household real estate"
# series (FRED HOEREPHRE, quarterly %). LTV2(year) = 1 - equity_share(year). A sourced,
# time-varying replacement for the flat LTV_MACRO guess. The share per ACS vintage ending in
# year Y is the mean of the quarterly series over the 5-year window Y-4..Y (matches the ACS
# 5-year span); computed at runtime by load_owners_equity_share() from the CSV the user placed
# under data/external. The fallback dict below is used only when that file is absent (sandbox /
# W1-only) and is APPROXIMATE -- the real values come from HOEREPHRE.csv.
OWNERS_EQUITY_FILE = EXTERNAL_DATA_DIR / "FRED Households Owners Equity" / "HOEREPHRE.csv"
OWNERS_EQUITY_WINDOW = 5       # years averaged (Y-4..Y) to match the ACS 5-year estimate
_OWNERS_EQUITY_FALLBACK = {
    2011: 0.465, 2012: 0.490, 2013: 0.535, 2014: 0.555, 2015: 0.575,
    2016: 0.585, 2017: 0.600, 2018: 0.605, 2019: 0.630, 2020: 0.655,
    2021: 0.680, 2022: 0.705, 2023: 0.700,
}
_OWNERS_EQUITY_CACHE = None    # lazily filled by _owners_equity_shares()

# W3 equity term: current LTV imputed as (ORIG_LTV * remaining-amortization) / appreciation.
ORIG_LTV = 0.80                # typical purchase-origination loan-to-value
MORTGAGE_RATE = 0.05           # nominal rate for the amortization schedule
MORTGAGE_TERM = 30             # standard fixed-rate term (years)


def load_owners_equity_share(path=OWNERS_EQUITY_FILE, window=OWNERS_EQUITY_WINDOW):
    """``{ACS end-year Y: owners'-equity share (fraction)}`` from FRED HOEREPHRE.

    Reads the quarterly ``observation_date,HOEREPHRE`` CSV (percent), averages to annual, then
    for each Y returns the mean over the 5-year window ``Y-4..Y`` divided by 100. Falls back to
    :data:`_OWNERS_EQUITY_FALLBACK` (approximate) when the file is absent.
    """
    if path is None or not Path(path).exists():
        warnings.warn(
            f"HOEREPHRE file not found at {path}; W2 LTV uses the approximate fallback "
            f"owners'-equity table. Drop the FRED CSV there for sourced, year-matched values."
        )
        return dict(_OWNERS_EQUITY_FALLBACK)
    raw = pd.read_csv(path)
    date_col, val_col = raw.columns[0], raw.columns[1]
    year = pd.to_datetime(raw[date_col], errors="coerce").dt.year
    val = pd.to_numeric(raw[val_col], errors="coerce") / 100.0
    annual = pd.DataFrame({"year": year, "v": val}).dropna().groupby("year")["v"].mean()
    if annual.empty:
        return dict(_OWNERS_EQUITY_FALLBACK)
    lo, hi = int(annual.index.min()), int(annual.index.max())
    return {
        Y: float(annual.loc[(annual.index >= Y - window + 1) & (annual.index <= Y)].mean())
        for Y in range(lo + window - 1, hi + 1)
        if annual.loc[(annual.index >= Y - window + 1) & (annual.index <= Y)].size
    }


def _owners_equity_shares():
    """Cached owners'-equity share table (loaded once from HOEREPHRE.csv or the fallback)."""
    global _OWNERS_EQUITY_CACHE
    if _OWNERS_EQUITY_CACHE is None:
        _OWNERS_EQUITY_CACHE = load_owners_equity_share()
    return _OWNERS_EQUITY_CACHE


def w_index_colname(r):
    """W_i column name at discount rate ``r`` (e.g. 0.02 -> 'W_i_r2pct')."""
    return f"W_i_r{round(r * 100, 2):g}pct"


def w2_index_colname(r):
    """W2_i column name at discount rate ``r`` (e.g. 0.02 -> 'W2_i_r2pct')."""
    return f"W2_i_r{round(r * 100, 2):g}pct"


def w3_index_colname(r):
    """W3_i column name at discount rate ``r`` (e.g. 0.02 -> 'W3_i_r2pct')."""
    return f"W3_i_r{round(r * 100, 2):g}pct"


def ltv_macro_w2(year, shares=None):
    """Year-matched aggregate LTV for W2 = 1 - Fed owners'-equity share (nearest year).

    ``shares`` defaults to the cached HOEREPHRE-derived table (:func:`_owners_equity_shares`).
    """
    shares = shares if shares is not None else _owners_equity_shares()
    share = shares.get(year)
    if share is None:
        share = shares[min(shares, key=lambda y: abs(y - year))]
    return 1.0 - share


def annuity_factor(r, n=HUMAN_CAPITAL_YEARS):
    """Present-value factor of an ordinary annuity of $1/yr for ``n`` years at rate r:
    ``PV = (1 - (1+r)**-n) / r``. Finite-horizon analogue of the 1/r perpetuity factor
    (to which it converges as n -> inf)."""
    return (1.0 - (1.0 + r) ** (-n)) / r


# Demographic-derivation constants (the same ACS codes download_acs.py pulls). Age
# midpoints feed the B01001 weighted-mean age; education bands group B15003 lines.
_AGE_MIDPOINTS = [
    2.5, 7.0, 12.0, 16.0, 18.5, 20.0, 21.0, 23.0,
    27.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0,
    60.5, 63.0, 65.5, 68.0, 72.0, 77.0, 82.0, 90.0,
]
AGE_MALE_VARS   = {f"B01001_{i+3:03d}E":  mp for i, mp in enumerate(_AGE_MIDPOINTS)}
AGE_FEMALE_VARS = {f"B01001_{i+27:03d}E": mp for i, mp in enumerate(_AGE_MIDPOINTS)}
EDU_LT_HS  = [f"B15003_{i:03d}E" for i in range(2, 17)]
EDU_HS_GED = ["B15003_017E", "B15003_018E"]
EDU_SOME   = ["B15003_019E", "B15003_020E", "B15003_021E"]
EDU_BACH   = ["B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"]

# B01001 age brackets that fall in the 16-64 working-age range (by midpoint). Used for
# the W2/W3 tract-varying human-capital horizon N_i (expected remaining working years).
WORKING_AGE_IDX = [i for i, mp in enumerate(_AGE_MIDPOINTS) if 16.0 <= mp <= 64.0]
WORKING_AGE_VARS = {
    f"B01001_{i + off:03d}E": _AGE_MIDPOINTS[i]
    for i in WORKING_AGE_IDX for off in (3, 27)   # 3 = male, 27 = female line offsets
}

# B25038 owner move-in brackets (003..008) used for W3 median tenure.
MOVEIN_OWNER_VARS = [f"B25038_{i:03d}E" for i in range(3, 9)]

# Raw ACS columns the wealth index needs. CORE cols are required for V_i + W1 (baseline)
# and raise if absent; the EXTRA cols power the W2/W3 extensions (capital income, age
# distribution for N_i, owner tenure for the W3 LTV imputation) and degrade gracefully
# (the helpers return 0 / fall back) so W1-only feathers still process.
WEALTH_CORE_COLS = [
    "aggregate_earnings_usd",      # B20003_001E -> per-capita labor-income flow
    "aggregate_owner_value_usd",   # B25082_001E -> V_i & equity numerator
    "total_population",            # B01001_001E -> shared per-capita denominator
    "B25003_001E", "B25003_002E",  # tenure: occupied total / owner-occupied
    "B25081_001E", "B25081_002E",  # mortgage status: owner total / with mortgage
]
WEALTH_EXTRA_COLS = [
    "aggregate_capital_income_usd",     # B19064_001E -> capitalized capital-income term
    *WORKING_AGE_VARS,                  # B01001 working-age brackets -> N_i
    "B25038_002E", *MOVEIN_OWNER_VARS,  # owner tenure distribution -> W3 LTV imputation
]
WEALTH_INPUT_COLS = WEALTH_CORE_COLS + WEALTH_EXTRA_COLS


# ── W2 / W3 building-block helpers (pure; unit-tested on synthetic data) ─────────

def expected_working_years(d, exit_age=WORKING_EXIT_AGE):
    """Tract-varying human-capital horizon N_i (expected remaining working years).

    ``N_i = sum_{a in 16-64} s_{a,i} * max(0, exit_age - a_bar)`` where ``s_{a,i}`` is the
    share of the 16-64 population in bracket ``a`` and ``a_bar`` its midpoint. Retiree-heavy
    tracts get a small N_i (short annuity -> the index leans on assets); young tracts get N_i
    near ``exit_age - 16``. Tracts with no working-age population get N_i = 0.
    """
    present = [(c, mp) for c, mp in WORKING_AGE_VARS.items() if c in d]
    if not present:
        return pd.Series(0.0, index=d.index)
    work_pop = sum(d[c].fillna(0) for c, _ in present)
    weighted = sum(d[c].fillna(0) * max(0.0, exit_age - mp) for c, mp in present)
    return (weighted / work_pop.replace(0, np.nan)).fillna(0.0)


def capital_income_pc(d, r_k=CAPITAL_YIELD):
    """Per-capita capital income (B19064) capitalized as a perpetuity 1/r_k. 0 if absent."""
    if "aggregate_capital_income_usd" not in d:
        return pd.Series(0.0, index=d.index)
    pop = d["total_population"].replace(0, np.nan)
    return ((1.0 / r_k) * (d["aggregate_capital_income_usd"] / pop)).fillna(0.0)


def amortization_remaining(t, rate=MORTGAGE_RATE, term=MORTGAGE_TERM):
    """Fraction of original principal still outstanding after ``t`` years of a level-payment
    ``term``-year loan at ``rate``: ``[(1+i)^T - (1+i)^t] / [(1+i)^T - 1]``. 1 at t=0, 0 at
    t>=T, monotonically decreasing. Accepts a scalar or array/Series ``t``."""
    t = np.clip(np.asarray(t, dtype="float64"), 0.0, None)
    g = (1.0 + rate) ** term
    rem = (g - (1.0 + rate) ** t) / (g - 1.0)
    return np.clip(rem, 0.0, 1.0)


def _owner_movein_bounds(year):
    """Approximate [lo, hi] move-in YEAR bounds for B25038 owner brackets 003..008 of the
    5-year ACS vintage ending in ``year``. The two most-recent brackets roll with the
    vintage; the older ones are ~decade bins. The 008 bin is open-ended below (a finite
    1960 floor is used only so the grouped median can interpolate within it).

    NOTE: the Census bracket cut-points have shifted across releases and the docs are
    egress-blocked in this sandbox, so these are APPROXIMATE. W3 uses the tenure ORDERING
    across tracts (robust to modest bound error); verify the exact shells per vintage before
    relying on absolute tenure. Bounds are clamped to stay ordered/non-degenerate.
    """
    raw = {
        "B25038_003E": (year - 2, year),
        "B25038_004E": (year - 5, year - 3),
        "B25038_005E": (2010, year - 6),
        "B25038_006E": (2000, 2009),
        "B25038_007E": (1990, 1999),
        "B25038_008E": (1960, 1989),
    }
    # Clamp each interval to lo<=hi and within [1960, year] so interpolation is well-defined.
    out = {}
    for code, (lo, hi) in raw.items():
        lo = min(max(lo, 1960), year)
        hi = min(max(hi, 1960), year)
        out[code] = (min(lo, hi), max(lo, hi))
    return out


def owner_median_tenure(d, year):
    """Tract owner median tenure (years) from the B25038 owner move-in distribution.

    Grouped-median move-in year by linear interpolation within the median bracket (bounds
    from :func:`_owner_movein_bounds`), then ``tenure = year - median_move_in``. Brackets
    absent in ``d`` are skipped; tracts with no owner records get NaN (W3 then falls back to
    no revaluation for that tract).
    """
    bounds = _owner_movein_bounds(year)
    codes = [c for c in MOVEIN_OWNER_VARS if c in d]
    if not codes:
        return pd.Series(np.nan, index=d.index)
    # Order brackets earliest -> latest move-in for the cumulative distribution.
    codes = sorted(codes, key=lambda c: bounds[c][0])
    counts = np.vstack([d[c].fillna(0).to_numpy(dtype="float64") for c in codes])  # (B, N)
    total = counts.sum(axis=0)
    target = total / 2.0
    cum = np.cumsum(counts, axis=0)
    median_year = np.full(total.shape, np.nan)
    prev_cum = np.zeros(total.shape)
    for b, c in enumerate(codes):
        lo, hi = bounds[c]
        in_bracket = (cum[b] >= target) & np.isnan(median_year) & (total > 0)
        cnt = counts[b]
        frac = np.where(cnt > 0, (target - prev_cum) / np.where(cnt == 0, 1, cnt), 0.0)
        frac = np.clip(frac, 0.0, 1.0)
        median_year = np.where(in_bracket, lo + frac * (hi - lo), median_year)
        prev_cum = cum[b]
    tenure = year - median_year
    return pd.Series(tenure, index=d.index)


def equity_pc(d, ltv):
    """Per-capita net owner home equity = (agg owner value / pop) * (1 - m_i * ltv).

    ``ltv`` is a scalar (W1: LTV_MACRO; W2: year LTV) or a per-tract Series (W3). The factor
    ``1 - m_i*ltv`` equals ``freeclear*1 + mortgaged*(1-ltv)`` with mortgaged share m_i. Zeroed
    where no owner value is published so the index stays defined (= non-equity terms there).
    """
    agg = d["aggregate_owner_value_usd"]
    pop = d["total_population"].replace(0, np.nan)
    owner_units = d["B25081_001E"].replace(0, np.nan)
    m = d["B25081_002E"] / owner_units                       # mortgaged share of owners
    phi = 1.0 - m * ltv
    eq = (agg / pop) * phi
    return eq.where(agg > 0, 0.0).fillna(0.0)


# FHFA all-transactions annual house-price indexes (developmental county + CBSA files), used
# for the W3 tenure x appreciation LTV imputation. The user's .xlsx files carry ~6 note rows
# before the header row (CBSA|Name|Year|...|HPI|... and State|County|FIPS code|Year|...|HPI|...).
# Absent files -> appreciation falls back to 1.0 (no revaluation), so W3 still computes.
FHFA_DIR = EXTERNAL_DATA_DIR / "FHFA Housing Price Index"
FHFA_COUNTY_FILE = FHFA_DIR / "hpi_at_county.xlsx"
FHFA_CBSA_FILE = FHFA_DIR / "hpi_at_cbsa.xlsx"


def _read_fhfa_hpi(path, key_needles, key_name):
    """Tidy ``[{key_name}, year, hpi]`` from one FHFA .xlsx (annual mean of the 'HPI' column).

    Auto-detects the header row (the first row containing both 'year' and an exact 'hpi' cell,
    skipping FHFA's note rows) and fuzzy-matches the key / year / HPI columns. Returns ``None``
    (with a warning) when the file is absent so the W3 appreciation term degrades to 1.0.
    """
    if path is None or not Path(path).exists():
        warnings.warn(
            f"FHFA HPI not found at {path}; that tier of the W3 appreciation term is skipped "
            f"(falls back to the other tier / 1.0). Drop the FHFA .xlsx there to enable it."
        )
        return None
    preview = pd.read_excel(path, header=None, nrows=15)
    hdr = next(
        (i for i in range(len(preview))
         if "year" in [str(x).strip().lower() for x in preview.iloc[i]]
         and "hpi" in [str(x).strip().lower() for x in preview.iloc[i]]),
        None,
    )
    if hdr is None:
        raise ValueError(f"Could not locate the FHFA header row (year/HPI) in {path}")
    df = pd.read_excel(path, header=hdr)
    kcol, ycol, hcol = (_find_col(df.columns, *key_needles),
                        _find_col(df.columns, "year"), _find_col(df.columns, "hpi"))
    out = pd.DataFrame({
        key_name: df[kcol].astype(str).str.strip(),
        "year": pd.to_numeric(df[ycol], errors="coerce"),
        "hpi": pd.to_numeric(df[hcol], errors="coerce"),  # "." missing -> NaN
    }).dropna(subset=["year", "hpi"])
    out["year"] = out["year"].astype(int)
    return out.groupby([key_name, "year"], as_index=False)["hpi"].mean()


def load_fhfa_county_hpi(path: "Path | None" = FHFA_COUNTY_FILE):
    """FHFA county HPI -> ``[county_fips, year, hpi]`` (county_fips zero-padded to 5 = geoid[:5])."""
    out = _read_fhfa_hpi(path, ("fips",), "county_fips")
    if out is not None:
        out["county_fips"] = out["county_fips"].str.zfill(5)
    return out


def load_fhfa_metro_hpi(path: "Path | None" = FHFA_CBSA_FILE):
    """FHFA CBSA HPI -> ``[cbsa_code, year, hpi]`` (matches the crosswalk's 5-digit cbsa_code)."""
    return _read_fhfa_hpi(path, ("cbsa",), "cbsa_code")


def load_fhfa_hpi(county_path=FHFA_COUNTY_FILE, metro_path=FHFA_CBSA_FILE):
    """Bundle both FHFA tiers for the W3 appreciation term: ``{'county': df|None, 'metro': df|None}``."""
    return {"county": load_fhfa_county_hpi(county_path), "metro": load_fhfa_metro_hpi(metro_path)}


def _hpi_lookups(frame, key):
    """``{key_value: (years[], hpi[])}`` sorted by year for np.interp; ``{}`` if frame is None."""
    out = {}
    if frame is not None:
        for k, sub in frame.groupby(key):
            s = sub.sort_values("year")
            out[k] = (s["year"].to_numpy("float64"), s["hpi"].to_numpy("float64"))
    return out


def appreciation_factor(d, year, hpi=None, crosswalk=None):
    """Per-tract house-price appreciation since the owner median move-in year.

    ``HPI(year) / HPI(median_move_in)`` with a COUNTY-first, CBSA-fallback, 1.0-last lookup:
    the county series for ``geoid[:5]`` is used if present, else the tract's CBSA series (via the
    crosswalk), else no revaluation. ``hpi`` is the ``{'county','metro'}`` bundle from
    :func:`load_fhfa_hpi`. ``np.interp`` handles gaps / pre-start move-in years (clips to the
    series endpoints). Falls back to 1.0 wherever HPI/tenure is missing, so W3 always computes.
    """
    fallback = pd.Series(1.0, index=d.index)
    if not hpi or "geoid" not in d:
        return fallback
    county_hpi, metro_hpi = hpi.get("county"), hpi.get("metro")
    if county_hpi is None and metro_hpi is None:
        return fallback

    county = d["geoid"].astype(str).str.zfill(11).str[:5]
    move_in = (year - owner_median_tenure(d, year)).round().to_numpy()
    cty = _hpi_lookups(county_hpi, "county_fips")
    met = _hpi_lookups(metro_hpi, "cbsa_code")
    cmap = _county_to_cbsa_map(crosswalk) if crosswalk is not None else {}

    county_arr = county.to_numpy()
    cur = np.full(len(d), np.nan)
    base = np.full(len(d), np.nan)
    for i in range(len(d)):
        if not np.isfinite(move_in[i]):
            continue
        ser = cty.get(county_arr[i]) or met.get(cmap.get(county_arr[i]))
        if ser is None or len(ser[0]) == 0:
            continue
        yrs, vals = ser
        cur[i] = np.interp(year, yrs, vals)
        base[i] = np.interp(move_in[i], yrs, vals)
    fac = pd.Series(cur / base, index=d.index)
    return fac.where(fac > 0).fillna(1.0)


def impute_ltv_w3(d, year, hpi=None, crosswalk=None):
    """Per-tract current LTV for W3 = (ORIG_LTV * remaining-amortization(tenure)) / appreciation,
    clipped to [0, 1]. Long-tenure / appreciated tracts -> low LTV (high equity); recent buyers
    -> high LTV. Missing tenure -> ORIG_LTV with no revaluation (appreciation = 1)."""
    tenure = owner_median_tenure(d, year).fillna(0.0).clip(lower=0.0)
    amort = pd.Series(amortization_remaining(tenure), index=d.index)
    appr = appreciation_factor(d, year, hpi, crosswalk)
    return ((ORIG_LTV * amort) / appr).clip(lower=0.0, upper=1.0)


# County->CBSA crosswalk. Primary source is the Census Bureau's authoritative OMB
# delineation file (List 1, ``list1_2023.xlsx``), placed locally under data/external.
# The former NBER CSV mirror is kept only as a network fallback (its URL now 404s).
# A normalized CSV is written to ``CBSA_CROSSWALK_CACHE`` after the first parse.
CBSA_CROSSWALK_XLSX = (
    EXTERNAL_DATA_DIR / "CVSA-FIPS Country Crosswalk" / "list1_2023.xlsx"
)
CBSA_XLSX_HEADER_ROW = 2  # 0-indexed: rows 0-1 are titles; row 2 holds column headers
CBSA_CROSSWALK_URL = "https://data.nber.org/cbsa-csa-fips-county-crosswalk/cbsa2fipsxw.csv"
CBSA_CROSSWALK_CACHE = EXTERNAL_DATA_DIR / "cbsa" / "county_cbsa_crosswalk.csv"
METRO_KIND = "Metropolitan Statistical Area"


# ══════════════════════════════════════════════════════════════════════════════
# ① County -> CBSA crosswalk + large-metro selection
# ══════════════════════════════════════════════════════════════════════════════

def _find_col(columns, *needles):
    """Return the first column whose alphanumeric-lowercased name contains all needles."""
    norm = {c: "".join(ch for ch in str(c).lower() if ch.isalnum()) for c in columns}
    for col, n in norm.items():
        if all(needle in n for needle in needles):
            return col
    raise KeyError(f"No column matching {needles} in {list(columns)}")


def _normalize_crosswalk(raw: pd.DataFrame) -> pd.DataFrame:
    """Reduce a raw CBSA delineation table to ``[county_fips, cbsa_code, cbsa_title]``.

    Works for both the Census ``list1`` layout and the NBER CSV mirror: columns are
    located by fuzzy name match (:func:`_find_col`) and only Metropolitan Statistical
    Areas are kept. Trailing note / blank rows (NaN CBSA code) are dropped.
    """
    code_col = _find_col(raw.columns, "cbsa", "code")
    title_col = _find_col(raw.columns, "cbsa", "title")
    kind_col = _find_col(raw.columns, "metropolitanmicropolitan")
    state_col = _find_col(raw.columns, "fips", "state")
    county_col = _find_col(raw.columns, "fips", "county")

    df = raw[[code_col, title_col, kind_col, state_col, county_col]].copy()
    df.columns = ["cbsa_code", "cbsa_title", "kind", "state_fips", "county_fips3"]
    df = df.dropna(subset=["cbsa_code", "kind", "state_fips", "county_fips3"])
    df = df[df["kind"].str.strip() == METRO_KIND]
    df["county_fips"] = (
        df["state_fips"].str.strip().str.zfill(2)
        + df["county_fips3"].str.strip().str.zfill(3)
    )
    df["cbsa_code"] = df["cbsa_code"].str.strip()
    out = df[["county_fips", "cbsa_code", "cbsa_title"]].dropna().drop_duplicates()
    return out.reset_index(drop=True)


def load_county_cbsa_crosswalk(xlsx: "Path | None" = CBSA_CROSSWALK_XLSX,
                               url: str = CBSA_CROSSWALK_URL,
                               cache: "Path | None" = CBSA_CROSSWALK_CACHE) -> pd.DataFrame:
    """County->CBSA crosswalk, restricted to Metropolitan Statistical Areas.

    Returns a tidy DataFrame ``[county_fips, cbsa_code, cbsa_title]`` where
    ``county_fips`` is the 5-char state+county FIPS string.

    Source precedence: (1) the normalized CSV cache, if present; (2) the local
    Census Bureau OMB delineation file (List 1 ``.xlsx``); (3) the NBER CSV mirror
    as a network fallback. The first successful source is normalized and written to
    the CSV cache for subsequent runs.
    """
    if cache is not None and cache.exists():
        return pd.read_csv(cache, dtype=str)

    if xlsx is not None and xlsx.exists():
        print(f"Reading county->CBSA crosswalk from {xlsx} ...")
        raw = pd.read_excel(xlsx, dtype=str, header=CBSA_XLSX_HEADER_ROW)
    else:
        print(f"Local crosswalk not found at {xlsx}; downloading from {url} ...")
        try:
            raw = pd.read_csv(url, dtype=str, encoding="latin-1")
        except Exception as exc:  # network blocked / bad URL
            raise RuntimeError(
                f"Could not obtain the county->CBSA crosswalk. Place the Census OMB "
                f"delineation file (List 1) at {xlsx}, or a normalized CSV with columns "
                f"[county_fips, cbsa_code, cbsa_title] at {cache}, and re-run."
            ) from exc

    out = _normalize_crosswalk(raw)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache, index=False)
    return out


def _county_to_cbsa_map(crosswalk: pd.DataFrame) -> dict:
    return dict(zip(crosswalk["county_fips"], crosswalk["cbsa_code"]))


def _cbsa_title_map(crosswalk: pd.DataFrame) -> dict:
    return dict(zip(crosswalk["cbsa_code"], crosswalk["cbsa_title"]))


def get_large_metros(base_year: int = BASE_YEAR,
                     min_pop: int = MIN_METRO_POP,
                     crosswalk: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """Large US metros by ACS population (task 1).

    Sums tract total population (``total_population``) within each CBSA for
    ``base_year`` and returns those above ``min_pop``. The CBSA code is the
    ACS-matching identifier requested by the user.

    Returns ``[cbsa_code, cbsa_title, population]`` sorted by population desc.
    """
    if crosswalk is None:
        crosswalk = load_county_cbsa_crosswalk()

    path = ACS_ROOT_DIR / str(base_year) / f"us_tracts_acs5_{base_year}.feather"
    print(f"Selecting metros > {min_pop:,} from {path} ...")
    gdf = gpd.read_feather(path)
    df = pd.DataFrame(gdf[["geoid", POP_COL]]).copy()
    df["geoid"] = df["geoid"].astype(str).str.zfill(11)
    df["county_fips"] = df["geoid"].str[:5]
    df["cbsa_code"] = df["county_fips"].map(_county_to_cbsa_map(crosswalk))

    pop = (
        df.dropna(subset=["cbsa_code"])
          .groupby("cbsa_code", as_index=False)[POP_COL].sum()
          .rename(columns={POP_COL: "population"})
    )
    pop["cbsa_title"] = pop["cbsa_code"].map(_cbsa_title_map(crosswalk))
    large = pop[pop["population"] > min_pop].copy()
    large = large.sort_values("population", ascending=False).reset_index(drop=True)
    print(f"  {len(large)} metros above {min_pop:,} inhabitants.")
    return large[["cbsa_code", "cbsa_title", "population"]]


def fix_ct_county_fips(county: pd.Series, year: int) -> pd.Series:
    """Recode pre-2022 CT county FIPS to 2022+ Planning Region FIPS.

    Starting with the 2022 ACS vintage, Connecticut replaced 8 traditional county FIPS
    (09001–09015) with 9 Planning Region FIPS (09110–09190). The OMB crosswalk references
    the new codes, so pre-2022 CT tracts need this recode before the CBSA lookup.
    Returns ``county`` unchanged for year >= 2022 or non-CT prefixes.
    """
    if year >= _CT_FIPS_CUTOVER_YEAR:
        return county
    return county.replace(_CT_FIPS_RECODE)


def tag_cbsa(gdf, geoid_col: str, crosswalk: pd.DataFrame, year: int = None):
    """Add a ``cbsa_code`` column to ``gdf`` from the county prefix of ``geoid_col``.

    Pass ``year`` so that pre-2022 Connecticut county FIPS are recoded to Planning Region
    codes before the crosswalk lookup (see :func:`fix_ct_county_fips`).
    """
    county = gdf[geoid_col].astype(str).str.zfill(11).str[:5]
    if year is not None:
        county = fix_ct_county_fips(county, year)
    gdf = gdf.copy()
    gdf["cbsa_code"] = county.map(_county_to_cbsa_map(crosswalk))
    return gdf


# ══════════════════════════════════════════════════════════════════════════════
# ② Per-year load / prep and per-CBSA relative scores
# ══════════════════════════════════════════════════════════════════════════════

def compute_acs_indicators(df, year, hpi=None, crosswalk=None):
    """Derive every ACS indicator from the raw columns persisted by download_acs.py.

    Generates the demographic ratios (vehicle / overcrowding / poverty / education),
    the weighted mean age, and the occupant-wealth indexes (V_i + the W1/W2/W3 families,
    one column per rate in DISCOUNT_RATES). This logic was moved out of download_acs.py so
    the download layer stays a pure fetch + persist step. Demographic indicators whose raw
    inputs are absent in a given vintage are skipped; the wealth index raises a clear error
    if its CORE inputs are missing and warns/degrades for the W2/W3 extras (see
    :func:`_compute_wealth_index`). ``hpi``/``crosswalk`` enable the W3 appreciation term.
    """
    d = df.copy()

    # --- % HH no vehicle ---------------------------------------------------
    if {"B08201_001E", "B08201_002E"} <= set(d.columns):
        d["pct_hh_no_vehicle"] = d["B08201_002E"] / d["B08201_001E"].replace(0, np.nan) * 100

    # --- % Overcrowded housing (> 1.00 occ / room) -------------------------
    oc = ["B25014_005E", "B25014_006E", "B25014_007E",
          "B25014_011E", "B25014_012E", "B25014_013E"]
    present_oc = [c for c in oc if c in d.columns]
    if present_oc and "B25014_001E" in d.columns:
        d["pct_overcrowded_housing"] = (
            d[present_oc].sum(axis=1) / d["B25014_001E"].replace(0, np.nan) * 100
        )

    # --- Mean commute (min) = aggregate minutes / (workers - WFH) ----------
    if {"B08135_001E", "B08301_001E"} <= set(d.columns):
        wfh = d["B08301_021E"].fillna(0) if "B08301_021E" in d.columns else 0
        d["mean_commute_min"] = d["B08135_001E"] / (d["B08301_001E"] - wfh).replace(0, np.nan)

    # --- % Below poverty ---------------------------------------------------
    if {"B17001_001E", "B17001_002E"} <= set(d.columns):
        d["pct_below_poverty"] = d["B17001_002E"] / d["B17001_001E"].replace(0, np.nan) * 100

    # --- Education shares --------------------------------------------------
    if "B15003_001E" in d.columns:
        edu_denom = d["B15003_001E"].replace(0, np.nan)
        for col, raw_list in [
            ("pct_edu_lt_hs", EDU_LT_HS), ("pct_edu_hs_ged", EDU_HS_GED),
            ("pct_edu_some_college", EDU_SOME), ("pct_edu_bach_plus", EDU_BACH),
        ]:
            present = [c for c in raw_list if c in d.columns]
            if present:
                d[col] = d[present].sum(axis=1) / edu_denom * 100

    # --- Weighted mean age -------------------------------------------------
    all_age = {**AGE_MALE_VARS, **AGE_FEMALE_VARS}
    age_codes = [c for c in all_age if c in d.columns]
    if age_codes and "total_population" in d.columns:
        w_sum = sum(d[c].fillna(0) * all_age[c] for c in age_codes)
        d["mean_age_years"] = w_sum / d["total_population"].replace(0, np.nan)

    # --- Occupant-wealth indexes (V_i, W1, W2, W3) -------------------------
    d = _compute_wealth_index(d, year, hpi=hpi, crosswalk=crosswalk)
    return d


def wealth_from_inputs(d, year, hpi=None, crosswalk=None, r_k=CAPITAL_YIELD):
    """Pure wealth-index arithmetic over a frame of raw ACS component columns.

    Returns a ``{column_name: Series}`` dict for V_i and the W1/W2/W3 families. Because it is
    pure in its inputs, the SAME function runs on the point estimate AND on each VRE replicate
    (see :func:`compute_replicate_se`), so the replicate variance captures the cross-table
    covariance that delta-method / independent-MC SEs miss.

      * W1  W_i(r)  = earnings_pc * annuity(r, 20yr) + equity_pc(LTV_MACRO)          [baseline]
      * W2  W2_i(r) = earnings_pc * annuity(r, N_i)  + cap_pc(r_k) + equity_pc(LTV2(year))
      * W3  W3_i(r) = earnings_pc * annuity(r, N_i)  + cap_pc(r_k) + equity_pc(LTV3_i)

    where N_i = expected_working_years, cap_pc = capitalized B19064 per capita, LTV2 is the
    year-matched Fed aggregate LTV, and LTV3_i is the tenure x appreciation imputed tract LTV.
    """
    out = {}
    pop = d["total_population"].replace(0, np.nan)
    agg_owner = d["aggregate_owner_value_usd"]

    # V_i: MEAN owner-occupied home value = aggregate value / owner-occupied units.
    owner_occupied = d["B25003_002E"].replace(0, np.nan)
    out["V_i"] = (agg_owner / owner_occupied).fillna(0.0)

    earnings_pc = (d["aggregate_earnings_usd"] / pop).fillna(0.0)

    # W1 (baseline, UNCHANGED): fixed 20-yr horizon, flat LTV_MACRO, no capital income.
    e1 = equity_pc(d, LTV_MACRO)
    for r in DISCOUNT_RATES:
        out[w_index_colname(r)] = earnings_pc * annuity_factor(r, HUMAN_CAPITAL_YEARS) + e1

    # W2 / W3 shared extensions: tract-varying horizon + capitalized capital income.
    n_i = expected_working_years(d)
    cap_pc = capital_income_pc(d, r_k)
    e2 = equity_pc(d, ltv_macro_w2(year))
    e3 = equity_pc(d, impute_ltv_w3(d, year, hpi=hpi, crosswalk=crosswalk))
    for r in DISCOUNT_RATES:
        hc = earnings_pc * annuity_factor(r, n_i)        # n_i is a Series -> elementwise
        out[w2_index_colname(r)] = hc + cap_pc + e2
        out[w3_index_colname(r)] = hc + cap_pc + e3
    return out


def _compute_wealth_index(d, year, hpi=None, crosswalk=None):
    """Assign V_i and the W1/W2/W3 columns to ``d`` via :func:`wealth_from_inputs`.

    CORE inputs (V_i + W1) are required and raise if absent; the W2/W3 EXTRA inputs only
    warn and degrade (capital term -> 0, N_i -> 0, W3 LTV -> origination LTV w/o revaluation).
    """
    missing_core = [c for c in WEALTH_CORE_COLS if c not in d.columns]
    if missing_core:
        raise KeyError(
            f"compute_acs_indicators: missing CORE wealth inputs {missing_core}. Re-download "
            f"with the strictly-raw download_acs.py (it persists raw ACS components + B20003)."
        )
    missing_extra = [c for c in WEALTH_EXTRA_COLS if c not in d.columns]
    if missing_extra:
        warnings.warn(
            f"compute_acs_indicators: missing W2/W3 inputs {missing_extra[:4]}"
            f"{'...' if len(missing_extra) > 4 else ''}; the affected W2/W3 terms degrade "
            f"(capital income / N_i / W3 LTV). Re-download to add B19064, the B01001 "
            f"working-age brackets, and B25038 owner tenure."
        )
    for col, series in wealth_from_inputs(d, year, hpi=hpi, crosswalk=crosswalk).items():
        d[col] = series
    return d


def load_and_prep(file_path, year, crs: str = ALIGN_CRS, hpi=None, crosswalk=None):
    """Load one ACS vintage, project to a metric CRS, and compute SE / log income.

    No NYC clip and no z-score here: relative scores are normalized per-CBSA later
    (see :func:`add_relative_scores`), after the metro filter is applied. ``hpi``/``crosswalk``
    are forwarded to the wealth index for the W3 tenure x appreciation LTV term.
    """
    print(f"Loading {year} data...")
    gdf = gpd.read_feather(file_path)

    if gdf.crs is None:
        warnings.warn(
            f"CRS is not set for {file_path}. Defaulting to EPSG:4326 -- verify this!"
        )
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Derive every ACS indicator (demographics + the V_i / W1 / W2 / W3 wealth indexes)
    # from the raw columns persisted by download_acs.py. Raises a clear KeyError if the
    # feather predates the strictly-raw download layer (missing CORE components / B20003).
    gdf = compute_acs_indicators(gdf, year, hpi=hpi, crosswalk=crosswalk)

    cols_to_keep = ["geoid", "geometry"] + KEEP_RAW_COLS
    missing = [c for c in cols_to_keep if c not in gdf.columns]
    if missing:
        raise KeyError(
            f"{file_path} is missing columns {missing} after derivation. Re-download "
            f"with the updated download_acs.py (it now persists raw ACS components)."
        )
    gdf = gdf[cols_to_keep].copy()
    gdf["geoid"] = gdf["geoid"].astype(str).str.zfill(11)
    gdf.rename(
        columns={c: f"{c}_{year}" for c in gdf.columns if c != "geometry"},
        inplace=True,
    )

    # MOE -> Standard Error (Census 90% CI -> 1.645).
    gdf[f"SE_{year}"] = gdf[f"{PCI_ERR_COL}_{year}"] / 1.645
    # Natural log of income (delta-method SE for the log).
    income = gdf[f"{PCI_COL}_{year}"].replace(0, np.nan)
    gdf[f"Log_PCI_{year}"] = np.log(income)
    gdf[f"Log_SE_{year}"] = gdf[f"SE_{year}"] / income
    return gdf


def add_relative_scores(gdf, year, group_col: str = "cbsa_code"):
    """Per-CBSA z-score of log income (``Rel_Score``) and its relative SE.

    Replaces the former single city-wide normalization: each metro's mean/std are
    used so scores are comparable within, not across, cities.
    """
    grp = gdf.groupby(group_col)[f"Log_PCI_{year}"]
    mean = grp.transform("mean")
    std = grp.transform("std")
    gdf[f"Rel_Score_{year}"] = (gdf[f"Log_PCI_{year}"] - mean) / std
    gdf[f"Rel_SE_{year}"] = gdf[f"Log_SE_{year}"] / std
    return gdf


def add_wealth_relative_scores(gdf, year, group_col: str = "cbsa_code",
                               wealth_vars=WEALTH_INDEX_VARS):
    """Per-CBSA z-score of log(W) for each wealth index -> ``Rel_Score_{var}_{year}``.

    Mirrors :func:`add_relative_scores` for income, but over the W1/W2/W3 columns
    (``{var}_{year}``). Non-positive / missing W values map to NaN before scoring. The
    matching ``Rel_SE_{var}_{year}`` comes from the VRE replicate variance
    (:func:`compute_replicate_se`), not the delta method.
    """
    for var in wealth_vars:
        col = f"{var}_{year}"
        if col not in gdf.columns:
            continue
        logw = np.log(gdf[col].where(gdf[col] > 0))
        gdf[f"Rel_Score_{var}_{year}"] = _groupwise_z(logw, gdf[group_col])
    return gdf


# ══════════════════════════════════════════════════════════════════════════════
# ③ Spatial alignment + significance (reused machinery)
# ══════════════════════════════════════════════════════════════════════════════

def spatial_align_max_overlap(target_gdf, source_gdf, target_year, source_year):
    """Align historical tracts to base-year tracts by maximum area overlap.

    For each target tract, keep the source tract covering the most of its area.
    Returns a plain DataFrame (no geometry) keyed by ``geoid_{target_year}``.
    """
    intersection = gpd.overlay(
        target_gdf, source_gdf, how="intersection", keep_geom_type=False
    )
    intersection["overlap_area"] = intersection.geometry.area
    intersection = intersection.sort_values(
        by=[f"geoid_{target_year}", "overlap_area"], ascending=[True, False]
    )
    best_match = intersection.drop_duplicates(
        subset=[f"geoid_{target_year}"], keep="first"
    )
    return pd.DataFrame(best_match.drop(columns=["geometry", "overlap_area"]))


def align_year_to_base(base_gdf, source_gdf, base_year, source_year, cbsa_codes):
    """Run :func:`spatial_align_max_overlap` per CBSA to bound overlay cost.

    Both frames carry ``cbsa_code``; ``cbsa_code`` is dropped from the source so it
    does not collide with the base frame's column on the downstream merge.
    """
    print(f"Spatially matching {source_year} tracts to {base_year} boundaries (per metro)...")
    pieces = []
    for code in cbsa_codes:
        tgt = base_gdf.loc[base_gdf["cbsa_code"] == code, [f"geoid_{base_year}", "geometry"]]
        src = source_gdf.loc[source_gdf["cbsa_code"] == code].drop(columns=["cbsa_code"])
        if tgt.empty or src.empty:
            continue
        pieces.append(spatial_align_max_overlap(tgt, src, base_year, source_year))
    if not pieces:
        return pd.DataFrame(columns=[f"geoid_{base_year}"])
    return pd.concat(pieces, ignore_index=True)


_CONFIDENCE_BINS  = [0.0, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
_CONFIDENCE_LABELS = ["p<0.50", "0.50≤p<0.75", "0.75≤p<0.90",
                      "0.90≤p<0.95", "0.95≤p<0.99", "p≥0.99"]


def _confidence_category(p_value: pd.Series) -> pd.Categorical:
    """Map a Series of two-tailed p-values to an ordered confidence-level category.

    Bins are defined on the confidence ``1 - p_value``:
      p<0.50 | 0.50≤p<0.75 | 0.75≤p<0.90 | 0.90≤p<0.95 | 0.95≤p<0.99 | p≥0.99
    where ``p`` here denotes the confidence level (1 - p_value), not the raw p-value.
    """
    confidence = 1.0 - p_value
    return pd.cut(
        confidence,
        bins=_CONFIDENCE_BINS,
        labels=_CONFIDENCE_LABELS,
        right=False,
        include_lowest=True,
    ).astype(pd.CategoricalDtype(categories=_CONFIDENCE_LABELS, ordered=True))


def test_significance(df, year1, year2,
                      score_prefix="Rel_Score", se_prefix="Rel_SE", label=""):
    """Z-test of the difference between two years' relative scores for one indicator.

    Z = |Rel2 - Rel1| / sqrt(SE1^2 + SE2^2); two-tailed p-value; flag at p < 0.01. Reads
    ``{score_prefix}_{year}`` / ``{se_prefix}_{year}`` and writes ``diff/zscore/pvalue/
    significant/significance_level_{label}{year1}_{year2}``. Defaults reproduce the income
    (Y) test exactly (label=""); pass e.g. ``score_prefix='Rel_Score_W2_i_r3pct'``,
    ``label='W2_i_r3pct_'`` for a wealth index.
    """
    print(f"Computing statistical significance ({label or 'income'}): {year1} vs {year2}...")
    diff = df[f"{score_prefix}_{year2}"] - df[f"{score_prefix}_{year1}"]
    se_diff = np.sqrt(df[f"{se_prefix}_{year2}"] ** 2 + df[f"{se_prefix}_{year1}"] ** 2)
    z_score = np.abs(diff) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(z_score))

    df[f"diff_{label}{year1}_{year2}"] = diff
    df[f"zscore_{label}{year1}_{year2}"] = z_score
    df[f"pvalue_{label}{year1}_{year2}"] = p_value
    df[f"significant_{label}{year1}_{year2}"] = p_value < 0.01
    df[f"significance_level_{label}{year1}_{year2}"] = _confidence_category(p_value)
    return df


def _groupwise_z(values, groups):
    """Within-group z-score of ``values`` (a Series) by ``groups`` (aligned array/Series)."""
    s = pd.Series(np.asarray(values, dtype="float64"), index=getattr(values, "index", None))
    g = s.groupby(np.asarray(groups))
    mean = g.transform("mean")
    std = g.transform("std")
    return (s - mean) / std


def compute_replicate_se(reps_by_col, geoid, cbsa, year, hpi=None, crosswalk=None,
                         wealth_vars=WEALTH_INDEX_VARS, n_rep=N_REPLICATES):
    """Replicate-variance SE of each wealth index's per-CBSA relative score, via SDR.

    ``reps_by_col`` maps each raw wealth-input column to an ``(n_tracts, n_rep + 1)`` array
    whose column 0 is the point estimate and 1..n_rep the 80 VRE replicates (uncovered
    components may repeat their point value or carry independent-MC draws). For each replicate
    r, the SAME :func:`wealth_from_inputs` is evaluated and z-scored per CBSA, so covariance
    among all VRE-covered inputs flows through. Returns a DataFrame with ``geoid``,
    ``Rel_Score_{var}_{year}`` (= replicate 0) and ``Rel_SE_{var}_{year}`` =
    sqrt( (4/n_rep) * sum_r (Rel_r - Rel_0)^2 ).
    """
    n = len(geoid)
    rel = {var: np.full((n, n_rep + 1), np.nan) for var in wealth_vars}
    for r in range(n_rep + 1):
        d = pd.DataFrame({col: np.asarray(arr)[:, r] for col, arr in reps_by_col.items()})
        d["geoid"] = np.asarray(geoid)
        idx = wealth_from_inputs(d, year, hpi=hpi, crosswalk=crosswalk)
        for var in wealth_vars:
            w = idx[var]
            logw = np.log(w.where(w > 0))
            rel[var][:, r] = _groupwise_z(logw, cbsa).to_numpy()

    out = {"geoid": np.asarray(geoid)}
    for var in wealth_vars:
        arr = rel[var]
        dev = arr[:, 1:] - arr[:, [0]]
        variance = SDR_SCALE * np.nansum(dev ** 2, axis=1)
        out[f"Rel_Score_{var}_{year}"] = arr[:, 0]
        out[f"Rel_SE_{var}_{year}"] = np.sqrt(variance)
    return pd.DataFrame(out)


def add_wealth_replicate_se(gdf, year, hpi=None, crosswalk=None,
                            wealth_vars=WEALTH_INDEX_VARS):
    """Merge VRE replicate-variance ``Rel_SE_{var}_{year}`` into one year's tract frame.

    Loads that vintage's VRE replicate inputs via :mod:`src.data.download_vre`, recomputes
    each W index on the 80 replicates, and attaches the SDR SE of the per-CBSA relative score
    by joining on ``geoid_{year}`` -- so it rides the same spatial alignment as the point
    relative scores. Returns ``gdf`` unchanged (with a warning) if the VRE store is absent.
    """
    try:
        from src.data.download_vre import load_replicate_inputs
    except Exception:
        load_replicate_inputs = None
    reps = load_replicate_inputs(year) if load_replicate_inputs is not None else None
    if reps is None:
        warnings.warn(
            f"VRE replicate tables not available for {year}; W1/W2/W3 Rel_SE not computed "
            f"(structural-change test for the wealth indexes will be skipped). Run "
            f"src/data/download_vre.py where the Census replicate_estimates host is reachable."
        )
        return gdf
    se_frame = compute_replicate_se(
        reps["inputs"], reps["geoid"], reps["cbsa"], year,
        hpi=hpi, crosswalk=crosswalk, wealth_vars=wealth_vars,
    )
    se_cols = [c for c in se_frame.columns if c.startswith("Rel_SE_")]
    se_frame = se_frame.rename(columns={"geoid": f"geoid_{year}"})
    se_frame[f"geoid_{year}"] = se_frame[f"geoid_{year}"].astype(str).str.zfill(11)
    return gdf.merge(se_frame[[f"geoid_{year}"] + se_cols], on=f"geoid_{year}", how="left")


def add_wealth_structural_change(final_gdf, start_year, end_year,
                                 wealth_vars=WEALTH_INDEX_VARS):
    """First-vs-last structural-change test for each wealth index (mirrors the income test).

    Expects ``Rel_Score_{var}_{start/end}`` (point) and ``Rel_SE_{var}_{start/end}`` (VRE
    replicate SE) already aligned into ``final_gdf``. Indexes lacking the replicate SE (no VRE
    store) are skipped. Adds ``Valid_Structural_Change_{var}`` for each index that has them.
    """
    for var in wealth_vars:
        needed = [f"Rel_Score_{var}_{start_year}", f"Rel_Score_{var}_{end_year}",
                  f"Rel_SE_{var}_{start_year}", f"Rel_SE_{var}_{end_year}"]
        if not all(c in final_gdf.columns for c in needed):
            continue
        final_gdf = test_significance(
            final_gdf, start_year, end_year,
            score_prefix=f"Rel_Score_{var}", se_prefix=f"Rel_SE_{var}", label=f"{var}_",
        )
        final_gdf[f"Valid_Structural_Change_{var}"] = (
            final_gdf[f"significant_{var}_{start_year}_{end_year}"]
        )
    return final_gdf


# ══════════════════════════════════════════════════════════════════════════════
# ④ Spearman robustness tables (task 5) -- CSV only
# ══════════════════════════════════════════════════════════════════════════════

def _spearman(x, y) -> float:
    """Pairwise-complete Spearman rho; NaN if undefined (too few / constant)."""
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    xv, yv = x[mask], y[mask]
    if xv.nunique() < 2 or yv.nunique() < 2:
        return np.nan
    return float(stats.spearmanr(xv, yv).statistic)


def compute_wealth_robustness(frame, year, cbsa_titles: "dict | None" = None) -> pd.DataFrame:
    """Per-metro and global Spearman rho between per-capita income and each wealth var.

    ``frame`` must carry ``cbsa_code``, ``per_capita_income_usd_{year}`` and
    ``{var}_{year}`` for each var in :data:`WEALTH_VARS`. Returns one row per CBSA
    plus a ``GLOBAL`` pooled row; columns ``rho_{var}`` and tract count ``n``.
    """
    cbsa_titles = cbsa_titles or {}
    pci = f"{PCI_COL}_{year}"
    var_cols = {v: f"{v}_{year}" for v in WEALTH_VARS}

    def _row(sub, code, title):
        rec = {"cbsa_code": code, "cbsa_title": title, "n": int(sub[pci].notna().sum())}
        for v, col in var_cols.items():
            rec[f"rho_{v}"] = _spearman(sub[pci], sub[col])
        return rec

    rows = [
        _row(sub, code, cbsa_titles.get(code, ""))
        for code, sub in frame.groupby("cbsa_code")
    ]
    rows.append(_row(frame, "GLOBAL", "GLOBAL (all metros pooled)"))

    cols = ["cbsa_code", "cbsa_title", "n"] + [f"rho_{v}" for v in WEALTH_VARS]
    return pd.DataFrame(rows)[cols].sort_values("cbsa_code").reset_index(drop=True)


def write_robustness_csvs(frame, year, variant, cbsa_titles=None,
                          out_dir=TABLES_DIR / "wealth_robustness"):
    """Write one Spearman CSV per (year, variant) -> ``spearman_{year}_{variant}.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table = compute_wealth_robustness(frame, year, cbsa_titles)
    out_path = out_dir / f"spearman_{year}_{variant}.csv"
    table.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")
    return out_path


def rk_sensitivity_table(d, year, hpi=None, crosswalk=None, grid=CAPITAL_YIELD_GRID):
    """W2/W3 sensitivity to the capital-income rate r_k: Spearman-with-PCI + moments.

    ``d`` is a frame of RAW wealth-input columns plus ``per_capita_income_usd``. For each
    ``r_k`` in ``grid`` and each rho in DISCOUNT_RATES, recomputes W2/W3 and reports the pooled
    Spearman against PCI and the index's mean / sd / skew. The point of the table is to show
    these barely move with r_k (only the additive capital term rescales).
    """
    pci = d[PCI_COL] if PCI_COL in d else d.get(f"{PCI_COL}_{year}")
    rows = []
    for r_k in grid:
        idx = wealth_from_inputs(d, year, hpi=hpi, crosswalk=crosswalk, r_k=r_k)
        for r in DISCOUNT_RATES:
            for fam, name in ((w2_index_colname(r), "W2"), (w3_index_colname(r), "W3")):
                w = idx[fam]
                rows.append({
                    "r_k": r_k, "rho": r, "index": name,
                    "spearman_pci": _spearman(pci, w) if pci is not None else np.nan,
                    "mean": float(w.mean()), "sd": float(w.std()),
                    "skew": float(stats.skew(w.dropna())) if w.notna().sum() > 2 else np.nan,
                })
    return pd.DataFrame(rows)


def write_rk_sensitivity_csv(year, hpi=None, crosswalk=None, grid=CAPITAL_YIELD_GRID,
                             out_dir=TABLES_DIR / "wealth_robustness"):
    """Re-read year ``year``'s raw feather, build the r_k sensitivity table, write CSV.

    Reads the raw ACS components (needed to recompute W2/W3 at alternative r_k, since the
    panel keeps only the headline index columns). Skipped with a warning if the feather is
    unavailable. -> ``rk_sensitivity_{year}.csv``.
    """
    path = ACS_ROOT_DIR / str(year) / f"us_tracts_acs5_{year}.feather"
    if not path.exists():
        warnings.warn(f"r_k sensitivity skipped for {year}: {path} not found.")
        return None
    d = pd.DataFrame(gpd.read_feather(path).drop(columns="geometry", errors="ignore"))
    out_dir.mkdir(parents=True, exist_ok=True)
    table = rk_sensitivity_table(d, year, hpi=hpi, crosswalk=crosswalk, grid=grid)
    out_path = out_dir / f"rk_sensitivity_{year}.csv"
    table.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ Structural-change share table (task 6) -- CSV only
# ══════════════════════════════════════════════════════════════════════════════

def valid_sc_shares_table(final_gdf, cbsa_titles=None):
    """Per-CBSA share of tracts flagged Valid_Structural_Change, for income and each W index.

    Returns one row per CBSA plus a NATIONAL pooled row. Columns:
      cbsa_code | cbsa_title | n_tracts | share_income | share_{var} x 12

    Share denominator is all tracts in the CBSA (tracts with insufficient data to test
    are treated as non-flagged, matching the flag-is-False convention in test_significance).
    Wealth columns are omitted when no Valid_Structural_Change_{var} exists in the panel
    (e.g. if the VRE store was absent and those tests were skipped).
    """
    cbsa_titles = cbsa_titles or {}
    sc_income  = "Valid_Structural_Change"
    sc_w_cols  = [f"Valid_Structural_Change_{v}" for v in WEALTH_INDEX_VARS
                  if f"Valid_Structural_Change_{v}" in final_gdf.columns]

    def _row(sub, code, title):
        n = len(sub)
        rec = {"cbsa_code": code, "cbsa_title": title, "n_tracts": n}
        if sc_income in sub.columns:
            rec["share_income"] = round(float(sub[sc_income].sum()) / n, 4) if n else np.nan
        for col in sc_w_cols:
            var = col.replace("Valid_Structural_Change_", "")
            rec[f"share_{var}"] = round(float(sub[col].sum()) / n, 4) if n else np.nan
        return rec

    rows = [
        _row(sub, code, cbsa_titles.get(code, ""))
        for code, sub in final_gdf.groupby("cbsa_code")
    ]
    rows.append(_row(final_gdf, "NATIONAL", "NATIONAL (all metros pooled)"))

    share_w_cols = [f"share_{v}" for v in WEALTH_INDEX_VARS
                    if f"Valid_Structural_Change_{v}" in final_gdf.columns]
    cols = ["cbsa_code", "cbsa_title", "n_tracts", "share_income"] + share_w_cols
    return pd.DataFrame(rows)[cols].sort_values("cbsa_code").reset_index(drop=True)


def significance_level_counts_table(final_gdf, start_year, end_year,
                                    wealth_vars=WEALTH_INDEX_VARS) -> pd.DataFrame:
    """Count of tracts in each confidence-level bracket, per indicator.

    Rows = confidence brackets (ordered low -> high). Columns = income + each wealth index
    that has a ``significance_level_*`` column in ``final_gdf``. Also adds a ``%`` column
    for each indicator (share of total tracts with non-null significance data).
    """
    income_col = f"significance_level_{start_year}_{end_year}"
    w_cols = {
        var: f"significance_level_{var}_{start_year}_{end_year}"
        for var in wealth_vars
        if f"significance_level_{var}_{start_year}_{end_year}" in final_gdf.columns
    }

    rows = []
    for label in _CONFIDENCE_LABELS:
        rec = {"significance_level": label}
        for name, col in ([("income", income_col)] if income_col in final_gdf.columns else []) + list(w_cols.items()):
            rec[f"n_{name}"] = int((final_gdf[col] == label).sum())
        rows.append(rec)

    table = pd.DataFrame(rows)

    # Add % columns
    for name, col in ([("income", income_col)] if income_col in final_gdf.columns else []) + list(w_cols.items()):
        total = final_gdf[col].notna().sum()
        if total > 0:
            table[f"pct_{name}"] = (table[f"n_{name}"] / total * 100).round(2)

    return table


def write_significance_level_counts_csv(final_gdf, start_year, end_year,
                                        wealth_vars=WEALTH_INDEX_VARS,
                                        out_dir=TABLES_DIR / "structural_change"):
    """Write the significance-level bracket count table.

    -> ``structural_change/significance_level_counts_{start}_{end}.csv``
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    table = significance_level_counts_table(final_gdf, start_year, end_year, wealth_vars)
    out_path = out_dir / f"significance_level_counts_{start_year}_{end_year}.csv"
    table.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")
    print(table.to_string(index=False))
    return out_path


def write_valid_sc_shares_csv(final_gdf, cbsa_titles=None,
                               out_dir=TABLES_DIR / "structural_change"):
    """Write the per-CBSA Valid_Structural_Change share table.

    -> ``structural_change/valid_sc_shares_by_metro.csv``
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    table = valid_sc_shares_table(final_gdf, cbsa_titles)
    out_path = out_dir / "valid_sc_shares_by_metro.csv"
    table.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# ⑥ Panel orchestration
# ══════════════════════════════════════════════════════════════════════════════

def check_alignment_fidelity(gdfs, final_gdf, years, base_year,
                             value_col: str = PCI_COL,
                             out_dir=TABLES_DIR / "alignment_fidelity"):
    """Quantify the measurement error introduced by aligning each year to base tracts.

    Every panel row is a *base-year* tract with a canonical ``geoid_{base_year}``. If
    that same geoid was published in ``year``'s own vintage (``gdfs[year]``), the value
    the panel SHOULD carry is that tract's own ``year`` value. The aligned panel instead
    carries the value of whichever ``year`` tract maximally overlaps the base geometry
    (see :func:`spatial_align_max_overlap`). When boundaries are stable the two coincide
    exactly; any difference is alignment-induced measurement error.

    Per-year categories over the base-tract universe in ``final_gdf`` (mutually
    exclusive, summing to ``len(final_gdf)``):
      * exact        -- base geoid exists in ``year`` AND the aligned value equals that
                        tract's own value (perfect match, no error introduced).
      * mismatch     -- base geoid exists in ``year`` but the aligned value came from a
                        *different* overlapping tract (genuine alignment error).
      * approximated -- base geoid absent in ``year`` (real boundary change, e.g. the
                        2010<->2020 vintage break); value necessarily borrowed from an
                        overlapping neighbour, so no same-geoid ground truth exists.
      * missing      -- no overlapping source tract was found (aligned value is NaN).

    Writes a one-row-per-year CSV and returns it. ``base_year`` should be ~100% exact
    (final_gdf starts as a copy of that vintage) -- a useful self-check.
    """
    n = len(final_gdf)
    base_geoid = final_gdf[f"geoid_{base_year}"]
    rows = []
    print("\nChecking alignment fidelity (own vintage vs aligned panel)...")
    for year in years:
        raw = gdfs[year].drop_duplicates(subset=[f"geoid_{year}"])
        raw_lookup = raw.set_index(f"geoid_{year}")[f"{value_col}_{year}"]

        raw_val = base_geoid.map(raw_lookup)             # this year's value for the SAME geoid
        aligned_val = final_gdf[f"{value_col}_{year}"]   # value the alignment actually assigned

        present = base_geoid.isin(raw_lookup.index)      # base tract existed in this vintage
        aligned_present = aligned_val.notna()
        eq = np.isclose(raw_val.to_numpy(dtype="float64"),
                        aligned_val.to_numpy(dtype="float64"),
                        rtol=1e-6, atol=1e-6, equal_nan=True)

        exact        = present & aligned_present & eq
        mismatch     = present & aligned_present & ~eq
        approximated = (~present) & aligned_present
        missing      = ~aligned_present

        if mismatch.any():
            rv, av = raw_val[mismatch], aligned_val[mismatch]
            pct = (av - rv).abs() / rv.abs().replace(0, np.nan)
            mean_abs_pct = float(pct.mean() * 100)       # NaN-safe mean over mismatched tracts
        else:
            mean_abs_pct = 0.0

        rows.append({
            "year": year,
            "n_base_tracts": n,
            "exact": int(exact.sum()),
            "mismatch": int(mismatch.sum()),
            "approximated": int(approximated.sum()),
            "missing": int(missing.sum()),
            "pct_exact": round(100 * exact.sum() / n, 3),
            "pct_mismatch": round(100 * mismatch.sum() / n, 3),
            "pct_approximated": round(100 * approximated.sum() / n, 3),
            "pct_missing": round(100 * missing.sum() / n, 3),
            "mismatch_mean_abs_pct_err": round(mean_abs_pct, 3),
        })
        print(f"  {year}: exact {100*exact.sum()/n:5.1f}% | "
              f"mismatch {100*mismatch.sum()/n:5.1f}% (mean |err| {mean_abs_pct:4.1f}%) | "
              f"approx {100*approximated.sum()/n:5.1f}% | missing {100*missing.sum()/n:5.1f}%")

    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"alignment_fidelity_{years[0]}_{years[-1]}.csv"
    table.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")
    return table


def process_panel(years: list[int] = PANEL_YEARS, base_year: int = BASE_YEAR):
    """Build the US-metros ACS panel and the Spearman robustness CSVs."""
    years = sorted(years)
    start_year, end_year = years[0], years[-1]

    # Income (Y) structural change spans the full panel (delta-method SEs exist every year), so it
    # uses start_year..end_year. The wealth indexes instead need the VRE replicate SE at BOTH
    # endpoints, and the VRE flat files only begin in VRE_FIRST_YEAR (2014) -- so the W test starts
    # at the first panel year with VRE coverage. (A start_year of 2011 has no replicate estimates,
    # which is why it previously emitted no Valid_Structural_Change_W* columns.) Lazy import keeps
    # the download_vre <-> process_acs dependency one-directional at module load.
    from src.data.download_vre import VRE_FIRST_YEAR
    wealth_start_year = max(start_year, VRE_FIRST_YEAR)

    print(f"\n{'='*60}\nUS metros panel | years {years} | base {base_year}\n{'='*60}")
    print(f"Income structural change: {start_year} vs {end_year} | "
          f"wealth structural change: {wealth_start_year} vs {end_year} (VRE >= {VRE_FIRST_YEAR})")

    crosswalk = load_county_cbsa_crosswalk()
    hpi = load_fhfa_hpi()   # {'county','metro'}; absent files -> W3 appreciation falls back to 1.0
    metros = get_large_metros(base_year, MIN_METRO_POP, crosswalk)
    large_codes = list(metros["cbsa_code"])
    large_set = set(large_codes)
    titles = dict(zip(metros["cbsa_code"], metros["cbsa_title"]))

    # 1. Load, tag to CBSA, filter to large metros, score per-CBSA (per year).
    gdfs = {}
    for year in years:
        path = ACS_ROOT_DIR / str(year) / f"us_tracts_acs5_{year}.feather"
        gdf = load_and_prep(path, year, hpi=hpi, crosswalk=crosswalk)
        gdf = tag_cbsa(gdf, f"geoid_{year}", crosswalk, year=year)
        gdf = gdf[gdf["cbsa_code"].isin(large_set)].copy()
        gdf = add_relative_scores(gdf, year)
        gdf = add_wealth_relative_scores(gdf, year)   # per-CBSA z-scores of log W1/W2/W3
        if year in (wealth_start_year, end_year):
            # Replicate-variance (VRE) SEs of the W relative scores -- only the two years the
            # wealth first-vs-last test needs (wealth_start_year..end_year, both VRE-covered).
            # Merged here so they ride the spatial alignment below.
            gdf = add_wealth_replicate_se(gdf, year, hpi=hpi, crosswalk=crosswalk)
        gdfs[year] = gdf
        print(f"  {year}: {len(gdf):,} tracts in {gdf['cbsa_code'].nunique()} metros.")

    # 2. Align every non-base year to the base-year tract boundaries (per metro).
    base_gdf = gdfs[base_year]
    final_gdf = base_gdf.copy()
    for year in years:
        if year == base_year:
            continue
        matched = align_year_to_base(base_gdf, gdfs[year], base_year, year, large_codes)
        final_gdf = final_gdf.merge(matched, on=f"geoid_{base_year}", how="left")

    # 3. Valid_Structural_Change = single first-vs-last significance test (income / Y).
    final_gdf = test_significance(final_gdf, start_year, end_year)
    final_gdf["Valid_Structural_Change"] = final_gdf[f"significant_{start_year}_{end_year}"]

    # 3b. Same first-vs-last test for the wealth indexes W1/W2/W3, using the replicate-variance
    #     (VRE) SEs aligned in step 1. Runs over wealth_start_year..end_year (VRE >= 2014), not the
    #     income start_year. Skipped per-index if the VRE store was absent -- the W relative scores
    #     are still carried into the panel.
    final_gdf = add_wealth_structural_change(final_gdf, wealth_start_year, end_year)

    # 4. Training labels = per-CBSA relative scores for every year.
    for year in years:
        final_gdf[f"Training_Label_{year}"] = final_gdf[f"Rel_Score_{year}"]

    n_valid = int(final_gdf["Valid_Structural_Change"].sum())
    print(f"\nValid structural change ({start_year} vs {end_year}): "
          f"{n_valid} of {len(final_gdf)} tracts.")

    # 5. Robustness Spearman CSVs -- raw (pre-alignment) and aligned (post-merge) -- and the
    #    r_k sensitivity table (W2/W3 vs PCI Spearman + moments across CAPITAL_YIELD_GRID).
    print("\nWriting wealth-robustness Spearman tables...")
    for year in years:
        write_robustness_csvs(gdfs[year], year, "raw", titles)
        write_robustness_csvs(final_gdf, year, "aligned", titles)
        write_rk_sensitivity_csv(year, hpi=hpi, crosswalk=crosswalk)

    # 6. Save the panel.
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_name = PROCESSED_DATA_DIR / f"us_metros_panel_{start_year}_{end_year}.feather"
    final_gdf.to_feather(output_name)
    print(f"\nPanel saved to {output_name}")

    # 7. Diagnostics on the single-period change.
    diff_col = f"diff_{start_year}_{end_year}"
    valid_mask = final_gdf["Valid_Structural_Change"] == True  # noqa: E712
    print(f"Mean |Rel_Score change| (valid):   "
          f"{final_gdf.loc[valid_mask, diff_col].abs().mean():.4f}")
    print(f"Mean |Rel_Score change| (invalid): "
          f"{final_gdf.loc[~valid_mask, diff_col].abs().mean():.4f}")

    # 8. Alignment fidelity: does max-overlap matching introduce measurement error?
    check_alignment_fidelity(gdfs, final_gdf, years, base_year)

    # 9. Per-CBSA share of tracts flagged Valid_Structural_Change (income + all W indexes).
    print("\nWriting structural-change share table...")
    write_valid_sc_shares_csv(final_gdf, titles)

    # 10. Significance-level bracket counts (n tracts per confidence bin x indicator).
    print("\nWriting significance-level bracket counts...")
    write_significance_level_counts_csv(final_gdf, start_year, end_year)

    return final_gdf

if __name__ == "__main__":
    process_panel(years=PANEL_YEARS, base_year=BASE_YEAR)
