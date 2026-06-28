"""
Fetch + persist the Census **Variance Replicate Estimate (VRE)** tables, and assemble the
80-replicate input arrays the wealth-index variance computation consumes.

Why this exists
---------------
A delta-method or independent-Monte-Carlo standard error for a *multi-table* index (W1/W2/W3
combine B20003, B25082, B25081, B01001, B19064, B25038) is wrong: those tables are correlated
and both methods ignore the cross-table covariance. The VRE tables ship, for each estimate
cell, the point estimate **plus 80 successive-difference replicates that share a common set of
replicate weights**. Recomputing the index on each replicate and taking the replicate variance
(``process_acs.compute_replicate_se``) captures the covariance automatically:

    Var(X) = (4 / 80) * sum_{r=1..80} (X_r - X_0)^2          # successive-difference replication

Access reality
--------------
VRE tables are **flat files** (NOT served by the Census API), published under
``www2.census.gov/programs-surveys/acs/replicate_estimates/{end_year}/data/5-year/``. Tract /
block-group files are named by **table ID + state FIPS**. Each row is a geography; columns are
``GEOID``, the estimate, the MOE, and ``VAR_REP1..VAR_REP80``. The VRE program covers only a
SUBSET of detailed tables, and the covered set / bracket layout has shifted across vintages.

Performance / robustness design
-------------------------------
The structural-change test in :mod:`src.data.process_acs` uses ONLY the first and last panel
years, so :func:`download_vre_tables` defaults to those two vintages (not all 13). Within that:
  * each (table, state) is fetched with a **timeout** and written to its **own parquet**, so
    files appear immediately and memory stays bounded;
  * an existing output is **skipped** -> the run is **resumable** after Ctrl-C / a dropped
    connection;
  * fetches run in a **thread pool** (network-bound).
Run :func:`probe_one` (``--test``) first to VERIFY the URL/path before the bulk pull -- the URL
pattern below is the documented layout but could not be checked from the build sandbox (egress
blocked). MUST run where ``www2.census.gov`` is reachable (the data / GPU box).
"""

from __future__ import annotations

import io
import sys
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import ACS_ROOT_DIR

# ── Where the normalized replicate store is persisted ───────────────────────────
VRE_ROOT = ACS_ROOT_DIR / "vre"
N_REPLICATES = 80
VAR_REP_COLS = [f"VAR_REP{i}" for i in range(1, N_REPLICATES + 1)]

# Census VRE flat-file base (tract summary level 140/).
# 2014–2018: files are gzip-compressed (.csv.gz); 2019+: zip-archived (.csv.zip).
_VRE_BASE = (
    "https://www2.census.gov/programs-surveys/acs/replicate_estimates/"
    "{end_year}/data/5-year/140/{table}_{state}"
)
_VRE_EXT_CUTOVER = 2019  # first year that uses .csv.zip instead of .csv.gz


def _vre_url(year: int, table: str, state: str) -> str:
    ext = ".csv.zip" if year >= _VRE_EXT_CUTOVER else ".csv.gz"
    return _VRE_BASE.format(end_year=year, table=table, state=state) + ext

# 50 states + DC (11) + Puerto Rico (72). Invalid FIPS simply 404 and are skipped.
ALL_STATES = [
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15", "16", "17",
    "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
    "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
    "47", "48", "49", "50", "51", "53", "54", "55", "56", "72",
]

# ── Manifest: which raw ACS cells each wealth input column is built from ─────────
# Maps the column name wealth_from_inputs() expects -> the source VRE table + cell line.
WEALTH_INPUT_CELLS: dict[str, str] = {
    "aggregate_earnings_usd":        "B20003_001",
    "aggregate_owner_value_usd":     "B25082_001",
    "aggregate_capital_income_usd":  "B19064_001",
    "total_population":              "B01001_001",
    "B25003_001E": "B25003_001", "B25003_002E": "B25003_002",
    "B25081_001E": "B25081_001", "B25081_002E": "B25081_002",
    "B25038_002E": "B25038_002",
    **{f"B25038_{i:03d}E": f"B25038_{i:03d}" for i in range(3, 9)},
    # B01001 working-age brackets (male offsets 6-19, female 30-43 -> midpoints 16..63).
    **{f"B01001_{i:03d}E": f"B01001_{i:03d}" for i in range(6, 20)},
    **{f"B01001_{i:03d}E": f"B01001_{i:03d}" for i in range(30, 44)},
}

# The detailed tables the manifest draws from (used by the downloader / coverage check).
VRE_TABLES = sorted({cell.split("_")[0] for cell in WEALTH_INPUT_CELLS.values()})


# VRE flat files are only published starting with the 2014 5-year ACS. PUBLIC so process_acs
# can clamp the wealth structural-change start year to the first VRE-covered vintage (the panel
# itself starts in 2011, which has no replicate estimates) -- one shared source of truth.
VRE_FIRST_YEAR = 2014

def _structural_change_years():
    """The two VRE vintages the W structural-change test needs (panel first/last, ≥ 2014)."""
    from src.data.process_acs import PANEL_YEARS
    first = max(PANEL_YEARS[0], VRE_FIRST_YEAR)
    return (first, PANEL_YEARS[-1])


def _fetch_csv(url, timeout=(10, 120), retries=4):
    """GET one VRE flat file into a DataFrame, with a connect/read timeout and backoff.

    Returns the parsed DataFrame, ``None`` for a 404 (uncovered table/state), and raises on
    persistent non-404 errors. ``timeout`` is (connect, read) seconds -- the read timeout is
    per network read, so large-but-flowing downloads do not trip it, but a stalled server does.
    """
    import requests  # local import: only needed where the download actually runs

    delay = 2
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            # Some state files (NM/35, PR/72) use Latin-1; apply to both formats.
            data = io.BytesIO(resp.content)
            if url.endswith(".csv.zip"):
                with zipfile.ZipFile(data) as zf:
                    with zf.open(zf.namelist()[0]) as f:
                        return pd.read_csv(f, dtype=str, encoding="latin-1")
            else:  # .csv.gz
                import gzip
                with gzip.open(data) as f:
                    return pd.read_csv(f, dtype=str, encoding="latin-1")
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return None


def _normalize_vre_file(raw: pd.DataFrame, table: str) -> pd.DataFrame:
    """Reduce one raw VRE flat file to ``[geoid, cell, estimate, moe, VAR_REP1..80]`` (long
    over cells). Defensive about column casing; the cell id is ``{table}_{line}``."""
    # Drop blank/header rows that have no GEOID (2 metadata rows at the top of each file).
    geo_col_raw = next((c for c in raw.columns if c.lower() in ("geoid", "geo_id")), raw.columns[0])
    raw = raw[raw[geo_col_raw].notna()].reset_index(drop=True)

    cols = {c.lower(): c for c in raw.columns}
    geo = cols.get("geoid") or cols.get("geo_id") or list(raw.columns)[0]
    # "ORDER" is the current column name for the cell line number (older files used "LINE").
    line = cols.get("order") or cols.get("line") or cols.get("orderid") or cols.get("line_number")
    est = cols.get("estimate") or cols.get("estimate_e") or cols.get("est")
    moe = cols.get("moe") or cols.get("margin_of_error")
    rep_map = {f"var_rep{i}": None for i in range(1, N_REPLICATES + 1)}
    for c in raw.columns:
        if c.lower() in rep_map:
            rep_map[c.lower()] = c
    out = pd.DataFrame({
        "geoid": raw[geo].astype(str).str[-11:].str.zfill(11),
        "cell": table + "_" + raw[line].astype(str).str.zfill(3),
        "estimate": pd.to_numeric(raw[est], errors="coerce"),
        "moe": pd.to_numeric(raw[moe], errors="coerce") if moe else np.nan,
    })
    for i in range(1, N_REPLICATES + 1):
        src = rep_map[f"var_rep{i}"]
        out[f"VAR_REP{i}"] = pd.to_numeric(raw[src], errors="coerce") if src else np.nan
    return out


def _out_path(out_root, year, table, state):
    return out_root / str(year) / f"{table}_{state}.parquet"


def _download_one(year, table, state, out_root, timeout):
    """Fetch + persist a single (table, state) file. Returns a status string.

    Skips (``'skip'``) if the parquet already exists -> resumable. Writes atomically (tmp +
    rename) so an interrupt never leaves a half-written parquet. ``'missing'`` = 404 (uncovered).
    """
    out = _out_path(out_root, year, table, state)
    if out.exists():
        return "skip"
    url = _vre_url(year, table, state)
    raw = _fetch_csv(url, timeout=timeout)
    if raw is None:
        return "missing"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    _normalize_vre_file(raw, table).to_parquet(tmp)
    tmp.replace(out)
    return "ok"


def download_vre_tables(years=None, states=ALL_STATES, tables=VRE_TABLES,
                        out_root: Path = VRE_ROOT, workers=4, timeout=(10, 120)):
    """Fetch + persist VRE flat files for ``tables`` x ``states`` x ``years`` (tract level).

    Writes one parquet per (table, state) to ``{out_root}/{year}/{table}_{state}.parquet`` and a
    ``coverage_{year}.csv`` per vintage. Resumable (skips existing), streamed (bounded memory),
    and thread-pooled. ``years`` defaults to the two structural-change vintages. MUST run where
    ``www2.census.gov`` egress is allowed (not the build sandbox).
    """
    years = years if years is not None else _structural_change_years()
    out_root.mkdir(parents=True, exist_ok=True)
    for year in years:
        tasks = [(table, state) for table in tables for state in states]
        counts = {"ok": 0, "skip": 0, "missing": 0, "error": 0}
        present = {t: False for t in tables}
        print(f"VRE {year}: {len(tasks)} (table,state) files -> {out_root / str(year)}")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_download_one, year, t, s, out_root, timeout): (t, s)
                    for t, s in tasks}
            done = 0
            for fut in as_completed(futs):
                t, s = futs[fut]
                done += 1
                try:
                    status = fut.result()
                except Exception as exc:  # noqa: BLE001 - keep going, report at end
                    status = "error"
                    warnings.warn(f"VRE {year} {t}_{s}: {exc}")
                counts[status] += 1
                if status in ("ok", "skip"):
                    present[t] = True
                if done % 25 == 0 or done == len(tasks):
                    print(f"  {year}: {done}/{len(tasks)}  ok={counts['ok']} "
                          f"skip={counts['skip']} missing={counts['missing']} err={counts['error']}")
        (out_root / str(year)).mkdir(parents=True, exist_ok=True)
        pd.Series(present, name="has_vre").to_csv(out_root / str(year) / f"coverage_{year}.csv")
        covered = sorted(t for t, p in present.items() if p)
        print(f"VRE {year} done: {counts}  covered tables: {covered}")
    return out_root


def probe_one(year=None, table="B25003", state="36", out_root: Path = VRE_ROOT):
    """Fetch a SINGLE VRE file and print the URL, HTTP result, shape and columns.

    Run this first to verify the URL/path is correct before any bulk pull (the path is the
    documented layout but was not verifiable from the build sandbox).
    """
    year = year or _structural_change_years()[-1]
    url = _vre_url(year, table, state)
    print(f"PROBE {url}")
    raw = _fetch_csv(url)
    if raw is None:
        print("  -> 404 / not found (table likely uncovered for this vintage, or wrong path)")
        return None
    print(f"  -> OK  shape={raw.shape}")
    print(f"  columns[:8]={list(raw.columns[:8])}")
    norm = _normalize_vre_file(raw, table)
    print(f"  normalized cells: {sorted(norm['cell'].unique())[:6]} ...  rows={len(norm)}")
    return raw


def _attach_uncovered_point_estimates(inputs: dict, geoid: np.ndarray, year: int):
    """Fill manifest inputs absent from VRE with their ACS point estimate (constant block).

    For every ``WEALTH_INPUT_CELLS`` column not already in ``inputs`` (i.e. tables the VRE
    program does not publish), read the point estimate from the raw vintage feather, align it
    to ``geoid``, and write an ``(n, 81)`` array that repeats the point value across all slots
    -- so it contributes to the index level on every replicate but adds no variance. Mutates
    ``inputs`` in place. The manifest keys ARE the raw feather column names (download_acs.py
    persists B25082_001E -> aggregate_owner_value_usd, etc.).
    """
    uncovered = [c for c in WEALTH_INPUT_CELLS if c not in inputs]
    if not uncovered:
        return
    feather = ACS_ROOT_DIR / str(year) / f"us_tracts_acs5_{year}.feather"
    if not feather.exists():
        raise FileNotFoundError(
            f"VRE replicate assembly needs the raw ACS feather for {year} to supply the "
            f"non-VRE inputs {uncovered}, but {feather} is missing. Run download_acs.py first."
        )
    raw = pd.read_feather(feather, columns=["geoid", *uncovered])
    raw["geoid"] = raw["geoid"].astype(str).str.zfill(11)
    raw = raw.drop_duplicates("geoid").set_index("geoid").reindex(geoid)
    for col in uncovered:
        pt = np.nan_to_num(pd.to_numeric(raw[col], errors="coerce").to_numpy(), nan=0.0)
        inputs[col] = np.repeat(pt[:, None], N_REPLICATES + 1, axis=1)


def load_replicate_inputs(year, out_root: Path = VRE_ROOT, rng_seed: int = 0):
    """Assemble the 80-replicate input arrays for the wealth index, or ``None`` if absent.

    Reads ALL per-(table,state) parquets under ``{out_root}/{year}/`` (glob). Returns
    ``{"geoid": ndarray, "cbsa": ndarray, "inputs": {col: (n, 81) ndarray}}`` where column 0 is
    the point estimate and 1..80 the replicates. Three input regimes:
      * **VRE-covered** cells (B20003, B01001, B25003, B25038) use the real 80 replicates.
      * cells present in VRE but with no replicates fall back to independent-MC draws from
        their MOE (still contribute variance, no covariance).
      * cells the VRE program does NOT publish at all (B25082, B19064, B25081) are pinned at
        their ACS POINT estimate -- read from the raw ``us_tracts_acs5_{year}.feather`` and
        tiled constant across all 81 slots. They shift the nonlinear per-CBSA z-score (so the
        point value is needed) but carry zero variance (no MOE is persisted to draw from).
    Returns ``None`` (callers skip gracefully) when no VRE files exist for ``year``.
    """
    files = sorted((out_root / str(year)).glob("*.parquet"))
    if not files:
        return None
    vre = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)

    geoid = np.sort(vre["geoid"].unique())
    n = len(geoid)
    pos = {g: i for i, g in enumerate(geoid)}
    rng = np.random.default_rng(rng_seed)

    inputs: dict[str, np.ndarray] = {}
    for col, cell in WEALTH_INPUT_CELLS.items():
        sub = vre[vre["cell"] == cell]
        if sub.empty:
            continue  # cell entirely absent -> leave it out (wealth helpers degrade)
        arr = np.zeros((n, N_REPLICATES + 1))
        rows = sub["geoid"].map(pos).to_numpy()
        arr[rows, 0] = sub["estimate"].to_numpy()
        rep_block = sub[VAR_REP_COLS].to_numpy()
        if np.isfinite(rep_block).any():
            arr[rows, 1:] = rep_block                       # covered: real replicates
        else:                                               # uncovered: independent-MC fallback
            se = np.where(np.isfinite(sub["moe"].to_numpy()), sub["moe"].to_numpy() / 1.645, 0.0)
            arr[rows, 1:] = arr[rows, [0]] + rng.normal(size=(len(rows), N_REPLICATES)) * se[:, None]
        inputs[col] = arr

    # Inputs the VRE program does not publish (B25082, B19064, B25081) still enter
    # wealth_from_inputs and shift the nonlinear per-CBSA z-score, so their point estimate
    # must be supplied -- read from the raw ACS feather and pinned constant across all 81
    # replicate slots (zero variance; no MOE is persisted for these tables to draw from).
    _attach_uncovered_point_estimates(inputs, geoid, year)

    from src.data.process_acs import load_county_cbsa_crosswalk, _county_to_cbsa_map
    xwalk = _county_to_cbsa_map(load_county_cbsa_crosswalk())
    cbsa = np.array([xwalk.get(g[:5]) for g in geoid], dtype=object)
    return {"geoid": geoid, "cbsa": cbsa, "inputs": inputs}


if __name__ == "__main__":
    # `python src/data/download_vre.py --test`  -> probe one file first (verify the URL).
    # `python src/data/download_vre.py`          -> full pull (all states, the 2 SC years).
    if "--test" in sys.argv:
        probe_one()
    else:
        download_vre_tables()
