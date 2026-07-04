"""End-to-end consistency report for the FINAL training data (ACS panel + buildings).

Runs AFTER ``run_data_pipeline`` (or the two stages individually) and answers a single
question: *is the data that training will actually consume internally consistent?* It is
deliberately broader than ``verify_panel`` / ``verify_buildings_index`` -- as well as the
structural PASS/FAIL schema checks it prints **summary statistics per city and globally**
so the numbers can be eyeballed:

  * ACS panel -- distribution of the main indicator (income ``Rel_Score`` by default, any
    :mod:`src.data.indicators` token via ``--indicator``): count / non-null / mean / std /
    min / p5 / p25 / median / p75 / p95 / max, both globally per year and per CBSA at the
    base year. Because the score is a within-CBSA z-score, every city-year should read
    mean ~ 0, std ~ 1 -- a broken join or mis-scoring shows up immediately here.
  * Buildings index -- building counts per CBSA and per tract (total, median/p5 per tract,
    tracts with no buildings), plus the global cross-state duplicate-id rate.

Structural checks (each prints PASS/WARN/FAIL; any FAIL exits non-zero):
  panel : geoid key unique/11-char; cbsa_code non-null; score + valid_change + training-label
          columns for every indicator token and year; per-CBSA z-moments ~ (0, 1); main
          indicator finite and within a sane |z| bound; geometry present and non-empty.
  bldgs : schema/dtypes; building_id non-negative and unpacking back to (cx, cy); coordinates
          inside CONUS EPSG:5070 bounds; every tract_id present in the panel. Cross-state
          border duplicates are reported as a WARN (the builder only dedupes within a state).

    python -m src.data.verify_final_data                          # panel + buildings
    python -m src.data.verify_final_data --indicator W2_r5         # a wealth token instead
    python -m src.data.verify_final_data --skip-buildings          # panel only (fast)

CSVs (per-city indicator stats + per-CBSA building coverage) land under
``results/tables/final_data_summary/`` for the record.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import indicators as ind
from src.data.build_buildings_index import DEFAULT_PANEL, unpack_building_id
from src.data.process_acs import BASE_YEAR, PANEL_YEARS
from src.utils.paths import PROCESSED_DATA_DIR, TABLES_DIR

DEFAULT_INDEX_DIR = PROCESSED_DATA_DIR / "buildings_index"
DEFAULT_OUT_DIR = TABLES_DIR / "final_data_summary"

# Quantiles reported for every distribution (labels reused as column names).
_QUANTILES = {"p5": 0.05, "p25": 0.25, "median": 0.50, "p75": 0.75, "p95": 0.95}

# Within-CBSA z-score moment tolerances (mirror verify_panel; alignment perturbs
# non-base years slightly, so the bound covers that, not a broken z-scoring).
Z_TOL_MEAN = 0.02          # |mean| of per-CBSA z-scores
Z_TOL_STD = 0.05           # |std - 1| of per-CBSA z-scores
MIN_TRACTS_FOR_Z = 30      # skip the moment check for tiny CBSAs (noisy)
MAX_ABS_Z = 25.0           # a real within-CBSA z-score never gets this extreme

# Generous CONUS bounds in EPSG:5070 meters (mirror verify_buildings_index).
X_RANGE = (-3.0e6, 3.0e6)
Y_RANGE = (-0.5e6, 3.5e6)
# State FIPS prefixes that project badly in EPSG:5070 (CONUS Albers): AK, HI, PR. Their
# building centroids fall well outside the CONUS bounds by design (build_buildings_index
# keeps them with a warning), so out-of-bounds coords from these are a WARN, not a FAIL.
NON_CONUS_FIPS = {"02", "15", "72"}
# Cross-state border duplicates are expected; only an implausibly high rate is a problem.
DUP_ID_WARN_FRAC = 0.02


class Checker:
    """Collects PASS/WARN/FAIL lines; only FAIL is fatal (accumulated in ``failures``)."""

    def __init__(self):
        self.failures = []

    def check(self, name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)

    def warn(self, name, ok, detail=""):
        print(f"[{'PASS' if ok else 'WARN'}] {name}" + (f" — {detail}" if detail else ""))


def _banner(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ══════════════════════════════════════════════════════════════════════════════

def describe_series(s: pd.Series) -> dict:
    """Count / non-null / mean / std / min / quantiles / max for one numeric series."""
    x = pd.to_numeric(s, errors="coerce")
    valid = x.dropna()
    rec = {
        "n": int(len(x)),
        "n_nonnull": int(valid.size),
        "n_null": int(x.isna().sum()),
        "mean": float(valid.mean()) if valid.size else np.nan,
        "std": float(valid.std()) if valid.size else np.nan,
        "min": float(valid.min()) if valid.size else np.nan,
    }
    q = valid.quantile(list(_QUANTILES.values())) if valid.size else None
    for name, frac in _QUANTILES.items():
        rec[name] = float(q.loc[frac]) if q is not None else np.nan
    rec["max"] = float(valid.max()) if valid.size else np.nan
    return rec


def indicator_summary(gdf, token, years, base_year):
    """(global-by-year, per-CBSA-at-base-year) summary tables for ``token``'s score.

    Global table: one row per year over the whole panel. Per-city table: one row per
    CBSA at ``base_year`` plus a pooled ``GLOBAL`` row, sorted by non-null count desc.
    """
    global_rows = []
    for year in years:
        col = ind.score_col(token, year)
        if col not in gdf.columns:
            continue
        rec = {"year": year, **describe_series(gdf[col])}
        global_rows.append(rec)
    global_df = pd.DataFrame(global_rows)

    base_col = ind.score_col(token, base_year)
    city_rows = []
    if base_col in gdf.columns:
        for code, sub in gdf.groupby("cbsa_code"):
            city_rows.append({"cbsa_code": code, "n_tracts": len(sub),
                              **describe_series(sub[base_col])})
        city_rows.append({"cbsa_code": "GLOBAL", "n_tracts": len(gdf),
                          **describe_series(gdf[base_col])})
    city_df = pd.DataFrame(city_rows)
    if not city_df.empty:
        pooled = city_df["cbsa_code"] == "GLOBAL"
        city_df = pd.concat([
            city_df[~pooled].sort_values("n_nonnull", ascending=False),
            city_df[pooled],
        ], ignore_index=True)
    return global_df, city_df


def buildings_summary(index, panel_tracts):
    """(global dict, per-CBSA coverage table) for the buildings index.

    Per-CBSA: n_tracts covered, n_buildings, median/p5 buildings per tract, and the count
    of panel tracts in that CBSA with zero indexed buildings.
    """
    merged = index.merge(panel_tracts, on="tract_id", how="left")
    per_tract = merged.groupby(["cbsa_code", "tract_id"], observed=True).size().rename("n")
    cov = per_tract.groupby("cbsa_code", observed=True).agg(
        n_tracts_with_bldgs="count",
        n_buildings="sum",
        median_per_tract="median",
        p5_per_tract=lambda s: s.quantile(0.05),
    ).reset_index()

    # Panel tracts (per covered CBSA) that ended up with zero indexed buildings.
    covered = set(cov["cbsa_code"].dropna())
    panel_cov = panel_tracts[panel_tracts["cbsa_code"].isin(covered)]
    panel_n = panel_cov.groupby("cbsa_code")["tract_id"].nunique()
    cov["n_panel_tracts"] = cov["cbsa_code"].map(panel_n).fillna(0).astype(int)
    cov["tracts_without_buildings"] = (
        cov["n_panel_tracts"] - cov["n_tracts_with_bldgs"]).clip(lower=0).astype(int)
    cov = cov.sort_values("n_buildings", ascending=False).reset_index(drop=True)

    n_dup = int(index["building_id"].duplicated().sum())
    glob = {
        "n_buildings": int(len(index)),
        "n_tracts_with_bldgs": int(index["tract_id"].nunique()),
        "n_panel_tracts": int(panel_tracts["tract_id"].nunique()),
        "n_cbsas_covered": int(cov["cbsa_code"].notna().sum()),
        "n_null_tract": int(index["tract_id"].isna().sum()),
        "n_duplicate_ids": n_dup,
        "dup_id_frac": n_dup / len(index) if len(index) else 0.0,
    }
    return glob, cov


# ══════════════════════════════════════════════════════════════════════════════
# Structural checks
# ══════════════════════════════════════════════════════════════════════════════

def check_panel_schema(gdf, token, years, base_year, c: Checker):
    key = f"geoid_{base_year}"
    c.check("panel geoid key present", key in gdf.columns, key)
    if key in gdf.columns:
        geoids = gdf[key].astype(str)
        c.check("geoid unique", gdf[key].is_unique, f"{gdf[key].duplicated().sum()} dup")
        c.check("geoid 11-char zero-padded", (geoids.str.len() == 11).all(),
                f"lengths: {sorted(geoids.str.len().unique())}")

    c.check("cbsa_code present", "cbsa_code" in gdf.columns)
    if "cbsa_code" in gdf.columns:
        c.check("cbsa_code non-null", gdf["cbsa_code"].notna().all(),
                f"{gdf['cbsa_code'].isna().sum()} nulls")

    # Score / training-label columns exist for every indicator token and year.
    for tok in ind.INDICATORS:
        miss = [y for y in years if ind.score_col(tok, y) not in gdf.columns]
        c.check(f"score columns for '{tok}' all years", not miss,
                f"missing: {miss}" if miss else f"{len(years)} years")
    miss_lbl = [y for y in years if f"Training_Label_{y}" not in gdf.columns]
    c.check("training-label columns all years", not miss_lbl,
            f"missing: {miss_lbl}" if miss_lbl else f"{len(years)} years")

    # Structural-change flags: income always; wealth tokens where VRE was available.
    inc_flag = ind.valid_change_col("inc")
    c.check(f"{inc_flag} present and boolean",
            inc_flag in gdf.columns and gdf[inc_flag].dropna().isin([True, False]).all())
    present = [t for t in ind.TOKEN_TO_VAR if ind.valid_change_col(t) in gdf.columns]
    c.check("wealth valid_change flags present", len(present) > 0,
            f"{len(present)}/{len(ind.TOKEN_TO_VAR)} tokens")

    has_geom = "geometry" in gdf.columns
    c.check("geometry present", has_geom)
    if has_geom:
        c.check("no empty geometries", (~gdf.geometry.is_empty).all(),
                f"{int(gdf.geometry.is_empty.sum())} empty")


def check_indicator_sanity(gdf, token, years, base_year, c: Checker):
    """Main-indicator values finite, bounded, and standardized within each CBSA."""
    base_col = ind.score_col(token, base_year)
    if base_col in gdf.columns:
        vals = pd.to_numeric(gdf[base_col], errors="coerce").dropna()
        c.check(f"main indicator '{token}' non-null present", vals.size > 0,
                f"{vals.size:,} non-null at {base_year}")
        c.check(f"main indicator '{token}' finite", np.isfinite(vals).all(),
                f"{int((~np.isfinite(vals)).sum())} non-finite")
        c.check(f"main indicator '{token}' within |z| < {MAX_ABS_Z:.0f}",
                vals.abs().max() < MAX_ABS_Z, f"max |z| = {vals.abs().max():.2f}")

    # Per-CBSA z-moments: mean ~ 0, std ~ 1. The scores are z-scored within each CBSA on
    # that CBSA's OWN vintage; only the base year is carried unaligned, so it must be tight.
    # Non-base years are spatially aligned to base tracts (max-overlap), which legitimately
    # perturbs the moments (see check_alignment_fidelity) -- reported as a WARN, not a FAIL.
    def worst_moments(yrs):
        wm, ws, n = 0.0, 0.0, 0
        for year in yrs:
            col = ind.score_col(token, year)
            if col not in gdf.columns:
                continue
            g = gdf.groupby("cbsa_code")[col]
            big = g.count() >= MIN_TRACTS_FOR_Z
            means, stds = g.mean()[big].abs(), (g.std()[big] - 1.0).abs()
            if means.empty:
                continue
            wm, ws, n = max(wm, float(means.max())), max(ws, float(stds.max())), n + 1
        return wm, ws, n

    bm, bs, bn = worst_moments([base_year])
    c.check(f"per-CBSA z-moments for '{token}' ~ (0, 1) at base year {base_year}",
            bn > 0 and bm < Z_TOL_MEAN and bs < Z_TOL_STD,
            f"worst |mean|={bm:.4f}, worst |std-1|={bs:.4f}")

    am, as_, an = worst_moments([y for y in years if y != base_year])
    if an > 0:
        c.warn(f"per-CBSA z-moments for '{token}' at aligned years within tolerance",
               am < Z_TOL_MEAN and as_ < Z_TOL_STD,
               f"{an} aligned yrs; worst |mean|={am:.4f}, worst |std-1|={as_:.4f} "
               f"(alignment-induced drift; see alignment_fidelity table)")


def check_buildings(index, panel_tracts, glob, c: Checker):
    c.check("buildings columns", set(index.columns) >= {"building_id", "cx", "cy", "tract_id"},
            str(list(index.columns)))
    c.check("building_id int64 non-negative",
            index["building_id"].dtype == np.int64 and (index["building_id"] >= 0).all())
    # Coordinate bounds: CONUS buildings must sit inside the EPSG:5070 design area. AK/HI/PR
    # legitimately fall outside it (kept-but-distorted), so those are a WARN, not a FAIL.
    oob = ~(index["cx"].between(*X_RANGE) & index["cy"].between(*Y_RANGE))
    non_conus = index["tract_id"].astype(str).str[:2].isin(NON_CONUS_FIPS)
    oob_conus = int((oob & ~non_conus).sum())
    oob_noncon = int((oob & non_conus).sum())
    c.check("CONUS buildings within EPSG:5070 bounds", oob_conus == 0,
            f"{oob_conus:,} out of bounds; cx {index['cx'].min():.0f}..{index['cx'].max():.0f}, "
            f"cy {index['cy'].min():.0f}..{index['cy'].max():.0f}")
    if oob_noncon:
        c.warn("non-CONUS (AK/HI/PR) coords inside CONUS bounds", False,
               f"{oob_noncon:,} buildings project outside EPSG:5070's design area — "
               f"distances/areas distorted, treat with care")

    sample = index.sample(min(100_000, len(index)), random_state=0)
    rx, ry = unpack_building_id(sample["building_id"].to_numpy())
    grid_ok = (np.abs(rx - sample["cx"].to_numpy()).max() <= 0.05 + 1e-9 and
               np.abs(ry - sample["cy"].to_numpy()).max() <= 0.05 + 1e-9)
    c.check("building_id unpacks to (cx, cy) on the 0.1 m grid", grid_ok)

    known = index["tract_id"].isin(set(panel_tracts["tract_id"]))
    n_unknown = int((~known & index["tract_id"].notna()).sum())
    c.check("all tract_ids exist in the panel", n_unknown == 0, f"{n_unknown:,} unknown")

    # Cross-state border duplicates: expected, non-fatal unless the rate is implausible.
    c.warn("building_id globally unique (cross-state)", glob["n_duplicate_ids"] == 0,
           f"{glob['n_duplicate_ids']:,} duplicates ({100 * glob['dup_id_frac']:.3f}%) — "
           f"border buildings shared across state files; dedupe before training")
    c.check("cross-state duplicate rate plausible", glob["dup_id_frac"] < DUP_ID_WARN_FRAC,
            f"{100 * glob['dup_id_frac']:.3f}% (threshold {100 * DUP_ID_WARN_FRAC:.0f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

_FLOAT_COLS = ["mean", "std", "min", *_QUANTILES, "max"]


def _fmt(df, floats=_FLOAT_COLS):
    df = df.copy()
    for col in floats:
        if col in df.columns:
            df[col] = df[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "nan")
    return df.to_string(index=False)


def report_panel(gdf, token, years, base_year, out_dir, c: Checker):
    _banner(f"ACS panel — main indicator '{token}' (income Rel_Score)" if token == "inc"
            else f"ACS panel — main indicator '{token}'")
    print(f"{len(gdf):,} tracts | {gdf['cbsa_code'].nunique()} CBSAs | "
          f"{len(gdf.columns)} columns | base year {base_year}\n")

    check_panel_schema(gdf, token, years, base_year, c)
    check_indicator_sanity(gdf, token, years, base_year, c)

    global_df, city_df = indicator_summary(gdf, token, years, base_year)
    print(f"\nGlobal distribution of '{token}' by year:")
    print(_fmt(global_df))

    if not city_df.empty:
        out_dir.mkdir(parents=True, exist_ok=True)
        city_path = out_dir / f"panel_{token}_stats_by_cbsa_{base_year}.csv"
        city_df.to_csv(city_path, index=False)
        print(f"\nPer-CBSA distribution of '{token}' at {base_year} "
              f"(pooled GLOBAL + top 10 by non-null) -> {city_path}")
        cities = city_df[city_df["cbsa_code"] != "GLOBAL"].head(10)
        head = pd.concat([city_df[city_df["cbsa_code"] == "GLOBAL"], cities])
        print(_fmt(head[["cbsa_code", "n_tracts", "n_nonnull", "mean", "std",
                         "min", "median", "max"]]))


def report_buildings(index, panel_tracts, out_dir, c: Checker):
    _banner("Buildings index — counts per city and global")
    print(f"{len(index):,} buildings | {index['tract_id'].nunique():,} tracts covered\n")

    glob, cov = buildings_summary(index, panel_tracts)
    check_buildings(index, panel_tracts, glob, c)

    print(f"\nGlobal: {glob['n_buildings']:,} buildings across "
          f"{glob['n_tracts_with_bldgs']:,} tracts in {glob['n_cbsas_covered']} CBSAs "
          f"({glob['n_panel_tracts']:,} panel tracts total, "
          f"{glob['n_panel_tracts'] - glob['n_tracts_with_bldgs']:,} with no buildings).")

    out_dir.mkdir(parents=True, exist_ok=True)
    cov_path = out_dir / "buildings_counts_by_cbsa.csv"
    cov.to_csv(cov_path, index=False)
    print(f"\nBuildings per CBSA (top 15 by count) -> {cov_path}")
    print(cov.head(15).to_string(index=False))

    thin = cov[cov["median_per_tract"] < 50]
    if len(thin):
        print(f"\nNote: {len(thin)} CBSAs have median <50 buildings/tract "
              f"(partial state coverage or sparse metros) — inspect before training.")


def load_panel_tracts_for_join(gdf, base_year):
    """[tract_id, cbsa_code] from the panel, keyed on the base-year geoid."""
    geoid_cols = sorted(c for c in gdf.columns if c.startswith("geoid_"))
    key = geoid_cols[-1] if geoid_cols else f"geoid_{base_year}"
    return (pd.DataFrame(gdf[[key, "cbsa_code"]])
            .rename(columns={key: "tract_id"}))


def run(panel_path, index_dir, token, out_dir, base_year, years,
        skip_buildings=False) -> int:
    """Load the final data, run all checks + summaries, return an exit code (0 ok)."""
    import geopandas as gpd

    c = Checker()
    print(f"Reading panel {panel_path} ...")
    gdf = gpd.read_feather(panel_path)
    report_panel(gdf, token, years, base_year, out_dir, c)

    if not skip_buildings:
        panel_tracts = load_panel_tracts_for_join(gdf, base_year)
        print(f"\nReading buildings index {index_dir} ...")
        index = pd.read_parquet(index_dir, columns=["building_id", "cx", "cy", "tract_id"])
        if "state" in index.columns:
            index["state"] = index["state"].astype(str)
        index["tract_id"] = index["tract_id"].astype(str)
        report_buildings(index, panel_tracts, out_dir, c)

    _banner("Final-data verification summary")
    if c.failures:
        print(f"{len(c.failures)} check(s) FAILED: {c.failures}")
        return 1
    print("All final-data checks passed.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--indicator", default="inc",
                        help=f"Indicator token to summarize. Default 'inc' (income "
                             f"Rel_Score). Valid: {sorted(ind.INDICATORS)}")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--base-year", type=int, default=BASE_YEAR)
    parser.add_argument("--start-year", type=int, default=PANEL_YEARS[0])
    parser.add_argument("--end-year", type=int, default=PANEL_YEARS[-1])
    parser.add_argument("--skip-buildings", action="store_true",
                        help="Verify the ACS panel only (skip the buildings index).")
    args = parser.parse_args(argv)

    if args.indicator not in ind.INDICATORS:
        parser.error(f"Unknown --indicator {args.indicator!r}. "
                     f"Valid: {sorted(ind.INDICATORS)}")
    years = list(range(args.start_year, args.end_year + 1))
    return run(Path(args.panel), Path(args.index_dir), args.indicator,
               Path(args.out_dir), args.base_year, years,
               skip_buildings=args.skip_buildings)


if __name__ == "__main__":
    sys.exit(main())
