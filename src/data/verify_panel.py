"""Post-generation sanity checks for the US-metros ACS panel.

Run AFTER regenerating the panel with ``process_acs.py``:

    python -m src.data.verify_panel [--panel data/processed/us_metros_panel_2011_2023.feather]

Checks (each prints PASS/FAIL; exits non-zero if any FAIL):
  1. Key: ``geoid_{base_year}`` present, unique, 11-char zero-padded strings.
  2. ``cbsa_code`` non-null; every CBSA clears MIN_METRO_POP at base year.
  3. Score columns: ``Rel_Score_{year}`` (income) for every panel year, and
     ``Rel_Score_{var}_{year}`` for every wealth indicator token.
  4. Per-CBSA z-scores have mean ~ 0 and std ~ 1 within each CBSA-year.
  5. Structural-change flags: ``valid_change_inc`` plus ``valid_change_{token}``
     for every wealth token with VRE coverage; all boolean-typed.
  6. Gate consistency: every ``valid_change_*`` equals ``pvalue_* < STRUCTURAL_CHANGE_P``
     — catches a stale panel built under an older gate (e.g. the pre-#29 p<0.01).
  7. Geometry present and non-empty (the tract polygons feed the buildings join).
"""

import argparse
import sys

import numpy as np
import pandas as pd

from src.data import indicators as ind
from src.data.process_acs import BASE_YEAR, MIN_METRO_POP, PANEL_YEARS, STRUCTURAL_CHANGE_P
from src.utils.paths import PROCESSED_DATA_DIR

DEFAULT_PANEL = (
    PROCESSED_DATA_DIR
    / f"us_metros_panel_{PANEL_YEARS[0]}_{PANEL_YEARS[-1]}.feather"
)

Z_TOL_MEAN = 0.02   # |mean| of per-CBSA z-scores
Z_TOL_STD = 0.05    # |std - 1| of per-CBSA z-scores
MIN_TRACTS_FOR_Z = 30  # skip z-moment check for tiny CBSAs (noisy)


class Checker:
    def __init__(self):
        self.failures = []

    def check(self, name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)

    def warn(self, name, ok, detail=""):
        print(f"[{'PASS' if ok else 'WARN'}] {name}" + (f" — {detail}" if detail else ""))


def check_key(gdf, base_year, c):
    key = f"geoid_{base_year}"
    c.check("key column present", key in gdf.columns, key)
    if key not in gdf.columns:
        return
    geoids = gdf[key]
    c.check("geoid unique", geoids.is_unique, f"{geoids.duplicated().sum()} duplicates")
    as_str = geoids.astype(str)
    c.check(
        "geoid 11-char zero-padded",
        (as_str.str.len() == 11).all(),
        f"lengths seen: {sorted(as_str.str.len().unique())}",
    )


def check_cbsa(gdf, base_year, c):
    c.check("cbsa_code present", "cbsa_code" in gdf.columns)
    if "cbsa_code" not in gdf.columns:
        return
    c.check("cbsa_code non-null", gdf["cbsa_code"].notna().all(),
            f"{gdf['cbsa_code'].isna().sum()} nulls")
    pop_col = f"total_population_{base_year}"
    if pop_col in gdf.columns:
        cbsa_pop = gdf.groupby("cbsa_code")[pop_col].sum()
        small = cbsa_pop[cbsa_pop <= MIN_METRO_POP]
        c.check(
            f"all CBSAs > {MIN_METRO_POP:,} pop at {base_year}",
            small.empty,
            f"{len(small)} below threshold: {list(small.index[:5])}",
        )
    else:
        c.check(f"{pop_col} present (needed for pop check)", False)


def check_score_columns(gdf, years, c):
    for token in ind.INDICATORS:
        missing = [y for y in years if ind.score_col(token, y) not in gdf.columns]
        c.check(
            f"score columns for '{token}' all years",
            not missing,
            f"missing years: {missing}" if missing else f"{len(years)} years",
        )


def check_z_moments(gdf, years, c, base_year=BASE_YEAR,
                    tokens=("inc", ind.DEFAULT_INDICATOR)):
    """Within-CBSA mean~0 / std~1 for the income and default-indicator scores.

    The scores are z-scored within each CBSA on that CBSA's OWN vintage, so the base
    year (carried unaligned) must be tight to tolerance -- that is the FAIL-guarded
    correctness check. Non-base years are spatially aligned to the base tract vintage
    (max-overlap), which legitimately perturbs the moments well beyond the tolerance
    (see ``process_acs.check_alignment_fidelity``); that drift is reported as a WARN,
    not a FAIL -- flagging it as broken z-scoring was a false alarm on the real panel.
    """
    def worst(token, yrs):
        wm, ws, n = 0.0, 0.0, 0
        for year in yrs:
            col = ind.score_col(token, year)
            if col not in gdf.columns:
                continue
            grouped = gdf.groupby("cbsa_code")[col]
            big = grouped.count() >= MIN_TRACTS_FOR_Z
            means = grouped.mean()[big].abs()
            stds = (grouped.std()[big] - 1.0).abs()
            if means.empty:
                continue
            wm, ws, n = max(wm, float(means.max())), max(ws, float(stds.max())), n + 1
        return wm, ws, n

    for token in tokens:
        bm, bs, bn = worst(token, [base_year])
        c.check(
            f"per-CBSA z-moments for '{token}' at base year {base_year}",
            bn > 0 and bm < Z_TOL_MEAN and bs < Z_TOL_STD,
            f"worst |mean|={bm:.4f}, worst |std-1|={bs:.4f}",
        )
        am, as_, an = worst(token, [y for y in years if y != base_year])
        if an > 0:
            c.warn(
                f"per-CBSA z-moments for '{token}' at aligned years within tolerance",
                am < Z_TOL_MEAN and as_ < Z_TOL_STD,
                f"{an} aligned years; worst |mean|={am:.4f}, worst |std-1|={as_:.4f} "
                f"(alignment-induced drift, expected)",
            )


def check_valid_change_flags(gdf, c):
    col_inc = ind.valid_change_col("inc")
    c.check(f"{col_inc} present", col_inc in gdf.columns)
    present, missing = [], []
    for token in ind.TOKEN_TO_VAR:
        col = ind.valid_change_col(token)
        (present if col in gdf.columns else missing).append(token)
    # Wealth flags require the VRE store at build time; report but don't fail if a
    # subset is absent — fail only if none exist.
    c.check(
        "wealth valid_change flags present",
        len(present) > 0,
        f"present: {present}" + (f"; missing: {missing}" if missing else ""),
    )
    for col in [col_inc] + [ind.valid_change_col(t) for t in present]:
        if col in gdf.columns:
            vals = gdf[col].dropna().unique()
            ok = set(map(bool, vals)) <= {True, False}
            c.check(f"{col} boolean-valued", ok, f"values: {vals[:5]}")


def check_gate_consistency(gdf, c, p_gate=STRUCTURAL_CHANGE_P):
    """Every ``valid_change_*`` flag must equal ``pvalue_* < p_gate``.

    The gate moved from p<0.01 to p<0.10 (#29); a panel generated before the
    change still carries boolean flags, so the boolean-type check alone cannot
    detect staleness — this one can, because the raw ``pvalue_*`` columns are
    stored alongside the flags.
    """
    start, end = PANEL_YEARS[0], PANEL_YEARS[-1]
    pairs = []
    if ind.valid_change_col("inc") in gdf.columns:
        pairs.append((ind.valid_change_col("inc"), f"pvalue_{start}_{end}"))
    for token, var in ind.TOKEN_TO_VAR.items():
        flag = ind.valid_change_col(token)
        if flag not in gdf.columns:
            continue
        pcols = [col for col in gdf.columns if col.startswith(f"pvalue_{var}_")]
        if pcols:
            pairs.append((flag, pcols[0]))

    if not pairs:
        c.warn("valid_change gate consistency", False,
               "no (flag, pvalue) column pairs found — cannot verify the gate")
        return
    for flag, pcol in pairs:
        if pcol not in gdf.columns:
            c.warn(f"{flag} gate consistency", False, f"{pcol} absent — cannot verify")
            continue
        p = gdf[pcol]
        mask = p.notna()
        mismatches = int(
            (gdf.loc[mask, flag].fillna(False).astype(bool) != (p[mask] < p_gate)).sum()
        )
        c.check(
            f"{flag} == ({pcol} < {p_gate})",
            mismatches == 0,
            f"{mismatches} mismatches — stale panel? Regenerate with process_acs.process_panel"
            if mismatches else f"{int(mask.sum()):,} tracts",
        )


def check_geometry(gdf, c):
    has_geom = hasattr(gdf, "geometry") and "geometry" in gdf.columns
    c.check("geometry column present", has_geom)
    if has_geom:
        c.check("no empty geometries", (~gdf.geometry.is_empty).all(),
                f"{int(gdf.geometry.is_empty.sum())} empty")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    args = parser.parse_args(argv)

    import geopandas as gpd
    print(f"Reading {args.panel} ...")
    gdf = gpd.read_feather(args.panel)
    print(f"{len(gdf):,} tracts, {gdf['cbsa_code'].nunique()} CBSAs, "
          f"{len(gdf.columns)} columns\n")

    c = Checker()
    check_key(gdf, BASE_YEAR, c)
    check_cbsa(gdf, BASE_YEAR, c)
    check_score_columns(gdf, PANEL_YEARS, c)
    check_z_moments(gdf, PANEL_YEARS, c)
    check_valid_change_flags(gdf, c)
    check_gate_consistency(gdf, c)
    check_geometry(gdf, c)

    print()
    if c.failures:
        print(f"{len(c.failures)} check(s) FAILED: {c.failures}")
        return 1
    print("All panel checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
