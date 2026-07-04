"""Orchestrates the raw-download -> training-ready-data pipeline.

Runs, in order (everything main.py needs, so after a clean pass you can go
straight to training):
  1. Verify the raw ACS downloads (per-year feathers, county->CBSA crosswalk, and the
     optional FHFA HPI / VRE replicate inputs) are actually on disk.
  2. Run ``process_acs.process_panel`` to (re)build the US-metros panel (this is
     where the ``valid_change_*`` p<0.10 gates take effect, #29).
  3. Verify the raw Microsoft building-footprint downloads are on disk.
  4. Run ``build_buildings_index`` to (re)build the buildings index/polygons.
  5. Run ``naip_coverage_audit`` for the processed states -> the per-state NAIP
     flight-year CSV that the whole-city split (#28) uses to pick each train
     city's temporal-holdout year (``--skip-naip-audit`` to skip; non-fatal on
     network failure -- the split falls back to the middle imagery year).
  6. Run ``verify_panel`` (incl. the p<0.10 gate-staleness check) and
     ``verify_final_data`` -- consistency reports over the FINAL data (panel +
     buildings) (``--skip-verify`` to skip both).
  7. Remove shard-cache directories left by the pre-#28 val sets
     (``val_spatial*``/``val_{year}``), which the new split can never read
     (``--keep-stale-caches`` to leave them).

The stage-1/3 raw-download checks are deliberately shallow -- file exists, on-disk size,
and a schema/columns peek -- to catch a failed/partial download, not a data-quality audit.
Stage 6 (and the standalone ``verify_panel.py`` / ``verify_buildings_index.py`` /
``verify_final_data.py``) is where the processed output itself is validated.

    python -m src.data.run_data_pipeline                      # verify + process everything
    python -m src.data.run_data_pipeline --check-only          # verify only, process nothing
    python -m src.data.run_data_pipeline --skip-buildings       # ACS panel only
    python -m src.data.run_data_pipeline --states Delaware NewYork   # dev-scale buildings run

Each processing stage is gated on its own raw-data check passing (``--force`` to
override); the stages are otherwise independent; a failure in one does not
block the others. NOTE: the first main.py run after a split change still needs a
fresh ``run_id`` (or ``retrain=True``) -- training shard caches embed the split.
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import pyarrow as pa

from src.data import process_acs
from src.data import build_buildings_index as bbi
from src.data import verify_final_data
from src.data import verify_panel
from src.data.cbsa_brackets import DEFAULT_COVERAGE_CSV
from src.data.download_vre import VRE_ROOT, VRE_FIRST_YEAR, ALL_STATES as VRE_ALL_STATES, \
    VRE_TABLES
from src.utils.paths import ACS_ROOT_DIR, CACHE_DIR, PROCESSED_DATA_DIR

# Minimum plausible on-disk size for a real download -- catches a truncated/empty
# file, not meant as a precise data-quality bound.
MIN_ACS_FEATHER_BYTES = 1_000_000       # real files run 70-500 MB
MIN_FOOTPRINTS_ZIP_BYTES = 10_000       # real files run several MB to >1 GB

# Raw columns process_acs.compute_acs_indicators needs from each year's ACS feather
# (mirrors process_acs.WEALTH_CORE_COLS + the income columns load_and_prep reads).
REQUIRED_ACS_COLUMNS = {"geoid", process_acs.PCI_COL, process_acs.PCI_ERR_COL,
                        *process_acs.WEALTH_CORE_COLS}

# 50 states + DC, in the Microsoft US Building Footprints naming convention (no spaces).
# Used only to report which official regions have not been downloaded yet -- processing
# itself always runs on whatever is actually present on disk (see --all-states default).
ALL_US_STATE_NAMES = sorted({
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "DistrictofColumbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
    "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
    "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
    "Nebraska", "Nevada", "NewHampshire", "NewJersey", "NewMexico", "NewYork",
    "NorthCarolina", "NorthDakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
    "RhodeIsland", "SouthCarolina", "SouthDakota", "Tennessee", "Texas", "Utah",
    "Vermont", "Virginia", "Washington", "WestVirginia", "Wisconsin", "Wyoming",
})


class Checker:
    """Collects PASS/WARN/FAIL lines; FAIL is the only status that blocks a stage."""

    def __init__(self):
        self.failures = []

    def check(self, name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)

    def warn(self, name, ok, detail=""):
        print(f"[{'PASS' if ok else 'WARN'}] {name}" + (f" -- {detail}" if detail else ""))


def _banner(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ══════════════════════════════════════════════════════════════════════════════
# Raw-data checks
# ══════════════════════════════════════════════════════════════════════════════

def check_acs_years(years, c: Checker):
    """Per-year ``us_tracts_acs5_{year}.feather``: exists, minimum size, required columns."""
    for year in years:
        path = ACS_ROOT_DIR / str(year) / f"us_tracts_acs5_{year}.feather"
        if not path.exists():
            c.check(f"ACS {year} feather present", False, str(path))
            continue
        size = path.stat().st_size
        if size < MIN_ACS_FEATHER_BYTES:
            c.check(f"ACS {year} feather size", False, f"{size:,} bytes at {path}")
            continue
        with pa.memory_map(str(path), "r") as source:
            columns = set(pa.ipc.open_file(source).schema.names)
        missing = REQUIRED_ACS_COLUMNS - columns
        c.check(f"ACS {year} feather columns", not missing,
                f"missing {sorted(missing)}" if missing else f"{size / 1e6:.0f} MB")


def check_crosswalk(c: Checker):
    """County->CBSA crosswalk: cache CSV or the OMB xlsx (the NBER network fallback 404s)."""
    ok = process_acs.CBSA_CROSSWALK_CACHE.exists() or process_acs.CBSA_CROSSWALK_XLSX.exists()
    c.check("county->CBSA crosswalk present",
            ok, str(process_acs.CBSA_CROSSWALK_CACHE) if not ok else "")


def check_fhfa(c: Checker):
    """FHFA HPI files: optional -- W3 appreciation degrades to 1.0 if absent."""
    c.warn("FHFA county HPI present", process_acs.FHFA_COUNTY_FILE.exists(),
           str(process_acs.FHFA_COUNTY_FILE))
    c.warn("FHFA CBSA HPI present", process_acs.FHFA_CBSA_FILE.exists(),
           str(process_acs.FHFA_CBSA_FILE))


def check_vre(years, c: Checker):
    """VRE replicate store for the two structural-change vintages: optional -- the wealth
    ``valid_change_*`` flags are simply skipped if absent (see process_acs.process_panel)."""
    start_year, end_year = min(years), max(years)
    wealth_start_year = max(start_year, VRE_FIRST_YEAR)
    max_files = len(VRE_TABLES) * len(VRE_ALL_STATES)
    for year in sorted({wealth_start_year, end_year}):
        n_files = len(list((VRE_ROOT / str(year)).glob("*.parquet"))) if (VRE_ROOT / str(year)).exists() else 0
        c.warn(f"VRE replicate files for {year}", n_files > 0,
               f"{n_files}/{max_files} (table, state) files under {VRE_ROOT / str(year)}")


def check_buildings_states(states, footprints_dir: Path):
    """Per-state ``{state}.geojson.zip``: exists + minimum size. Returns (present, missing)."""
    present, missing = [], []
    for state in states:
        zip_path = footprints_dir / f"{state}.geojson.zip"
        if zip_path.exists() and zipfile.is_zipfile(zip_path) and \
                zip_path.stat().st_size >= MIN_FOOTPRINTS_ZIP_BYTES:
            present.append(state)
        else:
            missing.append(state)
    return present, missing


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════════

def run_acs_stage(args, years) -> bool:
    _banner("Verifying ACS raw downloads")
    c = Checker()
    check_acs_years(years, c)
    check_crosswalk(c)
    check_fhfa(c)
    check_vre(years, c)
    ok = not c.failures

    if args.check_only:
        return ok
    if not ok and not args.force:
        print(f"\nSkipping ACS processing -- {len(c.failures)} check(s) failed above "
              f"(pass --force to run anyway).")
        return ok

    _banner("Running ACS processing (process_acs.process_panel)")
    process_acs.process_panel(years=years, base_year=args.base_year)
    return ok


def run_buildings_stage(args, panel_path: Path) -> bool:
    _banner("Verifying building-footprint raw downloads")
    footprints_dir = Path(args.footprints_dir)

    if args.states:
        requested = args.states
    else:
        # Default (and --all-states): process whatever is actually on disk.
        requested = bbi.available_states(footprints_dir)

    present, missing = check_buildings_states(requested, footprints_dir)
    for state in present:
        print(f"[PASS] {state}.geojson.zip present")
    for state in missing:
        print(f"[FAIL] {state}.geojson.zip present -- "
              f"{footprints_dir / f'{state}.geojson.zip'}")

    not_downloaded = sorted(set(ALL_US_STATE_NAMES) - set(bbi.available_states(footprints_dir)))
    if not_downloaded:
        print(f"\nNote: {len(not_downloaded)} of the 51 US states/DC have no zip under "
              f"{footprints_dir} yet: {not_downloaded}")

    ok = not missing
    if args.check_only:
        return ok
    if not present:
        print("\nNo building-footprint zips available -- nothing to process.")
        return ok
    if missing and not args.force:
        print(f"\n{len(missing)} requested state(s) missing their zip -- processing the "
              f"{len(present)} present ones only. Pass --force to silence this note.")

    _banner("Running buildings-index processing (build_buildings_index)")
    bbi_argv = [
        "--states", *present,
        "--footprints-dir", str(footprints_dir),
        "--panel", str(panel_path),
        "--out", str(PROCESSED_DATA_DIR),
        "--batch-size", str(args.batch_size),
    ]
    if args.all_buildings:
        bbi_argv.append("--all-buildings")
    if args.no_polygons:
        bbi_argv.append("--no-polygons")
    exit_code = bbi.main(bbi_argv)
    if exit_code:
        ok = False
    return ok


def run_naip_coverage_stage(args, footprints_dir: Path) -> bool:
    """Audit NAIP flight-year coverage per state into the CSV consumed by the
    whole-city split (#28): ``cbsa_brackets.pick_holdout_year`` reads it to place
    each train city's temporal-holdout year at the middle of real coverage.

    Network-dependent (Planetary Computer STAC) but non-fatal: on failure the
    split falls back to the middle of the imagery years and this stage WARNs.
    Skipped when the CSV already covers every requested state (``--force`` to
    re-audit).
    """
    _banner("Auditing NAIP coverage (naip_coverage_audit)")
    from src.data import naip_coverage_audit  # lazy: pulls the STAC client

    states = args.states or bbi.available_states(footprints_dir)
    states = sorted(set(states) & set(naip_coverage_audit.STATE_CENTERS))
    if not states:
        print("[WARN] no auditable states (none present / none in STATE_CENTERS); skipping.")
        return True

    out = Path(args.naip_coverage_out)
    if out.exists() and not args.force:
        try:
            import pandas as pd
            covered = set(pd.read_csv(out)["state"].unique())
        except Exception:
            covered = set()
        missing = sorted(set(states) - covered)
        if not missing:
            print(f"[PASS] {out} already covers all {len(states)} requested state(s); "
                  f"skipping (--force to re-audit).")
            return True
        print(f"{len(missing)} state(s) not yet audited ({missing[:5]}...); re-auditing "
              f"all {len(states)} requested states (the audit rewrites the CSV whole).")

    try:
        rc = naip_coverage_audit.main([
            "--states", *states,
            "--years", str(args.naip_years[0]), str(args.naip_years[1]),
            "--out", str(out),
        ])
        return rc == 0
    except Exception as e:  # network / STAC failure must not block the data stages
        print(f"[WARN] NAIP coverage audit failed ({e!r}); the split will fall back to "
              f"the middle imagery year per city. Re-run `python -m "
              f"src.data.naip_coverage_audit` when the network is available.")
        return True


def run_final_verification(args, panel_path: Path, skip_buildings: bool) -> bool:
    """Consistency reports over the FINAL data: ``verify_panel`` (schema, z-moments,
    p<0.10 gate staleness) then ``verify_final_data`` (panel + buildings). Non-fatal
    to the processing stages: they run only if the panel exists, and a failed check
    is reported in the Summary but does not raise."""
    _banner("Verifying final data (verify_panel + verify_final_data)")
    if not panel_path.exists():
        print(f"[WARN] panel not found ({panel_path}); skipping final-data verification.")
        return True

    try:
        panel_rc = verify_panel.main(["--panel", str(panel_path)])
    except Exception as e:
        print(f"[WARN] panel verification errored ({e!r}); "
              f"run `python -m src.data.verify_panel` separately.")
        panel_rc = 0

    index_dir = PROCESSED_DATA_DIR / "buildings_index"
    skip_b = skip_buildings or not index_dir.exists()
    if skip_buildings:
        print("Buildings stage skipped -> verifying the ACS panel only.")
    elif not index_dir.exists():
        print(f"[WARN] {index_dir} not found; verifying the ACS panel only.")

    years = list(range(args.start_year, args.end_year + 1))
    try:
        rc = verify_final_data.run(
            panel_path, index_dir, "inc", verify_final_data.DEFAULT_OUT_DIR,
            args.base_year, years, skip_buildings=skip_b,
        )
    except Exception as e:  # a verification bug must not discard hours of processing
        print(f"[WARN] final-data verification errored ({e!r}); "
              f"run `python -m src.data.verify_final_data` separately.")
        rc = 0
    return panel_rc == 0 and rc == 0


# Shard-cache dirs written by the pre-#28 split's val sets. The new val sets are
# val_cities/val_temporal, so these can never be read again; train_cache keeps its
# name and is cleared by main.py itself on retrain / fresh run_id.
STALE_CACHE_DIRS = ("val_spatial_cache", "val_spatial_temporal_cache")


def clean_stale_caches(cache_dir: Path = CACHE_DIR) -> list:
    """Remove orphaned val shard caches from the old split (returns removed names)."""
    removed = []
    if not cache_dir.exists():
        return removed
    candidates = [cache_dir / name for name in STALE_CACHE_DIRS]
    candidates += sorted(cache_dir.glob("val_[0-9][0-9][0-9][0-9]_cache"))
    for path in candidates:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
    return removed


def run_cache_cleanup_stage() -> None:
    _banner("Removing stale val shard caches (pre-#28 split)")
    removed = clean_stale_caches()
    if removed:
        print(f"Removed {len(removed)} stale cache dir(s) under {CACHE_DIR}: {removed}")
    else:
        print("No stale val caches found.")
    print("Reminder: the first main.py run on the new split needs a fresh run_id "
          "(or retrain=True) so the TRAIN shard cache is rebuilt too.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ACS stage
    parser.add_argument("--start-year", type=int, default=process_acs.PANEL_YEARS[0])
    parser.add_argument("--end-year", type=int, default=process_acs.PANEL_YEARS[-1])
    parser.add_argument("--base-year", type=int, default=process_acs.BASE_YEAR)
    parser.add_argument("--skip-acs", action="store_true")
    # Buildings stage
    parser.add_argument("--states", nargs="+", default=None,
                        help="State file stems, e.g. Delaware NewYork. Default: everything "
                             "present under --footprints-dir.")
    parser.add_argument("--footprints-dir", default=str(bbi.MS_FOOTPRINTS_DIR))
    parser.add_argument("--all-buildings", action="store_true",
                        help="Keep buildings outside panel tracts (tract_id=None)")
    parser.add_argument("--no-polygons", action="store_true",
                        help="Skip the cold buildings_polygons GeoParquet")
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--skip-buildings", action="store_true")
    # NAIP coverage audit (feeds the per-city temporal holdout of the #28 split)
    parser.add_argument("--skip-naip-audit", action="store_true",
                        help="Skip the NAIP flight-year coverage audit stage.")
    parser.add_argument("--naip-coverage-out", default=str(DEFAULT_COVERAGE_CSV),
                        help="Where to write the coverage CSV (default: the path "
                             "cbsa_brackets auto-probes at split time).")
    parser.add_argument("--naip-years", nargs=2, type=int, default=[2010, 2024],
                        metavar=("FROM", "TO"),
                        help="Imagery year range to audit (default: 2010 2024).")
    # Orchestration
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the final-data consistency reports "
                             "(verify_panel + verify_final_data).")
    parser.add_argument("--keep-stale-caches", action="store_true",
                        help="Do not delete the pre-#28 val shard caches under CACHE_DIR.")
    parser.add_argument("--check-only", action="store_true",
                        help="Run the raw-data checks only; process nothing.")
    parser.add_argument("--force", action="store_true",
                        help="Run a processing stage even if its raw-data checks failed "
                             "(also re-audits NAIP coverage even when the CSV is complete).")
    args = parser.parse_args(argv)

    years = list(range(args.start_year, args.end_year + 1))
    panel_path = PROCESSED_DATA_DIR / f"us_metros_panel_{years[0]}_{years[-1]}.feather"

    acs_ok = True
    if not args.skip_acs:
        acs_ok = run_acs_stage(args, years)

    buildings_ok = True
    if not args.skip_buildings:
        buildings_ok = run_buildings_stage(args, panel_path)

    naip_ok = True
    if not args.check_only and not args.skip_naip_audit:
        naip_ok = run_naip_coverage_stage(args, Path(args.footprints_dir))

    verify_ok = True
    if not args.check_only and not args.skip_verify:
        verify_ok = run_final_verification(args, panel_path, args.skip_buildings)

    if not args.check_only and not args.keep_stale_caches:
        run_cache_cleanup_stage()

    _banner("Summary")
    print(f"ACS stage:       {'OK' if acs_ok else 'FAILED CHECKS'}"
          f"{' (skipped)' if args.skip_acs else ''}")
    print(f"Buildings stage: {'OK' if buildings_ok else 'FAILED CHECKS'}"
          f"{' (skipped)' if args.skip_buildings else ''}")
    print(f"NAIP coverage:   {'OK' if naip_ok else 'FAILED CHECKS'}"
          f"{' (skipped)' if (args.check_only or args.skip_naip_audit) else ''}")
    print(f"Final-data check:{' OK' if verify_ok else ' FAILED CHECKS'}"
          f"{' (skipped)' if (args.check_only or args.skip_verify) else ''}")

    return 0 if (acs_ok and buildings_ok and naip_ok and verify_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
