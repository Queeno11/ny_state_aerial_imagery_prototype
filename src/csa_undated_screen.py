"""Restricted vs unrestricted event study under the undated-area screen (#36).

Why this exists
---------------
Most assessors record a construction year only for *residential* improvements.
Measured on the registered cities: Franklin County (Columbus) dates 93.1% of
residential parcels but 5.5% of commercial; Cook County's ``char_yrblt`` excludes
commercial outright; Allegheny dates 1.7% of commercial. Such a city cannot enter
the event study as-is, because ``build_tract_cohorts`` reads an unknown year as
"standing at baseline": the undated building's area is missing from the numerator
*and* inflating the denominator, so treatment intensity is biased toward zero
twice over, worst in exactly the tracts with the most non-residential stock.

:data:`~src.csa_event_study.MAX_UNDATED_AREA_SHARE` makes those cities usable by
dropping tracts whose undated buildings exceed a share of baseline area. That is
a sample restriction, and a referee is entitled to ask what it does to the
estimate. This module answers that with evidence rather than argument: it runs
the same city both ways and tabulates the ATT side by side.

The logic of the test
---------------------
Run it where the answer is *known* — a city whose assessor dates commercial stock
too (Tampa dates 97.4% of buildings; San Antonio 84% of commercial parcels). If
the restricted and unrestricted ATTs agree there, the screen is demonstrably
innocuous and applying it to a residential-only city is defensible. If they
diverge, the screen is not safe and such a city should stay excluded. Either way
the answer is measured, not asserted.

Note the test is only as strong as the screen is binding: at 20% the screen drops
0.8% of Tampa's tracts but 11.2% of San Antonio's, so San Antonio is the sharper
test of the two and Tampa is close to a null by construction.

Run
---
    python -m src.csa_undated_screen tampa --savename run_20260722
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from src import csa_event_study as ces
from src.utils.paths import PROCESSED_DATA_DIR, RESULTS_DIR, TABLES_DIR

# The point estimates this table compares do not depend on the bootstrap; only
# the confidence bands do. 2,000 keeps a multi-city, multi-threshold sweep cheap
# without touching the quantity under test.
DEFAULT_N_BOOT = 2_000


def _load_city(city: str, *, processed_dir: Path, csa_root: Path):
    """Footprints, tracts and tract-year outcomes for one city.

    Delegates to ``evaluation``'s loaders rather than reimplementing them: the
    tract set depends on the ACS split geometry and, for sub-county sources, on
    footprint coverage. A second implementation of that would be a second thing
    to keep in sync with ``part_d``.
    """
    from src import evaluation as ev

    spec = ces.CSA_CITIES[city]
    fp_path = Path(processed_dir) / spec.footprints_filename
    if not fp_path.exists():
        raise FileNotFoundError(
            f"{fp_path} missing — build it with "
            f"`python -m src.data.build_year_built {city}` first.")
    footprints = gpd.read_parquet(fp_path)

    construction_year = None
    if spec.id_index is not None and footprints.index.name == spec.id_index:
        construction_year = pd.to_numeric(footprints[spec.year_col],
                                          errors="coerce")

    years = spec.years()
    baseline_year = spec.baseline()
    tracts = ev._csa_city_tracts(Path(processed_dir), spec.geoid_prefixes,
                                 spec=spec, footprints=footprints)
    if tracts.empty:
        raise RuntimeError(f"{city}: no tracts matched {spec.geoid_prefixes}")

    bld = ev._csa_city_preds(Path(csa_root), spec, years)
    if bld.empty:
        raise RuntimeError(
            f"{city}: no per-city predictions under {csa_root}/csa/{city} for "
            f"{years}. Run `src.csa_predict` for this city first.")
    bld = bld[bld["GEOID_str"].isin(set(tracts["GEOID_str"]))]
    outcomes = ces.tract_outcomes(bld, construction_year=construction_year,
                                  baseline_year=baseline_year)
    return spec, footprints, tracts, outcomes, years, baseline_year


def compare_city(city: str, *,
                 savename: str | None = None,
                 processed_dir: Path = PROCESSED_DATA_DIR,
                 results_dir: Path | None = None,
                 thresholds=ces.DEFAULT_THRESHOLDS,
                 screen: float = ces.MAX_UNDATED_AREA_SHARE,
                 control_threshold: float | None = ces.CONTROL_THRESHOLD,
                 outcome_col: str = "pred_all",
                 n_boot: int = DEFAULT_N_BOOT,
                 verbose: bool = True) -> pd.DataFrame:
    """One row per (threshold x screen setting) with the ATT under each.

    ``screen=None`` is the unrestricted arm and is always run; the restricted arm
    uses ``screen``. Returns a tidy frame — the paired rows are what goes in the
    robustness table.

    ``control_threshold`` must match what ``part_d`` uses, or the same
    (city, threshold) label denotes a different estimand in the two tables. It
    defaults to :data:`src.csa_event_study.CONTROL_THRESHOLD` for that reason.
    """
    csa_root = Path(results_dir) if results_dir is not None \
        else Path(RESULTS_DIR) / (savename or "")
    spec, footprints, tracts, outcomes, years, baseline_year = _load_city(
        city, processed_dir=processed_dir, csa_root=csa_root)

    if outcome_col not in outcomes.columns:
        raise KeyError(f"{city}: no {outcome_col!r} outcome "
                       f"(have {list(outcomes.columns)})")

    if verbose:
        print(f"\n=== {spec.label}: undated-area screen at "
              f"{screen:.0%} vs unrestricted ===")
        print(f"  panel {years} (baseline {baseline_year}), "
              f"{outcomes['GEOID_str'].nunique():,} tracts with predictions")

    rows = []
    for arm in (None, float(screen)):
        for thresh in thresholds:
            coh = ces.build_tract_cohorts(
                footprints, tracts, years, baseline_year=baseline_year,
                threshold=thresh, year_col=spec.year_col,
                demolition_col=spec.demolition_col, area_epsg=spec.area_epsg,
                max_undated_area_share=arm,
                control_threshold=control_threshold,
            )
            panel = ces.build_event_panel(
                outcomes, coh.cohorts, outcome_col=outcome_col,
                panel_years=years,
            )
            row = {
                "city": spec.key,
                "threshold": float(thresh),
                "control_threshold": control_threshold,
                "screen": arm,
                "n_dropped_undated": int(coh.n_dropped_undated),
                "n_dropped_ambiguous": int(coh.n_dropped_ambiguous),
                "n_treated": int(panel.n_treated),
                "n_never_treated": int(panel.n_never_treated),
                "overall_att": float("nan"),
                "overall_se": float("nan"),
                "pretrend_p": float("nan"),
                "note": "; ".join(panel.notes),
            }
            if panel.usable:
                res = ces.estimate_event_study(
                    panel, n_boot=n_boot,
                    label=f"{spec.key}/{thresh:.0%}/"
                          f"{'unrestricted' if arm is None else f'{arm:.0%}'}")
                row.update(overall_att=res.overall_att,
                           overall_se=res.overall_se,
                           pretrend_p=res.pretrend_supt_p)
            rows.append(row)
            if verbose:
                arm_s = "unrestricted" if arm is None else f"screen {arm:.0%}"
                att = ("     n/a" if pd.isna(row["overall_att"])
                       else f"{row['overall_att']:+.4f}")
                print(f"  {thresh:>5.0%} {arm_s:<14s} "
                      f"treated {row['n_treated']:>4,} / never "
                      f"{row['n_never_treated']:>4,} "
                      f"(dropped {row['n_dropped_undated']:>3,})  "
                      f"ATT {att}")
                # An unusable panel is far more often an incomplete prediction
                # set than a real shortage of treated tracts — build_event_panel
                # requires a BALANCED panel, so a city missing one panel year
                # loses every tract at once and reports 0/0. Say which it is.
                if not panel.usable:
                    why = "; ".join(panel.notes) or "empty panel"
                    print(f"        ! not estimated: {why} "
                          f"(unbalanced dropped "
                          f"{panel.dropped_unbalanced:,} tracts)")

    out = pd.DataFrame(rows)
    return _add_delta(out)


def _add_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the restricted-minus-unrestricted ATT gap, in SE units.

    The gap in raw score units is not interpretable on its own — the question is
    whether the screen moves the estimate by more than its own noise, so the gap
    is also expressed against the unrestricted arm's standard error.
    """
    base = (df[df["screen"].isna()]
            .set_index("threshold")[["overall_att", "overall_se"]])
    df = df.copy()
    df["att_delta"] = df.apply(
        lambda r: (r["overall_att"] - base["overall_att"].get(r["threshold"],
                                                              float("nan"))),
        axis=1)
    df["att_delta_in_se"] = df.apply(
        lambda r: r["att_delta"] / base["overall_se"].get(r["threshold"],
                                                          float("nan")),
        axis=1)
    df.loc[df["screen"].isna(), ["att_delta", "att_delta_in_se"]] = float("nan")
    return df


def run(cities, *, savename: str | None = None, tables_dir: Path = TABLES_DIR,
        out_path: Path | None = None, **kwargs) -> pd.DataFrame:
    """Compare several cities and write ``csa_undated_screen_comparison.csv``.

    A city with no predictions yet is reported and skipped rather than aborting
    the sweep: the cities that matter most here are the ones whose panels are
    still being produced, so a partial table is the normal intermediate state.
    """
    frames = []
    for city in cities:
        try:
            frames.append(compare_city(city, savename=savename, **kwargs))
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            print(f"  skipped {city}: {exc}")
    if not frames:
        print("no city could be compared")
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    path = Path(out_path) if out_path is not None else \
        Path(tables_dir) / "csa_undated_screen_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\n-> {path}")
    return out


if __name__ == "__main__":                                  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cities", nargs="+")
    ap.add_argument("--savename", default=None,
                    help="run id under results/ holding csa/<city>/ predictions")
    ap.add_argument("--screen", type=float, default=ces.MAX_UNDATED_AREA_SHARE)
    ap.add_argument("--outcome", default="pred_all")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--out", default=None,
                    help="write the CSV here instead of results/tables/")
    args = ap.parse_args()
    run(args.cities, savename=args.savename, screen=args.screen,
        outcome_col=args.outcome, n_boot=args.n_boot,
        out_path=args.out)
