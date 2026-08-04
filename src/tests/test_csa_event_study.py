"""Synthetic-data tests for src/csa_event_study.py (CSA construction-cohort event study).

Everything here is fabricated in-memory: square tracts, square footprints, and a
panel whose data-generating process we control, so the estimator can be checked
against a known truth. No real data, no network.

The tests pin down the four things the previous ``part_d`` implementation got
wrong, each marked below with REGRESSION:

1. a cohort dated to the panel's first year must not be recoded as never-treated,
2. period indexing must not assume a biennial grid,
3. the ATT path must not be re-anchored by subtracting ATT(k=-1),
4. pre-trends must get an actual joint test.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import box

from src import csa_event_study as ces

BIENNIAL = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
ANNUAL = list(range(2016, 2025))
EPSG = 5070
X0, Y0 = 1_500_000.0, 2_000_000.0
TRACT_SIZE = 1000.0


# ─── fixtures / builders ──────────────────────────────────────────────────────

def _tracts(n: int, prefix: str = "36061") -> gpd.GeoDataFrame:
    """n square tracts in a row, GEOIDs prefix + zero-padded counter."""
    geoms, ids = [], []
    for i in range(n):
        x = X0 + i * TRACT_SIZE
        geoms.append(box(x, Y0, x + TRACT_SIZE, Y0 + TRACT_SIZE))
        ids.append(f"{prefix}{i:06d}")
    return gpd.GeoDataFrame({"GEOID_str": ids}, geometry=geoms, crs=EPSG)


def _footprints(spec: dict[str, list[tuple[int, float]]],
                tracts: gpd.GeoDataFrame,
                demolition: dict[int, int] | None = None) -> gpd.GeoDataFrame:
    """Build footprints from {GEOID_str: [(year_built, side_length), ...]}.

    Footprints are packed inside their tract so the centroid join is
    unambiguous. Returns a GeoDataFrame indexed by a synthetic building id.
    """
    rows, geoms, index = [], [], []
    bid = 1000
    centre = {r.GEOID_str: r.geometry.centroid for r in tracts.itertuples()}
    for geoid, blds in spec.items():
        c = centre[geoid]
        for j, (year, side) in enumerate(blds):
            # fan the buildings out around the tract centre, well inside it
            dx = ((j % 5) - 2) * 60.0
            dy = ((j // 5) - 2) * 60.0
            cx, cy = c.x + dx, c.y + dy
            geoms.append(box(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))
            rows.append({"CONSTRUCTION_YEAR": year,
                         "DEMOLITION_YEAR": (demolition or {}).get(bid, np.nan)})
            index.append(bid)
            bid += 1
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=EPSG,
                           index=pd.Index(index, name="DOITT_ID"))
    return gdf


# ═══════════════════════════════════════════════════════════════════════════════
# PeriodMap
# ═══════════════════════════════════════════════════════════════════════════════

def test_period_map_is_one_based_and_reserves_zero():
    """REGRESSION 1/2: periods start at 1 so 0 stays csa's never-treated sentinel."""
    pm = ces.PeriodMap.from_years(BIENNIAL)
    assert pm.n_periods == 8
    assert pm.period(2010) == 1          # NOT 0 — 0 means never-treated in csa
    assert pm.period(2024) == 8
    assert pm.year(1) == 2010
    assert pm.year(8) == 2024


@pytest.mark.parametrize("years", [BIENNIAL, ANNUAL, [2010, 2011, 2014, 2019, 2024]])
def test_period_map_handles_any_cadence(years):
    """REGRESSION 2: no (year - base) // step arithmetic — irregular panels work."""
    pm = ces.PeriodMap.from_years(years)
    assert [pm.year(pm.period(y)) for y in years] == list(years)
    assert list(pm.period_series(pd.Series(years))) == list(range(1, len(years) + 1))


def test_period_map_rejects_bad_input():
    with pytest.raises(ValueError):
        ces.PeriodMap(years=(2012, 2010))       # unsorted
    with pytest.raises(ValueError):
        ces.PeriodMap(years=(2010, 2010))       # duplicated
    with pytest.raises(ValueError):
        ces.PeriodMap(years=())                 # empty
    pm = ces.PeriodMap.from_years(BIENNIAL)
    with pytest.raises(ValueError):
        pm.year(0)
    with pytest.raises(ValueError):
        pm.year(9)


# ═══════════════════════════════════════════════════════════════════════════════
# Cohort construction
# ═══════════════════════════════════════════════════════════════════════════════

def test_cohorts_date_treatment_to_the_first_crossing_panel_year():
    tr = _tracts(3)
    a, b, c = tr["GEOID_str"].tolist()
    spec = {
        # a: 100x100 baseline (10,000 m2), one 60x60 (3,600 m2 = 36%) built 2015
        a: [(1950, 100.0), (2015, 60.0)],
        # b: same baseline, a tiny 10x10 (100 m2 = 1%) built 2013 -> below 5%
        b: [(1950, 100.0), (2013, 10.0)],
        # c: baseline only -> never treated
        c: [(1950, 100.0)],
    }
    fp = _footprints(spec, tr)
    res = ces.build_tract_cohorts(
        fp, tr, BIENNIAL, baseline_year=2009, threshold=0.05, area_epsg=EPSG,
        demolition_col="DEMOLITION_YEAR",
    )
    coh = res.cohorts.set_index("GEOID_str")["cohort_year"].to_dict()
    # built 2015 -> first panel year at/after it is 2016
    assert coh[a] == 2016
    assert coh[b] == 0          # 1% never crosses the 5% threshold
    assert coh[c] == 0
    assert res.n_treated == 1
    assert res.n_never_treated == 2


def test_cohort_attribution_respects_panel_cadence():
    """A building finished in an off-panel year lands in the next observed year."""
    tr = _tracts(1)
    a = tr["GEOID_str"].iloc[0]
    fp = _footprints({a: [(1950, 100.0), (2011, 60.0)]}, tr)

    biennial = ces.build_tract_cohorts(fp, tr, BIENNIAL, baseline_year=2009,
                                       threshold=0.05, area_epsg=EPSG)
    assert biennial.cohorts["cohort_year"].iloc[0] == 2012   # 2011 seen in 2012

    annual = ces.build_tract_cohorts(fp, tr, list(range(2010, 2025)),
                                     baseline_year=2009, threshold=0.05, area_epsg=EPSG)
    assert annual.cohorts["cohort_year"].iloc[0] == 2011     # seen the same year


def test_cohorts_accumulate_over_time():
    """Treatment is cumulative: three small additions cross where none would alone."""
    tr = _tracts(1)
    a = tr["GEOID_str"].iloc[0]
    # baseline 10,000 m2; three 40x40 = 1,600 m2 each -> 16%, 32%, 48% cumulative
    fp = _footprints({a: [(1950, 100.0), (2012, 40.0), (2014, 40.0), (2016, 40.0)]}, tr)
    res = ces.build_tract_cohorts(fp, tr, BIENNIAL, baseline_year=2009,
                                  threshold=0.30, area_epsg=EPSG)
    # 16% in 2012, 32% in 2014 -> crosses 30% in 2014, not 2012
    assert res.cohorts["cohort_year"].iloc[0] == 2014
    shares = res.shares.set_index("year")["share"]
    assert shares[2012] == pytest.approx(0.16, abs=1e-6)
    assert shares[2014] == pytest.approx(0.32, abs=1e-6)
    assert shares[2024] == pytest.approx(0.48, abs=1e-6)


def test_tracts_without_baseline_stock_are_dropped_not_used_as_controls():
    """The old code set base_area = inf -> share 0 -> silently a control."""
    tr = _tracts(2)
    a, b = tr["GEOID_str"].tolist()
    fp = _footprints({a: [(1950, 100.0)], b: [(2015, 60.0)]}, tr)  # b: no baseline
    res = ces.build_tract_cohorts(fp, tr, BIENNIAL, baseline_year=2009,
                                  threshold=0.05, area_epsg=EPSG)
    ids = set(res.cohorts["GEOID_str"])
    assert b not in ids, "zero-baseline tract must not appear at all"
    assert a in ids
    assert res.n_dropped_no_baseline == 1
    assert res.n_never_treated == 1


def test_pre_baseline_demolitions_leave_the_denominator():
    tr = _tracts(1)
    a = tr["GEOID_str"].iloc[0]
    fp = _footprints({a: [(1950, 100.0), (1960, 100.0)]}, tr)
    fp.loc[fp.index[1], "DEMOLITION_YEAR"] = 2005    # gone before the baseline
    res = ces.build_tract_cohorts(fp, tr, BIENNIAL, baseline_year=2009,
                                  threshold=0.05, area_epsg=EPSG,
                                  demolition_col="DEMOLITION_YEAR")
    assert res.cohorts["base_area"].iloc[0] == pytest.approx(10_000.0, rel=1e-6)


def test_unknown_year_built_counts_as_baseline_stock():
    """CONSTRUCTION_YEAR 0/NaN is a missing record, not a post-baseline build."""
    tr = _tracts(1)
    a = tr["GEOID_str"].iloc[0]
    fp = _footprints({a: [(0, 100.0), (2015, 10.0)]}, tr)
    res = ces.build_tract_cohorts(fp, tr, BIENNIAL, baseline_year=2009,
                                  threshold=0.05, area_epsg=EPSG)
    assert res.cohorts["base_area"].iloc[0] == pytest.approx(10_000.0, rel=1e-6)
    assert res.cohorts["cohort_year"].iloc[0] == 0    # 1% < 5%


def test_build_cohorts_rejects_a_baseline_after_the_panel():
    tr = _tracts(1)
    fp = _footprints({tr["GEOID_str"].iloc[0]: [(1950, 100.0)]}, tr)
    with pytest.raises(ValueError):
        ces.build_tract_cohorts(fp, tr, BIENNIAL, baseline_year=2024, area_epsg=EPSG)


# ═══════════════════════════════════════════════════════════════════════════════
# Outcomes
# ═══════════════════════════════════════════════════════════════════════════════

def test_tract_outcomes_all_vs_incumbent():
    preds = pd.DataFrame({
        "building_id": [1, 2, 3, 1, 2, 3],
        "GEOID": ["36061000100"] * 6,
        "year": [2010, 2010, 2010, 2024, 2024, 2024],
        "predicted_value": [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
    })
    cy = pd.Series({1: 1950, 2: 1960, 3: 2015})   # building 3 is new
    out = ces.tract_outcomes(preds, construction_year=cy, baseline_year=2009)
    out = out.set_index("year")
    # all-buildings mean is dragged up by the new building alone
    assert out.loc[2024, "pred_all"] == pytest.approx(1.0)
    # incumbent-only mean holds composition fixed and stays flat
    assert out.loc[2024, "pred_incumbent"] == pytest.approx(0.0)
    assert out.loc[2024, "n_all"] == 3
    assert out.loc[2024, "n_incumbent"] == 2


def test_tract_outcomes_without_construction_years_omits_incumbent():
    preds = pd.DataFrame({
        "building_id": [1, 2], "GEOID": ["36061000100"] * 2,
        "year": [2010, 2010], "predicted_value": [1.0, 3.0],
    })
    out = ces.tract_outcomes(preds)
    assert "pred_incumbent" not in out.columns
    assert out["pred_all"].iloc[0] == pytest.approx(2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Event panel
# ═══════════════════════════════════════════════════════════════════════════════

def _outcomes(geoids, years, value=0.0) -> pd.DataFrame:
    rows = [{"GEOID_str": g, "year": y, "pred_all": value} for g in geoids for y in years]
    return pd.DataFrame(rows)


def test_first_period_cohort_is_dropped_not_turned_into_a_control():
    """REGRESSION 1: the old `np.where(g == 0, 0, (g - 2010) // 2)` mapped a 2010
    cohort to 0 — csa's never-treated code — putting first-period-treated tracts
    into the comparison group. It must be dropped instead."""
    geoids = [f"36061{i:06d}" for i in range(5)]
    outcomes = _outcomes(geoids, BIENNIAL)
    cohorts = pd.DataFrame({
        "GEOID_str": geoids,
        "cohort_year": [2010, 2016, 2018, 0, 0],   # first one is the trap
    })
    panel = ces.build_event_panel(outcomes, cohorts, panel_years=BIENNIAL)

    assert panel.dropped_first_period_cohort == 1
    assert geoids[0] not in set(panel.data["GEOID_str"])
    assert panel.n_treated == 2
    assert panel.n_never_treated == 2
    # nothing that survived is a treated unit masquerading as a control
    surviving = panel.data.drop_duplicates("GEOID_str").set_index("GEOID_str")["cohort"]
    assert surviving[geoids[3]] == 0 and surviving[geoids[4]] == 0
    assert surviving[geoids[1]] == 4   # 2016 is the 4th biennial period
    assert surviving[geoids[2]] == 5


def test_event_panel_balances_and_reports():
    geoids = [f"36061{i:06d}" for i in range(4)]
    outcomes = _outcomes(geoids, BIENNIAL)
    outcomes = outcomes[~((outcomes["GEOID_str"] == geoids[0]) &
                          (outcomes["year"] == 2018))]     # punch a hole
    cohorts = pd.DataFrame({"GEOID_str": geoids, "cohort_year": [2016, 2016, 0, 0]})
    panel = ces.build_event_panel(outcomes, cohorts, panel_years=BIENNIAL)
    assert panel.dropped_unbalanced == 1
    assert geoids[0] not in set(panel.data["GEOID_str"])
    assert panel.data.groupby("GEOID_str")["period"].nunique().eq(8).all()


def test_event_panel_drops_tracts_without_a_cohort():
    geoids = [f"36061{i:06d}" for i in range(3)]
    outcomes = _outcomes(geoids, BIENNIAL)
    cohorts = pd.DataFrame({"GEOID_str": geoids[:2], "cohort_year": [2016, 0]})
    panel = ces.build_event_panel(outcomes, cohorts, panel_years=BIENNIAL)
    assert panel.dropped_no_cohort == 1
    assert geoids[2] not in set(panel.data["GEOID_str"])


def test_anticipation_shifts_adoption_earlier():
    geoids = [f"36061{i:06d}" for i in range(6)]
    outcomes = _outcomes(geoids, BIENNIAL)
    cohorts = pd.DataFrame({
        "GEOID_str": geoids,
        "cohort_year": [2012, 2016, 2020, 0, 0, 0],
    })
    base = ces.build_event_panel(outcomes, cohorts, panel_years=BIENNIAL)
    shifted = ces.build_event_panel(outcomes, cohorts, panel_years=BIENNIAL,
                                    anticipation=1)
    def coh(panel):
        return panel.data.drop_duplicates("GEOID_str").set_index("GEOID_str")["cohort"]
    # 2012->period 2, 2016->4, 2020->6; anticipation=1 moves each one earlier
    assert coh(base)[geoids[1]] == 4 and coh(shifted)[geoids[1]] == 3
    assert coh(base)[geoids[2]] == 6 and coh(shifted)[geoids[2]] == 5
    # the 2012 cohort (period 2) shifts to period 1 -> no pre-period -> dropped
    assert geoids[0] in set(base.data["GEOID_str"])
    assert geoids[0] not in set(shifted.data["GEOID_str"])
    assert shifted.dropped_first_period_cohort == 1
    assert shifted.anticipation == 1
    # never-treated units stay never-treated, never shifted into a cohort
    for g in geoids[3:]:
        assert coh(shifted)[g] == 0


def test_anticipation_clears_a_timing_induced_pretrend():
    """The motivating case: cohorts dated at completion, effect visible earlier.

    Treatment starts one period *before* the recorded cohort year (site work is
    visible in the imagery). With anticipation=0 that shows up as a pre-trend;
    allowing one period of anticipation should clear it.
    """
    pytest.importorskip("csa")
    rng = np.random.default_rng(21)
    pm = ces.PeriodMap.from_years(BIENNIAL)
    year_shock = {y: rng.normal(0, 0.2) for y in BIENNIAL}
    recs, coh_rows = [], []
    for i in range(150):
        g = (2016, 2020)[i % 2]
        fe = rng.normal(0, 1.0)
        for y in BIENNIAL:
            # effect switches on one period EARLY relative to the cohort year
            k = pm.period(y) - (pm.period(g) - 1)
            te = 0.4 * (k + 1) if k >= 0 else 0.0
            recs.append({"GEOID_str": f"36061{i:06d}", "year": y,
                         "pred_all": fe + year_shock[y] + te + rng.normal(0, 0.05)})
        coh_rows.append({"GEOID_str": f"36061{i:06d}", "cohort_year": g})
    for i in range(150, 300):
        fe = rng.normal(0, 1.0)
        for y in BIENNIAL:
            recs.append({"GEOID_str": f"36061{i:06d}", "year": y,
                         "pred_all": fe + year_shock[y] + rng.normal(0, 0.05)})
        coh_rows.append({"GEOID_str": f"36061{i:06d}", "cohort_year": 0})
    outcomes, cohorts = pd.DataFrame(recs), pd.DataFrame(coh_rows)

    naive = ces.estimate_event_study(
        ces.build_event_panel(outcomes, cohorts, panel_years=BIENNIAL),
        n_boot=1500, seed=23)
    adjusted = ces.estimate_event_study(
        ces.build_event_panel(outcomes, cohorts, panel_years=BIENNIAL,
                              anticipation=1),
        n_boot=1500, seed=23)

    assert naive.pretrend_supt_p < 0.05, "mis-timed treatment should show a pre-trend"
    assert adjusted.pretrend_supt_p > naive.pretrend_supt_p
    # Non-rejection at the conventional 5% level is the claim. The bar is not
    # higher because the sup-t here maximises over only 3 pre-periods, so an
    # ordinary null draw lands around p ~ 0.1 fairly often.
    assert adjusted.pretrend_supt_p > 0.05, (
        f"anticipation=1 should clear the timing artifact "
        f"(p={adjusted.pretrend_supt_p})"
    )
    # ...and the pre-treatment coefficients themselves must be small.
    pre = adjusted.event_times < 0
    assert np.abs(adjusted.att[pre]).max() < 0.05
    assert adjusted.anticipation == 1


def test_event_panel_flags_thin_samples():
    geoids = [f"36061{i:06d}" for i in range(4)]
    outcomes = _outcomes(geoids, BIENNIAL)
    cohorts = pd.DataFrame({"GEOID_str": geoids, "cohort_year": [2016, 2016, 0, 0]})
    panel = ces.build_event_panel(outcomes, cohorts, panel_years=BIENNIAL)
    assert not panel.usable                 # 2 treated / 2 controls is far too thin
    assert panel.notes


# ═══════════════════════════════════════════════════════════════════════════════
# Estimation against a known DGP
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate(n_treated=120, n_control=120, years=BIENNIAL, cohort_years=(2014, 2018),
              effect_per_period=0.25, pre_trend=0.0, noise=0.10, seed=7):
    """Panel with tract fixed effects, common year shocks, a staggered treatment
    effect that grows ``effect_per_period`` per period after adoption, and an
    optional differential pre-trend for the treated (to make the falsification
    test fail on purpose).
    """
    rng = np.random.default_rng(seed)
    pm = ces.PeriodMap.from_years(years)
    year_shock = {y: rng.normal(0, 0.3) for y in years}

    rows = []
    n = 0
    for i in range(n_treated):
        g = cohort_years[i % len(cohort_years)]
        rows.append((f"36061{n:06d}", g, rng.normal(0, 1.0)))
        n += 1
    for i in range(n_control):
        rows.append((f"36061{n:06d}", 0, rng.normal(0, 1.0)))
        n += 1

    recs = []
    for geoid, g, fe in rows:
        for y in years:
            k = (pm.period(y) - pm.period(g)) if g else None
            te = effect_per_period * (k + 1) if (k is not None and k >= 0) else 0.0
            trend = pre_trend * (pm.period(y) - 1) if g else 0.0
            recs.append({
                "GEOID_str": geoid, "year": y,
                "pred_all": fe + year_shock[y] + te + trend + rng.normal(0, noise),
            })
    outcomes = pd.DataFrame(recs)
    cohorts = pd.DataFrame([{"GEOID_str": g, "cohort_year": c} for g, c, _ in rows])
    return outcomes, cohorts


@pytest.fixture(scope="module")
def clean_dgp_result():
    pytest.importorskip("csa")
    pytest.importorskip("polars")
    outcomes, cohorts = _simulate()
    panel = ces.build_event_panel(outcomes, cohorts, panel_years=BIENNIAL)
    assert panel.usable
    return panel, ces.estimate_event_study(panel, n_boot=1500, seed=11)


def test_recovers_the_known_treatment_path(clean_dgp_result):
    _, res = clean_dgp_result
    # effect_per_period 0.25 -> ATT(k) = 0.25 * (k + 1)
    for k in (0, 1, 2):
        att, se = res.post_att(k)
        if att is None:
            continue
        assert att == pytest.approx(0.25 * (k + 1), abs=max(4 * se, 0.12)), f"k={k}"


def test_pre_treatment_atts_are_flat_under_the_null(clean_dgp_result):
    _, res = clean_dgp_result
    pre = res.event_times < 0
    assert pre.any(), "the simulated panel should have pre-periods"
    assert np.abs(res.att[pre]).max() < 0.15


def test_no_reanchoring_at_k_minus_one(clean_dgp_result):
    """REGRESSION 3: the old code subtracted ATT(k=-1) from att/lower/upper.

    csa uses a varying base period, so ATT(-1) is a real placebo estimate. It
    must be reported as estimated (generically non-zero), and the CI must stay
    symmetric around the point estimate — proof no rigid shift was applied.
    """
    _, res = clean_dgp_result
    at_minus_one = np.flatnonzero(res.event_times == -1)
    assert at_minus_one.size == 1
    i = int(at_minus_one[0])
    assert res.att[i] != 0.0, "ATT(-1) was zeroed out — re-anchoring reintroduced"
    mid = 0.5 * (res.lower[i] + res.upper[i])
    assert mid == pytest.approx(res.att[i], abs=1e-9)


def test_pretrend_test_passes_when_there_is_no_pre_trend(clean_dgp_result):
    """REGRESSION 4: there is now a joint pre-trend test at all."""
    _, res = clean_dgp_result
    assert res.pretrend_supt_p is not None
    assert res.pretrend_supt_p > 0.10, (
        f"clean DGP should not reject flat pre-trends (p={res.pretrend_supt_p})"
    )
    if res.pretrend_wald_p is not None:
        assert res.pretrend_df is not None and res.pretrend_df >= 1


def test_pretrend_test_rejects_a_planted_differential_trend():
    pytest.importorskip("csa")
    outcomes, cohorts = _simulate(pre_trend=0.30, effect_per_period=0.0,
                                  noise=0.05, seed=3)
    panel = ces.build_event_panel(outcomes, cohorts, panel_years=BIENNIAL)
    res = ces.estimate_event_study(panel, n_boot=1500, seed=13)
    assert res.pretrend_supt_p is not None
    assert res.pretrend_supt_p < 0.05, (
        f"a 0.30/period differential trend should be detected (p={res.pretrend_supt_p})"
    )


def test_annual_cadence_estimates_too():
    """REGRESSION 2 end-to-end: an annual panel needs no code change."""
    pytest.importorskip("csa")
    outcomes, cohorts = _simulate(years=ANNUAL, cohort_years=(2019, 2021),
                                  effect_per_period=0.30, seed=5)
    panel = ces.build_event_panel(outcomes, cohorts, panel_years=ANNUAL)
    assert panel.period_map.n_periods == len(ANNUAL)
    res = ces.estimate_event_study(panel, n_boot=800, seed=17)
    att, se = res.post_att(0)
    assert att is not None and att == pytest.approx(0.30, abs=max(4 * se, 0.15))


def test_notyet_control_group_also_runs():
    pytest.importorskip("csa")
    outcomes, cohorts = _simulate(n_control=40, seed=9)
    panel = ces.build_event_panel(outcomes, cohorts, panel_years=BIENNIAL)
    res = ces.estimate_event_study(panel, control="notyet", n_boot=600, seed=19)
    assert res.control == "notyet"
    assert res.event_times.size > 0


def test_estimate_refuses_an_impossible_panel():
    pytest.importorskip("csa")
    geoids = [f"36061{i:06d}" for i in range(4)]
    outcomes = _outcomes(geoids, BIENNIAL, value=1.0)
    cohorts = pd.DataFrame({"GEOID_str": geoids, "cohort_year": [2016] * 4})
    panel = ces.build_event_panel(outcomes, cohorts, panel_years=BIENNIAL)
    with pytest.raises(RuntimeError):
        ces.estimate_event_study(panel, control="never", n_boot=100)


def test_to_row_carries_the_table_fields(clean_dgp_result):
    _, res = clean_dgp_result
    row = res.to_row()
    for key in ("pretrend_joint_p", "n_treated", "n_never_treated",
                "att_k0", "se_k0", "overall_att"):
        assert key in row


# ═══════════════════════════════════════════════════════════════════════════════
# Registry / misc
# ═══════════════════════════════════════════════════════════════════════════════

def test_available_cities_splits_ready_from_missing(tmp_path):
    (tmp_path / "buildings_nyc.parquet").write_bytes(b"")
    ready, missing = ces.available_cities(tmp_path)
    assert [s.key for s in ready] == ["nyc"]
    missing_keys = {s.key for s, _ in missing}
    assert {"chicago", "seattle", "tampa", "nashville"} <= missing_keys
    # the reason must say what is needed, not just "missing"
    reasons = {s.key: r for s, r in missing}
    assert "issue #36" in reasons["chicago"]


def test_pooled_common_window():
    a = ces.EventStudyResult(np.array([-3, -2, -1, 0, 1]), *[np.zeros(5)] * 4,
                             n_units=1, n_treated=1, n_never_treated=1,
                             overall_att=0.0, overall_se=0.0)
    b = ces.EventStudyResult(np.array([-2, -1, 0, 1, 2, 3]), *[np.zeros(6)] * 4,
                             n_units=1, n_treated=1, n_never_treated=1,
                             overall_att=0.0, overall_se=0.0)
    assert ces.pooled_common_window({"a": a, "b": b}) == (-2, 1)
    assert ces.pooled_common_window({}) is None
