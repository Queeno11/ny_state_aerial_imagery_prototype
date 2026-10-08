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


# ─── per-building reuse (the expensive geometry step, hoisted) ────────────────

def _messy_city(seed: int = 0, n_tracts: int = 12):
    """A city with undated stock, demolitions and off-grid completion years."""
    rng = np.random.default_rng(seed)
    tr = _tracts(n_tracts)
    spec, demolition = {}, {}
    bid = 1000
    for geoid in tr["GEOID_str"]:
        blds = [(1950, float(rng.integers(40, 120)))]          # baseline stock
        for _ in range(int(rng.integers(0, 4))):
            blds.append((0, float(rng.integers(10, 40))))      # undated baseline
        for _ in range(int(rng.integers(0, 6))):
            # completion years deliberately land between panel years too
            blds.append((int(rng.integers(2010, 2025)),
                         float(rng.integers(10, 70))))
        spec[geoid] = blds
        for j in range(len(blds)):
            if rng.random() < 0.1:
                demolition[bid + j] = 2005
        bid += len(blds)
    return tr, _footprints(spec, tr, demolition=demolition)


@pytest.mark.parametrize("years", [BIENNIAL, ANNUAL, [2010, 2011, 2014, 2019, 2024]])
def test_per_building_table_reproduces_the_footprint_path(years):
    """Passing a pre-computed per-building table changes nothing but the cost.

    ``footprint_tract_areas`` exists so a threshold sweep reprojects the city's
    footprints once instead of once per column; that is only safe if the two
    routes are bit-for-bit the same cohort assignment.
    """
    tr, fp = _messy_city()
    per_bldg = ces.footprint_tract_areas(
        fp, tr, year_col="CONSTRUCTION_YEAR",
        demolition_col="DEMOLITION_YEAR", area_epsg=EPSG,
    )
    kw = dict(baseline_year=2009, area_epsg=EPSG,
              demolition_col="DEMOLITION_YEAR", control_threshold=0.01)
    for threshold in (0.01, 0.05, 0.10):
        direct = ces.build_tract_cohorts(fp, tr, years, threshold=threshold, **kw)
        reused = ces.build_tract_cohorts(None, tr, years, threshold=threshold,
                                         per_building=per_bldg, **kw)
        pd.testing.assert_frame_equal(direct.cohorts, reused.cohorts)
        pd.testing.assert_frame_equal(direct.shares, reused.shares)
        assert direct.summary() == reused.summary()


@pytest.mark.parametrize("years", [BIENNIAL, ANNUAL, [2010, 2011, 2014, 2019, 2024]])
def test_cumulative_new_area_matches_the_reference_loop(years):
    """The vectorised cumulation is the old per-tract loop, exactly.

    Reference implementation below is the original row-by-row accumulation:
    walk the panel, credit each building to the first panel year at or after its
    completion, carry the running total forward.
    """
    tr, fp = _messy_city(seed=3)
    per_bldg = ces.footprint_tract_areas(
        fp, tr, year_col="CONSTRUCTION_YEAR",
        demolition_col="DEMOLITION_YEAR", area_epsg=EPSG,
    )
    baseline_year = 2009
    tract_ids = sorted(tr["GEOID_str"].astype(str).unique())

    yb = per_bldg["year_built"]
    is_new = yb.notna() & (yb > baseline_year) & (yb <= years[-1])
    new = per_bldg.loc[is_new, ["GEOID_str", "area", "year_built"]].copy()
    new["year_built"] = new["year_built"].astype(int)
    new_by_year = (new.groupby(["GEOID_str", "year_built"])["area"].sum()
                   if len(new) else pd.Series(dtype=float))
    rows = []
    for tid in tract_ids:
        cum = 0.0
        per_year = (new_by_year.loc[tid] if len(new_by_year) and tid in
                    new_by_year.index.get_level_values(0) else None)
        prev = baseline_year
        for yr in years:
            if per_year is not None:
                built = per_year[(per_year.index > prev) & (per_year.index <= yr)]
                cum += float(built.sum())
            rows.append((tid, yr, cum))
            prev = yr
    expected = pd.DataFrame(rows, columns=["GEOID_str", "year", "cum_new_area"])

    # control_threshold=None and threshold=0 keep every tract in `shares`, so the
    # comparison sees the full tract x year grid rather than a filtered subset.
    got = ces.build_tract_cohorts(
        None, tr, years, baseline_year=baseline_year, threshold=0.0,
        demolition_col="DEMOLITION_YEAR", area_epsg=EPSG, per_building=per_bldg,
    ).shares
    merged = expected.merge(got[["GEOID_str", "year", "cum_new_area"]],
                            on=["GEOID_str", "year"], suffixes=("_ref", "_got"))
    assert len(merged) == len(expected)
    np.testing.assert_allclose(merged["cum_new_area_got"],
                               merged["cum_new_area_ref"], rtol=0, atol=1e-9)


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


@pytest.mark.parametrize("anticipation", [1, 2, 3])
def test_anticipation_never_turns_a_treated_tract_into_a_control(anticipation):
    """REGRESSION: the shift used to be followed by `cohort > 0`, so a cohort
    landing exactly on 0 was read as never-treated and a cohort landing below 0
    reached csa as a negative group. Both must be dropped instead.

    Concretely, with BIENNIAL the 2010 cohort is period 1: shifted by 1 it is 0
    (csa's never-treated sentinel) and shifted by 2 it is -1.
    """
    geoids = [f"36061{i:06d}" for i in range(6)]
    outcomes = _outcomes(geoids, BIENNIAL)
    cohorts = pd.DataFrame({
        "GEOID_str": geoids,
        # periods 1, 2, 3 treated; three genuine never-treated
        "cohort_year": [2010, 2012, 2014, 0, 0, 0],
    })
    panel = ces.build_event_panel(outcomes, cohorts, panel_years=BIENNIAL,
                                  anticipation=anticipation)
    # the control group is exactly the three tracts that never built
    assert panel.n_never_treated == 3
    surviving = panel.data.drop_duplicates("GEOID_str").set_index("GEOID_str")["cohort"]
    assert set(surviving[surviving == 0].index) == set(geoids[3:])
    # no cohort value is ever negative, and every treated unit kept a pre-period
    assert (surviving >= 0).all()
    assert (surviving[surviving > 0] >= 2).all()
    # treated tracts left with no pre-period are dropped, not relabelled
    assert panel.n_treated + panel.dropped_first_period_cohort == 3


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


def test_every_city_declares_its_own_panel_and_baseline():
    """Cadence is a per-city property, not a global. A shared year list would
    silently empty most cities' panels: under predict_exact_year a year with no
    flight yields no_year_match -> NaN, and the four states' NAIP grids intersect
    in only {2021, 2023}."""
    for key, spec in ces.CSA_CITIES.items():
        assert spec.panel_years, f"{key} has no panel_years"
        assert spec.baseline() < min(spec.panel_years), key
        assert spec.sensor in {"zarr", "naip", "ortho"}, key


def test_baseline_defaults_to_the_year_before_the_panel_opens():
    spec = ces.CityCohortSpec(key="x", label="X", geoid_prefixes=("1",),
                              footprints_filename="x.parquet",
                              panel_years=(2014, 2016, 2018))
    assert spec.baseline() == 2013
    assert spec.years() == [2014, 2016, 2018]


def test_city_without_a_panel_falls_back_to_the_callers_grid():
    spec = ces.CityCohortSpec(key="x", label="X", geoid_prefixes=("1",),
                              footprints_filename="x.parquet")
    assert spec.years([2020, 2018]) == [2018, 2020]
    assert spec.baseline(2009) == 2009
    with pytest.raises(ValueError):
        spec.years()


def test_chicago_is_annual_ortho_scoped_by_footprint_coverage():
    """The three choices that make Chicago the second main-figure city: annual
    Cook County ortho (15 periods vs Illinois NAIP's 8), an out-of-sensor
    holdout, and a tract set derived from footprint coverage because the
    municipal footprint layer stops at the city line inside Cook County."""
    spec = ces.CSA_CITIES["chicago"]
    assert spec.sensor == "ortho"
    assert spec.panel_years == tuple(range(2010, 2025))
    assert spec.tract_source == "footprints"
    assert spec.geoid_prefixes == ("17031",)
    # Predictions join footprints -> the composition-fixed outcome is available.
    assert spec.id_index == "building_id"


def test_nashville_stays_on_naip_because_tn_has_no_ortho_archive():
    """TNMap's IMAGERY service is a current mosaic refreshed county-by-county,
    not a year-indexed archive — there is no local time series to panel."""
    from src.data import ortho_fetcher as of
    assert ces.CSA_CITIES["nashville"].sensor == "naip"
    assert of.available_years("nashville") == ()


def test_pooled_common_window():
    a = ces.EventStudyResult(np.array([-3, -2, -1, 0, 1]), *[np.zeros(5)] * 4,
                             n_units=1, n_treated=1, n_never_treated=1,
                             overall_att=0.0, overall_se=0.0)
    b = ces.EventStudyResult(np.array([-2, -1, 0, 1, 2, 3]), *[np.zeros(6)] * 4,
                             n_units=1, n_treated=1, n_never_treated=1,
                             overall_att=0.0, overall_se=0.0)
    assert ces.pooled_common_window({"a": a, "b": b}) == (-2, 1)
    assert ces.pooled_common_window({}) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Undated-area screen
# ═══════════════════════════════════════════════════════════════════════════════

def _screen_case():
    """Three tracts differing only in how much of their baseline area is undated.

    Each has 10,000 m2 of dated 1950 stock plus one 2015 building at 36% of it,
    so all three are treated at 5% and identical in every respect the estimator
    cares about — except the undated mass, which is what the screen keys on.
    """
    tr = _tracts(3)
    a, b, c = tr["GEOID_str"].tolist()
    spec = {
        # clean: no undated stock at all
        a: [(1950, 100.0), (2015, 60.0)],
        # 100x100 dated + 50x50 undated = 2,500/12,500 = 20.0% (NOT over 20%)
        b: [(1950, 100.0), (np.nan, 50.0), (2015, 60.0)],
        # 100x100 dated + 80x80 undated = 6,400/16,400 = 39% -> over 20%
        c: [(1950, 100.0), (np.nan, 80.0), (2015, 60.0)],
    }
    return tr, _footprints(spec, tr), (a, b, c)


def test_undated_buildings_land_in_the_baseline_denominator():
    """The premise of the screen: an unknown year is read as 'standing at
    baseline', so undated area inflates the denominator instead of being ignored.
    Tract c must therefore show a LOWER treatment share than clean tract a even
    though both gained the identical 2015 building."""
    tr, fp, (a, b, c) = _screen_case()
    res = ces.build_tract_cohorts(fp, tr, BIENNIAL, baseline_year=2009,
                                  threshold=0.05, area_epsg=EPSG)
    base = res.cohorts.set_index("GEOID_str")["base_area"].to_dict()
    assert base[a] == pytest.approx(10_000, rel=1e-3)
    assert base[c] == pytest.approx(16_400, rel=1e-3)   # 6,400 undated added in
    final = (res.shares[res.shares["year"] == max(BIENNIAL)]
             .set_index("GEOID_str")["share"].to_dict())
    assert final[c] < final[a]


def test_screen_is_off_by_default():
    """Enabling it must be an explicit decision — it is a sample restriction."""
    tr, fp, _ = _screen_case()
    res = ces.build_tract_cohorts(fp, tr, BIENNIAL, baseline_year=2009,
                                  area_epsg=EPSG)
    assert res.max_undated_area_share is None
    assert res.n_dropped_undated == 0
    assert res.n_treated + res.n_never_treated == 3


def test_screen_drops_only_tracts_over_the_threshold():
    tr, fp, (a, b, c) = _screen_case()
    res = ces.build_tract_cohorts(fp, tr, BIENNIAL, baseline_year=2009,
                                  threshold=0.05, area_epsg=EPSG,
                                  max_undated_area_share=0.20)
    kept = set(res.cohorts["GEOID_str"])
    assert kept == {a, b}          # b is exactly at 20%, not over it
    assert res.n_dropped_undated == 1
    assert res.max_undated_area_share == 0.20


def test_screen_scores_every_tract_even_when_disabled():
    """The per-tract share is reported whether or not the screen runs, so the
    restricted/unrestricted comparison can be tabulated from one pass."""
    tr, fp, (a, b, c) = _screen_case()
    res = ces.build_tract_cohorts(fp, tr, BIENNIAL, baseline_year=2009,
                                  area_epsg=EPSG)
    s = res.undated_area_share
    assert s is not None and len(s) == 3
    assert s[a] == pytest.approx(0.0, abs=1e-9)
    assert s[b] == pytest.approx(2_500 / 12_500, rel=1e-3)
    assert s[c] == pytest.approx(6_400 / 16_400, rel=1e-3)


def test_screen_never_drops_a_fully_dated_city():
    """A city whose assessor dates everything must be untouched by the screen,
    so applying it uniformly costs nothing where it is not needed."""
    tr, fp, _ = _screen_case()
    clean = fp[fp["CONSTRUCTION_YEAR"].notna()].copy()
    on = ces.build_tract_cohorts(clean, tr, BIENNIAL, baseline_year=2009,
                                 threshold=0.05, area_epsg=EPSG,
                                 max_undated_area_share=0.20)
    off = ces.build_tract_cohorts(clean, tr, BIENNIAL, baseline_year=2009,
                                  threshold=0.05, area_epsg=EPSG)
    assert on.n_dropped_undated == 0
    assert on.n_treated == off.n_treated
    assert on.n_never_treated == off.n_never_treated


def test_screen_summary_distinguishes_off_from_dropped_nothing():
    """`None` (not applied) and 0.20-with-no-drops are different facts and must
    not render identically in the robustness table."""
    tr, fp, _ = _screen_case()
    off = ces.build_tract_cohorts(fp, tr, BIENNIAL, baseline_year=2009,
                                  area_epsg=EPSG).summary()
    on = ces.build_tract_cohorts(fp, tr, BIENNIAL, baseline_year=2009,
                                 area_epsg=EPSG,
                                 max_undated_area_share=0.99).summary()
    assert off["max_undated_area_share"] is None
    assert on["max_undated_area_share"] == 0.99
    assert off["n_dropped_undated"] == on["n_dropped_undated"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Pinned control group (CONTROL_THRESHOLD)
# ═══════════════════════════════════════════════════════════════════════════════

def _control_case():
    """Four tracts spanning the intensity range, all with 10,000 m^2 baseline.

    Final new-area share, by construction: none 0%, light 4%, mid 9%, heavy 25%.
    Against a 5% treated cut, `light` is the ambiguous tract — it built, but not
    enough to be treated — and `mid`/`heavy` are treated.
    """
    tr = _tracts(4)
    none_, light, mid, heavy = tr["GEOID_str"].tolist()
    spec = {
        # baseline 100x100 = 10,000 m^2 everywhere
        none_: [(1950, 100.0)],
        light: [(1950, 100.0), (2012, 20.0)],            #   400 ->  4%
        mid:   [(1950, 100.0), (2012, 30.0)],            #   900 ->  9%
        heavy: [(1950, 100.0), (2012, 50.0)],            # 2,500 -> 25%
    }
    return tr, _footprints(spec, tr), (none_, light, mid, heavy)


def test_control_cut_is_off_by_default():
    """Pinning changes the estimand, so it must be an explicit decision."""
    tr, fp, (none_, light, mid, heavy) = _control_case()
    res = ces.build_tract_cohorts(fp, tr, BIENNIAL, baseline_year=2009,
                                  threshold=0.05, area_epsg=EPSG)
    assert res.control_threshold is None
    assert res.n_dropped_ambiguous == 0
    # the old behaviour: `light` built, but is handed to the control group
    coh = res.cohorts.set_index("GEOID_str")["cohort_year"].to_dict()
    assert coh[light] == 0 and coh[none_] == 0
    assert res.n_never_treated == 2


def test_control_cut_drops_the_ambiguous_middle_instead_of_calling_it_a_control():
    """The fix: a tract between the control cut and the treated cut leaves the
    sample. It is neither a clean control nor treated at this threshold."""
    tr, fp, (none_, light, mid, heavy) = _control_case()
    res = ces.build_tract_cohorts(fp, tr, BIENNIAL, baseline_year=2009,
                                  threshold=0.05, area_epsg=EPSG,
                                  control_threshold=0.01)
    kept = set(res.cohorts["GEOID_str"])
    assert light not in kept                      # 4% is in (1%, 5%]
    assert kept == {none_, mid, heavy}
    assert res.n_dropped_ambiguous == 1
    assert res.control_threshold == 0.01
    assert res.n_treated == 2 and res.n_never_treated == 1
    # and it is gone from the long share frame too, not merely from `cohorts`
    assert light not in set(res.shares["GEOID_str"])


def test_control_group_is_identical_across_the_threshold_grid():
    """The point of the whole change: every column compares against the SAME
    tracts. Unpinned, the never-treated set grows with the threshold."""
    tr, fp, (none_, light, mid, heavy) = _control_case()

    def never(threshold, control_threshold):
        res = ces.build_tract_cohorts(fp, tr, BIENNIAL, baseline_year=2009,
                                      threshold=threshold, area_epsg=EPSG,
                                      control_threshold=control_threshold)
        c = res.cohorts.set_index("GEOID_str")["cohort_year"]
        return set(c.index[c == 0])

    pinned = [never(t, 0.01) for t in (0.01, 0.05, 0.10)]
    assert pinned[0] == pinned[1] == pinned[2] == {none_}

    unpinned = [never(t, None) for t in (0.01, 0.05, 0.10)]
    assert unpinned == [{none_}, {none_, light}, {none_, light, mid}]


def test_control_cut_leaves_the_lowest_threshold_untouched():
    """Pinning at the smallest cut in the grid must be a no-op for that column,
    so the grid stays nested and the 1% result is unchanged by the fix."""
    tr, fp, _ = _control_case()
    kw = dict(baseline_year=2009, threshold=0.01, area_epsg=EPSG)
    pinned = ces.build_tract_cohorts(fp, tr, BIENNIAL, control_threshold=0.01, **kw)
    plain = ces.build_tract_cohorts(fp, tr, BIENNIAL, **kw)
    assert pinned.n_dropped_ambiguous == 0
    assert pinned.n_treated == plain.n_treated
    assert pinned.n_never_treated == plain.n_never_treated
    pd.testing.assert_frame_equal(
        pinned.cohorts.reset_index(drop=True), plain.cohorts.reset_index(drop=True)
    )


def test_control_cut_uses_the_final_share_not_the_share_at_each_year():
    """A tract's share only grows, so membership must be decided once on the
    final share. Deciding year by year would let a tract be a control early and
    vanish later, unbalancing the panel."""
    tr = _tracts(2)
    a, b = tr["GEOID_str"].tolist()
    # `b` crosses 1% in 2012 but only reaches 9% by the end of the panel: it is
    # ambiguous against a 10% cut for the WHOLE panel, not just its later years.
    fp = _footprints({a: [(1950, 100.0)],
                      b: [(1950, 100.0), (2012, 20.0), (2020, 22.0)]}, tr)
    res = ces.build_tract_cohorts(fp, tr, BIENNIAL, baseline_year=2009,
                                  threshold=0.10, area_epsg=EPSG,
                                  control_threshold=0.01)
    assert set(res.shares["GEOID_str"]) == {a}
    assert res.n_dropped_ambiguous == 1


def test_control_cut_above_the_treated_cut_is_rejected():
    """Would make a tract both treated and never-treated."""
    tr, fp, _ = _control_case()
    with pytest.raises(ValueError, match="control_threshold"):
        ces.build_tract_cohorts(fp, tr, BIENNIAL, baseline_year=2009,
                                threshold=0.05, area_epsg=EPSG,
                                control_threshold=0.10)


def test_control_threshold_is_the_smallest_default_threshold():
    """CONTROL_THRESHOLD has to sit at or below every column of the grid, or
    part_d's own validation would reject its default."""
    assert ces.CONTROL_THRESHOLD <= min(ces.DEFAULT_THRESHOLDS)
    assert ces.CONTROL_THRESHOLD == min(ces.DEFAULT_THRESHOLDS)
