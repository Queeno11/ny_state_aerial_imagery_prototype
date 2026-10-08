# -*- coding: utf-8 -*-
"""
src/csa_event_study.py — Callaway–Sant'Anna event study on construction cohorts.

Validation of the *method*, not a substantive result: if the model reads real
neighbourhood change, tracts that absorb new real-estate development should show
flat pre-trends and a positive, growing post-construction effect on the model's
predicted wealth. See GitHub issue #36 for the sample design this implements.

Design notes that matter statistically
--------------------------------------
* **Cohorts.** ``g`` = the first *panel year* at which a tract's cumulative
  post-baseline new building area exceeds ``threshold`` × its baseline building
  area. Once treated, always treated (staggered adoption).
* **Period indexing is an explicit ordered map**, never year arithmetic. The
  panels differ in cadence (NYC zarr biennial, NAIP annual, Chicago annual,
  Tampa near-annual), so ``(year - 2010) // 2`` is wrong in general; and it
  silently maps a 2010 cohort onto 0, which is ``csa``'s *never-treated*
  sentinel — turning first-period-treated tracts into controls.
* **No re-anchoring.** ``csa`` uses a *varying* base period (``get_pt_period``:
  base = ``t-1`` for ``t < g``), so ``ATT(k=-1)`` is a genuine placebo estimate,
  not a mechanical zero. Subtracting it from the point estimates and both CI
  bounds — as the previous implementation did — discards a pre-trend
  coefficient and shifts every post-treatment ATT by a noisy estimate without
  propagating that estimate's uncertainty. The ATTs are reported as ``csa``
  produces them.
* **Pre-trends get a joint test.** Issue #36 makes flat, insignificant
  pre-trends the falsification test. ``pretrend_test`` reuses ``csa``'s own
  multiplier-bootstrap draws to form a sup-t statistic over the pre-period event
  times, so the test and the plotted simultaneous CIs come from one object.
* **Undefined treatment intensity is dropped, not called a control.** A tract
  with no baseline building stock has an undefined "share of stock replaced";
  the previous code gave it ``base_area = inf`` → ``share = 0`` → never-treated.
* **The threshold grid holds its control group fixed.** Each column compares
  "built more than ``threshold``" against one pinned comparison group — tracts
  that never crossed :data:`CONTROL_THRESHOLD` — and drops the tracts in
  between. Letting the never-treated group be the complement of the treated
  group, as it was, means the comparison group changes with every column, so
  the grid varies dose *and* control composition at once and cannot be read as
  a dose-response. Same reasoning as the sentence above, one level up: a tract
  that built 3% of its stock is not a valid counterfactual for one that built
  20%, and calling it one is a measurement choice, not a robustness check.
* **Composition.** The all-buildings tract mean moves mechanically when new
  buildings enter it. ``tract_outcomes`` therefore also emits an
  *incumbent-only* mean (buildings existing at the baseline year), which holds
  composition fixed: a positive ATT there is the model reading the
  *neighbourhood*, not just scoring the new building itself.

City coverage is a registry (:data:`CSA_CITIES`) so the per-city design in #36
has somewhere to attach. Only cities whose footprint + year-built table is on
disk can run; the rest are reported as unavailable with a reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "CityCohortSpec",
    "CSA_CITIES",
    "CONTROL_THRESHOLD",
    "DEFAULT_THRESHOLDS",
    "HEADLINE_THRESHOLD",
    "MAX_UNDATED_AREA_SHARE",
    "PeriodMap",
    "CohortResult",
    "EventPanel",
    "EventStudyResult",
    "available_cities",
    "build_tract_cohorts",
    "build_event_panel",
    "estimate_event_study",
    "footprint_tract_areas",
    "pretrend_test",
    "tract_outcomes",
]

# Cumulative-new-area thresholds defining "treated". Reported as a robustness
# grid; HEADLINE_THRESHOLD is the one that goes in the main figure.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.01, 0.05, 0.10)
HEADLINE_THRESHOLD: float = 0.05

# Every cell of that grid compares ">threshold" against this ONE control
# definition, and drops the tracts in between. Without it the grid is not a
# robustness check at all: raising the threshold moves a tract out of the
# treated group and straight into the *never-treated* group, so the comparison
# group changes with every column. Measured on the run_20260722 held-out panels,
# pinning the control cut here is worth about half of the difference between the
# 1% and 5% columns:
#
#   Seattle, overall ATT, treated cut across / control cut down
#            >1%      >5%     >10%
#     <=1%  +0.089   +0.049   +0.092      <- control pinned (this constant)
#     <=5%     --    -0.001   +0.045
#     <=10%    --       --    +0.040      <- the old diagonal
#
# The old diagonal reads 1%: +0.089, 5%: -0.001, 10%: +0.040 and looks like the
# model failing at 5%; the pinned row is +0.089 / +0.049 / +0.092 against one
# fixed comparison group. The reason the contamination bites so much harder in
# Seattle than in Tampa is the shape of the dose-response: Seattle's raw
# tract-mean change steps up at ~2% of baseline area and is flat above it, so
# the 5% cut splits a homogeneous plateau into "treated" and "control" halves
# that move identically, whereas Tampa's is a smooth monotone gradient over the
# whole range and any cut separates it. See issue #36.
#
# 1% is the natural pin because it is the lowest cut in DEFAULT_THRESHOLDS, so
# the 1% column is unchanged by the restriction and the grid stays nested.
CONTROL_THRESHOLD: float = 0.01

# Event-time horizons tabulated in the coefficient table (issue #36 asks for
# post-ATTs at a couple of horizons, not one lumped number).
POST_HORIZONS: tuple[int, ...] = (0, 1, 2, 3)

# Below this many treated (or never-treated) tracts a dynamic event study is not
# worth plotting — the sup-t critical value blows up and the curve is noise.
MIN_TREATED_UNITS = 20
MIN_CONTROL_UNITS = 20

# Tracts whose UNDATED buildings exceed this share of baseline footprint area are
# dropped when the screen is enabled. An undated building is not ignored by
# `build_tract_cohorts` — "unknown year" is read as "standing at baseline", so its
# area is simultaneously missing from the numerator and inflating the denominator.
# Both push treatment intensity down, which is why the screen is on AREA and not
# on a count or a land-use class: undated buildings run about twice average size
# (measured undated area shares of 3.0% Tampa / 8.7% Seattle / 10.8% San Antonio
# against count shares roughly half those), so a count-based rule would not bound
# the quantity that actually biases the estimate.
#
# 0.20 is chosen from that same measurement: it drops 0.8% of Tampa's tracts,
# 8.9% of Seattle's and 11.2% of San Antonio's, where 0.05 would have dropped
# 12.5%/57.5%/58.9% and 0.10 still 3.7%/27.3%/30.9%. The screen exists so that a
# city whose assessor dates only residential improvements can be used at all; it
# must be reported as a sample restriction, and validated by comparing the
# restricted and unrestricted ATT wherever the unrestricted one is available.
MAX_UNDATED_AREA_SHARE = 0.20


# ═══════════════════════════════════════════════════════════════════════════════
# City registry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CityCohortSpec:
    """Where a city's construction cohorts come from.

    Attributes
    ----------
    key, label
        Short identifier and display name.
    geoid_prefixes
        County FIPS prefixes that define the city geography (tracts are
        restricted to these, so a whole-CBSA holdout can still be scoped to the
        sub-geography its footprints cover — e.g. Seattle = King County only).
    footprints_filename
        Parquet under ``processed_dir`` holding footprint polygons plus a
        year-built column.
    year_col, demolition_col
        Construction / demolition year columns in that parquet.
    id_index
        Name of the footprint id (the parquet's index) that building-level
        predictions join on, or ``None`` if predictions cannot be joined —
        which disables the incumbent-only outcome for that city.
    area_epsg
        Projected CRS used for footprint areas.
    panel_years
        The city's own panel grid, or ``None`` to inherit the caller's. Cadence
        is a *per-city* property and cannot be a global: NYC's zarr is biennial,
        Cook County ortho is annual 2009-2025, and each state's NAIP grid
        differs (IL 2011/12/14/15/17/19/21/23, WA 2011/13/15/17/19/21/23,
        FL 2010/13/15/17/19/21/23, TN 2012/14/16/18/21/23 — a four-way
        intersection of just {2021, 2023}). Predicting a year off-grid yields
        ``no_year_match`` -> NaN under ``predict_exact_year``, so a shared year
        list would silently empty most cities' panels.
    baseline_year
        Year whose building stock is the treatment-intensity denominator.
        ``None`` -> ``min(panel_years) - 1``, i.e. the last year before the
        panel opens, so the whole panel is post-baseline.
    sensor
        Which imagery the city's predictions come from: ``"zarr"`` (NYC legacy),
        ``"naip"``, or ``"ortho"`` (the city's own municipal orthoimagery — a
        sensor never seen in fine-tuning, which makes the holdout spatial *and*
        out-of-sensor; see :mod:`src.data.ortho_fetcher`).
    state
        State name for the ``buildings_index`` / ``buildings_polygons``
        partition, when the city's footprints come from the national index.
    tract_source
        How the city's tract set is defined. ``"prefix"`` uses
        ``geoid_prefixes``; ``"footprints"`` derives it from actual footprint
        coverage, which is what a sub-county source needs — Chicago's municipal
        footprint layer stops at the city line, well inside Cook County, so a
        FIPS prefix would pull in ~500 suburban tracts with no year-built data
        and no way to distinguish them from genuinely undeveloped ones.
    min_tract_footprints
        Under ``tract_source="footprints"``, the count below which a tract is
        treated as outside coverage rather than as a real tract with few
        buildings.
    note
        Free text surfaced when the city is unavailable.
    """

    key: str
    label: str
    geoid_prefixes: tuple[str, ...]
    footprints_filename: str
    year_col: str = "CONSTRUCTION_YEAR"
    demolition_col: str | None = "DEMOLITION_YEAR"
    id_index: str | None = None
    area_epsg: int = 5070
    panel_years: tuple[int, ...] | None = None
    baseline_year: int | None = None
    sensor: str = "naip"
    state: str | None = None
    tract_source: str = "prefix"
    min_tract_footprints: int = 50
    note: str = ""

    def years(self, fallback: list[int] | tuple[int, ...] | None = None) -> list[int]:
        """This city's panel years, falling back to the caller's grid."""
        if self.panel_years:
            return sorted(self.panel_years)
        if not fallback:
            raise ValueError(f"{self.key}: no panel_years and no fallback given")
        return sorted(fallback)

    def baseline(self, fallback: int | None = None) -> int:
        """Baseline year: explicit, else the year before the panel opens.

        Falls back to the caller's global only when the city has no panel of its
        own — a city with annual ortho from 2010 must not inherit NYC's 2009
        baseline, which would put a year of construction before the first
        observed period and mis-date its cohorts.
        """
        if self.baseline_year is not None:
            return int(self.baseline_year)
        if self.panel_years:
            return int(min(self.panel_years)) - 1
        if fallback is None:
            raise ValueError(f"{self.key}: no baseline_year and no fallback given")
        return int(fallback)


# NYC is wired; the other four are the whole-CBSA CSA holdouts from issue #36,
# declared here so they light up the moment their year-built join lands. The
# national buildings_index carries only (building_id, cx, cy, tract_id, state) —
# no year built — so they cannot run off existing artifacts.
CSA_CITIES: dict[str, CityCohortSpec] = {
    "nyc": CityCohortSpec(
        key="nyc",
        label="New York City",
        geoid_prefixes=("36005", "36047", "36061", "36081", "36085"),
        footprints_filename="buildings_nyc.parquet",
        year_col="CONSTRUCTION_YEAR",
        demolition_col="DEMOLITION_YEAR",
        id_index="DOITT_ID",
        # EPSG:5070 (Conus Albers), not the legacy NYC state plane 6539: treatment
        # intensity is a ratio of areas, and Albers is equal-area, while 6539 is
        # conformal (preserves shape, distorts area). It is also the CRS the tract
        # geometries are already stored in, and the one every other city uses.
        area_epsg=5070,
        panel_years=(2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024),
        baseline_year=2009,
        sensor="zarr",
        state="NewYork",
    ),
    "chicago": CityCohortSpec(
        key="chicago",
        label="Chicago",
        # Cook County only, and within it the tract set is derived from footprint
        # coverage (tract_source="footprints"): the City of Chicago footprint
        # layer — the one source covering ALL building types with a year — stops
        # at the city line. The other six CBSA counties have no year-built source
        # at all, so listing them would add tracts that can never be dated.
        geoid_prefixes=("17031",),
        footprints_filename="buildings_chicago.parquet",
        year_col="year_built",
        # No usable demolition data. The municipal layer's `demolished` column is
        # a sentinel: all 124 non-null values are the year 1899, and bldg_statu
        # has exactly 1 DEMOLISHED row in 820,606 — implausible for 15 years of
        # Chicago. Declaring it would date those 124 as an 1899 cohort.
        demolition_col=None,
        # Predictions are keyed on the Microsoft building_id, which
        # build_year_built recomputes onto the dated footprints — so unlike the
        # other holdout cities Chicago gets the composition-fixed
        # incumbent-only outcome as well as the all-buildings one.
        id_index="building_id",
        area_epsg=5070,
        # Cook County flies the whole county every year and publishes each year
        # as its own 4-band 6-inch ImageServer (2009-2025). That is why Chicago
        # runs on ortho rather than NAIP: 15 annual periods instead of Illinois
        # NAIP's 8, and a sensor the model never saw in fine-tuning.
        panel_years=tuple(range(2010, 2025)),
        baseline_year=2009,
        sensor="ortho",
        state="Illinois",
        tract_source="footprints",
        note="BLOCKED: the City of Chicago footprint layer is a frozen 2015 "
             "snapshot (year_built maxes at 2015, 46% zeros, no post-2015 "
             "polygons), so construction cohorts cannot be dated over the panel. "
             "The Cook ortho panel itself is fine — this needs a maintained "
             "year-built source (Cook Assessor char_yrblt covers only "
             "residential <7 units, which misses the towers). See issue #36.",
    ),
    "seattle": CityCohortSpec(
        key="seattle",
        label="Seattle (King County)",
        geoid_prefixes=("53033",),
        footprints_filename="buildings_king_county.parquet",
        year_col="year_built",
        demolition_col=None,
        id_index="building_id",
        area_epsg=5070,
        # King County publishes discrete flights (2007/2017/2021) rather than an
        # annual panel, so the ortho cadence is too sparse for an event study on
        # its own; Washington NAIP (odd years 2011-2023) is the panel, and the
        # ortho years are available as a cross-sensor check.
        # Stops at 2019 for the same reason as Tampa: cohorts come from
        # Microsoft footprints, which are a static ~2019 snapshot, so post-2019
        # construction is invisible and later panel years would put developed
        # tracts in the control group. Verify with the coverage report's
        # `footprint_coverage_cliff_year` before extending this.
        panel_years=(2011, 2013, 2015, 2017, 2019),
        baseline_year=2010,
        sensor="naip",
        state="Washington",
        note="build it with `python -m src.data.build_year_built seattle "
             "--state Washington` (King County Assessor residential + commercial "
             "building extracts joined on PIN to the county parcel layer, then "
             "to Microsoft footprints; issue #36). Panel years verified against "
             "the Planetary Computer NAIP inventory: Washington NAIP exists only "
             "in odd years 2011-2023, so the biennial cadence is the data.",
    ),
    "tampa": CityCohortSpec(
        key="tampa",
        # ASCII hyphens on purpose: labels go straight into figure titles, which
        # paper.mplstyle renders through LaTeX (text.usetex), where a raw
        # en-dash is not a valid character.
        label="Tampa-St. Petersburg",
        geoid_prefixes=("12057", "12103", "12101", "12053"),
        footprints_filename="buildings_fl_fgdl.parquet",
        year_col="year_built",
        demolition_col=None,
        id_index="building_id",
        area_epsg=5070,
        # Florida NAIP runs 2010/2013/2015/2017/2019/2021/2023, but the panel
        # STOPS AT 2019 because the treatment is only observable to then. The
        # Microsoft footprint universe is a static ~2019-vintage snapshot:
        # measured footprint coverage of Tampa construction is 0.91-1.00 for
        # 2010-2018, 0.68 in 2019, and 0.08-0.24 for 2020-2024 (86,658 parcels
        # built 2020+ have only 10,982 footprints). Extending the panel past
        # 2019 would leave tracts that developed after the snapshot looking
        # never-treated — control-group contamination that attenuates the ATT
        # toward zero, in the one direction that would make the model look
        # unresponsive. At 5% this still gives 340 treated / 442 never-treated.
        panel_years=(2010, 2013, 2015, 2017, 2019),
        baseline_year=2009,
        sensor="naip",
        state="Florida",
        note="build it with `python -m src.data.build_year_built tampa "
             "--state Florida` (Florida statewide cadastral ACT_YR_BLT joined to "
             "Microsoft footprints; issue #36)",
    ),
    "san_antonio": CityCohortSpec(
        key="san_antonio",
        label="San Antonio (Bexar County)",
        # Bexar alone, not the 8-county CBSA: Texas has no statewide parcel
        # programme and each of the other seven counties runs its own appraisal
        # district. Bexar is 2.01M of 2.61M, and the sub-CBSA scope is the same
        # compromise Seattle makes with King County. Unlike Chicago the coverage
        # edge is a COUNTY line, which is also a tract boundary, so a FIPS prefix
        # is exact and `tract_source="footprints"` is unnecessary.
        geoid_prefixes=("48029",),
        footprints_filename="buildings_bexar.parquet",
        year_col="year_built",
        demolition_col=None,
        id_index="building_id",
        area_epsg=5070,
        # Texas NAIP is biennial EVEN years (2012/2014/2016/2018/2020/2022,
        # verified against the Planetary Computer inventory over a San Antonio
        # bbox) — the opposite parity to Washington's odd-year grid, which is
        # exactly why panel cadence cannot be a global. Stops at 2018 for the
        # Tampa/Seattle/Baltimore reason: the static ~2019 Microsoft footprint
        # snapshot cannot see later construction, so 2020 and 2022 would put
        # developed tracts in the never-treated control group. Four periods makes
        # this the thinnest panel in the registry — check the coverage report's
        # `footprint_coverage_cliff_year` before trimming it further.
        panel_years=(2012, 2014, 2016, 2018),
        baseline_year=2011,
        sensor="naip",
        state="Texas",
        note="build it with `python -m src.data.build_year_built san_antonio "
             "--state Texas` (Bexar County BCAD roll joined to Microsoft "
             "footprints; issue #36). Carries the South Central region: with NYC, "
             "Baltimore, Tampa and Seattle the holdout set spans five census "
             "divisions. Best all-class coverage measured anywhere — 84% of "
             "commercial (F1) parcels dated, against Baltimore's 65%, "
             "Allegheny's 1.7% and Cook's 0%.",
    ),
    "baltimore": CityCohortSpec(
        key="baltimore",
        label="Baltimore",
        # The whole CBSA, not a core county: Anne Arundel, Baltimore County,
        # Carroll, Harford, Howard, Queen Anne's and Baltimore City. Maryland's
        # parcel layer is genuinely statewide, so unlike Chicago there is no
        # sub-county coverage edge and `tract_source="prefix"` is exact.
        geoid_prefixes=("24003", "24005", "24013", "24025", "24027", "24035",
                        "24510"),
        footprints_filename="buildings_md_mdp.parquet",
        year_col="year_built",
        # MD_ParcelBoundaries carries no demolition column. Unlike Chicago's
        # `demolished` — which is a 1899 sentinel masquerading as data — this is
        # an honest absence, so the baseline stock is simply never decremented.
        demolition_col=None,
        id_index="building_id",
        area_epsg=5070,
        # Maryland NAIP on the Planetary Computer is 2011/2013/2015/2017/2018/
        # 2021/2023 (verified against the STAC inventory over a Baltimore bbox).
        # The panel stops before 2021 for the Tampa/Seattle reason, not a NAIP
        # one: cohorts come from the static ~2019-vintage Microsoft footprint
        # snapshot, so 2021 and 2023 would put post-snapshot development in the
        # never-treated control group and attenuate the ATT toward zero. Confirm
        # against the coverage report's `footprint_coverage_cliff_year` before
        # extending — that diagnostic exists precisely to set this bound.
        #
        # 2018 is DROPPED even though the imagery exists, because it is one year
        # after 2017 while every other gap in this panel is two. That single
        # irregular gap does real damage:
        #   * the 2018 cohort is "first crossed the threshold between 2017 and
        #     2018", a ONE-year window against everyone else's two, so it is
        #     mechanically about half the size — 9 / 4 / 2 tracts at the 1 / 5 /
        #     10% thresholds;
        #   * and because csa's varying base period needs t >= 2, that cohort is
        #     the ONLY one that can be observed at event time k = -3. The
        #     measured dynamic ATT at k = -3 is a single ATT(g,t) cell with
        #     weight pge = 1.0000 at every threshold — two tracts carrying a
        #     plotted coefficient and a pre-trend test at the 10% cut.
        # Dropping it moves the sup-t pre-trend p from 0.0005 to 0.018 (1%) and
        # from 0.0000 to 0.047 (10%), while the post-treatment ATT is unchanged
        # (0.086 -> 0.087 at the headline 5%). The 2018 flight is fine; a
        # one-year period on a biennial panel is not. See issue #36.
        panel_years=(2011, 2013, 2015, 2017),  # 2018 dropped: see above
        baseline_year=2010,
        sensor="naip",
        state="Maryland",
        note="build it with `python -m src.data.build_year_built baltimore "
             "--state Maryland` (Maryland statewide parcel YEARBLT joined to "
             "Microsoft footprints; issue #36). Chosen over the other test-split "
             "holdouts because its year built covers ALL property classes: "
             "18,870/29,147 commercial parcels dated (64.7%) against 1.7% for "
             "Allegheny/Pittsburgh and 0% for Cook, and 34,327 parcels built "
             "2012-2018 against 394 for Chicago's whole 2010-2015.",
    ),
    "nashville": CityCohortSpec(
        key="nashville",
        label="Nashville-Davidson",
        geoid_prefixes=("47037", "47149", "47189", "47165", "47187"),
        footprints_filename="buildings_tn_tnmap.parquet",
        year_col="year_built",
        demolition_col=None,
        id_index="building_id",
        area_epsg=5070,
        # Stops at 2018: same Microsoft-snapshot limit as Tampa and Seattle.
        panel_years=(2012, 2014, 2016, 2018),
        baseline_year=2011,
        # NAIP, not local ortho: TNMap's BASEMAPS/IMAGERY is a *current* mosaic
        # refreshed county-by-county as TDOT flies one region per year, not a
        # year-indexed archive, so there is no local time series to build a
        # panel from.
        sensor="naip",
        state="Tennessee",
        note="BLOCKED: no public bulk year-built source covers Davidson County. "
             "The TN Comptroller publishes per-county assessment tables, but only "
             "for the 86 counties in the state IMPACT CAMA system — Davidson, "
             "Rutherford and Williamson run their own and are excluded, which is "
             "3 of the 5 CBSA counties including the core. Metro Nashville's own "
             "parcel service carries ownership, land use and appraised values but "
             "no construction year, and its building-permits layer is a rolling "
             "3-year window that cannot reach a 2012-2018 panel. Unblocking is "
             "administrative: a PIN-keyed year extract from the Davidson County "
             "Assessor needs no new code — it is the same `year_from` table join "
             "Seattle uses. See src/data/parcel_sources.py and issue #36.",
    ),
}

# Cities that carry the main-body figure, in order: the deep-dive city plus a
# clean zero-shot one. Chicago was the intended second city (annual county ortho,
# Midwest mega) but its only all-building dated-footprint source — the City of
# Chicago layer — is a frozen 2015 snapshot (rowsUpdatedAt == createdAt ==
# 2015-08-14): year_built maxes at 2015, 46% of rows are 0, and post-2015
# buildings have no polygon at all, so essentially no tract crosses even the 1%
# threshold over a 2010-2024 panel. Tampa takes the slot: Florida's statewide
# cadastral carries ACT_YR_BLT for ALL property classes and is refreshed from the
# county appraisers' annual roll (verified carrying 2021/2022 construction).
MAIN_FIGURE_CITIES: tuple[str, ...] = ("nyc", "tampa")


def available_cities(processed_dir: Path,
                     keys: list[str] | tuple[str, ...] | None = None,
                     ) -> tuple[list[CityCohortSpec], list[tuple[CityCohortSpec, str]]]:
    """Split the registry into (runnable, [(spec, reason_unavailable), ...]).

    A city is runnable when its footprints parquet exists under
    ``processed_dir``. Nothing is read here — this is a cheap gate so the caller
    can report the skipped cities once instead of failing mid-estimation.
    """
    wanted = list(CSA_CITIES) if keys is None else [k for k in keys]
    ready: list[CityCohortSpec] = []
    missing: list[tuple[CityCohortSpec, str]] = []
    for key in wanted:
        spec = CSA_CITIES.get(key)
        if spec is None:
            continue
        path = Path(processed_dir) / spec.footprints_filename
        if path.exists():
            ready.append(spec)
        else:
            reason = f"{spec.footprints_filename} not found"
            if spec.note:
                reason += f" — {spec.note}"
            missing.append((spec, reason))
    return ready, missing


# ═══════════════════════════════════════════════════════════════════════════════
# Period indexing
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PeriodMap:
    """Bijection between the panel's calendar years and 1..T period indices.

    ``csa`` reserves ``0`` for never-treated units, so periods are 1-based and a
    cohort index can never collide with the sentinel. Using an explicit ordered
    map (rather than ``(year - base) // step``) is what makes the same code work
    for biennial, annual and irregular panels.
    """

    years: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.years) != len(set(self.years)):
            raise ValueError("PeriodMap years must be unique")
        if list(self.years) != sorted(self.years):
            raise ValueError("PeriodMap years must be sorted ascending")
        if not self.years:
            raise ValueError("PeriodMap needs at least one year")

    @classmethod
    def from_years(cls, years) -> "PeriodMap":
        return cls(tuple(sorted({int(y) for y in years})))

    @property
    def n_periods(self) -> int:
        return len(self.years)

    def period(self, year: int) -> int:
        """1-based period index of ``year``; raises if absent from the panel."""
        return self.years.index(int(year)) + 1

    def year(self, period: int) -> int:
        if not 1 <= int(period) <= self.n_periods:
            raise ValueError(f"period {period} outside 1..{self.n_periods}")
        return self.years[int(period) - 1]

    def period_series(self, years: pd.Series) -> pd.Series:
        """Vectorised :meth:`period`; unmapped years become NaN."""
        lookup = {y: i + 1 for i, y in enumerate(self.years)}
        return years.astype(int).map(lookup)

    def event_time_years(self, k: int) -> float:
        """Approximate calendar length of ``k`` periods (for axis labelling)."""
        if self.n_periods < 2:
            return float(k)
        spacing = np.diff(np.asarray(self.years, dtype=float)).mean()
        return float(k) * spacing


# ═══════════════════════════════════════════════════════════════════════════════
# Cohort construction
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CohortResult:
    """Per-tract construction cohorts plus the diagnostics behind them."""

    cohorts: pd.DataFrame        # GEOID_str, base_area, new_area, cohort_year
    shares: pd.DataFrame         # GEOID_str, year, cum_new_area, share (long)
    n_tracts_in: int = 0
    n_dropped_no_baseline: int = 0
    n_treated: int = 0
    n_never_treated: int = 0
    threshold: float = HEADLINE_THRESHOLD
    # Undated-area screen (see MAX_UNDATED_AREA_SHARE). `None` means not applied,
    # which is NOT the same as applied-and-dropped-nothing: the distinction is
    # what makes the restricted/unrestricted comparison legible in a table.
    max_undated_area_share: float | None = None
    n_dropped_undated: int = 0
    undated_area_share: pd.Series | None = None   # per tract, all tracts scored
    # Control cut (see CONTROL_THRESHOLD). `None` means the never-treated group
    # is "everything that did not cross `threshold`", which makes the comparison
    # group move with the threshold; a float pins it and drops the tracts whose
    # final share falls in (control_threshold, threshold].
    control_threshold: float | None = None
    n_dropped_ambiguous: int = 0

    def summary(self) -> dict:
        return {
            "threshold": self.threshold,
            "control_threshold": self.control_threshold,
            "n_tracts": int(self.n_tracts_in),
            "n_dropped_no_baseline": int(self.n_dropped_no_baseline),
            "max_undated_area_share": self.max_undated_area_share,
            "n_dropped_undated": int(self.n_dropped_undated),
            "n_dropped_ambiguous": int(self.n_dropped_ambiguous),
            "n_treated": int(self.n_treated),
            "n_never_treated": int(self.n_never_treated),
        }


def _check_reprojection(gdf, label: str, epsg: int) -> None:
    """Fail loudly if a reprojection produced non-finite coordinates.

    PROJ returns ``inf`` — rather than raising — when it cannot build a
    transformation pipeline (e.g. ``PROJ_NETWORK=ON`` with no route to the grid
    CDN, which is the situation inside a network-restricted sandbox). Silent
    ``inf`` geometry then yields an empty spatial join, which downstream looks
    exactly like "this city has no buildings": every tract gets dropped for an
    undefined baseline and the event study reports 0 treated / 0 control units
    with no error anywhere. Checking here converts that into a clear message.
    """
    bounds = gdf.geometry.bounds.to_numpy()
    if len(bounds) and not np.isfinite(bounds).all():
        n_bad = int((~np.isfinite(bounds).all(axis=1)).sum())
        raise RuntimeError(
            f"reprojecting {label} to EPSG:{epsg} produced non-finite coordinates "
            f"for {n_bad:,}/{len(bounds):,} geometries. PROJ could not build the "
            f"transformation pipeline — if PROJ_NETWORK=ON and this machine has no "
            f"access to the PROJ grid CDN, set PROJ_NETWORK=OFF and retry."
        )


def _tract_of_footprints(footprints, tracts, area_epsg: int) -> pd.DataFrame:
    """Assign each footprint to a tract by centroid, returning areas.

    ``footprints`` / ``tracts`` are GeoDataFrames; both are reprojected to
    ``area_epsg`` so polygon areas and the point-in-polygon test share one CRS.
    Returns a plain DataFrame (GEOID_str, area, year_built, demolition_year).
    """
    import geopandas as gpd

    fp = footprints.to_crs(area_epsg)
    tr = tracts.to_crs(area_epsg)[["GEOID_str", "geometry"]]
    _check_reprojection(fp, "building footprints", area_epsg)
    _check_reprojection(tr, "tract geometries", area_epsg)

    areas = fp.geometry.area.to_numpy()
    centroids = gpd.GeoDataFrame(
        {"area": areas},
        geometry=fp.geometry.centroid,
        crs=fp.crs,
        index=fp.index,
    )
    for col in ("year_built", "demolition_year"):
        if col in fp.columns:
            centroids[col] = fp[col].to_numpy()

    joined = gpd.sjoin(centroids, tr, how="inner", predicate="within")
    if len(centroids) and len(tr) and joined.empty:
        raise RuntimeError(
            f"no footprint centroid fell inside any of the {len(tr):,} tracts "
            f"(EPSG:{area_epsg}). The two layers do not overlap — check that the "
            f"footprint file covers this city's GEOID prefixes."
        )
    keep = ["GEOID_str", "area"] + [
        c for c in ("year_built", "demolition_year") if c in joined.columns
    ]
    return pd.DataFrame(joined[keep])


def footprint_tract_areas(
    footprints,
    tracts,
    year_col: str = "CONSTRUCTION_YEAR",
    demolition_col: str | None = None,
    area_epsg: int = 5070,
) -> pd.DataFrame:
    """Per-building ``(GEOID_str, area, year_built, demolition_year)`` table.

    This is the only geometry-bound step in :func:`build_tract_cohorts`, and by
    far its most expensive: it reprojects every footprint polygon to
    ``area_epsg``, takes areas and centroids, and point-in-polygon joins those
    centroids to tracts. For a city like NYC that is ~1.1M polygons reprojected
    twice over (polygon + centroid) — around a gigabyte of transient GEOS
    allocation per call.

    None of it depends on ``threshold``, ``max_undated_area_share`` or
    ``control_threshold``, so it is exposed separately: a caller sweeping a
    threshold grid — or re-running the same city under several anticipation arms
    — computes this once and passes the result to ``build_tract_cohorts`` via
    ``per_building``. The returned frame is a few tens of MB and holds no
    geometry, so it can be kept alive while the footprint GeoDataFrame is
    released.
    """
    fp = footprints
    cols = [c for c in (year_col, demolition_col) if c and c in fp.columns]
    # Carry only geometry + the two year columns into the reprojection; a full
    # `.copy()` of a wide footprint table drags every unused attribute column
    # through `to_crs` for nothing.
    fp = fp[cols + [fp.geometry.name]].copy()
    fp["year_built"] = pd.to_numeric(fp[year_col], errors="coerce")
    if demolition_col is not None and demolition_col in fp.columns:
        fp["demolition_year"] = pd.to_numeric(fp[demolition_col], errors="coerce")
    return _tract_of_footprints(fp, tracts, area_epsg)


def build_tract_cohorts(
    footprints,
    tracts,
    panel_years,
    baseline_year: int,
    threshold: float = HEADLINE_THRESHOLD,
    year_col: str = "CONSTRUCTION_YEAR",
    demolition_col: str | None = None,
    area_epsg: int = 5070,
    min_baseline_area: float = 0.0,
    max_undated_area_share: float | None = None,
    control_threshold: float | None = None,
    per_building: pd.DataFrame | None = None,
) -> CohortResult:
    """Tract-level construction cohorts from footprint polygons + year built.

    Treatment intensity is the cumulative area of buildings constructed after
    ``baseline_year``, as a share of the tract's *baseline* building area
    (everything standing at ``baseline_year``). A tract's cohort is the first
    panel year at which that share exceeds ``threshold``; ``0`` marks
    never-treated.

    Tracts whose baseline area is ``<= min_baseline_area`` (typically 0 — no
    pre-period building stock) have an undefined share and are **dropped**, not
    recorded as controls.

    With ``control_threshold`` set, "never-treated" stops meaning "did not cross
    ``threshold``" and becomes "did not cross ``control_threshold``"; the tracts
    in between are dropped from the sample instead of being handed to the
    control group. See :data:`CONTROL_THRESHOLD`.

    Parameters
    ----------
    footprints
        GeoDataFrame of footprint polygons with ``year_col`` (and optionally
        ``demolition_col``).
    tracts
        GeoDataFrame with ``GEOID_str`` and tract geometry.
    panel_years
        Years the outcome is observed in — cohorts can only be dated to these.
    baseline_year
        Last pre-treatment year: buildings built ``<= baseline_year`` form the
        denominator; anything later is potential treatment.
    max_undated_area_share
        Drop tracts whose *undated* buildings exceed this share of baseline
        footprint area; ``None`` disables the screen. See
        :data:`MAX_UNDATED_AREA_SHARE`. This exists for cities whose assessor
        records a construction year only for residential improvements — there the
        undated area is concentrated in commercial and large multifamily stock,
        and the intensity of an unscreened tract is biased toward zero in both
        the numerator and the denominator at once.
    control_threshold
        Share below which a tract is a valid never-treated control. Must not
        exceed ``threshold``. ``None`` (the default) reproduces the old
        behaviour, where the control group is the complement of the treated
        group and therefore changes whenever ``threshold`` does.
    per_building
        A pre-computed :func:`footprint_tract_areas` table. Supplying it skips
        the reprojection / centroid / spatial join entirely, which is the whole
        cost of this function and does not depend on any of the sample-design
        arguments; ``footprints`` is then unused and may be ``None``. Callers
        that sweep the threshold grid should build it once — see
        :func:`footprint_tract_areas`.
    """
    years = sorted({int(y) for y in panel_years})
    if not years:
        raise ValueError("panel_years is empty")
    if baseline_year >= years[-1]:
        raise ValueError(
            f"baseline_year {baseline_year} leaves no post-baseline panel year"
        )
    if control_threshold is not None and control_threshold > threshold:
        raise ValueError(
            f"control_threshold {control_threshold} exceeds threshold {threshold}: "
            "the control cut must sit at or below the treated cut, or treated and "
            "never-treated tracts overlap"
        )

    per_bldg = (
        footprint_tract_areas(footprints, tracts, year_col=year_col,
                              demolition_col=demolition_col, area_epsg=area_epsg)
        if per_building is None else per_building
    )
    tract_ids = sorted(tracts["GEOID_str"].astype(str).unique())

    yb = per_bldg["year_built"]
    demolished_pre = (
        per_bldg["demolition_year"].notna()
        & (per_bldg["demolition_year"] <= baseline_year)
        if "demolition_year" in per_bldg.columns
        else pd.Series(False, index=per_bldg.index)
    )
    # A CONSTRUCTION_YEAR of 0 / NaN means "unknown", not "ancient": such
    # buildings cannot date a cohort, but they are part of the standing stock,
    # so they count toward the baseline denominator.
    is_new = yb.notna() & (yb > baseline_year) & (yb <= years[-1])
    is_base = ~is_new & ~demolished_pre

    base_area = (
        per_bldg.loc[is_base].groupby("GEOID_str")["area"].sum()
        .reindex(tract_ids).fillna(0.0)
    )
    # Undated buildings are a SUBSET of the baseline by construction: an unknown
    # year cannot satisfy `yb > baseline_year`, so `is_new` is False and the
    # building lands in `is_base`. Scoring the share here — for every tract,
    # whether or not the screen is enabled — is what lets the caller report the
    # restriction rather than merely apply it.
    undated_base_area = (
        per_bldg.loc[is_base & yb.isna()].groupby("GEOID_str")["area"].sum()
        .reindex(tract_ids).fillna(0.0)
    )
    undated_share = undated_base_area / base_area.replace(0.0, np.nan)

    new = per_bldg.loc[is_new, ["GEOID_str", "area", "year_built"]]

    # Cumulative new area *as visible at each panel year*: a building finished
    # in 2011 shows up in the 2012 image, so a biennial panel credits it to
    # 2012. Attributing by "<= panel year" is what makes this cadence-agnostic.
    #
    # Each new building is credited to exactly one panel year — the earliest one
    # at or after its completion — so the whole tract x year cumulation is a
    # scatter-add into a (tract, year) matrix followed by a cumulative sum along
    # the year axis. `is_new` already guarantees baseline_year < year_built <=
    # years[-1], so every building lands inside the grid. The row-wise loop this
    # replaces re-derived `new_by_year.index.get_level_values(0)` once per tract,
    # making it quadratic in the number of tracts and allocating an array the
    # size of the whole tract x year table on every iteration.
    year_arr = np.asarray(years)
    tract_pos = {t: i for i, t in enumerate(tract_ids)}
    cum_new = np.zeros((len(tract_ids), len(years)), dtype=float)
    if len(new):
        rows_i = new["GEOID_str"].map(tract_pos).to_numpy()
        cols_i = np.searchsorted(year_arr, new["year_built"].to_numpy(),
                                 side="left")
        # A footprint whose centroid fell in a tract outside `tract_ids` cannot
        # be placed; dropping it here matches the old loop, which only ever
        # visited tracts in `tract_ids`.
        ok = pd.notna(rows_i) & (cols_i < len(years))
        np.add.at(cum_new,
                  (rows_i[ok].astype(np.intp), cols_i[ok].astype(np.intp)),
                  new["area"].to_numpy(dtype=float)[ok])
    cum_new = np.cumsum(cum_new, axis=1)
    shares = pd.DataFrame({
        "GEOID_str": np.repeat(np.asarray(tract_ids, dtype=object), len(years)),
        "year": np.tile(year_arr, len(tract_ids)),
        "cum_new_area": cum_new.ravel(),
    })
    shares = shares.merge(
        base_area.rename("base_area"), left_on="GEOID_str", right_index=True, how="left"
    )

    n_in = len(tract_ids)
    undefined = shares["base_area"] <= min_baseline_area
    dropped_ids = sorted(shares.loc[undefined, "GEOID_str"].unique())
    shares = shares[~undefined].copy()

    n_dropped_undated = 0
    if max_undated_area_share is not None:
        over = (shares["GEOID_str"].map(undated_share).fillna(0.0)
                > float(max_undated_area_share))
        n_dropped_undated = int(shares.loc[over, "GEOID_str"].nunique())
        shares = shares[~over].copy()

    shares["share"] = shares["cum_new_area"] / shares["base_area"]

    # Drop the ambiguous middle. `cum_new_area` is non-decreasing over the
    # panel, so a tract's final share is its max; a tract sitting in
    # (control_threshold, threshold] never becomes treated at this cut yet did
    # build, and handing it to the control group is what makes the threshold
    # grid move its own comparison group. Done AFTER the undated screen so the
    # two restrictions compose in a fixed order and their counts stay additive.
    n_dropped_ambiguous = 0
    if control_threshold is not None and len(shares):
        final_share = shares.groupby("GEOID_str")["share"].max()
        ambiguous = final_share.index[
            (final_share > float(control_threshold)) & (final_share <= threshold)
        ]
        n_dropped_ambiguous = int(len(ambiguous))
        if n_dropped_ambiguous:
            shares = shares[~shares["GEOID_str"].isin(set(ambiguous))].copy()

    treated = shares[shares["share"] > threshold]
    cohort_year = (
        treated.groupby("GEOID_str")["year"].min() if len(treated)
        else pd.Series(dtype=int)
    )

    cohorts = (
        shares.groupby("GEOID_str")
        .agg(base_area=("base_area", "first"), new_area=("cum_new_area", "max"))
        .reset_index()
    )
    cohorts["cohort_year"] = (
        cohorts["GEOID_str"].map(cohort_year).fillna(0).astype(int)
    )

    return CohortResult(
        cohorts=cohorts,
        shares=shares.reset_index(drop=True),
        n_tracts_in=n_in,
        n_dropped_no_baseline=len(dropped_ids),
        n_treated=int((cohorts["cohort_year"] > 0).sum()),
        n_never_treated=int((cohorts["cohort_year"] == 0).sum()),
        threshold=float(threshold),
        max_undated_area_share=(None if max_undated_area_share is None
                                else float(max_undated_area_share)),
        n_dropped_undated=n_dropped_undated,
        undated_area_share=undated_share,
        control_threshold=(None if control_threshold is None
                           else float(control_threshold)),
        n_dropped_ambiguous=n_dropped_ambiguous,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Outcomes
# ═══════════════════════════════════════════════════════════════════════════════

def tract_outcomes(
    bld_preds: pd.DataFrame,
    construction_year: pd.Series | None = None,
    baseline_year: int | None = None,
    pred_col: str = "predicted_value",
    id_col: str = "building_id",
) -> pd.DataFrame:
    """Tract-year outcome means from building-level predictions.

    Returns long ``GEOID_str, year, pred_all, n_all`` and — when
    ``construction_year`` (indexed by building id) and ``baseline_year`` are
    given — ``pred_incumbent, n_incumbent``: the same mean restricted to
    buildings that already existed at ``baseline_year``.

    The incumbent-only mean is the composition-constant outcome. The
    all-buildings mean moves partly by construction (a new, high-scoring
    building joins the average); the incumbent mean can only move if the model
    re-reads buildings that did not themselves change.
    """
    df = bld_preds.copy()
    if "GEOID_str" not in df.columns:
        df["GEOID_str"] = df["GEOID"].astype(str).str.zfill(11)
    df["year"] = df["year"].astype(int)
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    df = df[np.isfinite(df[pred_col])]

    out = (
        df.groupby(["GEOID_str", "year"])[pred_col]
        .agg(pred_all="mean", n_all="size")
        .reset_index()
    )

    if construction_year is not None and baseline_year is not None and id_col in df.columns:
        yb = df[id_col].map(construction_year)
        # Unknown year built (NaN / 0 sentinel) is treated as incumbent: these
        # are old buildings with missing records, not post-baseline construction.
        incumbent = ~(yb.notna() & (yb > baseline_year))
        inc = (
            df[incumbent].groupby(["GEOID_str", "year"])[pred_col]
            .agg(pred_incumbent="mean", n_incumbent="size")
            .reset_index()
        )
        out = out.merge(inc, on=["GEOID_str", "year"], how="left")

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Event panel
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EventPanel:
    """A balanced, ``csa``-ready panel plus the bookkeeping to report on it."""

    data: pd.DataFrame           # unit_id, period, cohort, outcome, GEOID_str
    period_map: PeriodMap
    outcome_col: str
    n_treated: int = 0
    n_never_treated: int = 0
    dropped_unbalanced: int = 0
    dropped_first_period_cohort: int = 0
    dropped_no_cohort: int = 0
    anticipation: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return (
            not self.data.empty
            and self.n_treated >= MIN_TREATED_UNITS
            and self.n_never_treated >= MIN_CONTROL_UNITS
        )

    def summary(self) -> dict:
        return {
            "n_units": int(self.data["unit_id"].nunique()) if len(self.data) else 0,
            "n_periods": self.period_map.n_periods,
            "n_treated": int(self.n_treated),
            "n_never_treated": int(self.n_never_treated),
            "dropped_unbalanced": int(self.dropped_unbalanced),
            "dropped_first_period_cohort": int(self.dropped_first_period_cohort),
            "dropped_no_cohort": int(self.dropped_no_cohort),
            "anticipation": int(self.anticipation),
        }


def build_event_panel(
    outcomes: pd.DataFrame,
    cohorts: pd.DataFrame,
    outcome_col: str = "pred_all",
    panel_years=None,
    anticipation: int = 0,
) -> EventPanel:
    """Turn tract-year outcomes + cohorts into a balanced ``csa`` panel.

    Steps, each of which the previous implementation got wrong or skipped:

    1. Restrict to ``panel_years`` (default: years present in ``outcomes``) and
       to tracts that have a cohort (tracts dropped for an undefined baseline
       never reach ``csa``).
    2. **Balance** the panel — ``csa.estimate(balanced=True)`` assumes it.
    3. Map years to 1-based periods through :class:`PeriodMap`, so ``cohort = 0``
       unambiguously means never-treated.
    4. Drop cohorts dated to the *first* period: they have no pre-period, so
       ``csa`` has no base year for them. (The old code instead folded them into
       the never-treated control group.)

    Parameters
    ----------
    anticipation
        Periods of anticipation to allow, moving each treated unit's effective
        adoption date that many periods *earlier*. This is the standard
        Callaway–Sant'Anna anticipation allowance, and it matters here for a
        concrete measurement reason: the cohort is dated from ``year_built``,
        which records **completion**, whereas demolition, excavation and
        superstructure are visible in aerial imagery well before that. With
        ``anticipation=0`` those pre-completion periods sit in the pre-treatment
        window and will fail a pre-trend test even when parallel trends hold.
        Event time ``k=0`` then refers to ``anticipation`` periods before
        completion. Units left without a pre-period after the shift are dropped.
    """
    df = outcomes.copy()
    if outcome_col not in df.columns:
        raise KeyError(f"outcome column {outcome_col!r} missing from outcomes")
    df["year"] = df["year"].astype(int)
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce")
    df = df[np.isfinite(df[outcome_col])]

    years = (
        sorted({int(y) for y in panel_years}) if panel_years is not None
        else sorted(df["year"].unique().tolist())
    )
    df = df[df["year"].isin(years)]
    pmap = PeriodMap.from_years(years)

    coh = cohorts[["GEOID_str", "cohort_year"]].drop_duplicates("GEOID_str")
    before = df["GEOID_str"].nunique()
    df = df.merge(coh, on="GEOID_str", how="inner")
    dropped_no_cohort = before - df["GEOID_str"].nunique()

    # 2. balance
    counts = df.groupby("GEOID_str")["year"].nunique()
    full = set(counts[counts == len(years)].index)
    dropped_unbalanced = int(counts.size - len(full))
    df = df[df["GEOID_str"].isin(full)].copy()

    if df.empty:
        return EventPanel(
            data=df.assign(unit_id=[], period=[], cohort=[]),
            period_map=pmap, outcome_col=outcome_col,
            dropped_unbalanced=dropped_unbalanced,
            dropped_no_cohort=int(dropped_no_cohort),
            notes=["no balanced tract-years"],
        )

    # 3. period indexing
    df["period"] = pmap.period_series(df["year"]).astype(int)
    cohort_period = df["cohort_year"].map(
        lambda y: pmap.period(y) if int(y) in pmap.years else 0
    )
    df["cohort"] = np.where(df["cohort_year"] > 0, cohort_period, 0).astype(int)
    # Who was treated is decided BEFORE the shift and carried through it. Asking
    # `cohort > 0` afterwards silently reclassifies: a period-1 cohort shifted by
    # 1 lands on 0, which is csa's never-treated sentinel, so a treated tract
    # becomes a control — the same failure this function was written to stop for
    # first-period cohorts, one step later in the pipeline. Shifted by 2 it lands
    # on -1 and is neither treated nor control, and a negative group reaches csa.
    was_treated = df["cohort"] > 0
    if anticipation:
        df["cohort"] = np.where(
            was_treated, df["cohort"] - int(anticipation), 0
        ).astype(int)

    # 4. cohorts in (or before) the first period have no pre-period to serve as
    #    csa's base year. An anticipation shift can push a cohort to <= 1, and to
    #    <= 0, so the test is on the shifted value for anything that was treated.
    first_period = 1
    bad = was_treated & (df["cohort"] <= first_period)
    dropped_first = int(df.loc[bad, "GEOID_str"].nunique())
    df = df[~bad].copy()

    df["unit_id"] = pd.factorize(df["GEOID_str"])[0]
    n_treated = int(df.loc[df["cohort"] > 0, "GEOID_str"].nunique())
    n_never = int(df.loc[df["cohort"] == 0, "GEOID_str"].nunique())

    notes = []
    if n_treated < MIN_TREATED_UNITS:
        notes.append(f"only {n_treated} treated tracts (< {MIN_TREATED_UNITS})")
    if n_never < MIN_CONTROL_UNITS:
        notes.append(f"only {n_never} never-treated tracts (< {MIN_CONTROL_UNITS})")

    return EventPanel(
        data=df[["GEOID_str", "unit_id", "period", "cohort", outcome_col]].reset_index(drop=True),
        period_map=pmap,
        outcome_col=outcome_col,
        n_treated=n_treated,
        n_never_treated=n_never,
        dropped_unbalanced=dropped_unbalanced,
        dropped_first_period_cohort=dropped_first,
        dropped_no_cohort=int(dropped_no_cohort),
        anticipation=int(anticipation),
        notes=notes,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Estimation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EventStudyResult:
    """Dynamic ATT path with simultaneous CIs and a pre-trend falsification test."""

    event_times: np.ndarray
    att: np.ndarray
    se: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    n_units: int
    n_treated: int
    n_never_treated: int
    overall_att: float
    overall_se: float
    simple_att: float | None = None
    simple_se: float | None = None
    pretrend_supt_p: float | None = None
    pretrend_wald_p: float | None = None
    pretrend_wald_stat: float | None = None
    pretrend_df: int | None = None
    pretrend_max_abs_t: float | None = None
    control: str = "never"
    method: str = "reg"
    anticipation: int = 0
    label: str = ""

    def post_att(self, k: int) -> tuple[float, float] | tuple[None, None]:
        """(ATT, se) at event time ``k``, or (None, None) if not estimated."""
        hit = np.flatnonzero(self.event_times == k)
        if not hit.size:
            return None, None
        i = int(hit[0])
        return float(self.att[i]), float(self.se[i])

    def to_row(self, horizons=POST_HORIZONS) -> dict:
        row = {
            "label": self.label,
            "control": self.control,
            "anticipation": self.anticipation,
            "n_units": self.n_units,
            "n_treated": self.n_treated,
            "n_never_treated": self.n_never_treated,
            "pretrend_joint_p": self.pretrend_supt_p,
            "pretrend_wald_p": self.pretrend_wald_p,
            "pretrend_max_abs_t": self.pretrend_max_abs_t,
            "overall_att": self.overall_att,
            "overall_se": self.overall_se,
            "simple_att": self.simple_att,
            "simple_se": self.simple_se,
        }
        for k in horizons:
            att, se = self.post_att(k)
            row[f"att_k{k}"] = att
            row[f"se_k{k}"] = se
        return row


def pretrend_test(agg, event_times: np.ndarray, att: np.ndarray,
                  n: int) -> dict:
    """Joint test that every pre-treatment ATT is zero.

    Two statistics, both built from ``csa``'s own multiplier-bootstrap draws so
    they are consistent with the simultaneous CIs the figure plots:

    * **sup-t** (primary) — the studentised maximum over pre-period event times,
      compared against its bootstrap distribution. This is the test that matches
      a uniform confidence band: if the band covers zero everywhere before
      treatment, this p-value is large. Robust to a singular covariance, which
      is common here because the dynamic ATTs are weighted averages of a small
      number of ATT(g,t) cells.
    * **Wald χ²** (secondary) — quadratic form in the bootstrap covariance,
      pseudo-inverted with the degrees of freedom set to its numerical rank.

    Returns a dict of ``supt_p, wald_p, wald_stat, df, max_abs_t`` (values
    ``None`` when there are no pre-periods to test).
    """
    empty = {"supt_p": None, "wald_p": None, "wald_stat": None,
             "df": None, "max_abs_t": None}
    pre = np.flatnonzero(np.asarray(event_times) < 0)
    if pre.size == 0:
        return empty

    boot = getattr(agg, "boot", None)
    aggregate = getattr(boot, "aggregate", None) if boot is not None else None
    if aggregate is None:
        return empty

    samples = np.asarray(aggregate.samples)          # (B, K), sqrt(n)-scaled
    se = np.asarray(aggregate.se)                    # (K,) = b_sigma / sqrt(n)
    if samples.ndim != 2 or samples.shape[1] != len(event_times):
        return empty

    theta = np.asarray(att, dtype=float)[pre]
    se_pre = se[pre]
    good = np.isfinite(theta) & np.isfinite(se_pre) & (se_pre > 0)
    if not good.any():
        return empty
    idx = pre[good]
    theta = np.asarray(att, dtype=float)[idx]
    se_pre = se[idx]

    # sup-t: studentise the draws by the same scale that produced `se`.
    b_sigma = se_pre * np.sqrt(n)
    draws = samples[:, idx] / b_sigma[None, :]
    obs_t = np.abs(theta / se_pre)
    max_obs = float(np.nanmax(obs_t))
    boot_max = np.nanmax(np.abs(draws), axis=1)
    supt_p = float(np.mean(boot_max >= max_obs))

    # Wald from the bootstrap covariance of the sqrt(n)-scaled draws.
    wald_p = wald_stat = df_used = None
    try:
        from scipy import stats as _stats

        V = np.asarray(aggregate.V)
        V_pre = V[np.ix_(idx, idx)] / float(n)
        # Rank via eigenvalues of the symmetrised matrix; tiny eigenvalues are
        # numerical noise, not information.
        V_pre = 0.5 * (V_pre + V_pre.T)
        evals = np.linalg.eigvalsh(V_pre)
        tol = max(evals.max(), 0.0) * 1e-8
        rank = int((evals > tol).sum())
        if rank > 0:
            stat = float(theta @ np.linalg.pinv(V_pre, rcond=1e-8) @ theta)
            wald_stat = stat
            df_used = rank
            wald_p = float(_stats.chi2.sf(stat, rank))
    except Exception:
        pass

    return {"supt_p": supt_p, "wald_p": wald_p, "wald_stat": wald_stat,
            "df": df_used, "max_abs_t": max_obs}


def estimate_event_study(
    panel: EventPanel,
    control: str = "never",
    method: str = "reg",
    n_boot: int = 10_000,
    seed: int = 42,
    label: str = "",
    verbose: bool = False,
) -> EventStudyResult:
    """Run ``csa`` on an :class:`EventPanel` and package the dynamic path.

    ``control="never"`` uses only never-treated tracts as comparisons;
    ``"notyet"`` also borrows not-yet-treated ones, which is more efficient when
    treatment is saturated (most NYC tracts see *some* construction) at the cost
    of a stronger no-anticipation requirement.

    Raises ``RuntimeError`` when the panel is too thin to estimate — callers
    should check :attr:`EventPanel.usable` first if they would rather skip.
    """
    import csa
    import polars as pl

    if panel.data.empty:
        raise RuntimeError("empty event panel")
    if control == "never" and panel.n_never_treated == 0:
        raise RuntimeError("control='never' but no never-treated tracts")
    if panel.n_treated == 0:
        raise RuntimeError("no treated tracts")

    df = panel.data.rename(columns={panel.outcome_col: "outcome"})
    df = df[["unit_id", "period", "cohort", "outcome"]]

    np.random.seed(seed)  # csa's multiplier bootstrap draws from the global RNG
    res = csa.estimate(
        data=pl.from_pandas(df),
        outcome="outcome",
        unit="unit_id",
        group="cohort",
        time="period",
        control=control,
        method=method,
        verbose=verbose,
    )
    agg = csa.agg_te(res, method="dynamic", boot=True, B=n_boot, verbose=False)

    boot = getattr(agg, "boot", None)
    if boot is None:
        raise RuntimeError("csa bootstrap did not run")
    est = boot.estimates.to_pandas()
    if "k" not in est.columns:
        raise RuntimeError(f"unexpected csa bootstrap columns: {list(est.columns)}")
    est = est.sort_values("k")

    ks = est["k"].to_numpy()
    att = est["att"].to_numpy(dtype=float)
    se = est["se"].to_numpy(dtype=float)
    lower = est["lower"].to_numpy(dtype=float)
    upper = est["upper"].to_numpy(dtype=float)

    pre = pretrend_test(agg, ks, att, n=int(res.n))

    simple_att = simple_se = None
    try:
        simple = csa.agg_te(res, method="simple")
        simple_att = float(simple.overall_att)
        simple_se = float(simple.overall_se)
    except Exception:
        pass

    return EventStudyResult(
        event_times=ks,
        att=att,
        se=se,
        lower=lower,
        upper=upper,
        n_units=int(res.n),
        n_treated=panel.n_treated,
        n_never_treated=panel.n_never_treated,
        overall_att=float(agg.overall_att),
        overall_se=float(agg.overall_se),
        simple_att=simple_att,
        simple_se=simple_se,
        pretrend_supt_p=pre["supt_p"],
        pretrend_wald_p=pre["wald_p"],
        pretrend_wald_stat=pre["wald_stat"],
        pretrend_df=pre["df"],
        pretrend_max_abs_t=pre["max_abs_t"],
        control=control,
        method=method,
        anticipation=panel.anticipation,
        label=label,
    )


def pooled_common_window(results: dict[str, EventStudyResult]) -> tuple[int, int] | None:
    """Event-time window common to every city's estimates.

    Issue #36: a pooled curve must be restricted to a common window because the
    cities' imagery cadences differ, and pooling across local sensors is an
    implicit sensor-invariance claim that should at least be stated on a
    like-for-like horizon.
    """
    if not results:
        return None
    lo = max(int(np.min(r.event_times)) for r in results.values())
    hi = min(int(np.max(r.event_times)) for r in results.values())
    return (lo, hi) if lo <= hi else None
