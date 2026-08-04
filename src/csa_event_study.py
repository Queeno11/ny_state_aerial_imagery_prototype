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
    "DEFAULT_THRESHOLDS",
    "HEADLINE_THRESHOLD",
    "PeriodMap",
    "CohortResult",
    "EventPanel",
    "EventStudyResult",
    "available_cities",
    "build_tract_cohorts",
    "build_event_panel",
    "estimate_event_study",
    "pretrend_test",
    "tract_outcomes",
]

# Cumulative-new-area thresholds defining "treated". Reported as a robustness
# grid; HEADLINE_THRESHOLD is the one that goes in the main figure.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.01, 0.05, 0.10)
HEADLINE_THRESHOLD: float = 0.05

# Event-time horizons tabulated in the coefficient table (issue #36 asks for
# post-ATTs at a couple of horizons, not one lumped number).
POST_HORIZONS: tuple[int, ...] = (0, 1, 2, 3)

# Below this many treated (or never-treated) tracts a dynamic event study is not
# worth plotting — the sup-t critical value blows up and the curve is noise.
MIN_TREATED_UNITS = 20
MIN_CONTROL_UNITS = 20


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
    note: str = ""


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
    ),
    "chicago": CityCohortSpec(
        key="chicago",
        label="Chicago",
        geoid_prefixes=("17031", "17043", "17089", "17093", "17097", "17111", "17197"),
        footprints_filename="buildings_chicago.parquet",
        year_col="year_built",
        demolition_col=None,
        area_epsg=5070,
        note="needs the Cook County assessor year-built join (issue #36 follow-up)",
    ),
    "seattle": CityCohortSpec(
        key="seattle",
        label="Seattle (King County)",
        geoid_prefixes=("53033",),
        footprints_filename="buildings_king_county.parquet",
        year_col="year_built",
        demolition_col=None,
        area_epsg=5070,
        note="needs the King County assessor year-built join (issue #36 follow-up)",
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
        area_epsg=5070,
        note="needs the FL FGDL parcel year-built join (issue #36 follow-up)",
    ),
    "nashville": CityCohortSpec(
        key="nashville",
        label="Nashville-Davidson",
        geoid_prefixes=("47037", "47149", "47189", "47165", "47187"),
        footprints_filename="buildings_tn_tnmap.parquet",
        year_col="year_built",
        demolition_col=None,
        area_epsg=5070,
        note="needs the TN TNMap parcel year-built join (issue #36 follow-up)",
    ),
}

# Cities whose panel is deep enough to carry the main-body figure, in order.
MAIN_FIGURE_CITIES: tuple[str, ...] = ("nyc", "chicago")


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

    def summary(self) -> dict:
        return {
            "threshold": self.threshold,
            "n_tracts": int(self.n_tracts_in),
            "n_dropped_no_baseline": int(self.n_dropped_no_baseline),
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
    """
    years = sorted({int(y) for y in panel_years})
    if not years:
        raise ValueError("panel_years is empty")
    if baseline_year >= years[-1]:
        raise ValueError(
            f"baseline_year {baseline_year} leaves no post-baseline panel year"
        )

    fp = footprints.copy()
    fp["year_built"] = pd.to_numeric(fp[year_col], errors="coerce")
    if demolition_col is not None and demolition_col in fp.columns:
        fp["demolition_year"] = pd.to_numeric(fp[demolition_col], errors="coerce")

    per_bldg = _tract_of_footprints(fp, tracts, area_epsg)
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

    new = per_bldg.loc[is_new, ["GEOID_str", "area", "year_built"]].copy()
    new["year_built"] = new["year_built"].astype(int)
    new_by_year = (
        new.groupby(["GEOID_str", "year_built"])["area"].sum()
        if len(new) else pd.Series(dtype=float)
    )

    # Cumulative new area *as visible at each panel year*: a building finished
    # in 2011 shows up in the 2012 image, so a biennial panel credits it to
    # 2012. Attributing by "<= panel year" is what makes this cadence-agnostic.
    rows = []
    for tid in tract_ids:
        cum = 0.0
        per_year = (
            new_by_year.loc[tid] if len(new_by_year) and tid in
            new_by_year.index.get_level_values(0) else None
        )
        prev = baseline_year
        for yr in years:
            if per_year is not None:
                built = per_year[(per_year.index > prev) & (per_year.index <= yr)]
                cum += float(built.sum())
            rows.append((tid, yr, cum))
            prev = yr
    shares = pd.DataFrame(rows, columns=["GEOID_str", "year", "cum_new_area"])
    shares = shares.merge(
        base_area.rename("base_area"), left_on="GEOID_str", right_index=True, how="left"
    )

    n_in = len(tract_ids)
    undefined = shares["base_area"] <= min_baseline_area
    dropped_ids = sorted(shares.loc[undefined, "GEOID_str"].unique())
    shares = shares[~undefined].copy()
    shares["share"] = shares["cum_new_area"] / shares["base_area"]

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
    if anticipation:
        # Shift adoption earlier for treated units only; 0 stays never-treated.
        df["cohort"] = np.where(
            df["cohort"] > 0, df["cohort"] - int(anticipation), 0
        ).astype(int)

    # 4. cohorts in (or before) the first period have no pre-period to serve as
    #    csa's base year. An anticipation shift can push a cohort to <= 1 too.
    first_period = 1
    bad = (df["cohort"] > 0) & (df["cohort"] <= first_period)
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
