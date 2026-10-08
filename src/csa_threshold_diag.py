# -*- coding: utf-8 -*-
"""Diagnostic: why does the Seattle 5% threshold flip sign?

Rebuilds exactly what :func:`src.evaluation.part_d` builds (unrestricted arm, no
undated screen) but keeps the intermediate objects, so cohort composition,
group-time cells and raw outcome trajectories can be inspected instead of only
the plotted ATT path.

Finding: the threshold grid is not a robustness check, because raising the
threshold moves three things at once — the dose, the adoption date, and the
never-treated pool. Stages ``donut`` / ``dating`` / ``bands`` separate them.

Run stages individually (``[city ...]``, default all four)::

    python -m src.csa_threshold_diag cohorts    # cohort sizes and timing
    python -m src.csa_threshold_diag estimate   # reproduce the plotted ATT paths
    python -m src.csa_threshold_diag cells      # ATT(g,t) cells behind them
    python -m src.csa_threshold_diag raw        # cohort x year means, model-free
    python -m src.csa_threshold_diag controls   # what the never-treated pool is
    python -m src.csa_threshold_diag donut      # treated cut x control cut grid
    python -m src.csa_threshold_diag dating     # adoption date varied alone
    python -m src.csa_threshold_diag bands      # dose-response, everything else fixed
    python -m src.csa_threshold_diag dose       # model-free dose curve

Writes CSVs to ``$CSA_DIAG_OUT`` (default ``<run>/evaluation/diagnostics``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src import csa_event_study as ces
from src import evaluation as ev
from src.utils.paths import PROCESSED_DATA_DIR, RESULTS_DIR

RUN = RESULTS_DIR / "run_20260722"
NYC_RESULTS = RUN / "nyc_zarr_check"
PROC = PROCESSED_DATA_DIR
THRESHOLDS = (0.01, 0.05, 0.10)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 300)


def load_city(key: str) -> dict:
    """Return every intermediate part_d builds for one city."""
    import os

    spec = ces.CSA_CITIES[key]
    city_years = spec.years(list(ev.YEARS))
    # CSA_DIAG_YEARS="2011,2013,2015,2017" overrides the registry panel, so an
    # irregular trailing period can be tested against a regular cadence.
    if os.environ.get("CSA_DIAG_YEARS"):
        city_years = sorted(int(y) for y in os.environ["CSA_DIAG_YEARS"].split(","))
        print(f"    panel overridden to {city_years}")
    baseline_year = spec.baseline(ev.CSA_BASELINE_YEAR)

    footprints = gpd.read_parquet(PROC / spec.footprints_filename)
    construction_year = None
    if spec.id_index is not None and footprints.index.name == spec.id_index:
        construction_year = pd.to_numeric(footprints[spec.year_col], errors="coerce")

    tracts = ev._csa_city_tracts(PROC, spec.geoid_prefixes, spec=spec,
                                 footprints=footprints)

    # CSA_DIAG_DROP_COUNTIES="24025,..." removes whole counties before anything
    # else, so a county-level imagery artifact can be excluded from the cohorts,
    # the controls and the outcomes in one place.
    import os

    drop = {c.strip() for c in os.environ.get("CSA_DIAG_DROP_COUNTIES", "").split(",")
            if c.strip()}
    if drop:
        keep = ~tracts["GEOID_str"].str[:5].isin(drop)
        print(f"    dropping counties {sorted(drop)}: "
              f"{(~keep).sum()}/{len(tracts)} tracts removed")
        tracts = tracts[keep].reset_index(drop=True)

    if key == "nyc":
        bld = ev._csa_building_preds(NYC_RESULTS, city_years)
    else:
        bld = ev._csa_city_preds(RUN, spec, city_years)
    bld = bld[bld["GEOID_str"].isin(set(tracts["GEOID_str"]))]

    outcomes = ces.tract_outcomes(bld, construction_year=construction_year,
                                  baseline_year=baseline_year)
    outcomes["split_group"] = outcomes["GEOID_str"].map(ev._csa_split_of(PROC))

    # CSA_DIAG_CONTROL_CUT="" reproduces the pre-fix complement control group;
    # unset uses ces.CONTROL_THRESHOLD, i.e. what part_d now runs.
    import os

    raw = os.environ.get("CSA_DIAG_CONTROL_CUT")
    ctrl_cut = (ces.CONTROL_THRESHOLD if raw is None
                else (float(raw) if raw.strip() else None))
    cohorts = {t: ces.build_tract_cohorts(
        footprints, tracts, city_years, baseline_year=baseline_year,
        threshold=t, year_col=spec.year_col,
        demolition_col=spec.demolition_col, area_epsg=spec.area_epsg,
        max_undated_area_share=None, control_threshold=ctrl_cut,
    ) for t in THRESHOLDS}

    return dict(key=key, spec=spec, years=city_years, baseline=baseline_year,
                tracts=tracts, outcomes=outcomes, cohorts=cohorts)


def panel_for(city: dict, thresh: float, group: str | None = "heldout",
              outcome_col: str = "pred_all", anticipation: int = 0):
    sub = city["outcomes"]
    if group is not None:
        sub = sub[sub["split_group"] == group]
    if sub.empty:
        return None
    return ces.build_event_panel(sub, city["cohorts"][thresh].cohorts,
                                 outcome_col=outcome_col,
                                 panel_years=city["years"],
                                 anticipation=anticipation)


# ── stage 1: cohort composition ──────────────────────────────────────────────

def stage_cohorts(keys: list[str]) -> None:
    for key in keys:
        city = load_city(key)
        pm = ces.PeriodMap.from_years(city["years"])
        print(f"\n{'='*78}\n{city['spec'].label}  years={city['years']}  "
              f"baseline={city['baseline']}\n{'='*78}")
        print(f"tracts with predictions: {city['outcomes']['GEOID_str'].nunique():,}"
              f"  split_group counts: "
              f"{city['outcomes'].drop_duplicates('GEOID_str')['split_group'].value_counts().to_dict()}")

        for t in THRESHOLDS:
            coh = city["cohorts"][t]
            panel = panel_for(city, t, "heldout")
            print(f"\n--- threshold {t:.0%} ---")
            print(f"  cohort build: {coh.summary()}")
            if panel is None:
                print("  no held-out panel")
                continue
            print(f"  held-out panel: {panel.summary()}")
            d = panel.data.drop_duplicates("unit_id")
            dist = d["cohort"].value_counts().sort_index()
            rows = [(int(g), int(n), ("never" if g == 0 else pm.year(int(g))))
                    for g, n in dist.items()]
            print("  cohort (period, n, calendar year): " +
                  ", ".join(f"g={g}({yr}):{n}" for g, n, yr in rows))
            # which event times each cohort can identify
            T = pm.n_periods
            ks = {}
            for g, n, _ in rows:
                if g == 0:
                    continue
                for tt in range(1, T + 1):
                    ks.setdefault(tt - g, 0)
                    ks[tt - g] += n
            print("  treated tracts contributing to each k: " +
                  ", ".join(f"k={k}:{ks[k]}" for k in sorted(ks)))


# ── stage 2: estimation across thresholds ────────────────────────────────────

def stage_estimate(keys: list[str], n_boot: int = 10_000) -> None:
    out = []
    for key in keys:
        city = load_city(key)
        for t in THRESHOLDS:
            for group in ("heldout", "train"):
                panel = panel_for(city, t, group)
                if panel is None:
                    continue
                res, reason = ev._csa_estimate_or_none(
                    panel, label=f"{key}/{group}/{t:.0%}", n_boot=n_boot)
                if res is None:
                    print(f"  skip {key}/{group}/{t:.0%}: {reason}")
                    continue
                for i, k in enumerate(res.event_times):
                    out.append(dict(city=key, split=group, threshold=t,
                                    k=int(k), att=res.att[i], se=res.se[i],
                                    lo=res.lower[i], hi=res.upper[i],
                                    n_treated=res.n_treated,
                                    n_control=res.n_never_treated,
                                    pretrend_p=res.pretrend_supt_p,
                                    overall=res.overall_att))
    df = pd.DataFrame(out)
    df.to_csv(_outpath("threshold_paths.csv"), index=False)
    for key in keys:
        for group in ("heldout", "train"):
            sub = df[(df.city == key) & (df.split == group)]
            if sub.empty:
                continue
            print(f"\n=== {key} / {group} ===")
            piv = sub.pivot(index="k", columns="threshold", values="att")
            sep = sub.pivot(index="k", columns="threshold", values="se")
            print(pd.concat({"att": piv, "se": sep}, axis=1).round(4))
            print(sub.groupby("threshold")[["n_treated", "n_control",
                                            "pretrend_p", "overall"]].first().round(4))


# ── stage 3: group-time cells behind the dynamic path ────────────────────────

def _gp_frame(panel) -> pd.DataFrame:
    """csa's ATT(g,t) cells with their dynamic aggregation weights."""
    import csa
    import polars as pl

    df = panel.data.rename(columns={panel.outcome_col: "outcome"})
    df = df[["unit_id", "period", "cohort", "outcome"]]
    np.random.seed(42)
    res = csa.estimate(data=pl.from_pandas(df), outcome="outcome", unit="unit_id",
                       group="cohort", time="period", control="never",
                       method="reg", verbose=False)
    agg = csa.agg_te(res, method="dynamic", boot=False)
    gp = agg.params.gp.to_pandas()
    gp["att"] = np.asarray(agg.params.att)
    return gp


def stage_cells(keys: list[str]) -> None:
    frames = []
    for key in keys:
        city = load_city(key)
        pm = ces.PeriodMap.from_years(city["years"])
        for t in THRESHOLDS:
            panel = panel_for(city, t, "heldout")
            if panel is None or not panel.usable:
                continue
            gp = _gp_frame(panel)
            gp["city"] = key
            gp["threshold"] = t
            gcol = [c for c in gp.columns if c.lower() in ("cohort", "g")][0]
            tcol = [c for c in gp.columns if c.lower() in ("period", "t", "time")][0]
            gp["g_year"] = gp[gcol].map(lambda g: pm.year(int(g)))
            gp["t_year"] = gp[tcol].map(lambda x: pm.year(int(x)))
            frames.append(gp)
            print(f"\n=== {key} {t:.0%}: ATT(g,t) cells ===")
            cols = [c for c in ("K", "att", "pge", "pg", "g_year", "t_year",
                                gcol, tcol) if c in gp.columns]
            print(gp[cols].round(4).to_string(index=False))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(_outpath("gt_cells.csv"),
                                                    index=False)


# ── stage 4: model-free trajectories + overlap between threshold samples ─────

def stage_raw(keys: list[str]) -> None:
    for key in keys:
        city = load_city(key)
        pm = ces.PeriodMap.from_years(city["years"])
        print(f"\n{'='*78}\n{city['spec'].label}: raw held-out tract means\n{'='*78}")
        base = None
        for t in THRESHOLDS:
            panel = panel_for(city, t, "heldout")
            if panel is None:
                continue
            d = panel.data
            g = d.drop_duplicates("unit_id").set_index("GEOID_str")["cohort"]
            if base is None:
                base = g
            wide = d.pivot(index="GEOID_str", columns="period",
                           values=panel.outcome_col)
            wide["cohort"] = g
            m = wide.groupby("cohort").mean()
            n = wide.groupby("cohort").size()
            m.columns = [pm.year(int(c)) for c in m.columns]
            m.insert(0, "n", n)
            print(f"\n--- {t:.0%}: mean prediction by cohort x calendar year ---")
            print(m.round(4).to_string())
            # never-treated minus each cohort, differenced from the cohort's g-1
            never = m.loc[0] if 0 in m.index else None
            if never is not None:
                print("  DiD vs never-treated, base = period g-1:")
                for gg in [i for i in m.index if i > 0]:
                    b = pm.year(int(gg) - 1)
                    row = {}
                    for yr in m.columns[1:]:
                        row[yr] = round(float((m.loc[gg, yr] - m.loc[gg, b])
                                              - (never[yr] - never[b])), 4)
                    print(f"    g={gg}({pm.year(int(gg))}) n={int(m.loc[gg,'n'])}: {row}")


# ── stage 5: donut design — treated cut and control cut varied separately ────
#
# Raising the threshold moves TWO things at once: which tracts count as treated
# (and when), and which tracts are left in the never-treated pool. A tract with
# 3% new area is treated at 1% and a control at 5%. This stage breaks that
# confound by dropping the ambiguous middle: treated = share > treat_cut,
# control = share <= ctrl_cut, nothing in between. With ctrl_cut fixed, moving
# treat_cut is a pure treatment-definition change; with treat_cut fixed, moving
# ctrl_cut is a pure control-group change.

def donut_cohorts(city: dict, treat_cut: float, ctrl_cut: float) -> pd.DataFrame:
    ct = city["cohorts"][treat_cut].cohorts
    cc = city["cohorts"][ctrl_cut].cohorts.set_index("GEOID_str")["cohort_year"]
    middle = (ct["cohort_year"] == 0) & (ct["GEOID_str"].map(cc).fillna(0) > 0)
    return ct[~middle].copy()


def stage_donut(keys: list[str], n_boot: int = 5_000) -> None:
    rows = []
    for key in keys:
        city = load_city(key)
        heldout = city["outcomes"][city["outcomes"]["split_group"] == "heldout"]
        for ctrl_cut in THRESHOLDS:
            for treat_cut in THRESHOLDS:
                if treat_cut < ctrl_cut:
                    continue
                coh = donut_cohorts(city, treat_cut, ctrl_cut)
                panel = ces.build_event_panel(heldout, coh, outcome_col="pred_all",
                                              panel_years=city["years"])
                for control in ("never", "notyet"):
                    res, reason = ev._csa_estimate_or_none(
                        panel, label=f"{key}", control=control, n_boot=n_boot)
                    if res is None:
                        print(f"  skip {key} treat>{treat_cut:.0%} ctrl<={ctrl_cut:.0%} "
                              f"{control}: {reason}")
                        continue
                    kmax = int(np.max(res.event_times))
                    rows.append(dict(
                        city=key, treat_cut=treat_cut, ctrl_cut=ctrl_cut,
                        control=control, n_treated=res.n_treated,
                        n_control=res.n_never_treated,
                        pretrend_p=res.pretrend_supt_p,
                        overall=res.overall_att, se=res.overall_se,
                        att_k0=res.post_att(0)[0], att_k1=res.post_att(1)[0],
                        att_k2=res.post_att(2)[0],
                        att_kmax=res.post_att(kmax)[0], kmax=kmax))
    df = pd.DataFrame(rows)
    df.to_csv(_outpath("donut.csv"), index=False)
    for key in keys:
        for control in ("never", "notyet"):
            sub = df[(df.city == key) & (df.control == control)]
            if sub.empty:
                continue
            print(f"\n=== {key} / control={control}: overall ATT "
                  "(rows = control cut, cols = treated cut) ===")
            print(sub.pivot(index="ctrl_cut", columns="treat_cut",
                            values="overall").round(4).to_string())
            print("  n_control:")
            print(sub.pivot(index="ctrl_cut", columns="treat_cut",
                            values="n_control").to_string())
            print("  n_treated:")
            print(sub.pivot(index="ctrl_cut", columns="treat_cut",
                            values="n_treated").to_string())


# ── stage 6: what the never-treated pool actually is at each threshold ───────

def stage_controls(keys: list[str]) -> None:
    for key in keys:
        city = load_city(key)
        pm = ces.PeriodMap.from_years(city["years"])
        y0, y1 = city["years"][0], city["years"][-1]
        wide = (city["outcomes"].pivot(index="GEOID_str", columns="year",
                                       values="pred_all"))
        nbld = (city["outcomes"].groupby("GEOID_str")["n_all"].mean())
        print(f"\n{'='*78}\n{city['spec'].label}: never-treated pool by threshold"
              f"\n{'='*78}")
        for t in THRESHOLDS:
            coh = city["cohorts"][t].cohorts.set_index("GEOID_str")
            never = coh.index[coh["cohort_year"] == 0]
            never = [g for g in never if g in wide.index]
            w = wide.loc[never]
            share = (coh.loc[never, "new_area"] / coh.loc[never, "base_area"])
            und = city["cohorts"][t].undated_area_share
            print(f"\n--- never-treated at {t:.0%}: n={len(never)} ---")
            print(f"  mean pred {y0}: {w[y0].mean():+.4f}   {y1}: {w[y1].mean():+.4f}"
                  f"   drift: {w[y1].mean() - w[y0].mean():+.4f}")
            print(f"  new-area share: mean {share.mean():.4f}  "
                  f"median {share.median():.4f}  p90 {share.quantile(0.9):.4f}")
            print(f"  baseline building area (m^2): median "
                  f"{coh.loc[never, 'base_area'].median():,.0f}   "
                  f"buildings/tract median {nbld.reindex(never).median():,.0f}")
            if und is not None:
                print(f"  undated area share: median "
                      f"{und.reindex(never).median():.4f}")
        # per-year city-wide drift, all held-out tracts
        allm = wide.mean()
        print(f"\n  city-wide mean prediction by year (all tracts with preds):")
        print("   " + "  ".join(f"{int(y)}:{allm[y]:+.4f}" for y in allm.index))


# ── stage 7: dating held separate from sample ────────────────────────────────
#
# The donut still moves two things: which tracts are treated, and WHEN. Raising
# the cut also pushes each tract's adoption date later, so a tract that has been
# building since period 2 is called "treated at period 4" and its own
# pre-treatment window already contains construction. This stage fixes the
# sample (treated at 10%, controls never at 1%) and varies only the crossing
# threshold used to date adoption.

def stage_dating(keys: list[str], n_boot: int = 5_000) -> None:
    rows = []
    for key in keys:
        city = load_city(key)
        heldout = city["outcomes"][city["outcomes"]["split_group"] == "heldout"]
        c10 = city["cohorts"][0.10].cohorts.set_index("GEOID_str")["cohort_year"]
        c01 = city["cohorts"][0.01].cohorts.set_index("GEOID_str")["cohort_year"]
        treated = set(c10.index[c10 > 0])
        controls = set(c01.index[c01 == 0])
        for date_cut in THRESHOLDS:
            cy = city["cohorts"][date_cut].cohorts.set_index("GEOID_str")["cohort_year"]
            coh = pd.DataFrame({
                "GEOID_str": sorted(treated | controls),
            })
            coh["cohort_year"] = [
                int(cy.get(g, 0)) if g in treated else 0 for g in coh["GEOID_str"]
            ]
            panel = ces.build_event_panel(heldout, coh, outcome_col="pred_all",
                                          panel_years=city["years"])
            res, reason = ev._csa_estimate_or_none(panel, label=key, n_boot=n_boot)
            if res is None:
                print(f"  skip {key} date@{date_cut:.0%}: {reason}")
                continue
            rows.append(dict(city=key, date_cut=date_cut, n_treated=res.n_treated,
                             n_control=res.n_never_treated,
                             pretrend_p=res.pretrend_supt_p,
                             overall=res.overall_att,
                             att_k0=res.post_att(0)[0], att_k1=res.post_att(1)[0],
                             att_k2=res.post_att(2)[0]))
    df = pd.DataFrame(rows)
    df.to_csv(_outpath("dating.csv"), index=False)
    print("\n=== sample fixed (treated = crosses 10%, control = never crosses 1%); "
          "only the dating threshold varies ===")
    print(df.round(4).to_string(index=False))


# ── stage 8: dose-response by final intensity band ───────────────────────────
#
# Same control pool and same dating rule for every band, so the only thing that
# changes is how much got built. This is the comparison the threshold grid is
# meant to approximate but does not, because it confounds dose with control-pool
# composition and with adoption timing.

BANDS = ((0.01, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, np.inf))


def stage_bands(keys: list[str], n_boot: int = 5_000) -> None:
    rows = []
    for key in keys:
        city = load_city(key)
        heldout = city["outcomes"][city["outcomes"]["split_group"] == "heldout"]
        c01 = city["cohorts"][0.01].cohorts.set_index("GEOID_str")
        final_share = c01["new_area"] / c01["base_area"]
        cy = c01["cohort_year"]
        controls = set(cy.index[cy == 0])
        for lo, hi in BANDS:
            treated = set(final_share.index[(final_share > lo) & (final_share <= hi)])
            treated &= set(cy.index[cy > 0])
            coh = pd.DataFrame({"GEOID_str": sorted(treated | controls)})
            coh["cohort_year"] = [int(cy.get(g, 0)) if g in treated else 0
                                  for g in coh["GEOID_str"]]
            panel = ces.build_event_panel(heldout, coh, outcome_col="pred_all",
                                          panel_years=city["years"])
            res, reason = ev._csa_estimate_or_none(panel, label=key, n_boot=n_boot)
            if res is None:
                print(f"  skip {key} band ({lo:.0%},{hi:.0%}]: {reason}")
                continue
            rows.append(dict(city=key, band=f"({lo:.0%},{hi:.0%}]",
                             n_treated=res.n_treated, n_control=res.n_never_treated,
                             pretrend_p=res.pretrend_supt_p,
                             overall=res.overall_att, se=res.overall_se,
                             att_k0=res.post_att(0)[0], att_k1=res.post_att(1)[0],
                             att_k2=res.post_att(2)[0]))
    df = pd.DataFrame(rows)
    df.to_csv(_outpath("bands.csv"), index=False)
    print("\n=== dose-response: control pool and dating fixed (never crosses 1% / "
          "first crossing of 1%), only the final intensity band varies ===")
    print(df.round(4).to_string(index=False))


# ── stage 9: model-free dose curve + control-contamination arithmetic ────────

def stage_dose(keys: list[str]) -> None:
    for key in keys:
        city = load_city(key)
        y0, y1 = city["years"][0], city["years"][-1]
        wide = city["outcomes"].pivot(index="GEOID_str", columns="year",
                                      values="pred_all")
        c01 = city["cohorts"][0.01].cohorts.set_index("GEOID_str")
        share = (c01["new_area"] / c01["base_area"]).reindex(wide.index).dropna()
        d = pd.DataFrame({"share": share,
                          "delta": (wide[y1] - wide[y0]).reindex(share.index),
                          "level": wide[y0].reindex(share.index)}).dropna()
        edges = [0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, np.inf]
        d["bin"] = pd.cut(d["share"], edges, right=True, include_lowest=True)
        g = d.groupby("bin", observed=True).agg(
            n=("delta", "size"), mean_delta=("delta", "mean"),
            se_delta=("delta", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            mean_level=("level", "mean"))
        base = g["mean_delta"].iloc[0]
        g["vs_lowest"] = g["mean_delta"] - base
        print(f"\n{'='*78}\n{city['spec'].label}: raw change in tract-mean "
              f"prediction {y0}->{y1} by final new-area share\n{'='*78}")
        print(g.round(4).to_string())

        # How much of the never-treated pool at each threshold is made up of
        # tracts that DID build (i.e. are treated at the 1% cut), and what the
        # ATT of that contaminating band is — the two terms whose product is the
        # bias that raising the threshold injects into the control group.
        print("  control-pool contamination:")
        for t in THRESHOLDS:
            ct = city["cohorts"][t].cohorts.set_index("GEOID_str")["cohort_year"]
            never = set(ct.index[ct == 0]) & set(d.index)
            clean = {g_ for g_ in never if share.get(g_, 1.0) <= 0.01}
            w = 1 - len(clean) / max(len(never), 1)
            dd = d.loc[sorted(never)]
            print(f"    never@{t:>4.0%}: n={len(never):4d}  clean(<=1%)={len(clean):4d}  "
                  f"contaminated share={w:.3f}  "
                  f"mean delta={dd['delta'].mean():+.4f}")


# ── stage 10: is a year-specific level shift clustered by county? ────────────
#
# A pre-trend that is a one-period SPIKE rather than a trend, and that does not
# move when the adoption date moves, is not a timing artifact — it is a level
# shift in one year. The two things that produce that are imagery (a NAIP year
# flown differently, or not at all, over part of the CBSA) and composition (the
# buildings that got predicted in that year are not the ones predicted in the
# others). `n_all` catches the second: the per-tract building sample is a stable
# hash, identical every year, so a year with fewer scored buildings in a tract
# means predictions came back non-finite there and were dropped.

def stage_county(keys: list[str]) -> None:
    for key in keys:
        city = load_city(key)
        o = city["outcomes"].copy()
        o["county"] = o["GEOID_str"].str[:5]

        print(f"\n{'='*78}\n{city['spec'].label}: mean prediction by county x year"
              f"\n{'='*78}")
        print(o.pivot_table(index="county", columns="year", values="pred_all",
                            aggfunc="mean").round(4).to_string())
        print("\n  year-over-year change:")
        piv = o.pivot_table(index="county", columns="year", values="pred_all",
                            aggfunc="mean")
        print(piv.diff(axis=1).round(4).to_string())

        print("\n  buildings scored per tract (median n_all) by county x year —"
              " a drop means predictions came back non-finite:")
        print(o.pivot_table(index="county", columns="year", values="n_all",
                            aggfunc="median").round(0).to_string())

        print("\n  tracts with predictions by county x year:")
        print(o.pivot_table(index="county", columns="year", values="GEOID_str",
                            aggfunc="count").to_string())

        # where the cohorts live
        for t in THRESHOLDS:
            coh = city["cohorts"][t].cohorts.set_index("GEOID_str")["cohort_year"]
            panel = panel_for(city, t, "heldout")
            if panel is None:
                continue
            d = panel.data.drop_duplicates("GEOID_str")
            d = d.assign(county=d["GEOID_str"].str[:5])
            print(f"\n  {t:.0%}: cohort x county counts")
            print(pd.crosstab(d["cohort"], d["county"]).to_string())


def _outpath(name: str) -> Path:
    import os

    p = Path(os.environ.get("CSA_DIAG_OUT",
                            RUN / "evaluation" / "diagnostics"))
    p.mkdir(parents=True, exist_ok=True)
    return p / name


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "cohorts"
    keys = sys.argv[2:] or ["nyc", "tampa", "seattle", "san_antonio"]
    {"cohorts": stage_cohorts, "estimate": stage_estimate,
     "cells": stage_cells, "raw": stage_raw, "donut": stage_donut,
     "controls": stage_controls, "dating": stage_dating,
     "bands": stage_bands, "dose": stage_dose, "county": stage_county}[stage](keys)
