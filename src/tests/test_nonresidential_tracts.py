"""Exclusion of non-residential tracts from the evaluation figures.

Special-land-use tracts (Census tract code 98xx) and near-zero ACS wealth
tract-years are removed from the shared figure sample and from the GB2 fits.
All synthetic.
"""

import numpy as np
import pandas as pd

import src.evaluation as ev


class _Ctx:
    """Minimal stand-in for _USContext: what _figure_test_sample reads."""

    def __init__(self, bld, tract, wealth=None):
        self.bld, self.tract = bld, tract
        if wealth is not None:
            self._exclusion_wealth_cache = wealth


def _wealth(rows):
    """rows = [(GEOID, acs_year, usd), ...] -> wealth_long frame."""
    df = pd.DataFrame(rows, columns=["GEOID", "year", "usd"])
    df["cbsa"] = "33860"
    return df


def test_special_land_use_is_tract_code_98xx():
    geoids = pd.Series(["17197980100", "17197981001", "17197880100", "17197098000",
                        1001980100])                       # int GEOID, lost leading zero
    assert ev._is_special_land_use(geoids).tolist() == [True, True, False, False, True]


def test_residential_wealth_drops_98xx_and_near_zero_only():
    w = _wealth([("17197980100", 2019, 500_000.0),         # 98xx: dropped
                 ("17197010100", 2019, 184.0),             # near zero: dropped
                 ("17197010200", 2019, 12_000.0),          # poor but real: kept
                 ("17197010300", 2019, np.nan)])           # missing: kept for the fit to skip
    out = ev._residential_wealth(w)
    assert out["GEOID"].tolist() == ["17197010200", "17197010300"]


def test_near_zero_rule_uses_the_nearest_acs_vintage():
    tract = pd.DataFrame({"GEOID": ["17197010100"] * 2, "year": [2014, 2017]})
    w = _wealth([("17197010100", 2013, 600.0),             # 2014 -> 2013: near zero
                 ("17197010100", 2018, 47_000.0)])         # 2017 -> 2018: fine
    flags = ev._nonresidential_tract_years(tract, w)
    assert flags["mask"].tolist() == [True, False]
    assert (flags["special"], flags["near_zero"], flags["wealth_rule"]) == (0, 1, True)


def test_without_wealth_only_the_98xx_rule_runs():
    tract = pd.DataFrame({"GEOID": ["17197980100", "17197010100"], "year": [2016, 2016]})
    flags = ev._nonresidential_tract_years(tract, None)
    assert flags["mask"].tolist() == [True, False]
    assert flags["wealth_rule"] is False


def _city(n_tracts, year=2016, cbsa="16980", prefix="17031"):
    """Tract and building frames for one test city-year; the label is the pred."""
    geoids = [f"{prefix}{i + 10000:06d}" for i in range(n_tracts)]
    vals = np.linspace(-1, 1, n_tracts)
    tract = pd.DataFrame({"GEOID": geoids, "year": year, "cbsa": cbsa,
                          "pred": vals, "label": vals})
    bld = tract.loc[tract.index.repeat(2)].assign(type="test").reset_index(drop=True)
    return tract, bld


def test_figure_sample_excludes_98xx_and_near_zero_tract_years():
    tract, bld = _city(25)
    special = tract.index[0]
    tract.loc[special, "GEOID"] = "17031980100"
    bld.loc[bld.index[:2], "GEOID"] = "17031980100"
    poor = tract["GEOID"].iloc[1]
    wealth = _wealth([(poor, 2016, 789.0)] + [(g, 2016, 300_000.0)
                                              for g in tract["GEOID"].iloc[2:]])
    s = ev._figure_test_sample(_Ctx(bld, tract, wealth))
    kept = set(s["tract_test"]["GEOID"])
    assert "17031980100" not in kept and poor not in kept and len(kept) == 23
    assert set(s["test_bld"]["GEOID"]) == kept
    assert int(s["cells_tract"]["n_tracts"].iloc[0]) == 23
    assert s["excluded"] == {"special": 1, "near_zero": 1, "wealth_rule": True, "n": 2}


def test_exclusion_runs_before_the_min_tract_floor():
    """A city-year left with < _MIN_FIGURE_TRACTS residential tracts is dropped."""
    big, big_b = _city(30, cbsa="16980", prefix="17031")
    thin, thin_b = _city(ev._MIN_FIGURE_TRACTS, cbsa="41180", prefix="29510")
    thin.loc[thin.index[0], "GEOID"] = "29510980100"
    thin_b.loc[thin_b.index[:2], "GEOID"] = "29510980100"
    ctx = _Ctx(pd.concat([big_b, thin_b], ignore_index=True),
               pd.concat([big, thin], ignore_index=True))
    s = ev._figure_test_sample(ctx)
    assert s["cells_tract"]["cbsa"].tolist() == ["16980"]
    assert s["excluded"]["wealth_rule"] is False      # no processed panel on this ctx


def test_sample_without_nonresidential_tracts_is_unchanged():
    tract, bld = _city(25)
    s = ev._figure_test_sample(_Ctx(bld, tract))
    assert len(s["tract_test"]) == 25 and s["excluded"]["n"] == 0
