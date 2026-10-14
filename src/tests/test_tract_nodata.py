"""Nodata (black-crop) retrofit: per-run sentinel detection and tract means.

Each run's black-crop value is detected from its own prediction histogram
(evaluation._detect_nodata_values). predictions_by_tract_<year>.parquet averages
black crops in, so the US context rebuilds those means from the filtered
building frame (evaluation._tract_means_from_buildings). All synthetic.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

import src.evaluation as ev
from src.data import indicators

BLACK = ev.NODATA_PREDICTION_VALUE


def _bld(rows):
    """rows = [(GEOID, year, pred, label), ...] -> canonical building frame."""
    return pd.DataFrame(rows, columns=["GEOID", "year", "pred", "label"])


def _tract_from(bld):
    """What main.py wrote: the raw per-tract mean/std of the building CSV."""
    return (bld.groupby(["GEOID", "year"])
            .agg(label=("label", "mean"), pred=("pred", "mean"),
                 predicted_value_std=("pred", "std"))
            .reset_index())


def test_clean_pass_is_a_noop():
    bld = _bld([("01001020100", 2016, 0.1 * i, 0.3) for i in range(5)]
               + [("01001020200", 2016, -0.2 * i, -0.4) for i in range(5)])
    tract = _tract_from(bld)
    out = ev._tract_means_from_buildings(tract.copy(), bld, [2016])
    pd.testing.assert_frame_equal(out, tract)


def test_black_buildings_are_averaged_out_and_label_kept():
    raw = _bld([("01001020100", 2016, 0.2, 0.7)] * 8
               + [("01001020100", 2016, BLACK, 0.7)] * 2)
    tract = _tract_from(raw)
    assert tract["pred"].iloc[0] != pytest.approx(0.2)        # parquet is contaminated
    out = ev._tract_means_from_buildings(tract, ev._drop_nodata_predictions(raw), [2016])
    assert len(out) == 1
    assert out["pred"].iloc[0] == pytest.approx(0.2)
    assert out["predicted_value_std"].iloc[0] == pytest.approx(0.0)
    assert out["label"].iloc[0] == pytest.approx(0.7)


def test_tract_year_removed_by_the_filter_is_dropped():
    """A >=50%-black tract-year leaves whole; the same tract's clean year stays."""
    raw = _bld([("01001020100", 2016, BLACK, 0.7)] * 9
               + [("01001020100", 2016, 0.5, 0.7)]
               + [("01001020100", 2017, 0.4, 0.6)] * 5)
    out = ev._tract_means_from_buildings(_tract_from(raw), ev._drop_nodata_predictions(raw),
                                     [2016, 2017])
    assert list(zip(out["GEOID"], out["year"])) == [("01001020100", 2017)]
    assert out["pred"].iloc[0] == pytest.approx(0.4)


def test_years_without_building_rows_are_untouched():
    bld = _bld([("01001020100", 2016, 0.1, 0.3)] * 3)
    tract = pd.concat([_tract_from(bld),
                       pd.DataFrame({"GEOID": ["01001020100"], "year": [2018],
                                     "label": [0.3], "pred": [0.9],
                                     "predicted_value_std": [0.1]})],
                      ignore_index=True)
    out = ev._tract_means_from_buildings(tract.copy(), bld, [2016])
    pd.testing.assert_frame_equal(out, tract)


def test_year_emptied_by_the_filter_is_still_covered():
    """REGRESSION: if every building of a year is black, the year disappears from
    the filtered frame. Its contaminated tract rows must go too, not be mistaken
    for a year without a CSV and kept."""
    raw = _bld([("01001020100", 2016, BLACK, 0.7)] * 5)
    out = ev._tract_means_from_buildings(_tract_from(raw), ev._drop_nodata_predictions(raw),
                                         [2016])
    assert out.empty


def test_us_context_tract_frame_uses_filtered_means(tmp_path, monkeypatch):
    """End to end: the parquet carries the contaminated mean; ctx.tract must not.

    The black crops sit at an L2-style value (not NODATA_PREDICTION_VALUE), so
    this also checks the loader detects the run's own sentinel.
    """
    monkeypatch.setattr(ev, "MIN_CBSA_TRACTS", 1)
    res, proc = tmp_path / "res", tmp_path / "proc"
    res.mkdir(), proc.mkdir()
    a, b, bg = "01001020100", "01001020200", "01001020300"
    gpd.GeoDataFrame(
        {"geoid_2023": [a, b, bg], "cbsa_code": ["33860"] * 3,
         indicators.valid_change_col("W2_r5"): [0, 0, 0],
         "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]},
        crs="EPSG:5070").to_feather(proc / ev._PANEL_FILENAME)

    l2_black = 0.3659668
    grid = np.round(np.linspace(-1, 1, 401), 6)          # smooth background, 10 each
    a_clean = grid[250:270]                               # tract A's real buildings
    rows = ([(bg, v, 0.0) for v in np.repeat(grid, 10)]
            + [(a, v, 0.7) for v in a_clean] + [(a, l2_black, 0.7)] * 10   # re-averaged
            + [(b, l2_black, -0.5)] * 60)                                  # dropped whole
    csv = pd.DataFrame([{"Rel_Score": lab, "predicted_value": p, "building_id": i,
                         "GEOID": g, "year": 2016, "type": "test"}
                        for i, (g, p, lab) in enumerate(rows)])
    csv.to_csv(res / "2016_predictions.csv", index=False)
    parquet = (csv.groupby("GEOID")
               .agg({"Rel_Score": "mean", "predicted_value": ["mean", "std"]}).reset_index())
    parquet.columns = ["GEOID", "Rel_Score", "predicted_value", "predicted_value_std"]
    parquet["GEOID"] = parquet["GEOID"].astype(np.int64)    # as written: numeric GEOID
    parquet.to_parquet(res / "predictions_by_tract_2016.parquet")

    ctx = ev._USContext(res, proc, res / "evaluation", "W2_r5")
    assert sorted(ctx.tract["GEOID"]) == [a, bg]
    got = ctx.tract.set_index("GEOID")
    assert got.loc[a, "pred"] == pytest.approx(a_clean.mean())
    assert got.loc[a, "label"] == pytest.approx(0.7)
    assert not (ctx.bld["pred"].round(6) == round(l2_black, 6)).any()


# ─── per-run sentinel detection (evaluation._detect_nodata_values) ────────────

def _smooth(n_values=401, per_value=10):
    return pd.Series(np.repeat(np.round(np.linspace(-1, 1, n_values), 6), per_value))


def test_detect_finds_nothing_in_a_smooth_histogram():
    assert ev._detect_nodata_values(_smooth()) == []


def test_detect_ignores_a_smooth_mode():
    """A busy region whose neighbours are busy too is a mode, not a sentinel
    (the ordinal run's busiest value, -1.25, is like this)."""
    vals = np.round(np.linspace(-1, 1, 401), 6)
    counts = (10 + 300 * np.exp(-(vals / 0.1) ** 2)).astype(int)
    assert ev._detect_nodata_values(pd.Series(np.repeat(vals, counts))) == []


def test_detect_finds_a_run_specific_sentinel():
    pred = pd.concat([_smooth(), pd.Series([0.3659668] * 200)], ignore_index=True)
    assert ev._detect_nodata_values(pred) == [pytest.approx(0.3659668)]


def test_detect_finds_two_adjacent_sentinels():
    """The L2 run puts its black crops on 0.3659668 and the next grid value."""
    pred = pd.concat([_smooth(), pd.Series([0.3659668] * 300 + [0.3657227] * 80)],
                     ignore_index=True)
    assert ev._detect_nodata_values(pred) == [pytest.approx(0.3657227),
                                              pytest.approx(0.3659668)]


def test_detect_ignores_a_spike_on_too_few_buildings():
    pred = pd.concat([_smooth(per_value=1), pd.Series([0.3659668] * 20)], ignore_index=True)
    assert ev._detect_nodata_values(pred) == []


def test_drop_uses_the_values_it_is_given():
    rows = ([("01001020100", 2016, 0.5, 0.1)] * 8
            + [("01001020100", 2016, 0.3659668, 0.1)] * 2
            + [("01001020100", 2016, BLACK, 0.1)] * 1)      # ordinal value: not this run's
    out = ev._drop_nodata_predictions(_bld(rows), values=[0.3659668])
    assert len(out) == 9 and (out["pred"].round(6) != 0.365967).all()
    assert len(ev._drop_nodata_predictions(_bld(rows), values=[])) == 11
