"""Tests for compute_val_metrics (#31): per-bracket / per-city validation metrics.

Pure synthetic DataFrames, hand-computed expectations. The MASD/DA numbers mirror
the pre-refactor inline logic in train_model, proving the extraction is
behavior-preserving.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.main import compute_val_metrics


def _val_df():
    """Cities 101/102 (mega) + 201 (small).

    Buildings:
      b1 (city 101, stable):  2014 pred 0.5 / 2018 pred 0.7  -> stable disp 0.2
      b2 (city 101, changed): 2014 pred 0.0 lbl -1 / 2018 pred 1.0 lbl 1
                              -> changed disp 1.0, DA correct
      b3 (city 102, single year) and b4 (city 201, single year): no temporal pairs
    """
    return pd.DataFrame({
        "building_id": [1, 1, 2, 2, 3, 4],
        "year":        [2014, 2018, 2014, 2018, 2016, 2016],
        "pred":        [0.5, 0.7, 0.0, 1.0, -0.3, 2.0],
        "label":       [0.4, 0.5, -1.0, 1.0, -0.5, 1.5],
        "change":      [0, 0, 1, 1, 0, 0],
        "cbsa":        [101, 101, 101, 101, 102, 201],
    })


def _meta():
    return pd.DataFrame({
        "cbsa_code": ["101", "102", "201"],
        "bracket": ["mega", "mega", "small"],
        "population": [9_000_000, 8_000_000, 600_000],
        "split": ["val", "val", "val"],
    })


def test_base_metrics_match_hand_computation():
    df = _val_df()
    m = compute_val_metrics(df)
    assert m["mse"] == pytest.approx(float(np.mean((df["pred"] - df["label"]) ** 2)))
    assert m["spearman"] == pytest.approx(spearmanr(df["pred"], df["label"])[0])
    assert m["pred_mean"] == pytest.approx(df["pred"].mean())
    assert m["pred_std"] == pytest.approx(df["pred"].std())
    # Hand-computed temporal metrics (same rules as the old inline loop)
    assert m["stable_masd"] == pytest.approx(0.2)
    assert m["changed_masd"] == pytest.approx(1.0)
    assert m["changed_da"] == pytest.approx(1.0)


def test_no_metrics_on_empty_df():
    assert compute_val_metrics(pd.DataFrame(
        columns=["building_id", "year", "pred", "label", "change", "cbsa"])) == {}


def test_bracket_breakdown():
    m = compute_val_metrics(_val_df(), cbsa_meta=_meta(), top_cities=None)
    # mega = cities 101+102 (5 rows), small = city 201 (1 row)
    assert m["bracket/mega/n"] == 5
    assert m["bracket/small/n"] == 1
    mega = _val_df().query("cbsa in (101, 102)")
    assert m["bracket/mega/mse"] == pytest.approx(
        float(np.mean((mega["pred"] - mega["label"]) ** 2)))
    assert m["bracket/mega/spearman"] == pytest.approx(
        spearmanr(mega["pred"], mega["label"])[0])
    assert m["bracket/mega/pred_mean"] == pytest.approx(mega["pred"].mean())
    # MASD lives entirely inside the mega bracket here
    assert m["bracket/mega/stable_masd"] == pytest.approx(0.2)
    assert m["bracket/small/pred_mean"] == pytest.approx(2.0)
    # single constant row -> no spearman key
    assert "bracket/small/spearman" not in m


def test_city_breakdown_respects_min_n():
    m = compute_val_metrics(_val_df(), cbsa_meta=_meta(),
                            top_cities=["101", "102", "201"], min_city_n=2)
    assert m["city/101/n"] == 4
    assert m["city/101/pred_mean"] == pytest.approx(0.55)
    # cities with fewer than min_city_n rows are silently skipped
    assert not any(k.startswith("city/102/") for k in m)
    assert not any(k.startswith("city/201/") for k in m)


def test_city_breakdown_only_for_top_cities():
    m = compute_val_metrics(_val_df(), cbsa_meta=_meta(),
                            top_cities=["102"], min_city_n=1)
    assert any(k.startswith("city/102/") for k in m)
    assert not any(k.startswith("city/101/") for k in m)


def test_no_cbsa_meta_gives_base_metrics_only():
    m = compute_val_metrics(_val_df(), cbsa_meta=None, top_cities=None)
    assert "mse" in m and "spearman" in m
    assert not any("/" in k for k in m)


def test_old_shards_cbsa_zero_no_crash():
    df = _val_df().assign(cbsa=0)
    m = compute_val_metrics(df, cbsa_meta=_meta(), top_cities=["101"])
    # cbsa 0 maps to no bracket -> only base metrics, no breakdown keys
    assert "mse" in m
    assert not any(k.startswith(("bracket/", "city/")) for k in m)
