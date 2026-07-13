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


# ── goal-aligned metrics: within-city cells, rank autocorr, drift guardrails ─

from src.main import _rank_autocorrelation, _within_city_cells


def _cell_df(cbsa, year, n, offset=0.0, seed=3, change=0, bid_base=0):
    """One (city, year) cell: pred = label + offset (perfect within-cell rank)."""
    rng = np.random.default_rng(seed + cbsa + year)
    lbl = rng.standard_normal(n)
    return pd.DataFrame({
        "building_id": bid_base + np.arange(n), "year": year,
        "pred": lbl + offset, "label": lbl, "change": change, "cbsa": cbsa,
    })


def test_within_spearman_immune_to_city_offsets():
    # perfect rank within each city, but offsets scramble the pooled ranking
    df = pd.concat([_cell_df(101, 2016, 8, offset=+2.0, bid_base=0),
                    _cell_df(102, 2016, 8, offset=-2.0, bid_base=100)])
    m = compute_val_metrics(df)
    assert m["within_spearman"] == pytest.approx(1.0)
    assert m["within_cells"] == 2 and m["within_n"] == 16
    assert m["spearman"] < 0.6  # the pooled metric is fooled by the offsets


def test_within_cells_drop_small_and_degenerate_cells():
    ok = _cell_df(101, 2016, 6)
    small = _cell_df(102, 2016, 4, bid_base=100)              # < 5 buildings
    flat = _cell_df(103, 2016, 6, bid_base=200).assign(pred=1.0)  # no pred variation
    cells = _within_city_cells(pd.concat([ok, small, flat]))
    assert len(cells) == 1
    assert cells.iloc[0]["cbsa"] == 101 and cells.iloc[0]["n"] == 6


def _autocorr_df(n=6, years=(2014, 2020), reversed_years=(), change=0, cbsa=101):
    rows = []
    for y in years:
        for i in range(n):
            pred = float(n - 1 - i) if y in reversed_years else float(i)
            rows.append({"building_id": i, "year": y, "pred": pred,
                         "label": float(i), "change": change, "cbsa": cbsa})
    return pd.DataFrame(rows)


def test_rank_autocorr_consistent_ranks_and_change_exclusion():
    stable = _autocorr_df(n=6)
    # changed buildings with reversed ranks must NOT contaminate the metric
    changed = _autocorr_df(n=6, reversed_years=(2020,), change=1)
    changed["building_id"] += 100
    out = _rank_autocorrelation(pd.concat([stable, changed]))
    assert out["rank_autocorr_pred"] == pytest.approx(1.0)
    assert out["rank_autocorr_label"] == pytest.approx(1.0)
    assert out["rank_autocorr_n"] == 6


def test_rank_autocorr_prefers_widest_span_on_ties():
    # all three years share the same 6 buildings (tie on n): the (2014, 2020)
    # pair must win, where predictions are rank-REVERSED
    df = _autocorr_df(n=6, years=(2014, 2016, 2020), reversed_years=(2020,))
    out = _rank_autocorrelation(df)
    assert out["rank_autocorr_pred"] == pytest.approx(-1.0)
    assert out["rank_autocorr_label"] == pytest.approx(1.0)  # labels never reversed


def test_rank_autocorr_requires_min_common_buildings():
    assert _rank_autocorrelation(_autocorr_df(n=4)) == {}
    # single-year set (val_temporal): no year pair at all
    assert _rank_autocorrelation(_autocorr_df(n=8, years=(2016,))) == {}


def test_masd_ratio_and_city_offset_sd():
    # city 101: 5 stable buildings over two years (disp 0.2) + 1 changed (disp 1.0)
    a = pd.concat([
        pd.DataFrame({"building_id": np.arange(5), "year": 2014,
                      "pred": np.linspace(0, 1, 5) + 0.4, "label": np.linspace(0, 1, 5),
                      "change": 0, "cbsa": 101}),
        pd.DataFrame({"building_id": np.arange(5), "year": 2018,
                      "pred": np.linspace(0, 1, 5) + 0.6, "label": np.linspace(0, 1, 5),
                      "change": 0, "cbsa": 101}),
        pd.DataFrame({"building_id": [50, 50], "year": [2014, 2018],
                      "pred": [0.0, 1.0], "label": [-1.0, 1.0],
                      "change": 1, "cbsa": 101}),
    ])
    # city 102: 6 single-year rows whose pred mean sits exactly 1.0 below 101's
    b = _cell_df(102, 2016, 6, bid_base=100)
    b["pred"] = b["label"] + (a[a["cbsa"] == 101]["pred"].mean() - 1.0 - b["label"].mean())
    m = compute_val_metrics(pd.concat([a, b]))
    assert m["masd_ratio"] == pytest.approx(1.0 / 0.2)
    assert m["city_offset_sd"] == pytest.approx(0.5)


def test_bracket_and_city_within_spearman_keys():
    df = pd.concat([_cell_df(101, 2016, 8, offset=+2.0),
                    _cell_df(102, 2016, 8, offset=-2.0, bid_base=100),
                    _cell_df(201, 2016, 8, offset=0.0, bid_base=200)])
    m = compute_val_metrics(df, cbsa_meta=_meta(), top_cities=["101"], min_city_n=5)
    assert m["bracket/mega/within_spearman"] == pytest.approx(1.0)
    assert m["bracket/small/within_spearman"] == pytest.approx(1.0)
    assert m["city/101/within_spearman"] == pytest.approx(1.0)
    # pooled bracket spearman for mega is still the offset-contaminated one
    assert m["bracket/mega/spearman"] < 0.6


def test_within_metrics_survive_nyc_legacy_cbsa_zero():
    # legacy zarr shards: cbsa == 0 everywhere -> one "city", cells = years,
    # so the within metric degrades gracefully to NYC per-year spearman
    df = pd.concat([_cell_df(0, 2016, 8), _cell_df(0, 2018, 8, bid_base=0)])
    m = compute_val_metrics(df)
    assert m["within_spearman"] == pytest.approx(1.0)
    assert m["within_cells"] == 2


def test_float16_preds_from_autocast_do_not_crash():
    """Regression: the val loop yields float16 preds under autocast; pandas'
    MASKED unstack (pivot_table with a missing building x year cell) has no
    float16 kernel and raises 'No matching signature found' — crashed
    _rank_autocorrelation at epoch 493 of run_20260710. A complete grid does
    NOT trigger it, so the missing cell below is load-bearing."""
    df = _autocorr_df(n=7)
    df = df.drop(df[(df["building_id"] == 6) & (df["year"] == 2020)].index)
    df["pred"] = df["pred"].astype(np.float16)
    df["label"] = df["label"].astype(np.float16)
    m = compute_val_metrics(df)
    assert m["rank_autocorr_pred"] == pytest.approx(1.0)
    assert m["rank_autocorr_n"] == 6      # the dropped building cannot pair
    assert m["within_spearman"] == pytest.approx(1.0)
