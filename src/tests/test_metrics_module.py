"""Tests for src/utils/metrics.py — the shared metric functions extracted from
main.py. Pure synthetic DataFrames, hand-computed expectations.

These import ONLY src.utils.metrics (numpy/pandas/scipy) — no torch, no paths —
so they run even where the heavy pipeline import chain is unavailable.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.utils.metrics import (
    within_city_cells,
    weighted_within_spearman,
    rank_autocorrelation,
    masd_by_change,
)


def _val_df():
    return pd.DataFrame({
        "building_id": [1, 1, 2, 2, 3, 4],
        "year":        [2014, 2018, 2014, 2018, 2016, 2016],
        "pred":        [0.5, 0.7, 0.0, 1.0, -0.3, 2.0],
        "label":       [0.4, 0.5, -1.0, 1.0, -0.5, 1.5],
        "change":      [0, 0, 1, 1, 0, 0],
        "cbsa":        [101, 101, 101, 101, 102, 201],
    })


def test_masd_by_change_hand_computed():
    m = masd_by_change(_val_df())
    # b1 stable: |0.7-0.5| = 0.2 ; b2 changed: |1.0-0.0| = 1.0, DA correct
    assert m["stable_masd"] == pytest.approx(0.2)
    assert m["changed_masd"] == pytest.approx(1.0)
    assert m["changed_da"] == pytest.approx(1.0)


def test_masd_empty_when_no_repeats():
    df = _val_df().drop_duplicates("building_id")  # every building once
    assert masd_by_change(df) == {}


def test_within_city_cells_and_weighting():
    # One city, one year, 6 buildings with a known positive rank relation.
    n = 6
    df = pd.DataFrame({
        "building_id": range(n),
        "year": [2016] * n,
        "pred": [1, 2, 3, 4, 5, 6],
        "label": [1, 2, 3, 4, 6, 5],  # near-monotone
        "change": [0] * n,
        "cbsa": [500] * n,
    })
    cells = within_city_cells(df)
    assert len(cells) == 1
    assert cells["n"].iloc[0] == n
    exp_rho = spearmanr(df["pred"], df["label"]).statistic
    assert cells["rho"].iloc[0] == pytest.approx(exp_rho)
    head = weighted_within_spearman(cells)
    assert head["within_spearman"] == pytest.approx(exp_rho)
    assert head["within_cells"] == 1
    assert head["within_n"] == n


def test_within_city_cells_drops_small_and_constant():
    # Cell A: 3 buildings (< min_bld=5) -> dropped.
    # Cell B: 5 buildings but constant label -> dropped.
    df = pd.DataFrame({
        "building_id": list(range(3)) + list(range(5)),
        "year": [2016] * 3 + [2018] * 5,
        "pred": [1, 2, 3] + [1, 2, 3, 4, 5],
        "label": [3, 2, 1] + [7, 7, 7, 7, 7],
        "change": [0] * 8,
        "cbsa": [1] * 3 + [1] * 5,
    })
    assert len(within_city_cells(df)) == 0
    assert weighted_within_spearman(within_city_cells(df)) == {}


def test_rank_autocorrelation_widest_span_and_benchmark():
    # City 1: stable buildings observed in 2010, 2016 and 2024. The pair with
    # the most common buildings AND widest span is (2010, 2024).
    ids = [10, 11, 12, 13, 14]
    rows = []
    for yr, preds, labels in [
        (2010, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        (2016, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        (2024, [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]),  # labels fully reverse
    ]:
        for i, bid in enumerate(ids):
            rows.append({"building_id": bid, "year": yr, "pred": preds[i],
                         "label": labels[i], "change": 0, "cbsa": 1})
    df = pd.DataFrame(rows)
    out = rank_autocorrelation(df)
    # pred is identical across years -> rho_p = 1; labels reverse over the wide
    # pair (2010 vs 2024) -> rho_l = -1.
    assert out["rank_autocorr_pred"] == pytest.approx(1.0)
    assert out["rank_autocorr_label"] == pytest.approx(-1.0)
    assert out["rank_autocorr_n"] == len(ids)


def test_rank_autocorrelation_empty_without_common():
    df = _val_df()  # only building 1 is stable & multi-year, n=1 < min_common
    assert rank_autocorrelation(df) == {}
