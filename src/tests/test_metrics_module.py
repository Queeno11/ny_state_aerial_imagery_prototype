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
    selection_score,
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
    assert cells["n_tracts"].iloc[0] == n  # distinct labels -> one tract each
    exp_rho = spearmanr(df["pred"], df["label"]).statistic
    assert cells["rho"].iloc[0] == pytest.approx(exp_rho)
    head = weighted_within_spearman(cells)
    assert head["within_spearman"] == pytest.approx(exp_rho)
    assert head["within_cells"] == 1
    assert head["within_n"] == n
    assert head["within_tracts"] == n


def test_within_weighting_is_by_tracts_not_buildings():
    # Cell A ("downtown"): 8 buildings but only 2 distinct tract labels.
    # Cell B ("suburb"):   5 buildings, 5 distinct tract labels.
    # Building-count weights would give A 8/13 of the mean; tract weights 2/7.
    a = pd.DataFrame({
        "building_id": range(8), "year": 2016,
        "pred": [1, 2, 3, 4, 5, 6, 7, 8],
        "label": [0.1] * 4 + [0.9] * 4,   # 2 tracts, 4 buildings each
        "change": 0, "cbsa": 101,
    })
    b = pd.DataFrame({
        "building_id": range(100, 105), "year": 2016,
        "pred": [1, 2, 3, 4, 5],
        "label": [0.2, 0.3, 0.5, 0.4, 0.6],  # 5 tracts
        "change": 0, "cbsa": 102,
    })
    cells = within_city_cells(pd.concat([a, b]))
    assert dict(zip(cells["cbsa"], cells["n_tracts"])) == {101: 2, 102: 5}
    rho_a = spearmanr(a["pred"], a["label"]).statistic
    rho_b = spearmanr(b["pred"], b["label"]).statistic
    head = weighted_within_spearman(cells)
    assert head["within_spearman"] == pytest.approx((2 * rho_a + 5 * rho_b) / 7)
    assert head["within_n"] == 13
    assert head["within_tracts"] == 7


def test_within_tract_count_prefers_geoid_column():
    # With a GEOID column present, tract count must come from it — here labels
    # are all distinct (6 would-be "tracts") but GEOID says 3.
    n = 6
    df = pd.DataFrame({
        "building_id": range(n), "year": 2016,
        "pred": [1, 2, 3, 4, 5, 6],
        "label": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "change": 0, "cbsa": 101,
        "GEOID": ["36061000100", "36061000100", "36061000200",
                  "36061000200", "36061000300", "36061000300"],
    })
    cells = within_city_cells(df)
    assert cells["n_tracts"].iloc[0] == 3
    assert cells["n"].iloc[0] == n


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


# ---- selection_score -----------------------------------------------------


def _sel_metrics():
    return {
        "val_cities":         {"within_spearman": 0.40, "within_tracts": 300, "spearman": 0.45},
        "val_within_nyc":     {"within_spearman": 0.30, "within_tracts": 100, "spearman": 0.35},
        "val_within_chicago": {"within_spearman": 0.60, "within_tracts": 100, "spearman": 0.65},
        "val_temporal":       {"within_spearman": 0.58, "within_tracts": 200, "spearman": 0.59},
    }


def test_selection_within_50_50_tract_weighted_pooling():
    score, comp = selection_score(_sel_metrics(), mode="within_50_50")
    # Cross-sectional side pools the three non-temporal sets by tract weight:
    cs = (0.40 * 300 + 0.30 * 100 + 0.60 * 100) / 500  # = 0.42
    assert comp["cross_sectional_within"] == pytest.approx(cs)
    assert comp["temporal_within"] == pytest.approx(0.58)
    assert score == pytest.approx(0.5 * cs + 0.5 * 0.58)


def test_selection_excludes_diagnostic_stability_set():
    # val_stability is a few noisy multi-year buildings; it must not perturb
    # the checkpoint-selection score (default diagnostic_sets excludes it).
    base = _sel_metrics()
    score_base, comp_base = selection_score(base, mode="within_50_50")
    poisoned = dict(base)
    poisoned["val_stability"] = {"within_spearman": -0.9, "within_tracts": 400}
    score, comp = selection_score(poisoned, mode="within_50_50")
    assert score == pytest.approx(score_base)
    assert comp["cross_sectional_within"] == pytest.approx(comp_base["cross_sectional_within"])
    # opting the set back in (empty exclusion) does drag the score down
    score_in, _ = selection_score(poisoned, mode="within_50_50", diagnostic_sets=())
    assert score_in < score_base


def test_selection_within_50_50_matches_pooled_cells():
    # Weighting set-level within_spearmans by within_tracts must equal the
    # tract-weighted mean over the union of the sets' cells.
    cells_a = pd.DataFrame({"rho": [0.2, 0.6], "n": [10, 10], "n_tracts": [10, 30]})
    cells_b = pd.DataFrame({"rho": [0.8], "n": [10], "n_tracts": [60]})
    m_a = weighted_within_spearman(cells_a)
    m_b = weighted_within_spearman(cells_b)
    score, comp = selection_score(
        {"val_cities": m_a, "val_within_chicago": m_b,
         "val_temporal": {"within_spearman": 0.0, "within_tracts": 1}},
        mode="within_50_50",
    )
    pooled = np.average([0.2, 0.6, 0.8], weights=[10, 30, 60])
    assert comp["cross_sectional_within"] == pytest.approx(pooled)
    assert score == pytest.approx(pooled / 2)


def test_selection_within_50_50_fallback_and_missing_sides():
    # Set without within cells falls back to pooled spearman (weight n).
    score, comp = selection_score(
        {"val_cities": {"spearman": 0.5, "n": 100},
         "val_temporal": {"within_spearman": 0.7, "within_tracts": 10}},
        mode="within_50_50",
    )
    assert comp["cross_sectional_within"] == pytest.approx(0.5)
    assert score == pytest.approx(0.6)
    # Missing temporal side: cross-sectional side carries the score alone.
    score, comp = selection_score(
        {"val_cities": {"within_spearman": 0.4, "within_tracts": 10}},
        mode="within_50_50",
    )
    assert "temporal_within" not in comp
    assert score == pytest.approx(0.4)
    # Nothing usable at all.
    score, comp = selection_score({"val_cities": {"mse": 1.0}}, mode="within_50_50")
    assert score is None and comp == {}


def test_selection_legacy_modes_are_unweighted_set_means():
    m = _sel_metrics()
    score_w, comp_w = selection_score(m, mode="within")
    assert score_w == pytest.approx(np.mean([0.40, 0.30, 0.60, 0.58]))
    score_p, _ = selection_score(m, mode="pooled")
    assert score_p == pytest.approx(np.mean([0.45, 0.35, 0.65, 0.59]))
    # "within" falls back per set to pooled spearman when within is missing.
    m["val_cities"] = {"spearman": 0.45}
    score_f, comp_f = selection_score(m, mode="within")
    assert comp_f["val_cities"] == pytest.approx(0.45)
    with pytest.raises(ValueError):
        selection_score(m, mode="nonsense")
