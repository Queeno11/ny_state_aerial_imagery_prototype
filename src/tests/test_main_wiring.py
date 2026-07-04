"""Wiring tests for main.py (no GPU, no network): savename, params, acs_lookup."""

import re

import numpy as np
import pandas as pd
import pytest

import src.build_dataset as bd
import src.main as main
from src.data import indicators


# ── generate_savename ────────────────────────────────────────────────────────

def test_savename_default_is_dated_run():
    assert re.fullmatch(r"run_\d{8}", main.generate_savename())


def test_savename_run_id_passthrough():
    assert main.generate_savename("my_experiment") == "my_experiment"


# ── fill_params_defaults ─────────────────────────────────────────────────────

def test_defaults_include_us_scale_params():
    params = main.fill_params_defaults({"sat_data": "NAIP", "nbands": 4,
                                        "years": [2016], "image_size": 224,
                                        "weights": None})
    assert params["indicator"] == indicators.DEFAULT_INDICATOR == "W2_r5"
    assert params["footprints_source"] == "ms_us"
    assert params["states"] is None
    assert params["run_id"] is None
    assert params["reject_padded_nir"] is False


def test_unknown_param_rejected():
    with pytest.raises(ValueError, match="Invalid parameter"):
        main.fill_params_defaults({"sat_data": "NAIP", "nbands": 4,
                                   "years": [2016], "image_size": 224,
                                   "weights": None, "bogus_key": 1})


def test_lambda_c_param_rejected():
    """lambda_c was removed with L_change (#30): stale configs must fail loudly."""
    with pytest.raises(ValueError, match="Invalid parameter"):
        main.fill_params_defaults({"sat_data": "NAIP", "nbands": 4,
                                   "years": [2016], "image_size": 224,
                                   "weights": None, "lambda_c": 0.0})


# ── CyclicCacheManager NAIP init: indicator-scoped acs_lookup ────────────────

def _fake_panel():
    years = list(range(2011, 2024))
    rows = []
    for i in range(4):
        row = {"geoid_2023": f"1000300010{i}", "cbsa_code": "37980"}
        for y in years:
            row[f"Rel_Score_{y}"] = float(i)             # income scores
            row[f"Rel_Score_W2_i_r5pct_{y}"] = 100.0 + i  # W2 scores (distinct)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(bd, "process_acs_panel", lambda: _fake_panel())
    df = pd.DataFrame({
        "building_id": [1, 1, 2, 2],
        "year": [2014, 2018, 2014, 2018],
        "GEOID": ["10003000100"] * 4,
        "cbsa_code": ["37980"] * 4,
        "Rel_Score": [0.1] * 4,
        "Valid_Structural_Change": [0, 0, 1, 1],
        "centroid_x": [1.7e6] * 4,
        "centroid_y": [2.0e6] * 4,
    })
    return main.CyclicCacheManager(
        df=df, all_years_datasets=None,
        params={"nbands": 4, "image_size": 64, "tau_meters": 100,
                "subsample_step": 1, "indicator": "W2_r5"},
        cache_dir=tmp_path, type="train", clear_cache=True, sat_data="NAIP",
    )


def test_acs_lookup_scoped_to_selected_indicator(manager):
    # Lookup must carry the W2 values (100+i), never the income values (i).
    val = manager.acs_lookup[("10003000100", 2016)]
    assert val == pytest.approx(100.0)
    assert all(v >= 100.0 for v in manager.acs_lookup.values())
    # Keyed on geoid_2023 x every panel year
    assert manager.panel_years == list(range(2011, 2024))
    assert len(manager.acs_lookup) == 4 * 13


def test_nearest_panel_year_selection(manager):
    # The fallback rule used in _extract: nearest panel year to the actual year.
    assert min(manager.panel_years, key=lambda y: abs(y - 2009)) == 2011
    assert min(manager.panel_years, key=lambda y: abs(y - 2024)) == 2023
    assert manager.year_sub_counts == {"exact": 0, "nearest_fallback": 0,
                                       "miss": 0, "holdout_reject": 0}


def test_stable_change_pools(manager):
    assert set(manager.stable_building_ids) == {1}
    assert set(manager.change_building_ids) == {2}


# ── _subsample_val_buildings (val cache subsampling, #28/#31) ────────────────

def test_subsample_val_buildings_keeps_all_years_of_chosen_buildings():
    df_val = pd.DataFrame({
        "GEOID": ["A"] * 8 + ["B"] * 2,
        "building_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "year": [2014, 2018] * 5,
    })
    out = main._subsample_val_buildings(df_val, n_buildings_per_tract=2, seed=0)
    # <= 2 buildings per tract, and every kept building retains BOTH years
    for geoid, grp in out.groupby("GEOID"):
        assert grp["building_id"].nunique() <= 2
        for _, bgrp in grp.groupby("building_id"):
            assert set(bgrp["year"]) == {2014, 2018}
    # tract B has only one building -> fully kept
    assert set(out.loc[out["GEOID"] == "B", "building_id"]) == {5}
    # deterministic
    out2 = main._subsample_val_buildings(df_val, n_buildings_per_tract=2, seed=0)
    pd.testing.assert_frame_equal(out, out2)
