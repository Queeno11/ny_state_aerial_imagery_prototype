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


def test_naip_manager_owns_tract_search_cache(manager):
    from src.data.naip_fetcher import TractSearchCache
    assert isinstance(manager._naip_search_cache, TractSearchCache)


def test_extract_raw_image_passes_tract_cache(manager, monkeypatch):
    """_extract_raw_image must route fetches through the manager's shared
    TractSearchCache, keyed on the row's GEOID (and disable the cache for
    rows without one)."""
    import src.data.naip_fetcher as nf

    seen = []

    def fake_fetch(lon, lat, crop_size_meters, nbands=4, out_pixels=250,
                   year_hint=None, search_cache=None, cache_key=None):
        seen.append({"search_cache": search_cache, "cache_key": cache_key,
                     "year_hint": year_hint})
        return nf.NaipFetchResult(None, None, failure="no_items")

    monkeypatch.setattr(nf, "fetch_naip", fake_fetch)

    row = manager.df.iloc[0]
    assert manager._extract_raw_image(row, n_bands=4) is None
    assert seen[0]["search_cache"] is manager._naip_search_cache
    assert seen[0]["cache_key"] == "10003000100"
    assert seen[0]["year_hint"] == 2014

    # No GEOID → cache disabled for that row, never a bogus key.
    row_no_geoid = dict(row)
    row_no_geoid["GEOID"] = None
    manager._extract_raw_image(row_no_geoid, n_bands=4)
    assert seen[1]["cache_key"] is None
    assert seen[1]["search_cache"] is manager._naip_search_cache


# NOTE: the manager-level stable/change building pools were removed with the
# normalization refactor — they were write-only (the sampler uses the
# shard-level dicts from InBatchRankingDataset.refresh()) and cost many GB at
# 71.8M buildings.


# ── CyclicCacheManager × LazyPairTable (normalized ms_us path) ───────────────

def _tiny_pair_table():
    from src.data.pair_table import LazyPairTable
    buildings = pd.DataFrame({
        "building_id": [1, 2],
        "GEOID": pd.Categorical(["10003000100"] * 2),
        "cbsa_code": pd.Categorical(["37980"] * 2),
        "centroid_x": np.float32([1.7e6, 1.7e6 + 50]),
        "centroid_y": np.float32([2.0e6, 2.0e6 + 50]),
        "dist_to_center": np.float32([1.0, 2.0]),
    })
    labels = pd.DataFrame({
        "GEOID": ["10003000100"] * 2,
        "year": [2014, 2018],
        "Rel_Score": np.float32([0.1, 0.2]),
        "Valid_Structural_Change": np.int8([0, 0]),
        "score_bin": np.int8([0, 0]),
    })
    return LazyPairTable(buildings, labels, [2014, 2018], tau_meters=100,
                         split_type="train")


def test_lazy_manager_init_and_slice(tmp_path, monkeypatch):
    monkeypatch.setattr(bd, "process_acs_panel", lambda: _fake_panel())
    table = _tiny_pair_table()
    mgr = main.CyclicCacheManager(
        df=table, all_years_datasets=None,
        params={"nbands": 4, "image_size": 64, "tau_meters": 100,
                "subsample_step": 1, "indicator": "W2_r5"},
        cache_dir=tmp_path, type="train", clear_cache=True, sat_data="NAIP",
    )
    assert mgr._is_lazy and len(mgr.df) == 4
    # The shard generator's slice call: cyclic materialization works
    sliced = mgr.df.materialize(2, 6)
    assert len(sliced) == 4
    assert list(sliced["building_id"]) == [2, 2, 1, 1]   # wraps around


def test_lazy_manager_tract_sampling_branch(tmp_path, monkeypatch):
    """tract_sampling=True routes shard sourcing through materialize_tract_sample
    (one building per drawn tract, all years adjacent) instead of the cyclic
    building-major slice; without the flag, the legacy slice is unchanged."""
    from src.data.tract_sampling import GradientHardnessRegistry

    monkeypatch.setattr(bd, "process_acs_panel", lambda: _fake_panel())
    base_params = {"nbands": 4, "image_size": 64, "tau_meters": 100,
                   "subsample_step": 1, "indicator": "W2_r5"}

    mgr = main.CyclicCacheManager(
        df=_tiny_pair_table(), all_years_datasets=None,
        params={**base_params, "tract_sampling": True, "sampling_seed": 3},
        cache_dir=tmp_path, type="train", clear_cache=True, sat_data="NAIP",
        shard_size=4, hardness_registry=GradientHardnessRegistry(alpha=0.3),
    )
    out = mgr._shard_source_df(0)
    # 1 tract in the table -> 1 building x both years, regardless of the
    # tract's 2 buildings (building-count weighting is gone)
    assert len(out) == 2
    assert out["building_id"].nunique() == 1
    assert sorted(out["year"]) == [2014, 2018]
    # deterministic per shard_id
    pd.testing.assert_frame_equal(out, mgr._shard_source_df(0))

    legacy = main.CyclicCacheManager(
        df=_tiny_pair_table(), all_years_datasets=None,
        params=base_params, cache_dir=tmp_path, type="train",
        clear_cache=True, sat_data="NAIP", shard_size=4,
    )
    out = legacy._shard_source_df(0)
    assert len(out) == 4                      # cyclic slice: both buildings
    assert out["building_id"].nunique() == 2


def test_lazy_manager_requires_naip(tmp_path, monkeypatch):
    monkeypatch.setattr(bd, "process_acs_panel", lambda: _fake_panel())
    with pytest.raises(NotImplementedError, match="NAIP"):
        main.CyclicCacheManager(
            df=_tiny_pair_table(), all_years_datasets=None,
            params={"nbands": 4, "image_size": 64, "tau_meters": 100,
                    "subsample_step": 1, "indicator": "W2_r5"},
            cache_dir=tmp_path, type="train", clear_cache=True, sat_data="aerial",
        )


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


def test_subsample_val_buildings_default_one_per_tract():
    df_val = pd.DataFrame({
        "GEOID": ["A"] * 8 + ["B"] * 2,
        "building_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "year": [2014, 2018] * 5,
    })
    out = main._subsample_val_buildings(df_val)
    per_tract = out.drop_duplicates("building_id").groupby("GEOID").size()
    assert (per_tract == 1).all()
    # all years of each kept building survive
    assert (out.groupby("building_id")["year"].nunique() == 2).all()


def test_subsample_val_buildings_shuffles_at_building_level():
    n = 200
    df_val = pd.DataFrame({
        "GEOID": [f"T{i}" for i in range(n) for _ in (0, 1)],
        "building_id": [i for i in range(n) for _ in (0, 1)],
        "year": [2014, 2018] * n,
    })
    out = main._subsample_val_buildings(df_val, seed=825)
    # building rows stay contiguous with years sorted...
    bids = out["building_id"].to_numpy()
    assert all(bids[i] == bids[i + 1] for i in range(0, len(bids), 2))
    assert (out.groupby("building_id", sort=False)["year"].apply(
        lambda s: s.is_monotonic_increasing)).all()
    # ...but building ORDER is shuffled (not the input head-first order),
    # so shard-cap truncation takes a random sample, not the first tracts.
    assert list(out["building_id"].drop_duplicates()) != list(range(n))
    # idempotent: re-applying (as happens to pre-sampled lazy vals) is a no-op
    out2 = main._subsample_val_buildings(out, seed=825)
    pd.testing.assert_frame_equal(
        out.sort_values(["building_id", "year"]).reset_index(drop=True),
        out2.sort_values(["building_id", "year"]).reset_index(drop=True),
    )


# ── wandb init fallback (deleted/tombstoned run ids) ─────────────────────────

def test_wandb_id_unusable_detection():
    CommError = main.wandb.errors.CommError
    # direct 409 message from the server
    assert main._wandb_id_is_unusable(
        CommError("run run_x was previously created and deleted; try a new id"))
    # same condition surfacing as an init timeout after wandb's internal retries
    assert main._wandb_id_is_unusable(
        CommError("Run initialization has timed out after 90.0 sec."))
    # unrelated comm failures must still propagate
    assert not main._wandb_id_is_unusable(CommError("permission denied"))


def test_init_wandb_run_falls_back_on_timeout(monkeypatch):
    CommError = main.wandb.errors.CommError
    calls = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise CommError("Run initialization has timed out after 90.0 sec.")
        return "run-handle"

    monkeypatch.setattr(main.wandb, "init", fake_init)
    out = main.init_wandb_run("run_x", {"p": 1}, wandb_resuming=True)
    assert out == "run-handle"
    assert calls[0]["id"] == "run_x"
    assert calls[1]["id"].startswith("run_x_") and calls[1]["id"].endswith("_retry")


def test_init_wandb_run_reraises_other_commerror(monkeypatch):
    CommError = main.wandb.errors.CommError

    def fake_init(**kwargs):
        raise CommError("permission denied")

    monkeypatch.setattr(main.wandb, "init", fake_init)
    with pytest.raises(CommError, match="permission denied"):
        main.init_wandb_run("run_x", {}, wandb_resuming=True)


def test_init_wandb_run_fresh_run_gets_suffixed_id(monkeypatch):
    calls = []
    monkeypatch.setattr(main.wandb, "init", lambda **kw: calls.append(kw) or "h")
    main.init_wandb_run("run_x", {}, wandb_resuming=False)
    assert calls[0]["id"] != "run_x"
    assert calls[0]["id"].startswith("run_x_")


# ── resume lr override (decay lr across a resume without resetting training) ─

def _tiny_adamw():
    import torch
    model = torch.nn.Linear(4, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # one real step so exp_avg/exp_avg_sq/step exist in the state
    model(torch.randn(2, 4)).sum().backward()
    opt.step()
    return model, opt


def test_override_optimizer_lr_none_is_noop():
    _, opt = _tiny_adamw()
    main.override_optimizer_lr(opt, None)
    assert all(pg["lr"] == 1e-4 for pg in opt.param_groups)


def test_override_optimizer_lr_survives_state_dict_roundtrip():
    import torch
    model, opt = _tiny_adamw()
    saved = opt.state_dict()

    fresh = torch.optim.AdamW(model.parameters(), lr=999.0)  # params say one thing...
    fresh.load_state_dict(saved)                             # ...checkpoint restores 1e-4
    assert fresh.param_groups[0]["lr"] == 1e-4

    main.override_optimizer_lr(fresh, 3e-5)
    assert all(pg["lr"] == 3e-5 for pg in fresh.param_groups)
    # only lr changed: moments and step counts from the checkpoint are intact
    old_state = saved["state"][0]
    new_state = fresh.state_dict()["state"][0]
    assert torch.equal(old_state["exp_avg"], new_state["exp_avg"])
    assert torch.equal(old_state["exp_avg_sq"], new_state["exp_avg_sq"])
    assert old_state["step"] == new_state["step"]
    assert fresh.param_groups[0]["weight_decay"] == saved["param_groups"][0]["weight_decay"]


def test_resume_lr_override_is_a_known_param():
    base = {"model_name": "scalemae", "kind": "reg", "sat_data": "NAIP",
            "years": [2022], "nbands": 3, "image_size": 224, "weights": None}
    merged = main.fill_params_defaults({**base, "resume_lr_override": 3e-5})
    assert merged["resume_lr_override"] == 3e-5
    # and defaults to None (no override) when not supplied
    assert main.fill_params_defaults(dict(base))["resume_lr_override"] is None
