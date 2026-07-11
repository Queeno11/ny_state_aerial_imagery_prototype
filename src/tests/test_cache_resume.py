"""Crash-resume of the NAIP shard cache (no GPU, no network).

Covers the three pieces that let an interrupted cache build resume instead of
refetching every NAIP tile:
  1. cache_run_marker.json read/write (run() keeps shards when the marker
     matches the current savename even with no model checkpoint yet),
  2. build_initial_cache() topping up only the missing shards,
  3. atomic shard saves (.pt.tmp -> rename) + stale-tmp cleanup on init.
"""

import numpy as np
import pandas as pd
import pytest
import torch

import src.build_dataset as bd
import src.main as main


# ── cache_run_marker ─────────────────────────────────────────────────────────

def test_marker_roundtrip(tmp_path):
    assert main.read_cache_marker(tmp_path) is None
    main.write_cache_marker("run_x", tmp_path)
    assert main.read_cache_marker(tmp_path) == "run_x"
    main.write_cache_marker("run_y", tmp_path)
    assert main.read_cache_marker(tmp_path) == "run_y"


def test_marker_corrupt_file_returns_none(tmp_path):
    main._cache_marker_path(tmp_path).write_text("not json {")
    assert main.read_cache_marker(tmp_path) is None


# ── CyclicCacheManager resume behaviour ──────────────────────────────────────

def _fake_panel():
    years = list(range(2011, 2024))
    rows = []
    for i in range(4):
        row = {"geoid_2023": f"1000300010{i}", "cbsa_code": "37980"}
        for y in years:
            row[f"Rel_Score_{y}"] = float(i)
            row[f"Rel_Score_W2_i_r5pct_{y}"] = 100.0 + i
        rows.append(row)
    return pd.DataFrame(rows)


def _make_manager(tmp_path, monkeypatch, *, num_shards=3, clear_cache=False,
                  type="train", holdout_years=None):
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
        cache_dir=tmp_path, type=type, num_shards=num_shards,
        shard_size=2, clear_cache=clear_cache, sat_data="NAIP",
        holdout_years=holdout_years,
    )


def _touch_shard(cache_dir, shard_id):
    path = cache_dir / "train_cache" / f"shard_{shard_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": torch.empty(0), "scores": torch.empty(0)}, path)
    return path


def test_partial_cache_tops_up_missing_shards(tmp_path, monkeypatch):
    """2 of 3 shards on disk -> only shards 2 is generated, 0/1 are kept."""
    _touch_shard(tmp_path, 0)
    _touch_shard(tmp_path, 1)
    mgr = _make_manager(tmp_path, monkeypatch, num_shards=3)
    assert len(mgr.active_shards) == 2
    assert mgr.next_shard_idx == 2

    generated = []
    monkeypatch.setattr(mgr, "_worker_generate",
                        lambda sid, show_progress=False: generated.append(sid))
    mgr.build_initial_cache()
    assert generated == [2]
    assert len(mgr.active_shards) == 3
    assert mgr._is_initialized


def test_complete_cache_generates_nothing(tmp_path, monkeypatch):
    for i in range(3):
        _touch_shard(tmp_path, i)
    mgr = _make_manager(tmp_path, monkeypatch, num_shards=3)
    monkeypatch.setattr(mgr, "_worker_generate",
                        lambda *a, **k: pytest.fail("should not regenerate"))
    mgr.build_initial_cache()
    assert len(mgr.active_shards) == 3
    assert mgr._is_initialized


def test_clear_cache_rebuilds_all(tmp_path, monkeypatch):
    _touch_shard(tmp_path, 0)
    _touch_shard(tmp_path, 1)
    mgr = _make_manager(tmp_path, monkeypatch, num_shards=3, clear_cache=True)
    assert mgr.active_shards == []
    generated = []
    monkeypatch.setattr(mgr, "_worker_generate",
                        lambda sid, show_progress=False: generated.append(sid))
    mgr.build_initial_cache()
    assert generated == [0, 1, 2]
    assert not list((tmp_path / "train_cache").glob("*.pt.tmp"))


def test_stale_tmp_files_removed_on_init(tmp_path, monkeypatch):
    """A crash mid-torch.save leaves shard_N.pt.tmp -- never picked up as a shard."""
    _touch_shard(tmp_path, 0)
    stale = tmp_path / "train_cache" / "shard_1.pt.tmp"
    stale.write_bytes(b"partial write")
    mgr = _make_manager(tmp_path, monkeypatch, num_shards=3)
    assert not stale.exists()
    assert [p.name for p in mgr.active_shards] == ["shard_0.pt"]


def test_worker_generate_saves_atomically(tmp_path, monkeypatch):
    """The final shard file only appears via rename; no lingering .tmp."""
    mgr = _make_manager(tmp_path, monkeypatch, num_shards=1)
    fake_crop = np.full((4, 64, 64), 7, dtype=np.uint8)
    monkeypatch.setattr(
        mgr, "_extract_raw_image",
        lambda row, n_bands=None, pad=0: (fake_crop, int(row["year"])))
    mgr._worker_generate(0)
    shard = tmp_path / "train_cache" / "shard_0.pt"
    assert shard.exists()
    assert not shard.with_suffix(".pt.tmp").exists()
    data = torch.load(shard, weights_only=False)
    assert len(data["images"]) == 2  # shard_size


# ── NAIP failure-rate halt vs holdout rejects ────────────────────────────────

def test_holdout_rejects_do_not_trip_failure_halt(tmp_path, monkeypatch):
    """val_temporal rejecting most crops (wrong flight year) is filtering, not an
    API failure: the >50% rate-limit halt must not fire."""
    mgr = _make_manager(tmp_path, monkeypatch, num_shards=1,
                        type="val_temporal", holdout_years={37980: 2020})
    fake_crop = np.full((4, 64, 64), 7, dtype=np.uint8)
    # actual_year == requested year (2014/2018) != holdout 2020 -> all rejected.
    monkeypatch.setattr(
        mgr, "_extract_raw_image",
        lambda row, n_bands=None, pad=0: (fake_crop, int(row["year"])))
    mgr._worker_generate(0)  # must not raise
    assert mgr.year_sub_counts["holdout_reject"] == 2
    data = torch.load(tmp_path / "val_temporal_cache" / "shard_0.pt",
                      weights_only=False)
    assert len(data["images"]) == 0


def test_genuine_fetch_failures_still_halt(tmp_path, monkeypatch):
    mgr = _make_manager(tmp_path, monkeypatch, num_shards=1)
    monkeypatch.setattr(mgr, "_extract_raw_image", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="rate limit"):
        mgr._worker_generate(0)
