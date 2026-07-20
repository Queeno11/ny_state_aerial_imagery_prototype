"""Synthetic-data tests: tract-first shard sampling + gradient hard-tract mining.

Covers the three new pieces end to end:
  * src/data/tract_sampling.py   (stable hash, hardness registry)
  * LazyPairTable.materialize_tract_sample (tract-first shard draws)
  * InBatchPairwiseRankingLoss lambda harvesting (analytic vs autograd)
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

import src.geo_utils  # noqa: F401  (breaks the pair_table<->build_dataset import cycle)
from src.data.pair_table import FLAT_COLUMNS, LazyPairTable
from src.data.tract_sampling import (
    GradientHardnessRegistry, groupby_mean, hash_geoids, stable_geoid_hash,
)

YEARS = [2012, 2014, 2018]
TAU = 100.0


def _synthetic_buildings(buildings_per_tract, seed=7):
    """Buildings over len(buildings_per_tract) tracts in one CBSA."""
    rng = np.random.default_rng(seed)
    rows, bid = [], 1
    for t, n in enumerate(buildings_per_tract):
        geoid = f"36{t:09d}"
        for _ in range(n):
            rows.append({
                "building_id": bid, "GEOID": geoid, "cbsa_code": "10000",
                "centroid_x": float(rng.uniform(0, 1e6)),
                "centroid_y": float(rng.uniform(0, 1e6)),
                "dist_to_center": float(rng.uniform(0, 30)),
            })
            bid += 1
    df = pd.DataFrame(rows)
    df["GEOID"] = df["GEOID"].astype("category")
    df["cbsa_code"] = df["cbsa_code"].astype("category")
    for c in ("centroid_x", "centroid_y", "dist_to_center"):
        df[c] = df[c].astype("float32")
    return df


def _synthetic_labels(buildings, years=YEARS, seed=11):
    rng = np.random.default_rng(seed)
    geoids = sorted(buildings["GEOID"].astype(str).unique())
    return pd.DataFrame([
        {"GEOID": g, "year": yr, "Rel_Score": np.float32(rng.normal()),
         "Valid_Structural_Change": np.int8(0), "score_bin": np.int8(i % 5)}
        for yr in years for i, g in enumerate(geoids)
    ])


def _table(buildings_per_tract=(3, 3, 3, 3), seed=7):
    b = _synthetic_buildings(buildings_per_tract, seed=seed)
    return LazyPairTable(b, _synthetic_labels(b), YEARS, tau_meters=TAU)


# ── stable hash ──────────────────────────────────────────────────────────────

def test_stable_geoid_hash_deterministic_and_bounded():
    h = stable_geoid_hash("36061000100")
    assert h == stable_geoid_hash("36061000100")      # no process salt
    assert 0 <= h < 2**31
    assert h != stable_geoid_hash("36061000200")
    arr = hash_geoids(["36061000100", "36061000200"])
    assert arr.dtype == np.int64 and arr[0] == h


def test_groupby_mean():
    keys = np.array([2, 1, 2, 1, 3])
    vals = np.array([1.0, 10.0, 3.0, 20.0, 7.0])
    uniq, means = groupby_mean(keys, vals)
    assert uniq.tolist() == [1, 2, 3]
    assert np.allclose(means, [15.0, 2.0, 7.0])


# ── registry ─────────────────────────────────────────────────────────────────

def test_registry_ema_and_weights():
    reg = GradientHardnessRegistry(alpha=0.5, beta=0.9, cap=10.0)
    # first update seeds the EMA directly; same-tract samples averaged first
    reg.update(np.array([1, 1, 2]), np.array([0.2, 0.4, 0.6]))
    w = reg.weights_for(np.array([1, 2, 99]))
    mean_ema = (0.3 + 0.6) / 2
    assert np.allclose(w, [0.5 + 0.5 * 0.3 / mean_ema,
                           0.5 + 0.5 * 0.6 / mean_ema,
                           1.0])                       # unseen tract == uniform
    # second update applies the EMA decay
    reg.update(np.array([1]), np.array([1.0]))
    assert np.isclose(reg._ema[1], 0.9 * 0.3 + 0.1 * 1.0)


def test_registry_alpha_zero_and_cap():
    reg = GradientHardnessRegistry(alpha=0.0, beta=0.9)
    reg.update(np.array([1, 2]), np.array([0.1, 5.0]))
    assert np.allclose(reg.weights_for(np.array([1, 2])), 1.0)  # mining off

    reg = GradientHardnessRegistry(alpha=1.0, beta=0.9, cap=2.0)
    reg.update(np.array([1, 2, 3]), np.array([0.001, 0.001, 10.0]))
    w = reg.weights_for(np.array([3]))                 # ratio ~3 > cap
    assert np.isclose(w[0], 2.0)                       # -> capped


def test_registry_save_load_roundtrip(tmp_path):
    reg = GradientHardnessRegistry(alpha=0.3, beta=0.9)
    reg.update(np.array([7, 8]), np.array([0.5, 1.5]))
    path = tmp_path / "hardness.json"
    reg.save(path)
    assert json.loads(path.read_text())["ema"]  # valid json with content
    fresh = GradientHardnessRegistry(alpha=0.3, beta=0.9)
    assert fresh.load(path)
    assert fresh._ema == reg._ema
    assert not fresh.load(tmp_path / "missing.json")


def test_registry_validates_inputs():
    with pytest.raises(ValueError):
        GradientHardnessRegistry(alpha=1.5)
    with pytest.raises(ValueError):
        GradientHardnessRegistry(cap=0.5)
    reg = GradientHardnessRegistry()
    with pytest.raises(ValueError):
        reg.update(np.array([1, 2]), np.array([0.1]))


# ── materialize_tract_sample ─────────────────────────────────────────────────

def test_tract_sample_shape_and_structure():
    table = _table((3, 3, 3, 3))
    out = table.materialize_tract_sample(9, seed=(825, 0))   # 9 // 3 yrs = 3 tracts
    assert list(out.columns) == FLAT_COLUMNS
    assert len(out) == 3 * len(YEARS)
    # one building per drawn tract, all its years adjacent (building-major)
    assert out["GEOID"].nunique() == 3
    assert out["building_id"].nunique() == 3
    for i in range(0, len(out), len(YEARS)):
        block = out.iloc[i:i + len(YEARS)]
        assert block["building_id"].nunique() == 1
        assert block["year"].tolist() == sorted(YEARS)


def test_tract_sample_deterministic_per_seed():
    table = _table()
    a = table.materialize_tract_sample(9, seed=(825, 3))
    b = table.materialize_tract_sample(9, seed=(825, 3))
    c = table.materialize_tract_sample(9, seed=(825, 4))
    pd.testing.assert_frame_equal(a, b)
    assert not a["building_id"].tolist() == c["building_id"].tolist() or \
           not a["GEOID"].tolist() == c["GEOID"].tolist()


def test_tract_sample_ignores_building_counts():
    """The core fix: a 50-building tract and a 2-building tract must be drawn
    equally often (legacy building-cyclic traversal gave the former 25x)."""
    table = _table((50, 2, 2, 2))
    counts = {g: 0 for g in table.buildings["GEOID"].astype(str).unique()}
    for s in range(300):
        out = table.materialize_tract_sample(2 * len(YEARS), seed=(1, s))
        for g in out["GEOID"].unique():
            counts[g] += 1
    freqs = np.array(list(counts.values()), dtype=float)
    # each tract drawn in ~half the 2-of-4 draws; big tract NOT over-drawn
    assert freqs.max() / freqs.min() < 1.4


def test_tract_sample_weighting_biases_draws():
    table = _table((3, 3, 3, 3))
    hot = str(table.buildings["GEOID"].astype(str).unique()[0])
    hot_hash = stable_geoid_hash(hot)

    def lookup(hashes):
        return np.where(np.asarray(hashes) == hot_hash, 10.0, 1.0)

    hits = 0
    for s in range(200):
        out = table.materialize_tract_sample(len(YEARS), seed=(2, s),
                                             weight_lookup=lookup)
        hits += int(hot in set(out["GEOID"]))
    # p(hot) = 10/13 ~ 0.77 vs 0.25 uniform
    assert hits > 120


def test_tract_sample_bad_weights_raise():
    table = _table()
    with pytest.raises(ValueError):
        table.materialize_tract_sample(9, weight_lookup=lambda h: np.ones(2))
    with pytest.raises(ValueError):
        table.materialize_tract_sample(0)


# ── lambda harvesting in the loss ────────────────────────────────────────────

def _loss_batch(B=6, seed=0):
    """Synthetic all-cross-sectional batch (same year, distinct tracts)."""
    rng = np.random.default_rng(seed)
    scores = torch.tensor(rng.normal(size=B), dtype=torch.float32,
                          requires_grad=True)
    labels = torch.tensor(rng.normal(size=B), dtype=torch.float32)
    geoids = torch.arange(100, 100 + B, dtype=torch.int64)
    years = torch.full((B,), 2018, dtype=torch.int64)
    bids = torch.arange(B, dtype=torch.int64)
    change = torch.zeros(B, dtype=torch.int64)
    return scores, labels, geoids, years, bids, change


def test_loss_lambda_matches_autograd():
    from src.main import InBatchPairwiseRankingLoss

    reg = GradientHardnessRegistry(alpha=0.3, beta=0.9)
    m = 1.0
    loss_fn = InBatchPairwiseRankingLoss(m_base=m, lambda_s=0.0, lambda_var=0.0,
                                         hardness_registry=reg)
    scores, labels, geoids, years, bids, change = _loss_batch()
    B = scores.shape[0]
    loss, diag = loss_fn(scores, labels, geoids, years, bids, change)

    # autograd reference: lambda_i = m * d(sum of pair losses)/ds_i
    # = m * N_pairs * d(L_cross)/ds_i  (loss uses the mean over pairs)
    n_pairs = diag["loss/cross_valid_pairs"]
    assert n_pairs == B * (B - 1) // 2                # no filters active
    (grad,) = torch.autograd.grad(loss, scores)
    expected_hardness = (m * n_pairs * grad).abs() / (B - 1)

    assert len(reg) == B
    got = np.array([reg._ema[int(g)] for g in geoids])
    assert np.allclose(got, expected_hardness.detach().numpy(), atol=1e-5)
    assert diag["mining/registry_tracts"] == B


def test_loss_without_registry_unchanged():
    from src.main import InBatchPairwiseRankingLoss

    loss_fn = InBatchPairwiseRankingLoss(m_base=1.0, lambda_s=0.0, lambda_var=0.0)
    scores, labels, geoids, years, bids, change = _loss_batch()
    loss, diag = loss_fn(scores, labels, geoids, years, bids, change)
    assert torch.isfinite(loss)
    assert "mining/registry_tracts" not in diag
