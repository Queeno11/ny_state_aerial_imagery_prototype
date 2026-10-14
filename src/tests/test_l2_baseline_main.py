"""Unit tests for the L2 baseline's pieces in src/main.py: PointwiseMSELoss,
repeat_occurrence_mask, validate_loss_config, the mixed-pool sampler,
substitution_label_columns and the per-run cache root. Synthetic only.
"""
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import src.main as m
from src.data import level_labels as ll


# ─── repeat_occurrence_mask / PointwiseMSELoss ───────────────────────────────

def test_repeat_occurrence_mask_flags_only_later_repeats():
    ids = torch.tensor([5, 7, 9, 7, 5, 11])
    assert m.repeat_occurrence_mask(ids).tolist() == [False, False, False, True, True, False]


def _batch():
    scores = torch.tensor([[0.0], [1.0], [2.0], [1.5], [0.5]], requires_grad=True)
    labels = torch.tensor([0.5, 1.0, 1.0, 2.0, 0.0])
    geoids = torch.tensor([1, 2, 3, 2, 1])
    years = torch.tensor([2016, 2017, 2016, 2019, 2020])
    bids = torch.tensor([10, 20, 30, 20, 10])        # last two are twins
    change = torch.tensor([0, 1, 0, 1, 0])
    return scores, labels, geoids, years, bids, change


def test_mse_loss_value_gradient_and_diagnostics():
    scores, labels, geoids, years, bids, change = _batch()
    loss, diag = m.PointwiseMSELoss()(scores, labels, geoids, years, bids, change,
                                      current_epoch=0, current_step=0)
    sq = (scores.detach().squeeze(-1) - labels) ** 2
    assert loss.item() == pytest.approx(sq.mean().item())
    loss.backward()
    expected_grad = 2 * (scores.detach().squeeze(-1) - labels) / len(labels)
    assert torch.allclose(scores.grad.squeeze(-1), expected_grad)

    assert diag["loss/mse_cs"] == pytest.approx(sq[:3].mean().item())
    assert diag["loss/mse_temporal"] == pytest.approx(sq[3:].mean().item())
    assert diag["loss/n_temporal"] == 2
    assert diag["loss/n_change"] == 1 and diag["loss/n_stable"] == 1
    assert diag["loss/n_years"] == 2      # CS core spans 2016 and 2017
    assert all(np.isfinite(v) for v in diag.values())


def test_mse_loss_without_twins_reports_zero_temporal():
    scores = torch.tensor([[0.0], [1.0]])
    _, diag = m.PointwiseMSELoss()(scores, torch.tensor([0.0, 0.0]), torch.tensor([1, 2]),
                                   torch.tensor([2016, 2016]), torch.tensor([1, 2]),
                                   torch.tensor([0, 0]))
    assert diag["loss/n_temporal"] == 0 and diag["loss/mse_temporal"] == 0.0


def test_mse_loss_handles_half_precision_scores():
    scores = torch.tensor([[0.25], [0.75]], dtype=torch.float16)
    loss, _ = m.PointwiseMSELoss()(scores, torch.tensor([0.0, 1.0]), torch.tensor([1, 2]),
                                   torch.tensor([2016, 2016]), torch.tensor([1, 2]),
                                   torch.tensor([0, 0]))
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(0.0625)


def test_set_model_and_loss_function_selects_mse(monkeypatch):
    monkeypatch.setattr(m.custom_models, "get_model", lambda **kw: torch.nn.Linear(1, 1))
    _, loss_fn = m.set_model_and_loss_function("x", "reg", 8, loss_type="mse")
    assert isinstance(loss_fn, m.PointwiseMSELoss) and loss_fn.hardness_registry is None
    _, loss_fn = m.set_model_and_loss_function("x", "reg", 8)
    assert isinstance(loss_fn, m.InBatchPairwiseRankingLoss)
    with pytest.raises(ValueError):
        m.set_model_and_loss_function("x", "reg", 8, loss_type="huber")


# ─── validate_loss_config ────────────────────────────────────────────────────

@pytest.mark.parametrize("params", [
    {"loss_type": "ranking", "mix_pools": True},
    {"loss_type": "ranking", "label_target": "log_level"},
    {"loss_type": "mse", "tract_sampling": True, "grad_mining_alpha": 0.3,
     "grad_mining_alpha_final": 0.0},
    {"loss_type": "mse", "tract_sampling": True, "grad_mining_alpha": 0.0,
     "grad_mining_alpha_final": 0.6},
    {"loss_type": "huber"},
    {"label_target": "levels"},
])
def test_validate_loss_config_rejects(params):
    with pytest.raises(ValueError):
        m.validate_loss_config(params)


@pytest.mark.parametrize("params", [
    {},  # ordinal defaults
    {"loss_type": "ranking", "tract_sampling": True, "grad_mining_alpha": 0.3,
     "grad_mining_alpha_final": 0.6},
    {"loss_type": "mse", "label_target": "log_level", "mix_pools": True,
     "tract_sampling": True, "grad_mining_alpha": 0.0, "grad_mining_alpha_final": 0.0},
    {"loss_type": "mse", "mix_pools": True, "tract_sampling": False,
     "grad_mining_alpha_final": 0.6},   # mining irrelevant without tract sampling
])
def test_validate_loss_config_accepts(params):
    m.validate_loss_config(params)


def test_fill_params_defaults_runs_the_guard():
    with pytest.raises(ValueError):
        m.fill_params_defaults({"loss_type": "ranking", "mix_pools": True})


# ─── HybridBatchSampler(mix_pools) ───────────────────────────────────────────

class _FakeDataset:
    """Minimal InBatchRankingDataset surface the sampler reads."""

    def __init__(self, n_cities=4, n_years=3, per_pool=12, seed=0):
        rng = np.random.default_rng(seed)
        years, cbsas, bids, change = [], [], [], []
        bid = 0
        for c in range(n_cities):
            for _ in range(per_pool):
                bid += 1
                for y in range(n_years):     # every building in every year -> twins exist
                    years.append(2016 + y)
                    cbsas.append(100 + c)
                    bids.append(bid)
                    change.append(int(rng.random() < 0.3))
        self.years = torch.tensor(years)
        self.cbsa_ids = torch.tensor(cbsas)
        self.building_ids = torch.tensor(bids)
        self.structural_change = torch.tensor(change)
        self.yearcbsa_to_idxs, self.year_to_idxs, self.building_to_idxs = {}, {}, {}
        for i, (y, c, b) in enumerate(zip(years, cbsas, bids)):
            self.yearcbsa_to_idxs.setdefault((y, c), []).append(i)
            self.year_to_idxs.setdefault(y, []).append(i)
            self.building_to_idxs.setdefault(b, []).append(i)

    def __len__(self):
        return len(self.years)


def _cores(sampler, cs):
    return [batch[:cs] for batch in sampler]


def test_single_pool_sampler_core_is_one_city_year():
    random.seed(1)
    ds = _FakeDataset()
    s = m.HybridBatchSampler(ds, batch_size_cs=8, max_temporal_per_batch=3)
    for core in _cores(s, 8):
        assert len({(ds.years[i].item(), ds.cbsa_ids[i].item()) for i in core}) == 1


def test_mixed_sampler_mixes_cities_and_years():
    random.seed(2)
    ds = _FakeDataset()
    s = m.HybridBatchSampler(ds, batch_size_cs=8, max_temporal_per_batch=3, mix_pools=True)
    cores = _cores(s, 8)
    n_cities = [len({ds.cbsa_ids[i].item() for i in c}) for c in cores]
    n_years = [len({ds.years[i].item() for i in c}) for c in cores]
    assert np.mean(n_cities) > 2.5 and np.mean(n_years) > 2.0


def test_mixed_sampler_twins_differ_from_their_own_anchor_year():
    random.seed(3)
    ds = _FakeDataset()
    s = m.HybridBatchSampler(ds, batch_size_cs=8, max_temporal_per_batch=3, mix_pools=True)
    for batch in s:
        core, twins = batch[:8], batch[8:]
        assert len(twins) == 3
        assert len(set(batch)) == len(batch)                      # no duplicated image
        assert len({ds.building_ids[i].item() for i in core}) == 8  # one image per building
        anchor_year = {ds.building_ids[i].item(): ds.years[i].item() for i in core}
        for t in twins:
            b = ds.building_ids[t].item()
            assert b in anchor_year and ds.years[t].item() != anchor_year[b]


def test_mixed_and_single_samplers_share_the_per_image_marginal():
    """Every pool holds >= batch_size images, so both samplers draw each image
    with probability B/N per batch; check frequencies over many epochs."""
    ds = _FakeDataset(n_cities=3, n_years=2, per_pool=10)
    n = len(ds)
    counts = {False: Counter(), True: Counter()}
    for mix in (False, True):
        random.seed(4)
        s = m.HybridBatchSampler(ds, batch_size_cs=8, max_temporal_per_batch=1, mix_pools=mix)
        for _ in range(400):
            for core in _cores(s, 8):
                counts[mix].update(core)
    for mix in (False, True):
        freq = np.array([counts[mix][i] for i in range(n)], dtype=float)
        freq /= freq.sum()
        # uniform within sampling noise
        assert np.abs(freq - 1 / n).max() < 0.25 / n


# ─── substitution_label_columns / cache root ─────────────────────────────────

def _panel():
    return pd.DataFrame({
        "Rel_Score_W2_i_r5pct_2016": [0.1, -0.1],
        "Rel_Score_W2_i_r5pct_2017": [0.2, -0.2],
        "W2_i_r5pct_2016": [100_000.0, 0.0],
        "W2_i_r5pct_2017": [120_000.0, 50_000.0],
    })


def test_substitution_labels_rel_score_reads_zscore_columns():
    cols = m.substitution_label_columns(_panel(), "W2_r5", [2015, 2016, 2017])
    assert sorted(cols) == [2016, 2017]
    assert cols[2017].tolist() == [0.2, -0.2]


def test_substitution_labels_log_level_uses_the_train_scale():
    stats = {"mu": 11.0, "sd": 0.5}
    cols = m.substitution_label_columns(_panel(), "W2_r5", [2016, 2017], "log_level", stats)
    want = (np.log(ll.deflate(120_000.0, 2017)) - 11.0) / 0.5
    assert cols[2017][0] == pytest.approx(want, rel=1e-5)
    assert np.isnan(cols[2016][1])          # non-positive dollars -> NaN, never z-score
    with pytest.raises(ValueError):
        m.substitution_label_columns(_panel(), "W2_r5", [2016], "log_level", None)


def test_run_cache_root_isolates_subdir():
    assert m.run_cache_root({}) == Path(m.CACHE_DIR)
    assert m.run_cache_root({"cache_subdir": "l2_baseline"}) == Path(m.CACHE_DIR) / "l2_baseline"
