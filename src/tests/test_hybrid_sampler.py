"""Tests for the US-scale HybridBatchSampler: same-(year,CBSA) cores + stable/change twins."""

import random

import pytest
import torch

from src.main import HybridBatchSampler, InBatchRankingDataset


def _fake_dataset(n_buildings=40, years=(2014, 2018), cbsas=(10001, 20002),
                  change_every=4):
    """InBatchRankingDataset stand-in without shard files.

    Each building appears once per year, buildings split evenly across CBSAs;
    every ``change_every``-th building is flagged structural change.
    """
    ds = InBatchRankingDataset.__new__(InBatchRankingDataset)
    yrs, bids, cbs, chg = [], [], [], []
    for b in range(n_buildings):
        cbsa = cbsas[b % len(cbsas)]
        flag = 1 if b % change_every == 0 else 0
        for yr in years:
            yrs.append(yr)
            bids.append(b)
            cbs.append(cbsa)
            chg.append(flag)
    n = len(yrs)
    ds.images = torch.zeros(n, 1)
    ds.scores = torch.zeros(n)
    ds.years = torch.tensor(yrs)
    ds.building_ids = torch.tensor(bids)
    ds.cbsa_ids = torch.tensor(cbs)
    ds.structural_change = torch.tensor(chg)
    ds.year_to_idxs = {}
    ds.yearcbsa_to_idxs = {}
    ds.building_to_idxs = {}
    for i in range(n):
        ds.year_to_idxs.setdefault(yrs[i], []).append(i)
        ds.yearcbsa_to_idxs.setdefault((yrs[i], cbs[i]), []).append(i)
        ds.building_to_idxs.setdefault(bids[i], []).append(i)
    return ds


def test_cs_core_single_year_and_cbsa():
    random.seed(0)
    ds = _fake_dataset()
    sampler = HybridBatchSampler(ds, batch_size_cs=8, max_temporal_per_batch=4)
    for batch in sampler:
        cs = batch[:8]
        assert len(set(ds.years[i].item() for i in cs)) == 1
        assert len(set(ds.cbsa_ids[i].item() for i in cs)) == 1


def test_twins_are_same_building_other_year():
    random.seed(1)
    ds = _fake_dataset()
    sampler = HybridBatchSampler(ds, batch_size_cs=8, max_temporal_per_batch=4)
    for batch in sampler:
        cs, twins = batch[:8], batch[8:]
        cs_year = ds.years[cs[0]].item()
        cs_bids = {ds.building_ids[i].item() for i in cs}
        for t in twins:
            assert ds.years[t].item() != cs_year
            assert ds.building_ids[t].item() in cs_bids


def test_change_twins_are_sampled():
    """The [PATCH] stable-only filter is gone: change buildings get twins too."""
    random.seed(2)
    ds = _fake_dataset(change_every=2)   # half the buildings are change-flagged
    sampler = HybridBatchSampler(ds, batch_size_cs=8, max_temporal_per_batch=6)
    change_twins = 0
    stable_twins = 0
    for _ in range(5):
        for batch in sampler:
            for t in batch[8:]:
                if ds.structural_change[t].item() == 1:
                    change_twins += 1
                else:
                    stable_twins += 1
    assert change_twins > 0, "change twins never sampled — [PATCH] filter still active?"
    assert stable_twins > 0


def test_budget_respected():
    random.seed(3)
    ds = _fake_dataset()
    max_temporal = 4
    sampler = HybridBatchSampler(ds, batch_size_cs=8, max_temporal_per_batch=max_temporal)
    for batch in sampler:
        assert len(batch) <= 8 + max_temporal


def test_degenerate_single_building_pool():
    random.seed(4)
    ds = _fake_dataset(n_buildings=2, years=(2014,), cbsas=(10001,))
    sampler = HybridBatchSampler(ds, batch_size_cs=8, max_temporal_per_batch=4)
    batches = list(sampler)
    assert batches, "sampler must still yield with tiny pools"
    for batch in batches:
        assert len(batch) >= 1


def test_batch_is_single_cbsa_including_twins():
    """Every batch — CS core AND temporal twins — must come from one CBSA (#29)."""
    for seed in range(6):
        random.seed(seed)
        ds = _fake_dataset(change_every=3)
        sampler = HybridBatchSampler(ds, batch_size_cs=8, max_temporal_per_batch=6)
        for batch in sampler:
            assert len({ds.cbsa_ids[i].item() for i in batch}) == 1


def test_fallback_warns_on_undersized_cbsa_pools(capsys):
    """Per-year fallback with real (nonzero) cbsa_ids must warn: batches can mix cities."""
    random.seed(6)
    # 2 CBSAs x 1 building each -> every (year, CBSA) pool is below min_pool for cs=8
    ds = _fake_dataset(n_buildings=2, years=(2014,), cbsas=(10001, 20002))
    sampler = HybridBatchSampler(ds, batch_size_cs=8, max_temporal_per_batch=4)
    list(sampler)
    assert "may MIX CBSAs" in capsys.readouterr().out
    list(sampler)   # one-shot: no second warning
    assert "may MIX CBSAs" not in capsys.readouterr().out


def test_zero_cbsa_backward_compat():
    """Old shards (cbsa_ids all zero) behave like the previous per-year sampler."""
    random.seed(5)
    ds = _fake_dataset(cbsas=(0,))
    sampler = HybridBatchSampler(ds, batch_size_cs=8, max_temporal_per_batch=4)
    for batch in sampler:
        cs = batch[:8]
        assert len(set(ds.years[i].item() for i in cs)) == 1
