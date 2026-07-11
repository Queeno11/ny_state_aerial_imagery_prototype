"""Tests for BatchImageDumper: first-30-batch image dumps for visual QA of ingested data."""

import json
import random

import pandas as pd
import pytest
import torch

from src.debug_batch_dump import BatchImageDumper, _norm_cbsa
from src.main import HybridBatchSampler, InBatchRankingDataset


def _fake_dataset(n_buildings=40, years=(2014, 2018), cbsas=(10001, 20002),
                  change_every=4):
    """InBatchRankingDataset stand-in without shard files (all shard tensors present)."""
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
    ds.images = torch.arange(n, dtype=torch.uint8).view(n, 1, 1, 1).expand(n, 4, 8, 8).contiguous()
    ds.scores = torch.arange(n, dtype=torch.float32)
    ds.geoids = torch.tensor(bids, dtype=torch.int64) * 7 + 3   # stand-in hashes
    ds.years = torch.tensor(yrs)
    ds.building_ids = torch.tensor(bids)
    ds.cbsa_ids = torch.tensor(cbs)
    ds.structural_change = torch.tensor(chg)
    ds.metas = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    ds.score_bins = torch.tensor(bids, dtype=torch.int64) % 5
    ds.year_to_idxs = {}
    ds.yearcbsa_to_idxs = {}
    ds.building_to_idxs = {}
    for i in range(n):
        ds.year_to_idxs.setdefault(yrs[i], []).append(i)
        ds.yearcbsa_to_idxs.setdefault((yrs[i], cbs[i]), []).append(i)
        ds.building_to_idxs.setdefault(bids[i], []).append(i)
    return ds


def test_dump_batch_payload_roundtrip(tmp_path):
    ds = _fake_dataset()
    dumper = BatchImageDumper(tmp_path / "dump", max_batches=5)
    batch = [0, 2, 4, 1]   # CS core [0, 2, 4] + one twin [1]
    dumper.dump_batch(ds, batch, cs_size=3, anchor_year=2014, anchor_cbsa=10001)

    payload = torch.load(tmp_path / "dump" / "batch_000.pt", weights_only=False)
    assert payload["cs_size"] == 3
    assert payload["anchor_year"] == 2014
    assert payload["anchor_cbsa"] == 10001
    assert payload["images"].dtype == torch.uint8
    assert payload["images"].shape == (4, 4, 8, 8)
    assert torch.equal(payload["images"], ds.images[torch.tensor(batch)])
    for key, src in [("scores", ds.scores), ("geoid_hashes", ds.geoids), ("years", ds.years),
                     ("building_ids", ds.building_ids), ("structural_change", ds.structural_change),
                     ("metas", ds.metas), ("score_bins", ds.score_bins), ("cbsa_ids", ds.cbsa_ids)]:
        assert torch.equal(payload[key], src[torch.tensor(batch)]), key


def test_dumper_stops_at_max_batches(tmp_path):
    ds = _fake_dataset()
    dumper = BatchImageDumper(tmp_path / "dump", max_batches=3)
    for _ in range(6):
        dumper.dump_batch(ds, [0, 1], cs_size=2, anchor_year=2014, anchor_cbsa=10001)
    assert dumper.finished
    assert sorted(p.name for p in (tmp_path / "dump").glob("batch_*.pt")) == [
        "batch_000.pt", "batch_001.pt", "batch_002.pt"
    ]


def test_dump_failure_disables_dumper(tmp_path):
    dumper = BatchImageDumper(tmp_path / "dump", max_batches=3)
    with pytest.warns(UserWarning, match="disabling batch dumping"):
        dumper.dump_batch(object(), [0], cs_size=1, anchor_year=2014, anchor_cbsa=0)
    assert dumper.finished
    assert not list((tmp_path / "dump").glob("batch_*.pt"))


def test_init_wipes_stale_dump(tmp_path):
    out = tmp_path / "dump"
    out.mkdir()
    (out / "batch_007.pt").write_bytes(b"stale")
    BatchImageDumper(out, max_batches=3)
    assert out.exists() and not list(out.iterdir())


def test_write_run_info(tmp_path):
    dumper = BatchImageDumper(tmp_path / "dump", max_batches=30)
    params = {"indicator": "median_income", "years": [2014, 2018], "image_size": 224,
              "nbands": 4, "max_jitter": 10, "tau_meters": 100.0, "sat_data": "NAIP"}
    dumper.write_run_info(params, "run_test")
    info = json.loads((tmp_path / "dump" / "dump_info.json").read_text())
    assert info["savename"] == "run_test"
    assert info["indicator"] == "median_income"
    assert info["years"] == [2014, 2018]
    assert info["max_batches"] == 30


def test_write_building_reference_joins_city_titles(tmp_path):
    dumper = BatchImageDumper(tmp_path / "dump", max_batches=30)
    df_train = pd.DataFrame({
        "building_id": [1, 1, 2, 3],           # duplicate rows across years collapse
        "GEOID": ["01001020100", "01001020100", "36061000100", "36061000200"],
        "cbsa_code": [33860, 33860, 35620.0, "35620"],  # mixed dtypes must normalize
        "Rel_Score": [0.1, 0.2, 0.3, 0.4],
    })
    cbsa_meta = pd.DataFrame({
        "cbsa_code": ["33860", "35620"],
        "cbsa_title": ["Montgomery, AL", "New York-Newark-Jersey City, NY-NJ-PA"],
    })
    dumper.write_building_reference(df_train, cbsa_meta=cbsa_meta)

    ref = pd.read_parquet(tmp_path / "dump" / "buildings_ref.parquet")
    assert len(ref) == 3
    assert ref["GEOID"].tolist()[0] == "01001020100"   # leading zero preserved
    by_bid = ref.set_index("building_id")
    assert by_bid.loc[1, "cbsa_title"] == "Montgomery, AL"
    assert by_bid.loc[2, "cbsa_title"].startswith("New York")
    assert by_bid.loc[3, "cbsa_title"].startswith("New York")


def test_norm_cbsa():
    assert _norm_cbsa(35620) == "35620"
    assert _norm_cbsa(35620.0) == "35620"
    assert _norm_cbsa("35620") == "35620"
    assert _norm_cbsa(None) == "None"


def test_sampler_integration_dumps_first_batches(tmp_path):
    """HybridBatchSampler with a dumper persists its first batches, CS core first."""
    random.seed(0)
    ds = _fake_dataset()
    dumper = BatchImageDumper(tmp_path / "dump", max_batches=3)
    sampler = HybridBatchSampler(ds, batch_size_cs=8, max_temporal_per_batch=4,
                                 batch_dumper=dumper)
    yielded = list(sampler)

    files = sorted((tmp_path / "dump").glob("batch_*.pt"))
    assert len(files) == min(3, len(yielded))
    for f, batch in zip(files, yielded):
        payload = torch.load(f, weights_only=False)
        assert torch.equal(payload["building_ids"],
                           ds.building_ids[torch.tensor(batch)])
        cs_years = payload["years"][: payload["cs_size"]]
        assert (cs_years == payload["anchor_year"]).all()
        cs_cbsas = payload["cbsa_ids"][: payload["cs_size"]]
        assert (cs_cbsas == payload["anchor_cbsa"]).all()

    # Second epoch: budget exhausted, nothing new is written
    list(sampler)
    assert len(list((tmp_path / "dump").glob("batch_*.pt"))) == min(3, len(yielded))
