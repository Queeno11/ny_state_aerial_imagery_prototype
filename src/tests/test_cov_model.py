"""Synthetic tests for the ACS-covariate model (scalemae_cov) and its wiring:
custom_models warm start, main.py helpers + datasets, prediction metadata,
and the src/run_cov_model.py launcher helpers."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

import src.custom_models as cm
import src.main as main
from src import prediction
from src import run_cov_model as rcm
from src.data import covariates as cv

EMBED = 8


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(EMBED, EMBED)
        self.fc1 = nn.Linear(EMBED, EMBED)
        self.fc2 = nn.Linear(EMBED, EMBED)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(self.qkv(x))))


def _tiny(meta_dim, targets=("qkv", "fc1", "fc2"), r=2, seed=0):
    torch.manual_seed(seed)
    m = nn.Module()
    m.meta_dim = meta_dim
    m.backbone = get_peft_model(_TinyBackbone(), LoraConfig(
        r=r, lora_alpha=r, target_modules=list(targets), lora_dropout=0.0, bias="none"))
    m.head = cm.LateFusionHead(EMBED, meta_dim, 1)
    return m


def _out(m, x, meta=None):
    return m.head(m.backbone(x).mean(dim=1), metadata=meta)


def _save_base(tmp_path, seed=1, **kw):
    base = _tiny(0, seed=seed, **kw)
    for n, p in base.backbone.named_parameters():
        if "lora_B" in n:                      # LoRA B starts at 0: make it non-trivial
            p.data.normal_()
    lora_dir = tmp_path / "base_best_lora"
    base.backbone.save_pretrained(lora_dir)
    head_path = tmp_path / "base_best.pth"
    torch.save(base.head.state_dict(), head_path)
    return base.eval(), head_path, lora_dir


# ─── custom_models ────────────────────────────────────────────────────────────

def test_expand_head_state_zero_covariate_columns():
    st = cm.LateFusionHead(EMBED, 0, 1).state_dict()
    wide = cm.expand_head_state(st, 2)
    assert wide["final_head.weight"].shape == (1, 66)
    assert torch.equal(wide["final_head.weight"][:, :64], st["final_head.weight"])
    assert torch.all(wide["final_head.weight"][:, 64:] == 0)
    assert cm.expand_head_state(wide, 2) is not wide and torch.equal(
        cm.expand_head_state(wide, 2)["final_head.weight"], wide["final_head.weight"])
    with pytest.raises(ValueError):
        cm.expand_head_state(wide, 3)


def test_warm_start_reproduces_base_and_keeps_lora_trainable(tmp_path):
    base, head_path, lora_dir = _save_base(tmp_path)
    # Same frozen base weights as the base run (in the real model both load the
    # same pretrained ScaleMAE; the adapter file carries only LoRA tensors).
    cov = _tiny(2, seed=1)
    lora_before = {n: id(p) for n, p in cov.backbone.named_parameters() if "lora_" in n}
    assert all(torch.all(p == 0) for n, p in cov.backbone.named_parameters() if "lora_B" in n)
    report = cm.warm_start_from_checkpoint(cov, head_path, lora_dir)
    cov.eval()
    assert report["lora_tensors"] == 6             # 3 target modules x (lora_A, lora_B)
    x = torch.randn(4, 5, EMBED)
    meta = torch.randn(4, 2)
    torch.testing.assert_close(_out(cov, x, meta), _out(base, x))   # zero covariate weights
    after = {n: (id(p), p.requires_grad) for n, p in cov.backbone.named_parameters() if "lora_" in n}
    assert {n: i for n, (i, _) in after.items()} == lora_before      # same objects: optimizer-safe
    assert all(rg for _, rg in after.values())                       # still trainable
    base_lora = dict(base.backbone.named_parameters())
    for n, p in cov.backbone.named_parameters():
        if "lora_" in n:
            torch.testing.assert_close(p, base_lora[n])                 # adapter really loaded


def test_warm_start_rejects_mismatched_adapter(tmp_path):
    _, head_path, lora_dir = _save_base(tmp_path, targets=("qkv",))
    with pytest.raises(RuntimeError, match="does not match"):
        cm.warm_start_from_checkpoint(_tiny(2), head_path, lora_dir)


def test_scalemae_cov_refuses_wrong_metadata_width():
    fake = SimpleNamespace(meta_dim=2)
    with pytest.raises(ValueError, match=r"\(B, 2\)"):
        cm.ScaleMAECov.forward(fake, torch.zeros(1, 3, 4, 4), metadata=torch.zeros(1, 1))
    with pytest.raises(ValueError):
        cm.ScaleMAECov.forward(fake, torch.zeros(1, 3, 4, 4), metadata=None)
    assert "scalemae_cov" in cm._MODEL_REGISTRY


# ─── main.py helpers ──────────────────────────────────────────────────────────

def test_meta_dim_and_covariate_validation():
    assert main.meta_dim_of({}) == 0
    assert main.meta_dim_of({"covariates": ("edu_ba_share", "age_25_54_share")}) == 2
    main.validate_covariate_config({"model_name": "scalemae"})
    main.validate_covariate_config({"model_name": "scalemae_cov", "covariates": ("edu_ba_share",)})
    for bad in ({"model_name": "scalemae_cov"},
                {"model_name": "scalemae", "covariates": ("edu_ba_share",)},
                {"model_name": "scalemae_cov", "covariates": ("income",)},
                {"model_name": "scalemae_cov", "covariates": ("edu_ba_share",), "train_scope": "x"}):
        with pytest.raises(ValueError):
            main.validate_covariate_config(bad)


def test_fill_params_defaults_accepts_covariate_keys():
    p = main.fill_params_defaults({"model_name": "scalemae_cov",
                                   "covariates": ("edu_ba_share", "age_25_54_share"),
                                   "init_from": "run_x", "reuse_cache_of": "run_x",
                                   "train_scope": "head"})
    assert p["covariates"] == ("edu_ba_share", "age_25_54_share") and p["train_scope"] == "head"
    assert main.fill_params_defaults({})["covariates"] is None


def test_resolve_cache_reuse_cases(tmp_path):
    r = main.resolve_cache_reuse
    assert r("run_a", "run_a") == (True, True)                         # own cache
    assert r("run_base", "run_cov", "run_base") == (True, False)        # reuse, keep owner
    assert r("run_other", "run_cov") == (False, True)                   # legacy: clear + claim
    assert r("run_other", "run_cov", resume_cache=True) == (True, True)
    assert r("run_base", "run_cov", "run_base", retrain=True) == (False, True)
    with pytest.raises(RuntimeError, match="belongs to 'run_other'"):
        r("run_other", "run_cov", "run_base")                           # never clear a reused cache


def test_apply_train_scope_freezes_backbone_only():
    m = _tiny(2)
    n_all = main.apply_train_scope(m, "all")
    n_head = main.apply_train_scope(m, "head")
    assert n_head == sum(p.numel() for p in m.head.parameters())
    assert n_head < n_all
    assert not any(p.requires_grad for p in m.backbone.parameters())


def test_warm_start_run_paths(tmp_path):
    ckpt = tmp_path / "models_by_epoch" / "run_x"
    ckpt.mkdir(parents=True)
    base, head_path, lora_dir = _save_base(tmp_path)
    head_path.rename(ckpt / "run_x_best.pth")
    lora_dir.rename(ckpt / "run_x_best_lora")
    cov = _tiny(2, seed=3)
    rep = main.warm_start_run(cov, "run_x", models_dir=tmp_path)
    assert rep["lora"] is not None
    with pytest.raises(FileNotFoundError):
        main.warm_start_run(cov, "run_missing", models_dir=tmp_path)


def _lookup():
    return cv.CovariateLookup(pd.DataFrame({
        "geoid_2023": ["36061000100", "36061000200"], "year": [2020, 2020],
        "edu_ba_share_z": [1.5, -0.5], "age_25_54_share_z": [0.2, 0.4]}))


def _shard(path, geoids=("36061000100", "36061000200"), years=(2020, 2020)):
    """A shard as CyclicCacheManager writes it: tract ids are stable_geoid_hash."""
    from src.data.tract_sampling import hash_geoids
    n = len(geoids)
    torch.save({"images": torch.zeros(n, 3, 4, 4, dtype=torch.uint8),
                "scores": torch.arange(n, dtype=torch.float32),
                "geoids": torch.tensor(hash_geoids(list(geoids)), dtype=torch.int64),
                "years": torch.tensor(years, dtype=torch.int64),
                "building_ids": torch.arange(n, dtype=torch.int64),
                "structural_change": torch.zeros(n, dtype=torch.int64),
                "metas": torch.full((n, 1), 9.0),
                "score_bins": torch.zeros(n, dtype=torch.int64),
                "cbsa_ids": torch.zeros(n, dtype=torch.int64)}, path)
    return path


def test_static_sharded_dataset_swaps_in_covariates(tmp_path):
    shard = _shard(tmp_path / "shard_0.pt")
    plain = main.StaticShardedDataset([shard], verbose=False)
    assert plain[0][2].tolist() == [9.0]                                 # legacy column untouched
    ds = main.StaticShardedDataset([shard], verbose=False, cov_lookup=_lookup())
    torch.testing.assert_close(ds[0][2], torch.tensor([1.5, 0.2]))
    torch.testing.assert_close(ds[1][2], torch.tensor([-0.5, 0.4]))
    bad = tmp_path / "shard_1.pt"
    torch.save({"images": torch.zeros(1, 3, 4, 4), "scores": torch.zeros(1)}, bad)
    with pytest.raises(RuntimeError, match="no geoids"):
        main.StaticShardedDataset([bad], verbose=False, cov_lookup=_lookup())[0]


def test_in_batch_dataset_swaps_in_covariates(tmp_path):
    shard = _shard(tmp_path / "shard_0.pt")
    mgr = SimpleNamespace(_is_initialized=True, active_shards=[shard])
    ds = main.InBatchRankingDataset(mgr, cov_lookup=_lookup())
    assert ds.metas.shape == (2, 2)
    torch.testing.assert_close(ds[1][6], torch.tensor([-0.5, 0.4]))
    assert main.InBatchRankingDataset(mgr).metas.shape == (2, 1)


# ─── prediction ───────────────────────────────────────────────────────────────

class _MetaModel(nn.Module):
    def forward(self, x, metadata=None):
        return x.mean(dim=(1, 2, 3)) * 0 + metadata.sum(dim=1)


def test_predict_chunk_rows_accepts_covariate_vectors():
    crops = [np.zeros((3, 4, 4), dtype=np.uint8) for _ in range(3)]
    tf = main.make_eval_transform(3)
    cov = np.array([[1.0, 2.0], [0.5, 0.5], [-1.0, 0.0]], dtype="float32")
    out = prediction.predict_chunk_rows(_MetaModel(), crops, cov, torch.device("cpu"), tf, 2)
    np.testing.assert_allclose(out, [3.0, 1.0, -1.0])
    legacy = prediction.predict_chunk_rows(_MetaModel(), crops, [1.0, 2.0, 3.0],
                                           torch.device("cpu"), tf, 2)
    np.testing.assert_allclose(legacy, [1.0, 2.0, 3.0])


def test_covariate_rows(monkeypatch):
    df = pd.DataFrame({"GEOID": ["36061000200", "36061000100", "x"]})
    assert prediction.covariate_rows(df, 2020, {}) is None
    monkeypatch.setattr(cv, "load_lookup", lambda names: _lookup())
    rows = prediction.covariate_rows(df, 2021, {"covariates": cv.COVARIATE_NAMES})
    np.testing.assert_allclose(rows, [[-0.5, 0.4], [1.5, 0.2], [0.0, 0.0]])


# ─── launcher helpers ─────────────────────────────────────────────────────────

def test_cov_params_overrides():
    p = rcm.cov_params({"model_name": "scalemae", "n_epochs": 750, "learning_rate": 1e-4,
                        "resume_lr_override": 1e-4, "indicator": "W2_r5"},
                       epochs=20, lr=1e-5, scope="head")
    assert p["model_name"] == "scalemae_cov" and p["covariates"] == rcm.COVARIATES
    assert p["init_from"] == p["reuse_cache_of"] == rcm.BASE_RUN_ID
    assert (p["n_epochs"], p["learning_rate"], p["train_scope"]) == (20, 1e-5, "head")
    assert p["run_id"] == "run_20260722_cov_head" and p["resume_lr_override"] is None
    assert p["indicator"] == "W2_r5"
    assert rcm.cov_savename() == "run_20260722_cov"
    assert rcm.input_params_from_log({"tau_meters": 112.0, "tau_meters_requested": 100,
                                      "a": 1}) == {"tau_meters": 100, "a": 1}


def test_subset_predictions(tmp_path):
    src = tmp_path / "2019_predictions.csv"
    pd.DataFrame({"building_id": [1, 2, 3], "GEOID": ["01", "02", "03"],
                  "predicted_value": [0.1, 0.2, 0.3]}).to_csv(src, index=False)
    n = rcm.subset_predictions(src, {1, 3}, tmp_path / "out" / "2019_predictions.csv")
    out = pd.read_csv(tmp_path / "out" / "2019_predictions.csv", dtype={"GEOID": str})
    assert n == 2 and out["building_id"].tolist() == [1, 3] and out["GEOID"].iloc[0] == "01"


def test_compare_table_and_edu_only():
    rng = np.random.default_rng(0)
    n = 60
    obs = rng.normal(size=n)
    pairs = pd.DataFrame({"GEOID": ["36061000100"] * 30 + ["36061000200"] * 30,
                          "dlog_label_usd": obs,
                          "in_stable": np.arange(n) < 30, "in_change_1pct": np.arange(n) >= 30,
                          "in_change_5pct": np.arange(n) >= 30, "in_change_10pct": np.arange(n) >= 50,
                          "split_group": np.where(np.arange(n) % 2, "heldout", "train")})
    good = pairs.assign(dlog_pred_usd=obs + rng.normal(0, 0.1, n))
    bad = pairs.assign(dlog_pred_usd=rng.normal(size=n))
    edu = pd.Series(obs, index=pairs.index)
    t = rcm.compare_table({"covariates": good, "imagery only": bad}, edu)
    piv = t.pivot_table(index="group", columns="model", values="rho")
    assert piv.loc["change_5pct", "covariates"] > 0.9 > abs(piv.loc["change_5pct", "imagery only"])
    assert piv.loc["change_5pct", "edu-only"] == pytest.approx(1.0)
    lk = cv.CovariateLookup(pd.DataFrame({
        "geoid_2023": ["36061000100"] * 2, "year": [2018, 2023],
        "edu_ba_share_z": [0.0, 1.0], "age_25_54_share_z": [0.0, 0.0]}))
    e = rcm.edu_only_change(pairs.iloc[:3], lk)
    np.testing.assert_allclose(e, [1.0, 1.0, 1.0])


def test_mask_credentials():
    assert rcm._mask_credentials("http://user:secret@localhost:3128") == "http://***@localhost:3128"
    assert rcm._mask_credentials("http://proxy:8080") == "http://proxy:8080"
    assert rcm._mask_credentials(None) is None
