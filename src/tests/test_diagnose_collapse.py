"""Synthetic-data tests for src/diagnose_collapse.py."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.diagnose_collapse import (
    build_eval_transform,
    head_weight_report,
    image_sensitivity,
    interpret,
    load_shard_tensors,
    lora_b_report,
    pick_most_different,
    predict,
    spearman_report,
)


# ── fixtures ──────────────────────────────────────────────────────────────

class MetaOnlyModel(nn.Module):
    """Simulates the collapsed model: score = 2 * meta, ignores the image."""
    def forward(self, x, metadata=None):
        return 2.0 * metadata[:, :1]


class ImageAwareModel(nn.Module):
    """Score depends on the image mean; metadata ignored."""
    def forward(self, x, metadata=None):
        return x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)


def make_shard(tmp_path, n=8, seed=0, name="shard_0.pt"):
    g = torch.Generator().manual_seed(seed)
    data = {
        "images": torch.randint(0, 255, (n, 4, 32, 32), generator=g, dtype=torch.uint8),
        "scores": torch.randn(n, generator=g),
        "metas": torch.randn(n, 1, generator=g),
    }
    p = tmp_path / name
    torch.save(data, p)
    return p, data


# ── data loading ──────────────────────────────────────────────────────────

def test_load_shard_tensors_concatenates(tmp_path):
    p1, _ = make_shard(tmp_path, n=8, name="a.pt")
    p2, _ = make_shard(tmp_path, n=5, seed=1, name="b.pt")
    out = load_shard_tensors([p1, p2])
    assert out["images"].shape == (13, 4, 32, 32)
    assert out["scores"].shape == (13,)
    assert out["metas"].shape == (13, 1)


def test_load_shard_tensors_skips_empty_and_raises_on_all_empty(tmp_path):
    p_empty = tmp_path / "empty.pt"
    torch.save({"images": torch.empty(0), "scores": torch.empty(0)}, p_empty)
    p_full, _ = make_shard(tmp_path, n=3)
    assert len(load_shard_tensors([p_empty, p_full])["images"]) == 3
    with pytest.raises(ValueError):
        load_shard_tensors([p_empty])


def test_eval_transform_scales_uint8_to_normalized_float():
    t = build_eval_transform(4)
    x = torch.randint(0, 255, (2, 4, 16, 16), dtype=torch.uint8)
    y = t(x)
    assert y.dtype == torch.float32
    assert y.abs().max() < 10.0  # normalized range, not raw 0-255


# ── check 1: spearman ─────────────────────────────────────────────────────

def test_spearman_report_detects_collapse():
    meta = np.arange(50, dtype=float)
    labels = np.random.default_rng(0).normal(size=50)
    preds = 3.0 * meta + 1.0  # monotone in meta
    rep = spearman_report(labels, meta.reshape(-1, 1), preds)
    assert rep["spearman_pred_vs_meta"] == pytest.approx(1.0)
    assert abs(rep["spearman_meta_vs_label"]) < 0.5


def test_spearman_report_without_preds():
    rep = spearman_report(np.arange(10.0), np.arange(10.0).reshape(-1, 1))
    assert rep == {"spearman_meta_vs_label": pytest.approx(1.0)}
    assert "spearman_pred_vs_meta" not in rep


# ── check 2: sensitivity ──────────────────────────────────────────────────

def test_pick_most_different_finds_extremes():
    imgs = torch.zeros(4, 1, 8, 8, dtype=torch.uint8)
    imgs[2] = 255
    i, j = pick_most_different(imgs)
    assert {i, j} == {2, 0} or {i, j} == {2, 1} or {i, j} == {2, 3}
    assert 2 in (i, j)


def test_image_sensitivity_flags_dead_path():
    imgs = torch.randint(0, 255, (8, 4, 32, 32), dtype=torch.uint8)
    metas = torch.randn(8, 1)
    t = build_eval_transform(4)
    dead = image_sensitivity(MetaOnlyModel(), imgs, metas, t, "cpu")
    assert dead["cross_image_diff"] == pytest.approx(0.0, abs=1e-6)
    alive = image_sensitivity(ImageAwareModel(), imgs, metas, t, "cpu")
    assert alive["cross_image_diff"] > 10 * max(alive["repeat_diff"], 1e-6)


def test_predict_batches_match_full_pass():
    imgs = torch.randint(0, 255, (10, 4, 32, 32), dtype=torch.uint8)
    metas = torch.randn(10, 1)
    t = build_eval_transform(4)
    p = predict(MetaOnlyModel(), imgs, metas, t, "cpu", batch_size=3)
    assert p.shape == (10,)
    assert torch.allclose(p, 2.0 * metas.view(-1))


# ── check 3: weight autopsy ───────────────────────────────────────────────

def _head_state(img_scale, meta_scale, meta_dim=1):
    w = torch.cat([img_scale * torch.ones(64), meta_scale * torch.ones(meta_dim)])
    return {
        "final_head.weight": w.view(1, -1),
        "image_compressor.0.weight": torch.ones(64, 1024) * img_scale,
    }


def test_head_weight_report_dominant_meta():
    rep = head_weight_report(_head_state(img_scale=1e-4, meta_scale=1.0))
    assert rep["meta_weight_abs_mean"] > 10 * rep["image_weight_abs_mean"]


def test_head_weight_report_meta_dim_zero():
    state = {
        "final_head.weight": torch.ones(1, 64),
        "image_compressor.0.weight": torch.ones(64, 1024),
    }
    rep = head_weight_report(state, meta_dim=0)
    assert rep["meta_weight_abs_mean"] == 0.0
    assert rep["image_weight_abs_mean"] == pytest.approx(1.0)


def test_lora_b_report_zero_init(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    tensors = {
        "base.layer0.lora_A.weight": torch.randn(16, 1024),
        "base.layer0.lora_B.weight": torch.zeros(1024, 16),
        "base.layer1.lora_A.weight": torch.randn(16, 1024),
        "base.layer1.lora_B.weight": torch.zeros(1024, 16),
    }
    safetensors.save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    rep = lora_b_report(tmp_path)
    assert rep["n_lora_modules"] == 2
    assert rep["lora_B_norm_max"] == 0.0
    assert rep["lora_A_norm_mean"] > 0


# ── verdicts ──────────────────────────────────────────────────────────────

def test_interpret_collapsed_case():
    verdicts = interpret(
        sp={"spearman_pred_vs_meta": 0.999, "spearman_meta_vs_label": -0.01},
        sens={"cross_image_diff": 0.0, "repeat_diff": 0.0},
        head={"meta_weight_abs_mean": 1.0, "image_weight_abs_mean": 1e-5},
        lora={"lora_B_norm_max": 0.0},
    )
    text = " ".join(verdicts)
    for token in ("COLLAPSED", "DEAD IMAGE PATH", "HEAD AUTOPSY", "LoRA NEVER TRAINED"):
        assert token in text


def test_interpret_healthy_case():
    verdicts = interpret(
        sp={"spearman_pred_vs_meta": 0.2, "spearman_meta_vs_label": 0.1},
        sens={"cross_image_diff": 1.5, "repeat_diff": 1e-6},
        head={"meta_weight_abs_mean": 0.1, "image_weight_abs_mean": 0.1},
        lora={"lora_B_norm_max": 3.2},
    )
    text = " ".join(verdicts)
    assert "COLLAPSED" not in text
    assert "DEAD IMAGE PATH" not in text
    assert "moved off zero-init" in text
