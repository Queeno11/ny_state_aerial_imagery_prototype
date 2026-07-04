"""Tests for InBatchPairwiseRankingLoss after the L_change removal (#30).

Synthetic hybrid batches on CPU: CS core first (anchor year), twins appended,
mirroring the layout produced by HybridBatchSampler.
"""

import torch
import pytest

from src.main import InBatchPairwiseRankingLoss


def _hybrid_batch(change_twin_score=0.5):
    """4 CS buildings (year 2020) + 1 stable twin (b0) + 1 change twin (b1)."""
    scores = torch.tensor(
        [0.0, 1.0, 2.0, 3.0, 0.2, change_twin_score], requires_grad=True
    )
    labels = torch.tensor([-1.5, -0.5, 0.5, 1.5, -1.5, -0.5])
    geoids = torch.tensor([1, 2, 3, 4, 1, 2])
    years = torch.tensor([2020, 2020, 2020, 2020, 2016, 2016])
    building_ids = torch.tensor([10, 11, 12, 13, 10, 11])
    structural_change = torch.tensor([0, 1, 0, 0, 0, 1])
    return scores, labels, geoids, years, building_ids, structural_change


def test_loss_has_no_change_component():
    loss_fn = InBatchPairwiseRankingLoss(m_base=1.0, m_min=0.1, lambda_s=1.0)
    loss, diag = loss_fn(*_hybrid_batch())
    assert torch.isfinite(loss)
    assert "loss/L_change" not in diag
    assert "grad_norm/change" not in diag
    assert "loss/change_hinge_active" not in diag


def test_twin_counts():
    loss_fn = InBatchPairwiseRankingLoss(lambda_s=1.0)
    _, diag = loss_fn(*_hybrid_batch())
    assert diag["loss/n_stable"] == 1
    assert diag["loss/n_change"] == 1


def test_change_twin_carries_no_gradient_or_loss():
    """Perturbing the change twin's score must leave the total loss unchanged."""
    loss_fn = InBatchPairwiseRankingLoss(lambda_s=1.0)
    loss_a, _ = loss_fn(*_hybrid_batch(change_twin_score=0.5))
    loss_b, _ = loss_fn(*_hybrid_batch(change_twin_score=5.0))
    assert loss_a.item() == pytest.approx(loss_b.item())

    scores, *rest = _hybrid_batch()
    loss, _ = loss_fn(scores, *rest)
    grad = torch.autograd.grad(loss, scores)[0]
    assert grad[5].item() == pytest.approx(0.0)   # change twin: no gradient
    assert grad[4].item() != pytest.approx(0.0)   # stable twin: L_stable gradient


def test_stable_penalty_responds_to_stable_twin():
    loss_fn = InBatchPairwiseRankingLoss(lambda_s=1.0)
    scores, labels, geoids, years, bids, chg = _hybrid_batch()
    _, diag = loss_fn(scores, labels, geoids, years, bids, chg)
    # stable twin score 0.2 vs CS partner 0.0 -> MSE = 0.04
    assert diag["loss/L_stable"] == pytest.approx(0.04, abs=1e-6)


def test_lambda_c_kwarg_rejected():
    with pytest.raises(TypeError):
        InBatchPairwiseRankingLoss(lambda_c=1.0)
