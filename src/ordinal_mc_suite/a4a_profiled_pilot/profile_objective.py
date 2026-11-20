"""Conditional head solves for the exact weighted scorer objective."""
import numpy as np
import torch

from prior import ORIGINAL
from fitting import cardinal, loss_parts, normalize
from heads import MonotoneHead, curvature, fit_heads


def solve(z, panel, spec, denoms, cfg, weights, start=None):
    """Fit the head to the branch objective on detached fitting scores."""
    w, d, gamma = weights
    if w <= 0:
        raise ValueError("A positive level coefficient anchors the two cell intercepts")
    z_numpy = z[:len(panel.x)].detach().cpu().double().numpy()
    return fit_heads(z_numpy, panel, spec, denoms, cfg["head_max_iter"],
                     level=1., change=d / w, gamma=gamma / w, start=start)


def fitted_module(spec, coefficients, data):
    head = MonotoneHead(spec, coefficients, data.inputs.dtype, data.inputs.device)
    return head.requires_grad_(False)


def objective(z, fit, head, denoms, cfg, weights):
    parts = loss_parts(z, fit, cfg["taus"][-1], denoms, head)
    rank = parts["rank"] + cfg["lambda_stability"] * parts["stability"]
    w, d, gamma = weights
    return rank + w * parts["level"] + d * parts["change"] + gamma * parts["curvature"], parts


def cardinal_objective(z, panel, spec, denoms, weights, coefficients):
    pred = spec.direct(coefficients, z[:len(panel.x)].detach().cpu().double().numpy(), panel.cells)
    r = pred - panel.w
    w, d, gamma = weights
    return float(w * np.mean(r**2) / denoms["W"]
                 + d * np.mean((r[panel.m:] - r[:panel.m])**2) / denoms["delta_W"]
                 + gamma * curvature(coefficients, spec))
