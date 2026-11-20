"""Exact within-cell pairwise surrogate and shared score normalization."""
from dataclasses import dataclass

import numpy as np
import torch

from ordinal_mc.core import Features, InadmissibleScoreError, Scorer

from data import fit_moments


@dataclass
class TensorPanel:
    inputs: torch.Tensor
    labels: torch.Tensor
    cells: torch.Tensor
    signs: torch.Tensor
    n_cells: int
    m: int
    views: int
    mean_f: float
    sd_f: float
    target_f: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    k: torch.Tensor

    @property
    def n(self):
        return 2 * self.m


def make_tensors(panel, aux, features, cfg, device, dtype):
    def arr(value):
        return torch.as_tensor(value, dtype=dtype, device=device)
    x = features(np.vstack((panel.x, aux)))
    w = arr(panel.w)
    c = torch.as_tensor(panel.w, dtype=torch.float64).reshape(panel.n_cells, -1)
    sorted_w, order = c.sort(dim=1)
    steps = torch.diff(sorted_w, dim=1, prepend=sorted_w[:, :1]) != 0
    ranks = torch.empty_like(order).scatter_(1, order, steps.cumsum(dim=1))
    ranks = ranks.to(device=device, dtype=torch.int16)
    signs = (ranks[:, None, :] - ranks[:, :, None]).sign().to(torch.int8)
    mean, var = fit_moments(panel.f, panel.n_cells)
    return TensorPanel(arr(x), w, torch.as_tensor(panel.cells, device=device), signs,
                       panel.n_cells, panel.m, cfg["views"], mean, np.sqrt(var),
                       arr(panel.f), arr(panel.a), arr(panel.b), arr(panel.k))


def fit_data(fit, val, aux_fit, aux_val, cfg, runtime, device, dtype):
    features = Features(fit.x, "nn", runtime)
    a = make_tensors(fit, aux_fit, features, cfg, device, dtype)
    b = make_tensors(val, aux_val, features, cfg, device, dtype)
    _, within_w = fit_moments(fit.w, fit.n_cells)
    denoms = dict(W=within_w, delta_W=float(np.var(fit.w[fit.m:] - fit.w[:fit.m])))
    assert min(denoms.values()) > 0
    return a, b, features, denoms


def new_scorer(seed, cfg, device, dtype):
    torch.manual_seed(seed)
    return Scorer(7, "nn", cfg["hidden"], cfg["weight_bound"]).to(device=device, dtype=dtype)


def normalize(raw, data, min_var):
    cells = raw[:data.n].reshape(data.n_cells, -1)
    mean = cells.mean()
    var = cells.var(dim=1, unbiased=True).mean()
    if not bool(torch.isfinite(var) & (var > min_var)):
        raise InadmissibleScoreError(f"Fitting score dispersion {var.detach().item()} outside domain")
    return (raw - mean) / var.sqrt(), (mean, var.sqrt())


def pair_stats(z, data, tau, actual=False):
    cells = z[:data.n].reshape(data.n_cells, -1)
    signed = data.signs * (cells[:, None, :] - cells[:, :, None])
    active = data.signs != 0
    count = data.n * (cells.shape[1] - 1)
    if actual:
        return ((signed < 0).sum() + 0.5 * ((signed == 0) & active).sum()) / count
    return (torch.sigmoid(-signed / tau) * active).sum() / count


def stability(z, data):
    return (z[data.n:].reshape(data.n, data.views) - z[:data.n, None]).square().mean()


def cardinal(predicted, data, denoms):
    residual = predicted - data.labels
    level = residual.square().mean() / denoms["W"]
    change = (residual[data.m:] - residual[:data.m]).square().mean() / denoms["delta_W"]
    return level, change


def known_prediction(z, panel, fit):
    u = fit.mean_f + fit.sd_f * z[:panel.n]
    return panel.a + panel.b * (u + panel.k * torch.tanh(u))


def loss_parts(z, data, tau, denoms, head=None, known_fit=None):
    rank = pair_stats(z, data, tau)
    stable = stability(z, data)
    if head is None and known_fit is None:
        return dict(rank=rank, stability=stable)
    pred = head(z[:data.n], data.cells) if head is not None else known_prediction(z, data, known_fit)
    level, change = cardinal(pred, data, denoms)
    penalty = head.penalty() if head is not None else z.new_zeros(())
    return dict(rank=rank, stability=stable, level=level, change=change, curvature=penalty)


@torch.no_grad()
def scores(model, fit, val, cfg):
    z, (mean, scale) = normalize(model(fit.inputs), fit, cfg["min_score_variance"])
    valid = (model(val.inputs) - mean) / scale
    return z, valid, (float(mean), float(scale))


def state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def selected_tau(update, total, cfg):
    return [tau for fraction, tau in zip(cfg["tau_fractions"], cfg["taus"])
            if update >= fraction * total][-1]
