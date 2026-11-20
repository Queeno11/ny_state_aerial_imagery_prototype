"""Independent cell intercepts and monotone PWL slopes in standardized W units."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
import torch
from torch import nn
from torch.nn import functional as F


def slope_spacing(knots):
    left, right = np.diff(knots)
    return np.array([left, (left + right) / 2, right])


def numpy_basis(z, knots):
    hinges = np.maximum(z[:, None] - knots, 0.)
    values = np.column_stack((z, hinges))
    return np.column_stack((np.ones(len(z)), values[:, :-1] - values[:, 1:], values[:, -1]))


@dataclass
class HeadSpec:
    knots: np.ndarray
    y_center: float
    y_scale: float
    min_slope: float

    def design(self, z):
        return numpy_basis(np.asarray(z, dtype=np.float64), self.knots)

    def direct(self, coefficients, z, cell):
        return self.y_center + self.y_scale * np.sum(self.design(z) * coefficients[cell], axis=1)


class MonotoneHead(nn.Module):
    def __init__(self, spec, coefficients, dtype, device):
        super().__init__()
        for name, values in dict(knots=spec.knots, spacing=slope_spacing(spec.knots),
                                 y_center=spec.y_center, y_scale=spec.y_scale).items():
            self.register_buffer(name, torch.as_tensor(values, dtype=dtype, device=device))
        self.min_slope = spec.min_slope
        intercept = torch.as_tensor(coefficients[:, 0], dtype=dtype, device=device)
        slopes = torch.as_tensor(coefficients[:, 1:], dtype=dtype, device=device)
        self.intercept = nn.Parameter(intercept.clone())
        self.cell_slopes = nn.Parameter(slopes.clone())

    def slopes(self):
        return self.cell_slopes

    @torch.no_grad()
    def project(self):
        self.cell_slopes.clamp_(min=self.min_slope)

    def forward(self, z, cell):
        if z.ndim != 1 or cell.shape != z.shape:
            raise ValueError("Cell heads require a scalar score and one cell ID")
        slopes = self.slopes()[cell]
        hinges = F.relu(z[:, None] - self.knots)
        y = self.intercept[cell] + slopes[:, 0] * z + (hinges * torch.diff(slopes, dim=1)).sum(1)
        return self.y_center + self.y_scale * y

    def penalty(self):
        return (torch.diff(self.slopes(), dim=1).square() / self.spacing).sum(1).mean()

    def coefficients(self):
        return torch.column_stack((self.intercept, self.slopes())).detach().cpu().double().numpy()


def initial_coefficients(z, w, cell, spec):
    y = (w - spec.y_center) / spec.y_scale
    results = []
    for i in range(int(cell.max()) + 1):
        x_i, y_i = z[cell == i], y[cell == i]
        slope = np.mean((x_i - x_i.mean()) * (y_i - y_i.mean())) / x_i.var()
        slope = max(slope, 0.01)
        results.append([y_i.mean() - slope * x_i.mean(), *([slope] * 4)])
    return np.array(results)


def curvature(coefficients, spec):
    return float(np.mean(np.sum(np.diff(coefficients[:, 1:], axis=1)**2
                                / slope_spacing(spec.knots), axis=1)))


def fit_heads(z, panel, spec, denoms, max_iter, level=1., change=1., gamma=0., start=None):
    """Constrained convex solve on paired city blocks, with exact quadratic gradient."""
    n_city = panel.n_cells // 2
    count = panel.m // n_city
    basis = spec.design(z).reshape(2, n_city, count, 5)
    target = ((panel.w - spec.y_center) / spec.y_scale).reshape(2, n_city, count)
    if start is None:
        start = initial_coefficients(z, panel.w, panel.cells, spec)
    lb = np.tile([-np.inf, *([spec.min_slope] * 4)], 2 * n_city)
    x0 = start.reshape(n_city, 10)
    hessian = np.zeros((n_city, 10, 10))
    linear = np.zeros((n_city, 10))
    if level:
        scale = level * spec.y_scale**2 / (2 * count * denoms["W"])
        for date in (0, 1):
            block = slice(date * 5, (date + 1) * 5)
            x = basis[date]
            hessian[:, block, block] += scale * (x.transpose(0, 2, 1) @ x)
            linear[:, block] += scale * (x.transpose(0, 2, 1) @ target[date, :, :, None]).squeeze(2)
    if change:
        scale = change * spec.y_scale**2 / (count * denoms["delta_W"])
        x = np.concatenate((-basis[0], basis[1]), axis=2)
        dy = target[1] - target[0]
        hessian += scale * (x.transpose(0, 2, 1) @ x)
        linear += scale * (x.transpose(0, 2, 1) @ dy[:, :, None]).squeeze(2)
    if gamma:
        diff = np.diff(np.eye(4), axis=0) / np.sqrt(slope_spacing(spec.knots))[:, None]
        for block in (slice(1, 5), slice(6, 10)):
            hessian[:, block, block] += gamma / 2 * diff.T @ diff

    def objective(flat):
        params = flat.reshape(n_city, 10)
        product = (hessian @ params[:, :, None]).squeeze(2)
        value = np.sum(params * (product - 2 * linear))
        return value, (2 * (product - linear)).ravel()

    result = minimize(objective, x0.ravel(), jac=True, method="L-BFGS-B",
                      bounds=list(zip(lb, np.full_like(lb, np.inf))),
                      options=dict(maxiter=max_iter, gtol=1e-7, ftol=1e-15, maxcor=20))
    gradient = objective(result.x)[1]
    active = result.x <= lb + 1e-9
    gradient[active] = np.minimum(gradient[active], 0)
    projected = float(np.max(np.abs(gradient)))
    coefficients = result.x.reshape(n_city * 2, 5)
    prediction = spec.direct(coefficients, z, panel.cells)
    residual = prediction - panel.w
    true_objective = (level * np.mean(residual**2) / denoms["W"]
                      + change * np.mean((residual[panel.m:] - residual[:panel.m])**2) / denoms["delta_W"]
                      + gamma * curvature(coefficients, spec))
    initial_residual = spec.direct(start, z, panel.cells) - panel.w
    initial_objective = (level * np.mean(initial_residual**2) / denoms["W"]
                         + change * np.mean(np.diff(initial_residual.reshape(2, -1), axis=0)**2) / denoms["delta_W"]
                         + gamma * curvature(start, spec))
    offset_check = (result.fun - objective(start.ravel())[0]) / n_city
    if abs(true_objective - initial_objective - offset_check) > 1e-10:
        raise ArithmeticError("Convex head and direct residual objectives disagree")
    if not np.isfinite(projected) or projected > 5e-5:
        raise ArithmeticError(f"Head calibration stalled at projected gradient {projected}: {result.message}")
    return coefficients, dict(iterations=int(result.nit), projected_gradient=projected,
                              solver_status=str(result.message), objective=float(true_objective))
