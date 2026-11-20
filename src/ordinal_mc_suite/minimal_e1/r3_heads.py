"""Round 3 second step: calibration heads on a FROZEN scorer, evaluated out of sample.

Heads (per city-date cell, fitted on the target calibration panel, applied to the test panel):
  map c : alpha + c1 exp(kappa s) + c2 exp(-kappa s),  c1 >= 0 >= c2   (family 'c')
  map a : alpha + beta s, beta >= 0                                     (family 'b')
  L2 head: heads.fit_head with q = 0.5 (least squares).
  L1 head: least absolute deviations, one linear program per cell-date (scipy linprog).
Metrics on the test panel: within-cell Kendall tau (head-free), and MAE / MSE of the
observed W against each head's prediction.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.stats import kendalltau

from .heads import fit_head
from .robust_dgp import noisy_panel

CAL_BLOCK, TEST_BLOCK, TEST_REP = 20, 21, 1


def head_family(map_name):
    return 'c' if map_name == 'c' else 'b'


def basis_columns(s, family, kappa):
    """(n, k) columns multiplying the sign-constrained coefficients."""
    if family == 'c':
        return np.stack([np.exp(kappa * s), np.exp(-kappa * s)], axis=-1)
    if family == 'b':
        return s[..., None]
    raise ValueError(family)


def coef_bounds(family):
    return [(0, None), (None, 0)] if family == 'c' else [(0, None)]


def lad_fit(s, w, family, kappa):
    """argmin_{alpha, coef in bounds} sum_i |w_i - alpha - B_i coef| for one cell-date,
    as an LP. Returns (alpha, coef, objective)."""
    n = len(s)
    B = basis_columns(s, family, kappa)
    k = B.shape[1]
    cost = np.concatenate([np.zeros(1 + k), np.ones(2 * n)])
    a_eq = np.hstack([np.ones((n, 1)), B, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] + coef_bounds(family) + [(0, None)] * (2 * n)
    res = linprog(cost, A_eq=a_eq, b_eq=w, bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(f'L1 head LP failed: {res.message}')
    return float(res.x[0]), res.x[1:1 + k].copy(), float(res.fun)


@dataclass
class L1Head:
    family: str
    kappa: float
    alpha: np.ndarray            # (C, 2)
    coef: np.ndarray             # (C, 2, k)

    def predict(self, s):
        """s: (C, n, 2) -> (C, n, 2)."""
        out = np.empty(s.shape)
        for c in range(s.shape[0]):
            for t in range(2):
                B = basis_columns(s[c, :, t], self.family, self.kappa)
                out[c, :, t] = self.alpha[c, t] + B @ self.coef[c, t]
        return out


def fit_l1_head(s, W, family, kappa):
    C, _, _ = s.shape
    k = len(coef_bounds(family))
    alpha, coef = np.zeros((C, 2)), np.zeros((C, 2, k))
    for c in range(C):
        for t in range(2):
            alpha[c, t], coef[c, t], _ = lad_fit(s[c, :, t], W[c, :, t], family, kappa)
    return L1Head(family, kappa, alpha, coef)


def fit_l2_head(s, W, family, kappa):
    return fit_head(s, W, family, 0.5, kappa)


def mean_kendall(s, W):
    """Average over city-date cells of Kendall's tau-b between score and outcome."""
    vals = [kendalltau(s[c, :, t], W[c, :, t])[0]
            for c in range(s.shape[0]) for t in range(s.shape[2])]
    return float(np.mean(vals))


def errors(pred, W):
    e = pred - W
    return float(np.mean(np.abs(e))), float(np.mean(e * e))


def step2_metrics(theta, cal, test, map_name, kappa):
    """Frozen scorer theta (unit vector in z coordinates) -> head-free and head-based
    out-of-sample metrics. cal/test = (z, W) with z (C, n, 2, p), W (C, n, 2)."""
    fam = head_family(map_name)
    s_cal, s_test = cal[0] @ theta, test[0] @ theta
    l1 = fit_l1_head(s_cal, cal[1], fam, kappa)
    l2 = fit_l2_head(s_cal, cal[1], fam, kappa)
    l1_mae, l1_mse = errors(l1.predict(s_test), test[1])
    l2_mae, l2_mse = errors(l2.predict(s_test), test[1])
    return dict(kendall=mean_kendall(s_test, test[1]), l1_mae=l1_mae, l1_mse=l1_mse,
                l2_mae=l2_mae, l2_mse=l2_mse)


_TEST_CACHE = {}


def target_panels(cfg, scenario, rep, cities=8, cal_n=20, test_n=500,
                  cal_block=CAL_BLOCK, test_block=TEST_BLOCK, test_rep=TEST_REP):
    """Calibration (fresh each rep) and test (fixed across reps) panels for the target.

    Both panels use the SAME 8 target cities (the test panel's city contexts, block
    test_block): a per-cell-date head can only be transported if the cell is the same
    city. Tract and noise draws are independent across the two panels (different blocks).
    The test panel needs n_draw >= test_n tracts, so n_draw is raised for it."""
    cal = noisy_panel(cfg, scenario, cities, cal_n, rep, cal_block,
                      city_key=(test_block, test_rep))
    key = (cfg['seed'], cfg['reference_size'], scenario, cities, test_n, test_block, test_rep)
    if key not in _TEST_CACHE:
        big = dict(cfg, n_draw=max(cfg['n_draw'], test_n))
        t = noisy_panel(big, scenario, cities, test_n, test_rep, test_block)
        _TEST_CACHE[key] = (t.z, t.W)
    return (cal.z, cal.W), _TEST_CACHE[key]
