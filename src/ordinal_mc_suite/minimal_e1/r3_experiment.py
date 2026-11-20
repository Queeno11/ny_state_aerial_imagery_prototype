"""Round 3: one dataset of a noise scenario -> scorer arms -> scorer metrics + step-2 heads."""
import json
import time
import traceback

import numpy as np

from .dgp import link_null_space
from .estimators import fixed_effects, fgls, cardinal, ordinal, maximum_likelihood, unit
from .heads import v_star
from .losses import PairData
from .r3_estimators import surn, surh
from .r3_heads import step2_metrics, target_panels
from .robust_dgp import noisy_panel
from .robust_estimators import ranknet_cell, mrc_exact_2d, normal_scores_fe
from .robust_experiment import metrics

FULL_ARMS = ['FE', 'NS', 'L2', 'ML', 'RNC', 'SURN', 'SURH']
S_ARMS = [a + '_S' for a in FULL_ARMS] + ['MRC_S']
STEP2_KEYS = ['kendall', 'l1_mae', 'l1_mse', 'l2_mae', 'l2_mse']


def arms(s_only):
    return S_ARMS if s_only else FULL_ARMS + S_ARMS


def _fit_space(z, W, pairs, map_name, cfg, suffix, basis, out, errors, seed):
    """All scorers in one coordinate system (full, or the link null space).
    RNG (global-temperature RankNet) is fitted as a helper (L2 start, tau0 fallback)
    but is not an arm."""
    kappa = cfg['kappa']
    vstar = v_star(W)
    two_d = z.shape[-1] == 2
    n = z.shape[1]

    def put(name, fun, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            theta, extras, info = fun(*args, **kwargs)
            out[name + suffix] = (basis @ theta if basis is not None else theta,
                                  extras, dict(info, seconds=time.perf_counter() - t0))
        except Exception as exc:
            errors[name + suffix] = ''.join(traceback.format_exception_only(type(exc), exc)).strip()

    local = lambda name: (basis.T @ out[name + suffix][0]) if basis is not None else out[name + suffix][0]
    put('FE', fixed_effects, z, W)
    put('NS', normal_scores_fe, z, W)
    put('RNG', ordinal, pairs, cfg['newton_tol'])
    put('RNC', ranknet_cell, pairs, n)
    starts = [local(a) for a in ('FE', 'RNG') if a + suffix in out]
    if map_name == 'a':
        put('L2', cardinal, z, W, 'a', 0.5, kappa, vstar, [])
        put('ML', fgls, z, W)
    else:
        put('L2', cardinal, z, W, map_name, 0.5, kappa, vstar, starts,
            gtol=cfg['sphere_gtol'], max_iter=cfg['max_iter'])
        ml_starts = [local(a) for a in ('L2', 'RNG', 'FE') if a + suffix in out]
        put('ML', maximum_likelihood, z, W, map_name, kappa, ml_starts,
            lam_bound=cfg['ml_lambda_bound'], gtol=cfg['ml_gtol'], max_iter=cfg['max_iter'])
    # Start and fixed RankNet temperature for the surrogates: RNC, else RNG.
    if 'RNC' + suffix in out:
        start, tau0 = local('RNC'), out['RNC' + suffix][1]['tau_median']
    elif 'RNG' + suffix in out:
        start, tau0 = local('RNG'), out['RNG' + suffix][1]['tau']
    else:
        start, tau0 = None, None
    if start is None and not two_d:
        start = local('FE')
    put('SURN', surn, pairs, n, start, seed)
    if tau0 is None or not np.isfinite(tau0):
        errors['SURH' + suffix] = 'no RankNet temperature available'
    else:
        put('SURH', surh, pairs, n, start, float(tau0), seed)
    if two_d:
        put('MRC', mrc_exact_2d, pairs)


def run_dataset(cfg, target_moments, task):
    """task: dict(scenario, cities, n, rep, block, s_only, [cal_n, test_n])."""
    t0 = time.perf_counter()
    theta_star, sigma_level, sigma_change, frame = target_moments
    panel = noisy_panel(cfg, task['scenario'], task['cities'], task['n'], task['rep'], task['block'])
    basis, rank, _ = link_null_space(panel.z, panel.stable)
    pairs = PairData.build(panel.z, panel.W)
    out, errors = {}, {}
    seed = int(1e6 * task['block'] + 1e3 * task['rep'] + task['cities'])
    if not task['s_only']:
        _fit_space(panel.z, panel.W, pairs, panel.map_name, cfg, '', None, out, errors, seed)
    _fit_space(panel.z @ basis, panel.W, pairs.project(basis), panel.map_name, cfg, '_S',
               basis, out, errors, seed)
    cal, test = target_panels(cfg, task['scenario'], task['rep'], cal_n=task.get('cal_n', 20),
                              test_n=task.get('test_n', 500))
    rows = []
    for arm in ['TRUE'] + arms(task['s_only']):
        row = dict(scenario=task['scenario'], block=task['block'], cities=task['cities'],
                   n=task['n'], rep=task['rep'], arm=arm, map=panel.map_name,
                   link_rank=rank, contaminated=panel.contaminated, pair_ties=pairs.ties)
        if arm == 'TRUE':
            theta, extras, info = theta_star, {}, dict(seconds=0.0)
        elif arm not in out:
            row.update(status='error', error=errors.get(arm, 'not fitted'))
            rows.append(row)
            continue
        else:
            theta, extras, info = out[arm]
            theta = unit(theta)
        row.update(status='ok' if info.get('success', True) else 'not_converged',
                   seconds=info.get('seconds'), objective=info.get('objective'),
                   start_spread=info.get('spread'), grad_norm=info.get('grad_norm'),
                   theta=json.dumps(np.round(theta, 12).tolist()),
                   extras=json.dumps({k: v for k, v in extras.items() if k != 'u'}, default=float))
        row.update(metrics(theta, theta_star, frame, sigma_level, sigma_change))
        try:
            row.update(step2_metrics(theta, cal, test, panel.map_name, cfg['kappa']))
        except Exception as exc:
            row['error'] = 'step2: ' + ''.join(traceback.format_exception_only(type(exc), exc)).strip()
        rows.append(row)
    return rows, time.perf_counter() - t0
