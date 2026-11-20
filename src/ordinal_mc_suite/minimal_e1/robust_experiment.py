"""Round 2: one dataset of a noise scenario -> every scorer arm -> metric rows."""
import json
import time
import traceback

import numpy as np

from .dgp import link_null_space
from .estimators import fixed_effects, fgls, cardinal, ordinal, maximum_likelihood, unit
from .heads import v_star
from .losses import PairData
from .robust_dgp import noisy_panel, evaluation_frame
from .robust_estimators import (ranknet_cell, surrogate, surrogate_2d, mrc_exact_2d,
                                normal_scores_fe)

TAUS = (1.0, 0.3, 0.1)
SUR_NAMES = {1.0: 'SUR1', 0.3: 'SUR03', 0.1: 'SUR01'}
FULL_ARMS = ['FE', 'NS', 'L2', 'ML', 'RNG', 'RNC', 'SUR1', 'SUR03', 'SUR01']
S_ARMS = [a + '_S' for a in FULL_ARMS] + ['MRC_S']


def arms(s_only):
    return S_ARMS if s_only else FULL_ARMS + S_ARMS


def _fit_space(z, W, pairs, map_name, cfg, suffix, basis, out, errors, seed):
    """All scorers in one coordinate system (full, or the link null space)."""
    kappa = cfg['kappa']
    vstar = v_star(W)
    two_d = z.shape[-1] == 2

    def put(name, fun, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            theta, extras, info = fun(*args, **kwargs)
            out[name + suffix] = (basis @ theta if basis is not None else theta,
                                  extras, dict(info, seconds=time.perf_counter() - t0))
        except Exception as exc:
            errors[name + suffix] = ''.join(traceback.format_exception_only(type(exc), exc)).strip()

    put('FE', fixed_effects, z, W)
    put('NS', normal_scores_fe, z, W)
    put('RNG', ordinal, pairs, cfg['newton_tol'])
    n = z.shape[1]
    put('RNC', ranknet_cell, pairs, n)
    local = lambda name: (basis.T @ out[name + suffix][0]) if basis is not None else out[name + suffix][0]
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
    t0 = time.perf_counter()
    try:
        if two_d:
            sur = surrogate_2d(pairs, TAUS)
        else:
            start = local('RNC') if 'RNC' + suffix in out else local('RNG')
            sur = surrogate(pairs, start, TAUS, seed=seed)
        sec = (time.perf_counter() - t0) / len(TAUS)
        for tau, (theta, info) in sur.items():
            out[SUR_NAMES[tau] + suffix] = (basis @ theta if basis is not None else theta, {},
                                            dict(info, seconds=sec))
    except Exception as exc:
        for tau in TAUS:
            errors[SUR_NAMES[tau] + suffix] = ''.join(
                traceback.format_exception_only(type(exc), exc)).strip()
    if two_d:
        put('MRC', mrc_exact_2d, pairs)


def metrics(theta, theta_star, frame, sigma_level, sigma_change):
    e1, e2, q, a1 = frame
    d = theta - theta_star
    return dict(cos_true=float(theta @ theta_star),
                f_mse=float(d @ sigma_level @ d),
                df_mse=float(d @ sigma_change @ d),
                signed_angle=float(np.arctan2(theta @ e2, theta @ e1)),
                off_plane=float(np.linalg.norm(theta - q @ (q.T @ theta))),
                a1_load=float(theta @ a1))


def run_dataset(cfg, target_moments, task):
    """task: dict(scenario, cities, n, rep, block, s_only)."""
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
    rows = []
    for arm in arms(task['s_only']):
        row = dict(scenario=task['scenario'], block=task['block'], cities=task['cities'],
                   n=task['n'], rep=task['rep'], arm=arm, map=panel.map_name,
                   link_rank=rank, contaminated=panel.contaminated, pair_ties=pairs.ties)
        if arm not in out:
            row.update(status='error', error=errors.get(arm, 'not fitted'))
            rows.append(row)
            continue
        theta, extras, info = out[arm]
        theta = unit(theta)
        row.update(status='ok' if info.get('success', True) else 'not_converged',
                   seconds=info.get('seconds'), objective=info.get('objective'),
                   start_spread=info.get('spread'), grad_norm=info.get('grad_norm'),
                   theta=json.dumps(np.round(theta, 12).tolist()),
                   extras=json.dumps({k: v for k, v in extras.items() if k != 'u'}, default=float))
        row.update(metrics(theta, theta_star, frame, sigma_level, sigma_change))
        rows.append(row)
    return rows, time.perf_counter() - t0
