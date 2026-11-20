"""One Monte Carlo dataset: generate, fit every arm for the requested maps,
evaluate on the fixed target environment, and return result rows."""
import json
import time
import traceback

import numpy as np

from .config import arms_for
from .dgp import (source_panel, calibration_sample, link_null_space, reference)
from .estimators import (fixed_effects, fgls, cardinal, ordinal, joint,
                         maximum_likelihood, mixed_model, unit)
from .evaluate import evaluate
from .heads import v_star
from .losses import PairData


def _timed(fun, *args, **kwargs):
    t0 = time.perf_counter()
    out = fun(*args, **kwargs)
    return out, time.perf_counter() - t0


class Fits:
    """Arm fits for one map, in full coordinates (restricted arms mapped back)."""

    def __init__(self):
        self.store = {}

    def put(self, arm, theta, extras, info, seconds, basis=None):
        if basis is not None:
            theta = unit(basis @ theta)
            extras = {k: v for k, v in extras.items() if k != 'u'}
        self.store[arm] = (theta, extras, dict(info, seconds=seconds))

    def theta(self, arm):
        return self.store[arm][0]


def fit_map(map_name, rho, z, zS, basis, W, pairs, pairsS, ordfit, cfg, theta_star):
    kappa, w = cfg['kappa'], cfg['joint_w']
    vstar = v_star(W)
    qstar = (1 + rho) / 2
    arms = arms_for(map_name, rho)
    fits = Fits()
    errors = {}
    gamma = lambda arm: basis.T @ fits.theta(arm)     # restricted coordinates

    (th, ex, inf), sec = _timed(fixed_effects, z, W)
    fits.put('FE', th, ex, inf, sec)
    fe_s, _, _ = fixed_effects(zS, W)                 # start value only
    fits.put('ORD', *ordfit['ORD'])
    fits.put('ORD_S', *ordfit['ORD_S'])
    u_ord, u_ord_s = ordfit['ORD'][1]['u'], ordfit['ORD_S_u']

    def run(arm, fun, *args, basis_=None, **kwargs):
        try:
            (th, ex, inf), sec = _timed(fun, *args, **kwargs)
            fits.put(arm, th, ex, inf, sec, basis_)
        except Exception as exc:                       # recorded, never invented
            errors[arm] = ''.join(traceback.format_exception_only(type(exc), exc)).strip()

    common = dict(gtol=cfg['sphere_gtol'], max_iter=cfg['max_iter'])
    jcommon = dict(gtol=cfg['joint_gtol'], max_iter=cfg['max_iter'])
    for arm, q in (('L2', 0.5), ('L2D', qstar)):
        if arm not in arms:
            continue
        run(arm, cardinal, z, W, map_name, q, kappa, vstar,
            [fits.theta('FE'), fits.theta('ORD')], **common)
        run(arm + '_S', cardinal, zS, W, map_name, q, kappa, vstar,
            [fe_s, gamma('ORD_S')], basis_=basis, **common)
    for arm, card_arm, q in (('ORD_L2', 'L2', 0.5), ('ORD_L2D', 'L2D', qstar)):
        if arm not in arms:
            continue
        starts = [u_ord] + ([fits.theta(card_arm) * np.linalg.norm(u_ord)]
                            if card_arm in fits.store else [])
        run(arm, joint, z, W, pairs, map_name, q, kappa, vstar, w, starts, **jcommon)
        starts_s = [u_ord_s] + ([gamma(card_arm + '_S') * np.linalg.norm(u_ord_s)]
                                if card_arm + '_S' in fits.store else [])
        run(arm + '_S', joint, zS, W, pairsS, map_name, q, kappa, vstar, w, starts_s,
            basis_=basis, **jcommon)
    if map_name == 'a':
        run('ML', fgls, z, W)
        run('ML_S', fgls, zS, W, basis_=basis)
    else:
        mlkw = dict(lam_bound=cfg['ml_lambda_bound'], gtol=cfg['ml_gtol'], max_iter=cfg['max_iter'])
        starts = [fits.theta(a) for a in ('L2', 'ORD', 'FE') if a in fits.store]
        run('ML', maximum_likelihood, z, W, map_name, kappa, starts, **mlkw)
        starts_s = [gamma(a) for a in ('L2_S', 'ORD_S') if a in fits.store] + [fe_s]
        run('ML_S', maximum_likelihood, zS, W, map_name, kappa, starts_s, basis_=basis, **mlkw)
    if 'MM' in arms:
        run('MM', mixed_model, z, W, cfg['mm_spec'])
    if 'L2aff' in arms:
        run('L2aff', cardinal, z, W, 'b', 0.5, kappa, vstar,
            [fits.theta('FE'), fits.theta('ORD')], **common)
    fits.put('TRUE', theta_star, {}, dict(success=True, n_starts=0, spread=0.0), 0.0)
    return fits, errors


def run_dataset(cfg, target, task):
    """task: dict(block, rho, cities, n, rep, maps)."""
    t0 = time.perf_counter()
    rho, rep = task['rho'], task['rep']
    ref = reference(cfg)
    theta_star = ref.theta_star
    panel = source_panel(cfg, rho, task['cities'], task['n'], rep, task['block'], task['maps'])
    basis, rank, sv = link_null_space(panel.z, panel.stable)
    zS = panel.z @ basis
    first = task['maps'][0]
    pairs = PairData.build(panel.z, panel.W[first])
    # Ordinal fits are shared across maps: monotone maps preserve within-cell order.
    same_ranks = all(np.array_equal(np.sign(np.diff(panel.W[m], axis=1)),
                                    np.sign(np.diff(panel.W[first], axis=1)))
                     for m in task['maps'])
    pairsS = pairs.project(basis)
    (th, ex, inf), sec = _timed(ordinal, pairs, cfg['newton_tol'])
    (ths, exs, infs), secs = _timed(ordinal, pairsS, cfg['newton_tol'])
    ordfit = {'ORD': (th, ex, inf, sec),
              'ORD_S': (unit(basis @ ths), {'tau': exs['tau']}, infs, secs),
              'ORD_S_u': exs['u']}
    cal_z, cal_W, _ = calibration_sample(cfg, target, rho, rep, task['maps'])
    restriction_gap = float(np.max(np.abs((panel.z[:, panel.stable, 1] - panel.z[:, panel.stable, 0])
                                          @ theta_star)))
    rows = []
    for map_name in task['maps']:
        W = panel.W[map_name]
        fits, errors = fit_map(map_name, rho, panel.z, zS, basis, W, pairs, pairsS,
                               ordfit, cfg, theta_star)
        for arm in arms_for(map_name, rho):
            row = dict(block=task['block'], rho=rho, cities=task['cities'], n=task['n'],
                       rep=rep, map=map_name, arm=arm, link_rank=rank,
                       free_dim=basis.shape[1], restriction_gap_true=restriction_gap,
                       ranks_shared=same_ranks, pair_ties=pairs.ties)
            if arm in errors or arm not in fits.store:
                row.update(status='error', error=errors.get(arm, 'not fitted'))
                rows.append(row)
                continue
            theta, extras, info = fits.store[arm]
            row.update(status='ok' if info.get('success', True) else 'not_converged',
                       seconds=info.get('seconds'), objective=info.get('objective'),
                       start_spread=info.get('spread'), n_starts=info.get('n_starts'),
                       grad_norm=info.get('grad_norm'), iterations=info.get('iterations'),
                       message=info.get('message'),
                       theta=json.dumps(np.round(theta, 12).tolist()),
                       extras=json.dumps({k: v for k, v in extras.items() if k != 'u'},
                                         default=float))
            row.update(evaluate(theta, arm, map_name, extras, panel.z, W, target,
                                cal_z, cal_W[map_name], cfg, theta_star))
            rows.append(row)
    return rows, time.perf_counter() - t0
