"""Target-city evaluation of a frozen source scorer.

Three tracks for the outcome mean:
  oracle  - true target conditional-mean map applied to the estimated score
            (representation-only; no map fitting)
  cal     - the correct-family head fitted on the fresh 20-tract-per-city target
            calibration sample, identical rule for every arm (q = 0.5)
  native  - each arm's own head family (differs from 'cal' only for the
            practitioner arms: FE and MM use a common slope with re-fitted
            intercepts, L2aff uses an affine city-date head)
In map a the common slope b is never re-estimated at the target: every arm
carries its own source b (from its fit, or from a source q=0.5 head on its
frozen score when the criterion has no slope of its own).
"""
import numpy as np

from .dgp import conditional_mean
from .heads import fit_head

NATIVE_FAMILY = dict(FE='a', MM='a', L2aff='b')


def source_slope(theta, extras, z, W, kappa, arm):
    if arm == 'TRUE':
        return 1.0
    if extras.get('b') is not None:
        return float(extras['b'])
    return float(fit_head(z @ theta, W, 'a', 0.5, kappa).coef)


def _mse(a, b):
    return float(np.mean((a - b) ** 2))


def _change(x):
    return x[..., 1] - x[..., 0]


def evaluate(theta, arm, map_name, extras, src_z, src_W, target, cal_z, cal_W, cfg, theta_star):
    kappa = cfg['kappa']
    s = target.z @ theta
    changed = ~target.stable
    out = dict(
        cos_true=float(theta @ theta_star),
        f_mse=_mse(s, target.f),
        df_mse=_mse(_change(s), _change(target.f)),
        df_mse_changed=_mse(_change(s)[:, changed], _change(target.f)[:, changed]),
        ds_rms_stable=float(np.sqrt(np.mean(_change(s)[:, ~changed] ** 2))),
    )
    mu = target.mu[map_name]
    oracle = conditional_mean(s, map_name, target.params, kappa)
    out['mu_oracle_mse'] = _mse(oracle, mu)
    out['dmu_oracle_mse'] = _mse(_change(oracle), _change(mu))
    b = None
    if map_name == 'a' or NATIVE_FAMILY.get(arm) == 'a':
        b = source_slope(theta, extras, src_z, src_W, kappa, arm)
        out['b_source'] = b
    s_cal = cal_z @ theta
    for track, family in (('cal', map_name), ('native', NATIVE_FAMILY.get(arm, map_name))):
        if track == 'native' and family == map_name:
            out['mu_native_mse'], out['dmu_native_mse'] = out['mu_cal_mse'], out['dmu_cal_mse']
            continue
        head = fit_head(s_cal, cal_W, family, 0.5, kappa, fixed_b=b if family == 'a' else None)
        pred = head.predict(s)
        out[f'mu_{track}_mse'] = _mse(pred, mu)
        out[f'dmu_{track}_mse'] = _mse(_change(pred), _change(mu))
    return out
