"""Calibration of the stability penalty (curvature share), policy file and the pilot.

lambda(omega) = omega tr(H_R) / ((1 - omega) tr(H_S))   [tangent space of the unit sphere, at the
hard estimate on TRAINING data]; lambda_g = ||grad R|| / ||grad P|| at the starting value.
policy file paper/calibration.json:
  {"mode": "fixed" | "per_rep", "lambda": {"0.5": .., "0.9": .., "0.99": ..}, "lambda_g": .., "cv": {...}, ...}
"""
import json
from pathlib import Path

import numpy as np

from . import stability as st

CAL_PATH = Path(__file__).resolve().parent / 'calibration.json'
OMEGAS = (0.5, 0.9, 0.99)
CV_THRESHOLD = 0.25


def load_policy(path=None):
    path = Path(path) if path else CAL_PATH
    if path.exists():
        return json.loads(path.read_text())
    return None


def lambda_diagnostics(cs, pen, th, M, th_start=None, rng=None, n_random=20):
    """Traces, lambda(omega) and gradient-balance lambdas at (th, M)."""
    trR, trS = st.curvature_traces(cs, pen, th, M)
    out = dict(trR=trR, trS=trS, **{f'lam_{w}': float(st.lambda_for_omega(w, trR, trS)) for w in OMEGAS})
    if th_start is not None:
        out['lam_g'] = st.lambda_gradient_balance(cs, pen, th_start / np.linalg.norm(th_start), M)
    rng = rng or np.random.default_rng(0)
    vals = []
    for _ in range(n_random):
        v = rng.normal(size=len(th))
        v /= np.linalg.norm(v)
        vals.append(st.lambda_gradient_balance(cs, pen, v, M))
    out['lam_g_random'] = float(np.mean(vals))
    return out


def summarise(rows):
    """rows: list of per-rep diagnostics -> policy dict (mode fixed iff CV small for all omega)."""
    def col(k):
        return np.array([r[k] for r in rows], float)
    cv = {}
    lam = {}
    for w in OMEGAS:
        x = col(f'lam_{w}')
        cv[str(w)] = float(x.std(ddof=1) / abs(x.mean()))
        lam[str(w)] = float(np.median(x))
    mode = 'fixed' if max(cv.values()) < CV_THRESHOLD else 'per_rep'
    out = {'mode': mode, 'lambda': lam, 'cv': cv, 'cv_threshold': CV_THRESHOLD, 'n_reps': len(rows),
           'lambda_mean': {str(w): float(col(f'lam_{w}').mean()) for w in OMEGAS},
           'lambda_q05': {str(w): float(np.quantile(col(f'lam_{w}'), .05)) for w in OMEGAS},
           'lambda_q95': {str(w): float(np.quantile(col(f'lam_{w}'), .95)) for w in OMEGAS},
           'trR_mean': float(col('trR').mean()), 'trS_mean': float(col('trS').mean())}
    for k in ('lam_g', 'lam_g_random', 'lam_u_0.9'):
        if all(k in r for r in rows):
            x = col(k)
            out[k] = dict(median=float(np.median(x)), cv=float(x.std(ddof=1) / abs(x.mean())))
    return out


def lambda_for(policy, omega, compute):
    """Penalty weight for omega: fixed from the policy, or per replication via compute() -> lambda."""
    if omega >= 1.0:
        return np.inf
    if omega <= 0.0:
        return 0.0
    if policy is not None and policy.get('mode') == 'fixed':
        return float(policy['lambda'][str(omega)])
    return float(compute(omega))
