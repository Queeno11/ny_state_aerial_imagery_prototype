"""Head-to-head of second-step heads (one replication, rep 0). From /mnt/user-data/outputs:
    PYTHONPATH=. python -m minimal_e1.paper.head_check
"""
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import arms, calibration, dgp, experiments, heads, heads_spline as hs

CELLS = [('E1_a', 8, 500), ('E1_a', 64, 500), ('E1_c', 8, 150), ('E1_c', 8, 500), ('E1_c', 64, 500),
         ('E1_cD', 8, 500)]
OUT = Path(__file__).parent / 'results_headcheck'
NREP_TIME = 5


def head_fns(panel):
    """name -> callable(s_tr, W_tr, s_te) -> (pred, df or nan)."""
    return {
        'oracle': lambda s_tr, W_tr, s_te: (heads.oracle_predict(panel.spec, panel.params, s_te), np.nan),
        'iso': lambda s_tr, W_tr, s_te: (heads.iso_predict(s_tr, W_tr, s_te), np.nan),
        'linear': lambda s_tr, W_tr, s_te: (hs.bspline_predict(s_tr, W_tr, s_te, 2), 2),
        'cv_spline': lambda s_tr, W_tr, s_te: hs.cv_spline_predict(s_tr, W_tr, s_te),
        'cv_mono': lambda s_tr, W_tr, s_te: hs.cv_spline_predict(s_tr, W_tr, s_te, monotone=True),
    }


def r2(obs, pred):
    return float(1.0 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2))


def score(pred, mu, obs):
    d = lambda a: a[..., 1] - a[..., 0]
    return dict(rmse_W=heads.rmse(pred, mu), rmse_dW=heads.rmse(d(pred), d(mu)),
                r2_W=r2(obs, pred), r2_dW=r2(d(obs), d(pred)))


def run_cell(scn, C, n, policy, rep=0):
    spec = experiments.SCENARIOS['E1'][scn]
    key = dgp.key_of('E1', scn, C, n, rep)
    panel = dgp.make_panel(spec, C, n, key)
    tr, te = dgp.split_tracts(panel, key)
    ctx = arms.Ctx(panel, tr, te, key, dict(policy=policy))
    z_tr, z_te = dgp.take(panel.z, tr), dgp.take(panel.z, te)
    W_tr, W_te = dgp.take(panel.W, tr), dgp.take(panel.W, te)
    mu = dgp.cond_mean(panel.spec, panel.params, dgp.take(panel.f, te), dgp.take(panel.a1c, te))
    fns = head_fns(panel)
    rows = []
    for src, theta in (('RNC_S', arms.get(ctx, 'RNC_S')['theta']), ('true', panel.theta0)):
        s_tr, s_te = z_tr @ theta, z_te @ theta
        for name, fn in fns.items():
            times = []
            for _ in range(NREP_TIME):
                t0 = time.perf_counter()
                pred, df = fn(s_tr, W_tr, s_te)
                times.append(time.perf_counter() - t0)
            rows.append(dict(scenario=scn, C=C, n=n, theta_src=src, head=name, df=df,
                             secs=statistics.median(times), **score(pred, mu, W_te)))
    return rows


def tabulate(df):
    keys = ['scenario', 'C', 'n', 'theta_src']
    out = []
    for _, g in df.groupby(keys, sort=False):
        g = g.set_index('head')
        for h in g.index:
            r = g.loc[h]
            out.append({**{k: r[k] for k in keys}, 'head': h, 'df': r['df'],
                        'time_x_iso': r['secs'] / g.loc['iso', 'secs'],
                        'W_x_iso': r['rmse_W'] / g.loc['iso', 'rmse_W'],
                        'W_x_orc': r['rmse_W'] / g.loc['oracle', 'rmse_W'] if g.loc['oracle', 'rmse_W'] > 1e-8 else np.nan,
                        'dW_x_iso': r['rmse_dW'] / g.loc['iso', 'rmse_dW'],
                        'dW_x_orc': r['rmse_dW'] / g.loc['oracle', 'rmse_dW'] if g.loc['oracle', 'rmse_dW'] > 1e-8 else np.nan,
                        'rmse_W': r['rmse_W'], 'rmse_dW': r['rmse_dW'], 'secs': r['secs'],
                        'R2_W': r['r2_W'], 'R2_dW': r['r2_dW'], 'R2_W_orc': g.loc['oracle', 'r2_W'],
                        'R2_dW_orc': g.loc['oracle', 'r2_dW']})
    return pd.DataFrame(out)


def main():
    policy = calibration.load_policy()
    rows = []
    for scn, C, n in CELLS:
        t0 = time.perf_counter()
        rows += run_cell(scn, C, n, policy)
        print(f'{scn} {C}x{n} done in {time.perf_counter() - t0:.0f}s', file=sys.stderr, flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / 'headcheck.csv', index=False)
    tab = tabulate(df)
    tab.to_csv(OUT / 'headcheck_table.csv', index=False)   # ratio table (NaN ratio vs oracle when oracle error is 0)
    pd.set_option('display.width', 250, 'display.max_rows', 500)
    print(tab.to_string(index=False, float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    main()
