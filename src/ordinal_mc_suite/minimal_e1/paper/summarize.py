"""Tidy CSVs from the per-replication JSON files.
python -m minimal_e1.paper.summarize --results minimal_e1/paper/results --out minimal_e1/paper/results/tables
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
def parse_metric(k):
    """'rmse_W_oracle' -> ('rmse', 'W', 'oracle'); 'rmse_f' -> ('rmse', 'f', 'index'); None if not a metric."""
    for stat in ('vdecC', 'vdec', 'rmse', 'r2'):
        if k.startswith(stat + '_'):
            rest = k[len(stat) + 1:].split('_', 1)
            return stat, rest[0], (rest[1] if len(rest) > 1 else 'index')
    return None


def load(results, exps=None):
    """Wide table: one row per (replication, arm), with every rmse_* / r2_* metric found in the files.
    Derived: r2_f = cos^2(angle) = (1 - rmse_f^2 / 2)^2 (same for fpool)."""
    rows = []
    dirs = [Path(r) for r in str(results).split(',')]
    paths = sorted(p for d in dirs for p in d.glob('E*/*/C*_n*/rep*.json'))
    for path in paths:
        rec = json.loads(path.read_text())
        if exps and rec['exp'] not in exps:
            continue
        for r in rec['rows']:
            d = dict(exp=rec['exp'], scenario=rec['scenario'], C=rec['C'], n=rec['n'], N=rec['N'], rep=rec['rep'],
                     arm=r['arm'], status=r['status'], ok=r.get('ok', np.nan), secs=r.get('secs', np.nan),
                     omega_sel=r.get('omega_sel', np.nan), fallback=r.get('fallback', 0),
                     spline_df=r.get('spline_df', np.nan), spline_df_upd=r.get('spline_df_upd', np.nan),
                     peak_rss_mb=rec['meta']['peak_rss_mb'], rep_secs=rec['meta']['seconds'],
                     q_hard=rec['meta']['q_hard'], n_links=rec['meta']['n_links'], backend=rec['meta']['backend'])
            d.update({k: v for k, v in r.items() if parse_metric(k) and isinstance(v, (int, float))})
            for idx in ('f', 'fpool'):
                if f'rmse_{idx}' in r:
                    d[f'r2_{idx}'] = (1.0 - r[f'rmse_{idx}'] ** 2 / 2.0) ** 2
            rows.append(d)
    return pd.DataFrame(rows)


def tidy(df):
    d = df[df.status == 'ok']
    metrics = [c for c in d.columns if parse_metric(c)]
    long = d.melt(id_vars=['exp', 'scenario', 'C', 'n', 'N', 'rep', 'arm', 'ok'], value_vars=metrics,
                  var_name='metric', value_name='value').dropna(subset=['value'])
    parts = long.metric.map(parse_metric)
    long['stat'] = parts.map(lambda x: x[0])
    long['target'] = parts.map(lambda x: x[1])
    long['head'] = parts.map(lambda x: x[2])
    return long.drop(columns='metric')


KEYS = ['exp', 'scenario', 'C', 'n', 'N', 'arm', 'stat', 'target', 'head']


def aggregate(long):
    g = long.groupby(KEYS)['value']
    out = g.agg(reps='count', mean='mean', sd='std', rms=lambda x: float(np.sqrt(np.mean(x ** 2)))).reset_index()
    out['mcse'] = out.sd / np.sqrt(out.reps)
    ok = long.groupby(KEYS)['ok'].apply(lambda s: int((s == False).sum())).reset_index(name='n_not_converged')
    return out.merge(ok, on=KEYS)


def timing(df):
    d = df[df.status == 'ok']
    t = d.groupby(['exp', 'C', 'n', 'arm']).agg(secs_mean=('secs', 'mean'), reps=('secs', 'count')).reset_index()
    r = df.groupby(['exp', 'C', 'n']).agg(rep_secs_mean=('rep_secs', 'mean'), peak_rss_mb=('peak_rss_mb', 'max')).reset_index()
    return t, r


def omega_table(df):
    d = df[(df.exp == 'E2') & df.arm.isin(['RNC', 'RNC_P0.5', 'RNC_P0.9', 'RNC_P0.99', 'RNC_S', 'RNC_SEL'])]
    d = d[d.status == 'ok']
    w = dict(RNC=0.0, **{'RNC_P0.5': 0.5, 'RNC_P0.9': 0.9, 'RNC_P0.99': 0.99}, RNC_S=1.0)
    d = d[d.arm.isin(list(w) + ['RNC_SEL'])].copy()
    d['omega'] = d.arm.map(w)
    g = d.groupby(['scenario', 'C', 'n', 'arm', 'omega'], dropna=False).agg(
        reps=('rmse_f', 'count'), rmse_f=('rmse_f', 'mean'), rmse_f_sd=('rmse_f', 'std'),
        rmse_W_oracle=('rmse_W_oracle', 'mean'), rmse_dW_oracle=('rmse_dW_oracle', 'mean'),
        omega_sel_mean=('omega_sel', 'mean')).reset_index()
    g['rmse_f_mcse'] = g.rmse_f_sd / np.sqrt(g.reps)
    return g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(HERE / 'results'))
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    out = Path(a.out or Path(a.results.split(',')[0]) / 'tables')
    out.mkdir(parents=True, exist_ok=True)
    df = load(a.results)
    if df.empty:
        print('no results')
        return
    df.to_csv(out / 'rows.csv', index=False)
    agg = aggregate(tidy(df))
    for exp, g in agg.groupby('exp'):
        g.to_csv(out / f'summary_{exp}.csv', index=False)
    agg.to_csv(out / 'summary_all.csv', index=False)
    t, r = timing(df)
    t.to_csv(out / 'timing_by_arm.csv', index=False)
    r.to_csv(out / 'timing_by_size.csv', index=False)
    vd = agg[agg['stat'].isin(['vdec', 'vdecC'])]
    if len(vd):
        vt = vd[vd.arm == 'RNC_S'].pivot_table(index=['exp', 'scenario', 'C', 'n', 'stat'], columns='target',
                                               values='mean').reset_index()
        vt.to_csv(out / 'variance_decomposition.csv', index=False)
    om = omega_table(df)
    if len(om):
        om.to_csv(out / 'omega_curve.csv', index=False)
    status = df.groupby(['exp', 'arm', 'status']).size().unstack(fill_value=0)
    status.to_csv(out / 'status_counts.csv')
    print(f'{len(df)} rows -> {out}')


if __name__ == '__main__':
    main()
