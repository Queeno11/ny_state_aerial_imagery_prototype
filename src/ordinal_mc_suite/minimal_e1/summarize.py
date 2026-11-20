"""Summarise rows.csv: means, Monte Carlo SEs, paired contrasts, bias/variance.

    python -m minimal_e1.summarize --out minimal_e1/run_5rep

Writes summary.csv (one line per block/rho/size/map/arm) and summary.md.
With only 5 replications, MCSEs are rough: read patterns, not significance.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ['f_mse', 'df_mse', 'mu_oracle_mse', 'dmu_oracle_mse', 'mu_cal_mse', 'dmu_cal_mse',
           'mu_native_mse', 'dmu_native_mse']
KEYS = ['block', 'rho', 'cities', 'n', 'map']
ARM_ORDER = ['TRUE', 'ML', 'ML_S', 'FE', 'MM', 'L2aff', 'L2', 'L2_S', 'L2D', 'L2D_S',
             'ORD', 'ORD_S', 'ORD_L2', 'ORD_L2_S', 'ORD_L2D', 'ORD_L2D_S']
TABLES = [('f_mse', 'Latent level f: MSE'), ('df_mse', 'Latent change Delta f: MSE'),
          ('mu_cal_mse', 'Outcome mean mu, calibrated head (20 tracts/city): MSE'),
          ('dmu_cal_mse', 'Outcome change Delta mu, calibrated head: MSE'),
          ('mu_oracle_mse', 'Outcome mean mu, oracle target map (representation only): MSE'),
          ('mu_native_mse', 'Outcome mean mu, each arm\'s own head family: MSE')]


def bias_variance(thetas, theta_star, sigma):
    """Exact empirical identity: mean_b err_b = bias^2 + variance (ddof 0)."""
    t = np.stack(thetas)
    mean = t.mean(0)
    bias2 = float((mean - theta_star) @ sigma @ (mean - theta_star))
    dev = t - mean
    variance = float(np.mean(np.einsum('bi,ij,bj->b', dev, sigma, dev)))
    return bias2, variance


def summarise(rows, theta_star, sig_level, sig_change):
    ok = rows[rows.status == 'ok'].copy()
    out = []
    for key, grp in rows.groupby(KEYS + ['arm']):
        good = ok.loc[ok.index.intersection(grp.index)]
        rec = dict(zip(KEYS + ['arm'], key))
        rec.update(B=len(good), not_ok=int((grp.status != 'ok').sum()),
                   seconds=float(good.seconds.mean()) if len(good) else np.nan,
                   start_spread_max=float(good.start_spread.max()) if len(good) else np.nan)
        for m in METRICS:
            vals = good[m].astype(float)
            rec[m] = float(vals.mean()) if len(vals) else np.nan
            rec[m + '_mcse'] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
        if len(good):
            thetas = [np.array(json.loads(t)) for t in good.theta]
            rec['f_bias2'], rec['f_var'] = bias_variance(thetas, theta_star, sig_level)
            rec['df_bias2'], rec['df_var'] = bias_variance(thetas, theta_star, sig_change)
        out.append(rec)
    summary = pd.DataFrame(out)
    # Paired contrasts against FE and ML on the replications both completed.
    for ref in ('FE', 'ML'):
        for m in ('f_mse', 'df_mse', 'mu_cal_mse', 'dmu_cal_mse'):
            ratio, diff, se = [], [], []
            for _, rec in summary.iterrows():
                cell = ok[(ok[KEYS] == pd.Series({k: rec[k] for k in KEYS})).all(axis=1)]
                a = cell[cell.arm == rec['arm']].set_index('rep')[m].astype(float)
                b = cell[cell.arm == ref].set_index('rep')[m].astype(float)
                common = a.index.intersection(b.index)
                if len(common) == 0 or rec['arm'] == ref:
                    ratio.append(np.nan); diff.append(np.nan); se.append(np.nan)
                    continue
                d = a[common] - b[common]
                ratio.append(float(a[common].mean() / b[common].mean()))
                diff.append(float(d.mean()))
                se.append(float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan)
            summary[f'{m}_rel_{ref}'] = ratio
            summary[f'{m}_diff_{ref}'] = diff
            summary[f'{m}_diff_{ref}_mcse'] = se
    summary['arm_order'] = summary.arm.map({a: i for i, a in enumerate(ARM_ORDER)})
    return summary.sort_values(KEYS + ['arm_order']).drop(columns='arm_order')


def _fmt(v):
    if not np.isfinite(v):
        return '–'
    return f'{v:.2e}' if (abs(v) < 1e-3 or abs(v) >= 1e3) else f'{v:.4f}'


def markdown(summary):
    lines = ['# Minimal Experiment 1: summary', '',
             'Cells show mean over replications, with the ratio to FE (same replications) in '
             'parentheses. Columns: block (S = small, A = anchor) / rho / C×n. '
             'With 5 replications treat differences below ~2 MCSE (see summary.csv) as noise.', '']
    summary = summary.copy()
    summary['col'] = (summary.block.map({0: 'S', 1: 'A'}) + ' ρ=' + summary.rho.map('{:g}'.format)
                      + ' ' + summary.cities.astype(str) + '×' + summary.n.astype(str))
    for map_name, mgrp in summary.groupby('map'):
        lines += [f'## Map ({map_name})', '']
        cols = list(dict.fromkeys(mgrp.sort_values(['block', 'rho', 'cities', 'n']).col))
        for metric, title in TABLES:
            lines += [f'### {title}', '', '| arm | ' + ' | '.join(cols) + ' |',
                      '|---|' + '---:|' * len(cols)]
            for arm in [a for a in ARM_ORDER if a in set(mgrp.arm)]:
                cells = []
                for col in cols:
                    rec = mgrp[(mgrp.arm == arm) & (mgrp.col == col)]
                    if rec.empty:
                        cells.append('')
                        continue
                    rec = rec.iloc[0]
                    rel = rec.get(f'{metric}_rel_FE', np.nan) if metric in (
                        'f_mse', 'df_mse', 'mu_cal_mse', 'dmu_cal_mse') else np.nan
                    cell = _fmt(rec[metric])
                    if np.isfinite(rel):
                        cell += f' ({rel:.2f})'
                    if rec['not_ok']:
                        cell += f' [{int(rec["not_ok"])}✗]'
                    cells.append(cell)
                lines.append(f'| {arm} | ' + ' | '.join(cells) + ' |')
            lines.append('')
        lines += ['### Bias² / variance of f (target panel, from coefficients)', '',
                  '| arm | ' + ' | '.join(cols) + ' |', '|---|' + '---:|' * len(cols)]
        for arm in [a for a in ARM_ORDER if a in set(mgrp.arm) and a != 'TRUE']:
            cells = []
            for col in cols:
                rec = mgrp[(mgrp.arm == arm) & (mgrp.col == col)]
                cells.append('' if rec.empty else
                             f"{_fmt(rec.iloc[0]['f_bias2'])} / {_fmt(rec.iloc[0]['f_var'])}")
            lines.append(f'| {arm} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'run_5rep'))
    args = ap.parse_args(argv)
    out = Path(args.out)
    rows = pd.read_csv(out / 'rows.csv')
    moments = np.load(out / 'target_moments.npz')
    summary = summarise(rows, moments['theta_star'], moments['sigma_level'], moments['sigma_change'])
    summary.to_csv(out / 'summary.csv', index=False)
    (out / 'summary.md').write_text(markdown(summary), encoding='utf-8')
    bad = rows[rows.status != 'ok']
    print(f'{len(rows)} rows, {len(bad)} not ok. Wrote {out / "summary.csv"} and {out / "summary.md"}')
    if len(bad):
        print(bad.groupby(['map', 'arm', 'status']).size().to_string())


if __name__ == '__main__':
    main()
