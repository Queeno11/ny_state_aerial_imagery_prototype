"""Summarise round 2 (run_robust): bias of the signed angle, precision, MSE
ratios to RankNet, and the pre-registered decision rules.

    python -m minimal_e1.summarize_robust --out minimal_e1/run_robust
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ['FE', 'NS', 'L2', 'ML', 'RNG', 'RNC', 'SUR1', 'SUR03', 'SUR01']
ORDER = ORDER + [a + '_S' for a in ORDER] + ['MRC_S']
SCENARIO_TEXT = dict(A='inside Gaussian (map c)', B='city-varying noise scale (map c)',
                     C='city-varying noise shape (map c)', D='inside + skewed outside (map c)',
                     E='X-dependent skewed noise (map a; ranking not valid)',
                     F='5% gross errors (map c)')


def per_arm(rows):
    ok = rows[rows.status == 'ok']
    recs = []
    for (sc, block, C, n, arm), g in ok.groupby(['scenario', 'block', 'cities', 'n', 'arm']):
        B = len(g)
        ang = g.signed_angle.astype(float)
        rec = dict(scenario=sc, block=block, cities=C, n=n, arm=arm, B=B,
                   not_ok=int(((rows.scenario == sc) & (rows.block == block) & (rows.arm == arm)
                               & (rows.status != 'ok')).sum()),
                   f_mse=g.f_mse.mean(), f_mse_mcse=g.f_mse.std(ddof=1) / np.sqrt(B),
                   df_mse=g.df_mse.mean(),
                   angle_bias=ang.mean(), angle_sd=ang.std(ddof=1),
                   angle_t=ang.mean() / (ang.std(ddof=1) / np.sqrt(B)) if B > 1 else np.nan,
                   off_plane=g.off_plane.mean(),
                   a1_load=g.a1_load.mean(),
                   a1_t=g.a1_load.mean() / (g.a1_load.std(ddof=1) / np.sqrt(B)) if B > 1 else np.nan,
                   seconds=g.seconds.mean())
        recs.append(rec)
    out = pd.DataFrame(recs)
    # Stability arms lie in the H-plane: their A1 loading is identically zero.
    out.loc[out.arm.str.endswith('_S'), ['a1_load', 'a1_t', 'off_plane']] = np.nan
    # Paired MSE ratio against RankNet (global temperature) in the same space.
    ratios, ses = [], []
    for _, r in out.iterrows():
        refarm = 'RNG_S' if r.arm.endswith('_S') else 'RNG'
        cell = ok[(ok.scenario == r.scenario) & (ok.block == r.block)]
        a = cell[cell.arm == r.arm].set_index('rep').f_mse
        b = cell[cell.arm == refarm].set_index('rep').f_mse
        common = a.index.intersection(b.index)
        if len(common) < 2:
            ratios.append(np.nan); ses.append(np.nan); continue
        ratio = a[common].mean() / b[common].mean()
        # delta-method SE of a ratio of paired means
        da, db = a[common] - a[common].mean(), b[common] - b[common].mean()
        var = ((da / b[common].mean() - ratio * db / b[common].mean()) ** 2).mean() / len(common)
        ratios.append(ratio); ses.append(np.sqrt(var))
    out['mse_ratio_RNG'] = ratios
    out['mse_ratio_RNG_se'] = ses
    out['order'] = out.arm.map({a: i for i, a in enumerate(ORDER)})
    return out.sort_values(['scenario', 'block', 'order']).drop(columns='order')


def decisions(s):
    lines = ['## Pre-registered decision rules', '']
    small = s[s.block == 2]
    anchor = s[s.block == 3]

    def get(df, sc, arm, col):
        v = df[(df.scenario == sc) & (df.arm == arm)][col]
        return float(v.iloc[0]) if len(v) else np.nan

    # Rule 1: per-cell temperature.
    b_ratio = get(small, 'B', 'RNC', 'f_mse') / get(small, 'B', 'RNG', 'f_mse')
    a_ratio = get(small, 'A', 'RNC', 'f_mse') / get(small, 'A', 'RNG', 'f_mse')
    rule1 = b_ratio < 0.9 and a_ratio <= 1.0 + 2 * get(small, 'A', 'RNC', 'mse_ratio_RNG_se')
    lines.append(f'1. Per-cell temperature (RNC) over global (RNG): MSE ratio in B = {b_ratio:.3f} '
                 f'(need < 0.90), in A = {a_ratio:.3f} (need <= 1 within noise) -> '
                 f'**{"adopt RNC" if rule1 else "keep global"}** (small cells; no-stability arms).')
    # Rule 2: switch primary to the surrogate.
    lines.append('2. Switch primary to the surrogate (tau = 0.3) only if, in B-D at the anchor, '
                 'RankNet (per-cell, with stability) has a significant signed-angle bias with '
                 'bias^2 >= its variance at (32,40), while SUR03_S does not, and SUR03_S costs '
                 '<= 1.3x the variance in A:')
    var_cost = (get(small, 'A', 'SUR03_S', 'angle_sd') / get(small, 'A', 'RNC_S', 'angle_sd')) ** 2
    switch = []
    for sc in 'BCD':
        t_rn = get(anchor, sc, 'RNC_S', 'angle_t')
        bias2 = get(anchor, sc, 'RNC_S', 'angle_bias') ** 2
        var_small = get(small, sc, 'RNC_S', 'angle_sd') ** 2
        t_sur = get(anchor, sc, 'SUR03_S', 'angle_t')
        trig = abs(t_rn) > 2 and bias2 >= var_small and abs(t_sur) <= 2 and var_cost <= 1.3
        switch.append(trig)
        lines.append(f'   - {sc}: RNC_S anchor bias t = {t_rn:.2f}, bias^2 = {bias2:.2e} vs '
                     f'small-cell variance {var_small:.2e}; SUR03_S anchor t = {t_sur:.2f}'
                     f' -> {"triggers" if trig else "does not trigger"}')
    lines.append(f'   - SUR03_S / RNC_S variance in A (small cells): {var_cost:.2f} (need <= 1.3)')
    lines.append(f'   -> **{"switch to the surrogate" if any(switch) else "keep RankNet as primary"}**')
    f_rn, f_sur = get(small, 'F', 'RNC_S', 'f_mse'), get(small, 'F', 'SUR03_S', 'f_mse')
    lines.append(f'3. Contamination (F, small cells, with stability): SUR03_S / RNC_S MSE = '
                 f'{f_sur / f_rn:.2f}. A win only here means: keep RankNet as primary and report '
                 f'the surrogate as the contamination-robust variant.')
    lines.append('4. Boundary case E (X-dependent skewed noise): ranking estimators should show a '
                 'significant A1 loading without stability while FE should not; see the table.')
    return lines


def table(s, block, cols):
    lines = []
    for sc, g in s[s.block == block].groupby('scenario'):
        lines += [f'### {sc}: {SCENARIO_TEXT[sc]}', '',
                  '| arm | B | ' + ' | '.join(c for c, _ in cols) + ' |',
                  '|---|---:|' + '---:|' * len(cols)]
        for _, r in g.iterrows():
            cells = [fmt(r[k]) for _, k in cols]
            lines.append(f'| {r.arm} | {r.B}' + (f' [{r.not_ok}✗]' if r.not_ok else '')
                         + ' | ' + ' | '.join(cells) + ' |')
        lines.append('')
    return lines


def fmt(v):
    if isinstance(v, str):
        return v
    if not np.isfinite(v):
        return '–'
    return f'{v:.2e}' if abs(v) < 1e-2 or abs(v) >= 1e3 else f'{v:.3f}'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'run_robust'))
    args = ap.parse_args(argv)
    out = Path(args.out)
    rows = pd.read_csv(out / 'rows.csv')
    s = per_arm(rows)
    s.to_csv(out / 'summary.csv', index=False)
    cols = [('f-MSE', 'f_mse'), ('ratio to RankNet', 'mse_ratio_RNG'), ('± se', 'mse_ratio_RNG_se'),
            ('angle bias', 'angle_bias'), ('angle sd', 'angle_sd'), ('bias t', 'angle_t'),
            ('off-plane', 'off_plane'), ('A1 load', 'a1_load'), ('A1 t', 'a1_t')]
    md = ['# Round 2: robustness of the ranking scorer to the noise law', '',
          'Signed angle = atan2(θ̂·e2, θ̂·e1) in the H-plane (e1 = θ*); its mean is the bias. '
          'Off-plane = norm of θ̂ outside the H-plane (zero for stability arms). '
          'Ratio = paired mean f-MSE over RankNet with a global temperature in the same space.', '']
    md += decisions(s) + ['']
    md += ['## Small cells (32 cities × 40 tracts)', ''] + table(s, 2, cols)
    md += ['## Anchor (64 × 250), stability arms only', ''] + table(s, 3, cols)
    (out / 'summary.md').write_text('\n'.join(md), encoding='utf-8')
    bad = rows[rows.status != 'ok']
    print(f'{len(rows)} rows, {len(bad)} not ok -> {out / "summary.md"}')
    if len(bad):
        print(bad.groupby(['scenario', 'arm', 'status']).size().to_string())


if __name__ == '__main__':
    main()
