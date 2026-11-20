"""Summarise round 3 (run_r3): scorer accuracy, signed-angle bias, paired MSE ratios to RNC,
second-step heads, and the decision sections.

    python -m minimal_e1.summarize_r3 --out minimal_e1/run_r3
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = ['FE', 'NS', 'L2', 'ML', 'RNC', 'SURN', 'SURH']
ORDER = ['TRUE'] + BASE + [a + '_S' for a in BASE] + ['MRC_S']
SCENARIO_TEXT = dict(A='inside Gaussian (map c)', A8='as A, persistence rho = 0.8 (map c)',
                     B='city-varying noise scale (map c)', C='city-varying noise shape (map c)',
                     D='inside + skewed outside (map c)',
                     E='X-dependent skewed noise (map a; ranking not valid)',
                     F='5% gross errors (map c)')
STEP2 = ['kendall', 'l1_mae', 'l1_mse', 'l2_mae', 'l2_mse']
KEYS = ['scenario', 'cities', 'n']


def paired_ratio(a, b):
    """Ratio of paired means mean(a)/mean(b) (Series indexed by rep) with delta-method SE."""
    common = a.index.intersection(b.index)
    if len(common) < 2 or b[common].mean() == 0:
        return np.nan, np.nan
    a, b = a[common], b[common]
    ratio = a.mean() / b.mean()
    da, db = a - a.mean(), b - b.mean()
    var = ((da / b.mean() - ratio * db / b.mean()) ** 2).mean() / len(common)
    return float(ratio), float(np.sqrt(var))


def _col(ok, scen, cities, n, arm, col):
    g = ok[(ok.scenario == scen) & (ok.cities == cities) & (ok.n == n) & (ok.arm == arm)]
    return g.set_index('rep')[col].astype(float)


def per_arm(rows):
    ok = rows[rows.status == 'ok']
    recs = []
    for (sc, C, n, arm), g in ok.groupby(KEYS + ['arm']):
        B = len(g)
        ang = g.signed_angle.astype(float)
        sd = ang.std(ddof=1) if B > 1 else np.nan
        rec = dict(scenario=sc, cities=C, n=n, arm=arm, B=B,
                   not_ok=int(((rows.scenario == sc) & (rows.cities == C) & (rows.n == n)
                               & (rows.arm == arm) & (rows.status != 'ok')).sum()),
                   f_mse=g.f_mse.mean(), f_mse_mcse=g.f_mse.std(ddof=1) / np.sqrt(B) if B > 1 else np.nan,
                   angle_bias=ang.mean(), angle_sd=sd,
                   angle_t=ang.mean() / (sd / np.sqrt(B)) if B > 1 and sd > 0 else np.nan,
                   seconds=g.seconds.mean())
        for k in STEP2:
            rec[k] = g[k].astype(float).mean()
        refarm = 'RNC_S' if arm.endswith('_S') else 'RNC'
        if arm == 'TRUE':
            ratio = se = np.nan
        else:
            ratio, se = paired_ratio(_col(ok, sc, C, n, arm, 'f_mse'), _col(ok, sc, C, n, refarm, 'f_mse'))
        rec.update(mse_ratio_RNC=ratio, mse_ratio_RNC_se=se)
        recs.append(rec)
    out = pd.DataFrame(recs)
    out['order'] = out.arm.map({a: i for i, a in enumerate(ORDER)})
    return out.sort_values(KEYS + ['order']).drop(columns='order').reset_index(drop=True)


def fmt(v, digits=3):
    if isinstance(v, str):
        return v
    if v is None or not np.isfinite(v):
        return '–'
    return f'{v:.2e}' if (abs(v) < 1e-2 and v != 0) or abs(v) >= 1e3 else f'{v:.{digits}f}'


def size_label(C, n):
    return f'{int(C)}×{int(n)}'


def table(s, cols):
    lines = []
    for (sc, C, n), g in s.groupby(KEYS, sort=False):
        lines += [f'### {sc} ({SCENARIO_TEXT.get(sc, "")}) — {size_label(C, n)}', '',
                  '| arm | B | ' + ' | '.join(c for c, _ in cols) + ' |',
                  '|---|---:|' + '---:|' * len(cols)]
        for _, r in g.iterrows():
            cells = [fmt(r[k]) for _, k in cols]
            lines.append(f'| {r.arm} | {r.B}' + (f' [{r.not_ok}✗]' if r.not_ok else '')
                         + ' | ' + ' | '.join(cells) + ' |')
        lines.append('')
    return lines


# ---------------------------------------------------------------------------
# Decision sections
# ---------------------------------------------------------------------------
def _get(s, sc, C, n, arm, col):
    v = s[(s.scenario == sc) & (s.cities == C) & (s.n == n) & (s.arm == arm)][col]
    return float(v.iloc[0]) if len(v) else np.nan


def _cells(s):
    return s[KEYS].drop_duplicates().itertuples(index=False, name=None)


def _pair_table(s, rows, a, b, title_a, title_b):
    """Bias t, variance ratio (angle) and paired MSE ratio of arm a over arm b,
    by scenario x size x space."""
    ok = rows[rows.status == 'ok']
    out = ['| scenario | size | space | t(' + title_a + ') | t(' + title_b + ') | var ratio | MSE ratio ± se |',
           '|---|---|---|---:|---:|---:|---:|']
    for sc, C, n in _cells(s):
        for suffix, space in (('', 'full'), ('_S', 'S')):
            aa, bb = a + suffix, b + suffix
            if _get(s, sc, C, n, aa, 'B') != _get(s, sc, C, n, aa, 'B'):
                continue
            var_ratio = (_get(s, sc, C, n, aa, 'angle_sd') / _get(s, sc, C, n, bb, 'angle_sd')) ** 2
            ratio, se = paired_ratio(_col(ok, sc, C, n, aa, 'f_mse'), _col(ok, sc, C, n, bb, 'f_mse'))
            out.append(f'| {sc} | {size_label(C, n)} | {space} | '
                       f'{fmt(_get(s, sc, C, n, aa, "angle_t"), 2)} | {fmt(_get(s, sc, C, n, bb, "angle_t"), 2)} | '
                       f'{fmt(var_ratio, 2)} | {fmt(ratio, 2)} ± {fmt(se, 2)} |')
    return out


def decisions(s, rows):
    ok = rows[rows.status == 'ok']
    L = ['## Decision summaries', '',
         'Signed-angle t = mean / (SD/√B) over replications (bias test); var ratio = ratio of '
         'angle variances; MSE ratio = paired mean f-MSE ratio (delta-method SE). '
         'Ratios < 1 favour the first arm.', '']
    L += ['### (i) SURN vs RNC', ''] + _pair_table(s, rows, 'SURN', 'RNC', 'SURN', 'RNC') + ['']
    L += ['### (ii) SURH vs SURN', ''] + _pair_table(s, rows, 'SURH', 'SURN', 'SURH', 'SURN') + ['']
    # (iii) persistence cost
    L += ['### (iii) Persistence cost: f-MSE in A8 / f-MSE in A (paired by replication; same X and innovations)', '']
    arms_p = [a for a in ORDER if a != 'TRUE']
    sizes = sorted({(C, n) for sc, C, n in _cells(s) if sc in ('A', 'A8')})
    L += ['| size | ' + ' | '.join(arms_p) + ' |', '|---|' + '---:|' * len(arms_p)]
    summary = {}
    for C, n in sizes:
        cells = []
        for arm in arms_p:
            a8, a = _col(ok, 'A8', C, n, arm, 'f_mse'), _col(ok, 'A', C, n, arm, 'f_mse')
            r, se = paired_ratio(a8, a)
            summary[(C, n, arm)] = r
            cells.append(f'{fmt(r, 2)} ± {fmt(se, 2)}' if np.isfinite(r) else '–')
        L.append(f'| {size_label(C, n)} | ' + ' | '.join(cells) + ' |')
    L.append('')
    for C, n in sizes:
        for suffix in ('', '_S'):
            ordv = [summary.get((C, n, a + suffix), np.nan) for a in ('RNC', 'SURN', 'SURH')]
            ml = summary.get((C, n, 'ML' + suffix), np.nan)
            if np.all(np.isfinite(ordv)) and np.isfinite(ml):
                L.append(f'- {size_label(C, n)} {"restricted" if suffix else "full"}: mean ordinal '
                         f'(RNC, SURN, SURH) persistence ratio {np.mean(ordv):.2f} vs ML {ml:.2f}.')
    L.append('')
    # (iv) step-2 heads
    L += ['### (iv) Step 2: L1 vs L2 head (observed-W errors on the fixed test panel; ratio = L1 / L2, paired)', '',
          '| scenario | size | arm | L1 MAE | L2 MAE | MAE ratio ± se | L1 MSE | L2 MSE | MSE ratio ± se |',
          '|---|---|---|---:|---:|---:|---:|---:|---:|']
    wins = dict(mae=[0, 0], mse=[0, 0])
    for sc, C, n in _cells(s):
        for arm in ('TRUE', 'ML_S', 'RNC_S', 'SURN_S'):
            if np.isnan(_get(s, sc, C, n, arm, 'B')):
                continue
            q = {k: _col(ok, sc, C, n, arm, k) for k in ('l1_mae', 'l2_mae', 'l1_mse', 'l2_mse')}
            r_mae, se_mae = paired_ratio(q['l1_mae'], q['l2_mae'])
            r_mse, se_mse = paired_ratio(q['l1_mse'], q['l2_mse'])
            for key, r, se in (('mae', r_mae, se_mae), ('mse', r_mse, se_mse)):
                wins[key][0] += int(r + 2 * se < 1)
                wins[key][1] += int(r - 2 * se > 1)
            L.append(f'| {sc} | {size_label(C, n)} | {arm} | {fmt(q["l1_mae"].mean())} | '
                     f'{fmt(q["l2_mae"].mean())} | {fmt(r_mae)} ± {fmt(se_mae)} | '
                     f'{fmt(q["l1_mse"].mean())} | {fmt(q["l2_mse"].mean())} | {fmt(r_mse)} ± {fmt(se_mse)} |')
    L += ['', f'Rows where L1 is better by more than 2 se: MAE {wins["mae"][0]}, MSE {wins["mse"][0]}; '
              f'L2 better by more than 2 se: MAE {wins["mae"][1]}, MSE {wins["mse"][1]}.', '']
    return L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'run_r3'))
    args = ap.parse_args(argv)
    out = Path(args.out)
    rows = pd.read_csv(out / 'rows.csv')
    s = per_arm(rows)
    s.to_csv(out / 'summary.csv', index=False)
    cols = [('f-MSE', 'f_mse'), ('± MCSE', 'f_mse_mcse'), ('ratio to RNC', 'mse_ratio_RNC'),
            ('± se', 'mse_ratio_RNC_se'), ('angle bias', 'angle_bias'), ('angle sd', 'angle_sd'),
            ('bias t', 'angle_t'), ('Kendall τ', 'kendall'), ('L1 MAE', 'l1_mae'), ('L2 MAE', 'l2_mae'),
            ('L1 MSE', 'l1_mse'), ('L2 MSE', 'l2_mse')]
    md = ['# Round 3: ordinal-only scorers (SURN, SURH) and second-step heads', '',
          'Signed angle = atan2(θ̂·e2, θ̂·e1) in the H-plane (e1 = θ*). Ratio = paired f-MSE over RNC '
          '(RNC_S for restricted arms). Kendall τ, L1/L2 head MAE and MSE are out of sample on the fixed '
          'test panel (observed W; TRUE = θ* floor). ✗ = fits not ok (excluded from means).', '']
    md += decisions(s, rows) + ['## Per-arm tables', ''] + table(s, cols)
    (out / 'summary.md').write_text('\n'.join(md), encoding='utf-8')
    bad = rows[rows.status != 'ok']
    print(f'{len(rows)} rows, {len(bad)} not ok -> {out / "summary.md"}')
    if len(bad):
        print(bad.groupby(['scenario', 'arm', 'status']).size().to_string())
    if 'error' in rows and rows.error.notna().any():
        print(rows[rows.error.notna()].groupby(['arm', 'error']).size().to_string())


if __name__ == '__main__':
    main()
