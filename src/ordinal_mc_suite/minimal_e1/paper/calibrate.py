"""Calibration pilot: distribution of lambda(omega) across replications (E1 map c, 8 x 150, 20 reps)
and its stability; writes paper/calibration.json.   python -m minimal_e1.paper.calibrate --reps 20
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import arms as arms_mod
from . import calibration as cal
from . import dgp
from . import experiments as ex
from . import fit
from . import stability as st

HERE = Path(__file__).resolve().parent


def one_rep(spec, C, n, rep, scen='E1_c'):
    key = dgp.key_of('CAL', scen, C, n, rep)
    p = dgp.make_panel(spec, C, n, key)
    tr, te = dgp.split_tracts(p, key)
    ctx = arms_mod.Ctx(p, tr, te, key, dict(policy=None))
    h = arms_mod.get(ctx, 'RNC_S')
    u = arms_mod.get(ctx, 'RNC')
    fe = arms_mod.get(ctx, 'FE')
    d = cal.lambda_diagnostics(ctx.cs, ctx.pen, h['theta'], h['M'], th_start=fe['theta'],
                               rng=np.random.default_rng(rep))
    trR_u, trS_u = st.curvature_traces(ctx.cs, ctx.pen, u['theta'], u['M'])
    d['lam_u_0.9'] = float(st.lambda_for_omega(0.9, trR_u, trS_u))
    d['P_unrestricted'] = ctx.pen.value(u['theta'])
    d['P_hard'] = ctx.pen.value(h['theta'])
    d['rep'] = rep
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--out', default=str(cal.CAL_PATH))
    ap.add_argument('--detail', default=str(HERE / 'results_pilot' / 'calibration_reps.json'))
    ap.add_argument('--check-sizes', default='8x50:6,8x500:4,64x150:2', help='extra (size:reps) stability checks')
    a = ap.parse_args(argv)
    spec = ex.SCENARIOS['E1']['E1_c']
    rows = [one_rep(spec, 8, 150, r) for r in range(a.reps)]
    policy = cal.summarise(rows)
    policy['design'] = dict(scenario='E1_c', C=8, n=150, reps=a.reps)
    checks = {}
    for item in filter(None, a.check_sizes.split(',')):
        sz, rr = item.split(':')
        C, n = (int(v) for v in sz.split('x'))
        rs = [one_rep(spec, C, n, r) for r in range(int(rr))]
        checks[sz] = {str(w): float(np.median([r[f'lam_{w}'] for r in rs])) for w in cal.OMEGAS}
    policy['size_check_median_lambda'] = checks
    Path(a.detail).parent.mkdir(parents=True, exist_ok=True)
    Path(a.detail).write_text(json.dumps(rows, indent=1))
    Path(a.out).write_text(json.dumps(policy, indent=1))
    print(json.dumps(policy, indent=1))


if __name__ == '__main__':
    main()
