"""Re-evaluate finished replications with the current heads and metrics, WITHOUT refitting the first step.

Every replication is deterministic in its key, so the panel and the train/test split are regenerated exactly,
and the stored index direction theta of each arm is re-scored with the current heads (oracle, CV B-spline,
native FE) and metrics (RMSE vs E[W|x] and within-cell R^2 vs observed W and dW). Only FE/FE_S are refitted
(milliseconds; needed for their native predictions), plus the arms listed in --refit, whose stored theta is
known to be stale.

Default --refit: the penalty arms in E2 N1_nuis_both / N2_hc_both (the city-level profiling fix of 18 Nov 2026).

Safety check: for every arm that is not refitted, the oracle-head RMSE recomputed from the stored theta must
equal the stored value (relative difference <= --check-tol); otherwise the file is not written and the
replication is reported as a mismatch.

    python -m minimal_e1.paper.rehead --src minimal_e1/paper/results --out minimal_e1/paper/results_v2 \
        --experiments E1,E2,E3 --workers 16 --cores 16
Resumable: replications already present in --out are skipped.
"""
import os
import sys

if __name__ == '__main__' or __name__.endswith('.rehead'):          # thread env must precede numpy import
    from minimal_e1.paper.parallel import thread_env_from_argv
    thread_env_from_argv(sys.argv)

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import arms as arms_mod
from . import calibration as cal
from . import dgp, experiments as ex, heads
from . import kernels as kn
from . import parallel as par

HERE = Path(__file__).resolve().parent
DEFAULT_REFIT = 'E2:N1_nuis_both,N2_hc_both:RNC_P0.5,RNC_P0.9,RNC_P0.99,RNC_SEL'
NATIVE_ARMS = ('FE', 'FE_S', 'MM')
_CFG = {}


def parse_refit(text):
    """'EXP:scen1,scen2:arm1,arm2;EXP2:...' -> {(exp, scen): {arms}}."""
    out = {}
    for block in filter(None, (text or '').split(';')):
        e, scens, arms = block.split(':')
        for s in scens.split(','):
            out.setdefault((e, s), set()).update(arms.split(','))
    return out


def _cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rehead_record(rec, refit_arms, cfg, check_tol=1e-6):
    """Return (new_record, max_rel_diff, n_mismatch). Rows keep their fit info; metrics are replaced."""
    spec = ex.SCENARIOS[rec['exp']][rec['scenario']]
    stored = {k: v for k, v in rec['spec'].items()}
    now = {k: getattr(spec, k) for k in spec.__dataclass_fields__}
    if any(json.dumps(now.get(k)) != json.dumps(v) for k, v in stored.items()):
        raise ValueError(f"scenario spec changed since the run: {rec['exp']}/{rec['scenario']}")
    key = dgp.key_of(rec['exp'], rec['scenario'], rec['C'], rec['n'], rec['rep'])
    panel = dgp.make_panel(spec, rec['C'], rec['n'], key)
    tr, te = dgp.split_tracts(panel, key)
    ctx = arms_mod.Ctx(panel, tr, te, key, dict(cfg))
    max_rel, n_bad, rows = 0.0, 0, []
    for r in rec['rows']:
        r = dict(r)
        if r.get('status') != 'ok':
            rows.append(r)
            continue
        name = r['arm']
        native = None
        if name in refit_arms:
            res = arms_mod.get(ctx, name)
            r['theta'] = [float(x) for x in res['theta']]
            r['secs'] = res['secs']
            for k in ('lambda', 'omega', 'omega_sel', 'fallback', 'q', 'tau_median', 'nit', 'kendall_val'):
                if k in res:
                    r[k] = res[k]
            r['refit'] = True
            native = arms_mod.native_for(ctx, name, res)
        elif name in NATIVE_ARMS and name != 'MM':
            res = arms_mod.get(ctx, name)
            if _cos(res['theta'], r['theta']) < 1 - 1e-8:
                n_bad += 1
            native = arms_mod.native_for(ctx, name, res)
        theta = np.asarray(r['theta'], float)
        met = heads.evaluate_theta(panel, tr, te, theta, native)
        if not r.get('refit') and 'rmse_W_oracle' in r:
            old = r['rmse_W_oracle']
            rel = abs(met['rmse_W_oracle'] - old) / max(abs(old), 1e-12)
            max_rel = max(max_rel, rel)
            if rel > check_tol:
                n_bad += 1
        for k in [k for k in r if k.startswith(('rmse_', 'r2_'))]:
            if not k.endswith('_iso'):                      # keep the old isotonic numbers for reference
                r.pop(k)
        r.update(met)
        rows.append(r)
    out = dict(rec, rows=rows)
    out['meta'] = dict(rec['meta'], rehead=dict(max_rel_diff=max_rel, n_mismatch=n_bad,
                                                refit=sorted(refit_arms), time=time.strftime('%Y-%m-%d %H:%M')))
    return out, max_rel, n_bad


def init_worker(cfg):
    _CFG.update(cfg)
    kn.MEM_GB = cfg['mem_gb']
    kn.set_threads(cfg.get('threads', 1))
    _CFG['policy'] = cal.load_policy()


def job(args):
    src, dst, refit_map, check_tol = args
    t0 = time.perf_counter()
    rec = json.loads(Path(src).read_text())
    refit_arms = refit_map.get((rec['exp'], rec['scenario']), set())
    new, max_rel, n_bad = rehead_record(rec, refit_arms, _CFG, check_tol)
    if n_bad:
        return src, 'mismatch', max_rel, time.perf_counter() - t0
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix('.tmp')
    tmp.write_text(json.dumps(new))
    os.replace(tmp, dst)
    return src, 'done', max_rel, time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=str(HERE / 'results'))
    ap.add_argument('--out', default=str(HERE / 'results_v2'))
    ap.add_argument('--experiments', default='E1,E2,E3')
    ap.add_argument('--scenarios', default=None)
    ap.add_argument('--refit', default=DEFAULT_REFIT, help="'EXP:scen,..:arm,..;...' (empty string = none)")
    ap.add_argument('--check-tol', type=float, default=1e-6)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--cores', type=int, default=None)
    ap.add_argument('--mem-gb', type=float, default=2.0)
    ap.add_argument('--sur-extra-starts', type=int, default=3)
    ap.add_argument('--limit', type=int, default=None, help='only the first N pending files (testing)')
    a = ap.parse_args(argv)
    exps = set(a.experiments.upper().split(','))
    scen = set(a.scenarios.split(',')) if a.scenarios else None
    src, out = Path(a.src), Path(a.out)
    files = sorted(p for p in src.glob('E*/*/C*_n*/rep*.json')
                   if p.parts[-4] in exps and (scen is None or p.parts[-3] in scen))
    todo = [(p, out / p.relative_to(src)) for p in files if not (out / p.relative_to(src)).exists()]
    todo.sort(key=lambda x: -int(x[0].parts[-2].split('_n')[0][1:]) * int(x[0].parts[-2].split('_n')[1]))
    if a.limit:
        todo = todo[:a.limit]
    workers, threads, cores = par.plan(min(a.workers, max(len(todo), 1)), a.threads, a.cores)
    refit_map = parse_refit(a.refit)
    print(f'{len(files)} replications in {src}, {len(todo)} to re-head into {out}; '
          f'workers={workers} x threads={threads}; refit={dict((f"{k[0]}/{k[1]}", sorted(v)) for k, v in refit_map.items())}',
          flush=True)
    cfg = dict(mem_gb=a.mem_gb, threads=threads, max_pairs=None, sur_extra_starts=a.sur_extra_starts)
    args = [(str(s), str(d), refit_map, a.check_tol) for s, d in todo]
    t0, n_done, bad, worst = time.time(), 0, [], 0.0

    def report(res):
        nonlocal n_done, worst
        path, status, rel, sec = res
        n_done += 1
        worst = max(worst, rel)
        if status != 'done':
            bad.append(path)
        if status != 'done' or n_done % 100 == 0 or n_done == len(args):
            print(f'[{n_done}/{len(args)}] {status} {Path(path).relative_to(src)} ({sec:.1f}s; '
                  f'elapsed {(time.time() - t0) / 60:.1f} min; max rel diff so far {worst:.1e})', flush=True)

    if workers == 1:
        init_worker(cfg)
        for x in args:
            report(job(x))
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'), initializer=init_worker,
                                 initargs=(cfg,)) as pool:
            futs = [pool.submit(job, x) for x in args]
            for f in as_completed(futs):
                try:
                    report(f.result())
                except Exception as e:                               # pragma: no cover
                    n_done += 1
                    bad.append(repr(e))
                    print(f'[{n_done}/{len(args)}] FAILED {type(e).__name__}: {e}', flush=True)
    print(f'finished: {n_done - len(bad)} written, {len(bad)} mismatched/failed (not written); '
          f'max rel diff of the oracle-head check {worst:.1e}', flush=True)
    for b in bad[:20]:
        print('  not written:', b)


if __name__ == '__main__':
    main()
