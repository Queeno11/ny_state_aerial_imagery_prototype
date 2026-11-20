"""Runner for the paper Monte Carlo: python -m minimal_e1.paper.run --experiment E1 --workers 16 --threads 1 --mem-gb 2

One JSON file per (experiment, scenario, size, rep) under <out>/<exp>/<scenario>/C<C>_n<n>/rep<r>.json;
existing files are skipped (resumable). Seeds are deterministic functions of (experiment, scenario, C, n, rep).
"""
import os
import sys

if __name__ == '__main__' or __name__.endswith('.run'):         # thread env must precede numpy import
    from minimal_e1.paper.parallel import thread_env_from_argv
    thread_env_from_argv(sys.argv)

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import platform
import resource
import time

from . import arms as arms_mod
from . import calibration as cal
from . import dgp
from . import experiments as ex
from . import kernels as kn
from . import parallel as par

HERE = Path(__file__).resolve().parent
_CFG = {}


def total_ram_gb():
    with open('/proc/meminfo') as fh:
        for line in fh:
            if line.startswith('MemTotal'):
                return int(line.split()[1]) / 1e6
    return 8.0


def cap_workers(workers, mem_gb):
    """workers x mem_gb must not exceed total RAM."""
    return max(1, min(int(workers), int(total_ram_gb() // max(mem_gb, 1e-9))))


def task_path(out, t):
    return Path(out) / t['exp'] / t['scenario'] / f"C{t['C']}_n{t['n']}" / f"rep{t['rep']:04d}.json"


def task_key(t):
    return dgp.key_of(t['exp'], t['scenario'], t['C'], t['n'], t['rep'])


def init_worker(cfg):
    _CFG.update(cfg)
    kn.MEM_GB = cfg['mem_gb']
    kn.set_threads(cfg.get('threads', 1))
    if cfg.get('policy_path'):
        _CFG['policy'] = cal.load_policy(cfg['policy_path'])
    else:
        _CFG['policy'] = cal.load_policy()


def run_task(t, out, cfg=None):
    """Run one replication (skips if the result file exists). Returns (path, status, seconds)."""
    cfg = cfg or _CFG
    path = task_path(out, t)
    if path.exists():
        return str(path), 'skipped', 0.0
    t0 = time.perf_counter()
    key = task_key(t)
    spec = t['spec']
    panel = dgp.make_panel(spec, t['C'], t['n'], key)
    tr, te = dgp.split_tracts(panel, key)
    t_dgp = time.perf_counter() - t0
    ctx = arms_mod.Ctx(panel, tr, te, key, dict(cfg))
    rows = arms_mod.run_arms(ctx, t['arms'])
    rec = dict(exp=t['exp'], scenario=t['scenario'], C=t['C'], n=t['n'], rep=t['rep'], N=t['C'] * t['n'],
               spec={k: getattr(spec, k) for k in spec.__dataclass_fields__}, rows=rows,
               meta=dict(seconds=time.perf_counter() - t0, seconds_dgp=t_dgp, backend=kn.BACKEND,
                         peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                         mem_gb=kn.MEM_GB, n_links=int(len(ctx.D)), q_hard=int(ctx.B_hard.shape[1]),
                         q_free=int(ctx.B_free.shape[1]), n_fake=int(panel.n_fake),
                         n_pairs_per_cell=int(ctx.cs.n_pairs() // ctx.cs.K), max_pairs=cfg.get('max_pairs'),
                         host=platform.node()))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, path)
    return str(path), 'done', time.perf_counter() - t0


def _job(args):
    t, out = args
    return run_task(t, out)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--experiment', default='E1', help='E1|E2|E3|E4|E5|all or comma list')
    ap.add_argument('--scenarios', default=None)
    ap.add_argument('--sizes', default=None, help='e.g. 8x150,64x500')
    ap.add_argument('--reps', type=int, default=None, help='cap/override repetitions per size')
    ap.add_argument('--workers', type=int, default=1, help='parallel processes (replications)')
    ap.add_argument('--threads', type=int, default=None, help='numba/BLAS threads per worker (default: cores // workers)')
    ap.add_argument('--cores', type=int, default=None, help='total core budget (default: all cores); workers x threads <= cores')
    ap.add_argument('--mem-gb', type=float, default=2.0, help='memory budget per worker (pair temporaries)')
    ap.add_argument('--max-pairs-per-cell', type=int, default=None, help='fixed pair subsample (fallback; default all)')
    ap.add_argument('--sur-extra-starts', type=int, default=3)
    ap.add_argument('--out', default=str(HERE / 'results'))
    ap.add_argument('--policy', default=None, help='calibration json (default paper/calibration.json)')
    ap.add_argument('--arms', default=None, help='restrict to these arms (comma list)')
    ap.add_argument('--exclude-arms', default=None, help='drop these arms (e.g. MM, to run it separately into another --out)')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--no-numba', action='store_true')
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    exps = ['E1', 'E2', 'E3', 'E4', 'E5'] if a.experiment.lower() == 'all' else a.experiment.upper().split(',')
    sizes = {tuple(int(v) for v in s.split('x')) for s in a.sizes.split(',')} if a.sizes else None
    scen = set(a.scenarios.split(',')) if a.scenarios else None
    tasks = ex.plan(exps, scen, sizes, a.reps, a.smoke)
    keep = set(a.arms.split(',')) if a.arms else None
    drop = set(a.exclude_arms.split(',')) if a.exclude_arms else set()
    for t in tasks:
        t['arms'] = [x for x in t['arms'] if (keep is None or x in keep) and x not in drop]
    tasks = [t for t in tasks if t['arms']]
    todo = [t for t in tasks if not task_path(a.out, t).exists()]
    if a.no_numba:
        kn.set_backend('numpy')
    workers = cap_workers(min(a.workers, max(len(todo), 1)), a.mem_gb)
    workers, threads, cores = par.plan(workers, a.threads, a.cores)
    print(f'{len(tasks)} tasks, {len(todo)} to run, backend={kn.BACKEND}, workers={workers} x threads={threads} '
          f'(cores budget {cores} of {os.cpu_count()}; requested workers {a.workers}; RAM {total_ram_gb():.1f} GB, '
          f'mem-gb {a.mem_gb})', flush=True)
    if a.list:
        for t in todo:
            print(t['exp'], t['scenario'], t['C'], t['n'], t['rep'], t['arms'])
        return
    cfg = dict(mem_gb=a.mem_gb, threads=threads, max_pairs=a.max_pairs_per_cell,
               sur_extra_starts=a.sur_extra_starts, policy_path=a.policy)
    todo.sort(key=lambda t: -t['C'] * t['n'])
    t0 = time.time()
    if workers == 1:
        init_worker(cfg)
        for i, t in enumerate(todo):
            p, s, sec = run_task(t, a.out)
            print(f'[{i + 1}/{len(todo)}] {t["exp"]} {t["scenario"]} {t["C"]}x{t["n"]} rep{t["rep"]} {s} {sec:.1f}s', flush=True)
        return
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=init_worker, initargs=(cfg,)) as pool:
        futs = {pool.submit(_job, (t, a.out)): t for t in todo}
        for i, f in enumerate(as_completed(futs)):
            t = futs[f]
            try:
                p, s, sec = f.result()
            except Exception as e:                                  # pragma: no cover
                s, sec = f'FAILED {type(e).__name__}: {e}', 0.0
            print(f'[{i + 1}/{len(todo)}] {t["exp"]} {t["scenario"]} {t["C"]}x{t["n"]} rep{t["rep"]} {s} {sec:.1f}s '
                  f'(elapsed {time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
