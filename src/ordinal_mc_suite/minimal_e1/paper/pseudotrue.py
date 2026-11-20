"""Pseudo-true (asymptotic-bias) directions of the unrestricted arms FE, MM, RNC, SUR.

Estimated once per DGP "law" on a large panel (default C = 512 cities x n = 500 tracts, all tracts used
for fitting) and evaluated on a fresh panel with the oracle head; cached to disk as JSON.
The cache key only contains the features that change the population criterion (map, noise law,
city-level coordinates, E4 case): link corruptions only affect the restricted arms, persistence and
spatial dependence leave the marginal pair distribution (hence the population criterion) unchanged.
python -m minimal_e1.paper.pseudotrue --experiments E1,E2,E4 --C 512 --n 500 --workers 9 --cores 16
  --workers: DGPs fitted in parallel (one process each; at most the number of distinct DGPs);
  --threads: numba/BLAS threads per worker (default cores // workers); --cores: total budget (default all).
Progress: one line per arm and per finished DGP (with times); finished DGPs are cached and skipped on rerun.
"""
import os
import sys

if __name__ == '__main__' or __name__.endswith('.pseudotrue'):    # thread env must precede numpy import
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
from . import dgp
from . import experiments as ex
from . import heads
from . import kernels as kn
from . import parallel as par

HERE = Path(__file__).resolve().parent
PT_ARMS = ('FE', 'MM', 'RNC', 'SUR')


def pt_key(spec):
    e4 = f"{spec.e4}clip" if spec.e4 else 'none'          # 'clip': bounded A1 (Nov 2026); old E4 caches are stale
    return f"map{spec.map}_{spec.noise}_nuis{int(spec.city_nuis)}_hc{int(spec.hc)}_e4{e4}"


def pt_path(spec, C, n, cache_dir=None):
    return Path(cache_dir or HERE / 'pseudotrue') / f"pt_{pt_key(spec)}_C{C}_n{n}.json"


def _log(tag, msg, verbose=True):
    if verbose:
        print(f'[{time.strftime("%H:%M:%S")}] {tag}: {msg}', flush=True)


def compute(spec, C=512, n=500, arms=PT_ARMS, max_pairs=None, eval_C=32, mem_gb=2.0, sur_starts=1, verbose=True):
    kn.MEM_GB = mem_gb
    tag = pt_key(spec)
    t0 = time.perf_counter()
    key = dgp.key_of('PT', pt_key(spec), C, n)
    panel = dgp.make_panel(spec, C, n, key)
    tr = np.tile(np.arange(n), (C, 1))
    ctx = arms_mod.Ctx(panel, tr, tr[:, :0], key, dict(policy=None, max_pairs=max_pairs, sur_extra_starts=sur_starts))
    ekey = dgp.key_of('PTE', pt_key(spec), eval_C, 500)
    ep = dgp.make_panel(spec, eval_C, 500, ekey)
    etr = np.tile(np.arange(500), (eval_C, 1))
    _log(tag, f'panel {C}x{n} built in {time.perf_counter() - t0:.0f}s; fitting {", ".join(arms)}', verbose)
    out = dict(spec=pt_key(spec), C=C, n=n, max_pairs=max_pairs, arms={})
    for a in arms:
        _log(tag, f'{a} started', verbose)
        res = arms_mod.get(ctx, a)
        m = heads.evaluate_theta(ep, etr, etr, res["theta"], with_spline=False)
        out['arms'][a] = dict(theta=[float(x) for x in res['theta']], secs=res['secs'], **m)
        _log(tag, f'{a} done in {res["secs"]:.0f}s (rmse_f {m["rmse_f"]:.4f})', verbose)
    out['seconds'] = time.perf_counter() - t0
    return out


def get_or_compute(spec, C=512, n=500, cache_dir=None, **kw):
    path = pt_path(spec, C, n, cache_dir)
    if path.exists():
        return json.loads(path.read_text())
    out = compute(spec, C, n, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


def _init_worker(threads, mem_gb):
    kn.MEM_GB = mem_gb
    kn.set_threads(threads)


def _job(args):
    spec, C, n, cache, kw = args
    t0 = time.perf_counter()
    out = get_or_compute(spec, C, n, cache, **kw)
    return out, time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiments', default='E1')
    ap.add_argument('--scenarios', default=None)
    ap.add_argument('--C', type=int, default=512)
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--max-pairs-per-cell', type=int, default=None)
    ap.add_argument('--mem-gb', type=float, default=2.0)
    ap.add_argument('--cache', default=None)
    ap.add_argument('--workers', type=int, default=1, help='DGPs fitted in parallel (processes)')
    ap.add_argument('--threads', type=int, default=None, help='numba/BLAS threads per worker (default cores // workers)')
    ap.add_argument('--cores', type=int, default=None, help='total core budget (default all cores)')
    a = ap.parse_args(argv)
    jobs, seen = [], set()
    for e in a.experiments.upper().split(','):
        for name, spec in ex.SCENARIOS[e].items():
            if a.scenarios and name not in a.scenarios.split(','):
                continue
            if pt_key(spec) in seen:
                continue
            seen.add(pt_key(spec))
            jobs.append((name, spec))
    todo = [(nm, sp) for nm, sp in jobs if not pt_path(sp, a.C, a.n, a.cache).exists()]
    workers, threads, cores = par.plan(min(a.workers, max(len(todo), 1)), a.threads, a.cores)
    print(f'{len(jobs)} distinct DGPs, {len(todo)} to compute ({len(jobs) - len(todo)} cached); '
          f'workers={workers} x threads={threads} (cores budget {cores} of {os.cpu_count()})', flush=True)
    kw = dict(max_pairs=a.max_pairs_per_cell, mem_gb=a.mem_gb)
    t0 = time.time()

    def report(name, out, sec, i):
        print(f'== [{i}/{len(todo)}] {name} {out["spec"]} finished in {sec / 60:.1f} min (elapsed {(time.time() - t0) / 60:.1f} min): '
              + str({k: round(v['rmse_f'], 4) for k, v in out['arms'].items()}), flush=True)

    if workers == 1:
        _init_worker(threads, a.mem_gb)
        for i, (name, spec) in enumerate(todo, 1):
            out, sec = _job((spec, a.C, a.n, a.cache, kw))
            report(name, out, sec, i)
        return
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'), initializer=_init_worker,
                             initargs=(threads, a.mem_gb)) as pool:
        futs = {pool.submit(_job, (spec, a.C, a.n, a.cache, kw)): name for name, spec in todo}
        for i, f in enumerate(as_completed(futs), 1):
            name = futs[f]
            try:
                out, sec = f.result()
                report(name, out, sec, i)
            except Exception as e:                                  # pragma: no cover
                print(f'== [{i}/{len(todo)}] {name} FAILED {type(e).__name__}: {e}', flush=True)


if __name__ == '__main__':
    main()
