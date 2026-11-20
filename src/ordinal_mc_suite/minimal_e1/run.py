"""Run the minimal loss-ablation Monte Carlo.

From src/ordinal_mc_suite:
    python -m minimal_e1.run --stage all --reps 5 --workers 4
Stages: 1 = map a, 2 = map b, 3 = map c (small cells); 4 = anchor (all maps).
Results are appended per dataset to <out>/rows.csv, so an interrupted run
resumes where it stopped. Then: python -m minimal_e1.summarize --out <out>
"""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent

FIELDS = ['block', 'rho', 'cities', 'n', 'rep', 'map', 'arm', 'status', 'error', 'seconds',
          'objective', 'start_spread', 'n_starts', 'grad_norm', 'iterations', 'message',
          'link_rank', 'free_dim', 'restriction_gap_true', 'ranks_shared', 'pair_ties',
          'cos_true', 'b_source', 'f_mse', 'df_mse', 'df_mse_changed', 'ds_rms_stable',
          'mu_oracle_mse', 'dmu_oracle_mse', 'mu_cal_mse', 'dmu_cal_mse',
          'mu_native_mse', 'dmu_native_mse', 'theta', 'extras']


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', default='all', help="'all' or comma list of 1,2,3,4")
    ap.add_argument('--reps', type=int, default=5, help='replications per small cell')
    ap.add_argument('--anchor-reps', type=int, default=None, help='replications at the anchor (default = --reps)')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--out', default=str(HERE / 'run_5rep'))
    ap.add_argument('--smoke', action='store_true', help='tiny sizes for a functional check (minutes)')
    return ap.parse_args(argv)


def build_tasks(cfg, stages, reps, anchor_reps):
    from .config import STAGES
    merged = {}
    for stage in stages:
        spec = STAGES[stage]
        if spec['sizes'] == 'small':
            block, sizes, nrep = 0, cfg['small_sizes'], reps
        else:
            block, sizes, nrep = 1, [cfg['anchor_size']], anchor_reps
        for rho in cfg['rhos']:
            for cities, n in sizes:
                for rep in range(1, nrep + 1):
                    key = (block, float(rho), int(cities), int(n), rep)
                    merged.setdefault(key, set()).update(spec['maps'])
    tasks = [dict(block=k[0], rho=k[1], cities=k[2], n=k[3], rep=k[4],
                  maps=[m for m in cfg['maps'] if m in maps]) for k, maps in merged.items()]
    # Large tasks first so parallel workers finish together.
    return sorted(tasks, key=lambda t: -t['cities'] * t['n'] * len(t['maps']))


def done_keys(path):
    if not path.exists():
        return set()
    with path.open(newline='') as fh:
        return {(int(r['block']), float(r['rho']), int(r['cities']), int(r['n']), int(r['rep']), r['map'])
                for r in csv.DictReader(fh)}


_TARGET = None


def _init_worker(cfg):
    global _TARGET
    from .dgp import target_environment
    _TARGET = target_environment(cfg)


def _work(args):
    cfg, task = args
    from .experiment import run_dataset
    return task, run_dataset(cfg, _TARGET, task)


def main(argv=None):
    args = parse(argv)
    if args.workers > 1:
        for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            os.environ.setdefault(var, '1')
    import numpy as np
    import scipy
    from .config import make_config
    from .dgp import reference, target_environment

    if args.smoke:
        cfg = make_config(small_sizes=[[4, 10], [6, 16]], anchor_size=[8, 40],
                          test_n=200, reference_size=40000)
    else:
        cfg = make_config()
    stages = ['1', '2', '3', '4'] if args.stage == 'all' else args.stage.split(',')
    anchor_reps = args.anchor_reps if args.anchor_reps is not None else args.reps
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / 'rows.csv'
    done = done_keys(rows_path)
    tasks = []
    for task in build_tasks(cfg, stages, args.reps, anchor_reps):
        todo = [m for m in task['maps'] if (task['block'], task['rho'], task['cities'], task['n'],
                                            task['rep'], m) not in done]
        if todo:
            tasks.append(dict(task, maps=todo))
    ref = reference(cfg)
    target = target_environment(cfg)
    manifest_path = out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(runs=[])
    if manifest.get('config') not in (None, cfg):
        sys.exit(f'{out} holds results from a different configuration; use another --out')
    manifest.update(config=cfg, theta_star=ref.theta_star.tolist())
    run_info = dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), argv=sys.argv[1:], stages=stages,
                    reps=args.reps, anchor_reps=anchor_reps, workers=args.workers,
                    tasks=len(tasks), python=platform.python_version(),
                    numpy=np.__version__, scipy=scipy.__version__, host=platform.node())
    manifest['runs'].append(run_info)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    np.savez(out / 'target_moments.npz', theta_star=ref.theta_star,
             sigma_level=target.sigma_level, sigma_change=target.sigma_change)
    print(f'{len(tasks)} datasets to run ({len(done)} map-datasets already in {rows_path})', flush=True)

    new_file = not rows_path.exists()
    t_start = time.perf_counter()
    failures = 0
    with rows_path.open('a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        if new_file:
            writer.writeheader()

        def record(task, result, k):
            nonlocal failures
            rows, seconds = result
            for row in rows:
                writer.writerow(row)
            fh.flush()
            bad = sum(r['status'] != 'ok' for r in rows)
            failures += bad
            print(f"[{k}/{len(tasks)}] block={task['block']} rho={task['rho']} C={task['cities']} "
                  f"n={task['n']} rep={task['rep']} maps={''.join(task['maps'])}: {seconds:.1f}s"
                  + (f'  ({bad} arm(s) not ok)' if bad else ''), flush=True)

        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
                for k, (task, result) in enumerate(pool.map(_work, [(cfg, t) for t in tasks]), 1):
                    record(task, result, k)
        else:
            from .experiment import run_dataset
            for k, task in enumerate(tasks, 1):
                record(task, run_dataset(cfg, target, task), k)

    run_info.update(finished=time.strftime('%Y-%m-%d %H:%M:%S'),
                    wall_seconds=round(time.perf_counter() - t_start, 1), arms_not_ok=failures)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'Done in {run_info["wall_seconds"]}s; {failures} arm fits not ok. Rows: {rows_path}')
    print(f'Summarise with: python -m minimal_e1.summarize --out {out}')


if __name__ == '__main__':
    main()
