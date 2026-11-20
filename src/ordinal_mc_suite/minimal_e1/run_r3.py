"""Round 3: ordinal-only scorers (SURN, SURH) and second-step heads (L1 vs L2).

From src/ordinal_mc_suite:
    python -m minimal_e1.run_r3 --reps-small 400 --reps-anchor 100 --workers 4
    python -m minimal_e1.summarize_r3
Scenarios A, A8, B-F (see robust_dgp.py) at (8 x 10) and (32 x 40) with all arms and at the
anchor (64 x 250) with stability arms only. Source blocks 4/5/6 (rounds 1-2 used 0, 2, 3).
Rows are appended per dataset to <out>/rows.csv; an interrupted run resumes where it stopped.
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
FIELDS = ['scenario', 'block', 'cities', 'n', 'rep', 'arm', 'map', 'status', 'error', 'seconds',
          'objective', 'start_spread', 'grad_norm', 'link_rank', 'contaminated', 'pair_ties',
          'cos_true', 'f_mse', 'df_mse', 'signed_angle', 'off_plane', 'a1_load',
          'kendall', 'l1_mae', 'l1_mse', 'l2_mae', 'l2_mse', 'theta', 'extras']
SIZE_BLOCKS = {'small1': 4, 'small2': 5, 'anchor': 6}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenarios', default='A,A8,B,C,D,E,F')
    ap.add_argument('--reps-small', type=int, default=400)
    ap.add_argument('--reps-anchor', type=int, default=100)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--out', default=str(HERE / 'run_r3'))
    ap.add_argument('--smoke', action='store_true', help='tiny sizes, functional check only')
    return ap.parse_args(argv)


def tasks_for(scenarios, small1, small2, anchor, reps_small, reps_anchor, test_n=500):
    extra = dict(cal_n=20, test_n=test_n)
    tasks = []
    for sc in scenarios:
        tasks += [dict(scenario=sc, block=SIZE_BLOCKS['anchor'], cities=anchor[0], n=anchor[1],
                       rep=r, s_only=True, **extra) for r in range(1, reps_anchor + 1)]
        for name, size in (('small1', small1), ('small2', small2)):
            tasks += [dict(scenario=sc, block=SIZE_BLOCKS[name], cities=size[0], n=size[1],
                           rep=r, s_only=False, **extra) for r in range(1, reps_small + 1)]
    return tasks


def done_keys(path):
    if not path.exists():
        return set()
    with path.open(newline='') as fh:
        return {(r['scenario'], int(r['block']), int(r['rep'])) for r in csv.DictReader(fh)}


_MOMENTS = None


def _moments(cfg):
    from .dgp import reference, target_environment
    from .robust_dgp import evaluation_frame
    target = target_environment(cfg, maps=('a',))
    return (reference(cfg).theta_star, target.sigma_level, target.sigma_change,
            evaluation_frame(cfg))


def _init(cfg):
    global _MOMENTS
    _MOMENTS = _moments(cfg)


def _work(args):
    cfg, task = args
    from .r3_experiment import run_dataset
    return task, run_dataset(cfg, _MOMENTS, task)


def main(argv=None):
    args = parse(argv)
    if args.workers > 1:
        for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            os.environ.setdefault(var, '1')
    import numpy as np
    import scipy
    from .config import make_config

    if args.smoke:
        cfg = make_config(test_n=200, reference_size=40000)
        small1, small2, anchor, test_n = (4, 8), (6, 12), (8, 30), 60
        args.reps_small, args.reps_anchor = min(args.reps_small, 2), min(args.reps_anchor, 1)
    else:
        cfg = make_config()
        small1, small2, anchor, test_n = (8, 10), (32, 40), (64, 250), 500
    scenarios = args.scenarios.split(',')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / 'rows.csv'
    done = done_keys(rows_path)
    tasks = [t for t in tasks_for(scenarios, small1, small2, anchor, args.reps_small, args.reps_anchor, test_n)
             if (t['scenario'], t['block'], t['rep']) not in done]
    manifest_path = out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(runs=[])
    design = dict(config=cfg, small1=list(small1), small2=list(small2), anchor=list(anchor),
                  test_n=test_n, cal_n=20)
    if manifest.get('design') not in (None, design):
        sys.exit(f'{out} holds results from a different design; use another --out')
    manifest['design'] = design
    run_info = dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), argv=sys.argv[1:],
                    scenarios=scenarios, reps_small=args.reps_small, reps_anchor=args.reps_anchor,
                    workers=args.workers, tasks=len(tasks), python=platform.python_version(),
                    numpy=np.__version__, scipy=scipy.__version__, host=platform.node())
    manifest['runs'].append(run_info)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'{len(tasks)} datasets to run ({len(done)} already in {rows_path})', flush=True)

    new_file = not rows_path.exists()
    t_start = time.perf_counter()
    failures = 0
    with rows_path.open('a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        if new_file:
            writer.writeheader()

        def record(k, task, result):
            nonlocal failures
            rows, sec = result
            for row in rows:
                writer.writerow(row)
            fh.flush()
            bad = sum(r['status'] != 'ok' for r in rows)
            failures += bad
            if k % 10 == 0 or bad or k == len(tasks):
                print(f"[{k}/{len(tasks)}] {task['scenario']} C={task['cities']} n={task['n']} "
                      f"rep={task['rep']}: {sec:.1f}s" + (f'  ({bad} not ok)' if bad else ''),
                      flush=True)

        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(cfg,)) as pool:
                for k, (task, result) in enumerate(pool.map(_work, [(cfg, t) for t in tasks],
                                                            chunksize=1), 1):
                    record(k, task, result)
        else:
            _init(cfg)
            from .r3_experiment import run_dataset
            for k, task in enumerate(tasks, 1):
                record(k, task, run_dataset(cfg, _MOMENTS, task))
    run_info.update(finished=time.strftime('%Y-%m-%d %H:%M:%S'),
                    wall_seconds=round(time.perf_counter() - t_start, 1), arms_not_ok=failures)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'Done in {run_info["wall_seconds"]}s; {failures} arm fits not ok.')
    print(f'Summarise with: python -m minimal_e1.summarize_r3 --out {out}')


if __name__ == '__main__':
    main()
