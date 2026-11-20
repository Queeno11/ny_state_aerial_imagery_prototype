"""Separate resumable E6 entry point; main process owns the SQLite ledger."""
import argparse
import hashlib
import json
import platform
import sqlite3
from pathlib import Path
from .config import load
from .data import coefficient_ranges
from .experiment import compute_job, job_key, atomic_json
from ..execution import iter_results


def jobs(d):
    return [dict(rep=r,cities=c,truth=truth,model=model) for r in range(1,d['replications']+1)
            for c in d['cities'] for truth,model in d['cases']]


def signature(d):
    config={k:v for k,v in d.items() if k!='workers'}
    package=Path(__file__).resolve().parents[1]
    files=sorted(package.glob('*.py'))+sorted((package/'e6').glob('*.py'))+[package.parent/'run_e6.py']
    sources={str(p.relative_to(package.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    value=json.dumps(dict(config=config,sources=sources),sort_keys=True)
    return hashlib.sha256(value.encode()).hexdigest(),sources


def ledger(out,d,resume):
    fingerprint,sources=signature(d);manifest=out/'manifest.json'
    import numpy, scipy, torch
    versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__)
    if manifest.exists():
        if not resume: raise ValueError('Existing E6 output: use --resume or choose a new directory')
        old=json.loads(manifest.read_text(encoding='utf-8'))
        if old['signature']!=fingerprint: raise ValueError('Configuration/source differs from saved run; use a new output directory')
        if old['runtime']!=versions: raise ValueError('Numerical runtime differs from saved run; use the original environment or a new output directory')
    else:
        if any(out.iterdir()): raise ValueError('Refusing a nonempty directory without an E6 manifest')
        atomic_json(manifest,dict(experiment='E6',signature=fingerprint,config=d,sources=sources,
            coefficient_ranges=coefficient_ranges(d),runtime=versions,
            target='latent within-tract changes; outcome calibration is a separate target'))
        atomic_json(out/'resolved_config.json',d)
    con=sqlite3.connect(out/'results.sqlite');con.execute('CREATE TABLE IF NOT EXISTS results (job TEXT PRIMARY KEY, payload TEXT NOT NULL)');con.commit()
    return con


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int);p.add_argument('--resume',action='store_true')
    p.add_argument('--plan',action='store_true',help='Print exact design and coefficient ranges; no training or output writes')
    p.add_argument('--report-only',action='store_true',help='Rebuild exports from completed ledger entries')
    args=p.parse_args(argv);d=load(args.config)
    if args.workers is not None:
        if args.workers<1: p.error('--workers must be positive')
        d['workers']=args.workers
    all_jobs=jobs(d)
    if args.plan:
        print(json.dumps(dict(config=d,jobs=len(all_jobs),baseline_restarts_per_job=d['restarts'],
            joint_candidates_per_job=len(d['maps'])*len(d['heads'])*len(d['level_weights']),
            coefficient_ranges=coefficient_ranges(d)),indent=2));return
    out=args.out.resolve()
    # E6 must not be pointed at prior suite output, including an empty named E5 directory.
    if out.name.lower() in ('change_recovery','optimization_validation_stopping'): p.error('Use a new E6 output directory')
    if args.report_only and not (out/'manifest.json').exists(): p.error('No E6 manifest to report')
    out.mkdir(parents=True,exist_ok=True)
    con=ledger(out,d,args.resume or args.report_only)
    try:
        done={row[0] for row in con.execute('SELECT job FROM results')}
        pending=[(d,j,str(out)) for j in all_jobs if job_key(j) not in done]
        print(f'E6: {len(done)}/{len(all_jobs)} complete; {len(pending)} pending; workers={d["workers"]}',flush=True)
        if not args.report_only:
            iterator=iter_results(compute_job,pending,workers=d['workers'],threads=d['threads'])
            try:
                for task,result in iterator:
                    key=job_key(task[1]);con.execute('INSERT INTO results VALUES (?,?)',(key,json.dumps(result,allow_nan=False)));con.commit()
                    done.add(key);print(f'Completed {key}: {len(done)}/{len(all_jobs)}',flush=True)
            finally: iterator.close()
        results=[json.loads(row[0]) for row in con.execute('SELECT payload FROM results ORDER BY job')]
        if results:
            from .report import export
            export(results,out,d,len(all_jobs));print('Report: '+str(out/'report.md'),flush=True)
    finally: con.close()

