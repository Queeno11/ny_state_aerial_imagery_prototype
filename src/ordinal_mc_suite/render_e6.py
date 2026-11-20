"""Rebuild E6 reporting in a new folder from completed SQLite jobs, without fitting.

This entry point needs NumPy, SciPy, pandas and matplotlib, but not PyTorch or
the original fitting runtime. Original manifests, fits and results stay intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
from ordinal_mc.e6.report import export


def render_saved(run,out):
    run=Path(run).resolve();out=Path(out).resolve();db=run/'results.sqlite'
    if not db.is_file(): raise ValueError('No completed E6 SQLite ledger at '+str(db))
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('experiment')!='E6': raise ValueError('Expected an E6 manifest')
    if out==run or (out.exists() and any(out.iterdir())): raise ValueError('Choose a new, empty review output directory')
    before=hashlib.sha256(db.read_bytes()).hexdigest()
    con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)
    try: results=[json.loads(r[0]) for r in con.execute('SELECT payload FROM results ORDER BY job')]
    finally: con.close()
    if not results: raise ValueError('No completed E6 jobs to report')
    d=manifest['config'];expected=d['replications']*len(d['cities'])*len(d['cases'])
    out.mkdir(parents=True,exist_ok=True);export(results,out,d,expected)
    after=hashlib.sha256(db.read_bytes()).hexdigest()
    if before!=after: raise RuntimeError('Fitting ledger changed during review; retry when fitting is idle')
    import ordinal_mc.e6.report as renderer
    provenance=dict(original_run=str(run),original_signature=manifest['signature'],ledger_sha256=before,
        original_fitting_results_modified=False,completed_jobs=len(results),expected_jobs=expected,
        renderer_sha256=hashlib.sha256(Path(renderer.__file__).read_bytes()).hexdigest(),
        python=platform.python_version())
    (out/'review_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    return provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();render_saved(args.run,args.out);print('Report: '+str(args.out.resolve()/'report.md'))


if __name__=='__main__': main()
