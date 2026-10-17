"""Optional real-PyTorch serial/parallel and resume equivalence smoke test.

Run from the suite with: python verification/check_multicore.py
No existing outputs are read or modified. Requires the suite's dependencies.
"""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def records(path):
    with path.open(newline='') as f:
        return sorted(list(csv.DictReader(f)),key=lambda r:(int(r['rep']),r['model'],r['arm'],r.get('reg',''),r.get('restart','')))

def compare(a,b):
    assert len(a)==len(b),(len(a),len(b))
    for left,right in zip(a,b):
        assert left.keys()==right.keys()
        for key,value in left.items():
            if key in ('seconds','total_candidate_seconds'):continue
            other=right[key]
            try:
                x,y=float(value),float(other)
            except ValueError:
                assert value==other,(key,value,other)
            else:
                assert (math.isnan(x) and math.isnan(y)) or math.isclose(x,y,rel_tol=1e-6,abs_tol=1e-8),(key,x,y)

def main():
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
        env[key]='1'
    config=dict(reps=2,models=['linear','nn'],truths=['linear'],city_grid=[2],n=[16],
        hidden=[4],views=1,epochs=8,lbfgs_steps=4,audit_steps=2,restarts=1,
        ridge_grid=[0.,.1],threads=1,device='cpu',dtype='float64',test_per_city=16,test_pairs=128,
        experiment3=dict(losses=['common_logit'],arms=['none','temporal'],nuisance_cases=[[1.,1.]]))
    with tempfile.TemporaryDirectory(prefix='ordinal_multicore_check_') as td:
        temp=Path(td);p=temp/'config.json';p.write_text(json.dumps(config))
        def run(out,workers,resume=False):
            command=[sys.executable,str(ROOT/'run.py'),'--config',str(p),'--experiments','3',
                     '--out',str(out),'--workers',str(workers)]
            if resume:command.append('--resume')
            subprocess.run(command,cwd=ROOT,env=env,check=True)
        a,b=temp/'serial',temp/'parallel'
        run(a,1);run(b,2)
        for filename in ('selected_fits.csv','candidates.csv'):
            compare(records(a/filename),records(b/filename))
        before=records(b/'selected_fits.csv')
        run(b,3,True)
        compare(before,records(b/'selected_fits.csv'))
    print('PASS: actual CPU/float64 serial and parallel fit records agree; changed worker count resumes without duplicates.')

if __name__=='__main__':main()
