"""Add only parametric ML to the already saved city-calibration experiments."""
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('rho_saved',ROOT/'rho_city_monte_carlo.py')
rho=importlib.util.module_from_spec(spec); spec.loader.exec_module(rho)


def read(path):
    import pandas as pd
    return pd.read_csv(path).to_dict('records')


def uniform(rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    return [{k:row.get(k) for k in keys} for row in rows]


def append(folder):
    original=folder/'metrics.csv'; before=hashlib.sha256(original.read_bytes()).hexdigest()
    metrics=read(original); cities=read(folder/'city_metrics.csv'); histories=read(folder/'histories.csv')
    a=SimpleNamespace(**json.loads((folder/'config.json').read_text()))
    a.out=folder/'ml_append'; a.out.mkdir(exist_ok=True)
    matched=sorted({(int(r['rep']),float(r['log_rho_sd'])) for r in metrics})
    added=[]; success=[]; max_truth_replay_error=0.
    for rep,sd in matched:
        d=rho.make_design(a,rep,sd)
        oldfolder=folder/f'rep_{rep:02d}_sd_{sd:g}'
        with np.load(oldfolder/'dgp.npz') as saved:
            for key in ('reference','rho','sizes','centers','mu','sigma','f_mean','f_sd'):
                # Float32 cubic arithmetic differs slightly across NumPy/BLAS
                # builds. Random draws and nuisance parameters must match exactly.
                tol=1e-7 if key in ('f_mean','f_sd') else 1e-12
                np.testing.assert_allclose(d[key],saved[key],rtol=tol,atol=tol,err_msg=f'{folder.name}: replay mismatch in {key}')
        available=list(oldfolder.glob('*_predictions.npz'))
        if available:
            with np.load(available[0]) as saved:
                for actual,expected in ((d['test']['f'],saved['f_true']),
                                        (d['changed_f']-d['test']['f'],saved['delta_f_true'])):
                    np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-7)
                    max_truth_replay_error=max(max_truth_replay_error,float(np.max(np.abs(actual-expected))))
        target=a.out/f'rep_{rep:02d}_sd_{sd:g}'; target.mkdir(exist_ok=True)
        m,c,h=rho.fit_one(a,d,'ml_correct',None,rep,sd,torch.device('cpu'),target)
        added.append(m); cities.extend(c); histories.extend(h); success.append(m['ml_success'])
        print(f'{folder.name}: appended ML rep={rep}, sd={sd:g}, accuracy={m["latent_pair_accuracy"]:.4f}',flush=True)
    if hashlib.sha256(original.read_bytes()).hexdigest()!=before: raise RuntimeError('Original metrics changed')
    rho.summarize(a,uniform(metrics+added),uniform(cities),uniform(histories))
    rho.summary_plot(a,metrics+added,cities,histories)
    for name in ('metrics','city_metrics','histories','summary','paired_differences'):
        (folder/(name+'_with_ml.csv')).write_bytes((a.out/(name+'.csv')).read_bytes())
    (folder/'experiment_summary_with_ml.png').write_bytes((a.out/'experiment_summary.png').read_bytes())
    manifest=dict(original_metrics_sha256=before,original_fits=len(metrics),added_ml_fits=len(added),
                  ml_not_converged=sum(not x for x in success),original_estimators_rerun=0,
                  sample_replay_verified=True,max_truth_replay_error=max_truth_replay_error,
                  truth_replay_atol=1e-7,truth_replay_rtol=1e-7)
    (folder/'ml_append_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    torch.set_num_threads(1)
    results={p.parent.name:append(p.parent) for p in sorted(ROOT.glob('rho_*/metrics.csv'))}
    (ROOT/'ordinal_mc_suite/rho_ml_append_summary.json').write_text(json.dumps(results,indent=2))
