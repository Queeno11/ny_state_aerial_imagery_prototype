"""Append ML to finished experiments without rerunning any existing estimator.

Replay the saved config/seed and audit sample diagnostics before fitting ML.
Original CSVs and SQLite checkpoints are read only. Augmented exports use the
suffix _with_ml, and presentation_figures automatically reads these exports.

    python append_ml_benchmarks.py              # all finished suite runs
    python append_ml_benchmarks.py overnight_measurement overnight_nn
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ordinal_mc.config import preset, merge
from ordinal_mc.core import misranking
from ordinal_mc.design import PopulationDGP, make_design
from ordinal_mc.experiments import scenario
from ordinal_mc.likelihood import fit_ml
from ordinal_mc.report import SCENARIO, export

ROOT=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(value):
    if value=='': return None
    if value in ('True','False'): return value=='True'
    try: return json.loads(value)
    except (ValueError,TypeError): return value


def load_rows(path):
    with path.open(newline='',encoding='utf-8-sig') as stream:
        return [{k:decode(v) for k,v in row.items()} for row in csv.DictReader(stream)]


def existing_metrics(predict,design):
    """Exactly the original ranking/spacing metrics; no new experiment is run."""
    result={}; ii,jj=design['pairs']; train=predict(design['train']['x'])
    sd=np.sqrt(.5*np.sum(design['pair_weights']*(train[jj]-train[ii])**2))
    for where,(data,pairs) in design['tests'].items():
        score=predict(data['x'])
        for kind,(i,j) in pairs.items():
            err=misranking(score[j]-score[i],data['f'][j]-data['f'][i])
            result[f'{where}_{kind}_latent']=err
            result[f'{where}_{kind}_accuracy']=1-err
            if kind=='within':
                result[f'{where}_within_observed']=misranking(score[j]-score[i],data['w'][j]-data['w'][i])
                result[f'{where}_oracle_observed']=misranking(data['f'][j]-data['f'][i],data['w'][j]-data['w'][i])
                if sd>0:
                    result[f'{where}_within_difference_mse']=float(np.mean(((score[j]-score[i])/sd-(data['f'][j]-data['f'][i]))**2))
    return result


def audit_replay(rows,design):
    actual=dict(n_fit=design['n_fit'],n_validation=design['n_val'],n_rank_pairs=len(design['pairs'][0]),
                noise_sd_effective=design['noise_sd_effective'],
                calibrated_baseline_oracle_risk=design['baseline_oracle_risk'])
    for where,(data,pairs) in design['tests'].items():
        i,j=pairs['within']
        actual[where+'_oracle_observed']=misranking(data['f'][j]-data['f'][i],data['w'][j]-data['w'][i])
    checked=0
    for row in rows:
        values={**actual,**{'offered_'+k:v for k,v in design['link_stats'][row['arm']].items()}}
        for key,value in values.items():
            if row.get(key) is not None:
                if not np.isclose(row[key],value,atol=1e-11,rtol=1e-10):
                    raise ValueError(f'Replayed sample differs for {key}: saved={row[key]}, replay={value}; refusing to append ML')
                checked+=1
    return checked


def append_run(folder,cache=None):
    cache={} if cache is None else cache
    original=folder/'selected_fits.csv'; original_hash=sha(original)
    rows=load_rows(original)
    if not rows: return None
    old=[r for r in rows if r['method']!='ml_correct']
    d=preset('smoke'); merge(d,json.loads((folder/'resolved_config.json').read_text()))
    # Data generation is NumPy float64 for both CPU and CUDA scorer runs.
    # ML is SciPy/CPU and does not need the original scorer's CUDA runtime.
    cfg=SimpleNamespace(**d); dgp=PopulationDGP(cfg)
    groups={}
    for row in old:
        s=scenario(**{k:row[k] for k in SCENARIO if row.get(k) is not None})
        for k,v in s.items(): row.setdefault(k,v)
        # A reference appears once per fitted-class/loss comparison panel.
        loss=row['loss'] if row['experiment'] in (3,4,5) else 'common_logit'
        key=json.dumps([row[k] for k in ('experiment','rep','n','cities','truth','model')]+[loss,s],sort_keys=True)
        groups.setdefault(key,[]).append(row)
    ml=[]; checks=0
    for ordinal,rr in enumerate(groups.values(),1):
        r=rr[0]; s={k:r[k] for k in SCENARIO}
        local=copy.copy(cfg); local.cities=r['cities']
        local.acquisition_strength=s['acquisition_strength']; local.group_strength=s['group_strength']
        # Optimizer settings and fitted scorer class do not affect the ML data.
        data_keys=('seed','function_seed','views','noise_sd','bayes_target','validation_fraction',
                   'pair_mode','pairs_per_label','test_per_city','test_pairs','false_jump')
        cache_key=json.dumps([{k:d[k] for k in data_keys},r['rep'],r['n'],r['cities'],r['truth'],s],sort_keys=True)
        if cache_key not in cache:
            design=make_design(local,dgp,r['rep'],r['n'],r['truth'],s)
            # Check all existing arms, not just the method chosen as reference.
            count=audit_replay(rr,design)
            predict,info=fit_ml(design,r['truth'],s,dgp)
            metrics=existing_metrics(predict,design)
            cache[cache_key]=(metrics,info,{k:design[k] for k in ('n_fit','n_val','noise_sd_effective','baseline_oracle_risk')})
            checks+=count
        else:
            metrics,info,counts=cache[cache_key]
            # Every cached result is tied to the complete data-generating config.
            for row in rr:
                for key in ('seen_oracle_observed','unseen_oracle_observed'):
                    if row.get(key) is not None and row[key]!=metrics[key]:
                        raise ValueError(f'Cached replay differs for {key}')
                    checks+=1
        metrics,info,counts=cache[cache_key]
        base={k:r[k] for k in ('experiment','rep','n','cities','truth','model')}
        loss=r['loss'] if r['experiment'] in (3,4,5) else 'common_logit'
        baseline='rank_common' if r['experiment']==1 else r['baseline'] if r['experiment']==2 else 'none'
        ml.append({**base,**s,**info,**metrics,'method':'ml_correct','loss':loss,'arm':'none',
                   'baseline':baseline,'reg':0.,'lambda_t':0.,'lambda_c':0.,'restart':0,
                   'n_fit':counts['n_fit'],'n_validation':counts['n_val'],
                   'noise_sd_effective':counts['noise_sd_effective'],
                   'first_order_small':None,'audit_stable':None,
                   'optimizer_status':'ok' if info['ml_success'] else 'ml_not_converged',
                   'benchmark_origin':'appended to saved experiment; original estimators not rerun'})
        if ordinal%50==0: print(f'{folder.name}: ML {ordinal}/{len(groups)}',flush=True)
    if sha(original)!=original_hash: raise RuntimeError('Source results changed while appending ML')
    # Export through the usual summary/paired formulas into a separate directory.
    out=folder/'ml_append'; out.mkdir(exist_ok=True)
    conn=sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE jobs (result TEXT)'); conn.execute('CREATE TABLE candidates (data TEXT)')
    for row in old+ml:
        conn.execute('INSERT INTO jobs VALUES (?)',(json.dumps(row,allow_nan=False),))
    export(conn,out,plots=False); conn.close()
    for name in ('selected_fits','summary','paired_replications','paired_summary'):
        (folder/(name+'_with_ml.csv')).write_bytes((out/(name+'.csv')).read_bytes())
    manifest=dict(original_csv_sha256=original_hash,original_fits=len(old),added_ml_fits=len(ml),
                  replay_checks=checks,ml_not_converged=sum(not r['ml_success'] for r in ml),
                  original_estimators_rerun=0,original_checkpoint_modified=False,
                  ml_information=ml[0]['ml_information'])
    (folder/'ml_append_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'{folder.name}: kept {len(old)}, added {len(ml)} ML; replay checks={checks}; failures={manifest["ml_not_converged"]}',flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='*'); args=p.parse_args()
    folders=[ROOT/name for name in args.runs] if args.runs else sorted(p.parent for p in ROOT.glob('*/selected_fits.csv') if (p.parent/'resolved_config.json').exists())
    summary_path=ROOT/'ml_append_summary.json'
    cache={}; results=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for folder in folders: results[folder.name]=append_run(folder,cache)
    summary_path.write_text(json.dumps(results,indent=2))


if __name__=='__main__': main()
