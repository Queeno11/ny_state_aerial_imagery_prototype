"""Four single-step probes at one fixed checkpoint; no fitting/model selection."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import diagnose_e6_gradients as diag


def main(a):
    folder=a.run/'replications'/'r001_c32_quadratic_nn';saved=torch.load(folder/'checkpoints.pt',map_location='cpu',weights_only=False)
    d=saved['config'];job=saved['job'];cfg=diag.e5_runtime(d,job['cities'])
    base,variants=diag.build(cfg,diag.PopulationDGP(cfg),job['rep'],job['truth'],d)
    features=diag.core.Features(base['train']['x'],job['model'],cfg)
    data=diag.core.prepare(base,features,cfg);edges={k:diag.core.tensor(base['links'][k],cfg,True) for k in ('temporal','cross_city')}
    model=diag.core.Estimator(data['x'].shape[1],job['model'],'common_logit',cfg).double();model.load_state_dict(saved['models']['A']['scorer'])
    design=variants['nonlinear'];levels=diag.tr.level_data(design['train'],cfg);v=diag.tr.validation_data(design['val'],features,cfg)
    head=diag.CalibrationHead(diag.core.predictions(model,features,design['train']['x'],cfg),design['train']['cell'],design['train']['w'],'affine',d['head_training'])
    head.load_state_dict(saved['models']['nonlinear_affine_lambda10']['B_head'])
    rows=[]
    for scale in [1.,.1,.01,.001]:
        m=copy.deepcopy(model);h=copy.deepcopy(head);params=list(m.scorer.parameters())+list(h.parameters())
        opt=torch.optim.Adam([dict(params=m.scorer.parameters(),lr=d['joint']['scorer_lr']*scale),dict(params=h.parameters(),lr=d['joint']['head_lr']*scale)])
        z=diag.components(m,h,data,edges,levels,v,cfg,d,10.)
        validation=z['val_rank']+z['val_level'];objective=z['rank']+z['stability']+z['complexity']+z['level']+z['head_penalty']
        gv=diag.vector(validation,params);gq=diag.vector(objective,params)
        before=[p.detach().clone() for p in params];v0=float(validation.detach());q0=float(objective.detach())
        opt.zero_grad(set_to_none=True);objective.backward();torch.nn.utils.clip_grad_norm_(params,20.);opt.step()
        delta=torch.cat([(p.detach()-old).reshape(-1) for p,old in zip(params,before)])
        with torch.no_grad():
            new=diag.components(m,h,data,edges,levels,v,cfg,d,10.)
            v1=float(new['val_rank']+new['val_level']);q1=float(new['rank']+new['stability']+new['complexity']+new['level']+new['head_penalty'])
        rows.append(dict(lr_multiplier=scale,scorer_lr=d['joint']['scorer_lr']*scale,head_lr=d['joint']['head_lr']*scale,
            predicted_validation_change=float(gv.dot(delta)),actual_validation_change=v1-v0,
            predicted_objective_change=float(gq.dot(delta)),actual_objective_change=q1-q0,parameter_step_norm=float(delta.norm())))
    out=a.run.parent/'e6_gradient_diagnostic'/'step_size_probe.csv'
    if out.exists(): raise ValueError('Probe output already exists')
    pd.DataFrame(rows).to_csv(out,index=False);print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1): main(a)
