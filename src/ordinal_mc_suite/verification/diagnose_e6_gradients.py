"""One saved replication; component gradients and short same-start Adam paths.

Does not modify MC sources/results, refit A/B, or use latent/test targets.
"""
import argparse
import copy
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ordinal_mc import core
from ordinal_mc.design import PopulationDGP
from ordinal_mc.e6.config import e5_runtime
from ordinal_mc.e6.data import build
from ordinal_mc.e6.heads import CalibrationHead
from ordinal_mc.e6 import training as tr


def vector(loss,params):
    if not loss.requires_grad: return torch.cat([torch.zeros_like(p).reshape(-1) for p in params])
    grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g,p in zip(grads,params)])


def cosine(a,b):
    denom=float(a.norm()*b.norm())
    return float(a.dot(b))/denom if denom>0 else None


def components(model,head,data,edges,levels,val,cfg,d,weight):
    _,parts=core.score_objective(model,data,edges,(d['lambda_t'],0.),d['alpha'],cfg.tau)
    w=tr.level_loss(model,head,data['x'],levels)
    return dict(rank=parts[0],stability=parts[1],complexity=d['alpha']*parts[2],
        level=weight*w,level_unweighted=w,head_penalty=d['head_training']['kappa_G']*head.penalty(),
        val_rank=tr.rank_loss(model,val),
        val_level=(head(model.scorer(val['x']),val['cell'])-val['w']).square().mean()/levels['variance'])


def audit(model,head,data,edges,levels,val,cfg,d,weight,tag):
    z=components(model,head,data,edges,levels,val,cfg,d,weight)
    params=list(model.scorer.parameters());hp=list(head.parameters())
    g={name:vector(z[name],params) for name in ('rank','stability','complexity','level','level_unweighted','val_rank','val_level')}
    e5=g['rank']+g['stability']+g['complexity'];total=e5+g['level']
    gv=d['validation_rank_weight']*g['val_rank']+d['validation_level_weight']*g['val_level']
    hlevel=vector(z['level'],hp);hpenalty=vector(z['head_penalty'],hp)
    row=dict(tag,weight=weight,**{name+'_loss':float(value.detach()) for name,value in z.items()},
        **{name+'_grad_norm':float(value.norm()) for name,value in g.items()},
        e5_grad_norm=float(e5.norm()),joint_grad_norm=float(total.norm()),
        head_level_grad_norm=float(hlevel.norm()),head_penalty_grad_norm=float(hpenalty.norm()),
        level_to_e5_ratio=float(g['level'].norm()/e5.norm()) if e5.norm()>0 else None,
        cosine_level_rank=cosine(g['level'],g['rank']),cosine_level_stability=cosine(g['level'],g['stability']),
        cosine_level_e5=cosine(g['level'],e5),cosine_level_val=cosine(g['level'],gv),
        gd_validation_derivative=float(-gv.dot(total)))
    joint=z['rank']+z['stability']+z['complexity']+z['level']+z['head_penalty']
    direct=vector(joint,params)
    torch.testing.assert_close(direct,total,atol=1e-11,rtol=1e-9)
    norm=float(torch.sqrt(total.square().sum()+(hlevel+hpenalty).square().sum()))
    row['all_parameter_grad_norm']=norm;row['global_clip_multiplier']=min(1.,20/(norm+1e-6))
    row['gradient_sum_verified']=True
    return row


def run(args):
    run=args.run.resolve();folder=run/'replications'/args.job;out=args.out.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Choose a new output directory')
    manifest=json.loads((run/'manifest.json').read_text());d=manifest['config']
    versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__)
    for key,value in versions.items():
        if str(manifest['runtime'][key])!=str(value): raise ValueError('Runtime mismatch: '+key)
    path=folder/'checkpoints.pt';before=hashlib.sha256(path.read_bytes()).hexdigest()
    saved=torch.load(path,map_location='cpu',weights_only=False);job=saved['job']
    cfg=e5_runtime(d,job['cities']);base,variants=build(cfg,PopulationDGP(cfg),job['rep'],job['truth'],d)
    features=core.Features(base['train']['x'],job['model'],cfg)
    for name,attr in [('feature_mu','mu'),('feature_sd','sd'),('feature_bmu','bmu'),('feature_bsd','bsd')]:
        np.testing.assert_allclose(saved[name],getattr(features,attr),atol=1e-12,rtol=1e-12)
    data=core.prepare(base,features,cfg);edges={k:core.tensor(base['links'][k],cfg,True) for k in ('temporal','cross_city')}
    A=core.Estimator(data['x'].shape[1],job['model'],'common_logit',cfg).double();A.load_state_dict(saved['models']['A']['scorer'])
    val=tr.validation_data(base['val'],features,cfg)
    np.testing.assert_allclose(float(tr.rank_loss(A,val).detach()),saved['models']['A']['info']['validation'],atol=1e-11,rtol=1e-11)
    rows=[]
    def load_head(kind,family,state):
        design=variants[kind]
        h=CalibrationHead(core.predictions(A,features,design['train']['x'],cfg),design['train']['cell'],
                          design['train']['w'],family,d['head_training']);h.load_state_dict(state);return h
    for kind in d['maps']:
        design=variants[kind];levels=tr.level_data(design['train'],cfg);v=tr.validation_data(design['val'],features,cfg)
        for family in d['heads']:
            for weight in d['level_weights']:
                state=saved['models'][f'{kind}_{family}_lambda{weight:g}']
                h=load_head(kind,family,state['B_head'])
                np.testing.assert_allclose(tr.observable_rule(A,h,v,levels['variance'],d),state['B_info']['validation'],atol=1e-11)
                rows.append(audit(A,h,data,edges,levels,v,cfg,d,weight,dict(map=kind,head=family,checkpoint='B_same_lambda',step=0)))
            result=json.loads((folder/'result.json').read_text())
            selection=next(x for x in result['selection'] if x['label']=='selected_C' and x['map']==kind and x['head']==family)
            state=saved['models'][f'{kind}_{family}_C_selected']
            m=copy.deepcopy(A);m.load_state_dict(state['scorer']);h=load_head(kind,family,state['head'])
            rows.append(audit(m,h,data,edges,levels,v,cfg,d,selection['weight'],dict(map=kind,head=family,checkpoint='C_selected',step=None)))
    # Fixed initial head across lambda: isolate weight changes from B initialization.
    kind='nonlinear';family='affine';design=variants[kind];levels=tr.level_data(design['train'],cfg)
    v=tr.validation_data(design['val'],features,cfg)
    selection=next(x for x in result['selection'] if x['label']=='selected_C' and x['map']==kind and x['head']==family)
    initial=saved['models'][f'{kind}_{family}_lambda{selection["weight"]:g}']['B_head']
    paths=[]
    for weight in [0.]+d['level_weights']:
        m=copy.deepcopy(A);h=load_head(kind,family,initial)
        params=list(m.scorer.parameters())+list(h.parameters())
        opt=torch.optim.Adam([dict(params=m.scorer.parameters(),lr=d['joint']['scorer_lr']),dict(params=h.parameters(),lr=d['joint']['head_lr'])])
        for step in range(args.steps+1):
            z=components(m,h,data,edges,levels,v,cfg,d,weight)
            row=dict(weight=weight,step=step,**{k:float(value.detach()) for k,value in z.items()})
            row['validation']=d['validation_rank_weight']*row['val_rank']+d['validation_level_weight']*row['val_level']
            paths.append(row)
            if step in (0,10,args.steps):
                rows.append(audit(m,h,data,edges,levels,v,cfg,d,weight,dict(map=kind,head=family,checkpoint='same_start_path',step=step)))
            if step==args.steps: break
            # Actual Adam delta, with the exact shared clipping rule from joint training.
            gv=vector(d['validation_rank_weight']*z['val_rank']+d['validation_level_weight']*z['val_level'],list(m.scorer.parameters()))
            hv=vector(d['validation_rank_weight']*z['val_rank']+d['validation_level_weight']*z['val_level'],list(h.parameters()))
            old=[p.detach().clone() for p in params]
            opt.zero_grad(set_to_none=True)
            objective=z['rank']+z['stability']+z['complexity']+z['level']+z['head_penalty']
            objective.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,20.));opt.step()
            ns=sum(p.numel() for p in m.scorer.parameters());delta=torch.cat([(p.detach()-q).reshape(-1) for p,q in zip(params,old)])
            row['gradient_norm_before_clipping']=norm;row['actual_scorer_update_norm']=float(delta[:ns].norm())
            row['actual_head_update_norm']=float(delta[ns:].norm())
            row['adam_validation_linear_scorer']=float(gv.dot(delta[:ns]));row['adam_validation_linear_head']=float(hv.dot(delta[ns:]))
        print(f'Finished same-start lambda={weight:g}',flush=True)
    out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'gradients.csv',index=False)
    pd.DataFrame(paths).to_csv(out/'short_paths.csv',index=False)
    after=hashlib.sha256(path.read_bytes()).hexdigest()
    assert before==after
    (out/'metadata.json').write_text(json.dumps(dict(job=job,config=d,runtime=versions,source_checkpoint_sha256=before,
        baseline_validation_reproduced=True,feature_transforms_reproduced=True,checkpoint_unchanged=True,
        monte_carlo_rerun=False,short_diagnostic_steps=args.steps,head_anchor_weight=selection['weight'],
        head_anchor_map=kind,head_anchor_family=family,test_or_latent_targets_used=False,
        note='lambda=0 path retains the joint head optimizer/penalty to isolate lambda; it is not the original D arm'),indent=2))
    print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--job',default='r001_c32_quadratic_nn')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=20)
    args=p.parse_args()
    if args.steps<1: p.error('steps must be positive')
    with threadpool_limits(limits=1): run(args)
