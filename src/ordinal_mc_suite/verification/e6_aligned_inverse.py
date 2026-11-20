"""One-shot affine-aligned oracle-inverse objective, with raw RankNet retained."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits
import diagnose_e6_gradients as diag
from e6_targeted_trials import targets
from ordinal_mc.design import inverse_measurement


class AffineInverse(nn.Module):
    def __init__(self,intercept,slope):
        super().__init__()
        self.a=nn.Parameter(torch.tensor(intercept,dtype=torch.float64))
        self.log_b=nn.Parameter(torch.tensor(np.log(slope),dtype=torch.float64))
    def forward(self,s): return self.a+self.log_b.exp()*s


def fitting_head(model,features,design,cfg,r):
    s=diag.core.predictions(model,features,design['train']['x'],cfg)
    intercept,slope=np.linalg.lstsq(np.column_stack([np.ones(len(s)),s]),r,rcond=None)[0]
    if slope<=0: raise ValueError('Nonpositive OLS slope; constrained fitting required')
    return AffineInverse(float(intercept),float(slope))


def run(a):
    root=a.run.resolve();out=a.out.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Choose new output directory')
    folder=root/'replications'/'r001_c32_quadratic_nn';source=folder/'checkpoints.pt'
    digest=hashlib.sha256(source.read_bytes()).hexdigest();saved=torch.load(source,map_location='cpu',weights_only=False)
    d=saved['config'];cfg=diag.e5_runtime(d,32);base,variants=diag.build(cfg,diag.PopulationDGP(cfg),1,'quadratic',d)
    features=diag.core.Features(base['train']['x'],'nn',cfg)
    for key,attr in [('feature_mu','mu'),('feature_sd','sd'),('feature_bmu','bmu'),('feature_bsd','bsd')]:
        np.testing.assert_allclose(saved[key],getattr(features,attr),atol=1e-12,rtol=1e-12)
    design=variants['nonlinear'];inverse={}
    for group in ['train','val']:
        q=design[group];inverse[group]=inverse_measurement(q['w'],q['a'],q['b'],q['kappa'],1.,'tanh')
        qa=variants['affine'][group]
        np.testing.assert_allclose(inverse[group],inverse_measurement(qa['w'],qa['a'],qa['b'],qa['kappa'],0.,'tanh'),atol=1e-12)
    r=diag.core.tensor(inverse['train'],cfg);vr=diag.core.tensor(inverse['val'],cfg);variance=float(np.var(inverse['train']))
    data=diag.core.prepare(base,features,cfg);val=diag.tr.validation_data(base['val'],features,cfg)
    edges={k:diag.core.tensor(base['links'][k],cfg,True) for k in ['temporal','cross_city']}
    A=diag.core.Estimator(7,'nn','common_logit',cfg).double();A.load_state_dict(saved['models']['A']['scorer'])
    H=fitting_head(A,features,base,cfg,inverse['train'])
    previous=json.loads((root.parent/'e6_inverse_rank_diagnostic'/'affine_validation_comparison.json').read_text())
    np.testing.assert_allclose([float(H.a.detach()),float(H.log_b.exp().detach())],[previous['intercept'],previous['slope']],atol=1e-12)
    def losses(model,head):
        e5,parts=diag.core.score_objective(model,data,edges,(d['lambda_t'],0.),d['alpha'],cfg.tau)
        fit=(head(model.scorer(data['x'][:len(r)]))-r).square().mean()/variance
        rankval=diag.tr.rank_loss(model,val)
        levelval=(head(model.scorer(val['x']))-vr).square().mean()/variance
        return e5,fit,rankval,levelval,parts
    with torch.no_grad():
        _,fit,rankval,levelval,_=losses(A,H)
        np.testing.assert_allclose(float(levelval),previous['metrics']['val']['frozen_affine']['standardized_mse'],atol=1e-11)
    # Verify initial calibration stationarity and actual scorer level-gradient flow.
    _,fit,_,_,_=losses(A,H)
    gh=diag.vector(fit,list(H.parameters()));gs=diag.vector(fit,list(A.scorer.parameters()))
    assert float(gh.norm())<1e-10 and float(gs.norm())>0
    trace=[];candidates=[];out.mkdir(parents=True,exist_ok=True)
    for weight in [.1,1.,10.,0.]:
        model=copy.deepcopy(A);head=copy.deepcopy(H)
        modules=[model,head] if weight else [model]
        params=list(model.scorer.parameters())+(list(head.parameters()) if weight else [])
        groups=[dict(params=model.scorer.parameters(),lr=.001)]
        if weight: groups.append(dict(params=head.parameters(),lr=.01))
        opt=torch.optim.Adam(groups);best=float('inf');states=None;best_step=None
        for step in range(101):
            e5,level,rankval,levelval,parts=losses(model,head)
            if step%10==0:
                criterion=float((rankval+levelval if weight else rankval).detach())
                trace.append(dict(weight=weight,step=step,fit_rank=float(parts[0].detach()),stability=float(parts[1].detach()),
                    fit_R_mse=float(level.detach()),validation_rank=float(rankval.detach()),validation_R_mse=float(levelval.detach()),
                    selection=criterion,intercept=float(head.a.detach()),slope=float(head.log_b.exp().detach())))
                if not np.isfinite(criterion): raise FloatingPointError('Nonfinite validation')
                if criterion<best:
                    best=criterion;states=[diag.core.cpu_state(m) for m in modules];best_step=step
            if step==100: break
            opt.zero_grad(set_to_none=True);q=e5+weight*level
            if not bool(torch.isfinite(q)): raise FloatingPointError('Nonfinite objective')
            q.backward();torch.nn.utils.clip_grad_norm_(params,20.);opt.step()
        for m,state in zip(modules,states): m.load_state_dict(state)
        if not weight: head=fitting_head(model,features,base,cfg,inverse['train'])
        candidates.append(dict(weight=weight,best_step=best_step,selection=best,model=model,head=head))
        pd.DataFrame(trace).to_csv(out/'training_validation_paths.csv',index=False)
        print(f'Finished aligned lambda_R={weight:g}; selected step {best_step}',flush=True)
    joint=min([c for c in candidates if c['weight']],key=lambda c:(c['selection'],c['weight']))
    D=next(c for c in candidates if not c['weight'])
    selection=[{k:v for k,v in c.items() if k not in ['model','head']} for c in candidates]
    pd.DataFrame(selection).to_csv(out/'candidate_selection.csv',index=False)
    rows=[];level_rows=[]
    for label,model,head in [('B_frozen_affine',A,H),('C_aligned_inverse',joint['model'],joint['head']),('D_ordinal_posthoc_affine',D['model'],D['head'])]:
        rows.extend(dict(arm=label,**row) for row in targets(model,features,base,cfg,d))
        with torch.no_grad():
            _,fit,rankval,levelval,_=losses(model,head)
        level_rows.append(dict(arm=label,fit_R_mse=float(fit),validation_R_mse=float(levelval),validation_rank=float(rankval),
            common_validation=float(rankval+levelval),intercept=float(head.a.detach()),slope=float(head.log_b.exp().detach())))
    pd.DataFrame(rows).to_csv(out/'selected_target_metrics.csv',index=False)
    pd.DataFrame(level_rows).to_csv(out/'selected_level_metrics.csv',index=False)
    torch.save(dict(scorer=diag.core.cpu_state(joint['model']),affine=diag.core.cpu_state(joint['head'])),out/'selected_checkpoint.pt')
    assert digest==hashlib.sha256(source.read_bytes()).hexdigest()
    meta=dict(job=saved['job'],total_scorer_updates=400,updates_per_candidate=100,learning_rates=dict(scorer=.001,head=.01),
        weights=[.1,1.,10.],selected={k:v for k,v in joint.items() if k not in ['model','head']},
        objective='L_rank(raw s)+0.3*S/V+lambda_R*mean((a+exp(log_b)*s-R)^2)/Var_fit(R)',
        head_penalty=0,calibration_fitted_on='fit observations only',
        initial_head_gradient_norm=float(gh.norm()),initial_scorer_R_gradient_norm=float(gs.norm()),
        selection='validation rank(raw s)+validation calibrated noisy-R MSE/Var_fit(R), fixed 1:1',
        D_selection='validation rank(raw s)',initial_checkpoint_eligible=True,
        initial_affine_fit_reproduced=True,positive_slope_parameterization='exp(log_b)',
        no_latent_selection=True,source_checkpoint_unchanged=True,head_or_scorer_architecture_unchanged=True)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1): run(a)
