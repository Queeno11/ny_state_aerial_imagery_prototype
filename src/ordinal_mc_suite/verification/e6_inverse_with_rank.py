"""One-shot noisy oracle-inverse supervision retaining the E5 rank term."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import diagnose_e6_gradients as diag
from e6_targeted_trials import targets
from ordinal_mc.design import inverse_measurement


def run(a):
    root=a.run.resolve();out=a.out.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Choose a new output directory')
    folder=root/'replications'/'r001_c32_quadratic_nn';path=folder/'checkpoints.pt'
    before=hashlib.sha256(path.read_bytes()).hexdigest();saved=torch.load(path,map_location='cpu',weights_only=False)
    d=saved['config'];job=saved['job'];cfg=diag.e5_runtime(d,job['cities'])
    base,variants=diag.build(cfg,diag.PopulationDGP(cfg),job['rep'],job['truth'],d)
    features=diag.core.Features(base['train']['x'],'nn',cfg)
    for name,attr in [('feature_mu','mu'),('feature_sd','sd'),('feature_bmu','bmu'),('feature_bsd','bsd')]:
        np.testing.assert_allclose(saved[name],getattr(features,attr),atol=1e-12,rtol=1e-12)
    inverse={}
    for kind,design in variants.items():
        inverse[kind]={}
        for group in ['train','val']:
            q=design[group]
            inverse[kind][group]=inverse_measurement(q['w'],q['a'],q['b'],q['kappa'],design['measurement_scenario']['psi_delta'],'tanh')
            # Implementation verification only, not a training target source.
            np.testing.assert_allclose(inverse[kind][group],q['r'],atol=1e-12)
    for group in ['train','val']:
        np.testing.assert_allclose(inverse['affine'][group],inverse['nonlinear'][group],atol=1e-12)
    r=diag.core.tensor(inverse['nonlinear']['train'],cfg);vr=diag.core.tensor(inverse['nonlinear']['val'],cfg)
    variance=float(np.var(inverse['nonlinear']['train']))
    data=diag.core.prepare(base,features,cfg);val=diag.tr.validation_data(base['val'],features,cfg)
    edges={k:diag.core.tensor(base['links'][k],cfg,True) for k in ['temporal','cross_city']}
    A=diag.core.Estimator(7,'nn','common_logit',cfg).double();A.load_state_dict(saved['models']['A']['scorer'])
    np.testing.assert_allclose(float(diag.tr.rank_loss(A,val).detach()),saved['models']['A']['info']['validation'],atol=1e-11)
    histories=[];candidates=[]
    out.mkdir(parents=True,exist_ok=True)
    def components(model):
        e5,parts=diag.core.score_objective(model,data,edges,(d['lambda_t'],0.),d['alpha'],cfg.tau)
        level=(model.scorer(data['x'][:len(r)])-r).square().mean()/variance
        rankval=diag.tr.rank_loss(model,val)
        r_val=(model.scorer(val['x'])-vr).square().mean()/variance
        return e5,level,rankval,r_val,parts
    # All candidates and controls share A and a 100-update ceiling. Initial A is eligible.
    for weight,lr in [(w,lr) for w in [.1,1.,10.] for lr in [.001,.0001]]+[(0.,.001),(0.,.0001)]:
        model=copy.deepcopy(A);opt=torch.optim.Adam(model.scorer.parameters(),lr=lr)
        best=float('inf');best_state=None;best_step=None
        for step in range(101):
            e5,level,rankval,r_val,parts=components(model)
            if step%10==0:
                criterion=float((rankval+r_val if weight else rankval).detach())
                histories.append(dict(weight=weight,lr=lr,step=step,rank_fit=float(parts[0].detach()),
                    stability=float(parts[1].detach()),R_fit_mse=float(level.detach()),rank_val=float(rankval.detach()),
                    R_val_mse=float(r_val.detach()),selection=criterion,fit_objective=float((e5+weight*level).detach())))
                if criterion<best: best=criterion;best_state=diag.core.cpu_state(model);best_step=step
            if step==100: break
            opt.zero_grad(set_to_none=True);(e5+weight*level).backward()
            torch.nn.utils.clip_grad_norm_(model.scorer.parameters(),20.);opt.step()
        model.load_state_dict(best_state)
        candidates.append(dict(weight=weight,lr=lr,best_step=best_step,selection=best,model=model))
        pd.DataFrame(histories).to_csv(out/'training_validation_paths.csv',index=False)
        print(f'Completed lambda_R={weight:g}, lr={lr:g}: selected step {best_step}',flush=True)
    # Finalize selection before exposing latent change targets.
    inverse_best=min([c for c in candidates if c['weight']],key=lambda c:(c['selection'],c['weight'],c['lr']))
    controls=[c for c in candidates if not c['weight']]
    Dmatched=next(c for c in controls if c['lr']==inverse_best['lr'])
    Dbest=min(controls,key=lambda c:(c['selection'],c['lr']))
    pd.DataFrame([{k:v for k,v in c.items() if k!='model'} for c in candidates]).to_csv(out/'candidate_selection.csv',index=False)
    metrics=[]
    for label,model,info in [('B_frozen',A,{}),('inverse_rank_selected',inverse_best['model'],inverse_best),
                              ('D_matched_lr',Dmatched['model'],Dmatched),('D_best_rank_validation',Dbest['model'],Dbest)]:
        metrics.extend(dict(arm=label,**{k:v for k,v in info.items() if k!='model'},**m) for m in targets(model,features,base,cfg,d))
    # Exploratory candidate test metrics remain separate and did not enter selection.
    exploratory=[]
    for c in candidates:
        if c['weight']:
            exploratory.extend(dict(**{k:v for k,v in c.items() if k!='model'},**m) for m in targets(c['model'],features,base,cfg,d))
    pd.DataFrame(metrics).to_csv(out/'selected_target_metrics.csv',index=False)
    pd.DataFrame(exploratory).to_csv(out/'exploratory_candidate_metrics.csv',index=False)
    torch.save(dict(inverse_rank=diag.core.cpu_state(inverse_best['model']),D_matched=diag.core.cpu_state(Dmatched['model'])),out/'diagnostic_checkpoints.pt')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    meta=dict(job=job,fit_variance_R=variance,n_fit=len(r),n_validation=len(vr),
        objective='L_E5 + lambda_R * mean((s(X)-Psi_inverse(W))^2)/Var_fit(R)',
        retained_rank=True,retained_stability=True,raw_rank_gaps_unchanged=True,scorer_architecture_unchanged=True,
        inverse_maps_agree=True,oracle_inverse_verified=True,noise_free_f_used_for_training=False,
        inverse_selection_rule='validation rank logistic + validation noisy R MSE / Var_fit(R); fixed 1:1 weights',
        D_selection_rule='validation rank logistic',total_scorer_updates=800,max_updates_per_candidate=100,
        initial_checkpoint_eligible=True,source_checkpoint_unchanged=True,
        selected={k:v for k,v in inverse_best.items() if k!='model'},
        note='One-replication diagnostic. Separate exploratory candidate test metrics are not selection evidence.')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1): run(a)
