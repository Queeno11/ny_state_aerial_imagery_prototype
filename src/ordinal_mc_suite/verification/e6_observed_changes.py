"""One bounded paired-outcome diagnostic: A/O/B/C/D, no searches or extra seeds."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import diagnose_e6_gradients as diag
from e6_targeted_trials import cell_variance
from ordinal_mc.design import matched_panel,make_inputs
from ordinal_mc.e6.data import cell_parameters,measurement


def labeled_pair(seed,contexts,n,cfg,dgp,d,settings):
    work=copy.copy(cfg);work.test_per_city=n
    p=matched_panel(seed,contexts,work,dgp,'quadratic',settings['training_change_step'],settings['changed_share'])
    c=np.repeat(np.arange(len(contexts)),n);m=len(c)
    x=np.vstack([p['x_before'],p['x_after']]);h=(x@dgp.mix.T)[:,:3]
    cc=np.tile(c,2);t=np.r_[np.zeros(m,int),np.ones(m,int)]
    f=dgp.score(h,'quadratic');errors=[]
    for endpoint in ['before','after']:
        rng=np.random.default_rng(diag.core.seed_for(seed,'outcome',endpoint))
        errors.append(dgp.noise_sd*(rng.gumbel(size=m)-np.euler_gamma)*np.sqrt(6)/np.pi)
    r=f+np.concatenate(errors);a,b,k=cell_parameters(contexts,cc,t,d)
    w=measurement(r,a,b,k,'nonlinear')
    # Actual observations only; H and stability indicators remain outside learner input.
    labels=dict(x=x,c=cc,t=t,cell=2*cc+t,w=w,tract_id=np.tile(np.arange(m)+seed*100000,2))
    return labels,h,dict(n_tracts=m,n_assigned_stable=int((~p['assigned_changed']).sum()),
        error_cross_date_correlation=float(np.corrcoef(errors[0],errors[1])[0,1]))


def all_pairs(labels):
    ii=[];jj=[];ww=[];n=len(labels['x'])
    for cell in np.unique(labels['cell']):
        loc=np.flatnonzero(labels['cell']==cell);i=np.repeat(loc,len(loc));j=np.tile(loc,len(loc));keep=i!=j
        ii.extend(i[keep]);jj.extend(j[keep]);ww.extend(np.full(int(keep.sum()),1/(n*(len(loc)-1))))
    return (np.asarray(ii,dtype=np.int64),np.asarray(jj,dtype=np.int64)),np.asarray(ww)


def head_losses(head,s,labels,denoms,cfg):
    y=diag.core.tensor(labels['w'],cfg);cells=diag.core.tensor(labels['cell'],cfg,True)
    predicted=head(s,cells);m=len(y)//2
    level=(predicted-y).square().mean()/denoms['W']
    change=((predicted[m:]-predicted[:m])-(y[m:]-y[:m])).square().mean()/denoms['delta_W']
    return level,change


@torch.no_grad()
def evaluate(model,head,features,design,test_design,cfg,d):
    predict=lambda x:diag.core.predictions(model,features,x,cfg)
    score=predict(design['train']['x']);sd=float(cell_variance(torch.tensor(score),torch.tensor(design['train']['cell'])).sqrt())
    cal=design['change_calibration'];movement=(predict(cal['x_after'])-predict(cal['x_before']))/sd
    cutoff=float(np.quantile(abs(movement),.95,method='higher'));rows=[]
    for strength in [.15,.4]:
        for where in ['seen','unseen']:
            p=test_design['e6_panels'][strength][where];s0=predict(p['x_before']);s1=predict(p['x_after'])
            delta=(s1-s0)/sd;truth=p['delta_f'];changed=abs(truth)>d['truth_threshold'];stable=~p['assigned_changed']
            x=truth[changed];y=delta[changed];slope=float(np.mean((x-x.mean())*(y-y.mean()))/np.var(x))
            corr=float(np.corrcoef(x,y)[0,1]);test,pp=test_design['tests'][where];i,j=pp['within'];s=predict(test['x'])
            row=dict(strength=strength,where=where,change_rmse=float(np.sqrt(np.mean((y-x)**2))),
                stable_rms=float(np.sqrt(np.mean(delta[stable]**2))),threshold=cutoff,sensitivity=float(np.mean(abs(delta[changed])>cutoff)),
                false_positive_rate=float(np.mean(abs(delta[stable])>cutoff)),slope=slope,correlation=corr,
                observed_rank_accuracy=1-diag.core.misranking(s[j]-s[i],test['w'][j]-test['w'][i]),
                latent_rank_accuracy=1-diag.core.misranking(s[j]-s[i],test['f'][j]-test['f'][i]),
                level_MSE=None,change_W_MSE=None,zero_change_W_MSE=None)
            if where=='seen':
                p0=head(diag.core.tensor(s0,cfg),diag.core.tensor(p['cell_before'],cfg,True)).numpy()
                p1=head(diag.core.tensor(s1,cfg),diag.core.tensor(p['cell_after'],cfg,True)).numpy()
                row.update(level_MSE=float(np.mean(np.r_[p0-p['w_before'],p1-p['w_after']]**2)),
                    change_W_MSE=float(np.mean((p1-p0-(p['w_after']-p['w_before']))**2)),
                    zero_change_W_MSE=float(np.mean((p['w_after']-p['w_before'])**2)))
            rows.append(row)
    return rows


def run(a):
    source=a.run.resolve();out=a.out.resolve();settings=json.loads(a.config.read_text())
    if out.exists() and any(out.iterdir()): raise ValueError('Choose a new output directory')
    manifest_path=source/'manifest.json';source_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    d=json.loads(manifest_path.read_text())['config'];cfg=diag.e5_runtime(d,32);dgp=diag.PopulationDGP(cfg)
    original,variants=diag.build(cfg,dgp,1,'quadratic',d);test_design=variants['nonlinear']
    fit_seed=diag.core.seed_for(original['seed'],'observed_change_one_shot','fit')
    val_seed=diag.core.seed_for(original['seed'],'observed_change_one_shot','val')
    train,h,train_info=labeled_pair(fit_seed,original['contexts'],settings['fit_tracts_per_city'],cfg,dgp,d,settings)
    val,_,val_info=labeled_pair(val_seed,original['contexts'],settings['validation_tracts_per_city'],cfg,dgp,d,settings)
    assert not set(train['tract_id'])&set(val['tract_id'])
    n=len(train['x']);m=n//2;pairs,weights=all_pairs(train)
    # Keep E5's two unlabeled same-H temporal views for EVERY fitting endpoint.
    # These are separate from the actual other-date outcome, which may have changed.
    src=np.repeat(np.arange(n),cfg.views)
    aux=make_inputs(h[src],train['c'][src],1-train['t'][src],original['contexts'],
                    np.random.default_rng(diag.core.seed_for(fit_seed,'external_stable_views')),cfg,dgp)
    np.testing.assert_allclose((aux@dgp.mix.T)[:,:3],h[src],atol=2e-13)
    links=dict(temporal=np.column_stack([src,n+np.arange(len(src))]),cross_city=np.empty((0,2),np.int64))
    design=dict(train=train,val=val,x=np.vstack([train['x'],aux]),pairs=pairs,pair_weights=weights,
                change_calibration=original['change_calibration'])
    denoms=dict(W=float(np.var(train['w'])),delta_W=float(np.var(train['w'][m:]-train['w'][:m])))
    assert min(denoms.values())>0
    features=diag.core.Features(train['x'],'nn',cfg);data=diag.core.prepare(design,features,cfg)
    valid=diag.tr.validation_data(val,features,cfg);edges={k:diag.core.tensor(v,cfg,True) for k,v in links.items()}
    torch.manual_seed(diag.core.seed_for(cfg.seed,'fit',1,512,32,'quadratic','nn',0))
    A=diag.core.Estimator(7,'nn','common_logit',cfg).double()
    baseline_d=copy.deepcopy(d);baseline_d['baseline'].update(adam_steps=settings['baseline_max_adam'],lbfgs_steps=0)
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(dict(source_config=d,diagnostic=settings,fit_seed=fit_seed,val_seed=val_seed),indent=2))
    info=diag.tr.baseline(A,data,edges,valid,cfg,baseline_d)
    pd.DataFrame(info['trace']).to_csv(out/'baseline_training.csv',index=False)
    print('Baseline fitted and validation checkpoint restored',flush=True)
    head=diag.CalibrationHead(diag.core.predictions(A,features,train['x'],cfg),train['cell'],train['w'],'spline',d['head_training'])
    with torch.no_grad(): frozen=A.scorer(data['x'][:n]).detach()
    initinfo,_=diag.tr.adam_selected([head],[dict(params=head.parameters(),lr=settings['head_lr'])],
        lambda:head_losses(head,frozen,train,denoms,cfg)[0],
        lambda:head_losses(head,A.scorer(valid['x']).detach(),val,denoms,cfg)[0],
        settings['head_initialization_max_updates'],10,20,d['min_delta'])
    pd.DataFrame(initinfo['trace']).to_csv(out/'head_initialization.csv',index=False)
    head.audit_monotonicity();print('Shared frozen level head initialized',flush=True)
    # Local component diagnostics at the common initialization; no population claim.
    _,parts=diag.core.score_objective(A,data,edges,(.3,0.),0.,cfg.tau)
    lw,ld=head_losses(head,A.scorer(data['x'][:n]),train,denoms,cfg)
    components=dict(rank=parts[0],weighted_stability=parts[1],level=lw,change=ld)
    gradients={k:diag.vector(v,list(A.scorer.parameters())) for k,v in components.items()}
    norm_rows=[dict(component=k,loss=float(components[k].detach()),scorer_gradient_norm=float(g.norm())) for k,g in gradients.items()]
    cosine_rows=[dict(first=k,second=j,cosine=diag.cosine(gradients[k],gradients[j])) for i,k in enumerate(gradients) for j in list(gradients)[i+1:]]
    torch.testing.assert_close(diag.vector(parts[0]+parts[1]+lw+ld,list(A.scorer.parameters())),sum(gradients.values()),atol=1e-10,rtol=1e-9)
    pd.DataFrame(norm_rows).to_csv(out/'initial_gradient_norms.csv',index=False)
    pd.DataFrame(cosine_rows).to_csv(out/'initial_gradient_cosines.csv',index=False)
    metrics=[];trace=[];states={}
    def observe(arm,step,model,hh):
        with torch.no_grad():
            e5,pp=diag.core.score_objective(model,data,edges,(.3,0.),0.,cfg.tau)
            fw,fd=head_losses(hh,model.scorer(data['x'][:n]),train,denoms,cfg)
            vw,vd=head_losses(hh,model.scorer(valid['x']),val,denoms,cfg)
            trace.append(dict(arm=arm,step=step,rank_fit=float(pp[0]),weighted_stability=float(pp[1]),
                fit_level=float(fw),fit_change=float(fd),val_rank=float(diag.tr.rank_loss(model,valid)),val_level=float(vw),val_change=float(vd)))
        metrics.extend(dict(arm=arm,step=step,**row) for row in evaluate(model,hh,features,design,test_design,cfg,d))
        states[f'{arm}_{step}']=dict(model=diag.core.cpu_state(model),head=diag.core.cpu_state(hh))
    observe('A',0,A,head)
    for arm in ['O','B','C','D']:
        model=copy.deepcopy(A);hh=copy.deepcopy(head)
        groups=[];params=[]
        if arm!='B':
            groups.append(dict(params=model.scorer.parameters(),lr=settings['scorer_lr']));params+=list(model.scorer.parameters())
        groups.append(dict(params=hh.parameters(),lr=settings['head_lr']));params+=list(hh.parameters())
        opt=torch.optim.Adam(groups)
        for step in range(settings['additional_updates']+1):
            if step in settings['checkpoints']: observe(arm,step,model,hh)
            if step==settings['additional_updates']: break
            opt.zero_grad(set_to_none=True)
            if arm=='B':
                scores=frozen;e5=scores.new_tensor(0.)
            else:
                e5,_=diag.core.score_objective(model,data,edges,(.3,0.),0.,cfg.tau)
                scores=model.scorer(data['x'][:n])
                if arm=='O': scores=scores.detach()
            fw,fd=head_losses(hh,scores,train,denoms,cfg)
            objective=e5+fw+(fd if arm in ['B','D'] else 0.)
            if not bool(torch.isfinite(objective)): raise FloatingPointError('Nonfinite objective')
            objective.backward();torch.nn.utils.clip_grad_norm_(params,20.);opt.step()
        hh.audit_monotonicity()
        if arm=='B':
            for k,v in A.state_dict().items(): torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
        pd.DataFrame(metrics).to_csv(out/'trajectory_metrics.csv',index=False)
        pd.DataFrame(trace).to_csv(out/'observable_paths.csv',index=False)
        print('Completed fixed-budget arm '+arm,flush=True)
    df=pd.DataFrame(metrics);final=df[(df.arm=='A')|(df.step==200)].copy()
    final.to_csv(out/'results.csv',index=False)
    # A/B scorer metrics must be identical at every checkpoint, including thresholds.
    keys=['change_rmse','stable_rms','threshold','sensitivity','false_positive_rate','slope','correlation','observed_rank_accuracy','latent_rank_accuracy']
    for _,brow in df[df.arm=='B'].iterrows():
        ref=df[(df.arm=='A')&(df.strength==brow.strength)&(df['where']==brow['where'])].iloc[0]
        np.testing.assert_array_equal(brow[keys].to_numpy(float),ref[keys].to_numpy(float))
    differences=[]
    for control in ['C','B','O']:
        for _,dr in final[final.arm=='D'].iterrows():
            cr=final[(final.arm==control)&(final.strength==dr.strength)&(final['where']==dr['where'])].iloc[0]
            differences.append(dict(contrast='D-'+control,strength=dr.strength,where=dr['where'],
                **{k:(float(dr[k]-cr[k]) if pd.notna(dr[k]) else None) for k in keys+['level_MSE','change_W_MSE']}))
    pd.DataFrame(differences).to_csv(out/'contrasts.csv',index=False)
    torch.save(dict(checkpoints=states,features=dict(mu=features.mu,sd=features.sd,bmu=features.bmu,bsd=features.bsd)),out/'checkpoints.pt')
    assert source_hash==hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    metadata=dict(denominators=denoms,fit_seed=fit_seed,validation_seed=val_seed,fit=train_info,validation=val_info,
        fit_endpoints=n,validation_endpoints=len(val['x']),stable_auxiliary_links=len(src),noise_sd_effective=dgp.noise_sd,
        errors='Independent centered Gumbel draws across dates, same E6 scaling; no persistence',
        baseline={k:v for k,v in info.items() if k!='trace'},head_initialization={k:v for k,v in initinfo.items() if k!='trace'},
        source_manifest_unchanged=True,scorer_AB_equivalence_verified=True,split_disjoint=True,
        exact_auxiliary_links_verified=True,head_monotonicity_verified=True,gradient_sum_verified=True,
        additional_scorer_updates=600,additional_head_updates_per_branch=200,
        O_heads='Levels on detached scores; no feedback into the ordinal scorer',
        head_penalty=0.,test_or_stability_indicator_used_for_selection=False,
        unseen_outcomes='Unavailable: cell-specific heads have no zero-shot unseen-city calibration',
        slope_correlation='Changing tracts only; slope includes an intercept',
        auxiliary_interpretation='Same-H counterfactual acquisition views, not the actual other-date endpoint of a changing tract')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1): run(a)
