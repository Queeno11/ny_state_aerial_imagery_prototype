"""Bounded one-replication diagnostics plus read-only paired completed-MC analysis."""
import argparse
import copy
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import diagnose_e6_gradients as diag
from ordinal_mc.design import panel,make_inputs
from ordinal_mc.e6.data import cell_parameters,measurement,conditional_mean
from ordinal_mc.e6.report import paired_differences


def cell_variance(scores,cells):
    # Exactly .5 sum ordered-pair weights * gap^2, evaluated in O(N).
    return sum((cells==c).sum()/len(cells)*scores[cells==c].var(unbiased=True) for c in torch.unique(cells))


@torch.no_grad()
def targets(model,features,base,cfg,d):
    predict=lambda x:diag.core.predictions(model,features,x,cfg)
    fit=predict(base['train']['x']);sd=float(cell_variance(torch.tensor(fit),torch.tensor(base['train']['cell'])).sqrt())
    rows=[]
    for strength in [.15,.4]:
        for where in ['seen','unseen']:
            p=base['e6_panels'][strength][where];change=(predict(p['x_after'])-predict(p['x_before']))/sd
            changed=abs(p['delta_f'])>d['truth_threshold'];stable=~p['assigned_changed']
            rows.append(dict(strength=strength,where=where,change_rmse=float(np.sqrt(np.mean((change[changed]-p['delta_f'][changed])**2))),
                             stable_rms=float(np.sqrt(np.mean(change[stable]**2)))))
    return rows


def paired_existing(run,out):
    source=sqlite3.connect((run/'results.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    dest=sqlite3.connect(out/'completed_snapshot.sqlite');source.backup(dest);dest.close();source.close()
    con=sqlite3.connect((out/'completed_snapshot.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    results=[json.loads(r[0]) for r in con.execute('SELECT payload FROM results ORDER BY job')];con.close()
    combinations={(r['job']['cities'],r['job']['truth']) for r in results}
    common=set.intersection(*[{r['job']['rep'] for r in results if (r['job']['cities'],r['job']['truth'])==key} for key in combinations])
    f=pd.DataFrame([x for r in results if r['job']['rep'] in common for x in r['metrics']])
    f.to_csv(out/'paired_metrics.csv',index=False)
    paired_differences(f).to_csv(out/'paired_differences.csv',index=False)
    return dict(completed_jobs=len(results),balanced_reps=sorted(common),balanced_jobs=sum(r['job']['rep'] in common for r in results))


def run(a):
    run=a.run.resolve();out=a.out.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Choose new output directory')
    out.mkdir(parents=True,exist_ok=True)
    folder=run/'replications'/'r001_c32_quadratic_nn';path=folder/'checkpoints.pt';before=hashlib.sha256(path.read_bytes()).hexdigest()
    saved=torch.load(path,map_location='cpu',weights_only=False);d=saved['config'];job=saved['job'];cfg=diag.e5_runtime(d,job['cities'])
    dgp=diag.PopulationDGP(cfg);base,variants=diag.build(cfg,dgp,job['rep'],job['truth'],d)
    features=diag.core.Features(base['train']['x'],job['model'],cfg)
    for name,attr in [('feature_mu','mu'),('feature_sd','sd'),('feature_bmu','bmu'),('feature_bsd','bsd')]:
        np.testing.assert_allclose(saved[name],getattr(features,attr),atol=1e-12,rtol=1e-12)
    A=diag.core.Estimator(7,'nn','common_logit',cfg).double();A.load_state_dict(saved['models']['A']['scorer'])
    data=diag.core.prepare(base,features,cfg);edges={k:diag.core.tensor(base['links'][k],cfg,True) for k in ('temporal','cross_city')}
    with torch.no_grad():
        s=A.scorer(data['x'][:base['n_fit']]);v=cell_variance(s,data['obs_cells'])
        _,parts=diag.core.score_objective(A,data,edges,(d['lambda_t'],0.),d['alpha'],cfg.tau)
        torch.testing.assert_close(v,parts[3],atol=1e-12,rtol=1e-12)
    def load_head(kind,family):
        tr=variants[kind]['train']
        h=diag.CalibrationHead(diag.core.predictions(A,features,tr['x'],cfg),tr['cell'],tr['w'],family,d['head_training'])
        h.load_state_dict(saved['models'][f'{kind}_{family}_lambda10']['B_head']);return h
    kind='nonlinear';head0=load_head(kind,'affine');levels=diag.tr.level_data(variants[kind]['train'],cfg)
    val=diag.tr.validation_data(variants[kind]['val'],features,cfg)
    paths=[];evaluations=[];matched=[];fractions=[.001,.0025,.005]
    for scorer_mult,head_mult in [(1.,1.),(.1,1.),(1.,.1),(.1,.1)]:
        model=copy.deepcopy(A);head=copy.deepcopy(head0);params=list(model.scorer.parameters())+list(head.parameters())
        opt=torch.optim.Adam([dict(params=model.scorer.parameters(),lr=d['joint']['scorer_lr']*scorer_mult),
                              dict(params=head.parameters(),lr=d['joint']['head_lr']*head_mult)])
        tag=dict(scorer_multiplier=scorer_mult,head_multiplier=head_mult);attained=set();initial=None
        for step in range(101):
            objective=diag.tr.joint_loss(model,head,data,edges,levels,cfg,d,10.)
            with torch.no_grad():
                level=float(diag.tr.level_loss(model,head,data['x'],levels))
                _,parts=diag.core.score_objective(model,data,edges,(d['lambda_t'],0.),d['alpha'],cfg.tau)
                validation=diag.tr.observable_rule(model,head,val,levels['variance'],d)
            if initial is None: initial=level
            reduction=(initial-level)/initial
            paths.append(dict(**tag,step=step,fit_W_mse=level,relative_W_reduction=reduction,objective=float(objective.detach()),
                              stability_penalty=float(parts[1]),validation=validation))
            events=[]
            if step in [0,1,5,10,20,50,100]: events.append(('equal_updates',None))
            for fraction in fractions:
                if fraction not in attained and reduction>=fraction:
                    attained.add(fraction);events.append(('matched_W_reduction',fraction))
            if events:
                measured=targets(model,features,base,cfg,d)
                for event,fraction in events:
                    for row in measured:
                        evaluations.append(dict(**tag,step=step,event=event,target_reduction=fraction,actual_reduction=reduction,**row))
                    if event=='matched_W_reduction': matched.append(dict(**tag,target_reduction=fraction,attained=True,step=step,actual_reduction=reduction))
            if step==100: break
            opt.zero_grad(set_to_none=True);objective.backward();torch.nn.utils.clip_grad_norm_(params,20.);opt.step()
        for fraction in fractions:
            if fraction not in attained: matched.append(dict(**tag,target_reduction=fraction,attained=False,step=None,actual_reduction=None))
        print('LR path complete',tag,flush=True)
        pd.DataFrame(paths).to_csv(out/'lr_paths.csv',index=False)
        pd.DataFrame(evaluations).to_csv(out/'lr_target_metrics.csv',index=False)
        pd.DataFrame(matched).to_csv(out/'matched_W_attainment.csv',index=False)
    # One new sample, fixed contexts and independent unit/noise/acquisition draws.
    fresh_seed=diag.core.seed_for(base['seed'],'targeted_fresh_gradient_v1')
    fresh=panel(fresh_seed,base['contexts'],base['cell_z'],d['labels_per_city'],cfg,dgp,job['truth'],variants[kind]['measurement_scenario'])
    source=np.repeat(np.arange(len(fresh['x'])),d['views'])
    aux=make_inputs(fresh['h'][source],fresh['c'][source],1-fresh['t'][source],base['contexts'],
                    np.random.default_rng(diag.core.seed_for(fresh_seed,'views')),cfg,dgp)
    tx=diag.core.tensor(features(fresh['x']),cfg);ta=diag.core.tensor(features(aux),cfg);cells=diag.core.tensor(fresh['cell'],cfg,True)
    params=list(A.scorer.parameters());s=A.scorer(tx);sa=A.scorer(ta)
    fresh_stability=d['lambda_t']*(s[diag.core.tensor(source,cfg,True)]-sa).square().mean()/cell_variance(s,cells)
    gs=diag.vector(fresh_stability,params);grads=[]
    for kind in ['affine','nonlinear']:
        aa,bb,kk=cell_parameters(base['contexts'],fresh['c'],fresh['t'],d)
        realized=measurement(fresh['r'],aa,bb,kk,kind)
        mu=conditional_mean(fresh['f'],aa,bb,kk,kind,base['noise_sd_effective'],d['quadrature_nodes'])
        variance=float(np.var(variants[kind]['train']['w']))
        for family in ['affine','spline']:
            head=load_head(kind,family);prediction=head(s,cells)
            mu_loss=(prediction-diag.core.tensor(mu,cfg)).square().mean()/variance
            noisy_loss=(prediction-diag.core.tensor(realized,cfg)).square().mean()/variance
            gn=diag.vector(10*noisy_loss,params);gm=diag.vector(10*mu_loss,params)
            grads.append(dict(map=kind,head=family,fresh_n=len(fresh['x']),fit_variance_W=variance,
                realized_loss=float(noisy_loss.detach()),oracle_mean_loss=float(mu_loss.detach()),
                realized_grad_norm=float(gn.norm()),oracle_grad_norm=float(gm.norm()),stability_grad_norm=float(gs.norm()),
                noisy_stability_cosine=diag.cosine(gn,gs),oracle_stability_cosine=diag.cosine(gm,gs),
                noisy_oracle_cosine=diag.cosine(gn,gm),noise_difference_norm=float((gn-gm).norm()),
                fresh_stability_penalty=float(fresh_stability.detach())))
    pd.DataFrame(grads).to_csv(out/'fresh_noise_oracle_gradients.csv',index=False)
    paired=paired_existing(run,out)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    metadata=dict(job=job,weight=10,head_anchor='nonlinear affine B at lambda10',max_steps_per_path=100,
        total_scorer_updates=400,fresh_gradient_sample_seed=fresh_seed,variance_identity_verified=True,
        source_checkpoint_unchanged=True,source_checkpoint_sha256=before,test_or_oracle_used_for_selection=False,
        matched_W_rule='first actual attainment; report actual reduction; no interpolation; no model selection',**paired)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    with threadpool_limits(limits=1): run(args)
