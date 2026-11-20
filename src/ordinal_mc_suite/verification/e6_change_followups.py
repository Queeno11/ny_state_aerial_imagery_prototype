"""Two bounded follow-ups; exact saved data; no oracle-driven selection."""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.integrate import quad_vec
from threadpoolctl import threadpool_limits
import e6_observed_changes as previous
import diagnose_e6_gradients as diag
from e6_targeted_trials import cell_variance
from ordinal_mc.design import make_inputs
from ordinal_mc.e6.data import cell_parameters


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct(source, saved):
    conf=json.loads((source/'config.json').read_text());d=conf['source_config'];settings=conf['diagnostic']
    cfg=diag.e5_runtime(d,32);dgp=diag.PopulationDGP(cfg)
    original,variants=diag.build(cfg,dgp,1,'quadratic',d)
    train,h,_=previous.labeled_pair(conf['fit_seed'],original['contexts'],settings['fit_tracts_per_city'],cfg,dgp,d,settings)
    val,_,_=previous.labeled_pair(conf['val_seed'],original['contexts'],settings['validation_tracts_per_city'],cfg,dgp,d,settings)
    src=np.repeat(np.arange(len(train['x'])),cfg.views)
    aux=make_inputs(h[src],train['c'][src],1-train['t'][src],original['contexts'],
        np.random.default_rng(diag.core.seed_for(conf['fit_seed'],'external_stable_views')),cfg,dgp)
    pairs,weights=previous.all_pairs(train)
    design=dict(train=train,val=val,x=np.vstack([train['x'],aux]),pairs=pairs,pair_weights=weights,
        change_calibration=original['change_calibration'])
    features=diag.core.Features(train['x'],'nn',cfg)
    for k,v in saved['features'].items(): np.testing.assert_array_equal(v,getattr(features,k))
    data=diag.core.prepare(design,features,cfg);valid=diag.tr.validation_data(val,features,cfg)
    edges=dict(temporal=diag.core.tensor(np.column_stack([src,len(train['x'])+np.arange(len(src))]),cfg,True),
        cross_city=diag.core.tensor(np.empty((0,2),np.int64),cfg,True))
    den=json.loads((source/'metadata.json').read_text())['denominators']
    np.testing.assert_equal(den['W'],float(np.var(train['w'])))
    m=len(train['w'])//2
    np.testing.assert_equal(den['delta_W'],float(np.var(train['w'][m:]-train['w'][:m])))
    A=diag.core.Estimator(7,'nn','common_logit',cfg).double();A.load_state_dict(saved['checkpoints']['A_0']['model'])
    head=diag.CalibrationHead(diag.core.predictions(A,features,train['x'],cfg),train['cell'],train['w'],'spline',d['head_training'])
    head.load_state_dict(saved['checkpoints']['A_0']['head'])
    observed=pd.read_csv(source/'observable_paths.csv').iloc[0]
    with torch.no_grad():
        fw,fd=previous.head_losses(head,A.scorer(data['x'][:len(train['w'])]),train,den,cfg)
        _,parts=diag.core.score_objective(A,data,edges,(.3,0.),0.,cfg.tau)
    np.testing.assert_allclose([float(fw),float(fd),float(parts[0])],observed[['fit_level','fit_change','rank_fit']].to_numpy(float),atol=1e-13,rtol=1e-13)
    return dict(cfg=cfg,d=d,dgp=dgp,design=design,test=variants['nonlinear'],features=features,data=data,
        valid=valid,edges=edges,den=den,A=A,head=head)


def quantities(model,head,z):
    cfg=z['cfg'];n=len(z['design']['train']['w'])
    _,parts=diag.core.score_objective(model,z['data'],z['edges'],(.3,0.),0.,cfg.tau)
    fw,fd=previous.head_losses(head,model.scorer(z['data']['x'][:n]),z['design']['train'],z['den'],cfg)
    return parts[0],parts[1]/.3,fw,fd


def flat(params): return torch.cat([p.detach().reshape(-1) for p in params])


@torch.no_grad()
def assign(params,values):
    start=0
    for p in params:
        count=p.numel();p.copy_(values[start:start+count].view_as(p));start+=count


def project(proposal,gradients,slacks):
    """Exact Euclidean projection onto two linear halfspaces, active-set enumeration."""
    norms=torch.linalg.vector_norm(gradients,dim=1)
    if bool(torch.any(norms<=1e-20)): raise FloatingPointError('Degenerate constraint gradient')
    J=gradients/norms[:,None];b=slacks/norms
    candidates=[]
    for nactive in range(3):
        for active in itertools.combinations(range(2),nactive):
            if not active: candidate=proposal
            else:
                index=list(active);j=J[index];rhs=j@proposal-b[index]
                lam=torch.linalg.lstsq(j@j.T,rhs).solution
                if bool(torch.any(lam < -1e-12)): continue
                candidate=proposal-j.T@lam
            if bool(torch.all(J@candidate<=b+1e-12)):
                candidates.append(candidate)
    if not candidates: raise FloatingPointError('Linearized feasible projection failed')
    return min(candidates,key=lambda v:float((v-proposal).square().sum()))


def fit_branches(z,out,settings):
    cfg=z['cfg'];A=z['A'];head=copy.deepcopy(z['head']);n=len(z['design']['train']['w'])
    with torch.no_grad(): fs=A.scorer(z['data']['x'][:n]).detach();vs=A.scorer(z['valid']['x']).detach()
    # Selection uses ONLY observed validation W labels. Oracle evaluation is not passed here.
    prep,unused=diag.tr.adam_selected([head],[dict(params=head.parameters(),lr=settings['head_lr'])],
        lambda:sum(previous.head_losses(head,fs,z['design']['train'],z['den'],cfg)),
        lambda:sum(previous.head_losses(head,vs,z['design']['val'],z['den'],cfg)),
        settings['head_preparation_updates'],settings['head_selection_every'],100000,0.,stop_on_patience=False)
    pd.DataFrame(prep['trace']).to_csv(out/'head_preparation.csv',index=False)
    torch.save(diag.core.cpu_state(head),out/'selected_common_head.pt')
    with torch.no_grad(): rank,stability,_,_=quantities(A,head,z)
    refs=dict(rank_A=float(rank),stability_A=float(stability))
    refs.update(rank_bound=refs['rank_A']+settings['rank_slack'],stability_bound=settings['stability_multiplier']*refs['stability_A'])
    (out/'reference_constraints.json').write_text(json.dumps(refs,indent=2))
    print('Common head selected at',prep['best_step'],'of',prep['updates'],'constraints',refs,flush=True)
    states={};traces=[];paths=[];metrics=[]
    def observe(arm,step,model,hh):
        with torch.no_grad():
            r,p,fw,fd=quantities(model,hh,z)
            vw,vd=previous.head_losses(hh,model.scorer(z['valid']['x']),z['design']['val'],z['den'],cfg)
            paths.append(dict(arm=arm,step=step,rank_fit=float(r),stability_ratio=float(p),
                rank_violation=max(0.,float(r)-refs['rank_bound']),stability_violation=max(0.,float(p)-refs['stability_bound']),
                fit_level=float(fw),fit_change=float(fd),val_level=float(vw),val_change=float(vd),
                val_rank=float(diag.tr.rank_loss(model,z['valid']))))
        if step in settings['checkpoints']:
            states[f'{arm}_{step}']=dict(model=diag.core.cpu_state(model),head=diag.core.cpu_state(hh))
            metrics.extend(dict(arm=arm,step=step,**row) for row in previous.evaluate(model,hh,z['features'],z['design'],z['test'],cfg,z['d']))
    for arm in ['F','U','Q']:
        model=copy.deepcopy(A);hh=copy.deepcopy(head);params=list(model.scorer.parameters())
        so=torch.optim.Adam(params,lr=settings['scorer_lr']);ho=torch.optim.Adam(hh.parameters(),lr=settings['head_lr'])
        for step in range(settings['updates']+1):
            observe(arm,step,model,hh)
            if step==settings['updates']: break
            rec=dict(arm=arm,attempt=step+1,accepted=False,backtracking_evaluations=0,scale=0.,fallback=False)
            if arm=='U':
                so.zero_grad(set_to_none=True);r,p,fw,fd=quantities(model,hh,z)
                (r+.3*p+fw+fd).backward();torch.nn.utils.clip_grad_norm_(params,20.);so.step()
                rec.update(accepted=True,scale=1.)
            elif arm=='Q':
                r,p,fw,fd=quantities(model,hh,z);card=fw+fd
                g=diag.vector(card,params).detach()
                gradients=torch.stack([diag.vector(r,params).detach(),diag.vector(p,params).detach()])
                slacks=g.new_tensor([refs['rank_bound']-float(r.detach()),refs['stability_bound']-float(p.detach())])
                theta=flat(params);opt_before=copy.deepcopy(so.state_dict())
                so.zero_grad(set_to_none=True);card.backward();torch.nn.utils.clip_grad_norm_(params,20.);so.step()
                proposal=flat(params)-theta;assign(params,theta)
                direction=project(proposal,gradients,slacks)
                if float(g@direction)>=-1e-16:
                    # Deterministic feasible steepest-descent fallback, same nominal step norm.
                    raw=-g*(proposal.norm()/g.norm().clamp_min(1e-30))
                    direction=project(raw,gradients,slacks);rec['fallback']=True
                derivative=float(g@direction);base=float(card.detach())
                rec.update(linear_card_derivative=derivative,proposal_norm=float(proposal.norm()),direction_norm=float(direction.norm()))
                if derivative < -1e-16:
                    for attempt in range(settings['max_backtracking_evaluations']):
                        scale=settings['backtracking_factor']**attempt;assign(params,theta+scale*direction)
                        with torch.no_grad(): nr,np_,nw,nd=quantities(model,hh,z)
                        feasible=(float(nr)<=refs['rank_bound']+settings['constraint_numerical_tolerance'] and
                            float(np_)<=refs['stability_bound']+settings['constraint_numerical_tolerance'])
                        improving=float(nw+nd)<=base+settings['armijo']*scale*derivative
                        rec['backtracking_evaluations']=attempt+1
                        if feasible and improving:
                            rec.update(accepted=True,scale=scale);break
                if not rec['accepted']:
                    assign(params,theta);so.load_state_dict(opt_before)
                rec['card_before_scorer']=base
                with torch.no_grad(): _,_,nw,nd=quantities(model,hh,z)
                rec['card_after_scorer']=float(nw+nd)
            # Separate heads: always exactly one update, no scorer gradient feedback here.
            ho.zero_grad(set_to_none=True)
            with torch.no_grad(): scores=model.scorer(z['data']['x'][:n]).detach()
            loss=sum(previous.head_losses(hh,scores,z['design']['train'],z['den'],cfg))
            loss.backward();torch.nn.utils.clip_grad_norm_(hh.parameters(),20.);ho.step()
            traces.append(rec)
            if (step+1)%50==0: print(arm,'attempt',step+1,flush=True)
        hh.audit_monotonicity()
        if arm=='F':
            for key,value in A.state_dict().items(): torch.testing.assert_close(value,model.state_dict()[key],atol=0,rtol=0)
        pd.DataFrame(paths).to_csv(out/'observable_paths.csv',index=False)
        pd.DataFrame(traces).to_csv(out/'optimizer_trace.csv',index=False)
        pd.DataFrame(metrics).to_csv(out/'latent_trajectory_metrics.csv',index=False)
        torch.save(dict(checkpoints=states,features={k:getattr(z['features'],k) for k in ['mu','sd','bmu','bsd']}),out/'checkpoints.pt')
    q=pd.DataFrame(paths).query("arm == 'Q'")
    assert q.rank_violation.max()<=settings['constraint_numerical_tolerance']
    assert q.stability_violation.max()<=settings['constraint_numerical_tolerance']
    return states,refs,{k:v for k,v in prep.items() if k!='trace'}


def moments(f,a,b,k,sd,tol,interval):
    """Integrate tanh, epsilon*tanh, tanh^2; exact centered-noise first two moments."""
    result=[]
    for start in range(0,len(f),2048):
        ff=f[start:start+2048]
        def integrand(z):
            eps=sd*np.sqrt(6)/np.pi*(z-np.euler_gamma)
            t=np.tanh(ff+eps);density=np.exp(-z-np.exp(-z))
            return np.stack([t,eps*t,t*t])*density
        v,err=quad_vec(integrand,*interval,epsabs=tol,epsrel=tol,norm='max')
        result.append(v)
    t,et,t2=np.concatenate(result,axis=1)
    mean=a+b*(f+k*t)
    variance=b*b*(sd*sd+2*k*et+k*k*(t2-t*t))
    if np.min(variance)<0: raise FloatingPointError('Negative conditional variance')
    return mean,variance


def oracle_panels(z,settings,out):
    panels={};checks=[]
    for strength in [.15,.4]:
        p=z['test']['e6_panels'][strength]['seen'];targets={}
        for endpoint in ['before','after']:
            aa,bb,kk=cell_parameters(p['contexts'],p['c'],p['t_'+endpoint],z['d'])
            answers=[moments(p['f_'+endpoint],aa,bb,kk,z['dgp'].noise_sd,tol,settings['integration_standard_gumbel_interval']) for tol in settings['integration_tolerances']]
            targets[endpoint]=answers[-1]
            checks.append(dict(strength=strength,endpoint=endpoint,
                max_mean_discrepancy=float(np.max(abs(answers[0][0]-answers[1][0]))),
                max_variance_discrepancy=float(np.max(abs(answers[0][1]-answers[1][1])))))
        panels[strength]=targets
    pd.DataFrame(checks).to_csv(out/'integration_checks.csv',index=False)
    print('Oracle integration precision checks',checks,flush=True)
    return panels


def residual_stats(e0,e1):
    means=[float(e.mean()) for e in [e0,e1]];variances=[float(e.var()) for e in [e0,e1]]
    product=float(np.mean(e0*e1));cov=float(np.mean((e0-means[0])*(e1-means[1])))
    level=float(np.mean((e0*e0+e1*e1)/2));change=float(np.mean((e1-e0)**2))
    rhs=2*level-2*product
    np.testing.assert_allclose(change,rhs,atol=2e-13,rtol=1e-13)
    return dict(residual_mean_0=means[0],residual_mean_1=means[1],residual_variance_0=variances[0],residual_variance_1=variances[1],
        residual_covariance=cov,residual_cross_product=product,level_MSE=level,change_MSE=change,
        change_bias_squared=(means[1]-means[0])**2,change_variance=variances[0]+variances[1]-2*cov,identity_discrepancy=change-rhs)


@torch.no_grad()
def outcome_evaluation(states,z,oracle,out,prefix,steps):
    rows=[];decomp=[];score_rows=[]
    model=copy.deepcopy(z['A']);head=copy.deepcopy(z['head'])
    for key,state in states.items():
        arm,step=key.rsplit('_',1);step=int(step);model.load_state_dict(state['model']);head.load_state_dict(state['head'])
        predict=lambda x:diag.core.predictions(model,z['features'],x,z['cfg'])
        ss=predict(z['design']['train']['x']);variance=float(cell_variance(torch.tensor(ss),torch.tensor(z['design']['train']['cell'])))
        for strength in [.15,.4]:
            p=z['test']['e6_panels'][strength]['seen'];s0=predict(p['x_before']);s1=predict(p['x_after'])
            w0=head(diag.core.tensor(s0,z['cfg']),diag.core.tensor(p['cell_before'],z['cfg'],True)).numpy()
            w1=head(diag.core.tensor(s1,z['cfg']),diag.core.tensor(p['cell_after'],z['cfg'],True)).numpy()
            m0,v0=oracle[strength]['before'];m1,v1=oracle[strength]['after']
            changed=abs(p['delta_f'])>z['d']['truth_threshold'];stable=~p['assigned_changed']
            score_rows.append(dict(arm=arm,step=step,strength=strength,train_score_variance=variance,
                raw_stable_score_RMS=float(np.sqrt(np.mean((s1[stable]-s0[stable])**2))),
                normalized_stable_RMS=float(np.sqrt(np.mean((s1[stable]-s0[stable])**2)/variance))))
            for group,mask in [('all',np.ones(len(s0),bool)),('stable',stable),('small' if strength==.15 else 'moderate',changed)]:
                mean_level=float(np.mean(np.r_[(w0-m0)[mask],(w1-m1)[mask]]**2))
                mean_change=float(np.mean(((w1-w0)-(m1-m0))[mask]**2))
                noise_level=float(np.mean((v0[mask]+v1[mask])/2));noise_change=2*noise_level
                realized_level=float(np.mean(np.r_[(w0-p['w_before'])[mask],(w1-p['w_after'])[mask]]**2))
                realized_change=float(np.mean(((w1-w0)-(p['w_after']-p['w_before']))[mask]**2))
                row=dict(arm=arm,step=step,strength=strength,group=group,n=int(mask.sum()),mean_level_MSE=mean_level,mean_change_MSE=mean_change,
                    realized_level_MSE=realized_level,realized_change_MSE=realized_change,noise_floor_level=noise_level,noise_floor_change=noise_change,
                    expected_noisy_level_risk=mean_level+noise_level,expected_noisy_change_risk=mean_change+noise_change)
                for k in list(row):
                    if k.endswith('_MSE') or k.startswith('noise_floor') or k.startswith('expected_noisy'):
                        row[k+'_standardized']=row[k]/z['den']['delta_W' if 'change' in k else 'W']
                rows.append(row)
                for target,tt0,tt1 in [('realized',p['w_before'],p['w_after']),('conditional_mean',m0,m1)]:
                    decomp.append(dict(arm=arm,step=step,strength=strength,group=group,target=target,**residual_stats((w0-tt0)[mask],(w1-tt1)[mask])))
    # A was saved only at zero; it is fixed and is explicitly repeated, never retrained.
    for collection in [rows,decomp,score_rows]:
        aa=[r for r in collection if r['arm']=='A']
        collection.extend(dict(r,step=step) for r in aa for step in steps if step!=0)
    pd.DataFrame(rows).to_csv(out/(prefix+'_outcome_trajectories.csv'),index=False)
    pd.DataFrame(decomp).to_csv(out/(prefix+'_residual_decomposition.csv'),index=False)
    pd.DataFrame(score_rows).to_csv(out/(prefix+'_score_scales.csv'),index=False)
    endpoints=pd.DataFrame(rows).query('step == 200')
    endpoints.to_csv(out/(prefix+'_endpoint_outcomes.csv'),index=False)
    return endpoints


def run(args):
    out=args.out.resolve();source=args.source.resolve();settings=json.loads(args.config.read_text())
    if out.exists() and any(out.iterdir()): raise ValueError('Choose a new output directory')
    out.mkdir(parents=True,exist_ok=True);(out/'config.json').write_text(json.dumps(settings,indent=2))
    checks={str(p):digest(p) for p in [source/'checkpoints.pt',source/'config.json',args.run/'manifest.json']}
    saved=torch.load(source/'checkpoints.pt',map_location='cpu',weights_only=False);z=reconstruct(source,saved)
    print('Exact data, feature transforms, denominators and baseline losses reproduced',flush=True)
    # Run observed-only training before computing oracle tables, preventing any oracle feedback.
    states,refs,prep=fit_branches(z,out,settings)
    oracle=oracle_panels(z,settings,out)
    outcome_evaluation(saved['checkpoints'],z,oracle,out,'experiment1',settings['checkpoints'])
    outcome_evaluation(states,z,oracle,out,'experiment2',settings['checkpoints'])
    all_latent=pd.read_csv(out/'latent_trajectory_metrics.csv')
    all_latent.query('step == 200').to_csv(out/'experiment2_endpoint_latent.csv',index=False)
    old=pd.read_csv(source/'trajectory_metrics.csv');f=all_latent.query("arm == 'F'")
    # Reuse all original independent seen/unseen scorer evaluations without refitting.
    old_A=old[old.arm=='A']
    old=pd.concat([old]+[old_A.assign(step=step) for step in settings['checkpoints'] if step!=0],ignore_index=True)
    old.to_csv(out/'experiment1_latent_trajectory_metrics.csv',index=False)
    keys=['change_rmse','stable_rms','threshold','sensitivity','false_positive_rate','slope','correlation','observed_rank_accuracy','latent_rank_accuracy']
    for _,r in f.iterrows():
        ref=old[(old.arm=='A')&(old.strength==r.strength)&(old['where']==r['where'])].iloc[0]
        np.testing.assert_array_equal(r[keys].to_numpy(float),ref[keys].to_numpy(float))
    for p,h in checks.items(): assert digest(Path(p))==h
    trace=pd.read_csv(out/'optimizer_trace.csv');q=trace[trace.arm=='Q']
    metadata=dict(source=str(source),source_hashes_unchanged=checks,denominators=z['den'],head_preparation=prep,
        reference_constraints=refs,exact_data_reproduced=True,F_A_exact_scorer_and_metric_equality=True,
        Q_accepted=int(q.accepted.sum()),Q_rejected=int((~q.accepted).sum()),Q_backtracking_evaluations=int(q.backtracking_evaluations.sum()),
        nominal_scorer_attempts_per_joint_arm=200,head_updates_per_branch=200,
        oracle_used_for_training_or_selection=False,oracle_computed_after_training=True,
        optimizer='Separate Adam scorer/head. Q projects nominal Adam displacement onto linear halfspaces; fallback projected steepest descent if not descending. Actual nonlinear constraint plus cardinal Armijo checks, max 12 evaluations. Rejected proposals roll back parameters AND Adam moments/counter; accepted backtracked proposals retain nominal-gradient Adam moments. Head state advances once per attempt independently.',
        integration='Adaptive deterministic quad_vec over standard Gumbel z in [-12,45], eps 1e-9 and 1e-11; analytic centered error mean and variance; omitted right tail probability < 2.87e-20. No simulation integration.',
        labels='Same independent date-specific centered Gumbel errors and exact original paired data; no extra seeds or searches')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    with threadpool_limits(limits=1):
        torch.set_num_threads(1);run(args)
