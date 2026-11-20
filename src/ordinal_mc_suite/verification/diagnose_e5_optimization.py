"""Single-sample E5 optimization diagnostics; does not rerun the MC battery.

From the suite: python verification/diagnose_e5_optimization.py
The sample, criterion, architecture and penalties match E5. Only optimizer paths
and training-label-based initializations vary. Test metrics never select a fit.
"""
from __future__ import annotations
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
from torch.nn.utils import parameters_to_vector, vector_to_parameters

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ordinal_mc import core
from ordinal_mc.config import runtime
from ordinal_mc.design import PopulationDGP, make_design
from ordinal_mc.experiments import scenario
from ordinal_mc.likelihood import fit_ml


def active_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def set_effective(layer,weight,bias,bound):
    if bool((weight.abs()>=bound).any()): raise ValueError('Embedding exceeds coefficient bound')
    with torch.no_grad():
        layer.weight.copy_(bound*torch.atanh(weight/bound))
        if layer.bias is not None:
            layer.bias.copy_(bound*torch.atanh(bias/bound))


def embed_linear(linear,cfg,seed=987):
    """Exact linear function in the same bounded ReLU NN, without oracle f.

Paired positive/negative input channels pass through two ReLU layers. Spare
units retain random incoming weights and zero outgoing weights, so capacity can
start learning without changing the initial linear score.
"""
    torch.manual_seed(seed)
    dimension=linear.scorer.layers[0].in_features
    model=core.Estimator(dimension,'nn','common_logit',cfg).to(device=cfg.device,dtype=cfg.torch_dtype)
    layers=model.scorer.layers; bound=cfg.weight_bound; channels=2*dimension
    if len(layers)!=3 or min(layers[0].out_features,layers[1].out_features)<channels:
        raise ValueError('Embedding requires two hidden layers with at least 2*input_dimension units')
    w=linear.scorer.effective(linear.scorer.layers[0].weight).detach()[0]
    first=model.scorer.effective(layers[0].weight).detach().clone()
    first_bias=model.scorer.effective(layers[0].bias).detach().clone()
    first[:channels]=0.; first_bias[:channels]=0.
    for j in range(dimension): first[2*j,j]=1.; first[2*j+1,j]=-1.
    set_effective(layers[0],first,first_bias,bound)
    second=model.scorer.effective(layers[1].weight).detach().clone()
    second_bias=model.scorer.effective(layers[1].bias).detach().clone()
    second[:channels]=0.; second_bias[:channels]=0.
    for j in range(channels): second[j,j]=1.
    set_effective(layers[1],second,second_bias,bound)
    final=torch.zeros_like(layers[-1].weight)
    for j in range(dimension): final[0,2*j]=w[j]; final[0,2*j+1]=-w[j]
    set_effective(layers[-1],final,None,bound)
    return model


def numpy_objective(model,data,edges,lam):
    """Independent NumPy forward/loss calculation of the E5 common-logit Q."""
    x=data['x'].detach().cpu().numpy().copy(); bound=model.scorer.weight_bound
    for k,layer in enumerate(model.scorer.layers):
        w=bound*np.tanh(layer.weight.detach().cpu().numpy()/bound)
        b=bound*np.tanh(layer.bias.detach().cpu().numpy()/bound) if layer.bias is not None else 0.
        x=x@w.T+b
        if k<len(model.scorer.layers)-1: x=np.maximum(x,0)
    s=x[:,0]+float(model.level_bias.detach())
    i=data['i'].cpu().numpy(); j=data['j'].cpu().numpy()
    y=data['y'].cpu().numpy(); weights=data['weights'].cpu().numpy()
    gap=s[j]-s[i]; variance=.5*np.sum(weights*gap**2)
    cross=np.sum(weights*np.logaddexp(0,-y*gap)*(y!=0))
    stable=0.
    for arm,weight in zip(('temporal','cross_city'),lam):
        e=edges[arm].cpu().numpy()
        if weight and len(e): stable+=weight*np.mean((s[e[:,0]]-s[e[:,1]])**2)/variance
    return np.array([cross+stable,cross,stable,variance])


def explicit_penalty_gradient(score,data,edges,lam):
    """Analytic score-space derivative of S/V, including the denominator."""
    s=np.asarray(score,float); i=data['i'].cpu().numpy(); j=data['j'].cpu().numpy()
    weights=data['weights'].cpu().numpy(); gap=s[j]-s[i]
    variance=.5*np.sum(weights*gap**2); dv=np.zeros_like(s)
    np.add.at(dv,j,weights*gap); np.add.at(dv,i,-weights*gap)
    total=0.; ds=np.zeros_like(s)
    for arm,weight in zip(('temporal','cross_city'),lam):
        e=edges[arm].cpu().numpy()
        if not weight or not len(e): continue
        a,b=e.T; difference=s[a]-s[b]
        total+=weight*np.mean(difference**2)
        np.add.at(ds,a,2*weight*difference/len(e)); np.add.at(ds,b,-2*weight*difference/len(e))
    return ds/variance-total*dv/variance**2


def gradient_checks(model,data,edges,lam,directions=5):
    q,parts=core.score_objective(model,data,edges,lam,0.,.1)
    reference=numpy_objective(model,data,edges,lam)
    torch_values=np.array([float(q.detach()),float(parts[0].detach()),float(parts[1].detach()),float(parts[3].detach())])
    np.testing.assert_allclose(torch_values,reference,rtol=1e-11,atol=1e-12)
    params=active_parameters(model); model.zero_grad(set_to_none=True); q.backward()
    # Parameters unused by Q receive exactly zero in the vector derivative.
    grad=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
    original=parameters_to_vector(params).detach().clone()
    rng=np.random.default_rng(701); results=[]
    try:
        for k in range(directions):
            direction=torch.as_tensor(rng.normal(size=len(original)),device=original.device,dtype=original.dtype)
            direction/=direction.norm(); analytic=float(grad@direction)
            errors=[]
            for eps in (1e-5,1e-6,1e-7):
                vector_to_parameters(original+eps*direction,params)
                plus=float(core.score_objective(model,data,edges,lam,0.,.1)[0].detach())
                vector_to_parameters(original-eps*direction,params)
                minus=float(core.score_objective(model,data,edges,lam,0.,.1)[0].detach())
                numeric=(plus-minus)/(2*eps)
                errors.append(dict(epsilon=eps,analytic=analytic,numeric=numeric,
                                   scaled_error=abs(numeric-analytic)/(1+abs(analytic))))
            best=min(errors,key=lambda row:row['scaled_error'])
            if best['scaled_error']>2e-7: raise AssertionError(f'NN directional gradient mismatch: {best}')
            results.append(dict(direction=k,**best))
    finally: vector_to_parameters(original,params)
    score=model.scorer(data['x']).detach().clone().requires_grad_(True)
    gap=score[data['j']]-score[data['i']]
    variance=.5*(data['weights']*gap.square()).sum()
    raw=score.new_tensor(0.)
    for arm,weight in zip(('temporal','cross_city'),lam):
        e=edges[arm]
        if weight and len(e): raw=raw+weight*(score[e[:,0]]-score[e[:,1]]).square().mean()
    numerical_gradient=torch.autograd.grad(raw/variance,score)[0].detach().cpu().numpy()
    analytic_gradient=explicit_penalty_gradient(score.detach().cpu().numpy(),data,edges,lam)
    np.testing.assert_allclose(numerical_gradient,analytic_gradient,rtol=1e-10,atol=1e-12)
    return dict(numpy_objective_max_error=float(np.max(np.abs(reference-torch_values))),
                directional_checks=results,penalty_gradient_max_error=float(np.max(abs(numerical_gradient-analytic_gradient))))


def metrics(model,data,edges,lam,features,design,cfg):
    model.zero_grad(set_to_none=True)
    q,parts=core.score_objective(model,data,edges,lam,0.,cfg.tau); q.backward()
    grad=max(float(p.grad.abs().max()) for p in model.parameters() if p.grad is not None and p.numel())
    with torch.no_grad():
        s=model.scorer(data['x']); gap=s[data['j']]-s[data['i']]
        train_error=core.misranking(gap.cpu().numpy(),data['y'].cpu().numpy())
        raw_link={}
        for arm in ('temporal','cross_city'):
            e=edges[arm]
            raw_link[arm+'_raw_gap_mse']=float((s[e[:,0]]-s[e[:,1]]).square().mean()) if len(e) else 0.
        effective=torch.cat([model.scorer.effective(layer.weight).reshape(-1) for layer in model.scorer.layers])
    return dict(objective=float(q.detach()),cross_loss=float(parts[0].detach()),stable_penalty=float(parts[1].detach()),
                score_variance=float(parts[3].detach()),gradient_inf=grad,train_misranking=train_error,
                effective_weight_saturation_share=float((effective.abs()>.95*cfg.weight_bound).double().mean()),
                validation_error=core.validation_error(model,features,design,cfg),
                **raw_link,**core.test_metrics(model,features,design,cfg))


def continue_lbfgs(model,data,edges,lam,cfg,chunks,chunk_steps,callback):
    """Keep optimizer history across chunks; preserve the pre-step admissible state."""
    params=active_parameters(model)
    opt=torch.optim.LBFGS(params,lr=1.,max_iter=chunk_steps,tolerance_grad=cfg.grad_tol,
        tolerance_change=1e-10,history_size=20,line_search_fn='strong_wolfe')
    for chunk in range(1,chunks+1):
        before=core.cpu_state(model); optimizer_before=copy.deepcopy(opt.state_dict())
        before_value=float(core.score_objective(model,data,edges,lam,0.,cfg.tau)[0].detach())
        state=opt.state[params[0]]; old_iterations=state.get('n_iter',0); calls=0
        def closure():
            nonlocal calls
            calls+=1; opt.zero_grad(set_to_none=True)
            value,_=core.score_objective(model,data,edges,lam,0.,cfg.tau)
            if not torch.isfinite(value): raise FloatingPointError('Nonfinite L-BFGS probe')
            value.backward(); return value
        status='ok'
        try:
            opt.step(closure)
            after=float(core.score_objective(model,data,edges,lam,0.,cfg.tau)[0].detach())
            if not np.isfinite(after) or after>before_value+1e-12:
                status='worse_endpoint_restored'; model.load_state_dict(before); opt.load_state_dict(optimizer_before)
        except (core.InadmissibleScoreError,FloatingPointError):
            status='invalid_probe_restored'; model.load_state_dict(before); opt.load_state_dict(optimizer_before)
        state=opt.state[params[0]]
        callback(model,chunk,dict(chunk_steps=chunk_steps,closure_evaluations=calls,
            actual_iterations=state.get('n_iter',0)-old_iterations,optimizer_status=status))
        if status!='ok': break
    return opt


def adam_continuation(model,data,edges,lam,cfg,steps,callback):
    opt=torch.optim.Adam(active_parameters(model),lr=.001)
    best=float(core.score_objective(model,data,edges,lam,0.,cfg.tau)[0].detach()); best_state=core.cpu_state(model)
    for step in range(1,steps+1):
        before=core.cpu_state(model); opt.zero_grad(set_to_none=True)
        value,_=core.score_objective(model,data,edges,lam,0.,cfg.tau); value.backward()
        torch.nn.utils.clip_grad_norm_(active_parameters(model),20.); opt.step()
        try: after=float(core.score_objective(model,data,edges,lam,0.,cfg.tau)[0].detach())
        except core.InadmissibleScoreError:
            model.load_state_dict(before); break
        if not np.isfinite(after): model.load_state_dict(before); break
        if after<best: best=after; best_state=core.cpu_state(model)
        if step in (100,300,steps):
            current=core.cpu_state(model); model.load_state_dict(best_state)
            callback(model,step,dict(optimizer_status='best_training_objective_so_far',adam_lr=.001))
            model.load_state_dict(current)
    model.load_state_dict(best_state)


def plot_trace(frame,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'none':'#677176','temporal':'#7459B4','both':'#2F7899'}
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for arm,color in colors.items():
        for path,style in [('random_best_lbfgs','-'),('linear_start_lbfgs','--')]:
            rows=frame[(frame.arm==arm)&(frame.path==path)].sort_values('extra_iterations')
            label=arm+(' · random start' if path=='random_best_lbfgs' else ' · linear start')
            axes[0].plot(rows.extra_iterations,rows.objective,style+'o',color=color,label=label,ms=3)
            axes[1].plot(rows.extra_iterations,100*rows.unseen_change_recall,style+'o',color=color,ms=3)
            axes[2].plot(rows.extra_iterations,100*rows.unseen_within_latent,style+'o',color=color,ms=3)
    axes[0].set(yscale='log',ylabel='Training objective (log scale)')
    axes[1].set(ylabel='Changed tracts detected (%)',ylim=(0,104))
    axes[2].set(ylabel='Test latent misranking (%)',ylim=(0,None))
    for ax in axes:
        ax.set_xlabel('Additional L-BFGS iteration budget'); ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.5,-.03))
    fig.suptitle('One fixed E5 sample · linear truth · clean links · change step 0.4',fontsize=12)
    fig.tight_layout(rect=(0,.15,1,.95))
    fig.savefig(out/'optimization_paths.png',dpi=180,bbox_inches='tight')
    fig.savefig(out/'optimization_paths.pdf',bbox_inches='tight'); plt.close(fig)


def run(args):
    source=args.run.resolve(); out=args.out.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Choose a new diagnostic output directory')
    out.mkdir(parents=True,exist_ok=True)
    original=source/'selected_fits.csv'; original_sha=hashlib.sha256(original.read_bytes()).hexdigest()
    d=json.loads((source/'resolved_config.json').read_text()); d.update(device='cpu',dtype='float64',workers=1,threads=1)
    cfg=runtime(d); cfg.fit_cell_effects=False
    s=scenario(link_case='changed',link_severity=0.,change_strength=.4,
        acquisition_strength=cfg.acquisition_strength,group_strength=cfg.group_strength)
    dgp=PopulationDGP(cfg); design=make_design(cfg,dgp,args.rep,32,'linear',s)
    saved=pd.read_csv(original); saved=saved[(saved.rep==args.rep)&(saved.truth=='linear')&(saved.change_strength==.4)&(saved.link_severity==0)]
    if saved.empty: raise ValueError('Requested fixed sample is not present in the saved E5 results')
    for field,value in [('n_fit',design['n_fit']),('n_validation',design['n_val']),
                        ('n_rank_pairs',len(design['pairs'][0])),('noise_sd_effective',design['noise_sd_effective'])]:
        available=saved[field].dropna()  # ML does not use or report rank-pair counts.
        if available.empty: raise ValueError(f'Missing saved sample diagnostic: {field}')
        np.testing.assert_allclose(available,value,rtol=1e-12,atol=1e-12)
    for name,(test,pairs) in design['tests'].items():
        i,j=pairs['within']; actual=core.misranking(test['f'][j]-test['f'][i],test['w'][j]-test['w'][i])
        np.testing.assert_allclose(saved[name+'_oracle_observed'],actual,rtol=0,atol=1e-14)
    features=core.Features(design['train']['x'],'nn',cfg); data=core.prepare(design,features,cfg)
    metadata=dict(replication=args.rep,truth='linear',n=32,cities=cfg.cities,change_strength=.4,
        contamination=0.,criterion='common_logit + lambda_t*S_t/V + lambda_c*S_c/V; ridge=0',
        original_csv_sha256=original_sha,config=d,python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
        original_runtime=json.loads((source/'run_metadata.json').read_text()),
        test_metrics_used_for_selection=False)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    records=[]; baselines=[]; checkpoints=out/'checkpoints'; checkpoints.mkdir()
    def record(model,arm,path,iterations,working,edges,lam,**extra):
        row=dict(arm=arm,path=path,extra_iterations=iterations,**extra,
                 **metrics(model,working,edges,lam,features,design,cfg))
        records.append(row); pd.DataFrame(records).to_csv(out/'trace.csv',index=False)
        print(f'{arm:8s} {path:22s} extra={iterations:4d} Q={row["objective"]:.6g} grad={row["gradient_inf"]:.3g} recall={row["unseen_change_recall"]:.3f} test_error={row["unseen_within_latent"]:.3f}',flush=True)
        return row
    # Tests use the actual full-size NN and actual E5 score inputs.
    torch.manual_seed(314)
    probe=core.Estimator(data['x'].shape[1],'nn','common_logit',cfg).double()
    probe_edges={arm:core.tensor(design['links'][arm],cfg,True) for arm in ('temporal','cross_city')}
    checks=gradient_checks(probe,data,probe_edges,(.3,.3))
    (out/'gradient_checks.json').write_text(json.dumps(checks,indent=2))
    for arm,lam in [('none',(0.,0.)),('temporal',(.3,0.)),('both',(.3,.3))]:
        edges={name:core.tensor(design['links'][name] if weight else design['links']['none'],cfg,True)
               for name,weight in zip(('temporal','cross_city'),lam)}
        working=data if any(lam) else {**data,'x':data['x'][:design['n_fit']]}
        models=[]
        for restart in range(args.starts):
            seed=core.seed_for(cfg.seed,'fit',args.rep,32,cfg.cities,'linear','nn',restart)
            model,stats=core.train_candidate(working,edges,'nn','common_logit',0.,lam,seed,cfg)
            row=record(model,arm,'original_budget_restart',0,working,edges,lam,restart=restart,seed=seed,
                original_audit_drop=stats['audit_objective_drop'])
            baselines.append(row); models.append((row['objective'],restart,model))
        models.sort(key=lambda item:item[0]); best=copy.deepcopy(models[0][2]); selected_restart=models[0][1]
        torch.save(dict(model_state=core.cpu_state(best),restart=selected_restart),checkpoints/f'{arm}_baseline.pt')
        # A convergent training-label-only linear control under exactly the same Q.
        seed=core.seed_for(cfg.seed,'fit',args.rep,32,cfg.cities,'linear','linear',0)
        linear,stats=core.train_candidate(working,edges,'linear','common_logit',0.,lam,seed,cfg)
        record(linear,arm,'linear_control',0,working,edges,lam,original_audit_drop=stats['audit_objective_drop'])
        warm=embed_linear(linear,cfg)
        np.testing.assert_allclose(core.predictions(warm,features,design['x'],cfg),
                                   core.predictions(linear,features,design['x'],cfg),atol=2e-12,rtol=1e-12)
        np.testing.assert_allclose(float(core.score_objective(warm,working,edges,lam,0.,cfg.tau)[0].detach()),
                                   float(core.score_objective(linear,working,edges,lam,0.,cfg.tau)[0].detach()),atol=1e-12)
        torch.save(dict(model_state=core.cpu_state(warm)),checkpoints/f'{arm}_linear_start.pt')
        for path,start in [('random_best_lbfgs',copy.deepcopy(best)),('linear_start_lbfgs',warm)]:
            record(start,arm,path,0,working,edges,lam)
            def callback(m,chunk,info):
                record(m,arm,path,chunk*args.chunk_steps,working,edges,lam,**info)
            opt=continue_lbfgs(start,working,edges,lam,cfg,args.chunks,args.chunk_steps,callback)
            torch.save(dict(model_state=core.cpu_state(start),optimizer_state=opt.state_dict()),checkpoints/f'{arm}_{path}_final.pt')
            # A fresh-history audit of the endpoint is separate from continuation.
            before=float(core.score_objective(start,working,edges,lam,0.,cfg.tau)[0].detach())
            continue_lbfgs(start,working,edges,lam,cfg,1,100,lambda *x:None)
            after=float(core.score_objective(start,working,edges,lam,0.,cfg.tau)[0].detach())
            record(start,arm,path+'_audit',args.chunks*args.chunk_steps+100,working,edges,lam,audit_objective_drop=max(0.,before-after))
        alternative=copy.deepcopy(best)
        record(alternative,arm,'adam_continuation',0,working,edges,lam)
        adam_continuation(alternative,working,edges,lam,cfg,args.adam_steps,
            lambda m,step,info:record(m,arm,'adam_continuation',step,working,edges,lam,**info))
        torch.save(dict(model_state=core.cpu_state(alternative)),checkpoints/f'{arm}_adam_final.pt')
        # Extend another starting point, retaining a fixed optimizer budget.
        if arm!='none' and len(models)>1:
            other=copy.deepcopy(models[1][2]); path='runner_up_lbfgs'
            record(other,arm,path,0,working,edges,lam,restart=models[1][1])
            continue_lbfgs(other,working,edges,lam,cfg,args.chunks,args.chunk_steps,
                lambda m,chunk,info:record(m,arm,'runner_up_lbfgs',chunk*args.chunk_steps,working,edges,lam,**info))
    ml,info=fit_ml(design,'linear',s,dgp)
    (out/'ml_reference.json').write_text(json.dumps({**info,**core.evaluate_predictions(ml,design,cfg,absolute_scale=True)},indent=2))
    if hashlib.sha256(original.read_bytes()).hexdigest()!=original_sha: raise AssertionError('Original E5 results changed')
    frame=pd.DataFrame(records); frame.to_csv(out/'trace.csv',index=False)
    pd.DataFrame(baselines).to_csv(out/'restart_comparison.csv',index=False)
    # Every L-BFGS trace should be nonincreasing in the same objective.
    for (arm,path),group in frame.groupby(['arm','path']):
        if path.endswith('lbfgs'):
            values=group.sort_values('extra_iterations').objective.to_numpy()
            if np.any(np.diff(values)>1e-10): raise AssertionError(f'Nonmonotone accepted L-BFGS path: {arm}/{path}')
    plot_trace(frame,out)
    summary=dict(completed=True,original_results_unchanged=True,sample_replay_verified=True,
        gradient_tests_passed=True,linear_embedding_test_passed=True,accepted_lbfgs_paths_nonincreasing=True,
        candidate_fits=len(baselines)+3,recorded_checkpoints=len(records),full_monte_carlo_run=False)
    (out/'verification.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'change_recovery')
    p.add_argument('--out',type=Path,default=ROOT/'optimization_single_run')
    p.add_argument('--rep',type=int,default=1); p.add_argument('--starts',type=int,default=3)
    p.add_argument('--chunks',type=int,default=5); p.add_argument('--chunk-steps',type=int,default=200)
    p.add_argument('--adam-steps',type=int,default=1000)
    args=p.parse_args()
    if min(args.rep,args.starts,args.chunks,args.chunk_steps,args.adam_steps)<1: p.error('Counts must be positive')
    run(args)
