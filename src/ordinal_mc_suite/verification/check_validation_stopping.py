"""Single-sample comparison with validation-selected early stopping.

No full Monte Carlo run. All methods share a 20%-target tract holdout, and test
truths are evaluated only after checkpoint/restart selection has finished.
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
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import diagnose_e5_optimization as base
core=base.core
ROOT=base.ROOT


def prepare_validation(design,features,cfg):
    """All ordered within-cell validation comparisons; neither f nor r is read."""
    val=design['val']; ii=[]; jj=[]; weights=[]; count=len(val['x'])
    for cell in np.unique(val['cell']):
        loc=np.flatnonzero(val['cell']==cell)
        a=np.repeat(loc,len(loc)); b=np.tile(loc,len(loc)); keep=a!=b
        ii.extend(a[keep]); jj.extend(b[keep]); weights.extend([1/(count*(len(loc)-1))]*int(keep.sum()))
    i=np.asarray(ii,int); j=np.asarray(jj,int)
    return dict(x=core.tensor(features(val['x']),cfg),i=core.tensor(i,cfg,True),
                j=core.tensor(j,cfg,True),y=core.tensor(np.sign(val['w'][j]-val['w'][i]),cfg),
                weights=core.tensor(np.asarray(weights),cfg))


@torch.no_grad()
def validation(model,val):
    s=model.scorer(val['x']); gap=s[val['j']]-s[val['i']]
    loss=(val['weights']*F.softplus(-val['y']*gap)*(val['y']!=0)).sum()
    error=core.misranking(gap.cpu().numpy(),val['y'].cpu().numpy())
    return float(loss),error


class ValidationTracker:
    """Always retain the lowest validation loss; patience uses min_delta."""
    def __init__(self,patience,min_delta=1e-5):
        self.patience=patience; self.min_delta=min_delta; self.best_loss=float('inf')
        self.reference=float('inf'); self.bad_checks=0; self.best_state=None; self.best_location=None

    def observe(self,model,loss,location):
        if not np.isfinite(loss): raise FloatingPointError('Nonfinite validation loss')
        if loss<self.best_loss:
            self.best_loss=loss; self.best_state=core.cpu_state(model); self.best_location=location
        if loss<self.reference-self.min_delta: self.reference=loss; self.bad_checks=0
        else: self.bad_checks+=1
        return self.bad_checks>=self.patience

    def restore(self,model):
        if self.best_state is None: raise RuntimeError('No validation checkpoint')
        model.load_state_dict(self.best_state)


def early_stop(model,data,edges,lam,val,cfg,*,max_adam=3000,check_every=10,patience=20,
               max_lbfgs=400,lbfgs_chunk=10,lbfgs_patience=10):
    """Train unchanged Q, select only observed validation BCE, restore best state.

No test data, latent labels or change metrics enter this function. Each optimizer
stage has its own patience counter, but both use the same best-checkpoint tracker.
The final restored checkpoint is not subsequently refined by an unvalidated audit.
"""
    tracker=ValidationTracker(patience); trace=[]
    def observe(stage,step):
        loss,error=validation(model,val)
        with torch.no_grad(): q,parts=core.score_objective(model,data,edges,lam,0.,cfg.tau)
        stop=tracker.observe(model,loss,(stage,step))
        trace.append(dict(stage=stage,step=step,train_objective=float(q),validation_loss=loss,
                          validation_error=error,best_validation_loss=tracker.best_loss,
                          score_variance=float(parts[3]),bad_checks=tracker.bad_checks))
        return stop
    observe('initial',0)
    opt=torch.optim.Adam(base.active_parameters(model),lr=cfg.lr)
    reason='maximum_adam_steps'
    for step in range(1,max_adam+1):
        before=core.cpu_state(model); opt.zero_grad(set_to_none=True)
        q,_=core.score_objective(model,data,edges,lam,0.,cfg.tau); q.backward()
        torch.nn.utils.clip_grad_norm_(base.active_parameters(model),20.); opt.step()
        try:
            value=core.score_objective(model,data,edges,lam,0.,cfg.tau)[0]
            if not torch.isfinite(value): raise FloatingPointError('Nonfinite training objective')
        except (core.InadmissibleScoreError,FloatingPointError):
            model.load_state_dict(before); reason='invalid_adam_step'; break
        if step%check_every==0 or step==max_adam:
            if observe('adam',step): reason='adam_validation_patience'; break
    tracker.restore(model)
    adam_stop=reason
    if max_lbfgs:
        tracker.bad_checks=0; tracker.reference=tracker.best_loss; tracker.patience=lbfgs_patience
        opt=torch.optim.LBFGS(base.active_parameters(model),lr=1.,max_iter=lbfgs_chunk,
            tolerance_grad=cfg.grad_tol,tolerance_change=1e-10,history_size=20,line_search_fn='strong_wolfe')
        reason='maximum_lbfgs_steps'
        for step in range(lbfgs_chunk,max_lbfgs+1,lbfgs_chunk):
            before=core.cpu_state(model); before_opt=copy.deepcopy(opt.state_dict())
            before_q=float(core.score_objective(model,data,edges,lam,0.,cfg.tau)[0].detach())
            def closure():
                opt.zero_grad(set_to_none=True)
                value,_=core.score_objective(model,data,edges,lam,0.,cfg.tau)
                if not torch.isfinite(value): raise FloatingPointError('Nonfinite L-BFGS probe')
                value.backward(); return value
            try:
                opt.step(closure)
                after=float(core.score_objective(model,data,edges,lam,0.,cfg.tau)[0].detach())
                if not np.isfinite(after) or after>before_q+1e-12: raise FloatingPointError('Worse L-BFGS endpoint')
            except (core.InadmissibleScoreError,FloatingPointError):
                model.load_state_dict(before); opt.load_state_dict(before_opt); reason='invalid_lbfgs_step'; break
            if observe('lbfgs',step): reason='lbfgs_validation_patience'; break
    else: reason='lbfgs_not_requested'
    tracker.restore(model)
    selected_loss,selected_error=validation(model,val)
    np.testing.assert_allclose(selected_loss,min(r['validation_loss'] for r in trace),atol=1e-12,rtol=1e-12)
    return model,dict(validation_loss=selected_loss,selection_validation_error=selected_error,
        best_stage=tracker.best_location[0],best_step=tracker.best_location[1],
        adam_stop_reason=adam_stop,lbfgs_stop_reason=reason),trace


def choose_validation(candidates):
    return min(candidates,key=lambda x:(x['info']['validation_loss'],x['restart']))


def case_scenario(measurement='affine',baseline_scale=10.):
    settings=dict(link_case='changed',link_severity=0.,change_strength=.4)
    if measurement=='exponential_additive':
        settings.update(positive_labels=True,psi_stage='additive',psi_strength=baseline_scale,psi_delta=0.)
    elif measurement!='affine': raise ValueError('Unknown measurement specification')
    return base.scenario(**settings)


def log_fe_prediction(design):
    """Exact common slope from log W with cell fixed effects; fitting labels only."""
    from e1_additive_exact import log_fe_beta
    tr=design['train']
    if np.any(tr['w']<=0): raise ValueError('Log + FE requires positive observations')
    beta=log_fe_beta(tr['x'],np.log(tr['w']),tr['cell'])
    return lambda x:np.asarray(x)@beta


def plot_results(frame,out,title=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    choices=[('fixed_budget_train_selected','Old stopping rule','#677176'),
             ('early_stopped_random','Validation stopping; random starts','#7459B4'),
             ('early_stopped_linear_start','Validation stopping; linear start','#2F7899')]
    fig,axes=plt.subplots(1,2,figsize=(10,4)); x=np.arange(3); width=.24
    for k,(method,label,color) in enumerate(choices):
        rows=frame[frame.method==method].set_index('arm').reindex(['none','temporal','both'])
        axes[0].bar(x+(k-1)*width,100*rows.unseen_change_recall,width,color=color,label=label)
        axes[1].bar(x+(k-1)*width,100*(1-rows.unseen_within_latent),width,color=color)
    for method,label,color,ls in [('ml_correct','ML (oracle nuisance)','#8F6802',':'),
                                  ('log_cell_fe','Log + cell fixed effects','#C8253D','--')]:
        reference=frame[frame.method==method]
        if len(reference):
            r=reference.iloc[0]
            axes[0].axhline(100*r.unseen_change_recall,color=color,ls=ls,lw=1.7,label=label)
            axes[1].axhline(100*(1-r.unseen_within_latent),color=color,ls=ls,lw=1.7)
    for ax in axes:
        ax.set(xticks=x,xticklabels=['No links','Temporal','Temporal + cross-city'],ylim=(0,104))
        ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Changed tracts detected (%)'); axes[1].set_ylabel('Test latent pair accuracy (%)')
    fig.suptitle(title or 'One E5 sample · linear truth · same validation split for every method',fontsize=12)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2 if len(labels)>3 else 1,bbox_to_anchor=(.5,-.08),frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.22,1,.94))
    fig.savefig(out/'validation_stopping.png',dpi=180,bbox_inches='tight')
    fig.savefig(out/'validation_stopping.pdf',bbox_inches='tight'); plt.close(fig)


def run(args):
    source=args.run.resolve(); out=args.out.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Choose a new output directory')
    out.mkdir(parents=True,exist_ok=True); (out/'checkpoints').mkdir()
    original=source/'selected_fits.csv'; before_hash=hashlib.sha256(original.read_bytes()).hexdigest()
    d=json.loads((source/'resolved_config.json').read_text())
    d.update(validation_fraction=.2,device='cpu',dtype='float64',workers=1,threads=1)
    cfg=base.runtime(d); cfg.fit_cell_effects=False
    truth=getattr(args,'truth','linear'); measurement=getattr(args,'measurement','affine')
    baseline_scale=getattr(args,'baseline_scale',10.)
    s=case_scenario(measurement,baseline_scale)
    dgp=base.PopulationDGP(cfg); design=base.make_design(cfg,dgp,1,32,truth,s)
    assert not set(map(tuple,design['train']['h'])) & set(map(tuple,design['val']['h']))
    features=core.Features(design['train']['x'],'nn',cfg)
    data=core.prepare(design,features,cfg); val=prepare_validation(design,features,cfg)
    metadata=dict(config=d,replication=1,truth=truth,measurement=measurement,scenario=s,
        baseline_scale=baseline_scale if measurement=='exponential_additive' else None,contamination=0.,change_strength=.4,
        n_fit=design['n_fit'],n_validation=design['n_val'],
        realized_validation_fraction=design['n_val']/(design['n_fit']+design['n_val']),
        validation_comparisons=len(val['i']),selection_metric='observed within-cell validation BCE',
        original_csv_sha256=before_hash,python=platform.python_version(),torch=torch.__version__,
        test_used_for_selection=False,original_results_modified=False,
        policy=dict(max_adam=args.max_adam,check_every=10,patience=20,min_delta=1e-5,
                    max_lbfgs=400,lbfgs_chunk=10,lbfgs_patience=10))
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    results=[]; traces=[]; candidates=[]
    def evaluate(model,arm,method,working,edges,lam,**info):
        loss,error=validation(model,val)
        row=dict(arm=arm,method=method,validation_loss=loss,selection_validation_error=error,
                 **base.metrics(model,working,edges,lam,features,design,cfg),**info)
        results.append(row); pd.DataFrame(results).to_csv(out/'results.csv',index=False)
        print(f'{arm:8s} {method:29s} val_loss={loss:.4f} recall={row["unseen_change_recall"]:.3f} test_accuracy={1-row["unseen_within_latent"]:.3f}',flush=True)
    # All candidate checkpoints and all restart choices are finalized before any
    # candidate is evaluated on latent test outcomes.
    selected=[]
    for arm,lam in [('none',(0.,0.)),('temporal',(.3,0.)),('both',(.3,.3))]:
        edges={name:core.tensor(design['links'][name] if weight else design['links']['none'],cfg,True)
               for name,weight in zip(('temporal','cross_city'),lam)}
        working=data if any(lam) else {**data,'x':data['x'][:design['n_fit']]}
        fixed=[]; early=[]
        for restart in range(args.starts):
            seed=core.seed_for(cfg.seed,'fit',1,32,cfg.cities,truth,'nn',restart)
            model,stats=core.train_candidate(working,edges,'nn','common_logit',0.,lam,seed,cfg)
            loss,error=validation(model,val)
            fixed.append(dict(model=model,restart=restart,info=dict(validation_loss=loss,
                selection_validation_error=error,train_objective=stats['train_objective'])))
            torch.manual_seed(seed)
            initial=core.Estimator(data['x'].shape[1],'nn','common_logit',cfg).double()
            model,info,trace=early_stop(initial,working,edges,lam,val,cfg,max_adam=args.max_adam)
            early.append(dict(model=model,restart=restart,info=info))
            traces.extend([dict(arm=arm,initialization='random',restart=restart,**r) for r in trace])
            candidates.extend([dict(arm=arm,policy='fixed_budget',restart=restart,**fixed[-1]['info']),
                               dict(arm=arm,policy='early_stopping',restart=restart,**info)])
            print(f'{arm}: restart {restart} finalized; early best={info["best_stage"]}:{info["best_step"]}, val={info["validation_loss"]:.4f}',flush=True)
        best_fixed=min(fixed,key=lambda x:x['info']['train_objective']); best_early=choose_validation(early)
        seed=core.seed_for(cfg.seed,'fit',1,32,cfg.cities,truth,'linear',0)
        linear,_=core.train_candidate(working,edges,'linear','common_logit',0.,lam,seed,cfg)
        warm=base.embed_linear(linear,cfg)
        np.testing.assert_allclose(core.predictions(warm,features,design['x'],cfg),core.predictions(linear,features,design['x'],cfg),atol=2e-12,rtol=1e-12)
        warm,info,trace=early_stop(warm,working,edges,lam,val,cfg,max_adam=args.max_adam)
        traces.extend([dict(arm=arm,initialization='linear_start',restart=0,**r) for r in trace])
        selected.extend([(arm,'fixed_budget_train_selected',best_fixed['model'],working,edges,lam,dict(restart=best_fixed['restart'])),
                         (arm,'fixed_budget_val_selected',choose_validation(fixed)['model'],working,edges,lam,dict()),
                         (arm,'early_stopped_random',best_early['model'],working,edges,lam,dict(restart=best_early['restart'],**{k:v for k,v in best_early['info'].items() if k not in ('validation_loss','selection_validation_error')})),
                         (arm,'early_stopped_linear_start',warm,working,edges,lam,{k:v for k,v in info.items() if k not in ('validation_loss','selection_validation_error')}),
                         (arm,'linear_control',linear,working,edges,lam,dict())])
    # Test metrics enter only here, after model selection for all arms.
    for arm,method,model,working,edges,lam,info in selected:
        evaluate(model,arm,method,working,edges,lam,**info)
        torch.save(dict(model_state=core.cpu_state(model),selection=info),out/'checkpoints'/f'{arm}_{method}.pt')
    # References are fitted on the same fitting tracts and are not used for NN
    # checkpoint selection. ML knows the generating measurement/noise mechanisms.
    ml,ml_info=base.fit_ml(design,truth,s,dgp)
    results.append(dict(arm='benchmark',method='ml_correct',**ml_info,
                        **core.evaluate_predictions(ml,design,cfg,absolute_scale=True)))
    if measurement=='exponential_additive':
        fe=log_fe_prediction(design)
        results.append(dict(arm='benchmark',method='log_cell_fe',
                            **core.evaluate_predictions(fe,design,cfg)))
    pd.DataFrame(results).to_csv(out/'results.csv',index=False)
    pd.DataFrame(traces).to_csv(out/'validation_history.csv',index=False)
    pd.DataFrame(candidates).to_csv(out/'candidate_selection.csv',index=False)
    if hashlib.sha256(original.read_bytes()).hexdigest()!=before_hash: raise AssertionError('Original E5 results changed')
    title=('Exponential + additive baseline' if measurement=='exponential_additive' else truth.capitalize()+' latent truth')+' · one sample · validation stopping'
    plot_results(pd.DataFrame(results),out,title=title)
    (out/'verification.json').write_text(json.dumps(dict(completed=True,full_mc_run=False,
        original_results_unchanged=True,split_disjoint=True,test_evaluated_only_after_selection=True,
        early_checkpoints_selected_by_validation=True,matched_split_for_fixed_and_early=True),indent=2))
    print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'change_recovery')
    p.add_argument('--out',type=Path,default=ROOT/'optimization_validation_stopping')
    p.add_argument('--starts',type=int,default=3); p.add_argument('--max-adam',type=int,default=3000)
    p.add_argument('--truth',choices=['linear','quadratic'],default='linear')
    p.add_argument('--measurement',choices=['affine','exponential_additive'],default='affine')
    p.add_argument('--baseline-scale',type=float,default=10.)
    args=p.parse_args()
    if min(args.starts,args.max_adam)<1: p.error('Training budgets and restart counts must be positive')
    if args.baseline_scale<0 or not np.isfinite(args.baseline_scale): p.error('Baseline scale must be finite and nonnegative')
    run(args)
