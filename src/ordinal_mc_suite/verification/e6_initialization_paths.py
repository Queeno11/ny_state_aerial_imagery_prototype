"""One fixed setting: ordinal scratch, joint scratch, extended joint warm start."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import e6_change_followups as follow
import e6_observed_changes as previous
import diagnose_e6_gradients as diag


def state_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load(source):
    saved=torch.load(source/'checkpoints.pt',map_location='cpu',weights_only=False)
    return follow.reconstruct(source,saved)


def prepare(source,out,settings):
    z=load(source);cfg=z['cfg'];n=len(z['design']['train']['w'])
    seed=diag.core.seed_for(cfg.seed,'fit',1,512,32,'quadratic','nn',0)
    torch.manual_seed(seed)
    model=diag.core.Estimator(7,'nn','common_logit',cfg).double()
    # Fitting scores/outcomes only: no frozen training or labels from oracle quantities.
    head=diag.CalibrationHead(diag.core.predictions(model,z['features'],z['design']['train']['x'],cfg),
        z['design']['train']['cell'],z['design']['train']['w'],'spline',z['d']['head_training'])
    scratch=dict(model=diag.core.cpu_state(model),head=diag.core.cpu_state(head))
    warm=dict(model=diag.core.cpu_state(z['A']),head=diag.core.cpu_state(z['head']))
    torch.save(dict(scratch=scratch,warm=warm),out/'initializations.pt')
    info=json.loads((source/'metadata.json').read_text())
    metadata=dict(scorer_initialization_seed=seed,fit_seed=info['fit_seed'],validation_seed=info['validation_seed'],
        fitting_endpoints=n,validation_endpoints=len(z['design']['val']['w']),denominators=z['den'],
        scratch_scorer_hash=state_hash(scratch['model']),scratch_head_hash=state_hash(scratch['head']),
        warm_scorer_hash=state_hash(warm['model']),warm_head_hash=state_hash(warm['head']),
        inherited_baseline_updates_executed=info['baseline']['updates'],
        inherited_baseline_selected_update=info['baseline']['best_step'],
        inherited_head_updates_executed=info['head_initialization']['updates'],
        source_config=json.loads((source/'config.json').read_text()))
    (out/'preparation.json').write_text(json.dumps(metadata,indent=2))
    print('Exact paired data reconstructed; identical scratch initialization saved; seed',seed,flush=True)


def train_one(path,source,out,settings):
    torch.set_num_threads(settings['threads_per_worker'])
    with threadpool_limits(limits=settings['threads_per_worker']):
        return _train_one(path,source,out,settings)


def _train_one(path,source,out,settings):
    started=time.monotonic();folder=out/path;folder.mkdir();z=load(source);cfg=z['cfg'];n=len(z['design']['train']['w'])
    initial=torch.load(out/'initializations.pt',map_location='cpu',weights_only=False)
    state=initial['warm' if path=='extended_warm' else 'scratch']
    model=copy.deepcopy(z['A']);model.load_state_dict(state['model'])
    # Scratch has fitting-only knots and normalization from its own initial scores.
    head=diag.CalibrationHead(diag.core.predictions(model,z['features'],z['design']['train']['x'],cfg),
        z['design']['train']['cell'],z['design']['train']['w'],'spline',z['d']['head_training'])
    head.load_state_dict(state['head']);head.audit_monotonicity()
    sp=list(model.scorer.parameters());hp=list(head.parameters())
    so=torch.optim.Adam(sp,lr=settings['scorer_lr']);ho=torch.optim.Adam(hp,lr=settings['head_lr'])
    tracker=diag.tr.Tracker(settings['validation_patience_checks'],settings['validation_min_delta'])
    stop_step=None;selected=None;rows=[];updates=[];states={}
    schedule=set(range(0,settings['updates_per_path']+1,settings['trajectory_every']))|set(settings['extra_checkpoints'])
    def snapshot(): return dict(model=diag.core.cpu_state(model),head=diag.core.cpu_state(head))
    for step in range(settings['updates_per_path']+1):
        if step%settings['validation_every']==0 or step in schedule:
            with torch.no_grad():
                r,p,fw,fd=follow.quantities(model,head,z)
                vw,vd=previous.head_losses(head,model.scorer(z['valid']['x']),z['design']['val'],z['den'],cfg)
                vr=diag.tr.rank_loss(model,z['valid'])
                value=float(vr if path=='ordinal_scratch' else vr+vw+vd)
                rows.append(dict(path=path,step=step,fit_rank=float(r),fit_stability_ratio=float(p),fit_level=float(fw),fit_change=float(fd),
                    val_rank=float(vr),val_level=float(vw),val_change=float(vd),validation_rule=value,
                    selection_still_active=stop_step is None))
            if step%settings['validation_every']==0 and stop_step is None:
                exhausted=tracker.observe([model,head],value,step)
                if tracker.step==step: selected=snapshot()
                if exhausted:
                    stop_step=step
                    print(path,'validation patience reached at',step,'selected',tracker.step,flush=True)
            if step in schedule: states[f'{path}_{step}']=snapshot()
        if step==settings['updates_per_path']: break
        so.zero_grad(set_to_none=True);ho.zero_grad(set_to_none=True)
        score_loss,parts=diag.core.score_objective(model,z['data'],z['edges'],(.3,0.),0.,cfg.tau)
        scores=model.scorer(z['data']['x'][:n])
        if path=='ordinal_scratch': scores=scores.detach()
        fw,fd=previous.head_losses(head,scores,z['design']['train'],z['den'],cfg)
        objective=score_loss+settings['lambda_W']*fw+settings['lambda_delta']*fd
        if not bool(torch.isfinite(objective)): raise FloatingPointError('Nonfinite objective')
        objective.backward()
        sg=float(torch.nn.utils.clip_grad_norm_(sp,20.));hg=float(torch.nn.utils.clip_grad_norm_(hp,20.))
        so.step();ho.step()
        updates.append(dict(step=step+1,scorer_gradient_norm=sg,head_gradient_norm=hg,
            scorer_clipped=sg>20.,head_clipped=hg>20.))
        if (step+1)%100==0:
            pd.DataFrame(rows).to_csv(folder/'observable_paths.csv',index=False)
            print(path,'completed',step+1,'/',settings['updates_per_path'],flush=True)
    head.audit_monotonicity()
    states[f'{path}_selected']=selected
    torch.save(dict(checkpoints=states,optimizer_final=dict(scorer=so.state_dict(),head=ho.state_dict())),folder/'checkpoints.pt')
    pd.DataFrame(rows).to_csv(folder/'observable_paths.csv',index=False)
    pd.DataFrame(updates).to_csv(folder/'optimizer_trace.csv',index=False)
    prep=json.loads((out/'preparation.json').read_text());warm=path=='extended_warm'
    summary=dict(path=path,selected_step=tracker.step,selected_validation=tracker.best,patience_stop_step=stop_step,
        validation_selection='rank logistic' if path=='ordinal_scratch' else 'rank logistic + standardized level MSE + standardized change MSE',
        executed_scorer_updates=settings['updates_per_path'],executed_head_updates=settings['updates_per_path'],
        inherited_scorer_updates_executed=prep['inherited_baseline_updates_executed'] if warm else 0,
        inherited_selected_scorer_ancestry=prep['inherited_baseline_selected_update'] if warm else 0,
        total_scorer_updates_executed=settings['updates_per_path']+(prep['inherited_baseline_updates_executed'] if warm else 0),
        selected_scorer_ancestry=tracker.step+(prep['inherited_baseline_selected_update'] if warm else 0),
        fixed_endpoint_scorer_ancestry=settings['updates_per_path']+(prep['inherited_baseline_selected_update'] if warm else 0),
        scorer_initial_hash=state_hash(state['model']),head_initial_hash=state_hash(state['head']),
        wall_seconds=time.monotonic()-started,scorer_clipped_updates=sum(r['scorer_clipped'] for r in updates),
        head_clipped_updates=sum(r['head_clipped'] for r in updates),selected_at_budget_boundary=tracker.step==settings['updates_per_path'])
    (folder/'summary.json').write_text(json.dumps(summary,indent=2));return summary


@torch.no_grad()
def evaluate_all(source,out,settings):
    z=load(source);cfg=z['cfg'];model=copy.deepcopy(z['A']);head=copy.deepcopy(z['head'])
    integration=json.loads((Path(__file__).resolve().parents[1]/'configs/e6_change_followups.json').read_text())
    oracle=follow.oracle_panels(z,integration,out)
    metrics=[];outcomes=[];selectedrows=[]
    for path in settings['paths']:
        summary=json.loads((out/path/'summary.json').read_text())
        saved=torch.load(out/path/'checkpoints.pt',map_location='cpu',weights_only=False)['checkpoints']
        for key,state in saved.items():
            suffix=key[len(path)+1:];kind='selected' if suffix=='selected' else 'trajectory'
            step=summary['selected_step'] if kind=='selected' else int(suffix)
            model.load_state_dict(state['model'])
            if state['head']['knots'].shape!=head.knots.shape: raise AssertionError('Head architecture mismatch')
            head.load_state_dict(state['head'])
            rows=previous.evaluate(model,head,z['features'],z['design'],z['test'],cfg,z['d'])
            tag=dict(path=path,kind=kind,step=step)
            metrics.extend(dict(**tag,**row) for row in rows)
            for strength in [.15,.4]:
                p=z['test']['e6_panels'][strength]['seen'];pred=[]
                for endpoint in ['before','after']:
                    s=diag.core.predictions(model,z['features'],p['x_'+endpoint],cfg)
                    pred.append(head(diag.core.tensor(s,cfg),diag.core.tensor(p['cell_'+endpoint],cfg,True)).numpy())
                m0,v0=oracle[strength]['before'];m1,v1=oracle[strength]['after'];p0,p1=pred
                for group,mask in [('all',np.ones(len(p0),bool)),('stable',~p['assigned_changed']),('changing',abs(p['delta_f'])>z['d']['truth_threshold'])]:
                    outcomes.append(dict(**tag,strength=strength,group=group,
                        mean_level_MSE=float(np.mean(np.r_[(p0-m0)[mask],(p1-m1)[mask]]**2)),
                        mean_change_MSE=float(np.mean(((p1-p0)-(m1-m0))[mask]**2)),
                        realized_level_MSE=float(np.mean(np.r_[(p0-p['w_before'])[mask],(p1-p['w_after'])[mask]]**2)),
                        realized_change_MSE=float(np.mean(((p1-p0)-(p['w_after']-p['w_before']))[mask]**2)),
                        noise_floor_level=float(np.mean((v0[mask]+v1[mask])/2)),noise_floor_change=float(np.mean(v0[mask]+v1[mask]))))
    pd.DataFrame(metrics).to_csv(out/'trajectory_metrics.csv',index=False)
    pd.DataFrame(outcomes).to_csv(out/'outcome_trajectories.csv',index=False)
    m=pd.DataFrame(metrics);o=pd.DataFrame(outcomes)
    selected=m[m.kind=='selected'];fixed=m[(m.kind=='trajectory')&(m.step==settings['updates_per_path'])]
    pd.concat([selected,fixed]).to_csv(out/'selected_and_fixed_metrics.csv',index=False)
    pd.concat([o[o.kind=='selected'],o[(o.kind=='trajectory')&(o.step==settings['updates_per_path'])]]).to_csv(out/'selected_and_fixed_outcomes.csv',index=False)
    return z


def run(args):
    source=args.source.resolve();out=args.out.resolve();settings=json.loads(args.config.read_text())
    if out.exists() and any(out.iterdir()): raise ValueError('Choose a new output directory')
    out.mkdir(parents=True);(out/'config.json').write_text(json.dumps(settings,indent=2))
    hashes={str(source/f):follow.digest(source/f) for f in ['checkpoints.pt','config.json','metadata.json']}
    with threadpool_limits(limits=1):
        torch.set_num_threads(1);prepare(source,out,settings)
    summaries=[]
    with ProcessPoolExecutor(max_workers=settings['workers'],mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(train_one,path,source,out,settings):path for path in settings['paths']}
        for future in as_completed(futures): summaries.append(future.result())
    bypath={r['path']:r for r in summaries}
    assert bypath['ordinal_scratch']['scorer_initial_hash']==bypath['joint_scratch']['scorer_initial_hash']
    assert bypath['ordinal_scratch']['head_initial_hash']==bypath['joint_scratch']['head_initial_hash']
    with threadpool_limits(limits=1):
        torch.set_num_threads(1);evaluate_all(source,out,settings)
    for path,digest in hashes.items(): assert follow.digest(Path(path))==digest
    metadata=dict(source=str(source),settings=settings,summaries=[bypath[p] for p in settings['paths']],source_hashes_unchanged=hashes,
        identical_scratch_scorer_and_heads_verified=True,all_training_before_test_evaluation=True,
        test_or_latent_targets_used_for_selection=False,total_new_scorer_updates=sum(r['executed_scorer_updates'] for r in summaries),
        selection_policy=settings['fixed_budget_policy'],runtime=dict(torch=torch.__version__,numpy=np.__version__),
        notes='One replication, one LR pair and one weight setting. Initial head knots/normalization use fitting scores only and remain fixed per path. Warm start inherits calibrated head and scorer; this does not isolate scorer initialization from head preparation.')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
