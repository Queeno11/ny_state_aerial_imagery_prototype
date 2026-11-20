"""One paired replication: train all arms first, then evaluate independent panels."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .. import core
from ..design import PopulationDGP, inverse_measurement
from ..likelihood import fit_ml
from .config import e5_runtime
from .data import build
from .heads import CalibrationHead
from . import training as train
from .evaluation import evaluate


def job_key(job):
    return f"r{job['rep']:03d}_c{job['cities']}_{job['truth']}_{job['model']}"


def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def compute_job(task):
    # Dynamic limits cover serial runs too, after numerical modules are imported.
    with threadpool_limits(limits=task[0]['threads'],user_api='blas'):
        return _compute_job(task)


def _compute_job(task):
    d,job,output=task;cfg=e5_runtime(d,job['cities']);dgp=PopulationDGP(cfg)
    if output: print('Starting '+job_key(job),flush=True)
    base,variants=build(cfg,dgp,job['rep'],job['truth'],d)
    features=core.Features(base['train']['x'],job['model'],cfg)
    data=core.prepare(base,features,cfg);val=train.validation_data(base['val'],features,cfg)
    edges={name:core.tensor(base['links'][name],cfg,True) for name in ('temporal','cross_city')}
    histories=[];selection=[];checkpoints={};candidates=[]
    def record(label,info,**extra):
        clean={k:v for k,v in info.items() if k!='trace'}
        selection.append(dict(label=label,**extra,**clean))
        histories.extend(dict(label=label,**extra,**row) for row in info.get('trace',[]))
    for restart in range(d['restarts']):
        seed=core.seed_for(cfg.seed,'fit',job['rep'],d['labels_per_city'],cfg.cities,job['truth'],job['model'],restart)
        torch.manual_seed(seed)
        model=core.Estimator(data['x'].shape[1],job['model'],'common_logit',cfg).double()
        info=train.baseline(model,data,edges,val,cfg,d)
        record('A_restart',info,restart=restart);candidates.append((info['validation'],restart,model,info))
    _,restart,A,ainfo=min(candidates,key=lambda x:(x[0],x[1]))
    if output: print(job_key(job)+': A selected by validation',flush=True)
    checkpoints['A']={'scorer':core.cpu_state(A),'info':{k:v for k,v in ainfo.items() if k!='trace'},'restart':restart}
    fitted=[];checks=dict(frozen_score_equivalence=True,matched_CD_updates=True,level_gradient=True,
        tract_split_disjoint=True,exact_links=True,head_monotone=True,test_used_for_selection=False)
    # The optional oracle sees noisy inverse labels and is kept outside ordinary selection.
    oracle=None
    if d['oracle_R']['enabled']:
        source=next(iter(variants.values()));tr=source['train'];va=source['val'];sc=source['measurement_scenario']
        r=inverse_measurement(tr['w'],tr['a'],tr['b'],tr['kappa'],sc['psi_delta'],'tanh')
        vr=inverse_measurement(va['w'],va['a'],va['b'],va['kappa'],sc['psi_delta'],'tanh')
        np.testing.assert_allclose(r,tr['r'],atol=1e-12)
        oracle=copy.deepcopy(A);info=train.oracle_noisy_R(oracle,data,edges,r,val['x'],vr,cfg,d)
        record('oracle_noisy_R',info);checkpoints['oracle_noisy_R']={'scorer':core.cpu_state(oracle)}
    for kind,design in variants.items():
        level=train.level_data(design['train'],cfg);v=train.validation_data(design['val'],features,cfg)
        for head_kind in d['heads']:
            fit_candidates=[];joint_candidates=[]
            scores=core.predictions(A,features,design['train']['x'],cfg)
            for weight in d['level_weights']:
                head=CalibrationHead(scores,design['train']['cell'],design['train']['w'],head_kind,d['head_training'])
                binfo=train.fit_head(A,head,data,level,v,cfg,d,weight)
                record('B_candidate',binfo,map=kind,head=head_kind,weight=weight)
                fit_candidates.append((binfo['validation'],weight,copy.deepcopy(head),binfo))
                joint=copy.deepcopy(A);jh=copy.deepcopy(head)
                # Audit actual level gradient into the scorer, without updating parameters.
                joint.zero_grad(set_to_none=True)
                train.level_loss(joint,jh,data['x'],level).backward()
                norm=sum(float(p.grad.square().sum()) for p in joint.scorer.parameters() if p.grad is not None)
                if not np.isfinite(norm) or norm<=0: raise AssertionError('Level loss has no scorer gradient')
                joint.zero_grad(set_to_none=True);jh.zero_grad(set_to_none=True)
                cinfo=train.fit_joint(joint,jh,data,edges,level,v,cfg,d,weight)
                cinfo['initial_level_gradient_norm']=float(np.sqrt(norm))
                record('C_candidate',cinfo,map=kind,head=head_kind,weight=weight)
                joint_candidates.append((cinfo['validation'],weight,joint,jh,cinfo,copy.deepcopy(head)))
                checkpoints[f'{kind}_{head_kind}_lambda{weight:g}']={
                    'B_head':core.cpu_state(head),'C_scorer':core.cpu_state(joint),'C_head':core.cpu_state(jh),
                    'B_info':{k:v for k,v in binfo.items() if k!='trace'},'C_info':{k:v for k,v in cinfo.items() if k!='trace'}}
            _,bw,B,binfo=min(fit_candidates,key=lambda x:(x[0],x[1]))
            _,cw,C,CH,cinfo,paired_B=min(joint_candidates,key=lambda x:(x[0],x[1]))
            D=copy.deepcopy(A)
            dinfo,Dcommon=train.continue_ordinal(D,paired_B,data,edges,level,v,cfg,d,cinfo['updates'])
            record('D',dinfo,map=kind,head=head_kind,matched_C_weight=cw)
            # Post-hoc calibration gives D observed-scale predictions, without scorer updates.
            dheads=[]
            for weight in d['level_weights']:
                ds=core.predictions(D,features,design['train']['x'],cfg)
                dh=CalibrationHead(ds,design['train']['cell'],design['train']['w'],head_kind,d['head_training'])
                info=train.fit_head(D,dh,data,level,v,cfg,d,weight)
                record('D_posthoc_head',info,map=kind,head=head_kind,weight=weight)
                dheads.append((info['validation'],weight,dh))
            _,dw,DH=min(dheads,key=lambda x:(x[0],x[1]))
            if output: print(f'{job_key(job)}: {kind}/{head_kind} finalized; C lambda={cw:g}, C/D updates={cinfo["updates"]}',flush=True)
            common=dict(map=kind,head=head_kind)
            fitted.extend([(dict(common,arm='A',weight=0.),A,None,{}),
                (dict(common,arm='B',weight=bw),A,B,{'frozen_same_scorer':True}),
                (dict(common,arm='C',weight=cw),C,CH,{'scorer_updates':cinfo['updates'],'best_step':cinfo['best_step']}),
                (dict(common,arm='D',weight=dw),D,DH,{'scorer_updates':dinfo['updates'],'best_step':dinfo['best_step']}),
                (dict(common,arm='D_common',weight=cw),Dcommon,paired_B,{'scorer_updates':dinfo['updates'],'best_step':dinfo['common_rule_best_step']})])
            for arm,model,head in [('B',A,B),('C',C,CH),('D',D,DH),('D_common',Dcommon,paired_B)]:
                checkpoints[f'{kind}_{head_kind}_{arm}_selected']={'scorer':core.cpu_state(model),'head':core.cpu_state(head)}
            selection.extend([dict(label='selected_B',map=kind,head=head_kind,weight=bw,validation=binfo['validation']),
                dict(label='selected_C',map=kind,head=head_kind,weight=cw,validation=cinfo['validation'],updates=cinfo['updates']),
                dict(label='B_at_C_weight_common_rule',map=kind,head=head_kind,weight=cw,
                     validation=train.observable_rule(A,paired_B,v,level['variance'],d))])
    # No candidate scorer/head can change after this boundary.
    rows=[];strata=[];arrays={}
    def collect(meta,predict,head,info,oracle_head=False):
        r,s,a=evaluate(predict,head,variants[meta['map']],d,cfg,meta['map'],oracle_head=oracle_head)
        rows.extend(dict(**job,**meta,**info,**x) for x in r)
        strata.extend(dict(**job,**meta,**x) for x in s)
        prefix=meta['map']+'_'+meta['head']+'_'+meta['arm']+'_'
        arrays.update({prefix+k:v for k,v in a.items()})
    for meta,model,head,info in fitted:
        collect(meta,lambda x,m=model:core.predictions(m,features,x,cfg),head,info)
    # A and B latent metrics must be bitwise equal, including calibrated cutoffs.
    latent=('change_rmse','sign_accuracy','stable_rms','cutoff','false_positive_rate','sensitivity','score_sd_fit',
            'predicted_change_mean','true_change_mean','within_latent_accuracy','normalization_only_rmse')
    for kind in d['maps']:
        for head in d['heads']:
            aa=[r for r in rows if r['map']==kind and r['head']==head and r['arm']=='A']
            bb=[r for r in rows if r['map']==kind and r['head']==head and r['arm']=='B']
            for a,b in zip(aa,bb):
                for key in latent:
                    if a[key]!=b[key]: raise AssertionError('Frozen calibration altered '+key)
        ml,info=fit_ml(variants[kind],job['truth'],variants[kind]['measurement_scenario'],dgp)
        collect(dict(map=kind,head='oracle',arm='oracle_ML',weight=0.),ml,None,info,oracle_head=True)
        if oracle is not None:
            collect(dict(map=kind,head='oracle',arm='oracle_noisy_R',weight=0.),
                lambda x:core.predictions(oracle,features,x,cfg),None,{'oracle_labels':'noisy inverse R, not f(H)'})
    result=dict(job=job,metrics=rows,strata=strata,selection=selection,history=histories,checks=checks,
        design=dict(n_fit=base['n_fit'],n_val=base['n_val'],realized_validation_fraction=base['n_val']/(base['n_val']+base['n_fit']),
            links=base['link_stats']['temporal'],lambda_t=d['lambda_t'],alpha=d['alpha'],noise_sd=base['noise_sd_effective']))
    if output:
        folder=Path(output)/'replications'/job_key(job);folder.mkdir(parents=True,exist_ok=True)
        temp=folder/'checkpoints.pt.tmp'
        torch.save(dict(models=checkpoints,feature_mu=features.mu,feature_sd=features.sd,
                        feature_bmu=features.bmu,feature_bsd=features.bsd,config=d,job=job),temp)
        temp.replace(folder/'checkpoints.pt')
        temp=folder/'changes.tmp.npz';np.savez_compressed(temp,**arrays);temp.replace(folder/'changes.npz')
        atomic_json(folder/'result.json',result)
    return result
