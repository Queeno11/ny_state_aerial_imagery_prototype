"""Paper-aligned configurations; explicit optional computational departures."""
import argparse, copy, json, math
from pathlib import Path
from types import SimpleNamespace
import torch
LOSSES=['common_logit','heterogeneous_logit','ordinal_smooth']
DEFAULT={
 'seed':72101,'function_seed':983,'models':['linear','nn'],'truths':['linear','quadratic'],
 'n':[32],'city_grid':[4,8],'reps':1,'cities':4,'views':2,'noise_sd':.7,'bayes_target':.2,
 'acquisition_strength':1.,'group_strength':1.,'validation_fraction':.25,
 'pair_mode':'all','pairs_per_label':8,'test_per_city':128,'test_pairs':3000,'hidden':[32,16],'weight_bound':10.,'rff':16,
 'epochs':50,'restarts':1,'lbfgs_steps':30,'audit_steps':15,'lr':.01,'tau':.1,
 'ridge_grid':[0.],'lambda_grid':[0.,.3],'lambda_t':.3,'lambda_c':.3,
 'dtype':'float64','device':'cpu','threads':1,'workers':1,'grad_tol':1e-5,'audit_tol':1e-5,'false_jump':1.,
 'experiment1':{'psi_shapes':['tanh'],'cases':[['common',0.],['multiplier',1.],['location_scale',1.],['shape',1.]],
   'composition_grid':[0.,.7],'losses':LOSSES,
   'methods':['rank_common','rank_heterogeneous','rank_ordinal','raw_levels','log_cell_fe','within_zscore','within_percentile','oracle_latent','oracle_inverse']},
 'experiment2':{'losses':LOSSES,'arms':['none'],'cases':[
   ['gumbel',0.],['student',3.],['cell_scale',.7],['feature_scale',.7],['median_violation',.7]]},
 'experiment3':{'losses':LOSSES,'arms':['none','temporal','cross_city','both'],'nuisance_cases':[[1.,1.],[0.,1.],[1.,0.]]},
 'experiment4':{'losses':LOSSES,'arms':['temporal','cross_city','both'],'cases':[
   ['exact',0.],['drift',.15],['random_false',.2],['systematic_false',.2]]}
}

def preset(name):
    d=copy.deepcopy(DEFAULT)
    if name in ('pilot','main'):
        d.update(city_grid=[8,16,32],reps=5,epochs=500,lbfgs_steps=200,audit_steps=100,restarts=2,
          models=['linear','basis','nn'],truths=['linear','quadratic','smooth','regime'],lambda_grid=[0.,.03,.3,1.])
        d['experiment1']['cases']=[['common',0.]]+[[stage,v] for stage in ('multiplier','location_scale','shape') for v in (.25,.5,1.,2.)]
        d['experiment2']['cases']=[['gumbel',0.],['normal',0.]]+[['student',v] for v in (3.,5.,10.,30.)]+[
          [family,v] for family in ('cell_scale','feature_scale','median_violation') for v in (0.,.3,.7,1.)]
        d['experiment3']['nuisance_cases']=[[1.,1.],[0.,1.],[.3,1.],[2.,1.],[1.,0.],[1.,.3],[1.,2.]]
        d['experiment4']['cases']=[['exact',0.]]+[['drift',v] for v in (.05,.15,.3)]+[
          [family,v] for family in ('random_false','systematic_false') for v in (.05,.1,.2,.4)]
    if name=='main': d.update(city_grid=[8,16,32,64],reps=30,epochs=1000,lbfgs_steps=300,audit_steps=300,restarts=3)
    return d

def merge(base,extra,path=''):
    for k,v in extra.items():
        if k not in base: raise ValueError(f'Unknown configuration key: {path}{k}')
        if isinstance(base[k],dict):
            if not isinstance(v,dict): raise ValueError(f'{path}{k} must be an object')
            merge(base[k],v,path+k+'.')
        else: base[k]=v
    return base

def validate(d):
    def finite(v):
        if isinstance(v,dict):
            for x in v.values(): finite(x)
        elif isinstance(v,list):
            for x in v: finite(x)
        elif isinstance(v,float) and not math.isfinite(v): raise ValueError('Values must be finite')
    finite(d)
    if d['pair_mode'] not in ('all','sampled'): raise ValueError('pair_mode must be all or sampled')
    for key in ('models','truths','n','city_grid'):
        if not d[key] or len(set(d[key]))!=len(d[key]): raise ValueError(f'Empty/duplicate {key}')
    for key in ('reps','cities','views','pairs_per_label','test_per_city','test_pairs','rff','restarts','threads','workers'):
        if not isinstance(d[key],int) or d[key]<1: raise ValueError(f'{key} must be a positive integer')
    if any(not isinstance(x,int) or x<2 for x in d['city_grid']): raise ValueError('city_grid requires integers >=2')
    if any(not isinstance(x,int) or x<8 for x in d['n']): raise ValueError('n requires integers >=8')
    if d['test_per_city']<4: raise ValueError('test_per_city>=4 required')
    for key in ('epochs','lbfgs_steps','audit_steps'):
        if not isinstance(d[key],int) or d[key]<0: raise ValueError(f'{key} must be nonnegative integer')
    for key in ('noise_sd','acquisition_strength','group_strength','lambda_t','lambda_c','false_jump'):
        if d[key]<0: raise ValueError(f'{key} must be nonnegative')
    for key in ('lr','tau','grad_tol','audit_tol','weight_bound'):
        if d[key]<=0: raise ValueError(f'{key} must be positive')
    if d['bayes_target'] is not None and not 0<d['bayes_target']<.5: raise ValueError('bayes_target must be null or between 0 and .5')
    if not 0<d['validation_fraction']<1: raise ValueError('Invalid validation_fraction')
    if not set(d['models'])<={'linear','basis','nn'} or not set(d['truths'])<={'linear','quadratic','smooth','regime'}: raise ValueError('Invalid model/truth')
    if not d['hidden'] or any(not isinstance(x,int) or x<1 for x in d['hidden']): raise ValueError('Invalid hidden widths')
    for key in ('ridge_grid','lambda_grid'):
        if not d[key] or min(d[key])<0: raise ValueError(f'Invalid {key}')
    if 0 not in d['lambda_grid']: raise ValueError('Tuned lambda_grid must include zero')
    if d['dtype'] not in ('float32','float64') or d['device'] not in ('cpu','cuda'): raise ValueError('Invalid dtype/device')
    if d['device']=='cuda' and d['workers']!=1: raise ValueError('Multiple workers require --device cpu; use --workers 1 for CUDA')
    for exp in (1,2,3,4):
        e=d[f'experiment{exp}']
        if not e['losses'] or not set(e['losses'])<=set(LOSSES): raise ValueError('Invalid losses')
        for key,value in e.items():
            if not value or len({json.dumps(x,sort_keys=True) for x in value})!=len(value): raise ValueError(f'Empty or duplicate experiment{exp}.{key}')
    e=d['experiment1']
    if not set(e['methods'])<=set(DEFAULT['experiment1']['methods']) or 'rank_common' not in e['methods']: raise ValueError('Retain rank_common reference')
    if 'common_logit' not in e['losses']: raise ValueError('Exp1 requires common_logit reference')
    if not set(e['psi_shapes'])<={'tanh','asinh'} or min(e['composition_grid'])<0: raise ValueError('Invalid exp1 transform/composition')
    for family,v in e['cases']:
        if family not in ('common','multiplier','location_scale','shape','additive') or v<0 or (family=='common' and v!=0): raise ValueError('Invalid transform case')
    for exp in (2,3,4):
        if not set(d[f'experiment{exp}']['arms'])<={'none','temporal','cross_city','both'}: raise ValueError('Invalid arm')
    if 'none' not in d['experiment3']['arms'] or 'none' in d['experiment4']['arms']: raise ValueError('Invalid baseline arms')
    for a,c in d['experiment3']['nuisance_cases']:
        if min(a,c)<0: raise ValueError('Nuisance strengths must be nonnegative')
    for family,v in d['experiment2']['cases']:
        if family not in ('gumbel','normal','student','chi2','mixture','cell_scale','feature_scale','median_violation') or v<0 or (family=='student' and v<=2) or (family=='chi2' and v<=0): raise ValueError('Invalid error case')
    for family,v in d['experiment4']['cases']:
        if family not in ('exact','drift','random_false','systematic_false') or v<0 or ('false' in family and v>1) or (family=='exact' and v!=0): raise ValueError('Invalid link case')

def parse_args():
    p=argparse.ArgumentParser(description='Manuscript-aligned Monte Carlo suite. --plan reports cost; --check audits formulas.')
    p.add_argument('--preset',choices=['smoke','pilot','main'],default='smoke'); p.add_argument('--config',type=Path)
    p.add_argument('--experiments',nargs='+',default=['all'],choices=['all','1','2','3','4']); p.add_argument('--out',type=Path,default=Path('results'))
    for k in ('resume','plan','check'): p.add_argument('--'+k,action='store_true')
    for k in ('models','truths','n','city_grid','ridge_grid','lambda_grid','hidden'):
        p.add_argument('--'+k.replace('_','-'),nargs='+',type=int if k in ('n','city_grid','hidden') else float if 'grid' in k else str)
    p.add_argument('--losses',nargs='+',choices=LOSSES)
    for k in ('reps','seed','function_seed','epochs','restarts','lbfgs_steps','audit_steps','threads','workers','cities'):
        p.add_argument('--'+k.replace('_','-'),type=int)
    p.add_argument('--device',choices=['cpu','cuda']); p.add_argument('--dtype',choices=['float64','float32'])
    args=p.parse_args(); d=preset(args.preset)
    if args.config: merge(d,json.loads(args.config.read_text()))
    for k in d:
        if hasattr(args,k) and getattr(args,k) is not None: d[k]=getattr(args,k)
    if args.cities is not None: d['city_grid']=[args.cities]
    if args.losses:
        for e in (1,2,3,4): d[f'experiment{e}']['losses']=args.losses
    validate(d); experiments=[1,2,3,4] if 'all' in args.experiments else sorted(set(map(int,args.experiments)))
    return args,d,experiments

def runtime(d):
    c=SimpleNamespace(**copy.deepcopy(d)); c.torch_dtype=getattr(torch,c.dtype)
    if c.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(c.threads); torch.use_deterministic_algorithms(True)
    return c
