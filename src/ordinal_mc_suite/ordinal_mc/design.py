"""Coherent unit-level DGPs; scenarios alter one restriction at a time."""
from __future__ import annotations
import numpy as np
from .core import seed_for, split_labels, pair_indices, StructuralDGP

BOUND = np.sqrt(3.)

def measurement(r, a, b, kappa, delta, shape):
    h = {'tanh':np.tanh, 'asinh':np.arcsinh, 'cubic':lambda z:z**3}[shape]
    return a+b*(r+delta*kappa*h(r))

def inverse_measurement(w, a, b, kappa, delta, shape):
    z=(w-a)/b
    if delta == 0:
        return z.copy()
    lo,hi=np.minimum(z,0.),np.maximum(z,0.)
    for _ in range(64):
        mid=(lo+hi)/2
        lower=measurement(mid,0.,1.,kappa,delta,shape)<z
        lo=np.where(lower,mid,lo); hi=np.where(lower,hi,mid)
    return (lo+hi)/2

def coefficients(contexts,c,t):
    return (3*contexts[c,0]+t, 1.5+np.exp(.2*contexts[c,1]),
            np.exp(.4*contexts[c,0]+.15*t))

def unit_errors(rng,h,c,t,cell_z,family,severity):
    """All marginal base errors have mean zero and variance one."""
    if family in ('gumbel','cell_scale','median_violation'):
        e=(rng.gumbel(size=len(h))-np.euler_gamma)*np.sqrt(6)/np.pi
    elif family in ('normal','feature_scale'):
        e=rng.normal(size=len(h))
    elif family=='mixture':
        e=(rng.normal(size=len(h))+severity*rng.choice([-1.,1.],len(h)))/np.sqrt(1+severity**2)
    elif family=='student':
        e=rng.standard_t(severity,len(h))*np.sqrt((severity-2)/severity)
    elif family=='chi2':
        e=(rng.chisquare(severity,len(h))-severity)/np.sqrt(2*severity)
    else:
        raise ValueError(family)
    if family=='cell_scale':
        e*=np.exp(severity*cell_z[c,t]-severity**2)
    if family in ('feature_scale','median_violation'):
        a=2*severity*BOUND
        rms=np.sqrt(np.sinh(a)/a) if a else 1.
        e*=np.exp(severity*h[:,0])/rms
    return e

def make_inputs(h,c,t,contexts,rng,cfg,dgp):
    temporal=cfg.acquisition_strength*(rng.uniform(-BOUND,BOUND,size=(len(h),2))+.6*t[:,None])
    local=cfg.group_strength*(BOUND*np.tanh(contexts[c]))
    return np.column_stack((h,temporal,local))@dgp.mix

def panel(seed,contexts,cell_z,n,cfg,dgp,truth,scenario):
    rng=np.random.default_rng(seed)
    c=np.repeat(np.arange(len(contexts)),n)
    t=np.tile(np.arange(n)%2,len(contexts))
    h=dgp.draw_h(rng,contexts[c,0])
    x=make_inputs(h,c,t,contexts,rng,cfg,dgp)
    f=dgp.score(h,truth)
    erng=np.random.default_rng(seed_for(seed,'errors'))
    r=f+dgp.noise_sd*unit_errors(erng,h,c,t,cell_z,scenario['error'],scenario['error_severity'])
    a,b,k=cell_coefficients(contexts,c,t,scenario)
    w=measurement(r,a,b,k,scenario['psi_delta'],scenario['psi_shape'])
    if scenario['positive_labels']:
        transformed=measurement(r,0.,1.,k,scenario['psi_delta'],scenario['psi_shape'])
        w=a+b*np.exp(.6*transformed)
    return dict(x=x,h=h,c=c,t=t,cell=2*c+t,f=f,w=w,r=r,a=a,b=b,kappa=k)

def structural_gradient(h,truth,dgp):
    u=h@dgp.rotation
    if truth=='linear': g=np.tile([1.,.6,-.4],(len(h),1))
    elif truth=='quadratic': g=np.column_stack((np.ones(len(h)),.7*u[:,2],.7*u[:,1]+.8*u[:,2]))
    elif truth=='smooth': g=np.column_stack((1.2*np.cos(1.2*u[:,0]),.5/np.cosh(u[:,1])**2+.4*u[:,2],.4*u[:,1]))
    else: g=np.column_stack((np.ones(len(h)),.8*(u[:,1]>0),.6*(u[:,2]<0)))
    return g@dgp.rotation.T/dgp.moments[truth][1]

def alter_links(h,rng,scenario,dgp,truth,cfg):
    kind,level=scenario['link_case'],scenario['link_severity']
    direction=rng.normal(size=h.shape)
    direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-12)
    selected=rng.random(len(h))<level
    replacement=dgp.draw_h(rng,rng.normal(size=len(h)))
    if kind=='exact': return h.copy(),np.zeros(len(h),bool)
    if kind=='drift': return np.clip(h+level*direction,dgp.hlow,dgp.hhigh),np.zeros(len(h),bool)
    if kind=='random_false': return np.where(selected[:,None],replacement,h),selected
    if kind=='changed':
        changed = change_h(h,rng,scenario['change_strength'],dgp,truth)
        return np.where(selected[:,None],changed,h),selected
    if kind=='systematic_false':
        g=structural_gradient(h,truth,dgp)
        g/=np.maximum(np.linalg.norm(g,axis=1,keepdims=True),1e-12)
        best=h.copy(); bestf=dgp.score(h,truth)
        for fraction in (1.,.5,.2,.05,.01,.0001):
            candidate=np.clip(h+cfg.false_jump*fraction*g,dgp.hlow,dgp.hhigh)
            score=dgp.score(candidate,truth)
            improve=selected & (score>bestf)
            best[improve]=candidate[improve]; bestf[improve]=score[improve]
        return best,selected
    raise ValueError(kind)


def change_h(h,rng,strength,dgp,truth):
    """Equal-probability improvement/decline along the true local gradient."""
    g=structural_gradient(h,truth,dgp)
    g/=np.maximum(np.linalg.norm(g,axis=1,keepdims=True),1e-12)
    direction=rng.choice([-1.,1.],len(h))
    return np.clip(h+strength*direction[:,None]*g,dgp.hlow,dgp.hhigh)


def matched_panel(seed,contexts,cfg,dgp,truth,strength,changed_share):
    rng=np.random.default_rng(seed)
    c=np.repeat(np.arange(len(contexts)),cfg.test_per_city)
    h=dgp.draw_h(rng,contexts[c,0]); after=change_h(h,rng,strength,dgp,truth)
    # Fixed counts give every city stable controls, even in small smoke runs.
    changed=np.zeros(len(h),bool)
    for city in range(len(contexts)):
        loc=np.flatnonzero(c==city)
        changed[rng.permutation(loc)[:int(round(len(loc)*changed_share))]]=True
    after=np.where(changed[:,None],after,h)
    x0=make_inputs(h,c,np.zeros(len(c),int),contexts,rng,cfg,dgp)
    x1=make_inputs(after,c,np.ones(len(c),int),contexts,rng,cfg,dgp)
    return dict(x_before=x0,x_after=x1,delta_f=dgp.score(after,truth)-dgp.score(h,truth),
                assigned_changed=changed)

def make_design(cfg,dgp,rep,n,truth,scenario):
    dgp.configure(truth,scenario['composition'])
    seed=seed_for(cfg.seed,'data',rep,n,cfg.cities)
    rng=np.random.default_rng(seed)
    contexts=rng.normal(size=(cfg.cities,2)); cell_z=rng.normal(size=(cfg.cities,2))
    labeled=panel(seed_for(seed,'labeled'),contexts,cell_z,n,cfg,dgp,truth,scenario)
    train,val=split_labels(labeled,cfg.validation_fraction,np.random.default_rng(seed_for(seed,'split')))
    count=len(train['x']); source=np.repeat(np.arange(count),cfg.views); budget=len(source)
    c=train['c'][source]; t=train['t'][source]
    crng=np.random.default_rng(seed_for(seed,'other_city'))
    other=(c+crng.integers(1,cfg.cities,len(c)))%cfg.cities
    xs=[train['x']]; allf=[train['f']]; allh=[train['h']]; flags=[np.zeros(count,bool)]
    links={'none':np.empty((0,2),np.int64)}
    for idx,(arm,cc,tt) in enumerate((('temporal',c,1-t),('cross_city',other,t))):
        h,flag=alter_links(train['h'][source],np.random.default_rng(seed_for(seed,'changes',arm)),scenario,dgp,truth,cfg)
        xs.append(make_inputs(h,cc,tt,contexts,np.random.default_rng(seed_for(seed,'view',arm)),cfg,dgp))
        allf.append(dgp.score(h,truth)); allh.append(h); flags.append(flag)
        links[arm]=np.column_stack((source,count+idx*budget+np.arange(budget)))
    order=np.random.default_rng(seed_for(seed,'mixed')).permutation(budget)
    links['both']=np.vstack((links['temporal'],links['cross_city']))
    ff=np.concatenate(allf); hh=np.vstack(allh); false=np.concatenate(flags)
    link_stats={}
    for arm,edges in links.items():
        if not len(edges):
            link_stats[arm]=dict(link_count=0,link_gap_rms=0.,link_gap_abs=0.,link_false_share=0.,link_h_max=0.,link_upward_share=0.)
        else:
            a,b=edges.T; gap=ff[b]-ff[a]
            link_stats[arm]=dict(link_count=len(edges),link_gap_rms=float(np.sqrt(np.mean(gap**2))),
                link_gap_abs=float(np.mean(abs(gap))),link_false_share=float(false[b].mean()),
                link_h_max=float(np.linalg.norm(hh[b]-hh[a],axis=1).max()),link_upward_share=float(np.mean(gap>1e-12)))
    tests={}; change_tests={}
    for name in ('seen','unseen'):
        trng=np.random.default_rng(seed_for(seed,name))
        cx=contexts if name=='seen' else trng.normal(size=(max(4,cfg.cities//2),2))
        zz=cell_z if name=='seen' else trng.normal(size=(len(cx),2))
        data=panel(seed_for(seed,'panel',name),cx,zz,cfg.test_per_city,cfg,dgp,truth,scenario)
        tests[name]=(data,{kind:pair_indices(data,trng,cfg.test_pairs,kind) for kind in ('within','cross_city','across_time')})
        change_tests[name]=matched_panel(seed_for(seed,'matched',name),cx,cfg,dgp,truth,
            scenario['change_strength'],cfg.experiment5['changed_share'])
    calibration_contexts=np.random.default_rng(seed_for(seed,'calibration_contexts')).normal(size=(max(4,cfg.cities),2))
    change_calibration=matched_panel(seed_for(seed,'stable_calibration'),calibration_contexts,cfg,dgp,truth,0.,0.)
    if cfg.pair_mode=='all':
        ii=[]; jj=[]; ww=[]
        for cell in np.unique(train['cell']):
            loc=np.flatnonzero(train['cell']==cell); a=np.repeat(loc,len(loc)); b=np.tile(loc,len(loc)); keep=a!=b
            ii.extend(a[keep]); jj.extend(b[keep]); ww.extend(np.full(keep.sum(),1/(count*(len(loc)-1))))
        pairs=(np.array(ii,dtype=np.int64),np.array(jj,dtype=np.int64)); pair_weights=np.array(ww)
    else:
        pairs=pair_indices(train,np.random.default_rng(seed_for(seed,'pairs')),cfg.pairs_per_label*count)
        pair_weights=np.full(len(pairs[0]),1/len(pairs[0]))
    return dict(train=train,val=val,x=np.vstack(xs),pairs=pairs,pair_weights=pair_weights,
        val_pairs=pair_indices(val,np.random.default_rng(seed_for(seed,'valpairs')),cfg.test_pairs),links=links,link_stats=link_stats,
        tests=tests,change_tests=change_tests,change_calibration=change_calibration,
        n_fit=count,n_val=len(val['x']),budget=budget,contexts=contexts,cell_z=cell_z,seed=seed,noise_sd_effective=dgp.noise_sd,baseline_oracle_risk=dgp.baseline_oracle_risk)

def level_target(method,design,cfg,scenario):
    tr=design['train']; w=tr['w']; diagnostics={}
    if method=='raw_levels': target=w.copy()
    elif method=='oracle_latent': target=tr['f'].copy()
    elif method=='oracle_inverse':
        if scenario['positive_labels']:
            observed=np.log((w-tr['a'])/tr['b'])/.6
            target=inverse_measurement(observed,0.,1.,tr['kappa'],scenario['psi_delta'],scenario['psi_shape'])
        else:
            target=inverse_measurement(w,tr['a'],tr['b'],tr['kappa'],scenario['psi_delta'],scenario['psi_shape'])
    elif method=='log_cell_fe':
        if np.any(w<=0): raise ValueError('Log benchmark requires strictly positive W; use the positive-label experiment-1 DGP.')
        target=np.log(w)
    elif method in ('within_zscore','within_percentile'):
        target=np.empty(len(w))
        for cell in np.unique(tr['cell']):
            keep=tr['cell']==cell; v=w[keep]
            if method=='within_zscore': target[keep]=(v-v.mean())/max(v.std(),1e-12)
            else:
                # Midrank empirical CDF, including a deterministic tie convention.
                ordered=np.sort(v)
                left=np.searchsorted(ordered,v,side='left'); right=np.searchsorted(ordered,v,side='right')
                target[keep]=.5*(left+right)/len(v)
    else: raise ValueError(method)
    return (target-target.mean())/max(target.std(),1e-8),diagnostics


def cell_coefficients(contexts,c,t,scenario):
    if not scenario['positive_labels']: return coefficients(contexts,c,t)
    # Common positive scale -> multiplier -> additive location/scale -> nonlinear shape.
    # 'additive' keeps the multiplier at strength 1 and scales only the additive baseline:
    # W = a + b exp(0.6R) with a = s(.6+.3tanh(z1)+.1t), so log + cell FE is misspecified.
    stage=scenario['psi_stage']; strength=scenario['psi_strength'] if stage!='shape' else 1.
    a=np.zeros_like(np.asarray(c),dtype=float); b=np.ones_like(a); k=np.ones_like(a)
    if stage!='common': b=np.exp((1. if stage=='additive' else strength)*(.5*np.tanh(contexts[c,1])+.1*t))
    if stage in ('location_scale','shape','additive'): a=strength*(.6+.3*np.tanh(contexts[c,0])+.1*t)
    if stage=='shape': k=.6+.2*np.tanh(contexts[c,0])
    return a,b,k



class PopulationDGP(StructuralDGP):
    """Fixed, independent population calibration of within-cell dispersion and baseline noise."""
    def __init__(self,cfg):
        super().__init__(cfg); self.cfg=cfg; self.cache={}
    def unscaled_h(self,rng,context,composition):
        u=rng.uniform(-1.,1.,size=(len(context),3))
        return BOUND*np.tanh(np.arctanh(u)+composition*np.asarray(context)[:,None]*np.array([1.,.5,-.5]))
    def configure(self,truth,composition):
        key=(truth,float(composition)); self.composition=composition
        if key not in self.cache:
            rng=np.random.default_rng(seed_for(self.cfg.function_seed,'population',composition))
            context=rng.normal(size=100_000)
            a=self.unscaled_h(rng,context,composition); b=self.unscaled_h(rng,context,composition)
            hmean=np.mean(np.vstack((a,b)),axis=0); hsd=np.std(np.vstack((a,b)),axis=0)
            a=(a-hmean)/hsd; b=(b-hmean)/hsd
            fa=self.raw(a,truth); fb=self.raw(b,truth)
            mean=float(np.mean(np.r_[fa,fb])); sd=float(np.sqrt(.5*np.mean((fb-fa)**2)))
            gap=abs((fb-fa)/sd)
            noise=self.cfg.noise_sd
            if self.cfg.bayes_target is not None:
                lo,hi=1e-4,100.
                for _ in range(60):
                    mid=(lo+hi)/2
                    risk=np.mean(1/(1+np.exp(np.minimum(gap/(mid*np.sqrt(6)/np.pi),700))))
                    if risk<self.cfg.bayes_target: lo=mid
                    else: hi=mid
                noise=(lo+hi)/2
            risk=float(np.mean(1/(1+np.exp(np.minimum(gap/(noise*np.sqrt(6)/np.pi),700))))) if noise else 0.
            self.cache[key]=(hmean,hsd,mean,sd,noise,risk)
        self.hmean,self.hsd,mean,sd,self.noise_sd,self.baseline_oracle_risk=self.cache[key]
        self.hlow=(-BOUND-self.hmean)/self.hsd; self.hhigh=(BOUND-self.hmean)/self.hsd
        self.moments[truth]=(mean,sd)
    def draw_h(self,rng,context):
        return (self.unscaled_h(rng,context,self.composition)-self.hmean)/self.hsd
