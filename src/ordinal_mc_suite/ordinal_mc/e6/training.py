"""Observable-only selection. No test panels, f, H, or true changes are accepted here."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from .. import core


class Tracker:
    def __init__(self, patience, delta):
        self.patience=patience; self.delta=delta; self.best=float('inf')
        self.reference=float('inf'); self.bad=0; self.state=None; self.step=None

    def observe(self, modules, value, step):
        if not np.isfinite(value): raise FloatingPointError('Nonfinite validation criterion')
        if value < self.best:
            self.best=value; self.step=step
            self.state=[core.cpu_state(m) for m in modules]
        if value < self.reference-self.delta: self.reference=value; self.bad=0
        else: self.bad+=1
        return self.bad>=self.patience

    def restore(self, modules):
        for m,s in zip(modules,self.state): m.load_state_dict(s)


def validation_data(val, features, cfg):
    ii=[];jj=[];ww=[];n=len(val['w'])
    for cell in np.unique(val['cell']):
        loc=np.flatnonzero(val['cell']==cell)
        if len(loc)<2: raise ValueError('Need two validation tracts per cell')
        i=np.repeat(loc,len(loc));j=np.tile(loc,len(loc));keep=i!=j
        ii.extend(i[keep]);jj.extend(j[keep]);ww.extend([1/(n*(len(loc)-1))]*int(keep.sum()))
    i=np.asarray(ii,int);j=np.asarray(jj,int)
    return dict(x=core.tensor(features(val['x']),cfg), i=core.tensor(i,cfg,True),
                j=core.tensor(j,cfg,True), y=core.tensor(np.sign(val['w'][j]-val['w'][i]),cfg),
                weights=core.tensor(ww,cfg), w=core.tensor(val['w'],cfg),cell=core.tensor(val['cell'],cfg,True))


def level_data(train, cfg):
    variance=float(np.var(train['w']))
    if variance<=0: raise ValueError('Zero fitting outcome variance')
    return dict(w=core.tensor(train['w'],cfg),cell=core.tensor(train['cell'],cfg,True),variance=variance)


def rank_loss(model,val):
    s=model.scorer(val['x']);gap=s[val['j']]-s[val['i']]
    return (val['weights']*F.softplus(-val['y']*gap)*(val['y']!=0)).sum()


def level_loss(model,head,x,levels):
    s=model.scorer(x[:len(levels['w'])])
    return (head(s,levels['cell'])-levels['w']).square().mean()/levels['variance']


@torch.no_grad()
def observable_rule(model,head,val,variance,d):
    r=float(rank_loss(model,val)); l=float((head(model.scorer(val['x']),val['cell'])-val['w']).square().mean()/variance)
    return d['validation_rank_weight']*r+d['validation_level_weight']*l


def e5_loss(model,data,edges,cfg,d):
    return core.score_objective(model,data,edges,(d['lambda_t'],0.),d['alpha'],cfg.tau)[0]


def joint_loss(model,head,data,edges,levels,cfg,d,weight):
    return e5_loss(model,data,edges,cfg,d)+weight*level_loss(model,head,data['x'],levels)+d['head_training']['kappa_G']*head.penalty()


def adam_selected(modules,groups,objective,validate,steps,check_every,patience,delta,
                  *,stop_on_patience=True,secondary=None):
    """Include initial checkpoint; restore best; count executed optimizer updates."""
    tracker=Tracker(patience,delta);other=Tracker(patience,delta) if secondary else None
    opt=torch.optim.Adam(groups);trace=[];executed=0
    params=[p for m in modules for p in m.parameters() if p.requires_grad]
    def observe(step):
        with torch.no_grad(): value=float(validate());q=float(objective())
        stop=tracker.observe(modules,value,step)
        row=dict(step=step,validation=value,objective=q,best_validation=tracker.best)
        if other:
            with torch.no_grad(): v=float(secondary())
            other.observe(modules,v,step);row['secondary_validation']=v
        trace.append(row)
        return stop
    observe(0)
    reason='budget'
    for step in range(1,steps+1):
        before=[core.cpu_state(m) for m in modules]
        opt.zero_grad(set_to_none=True);q=objective()
        if not bool(torch.isfinite(q)): raise FloatingPointError('Nonfinite objective')
        q.backward();torch.nn.utils.clip_grad_norm_(params,20.);opt.step();executed=step
        try:
            with torch.no_grad(): check=objective()
            if not bool(torch.isfinite(check)): raise FloatingPointError('Nonfinite step')
        except (core.InadmissibleScoreError,FloatingPointError):
            for m,s in zip(modules,before): m.load_state_dict(s)
            # Fail explicitly: silently shortening D would violate matched budgets.
            raise FloatingPointError('Invalid optimizer step; task not committed')
        if step%check_every==0 or step==steps:
            if observe(step) and stop_on_patience: reason='validation_patience';break
    tracker.restore(modules)
    return dict(updates=executed,best_step=tracker.step,validation=tracker.best,stop_reason=reason,trace=trace),other


def baseline(model,data,edges,val,cfg,d):
    b=d['baseline'];params=list(model.scorer.parameters())
    objective=lambda:e5_loss(model,data,edges,cfg,d)
    validate=lambda:rank_loss(model,val)
    info,_=adam_selected([model],[{'params':params,'lr':b['lr']}],objective,validate,
        b['adam_steps'],b['check_every'],b['patience'],d['min_delta'])
    tracker=Tracker(b['lbfgs_patience'],d['min_delta'])
    tracker.observe([model],info['validation'],('adam',info['best_step']))
    trace=info['trace'];calls=0;steps=0
    if b['lbfgs_steps']:
        opt=torch.optim.LBFGS(params,lr=1.,max_iter=b['lbfgs_chunk'],history_size=20,
            tolerance_grad=cfg.grad_tol,tolerance_change=1e-10,line_search_fn='strong_wolfe')
        for start in range(0,b['lbfgs_steps'],b['lbfgs_chunk']):
            before=core.cpu_state(model);oldopt=copy.deepcopy(opt.state_dict());old=float(objective().detach())
            def closure():
                nonlocal calls
                calls+=1;opt.zero_grad(set_to_none=True);q=objective()
                if not bool(torch.isfinite(q)): raise FloatingPointError('Nonfinite L-BFGS probe')
                q.backward();return q
            opt.param_groups[0]['max_iter']=min(b['lbfgs_chunk'],b['lbfgs_steps']-start)
            try:
                opt.step(closure)
                with torch.no_grad(): new=float(objective())
                if not np.isfinite(new) or new>old+1e-12: raise FloatingPointError('Worse L-BFGS endpoint')
            except (core.InadmissibleScoreError,FloatingPointError):
                model.load_state_dict(before);opt.load_state_dict(oldopt);break
            steps=start+opt.param_groups[0]['max_iter']
            with torch.no_grad(): value=float(validate())
            stop=tracker.observe([model],value,('lbfgs',steps))
            trace.append(dict(stage='lbfgs',step=steps,validation=value,objective=new,best_validation=tracker.best))
            if stop: break
    tracker.restore([model]);info.update(validation=tracker.best,best_location=tracker.step,
        lbfgs_steps=steps,lbfgs_closures=calls)
    return info


def fit_head(model,head,data,levels,val,cfg,d,weight):
    settings=d['head_training']
    # Only detached fitting scores are exposed to this optimization.
    with torch.no_grad(): scores=model.scorer(data['x'][:len(levels['w'])]).detach()
    before=core.cpu_state(model)
    objective=lambda:weight*(head(scores,levels['cell'])-levels['w']).square().mean()/levels['variance']+settings['kappa_G']*head.penalty()
    rule=lambda:observable_rule(model,head,val,levels['variance'],d)
    info,_=adam_selected([head],[{'params':head.parameters(),'lr':settings['lr']}],objective,rule,
        settings['steps'],settings['check_every'],settings['patience'],d['min_delta'])
    for k,v in before.items(): torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    head.audit_monotonicity();return info


def fit_joint(model,head,data,edges,levels,val,cfg,d,weight):
    j=d['joint']
    info,_=adam_selected([model,head],[{'params':model.scorer.parameters(),'lr':j['scorer_lr']},
        {'params':head.parameters(),'lr':j['head_lr']}],
        lambda:joint_loss(model,head,data,edges,levels,cfg,d,weight),
        lambda:observable_rule(model,head,val,levels['variance'],d),
        j['steps'],j['check_every'],j['patience'],d['min_delta'])
    head.audit_monotonicity();return info


def continue_ordinal(model,head,data,edges,levels,val,cfg,d,budget):
    j=d['joint']
    info,other=adam_selected([model],[{'params':model.scorer.parameters(),'lr':j['scorer_lr']}],
        lambda:e5_loss(model,data,edges,cfg,d),lambda:rank_loss(model,val),budget,
        j['check_every'],j['patience'],d['min_delta'],stop_on_patience=False,
        secondary=lambda:observable_rule(model,head,val,levels['variance'],d))
    common=copy.deepcopy(model);other.restore([common])
    if info['updates']!=budget: raise AssertionError('Unmatched C/D scorer update counts')
    info['common_rule_validation']=other.best;info['common_rule_best_step']=other.step
    return info,common


def oracle_noisy_R(model,data,edges,fit_r,val_x,val_r,cfg,d):
    """Privileged noisy inverse labels, not the noise-free structural function."""
    settings=d['oracle_R'];variance=float(np.var(fit_r));target=core.tensor(fit_r,cfg);vtarget=core.tensor(val_r,cfg)
    def objective():
        s=model.scorer(data['x']);_,parts=core.score_objective(model,data,edges,(d['lambda_t'],0.),d['alpha'],cfg.tau)
        return (s[:len(target)]-target).square().mean()/variance+parts[1]+d['alpha']*parts[2]
    info,_=adam_selected([model],[{'params':model.scorer.parameters(),'lr':settings['lr']}],objective,
        lambda:(model.scorer(val_x)-vtarget).square().mean()/variance,settings['steps'],
        d['joint']['check_every'],d['joint']['patience'],d['min_delta'])
    return info
