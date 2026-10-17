"""Direct checks against the two manuscripts' objective equations."""
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .config import preset,runtime,validate
from .core import Features,Estimator,prepare,tensor,score_objective
from .design import PopulationDGP,make_design,measurement,inverse_measurement,level_target,unit_errors
from .experiments import scenario,weights

def run_checks():
    d=preset('smoke'); d.update(models=['linear'],truths=['linear'],cities=4)
    cfg=runtime(d); dgp=PopulationDGP(cfg)
    base=make_design(cfg,dgp,1,16,'linear',scenario())
    tr=base['train']; i,j=base['pairs']; w=base['pair_weights']
    assert np.all(tr['cell'][i]==tr['cell'][j]) and np.all(i!=j)
    np.testing.assert_allclose(w.sum(),1.)
    for source in range(len(tr['x'])): np.testing.assert_allclose(w[i==source].sum(),1/len(tr['x']))
    assert len(base['links']['both'])==2*base['budget']
    for arm in ('temporal','cross_city','both'): assert base['link_stats'][arm]['link_gap_rms']<1e-12
    assert not set(map(tuple,tr['h'])) & set(map(tuple,base['val']['h']))
    for kind,v in [('drift',.15),('random_false',.3),('systematic_false',.3)]:
        x=make_design(cfg,dgp,1,16,'linear',scenario(link_case=kind,link_severity=v))
        np.testing.assert_array_equal(x['train']['x'],tr['x'])
        if kind=='drift': assert x['link_stats']['temporal']['link_h_max']<=v+1e-12
        if kind=='systematic_false': np.testing.assert_allclose(x['link_stats']['temporal']['link_upward_share'],x['link_stats']['temporal']['link_false_share'])
    cfg.acquisition_strength=0.
    zero=make_design(cfg,dgp,1,16,'linear',scenario(acquisition_strength=0.))
    ee=zero['links']['temporal']; np.testing.assert_array_equal(zero['x'][ee[:,0]],zero['x'][ee[:,1]])
    cfg.acquisition_strength=1.
    for shape in ('tanh','asinh'):
        for stage,v in [('common',0.),('multiplier',1.),('location_scale',1.),('shape',1.),('additive',10.)]:
            ss=scenario(positive_labels=True,psi_shape=shape,psi_stage=stage,psi_strength=v,psi_delta=v if stage=='shape' else 0.)
            x=make_design(cfg,dgp,1,16,'linear',ss)
            assert np.all(x['train']['w']>0)
            np.testing.assert_array_equal(np.sign(tr['w'][j]-tr['w'][i]),np.sign(x['train']['w'][j]-x['train']['w'][i]))
            inv,_=level_target('oracle_inverse',x,cfg,ss); r=x['train']['r']
            np.testing.assert_allclose(inv,(r-r.mean())/r.std(),atol=1e-10)
    data=prepare(base,Features(tr['x'],'linear',cfg),cfg)
    edges={arm:tensor(base['links'][arm],cfg,True) for arm in ('temporal','cross_city')}
    for loss in ('common_logit','heterogeneous_logit','ordinal_smooth','squared'):
        torch.manual_seed(10); cfg.fit_cell_effects=loss=='squared'
        model=Estimator(data['x'].shape[1],'linear',loss,cfg).double(); local=data.copy()
        local['target']=tensor(tr['f'],cfg)
        lam=(.2,.4); q,parts=score_objective(model,local,edges,lam,.0,.1)
        score=model.scorer(local['x']); gap=score[local['j']]-score[local['i']]; var=.5*(local['weights']*gap.square()).sum()
        if loss=='common_logit':
            ref=(local['weights']*F.binary_cross_entropy_with_logits(gap,(local['y']+1)/2,reduction='none')).sum()
        elif loss=='heterogeneous_logit':
            ref=(local['weights']*F.binary_cross_entropy_with_logits(model.rho()[local['cells']]*gap/var.sqrt(),(local['y']+1)/2,reduction='none')).sum()
        elif loss=='ordinal_smooth': ref=(local['weights']*torch.sigmoid(-local['y']*gap/var.sqrt()/.1)).sum()
        else: ref=(score[:len(tr['f'])]-local['target']).square().mean()
        np.testing.assert_allclose(float(parts[0].detach()),float(ref.detach()),atol=1e-12)
        if loss!='squared':
            penalty=sum(weight*(score[e[:,1]]-score[e[:,0]]).square().mean()/var for e,weight in zip(edges.values(),lam))
            np.testing.assert_allclose(float(q.detach()),float((ref+penalty).detach()),atol=1e-12)
        q.backward()
        for p in model.parameters():
            if not p.requires_grad: continue
            grad=p.grad.detach().flatten().clone()
            for k in range(p.numel()):
                flat=p.view(-1); old=float(flat[k].detach()); eps=1e-6
                with torch.no_grad(): flat[k]=old+eps
                plus=float(score_objective(model,local,edges,lam,0.,.1)[0].detach())
                with torch.no_grad(): flat[k]=old-eps
                minus=float(score_objective(model,local,edges,lam,0.,.1)[0].detach())
                with torch.no_grad(): flat[k]=old
                np.testing.assert_allclose((plus-minus)/(2*eps),grad[k],atol=4e-6,rtol=3e-4)
        if loss=='common_logit':
            assert model.raw_rho.numel()==0
            oldcross=float(parts[0].detach()); oldpen=float(parts[1].detach())
            with torch.no_grad():
                weight=model.scorer.layers[-1].weight; bound=model.scorer.weight_bound
                weight.copy_(bound*torch.atanh(3*torch.tanh(weight/bound)))
            _,new=score_objective(model,local,edges,lam,0.,.1)
            assert abs(float(new[0].detach())-oldcross)>1e-4
            np.testing.assert_allclose(float(new[1].detach()),oldpen,rtol=1e-12)
    for truth in ('linear','quadratic','smooth','regime'):
        for composition in (0.,.7):
            dgp.configure(truth,composition)
            np.testing.assert_allclose(dgp.baseline_oracle_risk,.2,atol=1e-12)
    assert (0.,0.) in weights(d,'both',True) and (.3,0.) in weights(d,'both',True) and (0.,.3) in weights(d,'both',True)
    from .runner import open_database
    with tempfile.TemporaryDirectory() as folder:
        c=open_database(Path(folder),{'a':1},False); c.close()
        c=open_database(Path(folder),{'a':1},True); c.close()
        try: open_database(Path(folder),{'a':2},True)
        except ValueError: pass
        else: raise AssertionError('Incompatible resume accepted')
    validate(d)
    print('Paper checks passed: exact BCE identities; separate lambda_t/lambda_c; tract-uniform weights; all four gradients including cell effects; common-score scale sensitivity; penalty invariance; positive-label inverses; exact/drifting/false links; baseline Bayes calibration; tuning grid; resume guard.')
