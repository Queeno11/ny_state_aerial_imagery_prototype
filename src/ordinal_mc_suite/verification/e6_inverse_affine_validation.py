"""Fit positive affine noisy-R calibration on fitting scores; evaluate held-out R."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import diagnose_e6_gradients as diag
from ordinal_mc.design import inverse_measurement
from e6_targeted_trials import targets


def run(a):
    folder=a.run/'replications'/'r001_c32_quadratic_nn';source=folder/'checkpoints.pt'
    before=hashlib.sha256(source.read_bytes()).hexdigest();saved=torch.load(source,map_location='cpu',weights_only=False)
    d=saved['config'];cfg=diag.e5_runtime(d,32);base,variants=diag.build(cfg,diag.PopulationDGP(cfg),1,'quadratic',d)
    features=diag.core.Features(base['train']['x'],'nn',cfg)
    A=diag.core.Estimator(7,'nn','common_logit',cfg).double();A.load_state_dict(saved['models']['A']['scorer'])
    labels={};scores={}
    for group in ['train','val']:
        q=variants['nonlinear'][group]
        labels[group]=inverse_measurement(q['w'],q['a'],q['b'],q['kappa'],1.,'tanh')
        scores[group]=diag.core.predictions(A,features,q['x'],cfg)
    s=scores['train'];r=labels['train'];variance=float(np.var(r))
    # Positive-slope constraint is inactive in this sample; no validation fitting.
    intercept,slope=np.linalg.lstsq(np.column_stack([np.ones(len(s)),s]),r,rcond=None)[0]
    if slope<=0: raise ValueError('Unconstrained fit has nonpositive slope; implement constrained fit explicitly')
    diagnostic=a.run.parent/'e6_inverse_rank_diagnostic';selected=torch.load(diagnostic/'diagnostic_checkpoints.pt',map_location='cpu',weights_only=False)
    inverse=diag.core.Estimator(7,'nn','common_logit',cfg).double();inverse.load_state_dict(selected['inverse_rank'])
    val=diag.tr.validation_data(base['val'],features,cfg);rank_A=float(diag.tr.rank_loss(A,val).detach());rank_I=float(diag.tr.rank_loss(inverse,val).detach())
    metrics={}
    for group in ['train','val']:
        q=base[group];target=labels[group]
        raw=scores[group];calibrated=intercept+slope*raw
        joint=diag.core.predictions(inverse,features,q['x'],cfg)
        metrics[group]={name:dict(raw_mse=float(np.mean((prediction-target)**2)),
                                 standardized_mse=float(np.mean((prediction-target)**2)/variance))
                        for name,prediction in [('frozen_uncalibrated',raw),('frozen_affine',calibrated),('joint_inverse',joint)]}
    # Explicit finite-sample verification of normalized change invariance.
    ii,jj=base['pairs'];v=.5*np.sum(base['pair_weights']*(s[jj]-s[ii])**2)
    calibrated=intercept+slope*s;vc=.5*np.sum(base['pair_weights']*(calibrated[jj]-calibrated[ii])**2)
    np.testing.assert_allclose(vc,slope**2*v,atol=1e-12,rtol=1e-12)
    maximum=0.
    for strength in [.15,.4]:
        for where in ['seen','unseen']:
            p=base['e6_panels'][strength][where]
            s0=diag.core.predictions(A,features,p['x_before'],cfg);s1=diag.core.predictions(A,features,p['x_after'],cfg)
            original=(s1-s0)/np.sqrt(v);changed=((intercept+slope*s1)-(intercept+slope*s0))/np.sqrt(vc)
            maximum=max(maximum,float(np.max(abs(changed-original))))
            np.testing.assert_allclose(changed,original,atol=1e-12,rtol=1e-12)
    assert before==hashlib.sha256(source.read_bytes()).hexdigest()
    result=dict(intercept=float(intercept),slope=float(slope),fit_variance_R=variance,n_fit=len(s),n_validation=len(labels['val']),
        positive_slope_constraint_satisfied=True,coefficients_fitted_on='fitting observations only',network_updates=0,
        metrics=metrics,validation_rank_frozen=rank_A,validation_rank_joint=rank_I,
        common_validation_rule_frozen_affine=rank_A+metrics['val']['frozen_affine']['standardized_mse'],
        common_validation_rule_joint=rank_I+metrics['val']['joint_inverse']['standardized_mse'],
        maximum_normalized_change_difference=maximum,source_checkpoint_unchanged=True)
    path=diagnostic/'affine_validation_comparison.json'
    if path.exists(): raise ValueError('Output exists')
    path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1): run(a)
