"""Read-only location/scale accounting; no gradient updates or new arm selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import diagnose_e6_gradients as diag
from ordinal_mc.design import inverse_measurement


def main(a):
    folder=a.run/'replications'/'r001_c32_quadratic_nn';saved=torch.load(folder/'checkpoints.pt',map_location='cpu',weights_only=False)
    d=saved['config'];cfg=diag.e5_runtime(d,32);base,variants=diag.build(cfg,diag.PopulationDGP(cfg),1,'quadratic',d)
    features=diag.core.Features(base['train']['x'],'nn',cfg)
    model=diag.core.Estimator(7,'nn','common_logit',cfg).double();model.load_state_dict(saved['models']['A']['scorer'])
    q=variants['nonlinear']['train'];r=inverse_measurement(q['w'],q['a'],q['b'],q['kappa'],1.,'tanh')
    s=diag.core.predictions(model,features,q['x'],cfg);variance=float(np.var(r))
    beta=np.linalg.lstsq(np.column_stack([np.ones(len(s)),s]),r,rcond=None)[0]
    means=dict(initial_score_mean=float(s.mean()),initial_score_sd=float(s.std()),R_mean=float(r.mean()),R_sd=float(r.std()),
        direct_standardized_R_mse=float(np.mean((s-r)**2)/variance),
        mean_offset_contribution=float((s.mean()-r.mean())**2/variance),
        after_centering_standardized_R_mse=float(np.mean((s-s.mean()+r.mean()-r)**2)/variance),
        fitting_affine_intercept=float(beta[0]),fitting_affine_slope=float(beta[1]),
        after_affine_fit_standardized_R_mse=float(np.mean((beta[0]+beta[1]*s-r)**2)/variance),
        diagnostic_only=True,additional_scorer_updates=0)
    path=a.run.parent/'e6_inverse_rank_diagnostic'/'location_scale_accounting.json'
    if path.exists(): raise ValueError('Accounting output exists')
    path.write_text(json.dumps(means,indent=2));print(json.dumps(means,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1): main(a)
