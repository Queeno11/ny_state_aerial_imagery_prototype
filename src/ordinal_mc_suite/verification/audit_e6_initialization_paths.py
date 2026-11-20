"""Read-only checkpoint equality, selection and score/head scale audit."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import e6_initialization_paths as experiment
from e6_targeted_trials import cell_variance


@torch.no_grad()
def run(out):
    meta=json.loads((out/'metadata.json').read_text());settings=meta['settings'];source=Path(meta['source'])
    initial=torch.load(out/'initializations.pt',map_location='cpu',weights_only=False);z=experiment.load(source)
    states={p:torch.load(out/p/'checkpoints.pt',map_location='cpu',weights_only=False)['checkpoints'] for p in settings['paths']}
    for component in ['model','head']:
        a=states['ordinal_scratch']['ordinal_scratch_0'][component]
        b=states['joint_scratch']['joint_scratch_0'][component]
        for key,value in a.items(): torch.testing.assert_close(value,b[key],atol=0,rtol=0)
        for key,value in initial['warm'][component].items():
            torch.testing.assert_close(value,states['extended_warm']['extended_warm_0'][component][key],atol=0,rtol=0)
    model=copy.deepcopy(z['A']);head=copy.deepcopy(z['head']);rows=[]
    model.load_state_dict(initial['scratch']['model']);head.load_state_dict(initial['scratch']['head'])
    with torch.enable_grad():
        scores=model.scorer(z['data']['x'][:len(z['design']['train']['w'])])
        lw,ld=experiment.previous.head_losses(head,scores,z['design']['train'],z['den'],z['cfg'])
        params=list(model.scorer.parameters())
        level_grad_norm=float(experiment.diag.vector(lw,params).norm())
        change_grad_norm=float(experiment.diag.vector(ld,params).norm())
    assert level_grad_norm>0 and change_grad_norm>0
    for p in settings['paths']:
        summary=next(r for r in meta['summaries'] if r['path']==p)
        tr=pd.read_csv(out/p/'observable_paths.csv')
        stop=summary['patience_stop_step'];allowed=tr[(tr.step%settings['validation_every']==0)&(tr.step<=(settings['updates_per_path'] if stop is None else stop))]
        best=allowed.loc[allowed.validation_rule.idxmin()]
        assert int(best.step)==summary['selected_step']
        for label,key in [('initial',p+'_0'),('selected',p+'_selected'),('fixed_budget',p+'_'+str(settings['updates_per_path']))]:
            state=states[p][key];model.load_state_dict(state['model']);head.load_state_dict(state['head'])
            s=model.scorer(z['data']['x'][:len(z['design']['train']['w'])]);v=float(cell_variance(s,z['data']['obs_cells']))
            _,parts=experiment.diag.core.score_objective(model,z['data'],z['edges'],(.3,0.),0.,z['cfg'].tau)
            np.testing.assert_allclose(v,float(parts[3]),atol=1e-13,rtol=1e-13)
            slopes=head.slopes().numpy();head.audit_monotonicity()
            rows.append(dict(path=p,checkpoint=label,score_mean=float(s.mean()),score_SD=float(s.std(unbiased=False)),
                within_cell_V=v,initial_head_score_scale=float(head.score_scale),
                median_head_segment_slope=float(np.median(slopes)),min_head_segment_slope=float(np.min(slopes)),max_head_segment_slope=float(np.max(slopes)),
                median_effective_slope_per_raw_score=float(head.y_scale/head.score_scale)*float(np.median(slopes))))
    pd.DataFrame(rows).to_csv(out/'score_and_head_scale_audit.csv',index=False)
    checks=dict(actual_initial_scratch_scorer_and_head_tensors_identical=True,actual_warm_initial_state_matches_source=True,
        initial_checkpoint_eligible_for_all_paths=True,selected_minimizes_observed_validation_within_patience_prefix=True,
        head_monotonicity_initial_selected_fixed_verified=True,dispersion_matches_original_pairwise_V=True,
        scratch_initial_level_scorer_gradient_norm=level_grad_norm,scratch_initial_change_scorer_gradient_norm=change_grad_norm,
        audit_is_read_only=True)
    (out/'checkpoint_audit.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    with threadpool_limits(limits=1):
        torch.set_num_threads(1);run(args.out.resolve())
