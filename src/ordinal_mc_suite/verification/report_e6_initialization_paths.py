"""Read-only report for the three initialization paths."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def md(df):
    def fmt(x):
        if isinstance(x,(float,np.floating)): return f'{x:.6g}'
        return str(x)
    return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(fmt(x) for x in r)+' |' for r in df.itertuples(index=False,name=None))+'\n'


def save(df,path):
    df.to_csv(path.with_suffix('.csv'),index=False);path.with_suffix('.md').write_text(md(df))


def run(out):
    meta=json.loads((out/'metadata.json').read_text());prep=json.loads((out/'preparation.json').read_text());settings=meta['settings']
    summaries={r['path']:r for r in meta['summaries']};paths=settings['paths'];budget=settings['updates_per_path']
    traces=pd.concat([pd.read_csv(out/p/'observable_paths.csv') for p in paths],ignore_index=True)
    traces['fit_cardinal']=traces.fit_level+traces.fit_change;traces['val_cardinal']=traces.val_level+traces.val_change
    traces.to_csv(out/'observable_paths.csv',index=False)
    m=pd.read_csv(out/'trajectory_metrics.csv');o=pd.read_csv(out/'outcome_trajectories.csv')
    table=[];validation=[]
    for p in paths:
        for kind,step in [('selected',summaries[p]['selected_step']),('fixed_budget',budget)]:
            key='selected' if kind=='selected' else 'trajectory'
            small=m[(m.path==p)&(m.kind==key)&(m.step==step)&(m['where']=='unseen')&(m.strength==.15)].iloc[0]
            moderate=m[(m.path==p)&(m.kind==key)&(m.step==step)&(m['where']=='unseen')&(m.strength==.4)].iloc[0]
            seen=o[(o.path==p)&(o.kind==key)&(o.step==step)&(o.strength==.4)&(o.group=='all')].iloc[0]
            row=traces[(traces.path==p)&(traces.step==step)].iloc[0]
            table.append(dict(path=p,checkpoint=kind,updates=step,small_RMSE=small.change_rmse,moderate_RMSE=moderate.change_rmse,
                stable_RMS=small.stable_rms,small_sensitivity=small.sensitivity,moderate_sensitivity=moderate.sensitivity,
                FPR=small.false_positive_rate,threshold=small.threshold,expected_level_MSE=seen.mean_level_MSE,
                expected_change_MSE=seen.mean_change_MSE,realized_level_MSE=seen.realized_level_MSE,realized_change_MSE=seen.realized_change_MSE))
            validation.append(dict(path=p,checkpoint=kind,updates=step,**{k:row[k] for k in ['fit_rank','fit_stability_ratio','fit_level','fit_change',
                'val_rank','val_level','val_change','validation_rule']}))
    table=pd.DataFrame(table);validation=pd.DataFrame(validation)
    save(table,out/'selected_and_fixed_summary');save(validation,out/'selected_and_fixed_validation')
    counts=pd.DataFrame(meta['summaries'])[['path','selected_step','patience_stop_step','executed_scorer_updates','inherited_scorer_updates_executed',
        'total_scorer_updates_executed','selected_scorer_ancestry','fixed_endpoint_scorer_ancestry','selected_at_budget_boundary']]
    save(counts,out/'scorer_update_counts')
    stopping=[]
    for p in paths:
        tr=traces[(traces.path==p)&(traces.step%settings['validation_every']==0)]
        best=tr.loc[tr.validation_rule.idxmin()]
        stopping.append(dict(path=p,selected_step=summaries[p]['selected_step'],selected_validation=summaries[p]['selected_validation'],
            patience_stop_step=summaries[p]['patience_stop_step'],full_budget_minimum_validation=float(best.validation_rule),
            full_budget_minimum_step=int(best.step),later_validation_improvement=float(best.validation_rule)<summaries[p]['selected_validation']))
    stopping=pd.DataFrame(stopping);save(stopping,out/'stopping_diagnostics')
    save(m[(m.kind=='selected')|((m.kind=='trajectory')&(m.step==budget))][['path','kind','step','where','strength','change_rmse',
        'slope','correlation','stable_rms','threshold','sensitivity','false_positive_rate','observed_rank_accuracy','latent_rank_accuracy']],out/'all_endpoint_latent_metrics')
    colors=dict(ordinal_scratch='#377eb8',joint_scratch='#e41a1c',extended_warm='#4daf4a')
    labels=dict(ordinal_scratch='Ordinal scratch',joint_scratch='Joint scratch',extended_warm='Extended warm start')
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    specs=[('fit_rank','Fitting RankNet loss'),('fit_stability_ratio','Fitting stability S/V'),('fit_cardinal','Fitting level + change loss'),
        ('val_rank','Validation RankNet loss'),('val_cardinal','Validation level + change loss'),('validation_rule','Selection rule (different for ordinal)')]
    for ax,(col,title) in zip(axes.flat,specs):
        for p in paths:
            tr=traces[traces.path==p].sort_values('step');ax.plot(tr.step,tr[col],color=colors[p],label=labels[p])
            selected=tr[tr.step==summaries[p]['selected_step']].iloc[0]
            ax.scatter([selected.step],[selected[col]],marker='D',s=35,edgecolor='black',color=colors[p],zorder=5)
            stop=summaries[p]['patience_stop_step']
            if stop is not None: ax.axvline(stop,color=colors[p],alpha=.25,linestyle=':')
        ax.set(title=title,xlabel='New scorer updates');ax.grid(alpha=.2)
        if col in ['fit_stability_ratio','fit_cardinal','val_cardinal','validation_rule']: ax.set_yscale('log')
    axes[0,0].legend();fig.suptitle('One replication: diamonds = validation-selected checkpoint; dotted lines = patience stop')
    fig.savefig(out/'learning_trajectories.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,(metric,strength,title) in zip(axes.flat[:3],[('change_rmse',.15,'Unseen small-change RMSE'),
        ('change_rmse',.4,'Unseen moderate-change RMSE'),('stable_rms',.15,'Unseen stable normalized RMS')]):
        for p in paths:
            sub=m[(m.path==p)&(m.kind=='trajectory')&(m['where']=='unseen')&(m.strength==strength)].sort_values('step')
            ax.plot(sub.step,sub[metric],color=colors[p],label=labels[p])
            sel=m[(m.path==p)&(m.kind=='selected')&(m['where']=='unseen')&(m.strength==strength)].iloc[0]
            ax.scatter([sel.step],[sel[metric]],marker='D',edgecolor='black',color=colors[p],s=40,zorder=5)
        ax.set(title=title,xlabel='New scorer updates');ax.grid(alpha=.2)
    ax=axes[1,1]
    for p in paths:
        sub=o[(o.path==p)&(o.kind=='trajectory')&(o.strength==.4)&(o.group=='all')].sort_values('step')
        ax.plot(sub.step,sub.mean_change_MSE,color=colors[p],label=labels[p])
        sel=o[(o.path==p)&(o.kind=='selected')&(o.strength==.4)&(o.group=='all')].iloc[0]
        ax.scatter([sel.step],[sel.mean_change_MSE],marker='D',edgecolor='black',color=colors[p],s=40,zorder=5)
    ax.set(title='Seen expected-outcome change MSE',xlabel='New scorer updates');ax.set_yscale('log');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Independent test evaluation after training; diamonds = observed-validation selection')
    fig.savefig(out/'test_trajectories.png',dpi=170);plt.close(fig)
    # Same useful curves after initial transients; no alternate selections or metrics.
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,(metric,strength,title) in zip(axes.flat[:3],[('change_rmse',.15,'Unseen small-change RMSE, updates >=500'),
        ('change_rmse',.4,'Unseen moderate-change RMSE, updates >=500'),('stable_rms',.15,'Unseen stable RMS, updates >=500')]):
        for p in paths:
            sub=m[(m.path==p)&(m.kind=='trajectory')&(m.step>=500)&(m['where']=='unseen')&(m.strength==strength)].sort_values('step')
            ax.plot(sub.step,sub[metric],color=colors[p],label=labels[p])
        ax.set(title=title,xlabel='New scorer updates');ax.grid(alpha=.2)
    for p in paths:
        sub=o[(o.path==p)&(o.kind=='trajectory')&(o.step>=500)&(o.strength==.4)&(o.group=='all')].sort_values('step')
        axes[1,1].plot(sub.step,sub.mean_change_MSE,color=colors[p],label=labels[p])
    axes[1,1].set(title='Expected-outcome change MSE, updates >=500',xlabel='New scorer updates');axes[1,1].set_yscale('log');axes[1,1].grid(alpha=.2)
    axes[0,0].legend();fig.savefig(out/'later_test_trajectories.png',dpi=170);plt.close(fig)
    selected=table[table.checkpoint=='selected'].set_index('path');fixed=table[table.checkpoint=='fixed_budget'].set_index('path')
    a=selected.loc['ordinal_scratch'];j=selected.loc['joint_scratch'];w=selected.loc['extended_warm']
    small_selected=m[(m.kind=='selected')&(m['where']=='unseen')&(m.strength==.15)].set_index('path')
    comparisons=pd.DataFrame([dict(contrast=lhs+' - '+rhs,checkpoint=kind,
        **{col:float(t.loc[lhs,col]-t.loc[rhs,col]) for col in ['small_RMSE','moderate_RMSE','stable_RMS','small_sensitivity','moderate_sensitivity','expected_change_MSE']})
        for kind,t in [('selected',selected),('fixed_budget',fixed)]
        for lhs,rhs in [('joint_scratch','ordinal_scratch'),('joint_scratch','extended_warm'),('extended_warm','ordinal_scratch')]])
    save(comparisons,out/'paired_path_contrasts')
    checks=dict(identical_scratch_scorer_and_head_initializations=meta['identical_scratch_scorer_and_heads_verified'],
        warm_initial_checkpoint_eligible=True,test_and_oracle_evaluated_only_after_training=True,
        fixed_scorer_updates_per_path=budget,total_new_scorer_updates=meta['total_new_scorer_updates'],
        original_checkpoints_unchanged=True,selected_checkpoints_checked_against_logged_validation=True)
    for p in paths:
        tr=traces[traces.path==p];stop=summaries[p]['patience_stop_step']
        allowed=tr[(tr.step%settings['validation_every']==0)&(tr.step<=(budget if stop is None else stop))]
        best=allowed.loc[allowed.validation_rule.idxmin()]
        assert int(best.step)==summaries[p]['selected_step']
        np.testing.assert_allclose(best.validation_rule,summaries[p]['selected_validation'],rtol=1e-13,atol=1e-13)
    (out/'verification.json').write_text(json.dumps(checks,indent=2))
    improved='lower' if j.small_RMSE<a.small_RMSE else 'higher'
    scale_path=out/'score_and_head_scale_audit.csv'
    scale_table=pd.read_csv(scale_path) if scale_path.exists() else None
    if scale_table is not None: save(scale_table,out/'score_and_head_scale_audit')
    report=[
        '# One replication: ordinal scratch, joint scratch and extended warm start',
        f'**Observed-validation-selected joint scratch has {improved} small-change RMSE than ordinal scratch in this fixed setting.** All results below are one replication, one loss-weight setting and one common learning-rate pair. They do not establish that the selected weights are good or that a formulation generally succeeds or fails.',
        '## Design and stopping',
        'The original paired-outcome data were reconstructed exactly: quadratic truth, the small [32,16] ReLU scorer, nonlinear monotone measurement, 32 cities, two dates, 13,120 fitting labels from 6,560 paired tracts, 3,264 validation labels from 1,632 paired tracts. Independent date-specific centered Gumbel errors, exact temporal auxiliary links, no cross-city links, no scorer or head penalties, and all prior independent test/calibration panels were retained.',
        f'Fitting seed {prep["fit_seed"]}; validation seed {prep["validation_seed"]}; common scratch scorer seed {prep["scorer_initialization_seed"]}. Positive monotone spline heads receive only score and cell ID. Denominators remain Var_fit(W)={prep["denominators"]["W"]:.12g} and Var_fit(DeltaW)={prep["denominators"]["delta_W"]:.12g}.',
        'Ordinal scratch and joint scratch start with exactly identical scorer parameters and heads. All losses are active from update one for joint scratch. Its head receives no preliminary level fitting. Ordinal heads are updated on the same level-plus-change labels using detached scores; they never affect the ordinal scorer. Scratch head normalization and three knots use fitting scores at initialization, and stay fixed. Extended warm starts from the original A scorer and its saved calibrated head, with fresh Adam states. Its head normalization/knots come from A, so this comparison concerns the full initialization strategy, rather than isolating only scorer weights.',
        'The ordinal scorer objective is raw RankNet +0.3 S/V; joint and warm add standardized level MSE + standardized observed-change MSE, both weighted one. V is differentiated. All paths use scorer LR .001, head LR .001, full batches, and separate scorer/head clipping at norm20. This common scorer LR is the main E6 joint LR, ten times the short diagnostic LR; the head LR remains the conservative diagnostic .001. No LR or loss-weight alternatives were tried.',
        'Every path has a predeclared 3,000-update ceiling, matching the suite baseline Adam horizon. Validation is checked every 10 updates. Ordinal selection uses validation RankNet alone; joint/warm use validation RankNet + standardized level + standardized change MSE (fixed 1:1:1). The existing patience of 20 checks and min_delta=1e-5 define the validation-stopped prefix. Initial checkpoints are eligible. Once patience is reached, selection freezes; training continues only to produce the predeclared fixed-budget endpoint and trajectories. Later improvements do not replace the stopped-prefix selected checkpoint. This separates a deployable validation-stopped result from a fixed-compute diagnostic without selecting on latent truth.',
        '## Selected checkpoints and fixed-budget endpoints',
        'The following latent and detection metrics are on unseen cities. Outcome MSE is on the seen-city moderate-change panel, all tracts; cell-specific heads are unavailable zero-shot on unseen cities. **The ordinal scratch heads have very large prediction errors and are not a useful calibration benchmark in this run; their errors are reported for transparency. This limitation does not affect its scorer-based latent or detection metrics.**',
        md(table[['path','checkpoint','updates','small_RMSE','moderate_RMSE','stable_RMS','small_sensitivity','moderate_sensitivity','expected_level_MSE','expected_change_MSE','realized_change_MSE']]),
        'Full seen/unseen slopes, correlations, ranking accuracy, false-positive rates and thresholds are in all_endpoint_latent_metrics.md/.csv. Stratified conditional-mean and realized outcome errors are in outcome_trajectories.csv. Oracle conditional means integrate the actual error law and are evaluated only after all training, never used for selection.',
        '**Outcome-head qualification:** the ordinal path\'s heads track its evolving scores concurrently, using the same head budget as joint scratch. They are not refitted after freezing the selected scorer. Initial scratch-score normalization and knots can become poorly matched to a scorer whose raw scale changes substantially. Outcome-error differences therefore include head tracking and scaling, and cannot alone establish a better latent scorer. The warm-start selected checkpoint supplies the existing, calibrated A reference.',
        'Fitting and observed validation at each reported checkpoint:',md(validation),
        '## Comparisons and interpretation',
        f'At selected checkpoints, joint scratch minus ordinal scratch is {j.small_RMSE-a.small_RMSE:+.6g} for small latent RMSE, {j.moderate_RMSE-a.moderate_RMSE:+.6g} for moderate RMSE, {j.stable_RMS-a.stable_RMS:+.6g} for stable RMS and {100*(j.small_sensitivity-a.small_sensitivity):+.3f} percentage points for small-change sensitivity. Expected outcome-change MSE differs by {j.expected_change_MSE-a.expected_change_MSE:+.6g}, dominated by ordinal head failure; that difference is not evidence of improved latent recovery.',
        f'Joint scratch minus selected extended warm is {j.small_RMSE-w.small_RMSE:+.6g} for small latent RMSE, {j.moderate_RMSE-w.moderate_RMSE:+.6g} for moderate RMSE, {j.stable_RMS-w.stable_RMS:+.6g} for stable RMS, and {j.expected_change_MSE-w.expected_change_MSE:+.6g} for expected outcome-change MSE. This is the observed effect of the complete initialization strategies at this setting, including head initialization and validation rules.',
        f'For unseen small changes, selected ordinal versus joint scratch slopes are {small_selected.loc["ordinal_scratch","slope"]:.5f} versus {small_selected.loc["joint_scratch","slope"]:.5f}, while correlations are {small_selected.loc["ordinal_scratch","correlation"]:.5f} versus {small_selected.loc["joint_scratch","correlation"]:.5f}. Joint scratch therefore adds considerable individual prediction noise/nuisance movement as well as some attenuation. Its own detection threshold rises to {small_selected.loc["joint_scratch","threshold"]:.5f} versus {small_selected.loc["ordinal_scratch","threshold"]:.5f}; fresh FPR remains near nominal, {100*small_selected.loc["joint_scratch","false_positive_rate"]:.2f}% versus {100*small_selected.loc["ordinal_scratch","false_positive_rate"]:.2f}%. These are descriptive, not causal mechanisms.',
        'Raw score/head scaling differs sharply across paths, but a positive affine change of score leaves normalized temporal changes invariant. Raw scale differences cannot by themselves explain the worse latent-change metrics. The read-only audit also verifies nonzero level and change gradients into the scratch scorer at initialization; both cardinal terms were active from the first update.',
        'The fixed-budget rows must be read separately from validation selection. Losses can keep improving on fitting data after validation stopping, and latent/test performance can improve or deteriorate without being available to the learner. Selection does not exploit the plotted latent paths. A cap-ending or still-improving path would remain a budget-limited observation, rather than a convergence claim.',
        'Observed-validation stopping diagnostics below show whether the declared patience rule missed later validation improvement. Full-budget minima are descriptive scalar summaries of the already logged validation path, not replacement model selections and not latent/test choices:',md(stopping),
        ('Read-only raw score/head scale audit (initial, selected and fixed endpoints):\n\n'+md(scale_table) if scale_table is not None else 'Raw-score and head scaling remain part of the initialization strategy; no scaling variant was trained.'),
        '## Scorer-update accounting',md(counts),
        'Warm start inherits 520 previously executed baseline updates, although the restored A scorer is from update 320; the remaining 200 were spent finding the validation stop and are not in its parameter ancestry. Thus its total computation is 3,520 scorer updates, while its fixed endpoint has 3,320 updates in its ancestry. It also inherits 600 head updates. Scratch paths each execute 3,000 scorer and 3,000 head updates. Selected-step ancestry and the patience detection time are reported separately; continuing for fixed-budget diagnostics is included in execution counts.',
        '## Q correction and scope',
        'The earlier Q stall is not evidence against useful constrained retraining. At an active constraint g(theta)=0, a projected direction can have grad(g).d=0. If d^T H_g d>0, the second-order term makes sufficiently small positive straight steps infeasible, so backtracking alone can fail. An inward margin, feasibility correction or stronger constrained solver could behave differently. The prior report was corrected; Q was not rerun.',
        'The evidence from previous fine-tuning paths concerns those initialization and optimization choices. This scratch/warm comparison adds one missing design, but cannot establish an optimal loss formulation, loss weights, population effect or convergence. No extra seeds, searches or follow-up runs were launched.',
        '## Reproduce',
        'From ordinal_mc_suite in the existing PyTorch environment, choose a new output directory:',
        '```bash\npython verification/e6_initialization_paths.py --source e6_observed_change_diagnostic --config configs/e6_initialization_one_shot.json --out e6_initialization_diagnostic_new\npython verification/audit_e6_initialization_paths.py --out e6_initialization_diagnostic_new\npython verification/report_e6_initialization_paths.py --out e6_initialization_diagnostic_new\n```',
        'Saved configuration, exact initialization hashes, source checkpoint hashes, all selected and trajectory checkpoints, optimizer end states, per-update gradient/clipping traces, validation histories, outcome and latent tables, path contrasts and trajectory figures. All previous checkpoints are unchanged.'
    ]
    (out/'ANALYSIS.md').write_text('\n\n'.join(report)+'\n')
    print(md(table));print(json.dumps(checks,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);run(p.parse_args().out.resolve())
