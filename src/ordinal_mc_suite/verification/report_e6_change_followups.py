"""Read-only tables, figures and interpretation of the two bounded follow-ups."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def md(frame):
    def fmt(v):
        if isinstance(v,(float,np.floating)): return f'{v:.6g}'
        return str(v)
    return '| '+' | '.join(frame.columns)+' |\n|'+ '|'.join(['---']*len(frame.columns))+'|\n'+ '\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'


def write_table(frame,path):
    frame.to_csv(path.with_suffix('.csv'),index=False);path.with_suffix('.md').write_text(md(frame))


def run(out):
    metadata=json.loads((out/'metadata.json').read_text());settings=json.loads((out/'config.json').read_text())
    e1=pd.read_csv(out/'experiment1_outcome_trajectories.csv');e2=pd.read_csv(out/'experiment2_outcome_trajectories.csv')
    latent=pd.read_csv(out/'latent_trajectory_metrics.csv');paths=pd.read_csv(out/'observable_paths.csv')
    prep=pd.read_csv(out/'head_preparation.csv');trace=pd.read_csv(out/'optimizer_trace.csv');integration=pd.read_csv(out/'integration_checks.csv')
    def endpoint(data,group='all',strength=.4):
        return data[(data.step==200)&(data.group==group)&(data.strength==strength)].set_index('arm')
    t1=endpoint(e1).loc[['A','O','B','C','D']]
    write_table(t1[['realized_level_MSE','realized_change_MSE','mean_level_MSE','mean_change_MSE','expected_noisy_change_risk']].reset_index(),out/'experiment1_endpoint_table')
    strat=e1[(e1.step==200)&(((e1.group=='stable')&(e1.strength==.4))|(e1.group.isin(['small','moderate'])))].copy()
    write_table(strat[['arm','group','n','mean_level_MSE','mean_change_MSE','mean_level_MSE_standardized','mean_change_MSE_standardized',
        'noise_floor_level','noise_floor_change','expected_noisy_level_risk','expected_noisy_change_risk']],out/'experiment1_stratified_endpoint_table')
    colors=dict(A='#333333',O='#377eb8',B='#4daf4a',C='#ff7f00',D='#e41a1c',F='#4daf4a',U='#e41a1c',Q='#984ea3')
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for arm in ['A','O','B','C','D']:
        for ax,(group,strength,col,title) in zip(axes.flat,[('all',.4,'mean_level_MSE','Expected-outcome level error: all tracts'),
            ('stable',.4,'mean_change_MSE','Expected-outcome change error: stable tracts'),
            ('small',.15,'mean_change_MSE','Expected-outcome change error: small changes'),
            ('moderate',.4,'mean_change_MSE','Expected-outcome change error: moderate changes')]):
            sub=e1[(e1.arm==arm)&(e1.group==group)&(e1.strength==strength)].sort_values('step')
            ax.plot(sub.step,sub[col],label=arm,color=colors[arm],marker='o',markersize=3,linestyle='--' if arm=='A' else '-')
            ax.set(title=title,xlabel='Additional updates',ylabel='Raw MSE');ax.grid(alpha=.2)
    axes[0,0].legend(ncol=5);fig.suptitle('Experiment 1: oracle means evaluate fixed checkpoints; no retraining')
    fig.savefig(out/'experiment1_oracle_trajectories.png',dpi=170);plt.close(fig)
    t2=endpoint(e2).loc[['F','U','Q']]
    write_table(t2[['realized_level_MSE','realized_change_MSE','mean_level_MSE','mean_change_MSE','expected_noisy_change_risk']].reset_index(),out/'experiment2_outcome_endpoint_table')
    lp=paths[paths.step==200].set_index('arm').loc[['F','U','Q']]
    refs=metadata['reference_constraints'];lp=lp.copy()
    lp['rank_relative_change']=lp.rank_fit/refs['rank_A']-1
    lp['stability_relative_change']=lp.stability_ratio/refs['stability_A']-1
    write_table(lp[['fit_level','fit_change','val_level','val_change','rank_fit','stability_ratio','rank_relative_change','stability_relative_change','rank_violation','stability_violation']].reset_index(),out/'experiment2_fitting_endpoint_table')
    final=latent[latent.step==200].copy()
    write_table(final[['arm','where','strength','change_rmse','slope','correlation','stable_rms','threshold','sensitivity','false_positive_rate','observed_rank_accuracy','latent_rank_accuracy']],out/'experiment2_latent_endpoint_table')
    contrasts=[]
    for control,frame in [('F',latent),('U',latent)]:
        for _,q in final[final.arm=='Q'].iterrows():
            ref=frame[(frame.arm==control)&(frame.step==200)&(frame['where']==q['where'])&(frame.strength==q.strength)].iloc[0]
            contrasts.append(dict(contrast='Q-'+control,where=q['where'],strength=q.strength,
                **{k:float(q[k]-ref[k]) for k in ['change_rmse','slope','correlation','stable_rms','threshold','sensitivity','false_positive_rate']}))
    source=Path(metadata['source']);old=pd.read_csv(source/'trajectory_metrics.csv')
    for filename in ['config.json','metadata.json']:
        (out/('source_'+filename)).write_text((source/filename).read_text())
    old_A=old[old.arm=='A']
    old_full=pd.concat([old]+[old_A.assign(step=step) for step in settings['checkpoints'] if step!=0],ignore_index=True)
    old_full.to_csv(out/'experiment1_latent_trajectory_metrics.csv',index=False)
    write_table(old_full[old_full.step==200][['arm','where','strength','change_rmse','slope','correlation','stable_rms','threshold','sensitivity','false_positive_rate']],out/'experiment1_latent_endpoint_table')
    for _,q in final[final.arm=='Q'].iterrows():
        ref=old[(old.arm=='A')&(old['where']==q['where'])&(old.strength==q.strength)].iloc[0]
        contrasts.append(dict(contrast='Q-A',where=q['where'],strength=q.strength,
            **{k:float(q[k]-ref[k]) for k in ['change_rmse','slope','correlation','stable_rms','threshold','sensitivity','false_positive_rate']}))
    write_table(pd.DataFrame(contrasts),out/'experiment2_latent_contrasts')
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for arm in ['F','U','Q']:
        p=paths[paths.arm==arm].copy();p['fit_card']=p.fit_level+p.fit_change;p['val_card']=p.val_level+p.val_change
        axes[0,0].plot(p.step,p.fit_card,color=colors[arm],label=arm)
        axes[0,1].plot(p.step,p.val_card,color=colors[arm],label=arm)
        for ax,col in [(axes[1,0],'change_rmse'),(axes[1,1],'stable_rms')]:
            pp=latent[(latent.arm==arm)&(latent['where']=='unseen')&(latent.strength==.15)].sort_values('step')
            ax.plot(pp.step,pp[col],marker='o',color=colors[arm],label=arm)
    for ax,title in zip(axes.flat,['Fitting cardinal loss','Observed validation cardinal loss','Unseen small-change RMSE','Unseen stable normalized RMS']):
        ax.set(title=title,xlabel='Additional updates');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Experiment 2: fixed budgets, common observed-validation head initialization')
    fig.savefig(out/'experiment2_trajectories.png',dpi=170);plt.close(fig)
    dec=pd.read_csv(out/'experiment1_residual_decomposition.csv')
    dec=dec[(dec.step==200)&(dec.group=='all')&(dec.strength==.4)].copy()
    write_table(dec[['arm','target','residual_mean_0','residual_mean_1','residual_variance_0','residual_variance_1','residual_covariance','residual_cross_product','level_MSE','change_MSE','change_bias_squared','change_variance','identity_discrepancy']],out/'experiment1_endpoint_decomposition')
    scale=pd.read_csv(out/'experiment1_score_scales.csv');scale=scale[(scale.step==200)&(scale.strength==.4)]
    write_table(scale,out/'experiment1_endpoint_score_scales')
    s2=pd.read_csv(out/'experiment2_score_scales.csv');s2=s2[(s2.step==200)&(s2.strength==.4)]
    write_table(s2,out/'experiment2_endpoint_score_scales')
    # Numerical audits only; these never select a model.
    q=trace[trace.arm=='Q'];accepted=q[q.accepted]
    assert np.all(accepted.card_after_scorer<=accepted.card_before_scorer+1e-12)
    assert (paths[paths.arm=='Q'].rank_violation<=1e-12).all()
    assert (paths[paths.arm=='Q'].stability_violation<=1e-12).all()
    maxidentity=float(pd.read_csv(out/'experiment1_residual_decomposition.csv').identity_discrepancy.abs().max())
    checks=dict(accepted_Q_scorer_steps_cardinal_nonincreasing=True,all_Q_training_constraints_satisfied=True,
        max_residual_identity_discrepancy=maxidentity,max_integration_mean_discrepancy=float(integration.max_mean_discrepancy.max()),
        max_integration_variance_discrepancy=float(integration.max_variance_discrepancy.max()),
        F_A_exact_scorer_and_metric_equality=metadata['F_A_exact_scorer_and_metric_equality'])
    (out/'verification.json').write_text(json.dumps(checks,indent=2))
    dm=dec[(dec.arm=='D')&(dec.target=='conditional_mean')].iloc[0]
    am=dec[(dec.arm=='A')&(dec.target=='conditional_mean')].iloc[0]
    qsmall=final[(final.arm=='Q')&(final['where']=='unseen')&(final.strength==.15)].iloc[0]
    fsmall=final[(final.arm=='F')&(final['where']=='unseen')&(final.strength==.15)].iloc[0]
    usmall=final[(final.arm=='U')&(final['where']=='unseen')&(final.strength==.15)].iloc[0]
    conclusions=[
        ('D improves expected outcome changes despite worse realized-label MSE in this replication.' if t1.loc['D','mean_change_MSE']<t1.loc['A','mean_change_MSE'] else 'Removing outcome noise does not reveal an expected-change benefit for D over A in this replication.'),
        ('D beats frozen B on expected changes, so that difference requires scorer updating in this comparison.' if t1.loc['D','mean_change_MSE']<t1.loc['B','mean_change_MSE'] else 'D does not beat frozen B on expected changes; scorer updating earns no expected-change advantage here.'),
        ('Q improves expected changes beyond frozen calibration F at this endpoint.' if t2.loc['Q','mean_change_MSE']<t2.loc['F','mean_change_MSE'] else 'Q does not improve expected changes beyond frozen calibration F at this endpoint.'),
        ('Q improves unseen small latent-change RMSE versus F.' if qsmall.change_rmse<fsmall.change_rmse else 'Q does not improve unseen small latent-change RMSE versus F.'),
        ('Q improves unseen small-change detection versus F.' if qsmall.sensitivity>fsmall.sensitivity else 'Q does not improve unseen small-change detection versus F.')]
    lines=[
        '# Two bounded follow-ups to the paired-outcome diagnostic',
        'One replication only: quadratic latent truth, nonlinear monotone measurement, 32 fitting cities, the same paired observations, splits, auxiliary links and independent evaluation panels. No baseline refit, new seeds, searches or automatic extensions. Experiment 2 training ran before oracle calculations, and its fitting function receives no oracle targets.',
        '**Fixed-budget findings:** '+' '.join(conclusions)+' These are descriptive comparisons from one replication.',
        '## Experiment 1: did observed-label noise hide expected-outcome gains?',
        'All saved A/O/B/C/D checkpoints were evaluated at updates 0,25,50,100,200. A is fixed: its saved update-zero state is repeated for the five evaluations. Cell-specific heads remain unavailable on unseen cities; their latent metrics remain in the prior report, without invented zero-shot outcome predictions.',
        'The conditional mean integrates the implemented centered Gumbel noise: epsilon = 0.8028980597616118 * sqrt(6)/pi * (z - EulerGamma), with standard Gumbel density exp(-z-exp(-z)). We integrate tanh(f+epsilon), epsilon*tanh(f+epsilon), and tanh(f+epsilon)^2 deterministically using adaptive quadrature, using the exact zero mean and variance of epsilon. The mean is a+b[f+kappa E tanh(f+epsilon)], rather than Psi(f). Conditional variance is b^2[sigma^2+2 kappa E(epsilon tanh)+kappa^2 Var(tanh)].',
        f'Integration over z in [-12,45] was repeated at tolerances 1e-9 and 1e-11. Maximum endpoint mean discrepancy: **{checks["max_integration_mean_discrepancy"]:.3g}**; maximum variance discrepancy: **{checks["max_integration_variance_discrepancy"]:.3g}**. Right-tail omitted probability is below 2.87e-20; its variance contribution is negligible at this accuracy. All reported oracle values use the tighter precision.',
        'All-tract seen-city endpoints, moderate-change panel (raw MSE):',md(t1[['realized_level_MSE','realized_change_MSE','mean_level_MSE','mean_change_MSE','expected_noisy_change_risk']].reset_index()),
        f'D-A expected-change MSE difference: **{t1.loc["D","mean_change_MSE"]-t1.loc["A","mean_change_MSE"]:+.6g}**; D-B: **{t1.loc["D","mean_change_MSE"]-t1.loc["B","mean_change_MSE"]:+.6g}**; D-C: **{t1.loc["D","mean_change_MSE"]-t1.loc["C","mean_change_MSE"]:+.6g}**. Negative differences favor D. These comparisons concern fixed predictions and independent conditional-mean targets, not checkpoint selection.',
        'Noise does not explain away the expected-change deterioration. O and C improve expected level prediction while worsening expected changes. B also worsens expected changes with the scorer exactly frozen: changing cell/date calibration alone can weaken cancellation of endpoint errors. D adds further deterioration beyond B. These are different phenomena from the scorer-based latent-change deterioration.',
        'Stratified expected errors, fixed-denominator standardized errors, conditional noise floors and expected noisy risks are in experiment1_stratified_endpoint_table.md/.csv. The independent-date change noise floor is the sum of the two endpoint variances. Expected noisy risk equals conditional-mean prediction MSE plus the corresponding noise floor; realized-sample MSE need not equal this expectation.',
        '### Error decomposition',
        'For identical paired weights, e_t = prediction_t - target_t; level MSE is (E e0^2 + E e1^2)/2. The verified identity is change MSE = 2 level MSE - 2 E(e0 e1). Equivalently, change MSE = (mean(e1)-mean(e0))^2 + Var(e0)+Var(e1)-2 Cov(e0,e1). Realized-label residuals and conditional-mean prediction errors are decomposed separately in experiment1_endpoint_decomposition.md and the full trajectory CSV.',
        f'For D versus A on conditional means, change-bias-squared changes by **{dm.change_bias_squared-am.change_bias_squared:+.6g}** and the change-variance component by **{dm.change_variance-am.change_variance:+.6g}**. Marginal variances change by **{dm.residual_variance_0+dm.residual_variance_1-am.residual_variance_0-am.residual_variance_1:+.6g}**, while the covariance contribution -2 Cov changes by **{-2*(dm.residual_covariance-am.residual_covariance):+.6g}**. These are finite-panel descriptive decompositions, not a causal identification of noise or optimizer mechanisms.',
        'Raw stable-score RMS, fitting V and normalized RMS are in experiment1_endpoint_score_scales.md. See experiment1_oracle_trajectories.png for the expected-level and stable/small/moderate expected-change paths.',
        '## Experiment 2: cardinal fitting while protecting ranking and stability',
        f'Common calibration preparation kept A frozen and ran exactly 1,000 additional head Adam updates, selecting every 10 updates with observed validation L_W+L_DeltaW and including update zero. Initial validation cardinal loss: **{prep.iloc[0].validation:.8f}**; selected: **{metadata["head_preparation"]["validation"]:.8f}** at update **{metadata["head_preparation"]["best_step"]}**. Selection is not at the budget boundary. This is validation selection, not a claim of converged fitting calibration. All branches use the identical restored scorer/head pair.',
        f'Fixed references: R_A={refs["rank_A"]:.12g}, P_A={refs["stability_A"]:.12g}. Absolute bounds: rank <= **{refs["rank_bound"]:.12g}**, S/V <= **{refs["stability_bound"]:.12g}**. The tolerances were fixed before execution and never relaxed.',
        'F freezes A and updates heads on L_card. U jointly updates the scorer on raw RankNet +0.3 S/V+L_card and heads on L_card. Q minimizes L_card with the two scorer constraints. All branches have 200 head updates; U and Q each have 200 scorer attempts. Nominal learning rates remain scorer 1e-4 and head 1e-3, with no added head penalty. Updates 0,25,50,100,200 are saved; no branch endpoint is selected using test or oracle performance.',
        'Q uses a genuine feasible-direction method. It projects the nominal Adam scorer displacement onto the two linearized halfspaces using exact enumeration of their active sets. If the result is not a cardinal descent direction, it uses a projected negative-gradient proposal of the same nominal step norm. At most 12 halving/backtracking evaluations enforce the actual nonlinear constraints and an Armijo fitting-cardinal decrease, with the head held fixed. Rejection restores both scorer parameters and Adam state/counter; an accepted shortened step retains nominal-gradient Adam moments. Head Adam updates occur separately once per attempt. This is local constrained optimization, not a global optimum certificate.',
        f'Q accepted **{metadata["Q_accepted"]}/200** scorer attempts and rejected **{metadata["Q_rejected"]}**; total backtracking evaluations **{metadata["Q_backtracking_evaluations"]}**. Every recorded training constraint satisfied its bound within 1e-12, and accepted scorer updates reduced fitting cardinal loss. F scorer parameters and all scorer-based test metrics equal A exactly.',
        f'**Constrained-optimization limitation:** the last accepted Q scorer attempt was **{int(accepted.attempt.max())}**. All remaining attempts were rejected by the bounded procedure while the stability constraint was at its boundary. Thus Q mainly demonstrates that this implementation can enforce the constraints and avoid U\'s deterioration; it does not settle whether a better feasible scorer update could improve cardinal predictions. A stalled optimizer is not evidence that useful feasible directions do not exist.',
        '**Correction on interpreting Q:** linear projection can produce a direction tangent to the active stability boundary. For a constraint g(theta)<=0, g(theta+alpha d)=g(theta)+alpha grad(g).d+(alpha^2/2)d^T H_g d+o(alpha^2). At the boundary, grad(g).d=0 and positive directional curvature can make every sufficiently small straight positive step infeasible. Halving alone cannot cure this geometry. An inward margin, a feasibility correction or another constrained solver could behave differently. Q\'s 178 rejections therefore provide little evidence against useful constrained retraining; no alternative Q run was performed.',
        'Fitting/validation losses below are standardized with the unchanged denominators Var_train(W)=30.124436698937078 and Var_train(DeltaW)=16.201447520252533:',
        md(lp[['fit_level','fit_change','val_level','val_change','rank_fit','stability_ratio','rank_relative_change','stability_relative_change']].reset_index()),
        'Seen-city all-tract outcome endpoints (moderate panel, raw MSE):',md(t2[['realized_level_MSE','realized_change_MSE','mean_level_MSE','mean_change_MSE']].reset_index()),
        f'Expected-change Q-F: **{t2.loc["Q","mean_change_MSE"]-t2.loc["F","mean_change_MSE"]:+.6g}**; Q-U: **{t2.loc["Q","mean_change_MSE"]-t2.loc["U","mean_change_MSE"]:+.6g}**. Realized-change Q-F: **{t2.loc["Q","realized_change_MSE"]-t2.loc["F","realized_change_MSE"]:+.6g}**; Q-U: **{t2.loc["Q","realized_change_MSE"]-t2.loc["U","realized_change_MSE"]:+.6g}**.',
        'Seen and unseen latent-score endpoints:',md(final[['arm','where','strength','change_rmse','slope','correlation','stable_rms','threshold','sensitivity','false_positive_rate']]),
        f'For unseen small changes, Q-F RMSE is **{qsmall.change_rmse-fsmall.change_rmse:+.6g}**, stable RMS **{qsmall.stable_rms-fsmall.stable_rms:+.6g}**, and sensitivity **{100*(qsmall.sensitivity-fsmall.sensitivity):+.3f} percentage points**. Q-U differences are **{qsmall.change_rmse-usmall.change_rmse:+.6g}**, **{qsmall.stable_rms-usmall.stable_rms:+.6g}**, and **{100*(qsmall.sensitivity-usmall.sensitivity):+.3f} points** respectively. F has A\'s scorer, so Q-F is also Q-A for latent metrics. Training-link stability protection must be judged against these independent stable observations.',
        'The fixed-budget distinction is clear: U achieves the largest fitting-cardinal improvement, but its observed validation loss, expected outcome changes, latent changes and small-change detection worsen. Q largely preserves independent stability and detection relative to F/A, without a material gain beyond frozen calibration. The tiny moderate latent-RMSE improvement for Q on unseen cities is descriptive and does not support a general recovery claim.',
        '## Limits and reproduction',
        'Better fitting loss, expected outcome-change prediction, latent-change recovery and detection are distinct outcomes. These tables provide all four, without statistical significance claims from one replication. Noisy labels, finite head fit, training-to-test generalization and local constrained optimization remain possible limitations. A rejected/stalled Q step would not prove that no feasible improvement exists. No further experiments were launched.',
        'From ordinal_mc_suite in the existing PyTorch environment, choose a new output directory:',
        '```bash\npython verification/e6_change_followups.py --run e6_main_512 --source e6_observed_change_diagnostic --config configs/e6_change_followups.json --out e6_change_followups_new\npython verification/report_e6_change_followups.py --out e6_change_followups_new\n```',
        'Saved: configuration, source hashes, common-head preparation, all branch checkpoints, per-attempt optimizer traces, constraint checks, fitting and validation paths, oracle/residual/latent tables, contrasts and two trajectory figures. Source checkpoints and main-run manifest remained unchanged.'
    ]
    (out/'ANALYSIS.md').write_text('\n\n'.join(lines)+'\n')
    print(json.dumps(checks,indent=2));print(out/'ANALYSIS.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out.resolve())
