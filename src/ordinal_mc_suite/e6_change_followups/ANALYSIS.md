# Two bounded follow-ups to the paired-outcome diagnostic

One replication only: quadratic latent truth, nonlinear monotone measurement, 32 fitting cities, the same paired observations, splits, auxiliary links and independent evaluation panels. No baseline refit, new seeds, searches or automatic extensions. Experiment 2 training ran before oracle calculations, and its fitting function receives no oracle targets.

**Fixed-budget findings:** Removing outcome noise does not reveal an expected-change benefit for D over A in this replication. D does not beat frozen B on expected changes; scorer updating earns no expected-change advantage here. Q does not improve expected changes beyond frozen calibration F at this endpoint. Q does not improve unseen small latent-change RMSE versus F. Q does not improve unseen small-change detection versus F. These are descriptive comparisons from one replication.

## Experiment 1: did observed-label noise hide expected-outcome gains?

All saved A/O/B/C/D checkpoints were evaluated at updates 0,25,50,100,200. A is fixed: its saved update-zero state is repeated for the five evaluations. Cell-specific heads remain unavailable on unseen cities; their latent metrics remain in the prior report, without invented zero-shot outcome predictions.

The conditional mean integrates the implemented centered Gumbel noise: epsilon = 0.8028980597616118 * sqrt(6)/pi * (z - EulerGamma), with standard Gumbel density exp(-z-exp(-z)). We integrate tanh(f+epsilon), epsilon*tanh(f+epsilon), and tanh(f+epsilon)^2 deterministically using adaptive quadrature, using the exact zero mean and variance of epsilon. The mean is a+b[f+kappa E tanh(f+epsilon)], rather than Psi(f). Conditional variance is b^2[sigma^2+2 kappa E(epsilon tanh)+kappa^2 Var(tanh)].

Integration over z in [-12,45] was repeated at tolerances 1e-9 and 1e-11. Maximum endpoint mean discrepancy: **1.78e-15**; maximum variance discrepancy: **5.33e-15**. Right-tail omitted probability is below 2.87e-20; its variance contribution is negligible at this accuracy. All reported oracle values use the tighter precision.

All-tract seen-city endpoints, moderate-change panel (raw MSE):

| arm | realized_level_MSE | realized_change_MSE | mean_level_MSE | mean_change_MSE | expected_noisy_change_risk |
|---|---|---|---|---|---|
| A | 7.53792 | 14.7592 | 0.197219 | 0.132194 | 14.832 |
| O | 7.50201 | 14.8058 | 0.173384 | 0.183051 | 14.8829 |
| B | 7.52347 | 14.8263 | 0.189639 | 0.20512 | 14.905 |
| C | 7.49863 | 14.8148 | 0.174162 | 0.195265 | 14.8951 |
| D | 7.5237 | 14.8547 | 0.204752 | 0.239074 | 14.9389 |


D-A expected-change MSE difference: **+0.10688**; D-B: **+0.0339539**; D-C: **+0.0438092**. Negative differences favor D. These comparisons concern fixed predictions and independent conditional-mean targets, not checkpoint selection.

Noise does not explain away the expected-change deterioration. O and C improve expected level prediction while worsening expected changes. B also worsens expected changes with the scorer exactly frozen: changing cell/date calibration alone can weaken cancellation of endpoint errors. D adds further deterioration beyond B. These are different phenomena from the scorer-based latent-change deterioration.

Stratified expected errors, fixed-denominator standardized errors, conditional noise floors and expected noisy risks are in experiment1_stratified_endpoint_table.md/.csv. The independent-date change noise floor is the sum of the two endpoint variances. Expected noisy risk equals conditional-mean prediction MSE plus the corresponding noise floor; realized-sample MSE need not equal this expectation.

### Error decomposition

For identical paired weights, e_t = prediction_t - target_t; level MSE is (E e0^2 + E e1^2)/2. The verified identity is change MSE = 2 level MSE - 2 E(e0 e1). Equivalently, change MSE = (mean(e1)-mean(e0))^2 + Var(e0)+Var(e1)-2 Cov(e0,e1). Realized-label residuals and conditional-mean prediction errors are decomposed separately in experiment1_endpoint_decomposition.md and the full trajectory CSV.

For D versus A on conditional means, change-bias-squared changes by **+9.80751e-05** and the change-variance component by **+0.106782**. Marginal variances change by **+0.0158147**, while the covariance contribution -2 Cov changes by **+0.0909675**. These are finite-panel descriptive decompositions, not a causal identification of noise or optimizer mechanisms.

Raw stable-score RMS, fitting V and normalized RMS are in experiment1_endpoint_score_scales.md. See experiment1_oracle_trajectories.png for the expected-level and stable/small/moderate expected-change paths.

## Experiment 2: cardinal fitting while protecting ranking and stability

Common calibration preparation kept A frozen and ran exactly 1,000 additional head Adam updates, selecting every 10 updates with observed validation L_W+L_DeltaW and including update zero. Initial validation cardinal loss: **1.10718252**; selected: **1.10717854** at update **10**. Selection is not at the budget boundary. This is validation selection, not a claim of converged fitting calibration. All branches use the identical restored scorer/head pair.

Fixed references: R_A=0.420883854316, P_A=0.00191767710772. Absolute bounds: rank <= **0.420983854316**, S/V <= **0.0019368538788**. The tolerances were fixed before execution and never relaxed.

F freezes A and updates heads on L_card. U jointly updates the scorer on raw RankNet +0.3 S/V+L_card and heads on L_card. Q minimizes L_card with the two scorer constraints. All branches have 200 head updates; U and Q each have 200 scorer attempts. Nominal learning rates remain scorer 1e-4 and head 1e-3, with no added head penalty. Updates 0,25,50,100,200 are saved; no branch endpoint is selected using test or oracle performance.

Q uses a genuine feasible-direction method. It projects the nominal Adam scorer displacement onto the two linearized halfspaces using exact enumeration of their active sets. If the result is not a cardinal descent direction, it uses a projected negative-gradient proposal of the same nominal step norm. At most 12 halving/backtracking evaluations enforce the actual nonlinear constraints and an Armijo fitting-cardinal decrease, with the head held fixed. Rejection restores both scorer parameters and Adam state/counter; an accepted shortened step retains nominal-gradient Adam moments. Head Adam updates occur separately once per attempt. This is local constrained optimization, not a global optimum certificate.

Q accepted **22/200** scorer attempts and rejected **178**; total backtracking evaluations **2281**. Every recorded training constraint satisfied its bound within 1e-12, and accepted scorer updates reduced fitting cardinal loss. F scorer parameters and all scorer-based test metrics equal A exactly.

**Constrained-optimization limitation:** the last accepted Q scorer attempt was **22**. All remaining attempts were rejected by the bounded procedure while the stability constraint was at its boundary. Thus Q mainly demonstrates that this implementation can enforce the constraints and avoid U's deterioration; it does not settle whether a better feasible scorer update could improve cardinal predictions. A stalled optimizer is not evidence that useful feasible directions do not exist.

Fitting/validation losses below are standardized with the unchanged denominators Var_train(W)=30.124436698937078 and Var_train(DeltaW)=16.201447520252533:

| arm | fit_level | fit_change | val_level | val_change | rank_fit | stability_ratio | rank_relative_change | stability_relative_change |
|---|---|---|---|---|---|---|---|---|
| F | 0.23902 | 0.890372 | 0.247134 | 0.866143 | 0.420884 | 0.00191768 | 0 | -3.10862e-14 |
| U | 0.238529 | 0.883986 | 0.247794 | 0.868868 | 0.420667 | 0.00357134 | -0.00051553 | 0.862326 |
| Q | 0.239005 | 0.890199 | 0.247238 | 0.866425 | 0.42088 | 0.00193685 | -8.34588e-06 | 0.01 |


Seen-city all-tract outcome endpoints (moderate panel, raw MSE):

| arm | realized_level_MSE | realized_change_MSE | mean_level_MSE | mean_change_MSE |
|---|---|---|---|---|
| F | 7.52506 | 14.8312 | 0.191793 | 0.209537 |
| U | 7.52611 | 14.8612 | 0.208008 | 0.246095 |
| Q | 7.52515 | 14.8309 | 0.193134 | 0.209973 |


Expected-change Q-F: **+0.000435085**; Q-U: **-0.0361228**. Realized-change Q-F: **-0.00026157**; Q-U: **-0.0302797**.

Seen and unseen latent-score endpoints:

| arm | where | strength | change_rmse | slope | correlation | stable_rms | threshold | sensitivity | false_positive_rate |
|---|---|---|---|---|---|---|---|---|---|
| F | seen | 0.15 | 0.053531 | 0.946796 | 0.95906 | 0.0429492 | 0.0901153 | 0.880078 | 0.0463867 |
| F | unseen | 0.15 | 0.0535695 | 0.953844 | 0.958214 | 0.0438895 | 0.0901153 | 0.880039 | 0.0546875 |
| F | seen | 0.4 | 0.0857602 | 0.952103 | 0.98425 | 0.0429492 | 0.0901153 | 0.992669 | 0.0463867 |
| F | unseen | 0.4 | 0.0881767 | 0.961938 | 0.982731 | 0.0438895 | 0.0901153 | 0.994118 | 0.0546875 |
| U | seen | 0.15 | 0.0673236 | 0.95241 | 0.9371 | 0.0590509 | 0.117872 | 0.743514 | 0.0537109 |
| U | unseen | 0.15 | 0.0648561 | 0.952071 | 0.93993 | 0.0590831 | 0.117872 | 0.719764 | 0.0566406 |
| U | seen | 0.4 | 0.0942307 | 0.953532 | 0.980807 | 0.0590509 | 0.117872 | 0.981916 | 0.0537109 |
| U | unseen | 0.4 | 0.0941626 | 0.960373 | 0.980279 | 0.0590831 | 0.117872 | 0.981373 | 0.0566406 |
| Q | seen | 0.15 | 0.0537034 | 0.948014 | 0.958817 | 0.0432286 | 0.0901116 | 0.879589 | 0.0454102 |
| Q | unseen | 0.15 | 0.0536409 | 0.954837 | 0.958128 | 0.0439813 | 0.0901116 | 0.878073 | 0.0566406 |
| Q | seen | 0.4 | 0.0857963 | 0.952962 | 0.984228 | 0.0432286 | 0.0901116 | 0.992669 | 0.0454102 |
| Q | unseen | 0.4 | 0.0881674 | 0.962651 | 0.982734 | 0.0439813 | 0.0901116 | 0.995098 | 0.0566406 |


For unseen small changes, Q-F RMSE is **+7.14024e-05**, stable RMS **+9.18257e-05**, and sensitivity **-0.197 percentage points**. Q-U differences are **-0.0112151**, **-0.0151017**, and **+15.831 points** respectively. F has A's scorer, so Q-F is also Q-A for latent metrics. Training-link stability protection must be judged against these independent stable observations.

The fixed-budget distinction is clear: U achieves the largest fitting-cardinal improvement, but its observed validation loss, expected outcome changes, latent changes and small-change detection worsen. Q largely preserves independent stability and detection relative to F/A, without a material gain beyond frozen calibration. The tiny moderate latent-RMSE improvement for Q on unseen cities is descriptive and does not support a general recovery claim.

## Limits and reproduction

**Correction on interpreting Q:** projection onto linearized constraints can produce a direction tangent to the active stability boundary. For a constraint g(theta)<=0, at the boundary with grad(g).d=0, positive directional curvature can make every sufficiently small straight positive step infeasible. Backtracking alone cannot cure this geometry. An inward margin, feasibility correction or another constrained solver could behave differently. Q's 178 rejections therefore provide little evidence against useful constrained retraining. No alternative Q run was performed. The earlier fine-tuning results do not settle the performance of joint training from scratch.

Better fitting loss, expected outcome-change prediction, latent-change recovery and detection are distinct outcomes. These tables provide all four, without statistical significance claims from one replication. Noisy labels, finite head fit, training-to-test generalization and local constrained optimization remain possible limitations. A rejected/stalled Q step would not prove that no feasible improvement exists. No further experiments were launched.

From ordinal_mc_suite in the existing PyTorch environment, choose a new output directory:

```bash
python verification/e6_change_followups.py --run e6_main_512 --source e6_observed_change_diagnostic --config configs/e6_change_followups.json --out e6_change_followups_new
python verification/report_e6_change_followups.py --out e6_change_followups_new
```

Saved: configuration, source hashes, common-head preparation, all branch checkpoints, per-attempt optimizer traces, constraint checks, fitting and validation paths, oracle/residual/latent tables, contrasts and two trajectory figures. Source checkpoints and main-run manifest remained unchanged.
