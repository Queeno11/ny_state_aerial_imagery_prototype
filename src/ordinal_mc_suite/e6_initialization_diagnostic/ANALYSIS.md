# One replication: ordinal scratch, joint scratch and extended warm start

**Observed-validation-selected joint scratch has higher small-change RMSE than ordinal scratch in this fixed setting.** All results below are one replication, one loss-weight setting and one common learning-rate pair. They do not establish that the selected weights are good or that a formulation generally succeeds or fails.

## Design and stopping

The original paired-outcome data were reconstructed exactly: quadratic truth, the small [32,16] ReLU scorer, nonlinear monotone measurement, 32 cities, two dates, 13,120 fitting labels from 6,560 paired tracts, 3,264 validation labels from 1,632 paired tracts. Independent date-specific centered Gumbel errors, exact temporal auxiliary links, no cross-city links, no scorer or head penalties, and all prior independent test/calibration panels were retained.

Fitting seed 1422665402; validation seed 1157853548; common scratch scorer seed 1765154216. Positive monotone spline heads receive only score and cell ID. Denominators remain Var_fit(W)=30.1244366989 and Var_fit(DeltaW)=16.2014475203.

Ordinal scratch and joint scratch start with exactly identical scorer parameters and heads. All losses are active from update one for joint scratch. Its head receives no preliminary level fitting. Ordinal heads are updated on the same level-plus-change labels using detached scores; they never affect the ordinal scorer. Scratch head normalization and three knots use fitting scores at initialization, and stay fixed. Extended warm starts from the original A scorer and its saved calibrated head, with fresh Adam states. Its head normalization/knots come from A, so this comparison concerns the full initialization strategy, rather than isolating only scorer weights.

The ordinal scorer objective is raw RankNet +0.3 S/V; joint and warm add standardized level MSE + standardized observed-change MSE, both weighted one. V is differentiated. All paths use scorer LR .001, head LR .001, full batches, and separate scorer/head clipping at norm20. This common scorer LR is the main E6 joint LR, ten times the short diagnostic LR; the head LR remains the conservative diagnostic .001. No LR or loss-weight alternatives were tried.

Every path has a predeclared 3,000-update ceiling, matching the suite baseline Adam horizon. Validation is checked every 10 updates. Ordinal selection uses validation RankNet alone; joint/warm use validation RankNet + standardized level + standardized change MSE (fixed 1:1:1). The existing patience of 20 checks and min_delta=1e-5 define the validation-stopped prefix. Initial checkpoints are eligible. Once patience is reached, selection freezes; training continues only to produce the predeclared fixed-budget endpoint and trajectories. Later improvements do not replace the stopped-prefix selected checkpoint. This separates a deployable validation-stopped result from a fixed-compute diagnostic without selecting on latent truth.

## Selected checkpoints and fixed-budget endpoints

The following latent and detection metrics are on unseen cities. Outcome MSE is on the seen-city moderate-change panel, all tracts; cell-specific heads are unavailable zero-shot on unseen cities. **The ordinal scratch heads have very large prediction errors and are not a useful calibration benchmark in this run; their errors are reported for transparency. This limitation does not affect its scorer-based latent or detection metrics.**

| path | checkpoint | updates | small_RMSE | moderate_RMSE | stable_RMS | small_sensitivity | moderate_sensitivity | expected_level_MSE | expected_change_MSE | realized_change_MSE |
|---|---|---|---|---|---|---|---|---|---|---|
| ordinal_scratch | selected | 1130 | 0.0582731 | 0.104573 | 0.042233 | 0.869223 | 0.989216 | 237.996 | 23.3632 | 37.894 |
| ordinal_scratch | fixed_budget | 3000 | 0.0621237 | 0.0937262 | 0.0546182 | 0.766962 | 0.989216 | 9.11965 | 1.22819 | 15.8383 |
| joint_scratch | selected | 670 | 0.194459 | 0.221067 | 0.178753 | 0.139626 | 0.569608 | 0.583537 | 0.567828 | 15.1291 |
| joint_scratch | fixed_budget | 3000 | 0.26901 | 0.288248 | 0.270108 | 0.0875123 | 0.328431 | 1.04095 | 1.36244 | 15.8924 |
| extended_warm | selected | 0 | 0.0535695 | 0.0881767 | 0.0438895 | 0.880039 | 0.994118 | 0.197219 | 0.132194 | 14.7592 |
| extended_warm | fixed_budget | 3000 | 0.262205 | 0.278241 | 0.264108 | 0.102262 | 0.390196 | 1.02399 | 1.48944 | 16.0982 |


Full seen/unseen slopes, correlations, ranking accuracy, false-positive rates and thresholds are in all_endpoint_latent_metrics.md/.csv. Stratified conditional-mean and realized outcome errors are in outcome_trajectories.csv. Oracle conditional means integrate the actual error law and are evaluated only after all training, never used for selection.

**Outcome-head qualification:** the ordinal path's heads track its evolving scores concurrently, using the same head budget as joint scratch. They are not refitted after freezing the selected scorer. Initial scratch-score normalization and knots can become poorly matched to a scorer whose raw scale changes substantially. Outcome-error differences therefore include head tracking and scaling, and cannot alone establish a better latent scorer. The warm-start selected checkpoint supplies the existing, calibrated A reference.

Fitting and observed validation at each reported checkpoint:

| path | checkpoint | updates | fit_rank | fit_stability_ratio | fit_level | fit_change | val_rank | val_level | val_change | validation_rule |
|---|---|---|---|---|---|---|---|---|---|---|
| ordinal_scratch | selected | 1130 | 0.424612 | 0.00181516 | 7.97271 | 2.28861 | 0.41997 | 7.62736 | 2.42641 | 0.41997 |
| ordinal_scratch | fixed_budget | 3000 | 0.419627 | 0.00280039 | 0.525304 | 0.955892 | 0.419278 | 0.51947 | 0.947944 | 0.419278 |
| joint_scratch | selected | 670 | 0.676162 | 0.0312426 | 0.249758 | 0.861958 | 0.676403 | 0.259405 | 0.87868 | 1.81449 |
| joint_scratch | fixed_budget | 3000 | 0.672694 | 0.0634971 | 0.238951 | 0.770548 | 0.673202 | 0.275897 | 0.941675 | 1.89077 |
| extended_warm | selected | 0 | 0.420884 | 0.00191768 | 0.241024 | 0.898826 | 0.420111 | 0.246829 | 0.860353 | 1.52729 |
| extended_warm | fixed_budget | 3000 | 0.416145 | 0.0565868 | 0.232116 | 0.767288 | 0.433866 | 0.275579 | 0.943349 | 1.65279 |


## Comparisons and interpretation

At selected checkpoints, joint scratch minus ordinal scratch is +0.136186 for small latent RMSE, +0.116494 for moderate RMSE, +0.13652 for stable RMS and -72.960 percentage points for small-change sensitivity. Expected outcome-change MSE differs by -22.7953, dominated by ordinal head failure; that difference is not evidence of improved latent recovery.

Joint scratch minus selected extended warm is +0.14089 for small latent RMSE, +0.13289 for moderate RMSE, +0.134864 for stable RMS, and +0.435634 for expected outcome-change MSE. This is the observed effect of the complete initialization strategies at this setting, including head initialization and validation rules.

For unseen small changes, selected ordinal versus joint scratch slopes are 0.92304 versus 0.87141, while correlations are 0.94976 versus 0.64302. Joint scratch therefore adds considerable individual prediction noise/nuisance movement as well as some attenuation. Its own detection threshold rises to 0.35725 versus 0.08906; fresh FPR remains near nominal, 5.27% versus 4.59%. These are descriptive, not causal mechanisms.

Raw score/head scaling differs sharply across paths, but a positive affine change of score leaves normalized temporal changes invariant. Raw scale differences cannot by themselves explain the worse latent-change metrics. The read-only audit also verifies nonzero level and change gradients into the scratch scorer at initialization; both cardinal terms were active from the first update.

The fixed-budget rows must be read separately from validation selection. Losses can keep improving on fitting data after validation stopping, and latent/test performance can improve or deteriorate without being available to the learner. Selection does not exploit the plotted latent paths. A cap-ending or still-improving path would remain a budget-limited observation, rather than a convergence claim.

Observed-validation stopping diagnostics below show whether the declared patience rule missed later validation improvement. Full-budget minima are descriptive scalar summaries of the already logged validation path, not replacement model selections and not latent/test choices:

| path | selected_step | selected_validation | patience_stop_step | full_budget_minimum_validation | full_budget_minimum_step | later_validation_improvement |
|---|---|---|---|---|---|---|
| ordinal_scratch | 1130 | 0.41997 | 1330 | 0.4192 | 2690 | True |
| joint_scratch | 670 | 1.81449 | 870 | 1.81449 | 670 | False |
| extended_warm | 0 | 1.52729 | 200 | 1.52729 | 0 | False |


Read-only raw score/head scale audit (initial, selected and fixed endpoints):

| path | checkpoint | score_mean | score_SD | within_cell_V | initial_head_score_scale | median_head_segment_slope | min_head_segment_slope | max_head_segment_slope | median_effective_slope_per_raw_score |
|---|---|---|---|---|---|---|---|---|---|
| ordinal_scratch | initial | 0.0589687 | 0.0549975 | 0.00254958 | 0.0549975 | 1 | 1 | 1 | 99.7967 |
| ordinal_scratch | selected | 1.16989 | 1.58938 | 2.51796 | 0.0549975 | 0.19765 | 0.0965066 | 0.283655 | 19.7248 |
| ordinal_scratch | fixed_budget | 0.902196 | 1.64658 | 2.66612 | 0.0549975 | 0.0718204 | 0.0235508 | 0.17101 | 7.16745 |
| joint_scratch | initial | 0.0589687 | 0.0549975 | 0.00254958 | 0.0549975 | 1 | 1 | 1 | 99.7967 |
| joint_scratch | selected | 0.0634829 | 0.0417125 | 0.00146029 | 0.0549975 | 0.956695 | 0.782816 | 1.19714 | 95.475 |
| joint_scratch | fixed_budget | 0.065443 | 0.0491031 | 0.00208468 | 0.0549975 | 0.814971 | 0.2704 | 1.51525 | 81.3315 |
| extended_warm | initial | 1.89125 | 1.67279 | 2.6315 | 1.67279 | 0.661437 | 0.610518 | 1.05549 | 2.17024 |
| extended_warm | selected | 1.89125 | 1.67279 | 2.6315 | 1.67279 | 0.661437 | 0.610518 | 1.05549 | 2.17024 |
| extended_warm | fixed_budget | 1.97473 | 1.75896 | 2.83517 | 1.67279 | 0.633296 | 0.0749688 | 1.81521 | 2.07791 |


## Scorer-update accounting

| path | selected_step | patience_stop_step | executed_scorer_updates | inherited_scorer_updates_executed | total_scorer_updates_executed | selected_scorer_ancestry | fixed_endpoint_scorer_ancestry | selected_at_budget_boundary |
|---|---|---|---|---|---|---|---|---|
| ordinal_scratch | 1130 | 1330 | 3000 | 0 | 3000 | 1130 | 3000 | False |
| joint_scratch | 670 | 870 | 3000 | 0 | 3000 | 670 | 3000 | False |
| extended_warm | 0 | 200 | 3000 | 520 | 3520 | 320 | 3320 | False |


Warm start inherits 520 previously executed baseline updates, although the restored A scorer is from update 320; the remaining 200 were spent finding the validation stop and are not in its parameter ancestry. Thus its total computation is 3,520 scorer updates, while its fixed endpoint has 3,320 updates in its ancestry. It also inherits 600 head updates. Scratch paths each execute 3,000 scorer and 3,000 head updates. Selected-step ancestry and the patience detection time are reported separately; continuing for fixed-budget diagnostics is included in execution counts.

## Q correction and scope

The earlier Q stall is not evidence against useful constrained retraining. At an active constraint g(theta)=0, a projected direction can have grad(g).d=0. If d^T H_g d>0, the second-order term makes sufficiently small positive straight steps infeasible, so backtracking alone can fail. An inward margin, feasibility correction or stronger constrained solver could behave differently. The prior report was corrected; Q was not rerun.

The evidence from previous fine-tuning paths concerns those initialization and optimization choices. This scratch/warm comparison adds one missing design, but cannot establish an optimal loss formulation, loss weights, population effect or convergence. No extra seeds, searches or follow-up runs were launched.

## Reproduce

From ordinal_mc_suite in the existing PyTorch environment, choose a new output directory:

```bash
python verification/e6_initialization_paths.py --source e6_observed_change_diagnostic --config configs/e6_initialization_one_shot.json --out e6_initialization_diagnostic_new
python verification/audit_e6_initialization_paths.py --out e6_initialization_diagnostic_new
python verification/report_e6_initialization_paths.py --out e6_initialization_diagnostic_new
```

Saved configuration, exact initialization hashes, source checkpoint hashes, all selected and trajectory checkpoints, optimizer end states, per-update gradient/clipping traces, validation histories, outcome and latent tables, path contrasts and trajectory figures. All previous checkpoints are unchanged.
