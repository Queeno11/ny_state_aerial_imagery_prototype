# Handoff for the ordinal Monte Carlo suite

Prepared November 9, 2026. The workspace is `C:\Working Papers\NY State Aerial Imagery Prototype\ny_state_aerial_imagery_prototype\src`. Paths in this document are relative to `src/ordinal_mc_suite` unless specified otherwise.

## Current state and the research question

The user is studying whether a common ordinal scorer can recover a tract's latent change across dates when outcomes are measured on different scales across cities and dates. They care about three distinct results: the direction of change, the magnitude of individual changes, and whether a detection rule flags a tract at a controlled false-positive rate. Their empirical application has trouble identifying changed tracts. The simulations test whether that difficulty appears under known truth and which part of the procedure contributes to it.

The latest task is complete. We ran one replication comparing ordinal training from scratch, joint training from scratch with level and change losses, and extended joint training from an existing ordinal checkpoint. At the chosen setting, the jointly trained scorers had substantially worse latent-change and detection metrics. Observed validation retained the original ordinal checkpoint for the warm start. The run also exposed a calibration problem in the detached heads accompanying ordinal scratch. Its outcome errors are a poor calibration benchmark, while its latent and detection metrics remain usable. [Latest report](e6_initialization_diagnostic/ANALYSIS.md)

The user stopped the large E6 Monte Carlo. A read-only count on November 9 found **62 of 120 jobs** in `e6_main_512/results.sqlite`, the same count as the interim snapshot. The separate `e6_main/results.sqlite` contains zero completed jobs. Keep the stopped run as the current state until the user requests resumption. Later work used explicitly bounded diagnostics, usually a single replication. Their results describe their specific loss weights, starting states and training budgets. [Interim report](e6_main_512_interim/ANALYSIS.md)

## Model, identification and evaluation

The DGP is `R = f(H) + ε` and `W = Ψ_ct(R)`. The scorer receives seven inputs formed by mixing three structural features, two acquisition nuisance features and two city nuisance features with a fixed orthogonal matrix. The structural rotation and mixing use function seed 983. With `u = H @ rotation`, the unnormalized truths are `u0 + 0.6*u1 - 0.4*u2` and `u0 + 0.7*u1*u2 + 0.4*(u2**2 - 1)`. The suite centers and scales the structural function using independent population calibration so its within-cell pair dispersion is one. The usual flexible scorer is a ReLU network with widths `[32,16]`, bounded effective weights and fitting-only feature standardization.

The baseline objective is

\[
L_{\rm ord}=\sum_{ij}q_{ij}\log(1+\exp[-y_{ij}(s_j-s_i)])
+\lambda_T S_T/V+\lambda_C S_C/V+\alpha J,
\qquad V=\tfrac12\sum_{ij}q_{ij}(s_j-s_i)^2.
\]

Here `y_ij = sign(W_j - W_i)` and the ordered pairs are within city/date. Each pair has weight `1/[N_fit*(n_cell-1)]`. RankNet uses raw score gaps and the stability ratio differentiates through `V`. The main E6 design and later diagnostics use `λ_T=0.3`, `λ_C=0`, `α=0`. Exact temporal links preserve `H` and city and change acquisition/date views. They supply external stability information without another labeled outcome. A within-cell strictly increasing measurement map preserves the rank labels. The raw logistic objective depends on the scale of the score gaps.

The latent estimator is `Δ̂ = (s(X1) - s(X0))/sqrt(V_fit(s))`, compared to `Δ = f(H1) - f(H0)`. A global shift or positive affine rescaling of the score cancels in this estimator. A city offset also cancels for the same tract within a city. A large change in normalized stable movement therefore reflects a change in the score's shape or nuisance sensitivity. Differences in raw scale can still affect the optimization because RankNet uses raw gaps. Finite-sample `V_fit(f)` can differ from one even though the population convention is one. E6 saves a normalization-only error diagnostic.

For evaluation, changing-tract metrics use `abs(Δ) > 0.05` and stable RMS uses the separately assigned pure stable tracts. The nominal E5 change step controls the perturbation along the structural gradient, with random sign and clipping to support. The resulting change sizes vary within nominal steps. Use the independent seen and unseen test panels. Each scorer's detection cutoff is its own empirical 95th percentile of `abs(Δ̂)` on the separate stable calibration panel, using `method='higher'`. Fresh stable FPR, cutoff, sensitivity, stable RMS, sign, RMSE, slope and correlation answer different questions. A high cutoff can hide accurately directed small changes. A near-unit slope can coexist with noisy individual changes.

On the outcome scale, report observed `W1-W0` and the oracle conditional-mean change separately. Under nonlinear measurement, compute `m_ct(H) = Eε[Ψ_ct(f(H)+ε)]` by integrating the centered Gumbel error. Evaluating `Ψ_ct(f(H))` generally gives a different target. A stable latent tract can have a change in expected `W` because its cell coefficients change over time. Fitted cell-specific heads predict on seen cities. Their outcome columns are unavailable for unseen cells without an explicitly separate calibration design. [Design and metric definitions](E6_README.md)

## E1–E5 and the appended ML benchmark

- **E1** compares ordinal scoring with pooled level OLS and log outcome OLS with city/date fixed effects. A multiplicative exponential map admits the log FE model. An additive baseline outside the exponential or changing nonlinear curvature breaks that representation. Ordinal comparisons survive increasing within-cell maps. Correctly specified FE is competitive, and even misspecified FE ranked well in the single additive-exponential check. The supported advantage is flexibility beyond the tested standard FE representations.
- **E2** varies noise families and reliability. The suite checks common logit, heterogeneous pair calibration and an ordinal surrogate. Most broad noise cases were implementation checks. In the separate city-reliability study, allowing city-specific calibration helped when reliability truly varied and could hurt when estimated unnecessarily.
- **E3** tests whether exact temporal views help the scorer ignore acquisition nuisance, with linear and flexible scorers and varying city counts. They generally help. Single-run temporal-plus-cross-city results were promising, but that arm also had more links and a larger combined stability coefficient.
- **E4** holds the labels and panels fixed and changes exact links to drifted or false matches. Mild drift preserves some benefit, while false links can penalize genuine structural differences. Validation tuning of link strength does not consistently repair bad matches.
- **E5** introduces matched before/after panels and measures direction, individual magnitude and detection. Original fixed-budget NN runs had weak change recovery and optimization audit failures. Follow-up diagnostics used a tract holdout targeting 20%, observed validation rank loss and restored best checkpoints. In one additive-exponential measurement run, moderate-change detection for no links, temporal links and both links was 25.6%, 47.5% and 91.8%. The corresponding quadratic-truth run gave 36.5%, 77.5% and 90.4%. The additive exponential and affine measurement fits matched exactly for the ordinal estimator with the same underlying data. These are single-run stopping diagnostics. The combined-link comparison changes information and link budget together. [E1–E5 synthesis](EXPERIMENT_SYNTHESIS.md) · [Nonlinear stopping comparison](nonlinear_validation_comparison/REPORT.md)

The user asked for ML as an addition to completed experiments. The append scripts replayed the saved samples and added privileged correctly specified likelihood fits to `_with_ml.csv` exports. Original estimator rows and ledgers were preserved. The ML fits know the true basis, inverse mixing, measurement inverse and error law, and estimate structural coefficients from fitting labels. Strong ML performance gives a reference with more information. It does not isolate an optimization gap. Saved charts are in `deck_figures_with_ml/`. The append audit reports 1,301 added suite ML rows, 23 rho ML rows and no added ML optimizer failures. [ML append audit](verification/ML_APPEND_VERIFICATION.md) · [E5 implementation checks](verification/ML_CHANGE_VERIFICATION.md)

## E6 main experiment and partial evidence

The primary E6 test asks whether observed level supervision improves the shared scorer's latent magnitudes. It uses affine truth `W=a+bR` and nonlinear truth `W=a+b*(R+κ*tanh(R))`, matched on structural draws and errors. With standard-normal city cues `Z`, `a=3Z[c,0]+t`, `b=1.5+exp(0.2Z[c,1])`, and `κ=0.6+0.2*tanh(Z[c,0]+0.5t)`. The noise is centered Gumbel scaled to a calibrated Bayes risk of 0.2. For the quadratic case the effective error SD is about 0.802898. Date-specific outcome errors are independent. Exact coefficient ranges for each run are in its manifest.

The original E6 labels are single-date observations. At 512 labels per city there are 256 per date, 205 fitting and 51 validation per cell. Each fitting labeled tract has two exact unlabeled temporal views. Every fitting label gets external links. That coverage is separate from the stable fraction in the evaluation panels. The primary E6 design has no labeled change differences.

| Arm in primary E6 | Training and comparison |
|---|---|
| A | Ordinal baseline with observed validation rank stopping and best-checkpoint restoration |
| B | Frozen A scorer with a fitted affine or monotone spline head |
| C | Joint scorer/head training initialized at the matching B candidate |
| D | Ordinal continuation with C's executed scorer-update budget, plus post-fit calibration for outcome comparisons |
| `D_common` | Alternative observable checkpoint comparison from the same ordinal continuation trajectory |

The positive-slope affine and four-segment monotone spline heads receive only scalar score and cell ID. Knots at fitting-score quantiles 0.2, 0.5 and 0.8 and the fitting normalization stay fixed through joint optimization. Primary E6 uses `0.01 J_G` with slope curvature and partial pooling, and `λ_W ∈ {0.1,1,10}` multiplying `MSE_fit(W)/Var_fit(W)`. B and C use a fixed observable validation rule of rank logistic plus standardized level MSE, irrespective of the training weight. A and B have identical scorer metrics by construction and automated check. The same-architecture `oracle_noisy_R` in primary E6 uses the true inverse measurement target `R` with a direct squared loss **replacing** RankNet. It is distinct from the later inverse experiments that retain RankNet. [E6 method and arms](E6_README.md)

Both eight-city smokes finished. The first used 32 labels per city and had quadratic unseen moderate-change RMSE 0.287, stable RMS 0.200 and detection 45.3%, even with 96.1% sign accuracy. All selected C fits retained update zero. The larger 512-label smoke and partial main results showed much better ordinal recovery. The first renderer had a `DataFrame.where` versus `['where']` column bug, producing empty charts/tables despite completed fits. The corrected read-only renderer rebuilds charts from SQLite. [Corrected first smoke](e6_smoke_review/SMOKE_REVIEW.md) · [512-label smoke](e6_smoke_512/report.md)

The interim snapshot contains 62 completed jobs. The balanced analysis uses replications 1–14 across both city counts and both structural cases, 56 jobs. In quadratic truth with unseen cities and nominal step 0.4, A's RMSE falls from 0.0904 at 32 fitting cities to 0.0745 at 64. Sign accuracy exceeds 99.9%, and stable RMS falls from 0.0466 to 0.0334. Small-change detection improves from 86.0% to 96.2%. At 64 cities, true moderate direction means of about −0.450 and +0.465 are predicted as −0.440 and +0.447. Average magnitudes can therefore be quite accurate with more data, while individual errors remain.

The partial replicated comparisons show no clear joint benefit over B or D. For 64-city quadratic truth with nonlinear measurement and an affine fitted head, unseen moderate C−B RMSE is +0.00232 with pointwise 95% interval [+0.00110, +0.00353]. C−D is +0.00261 [+0.00143, +0.00380]. Stable movement and cutoffs generally rise, and small-change sensitivity falls. Frozen calibration improves conditional-mean outcome changes relative to a zero-change prediction. Joint training provides no clear incremental outcome-change benefit. These are pointwise paired intervals over 14 replications, without multiplicity or sequential-monitoring adjustment. [Interim estimates and charts](e6_main_512_interim/ANALYSIS.md)

The empirical `λ_W` selections among 224 balanced C fits were 36 at 0.1, 115 at 1, and 73 at 10. Respectively 2, 64 and 53 retained their starting checkpoints. A substantial fraction of C fits moved the scorer. This rules out the simple description that every fit chose a negligible weight or stopped at its initialization. See `e6_main_512_interim/balanced_report/selection.csv`.

## Bounded diagnostics before paired outcome training

These diagnostics use the saved primary E6 replication 1, 32-city quadratic/ReLU fit. Its single-date training sample differs from the paired-label sample used later.

1. **Gradient scale and step size.** At the common nonlinear/affine start, RankNet and weighted stability gradients have norms about 0.00731 and 0.00709, with a net ordinal gradient of 0.00166. At `λ_W=1`, the level gradient norm is 0.01466. At 10 it is 0.14658. The fitting-sample level/stability cosine was about −0.54. The original Adam first step overshot locally and slowing both rates improved that step. Four 100-update paths crossed the original and 0.1× scorer/head rates, at fixed `λ_W=10`. At a comparable 0.25% reduction in fitting W-MSE, every path still worsened latent RMSE and stable RMS versus frozen B. The best of the four matched points moved them from 0.09196 and 0.04404 to 0.09463 and 0.05418. [Gradient audit](e6_gradient_diagnostic/ANALYSIS.md) · [Matched-progress trials](e6_targeted_diagnostic/ANALYSIS.md)
2. **Fresh gradient check.** On 16,384 fresh observations, both realized-W and oracle conditional-mean level gradients had positive cosines with fresh stability gradients in all four map/head combinations, about +0.08 to +0.19. The two level-gradient directions were closely aligned. The negative fitting-sample angle is a local result. These checks do not identify a population conflict or isolate outcome noise as its cause. [Fresh-panel analysis](e6_targeted_diagnostic/ANALYSIS.md)
3. **Raw oracle inverse with RankNet retained.** The target is noisy `R=Ψ⁻¹(W)`. Initially the loss forced raw `s` into R units. The validation-selected arm cut normalized R-MSE from 4.295 to 0.430, while unseen moderate latent RMSE rose from 0.09196 to 0.17772 and stable RMS from 0.04404 to 0.15578. About 85% of the initial fitting R-MSE came from location mismatch. A positive global affine fit on frozen fitting scores achieved **0.420588** on the same validation set, beating 0.429631 without changing normalized predictions. The offset explains much of the apparent level gain. Increased normalized stable movement shows the scorer changed in ways an offset cannot explain. [Inverse report and held-out affine comparison](e6_inverse_rank_diagnostic/ANALYSIS.md)
4. **Affine-aligned joint inverse.** Keep RankNet on raw `s` and use `a+b*s`, `b>0`, only in the standardized R-level loss. Initialize the two calibration parameters with fitting-only affine calibration. In the bounded trial, observed validation selected `λ_R=10`, update 40. Validation R-MSE improved from 0.420588 to 0.409149, so the scorer update added level information beyond frozen calibration. Unseen moderate and small latent RMSE still worsened to 0.09683 and 0.07020, and stable RMS rose to 0.05943. This is one aligned inverse trial with noisy labels in the right units. [Aligned inverse report](e6_aligned_inverse_diagnostic/ANALYSIS.md)

## Paired observed-outcome change diagnostic

The original E6 labels could not supply actual observed differences. This diagnostic uses a new paired fitting panel under the same 32-city quadratic/nonlinear DGP. Fitting seed 1422665402 gives 6,560 tracts with 13,120 labeled endpoints. Validation seed 1157853548 gives 1,632 tracts with 3,264 labeled endpoints. The fitting panel is about half stable, with 3,296 assigned stable tracts. Actual paired endpoints of changing tracts are never supplied as stable links. There are 26,240 same-H auxiliary links, two per fitting endpoint. All arms share all labeled endpoints. The E6 independent calibration and test panels are reused.

Add `L_W=MSE_fit(W)/30.124436698937078` and `L_ΔW=MSE_fit(ΔW)/16.201447520252533`, with both coefficients one in the full joint arm. The predicted difference uses the appropriate head at each endpoint. This bounded diagnostic omits the extra primary E6 head penalty. Refit A on the paired sample. Its original training executes 520 Adam updates and restores update 320. Its frozen level-only head preparation retains update 600.

**Arm letters change here.** A is the fitted ordinal scorer plus its calibrated head. O continues the ordinal scorer with detached level calibration. B freezes A and trains heads on levels plus changes. C jointly trains on levels. D jointly trains on levels plus changes. All branches have 200 additional updates at scorer/head LRs `0.0001/0.001`, with matched data and fixed-budget endpoints. Scorer-based A/B metrics are identical.

On unseen cities, A/B small and moderate latent RMSE are 0.05357 and 0.08818. D raises them to 0.06465 and 0.09387. Stable RMS rises from 0.04389 to 0.05861, the cutoff from 0.09011 to 0.11775, and small-change sensitivity falls from 88.0% to 72.0%. D also has worse seen observed-change MSE than A, B, C and O. B accounts for about 57% of D's **fitting** change-loss gain. Both B and D worsen held-out change prediction. The validation change loss diverges from the improving fitting loss. [Paired change-loss report](e6_observed_change_diagnostic/ANALYSIS.md) · [Full endpoint table](e6_observed_change_diagnostic/compact_results.md)

## Oracle conditional means and protected training

The next follow-up reevaluated the saved A/O/B/C/D checkpoints on noise-integrated expected outcomes. Adaptive deterministic quadrature integrated the actual centered Gumbel law and agreed across precision settings to roughly machine precision on this panel. On the seen moderate panel, observed change noise contributes a floor of 14.6998 MSE. Conditional-mean change MSE is 0.1322 for A, 0.2051 for frozen B and 0.2391 for joint D. D's worsening also appears separately for stable, small-change and moderate-change groups. The noise floor explains why realized outcome MSE looks insensitive to these differences. It does not conceal an expected-change improvement for D.

With `e_t = prediction_t - target_t`, change MSE equals `E[e0²]+E[e1²]-2E[e0*e1]`. About 0.091 of D's 0.107 increase over A comes from reduced covariance of the conditional-mean prediction errors across dates. Mean change bias contributes almost none. Frozen B's deterioration shows that changing date-specific calibration can weaken cancellation even with identical latent scores. This decomposition describes a finite panel. [Oracle evaluation and residual decomposition](e6_change_followups/ANALYSIS.md)

The protected-training branch used frozen F, unconstrained joint U and constrained joint Q. Common head preparation selected update 10 of 1,000, with a tiny validation gain. F/U/Q each received 200 head updates and U/Q each had 200 scorer attempts. Q minimizes cardinal loss subject to fitted rank loss ≤ `0.420983854316` and fitted `S/V` ≤ `0.001936853879`, bounds fixed from A plus the specified slacks. It projects the nominal Adam displacement onto linearized halfspaces, backtracks and checks both nonlinear bounds and cardinal descent before accepting. Rejected scorer proposals restore Adam state and parameters.

Q accepted 22 attempts, then stalled at the stability boundary for the remaining 178. It kept unseen stable RMS near 0.044 and small-change sensitivity near 88%, while U raised stable RMS to 0.059 and reduced sensitivity to 72%. Q yielded no meaningful gain over frozen F. The user corrected the interpretation of the stall. For an active constraint `g(θ)=0` and a tangent direction `∇g·d=0`, a positive quadratic term can make every sufficiently small straight positive step infeasible. Halving does not produce an inward direction. The rejection count provides little evidence against useful constrained retraining with an inward margin, a feasibility correction or a stronger solver. The prior report now includes that correction. An alternative Q solver remains untested. [Constrained-training results and correction](e6_change_followups/ANALYSIS.md)

## Latest experiment and exact results to carry forward

The user asked whether starting jointly from scratch or extending warm training would change the result. The three paths reuse the paired data and panels. Both scratch arms have bitwise identical initial scorer parameters and heads. The ordinal scorer uses RankNet plus stability, while its concurrent level/change heads receive detached scores. Joint scratch has both cardinal losses active from update one. Extended warm starts at the original A scorer and saved level-calibrated head. All paths use fresh Adam states, scorer/head LR `0.001/0.001`, and 3,000 new updates. The scorer rate is ten times the preceding short diagnostic rate and matches the primary E6 joint scorer rate. Both joint cardinal weights are one.

Observed validation is checked every ten steps with 20-check patience and minimum improvement `1e-5`. Ordinal selection uses validation rank logistic loss. Joint selection adds standardized level and change MSE with equal weights. Selection freezes at patience, with update zero eligible. Training continues to the prespecified endpoint for a separate fixed-budget comparison. Test and oracle evaluation runs after training. Ordinal patience fires at 1,330 and selects 1,130, although its validation rank loss improves again and reaches its logged full-budget minimum at 2,690. Joint scratch selects 670 at patience 870. Warm start selects zero at patience 200, and zero remains its best observable validation checkpoint over the full budget.

| Path | New updates at checkpoint | Unseen small RMSE | Unseen moderate RMSE | Unseen stable RMS | Unseen small sensitivity |
|---|---:|---:|---:|---:|---:|
| Ordinal scratch, selected | 1,130 | 0.05827 | 0.10457 | 0.04223 | 86.92% |
| Joint scratch, selected | 670 | 0.19446 | 0.22107 | 0.17875 | 13.96% |
| Warm, selected A | 0 | 0.05357 | 0.08818 | 0.04389 | 88.00% |
| Ordinal scratch, fixed | 3,000 | 0.06212 | 0.09373 | 0.05462 | 76.70% |
| Joint scratch, fixed | 3,000 | 0.26901 | 0.28825 | 0.27011 | 8.75% |
| Warm, fixed | 3,000 | 0.26221 | 0.27824 | 0.26411 | 10.23% |

The selected joint scratch cutoff is 0.357 versus 0.089 for ordinal scratch, at fresh FPR of 5.27% versus 4.59%. Small-change slopes are 0.871 versus 0.923, while correlations are 0.643 versus 0.950. Nuisance movement and scatter contribute substantially alongside some attenuation. At the selected checkpoints, the raw within-cell score variances are about 0.00146 for joint scratch and 2.518 for ordinal scratch. This scale disparity affects raw-gap RankNet. It cannot by itself explain normalized-change deterioration, since positive affine rescaling cancels in the metric.

The scratch heads' scale and knots were fixed using fitting scores at the common random initialization. The ordinal scorer's raw score SD grew from 0.055 to 1.589 by selection. Its detached heads failed to track that scale, producing expected level/change MSE of about 238/23.36. Treat these outcome errors as a head-tracking failure. Joint scratch cannot claim cardinal gains against that calibration benchmark. Against calibrated A, its expected change MSE is also worse, 0.568 versus 0.132. The warm comparison changes both the initial scorer and head preparation. The tests verify identical scratch tensors, nonzero cardinal gradients at initialization, monotone heads and original pairwise dispersion.

Total new computation was 9,000 scorer updates. Warm start also inherits the 520 updates originally executed to produce A, for **3,520** total executed scorer updates on that path. A's restored parameters come from baseline update 320, making the warm endpoint's parameter ancestry 3,320 updates. The two counts answer different questions. Neither joint path improves over the ordinal controls at this fixed setting. The evidence supports a bounded claim about these training strategies and leaves the choice of loss weights, head initialization and constrained solver open. [Latest complete report](e6_initialization_diagnostic/ANALYSIS.md) · [Selected and fixed endpoints](e6_initialization_diagnostic/selected_and_fixed_summary.md) · [Learning trajectories](e6_initialization_diagnostic/learning_trajectories.png) · [Test trajectories](e6_initialization_diagnostic/later_test_trajectories.png)

## Reproduction and file map

The E6 numerical environment used for the stopped main run and recent diagnostics is WSL Python at `/home/abbatenicolas/miniconda3/envs/torch_geo_env/bin/python`, with Python 3.10.20, PyTorch 2.5.1, NumPy 2.2.6 and SciPy 1.15.2. The WSL suite directory is `/mnt/c/Working Papers/NY State Aerial Imagery Prototype/ny_state_aerial_imagery_prototype/src/ordinal_mc_suite`. Use that environment's `python` for fitting commands below. The Windows Python available during the diagnostics lacked PyTorch. Prior Codex calls used `wsl.exe --exec` with the exact interpreter and quoted script/argument paths. The main runner pins fitting source and runtime versions. Consult `e6_main_512/manifest.json` if a resume rejects a mismatch.

| Purpose | Code and records |
|---|---|
| Original suite and append-only ML | `run.py`, `ordinal_mc/`, `append_ml_benchmarks.py`, `append_additive_benchmark.py`, `append_rho_benchmarks.py` |
| Primary E6 | `run_e6.py`, `ordinal_mc/e6/`, `configs/e6_main_512.json`, `e6_main_512/results.sqlite` |
| Read-only report regeneration | `render_e6.py`, `e6_main_512_interim/snapshot.sqlite`, `e6_main_512_interim/balanced_report/` |
| Local gradient and inverse checks | `verification/diagnose_e6_gradients.py`, `probe_e6_step_size.py`, `e6_targeted_trials.py`, `e6_inverse_with_rank.py`, `e6_inverse_affine_validation.py`, `e6_aligned_inverse.py` |
| Paired change experiment | `verification/e6_observed_changes.py`, `configs/e6_observed_change_one_shot.json`, `e6_observed_change_diagnostic/checkpoints.pt` |
| Oracle/Q follow-up | `verification/e6_change_followups.py`, `verification/report_e6_change_followups.py`, `configs/e6_change_followups.json`, `e6_change_followups/` |
| Latest three paths | `verification/e6_initialization_paths.py`, `verification/audit_e6_initialization_paths.py`, `verification/report_e6_initialization_paths.py`, `configs/e6_initialization_one_shot.json`, `e6_initialization_diagnostic/` |

From the suite directory, the read-only renderer writes to a fresh output directory. The fitting resume command is for use when the user requests restarting the stopped experiment. Completed paired jobs are skipped. Interrupted jobs start over at the job level. Eight workers run independent jobs with one numerical thread each.

```bash
python render_e6.py --run e6_main_512 --out e6_main_512_report_new
python run_e6.py --config configs/e6_main_512.json --out e6_main_512 --workers 8 --resume
```

The latest three-path report can be regenerated from completed artifacts with `python verification/report_e6_initialization_paths.py --out e6_initialization_diagnostic`. For an explicitly requested reproduction of the same fit, use a fresh directory and run the fit, audit and reporter in that order.

```bash
python verification/e6_initialization_paths.py --source e6_observed_change_diagnostic --config configs/e6_initialization_one_shot.json --out e6_initialization_diagnostic_new
python verification/audit_e6_initialization_paths.py --out e6_initialization_diagnostic_new
python verification/report_e6_initialization_paths.py --out e6_initialization_diagnostic_new
```

The paired diagnostic and Q report have their own runnable commands and artifact descriptions in their `ANALYSIS.md` files. Use `python -m unittest discover -s tests -p 'test_e6*.py' -v` for E6 implementation/report tests and `python verification/test_e6_change_followups.py` for the four focused numerical checks. The latest audit outputs are `checkpoint_audit.json` and `verification.json`. The saved artifacts are the first place to look when the user asks about an existing result. Preserve the distinction between the partial replicated main evidence and the later one-shot training comparisons when writing about the results.
