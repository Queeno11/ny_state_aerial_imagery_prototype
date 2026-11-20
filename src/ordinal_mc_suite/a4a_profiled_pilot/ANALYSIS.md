# Profiled outcome heads in the A4(a) pilot

Exact head profiling removed the conditional head-optimization gap. With correct head initialization, ordinary joint Adam already stayed close to the conditional optimum. The validation-selected profiled family gave nearly the same fresh-sample structural and outcome-change performance as ordinal continuation with equally capable calibration. Within the profiled arms, **regularized levels (LC) predicted outcomes better than regularized levels plus differences (LDC)**. On the fresh draws the own-head expected outcome-change MSE was .15110 for LC and .17466 for LDC. Giving both scorers the same validation-selected head library reduced this difference to .00021.

## Matched design

The replay used draws 201–203 from the balanced pilot. Fresh confirmation used independent draws 301–303. Each draw paired 256 tracts in each of 32 cities at two dates. The fit and validation split was at tract level, with 13,120 fitting endpoints, 3,264 validation endpoints and two external exact-H acquisition redraws per endpoint. A common ordinal anchor ran 1,500 Adam updates. All continuations started from that trained scorer, used the same full-batch data and spline knots, and ran 1,000 scorer updates at learning rate .0001. This compares local warm-start paths.

The realized A4(a) outcome is W = ψ_ct(f(H)) + e, with ψ_ct(f(H)) the exact conditional mean and independent conditionally Gaussian endpoint errors. The previous noise-before-map R belongs to a different design. The squared difference loss uses the same endpoint outcomes as the level loss, with paired residual differences emphasizing change. The objective retains ordinal weight 1 and external stability weight .3. The active level and change weights are .0279542993 and .0091976915, with curvature weight .001 when present. These were carried forward from the balanced pilot. The experiment used no new coefficient search.

| Path | Scorer updates | Primary outcome head |
|---|---:|---|
| F, frozen ordinal anchor | 0 after anchor | Validation choice among L, LC, LD and LDC |
| O, ordinal continuation | 1,000 | Same four options at every checked checkpoint |
| Legacy Adam LDC | 1,000 | Joint Adam, initialized at the historical equal-weight unpenalized head |
| Correct-init Adam LD and LDC | 1,000 each | Joint Adam, initialized at the fit-only optimum of the branch objective |
| Profile L, LC, LD and LDC | 1,000 each | Exact conditional head for the branch objective at every update |

L uses levels, D adds paired differences when combined with L, and C adds discrete squared slope-change curvature. Each head has one intercept and four positive piecewise-linear slopes per city–date cell. It receives the normalized scalar score and cell identity. The three knots stay fixed at fitting-score quantiles of the ordinal anchor. Cardinal losses are scaled by pooled within-cell fitting outcome variance and fitting paired outcome-change variance. Curvature penalizes adjacent slope jumps on the standardized outcome and score scales.

Every profiled step computes the differentiably normalized fitting score, solves the branch's convex head problem in float64 at that score, and uses the scorer partial derivative holding the minimizing head fixed. The solver objective is rescaled by the positive level coefficient for numerical conditioning. Curvature enters this conditional minimization and influences the scorer through the selected head. Native Adam updates its own head at learning rate .001, projecting slopes to the positive floor. The conditional solves for the native paths measure head gaps at checkpoints and do not feed back into their updates.

Primary selection at updates 0, 25, …, 1,000 used **observed rank misordering + .3 external stability + standardized level MSE + standardized change MSE** with each estimator's own head. Ties within 1e−6 favor earlier updates and then simpler head objectives. The profiled family chooses across all four trajectories by that same validation risk, with ties favoring simpler objectives and then earlier updates. That family expends four continuations while O expends one. Each individual trajectory uses the same 1,000-update scorer budget. A secondary comparison fits the identical four-option head library on every selected scorer and selects a head on validation. The historical equal-weight, unpenalized refit selector is retained for the legacy comparison. [Protocol](protocol.json) · [Run manifest](evaluation/manifest.json)

## What changed on the same legacy path?

All three replay anchors and historical ordinal checkpoints were reproduced bitwise. The legacy LDC scorer and native head at the fixed and historical-selected checkpoints also matched bitwise. Re-scoring that unchanged trajectory with its native head changed the selected updates from **525, 325, 0** to **525, 350, 350**. In replay draw 203, native expected change MSE fell from .30203 at the historical selection to .19931 at the own-head selection. The two selections were identical in draw 201 and had nearly identical expected change MSE in draw 202.

On fresh draws the same legacy control's historical selector chose 0, 50 and 0, while its own-head selector chose 350, 400 and 400. Native expected change MSE improved from (.31950, .24421, .27350) to (.22099, .19821, .19072). This is a checkpoint-selection effect on the same trajectory and head class. Its mean stable score movement was .05141, above ordinal continuation's .04650. Its mean expected outcome-change MSE remained above ordinal continuation with its validation-selected calibration.

## Initialization and profiling

On fresh draws the historical head initialization left a weighted fitting cardinal loss gap of .000335–.000909 versus the optimal head for the same initial scorer and LDC objective. At the selected legacy checkpoints the gap was .0000165–.000321. Correct-init Adam LDC reduced the initial gap to numerical precision. Its final gap stayed below 6.2e−9 on the fresh draws. Profile LDC was at the numerical floor. This gives little evidence of a material head-tracking problem for correctly initialized Adam LDC in these runs.

| Data | Selected method | Unseen latent change RMSE | Stable RMS | Seen own expected-W level MSE | Seen own expected-W change MSE |
|---|---|---:|---:|---:|---:|
| Replay | Legacy Adam LDC | .10751 | .05733 | .17126 | .20469 |
| Replay | Correct-init Adam LDC | .10992 | .05664 | .16756 | .18983 |
| Replay | Profile LDC | .10975 | .05642 | .16722 | .18933 |
| Fresh | Legacy Adam LDC | .10715 | .05141 | .16779 | .20331 |
| Fresh | Correct-init Adam LDC | .10732 | .04657 | .15502 | .17450 |
| Fresh | Profile LDC | .10728 | .04663 | .15482 | .17466 |

Correct initialization improved mean own-head outcome prediction and reduced stable movement relative to the legacy path, with mixed latent RMSE. All three fresh correct-init Adam LDC selections were at update zero. Profile LDC selected the same anchor twice and took 25 updates in the third draw. Comparing their own heads, profiling changed fresh expected change MSE by zero, +.00047 and zero. At the fixed 1,000-update endpoint, fresh expected change MSE was .20112 for correctly initialized Adam LDC and .19948 for Profile LDC, both above the ordinal endpoint's .17335. Profile LDC's fresh fixed latent RMSE was .11243 versus .11198 for correct-init Adam and .10988 for ordinal.

All profiled paths reduced their fitting objectives over the budget. None of the 24,000 transitions increased the recorded float32 profile objective by more than 1e−7. Scorer gradient norms at the fixed endpoints ranged from .00157 to .00299 across profiled arms. Conditional head convergence and fixed-budget scorer progress are separate. [Optimization by replication](results/optimization_by_rep.csv) · [Endpoint gradients](audit/endpoint_gradients.csv)

## Does the paired difference loss help once the head is regularized?

| Data | Selected method | Unseen latent RMSE | Stable RMS | Own expected-W level/change MSE | Equal-library expected-W level/change MSE |
|---|---|---:|---:|---:|---:|
| Replay | Frozen F | .10992 | .05382 | .16003 / .16245 | .16003 / .16245 |
| Replay | Ordinal O | .10892 | .05478 | .15619 / .16281 | .15619 / .16281 |
| Replay | Profile L | .10939 | .05560 | .20920 / .26910 | .15613 / .16339 |
| Replay | Profile LC | .10920 | .05541 | .15607 / .16310 | .15607 / .16310 |
| Replay | Profile LD | .11034 | .05763 | .21884 / .28149 | .15605 / .16580 |
| Replay | Profile LDC | .10975 | .05642 | .16722 / .18933 | .15695 / .16449 |
| Replay | Validation-selected profile family | .10920 | .05541 | .15607 / .16310 | .15607 / .16310 |
| Fresh | Frozen F | .10732 | .04657 | .14746 / .15722 | .14746 / .15722 |
| Fresh | Ordinal O | .10723 | .04650 | .14678 / .15713 | .14678 / .15713 |
| Fresh | Profile L | .10721 | .04652 | .19122 / .24695 | .14713 / .15717 |
| Fresh | Profile LC | .10730 | .04666 | .14302 / .15110 | .14654 / .15716 |
| Fresh | Profile LD | .10717 | .04654 | .20610 / .26426 | .14719 / .15734 |
| Fresh | Profile LDC | .10728 | .04663 | .15482 / .17466 | .14726 / .15737 |
| Fresh | Validation-selected profile family | .10726 | .04672 | .14634 / .15732 | .14634 / .15732 |

In the fresh draws, LDC minus LC own expected change MSE was **+.02533, +.01865, +.02669**. With the equal library, the differences were +.00017, +.00047 and 0. The same pattern appears on the replay. Adding curvature to a level-only head also materially reduced expected change MSE compared with unregularized levels (.15110 versus .24695 fresh). The absolute benefit of a head constraint is distinct from the benefit of changing the scorer.

The prespecified LC arm had lower fresh mean own expected-change error than ordinal O. The entire advantage came from draw 302, where LC and O used the **same ordinal anchor at update zero**. LC used its regularized level-only head. The observed validation library chose LDC for O, whose test expected change MSE was .18204 versus .16386 for LC. The equal library also chose LDC for the Profile LC scorer in that draw. On draw 301, active Profile LC had a .00008 higher expected change error than O and .00020 higher latent RMSE. Draw 303 tied. This distinguishes the prespecified LC comparison from the performance of choosing a method by validation. The validation-selected family chose LC at updates 50 and 0 in draws 301 and 303, and LDC at update 25 in draw 302.

| Fresh draw | Family head and update | Family minus O latent RMSE | Stable RMS difference | Own expected-W level MSE difference | Own expected-W change MSE difference |
|---|---|---:|---:|---:|---:|
| 301 | LC, 50 | +.000196 | +.000468 | −.000717 | +.000077 |
| 302 | LDC, 25 | −.000116 | +.000186 | −.000614 | +.000470 |
| 303 | LC, 0 | 0 | 0 | 0 | 0 |

The expected outcome-change comparison is more sensitive than realized change on these data. The fresh realized change noise floor was 7.97880 MSE and the zero-change predictor's expected change MSE was 2.40698. Mean realized change MSE was 8.13357 for O and 8.13422 for the profiled family. The comparisons across three fresh draws are descriptive. [Per-draw outcomes and errors](results/by_rep.csv) · [Matched contrasts](results/paired_by_rep.csv)

## Structural signs, detection and units

On fresh unseen cities, O flagged 84.13% and 84.00% at nominal steps −.15 and +.15, with approximately 98.8% correct signs. The profiled family flagged 83.88% and 83.82%. At nominal −.05 and +.05, O flagged 21.64% and 21.04% and signed 88.80% and 88.40% correctly. The profiled family was close at 21.54% and 20.88% flags. Fresh dose-zero false positives were 5.30% for O and 5.36% for the profiled family. Each signed step used 4,096 independently held-out unseen tracts per draw with acquisition redraws shared across arms. The cutoffs were the higher empirical 95th percentiles of absolute normalized movement in independent stable calibration panels.

Fresh O latent-change slopes were .876, 1.055 and .866 from low to high initial f tertiles. The profiled family slopes were .876, 1.055 and .866. Applying the finite-fit normalization to the true score itself contributed .01061 RMSE in the changed cohort. The held-out acquisition-only, measurement-map-only, structural-only and opposed-map cases use the same image contrasts as the prior pilot. Predictions are identical across each pair of regimes that differs only by the expected outcome mapping. Practical outcome MSE is restricted to supported seen-city cells. The dose figure's band describes tract heterogeneity, with no interpretation as uncertainty about the mean.

The monotone transformation check in the preceding pilot preserved ranks, exact stable equality and conditional-mean predictions while altering normalized latent gaps. These errors against the simulated f are DGP-specific descriptive measures. Outcome improvements can reflect calibration or a change of score coordinates. They do not establish identification of latent change magnitudes under A4(a). [Dose plot](results/signed_dose_response.png) · [Dose metrics](evaluation/dose_detection_by_rep.csv)

## Verification and reproduction

All six runs completed **57,000 scorer updates and 24,024 profiled head solves** in 782.6 seconds. The tiny finite-difference directional checks matched the envelope gradient to 6.03e−8. Maximum projected head gradient in the full profiled run was 9.31e−7. The selected and fixed cold head re-solves differed from saved conditional losses by at most 9.72e−14. Their maximum validation prediction difference was .000179. The GPU/CPU precision checks differed by at most 1.36e−6 in score and 5.92e−6 in outcome predictions. The audit verifies saved-scorer/head alignment, full budgets, replay equality, seed separation, own-head and library choices, positive slopes and outcome residual identities. The prior known-map and direct-f controls retain identical config and weights and are linked as prior runs. They add zero updates here.

The result reflects converged head subproblems, observed progress of the scorer objective, and local finite-budget warm starts. The comparison addresses initial head mismatch, profiling and own-head selection under the tested DGP, weights and head family. [Audit](audit/audit.json) · [Preflight](preflight/checks.json) · [Prior known-map controls](../a4a_balanced_pilot/evaluation/known)

From the suite directory in the existing WSL fitting environment, set PYTHONPATH=. and OPENBLAS_NUM_THREADS=1 and use fresh destinations.

```bash
python a4a_profiled_pilot/preflight_profile.py --out NEW/preflight.json
python a4a_profiled_pilot/run_profiled.py --out NEW/evaluation
python a4a_profiled_pilot/audit_profiled.py --run NEW/evaluation --out NEW/audit
python a4a_change_pilot/dose_metrics.py --run NEW/evaluation --out NEW/evaluation/dose_detection_by_rep.csv
python a4a_profiled_pilot/report_profiled.py --run NEW/evaluation --doses NEW/evaluation/dose_detection_by_rep.csv --out NEW/results
```
