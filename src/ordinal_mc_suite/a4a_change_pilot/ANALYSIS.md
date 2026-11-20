# Paired change under additive A4(a) outcome noise

Adding level and change losses to the trained ordinal scorer gave no reproducible improvement over budget-matched ordinal continuation. With common outcome calibration, joint level-plus-change training raised unseen latent-change RMSE in two of three replications, raised seen expected outcome-level MSE in all three, and raised expected outcome-change MSE in two. Both joint arms with level and change supervision selected the frozen starting scorer twice. Their full-budget endpoints moved farther and had worse change recovery.

The ordinal continuation usually recovered the direction of a structural change. It flagged roughly 83% of nominal ±.15 changes and 17% of nominal ±.05 changes, with about 4% false positives on fresh stable tracts. Magnitude error depended on starting level. Joint training changed detection in opposite directions across replications.

## Identification and positive controls

For any strictly increasing invertible transformation h, the transformed score h∘f and composed heads ψ_ct∘h⁻¹ produce the same conditional means. Recentring and rescaling either score on fitting observations can be absorbed into the heads. This preserves realized outcomes under identical additive errors, their differences, true ranks and exact stable equality. It leaves normalized nonzero latent gaps free to change. A restricted head class or curvature preference can favor a coordinate. A4(a) itself restricts the ordering.

With h(u) = u + .3 tanh(u), the bracketed float64 inverse reproduced held-out levels and differences within 3.6e−15. Ranks matched exactly and stable latent movement was zero. Normalized changes differed by **.0369 RMS** and up to **.1477** on assigned changing tracts. Finite-fit normalization of the original f alone contributed .0132 change RMSE. A discrete secant curvature proxy using matched latent quantiles and standardized outcome units fell from .0917 to .0213 after transformation. This proxy illustrates coordinate preference separately from the fitted PWL penalty. [Numerical preflight](preflight/checks.json)

The known-map control evaluated ψ_ct at u = μ_fit(f) + √V_fit(f) z. A perfect normalized scorer therefore returns the correct conditional mean, accounting explicitly for finite-sample versus population normalization. The two starts share one data replication and begin each comparison from identical scorer parameters.

| Training labels | Start | Seen expected-W level MSE, ordinal → full | Seen expected-W change MSE, ordinal → full | Unseen latent RMSE, ordinal → full | Unseen stable RMS, ordinal → full |
|---|---:|---:|---:|---:|---:|
| Noiseless | 0 | .1697 → .0622 | .0400 → .0274 | .0800 → .0731 | .0342 → .0298 |
| Noiseless | 1 | .1452 → .0403 | .0255 → .0194 | .0677 → .0597 | .0244 → .0244 |
| Additive noise | 0 | .1709 → .1463 | .0666 → .1054 | .1029 → .1100 | .0457 → .0749 |
| Additive noise | 1 | .9083 → .2260 | .0765 → .1101 | .1077 → .1229 | .0510 → .0688 |

Noiseless regression on normalized true f with the same architecture achieved validation pointwise MSE .00371 and .00378. Its unseen change RMSE was .0764 and .0756, with stable RMS .0624 and .0579. It used no stability penalty. Both clean full fits and both direct fits selected update 1,500, so these are finite-budget capacity checks. The noiseless control is deterministic. The noisy known-map fits are privileged diagnostics.

At update 1,500, the noisy known-map full arm had lower fitting standardized change MSE than ordinal in both starts (.8186 and .8183 versus .8409 and .8393). Validation reversed this ordering (.8991 and .8970 versus .8921 and .8916). From update 750 to 1,500, the fitting change error fell in both full fits while validation change error failed to improve consistently. This gap is consistent with fitting sample-specific outcome variation. The design does not isolate that explanation from objective tradeoffs. [Known-map outputs](known_control)

## Main pilot

Means over three independent draws. Latent RMSE uses the unseen mixed-panel tracts with |Δf| > .05. Outcome columns use all seen mixed-panel tracts and common level-plus-change head refits.

| Arm | Unseen latent change RMSE | Unseen stable RMS | Fresh stable FPR | Seen expected-W level MSE | Seen expected-W change MSE | Seen realized-W change MSE |
|---|---:|---:|---:|---:|---:|---:|
| Frozen anchor | .1056 | .0447 | 4.41% | .2622 | .2927 | 8.4061 |
| Ordinal continuation | .1037 | .0470 | 4.30% | .2523 | .2917 | 8.4038 |
| L, levels | .1048 | .0530 | 4.23% | .2568 | .3022 | 8.4167 |
| D, differences | .1058 | .0451 | 4.31% | .2625 | .2941 | 8.4077 |
| LD, levels and differences | .1060 | .0457 | 4.28% | .2627 | .2955 | 8.4083 |
| LDC, LD and curvature | .1059 | .0456 | 4.25% | .2623 | .2948 | 8.4079 |

| Selected contrast vs ordinal continuation | Rep | Unseen latent RMSE Δ | Stable RMS Δ | Fresh FPR Δ, pp | Seen expected-W level MSE Δ | Seen expected-W change MSE Δ |
|---|---:|---:|---:|---:|---:|---:|
| LD | 1 | +.00275 | +.00362 | −.39 | +.00592 | +.00988 |
| LD | 2 | +.00437 | −.00733 | +.29 | +.02288 | +.00168 |
| LD | 3 | −.00018 | −.00038 | +.05 | +.00240 | −.00047 |
| LDC | 1 | +.00240 | +.00350 | −.49 | +.00492 | +.00784 |
| LDC | 2 | +.00437 | −.00733 | +.29 | +.02288 | +.00168 |
| LDC | 3 | −.00018 | −.00038 | +.05 | +.00240 | −.00047 |

On the identical anchor scorer, the native level-only head had lower seen expected level and change MSE (.2050 and .2452) than the native frozen level-plus-change head (.2622 and .2927). The added loss worsened held-out outcome change even with a frozen scorer. LDC's native expected change MSE (.2789) was lower than LD's (.2956) and above the anchor's native level-only head. After the common refit, their difference shrank to .0007. LDC updated the selected scorer in one replication.

The realized change noise floor averaged 8.1084 MSE. The zero-change predictor had expected change MSE 2.4135. For common LD versus ordinal continuation, the sum of the two conditional-mean residual second moments increased by .0208, and twice the cross moment increased by .0171, leaving .0037 more expected change MSE. Endpoint errors grew more than their cross-date cancellation. The residual identity was checked exactly up to numerical tolerance.

The anchors selected at updates 1,200, 1,300 and 1,300. Ordinal continuation selected at further updates 50, 600 and 200, L at 50, 600 and 500, D at 25, 0 and 0, and LD and LDC each at 75, 0 and 0. Each branch still executed 1,000 updates. At that endpoint, ordinal continuation, LD and LDC had unseen change RMSE .1062, .1279 and .1257, stable RMS .0563, .0886 and .0859, and expected outcome-change MSE .3030, .4007 and .3670. LD's mean native fitting standardized change error fell from .8112 to .7729, while common-refit validation change loss rose from .8941 to .9022. The tables retain the best active update even where zero won, and the rank-and-stability-selected ordinal checkpoint.

At the joint start, mean scorer gradient norms for the weighted rank, stability, level and change terms were .00727, .00655, .00311 and .01448. The mean change–stability cosine was −.50 and rank–stability cosine −.90 on fitting data. Curvature acts on the heads and changes the scorer indirectly through joint updates. These local gradients and finite-budget trajectories do not establish a population conflict. [Matched comparisons and tables](results)

## Detection, sign and magnitude

| Unseen dose | True change median [5th, 95th percentiles] | Ordinal flag | Frozen anchor flag | LD flag | LDC flag | Ordinal sign accuracy |
|---|---:|---:|---:|---:|---:|---:|
| −.15 | −.159 [−.291, −.121] | 83.82% | 86.17% | 85.03% | 85.12% | 99.36% |
| −.05 | −.054 [−.099, −.041] | 16.98% | 18.37% | 17.85% | 17.85% | 89.40% |
| 0 | 0 [0, 0] | 4.16% | 4.31% | 4.31% | 4.26% | — |
| +.05 | +.054 [.040, .100] | 17.06% | 18.36% | 17.29% | 17.38% | 88.57% |
| +.15 | +.162 [.119, .299] | 82.50% | 84.77% | 83.25% | 83.26% | 98.95% |

Each replication used the same 4,096 unseen starting tracts at every dose, with independently redrawn acquisition views shared across arms and doses. Rates and quantiles are averaged across replications. At doses ±.05, for the subset with |Δf| > .05, ordinal flagged 22.3% in both directions and got about 92% of signs right. At −.15 and +.15, LD's flag-rate changes relative to ordinal continuation were (−4.22, +7.62, +.22) and (−5.66, +7.28, +.63) percentage points across the three draws. The mixed-panel actual .05–.15 bin averaged 27 unseen tracts per replication, so its rates remain secondary. Exact score ties receive half credit for ranking and zero credit for a nonzero direction.

Ordinal continuation misranked 15.91% of observed within-cell pairs and 2.52% of true-score pairs in unseen cities. Its mixed-panel changed-tract sign accuracy was 99.90%. The magnitude slopes were .91, 1.05 and .87 from the lowest to highest starting-score tertiles, with respective RMSE .096, .084 and .126. Mean true and predicted movements were +.464 and +.439 at dose +.4, and +.872 and +.821 at +.8. At +.05 the ordinal scorer beat the zero-change reference (.049 versus .064 RMSE), while individual error remained large relative to the mean movement. The figure's band averages the 10th–90th within-replication tract percentiles of predictions. It shows between-tract heterogeneity, with no interpretation as uncertainty in the mean or the prediction error. [Signed-dose metrics](results/dose_detection_by_rep.csv) · [Dose response](results/unseen_dose_response.png)

Acquisition-only and map-only regimes used identical image pairs. The map shift changed the expected outcome by about 1.00 in absolute value with zero structural change. The same score movement in both regimes separates the targets at inference. It cannot measure the effect of changing training maps. The structural-only and deliberately opposed-map regimes also shared their image pairs. Ordinal continuation agreed with the expected outcome direction in 99.87% of the structural-only pairs and 0.13% of the shifted opposed pairs, with identical latent sign behavior. These counterfactual maps receive structural metrics without practical head outcome MSE. Within trained maps at nominal ±.15, the predeclared true sign-opposition subset averaged 3,983 seen tracts per draw, all true declines. Ordinal continuation signed 99.14% and flagged 82.22% of them. The unselected supported panel is retained alongside this subset.

## Design and verification

- The existing quadratic f(H), bounded structural distribution, rotation and seven-input feature mix used function seed 983 and composition zero. Population normalization used the suite's independent calibration convention V(f) = 1. Changes used the existing gradient-aligned perturbation and support clipping. Latent magnitude comparisons are descriptive for this DGP.
- Additive labels were W = m_ct(H) + ν b_ct Z, with m_ct = a_ct + b_ct[f(H) + k_ct tanh(f(H))], independent standard Gaussian endpoint errors, and ν = .8. The previous noisy latent R = f(H) + ε belongs to the noise-before-map design. Here m_ct is the exact conditional mean and W its noisy realization. Conditional pair errors are symmetric and continuous. In 250,000 independent calibration pairs, Bayes misranking was .15993 and the noise-to-signal SD ratio .5702.
- Each replication paired 256 tracts in 32 cities over two dates, with half assigned nominal .4 structural changes. A tract-level 80/20 split yielded 13,120 fitting and 3,264 validation endpoints. Each fitting endpoint had two external exact-H opposite-date acquisition views. Actual change pairs did not enter the stability term. Cutoff calibration, seen test, unseen test and signed-dose panels had distinct seeds. A flag requires strictly exceeding the empirical higher 95th percentile of absolute movement on the independent stable calibration panel.
- Adam used all within-cell pairs, with score centering and pair-variance normalization differentiable on each full fitting batch. Stability was .3 times mean squared normalized movement on the auxiliary pairs. Scorer complexity weight was zero. The bounded sigmoid rank surrogate used τ = .5, .2 and .1 at 0%, 40% and 70% of baseline training and .1 during continuation. Actual misranking informed validation. No finite-temperature consistency or population optimum claim is made.
- The level and change MSE denominators were pooled fitting within-cell W variance and fitting paired ΔW variance. The former avoids dominance by between-cell offsets. Each active cardinal weight was .1. Heads saw a scalar normalized score and cell ID. They used independent intercepts, four positive-slope linear segments with three ordinal-anchor fitting-score quantile knots and affine tails. Fitting-only positive-slope regressions initialized the convex head solver. LDC penalized the cell average of adjacent standardized slope differences squared divided by segment-center spacing, weighted by .01. It is a discrete curvature penalty. Difference-only supervision leaves a common city head offset unidentified. Common head refits include levels to anchor the intercepts.
- The native A anchor has a level-only head and F has a level-plus-change head on the identical scorer. O continues the ordinal scorer. Joint arms start at the same pre-continuation anchor and level-plus-change head. All continuations use the same observed composite selector at 25-update checkpoints, with fitting-only convex head refits and update zero eligible. Selected scorers have cold final refits with the same objective, and the outputs preserve native heads separately. The study used fixed .1/.01 weights and warm starts. It included neither a tuning grid nor a logistic bridge.
- The main run completed **19,500 scorer updates**, and known controls completed **15,000**. The maximum recorded projected head gradient was 1.55e−6. The largest cold versus warm final refit prediction difference was 3.80e−5. At selected checkpoints the maximum GPU float32 versus CPU float64 discrepancies were 1.08e−6 in normalized scores and 9.61e−6 in head predictions. The audit verifies manifests, exact frozen equality, selected validation minima, budgets, monotonicity, supported outcomes and residual identities. [Audit](audit.json) · [Main manifest](main_pilot/manifest.json) · [Known-control manifest](known_control/manifest.json)

A targeted follow-up would pair observed noisy supervision with the exact conditional-mean targets on the same data, scorer start, flexible heads and loss weights. Include native level-only and frozen level-plus-change heads to separate changes in calibration from changes in the scorer. The existing controls motivate this comparison, while the present pilot leaves the mechanism open.

## Reproduction

From the suite directory in the WSL fitting environment, use PYTHONPATH=. and fresh output directories.

```bash
python a4a_change_pilot/preflight.py --out NEW/preflight.json
python a4a_change_pilot/run.py --stage known --out NEW/known
python a4a_change_pilot/run.py --stage main --out NEW/main
python a4a_change_pilot/report.py --known NEW/known --main NEW/main --out NEW/results
python a4a_change_pilot/audit.py --known NEW/known --main NEW/main --out NEW/audit.json
python a4a_change_pilot/dose_metrics.py --run NEW/main --out NEW/results/dose_detection_by_rep.csv
```
