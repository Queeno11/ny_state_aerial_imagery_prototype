# Bounded targeted E6 trials

All requested checks are complete. New optimization was limited to four paths
of 100 updates each, all from replication 1, 32 cities, quadratic/ReLU scoring,
512 labels per city, nonlinear measurement, and the identical saved A/B
checkpoint. Lambda_W stayed fixed at 10. The original scorer/head learning
rates were 0.001/0.01, independently crossed with a factor of 0.1. No Monte Carlo
replications were launched or resumed, and no production settings or source
checkpoints were changed. The original environment and saved transforms were
used. Exact fitting-pair dispersion was checked against an equivalent cellwise
variance formula. The fresh oracle analysis was diagnostic only.

## 1. Smaller steps help locally but do not resolve the scientific trade-off

At equal 100-update budgets, the four paths give the following independent
unseen-city results for moderate changes (nominal step 0.4):

| Scorer LR | Head LR | Fitting W-MSE reduction | Changing-tract RMSE | Fresh stable RMS |
|---|---|---:|---:|---:|
| Frozen B, before updates | — | 0 | 0.09196 | 0.04404 |
| 0.001 | 0.01 | 0.941% | 0.11192 | 0.08358 |
| 0.0001 | 0.01 | 0.310% | 0.09590 | 0.05661 |
| 0.001 | 0.001 | 1.173% | 0.11650 | 0.09105 |
| 0.0001 | 0.001 | 0.365% | 0.09672 | 0.05825 |

Smaller scorer steps appear substantially safer at equal updates, but they
also make much less progress on fitting W-MSE. A smaller head step alone
increases that progress and has greater latent/stability costs by update 100.
Thus equal-budget results alone would overstate the benefit of slowing the
scorer.

We predeclared 0.1%, 0.25% and 0.5% relative fitting W-MSE reductions and recorded
the first actual attainment without interpolation. All four paths attain the
0.25% target, with actual reductions tightly grouped at 0.2514%–0.2537%:

| Scorer LR | Head LR | Update | Actual W-MSE reduction | Changing-tract RMSE | Fresh stable RMS |
|---|---|---:|---:|---:|---:|
| 0.001 | 0.01 | 11 | 0.2537% | 0.09636 | 0.05783 |
| 0.0001 | 0.01 | 70 | 0.2514% | 0.09490 | 0.05447 |
| 0.001 | 0.001 | 8 | 0.2522% | 0.09584 | 0.05604 |
| 0.0001 | 0.001 | 52 | 0.2522% | 0.09463 | 0.05418 |

At comparable W-MSE progress, smaller rates still modestly improve the trade-off,
so the difference is not exclusively reduced progress. But every path has both
higher latent-change RMSE and more fresh stable movement than frozen B. The
best of these four matched points still raises RMSE from 0.09196 to 0.09463 and
stable RMS from 0.04404 to 0.05418. This is not a favorable exchange of a small
stability cost for better individual magnitudes.

The two slower-scorer paths do not attain 0.5% reduction within 100 updates;
they are marked unattained, not extrapolated. All equal-budget and attainment
metrics for both seen/unseen cities and steps 0.15/0.4 are saved. These fixed
diagnostic stages are not validation-selected production estimators, and no
path or hyperparameter was chosen using its latent test performance.

## 2. The negative fitting-sample gradient angle does not persist on fresh data

One fresh labeled panel contains 16,384 units in the same 32 cities, with new
structural features, outcome errors, acquisition draws, and fresh exact temporal
views. The A scorer stays frozen. For each measurement truth, both saved affine
and spline B heads use lambda=10; both receive exactly the same scalar scores
and cells. Realized and oracle-mean losses use the SAME original fitting
variance of W. The oracle target is E[W|H,c,t], integrating the centered Gumbel
error under nonlinear measurement; it is not Psi(f(H)).

Cosines between each level-loss scorer gradient and the fresh stability gradient:

| Measurement truth | Fitted head | Realized W | Oracle conditional mean |
|---|---|---:|---:|
| Affine | Affine | +0.187 | +0.172 |
| Affine | Spline | +0.177 | +0.159 |
| Nonlinear | Affine | +0.111 | +0.084 |
| Nonlinear | Spline | +0.167 | +0.131 |

All are positive, indicating local alignment rather than conflict. Realized-W
and oracle-mean gradient directions have cosines 0.899–0.965 with one another.
Noise changes their magnitudes and directions, but removing it does not change
the sign of the stability comparison here. The earlier nonlinear/affine
fitting-sample cosine of approximately -0.54 therefore does NOT establish a
general conflict between population level fitting and stability. It is sensitive
to which sample is used at this fitted checkpoint.

This fresh comparison cannot identify whether fitting-data reuse, scorer/head
estimation error, or finite-sample outcome noise generated the earlier negative
angle. In particular, there is no negative fresh angle that disappears only
when oracle means replace realized outcomes. The proposed specific evidence
for noise-driven conflict is therefore absent in this fresh sample.

Nor is the flexible head uniformly better: normalized oracle-mean level MSE is
0.00581 versus 0.00606 for spline versus affine under affine measurement, but
0.00791 versus 0.00695 under nonlinear measurement. Greater flexibility does not
guarantee a better fitted conditional mean. These comparisons do not isolate
head misspecification as the cause, and they do not prove a shared-scorer limit.
They are one fresh-panel diagnostic, without across-panel uncertainty estimates.

## 3. Existing paired replications favor keeping the targets separate

The stopped run has 62 completed jobs. This analysis reuses balanced replication
IDs 1–14 across both city counts and structural cases (56 jobs); it adds no
replications. Paired intervals use independent replications as their units and
are pointwise 95% t intervals, without multiplicity correction.

For quadratic/ReLU scoring at 64 cities, unseen moderate changes, nonlinear
measurement with an affine head:

| Contrast | Changing-tract RMSE difference [95% interval] | Fresh stable RMS difference [95% interval] |
|---|---:|---:|
| C-B | +0.00232 [+0.00110, +0.00353] | +0.00588 [+0.00391, +0.00784] |
| C-D | +0.00261 [+0.00143, +0.00380] | +0.00560 [+0.00368, +0.00752] |

Positive values worsen each target. Across all four head/measurement combinations
at that city count and size, every mean C-B and C-D difference is positive for
both latent RMSE and stable movement. Some individual intervals include zero;
none establishes a material latent-magnitude gain compensating for the stability
cost. The full paired table also includes the other city count, small and large
changes, and D_common. This concerns the original validation-selected methods,
whereas the short learning-rate paths above are fixed diagnostic trajectories.

## Interpretation

The learning-rate comparison confirms some avoidable local optimization damage,
but does not turn level fitting into better latent recovery at comparable
fitting progress in this trial. Fresh gradients also require revising the earlier
suggestion of a general level-versus-stability conflict: the fitting-sample angle
does not reproduce on the independent panel, with either noisy or noise-integrated
targets. The saved replicated results still show no compensating magnitude gain
for C's additional stable movement. Better W fit alone is not evidence of better
latent cardinality. These bounded checks do not establish what longer paths,
other checkpoints, or other replications would do, and none were run.

Artifacts: `lr_paths.csv`, `lr_target_metrics.csv`, `matched_W_attainment.csv`,
`fresh_noise_oracle_gradients.csv`, `paired_metrics.csv`, `paired_differences.csv`,
`completed_snapshot.sqlite`, `metadata.json`, `verification.json`, and
`targeted_results.png`. Reproducible code is
`verification/e6_targeted_trials.py` in the suite; it requires a new output folder.
