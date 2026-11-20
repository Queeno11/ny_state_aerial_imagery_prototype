# Persistent tract errors: five matched replications

The new residual law keeps the old marginal outcome variance and splits it equally between a tract component and transitory noise. The tract component persists only if the tract is unchanged. The ordinal baseline's mean structural-change MSE fell relative to the old independent-noise run, but its structural-level MSE rose. After retuning, the cubic level-loss branch improved structural levels substantially while worsening small-change detection. The controlled comparisons still show no mean structural-change gain from adding the difference loss at the tested coefficients.

## New error structure

For a city with total residual SD σ_c, draw v, v′, u₀ and u₁ independently with mean zero and variance σ_c²/2. Set e₀ = v + u₀ and e₁ = (1−D)v + Dv′ + u₁, where D is the simulator's indicator of genuine structural change. The outcome remains W = ψ_c(f(X)) + e. The distribution σ_c ∼ Uniform(.2,.6), structural features, acquisition features, city maps, true changes and original tract splits are unchanged.

| Quantity | Unchanged tract | Changed tract |
| --- | --- | --- |
| Endpoint residual variance | σ_c² | σ_c² |
| Cross-date residual correlation | .5 | 0 |
| Difference residual variance | σ_c² | 2σ_c² |

Thus this design does not double the old noise. Difference noise is halved on stable tracts and unchanged on changing tracts. The primary outcome target is the conditional mean ψ_c(f), integrating over both residual components. The innovation in a genuinely unobserved amenity remains unpredictable from X; these figures do not claim to recover its realized individual change.

## Estimation and calibration

Five replications reuse the original structural draws and city effects. Each original city has 200 paired tracts: 160 fitting and 40 validation. All methods use the same exact acquisition-stability links. Primary loss coefficients were retuned on the independent replication 101 using the original 39-candidate protocol and observed validation objective. No evaluation data selected weights. Two additional branches retain the old ordinal-level and full-loss coefficients, isolating their numerical weights from retuning. Variance normalizers are recomputed from each fitting sample.

All scores are frozen for the common cardinal comparison. Every method refits its map from scratch by identical unpenalized level least squares on 28 paired calibration tracts per city and is evaluated on 252 disjoint paired tracts. The 20 seen and 20 unseen cities, features and calibration/test splits match the previous 10% experiment. New residual innovations use a separate namespace. The outcome heads in A/B/C are the same families as before: A city intercept plus common positive slope; B city positive affine; C city monotone cubic. Maps remain fixed over dates.

The original Gaussian and mixed-model benchmarks ignore the new residual correlation and are working models. Two additional Gaussian benchmarks know the true changed/unchanged status and covariance shape (correlation .5 or zero), but estimate the city SDs and all mean parameters. They are explicitly oracle references. Their GLS head and profiled variance equations reduce to the old likelihood at zero correlation. All methods receive the same level-only calibration afterward, including the oracle-likelihood scores.

No hard-invariance estimator, noisy-stable-pair experiment, or additional neural architecture was implemented in this battery.

## Frozen weights

| Map | Level-only branch: level weight | Full branch: level weight | Full branch: change weight | Overall validation-grid winner |
| --- | --- | --- | --- | --- |
| A | 0.0304905 | 3.04905 | 1.0333 | rW0.1_rD0 |
| B | 0.0405872 | 0.405872 | 0.00878312 | rW0_rD0 |
| C | 3.17149 | 0.317149 | 0.00595012 | rW10_rD0 |

Rank weight is 1 and stability weight .3. Each matched all/changed/lower-weight branch uses the level-only branch's level coefficient and the full branch's change coefficient. Changed-only averages the masked squared residual difference over all pairs; it does not double each retained pair's weight. The lower-weight all-tract control multiplies the change coefficient by the actual fitting changed fraction. Masks use true simulation status. Curvature variants are retained in the full data.

The independently tuned full and level branches are packages with different level weights. In C the full branch's level coefficient is one-tenth of the level-only branch's coefficient. Its better change performance cannot therefore be attributed to adding the difference term. The matched comparisons below isolate that term.

## Four comparable tables

Each cell is mean city MSE ± city SD. We first calculate the equal-city mean and sample SD across 20 cities within each replication, then average those statistics over five replications. The SD describes city dispersion, not a standard error. The machine-readable summary also retains per-replication ranges. These are five-shot descriptive results.

### Structural levels f

Values in units of 10⁻⁶.

| Map | Scorer | Held-out seen cities | New cities after calibration |
| --- | --- | --- | --- |
| A | Ordinal + stability | 41.94 ± 4.753 | 41.73 ± 4.224 |
| A | Ordinal + levels + stability | 41.11 ± 4.657 | 40.9 ± 4.137 |
| A | Full loss + stability | 71.12 ± 5.258 | 71.65 ± 5.513 |
| A | Gaussian ML + stability, independence working model | 27.87 ± 2.647 | 27.99 ± 2.382 |
| A | Gaussian ML + stability, oracle pair covariance | 26.75 ± 2.47 | 26.95 ± 2.076 |
| B | Ordinal + stability | 53.22 ± 6.096 | 52.92 ± 5.355 |
| B | Ordinal + levels + stability | 50.36 ± 5.754 | 50.08 ± 5.054 |
| B | Full loss + stability | 40.93 ± 4.425 | 40.79 ± 3.95 |
| B | Gaussian ML + stability, independence working model | 23.25 ± 1.985 | 23.52 ± 1.792 |
| B | Gaussian ML + stability, oracle pair covariance | 22.86 ± 1.957 | 23.16 ± 1.701 |
| C | Ordinal + stability | 68.95 ± 7.871 | 68.6 ± 6.97 |
| C | Ordinal + levels + stability | 37.28 ± 2.71 | 37.53 ± 3.182 |
| C | Full loss + stability | 36.63 ± 3.92 | 36.51 ± 3.542 |
| C | Gaussian ML + stability, independence working model | 19.78 ± 1.563 | 19.97 ± 1.314 |
| C | Gaussian ML + stability, oracle pair covariance | 20.34 ± 1.568 | 20.58 ± 1.378 |

### Conditional-mean outcome W

Values in units of 10⁻³.

| Map | Scorer | Held-out seen cities | New cities after calibration |
| --- | --- | --- | --- |
| A | Ordinal + stability | 4.153 ± 5.194 | 3.972 ± 4.858 |
| A | Ordinal + levels + stability | 4.152 ± 5.194 | 3.971 ± 4.857 |
| A | Full loss + stability | 4.18 ± 5.187 | 3.986 ± 4.838 |
| A | Gaussian ML + stability, independence working model | 4.129 ± 5.156 | 3.951 ± 4.841 |
| A | Gaussian ML + stability, oracle pair covariance | 4.129 ± 5.156 | 3.947 ± 4.839 |
| B | Ordinal + stability | 7.647 ± 7.73 | 10.77 ± 11.53 |
| B | Ordinal + levels + stability | 7.644 ± 7.726 | 10.77 ± 11.53 |
| B | Full loss + stability | 7.634 ± 7.712 | 10.76 ± 11.55 |
| B | Gaussian ML + stability, independence working model | 7.599 ± 7.678 | 10.73 ± 11.52 |
| B | Gaussian ML + stability, oracle pair covariance | 7.607 ± 7.688 | 10.72 ± 11.51 |
| C | Ordinal + stability | 22.99 ± 46.01 | 25.97 ± 39.57 |
| C | Ordinal + levels + stability | 22.62 ± 44.42 | 26.19 ± 40.43 |
| C | Full loss + stability | 22.78 ± 45.16 | 26.06 ± 40.01 |
| C | Gaussian ML + stability, independence working model | 22.85 ± 45.56 | 26.09 ± 40.32 |
| C | Gaussian ML + stability, oracle pair covariance | 22.78 ± 45.19 | 26.13 ± 40.52 |

### Structural changes Δf

Values in units of 10⁻⁶.

| Map | Scorer | Held-out seen cities | New cities after calibration |
| --- | --- | --- | --- |
| A | Ordinal + stability | 3.854 ± 0.4187 | 3.798 ± 0.2619 |
| A | Ordinal + levels + stability | 4.03 ± 0.4383 | 3.972 ± 0.2775 |
| A | Full loss + stability | 84.51 ± 8.073 | 84.85 ± 7.971 |
| A | Gaussian ML + stability, independence working model | 13.19 ± 1.41 | 13 ± 1.12 |
| A | Gaussian ML + stability, oracle pair covariance | 17.36 ± 1.691 | 17.42 ± 1.598 |
| B | Ordinal + stability | 4.233 ± 0.4475 | 4.185 ± 0.2992 |
| B | Ordinal + levels + stability | 4.459 ± 0.4725 | 4.409 ± 0.3207 |
| B | Full loss + stability | 9.498 ± 0.9595 | 9.421 ± 0.7976 |
| B | Gaussian ML + stability, independence working model | 16.36 ± 1.728 | 16.15 ± 1.466 |
| B | Gaussian ML + stability, oracle pair covariance | 18.56 ± 1.835 | 18.55 ± 1.728 |
| C | Ordinal + stability | 4.497 ± 0.4663 | 4.447 ± 0.3112 |
| C | Ordinal + levels + stability | 33.91 ± 2.946 | 33.59 ± 3.311 |
| C | Full loss + stability | 8.495 ± 0.817 | 8.422 ± 0.7571 |
| C | Gaussian ML + stability, independence working model | 22.62 ± 2.269 | 22.36 ± 2.148 |
| C | Gaussian ML + stability, oracle pair covariance | 26.37 ± 2.548 | 26.28 ± 2.48 |

### Conditional-mean outcome changes ΔW

Values in units of 10⁻⁶.

| Map | Scorer | Held-out seen cities | New cities after calibration |
| --- | --- | --- | --- |
| A | Ordinal + stability | 17.64 ± 0.9167 | 28.4 ± 0.9663 |
| A | Ordinal + levels + stability | 17.69 ± 0.959 | 28.43 ± 0.9887 |
| A | Full loss + stability | 95.05 ± 8.81 | 102.4 ± 8.394 |
| A | Gaussian ML + stability, independence working model | 26.88 ± 2.311 | 37.83 ± 2.043 |
| A | Gaussian ML + stability, oracle pair covariance | 31.01 ± 2.646 | 42.47 ± 2.618 |
| B | Ordinal + stability | 179 ± 236.9 | 305.6 ± 387 |
| B | Ordinal + levels + stability | 179 ± 236.8 | 305.7 ± 386.9 |
| B | Full loss + stability | 183.2 ± 236.7 | 310 ± 386.6 |
| B | Gaussian ML + stability, independence working model | 188.5 ± 242.1 | 318.1 ± 390.6 |
| B | Gaussian ML + stability, oracle pair covariance | 191.8 ± 243.2 | 321.5 ± 391.8 |
| C | Ordinal + stability | 1883 ± 3558 | 2242 ± 4437 |
| C | Ordinal + levels + stability | 1899 ± 3506 | 2306 ± 4548 |
| C | Full loss + stability | 1879 ± 3525 | 2263 ± 4496 |
| C | Gaussian ML + stability, independence working model | 1885 ± 3527 | 2284 ± 4541 |
| C | Gaussian ML + stability, oracle pair covariance | 1887 ± 3501 | 2299 ± 4594 |

[Full mean/SD table](run_5rep/five_rep_summary.csv) · [All city rows](run_5rep/city_metrics.csv) · [Replication summaries](run_5rep/per_rep_city_summary.csv)

## Same old weights, different error law

The following comparison uses the ordinal baseline, which already has identical coefficients in both designs. Features and city effects are matched; residual draws change. MSE units are 10⁻⁶.

| Map | Old f | Persistent f | Old Δf | Persistent Δf |
| --- | --- | --- | --- | --- |
| A | 21.545 | 41.726 | 6.726 | 3.798 |
| B | 23.136 | 52.921 | 7.487 | 4.185 |
| C | 36.256 | 68.599 | 7.413 | 4.447 |

The complete same-old-coefficient comparison also includes ordinal + levels and full loss. It uses unseen cities and the same cardinal calibration protocol. This isolates the specified loss coefficients; it does not hold estimated normalizers or fitted representations fixed. Five replications do not establish a universal effect of persistence.

[Same-weight figure](run_5rep/figures/fixed_weight_noise_comparison/equal_calibration_comparison.png) · [Comparison data](run_5rep/figures/fixed_weight_comparison_data.csv)

## Matched difference-loss experiment

Unseen structural-change MSE, in units of 10⁻⁶:

| Map | No W-change loss | All tracts | Changed tracts only | All tracts, lower weight |
| --- | --- | --- | --- | --- |
| A | 3.972 | 117.270 | 234.270 | 93.598 |
| B | 4.409 | 6.056 | 6.243 | 5.185 |
| C | 33.594 | 33.875 | 34.180 | 33.731 |

Adding the difference term raises mean structural-change MSE in each map among these matched arms. Changed-only is also higher than the lower-weight control in all three means. The effects in B/C are small at their selected coefficients. These findings concern the tested weights, not the existence of an optimal positive difference coefficient.

[Matched mean-error figure](run_5rep/figures/matched_comparison/equal_calibration_comparison.png) · [Matched city SD](run_5rep/figures/matched_comparison/city_variability.png) · [Paired differences, ranges and counts](run_5rep/paired_summary.csv)

## Small-change detection

Each score gets a cutoff from a separate independent stable panel, using the empirical 95th percentile of absolute score movement. Fine change doses were evaluated without retraining or tuning. The numbers below average power over positive/negative changes and five replications on unseen cities. Doses .002–.05 expose differences that were hidden by near-perfect detection at the older larger doses.

| Map | Scorer | Change .005 | Change .01 | Change .02 |
| --- | --- | --- | --- | --- |
| A | Ordinal + stability | 85.2% | 94.7% | 100.0% |
| A | Ordinal + levels + stability | 85.0% | 94.5% | 100.0% |
| A | Full loss + stability | 12.2% | 32.5% | 67.9% |
| A | Gaussian ML + stability, independence working model | 65.4% | 79.0% | 97.2% |
| A | Gaussian ML + stability, oracle pair covariance | 51.5% | 73.0% | 96.0% |
| B | Ordinal + stability | 83.7% | 94.6% | 100.0% |
| B | Ordinal + levels + stability | 83.1% | 94.4% | 100.0% |
| B | Full loss + stability | 53.0% | 87.0% | 99.9% |
| B | Gaussian ML + stability, independence working model | 68.1% | 81.5% | 93.4% |
| B | Gaussian ML + stability, oracle pair covariance | 55.9% | 76.5% | 93.7% |
| C | Ordinal + stability | 84.0% | 94.7% | 100.0% |
| C | Ordinal + levels + stability | 49.7% | 68.0% | 84.7% |
| C | Full loss + stability | 57.2% | 88.3% | 100.0% |
| C | Gaussian ML + stability, independence working model | 67.6% | 81.1% | 90.2% |
| C | Gaussian ML + stability, oracle pair covariance | 43.9% | 76.1% | 90.1% |

Fresh stable false-alarm rates are approximately 4.71–4.72%. The same linear acquisition-noise scale cancellation discussed in the original pilot applies. At change .005, ordinal power is about 84–85% in all three unseen-city cases. In C, the retuned level-only branch improves mean f-level MSE by about 45% relative to ordinal but lowers power at .005 to about 50%. The full C branch has a smaller level weight and about 57% power; this is not evidence that the difference term caused that gain. The oracle covariance likelihood does not dominate detection in these five runs; all likelihood comparisons with stability also depend on the fixed numerical stability weight.

[Detection figure](run_5rep/figures/small_change_detection.png) · [Detection data](run_5rep/detection.csv) · [Fresh false alarms](run_5rep/figures/detection_false_alarms.csv)

## Audit and reproduction

Completed 280 scorer fits plus 39 independent tuning candidates and 560 common calibration head sets. All 560 calibration checks preserved scores exactly. Largest calibration gradient was 2.59e-14; largest selected score-gradient norm was 1.62e-07. Masked-loss finite differences and correlated-likelihood finite differences passed. A separate 500,000-pair noise diagnostic reproduced endpoint variance 1, stable covariance .5 and difference variance 1, and changed covariance 0 and difference variance 2 after normalizing by city sigma.

The two-stage index mixed model converged in 15/15 fits. The diagonal sensitivity model converged in 9/15, with warnings retained; neither models the new tract correlation. Original model files remain unchanged. The source manifest records both new and reused implementations. The stronger invariance proposal remains a proposed estimator, not a result of this run.

[Audit](run_5rep/audit.json) · [Noise and likelihood preflight](run_5rep/preflight.json) · [Manifest](run_5rep/manifest.json) · [Optimizer diagnostics](run_5rep/optimizer.csv)

Run from this folder, choosing a fresh run directory:

```text
python run_persistent.py --out NEW_RUN
python plot_persistent.py --run NEW_RUN
python report_persistent.py --run NEW_RUN
```
