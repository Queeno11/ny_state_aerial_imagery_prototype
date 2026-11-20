# Fair outcome calibration on frozen scores

The subsequent [changed-only difference-loss comparison](ANALYSIS_CHANGED_ONLY.md) holds the level weight fixed, tests the oracle changed mask, and includes an all-tract control with matching total difference-loss weight.

The same 10% calibration protocol gives the ordinal, joint, and classical methods comparable outcome predictions. After this calibration, adding level loss to the ordinal scorer barely changed expected outcome-change MSE. The full ordinal level-and-change package had higher mean expected outcome-change error than ordinal plus levels in every map and city panel. In the cubic map, the difference between fitting the head on the ordinal score and fitting it on the true score was negligible at this calibration sample size.

[Mean-error figure](calibration_10pct/figures/equal_calibration_comparison.png) · [City-SD figure](calibration_10pct/figures/city_variability.png) · [Mean-error PDF](calibration_10pct/figures/equal_calibration_comparison.pdf)

## What was fitted

The five saved linear-index replications supplied 175 scorers. Each scorer's original global unit-normalized coefficient vector was frozen. For each replication, the seen-city panel uses the original 20 cities with new tracts, and the unseen-city panel uses 20 independently drawn cities and new tracts. Both panels have 280 paired tracts per city. That count preserves the DGP's balanced stable and change categories and admits an exact 10% split.

A random, outcome-independent split selected 28 paired tracts per city for calibration, with 28 observed outcomes at each date. The other 252 pairs per city supplied evaluation. Each city panel in a replication therefore has 560 calibration pairs and 5,040 evaluation pairs. All methods use the same tracts and observed labels within a replication and map. The three maps also share the image draws, underlying Gaussian errors and split. The seen and unseen panels use fresh seeds, separate from the original scorer-fitting, validation and test draws.

All heads were fitted from scratch with unpenalized least squares on the observed calibration outcomes. A uses a common positive slope and city intercepts. B uses positive affine slopes and intercepts by city. C uses a city-specific monotone cubic with positive linear and nonnegative cubic coefficients. The city map is shared across dates. The fitter receives the frozen score and the calibration labels, with no access to true city coefficients or evaluation outcomes. Outcome change is the difference between the two fitted outcome predictions. In unseen cities these are **predictions after labeled local calibration**, so they answer a different question from zero-shot outcome prediction.

The earlier native and equal-library head results remain available in the original run. This comparison gives every scorer the same head-fitting data, family and criterion, so a difference in outcome error reflects the frozen score and how the common calibration learns from it. It does not evaluate the original heads or retain their different fitting advantages. [Original pilot](ANALYSIS.md) · [Recalibration manifest](calibration_10pct/manifest.json)

## Four primary comparisons

Each mean gives every city equal weight. Within each replication we compute the mean and sample SD of the 20 city MSEs. The table entries then average those means and SDs over the five replications. City SD describes variation across cities and is separate from replication ranges or standard errors.

The rows below show the main contrasts. The linked tables include all 35 map–method combinations and use unscaled values. All four abbreviated method labels include the same stability links. **Levels** and **Levels and changes** refer to the original scorer training losses. Every outcome head in these tables uses the new level-only calibration protocol.

### Structural score f MSE

Values in units of 10⁻⁶.

| Map | Scorer training | Seen mean | Seen city SD | Unseen mean | Unseen city SD |
|---|---|---:|---:|---:|---:|
| A | Ordinal | 20.99 | 1.976 | 21.54 | 1.939 |
| A | Ordinal + levels | 20.98 | 1.969 | 21.53 | 1.930 |
| A | Ordinal + levels and changes | 33.96 | 2.694 | 34.54 | 2.787 |
| A | Gaussian ML | 22.81 | 1.882 | 23.39 | 1.747 |
| B | Ordinal | 22.53 | 2.156 | 23.14 | 2.058 |
| B | Ordinal + levels | 21.97 | 2.087 | 22.56 | 2.003 |
| B | Ordinal + levels and changes | 32.10 | 2.033 | 32.40 | 2.395 |
| B | Gaussian ML | 16.70 | 1.337 | 17.24 | 1.408 |
| C | Ordinal | 35.40 | 3.557 | 36.26 | 3.330 |
| C | Ordinal + levels | 32.42 | 3.222 | 33.19 | 3.036 |
| C | Ordinal + levels and changes | 42.49 | 2.360 | 42.91 | 2.941 |
| C | Gaussian ML | 16.58 | 1.049 | 17.08 | 1.347 |

[All methods for f](calibration_10pct/table_f.csv)

### Conditional-mean outcome W MSE

Values in units of 10⁻³.

| Map | Scorer training | Seen mean | Seen city SD | Unseen mean | Unseen city SD |
|---|---|---:|---:|---:|---:|
| A | Ordinal | 4.357 | 6.536 | 3.250 | 4.481 |
| A | Ordinal + levels | 4.356 | 6.536 | 3.250 | 4.482 |
| A | Ordinal + levels and changes | 4.369 | 6.544 | 3.280 | 4.522 |
| A | Gaussian ML | 4.353 | 6.535 | 3.250 | 4.484 |
| B | Ordinal | 8.127 | 9.399 | 6.497 | 6.576 |
| B | Ordinal + levels | 8.126 | 9.398 | 6.497 | 6.577 |
| B | Ordinal + levels and changes | 8.128 | 9.397 | 6.522 | 6.613 |
| B | Gaussian ML | 8.112 | 9.374 | 6.490 | 6.591 |
| C | Ordinal | 14.45 | 15.72 | 17.20 | 25.53 |
| C | Ordinal + levels | 14.44 | 15.71 | 17.20 | 25.55 |
| C | Ordinal + levels and changes | 14.42 | 15.58 | 17.27 | 25.87 |
| C | Gaussian ML | 14.39 | 15.68 | 17.26 | 25.84 |

[All methods for conditional-mean W](calibration_10pct/table_W_conditional_mean.csv)

### Structural change Δf MSE

Values in units of 10⁻⁶. This table includes stable and changing tracts.

| Map | Scorer training | Seen mean | Seen city SD | Unseen mean | Unseen city SD |
|---|---|---:|---:|---:|---:|
| A | Ordinal | 6.455 | 0.511 | 6.726 | 0.538 |
| A | Ordinal + levels | 6.767 | 0.536 | 7.050 | 0.561 |
| A | Ordinal + levels and changes | 33.16 | 2.974 | 33.89 | 2.512 |
| A | Gaussian ML | 15.35 | 1.248 | 15.53 | 1.362 |
| B | Ordinal | 7.174 | 0.563 | 7.487 | 0.597 |
| B | Ordinal + levels | 7.453 | 0.586 | 7.776 | 0.618 |
| B | Ordinal + levels and changes | 45.15 | 3.803 | 45.85 | 3.650 |
| B | Gaussian ML | 10.88 | 0.911 | 11.27 | 0.876 |
| C | Ordinal | 7.100 | 0.555 | 7.413 | 0.585 |
| C | Ordinal + levels | 7.421 | 0.583 | 7.746 | 0.608 |
| C | Ordinal + levels and changes | 68.29 | 5.458 | 69.90 | 5.705 |
| C | Gaussian ML | 16.39 | 1.438 | 16.89 | 1.375 |

[All methods for Δf](calibration_10pct/table_delta_f.csv) · [Stable tracts](calibration_10pct/table_delta_f_stable.csv) · [Changing tracts](calibration_10pct/table_delta_f_changed.csv)

### Conditional-mean outcome change ΔW MSE

Values in units of 10⁻⁶.

| Map | Scorer training | Seen mean | Seen city SD | Unseen mean | Unseen city SD |
|---|---|---:|---:|---:|---:|
| A | Ordinal | 25.16 | 1.115 | 16.30 | 0.945 |
| A | Ordinal + levels | 25.45 | 1.145 | 16.64 | 0.970 |
| A | Ordinal + levels and changes | 51.86 | 4.040 | 44.00 | 3.199 |
| A | Gaussian ML | 32.28 | 2.240 | 24.37 | 1.847 |
| B | Ordinal | 184.1 | 266.6 | 149.9 | 196.3 |
| B | Ordinal + levels | 184.3 | 266.6 | 150.2 | 196.3 |
| B | Ordinal + levels and changes | 220.4 | 276.2 | 187.8 | 199.3 |
| B | Gaussian ML | 185.9 | 265.3 | 150.7 | 193.3 |
| C | Ordinal | 1,089 | 1,810 | 1,520 | 3,114 |
| C | Ordinal + levels | 1,089 | 1,811 | 1,521 | 3,117 |
| C | Ordinal + levels and changes | 1,153 | 1,844 | 1,582 | 3,129 |
| C | Gaussian ML | 1,100 | 1,828 | 1,524 | 3,115 |

[All methods for conditional-mean ΔW](calibration_10pct/table_delta_W_conditional_mean.csv)

## What the matched comparisons show

- **Ordinal versus ordinal plus levels.** The mean difference in expected change MSE was less than 0.64 × 10⁻⁶ in every map and panel, with the ordinal-only score slightly lower in all six means. The mean Δf MSE was also slightly lower for ordinal alone. For f levels in B and C, joint level training improved the mean relative to ordinal alone, and Gaussian ML plus stability had the smaller f MSE among these three scores. Lower f-level error and lower paired structural-change error can favor different methods.
- **Full ordinal package versus ordinal plus levels.** The full package raised expected change MSE in four of five draws in each map and panel. Mean increases, in units of 10⁻⁶, were 26.4 seen and 27.4 unseen in A, 36.0 and 37.6 in B, and 63.5 and 61.0 in C. This compares two independently tuned packages. Their original level weights differ as well as their change weights.
- **Adding changes at fixed level weight.** L2 plus differences and stability had higher mean expected change MSE than L2 plus stability in every map and panel. It lost on 4/5 seen and 5/5 unseen draws in A, 3/5 and 4/5 in B, and 5/5 and 3/5 in C. This contrast holds the original level weight fixed. The differences loss uses the same paired noisy endpoints and reweights their residual contrast.
- **Cubic calibration is noisy at this sample size.** For ordinal in unseen C cities, expected change MSE averaged .001520, while the average within-rep city median was .000479 and the average within-rep city maximum .012976. The same calibration using the true structural score gave .001525. This diagnostic estimates the head from the same noisy labels and shares the same test tracts, with the true score supplied solely to show the scale of calibration error. It is separate from the feasible comparisons and does not give a lower bound. No fitted linear coefficient hit the positivity floor. The cubic coefficient was zero in 78 of the 1,300 unseen C city–method fits, corresponding to six underlying city–replication calibrations across the 13 methods. None was zero in seen C cities. These diagnostics show uneven city error and head estimation variation under limited calibration labels.

The realized outcomes add Gaussian noise. For example, in unseen C cities the ordinal model's MSE was .2029 for realized W and .3740 for realized ΔW, compared with .01720 and .001520 against the conditional targets. The structural fitting-observation supplement reports f and Δf on the original scorer-fitting sample. The new outcome heads were fitted on external calibration tracts, so the original-fit supplement is structural only. [Paired contrasts and replication ranges](calibration_10pct/paired_summary.csv) · [Realized W](calibration_10pct/table_W_realized.csv) · [Realized ΔW](calibration_10pct/table_delta_W_realized.csv) · [Original fitting observations](calibration_10pct/original_fit_structural.csv) · [True-score calibration diagnostic](calibration_10pct/true_f_calibration_summary.csv) · [Head diagnostics](calibration_10pct/head_diagnostics.csv)

## Audit and files

The run fitted 350 feasible head sets and 30 separate true-score diagnostic head sets, with zero new scorer fits. All saved normalized coefficient vectors matched the original common-library vectors exactly. Predicted f and Δf were unchanged bit for bit in all 350 calibration comparisons. The audit checked the exact 10% tract split, disjoint calibration and evaluation IDs within panels, independent panel seeds relative to scorer fitting, matching calibration labels and features across methods, and correct city counts and aggregation. The largest projected head gradient was 3.59e−14.

The per-city file includes the MSE, number of endpoints or pairs, and mean predictions and targets for each metric. The per-replication file records city mean, SD, median and maximum. The five-replication file retains the range of the five city means separately from the city SD. Fitted head coefficients and split fingerprints are saved for reproduction. These are descriptive comparisons across five replications. [Audit](calibration_10pct/audit.json) · [Per-city data](calibration_10pct/city_metrics.csv) · [Per-replication summaries](calibration_10pct/per_rep_city_summary.csv) · [Five-replication summaries](calibration_10pct/five_rep_summary.csv) · [Fitted heads](calibration_10pct/calibrated_heads.json)

Run from the suite directory with the existing Windows Python environment and a fresh destination.

```text
python -B simple_index_pilot/calibrate_10pct.py --run simple_index_pilot/run_5rep --out NEW/calibration_10pct
```
