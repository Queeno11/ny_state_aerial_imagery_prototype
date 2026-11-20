# Restricting outcome-change loss to changed tracts

Restricting the loss to truly changed tracts gave mixed results over five matched draws. Relative to the all-tract difference loss, it improved unseen structural-change MSE in four of five draws in maps A and C, but one large loss reversed the mean in A. Map B worsened in four of five draws. The changed-only arm had higher mean expected outcome-change MSE in all three maps on unseen cities.

The supplementary control used all tracts with a smaller difference coefficient, chosen to match the changed-only loss's total weight. That control had lower mean structural and expected outcome-change error than changed-only in every map, in both seen and unseen panels. The anchor with no outcome-change loss retained the lowest mean change errors among these four arms.

## Controlled comparison

All arms retain the original ordinal + level + stability scorer's frozen level weight. Rank weight is 1, stability weight is .3, and curvature weight is zero. The outcome head is affine in A and B and a monotone cubic in C. The three difference-loss variants use the original full arm's frozen difference coefficient. The original full arm had a level weight **100 times larger**, so it appears as a separate reference in the expanded figure.

| Map | Shared level weight | Difference coefficient |
|---|---:|---:|
| A | .0242062 | .232849 |
| B | .0201294 | .306905 |
| C | .0253447 | .428864 |

The original five fitting panels, labels, exact stability links, score normalization, spherical grid, multistart initialization and optimizer settings were reused. There was no coefficient tuning.

| Label in the clean figure | Difference-loss definition |
|---|---|
| No difference loss | Original ordinal + level + stability anchor |
| All tracts | Original difference coefficient, all fitting pairs |
| Changed only | Same coefficient per included pair, true structural change ≠ 0 |
| All tracts, lower weight | All fitting pairs, coefficient multiplied by the fitting changed fraction |

The changed-only mask uses the simulator's true structural change status. It is an oracle training diagnostic. It does not threshold the noisy observed outcome change. Both its conditional head fit and its scorer gradient use the masked criterion.

The masked difference loss is the average over **all fitting pairs** of the squared change residual multiplied by the binary mask, divided by the original all-pair change variance. Stable pairs contribute zero. The average is kept over all pairs, so dropping stable pairs also reduces the total difference-loss weight. The control multiplies the all-tract coefficient by the actual global fitting changed fraction, which ranged from .495625 to .509688. It matches the total scalar weight, while the mask also changes which pairs and cities receive it.

After fitting, all new scores were frozen and received fresh unpenalized level-only heads on the exact same 28 paired calibration tracts per city as the prior comparison. Their remaining 252 test pairs were also identical. Both dates stay together, and seen and unseen city heads use only their own calibration labels. The original native-head and 10% results are retained. The supplementary control was a separate 15-fit stage after the 30 primary fits.

## Held-out results

The tables average equal-city means over the five draws. Structural entries are in units of 10⁻⁶. Each city's Δf MSE includes stable and changing tracts.

| Map | Loss arm | Seen f MSE | Unseen f MSE | Seen Δf MSE | Unseen Δf MSE |
|---|---|---:|---:|---:|---:|
| A | No difference loss | 20.98 | 21.53 | 6.767 | 7.050 |
| A | All tracts | 33.38 | 33.97 | 25.22 | 25.58 |
| A | Changed only | 34.46 | 35.17 | 27.91 | 28.77 |
| A | All tracts, lower weight | 28.24 | 28.83 | 16.89 | 17.25 |
| B | No difference loss | 21.97 | 22.56 | 7.453 | 7.776 |
| B | All tracts | 42.01 | 42.53 | 42.47 | 42.97 |
| B | Changed only | 49.91 | 50.52 | 58.81 | 59.82 |
| B | All tracts, lower weight | 33.15 | 33.71 | 26.24 | 26.69 |
| C | No difference loss | 32.42 | 33.19 | 7.421 | 7.746 |
| C | All tracts | 134.9 | 137.5 | 96.55 | 98.40 |
| C | Changed only | 127.1 | 129.6 | 85.85 | 87.46 |
| C | All tracts, lower weight | 83.22 | 84.72 | 60.27 | 61.55 |

The outcome-level entries below are in units of 10⁻³, and outcome-change entries are in units of 10⁻⁶. The targets are the exact conditional means.

| Map | Loss arm | Seen W MSE | Unseen W MSE | Seen ΔW MSE | Unseen ΔW MSE |
|---|---|---:|---:|---:|---:|
| A | No difference loss | 4.356 | 3.250 | 25.45 | 16.64 |
| A | All tracts | 4.375 | 3.275 | 44.23 | 35.48 |
| A | Changed only | 4.368 | 3.272 | 46.58 | 38.36 |
| A | All tracts, lower weight | 4.368 | 3.266 | 35.75 | 27.00 |
| B | No difference loss | 8.126 | 6.497 | 184.3 | 150.2 |
| B | All tracts | 8.154 | 6.528 | 217.4 | 186.1 |
| B | Changed only | 8.159 | 6.538 | 234.4 | 207.4 |
| B | All tracts, lower weight | 8.142 | 6.516 | 202.1 | 169.2 |
| C | No difference loss | 14.44 | 17.20 | 1,089 | 1,521 |
| C | All tracts | 14.51 | 17.33 | 1,175 | 1,598 |
| C | Changed only | 14.45 | 17.33 | 1,153 | 1,600 |
| C | All tracts, lower weight | 14.48 | 17.25 | 1,141 | 1,562 |

The mean differences across cities can be much smaller than city dispersion. The figure of city variability and the data retain that dispersion separately from the range across replications. No-difference loss has the smallest mean Δf and ΔW errors in all six map–city-panel combinations among these four arms. Some level-prediction comparisons differ, particularly the C seen-city W level error.

## Paired differences

The following differences subtract the named all-tract comparator from changed-only. Negative values favor changed-only. Values are in units of 10⁻⁶, with ranges over five draws. Wins count the draws with lower error.

| Map | Comparator | Unseen Δf MSE difference, mean [range] | Wins | Unseen ΔW MSE difference, mean [range] | Wins |
|---|---|---:|---:|---:|---:|
| A | All tracts | +3.20 [−14.66, +54.61] | 4/5 | +2.88 [−15.21, +53.52] | 4/5 |
| B | All tracts | +16.86 [−14.03, +57.85] | 1/5 | +21.29 [−12.46, +74.88] | 1/5 |
| C | All tracts | −10.94 [−43.88, +49.25] | 4/5 | +1.91 [−25.41, +53.00] | 4/5 |
| A | All tracts, lower weight | +11.52 [−2.81, +53.37] | 2/5 | +11.37 [−2.85, +52.44] | 2/5 |
| B | All tracts, lower weight | +33.14 [−2.54, +69.23] | 1/5 | +38.12 [−2.09, +77.49] | 1/5 |
| C | All tracts, lower weight | +25.92 [−19.74, +112.95] | 3/5 | +38.00 [−34.37, +129.16] | 1/5 |

Thus four wins in five draws do not imply a lower mean. Draw 1 reversed the mean in A. In C, changed-only had lower structural-change MSE than the full-weight all-tract loss, but its mean unseen expected outcome-change error was slightly higher because of draw 4. Compared with the lower-weight all-tract control, changed-only still improved C structural changes in three draws, while the two losses were larger. These results give no consistent advantage for selectively dropping stable pairs over reducing the overall difference weight at this setting.

The comparison is bounded to these five samples and the frozen coefficients. It also gives the changed-only method privileged change-status information. It cannot establish that differences are always unhelpful under a different DGP or regularization choice.

## Figures and machine-readable results

The clean four-arm figure holds the level coefficient fixed. The expanded overview also shows the original full-loss arm under its distinct, larger level weight.

- [Matched mean-error figure](changed_only_delta_mass_control/figures/matched_comparison/equal_calibration_comparison.png) and [PDF](changed_only_delta_mass_control/figures/matched_comparison/equal_calibration_comparison.pdf)
- [Matched city-dispersion figure](changed_only_delta_mass_control/figures/matched_comparison/city_variability.png)
- [Expanded original-reference figure](changed_only_delta_mass_control/figures/all_methods/equal_calibration_comparison.png)
- [Combined five-replication summary](changed_only_delta_mass_control/five_rep_summary.csv), [per-replication city summaries](changed_only_delta_mass_control/per_rep_city_summary.csv), and [per-city observations](changed_only_delta_mass_control/city_metrics.csv)
- [Paired per-replication differences](changed_only_delta_mass_control/paired_by_rep.csv) and [means, ranges and counts](changed_only_delta_mass_control/paired_summary.csv)

City SD is the sample SD of the 20 city MSEs within a replication, averaged over five replications. It is separate from a standard error or the five-replication range. The combined data retain realized noisy W and ΔW, stable and changed Δf, and original scorer-fitting f metrics.

## Audit and reproduction

The primary stage completed 30 scorer fits in 16.8 seconds, and the supplementary mass control completed 15 in 10.0 seconds. Conditional heads were solved for their actual masked or unmasked objectives. The all-one mask reproduced the original unmasked head bitwise and its objective and gradient numerically. The zero mask reproduced the level-only head and zero difference gradient. Stable rows had exactly zero masked difference contribution. Directional finite differences, including C's cubic head optimization, agreed within 3.39e−10 in the primary stage.

Across all 45 fits, the largest selected tangent gradient was 4.78e−8. Four starts reported success in 38 fits and three in seven. The final objective spread across starts was below 8.4e−15. The conditional head projected gradient was below 6.3e−16. All 90 post-fit calibration checks preserved the new f and Δf predictions bitwise, and their calibration/test input hashes matched the existing 10% comparison. The original score and head files were preserved. [Primary audit](changed_only_delta/audit.json) · [Mass-control audit](changed_only_delta_mass_control/audit.json) · [Primary manifest](changed_only_delta/manifest.json) · [Control manifest](changed_only_delta_mass_control/manifest.json)

From the suite directory, choose fresh output paths.

```text
python -B simple_index_pilot/changed_only_delta.py --run simple_index_pilot/run_5rep --calibration simple_index_pilot/calibration_10pct --out NEW/changed
python -B simple_index_pilot/changed_delta_mass_control.py --run simple_index_pilot/run_5rep --calibration simple_index_pilot/calibration_10pct --changed NEW/changed --out NEW/mass_control
```
