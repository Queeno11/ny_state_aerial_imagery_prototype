# E6 smoke review

Both eight-city jobs completed. Each uses 208 fitting and 48 validation tracts,
416 exact temporal links with weight 0.3, and the same splits/panels across arms.
All recorded implementation checks passed: frozen-score equivalence, matched
C/D updates, nonzero level gradients, monotone heads, disjoint tract splits,
exact links and no test-based selection. ML converged for both structural truths
and both measurement maps, with reported gradient infinity norm below 1e-7.

The original renderer incorrectly used `DataFrame.where` as a column, silently
filtering out the plotted data and summary table rows. That reporting defect is
fixed and covered by three reporting regressions, including explicit populated
curve/table checks. This review directory rebuilds reports from the completed
SQLite jobs. Its provenance file verifies that the original fitting ledger was
unchanged. No models were fitted or rerun during this review.

## Joint training and stopping

All eight selected C fits retained step zero and are exactly equivalent to their
selected B initialization. Each nevertheless executed 200 joint updates, with
nonzero initial level gradients and decreasing fitting objective. Validation
favored the starting checkpoint, so best-checkpoint restoration removed those
updates. Three of 24 candidates had their best checkpoint at step ten, but their
validation criterion did not beat the ultimately selected candidates. All eight
D fits received the matched 200 scorer updates. In the quadratic case, D also
restored A. These equalities are not evidence that gradients failed to flow.

## Latent changes and detection

The table uses unseen cities and nominal change step 0.4. A, B and selected C
have identical latent metrics under both measurement truths and both heads.

| Structural truth | Changing-tract RMSE | Sign accuracy | Stable RMS | Detection cutoff | Sensitivity | Fresh stable FPR |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 0.0352 | 100% | 0.0348 | 0.0701 | 100% | 3.9% |
| Quadratic/ReLU | 0.2872 | 96.1% | 0.1995 | 0.3973 | 45.3% | 5.9% |

The linear implementation check performs well. The quadratic case combines
attenuation/noisy individual magnitudes with nuisance movement among unchanged
tracts. At step 0.4, true mean changes are -0.448 and +0.477, while predicted
means are -0.374 and +0.378. Thus the direction is usually correct but average
magnitudes are attenuated, and individual RMSE remains substantial. Stable
movement raises the cutoff to about 0.40, limiting detection even when predicted
signs are correct. Quadratic sensitivity is only 11.8% at nominal step 0.15 and
rises to 78.1% at step 0.8. At the smaller step, sign accuracy also drops to
78.3%, so poor small-change detection is not exclusively a threshold issue.

Privileged ML has changing-tract RMSE about 0.011 for the quadratic case and
negligible stable movement. The noisy-R oracle using the same ReLU architecture
does not improve over A in this smoke sample (RMSE about 0.327). It is a noisy
label/capacity/optimization diagnostic, not a guaranteed performance ceiling.

## Outcome calibration

Calibration predicts observed changes better than the zero-change benchmark,
but all gains in these selected fits are already obtained by B. For example,
under quadratic truth and affine measurement, the affine head's observed-change
RMSE is 2.953 versus 3.125 for zero change; against the noise-integrated mean
change, its RMSE is 0.942 versus 1.327 for zero change. C equals B on both targets.
Under nonlinear measurement, the spline head's corresponding errors are 3.899
versus 4.029 and 1.274 versus 1.533. This demonstrates useful outcome prediction
in the smoke sample, without incremental evidence for joint latent recovery.

The corrected figure shows mean changes, so it can look good despite noisy
individual magnitudes; read it together with changing-tract RMSE and stable RMS.
Ordinal and joint curves overlap exactly because C retained its initialization.
There is one replication per structural case, so no across-replication confidence
intervals or replicated scientific conclusions are available. The smoke run
does not establish that level supervision cannot help at 32/64 cities. The main
design can proceed with the predeclared training/selection settings and corrected
renderer; no settings were tuned using these test results.
