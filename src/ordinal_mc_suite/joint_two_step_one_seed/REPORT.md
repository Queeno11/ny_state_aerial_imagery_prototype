# Joint versus two-step ordinal pilot

Fixed-hyperparameter Monte Carlo pilot

Preset full144; 1 seed(s); 144 scorer stages planned; 32 configurations completed; elapsed 72.3s on cuda.

Known-noise parametric benchmark estimates structural coefficients and exact cell maps by paired Gaussian likelihood; the true-parameter oracle predicts f and E[W|H,cell]. Both are distinct from realized R and W.

NN latent metrics use a single fitting-only evaluation affine alignment per scorer. Target heads use separate local calibration. Thresholds use independent stable calibration panels; no final test labels select parameters.

See metrics.csv, paired_contrasts.csv, per-configuration JSON diagnostics and the three PNG figures. Short smoke optimizer budgets do not certify convergence. Parametric convergence fields must be checked before performance interpretation.

M3 generating map: Psi_ct(r)=a_ct+b_ct[r+k_ct tanh(r)], b_ct>0, k_ct>0; its conditional mean integrates Gaussian error with 32-point Gauss-Hermite quadrature.

Persistent eta has fixed marginal sigma. Under changing slopes or nonlinear maps it need not cancel in delta W.

## Target test summaries

These are descriptive means over available grid cells; matched contrasts are in the raw CSV. A single seed supplies no uncertainty interval.

| Arm | delta f RMSE | delta mu RMSE | Stable FPR | Small sensitivity | Large sensitivity |
|---|---:|---:|---:|---:|---:|
| A | 0.0932 | 0.5197 | 0.0420 | 0.8330 | 1.0000 |
| B | 0.0932 | 0.5317 | 0.0420 | 0.8330 | 1.0000 |
| C | 0.0850 | 0.5005 | 0.0615 | 0.8979 | 0.9995 |
| D | 0.0851 | 0.5250 | 0.0640 | 0.9028 | 0.9995 |
| O | 0.0823 | 0.4991 | 0.0625 | 0.9292 | 1.0000 |
| scratch_C | 0.1609 | 0.4646 | 0.0527 | 0.3848 | 0.9395 |
| scratch_D | 0.1484 | 0.3537 | 0.0557 | 0.4668 | 0.9727 |
| estimated_parametric | 0.0075 | 0.1375 | 0.0000 | 1.0000 | 1.0000 |
| true_parameter_oracle | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

Scratch arms cover only settings 1 and 8; their averages are not comparable to eight-setting averages without matching the settings.

Paired C-A, D-B and D-C contrasts are keyed by seed, setting, rank loss and link regime. With fewer than three seeds, no uncertainty interval is reported.
