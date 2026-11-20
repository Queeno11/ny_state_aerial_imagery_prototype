# Joint versus two-step ordinal pilot

**IMPLEMENTATION CHECK / NOT SCIENTIFIC EVIDENCE**

Preset full144; 1 seed(s); 144 scorer stages planned; 32 configurations completed; elapsed 15.8s on cuda.

Known-noise parametric benchmark estimates structural coefficients and exact cell maps by paired Gaussian likelihood; the true-parameter oracle predicts f and E[W|H,cell]. Both are distinct from realized R and W.

NN latent metrics use a single fitting-only evaluation affine alignment per scorer. Target heads use separate local calibration. Thresholds use independent stable calibration panels; no final test labels select parameters.

See metrics.csv, paired_contrasts.csv, per-configuration JSON diagnostics and the three PNG figures. Short smoke optimizer budgets do not certify convergence. Parametric convergence fields must be checked before performance interpretation.

M3 generating map: Psi_ct(r)=a_ct+b_ct[r+k_ct tanh(r)], b_ct>0, k_ct>0; its conditional mean integrates Gaussian error with 32-point Gauss-Hermite quadrature.

Persistent eta has fixed marginal sigma. Under changing slopes or nonlinear maps it need not cancel in delta W.

## Target test summaries

These are descriptive means over available grid cells; matched contrasts are in the raw CSV. A single seed supplies no uncertainty interval.

| Arm | delta f RMSE | delta mu RMSE | Stable FPR | Small sensitivity | Large sensitivity |
|---|---:|---:|---:|---:|---:|
| A | 0.3745 | 1.8079 | 0.0000 | 0.0781 | 0.1641 |
| B | 0.3745 | 1.7505 | 0.0000 | 0.0781 | 0.1641 |
| C | 0.3683 | 2.6791 | 0.0000 | 0.0781 | 0.2227 |
| D | 0.3686 | 2.1379 | 0.0000 | 0.0781 | 0.2539 |
| O | 0.3673 | 1.8650 | 0.0000 | 0.0781 | 0.2266 |
| scratch_C | 0.4547 | 1.5046 | 0.0156 | 0.1250 | 0.1250 |
| scratch_D | 0.4479 | 1.2672 | 0.0156 | 0.1250 | 0.1250 |
| estimated_parametric | 0.0512 | 0.8425 | 0.0000 | 1.0000 | 1.0000 |
| true_parameter_oracle | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

Scratch arms cover only settings 1 and 8; their averages are not comparable to eight-setting averages without matching the settings.

Paired C-A, D-B and D-C contrasts are keyed by seed, setting, rank loss and link regime. With fewer than three seeds, no uncertainty interval is reported.
