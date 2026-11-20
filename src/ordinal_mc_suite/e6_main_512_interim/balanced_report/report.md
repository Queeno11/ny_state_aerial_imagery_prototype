![True versus predicted latent change](latent_change.png)

# E6: outcome levels and within-tract changes

Completed 56/120 paired jobs. PARTIAL RUN: do not interpret this as the complete replicated experiment.

The figure shows direction-specific mean changes at each nonzero E5 step size. Error bars are 95% t intervals across independent replications. The dashed line is exact mean recovery. Individual-change RMSE is reported separately; good group means do not imply precise individual magnitudes. All latent scorers, including both privileged references, use the E5 fitting-pair dispersion normalization. `normalization_only_rmse` records its finite-sample effect even for the true structural function.

Selection uses fitting/validation observations only. A uses validation rank logistic loss; B and C use the fixed rank-plus-standardized-level validation rule, independent of training lambda. D receives exactly C’s executed additional scorer updates and restores its best rank-validation checkpoint; D_common instead restores its best checkpoint under C’s common rule using the frozen B head. D outcome predictions use separate post-hoc calibration without scorer updates.

The following direct paired comparisons use the largest city count, the quadratic/ReLU case when present, and step size 0.4 when present. Negative differences improve RMSE; positive differences improve sensitivity. Detection also requires inspecting its cutoff and fresh-stable false-positive rate in `summary.csv`. All other conditions and D_common comparisons are in `paired_differences.csv`.

| Measurement | Head | Target | Contrast | Difference [95% CI] |
|---|---|---|---|---|
| affine | affine | Latent magnitudes | C-B | 0.0011 [-0.0005, 0.0026] (n=14) |
| affine | spline | Latent magnitudes | C-B | 0.0013 [-0.0007, 0.0032] (n=14) |
| nonlinear | affine | Latent magnitudes | C-B | 0.0023 [0.0011, 0.0035] (n=14) |
| nonlinear | spline | Latent magnitudes | C-B | 0.0006 [-0.0012, 0.0024] (n=14) |
| affine | affine | Latent magnitudes | C-D | 0.0013 [0.0000, 0.0027] (n=14) |
| affine | spline | Latent magnitudes | C-D | 0.0016 [-0.0005, 0.0036] (n=14) |
| nonlinear | affine | Latent magnitudes | C-D | 0.0026 [0.0014, 0.0038] (n=14) |
| nonlinear | spline | Latent magnitudes | C-D | 0.0009 [-0.0005, 0.0023] (n=14) |
| affine | affine | Detection sensitivity | C-B | -0.0012 [-0.0021, -0.0003] (n=14) |
| affine | spline | Detection sensitivity | C-B | -0.0012 [-0.0024, 0.0001] (n=14) |
| nonlinear | affine | Detection sensitivity | C-B | -0.0017 [-0.0026, -0.0008] (n=14) |
| nonlinear | spline | Detection sensitivity | C-B | -0.0009 [-0.0016, -0.0002] (n=14) |
| affine | affine | Detection sensitivity | C-D | -0.0012 [-0.0020, -0.0003] (n=14) |
| affine | spline | Detection sensitivity | C-D | -0.0011 [-0.0023, 0.0001] (n=14) |
| nonlinear | affine | Detection sensitivity | C-D | -0.0016 [-0.0026, -0.0006] (n=14) |
| nonlinear | spline | Detection sensitivity | C-D | -0.0008 [-0.0016, -0.0001] (n=14) |
| affine | affine | Reported outcome changes | C-B | -0.0003 [-0.0011, 0.0004] (n=14) |
| affine | spline | Reported outcome changes | C-B | 0.0003 [-0.0017, 0.0022] (n=14) |
| nonlinear | affine | Reported outcome changes | C-B | 0.0001 [-0.0014, 0.0017] (n=14) |
| nonlinear | spline | Reported outcome changes | C-B | -0.0021 [-0.0057, 0.0014] (n=14) |
| affine | affine | Reported outcome changes | C-D | -0.0003 [-0.0010, 0.0004] (n=14) |
| affine | spline | Reported outcome changes | C-D | 0.0003 [-0.0015, 0.0022] (n=14) |
| nonlinear | affine | Reported outcome changes | C-D | 0.0002 [-0.0014, 0.0017] (n=14) |
| nonlinear | spline | Reported outcome changes | C-D | -0.0021 [-0.0056, 0.0014] (n=14) |
| affine | affine | Conditional-mean changes | C-B | -0.0034 [-0.0091, 0.0022] (n=14) |
| affine | spline | Conditional-mean changes | C-B | 0.0027 [-0.0134, 0.0188] (n=14) |
| nonlinear | affine | Conditional-mean changes | C-B | 0.0009 [-0.0139, 0.0157] (n=14) |
| nonlinear | spline | Conditional-mean changes | C-B | -0.0132 [-0.0394, 0.0129] (n=14) |
| affine | affine | Conditional-mean changes | C-D | -0.0033 [-0.0090, 0.0023] (n=14) |
| affine | spline | Conditional-mean changes | C-D | 0.0030 [-0.0132, 0.0191] (n=14) |
| nonlinear | affine | Conditional-mean changes | C-D | 0.0008 [-0.0140, 0.0156] (n=14) |
| nonlinear | spline | Conditional-mean changes | C-D | -0.0138 [-0.0401, 0.0125] (n=14) |

C retained its initial scorer/head checkpoint in 119/224 selected fits. Executed training updates and the retained checkpoint step are different quantities; inspect `validation_history.csv` for the optimization trajectory.

## Separate conclusions

**Latent magnitudes.**
- affine, affine: Configured replications are incomplete; no replicated conclusion yet.
- affine, spline: Configured replications are incomplete; no replicated conclusion yet.
- nonlinear, affine: Configured replications are incomplete; no replicated conclusion yet.
- nonlinear, spline: Configured replications are incomplete; no replicated conclusion yet.

**Detection sensitivity.**
- affine, affine: Configured replications are incomplete; no replicated conclusion yet.
- affine, spline: Configured replications are incomplete; no replicated conclusion yet.
- nonlinear, affine: Configured replications are incomplete; no replicated conclusion yet.
- nonlinear, spline: Configured replications are incomplete; no replicated conclusion yet.

**Reported outcome changes.**
- affine, affine: Configured replications are incomplete; no replicated conclusion yet.
- affine, spline: Configured replications are incomplete; no replicated conclusion yet.
- nonlinear, affine: Configured replications are incomplete; no replicated conclusion yet.
- nonlinear, spline: Configured replications are incomplete; no replicated conclusion yet.

Better reported outcome changes alone are successful outcome calibration. They do not establish improved latent cardinal recovery. Stable-tract RMS, cutoff and sensitivity by actual size distinguish stronger true-change response from lower nuisance movement. Cell-specific heads have no zero-shot unseen-city outcome predictions. Privileged ML knows the structural basis, mixing, measurement map and noise law; the noisy-R oracle instead uses the same scorer architecture and noisy inverse labels.

Intervals treat replications as sampling units, retain all finite completed seeds, and use no seed selection. They are pointwise, with no multiplicity adjustment. This report describes this finite design and budget; it is not an identification claim.