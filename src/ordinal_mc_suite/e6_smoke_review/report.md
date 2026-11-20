![True versus predicted latent change](latent_change.png)

# E6: outcome levels and within-tract changes

Completed 2/2 paired jobs. Full configured design.

The figure shows direction-specific mean changes at each nonzero E5 step size. This is a single-replication smoke run; no across-replication intervals are available. The dashed line is exact mean recovery. Individual-change RMSE is reported separately; good group means do not imply precise individual magnitudes. All latent scorers, including both privileged references, use the E5 fitting-pair dispersion normalization. `normalization_only_rmse` records its finite-sample effect even for the true structural function.

Selection uses fitting/validation observations only. A uses validation rank logistic loss; B and C use the fixed rank-plus-standardized-level validation rule, independent of training lambda. D receives exactly C’s executed additional scorer updates and restores its best rank-validation checkpoint; D_common instead restores its best checkpoint under C’s common rule using the frozen B head. D outcome predictions use separate post-hoc calibration without scorer updates.

The following direct paired comparisons use the largest city count, the quadratic/ReLU case when present, and step size 0.4 when present. Negative differences improve RMSE; positive differences improve sensitivity. Detection also requires inspecting its cutoff and fresh-stable false-positive rate in `summary.csv`. All other conditions and D_common comparisons are in `paired_differences.csv`.

| Measurement | Head | Target | Contrast | Difference [95% CI] |
|---|---|---|---|---|
| affine | affine | Latent magnitudes | C-B | 0.0000 (n=1; CI unavailable) |
| affine | spline | Latent magnitudes | C-B | 0.0000 (n=1; CI unavailable) |
| nonlinear | affine | Latent magnitudes | C-B | 0.0000 (n=1; CI unavailable) |
| nonlinear | spline | Latent magnitudes | C-B | 0.0000 (n=1; CI unavailable) |
| affine | affine | Latent magnitudes | C-D | 0.0000 (n=1; CI unavailable) |
| affine | spline | Latent magnitudes | C-D | 0.0000 (n=1; CI unavailable) |
| nonlinear | affine | Latent magnitudes | C-D | 0.0000 (n=1; CI unavailable) |
| nonlinear | spline | Latent magnitudes | C-D | 0.0000 (n=1; CI unavailable) |
| affine | affine | Detection sensitivity | C-B | 0.0000 (n=1; CI unavailable) |
| affine | spline | Detection sensitivity | C-B | 0.0000 (n=1; CI unavailable) |
| nonlinear | affine | Detection sensitivity | C-B | 0.0000 (n=1; CI unavailable) |
| nonlinear | spline | Detection sensitivity | C-B | 0.0000 (n=1; CI unavailable) |
| affine | affine | Detection sensitivity | C-D | 0.0000 (n=1; CI unavailable) |
| affine | spline | Detection sensitivity | C-D | 0.0000 (n=1; CI unavailable) |
| nonlinear | affine | Detection sensitivity | C-D | 0.0000 (n=1; CI unavailable) |
| nonlinear | spline | Detection sensitivity | C-D | 0.0000 (n=1; CI unavailable) |
| affine | affine | Reported outcome changes | C-B | 0.0000 (n=1; CI unavailable) |
| affine | spline | Reported outcome changes | C-B | 0.0000 (n=1; CI unavailable) |
| nonlinear | affine | Reported outcome changes | C-B | 0.0000 (n=1; CI unavailable) |
| nonlinear | spline | Reported outcome changes | C-B | 0.0000 (n=1; CI unavailable) |
| affine | affine | Reported outcome changes | C-D | 0.0000 (n=1; CI unavailable) |
| affine | spline | Reported outcome changes | C-D | 0.0000 (n=1; CI unavailable) |
| nonlinear | affine | Reported outcome changes | C-D | 0.0000 (n=1; CI unavailable) |
| nonlinear | spline | Reported outcome changes | C-D | 0.0000 (n=1; CI unavailable) |
| affine | affine | Conditional-mean changes | C-B | 0.0000 (n=1; CI unavailable) |
| affine | spline | Conditional-mean changes | C-B | 0.0000 (n=1; CI unavailable) |
| nonlinear | affine | Conditional-mean changes | C-B | 0.0000 (n=1; CI unavailable) |
| nonlinear | spline | Conditional-mean changes | C-B | 0.0000 (n=1; CI unavailable) |
| affine | affine | Conditional-mean changes | C-D | 0.0000 (n=1; CI unavailable) |
| affine | spline | Conditional-mean changes | C-D | 0.0000 (n=1; CI unavailable) |
| nonlinear | affine | Conditional-mean changes | C-D | 0.0000 (n=1; CI unavailable) |
| nonlinear | spline | Conditional-mean changes | C-D | 0.0000 (n=1; CI unavailable) |

C retained its initial scorer/head checkpoint in 8/8 selected fits. Executed training updates and the retained checkpoint step are different quantities; inspect `validation_history.csv` for the optimization trajectory.

## Separate conclusions

**Latent magnitudes.**
- affine, affine: Single-replication smoke result; no replicated conclusion or across-replication interval is available.
- affine, spline: Single-replication smoke result; no replicated conclusion or across-replication interval is available.
- nonlinear, affine: Single-replication smoke result; no replicated conclusion or across-replication interval is available.
- nonlinear, spline: Single-replication smoke result; no replicated conclusion or across-replication interval is available.

**Detection sensitivity.**
- affine, affine: Single-replication smoke result; no replicated conclusion or across-replication interval is available.
- affine, spline: Single-replication smoke result; no replicated conclusion or across-replication interval is available.
- nonlinear, affine: Single-replication smoke result; no replicated conclusion or across-replication interval is available.
- nonlinear, spline: Single-replication smoke result; no replicated conclusion or across-replication interval is available.

**Reported outcome changes.**
- affine, affine: Single-replication smoke result; no replicated conclusion or across-replication interval is available.
- affine, spline: Single-replication smoke result; no replicated conclusion or across-replication interval is available.
- nonlinear, affine: Single-replication smoke result; no replicated conclusion or across-replication interval is available.
- nonlinear, spline: Single-replication smoke result; no replicated conclusion or across-replication interval is available.

Better reported outcome changes alone are successful outcome calibration. They do not establish improved latent cardinal recovery. Stable-tract RMS, cutoff and sensitivity by actual size distinguish stronger true-change response from lower nuisance movement. Cell-specific heads have no zero-shot unseen-city outcome predictions. Privileged ML knows the structural basis, mixing, measurement map and noise law; the noisy-R oracle instead uses the same scorer architecture and noisy inverse labels.

Intervals treat replications as sampling units, retain all finite completed seeds, and use no seed selection. They are pointwise, with no multiplicity adjustment. This report describes this finite design and budget; it is not an identification claim.