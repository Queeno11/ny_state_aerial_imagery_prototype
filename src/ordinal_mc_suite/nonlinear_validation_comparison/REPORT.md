# Two nonlinear specifications: one replication each

The same validation-selected stopping comparison was run on exactly one dataset
per specification, with three random initializations per NN arm as in the prior
diagnostic. This is not a full Monte Carlo run.

## Specifications

1. **Exponential measurement with an additive baseline:**
   W = a_ct + b_ct exp(0.6 R), with linear latent f in R = f(H) + epsilon.
   The existing additive-baseline scale is 10:
   a_ct = 10[0.6 + 0.3 tanh(z_c1) + 0.1 t], and
   b_ct = exp[0.5 tanh(z_c2) + 0.1 t].
   This breaks the exact log-linear representation with cell intercepts.
2. **Quadratic latent truth:**
   f(H) is the normalized version of u1 + 0.7 u2 u3 + 0.4(u3^2 - 1),
   using the suite's fixed rotation. Observations retain the usual affine
   measurement W = a_ct + b_ct R.

Both use root seed 20261102, replication 1, 16 cities, 32 labels per city, clean
links, feature-change step 0.4, and the original [32, 16] NN architecture.
The 20%-target city/date-stratified split yields 416 fitting and 96 validation
tracts. All comparisons within a case share the same split and test panels.

## Stopping and selection

Observed within-cell validation BCE selects checkpoints and random starts.
Adam checks every 10 steps, with patience 20 checks and min_delta 1e-5, under a
3,000-step ceiling. L-BFGS checks every 10 iterations with patience 10 checks,
under a 400-iteration ceiling. The best validation checkpoint is restored, with
no unvalidated training afterward. True test levels/changes are evaluated only
after selection is complete for every arm.

The old fixed-budget rule is rerun on the same split: 500 Adam steps, up to 200
L-BFGS steps and 100 audit steps, with restart selection by training objective.
The separate linear-start NN uses a training-label-fitted linear scorer;
that initialization is misspecified under quadratic truth and is not an oracle.
The ML reference knows the true basis, measurement map and error mechanisms,
while estimating coefficients from the same fitting labels. The log + cell FE
reference is exact OLS on log W with cell intercepts and uses no latent truths.

## Main results: validation-stopped NN from random starts

| Specification | Links | Old change recall | Validation-stopped change recall | Validation-stopped latent test pair accuracy |
|---|---|---:|---:|---:|
| Exponential + additive baseline | None | 9.0% | 25.6% | 95.1% |
| Exponential + additive baseline | Temporal | 19.9% | 47.5% | 94.2% |
| Exponential + additive baseline | Temporal + cross-city | 14.6% | 91.8% | 97.3% |
| Quadratic latent truth | None | 9.2% | 36.5% | 91.5% |
| Quadratic latent truth | Temporal | 14.3% | 77.5% | 92.2% |
| Quadratic latent truth | Temporal + cross-city | 12.7% | 90.4% | 94.2% |

The both-link false-positive rates under validation stopping are 5.86% in the
exponential case and 2.72% in the quadratic case. Each cutoff is calibrated on
an independent stable panel to target 5%, so the realized test rates can differ.

The ML reference achieves 100% change recall in both cases, with 99.43% and
99.00% latent test pair accuracy, respectively. Both ML fits report convergence.

## Interpretation of the exponential case

The exponential map is strictly increasing within cells. Accordingly, all
ranking estimator results match the previous affine/linear-truth run exactly:
the observation levels change, but the labels entering training and validation
comparisons do not. That invariance was checked numerically, including training
objectives, validation losses and test change metrics (maximum discrepancy 0).

Although log + cell FE is misspecified as a level relationship in this design,
it still ranks well in this particular sample: 97.4% latent pair accuracy and
99.4% change recall. Misspecification does not imply that these ranking/change
metrics must be poor in every sample. The result has not been tuned to make
fixed effects fail.

## Verification and scope

Ten automated tests passed, covering early-stop restoration and patience,
validation-only checkpoint/restart selection, tract-level separation, lack of
latent/test-label leakage, ranking invariance under the exponential map,
misspecification of log + cell intercepts, training-only fitting of the FE
reference, and the distinct quadratic latent signal.

Every selected random-start validation loss matches the minimum saved validation
trajectory. Original E5 result hashes remain unchanged. The run used CPU float64,
Python 3.13.7 and PyTorch 2.15.0. These are single-replication diagnostics and do
not estimate Monte Carlo uncertainty.

Reproduce from the source directory in an environment with the numerical packages:

```bash
python ordinal_mc_suite/verification/check_validation_stopping.py --truth linear --measurement exponential_additive --baseline-scale 10 --out ordinal_mc_suite/exponential_new
python ordinal_mc_suite/verification/check_validation_stopping.py --truth quadratic --measurement affine --out ordinal_mc_suite/quadratic_new
python -m unittest discover -s tests -p test_validation_early_stopping.py -v
```

The completed case folders are `validation_exponential_single_run/` and
`validation_quadratic_single_run/`, including results, validation histories,
selected model states and individual figures. This folder contains the combined
PNG/PDF figure, the percentage table and all result rows.
