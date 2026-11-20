# Affine-aligned joint inverse diagnostic

This corrects the earlier raw-output inverse experiment. The objective is exactly

L = L_rank(s) + 0.3 S(s)/V(s)
    + lambda_R mean_fit((a+b*s(X)-R)^2)/Var_fit(R), b>0.

RankNet uses the raw scorer output. Only the inverse-level loss uses a+b*s.
The same ReLU scorer and observed fitting/validation samples are retained. R is
computed by inverting observed W with the known simulation measurement map;
noise-free f is not used for training or selection. Affine and nonlinear
measurement inverses agree. No head penalty or cell-specific calibration enters
this diagnostic. The main configuration has alpha=0, so its retained E5
complexity term contributes zero.

## Bounded design and verification

One saved replication: r001, 32 cities, quadratic/ReLU, 512 observations/city.
Every arm starts from the same saved A scorer. The global affine calibration is
initialized at the fitting-only least-squares solution a=-1.5024055,
b=0.6083173. Its initial held-out R-MSE reproduces 0.420588. The affine gradient
at initialization is essentially zero (norm 4.4e-16), as expected for the fitting
least-squares optimum, while the level gradient into the scorer is nonzero.
The positive slope is parameterized as exp(log_b).

Learning rates are fixed at scorer=0.001 and affine parameters=0.01. There is
no learning-rate search. Three lambda_R values {0.1,1,10} receive 100 updates
each. One ordinal continuation receives the same 100-update budget and learning
rate. Total new scorer updates: 400. Initial checkpoints are eligible; validation
is checked every ten updates and the minimum is restored. The joint selection
rule is validation raw-score rank logistic plus standardized calibrated noisy-R
MSE, with fixed 1:1 weights independent of lambda. D selects by rank validation;
its reporting calibration is fitted afterward on fitting observations only.

All selections were finalized before evaluating latent changes. Source
checkpoints and main experiment code/settings are unchanged. No replication was
launched or resumed beyond this one-shot diagnostic.

## Results

The selected joint arm uses lambda_R=10 at update 40. Both other lambda values
also select update 40; the ordinal continuation restores update zero.

| Arm | Standardized validation R-MSE | Validation rank loss | Common validation criterion |
|---|---:|---:|---:|
| Frozen scorer + affine calibration | 0.420588 | 0.426439 | 0.847027 |
| Joint scorer + affine calibration | **0.409149** | 0.426473 | **0.835622** |
| Ordinal continuation + post-hoc affine calibration | 0.420588 | 0.426439 | 0.847027 |

The joint arm's fitting R-MSE also falls from 0.416394 to 0.400258. Since the
initial affine fit is already least-squares optimal for the frozen scores, this
fitting improvement cannot be explained by choosing a different global affine
conversion of that same score alone. The network changes beyond that conversion
and achieves an additional held-out noisy-R prediction gain. This does not say
which components of the function improve; between-cell position corrections
could contribute without benefiting within-tract changes.

Independent unseen-city change metrics are:

| Arm | Moderate-change RMSE (step 0.4) | Small-change RMSE (step 0.15) | Fresh stable RMS |
|---|---:|---:|---:|
| Frozen + affine | **0.09196** | **0.05623** | **0.04404** |
| Aligned joint inverse | 0.09683 | 0.07020 | 0.05943 |
| Matched ordinal continuation | **0.09196** | **0.05623** | **0.04404** |

The aligned joint arm worsens moderate-change RMSE by about 5.3%, small-change
RMSE by about 24.8%, and stable RMS by about 34.9% in this trial. Seen-city
results agree qualitatively: moderate-change RMSE rises from 0.08879 to 0.09233,
and stable RMS from 0.04375 to 0.05654.

For context, the earlier raw-output inverse arm had unseen moderate-change
RMSE 0.17772 and stable RMS 0.15578. Correcting the target alignment therefore
removes much of that deterioration, but does not deliver a latent-change gain
over frozen calibration in this sample. The earlier severe deterioration cannot
be attributed entirely to a scale conflict, because pure affine transformations
preserve normalized changes; the network's shape or nuisance sensitivity changed.

## Revised interpretation and next step

The earlier recommendation to move directly to matched-change supervision after
the raw-output inverse test was premature. The properly aligned comparison is
distinct and was necessary. It now demonstrates a genuine additional level
prediction improvement in one trial, while still failing to improve individual
changes. The correct conclusion is conditional on this replication, candidate
grid, optimization budget, and validation rule; no general impossibility of
useful level supervision follows.

If pursuing a broader statement about cardinal supervision, the appropriate
next comparison is paired replication of this affine-aligned inverse arm.
Matched-change supervision supplies a different training target and remains a
separate experiment. No new default, broad replicated run, or change-supervised
arm was launched here.

The paper's existing observed-W finding should remain narrow: in the tested
designs, feeding observed level errors back into the scorer did not improve
individual latent-change magnitudes, while fitting calibration after freezing
preserved the ordinal scorer's stronger change and stability performance. The
new aligned inverse result is a bounded diagnostic extension, not a replicated
finding about all correctly aligned cardinal supervision.

Artifacts: `metadata.json`, `candidate_selection.csv`,
`training_validation_paths.csv`, `selected_level_metrics.csv`,
`selected_target_metrics.csv`, `selected_checkpoint.pt`, `verification.json`.
Runnable code: `verification/e6_aligned_inverse.py` in the suite.
