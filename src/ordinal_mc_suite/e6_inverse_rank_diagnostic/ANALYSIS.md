# One-shot oracle-inverse supervision with RankNet retained

## Follow-up: held-out affine-calibration benchmark resolves the level comparison

The fitting-only regression R=a+b*s gives a=-1.5024055 and b=0.6083173>0.
On the EXACT SAME 3,264 validation observations, its standardized noisy-R MSE
is **0.420588**, compared with **0.429631** for the selected jointly trained
inverse arm and 4.295357 for uncalibrated A. The variance denominator is identical
in all three cases. No coefficients were fitted or selected on validation data.
No network updates were performed for this check.

| Method | Standardized validation R-MSE | Unseen moderate-change RMSE | Fresh stable RMS |
|---|---:|---:|---:|
| Frozen A/B, uncalibrated levels | 4.29536 | 0.09196 | 0.04404 |
| Frozen scorer + fitting-only positive affine R calibration | **0.42059** | **0.09196** | **0.04404** |
| Selected joint inverse + rank + stability | 0.42963 | 0.17772 | 0.15578 |

Thus all the observed held-out R-level prediction gain of the joint arm was
available through post-hoc affine calibration, with slightly better validation
MSE and without any loss of change accuracy. The common validation criterion
(raw-score rank loss plus standardized R prediction MSE) is also lower for the
frozen calibrated arm: 0.847027 versus 0.883283.

Positive-affine invariance of normalized tract changes was explicitly checked
on all seen/unseen diagnostic panels: maximum numerical difference was 8.3e-16.
The earlier 85% offset accounting explains the level objective's reward, not
the deterioration of changes. That deterioration necessarily involves changes
beyond a common positive affine transformation; a time-invariant city offset
also cancels from within-city tract differences. Missing cross-city links do
not directly explain it. This result remains one bounded trial and does not
establish how a converged, differently initialized inverse arm would perform.

Machine-readable evidence is in `affine_validation_comparison.json`; the
read-only runnable check is `verification/e6_inverse_affine_validation.py`.

This is distinct from the earlier `oracle_noisy_R` reference, which replaced
the rank term with a squared loss. This diagnostic retains the entire original
E5 scorer objective and adds noisy latent-unit supervision:

L = L_rank + 0.3 S/V + alpha J + lambda_R mean((s(X)-R)^2)/Var_fit(R),
where R = Psi_ct_inverse(W), and alpha remains zero as in the main design.

The original raw rank gaps, differentiated S/V, architecture and exact temporal
links are unchanged. Labels are explicitly inverted from observed W using the
known measurement coefficients; stored noise-free f is never a training target.
The affine and nonlinear measurement inverses were verified to agree. Thus one
inverse experiment suffices for both measurement maps.

## Scope and selection

One saved replication was used: replication 1, 32 cities, quadratic/ReLU scorer,
512 labels per city, 13,120 fitting and 3,264 validation observations. Every
candidate starts from exactly the same saved A scorer (also B's frozen scorer).
No calibration head is fitted in the oracle-inverse arm.

The predeclared grid was lambda_R in {0.1,1,10}, crossed with scorer learning
rates {0.001,0.0001}. Each of six candidates received 100 updates. Two ordinal
controls received 100 updates each at the same rates: 800 scorer updates total,
with no added replications. All candidates are checkpointed every ten updates,
including step zero. The inverse arm selects by validation rank logistic loss
plus validation noisy-R MSE/Var_fit(R), with fixed weights 1:1 independent of
lambda. D selects by validation rank loss. Candidate and learning-rate choices
were finalized before any latent test evaluation.

The original MC checkpoint is unchanged. New diagnostic checkpoints and CSVs
are confined to this directory; production MC settings were not modified.

## Result

Validation selects lambda_R=10, scorer LR=0.001, checkpoint 100. The standardized
noisy-R validation MSE falls from 4.29536 to 0.42963, while validation rank loss
rises from 0.42644 to 0.45365. The combined observable criterion falls from
4.72180 to 0.88328. Both ordinal controls restore their initial checkpoint.

Independent unseen-city targets are:

| Method | Moderate-change RMSE (step 0.4) | Small-change RMSE (step 0.15) | Fresh stable RMS |
|---|---:|---:|---:|
| Frozen B / initial A | 0.09196 | 0.05623 | 0.04404 |
| Selected inverse + rank + stability | 0.17772 | 0.15759 | 0.15578 |
| Ordinal continuation, same LR and update budget | 0.09196 | 0.05623 | 0.04404 |

Thus the selected arm improves prediction of noisy R levels while worsening
both latent-change magnitude and stable-tract movement in this bounded trial.
Seen-city results have the same qualitative pattern. Separate exploratory
candidate metrics also show no improvement over A/B for any of the six
validation-restored candidates, but these metrics did not select the arm.

## Important interpretation limits: score location and unfinished optimization

The starting ordinal scorer has arbitrary location. A read-only accounting
check finds a fitting score mean of 2.43085, compared with R's mean -0.02368.
Of its initial standardized R MSE 4.32298, 3.66463 (about 85%) is attributable
to the mean offset alone. Centering the scores analytically would reduce that
MSE to 0.65835. A descriptive positive affine regression of R on the frozen
score gives fitting MSE 0.41639, compared with 0.42547 for the selected inverse
arm after 100 updates. These analytic calculations are not additional trained
scorer arms and were not used for selection.

A positive affine transformation of the frozen score preserves the E5
normalized change estimator exactly. Therefore a large improvement in raw R
MSE can arise from resolving ordinal location/scale indeterminacy, without
improving individual latent changes. Here the coupled network updates also
alter its shape/nuisance sensitivity, and the measured latent targets worsen.
This limits the strength of a claim about intrinsic benefits of cardinal
supervision: the direct raw-score inverse loss spends substantial effort on
an initially mismatched location/scale.

All inverse candidates choose update 100, the budget endpoint. No convergence
claim is warranted, and this single replication does not establish that an
inverse-supervised scorer could never improve under another initialization,
normalization, or optimization budget. In particular, a gauge-aligned inverse
loss would be a distinct experiment; none was added after seeing these results.

## Implication

The bounded raw-output test does not provide the hoped-for evidence that a known
inverse rescues individual magnitude recovery. However, recommending a move to
matched-change supervision from that result alone was premature. A clean joint
inverse comparison must retain raw-score RankNet while fitting a positive affine
conversion a+b*s in its inverse-level loss, initialized from the fitting-only
calibration. That corrected one-shot comparison has now been run and documented
in `../e6_aligned_inverse_diagnostic/ANALYSIS.md`: it improves held-out R-level
prediction beyond frozen affine calibration but does not improve the latent
change targets in that trial. Broader claims about aligned cardinal supervision
require paired replication of the corrected arm. Matched-change supervision is
a separate training-target experiment. No matched-change run was launched here.

For the paper, the supported claim remains narrower than a general statement
about cardinal supervision: in the tested observed-level designs, feeding level
errors back into the scorer did not improve individual latent-change magnitudes;
post-hoc calibration of the frozen scorer preserved its stronger change and
stability performance. The new oracle-inverse result is a single, bounded
diagnostic, not a replicated extension of that finding.

Artifacts: `metadata.json`, `candidate_selection.csv`,
`training_validation_paths.csv`, `selected_target_metrics.csv`,
`exploratory_candidate_metrics.csv`, `diagnostic_checkpoints.pt`, and
`location_scale_accounting.json`. Reproducible scripts are
`verification/e6_inverse_with_rank.py` and
`verification/e6_inverse_location_check.py` in the suite.
