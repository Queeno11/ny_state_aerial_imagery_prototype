# Gradient-balanced cardinal supervision in the A4(a) pilot

Smaller cardinal weights reduced the deterioration from 1,000 joint updates relative to the original weights, on both the replay and the fresh confirmation draws. They did not improve the selected scorer over ordinal continuation on the fresh draws. The main improvement in predicting outcomes came from a weighted, curvature-regularized head that can be fitted on a frozen ordinal scorer. On fresh draws, it reduced expected outcome-change MSE from **.29555 to .18803** versus equal level-plus-change calibration, and from **.26792 to .18803** versus level-only calibration.

## Coefficients selected on fresh calibration data

Calibration used only fit and observed validation observations from draws 101 and 102. The replay used the original draws 1–3, whose test results were already known. Confirmation used new draws 201–203. The coefficient choice was frozen before either evaluation. The original rank loss, temperature schedule, score normalization and stability weight .3 stayed fixed.

At the common ordinal anchor, define G_ref = sqrt(||g_rank||² + ||.3 g_stability||²) on fitting data. Each cardinal coefficient equalled its requested relative budget times the median across the two calibration anchors of G_ref divided by the corresponding raw cardinal gradient norm. The two scale ratios were .337887 and .221199 for levels, and .100539 and .083415 for differences. The medians were .279543 and .091977. The original .1/.1 weights corresponded to relative budgets of about .36 and 1.09, respectively. The reference avoids using the norm of the partially cancelling sum of rank and stability gradients.

The search crossed relative budgets {0,.1,.3,1} for both terms and added the original .1/.1 point. For each branch, actual observed rank misordering plus .3 validation stability plus standardized level and change MSE selected a checkpoint from updates 0–1,000, checked every 25 updates. Every branch, including ordinal, got the same fit-only unpenalized level-plus-change head refit at each check. The two validation minima were averaged with equal weight. Within 1e−6 of the smallest mean, the rule preferred a smaller sum of relative budgets, then smaller curvature, then name.

| Candidate | Draw 101 | Draw 102 | Mean validation | Selected updates |
|---|---:|---:|---:|---:|
| Ordinal continuation, both zero | 1.270353 | 1.260527 | **1.265440** | 250, 100 |
| Level only, .1 relative | 1.271588 | 1.260966 | 1.266277 | — |
| Change only, .1 relative | 1.271636 | 1.261135 | 1.266385 | — |
| Balanced positive pair | 1.272115 | 1.261306 | 1.266710 | 0, 0 |
| Original .1/.1 | 1.272115 | 1.261306 | 1.266710 | 0, 0 |
| Balanced pair, curvature .001 | 1.271984 | 1.261306 | 1.266645 | 125, 0 |

Ordinal continuation won on both draws, by .000837 over the next best mean. Among positive pairs, the tie rule selected **λ_W = .0279542993** and **λ_ΔW = .0091976915**. Both selected the anchor, tying the original coefficients and other positive candidates. This identifies the smallest-budget choice in a validation tie.

For that pair, curvature γ ∈ {0,.001,.01,.1} was compared. The common-refit selector preferred **γ = .001** by .0000169 over the runner-up. Draw 102 individually preferred .1. The mean native-head validation composites for γ = 0, .001, .01 and .1 were 1.266710, 1.261756, 1.260707 and 1.263458. The primary common-refit selector intentionally removes the direct effect of curvature on the final head, so its choice is about the scorer. The reported native results expose the different ranking under native calibration. Curvature's direct scorer gradient was zero. The raw head-gradient norm of curvature was .293 and .364 at the two common starts. Shared Adam second-moment diagnostics at updates 25, 250 and 1,000 divide the instantaneous gradients by the same bias-corrected scale. These are descriptive directions, without a decomposition of the actual Adam update. [All 20 candidates](calibration/candidate_aggregate.csv) · [Frozen selection](calibration/selected.json)

## Scorer results

LD uses the selected positive pair with γ = 0. LDC adds γ = .001. The overall winner is the ordinal continuation, so its checkpoint serves as the global selected arm. L and D use the corresponding marginal coefficients from LD. Original-weight LD and LDC were trained again on the fresh confirmation draws.

| Data | Arm | Selected unseen change RMSE | Selected stable RMS | Selected expected-W level MSE | Selected expected-W change MSE | Fixed unseen change RMSE | Fixed expected-W change MSE |
|---|---|---:|---:|---:|---:|---:|---:|
| Replay | Ordinal continuation | .10374 | .04702 | .25225 | .29175 | .10621 | .30303 |
| Replay | Balanced LD | .10414 | .05001 | .25319 | .29455 | .10871 | .31213 |
| Replay | Balanced LDC | .10426 | .05034 | .25261 | .29460 | .10818 | .30792 |
| Replay | Original LD | .10605 | .04566 | .26265 | .29546 | .12785 | .40068 |
| Replay | Original LDC | .10593 | .04562 | .26231 | .29477 | .12571 | .36704 |
| Fresh | Ordinal continuation | .10867 | .05516 | .25160 | .29555 | .10729 | .30474 |
| Fresh | Balanced LD | .11034 | .05783 | .25363 | .29981 | .11126 | .32337 |
| Fresh | Balanced LDC | .11034 | .05800 | .25308 | .29972 | .11107 | .31511 |
| Fresh | Original LD | .11027 | .05510 | .25724 | .29676 | .13258 | .41338 |
| Fresh | Original LDC | .11087 | .05732 | .25809 | .30063 | .12902 | .38358 |

Outcome MSE in this table uses seen cities and common head refits. Change RMSE uses unseen tracts with actual |Δf| > .05. At the fixed endpoint on fresh draws, stable RMS was .05890 for ordinal, .06478 for balanced LD and .09063 for original LD. Smaller weights also selected positive-update checkpoints more often on the fresh draws. Balanced LD selected updates 500, 300 and 0, compared with 25, 0 and 0 for original LD. The fixed endpoints and positive-update checkpoints are available independently of the validation-selected endpoints. With three fresh draws, the paired differences are descriptive.

Fresh matched differences below subtract ordinal continuation. All differences are exactly zero in draw 203, where all three arms selected the common anchor. FPR is from the mixed panel's assigned stable tracts. Dose flag differences are in percentage points.

| Arm | Draw | Unseen Δ RMSE | Stable Δ RMS | Fresh FPR Δ, pp | Seen expected-W level/change Δ MSE | Flag Δ at −.15 / −.05 / +.05 / +.15 |
|---|---:|---:|---:|---:|---:|---:|
| LD | 201 | +.00350 | +.00547 | +.49 | +.00192 / +.00904 | −4.05 / −2.32 / −2.37 / −4.79 |
| LD | 202 | +.00153 | +.00255 | −.05 | +.00417 / +.00376 | −3.56 / −1.56 / −1.68 / −3.39 |
| LD | 203 | 0 | 0 | 0 | 0 / 0 | 0 / 0 / 0 / 0 |
| LDC | 201 | +.00314 | +.00547 | +.44 | +.00112 / +.00835 | −4.74 / −2.32 / −2.42 / −4.52 |
| LDC | 202 | +.00186 | +.00306 | +.15 | +.00333 / +.00417 | −3.10 / −1.56 / −1.12 / −3.05 |
| LDC | 203 | 0 | 0 | 0 | 0 / 0 | 0 / 0 / 0 / 0 |

With 4,096 unseen tracts per dose per fresh draw, ordinal continuation flagged 77.1% and 76.8% at nominal −.15 and +.15, with 98.8% and 98.6% correct signs. The median realized true changes were −.159 and +.163. At ±.05 it flagged 16% with roughly 86% correct signs. The fresh dose-zero false-positive rate was 5.15%. In the mixed panel, the lower, middle and higher baseline tertile slopes were .887, 1.058 and .851, with change RMSE .099, .091 and .132. Normalizing the true score on the finite fit sample contributed .0117 RMSE by itself. The dose figure's band is the spread across tracts, averaged across draws. It is neither uncertainty in the mean nor a prediction-error interval.

The four matched image regimes were rerun. On fresh draws the map-only regime moved expected W by about 1.00 in absolute value with zero structural change. Its predictions equal the acquisition-only predictions because the inputs are identical. In the structural-only and deliberately opposing-map panels, the same ordinal scorer agreed with the expected-W direction 99.85% and .15% of the time, respectively, despite identical structural predictions. This separates targets at inference. It does not test a change to the training maps. Practical W heads are evaluated only in seen supported cells, including the predeclared subset with expected-W and structural sign opposition. [Replay results](results/replay) · [Fresh results](results/confirmation)

## Outcome calibration on frozen scorers

Native LDC appeared to beat ordinal continuation with its common head on fresh expected-W change (.23893 versus .29555). For an equal comparison, each saved scorer was given the same four convex head options below and selected among them using the same observed validation sum of standardized level and change MSE. All 48 scorer–draw records chose the balanced, curvature-regularized option. Native and common results remain available separately. The earlier choice between native and common was asymmetric across arms and is retained as exploratory output.

All four option diagnostics below use the identical ordinal continuation scorer within each draw. Test outcomes for unselected options are displayed after selection and did not affect it.

| Data | Head | Standardized val level/change MSE | Test expected-W level/change MSE | Test realized-W level/change MSE |
|---|---|---:|---:|---:|
| Replay | Level only | .25302 / .88732 | .19442 / .24250 | 4.25316 / 8.35378 |
| Replay | Equal level and change | .25631 / .89234 | .25225 / .29175 | 4.31222 / 8.40381 |
| Replay | Balanced weights | .25384 / .88999 | .21043 / .26486 | 4.26925 / 8.37525 |
| Replay | Balanced and regularized | **.25036 / .87708** | **.15520 / .16881** | **4.21806 / 8.29155** |
| Fresh | Level only | .25083 / .87039 | .20799 / .26792 | 4.36535 / 8.52204 |
| Fresh | Equal level and change | .25301 / .87371 | .25160 / .29555 | 4.41055 / 8.54190 |
| Fresh | Balanced weights | .25117 / .87193 | .21735 / .27682 | 4.37556 / 8.52683 |
| Fresh | Balanced and regularized | **.24826 / .86264** | **.16516 / .18803** | **4.32013 / 8.44143** |

Reweighting alone improved on equal weights and remained worse than level-only. Adding curvature to the same balanced objective gave a further reduction. Fresh expected change MSE was about 30% lower than level-only and 36% lower than equal calibration. For the same selected head library on fresh draws, expected-W level/change MSE was .16516/.18803 for ordinal continuation, .16659/.19047 for balanced LD and .16636/.19079 for balanced LDC. The calibration gains therefore do not require joint scorer updates. [Ordinal option diagnostics](head_library/ordinal_options.csv) · [Equal-library choices](head_library/choices.csv)

## Known maps, invariance and audit

The known-map full loss used the frozen positive coefficients, without curvature because it used exact maps. The normalization passed the known map u = μ_fit(f) + SD_fit(f) z. Both matched starts and the noiseless direct-f references were retrained. The unchanged ordinal and direct-f tensors were bitwise equal to their original counterparts. Starts share one data replication.

| Condition | Start | Unseen latent RMSE, ordinal → full | Seen expected-W level MSE, ordinal → full | Seen expected-W change MSE, ordinal → full |
|---|---:|---:|---:|---:|
| Noiseless | 0 | .07996 → .07294 | .16971 → .06676 | .04001 → .03195 |
| Noiseless | 1 | .06770 → .06436 | .14523 → .04514 | .02546 → .02255 |
| Additive noise | 0 | .10285 → .09684 | .17086 → .12409 | .06658 → .06930 |
| Additive noise | 1 | .10773 → .09772 | .90829 → .16959 | .07646 → .06677 |

The two noisy full fits reversed the original weights' increases in latent RMSE, while the expected outcome-change result was mixed. Both noiseless direct-f fits attained the same validation pointwise MSE as before (.00371, .00378), and their unseen change RMSE was .07636 and .07565. All four known-map full fits selected the 1,500-update cap. At the selected known-map ordinal references, the frozen weights gave relative scorer-gradient norms .49–2.86 for levels and .126–.542 for differences. The same numerical coefficients thus have different leverage under privileged maps. Noiseless controls are deterministic capacity checks.

The nonlinear reparameterization check reproduced true rankings and exact stable equality. Maximum changes in conditional-mean levels and differences were 3.6e−15, while normalized latent differences changed by .03691 RMS. The discrete secant curvature proxy fell from .09173 to .02131. In this additive DGP, W is the realized outcome and ψ_ct(f(H)) its conditional mean. The older noise-before-map R is a different setup. Flexible increasing transformations can preserve both realized outcome loss and rank labels while changing normalized latent magnitudes. A4(a) gives an ordering restriction.

The search executed 43,000 scorer updates, replay 19,500, fresh confirmation 25,500 and known-map controls 15,000, for **103,000 substantive updates**. Head calibration and diagnostic scoring added zero scorer updates. The audit verified the coefficient freeze, 90 distinct calibration and evaluation seed entries, full checkpoint budgets, bitwise original ordinal replay, exact A/F scorer and common predictions, positive slopes and outcome residual identities. Selected GPU versus float64 CPU discrepancies were at most 1.04e−6 in normalized scores and 9.61e−6 in predicted outcomes. The equal head library audit checked 192 candidate records and all saved slopes. [Pilot audit](audit.json) · [Head audit](head_library/audit.json) · [Numerical rerun](invariance/checks.json)

## Reproduce

From the suite directory in the existing WSL Python environment, use PYTHONPATH=. and OPENBLAS_NUM_THREADS=1. Choose fresh output directories.

```bash
python a4a_balanced_pilot/calibrate.py --out NEW/calibration
for stage in replay confirmation known; do
  python a4a_balanced_pilot/rerun.py --selected NEW/calibration/selected.json --stage "$stage" --out "NEW/$stage"
done
python a4a_change_pilot/preflight.py --out NEW/invariance/checks.json
for stage in replay confirmation; do
  python a4a_change_pilot/dose_metrics.py --run "NEW/$stage" --out "NEW/$stage/dose_detection_by_rep.csv"
done
python a4a_balanced_pilot/render_balance.py --replay NEW/replay --confirmation NEW/confirmation --known NEW/known --out NEW/results
python a4a_balanced_pilot/audit_balance.py --calibration NEW/calibration --replay NEW/replay --confirmation NEW/confirmation --known NEW/known --out NEW/audit.json
python a4a_balanced_pilot/head_library.py --replay NEW/replay --confirmation NEW/confirmation --selected NEW/calibration/selected.json --out NEW/head_library
python a4a_balanced_pilot/ordinal_head_options.py --replay NEW/replay --confirmation NEW/confirmation --head-library NEW/head_library
```
