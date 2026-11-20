# Stability restrictions for ordinal estimation: Monte Carlo report (E1–E5)

Nov 19, 2026 · Nicolas Abbate

## Summary

Stability restrictions (links between observations known to share the same latent index) make every ordinal estimator we tried more precise, and they are what keeps the index consistent when the error depends on image features. Five Monte Carlo experiments support this:

- **Efficiency (E1).** Imposing the links exactly cuts the index error of the RankNet estimator by 2–3× (0.0149 against 0.0335 at 8 cities × 500 tracts, nonlinear map). The fixed-effects benchmark stalls at about 0.033 under a non-additive map; the restricted ranking estimators keep converging (0.004 at 128 × 500).
- **Which links (E2).** About 5% of the possible links already gives almost the full gain. Temporal links alone are enough for the within-city index. The hard restriction (with an eigengap rule) shrugs off 20% fake links; the squared penalty does not.
- **Dependence (E3).** Spatial and serial dependence in the errors leave the point estimates essentially unchanged.
- **Misspecification (E4).** When the error depends on an image feature outside the index, unrestricted estimators are inconsistent (index error 0.21–0.33, flat in sample size). The links cut this 2–8× for the same estimator, and the smoothed-MRC version keeps converging under heteroskedasticity.
- **Predicting change (E5).** With a true baseline survey, learning how much of a tract's baseline residual carries over gives 1.2–2.2× the R² of ΔW of the best naive rule, reaching 71–81% of the oracle. The images recover 87–92% of the change caused by structural change.
- **Weak point: the second step.** All remaining problems sit in the curve that maps the index back to W (the head), not in the index. It is fragile in heavy upper tails, and one smoothing choice shared by many cities does not scale.

## Setup

All experiments share one data-generating process and two estimation steps: rank the tracts with an index from the images, then map the index back to wealth.

**Model.** Tract *i* in city *c* at date *t* has observed wealth W = ψ_ct(v), with latent value v = f(x) + error. Here f(x) is a linear index of image features and ψ_ct is an unknown increasing map that differs by city and date (a city-date pair is a *cell*). Three map families are used: (a) additive, (b) a mild nonlinearity, and (c) a convex, non-additive map that is the default from E2 on. Each replication draws C cities × n tracts × 2 dates; the main sizes are 8×150, 8×500 and 64×500, with 100/100/30 replications.

**Stability restrictions.** A *link* declares two observations to have the same latent index: a tract whose structure did not change between dates (temporal link), or twin tracts in different cities (cross-city link). Half the tracts are stable. Links enter in two ways:

- **Hard:** the index is restricted to the null space of the linked feature differences (with an eigengap rule to discard directions when some links are wrong).
- **Penalty:** a squared penalty on linked index differences, its weight λ set so the penalty carries a chosen share ω of the objective's curvature (ω = 0.5, 0.9, 0.99).

**Estimators of the index.**

- **RankNet (RNC):** a pairwise logistic ranking loss within each cell, with a per-cell temperature.
- **Smoothed MRC (SUR):** Abrevaya-style smoothed maximum rank correlation, bandwidth n_c^−0.6.
- **Exact MRC (MRC_S):** maximum rank correlation under the hard restriction.
- **Fixed effects (FE):** a parametric benchmark with cell fixed effects; **ML:** parametric maximum likelihood where the map family is known.

The suffix _S marks the hard restriction and _Pω the penalty.

**Heads (second step).** The index is mapped back to W per cell by a monotone cubic B-spline whose degrees of freedom are chosen by 5-fold cross-validation (one df shared by all cells). It replaced isotonic regression, which was noisy. An *oracle* head (the true map at the estimated index) separates first-step from second-step error.

**Metrics.**

- Index error: RMSE of the standardised index direction, roughly the angle to the truth in radians; asymptotic floors come from large "pseudo-true" runs at 512×500.
- Out-of-sample R² of observed W and ΔW within cell, on held-out tracts, against a ceiling given by the true E[W | x].
- For E5 also: R² split by stable and changing tracts, and recovery of the structural part of the change.

## E1: baseline comparison

With correct links, the hard restriction makes every ranking estimator 2–3× more precise, and only the ranking estimators stay consistent under a non-additive map. E1 compares all estimators on four maps (a, b, c, and c with extra outer noise) from 8×50 to 128×500.

*[Chart in the online doc: index error by sample size under map c for fixed effects, RankNet without links and RankNet with hard links. Source: results_v2/tables/summary_E1.csv, scenario E1_c; 100 replications per size up to 8×500, 50 at 64×500, 30 at 128×500.]*

- **Links pay at every size.** At 8×500 under map c, RankNet's error falls from 0.0335 to 0.0149 with the hard links; under the additive map (a), from 0.0330 to 0.0115. The parametric ML benchmark sits next to unrestricted RankNet (0.0314).
- **Fixed effects are inconsistent under map c.** FE flattens out at about 0.033 even at 128×500, with or without links. The restricted ranking estimators keep converging (RNC_S and SUR_S about 0.004). Under maps a and b FE converges, as it should.
- **Smoothed MRC matches RankNet once restricted;** exact MRC is less efficient in small samples (0.020 against 0.015 at 8×500), because its objective is a step function.
- **Heads.** With the restricted index, the oracle head reaches the ceiling R² of W almost exactly, so what is left is second-step error. The cross-validated B-spline (which replaced a noisier isotonic head) is close under maps a and b (R² of W 0.445 against 0.448 at 8×500) but falls short under map c: 0.42 against 0.47 for W, and −0.01 against 0.08 for ΔW.

## E2: which links, how many, and wrong links

A few correct links are enough, and the hard restriction survives wrong ones; the squared penalty does not. E2 varies the type of link, the share of possible links declared, the share of fake links, and how exact the links are (map c throughout). Index error at 8×500, mean of 100 replications:

| Scenario | Hard links (RNC_S) | Penalty ω = 0.9 | Penalty ω = 0.99 | No links (RNC) |
| --- | --- | --- | --- | --- |
| Temporal and cross-city links | 0.0127 | 0.0132 | | 0.0314 |
| Temporal links only | 0.0124 | 0.0129 | | 0.0306 |
| Cross-city links only | 0.0110 | 0.0115 | | 0.0301 |
| 5% of links declared | 0.0131 | 0.0137 | 0.0132 | 0.0323 |
| 25% of links declared | 0.0124 | 0.0130 | 0.0124 | 0.0323 |
| 5% of links fake | 0.0151 | 0.0613 | 0.917 | 0.0300 |
| 20% of links fake | 0.0210 | 0.305 | 1.147 | 0.0316 |
| Approximate links, small error | 0.0132 | 0.0136 | 0.0132 | 0.0313 |
| Approximate links, large error | 0.0136 | 0.0143 | 0.0304 | 0.0318 |

- **5% of links is nearly enough.** Declaring 5% of the possible links gives almost the whole gain (0.0131 against 0.0127 with all of them).
- **Temporal links alone identify the within-city index.** Cross-city links add a level across cities, which only matters for a head shared across cities; with one head per city and date, as here, temporal links suffice.
- **Wrong links separate the two ways of imposing them.** With 20% fake links, the hard restriction (eigengap rule) loses about 0.008. The squared penalty is pulled off course: 0.31 at ω = 0.9, and worse than having no links at all. A strong penalty also suffers when links are only approximately right.
- **Choosing ω by validation (RNC_SEL) gave 0.020–0.030,** worse than simply imposing the links hard.
- **Nuisance variation in linked pairs,** in two variants, leaves the hard restriction unaffected (0.011–0.015).

## E3 and E4: dependence and misspecification

**E3: dependence does not hurt the point estimates.** E3 adds spatially correlated errors (short and long range, independent across dates or shared), a persistent tract component (correlation 0.8 across dates), and both combined. The restricted RankNet's index error at 8×500 stays between 0.0111 and 0.0178, against 0.0127 without dependence; the persistent tract component costs the most. A dependence-weighted variant of the loss (RNC_S_DW) gave no gain. Dependence will matter for standard errors, which these experiments do not yet assess.

**E4: when the error depends on the images, the links are what keep the index close.** E4 makes the error depend on an image feature that is not part of the true index. In (i) it shifts the error, in (ii) it acts through an omitted variable correlated with that feature, and in (iii) it changes the error's scale and skew. Index error at 64×500, mean of 30 replications:

| Scenario | Fixed effects | RankNet, no links | Smoothed MRC, no links | RankNet, hard links | Smoothed MRC, hard links | Exact MRC, hard links |
| --- | --- | --- | --- | --- | --- | --- |
| (i) error shifted by the feature | 0.263 | 0.304 | 0.307 | 0.042 | 0.041 | 0.041 |
| (ii) omitted variable tied to the feature | 0.213 | 0.239 | 0.241 | 0.034 | 0.032 | 0.032 |
| (iii) error scale depends on the feature | 0.282 | 0.089 | 0.068 | 0.038 | 0.010 | 0.010 |

- **Unrestricted estimators are inconsistent here.** Their error does not fall with sample size (0.21–0.33 from 8×150 to 64×500), because the nuisance feature acts as a proxy for the error.
- **The links cut the error 2–8× relative to the same estimator without them.** The feature varies between the two images of a stable tract while the latent index does not, so the restriction rules out loading on it. Under (i) and (ii) a floor of about 0.03–0.04 remains.
- **Under heteroskedasticity (iii), the MRC estimators keep converging** (0.024 at 8×150 to 0.010 at 64×500), while restricted RankNet stalls near 0.036. A rank-correlation objective is robust to a changing error scale; RankNet's logistic loss is not.
- **Fixed effects with links (FE_S) still sit at 0.07–0.22:** the parametric map is the problem there, not the links.

## E5: baseline updating and predicting ΔW

When a true baseline survey exists, learning how much of each tract's baseline residual carries over predicts the change in wealth far better than either naive rule. E5 adds a persistent tract component (75% of the latent noise) with four carry-over stories, each also in a version where half of it is visible in the images (`_x`). Half the tracts change structurally between the two dates. Predictors of date-1 wealth for held-out tracts whose baseline is observed:

- **Survey only:** carry the baseline rank forward; no images.
- **Image only:** the head at the new index, ignoring the baseline (carry-over κ = 0).
- **Baseline + image change:** W₀ plus the image-predicted change (implicitly κ = 1).
- **Learned carry-over:** the image head plus κ times the baseline residual, with κ constant or linear in the baseline index, κ(R), estimated on training tracts labelled at both dates.
- **Oracle:** the true E[W₁ | x, W₀].

*[Chart in the online doc: R² of observed ΔW over all tracts by predictor for P_const, P_rdep, P_reset and P_ferel. Source: results_v2/tables/rows.csv; E5, RNC_S first step, 64×500, medians over 30 replications.]*

R² of observed ΔW over all tracts, 64×500, medians (the values drawn in that chart):

| Scenario | Survey only | Image only | Baseline + image change | Learned carry-over, κ(R) | Oracle |
| --- | --- | --- | --- | --- | --- |
| P_const | 0.14 | 0.08 | 0.17 | 0.33 | 0.43 |
| P_rdep | 0.10 | 0.14 | 0.17 | 0.37 | 0.45 |
| P_reset | 0.08 | 0.32 | 0.12 | 0.39 | 0.52 |
| P_ferel | 0.18 | −0.22 | 0.25 | 0.36 | 0.51 |

Learning κ gives 1.2–2.2× the R² of the best naive rule and reaches 71–81% of the oracle. A constant κ and κ(R) perform alike. Medians by group, 64×500:

| Scenario | Changing tracts: κ(R) / oracle | Stable tracts: κ(R) / oracle | Structural change recovered (best achievable) | Structural share of ΔW variance, changing tracts |
| --- | --- | --- | --- | --- |
| P_const | 0.38 / 0.47 | 0.31 / 0.40 | 0.89 (0.95) | 15% |
| P_rdep | 0.42 / 0.48 | 0.32 / 0.41 | 0.89 (0.95) | 15% |
| P_reset | 0.50 / 0.57 | 0.15 / 0.41 | 0.92 (1.00) | 7% |
| P_ferel | 0.40 / 0.56 | 0.31 / 0.46 | 0.87 (0.92) | 21% |
| P_const_x | 0.49 / 0.68 | 0.30 / 0.61 | 0.78 (0.97) | 25% |
| P_rdep_x | 0.38 / 0.68 | 0.42 / 0.58 | 0.70 (0.97) | 27% |
| P_reset_x | 0.43 / 0.65 | 0.25 / 0.62 | 0.77 (1.00) | 19% |
| P_ferel_x | 0.52 / 0.76 | 0.50 / 0.64 | 0.74 (0.95) | 38% |

- **Splitting by group matters.** Changing tracts get close to the oracle in the main scenarios. In P_reset, one κ cannot serve stable tracts (true carry-over about 0.71) and changing ones (0): stable tracts fall to 0.15, and image only is best on changing tracts (0.55). A carry-over that depends on a detected-change flag would fix this.
- **The images detect structural change.** They recover 87–92% of the change caused by the index moving, close to the best achievable with the image curve (92–100%). The remainder is second-step error in the head.
- **Where ΔW comes from.** On changing tracts at 64×500, structural change explains 7–21% of the variance of observed ΔW (19–38% in the `_x` versions), the city map moving 7–14% (12–27%), residual dynamics 21–40%, and 45–52% is unpredictable even for the oracle. A low overall R² is therefore expected and is not a failure of the images.
- **The `_x` versions are fragile.** Typical replications are good, but in about 14% of them R² of ΔW is negative, at every size. In those, the index is perfect and the true curve at the estimated index fits as well as ever; the fitted spline head breaks on a few extreme tracts in a convex upper tail. Separately, median structural recovery falls from about 0.85 at 8 cities to 0.70–0.78 at 64, because one smoothing parameter is shared by all 128 city-date curves.

## Main conclusions

The restriction is worth making the centre of the paper: it buys precision when the model is right and protection when the error depends on the images.

1. **Impose the links hard.** The hard restriction matches the best penalty when links are correct, survives 20% fake links, and needs no tuning. The squared penalty adds a tuning choice and fails badly with wrong links, so it is better as a robustness check than as the method.
2. **Few, reliable links beat many noisy ones.** 5% of possible links captures almost the whole gain, and temporal links alone identify the within-city index. In practice this argues for a strict rule when declaring a tract stable (for example a building detector with a conservative threshold).
3. **Use ranking estimators, not parametric fixed effects.** Fixed effects are inconsistent once the city map is non-additive. Among ranking estimators, RankNet scales to neural networks; smoothed MRC is the more robust choice when the error scale varies with the images (E4 iii), so it is worth reporting both.
4. **Predicting change needs the baseline and a learned carry-over.** Neither ignoring the baseline nor adding the image change to it is adequate. Results should be reported separately for stable and changing tracts and for the structural component of the change; much of ΔW is unpredictable even for the oracle, so a modest overall R² is expected.
5. **The bottleneck is the second step.** Across E1–E5 the index is estimated well; the losses come from the curve that maps the index back to W. That is where methodological effort should go next.

## Open issues and next steps

- [ ] **Per-city smoothing for the head.** Choose the spline's flexibility separately for each city; one choice shared by all cells does not scale to many cities (the main issue for a national rollout).
- [ ] **Robustness to heavy tails.** Fit the head on a compressing scale (asinh or log of W) with a retransformation correction, or estimate the map by median regression and carry the residual over inside the map. First check whether the Monte Carlo tail is realistic, since ACS income is top-coded.
- [ ] **Carry-over by detected change.** Let κ depend on a building-detector change flag, with two thresholds (strictly stable, clearly changed, unsure in between), and simulate a misclassified flag feeding both the links and the head.
- [ ] **Choice of the smoothing criterion.** Compare the linear W + ΔW criterion now in use with a Cobb–Douglas version and with alternative ΔW groups.
- [ ] **Inference.** Standard errors and bootstrap coverage under spatial and serial dependence, and a consistency argument under serial dependence.
- [ ] **Reporting.** Plot E5 as medians with interquartile whiskers and the share of failed replications, since a few blow-ups distort the means.
- [ ] **Theory write-ups.** Links in one direction only; the temporal-only identification result; fixed effects inside versus outside the map; a Mundlak/correlated random effects variant.
