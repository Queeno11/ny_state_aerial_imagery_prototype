# Tract heterogeneity, persistence and ΔW prediction

2026-11-19 · Nico

> Snapshot of the online doc https://claude.ai/artifact/16wYoGuhTVKVJ9o1La5vRT (rev 37), saved 2026-11-19 for the progress log. The head described here (cross-validated B-spline) was replaced on 2026-11-19 by a fixed monotone cubic; see the progress log.


## Why: tract heterogeneity comes from sorting

A tract's income is the income of the households that sort into its bundle of amenities, so the tract effect is neither a classical fixed effect nor noise.

Households differ in how they value amenities. Some are neighbourhood amenities: closeness to the economic centre, security, schools, nightlife, stores, calm. Others belong to the property: build quality, structure, building amenities. In equilibrium the most wanted bundles go to the highest bidders, the allocation unravels downward, and each place's price equals the marginal bid for it (Epple & Sieg 1999; Bayer, Ferreira & McMillan 2007).

Three kinds of amenities matter for what we can learn from images:

- **Fixed for the life of a building:** structure, quality, building amenities. Images see most of these.
- **Fixed over the study span (about 10 years):** many neighbourhood amenities, such as location and street layout.
- **Unobserved or omitted:** schools, crime, shops, local services. Some exist as data in the US, but not everywhere in the world.

The tract effect is therefore the average type of household that the amenity bundle attracts. It stays put while the amenities stay put, because the same kind of residents keep bidding for them. When redevelopment changes the bundle, new people move in and the effect is reshuffled: a new average type, not a small shift of the old one. Neighbourhood change of this kind can also be abrupt once it starts (Card, Mas & Rothstein 2008).

A concrete picture over 10 periods: about 70% of tracts are stable, and some of them carry declared stability links. About 10% redevelop: stable for periods 1–3, changing in periods 4–7, stable again at a new wealth level in periods 8–10. The rest change subtly, with a new building here or a covariate shift there. A tract effect that is constant forever cannot describe the redevelopers. Independent noise cannot describe the stable majority.

## The model: an index that moves with x plus a residual that persists until reset

A tract effect that moves when x moves is the same model as a persistent error with resets; the two only label the pieces differently.

```latex
W_{ict} = \psi_{ct}\left(v_{ict}\right), \qquad v_{ict} = R_{ict} + \eta_{it} + e_{it}, \qquad R_{ict} = f^{*}(x_{ict}) = f(x_{ict}) + E[\text{tract effect} \mid x_{ict}]
```

- **f*(x)** (= R, the image index; R̂ is its estimate) is f plus the part of the tract effect that tracks the observables. v is the full latent value; R is the part of it the images reveal. It changes exactly as much as x changes: little for subtle changes, a lot for redevelopment. This is the "fixed effect correlated with X that moves when X moves".
- **η_it** is the part the images cannot see: who lives there beyond what the buildings and street show.
- **e_it** is transitory: idiosyncratic shocks plus ACS sampling error, whose variance s² we know from the margins of error (MOE/1.645).
- **ψ_ct** is the unknown monotone city-date map, as everywhere else in the project.

The only modelling question is how η evolves:

```latex
\eta_{it} = \rho(\Delta_{it}, x_{it}) \, \eta_{i,t-1} + \text{innovation}_{it}
```

where Δ_it measures how much the tract changed, for example |R̂_it − R̂_i,t−1| or the distance between embeddings.

|  | Pure fixed effect | Persistence with resets |
| --- | --- | --- |
| ρ | 1, and no innovation (with innovations, ρ = 1 is a random walk, not a fixed effect) | ≈ 1 for stable tracts, lower the more the tract changes |
| What happens at redevelopment | only f* moves; the old residual stays | f* moves and the old residual fades (re-sorting) |
| The 10-period tract | periods 1–3 and 8–10 share one residual | periods 1–3 share one residual, it erodes in 4–7, periods 8–10 share a new one |
| Fits the stable majority | yes | yes |
| Fits redevelopers | no: carries the old residents' residual into the new regime | yes |

The fixed-effect model is the special case ρ ≡ 1 with no innovation, so we do not have to choose in advance. The data tell us how much of a baseline surprise to keep for each kind of tract; a structural reading of that weight needs extra assumptions (worked example).

## What is identified, and from which data

With two censuses, f* comes from levels at each date and the persistence ρ comes from comparing the two dates' residuals across tracts; no differencing is needed.

**Data:** images for every tract at every date, and labels (census or ACS) at two dates, about 10 years apart.

**The image index R = f*(x) is identified cross-sectionally, and R̂ estimates it****.** The ranking estimator compares tracts within each city-date cell. Its target is f*, which by construction absorbs everything in the tract effect that is systematically related to the images.

**R̂ stays consistent under persistence (proof sketch, to be written up):**

1. The objective is a sum over city-date cells of pairwise statistics computed within a cell, between different, independent tracts.
2. Identification is per cell and uses only that cell's distribution: given x, the errors are exchangeable, with a distribution that does not depend on x beyond f*.
3. Persistence links a tract's errors across dates, so it never changes any single cell's distribution.
4. The stability term depends only on x, so it is unaffected.
5. Dependence only joins the same tract across cells. Treating the tract (or city) as a cluster, a law of large numbers for clustered pairwise statistics with a fixed number of dates gives consistency. The variance becomes a cluster sandwich: bootstrap by tract or city.

The Monte Carlo already agrees: with persistence ρ = 0.8 (E3), RNC_S has index RMSE 0.0057, against 0.0041 without it, at 64 cities × 500 tracts.

**ρ needs every tract, not only the stable ones.** Stable tracts identify persistence for tracts that did not change. Applying that to changers is biased, which is why both censuses are needed for all tracts.

**Per-tract persistence is not identified, and is not needed.** Each tract gives one residual pair, which is far too noisy on its own. What is identified is persistence as a function of things we observe: κ(Δ, x).

## How R̂ and the heads are estimated (the proposal in brief)

R̂ comes from an ordinal model on images with stability restrictions; the map from R̂ to W comes from a per-city B-spline head.

**Step 1: the ordinal model for R.** Within each city-date, tracts are compared in pairs: the loss rewards scoring the richer tract of each pair higher. Only within-cell orderings are used, so the unknown maps ψ_ct never need to be estimated, and R is identified up to a monotone transformation (in the linear case, R = x'θ with ‖θ‖ = 1). Two losses:

- **RankNet:** logistic pairwise loss with a temperature per city-date, shrunk to a common value. Efficient when noise is roughly Gaussian or logistic.
- **SUR:** Abrevaya's smoothed maximum rank correlation, a bounded sigmoid with bandwidth n_c^−0.6. Robust to noise whose shape depends on x; less precise in small samples.

**Stability restrictions.** Linked pairs are observations known to have the same R: a stable tract at two dates, or cross-city twins. The links require equal scores, f(x_a) = f(x_b). Two ways to impose them:

- **Hard:** in the linear model, θ is confined to the directions with zero gap on every link (the null space of the link differences; an eigen-gap rule picks it when some links may be wrong). Robust to fake links in the Monte Carlo.
- **Weighted loss (penalty):** loss + λ · mean squared link gap, scaled to be unit-free, with λ set by the share ω of the curvature that comes from the links. The only option for the neural net (ScaleMAE + LoRA), where the penalty acts on the score difference of each linked pair. Fragile to fake links at high ω.
- **Budget version (under consideration):** like a soft-margin SVM, minimize the loss subject to a budget on the average *absolute* link gap. Each link's pull is bounded, which should make the neural-net version robust to fake links; the budget is set relative to the gap of the unrestricted fit.

**Step 2: the head, R̂ → W.** Per city-date, a monotone cubic B-spline of W (or of the rank-scale Z) on R̂, with the degrees of freedom chosen by 5-fold cross-validation from {linear, 3, 4, 5, 6, 8}, one choice shared by all cells. It replaced isotonic regression (2.5–3× more accurate) and costs little against the true map in R² on observed W.

## The pipeline

Five steps; only step 4 is new relative to the current framework. Everything is fitted directly on y = log W, per city-date: the heads already absorb the unknown maps, and logs keep residuals comparable across dates when incomes grow proportionally.

1. **Image index.** Estimate R̂_it for every tract and date with the current framework (ranking loss, stability links).
2. **Heads.** At each census date, fit the CV B-spline head per city: ĥ_ct(R̂) = E[y_ct | R̂].
3. **Residuals.** For every tract at both censuses, r_it = y_it − ĥ_ct(R̂_it): how much richer (+) or poorer (−) the tract is than its images suggest. It contains the persistent part η and the transitory part e.
4. **Carry-over regression.** Across all tracts:
    - Regress r_i,t+k = a + κ(Δ_i, R̂_i, C_i) · r_it + v_i, with city terms so that a change in a city's income dispersion between dates is absorbed.
    - κ must depend on the change between censuses, Δ_i = |R̂_i,t+k − R̂_it| or the embedding distance.
    - C are baseline covariates (education, etc.). When C comes from the same survey as W, only baseline C exists at prediction time, so its change cannot enter. The raw 1,024-dimensional embedding is too much for κ.
    - The slope already includes the shrinkage for transitory error, so it is directly the coefficient the prediction needs.
    - Terms like d·H_i,t+k (residual predictable from embeddings) are head improvements, not persistence; keep them separate.
5. **Predict.** ŷ_i,t+k = ĥ_c,t+k(R̂_i,t+k) + κ̂ · r_it, and Ŵ = exp(ŷ) times a smearing factor per cell (the retransformation from logs). The head is not refitted.

**Why not a rank scale.** An earlier draft worked on within-city normal scores and mapped back with a quantile function (GB2). It adds estimation steps, and its units depend on each city-date distribution: a constant latent advantage need not be the same number of units at both dates. The direct head on y avoids both. A quantile mapping is needed only for target dates with no tract labels, to build that date's head from city-level aggregates.

**Dates between or after censuses.** Use the nearest census head, and let κ depend on the cumulative image change since the last label plus a slow geometric time decay. With more than two labelled points (non-overlapping ACS windows, at most one every 5 years), the time decay can be tested instead of assumed. The multi-period version is the same update repeated (a Kalman filter): carry the residual forward between labels, and move toward each new label with the persistent-share weight. It is postponed until the two-date version works in simulations and in the data.

## Worked example: three tracts in one city

The two models agree on stable tracts and disagree on redevelopers, and the carry-over regression tells us which one is right.

**Assumed estimates (illustrative numbers****; y = log income, in units of the city's standard deviation****).** The head gives ĥ(0.5) = 0.40, ĥ(0.7) = 0.55 and ĥ(1.5) = 1.20. Among stable tracts, Cov(r_0, r_1) = 0.30 (persistent variance) and Var(r_0) = 0.40, so the persistent share is λ = 0.75. The carry-over regression by change bin gives:

| Change bin | Slope κ | Implied ρ = κ / λ (only with equal baseline reliability across bins) |
| --- | --- | --- |
| Stable | 0.75 | 1.0 |
| Subtle change | 0.60 | 0.8 |
| Redevelopment | 0.15 | 0.2 |

**Predictions.** All three tracts start at R̂ = 0.5.

|  | Tract A: stable | Tract B: one new building | Tract C: redeveloped |
| --- | --- | --- | --- |
| R̂, census 1 → census 2 | 0.5 → 0.5 | 0.5 → 0.7 | 0.5 → 1.5 |
| Observed y, census 1 | 1.00 | 1.00 | −0.40 |
| Residual r = y − 0.40 | +0.60 | +0.60 | −0.80 |
| Image part ĥ(R̂ at census 2) | 0.40 | 0.55 | 1.20 |
| Predicted y, resets model (κ from the table) | 0.85 | 0.91 | 1.08 |
| Predicted y, fixed-effect model (κ = 0.75) | 0.85 | 1.00 | 0.60 |

- **A, stable:** both models agree. The tract stays well above what its images predict but loses the transitory quarter of its residual.
- **B, subtle change:** the images add +0.15 and most of the residual survives; the models differ only a little.
- **C, redeveloped:** the tract used to be poorer than its images suggested. The fixed-effect model carries that gap into the new regime (0.60). The resets model drops most of it, because the old residents left (1.08).

**The test between the two models** is not the change-bin slope by itself. A slope that falls with Δ, out of sample, shows that baseline surprises deserve less weight for visibly changing tracts, which is what prediction needs. Reading it as a structural reset (ρ = κ / λ) needs more: equal baseline reliability and equal persistent dispersion across bins, on a scale where persistence is constant. Counterexample: with a literal fixed effect, r_t = η_i + e_it, the slope within group D is Var(η|D) / (Var(η|D) + Var(e|D)); if future redevelopers have noisier baselines or less dispersed η, their slope is lower although nothing resets. Known ACS standard errors remove the reliability part; three non-overlapping waves would separate persistence from dispersion.

## What we gain, and what does not change

Persistence does not change R̂ or the heads; it improves tract-level predictions of later W and of ΔW, and for ΔW it removes a bias given the tract's history.

**Unchanged:** R̂ is consistent with or without modelling persistence, and each head is an unbiased estimate of E[y | R̂] at its date.

**What changes is the prediction target:**

- Without persistence we predict E[W_t+k | x]: what a tract with these images earns on average.
- With persistence we predict E[W_t+k | x, W_t]: what this tract earns, given its images and its own past.

Both are unbiased for their own target. The second has lower error for the observed outcome because it uses more information.

**For ΔW, ignoring persistence is also biased given the tract's history.** Predicting Ŵ_t+k(x) − W_t assumes the tract's whole baseline gap vanishes (κ = 0). Predicting ĥ_t+k − ĥ_t misses the mean reversion of the transitory part. With the estimated κ, the predicted change contains exactly the part of the gap that fades, −(1 − κ)·r_t.

**Size of the gain (example: persistent variance 0.30, transitory 0.10, total ****within-city variance of y**** = 1).** Prediction error variance of the tract's position at the second census:

| Predictor | Stable tract (ρ = 1) | Redeveloped tract (ρ = 0.2) |
| --- | --- | --- |
| Ignore the baseline (κ = 0) | 0.40 | 0.40 |
| Carry the full residual (κ = 1) | 0.20 | 0.68 |
| Constant κ fitted on all tracts (≈ 0.69 with 10% redevelopers) | 0.176 | 0.51 |
| Estimated κ(Δ) | 0.175 | 0.39 |

- Most of the gain comes from using the baseline at all: for the stable majority, R² rises from about 0.60 to about 0.82 in this example. A constant κ already achieves almost all of it: stable tracts reach 0.176 without measuring change.
- Estimating κ(Δ) protects the changers, where ΔW's signal is. Carrying the residual forward blindly makes redevelopers worse than ignoring the baseline. Against a fitted constant κ, measuring change lowers their error from 0.51 to 0.39; whether imagery delivers that in practice is the decisive experiment below.
- Without an estimated κ, one rule must serve everyone: the fixed-effect rule wins on stable tracts and fails on changers, and ignoring the baseline does the opposite.

## The decisive experiment: does current imagery beat an informed baseline?

The test that matters is not "κ(Δ) versus ignoring the baseline": that comparison shows a large win even if measuring change adds almost nothing, because a constant κ already captures nearly all of the stable tracts' gain.

Compare four predictors on the same held-out tracts, with the same target-date information available to every eligible model:

| Predictor | What its comparison establishes |
| --- | --- |
| Baseline income or rank carried forward, with a simple citywide adjustment if available | How well a survey-only forecast works without new imagery |
| Current-image head alone | How well imagery works without the tract's own baseline |
| Image history and baseline residual, with a fitted constant κ | What ordinary baseline adjustment buys |
| Same inputs and mean specification, with κ varying by visible change | What measuring change adds to the choice of carry-over weight |

Report the paired difference in held-out MSE between the last two models, overall and for redevelopers, alongside the full model's gain over the survey-only forecast. These answer the two paper questions:

1. Does new imagery improve a forecast for a place where we already have a survey?
2. Does imagery tell us how much of that survey's tract-specific information to retain?

In the Monte Carlo, include the fixed-effect counterexample with different baseline reliability across change groups. A good predictive method should adapt there too, and that case exposes any interpretation that equates a falling slope with a structural reset.

## Cautions and fallbacks

Four practical risks, plus what to do with a single census.

- **Overlapping ACS windows.** The moments assume transitory errors are independent over time. ACS 5-year estimates less than 5 years apart share sample years, so their sampling errors are correlated, and the stable-tract covariance overstates the persistent variance. This affects only the persistence moments. The main fit (R̂, heads) stays consistent, because shared sampling error is just more serial dependence; only standard errors need tract or city clustering. Estimate κ and validate ΔW on non-overlapping windows (2010, 2015, 2020). Timing: a 5-year estimate averages incomes over t−4…t while an image shows one year, so measured income changes lag image changes. Match images to the window's end year, and expect κ for changers to look smaller if images lead.
- **Changers are selected.** Redevelopment concentrates in particular places. Estimate persistence for changers on changers, model its dependence on x, and check overlap between the x of stable tracts and changers. If redevelopment is driven by things images do not show (an anticipated rezoning), only the changers themselves carry the information.
- **Heterogeneous persistence.** A single κ leaves the overall average unbiased but biases subgroups: if rich tracts are more persistent, a common κ over-shrinks them and under-shrinks poor ones. Allow at least κ(R̂ level, Δ).
- **Fine-tuning on the baseline.** Fine-tuning on the baseline is needed, because some cities look different (Mexico City). But a tract's residual must come from an f that never trained on that tract's label. Use K-fold cross-fitting with blocked spatial folds, about 5 blocks per city, each larger than the range of spatial correlation. With random folds, neighbours share unobserved amenities, the network learns them from the neighbourhood's look, residuals shrink and η leaks into f. Limit what fine-tuning changes: last layers or LoRA, regularized toward the pre-trained model, ideally with the within-city ordinal loss. Diagnostic: compare residual variance from in-sample fine-tuning with the cross-fitted one; a large gap means leakage.

**With a single census,** κ is not identified for the target tracts. Partial information remains:

- Known sampling error gives a floor on the transitory share: λ ≤ 1 − mean(s²) / Var(r_0).
- Spatial structure splits the variance: unobserved amenities are spatially smooth, sampling noise is not, so the residual's covariance with neighbouring tracts estimates the persistent variance. This counts tract-specific persistence as transitory, which errs toward less carry-over.
- Persistence after change has no information in one cross-section. Report bounds for changers: κ = λ (nothing resets) and κ = 0 (full reset). Stable tracts get the same prediction under both.

**Taste shifts.** If a type of tract becomes more valued without visible change (say, lofts becoming fashionable), resident characteristics such as education change. That is handled only if those covariates are in x; images alone will not show it.

## Covariates: what goes where

Keep f image-only and add covariates outside the index, in the head and in κ, then report how much each set adds.

|  | Image + education | Full sorting set (schools, crime, shops, …) |
| --- | --- | --- |
| Level prediction with no baseline | Good | Better: these are exactly the unobserved amenities |
| Prediction with a baseline | Nearly as good: the baseline residual already carries every time-invariant covariate's information | Adds only what changes over time and is observed at prediction dates |
| Transfer abroad | Close to image-only; education is widely available | Poor: many measures don't exist or aren't comparable outside the US |
| Stability links | Fine if C stays outside the linked index | Same; covariates inside x must also be stable for linked tracts, or they create fake links |
| Data effort | Low | High: crime data are patchy across jurisdictions, school measures vary by state |
| Later causal use | Education is close to the outcome and a mediator of gentrification (a bad control) | More mediators |

With a baseline, the full set should add little for ΔW, because the baseline residual already carries what time-invariant amenities explain. That is testable: an ablation of image only, image + education, and image + full set, each in the head and in κ.

## Open questions and next steps

- [ ] Write the consistency proof for R̂ under arbitrary serial dependence of a tract's errors (clustered pairwise statistics).
- [ ] Monte Carlo: a DGP where the tract effect is correlated with x, ρ = 1 for stable tracts and resets for changing ones (full and partial). Compare three predictors (ignore baseline, carry full residual, estimated κ) on R² of observed ΔW, by change group. Run the four-predictor comparison of the decisive experiment, and include the fixed-effect counterexample with different baseline reliability across groups.
- [ ] Later, after the two-date version is verified: multi-period version with gradual change spells (periods 4–7) and a Kalman filter.
- [ ] Run the fixed-effect-vs-resets test (change-bin slopes) on the real ACS panel, with non-overlapping windows.
- [ ] Covariate ablation (image only; + education; + full sorting set), covariates in the head and κ, not in f.
- [ ] Decide the change measure Δ: |ΔR̂| (stability-restricted, so free of sensor and season noise), embedding distance, footprints, permits.

- Open: how κ should scale with the time gap when predicting between censuses; three non-overlapping ACS points can test geometric decay.
- Open: whether resets spill over to neighbouring tracts, which would bring back spatial dependence.

## Sources

- Epple, D. & Sieg, H. (1999). Estimating equilibrium models of local jurisdictions. Journal of Political Economy.
- Bayer, P., Ferreira, F. & McMillan, R. (2007). A unified framework for measuring preferences for schools and neighborhoods. Journal of Political Economy.
- Card, D., Mas, A. & Rothstein, J. (2008). Tipping and the dynamics of segregation. Quarterly Journal of Economics.

Citations are from memory, not checked against the papers.
