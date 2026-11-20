# Project progress log

A diary of what we tried, what worked, what we discarded and why, so the project can be picked up again months later. Newest entry first. Each entry cites the documents saved next to this file (dated file names); online versions are linked inside each document.

Two papers share this repository and are kept separate:
- **Econometrics paper:** rank estimation of a common latent index with stability restrictions (Monte Carlo in `src/ordinal_mc_suite/minimal_e1/paper/`).
- **Satellite paper:** building-level / tract-level wealth from aerial imagery (ScaleMAE + LoRA pipeline in `src/main.py`).

---

## 19-nov-26

### Main results

- **The image index R is consistent and robust.** Across E1–E5 the ranking estimators with hard stability links recover the index almost perfectly, including under spatial and serial dependence (E3) and under errors that depend on image features (E4), where unrestricted estimators are inconsistent. In E5 even the replications that fail on ΔW have a perfect index (squared correlation of the estimated and true index change ≈ 1.00).
- **ΔW needs the baseline: the fix is to model the persistence of the tract residual.**
    - *Why the error is persistent.* A tract's income is the income of the households that sort into its bundle of amenities. Part of that bundle is invisible in images (schools, crime, who lives there), so each tract carries a residual η relative to what its images predict. It persists while the amenities stay put, because the same kind of households keep bidding for them, and it is reshuffled when redevelopment re-sorts the tract. On top of it sits transitory noise e (shocks plus ACS sampling error, whose variance is known from the MOE).
    - *Model.* W = ψ_ct(R + η + e), with η_t = ρ(Δ, x)·η_{t−1} + innovation, where Δ measures how much the tract changed. A pure fixed effect is the special case ρ ≡ 1; redevelopment lowers ρ.
    - *How to estimate the persistence.* Fit the head at each labelled date, take residuals r_t = y_t − ĥ_t(R̂_t) for all tracts labelled at both dates, and regress r_{t+k} on r_t with city intercepts: r_{t+k} = a_c + κ(Δ, R̂, C)·r_t. The slope κ = ρ × reliability already includes the shrinkage for transitory noise, so it is directly the weight the prediction needs. It is identified as a function of observables (change Δ, level R̂), not per tract. Stable tracts alone are not enough: κ for changers must be estimated on changers.
    - *Prediction.* ŷ_{t+k} = ĥ_{t+k}(R̂_{t+k}) + κ̂·r_t.
    - *Why E[W_t | W_{t−1}, R_t] beats E[W_t | R_t].* Both are unbiased for their own target, but the first conditions on more information: the baseline residual carries the persistent part that images cannot see. In the theory note's example (persistent variance 0.30, transitory 0.10) the error variance for stable tracts falls from 0.40 to 0.175 (R² from about 0.60 to 0.82). For ΔW, ignoring the baseline is also biased given the tract's history: κ = 0 assumes the whole baseline gap vanishes, κ = 1 carries the noise forward. Estimating κ(Δ) protects changing tracts (error 0.51 → 0.39 in the example, against a constant κ).
    - *Monte Carlo evidence (E5, 1,840 replications).* A learned carry-over gives 1.2–2.2× the R² of ΔW of the best naive rule and reaches 71–81% of the oracle; the images recover 87–92% of the structural change.
    - Details: `2026-11-19 Persistence and dW prediction (theory note).md`; results: `2026-11-19 MC report E1-E5.md`.
- **All remaining ΔW problems were in the head** (the curve mapping R̂ back to W), not in the index. The true curve at the estimated index recovers 92–100% of the structural change; the fitted heads lost the rest.
- **The fix for the head is a parsimonious specification.** At realistic cell sizes (150–250 tracts per city-year), one monotone cubic per cell (intercept + 3 coefficients, no knots) beat every flexible alternative, including cross-validated splines. See `2026-11-19 figures/estimation_gap (head failure, P_rdep_x city 2).png` for what a flexible head does in a sparse tail.

### What we tried

| Trial | Outcome | Status |
| --- | --- | --- |
| E5 baseline-updating experiment: 4 carry-over stories (constant, ρ rising with wealth, reset on change, fixed effect with noisier poor baselines) × `_x` versions (half the persistent part visible in images); predictors survey-only, image-only, κ = 1, constant κ, κ(R), oracle | Learned κ clearly best; P_reset needs κ to depend on change | ✔ kept |
| Split R² of ΔW by stable vs changing tracts | Changers get close to the oracle; mixing groups hides this | ✔ kept |
| Decompose ΔW into map moving / structural / residual dynamics (Shapley split) | On changers: structural 7–21%, map 7–14%, residual 21–40%, unpredictable 45–52% | ✔ kept |
| "Best achievable with the image curve" reference in the structural panel | Separates modelling gap from head estimation gap | ✔ kept |
| Spline df chosen jointly on W and ΔW (linear ½–½ criterion, ΔW on changing tracts) | Small gain (structural 0.853 → 0.865) | ✔ kept as option; superseded by fixed cubic |
| Ridge P-spline, lasso P-spline (trend filtering), median spline as heads | No consistent winner; lasso about 25× slower | ✘ discarded (median kept in mind for "median map" idea) |
| Fixed df vs cross-validation, true index, 150/250/500 tracts | Fixed cubic best at 150–250; knots hurt | ✔ adopted |
| Real-data check (`results/run_20260722/evaluation/figures/US_A_scatter_cells`) | z-score labels, 39–214 tracts per cell, Spearman 0.72–0.84, near-linear with low-end flattening, a few low-side outlier labels | power is the binding constraint |
| Realism audit of the simulated W | Map c (κ = 0.5) far too heavy-tailed; `_x` and E4 extreme | ✘ to recalibrate |
| Budget (SVM) vs weighted stability loss | Equivalent: fix λ, report achieved budget B | ✔ see `2026-11-19 Budget vs weighted-loss equivalence (note).md` |

### Decisions

- **Head:** fixed monotone cubic per city-year (intercept + 3 coefficients). Cross-validation only as a robustness check. Paper line: the head is problem-dependent, but parsimonious specifications are strongly recommended.
- **Target scale:** log W, with W a wealth composite (income plus discounted financial and housing assets). Simulations should produce near-normal log W.
- **Sample sizes:** add realistic small cells; minimum 100 tracts per cell.
- **Pooling:** no pooling across cities. Pooling across years is an optional variant: standardise each year by the stable tracts only (redevelopment would bias the mean and SD), then one head per city. Default stays per cell.
- **Smoothing criterion:** linear W + ΔW for now; Cobb–Douglas version deferred.
- **Stability:** impose links hard; the squared penalty only as a robustness check (fragile with fake links).

### Discarded, and why (check before re-trying)

- **Isotonic head:** too noisy (replaced earlier by the cross-validated B-spline, now by the fixed cubic).
- **Flexible heads (P-splines, lasso splines, knots, cross-validated df):** lose to a cubic at realistic n; their failures come from a handful of tail tracts.
- **Median spline as a drop-in head:** estimates the median, not the mean. Kept only as the first step of a possible "median map + carry-over inside the map" approach.
- **Per-city df tuning for scaling:** not needed. The 8 → 64 city drop in structural recovery was a pooled-metric artefact (median city ≈ 0.93 at both sizes).
- **Pooling heads across cities:** ruled out.
- **Selected penalty weight (RNC_SEL):** never beat the hard restriction.

### Problems found and corrections

- **Heavy-tailed simulated W** (map c, `_x` stretch, E4 iii): produced head blow-ups in about 14% of `_x` replications. Unrealistic for log income. Ranking estimators depend only on the order of W within each cell, so their index results stand; heads, fixed effects, ML and scenarios with noise added after the map must be rerun.
- **Fixed-effects headline at risk:** the 0.033 bias floor was produced under a strongly curved map; recheck after recalibration before claiming fixed effects are inconsistent.
- **Earlier statements corrected in the report:** E1 head gap (spline, not index), E4 gain (2–8×), E5 gain (1.2–2.2×), 64-city decline (metric artefact, not shared df).

### Next steps

1. **Calibrate the simulation to log W:** targets for skew, kurtosis and largest |z| per city-year from the ACS data (E: drive); lower map c curvature (≈ 0.2); equalise latent variance in `_x`; tone down E4 (iii); realism report plus a test that fails outside the band.
2. **Sizes:** add 8×100 and 8×250 (maybe 64×250); nothing below 100 tracts per cell.
3. **Head:** implement the fixed monotone cubic as the default (E5 and the W metrics).
4. **Reruns:** re-head stored fits for ranking arms; rerun fixed effects, ML, E1_cD, E4 (ii) and E5 under the new map; recheck the fixed-effects claim.
5. **Realistic label noise:** 1–2% low-side label outliers (6–10 SD) and ACS-like heteroskedastic noise; optionally a Huber head and inverse-variance weights.
6. **E5 additions:** κ depending on a detected-change flag (building detector, two thresholds: strictly stable / clearly changed / unsure; misclassification in the simulation); the stable-anchored single head per city (mean–SD and quantile-matching versions).
7. **Reporting:** medians with interquartile ranges, per-cell R² next to pooled R², R² on the log or z scale.
8. **Writing (econometrics paper):** the budget–λ equivalence and the parsimony argument; update the Monte Carlo report after the reruns.

### Documents saved here

| File | What it is |
| --- | --- |
| `2026-11-19 MC report E1-E5.md` | Results of all five experiments, decisions and open issues (charts in the online version) |
| `2026-11-19 Persistence and dW prediction (theory note).md` | Why the tract residual persists, how to estimate κ, why conditioning on the baseline helps, the decisive experiment |
| `2026-11-19 Budget vs weighted-loss equivalence (note).md` | λ vs B equivalence for the stability term |
| `2026-11-17 Paper outline - econometrics paper.md` | Outline of the econometrics paper (pilot numbers predate the full runs) |
| `2026-11-19 figures/` | E5 overview figures from the full run (`fig_update_overview_E5_*`, `fig_dW_sources_E5`) and `estimation_gap (...)`: true vs fitted head in a heavy-tailed city |

Other references:
- Design and metrics of the simulation: `src/ordinal_mc_suite/minimal_e1/paper/DESIGN.md`.
- Full results: `src/ordinal_mc_suite/minimal_e1/paper/results_v2/` (tables and figures).
- Earlier identification notes: `docs/Literature Review/Claude outputs/ordinal_identification_note-*.tex`.
- Claude project "satelites-us": `claude/minimal_e1_findings.md`, `proposal_common_index.tex`.

### Code state (not committed)

Changed in `src/ordinal_mc_suite/minimal_e1/`: `paper/heads.py`, `paper/heads_spline.py`, `paper/figures.py`, `paper/summarize.py`, `paper/DESIGN.md`, `tests/test_heads_spline.py`. Adds the joint W + ΔW df rule, P-spline / lasso / median fitters (options, not defaults), the fixed R² axis with ◄ markers, the all-tracts panel and the best-achievable reference. Tests: 128 passed.
