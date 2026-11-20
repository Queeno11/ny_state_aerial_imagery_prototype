# Paper Monte Carlo (E1–E5): experimental design

How to run it: `paper/README.md`. Code: `minimal_e1/paper/`. The paper outline is in the project doc
"stability restrictions".

## Glossary

### Estimators ("arms")

| Acronym | What it is | Uses links? | Loss |
|---|---|---|---|
| **RNC** | RankNet with **c**ell temperatures: logistic pairwise loss within each city-date cell, with one temperature per cell shrunk to a global value | no | ordinal |
| **RNC_S** | RNC under the **hard** stability restriction: θ is confined to the null space of the link gaps (θ = N u) | yes, hard | ordinal |
| **RNC_P0.5 / P0.9 / P0.99** | RNC + **penalty** λ·P(θ). λ is set so that stability supplies a share ω = 0.5 / 0.9 / 0.99 of the curvature (information); ω → 1 is the hard version | yes, soft | ordinal |
| **RNC_SEL** | RNC with ω picked from {0, 0.5, 0.9, 0.99, hard} by held-out within-cell Kendall τ on a validation split of the *training* tracts | yes | ordinal |
| **RNC_S_DW** | RNC_S with pairs **d**istance-**w**eighted (nearby tracts down-weighted; E3 only) | yes, hard | ordinal |
| **SUR** | **S**moothed **U**-process **R**ank estimator: Abrevaya's smoothed maximum rank correlation, a bounded sigmoid with per-cell bandwidth τ_c = n_c^-0.6. Search: annealed, warm-started from RNC, plus random restarts | no | ordinal |
| **SUR_S** | SUR under the hard restriction (exact 2-d grid search when the restricted space is 2-d) | yes, hard | ordinal |
| **MRC_S** | Exact **M**aximum **R**ank **C**orrelation (Han 1987 / Abrevaya 2000) under the hard restriction, by exact arc sweep. Only run when the restricted space is 2-d and C ≤ 64 | yes, hard | ordinal |
| **FE** | Within-cell OLS of W on x with city-date fixed effects (the practitioner's default) | no | cardinal |
| **FE_S** | FE with θ in the link null space | yes, hard | cardinal |
| **NS** | Normal-scores FE: FE on within-cell rank-based normal scores of W | no | rank → cardinal |
| **MM** | Mixed model (random cell intercept and slope). **Omitted from the Nov-2026 runs** (cost; asymptotically ≈ FE here) | no | cardinal |
| **ML** | Exact maximum likelihood of the true model; only where it is correctly specified (E1 maps a/b/c with Gaussian noise) | no | likelihood |

The suffix rule is: **_S** = hard restriction, **_P<ω>** = penalty at curvature share ω, no suffix = no
stability information.

### Heads (second step: from the index to W)

| Name | What it is |
|---|---|
| **oracle** | The true structural map m_ct(s) = E[W \| s, cell], evaluated at the estimated index s = x'θ̂. It isolates first-step (θ̂) error. |
| **spline** | **The feasible head (since Nov 18).** Per city-date cubic B-spline of W on s, monotone, with the degrees of freedom chosen by 5-fold CV from {2 (linear), 3, 4, 5, 6, 8}, one df shared by all cells. Linear extrapolation outside the training range. |
| **iso** | Isotonic regression (PAVA) per cell. Superseded by the spline (2.5–3× worse); kept only as a reference. |
| **native** | FE/MM's own fitted values (affine in x within the cell). |
| **index** | No head: only the index panel (f) uses it. |

### Targets and metrics

| Name | Definition | Units / scale |
|---|---|---|
| **f** (rmse_f) | RMSE of the *standardised* index on held-out tracts = √(2(1 − cos ∠(θ̂, θ0))) in the within-cell metric | ≈ angular error in radians (0.01 ≈ 0.57°). R²_f ≈ 1 − rmse_f² |
| **fpool** | Same, with the pooled (cross-city) covariance | as f |
| **W** (rmse) | RMSE of the predicted E[W\|x] against the true E[W\|x] (closed form) on held-out tracts, both dates | units of W, see the scale table below |
| **dW** (rmse) | RMSE of the predicted ΔE[W\|x] = E[W\|x, t=1] − E[W\|x, t=0] for the same tract | units of W |
| **R² of f** | squared correlation of the estimated and true index = (1 − rmse_f²/2)² | unit-free |
| **R² of W / dW** | out-of-sample R² against **observed** held-out W (ΔW), within city-date (city): 1 − Σ(W − Ŵ)² / Σ(W − cell mean of *training* W)² | unit-free; the main figures |
| **ceiling** | the same R² when the true E[W\|x] is the predictor: the best any method can reach | |
| **dWc** (E5) | RMSE of predicted ΔW against E[ΔW \| x, W_0], the target when the baseline is observed; dWcS / dWcC = stable / changing tracts | units of W |
| **dWo** (E5) | within-city out-of-sample R² of observed ΔW | unit-free |
| **MCSE** | Monte Carlo SE of the mean RMSE = sd / √reps; whiskers are ±1.96 MCSE | |
| **pseudo-true** | RMSE of the estimator's large-sample limit (one fit at 512 cities × 500 tracts): the asymptotic-bias floor | |

**Scale of the targets** (8×500, 5 draws, within-cell SDs). Divide an RMSE by these numbers to get a
unit-free *normalised RMSE*; normalised RMSE² = 1 − R².

| Scenario | sd E[W\|x] within cell | sd ΔE[W\|x] within city | noise sd of W |
|---|---|---|---|
| E1_a | 0.90 | 0.31 | 1.00 |
| E1_b | 0.97 | 0.47 | 1.11 |
| E1_c | 1.32 | 0.72 | 1.50 |
| E1_cD | 1.39 | 0.74 | 1.60 |
| E2/E3 baseline (map c) | ≈ 1.5 | ≈ 0.7–0.8 | ≈ 1.75 |

The E2/E3 baseline sd is larger than E1_c because the scenarios draw different cities.

### Other symbols

| Symbol | Meaning |
|---|---|
| C, n | Number of cities; tracts per city (per date). Cells = city × date, t ∈ {0, 1}. N = C·n on the plots |
| θ0, θ | True and estimated index direction, ‖θ‖ = 1 in whitened coordinates; f = x'θ0 |
| G_ct, ψ | City-date monotone map from latent index to outcome |
| link | A declared pair of observations with equal index: (x_a − x_b)'θ0 = 0 |
| ω | Curvature share of the stability term: ω = λ tr H_S / (tr H_R + λ tr H_S) |
| null space | Directions θ with zero gap on every link; "gap rule" = keep the eigen-directions below the largest log-gap of the link spectrum (used when links may be fake or approximate) |

## 1. Data-generating process (all experiments)

**Features.** p = 6. The raw coordinates are:
- H: 2 structural, within-city coordinates. Each is centred Exp(1), so skewed, plus a city shift of sd 0.5.
- A: tract nuisance, loaded on H, with a city shift and fresh noise at **every date**.
- Optionally, a 2-d city-level nuisance (N1, not in f) or a city-constant structural feature h_c (N2, in f with
  γ = 0.5). These replace nuisance dimensions so that p stays 6.

The raw coordinates are mixed by a fixed matrix and whitened with a large reference population.
Skewness is deliberate: with Gaussian x every single-index estimator would recover the direction for free.

**Index.** f = (1, 0.7)'H (+ γ h_c), standardised by the reference population; θ0 is f's direction in
whitened coordinates.

**Dates and stability.**
- Half of the tracts (even index) are *stable*: ΔH = 0, while A is redrawn, so their link gaps span the
  nuisance directions.
- The other half move by ΔH ~ N(0, 0.5² I).

**Outcome.** W = G_ct(f + η) with η ~ N(0, 1), independent unless E3 or E4 says otherwise. The maps are:
- **(a) additive / FE world:** G = α_ct + r.
- **(b) affine / mixed-effects world:** G = α_ct + β_ct r, with log β_ct ~ N(0, 0.3²).
- **(c) nonlinear, city-varying convexity:** G = α_ct + β_ct · S_λc(r), where
  S_λ(r) = ((1+λ) e^{κr} − (1−λ) e^{−κr}) / (2κ), κ = 0.5 and λ_c ~ U(−0.9, 0.9). Each city is a different
  mix of convex and concave.
- **cD (map c + noise D):** the two-error case. W = G(f + √0.5 ε) + √0.5 · 1.45 · β_ct · ν, where ν is
  standardised log-normal (median 0, right-skewed). So there is noise **inside** and **outside** the map.

**Links (all exact unless E2 corrupts them).**
- **Temporal:** every stable tract, linked to itself across dates (same index, different A).
- **Cross-city:** "twin" tracts. Cities sit on a ring graph with 4 neighbours (offsets ±1, ±2), and each edge
  gets 3 twin pairs. The twin's H is set so that its index equals its partner's *exactly*, and its nuisance
  is an independent draw.

**Split.** 80% of tracts per city are used for training and 20% are held out, with the same tracts on both
dates. Every metric is computed on held-out tracts of the *same* cities.

**Conditional means.** E[W | x] and ΔE[W | x] are in closed form, with no Monte Carlo error in the targets.

## 2. Stability restrictions as implemented

- **Hard (`_S`).** Stack the training link gaps into D. The restricted space is null(D) (tolerance 1e-4,
  or the gap rule), intersected with directions that vary within cells. In the baseline it is
  **2-dimensional**: 6 → 2 parameters. Directions with no within-cell variation (city-level coordinates)
  get their coefficients from the links when cross-city links span them; otherwise they are 0.
- **Penalty (`_P`).** Add λ·P(θ), with P = mean over links of (d'θ)² / Var_within(x'θ), which is unit-free.
  The ranking loss is a mean over pairs. City-level directions are profiled out of P, so cross-city links
  set them as in the hard arm. That fix was made on 18 Nov 2026; earlier E2 N1_*_both and N2_*_both results
  for the `_P` arms are invalid and must be re-run.
- **λ calibration** (`calibration.json`). In a 20-rep pilot (E1_c, 8×150), λ(ω) had an 8% CV across reps, so
  λ is **fixed** per ω: 0.5 → 0.12, 0.9 → 1.07, 0.99 → 11.7. The gradient-balance rule
  (λ‖∇P‖ = ‖∇R‖ at the start value) corresponds to ω ≈ 0.5 or lower.

## 3. Sizes, replications

| | Sizes (C × n) | Reps |
|---|---|---|
| E1 | 8×50, 8×150, 8×500, 64×500, 128×500 | 100, 100, 100, 50, 30 |
| E2, E3, E4 | 8×150, 8×500, 64×500 | 100, 100, 30 |

- MRC_S runs only for C ≤ 64.
- The ω curve (P0.5, P0.99, SEL) runs only at C = 8.
- Pseudo-true lines come from one fit at 512×500 per distinct DGP (E1 maps, E2 N1/N2, E4). E2's other cells
  and E3 share the E1_c population criterion, so they reuse its line.

## 4. Experiments

### E1: baseline. Value of the restriction with exact links

Question: do stability restrictions improve every estimator? Do the restricted ordinal estimators beat
(i) the practitioner's FE when it is misspecified (bias) and (ii) correctly specified but unrestricted
estimators (ML, MRC) in feasible samples (variance)?

| Scenario | Map | Noise | FE correctly specified? | ML run? |
|---|---|---|---|---|
| E1_a | a (additive) | Gaussian inside | yes | yes |
| E1_b | b (affine, city slopes) | Gaussian inside | direction yes (levels: MM) | yes |
| E1_c | c (nonlinear) | Gaussian inside | no → bias floor | yes |
| E1_cD | c | inside + skewed outside (D) | no | no |

Arms: RNC, RNC_S, RNC_P0.9, SUR, SUR_S, MRC_S, FE, FE_S, NS, (MM), ML.

### E2: links. What kind and how many links, and what if they are wrong

One factor at a time around the E1_c baseline (map c, both link types, all exact). Arms: RNC, RNC_S,
RNC_P0.9, SUR_S, MRC_S, FE, FE_S, plus the ω curve (P0.5, P0.99, SEL) in the F_, A_ and S_ cells at C = 8.

| Scenario | What changes | What it tests |
|---|---|---|
| L_none | no links | unrestricted reference (all `_S`/`_P` arms are NA) |
| L_temp | temporal links only | value of over-time links alone |
| L_cross | cross-city twins only | value of cross-city links alone |
| L_both | both (= baseline) | |
| N1_nuis_temp | + 2-d **city-level nuisance** in x (not in f); temporal links only | can within-city information alone handle city-level confounding directions? |
| N1_nuis_both | same, both link types | do cross-city links fix it? |
| N2_hc_temp | + **city-constant structural feature** h_c in f (γ = 0.5); temporal links only | h_c is invisible within cells and over time, so its coefficient is not identified: levels and ΔW go wrong |
| N2_hc_both | same, both link types | cross-city twins identify γ |
| F_5 / F_20 | 5% / 20% of declared links are **fake**. Fake temporal tracts drift along a fixed direction; fake cross pairs are random neighbour tracts | robustness to false links (hard uses the gap rule) |
| S_25 / S_5 | only 25% / 5% of the true links are **declared** | how many links you need |
| A_02 / A_10 | links are only **approximately** stable: index gap sd δ = 0.02 / 0.10 within-cell SD | robustness to near-stability |

### E3: dependence (exact links, map c)

Question: does dependence hurt point estimation, or only inference? It should only affect inference,
because exchangeability within cells still holds.

| Scenario | Dependence in η |
|---|---|
| D_rho08 | tract persistence across dates, ρ = 0.8 (corr of η_i0, η_i1) |
| D_sp005_ind / D_sp03_ind | spatial Gaussian random field over tract coordinates (unit square per city), exponential covariance, range r = 0.05 / 0.3, 50% of the variance; **independent** across dates |
| D_sp005_sh / D_sp03_sh | same field **shared** across dates (spatially correlated persistence) |
| D_comb | ρ = 0.8 + shared spatial field, r = 0.3 |

Arms: as E1 without ML, plus RNC_S_DW.

### E4: errors correlated with x (omitted-variable analogues)

Here x_j = A1, the first raw nuisance coordinate (skewed, centred and scaled).

| Scenario | Mechanism | Expected |
|---|---|---|
| X_i | inside-G mean shift 0.6·A1 | the identified index is f + 0.6·A1: every estimator converges to the same wrong direction |
| X_ii | omitted variable o = 0.6·A1 + noise, coefficient 0.7 | bias along cov(o, x), plus extra noise |
| X_iii | noise scale and skew depend on A1: exp(0.5·A1 − 0.125) · skewed mixture | E[η\|x] = 0 but the shape varies with x. FE is badly biased (pseudo-true 0.65); ranking-based estimators much less (RNC 0.10, SUR 0.07) |

**Fix (Nov 18).** A1 is heavy-tailed, and inside the sinh map E[W|x] exploded for a few extreme tracts
(pseudo-true W RMSE 10.6 in X_ii and 10^78 in X_iii). A1 is now clipped at ±2.5 SD where it enters the E4 error
terms (`dgp.E4_A1_CLIP`). The E4 pseudo-true cache key changed (`e4<case>clip`), so old E4 caches are ignored.

### E5: baseline updating (does the tract's own baseline label improve date-1 W and ΔW?)

Question: when a tract's baseline outcome W_0 is observed (a census), how much better can we predict W_1 and
ΔW = W_1 − W_0 by carrying over part of the tract's baseline residual, and what does it cost when the carry-over
model is wrong? Framework and reasoning: the project doc "Tract heterogeneity, persistence and ΔW prediction".

**DGP.** E1_c baseline (map c, exact temporal + cross-city links, Gaussian inside noise), with the latent noise
split into a persistent tract part ξ and a transitory part e, each date having unit total variance:
v_t = (1 + c) f_t + ξ_t + e_t, Var(ξ_t) = 0.75 (persistent share λ = 0.75), Var(e_t) = 0.25,
ξ_1 = ρ_i ξ_0 + √(1 − ρ_i²) ξ_new. In the `_x` versions half of the persistent variance is explained by x
through the index itself, ξ = c f + ξ̃ with c² = 0.5·λ, so the target index is f* = (1 + c) f (same direction;
the links stay exact).

| Scenario | Carry-over ρ_i | What it tests |
|---|---|---|
| P_const | 0.8 for every tract | constant κ is the correct model |
| P_rdep | 0.5 + 0.45 Φ(R_i), richer tracts more persistent | κ(R) is the correct model |
| P_reset | 0.95 for stable tracts, 0 (redraw) for changing tracts | resets after change; κ(R) is misspecified (it cannot see change) |
| P_ferel | 1 (literal fixed effect), transitory variance 0.25·(0.4 + 1.2 Φ(−R)): noisier baselines for poorer tracts | the coauthor's counterexample: κ varies with R without any reset |
| `_x` versions | as above | half of the tract effect moves with x |

**Predictors** of W_1 for held-out tracts (W_0 observed; ΔŴ = Ŵ_1 − W_0). Heads are the CV B-spline heads
per city-date fitted on training tracts; residuals r_t = W_t − ĥ_t(R̂_t); κ is estimated on training tracts
(labelled at both dates) by OLS of r_1 on r_0 with city intercepts.

| Predictor | Formula | Meaning |
|---|---|---|
| survey | baseline rank → same quantile of the city's date-1 training distribution | survey only, no imagery |
| image | ĥ_1(R̂_1) | imagery only; implies κ = 0 |
| imgchg | W_0 + ĥ_1(R̂_1) − ĥ_0(R̂_0) | levels-plus-changes practice; implies κ = 1 |
| kconst | ĥ_1(R̂_1) + a_c + κ̂ r_0 | constant carry-over |
| kR | ĥ_1(R̂_1) + a_c + (κ̂_0 + κ̂_1 R̂_0) r_0 | carry-over linear in the baseline index |
| oracle | E[W_1 \| x, W_0], closed form (invert the true date-0 map, condition the normal latent) | infeasible ceiling |

**Metrics** (stable and changing tracts are always reported separately: a modest overall R² can be large
for the 10–50% of tracts that change and near zero for the stable majority):

- `r2_dWo_*`, `r2_dWoS_*`, `r2_dWoC_*`: within-city out-of-sample R² of observed ΔW for all, stable and changing
  tracts; the benchmark is the city's training mean of the same group.
- `rmse_dWc_*` (and `S` / `C`): RMSE of ΔŴ against E[ΔW | x, W_0].
- **Two sources of change are separated.** The oracle's expected change splits into
  E[ΔW | x, W_0] = map + struct + resid:
  - **map** = ψ_1 − ψ_0 evaluated at the tract's structural position (1 + c) f_0: the city-date map moving. It
    is the same function of the index for every tract in the city, so it says nothing about detecting the
    change generated by f.
  - **struct** = the change caused by the structural index change (exactly 0 for stable tracts).
  - **resid** = expected movement of the tract's residual (mean reversion, persistence, the map acting on it).
- `r2_dWsC_*`, `rmse_dWsC_*`: how well each predictor recovers the **structural** part on changing tracts. A
  predictor's structural part is ĥ_1(R̂_1) − ĥ_1(R̂_0) (0 for survey only); this is the metric for "does the
  image index detect change driven by f". The oracle scores 1 by construction (it is the target).
- **Spline df for the updating heads** (`heads.UPDATE_DF_RULE`, `heads_spline.cv_select_df_joint`). The level
  heads used by the E5 predictors are still fitted to W per city-date, but their single shared df is chosen by
  loss = ½(1 − R²_W) + ½(1 − R²_ΔW). Both terms are pooled out-of-fold R² on training tracts. Folds are drawn
  per city at the tract level, so both dates of a tract are held out together. ΔW is predicted by the kR
  predictor actually used: ĥ₁(R̂₁) + carry-over refitted on the training folds, minus W₀. The ΔW term is scored
  on changing tracts (`group='changing'`); `'all'` and `'bigchange'` (image index moved more than the city
  median) are options. The W heads of E1–E4 keep the levels-only rule. Recorded per replication:
  `spline_df_upd`, `r2_W_upd`. Pilot (8×500, 4 reps, mean over scenarios), comparing levels-only with
  changing-tract scoring: R²_W 0.589 → 0.595, ΔW changing 0.426 → 0.451, structural 0.853 → 0.865 (the
  marginal-curve maximum is 0.954). Scoring all tracts dilutes the ΔW term with stable tracts and does worse
  on structure (0.833). Comparing Cobb–Douglas and other criteria is a later task.
  `kind` picks the curve fitter, all tuned by the same joint CV:
  - `'bs'` (default): regression spline with a df grid.
  - `'ps'`: ridge P-spline, 20 equal segments with a 2nd-difference penalty.
  - `'l1'`: lasso P-spline, an L1 penalty on the 2nd differences, solved by reweighted ridge.
  - `'med'`: median regression spline, which estimates the conditional *median*.

  Pilot, mean over scenarios (R²_W / ΔW all / ΔW changing / structural):

  | kind | R²_W | ΔW all | ΔW changing | structural |
  |---|---|---|---|---|
  | bs | 0.595 | 0.400 | 0.451 | 0.865 |
  | ps | 0.588 | 0.367 | 0.440 | 0.865 |
  | l1 | 0.583 | 0.366 | 0.438 | 0.836 |
  | med | 0.596 | 0.401 | 0.445 | 0.871 |

  No fitter wins consistently. Results swing widely across replications in the `_x` scenarios, because a few
  extreme tracts dominate both the fit and the squared-error CV loss: the chosen P-spline penalty ranged from
  0.003 to 100. The lasso is about 25× slower. The median shows no level bias in practice.
- `r2_dWsC_marginal`: reference for that panel (dashed mark): the structural R² of the *marginal* curve,
  E[W_1 | f_1] − E[W_1 | f_0], with the true map and true index. Every image predictor's structural part is a
  marginal-curve difference, so this is the most any of them can reach. Its gap to 1 is the part of the
  structural change that depends on the tract's baseline residual: with a nonlinear map, the effect of an index
  change depends on where the tract sits in latent space, and the residual tells you that (zero gap when the
  residual does not carry over, e.g. changing tracts in P_reset). The gap below it is estimation error. Pilot
  (8×500, 4 reps): the first-step index contributes almost nothing (corr² of ΔR̂ with Δf ≈ 1.00); the head is
  the main source, because CV picks df for the *level* of W, while the structural part is a *difference* of the
  head, which is closer to a derivative and wants a different df (too low in P_rdep_x/P_reset_x, too high in
  P_ferel).
- `vdec_*` / `vdecC_*` (all / changing tracts): Shapley split of the within-city R² of observed ΔW among map,
  struct and resid, plus noise = 1 − oracle R²; the four shares sum to 1. Table
  `tables/variance_decomposition.csv`, figure `fig_dW_sources_<exp>`.
- Fitted weights per replication: `kappa_const`, `kappa_R0`, `kappa_R1`.

**Arms:** RNC_S, RNC, SUR_S, FE (figures use RNC_S). **Sizes:** as E2–E4. The same updating metrics are
computed for E3 D_rho08, which is a literal fixed effect with persistent share 0.8 (re-scored, no refitting).

## 5. Versions of the results

- `results/`: the first E1–E3 run (Nov 17–18), RMSE metrics, isotonic head.
- `results_v2/`: the same replications re-scored by `rehead.py` with the spline head and the R² metrics (the first
  step is not refitted: each replication's panel is regenerated from its seed and the stored θ is re-used;
  the oracle-head RMSE must match the stored one exactly). The penalty arms of E2 N1_nuis_both / N2_hc_both
  are refitted (city-level profiling fix). E4 is run directly into `results_v2/`.

## 6. Known limitations

- ΔW: head error ≈ σ_u·√(2(1−ρ)·df/n). With independent errors across dates (E1) nothing cancels; with tract
  persistence it shrinks automatically (same tracts at both dates).
- The ω curve and the selector are evaluated only at 8 cities.
- MM is omitted. FE ≈ MM asymptotically in every finished DGP.
