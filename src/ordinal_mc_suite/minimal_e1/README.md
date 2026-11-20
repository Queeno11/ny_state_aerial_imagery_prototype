# minimal_e1 — loss-component ablation (Experiment 1, minimal version)

Question: does each component of the criterion (L2 on W, the ordinal RankNet term,
stability, L2 on ΔW) improve recovery of a linear latent index f, relative to
ordinal+stability and to L2, and how do they compare with the standard method for
each outcome map and with maximum likelihood, in small and large samples?

Independent of `parametric_e1/` (the earlier full-proposal code) and all older suites.

> **Acronyms used in the paper Monte Carlo (full definitions in `paper/DESIGN.md`).**
> Suffixes: **_S** = hard stability restriction (θ in the null space of the link gaps), **_P<ω>** = stability
> penalty with curvature share ω, no suffix = no stability information.
>
> | | |
> |---|---|
> | **RNC** | RankNet with per-cell temperatures (logistic pairwise loss within city-date cells) |
> | **SUR** | Abrevaya's smoothed rank estimator (smoothed MRC, sigmoid, bandwidth n_c^-0.6) |
> | **MRC** | exact maximum rank correlation (Han / Abrevaya), only as MRC_S in a 2-d restricted space |
> | **RNC_SEL** | RNC with ω chosen by held-out Kendall τ; **RNC_S_DW** = RNC_S with distance-weighted pairs |
> | **FE / FE_S** | within-cell OLS with city-date fixed effects / the same with the restriction; **NS** = normal-scores FE |
> | **MM / ML** | mixed model (random cell intercept and slope) / exact maximum likelihood where correctly specified |
> | heads | **oracle** = true map at θ̂; **spline** = per-cell cubic B-spline, df by CV (the feasible head); **iso** = per-cell isotonic (old reference); **native** = FE/MM's fitted values |
> | targets | **f** = standardised index (RMSE ≈ angle in radians); **W** = E[W\|x]; **dW** = ΔE[W\|x] over dates |
> | maps | **a** additive (FE world), **b** affine (mixed-effects world), **c** nonlinear city-varying convexity; **cD** = map c with noise inside and outside the map |

## Run (WSL, `torch_geo_env`)

```bash
wsl bash -ic 'cd "/mnt/c/Working Papers/NY State Aerial Imagery Prototype/ny_state_aerial_imagery_prototype/src/ordinal_mc_suite" && conda activate torch_geo_env && python -m unittest minimal_e1.tests.test_minimal_e1 -v'
wsl bash -ic 'cd "/mnt/c/Working Papers/NY State Aerial Imagery Prototype/ny_state_aerial_imagery_prototype/src/ordinal_mc_suite" && conda activate torch_geo_env && python -m minimal_e1.run --stage all --reps 5 --workers 4 && python -m minimal_e1.summarize'
```

* `--stage`: `1` map a, `2` map b, `3` map c (small cells), `4` anchor (64×250, all maps), or `all`.
  Stages can be run separately into the same `--out`; finished map-datasets are skipped, so
  an interrupted run resumes.
* `--reps 5` per small cell; `--anchor-reps` defaults to `--reps`.
* `--smoke` runs tiny sizes in seconds (functional check only).
* Needs only numpy, scipy, pandas (no statsmodels, no torch).

Measured on 2 cores (cloud container, not your machine): stages 1–3 with 5 reps =
40 datasets, 83 s with `--workers 2`; one anchor dataset ≈ 105–140 s on one core,
so stage 4 with 5 reps × 2 ρ ≈ 25 min single-core (≈ 6 min with 4 workers).

Outputs in `minimal_e1/run_5rep/` (or `--out`): `rows.csv` (one row per
replication × map × arm, with θ̂), `summary.csv`, `summary.md`, `manifest.json`,
`target_moments.npz`.

## Design

* **Covariates.** H ∈ R² (centered Exp(1) + city mean, skewed on purpose), nuisance
  A = HB' + city shift + N(0, I₄) redrawn each date; X = M(H, A) with a fixed
  orthogonal M; estimators see whitened z only. f = H₁ + 0.7H₂, standardized on a
  200k-endpoint reference sample, so f = z'θ* with |θ*| = 1 exactly.
* **Change and links.** Even-indexed tracts are stable (ΔH = 0) and supplied as
  exact temporal links; the rest move by 0.5·N(0, I). Links span the 4 nuisance
  directions (rank 4 checked every dataset); stability is imposed as a hard
  constraint θ ∈ null(link gaps) (the λ = ∞ limit), i.e. `_S` arms fit in 2 dims.
* **Residual.** η = √ρ u_i + √(1−ρ) ε_it, Gaussian, σ = 1, ρ ∈ {0, 0.8}. The marginal
  is N(0,1) at every ρ (no shape/persistence confound).
* **Maps** (same X, η, ranks across maps):
  (a) W = α_ct + r; (b) W = α_ct + β_ct r, log β ~ N(0, 0.3²);
  (c) W = α_ct + β_ct S_λc(r), S_λ(r) = [(1+λ)e^{κr} − (1−λ)e^{−κr}]/(2κ),
  κ = 0.5 known, λ_c ~ U(−0.9, 0.9) by city (convex for λ>0, concave for λ<0).
* **Sample axis.** (C, n) ∈ {8, 32} × {10, 40} plus the anchor (64, 250). All panels
  are prefixes of one draw per replication (common random numbers across ρ, sizes, arms).
* **Target.** 8 fixed target cities, fixed 1,000-tract test panel; fresh 20-tract
  calibration sample per city each replication.

### Arms

| Arm | Criterion | Notes |
|---|---|---|
| TRUE | θ* | zero f-error; calibration-only floor for μ |
| FE | pooled within OLS | practitioner (all maps); identical to L2 in map a |
| MM | random-slope linear mixed model (Gaussian ML) | practitioner, map b |
| L2aff | L2 with an affine city-date head | practitioner, map c |
| L2 / L2_S | C_q, q = 0.5, correct-family head | `_S` = with stability |
| L2D / L2D_S | C_q, q* = (1+ρ)/2 | ρ > 0 only (identical to L2 at ρ = 0) |
| ORD / ORD_S | RankNet, free temperature | one fit shared by all maps |
| ORD_L2 / ORD_L2_S | 0.5·C_q/C_q(0) + 0.5·P_RN/log 2 | |
| ORD_L2D / ORD_L2D_S | same with q* | ρ > 0 only |
| ML / ML_S | exact likelihood with Jacobian, σ and ρ estimated | map a: FGLS (= exact ML) |

Correct-family heads (profiled in closed form for a frozen score):
a: α_ct + b s (common b); b: α_ct + β_ct s, β > 0; c: α_ct + c₁e^{κs} + c₂e^{−κs},
c₁ ≥ 0 ≥ c₂ (exactly the map-c conditional mean family).

### Metrics (`rows.csv`)
`f_mse`, `df_mse` (latent level/change on the target panel; also `df_mse_changed`,
`ds_rms_stable`, `cos_true`); outcome mean/change with three heads:
`*_oracle_*` (true target map on the estimated score: representation only),
`*_cal_*` (correct-family head fitted on the 20 calibration tracts, same rule for
every arm), `*_native_*` (practitioner arms' own head family). `summary.csv` adds
MCSEs, paired contrasts vs FE and ML (ratio, mean difference, MCSE) and an exact
bias² + variance split of f and Δf from the stored θ̂.

## Deviations log

1. Map (c) cardinal head relaxes the city-level λ: (c₁, c₂) are free per city-date
   (still the exact conditional-mean family, monotone by sign constraints). The mean
   does not identify λ at these sizes; read μ, not head coefficients.
2. Map (a): the common slope b is estimated (not fixed at 1); every arm keeps its own
   source b at the target and only re-fits intercepts. Arms without a slope of their
   own (ORD, joint) get b from a q = 0.5 source head on their frozen score.
3. ML(a) is feasible GLS (cell fixed intercepts, tract random intercept, free variance,
   ρ ∈ [0, 0.98]); this is the exact Gaussian ML for map a. ML(b,c) estimate ρ in (−1, 1).
4. Target calibration uses the same least-squares correct-family head for every arm,
   ML included (not a target-side likelihood).
5. q* = (1+ρ)/2 uses the true ρ: a fixed design weight, not an estimate.
6. MM is our own Gaussian ML implementation (Woodbury; checked against a dense
   likelihood in the tests), not statsmodels and not REML. Ω unstructured with
   automatic fallback to diagonal (recorded in `extras.spec`).
7. Stability only as the hard constraint; the finite-λ path, approximate/false links
   and cross-city links are deferred.
8. ML(c) bounds |λ| ≤ 0.95. At n = 10 many cities hit the bound and an occasional fit
   reaches the iteration limit; such fits are flagged `not_converged`, kept in
   `rows.csv` and excluded from summary means (counted in `not_ok`), never imputed.
9. Convergence = optimizer success or max |projected gradient| ≤ 1e-6 (L-BFGS-B
   often ends with a line-search warning at machine precision).
10. 5 replications per cell and at the anchor (pilot); MCSEs are rough.

---

# Round 2 — robustness of the ranking scorer to the noise law (`run_robust`)

Question: is RankNet's logistic, single-temperature pair model safe when the noise
law is unknown and varies by city (noise inside and outside ψ, city-varying scale and
shape, gross errors), or should the primary criterion be a bounded surrogate / MRC?
Design follows the Fable review of 17 Nov 2026.

```bash
wsl bash -ic 'cd "/mnt/c/Working Papers/NY State Aerial Imagery Prototype/ny_state_aerial_imagery_prototype/src/ordinal_mc_suite" && conda activate torch_geo_env && python -m unittest minimal_e1.tests.test_robust -v'
wsl bash -ic 'cd "/mnt/c/Working Papers/NY State Aerial Imagery Prototype/ny_state_aerial_imagery_prototype/src/ordinal_mc_suite" && conda activate torch_geo_env && python -m minimal_e1.run_robust --reps-small 400 --reps-anchor 100 --workers 4 && python -m minimal_e1.summarize_robust'
```

Cost (measured on the cloud container, 1 core): ≈ 5.5 s per (32×40) dataset and
≈ 80 s per anchor (64×250) dataset, ≈ 1.1 GB per worker at the anchor. The full
400/100 design is ≈ 17 CPU-hours (≈ 4–5 h with 4 workers); `--reps-anchor 50`
halves the anchor part. Interrupted runs resume. Output: `minimal_e1/run_robust/`.

**Scenarios** (`robust_dgp.py`; ρ = 0; same covariates, links and maps as round 1):
A inside Gaussian (map c) · B city-varying scale σ_c = exp(0.5 z_c) · C city-varying
shape (half Gaussian, half skewed mixture) · D inside + skewed median-zero outside
noise (equal shares) · E map a with skewed noise whose scale depends on the nuisance
coordinate A1 (the boundary case where within-cell ranking is not valid) · F 5% of W
replaced by gross errors.

**Arms** (all also with stability `_S`; at the anchor only `_S`):
FE · NS (FE on within-cell normal scores of W; ψ-invariant) · L2 (correct head) ·
ML (inside-Gaussian working likelihood; misspecified in B–F) · RNG (RankNet, global
temperature) · RNC (RankNet, per-cell temperatures shrunk to the global one; joint
L-BFGS) · SUR1/SUR03/SUR01 (bounded sigmoid on the unit-variance score, annealed
τ = 1 → 0.3 → 0.1; 4 starts on the sphere, global grid + refinement in the
one-angle case) · MRC_S (exact maximum rank correlation, global by arc sweep).

**Metrics**: signed angle in the H-plane (mean = bias, t-test), its SD, f-MSE, paired
MSE ratio to RNG, off-plane norm and the loading on the A1 direction (scenario E).
`summary.md` evaluates the pre-registered decision rules (per-cell vs global
temperature; switch primary to the surrogate; contamination; boundary case).

Round-2 deviations: decision rule 1 uses (32×40) without stability (no n = 10 cell);
the RNC shrinkage prior (SD 0.5 on log temperature, scaled by 1/n) is a design choice;
ML is kept only as a labelled misspecified working likelihood (exact ML with both
noise components needs a convolution).

---

# Round 3 — ordinal-only scorer and second-step heads (`run_r3`)

Question: does the smoothed-OBJ1 scorer with a shrinking per-cell bandwidth (SURN), or
with a small RankNet anchor (SURH), remove RankNet's misspecification bias without a
variance cost; how costly is tract-level persistence (A8) for ordinal vs likelihood
scorers; and, with the scorer frozen, does an L1 calibration head beat the L2 head for
predicting observed W in new data?

```bash
python -m unittest minimal_e1.tests.test_r3 -v
python -m minimal_e1.run_r3 --smoke --out /tmp/r3_smoke          # seconds
python -m minimal_e1.run_r3 --reps-small 400 --reps-anchor 100 --workers 4
python -m minimal_e1.summarize_r3                                 # -> run_r3/summary.{csv,md}
```

**Scenarios** A, A8, B–F (`robust_dgp.SCENARIOS_R3`; `SCENARIOS` is unchanged). A8 = A with
tract-level persistence ρ = 0.8 via `dgp.residuals(..., 0.8, n_draw)` on the same stream keys,
so X and the innovations are shared with A (common random numbers; A–F data are bit-identical
to round 2). **Sizes** (8 × 10) block 4, (32 × 40) block 5 (all arms), anchor (64 × 250) block 6
(`_S` arms only). Defaults: 400 / 400 / 100 reps.

**Scorers** (`r3_estimators.py`; full space and, with `_S`, the link null space)

* `SUR(θ; τ_c) = Σ_k w_k σ(−v_kᵀθ/τ_c(k))` on the unit sphere, with `v_k = y_k Δz_k`. This is
  Abrevaya (2000) Eq. 9, smoothed OBJ1 with the logistic kernel K: `SUR = const − Σ w y K(Δzᵀθ/τ)`
  (K(−v) = 1 − K(v); tested numerically). Consistent only as τ → 0; at fixed τ it is biased.
* **SURN**: minimise SUR with `τ_c = n_c^(−0.6)` (n = 10 → 0.251, 40 → 0.109, 250 → 0.036).
  Per-cell τ vector supported (`pairs.cell`). Anneal 1.0 → 0.3 → τ_c (levels at or below the
  target are skipped). Full space: start from RNC plus 3 seeded random starts, keep the best final
  objective. Link null space: global grid over the angle (180 points) plus bounded refinement at
  every level (around the grid minimum and the previous level's solution). For > 400k pairs
  the *grid scan only* uses a fixed unbiased 400k-pair subsample; refinement and reported
  objective use all pairs.
* **SURH**: minimise `SUR(θ; τ_c) + ε_n · P_RN(θ/τ0)`, `P_RN(u) = Σ w log(1 + e^{−vᵀu})`,
  `ε_n = 1/n`, `τ0` = median RNC temperature of the starting fit, fixed (not optimised).
* Unchanged from round 2: FE, NS, L2 (correct-family cardinal, labelled cardinal reference),
  ML (inside-Gaussian working likelihood: `maximum_likelihood` for map c, `fgls` for map a),
  RNC, MRC_S. RNG (global-temperature RankNet) is fitted as a helper (L2 start; τ0 fallback) but
  is not an arm.

**Second step** (`r3_heads.py`, frozen scorer, observed W, out of sample): calibration panel
`noisy_panel(cfg, scenario, 8, 20, rep, block 20)` per replication, fixed test panel
`(8, 500, rep 1, block 21)`. Heads per cell-date: map c `α + c₁e^{κs} + c₂e^{−κs}`, c₁ ≥ 0 ≥ c₂;
map a `α + βs`, β ≥ 0. L2 = `heads.fit_head(q = 0.5)`; L1 = least absolute deviations by
`scipy.optimize.linprog` (HiGHS). Metrics: within-cell Kendall τ (head-free), MAE and MSE of the
L1 and L2 heads; the `TRUE` arm (θ*) is the floor. Scorer metrics are those of round 2
(signed angle, f-MSE, ...).

**Round-3 deviations**

1. *Shared target cities.* A per-cell-date head only transfers to the same city, so the
   calibration panel (block 20, fresh tract/noise draws each rep) uses the **city contexts of
   the test panel** (block 21, rep 1) via a new optional `city_key` argument of `noisy_panel`
   (default `None` = unchanged behaviour; tested). Tract and noise draws are independent of the
   test panel's.
2. The test panel has n = 500 > `n_draw` (250), which would silently truncate to 250 tracts, so
   `n_draw` is raised to 500 for the test panel only.
3. 2-d grid scan on a pair subsample at the anchor (see above), for speed.
4. `--smoke` also caps reps (2 small / 1 anchor) and shrinks the test panel to 60 tracts.

---

# Paper Monte Carlo (E1–E4) — `minimal_e1/paper/`

* **Design and glossary:** `paper/DESIGN.md`
* **How to run:** `paper/README.md`
