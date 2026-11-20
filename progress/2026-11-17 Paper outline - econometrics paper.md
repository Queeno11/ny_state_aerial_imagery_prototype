# Paper outline: rank estimation of a common latent index with stability restrictions

2026-11-17 · Nico

> Snapshot of the online doc https://claude.ai/artifact/Y1SmciBcaa1tmi2mkfyoVC (rev 30), saved 2026-11-19 for the progress log. Pilot numbers in it predate the full E1–E5 runs; see `2026-11-19 MC report E1-E5.md` for current results.

**Working title:** Stability restrictions make rank estimation feasible: a common latent index under unknown city-date monotone maps.

**Pitch.** Outcomes such as tract income are an unknown, city- and date-specific monotone transformation of a common latent index of observed features (imagery). Within-cell ranking (Han 1987; Abrevaya 2000) identifies the index without modelling those maps, but rank estimators have been impractical: too noisy at feasible sample sizes, hard to compute, hard to do inference with. The paper's central contribution is **stability restrictions**: external knowledge that some units did not change (e.g. tracts with no construction, photographed under different conditions) imposes restrictions on the index function that remove nuisance directions and make Abrevaya-type estimation feasible in realistic samples. Supporting pieces are a smoothed and a convex estimator, a second step for outcome maps, changes and effects, and inference.

**Status.** Outline from the November 2026 working sessions. Monte Carlo pilots exist (rounds 1–3 in `minimal_e1`); the full runs and all proofs are still to do.

## 1. Introduction

The paper's claim: under assumptions that need no model for the outcome map or the noise law, a stability-restricted rank estimator recovers the latent index and outcome changes better in realistic samples than misspecified fixed or random effects (less bias) and than correctly specified rank estimators without restrictions (less variance).

Contributions:

1. **Stability restrictions (central).** Known-unchanged units give restrictions on the feature-to-index function, linear restrictions (x_i1 − x_i0)'θ = 0 in the linear case and an invariance penalty for a network. They remove nuisance directions, cut estimation error several-fold at feasible sizes, and make changes comparable over time. This is what turns Abrevaya-type rank estimation from a theoretical object into a usable estimator.
2. **Identification framework.** A common index with city-date-specific monotone maps, identified by within-cell ranking: Abrevaya's (2000) framework with city-date cells as clusters, under pairwise exchangeability rather than i.i.d. errors.
3. **Feasible estimators.** A smoothed within-cell OBJ1 (Abrevaya Eq. 9) with a shrinking per-cell bandwidth, and RankNet with per-cell temperatures as a convex quasi-likelihood alternative; conditions under which each is consistent.
4. **Second step and effects.** Consistent two-step estimation of the city-date conditional-mean maps, giving outcome predictions, outcome changes ΔW and average structural effects in outcome units.
5. **Inference and a specification test.** City-cluster bootstrap, and a Hausman-type contrast between RankNet and the smoothed estimator that tests RankNet's extra assumption.
6. **Neural-network implementation**, with one proposition on order consistency and consistency of the composite prediction.

Preview of results (to fill in from the full Monte Carlo runs): size of the stability gain; RankNet vs smoothed OBJ1 efficiency; where FE/RE are biased; where unrestricted MRC is too noisy.

## 2. Model and identification

The outcome of tract i in city c at date t is a city-date monotone function of a common index and noise:

```latex
W_{ict} = G_{ct}\big(f(x_{ict}),\, U_{ict}\big), \qquad G_{ct} \text{ strictly increasing in } f, \qquad f(x) = x'\theta \text{ in the core theory}
```

The leading case is W = ψ_ct(f + η) + e, noise inside and outside the map. Folding the outside noise in gives an error whose law depends on f, which is why the general form G_ct(f, U) is used.

Assumptions to state:

- **A1 Monotone maps.** G_ct weakly increasing in f; may differ by city and date; never modelled.
- **A2 Pairwise exchangeability.** Within a cell, (U_i, U_j) given X has the same law as (U_j, U_i). Allows spatial correlation, city-wide shocks and tract persistence; excludes noise whose law depends on X beyond f (e.g. variance or skew rising with a covariate).
- **A3 Weak dependence.** Spatial correlation decays with distance (mixing), so within-cell U-statistic laws of large numbers and CLTs hold; cities independent.
- **A4 Support and normalisation.** One continuous covariate with non-zero coefficient and full conditional support (Abrevaya Assumption 3); scale fixed by ‖θ‖ = 1 on a reference distribution. Covariates constant within a cell are not identified.

What is identified:

- **Linear f:** θ up to positive scale.
- **Flexible f:** f up to a global strictly increasing transform g (the ordering), when X supports overlap across cells.
- **Maps:** the conditional mean m_ct(s) = E[W | s, c, t] (or median) given the index; ψ_ct itself is not separable from the noise.
- **Composite:** m_ct(f(x)) is identified even though f and m_ct separately are not, so predictions and ΔW are identified.
- **Changes:** Δf is identified only up to g; its magnitude needs an extra normalisation. The paper therefore targets ΔW, not Δf.

Proof sketch: Abrevaya's swap argument (pp. 6–7, A.3), with the fixed effect replaced by the cell and i.i.d. replaced by exchangeability.

## 3. Stability restrictions

Stability restrictions are the paper's central contribution: they make within-cell rank estimation precise enough to use at feasible sample sizes, and they make the index comparable over time.

**Definition.** A stability link is a pair of observations whose latent index is known to be equal: the same tract at two dates with no structural change (only acquisition conditions such as season, sensor or lighting differ), or, more generally, two units judged equal across contexts. Each link imposes f(x_a) = f(x_b) on the feature-to-index function, not on the outcome maps.

```latex
(x_{i1} - x_{i0})'\theta = 0 \;\; \text{for every linked } i \quad\Longleftrightarrow\quad \theta \in \operatorname{null}(D), \qquad D = \big[\,x_{i1} - x_{i0}\,\big]_{i \in \text{links}}
```

**Assumption S (stability).** Linked pairs truly have equal latent index: no unobserved change. Its failure mode is a supposedly stable unit that did change; Experiment 2 measures the cost of fake and missing links, and over-identifying links allow a consistency check.

Results to establish:

- **Theorem S1 (variance).** For an efficient estimator, imposing Rθ = 0 weakly reduces asymptotic variance (P A P′ ≤ A); for any √n estimator, imposing it by minimum distance with weight A⁻¹ weakly dominates. The rate is unchanged; the effective dimension falls from p to p − r.
- **Theorem S2 (finite samples).** Restricting the function class lowers its complexity and tightens excess-risk bounds for the ranking loss.
- **Proposition S3 (cross-context links, open).** Whether links across cities identify cross-city comparisons that within-cell ranking alone leaves unidentified.
- **Proposition S4 (network).** Stability as an invariance penalty on f inside the ranking loss, tightened during training.

**Why it matters (pilot evidence).** Within-cell rank estimators are √n-consistent but noisy at feasible sizes. Exact stability links cut index error 3–50× and outcome-change error about 10× for every estimator. With them, the rank estimators beat misspecified FE/RE on bias and unrestricted MRC on variance.

**Origin.** The idea comes from satellite and remote-sensing practice: pseudo-invariant features for normalising images across dates, and same-location consistency in imagery models (see the literature section). It applies to any setup with unknown monotone maps and external knowledge of units that did not change.

## 4. Estimators and theory

Two estimators share one identification argument and differ in what they need to reach it. Both have a version restricted by the stability links.

**4.1 Smoothed within-cell OBJ1 (SUR).** Minimise over ‖θ‖ = 1

```latex
\sum_{c,t} \frac{1}{\binom{n_c}{2}} \sum_{i<j} \sigma\!\left(-\,\mathrm{sign}(W_j - W_i)\,\frac{(x_j - x_i)'\theta}{\tau_c}\right), \qquad \tau_c = n_c^{-0.6} \to 0
```

This is exactly Abrevaya's Eq. 9 with a logistic kernel (constant minus Σ y·K). Theory: consistency from Abrevaya Theorem 2 (bounded kernel, τ → 0, fixed scale); asymptotic normality in the large-cell regime with bias o(n^{-1/2}) by undersmoothing. Unsmoothed within-cell MRC is √n there (Sherman 1993), so smoothing is a computational device. Computation: non-convex; anneal τ from a RankNet start; exact global solution by grid or arc sweep when the restricted problem is one-dimensional.

**4.2 RankNet with per-cell temperatures (RNC).** Logistic loss on within-cell pair signs with an estimated temperature per cell, shrunk to a common value. Convex in θ for fixed temperatures. Results to show:

- **Exact quasi-likelihood** when pair probabilities are logistic in the index difference (e.g. Gumbel inside noise).
- **Direction-consistent for any link** when E[x | x'θ] is linear (Ruud; Li–Duan).
- **Order-consistent for flexible f** under A1–A2 alone: the swap argument shows its population minimiser preserves the ordering of f (unlike the general case in Duchi et al. 2013, because A1–A2 imply item dominance).
- **Per-cell temperatures** remove the scale part of cross-cell heterogeneity and weight noisy cells down.

**4.3 Stability-restricted versions.** Both estimators are fitted on the restricted parameter space of Section 3: on the null space of the link gaps in the linear case (exact global solutions when one free direction remains), with a penalty for a network.

**4.4 Pilot evidence (to be replaced by full runs).** Stability cut index error 3–50× for every estimator. Where both are valid, RankNet is about 1.3–1.6× more efficient than SUR. Under X-dependent skewed noise (outside A2), SUR and exact MRC are far less biased than RankNet.

## 5. Second step: maps, outcome changes and effects

Given the estimated index ŝ = x'θ̂, the city-date maps are estimated cell by cell, which gives everything the user needs in outcome units.

- **Estimator.** Per cell, a monotone regression of W on ŝ: nonparametric (isotonic) with many tracts per cell, or a low-dimensional monotone family with few (e.g. the 3-coefficient head; L1 for the conditional median, L2 for the mean). Follows Han (1987), Section 4(A).
- **Consistency (short proof).** θ̂ consistent, m_ct Lipschitz in the index, calibration samples per cell growing:

```latex
\big|\hat m_{ct}(x'\hat\theta) - m_{ct}(x'\theta)\big| \le \big|\hat m_{ct} - m_{ct}\big| + L\,\|x\|\,\|\hat\theta - \theta\|
```

- **Estimated index.** With a nonparametric step 2, the first step is faster and does not affect step-2 inference to first order. With a parametric step 2, correct it (Murphy–Topel) or bootstrap both steps.
- **Outcome changes.** ΔŴ = m̂_c1(ŝ₁) − m̂_c0(ŝ₀), the change in expected (or median) outcome. This is the change target of the paper.
- **Effects in outcome units.** If a randomised D enters the index with coefficient β_D, the effect on W is an average structural function contrast, not β_D:

```latex
\mathrm{ATE}_{ct} = E_x\big[\, m_{ct}(x'\theta + \beta_D) - m_{ct}(x'\theta) \,\big]
```

It is identified in W units although β_D is identified only relative to the normalisation, because m_ct absorbs the scale. It varies by city-date; report the cell-level effects and their average (Blundell–Powell average structural function; Chernozhukov et al. on effects in nonseparable panel models).

- **Pilot evidence.** With 20 calibration tracts per city, outcome-level errors are dominated by calibration noise. L1 heads were slightly better than L2 heads in most scenarios.

## 6. Inference

Confidence intervals come from the linear-index estimators through a city-cluster bootstrap. A Hausman-type contrast between RankNet and SUR tests RankNet's extra assumption.

- **Pre-specified estimators, not selection by MSE.** MSE is unobservable in data, and choosing by it is model selection that invalidates naive intervals. RankNet is the primary estimator; its interval is valid for its pseudo-true value, which equals θ under its conditions. SUR is the robust estimator.
- **Hausman-type test.** Under A1–A2 plus RankNet's link condition both are consistent and RankNet is more efficient; under link misspecification only SUR is consistent. The contrast between them, standardised with a bootstrap covariance, tests the extra condition. In the pilot it would flag the X-dependent-noise scenario.
- **City-cluster bootstrap.** Resample cities. This is robust to any dependence within a city (spatial, serial) if cities are many and roughly independent. Bootstrap validity rests on both estimators being smooth √n estimators; the Abrevaya–Huang (2005) failure applies to cube-root estimators only.
- **Few cities.** Use a spatial block bootstrap or Conley-type variance within cities; flag the US-scale data as what makes city-clustered inference credible.
- **Estimated restrictions.** When the links are noisy, re-estimate the restriction inside each bootstrap draw.
- **Second step.** Bootstrap both steps together for predictions, ΔW and effects.

## 7. Neural-network implementation

The network replaces x'θ with a flexible f; the paper states one proposition for it and keeps all inference on the linear index.

- **Training.** The loss is a U-statistic over within-cell pairs: minibatches sample cells, then pairs. Fix the scale by normalising f to unit variance on a reference batch, so the bandwidth or temperature means something. Anneal the bandwidth for SUR.
- **Stability.** Imposed as a penalty that tightens over training (or an augmented Lagrangian), since a hard null-space constraint is not available for a network.
- **Proposition.** Under A1–A2 and a rich enough sieve: (a) the population minimiser of the RankNet or SUR loss orders units as f does (identified up to g); (b) the two-step prediction m̂_ct(f̂(x)) is consistent for the composite m_ct(f(x)). Route: sieve extremum estimation for (a); the Section 5 argument for (b).
- **Tools to check.** U-process ranking theory (Clémençon, Lugosi & Vayatis 2008), network M-estimation rates (Farrell, Liang & Misra 2021), and generalisation benefits of invariance for the stability penalty.
- **Out of scope.** Rates and uncertainty quantification for the network (or only via sample splitting).

## 8. Monte Carlo design

Three experiments, each reporting MSE of f, W, Δf and ΔW (oracle-map and calibrated), plus bias/variance splits and bootstrap coverage. The headline is one feasible-sample case where the proposed estimator beats FE/RE through lower bias and unrestricted MRC through lower variance.

| Experiment | Question | Design | Arms |
| --- | --- | --- | --- |
| E1 Core comparison | Does the stability-restricted rank estimator win at realistic sizes, and against what? | Linear f, skewed X; maps: additive, affine, convex/concave by city; noise inside, outside, city-varying scale/shape; small cells (8×10, 32×40) and an anchor (64×250) | RNC, SUR (with/without stability); MRC with/without stability; FE, RE/mixed model, correctly specified ML; network regression on W with city effects (practitioner baseline) |
| E2 Imperfect links | How robust is the gain to bad link information? | Fake (changed units declared stable), missing, noisy links; temporal-only vs cross-city-only; how many nuisance directions links span; finite penalty and its selection | Same estimators, hard constraint vs penalty |
| E3 Dependence | What do persistence and spatial correlation cost, and can they be exploited? | Tract persistence ρ ∈ {0, 0.8}; Gaussian random field over tract locations (2 ranges); non-stationary variance as the boundary case | RNC, SUR, pairwise panel maximum-score term, distance-weighted pairs; cluster-bootstrap coverage |

Notes: separate the two contributions by including MRC with stability (shares the restriction, not the estimator). Report where correctly specified ML wins. Existing pilots: `minimal_e1` rounds 1–3.

## 9. Empirical application

Aerial imagery (NAIP) and ACS tract income: estimate the index on training cities, calibrate maps with few labelled tracts in held-out cities, and predict levels and changes.

- **Validation.** Whole-city holdouts and a temporal holdout year; within-cell Kendall τ (head-free), calibrated L1/L2 error on W, and ΔW against ACS changes. Compare with FE, mixed models and a standard network regression on standardized income.
- **What a stability link is empirically.** Tracts or buildings with no construction between dates, and acquisition differences (season, sensor, flight line) as the nuisance the links remove. Document how well the links hold.
- **Diagnostics for A2.** RankNet–SUR (Hausman) contrast; within-cell residual size and skew against covariates. ACS sampling error scales with tract population, which imagery may reveal: the main empirical threat.
- **Effects.** If a policy shock enters the index, report average structural effects in income units by city.

## 10. Extensions

Each extension is labelled by what it touches: identification, efficiency, or inference only.

- **Spatial correlation (inference and efficiency only).** Stationary spatial fields keep pairs exchangeable, so identification holds; city-wide shocks are absorbed by the map. It affects variance and inference: handle with the city-cluster bootstrap, or a spatial block bootstrap / Conley-type variance with few cities, under Jenish–Prucha-type mixing. It can also be exploited: distance-weighted pairs or neighbourhood-level ranking cells difference out local shocks like a local fixed effect.
- **Spatially correlated omitted covariates (identification).** If noise depends on neighbours' characteristics, exchangeability given own X fails for every estimator. Fix for prediction: let f use neighbourhood context (larger tiles, SLX-style spatial lags of image features).
- **Serial correlation and tract effects (efficiency and inference).** Within-date ranking stays valid; information in persistence is lost (about 3× worse than ML at ρ = 0.8 in pilots). Extension: a pairwise panel maximum-score term on order flips of tract pairs across dates, 1(W_jt > W_it). The map cancels within the cell and the pair difference u_j − u_i is a fixed effect, so this is Manski's (1987) panel binary-choice model at the pair level. Needs exchangeable errors across dates.
- **X-dependent heteroskedasticity or skew (identification).** Outside A2 for every rank estimator. Detect with the Hausman contrast; SUR and exact MRC degrade least in pilots.
- **Noisy or partial links (efficiency vs bias).** Penalty instead of hard constraint; graph connectivity and spectral gap of the link network govern how link noise averages out (E2).
- **A ψ-invariant change term (efficiency).** Changes in within-cell normal scores of W; only approximately valid, exploratory.
- **Censoring and top-coding (none for ranks).** Ranks are robust to top-coded W (Han allows a non-decreasing outer map); cardinal methods are not.
- **Cardinal Δf (identification).** Needs an extra normalisation, e.g. inside noise with a known law, which pins f up to an affine transform.
- **Network rates and uncertainty (theory).** Left to future work.

## Open questions and scope

- [ ] **Scope:** one paper (linear theory + network proposition + application) or split the network and US-scale work into a second paper?
- [ ] **Stability theorem:** is the efficient-estimator / minimum-distance version enough, or do we need a finite-sample (complexity) bound as well?
- [ ] **SUR asymptotics in the large-cell regime:** bandwidth rate and undersmoothing condition; look for an existing result for clustered smoothed MRC.
- [ ] **RankNet bias size:** how large is the link-misspecification bias under realistic skewed X and outside noise (full E1)?
- [ ] **Stability gain realism:** in the pilots links remove 4 of 6 directions; rerun with fewer nuisance directions and imperfect links before claiming magnitudes.
- [ ] **Is A2 plausible for ACS income?** Check residual skew against tract population and imagery features.
- [ ] **Pairwise panel maximum-score term:** prototype on the persistence scenario; rate with many pairs.
- [ ] **Bootstrap validity reference** for √n rank estimators (a paper by Subbotin, unverified).

## Literature to position against

The stability-restriction idea comes from satellite and remote-sensing work, so those papers are the primary citations for it. Anchoring ideas in survey measurement, psychometrics, political science and price indices are related but more distant: they use known-equal units to align outcome scales, not to restrict the feature-to-index function. A November 2026 search found no prior use of known-unchanged units as restrictions on the index function in a rank or monotone-index estimator. References without a link are from memory and need checking.

- **Satellite and remote sensing (primary for stability restrictions).**
    - Schott, Salvaggio & Volchok (1988), "Radiometric scene normalization using pseudoinvariant features," *Remote Sensing of Environment* 26: 1–16: unchanged sites calibrate images across dates. Verified only through [Elvidge et al. (1995)](https://asprs.org/wp-content/uploads/pers/1995journal/oct/1995_oct_1255-1260.pdf), which describes it.
    - [Zheng et al. (2026), "A satellite foundation model for improved wealth monitoring" (Tempov)](https://arxiv.org/pdf/2604.23166): consistency loss on same-location image pairs across seasons; the nearest imagery-wealth use of same-location pairs (no ranking, no unknown maps).
    - Jean et al. (2016); [Yeh et al. (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7244551); [Rolf et al. (2021)](https://www.nber.org/papers/w28045): imagery-based economic measurement; the practitioner baseline (regression on survey labels).
- **Rank estimation and transformation models (the framework).** Han (1987), [Publication record](https://portal.mardi4nfdi.de/wiki/Publication:1099547); Sherman (1993); [Abrevaya (2000)](https://ideas.repec.org/a/eee/econom/v95y2000i1p1-23.html); [Manski (1987)](https://econometricsociety.org/publications/econometrica/1987/03/01/semiparametric-analysis-random-effects-linear-models-binary); [Botosaru, Muris & Pendakur](https://arxiv.org/abs/2008.05507); Cavanagh & Sherman; Khan & Tamer; Chen (2002). Why rank estimators faded: [Chen (2010)](https://ideas.repec.org/a/eee/econom/v159y2010i1p222-234.html) (slow rates in fixed-T panels), [Abrevaya & Huang (2005)](https://doi.org/10.1111/j.1468-0262.2005.00613.x) (bootstrap failure for cube-root estimators), [Shin & Todorov (2021)](https://arxiv.org/abs/2009.03844) (computation).
- **Learning to rank.** Duchi, Mackey & Jordan (2013); Clémençon, Lugosi & Vayatis (2008).
- **Effects and structural functions.** Blundell & Powell (average structural function); Chernozhukov et al. (nonseparable panel models); Ruud; Li & Duan.
- **More distant analogues: anchors that align outcome scales.**
    - Anchoring vignettes: [King, Murray, Salomon & Tandon (2004)](https://gking.harvard.edu/publication/enhancing-the-validity-and-cross-cultural-comparability-of-measurement-in-survey-research/); [King & Wand (2007)](https://econpapers.repec.org/RePEc:cup:polals:v:15:y:2007:i:01:p:46-66_00); [Greene, Harris, Knott & Rice (2019)](https://www.york.ac.uk/media/economics/documents/hedg/workingpapers/1918.pdf) (tests of the anchoring assumptions).
    - IRT linking with anchor items: [Kolen & Brennan (2014)](https://www.springer.com/978-1-4939-0316-0); [Xu, Douglas & Lee (2009)](https://rd.springer.com/chapter/10.1007/978-0-387-98138-3_15); [Lee, Joo & Stark (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5978619).
    - Bridging in ideal-point estimation: [Bailey (2007)](https://ideas.repec.org/a/wly/amposc/v51y2007i3p433-448.html); [Shor & McCarty (2011)](https://ideas.repec.org/a/cup/apsrev/v105y2011i03p530-551_00.html); [Treier (2011)](https://researchportalplus.anu.edu.au/en/publications/comparing-ideal-points-across-institutions-and-time/).
    - Repeat-sales indices: Bailey, Muth & Nourse (1963), not opened; [Case & Shiller (1987)](https://www.nber.org/papers/w2393).
    - Machine learning: [Arjovsky et al. (2019), Invariant Risk Minimization](https://arxiv.org/abs/1907.02893); [Wagstaff et al. (2001), must-link constraints](https://www.wkiri.com/research/papers/wagstaff-kmeans-01.pdf).
