# E6 — Can outcome levels improve latent change magnitudes?

E6 is separate from E1–E5: `run_e6.py`, `ordinal_mc/e6/`, and the two
`configs/e6_*.json` files. It does not append to or overwrite previous experiments.
Small two-city implementation fixtures exercise the code only; they are not
Monte Carlo performance results. Run the eight-city smoke design before scaling
to the replicated main design.

## Run it

From the `src` directory, use your existing Python/PyTorch environment:

```powershell
cd ordinal_mc_suite
python -m pip install -r requirements_e6.txt
python -m unittest discover -s tests -p 'test_e6*.py' -v

# Inspect the full resolved specification and coefficient ranges without fitting.
python run_e6.py --config configs/e6_main.json --out e6_main --plan

# Eight-city smoke run: both truths, both maps, both heads, full stopping rules.
python run_e6.py --config configs/e6_smoke.json --out e6_smoke --workers 2

# Larger smoke: 512 labeled tracts per city, one replication of both cases.
# Each city-year has 205 fitting and 51 validation observations.
python run_e6.py --config configs/e6_smoke_512.json --out e6_smoke_512 --workers 2 --resume

# Main design: 30 paired replications at 32 and 64 fitting cities.
python run_e6.py --config configs/e6_main.json --out e6_main --workers 4

# Resume after interruption; a completed paired job is never retrained.
python run_e6.py --config configs/e6_main.json --out e6_main --workers 4 --resume

# Generate a report from completed jobs, including an interrupted/partial run.
python run_e6.py --config configs/e6_main.json --out e6_main --report-only

# Refresh reporting after a renderer fix or in a different numerical environment.
# Reads completed jobs only; writes to a NEW folder; requires no PyTorch.
python render_e6.py --run e6_smoke --out e6_smoke_review
```

The same commands work in a Linux/macOS shell after changing into the suite.
Fitting resume remains pinned to the original source/runtime. The separate
`render_e6.py` command can refresh completed results after reporting-only fixes
without changing their fitting manifest or checkpoints.
Use `--workers 1` for serial execution or a larger number for more concurrent
paired jobs. Each worker uses one numerical thread, including BLAS libraries.
Worker count may change on resume; design, sources, numerical-library versions,
and numerical thread count must match. Use one coordinator per output directory.
Completed jobs are committed to SQLite by the parent process. An interrupted
incomplete job restarts from its beginning. The main design has 120 paired jobs;
each includes both measurement maps and both fitted heads, with 12 joint
candidates. No runtime estimate has been inferred from the tiny tests.

Choose new output names for changed specifications. `--report-only` rebuilds
exports without fitting; a partial report states its completion count and does
not claim a completed replicated finding. Full configuration and numerical
versions are saved in `manifest.json` **before any fitting**.

## DGP and matching

Let `u = H @ rotation`. The original E5 structural functions are

\[
f_{lin}(H)=(u_0+0.6u_1-0.4u_2-\mu_{lin})/\sigma_{lin},
\]
\[
f_{quad}(H)=[u_0+0.7u_1u_2+0.4(u_2^2-1)-\mu_{quad}]/\sigma_{quad}.
\]

E5's independent population calibration fixes the structural within-cell
pair dispersion to one. Its 100,000 calibration draws, rotation, and orthogonal
input mixing use fixed function seed 983. Seven mixed inputs contain three
structural features, two acquisition features, and two city nuisance features.
Only the privileged ML reference inverts the known mixing. Ordinary scorers use
fit-only standardized mixed inputs. The main flexible scorer has ReLU hidden
layers of widths 32 and 16, with the original bounded effective weights.

`R = f(H) + epsilon`, where
`epsilon = noise_sd_effective * sqrt(6)/pi * (Gumbel(0,1) - EulerGamma)`.
As in E5, the default `bayes_target=0.2` calibrates the noise scale separately
for each structural truth; `noise_sd=0.7` is the fallback if that target is null.
The actual effective scale is saved in each replication's design metadata.

For standard-normal city contexts Z and dates t in {0,1},

\[
a_{ct}=3Z_{c0}+t,\quad b_{ct}=1.5+\exp(0.2Z_{c1}),\quad
\kappa_{ct}=0.6+0.2\tanh(Z_{c0}+0.5t).
\]

Affine truth is `W=a+b*R`; nonlinear truth is
`W=a+b*(R+kappa*tanh(R))`. Both are strictly increasing. The latter changes
curvature without requiring logs or positive outcomes. Theoretical ranges are
`a` unbounded with Gaussian moments, `b>1.5` with lognormal moments, and
`0.4<kappa<0.8`. All maps have finite second moments. The preflight main-design
ranges over fitting/seen cells are approximately:

| Coefficient | Minimum | Maximum |
|---|---:|---:|
| a | -10.5836577431 | 10.6177259227 |
| b | 1.9946178102 | 3.5995327504 |
| kappa | 0.4003446695 | 0.7997585171 |

The machine-readable exact ranges for fitting and unseen cells are regenerated
by `--plan` and saved before fitting in the run manifest. These ranges enumerate
the predetermined seeds, rather than selecting favorable seeds. Affine and
nonlinear truths share the exact same contexts, coefficients, H, X, errors,
tract split, links, and test panels. Within-cell orderings therefore coincide,
and A is fitted just once per structural-truth/replication/city-count job.

Each city has 32 labeled tracts, alternating observation dates. Original tract
IDs are allocated before the E5 stratified split. The 20% target is rounded
within city/date (three of sixteen tracts per cell with defaults, or 18.75%
overall). Each labeled tract has one observed date. All additional dates/views
are unlabeled, generated only from fitting tracts; none of a held-out tract's
views enters training. There are two exact temporal links per fitting tract,
with weight 0.3 and no cross-city links, identically across A–D.

The E5 change generator moves half the evaluation tracts in an equally likely
upward/downward local structural-gradient direction, clips to the structural
support, and redraws acquisition nuisance. Step sizes are 0, 0.15, 0.4, 0.8.
Structural change is calculated after clipping, so actual sizes can differ from
the nominal step. Seen and unseen panels are independent; a third independent
pure-stable panel calibrates detection. Dates' outcome errors are independent,
and those errors are matched across maps, arms, and step sizes.

## Objective, heads, and selection

The scorer criterion reuses `core.score_objective` without altering its weights,
raw gaps, or derivatives:

\[
L_{E5}=\sum_{ij}q_{ij}\log[1+\exp(-y_{ij}(s_j-s_i))]
+0.3\,S(s)/V_{fit}(s)+\alpha J(s),
\quad V_{fit}=\tfrac12\sum_{ij}q_{ij}(s_j-s_i)^2.
\]

All ordered distinct same-cell fitting pairs have
`q_ij=1/[N_fit*(n_cell-1)]`. Exact observed ties are excluded from the logistic
term but retained in V. S is the average squared raw score difference across
the fixed exact temporal links. V is **differentiated**, not detached or replaced
by an epsilon. Complexity is the existing squared path norm; default alpha=0
matches the E5 paper preset. Nonzero alpha can be specified in a new config.

The calibrated objective is

\[
L_{joint}=L_{E5}+\lambda_W\operatorname{mean}_{fit}
[(W-G_{ct}(s))^2]/\operatorname{Var}_{fit}(W)+0.01J_G,
\quad \lambda_W\in\{0.1,1,10\}.
\]

G receives **only one scalar score and a cell ID**. It never receives structural
features, hidden activations, H, R, true coefficients, or true changes. The affine
head has a positive softplus slope. The continuous piecewise-linear monotone
spline has three fit-score quantile knots (20%, 50%, 80%) and four positive
segment slopes, allowing slope increases and decreases. Score/response scaling
and knots use fitting observations only, with score normalization/knots fixed
at A even when C moves the scorer. Its penalty is mean squared slope jumps
(zero for affine) plus 10 times partially pooled cell intercept/raw-slope
deviations. A cell's pooling multiplier is `16/n_fit_cell`, giving smaller
cells more pooling. Both heads pool cell intercepts and slopes; the global
coefficients are learned. Knot locations are common across cells; coefficients
are cell specific. Positivity holds also outside the knot range.

| Arm | Fitting and checkpoint choice |
|---|---|
| A, ordinal | Three seeded starts. Up to 3,000 Adam steps at 0.01 and 400 L-BFGS steps in 10-step chunks. Select checkpoints and restart by observed validation rank logistic loss. |
| B, frozen | For each lambda, fit only G from A's detached fitting scores for at most 600 Adam steps at 0.02. A's scorer remains bitwise unchanged. |
| C, joint | Initialize from the corresponding B candidate; unfreeze the scorer. At most 600 Adam steps, scorer LR 0.001 and head LR 0.01. |
| D, continuation | Start from A with a fresh Adam optimizer at 0.001. Execute exactly the selected C candidate's additional scorer-update count, without level loss. Restore the best validation-rank checkpoint, including A. Then calibrate a head post hoc without scorer updates for outcome-scale comparisons. |

Check validation every ten updates, with patience 20 checks and min_delta=1e-5
(L-BFGS patience ten checks). Always retain the actual minimum validation loss;
min_delta affects patience, not which minimum is retained. Initial checkpoints
are eligible. Restore the best checkpoint after every stage. No final
unvalidated optimization audit is performed. D runs its matched budget without
patience truncation and still restores its best checkpoint; both executed and
selected update counts are recorded. Restarts are shared via A's selected
checkpoint, with no new random scorer initialization in B–D.

For B and C the **same fixed validation rule** is
`validation_rank_logistic + validation_level_MSE/Var_fit(W)`, with weights 1 and
1 independent of the training lambda. B and C are separately selected across
the predeclared grid using that rule, with deterministic lower-lambda tie
breaking. Each C candidate starts from its same-lambda B, and the selected C's
starting B is additionally scored under the common rule in `selection.csv`.
B and C receive identical labeled observations. There is no labeled change
loss, no test/oracle-latent tuning, and no seed selection. D's budget comes from
the observable-selected C candidate, including C's patience tail, not just its
best checkpoint step. `D_common` is an additional checkpoint diagnostic from
the same ordinal trajectory, selected under the common rule using C's frozen
starting B head; it adds no level gradients.

## Privileged references and targets

`oracle_ML` reuses the existing correctly specified observation likelihood.
It knows the structural basis, input mixing, measurement inverse, centered
Gumbel law and noise scale, and estimates coefficients on fitting observations.
Its optimizer status and gradient are reported rather than silently dropping
failed numerical convergence. Its optional reported-scale prediction applies
the known measurement conditional mean to the fitted structural index.

`oracle_noisy_R` uses the **same scorer architecture** as the ordinary arm,
initialized at A, trained on the known inverse `Psi_ct^-1(W)=R` with a direct
standardized noisy-R squared loss plus the same stability/complexity terms.
It never trains on noise-free f(H). It uses its own privileged validation noisy-R
MSE and the same 600-step ceiling. It is a diagnostic of scorer capacity and
inverse-label access, not an available empirical estimator or a guaranteed
upper bound. Its objective replaces the rank term, and it has no fitted G head.

1. **Latent change:** for every scorer, including both privileged references,
   `delta_hat=(s(X1)-s(X0))/sqrt(V_fit(s))` is compared to `f(H1)-f(H0)`.
   Report RMSE and sign accuracy on actual changes above 0.05, and mean
   predicted/true changes by direction and nominal/actual size. Actual absolute
   size bins (indices 0–3) are (0.05,0.15), [0.15,0.3), [0.3,0.6), [0.6,infinity).
   Direction accuracy gives a tied predicted change half credit, as in E5.
   Pure stable tracts are `assigned_changed=False`, reported separately
   as RMS movement, so genuine tiny changes are not mixed into that measure.
   At nominal step zero all tracts are stable.
   A/B equality of latent metrics is enforced automatically. Population V(f)=1
   does not force sample V_fit(f)=1: `true_f_sd_fit` and
   `normalization_only_rmse` explicitly show this finite-sample convention.
2. **Detection:** set a model-specific cutoff to the 95th percentile of absolute
   normalized predictions on the independent stable calibration panel, using
   E5's `method='higher'` empirical quantile. Apply
   it unchanged to fresh stable/changed panels. Report cutoff, fresh pure-stable
   FPR, and sensitivity by actual true size. Lower stable movement can improve
   sensitivity by lowering the cutoff without improving the change response;
   this is distinct from more accurate latent magnitudes.
3. **Observed scale (seen cities only):** report level RMSE and predicted change
   `G_c1(s1)-G_c0(s0)` RMSE against observed `W1-W0`, plus a zero-change benchmark.
   Also compare levels and changes to the oracle conditional mean
   `m_ct(H)=E_epsilon[Psi_ct(f(H)+epsilon)]`. The affine mean is exactly
   `a+b*f(H)`; the nonlinear mean integrates tanh using 128 Gauss–Legendre
   uniform-quantile nodes and uses exact zero mean for the linear noise term.
   A test checks 128/256-node agreement. G targets `E[W|s,c,t]`; this need not
   equal the structural Psi, and `Psi(f)` is generally not the noisy nonlinear
   conditional mean. The reported oracle mean benchmark conditions on H.
   Cell-specific heads are unavailable zero-shot in unseen cities, where only
   latent/detection metrics are reported. No pooled/few-label result is implied.
   A tract with unchanged f can still have a nonzero observed or conditional-mean
   W change because measurement coefficients change across dates, in addition
   to independent outcome noise. That does not count as latent change.

## Outputs and interpretation

`report.md` leads with `latent_change.png` (also PDF): true versus predicted
mean latent change by direction and step size, with a 45-degree line, ordinal,
both joint heads, and both clearly labeled privileged references. This lead
figure uses unseen cities at the largest fitting-city count. Error bars are
pointwise 95% t intervals across replications, not individual prediction bands.

The output folder also contains:

- `manifest.json`, `resolved_config.json`: frozen design, source hashes,
  numerical versions, and exact coefficient ranges recorded before fitting.
- `results.sqlite`: parent-owned durable resume ledger, one complete paired
  job per transaction.
- `per_replication.csv`, `change_strata.csv`: all arm/map/head/city/truth/size
  results, including cutoffs, FPR, individual RMSE and sign accuracy.
- `summary.csv`, `strata_summary.csv`: means and pointwise t intervals, treating
  independent replications as the sampling units.
- `paired_differences.csv`: **C–B, C–D and C–D_common**, matched within
  replication and condition, with 95% intervals across replications.
- `selection.csv`, `validation_history.csv`, `checks.json`: observable candidate
  selection, stopping paths, matched update counts and implementation checks.
- `replications/<job>/result.json`, `checkpoints.pt`, `changes.npz`: per-job
  results, A and candidate/selected model states, fit-only feature transforms,
  and individual normalized predicted/true test changes. Checkpoint head
  state includes its fit-only normalizations and knots.

Read the latent RMSE and direction/size means first, then detection jointly with
stable RMS and its calibrated cutoff, then observed-scale errors against both
W and its noise-integrated mean. The report gives separate conclusions for
latent magnitudes, detection and reported outcome changes. It compares C with
both B and D directly; it does not promote improved outcome prediction alone
as evidence of improved latent cardinal recovery. Pointwise intervals do not
adjust for the multiple conditions/metrics. No empirical conclusion is supplied
before running the replicated design.

Implementation verification uses standard `unittest`, including matched-map
draws, tract/view separation, exact link gaps, monotonicity with changing spline
curvature, fitting-only knots, nonzero level gradients into the scorer,
zero-level-weight preservation of E5 gradients, differentiated S/V, bitwise
frozen-score equality, fixed observable selection, best-checkpoint restoration,
exact C/D update budgets, affine oracle means, nonlinear quadrature, resume
guards, replication-level intervals, exports, and a tiny quadratic/ReLU job in
a spawned worker. These checks do not estimate Monte Carlo performance.
