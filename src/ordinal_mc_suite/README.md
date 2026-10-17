# Zero-dispersion recovery fix

The optimizer may produce a constant ReLU score (V=0), particularly with strong
path-norm regularization. V=0 remains outside the ranking criterion's domain.
This update does not add an epsilon, clip V, change the loss, or drop simulation
replications. It changes optimizer failure handling:

- Inadmissible initializations are retried with deterministic seeds, up to ten.
- An Adam step that collapses dispersion restores the best valid checkpoint
  (or valid initialization if the final-temperature phase has not started),
  then hands that checkpoint to refinement.
- An inadmissible L-BFGS trial/endpoint restores the checkpoint before that
  refinement. A failed audit probe is flagged instead of reporting a clean audit.
- A restart without any admissible initialization is recorded as inadmissible.
  Other restarts/grid points can still be selected; they use the same objective
  and validation rule. If all grid points have no admissible restart, the job
  fails explicitly instead of dropping the replication or inventing a result.
- candidates.csv and selected_fits.csv report initialization_retries,
  adam_dispersion_recoveries, lbfgs_dispersion_recoveries, optimizer_status,
  recovered_candidates and inadmissible_candidates where applicable.
  diagnostics.json reports recovered and inadmissible candidate counts.

Compatibility: both the original uploaded serial source and the first multicore
release are recognized. With identical scientific settings and numerical
runtime, existing completed jobs are retained. Only incomplete jobs are rerun.
No other scientific/source versions are silently accepted.

Extract this archive into the existing ordinal_mc_suite folder, replacing source
files. Existing output folders are not in the archive and are not overwritten.
Then resume with: bash run_overnight_multicore.sh 8
The batch skips saved work and proceeds to later stages as usual.

Tests executed here (PyTorch is not installed): actual process scheduling,
SQLite checkpoint/resume/migration, and optimizer control-flow regression tests
with deterministic backend fixtures. The latter cover dead initialization,
Adam interior/final-step collapse, L-BFGS invalid probes and endpoints, truthful
audit flags, failed-restart accounting, and unchanged valid-fit behavior.
All source files parse as Python 3.10, matching the user's environment.

An optional real-PyTorch regression test, which deliberately injects collapsed
steps, is included: python verification/check_dispersion_recovery.py
It does not access or modify existing results. It was not executed here.

The five-fold validation upgrade is separate and is not part of this repair.
Existing selection settings stay unchanged.

# Multicore update

Use `--workers 4 --threads 1` to run four independent CPU fits concurrently.
The default remains one worker. `--threads` is the number of numerical threads
per worker, not the number of independent fits. The parallel scheduler uses
spawned processes on Linux, WSL and Windows. CUDA requires `--workers 1`.

For the complete presentation battery, activate `torch_geo_env`, then run:

```bash
bash run_overnight_multicore.sh 4
```

Start with four workers. You can also use `bash run_overnight_multicore.sh 6`
or set `WORKERS=6`. More workers require more RAM and do not guarantee linear
speedups. Each worker constructs its own small DGP and keeps one design cache.
NumPy/BLAS and PyTorch thread counts are limited per worker to avoid
oversubscribing the machine. Candidate grids and restarts inside one fit stay
serial; independent selected-fit jobs run in parallel. The three experiments
run sequentially so they share the chosen CPU budget.

Stop any currently running serial batch with Ctrl+C before updating the code
or launching this runner against the same output directories. Extract this
archive into the existing suite folder, overwriting source files and the
three included overnight configs. It does not contain old result folders.
The new launcher automatically adds --resume when a matching checkpoint
exists. It retains completed jobs and reruns only jobs not yet saved. The
parent process alone writes SQLite and exports result CSVs. Progress lines
can arrive in a different job order; the estimator, data seeds, losses and
selection rules are unchanged.

Worker count is an execution setting and may change on resume. Other config
changes, including threads, dtype, device, training budget and seeds, remain
guarded. Checkpoints from the exact serial release uploaded in this conversation
(source SHA256 25190dbdd3174ebf4f890cbfd4690ea056f78bf2b225b528e888aa6a97806f2e)
are accepted after verifying all scientific/runtime settings. Their original
signature is retained in checkpoint metadata. Unknown source versions or
incompatible settings are refused rather than mixed.

For an individual run, append `--workers 4 --threads 1` to its existing command.
To resume it, also append `--resume`. Do not use a second terminal to write to
the same output folder while another runner is active.

The bundled configs retain the previously supplied overnight battery:
measurement (E1, linear), flexible temporal comparison (E3, small NN), and
imperfect links (E4, linear), each with 30 replications and fresh simulation
seed 20261015. Existing completed experiments are not otherwise rerun.

## Verification

Executed here: real spawned-process concurrency; deterministic seeded task
outputs; bounded scheduling; NumPy/BLAS environment limits; worker exceptions
and abrupt crashes; cleanup on cancellation; and real runner/SQLite/CSV
integration using a deterministic fitting fixture. The latter checks serial
and parallel records, partial-run recovery, no duplicate jobs/candidates,
worker-count changes, guarded serial-checkpoint migration and rejection of
incompatible resumes. All passed. Python and shell syntax checks also passed.

This environment has no PyTorch, so actual numerical fits could not be tested
here. An optional real-PyTorch check is included:

```bash
python verification/check_multicore.py
```

It runs tiny linear/NN fits serially and with two workers, compares fit and
candidate records excluding elapsed times, and verifies a worker-count change
on resume. It uses a temporary directory and never touches existing results.

The existing serial README and verification reports below describe the
original numerical implementation; they do not certify this process update.

# Ordinal Monte Carlo suite — paper-aligned revision (2.0)

This revision follows the supplied October 5 proposal and longer technical note. Read `PAPER_ALIGNMENT.md` for the audit, corrected discrepancies, and qualifications that remain in the theory. The manuscripts themselves have not been edited.

## Run

Extract the ZIP, enter `ordinal_mc_suite`, and use your existing `torch_geo_env`:

```bash
python run.py --check
python run.py --config configs/verification_all.json --experiments all --out smoke_all
```

NumPy, Matplotlib and PyTorch are required. Install missing dependencies with `python -m pip install -r requirements.txt`; retain an existing working CUDA-enabled PyTorch build. The implementation uses vectorized NumPy/PyTorch and automatic differentiation, so Numba is unnecessary.

The verification config runs every experiment with linear and neural estimators and a linear truth at C=4. It is a smoke check, not a Monte Carlo result. Other presets include nonlinear truths and group-count curves.

For the focused temporal comparison:

```bash
python run.py --config configs/temporal_focus.json --experiments 3 --plan
python run.py --config configs/temporal_focus.json --experiments 3 --out temporal_focus
```

It uses NN estimators, four true functions, common logit, no-link/temporal arms, C=8/16/32, n=32 tracts per group, ten replications, and three starts. Ten replications are exploratory. **C, not n, is the main sample-size axis**, as in the paper.

Plan larger grids before training:

```bash
python run.py --preset pilot --experiments 1 --models nn --truths linear quadratic --reps 3 --plan
python run.py --preset pilot --experiments 2 --models nn --truths linear quadratic --reps 3 --out exp2_pilot
python run.py --preset pilot --experiments 4 --models nn --truths linear quadratic --reps 3 --out exp4_pilot
```

The main factorial is large. Start with the focused Q3 run and selected pilot cases. `--plan` performs no fitting.

## Implemented experiments

| Experiment | Varied factors | Methods |
|---|---|---|
| Q1 | Common map → multiplier → location/scale → shape; group composition; labeled group count | Three ranking losses; pooled levels; logs with cell effects; within-cell z-scores/percentiles; latent-f and inverse-transformation oracles |
| Q2 | Student tail degrees of freedom; cell scale dispersion; feature scale; feature-dependent skew | Three losses; fixed group/sample size and no links by default |
| Q3 | None/temporal/cross-group/both; separate acquisition/group-cue strengths; labeled group count | Three losses and requested model classes |
| Q4 | Exact links; small drift; random false links; one-sided false links | Each loss with unlinked, fixed-weight, and jointly tuned temporal/cross-group/both arms |

Estimators: `linear`, `basis`, `nn`. True functions: `linear`, `quadratic`, `smooth`, `regime`. True function and fitted class are separate factors. The affine observation transformation in Q2–Q4 keeps the baseline common-logit and literal additive-error median restrictions compatible. Q1 deliberately varies the transformation, using the technical note's latent-noise construction.

The common-logit cross loss is the paper's unnormalized `softplus(-Y * score_difference)`. Heterogeneous logit normalizes the score to V=1 and estimates positive cell scales. Ordinal uses an explicit smooth surrogate and evaluates exact misranking. Stability is `lambda_t*S_t/V + lambda_c*S_c/V`. Both uses both full edge sets; its weights are not halved.

## Configure

Precedence: preset → optional JSON → CLI. Unknown keys are rejected. For example:

```bash
python run.py --preset pilot --experiments 3 --models linear nn \
  --truths linear quadratic --city-grid 8 16 32 --n 32 --reps 5 \
  --losses common_logit --epochs 1000 --restarts 3 \
  --lbfgs-steps 300 --audit-steps 300 --out exp3_pilot
```

- `city_grid` / `--city-grid`: group counts for Q1/Q3.
- `cities` / `--cities`: fixed group count for Q2/Q4; the CLI `--cities` also fixes Q1/Q3 to that count.
- `n`: labeled tracts per group, split over two dates, including the validation holdout. Defaults keep n fixed.
- `lambda_t`, `lambda_c`: fixed-arm coefficients; unavailable link types receive zero.
- `lambda_grid`: tuned values for each available coefficient. Both-arm tuning takes their Cartesian product, including either coefficient equal to zero.
- `experiment3.nuisance_cases`: pairs `[acquisition_strength, group_strength]`; defaults change one factor at a time.
- `experiment1.composition_grid`: amount of between-group variation in H and f distributions.
- `bayes_target`: default 0.2 for independent calibration of the common-Gumbel baseline. Set null to use `noise_sd` literally. Q2 retains baseline scale while changing errors.
- `pair_mode`: `all` by default, exact tract-uniform ordered comparisons; `sampled` is an explicit computational approximation controlled by `pairs_per_label`.
- `weight_bound`: bound on effective scorer coefficients, default 10, defining the bounded function class.
- `tau`: smoothing bandwidth, default 0.1; a sensitivity parameter, not an error assumption.

All settings are recorded. The default `ridge_grid=[0]` implements the paper's criterion without a new complexity term. For an explicitly regularized sensitivity experiment, use `--ridge-grid 0 0.001 0.01 0.1 1`, or overlay `configs/regularized_sensitivity.json`. That changes the training estimator and should be labeled accordingly in the paper.

`--device cuda --dtype float32` is available but not GPU-verified. These small full-batch fits may be faster on CPU; float64 CPU is the reference.

## Selection and evaluation

Every arm shares fitting data, validation/test tracts, rank comparisons, and initial seeds within a replication/design. Auxiliary views are constructed only for fitting tracts. Feature preprocessing uses fitting X only. Restarts are chosen by training objective; hyperparameters by observed within-cell validation misranking. Exact validation ties favor larger ridge, then smaller total stability weight, then the temporal/cross-city weight tuple. Test f and true link gaps do not tune fits. The explicit oracle benchmarks alone use their declared oracle targets.

All metrics are evaluated on independent new tracts in seen and unseen groups:

- Latent within-cell, across-time, and cross-group misranking.
- Observed within-cell misranking and f-versus-observed-label error.
- Normalized within-cell difference MSE for logistic fits only; normalization uses fitting pairs and never an oracle regression against test f.

The oracle-label error is not a Bayes-risk claim in median-violating designs. All scores with predicted ties receive half ranking error; exact true-f ties are excluded from latent comparisons.

The main figures use labeled group count for Q1/Q3 and departure severity for Q2/Q4. Results contain both predictions' ranking quality and spacing diagnostics. Cross-group recovery is a diagnostic; the supplied theory does not automatically identify all cross-group offsets/scales.

## Outputs and resume

`selected_fits.csv` contains selected fits, test metrics, fitting/validation/pair counts, true link-gap diagnostics, and optimization flags. `candidates.csv` includes every candidate's validation score and restart/selection flags. `summary.csv`, `paired_replications.csv`, and `paired_summary.csv` report MC means and SEs across independent whole-simulation repetitions, not across dependent pairs. Positive paired gains mean lower error than the baseline. Q1 references common logit; Q2 references common logit with the same arm; Q3/Q4 reference no links with the same loss.

`figures/` contains ranking and logistic difference-MSE plots. `resolved_config.json` and `run_metadata.json` record the design and runtime. `checkpoint.sqlite` stores completed jobs atomically. Repeat the exact command with `--resume` to skip completed jobs. Only an interrupted current job is rerun. Config/source/runtime changes are rejected; use a new output directory. Do not run concurrent processes against one output directory. Model weights are not saved.

Optimization flags do not establish global optimality. Low-budget smoke fits can remain far from a stationary point or training minimum; paper results require adequate budgets and robustness checks. See `VERIFICATION.md` for executed checks.

## Limits

Full-batch ratio evaluation replaces the proposed image-scale minibatch workflow for these small synthetic experiments. One labeled tile per tract is used. This package is not an imagery encoder, an empirical standard-error procedure, or a proof of consistency/rates. Fixed smoothing, fixed classes, approximate links, and heterogeneous scale identification remain explicit theoretical qualifications in `PAPER_ALIGNMENT.md`.
