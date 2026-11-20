# ML and matched-tract change extension — November 2, 2026

Validated on Windows, Python 3.13.7, CPU PyTorch 2.15.0, float64.
The temporary dependency environment was isolated from the user's Python setup.

## Executed

- **40 tests passed** across `tests/test_mc_likelihood_change.py`,
  `tests/test_e1_additive.py`, and `tests/test_mc_presentation_figures.py`.
  These cover density identities against SciPy, Gaussian ML versus exact least
  squares, Gumbel coefficient recovery, all four true structural bases,
  noiseless/unbounded-likelihood cases, no fitting on stored latent/test truths,
  independence of the change cutoff from test data, matched stable controls,
  positive/negative changes, contaminated training links, all-experiment ML jobs,
  undefined metrics, paired-gain orientation, additive-exact ML summaries, the
  separate rho benchmark, CSV output, and plots.
- `python run.py --check`: all existing objective identities, gradients, sampling,
  transformations, calibration, link and resume-guard checks passed.
- `python verification/check_dispersion_recovery.py`: actual PyTorch dead-start,
  Adam-collapse and L-BFGS-probe recovery checks passed.
- `python verification/check_multicore.py`: actual serial/two-worker records
  matched for linear and NN fits including ML; changing to three workers on
  resume reused all jobs without duplicates.
- CLI battery with `configs/verification_ml_change.json`, all five experiments,
  and two workers: **57 selected fits**, including **19 ML fits**. Every ML fit
  reported convergence. All eight supported error families were exercised.
  CSV summaries, paired comparisons and **38 figures** were exported.
  A second invocation with three workers and `--resume` reused all 57 saved jobs.
- Separate rho CLI smoke: shared/city/oracle NN arms plus parametric ML,
  five cities and one optimization step for each NN; outputs, checkpoints,
  paired summaries and plots completed.
- Modified Python sources parsed with the Python 3.10 grammar; JSON configs
  parsed. The shell launcher was reviewed statically; Git Bash could not start
  inside the Windows sandbox, so its syntax was not independently executed.

## Saved smoke outputs

`verification/ml_change_smoke/` contains the final suite CLI outputs and
`verification/rho_ml_smoke/` contains the independent rho CLI outputs.
These are implementation checks, **not Monte Carlo performance evidence**.
The suite smoke has one replication, two cities, 16 labels per city, and a very
short training budget; 30 scorer fits failed the objective-audit stability check
and 31 had gradients above tolerance. These flags are retained in the outputs.
ML convergence is a numerical diagnostic, not a certificate of global optimality.

## Reproduce

From `ordinal_mc_suite`, in an environment with the suite requirements:

```bash
python run.py --config configs/verification_ml_change.json --experiments all --workers 2 --out verification_new
python run.py --config configs/verification_ml_change.json --experiments all --workers 3 --out verification_new --resume
python verification/check_multicore.py
python verification/check_dispersion_recovery.py
```

The full `configs/change_recovery.json` design was planned but **not run**:
5,760 selected jobs / 14,400 candidate fits including the ML jobs, across 30
replications, two fitted classes and two truths. Run it in a new output directory:

```bash
python run.py --config configs/change_recovery.json --experiments 5 --out change_recovery --workers 4 --threads 1
```

The experiment measures same-tract recovery and its sensitivity to treating
actually changing endpoints as stable. It does not establish whether the
empirical application has an estimation bug. Existing empirical data and
previous Monte Carlo result directories were not modified.
