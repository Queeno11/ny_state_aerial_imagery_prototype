# Focused E5 optimization investigation

## What was executed

One fixed sample: E5 replication 1, linear truth, 16 cities, 32 labels per city,
clean auxiliary links, feature-change step 0.4. The original architecture
([32, 16]), coefficient bound (10), common-logit objective, zero ridge, link
weights (0.3), sample splits, and evaluation panels were retained.

This was **not a Monte Carlo rerun**. There were 12 initial fits: three NN starts
and one linear control for each of the no-link, temporal-link, and both-link
arms. Subsequent diagnostics continued those fits or initialized the same NN
to the exact training-label-fitted linear function. Initial restarts were chosen
by training objective, never test truth.

Each selected random NN and each linear-initialized NN received five chunks of
200 additional L-BFGS iterations, preserving optimizer history between chunks,
followed by a separate 100-iteration audit. Two linked-arm runner-up starts were
also extended. An alternative branch used 1,000 Adam steps from the same baseline.
All 78 recorded checkpoints include training, validation and independent test
metrics; accepted L-BFGS objectives were verified to be nonincreasing. Finite,
positive score dispersion was required throughout.

## Numerical tests: 7 passed

1. Full NN loss agrees with an independent NumPy forward/objective calculation,
   and numerical directional derivatives agree with automatic parameter gradients.
2. The score-space gradient of S/V agrees with its explicit quotient-rule formula;
   omitting the denominator derivative is detectably wrong.
3. The fitted linear function embeds exactly into the same bounded two-hidden-layer
   ReLU architecture, preserving scores and criterion values.
4. The stability penalty is invariant to uniform score rescaling, while the raw
   common-logit likelihood changes as intended.
5. Repeated seeded optimizer runs are identical.
6. Continuation keeps admissible, nonincreasing accepted endpoints.
7. Change recall, direction, false positives and AUROC are invariant to a positive
   global score rescaling/shift, ruling out arbitrary score units as the cause
   of poor detection in this check.

The full-size sample additionally passed five
directional checks: maximum scaled derivative discrepancy 3.10e-10; explicit
penalty-gradient discrepancy 1.67e-16; independent forward-value discrepancy
4.44e-16.

## Main result: lowering the objective can harm recovery

Temporal links, same NN, initialized to reproduce the fitted linear control:

| Metric | Linear initialization | After 1,000 additional L-BFGS iterations |
|---|---:|---:|
| Training objective | 0.409385 | 0.018369 |
| Training pair misranking | 19.08% | 0.00% |
| Observed validation pair misranking | 17.63% | 24.47% |
| Independent latent test pair misranking | 1.93% | 18.97% |
| Changed-tract detection recall | 100.0% | 17.0% |
| Test false-positive rate | 3.91% | 5.66% |
| Fitting score dispersion V | 3.03 | 1,095,307 |

The both-link warm start similarly falls from 100% to 12.9% change recall as its
training objective falls from 0.409720 to 0.088189. Thus a good recoverer is
feasible in the NN class, but minimizing the empirical criterion can move away
from it and interpolate noisy training orderings. This is evidence of a
generalization/overfitting problem on this sample, not just insufficient time.

The random-start temporal NN's objective falls from 0.019107 to 0.004499, while
change recall changes from 20.3% to 17.8%. Its terminal gradient remains 8.83e-4,
above the 1e-5 target. Linked fits are therefore still not certified converged,
but the extra budget does not solve their change recovery here. A runner-up
start obtains 34.8% recall with a slightly worse objective (0.004876), illustrating
why selecting the lowest objective does not necessarily select the best changes.

The unlinked NN has zero training ranking error and already passes the gradient
tolerance. Extra Adam optimization reduces its objective further (6.95e-7 to
3.89e-8), with recall remaining around 4–5%. This confirms that passing a
convergence check is insufficient to establish useful generalization.

## Interpretation and practical follow-up

No error was found in the objective or gradients checked here. The results
separate incomplete convergence in linked NN fits from a distinct tendency of
the unregularized empirical objective to reward training interpolation and very
large score dispersion. Uniform scaling alone cannot reduce S/V; the observed
behavior also changes the fitted function, not just its units.

Observed validation errors already flag the degradation along the linear-start
paths, without access to latent test truth. `selection_diagnostic.csv` compares
training-objective selection with validation-checkpoint selection on the recorded
paths; it is a diagnostic, not a separately validated new estimator.

A subsequent sensitivity experiment can examine validation-based early stopping,
capacity or regularization. Such changes must be labeled as changes to the
estimation procedure, rather than silently presented as optimizer fixes. These
single-sample tests do not prove a global optimum or establish performance across
all samples, and they do not isolate an empirical-data-only explanation.

## Artifacts and reproducibility

- `trace.csv`: all optimizer paths and metrics.
- `restart_comparison.csv`: the three random starts for each arm.
- `gradient_checks.json` and `verification.json`: numerical check results.
- `checkpoints/`: baseline, linear-start, and final model states; L-BFGS states
  are also saved for the continued paths.
- `optimization_paths.png` / `.pdf`: objective and performance trajectories.
- `metadata.json`: complete configuration, data seed, software versions and
  original source/result metadata.

Executed in CPU float64 with Python 3.13.7 and PyTorch 2.15.0. The original E5
used Python 3.10.20 and PyTorch 2.5.1. Data/sample diagnostics reproduce the saved
E5 sample, but recreated optimizer trajectories are not expected to match old
weights bit for bit across those versions. Original E5 result hashes remain
unchanged, and no existing fit or chart was overwritten.

From the source directory, using an environment with PyTorch and pytest:

```bash
python -m pytest tests/test_e5_optimization_diagnostic.py -q
python ordinal_mc_suite/verification/diagnose_e5_optimization.py --out ordinal_mc_suite/optimization_single_run_new
```

The diagnostic refuses to overwrite a nonempty output directory.
