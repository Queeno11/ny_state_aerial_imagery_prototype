# Validation-selected stopping: single-sample check

Yes: checkpoint selection on a held-out validation set provides a substantially
fairer DNN evaluation than minimizing training loss for a fixed budget. This
diagnostic implements that policy and evaluates it on the same fixed sample.

## Design and stopping rule

- E5 replication 1, linear truth, 16 cities, 32 labeled tracts per city, clean
  stability links, structural change step 0.4, and the original [32, 16] NN.
- A 20% validation target is stratified by city/date using the existing splitter.
  With 16 labels per cell, rounding gives **96 validation / 416 fitting tracts**
  (18.75% actual holdout). Both old and new stopping rules use this same split.
- Auxiliary views come only from fitting tracts. Fitting and validation tracts
  are disjoint. Feature preprocessing uses fitting observations only.
- Checkpoint selection uses **observed within-cell validation binary cross
  entropy**, computed over all 192 ordered validation comparisons. It never
  reads latent f, test outcomes, true changes, or stability calibration outputs.
- Adam checks validation every 10 steps, with patience 20 checks and min_delta
  1e-5, under a 3,000-step ceiling. The minimum-loss validation checkpoint is
  retained even if its improvement is smaller than min_delta.
- L-BFGS starts from the best Adam checkpoint, checks validation every 10
  iterations, uses patience 10 checks, and has a 400-iteration ceiling. The
  minimum-validation-loss checkpoint across both phases is restored. There is
  **no subsequent unvalidated optimization audit** that changes the weights.
- Three random starts are compared using validation loss. Training criterion,
  link coefficients, coefficient bounds, architecture, and ridge remain fixed.
- A separate NN initialization reproduces a fitted linear control exactly,
  using fitting labels only; it is also validation-stopped.
- All candidate selections for all arms are finalized before test metrics are
  evaluated. This is one dataset, not a new Monte Carlo experiment.

## Results: random NN starts

| Links | Old fixed-budget rule: change recall | Validation stopping: change recall | Old latent test pair accuracy | Validation stopping: latent test pair accuracy |
|---|---:|---:|---:|---:|
| None | 9.0% | 25.6% | 77.7% | 95.1% |
| Temporal | 19.9% | 47.5% | 84.6% | 94.2% |
| Temporal + cross-city | 14.6% | 91.8% | 85.9% | 97.3% |

Old-rule restarts are selected by training objective, matching E5. New-rule
restarts and checkpoints are selected by validation loss. For separation of
these effects, `fixed_budget_val_selected` also chooses among the same fully
trained endpoints by validation: its change recalls are only 5.3%, 22.5%, and
14.5%. The gains therefore cannot be explained solely by choosing a different
restart after the original training budget; retaining earlier checkpoints matters.

The best validation checkpoints among selected random starts occurred at Adam
steps 50 (none), 130 (temporal), and 160 (both), well before the original
500-step Adam budget. The original fixed-budget model validation BCE values
were 37.51, 19.99, and 2.86, versus 0.421, 0.413, and 0.396 with early stopping,
respectively. Validation thus clearly detects deterioration without latent truth.

For the both-link arm, the test false-positive rate is 5.86% under both rules,
so its recall increase is not attributable to accepting more false positives.
The independent stable calibration panel continues to set a nominal 5% cutoff.

## Linear-start control

With validation stopping, the NN initialized from the fitted linear model
retains 100% change recall with temporal links and 92.2% with both links. The
no-link control retains 82.2%. The temporal and no-link arms select their
initial checkpoints; the both-link arm selects Adam step 10. There was no
selection on the true change outcomes.

## Verification and interpretation

Six automated tests passed: best-checkpoint restoration, patience/min_delta
semantics, disjoint tract/cell comparisons, independence from latent/test labels,
restoration of the minimum recorded validation loss, and restart selection by
validation rather than training objective. Each selected random-start validation
loss was also checked against the minimum of its saved trajectory.

Original E5 result hashes remain unchanged. Results, validation histories,
candidate decisions, selected model states and figures are saved in this folder.
The run used CPU float64, Python 3.13.7, and PyTorch 2.15.0; the old-rule controls
were rerun in that same environment and on the same holdout for a matched
comparison. These new numbers are not the old E5 aggregates.

This single-sample result supports validation-selected stopping and corrects the
interpretation of the earlier fixed-budget diagnostic. It does not establish
general performance across all E5 samples, truths or contaminated links. The
original NN curves should not be treated as evidence against a properly
validation-selected NN estimator. Early stopping is an explicit regularization/
selection choice and should be documented in future experimental protocols.

Reproduce from the source directory, with the required numerical packages:

```bash
python -m unittest discover -s tests -p test_validation_early_stopping.py -v
python ordinal_mc_suite/verification/check_validation_stopping.py --out ordinal_mc_suite/optimization_validation_stopping_new
```
