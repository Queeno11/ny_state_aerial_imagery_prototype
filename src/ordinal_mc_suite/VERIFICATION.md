# Verification of the paper-aligned revision

Verified on CPU, float64, with Python 3.12.14, PyTorch 2.14.1+cpu, NumPy 2.3.5, and Matplotlib 3.10.8. GPU execution was not tested.

## Formula and design checks

`python run.py --check` passed direct comparisons against the manuscript equations:

- Common-logit softplus equals binary cross-entropy on the raw margin, with no fitted common scale.
- Heterogeneous-logit binary cross-entropy uses the explicitly V-normalized score and positive cell scales.
- Temporal and cross-group penalties enter separately with their stated coefficients.
- Tract-uniform ordered-pair weights sum to one and each source tract receives weight 1/N.
- Finite-difference gradients for all three ranking objectives and the level benchmark, including jointly fitted cell intercepts.
- The common-logit cross loss changes under score rescaling, while its stability penalty does not.
- Positive Q1 labels and exact numerical inverse recovery for every transformation stage.
- Rank-label invariance under strictly increasing measurement changes.
- Exact link equality, bounded H drift, upward systematic false links, and fitting/validation separation.
- With acquisition strength zero, exact temporal links have identical X.
- Independent baseline Bayes-risk calibration across four truths and two composition settings.
- Joint tuning includes either or both link coefficients equal to zero.
- Incompatible checkpoint signatures are rejected.

## End-to-end checks

Run 1:

```bash
python run.py --config configs/verification_all.json --experiments all --out verification/all_experiments
```

Completed **414 selected fits and 534 candidate fits**, covering all four experiments, linear and NN scorers, all three ranking losses, every implemented default benchmark, two composition settings, all nuisance controls, and all link arms. This uses a linear truth and C=4; other true functions are checked in the population calibration and remain available in the presets. Runtime was about 78 seconds on the hosted CPU, including plots. This is not a runtime forecast for larger runs.

Run 2:

```bash
python run.py --experiments 3 --models basis --truths linear --city-grid 3 6 --n 16 \
  --reps 2 --restarts 2 --epochs 20 --lbfgs-steps 30 --audit-steps 15 \
  --out verification/group_curves
```

Completed **144 selected fits and 288 candidates**, exercising increasing group counts, two independent repetitions, two optimization starts, all three losses, four arms, and three nuisance cases. Runtime was about 19 seconds. These repetitions test aggregation and plotting; they do not estimate precise Monte Carlo effects.

`python verification/review_outputs.py` independently reconstructs candidate selection and paired means/SEs. It confirms finite errors, logistic-only difference MSEs, invariance of rank fits to measurement transforms, unchanged unlinked results under link contamination, and correct joint weight selection. The completed all-experiment run was resumed without repeating fits or duplicating rows.

## Optimization limitations

These are intentionally low-budget execution checks. The all-experiment run flags 238/414 selected fits for further objective improvement during the audit and 239/414 for non-small raw gradients. The basis/group-curve check flags 136/144 and 132/144 respectively. The code reports these diagnostics explicitly. Successful execution does not mean the fits adequately approximate the global training minimum.

Use larger budgets, multiple starts, bandwidth sensitivity and an explicitly labeled regularized-baseline sensitivity before paper conclusions. The theorem's approximate-minimizer requirement and heterogeneous-scale identification are not established by these tests.

The source hash and run metadata in each output folder refer to this revised implementation. Original version-1 results are not included or mixed with these results.
