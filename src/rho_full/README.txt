Monte Carlo: does city-specific logistic calibration improve a common NN index?

QUICK START
-----------
Requires Python >= 3.10, numpy, scipy, matplotlib, and PyTorch. Install a CUDA
build of PyTorch appropriate for your computer using https://pytorch.org/get-started/locally/.
Other dependencies:  python -m pip install numpy scipy matplotlib

Default experiment (finite budget; CUDA is selected automatically):
    python rho_city_monte_carlo.py
Shorter first run:
    python rho_city_monte_carlo.py --reps 1 --steps 600 --out rho_quick
Larger Monte Carlo:
    python rho_city_monte_carlo.py --reps 10 --steps 2500 --out rho_large
Check sensitivity to large-city dominance with identical underlying seeds:
    python rho_city_monte_carlo.py --weighting pairs --out rho_pair_weighted
Optional shrinkage of city log-calibrations (zero by default):
    python rho_city_monte_carlo.py --shrinkage 0.01 --out rho_shrunk

DEFAULT DESIGN
--------------
3 replications x 2 noise scenarios x 3 estimators = 18 bounded fits, each with
1,500 optimizer steps. 120 cities; 4 years; total TRAINING observations per city
are 20, 80, 300, 1,000, or 2,000, allocated across years. The balanced default
design has 81,600 training observations. Validation/test populations are freshly
generated and equally large in every city, so small-city performance is measured
precisely without adding information to their training samples. Runtime is
hardware-dependent; increase --steps or --reps only after a first run.

DGP AND IDENTIFICATION
----------------------
* Six numerical amenities with overlapping but city-dependent distributions.
  Every city also has a 25% common-distribution mixture component. City ID is NOT
  a score-network input; it is used only for calibration and sampling.
* One strictly coordinatewise increasing cubic polynomial f(X). Its location
  and scale are fixed on a shared, unlabeled reference sample. The NN uses this
  SAME reference sample for differentiable normalization: no true f values enter
  NN training, normalization, or checkpoint selection.
* R_ict = f(X_ict) + (G_ict - EulerGamma)/rho_c, with independent standard Gumbel
  G. Differences of independent Gumbels are logistic, so exactly
      P(W_jct > W_ict | X_i, X_j, c, t) = sigmoid(rho_c * (f_j - f_i)).
  Giving each observation a logistic error would NOT yield this pairwise model.
* rho_c = clip(exp(log_sd * Z_c), 0.2, 5.0), Z_c iid N(0,1), independent of size
  and city feature centers. log_sd = 0 is the homogeneous control; 0.9 is the
  heterogeneous scenario. Common random numbers couple both scenarios.
* psi_ct(r) = exp(mu_ct + sigma_ct*r), with random mu_ct and positive, lognormally
  distributed sigma_ct. This is the familiar lognormal quantile construction
  Q_LogNormal(mu,sigma)(Phi(r)). W itself is NOT necessarily marginally lognormal,
  because R is not standard normal. Log(W) is stored to preserve ordering without
  overflow; it is exactly the same ranking supervision as W.

ESTIMATORS (same architecture, initial weights, and minibatch stream)
------------------------------------------------------------------
shared: learn ONE positive calibration, constrained equal across all cities.
city:   learn ONE positive calibration PER CITY, shared across years.
oracle: use the true rho_c while still learning f with exactly the same NN.

A learned shared scalar is a fairer homogeneous baseline than fixing rho=1:
with a normalized score, a fixed value would impose an arbitrary noise level.
There is no temporal-stability penalty in this experiment, to isolate the effect
of calibration. Years pool independent training information, not stable twins.
This is a diagnostic experiment, not a proof for the full paper's estimator.

TRAINING AND OUTPUTS
--------------------
Default: equal-city/equal-year loss; draw groups uniformly, take all unique pairs
of distinct observations within each sampled group, average within groups, then
average groups. Small groups use all their observations, without replacement.
--weighting pairs changes group selection probability to n_ct*(n_ct-1), exposing
the consequences of pooling all pairs. No outcome-dependent pair filtering.
All calibrations are constrained to [0.1, 10] for numerical stability. Optional
shrinkage is a common penalty per city; it is NOT advertised as sample-size-
adaptive hierarchical estimation. Exact fixed-reference score normalization
removes the score/calibration scale ambiguity without a variance penalty.

Outputs: metrics.csv, city_metrics.csv, histories.csv, summary.csv,
paired_differences.csv, experiment_summary.png, per-fit diagnostic PNGs and
checkpoints, and reproducible DGP metadata. Key diagnostics:
  - test latent R2 / RMSE: recovery of f, NOT noisy realized R or dollar labels;
  - one globally affine-aligned R2 (alignment fitted on validation truths ONLY
    for evaluation; never used for training, early stopping, or probabilities);
  - within-cell observed NLL, expected excess NLL (KL to true pair probability),
    probability RMSE, latent pair accuracy and cross-city latent pair accuracy;
  - counterfactual temporal-difference RMSE, with changed numerical amenities;
  - calibration recovery and performance broken down by training city size.

Checkpoint selection uses held-out observed ranking NLL only. Look at both
probabilities and index recovery: improved calibration need not improve f much,
especially under substantial overlap and a correctly shared f. With 3 reps the
Monte Carlo SE is descriptive, not strong evidence. Training convergence and
the oracle gap should be inspected before drawing conclusions.

References for implementation / DGP:
https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html
https://eml.berkeley.edu/books/choice2nd/Ch03_p34-75.pdf

This file was authored and statically checked without running its simulation.
