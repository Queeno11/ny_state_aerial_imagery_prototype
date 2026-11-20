# Joint versus two-step ordinal pilot

Entry point: `joint_two_step_pilot.py`. Fixed weights and fixed training endpoints; no hyperparameter search. **No smoke or training was executed here.** Python needs NumPy, SciPy, Matplotlib and PyTorch (`requirements.txt`). Local Python 3.13 has the first three but no PyTorch; the repository `.venv` lacks NumPy. CUDA on the intended RTX 3070 is unverified.

## PowerShell commands from `src`

```powershell
python .\ordinal_mc_suite\joint_two_step_pilot.py --plan --preset full144
python .\ordinal_mc_suite\joint_two_step_pilot.py --self-check
New-Item -ItemType Directory -Force .\ordinal_mc_suite\joint_two_step_smoke | Out-Null
python .\ordinal_mc_suite\joint_two_step_pilot.py --smoke --preset full144 --seeds 1 --device cpu --output .\ordinal_mc_suite\joint_two_step_smoke 2>&1 | Tee-Object .\ordinal_mc_suite\joint_two_step_smoke\console.log
$LASTEXITCODE
```

The smoke covers all eight settings, both ranking losses, both link regimes, A/B/C/D/O and representative scratch arms, with four source cities, two target cities, 16 paired fitting tracts/city, five ordinal and three joint updates, and 12 likelihood iterations. Plots are labeled **IMPLEMENTATION CHECK / NOT SCIENTIFIC EVIDENCE**. Short likelihood optimization is uncertified; inspect benchmark and configuration JSON. There is no automatic retry.

```powershell
python .\ordinal_mc_suite\joint_two_step_pilot.py --plan --preset full144 --seeds 1
python .\ordinal_mc_suite\joint_two_step_pilot.py --preset full144 --seeds 1 --device cuda --output .\ordinal_mc_suite\joint_two_step_one_seed 2>&1 | Tee-Object .\ordinal_mc_suite\joint_two_step_one_seed.log
$LASTEXITCODE
python .\ordinal_mc_suite\joint_two_step_pilot.py --plan --preset full144 --seeds 3
python .\ordinal_mc_suite\joint_two_step_pilot.py --preset full144 --seeds 3 --device cuda --output .\ordinal_mc_suite\joint_two_step_three_seeds 2>&1 | Tee-Object .\ordinal_mc_suite\joint_two_step_three_seeds.log
$LASTEXITCODE
```

These run sequentially. Defaults: 16 source cities, 128 paired fitting tracts/city, four target cities, 64 tracts/city in each independent validation/calibration/final panel, 400 ordinal and 200 additional updates. `core96`, `controlled128`, `full144` have respectively 96/128/144 scorer stages per seed over 32 rank/link configurations. The last includes scratch C/D in settings 1 and 8. There are eight source plus eight target known-noise likelihood jobs/seed, separately counted, 64 source frozen-head QPs and additional target refits. A completed configuration resumes only under an identical manifest and source SHA-256; use a fresh output for changed code or inputs. Outputs include metrics, paired contrasts, report, per-job diagnostics and three PNGs.

## Design

### Resuming after a frozen-head iteration-limit error

The original SLSQP calibration solver remains the first attempt. If it fails,
calibration now solves the same convex least-squares objective in segment-slope
coordinates with SciPy BVLS. Intercepts remain unrestricted; every segment slope
retains its original lower bound of 1e-5. Column scaling is numerical only. The
fallback checks feasibility and the projected-gradient optimality condition and
records solver diagnostics; it does not drop an arm or change loss weights.

Rerun the same command and output directory. The patch recognizes the exact
pre-fallback source hash with otherwise identical settings, preserves completed
configurations, and restarts only incomplete configurations. It saves the original
manifest as `manifest_before_solver_fallback.json` and records reused jobs and
both source hashes in `solver_patch_resume.json`. Other code/configuration changes
still require a fresh output directory. In WSL/Bash:

```bash
set -o pipefail
python -u ./ordinal_mc_suite/joint_two_step_pilot.py \
  --preset full144 --seeds 1 --device cuda \
  --output ./ordinal_mc_suite/joint_two_step_one_seed \
  2>&1 | tee -a ./ordinal_mc_suite/joint_two_step_one_seed.log
echo "Exit code: $?"
```

The patch passed focused NumPy/SciPy checks against the original constrained
objective for M1/M2/M3, both cardinal weights, repeated spline knots, slope
feasibility, optimality, and resume guards. No Monte Carlo or NN training was
run while applying the patch; the failed setting must still be retried by the user.

The fixed seven-input ReLU scorer is 7 to 32 to 16 to 1. Structural H has direct coordinates; a fixed orthogonal mix hides H with acquisition nuisance and city cues in X. The known-family benchmark has access to the structural basis obtained by the specified inverse mix, an information advantage. Linear truth is `H0+0.6*H1-0.4*H2`; quadratic truth is `H0+0.7*H1*H2+0.4*(H2^2-1)`. Both are centered and scaled against an independent 100,000-draw population basis reference.

Stable (50%) pairs retain identical H but independently vary acquisition nuisance. Their **observed endpoints** supply temporal links; there are no extra temporal views. Small and large groups (25% each) use `abs(delta f)` bins `[.12,.38)` and `[.48,1.10)`. Simple changes move H0. Nonlinear changes fix H0 while moving H1/H2. Both baseline and candidate are redrawn on rejection because bounded support can make a fixed baseline's requested signed bin impossible. Changed-group support is intentionally conditioned on sign and magnitude; JSON records attempts, rejection rates and endpoint ranges. Signs alternate where feasible.

Residuals are `eta_t=sigma*(sqrt(rho)*u+sqrt(1-rho)*eps_t)` with marginal sigma=.8, rho=0/.9, independent of full H history. Shared streams match marginal draws where feasible. M1 is `Psi=a+r`; M2 is `a+b*r`, positive b; M3 is `Psi=a+b*(r+k*tanh(r))`, positive b,k. Persistence does not generally cancel in delta W under changing maps.

Each update samples up to 12 tracts/source city and uses **all eligible ordered within-cell pairs in the minibatch**, weighted `1/(number_of_cells*m*(m-1))`; it does not enumerate full-fitting-data pairs. `V=0.5*sum(weight*gap^2)` normalizes temporal and cross stability, differentiated through V. RankNet is `softplus(-signed_gap)`; the bounded loss is `sigmoid(-signed_gap/tau)` with tau=.3. ON adds exact cross-city same-H auxiliary links for 20% of fitting tracts. A reserved city-chain spanning tree guarantees connected support; additional links use distinct fitting origins. A custom sample too small to cover `cities-1` links fails clearly. Exact coverage, city counts and graph components are logged. Default lambda_T=lambda_C=.3.

The cardinal objective is `kappa/vstar*((1-q)*mean(a^2)+q*mean(d^2)/4)`, `a=(e0+e1)/2`, `d=e1-e0`, q=.5 for levels and .8 for levels+changes, kappa=1. q=.5 equals endpoint MSE times `kappa/(2*vstar)`. A deterministic rho-zero fitting marginal reference fixes within-cell vstar across persistence variants. Both effective coefficient sets are logged. CLI lambda/q overrides are deliberate later experiments; no adaptive weighting or tuning occurs.

A freezes an ordinal scorer and solves a constrained level-head QP. B solves a level/change QP on that identical scorer. C/D warm start joint scorer/head updates from A/B. O continues ordinal training then refits a head. Scratch C/D use a common original scorer/head initialization in settings 1 and 8. M1 uses one positive global score conversion and cell intercepts; **target M1 keeps the source conversion** and fits only target intercepts. M2 uses positive cell slopes; M3 uses fixed-quantile monotone spline heads. Failed QPs raise. All paths retain prespecified **final** training checkpoints; validation is diagnostic only. Gradient diagnostics include weighted stability/cardinal norms and conflict cosines.

The estimated parametric benchmark knows the correct basis, map family and population centering/scale, but estimates coefficients and cell maps from fitting W by paired Gaussian likelihood with **known design sigma/rho**. M2 includes slope-dependent paired covariance. M3 has safeguarded inverse roots, implicit first derivatives and log Jacobian. Estimated M3 k is bounded in (0,2); target likelihood bounds intercepts to [-10,10], log slopes to [-3,3] and raw k logits to [-8,8]. Independent target calibration fits target map parameters with estimated f frozen. Final gradient norms and solver messages are exposed; finite predictions do not certify convergence. The separate true-parameter oracle predicts f and Gaussian-integrated conditional mean mu, never realized R/W.

NN raw scores receive one positive fitting-only affine alignment to true f **for evaluation only**, reused across cities/dates. Negative fitting correlation is flagged and uses a positive score-SD fallback without oracle sign reversal. Source fitting/final and target calibration/final metrics cover f, R, mu, W and their changes, bias, slope, sign, stable movement and detection. Separate stable calibration panels set 5% false-positive thresholds; independent tests measure actual rates. Plots facet ranking loss and cross-city regime. `paired_contrasts.csv` gives C-A, D-B, D-C within seed/setting/rank/link. One seed has no confidence interval. Scratch covers only two settings, so its overall average is not directly comparable to eight-setting arms.
