# Fixed-budget ordinal latent diagnostic

`ordinal_mc_suite/ordinal_latent_diagnostic.py` is separate from the joint two-step pilot and its results. **The implementer did not run self-check, smoke, training, or Monte Carlo.** Use WSL Bash in the CUDA `torch_geo_env` conda environment.

## Commands from `src` in WSL Bash

```bash
conda activate torch_geo_env
python ordinal_mc_suite/ordinal_latent_diagnostic.py --plan
python ordinal_mc_suite/ordinal_latent_diagnostic.py --self-check
set -o pipefail
python ordinal_mc_suite/ordinal_latent_diagnostic.py --smoke --seeds 1 --device cuda --output ordinal_mc_suite/ordinal_latent_smoke 2>&1 | tee ordinal_latent_smoke.log
python ordinal_mc_suite/ordinal_latent_diagnostic.py --stage core --seeds 3 --device cuda --output ordinal_mc_suite/ordinal_latent_core 2>&1 | tee ordinal_latent_core.log
```

The smoke uses tiny panels and two optimizer updates per trajectory; it is an implementation check. `--plan` prints actual trajectory/checkpoint counts and tract exposure without outputs. `--self-check` makes structural and oracle assertions without training. Resume an interrupted job with the same code, options, and output directory. Use a fresh output for changed code or options.

Optional scaling stages, each in a separate directory:

```bash
python ordinal_mc_suite/ordinal_latent_diagnostic.py --plan --stage scaling --scaling-arm wide
python ordinal_mc_suite/ordinal_latent_diagnostic.py --stage scaling --scaling-arm wide --device cuda --output ordinal_mc_suite/ordinal_latent_wide 2>&1 | tee ordinal_latent_wide.log
python ordinal_mc_suite/ordinal_latent_diagnostic.py --stage scaling --scaling-arm within_city --device cuda --output ordinal_mc_suite/ordinal_latent_within_city 2>&1 | tee ordinal_latent_within_city.log
python ordinal_mc_suite/ordinal_latent_diagnostic.py --stage scaling --scaling-arm city_coverage --device cuda --output ordinal_mc_suite/ordinal_latent_city_coverage 2>&1 | tee ordinal_latent_city_coverage.log
```

Core defaults to `--methods nn basis supervised`, `--rank ranknet`, and `--links both`. Scaling defaults to `--methods nn` to avoid redundant basis and supervision runs. Optional `--rank sigmoid` or `--rank both` expands loss arms. The new core baseline is **64 fitting cities x 250 paired tracts/city**, widths 32/16, and checkpoints 400/600/1200/2400/4800 continuous Adam updates. The old pilot defaulted to 16 x 128; its results use different data. Four latent scenarios are linear/simple/rho0, quadratic/simple/rho0, quadratic/nonlinear/rho0, and quadratic/nonlinear/rho.9, with M1/M2/M3 evaluation on each.

## Design and interpretation

Each optimizer update selects 12 distinct paired tracts from each of 16 distinct fitting cities, with two endpoints per tract. **Effective tract passes** count selected tract records (including repeat selections) divided by fitting tract records: final core exposure is 4800 x 16 x 12 / (64 x 250) = 57.6 passes. These are not sequential epochs. Wide uses widths 128/64 at 64 x 250; within-city uses 64 x 1000 and coverage uses 256 x 250. The latter two multiply the default update checkpoints by four to preserve final passes with the same batch. Explicit `--checkpoints ...` overrides this. `--plan` shows the actual mapped budget.

Temporal links use all truly stable fitting tracts, as in the original pilot. Cross links default to `--link-fraction .2` of each panel's tracts, with at least one edge connecting each neighboring source city when cross links are on. `--link-limit N` instead caps both link sets and changes coverage. Every checkpoint records actual link counts and coverage. Validation links are independent of fitting links. The same links are held throughout each optimizer trajectory. Reported tract passes count sampled ranking/direct-supervision records; stability links are additionally evaluated at every update.

All source panels share fitting-city contexts drawn independently of fitting city count: validation, source test, and source threshold use independent **held-out tracts in the first fitting source cities**, preserving those city contexts across scaling arms. Target cities and contexts are independent. Source evaluation panels and target evaluation panels remain fixed across scaling arms. Fitting draws are deterministic but **not nested** across data sizes because the pilot generator seeds/draws whole panels. Data, network initialization, basis initialization, minibatches, links, and evaluation draws use distinct reproducible streams. `--initialization-seed N` varies only NN/basis initialization while keeping `--seed` and `--seeds 1` fixed; use distinct output directories to compare initial paths on identical generated data and batches. The default three replicates vary both data and initialization and alone cannot isolate initialization effects.

The ordinal network sees seven observed mixed features. The privileged basis arm sees true H with the correct linear or full quadratic basis and learns coefficients under the *same* ordinal plus stability objective, without f labels or true coefficients. Its stable-H temporal numerator is zero by construction. Direct-f supervision receives fitting f labels and the same seven-feature network, and evaluates **raw** f predictions without affine fitting. All networks in the same latent scenario share initialization. Every true cell map is strictly increasing, giving identical within-cell W ordering across M1/M2/M3. Each ordinal scorer trains once on M1 outcome ranks per scenario/rank/link/seed, and evaluates all three maps. The self-check verifies this ranking equality.

At each checkpoint ordinal/basis scores get one **source-fitting-only** positive affine calibration against true f, reused on source/target panels and all maps. A nonpositive fitting correlation receives the flagged positive-SD fallback, with no oracle sign reversal. Predicted conditional mu applies the **true** city/date outcome map to aligned f and integrates Gaussian noise for M3 with 32-point Gauss-Hermite quadrature. This oracle map is a latent recovery diagnostic, not a deployable outcome head. Four MSEs cover f, conditional mu, delta f, and delta mu for all/stable/small/large strata. Change bias/slope, stable movement RMS, and detection rate use independent stable-only threshold panels. The self-check asserts all four errors are zero when oracle input is true f.

`core_objectives.csv` (or `scaling_objectives.csv`) holds pair loss, separate unweighted stability numerators, within-cell dispersion, weighted penalties, and training/validation objectives on a **fixed sampled monitoring set of up to 12 tracts per city**, with all selected links. Rows show sampled and full tract counts. These are descriptive sampled objectives, **not exhaustive full-panel objectives**. Validation and test truth never select checkpoints. The training pair objective groups comparisons within city/date to avoid a global all-pairs matrix.

Outputs also include stage metric CSV, per-scenario/map four-metric plots, change-strata plots, sampled objective plots, generated `REPORT.md`, and atomic model/optimizer/RNG checkpoints per trajectory. The manifest fingerprints this driver and imported pilot code, rejects inconsistent resumes, and requires all checkpoint outputs before marking a trajectory complete. An interrupted checkpoint regenerates missing diagnostics from its saved state. Only load trusted local `.pt` checkpoints. Plots average seeds within clearly labeled scenario/map; the CSV preserves paired seed results.

Fixed-budget trajectories do not establish a local optimum. Objective plots compare progress within a method; supervised MSE, RankNet, and bounded sigmoid losses have different scales and must not be ranked by their plotted objective levels. This suite does not fit likelihood benchmarks or outcome heads; the earlier pilot's likelihood jobs used different data. A same-panel likelihood gap would require a separately fitted reference. There is no oracle early stopping, automatic budget extension, or hyperparameter search.
