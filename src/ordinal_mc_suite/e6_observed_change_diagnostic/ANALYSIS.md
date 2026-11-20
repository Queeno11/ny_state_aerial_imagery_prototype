# One paired-outcome change-loss diagnostic

**At the fixed 200-update endpoint, adding the observed-change loss did not
improve held-out observed changes, latent changes, or detection in this
replication.** D reduced fitting change loss while increasing validation change
loss and fresh stable movement. This is a bounded diagnostic, not a general
negative finding about change supervision.

## Data and exact setup

Quadratic truth, 7 mixed inputs, the established ReLU [32,16] scorer, 32 cities,
two dates, function seed 983 and replication 1 were retained. The nonlinear
measurement coefficients and independent population/noise calibration were
read from the existing E6 configuration. Effective centered-Gumbel error SD is
**0.8028980598**. The errors are independent across dates, as in E6's paired
outcome evaluation; there is no persistent tract error. The realized error
correlation is -0.0055 in fitting and 0.0444 in validation.

The original outcome labels were single-date observations, so the old baseline
could not be reused fairly. New paired labels use independent fixed seeds:
fitting **1422665402**, validation **1157853548**. Every arm receives both labeled
endpoints of the same 6,560 fitting tracts (13,120 labels) and 1,632 validation
tracts (3,264 labels). These match the earlier counts of labeled endpoints.
There are 205 fitting and 51 validation tracts per city, with about half stable
(3,296 fitting tracts) and half changing. Changing training tracts use the
established moderate step 0.4 with random direction and support clipping. No
stability assignment or true change magnitude is supplied to the learner.

The 26,240 external stability links preserve the E5 construction: two unlabeled
same-H acquisition views per labeled endpoint, at the other date. For a changing
tract, this synthetic stable view is distinct from its actual changed-date
observation. The actual changed endpoints are not incorrectly linked as stable.
There are no cross-city links or scorer complexity penalties. Scalar-score-only,
cell-specific monotone spline heads use three fitting-score quantile knots.
No additional head penalty was introduced, matching the requested displayed
objective; monotonicity is imposed by positive segment slopes.

All arms share the fixed fitting denominators:

- **Var_train(W) = 30.1244366989** across all labeled endpoints.
- **Var_train(Delta W) = 16.2014475203** across labeled tract pairs.

Both loss weights equal one. Raw-score RankNet and differentiated S/V are
unchanged. The independent E6 stable-only calibration panel and seen/unseen
test panels were reused, with no overlap with the new fitting/validation draws.
Test steps are 0.15 (small) and 0.4 (moderate). Each scorer uses its own 95th
percentile absolute normalized movement cutoff (`method='higher'`). The stable
calibration panel has 4,096 tracts; each seen-city test panel has 4,096 tracts
and each unseen-city test panel 2,048, with half assigned stable in each city.

## Training and controls

One random baseline start was refitted using validation rank stopping: it
executed 520 Adam updates and restored update 320. To bound this diagnostic,
its ceiling was 1,000 Adam updates, with no L-BFGS stage; neither cap was
extended. Frozen level-head initialization executed its 600-update ceiling and
restored update 600 using observed validation level loss. It was still improving
at that boundary. All branches start from this identical scorer/head pair.

- A: original selected scorer and frozen initialized head.
- O: 200 RankNet/stability scorer updates; 200 level-head updates on detached
  scores, with no level feedback into the scorer.
- B: A's scorer frozen; 200 head updates on levels plus observed changes.
- C: 200 joint scorer/head updates on RankNet, stability and levels.
- D: 200 joint scorer/head updates on RankNet, stability, levels and changes.

Scorer LR is 0.0001 and head LR 0.001. All branches use full batches, identical
data ordering, fresh Adam state, and record updates 0,25,50,100,200. No branch
checkpoint or hyperparameter is selected using latent or test performance.
Scorer updates total 600 across O/C/D; B/C/D each get exactly 200 head updates.
Initial baseline/head preparation is separate and fully logged.

## Compact results

The [complete compact table](compact_results.md) and its CSV contain all ten
arm-by-seen/unseen rows, with observed level/change MSE, observed within-cell
ranking accuracy, small/moderate latent RMSE and sensitivity, stable RMS, actual
FPR, cutoff, slope and correlation. W errors in that table use the moderate
test panel. Unseen-cell W prediction is unavailable zero-shot for these
cell-specific heads, so those entries are NA; latent and ranking metrics remain
reported. Slope and correlation refer to changing tracts and include an
intercept in the slope regression.

The principal fixed endpoints are:

| Arm | Seen W level MSE | Seen W change MSE | Unseen small latent RMSE | Unseen moderate latent RMSE | Unseen stable RMS | Unseen small sensitivity | Unseen moderate sensitivity |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 7.5379 | **14.7592** | **0.05357** | 0.08818 | **0.04389** | 88.00% | 99.41% |
| O | 7.5020 | 14.8058 | 0.05404 | **0.08658** | 0.04556 | **88.59%** | **99.51%** |
| B | 7.5235 | 14.8263 | **0.05357** | 0.08818 | **0.04389** | 88.00% | 99.41% |
| C | **7.4986** | 14.8148 | 0.05930 | 0.08946 | 0.05188 | 81.32% | 99.12% |
| D | 7.5237 | 14.8547 | 0.06465 | 0.09387 | 0.05861 | 71.98% | 98.24% |

A/B scorer metrics are numerically identical at every recorded checkpoint,
including slopes, correlations, thresholds and detection rates. This is
verified by exact array comparison and by equality of scorer parameters.

## Key comparisons and interpretation

**Observed changes:** D's held-out W change MSE is worse than C by 0.03990,
B by 0.02843 and O by 0.04889 on the seen moderate panel. These are small
single-sample differences, not statistically established effects. There is no
held-out gain for which changing the scorer can claim credit. B accounts for
about 57% of D's reduction in FITTING change loss, but B also fails to improve
held-out change MSE. That training fraction therefore does not demonstrate
successful out-of-sample calibration.

**Latent changes and detection:** on unseen moderate changes, D-C RMSE is
+0.00442, D-B +0.00569 and D-O +0.00729. D also raises stable RMS by +0.00674,
+0.01472 and +0.01305 respectively. On small changes, sensitivity falls by
9.34 percentage points versus C, 16.03 versus B, and 16.62 versus O. D's cutoff
is 0.11775 versus A/B's 0.09011; fresh stable FPR is 5.57% versus 5.47%, close
to the same nominal operating point. Larger nuisance movement raises the
detection threshold and limits sensitivity.

**Attenuation versus noise:** unseen moderate slopes are very similar, 0.96194
for A/B and 0.96027 for D. Correlation falls from 0.98273 to 0.98040. For small
changes, slopes are 0.95384 versus 0.95163, while correlation falls more clearly
from 0.95821 to 0.94025. These results point more toward added prediction noise
or nuisance sensitivity than a large additional attenuation of mean changes.

**Optimization/head fit/noisy labels:** D's standardized fitting change MSE
keeps falling, 0.89883 initially, 0.89000 at 100 and 0.88451 at 200. Its
validation change MSE rises from 0.86035 to 0.86438 to 0.86839. The head-only B
path exhibits the same fitting/validation divergence. Some fitting optimization
is still progressing at update 200, and the common head initialization was
still improving at its cap, so no convergence claim is made. Independent
date-specific outcome noise makes observed differences noisy. Finite head fit,
the absence of extra head regularization, sample variation and optimization
all limit interpretation; this trial does not isolate which is decisive.

## Local gradients at the shared initialization

| Scorer-gradient component | Norm |
|---|---:|
| RankNet | 0.00670 |
| Weighted temporal stability | 0.00618 |
| Standardized level loss | 0.04358 |
| Standardized change loss | 0.03604 |

Pairwise cosines: rank/stability -0.9392; rank/levels +0.2169;
rank/changes +0.0258; stability/levels -0.2013; stability/changes +0.0281;
levels/changes +0.0269. The change gradient is substantial and nearly orthogonal
to the other components at this initialization. These are local diagnostics,
not evidence of a population conflict. Gradient additivity was verified against
direct autograd.

## Reproduction and limits

From the suite directory, in the existing PyTorch environment:

```bash
python verification/e6_observed_changes.py --run e6_main_512 --config configs/e6_observed_change_one_shot.json --out e6_observed_change_diagnostic_new
```

Use a new output directory. Config/seeds, baseline and head traces, every branch
checkpoint, all metrics, D-C/D-B/D-O contrasts, and numerical checks are saved.
`trajectories.png` shows fitting/validation change losses alongside independent
latent and stable-movement trajectories. No extra seeds, weight searches,
convergence extensions or follow-up experiments were run. This null result
applies to one fixed design and budget and does not establish that observed
change supervision cannot work.
