# E6 interim analysis — 512 labeled observations per city

The read-only SQLite snapshot contains 62/120 completed jobs: 17 replications
at 32 cities and 14 at 64 cities, for each structural case. The comparisons below
use the balanced, prespecified replication IDs 1–14 across all four city/case
combinations (56 jobs). All 62 completed jobs' metrics are also exported. No
configuration, source, checkpoint or live-run result was modified. The live
experiment can continue uninterrupted. Snapshot provenance is in
`snapshot_info.json`; `snapshot.sqlite` preserves the exact data analyzed.

This is a provisional analysis. Intervals are pointwise paired 95% t intervals
across 14 replications, without multiplicity or sequential-monitoring adjustment.
They describe this snapshot and should not be used as a stopping rule. The
remaining replications are needed for the planned final analysis.

## 1. The ordinal baseline recovers magnitudes reasonably well with more data

For quadratic truth/ReLU scoring, unseen-city changes at nominal step 0.4:

| Fitting cities | Changing-tract RMSE | Sign accuracy | Stable RMS | Cutoff | Sensitivity | Fresh stable FPR |
|---|---:|---:|---:|---:|---:|---:|
| 32 | 0.0904 | 99.91% | 0.0466 | 0.0964 | 99.07% | 5.18% |
| 64 | 0.0745 | 99.98% | 0.0334 | 0.0698 | 99.61% | 5.02% |

At 64 cities, direction-specific true mean changes of -0.450/+0.465 become
-0.440/+0.447. At the larger step 0.8, true means -0.826/+0.873 become
-0.811/+0.835. Average attenuation remains modest; individual errors persist.
The linear implementation check is stronger still: corresponding change RMSE
is 0.0053 at 32 cities and 0.0034 at 64 cities.

Small changes remain harder, but the earlier severe detection failure is much
reduced. For quadratic truth at step 0.15, baseline sensitivity is 86.0% at
32 cities and 96.2% at 64, with sign accuracy 99.45% and 99.84%. These results
support revising the broad interpretation that flexible ordinal scoring cannot
recover magnitudes: it can recover average magnitudes well in this DGP with
sufficient data, although it remains less precise than privileged ML.

## 2. Joint level supervision has not improved latent magnitudes

At 64 cities, quadratic truth, unseen cities, and step 0.4, the mean C-B latent
RMSE differences are:

| Measurement truth | Fitted head | C-B RMSE difference [95% interval] |
|---|---|---:|
| Affine | Affine | +0.00105 [-0.00046, +0.00256] |
| Affine | Spline | +0.00127 [-0.00070, +0.00324] |
| Nonlinear | Affine | +0.00232 [+0.00110, +0.00353] |
| Nonlinear | Spline | +0.00059 [-0.00122, +0.00240] |

Positive differences worsen recovery. None of these conditions favors joint
training on average; the nonlinear-truth/affine-head condition has an interval
entirely above zero. C-D comparisons tell the same substantive story. In that
condition C-D is +0.00261 [+0.00143, +0.00380]. Differences are numerically small
at moderate sizes, but do not support the intended latent-recovery hypothesis.

This cannot be explained solely by restoring the initial checkpoint. Across all
224 selected C fits, 119 retain step zero and 105 retain later checkpoints.
In the 64-city quadratic case, 38/56 selected fits retain positive update counts.
The reported metrics evaluate the validation-selected checkpoints, not the
unrestored ends of training.

## 3. Joint training often raises nuisance movement and harms small-change detection

For quadratic truth at 64 cities, C's stable RMS rises from the baseline 0.0334
to 0.0371–0.0393 across the four measurement/head combinations. Cutoffs rise
from 0.0698 to 0.0775–0.0827. Fresh stable FPR remains approximately 5%.

Moderate-change sensitivity therefore has limited room to improve and generally
falls slightly. The effect is clearer for small changes. At step 0.15, C-B
sensitivity differences are -3.55, -3.25, -3.98 and -2.27 percentage points for
affine/affine, affine/spline, nonlinear/affine and nonlinear/spline respectively.
All corresponding mean latent RMSE differences are positive as well. For
nonlinear truth with an affine head, C-B sensitivity is -3.98 percentage points
[-5.38, -2.59]; C-D is -3.80 [-5.19, -2.41]. Thus less precise change predictions
and larger nuisance cutoffs both contribute, rather than a breakdown of the
direction signal alone. `paired_effects.png` displays the paired comparisons.

## 4. Frozen outcome calibration is useful; joint training adds no clear benefit

At 64 cities and step 0.4, an affine head fitted to the frozen quadratic scorer
has conditional-mean-change RMSE 0.311 under affine truth, versus 1.325 for
predicting zero change. Under nonlinear truth, the corresponding errors are
0.397 versus 1.528. These are useful calibration results and do not require
moving the latent scorer.

Across the four head/measurement combinations, all C-B and C-D intervals for
observed-change RMSE and conditional-mean-change RMSE include zero at this
city count and step size. The nonlinear-truth/spline-head case has the most
encouraging mean conditional-mean-change difference, C-B=-0.0132, but its interval
[-0.0394, +0.0129] remains inconclusive. This is not evidence of improved latent
cardinal recovery; even its latent RMSE difference is positive on average.

Observed-change errors are heavily influenced by irreducible outcome noise.
Under affine truth at 64 cities, B's observed-change RMSE is 2.901 versus 2.884
for privileged ML with the known conditional-mean measurement map. Nevertheless,
B's conditional-mean-change RMSE is 0.311 versus 0.00364 for that reference.
The observed-outcome comparison alone would conceal this meaningful prediction
gap. The flexible spline head does not consistently beat the affine head, even
under nonlinear truth, so greater calibration flexibility alone is not a cure.

## 5. Benchmarks and interpretation

At 64 cities, privileged ML's quadratic latent-change RMSE is about 0.0019,
compared with 0.0745 for A. ML knows the structural basis and nuisance mechanisms;
it is not an attainable empirical benchmark. The same-architecture noisy-R
oracle has RMSE 0.0906 and does not outperform A. This cautions against assuming
that noisy level labels necessarily improve this scorer. It does not prove a
capacity bound: architecture approximation, objective choice, and optimization
can all matter for that diagnostic.

All recorded implementation checks pass in the balanced snapshot; all privileged
ML fits report convergence. A and B share latent metrics exactly. The empirical
pattern is therefore: sufficient data makes the ordinal/stability scorer much
more effective, frozen calibration helps express predictions on the observed
scale, and joint level training has not yet provided an incremental advantage
for latent magnitudes, detection, or reported outcome changes. A plausible
explanation is that additional noisy level fitting perturbs an already effective
ordinal scorer while introducing calibration estimation error, but this is an
interpretation, not a mechanism established by the experiment.

No training settings were changed in response to this interim analysis. Complete
the planned run before drawing final conclusions. Full balanced tables, paired
contrasts including D_common, and the true-versus-predicted change figure are in
`balanced_report/`; all strength/seen/unseen conditions remain available there.
