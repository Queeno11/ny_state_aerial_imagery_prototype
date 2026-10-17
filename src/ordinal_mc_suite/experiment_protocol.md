# Exact protocol for the focused E1, E3 and E4 Monte Carlo battery

This document describes the **overnight configurations delivered in this conversation**, including multicore execution and the zero-dispersion recovery repair. It distinguishes those configurations from the much broader experiments supported by the package.

The protocol was checked against `ordinal_mc/core.py`, `design.py`, `experiments.py`, `runner.py`, `report.py`, and the three `configs/overnight_*.json` files. The uploaded E1 and partial E3 CSVs agree with the relevant configuration fields.

For an exact historical audit of a run on your machine, its output directory's `resolved_config.json` and `run_metadata.json` are authoritative. CLI arguments override the input JSON. In particular, the number of CPU workers can differ from a JSON default.

## 1. What was actually requested by the overnight launcher?

| Setting | E1: measurement | E3: temporal information | E4: imperfect links |
|---|---|---|---|
| Input file | `configs/overnight_measurement.json` | `configs/overnight_nn.json` | `configs/overnight_links.json` |
| Output directory | `overnight_measurement` | `overnight_nn` | `overnight_links` |
| True latent function | Linear | Linear and quadratic | Linear |
| Fitted scorer | Linear | ReLU NN, hidden widths `[8,4]` | Linear |
| Cities | 8, 16, 32, 64 | 8, 16, 32, 64 | 32 fixed |
| Labeled observations per city | 32 total over two dates | Same | Same |
| Replications per setting | 30 | 30 | 30 |
| Label measurement map | Two positive monotone regimes | Affine | Affine |
| Composition settings | 0 and 0.7 | 0 | 0 |
| Methods | Raw levels, log + cell FE, common logit | Common logit without/with temporal links | No links, fixed temporal weight, tuned temporal weight |
| Additional complexity penalty | None | Validation-selected squared path norm | None |
| Temporal weight | 0 | 0 or 0.3 fixed | 0, 0.3 fixed, or selected from `[0,.03,.3,1]` |
| Planned selected fits | 1,440 | 480 | 360 |
| Planned optimization candidates | 4,320 | 11,520 | 2,160 |

The focus on linear truth/linear estimation in E1 and E4 was a deliberate configuration choice for controlled benchmarks. E3 introduces a flexible estimator and a nonlinear truth. It is **not** a software restriction.

The package supports fitted models `linear`, `basis`, and `nn`, and true functions `linear`, `quadratic`, `smooth`, and `regime`. Broadening the model/truth lists creates a different, larger run. A JSON also contains sections for experiments that are not selected; for example, the large `experiment1` section in `overnight_nn.json` is inactive when the command selects only experiment 3.

## 2. What do the E1 figure names and `linear / linear` mean?

The first word is the **true latent function**, not the monetary observation map. The second is the **fitted scorer class**.

Thus `linear / linear` means:

- The latent signal is linear in three signal coordinates.
- The fitted score is linear in the seven observed, mixed covariates.
- The relationship between observed monetary labels and that score can still be nonlinear and city/date dependent.

In the supplied eight E1 images:

| Image | Measurement regime | Composition | Vertical-axis metric | Plotted methods |
|---|---|---:|---|---|
| `exp1_0000.png` | Multiplier | 0 | Latent within-cell misranking | All three |
| `exp1_0001.png` | Multiplier | 0 | Normalized latent-difference MSE | `rank_common` only |
| `exp1_0002.png` | Multiplier | 0.7 | Latent within-cell misranking | All three |
| `exp1_0003.png` | Multiplier | 0.7 | Normalized latent-difference MSE | `rank_common` only |
| `exp1_0004.png` | Shape | 0 | Latent within-cell misranking | All three |
| `exp1_0005.png` | Shape | 0 | Normalized latent-difference MSE | `rank_common` only |
| `exp1_0006.png` | Shape | 0.7 | Latent within-cell misranking | All three |
| `exp1_0007.png` | Shape | 0.7 | Normalized latent-difference MSE | `rank_common` only |

This numbering describes the uploaded figures. In general, panel numbering follows the order in which groups are encountered; parallel job completion can affect that order. Use the title and y-axis rather than treating a numeric filename as a permanent semantic identifier.

### Why only `rank_common` on the odd-numbered charts?

`core.test_metrics()` calculates the normalized difference-MSE metric only when the fit's loss is `common_logit` or `heterogeneous_logit`. In this E1 configuration:

- `rank_common` has loss `common_logit`.
- `raw_levels` and `log_cell_fe` have loss `squared`.

Therefore only the ranking method has a value for that MSE column. `report.export()` does not summarize a metric absent from a method's records, and `report.make_plots()` plots only the summarized series available for that metric.

The L2 fits are **not missing**: they are present in the misranking figures and CSVs. This is a reporting-scope choice in the implementation, not a mathematical impossibility of defining a normalized score-difference diagnostic for an L2 scorer. The current difference-MSE plots cannot establish comparative spacing performance against those L2 benchmarks.

The default Matplotlib color cycle restarts in each figure. Consequently `rank_common` is green when it is the third series in the misranking plot, but blue when it is the only series in the MSE plot. The label identifies the method.

The stock plot footer says that smoke runs only check implementation. That generic sentence does **not** identify this run as a smoke test. E1 has 30 replications per setting. Original suite plots show **plus/minus one Monte Carlo SE**, not 95% intervals. The separately reconstructed analysis figures use explicitly labeled 95% t intervals instead.

## 3. Shared data-generating process

### 3.1 Units, cities, dates and sample size

A city is indexed by `c`, a date by `t` in `{0,1}`, and a city-date pair is a cell. The label budget is `n=32` observations per city **across both dates**, giving 16 in each city-date cell. These are synthetic units; there are no actual satellite pixels, images or pretrained encoders in this suite.

The labeled observations at the two dates are separate sampled units. Temporal links are additional auxiliary views of fitting units, described below; they are not a second observed monetary label for each source.

For each cell, a random permutation selects 25% of labels for validation. Therefore:

- Each cell has 12 fitting and 4 validation observations.
- Each city has 24 fitting and 8 validation observations.
- At C cities, `n_fit=24C` and `n_validation=8C`.

The final selected model is **not refitted on all 32C labels**. The validation labels remain held out. The current battery does **not** use five-fold cross-validation: that upgrade was discussed separately and was not included in these completed runs.

### 3.2 City contexts and signal coordinates

Each city has a two-dimensional context `Z_c`, drawn independently from a standard normal distribution. For each observation draw a three-dimensional vector U with independent uniform(-1,1) entries. Let d denote the composition setting. Before population standardization,

\[
H^{raw}=\sqrt{3}\tanh\{\operatorname{arctanh}(U)
 +d Z_{c1}(1,0.5,-0.5)\}.
\]

All operations inside this expression are coordinatewise. With d=0, the signal distribution is the same across cities. With d=0.7, city context shifts the distribution of signal characteristics. It does not create a different structural function in every city.

H is obtained by centering and scaling these coordinates using an independent population calibration, not the realized training data.

### 3.3 Structural function and normalization

Let u=H times a fixed 3-by-3 orthogonal matrix. The active raw structural functions are:

\[
g_{linear}(H)=u_1+0.6u_2-0.4u_3,
\]

\[
g_{quadratic}(H)=u_1+0.7u_2u_3+0.4(u_3^2-1).
\]

Each function is centered and scaled as

\[
f(H)=\frac{g(H)-m_g}{D_g},\qquad
D_g^2=\tfrac12 E[(g(H_j)-g(H_i))^2],
\]

where the pair is drawn independently within the same population city context. Moments are approximated using 100,000 independent population pairs. The calibration is fixed for each truth/composition combination and does not change with the number of training cities.

### 3.4 Seven observed covariates

The estimator receives an orthogonal mixture of seven coordinates:

1. Three standardized signal coordinates H.
2. Two acquisition coordinates, `uniform(-sqrt(3),sqrt(3)) + 0.6*t`.
3. Two city-cue coordinates, `sqrt(3)*tanh(Z_c)`.

The acquisition and city-cue strengths are both 1 in the focused battery. A fixed 7-by-7 orthogonal matrix mixes these coordinates into X. The estimator receives X, not the separate signal/nuisance blocks. City cues are constant within a city and can encode its identity/context.

Before fitting, X is standardized using only the labeled fitting observations. Validation, test observations and auxiliary views do not determine that feature preprocessing. For linear/NN models, the additional feature-standardization pass in `Features` is approximately an identity adjustment after the initial standardization.

### 3.5 Noise and difficulty calibration

One error is drawn for each labeled unit, rather than drawing independent noise for every comparison pair:

\[
R=f(H)+\sigma E,\qquad
E=(G-\gamma)\sqrt6/\pi,\quad G\sim\text{standard Gumbel}.
\]

E has mean zero and variance one. The difference of two independent unit errors is logistic. The code calibrates sigma independently for each truth/composition combination so that

\[
E\left[\Lambda\left(-\frac{|\Delta f|}{\sigma\sqrt6/\pi}\right)\right]=0.2
\]

under the independent population calibration. Thus the default target is approximately 20% observed-label ordering error even for an oracle using true f. Although the JSON also contains `noise_sd=0.7`, that value is overridden by this calibration when `bayes_target=0.2`; `noise_sd_effective` records the value actually used.

This 20% oracle error concerns predicting noisy observed comparisons. It is **not** a 20% lower bound on the latent-f misranking plotted in the main figures.

## 4. Shared scorer, ranking loss, stability term and optimization

### 4.1 Scorers

The linear scorer is a linear function of preprocessed X. Ranking fits have no output intercept. L2 benchmarks additionally fit a global intercept, and the log/FE benchmark also fits unrestricted training-cell intercepts.

E3's NN has architecture `7 -> 8 -> 4 -> 1`, ReLU activation after each hidden layer, hidden-layer biases, and no scorer output bias. It has 104 trainable scorer parameters. Each effective weight/bias is parameterized as `10*tanh(raw_parameter/10)`, imposing a coefficient bound of 10.

### 4.2 Fitting comparisons and weights

All ordered distinct fitting pairs within each city-date cell are used. With 12 fitting observations in every cell there are 12*11=132 ordered pairs per cell, or **264C fitting pairs** across 2C cells. The reversed direction of each pair is included; these are not independent observations.

If N=24C and a cell has m=12 fitting observations, each ordered pair receives weight

\[
w_{ij}=\frac1{N(m-1)}.
\]

The weights sum to one, and each source observation receives total outgoing weight 1/N. Define the observed pair label, margin and score dispersion as

\[
Y_{ij}=\operatorname{sign}(W_j-W_i),\quad
\Delta s_{ij}=s(X_j)-s(X_i),\quad
\widehat V(s)=\tfrac12\sum w_{ij}(\Delta s_{ij})^2.
\]

The common-logit objective is

\[
\widehat Q_{cross}(s)=\sum w_{ij}\log\{1+\exp(-Y_{ij}\Delta s_{ij})\}.
\]

Its margins are **not normalized by V**. Its scale absorbs the common logistic slope.

For offered temporal links S, the added term is

\[
\lambda_t\frac{|S|^{-1}\sum_{(a,b)\in S}[s(X_a)-s(X_b)]^2}
 {\widehat V(s)}.
\]

The denominator uses labeled fitting comparisons, and gradients flow through it. A cross-city link term exists in the suite, but none of the three focused configurations activates it.

### 4.3 Additional regularization

E3 adds alpha times the squared path norm of the NN's effective parameters. This is not ordinary weight decay or the sum of squared NN parameters. It sums products of squared weights along input/bias-to-output paths. It is a separate regularized sensitivity estimator relative to the paper's displayed two-part criterion.

For linear scorers, the same helper reduces to a squared coefficient norm. E1 and E4 set alpha=0, so no additional complexity penalty is used there.

### 4.4 Restarts, training and validation selection

Every candidate uses:

- 1,000 full-batch Adam epochs, learning rate 0.01.
- Gradient clipping at norm 20 during Adam.
- Up to 300 L-BFGS iterations with strong-Wolfe line search.
- An additional audit refinement of up to 300 L-BFGS iterations.
- Three deterministic initialization restarts per hyperparameter setting.
- CPU float64 in the provided run configurations.

Adam's best checkpoint is retained according to its training objective. For each hyperparameter setting, **the restart with lowest final training objective is chosen**. Its observed validation misranking is then used to choose hyperparameters. Restarts are not chosen using test f or validation accuracy.

Validation samples 3,000 within-cell comparisons with replacement from the held-out labeled units. At eight cities this uses 64 held-out observations, with only 96 distinct unordered within-cell pairs available. Repeating comparisons does not create additional independent labels. The validation set is drawn from training cities; it is not a whole-city holdout.

Validation ties favor the larger alpha, then the smaller sum of link weights, then the weight tuple. The exact key in `runner.fit_job()` is `(validation_misranking, -alpha, sum(weights), weights)`.

True f, true link gaps and test panels do not enter this selection rule.

The objective-drop audit tests continued optimization on the fitting objective; it is not validation or a global-optimality certificate. The raw gradient infinity-norm threshold is 1e-5. Audit objective-drop tolerance is 1e-5. The ReLU objective is nonsmooth, so both diagnostics must be interpreted with that limitation.

## 5. E1: measurement transformations

### 5.1 Question and exact regimes

E1 asks how well three estimators recover the latent within-cell ordering when monetary labels are transformed differently by city/date. Both latent truth and fitted scorer are linear in this focused run.

For the multiplier regime:

\[
a_{ct}=0,\quad
b_{ct}=\exp\{0.5\tanh(Z_{c2})+0.1t\},\quad
W=b_{ct}\exp(0.6R).
\]

Therefore `log W = log b_ct + 0.6 f(H) + 0.6 sigma E`: a linear global score plus cell intercepts is correctly specified for the conditional mean of the logged label.

For the shape regime:

\[
a_{ct}=0.6+0.3\tanh(Z_{c1})+0.1t,\quad
b_{ct}=\exp\{0.5\tanh(Z_{c2})+0.1t\},
\]

\[
\kappa_{ct}=0.6+0.2\tanh(Z_{c1}),\qquad
W=a_{ct}+b_{ct}\exp\{0.6[R+\kappa_{ct}\tanh(R)]\}.
\]

Both maps are strictly increasing. **The contrast between the two selected regimes adds both an additive level component and a nonlinear shape component.** It does not isolate the effect of the nonlinear term alone. The broader suite has a separate location/scale stage, but that stage is not in this focused battery.

Both regimes are crossed with composition d in `{0,0.7}` and C in `{8,16,32,64}`. The same latent observations/noise and pair labels are shared across transformations at matched replication/composition/C. Their observed values change, while their within-cell ordering does not.

### 5.2 Three estimators

| Method | Training target/objective | Cell intercepts? | Stability links? |
|---|---|---|---|
| `raw_levels` | Squared error on observed W | No | No |
| `log_cell_fe` | Squared error on log W | Yes, one per city-date | No |
| `rank_common` | Common-logit loss on within-cell observed orderings | No explicit cell effects | No |

For both squared-loss benchmarks, the target is globally centered/scaled using fitting observations only before optimization. Their global intercepts are trained. The log benchmark's cell intercepts are also trained jointly. At test time, ranking uses only the global scorer: a cell constant is irrelevant to a within-cell ordering, and no intercept is estimated from unseen-city labels.

Alpha=0 and all link weights are zero. The three restarts improve numerical reliability; they do not provide additional observations.

### 5.3 What this design does and does not isolate

The linear latent truth is within the fitted linear score class. That does not make the raw-level conditional mean linear: even the multiplier map contains an exponential. Thus a raw linear-L2 disadvantage can reflect both a misspecified monetary conditional mean and non-comparable measurement. A common-map control or a flexible-level benchmark would be needed to separate those mechanisms more sharply.

A misspecified log/FE conditional mean can nevertheless preserve ordering well. Low ranking risk under the shape regime does not prove that it correctly predicts monetary levels.

The count is `30 reps * 4 C * 2 transformations * 2 compositions * 3 methods = 1,440` selected fits. With three restarts and no hyperparameter grid beyond alpha=0, there are 4,320 candidate fits.

## 6. E3: exact temporal links with flexible estimation

### 6.1 Labels and cases

The true functions are linear and quadratic. Both are estimated with the same `[8,4]` NN. Composition d=0, with acquisition and city-cue strengths both 1. Unlike E1, observed labels use the affine map

\[
W_{ict}=a_{ct}+b_{ct}R_{ict},\qquad
a_{ct}=3Z_{c1}+t,\quad b_{ct}=1.5+\exp(0.2Z_{c2})>0.
\]

The `psi_stage='shape'` string in E3 rows is an inherited scenario label; `positive_labels=False` and `psi_delta=0` mean the actual map is affine. It is not E1's exponential shape map.

### 6.2 Exact temporal auxiliary views

Each of the 24C fitting sources gets two unlabeled temporal views, creating **48C temporal links**. The auxiliary view:

- Keeps the source's H and true f exactly fixed.
- Keeps the city fixed.
- Switches t to `1-t`.
- Redraws acquisition nuisance coordinates at that date.
- Preserves the same city-cue coordinates.

The learner receives the auxiliary X and the fact that a link was offered. It receives no auxiliary monetary labels or latent scores. Exact stability means equality of the systematic signal f; it does not mean equality of independent noisy realized R values.

The implementation also constructs a cross-city auxiliary pool, but no cross-city penalty is activated in this focused E3. The `auxiliary_pool_size` column can include constructed but unused views. The active `link_count` is the clearer measure of the information actually used.

### 6.3 Two arms and regularization selection

| Arm | Temporal coefficient | Additional penalty |
|---|---:|---|
| `none` | 0 | Alpha times squared path norm |
| `temporal` | 0.3 fixed | Same kind of penalty, independently selected alpha |

Each arm selects alpha from `[0,.001,.01,.1,.3,1,3,10]` using the validation procedure above. Lambda is **not tuned in E3**. A `lambda_grid` field in its JSON does not activate tuning: `experiments.specifications()` supplies fixed weights for E3.

The no-link arm excludes unused auxiliary X from its scorer computations. The paired arms share source data, preprocessing inputs, validation/test units, comparison pairs and initialization seeds. Their selected regularization can differ.

The count is `30 reps * 4 C * 2 truths * 2 arms = 480` selected fits. Each searches `8 alpha values * 3 restarts = 24` candidates, giving 11,520 planned candidate fits.

This experiment compares prediction risk under flexible estimation. It does not by itself identify a variance reduction or a different convergence rate.

## 7. E4: approximate and incorrect temporal links

E4 fixes C=32 and uses linear truth with a linear scorer. The labeled DGP is the same affine/common-Gumbel setup as E3. There are 768 fitting labels and 1,536 offered temporal links in each replication. Alpha remains zero.

### 7.1 Link regimes

| Regime | Exact operation on auxiliary H |
|---|---|
| `exact`, severity 0 | Copy H exactly |
| `drift`, severity 0.15 | Add 0.15 times an independent random unit vector, then clip to signal support |
| `random_false`, severity 0.2 | Independently select links with probability 0.2 and replace their H with an independent signal draw |
| `systematic_false`, severity 0.2 | Independently select links with probability 0.2 and move H in an upward-f direction, clipping to support |

In the drift case, 0.15 is a displacement in standardized three-dimensional H-space, **not** a 15% false-link share. Clipping can make the actual displacement smaller.

In false-link cases, 0.2 is a Bernoulli selection probability per offered edge. The realized share need not equal exactly 20%. False links are treated as stable by the estimator. Their actual gaps and false-link indicators are available only as simulation diagnostics.

For systematic errors, the simulator uses the normalized gradient of true f and candidate step sizes `1, .5, .2, .05, .01, .0001`, multiplied by `false_jump=1`. It retains the admissible candidate with largest upward change from the source. The learner is not given the gradient or that true change. This is a deliberate one-sided contamination scenario, not a model of an empirically estimated change detector.

### 7.2 Three methods per link regime

| Method | Temporal coefficient |
|---|---|
| `none` | 0 |
| `temporal_fixed` | 0.3 |
| `temporal_tuned` | Chosen from `[0,.03,.3,1]` by observed validation misranking |

Zero is included, so tuning can turn off the temporal information. When lambda=0 is selected, the effective selected link count can be zero even though links were offered. `offered_link_*` columns record the offered pool, while the ordinary `link_*` columns describe the selected active link type.

The labeled source data, fitting/validation split and tests are held fixed across link regimes within a replication. The unlinked baseline should therefore be identical across those regimes; its repeated appearance is a matched comparison, not four independent draws of a baseline.

E4 varies **the kind/severity of link error at fixed C**. It does not provide a learning curve over cities, and this particular battery does not vary the false-link share over a broad grid.

There are `30 reps * 4 link regimes * 3 methods = 360` selected fits. Per replication/regime, the baseline and fixed method each use three starts, and tuning searches four weights with three starts: 18 candidate fits. Total: 2,160 planned candidates.

## 8. Evaluation, uncertainty and interpretation

### 8.1 Test data

Every design generates independent new units in seen cities and in new cities. Each test city contains 128 observations, 64 per date. The unseen-city count is `max(4, floor(C/2))`. The code samples 3,000 pairs for each of three latent evaluation types:

- **Within:** same city-date cell.
- **Across time:** same city, different dates, different sampled units.
- **Cross city:** different cities, with dates unrestricted.

The across-time metric does not compare a source to its matched stable auxiliary view. The test panels are independent of fitting sources and their auxiliary views.

Different C designs have different deterministic data seeds. The curves average independent simulation datasets at each C; the samples are not nested as C increases.

### 8.2 Primary misranking metric

For evaluated pairs, an error is a disagreement between predicted ordering and the ordering of true f. True-f ties are excluded; a predicted tie against a nonzero true difference receives half an error. `unseen_within_latent=0.02` means approximately 2% of sampled within-cell pairs in new cities are misordered.

It is not monetary prediction MSE and is not observed-label classification error. The suite also records observed comparison errors and the oracle true-f versus noisy-label error separately.

### 8.3 Normalized difference MSE

For logistic losses only, the current implementation reports

\[
\frac1{K}\sum_{(i,j)\in test}
\left[\frac{s(X_j)-s(X_i)}{\sqrt{\widehat V_{fit}(s)}}
 -(f(H_j)-f(H_i))\right]^2.
\]

Normalization is determined by fitting comparisons. No best scale is fitted using test f. Within-cell differencing removes score intercepts; this metric tests latent spacing within cells, not recovery of city-level score offsets.

### 8.4 Monte Carlo uncertainty

Each replication is a complete new simulated sample of cities, labeled observations, auxiliary views and test units, conditional on the fixed structural function/rotations/calibration. Replications, not comparison pairs, are the units for Monte Carlo standard errors.

`summary.csv` reports the mean, sample SD, and SD/sqrt(R), with approximate intervals mean +/- 1.96 SE. Stock figures display +/- 1 SE. The separate analysis bundle uses t critical values, which is why those plotted intervals are wider.

`paired_replications.csv` subtracts each method's error from its matched baseline error. Positive gain means improvement over that baseline. E1's designated reference is `rank_common`; therefore a negative gain for `raw_levels` means raw levels lose to ranking. E3 and E4 use `none` as the reference, so positive temporal gains favor links.

`paired_summary.csv` summarizes those within-replication differences. Do not infer a paired confidence interval by comparing or subtracting two marginal intervals.

### 8.5 Scope of common-index recovery

The DGP includes observed city-constant nuisance cues. Within-cell comparisons cannot distinguish f from f plus a city-dependent score offset. Temporal links within the same city also leave that offset unchanged. Thus low within-cell misranking in unseen cities does not by itself establish globally comparable score levels or accurate cross-city rankings. The suite reports cross-city errors to make this limitation visible.

## 9. Random seeds, parallel execution, checkpoints and the collapse repair

The overnight battery uses simulation seed **20261015** and function seed **983**. The function seed fixes rotations and population calibration. Simulation seeds are deterministically derived using SHA256 of a JSON list of the root seed and job tags, taking the first four digest bytes as a little-endian integer.

Data seeds depend on replication, n and C. Fit seeds additionally depend on truth, fitted model and restart, and are shared across paired arms/methods at matched settings. Parallel scheduling does not supply the random seeds and does not create additional statistical replications.

`--workers 8 --threads 1` runs independent fit jobs in eight spawned processes. Candidate grids/restarts within a job remain serial. One parent writes SQLite transactions and exports CSVs. The three experiments run sequentially. Worker count can change on resume, while scientific and numerical configuration compatibility is checked.

Each job is saved only after its hyperparameter search, selected model evaluation and candidate records are complete. Interrupted jobs are recomputed. Model weights are not persisted. CSVs/figures are exported after completion; an error handler exports partial CSVs without figures. This explains the missing E3 charts in the 149-fit snapshot.

The recovery patch keeps the V>0 domain requirement. It does not introduce a variance floor or epsilon. Dead initializations receive deterministic retries; Adam collapse restores a valid checkpoint; an inadmissible L-BFGS probe restores its pre-refinement checkpoint. Recovery is recorded. Invalid audit probes are flagged rather than certified stable. If a restart never has an admissible initialization, its failure is recorded and other valid candidates can be considered; a job with no admissible candidate fails explicitly instead of dropping the simulation replication.

Successful valid-path fits already saved by the supported earlier releases are retained when resuming compatible checkpoints. New recovery counters distinguish repaired candidates from ordinary successful candidates.

## 10. Where to inspect or change each part

| Question | File/function |
|---|---|
| General DGP and mathematical conventions | `DESIGN.md` |
| Relationship to manuscript and theoretical qualifications | `PAPER_ALIGNMENT.md` |
| Available commands and output meanings | `README.md` |
| Checks actually executed on the original numerical implementation | `VERIFICATION.md` |
| Exact requested focused grids | The three `configs/overnight_*.json` files |
| Exact settings and runtime of a completed/partial run | Its `resolved_config.json` and `run_metadata.json` |
| Structural functions, scorer, objective, optimizer and metric formulas | `ordinal_mc/core.py` |
| Label generation, transforms, split, links and tests | `ordinal_mc/design.py` |
| Which factors/methods/weights become jobs | `ordinal_mc/experiments.py` |
| Restart/hyperparameter selection and checkpoint compatibility | `ordinal_mc/runner.py` |
| Process execution and thread limits | `ordinal_mc/execution.py` |
| Summary, paired differences and figure grouping | `ordinal_mc/report.py` |
| E1's logistic-only MSE restriction | `core.test_metrics()` |
| E4's exact drift/false-link generation | `design.alter_links()` |

The existing documents were distributed across several files and describe more cases than the focused run uses. This reference consolidates the actual focused procedure; an inactive configuration section or an available method is not evidence that it was run.
