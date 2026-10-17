# Audit against the supplied proposal and technical note

The October 5 proposal (`Pasted markdown(7).md`) is the primary specification. The longer technical note (`Pasted text (2)(2).txt`) supplies the detailed common-logit criterion, tract weights, normalization, and sampling assumptions. Neither manuscript has been edited. This package corrects the implementation; qualifications about the manuscripts are identified separately below.

## Answer about the three losses

The proposal explicitly says the ordinal criterion is trained through a smooth surrogate. Therefore using a smooth approximation was intended. The earlier response suggesting that smoothing itself departed from the proposal was too strong. What remains unspecified in the proposal is the precise surrogate, bandwidth, and smoothing-error argument.

Let Y=2y−1, d=s(X_j)−s(X_i), and V=0.5 E[d²]. The revised implementation is:

| Version | Manuscript | Revised implementation |
|---|---|---|
| Common logit | E[log(1+exp(d))−yd], scale absorbed into s | Weighted mean softplus(−Yd). No extra fitted rho and no division of this margin by sqrt(V). |
| Heterogeneous logit | E[log(1+exp(rho_ct d))−y rho_ct d], V(s)=1 | Define the actual normalized score as raw_score/sqrt(V(raw_score)), then fit positive rho_ct. This implements the normalization, rather than adding a different comparison model. |
| Ordinal | Misranking probability, half loss for predicted ties, trained through a smooth surrogate | Weighted mean sigmoid(−Yd/(sqrt(V) tau)); tau is explicit. Exact misranking with half ties is used for validation and evaluation. |

Softplus is a numerically stable evaluation of the paper's binary cross-entropy, not an alternative logistic loss. The +/-1 coding is exactly equivalent to the paper's 0/1 coding.

The previous common-logit implementation had a normalized score and a bounded fitted rho. Without bounds and with compatible function classes, a free rho could reparameterize the unnormalized score. Its hard bounds and regularizer meant that calling it literally the paper's estimator was inaccurate. That departure is removed. Heterogeneous scales now use a positive softplus parameterization with no finite upper cap.

## Implementation discrepancies corrected

| Item | Previous package | Paper-aligned revision |
|---|---|---|
| Stability terms | Both arm split a fixed edge budget and averaged the two types | Separate lambda_t S_t/V + lambda_c S_c/V; both arm uses all available links of both types. |
| Tuning both links | One scalar weight for a mixed pool | Independent Cartesian grid over lambda_t and lambda_c; either or both can be zero. |
| Additional complexity penalty | Selected alpha by default | Alpha=0 in paper presets. Optional regularization is explicitly a sensitivity analysis, not part of the stated criterion. |
| Score class | Unbounded network coefficients and unbounded nuisance covariates | Bounded covariates and bounded effective scorer coefficients, consistent with the long note's compact-class/bounded-score setup for the common-logit result. |
| Pair measure | Cell-uniform sampling, equivalent to tract weights only in balanced cases | Exact ordered within-cell pairs by default, weight 1/[N(m_cell−1)]; sampled tract-uniform pairs remain an explicit option. |
| Zero dispersion | Added epsilon to V | Positive V required; a zero-dispersion ranking score raises an error rather than silently changing the domain. |
| Sample-size axis | Tracts per city grew, number of cities fixed | Q1 and Q3 grow the number of labeled groups C, with n tracts/group fixed. n remains an optional secondary design dimension. |
| Q1 competitors | Raw levels, inverse corrections, invented external calibration-anchor dataset | Pooled levels, log outcomes with jointly fitted cell effects, within-cell z-scores, within-cell percentiles, and oracle benchmarks. No external anchor dataset is assumed. |
| Q1 distortion design | Only one nonlinear departure from an affine map | Common map, cell multiplier, location/scale, and nonlinear shape stages, plus variation in the distribution of H and f across groups. |
| Q2 tails | Distribution substitutions but no direct heavy-tail dial | Variance-standardized Student-t unit errors, with degrees of freedom controlling the tails of their differences. Other original families remain available. |
| Q3 nuisance factors | One combined nuisance-strength parameter | Separate acquisition-noise and group-cue controls, varied one at a time. With no acquisition variation, temporal exact links are identical X and add no information. |
| Q3 losses | Common logit default | All three losses in paper presets. A focused common-logit run is separately configured. |
| Q4 links | Temporal by default | Temporal, cross-group, and both arms, each fixed and tuned; exact, gradual, random false, and one-sided false links. |
| Evaluation | Rankings only | Add normalized within-cell index-difference MSE for both logistic losses, using fitting-sample dispersion and no oracle rescaling. |
| Baseline difficulty | Equal approximate variance of f only | Independently calibrate within-cell V(f) and Gumbel noise to match a baseline observed-label Bayes misranking target. |

The effective scorer coefficients are B*tanh(raw/B), B=10 by default. The optimizer searches the interior of a fixed bounded coefficient class; its closure is compact. This is not a new penalty. It can restrict approximation, as any fixed class can, and B is recorded in the configuration. Raw optimization variables are auxiliary; the bounded effective coefficients are the statistical class parameters. Heterogeneous normalization and growing cell scales still require their own theory.

## Necessary qualifications to the manuscript

### 1. Unit variance does not equalize irreducible ranking error

Under common logistic differences, the Bayes observed-label misranking risk is

    E[Lambda(−rho * |Delta f|)].

V(f)=1 fixes the second moment of Delta f, not this nonlinear expectation. Two distributions of Delta f with the same second moment can have different risks. The code now explicitly matches this baseline risk, using a fixed independent population calibration draw, rather than claiming the variance normalization implies it.

Suggested narrow clarification: “We normalize V(f)=1 and separately calibrate baseline noise so that the oracle observed-label misranking rate is matched across structural designs. Under error misspecification we retain that baseline noise scale and report the resulting oracle error.”

The equalization is numerical, not an exact analytic population identity: both within-cell dispersion and risk are calibrated on 100,000 independent population pairs. Q2 deliberately changes error shape/reliability and therefore need not retain the baseline Bayes error. In a median-violating scenario, f need not be the Bayes ordering; the exported f-versus-label error is then an oracle-index diagnostic, not a Bayes-risk claim. The latent-f ranking target itself has zero irreducible error when f is known.

### 2. The two model displays use different transformations

The technical note uses W=Psi_ct(f(X)+epsilon). The proposal's main display uses W=psi_ct(f(X))+e, and its footnote explains the connection. With feature-independent epsilon within a cell and suitable integrability,

    psi_ct(u) = E[Psi_ct(u+epsilon) | c,t]
    e = Psi_ct(f(X)+epsilon) − psi_ct(f(X)).

Then E[e|X,c,t]=0 and psi is increasing, but psi generally differs from Psi. One cannot put the same arbitrary iid additive e after a nonlinear psi and still obtain a logistic comparison probability in Delta f. The package keeps the technical note's unit-level mechanism for the logistic designs.

For the error-assumption comparison Q2, the fixed observation transformation is affine, so e=b_ct epsilon and the literal median-difference restriction in the proposal is satisfied by the symmetric feature-scale scenarios. Q3/Q4 use the same affine baseline to keep all three comparison assumptions aligned. Q1 studies nonlinear transformations under the proposal's directly stated logistic probability restriction and its latent-noise construction. No median-zero assertion about the induced additive e is inferred merely from that construction.

### 3. Cross-group links need a different sampling argument

The longer note's iid city totals contain only within-city comparisons and temporal links. A cross-group link depends on two city draws, so including it in a city total does not leave independent totals of the same form. Cross-group simulation arms remain appropriate, but the existing iid-city ULLN proof cannot be cited unchanged for them. One must specify and analyze the link-sampling design, for example as a cross-cluster pair statistic. Monte Carlo SEs here use independent whole-simulation repetitions, so they do not rely on that unproved within-run extension.

### 4. Heterogeneous scales introduce an identification issue as well as a growing-parameter issue

A single global normalization need not identify all relative cell scales. If X reveals group membership, scores of the form s(x)=a_c f(x)+b_c, with correspondingly adjusted rho_ct, reproduce the same within-cell logistic probabilities. Exact temporal stability can hold for these scores with a group-specific a_c. Global V(s)=1 removes one common scale, not every group-specific scale.

Thus the proposal's “same holds jointly with the scales” needs an overlap/anchor/scale-identification condition before it becomes a theorem. This is not solved solely by controlling the number of rho_ct parameters. The heterogeneous difference-MSE output is a recovery diagnostic, not an assertion that current assumptions guarantee its convergence to zero. Cross-group stable links may constrain these ambiguities if their coverage is informative; merely having a nonempty set is insufficient.

### 5. Approximation and smoothing qualifications must remain explicit

With fixed positive lambda and approximate links, even the longer common-logit theorem gives a neighborhood bound containing lambda L² delta² and class approximation error. A fixed class or fixed nonzero smoothing bandwidth does not by itself give exact consistency for f. The proposal already mentions smoothing error, which is the correct qualification. This package retains that distinction.

No sigmoid surrogate is uniquely prescribed by the paper. The chosen surrogate is a declared computational choice, not an additional error-distribution assumption. Bandwidth sensitivity is necessary before using its MC results to claim robustness under only median restrictions.

## Computational choices retained, with their limits

- Full-batch Adam plus L-BFGS evaluates the numerator and denominator at the same parameters and differentiates through V. For these small simulations it avoids minibatch-ratio bias. It does not change the empirical criterion. The proposed large-image minibatch implementation is not included.
- One labeled tile per tract is the current specialization. The technical note's multiple-tile average reduces to this case; multi-tile label aggregation is not tested.
- The group-cue block is a bounded deterministic function of group context, and acquisition noise is redrawn across views. This makes the proposal's two nuisance mechanisms separately controllable.
- Smooth polynomial/interaction/ReLU truths and a small MLP instantiate the proposal's linear/nonlinear cases. They are not satellite imagery or a pretrained image backbone.
- Gradual link drift is controlled in standardized H distance, with the realized f gap reported. This implements the Lipschitz premise; it does not force an identical numerical f gap for every link.
- Fixed-function and finite-sample MC curves do not prove an asymptotic rate. Growing-group curves follow the stated asymptotic axis but do not establish a theorem.
- Optimizer audits measure additional objective improvement and gradients, not global optimality. Approximate minimization remains a substantive condition.

The original package's numeric results must not be combined with the revised runs: the objective, pairing measure, DGP controls, and sample-size axis changed. Use fresh output directories.
