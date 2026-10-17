# Exact design choices in version 2

`PAPER_ALIGNMENT.md` maps these choices to the manuscript. This file specifies the simulator so that its results are reproducible and interpretable.

## Signal, representation, and normalization

Draw group contexts independently. In each of two dates, draw tracts' base signal coordinates U uniformly on (-1,1)^3. With composition parameter d and first group context z_c,

    H_raw = sqrt(3) tanh(arctanh(U) + d z_c (1, 0.5, -0.5)).

Standardize coordinates using an independent fixed population calibration, not the training sample. At d=0 this is a standardized uniform signal; d>0 shifts its distribution by group while preserving a bounded support and one global score function. The standardized signal is mixed orthogonally with two bounded acquisition-noise coordinates and two bounded group-cue coordinates. Acquisition noise is redrawn across views with a date shift; group cues are deterministic given the group, so they actually reveal group membership. The learner receives the seven mixed coordinates, not H or group identity separately (except the explicit cell-effects benchmark).

For u=H times a fixed orthogonal matrix, the four raw truths are:

    linear:    u1 + 0.6 u2 − 0.4 u3
    quadratic: u1 + 0.7 u2 u3 + 0.4(u3²−1)
    smooth:    sin(1.2u1) + 0.5 tanh(u2) + 0.4 u2 u3
    regime:    u1 + 0.8 max(u2,0) − 0.6 max(−u3,0).

Each is centered and divided by the square root of 0.5 E[(f_j−f_i)²] for two independent signal draws in the same group, estimated on 100,000 independent population pairs. This is within-group dispersion, not total pooled dispersion, which differs when composition changes. Structural calibration and rotations are fixed by `function_seed`, and remain fixed across repetitions. The calibrated function is unchanged across C and n.

For baseline centered Gumbel unit errors, choose their scale to attain `bayes_target` under this same independent population calibration:

    mean Lambda(−|Delta f| / [noise_sd_effective sqrt(6)/pi]) = bayes_target.

Default target is 0.2. Setting `bayes_target=null` instead uses `noise_sd`. This matches baseline difficulty across truths/compositions separately from variance normalization. In Q2 the calibrated baseline scale remains fixed while the error law changes.

## Labels and transformations

Generate one error per labeled unit, then R=f(X)+noise_sd_effective*error. Pair labels are computed from those unit observations. They are never simulated as mutually independent Bernoulli comparisons.

Q1 uses positive labels so the log benchmark has its stated meaning:

    W = a_ct + b_ct exp(0.6 [R + delta kappa_ct h(R)]),

where b_ct>0, a_ct>=0, h is tanh or asinh, and delta>=0. These maps are strictly increasing. The four stages are:

- Common: a=0, b=1, delta=0.
- Multiplier: a=0, b varies by group/date, delta=0. Logs plus cell effects correctly remove this multiplicative distortion.
- Location/scale: a and b vary by group/date, delta=0. An additive shift in positive levels is no longer generally removed by logs plus cell effects.
- Shape: hold the location/scale heterogeneity at its reference setting and vary delta with cell-specific kappa. This isolates a further nonlinear departure.

The strength dial controls b heterogeneity for the multiplier stage and a/b heterogeneity jointly for the location/scale stage; it controls only delta for the shape stage. Composition variation is separately controlled. Baseline centered Gumbel errors with the delivered calibration and transform settings yield well-behaved finite moments; arbitrary extreme user settings should not be assumed to retain them.

Q2–Q4 use the affine observation map W=a_ct+b_ct R with positive b_ct, which is an allowed special case of the paper. In these experiments the regression-form residual is exactly e=b_ct*epsilon. This keeps the literal conditional-median formulation and the common/heterogeneous logistic formulations aligned in the baseline. It also isolates error and link departures from measurement-map departures.

## Comparison errors

Base errors are centered Gumbel with SD one. The Gumbel difference has logistic scale sqrt(6)/pi before multiplication by the noise scale. Other options:

- Standard normal.
- Student-t with degrees of freedom nu>2, multiplied by sqrt((nu−2)/nu). The *unit* error is Student-t; its difference is not itself asserted to be Student-t. Nu controls heavy tails of the difference.
- Centered, standardized chi-square, and standardized two-normal mixtures, retained as optional shape checks.
- Cell-scale Gumbel: multiplier exp(tau z_ct−tau²), population RMS one.
- Feature-scale Gaussian: multiplier exp(tau H1) divided by its uniform-reference RMS. At the Q2 zero-composition baseline this is approximately population RMS one, up to the numerical coordinate calibration. Conditional symmetry preserves the pair median restriction.
- Feature-scale centered Gumbel: same multiplier, but conditional difference medians can be nonzero. This deliberately violates the median restriction.

Scale standardization is a population statement; finite sampled groups can have different realized noise variance. Gumbel cell scales preserve the heterogeneous logistic form. Symmetric feature scales preserve the ordering condition but not an additive-logit probability representation. These facts do not force any particular fitted method to win or fail in finite samples.

## Estimators and objectives

The linear scorer uses standardized raw X. The basis scorer adds all quadratic monomials and fixed random cosine features. The NN is a ReLU network with widths [32,16] by default. Preprocessing is fitted on labeled fitting X only. Effective scorer weights/biases are bounded by `weight_bound`; this gives a bounded score class on the bounded covariate support. Scores are nonconstant in the admissible ranking domain.

For all ordered within-cell fitting pairs, use weight w_ij=1/[N(m_cell−1)]. Their weights sum to one and each source tract has total weight 1/N. Define

    V = 0.5 sum w_ij d_ij².

Then the actual ranking training objectives are

    common: sum w_ij softplus(−Y_ij d_ij)
    heterogeneous: sum w_ij softplus(−Y_ij rho_ct d_ij/sqrt(V))
    ordinal surrogate: sum w_ij sigmoid(−Y_ij d_ij/[sqrt(V) tau]),

plus the separate penalties lambda_t mean_temporal(delta_s²)/V and lambda_c mean_cross_group(delta_s²)/V. The denominator is differentiated through. Tied label pairs contribute zero to the ranking numerator and retain their original weights in V, following the longer note; labels are continuous in delivered DGPs.

A zero V raises an explicit error. The common-logit score absorbs its scale, and only its stability ratio is scale-invariant. The heterogeneous score is explicitly normalized, while each rho_ct is positive with no finite upper cap. Exact ordinal misranking is used for validation/evaluation. A fixed smooth bandwidth is an explicit approximation, not a consistency result.

The main presets use alpha=0. An optional alpha penalty uses squared effective scorer coefficients for linear/basis and squared path norm for NN. For heterogeneous/ordinal normalized scores it is also normalized by V; heterogeneous scales enter the penalty through mean rho_ct². It is a separate regularized estimator and is not included in the paper's displayed objective.

## Level benchmarks

Targets are built only from fitting labels and then globally centered/scaled on the fitting sample:

- Raw levels: W.
- Logs with cell effects: log W, jointly regressed on a global score and unrestricted training-cell intercepts. The intercepts are discarded for ranking within an unseen cell, where a constant does not affect order.
- Within-cell z-scores: (W−cell mean)/cell SD.
- Within-cell percentiles: empirical midrank CDF values.
- Oracle latent: true f(X), an explicit noiseless upper-information benchmark.
- Oracle inverse: the true inverse map applied to W, yielding noisy R. This separates oracle transformation knowledge from absence of label noise.

No additional calibration-anchor observations are invented. For nonlinear true f, “correct correction” still does not imply a misspecified linear scorer is adequate; truth and model class are separate design factors. All methods select hyperparameters using observed within-cell validation ranking, so the primary comparison is prediction of order rather than levels MSE.

## Links and metrics

Each fitting source has `views` temporal and cross-group auxiliary images. Exact links preserve H and f but alter the relevant nuisance coordinates. Both uses all available links of each type. Drift shifts each H by a vector of length at most the dial, clipping to support. Random false links replace a selected fraction by independent H. One-sided false links use bounded ascent steps on the true simulated f solely to construct an upward change. The estimator never receives that true change.

Validation and test tracts are separate from fitting sources and auxiliary views. Tests include both new tracts from seen groups and new groups. Across-time metrics compare different tracts in different dates of one group; cross-group comparisons may span dates.

For the logistic difference MSE, normalize the fitted score by sqrt(V_hat) from fitting pairs and compute mean[(Delta normalized_score−Delta true_f)²] on held-out within-cell pairs. Do not estimate a best scale against test f. This is the metric requested in the proposal; it can expose spacing and scale errors that ranking alone misses.

The held-out group count is held at max(4, floor(C/2)), with an independent context draw. All methods within a design share these groups and test pairs. Different C designs are independent simulation datasets indexed by the seed; they are not assumed nested. MC uncertainty uses independent repetitions of the full design.
