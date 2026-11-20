# One-shot E6 gradient analysis

Analyzed replication 1, 32 cities, quadratic truth/ReLU scorer, 512 labeled
observations per city, using the saved A/B/C checkpoints. The matching original
WSL runtime was used: Python 3.10.20, PyTorch 2.5.1, NumPy 2.2.6. No Monte Carlo
sources, fitting settings, results, or checkpoints were modified. The diagnostic
does not train on latent/test targets or select settings using them.

## Checks and scope

The fitting feature transforms and A's saved validation loss were reconstructed
to numerical tolerance. Component scorer gradients sum to the joint gradient
computed directly by autograd. The 20-step replay at lambda=10 reproduces the
saved selected nonlinear/affine C checkpoint's component losses and gradient
norms to tolerance. The source checkpoint SHA256 is unchanged. These checks
support interpreting the measured gradients as those of the original pipeline.

Saved B starts for both measurement truths, both heads and all three weights
were audited, along with the four selected C checkpoints. Additionally, four
20-step Adam paths used the SAME A scorer and SAME nonlinear-measurement affine
B head, varying only lambda in {0,0.1,1,10}. That fixed head is the B initialization
of the selected C candidate at lambda=10. These paths are diagnostics, not new
replicated experiments or a hyperparameter-selection procedure. The lambda=0
path still updates the head penalty and is not the original ordinal D arm.

## 1. The W gradient is not negligible

At that common starting point, the scorer-gradient norms are:

| Component/weight | Gradient L2 norm |
|---|---:|
| Rank logistic | 0.007311 |
| Weighted temporal stability | 0.007085 |
| Net E5: rank + stability (alpha=0) | 0.001657 |
| Weighted W loss, lambda=0.1 | 0.001466 |
| Weighted W loss, lambda=1 | 0.014658 |
| Weighted W loss, lambda=10 | 0.146583 |

The W/E5 norm ratios are 0.88, 8.84 and 88.44. The net E5 gradient is small partly
because rank and stability gradients nearly cancel at A. Even against either
individual component, W's gradient is about twice as large at lambda=1 and
twenty times as large at lambda=10. This rejects the simple explanation that W
has no effective scorer gradient in this replication.

The conclusion also holds across the actual different B initializations: at
lambda=10, W/net-E5 norm ratios range from 88 to 360 across the four map/head
combinations. Those comparisons additionally reflect differing fitted heads;
the common-head paths isolate the weight itself.

## 2. Level fitting conflicts with stability

At the common start, cosine(W gradient, stability gradient) is -0.537; cosine(W,
rank) is +0.555. A negative gradient dot product means descending W alone would
increase the stability penalty to first order. The W gradient is not simply a
stronger version of the ordinal/stability direction.

Along the actual lambda=10 Adam path, from step 0 to 20:

- standardized fitting W MSE falls from 0.254010 to 0.253112 (about 0.35%);
- the weighted S/V stability penalty rises from 0.000558 to 0.001006 (about 80%);
- the fixed validation criterion improves slightly, from 0.689583 to 0.689178.

The S/V increase is training-link movement, not a directly measured fresh-stable
test metric in this diagnostic. It provides a local mechanism consistent with
the main experiment's higher stable movement after joint training. It does not
prove that every joint update or replication behaves the same way.

## 3. Adam and step size matter

The first scorer parameter-update norm is approximately 0.0277 at all three
positive weights, despite the 100-fold gradient-norm difference between lambda
0.1 and 10. Adam's adaptive normalization makes gradient magnitude and update
magnitude different quantities. The update directions and later trajectories
still differ. Global clipping never activates in these audited paths, so it
does not explain this effect.

At lambda=10, the first step's local gradient predicts a validation improvement
of -0.000361, but the actual finite step worsens validation by +0.000241. The
joint training objective also increases. Four single-step probes from the exact
same state demonstrate that reducing both learning rates resolves this local
overshooting:

| Scorer LR | Head LR | Actual training-objective change | Actual validation change |
|---|---|---:|---:|
| 0.001 | 0.01 | +0.002911 | +0.0002413 |
| 0.0001 | 0.001 | -0.0002155 | -0.0000303 |
| 0.00001 | 0.0001 | -0.0000268 | -0.00000356 |
| 0.000001 | 0.00001 | about -0.00000273 | -0.000000361 |

At the smallest step, the observed validation change closely agrees with the
first-order prediction. This supports a finite-step/curvature explanation at
this point, rather than an incorrect gradient or a missing backward path.
The probe scales both learning rates together; it does not identify whether
the scorer LR, head LR, or their interaction is primarily responsible.

## Interpretation

The evidence favors two concerns over an insufficient level-loss weight:
(1) level fitting pushes against the stability objective, and (2) the current
joint Adam step can overshoot locally. Some large-lambda updates may therefore
be rejected by validation even when a smaller step in the initial direction
would improve its observable criterion. However, the 20-step path eventually
recovers a small validation gain, so the first-step overshoot alone does not
explain all selection outcomes.

Increasing lambda further is not supported by this diagnostic. A targeted next
comparison would hold lambda, A, B, and scorer-update budget fixed while reducing
joint learning rates, then use the same validation rule. It would still need
post-selection latent/detection evaluation: better observable validation alone
does not establish better latent cardinal recovery. No new default or Monte
Carlo setting has been adopted on the basis of this one replication.

Artifacts: `gradients.csv`, `short_paths.csv`, `step_size_probe.csv`,
`metadata.json`, `verification.json`, and `gradient_diagnostic.png`. Runnable
scripts are `verification/diagnose_e6_gradients.py` and
`verification/probe_e6_step_size.py` in the suite. Diagnostic outputs require
new folders/filenames on rerun to avoid overwriting previous evidence.
