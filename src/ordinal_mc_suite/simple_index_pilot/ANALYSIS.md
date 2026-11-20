# Five-shot linear index ladder

The separate [fair 10% calibration comparison](ANALYSIS_10PCT.md) freezes these same scorers and fits fresh heads for every method on matched calibration tracts in seen and unseen cities.

The simple design separated structural scoring from outcome calibration. The ordinal scorer with exact stability links had the smallest mean unseen latent-change RMSE in all three maps, about .0021. Joint level supervision gave essentially the same change error. In the comparison that held level and stability weights fixed, adding paired outcome differences raised mean latent-change and expected outcome-change error in all three maps. The correct conditional Gaussian likelihood and the calibrated level-loss methods remained strong outcome predictors.

The cubic map exposed a larger failure of the **linear outcome head** than of the linear index. In case C, fixed effects had expected outcome-change MSE .011381 using its own linear head. Calibrating a cubic head on the unchanged fixed-effects score lowered it to .000198. Correct shared-index Gaussian ML reached .000130. The result is consistent with an isotropic Gaussian design whose linear projection retains much of the single-index direction even when its outcome mean is nonlinear.

## Simulation and methods

The only evaluation draws were 1–5 under master seed 2026110917. Each draw paired 200 tracts per city in 20 cities at two dates. There were 160 fitting and 40 validation tracts per city, held out at tract level. All cases and methods shared inputs, city draws, error innovations and splits. Baseline features H₁, H₂ and A were independent standard normals, and the structural index was f = (H₁ + 0.5H₂) / √1.25. A was independently redrawn at date 1. Each city had 100 stable tracts and 25 tracts at each of −.4, −.15, +.15 and +.4 true structural change. Increasing H₁ by the true change times √1.25 delivered those gaps without clipping.

The city maps were fixed over dates.

| Case | Conditional mean |
|---|---|
| A | `a_c + f` |
| B | `a_c + b_c × f` |
| C | `a_c + b_c × (f + k_c × f³) / √(1 + 6k_c + 15k_c²)` |

Independent city draws came from the specified uniforms for intercept [−1,1], slope [.5,1.5], curvature [.05,.20] and noise SD [.2,.6]. The realized outcome W added independent centered Gaussian noise with that city's SD. Here the structural score is f, and the mapped conditional mean and realized W are separate targets. Noise enters the outcome after mapping.

Baseline structural variance is one. The cubic denominator makes its baseline signal variance equal to B. At date 1 the change mixture gives structural variance 1.045625. The cubic signal variance relative to baseline is about 1.059–1.087 over the curvature range. The maps were held fixed across dates. For baseline X ∼ N(0, I) and a unit direction β, the Gaussian moment identity is E[X f³] = 3β. Therefore the population linear projection of the cubic map remains proportional to β at baseline. The date-1 mixture can change the exact proportionality.

Every estimator sees all three features. Fixed effects uses a common raw slope vector and city intercepts. City-specific slope OLS estimates separate intercepts and three slopes by city and uses the mean slope as its global score. Both standard mixed models use statsmodels MixedLM REML with a fixed intercept and all three fixed effects. Fitting-only city-specific SD estimates use residual sum of squares from city-specific OLS divided by (n_c-4). They prewhiten the outcome and **all** fixed and random design columns, including intercepts. This is a feasible plug-in variance approximation and the SD includes linear mean misspecification in C. The all-feature diagonal model estimates four independent random coefficients. The two-stage index model estimates a random intercept and a random slope along the pooled FE direction, using the observed fitting data to generate that direction. Both use the fixed-effect slope vector as the structural score and fitted random effects for seen-city W predictions. The uniform coefficient draws make the Gaussian random-effects distribution an approximation.

Conditional Gaussian ML estimates the city means and unknown city noise SDs using the correct mean family. It profiles variance at SSE divided by the number of city endpoints. A estimates an unrestricted raw coefficient vector β with fixed unit map slope. B estimates a unit-norm direction and separate positive affine slopes. C estimates a unit-norm direction and separate cubic heads of the form a + ur + vr³, with u > 0 and v ≥ 0. This includes the true family without revealing the true curvature interval. At a given direction the B/C conditional head minimizes city SSE, also minimizing the city profiled log SSE. City coefficients are treated as unknown parameters, with no distributional requirement on their draws. Gaussian ML+S adds the same exact stability term used by the custom arms.

The custom losses used the same exact affine or cubic head families. Case A's common positive fitted slope times the normalized direction can represent every nonzero raw fixed-effects slope. A numerical check reproduced the FE fitted values. All ordinal terms used the bounded sigmoid loss on within-city-date pair margins with temperature .1. The stability term was .3 times mean squared score movement over two independent same-H acquisition redraws per fitting endpoint. Actual paired outcomes entered level and paired residual-difference losses, and changing tracts never served as stability links. The level denominator is pooled within-city fitting W variance and the difference denominator is fitting paired ΔW variance. In C, the curvature penalty is `36 × mean(v_c²) / V_W`, where the mean is over cities and `V_W` is the level denominator. This is proportional to integrated squared second derivative under a standard normal reference score. The curvature weight equals .001 times the branch's level weight.

**Ordinal_S is the frozen ordinal comparator.** It fits the ordinal scorer plus stability and then fits outcome heads on the frozen score, using validation to choose from the same exact-family library available to the other scores. Other methods report their own prediction heads, and the secondary equal-library columns put the same calibration options on all fitted scores. In case C, this distinction matters for FE and the mixed methods, whose own outcome heads are linear. [Configuration](config.json) · [Numerical preflight](preflight_final.json)

## Weight selection and optimization

The independent draw 101 generated fitting and validation data only. At its ordinal anchor, weights were scaled using the ratio of the joint norm of the separate rank and weighted-stability gradients to each cardinal gradient norm on the score sphere. The grid crossed relative budgets {0,.1,1} for level and difference and added (10,0) and (10,10). Separately, the pure L2+S family compared paired-difference weights .1 and 1 at fixed level weight 1. Observed validation rank + .3 stability + standardized level and change errors selected the weights, with lower budget breaking ties. The overall grid winner in B was ordinal-only. The cardinal-dominant (10,10) point won in A and C. Positive weights were also retained to compare the specified branches.

| Map | Level weight in ordinal + L2 | Level / difference weights in ordinal + L2 + Δ | Difference weight in pure L2 + Δ |
|---|---:|---:|---:|
| A | .02421 | 2.42062 / .23285 | 1 |
| B | .02013 | 2.01294 / .30690 | .1 |
| C | .02534 | 2.53447 / .42886 | .1 |

The ordinal branches above change the level weight as well as the difference weight, so their contrast is between two tuned packages. The pure L2 comparison holds the level weight fixed and isolates adding differences at the selected positive weight. All weight selection used observed validation criteria, and the first evaluation test panel was constructed after the weights were frozen. The grid examines a limited set of weights on one calibration draw per map.

Direction searches used a 48-direction spherical grid and four BFGS starts, with analytic profiled gradients. For the 115 evaluation optimizer-based fits, four starts reported success in 101 fits, three in 13, and two in one. In 51 fits the final objectives across starts spanned more than 1e−6, particularly for loss families with near-opposite orientations and positive map slopes. The selected fits beat or matched the grid and their largest tangent or raw gradient norm was 1.15e−7. Solver status across starts and the selected stationarity check are distinct. This is a numerical multistart check rather than a proof of the global optimum.

The two-stage index mixed model converged in all 15 fits. The diagonal model converged in 10 of 15, with two failures in A, one in B and two in C. All returned predictions are retained, with convergence and boundary warnings attached. The unrestricted covariance failed to converge in A and B in preflight. Estimated covariance near the boundary is expected when proportional random slopes have low rank. The predictions use BLUPs that agree with the package where it returns them. [Calibration candidates](run_5rep/calibration_candidates.csv) · [Frozen weights](run_5rep/frozen_selection.json) · [Optimizer and mixed-model diagnostics](run_5rep/optimizer.csv)

## Structural change and outcome prediction

Each cell below is the mean across five matched draws. Structural RMSE is on the changing tracts in the unseen mixed panel. Expected W error is on all paired tracts in the seen mixed panel, using the method's own head.

| Method | Unseen latent RMSE A / B / C | Seen expected W change MSE A / B / C |
|---|---:|---:|
| Fixed effects | .004951 / .005579 / .006502 | .000049 / .004299 / .011381 |
| City-specific slopes | .006191 / .006763 / .004179 | .001555 / .001555 / .008927 |
| Mixed, two-stage index | .003711 / .004020 / .003976 | .000022 / .000055 / .007184 |
| Mixed, diagonal | .003810 / .004288 / .003635 | .000069 / .000112 / .007220 |
| Correct Gaussian ML | .003707 / .003068 / .004198 | .000022 / .000042 / .000130 |
| Correct Gaussian ML + S | .003241 / .002629 / .003642 | .000017 / .000038 / .000125 |
| L2 + S | .002954 / .003557 / .004015 | .000018 / .000047 / .000134 |
| L2 + Δ + S | .006112 / .004421 / .005495 | .000054 / .000057 / .000159 |
| Frozen ordinal + S | .002051 / .002055 / .002118 | .000011 / .000046 / .000147 |
| Ordinal + L2 + S | .002091 / .002081 / .002125 | .000009 / .000035 / .000116 |
| Ordinal + L2 + Δ + S | .004722 / .005519 / .007064 | .000036 / .000072 / .000196 |

Adding the same exact links to Gaussian ML lowered changed-tract RMSE in all 15 case–draw pairs. FE and mixed models lack these external links, so their comparison with ordinal + S combines information and objective differences. Gaussian ML+S uses the links, although the same numerical stability weight can have different relative leverage with likelihood and rank losses. The ordinal scorer suppressed the nuisance coefficient and had lower stable RMS (.00184/.00179/.00171) than ML+S (.00316/.00251/.00358). In B and C, ML+S had lower angular error than ordinal alone. Small angular error and small paired change error therefore rank methods differently. In A, the raw coefficient error was .00713 for FE and .00625 for ML. These results concern finite held-out change risk with this design.

Let b₁ and bₐ be the fitted unit-norm coefficients on H₁ and acquisition A, respectively. A tract's latent change error at true change d is exactly

**Change error = (√1.25 × b₁ − 1)d + bₐ(A₁ − A₀).**

On an independent changing panel the expected squared error conditional on the fitted coefficients is the sum of .09125 times the squared dose-slope error and 2bₐ². The cross term averages to zero by independence. Acquisition contributed 90–94% of this model-implied error for ordinal + S and 98–99% for Gaussian ML+S, averaged within cases over the five fits. In C, ordinal's smaller acquisition term (6.73e−6 versus 1.63e−5) outweighed its larger dose-slope term (7.45e−7 versus 1.95e−7), despite a larger mean angular error (.303° versus .226°). This decomposition explains how stronger nuisance suppression can improve change RMSE even when direction recovery is weaker. [Coefficient-based error decomposition](results/change_mse_decomposition_by_rep.csv)

For pure L2, adding paired differences raised latent RMSE in 5/5 A, 4/5 B and 5/5 C draws and raised expected W change MSE in 5/5 A, 3/5 B and 4/5 C. It uses the same two noisy endpoint labels, so it reweights residual contrasts. The joint ordinal plus level branch changed unseen RMSE little relative to the frozen scorer, with a modest mean improvement in expected W change. In C, adding the fixed cubic curvature penalty to ordinal + levels changed expected W change MSE from .000116 to .000114 and left latent RMSE about .00212. The larger difference between the tuned regularized full branch (.000195) and regularized level branch (.000114) also includes their different level weights.

Case C expected-W level MSE was .143756 for FE and .053182 for the index mixed model with their own heads. Their equal-library cubic calibration yielded .002165 and .002132. Correct Gaussian ML with its own cubic mean gave .001751, and ordinal + level + stability gave .001759. Expected changes show the same distinction. This quantifies the impact of the head family independently of the latent change RMSE. The realized change noise floor was about .357881 across cases. The zero-change predictor's expected change MSE was .045625/.043454/.049593, so the small expected differences among well-calibrated methods were largely obscured on the realized scale. [Means](results/means.csv) · [All per-draw metrics](results/by_rep.csv) · [Matched differences](results/paired_by_rep.csv)

Paired differences below subtract the second method from the first. A win means a lower error on a matched draw. Brackets give the minimum and maximum difference across the five draws. The supplementary table includes paired means, ranges, and lower/higher/tie counts for all reported contrasts and the six key error and coefficient metrics.

| Comparison | Map | Unseen latent RMSE difference, mean [range] | Wins | Own expected-W change MSE difference, mean [range] | Wins |
|---|---|---:|---:|---:|---:|
| Ordinal + S vs Gaussian ML + S | A | −.001189 [−.004196, +.002142] | 4/5 | −.000006 [−.000030, +.000020] | 4/5 |
| Ordinal + S vs Gaussian ML + S | B | −.000574 [−.002075, +.001566] | 4/5 | +.000008 [−.000016, +.000056] | 3/5 |
| Ordinal + S vs Gaussian ML + S | C | −.001524 [−.003090, +.000100] | 4/5 | +.000022 [−.000023, +.000149] | 3/5 |
| L2 + Δ + S vs L2 + S | A | +.003158 [+.000760, +.005609] | 0/5 | +.000036 [+.000017, +.000081] | 0/5 |
| L2 + Δ + S vs L2 + S | B | +.000864 [−.000310, +.001863] | 1/5 | +.000011 [−.00000023, +.000021] | 2/5 |
| L2 + Δ + S vs L2 + S | C | +.001480 [+.000751, +.002484] | 0/5 | +.000025 [−.00000755, +.000064] | 1/5 |
| Ordinal + L2 + S vs frozen ordinal + S | A | +.000039 [−.000024, +.000163] | 2/5 | −.000002 [−.000013, +.000002] | 3/5 |
| Ordinal + L2 + S vs frozen ordinal + S | B | +.000026 [−.000051, +.000149] | 2/5 | −.000011 [−.000050, +.000001] | 3/5 |
| Ordinal + L2 + S vs frozen ordinal + S | C | +.000007 [−.000170, +.000240] | 3/5 | −.000031 [−.000158, +.000008] | 2/5 |

[Paired means, ranges and counts](results/paired_sign_counts.csv)

## Detection and scope

Each seen and unseen test panel used 5,120 tracts. A separate independent stable panel supplied the higher empirical 95th percentile cutoff for each score. All linear scorers had one acquisition coefficient. Stable movement is therefore the absolute value of that coefficient times the common acquisition difference. For nonzero coefficients this scale cancels in the strict exceedance rule and every method had the same mean fresh stable false-positive rate, 4.625%. Stable RMS and the cutoff itself still differ. Across all five draws, ordinal + S detected and signed all nominal ±.05 doses on the unseen panels. FE detection at those doses ranged from about 97% in B to about 99.9% in C, and its direction was almost always correct. O mean structural dose-response slopes were 1.00040/1.00048/1.00048. The narrow change errors and high detection rates are features of this simple linear rung.

The lower- and upper-noise city halves had similar latent change RMSE. For example C ordinal + S was .002112 and .002124. City noise affects the labels and fitting, but the images in the test panels have the same structural and acquisition laws across noise groups. Both groups' outcome and structural results, including all signed doses, are recorded. Directions use the analytic unit norm under baseline feature covariance I. This compares normalized linear scores under an explicit structural restriction and does not identify arbitrary monotone reparameterizations in the broader A4(a) model. With five draws, the results and paired ranges are descriptive. [Dose and noise-stratified tables](run_5rep/structural.csv) · [Comparison figure](results/structural_outcome_comparison.png)

## Reproduction and audit

The run completed 175 fitted estimators and 39 validation-only candidate fits in 71.1 seconds. The audit verified 72 distinct panel seeds, exact fitted-head and Gaussian variance reconstruction, FE fitted values to 3.6e−15, and the outcome change residual decomposition. All earlier pilots are separate. [Audit](results/audit.json) · [Fit manifest](run_5rep/manifest.json) · [SQLite results](run_5rep/results.sqlite)

Run from the suite directory with the existing Windows Python environment. Choose fresh destinations.

```text
python -B simple_index_pilot/run_simple.py --out NEW/run
python -B simple_index_pilot/report_simple.py --run NEW/run --out NEW/results
```
