# Presenter notes

## Title
0:15. A measurement agenda motivated by economic panels. Target 35 minutes including the forthcoming simulation figures, leaving about 25 minutes for lab discussion.

## 1 — Can imagery fill gaps between surveys?
1:00. Motivate the economic problem before the method. Satellite maps supported geographic targeting, not direct household identification. Source: https://cega.berkeley.edu/collection/ai-assisted-cash-transfers-togo/ . The World Bank Geospatial Poverty Portal provides a second example of institutional demand.

## 2 — Good rankings; economic change remains difficult
1:00. Preview findings now, following Shapiro. The 35/36 comparison is a comparison of fitted runs, with unequal total training budgets (750 versus 200 epochs), not a clean equal-compute loss ablation. Change result is after GB2 calibration. Source: run_20260722 evaluation tables and NYC F_construction_scatter.pdf.

## 3 — Four questions in one research agenda
0:30. Four connected research questions, not completion percentages or a promise of four papers. Global pretraining plus local survey calibration is the ambition; causal use requires additional validation.

## 4 — What existing imagery models already deliver
0:45. Avoid claiming that nobody predicts change. Tempov reports decadal asset-wealth change R2 around .69 and .46 in Malawi and Mozambique. Source: https://arxiv.org/html/2604.23166v1 . Khachiyan et al.: https://doi.org/10.1257/aeri.20210422 . The contribution is a measurement framework, not merely applying AI to a new geography.

## 5 — The US measurement setting
1:00. Establish observation units early. Illustrative local imagery from the previous deck; these are not certified training crops. Logged tau is 112m, after snapping requested 100m; 224 pixels. Data sources: USDA NAIP, ACS, Microsoft US Building Footprints. Saved run log and debug_batches/dump_info.json support geometry. Avoid claiming household or building wealth is directly observed.

## 6 — The label is a constructed wealth measure
1:15. Formula checked against src/data/process_acs.py annuity_factor, equity_pc and wealth_from_inputs. Ordinary annuity begins at period 1, not 0. Mortgage share m is required by code. Use r_d for discount rate to avoid collision with the comparison-error precision rho. This describes the implemented index, not an independent validation of its economic construction. LTV source and potential aggregate/share interpretation are in backup.

## 7 — Similar structures can carry different dollar labels
1:00. Six images will arrive tomorrow. Replace exact filenames in figures/puzzle; no invented labels or footprint overlays. Give each its actual tract/year label using the configurable labels in the preamble if desired. Similarity is illustrative, not evidence that A2 holds. Keep this slide to one minute; footprints can be discussed using a backup image if useful.

## 8 — Image changes need not be economic changes
0:45. Images copied from supplied NYC time-series location. Inspect structures and acquisition differences live; do not assert that the entire scene is physically unchanged. Label-scale variation and acquisition variation are distinct problems. Stability restrictions concern selected pairs, not every location.

## 9 — A common index with a local measurement scale
1:45. Define all notation explicitly. Moving an additive outcome error through an invertible psi is only a reparameterization; it does not imply latent mean zero or logistic differences. Those restrictions are additional. Model is not an empirical proof that all city differences are monetary rescalings. Pooled-level proposition and derivation in backup.

## 10 — The central economic restriction is invariance
1:00. This is where the audience should interrogate the model. A pool or density is not guaranteed to mean the same thing everywhere. Shared function plus geographic cues can also reproduce city-specific behavior on disjoint supports. No claim of universal extrapolation. Units here are tract labels paired with local views.

## 11 — Stable pairs add a second source of information
1:15. A3 restricts the true systematic score, not realized occupant wealth. No footprint change is an imperfect proxy, not a proof. L and delta are not observed; tightening the selection rule is intended to improve pair validity, not guaranteed to reduce latent delta. Occupant sorting and earnings can change without visible structure changes.

## 12 — Within-cell comparisons eliminate the local scale
1:15. Logistic is imposed on epsilon_i minus epsilon_j, not on individual residuals. Common rho fixes a common comparison-noise scale. Feature-dependent residual dispersion is excluded by this version. The weaker median-zero version requires a suitable strict crossing condition; results for it are not yet claimed.

## 13 — The criterion combines ranking and stability
1:30. Labels enter only through order, provided pair selection and weights preserve that invariance. Define ratio as infinite when V=0. Cross-city pairs omitted in this criterion; adding them changes the analysis unless their validity is established. Ranking term itself is not scale-free; only the penalty is. Normalization V(f)=1 used in theorem.

## 14 — Population recovery has a price for imperfect pairs
2:00. Present as supplied theoretical result, not a newly independently peer-reviewed proof. Norm is mean squared error of within-cell differences. Exact zero slack is special. Logistic curvature turns excess risk into difference error. Finite bound may be loose; it is not a confidence band and does not establish finite-sample improvement.

## 15 — More cities remove sampling error, not every error
1:30. Require bounded city and pair counts, positive expected counts, score continuity and optimization error tending to zero. Sampling asymptotics are C to infinity with fixed T. B=max(2 Sbar,rho L D). This result is an asymptotic error bound, not pointwise consistency on all new cities; common superpopulation and support matter.

## 16 — Cross-city alignment is still an open problem
0:45. Offset indeterminacy is especially transparent on disjoint city supports. Overlap can restrict separate offsets, so do not claim arbitrary city shifts are always feasible. Do not repeat the unsupported 2 rho L delta bound for learned offsets. Cell common shifts may cancel in an appropriate within-city differencing design, but that is not a generic exemption for every event study.

## 17 — Monte Carlo: robustness to non-comparable labels
1:30. Results arrive tomorrow; do not state a simulated finding before importing the actual outputs. Introduce DGP: common f, cell-specific psi, comparison-error assumption; report cities, dates, units, replications, and Monte Carlo uncertainty with final figures. Use same held-out metric and axes across panels. Broad design from supplied outline; parameters must come from the finished run.

## 18 — Monte Carlo: what do temporal pairs add?
1:30. User previously reports this experiment completed; plot to arrive tomorrow. Do not invent gain or number of repetitions. Explain why it tests added information when the ranking model is correctly specified. Keep model class, labels, tuning, and evaluation constant; distinguish label budget from image/computation budget.

## 19 — Monte Carlo: does the gain survive flexibility?
1:30. Results will be added tomorrow. Prespecified nonlinear truth; plot no links versus links under the same tuning budget. If unfinished, omit this slide and describe it as next experiment. Imperfect/false links design in backup.

## 20 — The current US run approximates the framework
1:15. Weights from saved run log: lambda_s=.3, lambda_var=1.5, temporal_fraction=.4. Rank loss uses current training implementation, including pair-mining/sampling; do not claim it verifies A4(c). A soft variance penalty neither fixes V=1 exactly nor generally duplicates the ratio objective. Theory is guidance for an improved implementation; no claim that current empirical estimator is exactly covered.

## 21 — Generalization tests separate cities, dates and sensors
1:00. City partitions from supplied split map and manuscript. Separate buffered NYC/Chicago tract holdouts can exist within training cities. Avoid absolute no-spatial-leakage claim across every evaluation. Temporal heldout year is interpolation, not future forecasting. Full split and selection details in backup; source map is inherited, not regenerated.

## 22 — Tract rankings transfer to held-out cities
1:00. Replotted from evaluation/US_summary.json l2/bracket/rho and l2/all entries. This is matched-cell, tract-aggregate comparison. Do not mix with US_A_summary.csv building-level within Spearman .580 or pooled tract .727, which use different units/samples. L2 selected from 3 learning rates; full runs 750 vs 200 epochs.

## 23 — The ordinal run improves rankings in 35 of 36 cities
0:45. Replotted without changes to data from US_L2_city_distance.csv, sorted by difference. Broad advantage of these fitted runs. Do not infer causal effect of loss alone. Matched-budget selection diagnostics in backup show advantage remains at <=200 epochs, but are validation scores, not matched test-city results. Signed-distance slope includes zero; original figure retained in backup.

## 24 — Persistence is high; detecting change is harder
1:00. Exact values from US_B_rank_autocorr.csv. Autocorrelation summarizes stable-building pairs by city with pair-count weights; movement statistics use all available year pairs in each flag group. Mean absolute score displacement is in raw score units, not standardized change. Sign agreement is with label change in changed group (ties excluded). Different denominators; do not infer statistically significant separation from 1.02 ratio.

## 25 — Construction tests responsiveness of the fitted index
1:00. q denotes construction cohort, avoiding conflict with head g. Callaway-SantAnna estimates on generated outcome require identification assumptions, including appropriate parallel trends and anticipation. Lipschitz bounds near pairs; does not imply direction for construction. Economic expectation of positive score change is a hypothesis, not a theorem. New construction is heterogeneous and need not always house wealthier residents.

## 26 — NYC scores rise around construction
1:15. Top held-out row of the original plot, not a newly estimated event study. All three panels and confidence bands retained. First-threshold p=.570. Anticipation two means event adoption shifted two periods (~four years) early as in original title; labels refer to recorded construction. Do not claim clean causal dose response or robust identification. Full train/test figure and other cities in backup.

## 27 — Fine spatial detail, with tract-level validation
0:45. Inspect original legend and dates. Inherited map compares ACS and model geography at different resolution. Use as a visual illustration, not proof of within-tract accuracy. Building scores inherit tract supervision, and local tiles contain surrounding neighborhood context.

## 28 — Held-out tract changes remain poorly predicted
1:45. Replotted from explicit heldout columns of F_construction_scatter.csv. Heldout n=243,371,196,86 for stable, >=1%, >=5%, >=10%. Spearman .101,.283,.327,.411; predictive R2 -1.304,-.566,-.521,-.682. These are point estimates without uncertainty intervals. Full mixed-sample scatter retained in backup. Both sides use GB2 dollar mappings; not raw Delta f, and mapping uses target-year survey information. Negative predictive R2 indicates poor change-level calibration, even when rank correlation is positive.

## 29 — Temporal measurement requires more than stable images
1:00. Corrects original assertion that all label change at stable places is epsilon. That ignores psi changes. Do not claim most tracts meet A3 or delta f=0 by design. Noise-ceiling calculation needs covariance across ACS releases, generated-index uncertainty and measurement-error assumptions; a simple formula is not an empirical ceiling for Delta f.

## 30 — What is established, and what comes next
1:15. End on the substantive result and three research decisions. Monte Carlo evidence should be incorporated verbally after actual figures arrive. Near-term work: label-independent pair selection, ratio penalty, heteroskedastic comparison models, held-out change validation, calibration, inference for target functional. Inference cannot be dismissed because neural weights are nonidentified; downstream design determines dependence and resampling. Leave this slide up for discussion rather than a separate thank-you.

## 31 — A4: three comparison-error models
Median zero requires the appropriate strict crossing/no-flat-region condition for the equivalence, not simply a loose median definition. Homogeneous logistic errors refer to residual differences.

## 32 — A5--A6: sampling and estimation
The near-minimizer condition is an assumption about optimization, not a guarantee from adding more cities.

## 33 — Theorem 1: why the population bound holds
Second-order expansion is for conditional expected pairwise loss with true p; integrate over pair distribution. B must bound both differences along segment.

## 35 — Label construction and interpretation
This is a source-code audit, not a change to label data. An aggregate owners-equity ratio is not necessarily a conditional mortgaged-owner LTV, so using both factors deserves conceptual checking. Missing component fallbacks also affect interpretation. No underlying model/data edits were made.

## 36 — Training details from the saved run
Source models/run_20260722/run_20260722_logs.txt. Architectural LoRA specifics in older prose were not re-verified from a saved checkpoint, so omitted here. Effective batch details should be confirmed if discussed.

## 37 — Current pair gate uses survey outcomes
W2 uses replicate-based component covariance in current process_acs. Do not use a generic independent-component delta method for the entire index. Exact vintages in saved pair table should be confirmed before claims about the run gate.

## 38 — Comparison at a common training budget
At <=200, ordinal best epoch197; L2 best98. Full ordinal best708. Do not label these Spearman on the same test cities: selection is a combined validation metric. This qualifies only-loss-differs claims.

## 42 — NYC construction event study: all reported panels
Top = held-out; bottom = training. Retain pretrend tests and simultaneous intervals. Higher thresholds have smaller samples and different treated compositions; not a causal dose-response comparison.

## 43 — NYC change prediction: all construction groups
Open training and filled held-out points; displayed fit statistics aggregate available plotted sample. GB2 mapping uses each target release. This is not raw delta f.

## 47 — Testing sensitivity to imperfect stable pairs
Proposed MC4 design. No results are claimed; add actual figures when available.

## 48 — Inference and downstream evaluation
Conformal coverage and prediction-powered inference are not automatic solutions for dependent spatial panels or causal outcomes. Need setting-specific assumptions, labeled validation, and target functional. No generic independence of treatment and prediction error sufficient for every design.
