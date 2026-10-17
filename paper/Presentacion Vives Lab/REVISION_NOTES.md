# Revision notes

The revision follows Shapiro's applied-micro talk guidance: motivate economically, state the question and preview tangible findings early, define units and estimand, make assumptions explicit, use readable figures, and move documentation to backup. No automatic overlays or section-divider pages. White background, retained Marginal font family and purple accent, fewer decorative boxes, larger body text, full-width evidence.

## Corrections and qualifications

- Wealth formula now matches annuity_factor and equity_pc: ordinary annuity, mortgage share, and distinct discount-rate notation. Code agreement does not validate the economic construction; the aggregate-LTV interpretation is noted in backup.
- Saved run has tau=112m (224m footprint), not 100m/200m; requested 100m was snapped. lambda_var=1.5, not 1.
- Current run is not an implementation of the exact ratio-penalty theorem. Soft variance regularization does not make objectives equivalent. Logistic loss does not verify logistic errors.
- Survey-based gate is not independent of labels, and a non-rejection is not evidence of physical stability.
- Main results distinguish tract-aggregate paired-cell Spearman from building-level and pooled metrics. The 0.782/0.711 comparison uses 158 paired city-year cells.
- Ordinal/L2 comparison is not an equal-compute ablation: full budgets 750 vs 200 epochs. Matched-budget validation-selection results are retained in backup and labeled accordingly.
- Autocorrelation is not proof of temporal validity; changed/stable movement ratio=1.021 and direction agreement=50.1%.
- Theorem guarantees concern within-cell differences. Shared networks do not alone establish cross-city alignment. Removed unsupported learned-offset inequality and unconditional claim that offsets never matter for event studies.
- Lipschitz smoothness does not imply a directional construction response. Removed unsupported claim that NYC new construction is mostly high-end replacement.
- NYC event-study pretrend p=.004/.008 at higher thresholds is visible. Removed causal dose-response and unverified anticipation-robustness assertions.
- The main change slide uses the explicitly held-out columns of F_construction_scatter.csv; its four groups have n=243,371,196,86. The original mixed train/held-out scatters remain in backup. Both are calibrated-dollar diagnostics, not direct tests of raw Delta f.
- Temporal decomposition now distinguishes Delta R from Delta W and allows changes in psi. Removed the claim that all label changes at stable sites are epsilon.
- Removed the unqualified noise-ceiling formula and assertion that nonidentified network parameters rule out sandwich/functional inference.
- No research data, model code, or fitted results were changed. Plots derived from CSVs preserve source numbers; PDF panel crops retain all within-panel points, tests and confidence bands.

## Remaining inputs

Six puzzle images and five Monte Carlo figure PDFs are intentionally pending until tomorrow, as confirmed by the presenter. Their automatic insertion paths are in REHEARSAL.md. No numeric MC results or wealth labels were fabricated.

Confirm the gate vintages from the saved pair table if discussing the detailed implementation. The available code/prose uses the first/last W2 releases; the saved hyperparameter log itself does not list those endpoints.

The split map is inherited from the supplied deck. The run's NYC evaluation should be described as held-out tracts and sensor transfer, not as a wholly held-out city.

Sources for external claims are in slide notes. Internal sources are the saved run log, evaluation CSV/JSON files, and process_acs.py; detailed file names are recorded in notes.
