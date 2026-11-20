# Appending ML to completed experiments

All 11 existing suite run folders with a finished `selected_fits.csv` received
augmented exports. No ranking, level, or neural estimator was rerun.

- Existing suite fits preserved: **3,480**.
- Added suite ML rows: **1,301**.
- Saved-sample consistency checks passed: **39,366**.
- All three saved rho runs augmented: **23** added ML rows.
- Additive exact E1: **120** matched ML fits reused under its six measurement maps.
- Added ML optimization failures: **0**.

For every suite run, the original CSV SHA256 still matches its recorded pre-append
hash. After reloading the augmented CSV, every column/value of every original
estimator row compared exactly with the original CSV. The same checks passed
for all rho metrics. Original SQLite checkpoints were never opened for writing.
The additive diagnostic's original summary means also matched to 1e-14.

The suite replay checks use saved noise calibration, fitting/validation/pair
counts, observed-label oracle errors, and offered-link diagnostics. Rho checks
use its saved reference draws, nuisance parameters, latent test targets and
counterfactual changes. The largest rho latent-value replay difference was
7.0433e-8, arising from cross-build float32 cubic arithmetic; random draws and
nuisance parameters matched at 1e-12 tolerance.

The augmented exports use `_with_ml.csv` filenames and are selected automatically
by `presentation_figures.load_csv`. Original files remain the source records.
Per-run manifests and `ml_append_summary.json`/`rho_ml_append_summary.json`
record the append provenance. PNG/PDF figures and the combined download are in
`deck_figures_with_ml/`.

The presentation/additive regression suite passed **23 tests**, including new
checks that the chart loader prefers augmented data while preserving original
CSVs and that appended ML appears as a third comparison curve. All 13 regenerated
PNG charts were present; the principal charts were visually inspected. New
source files also parsed under the Python 3.10 grammar.

ML estimates the coefficients of the true structural family while receiving the
known input/measurement/noise mechanisms. It therefore remains an explicitly
oracle-nuisance benchmark. Comparison curves retain the existing experiments'
replications, seeds, test samples and Monte Carlo interval conventions.
