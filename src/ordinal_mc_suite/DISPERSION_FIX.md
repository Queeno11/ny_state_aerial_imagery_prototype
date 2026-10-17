# Zero-dispersion recovery fix

The optimizer may produce a constant ReLU score (V=0), particularly with strong
path-norm regularization. V=0 remains outside the ranking criterion's domain.
This update does not add an epsilon, clip V, change the loss, or drop simulation
replications. It changes optimizer failure handling:

- Inadmissible initializations are retried with deterministic seeds, up to ten.
- An Adam step that collapses dispersion restores the best valid checkpoint
  (or valid initialization if the final-temperature phase has not started),
  then hands that checkpoint to refinement.
- An inadmissible L-BFGS trial/endpoint restores the checkpoint before that
  refinement. A failed audit probe is flagged instead of reporting a clean audit.
- A restart without any admissible initialization is recorded as inadmissible.
  Other restarts/grid points can still be selected; they use the same objective
  and validation rule. If all grid points have no admissible restart, the job
  fails explicitly instead of dropping the replication or inventing a result.
- candidates.csv and selected_fits.csv report initialization_retries,
  adam_dispersion_recoveries, lbfgs_dispersion_recoveries, optimizer_status,
  recovered_candidates and inadmissible_candidates where applicable.
  diagnostics.json reports recovered and inadmissible candidate counts.

Compatibility: both the original uploaded serial source and the first multicore
release are recognized. With identical scientific settings and numerical
runtime, existing completed jobs are retained. Only incomplete jobs are rerun.
No other scientific/source versions are silently accepted.

Extract this archive into the existing ordinal_mc_suite folder, replacing source
files. Existing output folders are not in the archive and are not overwritten.
Then resume with: bash run_overnight_multicore.sh 8
The batch skips saved work and proceeds to later stages as usual.

Tests executed here (PyTorch is not installed): actual process scheduling,
SQLite checkpoint/resume/migration, and optimizer control-flow regression tests
with deterministic backend fixtures. The latter cover dead initialization,
Adam interior/final-step collapse, L-BFGS invalid probes and endpoints, truthful
audit flags, failed-restart accounting, and unchanged valid-fit behavior.
All source files parse as Python 3.10, matching the user's environment.

An optional real-PyTorch regression test, which deliberately injects collapsed
steps, is included: python verification/check_dispersion_recovery.py
It does not access or modify existing results. It was not executed here.

The five-fold validation upgrade is separate and is not part of this repair.
Existing selection settings stay unchanged.

