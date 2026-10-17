Overnight presentation battery

Extract this archive into your existing ordinal_mc_suite directory. The new
run_overnight.sh must sit beside run.py. The bundle does not replace source code.
Activate torch_geo_env, then run: bash run_overnight.sh
All three runs execute sequentially. They use CPU/float64 and 30 replications.
No changes are required to the simulator.
All three configs use a new simulation seed (20261015), retaining function_seed
983. The 30 confirmation replications therefore do not reuse the pilot data.

1. overnight_measurement (experiment 1): linear truth and linear scorer;
   raw monetary L2, log with city-year fixed effects, common-logit ranking;
   multiplier versus nonlinear shape transformations; composition 0 and 0.7;
   8/16/32/64 cities. No extra complexity penalty.
   The log/FE transformation is correctly specified in the multiplier regime,
   and misspecified in the shape regime. Comparisons use latent within-cell
   misranking in unseen cities. Composition=0 is a control; 0.7 introduces
   differences in latent feature distributions across city contexts.

2. overnight_nn (experiment 3): linear and quadratic truths; ReLU widths [8,4];
   no links versus exact temporal links; fixed lambda_t=0.3; validation-selected
   squared path-norm penalty from [0,.001,.01,.1,.3,1,3,10];
   8/16/32/64 cities. Label as a regularized sensitivity estimator.

3. overnight_links (experiment 4): linear truth and linear scorer at 32 cities;
   exact links, drift 0.15, random false links at probability 0.2, one-sided
   false links at probability 0.2. No links versus fixed temporal weight 0.3
   versus validation-tuned temporal weight from [0,.03,.3,1].
   No extra complexity penalty. Actual realized link shares and gaps are
   recorded in selected_fits.csv. This is an optional fourth presentation slide.

Outputs: each output directory contains selected_fits.csv, candidates.csv,
summary.csv, paired_summary.csv, diagnostics.json, resolved_config.json,
run_metadata.json, checkpoint.sqlite, and figures/. Upload selected_fits.csv,
paired_summary.csv, diagnostics.json, and resolved_config.json for review.
Keep candidates.csv available if tuning needs inspection.

Resume: do not rerun the batch script after any completed task. Run the exact
interrupted python command from the script with --resume appended. Existing
output folders are protected against silent overwrite. To skip completed
steps, run only the remaining commands.

The existing completed 50-replication linear temporal benchmark is reused;
it is not rerun by this bundle. The new NN case is a prespecified working
architecture, not a claim of architecture selection validated on independent data.

Compute plans: measurement 1,440 selected / 4,320 candidate fits;
NN 480 selected / 11,520 candidate fits;
links 360 selected / 2,160 candidate fits;
total 2,280 selected / 18,000 candidate fits.
The latest small-NN C=8 diagnostic took about 10 minutes of accumulated
candidate fitting for 480 candidates. At that per-fit cost the NN stage would
be about 4 hours, but larger city counts and optimizer behavior can change this
substantially. The linear stages add runtime. This is not a completion guarantee.

Verification: JSON configs pass the simulator's own merge/validation functions;
its job generator produces the stated counts. Bash syntax is checked. Training
was not executed in this review environment because PyTorch is unavailable.
