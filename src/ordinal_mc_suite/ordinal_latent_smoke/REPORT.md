# Core ordinal latent diagnostic

Implementation run summary; fixed checkpoints and no outcome-based selection.

Completed trajectories: 20.
Metric rows: 960. Objective rows: 80.
Seed count: 1; fitting cities: 4; paired tracts/city: 16.

The basis arm observes true H; supervised NN observes fitting f; true maps and Gaussian noise marginalization are oracle evaluation only.
Objective rows use a deterministic sampled monitoring batch per city and selected links. They are descriptive and do not select checkpoints.
Static plots are descriptive means across seeds. No local-optimum or likelihood-gap claim follows.
See ORDINAL_LATENT_DIAGNOSTIC.md for methods and limits, and the CSV for paired seed-level results.
