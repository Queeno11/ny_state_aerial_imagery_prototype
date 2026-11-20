# Ordinal continuation and matched likelihood

Baseline endpoint: 2; new endpoints: [3, 4].
Baseline rows: 768; extension rows: 192; ML rows: 192.
Source truth enters only fit-only evaluation alignment; true target maps enter oracle evaluation.
Likelihood fits use source W, known basis family and noise, population latent normalization, and constrained positive first coefficient; no target map is fitted.
Seed ranges are descriptive and paired by seed; optimizer convergence is reported per ML fit.
Continuation trajectories completed: 4 / 4.
Distinct ML fits completed: 12 / 12.
ML fits without certified gradient tolerance: 8.

- linear/simple/rho0 / M3 / seed 1: not certified; iterations=20; gradient=0.05544747757267617; selected start=0
- quadratic/nonlinear/rho0 / M2 / seed 1: not certified; iterations=20; gradient=0.0024579026232318586; selected start=0
- quadratic/nonlinear/rho0 / M3 / seed 1: not certified; iterations=20; gradient=0.046211075891632626; selected start=1
- quadratic/nonlinear/rho0.9 / M1 / seed 1: not certified; iterations=20; gradient=0.0004291018105846276; selected start=0
- quadratic/nonlinear/rho0.9 / M2 / seed 1: not certified; iterations=20; gradient=0.0046355709152077415; selected start=0
- quadratic/nonlinear/rho0.9 / M3 / seed 1: not certified; iterations=20; gradient=0.07924462808940577; selected start=0
- quadratic/simple/rho0 / M2 / seed 1: not certified; iterations=20; gradient=0.004606465226140276; selected start=1
- quadratic/simple/rho0 / M3 / seed 1: not certified; iterations=20; gradient=0.09082784452175094; selected start=1

See combined_metrics.csv for all/stable/small/large strata and source/target splits.
