# Paper Monte Carlo (E1–E5): how to run

The experimental design and the glossary of every acronym (RNC, RNC_S, RNC_P0.9, SUR, MRC_S, FE_S, heads,
metrics, scenario names) are in **`DESIGN.md`**.

Every command runs from `src/ordinal_mc_suite` in WSL with `torch_geo_env` active:
```bash
cd "/mnt/c/Working Papers/NY State Aerial Imagery Prototype/ny_state_aerial_imagery_prototype/src/ordinal_mc_suite"
conda activate torch_geo_env
```

## 0. Once
```bash
pip install numba                     # optional; 2-3x faster pair kernels (numpy kernels otherwise)
python -m unittest minimal_e1.tests.test_paper minimal_e1.tests.test_minimal_e1 minimal_e1.tests.test_robust minimal_e1.tests.test_r3
```

## 1. Asymptotic-bias (pseudo-true) lines: once per DGP, cached
```bash
python -m minimal_e1.paper.pseudotrue --experiments E1,E2,E4 --C 512 --n 500 --workers 9 --cores 16 --cache minimal_e1/paper/pseudotrue
```
This fits one 512×500 sample per distinct DGP (9 in total). It prints a line per estimator and per finished
DGP, and skips DGPs that are already cached.

## 2. Experiments: resumable
```bash
python -m minimal_e1.paper.run --experiment E1,E2,E3 --workers 16 --cores 16 --exclude-arms MM
```
- `--experiment` takes `E1|E2|E3|E4|E5|all` or a comma list. E5 (baseline updating) needs no pseudo-true step. `--scenarios L_both,F_20` and `--sizes 8x150,64x500`
  restrict the run.
- `--reps N` caps the replications per size; rerunning without it fills in the rest.
- `--list` shows the pending tasks; `--smoke` does 2 reps at the smallest size.
- One JSON file is written per (experiment, scenario, size, rep) under `minimal_e1/paper/results/`. Existing
  files are skipped, so rerunning the same command resumes.
- `--arms A,B` / `--exclude-arms A,B` select arms. To add an arm later, run it into a separate folder with
  `--out` and merge at summary time (below).
- MM is slow, so run it separately: `--experiment E1 --arms MM --reps 20 --out minimal_e1/paper/results_mm`.

## 2b. Re-score finished replications with new heads/metrics (no refitting of the first step)
```bash
python -m minimal_e1.paper.rehead --src minimal_e1/paper/results --out minimal_e1/paper/results_v2 --experiments E1,E2,E3 --workers 16 --cores 16
```
Regenerates each replication's data from its seed, re-uses the stored θ, and writes the new metrics (CV-spline
head, R² vs observed W/ΔW, ceiling). Refits only FE/FE_S (for their native predictions) and the arms in
`--refit` (default: penalty arms of E2 N1_nuis_both/N2_hc_both). A file is not written unless the recomputed
oracle-head RMSE matches the stored one. Resumable.

## 3. Tables and figures
```bash
python -m minimal_e1.paper.summarize            # -> minimal_e1/paper/results/tables/
python -m minimal_e1.paper.figures              # -> minimal_e1/paper/results/figures/
# merge folders:  --results minimal_e1/paper/results,minimal_e1/paper/results_mm
```

**Tables:**
- `summary_<exp>.csv`: mean, sd and MCSE of each RMSE per scenario × size × arm × target × head.
- `omega_curve.csv`, `timing_by_arm.csv`, `timing_by_size.csv`.
- `status_counts.csv`: `na` = the arm does not apply, e.g. a restricted arm with no links.

**Figures:** coefficient plots with three panels (index f | W | ΔW) and a footnote defining the scenario, arms
and metrics. `--stat r2` (default, `figures/`): R² of the index and out-of-sample R² vs observed W and ΔW,
dashed line = ceiling per size. `--stat rmse` (`figures/rmse/`): RMSE vs the true conditional mean, log axis,
grey bar = pseudo-true.
- `fig_<exp>_<scenario>`: rows = arms; colour = sample size (light = small, dark = large); filled = oracle head
  (the index itself in panel 1), open = CV-spline head, diamond = FE's native prediction; whiskers = ±1.96 MCSE.
- `fig_overview_<exp>_<C>x<n>`: rows = scenarios; colour and marker = RNC_S, RNC, SUR_S, FE, MM (spline head
  for R², oracle head for RMSE).
- `fig_omega_<scenario>_<C>x<n>`: E2 bias–variance curve over ω.
- `fig_update_<exp>_<scenario>` and `fig_update_overview_<exp>_<C>x<n>` (E5, E3 D_rho08): baseline-updating predictors
  (survey only, image head, baseline + image change, constant κ, κ(R), oracle); panels = R² of observed ΔW, RMSE vs
  E[ΔW | x, W_0] for stable and for changing tracts; RNC_S first step.

## Cores, workers, threads
- `--workers W` = parallel processes; `--threads T` = numba/BLAS threads per process; `--cores N` = total budget
  (default all cores). Without `--threads`, T = N // W, and W × T never exceeds N.
- `run`: many small tasks, so use `--workers 16 --cores 16` (1 thread each). `pseudotrue`: at most 9 DGPs, so use
  `--workers 9 --cores 16`.
- Hybrid CPUs under WSL cannot pin processes to P-cores. `--cores 8`–`10` is the closest to "P-cores only".
- `--mem-gb` (default 2) is the memory budget per worker for the pair kernels; peak use is about 250 MB per worker
  at 128×500. `--max-pairs-per-cell N` turns on a fixed pair subsample (off by default; fallback only).

## Other
- Calibration of λ(ω): `python -m minimal_e1.paper.calibrate --reps 20` (writes `paper/calibration.json`).
- Runtime (Core Ultra 7 265KF, 16 workers, no MM): E1–E3 took about 2.5 h. One replication takes about
  85 s at 64×500 and 150 s at 128×500.
