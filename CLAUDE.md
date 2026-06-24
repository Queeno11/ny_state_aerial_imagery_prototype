# CLAUDE.md

Guidance for working in this repository. Read this before touching code.

## What this project is

A research codebase (Nicolas Abbate, NYU PhD) that estimates **building-level relative
wealth from high-resolution aerial imagery**. The model is a **ScaleMAE (ViT-L/16) backbone
with LoRA adapters** and a late-fusion scalar head, trained with a **purely ordinal loss**
(RankNet cross-sectional ranking + temporal stability penalty + variance regularizer) so that
predictions are invariant to city-wide macro drift and transferable across cities without
retraining. Applied to NYC aerial orthoimagery, 2010–2024 (even years), with ACS tract income
as supervision. The paper is the source of truth for the method: `paper/latex/main.tex`.

**North star:** scale this from NYC to the whole US. Don't build NYC-only assumptions in.
Don't *act* on the US scale-up unless asked — just don't foreclose it.

**Imagery source — already NAIP, not zarr.** `CyclicCacheManager` in `main.py` is dual-mode,
switched by the `sat_data` param: `"NAIP"` fetches tiles programmatically from Planetary Computer
(`src/data/naip_fetcher.py` `fetch_naip_crop`) into a rolling disk cache — each shard is consumed
until it "dies," then a fresh one is generated — while any other value reads the legacy zarr store.
Current runs use `sat_data="NAIP"`, so even NYC now trains off live NAIP fetches, not zarr. This is
the path that scales to the US: don't reintroduce or extend the zarr branch for new work.

## Environment & how to run — IMPORTANT

- **Always run Python through WSL + the `torch_geo_env` conda env** (not the Windows `.venv` or
  `tf_updated`). WSL doesn't inherit the Windows cwd, so `cd` in first:
  ```bash
  wsl bash -ic 'cd "/mnt/c/Working Papers/NY State Aerial Imagery Prototype/ny_state_aerial_imagery_prototype" && conda activate torch_geo_env && python <script>'
  ```
- Editable install (`pip install -e .`), so `import src...` works; the full pipeline is launched
  as `python src/main.py` (params in the `__main__` block, no CLI flags).
- **`src/main.py` is the only script you must never run** (full training / full-city prediction /
  large NAIP pulls — Nicolas runs those on the GPU box). You *may and should* run smaller
  subtasks to verify functions you edit: exercise a single function, run `verify_pipeline.py`, or
  set `small_sample=True`.
- GPU: `main.py` sets a VRAM safety cap (~7 GB) for 8 GB cards. Experiment tracking is **wandb**
  (`WANDB_API_KEY` in `.env`); model weights gated by `HF_TOKEN`.

## Data locations

- **Imagery**: current runs fetch NAIP on demand from Planetary Computer (see imagery-source note
  above), cached under `CACHE_DIR`. The **legacy zarr** store still lives on the **WSL filesystem**
  (`IMAGERY_ROOT` in `.env`), used only when `sat_data` is not `"NAIP"`. Neither is in the repo.
- **ACS data**: on the **Windows** drive `E:\Datasets\US ACS 5-year Census Tract Estimates`
  (from WSL: `/mnt/e/...`), pointed to by `ACS_ROOT_DIR`.
- Cache: `CACHE_DIR = /home/abbatenicolas/data/cache` (WSL).
- `data/`, `models/`, `results/`, `logs/`, `wandb/` are gitignored. Secrets/paths in `.env`
  (`IMAGERY_ROOT`, `ACS_ROOT_DIR`, `WANDB_API_KEY`, `HF_TOKEN`, `CENSUS_API_KEY`).
- Path constants come from `src/utils/paths.py` — use those, don't hard-code paths.

## Repo map (what to touch)

- `src/main.py` (~2.6k lines) — **the pipeline monolith**: zarr chunk cache, cyclic shard cache
  manager, hybrid batch sampler, `InBatchPairwiseRankingLoss`, `train_model`, chunked
  prediction. Fragile and central — change wiring, labels, and the train/val/test split logic
  carefully.
- `src/custom_models.py` — model registry + `ScaleMAE` and `LateFusionHead`. **Canonical model
  code.**
- `src/build_dataset.py` — dataset assembly: building footprints, ACS income labels, tract
  train/val/test/dead-zone split, tile extraction wiring.
- `src/data/` — `download_acs.py`, `process_acs.py` (panel construction, MOE→SE, z-scores,
  structural-change indicator), `dataset_generation.py`, and the **US scale-up** path
  `naip_fetcher.py` / `query_naipp.py` (NAIP via Planetary Computer).
- `src/evaluation.py` (~2.5k lines) — produces the paper figures/tables, organized as
  `part_a`…`part_e` (maps, stability, GB2 quantile mapping, CSA event study, Hudson Yards).
- `src/geo_utils.py` — projection/tile geometry helpers (EPSG:6539, meters↔pixels, tau).
- `paper/latex/main.tex` — the paper. Other `.tex` files in that dir are notes/older drafts.

**Legacy — ignore unless explicitly asked:** `src/_old/`, `custom_models_tf.py`, and the
EfficientNet/DINOv2 runs under `models/` and `logs/`. Only ScaleMAE/PyTorch is current.

## Working conventions

- **Keep paper and code in sync.** The method described in `paper/latex/main.tex` (loss terms,
  λ_s, τ=100m, temporal_fraction, split design, GB2 mapping, CSA thresholds) must match the
  code. If you change one, flag the other; if you spot drift, surface it rather than silently
  picking a side.
- **Econometric rigor matters.** Be careful with ACS standard errors (MOE/1.645, delta method),
  the cross-sectional z-score labels, leakage/identification (the 150m dead-zone buffer and the
  2016 temporal holdout exist to prevent spatial/temporal leakage — don't undermine them), the
  Callaway–Sant'Anna event study, and the GB2 quantile mapping. Don't hand-wave statistics.
- **Don't break the split logic.** Test tracts (~5%), validation tracts (~10%), dead-zone, and
  the 2016 temporal holdout are load-bearing for every result. Touch with care.
- **US-generalizable changes preferred.** Avoid new hard dependencies on NYC-specific data
  (DoITT IDs, NYC boundaries) in core paths; the model is fed imagery + minimal covariates by
  design, not footprint geometry.
- **Git:** never `git commit`/`push` unless asked. Make edits; Nicolas commits.

## Useful skills

- `/code-review` — best fit here: hunt correctness bugs in the fragile monolith before runs.
- `/simplify` — targeted cleanup of changed code (quality only, no bug hunting).
