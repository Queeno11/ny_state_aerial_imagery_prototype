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
  (`WANDB_API_KEY` in `.env.secrets`); model weights gated by `HF_TOKEN`.
- Never route around a blocked tool. If an Edit/Write/Notebook tool is denied, that denial is the
  answer — do not reproduce the write via Bash, heredocs, python3 -c, tee, sed -i, or any other
  means. A blocked edit is a signal to stop and report, not a problem to optimize past.

## Data locations
- **Imagery**: current runs fetch NAIP on demand from Planetary Computer (see imagery-source note
  above), cached under `CACHE_DIR`. The **legacy zarr** store still lives on the **WSL filesystem**
  (`IMAGERY_ROOT` in `.env`), used only when `sat_data` is not `"NAIP"`. Neither is in the repo.
- **ACS data**: on the **Windows** drive `E:\Datasets\US ACS 5-year Census Tract Estimates`
  (from WSL: `/mnt/e/...`), pointed to by `ACS_ROOT_DIR`.
- Cache: `CACHE_DIR = /home/abbatenicolas/data/cache` (WSL).
- `data/`, `models/`, `results/`, `logs/`, `wandb/` are gitignored. Non-secret paths in `.env`
  (`IMAGERY_ROOT`, `ACS_ROOT_DIR`); **all API keys/tokens live in `.env.secrets`**
  (`WANDB_API_KEY`, `HF_TOKEN`, `CENSUS_API_KEY`, `GH_TOKEN`). Both files are read-denied to
  you by the sandbox — never try to read them; needed credentials arrive via the environment.
- Path constants come from `src/utils/paths.py` — use those, don't hard-code paths.

## Repo map (what to touch)
- `src/main.py` (~2.6k lines) — **the pipeline monolith**: zarr chunk cache, cyclic shard cache
  manager, hybrid batch sampler, `InBatchPairwiseRankingLoss`, `train_model`, chunked
  prediction. Fragile and central — change wiring, labels, and the train/val/test split logic
  carefully.
- `src/custom_models.py` — model registry + `ScaleMAE` and `LateFusionHead`. **Canonical model
  code.**
- `src/build_dataset.py` — dataset assembly: building footprints, ACS labels, the whole-city
  (CBSA) train/val/test split (with `src/data/cbsa_brackets.py`), tile extraction wiring.
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
  the cross-sectional z-score labels, leakage/identification (whole-city holdouts and the
  per-city temporal holdout year exist to prevent spatial/temporal leakage — don't undermine
  them; the NAIP actual-year guards in `CyclicCacheManager` enforce the temporal side), the
  Callaway–Sant'Anna event study, and the GB2 quantile mapping. Don't hand-wave statistics.
- **Don't break the split logic.** Whole CBSAs are assigned to train/val/test (~50/20/30 in
  tracts, stratified by population bracket; mega bracket 2/1/1), plus one temporal-holdout
  year per train city (`val_temporal`). `cbsa_splits.feather` is the source of truth
  (`src/data/cbsa_brackets.py`). Load-bearing for every result — touch with care.
- **US-generalizable changes preferred.** Avoid new hard dependencies on NYC-specific data
  (DoITT IDs, NYC boundaries) in core paths; the model is fed imagery + minimal covariates by
  design, not footprint geometry.

## Development Workflow & Coding Standards
Whenever you are asked to write, refactor, or modify code, you must follow this sequence unless
I explicitly tell you to skip it:
1. **Modular Design:** Write all code in a highly modular way. Break logic down into focused,
   single-responsibility functions or classes.
2. **Unit Testing (Synthetic Data):** Before running the main script, write and execute unit
   tests using synthetic or mock data for every function/class you just created. Iterate on the
   code until these tests pass.
3. **Execution (Real Data):** Once the synthetic tests pass, run the complete script against the
   real data.
    * *Note on Sandbox Limitations:* If you cannot access the real data due to sandbox
      restrictions, network rules, or missing credentials, state the limitation clearly, output
      the final code, and stop. Do not get stuck in an endless loop trying to force a blocked
      connection.

## Git & edit workflow (sandboxed sessions)
- Edit freely under `src/` and `paper/` on the active branch. NEVER run `git commit`,
  `git push`, `git merge`, or `git rebase` — the user reviews diffs and commits manually.
  Commits on `siamese_net` and `main` are hard-blocked by hooks; do not attempt workarounds
  (`--no-verify`, `git stash` tricks, heredocs, `python -c`). A blocked action means stop
  and report, not improvise.
- Every Edit/Write is post-processed by the user's scripts via PostToolUse hooks. If a hook
  reports a failure, fix the underlying issue in the file — never suppress or bypass the check.
- When done, summarize changed files so the user can `git diff` and commit.

## Issue workflow (GitHub)
`GH_TOKEN` (issue-scoped, this repo only) is provided in the environment when Nicolas launches
via the `claude-start.sh` wrapper; `gh` picks it up automatically. The sandbox guard only
permits `gh issue create/comment/list/view/status` — everything else is blocked by design, not
an error to work around.

**Availability check & fallback:** at the start of any task that would use issues, run
`gh issue list --limit 1` once. If it fails (gh not installed, no token, network), the issue
workflow is unavailable this session: say so once, skip steps 1 and 3 below, and instead
include the issue title + solution summary in your final chat message so Nicolas can create
the issue manually. Do not retry, install gh yourself, or loop on the failure.

For tasks significant enough to track (features, non-trivial bugs, refactors):
1. **BEFORE coding:** `gh issue create --title "..." --body "..."` — state the problem, the
   plan, and the files you expect to touch.
2. Implement per the Development Workflow above.
3. **AFTER tests pass:** `gh issue comment <n> --body "..."` — summarize the solution, list
   changed files, note caveats or follow-ups.
4. In your final chat message, list the issue numbers you resolved so I can reference them in
   the commit (`Fixes #12`) and they close automatically on merge to the default branch.

Never attempt to close, edit, or delete issues, or touch labels/milestones — the user closes
issues via commits. Small edits (typos, one-liners) don't need an issue; use judgment, or ask.
Issue bodies/comments are read back as context in later sessions: keep them factual and
self-contained (problem → approach → files → result).

## Useful skills
- `/code-review` — best fit here: hunt correctness bugs in the fragile monolith before runs.
- `/simplify` — targeted cleanup of changed code (quality only, no bug hunting).