"""Cardinal L2 baseline (B1, "cardinal-levels") — standalone trainer/predictor.

The comparison model for the paper's main figure (Panel B) and the L2
comparison figure in evaluation.py. Everything is held fixed against the
ordinal run except three things:

* **loss**   — pointwise MSE on every image (main.PointwiseMSELoss): no pairs,
  no stability term, no variance regularizer, no SE weighting, no curriculum;
* **label**  — log W in real (CPI-U) dollars, standardized with ONE pooled
  mean/SD fitted on train CBSAs only (src/data/level_labels.py), so
  between-city level differences stay in the target;
* **batches** — the cross-sectional core mixes cities and years
  (``mix_pools``), with the same per-image draw probability as the ordinal
  sampler and the same temporal twins.

Gradient hard-tract mining is off (tract hardness is defined by ranking-loss
gradients). The baseline uses its own shard-cache subdirectory, so it never
clears the ordinal run's NAIP shards.

Protocol (fixed before running — no hyperparameter search beyond it)
--------------------------------------------------------------------
1. ``screen``  — one run per LR in ``LR_GRID`` for ``SCREEN_EPOCHS`` epochs.
2. ``select``  — pick the LR with the best mean validation selection score
   (``within_50_50``, the ordinal model's own metric) over the last
   ``SELECTION_WINDOW`` screen epochs; within ``TIE_TOL`` of the best, the
   LOWER lr wins (short screens favour large learning rates). Never test data.
3. ``final``   — resume the winning screen run from its epoch-50 checkpoint
   and continue to ``FINAL_EPOCHS``.
4. ``predict`` — test-split predictions with the ordinal run's exact sampling
   (same tracts / buildings), written to results/<baseline>/ in the layout
   evaluation.py reads.
5. ``link``    — write results/<ordinal>/l2_baseline.json so evaluation picks
   the baseline up automatically (main figure Panel B, 'l2', 'l2_training').
6. ``export-history`` — pull each run's per-epoch validation curve from wandb
   into results/<run>/wandb_history.csv (training-curve / matched-budget part).

Usage (WSL + torch_geo_env, from the repo root; GPU box)::

    python src/run_L2_model.py                    # screen -> final -> predict -> link
    python src/run_L2_model.py --stage screen     # one stage at a time
    python src/run_L2_model.py --stage export-history
    python -m src.evaluation --savename run_20260722 --mode us \\
        --parts main_figure l2 l2_training

The selection / link stages need the screen histories, which come from wandb:
run ``export-history`` after ``screen`` (``--stage all`` does it for you).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import MODELS_DIR, RESULTS_DIR

# ─── protocol constants ───────────────────────────────────────────────────────
ORDINAL_RUN_ID = "run_20260722"
LR_GRID = (3e-5, 1e-4, 3e-4)      # log-spaced around the ordinal run's 1e-4
SCREEN_EPOCHS = 50
FINAL_EPOCHS = 200
SELECTION_WINDOW = 10             # mean over the last N screen epochs (val noise)
TIE_TOL = 0.01                    # within this of the best -> prefer the lower lr
L2_CACHE_SUBDIR = "l2_baseline"
WANDB_PROJECT = "urban-income-prediction"   # must match main.init_wandb_run
POINTER_FILENAME = "l2_baseline.json"       # evaluation.L2_POINTER_FILENAME
HISTORY_FILENAME = "wandb_history.csv"      # evaluation.WANDB_HISTORY_FILENAME

# What makes the run the L2 baseline; everything else is the ordinal run's.
L2_OVERRIDES = {
    "loss_type": "mse",
    "label_target": "log_level",
    "mix_pools": True,
    "grad_mining_alpha": 0.0,
    "grad_mining_alpha_final": 0.0,
    "cache_subdir": L2_CACHE_SUBDIR,
    "resume_lr_override": None,     # resuming the screen winner keeps its own lr
    # Test-split comparison only: no NYC full-universe pass (that serves the
    # NYC CSA event study, not this comparison); the sampled test tracts are
    # the ordinal run's by construction (hash-deterministic).
    "predict_full_universe_geoid_prefixes": (),
    "level_label_stats": None,      # refitted by generate_datasets each run
}

# Keys run() adds after the defaults are filled; not valid as input params.
_DERIVED_KEYS = ("tau_meters_requested",)


# ─── parameters ──────────────────────────────────────────────────────────────

def screen_savename(lr: float, ordinal_run_id: str = ORDINAL_RUN_ID) -> str:
    """'run_20260722_l2_lr1e-04' — one savename (checkpoints, cache marker,
    wandb name, results dir) per screen learning rate."""
    return f"{ordinal_run_id}_l2_lr{lr:.0e}"


def load_logged_params(run_id: str, models_dir: Path = MODELS_DIR) -> dict:
    """The params a run last logged (main.generate_parameters_log)."""
    path = Path(models_dir) / run_id / f"{run_id}_logs.txt"
    if not path.exists():
        raise FileNotFoundError(f"No parameter log for '{run_id}' at {path}.")
    return json.loads(path.read_text())


def input_params_from_log(logged: dict) -> dict:
    """Turn a logged (post-snap) params dict back into valid run() input.

    run() snaps tau_meters to the exact-tile value and records the original
    request as ``tau_meters_requested``; feeding the snapped value back would
    snap a second time, so the request is restored and the derived keys dropped.
    """
    params = {k: v for k, v in logged.items() if k not in _DERIVED_KEYS}
    if "tau_meters_requested" in logged:
        params["tau_meters"] = logged["tau_meters_requested"]
    return params


def l2_params(ordinal_params: dict, lr: float, n_epochs: int, run_id: str) -> dict:
    """The ordinal run's params + the L2 overrides, at one lr / budget."""
    params = dict(ordinal_params)
    params.update(L2_OVERRIDES)
    params.update({"learning_rate": float(lr), "n_epochs": int(n_epochs), "run_id": run_id})
    return params


# ─── learning-rate selection ─────────────────────────────────────────────────

def screen_score(history: pd.DataFrame, max_epoch: int = SCREEN_EPOCHS,
                 window: int = SELECTION_WINDOW) -> float:
    """Mean selection score over the last ``window`` epochs <= ``max_epoch``.

    ``history``: [epoch, score]. NaN when the run has fewer than ``window``
    scored epochs in range — an incomplete screen must not win by luck.
    """
    h = history[(history["epoch"] <= max_epoch)].dropna(subset=["score"])
    h = h.sort_values("epoch").tail(window)
    if len(h) < window:
        return float("nan")
    return float(h["score"].mean())


def select_screen_lr(histories: dict, max_epoch: int = SCREEN_EPOCHS,
                     window: int = SELECTION_WINDOW, tie_tol: float = TIE_TOL) -> dict:
    """Pick the screen winner: best windowed score; ties (within ``tie_tol``
    of the best) go to the LOWEST lr. Returns {lr, scores}; raises when no
    run completed the screen."""
    scores = {float(lr): screen_score(h, max_epoch, window) for lr, h in histories.items()}
    valid = {lr: s for lr, s in scores.items() if np.isfinite(s)}
    if not valid:
        raise ValueError(f"No screen run has {window} scored epochs <= {max_epoch}: {scores}")
    best = max(valid.values())
    lr = min(lr for lr, s in valid.items() if s >= best - tie_tol)
    return {"lr": lr, "scores": scores, "window": window, "max_epoch": max_epoch,
            "tie_tol": tie_tol}


# ─── wandb history export ────────────────────────────────────────────────────

def merge_histories(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """[epoch, selection/score] across a savename's wandb runs (resumes create
    new run ids), in run-creation order; a re-logged epoch keeps its LAST value."""
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame(columns=["epoch", "selection/score"])
    h = pd.concat(frames, ignore_index=True)[["epoch", "selection/score"]].dropna()
    h["epoch"] = h["epoch"].astype(int)
    return h.drop_duplicates("epoch", keep="last").sort_values("epoch").reset_index(drop=True)


def fetch_wandb_history(savename: str, project: str = WANDB_PROJECT) -> pd.DataFrame:
    """Per-epoch [epoch, selection/score] of every wandb run named ``savename``."""
    import wandb
    api = wandb.Api()
    runs = api.runs(f"{api.default_entity}/{project}", filters={"display_name": savename})
    runs = sorted(runs, key=lambda r: r.created_at)
    frames = [pd.DataFrame(list(r.scan_history(keys=["epoch", "selection/score"])))
              for r in runs]
    return merge_histories(frames)


def export_history(savename: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Write results/<savename>/wandb_history.csv; returns its path."""
    h = fetch_wandb_history(savename)
    if h.empty:
        raise RuntimeError(f"wandb has no epoch/selection history for '{savename}'.")
    out = Path(results_dir) / savename / HISTORY_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    h.to_csv(out, index=False)
    print(f"  {savename}: {len(h)} epochs -> {out}")
    return out


def read_history(savename: str, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """[epoch, score] from an exported wandb_history.csv."""
    path = Path(results_dir) / savename / HISTORY_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `--stage export-history` first.")
    return pd.read_csv(path).rename(columns={"selection/score": "score"})


# ─── prediction outputs ──────────────────────────────────────────────────────

def predicted_years(run_dir: Path) -> list[int]:
    """Years with a ``<year>_predictions.csv`` in ``run_dir`` (sorted). The
    baseline predicts only these — a year the ordinal run never predicted has
    nothing to pair with, so fetching it would be wasted NAIP traffic."""
    years = []
    for f in Path(run_dir).glob("*_predictions.csv"):
        stem = f.name.split("_predictions.csv")[0]
        if stem.isdigit():
            years.append(int(stem))
    return sorted(years)


def dissolve_to_tracts(df_result: pd.DataFrame) -> pd.DataFrame:
    """Building predictions -> [GEOID, Rel_Score, predicted_value, predicted_value_std],
    the predictions_by_tract_<year>.parquet layout evaluation reads."""
    tracts = df_result.groupby("GEOID").agg({
        "Rel_Score": "mean",
        "predicted_value": ["mean", "std"],
    }).reset_index()
    tracts.columns = ["GEOID", "Rel_Score", "predicted_value", "predicted_value_std"]
    return tracts


def write_pointer(ordinal_run_id: str, final_savename: str, selection: dict,
                  results_dir: Path = RESULTS_DIR) -> Path:
    """results/<ordinal>/l2_baseline.json — evaluation's link to the baseline."""
    payload = {
        "savename": final_savename,
        "screen": {f"{lr:.0e}": screen_savename(lr, ordinal_run_id) for lr in LR_GRID},
        "chosen_lr": selection.get("lr"),
        "selection": {f"{k:.0e}" if isinstance(k, float) else k: v
                      for k, v in selection.get("scores", {}).items()},
        "protocol": {"screen_epochs": SCREEN_EPOCHS, "final_epochs": FINAL_EPOCHS,
                     "window": SELECTION_WINDOW, "tie_tol": TIE_TOL},
        "written_at": datetime.now().isoformat(),
    }
    out = Path(results_dir) / ordinal_run_id / POINTER_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"🔗 {out} -> {final_savename}")
    return out


# ─── stages (GPU / network: run on the training box) ─────────────────────────

def run_training(params: dict) -> None:
    """Train (or resume) one L2 run through main.run — same pipeline as the
    ordinal model; predictions and evaluation are separate stages here."""
    from src import main
    main.run(params, train=True, generate_predictions=False, evaluate=False)


def train_in_subprocess(lr: float, epochs: int, ordinal_run_id: str) -> None:
    """One training job per process. main.run leaves shard-generation threads
    and CUDA allocations behind; a second in-process run would clear the
    shared L2 shard cache while such a thread may still be writing to it."""
    import subprocess
    import sys
    cmd = [sys.executable, str(Path(__file__).resolve()), "--stage", "train-one",
           "--lr", repr(float(lr)), "--epochs", str(int(epochs)),
           "--ordinal-run-id", ordinal_run_id]
    subprocess.run(cmd, check=True)


def stage_screen(ordinal_run_id: str = ORDINAL_RUN_ID) -> None:
    for lr in LR_GRID:
        print(f"\n{'=' * 80}\n🔬 LR screen: lr={lr:.0e}, {SCREEN_EPOCHS} epochs\n{'=' * 80}")
        train_in_subprocess(lr, SCREEN_EPOCHS, ordinal_run_id)


def stage_select() -> dict:
    histories = {lr: read_history(screen_savename(lr)) for lr in LR_GRID}
    sel = select_screen_lr(histories)
    for lr, s in sorted(sel["scores"].items()):
        mark = "  <- chosen" if lr == sel["lr"] else ""
        print(f"  lr={lr:.0e}: mean selection score (epochs "
              f"{SCREEN_EPOCHS - SELECTION_WINDOW + 1}-{SCREEN_EPOCHS}) = {s:.4f}{mark}")
    return sel


def stage_final(lr: float, ordinal_run_id: str = ORDINAL_RUN_ID) -> str:
    """Resume the winning screen run to FINAL_EPOCHS (same savename, so
    main.run picks up its epoch-50 _last.pth checkpoint)."""
    name = screen_savename(lr, ordinal_run_id)
    print(f"\n{'=' * 80}\n🏁 Final: resuming {name} to {FINAL_EPOCHS} epochs\n{'=' * 80}")
    train_in_subprocess(lr, FINAL_EPOCHS, ordinal_run_id)
    return name


def stage_predict(ordinal_params: dict, savename: str, lr: float,
                  ordinal_run_id: str = ORDINAL_RUN_ID) -> None:
    """Test-split predictions for the ordinal run's predicted years, in its layout."""
    import gc
    import torch
    from src import main, prediction
    from src import build_dataset
    from src.data import cbsa_brackets
    from src.utils.paths import CACHE_DIR

    params = main.resolve_runtime_params(l2_params(ordinal_params, lr, FINAL_EPOCHS, savename))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, best_model_path = main.load_best_model(params, savename, device)
    eval_transform = main.make_eval_transform(params["nbands"])

    # Labels here are the ordinal run's z-scores (no label_target swap): the
    # comparison scores both models against the same Rel_Score columns.
    table = build_dataset.load_income_dataset(
        params["years"], tau_meters=params["tau_meters"], indicator=params["indicator"],
        footprints_source="ms_us", states=params["states"],
    )
    city_split = cbsa_brackets.load_city_split()
    split_map = dict(zip(city_split["cbsa_code"].astype(str), city_split["split"]))
    holdout_years = cbsa_brackets.holdout_year_map(city_split)

    chunk_root = CACHE_DIR / "pred_chunks" / savename / params["predict_split"]
    prediction.init_chunk_root(chunk_root, prediction.prediction_fingerprint(params, best_model_path))
    out_dir = RESULTS_DIR / savename
    out_dir.mkdir(parents=True, exist_ok=True)

    years = [y for y in predicted_years(RESULTS_DIR / ordinal_run_id) if y in params["years"]]
    if not years:
        raise RuntimeError(f"No <year>_predictions.csv under results/{ordinal_run_id} — "
                           f"nothing to pair the baseline with.")
    print(f"Predicting the ordinal run's years: {years}")
    
    for year in years:
        if year <= 2020:
            print(f"  skipping {year}")
            continue
        print(f"\n🚀 L2 predictions, year {year}")
        df_year = table.materialize_year(year)
        df_year = prediction.assign_prediction_types(df_year, year, split_map, holdout_years)
        df_year = prediction.select_prediction_rows(df_year, params)
        if df_year.empty:
            print(f"  no rows for {year}, skipping")
            continue
        output_csv = out_dir / f"{year}_predictions.csv"
        produced = prediction.predict_year_chunked(
            model=model, df_year=df_year, year=year, params=params, device=device,
            eval_transform=eval_transform, chunk_root=chunk_root, output_csv=output_csv,
            max_workers=int(params.get("predict_fetch_workers", 16)), verbose=True,
        )
        del df_year
        if produced:
            df_result = pd.read_csv(output_csv)
            if len(df_result):
                dissolve_to_tracts(df_result).to_parquet(
                    out_dir / f"predictions_by_tract_{year}.parquet")
            del df_result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def stage_export_history(ordinal_run_id: str = ORDINAL_RUN_ID) -> None:
    for name in [ordinal_run_id] + [screen_savename(lr, ordinal_run_id) for lr in LR_GRID]:
        try:
            export_history(name)
        except Exception as exc:   # one missing run must not block the others
            print(f"  ⚠️ {name}: {type(exc).__name__}: {exc}")


STAGES = ("all", "screen", "export-history", "select", "final", "predict", "link",
          "train-one")


def main_cli(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", choices=STAGES, default="all",
                        help="'train-one' is the per-process worker used by screen/final.")
    parser.add_argument("--ordinal-run-id", default=ORDINAL_RUN_ID)
    parser.add_argument("--lr", type=float, default=None,
                        help="Skip selection and use this lr (final/predict/link); "
                             "required for train-one.")
    parser.add_argument("--epochs", type=int, default=None, help="train-one only.")
    args = parser.parse_args(argv)
    stage = args.stage

    if stage == "train-one":
        if args.lr is None or args.epochs is None:
            parser.error("--stage train-one needs --lr and --epochs")
        ordinal = input_params_from_log(load_logged_params(args.ordinal_run_id))
        run_training(l2_params(ordinal, args.lr, args.epochs,
                               screen_savename(args.lr, args.ordinal_run_id)))
        return

    if stage in ("all", "screen"):
        stage_screen(args.ordinal_run_id)
    if stage in ("all", "export-history"):
        stage_export_history(args.ordinal_run_id)
    if stage == "export-history" or stage == "screen":
        return

    selection = {"lr": args.lr, "scores": {}} if args.lr is not None else stage_select()
    lr = selection["lr"]
    if stage == "select":
        return
    final_name = screen_savename(lr, args.ordinal_run_id)
    if stage in ("all", "final"):
        stage_final(lr, args.ordinal_run_id)
        if stage == "all":
            export_history(final_name)
    ordinal = input_params_from_log(load_logged_params(args.ordinal_run_id))
    if stage in ("all", "predict"):
        stage_predict(ordinal, final_name, lr, args.ordinal_run_id)
    if stage in ("all", "link"):
        write_pointer(args.ordinal_run_id, final_name, selection)


if __name__ == "__main__":
    main_cli()
