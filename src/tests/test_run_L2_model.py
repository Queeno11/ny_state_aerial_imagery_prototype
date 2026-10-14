"""Unit tests for src/run_L2_model.py — parameter construction, the LR-screen
selection rule, wandb-history merging, prediction-output helpers and the
evaluation pointer. No training, no network.
"""
import json

import numpy as np
import pandas as pd
import pytest

import src.main as main
import src.run_L2_model as r
import src.evaluation as ev

# Shape of a real logged params file (models/<run>/<run>_logs.txt): post-snap.
LOGGED = {
    "model_name": "scalemae", "kind": "reg", "weights": None, "image_size": 224, "nbands": 3,
    "batch_size": 8, "small_sample": False, "n_epochs": 750, "learning_rate": 1e-4,
    "sat_data": "NAIP", "years": list(range(2010, 2025)), "run_id": "run_20260722",
    "resume_lr_override": 1e-4, "selection_metric": "within_50_50", "indicator": "W2_r5",
    "footprints_source": "ms_us", "predict_full_universe_geoid_prefixes": ["36005"],
    "tau_meters": 112.0, "subsample_step": 2, "tau_meters_requested": 100,
    "m_min": 0.05, "lambda_s": 0.3, "lambda_var": 1.5, "temporal_fraction": 0.4,
    "tract_sampling": True, "grad_mining_alpha": 0.3, "grad_mining_alpha_final": 0.6,
    "val_years_per_city": 1,
}


def test_input_params_from_log_restores_the_tau_request():
    p = r.input_params_from_log(LOGGED)
    assert p["tau_meters"] == 100 and "tau_meters_requested" not in p
    assert LOGGED["tau_meters"] == 112.0                  # input not mutated


def test_l2_params_change_only_what_makes_the_baseline():
    ordinal = r.input_params_from_log(LOGGED)
    p = r.l2_params(ordinal, 3e-4, 50, "run_20260722_l2_lr3e-04")
    changed = {k for k in set(p) | set(ordinal) if p.get(k) != ordinal.get(k)}
    assert changed <= set(r.L2_OVERRIDES) | {"learning_rate", "n_epochs", "run_id"}
    assert p["loss_type"] == "mse" and p["label_target"] == "log_level" and p["mix_pools"]
    assert p["temporal_fraction"] == 0.4 and p["batch_size"] == 8   # held fixed
    assert p["cache_subdir"] == r.L2_CACHE_SUBDIR


def test_l2_params_pass_main_validation_and_snap_tau_once():
    p = main.resolve_runtime_params(r.l2_params(r.input_params_from_log(LOGGED), 1e-4, 200, "x"))
    assert p["tau_meters"] == pytest.approx(LOGGED["tau_meters"])   # same snap as the ordinal run
    assert p["subsample_step"] == LOGGED["subsample_step"]


def test_screen_savename_is_stable_and_distinct():
    names = {r.screen_savename(lr) for lr in r.LR_GRID}
    assert len(names) == len(r.LR_GRID)
    assert r.screen_savename(1e-4) == "run_20260722_l2_lr1e-04"


# ─── selection rule ──────────────────────────────────────────────────────────

def _hist(final, n=50, start=0.2):
    return pd.DataFrame({"epoch": range(1, n + 1), "score": np.linspace(start, final, n)})


def test_screen_score_uses_last_window_and_needs_a_full_window():
    h = _hist(0.5)
    assert r.screen_score(h) == pytest.approx(h["score"].tail(10).mean())
    assert np.isnan(r.screen_score(_hist(0.5, n=5)))
    # epochs past the screen budget are ignored
    longer = pd.concat([h, pd.DataFrame({"epoch": [51, 52], "score": [9.0, 9.0]})])
    assert r.screen_score(longer) == pytest.approx(r.screen_score(h))


def test_select_screen_lr_best_wins_and_ties_go_to_lower_lr():
    sel = r.select_screen_lr({3e-5: _hist(0.40), 1e-4: _hist(0.60), 3e-4: _hist(0.50)})
    assert sel["lr"] == 1e-4
    sel = r.select_screen_lr({3e-5: _hist(0.40), 1e-4: _hist(0.595), 3e-4: _hist(0.60)})
    assert sel["lr"] == 1e-4                               # within TIE_TOL -> lower lr
    sel = r.select_screen_lr({1e-4: _hist(0.5, n=5), 3e-4: _hist(0.3)})
    assert sel["lr"] == 3e-4                               # incomplete screen can't win
    with pytest.raises(ValueError):
        r.select_screen_lr({1e-4: _hist(0.5, n=5)})


# ─── stage wiring (no training) ──────────────────────────────────────────────

def test_screen_runs_one_subprocess_per_lr(monkeypatch):
    calls = []
    monkeypatch.setattr(r, "train_in_subprocess", lambda lr, ep, rid: calls.append((lr, ep, rid)))
    r.stage_screen("run_X")
    assert calls == [(lr, r.SCREEN_EPOCHS, "run_X") for lr in r.LR_GRID]
    calls.clear()
    assert r.stage_final(1e-4, "run_X") == "run_X_l2_lr1e-04"
    assert calls == [(1e-4, r.FINAL_EPOCHS, "run_X")]


def test_train_in_subprocess_builds_the_worker_command(monkeypatch):
    seen = {}
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: seen.update(cmd=cmd, check=check))
    r.train_in_subprocess(3e-4, 50, "run_X")
    cmd = seen["cmd"]
    assert seen["check"] is True and cmd[1].endswith("run_L2_model.py")
    assert cmd[cmd.index("--stage") + 1] == "train-one"
    assert float(cmd[cmd.index("--lr") + 1]) == 3e-4
    assert cmd[cmd.index("--epochs") + 1] == "50"


def test_train_one_requires_lr_and_epochs():
    with pytest.raises(SystemExit):
        r.main_cli(["--stage", "train-one", "--lr", "1e-4"])


# ─── histories / outputs / pointer ───────────────────────────────────────────

def test_merge_histories_keeps_last_value_per_epoch():
    a = pd.DataFrame({"epoch": [1, 2, 3], "selection/score": [0.1, 0.2, 0.3]})
    b = pd.DataFrame({"epoch": [3, 4], "selection/score": [0.35, 0.4]})   # resumed run
    h = r.merge_histories([a, None, b])
    assert h["epoch"].tolist() == [1, 2, 3, 4]
    assert h["selection/score"].tolist() == [0.1, 0.2, 0.35, 0.4]
    assert r.merge_histories([]).empty


def test_dissolve_to_tracts_layout():
    df = pd.DataFrame({"GEOID": ["a", "a", "b"], "Rel_Score": [1.0, 1.0, -1.0],
                       "predicted_value": [0.0, 2.0, 5.0]})
    t = r.dissolve_to_tracts(df).set_index("GEOID")
    assert list(t.reset_index().columns) == ["GEOID", "Rel_Score", "predicted_value",
                                            "predicted_value_std"]
    assert t.loc["a", "predicted_value"] == 1.0
    assert t.loc["a", "predicted_value_std"] == pytest.approx(np.sqrt(2))


def test_predicted_years_reads_the_ordinal_csvs(tmp_path):
    for name in ("2016_predictions.csv", "2018_predictions.csv", "nyc_predictions.csv"):
        (tmp_path / name).write_text("x\n")
    assert r.predicted_years(tmp_path) == [2016, 2018]


def test_pointer_roundtrips_into_evaluation(tmp_path):
    base = tmp_path / "run_20260722_l2_lr1e-04"
    base.mkdir()
    (base / "2016_predictions.csv").write_text("x\n")
    sel = {"lr": 1e-4, "scores": {3e-5: 0.4, 1e-4: 0.6, 3e-4: 0.5}}
    path = r.write_pointer("run_20260722", base.name, sel, results_dir=tmp_path)
    payload = json.loads(path.read_text())
    assert payload["chosen_lr"] == 1e-4 and payload["selection"]["1e-04"] == 0.6
    assert payload["screen"]["3e-04"] == "run_20260722_l2_lr3e-04"
    assert r.POINTER_FILENAME == ev.L2_POINTER_FILENAME
    assert r.HISTORY_FILENAME == ev.WANDB_HISTORY_FILENAME
    assert ev.resolve_l2_baseline_dir(tmp_path / "run_20260722") == base
