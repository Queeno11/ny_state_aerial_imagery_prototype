"""Verify saved pilot accounting and comparisons without refitting any scorer."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from training import tensor_hash

HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(run):
    manifest = read_json(run / "manifest.json")
    for name, sha256 in manifest["source_sha256"].items():
        path = HERE.parent / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256, path
    assert hashlib.sha256((HERE / "config.json").read_bytes()).hexdigest() == manifest["config_sha256"]
    assert (run / "completion.json").exists() and not (run / "failure.txt").exists()
    return manifest["config"]


def equal_arms(frame, where_clause, arm1, arm2):
    filtered = frame.query(where_clause)
    x, y = (filtered[filtered.arm == arm].drop(columns="arm").reset_index(drop=True)
            for arm in (arm1, arm2))
    assert len(x) > 0
    pd.testing.assert_frame_equal(x, y, check_exact=True)


def verify_states(checkpoint):
    for records in checkpoint["results"].values():
        for state in records.values():
            if isinstance(state, dict):
                assert tensor_hash(state["model"]) == state["state_hash"]
                for name in ("calibrated", "native"):
                    if state.get(name) is not None:
                        assert np.min(state[name][:, 1:]) > 0


def verify_main(run, cfg):
    rows = []
    reps = sorted(run.glob("rep_*"))
    assert len(reps) == len(cfg["replications"]) * len(cfg["start_ids"])
    for rep in reps:
        checkpoint = torch.load(rep / "checkpoints.pt", weights_only=False, map_location="cpu")
        verify_states(checkpoint)
        trace = pd.read_csv(rep / "training.csv")
        results = checkpoint["results"]
        structure = pd.read_csv(rep / "structural.csv")
        outcomes = pd.read_csv(rep / "outcome.csv")
        ranks = pd.read_csv(rep / "ranks.csv")
        equal_arms(structure, "endpoint == 'selected'", "A_anchor", "F_anchor")
        equal_arms(ranks, "endpoint == 'selected'", "A_anchor", "F_anchor")
        equal_arms(outcomes, "endpoint == 'selected' and head == 'common'", "A_anchor", "F_anchor")
        assert set(outcomes["where"]) == {"seen"}
        assert set(outcomes["panel"]) == {"supported_mixed", "supported_small_mixed"}
        assert np.array_equal(results["A_anchor"]["selected"]["calibrated"],
                              results["F_anchor"]["selected"]["calibrated"])
        assert results["A_anchor"]["selected"]["state_hash"] == results["F_anchor"]["selected"]["state_hash"]
        np.testing.assert_allclose(outcomes.mean_change_mse,
            outcomes.mean_residual_date0_mse + outcomes.mean_residual_date1_mse - 2 * outcomes.mean_residual_cross_moment,
            rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(outcomes.realized_change_mse,
            outcomes.mean_change_mse + outcomes.mean_change_noise_floor + outcomes.realized_change_cross_term,
            rtol=1e-10, atol=1e-10)
        for left, right in [("acquisition_only", "map_only"), ("structural_only", "opposing_shifted_map")]:
            common = ["rep", "start", "arm", "endpoint", "where", "stratum", "kind", "count",
                      "prediction_mean", "rms_score_movement", "rmse", "sign", "sensitivity", "false_positive"]
            a, b = (structure[structure.panel == name][common].reset_index(drop=True) for name in (left, right))
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        selection = pd.read_csv(rep / "selections.csv")
        assert trace[trace.path == "anchor"]["update"].max() == cfg["baseline_updates"]
        for branch in ("O_continue", "L", "D", "LD", "LDC"):
            branch_trace = trace[trace.path == branch]
            assert branch_trace["update"].max() == cfg["joint_updates"]
            assert np.array_equal(branch_trace["update"].to_numpy(), np.arange(0, cfg["joint_updates"] + 1, cfg["check_every"]))
            select = selection[(selection.arm == branch) & (selection.endpoint == "selected")].iloc[0]
            assert select["update"] == branch_trace.loc[branch_trace.selector.idxmin(), "update"]
            assert np.isclose(select.selector, branch_trace.selector.min(), atol=1e-12, rtol=1e-12)
        precision = pd.read_csv(rep / "precision_audit.csv")
        heads = pd.read_csv(rep / "head_convergence.csv")
        assert np.max(precision.score_max_abs_error) < 2e-4
        assert np.max(precision.outcome_max_abs_error) < 2e-3
        assert np.max(trace.head_projected_grad.dropna()) <= 5e-5
        assert np.max(heads.projected_gradient) <= 5e-5
        elapsed = read_json(rep / "completion.json")
        rows.append(dict(rep=rep.name, **elapsed, max_cpu_gpu_score_error=float(precision.score_max_abs_error.max()),
                         max_cpu_gpu_outcome_error=float(precision.outcome_max_abs_error.max()),
                         max_head_projected_gradient=float(trace.head_projected_grad.max()),
                         max_fresh_head_prediction_change=float(heads.fresh_cold_warm_prediction_max.max())))
    return rows


def verify_known(run, cfg):
    paths = sorted(run.glob("noise_*"))
    assert len(paths) == 4
    rows = []
    for path in paths:
        checkpoint = torch.load(path / "checkpoints.pt", weights_only=False, map_location="cpu")
        verify_states(checkpoint)
        assert len({arm["initial_hash"] for arm in checkpoint["results"].values()}) == 1
        trace = pd.read_csv(path / "training.csv")
        assert trace.groupby("path")["update"].max().eq(cfg["baseline_updates"]).all()
        rows.append(dict(case=path.name, **read_json(path / "completion.json")))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", type=Path, required=True)
    p.add_argument("--known", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cfg = verify_manifest(args.main)
    assert cfg == verify_manifest(args.known)
    main_audit = verify_main(args.main, cfg)
    known_audit = verify_known(args.known, cfg)
    with args.out.open("x", encoding="utf-8") as file:
        json.dump(dict(passed=True, main=main_audit, known=known_audit), file, indent=2)


if __name__ == "__main__":
    main()
