"""Audit the frozen selection, disjoint sample roles, and read-only replay outputs."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from prior import HERE, ORIGINAL
from audit import verify_known, verify_main, verify_manifest as verify_original
from run import dump_json, file_hash
from calibrate import aggregate, choose, read
from rerun import plan


def check_hashes(manifest):
    for name, sha in manifest["source_sha256"].items():
        assert file_hash(HERE.parent / name) == sha, name
    assert file_hash(HERE / "design.json") == manifest["design_sha256"]
    assert file_hash(ORIGINAL / "config.json") == manifest["config_sha256"]


def calibration_checks(folder):
    manifest = read(folder / "manifest.json")
    check_hashes(manifest)
    cfg, design = manifest["config"], manifest["design"]
    assert not (folder / "failure.txt").exists()
    selected = read(folder / "selected.json")
    rows = pd.read_csv(folder / "candidate_by_rep.csv").to_dict("records")
    scales = pd.read_csv(folder / "anchor_gradient_scales.csv")
    w, d = float(scales.ratio_W.median()), float(scales.ratio_D.median())
    np.testing.assert_allclose([w, d], [selected["scale_W"], selected["scale_D"]], atol=1e-14)
    n = len(design["calibration_reps"])
    table = aggregate(rows, w, d, n)
    initial = [a for a in table if a["gamma"] == 0]
    positive = [a for a in initial if min(a["lambda_W"], a["lambda_D"]) > 0]
    gamma_group = [a for a in table if a["candidate"] == selected["best_positive_pair"]["candidate"] or a["gamma"] > 0]
    for label, candidates in (("best_grid", initial), ("best_positive_pair", positive),
                              ("best_global", table), ("best_full", gamma_group),
                              ("best_positive_curvature", [a for a in gamma_group if a["gamma"] > 0])):
        assert selected[label]["candidate"] == choose(candidates, design["tie_tolerance"])["candidate"]
    assert len(rows) == n * selected["final_candidates"] == 40
    assert n * cfg["baseline_updates"] + sum(r["executed_updates"] for r in rows) == selected["total_updates"]
    assert set(a["gamma"] for a in gamma_group) == set(design["gamma_grid"])
    assert not list(folder.glob("**/structural.csv")) and not list(folder.glob("**/outcome.csv"))
    seeds = set()
    for rep in design["calibration_reps"]:
        path = folder / f"rep_{rep}"
        anchor = read(path / "anchor.json")
        keys = [v for k, v in anchor.items() if k.endswith("_seed")]
        assert len(keys) == len(set(keys)) and not set(keys) & seeds
        seeds.update(keys)
        candidate_rows = [a for a in rows if a["rep"] == rep]
        for a in candidate_rows:
            t = pd.read_csv(path / a["candidate"] / "trajectory.csv")
            assert t["update"].tolist() == list(range(0, cfg["joint_updates"] + 1, cfg["check_every"]))
            assert t.loc[t.selector.idxmin(), "update"] == a["selected_update"]
            assert np.isclose(t.selector.min(), a["validation"], atol=1e-12)
            assert t.head_projected_grad.max() <= 5e-5
        assert np.ptp([r["zero_validation"] for r in candidate_rows]) < 1e-10
    return manifest, selected, seeds


def audit_heads(path, tol):
    choices = pd.read_csv(path / "head_choices.csv")
    outcomes = pd.read_csv(path / "outcome.csv")
    for row in choices.itertuples():
        native = row.val_native_level + row.val_native_change
        common = row.val_common_level + row.val_common_change
        expected = "native" if native < common - tol else "common"
        assert row.chosen_head == expected
        subset = outcomes[(outcomes.arm == row.arm) & (outcomes.endpoint == "selected")]
        canonical = subset[subset["head"] == expected].drop(columns="head").reset_index(drop=True)
        deployed = subset[subset["head"] == "deployed"].drop(columns="head").reset_index(drop=True)
        pd.testing.assert_frame_equal(canonical, deployed, check_exact=True)


def evaluation_checks(folder, stage, cal_manifest, selected, selected_file, seeds):
    manifest = read(folder / "manifest.json")
    check_hashes(manifest)
    cfg, design = manifest["config"], manifest["design"]
    assert cfg == cal_manifest["config"] and design == cal_manifest["design"]
    assert file_hash(selected_file) == manifest["selected_sha256"]
    assert (folder / "completion.json").exists() and not (folder / "failure.txt").exists()
    if stage == "known":
        out = verify_known(folder, cfg)
        for p in folder.glob("noise_*"):
            assert all(read(p / "completion.json")["old_reference_equality"].values())
        return out
    out = verify_main(folder, cfg)
    weights, aliases = plan(design, selected, stage)
    assert manifest["arms"] == {key: list(value) for key, value in weights.items()}
    assert manifest["aliases"] == aliases
    for rep in manifest["reps"]:
        path = folder / f"rep_{rep}_start_0"
        record = read(path / "design.json")
        identifiers = set(record["seeds"].values()) | {record["scorer_seed"]}
        assert not seeds & identifiers
        seeds.update(identifiers)
        completion = read(path / "completion.json")
        if stage == "replay":
            assert completion["replay"] and all(completion["replay"].values())
        assert completion["executed_scorer_updates"] == cfg["baseline_updates"] + len(weights) * cfg["joint_updates"]
        trace = pd.read_csv(path / "training.csv")
        for arm in weights:
            assert trace.loc[trace.path == arm, "update"].max() == cfg["joint_updates"]
        audit_heads(path, design["tie_tolerance"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--confirmation", type=Path, required=True)
    p.add_argument("--known", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    verify_original(ORIGINAL / "main_pilot")
    verify_original(ORIGINAL / "known_control")
    cm, selected, seeds = calibration_checks(args.calibration)
    results = {}
    for stage in ("replay", "confirmation", "known"):
        results[stage] = evaluation_checks(getattr(args, stage), stage, cm, selected,
                                          args.calibration / "selected.json", seeds)
    dump_json(args.out, dict(passed=True, calibration_updates=selected["total_updates"],
                            evaluation=results, distinct_checked_seeds=len(seeds)))


if __name__ == "__main__":
    main()
