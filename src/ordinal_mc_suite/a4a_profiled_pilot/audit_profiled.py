"""Audit saved trajectories, profile heads, seed separation, and primary selections."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from prior import BALANCED, HERE, ORIGINAL
from fitting import loss_parts, new_scorer, normalize
from paths import choose
from profile_objective import fitted_module, objective
from run import dump_csv, dump_json, file_hash, training_sample
from run_profiled import read
from training import tensor_hash
from heads import HeadSpec


def seeds_from_design(path):
    record = read(path)
    return set(record["seeds"].values()) | {record["scorer_seed"]}


def verify_preflight():
    checks = read(HERE / "preflight/checks.json")
    errors = [x["best_abs_error"] for x in checks["finite_difference"].values()]
    assert len(errors) == 4 and max(errors) < 2e-5
    return max(errors)


def verify_previous(manifest):
    assert manifest["config_sha256"] == file_hash(ORIGINAL / "config.json")
    assert manifest["protocol_sha256"] == file_hash(HERE / "protocol.json")
    assert manifest["coefficient_sha256"] == file_hash(BALANCED / "calibration/selected.json")
    for name, sha in manifest["source_sha256"].items():
        assert file_hash(HERE.parent / name) == sha
    known = manifest["known_controls"]
    assert file_hash(HERE.parent / known["manifest"]) == known["sha256"]
    assert known["reused"] and known["new_scorer_updates"] == 0
    seen = set()
    for stage in ("replay", "confirmation"):
        for path in (BALANCED / "evaluation" / stage).glob("rep_*/design.json"):
            seen.update(seeds_from_design(path))
    for path in (BALANCED / "calibration").glob("rep_*/anchor.json"):
        seen.update(v for k, v in read(path).items() if k.endswith("_seed"))
    return seen


def match_checkpoint(row, record):
    assert row["update"] == record["update"]
    assert row["path"] == record["path"]
    assert row["state_hash"] == record["state_hash"]
    assert row["head_name"] == record["head_name"]
    assert abs(row["selector"] - record["selector"]) < 1e-12


def verify_choices(folder, checkpoint, manifest):
    protocol, cfg = manifest["protocol"], manifest["config"]
    traced = pd.read_csv(folder / "own_validation.csv")
    history = pd.read_csv(folder / "historical_common_validation.csv")
    library = pd.read_csv(folder / "equal_library.csv")
    selected = checkpoint["results"]
    updates = list(range(0, cfg["joint_updates"] + 1, cfg["check_every"]))
    start_hash = checkpoint["anchor"]["state_hash"]
    assert traced[traced["update"] == 0].state_hash.eq(start_hash).all()
    assert set(traced[traced["path"] == "O"].head_name) == set(manifest["weights"])
    for arm, endpoints in selected.items():
        if arm == "F":
            rows = traced[(traced["path"] == "O") & (traced["update"] == 0)].to_dict("records")
            match_checkpoint(choose(rows, protocol), {**endpoints["selected"], "path": "O"})
        else:
            source = (traced[traced.path.str.startswith("Profile_")] if arm == "Profile_selected"
                      else traced[traced["path"] == arm])
            for _, block in source.groupby(["path", "head_name"]):
                assert block["update"].tolist() == updates
            for endpoint, cutoff in (("selected", -1), ("selected_active", 0), ("fixed", cfg["joint_updates"] - 1)):
                rows = source[source["update"] > cutoff].to_dict("records")
                match_checkpoint(choose(rows, protocol, family=arm == "Profile_selected"), endpoints[endpoint])
        for record in endpoints.values():
            assert tensor_hash(record["model"]) == record["state_hash"]
            assert record["native"][:, 1:].min() >= cfg["min_slope"] - 1e-9
        candidates = library[library.arm == arm].to_dict("records")
        assert len(candidates) == 4
        chosen = choose(candidates, protocol, family=True)
        assert chosen["chosen"] and chosen["head_name"] == endpoints["selected"]["equal_library_head_name"]
    legacy = history[history["path"] == "Legacy_Adam_LDC"]
    best_historical = legacy.loc[legacy.selector.idxmin()]
    legacy_record = selected["Legacy_Adam_LDC"]["selected_common"]
    assert best_historical["update"] == legacy_record["update"]
    assert abs(best_historical["selector"] - legacy_record["historical_selector"]) < 1e-12


def verify_evaluation(folder):
    data = pd.read_csv(folder / "outcome.csv")
    assert set(data["where"]) == {"seen"}
    assert set(data["panel"]) == {"supported_mixed", "supported_small_mixed"}
    np.testing.assert_allclose(data.mean_change_mse,
        data.mean_residual_date0_mse + data.mean_residual_date1_mse - 2 * data.mean_residual_cross_moment,
        atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(data.realized_change_mse, data.mean_change_mse + data.mean_change_noise_floor
                               + data.realized_change_cross_term, atol=1e-10, rtol=1e-10)
    structural = pd.read_csv(folder / "structural.csv")
    for left, right in (("acquisition_only", "map_only"), ("structural_only", "opposing_shifted_map")):
        a, b = (structural[structural["panel"] == panel].drop(columns="panel").reset_index(drop=True)
                for panel in (left, right))
        pd.testing.assert_frame_equal(a, b, check_exact=True)


def endpoint_gradients(rep, checkpoint, manifest):
    """Differentiate saved estimators at their saved heads without optimizer steps."""
    cfg = manifest["config"]
    device = manifest["device"]
    dtype = torch.float32 if device == "cuda" else torch.float64
    _, _, _, _, _, fit, _, _, denoms, _, _ = training_sample(cfg, rep, device, dtype)
    model = new_scorer(0, cfg, device, dtype)
    spec = HeadSpec(**checkpoint["spec"])
    rows = []
    for arm, endpoints in checkpoint["results"].items():
        if arm == "Profile_selected":
            continue
        for label in ("selected", "fixed"):
            if label not in endpoints:
                continue
            record = endpoints[label]
            model.load_state_dict(record["model"])
            z, _ = normalize(model(fit.inputs), fit, cfg["min_score_variance"])
            if arm in ("F", "O"):
                parts = loss_parts(z, fit, cfg["taus"][-1], denoms)
                value = parts["rank"] + cfg["lambda_stability"] * parts["stability"]
            else:
                weights = manifest["weights"][record["head_name"]]
                value, _ = objective(z, fit, fitted_module(spec, record["native"], fit), denoms, cfg, weights)
            grads = torch.autograd.grad(value, tuple(model.parameters()))
            norm = float(torch.sqrt(sum(g.square().sum() for g in grads)))
            rows.append(dict(rep=rep, arm=arm, endpoint=label, update=record["update"],
                             gradient_norm=norm, objective=float(value.detach())))
    return rows


def verify_run(run, out):
    manifest = read(run / "manifest.json")
    cfg, protocol = manifest["config"], manifest["protocol"]
    previous_seeds = verify_previous(manifest)
    roles = {rep: role for role in ("replay", "confirmation") for rep in protocol[role]}
    completed = read(run / "completion.json")
    assert not (run / "failure.txt").exists()
    assert completed["executed_scorer_updates"] == 57000
    assert sum(item["profile_head_solves"] for item in completed["replications"]) == 24024
    norms, rows = [], []
    for rep, role in roles.items():
        folder = run / f"rep_{rep}_start_0"
        seeds = seeds_from_design(folder / "design.json")
        status = read(folder / "completion.json")
        if role == "confirmation":
            assert not previous_seeds & seeds
            previous_seeds.update(seeds)
        else:
            assert seeds <= previous_seeds and status["historical_equivalence"] and all(status["historical_equivalence"].values())
        checkpoint = torch.load(folder / "checkpoints.pt", map_location="cpu", weights_only=False)
        verify_choices(folder, checkpoint, manifest)
        verify_evaluation(folder)
        fit = pd.read_csv(folder / "fitting_trajectories.csv")
        profile = fit[fit.path.str.startswith("Profile_")]
        assert profile.groupby("path").size().eq(cfg["joint_updates"] + 1).all()
        assert np.max(abs(profile.head_gap)) < 5e-7
        own_adam = fit[(fit["update"] == 0) & fit.path.isin(["Adam_LD", "Adam_LDC"])]
        assert np.max(abs(own_adam.head_gap)) < 5e-7
        cold = pd.read_csv(folder / "conditional_cold_audit.csv")
        profile_cold = cold[cold.arm.str.startswith("Profile_")]
        assert np.max(abs(profile_cold.saved_minus_cold)) < 5e-8
        assert np.max(profile.projected_head_gradient) < 5e-5
        assert np.max(cold.cold_projected_gradient) < 5e-5
        precision = pd.read_csv(folder / "precision_audit.csv")
        assert precision.score_max_abs_error.max() < 2e-4
        assert precision.outcome_max_abs_error.max() < 2e-3
        norms.extend(endpoint_gradients(rep, checkpoint, manifest))
        rows.append(dict(rep=rep, role=role, runtime_seconds=status["elapsed_seconds"],
                         historical_equivalence=status["historical_equivalence"],
                         max_profile_gradient=float(profile.projected_head_gradient.max()),
                         max_profile_cold_loss_gap=float(np.max(abs(profile_cold.saved_minus_cold))),
                         max_profile_cold_prediction_diff=float(np.max(profile_cold.validation_prediction_max_diff)),
                         max_precision_score=float(precision.score_max_abs_error.max()),
                         max_precision_outcome=float(precision.outcome_max_abs_error.max())))
    out.mkdir(parents=True, exist_ok=False)
    dump_json(out / "audit.json", dict(passed=True, substantive_scorer_updates=57000,
                                      profile_head_solves=24024, finite_difference_max_error=verify_preflight(),
                                      reps=rows))
    dump_csv(out / "endpoint_gradients.csv", norms)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        verify_run(args.run, args.out)


if __name__ == "__main__":
    main()
