"""Replay the A4(a) pilot using frozen coefficients and confirm on fresh data."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from prior import HERE, ORIGINAL
from ordinal_mc.core import seed_for
from evaluation import evaluation
from fitting import new_scorer, scores
from heads import HeadSpec
from run import dump_csv, dump_json, hardware_audit, new_tables, put_tables, training_sample
from training import baseline, calibrate, finalize, mse, refit_record, tensor_hash
from calibrate import configs, manifest, read
from weighted import continuation, gradient_table, known_control


def as_weights(record):
    return tuple(record[key] for key in ("lambda_W", "lambda_D", "gamma"))


def plan(design, selected, stage):
    w, d, _ = as_weights(selected["best_positive_pair"])
    curvature = selected["best_positive_curvature"]["gamma"]
    arms = {"O_continue": (0., 0., 0.), "L": (w, 0., 0.), "D": (0., d, 0.),
            "LD": (w, d, 0.), "LDC": (w, d, curvature)}
    aliases = {}
    def add(name, weights):
        equivalent = next((a for a, value in arms.items() if value == weights), None)
        if equivalent is None:
            arms[name] = weights
        else:
            aliases[name] = equivalent
    add("Full_selected", as_weights(selected["best_full"]))
    add("Global_selected", as_weights(selected["best_global"]))
    if stage == "confirmation":
        ow, od, _ = design["old_weights"]
        add("Old_LD", (ow, od, 0.))
        add("Old_LDC", (ow, od, design["old_curvature"]))
    return arms, aliases


def choose_deployed(record, model, fit, val, val_panel, spec, denoms, cfg, tol):
    """Choose the head by validation level+change, with ranks and stability held fixed."""
    model.load_state_dict(record["model"])
    _, zv, _ = scores(model, fit, val, cfg)
    z = zv[:val.n].cpu().double().numpy()
    common_l, common_d = mse(spec.direct(record["calibrated"], z, val_panel.cells), val_panel, denoms)
    native_l = native_d = np.nan
    choice = "common"
    if record["native"] is not None:
        native_l, native_d = mse(spec.direct(record["native"], z, val_panel.cells), val_panel, denoms)
        if native_l + native_d < common_l + common_d - tol:
            choice = "native"
    return dict(chosen_head=choice, val_common_level=common_l, val_common_change=common_d,
                val_native_level=native_l, val_native_change=native_d)


def evaluation_rep(out, cfg, design, rep, stage, arms, device, dtype):
    dest = out / f"rep_{rep}_start_0"
    dest.mkdir()
    started = time.perf_counter()
    dgp, seeds, panel, valid, aux, fit, val, features, denoms, cutoff, panels = training_sample(cfg, rep, device, dtype)
    start_seed = seed_for(cfg["seed"], "scorer", rep, 0)
    hashes = dict(train_tract_ids_sha256=hashlib.sha256(panel.ids.tobytes()).hexdigest(),
                  val_tract_ids_sha256=hashlib.sha256(valid.ids.tobytes()).hexdigest())
    dump_json(dest / "design.json", dict(role=stage, seeds=seeds, scorer_seed=start_seed, **hashes,
                                        fit_tracts=panel.m, validation_tracts=valid.m,
                                        exact_fit_links=len(aux), denoms=denoms))
    if stage == "replay":
        original = read(ORIGINAL / f"main_pilot/rep_{rep}_start_0/design.json")
        assert seeds == original["seeds"] and denoms == original["denominators"]
        assert all(original[key] == value for key, value in hashes.items())
    model = new_scorer(start_seed, cfg, device, dtype)
    trace, gradients, head_trace, precision = [], [], [], []
    anchor, fixed_anchor = baseline(model, fit, val, cfg, denoms, trace)
    z, _, _ = scores(model, fit, val, cfg)
    z = z[:fit.n].cpu().double().numpy()
    spec = HeadSpec(np.quantile(z, cfg["knot_quantiles"]), float(panel.w.mean()),
                    np.sqrt(denoms["W"]), cfg["min_slope"])
    head0, both_info = calibrate(z, panel, spec, denoms, cfg)
    level, level_info = calibrate(z, panel, spec, denoms, cfg, change=0.)
    head_trace.extend([dict(path="anchor_both", **both_info), dict(path="anchor_level", **level_info)])
    frozen = refit_record(model, fit, val, panel, valid, spec, denoms, cfg, 0, head0, head0, "F_anchor", trace)
    frozen = finalize(frozen, model, fit, val, panel, valid, spec, denoms, cfg, head_trace, "F_anchor")
    results = dict(A_anchor=dict(selected={**frozen, "native": level}), F_anchor=dict(selected=frozen))
    for arm, weights in arms.items():
        print(f"{stage} rep {rep}: {arm}", flush=True)
        records = continuation(arm, weights, model, anchor, fit, val, panel, valid, spec, denoms, cfg,
                               head0, trace, gradients)
        records["selected"] = finalize(records["selected"], model, fit, val, panel, valid, spec,
                                        denoms, cfg, head_trace, arm)
        results[arm] = records
        zero = next(row for row in trace if row["path"] == arm and row["update"] == 0)
        assert abs(zero["selector"] - frozen["selector"]) < 1e-8
    comparison = {}
    if stage == "replay":
        old = torch.load(ORIGINAL / f"main_pilot/rep_{rep}_start_0/checkpoints.pt",
                         weights_only=False, map_location="cpu")
        comparison = dict(anchor_identical=anchor["state_hash"] == old["anchor"]["state_hash"],
                          anchor_fixed_identical=fixed_anchor["state_hash"] == old["fixed_anchor"]["state_hash"],
                          continuation_selected_identical=results["O_continue"]["selected"]["state_hash"]
                          == old["results"]["O_continue"]["selected"]["state_hash"],
                          continuation_fixed_identical=results["O_continue"]["fixed"]["state_hash"]
                          == old["results"]["O_continue"]["fixed"]["state_hash"])
        assert all(comparison.values()), comparison
    choices = {}
    for arm, records in results.items():
        record = records["selected"]
        numerical = hardware_audit(record, panel, aux, fit, val, features, model, spec, cfg)
        for kind in ("calibrated", "native"):
            coefficients = record[kind]
            if coefficients is not None:
                numerical[kind + "_intercept_sd"] = float(np.std(coefficients[:, 0]))
                numerical[kind + "_mean_segment_slope_sd"] = float(np.std(coefficients[:, 1:], axis=0).mean())
        precision.append(dict(arm=arm, **numerical))
        choices[arm] = choose_deployed(record, model, fit, val, valid, spec, denoms, cfg, design["tie_tolerance"])
    torch.save(dict(results=results, anchor=anchor, fixed_anchor=fixed_anchor, spec=vars(spec),
                    feature_mu=features.mu, feature_sd=features.sd, feature_bmu=features.bmu,
                    feature_bsd=features.bsd, denoms=denoms), dest / "checkpoints.pt")
    dump_csv(dest / "training.csv", trace)
    dump_csv(dest / "head_convergence.csv", head_trace)
    dump_csv(dest / "gradient_audit.csv", gradients)
    dump_csv(dest / "precision_audit.csv", precision)
    dump_csv(dest / "head_choices.csv", [dict(rep=rep, arm=arm, **row) for arm, row in choices.items()])
    tables, selections = new_tables(), []
    for arm, endpoints in results.items():
        for endpoint, record in endpoints.items():
            if endpoint == "selected_rank" and arm != "O_continue":
                continue
            prefix = dict(rep=rep, start=0, arm=arm, endpoint=endpoint)
            selections.append(dict(**prefix, update=record["update"], selector=record["selector"],
                                   state_hash=record["state_hash"], score_mu=record["score_mu"],
                                   raw_v=record["score_sd"]**2, **record["metrics"]))
            prior_outcome_rows = len(tables["outcome"])
            evaluation(record, model, features, panel, cutoff, panels, spec, dgp, cfg,
                       prefix, tables, compact=endpoint != "selected")
            if endpoint == "selected":
                chosen = choices[arm]["chosen_head"]
                for row in tables["outcome"][prior_outcome_rows:]:
                    if row["head"] == chosen:
                        tables["outcome"].append({**row, "head": "deployed"})
    put_tables(dest, tables)
    dump_csv(dest / "selections.csv", selections)
    dump_json(dest / "completion.json", dict(executed_scorer_updates=cfg["baseline_updates"] +
                                            len(arms) * cfg["joint_updates"],
                                            elapsed_seconds=time.perf_counter() - started, replay=comparison))


def run_known(out, cfg, weights, device, dtype):
    for noise in (0., cfg["noise_ratio"]):
        dgp, seeds, panel, valid, _, fit, val, features, denoms, cutoff, panels = training_sample(cfg, 1, device, dtype, noise)
        for start in (0, 1):
            started = time.perf_counter()
            dest = out / f"noise_{noise:g}_start_{start}"
            dest.mkdir()
            start_seed = seed_for(cfg["seed"], "scorer", 1, start)
            dump_json(dest / "design.json", dict(seeds=seeds, scorer_seed=start_seed, denoms=denoms,
                                                 known_fit_f_mean=fit.mean_f, known_fit_f_sd=fit.sd_f, weights=weights))
            model = new_scorer(start_seed, cfg, device, dtype)
            gradients, _ = gradient_table(model, fit, denoms, cfg, None, weights, path="matched_start", update=0, known=True)
            trace, selections, tables, results = [], [], new_tables(), {}
            paths = ["known_ordinal", "known_full"] + (["direct_f"] if noise == 0. else [])
            for path in paths:
                print(f"Known noise {noise:g}, start {start}: {path}", flush=True)
                model = new_scorer(start_seed, cfg, device, dtype)
                results[path] = known_control(model, fit, val, denoms, cfg, weights, path, trace, gradients)
            assert len({v["initial_hash"] for v in results.values()}) == 1
            original = torch.load(ORIGINAL / f"known_control/{dest.name}/checkpoints.pt",
                                  weights_only=False, map_location="cpu")
            matched = {}
            for path in ("known_ordinal", "direct_f"):
                if path in results:
                    for endpoint in ("selected", "fixed"):
                        matched[f"{path}_{endpoint}"] = (
                            results[path][endpoint]["state_hash"] == original["results"][path][endpoint]["state_hash"])
            assert all(matched.values()), matched
            torch.save(dict(results=results, known_fit_f_mean=fit.mean_f, known_fit_f_sd=fit.sd_f,
                            feature_mu=features.mu, feature_sd=features.sd, feature_bmu=features.bmu,
                            feature_bsd=features.bsd), dest / "checkpoints.pt")
            dump_csv(dest / "training.csv", trace)
            dump_csv(dest / "gradient_audit.csv", gradients)
            for path, endpoints in results.items():
                for endpoint in ("selected", "fixed"):
                    record = endpoints[endpoint]
                    prefix = dict(rep=1, noise=noise, start=start, arm=path, endpoint=endpoint)
                    selections.append(dict(**prefix, update=record["update"], selector=record["selector"],
                                           state_hash=record["state_hash"], score_mu=record["score_mu"],
                                           raw_v=record["score_sd"]**2, **record["metrics"]))
                    evaluation(record, model, features, panel, cutoff, panels, None, dgp, cfg, prefix, tables,
                               known_units=(fit.mean_f, fit.sd_f), compact=True)
            put_tables(dest, tables)
            dump_csv(dest / "selections.csv", selections)
            dump_json(dest / "completion.json", dict(executed_scorer_updates=len(paths) * cfg["baseline_updates"],
                                                     elapsed_seconds=time.perf_counter() - started,
                                                     old_reference_equality=matched))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--design", type=Path, default=HERE / "design.json")
    p.add_argument("--selected", type=Path, required=True)
    p.add_argument("--stage", choices=("known", "replay", "confirmation"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()
    design, cfg, base = configs(args.design.resolve())
    selected = read(args.selected)
    calibration_manifest = read(args.selected.parent / "manifest.json")
    assert calibration_manifest["config"] == cfg and calibration_manifest["design"] == design
    assert calibration_manifest["stage"] == "calibration"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dtype = torch.float32 if args.device == "cuda" else torch.float64
    args.out.mkdir(parents=True, exist_ok=False)
    details = manifest(args.design, base, cfg, design, args.stage, args.device, args.selected)
    args_stage = args.stage
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            if args_stage == "known":
                w, d, _ = as_weights(selected["best_positive_pair"])
                dump_json(args.out / "manifest.json", {**details, "known_weights": [w, d, 0.]})
                run_known(args.out, cfg, (w, d, 0.), args.device, dtype)
            else:
                arms, aliases = plan(design, selected, args_stage)
                reps = design[args_stage + "_reps"]
                dump_json(args.out / "manifest.json", {**details, "reps": reps, "arms": arms, "aliases": aliases})
                for rep in reps:
                    evaluation_rep(args.out, cfg, design, rep, args_stage, arms, args.device, dtype)
        dump_json(args.out / "completion.json", dict(elapsed_seconds=time.perf_counter() - started))
    except Exception:
        (args.out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
