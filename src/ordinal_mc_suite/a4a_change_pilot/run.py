"""Run a fresh, explicitly bounded A4(a) stage and save reproducible checkpoints."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_limits

from ordinal_mc.core import seed_for
from data import replication, test_panels
from evaluation import evaluation
from fitting import fit_data, new_scorer, normalize, scores
from heads import HeadSpec, MonotoneHead
from training import (baseline, branch, calibrate, finalize, privileged, refit_record, tensor_hash)

HERE = Path(__file__).resolve().parent


def dump_json(path, payload):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def dump_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(rows)


def new_tables():
    return {name: [] for name in ("structural", "outcome", "ranks", "regimes", "dose")}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(config, settings, device, stage):
    inherited = [HERE.parent / name for name in ("ordinal_mc/core.py", "ordinal_mc/design.py",
                                                  "ordinal_mc/e6/data.py")]
    sources = {str(p.relative_to(HERE.parent)): file_hash(p)
               for p in sorted(HERE.glob("*.py")) + inherited}
    return dict(config=settings, config_sha256=file_hash(config), source_sha256=sources,
                stage=stage, device=device, fit_dtype="float32" if device == "cuda" else "float64",
                calibration_dtype="float64", torch=str(torch.__version__), numpy=np.__version__,
                scipy=scipy.__version__, python=platform.python_version(), platform=platform.platform(),
                gpu=torch.cuda.get_device_name() if device == "cuda" else None,
                selector="actual validation misranking + .3*S(z_val) + W_MSE/V_fit(W) + dW_MSE/Var_fit(dW)",
                anchor_selector="actual validation misranking + .3*S(z_val)",
                known_rep=1, known_start_ids=[0, 1], known_noises=[0., settings["noise_ratio"]],
                calibration="monotone PWL: three anchor knots, independent cell intercepts and four slopes, "
                            "convex L-BFGS-B equal level/change; cold final refits",
                head_projection="joint Adam directly on slopes then clamp at positive lower bound")


def hardware_audit(record, fit_panel, aux, fit, val, features, model, spec, cfg):
    model.load_state_dict(record["model"])
    device_z, _, (mean, sd) = scores(model, fit, val, cfg)
    reference = new_scorer(0, cfg, "cpu", torch.float64)
    reference.load_state_dict(record["model"])
    with torch.no_grad():
        inputs = torch.tensor(features(np.vstack((fit_panel.x, aux))), dtype=torch.float64)
        z64, (mean64, sd64) = normalize(reference(inputs), fit, cfg["min_score_variance"])
    z_device = device_z[:fit.n].cpu().double().numpy()
    z64 = z64[:fit.n].numpy()
    z_error = float(np.max(np.abs(z_device - z64)))
    coefs = record["calibrated"]
    head64 = spec.direct(coefs, z64, fit_panel.cells)
    head_device = MonotoneHead(spec, coefs, fit.inputs.dtype, fit.inputs.device)
    with torch.no_grad():
        preds = head_device(device_z[:fit.n], fit.cells).cpu().double().numpy()
    w_error = float(np.max(np.abs(head64 - preds)))
    assert z_error < 2e-4 and w_error < 2e-3, (z_error, w_error)
    return dict(score_max_abs_error=z_error, outcome_max_abs_error=w_error,
                mean_abs_error=float(abs(mean - mean64)), sd_abs_error=float(abs(sd - sd64)),
                head_slope_min=float(coefs[:, 1:].min()), head_slope_max=float(coefs[:, 1:].max()),
                fit_score_min=float(z_device.min()), fit_score_max=float(z_device.max()),
                fit_head_min=float(preds.min()), fit_head_max=float(preds.max()))


def put_tables(out, tables):
    for name, rows in tables.items():
        dump_csv(out / (name + ".csv"), rows)


def training_sample(cfg, rep, device, dtype, noise=None):
    runtime, dgp, seeds, contexts, panel, valid, aux, aux_val = replication(cfg, rep, noise)
    fit, val, features, denoms = fit_data(panel, valid, aux, aux_val, cfg, runtime, device, dtype)
    cutoff, panels = test_panels(cfg, runtime, dgp, seeds, contexts, noise=noise)
    ids = [panel.ids, valid.ids, cutoff.ids]
    ids += [p.ids for pair in panels.values() for p in pair]
    combined = np.concatenate(ids)
    assert len(np.unique(combined)) == len(combined)
    return dgp, seeds, panel, valid, aux, fit, val, features, denoms, cutoff, panels


def run_main_rep(out, cfg, rep, device, dtype):
    dgp, seeds, panel, valid, aux, fit, val, features, denoms, cutoff, panels = training_sample(cfg, rep, device, dtype)
    for start in cfg["start_ids"]:
        dest = out / f"rep_{rep}_start_{start}"
        dest.mkdir()
        start_seed = seed_for(cfg["seed"], "scorer", rep, start)
        dump_json(dest / "design.json", dict(seeds=seeds, scorer_seed=start_seed, fit_tracts=panel.m,
                                             validation_tracts=valid.m, exact_fit_links=len(aux),
                                             fit_stable=int((~panel.changed).sum()), denominators=denoms,
                                             train_tract_ids_sha256=hashlib.sha256(panel.ids.tobytes()).hexdigest(),
                                             val_tract_ids_sha256=hashlib.sha256(valid.ids.tobytes()).hexdigest()))
        model = new_scorer(start_seed, cfg, device, dtype)
        traces, head_traces, gradients, precision = [], [], [], []
        start_wall = time.perf_counter()
        anchor, fixed_anchor = baseline(model, fit, val, cfg, denoms, traces)
        z, _, _ = scores(model, fit, val, cfg)
        z = z[:fit.n].cpu().double().numpy()
        spec = HeadSpec(np.quantile(z, cfg["knot_quantiles"]), float(panel.w.mean()),
                        np.sqrt(denoms["W"]), cfg["min_slope"])
        common, common_info = calibrate(z, panel, spec, denoms, cfg)
        level, level_info = calibrate(z, panel, spec, denoms, cfg, change=0.)
        head_traces.extend([dict(path="anchor_both", **common_info), dict(path="anchor_level", **level_info)])
        frozen = refit_record(model, fit, val, panel, valid, spec, denoms, cfg, 0, common, common, "F_anchor", traces)
        results = dict(A_anchor=dict(selected={**frozen, "native": level}),
                       F_anchor=dict(selected=frozen))
        print(f"rep {rep}, start {start} anchor selected {anchor['update']}", flush=True)
        for label, weights in (("O_continue", (0., 0., 0.)),
                               *((label, (weight * int("L" in label), weight * int("D" in label),
                                          cfg["gamma"] if label == "LDC" else 0.))
                                 for label in ("L", "D", "LD", "LDC")
                                 for weight in cfg["joint_weights"])):
            print(f"rep {rep}, start {start} fitting {label}", flush=True)
            records = branch(label, weights, model, anchor, fit, val, panel, valid, spec, denoms, cfg,
                             common, traces, gradients)
            records["selected"] = finalize(records["selected"], model, fit, val, panel, valid, spec,
                                            denoms, cfg, head_traces, label)
            results[label] = records
        assert tensor_hash(results["A_anchor"]["selected"]["model"]) == tensor_hash(frozen["model"])
        assert np.array_equal(results["A_anchor"]["selected"]["calibrated"], frozen["calibrated"])
        for arm, records in results.items():
            if arm not in ("A_anchor", "F_anchor"):
                zero = next(row for row in traces if row["path"] == arm and row["update"] == 0)
                assert abs(zero["selector"] - frozen["selector"]) < 1e-8
            precision.append(dict(arm=arm, **hardware_audit(records["selected"], panel, aux, fit, val,
                                                             features, model, spec, cfg)))
        torch.save(dict(results=results, anchor=anchor, fixed_anchor=fixed_anchor,
                        feature_mu=features.mu, feature_sd=features.sd, feature_bmu=features.bmu,
                        feature_bsd=features.bsd, spec=vars(spec), denoms=denoms),
                   dest / "checkpoints.pt")
        dump_csv(dest / "training.csv", traces)
        dump_csv(dest / "head_convergence.csv", head_traces)
        dump_csv(dest / "gradient_audit.csv", gradients)
        dump_csv(dest / "precision_audit.csv", precision)
        tables = new_tables()
        selections = []
        for arm, records in results.items():
            for endpoint, record in records.items():
                if endpoint == "selected_rank" and arm != "O_continue":
                    continue
                prefix = dict(rep=rep, start=start, arm=arm, endpoint=endpoint)
                selections.append(dict(**prefix, update=record["update"], selector=record["selector"],
                                       state_hash=record["state_hash"], score_mu=record["score_mu"],
                                       raw_v=record["score_sd"]**2, **record["metrics"]))
                evaluation(record, model, features, panel, cutoff, panels, spec, dgp, cfg,
                           prefix, tables, compact=endpoint != "selected")
        put_tables(dest, tables)
        dump_csv(dest / "selections.csv", selections)
        dump_json(dest / "completion.json", dict(elapsed_seconds=time.perf_counter() - start_wall,
                                                executed_scorer_updates=cfg["baseline_updates"] + 5 * cfg["joint_updates"]))


def run_known(out, cfg, device, dtype):
    for noise in (0., cfg["noise_ratio"]):
        dgp, seeds, panel, valid, _, fit, val, features, denoms, cutoff, panels = training_sample(cfg, 1, device, dtype, noise)
        for start in (0, 1):
            dest = out / f"noise_{noise:g}_start_{start}"
            dest.mkdir()
            start_seed = seed_for(cfg["seed"], "scorer", 1, start)
            dump_json(dest / "design.json", dict(seeds=seeds, scorer_seed=start_seed, denominators=denoms,
                                                 known_fit_f_mean=fit.mean_f, known_fit_f_sd=fit.sd_f))
            paths = ["known_ordinal", "known_full"] + (["direct_f"] if noise == 0. else [])
            results, trace, tables = {}, [], new_tables()
            start_wall = time.perf_counter()
            for path in paths:
                print(f"known-map noise {noise:g}, start {start} fitting {path}", flush=True)
                model = new_scorer(start_seed, cfg, device, dtype)
                results[path] = privileged(model, fit, val, denoms, cfg, path, trace)
            assert len({records["initial_hash"] for records in results.values()}) == 1
            torch.save(dict(results=results, known_fit_f_mean=fit.mean_f, known_fit_f_sd=fit.sd_f,
                            feature_mu=features.mu, feature_sd=features.sd, feature_bmu=features.bmu,
                            feature_bsd=features.bsd), dest / "checkpoints.pt")
            dump_csv(dest / "training.csv", trace)
            selections = []
            for path, endpoints in results.items():
                for endpoint in ("selected", "fixed"):
                    record = endpoints[endpoint]
                    prefix = dict(rep=1, noise=noise, start=start, arm=path, endpoint=endpoint)
                    selections.append(dict(**prefix, update=record["update"], selector=record["selector"],
                                           state_hash=record["state_hash"], score_mu=record["score_mu"],
                                           raw_v=record["score_sd"]**2, **record["metrics"]))
                    evaluation(record, model, features, panel, cutoff, panels, None, dgp, cfg, prefix,
                               tables, known_units=(fit.mean_f, fit.sd_f), compact=True)
            put_tables(dest, tables)
            dump_csv(dest / "selections.csv", selections)
            dump_json(dest / "completion.json", dict(elapsed_seconds=time.perf_counter() - start_wall,
                                                    executed_scorer_updates=len(paths) * cfg["baseline_updates"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--stage", required=True, choices=("known", "main"))
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if len(cfg["joint_weights"]) != 1:
        raise ValueError("This pilot is registered for one fixed cardinal weight")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dtype = torch.float32 if args.device == "cuda" else torch.float64
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dump_json(out / "manifest.json", manifest(args.config, cfg, args.device, args.stage))
    start = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            if args.stage == "known":
                run_known(out, cfg, args.device, dtype)
            else:
                for rep in cfg["replications"]:
                    run_main_rep(out, cfg, rep, args.device, dtype)
        dump_json(out / "completion.json", dict(elapsed_seconds=time.perf_counter() - start))
    except Exception:
        (out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
