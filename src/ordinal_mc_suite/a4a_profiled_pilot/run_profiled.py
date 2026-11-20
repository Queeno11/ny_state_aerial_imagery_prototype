"""Execute the frozen warm-start profiling comparison on replay and fresh draws."""
import argparse
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

from prior import BALANCED, HERE, ORIGINAL
from evaluation import evaluation
from fitting import new_scorer, scores
from heads import HeadSpec
from ordinal_mc.core import seed_for
from paths import Problem, cardinal_path, choose, cold_check, frozen_options, ordinal_path
from preflight_profile import load_parameters
from run import dump_csv, dump_json, file_hash, hardware_audit, new_tables, put_tables, training_sample
from training import baseline, calibrate, composite, measure_rank, mse


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def root_manifest(cfg, options, protocol, path, device):
    sources = list(HERE.glob("*.py")) + list(ORIGINAL.glob("*.py"))
    sources += [HERE.parent / name for name in ("ordinal_mc/core.py", "ordinal_mc/design.py",
                                                "ordinal_mc/e6/data.py")]
    known = BALANCED / "evaluation/known/manifest.json"
    prior_selected = BALANCED / "calibration/selected.json"
    known_manifest = read(known)
    assert known_manifest["config"] == cfg
    assert known_manifest["known_weights"] == list(options["LD"])
    assert known_manifest["selected_sha256"] == file_hash(prior_selected)
    return dict(stage="profiled", device=device, config=cfg, protocol=protocol, weights=options,
                config_sha256=file_hash(ORIGINAL / "config.json"), protocol_sha256=file_hash(path),
                coefficient_sha256=file_hash(prior_selected),
                versions=dict(python=platform.python_version(), torch=str(torch.__version__),
                              numpy=np.__version__, scipy=scipy.__version__),
                gpu=torch.cuda.get_device_name() if device == "cuda" else None,
                source_sha256={str(p.relative_to(HERE.parent)): file_hash(p) for p in sources},
                selector="actual validation rank + .3*external stability + standardized level and change MSE, "
                         "evaluated with the primary trained head",
                known_controls=dict(reused=True, new_scorer_updates=0,
                                    manifest=str(known.relative_to(HERE.parent)), sha256=file_hash(known)))


def verify_replay(rep, anchor, fixed_anchor, results, old_o, design):
    folder = BALANCED / f"evaluation/confirmation/rep_{rep}_start_0"
    old_design = read(folder / "design.json")
    for key in ("seeds", "scorer_seed", "train_tract_ids_sha256", "val_tract_ids_sha256", "denoms"):
        assert design[key] == old_design[key], key
    old = torch.load(folder / "checkpoints.pt", weights_only=False, map_location="cpu")
    hashes = {
        "anchor": (anchor, old["anchor"]),
        "anchor_fixed": (fixed_anchor, old["fixed_anchor"]),
        "ordinal_fixed": (results["O"]["fixed"], old["results"]["O_continue"]["fixed"]),
        "ordinal_historical_selected": (old_o, old["results"]["O_continue"]["selected"]),
        "legacy_fixed": (results["Legacy_Adam_LDC"]["fixed"], old["results"]["LDC"]["fixed"]),
        "legacy_historical_selected": (results["Legacy_Adam_LDC"]["selected_common"], old["results"]["LDC"]["selected"]),
    }
    checks = {name: left["state_hash"] == right["state_hash"] for name, (left, right) in hashes.items()}
    checks["legacy_fixed_head"] = np.array_equal(results["Legacy_Adam_LDC"]["fixed"]["native"],
                                                  old["results"]["LDC"]["fixed"]["native"])
    checks["legacy_historical_native_head"] = np.array_equal(
        results["Legacy_Adam_LDC"]["selected_common"]["native"], old["results"]["LDC"]["selected"]["native"])
    assert all(checks.values()), checks
    return checks


def library_head(p, record, cache):
    key = record["state_hash"]
    if key in cache:
        return cache[key]
    p.model.load_state_dict(record["model"])
    z, zv, _ = scores(p.model, p.fit, p.val, p.cfg)
    rank = measure_rank(zv, p.val, p.cfg)
    zv_np = zv[:p.val.n].cpu().double().numpy()
    rows = []
    for head, (coefs, info) in frozen_options(p, z).items():
        val_w, val_d = mse(p.spec.direct(coefs, zv_np, p.valid.cells), p.valid, p.denoms)
        rows.append(dict(head_name=head, selector=composite(rank, val_w, val_d, p.cfg), level=val_w, change=val_d,
                         update=record["update"], projected_gradient=info["projected_gradient"], coefs=coefs))
    best = choose(rows, p.protocol, family=True)
    cache[key] = (best, rows)
    return best, rows


def evaluation_outputs(p, dgp, features, cutoff, panels, results, role, rep, dest):
    tables = new_tables()
    selections, audits, library_rows, precision = [], [], [], []
    cache = {}
    for arm, endpoints in results.items():
        for endpoint, record in endpoints.items():
            prefix = dict(role=role, rep=rep, start=0, arm=arm, endpoint=endpoint,
                          source_path=record["path"], head_name=record["head_name"])
            selections.append(dict(**prefix, update=record["update"], selector=record["selector"],
                                   selection_protocol=record["selection_protocol"], state_hash=record["state_hash"],
                                   historical_selector=record.get("historical_selector"), **record["metrics"]))
            view = record.copy()
            if endpoint in ("selected", "fixed") and arm != "Profile_selected":
                audits.append(cold_check(p, arm, endpoint, record))
            if endpoint == "selected":
                best, rows = library_head(p, record, cache)
                for row in rows:
                    library_rows.append(dict(role=role, rep=rep, arm=arm, state_hash=record["state_hash"],
                                             chosen=best["head_name"] == row["head_name"],
                                             **{k: v for k, v in row.items() if k != "coefs"}))
                view["calibrated"] = best["coefs"]
                record["equal_library"] = best["coefs"]
                record["equal_library_head_name"] = best["head_name"]
                precision.append(dict(arm=arm, **hardware_audit(
                    {**record, "calibrated": record["native"]}, p.panel,
                    p.fit_aux, p.fit, p.val, features, p.model, p.spec, p.cfg)))
            before = len(tables["outcome"])
            evaluation(view, p.model, features, p.panel, cutoff, panels, p.spec, dgp, p.cfg,
                       prefix, tables, compact=endpoint != "selected")
            for row in tables["outcome"][before:]:
                row["head"] = "own" if row["head"] == "native" else "equal_library"
    put_tables(dest, tables)
    dump_csv(dest / "selections.csv", selections)
    dump_csv(dest / "conditional_cold_audit.csv", audits)
    dump_csv(dest / "equal_library.csv", library_rows)
    dump_csv(dest / "precision_audit.csv", precision)


def run_rep(rep, role, out, cfg, options, protocol, device, dtype):
    dest = out / f"rep_{rep}_start_0"
    dest.mkdir()
    started = time.perf_counter()
    dgp, seeds, panel, valid, aux, fit, val, features, denoms, cutoff, panels = training_sample(cfg, rep, device, dtype)
    start_seed = seed_for(cfg["seed"], "scorer", rep, 0)
    design = dict(role=role, seeds=seeds, scorer_seed=start_seed, fit_tracts=panel.m, validation_tracts=valid.m,
                  exact_fit_links=len(aux), denoms=denoms,
                  train_tract_ids_sha256=hashlib.sha256(panel.ids.tobytes()).hexdigest(),
                  val_tract_ids_sha256=hashlib.sha256(valid.ids.tobytes()).hexdigest())
    dump_json(dest / "design.json", design)
    model = new_scorer(start_seed, cfg, device, dtype)
    baseline_trace, own_trace, common_trace, fit_trace = [], [], [], []
    anchor, fixed_anchor = baseline(model, fit, val, cfg, denoms, baseline_trace)
    z, _, _ = scores(model, fit, val, cfg)
    zf = z[:fit.n].cpu().double().numpy()
    spec = HeadSpec(np.quantile(zf, cfg["knot_quantiles"]), float(panel.w.mean()),
                    np.sqrt(denoms["W"]), cfg["min_slope"])
    common, common_info = calibrate(zf, panel, spec, denoms, cfg)
    p = Problem(model, anchor, fit, val, aux, panel, valid, cfg, protocol, spec, denoms, options, {}, common)
    p.initial = frozen_options(p, z)
    dump_json(dest / "initial_head_solves.json",
              dict(common=common_info, own={head: info for head, (_, info) in p.initial.items()}))
    results, family = {}, []
    old_o = None
    for name, (mode, head) in protocol["paths"].items():
        started_path = time.perf_counter()
        print(f"{role} {rep} {name}", flush=True)
        if mode == "ordinal":
            results[name], old_o, frozen = ordinal_path(p, own_trace, common_trace)
            results["F"] = dict(selected={**frozen, "path": "F"})
        else:
            results[name], records = cardinal_path(p, name, mode, head, own_trace, common_trace, fit_trace)
            if mode == "profile":
                family.extend(records)
        print(f"{role} {rep} {name}: own checkpoint {results[name]['selected']['update']}, "
              f"{time.perf_counter() - started_path:.1f}s", flush=True)
    results["Profile_selected"] = dict(
        selected=choose(family, protocol, family=True),
        selected_active=choose([r for r in family if r["update"] > 0], protocol, family=True),
        fixed=choose([r for r in family if r["update"] == cfg["joint_updates"]], protocol, family=True))
    verified = verify_replay(rep, anchor, fixed_anchor, results, old_o, design) if role == "replay" else {}
    for record in (r for r in own_trace if r["update"] == 0):
        assert record["state_hash"] == anchor["state_hash"]
    dump_csv(dest / "anchor_training.csv", baseline_trace)
    dump_csv(dest / "own_validation.csv", own_trace)
    dump_csv(dest / "historical_common_validation.csv", common_trace)
    dump_csv(dest / "fitting_trajectories.csv", fit_trace)
    evaluation_outputs(p, dgp, features, cutoff, panels, results, role, rep, dest)
    torch.save(dict(results=results, anchor=anchor, fixed_anchor=fixed_anchor, historical_ordinal=old_o,
                    head_initial={key: value[0] for key, value in p.initial.items()},
                    common_head_initial=common, spec=vars(spec), denoms=denoms,
                    feature_mu=features.mu, feature_sd=features.sd, feature_bmu=features.bmu,
                    feature_bsd=features.bsd), dest / "checkpoints.pt")
    report = dict(role=role, rep=rep, elapsed_seconds=time.perf_counter() - started,
                  executed_scorer_updates=cfg["baseline_updates"] + len(protocol["paths"]) * cfg["joint_updates"],
                  profile_head_solves=4 * (1 + cfg["joint_updates"]), historical_equivalence=verified)
    dump_json(dest / "completion.json", report)
    print(f"{role} {rep} completed: {report}", flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = p.parse_args()
    cfg, options = load_parameters()
    protocol = read(args.protocol)
    reps = protocol["replay"] + protocol["confirmation"]
    assert len(set(reps)) == 6
    assert set(protocol["replay"]) == set(read(BALANCED / "design.json")["confirmation_reps"])
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=False)
    dump_json(args.out / "manifest.json", root_manifest(cfg, options, protocol, args.protocol, args.device))
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            results = [run_rep(rep, role, args.out, cfg, options, protocol, args.device,
                               torch.float32 if args.device == "cuda" else torch.float64)
                       for role in ("replay", "confirmation") for rep in protocol[role]]
        dump_json(args.out / "completion.json", dict(elapsed_seconds=time.perf_counter() - started,
                  executed_scorer_updates=sum(r["executed_scorer_updates"] for r in results), replications=results))
    except Exception:
        (args.out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
