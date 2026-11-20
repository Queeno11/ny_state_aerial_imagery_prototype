"""Fit-only gradient scales and fresh, observed-validation coefficient selection."""
import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_limits

from prior import HERE, ORIGINAL
from ordinal_mc.core import seed_for
from data import replication
from fitting import fit_data, new_scorer, scores
from heads import HeadSpec, MonotoneHead
from run import dump_csv, dump_json, file_hash
from training import baseline, calibrate as fit_calibration
from weighted import continuation, gradient_table


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def configs(design_path):
    design = read(design_path)
    base_path = (design_path.parent / design["base_config"]).resolve()
    cfg = read(base_path)
    roles = [design[k] for k in ("calibration_reps", "replay_reps", "confirmation_reps")]
    ids = sum(roles, [])
    assert len(ids) == len(set(ids))
    assert roles[1] == cfg["replications"]
    roots = [seed_for(cfg["seed"], "rep", rep) for rep in ids]
    assert len(set(roots)) == len(roots)
    return design, cfg, base_path


def manifest(design_path, base_path, cfg, design, stage, device, selected_path=None):
    sources = sorted(HERE.glob("*.py")) + sorted(ORIGINAL.glob("*.py"))
    sources += [HERE.parent / name for name in ("ordinal_mc/core.py", "ordinal_mc/design.py",
                                                "ordinal_mc/e6/data.py")]
    return dict(stage=stage, config=cfg, design=design, device=device,
                config_sha256=file_hash(base_path), design_sha256=file_hash(design_path),
                selected_sha256=file_hash(selected_path) if selected_path else None,
                source_sha256={str(p.relative_to(HERE.parent)): file_hash(p) for p in sources},
                versions=dict(python=platform.python_version(), torch=str(torch.__version__),
                              numpy=np.__version__, scipy=scipy.__version__),
                scorer_dtype="float32" if device == "cuda" else "float64", head_refit_dtype="float64",
                gpu=torch.cuda.get_device_name() if device == "cuda" else None,
                selector="actual observed misranking + .3*external stability + level MSE/Vfit(W) + "
                         "paired change MSE/Varfit(delta W), common unpenalized head refit",
                coefficient_rule="median over fitting-only anchors of hypot(rank_grad,.3*stable_grad)/cardinal_grad "
                                 "times the candidate relative budget",
                candidate_tie_rule="among mean validation minima within design tolerance, prefer lower "
                                   "sum of relative cardinal budgets, then lower gamma, then candidate name")


@dataclass
class FitAnchor:
    rep: int
    model: torch.nn.Module
    fit: object
    val: object
    panel: object
    valid: object
    spec: HeadSpec
    denoms: dict
    anchor: dict
    fixed: dict
    head0: np.ndarray
    trace: list
    diagnostics: list
    norms: dict


def make_anchor(rep, cfg, device, dtype, out):
    runtime, _, seeds, _, panel, valid, aux, aux_val = replication(cfg, rep)
    fit, val, _, denoms = fit_data(panel, valid, aux, aux_val, cfg, runtime, device, dtype)
    start_seed = seed_for(cfg["seed"], "scorer", rep, 0)
    model = new_scorer(start_seed, cfg, device, dtype)
    trace = []
    anchor, fixed = baseline(model, fit, val, cfg, denoms, trace)
    z, _, _ = scores(model, fit, val, cfg)
    z = z[:fit.n].cpu().double().numpy()
    spec = HeadSpec(np.quantile(z, cfg["knot_quantiles"]), float(panel.w.mean()),
                    np.sqrt(denoms["W"]), cfg["min_slope"])
    head, info = fit_calibration(z, panel, spec, denoms, cfg)
    module = MonotoneHead(spec, head, dtype, device)
    rows, norms = gradient_table(model, fit, denoms, cfg, module, (0., 0., 0.), path="anchor", update=0)
    for k in ("level", "change", "ref"):
        if not math.isfinite(norms[k]) or norms[k] <= 1e-12:
            raise ArithmeticError(f"Degenerate anchor gradient: rep {rep}, {k} = {norms[k]}")
    norms.update(rep=rep, ratio_W=norms["ref"] / norms["level"], ratio_D=norms["ref"] / norms["change"])
    dump_json(out / "anchor.json", dict(role="fit and observed validation only", rep=rep,
                                        fit_seed=seeds["paired"], split_seed=seeds["split"],
                                        context_seed=seeds["contexts"], fit_aux_seed=seeds["fit_aux"],
                                        val_aux_seed=seeds["val_aux"], scorer_seed=start_seed,
                                        train_ids_sha256=hashlib.sha256(panel.ids.tobytes()).hexdigest(),
                                        validation_ids_sha256=hashlib.sha256(valid.ids.tobytes()).hexdigest(),
                                        anchor_update=anchor["update"], anchor_hash=anchor["state_hash"],
                                        fit_tracts=panel.m, validation_tracts=valid.m, denoms=denoms,
                                        gradient_norms=norms, head_fit=info))
    torch.save(dict(anchor=anchor, fixed=fixed, coefficients=head, spec=vars(spec)), out / "anchor.pt")
    dump_csv(out / "anchor_training.csv", trace)
    return FitAnchor(rep, model, fit, val, panel, valid, spec, denoms, anchor, fixed, head, [], rows, norms)


def candidate_name(r1, r2):
    return f"rW{r1:g}_rD{r2:g}"


def run_candidate(anchor, name, weights, cfg, design, out):
    started = time.perf_counter()
    result = continuation(name, weights, anchor.model, anchor.anchor, anchor.fit, anchor.val,
                          anchor.panel, anchor.valid, anchor.spec, anchor.denoms, cfg, anchor.head0,
                          anchor.trace, anchor.diagnostics, design["adam_checkpoints"])
    subset = [r for r in anchor.trace if r["path"] == name]
    best = result["selected"]
    row = next(r for r in subset if r["update"] == best["update"])
    native_level = row.get("val_native_level", row["val_level"])
    native_change = row.get("val_native_change", row["val_change"])
    valid_rank = row["val_rank"] + cfg["lambda_stability"] * row["val_stability"]
    summary = dict(rep=anchor.rep, candidate=name, lambda_W=weights[0], lambda_D=weights[1], gamma=weights[2],
                   selected_update=best["update"], validation=best["selector"],
                   val_rank=row["val_rank"], val_stability=row["val_stability"],
                   val_common_level=row["val_level"], val_common_change=row["val_change"],
                   val_native_level=native_level, val_native_change=native_change,
                   val_native_composite=valid_rank + native_level + native_change,
                   zero_validation=subset[0]["selector"], fixed_validation=result["fixed"]["selector"],
                   active_validation=result["selected_active"]["selector"],
                   max_head_projected_grad=max(r["head_projected_grad"] for r in subset),
                   elapsed_seconds=time.perf_counter() - started, executed_updates=cfg["joint_updates"])
    destination = out / f"rep_{anchor.rep}" / name
    destination.mkdir()
    dump_csv(destination / "trajectory.csv", subset)
    dump_json(destination / "summary.json", summary)
    torch.save(result, destination / "checkpoints.pt")
    return summary


def aggregate(rows, scale_w, scale_d, n_reps):
    results = []
    for name in dict.fromkeys(x["candidate"] for x in rows):
        subset = [x for x in rows if x["candidate"] == name]
        assert len(subset) == n_reps
        a = subset[0]
        results.append(dict(candidate=name, lambda_W=a["lambda_W"], lambda_D=a["lambda_D"], gamma=a["gamma"],
                            gradient_budget=a["lambda_W"] / scale_w + a["lambda_D"] / scale_d,
                            mean_validation=float(np.mean([x["validation"] for x in subset])),
                            mean_native_validation=float(np.mean([x["val_native_composite"] for x in subset])),
                            mean_common_level=float(np.mean([x["val_common_level"] for x in subset])),
                            mean_common_change=float(np.mean([x["val_common_change"] for x in subset])),
                            mean_native_level=float(np.mean([x["val_native_level"] for x in subset])),
                            mean_native_change=float(np.mean([x["val_native_change"] for x in subset])),
                            **{f"rep_{x['rep']}": x["validation"] for x in subset},
                            **{f"update_{x['rep']}": x["selected_update"] for x in subset}))
    return sorted(results, key=lambda a: (a["mean_validation"], a["gradient_budget"], a["gamma"], a["candidate"]))


def choose(rows, tolerance, column="mean_validation"):
    best = min(row[column] for row in rows)
    eligible = [row for row in rows if row[column] <= best + tolerance]
    return min(eligible, key=lambda a: (a["gradient_budget"], a["gamma"], a["candidate"]))


def run(cfg, design, out, device, dtype):
    anchors = []
    started = time.perf_counter()
    for rep in design["calibration_reps"]:
        path = out / f"rep_{rep}"
        path.mkdir()
        anchor = make_anchor(rep, cfg, device, dtype, path)
        anchors.append(anchor)
        print(f"Calibration rep {rep}: anchor {anchor.anchor['update']}, ratios "
              f"{anchor.norms['ratio_W']:.6g}/{anchor.norms['ratio_D']:.6g}", flush=True)
    scale_w = float(np.median([a.norms["ratio_W"] for a in anchors]))
    scale_d = float(np.median([a.norms["ratio_D"] for a in anchors]))
    grid = [(candidate_name(rw, rd), (rw * scale_w, rd * scale_d, 0.))
            for rw, rd in itertools.product(design["gradient_budgets"], repeat=2)]
    grid.append(("original_0.1_0.1", tuple(design["old_weights"])))
    dump_json(out / "grid.json", dict(scale_W=scale_w, scale_D=scale_d,
                                     candidates=[dict(name=name, weights=weights) for name, weights in grid]))
    rows = []
    for name, weights in grid:
        for a in anchors:
            print(f"Grid {name}, rep {a.rep}", flush=True)
            rows.append(run_candidate(a, name, weights, cfg, design, out))
    n = len(anchors)
    initial = aggregate(rows, scale_w, scale_d, n)
    positive = [r for r in initial if min(r["lambda_W"], r["lambda_D"]) > 0]
    tol = design["tie_tolerance"]
    best_pair = choose(positive, tol)
    best_grid = choose(initial, tol)
    gamma_names = [best_pair["candidate"]]
    for gamma in design["gamma_grid"]:
        if gamma == 0:
            continue
        name = f"{best_pair['candidate']}_gamma{gamma:g}"
        gamma_names.append(name)
        weights = best_pair["lambda_W"], best_pair["lambda_D"], gamma
        for a in anchors:
            print(f"Curvature {name}, rep {a.rep}", flush=True)
            rows.append(run_candidate(a, name, weights, cfg, design, out))
    table = aggregate(rows, scale_w, scale_d, n)
    curvatures = [r for r in table if r["candidate"] in gamma_names]
    best_full = choose(curvatures, tol)
    best_curved = choose([r for r in curvatures if r["gamma"] > 0], tol)
    best_global = choose(table, tol)
    margin = lambda group: sorted(r["mean_validation"] for r in group)[1] - min(r["mean_validation"] for r in group)
    draw_winners = {}
    for a in anchors:
        column = f"rep_{a.rep}"
        draw_winners[str(a.rep)] = dict(overall=choose(table, tol, column)["candidate"],
                                        grid=choose(initial, tol, column)["candidate"],
                                        both=choose(positive, tol, column)["candidate"],
                                        curvature=choose(curvatures, tol, column)["candidate"])
    dump_csv(out / "candidate_by_rep.csv", rows)
    dump_csv(out / "candidate_aggregate.csv", table)
    dump_csv(out / "anchor_gradient_scales.csv", [a.norms for a in anchors])
    for a in anchors:
        dump_csv(out / f"rep_{a.rep}" / "gradient_diagnostics.csv", a.diagnostics)
    dump_json(out / "selected.json", dict(scale_W=scale_w, scale_D=scale_d, best_grid=best_grid,
                                         best_positive_pair=best_pair, best_full=best_full,
                                         best_positive_curvature=best_curved, best_global=best_global,
                                         winner_by_draw=draw_winners,
                                         runner_up_margin=dict(grid=margin(initial), positive_pair=margin(positive),
                                                               global_=margin(table), curvature=margin(curvatures)),
                                         stage1_candidates=len(initial), final_candidates=len(table),
                                         total_updates=n * cfg["baseline_updates"] + sum(r["executed_updates"] for r in rows),
                                         runtime_seconds=time.perf_counter() - started))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--design", type=Path, default=HERE / "design.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = p.parse_args()
    design, cfg, base = configs(args.design.resolve())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=False)
    dump_json(args.out / "manifest.json", manifest(args.design, base, cfg, design, "calibration", args.device))
    try:
        with threadpool_limits(limits=1):
            run(cfg, design, args.out, args.device,
                torch.float32 if args.device == "cuda" else torch.float64)
    except Exception:
        (args.out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
