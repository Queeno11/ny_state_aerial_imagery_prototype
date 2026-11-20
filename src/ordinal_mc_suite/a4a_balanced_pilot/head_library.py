"""Give every saved selected scorer the same validation-only head choice."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from prior import HERE
from data import counterfactual
from evaluation import delta, evaluator, outcome, supported_masks
from fitting import new_scorer, scores
from heads import HeadSpec, fit_heads
from ordinal_mc.e6.data import measurement
from run import dump_csv, dump_json, file_hash, training_sample
from training import mse
from calibrate import read


def library(selected):
    pair = selected["best_positive_pair"]
    gamma = selected["best_full"]["gamma"]
    if gamma == 0:
        gamma = selected["best_positive_curvature"]["gamma"]
    w, d = pair["lambda_W"], pair["lambda_D"]
    return {"level_only": (1., 0., 0.), "equal_level_change": (1., 1., 0.),
            "balanced": (w, d, 0.), "balanced_curvature": (w, d, gamma)}


def calibrations(library, record, fit, val, train, valid, spec, denoms, cfg, model, tolerance):
    model.load_state_dict(record["model"])
    zf, zv, (mu, sd) = scores(model, fit, val, cfg)
    assert mu == record["score_mu"] and sd == record["score_sd"]
    zf, zv = zf[:fit.n].cpu().double().numpy(), zv[:val.n].cpu().double().numpy()
    candidates, states = [], {}
    for key, (w, d, gamma) in library.items():
        # Rescale the whole convex objective for a comparable numerical tolerance.
        coeffs, audit = fit_heads(zf, train, spec, denoms, cfg["head_max_iter"],
                                  level=1., change=d / w, gamma=gamma / w)
        level, change = mse(spec.direct(coeffs, zv, valid.cells), valid, denoms)
        states[key] = coeffs
        candidates.append(dict(head_candidate=key, lambda_W=w, lambda_D=d, gamma=gamma,
                               val_level=level, val_change=change, val_criterion=level + change,
                               projected_gradient=audit["projected_gradient"], head_iterations=audit["iterations"]))
    best = min(x["val_criterion"] for x in candidates)
    chosen = next(x["head_candidate"] for x in candidates if x["val_criterion"] <= best + tolerance)
    return chosen, candidates, states


def one_rep(folder, out, manifest, choices):
    stage = manifest["stage"]
    cfg = manifest["config"]
    device = manifest["device"]
    dtype = torch.float32 if device == "cuda" else torch.float64
    rep = int(folder.name.split("_")[1])
    dgp, _, train, valid, _, fit, val, features, denoms, _, panels = training_sample(cfg, rep, device, dtype)
    checkpoint = torch.load(folder / "checkpoints.pt", weights_only=False, map_location="cpu")
    spec = HeadSpec(**checkpoint["spec"])
    assert denoms == checkpoint["denoms"]
    model = new_scorer(0, cfg, device, dtype)
    p, fixture = panels["seen"]
    n = fixture.m
    f0 = fixture.f[:n]
    x1, f1 = counterfactual(fixture, .15 * np.where(np.arange(n) % 2, 1, -1), dgp)
    w0 = fixture.mean_w[:n]
    w1 = measurement(f1, fixture.a[n:], fixture.b[n:], fixture.k[n:], "nonlinear")
    exact = np.r_[w0, w1]
    opposite = ((w1 - w0) * (f1 - f0) < 0) & (abs(f1 - f0) > .05)
    mixed_masks = [(name, mask) for name, _, mask in supported_masks(p, delta(p.f), cfg)[:5]]
    candidates, outcomes, selection, saved, seen = [], [], [], {}, {}
    for arm, records in checkpoint["results"].items():
        record = records["selected"]
        key = record["state_hash"]
        if key not in seen:
            seen[key] = calibrations(choices, record, fit, val, train, valid, spec, denoms, cfg, model,
                                     manifest["design"]["tie_tolerance"])
        chosen, rows, states = seen[key]
        saved[arm] = dict(chosen=chosen, heads=states)
        tags = dict(stage=stage, rep=rep, arm=arm, state_hash=key)
        candidates.extend({**tags, **row} for row in rows)
        row = next(row for row in rows if row["head_candidate"] == chosen)
        selection.append({**tags, **row})
        tags["head"] = chosen
        predict = evaluator(record, model, features)
        estimates = spec.direct(states[chosen], predict(p.x), p.cells)
        for label, mask in mixed_masks:
            if mask.any():
                outcomes.append(dict(**tags, panel="supported_mixed", stratum=label,
                                     **outcome(estimates, p.w, p.mean_w, np.flatnonzero(mask))))
        z0 = predict(fixture.x)[:n]
        small = spec.direct(states[chosen], np.r_[z0, predict(x1)], fixture.cells)
        for label, mask in (("all", np.ones(n, dtype=bool)), ("opposed", opposite)):
            if mask.any():
                outcomes.append(dict(**tags, panel="supported_small_mixed", stratum=label,
                                     **outcome(small, exact + fixture.errors, exact, np.flatnonzero(mask))))
    assert saved["A_anchor"]["chosen"] == saved["F_anchor"]["chosen"]
    assert np.array_equal(saved["A_anchor"]["heads"][saved["A_anchor"]["chosen"]],
                          saved["F_anchor"]["heads"][saved["F_anchor"]["chosen"]])
    dest = out / f"{stage}_{folder.name}"
    dest.mkdir()
    torch.save(dict(heads=saved, spec=vars(spec)), dest / "head_checkpoints.pt")
    return candidates, selection, outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected = read(args.selected)
    choices = library(selected)
    args.out.mkdir(parents=True, exist_ok=False)
    dump_json(args.out / "manifest.json",
              dict(library=choices, selected_sha256=file_hash(args.selected), self_sha256=file_hash(Path(__file__)),
                   selector="validation standardized level + change MSE", optimized_scorer_updates=0,
                   tie_order=list(choices)))
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    candidates, selection, outcomes = [], [], []
    with threadpool_limits(limits=1):
        for root in (args.replay, args.confirmation):
            manifest = read(root / "manifest.json")
            assert manifest["selected_sha256"] == file_hash(args.selected)
            assert (root / "completion.json").exists()
            for folder in sorted(root.glob("rep_*")):
                c, s, o = one_rep(folder, args.out, manifest, choices)
                candidates.extend(c)
                selection.extend(s)
                outcomes.extend(o)
    dump_csv(args.out / "candidates.csv", candidates)
    dump_csv(args.out / "choices.csv", selection)
    dump_csv(args.out / "outcomes.csv", outcomes)
    dump_json(args.out / "completion.json",
              dict(elapsed_seconds=time.perf_counter() - started, scorer_updates=0,
                   candidate_rows=len(candidates), selected_rows=len(selection)))


if __name__ == "__main__":
    main()
