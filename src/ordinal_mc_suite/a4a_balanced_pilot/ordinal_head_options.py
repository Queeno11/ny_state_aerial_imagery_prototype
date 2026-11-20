"""Compare already fitted ordinal head options after the validation choice is frozen."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from prior import HERE
from calibrate import read
from evaluation import evaluator, outcome
from fitting import new_scorer
from heads import HeadSpec
from run import dump_csv, dump_json, training_sample


def check_candidates(root, tolerance):
    candidates = pd.read_csv(root / "candidates.csv")
    choices = pd.read_csv(root / "choices.csv")
    for (stage, rep, arm), block in candidates.groupby(["stage", "rep", "arm"], sort=False):
        expected = block.loc[block["val_criterion"] <= block["val_criterion"].min() + tolerance].iloc[0]
        saved = choices.query("stage == @stage and rep == @rep and arm == @arm").iloc[0]
        assert len(block) == 4 and saved["head_candidate"] == expected["head_candidate"]
    reported = pd.read_csv(root / "outcomes.csv")
    np.testing.assert_allclose(reported.mean_change_mse,
        reported.mean_residual_date0_mse + reported.mean_residual_date1_mse - 2 * reported.mean_residual_cross_moment,
        rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(reported.realized_change_mse, reported.mean_change_mse
        + reported.mean_change_noise_floor + reported.realized_change_cross_term, rtol=1e-10, atol=1e-10)
    return candidates


def one_rep(path, manifest, root, candidates):
    cfg = manifest["config"]
    stage = manifest["stage"]
    device = manifest["device"]
    dtype = torch.float32 if device == "cuda" else torch.float64
    rep = int(path.name.split("_")[1])
    _, _, _, _, _, _, _, features, _, _, panels = training_sample(cfg, rep, device, dtype)
    panel = panels["seen"][0]
    checkpoint = torch.load(path / "checkpoints.pt", map_location="cpu", weights_only=False)
    saved = torch.load(root / f"{stage}_{path.name}" / "head_checkpoints.pt",
                       map_location="cpu", weights_only=False)
    spec = HeadSpec(**saved["spec"])
    for heads in saved["heads"].values():
        assert all(np.min(coeffs[:, 1:]) > 0 for coeffs in heads["heads"].values())
    model = new_scorer(0, cfg, device, dtype)
    rows = []
    for arm in ("A_anchor", "O_continue"):
        record = checkpoint["results"][arm]["selected"]
        z = evaluator(record, model, features)(panel.x)
        arm_candidates = candidates.query("stage == @stage and rep == @rep and arm == @arm")
        chosen = saved["heads"][arm]["chosen"]
        for candidate in arm_candidates.to_dict("records"):
            name = candidate["head_candidate"]
            prediction = spec.direct(saved["heads"][arm]["heads"][name], z, panel.cells)
            rows.append(dict(**candidate, validation_selected=name == chosen,
                             **outcome(prediction, panel.w, panel.mean_w, np.arange(panel.m))))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--confirmation", type=Path, required=True)
    p.add_argument("--head-library", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = args.head_library
    tolerance = read(args.replay / "manifest.json")["design"]["tie_tolerance"]
    candidates = check_candidates(root, tolerance)
    rows = []
    with threadpool_limits(limits=1):
        for group in (args.replay, args.confirmation):
            manifest = read(group / "manifest.json")
            for path in sorted(group.glob("rep_*")):
                rows.extend(one_rep(path, manifest, root, candidates))
    dump_csv(root / "ordinal_options.csv", rows)
    dump_json(root / "audit.json", dict(passed=True, validated_candidates=len(candidates),
                                       ordinal_diagnostic_rows=len(rows), scorer_updates=0,
                                       max_head_projected_gradient=float(candidates.projected_gradient.max())))


if __name__ == "__main__":
    main()
