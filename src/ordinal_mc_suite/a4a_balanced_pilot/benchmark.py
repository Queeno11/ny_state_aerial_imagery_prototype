"""Time a handful of actual-size updates at a saved ordinal anchor."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from prior import ORIGINAL
from data import replication
from fitting import fit_data, loss_parts, new_scorer, normalize, scores
from heads import HeadSpec, MonotoneHead, fit_heads


def benchmark(cfg):
    runtime, _, _, _, panel, valid, aux, aux_val = replication(cfg, 1)
    fit, val, _, denoms = fit_data(panel, valid, aux, aux_val, cfg, runtime, "cuda", torch.float32)
    saved = torch.load(ORIGINAL / "main_pilot/rep_1_start_0/checkpoints.pt",
                       weights_only=False, map_location="cpu")
    model = new_scorer(0, cfg, "cuda", torch.float32)
    model.load_state_dict(saved["anchor"]["model"])
    spec = HeadSpec(**saved["spec"])
    z, _, _ = scores(model, fit, val, cfg)
    start = time.perf_counter()
    coeffs, head_info = fit_heads(z[:fit.n].cpu().numpy(), panel, spec, denoms, cfg["head_max_iter"])
    refit_seconds = time.perf_counter() - start
    head = MonotoneHead(spec, coeffs, torch.float32, "cuda")
    opt = torch.optim.Adam([dict(params=model.parameters(), lr=cfg["joint_lr"]),
                            dict(params=head.parameters(), lr=cfg["head_lr"])])
    seconds = []
    for step in range(14):
        torch.cuda.synchronize()
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        z, _ = normalize(model(fit.inputs), fit, cfg["min_score_variance"])
        terms = loss_parts(z, fit, cfg["taus"][-1], denoms, head)
        loss = terms["rank"] + .3 * terms["stability"] + .1 * terms["level"] + .02 * terms["change"]
        loss.backward()
        opt.step()
        head.project()
        torch.cuda.synchronize()
        if step >= 4:
            seconds.append(time.perf_counter() - start)
    return dict(timed_steps=len(seconds), warmup_steps=4, fit_endpoints=fit.n,
                fit_pairs=int(fit.signs.count_nonzero()), median_seconds=float(np.median(seconds)),
                mean_seconds=float(np.mean(seconds)), refit_seconds=refit_seconds, head_info=head_info,
                gpu=torch.cuda.get_device_name(), illustrative_weights=[.1, .02])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    cfg = json.loads((ORIGINAL / "config.json").read_text())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        result = benchmark(cfg)
    with args.out.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
