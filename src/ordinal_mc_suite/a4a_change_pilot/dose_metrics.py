"""Threshold the already specified dose panel using saved scores and stable cutoffs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ordinal_mc.core import Features
from data import counterfactual, replication, test_panels
from evaluation import delta, evaluator, rms
from fitting import new_scorer


def score_dose(folder, cfg, device):
    rep, start = [int(token) for token in folder.name.split("_")[1::2]]
    runtime, dgp, seeds, contexts, fit, _, _, _ = replication(cfg, rep)
    features = Features(fit.x, "nn", runtime)
    cutoff_panel, panels = test_panels(cfg, runtime, dgp, seeds, contexts)
    checkpoint = torch.load(folder / "checkpoints.pt", weights_only=False, map_location="cpu")
    dtype = torch.float32 if device == "cuda" else torch.float64
    model = new_scorer(0, cfg, device, dtype)
    with (folder / "ranks.csv").open(encoding="utf-8") as file:
        saved = {(row["arm"], row["where"]): float(row["cutoff"]) for row in csv.DictReader(file)
                 if row["endpoint"] == "selected"}
    rows = []
    for arm, records in checkpoint["results"].items():
        predict = evaluator(records["selected"], model, features)
        cutoff = float(np.quantile(abs(delta(predict(cutoff_panel.x))), cfg["cutoff_quantile"], method="higher"))
        for where, (_, panel) in panels.items():
            assert np.isclose(cutoff, saved[arm, where], atol=1e-9, rtol=1e-9)
            f0 = panel.f[:panel.m]
            z0 = predict(panel.x)[:panel.m]
            start_group = np.searchsorted(np.quantile(f0, [1 / 3, 2 / 3]), f0)
            for dose in cfg["dose_steps"]:
                x1, f1 = counterfactual(panel, dose, dgp)
                truth, pred = f1 - f0, predict(x1) - z0
                all_rows = np.ones(panel.m, bool)
                bins = cfg["change_bins"][1:] + [np.inf]
                masks = [("all", all_rows), ("abs_gt_0.05", abs(truth) > 0.05)]
                masks += [(f"abs_{lo:g}_to_{hi:g}", (abs(truth) > lo) & (abs(truth) <= hi))
                          for lo, hi in zip(bins, bins[1:])]
                masks += [(f"start_tertile_{k + 1}", start_group == k) for k in range(3)]
                for stratum, keep in masks:
                    if not keep.any():
                        continue
                    d_true, d_pred = truth[keep], pred[keep]
                    detected = abs(d_pred) > cutoff
                    sign = np.sign(d_pred) == np.sign(d_true)
                    q = np.quantile(d_true, [0.05, 0.5, 0.95])
                    rows.append(dict(rep=rep, start=start, arm=arm, where=where, dose=dose, stratum=stratum,
                                     count=len(d_true), cutoff=cutoff, true_delta_mean=float(d_true.mean()),
                                     true_delta_q05=q[0], true_delta_q50=q[1], true_delta_q95=q[2],
                                     predicted_delta_mean=float(d_pred.mean()), rmse=rms(d_pred - d_true),
                                     zero_change_rmse=rms(d_true), flag_rate=float(detected.mean()),
                                     sign_accuracy=float(sign.mean()) if dose != 0 else np.nan,
                                     correct_signed_flag_rate=float((detected & sign).mean()) if dose != 0 else np.nan))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    cfg = manifest["config"]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    with threadpool_limits(limits=1):
        for folder in sorted(args.run.glob("rep_*")):
            rows.extend(score_dose(folder, cfg, manifest["device"]))
    with args.out.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
