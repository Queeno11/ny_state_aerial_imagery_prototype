"""Diagnose the dist_to_center collapse (frozen val spearman, stable MASD == 0).

Three checks, all runnable independently:

  1. Shortcut check     — spearman(pred, dist_to_center) on a val shard. If the
                          model collapsed onto the covariate, |rho| ~ 1.0. Also
                          reports spearman(dist_to_center, label), which should
                          reproduce the frozen wandb val spearmans.
  2. Sensitivity check  — forward two visibly different images with the SAME
                          metadata. Identical outputs => the image pathway is dead.
  3. Weight autopsy     — magnitude of the image slice vs. the meta slice of
                          head.final_head.weight, and the norms of the saved
                          LoRA B matrices (zero-init; still ~0 => the backbone
                          never received gradient).

Checks 1-2 need --checkpoint (head .pth) and at least one --shard; check 3 runs
from the checkpoint files alone. Without --checkpoint, only the label-vs-meta
spearman of check 1 runs (no model needed).

Usage:
    python src/diagnose_collapse.py \
        --shard /home/abbatenicolas/data/cache/val_cities/shard_0.pt \
        --checkpoint models/models_by_epoch/<run>/<run>_best.pth \
        [--lora-dir models/models_by_epoch/<run>/<run>_best_lora] \
        [--meta-dim 1] [--nbands 4] [--device cuda]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torchvision.transforms import v2 as transforms


# ──────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────

def load_shard_tensors(shard_paths):
    """Concatenate images / scores / metas from one or more shard .pt files."""
    imgs, scores, metas = [], [], []
    for p in shard_paths:
        data = torch.load(p, weights_only=False)
        n = len(data["images"])
        if n == 0:
            continue
        imgs.append(data["images"])
        scores.append(data["scores"])
        metas.append(data.get("metas", torch.zeros(n, 1)))
    if not imgs:
        raise ValueError(f"No non-empty shards among: {shard_paths}")
    return {
        "images": torch.cat(imgs),
        "scores": torch.cat(scores),
        "metas": torch.cat(metas),
    }


def build_eval_transform(nbands):
    """Mirror of the eval_transform in main.setup_dataloaders."""
    mean = [0.485, 0.456, 0.406] + [0.5] * max(0, nbands - 3)
    std = [0.229, 0.224, 0.225] + [0.5] * max(0, nbands - 3)
    return transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=mean, std=std),
    ])


# ──────────────────────────────────────────────────────────────────────────
# Model loading (heavy: downloads the ScaleMAE backbone)
# ──────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, lora_dir=None, meta_dim=1, nbands=4,
               image_size=224, device="cpu"):
    from peft import PeftModel
    from src.custom_models import ScaleMAE

    model = ScaleMAE(image_size=image_size, bands=nbands, kind="reg", meta_dim=meta_dim)
    head_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(head_state, dict) and "model_state_dict" in head_state:
        head_state = head_state["model_state_dict"]  # _last.pth full checkpoint
        model.load_state_dict(head_state)
    else:
        model.head.load_state_dict(head_state)  # _best.pth: head weights only
        if lora_dir is not None and Path(lora_dir).exists():
            model.backbone = PeftModel.from_pretrained(
                model.backbone.base_model.model, str(lora_dir)
            )
    model.eval()
    return model.to(device)


@torch.no_grad()
def predict(model, images, metas, transform, device, batch_size=16):
    """Run the eval-path forward over a stack of raw (uint8) shard images."""
    preds = []
    for i in range(0, len(images), batch_size):
        x = transform(images[i:i + batch_size]).to(device)
        m = metas[i:i + batch_size].to(device)
        preds.append(model(x, metadata=m).view(-1).float().cpu())
    return torch.cat(preds)


# ──────────────────────────────────────────────────────────────────────────
# Check 1: shortcut spearmans
# ──────────────────────────────────────────────────────────────────────────

def spearman_report(labels, metas, preds=None):
    """Spearman correlations that identify a metadata-shortcut collapse."""
    labels = np.asarray(labels, dtype=float)
    meta = np.asarray(metas, dtype=float).reshape(len(labels), -1)[:, 0]
    out = {"spearman_meta_vs_label": float(spearmanr(meta, labels).statistic)}
    if preds is not None:
        preds = np.asarray(preds, dtype=float)
        out["spearman_pred_vs_meta"] = float(spearmanr(preds, meta).statistic)
        out["spearman_pred_vs_label"] = float(spearmanr(preds, labels).statistic)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Check 2: image sensitivity
# ──────────────────────────────────────────────────────────────────────────

def pick_most_different(images, n_probe=64):
    """Indices of the two most pixel-different images among the first n_probe."""
    sub = images[:n_probe].reshape(min(len(images), n_probe), -1).float()
    d = torch.cdist(sub, sub)
    i, j = divmod(d.argmax().item(), d.shape[1])
    return i, j


@torch.no_grad()
def image_sensitivity(model, images, metas, transform, device, n_probe=64):
    """Forward the two most different images with IDENTICAL metadata.

    Returns |pred_i - pred_j| alongside a same-image-twice determinism baseline.
    A dead image pathway gives cross_image_diff == 0 (== repeat_diff).
    """
    i, j = pick_most_different(images, n_probe)
    meta = metas[i:i + 1]  # same covariate row for both forwards
    x = transform(torch.stack([images[i], images[j], images[i]])).to(device)
    m = meta.repeat(3, 1).to(device)
    p = model(x, metadata=m).view(-1).float().cpu()
    return {
        "idx_pair": (i, j),
        "cross_image_diff": float((p[0] - p[1]).abs()),
        "repeat_diff": float((p[0] - p[2]).abs()),  # ~0: determinism baseline
    }


# ──────────────────────────────────────────────────────────────────────────
# Check 3: weight autopsy
# ──────────────────────────────────────────────────────────────────────────

def head_weight_report(head_state, meta_dim=1):
    """Compare image-slice vs meta-slice magnitudes of final_head.weight."""
    w = head_state["final_head.weight"].float().view(-1)
    if meta_dim > 0:
        img_w, meta_w = w[:-meta_dim], w[-meta_dim:]
    else:
        img_w, meta_w = w, torch.zeros(0)
    return {
        "image_weight_abs_mean": float(img_w.abs().mean()),
        "image_weight_abs_max": float(img_w.abs().max()),
        "meta_weight_abs_mean": float(meta_w.abs().mean()) if meta_dim > 0 else 0.0,
        "compressor_weight_abs_mean": float(
            head_state["image_compressor.0.weight"].float().abs().mean()
        ),
    }


def lora_b_report(lora_dir):
    """Norms of the saved LoRA A/B matrices. B is zero-init: still ~0 => no learning."""
    from safetensors.torch import load_file
    tensors = load_file(str(Path(lora_dir) / "adapter_model.safetensors"))
    a = [v.float().norm().item() for k, v in tensors.items() if "lora_A" in k]
    b = [v.float().norm().item() for k, v in tensors.items() if "lora_B" in k]
    if not b:
        raise ValueError(f"No lora_B tensors found in {lora_dir}")
    return {
        "n_lora_modules": len(b),
        "lora_B_norm_mean": float(np.mean(b)),
        "lora_B_norm_max": float(np.max(b)),
        "lora_A_norm_mean": float(np.mean(a)) if a else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────
# Interpretation + CLI
# ──────────────────────────────────────────────────────────────────────────

def interpret(sp, sens, head, lora):
    verdicts = []
    if sp and "spearman_pred_vs_meta" in sp:
        if abs(sp["spearman_pred_vs_meta"]) > 0.95:
            verdicts.append("COLLAPSED: predictions are a monotone function of dist_to_center.")
        else:
            verdicts.append("Predictions are NOT purely the covariate — collapse hypothesis weakened.")
    if sens:
        if sens["cross_image_diff"] < 10 * max(sens["repeat_diff"], 1e-6):
            verdicts.append("DEAD IMAGE PATH: different images give (near-)identical scores.")
        else:
            verdicts.append("Image pathway responds to input changes.")
    if head:
        if head["meta_weight_abs_mean"] > 10 * head["image_weight_abs_mean"]:
            verdicts.append("HEAD AUTOPSY: meta weight dominates the image weights >10x.")
    if lora:
        if lora["lora_B_norm_max"] < 1e-4:
            verdicts.append("LoRA NEVER TRAINED: all lora_B matrices are still at zero-init.")
        else:
            verdicts.append("LoRA B matrices moved off zero-init (backbone received gradient).")
    return verdicts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", nargs="+", default=[], help="val shard .pt file(s)")
    ap.add_argument("--checkpoint", default=None, help="head .pth (or _last.pth) checkpoint")
    ap.add_argument("--lora-dir", default=None, help="saved LoRA adapter dir")
    ap.add_argument("--meta-dim", type=int, default=1, help="meta_dim the checkpoint was trained with")
    ap.add_argument("--nbands", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-images", type=int, default=512, help="cap for the prediction pass")
    args = ap.parse_args()

    sp, sens, head_rep, lora_rep = {}, {}, {}, {}
    data = load_shard_tensors(args.shard) if args.shard else None
    transform = build_eval_transform(args.nbands)

    if args.checkpoint:
        # Check 3a runs from the raw state dict, before the heavy model build
        head_state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(head_state, dict) and "model_state_dict" in head_state:
            head_state = {k.removeprefix("head."): v for k, v in head_state["model_state_dict"].items()
                          if k.startswith("head.")}
        head_rep = head_weight_report(head_state, meta_dim=args.meta_dim)

    if args.lora_dir:
        lora_rep = lora_b_report(args.lora_dir)

    if data is not None:
        preds = None
        if args.checkpoint:
            model = load_model(args.checkpoint, args.lora_dir, meta_dim=args.meta_dim,
                               nbands=args.nbands, device=args.device)
            n = min(len(data["images"]), args.max_images)
            preds = predict(model, data["images"][:n], data["metas"][:n], transform, args.device)
            sp = spearman_report(data["scores"][:n], data["metas"][:n], preds)
            sens = image_sensitivity(model, data["images"], data["metas"], transform, args.device)
        else:
            sp = spearman_report(data["scores"], data["metas"])

    print("\n=== diagnose_collapse report ===")
    for name, rep in [("check1_spearman", sp), ("check2_sensitivity", sens),
                      ("check3a_head_weights", head_rep), ("check3b_lora", lora_rep)]:
        if rep:
            print(f"\n[{name}]")
            for k, v in rep.items():
                print(f"  {k}: {v}")
    print("\n[verdicts]")
    for v in interpret(sp, sens, head_rep, lora_rep):
        print(f"  - {v}")
    if not any([sp, sens, head_rep, lora_rep]):
        print("  - nothing to check: pass --shard and/or --checkpoint")


if __name__ == "__main__":
    main()
