"""Score a cached validation set with a checkpoint and break the within-city
Spearman down PER CITY (not just per bracket).

Why: the per-bracket val metric pools whole cities, and the thin brackets pool
very few of them (the small bracket is exactly two: Lancaster PA + Modesto CA).
A bracket number can therefore be dragged by one idiosyncratic city and read as
a systematic size effect. This prints the per-city distribution so between-city
spread can be compared against the bracket gap.

CPU-only and thread-capped, so it is safe to run alongside a live GPU training
run. Reads the STATIC val shard cache (no NAIP fetches, no network).

Usage:
    python src/probe_val_by_city.py [run_id] [val_set]
    python src/probe_val_by_city.py run_20260722 val_cities
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

TORCH_THREADS = 4
BATCH = 8


def load_val_cache(val_set: str) -> tuple[pd.DataFrame, list]:
    """Metadata frame + image list for a cached val set (shard order)."""
    rows, imgs = [], []
    paths = sorted(glob.glob(f"/home/abbatenicolas/data/cache/{val_set}_cache/shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no shards for {val_set}")
    for p in paths:
        d = torch.load(p, weights_only=False, map_location="cpu")
        n = len(d["scores"])
        rows.append(pd.DataFrame({
            "cbsa": np.asarray(d["cbsa_ids"]).ravel(),
            "year": np.asarray(d["years"]).ravel(),
            "building_id": np.asarray(d["building_ids"]).ravel(),
            "label": np.asarray(d["scores"], dtype=float).ravel(),
            "change": np.asarray(d["structural_change"]).ravel(),
        }))
        imgs.extend(d["images"][i].clone() for i in range(n))
        del d
    return pd.concat(rows, ignore_index=True), imgs


def load_checkpoint(run_id: str):
    from peft import PeftModel

    from src.custom_models import get_model

    import time

    torch.set_num_threads(TORCH_THREADS)
    ckpt = PROJ / "models" / "models_by_epoch" / run_id

    # ScaleMAE.__init__ silently falls back to an untrained timm ViT-L when the
    # HF fetch fails (e.g. a transient 403 rate-limit). That fallback would still
    # accept the LoRA adapter and emit plausible-but-meaningless scores, so retry
    # the pretrained load and refuse to score with the wrong backbone.
    for attempt in range(6):
        model = get_model("scalemae", image_size=224, bands=3, kind="reg", meta_dim=0)
        inner = type(model.backbone.base_model.model).__name__  # under the PeftModel wrapper
        if "ScaleMAE" in inner:
            break
        print(f"⚠️ backbone came back as {inner!r} (HF fetch failed); retry {attempt+1}/6", flush=True)
        time.sleep(20)
    else:
        raise RuntimeError(
            "pretrained ScaleMAE backbone never loaded (kept falling back to an "
            "untrained ViT) — any metric here would be garbage. Warm the HF cache "
            "or wait out the rate-limit."
        )
    model.head.load_state_dict(torch.load(ckpt / f"{run_id}_best.pth",
                                          map_location="cpu", weights_only=True))
    model.backbone = PeftModel.from_pretrained(model.backbone.base_model.model,
                                               ckpt / f"{run_id}_best_lora")
    model.eval()
    return model


def predict(model, imgs) -> np.ndarray:
    from torchvision.transforms import v2 as T

    tf = T.Compose([T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    out = []
    with torch.no_grad():
        for s in range(0, len(imgs), BATCH):
            batch = torch.stack([tf(im) for im in imgs[s:s + BATCH]])
            out.append(model(batch).squeeze(-1).numpy())
            if (s // BATCH) % 25 == 0:
                print(f"  {s + len(batch)}/{len(imgs)}", flush=True)
    return np.concatenate(out)


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else "run_20260722"
    val_set = sys.argv[2] if len(sys.argv) > 2 else "val_cities"

    from src.utils.metrics import within_city_cells, weighted_within_spearman

    meta, imgs = load_val_cache(val_set)
    print(f"{val_set}: {len(meta)} rows, {meta.cbsa.nunique()} cities", flush=True)

    split = pd.read_feather(PROJ / "data/processed/cbsa_splits.feather")
    bracket = dict(zip(split.cbsa_code.astype(int), split.bracket))
    title = dict(zip(split.cbsa_code.astype(int), split.cbsa_title))
    ntracts = dict(zip(split.cbsa_code.astype(int), split.n_tracts))

    model = load_checkpoint(run_id)
    print("model ready", flush=True)
    meta["pred"] = predict(model, imgs)
    out = Path.home() / "outputs" / f"val_by_city_{run_id}_{val_set}.parquet"
    meta.to_parquet(out)
    print(f"saved {out}\n", flush=True)

    cells = within_city_cells(meta)
    cells["bracket"] = cells["cbsa"].astype(int).map(bracket)
    cells["city"] = cells["cbsa"].astype(int).map(title)
    per_city = (cells.groupby(["bracket", "city", "cbsa"])
                .apply(lambda g: pd.Series(weighted_within_spearman(g)))
                .reset_index())
    per_city["universe_tracts"] = per_city["cbsa"].astype(int).map(ntracts)
    print("PER-CITY within-Spearman:")
    print(per_city[["bracket", "city", "within_spearman", "within_tracts",
                    "universe_tracts", "within_cells"]]
          .sort_values(["bracket", "within_spearman"]).round(3).to_string(index=False))

    print("\nPER-BRACKET (tract-weighted over its cities):")
    for b, g in cells.groupby("bracket"):
        m = weighted_within_spearman(g)
        rhos = per_city.loc[per_city.bracket == b, "within_spearman"]
        print(f"  {b:7s} rho={m['within_spearman']:+.3f}  "
              f"n_cities={len(rhos)}  city spread=[{rhos.min():+.3f}, {rhos.max():+.3f}]  "
              f"sd={rhos.std():.3f}" if len(rhos) > 1 else
              f"  {b:7s} rho={m['within_spearman']:+.3f}  n_cities={len(rhos)}")

    all_rhos = per_city["within_spearman"]
    print(f"\nBetween-city SD across ALL {len(all_rhos)} val cities: {all_rhos.std():.3f}")
    print("A bracket built from k cities has a between-city SE of about "
          f"sd/sqrt(k) = {all_rhos.std():.3f}/sqrt(k).")


if __name__ == "__main__":
    main()
