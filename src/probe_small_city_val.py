"""Diagnostic probe: why is val_cities/bracket/small so bad? (issue: small-city debug)

Runs the run_20260721 best checkpoint (CPU, thread-capped — safe next to a live
GPU training run) over the small-bracket rows of the cached val_cities shards
(Lancaster 29540, Modesto 33700), then prints:
  * label-integrity checks (geoid hash + Rel_Score vs the source label panel),
  * per-(city, year) Spearman cells with label/pred spread,
  * per-building cross-year means — the actual 20 buildings the bracket
    metric stands on.

Writes predictions to ~/outputs/small_city_val_preds.parquet.
Usage: python src/probe_small_city_val.py
"""
import sys, zlib
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)
PROJ = "/mnt/c/Working Papers/NY State Aerial Imagery Prototype/ny_state_aerial_imagery_prototype"
sys.path.insert(0, PROJ)

TARGET_CBSAS = {29540: "Lancaster", 33700: "Modesto"}

# ── data: collect target rows from val shards ────────────────────────────────
rows, imgs = [], []
for i in range(5):
    d = torch.load(f"/home/abbatenicolas/data/cache/val_cities_cache/shard_{i}.pt",
                   weights_only=False, map_location="cpu")
    cbsa = np.asarray(d["cbsa_ids"]).ravel()
    idxs = np.where(np.isin(cbsa, list(TARGET_CBSAS)))[0]
    for j in idxs:
        rows.append({
            "cbsa": int(cbsa[j]),
            "year": int(np.asarray(d["years"]).ravel()[j]),
            "building_id": int(np.asarray(d["building_ids"]).ravel()[j]),
            "label": float(np.asarray(d["scores"]).ravel()[j]),
            "geoid_hash": int(np.asarray(d["geoids"]).ravel()[j]),
            "change": int(np.asarray(d["structural_change"]).ravel()[j]),
        })
        imgs.append(d["images"][j].clone())
    del d
meta = pd.DataFrame(rows)
print(f"selected {len(meta)} rows: ", meta.groupby("cbsa").size().to_dict(), flush=True)

# ── label integrity check (no model needed) ──────────────────────────────────
bld = pd.read_parquet(PROJ + "/data/processed/pair_buildings_ms_us_epsg5070_all.parquet",
                      columns=["building_id", "GEOID", "cbsa_code"])
bmap = bld.drop_duplicates("building_id").set_index("building_id")
meta["GEOID"] = meta.building_id.map(bmap["GEOID"])
meta["hash_check"] = [zlib.crc32(str(g).encode()) % (2**31) == h
                      for g, h in zip(meta.GEOID, meta.geoid_hash)]
print("geoid hash matches:", meta.hash_check.mean(), flush=True)

lab = pd.read_parquet(PROJ + "/data/processed/pair_labels_ms_us_W2_r5_years2010-2024n15_all.parquet")
lab = lab.set_index(["GEOID", "year"])
meta["label_src"] = [
    lab["Rel_Score"].get((g, y), np.nan) for g, y in zip(meta.GEOID, meta.year)
]
meta["label_match"] = np.isclose(meta.label, meta.label_src, atol=1e-4)
print("label matches source panel:", meta.label_match.mean(),
      "| n mismatched:", int((~meta.label_match).sum()), flush=True)

# ── model ────────────────────────────────────────────────────────────────────
from src.custom_models import get_model  # noqa: E402
from torchvision.transforms import v2 as T  # noqa: E402
from peft import PeftModel  # noqa: E402

model = get_model("scalemae", image_size=224, bands=3, kind="reg", meta_dim=0)
ckpt_dir = PROJ + "/models/models_by_epoch/run_20260721"
model.head.load_state_dict(torch.load(f"{ckpt_dir}/run_20260721_best.pth",
                                      map_location="cpu", weights_only=True))
model.backbone = PeftModel.from_pretrained(model.backbone.base_model.model,
                                           f"{ckpt_dir}/run_20260721_best_lora")
model.eval()
print("model ready", flush=True)

tf = T.Compose([T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

preds = []
with torch.no_grad():
    for s in range(0, len(imgs), 8):
        batch = torch.stack([tf(im) for im in imgs[s:s + 8]])
        preds.append(model(batch).squeeze(-1).numpy())
        if (s // 8) % 5 == 0:
            print(f"  {s + len(batch)}/{len(imgs)}", flush=True)
meta["pred"] = np.concatenate(preds)
meta.to_parquet("/home/abbatenicolas/outputs/small_city_val_preds.parquet")

# ── metrics ──────────────────────────────────────────────────────────────────
from scipy.stats import spearmanr  # noqa: E402

print("\nper-(city, year) cells (dedup building, mean pred per building):")
for (c, y), g in meta.groupby(["cbsa", "year"]):
    u = g.groupby("building_id").agg(pred=("pred", "mean"), label=("label", "first"))
    rho = spearmanr(u.pred, u.label)[0] if len(u) >= 5 else np.nan
    print(f"  {TARGET_CBSAS[c]:9s} {y}: n_bld={len(u):3d} rho={rho:+.3f} "
          f"label_sd={u.label.std():.2f} pred_sd={u.pred.std():.2f}")

print("\nper-city pooled (building-year level):")
for c, g in meta.groupby("cbsa"):
    print(f"  {TARGET_CBSAS[c]:9s} pooled rho={spearmanr(g.pred, g.label)[0]:+.3f} n={len(g)}")

print("\nper-building means (cross-year avg pred vs avg label):")
for c, g in meta.groupby("cbsa"):
    u = g.groupby("building_id").agg(pred=("pred", "mean"), label=("label", "mean"),
                                     GEOID=("GEOID", "first"))
    u = u.sort_values("label")
    print(f"-- {TARGET_CBSAS[c]}  (building-level rho={spearmanr(u.pred, u.label)[0]:+.3f})")
    print(u.round(3).to_string())
