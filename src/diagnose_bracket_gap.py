"""Why do the medium/large val brackets underperform mega/small?

The logged ``bracket/{name}/spearman`` POOLS every building of every city in the
bracket (compute_val_metrics), so it mixes two different things:
  (a) within-city ranking quality — what the model is actually trained for, and
  (b) cross-city score-level alignment — unconstrained by the loss (single-CBSA
      batches, per-CBSA z labels), so city offsets scramble pooled ranks.
Brackets differ hugely in how much (b) they contain: val_cities mega is a single
city (Chicago, no cross-city pairs at all) while large pools 13 cities and
medium 5 — including McAllen, whose income level (and hence predicted level)
is a national outlier.

Hypotheses this script quantifies, per val set and bracket:
  H1 pooling artifact  rho_pooled vs rho_within (n-weighted per-city Spearman)
                       vs rho_demeaned (pooled after removing per-city offsets).
                       H1 explains the gap iff rho_within/rho_demeaned are flat
                       across brackets while rho_pooled is not.
  H2 label noise       per-city rho vs mean ACS relative SE of the z-label
                       (Rel_SE^2 from the metros panel): noisier labels cap the
                       attainable Spearman.
  H3 city covariates   per-city rho vs population, n_tracts, mean log-PCI,
                       val label spread, |pred offset|, val n.

Runs in two modes:
  inference (GPU box):  loads val shard caches + the run's best checkpoint,
      predicts, dumps a predictions CSV, then analyzes:
        python src/diagnose_bracket_gap.py \
            --val-cache ~/data/cache/val_cities_cache ~/data/cache/val_temporal_cache \
            --checkpoint-dir models/models_by_epoch/run_20260710 --savename run_20260710 \
            --preds-out results/run_20260710/val_bracket_preds.csv
  analysis-only (no torch/GPU):  re-analyze a dumped CSV:
        python src/diagnose_bracket_gap.py --from-preds results/run_20260710/val_bracket_preds.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.diagnose_city_coding import (
    PRED_COL, LABEL_COL, CITY_COL,
    _spearman, per_city_stats, weighted_within_spearman, demeaned_pooled_spearman,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = PROJECT_ROOT / "data/processed/cbsa_splits.feather"
DEFAULT_PANEL = PROJECT_ROOT / "data/processed/us_metros_panel_2011_2023.feather"
BRACKET_ORDER = ["small", "medium", "large", "mega"]

# columns a predictions frame must carry ("set" tags val_cities / val_temporal)
REQUIRED = [PRED_COL, LABEL_COL, CITY_COL, "set"]


# ── inference mode: shards -> predictions ────────────────────────────────────

def load_val_shards(cache_dirs) -> tuple[pd.DataFrame, "torch.Tensor"]:
    """Concatenate every shard_*.pt in the given cache dirs.

    Returns (meta_df, images): meta_df carries Rel_Score/cbsa_code/etc. named
    like the prediction CSVs so the analysis functions are shared; the "set"
    column is the cache dir's name. Rows with cbsa_id == 0 (pre-US shards) are
    dropped with a warning — they cannot be attributed to a city.
    """
    import torch

    metas, images = [], []
    for cache_dir in cache_dirs:
        cache_dir = Path(cache_dir).expanduser()
        shard_paths = sorted(cache_dir.glob("shard_*.pt"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard_*.pt in {cache_dir}")
        for sp in shard_paths:
            d = torch.load(sp, map_location="cpu", weights_only=False)
            metas.append(pd.DataFrame({
                LABEL_COL: d["scores"].numpy(),
                CITY_COL: d["cbsa_ids"].numpy(),
                "year": d["years"].numpy(),
                "building_id": d["building_ids"].numpy(),
                "GEOID": d["geoids"].numpy(),
                "change": d["structural_change"].numpy(),
                "meta": d["metas"].squeeze(-1).numpy(),
                "set": cache_dir.name,
            }))
            images.append(d["images"])
    df = pd.concat(metas, ignore_index=True)
    imgs = torch.cat(images)
    n_zero = int((df[CITY_COL] == 0).sum())
    if n_zero:
        print(f"⚠️ {n_zero}/{len(df)} rows have cbsa_id == 0 (pre-US shards) — dropped.")
        keep = df[CITY_COL] != 0
        imgs = imgs[torch.as_tensor(keep.to_numpy())]
        df = df[keep].reset_index(drop=True)
    return df, imgs


def build_eval_transform(nbands: int):
    """The exact transform generate_predictions uses (must match training)."""
    import torch
    from torchvision.transforms import v2 as transforms

    mean = [0.485, 0.456, 0.406] + [0.5] * max(0, nbands - 3)
    std = [0.229, 0.224, 0.225] + [0.5] * max(0, nbands - 3)
    return transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_model(checkpoint_dir, savename, which="best", model_name="scalemae",
               image_size=224, nbands=3, meta_dim=0, device="cuda"):
    """Rebuild the model exactly like generate_predictions: head + LoRA adapter."""
    import torch
    from src.main import set_model_and_loss_function

    model, _ = set_model_and_loss_function(
        model_name=model_name, kind="reg", image_size=image_size,
        bands=nbands, weights=None, meta_dim=meta_dim,
    )
    ckpt = Path(checkpoint_dir) / f"{savename}_{which}.pth"
    if which == "best":
        model.head.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        lora_dir = ckpt.parent / f"{ckpt.stem}_lora"
        if lora_dir.exists():
            from peft import PeftModel
            model.backbone = PeftModel.from_pretrained(model.backbone.base_model.model, str(lora_dir))
            print(f"✅ LoRA adapter loaded from {lora_dir}")
    else:  # last: full state dict
        checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def predict_scores(model, images, meta_values, device="cuda", batch_size=64,
                   transform=None) -> np.ndarray:
    """Batched forward pass; order-preserving. images uint8 [N,C,H,W]."""
    import torch

    if transform is None:
        transform = build_eval_transform(images.shape[1])
    meta_t = torch.as_tensor(np.asarray(meta_values), dtype=torch.float32).reshape(len(images), -1)
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = transform(images[i:i + batch_size]).to(device)
            metas = meta_t[i:i + batch_size].to(device)
            out = model(batch, metadata=metas)
            preds.append(out.view(-1).float().cpu().numpy())
    return np.concatenate(preds)


# ── analysis ─────────────────────────────────────────────────────────────────

def bracket_decomposition(df: pd.DataFrame, bracket_of: dict, min_n: int = 20) -> pd.DataFrame:
    """Per (set, bracket): pooled vs within-city vs demeaned Spearman (H1)."""
    df = df.copy()
    df["bracket"] = df[CITY_COL].astype(int).map(bracket_of)
    rows = []
    for (val_set, bracket), grp in df.groupby(["set", "bracket"]):
        stats = per_city_stats(grp, min_n=min_n)
        rows.append({
            "set": val_set, "bracket": bracket,
            "n": len(grp), "n_cities": grp[CITY_COL].nunique(),
            "rho_pooled": _spearman(grp[PRED_COL], grp[LABEL_COL]),
            "rho_within": weighted_within_spearman(stats),
            "rho_demeaned": demeaned_pooled_spearman(grp),
            "offset_spread": float(stats["pred_mean"].std(ddof=0)) if len(stats) > 1 else 0.0,
        })
    out = pd.DataFrame(rows)
    out["pooling_gap"] = out["rho_within"] - out["rho_pooled"]
    out["bracket"] = pd.Categorical(out["bracket"], BRACKET_ORDER, ordered=True)
    return out.sort_values(["set", "bracket"]).reset_index(drop=True)


def city_covariates(panel_path=DEFAULT_PANEL, se_prefix="Rel_SE_W2_i_r5pct") -> pd.DataFrame:
    """Per-CBSA label-noise and income-level covariates from the metros panel."""
    panel = pd.read_feather(panel_path)
    se_cols = [c for c in panel.columns if c.startswith(se_prefix)]
    pci_cols = [c for c in panel.columns if c.startswith("Log_PCI_")]
    if not se_cols:
        raise ValueError(f"No columns starting with '{se_prefix}' in {panel_path}")
    g = panel.groupby(panel["cbsa_code"].astype(int))
    out = pd.DataFrame({
        # mean squared relative SE of the z-label: the noise share that caps rho
        "label_noise_se2": g[se_cols].mean().pow(2).mean(axis=1),
        "mean_log_pci": g[pci_cols].mean().mean(axis=1),
    })
    return out.reset_index().rename(columns={"cbsa_code": CITY_COL})


def build_city_table(df: pd.DataFrame, cbsa_meta: pd.DataFrame,
                     covariates: pd.DataFrame | None = None, min_n: int = 20) -> pd.DataFrame:
    """One row per (set, city): rho + everything we might explain rho with."""
    rows = []
    for val_set, grp in df.groupby("set"):
        stats = per_city_stats(grp, min_n=min_n)
        stats["label_std"] = stats[CITY_COL].map(
            grp.groupby(CITY_COL)[LABEL_COL].std().to_dict())
        stats["set"] = val_set
        rows.append(stats)
    out = pd.concat(rows, ignore_index=True)
    out[CITY_COL] = out[CITY_COL].astype(int)

    meta = cbsa_meta.copy()
    meta[CITY_COL] = meta["cbsa_code"].astype(int)
    out = out.merge(meta[[CITY_COL, "cbsa_title", "bracket", "population", "n_tracts"]],
                    on=CITY_COL, how="left")
    if covariates is not None:
        out = out.merge(covariates, on=CITY_COL, how="left")
    out["abs_offset"] = out["pred_mean"].abs()
    return out


COVARIATES = ["label_noise_se2", "mean_log_pci", "population", "n_tracts",
              "label_std", "abs_offset", "n"]


def covariate_correlations(city_df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of per-city rho with each covariate (H2/H3)."""
    from scipy.stats import spearmanr

    sub = city_df.dropna(subset=["rho"])
    rows = []
    for cov in COVARIATES:
        if cov not in sub.columns:
            continue
        pair = sub[["rho", cov]].dropna()
        if len(pair) < 5 or pair[cov].nunique() < 2:
            continue
        r, p = spearmanr(pair["rho"], pair[cov])
        rows.append({"covariate": cov, "spearman_r": float(r), "p_value": float(p),
                     "n_cities": len(pair)})
    if not rows:
        return pd.DataFrame(columns=["covariate", "spearman_r", "p_value", "n_cities"])
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def format_report(bracket_df, city_df, corr_df) -> str:
    lines = ["═" * 96,
             "BRACKET GAP DECOMPOSITION — pooled (as logged) vs within-city vs city-demeaned Spearman",
             "═" * 96]
    with pd.option_context("display.width", 200):
        lines.append(bracket_df.round(3).to_string(index=False))
    lines += [
        "",
        "H1 (pooling artifact) holds where pooling_gap is large and rho_within is flat across brackets.",
        "Residual within-city differences (rho_within still low for a bracket) are real and need H2/H3.",
        "─" * 96,
        "per-city (worst within-city rho first):",
    ]
    show_cols = ["set", "cbsa_title", "bracket", "n", "n_bld", "rho", "rho_bld",
                 "pred_mean", "label_std", "label_noise_se2", "mean_log_pci"]
    show = city_df.dropna(subset=["rho"]).sort_values("rho")
    with pd.option_context("display.width", 200):
        lines.append(show[[c for c in show_cols if c in show.columns]].round(3).to_string(index=False))
    lines += ["─" * 96, "correlates of per-city rho (H2: label_noise_se2; H3: the rest):"]
    lines.append(corr_df.round(4).to_string(index=False) if len(corr_df) else "  (too few cities)")
    return "\n".join(lines)


# ── entry point ──────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from-preds", default=None,
                    help="analysis-only: CSV with predicted_value, Rel_Score, cbsa_code, set")
    ap.add_argument("--val-cache", nargs="+", default=None,
                    help="val shard cache dir(s), e.g. ~/data/cache/val_cities_cache ...")
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--savename", default=None)
    ap.add_argument("--which", choices=["best", "last"], default="best")
    ap.add_argument("--nbands", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=224)
    # training builds the model with meta_dim=0 (dist_to_center disabled, main.py
    # set_model_and_loss_function call) — the head ignores metadata entirely
    ap.add_argument("--meta-dim", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--preds-out", default=None, help="dump the predictions CSV here")
    ap.add_argument("--splits-feather", default=str(DEFAULT_SPLITS))
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--se-prefix", default="Rel_SE_W2_i_r5pct")
    ap.add_argument("--min-n", type=int, default=20)
    args = ap.parse_args(argv)

    if args.from_preds:
        df = pd.read_csv(args.from_preds)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{args.from_preds} is missing columns: {missing}")
    else:
        if not (args.val_cache and args.checkpoint_dir and args.savename):
            ap.error("need either --from-preds or all of --val-cache/--checkpoint-dir/--savename")
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        df, images = load_val_shards(args.val_cache)
        model = load_model(args.checkpoint_dir, args.savename, which=args.which,
                           image_size=args.image_size, nbands=args.nbands,
                           meta_dim=args.meta_dim, device=device)
        df[PRED_COL] = predict_scores(model, images, df["meta"].values,
                                      device=device, batch_size=args.batch_size)
        if args.preds_out:
            out = Path(args.preds_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.drop(columns=["meta"]).to_csv(out, index=False)
            print(f"💾 predictions -> {out}")

    cbsa_meta = pd.read_feather(args.splits_feather)
    bracket_of = {int(c): b for c, b in zip(cbsa_meta["cbsa_code"], cbsa_meta["bracket"])}
    try:
        covs = city_covariates(args.panel, se_prefix=args.se_prefix)
    except (FileNotFoundError, ValueError) as e:
        print(f"⚠️ panel covariates unavailable ({e}); H2 columns will be empty.")
        covs = None

    bracket_df = bracket_decomposition(df, bracket_of, min_n=args.min_n)
    city_df = build_city_table(df, cbsa_meta, covariates=covs, min_n=args.min_n)
    corr_df = covariate_correlations(city_df)
    print(format_report(bracket_df, city_df, corr_df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
