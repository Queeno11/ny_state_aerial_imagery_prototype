"""Offline small-city evaluator (Tier 0 of the small-bracket val fix).

Background: the training-time ``val_cities`` cache truncates to ~272 buildings,
leaving the small bracket with 11 (Lancaster) + 9 (Modesto) buildings — see
``src/probe_small_city_val.py``. This script builds an HONEST sample offline —
1 building for EVERY tract of the small val cities, a few NAIP flight years —
fetches the crops itself (same geometry as ``CyclicCacheManager``: tau from
``calculate_exact_tau``, 224 px, EPSG:5070 centroids), scores a checkpoint on
CPU (thread-capped; safe next to a live GPU run), and reports the same
tract-weighted within-city Spearman the training loop logs.

It never touches the training process, its caches, or the model weights.

Usage (WSL, torch_geo_env):
    python src/eval_small_city_offline.py

Outputs:
    ~/outputs/offline_small_city_eval.parquet   (per-row preds)
    stdout report: per-(city, year) cells + tract-weighted bracket number,
    with the cached-val 20-building number alongside for contrast.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.utils.metrics import weighted_within_spearman, within_city_cells  # noqa: E402

# ── run/eval configuration ───────────────────────────────────────────────────
SAVENAME = "run_20260721"
CKPT_DIR = PROJ / "models" / "models_by_epoch" / SAVENAME
OUT_PATH = Path.home() / "outputs" / "offline_small_city_eval.parquet"
CACHED_VAL_PREDS = Path.home() / "outputs" / "small_city_val_preds.parquet"

# Known NAIP flight years per city (observed in the cached val shards).
CBSA_YEARS = {
    29540: [2017, 2019, 2022],  # Lancaster, PA
    33700: [2018, 2020, 2022],  # Modesto, CA
}
CBSA_NAMES = {29540: "Lancaster", 33700: "Modesto"}

TAU_METERS_REQUESTED = 100
IMAGE_SIZE = 224
NBANDS = 3
SEED = 825
FETCH_WORKERS = 12
TORCH_THREADS = 4


# ── pure logic (unit-tested in src/tests/test_eval_small_city_offline.py) ────
def build_eval_frame(buildings: pd.DataFrame, cbsa_years: dict[int, list[int]],
                     n_tracts_per_cbsa: int | None = None,
                     seed: int = SEED) -> pd.DataFrame:
    """One building per tract for each target CBSA, crossed with its years.

    ``buildings`` needs columns building_id, GEOID, cbsa_code, centroid_x,
    centroid_y (cbsa_code coercible to int). Building choice within a tract is
    deterministic under ``seed``; ``n_tracts_per_cbsa`` (None = all) subsamples
    tracts, also deterministically. Returns one row per (building, year) with
    columns [building_id, GEOID, cbsa, year, centroid_x, centroid_y].
    """
    df = buildings.copy()
    df["cbsa"] = pd.to_numeric(df["cbsa_code"], errors="coerce")
    frames = []
    for cbsa, years in cbsa_years.items():
        city = df[df["cbsa"] == cbsa]
        if city.empty:
            continue
        picks = (
            city.sort_values("building_id")
            .groupby("GEOID", observed=True, group_keys=False)
            [["building_id", "GEOID", "centroid_x", "centroid_y"]]
            .apply(lambda g: g.sample(n=1, random_state=seed))
        )
        if n_tracts_per_cbsa is not None and len(picks) > n_tracts_per_cbsa:
            picks = picks.sample(n=n_tracts_per_cbsa, random_state=seed)
        picks = picks.assign(cbsa=cbsa)
        frames.append(picks.merge(pd.DataFrame({"year": years}), how="cross"))
    if not frames:
        return pd.DataFrame(columns=["building_id", "GEOID", "cbsa", "year",
                                     "centroid_x", "centroid_y"])
    return pd.concat(frames, ignore_index=True)


def label_for_effective_year(geoid: str, actual_year: int,
                             label_lookup: dict, panel_years: list[int]):
    """Label for the year the imagery actually comes from.

    Mirrors ``CyclicCacheManager._worker_generate``: exact (GEOID, year) hit,
    else the NEAREST panel year (never another year's label for substituted
    imagery), else None. Returns (label, label_year) or (None, None).
    """
    key = (geoid, int(actual_year))
    val = label_lookup.get(key)
    if val is not None and not pd.isna(val):
        return float(val), int(actual_year)
    nearest = min(panel_years, key=lambda y: abs(y - int(actual_year)))
    val = label_lookup.get((geoid, nearest))
    if val is not None and not pd.isna(val):
        return float(val), nearest
    return None, None


def dedupe_effective_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate (building_id, eff_year) rows (flight-year substitution
    can map two requested years onto the same flight)."""
    return df.drop_duplicates(subset=["building_id", "eff_year"]).reset_index(drop=True)


def compute_report(df: pd.DataFrame) -> dict:
    """Per-city cells + tract-weighted within-Spearman, per city and pooled.

    ``df`` needs columns cbsa, year, pred, label, building_id (one row per
    building-year). Reuses the canonical metric functions so the number is
    directly comparable to ``val_cities/bracket/small/within_spearman``.
    """
    out = {"cells": within_city_cells(df)}
    out["bracket"] = weighted_within_spearman(out["cells"])
    out["per_city"] = {
        int(cbsa): weighted_within_spearman(cells_c)
        for cbsa, cells_c in (
            (c, out["cells"][out["cells"]["cbsa"] == c]) for c in df["cbsa"].unique()
        )
    }
    return out


# ── I/O + network + model (thin wrappers, not unit-tested) ───────────────────
def load_inputs(target_cbsas=tuple(CBSA_YEARS)):
    # The buildings parquet is the full US (~72M rows, 1.5 GB); push the CBSA
    # filter into the read so only the target cities load — a full read OOMs
    # next to a live GPU training run. cbsa_code is dictionary<string>.
    import pyarrow.dataset as ds

    codes = [str(c) for c in target_cbsas]
    dset = ds.dataset(PROJ / "data/processed/pair_buildings_ms_us_epsg5070_all.parquet")
    buildings = dset.to_table(
        columns=["building_id", "GEOID", "cbsa_code", "centroid_x", "centroid_y"],
        filter=ds.field("cbsa_code").isin(codes),
    ).to_pandas()
    labels = pd.read_parquet(
        PROJ / "data/processed/pair_labels_ms_us_W2_r5_years2010-2024n15_all.parquet",
        columns=["GEOID", "year", "Rel_Score"],
    )
    label_lookup = {(g, int(y)): s for g, y, s in
                    zip(labels["GEOID"], labels["year"], labels["Rel_Score"])}
    panel_years = sorted(labels["year"].unique().tolist())
    return buildings, label_lookup, panel_years


def fetch_eval_crops(frame: pd.DataFrame, tau_meters: float):
    """Fetch one NAIP crop per row. Returns (images, eff_years, failures) lists
    aligned to ``frame`` rows; failed rows have image None."""
    from concurrent.futures import ThreadPoolExecutor

    from pyproj import Transformer

    from src import geo_utils
    from src.data.naip_fetcher import TractSearchCache, fetch_naip
    from src.utils.paths import CACHE_DIR

    # Transform ALL centroids to lon/lat once, up front: a single pyproj
    # Transformer is not thread-safe, so sharing one across the fetch pool
    # corrupts every search (observed as a 100% search_error rate). The
    # transform is vectorized and cheap, so this also beats per-row calls.
    to_4326 = Transformer.from_crs(geo_utils.METRIC_CRS, "EPSG:4326", always_xy=True)
    lons, lats = to_4326.transform(frame["centroid_x"].to_numpy(),
                                   frame["centroid_y"].to_numpy())
    search_cache = TractSearchCache(cache_dir=Path(CACHE_DIR) / "naip_search_cache" / "offline_eval")
    tasks = list(zip(lons, lats, frame["year"].to_numpy(), frame["GEOID"].to_numpy()))

    def _one(task):
        lon, lat, year, geoid = task
        res = fetch_naip(
            lon=float(lon), lat=float(lat), crop_size_meters=tau_meters * 2,
            nbands=NBANDS, out_pixels=IMAGE_SIZE, year_hint=int(year),
            search_cache=search_cache, cache_key=str(geoid),
        )
        return res.crop, res.actual_year, res.failure

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        results = list(pool.map(_one, tasks))
    images = [r[0] for r in results]
    eff_years = [r[1] for r in results]
    failures = [r[2] for r in results]
    return images, eff_years, failures


def load_run_model():
    import torch

    from peft import PeftModel

    from src.custom_models import get_model

    torch.set_num_threads(TORCH_THREADS)
    model = get_model("scalemae", image_size=IMAGE_SIZE, bands=NBANDS,
                      kind="reg", meta_dim=0)
    model.head.load_state_dict(torch.load(
        CKPT_DIR / f"{SAVENAME}_best.pth", map_location="cpu", weights_only=True))
    model.backbone = PeftModel.from_pretrained(
        model.backbone.base_model.model, CKPT_DIR / f"{SAVENAME}_best_lora")
    model.eval()
    return model


def run_inference(model, images: list[np.ndarray], batch_size: int = 8) -> np.ndarray:
    import torch
    from torchvision.transforms import v2 as T

    tf = T.Compose([T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
    preds = []
    with torch.no_grad():
        for s in range(0, len(images), batch_size):
            batch = torch.stack([tf(torch.from_numpy(np.ascontiguousarray(im)))
                                 for im in images[s:s + batch_size]])
            preds.append(model(batch).squeeze(-1).numpy())
            done = s + len(batch)
            if (s // batch_size) % 10 == 0:
                print(f"  inference {done}/{len(images)}", flush=True)
    return np.concatenate(preds) if preds else np.array([])


def main():
    from src import geo_utils

    tau_meters, step = geo_utils.calculate_exact_tau(TAU_METERS_REQUESTED, IMAGE_SIZE)
    print(f"tau={tau_meters:.2f} m (step {step}); crop={tau_meters * 2:.0f} m -> {IMAGE_SIZE}px", flush=True)

    buildings, label_lookup, panel_years = load_inputs()
    frame = build_eval_frame(buildings, CBSA_YEARS)
    print(f"eval frame: {len(frame)} rows, "
          f"{frame.groupby('cbsa')['GEOID'].nunique().to_dict()} tracts/city", flush=True)

    print("fetching NAIP crops...", flush=True)
    images, eff_years, failures = fetch_eval_crops(frame, tau_meters)
    frame = frame.assign(eff_year=eff_years, failure=failures)
    keep = np.array([im is not None for im in images])
    from collections import Counter
    print(f"fetched {int(keep.sum())}/{len(frame)}; failures: "
          f"{Counter(f for f in failures if f)}", flush=True)
    frame = frame[keep].reset_index(drop=True)
    images = [im for im in images if im is not None]

    # Relabel to the effective flight year (worker semantics), then dedupe.
    lab = [label_for_effective_year(g, y, label_lookup, panel_years)
           for g, y in zip(frame["GEOID"], frame["eff_year"])]
    frame = frame.assign(label=[l for l, _ in lab], label_year=[y for _, y in lab])
    ok = frame["label"].notna().to_numpy()
    frame, images = frame[ok].reset_index(drop=True), [im for im, k in zip(images, ok) if k]
    before = len(frame)
    frame["_row"] = range(len(frame))
    frame = dedupe_effective_rows(frame)
    images = [images[i] for i in frame["_row"]]
    frame = frame.drop(columns="_row")
    print(f"labeled rows: {before}, after (building, eff_year) dedupe: {len(frame)}", flush=True)

    model = load_run_model()
    print("model ready", flush=True)
    frame["pred"] = run_inference(model, images)
    frame["year"] = frame["eff_year"]  # metric cells key on the effective year
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.drop(columns=["centroid_x", "centroid_y"]).to_parquet(OUT_PATH)
    print(f"saved {OUT_PATH}", flush=True)

    rep = compute_report(frame)
    print("\nper-(city, year) cells:")
    for r in rep["cells"].itertuples(index=False):
        print(f"  {CBSA_NAMES.get(int(r.cbsa), r.cbsa):9s} {int(r.year)}: "
              f"n={r.n:3d} tracts={r.n_tracts:3d} rho={r.rho:+.3f}")
    print("\ntract-weighted within-Spearman (canonical metric):")
    for cbsa, m in rep["per_city"].items():
        print(f"  {CBSA_NAMES.get(cbsa, cbsa):9s}: {m.get('within_spearman', float('nan')):+.3f} "
              f"({m.get('within_tracts', 0)} tract-cells)")
    print(f"  SMALL BRACKET (honest, {rep['bracket'].get('within_tracts', 0)} tract-cells): "
          f"{rep['bracket'].get('within_spearman', float('nan')):+.3f}")

    if CACHED_VAL_PREDS.exists():
        cached = pd.read_parquet(CACHED_VAL_PREDS)
        cached_rep = compute_report(cached.rename(columns={"label_src": "_ls"}))
        print(f"  cached-val 20-building sample, same checkpoint: "
              f"{cached_rep['bracket'].get('within_spearman', float('nan')):+.3f}")


if __name__ == "__main__":
    main()
