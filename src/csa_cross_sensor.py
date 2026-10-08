"""Cross-sensor agreement check: does the model survive a camera swap? (issue #36)

**Run this before committing a machine to a multi-hour Chicago pass.** The whole
out-of-sensor design rests on an assumption that has never been tested: the model
is ScaleMAE + LoRA fine-tuned on NAIP, and Chicago's event study reads Cook
County's 6-inch municipal ortho instead. Whether predictions transfer across that
gap is an *empirical question*, not something to assert in a caption.

The measurement
---------------
Take a sample of Chicago buildings, predict each one twice for the same year —
once from NAIP, once from Cook ortho — and compare.

The headline number is the **within-tract** Spearman correlation, not the pooled
one. Pooled correlation across a whole city is dominated by between-tract
variation, which is easy: any sensor showing that downtown differs from a
residential fringe scores well. The event study lives on *within*-tract movement
over time, so within-tract agreement is what has to hold.

Reading the result
------------------
* **High within-tract rho** — the out-of-sensor claim is supported, and this
  table becomes a paper exhibit that demonstrates it rather than asserting it.
* **Low within-tract rho** — that is itself a finding (the model is sensor-bound,
  which bears on the transferability claim the paper makes) and Chicago should
  fall back to NAIP, losing the annual cadence but keeping the event study.

Either way it costs a few thousand crops instead of 2.4M.

Overlapping years only
----------------------
The comparison needs a year both sensors flew. Illinois NAIP is
2011/12/14/15/17/19/21/23, Cook ortho is annual 2009-2025, so the overlap is the
NAIP grid. A mean/SD shift between sensors is expected and harmless — the loss is
purely ordinal, so only the ranking has to agree, and the report includes the
shift so it can be seen rather than guessed at.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import csa_event_study as ces
from src.utils.paths import PROCESSED_DATA_DIR, TABLES_DIR

CROP_SIZE_M = 200.0
OUT_PIXELS = 224

# Years both sensors cover for Chicago (Illinois NAIP grid ∩ Cook ortho).
DEFAULT_OVERLAP_YEARS = (2011, 2012, 2014, 2015, 2017, 2019, 2021, 2023)

# Enough tracts for a stable within-tract rho, few enough to stay cheap:
# 250 tracts x 20 buildings x 4 years = 20k crops per sensor.
DEFAULT_N_TRACTS = 250
DEFAULT_PER_TRACT = 20

# Below this, treat the sensor swap as not demonstrated. Not a hard threshold in
# the statistics — a judgement call about what "the model reads the same
# neighbourhood" should mean, stated up front so it is not chosen after seeing
# the number.
WITHIN_RHO_FLOOR = 0.5


def sample_frame(city: str, years, *, n_tracts: int = DEFAULT_N_TRACTS,
                 per_tract: int = DEFAULT_PER_TRACT,
                 processed_dir: Path = PROCESSED_DATA_DIR,
                 seed: int = 825) -> pd.DataFrame:
    """Buildings x years to predict from both sensors.

    Reuses :class:`src.csa_predict.CSAPredictionRunner` so the sample is drawn
    exactly the way the real pass draws it — same stable hash, same per-tract
    cap semantics — rather than by a second, subtly different sampler.
    """
    from src.csa_predict import CSAPredictionRunner

    spec = ces.CSA_CITIES[city]
    runner = CSAPredictionRunner("cross_sensor", {}, cities=[city],
                                 buildings_per_tract=per_tract,
                                 processed_dir=processed_dir)
    buildings = runner.sample_buildings(spec)
    if buildings.empty:
        raise RuntimeError(f"no buildings sampled for {city}")

    keep = (buildings["GEOID"].drop_duplicates()
            .sample(n=min(n_tracts, buildings["GEOID"].nunique()),
                    random_state=seed))
    buildings = buildings[buildings["GEOID"].isin(set(keep))]
    return buildings.merge(pd.DataFrame({"year": list(years)}), how="cross")


def fetch_both(frame: pd.DataFrame, city: str, *, max_workers: int = 8,
               max_rps: float | None = None, verbose: bool = True):
    """Fetch each row's crop from NAIP and from the local ortho.

    Returns ``(naip_crops, ortho_crops)`` aligned to ``frame`` rows, with None
    where a sensor had no imagery. Both use the same 200 m / 224 px geometry, so
    the only difference between the two crops is the camera.
    """
    from concurrent.futures import ThreadPoolExecutor

    from pyproj import Transformer

    from src.data.naip_fetcher import TractSearchCache, fetch_naip
    from src.data.ortho_fetcher import make_fetch_fn
    from src.utils.paths import CACHE_DIR

    # One vectorized transform up front: pyproj Transformers are thread-affine,
    # and sharing one across a fetch pool corrupts every lookup.
    to_4326 = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    lons, lats = to_4326.transform(frame["centroid_x"].to_numpy(),
                                   frame["centroid_y"].to_numpy())
    if not np.isfinite(lons).all():
        raise RuntimeError("non-finite lon/lat; set PROJ_NETWORK=OFF and retry")

    years = frame["year"].astype(int).to_numpy()
    geoids = frame["GEOID"].astype(str).to_numpy()
    search_cache = TractSearchCache(
        cache_dir=Path(CACHE_DIR) / "naip_search_cache" / "cross_sensor")
    ortho_fetch = make_fetch_fn(city, max_rps=max_rps)

    def _naip(i):
        return fetch_naip(lons[i], lats[i], CROP_SIZE_M, nbands=4,
                          out_pixels=OUT_PIXELS, year_hint=int(years[i]),
                          search_cache=search_cache, cache_key=geoids[i],
                          exact_year=True).crop

    def _ortho(i):
        return ortho_fetch(lons[i], lats[i], CROP_SIZE_M, nbands=4,
                           out_pixels=OUT_PIXELS, year_hint=int(years[i]),
                           exact_year=True).crop

    idx = range(len(frame))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        naip = list(pool.map(_naip, idx))
    if verbose:
        print(f"  NAIP : {sum(c is not None for c in naip):,}/{len(frame):,} crops")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        ortho = list(pool.map(_ortho, idx))
    if verbose:
        print(f"  ortho: {sum(c is not None for c in ortho):,}/{len(frame):,} crops")
    return naip, ortho


def agreement(df: pd.DataFrame) -> dict:
    """Agreement statistics from a frame with ``pred_naip`` / ``pred_ortho``.

    ``within_rho`` is the tract-size-weighted mean of per-tract Spearman
    correlations — the headline. ``pooled_rho`` is reported alongside because a
    large gap between them is itself informative: it means the sensors agree on
    which neighbourhoods are rich but not on within-neighbourhood ordering,
    which is exactly the part the event study depends on.
    """
    from scipy.stats import spearmanr

    sub = df.dropna(subset=["pred_naip", "pred_ortho"])
    if len(sub) < 10:
        return {"n": len(sub), "pooled_rho": float("nan"),
                "within_rho": float("nan")}

    pooled = float(spearmanr(sub["pred_naip"], sub["pred_ortho"]).statistic)

    rhos, weights = [], []
    for _, g in sub.groupby("GEOID"):
        if len(g) < 5 or g["pred_naip"].nunique() < 2 or g["pred_ortho"].nunique() < 2:
            continue
        r = spearmanr(g["pred_naip"], g["pred_ortho"]).statistic
        if np.isfinite(r):
            rhos.append(r)
            weights.append(len(g))
    within = float(np.average(rhos, weights=weights)) if rhos else float("nan")

    return {
        "n": int(len(sub)),
        "n_tracts": int(sub["GEOID"].nunique()),
        "n_tracts_scored": len(rhos),
        "pooled_rho": pooled,
        "within_rho": within,
        # Ordinal loss ⇒ a level/scale shift between sensors is harmless. Report
        # it so that is visible rather than assumed.
        "mean_naip": float(sub["pred_naip"].mean()),
        "mean_ortho": float(sub["pred_ortho"].mean()),
        "sd_naip": float(sub["pred_naip"].std()),
        "sd_ortho": float(sub["pred_ortho"].std()),
        "mean_shift": float(sub["pred_ortho"].mean() - sub["pred_naip"].mean()),
        "sd_ratio": float(sub["pred_ortho"].std() / sub["pred_naip"].std())
        if sub["pred_naip"].std() else float("nan"),
    }


def report(df: pd.DataFrame, *, city: str, tables_dir: Path = TABLES_DIR
           ) -> pd.DataFrame:
    """Per-year and overall agreement, written to results/tables/."""
    rows = []
    for year, g in df.groupby("year"):
        rows.append({"city": city, "year": int(year), **agreement(g)})
    rows.append({"city": city, "year": "all", **agreement(df)})
    out = pd.DataFrame(rows)

    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    path = Path(tables_dir) / f"csa_cross_sensor_agreement_{city}.csv"
    out.to_csv(path, index=False)
    print(f"\n{out.to_string(index=False)}")
    print(f"\n-> {path}")

    overall = out.iloc[-1]
    print("\n" + "=" * 74)
    if not np.isfinite(overall["within_rho"]):
        print("VERDICT: not enough paired predictions to judge. Check coverage.")
    elif overall["within_rho"] >= WITHIN_RHO_FLOOR:
        print(f"VERDICT: within-tract rho = {overall['within_rho']:.3f} "
              f">= {WITHIN_RHO_FLOOR} — the sensor swap holds.")
        print("  Proceed with the ortho pass; cite this table as the evidence")
        print("  for the out-of-sensor claim rather than asserting it.")
    else:
        print(f"VERDICT: within-tract rho = {overall['within_rho']:.3f} "
              f"< {WITHIN_RHO_FLOOR} — the model looks sensor-bound.")
        print("  This is a finding, not a bug: report it, and run Chicago on")
        print("  NAIP (spec.sensor='naip'), losing the annual cadence but")
        print("  keeping the event study on a sensor the model handles.")
    print("=" * 74)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--city", default="chicago")
    ap.add_argument("--savename", required=True, help="run id for the checkpoint")
    ap.add_argument("--years", default=None,
                    help="comma-separated overlap years (default: NAIP grid)")
    ap.add_argument("--n-tracts", type=int, default=DEFAULT_N_TRACTS)
    ap.add_argument("--per-tract", type=int, default=DEFAULT_PER_TRACT)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--max-rps", type=float, default=None)
    args = ap.parse_args()

    years = ([int(y) for y in args.years.split(",")] if args.years
             else list(DEFAULT_OVERLAP_YEARS))
    naip_years = set(ces.CSA_CITIES[args.city].panel_years or ())
    print(f"Cross-sensor check: {args.city}, years {years}")
    if naip_years and not set(years) <= naip_years:
        print(f"  note: {sorted(set(years) - naip_years)} are outside this "
              f"city's registered panel years")

    frame = sample_frame(args.city, years, n_tracts=args.n_tracts,
                         per_tract=args.per_tract)
    print(f"  {len(frame):,} building-years "
          f"({frame['GEOID'].nunique():,} tracts) x 2 sensors")

    naip, ortho = fetch_both(frame, args.city, max_workers=args.max_workers,
                             max_rps=args.max_rps)

    model = load_model(args.savename)
    frame = frame.assign(
        pred_naip=predict(model, naip),
        pred_ortho=predict(model, ortho),
    )
    report(frame, city=args.city)


def load_model(savename: str, *, nbands: int = 4, image_size: int = OUT_PIXELS,
               device=None):
    """Load ``{savename}_best.pth`` + its LoRA adapter, exactly as main.run does.

    Not reusing ``eval_small_city_offline.load_run_model``: that one pins a
    different run id at module scope and normalises with a 3-band ImageNet
    statistic, which is wrong for the 4-band crops here.
    """
    import torch
    from peft import PeftModel

    from src.custom_models import get_model
    from src.utils.paths import MODELS_DIR

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(MODELS_DIR) / "models_by_epoch" / savename
    best = ckpt_dir / f"{savename}_best.pth"
    if not best.exists():
        raise FileNotFoundError(f"no checkpoint at {best}")

    model = get_model("scalemae", image_size=image_size, bands=nbands,
                      kind="reg", meta_dim=0)
    model.head.load_state_dict(
        torch.load(best, map_location=device, weights_only=True))
    lora = ckpt_dir / f"{savename}_best_lora"
    if lora.exists():
        model.backbone = PeftModel.from_pretrained(
            model.backbone.base_model.model, lora)
    model.to(device).eval()
    model._device = device
    model._nbands = nbands
    return model


def predict(model, crops, *, batch_size: int = 32) -> np.ndarray:
    """Run inference over a crop list, keeping missing rows as NaN.

    Both sensors go through the identical transform — same normalisation, same
    dtype scaling — so any difference in the predictions is the imagery, not the
    preprocessing.
    """
    import torch
    from torchvision.transforms import v2 as T

    nbands = getattr(model, "_nbands", 4)
    device = getattr(model, "_device", torch.device("cpu"))
    # Matches main.run's eval transform: ImageNet stats on RGB, 0.5 on any extra
    # band (NIR).
    mean = [0.485, 0.456, 0.406] + [0.5] * max(0, nbands - 3)
    std = [0.229, 0.224, 0.225] + [0.5] * max(0, nbands - 3)
    tf = T.Compose([T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=mean, std=std)])

    ok = [i for i, c in enumerate(crops) if c is not None]
    out = np.full(len(crops), np.nan, dtype="float64")
    with torch.no_grad():
        for s in range(0, len(ok), batch_size):
            idx = ok[s:s + batch_size]
            batch = torch.stack([
                tf(torch.from_numpy(np.ascontiguousarray(crops[i]))) for i in idx
            ]).to(device)
            out[idx] = model(batch).squeeze(-1).cpu().numpy().astype("float64")
    return out


if __name__ == "__main__":
    main()
