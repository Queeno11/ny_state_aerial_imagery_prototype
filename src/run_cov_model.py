"""ACS-covariate diagnostic model (``scalemae_cov``) — train, predict NYC, compare.

Question: once the model is TOLD who lives in a tract (share of adults with a
bachelor's degree, share aged 25-54 — src/data/covariates.py), does it predict
wealth *change* in NYC, in particular in tracts with new construction? The
imagery-only run (``run_20260722``) moves every construction tract by about the
same amount whether residents changed or not.

Design (everything else is the base run's logged config):

* ``model_name="scalemae_cov"``: same backbone + LoRA + head, with
  ``meta_dim = len(COVARIATES)`` late-fusion inputs;
* warm start from the base run's ``_best`` head + LoRA. The new covariate
  weights start at exactly zero, so epoch 0 reproduces the base model;
* trains on the base run's shard cache (``reuse_cache_of``) — shards are images
  + labels, independent of the model — so no NAIP refetch;
* ``train_scope``: "all" (LoRA + head, default) or "head" (backbone frozen).
  With "all", watch the image pathway: covariates that rank tracts on their own
  can let weight decay shrink it (the dist_to_center collapse noted in main.run).

Reading the result. Covariates come from the same ACS survey as the label, so
their change shares sampling noise with the label change. The compare stage
therefore reports, next to both models, how far the education change ALONE
goes ("edu-only"); imagery adds something only if the covariate model beats it.

Usage (WSL + torch_geo_env, repo root, GPU box)::

    python -m src.data.covariates                  # once: builds tract_covariates.parquet
    python src/run_cov_model.py --stage check      # inputs, checkpoint, cache owner
    python src/run_cov_model.py --stage train --epochs 200 --lr 3e-5
    python src/run_cov_model.py --stage predict-nyc   # NAIP 2017/19/21/22, 50 bldg/tract
    python src/run_cov_model.py --stage compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import MODELS_DIR, RESULTS_DIR

BASE_RUN_ID = "run_20260722"
COVARIATES = ("edu_ba_share", "age_25_54_share")
DEFAULT_EPOCHS = 200
DEFAULT_LR = 3e-5
NYC_NAIP_YEARS = (2017, 2019, 2021, 2022)   # flights with base-run NYC predictions
NYC_BUILDINGS_PER_TRACT = 50
COMPARE_ACS = (2018, 2023)                   # 5-year window matching the flights

# Keys run() adds after the defaults are filled; not valid as input params.
_DERIVED_KEYS = ("tau_meters_requested",)


# ─── parameters ──────────────────────────────────────────────────────────────

def cov_savename(base_run_id: str = BASE_RUN_ID, scope: str = "all") -> str:
    return f"{base_run_id}_cov" + ("" if scope == "all" else f"_{scope}")


def load_logged_params(run_id: str, models_dir: Path = MODELS_DIR) -> dict:
    import json
    path = Path(models_dir) / run_id / f"{run_id}_logs.txt"
    if not path.exists():
        raise FileNotFoundError(f"No parameter log for '{run_id}' at {path}.")
    return json.loads(path.read_text())


def input_params_from_log(logged: dict) -> dict:
    """Logged (post-snap) params -> valid run() input (restore the tau request)."""
    params = {k: v for k, v in logged.items() if k not in _DERIVED_KEYS}
    if "tau_meters_requested" in logged:
        params["tau_meters"] = logged["tau_meters_requested"]
    return params


def cov_params(base_params: dict, *, base_run_id: str = BASE_RUN_ID, epochs: int = DEFAULT_EPOCHS,
               lr: float = DEFAULT_LR, scope: str = "all", covariates=COVARIATES) -> dict:
    """The base run's params + the covariate-model overrides."""
    params = dict(base_params)
    params.update({
        "model_name": "scalemae_cov",
        "covariates": tuple(covariates),
        "init_from": base_run_id,
        "reuse_cache_of": base_run_id,
        "train_scope": scope,
        "n_epochs": int(epochs),
        "learning_rate": float(lr),
        "run_id": cov_savename(base_run_id, scope),
        "resume_lr_override": None,   # a resumed covariate run keeps its own lr
    })
    return params


# ─── comparison helpers ──────────────────────────────────────────────────────

def subset_predictions(src_csv: Path, keep_ids: set, dst_csv: Path) -> int:
    """Rows of ``src_csv`` whose building_id is in ``keep_ids`` -> ``dst_csv``.
    Puts the base run on exactly the buildings the covariate run predicted."""
    df = pd.read_csv(src_csv, dtype={"GEOID": str})
    df = df[df["building_id"].isin(keep_ids)]
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_csv, index=False)
    return len(df)


def edu_only_change(pairs: pd.DataFrame, lookup, years=COMPARE_ACS) -> pd.Series:
    """Change in the standardized education share between the two ACS
    releases, aligned with ``pairs`` — the covariates-alone reference."""
    g = pairs["GEOID"].astype(str).to_numpy()
    i = list(lookup.names).index("edu_ba_share")
    pre = lookup.lookup(g, np.full(len(g), years[0]))[:, i]
    post = lookup.lookup(g, np.full(len(g), years[1]))[:, i]
    return pd.Series(post - pre, index=pairs.index)


def compare_table(pairs_by_model: dict, edu_change: pd.Series | None = None) -> pd.DataFrame:
    """Per construction group: Spearman of observed vs predicted Δlog (GB2)
    and the predicted shift vs stable, for each model (+ edu-only rho)."""
    from scipy.stats import spearmanr
    rows = []
    groups = ("in_stable", "in_change_1pct", "in_change_5pct", "in_change_10pct")
    for name, p in pairs_by_model.items():
        st = p[p["in_stable"].astype(bool)]
        for g in groups:
            s = p[p[g].astype(bool)]
            if len(s) < 5:
                continue
            row = {"model": name, "group": g[3:], "n": len(s),
                   "rho": float(spearmanr(s["dlog_label_usd"], s["dlog_pred_usd"])[0]),
                   "shift_pred": float(s["dlog_pred_usd"].mean() - st["dlog_pred_usd"].mean()),
                   "shift_obs": float(s["dlog_label_usd"].mean() - st["dlog_label_usd"].mean())}
            held = s[s["split_group"] == "heldout"]
            row["rho_heldout"] = (float(spearmanr(held["dlog_label_usd"], held["dlog_pred_usd"])[0])
                                  if len(held) >= 5 else np.nan)
            if edu_change is not None and name == next(iter(pairs_by_model)):
                e = edu_change.reindex(s.index)
                rows.append({"model": "edu-only", "group": g[3:], "n": len(s),
                             "rho": float(spearmanr(s["dlog_label_usd"], e)[0])})
            rows.append(row)
    return pd.DataFrame(rows)


# ─── stages ──────────────────────────────────────────────────────────────────

def stage_check(base_run_id: str = BASE_RUN_ID) -> dict:
    """Inputs present? Prints what a train run would do; never trains."""
    from src.data import covariates
    from src.main import read_cache_marker, run_cache_root

    ckpt = Path(MODELS_DIR) / "models_by_epoch" / base_run_id
    status = {
        "covariate_table": covariates.table_path().exists(),
        "base_best": (ckpt / f"{base_run_id}_best.pth").exists(),
        "base_lora": (ckpt / f"{base_run_id}_best_lora" / "adapter_model.safetensors").exists(),
    }
    base = input_params_from_log(load_logged_params(base_run_id))
    owner = read_cache_marker(run_cache_root(base))
    status["cache_owner"] = owner
    status["cache_reusable"] = owner == base_run_id
    for k, v in status.items():
        print(f"  {k:16s}: {v}")
    if not status["covariate_table"]:
        print("  -> build it: python -m src.data.covariates")
    if not status["cache_reusable"]:
        print(f"  -> the shard cache in {run_cache_root(base)} belongs to '{owner}', not "
              f"'{base_run_id}'; training would refuse to start (it never clears a reused cache).")
    return status


def stage_build_covariates(out: Path | None = None) -> Path:
    """Build data/processed/tract_covariates.parquet from the ACS tract files."""
    from src.data import covariates
    from src.data.process_acs import PANEL_YEARS
    from src.utils.paths import ACS_ROOT_DIR, PROCESSED_DATA_DIR
    panel = Path(PROCESSED_DATA_DIR) / f"us_metros_panel_{PANEL_YEARS[0]}_{PANEL_YEARS[-1]}.feather"
    table = covariates.build_covariate_table(Path(ACS_ROOT_DIR), panel, list(PANEL_YEARS))
    out = Path(out) if out else covariates.table_path()
    table.to_parquet(out, index=False)
    print(f"wrote {out}: {len(table):,} tract-years")
    return out


def stage_train(base_run_id: str, epochs: int, lr: float, scope: str) -> str:
    from src import main
    params = cov_params(input_params_from_log(load_logged_params(base_run_id)),
                        base_run_id=base_run_id, epochs=epochs, lr=lr, scope=scope)
    main.run(params, train=True, generate_predictions=False, evaluate=False)
    return params["run_id"]


def stage_predict_nyc(base_run_id: str, scope: str, years=NYC_NAIP_YEARS,
                      buildings_per_tract: int = NYC_BUILDINGS_PER_TRACT) -> Path:
    """NYC NAIP predictions of the covariate model (every tract, capped
    buildings) -> results/<savename>/csa/nyc/{year}_predictions.csv."""
    import torch
    from src import main
    from src import csa_event_study as ces
    from src.csa_predict import CSAPredictionRunner

    params = main.resolve_runtime_params(
        cov_params(input_params_from_log(load_logged_params(base_run_id)),
                   base_run_id=base_run_id, scope=scope))
    savename = params["run_id"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, best_path = main.load_best_model(params, savename, device)
    runner = CSAPredictionRunner(savename, params, cities=["nyc"],
                                 buildings_per_tract=buildings_per_tract)
    runner.run_city(ces.CSA_CITIES["nyc"], model=model, device=device,
                    eval_transform=main.make_eval_transform(params["nbands"]),
                    model_path=best_path, years=list(years))
    return runner.city_dir(ces.CSA_CITIES["nyc"])


def _mask_credentials(value):
    """'http://user:pass@host:3128' -> 'http://***@host:3128' (never print secrets)."""
    import re
    return None if value is None else re.sub(r"//[^/@]+@", "//***@", str(value))


def stage_diagnose_fetch(base_run_id: str, geoid: str, year: int,
                         buildings_per_tract: int = NYC_BUILDINGS_PER_TRACT) -> dict:
    """Fetch one NYC tract-year through the real prediction fetch path and
    print every failure with its traceback, plus the environment that shapes
    reads (PROJ / GDAL / proxies / versions). No model, no GPU; ~1 minute."""
    import collections
    import os
    import traceback
    import pyproj
    import rasterio
    from pyproj import Transformer
    from src import csa_event_study as ces
    from src import geo_utils, main, prediction
    from src.csa_predict import CSAPredictionRunner
    from src.data import naip_fetcher as nf

    env_keys = ("PROJ_NETWORK", "PROJ_DATA", "PROJ_LIB", "GDAL_DATA", "CURL_CA_BUNDLE",
                "HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy")
    print("env:", {k: _mask_credentials(os.environ.get(k)) for k in env_keys})
    print(f"pyproj {pyproj.__version__} (PROJ {pyproj.proj_version_str}, network "
          f"{pyproj.network.is_network_enabled()}, data {pyproj.datadir.get_data_dir()}) | "
          f"rasterio {rasterio.__version__} (GDAL {rasterio.__gdal_version__})")

    params = main.resolve_runtime_params(
        cov_params(input_params_from_log(load_logged_params(base_run_id)), base_run_id=base_run_id))
    runner = CSAPredictionRunner(params["run_id"], params, cities=["nyc"],
                                 buildings_per_tract=buildings_per_tract)
    spec = ces.CSA_CITIES["nyc"]
    city_params = runner._city_params(spec)
    df = runner.city_frame(spec, int(year))
    df = df[df["GEOID"].astype(str) == str(geoid)].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"no labeled {year} rows for tract {geoid}")
    to_4326 = Transformer.from_crs(geo_utils.METRIC_CRS, "EPSG:4326", always_xy=True)
    lon, lat = to_4326.transform(df["centroid_x"].to_numpy(), df["centroid_y"].to_numpy())
    df = df.assign(lon=lon, lat=lat)
    print(f"tract {geoid} {year}: {len(df)} rows | non-finite lon/lat: "
          f"{int((~np.isfinite(df['lon']) | ~np.isfinite(df['lat'])).sum())}")

    failures_tb = collections.OrderedDict()

    def loud_read(src, bbox, nbands, out_pixels):
        crop, nir, partial, failure = nf.read_naip_crop_from_src(src, bbox, nbands, out_pixels)
        if failure is not None and len(failures_tb) < 3:
            try:   # re-run the same steps un-guarded to capture the traceback
                native = nf.bbox_to_native_checked(bbox, src.crs)
                from rasterio.windows import from_bounds
                w = from_bounds(*native, transform=src.transform)
                src.read(indexes=list(range(1, min(nbands, src.count) + 1)), window=w,
                         out_shape=(min(nbands, src.count), out_pixels, out_pixels))
            except Exception:
                failures_tb.setdefault(traceback.format_exc(), (list(bbox), str(src.crs)))
        return crop, nir, partial, failure

    nf.reset_error_examples()
    crops, years, fails = prediction.fetch_prediction_chunk(
        df, city_params, None, to_4326, max_passes=1, read_fn=loud_read, verbose=True)
    counts = dict(collections.Counter(fails))
    print("outcomes:", counts)
    print("error examples:", nf.get_error_examples())
    print("fetch stats:", {k: v for k, v in nf.get_fetch_stats().items() if v})
    for tb, (bbox, crs) in failures_tb.items():
        print(f"\n--- failure (bbox {bbox}, crs {crs}) ---\n{tb}")
    return counts


def stage_compare(base_run_id: str, scope: str, years=NYC_NAIP_YEARS,
                  acs=COMPARE_ACS) -> pd.DataFrame:
    """Construction scatter (ACS ``acs``) for the covariate model and the base
    model on the same buildings, plus the edu-only reference."""
    from src import evaluation as ev
    from src.data.covariates import load_lookup
    from src.utils.paths import PROCESSED_DATA_DIR

    savename = cov_savename(base_run_id, scope)
    cov_dir = Path(RESULTS_DIR) / savename / "csa" / "nyc"
    base_dir = Path(RESULTS_DIR) / savename / "compare" / "base_same_buildings"
    for y in years:
        cov_csv = cov_dir / f"{y}_predictions.csv"
        if not cov_csv.exists():
            raise FileNotFoundError(f"{cov_csv} missing — run --stage predict-nyc first")
        keep = set(pd.read_csv(cov_csv, usecols=["building_id"])["building_id"])
        n = subset_predictions(Path(RESULTS_DIR) / base_run_id / f"{y}_predictions.csv",
                               keep, base_dir / f"{y}_predictions.csv")
        print(f"  {y}: base run on the same {len(keep):,} buildings -> {n:,} rows")

    pairs = {}
    for name, pred_dir in (("covariates", cov_dir), ("imagery only", base_dir)):
        out = Path(RESULTS_DIR) / savename / "compare" / name.replace(" ", "_")
        ev._make_dirs(out)
        ev.part_f_construction_scatter(pred_dir, PROCESSED_DATA_DIR, out, acs_pre=acs[0],
                                       acs_post=acs[1], pred_results_dir=pred_dir,
                                       pred_years=list(years))
        pairs[name] = pd.read_csv(out / "tables" / "F_construction_pairs.csv",
                                  dtype={"GEOID": str})
    edu = edu_only_change(pairs["covariates"], load_lookup(COVARIATES), acs)
    table = compare_table(pairs, edu)
    path = Path(RESULTS_DIR) / savename / "compare" / "cov_vs_base.csv"
    table.to_csv(path, index=False)
    print(table.round(3).to_string(index=False))
    print(f"wrote {path}")
    return table


def main_cli(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True,
                    choices=["check", "build-covariates", "train", "predict-nyc", "compare",
                             "diagnose-fetch"])
    ap.add_argument("--base-run-id", default=BASE_RUN_ID)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--scope", choices=["all", "head"], default="all")
    ap.add_argument("--buildings-per-tract", type=int, default=NYC_BUILDINGS_PER_TRACT)
    ap.add_argument("--geoid", default="36005029600", help="diagnose-fetch: tract to fetch")
    ap.add_argument("--year", type=int, default=2017, help="diagnose-fetch: NAIP year")
    args = ap.parse_args(argv)
    if args.stage == "check":
        stage_check(args.base_run_id)
    elif args.stage == "build-covariates":
        stage_build_covariates()
    elif args.stage == "train":
        stage_train(args.base_run_id, args.epochs, args.lr, args.scope)
    elif args.stage == "predict-nyc":
        stage_predict_nyc(args.base_run_id, args.scope,
                          buildings_per_tract=args.buildings_per_tract)
    elif args.stage == "compare":
        stage_compare(args.base_run_id, args.scope)
    elif args.stage == "diagnose-fetch":
        stage_diagnose_fetch(args.base_run_id, args.geoid, args.year,
                             buildings_per_tract=args.buildings_per_tract)


if __name__ == "__main__":
    main_cli()
