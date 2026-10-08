"""Per-city prediction driver for the CSA event study (issue #36).

``evaluation.part_d`` is already city-generic — it loops the
:data:`src.csa_event_study.CSA_CITIES` registry and keys every output on the city.
What it lacked was *predictions* for anywhere but NYC: the ordinary US pass
samples 10% of each CBSA's tracts (``predict_tract_sample_frac``), which left
Chicago with ~208 of 2,070 tracts — far too sparse for a tract-level
treated/never-treated contrast. This module produces the dense per-city panel
``part_d`` needs.

What it does differently from the main prediction pass
------------------------------------------------------
* **Every tract, capped buildings.** The event study needs the full tract
  cross-section (a tract absent from the panel is neither treated nor control,
  it simply vanishes), but not every building in it. So: all tracts,
  ``buildings_per_tract`` (default 200) each. Sampling the *outcome* this way
  adds classical measurement error to the tract mean — it inflates the ATT's
  standard errors, it does not bias the ATT.
* **Per-city panel years.** Taken from ``spec.panel_years``, never a global —
  Cook County ortho is annual 2009-2025 while each state's NAIP grid differs and
  the four intersect in only {2021, 2023}. See :class:`CityCohortSpec`.
* **Per-city sensor.** Chicago runs on Cook County's own 6-inch annual ortho, a
  camera the model never saw in fine-tuning, which makes the holdout spatial
  *and* out-of-sensor. Selected through ``params["predict_sensor"]``; the whole
  fetch/GPU/resume engine below it is unchanged.

Why not reuse ``select_prediction_rows``
----------------------------------------
That function implements the *evaluation* sampling design (10% of tracts per
CBSA, 100 buildings each, plus the NYC full-universe bypass). Its tract tier is
precisely what this module replaces, and bolting a fourth mode onto it would
make the paper run's row selection depend on a flag only the event study sets.
The runner instead hands ``predict_year_chunked`` an already-selected frame,
which that function accepts as-is.

Why not ``LazyPairTable.materialize_year``
------------------------------------------
It materialises all 71.8M building-years before filtering — the RAM blow-up
tracked in issue #35. The city filter is pushed into the parquet read instead
(the ``eval_small_city_offline.load_inputs`` pattern), so a city costs its own
row count and nothing more.

Determinism
-----------
The building sample is a stable hash of ``building_id``, never RNG state, so it
is byte-identical in every panel year and across restarts. That is not a nicety:
``csa.estimate(balanced=True)`` assumes a balanced panel, and a tract whose
sampled buildings drifted between years would contribute a spurious year-to-year
outcome change indistinguishable from a real one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import csa_event_study as ces
from src.utils.paths import CACHE_DIR, PROCESSED_DATA_DIR, RESULTS_DIR

BUILDINGS_PARQUET = "pair_buildings_ms_us_epsg5070_all.parquet"
LABELS_PARQUET = "pair_labels_ms_us_W2_r5_years2010-2024n15_all.parquet"
TRACT_SPLITS = "tract_splits.feather"

# Default buildings per tract. At sigma_within = 0.5 the tract mean's 95% CI is
# +-0.098 at 100 and +-0.069 at 200; the loss anchors the global cross-sectional
# SD at 1.0, so this is well inside the noise the rank metrics already tolerate.
BUILDINGS_PER_TRACT_DEFAULT = 200


@dataclass(frozen=True)
class CityYearStatus:
    city: str
    year: int
    path: Path
    done: bool
    n_rows: int | None = None


class CSAPredictionRunner:
    """Dense per-city building predictions for the construction-cohort event study.

    Parameters
    ----------
    savename
        Run id; outputs land under ``<results>/<savename>/csa/<city>/``.
    params
        The training/prediction params dict. Per-city sensor keys are set by the
        runner, so the caller's dict does not need to know about ortho.
    cities
        Registry keys to run; ``None`` means every city that has a footprints
        table on disk.
    buildings_per_tract
        Cap per tract. ``None`` = every building (only sane for a small city).
    shard
        ``(i, n)`` — process only tracts whose stable hash falls in shard ``i``
        of ``n``. Lets several processes run in parallel; note that for ortho
        cities the binding constraint is the *municipal server's* rate limit,
        not local CPU, and the limiter is per-process — so sharding across
        processes multiplies the load the server sees. Divide ``max_rps``
        accordingly.
    max_rps
        Per-process request cap for ortho sources. ``None`` keeps the polite
        module default.
    """

    def __init__(self, savename: str, params: dict, *,
                 cities: list[str] | tuple[str, ...] | None = None,
                 buildings_per_tract: int | None = BUILDINGS_PER_TRACT_DEFAULT,
                 shard: tuple[int, int] = (0, 1),
                 max_rps: float | None = None,
                 processed_dir: Path = PROCESSED_DATA_DIR,
                 results_dir: Path = RESULTS_DIR,
                 cache_root: Path = CACHE_DIR):
        self.savename = str(savename)
        self.params = dict(params)
        self.buildings_per_tract = buildings_per_tract
        self.max_rps = max_rps
        self.processed_dir = Path(processed_dir)
        self.results_dir = Path(results_dir)
        self.cache_root = Path(cache_root)

        shard_i, shard_n = int(shard[0]), int(shard[1])
        if shard_n < 1 or not (0 <= shard_i < shard_n):
            raise ValueError(f"invalid shard {shard!r}: need 0 <= i < n, n >= 1")
        self.shard = (shard_i, shard_n)

        ready, missing = ces.available_cities(self.processed_dir, keys=cities)
        if cities is not None:
            unknown = set(cities) - set(ces.CSA_CITIES)
            if unknown:
                raise KeyError(f"unknown CSA cities: {sorted(unknown)}")

        # A city whose sensor is "zarr" is served by the dedicated whole-city
        # pass (main.run_nyc_zarr_validation_predictions), not by this runner.
        # Predicting it here would be actively harmful, not merely redundant:
        # this runner samples 200 buildings/tract from NAIP, while that pass
        # produces the FULL city universe from the zarr imagery — and
        # part_d._csa_city_preds prefers csa/<city>/ over the zarr directory, so
        # the sampled, wrong-sensor predictions would silently displace the good
        # ones. Naming the city explicitly overrides this.
        self.skipped_zarr = []
        if cities is None:
            self.skipped_zarr = [s for s in ready if s.sensor == "zarr"]
            ready = [s for s in ready if s.sensor != "zarr"]

        self.specs = ready
        self.missing = missing

    # ── paths ────────────────────────────────────────────────────────────────

    def city_dir(self, spec) -> Path:
        return self.results_dir / self.savename / "csa" / spec.key

    def output_csv(self, spec, year: int) -> Path:
        return self.city_dir(spec) / f"{int(year)}_predictions.csv"

    def chunk_root(self, spec) -> Path:
        return self.cache_root / "pred_chunks" / self.savename / "csa" / spec.key

    # ── row selection ────────────────────────────────────────────────────────

    def city_tracts(self, spec) -> pd.DataFrame:
        """Tract GEOIDs for this city, honouring ``spec.tract_source``.

        ``"prefix"`` filters ``tract_splits.feather`` by county FIPS.
        ``"footprints"`` instead takes the tracts the city's *dated footprints*
        actually cover, which is what a sub-county source needs: Chicago's
        municipal footprint layer stops at the city line well inside Cook
        County, so a FIPS prefix would admit ~500 suburban tracts that can never
        be dated and would enter the event study as permanent never-treated
        controls — biasing the control group toward places the treatment
        variable cannot even be measured in.
        """
        splits = pd.read_feather(self.processed_dir / TRACT_SPLITS)
        splits["GEOID_str"] = splits["GEOID"].astype(str)
        sub = splits[splits["GEOID_str"].str.startswith(tuple(spec.geoid_prefixes))]
        if spec.tract_source != "footprints":
            return sub[["GEOID_str", "type"]].reset_index(drop=True)

        covered = self._footprint_tracts(spec, sub)
        return (sub[sub["GEOID_str"].isin(covered)][["GEOID_str", "type"]]
                .reset_index(drop=True))

    def _footprint_tracts(self, spec, tract_splits) -> set[str]:
        """GEOIDs with at least ``spec.min_tract_footprints`` dated footprints."""
        import geopandas as gpd

        footprints = gpd.read_parquet(
            self.processed_dir / spec.footprints_filename)
        tracts = gpd.read_feather(self.processed_dir / TRACT_SPLITS)
        tracts["GEOID_str"] = tracts["GEOID"].astype(str)
        tracts = tracts[tracts["GEOID_str"].isin(set(tract_splits["GEOID_str"]))]
        tracts = tracts.to_crs(footprints.crs)

        cent = gpd.GeoDataFrame(geometry=footprints.geometry.centroid,
                                crs=footprints.crs)
        joined = gpd.sjoin(cent, tracts[["GEOID_str", "geometry"]],
                           how="inner", predicate="within")
        counts = joined.groupby("GEOID_str").size()
        return set(counts[counts >= int(spec.min_tract_footprints)].index)

    def _load_city_buildings(self, spec, geoids: set[str]) -> pd.DataFrame:
        """Buildings in the city, read with a GEOID-prefix pushdown.

        The national table is ~72M rows / 1.5 GB; reading it whole to filter
        afterwards is the RAM blow-up of issue #35. The prefix filter goes into
        the parquet scan, then the exact GEOID set is applied in memory.
        """
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.dataset as ds

        dset = ds.dataset(self.processed_dir / BUILDINGS_PARQUET)
        # GEOID is stored dictionary-encoded, and `starts_with` has no kernel for
        # dictionary<string> — it raises ArrowNotImplementedError rather than
        # falling back. Cast to plain string inside the filter so the predicate
        # binds; the cast is applied to the dictionary's small value set, not
        # per row, so it costs nothing.
        geoid = ds.field("GEOID").cast(pa.string())
        expr = None
        for prefix in spec.geoid_prefixes:
            e = pc.starts_with(geoid, str(prefix))
            expr = e if expr is None else (expr | e)
        table = dset.to_table(
            columns=["building_id", "GEOID", "cbsa_code",
                     "centroid_x", "centroid_y", "dist_to_center"],
            filter=expr,
        )
        df = table.to_pandas()
        df["GEOID"] = df["GEOID"].astype(str)
        return df[df["GEOID"].isin(geoids)].reset_index(drop=True)

    def sample_buildings(self, spec) -> pd.DataFrame:
        """Deterministic per-tract building sample, identical in every year.

        Ranks each tract's buildings by a stable hash and keeps the lowest
        ``buildings_per_tract``. Hash, not RNG: the same buildings must be drawn
        in every panel year, or a tract's outcome would move between years for
        reasons unrelated to what the imagery shows.
        """
        from src.prediction import _stable_hash

        tracts = self.city_tracts(spec)
        geoids = set(tracts["GEOID_str"])
        if not geoids:
            return pd.DataFrame()

        df = self._load_city_buildings(spec, geoids)
        if df.empty:
            return df

        if self.shard[1] > 1:
            keep = (_stable_hash(df["GEOID"]) % self.shard[1]) == self.shard[0]
            df = df[keep].reset_index(drop=True)
            if df.empty:
                return df

        if self.buildings_per_tract is not None:
            h = _stable_hash(df["building_id"])
            df = df.assign(_h=h)
            df["_rank"] = df.groupby("GEOID")["_h"].rank(method="first").astype(int)
            df = df[df["_rank"] <= int(self.buildings_per_tract)]
            df = df.drop(columns=["_h", "_rank"]).reset_index(drop=True)

        return df.merge(tracts.rename(columns={"GEOID_str": "GEOID"}),
                        on="GEOID", how="left")

    def city_frame(self, spec, year: int, buildings: pd.DataFrame | None = None
                   ) -> pd.DataFrame:
        """One year's frame in exactly the shape ``predict_year_chunked`` wants.

        Required columns: ``Rel_Score, GEOID, building_id, year, type,
        centroid_x, centroid_y``. Rows with no label are dropped here rather
        than fetched and thrown away — an unlabeled tract-year cannot enter the
        event study, and a crop costs a network round trip.
        """
        if buildings is None:
            buildings = self.sample_buildings(spec)
        if buildings.empty:
            return buildings

        labels = pd.read_parquet(self.processed_dir / LABELS_PARQUET,
                                 columns=["GEOID", "year", "Rel_Score"])
        labels = labels[labels["year"] == int(year)]
        labels["GEOID"] = labels["GEOID"].astype(str)

        df = buildings.merge(labels[["GEOID", "Rel_Score"]], on="GEOID", how="left")
        df["year"] = int(year)
        if "type" not in df.columns:
            df["type"] = "test"
        df["type"] = df["type"].fillna("unassigned").astype(str)
        return df[df["Rel_Score"].notna()].reset_index(drop=True)

    # ── running ──────────────────────────────────────────────────────────────

    def _city_params(self, spec) -> dict:
        """Params for one city: pin the sensor, disable the sampling tiers.

        The sampling knobs are set to None (not merely ignored) because they are
        part of ``prediction_fingerprint``: leaving the paper run's 10%/100
        values in place would let this dense pass share a chunk cache with a
        sparse one.
        """
        p = dict(self.params)
        p["predict_sensor"] = spec.sensor if spec.sensor != "zarr" else "naip"
        if spec.sensor == "ortho":
            p["predict_ortho_city"] = spec.key
            if self.max_rps is not None:
                p["predict_ortho_max_rps"] = self.max_rps
        p["predict_split"] = f"csa_{spec.key}"
        p["predict_tract_sample_frac"] = None
        p["predict_tract_sample_min"] = None
        p["predict_buildings_per_tract"] = self.buildings_per_tract
        p["predict_full_universe_geoid_prefixes"] = tuple(spec.geoid_prefixes)
        p["predict_cbsa_whitelist"] = None
        return p

    def run_city(self, spec, *, model, device, eval_transform,
                 model_path: Path, years: list[int] | None = None,
                 verbose: bool = True) -> dict:
        """Predict every panel year for one city. Resumable; skips finished years."""
        from src import prediction

        years = years or spec.years()
        params = self._city_params(spec)
        chunk_root = self.chunk_root(spec)
        self.city_dir(spec).mkdir(parents=True, exist_ok=True)
        prediction.init_chunk_root(
            chunk_root, prediction.prediction_fingerprint(params, model_path))

        if spec.sensor == "ortho":
            self._assert_band_order(spec, years, verbose=verbose)

        buildings = self.sample_buildings(spec)
        if verbose:
            n_tracts = buildings["GEOID"].nunique() if len(buildings) else 0
            print(f"\n=== CSA predictions: {spec.label} ===")
            print(f"  sensor      : {spec.sensor}")
            print(f"  panel years : {years}")
            print(f"  tracts      : {n_tracts:,}")
            print(f"  buildings   : {len(buildings):,} "
                  f"(<={self.buildings_per_tract}/tract)")
            print(f"  crops total : {len(buildings) * len(years):,}")
            if self.shard[1] > 1:
                print(f"  shard       : {self.shard[0]}/{self.shard[1]}")

        done = {}
        for year in years:
            out_csv = self.output_csv(spec, year)
            if out_csv.exists():
                if verbose:
                    print(f"  ✔ {year}: already complete ({out_csv.name})")
                done[year] = True
                continue
            df_year = self.city_frame(spec, year, buildings=buildings)
            if df_year.empty:
                if verbose:
                    print(f"  ⏭️  {year}: no labeled rows")
                done[year] = False
                continue
            done[year] = prediction.predict_year_chunked(
                model=model, df_year=df_year, year=year, params=params,
                device=device, eval_transform=eval_transform,
                chunk_root=chunk_root, output_csv=out_csv,
                max_workers=int(params.get("predict_fetch_workers", 16)),
                verbose=verbose,
            )
        return {"city": spec.key, "years": done}

    def _assert_band_order(self, spec, years, *, verbose: bool = True) -> None:
        """Fail before fetching millions of crops if band 4 is not NIR.

        A permuted band order raises nowhere — it just feeds the model wrong
        channels and corrupts every prediction — so it must be caught up front,
        not inferred later from an odd-looking ATT.
        """
        from src.data.ortho_fetcher import NDVI_PROBE_POINTS, check_band_order

        if spec.key not in NDVI_PROBE_POINTS:
            return
        year = max(years)
        ok, ndvi = check_band_order(spec.key, year)
        if verbose:
            print(f"  band-order check ({spec.key} {year}): "
                  f"median NDVI over vegetation = {ndvi:.3f}")
        if not ok:
            raise RuntimeError(
                f"{spec.key} {year}: band-order check failed (median NDVI "
                f"{ndvi:.3f} over a known park). Band 4 is probably not NIR — "
                f"refusing to predict, as every crop would feed the model "
                f"permuted channels with nothing raising downstream."
            )

    def run(self, *, model, device, eval_transform, model_path: Path,
            verbose: bool = True) -> dict:
        """Predict every registered, available city."""
        for spec in self.skipped_zarr:
            print(f"⏭️  {spec.label}: served by the dedicated whole-city zarr "
                  f"pass (full universe, zarr imagery) — not re-predicted here.")
        for spec, reason in self.missing:
            print(f"⏭️  {spec.label}: {reason}")
        if not self.specs:
            print("no CSA city has a footprint + year-built table — nothing to do.")
            return {}
        return {spec.key: self.run_city(spec, model=model, device=device,
                                        eval_transform=eval_transform,
                                        model_path=model_path, verbose=verbose)
                for spec in self.specs}

    # ── reporting ────────────────────────────────────────────────────────────

    def status(self) -> pd.DataFrame:
        """One row per (city, year): whether its prediction CSV exists yet."""
        rows = []
        for spec in self.specs:
            for year in spec.years():
                path = self.output_csv(spec, year)
                n = None
                if path.exists():
                    try:
                        n = sum(1 for _ in path.open()) - 1
                    except OSError:
                        n = None
                rows.append(CityYearStatus(spec.key, year, path,
                                           path.exists(), n).__dict__)
        return pd.DataFrame(rows)

    def plan(self) -> pd.DataFrame:
        """Crop budget per city, without fetching anything.

        Run this before committing a machine to a multi-hour pass.
        """
        rows = []
        for spec in self.specs:
            buildings = self.sample_buildings(spec)
            years = spec.years()
            rows.append({
                "city": spec.key,
                "sensor": spec.sensor,
                "n_tracts": int(buildings["GEOID"].nunique()) if len(buildings) else 0,
                "n_buildings": len(buildings),
                "n_years": len(years),
                "n_crops": len(buildings) * len(years),
                "years": ",".join(str(y) for y in years),
            })
        return pd.DataFrame(rows)


if __name__ == "__main__":                                  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(
        description="Plan / inspect CSA per-city prediction passes. Running the "
                    "predictions themselves goes through src/main.py, which "
                    "builds the model, device and eval transform exactly as "
                    "training does.")
    ap.add_argument("--savename", required=True)
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--buildings-per-tract", type=int,
                    default=BUILDINGS_PER_TRACT_DEFAULT)
    ap.add_argument("--shard", default="0/1", help="i/n tract shard")
    ap.add_argument("--max-rps", type=float, default=None,
                    help="per-process request cap for ortho sources")
    ap.add_argument("--status", action="store_true",
                    help="per city-year completion instead of the crop budget")
    args = ap.parse_args()

    i, n = (int(v) for v in args.shard.split("/"))
    # Both --plan and --status are pure bookkeeping over parquet + the registry:
    # no model, no GPU, no network. Params only supply paths here, so an empty
    # dict is enough and avoids importing src.main (torch, wandb, credentials).
    runner = CSAPredictionRunner(
        args.savename, {}, cities=args.cities,
        buildings_per_tract=args.buildings_per_tract,
        shard=(i, n), max_rps=args.max_rps)
    for spec, reason in runner.missing:
        print(f"⏭️  {spec.label}: {reason}")
    frame = runner.status() if args.status else runner.plan()
    print(frame.to_string(index=False) if len(frame) else "(nothing available)")
