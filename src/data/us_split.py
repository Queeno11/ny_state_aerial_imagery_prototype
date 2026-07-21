"""US scale-up sampling overrides for the whole-city (CBSA) split.

This module holds the *policy* for the US Callaway–Sant'Anna validation sample —
which cities are pinned into the test set, how NYC is handled, and how a small
spatially-clustered validation holdout is carved out of the large test cities —
plus the pure functions that realise it. It deliberately contains **no CSA /
event-study code**: its only job is to produce train/val/test *assignments* so
the model can be trained.

Design (see the project issue "CSA validation sample design for the US scale-up"):

* **Forced test cities** (:data:`FORCED_TEST_CBSAS`) are the whole-CBSA holdouts
  where the event study will later run. Chicago/Seattle/Tampa/Nashville anchor
  the region/size spread; a few mid/small **FL and TN** metros ride along (TN's
  statewide biennial imagery makes them cheap panels) so a referee asking for a
  smaller city is already covered. They are pinned via
  :func:`cbsa_brackets.build_city_split`'s ``forced_splits`` argument.

* **NYC** (:data:`NYC_CBSA`) is the sole *within-city* exception: it never gets a
  whole-city split. Its tracts inherit the validated notebook holdout
  (``nyc_tract_splits.feather``, seed 25, spatially-clustered, income-quintile
  stratified, 100 m dead-zone buffer). NYC therefore contributes train tracts to
  training *and* a spatial val/test holdout. It is passed to
  ``build_city_split`` via ``exclude_cbsas``.

* **Clustered test-city validation** (NYC + Chicago only): NYC's val comes from
  the notebook holdout (reported as ``val_within_nyc``); Chicago
  (:data:`VAL_CLUSTER_CBSAS`) gets the *same* method applied fresh — contiguous,
  income-stratified tract clusters quarantined by a dead-zone buffer, reported as
  ``val_within_chicago``. These per-city validation metrics are the closest
  proxies to the two headline CSA panels, so they drive model selection. The
  other test cities (Seattle/Tampa/Nashville and the mid/small FL–TN panels) are
  kept **whole** in test — undiluted CSA panels. Chicago's clusters are cut a
  little larger than the NYC sample (:data:`VAL_EVAL_FRACTION`).

All geometry work assumes/forces a metric CRS (EPSG:5070, the panel's
``geo_utils.METRIC_CRS``) so buffer radii are plain metres and the code is
US-general (no NYC-only state-plane assumption).
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import PROCESSED_DATA_DIR

# --- Policy constants ---------------------------------------------------------

NYC_CBSA = "35620"  # New York-Newark-Jersey City, NY-NJ — within-city split only.

# cbsa_code -> (title, region, size tier). Pinned wholesale into the test set.
FORCED_TEST_CBSAS: dict[str, tuple[str, str, str]] = {
    "16980": ("Chicago-Naperville-Elgin, IL-IN", "Midwest", "mega"),
    "42660": ("Seattle-Tacoma-Bellevue, WA", "West/PNW", "large"),
    "45300": ("Tampa-St. Petersburg-Clearwater, FL", "South/FL", "large"),
    "34980": ("Nashville-Davidson--Murfreesboro--Franklin, TN", "South/TN", "large"),
    # Mid/small FL & TN ride-alongs (state backbones → cheap panels; TN biennial).
    "28940": ("Knoxville, TN", "South/TN", "medium"),
    "16860": ("Chattanooga, TN-GA", "South/TN", "small"),
    "35840": ("North Port-Bradenton-Sarasota, FL", "South/FL", "medium"),
    "15980": ("Cape Coral-Fort Myers, FL", "South/FL", "medium"),
}

# Test cities that donate a *carved* clustered val holdout. Only Chicago: NYC's
# val comes from the notebook holdout instead, and the other test cities are kept
# whole (undiluted CSA panels). Each maps to its own reported val-set name.
VAL_CLUSTER_CBSAS: tuple[str, ...] = ("16980",)
CARVE_VAL_TYPE: dict[str, str] = {"16980": "val_within_chicago"}

# Named spatial-validation sets (reported separately, never pooled). NYC's comes
# from the notebook holdout; Chicago's from the carve above.
SPATIAL_VAL_TYPES: tuple[str, ...] = ("val_within_nyc", "val_within_chicago")

# Clustering parameters (metres). Mirror the NYC notebook, but cut the val
# fraction a little larger so each city yields enough clustered tracts.
CLUSTER_RADIUS_M = 300.0
DEAD_ZONE_M = 100.0            # == tau; quarantines val clusters from test.
VAL_EVAL_FRACTION = 0.08       # NYC notebook used 0.06; "a bit larger".
CLUSTER_SEED = 25             # same seed family as the validated NYC split.

METRIC_EPSG = 5070            # geo_utils.METRIC_CRS; metres.

NYC_TRACT_SPLIT_PATH = PROCESSED_DATA_DIR / "nyc_tract_splits.feather"

# Notebook tract-type -> canonical building-split type. NYC's spatial val is
# reported under its own name so it is never pooled with Chicago's.
_NYC_TYPE_MAP = {
    "train": "train",
    "val": "val_within_nyc",
    "test": "test",
    "dead_zone": "dead_zone",
}

_REL_SCORE_RE = re.compile(r"^Rel_Score_\d{4}$")


def forced_test_split_map() -> dict[str, str]:
    """``{cbsa_code -> "test"}`` for every forced test city (for ``build_city_split``)."""
    return {code: "test" for code in FORCED_TEST_CBSAS}


def mean_rel_score(panel: pd.DataFrame) -> pd.Series:
    """Row-wise mean of the per-year ``Rel_Score_YYYY`` columns (NaN-robust).

    The cross-sectional z-score labels are already city-relative, so their mean
    over time is a stable per-tract wealth level to stratify clusters on.
    """
    cols = [c for c in panel.columns if _REL_SCORE_RE.match(c)]
    if not cols:
        raise KeyError("No Rel_Score_YYYY columns found for stratification.")
    return panel[cols].mean(axis=1, skipna=True)


def nyc_tract_type_map(path: Path | str | None = None) -> dict[str, str]:
    """``{GEOID -> canonical type}`` from the validated NYC notebook holdout.

    Maps notebook ``val`` -> ``val_within_nyc`` and passes train/test/dead_zone
    through. Returns ``{}`` (with a warning) if the artifact is missing, so a
    build over a non-NYC universe still proceeds. ``path`` defaults to the module
    attribute :data:`NYC_TRACT_SPLIT_PATH` (resolved at call time, so tests can
    monkeypatch it).
    """
    path = Path(path) if path is not None else NYC_TRACT_SPLIT_PATH
    if not path.exists():
        warnings.warn(
            f"NYC tract-split artifact absent ({path}); NYC will not receive a "
            "within-city split. Run notebooks/split_train_test_val.ipynb first."
        )
        return {}
    nyc = pd.read_feather(path, columns=["GEOID", "type"])
    nyc["GEOID"] = nyc["GEOID"].astype(str)
    unknown = set(nyc["type"]) - set(_NYC_TYPE_MAP)
    if unknown:
        warnings.warn(f"Unmapped NYC tract types ignored: {sorted(unknown)}")
    return {
        g: _NYC_TYPE_MAP[t]
        for g, t in zip(nyc["GEOID"], nyc["type"])
        if t in _NYC_TYPE_MAP
    }


def grow_stratified_tract_clusters(
    gdf,
    cluster_radius: float,
    score: pd.Series,
    eval_fraction: float,
    rng: np.random.Generator,
    n_quantiles: int = 5,
) -> set:
    """Grow contiguous, income-stratified tract clusters (notebook method).

    Ports ``create_stratified_tract_holdout`` from
    ``notebooks/split_train_test_val.ipynb``: within each income quintile, seed
    tracts are drawn (area-weighted) and every tract within ``cluster_radius`` of
    a seed is captured, until ``eval_fraction`` of that quintile's tracts are
    held out. ``gdf`` must be in a metric CRS (radius in metres).

    Returns the set of captured GEOIDs. Deterministic given ``rng``.
    """
    if gdf.empty:
        return set()
    work = gdf.reset_index(drop=True).copy()
    work["_score"] = score.to_numpy()
    valid = work["_score"].notna()
    if valid.sum() < n_quantiles:
        return set()
    # Income quintiles (fewer bins if the city is tiny / ties dominate).
    try:
        work.loc[valid, "_q"] = pd.qcut(
            work.loc[valid, "_score"], q=n_quantiles, labels=False, duplicates="drop"
        )
    except ValueError:
        work.loc[valid, "_q"] = 0

    captured: set = set()
    areas_all = work.geometry.area.to_numpy()
    for q, group in work[valid].groupby("_q"):
        group_geoids = set(group["GEOID"])
        target = math.ceil(len(group_geoids) * eval_fraction)
        if target == 0:
            continue
        idx = group.index.to_numpy()
        guard = 0
        while len(group_geoids & captured) < target and guard < 10 * target + 50:
            guard += 1
            areas = areas_all[idx]
            if areas.sum() == 0:
                break
            seed_i = rng.choice(idx, p=areas / areas.sum())
            seed_geom = work.geometry.iloc[seed_i]
            hit = work.geometry.intersects(seed_geom.buffer(cluster_radius))
            captured.update(work.loc[hit, "GEOID"])
    return captured


def carve_test_city_val(
    tract_panel,
    val_cluster_cbsas=VAL_CLUSTER_CBSAS,
    cluster_radius: float = CLUSTER_RADIUS_M,
    dead_zone: float = DEAD_ZONE_M,
    eval_fraction: float = VAL_EVAL_FRACTION,
    seed: int = CLUSTER_SEED,
) -> dict[str, str]:
    """``{GEOID -> "val_within_<city>"|"dead_zone"}`` carved from test cities.

    For each CBSA in ``val_cluster_cbsas`` present in ``tract_panel``: grow
    income-stratified clusters -> that city's val type (from
    :data:`CARVE_VAL_TYPE`, e.g. Chicago -> ``val_within_chicago``); every
    remaining tract touching the ``dead_zone``-buffered val geometry ->
    ``dead_zone`` (dropped from every split so no test tract sits adjacent to a
    val tract). ``tract_panel`` must be a GeoDataFrame with GEOID, cbsa_code,
    geometry and Rel_Score_YYYY columns; it is reprojected to the metric CRS.
    """
    import geopandas as gpd  # local import: pure-pandas tests don't need geopandas

    if not isinstance(tract_panel, gpd.GeoDataFrame) or tract_panel.empty:
        return {}
    gdf = tract_panel.to_crs(epsg=METRIC_EPSG)
    gdf = gdf.assign(cbsa_code=gdf["cbsa_code"].astype(str),
                     GEOID=gdf["GEOID"].astype(str))
    score_all = mean_rel_score(gdf)

    overrides: dict[str, str] = {}
    for i, code in enumerate(val_cluster_cbsas):
        city = gdf[gdf["cbsa_code"] == code]
        if city.empty:
            warnings.warn(f"val-cluster CBSA {code} absent from the panel; skipped.")
            continue
        # Distinct, deterministic stream per city.
        rng = np.random.default_rng(seed + i)
        val_geoids = grow_stratified_tract_clusters(
            city, cluster_radius, score_all.loc[city.index], eval_fraction, rng
        )
        if not val_geoids:
            continue
        val_type = CARVE_VAL_TYPE.get(code, "val_spatial")
        val_city = city[city["GEOID"].isin(val_geoids)]
        buffer_geom = val_city.geometry.union_all().buffer(dead_zone)
        in_buffer = city.geometry.intersects(buffer_geom)
        for g in city.loc[in_buffer, "GEOID"]:
            overrides[g] = "dead_zone"      # buffer first...
        for g in val_geoids:
            overrides[g] = val_type         # ...val wins over its own buffer.
    return overrides


def nyc_greater_metro_overrides(
    tract_panel, nyc_types: dict[str, str], dead_zone: float = DEAD_ZONE_M
) -> dict[str, str]:
    """Route the greater-NYC CBSA (NJ / Long Island / Westchester) to ``train``.

    The validated notebook split only covers the five boroughs; the rest of the
    NYC CBSA would otherwise fall to ``unassigned`` and be dropped from training.
    Those tracts are supplementary training data (they are *not* the CSA
    geography), so they are labelled ``train`` — except any tract within
    ``dead_zone`` metres of a held-out NYC-proper tract (test or
    ``val_within_nyc``), which becomes ``dead_zone`` so no borough holdout leaks
    into training across the boundary. ``tract_panel`` needs GEOID, cbsa_code,
    geometry; it is reprojected to the metric CRS.
    """
    import geopandas as gpd

    if not isinstance(tract_panel, gpd.GeoDataFrame) or not nyc_types:
        return {}
    gdf = tract_panel.to_crs(epsg=METRIC_EPSG)
    gdf = gdf.assign(cbsa_code=gdf["cbsa_code"].astype(str),
                     GEOID=gdf["GEOID"].astype(str))
    nyc = gdf[gdf["cbsa_code"] == NYC_CBSA]
    uncovered = nyc[~nyc["GEOID"].isin(nyc_types)]
    if uncovered.empty:
        return {}

    holdout_geoids = {g for g, t in nyc_types.items()
                      if t in ("test", "val_within_nyc")}
    hold = nyc[nyc["GEOID"].isin(holdout_geoids)]
    if hold.empty:
        in_buf = pd.Series(False, index=uncovered.index)
    else:
        buffer_geom = hold.geometry.union_all().buffer(dead_zone)
        in_buf = uncovered.geometry.intersects(buffer_geom)
    return {g: ("dead_zone" if b else "train")
            for g, b in zip(uncovered["GEOID"], in_buf)}


def build_tract_overrides(tract_panel) -> dict[str, str]:
    """Full ``{GEOID -> type}`` overlay applied on top of the whole-city split.

    Combines, in increasing precedence:
      1. clustered val carved from Chicago (``val_within_chicago`` + dead_zone),
      2. greater-NYC-CBSA → train (with a borough-holdout dead-zone buffer),
      3. the validated NYC notebook split for the five boroughs
         (train / ``val_within_nyc`` / test / dead_zone) — authoritative.

    A GEOID present here wins over its city's whole-city label in the
    building-assignment step.
    """
    nyc_types = nyc_tract_type_map()
    overrides = carve_test_city_val(tract_panel)
    overrides.update(nyc_greater_metro_overrides(tract_panel, nyc_types))
    overrides.update(nyc_types)             # boroughs (validated split) win.
    return overrides


def resolve_building_types(
    cbsa_codes: pd.Series,
    geoids: pd.Series,
    years: pd.Series | None,
    city_split_map: dict[str, str],
    holdout_map: dict[int, int],
    tract_overrides: dict[str, str],
) -> pd.Series:
    """Final per-building split ``type`` combining every layer.

    Precedence: tract-level override (NYC / clustered val) > whole-city split.
    Then, **when ``years`` is provided**, train-city buildings in that city's
    temporal-holdout year become ``val_temporal``. Values are in {train, test,
    val_cities, val_temporal, val_within_nyc, val_within_chicago, dead_zone,
    unassigned}. Index follows the inputs.

    Pass ``years=None`` on the lazy (static-buildings) path, where a building has
    no single year and ``val_temporal`` is materialized downstream via
    ``materialize_holdout_years`` instead — those buildings stay ``train`` here.

    Args:
        cbsa_codes/geoids/years: aligned building-level Series (cbsa & GEOID as
            str-able; year as int-able). ``years`` may be None (see above).
        city_split_map: ``{cbsa_code(str) -> train|val|test}`` (NYC absent).
        holdout_map: ``{cbsa_code(int) -> holdout_year}`` for train cities.
        tract_overrides: output of :func:`build_tract_overrides`.
    """
    cbsa = cbsa_codes.astype(str)
    geo = geoids.astype(str)

    city = cbsa.map(city_split_map)
    base = pd.Series("unassigned", index=cbsa.index, dtype=object)
    base[city == "train"] = "train"
    base[city == "val"] = "val_cities"
    base[city == "test"] = "test"

    override = geo.map(tract_overrides)
    final = override.where(override.notna(), base)

    if years is not None:
        # Temporal holdout: only real train CITIES (in holdout_map) — never NYC.
        holdout_year = cbsa.map({str(k): v for k, v in holdout_map.items()})
        is_holdout = (final == "train") & holdout_year.notna() & (
            pd.to_numeric(years, errors="coerce") == holdout_year.astype("float")
        )
        final[is_holdout] = "val_temporal"
    return final
