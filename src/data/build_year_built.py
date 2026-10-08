"""Build ``buildings_<city>.parquet``: dated footprints for the CSA event study (issue #36).

Consumes the raw downloads from :mod:`src.data.parcel_sources` and produces the
one artifact :func:`src.csa_event_study.available_cities` gates on — footprint
polygons plus a construction-year column, per city.

Output contract
---------------
A GeoParquet under ``data/processed/`` whose

* geometry is polygons in ``EPSG:5070`` (what ``build_tract_cohorts`` reprojects
  from and what the tract polygons already use),
* ``year_built`` column is nullable Int64,
* index is named ``building_id`` and holds the **Microsoft footprint id**.

That last point is the load-bearing one. ``building_id`` is what the prediction
CSVs key on, so indexing the footprints by the same id is what lets ``part_d``
join predictions to construction years and compute the composition-fixed
``pred_incumbent`` outcome — the one that shows an effect is not just new
buildings being added to the tract mean. Cities registered with
``id_index=None`` silently lose that second outcome, so ids are recomputed here
via :func:`src.data.build_buildings_index.compute_building_id` (the *same*
0.1 m-grid centroid packing the index itself uses) rather than carrying the
municipal id through.

Two join modes
--------------
``direct``
    The source is already dated building polygons (Chicago, NYC/DoITT). No
    spatial join: reproject, normalise the year, recompute ids.

``parcel_join``
    The source is parcel polygons with a year, which must be attached to
    Microsoft footprints from ``data/processed/buildings_polygons/<State>/``.
    Footprint centroid within parcel first, then nearest parcel within
    ``max_distance`` for the remainder (a centroid can fall in a right-of-way
    gap between parcel polygons). Lossy where one parcel carries several
    buildings of different vintages — recorded in the coverage report as
    ``buildings_per_matched_parcel``.

Sentinel handling
-----------------
Unknown years are ``NA``, never a number. ``build_tract_cohorts`` deliberately
counts unknown-year buildings in the *baseline* denominator ("unknown != ancient"
is an explicit comment there), so coercing a 0 sentinel to, say, 1800 would
inflate the baseline stock and shrink every tract's measured treatment intensity.
Implausible years (before ``MIN_PLAUSIBLE_YEAR`` or after the current year) are
also nulled rather than clipped, for the same reason.

Coverage report
---------------
Every build writes ``results/tables/csa_year_built_coverage_<city>.csv``. Read it
before trusting a city's cohorts: a low non-null ``year_built`` share or a thin
footprint-per-tract count means the cohorts are being dated off a biased subset,
which no downstream diagnostic will catch.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src import geo_utils
from src.data import parcel_sources as ps
from src.data.build_buildings_index import compute_building_id
from src.utils.paths import PROCESSED_DATA_DIR, TABLES_DIR

# Buildings older than this are almost certainly data-entry errors in an
# assessor file (year 1, year 190, etc). Nulled, not clipped — see module docstring.
MIN_PLAUSIBLE_YEAR = 1800

# A footprint centroid can land in a right-of-way sliver between parcel polygons.
# 25 m is roughly a city lot's half-width: wide enough to bridge a road-edge gap,
# narrow enough not to steal a neighbouring block's year.
NEAREST_MAX_DISTANCE_M = 25.0

METRIC_EPSG = geo_utils.METRIC_EPSG          # 5070


# --------------------------------------------------------------------------- #
# Normalisation                                                                #
# --------------------------------------------------------------------------- #

def normalize_years(values, *, max_year: int | None = None) -> pd.Series:
    """Coerce a raw year OR date column to nullable Int64, nulling the implausible.

    Sources are inconsistent about this: construction year is usually a number
    (``1995``), but demolition is often a timestamp (Chicago's ``demolished`` is
    an ISO date). Both are accepted — numeric parsing first, then datetime for
    whatever failed — so a caller never has to know which shape a column is.

    Zero/negative sentinels, blanks and out-of-range values all become ``NA``.
    Deliberately *not* clipped to a boundary year: an unknown year must stay
    unknown so ``build_tract_cohorts`` can keep it in the baseline denominator
    instead of treating it as a dated building.
    """
    if max_year is None:
        max_year = _dt.date.today().year
    raw = pd.Series(values).replace({"": None})
    s = pd.to_numeric(raw, errors="coerce")

    # Anything numeric parsing rejected may still be a date/timestamp.
    # format="mixed" is required, not cosmetic: pandas >= 2.0 infers ONE format
    # from the first element and applies it strictly to the rest, so a column
    # holding both "2015-03-01T00:00:00.000" and "1998-12-31" — which assessor
    # exports routinely do — silently coerces every value after the first shape
    # to NaT, and the demolition column comes back all-NA.
    unparsed = s.isna() & raw.notna()
    if unparsed.any():
        try:
            dates = pd.to_datetime(raw[unparsed], errors="coerce", utc=True,
                                   format="mixed")
        except (TypeError, ValueError):
            # pandas < 2.0 has no format="mixed"; its default is per-element.
            dates = pd.to_datetime(raw[unparsed], errors="coerce", utc=True)
        s.loc[unparsed] = dates.dt.year.astype("float64")

    s = s.where(np.isfinite(s.astype("float64")))
    s = s.where((s >= MIN_PLAUSIBLE_YEAR) & (s <= max_year))
    return s.round().astype("Int64")


def clean_geometry(gdf: gpd.GeoDataFrame, label: str) -> tuple[gpd.GeoDataFrame, dict]:
    """Drop unusable source geometry and repair invalid polygons.

    Municipal footprint layers carry a few placeholder records with null geometry
    and every attribute zeroed (Chicago has 6 in 820,606). Those cannot be
    reprojected, assigned to a tract, or given a ``building_id``, so they are
    dropped — loudly, and counted into the coverage report, because a silent drop
    is how a real coverage problem hides.

    Self-intersecting polygons are repaired rather than dropped. This is not
    cosmetic: treatment intensity is a *ratio of areas*, and a bow-tie polygon's
    ``.area`` counts its two lobes with opposite sign, so an unrepaired invalid
    footprint understates the stock it belongs to.
    """
    stats = {"n_input": len(gdf)}

    unusable = gdf.geometry.isna() | gdf.geometry.is_empty
    stats["n_dropped_null_geometry"] = int(unusable.sum())
    if unusable.any():
        print(f"    ⚠️ {label}: dropped {int(unusable.sum()):,} rows with null/empty "
              f"geometry (placeholder records in the source)")
        gdf = gdf[~unusable].copy()

    if len(gdf):
        bounds = gdf.geometry.bounds.to_numpy()
        nonfinite = ~np.isfinite(bounds).all(axis=1)
        stats["n_dropped_nonfinite_source"] = int(nonfinite.sum())
        if nonfinite.any():
            print(f"    ⚠️ {label}: dropped {int(nonfinite.sum()):,} rows whose "
                  f"SOURCE coordinates are already non-finite")
            gdf = gdf[~nonfinite].copy()
    else:
        stats["n_dropped_nonfinite_source"] = 0

    invalid = ~gdf.geometry.is_valid
    stats["n_repaired_invalid"] = int(invalid.sum())
    if invalid.any():
        print(f"    {label}: repairing {int(invalid.sum()):,} invalid geometries "
              f"(self-intersections distort footprint area)")
        geom_col = gdf.geometry.name
        try:
            gdf.loc[invalid, geom_col] = gdf.loc[invalid, geom_col].make_valid()
        except AttributeError:                      # geopandas < 0.13
            gdf.loc[invalid, geom_col] = gdf.loc[invalid, geom_col].buffer(0)

    stats["n_kept"] = len(gdf)
    return gdf, stats


def _to_metric(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    """Reproject to EPSG:5070, distinguishing a PROJ failure from bad input.

    The two look identical in the output — non-finite coordinates — but have
    opposite fixes, so they are told apart by what went IN. A row that was finite
    before the transform and is not after means PROJ could not build the
    pipeline, which is systemic (typically ``PROJ_NETWORK=ON`` with no route to
    the grid CDN) and must abort. A row that was already non-finite is a source
    data problem, handled by :func:`clean_geometry` before we get here.
    """
    if gdf.crs is None:
        raise ValueError(f"{label}: geometry has no CRS; refusing to guess")
    if not len(gdf):
        return gdf

    finite_before = np.isfinite(gdf.geometry.bounds.to_numpy()).all(axis=1)
    out = gdf.to_crs(geo_utils.METRIC_CRS)
    finite_after = np.isfinite(out.geometry.bounds.to_numpy()).all(axis=1)

    broken = finite_before & ~finite_after
    if broken.any():
        raise RuntimeError(
            f"reprojecting {label} to EPSG:{METRIC_EPSG} turned "
            f"{int(broken.sum()):,}/{int(finite_before.sum()):,} finite geometries "
            f"non-finite. PROJ could not build the transformation pipeline — if "
            f"PROJ_NETWORK=ON and this machine cannot reach cdn.proj.org, set "
            f"PROJ_NETWORK=OFF and retry."
        )
    return out


def _with_building_ids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Attach the Microsoft ``building_id`` from EPSG:5070 centroids.

    Same packing as the national index, so a footprint dated here and the same
    footprint predicted from the index share an id and join. Duplicate ids
    (centroids identical on the 0.1 m grid) are dropped keeping the first, which
    is what ``build_buildings_index`` does upstream.
    """
    cent = gdf.geometry.centroid
    gdf = gdf.copy()
    gdf["building_id"] = compute_building_id(cent.x.to_numpy(), cent.y.to_numpy())
    dupes = int(gdf["building_id"].duplicated().sum())
    if dupes:
        print(f"    dropped {dupes:,} duplicate building_id "
              f"(coincident centroids on the 0.1 m grid)")
        gdf = gdf[~gdf["building_id"].duplicated(keep="first")]
    return gdf.set_index("building_id")


# --------------------------------------------------------------------------- #
# Join modes                                                                   #
# --------------------------------------------------------------------------- #

def build_direct(raw: gpd.GeoDataFrame, source: ps.ParcelSource,
                 stats: dict | None = None) -> gpd.GeoDataFrame:
    """Dated building polygons -> the output contract, with no spatial join.

    Carries a demolition year through when the source has one. That column is
    load-bearing for the cohort denominator: a building demolished before the
    baseline year is not part of the baseline stock, and counting it would
    overstate every affected tract's denominator and so understate its treatment
    intensity — pushing tracts below the threshold and out of the treated group.
    """
    if source.year_col not in raw.columns:
        raise KeyError(f"{source.key}: no {source.year_col!r} column "
                       f"(has {sorted(raw.columns)[:12]}...)")
    cols = [source.year_col, "geometry"]
    demo = source.demolition_col
    if demo:
        if demo not in raw.columns:
            # Socrata omits all-null fields from its responses, so a declared
            # demolition column can legitimately be absent. Warn rather than
            # fail: the cohorts are still correct, just without demolitions.
            print(f"    ⚠️ {source.key}: no {demo!r} column in the download; "
                  f"proceeding without demolition dates")
            demo = None
        else:
            cols.insert(1, demo)

    clean, geom_stats = clean_geometry(raw[cols].copy(), source.key)
    if stats is not None:
        stats.update(geom_stats)
    gdf = _to_metric(clean, source.key)
    gdf["year_built"] = normalize_years(gdf.pop(source.year_col))
    out_cols = ["year_built", "geometry"]
    if demo:
        gdf["demolition_year"] = normalize_years(gdf.pop(demo))
        n_demo = int(gdf["demolition_year"].notna().sum())
        print(f"    {n_demo:,} footprints carry a demolition year")
        out_cols.insert(1, "demolition_year")
    return _with_building_ids(gdf)[out_cols]


# --------------------------------------------------------------------------- #
# Tabular year sources (CAMA extracts joined to parcel geometry)               #
# --------------------------------------------------------------------------- #

def parcel_key(frame: pd.DataFrame, cols, widths) -> pd.Series:
    """Zero-padded concatenation of key columns, as a string Series.

    King County stores its parcel id split across ``Major`` and ``Minor`` and
    pads NEITHER, so parcel 200660/1340 and parcel 20066/134 both concatenate to
    a 10-character string but describe different land. Left-padding each part to
    its documented width is what makes the key join to the GIS layer's ``PIN``,
    which *is* padded.

    A value wider than its declared width means the width is wrong for this
    source; that raises rather than truncating, because a silently truncated key
    joins to the wrong parcel instead of to none.
    """
    if len(cols) != len(widths):
        raise ValueError(f"cols {tuple(cols)} and widths {tuple(widths)} differ in length")
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"key columns {missing} absent (have {list(frame.columns)[:12]})")

    parts = []
    for col, width in zip(cols, widths):
        s = frame[col].astype("string").str.strip()
        too_long = s.str.len() > int(width)
        if too_long.any():
            sample = s[too_long].head(3).tolist()
            raise ValueError(
                f"{col}: {int(too_long.sum()):,} values exceed the declared width "
                f"{width} (e.g. {sample}); the key spec is wrong for this source")
        parts.append(s.str.zfill(int(width)))

    key = parts[0]
    for part in parts[1:]:
        key = key + part
    # Any null component makes the whole key meaningless, not merely shorter.
    null_any = np.zeros(len(frame), dtype=bool)
    for col in cols:
        null_any |= frame[col].isna().to_numpy()
    return key.mask(pd.Series(null_any, index=frame.index))


def building_year_table(tables: dict, sources: dict) -> pd.DataFrame:
    """Stack per-building CAMA extracts into ``[parcel_key, year_built, size]``.

    One row per *building*, not per parcel — which is the point. King County's
    residential extract covers 1-3 unit buildings and its commercial extract
    covers everything else including apartments, so only the union describes the
    building stock, and only per-building rows let
    :func:`parcel_year_from_buildings` tell a 1950 house with a new garage apart
    from a 1950 parcel redeveloped as a 2016 tower.
    """
    frames = []
    for key, frame in tables.items():
        source = sources[key]
        if source.year_col not in frame.columns:
            raise KeyError(f"{key}: no {source.year_col!r} column "
                           f"(has {list(frame.columns)[:12]})")
        out = pd.DataFrame(index=frame.index)
        out["parcel_key"] = parcel_key(frame, source.key_cols, source.key_widths)
        out["year_built"] = normalize_years(frame[source.year_col])
        if source.size_col and source.size_col in frame.columns:
            out["size"] = pd.to_numeric(frame[source.size_col], errors="coerce")
        else:
            out["size"] = np.nan
        out["source"] = key
        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["parcel_key", "year_built", "size", "source"])
    stacked = pd.concat(frames, ignore_index=True)
    return stacked[stacked["parcel_key"].notna()].reset_index(drop=True)


def parcel_year_from_buildings(bldgs: pd.DataFrame, *, agg: str = "dominant"
                               ) -> tuple[pd.DataFrame, dict]:
    """Collapse per-building rows to one construction year per parcel.

    ``build_parcel_join`` can only carry ONE year per parcel polygon, so a parcel
    holding buildings of several vintages has to be summarised. Which summary is
    used changes the treatment measure materially:

    ``"dominant"`` (default)
        The year of the parcel's LARGEST building, ties broken toward the oldest.
        A 1950 house with a 2016 garage stays 1950; a 1950 lot redeveloped as a
        2016 tower becomes 2016. This is the only rule that is right in both
        directions, and floor area is the right tiebreaker because treatment
        intensity is itself an area ratio.
    ``"min"``
        The oldest building. Understates treatment — redevelopment is missed —
        which attenuates the ATT toward zero.
    ``"max"``
        The newest. Dates the whole parcel's footprint area to the newest
        structure, so one new shed converts an entire block of 1920s stock into
        the current cohort. Available for sensitivity only; never a default.

    Returns ``(frame[parcel_key, year_built], diagnostics)``. The diagnostics
    quantify how much the choice can matter — if ``frac_parcels_multi_vintage``
    is small, all three rules agree almost everywhere.
    """
    if agg not in ("dominant", "min", "max"):
        raise ValueError(f"unknown agg {agg!r}; use dominant/min/max")

    dated = bldgs[bldgs["year_built"].notna()]
    if dated.empty:
        return (pd.DataFrame(columns=["parcel_key", "year_built"]),
                {"n_parcels_with_year": 0, "n_building_records": len(bldgs)})

    per_parcel_years = dated.groupby("parcel_key")["year_built"]
    spread = per_parcel_years.max() - per_parcel_years.min()
    n_parcels = int(spread.size)
    counts = per_parcel_years.size()

    if agg == "dominant":
        # NaN sizes sort last, so a parcel whose buildings have no recorded area
        # falls through to the year tiebreaker and takes its oldest — the
        # conservative choice, matching agg="min" exactly where size is unknown.
        ordered = dated.sort_values(
            ["parcel_key", "size", "year_built"],
            ascending=[True, False, True], na_position="last")
        picked = ordered.groupby("parcel_key", as_index=False).head(1)
        out = picked[["parcel_key", "year_built"]].reset_index(drop=True)
    else:
        out = (per_parcel_years.min() if agg == "min" else per_parcel_years.max())
        out = out.reset_index()

    diagnostics = {
        "parcel_year_agg": agg,
        "n_building_records": len(bldgs),
        "n_building_records_dated": len(dated),
        "n_parcels_with_year": n_parcels,
        "frac_parcels_multi_building": float((counts > 1).mean()),
        # The share of parcels where the aggregation rule actually bites. Every
        # rule agrees on the rest, so this bounds the sensitivity of the cohorts
        # to the choice above.
        "frac_parcels_multi_vintage": float((spread > 0).mean()),
        "median_vintage_spread_years": float(spread[spread > 0].median())
        if (spread > 0).any() else 0.0,
    }
    n_multi = int((spread > 0).sum())
    if n_multi:
        print(f"    {n_multi:,} of {n_parcels:,} parcels "
              f"({100 * n_multi / n_parcels:.1f}%) hold buildings of more than one "
              f"vintage; resolved by agg={agg!r}")
    return out.astype({"year_built": "Int64"}), diagnostics


def attach_table_years(parcels: gpd.GeoDataFrame, per_parcel: pd.DataFrame,
                       source) -> tuple[gpd.GeoDataFrame, dict]:
    """Join per-parcel years onto parcel geometry via ``source.join_key``.

    The match rate is returned rather than merely warned about: an unmatched
    parcel is not a neutral loss. It keeps its footprints in the tract's baseline
    denominator (undated buildings are baseline stock by construction) while
    contributing nothing to the numerator, so a bad key silently pushes every
    tract's treatment intensity down and out of the treated group.
    """
    if not source.join_key:
        raise ValueError(f"{source.key}: year_from set but no join_key")
    if source.join_key not in parcels.columns:
        raise KeyError(f"{source.key}: no {source.join_key!r} column in the parcel "
                       f"layer (has {list(parcels.columns)[:12]})")

    out = parcels.copy()
    keys = out[source.join_key].astype("string").str.strip()
    lut = (per_parcel.dropna(subset=["parcel_key"])
           .drop_duplicates("parcel_key")
           .set_index("parcel_key")["year_built"])
    out[source.year_col] = keys.map(lut).astype("Int64")

    n_matched = int(out[source.year_col].notna().sum())
    stats = {
        "n_parcel_polygons": len(out),
        "n_parcel_polygons_dated": n_matched,
        "parcel_year_match_fraction": n_matched / len(out) if len(out) else float("nan"),
    }
    print(f"    parcel year join: {n_matched:,} / {len(out):,} polygons dated "
          f"({100 * stats['parcel_year_match_fraction']:.1f}%)")
    if len(out) and stats["parcel_year_match_fraction"] < 0.5:
        print(f"    ⚠️ fewer than half the parcels matched a year. Check that "
              f"{source.join_key!r} and the extract's key columns "
              f"{source.key_cols} describe the same id with the same padding.")
    return out, stats


def load_year_tables(city: str, source, *, download: bool = True,
                     force: bool = False) -> tuple[pd.DataFrame, dict]:
    """Download (if needed) and collapse a geometry source's ``year_from`` tables."""
    tables, sources = {}, {}
    for key in source.year_from:
        table_source = ps.source_by_key(city, key)
        if download:
            ps.download_source(table_source, force=force)
        if not table_source.out_path.exists():
            raise FileNotFoundError(
                f"{table_source.out_path} missing — run "
                f"`python -m src.data.parcel_sources {city}` first.")
        tables[key] = pd.read_parquet(table_source.out_path)
        sources[key] = table_source
        print(f"    {key}: {len(tables[key]):,} building records")
    bldgs = building_year_table(tables, sources)
    return parcel_year_from_buildings(bldgs)


def city_bbox_4326(spec, *, processed_dir: Path = PROCESSED_DATA_DIR,
                   pad_deg: float = 0.02) -> tuple[float, float, float, float]:
    """WGS84 envelope of a city's tracts, for spatially filtering a parcel service.

    Derived from ``tract_splits.feather`` rather than hardcoded, so it stays
    correct if the split or the city's county set changes. ``pad_deg`` (~2 km)
    guards against a parcel whose centroid sits just outside a boundary tract.
    """
    splits = gpd.read_feather(Path(processed_dir) / "tract_splits.feather")
    splits["GEOID_str"] = splits["GEOID"].astype(str)
    sub = splits[splits["GEOID_str"].str.startswith(tuple(spec.geoid_prefixes))]
    if sub.empty:
        raise ValueError(f"{spec.key}: no tracts match {spec.geoid_prefixes}")
    xmin, ymin, xmax, ymax = sub.to_crs("EPSG:4326").total_bounds
    return (xmin - pad_deg, ymin - pad_deg, xmax + pad_deg, ymax + pad_deg)


def city_building_ids(spec, *, processed_dir: Path = PROCESSED_DATA_DIR
                      ) -> set[int]:
    """Microsoft ``building_id``s whose tract belongs to this city.

    Read from the hot ``buildings_index`` (id + tract only), so the city's
    footprint set is defined by the *tract* geography rather than by a bounding
    box. That matters because a bbox around Tampa also catches Manatee, Polk,
    Sumter, Citrus and Hardee parcels; filtering on tracts drops them exactly.
    """
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    part = Path(processed_dir) / "buildings_index" / f"state={spec.state}" / "part.parquet"
    if not part.exists():
        raise FileNotFoundError(
            f"no buildings_index partition for {spec.state!r} at {part}")
    expr = None
    for prefix in spec.geoid_prefixes:
        e = pc.starts_with(ds.field("tract_id"), str(prefix))
        expr = e if expr is None else (expr | e)
    table = ds.dataset(part).to_table(columns=["building_id"], filter=expr)
    return set(table.column("building_id").to_pylist())


def load_ms_polygons(state: str, *, processed_dir: Path = PROCESSED_DATA_DIR,
                     building_ids: set[int] | None = None,
                     verbose: bool = True) -> gpd.GeoDataFrame:
    """Microsoft footprint polygons for one state (EPSG:5070), keyed building_id.

    ``building_ids`` restricts the read to a city. Applied per part file rather
    than after concatenating: Florida's full polygon store is several GB, and a
    city needs a small fraction of it.
    """
    root = Path(processed_dir) / "buildings_polygons" / state
    if not root.exists():
        raise FileNotFoundError(
            f"no Microsoft polygons for {state!r} at {root}. Run "
            f"src/data/build_buildings_index.py first."
        )
    parts = sorted(root.glob("part_*.parquet")) or sorted(root.glob("*.parquet"))
    frames = []
    for p in parts:
        g = gpd.read_parquet(p)
        if building_ids is not None:
            g = g[g["building_id"].isin(building_ids)]
        if len(g):
            frames.append(g)
    if not frames:
        raise RuntimeError(f"no Microsoft polygons matched for {state!r}")
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                           geometry="geometry", crs=geo_utils.METRIC_CRS)
    if verbose:
        print(f"    {len(gdf):,} Microsoft footprints"
              + (" in this city" if building_ids is not None else f" in {state}"))
    return gdf.set_index("building_id")


def build_parcel_join(parcels: gpd.GeoDataFrame, footprints: gpd.GeoDataFrame,
                      source: ps.ParcelSource, *,
                      max_distance: float = NEAREST_MAX_DISTANCE_M
                      ) -> tuple[gpd.GeoDataFrame, dict]:
    """Attach parcel years to footprints; returns (dated footprints, join stats).

    Centroid-within first (unambiguous), then nearest-within-``max_distance`` for
    footprints whose centroid fell in a gap between parcel polygons.
    """
    if source.year_col not in parcels.columns:
        raise KeyError(f"{source.key}: no {source.year_col!r} column")
    parcels, _ = clean_geometry(parcels[[source.year_col, "geometry"]].copy(),
                                source.key)
    parcels = _to_metric(parcels, source.key)
    parcels["year_built"] = normalize_years(parcels.pop(source.year_col))
    parcels = parcels[["year_built", "geometry"]].reset_index(drop=True)

    cent = gpd.GeoDataFrame(geometry=footprints.geometry.centroid,
                            index=footprints.index, crs=footprints.crs)

    within = gpd.sjoin(cent, parcels, how="left", predicate="within")
    within = within[~within.index.duplicated(keep="first")]
    year = within["year_built"]
    n_within = int(year.notna().sum())

    missing = year.index[year.isna()]
    n_nearest = 0
    if len(missing):
        near = gpd.sjoin_nearest(cent.loc[missing], parcels, how="left",
                                 max_distance=max_distance,
                                 distance_col="_dist")
        near = near[~near.index.duplicated(keep="first")]
        year = year.copy()
        year.loc[near.index] = near["year_built"]
        n_nearest = int(near["year_built"].notna().sum())

    out = footprints.copy()
    out["year_built"] = year.astype("Int64")

    matched = int(out["year_built"].notna().sum())
    stats = {
        "n_footprints": len(out),
        "n_matched_within": n_within,
        "n_matched_nearest": n_nearest,
        "n_unmatched": len(out) - matched,
        "matched_fraction": matched / len(out) if len(out) else float("nan"),
        # >1 means one parcel's year is being shared by several buildings of
        # possibly different vintages — the main lossiness of this mode.
        "buildings_per_matched_parcel": (
            matched / max(1, int(parcels["year_built"].notna().sum()))
        ),
    }
    stats.update(footprint_vintage_cliff(parcels["year_built"], out["year_built"]))
    return out[["year_built", "geometry"]], stats


# Below this share of a construction year's parcels having a footprint, the
# footprint universe is treated as not covering that year.
VINTAGE_CLIFF_SHARE = 0.5


def footprint_vintage_cliff(parcel_years, footprint_years, *,
                            min_parcels: int = 500) -> dict:
    """Detect the year after which the footprint snapshot stops seeing construction.

    Microsoft's footprint universe is a STATIC snapshot, so buildings completed
    after its vintage simply do not exist as polygons — measured for Tampa at
    0.91-1.00 coverage for 2010-2018 but 0.08-0.24 for 2020-2024. That is
    invisible in every other diagnostic and is dangerous in a specific way: a
    tract that developed after the snapshot still *looks* developed to the model
    in later imagery, but is classified never-treated, so it lands in the control
    group and attenuates the ATT toward zero — i.e. it makes a responsive model
    look unresponsive.

    Returns the first year at which coverage falls below
    :data:`VINTAGE_CLIFF_SHARE` and stays there, so a city's ``panel_years`` can
    be capped just before it.
    """
    pc = pd.Series(parcel_years).dropna().astype(int).value_counts()
    fc = pd.Series(footprint_years).dropna().astype(int).value_counts()
    years = sorted(y for y in pc.index if pc[y] >= min_parcels)
    cliff = None
    for y in years:
        share = float(fc.get(y, 0)) / float(pc[y])
        if share < VINTAGE_CLIFF_SHARE:
            later = [yy for yy in years if yy > y]
            if all(float(fc.get(yy, 0)) / float(pc[yy]) < VINTAGE_CLIFF_SHARE
                   for yy in later):
                cliff = y
                break
    out = {"footprint_coverage_cliff_year": cliff}
    if cliff is not None:
        print(f"    ⚠️ footprint coverage falls below "
              f"{VINTAGE_CLIFF_SHARE:.0%} from {cliff} onward — the footprint "
              f"snapshot predates it. Cap this city's panel_years at "
              f"<= {cliff - 1}, or tracts developed after {cliff} will sit in "
              f"the never-treated control group and attenuate the ATT.")
    return out


# --------------------------------------------------------------------------- #
# Coverage report                                                              #
# --------------------------------------------------------------------------- #

def coverage_report(gdf: gpd.GeoDataFrame, city: str, *,
                    tracts: gpd.GeoDataFrame | None = None,
                    join_stats: dict | None = None) -> pd.DataFrame:
    """Diagnostics that decide whether a city's cohorts are trustworthy.

    The decisive numbers are the non-null ``year_built`` share (cohorts are dated
    off that subset only) and, when tracts are supplied, how many tracts have
    enough footprints to give a stable baseline area.
    """
    yb = gdf["year_built"]
    rows = {
        "city": city,
        "n_footprints": len(gdf),
        "n_year_built": int(yb.notna().sum()),
        "year_built_share": float(yb.notna().mean()) if len(gdf) else float("nan"),
        "year_min": int(yb.min()) if yb.notna().any() else None,
        "year_max": int(yb.max()) if yb.notna().any() else None,
        "n_built_2000plus": int((yb >= 2000).sum()),
        "n_built_2010plus": int((yb >= 2010).sum()),
    }
    if join_stats:
        rows.update(join_stats)
    if tracts is not None and len(tracts):
        assigned = gpd.sjoin(
            gpd.GeoDataFrame(geometry=gdf.geometry.centroid, crs=gdf.crs),
            tracts[["GEOID_str", "geometry"]], how="inner", predicate="within")
        per_tract = assigned.groupby("GEOID_str").size()
        rows.update({
            "n_tracts_with_footprints": int(per_tract.size),
            "median_footprints_per_tract": float(per_tract.median()),
            "n_tracts_under_50_footprints": int((per_tract < 50).sum()),
        })
    return pd.DataFrame([rows])


def year_histogram(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Footprint counts by construction year — eyeball this for spikes.

    Assessor files often park unknown vintages on a round year (1900, 1950),
    which would create a spurious mega-cohort. A spike is a reason to null that
    year, not to proceed.
    """
    yb = gdf["year_built"].dropna().astype(int)
    return (yb.value_counts().sort_index().rename("n_footprints")
            .rename_axis("year_built").reset_index())


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #

def build_city_footprints(city: str, *, processed_dir: Path = PROCESSED_DATA_DIR,
                          tables_dir: Path = TABLES_DIR,
                          state: str | None = None,
                          tracts: gpd.GeoDataFrame | None = None,
                          download: bool = True, force: bool = False) -> Path:
    """Build and write ``buildings_<city>.parquet`` plus its coverage report.

    Returns the output path. The filename is taken from the CSA registry so the
    artifact lands exactly where ``available_cities`` looks for it.
    """
    from src.csa_event_study import CSA_CITIES

    spec = CSA_CITIES.get(city)
    if spec is None:
        raise KeyError(f"{city!r} is not in CSA_CITIES")
    out_path = Path(processed_dir) / spec.footprints_filename
    if out_path.exists() and not force:
        print(f"✔ {city}: {out_path.name} already exists (use force=True to rebuild)")
        return out_path

    sources = ps.sources_for(city, primary_only=True)
    if not sources:
        raise KeyError(f"no primary parcel source registered for {city!r}")
    source = sources[0]

    print(f"\n=== Building {out_path.name} from {source.label} ===")
    if download:
        kwargs = {}
        if source.kind in ("arcgis", "http_archive"):
            # Statewide sources (Florida is 10.8M parcels) must be cut to the
            # city. For an archive that is a GDAL bbox push-down at read time;
            # for a service it is a query envelope. Either way the box comes
            # from the city's own tracts, never a hardcoded county-code table.
            bbox = city_bbox_4326(spec, processed_dir=processed_dir)
            print(f"  spatial filter: bbox {tuple(round(v, 4) for v in bbox)}")
            kwargs["bbox"] = bbox
        ps.download_source(source, force=force, **kwargs)
    if not source.out_path.exists():
        raise FileNotFoundError(
            f"{source.out_path} missing — run "
            f"`python -m src.data.parcel_sources {city}` first."
        )

    # Geometry-cleaning counts (null/non-finite dropped, invalid repaired) belong
    # in the coverage report next to the year-built share: both are ways the
    # footprint universe can be thinner than it looks.
    join_stats: dict = {}
    if source.geom_kind == "footprint":
        raw = gpd.read_parquet(source.out_path)
        gdf = build_direct(raw, source, stats=join_stats)
    else:
        if state is None:
            raise ValueError(
                f"{city}: parcel_join mode needs `state=` to locate the "
                f"Microsoft polygons partition")
        parcels = gpd.read_parquet(source.out_path)
        if source.year_from:
            # The geometry layer carries no year of its own (King County's is
            # literally just PIN + shape), so the CAMA extracts have to be joined
            # on before the spatial join can date anything.
            per_parcel, year_stats = load_year_tables(
                city, source, download=download, force=force)
            parcels, attach_stats = attach_table_years(parcels, per_parcel, source)
            join_stats.update(year_stats)
            join_stats.update(attach_stats)
        # Scope the footprints to the city's TRACTS, not the state or the bbox:
        # the download envelope necessarily spills into neighbouring counties,
        # and a statewide polygon store is several GB.
        ids = city_building_ids(spec, processed_dir=processed_dir)
        print(f"  city footprint universe: {len(ids):,} buildings "
              f"across tracts matching {spec.geoid_prefixes}")
        footprints = load_ms_polygons(state, processed_dir=processed_dir,
                                      building_ids=ids)
        gdf, parcel_stats = build_parcel_join(parcels, footprints, source)
        join_stats.update(parcel_stats)

    gdf.to_parquet(out_path)
    print(f"✔ {len(gdf):,} dated footprints -> {out_path}")

    report = coverage_report(gdf, city, tracts=tracts, join_stats=join_stats)
    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(tables_dir) / f"csa_year_built_coverage_{city}.csv"
    report.to_csv(report_path, index=False)
    hist_path = Path(tables_dir) / f"csa_year_built_histogram_{city}.csv"
    year_histogram(gdf).to_csv(hist_path, index=False)
    print(f"  coverage report -> {report_path}")
    print(report.T.to_string(header=False))
    return out_path


if __name__ == "__main__":                                  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("city")
    ap.add_argument("--state", default=None,
                    help="Microsoft polygons partition (parcel_join mode only)")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    build_city_footprints(args.city, state=args.state,
                          download=not args.no_download, force=args.force)
