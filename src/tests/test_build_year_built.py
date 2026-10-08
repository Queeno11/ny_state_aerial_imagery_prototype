"""Synthetic-data tests for src/data/build_year_built.py and parcel_sources.py.

Hand-built fixtures only: a handful of square parcels and footprints whose
correct join is obvious by eye, and fake HTTP payloads for the paginators. No
network, no real assessor files.

What matters here:

1. **Sentinels stay unknown.** ``build_tract_cohorts`` counts unknown-year
   buildings in the *baseline* denominator ("unknown != ancient"), so coercing a
   0 sentinel into a real year would inflate every tract's baseline stock and
   shrink its measured treatment intensity — biasing the ATT toward zero with
   nothing raising.
2. **building_id round-trips.** The id is what joins predictions to footprints;
   get it wrong and the composition-fixed incumbent outcome silently disappears.
3. **Paginators terminate and cover exactly once.** A repeated page would
   double-count footprints in a tract's baseline area; a dropped page would
   under-count it.
"""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import box

from src.data import build_year_built as byb
from src.data import parcel_sources as ps
from src.data.build_buildings_index import compute_building_id, unpack_building_id

X0, Y0 = 700_000.0, 2_100_000.0     # somewhere sane in EPSG:5070


@pytest.fixture(autouse=True)
def offline_proj():
    """Ballpark transforms only — see test_ortho_fetcher for the same guard."""
    was = pyproj.network.is_network_enabled()
    pyproj.network.set_network_enabled(False)
    yield
    pyproj.network.set_network_enabled(was)


def _squares(n, size=20.0, gap=30.0, crs=5070, x0=X0, y0=Y0):
    geoms = [box(x0 + i * gap, y0, x0 + i * gap + size, y0 + size) for i in range(n)]
    return gpd.GeoDataFrame(geometry=geoms, crs=f"EPSG:{crs}")


# ─── year normalisation ───────────────────────────────────────────────────────

def test_zero_sentinel_becomes_na_not_a_year():
    """REGRESSION GUARD: DoITT-style files use 0 for 'unknown'. Mapping it to a
    real year would move those buildings out of the baseline denominator and
    into a construction cohort."""
    out = byb.normalize_years([0, 1995, None, ""])
    assert pd.isna(out[0]) and pd.isna(out[2]) and pd.isna(out[3])
    assert out[1] == 1995


def test_implausible_years_are_nulled_not_clipped():
    """Clipping to MIN_PLAUSIBLE_YEAR would turn a typo into a dated 1800
    building, which then counts as real baseline stock."""
    out = byb.normalize_years([1, 190, 1799, 1800, 3000])
    assert list(pd.isna(out)) == [True, True, True, False, True]
    assert out[3] == 1800


def test_future_years_are_nulled():
    import datetime as dt
    future = dt.date.today().year + 5
    assert pd.isna(byb.normalize_years([future])[0])


def test_normalize_returns_nullable_int():
    out = byb.normalize_years([1990.0, np.nan])
    assert out.dtype == "Int64"
    assert out[0] == 1990


def test_normalize_extracts_the_year_from_dates():
    """Construction year is a number but demolition is often a timestamp
    (Chicago's `demolished` is an ISO date). Both must work, or the demolition
    column silently parses to all-NA and pre-baseline teardowns stay in the
    baseline stock."""
    out = byb.normalize_years(["2015-03-01T00:00:00.000", "1998-12-31", None])
    assert out[0] == 2015
    assert out[1] == 1998
    assert pd.isna(out[2])


def test_normalize_handles_mixed_numbers_and_dates():
    out = byb.normalize_years([1990, "2015-03-01T00:00:00.000", 0, "garbage"])
    assert out[0] == 1990
    assert out[1] == 2015
    assert pd.isna(out[2]) and pd.isna(out[3])


# ─── building ids ─────────────────────────────────────────────────────────────

def test_building_id_round_trips_to_the_centroid():
    """The id is the join key between predictions and footprints; if it did not
    reproduce the index's packing, part_d would find zero matches and quietly
    drop the composition-fixed outcome."""
    gdf = _squares(4)
    out = byb._with_building_ids(gdf)
    cx, cy = unpack_building_id(out.index.to_numpy())
    expected = gdf.geometry.centroid
    np.testing.assert_allclose(cx, expected.x.to_numpy(), atol=0.05)
    np.testing.assert_allclose(cy, expected.y.to_numpy(), atol=0.05)


def test_building_id_matches_the_national_index_packing():
    gdf = _squares(3)
    out = byb._with_building_ids(gdf)
    cent = gdf.geometry.centroid
    expected = compute_building_id(cent.x.to_numpy(), cent.y.to_numpy())
    np.testing.assert_array_equal(out.index.to_numpy(), expected)


def test_coincident_centroids_are_deduplicated():
    gdf = gpd.GeoDataFrame(geometry=[box(X0, Y0, X0 + 10, Y0 + 10)] * 3,
                           crs="EPSG:5070")
    out = byb._with_building_ids(gdf)
    assert len(out) == 1


def test_index_is_named_building_id():
    out = byb._with_building_ids(_squares(2))
    assert out.index.name == "building_id"


# ─── direct mode ──────────────────────────────────────────────────────────────

def _direct_source():
    return ps.ParcelSource(key="k", city="chicago", label="L", kind="socrata",
                           url="https://example.org", dataset_id="aaaa-bbbb",
                           year_col="year_built", geom_kind="footprint")


def test_null_geometry_rows_are_dropped_and_counted():
    """Chicago's layer carries 6 placeholder records (null geometry, every
    attribute zeroed) in 820,606. They cannot be reprojected, tract-assigned or
    given a building_id — so drop them, but count them, because a silent drop is
    how a real coverage problem hides."""
    gdf = _squares(4)
    gdf["year_built"] = [1990, 0, 2015, 2020]
    gdf.loc[1, "geometry"] = None
    out, stats = byb.clean_geometry(gdf, "test")
    assert len(out) == 3
    assert stats["n_dropped_null_geometry"] == 1
    assert stats["n_kept"] == 3


def test_invalid_polygons_are_repaired_not_dropped():
    """REGRESSION GUARD: treatment intensity is a RATIO OF AREAS, and a bow-tie
    polygon's .area counts its two lobes with opposite sign. Dropping the
    footprint loses stock; leaving it unrepaired understates it."""
    from shapely.geometry import Polygon

    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    assert not bowtie.is_valid
    gdf = gpd.GeoDataFrame({"year_built": [1990]}, geometry=[bowtie],
                           crs="EPSG:5070")
    out, stats = byb.clean_geometry(gdf, "test")
    assert len(out) == 1                       # kept, not dropped
    assert stats["n_repaired_invalid"] == 1
    assert out.geometry.iloc[0].is_valid
    assert out.geometry.iloc[0].area > 0


def test_clean_geometry_is_a_noop_on_clean_input():
    gdf = _squares(3)
    out, stats = byb.clean_geometry(gdf, "test")
    assert len(out) == 3
    assert stats["n_dropped_null_geometry"] == 0
    assert stats["n_repaired_invalid"] == 0


def test_to_metric_raises_when_finite_input_comes_back_non_finite(monkeypatch):
    """A row finite going IN and non-finite coming OUT means PROJ could not build
    the pipeline — systemic, and the PROJ_NETWORK hint is the right advice."""
    from shapely.geometry import Polygon

    def fake_to_crs(self, *a, **k):
        return gpd.GeoDataFrame(
            geometry=[Polygon([(np.inf, np.inf), (np.inf, 1), (1, 1)])] * len(self),
            crs="EPSG:5070")

    monkeypatch.setattr(gpd.GeoDataFrame, "to_crs", fake_to_crs)
    with pytest.raises(RuntimeError, match="PROJ_NETWORK"):
        byb._to_metric(_squares(2), "test")


def test_to_metric_does_not_blame_proj_for_bad_source_rows():
    """REGRESSION: the real Chicago run aborted on 6 null-geometry rows with a
    message telling the user to set PROJ_NETWORK=OFF — the wrong fix for the
    wrong cause. After clean_geometry removes them, reprojection must succeed."""
    gdf = _squares(4)
    gdf["year_built"] = [1990, 1991, 2015, 2020]
    gdf.loc[1, "geometry"] = None
    clean, _ = byb.clean_geometry(gdf, "test")
    out = byb._to_metric(clean, "test")
    assert len(out) == 3
    assert out.crs.to_epsg() == 5070
    assert np.isfinite(out.geometry.bounds.to_numpy()).all()


def test_build_direct_drops_null_geometry_end_to_end():
    """The real Chicago failure: 6 null-geometry rows aborted the whole build
    with a message blaming PROJ."""
    raw = _squares(4)
    raw["year_built"] = [1990, 1991, 2015, 2020]
    raw.loc[2, "geometry"] = None
    stats = {}
    out = byb.build_direct(raw, _direct_source(), stats=stats)
    assert len(out) == 3
    assert stats["n_dropped_null_geometry"] == 1


def test_build_direct_produces_the_output_contract():
    raw = _squares(3)
    raw["year_built"] = [1990, 0, 2015]
    out = byb.build_direct(raw, _direct_source())
    assert out.index.name == "building_id"
    assert list(out.columns) == ["year_built", "geometry"]
    assert out.crs.to_epsg() == 5070
    assert out["year_built"].tolist()[0] == 1990
    assert pd.isna(out["year_built"].tolist()[1])


def test_build_direct_reprojects_to_5070():
    raw = _squares(2, size=0.001, gap=0.002, crs=4326, x0=-87.63, y0=41.88)
    raw["year_built"] = [2001, 2002]
    out = byb.build_direct(raw, _direct_source())
    assert out.crs.to_epsg() == 5070
    assert np.isfinite(out.geometry.bounds.to_numpy()).all()


def test_build_direct_carries_a_demolition_year_through():
    """A building demolished before the baseline year is not baseline stock;
    counting it would overstate the tract's denominator and understate its
    treatment intensity, pushing tracts out of the treated group."""
    source = ps.ParcelSource(key="k", city="chicago", label="L", kind="socrata",
                             url="https://example.org", dataset_id="a-b",
                             year_col="year_built", demolition_col="demolished",
                             geom_kind="footprint")
    raw = _squares(3)
    raw["year_built"] = [1950, 1960, 2015]
    raw["demolished"] = ["2012-06-01T00:00:00.000", None, None]
    out = byb.build_direct(raw, source)
    assert "demolition_year" in out.columns
    assert out["demolition_year"].iloc[0] == 2012
    assert pd.isna(out["demolition_year"].iloc[1])


def test_build_direct_tolerates_an_absent_demolition_column():
    """Socrata omits all-null fields from its responses, so a declared demolition
    column can legitimately be missing — that must not fail the build."""
    source = ps.ParcelSource(key="k", city="chicago", label="L", kind="socrata",
                             url="https://example.org", dataset_id="a-b",
                             year_col="year_built", demolition_col="demolished",
                             geom_kind="footprint")
    raw = _squares(2)
    raw["year_built"] = [1990, 1991]
    out = byb.build_direct(raw, source)          # no 'demolished' column at all
    assert list(out.columns) == ["year_built", "geometry"]


def test_chicago_declares_no_demolition_data():
    """REGRESSION GUARD: Chicago's `demolished` column is a SENTINEL — all 124
    non-null values are the year 1899, and bldg_statu has 1 DEMOLISHED row in
    820,606. Declaring it would date those 124 into an 1899 cohort. The registry
    must say None, whatever the source nominally offers."""
    from src import csa_event_study as ces
    assert ces.CSA_CITIES["chicago"].demolition_col is None


def test_registry_demolition_columns_match_what_the_builder_emits():
    """A spec's demolition_col must be a column build_year_built actually writes
    ('demolition_year'), or part_d silently loses demolitions."""
    from src import csa_event_study as ces
    for key, spec in ces.CSA_CITIES.items():
        if spec.demolition_col is None:
            continue
        assert spec.demolition_col in {"demolition_year", "DEMOLITION_YEAR"}, key


def test_tampa_is_the_second_main_figure_city():
    """Chicago's dated-footprint source is a frozen 2015 snapshot, so it cannot
    date cohorts over a 2010-2024 panel; Tampa's statewide cadastral is
    maintained and covers all property classes."""
    from src import csa_event_study as ces
    assert ces.MAIN_FIGURE_CITIES == ("nyc", "tampa")
    assert "chicago" not in ces.MAIN_FIGURE_CITIES


def test_tampa_has_a_registered_maintained_parcel_source():
    src = ps.sources_for("tampa", primary_only=True)
    assert len(src) == 1
    s = src[0]
    # Bulk archive, not the ArcGIS service: that service rejects returnIdsOnly,
    # returnCountOnly and geometry filters, and CO_NO is unindexed.
    assert s.kind == "http_archive" and s.geom_kind == "parcel"
    # ACT_YR_BLT (actual), not EFF_YR_BLT: the "effective" year is reset by major
    # renovation and would date a rehabbed 1920 house into a recent cohort.
    assert s.year_col == "ACT_YR_BLT"
    assert "EFF_YR_BLT" not in s.select
    assert {"PARCEL_ID", "ACT_YR_BLT"} <= set(s.select)


def test_embedded_arcgis_error_in_a_200_raises_instead_of_reading_empty():
    """REGRESSION GUARD: ArcGIS answers a rejected query with HTTP 200 and an
    {"error": ...} body. Unchecked, the caller sees no 'features'/'objectIds'
    key and reads it as an EMPTY RESULT — which is exactly how the Florida
    query surfaced as '0 features in the selection' and a zero-row download
    rather than an error."""
    class Sess:
        def get(self, url, params=None, timeout=None):
            return _Resp(200, payload={"error": {
                "code": 400, "message": "Cannot perform query.",
                "details": ["Unable to perform query."]}})

    with pytest.raises(RuntimeError, match="embedded error"):
        ps._get_json(Sess(), "https://example.org", {}, sleep_fn=lambda s: None)


def test_keyset_paging_advances_by_the_largest_id_seen(monkeypatch, tmp_path):
    """Keyset (OBJECTID > last) rather than resultOffset: some services reject
    id-only queries outright, and deep offsets degrade badly on a 10.8M-row
    layer. Terminating on an empty page also survives a resultRecordCount clamp."""
    source = ps.ParcelSource(key="p", city="tampa", label="L", kind="arcgis",
                             url="https://example.org/0", geom_kind="parcel",
                             select=("OBJECTID", "ACT_YR_BLT"))
    monkeypatch.setattr(type(source), "cache_dir",
                        property(lambda self: tmp_path / "pages"))
    wheres = []
    served = {"n": 0}

    def fake_get_json(session, url, params, sleep_fn=None):
        if params.get("f") == "json":
            return {"maxRecordCount": 2}
        wheres.append(params["where"])
        served["n"] += 1
        if served["n"] > 3:
            return {"type": "FeatureCollection", "features": []}
        base = (served["n"] - 1) * 2 + 1
        return {"type": "FeatureCollection", "features": [
            {"type": "Feature", "id": base + i,
             "geometry": {"type": "Point", "coordinates": [-82.5, 27.9]},
             "properties": {"OBJECTID": base + i, "ACT_YR_BLT": 2019}}
            for i in range(2)]}

    monkeypatch.setattr(ps, "_get_json", fake_get_json)
    out = ps.fetch_arcgis(source, session=object(), verbose=False)
    assert len(out) == 6
    assert "OBJECTID > -1" in wheres[0]
    assert "OBJECTID > 2" in wheres[1]
    assert "OBJECTID > 4" in wheres[2]


def test_chicago_spec_records_why_it_is_blocked():
    """The note is what part_d prints when the city is skipped; it must name the
    actual cause so the next person does not re-discover the 2015 freeze."""
    from src import csa_event_study as ces
    note = ces.CSA_CITIES["chicago"].note
    assert "2015" in note and "BLOCKED" in note


def test_chicago_select_uses_the_real_truncated_field_names():
    """REGRESSION GUARD: the layer's machine names are the truncated
    shapefile-era ones (bldg_statu / no_of_unit / bldg_sq_fo), not the
    spelled-out display labels. Socrata answers a wrong name with a 400 for the
    whole request, so one bad name breaks the entire download."""
    select = set(ps.sources_for("chicago", primary_only=True)[0].select)
    assert {"bldg_statu", "no_of_unit", "bldg_sq_fo"} <= select
    assert not ({"bldg_status", "no_of_units", "bldg_sq_footage"} & select)
    assert {"bldg_id", "year_built", "demolished", "the_geom"} <= select


def test_build_direct_names_the_missing_column():
    raw = _squares(2)
    raw["built"] = [1990, 1991]
    with pytest.raises(KeyError, match="year_built"):
        byb.build_direct(raw, _direct_source())


# ─── parcel join ──────────────────────────────────────────────────────────────

def _parcel_source():
    return ps.ParcelSource(key="p", city="seattle", label="L", kind="arcgis",
                           url="https://example.org/0", year_col="year_built",
                           geom_kind="parcel")


def test_parcel_join_matches_footprints_inside_parcels():
    parcels = _squares(3, size=100.0, gap=200.0)
    parcels["year_built"] = [1970, 1985, 2012]
    # One small footprint centred inside each parcel.
    foot = gpd.GeoDataFrame(
        geometry=[box(X0 + i * 200 + 40, Y0 + 40, X0 + i * 200 + 60, Y0 + 60)
                  for i in range(3)], crs="EPSG:5070")
    foot = byb._with_building_ids(foot)

    out, stats = byb.build_parcel_join(parcels, foot, _parcel_source())
    assert out["year_built"].tolist() == [1970, 1985, 2012]
    assert stats["n_matched_within"] == 3
    assert stats["matched_fraction"] == 1.0


def test_parcel_join_falls_back_to_nearest_within_the_distance_cap():
    """A footprint centroid can land in a right-of-way sliver between parcels."""
    parcels = _squares(1, size=100.0)
    parcels["year_built"] = [1960]
    # 10 m to the right of the parcel's edge — outside, but well within 25 m.
    foot = gpd.GeoDataFrame(
        geometry=[box(X0 + 105, Y0 + 40, X0 + 115, Y0 + 50)], crs="EPSG:5070")
    foot = byb._with_building_ids(foot)

    out, stats = byb.build_parcel_join(parcels, foot, _parcel_source())
    assert out["year_built"].tolist() == [1960]
    assert stats["n_matched_within"] == 0
    assert stats["n_matched_nearest"] == 1


def test_parcel_join_leaves_far_footprints_unmatched():
    parcels = _squares(1, size=100.0)
    parcels["year_built"] = [1960]
    foot = gpd.GeoDataFrame(
        geometry=[box(X0 + 5000, Y0, X0 + 5010, Y0 + 10)], crs="EPSG:5070")
    foot = byb._with_building_ids(foot)

    out, stats = byb.build_parcel_join(parcels, foot, _parcel_source())
    assert pd.isna(out["year_built"].iloc[0])
    assert stats["n_unmatched"] == 1


def test_parcel_join_shares_one_year_across_a_multi_building_parcel():
    """Documented lossiness: a parcel with a 1920 house and a 2015 addition has
    one year, and every building on it inherits it. The coverage report surfaces
    this as buildings_per_matched_parcel > 1."""
    parcels = _squares(1, size=200.0)
    parcels["year_built"] = [1920]
    foot = gpd.GeoDataFrame(
        geometry=[box(X0 + 20 + 40 * i, Y0 + 20, X0 + 40 + 40 * i, Y0 + 40)
                  for i in range(4)], crs="EPSG:5070")
    foot = byb._with_building_ids(foot)

    out, stats = byb.build_parcel_join(parcels, foot, _parcel_source())
    assert (out["year_built"] == 1920).all()
    assert stats["buildings_per_matched_parcel"] == 4.0


def test_parcel_join_preserves_unknown_years():
    parcels = _squares(2, size=100.0, gap=200.0)
    parcels["year_built"] = [0, 1999]
    foot = gpd.GeoDataFrame(
        geometry=[box(X0 + i * 200 + 40, Y0 + 40, X0 + i * 200 + 60, Y0 + 60)
                  for i in range(2)], crs="EPSG:5070")
    foot = byb._with_building_ids(foot)

    out, _ = byb.build_parcel_join(parcels, foot, _parcel_source())
    assert pd.isna(out["year_built"].iloc[0])
    assert out["year_built"].iloc[1] == 1999


# ─── coverage report ──────────────────────────────────────────────────────────

def test_vintage_cliff_detects_a_static_footprint_snapshot():
    """REGRESSION GUARD: Microsoft footprints are a static ~2019 snapshot, so
    post-snapshot construction has no polygon. Measured for Tampa: 0.91-1.00
    coverage through 2018, 0.08-0.24 from 2020. Undetected, those tracts sit in
    the NEVER-TREATED group while still looking developed in later imagery,
    attenuating the ATT toward zero — making a responsive model look dead."""
    parcels, foots = [], []
    for y in range(2010, 2025):
        parcels += [y] * 1000
        foots += [y] * (950 if y <= 2018 else 100)     # cliff at 2019
    out = byb.footprint_vintage_cliff(pd.Series(parcels), pd.Series(foots))
    assert out["footprint_coverage_cliff_year"] == 2019


def test_vintage_cliff_is_none_when_coverage_holds():
    parcels, foots = [], []
    for y in range(2010, 2025):
        parcels += [y] * 1000
        foots += [y] * 900
    out = byb.footprint_vintage_cliff(pd.Series(parcels), pd.Series(foots))
    assert out["footprint_coverage_cliff_year"] is None


def test_vintage_cliff_ignores_a_single_thin_year():
    """One sparse year mid-panel is noise, not a snapshot boundary; only a drop
    that PERSISTS to the end of the record is a vintage cliff."""
    parcels, foots = [], []
    for y in range(2010, 2025):
        parcels += [y] * 1000
        foots += [y] * (100 if y == 2015 else 950)
    out = byb.footprint_vintage_cliff(pd.Series(parcels), pd.Series(foots))
    assert out["footprint_coverage_cliff_year"] is None


def test_msbased_cities_stop_before_the_snapshot_cliff():
    """Every city whose cohorts come from Microsoft footprints must end its panel
    before post-snapshot construction becomes invisible."""
    from src import csa_event_study as ces
    for key in ("tampa", "seattle", "nashville"):
        spec = ces.CSA_CITIES[key]
        assert max(spec.panel_years) <= 2019, (key, spec.panel_years)
    # NYC is exempt: DoITT is a maintained cadastral, not a static snapshot.
    assert max(ces.CSA_CITIES["nyc"].panel_years) == 2024


def test_coverage_report_surfaces_the_dated_share():
    gdf = _squares(4)
    gdf["year_built"] = pd.array([1990, None, 2015, 2020], dtype="Int64")
    gdf = byb._with_building_ids(gdf)
    rep = byb.coverage_report(gdf, "chicago")
    assert rep.loc[0, "n_footprints"] == 4
    assert rep.loc[0, "n_year_built"] == 3
    assert rep.loc[0, "year_built_share"] == pytest.approx(0.75)
    assert rep.loc[0, "n_built_2010plus"] == 2


def test_year_histogram_would_expose_a_round_year_spike():
    """Assessor files often park unknown vintages on 1900/1950, which would
    become a spurious mega-cohort. The histogram is how that gets caught."""
    gdf = _squares(6)
    gdf["year_built"] = pd.array([1900] * 5 + [2014], dtype="Int64")
    gdf = byb._with_building_ids(gdf)
    hist = byb.year_histogram(gdf).set_index("year_built")["n_footprints"]
    assert hist.loc[1900] == 5
    assert hist.loc[2014] == 1


# ─── paginators ───────────────────────────────────────────────────────────────

class FakeSocrata:
    """Serves ``total`` rows in pages, recording the params it was asked for.

    ``server_cap`` emulates Socrata clamping ``$limit`` below what was asked.
    """

    def __init__(self, total, fmt="geojson", server_cap=None):
        self.total = total
        self.fmt = fmt
        self.server_cap = server_cap
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params or {}))
        offset = int(params["$offset"])
        limit = int(params["$limit"])
        if self.server_cap is not None:
            limit = min(limit, self.server_cap)
        n = max(0, min(limit, self.total - offset))
        feats = [{"type": "Feature",
                  "geometry": {"type": "Point", "coordinates": [-87.6, 41.9]},
                  "properties": {"year_built": 2000 + (offset + i) % 5,
                                 "bldg_id": offset + i}}
                 for i in range(n)]
        payload = ({"type": "FeatureCollection", "features": feats}
                   if self.fmt == "geojson"
                   else [f["properties"] for f in feats])

        class R:
            status_code = 200

            @staticmethod
            def raise_for_status():
                pass

            @staticmethod
            def json():
                return payload
        return R


def test_socrata_paginator_terminates_and_covers_exactly_once(tmp_path,
                                                              monkeypatch):
    source = ps.ParcelSource(key="k", city="chicago", label="L", kind="socrata",
                             url="https://example.org", dataset_id="a-b",
                             geom_kind="footprint")
    monkeypatch.setattr(ps, "PARCELS_DIR", tmp_path)
    monkeypatch.setattr(type(source), "cache_dir",
                        property(lambda self: tmp_path / "pages"))
    sess = FakeSocrata(total=250)
    out = ps.fetch_socrata(source, page_size=100, session=sess, verbose=False)
    assert len(out) == 250
    assert len(set(out["bldg_id"])) == 250          # no page repeated
    # Offsets advance by rows actually returned, and the loop terminates only on
    # an empty page — hence the trailing probe at 250. That extra request is the
    # price of being safe against a server-side $limit clamp.
    assert [c["$offset"] for c in sess.calls] == [0, 100, 200, 250]


def test_socrata_survives_a_server_side_limit_clamp(tmp_path, monkeypatch):
    """REGRESSION GUARD: Socrata clamps $limit server-side on some endpoints.
    Treating a short page as the last page truncates the download at the first
    request — and a partial footprint universe understates every tract's
    baseline area, biasing treatment intensity upward, with nothing raising."""
    source = ps.ParcelSource(key="k", city="chicago", label="L", kind="socrata",
                             url="https://example.org", dataset_id="a-b",
                             geom_kind="footprint")
    monkeypatch.setattr(type(source), "cache_dir",
                        property(lambda self: tmp_path / "pages"))
    # Ask for 50k a page; the server only ever hands back 1k.
    sess = FakeSocrata(total=4_300, server_cap=1_000)
    out = ps.fetch_socrata(source, page_size=50_000, session=sess, verbose=False)
    assert len(out) == 4_300
    assert len(set(out["bldg_id"])) == 4_300


def test_socrata_paging_is_ordered(tmp_path, monkeypatch):
    """REGRESSION GUARD: Socrata's $offset without $order is not stable across
    requests — rows repeat or vanish between pages, which would double- or
    under-count footprints in a tract's baseline area."""
    source = ps.ParcelSource(key="k", city="chicago", label="L", kind="socrata",
                             url="https://example.org", dataset_id="a-b",
                             geom_kind="parcel")
    monkeypatch.setattr(type(source), "cache_dir",
                        property(lambda self: tmp_path / "pages"))
    sess = FakeSocrata(total=5, fmt="json")
    ps.fetch_socrata(source, page_size=10, session=sess, verbose=False)
    assert all(c["$order"] == ":id" for c in sess.calls)


def test_socrata_resumes_from_cached_pages(tmp_path, monkeypatch):
    source = ps.ParcelSource(key="k", city="chicago", label="L", kind="socrata",
                             url="https://example.org", dataset_id="a-b",
                             geom_kind="parcel")
    pages = tmp_path / "pages"
    pages.mkdir(parents=True)
    monkeypatch.setattr(type(source), "cache_dir", property(lambda self: pages))
    (pages / "page_00000.json").write_text(json.dumps(
        [{"bldg_id": 1, "year_built": 1999}, {"bldg_id": 2, "year_built": 2000}]))

    sess = FakeSocrata(total=0, fmt="json")
    out = ps.fetch_socrata(source, page_size=2, session=sess, verbose=False)
    assert len(out) == 2
    # Page 0 came off disk; only page 1 was requested.
    assert [c["$offset"] for c in sess.calls] == [2]


def test_envelope_params_are_emitted_for_a_bbox():
    p = ps.envelope_params((-82.9, 27.6, -82.2, 28.4))
    assert p["geometry"] == "-82.9,27.6,-82.2,28.4"
    assert p["geometryType"] == "esriGeometryEnvelope"
    assert p["inSR"] == "4326"
    assert p["spatialRel"] == "esriSpatialRelIntersects"
    assert ps.envelope_params(None) == {}


def _keyset_fake(page_params, n_pages=2, page_size=2):
    """A fake ArcGIS that serves ``n_pages`` keyset pages then an empty one."""
    served = {"n": 0}

    def fake_get_json(session, url, params, sleep_fn=None):
        if params.get("f") == "json":
            return {"maxRecordCount": page_size}
        page_params.append(params)
        served["n"] += 1
        if served["n"] > n_pages:
            return {"type": "FeatureCollection", "features": []}
        base = (served["n"] - 1) * page_size + 1
        return {"type": "FeatureCollection", "features": [
            {"type": "Feature", "id": base + i,
             "geometry": {"type": "Point", "coordinates": [-82.5, 27.9]},
             "properties": {"OBJECTID": base + i, "ACT_YR_BLT": 2019}}
            for i in range(page_size)]}
    return fake_get_json


def test_bbox_is_applied_to_every_page_query(monkeypatch, tmp_path):
    """A spatial filter must go on each page, not only on a one-off selection
    step — otherwise later pages silently widen back out to the whole layer."""
    source = ps.ParcelSource(key="p", city="tampa", label="L", kind="arcgis",
                             url="https://example.org/0", geom_kind="parcel",
                             select=("OBJECTID", "ACT_YR_BLT"))
    monkeypatch.setattr(type(source), "cache_dir",
                        property(lambda self: tmp_path / "pages"))
    page_params: list = []
    monkeypatch.setattr(ps, "_get_json", _keyset_fake(page_params))
    ps.fetch_arcgis(source, session=object(), bbox=(-82.9, 27.6, -82.2, 28.4),
                    verbose=False)
    queries = [p for p in page_params if p.get("f") == "geojson"]
    assert queries, "no page queries issued"
    for p in queries:
        assert p.get("geometryType") == "esriGeometryEnvelope"
        assert p.get("geometry") == "-82.9,27.6,-82.2,28.4"


def test_paging_never_uses_result_offset(monkeypatch, tmp_path):
    """REGRESSION GUARD: deep resultOffset degrades badly on a 10.8M-row layer
    (the server walks every skipped row), and some services reject the id-only
    query a range-based paginator needs. Keyset paging uses neither."""
    source = ps.ParcelSource(key="p", city="tampa", label="L", kind="arcgis",
                             url="https://example.org/0", geom_kind="parcel",
                             select=("OBJECTID", "ACT_YR_BLT"))
    monkeypatch.setattr(type(source), "cache_dir",
                        property(lambda self: tmp_path / "pages"))
    page_params: list = []
    monkeypatch.setattr(ps, "_get_json", _keyset_fake(page_params))
    ps.fetch_arcgis(source, session=object(), verbose=False)
    for p in page_params:
        assert "resultOffset" not in p
        assert p.get("returnIdsOnly") is None
    assert any(p.get("orderByFields") == "OBJECTID"
               for p in page_params if p.get("f") == "geojson")


class _Resp:
    def __init__(self, status_code, text="", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload if payload is not None else []

    def json(self):
        return self._payload


def test_client_errors_are_not_retried():
    """REGRESSION GUARD: a 400 is a permanent client error (mistyped column, bad
    filter). Retrying it four times only delays the message by the backoff
    ladder."""
    calls = []

    class Sess:
        def get(self, url, params=None, timeout=None):
            calls.append(params)
            return _Resp(400, "no such column: bldg_status")

    with pytest.raises(RuntimeError, match="permanently"):
        ps._get_json(Sess(), "https://example.org", {"$select": "x"},
                     sleep_fn=lambda s: None)
    assert len(calls) == 1


def test_client_error_surfaces_the_server_message():
    """Socrata names the offending column in the body; swallowing it turns a
    one-line fix into a guessing game."""
    class Sess:
        def get(self, url, params=None, timeout=None):
            return _Resp(400, "Invalid SoQL: no such column: bldg_status")

    with pytest.raises(RuntimeError, match="no such column: bldg_status"):
        ps._get_json(Sess(), "https://example.org", {}, sleep_fn=lambda s: None)


def test_rate_limits_and_server_errors_are_still_retried():
    seen = []

    class Sess:
        def get(self, url, params=None, timeout=None):
            seen.append(1)
            if len(seen) < 3:
                return _Resp(503, "busy")
            return _Resp(200, payload=[{"ok": 1}])

    out = ps._get_json(Sess(), "https://example.org", {}, sleep_fn=lambda s: None)
    assert out == [{"ok": 1}]
    assert len(seen) == 3


def test_429_is_retried_not_treated_as_permanent():
    seen = []

    class Sess:
        def get(self, url, params=None, timeout=None):
            seen.append(1)
            return _Resp(429, "slow down") if len(seen) < 2 else _Resp(200, payload=[])

    ps._get_json(Sess(), "https://example.org", {}, sleep_fn=lambda s: None)
    assert len(seen) == 2


def test_registry_marks_the_cook_supplement_as_non_primary():
    """x54s-btds covers residential <7 units only — using it as Chicago's cohort
    source would exclude exactly the commercial and large-multifamily
    construction that drives the city's development."""
    primary = ps.sources_for("chicago", primary_only=True)
    assert [s.key for s in primary] == ["chicago_footprints"]
    supplement = [s for s in ps.sources_for("chicago") if not s.primary][0]
    assert supplement.dataset_id == "x54s-btds"
    assert "supplement" in supplement.note


# ═══════════════════════════════════════════════════════════════════════════════
# Tabular year sources — King County / Seattle
#
# King County is the one registered city whose parcel geometry carries no year at
# all, so its years arrive as two separate CAMA extracts joined on PIN. Three
# things can go wrong silently, and each has a test below:
#
#   * the key. Major/Minor are stored UNPADDED, so a naive concat produces a key
#     that matches nothing (or, worse, matches the wrong parcel).
#   * the aggregation. A parcel holding several vintages must collapse to one
#     year, and the naive choices are wrong in opposite directions.
#   * the union. Residential covers 1-3 unit buildings only; apartments are in
#     the COMMERCIAL file. Dropping either deletes half the building stock.
# ═══════════════════════════════════════════════════════════════════════════════

def _res_source():
    return ps.source_by_key("seattle", "kingco_resbldg")


def _comm_source():
    return ps.source_by_key("seattle", "kingco_commbldg")


def _bldg_frame(rows, size_col="SqFtTotLiving"):
    """Rows of (Major, Minor, YrBuilt, size) as the raw string-typed CSV would be."""
    return pd.DataFrame({
        "Major": [str(r[0]) for r in rows],
        "Minor": [str(r[1]) for r in rows],
        "YrBuilt": [str(r[2]) for r in rows],
        size_col: [str(r[3]) for r in rows],
    })


# ─── parcel_key ───────────────────────────────────────────────────────────────

def test_parcel_key_zero_pads_each_component():
    """REGRESSION GUARD: the assessor stores Major/Minor unpadded ('200660',
    '1340') while the GIS layer's PIN is a padded 10-character string. Plain
    concatenation gives '2006601340' here but '200660134' for Minor='134', so the
    short-Minor parcels would silently fail to join."""
    frame = pd.DataFrame({"Major": ["200660", "20066"], "Minor": ["1340", "134"]})
    key = byb.parcel_key(frame, ("Major", "Minor"), (6, 4))
    assert list(key) == ["2006601340", "0200660134"]
    assert all(len(k) == 10 for k in key)


def test_parcel_key_strips_mainframe_padding():
    """These are fixed-width mainframe extracts; fields arrive space-padded."""
    frame = pd.DataFrame({"Major": [" 200660 "], "Minor": ["1340  "]})
    assert list(byb.parcel_key(frame, ("Major", "Minor"), (6, 4))) == ["2006601340"]


def test_parcel_key_raises_rather_than_truncating_an_overwide_value():
    """A truncated key does not fail to join — it joins to the WRONG parcel, and
    hands that parcel someone else's construction year."""
    frame = pd.DataFrame({"Major": ["1234567"], "Minor": ["1340"]})
    with pytest.raises(ValueError, match="exceed the declared width"):
        byb.parcel_key(frame, ("Major", "Minor"), (6, 4))


def test_parcel_key_nulls_the_whole_key_when_a_component_is_missing():
    frame = pd.DataFrame({"Major": ["200660", None], "Minor": ["1340", "0010"]})
    key = byb.parcel_key(frame, ("Major", "Minor"), (6, 4))
    assert key[0] == "2006601340"
    assert pd.isna(key[1])


def test_parcel_key_rejects_a_mismatched_spec():
    frame = pd.DataFrame({"Major": ["200660"], "Minor": ["1340"]})
    with pytest.raises(ValueError, match="differ in length"):
        byb.parcel_key(frame, ("Major", "Minor"), (6,))
    with pytest.raises(KeyError):
        byb.parcel_key(frame, ("Major", "Parcel"), (6, 4))


# ─── building_year_table ──────────────────────────────────────────────────────

def test_building_year_table_unions_residential_and_commercial():
    """Apartments live in the COMMERCIAL extract. A union that dropped it would
    remove the multifamily towers driving Seattle's construction boom and leave a
    cohort panel made of single-family infill."""
    res = _bldg_frame([("200660", "1340", "1950", "1800")])
    comm = _bldg_frame([("300111", "0020", "2016", "90000")],
                       size_col="BldgGrossSqFt")
    out = byb.building_year_table(
        {"kingco_resbldg": res, "kingco_commbldg": comm},
        {"kingco_resbldg": _res_source(), "kingco_commbldg": _comm_source()})
    assert len(out) == 2
    assert set(out["source"]) == {"kingco_resbldg", "kingco_commbldg"}
    assert set(out["parcel_key"]) == {"2006601340", "3001110020"}
    assert out.loc[out["parcel_key"] == "3001110020", "year_built"].iloc[0] == 2016


def test_building_year_table_keeps_one_row_per_building():
    """Per-building rows are what let the aggregation below tell a new garage
    apart from a redevelopment; collapsing here would destroy that."""
    res = _bldg_frame([("200660", "1340", "1950", "1800"),
                       ("200660", "1340", "2016", "400")])
    out = byb.building_year_table({"kingco_resbldg": res},
                                  {"kingco_resbldg": _res_source()})
    assert len(out) == 2
    assert out["parcel_key"].nunique() == 1


def test_building_year_table_nulls_sentinel_years():
    res = _bldg_frame([("200660", "1340", "0", "1800"),
                       ("200661", "1340", "1995", "1800")])
    out = byb.building_year_table({"kingco_resbldg": res},
                                  {"kingco_resbldg": _res_source()})
    assert pd.isna(out["year_built"].iloc[0])
    assert out["year_built"].iloc[1] == 1995


def test_building_year_table_coerces_space_padded_sizes():
    """BldgGrossSqFt arrives as '512      ' — a string. Left uncoerced it sorts
    lexically, and '9' would outrank '90000'."""
    comm = _bldg_frame([("300111", "0020", "2016", "512      ")],
                       size_col="BldgGrossSqFt")
    out = byb.building_year_table({"kingco_commbldg": comm},
                                  {"kingco_commbldg": _comm_source()})
    assert out["size"].iloc[0] == 512.0


def test_building_year_table_names_a_missing_year_column():
    res = _bldg_frame([("200660", "1340", "1950", "1800")]).drop(columns=["YrBuilt"])
    with pytest.raises(KeyError, match="YrBuilt"):
        byb.building_year_table({"kingco_resbldg": res},
                                {"kingco_resbldg": _res_source()})


# ─── parcel_year_from_buildings ───────────────────────────────────────────────

def _stacked(rows):
    """rows of (parcel_key, year, size) already normalised."""
    return pd.DataFrame({
        "parcel_key": [r[0] for r in rows],
        "year_built": pd.array([r[1] for r in rows], dtype="Int64"),
        "size": [r[2] for r in rows],
        "source": "test",
    })


def test_dominant_keeps_the_old_house_when_a_new_garage_appears():
    """THE case agg='max' gets wrong. A 1950 parcel with a 2016 garage is not
    2016 construction; dating it so would hand the tract the whole parcel's
    footprint area as new build and manufacture treatment where none exists."""
    out, _ = byb.parcel_year_from_buildings(
        _stacked([("A", 1950, 1800.0), ("A", 2016, 400.0)]), agg="dominant")
    assert out.set_index("parcel_key")["year_built"]["A"] == 1950


def test_dominant_takes_the_new_tower_on_redevelopment():
    """THE case agg='min' gets wrong. A 1950 lot rebuilt as a 2016 tower is
    genuine treatment; dating it 1950 erases it and attenuates the ATT."""
    out, _ = byb.parcel_year_from_buildings(
        _stacked([("A", 1950, 900.0), ("A", 2016, 90000.0)]), agg="dominant")
    assert out.set_index("parcel_key")["year_built"]["A"] == 2016


def test_dominant_falls_back_to_the_oldest_when_sizes_are_unknown():
    """No recorded area means no basis to call one structure dominant, so the
    rule degrades to the conservative choice rather than picking arbitrarily."""
    out, _ = byb.parcel_year_from_buildings(
        _stacked([("A", 2016, np.nan), ("A", 1950, np.nan)]), agg="dominant")
    assert out.set_index("parcel_key")["year_built"]["A"] == 1950


def test_dominant_breaks_equal_size_ties_toward_the_oldest():
    out, _ = byb.parcel_year_from_buildings(
        _stacked([("A", 2016, 1000.0), ("A", 1950, 1000.0)]), agg="dominant")
    assert out.set_index("parcel_key")["year_built"]["A"] == 1950


def test_min_and_max_aggregations_bracket_the_dominant_rule():
    rows = _stacked([("A", 1950, 1800.0), ("A", 2016, 400.0)])
    lo, _ = byb.parcel_year_from_buildings(rows, agg="min")
    hi, _ = byb.parcel_year_from_buildings(rows, agg="max")
    assert lo.set_index("parcel_key")["year_built"]["A"] == 1950
    assert hi.set_index("parcel_key")["year_built"]["A"] == 2016


def test_single_building_parcels_are_unaffected_by_the_rule():
    rows = _stacked([("A", 1995, 1200.0), ("B", 2008, 3000.0)])
    years = {agg: byb.parcel_year_from_buildings(rows, agg=agg)[0]
             .set_index("parcel_key")["year_built"].to_dict()
             for agg in ("dominant", "min", "max")}
    assert years["dominant"] == years["min"] == years["max"] == {"A": 1995, "B": 2008}


def test_aggregation_diagnostics_quantify_how_much_the_rule_matters():
    """If frac_parcels_multi_vintage is small the three rules agree almost
    everywhere, so the diagnostic bounds the cohorts' sensitivity to the choice
    rather than leaving it as an unquantified caveat."""
    _, diag = byb.parcel_year_from_buildings(
        _stacked([("A", 1950, 1800.0), ("A", 2016, 400.0),
                  ("B", 1995, 1200.0)]), agg="dominant")
    assert diag["n_parcels_with_year"] == 2
    assert diag["frac_parcels_multi_building"] == 0.5
    assert diag["frac_parcels_multi_vintage"] == 0.5
    assert diag["median_vintage_spread_years"] == 66.0
    assert diag["parcel_year_agg"] == "dominant"


def test_undated_buildings_do_not_create_a_parcel_year():
    """An undated building must leave its parcel undated, so its footprints stay
    in the baseline denominator instead of entering a cohort."""
    out, diag = byb.parcel_year_from_buildings(
        _stacked([("A", None, 1800.0), ("B", 1995, 1200.0)]))
    assert list(out["parcel_key"]) == ["B"]
    assert diag["n_building_records"] == 2
    assert diag["n_building_records_dated"] == 1


def test_unknown_aggregation_is_rejected():
    with pytest.raises(ValueError, match="unknown agg"):
        byb.parcel_year_from_buildings(_stacked([("A", 1995, 1.0)]), agg="mean")


# ─── attach_table_years ───────────────────────────────────────────────────────

def _parcel_layer(pins):
    gdf = _squares(len(pins))
    gdf["PIN"] = list(pins)
    return gdf


def test_attach_table_years_joins_on_the_padded_pin():
    parcels = _parcel_layer(["2006601340", "3001110020"])
    per_parcel = pd.DataFrame({"parcel_key": ["2006601340", "3001110020"],
                               "year_built": pd.array([1950, 2016], dtype="Int64")})
    out, stats = byb.attach_table_years(parcels, per_parcel, _seattle_geom_source())
    assert list(out["year_built"]) == [1950, 2016]
    assert stats["parcel_year_match_fraction"] == 1.0


def test_attach_table_years_leaves_unmatched_parcels_unknown():
    """Not a neutral loss: an unmatched parcel keeps its footprints in the tract's
    baseline denominator while contributing nothing to the numerator, so a bad key
    pushes every tract's treatment intensity DOWN and out of the treated group."""
    parcels = _parcel_layer(["2006601340", "9999999999"])
    per_parcel = pd.DataFrame({"parcel_key": ["2006601340"],
                               "year_built": pd.array([1950], dtype="Int64")})
    out, stats = byb.attach_table_years(parcels, per_parcel, _seattle_geom_source())
    assert out["year_built"].iloc[0] == 1950
    assert pd.isna(out["year_built"].iloc[1])
    assert stats["parcel_year_match_fraction"] == 0.5


def test_attach_table_years_tolerates_padding_in_the_gis_pin():
    parcels = _parcel_layer([" 2006601340 "])
    per_parcel = pd.DataFrame({"parcel_key": ["2006601340"],
                               "year_built": pd.array([1950], dtype="Int64")})
    out, _ = byb.attach_table_years(parcels, per_parcel, _seattle_geom_source())
    assert out["year_built"].iloc[0] == 1950


def test_attach_table_years_names_a_missing_join_key():
    parcels = _squares(1)
    with pytest.raises(KeyError, match="PIN"):
        byb.attach_table_years(parcels, pd.DataFrame(
            {"parcel_key": [], "year_built": []}), _seattle_geom_source())


def _seattle_geom_source():
    return ps.source_by_key("seattle", "kingco_parcels")


def test_attached_years_flow_into_the_spatial_join():
    """End-to-end for the mode: CAMA table -> parcel polygon -> footprint. The
    output must reach build_parcel_join's contract with no special-casing."""
    source = _seattle_geom_source()
    parcels = _parcel_layer(["2006601340", "3001110020"])
    res = _bldg_frame([("200660", "1340", "1950", "1800")])
    comm = _bldg_frame([("300111", "0020", "2016", "90000")],
                       size_col="BldgGrossSqFt")
    stacked = byb.building_year_table(
        {"kingco_resbldg": res, "kingco_commbldg": comm},
        {"kingco_resbldg": _res_source(), "kingco_commbldg": _comm_source()})
    per_parcel, _ = byb.parcel_year_from_buildings(stacked)
    dated_parcels, _ = byb.attach_table_years(parcels, per_parcel, source)

    # One footprint inside each parcel square.
    cents = dated_parcels.geometry.centroid
    foot = gpd.GeoDataFrame(
        geometry=[c.buffer(3.0) for c in cents],
        index=pd.Index([11, 22], name="building_id"), crs=dated_parcels.crs)
    out, stats = byb.build_parcel_join(dated_parcels, foot, source)
    assert list(out["year_built"]) == [1950, 2016]
    assert stats["matched_fraction"] == 1.0


# ─── registry ─────────────────────────────────────────────────────────────────

def test_seattle_parcel_layer_declares_it_has_no_year_of_its_own():
    """KingCo_Parcels is (OBJECTID, MAJOR, MINOR, PIN, Shape) and nothing else, so
    the build must know to fetch years elsewhere before the spatial join."""
    source = _seattle_geom_source()
    assert source.primary
    assert source.join_key == "PIN"
    assert set(source.year_from) == {"kingco_resbldg", "kingco_commbldg"}


def test_seattle_registers_both_assessor_extracts():
    """Residential covers 1-3 unit buildings ONLY; apartments are commercial."""
    keys = {s.key for s in ps.sources_for("seattle")}
    assert {"kingco_resbldg", "kingco_commbldg"} <= keys
    for key in ("kingco_resbldg", "kingco_commbldg"):
        source = ps.source_by_key("seattle", key)
        assert source.geom_kind == "table"
        assert source.kind == "csv_archive"
        assert source.key_cols == ("Major", "Minor")
        assert source.key_widths == (6, 4)
        assert source.size_col in source.select


def test_seattle_dates_on_actual_year_not_renovation_year():
    """YrRenovated is reset by a major remodel, so using it would date a 1920
    bungalow gut-rehabbed in 2016 into the 2016 construction cohort — inventing
    treatment on a parcel where no new structure was built. Same trap as
    Florida's EFF_YR_BLT."""
    for key in ("kingco_resbldg", "kingco_commbldg"):
        source = ps.source_by_key("seattle", key)
        assert source.year_col == "YrBuilt"
        assert "YrRenovated" not in source.select


def test_seattle_keyset_paging_can_run_on_the_parcel_layer():
    """_arcgis_keyset_pages raises if OBJECTID is absent from the response, so a
    select that omits it makes the download impossible."""
    assert "OBJECTID" in _seattle_geom_source().select


def test_seattle_panel_years_match_the_washington_naip_grid():
    """Verified against the Planetary Computer inventory: Washington NAIP exists
    only in odd years. An even year in the panel would fetch nothing."""
    from src.csa_event_study import CSA_CITIES
    spec = CSA_CITIES["seattle"]
    assert spec.panel_years == (2011, 2013, 2015, 2017, 2019)
    assert all(y % 2 == 1 for y in spec.panel_years)
    assert spec.baseline_year == 2010
    assert spec.sensor == "naip"


def test_nashville_records_why_it_is_blocked():
    """Its core county's years are not public in bulk. The note must say so and
    say what would unblock it, or the next session re-derives the whole dead end."""
    from src.csa_event_study import CSA_CITIES
    spec = CSA_CITIES["nashville"]
    assert spec.note.startswith("BLOCKED:")
    assert "Davidson" in spec.note
    assert ps.sources_for("nashville") == ()


# ─── san antonio ──────────────────────────────────────────────────────────────

def _san_antonio_source():
    return ps.source_by_key("san_antonio", "bexar_parcels")


def test_san_antonio_parcel_layer_carries_its_own_year():
    source = _san_antonio_source()
    assert source.primary
    assert source.kind == "arcgis" and source.geom_kind == "parcel"
    assert source.year_col == "YrBlt"
    assert source.year_from == () and source.join_key is None
    assert "OBJECTID" in source.select          # keyset paging needs it


def test_san_antonio_null_string_sentinel_normalizes_to_na():
    """REGRESSION GUARD: Bexar's YrBlt is a STRING whose missing value is the
    literal text "NULL". Read naively that is neither a year nor a null, and a
    cohort built from it would be garbage."""
    out = byb.normalize_years(pd.Series(["1961", "NULL", "2015", "0", None]))
    assert out.iloc[0] == 1961
    assert pd.isna(out.iloc[1])                 # the literal "NULL"
    assert out.iloc[2] == 2015
    assert pd.isna(out.iloc[3]) and pd.isna(out.iloc[4])
    assert str(out.dtype) == "Int64"


def test_san_antonio_does_not_filter_undated_parcels_server_side():
    """Filtering `YrBlt <> 'NULL'` would cut the download 13% and CORRUPT the
    join: build_parcel_join falls back to sjoin_nearest within
    NEAREST_MAX_DISTANCE_M, so a deleted parcel does not leave its buildings
    undated — it lets them inherit a NEIGHBOUR's year, inventing construction on
    a parcel whose year is merely unknown."""
    assert _san_antonio_source().where is None


def test_san_antonio_keeps_an_unknown_year_unknown_through_the_join():
    """The behavioural half of the guard above: a footprint on an undated parcel
    must come out NA, not carry the dated neighbour's year."""
    parcels = _squares(2, size=100.0, gap=200.0)
    parcels["YrBlt"] = ["1961", "NULL"]
    footprints = _squares(2, size=10.0, gap=200.0, x0=X0 + 20.0)
    footprints = footprints.set_index(pd.Index([1, 2], name="building_id"))

    out, _ = byb.build_parcel_join(parcels, footprints, _san_antonio_source())
    assert out["year_built"].iloc[0] == 1961
    assert pd.isna(out["year_built"].iloc[1])


def test_san_antonio_panel_years_match_the_texas_naip_grid():
    """Texas NAIP is biennial EVEN years — the opposite parity to Washington's
    odd-year grid, which is why panel cadence cannot be a global. An odd year
    here would fetch nothing."""
    from src.csa_event_study import CSA_CITIES
    spec = CSA_CITIES["san_antonio"]
    assert spec.panel_years == (2012, 2014, 2016, 2018)
    assert all(y % 2 == 0 for y in spec.panel_years)
    assert spec.baseline_year == 2011 < min(spec.panel_years)
    assert spec.sensor == "naip" and spec.state == "Texas"


def test_san_antonio_is_scoped_to_bexar_county():
    """Texas has no statewide parcel programme, so the source is one county's
    appraisal district. The county line is also a tract boundary, so a FIPS
    prefix is exact — unlike Chicago, which needs footprint-derived tracts."""
    from src.csa_event_study import CSA_CITIES
    spec = CSA_CITIES["san_antonio"]
    assert spec.geoid_prefixes == ("48029",)
    assert spec.tract_source == "prefix"


def test_holdout_cities_span_more_than_one_region():
    """The registry is the answer to a cherry-picking objection, so it must keep
    spanning distinct census divisions. State FIPS: 36 NY, 24 MD, 12 FL, 53 WA,
    48 TX."""
    from src.csa_event_study import CSA_CITIES
    states = {p[:2] for k, s in CSA_CITIES.items() if not s.note.startswith("BLOCKED")
              for p in s.geoid_prefixes}
    assert {"36", "24", "12", "53", "48"} <= states


# ─── baltimore ────────────────────────────────────────────────────────────────

def _baltimore_source():
    return ps.source_by_key("baltimore", "md_parcel_boundaries")


def test_baltimore_parcel_layer_carries_its_own_year():
    """The only registered geometry source that needs no `year_from`: Maryland
    puts YEARBLT on the polygon, unlike King County whose layer is PIN + shape.
    A stray year_from would send the build looking for tables that do not exist."""
    source = _baltimore_source()
    assert source.primary
    assert source.kind == "arcgis" and source.geom_kind == "parcel"
    assert source.year_col == "YEARBLT"
    assert source.year_from == ()
    assert source.join_key is None


def test_baltimore_keyset_paging_can_run_on_the_parcel_layer():
    """_arcgis_keyset_pages raises if OBJECTID is absent from the response, so a
    select that omits it makes the download impossible."""
    assert "OBJECTID" in _baltimore_source().select
    assert {"ACCTID", "YEARBLT"} <= set(_baltimore_source().select)


def test_baltimore_dates_on_actual_year_with_no_effective_year_trap():
    """Florida's EFF_YR_BLT and King County's YrRenovated are reset by major
    remodels and would date a rehabbed rowhouse into a recent construction
    cohort. MD_ParcelBoundaries has no such column at all — this guards against
    one being added to `select` later on the assumption that it is a better year."""
    select = set(_baltimore_source().select)
    assert not any("EFF" in c.upper() or "RENOV" in c.upper() for c in select)


def test_baltimore_declares_no_demolition_data():
    """MD_ParcelBoundaries has no demolition column. That is an honest absence
    (unlike Chicago's 1899 sentinel), and the spec must not invent one."""
    from src.csa_event_study import CSA_CITIES
    assert CSA_CITIES["baltimore"].demolition_col is None
    assert _baltimore_source().demolition_col is None


def test_baltimore_panel_years_match_the_maryland_naip_grid():
    """Verified against the Planetary Computer inventory: Maryland NAIP is
    2011/2013/2015/2017/2018/2021/2023.

    The panel stops before 2021 because cohorts come from the static ~2019
    Microsoft footprint snapshot — 2021 and 2023 would put post-snapshot
    development in the never-treated control group.

    2018 is excluded for a different reason: it is one year after 2017 while
    every other gap is two, so its cohort is defined over half the calendar
    window of the others and ends up the sole identifier of event time k = -3
    (one ATT(g,t) cell at weight 1.0 — two tracts at the 10% cut). Dropping it
    moves the sup-t pre-trend p from 0.0005 to 0.018 at the 1% threshold and
    from 0.0000 to 0.047 at 10%, leaving the post-treatment ATT unchanged. See
    test_csa_predict.test_no_city_panel_ends_in_a_short_stub_period.
    """
    from src.csa_event_study import CSA_CITIES
    spec = CSA_CITIES["baltimore"]
    assert spec.panel_years == (2011, 2013, 2015, 2017)
    assert 2018 not in spec.panel_years
    assert spec.baseline_year == 2010
    assert spec.baseline_year < min(spec.panel_years)
    assert spec.sensor == "naip"
    assert spec.state == "Maryland"


def test_baltimore_covers_the_whole_cbsa_by_prefix():
    """Maryland's layer is genuinely statewide, so unlike Chicago (whose municipal
    footprints stop at the city line) there is no sub-county coverage edge and the
    tract set can come from FIPS prefixes. All seven CBSA jurisdictions must be
    listed — dropping one silently deletes its tracts from the panel."""
    from src.csa_event_study import CSA_CITIES
    spec = CSA_CITIES["baltimore"]
    assert spec.tract_source == "prefix"
    assert set(spec.geoid_prefixes) == {
        "24003", "24005", "24013", "24025", "24027", "24035", "24510"}
    assert all(p.startswith("24") for p in spec.geoid_prefixes)


def test_baltimore_string_years_normalize_including_the_zero_sentinel():
    """REGRESSION GUARD: YEARBLT is esriFieldTypeString(4), so the column arrives
    as text ('1947') with '0000' as the missing sentinel. Both must survive
    normalisation — a string year silently coerced to NA would empty the panel,
    and a '0000' read as year 0 would date a cohort in antiquity."""
    out = byb.normalize_years(pd.Series(["1947", "0000", "2015", "", None]))
    assert out.iloc[0] == 1947
    assert pd.isna(out.iloc[1])          # '0000' sentinel
    assert out.iloc[2] == 2015
    assert pd.isna(out.iloc[3]) and pd.isna(out.iloc[4])
    assert str(out.dtype) == "Int64"


def test_baltimore_parcel_join_runs_on_string_years_end_to_end():
    """The whole point of the city: polygons carrying string years join straight
    to footprints with no table step. Exercises the real registered source."""
    parcels = _squares(3, size=100.0, gap=200.0)
    parcels["YEARBLT"] = ["1970", "0000", "2015"]
    footprints = _squares(3, size=10.0, gap=200.0, x0=X0 + 20.0)
    footprints = footprints.set_index(
        pd.Index([1, 2, 3], name="building_id"))

    out, stats = byb.build_parcel_join(parcels, footprints, _baltimore_source())

    assert list(out.columns) == ["year_built", "geometry"]
    assert out["year_built"].tolist()[0] == 1970
    assert pd.isna(out["year_built"].iloc[1])     # '0000' stays unknown
    assert out["year_built"].tolist()[2] == 2015
    assert stats["n_footprints"] == 3


def test_baltimore_fetch_parses_the_shape_the_mapserver_really_returns(
        monkeypatch, tmp_path):
    """CONTRACT GUARD for a service the sandbox cannot reach. Verified live that
    MD_ParcelBoundaries (a MapServer, not a FeatureServer) answers `f=geojson`
    with the feature `id` populated from OBJECTID and geometry in outSR 4326.
    This replays that exact shape — including a 'NOT LOCATED' row, the unmapped
    account whose geometry is null — so a regression in the pager or in geometry
    cleaning surfaces here rather than after a million-row download."""
    source = _baltimore_source()
    monkeypatch.setattr(type(source), "cache_dir",
                        property(lambda self: tmp_path / "pages"))
    served = {"n": 0}

    def fake_get_json(session, url, params, sleep_fn=None):
        if params.get("f") == "json":
            return {"maxRecordCount": 1000}
        served["n"] += 1
        if served["n"] > 1:
            return {"type": "FeatureCollection", "features": []}
        return {"type": "FeatureCollection", "features": [
            {"type": "Feature", "id": 507043,
             "geometry": {"type": "Polygon", "coordinates":
                          [[[-76.61, 39.29], [-76.60, 39.29],
                            [-76.60, 39.30], [-76.61, 39.30], [-76.61, 39.29]]]},
             "properties": {"OBJECTID": 507043, "ACCTID": "04010101020100",
                            "YEARBLT": "1947", "LU": "R", "SQFTSTRC": 1566,
                            "JURSCODE": "BACO", "CT2020": "24005400500"}},
            {"type": "Feature", "id": 507044,
             "geometry": None,
             "properties": {"OBJECTID": 507044, "ACCTID": "NOT LOCATED",
                            "YEARBLT": None, "LU": None, "SQFTSTRC": None,
                            "JURSCODE": "BACO", "CT2020": None}},
        ]}

    monkeypatch.setattr(ps, "_get_json", fake_get_json)
    out = ps.fetch_arcgis(source, session=object(), verbose=False)

    assert len(out) == 2
    assert set(source.select) <= set(out.columns)
    # YEARBLT arrives as TEXT, which is what makes normalize_years load-bearing.
    assert out["YEARBLT"].iloc[0] == "1947"
    # The unmapped account must not survive into a dated footprint universe.
    cleaned, stats = byb.clean_geometry(out, "md_parcel_boundaries")
    assert stats["n_dropped_null_geometry"] == 1
    assert len(cleaned) == 1
    assert cleaned["ACCTID"].iloc[0] == "04010101020100"


def _paging_source(tmp_path, monkeypatch, **kw):
    source = ps.ParcelSource(key="p", city="baltimore", label="L", kind="arcgis",
                             url="https://example.org/0", geom_kind="parcel",
                             select=("OBJECTID", "YEARBLT"), **kw)
    monkeypatch.setattr(type(source), "cache_dir",
                        property(lambda self: tmp_path / "pages"))
    return source


def _feature(oid):
    return {"type": "Feature", "id": oid,
            "geometry": {"type": "Point", "coordinates": [-76.6, 39.3]},
            "properties": {"OBJECTID": oid, "YEARBLT": "1947"}}


def test_page_size_halves_when_the_service_cannot_serialize_the_page(
        tmp_path, monkeypatch):
    """REGRESSION GUARD (Baltimore, page 21): maxRecordCount is a promise about
    ROW COUNT, not response size. Maryland advertises 1000, serves 1000 rows
    without geometry, and 500s on the same 1000 WITH polygons. A pager that takes
    the advertised number on faith dies partway through a million-row download."""
    source = _paging_source(tmp_path, monkeypatch)
    sizes, served = [], {"n": 0}

    def fake_get_json(session, url, params, sleep_fn=None):
        if params.get("f") == "json":
            return {"maxRecordCount": 1000}
        size = int(params["resultRecordCount"])
        sizes.append(size)
        if size > 250:                      # what the real service does
            raise RuntimeError("HTTP 500: Error performing query operation")
        served["n"] += 1
        if served["n"] > 2:
            return {"type": "FeatureCollection", "features": []}
        base = (served["n"] - 1) * 2 + 1
        return {"type": "FeatureCollection",
                "features": [_feature(base), _feature(base + 1)]}

    monkeypatch.setattr(ps, "_get_json", fake_get_json)
    out = ps.fetch_arcgis(source, session=object(), verbose=False)

    assert len(out) == 4
    assert sizes[0] == 1000                 # tried the advertised value
    assert 250 in sizes                     # degraded to something that works
    # Sticky: never climbs back and re-spends a doomed request on later pages.
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[-1] <= 250


def test_page_size_degradation_stops_at_the_floor(tmp_path, monkeypatch):
    """Halving forever would turn a server outage into an infinite loop. Below the
    floor the failure is no longer plausibly about response size, so it re-raises."""
    source = _paging_source(tmp_path, monkeypatch)

    def always_500(session, url, params, sleep_fn=None):
        if params.get("f") == "json":
            return {"maxRecordCount": 1000}
        raise RuntimeError("HTTP 500: Error performing query operation")

    monkeypatch.setattr(ps, "_get_json", always_500)
    with pytest.raises(RuntimeError, match="500"):
        ps.fetch_arcgis(source, session=object(), verbose=False)


def test_declared_page_size_skips_the_metadata_probe(tmp_path, monkeypatch):
    """A source that already knows the service's real limit should not spend a
    doomed request to rediscover it — nor even ask for maxRecordCount."""
    source = _paging_source(tmp_path, monkeypatch, page_size=250)
    seen = []

    def fake_get_json(session, url, params, sleep_fn=None):
        seen.append(params)
        assert params.get("f") != "json", "should not probe maxRecordCount"
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr(ps, "_get_json", fake_get_json)
    ps.fetch_arcgis(source, session=object(), verbose=False)
    assert seen and int(seen[0]["resultRecordCount"]) == 250


def test_baltimore_declares_the_page_size_its_service_can_actually_serve():
    """1000 (the advertised maxRecordCount) returns a deterministic 500 on heavy
    geometry; 500 is verified at the offset that killed the first pull."""
    source = _baltimore_source()
    assert source.page_size == 500
    assert ps.ARCGIS_MIN_PAGE <= source.page_size < ps.ARCGIS_DEFAULT_PAGE


def test_baltimore_narrows_the_download_to_its_own_jurisdictions():
    """The envelope is a rectangle and the CBSA is not: the box alone selects
    1,731,656 parcels against 1,044,818 in the metro, so 40% of the pages would
    be fetched only to be discarded when footprints are scoped to city tracts."""
    where = _baltimore_source().where
    assert where and "JURSCODE" in where
    for code in ("BACI", "BACO", "ANNE", "HOWA", "HARF", "CARR", "QUEE"):
        assert f"'{code}'" in where
    # DESCLU is not indexed on this service and answers a filter with HTTP 500.
    assert "DESCLU" not in where


def test_source_where_is_anded_with_the_keyset_predicate(tmp_path, monkeypatch):
    """A registry filter must NARROW the page, not replace the OBJECTID cursor —
    dropping the cursor would re-request page 0 forever."""
    source = _paging_source(tmp_path, monkeypatch, where="JURSCODE IN ('BACO')")
    wheres = []

    def fake_get_json(session, url, params, sleep_fn=None):
        if params.get("f") == "json":
            return {"maxRecordCount": 2}
        wheres.append(params["where"])
        if len(wheres) > 1:
            return {"type": "FeatureCollection", "features": []}
        return {"type": "FeatureCollection", "features": [_feature(7)]}

    monkeypatch.setattr(ps, "_get_json", fake_get_json)
    ps.fetch_arcgis(source, session=object(), verbose=False)
    assert "JURSCODE IN ('BACO')" in wheres[0]
    assert "OBJECTID > -1" in wheres[0]
    assert "OBJECTID > 7" in wheres[1]


def test_explicit_where_argument_overrides_the_registry(tmp_path, monkeypatch):
    source = _paging_source(tmp_path, monkeypatch, where="JURSCODE IN ('BACO')")

    def fake_get_json(session, url, params, sleep_fn=None):
        if params.get("f") == "json":
            return {"maxRecordCount": 2}
        assert "JURSCODE" not in params["where"]
        assert "ACCTID" in params["where"]
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr(ps, "_get_json", fake_get_json)
    ps.fetch_arcgis(source, session=object(), where="ACCTID <> 'NOT LOCATED'",
                    verbose=False)


def test_pages_of_different_sizes_still_compose_into_one_download(
        tmp_path, monkeypatch):
    """Degrading mid-download is only safe because the pager advances by the
    largest id SEEN, not by an assumed stride. This also makes a cache written by
    an earlier run at a different page size valid to resume from."""
    source = _paging_source(tmp_path, monkeypatch)
    served = {"n": 0}

    def fake_get_json(session, url, params, sleep_fn=None):
        if params.get("f") == "json":
            return {"maxRecordCount": 4}
        served["n"] += 1
        if served["n"] == 1:
            return {"type": "FeatureCollection",
                    "features": [_feature(i) for i in (1, 2, 3, 4)]}
        if served["n"] == 2:                # a smaller page, as after a degrade
            return {"type": "FeatureCollection",
                    "features": [_feature(i) for i in (5, 6)]}
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr(ps, "_get_json", fake_get_json)
    out = ps.fetch_arcgis(source, session=object(), verbose=False)
    assert sorted(out["OBJECTID"].tolist()) == [1, 2, 3, 4, 5, 6]


def test_baltimore_is_registered_but_not_a_main_figure_city():
    """It is the 4th city — an annex holdout. Promoting it into the main figure is
    a paper decision, not a side effect of registering a source."""
    from src import csa_event_study as ces
    assert "baltimore" in ces.CSA_CITIES
    assert "baltimore" not in ces.MAIN_FIGURE_CITIES


def test_baltimore_footprints_filename_is_unique_in_the_registry():
    """available_cities() gates purely on this filename; a collision would make
    one city silently read another's footprints."""
    from src import csa_event_study as ces
    names = [s.footprints_filename for s in ces.CSA_CITIES.values()]
    assert len(names) == len(set(names))
    assert ces.CSA_CITIES["baltimore"].footprints_filename == "buildings_md_mdp.parquet"


# ─── csv_archive transport ────────────────────────────────────────────────────

def _zip_with(tmp_path, name, frame):
    import zipfile
    path = tmp_path / "extract.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(name, frame.to_csv(index=False))
    return path


def test_archive_member_matches_on_basename(tmp_path):
    """These archives are regenerated weekly and have moved their CSV in and out
    of a top-level directory; an exact-path match would fail on a file that is
    plainly present."""
    path = _zip_with(tmp_path, "SomeFolder/EXTR_ResBldg.csv",
                     pd.DataFrame({"a": [1]}))
    assert ps.archive_member(path, "EXTR_ResBldg.csv") == "SomeFolder/EXTR_ResBldg.csv"


def test_archive_member_reports_what_the_archive_actually_holds(tmp_path):
    path = _zip_with(tmp_path, "EXTR_Other.csv", pd.DataFrame({"a": [1]}))
    with pytest.raises(RuntimeError, match="EXTR_Other.csv"):
        ps.archive_member(path, "EXTR_ResBldg.csv")


def test_csv_archive_reads_the_named_member(tmp_path, monkeypatch):
    frame = pd.DataFrame({"Major": ["200660"], "Minor": ["1340"],
                          "BldgNbr": ["1"], "YrBuilt": ["1950"],
                          "SqFtTotLiving": ["1800"], "NbrLivingUnits": ["1"],
                          "Unwanted": ["x"]})
    path = _zip_with(tmp_path, "EXTR_ResBldg.csv", frame)
    monkeypatch.setattr(ps, "download_archive", lambda source, **kw: path)
    out = ps.fetch_csv_archive(_res_source(), verbose=False)
    assert list(out.columns) == list(_res_source().select)
    assert out["YrBuilt"].iloc[0] == "1950"


def test_csv_archive_names_the_columns_that_went_missing(tmp_path, monkeypatch):
    """A renamed upstream column must say WHICH one; these files have 26-40
    columns and pandas' own message names none of them."""
    frame = pd.DataFrame({"Major": ["200660"], "Minor": ["1340"]})
    path = _zip_with(tmp_path, "EXTR_ResBldg.csv", frame)
    monkeypatch.setattr(ps, "download_archive", lambda source, **kw: path)
    with pytest.raises(RuntimeError, match="YrBuilt"):
        ps.fetch_csv_archive(_res_source(), verbose=False)


def test_csv_archive_reads_everything_as_string(tmp_path, monkeypatch):
    """Mainframe extracts space-pad their numerics ('5  ', '512      '), so type
    inference is unreliable; coercion happens once, explicitly, downstream."""
    frame = pd.DataFrame({"Major": ["200660"], "Minor": ["1340"],
                          "BldgNbr": ["5  "], "YrBuilt": ["1950"],
                          "SqFtTotLiving": ["1800   "], "NbrLivingUnits": ["1"]})
    path = _zip_with(tmp_path, "EXTR_ResBldg.csv", frame)
    monkeypatch.setattr(ps, "download_archive", lambda source, **kw: path)
    out = ps.fetch_csv_archive(_res_source(), verbose=False)
    assert out["SqFtTotLiving"].iloc[0].strip() == "1800"
    assert out.dtypes.eq(object).all()


def test_table_sources_never_receive_a_bbox(tmp_path, monkeypatch):
    """download_source hands spatial sources a city envelope; a non-spatial CSV
    would raise TypeError on a keyword it never accepts."""
    seen = {}
    monkeypatch.setattr(ps, "fetch_csv_archive",
                        lambda source, **kw: (seen.update(kw), pd.DataFrame({"a": [1]}))[1])
    monkeypatch.setattr(ps.ParcelSource, "out_path",
                        property(lambda self: tmp_path / f"{self.key}.parquet"))
    ps.download_source(_res_source(), bbox=(-1, -1, 1, 1), verbose=False)
    assert "bbox" not in seen
