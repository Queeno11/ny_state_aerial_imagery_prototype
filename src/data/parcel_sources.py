"""Downloaders for per-city building/parcel year-built data — issue #36.

The Callaway–Sant'Anna event study dates tracts into construction cohorts, so
every CSA city needs a footprint universe carrying a *construction year*. NYC has
one (DoITT ``CONSTRUCTION_YEAR``). Nothing else does: the national Microsoft
footprint index is ``(building_id, cx, cy, tract_id)`` — a static universe with no
dates at all (see ``src/data/build_buildings_index.py``). This module fetches the
missing years; :mod:`src.data.build_year_built` joins them to footprints.

Two shapes of source
--------------------
``geom_kind="footprint"``
    The source *is* dated building polygons — one row per building, with a year.
    Chicago's municipal footprint layer is this, which makes it a direct DoITT
    analogue needing no spatial join at all. Strongly preferred when available.

``geom_kind="parcel"``
    Assessor/parcel polygons carrying a year, which must be spatially joined to
    Microsoft footprints. Lossier: a parcel with a 1920 house and a 2015 addition
    has one year, and every building on it inherits that year.

Why Chicago uses the municipal footprints, not the Cook assessor
----------------------------------------------------------------
Cook County's ``Assessor - Parcel Universe`` (``nj4t-kc8j``) has no year-built
column at all, and ``Single and Multi-Family Improvement Characteristics``
(``x54s-btds``) has ``char_yrblt`` but covers only residential under 7 units —
excluding precisely the commercial and large-multifamily construction that drives
Chicago development, which would bias cohorts toward low-rise infill. The City of
Chicago footprint layer covers all building types. Its cost is geography: city
proper (~800 tracts), not the 7-county CBSA — which is why Chicago's tracts are
derived from footprint coverage rather than a county FIPS prefix
(``CityCohortSpec.tract_source``). ``x54s-btds`` stays registered as a
supplement/cross-check, not the primary.

Why Baltimore is the cheapest city in the registry
--------------------------------------------------
Maryland's statewide parcel layer is the only registered geometry source that
carries the construction year *on the polygon itself*. Seattle needs two CAMA
extracts joined on PIN before its geometry can be dated (``year_from``) and Tampa
needs a 2.6 GB statewide archive; Baltimore needs one paged ArcGIS query. It also
ships ``CT2020`` — the 11-digit tract GEOID — so the tract assignment arrives with
the data rather than being derived.

The reason it survives selection where Chicago and Pittsburgh do not is
``LU``/``DESCLU`` coverage. The recurring failure mode of assessor sources is that
year built lives in a *residential* characteristics table: Cook County's
``x54s-btds`` excludes commercial and 7+ unit multifamily outright, and Allegheny
County's assessment file dates 447,982/523,741 residential parcels but only
594/35,871 commercial (1.7%). Measured over Baltimore's seven CBSA jurisdictions,
Maryland dates 18,870/29,147 commercial parcels (64.7%) and 876,532/1,044,818
overall — commercial construction is actually observable, which is what the event
study's treatment intensity depends on.

Paging correctness
------------------
Both paginators are ordered. Socrata's ``$offset`` without ``$order`` is *not*
stable across requests — rows can repeat or vanish between pages — so every query
pins ``$order=:id``. ArcGIS ``resultOffset`` is worse: on large layers it is
unreliable regardless of ordering, so :func:`fetch_arcgis` pages by explicit
OBJECTID ranges obtained from ``returnIdsOnly=true``, which is exact by
construction.

Everything is resumable: each page lands in a per-source directory, and a
completed source is written once to ``<external>/parcels/<city>/<key>.parquet``
via ``.part`` -> ``os.replace``. Re-running skips finished sources.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from src.utils.paths import EXTERNAL_DATA_DIR

# One page of a Socrata export. 50k is the documented ceiling for the
# JSON/GeoJSON endpoints; larger values are silently clamped, which would make
# the offset arithmetic skip rows.
SOCRATA_PAGE = 50_000

# ArcGIS services advertise maxRecordCount (commonly 1000-2000). We ask for the
# service's own value and never exceed it.
ARCGIS_DEFAULT_PAGE = 1000

# ...but maxRecordCount is a promise about ROW COUNT, not about response size, and
# a server can advertise a count it cannot actually serialize. Maryland's parcel
# layer advertises 1000 and answers 1000 rows happily with `returnGeometry=false`,
# yet returns a deterministic HTTP 500 for the same 1000 rows WITH polygons once
# the page lands on heavy geometry — it died at page 21 of the Baltimore pull
# while pages 0-20 (lighter parcels) had succeeded. The page size is therefore
# halved on failure down to this floor rather than taken on faith. Below it, the
# failure is no longer plausibly about response size and is re-raised.
ARCGIS_MIN_PAGE = 25

HTTP_TIMEOUT_S = (10.0, 300.0)
RETRIES = 4
BACKOFF_S = 2.0

PARCELS_DIR = Path(EXTERNAL_DATA_DIR) / "parcels"


@dataclass(frozen=True)
class ParcelSource:
    """One downloadable year-built source.

    Attributes
    ----------
    key, city, label
        Registry identity; ``city`` matches :data:`src.csa_event_study.CSA_CITIES`.
    kind
        Transport: ``"socrata"`` (SODA 2.x), ``"arcgis"`` (FeatureServer/MapServer),
        ``"http_archive"`` (bulk spatial archive) or ``"csv_archive"`` (bulk zip
        holding a *non-spatial* CSV table).
    url
        Socrata: the portal host root. ArcGIS: the layer URL (``.../MapServer/0``).
        Archives: the file URL.
    dataset_id
        Socrata four-by-four (e.g. ``syp8-uezg``). Unused for ArcGIS.
    year_col
        Column holding the construction year *in the source's own naming*.
    demolition_col
        Column holding the demolition year or date, or ``None`` if the source has
        none. May be a date; :func:`build_year_built.normalize_years` extracts the
        year either way.
    id_col
        Source's own building/parcel id, kept for provenance and debugging.
    geom_kind
        ``"footprint"`` (dated building polygons), ``"parcel"`` (polygons needing a
        spatial join) or ``"table"`` (no geometry at all — a CAMA extract that
        supplies years to a *separate* geometry source via ``year_from``).
    primary
        Whether this is the city's authoritative source. Non-primary sources are
        supplements/cross-checks and are never used to build cohorts alone.
    select
        Optional explicit column list, to avoid pulling 100-column parcel rows.
    member
        ``csv_archive`` only: the CSV inside the zip. Matched on basename,
        case-insensitively, so a repackaged archive does not break the build.
    key_cols, key_widths
        ``csv_archive`` only: the columns whose zero-padded concatenation forms
        the parcel key, and the width of each. King County splits its parcel id
        across ``Major`` (6) + ``Minor`` (4) and stores both *unpadded*, so a naive
        string concat produces ``"2006601340"`` for one parcel and a different
        length for the next — see :func:`build_year_built.parcel_key`.
    size_col
        ``csv_archive`` only: per-building floor area, used to pick the parcel's
        dominant structure when several buildings of different vintages share it.
    join_key
        Geometry sources only: the column that ``year_from`` tables join to.
    year_from
        Geometry sources only: keys of the ``"table"`` sources supplying this
        layer's construction years. Non-empty means the layer has no year of its
        own and one must be joined in before the spatial join can run.
    page_size
        ArcGIS only: rows per request, overriding the service's advertised
        ``maxRecordCount``. Set this when a layer is known to advertise more than
        it can serialize (see :data:`ARCGIS_MIN_PAGE`); ``None`` trusts the
        service. The pager degrades on failure anyway, so this is an optimisation
        — it avoids spending a doomed request and its retry budget to rediscover
        a limit that is already known.
    where
        ArcGIS only: a server-side attribute filter ANDed with the keyset
        predicate, for layers where one is *demonstrably* cheap. This is a
        deliberate exception to the envelope-only rule in :func:`envelope_params`
        and must be justified per source, because the usual reason to avoid it is
        that county columns are unindexed. It narrows a download that the bbox
        alone cannot: an envelope is a rectangle and a CBSA is not, so Baltimore's
        box pulls 1,731,656 parcels against the 1,044,818 actually in the metro —
        40% of the pages fetched only to be discarded at the tract-scoping step.
    note
        Free text surfaced in coverage reports and skip messages.
    """

    key: str
    city: str
    label: str
    kind: Literal["socrata", "arcgis", "http_archive", "csv_archive"]
    url: str
    dataset_id: str | None = None
    year_col: str = "year_built"
    demolition_col: str | None = None
    id_col: str | None = None
    geom_kind: Literal["footprint", "parcel", "table"] = "footprint"
    primary: bool = True
    select: tuple[str, ...] = ()
    member: str | None = None
    key_cols: tuple[str, ...] = ()
    key_widths: tuple[int, ...] = ()
    size_col: str | None = None
    join_key: str | None = None
    year_from: tuple[str, ...] = ()
    page_size: int | None = None
    where: str | None = None
    note: str = ""

    @property
    def out_path(self) -> Path:
        return PARCELS_DIR / self.city / f"{self.key}.parquet"

    @property
    def cache_dir(self) -> Path:
        return PARCELS_DIR / self.city / f"_{self.key}_pages"


PARCEL_SOURCES: dict[str, tuple[ParcelSource, ...]] = {
    "san_antonio": (
        ParcelSource(
            key="bexar_parcels",
            city="san_antonio",
            label="Bexar County Parcels (BCAD appraisal roll)",
            kind="arcgis",
            url="https://maps.bexar.org/arcgis/rest/services/Parcels/MapServer/0",
            # `YrBlt` is esriFieldTypeString whose MISSING SENTINEL IS THE LITERAL
            # TEXT "NULL", not a null. That is a live trap in any server-side
            # predicate: `YrBlt > '1800'` MATCHES "NULL", because the comparison
            # is lexical and "N" > "1". Filters must say `YrBlt <> 'NULL'`.
            # `normalize_years` is already safe — to_numeric coerces "NULL" to NA.
            year_col="YrBlt",
            id_col="AcctNumb",
            geom_kind="parcel",
            primary=True,
            # State_cd is the Texas comptroller class code (A residential, B
            # multifamily, F1 commercial, F2 industrial), kept so the coverage
            # report can show that the all-class criterion actually holds here:
            # 20,782/24,664 F1 commercial parcels are dated (84%), against 1.7%
            # for Allegheny and 0% for Cook. TOT_GBA is gross building area.
            select=("OBJECTID", "AcctNumb", "YrBlt", "State_cd", "PropUse",
                    "TOT_GBA"),
            # NOTE: deliberately NOT filtered to `YrBlt <> 'NULL'`, though that
            # would cut the download by 13%. Undated parcels must stay in the
            # frame: `build_parcel_join` falls back to sjoin_nearest within
            # NEAREST_MAX_DISTANCE_M for footprints that land in a gap, so
            # deleting a parcel does not make its buildings undated — it makes
            # them inherit a NEIGHBOUR's year, inventing construction on a parcel
            # whose year is simply unknown.
            note="Bexar County only (2.01M of the CBSA's 2.61M), the same "
                 "core-county scope Seattle uses. Verified live: 615,867/710,772 "
                 "parcels dated (87%), 55,146 built 2012-2018, and 23,876 dated "
                 "2023-2026 — a live appraisal roll, not a frozen snapshot. "
                 "Serializes 1000 rows with geometry, so it takes the service's "
                 "advertised maxRecordCount rather than an override.",
        ),
    ),
    "baltimore": (
        ParcelSource(
            key="md_parcel_boundaries",
            city="baltimore",
            label="Maryland Parcel Boundaries (MDP/SDAT)",
            kind="arcgis",
            url=("https://mdgeodata.md.gov/imap/rest/services/PlanningCadastre/"
                 "MD_ParcelBoundaries/MapServer/0"),
            # YEARBLT is esriFieldTypeString(4), NOT a number: server-side
            # predicates on it must be string comparisons (`YEARBLT > '1800'`) or
            # the service answers HTTP 400. `normalize_years` coerces it here, and
            # the "0000" missing-sentinel falls out below MIN_PLAUSIBLE_YEAR.
            year_col="YEARBLT",
            id_col="ACCTID",
            geom_kind="parcel",
            primary=True,
            # OBJECTID is load-bearing, not decorative: `_arcgis_keyset_pages`
            # advances on it and raises if it is absent. Verified live that this
            # MapServer answers `f=geojson` with the feature `id` populated from
            # OBJECTID, so the keyset pager works against it unmodified.
            #
            # This layer has no demolition and no "effective year" column — the
            # trap that makes Florida's EFF_YR_BLT and King County's YrRenovated
            # unusable (a 1920 rowhouse gut-rehabbed in 2016 would enter the 2016
            # construction cohort) simply does not exist here. YEARBLT is the
            # actual construction year and it is the only year on the record.
            select=("OBJECTID", "ACCTID", "YEARBLT", "LU", "SQFTSTRC",
                    "JURSCODE", "CT2020"),
            # The service advertises maxRecordCount=1000 and MEANS it only for
            # attribute-only responses: 1000 rows with `returnGeometry=false`
            # succeed, the identical 1000 rows WITH polygons return a
            # deterministic HTTP 500 once the page reaches heavy geometry (it
            # killed the first Baltimore pull at page 21). 500 is verified to
            # serialize at that same offset; see ARCGIS_MIN_PAGE for the general
            # fallback if some other stretch turns out to be heavier still.
            page_size=500,
            # The tract-derived envelope is a rectangle and the CBSA is not, so
            # the box alone selects 1,731,656 parcels against the 1,044,818 in
            # the metro. JURSCODE is the exception that earns an attribute
            # filter: unlike Florida's unindexed CO_NO it answers a count over
            # all seven jurisdictions immediately, and the codes are the state's
            # own stable mnemonics rather than a FIPS table we would have to
            # hardcode. DESCLU, by contrast, 500s — do not filter on it.
            where=("JURSCODE IN ('BACI','BACO','ANNE','HOWA','HARF','CARR',"
                   "'QUEE')"),
            note="statewide parcel polygons carrying the year themselves — no "
                 "`year_from` join and no bulk archive. Downloaded through the "
                 "tract-derived bbox, which also drops the 'NOT LOCATED' rows: "
                 "unmapped accounts with null geometry that would otherwise "
                 "inflate the denominator (they cannot match a spatial filter). "
                 "Measured over the 7 CBSA jurisdictions: 876,532/1,044,818 "
                 "parcels dated, 18,870/29,147 commercial (64.7%), 34,327 built "
                 "2012-2018.",
        ),
    ),
    "chicago": (
        ParcelSource(
            key="chicago_footprints",
            city="chicago",
            label="City of Chicago Building Footprints",
            kind="socrata",
            url="https://data.cityofchicago.org",
            dataset_id="syp8-uezg",
            year_col="year_built",
            # The layer carries a `demolished` DATE, so Chicago gets the same
            # demolition handling NYC has: a building torn down before the
            # baseline year must leave the baseline stock, or every tract's
            # denominator is overstated and its treatment intensity understated.
            demolition_col="demolished",
            id_col="bldg_id",
            geom_kind="footprint",
            primary=True,
            # Field names are the TRUNCATED shapefile-era ones (`bldg_statu`,
            # `no_of_unit`, `bldg_sq_fo`), not the spelled-out display labels.
            # Verified against /api/views/syp8-uezg/columns.json — Socrata answers
            # a wrong name with a 400, not a silently missing column.
            select=("bldg_id", "year_built", "demolished", "bldg_statu",
                    "stories", "no_of_unit", "bldg_sq_fo", "the_geom"),
            note="all building types, dated; direct DoITT analogue, no parcel join",
        ),
        ParcelSource(
            key="cook_improvement_chars",
            city="chicago",
            label="Cook County Assessor - Single/Multi-Family Improvement Characteristics",
            kind="socrata",
            url="https://datacatalog.cookcountyil.gov",
            dataset_id="x54s-btds",
            year_col="char_yrblt",
            id_col="pin",
            geom_kind="parcel",
            primary=False,
            select=("pin", "char_yrblt", "tax_year", "class"),
            note="residential <7 units ONLY — supplement/cross-check, never the "
                 "cohort source on its own (excludes commercial + large multifamily)",
        ),
    ),
    # King County is the one registered city whose geometry layer carries NO year
    # at all: KingCo_Parcels is (OBJECTID, MAJOR, MINOR, PIN, Shape) and nothing
    # else — verified against the layer's own metadata. The years live in two
    # separate Assessor CAMA extracts that must be joined on PIN *before* the
    # spatial join to Microsoft footprints can run. Hence `year_from`.
    "seattle": (
        ParcelSource(
            key="kingco_parcels",
            city="seattle",
            label="King County parcel geometry",
            kind="arcgis",
            url=("https://gismaps.kingcounty.gov/arcgis/rest/services/"
                 "Property/KingCo_Parcels/MapServer/0"),
            year_col="year_built",
            id_col="PIN",
            geom_kind="parcel",
            primary=True,
            # OBJECTID is required, not decorative: `_arcgis_keyset_pages` pages
            # on it and raises if it is absent from the response.
            select=("OBJECTID", "PIN", "MAJOR", "MINOR"),
            join_key="PIN",
            year_from=("kingco_resbldg", "kingco_commbldg"),
            note="no year of its own; years joined on PIN from the Assessor's "
                 "residential + commercial building extracts",
        ),
        # BOTH extracts are needed and neither is optional. Residential covers
        # buildings with 1-3 living units only (~533k records); commercial covers
        # everything else INCLUDING APARTMENTS (~42k records). Dropping the
        # commercial file would delete exactly the multifamily and mixed-use
        # towers that drive Seattle's 2010s construction boom, leaving a cohort
        # panel made of single-family infill.
        ParcelSource(
            key="kingco_resbldg",
            city="seattle",
            label="King County Assessor - Residential Building extract",
            kind="csv_archive",
            url="https://aqua.kingcounty.gov/extranet/assessor/Residential%20Building.zip",
            member="EXTR_ResBldg.csv",
            # YrBuilt is the actual construction year. YrRenovated is deliberately
            # NOT used, for the same reason Florida's EFF_YR_BLT is not: it is
            # reset by major remodels, and would date a 1920 bungalow gut-rehabbed
            # in 2016 into the 2016 construction cohort — inventing treatment on a
            # parcel where no new structure exists.
            year_col="YrBuilt",
            id_col="PIN",
            geom_kind="table",
            primary=False,
            key_cols=("Major", "Minor"),
            key_widths=(6, 4),
            size_col="SqFtTotLiving",
            select=("Major", "Minor", "BldgNbr", "YrBuilt", "SqFtTotLiving",
                    "NbrLivingUnits"),
            note="one row per residential building (1-3 units); weekly refresh",
        ),
        ParcelSource(
            key="kingco_commbldg",
            city="seattle",
            label="King County Assessor - Commercial Building extract",
            kind="csv_archive",
            url="https://aqua.kingcounty.gov/extranet/assessor/Commercial%20Building.zip",
            member="EXTR_CommBldg.csv",
            year_col="YrBuilt",
            id_col="PIN",
            geom_kind="table",
            primary=False,
            key_cols=("Major", "Minor"),
            key_widths=(6, 4),
            size_col="BldgGrossSqFt",
            select=("Major", "Minor", "BldgNbr", "NbrBldgs", "YrBuilt",
                    "BldgGrossSqFt", "PredominantUse"),
            note="one row per commercial building; INCLUDES APARTMENTS. Verified "
                 "live carrying YrBuilt through 2026, 149-355 buildings/year over "
                 "2010-2024, so it is a live roll and not a frozen snapshot.",
        ),
    ),
    "tampa": (
        ParcelSource(
            key="fl_cadastral",
            city="tampa",
            label="Florida Statewide Cadastral (FDOR tax roll, bulk)",
            kind="http_archive",
            url="https://publicfiles.dep.state.fl.us/otis/gis/data/Cadastral_Statewide.zip",
            year_col="ACT_YR_BLT",
            id_col="PARCEL_ID",
            geom_kind="parcel",
            primary=True,
            # ACT_YR_BLT is the *actual* year built; EFF_YR_BLT is the assessor's
            # "effective" year, reset by major renovation, which would date a
            # 1920 house rehabbed in 2018 into a 2018 cohort. NO_BULDNG lets the
            # coverage report quantify the multi-building-parcel approximation
            # rather than merely warning about it.
            select=("PARCEL_ID", "ACT_YR_BLT", "CO_NO", "NO_BULDNG"),
            note="~2.6 GB statewide zip, refreshed from the county property "
                 "appraisers' annual roll; the city's window is read out with a "
                 "GDAL bbox push-down. Bulk rather than the ArcGIS service "
                 "because that service rejects returnIdsOnly, returnCountOnly "
                 "AND geometry filters (all HTTP 400), leaving only unfiltered "
                 "OBJECTID keyset paging over 10.8M parcels. NOTE: FGDL removed "
                 "its parcel layers in Oct 2025 at the state GIO's request, so "
                 "issue #36's 'FGDL backbone' no longer exists.",
        ),
        ParcelSource(
            key="fl_cadastral_service",
            city="tampa",
            label="Florida Statewide Cadastral (ArcGIS service)",
            kind="arcgis",
            url=("https://services9.arcgis.com/Gh9awoU677aKree0/arcgis/rest/"
                 "services/Florida_Statewide_Cadastral/FeatureServer/0"),
            year_col="ACT_YR_BLT",
            id_col="PARCEL_ID",
            geom_kind="parcel",
            primary=False,
            select=("OBJECTID", "PARCEL_ID", "ACT_YR_BLT", "CO_NO", "NO_BULDNG"),
            note="fallback only. Verified live to carry 2021/2022 ACT_YR_BLT, but "
                 "supports neither id/count-only queries nor spatial filters, and "
                 "CO_NO is unindexed (a county where-clause times out), so a "
                 "city extract means keyset-paging the whole 10.8M-row layer.",
        ),
    ),
    # Nashville has no public bulk year-built source covering its core county.
    # Three candidates were checked live and all three fail:
    #
    #  1. TN Comptroller statewide parcels + `Assessment_Data_##.dbf`. The state
    #     publishes this per county from its IMPACT CAMA system, but only for the
    #     86 counties IN that system. Davidson, Rutherford and Williamson run
    #     their own CAMA and are explicitly excluded — three of the five CBSA
    #     counties, including Davidson itself (~40% of CBSA population). What
    #     remains (Wilson, Sumner) is the exurban ring, which is not Nashville.
    #  2. Metro Nashville's own parcel service
    #     (maps.nashville.gov/arcgis/.../Cadastral/Parcels/MapServer/0). Verified
    #     field-by-field: it carries ownership, land use, and appraised values
    #     (LandAppr/ImprAppr/TotlAppr) but NO construction year. No other folder
    #     on that server holds buildings, structures or CAMA data.
    #  3. Nashville "Building Permits Issued" (ArcGIS item
    #     2576bfb2d74f418b8ba8c4538e4f729f). Would be a defensible cohort source —
    #     except its own metadata says it is "a rolling three-year period", so it
    #     cannot reach back to a 2012-2018 panel.
    #
    # The unblocking move is administrative, not technical: the Davidson County
    # Assessor (padctn.org) maintains year built in CAMA and can supply a bulk
    # extract on request. Given a PIN/APN-keyed CSV with a year, Nashville needs
    # no new code — it is the same `year_from` table join Seattle now uses, and
    # the parcel geometry above is already public.
    "nashville": (),
}


def sources_for(city: str, *, primary_only: bool = False) -> tuple[ParcelSource, ...]:
    src = PARCEL_SOURCES.get(city, ())
    return tuple(s for s in src if s.primary) if primary_only else src


# --------------------------------------------------------------------------- #
# HTTP                                                                         #
# --------------------------------------------------------------------------- #

def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = (
        "ny-state-aerial-imagery-prototype/1.0 (academic research; "
        "building-level wealth estimation)"
    )
    token = os.environ.get("SOCRATA_APP_TOKEN")
    if token:
        # Optional: raises the anonymous throttling ceiling. Not required.
        s.headers["X-App-Token"] = token
    return s


def _get_json(session, url, params, *, sleep_fn=time.sleep):
    """GET with retries, returning parsed JSON.

    Only 429/5xx are retried. A 4xx is a *permanent* client error — a mistyped
    column, a bad filter — so retrying it four times just delays the error by
    the backoff ladder and tells you nothing new.

    The server's response body is included in the raised message. Socrata answers
    a bad ``$select`` with an explicit "no such column" naming the offender, and
    swallowing that turns a one-line fix into a guessing game.
    """
    last = None
    for attempt in range(RETRIES):
        try:
            resp = session.get(url, params=params, timeout=HTTP_TIMEOUT_S)
        except Exception as exc:                       # noqa: BLE001  (transport)
            last = exc
            if attempt < RETRIES - 1:
                sleep_fn(BACKOFF_S * (2 ** attempt))
            continue

        if resp.status_code < 400:
            payload = resp.json()
            # ArcGIS returns HTTP 200 with an {"error": {...}} body for a
            # rejected query. Left unchecked the caller sees a payload with no
            # "features"/"objectIds" key and reads it as an EMPTY RESULT — which
            # is how a failed Florida query surfaced as "0 features in the
            # selection" and a zero-row download rather than an error.
            if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
                err = payload["error"]
                raise RuntimeError(
                    f"GET {url} returned HTTP {resp.status_code} with an embedded "
                    f"error: code={err.get('code')} message={err.get('message')!r} "
                    f"details={err.get('details')}\n  params: {params}")
            return payload

        detail = (resp.text or "")[:500].strip()
        last = f"HTTP {resp.status_code}: {detail}"
        if 400 <= resp.status_code < 500 and resp.status_code != 429:
            raise RuntimeError(f"GET {url} failed permanently — {last}\n"
                               f"  params: {params}")
        if attempt < RETRIES - 1:
            sleep_fn(BACKOFF_S * (2 ** attempt))
    raise RuntimeError(f"GET {url} failed after {RETRIES} attempts: {last}")


def _write_atomic(gdf, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".part.{os.getpid()}")
    if isinstance(gdf, gpd.GeoDataFrame):
        gdf.to_parquet(tmp)
    else:
        gdf.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Socrata                                                                      #
# --------------------------------------------------------------------------- #

def fetch_socrata(source: ParcelSource, *, page_size: int = SOCRATA_PAGE,
                  max_pages: int | None = None, session=None,
                  sleep_fn=time.sleep, verbose: bool = True):
    """Page a Socrata dataset to a GeoDataFrame (or DataFrame if non-spatial).

    Pages are cached individually under ``source.cache_dir`` so an interrupted
    download resumes instead of restarting. ``$order=:id`` is mandatory — without
    it Socrata's offset paging is not stable and rows silently repeat or vanish.
    """
    session = session or _session()
    fmt = "geojson" if source.geom_kind == "footprint" else "json"
    url = f"{source.url.rstrip('/')}/resource/{source.dataset_id}.{fmt}"
    source.cache_dir.mkdir(parents=True, exist_ok=True)

    frames, offset, page_idx = [], 0, 0
    while True:
        if max_pages is not None and page_idx >= max_pages:
            break
        page_path = source.cache_dir / f"page_{page_idx:05d}.json"
        if page_path.exists():
            payload = json.loads(page_path.read_text())
        else:
            params = {"$limit": page_size, "$offset": offset, "$order": ":id"}
            if source.select:
                params["$select"] = ",".join(source.select)
            payload = _get_json(session, url, params, sleep_fn=sleep_fn)
            tmp = page_path.with_suffix(".part")
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, page_path)

        n = _page_len(payload, fmt)
        if verbose:
            print(f"    {source.key}: page {page_idx} -> {n:,} rows "
                  f"(offset {offset:,})")
        # Terminate ONLY on an empty page, and advance by the rows actually
        # returned rather than by the requested page size. Socrata clamps $limit
        # server-side on some endpoints (the geojson one caps lower than the JSON
        # one), so treating a short page as the last page would silently truncate
        # the download at the first request — and a partial footprint universe
        # understates every tract's baseline area with nothing raising. The cost
        # of this is one extra empty request at the end.
        if n == 0:
            break
        frames.append(_page_frame(payload, fmt))
        offset += n
        page_idx += 1

    if not frames:
        return gpd.GeoDataFrame() if fmt == "geojson" else pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if fmt == "geojson":
        out = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
    return out


def _page_len(payload, fmt: str) -> int:
    return len(payload.get("features", []) if fmt == "geojson" else payload)


def _page_frame(payload, fmt: str):
    if fmt == "geojson":
        if not payload.get("features"):
            return gpd.GeoDataFrame()
        return gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    return pd.DataFrame(payload)


# --------------------------------------------------------------------------- #
# ArcGIS                                                                       #
# --------------------------------------------------------------------------- #

def envelope_params(bbox: tuple[float, float, float, float] | None) -> dict:
    """ArcGIS spatial-filter params for a WGS84 ``(xmin, ymin, xmax, ymax)`` box.

    Spatial filtering rather than an attribute filter is a deliberate choice for
    statewide parcel layers. The geometry column is spatially indexed while the
    county column generally is not, so a ``CO_NO`` where-clause over Florida's
    10.8M parcels times out, and the county codes are the FDOR's own numbering
    rather than FIPS — a table we would have to hardcode and could silently get
    wrong. An envelope derived from the city's own tract geometry needs no such
    table and is exact.
    """
    if bbox is None:
        return {}
    xmin, ymin, xmax, ymax = (float(v) for v in bbox)
    return {
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
    }


def _arcgis_keyset_pages(source, session, *, page_size, where, bbox, oid_field,
                         max_pages, sleep_fn, verbose):
    """Yield pages via keyset pagination on OBJECTID.

    ``OBJECTID > last_seen ORDER BY OBJECTID`` rather than ``resultOffset``, for
    two reasons measured against Florida's statewide cadastral:

    * ``returnIdsOnly`` and ``returnCountOnly`` are simply rejected by some
      services (that one answers both with HTTP 400), so a paginator that needs
      an id enumeration up front cannot run at all;
    * deep ``resultOffset`` degrades badly on a 10.8M-row layer, because the
      server must walk every skipped row, whereas an indexed ``OBJECTID >``
      predicate is O(log n) per page.

    Terminates on an empty page and advances by the largest id actually seen, so
    a server-side clamp of ``resultRecordCount`` cannot truncate the download.

    A page that fails outright is retried at half the size, down to
    :data:`ARCGIS_MIN_PAGE` — see that constant for the failure it exists to
    survive. The reduction is *sticky*: once a layer has proved it cannot
    serialize a given page size, resetting to the advertised value would spend a
    doomed request (and its full retry budget) on every subsequent page. Shrinking
    is always safe here because the pager advances by the largest id actually
    seen rather than by an assumed stride, so pages of different sizes compose
    into the same complete download — which is also why a cache written by an
    earlier run at a different page size stays valid.
    """
    last_oid = -1
    page_idx = 0
    while max_pages is None or page_idx < max_pages:
        page_path = source.cache_dir / f"page_{page_idx:05d}.json"
        if page_path.exists():
            payload = json.loads(page_path.read_text())
        else:
            clause = f"{oid_field} > {last_oid}"
            if where and where != "1=1":
                clause = f"({where}) AND {clause}"
            params = {
                "where": clause,
                "outFields": ",".join(source.select) if source.select else "*",
                "returnGeometry": "true",
                "outSR": "4326",
                "orderByFields": oid_field,
                "f": "geojson",
            }
            params.update(envelope_params(bbox))
            while True:
                try:
                    payload = _get_json(
                        session, f"{source.url.rstrip('/')}/query",
                        {**params, "resultRecordCount": page_size},
                        sleep_fn=sleep_fn)
                    break
                except RuntimeError:
                    if page_size <= ARCGIS_MIN_PAGE:
                        raise
                    page_size = max(ARCGIS_MIN_PAGE, page_size // 2)
                    print(f"    ⚠️ {source.key}: page {page_idx} failed; retrying "
                          f"at resultRecordCount={page_size} (the service "
                          f"advertises more than it can serialize)")
            tmp = page_path.with_suffix(".part")
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, page_path)

        feats = payload.get("features") or []
        if not feats:
            break
        oids = [f.get("id") for f in feats if f.get("id") is not None]
        if not oids:
            oids = [f.get("properties", {}).get(oid_field) for f in feats]
            oids = [o for o in oids if o is not None]
        if not oids:
            raise RuntimeError(
                f"{source.key}: keyset paging needs {oid_field} in the response; "
                f"add it to the source's `select`.")
        last_oid = max(int(o) for o in oids)
        if verbose:
            print(f"    {source.key}: page {page_idx} -> {len(feats):,} rows "
                  f"(through {oid_field} {last_oid})")
        yield feats
        page_idx += 1


def arcgis_object_ids(source: ParcelSource, *, session=None,
                      where: str = "1=1", bbox=None,
                      sleep_fn=time.sleep) -> list[int]:
    """Every OBJECTID matching the filter, via ``returnIdsOnly``.

    This is what makes ArcGIS paging exact: ``resultOffset`` is unreliable on
    large layers (it can skip or duplicate under concurrent edits or when the
    service reorders), whereas an explicit id list partitions the layer with no
    ambiguity. ``returnIdsOnly`` is also not subject to ``maxRecordCount``, so a
    single call enumerates the whole selection.
    """
    session = session or _session()
    params = {"where": where, "returnIdsOnly": "true", "f": "json"}
    params.update(envelope_params(bbox))
    payload = _get_json(session, f"{source.url.rstrip('/')}/query", params,
                        sleep_fn=sleep_fn)
    return sorted(int(i) for i in payload.get("objectIds") or [])


def fetch_arcgis(source: ParcelSource, *, page_size: int | None = None,
                 where: str = "1=1", bbox=None, session=None,
                 max_pages: int | None = None,
                 sleep_fn=time.sleep, verbose: bool = True) -> gpd.GeoDataFrame:
    """Page an ArcGIS layer by OBJECTID ranges to a GeoDataFrame.

    ``bbox`` is an optional WGS84 ``(xmin, ymin, xmax, ymax)`` envelope; see
    :func:`envelope_params` for why that is preferred over an attribute filter.
    """
    session = session or _session()
    if page_size is None:
        page_size = source.page_size
    if page_size is None:
        meta = _get_json(session, source.url.rstrip("/"), {"f": "json"},
                         sleep_fn=sleep_fn)
        page_size = int(meta.get("maxRecordCount") or ARCGIS_DEFAULT_PAGE)
    # An explicit caller argument still wins, so a smoke test can narrow further.
    if where == "1=1" and source.where:
        where = source.where

    oid_field = "OBJECTID"
    source.cache_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
        for feats in _arcgis_keyset_pages(
            source, session, page_size=page_size, where=where, bbox=bbox,
            oid_field=oid_field, max_pages=max_pages, sleep_fn=sleep_fn,
            verbose=verbose)
    ]

    if not frames:
        return gpd.GeoDataFrame()
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                            geometry="geometry", crs="EPSG:4326")


# --------------------------------------------------------------------------- #
# Bulk archive                                                                 #
# --------------------------------------------------------------------------- #

def download_archive(source: ParcelSource, *, chunk_bytes: int = 8 << 20,
                     session=None, verbose: bool = True) -> Path:
    """Stream a bulk archive to disk, resuming a partial download via Range.

    Returns the local path. The archive is written to ``.part`` and renamed only
    once the full length has arrived, so an interrupted download can never be
    mistaken for a complete one.
    """
    session = session or _session()
    # Beside the city's outputs, not under the paging cache: this is one large
    # file, and someone looking for a 2.6 GB download should not have to find it
    # inside a directory named "_pages".
    dest = source.out_path.parent / Path(source.url).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if verbose:
            print(f"    {source.key}: archive already present ({dest})")
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with session.get(source.url, headers=headers, stream=True,
                     timeout=HTTP_TIMEOUT_S) as resp:
        resp.raise_for_status()
        # 200 to a Range request means the server ignored it: restart from zero
        # rather than appending to what we already have and corrupting the file.
        if have and resp.status_code == 200:
            have = 0
            part.unlink(missing_ok=True)
        total = int(resp.headers.get("Content-Length", 0)) + have
        mode = "ab" if have else "wb"
        got = have
        with open(part, mode) as fh:
            for block in resp.iter_content(chunk_bytes):
                fh.write(block)
                got += len(block)
                if verbose and total:
                    print(f"\r    {source.key}: {got / 2**30:5.2f} / "
                          f"{total / 2**30:5.2f} GiB", end="", flush=True)
    if verbose:
        print()
    os.replace(part, dest)
    return dest


ARCHIVE_DATASET_SUFFIXES = (".gdb", ".gpkg", ".shp", ".geojson")


def archive_dataset_uri(path: Path) -> str:
    """GDAL URI for the dataset *inside* an archive.

    ``/vsizip/foo.zip`` alone is not openable unless the archive root happens to
    be the dataset; Florida's ships a File Geodatabase directory
    (``CADASTRAL_DOR.gdb/``) one level down, and GDAL must be pointed at that.
    Rather than hardcode the name, find the first recognised dataset entry.
    """
    if not str(path).lower().endswith(".zip"):
        return str(path)

    import zipfile

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    roots = []
    for name in names:
        head = name.split("/")[0]
        if head.lower().endswith(ARCHIVE_DATASET_SUFFIXES) and head not in roots:
            roots.append(head)
    if not roots:
        for name in names:
            if name.lower().endswith(ARCHIVE_DATASET_SUFFIXES):
                return f"/vsizip/{path}/{name}"
        raise RuntimeError(
            f"{path}: no recognised dataset inside the archive "
            f"(looked for {ARCHIVE_DATASET_SUFFIXES}); first entries: {names[:5]}")
    return f"/vsizip/{path}/{roots[0]}"


def read_archive(source: ParcelSource, path: Path, *, bbox=None,
                 layer: str | None = None, verbose: bool = True
                 ) -> gpd.GeoDataFrame:
    """Read a (possibly huge) archive, pushing the bbox filter down to GDAL.

    The spatial filter is applied by the driver using the dataset's own index, so
    a statewide 10.8M-parcel geodatabase yields only the city's parcels without
    ever materialising the rest. ``columns`` likewise avoids carrying the ~120
    tax-roll attributes we do not use.

    ``bbox`` is given in WGS84 and reprojected into the dataset's CRS here.
    pyogrio interprets ``bbox`` in the *dataset's* coordinates, so handing it
    degrees against Florida's EPSG:6439 Albers metres would match nothing and
    return an empty frame — a silent wrong answer, not an error.
    """
    import pyogrio

    uri = archive_dataset_uri(path)
    info_kwargs = {"layer": layer} if layer is not None else {}
    info = pyogrio.read_info(uri, **info_kwargs)
    native_crs = info.get("crs")

    kwargs = dict(info_kwargs)
    if bbox is not None:
        if native_crs and str(native_crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
            from pyproj import Transformer
            tr = Transformer.from_crs("EPSG:4326", native_crs, always_xy=True)
            xs, ys = tr.transform([bbox[0], bbox[2], bbox[0], bbox[2]],
                                  [bbox[1], bbox[3], bbox[3], bbox[1]])
            if not all(map(np.isfinite, list(xs) + list(ys))):
                raise RuntimeError(
                    f"reprojecting the bbox into {native_crs} gave non-finite "
                    f"values; if PROJ_NETWORK=ON and cdn.proj.org is "
                    f"unreachable, set PROJ_NETWORK=OFF and retry.")
            bbox = (min(xs), min(ys), max(xs), max(ys))
        kwargs["bbox"] = tuple(float(v) for v in bbox)
    if source.select:
        # Geometry always comes back; only attribute columns are listed.
        kwargs["columns"] = [c for c in source.select if c.lower() != "geometry"]

    if verbose:
        print(f"    {source.key}: reading {uri}")
        print(f"      layer={info.get('name') or layer} "
              f"features={info.get('features'):,} crs={native_crs}")
        if bbox is not None:
            print(f"      bbox push-down (in {native_crs}): "
                  f"{tuple(round(v, 1) for v in kwargs['bbox'])}")

    try:
        gdf = gpd.read_file(uri, engine="pyogrio", **kwargs)
    except TypeError:
        kwargs.pop("columns", None)          # older pyogrio without `columns=`
        gdf = gpd.read_file(uri, engine="pyogrio", **kwargs)

    if len(gdf) == 0:
        raise RuntimeError(
            f"{source.key}: the bbox-filtered read returned no features. The "
            f"archive has {info.get('features'):,} in {native_crs}; check the "
            f"bbox actually overlaps the dataset's extent.")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def fetch_http_archive(source: ParcelSource, *, bbox=None, session=None,
                       layer: str | None = None, verbose: bool = True,
                       **_ignored) -> gpd.GeoDataFrame:
    """Download a bulk archive once, then read the city's window out of it."""
    path = download_archive(source, session=session, verbose=verbose)
    return read_archive(source, path, bbox=bbox, layer=layer, verbose=verbose)


def archive_member(path: Path, member: str | None) -> str:
    """Resolve ``member`` to an entry inside a zip, tolerating repackaging.

    Matched on basename and case-insensitively, because these archives are
    regenerated weekly by the county and have historically moved their CSV in and
    out of a top-level directory. An exact-path match would turn that into a
    build failure on a file that is plainly present.
    """
    import zipfile

    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
    if member is None:
        csvs = [n for n in names if n.lower().endswith(".csv")]
        if len(csvs) == 1:
            return csvs[0]
        raise RuntimeError(
            f"{path}: `member` is required — the archive holds {len(csvs)} CSVs "
            f"({csvs[:5]})")
    for name in names:
        if name == member:
            return name
    wanted = member.rsplit("/", 1)[-1].lower()
    hits = [n for n in names if n.rsplit("/", 1)[-1].lower() == wanted]
    if len(hits) == 1:
        return hits[0]
    raise RuntimeError(
        f"{path}: no member matching {member!r} "
        f"({'ambiguous: ' + str(hits) if hits else 'archive holds ' + str(names[:8])})")


def fetch_csv_archive(source: ParcelSource, *, session=None,
                      verbose: bool = True, **_ignored) -> pd.DataFrame:
    """Download a zip once and read one non-spatial CSV out of it.

    Read with ``encoding="latin-1"`` and ``dtype=str``. Both are deliberate:
    these are mainframe extracts that carry occasional non-UTF-8 bytes in address
    fields (which would abort a strict decode over an otherwise perfect file),
    and their numeric columns are space-padded fixed-width strings (``"5  "``,
    ``"512      "``) that pandas' type inference turns into object columns
    anyway. Coercion happens once, explicitly, in
    :func:`build_year_built.building_year_table`.
    """
    path = download_archive(source, session=session, verbose=verbose)
    name = archive_member(path, source.member)

    import zipfile

    usecols = list(source.select) if source.select else None
    with zipfile.ZipFile(path) as zf, zf.open(name) as fh:
        try:
            frame = pd.read_csv(fh, dtype=str, encoding="latin-1",
                                usecols=usecols)
        except ValueError as exc:
            # A renamed column is a real error, but the message pandas raises
            # ("Usecols do not match columns") does not say which one, and these
            # files have 25-40 columns.
            with zipfile.ZipFile(path) as zf2, zf2.open(name) as probe:
                have = list(pd.read_csv(probe, dtype=str, encoding="latin-1",
                                        nrows=0).columns)
            missing = [c for c in (usecols or []) if c not in have]
            raise RuntimeError(
                f"{source.key}: {name} is missing {missing!r}; it has {have}"
            ) from exc
    if verbose:
        print(f"    {source.key}: {len(frame):,} rows from {name}")
    return frame


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #

def source_by_key(city: str, key: str) -> ParcelSource:
    """One registered source by key, for resolving ``year_from`` references."""
    for source in PARCEL_SOURCES.get(city, ()):
        if source.key == key:
            return source
    raise KeyError(f"{city!r} has no registered source {key!r}")


def download_source(source: ParcelSource, *, force: bool = False, **kwargs):
    """Fetch one source to ``source.out_path``, skipping if already complete."""
    if source.out_path.exists() and not force:
        print(f"  ✔ {source.key}: already downloaded ({source.out_path})")
        return gpd.read_parquet(source.out_path) if source.geom_kind != "table" \
            else pd.read_parquet(source.out_path)

    print(f"  ↓ {source.label} ({source.kind})")
    fetch = {"socrata": fetch_socrata, "arcgis": fetch_arcgis,
             "http_archive": fetch_http_archive,
             "csv_archive": fetch_csv_archive}[source.kind]
    # A non-spatial table has no geometry to filter, and passing an envelope down
    # to it would be a TypeError on a keyword it never accepts.
    if source.geom_kind == "table":
        kwargs = {k: v for k, v in kwargs.items() if k != "bbox"}
    frame = fetch(source, **kwargs)
    if len(frame) == 0:
        raise RuntimeError(f"{source.key}: download returned zero rows")
    _write_atomic(frame, source.out_path)
    print(f"  ✔ {source.key}: {len(frame):,} rows -> {source.out_path}")
    return frame


def download_city(city: str, *, primary_only: bool = True, force: bool = False,
                  **kwargs) -> dict[str, Path]:
    """Fetch a city's registered sources. Returns ``{source key: path}``."""
    srcs = sources_for(city, primary_only=primary_only)
    if not srcs:
        raise KeyError(f"no parcel sources registered for {city!r}")
    out = {}
    for source in srcs:
        download_source(source, force=force, **kwargs)
        out[source.key] = source.out_path
    return out


if __name__ == "__main__":                                  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("city", choices=sorted(PARCEL_SOURCES))
    ap.add_argument("--all-sources", action="store_true",
                    help="also fetch non-primary supplements")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="stop after N pages (smoke test)")
    args = ap.parse_args()
    download_city(args.city, primary_only=not args.all_sources,
                  force=args.force, max_pages=args.max_pages)
