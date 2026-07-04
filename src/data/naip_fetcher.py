"""NAIP crop fetcher (Planetary Computer STAC) — US-scale robust version.

Improvements over the original NYC prototype (issue #25):
  * Geodesic bbox: the meters->degrees window is computed with true WGS84
    geodesic offsets (pyproj.Geod) instead of a flat-earth 111km/deg
    approximation, so the footprint is exact at any latitude. (Offsetting in
    EPSG:5070 was considered and rejected: Albers is equal-area, not conformal —
    its linear scale distorts up to ~20% far from the standard parallels.)
  * Observability: no more silent ``(None, None)`` — every failure is tagged with
    a mode (``no_items`` / ``asset_missing`` / ``read_error``) and counted in a
    process-wide stats dict (:func:`get_fetch_stats`) that the cache manager logs.
  * Silent NIR padding is now flagged: when only the 3-band ``visual`` asset is
    available, the zero-padded NIR is reported via ``nir_padded`` so callers can
    track (or reject) synthetic-NIR crops.
  * Deterministic item choice when ``year_hint`` is None (newest first, id tiebreak).
  * Full-coverage item preference (issue: stretched crops near DOQQ edges): the
    STAC search returns every quarter-quad that merely *intersects* the request
    bbox, and picking purely by year could select a tile that covers only a
    sliver of the window; rasterio then clips the window to the raster and
    resamples that sliver to ``out_pixels`` square — a heavily stretched crop.
    Items whose footprint fully contains the bbox are now preferred (NAIP DOQQs
    overlap ~300 m, so a neighboring tile almost always qualifies). If no item
    fully covers the window, the read falls back to ``boundless=True`` with
    zero fill — geometry stays correct, the missing margin is black — and the
    crop is flagged via ``partial_coverage``.

GSD note: NAIP is 0.6 m or 1 m depending on state/year, but crops are always
resampled so meters/pixel = crop_size_meters / out_pixels — the *effective*
resolution is constant by construction; the native GSD only affects sharpness.
"""

import threading
from dataclasses import dataclass

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from pyproj import Geod
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

_local = threading.local()
_GEOD = Geod(ellps="WGS84")

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MAX_RETRY="3",
    GDAL_HTTP_RETRY_DELAY="2",
)

# Process-wide fetch statistics (thread-safe). Reset with reset_fetch_stats().
_STATS_LOCK = threading.Lock()
_FETCH_STATS = {"ok": 0, "no_items": 0, "asset_missing": 0, "read_error": 0,
                "nir_padded": 0, "partial_coverage": 0}


def get_fetch_stats() -> dict:
    """Snapshot of cumulative fetch outcome counts for this process."""
    with _STATS_LOCK:
        return dict(_FETCH_STATS)


def reset_fetch_stats() -> None:
    with _STATS_LOCK:
        for k in _FETCH_STATS:
            _FETCH_STATS[k] = 0


def _count(key: str) -> None:
    with _STATS_LOCK:
        _FETCH_STATS[key] += 1


@dataclass
class NaipFetchResult:
    """Outcome of one fetch: crop (or None) + provenance/diagnostics."""
    crop: np.ndarray | None          # (C, H, W) uint8, or None on failure
    actual_year: int | None          # flight year of the item actually used
    nir_padded: bool = False         # True when NIR was zero-padded (visual asset)
    partial_coverage: bool = False   # True when the item didn't cover the full
                                     # window (missing margin is zero-filled)
    failure: str | None = None       # None | no_items | asset_missing | read_error


def get_catalog() -> pystac_client.Client:
    """One STAC client per thread — avoids connection pool contention."""
    if not hasattr(_local, "catalog"):
        _local.catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
    return _local.catalog


def lonlat_bbox(lon: float, lat: float, crop_size_meters: float) -> list[float]:
    """EPSG:4326 bbox of a square window of ``crop_size_meters`` centered on (lon, lat).

    Uses WGS84 geodesic offsets (east/west/north/south by half the window), so
    the ground size is exact at any latitude — unlike the previous constant-
    111km/deg approximation (or projected offsets in an equal-area CRS).
    """
    half = crop_size_meters / 2.0
    lon_e, _, _ = _GEOD.fwd(lon, lat, 90.0, half)
    lon_w, _, _ = _GEOD.fwd(lon, lat, 270.0, half)
    _, lat_n, _ = _GEOD.fwd(lon, lat, 0.0, half)
    _, lat_s, _ = _GEOD.fwd(lon, lat, 180.0, half)
    return [min(lon_w, lon_e), min(lat_s, lat_n),
            max(lon_w, lon_e), max(lat_s, lat_n)]


def item_contains_bbox(item, bbox: list[float]) -> bool:
    """True if the STAC item's EPSG:4326 bbox fully contains ``bbox``.

    Items without a usable bbox count as non-containing (they sort after
    items known to cover the window; the boundless-read guard catches any
    residual under-coverage).
    """
    ib = getattr(item, "bbox", None)
    if not ib or len(ib) < 4:
        return False
    return ib[0] <= bbox[0] and ib[1] <= bbox[1] and ib[2] >= bbox[2] and ib[3] >= bbox[3]


def window_exceeds_raster(window, width: int, height: int, tol: float = 1e-3) -> bool:
    """True if a float window sticks out of a (width, height) raster.

    ``tol`` (in pixels) absorbs float noise from CRS round-trips so exact
    edge-aligned reads are not sent down the boundless path.
    """
    return (
        window.col_off < -tol
        or window.row_off < -tol
        or window.col_off + window.width > width + tol
        or window.row_off + window.height > height + tol
    )


def fetch_naip(
    lon: float, lat: float,
    crop_size_meters: float,
    nbands: int = 4,
    out_pixels: int = 250,
    year_hint: int | None = None,
) -> NaipFetchResult:
    """Fetch a single NAIP crop from Planetary Computer.

    Returns a :class:`NaipFetchResult`; ``result.crop`` is (C, H, W) uint8 or
    None with ``result.failure`` set to the failure mode.
    """
    bbox = lonlat_bbox(lon, lat, crop_size_meters)

    try:
        items = list(get_catalog().search(
            collections=["naip"],
            bbox=bbox,
            max_items=50,
        ).items())
    except Exception:
        _count("read_error")
        return NaipFetchResult(None, None, failure="read_error")

    if not items:
        _count("no_items")
        return NaipFetchResult(None, None, failure="no_items")

    # Full-window coverage first: an intersecting-but-not-containing DOQQ yields
    # a sliver crop (see module docstring). Adjacent DOQQs overlap, so a fully
    # covering same-year tile almost always exists; when it doesn't, a fully
    # covering off-year tile beats a same-year sliver.
    if year_hint is not None:
        # Then closest flight year to the requested panel year (stable id tiebreak).
        items.sort(key=lambda item: (not item_contains_bbox(item, bbox),
                                     abs(item.datetime.year - year_hint), item.id))
    else:
        # Then deterministic default: newest first (STAC order is not guaranteed).
        items.sort(key=lambda item: (not item_contains_bbox(item, bbox),
                                     -item.datetime.year, item.id))

    item = items[0]
    actual_year = item.datetime.year

    asset = item.assets.get("image") or item.assets.get("visual")
    if not asset:
        _count("asset_missing")
        return NaipFetchResult(None, actual_year, failure="asset_missing")

    try:
        with rasterio.Env(**GDAL_ENV):
            with rasterio.open(asset.href) as src:
                native_bb = transform_bounds(CRS.from_epsg(4326), src.crs, *bbox)
                window = from_bounds(*native_bb, transform=src.transform)

                # If the window sticks out of the raster, a plain read would
                # clip it and stretch the remainder to out_pixels square. Read
                # boundless instead: geometry stays correct, missing margin is
                # zero-filled, and the crop is flagged as partial.
                partial = window_exceeds_raster(window, src.width, src.height)
                if partial:
                    _count("partial_coverage")

                # Request out_shape with out_pixels size
                # And only the required number of bands
                channels = min(nbands, src.count)
                crop = src.read(
                    indexes=list(range(1, channels + 1)),
                    window=window,
                    out_shape=(channels, out_pixels, out_pixels),
                    resampling=Resampling.bilinear,
                    boundless=partial,
                    fill_value=0,
                )

                # If requested more bands than available (3-band visual asset),
                # pad with zeros and FLAG it — synthetic NIR must be trackable.
                nir_padded = channels < nbands
                if nir_padded:
                    padded_crop = np.zeros((nbands, out_pixels, out_pixels), dtype=crop.dtype)
                    padded_crop[:channels] = crop
                    crop = padded_crop
                    _count("nir_padded")

                _count("ok")
                return NaipFetchResult(crop, actual_year, nir_padded=nir_padded,
                                       partial_coverage=partial)
    except Exception:
        _count("read_error")
        return NaipFetchResult(None, actual_year, failure="read_error")


def fetch_naip_crop(
    lon: float, lat: float,
    crop_size_meters: float,
    nbands: int = 4,
    out_pixels: int = 250,
    year_hint: int | None = None,
) -> tuple[np.ndarray | None, int | None]:
    """Backward-compatible wrapper returning ``(crop, actual_year)``.

    Prefer :func:`fetch_naip` in new code — it also reports ``nir_padded`` and
    the failure mode. Stats are counted either way.
    """
    res = fetch_naip(lon, lat, crop_size_meters, nbands, out_pixels, year_hint)
    return res.crop, res.actual_year
