"""NAIP crop fetcher (Planetary Computer STAC) — US-scale robust version.

Improvements over the original NYC prototype (issue #25):
  * Geodesic bbox: the meters->degrees window is computed with true WGS84
    geodesic offsets (pyproj.Geod) instead of a flat-earth 111km/deg
    approximation, so the footprint is exact at any latitude. (Offsetting in
    EPSG:5070 was considered and rejected: Albers is equal-area, not conformal —
    its linear scale distorts up to ~20% far from the standard parallels.)
  * Observability: no more silent ``(None, None)`` — every failure is tagged with
    a mode (``no_items`` / ``asset_missing`` / ``search_error`` / ``read_error``)
    and counted in a process-wide stats dict (:func:`get_fetch_stats`) that the
    cache manager logs.
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

Rate-limit hardening (issue: ~23% of training fetches failed as ``read_error``;
a load probe traced the cause to Planetary Computer *STAC search* throttling,
not blob reads, which were clean):
  * ``search_error`` is counted separately from ``read_error`` — the STAC
    search and the raster read are different services with different failure
    modes, and lumping them made the throttling invisible.
  * Searches retry with exponential backoff + jitter (:func:`_search_items`).
  * :class:`TractSearchCache` — one STAC search per *tract* instead of per
    crop. The training sampler draws many buildings per tract, so per-crop
    searches re-ask the same question hundreds of times. The cache keys on
    GEOID, searches a buffered bbox around the first crop, and serves every
    later crop in the tract locally; when a crop falls outside the searched
    bbox, the entry is re-searched over the union (self-correcting — no tract
    geometry needed, works for lazily materialized pair tables). Cached items
    are fetched UNSIGNED and their asset hrefs signed at read time, because a
    cached SAS token would expire mid-run. Failed searches are never cached
    (a poisoned entry would drop an entire tract — the label unit — from every
    shard). Per-key single-flight locking keeps a cold tract from firing one
    identical search per extract worker.

GSD note: NAIP is 0.6 m or 1 m depending on state/year, but crops are always
resampled so meters/pixel = crop_size_meters / out_pixels — the *effective*
resolution is constant by construction; the native GSD only affects sharpness.
"""

import random
import threading
import time
from collections import OrderedDict
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
    # Without a timeout one stalled TCP connection blocks its fetch thread
    # forever; upstream this froze CyclicCacheManager's background rotation
    # for an entire training run. Bound every phase of the transfer.
    GDAL_HTTP_CONNECTTIMEOUT="10",     # seconds to establish the connection
    GDAL_HTTP_TIMEOUT="60",            # seconds for the whole request
    GDAL_HTTP_LOW_SPEED_LIMIT="1024",  # abort if throughput < 1 KiB/s ...
    GDAL_HTTP_LOW_SPEED_TIME="30",     # ... for 30 consecutive seconds
)

# pystac-client defaults to NO request timeout: a dead connection hangs the
# STAC search (and the shard-generation thread running it) indefinitely.
STAC_TIMEOUT_S = 30

# STAC search retry policy. Planetary Computer rate-limits the search endpoint
# under sustained load; a failed search with no retry permanently drops the
# sample from its shard. Backoff is exponential with jitter so the 16 extract
# workers don't retry in lockstep.
SEARCH_RETRIES = 3
SEARCH_BACKOFF_S = 2.0

# Process-wide fetch statistics (thread-safe). Reset with reset_fetch_stats().
_STATS_LOCK = threading.Lock()
_FETCH_STATS = {"ok": 0, "no_items": 0, "asset_missing": 0, "read_error": 0,
                "search_error": 0, "nir_padded": 0, "partial_coverage": 0,
                "search_cache_hit": 0, "search_cache_miss": 0,
                "search_cache_expand": 0}


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
    failure: str | None = None       # None | no_items | asset_missing |
                                     # search_error | read_error


def get_catalog(signed: bool = True) -> pystac_client.Client:
    """One STAC client per thread — avoids connection pool contention.

    ``signed=False`` returns a client WITHOUT the sign_inplace modifier: items
    meant for the :class:`TractSearchCache` must be cached unsigned (SAS tokens
    expire in ~1h) and signed at read time instead.
    """
    attr = "catalog" if signed else "catalog_unsigned"
    if not hasattr(_local, attr):
        kwargs = {"modifier": planetary_computer.sign_inplace} if signed else {}
        try:
            client = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                timeout=STAC_TIMEOUT_S, **kwargs,
            )
        except TypeError:  # pystac-client < 0.7 has no timeout kwarg
            client = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1", **kwargs,
            )
        setattr(_local, attr, client)
    return getattr(_local, attr)


def _search_items(bbox: list[float], max_items: int | None = None,
                  signed: bool = True, retries: int | None = None,
                  backoff_s: float | None = None) -> list:
    """All NAIP items intersecting ``bbox``, with retry on search failure.

    ``max_items=None`` paginates the search fully — required for tract-level
    bboxes, where truncation would silently drop specific flight years and
    bias which panel years survive for that tract. Raises the last exception
    when every attempt fails (callers count ``search_error``).

    ``retries``/``backoff_s`` default to the module-level SEARCH_RETRIES /
    SEARCH_BACKOFF_S, resolved at call time so tests can monkeypatch them.
    """
    retries = SEARCH_RETRIES if retries is None else retries
    backoff_s = SEARCH_BACKOFF_S if backoff_s is None else backoff_s
    for attempt in range(retries):
        try:
            return list(get_catalog(signed=signed).search(
                collections=["naip"],
                bbox=bbox,
                max_items=max_items,
            ).items())
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(backoff_s * (2 ** attempt) * (0.5 + random.random()))


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


def item_intersects_bbox(item, bbox: list[float]) -> bool:
    """True if the STAC item's EPSG:4326 bbox intersects ``bbox``.

    Used to filter a tract-level cached item list down to the candidates a
    per-crop STAC search would have returned (STAC bbox search = intersection),
    so item selection is identical with and without the cache. Items without a
    usable bbox are kept: a per-crop search would have returned them too, and
    dropping them here would silently change selection.
    """
    ib = getattr(item, "bbox", None)
    if not ib or len(ib) < 4:
        return True
    return ib[0] <= bbox[2] and ib[2] >= bbox[0] and ib[1] <= bbox[3] and ib[3] >= bbox[1]


def buffer_bbox(bbox: list[float], meters: float) -> list[float]:
    """Expand an EPSG:4326 bbox outward by ``meters`` on every side (geodesic)."""
    w, s, e, n = bbox
    mid_lat = (s + n) / 2.0
    lon_e, _, _ = _GEOD.fwd(e, mid_lat, 90.0, meters)
    lon_w, _, _ = _GEOD.fwd(w, mid_lat, 270.0, meters)
    _, lat_n, _ = _GEOD.fwd((w + e) / 2.0, n, 0.0, meters)
    _, lat_s, _ = _GEOD.fwd((w + e) / 2.0, s, 180.0, meters)
    return [min(lon_w, w), min(lat_s, s), max(lon_e, e), max(lat_n, n)]


def bbox_union(a: list[float], b: list[float]) -> list[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def bbox_contains(outer: list[float], inner: list[float]) -> bool:
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3])


class TractSearchCache:
    """Per-tract (GEOID-keyed) cache of unsigned STAC item lists.

    One STAC search per tract instead of one per crop: the shard generator
    draws many buildings per tract, and NAIP DOQQs (~6x7.5 km) dwarf the
    ~100 m crops, so consecutive fetches keep re-asking Planetary Computer
    the same question — the direct cause of the search-endpoint throttling
    observed in training.

    Correctness invariant: the entry's ``searched_bbox`` always contains every
    crop bbox it has served. STAC bbox search returns items *intersecting* the
    query bbox, so for any crop bbox inside ``searched_bbox`` the cached list
    is a superset of that crop's own search result; filtering it down with
    :func:`item_intersects_bbox` (done by the caller) reproduces the per-crop
    search exactly. When a crop falls outside, the entry is re-searched over
    the union of the old bbox and the buffered crop bbox — self-correcting,
    so no precomputed tract geometry is needed.

    Thread-safe. Per-key single-flight: a cold tract triggers exactly one
    search even with all extract workers asking at once. Failed searches
    propagate and are never cached (a cached failure would delete an entire
    tract — the label unit — from every shard until eviction). LRU-bounded;
    shard slices are sequential over a spatially ordered pair table, so a few
    hundred entries give near-perfect hit rates.
    """

    def __init__(self, buffer_meters: float = 1500.0, max_entries: int = 512):
        self.buffer_meters = buffer_meters
        self.max_entries = max_entries
        self._lock = threading.Lock()                 # guards _entries/_key_locks
        self._entries: OrderedDict[str, tuple[list[float], list]] = OrderedDict()
        self._key_locks: dict[str, threading.Lock] = {}

    def _get_key_lock(self, key: str) -> threading.Lock:
        with self._lock:
            return self._key_locks.setdefault(key, threading.Lock())

    def _lookup(self, key: str, crop_bbox: list[float]):
        """Entry for ``key`` if it already covers ``crop_bbox`` (LRU-touched)."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and bbox_contains(entry[0], crop_bbox):
                self._entries.move_to_end(key)
                return entry
        return None

    def get_items(self, key: str, crop_bbox: list[float]) -> list:
        """Unsigned STAC items whose search bbox covers ``crop_bbox``.

        Raises on search failure (after retries) — the caller counts
        ``search_error``; nothing is cached in that case.
        """
        entry = self._lookup(key, crop_bbox)
        if entry is not None:
            _count("search_cache_hit")
            return entry[1]

        with self._get_key_lock(key):
            # Re-check: another thread may have searched while we waited.
            entry = self._lookup(key, crop_bbox)
            if entry is not None:
                _count("search_cache_hit")
                return entry[1]

            with self._lock:
                stale = self._entries.get(key)
            if stale is None:
                search_bbox = buffer_bbox(crop_bbox, self.buffer_meters)
                _count("search_cache_miss")
            else:
                # Entry exists but doesn't cover this crop: expand, re-search.
                search_bbox = bbox_union(
                    stale[0], buffer_bbox(crop_bbox, self.buffer_meters))
                _count("search_cache_expand")

            items = _search_items(search_bbox, max_items=None, signed=False)

            with self._lock:
                self._entries[key] = (search_bbox, items)
                self._entries.move_to_end(key)
                while len(self._entries) > self.max_entries:
                    evicted, _ = self._entries.popitem(last=False)
                    self._key_locks.pop(evicted, None)
            return items


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


def _select_item(items: list, bbox: list[float], year_hint: int | None):
    """Pick the item to read for ``bbox`` from candidate ``items`` (or None).

    Full-window coverage first: an intersecting-but-not-containing DOQQ yields
    a sliver crop (see module docstring). Adjacent DOQQs overlap, so a fully
    covering same-year tile almost always exists; when it doesn't, a fully
    covering off-year tile beats a same-year sliver.
    """
    if not items:
        return None
    if year_hint is not None:
        # Then closest flight year to the requested panel year (stable id tiebreak).
        return min(items, key=lambda item: (not item_contains_bbox(item, bbox),
                                            abs(item.datetime.year - year_hint), item.id))
    # Then deterministic default: newest first (STAC order is not guaranteed).
    return min(items, key=lambda item: (not item_contains_bbox(item, bbox),
                                        -item.datetime.year, item.id))


def fetch_naip(
    lon: float, lat: float,
    crop_size_meters: float,
    nbands: int = 4,
    out_pixels: int = 250,
    year_hint: int | None = None,
    search_cache: "TractSearchCache | None" = None,
    cache_key: str | None = None,
) -> NaipFetchResult:
    """Fetch a single NAIP crop from Planetary Computer.

    Returns a :class:`NaipFetchResult`; ``result.crop`` is (C, H, W) uint8 or
    None with ``result.failure`` set to the failure mode.

    ``search_cache`` + ``cache_key`` (typically the tract GEOID) route the STAC
    search through a :class:`TractSearchCache`: one search serves every crop in
    the tract, and item choice is provably identical to a per-crop search (the
    cached list, filtered to items intersecting this crop's bbox, is exactly
    what the per-crop search would return). Cached items are unsigned, so the
    asset href is signed at read time. Either argument None ⇒ direct search.
    """
    bbox = lonlat_bbox(lon, lat, crop_size_meters)
    use_cache = search_cache is not None and cache_key is not None

    try:
        if use_cache:
            items = [it for it in search_cache.get_items(cache_key, bbox)
                     if item_intersects_bbox(it, bbox)]
        else:
            items = _search_items(bbox, max_items=50, signed=True)
    except Exception:
        _count("search_error")
        return NaipFetchResult(None, None, failure="search_error")

    if not items:
        _count("no_items")
        return NaipFetchResult(None, None, failure="no_items")

    item = _select_item(items, bbox, year_hint)
    actual_year = item.datetime.year

    asset = item.assets.get("image") or item.assets.get("visual")
    if not asset:
        _count("asset_missing")
        return NaipFetchResult(None, actual_year, failure="asset_missing")

    try:
        # Cached items carry unsigned hrefs (cached SAS tokens would expire);
        # sign here, at read time. The token itself is cached client-side by
        # planetary_computer, so this is not an extra API round-trip.
        href = planetary_computer.sign(asset.href) if use_cache else asset.href
        with rasterio.Env(**GDAL_ENV):
            with rasterio.open(href) as src:
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
