"""Local high-resolution ortho fetcher (ArcGIS Image/Map services) — issue #36.

Companion to :mod:`src.data.naip_fetcher`. NAIP is the *training* sensor; this
module fetches each CSA city's own municipal orthoimagery, which is a sensor the
model never saw during fine-tuning. Predicting the Callaway–Sant'Anna event study
off it makes the validation **spatially held out x out of sensor** — the
"double-holdout" issue #36 asks for — and, for Chicago, buys an *annual* panel
(2009-2025) where Illinois NAIP is only 2011/12/14/15/17/19/21/23.

Why this is a separate module and not a branch inside ``naip_fetcher``
---------------------------------------------------------------------
NAIP access is a three-step dance: STAC search -> item selection -> windowed COG
read, with SAS signing and a per-tract search cache to amortise the search. An
ArcGIS image service has none of that: one HTTP GET returns exactly the window
you asked for, already mosaicked and resampled server-side. Sharing code would
mean threading "does this sensor have a catalogue?" through every function; the
two fetchers instead share a *contract* (see below) and nothing else.

The contract
------------
:func:`fetch_ortho` is keyword-compatible with :func:`src.data.naip_fetcher.fetch_naip`
and returns an :class:`OrthoFetchResult` carrying the same ``crop`` /
``actual_year`` / ``failure`` attributes as ``NaipFetchResult``. That is what lets
:func:`src.prediction.fetch_prediction_chunk` swap sensors through its existing
``fetch_fn`` injection point, reusing the whole retry/cooldown/halt orchestration,
the resumable chunk parquets and the GPU path unchanged. ``search_cache`` and
``cache_key`` are accepted and ignored — there is no catalogue to cache.

Failure modes reuse ``naip_fetcher``'s taxonomy so callers' retry logic is
identical: ``no_items`` / ``no_year_match`` are permanent, ``read_error`` is
retryable. Throttling (HTTP 429/503) is reported as ``read_error`` — retryable —
but counted separately in the stats so it stays visible.

Geometry
--------
Crops are built in the *service's own projected CRS* rather than in EPSG:5070.
5070 (CONUS Albers) is equal-area but not conformal: its linear scale distorts
the window's shape away from the standard parallels, which would shear the crop.
Each source declares a conformal local CRS (e.g. EPSG:3435, Illinois East ftUS)
in which a square really is square, plus ``units_per_meter`` for the offset.
``test_ortho_fetcher`` validates every registered source's ``units_per_meter``
against pyproj, so a wrong constant fails in CI rather than silently producing
crops of the wrong ground size.

The request asks for ``out_pixels`` square over a ``crop_size_meters`` window, so
the effective GSD is ``crop_size_meters / out_pixels`` — identical to the NAIP
path by construction. At the production tau=100 m / 224 px that is 0.89 m/px,
matching what the model was fine-tuned on, even though Cook County's native
imagery is 6-inch. The server does the downsampling. ScaleMAE's scale metadata
therefore needs no adjustment.

Politeness
----------
Planetary Computer is hyperscale infrastructure that absorbed 64 concurrent
workers with zero throttling. These are *municipal GIS servers*. Every request
goes through a shared token bucket (:class:`RateLimiter`, default 10 req/s),
``Retry-After`` is honoured, and repeated throttling escalates the cooldown.
Do not raise the default rate limit to make a run finish sooner.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import requests
from pyproj import Transformer
from rasterio.io import MemoryFile

from src.data.naip_fetcher import (  # noqa: F401  (re-exported for callers)
    PERMANENT_FAILURES,
    RETRYABLE_FAILURES,
    backoff_sleep,
)

# HTTP status codes that mean "you are asking too fast", as opposed to a real
# error. Treated as retryable and counted separately so throttling never hides
# inside a generic read_error count.
THROTTLE_STATUS = frozenset({429, 502, 503, 504})

# Per-request timeouts (connect, read). An ArcGIS exportImage of a 224x224
# window is small and fast; a slow response means the server is struggling, and
# waiting longer only deepens the queue we are contributing to.
HTTP_TIMEOUT_S = (10.0, 60.0)

# Retries *inside* one fetch_ortho call. The caller (fetch_prediction_chunk)
# runs its own multi-pass cooldown ladder on top, so this stays small: it exists
# to ride out a single blip, not to grind through an outage.
HTTP_RETRIES = 3
HTTP_BACKOFF_S = 1.5

_local = threading.local()


# --------------------------------------------------------------------------- #
# Rate limiting                                                                #
# --------------------------------------------------------------------------- #

class RateLimiter:
    """Thread-safe token bucket, shared by every worker hitting one host.

    ``rate`` tokens accrue per second up to ``burst``. :meth:`acquire` blocks
    until a token is available. ``rate <= 0`` disables limiting entirely (used
    only by the throughput probe, never in production runs).

    A bucket rather than a fixed sleep because fetch latency is bursty: a token
    bucket lets a run of fast responses proceed at full speed while still
    bounding the *average* rate the server sees, which is the number that
    matters to whoever operates it.
    """

    def __init__(self, rate: float = 10.0, burst: float | None = None,
                 time_fn=time.monotonic, sleep_fn=time.sleep):
        self.rate = float(rate)
        self.burst = float(burst) if burst is not None else max(1.0, float(rate))
        self._tokens = self.burst
        self._last = time_fn()
        self._lock = threading.Lock()
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn

    def acquire(self, n: float = 1.0) -> float:
        """Consume ``n`` tokens, blocking as needed. Returns seconds waited.

        Uses *virtual scheduling*: the shortfall is debited immediately (letting
        ``_tokens`` go negative, to be repaid as time passes) and the caller
        sleeps exactly long enough to mint it. There is deliberately no
        re-check loop — a loop that recomputes the balance after sleeping can
        spin forever once the residual deficit drops below the clock's ULP,
        because ``t + tiny == t`` in floating point, so elapsed time reads as
        zero and the balance never grows. Debiting under the lock also means a
        concurrent caller sees the debt instead of racing for the same token.
        """
        if self.rate <= 0:
            return 0.0
        with self._lock:
            now = self._time_fn()
            self._tokens = min(self.burst,
                               self._tokens + (now - self._last) * self.rate)
            self._last = now
            if self._tokens >= n:
                self._tokens -= n
                return 0.0
            deficit = (n - self._tokens) / self.rate
            self._tokens -= n
        self._sleep_fn(deficit)
        return deficit

    def penalise(self, seconds: float) -> None:
        """Drain the bucket for ``seconds`` after a throttle response.

        Called on 429/503 so *every* worker slows down, not just the one that
        got throttled — otherwise the remaining threads keep hammering while one
        backs off, and the server never gets relief.
        """
        if self.rate <= 0:
            return
        with self._lock:
            self._tokens = min(self._tokens, 0.0) - seconds * self.rate


# Process-wide default. Deliberately conservative; see the module docstring.
DEFAULT_MAX_RPS = 10.0
_LIMITERS: dict[str, RateLimiter] = {}
_LIMITER_LOCK = threading.Lock()


def get_limiter(host: str, max_rps: float | None = None) -> RateLimiter:
    """One shared limiter per host, created on first use."""
    with _LIMITER_LOCK:
        lim = _LIMITERS.get(host)
        if lim is None:
            lim = RateLimiter(DEFAULT_MAX_RPS if max_rps is None else max_rps)
            _LIMITERS[host] = lim
        elif max_rps is not None and max_rps != lim.rate:
            lim.rate = float(max_rps)
            lim.burst = max(1.0, float(max_rps))
    return lim


def set_max_rps(max_rps: float) -> None:
    """Set the rate cap for every host, current and future."""
    global DEFAULT_MAX_RPS
    DEFAULT_MAX_RPS = float(max_rps)
    with _LIMITER_LOCK:
        for lim in _LIMITERS.values():
            lim.rate = float(max_rps)
            lim.burst = max(1.0, float(max_rps))


# --------------------------------------------------------------------------- #
# Stats                                                                        #
# --------------------------------------------------------------------------- #

_STATS_LOCK = threading.Lock()
_FETCH_STATS = {"ok": 0, "no_items": 0, "no_year_match": 0, "read_error": 0,
                "throttled": 0, "nir_padded": 0, "partial_coverage": 0,
                "rate_limit_wait_s": 0.0}


def get_fetch_stats() -> dict:
    """Snapshot of cumulative ortho fetch outcomes for this process."""
    with _STATS_LOCK:
        return dict(_FETCH_STATS)


def reset_fetch_stats() -> None:
    with _STATS_LOCK:
        for k in _FETCH_STATS:
            _FETCH_STATS[k] = type(_FETCH_STATS[k])(0)


def _count(key: str, amount: float = 1) -> None:
    with _STATS_LOCK:
        _FETCH_STATS[key] += amount


# --------------------------------------------------------------------------- #
# Source registry                                                              #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OrthoSource:
    """One city-year of local orthoimagery served by an ArcGIS endpoint.

    Attributes
    ----------
    city, year
        Registry keys; ``city`` matches :data:`src.csa_event_study.CSA_CITIES`.
    url
        Service root, e.g. ``.../CookOrtho2023/ImageServer``. The operation
        (``exportImage`` vs ``export``) is derived from ``kind``.
    kind
        ``"image"`` for an ImageServer (true raster, 4-band, supports
        ``pixelType``), ``"map"`` for a MapServer tile cache (rendered RGB —
        NIR must be zero-padded, see ``bands``).
    native_wkid
        A *conformal* projected CRS for this service's area, in which the crop
        window is built. Not EPSG:5070: Albers is equal-area, so a square in
        5070 is not a square on the ground, and the crop would be sheared.
    units_per_meter
        Linear units of ``native_wkid`` per metre (1.0 metric, 3.2808333... for
        US survey feet). Validated against pyproj in the unit tests.
    bands
        Bands the service actually returns. 3 triggers NIR zero-padding, which
        is reported via ``nir_padded`` exactly as the NAIP ``visual`` asset is.
    layer
        MapServer layer id, if the export needs one pinned.
    """

    city: str
    year: int
    url: str
    kind: Literal["image", "map"] = "image"
    native_wkid: int = 3435
    units_per_meter: float = 3937.0 / 1200.0   # US survey feet
    bands: int = 4
    layer: str | None = None

    @property
    def host(self) -> str:
        return self.url.split("//", 1)[-1].split("/", 1)[0]

    @property
    def export_url(self) -> str:
        op = "exportImage" if self.kind == "image" else "export"
        return f"{self.url.rstrip('/')}/{op}"


# Cook County flies the whole county every year and publishes each year as its
# own ImageServer: 4-band, 6-inch (0.5 ft pixel), U8, no authentication.
# Verified live against the services directory; 1998/2003 exist but are
# panchromatic and excluded (a 1-band crop is not comparable to the 4-band
# panel the model consumes).
_COOK_ORTHO_YEARS = tuple(range(2009, 2026))
_COOK_URL = ("https://gis.cookcountyil.gov/imagery/rest/services/"
             "CookOrtho{year}/ImageServer")

# EPSG:3435 = NAD83 / Illinois East (ftUS) — the service's own reported CRS
# (wkid 102671, latestWkid 3435) and conformal over Cook County.
_CHICAGO = {
    year: OrthoSource(city="chicago", year=year, url=_COOK_URL.format(year=year),
                      kind="image", native_wkid=3435,
                      units_per_meter=3937.0 / 1200.0, bands=4)
    for year in _COOK_ORTHO_YEARS
}

# King County publishes discrete flights as cached MapServers (rendered RGB, so
# bands=3 and NIR is padded). EPSG:2926 = NAD83(HARN) / Washington North (ftUS).
# Sparser than Cook: no annual panel, so Seattle's CSA cadence is irregular.
_KINGCO_URL = ("https://gismaps.kingcounty.gov/arcgis/rest/services/"
               "BaseMaps/KingCo_Aerial_{year}/MapServer")
_SEATTLE = {
    year: OrthoSource(city="seattle", year=year, url=_KINGCO_URL.format(year=year),
                      kind="map", native_wkid=2926,
                      units_per_meter=3937.0 / 1200.0, bands=3)
    for year in (2007, 2017, 2021)
}

# Tampa (FL) and Nashville (TN) have no year-indexed local ortho service:
# TNMap's BASEMAPS/IMAGERY is a *current* mosaic refreshed county-by-county as
# TDOT flies one region per year, not a time series, so there is nothing to
# build a panel from. Both fall back to NAIP (spec.sensor == "naip"). Left
# empty rather than absent so the registry documents the gap.
ORTHO_SOURCES: dict[str, dict[int, OrthoSource]] = {
    "chicago": _CHICAGO,
    "seattle": _SEATTLE,
    "tampa": {},
    "nashville": {},
}


def available_years(city: str) -> tuple[int, ...]:
    """Years with a local ortho service for ``city`` (empty if none)."""
    return tuple(sorted(ORTHO_SOURCES.get(city, {})))


def resolve_source(city: str, year: int, *, exact_year: bool = True
                   ) -> tuple[OrthoSource | None, str | None]:
    """Pick the service for (city, year), or explain why there isn't one.

    With ``exact_year`` (the default, matching ``predict_exact_year``) only the
    requested year is acceptable — substituting a neighbouring flight would put
    imagery from the wrong side of a construction event into the panel and
    manufacture the effect we are trying to measure.
    """
    sources = ORTHO_SOURCES.get(city)
    if not sources:
        return None, "no_items"
    src = sources.get(int(year))
    if src is not None:
        return src, None
    if exact_year:
        return None, "no_year_match"
    nearest = min(sources, key=lambda y: (abs(y - int(year)), y))
    return sources[nearest], None


# --------------------------------------------------------------------------- #
# Geometry                                                                     #
# --------------------------------------------------------------------------- #

def _transformer(wkid: int) -> Transformer:
    """Thread-local EPSG:4326 -> native transformer (pyproj is not thread-safe
    to share, and rebuilding one per crop measurably dominated the NAIP path)."""
    cache = getattr(_local, "transformers", None)
    if cache is None:
        cache = _local.transformers = {}
    tr = cache.get(wkid)
    if tr is None:
        tr = cache[wkid] = Transformer.from_crs("EPSG:4326", f"EPSG:{wkid}",
                                                always_xy=True)
    return tr


def native_bbox(lon: float, lat: float, crop_size_meters: float,
                source: OrthoSource) -> tuple[float, float, float, float]:
    """Square window of ``crop_size_meters`` centred on (lon, lat), in native units.

    Built by projecting the centre and offsetting by half the window in the
    service's conformal CRS, so the result is a true ground square. Returned as
    ``(xmin, ymin, xmax, ymax)``.

    Raises on non-finite output. PROJ returns ``inf`` rather than raising when it
    cannot build a pipeline — notably ``EPSG:4326 -> EPSG:3435``, which wants a
    NAD83 shift grid from cdn.proj.org, under ``PROJ_NETWORK=ON`` with no route
    to that CDN. Unchecked, the bbox would be formatted as the literal string
    "inf", every request would come back an error, and the run would look like a
    server outage rather than a misconfigured environment. Same guard, same
    reason, as ``csa_event_study._check_reprojection``.
    """
    x, y = _transformer(source.native_wkid).transform(lon, lat)
    if not (np.isfinite(x) and np.isfinite(y)):
        raise RuntimeError(
            f"projecting ({lon}, {lat}) to EPSG:{source.native_wkid} produced "
            f"non-finite coordinates. PROJ could not build the transformation "
            f"pipeline — if PROJ_NETWORK=ON and this machine has no access to "
            f"the PROJ grid CDN (cdn.proj.org), set PROJ_NETWORK=OFF and retry."
        )
    half = (crop_size_meters / 2.0) * source.units_per_meter
    return (x - half, y - half, x + half, y + half)


# --------------------------------------------------------------------------- #
# Fetch                                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class OrthoFetchResult:
    """Outcome of one fetch — mirrors :class:`~src.data.naip_fetcher.NaipFetchResult`.

    Field-for-field compatible on the three attributes ``prediction.py`` reads
    (``crop``, ``actual_year``, ``failure``), so the two fetchers are
    interchangeable through ``fetch_prediction_chunk(fetch_fn=...)``.
    """

    crop: np.ndarray | None          # (C, H, W) uint8, or None on failure
    actual_year: int | None          # flight year actually used
    nir_padded: bool = False         # NIR synthesised (3-band MapServer source)
    partial_coverage: bool = False   # window fell partly outside the service
    failure: str | None = None       # None | no_items | no_year_match | read_error
    status: int | None = field(default=None, repr=False)   # last HTTP status


def _session() -> requests.Session:
    """Thread-local session so connections are pooled per worker."""
    s = getattr(_local, "session", None)
    if s is None:
        s = _local.session = requests.Session()
        s.headers["User-Agent"] = (
            "ny-state-aerial-imagery-prototype/1.0 (academic research; "
            "building-level wealth estimation)"
        )
    return s


def _export_params(source: OrthoSource, bbox, out_pixels: int) -> dict:
    """Query string for one exportImage/export call returning raw bytes."""
    p = {
        "bbox": ",".join(f"{v:.4f}" for v in bbox),
        "bboxSR": str(source.native_wkid),
        "imageSR": str(source.native_wkid),
        "size": f"{out_pixels},{out_pixels}",
        "format": "tiff",
        "f": "image",
    }
    if source.kind == "image":
        # ImageServer only. Bilinear matches the NAIP path's resampling, so the
        # two sensors are downsampled the same way and a cross-sensor
        # comparison is not confounded by interpolation choice.
        p["pixelType"] = "U8"
        p["interpolation"] = "RSP_BilinearInterpolation"
        p["noData"] = "0"
    else:
        p["transparent"] = "false"
        if source.layer is not None:
            p["layers"] = f"show:{source.layer}"
    return p


def _decode(content: bytes, nbands: int, out_pixels: int
            ) -> tuple[np.ndarray | None, bool, bool, str | None]:
    """Decode TIFF bytes to (C, H, W) uint8, padding/truncating to ``nbands``.

    Returns ``(crop, nir_padded, partial_coverage, failure)``. ``partial_coverage``
    is inferred from an all-zero band-1, which is how ArcGIS reports a window
    outside the service footprint (it returns a valid, black image rather than
    an error).
    """
    try:
        with MemoryFile(content) as mem, mem.open() as ds:
            arr = ds.read(out_shape=(ds.count, out_pixels, out_pixels))
    except Exception:
        return None, False, False, "read_error"

    if arr.dtype != np.uint8:
        # Services occasionally hand back 16-bit despite pixelType=U8.
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    nir_padded = False
    if arr.shape[0] >= nbands:
        crop = arr[:nbands]
    else:
        pad = np.zeros((nbands - arr.shape[0], out_pixels, out_pixels), np.uint8)
        crop = np.concatenate([arr, pad], axis=0)
        nir_padded = True

    # A fully black crop means the window is outside coverage. Distinguished
    # from a failure because the geometry is still correct; the caller decides
    # whether to keep it (the NAIP path makes the same distinction).
    partial = not bool(crop[0].any())
    return crop, nir_padded, partial, None


def fetch_ortho(lon: float, lat: float, crop_size_meters: float,
                nbands: int = 4, out_pixels: int = 224,
                year_hint: int | None = None,
                search_cache=None, cache_key: str | None = None,
                exact_year: bool = True,
                *, city: str | None = None, max_rps: float | None = None,
                session=None, sleep_fn=time.sleep) -> OrthoFetchResult:
    """Fetch one crop from a city's local ortho service.

    Keyword-compatible with :func:`src.data.naip_fetcher.fetch_naip` so it can be
    injected straight into :func:`src.prediction.fetch_prediction_chunk` as
    ``fetch_fn``. ``search_cache`` and ``cache_key`` are accepted and ignored:
    an image service needs no catalogue lookup. ``city`` must be bound by the
    caller (typically ``functools.partial(fetch_ortho, city="chicago")``).

    Retries throttles and transient HTTP errors ``HTTP_RETRIES`` times with
    jittered backoff, penalising the shared rate limiter so *all* workers slow
    down, then reports ``read_error`` for the caller's coarser cooldown ladder.
    """
    if city is None:
        raise ValueError("fetch_ortho requires a city (bind it with functools.partial)")
    if year_hint is None:
        return OrthoFetchResult(None, None, failure="no_year_match")

    source, failure = resolve_source(city, year_hint, exact_year=exact_year)
    if source is None:
        _count(failure)
        return OrthoFetchResult(None, None, failure=failure)

    bbox = native_bbox(lon, lat, crop_size_meters, source)
    params = _export_params(source, bbox, out_pixels)
    sess = session if session is not None else _session()
    limiter = get_limiter(source.host, max_rps)
    last_status = None

    for attempt in range(HTTP_RETRIES):
        _count("rate_limit_wait_s", limiter.acquire())
        try:
            resp = sess.get(source.export_url, params=params, timeout=HTTP_TIMEOUT_S)
            last_status = resp.status_code
            if resp.status_code in THROTTLE_STATUS:
                _count("throttled")
                # Retry-After may be absent; fall back to the exponential ladder.
                try:
                    wait = float(resp.headers.get("Retry-After", ""))
                except ValueError:
                    wait = HTTP_BACKOFF_S * (2 ** attempt)
                limiter.penalise(wait)
                if attempt < HTTP_RETRIES - 1:
                    sleep_fn(wait)
                    continue
                _count("read_error")
                return OrthoFetchResult(None, None, failure="read_error",
                                        status=last_status)
            resp.raise_for_status()
            content = resp.content
        except Exception:
            if attempt < HTTP_RETRIES - 1:
                sleep_fn(HTTP_BACKOFF_S * (2 ** attempt))
                continue
            _count("read_error")
            return OrthoFetchResult(None, None, failure="read_error",
                                    status=last_status)

        crop, nir_padded, partial, dec_failure = _decode(content, nbands, out_pixels)
        if dec_failure is not None:
            if attempt < HTTP_RETRIES - 1:
                sleep_fn(HTTP_BACKOFF_S * (2 ** attempt))
                continue
            _count("read_error")
            return OrthoFetchResult(None, None, failure=dec_failure, status=last_status)

        _count("ok")
        if nir_padded:
            _count("nir_padded")
        if partial:
            _count("partial_coverage")
        return OrthoFetchResult(crop, source.year, nir_padded=nir_padded,
                                partial_coverage=partial, status=last_status)

    _count("read_error")
    return OrthoFetchResult(None, None, failure="read_error", status=last_status)


def make_fetch_fn(city: str, *, max_rps: float | None = None):
    """``fetch_naip``-compatible callable bound to one city's ortho service.

    Hand the result to ``fetch_prediction_chunk(fetch_fn=...)`` to run the whole
    prediction engine off local ortho instead of NAIP.
    """
    def _fetch(lon, lat, crop_size_meters, nbands=4, out_pixels=224,
               year_hint=None, search_cache=None, cache_key=None,
               exact_year=True):
        return fetch_ortho(lon, lat, crop_size_meters, nbands=nbands,
                           out_pixels=out_pixels, year_hint=year_hint,
                           search_cache=search_cache, cache_key=cache_key,
                           exact_year=exact_year, city=city, max_rps=max_rps)
    _fetch.__name__ = f"fetch_ortho_{city}"
    return _fetch


# --------------------------------------------------------------------------- #
# Band-order guard                                                             #
# --------------------------------------------------------------------------- #

# Known heavily-vegetated points, used to verify band order. A park in high
# summer is unambiguously vegetated, so NDVI from the true NIR band must be
# strongly positive; if the service returned BGRN (or any other permutation)
# the computed NDVI collapses or goes negative.
NDVI_PROBE_POINTS = {
    "chicago": (-87.6367, 41.7920),   # Washington Park, Chicago
    "seattle": (-122.3035, 47.5550),  # Seward Park, Seattle
}


def check_band_order(city: str, year: int, *, crop_size_meters: float = 200.0,
                     out_pixels: int = 224, min_ndvi: float = 0.10,
                     fetch_fn=None) -> tuple[bool, float]:
    """Assert band 4 really is NIR, by measuring NDVI over a known park.

    Returns ``(ok, median_ndvi)``. A silent RGBN-vs-BGRN permutation would not
    raise anywhere — it would just quietly feed the model wrong channels and
    corrupt every prediction — so this runs as a startup check before a city's
    prediction pass, not only in the test suite.

    ``fetch_fn`` is injectable so tests can exercise it without network.
    """
    if city not in NDVI_PROBE_POINTS:
        raise KeyError(f"no NDVI probe point registered for {city!r}")
    lon, lat = NDVI_PROBE_POINTS[city]
    fetch = fetch_fn if fetch_fn is not None else fetch_ortho
    res = fetch(lon, lat, crop_size_meters, nbands=4, out_pixels=out_pixels,
                year_hint=year, city=city)
    if res.crop is None:
        raise RuntimeError(f"band-order check could not fetch {city} {year}: "
                           f"{res.failure}")
    if res.nir_padded:
        # 3-band source: NIR is synthetic by construction, nothing to verify.
        return True, float("nan")
    red = res.crop[0].astype(np.float32)
    nir = res.crop[3].astype(np.float32)
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / np.maximum(denom, 1e-6), 0.0)
    median = float(np.median(ndvi))
    return median >= min_ndvi, median
