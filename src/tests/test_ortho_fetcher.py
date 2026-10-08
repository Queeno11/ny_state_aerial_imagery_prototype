"""Synthetic-data tests for src/data/ortho_fetcher.py (local high-res ortho).

No network: every HTTP call goes through a fake session that returns TIFF bytes
built in-memory by rasterio. The geometry tests DO use pyproj, because the whole
point of them is to catch a wrong ``units_per_meter`` constant in the registry —
validating that against a hand-copy of the same constant would prove nothing.

What these pin down:

1. the crop window is a true 200 m ground square in the service's native CRS
   (a wrong units_per_meter silently produces crops of the wrong ground size,
   which no downstream code can detect),
2. band 4 is NIR and the NDVI guard actually fires on a permuted response,
3. throttling (429/503) is retried, honours Retry-After, and slows *every*
   worker via the shared limiter,
4. the result is drop-in compatible with the NAIP fetcher's contract, since
   ``prediction.fetch_prediction_chunk`` swaps them through one injection point.
"""

import threading
import time

import numpy as np
import pyproj
import pytest
import rasterio
from pyproj import Geod, Transformer
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from src.data import ortho_fetcher as of

GEOD = Geod(ellps="WGS84")
CHICAGO_LONLAT = (-87.6298, 41.8781)
SEATTLE_LONLAT = (-122.3321, 47.6062)


# ─── fakes ────────────────────────────────────────────────────────────────────

def _tiff_bytes(bands: int, size: int = 224, fill=None, dtype="uint8") -> bytes:
    """A valid single-tile GeoTIFF with ``bands`` bands."""
    arr = np.zeros((bands, size, size), dtype=dtype)
    for b in range(bands):
        arr[b] = (b + 1) * 10 if fill is None else fill[b]
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=size, height=size, count=bands,
                      dtype=dtype, crs="EPSG:3435",
                      transform=from_bounds(0, 0, size, size, size, size)) as ds:
            ds.write(arr)
        return mem.read()


class FakeResponse:
    def __init__(self, content=b"", status_code=200, headers=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Records every request; replays a scripted list of responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        resp = self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]
        if isinstance(resp, Exception):
            raise resp
        return resp


@pytest.fixture(autouse=True)
def _reset_stats():
    of.reset_fetch_stats()
    yield
    of.reset_fetch_stats()


@pytest.fixture(autouse=True)
def offline_proj():
    """EPSG:4326 -> EPSG:3435 needs a NAD83 shift grid from cdn.proj.org; with
    PROJ_NETWORK=ON and no route to it, PROJ returns inf instead of raising.
    Force the ballpark transform so these tests never depend on the network.
    Ballpark accuracy is ~1 m, irrelevant against a 200 m crop window.
    """
    was_enabled = pyproj.network.is_network_enabled()
    pyproj.network.set_network_enabled(False)
    _local_caches = getattr(of._local, "transformers", None)
    if _local_caches is not None:
        _local_caches.clear()
    yield
    pyproj.network.set_network_enabled(was_enabled)
    if _local_caches is not None:
        _local_caches.clear()


# ─── registry ─────────────────────────────────────────────────────────────────

def test_chicago_registry_is_annual_2009_2025():
    """Cook County's annual cadence is the reason Chicago uses ortho over NAIP
    (Illinois NAIP has only 8 years); a gap here silently shortens the panel."""
    years = of.available_years("chicago")
    assert years == tuple(range(2009, 2026))
    assert all(of.ORTHO_SOURCES["chicago"][y].bands == 4 for y in years)
    assert all(of.ORTHO_SOURCES["chicago"][y].kind == "image" for y in years)


def test_cities_without_a_year_indexed_service_are_declared_empty():
    """TN/FL have no historical ortho time series — they must fall back to NAIP,
    and the registry documents that rather than omitting the key."""
    assert of.available_years("tampa") == ()
    assert of.available_years("nashville") == ()
    assert "tampa" in of.ORTHO_SOURCES and "nashville" in of.ORTHO_SOURCES


def test_export_url_matches_service_kind():
    assert of.ORTHO_SOURCES["chicago"][2023].export_url.endswith(
        "/CookOrtho2023/ImageServer/exportImage")
    assert of.ORTHO_SOURCES["seattle"][2021].export_url.endswith(
        "/KingCo_Aerial_2021/MapServer/export")


@pytest.mark.parametrize("city", ["chicago", "seattle"])
def test_units_per_meter_matches_pyproj(city):
    """REGRESSION GUARD: a wrong units_per_meter yields crops of the wrong ground
    size with no error anywhere downstream — the model just sees the wrong zoom."""
    for source in of.ORTHO_SOURCES[city].values():
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{source.native_wkid}",
                                  always_xy=True)
        lon, lat = CHICAGO_LONLAT if city == "chicago" else SEATTLE_LONLAT
        x0, y0 = tr.transform(lon, lat)
        # Move exactly 1000 m east on the ellipsoid, measure it in native units.
        lon2, lat2, _ = GEOD.fwd(lon, lat, 90.0, 1000.0)
        x1, y1 = tr.transform(lon2, lat2)
        measured = np.hypot(x1 - x0, y1 - y0) / 1000.0
        assert measured == pytest.approx(source.units_per_meter, rel=2e-3)


# ─── geometry ─────────────────────────────────────────────────────────────────

def test_native_bbox_is_a_true_ground_square():
    source = of.ORTHO_SOURCES["chicago"][2023]
    lon, lat = CHICAGO_LONLAT
    xmin, ymin, xmax, ymax = of.native_bbox(lon, lat, 200.0, source)

    width_native, height_native = xmax - xmin, ymax - ymin
    assert width_native == pytest.approx(height_native)
    # 200 m in US survey feet.
    assert width_native == pytest.approx(200.0 * 3937.0 / 1200.0, rel=1e-9)

    # And it is 200 m on the ground, measured geodesically after inverting.
    inv = Transformer.from_crs(f"EPSG:{source.native_wkid}", "EPSG:4326",
                               always_xy=True)
    lon_w, lat_s = inv.transform(xmin, ymin)
    lon_e, _ = inv.transform(xmax, ymin)
    _, _, ground_m = GEOD.inv(lon_w, lat_s, lon_e, lat_s)
    assert ground_m == pytest.approx(200.0, rel=1e-3)


def test_non_finite_projection_raises_instead_of_sending_inf():
    """PROJ answers an unbuildable pipeline with inf rather than an exception.
    Unchecked, the bbox formats as the literal "inf", every request errors, and
    the run looks like a server outage instead of a bad environment."""
    source = of.ORTHO_SOURCES["chicago"][2023]

    class InfTransformer:
        @staticmethod
        def transform(lon, lat):
            return float("inf"), float("inf")

    of._local.transformers = {source.native_wkid: InfTransformer}
    try:
        with pytest.raises(RuntimeError, match="PROJ_NETWORK"):
            of.native_bbox(-87.6, 41.9, 200.0, source)
    finally:
        of._local.transformers = {}


def test_bbox_is_centred_on_the_request_point():
    source = of.ORTHO_SOURCES["chicago"][2023]
    lon, lat = CHICAGO_LONLAT
    xmin, ymin, xmax, ymax = of.native_bbox(lon, lat, 200.0, source)
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{source.native_wkid}",
                              always_xy=True)
    cx, cy = tr.transform(lon, lat)
    assert (xmin + xmax) / 2 == pytest.approx(cx)
    assert (ymin + ymax) / 2 == pytest.approx(cy)


def test_effective_gsd_matches_the_naip_training_footprint():
    """tau=100 m -> 200 m window over 224 px = 0.89 m/px, the GSD the model was
    fine-tuned at. The server does the downsampling from 6-inch native."""
    sess = FakeSession([FakeResponse(_tiff_bytes(4))])
    res = of.fetch_ortho(*CHICAGO_LONLAT, 200.0, year_hint=2023, city="chicago",
                         session=sess, max_rps=0)
    assert res.crop.shape == (4, 224, 224)
    params = sess.calls[0]["params"]
    assert params["size"] == "224,224"
    assert params["bboxSR"] == "3435" and params["imageSR"] == "3435"


# ─── year resolution ──────────────────────────────────────────────────────────

def test_exact_year_refuses_substitution():
    """Substituting a neighbouring flight would put imagery from the wrong side
    of a construction event into the panel and manufacture the ATT."""
    source, failure = of.resolve_source("chicago", 2005, exact_year=True)
    assert source is None and failure == "no_year_match"


def test_inexact_year_falls_back_to_nearest():
    source, failure = of.resolve_source("chicago", 2005, exact_year=False)
    assert failure is None and source.year == 2009


def test_unregistered_city_is_a_permanent_failure():
    source, failure = of.resolve_source("tampa", 2019)
    assert source is None and failure == "no_items"
    assert failure in of.PERMANENT_FAILURES


def test_fetch_reports_actual_year_of_the_service_used():
    sess = FakeSession([FakeResponse(_tiff_bytes(4))])
    res = of.fetch_ortho(*CHICAGO_LONLAT, 200.0, year_hint=2017, city="chicago",
                         session=sess, max_rps=0)
    assert res.actual_year == 2017 and res.failure is None


# ─── decoding ─────────────────────────────────────────────────────────────────

def test_three_band_source_pads_nir_and_flags_it():
    """King County's MapServer renders RGB; a silently-3-band crop would feed the
    model a zero channel with no record of it."""
    sess = FakeSession([FakeResponse(_tiff_bytes(3))])
    res = of.fetch_ortho(*SEATTLE_LONLAT, 200.0, year_hint=2021, city="seattle",
                         session=sess, max_rps=0)
    assert res.crop.shape == (4, 224, 224)
    assert res.nir_padded is True
    assert not res.crop[3].any()
    assert of.get_fetch_stats()["nir_padded"] == 1


def test_all_black_response_is_flagged_partial_not_failed():
    """ArcGIS answers an out-of-coverage window with a valid black image, not an
    error; the geometry is still right, so it is partial_coverage, not a failure."""
    sess = FakeSession([FakeResponse(_tiff_bytes(4, fill=[0, 0, 0, 0]))])
    res = of.fetch_ortho(*CHICAGO_LONLAT, 200.0, year_hint=2023, city="chicago",
                         session=sess, max_rps=0)
    assert res.failure is None and res.partial_coverage is True


def test_undecodable_bytes_are_a_retryable_read_error():
    sess = FakeSession([FakeResponse(b"<html>error</html>")])
    res = of.fetch_ortho(*CHICAGO_LONLAT, 200.0, year_hint=2023, city="chicago",
                         session=sess, max_rps=0, sleep_fn=lambda s: None)
    assert res.crop is None and res.failure == "read_error"
    assert res.failure in of.RETRYABLE_FAILURES


# ─── throttling / politeness ──────────────────────────────────────────────────

def test_429_is_retried_and_then_succeeds():
    sess = FakeSession([FakeResponse(status_code=429, headers={"Retry-After": "0"}),
                        FakeResponse(_tiff_bytes(4))])
    slept = []
    res = of.fetch_ortho(*CHICAGO_LONLAT, 200.0, year_hint=2023, city="chicago",
                         session=sess, max_rps=0, sleep_fn=slept.append)
    assert res.failure is None and res.crop is not None
    assert of.get_fetch_stats()["throttled"] == 1
    assert slept == [0.0]           # honoured Retry-After rather than the ladder


def test_sustained_throttling_gives_up_as_retryable():
    sess = FakeSession([FakeResponse(status_code=503)])
    res = of.fetch_ortho(*CHICAGO_LONLAT, 200.0, year_hint=2023, city="chicago",
                         session=sess, max_rps=0, sleep_fn=lambda s: None)
    assert res.failure == "read_error" and res.failure in of.RETRYABLE_FAILURES
    assert of.get_fetch_stats()["throttled"] == of.HTTP_RETRIES


def test_retry_after_header_is_honoured_over_the_backoff_ladder():
    sess = FakeSession([FakeResponse(status_code=429, headers={"Retry-After": "7"}),
                        FakeResponse(_tiff_bytes(4))])
    slept = []
    of.fetch_ortho(*CHICAGO_LONLAT, 200.0, year_hint=2023, city="chicago",
                   session=sess, max_rps=0, sleep_fn=slept.append)
    assert slept == [7.0]


# ─── rate limiter ─────────────────────────────────────────────────────────────

def test_limiter_bounds_the_average_rate():
    clock = {"t": 0.0}
    lim = of.RateLimiter(rate=10.0, burst=1.0,
                         time_fn=lambda: clock["t"],
                         sleep_fn=lambda s: clock.__setitem__("t", clock["t"] + s))
    lim.acquire()                       # consumes the single burst token
    for _ in range(5):
        lim.acquire()
    # 5 further tokens at 10/s must have advanced the clock by ~0.5 s.
    assert clock["t"] == pytest.approx(0.5, abs=1e-6)


def test_limiter_rate_zero_disables_limiting():
    lim = of.RateLimiter(rate=0.0)
    assert lim.acquire() == 0.0


def test_limiter_terminates_when_the_deficit_falls_below_clock_resolution():
    """REGRESSION: the first implementation re-checked the balance after
    sleeping. Once the residual deficit dropped below the clock's ULP, ``t +
    tiny == t``, so elapsed time read as zero, the balance never grew, and
    acquire() spun forever — a livelock that would have hung a production
    prediction run, not just this test. Virtual scheduling has no such loop.
    """
    # A large clock value makes the ULP coarse, which is what real
    # time.monotonic() looks like on a long-running host.
    clock = {"t": 1e9}
    lim = of.RateLimiter(rate=7.3, burst=1.0,
                         time_fn=lambda: clock["t"],
                         sleep_fn=lambda s: clock.__setitem__("t", clock["t"] + s))
    for _ in range(50):
        lim.acquire()          # must return, not spin
    assert clock["t"] > 1e9


def test_penalise_slows_every_worker_not_just_the_throttled_one():
    """If only the throttled thread backed off, the rest would keep hammering and
    the server would never get relief."""
    clock = {"t": 0.0}
    lim = of.RateLimiter(rate=10.0, burst=10.0,
                         time_fn=lambda: clock["t"],
                         sleep_fn=lambda s: clock.__setitem__("t", clock["t"] + s))
    lim.penalise(2.0)
    lim.acquire()
    assert clock["t"] >= 2.0


def test_limiter_is_shared_per_host():
    a = of.get_limiter("gis.cookcountyil.gov")
    b = of.get_limiter("gis.cookcountyil.gov")
    c = of.get_limiter("gismaps.kingcounty.gov")
    assert a is b and a is not c


def test_limiter_is_threadsafe():
    lim = of.RateLimiter(rate=0.0)
    errors = []

    def worker():
        try:
            for _ in range(200):
                lim.acquire()
        except Exception as exc:      # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


# ─── band-order guard ─────────────────────────────────────────────────────────

def test_band_order_guard_passes_on_vegetated_rgbn():
    """NIR high, red low over a park -> NDVI strongly positive."""
    veg = _tiff_bytes(4, fill=[40, 60, 40, 200])
    ok, ndvi = of.check_band_order(
        "chicago", 2023,
        fetch_fn=lambda *a, **k: of.OrthoFetchResult(
            crop=rasterio.open(MemoryFile(veg)).read(), actual_year=2023))
    assert ok and ndvi > 0.5


def test_band_order_guard_catches_a_permuted_response():
    """REGRESSION GUARD: a BGRN service would feed the model swapped channels and
    corrupt every prediction with nothing raising anywhere."""
    permuted = np.zeros((4, 8, 8), dtype=np.uint8)
    permuted[0] = 200      # "red" slot actually holding NIR
    permuted[3] = 40       # "NIR" slot actually holding blue
    ok, ndvi = of.check_band_order(
        "chicago", 2023,
        fetch_fn=lambda *a, **k: of.OrthoFetchResult(crop=permuted, actual_year=2023))
    assert not ok and ndvi < 0


def test_band_order_guard_skips_three_band_sources():
    ok, ndvi = of.check_band_order(
        "seattle", 2021,
        fetch_fn=lambda *a, **k: of.OrthoFetchResult(
            crop=np.zeros((4, 8, 8), np.uint8), actual_year=2021, nir_padded=True))
    assert ok and np.isnan(ndvi)


# ─── contract with the NAIP fetcher ───────────────────────────────────────────

def test_result_is_drop_in_compatible_with_naip_result():
    """prediction.fetch_prediction_chunk reads exactly these three attributes off
    whatever fetch_fn returns; divergence here breaks the sensor swap silently."""
    from src.data.naip_fetcher import NaipFetchResult
    for attr in ("crop", "actual_year", "failure", "nir_padded", "partial_coverage"):
        assert hasattr(of.OrthoFetchResult(None, None), attr)
        assert hasattr(NaipFetchResult(None, None), attr)


def test_make_fetch_fn_matches_the_fetch_naip_keyword_signature():
    import inspect
    from src.data.naip_fetcher import fetch_naip

    bound = of.make_fetch_fn("chicago")
    naip_params = set(inspect.signature(fetch_naip).parameters)
    ortho_params = set(inspect.signature(bound).parameters)
    assert naip_params <= ortho_params, naip_params - ortho_params


def test_bound_fetch_fn_accepts_and_ignores_search_cache():
    """The NAIP path always passes search_cache/cache_key; an image service has
    no catalogue, so they must be accepted and ignored rather than raising."""
    sess = FakeSession([FakeResponse(_tiff_bytes(4))])
    res = of.fetch_ortho(*CHICAGO_LONLAT, 200.0, year_hint=2023, city="chicago",
                         search_cache=object(), cache_key="17031010100",
                         session=sess, max_rps=0)
    assert res.crop is not None


def test_fetch_requires_a_bound_city():
    with pytest.raises(ValueError, match="requires a city"):
        of.fetch_ortho(*CHICAGO_LONLAT, 200.0, year_hint=2023)
