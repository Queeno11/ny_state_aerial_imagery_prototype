"""Tests for src/data/naip_fetcher.py (geometry + failure modes; one optional live smoke)."""

from contextlib import contextmanager

import numpy as np
import pytest
from pyproj import Geod, Transformer
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from src.data import naip_fetcher as nf


# ── bbox geometry (no network) ────────────────────────────────────────────────

@pytest.mark.parametrize("lat", [25.8, 38.5, 48.5])   # southern / mid / northern CONUS
@pytest.mark.parametrize("lon", [-118.0, -95.0, -74.0])
def test_lonlat_bbox_width_matches_meters(lon, lat):
    size = 200.0
    bbox = nf.lonlat_bbox(lon, lat, size)
    geod = Geod(ellps="WGS84")
    mid_lat = (bbox[1] + bbox[3]) / 2
    _, _, width_m = geod.inv(bbox[0], mid_lat, bbox[2], mid_lat)
    mid_lon = (bbox[0] + bbox[2]) / 2
    _, _, height_m = geod.inv(mid_lon, bbox[1], mid_lon, bbox[3])
    # EPSG:5070 is equal-area, not conformal: allow 1% distance distortion.
    assert width_m == pytest.approx(size, rel=0.01)
    assert height_m == pytest.approx(size, rel=0.01)


def test_lonlat_bbox_contains_center():
    bbox = nf.lonlat_bbox(-100.0, 40.0, 200.0)
    assert bbox[0] < -100.0 < bbox[2]
    assert bbox[1] < 40.0 < bbox[3]


# ── failure modes (mocked STAC) ───────────────────────────────────────────────

class _FakeSearch:
    def __init__(self, items):
        self._items = items

    def items(self):
        return iter(self._items)


class _FakeCatalog:
    def __init__(self, items=None, raise_search=False):
        self._items = items or []
        self._raise = raise_search

    def search(self, **kwargs):
        if self._raise:
            raise ConnectionError("boom")
        return _FakeSearch(self._items)


class _FakeItem:
    def __init__(self, year, item_id="a", assets=None, bbox=None):
        import datetime
        self.datetime = datetime.datetime(year, 6, 1)
        self.id = item_id
        self.assets = assets if assets is not None else {}
        if bbox is not None:
            self.bbox = bbox


# EPSG:4326 footprints relative to the 200m search bbox around (-100, 40):
# one fully containing it, one only overlapping its western half (a "sliver"
# DOQQ — the stretched-crop failure mode).
_FULL_BBOX = [-100.1, 39.9, -99.9, 40.1]
_SLIVER_BBOX = [-100.1, 39.9, -100.0, 40.1]


def test_no_items_failure(monkeypatch):
    nf.reset_fetch_stats()
    monkeypatch.setattr(nf, "get_catalog", lambda signed=True: _FakeCatalog(items=[]))
    res = nf.fetch_naip(-100.0, 40.0, 200.0)
    assert res.crop is None
    assert res.failure == "no_items"
    assert nf.get_fetch_stats()["no_items"] == 1


def test_search_error_counted_separately_from_read_error(monkeypatch):
    # Search failures used to be lumped into read_error, which hid the STAC
    # rate-limiting behind blob-read noise. They are now their own mode, and
    # the search retries (backoff zeroed here) before giving up.
    nf.reset_fetch_stats()
    monkeypatch.setattr(nf, "SEARCH_BACKOFF_S", 0.0)
    monkeypatch.setattr(nf, "get_catalog", lambda signed=True: _FakeCatalog(raise_search=True))
    res = nf.fetch_naip(-100.0, 40.0, 200.0)
    assert res.crop is None
    assert res.failure == "search_error"
    stats = nf.get_fetch_stats()
    assert stats["search_error"] == 1 and stats["read_error"] == 0


def test_asset_missing(monkeypatch):
    nf.reset_fetch_stats()
    item = _FakeItem(2020, assets={})
    monkeypatch.setattr(nf, "get_catalog", lambda signed=True: _FakeCatalog(items=[item]))
    res = nf.fetch_naip(-100.0, 40.0, 200.0)
    assert res.crop is None
    assert res.failure == "asset_missing"
    assert res.actual_year == 2020   # provenance survives asset failure


def test_year_hint_picks_closest(monkeypatch):
    items = [_FakeItem(2012, "a"), _FakeItem(2018, "b"), _FakeItem(2021, "c")]
    monkeypatch.setattr(nf, "get_catalog", lambda signed=True: _FakeCatalog(items=items))
    res = nf.fetch_naip(-100.0, 40.0, 200.0, year_hint=2017)
    assert res.actual_year == 2018   # asset missing, but year selection ran first


def test_exact_year_no_same_year_flight_is_no_year_match(monkeypatch):
    # Imagery exists for the window but none flown in year_hint: exact mode
    # must fail (permanently) instead of substituting the closest year.
    nf.reset_fetch_stats()
    items = [_FakeItem(2012, "a"), _FakeItem(2018, "b"), _FakeItem(2021, "c")]
    monkeypatch.setattr(nf, "get_catalog", lambda signed=True: _FakeCatalog(items=items))
    res = nf.fetch_naip(-100.0, 40.0, 200.0, year_hint=2017, exact_year=True)
    assert res.crop is None
    assert res.failure == "no_year_match"
    assert res.actual_year is None                    # nothing was substituted
    stats = nf.get_fetch_stats()
    assert stats["no_year_match"] == 1 and stats["no_items"] == 0
    assert "no_year_match" in nf.PERMANENT_FAILURES   # never retried
    assert "no_year_match" not in nf.RETRYABLE_FAILURES


def test_exact_year_same_year_available_selects_it(monkeypatch):
    items = [_FakeItem(2012, "a"), _FakeItem(2017, "b"), _FakeItem(2021, "c")]
    monkeypatch.setattr(nf, "get_catalog", lambda signed=True: _FakeCatalog(items=items))
    res = nf.fetch_naip(-100.0, 40.0, 200.0, year_hint=2017, exact_year=True)
    assert res.actual_year == 2017   # asset missing, but same-year selection ran


def test_exact_year_prefers_same_year_sliver_over_offyear_full():
    # The year filter must run BEFORE the coverage preference: a same-year
    # partial DOQQ (boundless read, flagged) beats rejection, while
    # substitution mode keeps preferring the fully-covering off-year tile.
    bbox = nf.lonlat_bbox(-100.0, 40.0, 200.0)
    offyear_full = _FakeItem(2018, "full", bbox=_FULL_BBOX)
    sameyear_sliver = _FakeItem(2017, "sliver", bbox=_SLIVER_BBOX)
    items = [offyear_full, sameyear_sliver]
    assert nf._select_item(items, bbox, 2017) is offyear_full
    assert nf._select_item(items, bbox, 2017, exact_year=True) is sameyear_sliver


def test_no_year_hint_is_deterministic_newest_first(monkeypatch):
    items = [_FakeItem(2012, "z"), _FakeItem(2021, "b"), _FakeItem(2021, "a")]
    monkeypatch.setattr(nf, "get_catalog", lambda signed=True: _FakeCatalog(items=items))
    res1 = nf.fetch_naip(-100.0, 40.0, 200.0)
    res2 = nf.fetch_naip(-100.0, 40.0, 200.0)
    assert res1.actual_year == res2.actual_year == 2021


# ── full-coverage item preference (issue: stretched crops at DOQQ edges) ─────

def test_item_contains_bbox():
    bbox = nf.lonlat_bbox(-100.0, 40.0, 200.0)
    assert nf.item_contains_bbox(_FakeItem(2020, bbox=_FULL_BBOX), bbox)
    assert not nf.item_contains_bbox(_FakeItem(2020, bbox=_SLIVER_BBOX), bbox)
    assert not nf.item_contains_bbox(_FakeItem(2020), bbox)          # no bbox attr
    assert not nf.item_contains_bbox(_FakeItem(2020, bbox=[]), bbox)  # empty bbox


def test_containing_item_beats_closer_year_sliver(monkeypatch):
    # 2013 tile only covers a sliver of the window; 2012 neighbor covers it all.
    items = [_FakeItem(2013, "sliver", bbox=_SLIVER_BBOX),
             _FakeItem(2012, "full", bbox=_FULL_BBOX)]
    monkeypatch.setattr(nf, "get_catalog", lambda signed=True: _FakeCatalog(items=items))
    res = nf.fetch_naip(-100.0, 40.0, 200.0, year_hint=2013)
    assert res.actual_year == 2012   # coverage outranks year proximity


def test_same_year_tie_broken_by_containment(monkeypatch):
    items = [_FakeItem(2013, "a_sliver", bbox=_SLIVER_BBOX),
             _FakeItem(2013, "b_full", bbox=_FULL_BBOX)]
    monkeypatch.setattr(nf, "get_catalog", lambda signed=True: _FakeCatalog(items=items))
    res = nf.fetch_naip(-100.0, 40.0, 200.0, year_hint=2013)
    assert res.actual_year == 2013
    # asset lookup ran on the full-coverage item (both fail, but stats prove order)
    assert res.failure == "asset_missing"


def test_no_year_hint_prefers_containing_over_newer(monkeypatch):
    items = [_FakeItem(2022, "new_sliver", bbox=_SLIVER_BBOX),
             _FakeItem(2018, "old_full", bbox=_FULL_BBOX)]
    monkeypatch.setattr(nf, "get_catalog", lambda signed=True: _FakeCatalog(items=items))
    res = nf.fetch_naip(-100.0, 40.0, 200.0)
    assert res.actual_year == 2018


class _FakeWindow:
    def __init__(self, col_off, row_off, width, height):
        self.col_off, self.row_off = col_off, row_off
        self.width, self.height = width, height


def test_window_exceeds_raster():
    inside = _FakeWindow(10.0, 10.0, 100.0, 100.0)
    assert not nf.window_exceeds_raster(inside, 500, 500)
    # sticks out on each side
    assert nf.window_exceeds_raster(_FakeWindow(-5.0, 10.0, 100.0, 100.0), 500, 500)
    assert nf.window_exceeds_raster(_FakeWindow(10.0, -5.0, 100.0, 100.0), 500, 500)
    assert nf.window_exceeds_raster(_FakeWindow(450.0, 10.0, 100.0, 100.0), 500, 500)
    assert nf.window_exceeds_raster(_FakeWindow(10.0, 450.0, 100.0, 100.0), 500, 500)
    # float noise at an exact edge is tolerated
    edge = _FakeWindow(0.0, 0.0, 500.0 + 1e-6, 500.0)
    assert not nf.window_exceeds_raster(edge, 500, 500)


def test_backward_compatible_wrapper(monkeypatch):
    monkeypatch.setattr(nf, "get_catalog", lambda signed=True: _FakeCatalog(items=[]))
    crop, year = nf.fetch_naip_crop(-100.0, 40.0, 200.0)
    assert crop is None and year is None


# ── mosaic read (option #2: read a tract's window once, crop locally) ────────
#
# The contract: reading a crop from open_naip_mosaic(...) must be byte-identical
# — pixels, NIR padding, and the partial-coverage flag — to reading it straight
# from the DOQQ with read_naip_crop_from_src. These build a synthetic NAIP-like
# COG in EPSG:5070 (Albers, as NAIP is) with a per-pixel gradient so any pixel
# misplacement or resampling drift surfaces, then compare the two paths.

_SYN_EPSG = 5070
_SYN_X0, _SYN_Y0 = -100000.0, 2000000.0   # arbitrary CONUS Albers origin
_SYN_PX = 0.6                             # native GSD (m), like 0.6 m NAIP
_SYN_SIZE = 400                           # 400 px = 240 m square DOQQ stand-in


@contextmanager
def _synthetic_naip_src(bands=4):
    """An in-memory NAIP-like COG with a distinctive per-(row,col,band) pattern."""
    transform = from_origin(_SYN_X0, _SYN_Y0, _SYN_PX, _SYN_PX)
    rows = np.arange(_SYN_SIZE, dtype=np.int64)[:, None]
    cols = np.arange(_SYN_SIZE, dtype=np.int64)[None, :]
    data = np.stack([((rows * 7 + cols * 3 + b * 29) % 251).astype(np.uint8)
                     for b in range(bands)])
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", height=_SYN_SIZE, width=_SYN_SIZE,
                     count=bands, dtype="uint8", crs=f"EPSG:{_SYN_EPSG}",
                     transform=transform) as dst:
            dst.write(data)
        with mf.open() as src:
            yield src


def _bbox_for_native(px_x, px_y, crop_m, to4326):
    """EPSG:4326 crop bbox centered on a pixel-space (col, row) in the synthetic raster."""
    east = _SYN_X0 + px_x * _SYN_PX
    north = _SYN_Y0 - px_y * _SYN_PX
    lon, lat = to4326.transform(east, north)
    return nf.lonlat_bbox(lon, lat, crop_m)


def test_mosaic_read_matches_direct_read_interior():
    to4326 = Transformer.from_crs(f"EPSG:{_SYN_EPSG}", "EPSG:4326", always_xy=True)
    # Three overlapping 30 m windows clustered near the raster center, so their
    # covering union re-reads shared blocks the mosaic must serve once.
    bboxes = [_bbox_for_native(cx, cy, 30.0, to4326)
              for cx, cy in [(200, 200), (210, 205), (195, 210)]]
    with _synthetic_naip_src(bands=4) as src:
        direct = [nf.read_naip_crop_from_src(src, bb, 4, 64) for bb in bboxes]
        with nf.open_naip_mosaic(src, bboxes, 4) as mem:
            assert mem is not None                      # union fits → mosaicked
            mosaic = [nf.read_naip_crop_from_src(mem, bb, 4, 64) for bb in bboxes]
    for (dc, dn, dp, df), (mc, mn, mp, mf_) in zip(direct, mosaic):
        assert np.array_equal(dc, mc)                   # identical pixels
        assert (dn, dp, df) == (mn, mp, mf_)            # identical flags
        assert not dp                                   # interior → not partial


def test_mosaic_read_matches_direct_read_partial_edge():
    to4326 = Transformer.from_crs(f"EPSG:{_SYN_EPSG}", "EPSG:4326", always_xy=True)
    # One window jammed into the top-left corner so it overruns the DOQQ: the
    # direct read goes boundless + zero-fill + partial flag; the mosaic (union
    # clipped to the raster) must reproduce that exactly.
    edge = _bbox_for_native(6, 6, 40.0, to4326)
    interior = _bbox_for_native(200, 200, 40.0, to4326)
    bboxes = [edge, interior]
    with _synthetic_naip_src(bands=4) as src:
        direct = [nf.read_naip_crop_from_src(src, bb, 4, 48) for bb in bboxes]
        with nf.open_naip_mosaic(src, bboxes, 4) as mem:
            assert mem is not None
            mosaic = [nf.read_naip_crop_from_src(mem, bb, 4, 48) for bb in bboxes]
    assert direct[0][2] and mosaic[0][2]                # both flag partial
    for (dc, dn, dp, df), (mc, mn, mp, mf_) in zip(direct, mosaic):
        assert np.array_equal(dc, mc)
        assert (dn, dp, df) == (mn, mp, mf_)


def test_mosaic_pads_nir_like_direct_for_three_band_source():
    to4326 = Transformer.from_crs(f"EPSG:{_SYN_EPSG}", "EPSG:4326", always_xy=True)
    bboxes = [_bbox_for_native(cx, cy, 30.0, to4326)
              for cx, cy in [(180, 190), (185, 195)]]
    with _synthetic_naip_src(bands=3) as src:      # visual asset: no NIR band
        direct = [nf.read_naip_crop_from_src(src, bb, 4, 64) for bb in bboxes]
        with nf.open_naip_mosaic(src, bboxes, 4) as mem:
            assert mem is not None
            mosaic = [nf.read_naip_crop_from_src(mem, bb, 4, 64) for bb in bboxes]
    for (dc, dn, dp, df), (mc, mn, mp, mf_) in zip(direct, mosaic):
        assert dc.shape == (4, 64, 64) and dn and mn   # 4th band synthesized
        assert np.array_equal(dc, mc)
        assert np.array_equal(dc[3], np.zeros((64, 64), np.uint8))  # padded NIR


def test_mosaic_falls_back_to_none_when_union_too_large():
    to4326 = Transformer.from_crs(f"EPSG:{_SYN_EPSG}", "EPSG:4326", always_xy=True)
    bboxes = [_bbox_for_native(50, 50, 30.0, to4326),
              _bbox_for_native(350, 350, 30.0, to4326)]  # spread → large union
    with _synthetic_naip_src(bands=4) as src:
        with nf.open_naip_mosaic(src, bboxes, 4, max_union_pixels=64) as mem:
            assert mem is None                          # caller reads per-window


def test_mosaic_none_for_empty_and_broken_src():
    with _synthetic_naip_src(bands=4) as src:
        with nf.open_naip_mosaic(src, [], 4) as mem:
            assert mem is None
    # A non-raster src (e.g. a test double) must not raise — just fall back.
    with nf.open_naip_mosaic(("not", "a", "raster"), [[-100.0, 40.0, -99.9, 40.1]], 4) as mem:
        assert mem is None


# ── optional live smoke (network to Planetary Computer) ─────────────────────

@pytest.mark.network
def test_live_fetch_smoke():
    """One tiny real fetch over Wilmington, DE. Skipped unless -m network."""
    res = nf.fetch_naip(-75.55, 39.75, crop_size_meters=200, nbands=4,
                        out_pixels=64, year_hint=2018)
    assert res.crop is not None, f"live fetch failed: {res.failure}"
    assert res.crop.shape == (4, 64, 64)
    assert res.crop.dtype == np.uint8
    assert res.actual_year is not None
