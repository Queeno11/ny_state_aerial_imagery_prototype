"""Tests for src/data/naip_fetcher.py (geometry + failure modes; one optional live smoke)."""

import numpy as np
import pytest
from pyproj import Geod

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
    monkeypatch.setattr(nf, "get_catalog", lambda: _FakeCatalog(items=[]))
    res = nf.fetch_naip(-100.0, 40.0, 200.0)
    assert res.crop is None
    assert res.failure == "no_items"
    assert nf.get_fetch_stats()["no_items"] == 1


def test_search_error_counts_read_error(monkeypatch):
    nf.reset_fetch_stats()
    monkeypatch.setattr(nf, "get_catalog", lambda: _FakeCatalog(raise_search=True))
    res = nf.fetch_naip(-100.0, 40.0, 200.0)
    assert res.crop is None
    assert res.failure == "read_error"


def test_asset_missing(monkeypatch):
    nf.reset_fetch_stats()
    item = _FakeItem(2020, assets={})
    monkeypatch.setattr(nf, "get_catalog", lambda: _FakeCatalog(items=[item]))
    res = nf.fetch_naip(-100.0, 40.0, 200.0)
    assert res.crop is None
    assert res.failure == "asset_missing"
    assert res.actual_year == 2020   # provenance survives asset failure


def test_year_hint_picks_closest(monkeypatch):
    items = [_FakeItem(2012, "a"), _FakeItem(2018, "b"), _FakeItem(2021, "c")]
    monkeypatch.setattr(nf, "get_catalog", lambda: _FakeCatalog(items=items))
    res = nf.fetch_naip(-100.0, 40.0, 200.0, year_hint=2017)
    assert res.actual_year == 2018   # asset missing, but year selection ran first


def test_no_year_hint_is_deterministic_newest_first(monkeypatch):
    items = [_FakeItem(2012, "z"), _FakeItem(2021, "b"), _FakeItem(2021, "a")]
    monkeypatch.setattr(nf, "get_catalog", lambda: _FakeCatalog(items=items))
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
    monkeypatch.setattr(nf, "get_catalog", lambda: _FakeCatalog(items=items))
    res = nf.fetch_naip(-100.0, 40.0, 200.0, year_hint=2013)
    assert res.actual_year == 2012   # coverage outranks year proximity


def test_same_year_tie_broken_by_containment(monkeypatch):
    items = [_FakeItem(2013, "a_sliver", bbox=_SLIVER_BBOX),
             _FakeItem(2013, "b_full", bbox=_FULL_BBOX)]
    monkeypatch.setattr(nf, "get_catalog", lambda: _FakeCatalog(items=items))
    res = nf.fetch_naip(-100.0, 40.0, 200.0, year_hint=2013)
    assert res.actual_year == 2013
    # asset lookup ran on the full-coverage item (both fail, but stats prove order)
    assert res.failure == "asset_missing"


def test_no_year_hint_prefers_containing_over_newer(monkeypatch):
    items = [_FakeItem(2022, "new_sliver", bbox=_SLIVER_BBOX),
             _FakeItem(2018, "old_full", bbox=_FULL_BBOX)]
    monkeypatch.setattr(nf, "get_catalog", lambda: _FakeCatalog(items=items))
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
    monkeypatch.setattr(nf, "get_catalog", lambda: _FakeCatalog(items=[]))
    crop, year = nf.fetch_naip_crop(-100.0, 40.0, 200.0)
    assert crop is None and year is None


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
