"""Unit tests for the tract-keyed STAC search cache + search retry/observability
(src/data/naip_fetcher.py). Synthetic items only — no network, no real STAC.

Covers:
  * bbox helpers (buffer/union/contains/intersects),
  * TractSearchCache: miss→hit reuse, self-correcting expansion, failures never
    cached, LRU eviction, single-flight under concurrency,
  * _select_item: containment preference + year_hint ordering (must replicate
    the old inline sort exactly),
  * fetch_naip: search_error vs read_error split, cache-path item selection
    identical to the direct path, sign-at-read for cached (unsigned) items,
  * _search_items retry with backoff.
"""
import threading
import time
from datetime import datetime

import numpy as np
import pytest

from src.data import naip_fetcher as nf


# ── synthetic STAC objects ───────────────────────────────────────────────────

class _Asset:
    def __init__(self, href):
        self.href = href


class _Item:
    def __init__(self, id, bbox, year, assets=None):
        self.id = id
        self.bbox = bbox
        self.datetime = datetime(year, 6, 1)
        self.assets = {"image": _Asset(f"https://example.com/{id}.tif")} \
            if assets is None else assets


def _bbox_around(lon, lat, half_deg=0.01):
    return [lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg]


@pytest.fixture(autouse=True)
def _reset_stats():
    nf.reset_fetch_stats()
    yield
    nf.reset_fetch_stats()


# ── bbox helpers ─────────────────────────────────────────────────────────────

def test_buffer_bbox_contains_original_and_grows_by_meters():
    bbox = _bbox_around(-74.0, 40.7, 0.001)
    buffered = nf.buffer_bbox(bbox, 1500.0)
    assert nf.bbox_contains(buffered, bbox)
    # Each side must move out by ~1500 m: check the west edge geodesically.
    _, _, dist_w = nf._GEOD.inv(bbox[0], 40.7, buffered[0], 40.7)
    _, _, dist_n = nf._GEOD.inv(-74.0, bbox[3], -74.0, buffered[3])
    assert dist_w == pytest.approx(1500.0, rel=0.01)
    assert dist_n == pytest.approx(1500.0, rel=0.01)


def test_bbox_union_and_contains():
    a = [0.0, 0.0, 1.0, 1.0]
    b = [0.5, -1.0, 2.0, 0.5]
    u = nf.bbox_union(a, b)
    assert u == [0.0, -1.0, 2.0, 1.0]
    assert nf.bbox_contains(u, a) and nf.bbox_contains(u, b)
    assert not nf.bbox_contains(a, u)


def test_item_intersects_bbox():
    bbox = [0.0, 0.0, 1.0, 1.0]
    assert nf.item_intersects_bbox(_Item("a", [0.5, 0.5, 2.0, 2.0], 2020), bbox)
    assert nf.item_intersects_bbox(_Item("b", [-1.0, -1.0, 0.0, 0.0], 2020), bbox)  # edge touch
    assert not nf.item_intersects_bbox(_Item("c", [1.1, 1.1, 2.0, 2.0], 2020), bbox)
    # No usable bbox → kept (a per-crop search would have returned it too).
    assert nf.item_intersects_bbox(_Item("d", None, 2020), bbox)


# ── _select_item: must replicate the old inline sort ─────────────────────────

def test_select_item_prefers_containing_then_year_hint():
    bbox = _bbox_around(-74.0, 40.7, 0.001)
    covering = [-74.05, 40.65, -73.95, 40.75]
    sliver = [-74.0005, 40.65, -73.95, 40.75]  # intersects, doesn't contain
    items = [
        _Item("sliver_exact_year", sliver, 2018),
        _Item("covering_off_year", covering, 2016),
        _Item("covering_exact_year", covering, 2018),
    ]
    # Containment beats year proximity; among covering, closest year wins.
    assert nf._select_item(items, bbox, year_hint=2018).id == "covering_exact_year"
    assert nf._select_item(items[:2], bbox, year_hint=2018).id == "covering_off_year"
    # No hint → newest covering first.
    assert nf._select_item(items, bbox, year_hint=None).id == "covering_exact_year"
    assert nf._select_item(items, bbox, year_hint=2016).id == "covering_off_year"
    # Deterministic id tiebreak.
    tie = [_Item("b", covering, 2018), _Item("a", covering, 2018)]
    assert nf._select_item(tie, bbox, year_hint=2018).id == "a"
    assert nf._select_item([], bbox, year_hint=2018) is None


# ── TractSearchCache ─────────────────────────────────────────────────────────

def _install_fake_search(monkeypatch, items=None, fail_times=0):
    """Replace _search_items with a recorder returning ``items``."""
    calls = []
    failures = {"left": fail_times}

    def fake_search(bbox, max_items=None, signed=True, **kw):
        calls.append({"bbox": list(bbox), "max_items": max_items, "signed": signed})
        if failures["left"] > 0:
            failures["left"] -= 1
            raise RuntimeError("simulated STAC rate limit")
        return items if items is not None else []

    monkeypatch.setattr(nf, "_search_items", fake_search)
    return calls


def test_cache_one_search_serves_the_tract(monkeypatch):
    item = _Item("doqq", [-74.1, 40.6, -73.9, 40.8], 2018)
    calls = _install_fake_search(monkeypatch, items=[item])
    cache = nf.TractSearchCache(buffer_meters=1500.0)

    crop1 = _bbox_around(-74.0, 40.7, 0.0005)
    crop2 = _bbox_around(-74.001, 40.701, 0.0005)  # ~100m away, same tract
    assert cache.get_items("36061000100", crop1) == [item]
    assert cache.get_items("36061000100", crop2) == [item]

    assert len(calls) == 1
    # The one search must be unsigned (items are cached; tokens would expire)
    # and fully paginated (truncation would drop flight years for the tract).
    assert calls[0]["signed"] is False and calls[0]["max_items"] is None
    # The searched bbox covers both crops (buffer >> crop spacing).
    stats = nf.get_fetch_stats()
    assert stats["search_cache_miss"] == 1 and stats["search_cache_hit"] == 1
    assert stats["search_cache_expand"] == 0


def test_cache_expands_when_crop_falls_outside(monkeypatch):
    item = _Item("doqq", [-75.0, 40.0, -73.0, 41.5], 2018)
    calls = _install_fake_search(monkeypatch, items=[item])
    cache = nf.TractSearchCache(buffer_meters=1000.0)

    near = _bbox_around(-74.0, 40.7, 0.0005)
    far = _bbox_around(-74.3, 40.9, 0.0005)  # ~30 km: outside the 1 km buffer
    cache.get_items("g", near)
    cache.get_items("g", far)

    assert len(calls) == 2
    # Second search covers the union: both crops are inside it,
    # so a third request for either is a pure hit.
    assert nf.bbox_contains(calls[1]["bbox"], near)
    assert nf.bbox_contains(calls[1]["bbox"], far)
    cache.get_items("g", near)
    cache.get_items("g", far)
    assert len(calls) == 2
    stats = nf.get_fetch_stats()
    assert stats["search_cache_miss"] == 1 and stats["search_cache_expand"] == 1
    assert stats["search_cache_hit"] == 2


def test_cache_never_caches_failures(monkeypatch):
    item = _Item("doqq", [-74.1, 40.6, -73.9, 40.8], 2018)
    calls = _install_fake_search(monkeypatch, items=[item], fail_times=1)
    cache = nf.TractSearchCache()
    crop = _bbox_around(-74.0, 40.7, 0.0005)

    with pytest.raises(RuntimeError):
        cache.get_items("g", crop)
    # The failure was not cached: the next request searches again and succeeds.
    assert cache.get_items("g", crop) == [item]
    assert len(calls) == 2


def test_cache_lru_eviction(monkeypatch):
    item = _Item("doqq", [-180.0, -90.0, 180.0, 90.0], 2018)
    calls = _install_fake_search(monkeypatch, items=[item])
    cache = nf.TractSearchCache(max_entries=1)
    crop = _bbox_around(-74.0, 40.7, 0.0005)

    cache.get_items("A", crop)
    cache.get_items("B", crop)   # evicts A
    cache.get_items("A", crop)   # miss again
    assert len(calls) == 3


def test_cache_single_flight_under_concurrency(monkeypatch):
    item = _Item("doqq", [-74.1, 40.6, -73.9, 40.8], 2018)
    calls = []
    lock = threading.Lock()

    def slow_search(bbox, max_items=None, signed=True, **kw):
        with lock:
            calls.append(1)
        time.sleep(0.2)
        return [item]

    monkeypatch.setattr(nf, "_search_items", slow_search)
    cache = nf.TractSearchCache()
    crop = _bbox_around(-74.0, 40.7, 0.0005)

    threads = [threading.Thread(target=cache.get_items, args=("g", crop))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1   # cold tract, 8 workers, exactly one search


# ── _search_items retry ──────────────────────────────────────────────────────

class _FakeSearch:
    def __init__(self, items):
        self._items = items

    def items(self):
        return iter(self._items)


class _FlakyCatalog:
    def __init__(self, fail_times, items):
        self.fail_times = fail_times
        self.items = items
        self.calls = 0

    def search(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("simulated 429")
        return _FakeSearch(self.items)


def test_search_items_retries_then_succeeds(monkeypatch):
    catalog = _FlakyCatalog(fail_times=2, items=[_Item("x", [0, 0, 1, 1], 2020)])
    monkeypatch.setattr(nf, "get_catalog", lambda signed=True: catalog)
    items = nf._search_items([0, 0, 1, 1], retries=3, backoff_s=0.0)
    assert [it.id for it in items] == ["x"]
    assert catalog.calls == 3


def test_search_items_raises_after_exhausting_retries(monkeypatch):
    catalog = _FlakyCatalog(fail_times=99, items=[])
    monkeypatch.setattr(nf, "get_catalog", lambda signed=True: catalog)
    with pytest.raises(ConnectionError):
        nf._search_items([0, 0, 1, 1], retries=3, backoff_s=0.0)
    assert catalog.calls == 3


# ── fetch_naip failure modes + cache path (no network: local GeoTIFF) ────────

def _write_local_geotiff(path, bounds, size=64, count=4):
    import rasterio
    from rasterio.transform import from_bounds as transform_from_bounds
    data = np.random.default_rng(0).integers(
        1, 255, size=(count, size, size), dtype=np.uint8)
    transform = transform_from_bounds(*bounds, size, size)
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=count,
        dtype="uint8", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data)


def test_fetch_naip_counts_search_error(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("simulated 429")
    monkeypatch.setattr(nf, "_search_items", boom)
    res = nf.fetch_naip(-74.0, 40.7, 100.0, out_pixels=8)
    assert res.crop is None and res.failure == "search_error"
    stats = nf.get_fetch_stats()
    assert stats["search_error"] == 1 and stats["read_error"] == 0


def test_fetch_naip_cache_search_error(monkeypatch):
    monkeypatch.setattr(
        nf, "_search_items",
        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("429")))
    res = nf.fetch_naip(-74.0, 40.7, 100.0, out_pixels=8,
                        search_cache=nf.TractSearchCache(), cache_key="g")
    assert res.failure == "search_error"


def test_fetch_naip_cached_path_reads_and_signs_at_read_time(
        monkeypatch, tmp_path):
    bounds = [-74.02, 40.68, -73.98, 40.72]
    tif = tmp_path / "doqq.tif"
    _write_local_geotiff(tif, bounds)
    item = _Item("doqq", bounds, 2018,
                 assets={"image": _Asset(str(tif))})
    _install_fake_search(monkeypatch, items=[item])

    signed = []
    monkeypatch.setattr(nf.planetary_computer, "sign",
                        lambda href: (signed.append(href), href)[1])

    cache = nf.TractSearchCache()
    res = nf.fetch_naip(-74.0, 40.7, 100.0, nbands=4, out_pixels=16,
                        year_hint=2018, search_cache=cache, cache_key="g")
    assert res.failure is None
    assert res.crop.shape == (4, 16, 16) and res.actual_year == 2018
    assert signed == [str(tif)]          # cached (unsigned) href signed at read

    # Direct path must NOT re-sign (hrefs come pre-signed from the client).
    signed.clear()
    res2 = nf.fetch_naip(-74.0, 40.7, 100.0, nbands=4, out_pixels=16,
                         year_hint=2018)
    assert res2.failure is None and signed == []
    assert np.array_equal(res.crop, res2.crop)   # identical selection + read


def test_fetch_naip_cache_filters_to_crop_intersection(monkeypatch, tmp_path):
    """A tract-level item list may contain DOQQs that don't touch this crop —
    they must be filtered out before selection (else a nearer-year item from
    a non-intersecting DOQQ could win and yield an empty/black read)."""
    bounds = [-74.02, 40.68, -73.98, 40.72]
    tif = tmp_path / "doqq.tif"
    _write_local_geotiff(tif, bounds)
    covering = _Item("covering", bounds, 2016,
                     assets={"image": _Asset(str(tif))})
    elsewhere = _Item("elsewhere_exact_year", [-74.5, 40.68, -74.4, 40.72],
                      2018)  # exact year, but does not intersect the crop
    _install_fake_search(monkeypatch, items=[elsewhere, covering])
    monkeypatch.setattr(nf.planetary_computer, "sign", lambda href: href)

    res = nf.fetch_naip(-74.0, 40.7, 100.0, nbands=4, out_pixels=16,
                        year_hint=2018, search_cache=nf.TractSearchCache(),
                        cache_key="g")
    assert res.failure is None and res.actual_year == 2016


def test_fetch_naip_no_items_after_intersection_filter(monkeypatch):
    elsewhere = _Item("elsewhere", [-74.5, 40.68, -74.4, 40.72], 2018)
    _install_fake_search(monkeypatch, items=[elsewhere])
    res = nf.fetch_naip(-74.0, 40.7, 100.0, out_pixels=8,
                        search_cache=nf.TractSearchCache(), cache_key="g")
    assert res.crop is None and res.failure == "no_items"
