# -*- coding: utf-8 -*-
"""The nodata-hole guard in ``CyclicCacheManager._extract_raw_image``.

Why this exists
---------------
A NAIP or zarr nodata hole does not fail. It returns a structurally perfect
array of zeros: the STAC item is present, the window sits inside the raster,
the shape and dtype are right, ``failure`` is None and no flag is set. The
model then scores a black image and the pipeline writes the result out as a
real, finite, plausible-looking prediction.

Measured consequence before the guard (issue #36): Harford County NAIP 2013
produced 3,919 of 10,028 buildings carrying the identical value 0.68359375,
with 17 whole tracts constant; NYC zarr 2022 the same signature at 80,605 of
1,079,005 buildings and 146 Bronx tracts. Those tract-years then passed the
event study's balanced-panel check — balance tests presence, not validity — and
went into the ATT.

The guard drops the crop, which drops the building, which empties the
tract-year, which drops the tract in ``build_event_panel``'s balance step.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.main import CyclicCacheManager

BANDS, SIZE = 4, 8


class _Guard:
    """The two guard methods bound to a params dict, without constructing a
    CyclicCacheManager (whose __init__ wants a zarr store / NAIP catalog)."""

    def __init__(self, **params):
        self.params = params

    _zero_pixel_share = CyclicCacheManager._zero_pixel_share
    _reject_for_zero_pixels = CyclicCacheManager._reject_for_zero_pixels


def _crop(zero_rows: int, fill: int = 120) -> np.ndarray:
    """(C, H, W) uint8 crop whose first ``zero_rows`` rows are nodata."""
    a = np.full((BANDS, SIZE, SIZE), fill, dtype=np.uint8)
    a[:, :zero_rows, :] = 0
    return a


# ─── the share statistic ──────────────────────────────────────────────────────

@pytest.mark.parametrize("zero_rows,expected", [
    (0, 0.0), (2, 0.25), (4, 0.5), (6, 0.75), (SIZE, 1.0),
])
def test_zero_pixel_share_counts_all_band_zero_positions(zero_rows, expected):
    g = _Guard()
    assert g._zero_pixel_share(_crop(zero_rows)) == pytest.approx(expected)


def test_a_dark_but_real_pixel_is_not_nodata():
    """Zero in ONE band is ordinary (deep shadow, water). Only a position that
    is zero in every band is missing data — counting zero *values* instead
    would throw away real imagery."""
    g = _Guard()
    a = np.full((BANDS, SIZE, SIZE), 40, dtype=np.uint8)
    a[0] = 0                      # whole red band dark, other bands fine
    assert g._zero_pixel_share(a) == 0.0
    assert not g._reject_for_zero_pixels(a)


def test_padded_nir_alone_does_not_trip_the_zero_guard():
    """A 3-band `visual` asset gets a zero-filled NIR band. That is
    `reject_padded_nir`'s business; the two guards must stay independent or a
    padded-NIR crop would be indistinguishable from a nodata hole."""
    g = _Guard()
    a = np.full((BANDS, SIZE, SIZE), 200, dtype=np.uint8)
    a[3] = 0                      # NIR zero-padded, RGB intact
    assert g._zero_pixel_share(a) == 0.0
    assert not g._reject_for_zero_pixels(a)


def test_share_is_zero_for_degenerate_shapes():
    g = _Guard()
    assert g._zero_pixel_share(np.zeros((0,))) == 0.0
    assert g._zero_pixel_share(np.zeros((SIZE, SIZE))) == 0.0   # not (C, H, W)


# ─── the threshold ────────────────────────────────────────────────────────────

def test_rejects_above_half_keeps_at_or_below():
    """'More than 50%' — strictly greater, so an exactly-half crop survives."""
    g = _Guard(max_zero_pixel_share=0.5)
    assert not g._reject_for_zero_pixels(_crop(3))    # 0.375
    assert not g._reject_for_zero_pixels(_crop(4))    # 0.500 exactly -> keep
    assert g._reject_for_zero_pixels(_crop(5))        # 0.625
    assert g._reject_for_zero_pixels(_crop(SIZE))     # the Harford case


def test_default_is_active_at_one_half():
    """Absent from params, the guard must still run: the failure it prevents is
    silent, so defaulting to off would restore the original bug."""
    g = _Guard()
    assert g._reject_for_zero_pixels(_crop(SIZE))
    assert not g._reject_for_zero_pixels(_crop(0))


def test_none_disables_the_guard():
    g = _Guard(max_zero_pixel_share=None)
    assert not g._reject_for_zero_pixels(_crop(SIZE))


def test_threshold_is_configurable():
    strict = _Guard(max_zero_pixel_share=0.1)
    lax = _Guard(max_zero_pixel_share=0.9)
    quarter = _crop(2)                                 # 0.25
    assert strict._reject_for_zero_pixels(quarter)
    assert not lax._reject_for_zero_pixels(quarter)


def test_the_registered_default_matches_the_guard_default():
    """params in main.run() and the .get() fallback must not drift apart."""
    import inspect

    from src import main

    src = inspect.getsource(main)
    assert '"max_zero_pixel_share": 0.5,' in src


def test_blank_crop_is_a_countable_fetch_outcome():
    """The NAIP branch records the drop so it shows up in get_fetch_stats(),
    which is what makes the rate visible per (city, year) instead of the
    buildings just quietly vanishing."""
    from src.data import naip_fetcher as nf

    nf.reset_fetch_stats()
    assert "blank_crop" in nf.get_fetch_stats()
    nf.record_failure("blank_crop")
    assert nf.get_fetch_stats()["blank_crop"] == 1
    nf.reset_fetch_stats()
