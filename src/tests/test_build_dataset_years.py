"""Year handling: label-cache tag identity + panel-year -> ACS vintage clamp.

The study years are now every year the annual ACS panel supports (plus one
clamped edge year each side), not the legacy even-years grid — these tests
pin the two pieces that make that safe: the label cache tag must identify the
year SET (not just its span), and edge years must clamp to the nearest ACS
vintage.
"""

from src import build_dataset


def test_pair_years_tag_distinguishes_sets_with_same_span():
    even = build_dataset.pair_years_tag(range(2010, 2025, 2))
    full = build_dataset.pair_years_tag(range(2010, 2025))
    assert even == "years2010-2024n8"
    assert full == "years2010-2024n15"
    assert even != full


def test_get_closest_acs_year_clamps_edges():
    assert build_dataset.get_closest_acs_year(2010) == 2011   # clamped up
    assert build_dataset.get_closest_acs_year(2024) == 2023   # clamped down
    assert build_dataset.get_closest_acs_year(2016) == 2016   # annual: exact
