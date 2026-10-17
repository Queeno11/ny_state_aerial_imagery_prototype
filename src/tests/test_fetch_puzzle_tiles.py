"""Synthetic tests for src/fetch_puzzle_tiles.py (no network)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.fetch_puzzle_tiles as fp


def test_site_table_balanced_for_rich():
    rich = fp.select_sites(fp.SITES, ["rich"])
    assert sum(s.city == "nyc" for s in rich) >= 4 and sum(s.city == "chi" for s in rich) >= 4
    for s in rich:  # points sit in the right city
        if s.city == "nyc":
            assert 40.5 < s.lat < 41.0 and -74.3 < s.lon < -73.7
        else:
            assert 41.6 < s.lat < 42.1 and -88.0 < s.lon < -87.5


def test_slug_and_tile_path(tmp_path):
    s = fp.Site("nyc", "rich", "Billionaires Row 57th St", 40.0, -74.0)
    assert s.slug == "billionaires_row_57th_st"
    assert fp.tile_path(tmp_path, s, 3) == tmp_path / "rich" / "nyc_rich_3_billionaires_row_57th_st.png"


def test_to_rgb_image_takes_first_three_bands():
    crop = np.zeros((4, 5, 6), np.uint8)
    crop[0], crop[1], crop[2], crop[3] = 10, 20, 30, 99
    im = fp.to_rgb_image(crop)
    assert im.size == (6, 5) and im.getpixel((0, 0)) == (10, 20, 30)
    with pytest.raises(ValueError):
        fp.to_rgb_image(np.zeros((2, 5, 5), np.uint8))


def test_contact_sheet_grid():
    ims = [("a", fp.to_rgb_image(np.full((3, 8, 8), v, np.uint8))) for v in (0, 100, 200, 255, 50)]
    sheet = fp.contact_sheet(ims, cols=4, size=32)
    assert sheet.size == (4 * 32, 2 * (32 + 18))
    with pytest.raises(ValueError):
        fp.contact_sheet([])


def test_run_writes_tiles_sheet_and_table(tmp_path, monkeypatch):
    def fake_fetch(site, year, pixels):
        if site.name == "Streeterville":
            return SimpleNamespace(crop=None, actual_year=None, failure="no_items", partial_coverage=False)
        return SimpleNamespace(crop=np.full((3, pixels, pixels), 120, np.uint8), actual_year=2021,
                               failure=None, partial_coverage=False)
    monkeypatch.setattr(fp, "fetch_site", fake_fetch)
    table = fp.run(["rich"], 2021, 16, tmp_path)
    n = len(fp.select_sites(fp.SITES, ["rich"]))
    assert len(table) == n and table.failure.notna().sum() == 1
    assert (tmp_path / "rich" / "sheet.png").exists()
    assert len(list((tmp_path / "rich").glob("*_rich_*.png"))) == n - 1
    saved = pd.read_csv(tmp_path / "rich" / "sites.csv")
    assert set(saved.city) == {"nyc", "chi"}


def test_site_year_overrides_default():
    assert fp.site_year(fp.Site("nyc", "rich", "x", 40.7, -74.0, 2022), 2021) == 2022
    assert fp.site_year(fp.Site("nyc", "rich", "x", 40.7, -74.0), 2021) == 2021


def test_grid_overlay_marks_centre_and_tile_box():
    from PIL import Image
    im = Image.new("RGB", (400, 400), (0, 0, 0))
    g = fp.grid_overlay(im, meters=400.0, step=50.0, box_meters=200.0)
    assert im.getpixel((200, 50)) == (0, 0, 0)            # input untouched
    assert g.getpixel((200, 50)) == (255, 0, 0)           # red centre line
    assert g.getpixel((100, 120)) == (0, 255, 255)        # tile box edge at 100 px
    assert g.getpixel((250, 210)) == (255, 255, 0)        # +50 m grid line


def test_mid_and_poor_sites_balanced():
    for wealth in ("mid", "poor"):
        sites = fp.select_sites(fp.SITES, [wealth])
        assert sum(s.city == "nyc" for s in sites) >= 4 and sum(s.city == "chi" for s in sites) >= 4


def _toy_panel():
    import geopandas as gpd
    from shapely.geometry import box
    # two CBSAs, three tracts each; unit squares in a projected CRS
    geoms, rows = [], []
    for i, (cbsa, w) in enumerate([("A", 10e3), ("A", 20e3), ("A", 30e3), ("B", 5e3), ("B", 50e3), ("B", 60e3)]):
        geoms.append(box(i, 0, i + 1, 1))
        rows.append(dict(geoid_2023=f"{i:011d}", cbsa_code=cbsa, W2_i_r5pct_2022=w,
                         per_capita_income_usd_2022=w / 2))
    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:3857")


def test_acs_year_matches_flight_and_caps():
    assert fp.acs_year(2022) == 2022 and fp.acs_year(2024) == 2023 and fp.acs_year(None) == 2023


def test_within_cbsa_percentile_ranks_per_metro():
    v = pd.Series([10, 20, 30, 5, 50, 60.])
    pct = fp.within_cbsa_percentile(v, pd.Series(list("AAABBB")))
    assert list(pct.round(1)) == [33.3, 66.7, 100.0, 33.3, 66.7, 100.0]


def test_tract_labels_point_in_polygon():
    import geopandas as gpd
    panel = _toy_panel()
    # points at the centres of tract 1 (CBSA A) and tract 4 (CBSA B), given in lon/lat
    xy = gpd.GeoSeries(gpd.points_from_xy([1.5, 4.5], [0.5, 0.5]), crs="EPSG:3857").to_crs("EPSG:4326")
    pts = pd.DataFrame(dict(city=["x", "y"], k=[1, 1], name=["a", "b"], lon=xy.x, lat=xy.y, year=[2022, 2022]))
    out = fp.tract_labels(pts, panel)
    assert list(out.geoid) == ["00000000001", "00000000004"]
    assert list(out.w2_usd) == [20e3, 50e3] and list(out.income_usd) == [10e3, 25e3]
    assert list(out.w2_pct.round(1)) == [66.7, 66.7]
    assert "W2 $20k" in fp.label_text(out.iloc[0]) and "pct 67" in fp.label_text(out.iloc[0])


def test_numbered_sites_keep_numbers_when_filtered():
    sites = [fp.Site("nyc", "mid", n, 40.7, -74.0) for n in ("A one", "B two", "C three")]
    allk = fp.numbered_sites(["mid"], sites=sites)
    assert [k for _, k in allk] == [1, 2, 3]
    only = fp.numbered_sites(["mid"], names=["c_three"], sites=sites)
    assert [(s.name, k) for s, k in only] == [("C three", 3)]


def test_footprint_overlay_maps_native_bounds_to_pixels():
    from PIL import Image
    from shapely.geometry import box
    im = Image.new("RGB", (100, 100), (0, 0, 0))
    # window [0,200]x[0,200]; a square at x 50-100, y 150-200 -> px x 25-50, y 0-25 (top)
    out = fp.footprint_overlay(im, [box(50, 150, 100, 200)], (0, 0, 200, 200))
    inside, outside = out.getpixel((37, 12)), out.getpixel((80, 80))
    assert outside == (0, 0, 0) and inside != (0, 0, 0) and inside[2] > inside[1]


def test_load_footprints_filters_parts_by_bbox(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box
    d = tmp_path / "State"
    d.mkdir()
    gpd.GeoDataFrame({"building_id": [1, 2]}, geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
                     crs="EPSG:5070").to_parquet(d / "part_0000.parquet")
    gpd.GeoDataFrame({"building_id": [3]}, geometry=[box(500, 500, 501, 501)],
                     crs="EPSG:5070").to_parquet(d / "part_0001.parquet")
    got = fp.load_footprints("State", (-1, -1, 5, 5), root=tmp_path)
    assert list(got.building_id) == [1]
    assert fp.load_footprints("State", (900, 900, 901, 901), root=tmp_path).empty


def test_puzzle_picks_exist_in_site_table():
    names = {(s.wealth, s.name) for s in fp.SITES}
    assert set(fp.PUZZLE_PICKS.values()) <= names
    assert set(fp.PUZZLE_PICKS) == {f"{c}_{w}" for c in ("nyc", "chi") for w in ("rich", "mid", "poor")}
