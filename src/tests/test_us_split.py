"""Tests for the US scale-up sampling overrides — pure synthetic data.

Covers (1) forced-split / NYC-exclusion support in cbsa_brackets.build_city_split
and (2) the src.data.us_split policy functions (NYC tract map, clustered val
carve, building-type resolution). No real feathers, no .env, no network.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.data import cbsa_brackets as cb
from src.data import us_split as us


# ── helpers ──────────────────────────────────────────────────────────────────

def _pop_df(pops: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "cbsa_code": list(pops),
        "cbsa_title": [f"City {c}" for c in pops],
        "population": list(pops.values()),
    })


def _universe():
    pops = {
        "10010": 19_000_000, "10020": 13_000_000, "10030": 9_500_000, "10040": 7_000_000,
        "20010": 4_000_000, "20020": 2_500_000, "20030": 1_200_000,
        "30010": 950_000, "30020": 850_000, "30030": 760_000,
        "40010": 700_000, "40020": 550_000,
    }
    tracts = {
        "10010": 2200, "10020": 1500, "10030": 1100, "10040": 800,
        "20010": 600, "20020": 380, "20030": 200,
        "30010": 160, "30020": 140, "30030": 120,
        "40010": 110, "40020": 80,
    }
    return pd.Series(tracts), _pop_df(pops)


def _grid_city(cbsa_code, x0, n=10, cell=1000.0, score_base=0.0):
    """n×n grid of square 'tracts' (cell metres) offset at x0. Returns a GeoDataFrame
    with GEOID, cbsa_code, geometry, and one Rel_Score_2020 column."""
    import geopandas as gpd
    from shapely.geometry import box
    rows = []
    for i in range(n):
        for j in range(n):
            gx = x0 + i * cell
            gy = j * cell
            rows.append({
                "GEOID": f"{cbsa_code}{i:02d}{j:02d}",
                "cbsa_code": cbsa_code,
                "geometry": box(gx, gy, gx + cell, gy + cell),
                # varied score so income quintiles are non-degenerate
                "Rel_Score_2020": score_base + i - j + 0.1 * (i * j % 5),
            })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{us.METRIC_EPSG}")


# ── build_city_split: forced_splits + exclude_cbsas ──────────────────────────

def test_forced_splits_pin_cities():
    tract_counts, pop_df = _universe()
    forced = {"20020": "test", "30010": "test"}
    out = cb.build_city_split(tract_counts, pop_df, forced_splits=forced)
    split = dict(zip(out["cbsa_code"], out["split"]))
    assert split["20020"] == "test"
    assert split["30010"] == "test"
    # every city still assigned
    assert out["split"].notna().all()


def test_exclude_cbsas_drops_city_entirely():
    tract_counts, pop_df = _universe()
    out = cb.build_city_split(tract_counts, pop_df, exclude_cbsas=["10010"])
    assert "10010" not in set(out["cbsa_code"])
    # excluded city does not consume a split slot; others intact
    assert len(out) == 11
    assert {"train", "val", "test"} <= set(out["split"])


def test_forced_and_exclude_together_nyc_like():
    """NYC excluded (within-city) + Chicago-like mega + FL/TN forced to test."""
    tract_counts, pop_df = _universe()
    forced = {"10030": "test", "20010": "test", "30020": "test", "40020": "test"}
    out = cb.build_city_split(
        tract_counts, pop_df, forced_splits=forced, exclude_cbsas=["10010"]
    )
    split = dict(zip(out["cbsa_code"], out["split"]))
    assert "10010" not in split
    for c in forced:
        assert split[c] == "test"
    assert (out["split"] == "train").any() and (out["split"] == "val").any()


def test_missing_forced_code_warns_and_is_ignored():
    tract_counts, pop_df = _universe()
    with pytest.warns(UserWarning):
        out = cb.build_city_split(tract_counts, pop_df, forced_splits={"99999": "test"})
    assert "99999" not in set(out["cbsa_code"])


def test_forced_split_is_deterministic():
    tract_counts, pop_df = _universe()
    a = cb.build_city_split(tract_counts, pop_df, forced_splits={"20020": "test"})
    b = cb.build_city_split(tract_counts, pop_df, forced_splits={"20020": "test"})
    pd.testing.assert_frame_equal(a, b)


def test_forced_test_counts_toward_bracket_target_not_on_top():
    """Net-of-manual-selection: a forced-test city eats into the bracket's 30%
    test target instead of adding on top, so the large-bracket test share stays
    near 30% overall (not 30% + the forced city's share)."""
    tract_counts, pop_df = _universe()
    # large bracket = {20010:600, 20020:380, 20030:200} -> 1180 tracts.
    forced = {"20030": "test"}          # 200/1180 = 17% of large already in test
    out = cb.build_city_split(tract_counts, pop_df, forced_splits=forced)
    large = out[out["bracket"] == "large"]
    test_share = large.loc[large["split"] == "test", "n_tracts"].sum() / large["n_tracts"].sum()
    # With forced 17% counted toward a 30% target, the free cities add little more
    # test; total test share should stay well under "30% + 17%".
    assert test_share <= 0.47
    assert (large["split"] == "test").any()          # forced city is in test
    assert large.set_index("cbsa_code").loc["20030", "split"] == "test"


def test_bracket_targets_are_65_5_30():
    assert cb.BRACKET_TARGETS == {"train": 0.65, "val": 0.05, "test": 0.30}


def test_spatial_val_type_names():
    assert us.SPATIAL_VAL_TYPES == ("val_within_nyc", "val_within_chicago")
    assert us.VAL_CLUSTER_CBSAS == ("16980",)      # Chicago only; NYC via notebook


# ── us_split policy constants ────────────────────────────────────────────────

def test_forced_test_map_covers_all_and_is_test():
    m = us.forced_test_split_map()
    assert set(m) == set(us.FORCED_TEST_CBSAS)
    assert set(m.values()) == {"test"}
    assert us.NYC_CBSA not in m                      # NYC is never a forced test city
    assert set(us.VAL_CLUSTER_CBSAS) <= set(m)       # val cities are forced test cities


def test_mean_rel_score_averages_year_columns():
    df = pd.DataFrame({
        "Rel_Score_2011": [0.0, 2.0, np.nan],
        "Rel_Score_2020": [2.0, np.nan, np.nan],
        "other": [9, 9, 9],
    })
    s = us.mean_rel_score(df)
    assert s.iloc[0] == pytest.approx(1.0)
    assert s.iloc[1] == pytest.approx(2.0)
    assert np.isnan(s.iloc[2])


def test_mean_rel_score_requires_columns():
    with pytest.raises(KeyError):
        us.mean_rel_score(pd.DataFrame({"foo": [1]}))


def test_nyc_tract_type_map_reads_and_remaps(tmp_path):
    p = tmp_path / "nyc_tract_splits.feather"
    pd.DataFrame({
        "GEOID": ["36061000100", "36061000200", "36061000300", "36061000400"],
        "type": ["train", "val", "test", "dead_zone"],
    }).to_feather(p)
    m = us.nyc_tract_type_map(p)
    assert m["36061000100"] == "train"
    assert m["36061000200"] == "val_within_nyc"   # val -> val_within_nyc
    assert m["36061000300"] == "test"
    assert m["36061000400"] == "dead_zone"


def test_nyc_tract_type_map_missing_file_warns(tmp_path):
    with pytest.warns(UserWarning):
        assert us.nyc_tract_type_map(tmp_path / "nope.feather") == {}


# ── clustered val carve (geometry) ───────────────────────────────────────────

def test_grow_clusters_captures_a_fraction():
    gdf = _grid_city("16980", x0=0.0, n=10)          # 100 tracts
    score = us.mean_rel_score(gdf)
    rng = np.random.default_rng(0)
    caught = us.grow_stratified_tract_clusters(
        gdf, cluster_radius=us.CLUSTER_RADIUS_M, score=score,
        eval_fraction=0.10, rng=rng,
    )
    assert 0 < len(caught) < len(gdf)                # some, not all
    assert caught <= set(gdf["GEOID"])


def test_grow_clusters_deterministic_given_rng():
    gdf = _grid_city("16980", x0=0.0, n=8)
    score = us.mean_rel_score(gdf)
    a = us.grow_stratified_tract_clusters(
        gdf, us.CLUSTER_RADIUS_M, score, 0.15, np.random.default_rng(7))
    b = us.grow_stratified_tract_clusters(
        gdf, us.CLUSTER_RADIUS_M, score, 0.15, np.random.default_rng(7))
    assert a == b


def test_carve_test_city_val_labels_and_quarantines():
    # Chicago (16980) carves val_within_chicago; a non-mapped city defaults to
    # val_spatial. Both quarantined by a dead_zone buffer.
    import geopandas as gpd
    c1 = _grid_city("16980", x0=0.0, n=10)
    c2 = _grid_city("42660", x0=1_000_000.0, n=10, score_base=5.0)
    panel = gpd.GeoDataFrame(pd.concat([c1, c2], ignore_index=True),
                             geometry="geometry", crs=f"EPSG:{us.METRIC_EPSG}")
    ov = us.carve_test_city_val(
        panel, val_cluster_cbsas=("16980", "42660"),
        eval_fraction=0.10, seed=1,
    )
    assert ov, "expected some val/dead_zone assignments"
    assert set(ov.values()) <= {"val_within_chicago", "val_spatial", "dead_zone"}
    # Chicago tracts labelled with its named val type
    chi_vals = {v for g, v in ov.items() if g.startswith("16980") and v != "dead_zone"}
    assert chi_vals == {"val_within_chicago"}
    # the unmapped city falls back to the generic bucket
    other_vals = {v for g, v in ov.items() if g.startswith("42660") and v != "dead_zone"}
    assert other_vals == {"val_spatial"}


def test_carve_default_uses_module_config():
    """With module defaults, only Chicago carves and it is val_within_chicago."""
    import geopandas as gpd
    chi = _grid_city("16980", x0=0.0, n=10)
    ov = us.carve_test_city_val(chi, eval_fraction=0.10, seed=1)  # defaults
    assert {v for v in ov.values() if v != "dead_zone"} == {"val_within_chicago"}


def test_greater_nyc_routed_to_train_with_buffer():
    """Uncovered NYC-CBSA tracts -> train, except those buffered by a borough
    holdout (test/val_within_nyc) -> dead_zone."""
    import geopandas as gpd
    from shapely.geometry import box
    cell = 1000.0
    rows = []
    # A 1x5 strip of NYC-CBSA tracts along x. tract 0 is a borough TEST holdout
    # (covered); 1 is adjacent (should be dead_zone); 4 is far (train).
    for i in range(5):
        rows.append({"GEOID": f"{us.NYC_CBSA}00{i}", "cbsa_code": us.NYC_CBSA,
                     "geometry": box(i * cell, 0, i * cell + cell, cell),
                     "Rel_Score_2020": float(i)})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{us.METRIC_EPSG}")
    nyc_types = {f"{us.NYC_CBSA}000": "test"}     # tract 0 is a borough holdout
    ov = us.nyc_greater_metro_overrides(gdf, nyc_types, dead_zone=us.DEAD_ZONE_M)
    assert f"{us.NYC_CBSA}000" not in ov          # covered tract untouched here
    assert ov[f"{us.NYC_CBSA}001"] == "dead_zone" # adjacent to the holdout
    assert ov[f"{us.NYC_CBSA}004"] == "train"     # far away
    # far tracts dominate -> most are train
    assert sum(v == "train" for v in ov.values()) >= 2


def test_build_tract_overrides_borough_split_wins(tmp_path, monkeypatch):
    """NYC notebook split (boroughs) is authoritative over the greater-metro fill."""
    import geopandas as gpd
    from shapely.geometry import box
    # write a tiny NYC notebook split
    p = tmp_path / "nyc_tract_splits.feather"
    pd.DataFrame({"GEOID": [f"{us.NYC_CBSA}000"], "type": ["val"]}).to_feather(p)
    monkeypatch.setattr(us, "NYC_TRACT_SPLIT_PATH", p)
    rows = [{"GEOID": f"{us.NYC_CBSA}00{i}", "cbsa_code": us.NYC_CBSA,
             "geometry": box(i * 1000.0, 0, i * 1000.0 + 1000.0, 1000.0),
             "Rel_Score_2020": float(i)} for i in range(4)]
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{us.METRIC_EPSG}")
    ov = us.build_tract_overrides(gdf)
    assert ov[f"{us.NYC_CBSA}000"] == "val_within_nyc"   # borough split wins
    assert ov[f"{us.NYC_CBSA}003"] == "train"            # greater-metro fill


def test_carve_skips_absent_city():
    import geopandas as gpd
    c1 = _grid_city("16980", x0=0.0, n=6)
    with pytest.warns(UserWarning):
        ov = us.carve_test_city_val(c1, val_cluster_cbsas=("16980", "99999"),
                                    eval_fraction=0.2, seed=1)
    assert all(g.startswith("16980") for g in ov)


# ── resolve_building_types ───────────────────────────────────────────────────

def test_resolve_building_types_precedence():
    # buildings across: train city (with holdout yr), val city, test city,
    # NYC (excluded from city map, typed via override), and a val cluster tract.
    df = pd.DataFrame({
        "cbsa_code": ["100", "100", "200", "300", "35620", "35620", "300"],
        "GEOID":     ["100a", "100a", "200a", "300a", "nyc_tr", "nyc_val", "300vAl"],
        "year":      [2018,   2016,   2018,   2018,   2018,      2018,      2018],
    })
    city_split_map = {"100": "train", "200": "val", "300": "test"}  # NYC absent
    holdout_map = {100: 2016}
    tract_overrides = {
        "nyc_tr": "train",              # NYC within-city train
        "nyc_val": "val_within_nyc",    # NYC within-city val
        "300vAl": "val_within_chicago", # clustered val carved from a test city
    }
    out = us.resolve_building_types(
        df["cbsa_code"], df["GEOID"], df["year"],
        city_split_map, holdout_map, tract_overrides,
    )
    assert out.tolist() == [
        "train",             # train city, non-holdout year
        "val_temporal",      # train city, holdout year 2016
        "val_cities",        # val city
        "test",              # test city
        "train",             # NYC train tract (no holdout — NYC not in holdout_map)
        "val_within_nyc",    # NYC val tract
        "val_within_chicago",# override beats the test-city base label
    ]


def test_resolve_unassigned_for_unknown_city():
    df = pd.DataFrame({"cbsa_code": ["999"], "GEOID": ["x"], "year": [2018]})
    out = us.resolve_building_types(
        df["cbsa_code"], df["GEOID"], df["year"], {}, {}, {})
    assert out.iloc[0] == "unassigned"


def test_resolve_nyc_never_gets_temporal_holdout():
    # NYC train tract in a year that IS some other city's holdout must stay train.
    df = pd.DataFrame({"cbsa_code": ["35620"], "GEOID": ["nyc"], "year": [2016]})
    out = us.resolve_building_types(
        df["cbsa_code"], df["GEOID"], df["year"],
        {}, {100: 2016}, {"nyc": "train"})
    assert out.iloc[0] == "train"


def test_nyc_split_port_deterministic_and_labels():
    """The ported NYC split (src.data.nyc_split) yields the four labels and is
    reproducible given the seed — synthetic grid so no NYC data needed."""
    import geopandas as gpd
    from shapely.geometry import box
    from src.data import nyc_split as ns

    n = 20
    rows = []
    for i in range(n):
        for j in range(n):
            rows.append({"GEOID": f"36061{i:02d}{j:02d}",
                         "geometry": box(i * 1000.0, j * 1000.0,
                                         i * 1000.0 + 1000.0, j * 1000.0 + 1000.0),
                         "income_quintile": (i + j) % 5})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{ns.NYC_EVAL_EPSG}")

    a = ns.build_nyc_split(gdf.copy(), save=False)
    b = ns.build_nyc_split(gdf.copy(), save=False)
    assert set(a["type"]) <= {"train", "val", "test", "dead_zone"}
    assert {"train", "val", "test"} <= set(a["type"])          # all present
    assert a["type"].tolist() == b["type"].tolist()            # deterministic
    # test and val holdouts are disjoint
    assert not ((a["type"] == "test") & (b["type"] == "val")).any()


def test_resolve_lazy_years_none_skips_temporal():
    # years=None: holdout-year rows stay 'train' (val_temporal materialized later).
    df = pd.DataFrame({"cbsa_code": ["100", "100"], "GEOID": ["a", "a"]})
    out = us.resolve_building_types(
        df["cbsa_code"], df["GEOID"], None,
        {"100": "train"}, {100: 2016}, {})
    assert out.tolist() == ["train", "train"]


# ── integration: build_dataset.assign_buildings_by_city rewiring ─────────────

def test_assign_buildings_by_city_routes_overrides(tmp_path, monkeypatch):
    """The rewired flat-path assignment honours NYC + clustered-val overrides."""
    from src import build_dataset as bd
    # redirect the building_splits.feather side-effect write to a writable dir
    monkeypatch.setattr(bd, "PROCESSED_DATA_DIR", tmp_path)

    buildings = pd.DataFrame({
        "building_id": range(7),
        "cbsa_code": ["100", "100", "200", "300", "35620", "35620", "300"],
        "GEOID":     ["100a", "100a", "200a", "300a", "nyc_tr", "nyc_val", "300v"],
        "year":      [2018,   2016,   2018,   2018,   2018,      2018,      2018],
    })
    city_split_df = pd.DataFrame({
        "cbsa_code": ["100", "200", "300"],
        "cbsa_title": ["train city", "val city", "test city"],
        "split": ["train", "val", "test"],
        "holdout_year": [2016.0, np.nan, np.nan],
    })
    tract_overrides = {"nyc_tr": "train", "nyc_val": "val_within_nyc",
                       "300v": "val_within_chicago"}

    train_mask, test_mask, val_masks, dead_zone_mask = bd.assign_buildings_by_city(
        buildings, city_split_df, tract_overrides
    )
    assert buildings["type"].tolist() == [
        "train", "val_temporal", "val_cities", "test",
        "train", "val_within_nyc", "val_within_chicago",
    ]
    assert val_masks["val_within_nyc"].sum() == 1
    assert val_masks["val_within_chicago"].sum() == 1
    assert test_mask.sum() == 1                      # 300v got pulled to val
    assert not dead_zone_mask.any()
