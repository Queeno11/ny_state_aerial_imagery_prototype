"""Tests for LazyPairTable: the implicit building×year pair table (US-scale path).

The reference implementation throughout is the LEGACY flat construction: unroll
buildings × years row-by-row and merge tract-year labels. LazyPairTable must
reproduce it slice-for-slice.
"""

import numpy as np
import pandas as pd
import pytest

import src.geo_utils as geo_utils
from src.data.pair_table import (
    FLAT_COLUMNS, LazyPairTable, weighted_qcut,
)

YEARS = [2012, 2014, 2018]
TAU = 100.0


def _synthetic_buildings(n_tracts=6, buildings_per_tract=5, seed=7):
    """Buildings across n_tracts tracts in 3 CBSAs (2 tracts each)."""
    rng = np.random.default_rng(seed)
    rows = []
    bid = 1
    for t in range(n_tracts):
        geoid = f"36{t:09d}"
        cbsa = str(10000 + (t // 2))
        for _ in range(buildings_per_tract):
            rows.append({
                "building_id": bid,
                "GEOID": geoid,
                "cbsa_code": cbsa,
                "centroid_x": float(rng.uniform(0, 1e6)),
                "centroid_y": float(rng.uniform(0, 1e6)),
                "dist_to_center": float(rng.uniform(0, 30)),
            })
            bid += 1
    df = pd.DataFrame(rows)
    df["GEOID"] = df["GEOID"].astype("category")
    df["cbsa_code"] = df["cbsa_code"].astype("category")
    df["centroid_x"] = df["centroid_x"].astype("float32")
    df["centroid_y"] = df["centroid_y"].astype("float32")
    df["dist_to_center"] = df["dist_to_center"].astype("float32")
    return df


def _synthetic_labels(buildings, years=YEARS, nan_tract_year=None, seed=11):
    rng = np.random.default_rng(seed)
    geoids = sorted(buildings["GEOID"].astype(str).unique())
    rows = []
    for yr in years:
        for i, g in enumerate(geoids):
            score = float(rng.normal())
            if nan_tract_year is not None and (g, yr) == nan_tract_year:
                score = np.nan
            rows.append({
                "GEOID": g, "year": yr, "Rel_Score": np.float32(score),
                "Valid_Structural_Change": np.int8(i % 2),
                "score_bin": np.int8(i % 5),
            })
    return pd.DataFrame(rows)


def _brute_force_flat(buildings, labels, years, tau_meters, split_type="all",
                      holdout_map=None):
    """Legacy-style unroll: one row per (building, year), labels merged in."""
    tau_units = geo_utils.meters_to_projected_units(tau_meters, geo_utils.METRIC_EPSG)
    frames = []
    for _, b in buildings.iterrows():
        for yr in sorted(years):
            lab = labels[(labels["GEOID"] == str(b["GEOID"])) & (labels["year"] == yr)]
            rel = float(lab["Rel_Score"].iloc[0]) if len(lab) else np.nan
            vsc = int(lab["Valid_Structural_Change"].iloc[0]) if len(lab) else 0
            sbin = int(lab["score_bin"].iloc[0]) if len(lab) else -1
            if holdout_map and int(holdout_map.get(str(b["cbsa_code"]), -1)) == yr:
                rel = np.nan
            frames.append({
                "building_id": b["building_id"], "GEOID": str(b["GEOID"]),
                "cbsa_code": str(b["cbsa_code"]), "year": yr, "type": split_type,
                "Rel_Score": rel, "Valid_Structural_Change": vsc, "score_bin": sbin,
                "dataset": "NAIP",
                "bbox_minx": b["centroid_x"] - tau_units,
                "bbox_miny": b["centroid_y"] - tau_units,
                "bbox_maxx": b["centroid_x"] + tau_units,
                "bbox_maxy": b["centroid_y"] + tau_units,
                "row_start": 0, "row_stop": 0, "col_start": 0, "col_stop": 0,
                "dist_to_center": b["dist_to_center"],
                "centroid_x": b["centroid_x"], "centroid_y": b["centroid_y"],
            })
    return pd.DataFrame(frames)[FLAT_COLUMNS]


def _assert_flat_equal(got, expected):
    assert list(got.columns) == FLAT_COLUMNS
    assert len(got) == len(expected)
    for col in FLAT_COLUMNS:
        g, e = got[col].to_numpy(), expected[col].to_numpy()
        if got[col].dtype.kind in "fc":
            np.testing.assert_allclose(g.astype("float64"), e.astype("float64"),
                                       rtol=1e-5, equal_nan=True, err_msg=col)
        else:
            assert (g.astype(str) == e.astype(str)).all(), col


@pytest.fixture
def table():
    buildings = _synthetic_buildings()
    labels = _synthetic_labels(buildings)
    return LazyPairTable(buildings, labels, YEARS, TAU, split_type="all")


# --------------------------------------------------------------------------- #
# weighted_qcut                                                                #
# --------------------------------------------------------------------------- #
class TestWeightedQcut:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_matches_qcut_on_expanded_rows(self, seed):
        rng = np.random.default_rng(seed)
        values = rng.normal(size=40)
        weights = rng.integers(1, 50, size=40)
        expected = pd.qcut(np.repeat(values, weights), q=5, labels=False,
                           duplicates="drop")
        got = np.repeat(weighted_qcut(values, weights, q=5), weights)
        np.testing.assert_array_equal(got, expected)

    def test_ties_match_qcut(self):
        values = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
        weights = np.array([10, 5, 5, 10, 20])
        expected = pd.qcut(np.repeat(values, weights), q=5, labels=False,
                           duplicates="drop")
        got = np.repeat(weighted_qcut(values, weights, q=5), weights)
        np.testing.assert_array_equal(got, expected)

    def test_nan_gets_minus_one(self):
        out = weighted_qcut(np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0]),
                            np.array([3, 1, 1, 1, 1, 1]), q=5)
        assert out[0] == -1
        assert (out[1:] >= 0).all()

    def test_constant_values_single_bin(self):
        out = weighted_qcut(np.array([2.0, 2.0, 2.0]), np.array([1, 2, 3]), q=5)
        np.testing.assert_array_equal(out, [0, 0, 0])

    def test_rejects_nonpositive_weights(self):
        with pytest.raises(ValueError):
            weighted_qcut(np.array([1.0, 2.0]), np.array([1, 0]), q=5)


# --------------------------------------------------------------------------- #
# materialize                                                                  #
# --------------------------------------------------------------------------- #
class TestMaterialize:
    def test_full_table_matches_brute_force(self, table):
        expected = _brute_force_flat(table.buildings, table.labels, YEARS, TAU)
        got = table.materialize(0, len(table))
        _assert_flat_equal(got, expected)

    def test_building_major_year_minor_and_twins_adjacent(self, table):
        got = table.materialize(0, len(table))
        n_years = len(YEARS)
        bids = got["building_id"].to_numpy()
        yrs = got["year"].to_numpy()
        for k in range(0, len(got), n_years):
            assert len(set(bids[k:k + n_years])) == 1      # same building block
            np.testing.assert_array_equal(yrs[k:k + n_years], sorted(YEARS))

    def test_arbitrary_slice_matches_full(self, table):
        full = table.materialize(0, len(table))
        got = table.materialize(7, 29)
        _assert_flat_equal(got, full.iloc[7:29].reset_index(drop=True))

    def test_cyclic_wraparound(self, table):
        full = table.materialize(0, len(table))
        n = len(table)
        got = table.materialize(n - 5, n + 4)
        expected = pd.concat([full.iloc[n - 5:], full.iloc[:4]], ignore_index=True)
        _assert_flat_equal(got, expected)

    def test_missing_tract_year_label_is_nan(self):
        buildings = _synthetic_buildings()
        geoid0 = str(buildings["GEOID"].astype(str).iloc[0])
        labels = _synthetic_labels(buildings)
        labels = labels[~((labels["GEOID"] == geoid0) & (labels["year"] == YEARS[1]))]
        table = LazyPairTable(buildings, labels, YEARS, TAU)
        got = table.materialize(0, len(table))
        hole = got[(got["GEOID"] == geoid0) & (got["year"] == YEARS[1])]
        assert len(hole) > 0 and hole["Rel_Score"].isna().all()
        rest = got[~((got["GEOID"] == geoid0) & (got["year"] == YEARS[1]))]
        assert rest["Rel_Score"].notna().all()

    def test_nan_label_value_propagates(self):
        buildings = _synthetic_buildings()
        geoid0 = str(buildings["GEOID"].astype(str).iloc[0])
        labels = _synthetic_labels(buildings, nan_tract_year=(geoid0, YEARS[0]))
        table = LazyPairTable(buildings, labels, YEARS, TAU)
        got = table.materialize(0, len(table))
        hole = got[(got["GEOID"] == geoid0) & (got["year"] == YEARS[0])]
        assert hole["Rel_Score"].isna().all()

    def test_holdout_year_pairs_get_nan_label(self):
        buildings = _synthetic_buildings()
        labels = _synthetic_labels(buildings)
        cbsas = sorted(buildings["cbsa_code"].astype(str).unique())
        holdout_map = {cbsas[0]: YEARS[0], cbsas[1]: YEARS[2]}
        table = LazyPairTable(buildings, labels, YEARS, TAU,
                              holdout_map=holdout_map, split_type="train")
        expected = _brute_force_flat(buildings, labels, YEARS, TAU,
                                     split_type="train", holdout_map=holdout_map)
        got = table.materialize(0, len(table))
        _assert_flat_equal(got, expected)
        on_holdout = got["cbsa_code"].map(holdout_map) == got["year"]
        assert on_holdout.any()
        assert got.loc[on_holdout, "Rel_Score"].isna().all()
        assert got.loc[~on_holdout, "Rel_Score"].notna().all()

    def test_str_keys_in_slices(self, table):
        got = table.materialize(0, 10)
        assert got["GEOID"].dtype == object and isinstance(got["GEOID"].iloc[0], str)
        assert got["cbsa_code"].dtype == object

    def test_empty_range_raises(self, table):
        with pytest.raises(ValueError):
            table.materialize(5, 5)


# --------------------------------------------------------------------------- #
# materialize_year / sampling / subset                                         #
# --------------------------------------------------------------------------- #
class TestOtherMaterializers:
    def test_materialize_year(self, table):
        full = table.materialize(0, len(table))
        got = table.materialize_year(YEARS[1])
        expected = (full[full["year"] == YEARS[1]]).reset_index(drop=True)
        assert len(got) == table.n_buildings
        # categorical keys are kept for the big frame; compare as str
        got = got.copy()
        got["GEOID"] = got["GEOID"].astype(str)
        got["cbsa_code"] = got["cbsa_code"].astype(str)
        _assert_flat_equal(got, expected)

    def test_materialize_unknown_year_is_empty(self, table):
        got = table.materialize_year(1999)
        assert len(got) == 0 and list(got.columns) == FLAT_COLUMNS

    def test_sample_buildings_per_tract(self, table):
        got = table.sample_buildings_per_tract(2, seed=825)
        per_tract = got.drop_duplicates("building_id").groupby("GEOID").size()
        assert (per_tract <= 2).all()
        # every kept building retains ALL years
        per_building = got.groupby("building_id")["year"].apply(
            lambda s: sorted(s) == sorted(YEARS))
        assert per_building.all()
        again = table.sample_buildings_per_tract(2, seed=825)
        pd.testing.assert_frame_equal(got, again)

    def test_sample_cross_sectional_one_year_per_city(self, table):
        got = table.sample_cross_sectional(years_per_city=1, seed=825)
        # one building per tract
        assert (got.groupby("GEOID")["building_id"].nunique() <= 1).all()
        # exactly one row per building (one year) -> tract coverage, not depth
        assert (got.groupby("building_id").size() == 1).all()
        # every tract of every city is present (full coverage within budget)
        for cbsa, grp in got.groupby("cbsa_code"):
            city_geoids = set(table.buildings.loc[
                table.buildings["cbsa_code"].astype(str) == cbsa, "GEOID"].astype(str))
            assert set(grp["GEOID"]) == city_geoids
            # a city's tracts all share ITS chosen year -> one dense cell
            assert grp["year"].nunique() == 1
        # labels attach at the chosen year
        assert got["Rel_Score"].notna().all()

    def test_sample_cross_sectional_years_vary_across_cities(self, table):
        # With one year per city drawn per-CBSA, different cities can pick
        # different vintages (deterministically) — not all pinned to one year.
        got = table.sample_cross_sectional(years_per_city=1, seed=825)
        year_by_city = got.groupby("cbsa_code")["year"].first()
        assert year_by_city.nunique() >= 2

    def test_sample_cross_sectional_two_years_dense_cells(self, table):
        got = table.sample_cross_sectional(years_per_city=2, seed=825)
        assert (got.groupby("building_id").size() == 2).all()
        for _cbsa, grp in got.groupby("cbsa_code"):
            assert grp["year"].nunique() == 2  # two dense cells per city

    def test_sample_cross_sectional_deterministic(self, table):
        a = table.sample_cross_sectional(years_per_city=1, seed=825)
        b = table.sample_cross_sectional(years_per_city=1, seed=825)
        pd.testing.assert_frame_equal(a, b)

    def test_sample_temporal_stability(self, table):
        got = table.sample_temporal_stability(n_buildings_per_cbsa=3, seed=825)
        # cap is per CITY, not per tract
        assert (got.groupby("cbsa_code")["building_id"].nunique() <= 3).all()
        # every kept building carries ALL years (temporal tracking)
        per_building = got.groupby("building_id")["year"].apply(
            lambda s: sorted(s) == sorted(YEARS))
        assert per_building.all()
        again = table.sample_temporal_stability(n_buildings_per_cbsa=3, seed=825)
        pd.testing.assert_frame_equal(got, again)

    def test_sample_pairs_deterministic(self, table):
        got = table.sample_pairs(10, seed=825)
        again = table.sample_pairs(10, seed=825)
        assert len(got) == 10
        pd.testing.assert_frame_equal(got, again)

    def test_materialize_holdout_years(self, table):
        cbsas = sorted(table.buildings["cbsa_code"].astype(str).unique())
        holdout_map = {cbsas[0]: YEARS[0], cbsas[1]: YEARS[1]}  # cbsas[2] has none
        got = table.materialize_holdout_years(holdout_map, n_buildings_per_tract=2,
                                              seed=825)
        assert set(got["cbsa_code"]) == {cbsas[0], cbsas[1]}
        assert (got["year"] == got["cbsa_code"].map(holdout_map)).all()
        per_tract = got.groupby("GEOID")["building_id"].nunique()
        assert (per_tract <= 2).all() and (per_tract > 0).all()
        assert got["Rel_Score"].notna().all()   # labels attach at the holdout year

    def test_subset_and_shuffle(self, table):
        mask = (table.buildings["cbsa_code"].astype(str)
                == sorted(table.buildings["cbsa_code"].astype(str).unique())[0]).to_numpy()
        sub = table.subset(mask, split_type="test", holdout_map={})
        assert sub.n_buildings == int(mask.sum())
        assert (sub.materialize(0, len(sub))["type"] == "test").all()

        shuffled = sub.shuffle_buildings(seed=825)
        assert set(shuffled.buildings["building_id"]) == set(sub.buildings["building_id"])
        again = sub.shuffle_buildings(seed=825)
        np.testing.assert_array_equal(shuffled.buildings["building_id"].to_numpy(),
                                      again.buildings["building_id"].to_numpy())
        # shuffling permutes buildings but keeps each building's years adjacent
        got = shuffled.materialize(0, len(shuffled))
        n_years = len(YEARS)
        bids = got["building_id"].to_numpy()
        for k in range(0, len(got), n_years):
            assert len(set(bids[k:k + n_years])) == 1

    def test_building_reference(self, table):
        ref = table.building_reference()
        assert list(ref.columns) == ["building_id", "GEOID", "cbsa_code"]
        assert len(ref) == table.n_buildings


# --------------------------------------------------------------------------- #
# unavailable (NAIP coverage-gap) masking                                     #
# --------------------------------------------------------------------------- #
class TestUnavailableMasking:
    def _table(self, unavailable):
        buildings = _synthetic_buildings()
        labels = _synthetic_labels(buildings)
        return (buildings, labels,
                LazyPairTable(buildings, labels, YEARS, TAU, unavailable=unavailable))

    def test_cbsa_gap_masks_whole_metro_all_years(self):
        buildings = _synthetic_buildings()
        cbsa0 = sorted(buildings["cbsa_code"].astype(str).unique())[0]
        unavail = pd.DataFrame([{"level": "cbsa", "key": cbsa0, "year": y}
                                for y in YEARS])
        table = LazyPairTable(buildings, _synthetic_labels(buildings), YEARS, TAU,
                              unavailable=unavail)
        got = table.materialize(0, len(table))
        masked = got["cbsa_code"] == cbsa0
        assert masked.any()
        assert got.loc[masked, "Rel_Score"].isna().all()
        assert got.loc[~masked, "Rel_Score"].notna().all()

    def test_tract_gap_masks_single_tract_single_year(self):
        buildings = _synthetic_buildings()
        geoid0 = sorted(buildings["GEOID"].astype(str).unique())[0]
        unavail = pd.DataFrame([{"level": "tract", "key": geoid0, "year": YEARS[1]}])
        table = LazyPairTable(buildings, _synthetic_labels(buildings), YEARS, TAU,
                              unavailable=unavail)
        got = table.materialize(0, len(table))
        hit = (got["GEOID"] == geoid0) & (got["year"] == YEARS[1])
        assert hit.any() and got.loc[hit, "Rel_Score"].isna().all()
        # other years of the same tract are untouched
        other = (got["GEOID"] == geoid0) & (got["year"] != YEARS[1])
        assert got.loc[other, "Rel_Score"].notna().all()

    def test_gap_applies_in_materialize_year(self):
        buildings = _synthetic_buildings()
        cbsa0 = sorted(buildings["cbsa_code"].astype(str).unique())[0]
        unavail = pd.DataFrame([{"level": "cbsa", "key": cbsa0, "year": YEARS[0]}])
        table = LazyPairTable(buildings, _synthetic_labels(buildings), YEARS, TAU,
                              unavailable=unavail)
        yr0 = table.materialize_year(YEARS[0])
        yr0_cbsa = yr0["cbsa_code"].astype(str)
        assert yr0.loc[yr0_cbsa == cbsa0, "Rel_Score"].isna().all()
        assert yr0.loc[yr0_cbsa != cbsa0, "Rel_Score"].notna().all()
        # a year NOT in the gap list is fully available
        yr1 = table.materialize_year(YEARS[1])
        assert yr1["Rel_Score"].notna().all()

    def test_unavailable_survives_subset_and_shuffle(self):
        buildings = _synthetic_buildings()
        cbsa0 = sorted(buildings["cbsa_code"].astype(str).unique())[0]
        unavail = pd.DataFrame([{"level": "cbsa", "key": cbsa0, "year": y}
                                for y in YEARS])
        table = LazyPairTable(buildings, _synthetic_labels(buildings), YEARS, TAU,
                              unavailable=unavail)
        # shuffle + subset (keep all) must both carry the unavailable table through
        derived = table.shuffle_buildings(seed=1).subset(
            np.arange(table.n_buildings))
        got = derived.materialize(0, len(derived))
        assert got.loc[got["cbsa_code"] == cbsa0, "Rel_Score"].isna().all()
        assert got.loc[got["cbsa_code"] != cbsa0, "Rel_Score"].notna().all()

    def test_none_unavailable_is_noop(self):
        buildings = _synthetic_buildings()
        table = LazyPairTable(buildings, _synthetic_labels(buildings), YEARS, TAU,
                              unavailable=None)
        assert table.materialize(0, len(table))["Rel_Score"].notna().all()

    def test_fully_unavailable_needs_all_years(self):
        buildings = _synthetic_buildings()
        cbsa0, cbsa1 = sorted(buildings["cbsa_code"].astype(str).unique())[:2]
        geoid0 = sorted(buildings["GEOID"].astype(str).unique())[0]
        rows = (
            [{"level": "cbsa", "key": cbsa0, "year": y} for y in YEARS]        # all years
            + [{"level": "cbsa", "key": cbsa1, "year": YEARS[0]}]              # one year only
            + [{"level": "tract", "key": geoid0, "year": y} for y in YEARS]    # all years
        )
        table = LazyPairTable(buildings, _synthetic_labels(buildings), YEARS, TAU,
                              unavailable=pd.DataFrame(rows))
        dead_cbsas, dead_tracts = table.fully_unavailable()
        assert dead_cbsas == {cbsa0}          # cbsa1 flagged only 1 year → stays
        assert dead_tracts == {geoid0}

    def test_fully_unavailable_empty_when_none(self):
        buildings = _synthetic_buildings()
        table = LazyPairTable(buildings, _synthetic_labels(buildings), YEARS, TAU,
                              unavailable=None)
        assert table.fully_unavailable() == (set(), set())

    def test_bad_unavailable_columns_raise(self):
        buildings = _synthetic_buildings()
        with pytest.raises(KeyError):
            LazyPairTable(buildings, _synthetic_labels(buildings), YEARS, TAU,
                          unavailable=pd.DataFrame({"region": ["x"], "year": [2010]}))

    def test_len_and_shape(self, table):
        assert len(table) == table.n_buildings * len(YEARS)
        assert table.shape == (len(table), len(FLAT_COLUMNS))
        assert not table.empty
