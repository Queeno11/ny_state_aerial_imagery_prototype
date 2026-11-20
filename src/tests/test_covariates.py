"""Synthetic tests for src/data/covariates.py."""

import numpy as np
import pandas as pd
import pytest

from src.data import covariates as cv


def _acs(geoids, ba, pop25, age_counts, total):
    """ACS-like frame: ``ba`` bachelor's+ counts split over 022-025."""
    d = {"geoid": geoids, "B15003_001E": pop25, "total_population": total}
    for i, c in enumerate(cv._EDU_NUM):
        d[c] = [b / 4 for b in ba]
    for c in cv._AGE_NUM:
        d[c] = [a / len(cv._AGE_NUM) for a in age_counts]
    return pd.DataFrame(d)


def test_raw_covariates_shares_and_guards():
    acs = _acs(["36061000100", "36061000200", "6037000100"],
               ba=[50, 0, 10], pop25=[100, 0, 40], age_counts=[60, 30, 20], total=[200, 0, 80])
    r = cv.raw_covariates(acs).set_index("geoid")
    assert r.loc["36061000100", "edu_ba_share"] == pytest.approx(0.5)
    assert r.loc["36061000100", "age_25_54_share"] == pytest.approx(0.3)
    assert np.isnan(r.loc["36061000200", "edu_ba_share"])        # zero denominator
    assert "06037000100" in r.index                               # zero-padded


def test_crosswalk_2023_uses_its_own_id_and_flags_missing():
    raw = pd.DataFrame({"geoid": ["00000000021"], "edu_ba_share": [0.9], "age_25_54_share": [0.6]})
    panel = pd.DataFrame({"geoid_2023": ["00000000021", "00000000022"], "cbsa_code": ["1", "1"]})
    new = cv.crosswalk_to_panel(raw, panel, 2023)
    assert new["year"].eq(2023).all()
    assert new["edu_ba_share"].iloc[0] == pytest.approx(0.9)
    assert np.isnan(new["edu_ba_share"].iloc[1])


def test_crosswalk_maps_values_through_old_ids():
    raw = pd.DataFrame({"geoid": ["00000000011", "00000000012"],
                        "edu_ba_share": [0.1, 0.2], "age_25_54_share": [0.4, 0.5]})
    panel = pd.DataFrame({"geoid_2023": ["00000000021", "00000000022"], "cbsa_code": ["1", "1"],
                          "geoid_2015": ["00000000012", "00000000011"]})
    out = cv.crosswalk_to_panel(raw, panel, 2015).set_index("geoid_2023")
    assert out.loc["00000000021", "edu_ba_share"] == pytest.approx(0.2)
    assert out.loc["00000000022", "edu_ba_share"] == pytest.approx(0.1)


def test_standardize_within_groups_clips_and_fills():
    long = pd.DataFrame({"cbsa": ["1"] * 4 + ["2"] * 3, "year": [2020] * 7,
                         "edu_ba_share": [0.1, 0.2, 0.3, np.nan, 0.5, 0.5, 0.5],
                         "age_25_54_share": [0.3, 0.3, 0.3, 0.3, 0.1, 0.2, 0.9]})
    z = cv.standardize_within(long)
    g1 = z.iloc[:3]["edu_ba_share_z"]
    assert g1.mean() == pytest.approx(0.0, abs=1e-6) and g1.std() == pytest.approx(1.0, rel=1e-5)
    assert z["edu_ba_share_z"].iloc[3] == 0.0                     # missing -> group mean
    assert (z["edu_ba_share_z"].iloc[4:] == 0.0).all()            # zero spread -> 0
    assert (z["age_25_54_share_z"].iloc[:4] == 0.0).all()
    assert z["edu_ba_share_z"].dtype == np.float32
    big = cv.standardize_within(pd.DataFrame({
        "cbsa": ["1"] * 101, "year": [2020] * 101,
        "edu_ba_share": [0.0] * 100 + [100.0], "age_25_54_share": [0.0] * 101}))
    assert big["edu_ba_share_z"].max() == pytest.approx(cv.CLIP_SD)


def _table():
    return pd.DataFrame({
        "geoid_2023": ["36061000100", "36061000100", "36061000200"],
        "year": [2014, 2023, 2014],
        "edu_ba_share_z": [1.0, 2.0, -1.0], "age_25_54_share_z": [0.5, 0.6, -0.5]})


def test_lookup_ints_strings_year_snapping_and_unknowns():
    lk = cv.CovariateLookup(_table())
    assert lk.dim == 2
    out = lk.lookup([36061000100, "36061000100", "36061000200", "99999999999", "36061000100"],
                    [2014, 2023, 2010, 2014, 2030])
    np.testing.assert_allclose(out, [[1.0, 0.5], [2.0, 0.6], [-1.0, -0.5], [0, 0], [2.0, 0.6]])
    assert out.dtype == np.float32
    assert lk.coverage(["36061000100", "x"], [2014, 2014]) == 0.5
    # a year the tract lacks (2023 for ...200) snaps to the nearest table year
    # (2023) and finds no row -> zeros
    np.testing.assert_allclose(lk.lookup(["36061000200"], [2023]), [[0.0, 0.0]])


def test_lookup_subset_and_load_errors(tmp_path):
    lk = cv.CovariateLookup(_table(), names=("age_25_54_share",))
    np.testing.assert_allclose(lk.lookup(["36061000100"], [2014]), [[0.5]])
    with pytest.raises(FileNotFoundError):
        cv.load_lookup(path=tmp_path / "missing.parquet")
    with pytest.raises(ValueError):
        cv.load_lookup(names=("income",), path=tmp_path / "x.parquet")
    p = tmp_path / "t.parquet"
    _table().to_parquet(p)
    assert cv.load_lookup(path=p).dim == 2


def test_fill_uncovered_years_takes_nearest_covered_year():
    long = pd.DataFrame({"geoid_2023": ["a", "b"] * 3, "cbsa": ["1"] * 6,
                         "year": [2011, 2011, 2012, 2012, 2013, 2013],
                         "edu_ba_share": [np.nan, np.nan, 0.2, 0.4, 0.3, 0.5],
                         "age_25_54_share": [0.1, 0.2, 0.3, 0.4, 0.5, np.nan]})
    out = cv.fill_uncovered_years(long)
    np.testing.assert_allclose(out.loc[out.year == 2011, "edu_ba_share"], [0.2, 0.4])
    # 50% coverage is enough: the 2013 age gap stays missing (-> group mean later)
    assert np.isnan(out.loc[(out.year == 2013) & (out.geoid_2023 == "b"), "age_25_54_share"]).all()
    np.testing.assert_allclose(out.loc[out.year == 2011, "age_25_54_share"], [0.1, 0.2])


def test_lookup_by_shard_hash():
    from src.data.tract_sampling import hash_geoids
    lk = cv.CovariateLookup(_table())
    h = hash_geoids(["36061000100", "36061000200", "99999999999"])
    np.testing.assert_allclose(lk.lookup(h, [2023, 2014, 2014], hashed=True),
                               [[2.0, 0.6], [-1.0, -0.5], [0.0, 0.0]])
    assert lk.coverage(h, [2014] * 3, hashed=True) == pytest.approx(2 / 3)
    # plain GEOID ints are NOT hashes: the hashed index must not match them
    assert lk.coverage([36061000100], [2014], hashed=True) == 0.0
