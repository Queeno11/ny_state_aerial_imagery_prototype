"""Tests for the whole-city (CBSA) split machinery (#28) — pure synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.data import cbsa_brackets as cb


def _pop_df(pops: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "cbsa_code": list(pops),
        "cbsa_title": [f"City {c}" for c in pops],
        "population": list(pops.values()),
    })


def _universe(n_cbsa=12):
    """12 CBSAs: 4 mega-sized, 3 large, 3 medium, 2 small; varied tract counts."""
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
    keys = list(pops)[:n_cbsa]
    pop_df = _pop_df({k: pops[k] for k in keys})
    tract_counts = pd.Series({k: tracts[k] for k in keys})
    return tract_counts, pop_df


# ── assign_brackets ──────────────────────────────────────────────────────────

def test_assign_brackets_thresholds():
    _, pop_df = _universe()
    out = cb.assign_brackets(pop_df)
    b = dict(zip(out["cbsa_code"], out["bracket"]))
    assert [b[c] for c in ("10010", "10020", "10030", "10040")] == ["mega"] * 4
    assert b["20030"] == "large"      # 1.2M > 1M
    assert b["30010"] == "medium"     # 950K in [750K, 1M]
    assert b["30030"] == "medium"     # 760K boundary side
    assert b["40010"] == "small"      # 700K < 750K
    assert b["40020"] == "small"


def test_no_mega_bracket_below_min_universe():
    pop_df = _pop_df({"1": 9_000_000, "2": 5_000_000, "3": 800_000, "4": 600_000})
    out = cb.assign_brackets(pop_df)
    assert "mega" not in set(out["bracket"])
    assert set(out.loc[out["population"] > cb.LARGE_MIN_POP, "bracket"]) == {"large"}


# ── build_city_split ─────────────────────────────────────────────────────────

def test_split_deterministic_given_seed():
    tract_counts, pop_df = _universe()
    a = cb.build_city_split(tract_counts, pop_df, seed=cb.SPLIT_SEED)
    b = cb.build_city_split(tract_counts, pop_df, seed=cb.SPLIT_SEED)
    pd.testing.assert_frame_equal(a, b)


def test_mega_bracket_is_2_1_1():
    tract_counts, pop_df = _universe()
    out = cb.build_city_split(tract_counts, pop_df)
    mega = out[out["bracket"] == "mega"]
    assert mega["split"].value_counts().to_dict() == {"train": 2, "val": 1, "test": 1}


def test_every_split_nonempty_and_universe_covered():
    tract_counts, pop_df = _universe()
    out = cb.build_city_split(tract_counts, pop_df)
    assert set(out["split"]) == {"train", "val", "test"}
    assert set(out["cbsa_code"]) == set(tract_counts.index.astype(str))
    assert out["split"].notna().all()


def test_greedy_tract_shares_train_is_largest():
    tract_counts, pop_df = _universe()
    out = cb.build_city_split(tract_counts, pop_df)
    shares = out.groupby("split")["n_tracts"].sum() / out["n_tracts"].sum()
    assert shares["train"] == shares.max()
    # coarse sanity: nothing wildly off given whole cities as atoms
    assert 0.30 <= shares["train"] <= 0.75


def test_two_city_bracket_goes_train_test():
    tract_counts, pop_df = _universe()
    out = cb.build_city_split(tract_counts, pop_df)
    small = out[out["bracket"] == "small"]
    assert set(small["split"]) == {"train", "test"}
    # larger of the two -> train
    assert small.sort_values("n_tracts", ascending=False)["split"].iloc[0] == "train"


def test_guard_raises_below_3_cbsas():
    pop_df = _pop_df({"1": 9_000_000, "2": 5_000_000})
    with pytest.raises(ValueError, match=">= 3 CBSAs"):
        cb.build_city_split(pd.Series({"1": 100, "2": 50}), pop_df)


def test_small_universe_backfills_val_and_test():
    pop_df = _pop_df({"1": 2_000_000, "2": 1_500_000, "3": 1_200_000})
    out = cb.build_city_split(pd.Series({"1": 300, "2": 200, "3": 100}), pop_df)
    assert set(out["split"]) == {"train", "val", "test"}


# ── holdout years ────────────────────────────────────────────────────────────

def _coverage(state="NewYork", first=2011, last=2021):
    return pd.DataFrame({
        "state": [state] * (last - first + 1),
        "year": list(range(first, last + 1)),
        "n_items": [5] * (last - first + 1),
    })


def test_pick_holdout_year_from_coverage():
    years = list(range(2010, 2025, 2))
    # coverage 2011-2021 -> mid 2016 -> exact imagery year
    assert cb.pick_holdout_year("NewYork", years, _coverage()) == 2016
    # coverage 2016-2022 -> mid 2019 -> 2018 vs 2020 tie -> earlier
    assert cb.pick_holdout_year("NewYork", years, _coverage(first=2016, last=2022)) == 2018


def test_pick_holdout_year_fallback_middle_tie_earlier():
    years = list(range(2010, 2025, 2))   # mid 2017: 2016/2018 tie -> 2016
    assert cb.pick_holdout_year("Nowhere", years, None) == 2016
    assert cb.pick_holdout_year(None, years, _coverage()) == 2016


def test_attach_and_map_holdout_years():
    tract_counts, pop_df = _universe()
    split_df = cb.build_city_split(tract_counts, pop_df)
    state_map = {c: "NewYork" for c in split_df["cbsa_code"]}
    out = cb.attach_holdout_years(split_df, state_map, list(range(2010, 2025, 2)),
                                  _coverage())
    train = out[out["split"] == "train"]
    assert (train["holdout_year"] == 2016).all()
    assert out.loc[out["split"] != "train", "holdout_year"].isna().all()
    hmap = cb.holdout_year_map(out)
    assert set(hmap) == {int(c) for c in train["cbsa_code"]}
    assert all(v == 2016 for v in hmap.values())


# ── populations fallback + persistence ───────────────────────────────────────

def test_cbsa_populations_panel_fallback(monkeypatch):
    from src.data import process_acs

    def boom(*a, **k):
        raise FileNotFoundError("no ACS store")

    monkeypatch.setattr(process_acs, "get_large_metros", boom)
    panel = pd.DataFrame({
        "cbsa_code": ["1", "1", "2"],
        f"total_population_{process_acs.BASE_YEAR}": [600_000, 500_000, 900_000],
    })
    with pytest.warns(UserWarning, match="falling back"):
        pop = cb.cbsa_populations(panel=panel)
    assert dict(zip(pop["cbsa_code"], pop["population"])) == {"1": 1_100_000, "2": 900_000}


def test_save_and_load_city_split(tmp_path):
    tract_counts, pop_df = _universe()
    split_df = cb.build_city_split(tract_counts, pop_df)
    path = tmp_path / "cbsa_splits.feather"
    cb.save_city_split(split_df, path=path)
    loaded = cb.load_city_split(path=path)
    pd.testing.assert_frame_equal(split_df.reset_index(drop=True), loaded)


def test_top_cbsas_by_population():
    tract_counts, pop_df = _universe()
    split_df = cb.build_city_split(tract_counts, pop_df)
    top = cb.top_cbsas(split_df, n=3)
    assert top == ["10010", "10020", "10030"]
