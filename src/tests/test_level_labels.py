"""Unit tests for the L2 baseline's level labels (src/data/level_labels.py) and
their wiring into the split (build_dataset.apply_label_target,
LazyPairTable.with_labels). Synthetic frames only — no panel, no imagery.
"""
import json

import numpy as np
import pandas as pd
import pytest

import src.build_dataset as bd  # first: geo_utils <-> build_dataset import cycle
from src.data import level_labels as ll
from src.data.pair_table import LazyPairTable, FLAT_COLUMNS

YEARS = [2014, 2016, 2018]
TRAIN, VAL, TEST = "10001", "20002", "30003"


# ─── synthetic world ─────────────────────────────────────────────────────────
# Three cities with very different wealth LEVELS (train rich, test poor) and
# two tracts each; dollars grow 10%/yr nominal so the deflator matters.

def _usd(cbsa, tract_i, year):
    base = {TRAIN: 400_000.0, VAL: 200_000.0, TEST: 100_000.0}[cbsa]
    return base * (1 + 0.5 * tract_i) * (1.10 ** (year - 2014))


def _panel():
    rows = []
    for cbsa in (TRAIN, VAL, TEST):
        for i in range(2):
            row = {"geoid_2024": f"{cbsa}{i}", "cbsa_code": int(cbsa)}
            for y in YEARS:
                row[f"W2_i_r5pct_{y}"] = _usd(cbsa, i, y)
            rows.append(row)
    return pd.DataFrame(rows)


def _labels():
    """(tract, imagery-year) z-score labels — values irrelevant, layout real."""
    return pd.DataFrame([
        {"GEOID": f"{c}{i}", "year": y, "Rel_Score": np.float32(i - 0.5),
         "Valid_Structural_Change": np.int8(0), "score_bin": np.int8(i)}
        for c in (TRAIN, VAL, TEST) for i in range(2) for y in YEARS
    ])


def _buildings():
    return pd.DataFrame([
        {"building_id": f"b{c}{i}{k}", "GEOID": f"{c}{i}", "cbsa_code": c,
         "centroid_x": 1000.0 * k, "centroid_y": 0.0, "dist_to_center": 1.0}
        for c in (TRAIN, VAL, TEST) for i in range(2) for k in range(2)
    ])


def _city_split(holdout_year=2016):
    return pd.DataFrame({"cbsa_code": [int(TRAIN), int(VAL), int(TEST)],
                         "split": ["train", "val", "test"],
                         "holdout_year": [holdout_year, np.nan, np.nan]})


def _wealth_long():
    return ll.panel_log_real_wealth(_panel(), "geoid_2024", "W2_r5", YEARS)


def _acs_of(y):  # identity vintage map for the synthetic world
    return y


# ─── deflation / logs ────────────────────────────────────────────────────────

def test_deflate_to_base_year_and_unknown_year_raises():
    assert ll.deflate(100.0, 2023) == pytest.approx(100.0)
    assert ll.deflate(100.0, 2016) == pytest.approx(100.0 * 304.702 / 240.007)
    with pytest.raises(KeyError):
        ll.deflate(100.0, 1999)


def test_log_real_dollars_nonpositive_is_nan():
    out = ll.log_real_dollars(np.array([0.0, -5.0, np.nan, 304.702]), 2023)
    assert np.isnan(out[:3]).all()
    assert out[3] == pytest.approx(np.log(304.702))


def test_wealth_dollar_column_matches_indicator_tokens():
    assert ll.wealth_dollar_column("W2_r5", 2016) == "W2_i_r5pct_2016"
    assert ll.wealth_dollar_column("inc", 2016) == "per_capita_income_usd_2016"


def test_cbsa_key_normalizes_types():
    assert ll.cbsa_key(35620) == ll.cbsa_key("35620") == ll.cbsa_key(35620.0) == "35620"


def test_panel_log_real_wealth_long_layout_and_missing_column_raises():
    w = _wealth_long()
    assert set(w.columns) == {"GEOID", "cbsa_code", "acs_year", "log_real"}
    assert len(w) == 6 * len(YEARS)
    r = w[(w["GEOID"] == f"{TRAIN}0") & (w["acs_year"] == 2016)]["log_real"].iloc[0]
    assert r == pytest.approx(np.log(ll.deflate(_usd(TRAIN, 0, 2016), 2016)))
    with pytest.raises(KeyError):
        ll.panel_log_real_wealth(_panel(), "geoid_2024", "W2_r5", [2014, 2020])


# ─── attach / standardize ────────────────────────────────────────────────────

def test_attach_log_level_uses_the_vintage_map_and_keeps_order():
    labels = _labels()
    shifted = ll.attach_log_level(labels, _wealth_long(), lambda y: 2014)
    assert list(shifted["GEOID"]) == list(labels["GEOID"])
    # every imagery year reads the 2014 vintage
    g = shifted[shifted["GEOID"] == f"{TRAIN}0"]["log_real"]
    assert g.nunique() == 1


def test_standardizer_uses_train_cities_only_and_drops_holdout_year():
    frame = ll.attach_log_level(_labels(), _wealth_long(), _acs_of)
    stats = ll.fit_level_standardizer(frame, [TRAIN], {int(TRAIN): 2016})
    use = frame[(frame["cbsa_code"] == TRAIN) & (frame["year"] != 2016)]["log_real"]
    assert stats["mu"] == pytest.approx(use.mean())
    assert stats["sd"] == pytest.approx(use.std(ddof=1))
    assert stats["n_rows"] == len(use) == 4 and stats["n_cbsas"] == 1


def test_standardizer_raises_without_train_rows():
    frame = ll.attach_log_level(_labels(), _wealth_long(), _acs_of)
    with pytest.raises(ValueError):
        ll.fit_level_standardizer(frame, ["99999"])


def test_standardized_labels_keep_cross_city_level_gap():
    new, stats = ll.standardized_level_labels(_labels(), _wealth_long(), _acs_of,
                                              [TRAIN], {int(TRAIN): 2016})
    assert new["Rel_Score"].dtype == np.float32
    # only Rel_Score changes
    pd.testing.assert_frame_equal(new.drop(columns="Rel_Score"),
                                  _labels().drop(columns="Rel_Score"))
    by_city = new.assign(c=new["GEOID"].str[:5]).groupby("c")["Rel_Score"].mean()
    # the poor test city sits well below the rich train city: the level gap
    # the per-CBSA z-score would have erased stays in the label
    assert by_city[TEST] < by_city[VAL] < by_city[TRAIN]
    expected = (np.log(ll.deflate(_usd(TEST, 1, 2018), 2018)) - stats["mu"]) / stats["sd"]
    got = new[(new["GEOID"] == f"{TEST}1") & (new["year"] == 2018)]["Rel_Score"].iloc[0]
    assert got == pytest.approx(expected, rel=1e-5)


def test_relabel_flat_maps_by_geoid_year_and_nans_missing():
    new, _ = ll.standardized_level_labels(_labels(), _wealth_long(), _acs_of, [TRAIN])
    flat = pd.DataFrame({"GEOID": [f"{VAL}0", f"{VAL}1", "unknown"],
                         "year": [2014, 2018, 2014], "Rel_Score": [9.0, 9.0, 9.0]})
    out = ll.relabel_flat(flat, new)
    want0 = new[(new["GEOID"] == f"{VAL}0") & (new["year"] == 2014)]["Rel_Score"].iloc[0]
    assert out["Rel_Score"].iloc[0] == pytest.approx(want0)
    assert np.isnan(out["Rel_Score"].iloc[2])
    assert flat["Rel_Score"].tolist() == [9.0, 9.0, 9.0]  # input untouched
    assert ll.relabel_flat(pd.DataFrame(), new).empty


# ─── LazyPairTable.with_labels + apply_label_target ──────────────────────────

def _table(split_type="train", holdout_map=None):
    return LazyPairTable(_buildings(), _labels(), YEARS, 100.0,
                         split_type=split_type, holdout_map=holdout_map)


def test_with_labels_keeps_order_and_holdout_masking():
    t = _table(holdout_map={TRAIN: 2016})
    new, _ = ll.standardized_level_labels(_labels(), _wealth_long(), _acs_of, [TRAIN])
    t2 = t.with_labels(new)
    a, b = t.materialize(0, len(t)), t2.materialize(0, len(t2))
    assert list(a["building_id"]) == list(b["building_id"])
    on_holdout = (b["cbsa_code"] == TRAIN) & (b["year"] == 2016)
    assert b.loc[on_holdout, "Rel_Score"].isna().all()
    assert not np.allclose(a.loc[~on_holdout, "Rel_Score"], b.loc[~on_holdout, "Rel_Score"])


def test_apply_label_target_rel_score_is_identity():
    t = _table()
    vals = {"val_cities": pd.DataFrame(columns=FLAT_COLUMNS)}
    out = bd.apply_label_target(t, vals, t, "rel_score", "W2_r5", YEARS, "unused")
    assert out[0] is t and out[1] is vals and out[2] is t


def test_apply_label_target_rejects_unknown_and_flat_frames():
    with pytest.raises(ValueError):
        bd.apply_label_target(_table(), {}, _table(), "levels", "W2_r5", YEARS, "x")
    flat = _table().materialize(0, 4)
    with pytest.raises(NotImplementedError):
        bd.apply_label_target(flat, {}, flat, "log_level", "W2_r5", YEARS, "x")


def test_apply_label_target_log_level_swaps_every_split_and_records_scale(tmp_path, monkeypatch):
    monkeypatch.setattr(bd, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(bd, "get_closest_acs_year", _acs_of)
    train = _table("train", holdout_map={TRAIN: 2016})
    test = _table("test")
    val_flat = test.materialize(0, len(test)).query("cbsa_code == @VAL").reset_index(drop=True)
    tr, vals, te = bd.apply_label_target(
        train, {"val_cities": val_flat}, test, "log_level", "W2_r5", YEARS, "l2run",
        city_split=_city_split(), wealth_long=_wealth_long())

    stats = json.loads((tmp_path / "l2run" / "level_label_stats.json").read_text())
    assert stats == bd.load_level_label_stats("l2run")
    assert stats["n_cbsas"] == 1 and stats["n_rows"] == 4  # train city, holdout dropped
    assert stats["label_target"] == "log_level"

    m_te = te.materialize(0, len(te))
    row = m_te[(m_te["GEOID"] == f"{TEST}0") & (m_te["year"] == 2014)].iloc[0]
    want = (np.log(ll.deflate(_usd(TEST, 0, 2014), 2014)) - stats["mu"]) / stats["sd"]
    assert row["Rel_Score"] == pytest.approx(want, rel=1e-5)
    v = vals["val_cities"]
    assert len(v) == len(val_flat) and (v["Rel_Score"] != val_flat["Rel_Score"]).all()
    # train-city holdout year still masked after the swap
    m_tr = tr.materialize(0, len(tr))
    assert m_tr[(m_tr["cbsa_code"] == TRAIN) & (m_tr["year"] == 2016)]["Rel_Score"].isna().all()


def test_load_level_label_stats_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(bd, "RESULTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        bd.load_level_label_stats("nope")
