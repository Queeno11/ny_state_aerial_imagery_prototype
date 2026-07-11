"""Synthetic-data tests for the city-coding audit (src/diagnose_city_coding.py)."""

import numpy as np
import pandas as pd
import pytest

import src.diagnose_city_coding as dcc


def _make_df(rng, n_cities=4, n_per_city=300, years=(2020,), signal=0.9,
             noise=0.4, offsets=None):
    """Predictions = signal*label + city_offset + noise; labels z per city-year."""
    rows = []
    for year in years:
        for c in range(n_cities):
            z = rng.standard_normal(n_per_city)
            z = (z - z.mean()) / z.std()  # exact per-city z, like the real labels
            off = 0.0 if offsets is None else offsets[c]
            pred = signal * z + off + noise * rng.standard_normal(n_per_city)
            rows.append(pd.DataFrame({
                "predicted_value": pred, "Rel_Score": z,
                "cbsa_code": f"C{c}", "year": year,
            }))
    return pd.concat(rows, ignore_index=True)


# ── eta_squared ──────────────────────────────────────────────────────────────

def test_eta_squared_zero_when_group_means_equal():
    v = np.tile([1.0, 2.0, 3.0], 2)
    g = ["a"] * 3 + ["b"] * 3
    assert dcc.eta_squared(v, g) == pytest.approx(0.0, abs=1e-12)


def test_eta_squared_one_when_groups_disjoint_constants():
    v = [1.0] * 50 + [5.0] * 50
    g = ["a"] * 50 + ["b"] * 50
    assert dcc.eta_squared(v, g) == pytest.approx(1.0)


def test_eta_squared_constant_values_and_nans():
    assert dcc.eta_squared([2.0] * 10, ["a"] * 5 + ["b"] * 5) == 0.0
    v = [1.0, np.nan, 2.0, 3.0, np.nan, 4.0]
    g = ["a", "a", "a", "b", "b", "b"]
    assert np.isfinite(dcc.eta_squared(v, g))


def test_eta_squared_matches_analytic_two_group_case():
    # two groups, means 0 and 1, zero within variance -> eta2 = 1; add equal
    # within-variance 1 via +/-1 points -> eta2 = between/(between+within)
    v = [-1.0, 1.0, 0.0, 2.0]  # means 0 and 1, within SS = 2 per group
    g = ["a", "a", "b", "b"]
    between = 2 * (0.5) ** 2 * 2  # n_g * (mean_g - grand)^2 summed = 1
    total = between + 4
    assert dcc.eta_squared(v, g) == pytest.approx(between / total)


# ── scenario: clean model (no city coding) ───────────────────────────────────

def test_clean_model_passes_audit():
    rng = np.random.default_rng(825)
    df = _make_df(rng)
    res = dcc.diagnose(df)
    a = res["aggregate"]
    assert a["eta2_pred"] < 0.02
    assert a["eta2_label"] == pytest.approx(0.0, abs=1e-10)
    assert abs(a["rho_demeaned"] - a["rho_pooled"]) < 0.01
    assert res["verdict"].startswith("NEGLIGIBLE")


# ── scenario: range coder (the feared failure mode) ──────────────────────────

def test_range_coder_is_flagged():
    rng = np.random.default_rng(825)
    # big city offsets, weak within-city signal: ranks fine within, scrambled across
    df = _make_df(rng, signal=0.3, noise=0.1, offsets=[-3.0, -1.0, 1.0, 3.0])
    res = dcc.diagnose(df)
    a = res["aggregate"]
    assert a["eta2_pred"] > 0.8
    assert a["rho_within"] > 0.9
    assert a["rho_demeaned"] > a["rho_pooled"] + 0.2  # demeaning recovers the ranking
    assert res["verdict"].startswith("SUBSTANTIAL")


def test_mild_offsets_get_mild_verdict():
    rng = np.random.default_rng(825)
    df = _make_df(rng, offsets=[-0.3, -0.1, 0.1, 0.3])
    res = dcc.diagnose(df)
    assert res["verdict"].startswith("MILD")


# ── mechanics ────────────────────────────────────────────────────────────────

def test_min_n_excludes_small_cities_from_within_rho_only():
    rng = np.random.default_rng(825)
    df = _make_df(rng, n_cities=2, n_per_city=100)
    tiny = _make_df(rng, n_cities=1, n_per_city=5)
    tiny["cbsa_code"] = "TINY"
    res = dcc.diagnose(pd.concat([df, tiny], ignore_index=True), min_n=20)
    cs = res["city_stats"]
    assert np.isnan(cs.loc[cs["cbsa_code"] == "TINY", "rho"].iloc[0])
    assert res["aggregate"]["n"] == len(df) + 5  # still counted in pooled stats


def test_per_year_grouping_isolates_yearly_offsets():
    rng = np.random.default_rng(825)
    # same city gets opposite offsets in two years: pooled-over-years eta2 would
    # hide it (offsets cancel), per-year must not
    d1 = _make_df(rng, n_cities=2, years=(2020,), offsets=[-1.5, 1.5])
    d2 = _make_df(rng, n_cities=2, years=(2022,), offsets=[1.5, -1.5])
    res = dcc.diagnose(pd.concat([d1, d2], ignore_index=True))
    for r in res["years"].values():
        assert r["eta2_pred"] > 0.4
    assert res["aggregate"]["eta2_pred"] > 0.4


def test_diagnose_raises_on_empty():
    df = pd.DataFrame({"predicted_value": [np.nan], "Rel_Score": [1.0],
                       "cbsa_code": ["C0"], "year": [2020]})
    with pytest.raises(ValueError):
        dcc.diagnose(df)


# ── input handling ───────────────────────────────────────────────────────────

def test_attach_cbsa_maps_county_prefix_and_drops_unmapped(capsys):
    xw = pd.DataFrame({"county_fips": ["36005"], "cbsa_code": ["35620"]})
    df = pd.DataFrame({"GEOID": ["36005006400", "06037123456"], "predicted_value": [1.0, 2.0]})
    out = dcc.attach_cbsa(df, xw)
    assert list(out["cbsa_code"]) == ["35620"]
    assert "1/2" in capsys.readouterr().out


def test_load_predictions_csv_split_filter_and_crosswalk(tmp_path):
    xw_path = tmp_path / "xw.csv"
    pd.DataFrame({"county_fips": ["36005", "06037"], "cbsa_code": ["35620", "31080"],
                  "cbsa_title": ["NY", "LA"]}).to_csv(xw_path, index=False)
    pred_path = tmp_path / "2020_predictions.csv"
    pd.DataFrame({
        "Rel_Score": [0.1, -0.2, 0.3], "predicted_value": [0.2, -0.1, 0.4],
        "building_id": [1, 2, 3], "GEOID": ["36005006400", "06037123456", "36005006500"],
        "year": [2020] * 3, "type": ["test", "test", "train"],
    }).to_csv(pred_path, index=False)
    df = dcc.load_predictions([pred_path], split="test", crosswalk_path=xw_path)
    assert len(df) == 2 and set(df["cbsa_code"]) == {"35620", "31080"}
    with pytest.raises(ValueError):
        dcc.load_predictions([pred_path], split="nope", crosswalk_path=xw_path)


def test_cli_end_to_end(tmp_path, capsys):
    rng = np.random.default_rng(825)
    df = _make_df(rng)
    p = tmp_path / "preds.csv"
    df.to_csv(p, index=False)  # already has cbsa_code -> no crosswalk needed
    assert dcc.main([str(p)]) == 0
    out = capsys.readouterr().out
    assert "VERDICT: NEGLIGIBLE" in out and "eta2_pred" in out
