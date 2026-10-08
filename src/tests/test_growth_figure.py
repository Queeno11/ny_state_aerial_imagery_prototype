"""Unit tests for the long-difference growth figure (US mode).

Synthetic data only — the end-to-end tests use a minimal fake context (as in
test_main_figure.py) and pass the income/population panel in directly, so no
real results tree or ACS panel is needed.
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.evaluation as ev


# ─── _long_difference_pairs ─────────────────────────────────────────────────

def _obs(geoid, cbsa, years, label=None, pred=None):
    label = np.arange(len(years), dtype=float) if label is None else label
    pred = np.arange(len(years), dtype=float) * 2 if pred is None else pred
    return pd.DataFrame({"GEOID": geoid, "cbsa": cbsa, "year": years,
                         "label": label, "pred": pred})


def test_pairs_pick_latest_and_closest_to_ten_years_back():
    df = pd.concat([
        _obs("A", "1", [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]),
        _obs("B", "1", [2016, 2017, 2019, 2023]),   # 2013 target -> 2016 (7y)
    ], ignore_index=True)
    p = ev._long_difference_pairs(df).set_index("GEOID")
    assert (p.loc["A", "t0"], p.loc["A", "t1"], p.loc["A", "gap"]) == (2014, 2024, 10)
    assert (p.loc["B", "t0"], p.loc["B", "t1"], p.loc["B", "gap"]) == (2016, 2023, 7)


def test_pairs_tie_prefers_earlier_year():
    # t1 = 2020, target 2010: 2008 and 2012 are equidistant -> longer horizon.
    p = ev._long_difference_pairs(_obs("A", "1", [2008, 2012, 2020]))
    assert int(p["t0"].iloc[0]) == 2008


def test_pairs_values_and_differences_are_aligned():
    df = _obs("A", "7", [2014, 2018, 2024], label=[0.1, 0.5, 0.9], pred=[-1.0, 0.0, 2.0])
    p = ev._long_difference_pairs(df).iloc[0]
    assert p["cbsa"] == "7"
    assert p["label_0"] == pytest.approx(0.1) and p["label_1"] == pytest.approx(0.9)
    assert p["d_label"] == pytest.approx(0.8)
    assert p["d_pred"] == pytest.approx(3.0)


def test_pairs_drop_single_year_tracts():
    df = pd.concat([_obs("A", "1", [2016, 2022]), _obs("B", "1", [2020])], ignore_index=True)
    p = ev._long_difference_pairs(df)
    assert list(p["GEOID"]) == ["A"]


def test_pairs_reject_duplicate_tract_years():
    df = pd.concat([_obs("A", "1", [2016, 2022]), _obs("A", "1", [2022])], ignore_index=True)
    with pytest.raises(ValueError):
        ev._long_difference_pairs(df)


def test_pairs_custom_value_cols():
    df = _obs("A", "1", [2016, 2022])
    df["pred_z"] = [0.0, 1.5]
    p = ev._long_difference_pairs(df, value_cols=("label", "pred_z"))
    assert "d_pred_z" in p.columns and "d_pred" not in p.columns
    assert p["d_pred_z"].iloc[0] == pytest.approx(1.5)


# ─── _zscore_within ─────────────────────────────────────────────────────────

def test_zscore_within_standardises_each_group():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"cbsa": np.repeat(["a", "b"], 50), "year": 2020,
                       "pred": np.r_[rng.normal(5, 3, 50), rng.normal(-2, 0.1, 50)]})
    z = ev._zscore_within(df, "pred")
    for _, g in z.groupby(df["cbsa"]):
        assert g.mean() == pytest.approx(0, abs=1e-12)
        assert g.std() == pytest.approx(1)


def test_zscore_within_constant_group_is_nan_not_inf():
    df = pd.DataFrame({"cbsa": "a", "year": 2020, "pred": [1.0, 1.0, 1.0]})
    assert ev._zscore_within(df, "pred").isna().all()


# ─── R^2 / metrics ──────────────────────────────────────────────────────────

def test_r2_oos_reference_points():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert ev._r2_oos(y, y) == pytest.approx(1.0)
    assert ev._r2_oos(y, np.full(4, y.mean())) == pytest.approx(0.0)
    # Predicting zero change for a series with a non-zero mean is worse than
    # the mean -> negative, as in Khachiyan et al.'s 2007-2017 income cell.
    assert ev._r2_oos(y, np.zeros(4)) < 0


def test_r2_ols_is_squared_pearson_and_scale_free():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    y = 0.5 * x + rng.normal(size=200)
    r = np.corrcoef(x, y)[0, 1]
    assert ev._r2_ols(y, x) == pytest.approx(r ** 2)
    assert ev._r2_ols(y, -10 * x + 3) == pytest.approx(r ** 2)


def test_growth_metrics_drop_nonfinite_and_handle_degenerate():
    y = np.array([0.0, 1.0, 2.0, np.nan, 4.0])
    yhat = np.array([0.0, 1.0, 2.0, 3.0, np.inf])
    m = ev._growth_metrics(y, yhat)
    assert m["n"] == 3 and m["r2_oos"] == pytest.approx(1.0)
    flat = ev._growth_metrics(np.arange(5.0), np.zeros(5))
    assert np.isnan(flat["r2_ols"]) and np.isnan(flat["spearman"])


def test_cluster_bootstrap_ci_brackets_estimate_and_needs_two_clusters():
    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(20), 30)
    x = rng.normal(size=600)
    y = x + rng.normal(0, 0.5, 600)
    lo, hi = ev._cluster_bootstrap_ci(y, x, clusters, ev._r2_ols, n_boot=200)
    est = ev._r2_ols(y, x)
    assert lo <= est <= hi and 0.6 < lo
    assert all(np.isnan(ev._cluster_bootstrap_ci(y, x, np.zeros(600), ev._r2_ols, n_boot=10)))


def test_demean_within():
    s = pd.Series([1.0, 3.0, 10.0, 20.0])
    g = pd.Series(["a", "a", "b", "b"])
    np.testing.assert_allclose(ev._demean_within(s, g), [-1, 1, -5, 5])


# ─── panel helper ───────────────────────────────────────────────────────────

def test_panel_growth_scatter_draws_45_line_and_bins():
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    y = x + rng.normal(0, 0.3, 400)
    fig, ax = plt.subplots()
    n_off = ev._panel_growth_scatter(ax, x, y, letter="A", xlabel="x", ylabel="y",
                                     r2_text="R2", central_frac=0.98)
    labels = [l.get_label() for l in ax.get_lines()]
    assert any("45" in l for l in labels) and any("conditional mean" in l for l in labels)
    assert 0 < n_off < 40
    plt.close(fig)


def test_panel_growth_scatter_insufficient_data():
    fig, ax = plt.subplots()
    assert ev._panel_growth_scatter(ax, np.arange(5.0), np.arange(5.0), letter="A",
                                    xlabel="x", ylabel="y", r2_text="") == 0
    plt.close(fig)


def test_tex_year_range_only_touches_year_ranges():
    assert ev._tex_year_range("2007-2017 (out-of-sample)") == "2007--2017 (out-of-sample)"


# ─── GB2 dollar mapping ─────────────────────────────────────────────────────

def test_gb2_ppf_finite_for_degenerate_shape_parameters():
    # Shape fitted on real W2 wealth (CBSA 30780, 2023): large c, tiny p, q.
    # The naive v/(1-v) overflowed to inf above the median.
    c, p, q, b = 124.274, 0.02, 0.023, 443795.0
    qs = ev.gb2.ppf(np.array([0.1, 0.5, 0.9, 0.995]), c, p, q, loc=0, scale=b)
    assert np.all(np.isfinite(qs)) and np.all(np.diff(qs) > 0)
    # ...and round-trips through the CDF.
    np.testing.assert_allclose(ev.gb2.cdf(qs, c, p, q, loc=0, scale=b),
                               [0.1, 0.5, 0.9, 0.995], atol=1e-6)


def test_gb2_ppf_unchanged_for_well_conditioned_parameters():
    from scipy import special as sp
    c, p, q, b = 2.0, 0.8, 2.0, 5e4
    u = np.array([0.05, 0.5, 0.95])
    v = sp.betaincinv(p, q, u)
    naive = b * (v / (1 - v)) ** (1 / c)
    np.testing.assert_allclose(ev.gb2.ppf(u, c, p, q, loc=0, scale=b), naive, rtol=1e-10)
    x = np.array([1e4, 5e4, 2e5])
    z = (x / b) ** c
    np.testing.assert_allclose(ev.gb2.cdf(x, c, p, q, loc=0, scale=b),
                               sp.betainc(p, q, z / (1 + z)), rtol=1e-10)
    np.testing.assert_allclose(ev.gb2.sf(x, c, p, q, loc=0, scale=b),
                               1 - sp.betainc(p, q, z / (1 + z)), rtol=1e-10)


def test_deflate_to_base_year():
    assert ev._deflate(100.0, 2023) == pytest.approx(100.0)
    assert ev._deflate(100.0, 2016) == pytest.approx(100.0 * 304.702 / 240.007)


def test_wealth_dollar_column_per_indicator():
    assert ev._wealth_dollar_column("W2_r5", 2016) == "W2_i_r5pct_2016"
    assert ev._wealth_dollar_column("inc", 2016) == "per_capita_income_usd_2016"


def test_load_wealth_dollars_long_deflates_and_skips_missing_years(tmp_path):
    geo = ev._PANEL_GEOID_COL
    panel = pd.DataFrame({
        geo: ["1001020100", "1001020100", "36061000100"],
        "cbsa_code": [10000, 10000, 35620],
        "W2_i_r5pct_2016": [1000.0, 1000.0, 2000.0],
        "W2_i_r5pct_2023": [1500.0, 1500.0, 2500.0],
    })
    panel.to_feather(tmp_path / ev._PANEL_FILENAME)
    long = ev._load_wealth_dollars_long(tmp_path, "W2_r5", [2016, 2020, 2023])
    assert sorted(long["year"].unique()) == [2016, 2023]          # 2020 absent
    assert len(long) == 4 and set(long["cbsa"]) == {"10000", "35620"}
    r = long[(long["GEOID"] == "01001020100") & (long["year"] == 2016)].iloc[0]
    assert r["usd"] == pytest.approx(1000.0 * 304.702 / 240.007)


def _wealth_long_lognormal(cbsas=("1", "2"), years=(2016, 2020, 2023), n=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in cbsas:
        for y in years:
            med = 4e5 * (1.03 ** (y - years[0]))
            usd = med * np.exp(rng.normal(0, 0.5, n))
            rows.append(pd.DataFrame({"GEOID": [f"{c}_{i}" for i in range(n)],
                                      "cbsa": c, "year": y, "usd": usd}))
    return pd.concat(rows, ignore_index=True)


def test_fit_city_gb2_recovers_medians_per_year():
    wl = _wealth_long_lognormal()
    params = ev._fit_city_gb2(wl, ["1", "2", "absent"])
    assert set(params) == {"1", "2"}
    assert set(params["1"]) == {2016, 2020, 2023}
    for c in ("1", "2"):
        for y in (2016, 2023):
            emp = np.median(wl[(wl.cbsa == c) & (wl.year == y)]["usd"])
            assert ev._gb2_quantile(0.5, params[c][y]) == pytest.approx(emp, rel=0.12)


def test_hazen_within_and_gb2_dollars_are_rank_preserving():
    df = pd.DataFrame({"cbsa": ["1"] * 4 + ["2"] * 2, "year": 2016,
                       "pred": [3.0, 1.0, 2.0, 4.0, 0.0, 9.0]})
    pr = ev._hazen_within(df, "pred")
    np.testing.assert_allclose(pr, [0.625, 0.125, 0.375, 0.875, 0.25, 0.75])
    df["prob"] = pr
    params = {"1": {2016: (2.0, 0.8, 2.0, 5e4)}}          # city "2" has no fit
    usd = ev._gb2_dollars(df, "prob", params)
    assert np.all(np.isfinite(usd[:4])) and np.all(np.isnan(usd[4:]))
    assert list(np.argsort(usd[:4])) == list(np.argsort(df["pred"].values[:4]))


def test_add_gb2_dollar_pairs_frozen_uses_t0_rank_at_t1_map():
    params = {"1": {2016: (2.0, 0.8, 2.0, 5e4), 2023: (2.0, 0.8, 2.0, 6e4)}}
    pairs = pd.DataFrame({"cbsa": ["1"], "t0": [2016], "t1": [2023], "pred_prob_0": [0.5],
                          "pred_usd_0": [ev._gb2_quantile(0.5, params["1"][2016])],
                          "pred_usd_1": [1.0], "label_usd_0": [1.0], "label_usd_1": [2.0],
                          "acs_usd_0": [1.0], "acs_usd_1": [-1.0]})
    out = ev._add_gb2_dollar_pairs(pairs, params).iloc[0]
    # Same quantile, scale 5e4 -> 6e4: frozen growth is exactly +20%.
    assert out["frozen_usd_1"] == pytest.approx(1.2 * out["pred_usd_0"])
    assert out["dlog_frozen_usd"] == pytest.approx(np.log(1.2))
    assert out["dlog_label_usd"] == pytest.approx(np.log(2.0))
    assert np.isnan(out["dlog_acs_usd"])                      # non-positive -> NaN


# ─── part_growth_figure_us end-to-end ───────────────────────────────────────

class _FakeCtx:
    def __init__(self, bld, tract, cbsa_meta, out):
        self.bld, self.tract, self.cbsa_meta, self.out = bld, tract, cbsa_meta, out


def _synthetic_growth_ctx(out, tracks_change: bool, seed=0, n_tracts=40,
                          years=(2012, 2014, 2016, 2018, 2020, 2022)):
    """Tract log-wealth = city level + persistent tract level + tract trend.
    Labels are the within-city-year z-score of log wealth (as in
    process_acs); the prediction follows current log wealth (tracks change)
    or is frozen at its first-year value (persistence only)."""
    rng = np.random.default_rng(seed)
    cbsas = ["10420", "35620", "41860", "99999"]
    bld_rows, tract_rows, w_rows = [], [], []
    for k, cbsa in enumerate(cbsas):
        level = rng.normal(0, 0.5, n_tracts)
        trend = rng.normal(0, 0.06, n_tracts)
        for yr in years:
            logw = 13.0 + 0.1 * k + 0.02 * (yr - years[0]) + level + trend * (yr - years[0])
            label = (logw - logw.mean()) / logw.std(ddof=1)
            base = level if not tracks_change else logw
            pred = 3.0 * base + 1.0 + rng.normal(0, 0.01, n_tracts)   # arbitrary scale
            for i in range(n_tracts):
                gid = f"{cbsa}{i:06d}"
                tract_rows.append({"cbsa": cbsa, "year": yr, "GEOID": gid,
                                   "pred": pred[i], "label": label[i]})
                bld_rows.append({"cbsa": cbsa, "year": yr, "type": "test",
                                 "pred": pred[i], "label": label[i],
                                 "building_id": f"{gid}_{yr}", "GEOID": gid})
                w_rows.append({"GEOID": gid, "cbsa": cbsa, "year": yr,
                               "usd": float(np.exp(logw[i]))})
    cbsa_meta = pd.DataFrame({"cbsa_code": cbsas, "bracket": ["small", "mega", "large", "medium"]})
    ctx = _FakeCtx(pd.DataFrame(bld_rows), pd.DataFrame(tract_rows), cbsa_meta, out)
    return ctx, pd.DataFrame(w_rows)


def _out(tmp_path):
    out = tmp_path / "evaluation"
    (out / "figures").mkdir(parents=True)
    (out / "tables").mkdir(parents=True)
    return out


def test_part_growth_figure_tracking_model_beats_rank_frozen(tmp_path):
    out = _out(tmp_path)
    ctx, wl = _synthetic_growth_ctx(out, tracks_change=True)
    h = ev.part_growth_figure_us(ctx, wealth_long=wl, n_boot=30)
    assert h["growth/n_tract_pairs"] == 4 * 40
    assert h["growth/gap_median"] == 10                       # 2022 -> 2012
    assert h["growth/usd_model_pooled_r2_oos"] > 0.9
    assert h["growth/log_model_within_r2_oos"] > 0.9
    assert h["growth/log_model_within_r2_oos"] > h["growth/log_frozen_within_r2_oos"] + 0.3
    assert (out / "figures" / "US_growth_figure.pdf").exists()
    tbl = pd.read_csv(out / "tables" / "US_growth_metrics.csv")
    assert (tbl["source"] == "this paper").sum() == 8 + 3
    bench = tbl[tbl["source"].str.startswith("Khachiyan")]
    assert set(bench["r2_oos"].round(4)) == {0.4331, -0.0999, 0.3731}
    pairs = pd.read_csv(out / "tables" / "US_growth_pairs.csv")
    assert {"d_label_usd", "d_pred_usd", "d_frozen_usd", "dlog_label_usd"} <= set(pairs.columns)


def test_part_growth_figure_frozen_model_matches_rank_frozen_benchmark(tmp_path):
    out = _out(tmp_path)
    ctx, wl = _synthetic_growth_ctx(out, tracks_change=False, seed=4)
    h = ev.part_growth_figure_us(ctx, wealth_long=wl, n_boot=30)
    # A model whose ranks never move is the rank-frozen benchmark (up to rank
    # ties), and neither explains within-city log change.
    assert h["growth/log_model_within_r2_oos"] == pytest.approx(
        h["growth/log_frozen_within_r2_oos"], abs=0.02)
    assert h["growth/log_model_within_r2_ols"] < 0.1


def test_part_growth_figure_uses_main_figure_sample(tmp_path, monkeypatch):
    out = _out(tmp_path)
    ctx, wl = _synthetic_growth_ctx(out, tracks_change=True)
    calls = []
    orig = ev._figure_test_sample

    def spy(c, what="figure", **kw):
        calls.append(what)
        return orig(c, what=what, **kw)

    monkeypatch.setattr(ev, "_figure_test_sample", spy)
    ev.part_growth_figure_us(ctx, wealth_long=wl, n_boot=10)
    ev.part_main_figure_us(ctx)
    assert calls == ["growth figure", "main figure"]


def test_part_growth_figure_without_wealth_panel_skips(tmp_path):
    out = _out(tmp_path)
    ctx, _ = _synthetic_growth_ctx(out, tracks_change=True)
    # _FakeCtx has no processed_dir -> loader fails -> clean skip, no figure.
    assert ev.part_growth_figure_us(ctx, n_boot=10) == {}
    assert not (out / "figures" / "US_growth_figure.pdf").exists()


def test_part_growth_figure_empty_inputs_skip(tmp_path):
    out = _out(tmp_path)
    empty = pd.DataFrame(columns=["cbsa", "year", "type", "pred", "label", "GEOID"])
    assert ev.part_growth_figure_us(_FakeCtx(empty, empty, None, out)) == {}
    assert not (out / "figures" / "US_growth_figure.pdf").exists()


def test_growth_part_is_registered():
    assert ev._US_PARTS["growth"] is ev.part_growth_figure_us
    us, _ = ev._split_parts(["growth"], "us")
    assert us == ["growth"]
    assert "growth" in ev._ALL_US_PARTS and "growth" not in ev._DEFAULT_US_PARTS


def test_fit_city_gb2_unsmoothed_by_default_and_smoothing_opt_in(monkeypatch):
    wl = _wealth_long_lognormal(years=(2016, 2018, 2020, 2023))
    calls = []
    orig = ev._smooth_gb2_params
    monkeypatch.setattr(ev, "_smooth_gb2_params",
                        lambda y, p, **kw: calls.append(1) or orig(y, p, **kw))
    raw = ev._fit_city_gb2(wl, ["1"])
    assert not calls
    ev._fit_city_gb2(wl, ["1"], smooth=True)
    assert calls
    assert set(raw["1"]) == {2016, 2018, 2020, 2023}


def test_gb2_fit_error_small_for_good_fit_large_for_wrong_params():
    wl = _wealth_long_lognormal(cbsas=("1",), years=(2016,))
    params = ev._fit_city_gb2(wl, ["1"])
    good = ev._gb2_fit_error(wl, params, [("1", 2016)])
    assert good["max_log_err"].iloc[0] < 0.1
    bad = {"1": {2016: (params["1"][2016][:3] + (params["1"][2016][3] * 4,))}}
    assert ev._gb2_fit_error(wl, bad, [("1", 2016)])["max_log_err"].iloc[0] > 1.0
    assert ev._gb2_fit_error(wl, params, [("9", 2016)]).empty


# ─── stable vs change (training's MOE gate) ─────────────────────────────────

def test_attach_change_pvalue_matches_training_z_test():
    from scipy.stats import norm
    pairs = pd.DataFrame({"GEOID": ["A", "B"], "t0": [2016, 2016], "t1": [2022, 2022],
                          "label_0": [0.0, 0.0], "label_1": [0.5, 0.05]})
    se = pd.DataFrame({"GEOID": ["A", "A", "B", "B"], "year": [2016, 2022] * 2,
                       "se": [0.1, 0.2, 0.1, 0.2], "valid_change": [1, 1, 0, 0]})
    out = ev._attach_change_pvalue(pairs, se).set_index("GEOID")
    z = 0.5 / np.sqrt(0.1 ** 2 + 0.2 ** 2)
    assert out.loc["A", "change_z"] == pytest.approx(z)
    assert out.loc["A", "change_p"] == pytest.approx(2 * (1 - norm.cdf(z)))
    assert out.loc["A", "valid_change"] == 1 and out.loc["B", "valid_change"] == 0


def test_change_groups_are_nested_and_exclude_missing_se():
    pairs = pd.DataFrame({"change_p": [0.5, 0.08, 0.03, 0.001, np.nan],
                          "valid_change": [0, 1, 1, 1, np.nan]})
    g = ev._change_groups(pairs)
    assert list(g["stable (p>=0.10)"]) == [True, False, False, False, False]
    assert g["change p<0.10"].sum() == 3
    assert g["change p<0.05"].sum() == 2
    assert g["change p<0.01"].sum() == 1
    assert (g["change p<0.01"] <= g["change p<0.05"]).all()
    assert g["training flag: change"].sum() == 3 and g["training flag: stable"].sum() == 1


def _se_long_for(ctx, se=0.05):
    t = ctx.tract
    return pd.DataFrame({"GEOID": t["GEOID"], "year": t["year"], "se": se,
                         "valid_change": 0.0})


def test_part_growth_by_change_tracking_model_scores_higher_in_change_group(tmp_path):
    out = _out(tmp_path)
    ctx, wl = _synthetic_growth_ctx(out, tracks_change=True, n_tracts=60)
    ev.part_growth_figure_us(ctx, wealth_long=wl, se_long=_se_long_for(ctx), n_boot=20)
    bc = pd.read_csv(out / "tables" / "US_growth_by_change.csv")
    m = bc[(bc["predictor"] == "model") & (bc["target"].str.startswith("d W2"))].set_index("group")
    assert m.loc["change p<0.01", "r2_oos"] > 0.8
    assert m.loc["change p<0.01", "sd_actual"] > m.loc["stable (p>=0.10)", "sd_actual"]
    pairs = pd.read_csv(out / "tables" / "US_growth_pairs.csv")
    assert {"change_p", "change_z", "se_0", "se_1"} <= set(pairs.columns)


def test_part_growth_by_change_skipped_without_se(tmp_path):
    out = _out(tmp_path)
    ctx, wl = _synthetic_growth_ctx(out, tracks_change=True)
    h = ev.part_growth_figure_us(ctx, wealth_long=wl, n_boot=10)   # no processed_dir
    assert "growth/n_tract_pairs" in h
    assert not (out / "tables" / "US_growth_by_change.csv").exists()


def test_attach_change_pvalue_uses_closest_se_year_like_training():
    pairs = pd.DataFrame({"GEOID": ["A"], "t0": [2017], "t1": [2022],
                          "label_0": [0.0], "label_1": [0.3]})
    se = pd.DataFrame({"GEOID": ["A", "A"], "year": [2014, 2023],
                       "se": [0.1, 0.2], "valid_change": [0, 0]})
    out = ev._attach_change_pvalue(pairs, se).iloc[0]
    assert (out["se_year_0"], out["se_year_1"]) == (2014, 2023)
    assert out["change_z"] == pytest.approx(0.3 / np.sqrt(0.1 ** 2 + 0.2 ** 2))


def test_change_detection_stats_reference_values():
    rng = np.random.default_rng(7)
    stable = pd.DataFrame({"d_label": rng.normal(0, 0.05, 300), "se_0": 0.05, "se_1": 0.05,
                           "d_pred": rng.normal(0, 0.05, 300)})
    d = rng.normal(0, 1.0, 300)
    change = pd.DataFrame({"d_label": d, "se_0": 0.05, "se_1": 0.05, "d_pred": d},
                          index=np.arange(300, 600))
    st = ev._change_detection_stats(change, stable, "d_label", "d_pred")
    assert st["label_reliability"] > 0.99
    assert st["sign_agree"] == pytest.approx(1.0)
    assert st["auc_abs_vs_stable"] > 0.9
    blind = change.assign(d_pred=rng.normal(0, 0.05, 300))
    st2 = ev._change_detection_stats(blind, stable, "d_label", "d_pred")
    assert abs(st2["auc_abs_vs_stable"] - 0.5) < 0.1 and abs(st2["sign_agree"] - 0.5) < 0.1
    # The stable group against itself: no AUC.
    assert np.isnan(ev._change_detection_stats(stable, stable, "d_label", "d_pred")["auc_abs_vs_stable"])


# ─── Part F: NYC growth by construction cohort ──────────────────────────────

def test_construction_window_groups_window_and_pinned_stable():
    pairs = pd.DataFrame({"GEOID": ["s", "a", "b", "pre", "amb"],
                          "t0": 2014, "t1": 2024})
    cohorts = {
        0.01: pd.DataFrame({"GEOID_str": ["s", "a", "b", "pre", "amb"],
                            "cohort_year": [0, 2018, 2016, 2012, 2020]}),
        0.05: pd.DataFrame({"GEOID_str": ["s", "a", "pre"],          # b, amb dropped
                            "cohort_year": [0, 2022, 2012]}),
    }
    g = ev._construction_window_groups(pairs, cohorts)
    assert list(pairs["GEOID"][g["stable"]]) == ["s"]
    assert list(pairs["GEOID"][g["change 1%"]]) == ["a", "b", "amb"]
    assert list(pairs["GEOID"][g["change 5%"]]) == ["a"]
    assert list(pairs["GEOID"][g["pre-window 1%"]]) == ["pre"]
    assert not (g["stable"] & g["change 1%"]).any()


def test_construction_group_row_mean_shifts_and_r2():
    rng = np.random.default_rng(0)
    d = rng.normal(0.5, 1, 200)
    sub = pd.DataFrame({"GEOID": [f"c{i}" for i in range(200)], "d_label": d,
                        "d_pred": d + 1.0, "d_pred_z": d}, index=range(200, 400))
    stable = pd.DataFrame({"GEOID": [f"s{i}" for i in range(200)],
                           "d_label": rng.normal(0, 0.1, 200), "d_pred": 1.0,
                           "d_pred_z": rng.normal(0, 0.1, 200)})
    r = ev._construction_group_row(sub, stable, n_boot=20)
    assert r["n"] == 200 and r["r2_oos"] == pytest.approx(1.0)
    assert r["d_pred_vs_stable"] == pytest.approx(d.mean(), abs=1e-9)
    assert r["d_label_vs_stable"] == pytest.approx(d.mean() - stable["d_label"].mean())
    assert r["sign_agree"] == pytest.approx(1.0) and r["auc_abs_vs_stable"] > 0.9
    small = ev._construction_group_row(sub.iloc[:5], stable, n_boot=5)
    assert small["n"] == 5 and "r2_oos" not in small


def test_part_f_end_to_end_with_stubbed_geometry(tmp_path, monkeypatch):
    from src import csa_event_study as ces
    out = tmp_path / "evaluation"
    (out / "tables").mkdir(parents=True)
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / ces.CSA_CITIES["nyc"].footprints_filename).write_bytes(b"")
    rng = np.random.default_rng(1)
    years = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
    gids = [f"36061{i:06d}" for i in range(120)]
    built = {g: (2018 if i < 40 else 0) for i, g in enumerate(gids)}   # 40 changers
    size = {g: (0.5 + 1.0 * (i % 40) / 39 if built[g] else 0.0) for i, g in enumerate(gids)}
    level = {g: rng.normal(0, 1) for g in gids}
    rows, labs = [], []
    for y in years:
        latent = np.array([level[g] + rng.normal(0, 0.02)
                           + (size[g] if (built[g] and y >= built[g]) else 0.0) for g in gids])
        z = (latent - latent.mean()) / latent.std(ddof=1)   # labels are within-year z
        for g, lab in zip(gids, z):
            rows.append({"GEOID_str": g, "year": y, "pred_all": 2 * lab})
            labs.append({"GEOID_str": g, "year": y, "Rel_Score": lab})
    monkeypatch.setattr(ev, "_csa_city_geometry", lambda p, s: (None, pd.DataFrame(
        {"GEOID_str": gids}), None))
    monkeypatch.setattr(ev, "_csa_city_outcomes", lambda *a, **k: pd.DataFrame(rows))
    monkeypatch.setattr(ev, "_load_tract_long", lambda r, y=None: pd.DataFrame(labs))
    monkeypatch.setattr(ev, "_csa_split_of",
                        lambda p: {g: ("heldout" if i % 2 else "train") for i, g in enumerate(gids)})

    class _Coh:
        def __init__(self, t):
            self.cohorts = pd.DataFrame({"GEOID_str": gids,
                                         "cohort_year": [built[g] for g in gids]})
    monkeypatch.setattr(ces, "build_tract_cohorts", lambda *a, threshold, **k: _Coh(threshold))
    t = ev.part_f(tmp_path / "res", proc, out, years=years, n_boot=20)
    assert (out / "tables" / "F_growth_by_construction.csv").exists()
    held = t[t["split"] == "Held out"].set_index("group")
    assert held.loc["stable", "n"] == 40 and held.loc["change 1%", "n"] == 20
    assert held.loc["change 1%", "r2_ols"] > 0.9
    assert held.loc["change 1%", "d_label_vs_stable"] > 0.3
    assert held.loc["change 1%", "d_pred_vs_stable"] == pytest.approx(
        2 * held.loc["change 1%", "d_label_vs_stable"], rel=1e-6)


# ─── Part F ACS-window scatter ──────────────────────────────────────────────

NYC_YEARS = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]


def test_acs_window_years_match_5yr_survey_window():
    assert ev._acs_window_years(2018, NYC_YEARS) == [2014, 2016, 2018]
    assert ev._acs_window_years(2023, NYC_YEARS) == [2020, 2022]
    assert ev._acs_window_years(2013, [2011, 2013, 2015]) == [2011, 2013]


def test_window_average_and_counts():
    t = pd.DataFrame({"GEOID": ["a"] * 3 + ["b"], "year": [2014, 2016, 2020, 2016],
                      "pred": [1.0, 3.0, 9.0, 5.0]})
    w = ev._window_average(t, [2014, 2016, 2018]).set_index("GEOID")
    assert w.loc["a", "mean"] == 2.0 and w.loc["a", "n_years"] == 2
    assert w.loc["b", "mean"] == 5.0 and w.loc["b", "n_years"] == 1


def test_acs_window_pairs_hand_checked():
    rows = []
    for g, pre, post in (("a", 0.0, 1.0), ("b", 1.0, 1.0), ("c", 2.0, 4.0)):
        for y in (2014, 2016, 2018):
            rows.append({"GEOID": g, "year": y, "pred": pre})
        for y in (2020, 2022):
            rows.append({"GEOID": g, "year": y, "pred": post})
    labels = pd.DataFrame({"GEOID": ["a", "b", "c"] * 2, "year": [2018] * 3 + [2023] * 3,
                           "label": [0.0, 1.0, 2.0, 1.0, 1.0, 3.0]})
    p = ev._acs_window_pairs(pd.DataFrame(rows), labels, 2018, 2023, NYC_YEARS).set_index("GEOID")
    assert p.attrs["window_pre"] == [2014, 2016, 2018]
    assert list(p["d_label"]) == [1.0, 0.0, 1.0]
    assert list(p["d_pred"]) == [1.0, 0.0, 2.0]
    z = lambda s: (s - s.mean()) / s.std()
    np.testing.assert_allclose(p["d_pred_z"], z(pd.Series([1.0, 1, 4])).values
                               - z(pd.Series([0.0, 1, 2])).values)
    assert list(p["pred_single_pre"]) == [0.0, 1.0, 2.0]   # the 2018 flight


def test_load_panel_scores(tmp_path):
    geo = ev._PANEL_GEOID_COL
    pd.DataFrame({geo: ["36061000100", "36061000100"],
                  "Rel_Score_W2_i_r5pct_2018": [0.5, 0.5],
                  "Rel_Score_W2_i_r5pct_2023": [0.7, 0.7]}).to_feather(tmp_path / ev._PANEL_FILENAME)
    s = ev._load_panel_scores(tmp_path, "W2_r5", [2018, 2023])
    assert len(s) == 2 and set(s["label"]) == {0.5, 0.7}


def test_part_f_acs_window_end_to_end(tmp_path, monkeypatch):
    from src import csa_event_study as ces
    out = tmp_path / "evaluation"
    (out / "tables").mkdir(parents=True)
    (out / "figures").mkdir(parents=True)
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / ces.CSA_CITIES["nyc"].footprints_filename).write_bytes(b"")
    rng = np.random.default_rng(3)
    gids = [f"36061{i:06d}" for i in range(160)]
    cohort = {g: (2020 if i < 50 else (2016 if i < 60 else 0)) for i, g in enumerate(gids)}
    jump = {g: (rng.uniform(0.3, 1.5) if cohort[g] == 2020 else 0.0) for g in gids}
    level = {g: rng.normal() for g in gids}
    rows = [{"GEOID_str": g, "year": y,
             "pred_all": level[g] + (jump[g] if y >= 2020 else 0) + rng.normal(0, 0.01)}
            for y in NYC_YEARS for g in gids]
    lab = pd.DataFrame([{"GEOID": g, "year": y, "label": level[g] + (jump[g] if y == 2023 else 0)}
                        for y in (2018, 2023) for g in gids])
    monkeypatch.setattr(ev, "_csa_city_geometry", lambda p, s: (None, pd.DataFrame(
        {"GEOID_str": gids}), None))
    monkeypatch.setattr(ev, "_csa_city_outcomes", lambda *a, **k: pd.DataFrame(rows))
    monkeypatch.setattr(ev, "_load_panel_scores", lambda *a, **k: lab)
    monkeypatch.setattr(ev, "_csa_split_of",
                        lambda p: {g: ("heldout" if i % 2 else "train") for i, g in enumerate(gids)})

    class _Coh:
        def __init__(self):
            self.cohorts = pd.DataFrame({"GEOID_str": gids,
                                         "cohort_year": [cohort[g] for g in gids]})
    monkeypatch.setattr(ces, "build_tract_cohorts", lambda *a, **k: _Coh())
    wl = pd.DataFrame([{"GEOID": g, "cbsa": "35620", "year": y,
                        "usd": float(4e5 * np.exp(0.5 * (level[g] + (jump[g] if y == 2023 else 0))))}
                       for y in (2018, 2023) for g in gids])
    t = ev.part_f_acs_window(tmp_path / "res", proc, out, years=NYC_YEARS, n_boot=10,
                             wealth_long=wl)
    assert (out / "figures" / "F_acs_window_scatter.pdf").exists()
    sel = (t["split"] == "All (held out + train)") & (t["prediction"] == "window-averaged")
    usd = t[sel & (t["target"] == "dlog wealth USD (GB2)")].set_index("group")
    assert usd.loc["stable", "n"] == 100 and usd.loc["change 5%", "n"] == 50  # 2016 cohort excluded
    assert usd.loc["change 5%", "r2_ols"] > 0.8
    assert set(t["target"]) >= {"d wealth USD (GB2)", "d wealth USD (raw ACS W2)"}


def test_add_window_gb2_dollars_maps_ranks_per_vintage():
    params = {2018: (2.0, 0.8, 2.0, 5e4), 2023: (2.0, 0.8, 2.0, 6e4)}
    pairs = pd.DataFrame({"GEOID": ["a", "b", "c"],
                          "label_pre": [0.0, 1.0, 2.0], "label_post": [2.0, 1.0, 0.0],
                          "pred_pre": [0.0, 1.0, 2.0], "pred_post": [0.0, 1.0, 2.0],
                          "pred_single_pre": [0.0, np.nan, 2.0], "pred_single_post": [0.0, 1.0, 2.0]})
    acs = pd.DataFrame({"GEOID": ["a", "b", "c"] * 2, "year": [2018] * 3 + [2023] * 3,
                        "usd": [1.0, 2, 3, 2, 2, 6]})
    o = ev._add_window_gb2_dollars(pairs, params, 2018, 2023, acs_usd=acs,
                                   winsor=None).set_index("GEOID")
    # Unchanged rank, scale 5e4 -> 6e4: +20% for every tract, exactly the frozen benchmark.
    np.testing.assert_allclose(np.exp(o["dlog_pred_usd"]), 1.2)
    np.testing.assert_allclose(o["frozen_usd_post"], o["pred_usd_post"])
    # Rank reversal in the label: the poorest becomes the richest.
    assert o.loc["a", "d_label_usd"] > 0 > o.loc["c", "d_label_usd"]
    # Missing single flight -> NaN for that tract only, others still ranked.
    assert np.isnan(o.loc["b", "pred_single_usd_pre"]) and np.isfinite(o.loc["a", "pred_single_usd_pre"])
    assert o.loc["c", "d_acs_usd"] == 3.0 and o.loc["a", "dlog_acs_usd"] == pytest.approx(np.log(2))


def test_winsorize_clips_to_own_percentiles_and_keeps_nan():
    v = np.r_[np.arange(100.0), 1e9, np.nan]
    w = ev._winsorize(v, 0.01)
    assert np.isnan(w[-1])
    assert w[-2] == pytest.approx(np.percentile(v[:-1], 99))
    assert w[:-1].min() == pytest.approx(np.percentile(v[:-1], 1))
    np.testing.assert_array_equal(ev._winsorize(np.array([np.nan]), 0.01), [np.nan])


def test_add_window_gb2_dollars_winsorizes_tail_leverage():
    n = 300
    params = {2018: (124.0, 0.02, 0.023, 4.4e5), 2023: (124.0, 0.02, 0.023, 4.6e5)}  # NYC-like tail
    r = np.arange(n, dtype=float)
    pairs = pd.DataFrame({"GEOID": [str(i) for i in range(n)], "label_pre": r,
                          "label_post": np.r_[r[:-2], r[-1], r[-2]],      # top two swap
                          "pred_pre": r, "pred_post": r,
                          "pred_single_pre": r, "pred_single_post": r})
    raw = ev._add_window_gb2_dollars(pairs, params, 2018, 2023, winsor=None)
    win = ev._add_window_gb2_dollars(pairs, params, 2018, 2023)
    share = lambda v: ((v - v.mean()) ** 2).nlargest(2).sum() / ((v - v.mean()) ** 2).sum()
    assert share(raw["d_label_usd"]) > 0.9           # the swap dominates unclipped
    assert win["label_usd_post"].max() < raw["label_usd_post"].max()
    assert share(win["d_label_usd"]) < share(raw["d_label_usd"])


# ─── direction accuracy ─────────────────────────────────────────────────────

def test_direction_metrics_reference_values():
    a = np.r_[np.ones(80), -np.ones(20)]
    m = ev._direction_metrics(a, a, n_boot=50)
    assert m["accuracy"] == 1.0 and m["balanced"] == 1.0 and m["kappa"] == pytest.approx(1.0)
    assert m["majority"] == pytest.approx(0.8)
    # "Always up" matches the majority baseline but is chance on balanced accuracy.
    up = ev._direction_metrics(a, np.ones(100), n_boot=50)
    assert up["accuracy"] == pytest.approx(0.8) and up["balanced"] == pytest.approx(0.5)
    assert up["kappa"] == pytest.approx(0.0)
    # Zeros and NaNs are dropped.
    z = ev._direction_metrics([1.0, 0.0, np.nan, -1.0], [1.0, 1.0, 1.0, -1.0], n_boot=10)
    assert z["n"] == 2 and z["accuracy"] == 1.0
    assert ev._direction_metrics([], [], n_boot=5)["n"] == 0


def test_direction_table_relative_removes_common_growth():
    rng = np.random.default_rng(0)
    rel = rng.normal(0, 1, 400)
    pairs = pd.DataFrame({"d_label_usd": 5 + rel, "d_pred_usd": 5 + rng.normal(0, 1, 400),
                          "split_group": "heldout"})
    t = ev._direction_table(pairs, {"all": pd.Series(True, index=pairs.index)},
                            {"model": "d_pred_usd"}, n_boot=20).set_index("definition")
    held = t[t["split"] == "Held out"]
    assert held.loc["absolute", "accuracy"] > 0.95          # everyone went up
    assert abs(held.loc["relative to city", "balanced"] - 0.5) < 0.1   # no real skill


# ─── Khachiyan per-capita benchmark ─────────────────────────────────────────

def _pc_synthetic(n=60, cities=("1", "2"), seed=0, tracks=True):
    """Tracts with flights in state-specific years; per-capita income grows by
    a tract-specific rate; the prediction follows log income (tracks=True) or
    stays at its 2016 value."""
    rng = np.random.default_rng(seed)
    flights = {"1": [2016, 2018, 2020, 2022], "2": [2017, 2019, 2021, 2023]}
    trows, irows = [], []
    for c in cities:
        lvl = rng.normal(10.5, 0.5, n)
        g = rng.normal(0.1, 0.15, n)
        for i in range(n):
            gid = f"{c}{i:05d}"
            for y in flights[c]:
                frac = (y - 2016) / 7
                trows.append({"GEOID": gid, "cbsa": c, "year": y,
                              "pred": lvl[i] + (g[i] * frac if tracks else 0) + rng.normal(0, 0.005)})
            for y, frac in ((2018, 2 / 7), (2023, 1.0)):
                irows.append({"GEOID": gid, "cbsa": c, "year": y,
                              "usd": float(np.exp(lvl[i] + g[i] * frac))})
    return pd.DataFrame(trows), pd.DataFrame(irows)


def test_window_pc_income_pairs_ranks_within_city_and_frozen_is_exact():
    tract, inc = _pc_synthetic()
    params = {"1": {2018: (2.0, 0.8, 2.0, 3e4), 2023: (2.0, 0.8, 2.0, 3.6e4)},
              "2": {2018: (2.0, 0.8, 2.0, 5e4), 2023: (2.0, 0.8, 2.0, 5e4)}}
    p = ev._window_pc_income_pairs(tract, inc, params, 2018, 2023,
                                   sorted(tract["year"].unique()), winsor=None)
    assert p.attrs["window_pre"] == [2016, 2017, 2018]
    assert len(p) == 120 and set(p["cbsa"]) == {"1", "2"}
    # Hazen ranks are within city: each city spans the full (0, 1) range.
    for c, g in p.groupby("cbsa"):
        assert g["prob_pre"].min() < 0.02 and g["prob_pre"].max() > 0.98
    # Same shape, scale x1.2 in city 1 and x1.0 in city 2: frozen dlog is exact.
    np.testing.assert_allclose(p.loc[p.cbsa == "1", "dlog_frozen"], np.log(1.2))
    np.testing.assert_allclose(p.loc[p.cbsa == "2", "dlog_frozen"], 0.0, atol=1e-12)
    assert np.isfinite(p["dlog_actual"]).all()


def test_khachiyan_rows_tracking_vs_frozen():
    tract, inc = _pc_synthetic(n=150)
    wl = inc.copy()
    params = ev._fit_city_gb2(wl, ["1", "2"])
    p = ev._window_pc_income_pairs(tract, inc, params, 2018, 2023, sorted(tract["year"].unique()))
    rows = {(r["predictor"], r["scope"]): r for r in ev._khachiyan_pc_rows(p, "x", n_boot=20)}
    assert rows[("model", "within-city")]["r2_ols"] > 0.5
    assert rows[("model", "within-city")]["r2_ols"] > rows[("rank-frozen", "within-city")]["r2_ols"]
    frozen_tract, _ = _pc_synthetic(n=150, tracks=False)
    pf = ev._window_pc_income_pairs(frozen_tract, inc, params, 2018, 2023,
                                    sorted(frozen_tract["year"].unique()))
    rf = {(r["predictor"], r["scope"]): r for r in ev._khachiyan_pc_rows(pf, "x", n_boot=20)}
    assert rf[("model", "within-city")]["r2_ols"] < 0.1


def test_part_khachiyan_pc_us_end_to_end(tmp_path):
    out = _out(tmp_path)
    tract, inc = _pc_synthetic(n=60)
    bld = tract.assign(type="test", label=0.0, building_id=lambda d: d.GEOID + d.year.astype(str))
    tract = tract.assign(label=rng_label(len(tract)))
    ctx = _FakeCtx(bld.assign(label=tract["label"].values), tract,
                   pd.DataFrame({"cbsa_code": ["1", "2"], "bracket": ["small", "large"]}), out)
    h = ev.part_khachiyan_pc_us(ctx, income_long=inc, n_boot=10)
    assert h["khachiyan_pc/n"] == 120
    t = pd.read_csv(out / "tables" / "US_khachiyan_pc.csv")
    bench = t[t["source"].str.startswith("Khachiyan")]
    assert set(bench["r2_oos"].round(4)) == {0.0461, 0.0674, 0.0306, 0.0653}
    assert "khachiyan" in ev._US_PARTS and "khachiyan" in ev._ALL_US_PARTS


def rng_label(n, seed=11):
    return np.random.default_rng(seed).normal(0, 1, n)


def test_part_f_khachiyan_end_to_end(tmp_path, monkeypatch):
    out = _out(tmp_path)
    tract, inc = _pc_synthetic(n=80, cities=("1",))
    tract["GEOID"] = "36061" + tract["GEOID"].str[1:].str.zfill(6)
    inc["GEOID"] = "36061" + inc["GEOID"].str[1:].str.zfill(6)
    tl = tract.rename(columns={"GEOID": "GEOID_str", "pred": "predicted_value"})
    tl["GEOID"] = tl["GEOID_str"].astype("int64")   # the real loader carries both
    monkeypatch.setattr(ev, "_load_tract_long", lambda r, y=None: tl)
    monkeypatch.setattr(ev, "_load_wealth_dollars_long", lambda *a, **k: inc.assign(cbsa="35620"))
    split = {g: ("heldout" if i % 2 else "train") for i, g in enumerate(sorted(tl.GEOID_str.unique()))}
    t = ev.part_f_khachiyan(tmp_path / "r", tmp_path, out, years=[2016, 2018, 2020, 2022],
                            split_of=split, n_boot=10)
    assert set(t["tracts"]) == {"all", "heldout", "train"}
    assert (out / "tables" / "F_khachiyan_pc.csv").exists()
    m = t[(t["tracts"] == "all") & (t["predictor"] == "model")].iloc[0]
    assert m["n"] == 80 and m["r2_ols"] > 0.5


def test_cross_fitted_calibration_never_uses_scored_rows():
    rng = np.random.default_rng(0)
    g = pd.Series(np.repeat(["a", "b", "c"], 50))
    p = pd.Series(rng.normal(0, 1, 150))
    a = 0.2 * p + 0.5 + rng.normal(0, 0.01, 150)
    cal = ev._cross_fitted_calibration(a, p, g)
    np.testing.assert_allclose(cal, a, atol=0.05)            # learns the shrink, out of sample
    # A city whose own relation differs is NOT fitted to itself.
    a2 = a.copy(); a2[g == "c"] = -5 * p[g == "c"]
    cal2 = ev._cross_fitted_calibration(a2, p, g)
    assert np.corrcoef(cal2[g == "c"], a2[g == "c"])[0, 1] < 0
    # fit_mask: one fit on the masked rows applied to all.
    fm = pd.Series(np.r_[np.ones(100, bool), np.zeros(50, bool)])
    cal3 = ev._cross_fitted_calibration(a, p, g, fit_mask=fm)
    assert cal3.notna().all()
