"""Synthetic tests for src/change_reliability.py and its Part F wiring."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import change_reliability as cr
from src import evaluation as ev

DIFF = lambda pre, post: np.asarray(post, float) - np.asarray(pre, float)


# ─── ACS side ─────────────────────────────────────────────────────────────────

def test_fill_missing_se_uses_median_of_valid():
    se, n = cr.fill_missing_se(np.array([0.1, np.nan, 0.3, 0.0, 0.2]))
    np.testing.assert_allclose(se, [0.1, 0.2, 0.3, 0.2, 0.2])
    assert n == 2
    with pytest.raises(ValueError):
        cr.fill_missing_se(np.array([np.nan, 0.0]))


def test_simulated_noise_matches_analytic_variance_for_linear_map():
    n = 50
    se_pre, se_post = np.full(n, 0.3), np.linspace(0.1, 0.5, n)
    nv = cr.simulate_label_noise(np.zeros(n), np.zeros(n), se_pre, se_post, DIFF,
                                 n_draws=4000, seed=1)
    np.testing.assert_allclose(nv, se_pre ** 2 + se_post ** 2, rtol=0.1)


# ─── prediction side ──────────────────────────────────────────────────────────

def test_split_half_assignments_are_disjoint_and_unique():
    splits = cr.split_half_assignments([2010, 2012, 2014], [2020, 2022])
    # 6 ordered splits of 3 flights x 2 of 2 flights, halved by the A/B swap.
    assert len(splits) == 6
    for (pa, qa), (pb, qb) in splits:
        assert pa and pb and qa and qb
        assert not set(pa) & set(pb) and not set(qa) & set(qb)
        assert set(pa) | set(pb) == {2010, 2012, 2014}
    assert len(cr.split_half_assignments([1, 2], [3, 4])) == 2
    with pytest.raises(ValueError):
        cr.split_half_assignments([2010], [2020, 2022])


def test_window_mean_skips_missing_flights():
    w = pd.DataFrame({2010: [1.0, np.nan], 2012: [3.0, 5.0]})
    np.testing.assert_allclose(cr.window_mean(w, [2010, 2012]), [2.0, 5.0])
    assert np.isnan(cr.window_mean(w, [2030])).all()


# ─── statistics ───────────────────────────────────────────────────────────────

def _synthetic(n=20000, beta=0.8, sd_acs=1.0, sd_flight=0.6, seed=0):
    """True change s ~ N(0,1). ACS: Δy = s + noise(sd_acs). Predictions: two
    flights per window, each with independent noise sd_flight."""
    rng = np.random.default_rng(seed)
    s = rng.standard_normal(n)
    dy = s + sd_acs * rng.standard_normal(n)
    level = rng.standard_normal(n)
    flights = {y: level + (beta * s if y >= 2020 else 0) + sd_flight * rng.standard_normal(n)
               for y in (2010, 2012, 2020, 2022)}
    wide = pd.DataFrame(flights)
    dp = DIFF(cr.window_mean(wide, [2010, 2012]), cr.window_mean(wide, [2020, 2022]))
    splits = cr.split_half_assignments([2010, 2012], [2020, 2022])
    halves = np.array([[a, b] for a, b in cr.split_half_deltas(wide, splits, DIFF)])
    return dy, dp, np.full(n, sd_acs ** 2), halves, beta, sd_flight


def test_reliability_stats_recover_known_decomposition():
    dy, dp, nv, halves, beta, sdf = _synthetic()
    st = cr.reliability_stats(dy, dp, nv, halves)
    # rel_y = 1 / (1 + 1); Δp noise = 2 * sdf^2 / 2 (two-flight means) -> sdf^2.
    assert st["rel_y"] == pytest.approx(0.5, abs=0.03)
    assert st["signal_var_dp"] == pytest.approx(beta ** 2, rel=0.08)
    assert st["rel_p"] == pytest.approx(beta ** 2 / (beta ** 2 + sdf ** 2), abs=0.03)
    assert st["r_disattenuated"] == pytest.approx(1.0, abs=0.06)   # same true change
    assert st["r"] < 0.75
    assert st["slope_disattenuated"] == pytest.approx(1 / beta, rel=0.08)
    assert st["r2_share_of_ceiling"] == pytest.approx(st["r2_ols"] / st["rel_y"])


def test_reliability_stats_unrelated_signal_gives_zero_r_star():
    dy, dp, nv, halves, *_ = _synthetic(seed=3)
    rng = np.random.default_rng(9)
    st = cr.reliability_stats(rng.permutation(dy), dp, nv, halves)
    assert abs(st["r_disattenuated"]) < 0.06
    assert st["rel_p"] > 0.5            # reliable but wrong


def test_reliability_stats_drops_rows_with_missing_inputs():
    dy, dp, nv, halves, *_ = _synthetic(n=200)
    dy[:5] = np.nan
    halves[0, 1, 5:8] = np.nan
    assert cr.reliability_stats(dy, dp, nv, halves)["n"] == 192
    assert cr.reliability_stats(dy[10:13], dp[10:13], nv[10:13], halves[:, :, 10:13]) == {"n": 3}


def test_disattenuated_corr_edge_cases():
    assert cr.disattenuated_corr(0.2, 0.25, 0.64) == pytest.approx(0.5)
    assert np.isnan(cr.disattenuated_corr(0.2, -0.1, 0.5))
    assert np.isnan(cr.disattenuated_corr(np.nan, 0.5, 0.5))


def test_bootstrap_ci_covers_truth():
    dy, dp, nv, halves, *_ = _synthetic(n=3000, seed=4)
    ci = cr.bootstrap_reliability(dy, dp, nv, halves, n_boot=200, seed=0)
    assert ci["rel_y_ci_lo"] < 0.5 < ci["rel_y_ci_hi"]
    assert ci["r_disattenuated_ci_lo"] < 1.0 < ci["r_disattenuated_ci_hi"] + 0.1
    assert cr.bootstrap_reliability(dy[:5], dp[:5], nv[:5], halves[:, :, :5]) == {}


# ─── evaluation wiring ────────────────────────────────────────────────────────

def test_rank_gb2_dlog_matches_window_gb2_dollars():
    rng = np.random.default_rng(2)
    n = 300
    pairs = pd.DataFrame({"GEOID": [f"g{i}" for i in range(n)],
                          "label_pre": rng.normal(size=n), "label_post": rng.normal(size=n),
                          "pred_pre": rng.normal(size=n), "pred_post": rng.normal(size=n),
                          "pred_single_pre": rng.normal(size=n),
                          "pred_single_post": rng.normal(size=n)})
    params = {2014: (2.0, 1.5, 1.2, 4e4), 2023: (2.2, 1.4, 1.1, 5e4)}
    ref = ev._add_window_gb2_dollars(pairs, params, 2014, 2023)
    got = ev._rank_gb2_dlog(pairs["label_pre"], pairs["label_post"], params[2014], params[2023])
    np.testing.assert_allclose(got, ref["dlog_label_usd"], atol=1e-10)
    got = ev._rank_gb2_dlog(pairs["pred_pre"], pairs["pred_post"], params[2014], params[2023])
    np.testing.assert_allclose(got, ref["dlog_pred_usd"], atol=1e-10)


def test_label_se_pair_uses_nearest_year():
    se_long = pd.DataFrame({"GEOID": ["a", "b", "a", "b"], "year": [2014, 2014, 2023, 2023],
                            "se": [0.1, 0.2, 0.3, 0.4]})
    pre, post, used = ev._label_se_pair(pd.Series(["b", "a", "c"]), se_long, 2013, 2023)
    assert used == (2014, 2023)
    np.testing.assert_allclose(pre[:2], [0.2, 0.1])
    np.testing.assert_allclose(post[:2], [0.4, 0.3])
    assert np.isnan(pre[2])


NYC_YEARS = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]


def test_part_f_change_reliability_end_to_end(tmp_path, monkeypatch):
    from src import csa_event_study as ces
    out = tmp_path / "evaluation"
    (out / "tables").mkdir(parents=True)
    (out / "figures").mkdir(parents=True)
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / ces.CSA_CITIES["nyc"].footprints_filename).write_bytes(b"")
    rng = np.random.default_rng(7)
    gids = [f"36061{i:06d}" for i in range(400)]
    cohort = {g: (2016 if i < 150 else 0) for i, g in enumerate(gids)}
    true = {g: rng.normal(0.6, 0.5) if cohort[g] else rng.normal(0, 0.3) for g in gids}
    level = {g: rng.normal() for g in gids}
    rows = [{"GEOID_str": g, "year": y,
             "pred_all": level[g] + (true[g] if y >= 2016 else 0) + rng.normal(0, 0.3)}
            for y in NYC_YEARS for g in gids]
    se = 0.2
    lab = pd.DataFrame([{"GEOID": g, "year": y,
                         "label": level[g] + (true[g] if y == 2023 else 0) + rng.normal(0, se)}
                        for y in (2014, 2023) for g in gids])
    wl = pd.DataFrame([{"GEOID": g, "cbsa": "35620", "year": y,
                        "usd": float(4e5 * np.exp(0.5 * (level[g] + (true[g] if y == 2023 else 0))))}
                       for y in (2014, 2023) for g in gids])
    se_long = pd.DataFrame([{"GEOID": g, "year": y, "se": se} for y in (2014, 2023) for g in gids])
    monkeypatch.setattr(ev, "_csa_city_geometry",
                        lambda p, s: (None, pd.DataFrame({"GEOID_str": gids}), None))
    monkeypatch.setattr(ev, "_csa_city_outcomes", lambda *a, **k: pd.DataFrame(rows))
    monkeypatch.setattr(ev, "_load_panel_scores", lambda *a, **k: lab)
    monkeypatch.setattr(ev, "_csa_split_of",
                        lambda p: {g: ("heldout" if i % 2 else "train") for i, g in enumerate(gids)})

    class _Coh:
        cohorts = pd.DataFrame({"GEOID_str": gids, "cohort_year": [cohort[g] for g in gids]})
    monkeypatch.setattr(ces, "build_tract_cohorts", lambda *a, **k: _Coh())
    t = ev.part_f_change_reliability(tmp_path / "res", proc, out, years=NYC_YEARS,
                                     wealth_long=wl, se_long=se_long, n_draws=30, n_boot=30)
    t = t.set_index(["group", "sample"])
    assert set(t.index.get_level_values("group")) == {"stable", "change 1%", "change 5%",
                                                       "change 10%"}
    ch = t.loc[("change 5%", "all")]
    assert ch["n"] == 150
    assert 0 < ch["rel_y"] < 1 and 0 < ch["rel_p"] < 1
    assert ch["r_disattenuated"] > ch["r"] > 0.3
    for f in ("figures/F_change_reliability.pdf", "tables/F_change_reliability.csv",
              "tables/F_change_reliability_pairs.csv", "tables/F_change_reliability_splits.csv",
              "tables/F_change_reliability_counts.csv"):
        assert (out / f).exists(), f
    assert not (out / "tables" / "F_construction_counts.csv").exists()
    pairs = pd.read_csv(out / "tables" / "F_change_reliability_pairs.csv")
    assert {"noise_var_dy", "d_half0_a", "d_half0_b", "in_stable"} <= set(pairs.columns)
    assert len(pd.read_csv(out / "tables" / "F_change_reliability_splits.csv")) == 6


def test_reliability_stats_without_splits_leaves_prediction_side_unidentified():
    dy, dp, nv, halves, *_ = _synthetic(n=500)
    st = cr.reliability_stats(dy, dp, nv, np.empty((0, 2, len(dy))))
    assert st["n"] == 500 and 0 < st["rel_y"] < 1 and np.isfinite(st["r"])
    for k in ("rel_p", "r_disattenuated", "signal_var_dp", "sd_noise_dp", "split_half_corr"):
        assert np.isnan(st[k]), k
    ci = cr.bootstrap_reliability(dy, dp, nv, np.empty((0, 2, len(dy))), n_boot=20)
    assert np.isfinite(ci["rel_y_ci_lo"]) and np.isnan(ci["rel_p_ci_lo"])


def test_part_f_change_reliability_other_pass_single_pre_flight(tmp_path, monkeypatch):
    """NAIP-style pass: flights 2017/2019/2021/2022, ACS 2018 -> 2023. The
    2014-2018 window holds one flight, so only the ACS side is identified."""
    from src import csa_event_study as ces
    out = tmp_path / "evaluation"
    (out / "tables").mkdir(parents=True)
    (out / "figures").mkdir(parents=True)
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / ces.CSA_CITIES["nyc"].footprints_filename).write_bytes(b"")
    rng = np.random.default_rng(11)
    gids = [f"36047{i:06d}" for i in range(300)]
    cohort = {g: (2020 if i < 100 else 0) for i, g in enumerate(gids)}
    true = {g: rng.normal(0.5, 0.4) if cohort[g] else rng.normal(0, 0.2) for g in gids}
    level = {g: rng.normal() for g in gids}
    flights = [2017, 2019, 2021, 2022]
    rows = [{"GEOID_str": g, "year": y,
             "pred_all": level[g] + (true[g] if y >= 2021 else 0) + rng.normal(0, 0.2)}
            for y in flights for g in gids]
    seen = {}
    def outcomes(csa_root, results_dir, processed_dir, spec, city_years, *a, **k):
        seen["dir"], seen["years"] = Path(results_dir), list(city_years)
        return pd.DataFrame(rows)
    lab = pd.DataFrame([{"GEOID": g, "year": y,
                         "label": level[g] + (true[g] if y == 2023 else 0) + rng.normal(0, 0.15)}
                        for y in (2018, 2023) for g in gids])
    wl = pd.DataFrame([{"GEOID": g, "cbsa": "35620", "year": y,
                        "usd": float(4e5 * np.exp(0.5 * (level[g] + (true[g] if y == 2023 else 0))))}
                       for y in (2018, 2023) for g in gids])
    se_long = pd.DataFrame([{"GEOID": g, "year": y, "se": 0.15} for y in (2014, 2023) for g in gids])
    monkeypatch.setattr(ev, "_csa_city_geometry",
                        lambda p, s: (None, pd.DataFrame({"GEOID_str": gids}), None))
    monkeypatch.setattr(ev, "_csa_city_outcomes", outcomes)
    monkeypatch.setattr(ev, "_load_panel_scores", lambda *a, **k: lab)
    monkeypatch.setattr(ev, "_csa_split_of", lambda p: {g: "heldout" for g in gids})

    class _Coh:
        cohorts = pd.DataFrame({"GEOID_str": gids, "cohort_year": [cohort[g] for g in gids]})
    monkeypatch.setattr(ces, "build_tract_cohorts", lambda *a, **k: _Coh())
    asked = {}
    def load_se(processed_dir, indicator, years):
        asked["years"] = list(years)
        return se_long
    monkeypatch.setattr(ev, "_load_rel_se_long", load_se)
    naip = tmp_path / "naip"
    t = ev.part_f_change_reliability(tmp_path / "res", proc, out, years=NYC_YEARS, acs_pre=2018,
                                     wealth_long=wl, n_draws=20, n_boot=20,
                                     pred_results_dir=naip, pred_years=flights,
                                     pred_tag="_naip").set_index(["group", "sample"])
    assert seen == {"dir": naip, "years": flights}
    assert 2014 in asked["years"]   # ACS 2018 must be able to borrow the nearer 2014 SE
    row = t.loc[("change 5%", "all")]
    assert row["n"] == 100 and 0 < row["rel_y"] < 1 and row["r"] > 0.3
    assert np.isnan(row["rel_p"]) and np.isnan(row["r_disattenuated"])
    assert (out / "figures" / "F_change_reliability_naip.pdf").exists()
    pairs = pd.read_csv(out / "tables" / "F_change_reliability_pairs_naip.csv")
    assert not any(c.startswith("d_half") for c in pairs.columns)
    sc = ev.part_f_construction_scatter(tmp_path / "res", proc, out, years=NYC_YEARS, acs_pre=2018,
                                        wealth_long=wl, n_boot=5, pred_results_dir=naip,
                                        pred_years=flights, pred_tag="_naip")
    assert (out / "tables" / "F_construction_pairs_naip.csv").exists()
    p = pd.read_csv(out / "tables" / "F_construction_pairs_naip.csv")
    assert (p["n_pre"] == 1).all() and (p["n_post"] == 3).all()
    with pytest.raises(ValueError):
        ev.part_f_construction_scatter(tmp_path / "res", proc, out, years=NYC_YEARS, acs_pre=2018,
                                       wealth_long=wl, pred_results_dir=naip, pred_years=flights,
                                       weighting="volume", heights=pd.DataFrame())


def test_flight_span_labels():
    assert ev._flight_span([2017]) == "2017"
    assert ev._flight_span([2014, 2016, 2018]) == "2014--2018"
