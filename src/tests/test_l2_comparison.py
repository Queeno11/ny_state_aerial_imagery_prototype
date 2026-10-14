"""Unit tests for the ordinal-vs-L2 comparison in src/evaluation.py:
pairing/alignment, bracket table, city-distance slope, training-curve tables,
baseline resolution, and the 'l2' / main-figure parts end-to-end on fake
contexts (no real results tree).
"""
import json

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")

import src.evaluation as ev

CBSAS = {"10420": "small", "35620": "mega", "41860": "large", "99999": "medium"}
YEARS = (2016, 2017, 2018)


class _FakeCtx:
    """Stand-in for _USContext with what the comparison parts read."""

    def __init__(self, bld, tract, out, results_dir, baseline=None):
        self.bld, self.tract, self.out = bld, tract, out
        self.cbsa_meta = pd.DataFrame({"cbsa_code": list(CBSAS), "bracket": list(CBSAS.values())})
        self.results_dir = results_dir
        self.processed_dir = results_dir
        self.indicator = "W2_r5"
        self.baseline = baseline


def _frames(noise_by_cbsa, seed, n_tracts=25, drop=None):
    """Building + tract frames; prediction noise per city sets its rho."""
    rng = np.random.default_rng(seed)
    bld, tract = [], []
    for cbsa in CBSAS:
        for yr in YEARS:
            for i in range(n_tracts):
                if drop and (cbsa, yr, i) in drop:
                    continue
                label = np.sin(i + 3 * yr) + 0.1 * i / n_tracts   # same labels in both runs
                pred = label + rng.normal(0, noise_by_cbsa[cbsa])
                geoid = f"{cbsa}{i:06d}"
                bld.append({"cbsa": cbsa, "year": yr, "type": "test", "pred": pred,
                            "label": label, "building_id": f"b{geoid}", "GEOID": geoid})
                tract.append({"cbsa": cbsa, "year": yr, "GEOID": geoid,
                              "pred": pred, "label": label})
    return pd.DataFrame(bld), pd.DataFrame(tract)


def _ctx_pair(tmp_path, drop=None):
    out = tmp_path / "ordinal" / "evaluation"
    (out / "figures").mkdir(parents=True)
    (out / "tables").mkdir(parents=True)
    ord_noise = {c: 0.2 for c in CBSAS}
    # L2 degrades with distance from the (rich, mega) training level
    l2_noise = {"35620": 0.25, "41860": 0.5, "99999": 0.8, "10420": 1.2}
    b_ord, t_ord = _frames(ord_noise, seed=1)
    b_l2, t_l2 = _frames(l2_noise, seed=2, drop=drop)
    base_dir = tmp_path / "l2run"
    base_dir.mkdir()
    (base_dir / "level_label_stats.json").write_text(json.dumps({"mu": 12.0, "sd": 1.0}))
    base = _FakeCtx(b_l2, t_l2, out, base_dir)
    return _FakeCtx(b_ord, t_ord, out, tmp_path / "ordinal", baseline=base)


def _wealth_long(*_args, **_kw):
    """City wealth: mega at the train mean, the others progressively poorer."""
    level = {"35620": 12.0, "41860": 11.5, "99999": 11.0, "10420": 10.2}
    rows = [{"GEOID": f"{c}{i:06d}", "cbsa": c, "year": y, "usd": float(np.exp(level[c]))}
            for c in CBSAS for i in range(3) for y in range(2011, 2024)]
    return pd.DataFrame(rows)


# ─── pure helpers ────────────────────────────────────────────────────────────

def test_align_rows_keeps_only_shared_keys():
    a = pd.DataFrame({"GEOID": ["1", "2", "3"], "year": [2016, 2016, 2017], "v": [1, 2, 3]})
    b = pd.DataFrame({"GEOID": ["2", "3", "4"], "year": [2016, 2018, 2016], "v": [9, 9, 9]})
    a2, b2 = ev._align_rows(a, b, ["GEOID", "year"])
    assert a2["GEOID"].tolist() == ["2"] and b2["GEOID"].tolist() == ["2"]


def test_l2_bracket_table_weights_and_ci():
    paired = pd.DataFrame({
        "cbsa": ["10420", "10420", "35620", "41860"], "year": [2016, 2017, 2016, 2016],
        "n_tracts": [10, 30, 20, 20], "rho_ord": [0.8, 0.8, 0.9, 0.7], "rho_l2": [0.4, 0.6, 0.9, 0.5],
    })
    tab = ev.l2_bracket_table(paired, CBSAS).set_index("bracket")
    assert tab.loc["small", "rho_l2"] == pytest.approx((0.4 * 10 + 0.6 * 30) / 40)
    assert tab.loc["small", "diff"] == pytest.approx(0.8 - 0.55)
    assert tab.loc["all", "n_cities"] == 3
    lo, hi = tab.loc["all", ["diff_ci_lo", "diff_ci_hi"]]
    assert lo <= tab.loc["all", "diff"] <= hi
    assert np.isnan(tab.loc["mega", "diff_ci_lo"])          # one city -> no CI


def test_city_level_distance_in_train_sd_units():
    d = ev.city_level_distance(_wealth_long(), range(2016, 2019), mu=12.0, sd=0.5).set_index("cbsa")
    assert d.loc["35620", "dist_sd"] == pytest.approx(0.0)
    assert d.loc["10420", "dist_sd"] == pytest.approx((10.2 - 12.0) / 0.5)


def test_distance_slope_detects_rising_gap():
    city = pd.DataFrame({"cbsa": list("abcdef"), "dist_sd": [0, -0.5, 1, -1.5, 2, -2.5],
                         "diff": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]})
    fit = ev.distance_slope(city, n_boot=300)
    assert fit["slope"] == pytest.approx(0.1)
    assert fit["slope_ci_lo"] <= 0.1 <= fit["slope_ci_hi"]
    assert fit["spearman"] == pytest.approx(1.0)


def test_matched_budget_table_and_history_loader(tmp_path):
    pd.DataFrame({"epoch": [1, 2, 3, 3, 4], "selection/score": [0.1, 0.5, 0.2, 0.3, 0.9]}).to_csv(
        tmp_path / "wandb_history.csv", index=False)
    h = ev.load_selection_history(tmp_path)
    assert h["epoch"].tolist() == [1, 2, 3, 4] and h["score"].tolist()[2] == 0.3  # last wins
    base = pd.DataFrame({"epoch": [1, 2, 3], "score": [0.2, 0.4, 0.35]})
    tab = ev.matched_budget_table(h, base)
    ord_budget = tab[(tab["model"] == "ordinal") & tab["budget"].str.startswith("<=")].iloc[0]
    assert ord_budget["best_epoch"] == 2 and ord_budget["best_selection_score"] == 0.5
    full = tab[tab["budget"].str.startswith("full")].iloc[0]
    assert full["best_epoch"] == 4
    assert tab[tab["model"] == "l2"].iloc[0]["best_selection_score"] == 0.4
    assert ev.load_selection_history(tmp_path / "missing") is None


def test_resolve_l2_baseline_dir_explicit_pointer_and_missing(tmp_path):
    ordinal, base = tmp_path / "ord", tmp_path / "l2run"
    ordinal.mkdir()
    base.mkdir()
    assert ev.resolve_l2_baseline_dir(ordinal) is None                 # no pointer
    (ordinal / ev.L2_POINTER_FILENAME).write_text(json.dumps({"savename": "l2run"}))
    assert ev.resolve_l2_baseline_dir(ordinal) is None                 # no predictions yet
    (base / "2016_predictions.csv").write_text("x\n")
    assert ev.resolve_l2_baseline_dir(ordinal) == base
    assert ev.resolve_l2_baseline_dir(ordinal, "l2run") == base
    assert ev.resolve_l2_baseline_dir(ordinal, "other") is None


# ─── parts end-to-end ────────────────────────────────────────────────────────

def test_paired_samples_drop_rows_only_one_model_predicted(tmp_path):
    ctx = _ctx_pair(tmp_path, drop={("10420", 2016, 0), ("10420", 2016, 1)})
    s = ev._paired_test_samples(ctx, ctx.baseline)
    assert len(s["tract_ord"]) == len(s["tract_l2"]) == 4 * 3 * 25 - 2
    pairs = s["tract_ord"][["GEOID", "year"]].astype(str)
    assert not ((pairs["GEOID"] == "10420000000") & (pairs["year"] == "2016")).any()


def test_part_l2_comparison_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "_load_wealth_dollars_long", _wealth_long)
    ctx = _ctx_pair(tmp_path)
    head = ev.part_l2_comparison_us(ctx)
    out = ctx.out
    assert (out / "figures" / "US_L2_city_gap_vs_distance.pdf").exists()
    assert (out / "figures" / "US_L2_bracket_raincloud.pdf").exists()
    assert not (out / "figures" / "US_L2_comparison_figure.pdf").exists()
    for t in ("US_L2_paired_cells.csv", "US_L2_by_bracket.csv", "US_L2_city_distance.csv"):
        assert (out / "tables" / t).exists()
    assert head["l2/all/diff"] > 0                      # baseline noisier everywhere
    assert head["l2/small/diff"] > head["l2/mega/diff"]
    assert head["l2/distance_slope"] > 0                # gap grows with level distance
    assert head["l2/n_paired_cells"] == 4 * 3


def test_city_short_name_and_labels():
    assert ev._city_short_name("Dallas-Fort Worth-Arlington, TX") == "Dallas, TX"
    assert ev._city_short_name("Chicago-Naperville-Elgin, IL-IN") == "Chicago, IL"
    assert ev._city_short_name(None) is None
    city = pd.DataFrame({"cbsa": list("abcdefg"), "dist_sd": [0.1] * 7,
                         "diff": [0.3, 0.2, 0.1, 0.05, 0.04, 0.03, -0.01]})
    picks = ev._l2_cities_to_label(city, {"g": "small", "f": "mega"}, n_top=2)
    assert picks == ["a", "b", "g", "f"]       # top gaps, L2 wins, mega; no duplicates


def test_ols_band_brackets_the_fit():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 40)
    y = 0.1 * x + rng.normal(0, 0.02, 40)
    xs = np.linspace(0, 1, 5)
    lo, hi = ev._ols_band(x, y, xs, n_boot=300)
    fit = np.polyval(np.polyfit(x, y, 1), xs)
    assert np.all(lo <= fit) and np.all(fit <= hi)


def test_part_l2_comparison_skips_without_baseline(tmp_path):
    ctx = _ctx_pair(tmp_path)
    ctx.baseline = None
    assert ev.part_l2_comparison_us(ctx) == {}


def test_main_figure_uses_measured_baseline_when_linked(tmp_path):
    ctx = _ctx_pair(tmp_path)
    head = ev.part_main_figure_us(ctx)
    assert head["main_figure/l2_baseline"] == "l2run"
    for yr in YEARS:
        assert head[f"main_figure/panel_b_ordinal_rho_year_{yr}"] > \
               head[f"main_figure/panel_b_l2_rho_year_{yr}"]
    ctx.baseline = None
    head = ev.part_main_figure_us(ctx)
    assert "main_figure/l2_baseline" not in head


def test_part_l2_training_curves(tmp_path):
    ctx = _ctx_pair(tmp_path)
    pd.DataFrame({"epoch": range(1, 301), "selection/score": np.linspace(0.3, 0.7, 300)}).to_csv(
        ctx.results_dir / "wandb_history.csv", index=False)
    head = ev.part_l2_training_curves_us(ctx)
    assert head == {}                                   # baseline history not exported yet
    pd.DataFrame({"epoch": range(1, 201), "selection/score": np.linspace(0.2, 0.5, 200)}).to_csv(
        ctx.baseline.results_dir / "wandb_history.csv", index=False)
    head = ev.part_l2_training_curves_us(ctx)
    assert (ctx.out / "figures" / "US_L2_training_curves.pdf").exists()
    assert head["l2_budget/l2/<= epoch 200"] == pytest.approx(0.5)
    assert head["l2_budget/ordinal/<= epoch 200"] < head["l2_budget/ordinal/full run (300 epochs)"]
