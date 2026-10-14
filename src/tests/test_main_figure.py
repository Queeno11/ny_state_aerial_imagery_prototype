"""Unit tests for the 3-panel Science-style main performance figure (US mode).

Covers the pure/panel-building helpers with synthetic data — no real results
tree is needed for these; the end-to-end test builds a minimal fake context
object rather than the real ``_USContext`` (which does file I/O) so it stays
fast and self-contained.
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.evaluation as ev


# ─── Panel A: raincloud ─────────────────────────────────────────────────────

def _make_cells(seed=0):
    rng = np.random.default_rng(seed)
    cbsas = {"10420": "small", "35620": "mega", "41860": "large", "99999": "medium"}
    rows = []
    for cbsa in cbsas:
        for yr in (2010, 2012, 2014, 2016):
            rows.append({"cbsa": cbsa, "year": yr, "n": 20,
                        "rho": float(np.clip(rng.normal(0.80, 0.05), -1, 1))})
    return pd.DataFrame(rows), cbsas


def test_panel_a_raincloud_orders_present_brackets_only():
    cells, cbsas = _make_cells()
    fig, ax = plt.subplots()
    ev._panel_a_raincloud(ax, cells, cbsas)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["Mega", "Large", "Medium", "Small"]
    plt.close(fig)


def test_panel_a_raincloud_drops_unmapped_cbsa():
    cells, cbsas = _make_cells()
    extra = pd.DataFrame([{"cbsa": "00000", "year": 2010, "n": 20, "rho": 0.5}])
    cells = pd.concat([cells, extra], ignore_index=True)
    fig, ax = plt.subplots()
    ev._panel_a_raincloud(ax, cells, cbsas)
    labels = {t.get_text() for t in ax.get_xticklabels()}
    assert labels == {"Mega", "Large", "Medium", "Small"}
    plt.close(fig)


def test_panel_a_raincloud_empty_data_does_not_crash():
    fig, ax = plt.subplots()
    ev._panel_a_raincloud(ax, pd.DataFrame(columns=["cbsa", "year", "n", "rho"]), {})
    plt.close(fig)


# ─── Panel B: kill shot ─────────────────────────────────────────────────────

def test_panel_b_kill_shot_without_baseline_plots_only_the_ordinal_line():
    cells, _ = _make_cells()
    fig, ax = plt.subplots()
    ev._panel_b_kill_shot(ax, cells, holdout_year=2016)
    assert len(ax.get_lines()) == 1
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert any("Ordinal" in l for l in labels)
    assert not any("baseline" in l for l in labels)   # no synthetic stand-in
    plt.close(fig)


def test_panel_b_kill_shot_plots_measured_baseline_with_bands():
    cells, _ = _make_cells()
    base, _ = _make_cells(seed=5)
    base["rho"] -= 0.2
    fig, ax = plt.subplots()
    ev._panel_b_kill_shot(ax, cells, holdout_year=2016, baseline_cells=base, bands=True)
    lines = ax.get_lines()
    assert len(lines) == 2
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert any("Ordinal" in l for l in labels) and any("L2 baseline" in l for l in labels)
    by_label = {l.get_label(): l for l in lines}
    ord_y = by_label["Ordinal model (this paper)"].get_ydata()
    base_y = by_label["Cardinal L2 baseline"].get_ydata()
    assert np.all(np.asarray(base_y) < np.asarray(ord_y))   # the measured curve, not a stand-in
    assert np.asarray(base_y) == pytest.approx(ev._yearly_weighted_rho(base)["rho"].to_numpy())
    assert len(ax.collections) >= 2                      # one band per line
    plt.close(fig)


def test_yearly_weighted_rho_and_band():
    cells = pd.DataFrame([
        {"cbsa": "A", "year": 2020, "n": 5, "n_tracts": 10, "rho": 0.2},
        {"cbsa": "B", "year": 2020, "n": 5, "n_tracts": 30, "rho": 0.6},
        {"cbsa": "A", "year": 2021, "n": 5, "n_tracts": 10, "rho": 0.5},
    ])
    curve = ev._yearly_weighted_rho(cells)
    assert curve["rho"].tolist() == pytest.approx([0.5, 0.5])
    band = ev._yearly_rho_band(cells, n_boot=200)
    lo, hi = band.loc[band["year"] == 2020, ["lo", "hi"]].iloc[0]
    assert 0.2 - 1e-9 <= lo <= 0.5 <= hi <= 0.6 + 1e-9
    assert band.loc[band["year"] == 2021, ["lo", "hi"]].isna().all(axis=None)  # 1 city


def test_panel_b_kill_shot_empty_data_does_not_crash():
    fig, ax = plt.subplots()
    ev._panel_b_kill_shot(ax, pd.DataFrame(columns=["cbsa", "year", "n", "rho"]))
    plt.close(fig)


# ─── Panel C: binned scatter ────────────────────────────────────────────────

def _make_tract_df(n=300, year=2016, seed=1):
    rng = np.random.default_rng(seed)
    label = rng.normal(0, 1, n)
    pred = label + rng.normal(0, 0.35, n)
    return pd.DataFrame({"year": year, "label": label, "pred": pred})


def test_panel_c_binned_scatter_returns_hexbin_and_monotonic_means():
    df = _make_tract_df()
    fig, ax = plt.subplots()
    hb, rho, note = ev._panel_c_binned_scatter(ax, df, 2016)
    assert hb is not None
    assert rho is not None and 0.5 < rho <= 1.0  # _make_tract_df's pred is label + small noise
    lines = ax.get_lines()
    assert len(lines) == 1
    ys = lines[0].get_ydata()
    assert ys[-1] > ys[0]
    plt.close(fig)


def test_panel_c_binned_scatter_insufficient_data_returns_none():
    df = _make_tract_df(n=5)
    fig, ax = plt.subplots()
    hb, rho, note = ev._panel_c_binned_scatter(ax, df, 2016)
    assert hb is None
    assert rho is None
    assert note is None
    plt.close(fig)


def test_panel_c_binned_scatter_clips_axes_to_central_range_and_reports_outliers():
    # Clipping is opt-in (central_frac < 1.0). When requested, a handful of
    # extreme outliers should be excluded from the axis limits and counted in
    # the returned note, without changing the panel type.
    rng = np.random.default_rng(2)
    n_core = 300
    label = rng.normal(0, 1, n_core)
    pred = label + rng.normal(0, 0.2, n_core)
    core = pd.DataFrame({"year": 2016, "label": label, "pred": pred})
    outliers = pd.DataFrame({"year": 2016, "label": [50.0, -50.0], "pred": [50.0, -50.0]})
    df = pd.concat([core, outliers], ignore_index=True)

    fig, ax = plt.subplots()
    hb, rho, note = ev._panel_c_binned_scatter(ax, df, 2016, central_frac=0.98)
    assert hb is not None
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    assert xhi < 50 and xlo > -50
    assert yhi < 50 and ylo > -50
    assert note is not None
    assert f"{n_core + 2:,}" in note  # denominator is the full pooled n
    plt.close(fig)


def test_panel_c_binned_scatter_shows_full_range_by_default():
    # Default (central_frac == 1.0): nothing is trimmed, so the axes must span
    # the extreme points and no outlier note is returned.
    rng = np.random.default_rng(2)
    n_core = 300
    label = rng.normal(0, 1, n_core)
    pred = label + rng.normal(0, 0.2, n_core)
    core = pd.DataFrame({"year": 2016, "label": label, "pred": pred})
    outliers = pd.DataFrame({"year": 2016, "label": [50.0, -50.0], "pred": [50.0, -50.0]})
    df = pd.concat([core, outliers], ignore_index=True)

    fig, ax = plt.subplots()
    hb, rho, note = ev._panel_c_binned_scatter(ax, df, 2016)
    assert hb is not None
    assert note is None
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    assert xlo <= -50.0 and xhi >= 50.0
    assert ylo <= -50.0 and yhi >= 50.0
    plt.close(fig)


def test_panel_c_binned_scatter_pools_all_years_by_default():
    # year=None (the default) must pool every year in the frame, not just one.
    df = pd.concat([
        _make_tract_df(n=100, year=2010, seed=1),
        _make_tract_df(n=100, year=2016, seed=2),
        _make_tract_df(n=100, year=2022, seed=3),
    ], ignore_index=True)
    fig, ax = plt.subplots()
    hb, rho, note = ev._panel_c_binned_scatter(ax, df)
    assert hb is not None
    assert rho is not None
    assert hb.get_offsets().shape[0] <= 300  # scatter sees the pooled set (minus clipped-out pts)
    text = ax.texts[0].get_text()
    assert "300" in text and "tract-years" in text
    plt.close(fig)


def test_panel_c_binned_scatter_explicit_year_still_single_cross_section():
    df = pd.concat([
        _make_tract_df(n=100, year=2010, seed=1),
        _make_tract_df(n=100, year=2016, seed=2),
    ], ignore_index=True)
    fig, ax = plt.subplots()
    hb, rho, note = ev._panel_c_binned_scatter(ax, df, year=2016)
    assert hb is not None
    assert rho is not None
    text = ax.texts[0].get_text()
    assert "100" in text and "tracts" in text and "tract-years" not in text
    plt.close(fig)


def test_panel_c_binned_scatter_no_outliers_at_full_central_range():
    # central_frac=1.0 clips to [min, max] exactly, so nothing is excluded.
    df = _make_tract_df(n=500, seed=9)
    fig, ax = plt.subplots()
    hb, rho, note = ev._panel_c_binned_scatter(ax, df, 2016, central_frac=1.0)
    assert hb is not None
    assert rho is not None
    assert note is None
    plt.close(fig)


# ─── part_main_figure_us end-to-end ─────────────────────────────────────────

class _FakeCtx:
    """Minimal stand-in for _USContext exposing only what the figure reads."""

    def __init__(self, bld, tract, cbsa_meta, out):
        self.bld = bld
        self.tract = tract
        self.cbsa_meta = cbsa_meta
        self.out = out


def test_part_main_figure_us_end_to_end(tmp_path):
    out = tmp_path / "evaluation"
    (out / "figures").mkdir(parents=True)
    (out / "tables").mkdir(parents=True)

    cbsas = {"10420": "small", "35620": "mega", "41860": "large", "99999": "medium"}
    cbsa_meta = pd.DataFrame({"cbsa_code": list(cbsas.keys()), "bracket": list(cbsas.values())})

    rng = np.random.default_rng(3)
    bld_rows, tract_rows = [], []
    for cbsa in cbsas:
        for yr in (2010, 2012, 2014, 2016):
            for i in range(20):
                label = rng.normal(0, 1)
                pred = label + rng.normal(0, 0.3)
                bld_rows.append({"cbsa": cbsa, "year": yr, "type": "test",
                                 "pred": pred, "label": label,
                                 "building_id": f"{cbsa}_{yr}_{i}"})
                tract_rows.append({"cbsa": cbsa, "year": yr,
                                   "GEOID": f"{cbsa}{i:05d}", "pred": pred, "label": label})
    bld = pd.DataFrame(bld_rows)
    tract = pd.DataFrame(tract_rows)

    ctx = _FakeCtx(bld, tract, cbsa_meta, out)
    headline = ev.part_main_figure_us(ctx)

    assert headline["main_figure/n_test_cities"] == 4
    assert (out / "figures" / "US_main_performance_figure.pdf").exists()
    # Panel C pools every eligible year by default (4 years x 4 cities x 20
    # tracts), not a single cross-section.
    assert "main_figure/panel_c_year" not in headline
    assert headline["main_figure/panel_c_n_years"] == 4
    assert headline["main_figure/panel_c_n_tract_years"] == 4 * 4 * 20
    # Pooled rho (as plotted) plus one rho per individual year, so the pooled
    # number can be checked against the single-year picture it's built from.
    assert -1.0 <= headline["main_figure/panel_c_rho"] <= 1.0
    for yr in (2010, 2012, 2014, 2016):
        key = f"main_figure/panel_c_rho_year_{yr}"
        assert key in headline
        assert -1.0 <= headline[key] <= 1.0


def test_part_main_figure_us_explicit_year_uses_single_cross_section(tmp_path):
    out = tmp_path / "evaluation"
    (out / "figures").mkdir(parents=True)
    (out / "tables").mkdir(parents=True)

    cbsas = {"10420": "small", "35620": "mega", "41860": "large", "99999": "medium"}
    cbsa_meta = pd.DataFrame({"cbsa_code": list(cbsas.keys()), "bracket": list(cbsas.values())})

    rng = np.random.default_rng(3)
    bld_rows, tract_rows = [], []
    for cbsa in cbsas:
        for yr in (2010, 2012, 2014, 2016):
            for i in range(20):
                label = rng.normal(0, 1)
                pred = label + rng.normal(0, 0.3)
                bld_rows.append({"cbsa": cbsa, "year": yr, "type": "test",
                                 "pred": pred, "label": label,
                                 "building_id": f"{cbsa}_{yr}_{i}"})
                tract_rows.append({"cbsa": cbsa, "year": yr,
                                   "GEOID": f"{cbsa}{i:05d}", "pred": pred, "label": label})
    bld = pd.DataFrame(bld_rows)
    tract = pd.DataFrame(tract_rows)

    ctx = _FakeCtx(bld, tract, cbsa_meta, out)
    headline = ev.part_main_figure_us(ctx, year=2016)

    assert headline["main_figure/panel_c_year"] == 2016
    assert "main_figure/panel_c_n_years" not in headline
    # rho-by-year is reported regardless of which mode Panel C itself is in.
    assert -1.0 <= headline["main_figure/panel_c_rho"] <= 1.0
    for yr in (2010, 2012, 2014, 2016):
        assert f"main_figure/panel_c_rho_year_{yr}" in headline


def test_part_main_figure_us_panels_a_and_b_use_tract_level_rho(tmp_path, monkeypatch):
    """Panels A and B must be fed tract-level cells (one row per tract), not
    building-level cells (one row per building, tract label repeated) — the
    two diverge here on purpose so a regression to building-level data would
    be caught. (Panel B was building-level until it was aligned with A: the
    label only exists per tract, so building-level rho is attenuated by
    within-tract spread.)"""
    out = tmp_path / "evaluation"
    (out / "figures").mkdir(parents=True)
    (out / "tables").mkdir(parents=True)

    cbsas = {"10420": "small", "35620": "mega", "41860": "large", "99999": "medium"}
    cbsa_meta = pd.DataFrame({"cbsa_code": list(cbsas.keys()), "bracket": list(cbsas.values())})

    rng = np.random.default_rng(5)
    bld_rows, tract_rows = [], []
    for cbsa in cbsas:
        for yr in (2010, 2012, 2014, 2016):
            tract_labels = rng.normal(0, 1, 25)
            # Tract-level prediction: near-perfect rank agreement with label.
            tract_preds = tract_labels + rng.normal(0, 0.02, 25)
            for i, (label, tpred) in enumerate(zip(tract_labels, tract_preds)):
                tract_rows.append({"cbsa": cbsa, "year": yr,
                                   "GEOID": f"{cbsa}{i:05d}", "pred": tpred, "label": label})
                # Building-level predictions for the same tract carry a
                # deliberately unrelated (randomized) value, so the
                # building-level rho for this cell is near zero.
                for b in range(4):
                    bld_rows.append({"cbsa": cbsa, "year": yr, "type": "test",
                                     "pred": rng.normal(0, 1), "label": label,
                                     "building_id": f"{cbsa}_{yr}_{i}_{b}"})
    bld = pd.DataFrame(bld_rows)
    tract = pd.DataFrame(tract_rows)

    captured = {}
    orig_panel_a = ev._panel_a_raincloud
    orig_panel_b = ev._panel_b_kill_shot

    def spy_panel_a(ax, cells, bracket_of):
        captured["a"] = cells.copy()
        return orig_panel_a(ax, cells, bracket_of)

    def spy_panel_b(ax, cells, holdout_year=None, **kw):
        captured["b"] = cells.copy()
        return orig_panel_b(ax, cells, holdout_year=holdout_year, **kw)

    monkeypatch.setattr(ev, "_panel_a_raincloud", spy_panel_a)
    monkeypatch.setattr(ev, "_panel_b_kill_shot", spy_panel_b)

    ctx = _FakeCtx(bld, tract, cbsa_meta, out)
    ev.part_main_figure_us(ctx)

    assert captured["a"]["rho"].mean() > 0.9   # tract-level: near-perfect
    assert captured["b"]["rho"].mean() > 0.9   # tract-level too, same cells as A
    pd.testing.assert_frame_equal(captured["a"].reset_index(drop=True),
                                  captured["b"].reset_index(drop=True))


def test_part_main_figure_us_drops_cities_below_min_tracts(tmp_path):
    out = tmp_path / "evaluation"
    (out / "figures").mkdir(parents=True)
    (out / "tables").mkdir(parents=True)

    # Three cities at/above the 20-tract floor, one small city below it.
    cbsas = {"10420": "small", "35620": "mega", "41860": "large", "99999": "medium"}
    n_tracts_by_cbsa = {"10420": 20, "35620": 20, "41860": 20, "99999": 5}
    cbsa_meta = pd.DataFrame({"cbsa_code": list(cbsas.keys()), "bracket": list(cbsas.values())})

    rng = np.random.default_rng(3)
    bld_rows, tract_rows = [], []
    for cbsa, n_tracts in n_tracts_by_cbsa.items():
        for yr in (2010, 2012, 2014, 2016):
            for i in range(n_tracts):
                label = rng.normal(0, 1)
                pred = label + rng.normal(0, 0.3)
                bld_rows.append({"cbsa": cbsa, "year": yr, "type": "test",
                                 "pred": pred, "label": label,
                                 "building_id": f"{cbsa}_{yr}_{i}"})
                tract_rows.append({"cbsa": cbsa, "year": yr,
                                   "GEOID": f"{cbsa}{i:05d}", "pred": pred, "label": label})
    bld = pd.DataFrame(bld_rows)
    tract = pd.DataFrame(tract_rows)

    ctx = _FakeCtx(bld, tract, cbsa_meta, out)
    headline = ev.part_main_figure_us(ctx)

    # The 5-tract city ("99999") must be excluded from all three panels.
    assert headline["main_figure/n_test_cities"] == 3
    assert (out / "figures" / "US_main_performance_figure.pdf").exists()


def test_part_main_figure_us_drops_only_the_thin_year_not_whole_city(tmp_path, monkeypatch):
    """The tract-count floor is a per-(CBSA, year) cell filter, not a
    whole-city one: a city with plenty of tracts most years but one sparse
    year should lose only that year's cell, not disappear from every panel.
    (Filtering by total tracts across all years would wrongly keep or drop
    the whole city instead of just the offending cell.)"""
    out = tmp_path / "evaluation"
    (out / "figures").mkdir(parents=True)
    (out / "tables").mkdir(parents=True)

    cbsas = {"10420": "small", "35620": "mega", "41860": "large"}
    cbsa_meta = pd.DataFrame({"cbsa_code": list(cbsas.keys()), "bracket": list(cbsas.values())})
    # "10420" has 100 tracts total across 4 years (way above any whole-city
    # floor) but only 5 in 2016 specifically.
    n_tracts_by_cbsa_year = {
        ("10420", 2010): 35, ("10420", 2012): 35, ("10420", 2014): 25, ("10420", 2016): 5,
        ("35620", 2010): 20, ("35620", 2012): 20, ("35620", 2014): 20, ("35620", 2016): 20,
        ("41860", 2010): 20, ("41860", 2012): 20, ("41860", 2014): 20, ("41860", 2016): 20,
    }

    rng = np.random.default_rng(11)
    bld_rows, tract_rows = [], []
    for (cbsa, yr), n_tracts in n_tracts_by_cbsa_year.items():
        for i in range(n_tracts):
            label = rng.normal(0, 1)
            pred = label + rng.normal(0, 0.3)
            bld_rows.append({"cbsa": cbsa, "year": yr, "type": "test",
                             "pred": pred, "label": label,
                             "building_id": f"{cbsa}_{yr}_{i}"})
            tract_rows.append({"cbsa": cbsa, "year": yr,
                               "GEOID": f"{cbsa}{i:05d}", "pred": pred, "label": label})
    bld = pd.DataFrame(bld_rows)
    tract = pd.DataFrame(tract_rows)

    captured = {}
    orig_panel_a = ev._panel_a_raincloud

    def spy_panel_a(ax, cells, bracket_of):
        captured["a"] = cells.copy()
        return orig_panel_a(ax, cells, bracket_of)

    monkeypatch.setattr(ev, "_panel_a_raincloud", spy_panel_a)

    ctx = _FakeCtx(bld, tract, cbsa_meta, out)
    headline = ev.part_main_figure_us(ctx)

    # "10420" must still appear (it has 3 qualifying years), just without 2016.
    assert headline["main_figure/n_test_cities"] == 3
    a_cells = captured["a"]
    assert not ((a_cells["cbsa"] == "10420") & (a_cells["year"] == 2016)).any()
    assert ((a_cells["cbsa"] == "10420") & (a_cells["year"] == 2010)).any()


def test_part_main_figure_us_missing_bracket_meta_skips(tmp_path):
    out = tmp_path / "evaluation"
    (out / "figures").mkdir(parents=True)
    bld = pd.DataFrame({"cbsa": ["1"], "year": [2010], "type": ["test"],
                        "pred": [0.1], "label": [0.1]})
    tract = pd.DataFrame({"cbsa": ["1"], "year": [2010], "GEOID": ["1"],
                          "pred": [0.1], "label": [0.1]})
    ctx = _FakeCtx(bld, tract, None, out)
    headline = ev.part_main_figure_us(ctx)
    assert headline == {}
    assert not (out / "figures" / "US_main_performance_figure.pdf").exists()


def test_part_main_figure_us_empty_bld_skips(tmp_path):
    out = tmp_path / "evaluation"
    (out / "figures").mkdir(parents=True)
    empty = pd.DataFrame(columns=["cbsa", "year", "type", "pred", "label"])
    ctx = _FakeCtx(empty, empty, None, out)
    headline = ev.part_main_figure_us(ctx)
    assert headline == {}


# ─── Panel C: GB2 cardinal reconstruction ───────────────────────────────────

def _gb2_world(n_tracts=80, years=(2016, 2017), seed=11, noise=0.3):
    """Cities at very different wealth levels; tract truth is lognormal and the
    prediction is a noisy within-city signal (no level information)."""
    rng = np.random.default_rng(seed)
    level = {"10420": 11.0, "35620": 12.5, "41860": 12.0, "99999": 11.5}
    wl, tr = [], []
    for cbsa, mu in level.items():
        z = rng.normal(0, 1, n_tracts)
        for yr in years:
            for i in range(n_tracts):
                geoid = f"{cbsa}{i:06d}"
                wl.append({"GEOID": geoid, "cbsa": cbsa, "year": yr,
                           "usd": float(np.exp(mu + 0.6 * z[i]))})
                tr.append({"GEOID": geoid, "cbsa": cbsa, "year": yr, "label": z[i],
                           "pred": z[i] + rng.normal(0, noise)})
    return pd.DataFrame(wl), pd.DataFrame(tr)


def test_gb2_mapped_tracts_preserves_within_city_order_and_beats_city_only():
    wl, tr = _gb2_world()
    params = ev._fit_city_gb2(wl, set(wl["cbsa"]))
    f = ev.gb2_mapped_tracts(tr, wl, params)
    assert len(f) == len(tr)
    for _, g in f.groupby(["cbsa", "year"]):
        assert g["city_usd"].nunique() == 1                        # constant benchmark
        assert np.array_equal(g["pred"].rank().values, g["pred_usd"].rank().values)
    s = ev.cardinal_fit_stats(f)
    assert s["between_share"] > 0.5                                # level gaps dominate here
    assert s["city_r2"] > 0.4                                      # survey alone is strong...
    assert s["r2"] > s["city_r2"] and s["rho"] > s["city_rho"]     # ...the model adds to it


def test_gb2_mapped_tracts_perfect_ranks_recover_dollars():
    wl, tr = _gb2_world(noise=0.0)
    f = ev.gb2_mapped_tracts(tr, wl, ev._fit_city_gb2(wl, set(wl["cbsa"])))
    assert ev.cardinal_fit_stats(f)["r2"] > 0.95


def test_gb2_mapped_tracts_uses_nearest_fitted_vintage_and_drops_unfitted():
    wl, tr = _gb2_world(years=(2016,))
    params = ev._fit_city_gb2(wl, {"10420", "35620"})
    tr = pd.concat([tr, tr.assign(year=2018)])                     # no 2018 survey -> use 2016
    f = ev.gb2_mapped_tracts(tr, wl, params)
    assert set(f["cbsa"]) == {"10420", "35620"}
    assert set(f["acs_year"]) == {2016} and set(f["year"]) == {2016, 2018}


def test_part_main_figure_us_panel_c_gb2_mode(tmp_path, monkeypatch):
    out = tmp_path / "evaluation"
    (out / "figures").mkdir(parents=True)
    (out / "tables").mkdir(parents=True)
    wl, tr = _gb2_world(n_tracts=40, years=(2016, 2017))
    cbsas = {"10420": "small", "35620": "mega", "41860": "large", "99999": "medium"}
    cbsa_meta = pd.DataFrame({"cbsa_code": list(cbsas), "bracket": list(cbsas.values())})
    bld = tr.assign(type="test", building_id=tr["GEOID"] + "_b")
    ctx = _FakeCtx(bld, tr, cbsa_meta, out)
    ctx.processed_dir, ctx.indicator = tmp_path, "W2_r5"
    monkeypatch.setattr(ev, "_load_wealth_dollars_long", lambda *a, **k: wl)
    head = ev.part_main_figure_us(ctx)
    assert head["main_figure/panel_c_mode"] == "gb2"
    assert head["main_figure/panel_c_r2"] > head["main_figure/panel_c_city_r2"]
    assert head["main_figure/panel_c_n"] == len(tr)
    assert (out / "figures" / "US_main_performance_figure.pdf").exists()


def test_part_main_figure_us_panel_c_falls_back_without_panel_dollars(tmp_path):
    out = tmp_path / "evaluation"
    (out / "figures").mkdir(parents=True)
    (out / "tables").mkdir(parents=True)
    _, tr = _gb2_world(n_tracts=30, years=(2016,))
    cbsas = {"10420": "small", "35620": "mega", "41860": "large", "99999": "medium"}
    cbsa_meta = pd.DataFrame({"cbsa_code": list(cbsas), "bracket": list(cbsas.values())})
    ctx = _FakeCtx(tr.assign(type="test", building_id=tr["GEOID"]), tr, cbsa_meta, out)
    assert ev.part_main_figure_us(ctx)["main_figure/panel_c_mode"] == "latent"
