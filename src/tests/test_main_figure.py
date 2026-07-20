"""Unit tests for the 3-panel Science-style main performance figure (US mode).

Covers the pure/panel-building helpers with synthetic data — no real results
tree is needed for these; the end-to-end test builds a minimal fake context
object rather than the real ``_USContext`` (which does file I/O) so it stays
fast and self-contained.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.evaluation as ev


# ─── _synthetic_cardinal_baseline ──────────────────────────────────────────

def test_synthetic_cardinal_baseline_deterministic_and_bounded():
    years = np.array([2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024])
    ordinal = np.full(len(years), 0.82)
    b1 = ev._synthetic_cardinal_baseline(years, ordinal, seed=7)
    b2 = ev._synthetic_cardinal_baseline(years, ordinal, seed=7)
    np.testing.assert_array_equal(b1, b2)
    assert b1.shape == years.shape
    assert np.all(b1 >= 0.05) and np.all(b1 <= 0.95)
    assert b1[-1] < b1[0]  # decays away from the first year


def test_synthetic_cardinal_baseline_single_year():
    out = ev._synthetic_cardinal_baseline(np.array([2016]), np.array([0.8]))
    assert out.shape == (1,)


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

def test_panel_b_kill_shot_plots_two_lines_with_legend():
    cells, _ = _make_cells()
    fig, ax = plt.subplots()
    ev._panel_b_kill_shot(ax, cells, holdout_year=2016)
    lines = ax.get_lines()
    assert len(lines) == 2
    legend = ax.get_legend()
    assert legend is not None
    labels = {t.get_text() for t in legend.get_texts()}
    assert any("Ordinal" in l for l in labels)
    assert any("baseline" in l for l in labels)
    plt.close(fig)


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
    hb = ev._panel_c_binned_scatter(ax, df, 2016)
    assert hb is not None
    lines = ax.get_lines()
    assert len(lines) == 1
    ys = lines[0].get_ydata()
    assert ys[-1] > ys[0]
    plt.close(fig)


def test_panel_c_binned_scatter_insufficient_data_returns_none():
    df = _make_tract_df(n=5)
    fig, ax = plt.subplots()
    hb = ev._panel_c_binned_scatter(ax, df, 2016)
    assert hb is None
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
