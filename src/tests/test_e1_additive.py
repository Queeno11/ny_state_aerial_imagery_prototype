"""Synthetic tests for the additive-baseline E1: suite stage, exact diagnostic, deck figure."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SUITE = Path(__file__).resolve().parents[1] / "ordinal_mc_suite"
sys.path.insert(0, str(SUITE))

import e1_additive_exact as ex  # noqa: E402
from ordinal_mc.config import preset, validate  # noqa: E402
from ordinal_mc.design import cell_coefficients  # noqa: E402
from ordinal_mc.experiments import scenario  # noqa: E402

_spec = importlib.util.spec_from_file_location("presentation_figures", SUITE / "presentation_figures.py")
pf = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pf
_spec.loader.exec_module(pf)

CONTEXTS = np.array([[0.3, -1.2], [-0.8, 0.5], [1.5, 0.1]])
C = np.array([0, 0, 1, 1, 2, 2])
T = np.array([0, 1, 0, 1, 0, 1])


def _coef(stage, strength):
    s = scenario(positive_labels=True, psi_stage=stage, psi_strength=strength)
    return cell_coefficients(CONTEXTS, C, T, s)


def test_additive_stage_keeps_multiplier_and_scales_baseline():
    a_m, b_m, k_m = _coef("multiplier", 1.0)
    a, b, k = _coef("additive", 10.0)
    np.testing.assert_allclose(b, b_m)                      # b_ct unchanged
    np.testing.assert_allclose(a, ex.additive_baseline(CONTEXTS, C, T, 10.0))
    assert np.all(a_m == 0) and np.all(k == 1) and np.all(a > 0)
    a0, b0, _ = _coef("additive", 0.0)
    np.testing.assert_allclose(a0, 0)
    np.testing.assert_allclose(b0, b_m)


def test_additive_baseline_range_at_s10():
    a = ex.additive_baseline(np.array([[-50.0, 0], [50.0, 0]]), np.array([0, 1]), np.array([0, 1]), 10.0)
    assert 3.0 - 1e-9 <= a.min() and a.max() <= 10.0 + 1e-9


def test_validate_accepts_additive_and_rejects_unknown_stage():
    d = preset("smoke")
    d["experiment1"]["cases"] = [["multiplier", 1.0], ["additive", 10.0]]
    validate(d)
    d["experiment1"]["cases"] = [["additive_typo", 10.0]]
    with pytest.raises(ValueError, match="Invalid transform case"):
        validate(d)


def test_demean_and_log_fe_beta_recover_common_slope_with_cell_intercepts():
    rng = np.random.default_rng(0)
    cells = np.repeat(np.arange(4), 25)
    x = rng.normal(size=(100, 3))
    beta = np.array([0.5, -1.0, 2.0])
    y = x @ beta + np.array([3.0, -2.0, 7.0, 0.1])[cells]
    d = ex.demean_by_cell(x, cells)
    for cell in range(4):
        np.testing.assert_allclose(d[cells == cell].mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(ex.log_fe_beta(x, y, cells), beta, atol=1e-10)


def _records_and_fits():
    rows, fits = [], []
    for c in (8, 16):
        for rep in (1, 2, 3):
            fits.append(dict(cities=c, rep=rep, method="log_cell_fe", unseen_within_latent=0.03 + 0.001 * rep))
            fits.append(dict(cities=c, rep=rep, method="rank_common", unseen_within_latent=0.035))
            for s in (0.0, 10.0):
                rows.append(dict(cities=c, rep=rep, scale=s, risk=0.03 + 0.001 * rep + 0.002 * (s > 0)))
    return pd.DataFrame(rows), pd.DataFrame(fits)


def test_check_baseline_passes_on_match_and_fails_on_mismatch():
    records, fits = _records_and_fits()
    assert ex.check_baseline(records, fits) == pytest.approx(0)
    bad = fits.copy()
    bad.loc[bad.method == "log_cell_fe", "unseen_within_latent"] += 0.01
    with pytest.raises(ValueError, match="does not reproduce"):
        ex.check_baseline(records, bad)


def test_summarize_suite_schema_and_ranking_repeated_per_map():
    records, fits = _records_and_fits()
    s = ex.summarize(records, fits)
    assert {"method", "psi_stage", "psi_strength", "cities", "metric", "mean",
            "ci95_low", "ci95_high", "mc_se"} <= set(s.columns)
    rank = s[s.method == "rank_common"]
    assert set(rank.psi_stage) == {"multiplier", "additive"}
    assert rank["mean"].nunique() == 1                     # same fit under both maps
    fe = s[(s.method == "log_cell_fe") & (s.psi_stage == "additive")]
    assert (fe.psi_strength == 10.0).all()
    assert np.all(fe.ci95_low <= fe["mean"]) and np.all(fe["mean"] <= fe.ci95_high)


def test_paired_gap_sign():
    records, fits = _records_and_fits()
    p = ex.paired(records, fits)
    assert (p[p.scale == 10.0].gap_pp > p[p.scale == 0.0].gap_pp.values).all()


def test_fig_e1_additive_two_panels_and_missing_map():
    records, fits = _records_and_fits()
    s = ex.summarize(records, fits)
    fig = pf.fig_e1_additive(s, 10.0)
    assert len(fig.axes) == 2
    assert all(len(ax.get_lines()) == 2 for ax in fig.axes)  # log + FE and ranking
    with pytest.raises(ValueError, match="was this map run"):
        pf.fig_e1_additive(s, 25.0)


def test_e1_additive_run_prefers_trained_run(tmp_path):
    assert pf.e1_additive_run(tmp_path) is None
    for name in ("e1_additive_exact", "e1_additive"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.csv").write_text("x\n")
    assert pf.e1_additive_run(tmp_path).name == "e1_additive"
