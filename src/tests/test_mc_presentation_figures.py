"""Synthetic tests for src/ordinal_mc_suite/presentation_figures.py."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_MOD = Path(__file__).resolve().parents[1] / "ordinal_mc_suite" / "presentation_figures.py"
_spec = importlib.util.spec_from_file_location("presentation_figures", _MOD)
pf = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pf  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(pf)

CITIES = [8, 16, 32, 64]


def _rows(method, metric, base, **extra):
    """One summary row per C with a decreasing mean and a +/-10% band."""
    out = []
    for k, c in enumerate(CITIES):
        m = base / (k + 1)
        out.append(dict(cities=c, method=method, metric=metric, mean=m,
                        ci95_low=0.9 * m, ci95_high=1.1 * m, **extra))
    return out


def _e1_summary():
    rows = []
    for stage in ("multiplier", "shape"):
        for comp in (0.0, 0.7):
            for metric in ("unseen_within_latent", "unseen_cross_city_latent"):
                for meth, base in (("raw_levels", .07), ("log_cell_fe", .04), ("rank_common", .038)):
                    rows += _rows(meth, metric, base, psi_stage=stage, composition=comp)
    return pd.DataFrame(rows)


def _e3_summary(truths=("linear", "quadratic")):
    rows = []
    for truth in truths:
        for metric in ("unseen_within_latent", "unseen_across_time_latent",
                       "unseen_cross_city_latent"):
            rows += _rows("none", metric, .07, truth=truth)
            rows += _rows("temporal", metric, .05, truth=truth)
    return pd.DataFrame(rows)


def _e4_paired():
    rows = []
    for meth in ("temporal_fixed", "temporal_tuned"):
        for case in ("exact", "drift", "random_false", "systematic_false"):
            rows.append(dict(method=meth, link_case=case, metric="unseen_within_latent",
                             mean=.004, ci95_low=-.001, ci95_high=.009,
                             relative_reduction_of_mean=.2))
    return pd.DataFrame(rows)


def test_select_applies_all_filters():
    df = _e1_summary()
    out = pf.select(df, method="rank_common", psi_stage="shape", composition=0.7,
                    metric="unseen_within_latent")
    assert sorted(out["cities"]) == CITIES


def test_curve_sorts_and_scales_to_percent():
    df = pd.DataFrame(_rows("x", "m", .04)).iloc[::-1]
    c = pf.curve(df, "x", pf.INK)
    assert c.x == CITIES
    assert c.mean[0] == pytest.approx(4.0)
    assert all(lo <= m <= hi for lo, m, hi in zip(c.lo, c.mean, c.hi))


def test_curve_rejects_empty_and_duplicated_x():
    with pytest.raises(ValueError, match="no rows"):
        pf.curve(pd.DataFrame(columns=["cities", "mean", "ci95_low", "ci95_high"]), "e", pf.INK)
    dup = pd.DataFrame(_rows("x", "m", .04) + _rows("x", "m", .05))
    with pytest.raises(ValueError, match="duplicated"):
        pf.curve(dup, "d", pf.INK)


def test_fig_e1_has_two_panels_with_three_methods():
    fig = pf.fig_e1(_e1_summary(), 0.0)
    assert len(fig.axes) == 2
    assert all(len(ax.get_lines()) == 3 for ax in fig.axes)


def test_fig_links_supports_per_panel_metric_override():
    e3 = _e3_summary()
    fig = pf.fig_links([(e3, {"truth": "linear"}, "a"),
                        (e3, {"truth": "quadratic", "metric": "unseen_across_time_latent"}, "b")])
    assert len(fig.axes) == 2


def test_fig_e4_requires_exactly_one_row_per_bar():
    pf.fig_e4(_e4_paired())
    with pytest.raises(ValueError, match="expected one row"):
        pf.fig_e4(pd.concat([_e4_paired(), _e4_paired()]))


def _e3_paired_reps(truths=("linear", "quadratic"), reps=10):
    """Per-replication paired gains; half the reps improve, half do not."""
    rows = []
    for truth in truths:
        for c in CITIES:
            for r in range(reps):
                rows.append(dict(truth=truth, cities=c, rep=r, method="temporal",
                                 metric="unseen_within_latent",
                                 gain=0.01 if r % 2 == 0 else -0.002))
    return pd.DataFrame(rows)


def test_invariance_shift_is_zero_for_identical_stages():
    df = _e1_summary()
    df.loc[(df.psi_stage == "shape") & (df.method == "log_cell_fe"), "mean"] += 0.003
    shift = pf.invariance_shift(df)
    rank = pf.select(shift, method="rank_common")["shift_pp"]
    fe = pf.select(shift, method="log_cell_fe")["shift_pp"]
    assert len(shift) == 3 * len(CITIES)
    assert (rank == 0).all()
    assert fe.to_numpy() == pytest.approx([0.3] * len(CITIES))


def test_invariance_shift_requires_both_stages():
    df = _e1_summary()
    with pytest.raises(ValueError, match="lacks psi_stage"):
        pf.invariance_shift(df[df.psi_stage == "shape"])


def test_equivalent_cities_interpolates_on_log2_axis():
    x, y = [8, 16, 32, 64], [4.0, 3.0, 2.0, 1.0]
    assert pf.equivalent_cities(x, y, 3.0) == pytest.approx(16)
    assert pf.equivalent_cities(x, y, 2.5) == pytest.approx(16 * 2 ** 0.5)
    assert pf.equivalent_cities(x, y, 0.5) is None
    assert pf.equivalent_cities(x, y, 5.0) is None


def test_fig_doubling_draws_two_curves():
    fig = pf.fig_doubling(_e3_summary(("linear",)))
    assert len(fig.axes[0].get_lines()) == 2


def test_fig_paired_strip_annotates_positive_share():
    fig = pf.fig_paired_strip([(_e3_paired_reps(), {"truth": "linear"}, "a"),
                               (_e3_paired_reps(), {"truth": "quadratic"}, "b")])
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert texts == ["50%"] * len(CITIES)
    with pytest.raises(ValueError, match="no paired rows"):
        pf.fig_paired_strip([(_e3_paired_reps(), {"truth": "smooth"}, "c")])


def test_build_all_writes_every_figure(tmp_path):
    runs = {}
    files = {"E1": [("summary.csv", _e1_summary())],
             "E3_nn": [("summary.csv", _e3_summary()),
                       ("paired_replications.csv", _e3_paired_reps())],
             "E3_linear": [("summary.csv", _e3_summary(("linear",))),
                           ("paired_replications.csv", _e3_paired_reps(("linear",)))],
             "E4": [("paired_summary.csv", _e4_paired())]}
    for key, items in files.items():
        d = tmp_path / key
        d.mkdir()
        for name, df in items:
            df.to_csv(d / name, index=False)
        runs[key] = d
    # the linear run has no truth filter in build_all: give it a single truth
    written = pf.build_all(tmp_path / "out", runs)
    assert len(written) == 10
    assert all(p.exists() and p.stat().st_size > 0 for p in written)
