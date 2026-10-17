"""Synthetic tests for src/consistent_change.py."""

import numpy as np
import pandas as pd
import pytest

from src import consistent_change as cc


def test_max_drawdown_and_drawup():
    s = np.array([[0.0, 1.0, 0.7, 1.5],      # dips 0.3 below running max
                  [0.0, 0.1, 0.2, 0.3],      # monotone up
                  [np.nan, 1.0, np.nan, 0.4],
                  [np.nan, np.nan, np.nan, np.nan]])
    dd = cc.max_drawdown(s)
    np.testing.assert_allclose(dd[:3], [0.3, 0.0, 0.6])
    assert np.isnan(dd[3])
    np.testing.assert_allclose(cc.max_drawup(np.array([[1.0, 0.0, 0.5, -1.0]])), [0.5])


def _traj():
    idx = ["up_clean", "up_reverts", "up_insig", "down_clean", "flat"]
    scores = pd.DataFrame({2013: [0.0, 0.0, 0.0, 1.0, 0.5],
                           2018: [0.5, 0.9, 0.2, 0.5, 0.5],
                           2023: [1.0, 0.8, 0.4, 0.0, 0.5]}, index=idx)
    diff = pd.Series([1.0, 0.8, 0.4, -1.0, 0.0], index=idx)
    pval = pd.Series([0.001, 0.001, 0.30, 0.001, 0.9], index=idx)
    return scores, diff, pval


def test_consistent_direction_independent_rule():
    scores, diff, pval = _traj()
    d = cc.consistent_direction(scores, diff, pval, alpha=0.05)
    assert d.to_dict() == {"up_clean": "up", "up_reverts": "none", "up_insig": "none",
                           "down_clean": "down", "flat": "none"}


def test_consistent_direction_tolerance_rule():
    scores, diff, pval = _traj()
    se = pd.Series(0.05, index=scores.index)
    strict = cc.consistent_direction(scores, diff, pval, se=se, rule="tolerance", tol_se=1.0)
    assert strict["up_reverts"] == "none"          # 0.1 dip > 0.05
    loose = cc.consistent_direction(scores, diff, pval, se=se, rule="tolerance", tol_se=2.0)
    assert loose["up_reverts"] == "up"             # 0.1 dip <= 0.10
    with pytest.raises(ValueError):
        cc.consistent_direction(scores, diff, pval, rule="tolerance")
    with pytest.raises(ValueError):
        cc.consistent_direction(scores, diff, pval, rule="other")


def test_auc_known_values():
    assert cc.auc([3, 4], [1, 2]) == 1.0
    assert cc.auc([1, 2], [3, 4]) == 0.0
    assert cc.auc([1, 2], [1, 2]) == 0.5
    assert np.isnan(cc.auc([], [1.0]))


def _pairs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    g = [f"36061{i:06d}" for i in range(n)]
    built = np.arange(n) < 80
    stable = ~built
    up = (np.arange(n) % 4 == 0)                     # 50 tracts consistently rise
    lab_pre = rng.uniform(size=n)
    lab_post = np.clip(lab_pre + np.where(up, 0.2, 0.0) + rng.normal(0, 0.02, n), 0, 1)
    pred_pre = rng.uniform(size=n)
    pred_post = np.clip(pred_pre + np.where(up & built, 0.1, 0.0) + rng.normal(0, 0.02, n), 0, 1)
    p = pd.DataFrame({"GEOID": g, "label_prob_pre": lab_pre, "label_prob_post": lab_post,
                      "pred_prob_pre": pred_pre, "pred_prob_post": pred_post,
                      "dlog_label_usd": lab_post - lab_pre, "dlog_pred_usd": pred_post - pred_pre,
                      "in_stable": stable, "in_change_1pct": built, "in_change_5pct": built,
                      "in_change_10pct": built,
                      "split_group": np.where(np.arange(n) % 2, "heldout", "train")})
    direction = pd.Series(np.where(up, "up", "none"), index=g)
    return p, direction


def test_group_summary_and_table_on_synthetic_signal():
    p, direction = _pairs()
    t = cc.consistent_change_table(p, direction, n_boot=50).set_index(["group", "sample"])
    main = t.loc[("up & built>=5%", "all")]
    assert main["n"] == 20 and main["n_train"] + main["n_heldout"] == 20
    assert main["mean_dpct_obs"] == pytest.approx(20, abs=2)
    assert main["mean_dpct_pred"] == pytest.approx(10, abs=2)
    assert main["pass_through_pct"] == pytest.approx(0.5, abs=0.12)
    assert main["share_pred_up"] == 1.0 and main["sign_test_p"] < 0.001
    assert main["auc_pred_vs_ref"] > 0.95                # risers beat same-construction non-risers
    stable_up = t.loc[("up & stable", "all")]
    assert stable_up["n"] == 30
    assert abs(stable_up["mean_dpct_pred"]) < 2          # model blind to non-construction rise
    assert 0.3 < stable_up["auc_pred_vs_ref"] < 0.7
    assert t.loc[("down & built>=5%", "all"), "n"] == 0
    assert set(t.index.get_level_values("sample")) == {"all", "train", "heldout"}


def test_with_flags_replaces_construction_flags():
    p, _ = _pairs(n=10)
    other = p[["GEOID"]].assign(in_stable=True, in_change_1pct=False, in_change_5pct=False,
                                in_change_10pct=False)
    out = cc.with_flags(p.drop(columns=["in_stable"]), other.iloc[:5])
    assert out["in_stable"].tolist() == [True] * 5 + [False] * 5
    assert not out["in_change_5pct"].any()
    assert cc.with_flags(p, None) is p
