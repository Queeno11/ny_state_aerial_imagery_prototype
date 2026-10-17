"""Synthetic tests for rank_autocorrelation_by_city and src/plot_autocorr_dumbbell.py."""

import numpy as np
import pandas as pd
import pytest

from src.plot_autocorr_dumbbell import city_autocorr, fig_dumbbell, weighted_means
from src.utils.metrics import rank_autocorrelation, rank_autocorrelation_by_city


def _bld(n_per_city=(6, 12), split="test", seed=0):
    """Stable buildings in two cities, two years; city 2's labels are reversed in 2024."""
    rng = np.random.default_rng(seed)
    rows = []
    for city, n in zip((1, 2), n_per_city):
        base = rng.normal(size=n)
        for yr in (2016, 2024):
            lab = base if (city == 1 or yr == 2016) else -base
            for i in range(n):
                rows.append(dict(building_id=city * 100 + i, year=yr, cbsa=city,
                                 pred=base[i], label=lab[i], change=0, type=split))
    # a changed building that must be ignored
    rows.append(dict(building_id=999, year=2016, cbsa=1, pred=9, label=9, change=1, type=split))
    return pd.DataFrame(rows)


def test_by_city_matches_pooled_average():
    df = _bld()
    per = rank_autocorrelation_by_city(df)
    pooled = rank_autocorrelation(df)
    assert list(per.columns) == ["cbsa", "t1", "t2", "n", "rho_p", "rho_l"]
    assert per.set_index("cbsa")["rho_l"].to_dict() == pytest.approx({1: 1.0, 2: -1.0})
    mp, ml = weighted_means(per)
    assert mp == pytest.approx(pooled["rank_autocorr_pred"])
    assert ml == pytest.approx(pooled["rank_autocorr_label"])
    assert ml == pytest.approx((6 - 12) / 18)


def test_by_city_empty_frame_keeps_columns():
    per = rank_autocorrelation_by_city(_bld(n_per_city=(3, 3)))
    assert per.empty and "rho_p" in per.columns
    assert rank_autocorrelation(_bld(n_per_city=(3, 3))) == {}


def test_city_autocorr_filters_split_names_and_sorts():
    df = pd.concat([_bld(), _bld(split="train").assign(cbsa=lambda d: d.cbsa + 10)])
    out = city_autocorr(df, "test", names={"1": "Alpha", "2": "Beta"})
    assert out["name"].tolist() == ["Beta", "Alpha"]  # sorted by label autocorrelation


def test_fig_dumbbell_rows_and_empty():
    fig = fig_dumbbell(city_autocorr(_bld()))
    assert len(fig.axes[0].get_xticklabels()) == 2
    with pytest.raises(ValueError, match="no cities"):
        fig_dumbbell(city_autocorr(_bld(n_per_city=(3, 3))))
