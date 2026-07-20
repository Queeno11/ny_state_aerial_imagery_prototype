"""Synthetic-data tests for src/diagnose_pair_exclusion.py.

Checks the batch/pair simulator against analytic values:
  1. Pool choice is proportional to pool size (HybridBatchSampler semantics).
  2. With iid N(0,1) labels and constant SE=s, the SE rule's exclusion
     probability matches P(|N(0,2)| < k*s*sqrt(2)) = 2*Phi(k*s) - 1.
  3. With uniform labels binned into exact quintiles, P(bin_diff == 0) -> 1/5.
  4. Small pools (< batch_size_cs // 4) are dropped, mirroring _pools().
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.diagnose_pair_exclusion import build_pools, sample_batch_pairs, simulate


def _synth_universe(n_per_city=2000, n_cities=3, se=0.2, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for c in range(n_cities):
        z = rng.normal(0.0, 1.0, n_per_city)
        frames.append(pd.DataFrame({
            "GEOID": [f"{c:02d}{i:09d}" for i in range(n_per_city)],
            "year": 2012,
            "cbsa_code": f"c{c}",
            "bracket": "large",
            "county_fips": f"{c:05d}",
            "z": z,
            # exact empirical quintiles => P(same bin) ~ 0.2 within a city
            "score_bin": pd.qcut(z, 5, labels=False),
            "score_bin_cbsa": pd.qcut(z, 5, labels=False),
            "se": se,
        }))
    return pd.concat(frames, ignore_index=True)


def test_pool_weighting_proportional_to_size():
    df = _synth_universe(n_per_city=1000)
    # triple city 0's pool
    extra = df[df.cbsa_code == "c0"].copy()
    df = pd.concat([df, extra.assign(year=2014), extra.assign(year=2016)],
                   ignore_index=True)
    pools, keys, probs = build_pools(df, batch_size_cs=32)
    p_c0 = sum(p for k, p in zip(keys, probs) if k[1] == "c0")
    assert p_c0 == pytest.approx(3000 / 5000, abs=1e-9)


def test_se_rule_matches_analytic_probability():
    se = 0.2
    df = _synth_universe(n_per_city=5000, n_cities=1, se=se, seed=1)
    pairs = simulate(df, batch_size_cs=32, n_batches=1500, seed=2)
    for k in (1.0, 2.0):
        # dz ~ N(0, 2) => P(|dz| < k*se*sqrt(2)) = 2*Phi(k*se) - 1
        expected = 2 * norm.cdf(k * se) - 1
        got = pairs[f"se_excl_{k:g}"].mean()
        assert got == pytest.approx(expected, abs=0.02), (k, got, expected)


def test_bin_rule_matches_quintile_probability():
    df = _synth_universe(n_per_city=5000, n_cities=1, seed=3)
    pairs = simulate(df, batch_size_cs=32, n_batches=1500, seed=4)
    assert pairs["bin_excl"].mean() == pytest.approx(0.2, abs=0.02)


def test_cbsa_bin_rule_is_two_tenths_regardless_of_city_shift():
    # Shift one city's labels far away: its NATIONAL bins would collapse into
    # one, but intra-CBSA quintiles must still exclude ~20% of its pairs.
    df = _synth_universe(n_per_city=5000, n_cities=2, seed=5)
    df.loc[df.cbsa_code == "c1", "z"] -= 10.0
    df["score_bin_cbsa"] = df.groupby(["year", "cbsa_code"])["z"].transform(
        lambda x: pd.qcut(x, 5, labels=False))
    pairs = simulate(df, batch_size_cs=32, n_batches=1500, seed=6)
    sub = pairs[pairs.cbsa_code == "c1"]
    assert sub["bin_cbsa_excl"].mean() == pytest.approx(0.2, abs=0.02)


def test_small_pools_dropped():
    df = _synth_universe(n_per_city=1000, n_cities=1)
    tiny = df.head(5).copy().assign(cbsa_code="tiny")
    pools, keys, _ = build_pools(pd.concat([df, tiny], ignore_index=True),
                                 batch_size_cs=32)
    assert all(k[1] != "tiny" for k in keys)  # 5 < 32 // 4
