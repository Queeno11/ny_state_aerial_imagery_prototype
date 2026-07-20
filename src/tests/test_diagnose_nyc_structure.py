"""Synthetic-data tests for diagnose_nyc_structure decomposition logic."""

import numpy as np
import pandas as pd

from src.diagnose_nyc_structure import (add_borough, decompose_city,
                                        density_lean, same_bin_pair_fraction)


def _frame(geoids, labels, preds):
    return pd.DataFrame({"GEOID": geoids, "Rel_Score": labels,
                         "predicted_value": preds})


def test_add_borough_filters_and_maps():
    df = _frame(["36061000100", "36005000200", "4013040514", "36047001500"],
                [1.0, -1.0, 0.0, 0.2], [0.5, -0.5, 0.1, 0.0])
    out = add_borough(df)
    assert len(out) == 3
    assert set(out["boro"]) == {"Manhattan", "Bronx", "Brooklyn"}


def test_within_signal_only():
    """Groups ordered correctly within, but between-group means inverted."""
    rng = np.random.default_rng(0)
    rows = []
    # rich group A (label mean +1) predicted LOW (-1); poor group B predicted HIGH
    for boro_fips, lab_mu, pred_mu in [("36061", 1.0, -1.0), ("36005", -1.0, 1.0)]:
        within = rng.normal(0, 1, 50)
        for i, w in enumerate(within):
            rows.append({"GEOID": f"{boro_fips}{i:06d}", "Rel_Score": lab_mu + w,
                         "predicted_value": pred_mu + w})  # perfect within
    df = add_borough(pd.DataFrame(rows))
    d = decompose_city(df)
    assert d["pooled_within"] > 0.99          # perfect within-group ranking
    # only 2 groups -> between is nan by design (needs >= 3)
    assert np.isnan(d["between"])
    assert d["overall"] < 0.5                 # inversion drags pooled rho down


def test_between_signal_only():
    """Group means ordered correctly, within-group prediction pure noise."""
    rng = np.random.default_rng(1)
    rows = []
    for boro_fips, mu in [("36061", 1.0), ("36047", 0.0), ("36005", -1.0)]:
        for i in range(60):
            rows.append({"GEOID": f"{boro_fips}{i:06d}",
                         "Rel_Score": mu + rng.normal(0, 1),
                         "predicted_value": mu + rng.normal(0, 1) * 0.0
                         + rng.normal(0, 0.01)})
    df = add_borough(pd.DataFrame(rows))
    d = decompose_city(df)
    assert d["between"] == 1.0
    assert abs(d["pooled_within"]) < 0.25     # no within signal


def test_same_bin_pair_fraction():
    # all same bin -> every pair excluded
    assert same_bin_pair_fraction([1, 1, 1, 1]) == 1.0
    # all distinct -> no pair excluded
    assert same_bin_pair_fraction([1, 2, 3, 4]) == 0.0
    # two bins of two: 2 same-bin pairs of 6 total
    assert abs(same_bin_pair_fraction([1, 1, 2, 2]) - 2 / 6) < 1e-12
    # degenerate
    assert np.isnan(same_bin_pair_fraction([1]))
    # NaNs ignored
    assert same_bin_pair_fraction([1, 1, np.nan]) == 1.0


def test_density_lean_signs():
    rng = np.random.default_rng(3)
    dens = rng.normal(0, 1, 300)
    # labels uncorrelated with density; predictions strongly ANTI-density
    df = pd.DataFrame({"Rel_Score": rng.normal(0, 1, 300),
                       "predicted_value": -dens + rng.normal(0, 0.1, 300),
                       "log_density": dens})
    d = density_lean(df)
    assert d["rho_pred_dens"] < -0.9
    assert abs(d["rho_label_dens"]) < 0.2
    assert d["lean_gap"] < -0.7


def test_per_group_min_size():
    rng = np.random.default_rng(2)
    rows = [{"GEOID": f"36061{i:06d}", "Rel_Score": rng.normal(),
             "predicted_value": rng.normal()} for i in range(9)]
    rows += [{"GEOID": f"36005{i:06d}", "Rel_Score": rng.normal(),
              "predicted_value": rng.normal()} for i in range(15)]
    d = decompose_city(add_borough(pd.DataFrame(rows)))
    assert "Manhattan" not in d["per_group"]  # n=9 < 10
    assert "Bronx" in d["per_group"]
