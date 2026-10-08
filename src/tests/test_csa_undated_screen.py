"""Synthetic tests for src/csa_undated_screen.py (restricted vs unrestricted ATT).

The heavy lifting — cohorts, panels, the estimator — is tested in
test_csa_event_study.py. What is specific here is the *comparison*: pairing the
two arms on threshold and expressing the gap in units a reader can judge.
"""

import numpy as np
import pandas as pd
import pytest

from src import csa_undated_screen as cus


def _frame(rows):
    return pd.DataFrame(
        rows, columns=["city", "threshold", "screen", "overall_att", "overall_se"])


def test_delta_is_measured_against_the_matching_unrestricted_threshold():
    """Each screened row must be compared with the SAME threshold's unrestricted
    arm — pairing on the wrong threshold would silently compare 5% against 1%,
    where the ATT genuinely differs (dose response), and manufacture a gap."""
    df = _frame([
        ("tampa", 0.01, np.nan, 0.0346, 0.010),
        ("tampa", 0.05, np.nan, 0.0508, 0.012),
        ("tampa", 0.01, 0.20, 0.0345, 0.010),
        ("tampa", 0.05, 0.20, 0.0509, 0.012),
    ])
    out = cus._add_delta(df)
    screened = out[out["screen"].notna()].set_index("threshold")["att_delta"]
    assert screened.loc[0.01] == pytest.approx(-0.0001, abs=1e-9)
    assert screened.loc[0.05] == pytest.approx(+0.0001, abs=1e-9)


def test_delta_is_also_expressed_in_standard_errors():
    """A raw gap in score units is not interpretable on its own; the question is
    whether the screen moves the estimate by more than its own noise."""
    df = _frame([
        ("x", 0.05, np.nan, 0.0500, 0.010),
        ("x", 0.05, 0.20, 0.0550, 0.010),
    ])
    out = cus._add_delta(df)
    row = out[out["screen"].notna()].iloc[0]
    assert row["att_delta"] == pytest.approx(0.005, abs=1e-9)
    assert row["att_delta_in_se"] == pytest.approx(0.5, abs=1e-9)


def test_unrestricted_rows_carry_no_delta():
    """The baseline arm has nothing to be compared against; a 0.0 there would
    read as 'measured and unchanged' rather than 'not applicable'."""
    df = _frame([
        ("x", 0.05, np.nan, 0.05, 0.01),
        ("x", 0.05, 0.20, 0.05, 0.01),
    ])
    out = cus._add_delta(df)
    base = out[out["screen"].isna()].iloc[0]
    assert pd.isna(base["att_delta"]) and pd.isna(base["att_delta_in_se"])


def test_delta_is_nan_when_an_arm_could_not_be_estimated():
    """A city whose panel is unusable (e.g. predictions missing for one panel
    year) yields NaN ATTs; the comparison must propagate that rather than
    report a spurious zero gap."""
    df = _frame([
        ("seattle", 0.05, np.nan, np.nan, np.nan),
        ("seattle", 0.05, 0.20, np.nan, np.nan),
    ])
    out = cus._add_delta(df)
    assert out["att_delta"].isna().all()


def test_screen_column_distinguishes_the_two_arms():
    """`screen` is NaN for unrestricted and the threshold for restricted; the
    pairing logic depends on that and on nothing else."""
    df = _frame([
        ("x", 0.05, np.nan, 0.05, 0.01),
        ("x", 0.05, 0.20, 0.06, 0.01),
    ])
    out = cus._add_delta(df)
    assert out["screen"].isna().sum() == 1
    assert (out["screen"] == 0.20).sum() == 1
