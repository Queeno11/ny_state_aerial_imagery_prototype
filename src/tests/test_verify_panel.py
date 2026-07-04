"""Tests for verify_panel's p<0.10 gate-staleness check (#29).

Synthetic frames only — the full verify_panel.main path needs a real panel and is
exercised by run_data_pipeline on real data.
"""

import pandas as pd

from src.data import verify_panel as vp
from src.data import indicators as ind
from src.data.process_acs import PANEL_YEARS, STRUCTURAL_CHANGE_P


START, END = PANEL_YEARS[0], PANEL_YEARS[-1]


def _frame(pvals, flags, var="W2_i_r5pct", token="W2_r5"):
    return pd.DataFrame({
        f"pvalue_{START}_{END}": pvals,
        ind.valid_change_col("inc"): flags,
        f"pvalue_{var}_{START}_{END}": pvals,
        ind.valid_change_col(token): flags,
    })


def test_gate_consistent_passes():
    pvals = [0.005, 0.05, 0.09, 0.11, 0.5]
    flags = [p < STRUCTURAL_CHANGE_P for p in pvals]
    c = vp.Checker()
    vp.check_gate_consistency(_frame(pvals, flags), c)
    assert not c.failures


def test_stale_p01_panel_fails():
    """A panel built under the old p<0.01 gate must FAIL the consistency check."""
    pvals = [0.005, 0.05, 0.09, 0.11, 0.5]
    stale_flags = [p < 0.01 for p in pvals]   # old gate
    c = vp.Checker()
    vp.check_gate_consistency(_frame(pvals, stale_flags), c)
    assert c.failures  # 0.05 and 0.09 rows mismatch under p<0.10


def test_missing_pvalue_columns_warn_only():
    df = pd.DataFrame({ind.valid_change_col("inc"): [True, False]})
    c = vp.Checker()
    vp.check_gate_consistency(df, c)
    assert not c.failures  # cannot verify -> WARN, not FAIL


def test_nan_pvalues_ignored():
    pvals = [0.05, None, 0.5]
    flags = [True, False, False]
    c = vp.Checker()
    vp.check_gate_consistency(_frame(pvals, flags), c)
    assert not c.failures
