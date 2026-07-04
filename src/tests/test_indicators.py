"""Unit tests for the indicator token registry (src/data/indicators.py)."""

import pytest

from src.data import indicators as ind


def test_registry_covers_income_and_all_wealth_variants():
    # income + 3 families x 4 rates = 13 tokens
    assert len(ind.INDICATORS) == 13
    assert "inc" in ind.INDICATORS
    for fam in ("W", "W2", "W3"):
        for rate in (2, 3, 5, 7):
            assert f"{fam}_r{rate}" in ind.INDICATORS


def test_registry_matches_process_acs_wealth_vars():
    # The literal token->var map must mirror process_acs.WEALTH_INDEX_VARS.
    from src.data.process_acs import WEALTH_INDEX_VARS
    assert set(ind.TOKEN_TO_VAR.values()) == set(WEALTH_INDEX_VARS)


def test_default_indicator():
    assert ind.DEFAULT_INDICATOR == "W2_r5"
    assert ind.token_to_var("W2_r5") == "W2_i_r5pct"


def test_score_col_income():
    assert ind.score_col("inc", 2019) == "Rel_Score_2019"


def test_score_col_wealth():
    assert ind.score_col("W2_r5", 2019) == "Rel_Score_W2_i_r5pct_2019"
    assert ind.score_col("W3_r7", 2011) == "Rel_Score_W3_i_r7pct_2011"


def test_valid_change_col():
    assert ind.valid_change_col("inc") == "valid_change_inc"
    assert ind.valid_change_col("W2_r5") == "valid_change_W2_r5"


def test_var_token_round_trip():
    for token, var in ind.TOKEN_TO_VAR.items():
        assert ind.var_to_token(var) == token
        assert ind.token_to_var(token) == var


def test_token_to_var_income_is_none():
    assert ind.token_to_var("inc") is None


def test_unknown_token_raises():
    with pytest.raises(KeyError):
        ind.score_col("W4_r5", 2019)
    with pytest.raises(KeyError):
        ind.valid_change_col("bogus")
    with pytest.raises(KeyError):
        ind.var_to_token("W9_i_r5pct")
