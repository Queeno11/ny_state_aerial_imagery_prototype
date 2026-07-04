"""Indicator token registry: short names for the training-label candidates.

The ACS panel (``process_acs.py``) carries one per-CBSA relative score per year for
income and for each occupant-wealth index (W1/W2/W3 x discount rate rho), plus one
structural-change flag per indicator (``valid_change_{token}``). Downstream code
(``build_dataset.py``, ``main.py``) selects which indicator supervises training via a
single short token, e.g. ``params["indicator"] = "W2_r5"``.

Token grammar
-------------
* ``inc``            -- per-capita income (panel columns ``Rel_Score_{year}``).
* ``{fam}_r{rate}``  -- wealth index family ``W`` (W1 baseline), ``W2`` or ``W3`` at
                        human-capital discount rate rho = rate% (2, 3, 5, 7), e.g.
                        ``W2_r5`` -> panel columns ``Rel_Score_W2_i_r5pct_{year}``.

The capital yield r_k (W2/W3 capital-income term) is fixed at build time by
``process_acs.CAPITAL_YIELD`` (0.045) and is not part of the token.
"""

# Wealth families and discount rates must mirror process_acs.WEALTH_INDEX_VARS /
# DISCOUNT_RATES. Kept literal here so importing this module stays dependency-free
# (build_dataset/main must not pull in the heavy process_acs machinery).
_WEALTH_FAMILIES = ("W", "W2", "W3")
_RATE_PCTS = (2, 3, 5, 7)

# token -> panel wealth variable name ("W2_r5" -> "W2_i_r5pct"); income has no variable.
TOKEN_TO_VAR = {
    f"{fam}_r{rate}": f"{fam}_i_r{rate}pct"
    for fam in _WEALTH_FAMILIES
    for rate in _RATE_PCTS
}
VAR_TO_TOKEN = {var: token for token, var in TOKEN_TO_VAR.items()}

# token -> panel score-column prefix (score columns are f"{prefix}_{year}").
INDICATORS = {"inc": "Rel_Score"}
INDICATORS.update(
    {token: f"Rel_Score_{var}" for token, var in TOKEN_TO_VAR.items()}
)

# Default training indicator: W2 at rho = 0.05 (r_k = 0.045 via CAPITAL_YIELD).
DEFAULT_INDICATOR = "W2_r5"


def _check(token: str) -> str:
    if token not in INDICATORS:
        raise KeyError(
            f"Unknown indicator token {token!r}. Valid tokens: {sorted(INDICATORS)}"
        )
    return token


def score_col(token: str, year: int) -> str:
    """Panel column holding the per-CBSA relative score of ``token`` in ``year``."""
    return f"{INDICATORS[_check(token)]}_{year}"


def score_prefix(token: str) -> str:
    """Panel column prefix for ``token``'s relative scores (columns are prefix_{year})."""
    return INDICATORS[_check(token)]


def valid_change_col(token: str) -> str:
    """Panel column holding the structural-change flag for ``token``."""
    return f"valid_change_{_check(token)}"


def var_to_token(var: str) -> str:
    """Map a panel wealth variable name to its token ("W2_i_r5pct" -> "W2_r5")."""
    if var not in VAR_TO_TOKEN:
        raise KeyError(
            f"Unknown wealth variable {var!r}. Valid: {sorted(VAR_TO_TOKEN)}"
        )
    return VAR_TO_TOKEN[var]


def token_to_var(token: str) -> str | None:
    """Map a token to its panel wealth variable name; None for income."""
    _check(token)
    return TOKEN_TO_VAR.get(token)
