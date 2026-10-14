"""Cardinal level labels for the L2 baseline (B1, "cardinal-levels").

The ordinal model trains on ``Rel_Score`` — the per-CBSA-year z-score of log
wealth — so every city-year is centred and scaled away before the model sees
it. The cardinal baseline instead regresses on the wealth LEVEL, the way a
standard satellite-income regression would:

    label = (log W_real - mu_train) / sd_train

* ``W_real`` is the indicator's dollar variable (e.g. ``W2_i_r5pct``) deflated
  to constant ``DOLLAR_BASE_YEAR`` dollars with CPI-U, so pure inflation does
  not masquerade as growth.
* ``mu_train`` / ``sd_train`` are ONE pooled pair of constants, fitted on train
  CBSAs only (each train city's temporal-holdout year excluded), over every
  (tract, imagery-year) row the training table can draw. They never vary by
  city or year: between-city level differences — the corruption the baseline
  is meant to expose — stay in the label.

No regional price parity (RPP) adjustment is applied: the baseline is the
naive "predict dollars" regression the paper argues against, not a
cost-of-living-corrected one.

Pure pandas/numpy (no torch, no geopandas) so it unit-tests on synthetic
frames and can be imported from evaluation as well as from build_dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import indicators

# BLS CPI-U, U.S. city average, all items (series CUUR0000SA0), annual
# averages. ACS 5-year dollar estimates are in the nominal dollars of the
# vintage's final year; deflating to one base year keeps ~25% inflation
# between 2016 and 2023 from masquerading as wealth growth proportional to
# each tract's level (which any levels-only model would "predict").
CPI_U_ANNUAL = {
    2010: 218.056, 2011: 224.939, 2012: 229.594, 2013: 232.957, 2014: 236.736,
    2015: 237.017, 2016: 240.007, 2017: 245.120, 2018: 251.107, 2019: 255.657,
    2020: 258.811, 2021: 270.970, 2022: 292.655, 2023: 304.702,
}
DOLLAR_BASE_YEAR = 2023

LABEL_TARGETS = ("rel_score", "log_level")


def deflate(values, year: int, base: int = DOLLAR_BASE_YEAR):
    """Nominal ``year`` dollars -> constant ``base``-year dollars (CPI-U)."""
    year, base = int(year), int(base)
    missing = [y for y in (year, base) if y not in CPI_U_ANNUAL]
    if missing:
        raise KeyError(f"No CPI-U annual average for year(s) {missing}; "
                       f"covered: {min(CPI_U_ANNUAL)}-{max(CPI_U_ANNUAL)}")
    return values * (CPI_U_ANNUAL[base] / CPI_U_ANNUAL[year])


def wealth_dollar_column(indicator: str, year: int) -> str:
    """Panel column holding the dollar variable behind ``indicator``'s label
    (the label is the per-CBSA z-score of its log): W-family tokens -> the
    wealth index (e.g. 'W2_i_r5pct_2016'), 'inc' -> per-capita income."""
    var = indicators.token_to_var(indicator)
    return f"{var}_{year}" if var else f"per_capita_income_usd_{year}"


def cbsa_key(code) -> str:
    """Canonical string form of a CBSA code ('35620', 35620, 35620.0 -> '35620')."""
    try:
        return str(int(float(code)))
    except (TypeError, ValueError):
        return str(code)


def log_real_dollars(usd, year: int, base: int = DOLLAR_BASE_YEAR) -> np.ndarray:
    """log of CPI-deflated dollars; non-positive / missing values -> NaN."""
    real = deflate(np.asarray(usd, dtype="float64"), year, base)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(real > 0, np.log(real), np.nan)


def panel_log_real_wealth(panel: pd.DataFrame, geoid_col: str, indicator: str,
                          acs_years, base: int = DOLLAR_BASE_YEAR) -> pd.DataFrame:
    """Long [GEOID, cbsa_code, acs_year, log_real] from a wide tract panel.

    ``GEOID`` is ``panel[geoid_col].astype(str)`` — the exact transform
    build_dataset._build_labels_frame applies, so the two join without any
    zero-padding mismatch. Raises if a requested vintage's column is absent:
    a silently missing year would drop its rows from training without notice.
    """
    acs_years = sorted({int(y) for y in acs_years})
    cols = {y: wealth_dollar_column(indicator, y) for y in acs_years}
    missing = [c for c in cols.values() if c not in panel.columns]
    if missing:
        raise KeyError(f"Panel is missing dollar columns for indicator "
                       f"'{indicator}': {missing}")
    base_df = pd.DataFrame({
        "GEOID": panel[geoid_col].astype(str).to_numpy(),
        "cbsa_code": panel["cbsa_code"].map(cbsa_key).to_numpy(),
    })
    keep = ~base_df["GEOID"].duplicated().to_numpy()
    frames = []
    for y, col in cols.items():
        frames.append(pd.DataFrame({
            "GEOID": base_df["GEOID"].to_numpy()[keep],
            "cbsa_code": base_df["cbsa_code"].to_numpy()[keep],
            "acs_year": np.int64(y),
            "log_real": log_real_dollars(panel[col].to_numpy()[keep], y, base),
        }))
    return pd.concat(frames, ignore_index=True)


def read_panel_log_real_wealth(panel_path: Path, geoid_col: str, indicator: str,
                               acs_years, base: int = DOLLAR_BASE_YEAR) -> pd.DataFrame:
    """Column-pruned feather read + :func:`panel_log_real_wealth` (no geometry)."""
    cols = [geoid_col, "cbsa_code"] + [wealth_dollar_column(indicator, y)
                                       for y in sorted({int(y) for y in acs_years})]
    panel = pd.read_feather(panel_path, columns=cols)
    return panel_log_real_wealth(panel, geoid_col, indicator, acs_years, base)


def attach_log_level(labels: pd.DataFrame, wealth_long: pd.DataFrame,
                     acs_year_of) -> pd.DataFrame:
    """Labels frame + ``cbsa_code`` and ``log_real`` for each (GEOID, year).

    ``labels`` is keyed by IMAGERY year; ``acs_year_of(year)`` maps it to the
    ACS vintage that supervises it (the same clamp the z-score labels use), so
    both targets come from the same survey vintage. Row order is preserved.
    """
    out = labels.copy()
    out["GEOID"] = out["GEOID"].astype(str)
    out["acs_year"] = out["year"].map(lambda y: int(acs_year_of(int(y)))).astype("int64")
    lookup = wealth_long[["GEOID", "acs_year", "cbsa_code", "log_real"]]
    merged = out.merge(lookup, on=["GEOID", "acs_year"], how="left", validate="many_to_one")
    return merged.drop(columns=["acs_year"])


def fit_level_standardizer(frame: pd.DataFrame, train_cbsas,
                           holdout_map: dict | None = None) -> dict:
    """Pooled mean/SD of ``log_real`` over train-city rows only.

    ``frame`` carries ``cbsa_code``, ``year`` (imagery year) and ``log_real``
    (see :func:`attach_log_level`). Rows of a train city at its temporal-holdout
    year are excluded — they are ``val_temporal`` and must not inform the
    label scale either. Raises when nothing usable remains.
    """
    train = {cbsa_key(c) for c in train_cbsas}
    holdout = {cbsa_key(k): int(v) for k, v in (holdout_map or {}).items()}
    cb = frame["cbsa_code"].map(cbsa_key)
    in_train = cb.isin(train).to_numpy()
    on_holdout = (cb.map(holdout).to_numpy(dtype="float64", na_value=np.nan)
                  == frame["year"].to_numpy(dtype="float64"))
    vals = frame["log_real"].to_numpy(dtype="float64")
    use = in_train & ~on_holdout & np.isfinite(vals)
    if use.sum() < 2:
        raise ValueError("fit_level_standardizer: fewer than 2 finite train rows — "
                         "check the train CBSA list and the panel dollar columns.")
    mu = float(vals[use].mean())
    sd = float(vals[use].std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"fit_level_standardizer: degenerate SD ({sd}).")
    return {"mu": mu, "sd": sd, "n_rows": int(use.sum()),
            "n_cbsas": int(cb[use].nunique()), "base_year": DOLLAR_BASE_YEAR}


def standardized_level_labels(labels: pd.DataFrame, wealth_long: pd.DataFrame,
                              acs_year_of, train_cbsas,
                              holdout_map: dict | None = None):
    """Copy of ``labels`` with ``Rel_Score`` replaced by the B1 level label.

    Returns ``(new_labels, stats)``. Only ``Rel_Score`` changes: the
    structural-change flag and ``score_bin`` (unused by the MSE loss) are kept,
    and tracts without a positive dollar value get NaN, which the shard
    generator's existing NaN guard drops before any fetch.
    """
    with_level = attach_log_level(labels, wealth_long, acs_year_of)
    stats = fit_level_standardizer(with_level, train_cbsas, holdout_map)
    new = labels.copy()
    new["Rel_Score"] = ((with_level["log_real"].to_numpy(dtype="float64") - stats["mu"])
                        / stats["sd"]).astype("float32")
    stats["n_labels"] = int(len(new))
    stats["n_nan"] = int(np.isnan(new["Rel_Score"].to_numpy()).sum())
    return new, stats


def relabel_flat(df: pd.DataFrame, new_labels: pd.DataFrame) -> pd.DataFrame:
    """Swap ``Rel_Score`` in a materialized flat frame via (GEOID, year).

    Used for the validation frames, which the lazy split hands back already
    materialized. Rows whose (GEOID, year) has no new label become NaN.
    """
    if df is None or len(df) == 0:
        return df
    key = new_labels.assign(GEOID=new_labels["GEOID"].astype(str)).set_index(
        ["GEOID", "year"])["Rel_Score"]
    idx = pd.MultiIndex.from_arrays([df["GEOID"].astype(str).to_numpy(),
                                     df["year"].astype("int64").to_numpy()])
    out = df.copy()
    out["Rel_Score"] = key.reindex(idx).to_numpy(dtype="float32")
    return out
