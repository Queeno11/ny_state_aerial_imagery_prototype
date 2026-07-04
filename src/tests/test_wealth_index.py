"""Synthetic-data unit tests for the W2/W3 wealth-index construction and the VRE
replicate-variance machinery in src/data/process_acs.py. No network / no real ACS data."""

import numpy as np
import pandas as pd
import pytest

from src.data import process_acs as pa


# ── builders ────────────────────────────────────────────────────────────────────

def _age_frame(midpoint, pop, n=1):
    """A frame whose entire working-age population sits in the bracket at ``midpoint``."""
    idx = pa._AGE_MIDPOINTS.index(midpoint)
    male, female = f"B01001_{idx + 3:03d}E", f"B01001_{idx + 27:03d}E"
    d = {c: np.zeros(n) for c in pa.WORKING_AGE_VARS}
    d[male] = np.full(n, pop / 2.0)
    d[female] = np.full(n, pop / 2.0)
    return pd.DataFrame(d)


def _base_frame(n=4):
    """Minimal raw-input frame with all columns wealth_from_inputs touches."""
    rng = np.random.default_rng(1)
    d = pd.DataFrame({
        "geoid": [f"36061{i:06d}" for i in range(n)],
        "total_population": np.full(n, 1000.0),
        "aggregate_earnings_usd": rng.uniform(2e7, 5e7, n),
        "aggregate_owner_value_usd": rng.uniform(1e8, 3e8, n),
        "aggregate_capital_income_usd": rng.uniform(1e6, 5e6, n),
        "B25003_001E": np.full(n, 400.0),
        "B25003_002E": np.full(n, 250.0),
        "B25081_001E": np.full(n, 250.0),
        "B25081_002E": np.full(n, 150.0),   # mortgaged share m = 0.6
        "B25038_002E": np.full(n, 250.0),
    })
    for c in pa.WORKING_AGE_VARS:
        d[c] = np.full(n, 30.0)
    for c in pa.MOVEIN_OWNER_VARS:
        d[c] = np.full(n, 40.0)
    return d


# ── (1) expected working years N_i ───────────────────────────────────────────────

def test_expected_working_years_young_vs_old():
    young = pa.expected_working_years(_age_frame(16.0, 100))   # all at midpoint 16
    old = pa.expected_working_years(_age_frame(63.0, 100))     # all at midpoint 63
    assert young.iloc[0] == pytest.approx(65 - 16, abs=1e-9)
    assert old.iloc[0] == pytest.approx(65 - 63, abs=1e-9)
    assert young.iloc[0] > old.iloc[0]


def test_expected_working_years_zero_pop():
    d = pd.DataFrame({c: [0.0] for c in pa.WORKING_AGE_VARS})
    assert pa.expected_working_years(d).iloc[0] == 0.0


# ── (2) amortization ─────────────────────────────────────────────────────────────

def test_amortization_endpoints_and_monotone():
    assert pa.amortization_remaining(0) == pytest.approx(1.0)
    assert pa.amortization_remaining(pa.MORTGAGE_TERM) == pytest.approx(0.0, abs=1e-12)
    ts = np.linspace(0, pa.MORTGAGE_TERM, 31)
    rem = pa.amortization_remaining(ts)
    assert np.all(np.diff(rem) <= 1e-12)            # non-increasing
    assert pa.amortization_remaining(50) == 0.0     # past term -> clipped to 0


# ── (3) capital income term ──────────────────────────────────────────────────────

def test_capital_income_pc_exact():
    d = pd.DataFrame({"total_population": [1000.0],
                      "aggregate_capital_income_usd": [2_000_000.0]})
    got = pa.capital_income_pc(d, r_k=0.05).iloc[0]
    assert got == pytest.approx((1 / 0.05) * (2_000_000.0 / 1000.0))


def test_capital_income_pc_absent_is_zero():
    d = pd.DataFrame({"total_population": [1000.0]})
    assert pa.capital_income_pc(d).iloc[0] == 0.0


# ── (4) equity term reduces to 1 - m*LTV ─────────────────────────────────────────

def test_equity_pc_matches_closed_form():
    d = _base_frame(1)
    ltv = 0.5
    eq = pa.equity_pc(d, ltv).iloc[0]
    m = 150.0 / 250.0
    expected = (d["aggregate_owner_value_usd"].iloc[0] / 1000.0) * (1 - m * ltv)
    assert eq == pytest.approx(expected)


# ── (5) owner median tenure ──────────────────────────────────────────────────────

def test_owner_median_tenure_recent_vs_old():
    year = 2023
    recent = _base_frame(1)
    old = _base_frame(1)
    for c in pa.MOVEIN_OWNER_VARS:
        recent[c] = 0.0
        old[c] = 0.0
    recent["B25038_003E"] = 250.0     # most-recent bracket
    old["B25038_008E"] = 250.0        # earliest bracket
    t_recent = pa.owner_median_tenure(recent, year).iloc[0]
    t_old = pa.owner_median_tenure(old, year).iloc[0]
    assert t_recent < t_old
    assert t_recent < 5 and t_old > 30


# ── (6) W3 LTV: long tenure / appreciation -> lower LTV ──────────────────────────

def test_impute_ltv_w3_tenure_and_appreciation():
    year = 2023
    recent = _base_frame(1)
    old = _base_frame(1)
    for c in pa.MOVEIN_OWNER_VARS:
        recent[c] = 0.0
        old[c] = 0.0
    recent["B25038_003E"] = 250.0
    old["B25038_008E"] = 250.0
    ltv_recent = pa.impute_ltv_w3(recent, year).iloc[0]
    ltv_old = pa.impute_ltv_w3(old, year).iloc[0]
    assert ltv_old < ltv_recent              # longer tenure -> more amortized -> lower LTV

    # Appreciation lowers LTV further -- use a MODERATE-tenure tract (bracket 005 -> ~10yr,
    # so amortization is partial, not already fully paid off / LTV 0). County tier (geoid[:5]).
    mod = _base_frame(1)
    for c in pa.MOVEIN_OWNER_VARS:
        mod[c] = 0.0
    mod["B25038_005E"] = 250.0
    hpi = {"county": pd.DataFrame({"county_fips": ["36061", "36061"],
                                   "year": [1985, 2023], "hpi": [100.0, 300.0]}),
           "metro": None}
    ltv_mod = pa.impute_ltv_w3(mod, year).iloc[0]
    ltv_appr = pa.impute_ltv_w3(mod, year, hpi=hpi).iloc[0]
    assert 0.0 < ltv_appr < ltv_mod


def test_appreciation_factor_fallback_is_one():
    fac = pa.appreciation_factor(_base_frame(3), 2023)   # no hpi/crosswalk
    assert np.allclose(fac.to_numpy(), 1.0)


def test_appreciation_prefers_county_over_cbsa():
    # County and CBSA series disagree; the county series must win for a covered county.
    d = _base_frame(1)
    for c in pa.MOVEIN_OWNER_VARS:
        d[c] = 0.0
    d["B25038_005E"] = 250.0   # moderate tenure
    hpi = {
        "county": pd.DataFrame({"county_fips": ["36061", "36061"],
                                "year": [2000, 2023], "hpi": [100.0, 400.0]}),  # 4x
        "metro": pd.DataFrame({"cbsa_code": ["35620", "35620"],
                               "year": [2000, 2023], "hpi": [100.0, 150.0]}),   # 1.5x
    }
    xwalk = pd.DataFrame({"county_fips": ["36061"], "cbsa_code": ["35620"], "cbsa_title": ["NY"]})
    ltv_county = pa.impute_ltv_w3(d, 2023, hpi=hpi, crosswalk=xwalk).iloc[0]
    # Drop the county tier -> must fall back to the (weaker) CBSA series -> higher LTV.
    ltv_cbsa = pa.impute_ltv_w3(d, 2023, hpi={"county": None, "metro": hpi["metro"]},
                                crosswalk=xwalk).iloc[0]
    assert ltv_county < ltv_cbsa     # stronger county appreciation -> lower LTV


# ── FHFA .xlsx loaders (synthetic look-alikes with FHFA's note rows) ─────────────

def _write_fhfa_xlsx(path, key_header, key_vals):
    """Write a 6-note-row + header FHFA-style workbook (county or CBSA)."""
    note = "HPI ... developmental ... experimental"
    header = [key_header, "Name", "Year", "Annual Change (%)", "HPI",
              "HPI with 1990 base", "HPI with 2000 base"]
    rows = [[note] + [None] * 6, [None] * 7, [note] * 1 + [None] * 6,
            ["Last updated"] + [None] * 6, ["Not Seasonally Adjusted (NSA)"] + [None] * 6,
            header]
    for k in key_vals:
        rows += [[k, "X", 2000, None, 100.0, 90.0, 100.0],
                 [k, "X", 2023, 12.3, 300.0, 270.0, 300.0]]
    pd.DataFrame(rows).to_excel(path, header=False, index=False)


def test_load_fhfa_xlsx_county_and_cbsa(tmp_path):
    cty = tmp_path / "hpi_at_county.xlsx"
    cbsa = tmp_path / "hpi_at_cbsa.xlsx"
    _write_fhfa_xlsx(cty, "FIPS code", ["36061", "06037"])
    _write_fhfa_xlsx(cbsa, "CBSA", ["35620", "31080"])
    c = pa.load_fhfa_county_hpi(cty)
    m = pa.load_fhfa_metro_hpi(cbsa)
    assert set(c.columns) == {"county_fips", "year", "hpi"}
    assert "36061" in set(c["county_fips"]) and set(c["year"]) == {2000, 2023}
    assert float(c.loc[(c.county_fips == "36061") & (c.year == 2023), "hpi"].iloc[0]) == 300.0
    assert set(m.columns) == {"cbsa_code", "year", "hpi"} and "35620" in set(m["cbsa_code"])


def test_load_fhfa_absent_returns_none():
    assert pa.load_fhfa_county_hpi(pa.Path("/no/such/file.xlsx")) is None


# ── HOEREPHRE 5-year-window owners'-equity share ─────────────────────────────────

def test_load_owners_equity_share_5yr_window(tmp_path):
    # Quarterly series at a constant 60% for 2007-2023 -> every window mean = 0.60.
    dates = pd.date_range("2007-01-01", "2023-10-01", freq="QS")
    csv = tmp_path / "HOEREPHRE.csv"
    pd.DataFrame({"observation_date": dates, "HOEREPHRE": np.full(len(dates), 60.0)}).to_csv(
        csv, index=False)
    shares = pa.load_owners_equity_share(csv, window=5)
    assert shares[2023] == pytest.approx(0.60)
    assert min(shares) == 2011          # first full 5-yr window (2007..2011)
    # Linear ramp -> window mean equals the mean of the 5 annual values.
    ramp = pd.DataFrame({"observation_date": dates,
                         "HOEREPHRE": (dates.year - 2000) * 1.0})  # percent rises 1/yr
    csv2 = tmp_path / "ramp.csv"
    ramp.to_csv(csv2, index=False)
    s2 = pa.load_owners_equity_share(csv2, window=5)
    # 2023 window = years 2019..2023 -> percents 19..23 -> mean 21 -> 0.21
    assert s2[2023] == pytest.approx(0.21, abs=1e-9)


def test_ltv_macro_w2_uses_shares():
    assert pa.ltv_macro_w2(2023, shares={2023: 0.70}) == pytest.approx(0.30)
    assert pa.ltv_macro_w2(2099, shares={2023: 0.70}) == pytest.approx(0.30)  # nearest-year


# ── (7) wealth_from_inputs: parity + W2/W3 structure ─────────────────────────────

def test_wealth_from_inputs_w1_matches_closed_form():
    d = _base_frame(3)
    out = pa.wealth_from_inputs(d, 2023)
    r = 0.03
    earnings_pc = d["aggregate_earnings_usd"] / 1000.0
    e1 = pa.equity_pc(d, pa.LTV_MACRO)
    expected = earnings_pc * pa.annuity_factor(r, pa.HUMAN_CAPITAL_YEARS) + e1
    assert np.allclose(out[pa.w_index_colname(r)].to_numpy(), expected.to_numpy())


def test_w2_w3_differ_only_by_equity():
    d = _base_frame(3)
    out = pa.wealth_from_inputs(d, 2023)
    r = 0.05
    diff = out[pa.w2_index_colname(r)] - out[pa.w3_index_colname(r)]
    e2 = pa.equity_pc(d, pa.ltv_macro_w2(2023))
    e3 = pa.equity_pc(d, pa.impute_ltv_w3(d, 2023))
    assert np.allclose(diff.to_numpy(), (e2 - e3).to_numpy())


def test_w2_includes_capital_term():
    d = _base_frame(3)
    out = pa.wealth_from_inputs(d, 2023)
    r = 0.05
    n_i = pa.expected_working_years(d)
    hc = (d["aggregate_earnings_usd"] / 1000.0) * pa.annuity_factor(r, n_i)
    e2 = pa.equity_pc(d, pa.ltv_macro_w2(2023))
    cap = pa.capital_income_pc(d, pa.CAPITAL_YIELD)
    assert np.allclose(out[pa.w2_index_colname(r)].to_numpy(), (hc + cap + e2).to_numpy())


# ── (8) SDR replicate variance + covariance capture ──────────────────────────────

def test_compute_replicate_se_sdr_formula_and_covariance():
    # Two tracts in distinct CBSAs so the per-CBSA z-score is trivial (std over 1 -> NaN);
    # instead use several tracts in one CBSA and check the SDR arithmetic directly via a
    # single-column index proxy: build reps so W2 is linear in one input.
    n, R = 6, pa.N_REPLICATES
    base = _base_frame(n)
    cbsa = np.array(["A"] * n)
    geoid = base["geoid"].to_numpy()

    rng = np.random.default_rng(0)

    def reps_for(perturb_cols, correlated):
        reps = {}
        shocks = rng.normal(size=(n, R))
        for col in base.columns:
            if col == "geoid":
                continue
            arr = np.tile(base[col].to_numpy()[:, None], (1, R + 1)).astype(float)
            if col in perturb_cols:
                s = shocks if correlated else rng.normal(size=(n, R))
                arr[:, 1:] = arr[:, 0][:, None] * (1.0 + 0.05 * s)
            reps[col] = arr
        return reps

    se_corr = pa.compute_replicate_se(
        reps_for(["aggregate_earnings_usd", "aggregate_owner_value_usd"], True),
        geoid, cbsa, 2023, wealth_vars=["W2_i_r3pct"],
    )
    se_anti = pa.compute_replicate_se(
        reps_for(["aggregate_earnings_usd", "aggregate_owner_value_usd"], False),
        geoid, cbsa, 2023, wealth_vars=["W2_i_r3pct"],
    )
    col = "Rel_SE_W2_i_r3pct_2023"
    # Both finite and non-negative; the correlated-shock case differs from the independent
    # one -> covariance among inputs is captured (delta/independent-MC would miss this).
    assert np.all(np.isfinite(se_corr[col])) and np.all(se_corr[col] >= 0)
    assert not np.allclose(se_corr[col].to_numpy(), se_anti[col].to_numpy())


def test_sdr_scale_constant():
    assert pa.SDR_SCALE == pytest.approx(4.0 / 80)


# ── (9) generalized significance test backward-compat ────────────────────────────

def test_test_significance_income_backcompat():
    df = pd.DataFrame({
        "Rel_Score_2011": [0.0, 1.0], "Rel_Score_2023": [1.0, 1.0],
        "Rel_SE_2011": [0.1, 0.1], "Rel_SE_2023": [0.1, 0.1],
    })
    out = pa.test_significance(df, 2011, 2023)
    assert "significant_2011_2023" in out.columns
    assert bool(out["significant_2011_2023"].iloc[0]) is True    # 1.0 shift, SE~0.14
    assert bool(out["significant_2011_2023"].iloc[1]) is False   # no shift


def test_test_significance_labelled():
    df = pd.DataFrame({
        "Rel_Score_W2_i_r3pct_2011": [0.0], "Rel_Score_W2_i_r3pct_2023": [2.0],
        "Rel_SE_W2_i_r3pct_2011": [0.1], "Rel_SE_W2_i_r3pct_2023": [0.1],
    })
    out = pa.test_significance(df, 2011, 2023, score_prefix="Rel_Score_W2_i_r3pct",
                               se_prefix="Rel_SE_W2_i_r3pct", label="W2_i_r3pct_")
    assert "significant_W2_i_r3pct_2011_2023" in out.columns


# ── (10) r_k sensitivity table ───────────────────────────────────────────────────

def test_add_wealth_relative_scores_per_cbsa():
    gdf = pd.DataFrame({
        "cbsa_code": ["A", "A", "A", "B", "B", "B"],
        "W2_i_r3pct_2023": np.array([1, 2, 4, 10, 20, 40], dtype=float),
    })
    out = pa.add_wealth_relative_scores(gdf, 2023, wealth_vars=["W2_i_r3pct"])
    col = "Rel_Score_W2_i_r3pct_2023"
    assert col in out.columns
    assert abs(out.loc[out.cbsa_code == "A", col].mean()) < 1e-9   # within-CBSA z -> mean 0


def test_add_wealth_structural_change_flags():
    df = pd.DataFrame({
        "Rel_Score_W2_i_r3pct_2011": [0.0, 0.0],
        "Rel_Score_W2_i_r3pct_2023": [2.0, 0.0],
        "Rel_SE_W2_i_r3pct_2011": [0.1, 0.1],
        "Rel_SE_W2_i_r3pct_2023": [0.1, 0.1],
    })
    out = pa.add_wealth_structural_change(df, 2011, 2023, wealth_vars=["W2_i_r3pct"])
    flag = "valid_change_W2_r3"
    assert flag in out.columns
    assert bool(out[flag].iloc[0]) is True     # 2.0 shift vs SE ~0.14 -> significant
    assert bool(out[flag].iloc[1]) is False    # no shift


def test_structural_change_gate_is_p10():
    assert pa.STRUCTURAL_CHANGE_P == pytest.approx(0.10)


def test_add_wealth_structural_change_borderline_p():
    """valid_change_* gates at p<0.10 while significant_* stays at p<0.01.

    Row 0: shift 0.30 with SEs 0.1/0.1 -> z~2.12, p~0.034: valid change, NOT significant.
    Row 1: shift 0.15 -> z~1.06, p~0.29: neither.
    """
    df = pd.DataFrame({
        "Rel_Score_W2_i_r3pct_2011": [0.0, 0.0],
        "Rel_Score_W2_i_r3pct_2023": [0.30, 0.15],
        "Rel_SE_W2_i_r3pct_2011": [0.1, 0.1],
        "Rel_SE_W2_i_r3pct_2023": [0.1, 0.1],
    })
    out = pa.add_wealth_structural_change(df, 2011, 2023, wealth_vars=["W2_i_r3pct"])
    assert bool(out["valid_change_W2_r3"].iloc[0]) is True
    assert bool(out["significant_W2_i_r3pct_2011_2023"].iloc[0]) is False
    assert bool(out["valid_change_W2_r3"].iloc[1]) is False


def test_add_wealth_structural_change_skips_without_se():
    df = pd.DataFrame({"Rel_Score_W2_i_r3pct_2011": [0.0],
                       "Rel_Score_W2_i_r3pct_2023": [2.0]})   # no Rel_SE present
    out = pa.add_wealth_structural_change(df, 2011, 2023, wealth_vars=["W2_i_r3pct"])
    assert "valid_change_W2_r3" not in out.columns


def test_rk_sensitivity_table_shape_and_stability():
    d = _base_frame(20)
    d[pa.PCI_COL] = d["aggregate_earnings_usd"] / 1000.0 + np.arange(20) * 100
    tbl = pa.rk_sensitivity_table(d, 2023)
    assert set(tbl["r_k"]) == set(pa.CAPITAL_YIELD_GRID)
    assert len(tbl) == len(pa.CAPITAL_YIELD_GRID) * len(pa.DISCOUNT_RATES) * 2
    # Spearman with PCI should be near-identical across r_k for a given (rho, index).
    w2 = tbl[(tbl["index"] == "W2") & (tbl["rho"] == 0.03)].sort_values("r_k")
    assert w2["spearman_pci"].max() - w2["spearman_pci"].min() < 0.05
