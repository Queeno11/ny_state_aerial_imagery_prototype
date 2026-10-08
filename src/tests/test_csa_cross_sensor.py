"""Synthetic-data tests for src/csa_cross_sensor.py and probe_ortho_throughput.py.

The agreement statistic is what decides whether Chicago runs on ortho at all, so
it needs to be right on data whose answer we control. No network, no model.

The property that matters most: **within-tract rho, not pooled rho, is the
headline**. Pooled correlation over a city is dominated by between-tract
variation — any sensor that can tell downtown from a residential fringe scores
well on it — while the event study lives on within-tract movement. The tests
below construct a case where those two numbers diverge sharply, which is exactly
the situation the report has to catch.
"""

import numpy as np
import pandas as pd
import pytest

from src import csa_cross_sensor as ccs


def _frame(n_tracts=20, per_tract=10, *, noise=0.05, within_signal=1.0,
           between_spread=5.0, seed=0):
    """Paired predictions with controllable within- vs between-tract agreement."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tracts):
        level = t * between_spread
        for _ in range(per_tract):
            base = rng.normal(0, 1)
            naip = level + base
            ortho = level + within_signal * base + rng.normal(0, noise)
            rows.append({"GEOID": f"17031{t:06d}", "year": 2019,
                         "pred_naip": naip, "pred_ortho": ortho})
    return pd.DataFrame(rows)


# ─── agreement statistics ─────────────────────────────────────────────────────

def test_perfect_agreement_scores_one_on_both_measures():
    df = _frame(noise=0.0)
    out = ccs.agreement(df)
    assert out["pooled_rho"] == pytest.approx(1.0, abs=1e-6)
    assert out["within_rho"] == pytest.approx(1.0, abs=1e-6)


def test_within_rho_catches_what_pooled_rho_hides():
    """REGRESSION GUARD: this is the whole reason within_rho is the headline.
    Here the sensors agree perfectly on which tract is richer but carry NO
    within-tract information — pooled rho stays high while within rho collapses.
    Reporting only the pooled number would greenlight a sensor that cannot
    support the event study."""
    rng = np.random.default_rng(1)
    rows = []
    for t in range(30):
        level = t * 10.0
        for _ in range(12):
            rows.append({"GEOID": f"17031{t:06d}", "year": 2019,
                         "pred_naip": level + rng.normal(0, 1),
                         "pred_ortho": level + rng.normal(0, 1)})
    out = ccs.agreement(pd.DataFrame(rows))
    assert out["pooled_rho"] > 0.9
    assert abs(out["within_rho"]) < 0.35


def test_within_rho_is_weighted_by_tract_size():
    """A 100-building tract should not count the same as a 5-building one."""
    small = _frame(n_tracts=1, per_tract=6, within_signal=-1.0, noise=0.01, seed=2)
    small["GEOID"] = "17031000001"
    big = _frame(n_tracts=1, per_tract=200, within_signal=1.0, noise=0.01, seed=3)
    big["GEOID"] = "17031000002"
    out = ccs.agreement(pd.concat([small, big], ignore_index=True))
    assert out["within_rho"] > 0.8      # dominated by the large tract


def test_level_and_scale_shift_do_not_hurt_rho():
    """The loss is purely ordinal, so a sensor that is uniformly brighter or
    more spread out is fine — only the ranking has to agree."""
    df = _frame(noise=0.0)
    df["pred_ortho"] = df["pred_ortho"] * 3.0 + 17.0
    out = ccs.agreement(df)
    assert out["within_rho"] == pytest.approx(1.0, abs=1e-6)
    assert out["mean_shift"] != 0
    assert out["sd_ratio"] == pytest.approx(3.0, rel=0.05)


def test_shift_diagnostics_are_reported():
    df = _frame(noise=0.0)
    df["pred_ortho"] = df["pred_ortho"] + 2.0
    out = ccs.agreement(df)
    assert out["mean_shift"] == pytest.approx(2.0, abs=1e-6)


def test_missing_predictions_are_dropped_pairwise():
    df = _frame(n_tracts=5, per_tract=10, noise=0.0)
    df.loc[df.index[:7], "pred_ortho"] = np.nan
    out = ccs.agreement(df)
    assert out["n"] == len(df) - 7


def test_too_few_pairs_returns_nan_not_a_spurious_number():
    df = _frame(n_tracts=1, per_tract=3)
    out = ccs.agreement(df)
    assert np.isnan(out["within_rho"])


def test_constant_tract_is_skipped_not_counted_as_disagreement():
    """A tract where one sensor returns a constant has undefined Spearman; it
    must not be silently folded in as rho=0."""
    df = _frame(n_tracts=3, per_tract=8, noise=0.0)
    flat = df["GEOID"] == df["GEOID"].iloc[0]
    df.loc[flat, "pred_ortho"] = 1.0
    out = ccs.agreement(df)
    assert out["n_tracts_scored"] == 2
    assert out["within_rho"] == pytest.approx(1.0, abs=1e-6)


# ─── report ───────────────────────────────────────────────────────────────────

def test_report_writes_per_year_and_overall_rows(tmp_path):
    df = pd.concat([_frame(noise=0.0).assign(year=y) for y in (2019, 2021)],
                   ignore_index=True)
    out = ccs.report(df, city="chicago", tables_dir=tmp_path)
    assert list(out["year"]) == [2019, 2021, "all"]
    assert (tmp_path / "csa_cross_sensor_agreement_chicago.csv").exists()


def test_report_overall_row_pools_every_year(tmp_path):
    df = pd.concat([_frame(noise=0.0).assign(year=y) for y in (2019, 2021)],
                   ignore_index=True)
    out = ccs.report(df, city="chicago", tables_dir=tmp_path)
    assert out.iloc[-1]["n"] == len(df)


def test_overlap_years_are_inside_the_illinois_naip_grid():
    """The comparison needs a year BOTH sensors flew; Cook ortho is annual, so
    the binding constraint is the Illinois NAIP grid."""
    from src.data import ortho_fetcher as of
    cook = set(of.available_years("chicago"))
    assert set(ccs.DEFAULT_OVERLAP_YEARS) <= cook


# ─── probe ────────────────────────────────────────────────────────────────────

def test_probe_level_result_computes_throughput():
    from src import probe_ortho_throughput as probe

    r = probe.LevelResult(workers=4, max_rps=10, n_ok=40, n_fail=0,
                          n_throttled=0, elapsed_s=2.0, p50_ms=90, p95_ms=180)
    assert r.throughput == pytest.approx(20.0)
    assert "crops/s" in r.line()


def test_probe_uses_production_crop_geometry():
    """The probe's numbers only transfer if it fetches what production fetches:
    tau=100 m -> a 200 m window at image_size=224."""
    from src import probe_ortho_throughput as probe

    assert probe.CROP_SIZE_M == 200.0
    assert probe.OUT_PIXELS == 224
    assert probe.BYTES_PER_CROP == 224 * 224 * 4


def test_probe_verdict_reports_no_usable_level(capsys):
    from src import probe_ortho_throughput as probe

    throttled = probe.LevelResult(workers=8, max_rps=20, n_ok=5, n_fail=1,
                                  n_throttled=3, elapsed_s=1.0, p50_ms=1, p95_ms=2)
    probe.verdict([throttled], n_crops_total=1000)
    out = capsys.readouterr().out
    assert "Do not escalate" in out


def test_probe_verdict_recommends_online_when_fast(capsys):
    from src import probe_ortho_throughput as probe

    fast = probe.LevelResult(workers=8, max_rps=20, n_ok=100, n_fail=0,
                             n_throttled=0, elapsed_s=1.0, p50_ms=10, p95_ms=20)
    probe.verdict([fast], n_crops_total=100_000)
    out = capsys.readouterr().out
    assert "ONLINE" in out


def test_probe_verdict_warns_about_per_process_rate_limits(capsys):
    """The limiter is per-process, so N shards multiply the load the municipal
    service sees by N — the verdict must say so."""
    from src import probe_ortho_throughput as probe

    slow = probe.LevelResult(workers=4, max_rps=10, n_ok=10, n_fail=0,
                             n_throttled=0, elapsed_s=10.0, p50_ms=900, p95_ms=1800)
    probe.verdict([slow], n_crops_total=5_000_000)
    out = capsys.readouterr().out
    assert "DIVIDE" in out or "per process" in out
