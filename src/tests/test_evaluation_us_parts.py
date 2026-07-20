"""End-to-end tests for US-mode run_evaluation on a synthetic results tree.

Builds a small but realistic full-US artifact set (3 CBSAs x 2 years, tract +
building predictions, an ACS panel, a city split), runs run_evaluation, and
checks the tables/figures/summary it produces — including the per-(CBSA, year)
scatter grid and the Spearman histogram the user asked for. Also covers fault
isolation between parts and the empty-results short-circuit.
"""

import json

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point

import src.evaluation as ev
from src.data import indicators

YEARS = [2016, 2018]
CBSAS = ["10420", "35620", "41860"]   # Akron, New York, San Francisco (real codes)
N_TRACTS = 12
N_BLD = 6                              # buildings per tract per year


def _build_synth(tmp_path, seed=0):
    """Create results_dir + processed_dir with correlated synthetic data.

    Predictions are label + noise so within-city Spearman is high (~0.9) and
    known-positive. Returns (results_dir, processed_dir, expected_cbsas).
    """
    rng = np.random.default_rng(seed)
    results_dir = tmp_path / "results" / "run_synth"
    results_dir.mkdir(parents=True)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    # Tract GEOIDs per CBSA (11-char). county prefix = cbsa[:2] + "001".
    geoids = {c: [f"{c[:2]}001{t:06d}" for t in range(N_TRACTS)] for c in CBSAS}

    change_col = indicators.valid_change_col("W2_r5")
    panel_rows = []
    for c in CBSAS:
        for i, g in enumerate(geoids[c]):
            panel_rows.append({
                "geoid_2023": g,
                "cbsa_code": c,
                change_col: 1 if i == 0 else 0,           # one changed tract each
                "per_capita_income_usd_2016": 25000 + 3000 * i,
                "per_capita_income_usd_2018": 26000 + 3100 * i,
                "geometry": Point(float(i), float(CBSAS.index(c))),
            })
    panel = gpd.GeoDataFrame(panel_rows, crs="EPSG:5070")
    panel.to_feather(processed_dir / ev._PANEL_FILENAME)

    # City split (cbsa_splits.feather).
    split = pd.DataFrame({
        "cbsa_code": CBSAS,
        "cbsa_title": ["Akron", "New York", "San Francisco"],
        "population": [700_000, 20_000_000, 4_700_000],
        "bracket": ["small", "mega", "large"],
        "split": ["test", "test", "test"],
        "holdout_year": [np.nan, np.nan, np.nan],
    })
    split.to_feather(processed_dir / "cbsa_splits.feather")

    # Building + tract predictions per year.
    for yr in YEARS:
        bld_rows = []
        for c in CBSAS:
            for i, g in enumerate(geoids[c]):
                # tract "true" wealth level within the city, z-scored later per cell
                base = i - N_TRACTS / 2
                for b in range(N_BLD):
                    label = base + rng.normal(0, 0.3)
                    pred = label + rng.normal(0, 0.5)   # correlated but noisy
                    bld_rows.append({
                        "Rel_Score": label, "predicted_value": pred,
                        "building_id": int(f"{CBSAS.index(c)}{i:02d}{b}"),
                        "GEOID": g, "year": yr, "type": "test",
                    })
        bld = pd.DataFrame(bld_rows)
        bld.to_csv(results_dir / f"{yr}_predictions.csv", index=False)

        tract = bld.groupby("GEOID").agg(
            Rel_Score=("Rel_Score", "mean"),
            predicted_value=("predicted_value", "mean"),
            predicted_value_std=("predicted_value", "std"),
        ).reset_index()
        tract.to_parquet(results_dir / f"predictions_by_tract_{yr}.parquet")

    return results_dir, processed_dir


def test_run_evaluation_produces_outputs(tmp_path):
    results_dir, processed_dir = _build_synth(tmp_path)
    summary = ev.run_evaluation(
        "run_synth",
        params={"footprints_source": "ms_us", "indicator": "W2_r5"},
        results_dir=results_dir,
        processed_dir=processed_dir,
    )
    out = results_dir / "evaluation"

    # Summary + headline.
    assert (out / "US_summary.json").exists()
    saved = json.loads((out / "US_summary.json").read_text())
    assert saved["mode"] == "us"
    assert "test/within_spearman" in summary
    assert summary["test/within_spearman"] > 0.5      # constructed positive corr

    # Part A tables.
    assert (out / "tables" / "US_A_cells_test.csv").exists()
    assert (out / "tables" / "US_A_summary.csv").exists()
    assert (out / "tables" / "US_A_by_bracket_test.csv").exists()

    # Part A figures: one scatter per (CBSA, year) cell + the Spearman histogram.
    scatter_dir = out / "figures" / "US_A_scatter_cells"
    pngs = list(scatter_dir.glob("*.png"))
    assert len(pngs) == len(CBSAS) * len(YEARS)
    assert (out / "figures" / "US_A_spearman_hist_test.pdf").exists()

    # Part B + C outputs.
    assert (out / "tables" / "US_B_rank_autocorr.csv").exists()
    assert (out / "tables" / "US_C_gb2_dollar_mapping.csv").exists()


def test_part_fault_isolation(tmp_path, monkeypatch):
    results_dir, processed_dir = _build_synth(tmp_path)
    # Force Part B to blow up; A and C must still run and A results survive.
    monkeypatch.setattr(ev, "part_b_us",
                        lambda ctx: (_ for _ in ()).throw(RuntimeError("boom")))
    summary = ev.run_evaluation(
        "run_synth",
        params={"footprints_source": "ms_us", "indicator": "W2_r5"},
        results_dir=results_dir,
        processed_dir=processed_dir,
    )
    assert "test/within_spearman" in summary          # Part A still returned
    assert (results_dir / "evaluation" / "tables" / "US_C_gb2_dollar_mapping.csv").exists()


def test_empty_results_returns_empty(tmp_path):
    empty = tmp_path / "results" / "empty_run"
    empty.mkdir(parents=True)
    out = ev.run_evaluation(
        "empty_run",
        params={"footprints_source": "ms_us"},
        results_dir=empty,
        processed_dir=tmp_path,
    )
    assert out == {}


def test_resolve_mode_prefers_params_and_artifacts(tmp_path):
    rd = tmp_path / "r"
    rd.mkdir()
    assert ev._resolve_mode(rd, {"footprints_source": "ms_us"}, "auto") == "us"
    assert ev._resolve_mode(rd, {"footprints_source": "doitt_nyc"}, "auto") == "nyc"
    assert ev._resolve_mode(rd, None, "us") == "us"          # explicit wins
    # NYC artifact present -> auto-detects nyc
    (rd / "predictions_2016.parquet").write_bytes(b"")
    assert ev._resolve_mode(rd, None, "auto") == "nyc"
