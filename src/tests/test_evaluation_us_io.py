"""Tests for the US-mode loaders in src/evaluation.py.

All synthetic, written under tmp_path. Verifies GEOID zero-padding survives the
CSV round-trip, NaN predictions are dropped, year globbing, and the panel/CBSA
join (with an intentionally unmatched GEOID).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point

import src.evaluation as ev
from src.data import indicators


def _write_pred_csv(results_dir, year, rows):
    """rows: list of dicts with the legacy CSV columns."""
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / f"{year}_predictions.csv", index=False)


def _panel(tmp_path):
    """Minimal us_metros_panel_2011_2023.feather with the columns loaders need."""
    change_col = indicators.valid_change_col("W2_r5")
    gdf = gpd.GeoDataFrame(
        {
            "geoid_2023": ["01001020100", "36061000100", "36061000200"],
            "cbsa_code": ["33860", "35620", "35620"],
            change_col: [0, 1, 0],
            "per_capita_income_usd_2016": [30000.0, 55000.0, 42000.0],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:5070",
    )
    gdf.to_feather(tmp_path / ev._PANEL_FILENAME)
    return tmp_path


def test_available_years_globs_and_sorts(tmp_path):
    for y in (2016, 2010, 2024):
        _write_pred_csv(tmp_path, y, [{"Rel_Score": 0.1, "predicted_value": 0.2,
                                       "building_id": 1, "GEOID": "36061000100",
                                       "year": y, "type": "test"}])
    # a non-matching file must be ignored
    (tmp_path / "predictions_by_tract_2016.parquet").write_bytes(b"")
    assert ev._available_years(tmp_path) == [2010, 2016, 2024]


def test_load_building_preds_geoid_zfill_and_nan_drop(tmp_path):
    # GEOID for state 01 stored as an int loses its leading zero in the CSV.
    _write_pred_csv(tmp_path, 2016, [
        {"Rel_Score": 0.5, "predicted_value": 0.4, "building_id": 1,
         "GEOID": 1001020100, "year": 2016, "type": "test"},          # int GEOID
        {"Rel_Score": 0.1, "predicted_value": np.nan, "building_id": 2,
         "GEOID": "36061000100", "year": 2016, "type": "test"},        # NaN pred -> drop
    ])
    df = ev._load_building_preds_us(tmp_path, [2016])
    assert len(df) == 1                       # NaN-pred row dropped
    assert df["GEOID"].iloc[0] == "01001020100"   # zero-padded to 11 chars
    assert {"pred", "label"}.issubset(df.columns)
    assert df["pred"].iloc[0] == pytest.approx(0.4)
    assert df["label"].iloc[0] == pytest.approx(0.5)


def test_load_panel_us_columns_and_income(tmp_path):
    proc = _panel(tmp_path)
    panel = ev._load_panel_us("W2_r5", proc, want_income=True)
    assert {"GEOID", "cbsa_code", "change", "geometry"}.issubset(panel.columns)
    assert panel["GEOID"].iloc[0] == "01001020100"
    assert any(c.startswith("per_capita_income_usd_") for c in panel.columns)
    assert panel["change"].tolist() == [0, 1, 0]


def test_attach_cbsa_drops_unmatched(tmp_path):
    proc = _panel(tmp_path)
    panel = ev._load_panel_us("W2_r5", proc)
    bld = pd.DataFrame({
        "building_id": [1, 2, 3],
        "GEOID": ["36061000100", "36061000200", "99999999999"],  # last unmatched
        "year": [2016, 2016, 2016],
        "type": ["test", "test", "test"],
        "pred": [0.1, 0.2, 0.3],
        "label": [0.0, 0.1, 0.2],
    })
    out = ev._attach_cbsa(bld, panel)
    assert len(out) == 2                     # unmatched GEOID dropped
    assert set(out["cbsa"].unique()) == {"35620"}
    assert "change" in out.columns
