"""Synthetic-data tests for src/data/verify_final_data.py.

A small in-memory panel (2 CBSAs, per-CBSA z-scored indicator columns for every
indicator token) and a matching buildings index are built from scratch — no real
ACS/buildings data, no network. Exercises both the summary tables and the PASS/FAIL
structural checks.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import box

import src.geo_utils as geo_utils
from src.data import indicators as ind
from src.data import build_buildings_index as bbi
from src.data import verify_final_data as vfd

YEARS = [2022, 2023]
BASE_YEAR = 2023
# Two CBSAs at plausible EPSG:5070 coordinates, laid out as a grid of 1 km tracts.
_CBSAS = {"10580": (1_500_000.0, 2_000_000.0), "35620": (1_800_000.0, 2_300_000.0)}
_N_PER_CBSA = 40


def _zscore(rng, n):
    v = rng.normal(size=n)
    return (v - v.mean()) / v.std(ddof=1)   # sample std, matches pandas .std()


@pytest.fixture
def panel():
    """GeoDataFrame mimicking us_metros_panel: geoid/cbsa/geometry + per-CBSA z-scores,
    training labels, and boolean valid_change flags for every indicator token."""
    rng = np.random.default_rng(0)
    rows, geoms = [], []
    for ci, (cbsa, (x0, y0)) in enumerate(_CBSAS.items()):
        # Per-CBSA standardized scores so the z-moment check reads mean~0/std~1.
        scores = {tok: {y: _zscore(rng, _N_PER_CBSA) for y in YEARS} for tok in ind.INDICATORS}
        for i in range(_N_PER_CBSA):
            geoid = f"{36 + ci:02d}{i:09d}"           # 11-char, unique
            rec = {f"geoid_{BASE_YEAR}": geoid, "cbsa_code": cbsa}
            for y in YEARS:
                for tok in ind.INDICATORS:
                    rec[ind.score_col(tok, y)] = scores[tok][y][i]
                rec[f"Training_Label_{y}"] = scores["inc"][y][i]
            rec[ind.valid_change_col("inc")] = bool(i % 3 == 0)
            for tok in ind.TOKEN_TO_VAR:
                rec[ind.valid_change_col(tok)] = bool(i % 4 == 0)
            rows.append(rec)
            gx, gy = x0 + (i % 8) * 1000, y0 + (i // 8) * 1000
            geoms.append(box(gx, gy, gx + 1000, gy + 1000))
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=geo_utils.METRIC_CRS)


@pytest.fixture
def buildings(panel):
    """Buildings index: 10 buildings inside each tract's polygon, ids from centroids."""
    rng = np.random.default_rng(1)
    parts = []
    for _, tract in panel.iterrows():
        minx, miny, maxx, maxy = tract.geometry.bounds
        cx = rng.uniform(minx + 50, maxx - 50, 10)
        cy = rng.uniform(miny + 50, maxy - 50, 10)
        parts.append(pd.DataFrame({
            "building_id": bbi.compute_building_id(cx, cy),
            "cx": cx, "cy": cy,
            "tract_id": tract[f"geoid_{BASE_YEAR}"],
        }))
    idx = pd.concat(parts, ignore_index=True)
    return idx.drop_duplicates(subset="building_id").reset_index(drop=True)


# ── summary statistics ────────────────────────────────────────────────────────

def test_describe_series_matches_pandas():
    s = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0])
    rec = vfd.describe_series(s)
    assert rec["n"] == 5 and rec["n_nonnull"] == 4 and rec["n_null"] == 1
    assert rec["min"] == 1.0 and rec["max"] == 5.0
    assert rec["median"] == pytest.approx(2.5)
    assert rec["mean"] == pytest.approx(2.75)


def test_indicator_summary_shapes_and_global_row(panel):
    global_df, city_df = vfd.indicator_summary(panel, "inc", YEARS, BASE_YEAR)
    assert list(global_df["year"]) == YEARS
    # per-CBSA rows + one pooled GLOBAL row, which must be last and cover all tracts.
    assert set(city_df["cbsa_code"]) == set(_CBSAS) | {"GLOBAL"}
    assert city_df.iloc[-1]["cbsa_code"] == "GLOBAL"
    assert city_df.iloc[-1]["n_tracts"] == len(panel)
    # Each CBSA's within-city z-score is standardized.
    per_city = city_df[city_df["cbsa_code"] != "GLOBAL"]
    assert per_city["mean"].abs().max() < 1e-6
    assert (per_city["std"] - 1.0).abs().max() < 1e-6


def test_buildings_summary_counts(panel, buildings):
    tracts = vfd.load_panel_tracts_for_join(panel, BASE_YEAR)
    glob, cov = vfd.buildings_summary(buildings, tracts)
    assert glob["n_buildings"] == len(buildings)
    assert glob["n_duplicate_ids"] == 0
    assert glob["n_cbsas_covered"] == len(_CBSAS)
    # Every tract seeded 10 buildings, so no covered tract is empty.
    assert (cov["tracts_without_buildings"] == 0).all()
    assert cov["n_buildings"].sum() == len(buildings)


def test_buildings_summary_flags_empty_tracts(panel, buildings):
    """Dropping one tract's buildings shows up as tracts_without_buildings."""
    dropped_tract = panel.iloc[0][f"geoid_{BASE_YEAR}"]
    trimmed = buildings[buildings["tract_id"] != dropped_tract]
    tracts = vfd.load_panel_tracts_for_join(panel, BASE_YEAR)
    _, cov = vfd.buildings_summary(trimmed, tracts)
    assert cov["tracts_without_buildings"].sum() == 1


# ── structural checks ─────────────────────────────────────────────────────────

def test_panel_checks_pass_on_valid_panel(panel):
    c = vfd.Checker()
    vfd.check_panel_schema(panel, "inc", YEARS, BASE_YEAR, c)
    vfd.check_indicator_sanity(panel, "inc", YEARS, BASE_YEAR, c)
    assert not c.failures, c.failures


def test_panel_check_flags_bad_geoid(panel):
    panel.loc[panel.index[0], f"geoid_{BASE_YEAR}"] = "123"   # not 11 chars
    c = vfd.Checker()
    vfd.check_panel_schema(panel, "inc", YEARS, BASE_YEAR, c)
    assert any("11-char" in f for f in c.failures)


def test_panel_check_flags_broken_zscore(panel):
    # Blow up one CBSA-year's scale so std deviates from 1.
    col = ind.score_col("inc", BASE_YEAR)
    mask = panel["cbsa_code"] == "10580"
    panel.loc[mask, col] = panel.loc[mask, col] * 10.0
    c = vfd.Checker()
    vfd.check_indicator_sanity(panel, "inc", YEARS, BASE_YEAR, c)
    assert any("z-moments" in f for f in c.failures)


def test_buildings_checks_pass(panel, buildings):
    tracts = vfd.load_panel_tracts_for_join(panel, BASE_YEAR)
    glob, _ = vfd.buildings_summary(buildings, tracts)
    c = vfd.Checker()
    vfd.check_buildings(buildings, tracts, glob, c)
    assert not c.failures, c.failures


def test_buildings_check_flags_unknown_tract(panel, buildings):
    buildings.loc[buildings.index[0], "tract_id"] = "99999999999"   # not in panel
    tracts = vfd.load_panel_tracts_for_join(panel, BASE_YEAR)
    glob, _ = vfd.buildings_summary(buildings, tracts)
    c = vfd.Checker()
    vfd.check_buildings(buildings, tracts, glob, c)
    assert any("tract_ids exist" in f for f in c.failures)


def test_duplicate_ids_warn_not_fatal(panel, buildings):
    """A handful of cross-state duplicate ids WARNs but must not fail (documented)."""
    dup = pd.concat([buildings, buildings.iloc[:3]], ignore_index=True)
    tracts = vfd.load_panel_tracts_for_join(panel, BASE_YEAR)
    glob, _ = vfd.buildings_summary(dup, tracts)
    assert glob["n_duplicate_ids"] == 3
    c = vfd.Checker()
    vfd.check_buildings(dup, tracts, glob, c)
    assert not any("globally unique" in f for f in c.failures)


# ── end-to-end run() on disk ──────────────────────────────────────────────────

def test_run_end_to_end(tmp_path, panel, buildings):
    panel_path = tmp_path / "panel.feather"
    panel.to_feather(panel_path)
    index_dir = tmp_path / "buildings_index" / "state=Test"
    index_dir.mkdir(parents=True)
    buildings.to_parquet(index_dir / "part.parquet", index=False)
    rc = vfd.run(panel_path, tmp_path / "buildings_index", "inc",
                 tmp_path / "out", BASE_YEAR, YEARS)
    assert rc == 0
    assert (tmp_path / "out" / f"panel_inc_stats_by_cbsa_{BASE_YEAR}.csv").exists()
    assert (tmp_path / "out" / "buildings_counts_by_cbsa.csv").exists()
