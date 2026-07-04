"""Synthetic-data tests for the US-scale build_dataset path (issues #26/#32/#33/#34).

Everything runs on a fake 2-CBSA panel + fake buildings_index written to tmp_path;
no real data, no network.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import box

import src.geo_utils as geo_utils
import src.build_dataset as bd
from src.data import indicators
from src.data.build_buildings_index import compute_building_id

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

YEARS = list(range(2011, 2024))
X0, Y0 = 1_500_000.0, 2_000_000.0     # plausible EPSG:5070 CONUS coords
TRACT_SIZE = 1000.0
N_TRACTS_PER_CBSA = 25
N_BLDG_PER_TRACT = 4

# 12 CBSAs spanning every population bracket of the whole-city split (#28):
# 4 mega-sized (top-4), 3 large (>1M), 3 medium (750K-1M), 2 small (500-750K).
CBSAS = [
    ("10010", 19_000_000), ("10020", 13_000_000), ("10030", 9_500_000), ("10040", 7_000_000),
    ("20010", 4_000_000), ("20020", 2_500_000), ("20030", 1_200_000),
    ("30010", 950_000), ("30020", 850_000), ("30030", 760_000),
    ("40010", 700_000), ("40020", 550_000),
]
N_CBSA = len(CBSAS)


def _make_panel():
    """12 CBSAs x 25 tracts, income + W2_r5 scores for all years + change flags."""
    rng = np.random.default_rng(7)
    rows, geoms = [], []
    for c, (cbsa, population) in enumerate(CBSAS):
        for t in range(N_TRACTS_PER_CBSA):
            geoid = f"{c + 1:02d}001{t:06d}"
            x = X0 + t * TRACT_SIZE + c * 200_000.0
            geoms.append(box(x, Y0, x + TRACT_SIZE, Y0 + TRACT_SIZE))
            row = {"geoid_2023": geoid, "cbsa_code": cbsa,
                   "total_population_2023": population // N_TRACTS_PER_CBSA,
                   "valid_change_inc": bool(t % 3 == 0),
                   "valid_change_W2_r5": bool(t % 4 == 0)}
            for y in YEARS:
                row[f"Rel_Score_{y}"] = rng.normal()
                row[f"Rel_Score_W2_i_r5pct_{y}"] = rng.normal()
            rows.append(row)
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=geo_utils.METRIC_CRS)


def _make_index(panel, out_dir):
    """buildings_index partition matching the synthetic panel tracts."""
    rng = np.random.default_rng(11)
    recs = []
    for _, tract in panel.iterrows():
        minx, miny, maxx, maxy = tract.geometry.bounds
        cx = rng.uniform(minx + 50, maxx - 50, N_BLDG_PER_TRACT)
        cy = rng.uniform(miny + 50, maxy - 50, N_BLDG_PER_TRACT)
        recs.append(pd.DataFrame({
            "building_id": compute_building_id(cx, cy),
            "cx": cx, "cy": cy,
            "tract_id": tract["geoid_2023"],
        }))
    df = pd.concat(recs, ignore_index=True)
    part_dir = out_dir / "buildings_index" / "state=TestState"
    part_dir.mkdir(parents=True)
    df.to_parquet(part_dir / "part.parquet", index=False)
    return df


@pytest.fixture
def env(tmp_path, monkeypatch):
    panel = _make_panel()
    index_df = _make_index(panel, tmp_path)
    monkeypatch.setattr(bd, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(bd, "FIGURES_DIR", tmp_path)
    monkeypatch.setattr(bd, "process_acs_panel", lambda: panel.copy())
    for sub in ["train_datasets", "test_datasets", "val_datasets"]:
        (tmp_path / sub).mkdir()
    return {"panel": panel, "index": index_df, "tmp": tmp_path}


# ── load_buildings_index ─────────────────────────────────────────────────────

def test_load_buildings_index_schema(env):
    df = bd.load_buildings_index(index_dir=env["tmp"] / "buildings_index")
    assert {"building_id", "centroid_x", "centroid_y", "GEOID", "state"} <= set(df.columns)
    assert (df["CONSTRUCTION_YEAR"] == 0).all()
    assert (df["DEMOLITION_YEAR"] == 2999).all()
    assert len(df) == len(env["index"])


# ── load_income_dataset (flat table) ─────────────────────────────────────────

def test_flat_table_schema_and_static_universe(env):
    years = [2012, 2016, 2022]
    flat = bd.load_income_dataset(years, tau_meters=100, indicator="W2_r5")
    expected_cols = {
        "building_id", "GEOID", "cbsa_code", "year",
        "bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy",
        "centroid_x", "centroid_y",
        "Rel_Score", "Valid_Structural_Change", "score_bin", "dist_to_center",
    }
    assert set(flat.columns) == expected_cols
    # Static universe: every building appears in every requested year.
    assert len(flat) == len(env["index"]) * len(years)
    assert set(flat["year"]) == set(years)
    assert flat["cbsa_code"].nunique() == N_CBSA


def test_indicator_selects_correct_label(env):
    flat_w2 = bd.load_income_dataset([2016], indicator="W2_r5")
    flat_inc = bd.load_income_dataset([2016], indicator="inc")
    panel = env["panel"].set_index("geoid_2023")
    for flat, col, vc in [
        (flat_w2, "Rel_Score_W2_i_r5pct_2016", "valid_change_W2_r5"),
        (flat_inc, "Rel_Score_2016", "valid_change_inc"),
    ]:
        sample = flat.sample(20, random_state=0)
        expected = panel.loc[sample["GEOID"], col].to_numpy()
        assert np.allclose(sample["Rel_Score"].to_numpy(), expected)
        expected_vc = panel.loc[sample["GEOID"], vc].to_numpy()
        assert (sample["Valid_Structural_Change"].to_numpy() == expected_vc).all()


def test_bbox_is_centroid_plus_minus_tau(env):
    flat = bd.load_income_dataset([2016], tau_meters=100, indicator="W2_r5")
    assert np.allclose(flat["bbox_maxx"] - flat["centroid_x"], 100.0)
    assert np.allclose(flat["centroid_x"] - flat["bbox_minx"], 100.0)
    assert np.allclose(flat["bbox_maxy"] - flat["bbox_miny"], 200.0)


def test_artifact_name_carries_source_indicator_crs(env):
    bd.load_income_dataset([2016], indicator="W2_r5")
    files = list(env["tmp"].glob("temporal_data_*.parquet"))
    assert len(files) == 1
    name = files[0].name
    assert "ms_us" in name and "W2_r5" in name and "epsg5070" in name


def test_missing_indicator_columns_raise(env, monkeypatch):
    panel = env["panel"].drop(columns=["valid_change_W2_r5"])
    monkeypatch.setattr(bd, "process_acs_panel", lambda: panel.copy())
    with pytest.raises(KeyError, match="W2_r5"):
        bd.load_income_dataset([2016], indicator="W2_r5")


def test_dist_to_center_is_within_cbsa_scale(env):
    flat = bd.load_income_dataset([2016], indicator="W2_r5")
    # Tracts span ~25 km per CBSA; distance to own-CBSA center must stay local,
    # far below the ~200 km separation between the two synthetic CBSAs.
    assert flat["dist_to_center"].max() < 50.0
    assert (flat["dist_to_center"] >= 0).all()


# ── get_closest_acs_year ─────────────────────────────────────────────────────

def test_get_closest_acs_year_clamps_to_panel():
    assert bd.get_closest_acs_year(2010) == 2011
    assert bd.get_closest_acs_year(2016) == 2016
    assert bd.get_closest_acs_year(2024) == 2023


# ── create_train_test_dataframes (whole-city split integrity, #28) ──────────

def _run_split(env, years=(2014, 2016, 2022)):
    flat = bd.load_income_dataset(list(years), indicator="W2_r5")
    # NAIP-mode placeholder columns expected by the split/export code
    flat["dataset"] = "NAIP"
    for c in ["row_start", "row_stop", "col_start", "col_stop"]:
        flat[c] = 0
    df_train, df_vals, df_test, df_dead = bd.create_train_test_dataframes(
        flat, "testsave", small_sample=False, indicator="W2_r5",
        naip_coverage_csv=env["tmp"] / "no_such_coverage.csv",  # force deterministic fallback
    )
    return df_train, df_vals, df_test, df_dead


def test_splits_are_disjoint_rows_and_whole_cities(env):
    df_train, df_vals, df_test, df_dead = _run_split(env)

    # Row-level disjointness
    train_ids = set(zip(df_train["building_id"], df_train["year"]))
    test_ids = set(zip(df_test["building_id"], df_test["year"]))
    assert not train_ids & test_ids
    for name, df_val in df_vals.items():
        val_ids = set(zip(df_val["building_id"], df_val["year"]))
        assert not train_ids & val_ids, name
        assert not test_ids & val_ids, name

    # City-level purity: no CBSA straddles train / val_cities / test
    train_cbsas = set(df_train["cbsa_code"])
    test_cbsas = set(df_test["cbsa_code"])
    val_cities_cbsas = set(df_vals["val_cities"]["cbsa_code"])
    assert not train_cbsas & test_cbsas
    assert not train_cbsas & val_cities_cbsas
    assert not test_cbsas & val_cities_cbsas
    assert test_cbsas and val_cities_cbsas and train_cbsas

    # Dead zone is gone under the whole-city split
    assert len(df_dead) == 0


def test_val_temporal_is_one_holdout_year_per_train_city(env):
    df_train, df_vals, df_test, _ = _run_split(env)
    df_vt = df_vals["val_temporal"]

    for cbsa, grp in df_vt.groupby("cbsa_code"):
        held_years = set(grp["year"])
        assert len(held_years) == 1, f"CBSA {cbsa} holds out {held_years}"
        held = held_years.pop()
        # The holdout year never appears in that city's train rows
        assert held not in set(df_train.loc[df_train["cbsa_code"] == cbsa, "year"])
    # Every train city held out exactly one year
    assert set(df_vt["cbsa_code"]) == set(df_train["cbsa_code"])


def test_cbsa_splits_artifact_consistent(env):
    from src.data import cbsa_brackets as cb
    df_train, df_vals, df_test, _ = _run_split(env)

    meta = pd.read_feather(env["tmp"] / "cbsa_splits.feather")
    assert {"cbsa_code", "cbsa_title", "population", "n_tracts",
            "bracket", "split", "holdout_year"} <= set(meta.columns)
    assert set(meta["split"]) == {"train", "val", "test"}
    assert set(meta["bracket"]) == {"mega", "large", "medium", "small"}

    split_of = dict(zip(meta["cbsa_code"].astype(str), meta["split"]))
    assert all(split_of[str(c)] == "train" for c in df_train["cbsa_code"].unique())
    assert all(split_of[str(c)] == "test" for c in df_test["cbsa_code"].unique())
    assert all(split_of[str(c)] == "val" for c in df_vals["val_cities"]["cbsa_code"].unique())

    # holdout_year set exactly for train cities; deterministic fallback year
    # (middle of [2014, 2016, 2022] -> 2018 midpoint -> 2016 is closest)
    train_meta = meta[meta["split"] == "train"]
    assert train_meta["holdout_year"].notna().all()
    assert set(train_meta["holdout_year"].astype(int)) == {2016}
    assert meta.loc[meta["split"] != "train", "holdout_year"].isna().all()

    hmap = cb.holdout_year_map(meta)
    assert set(hmap.values()) == {2016}


def test_tract_splits_artifact_types(env):
    _run_split(env)
    tract_splits = gpd.read_feather(env["tmp"] / "tract_splits.feather")
    assert set(tract_splits["type"]) <= {"train", "val", "test"}
    assert len(tract_splits) == N_CBSA * N_TRACTS_PER_CBSA
