"""Synthetic-data tests for the US-scale build_dataset path (issues #26/#32/#33/#34).

Everything runs on a fake 12-CBSA panel + fake buildings_index written to tmp_path;
no real data, no network. Since the normalization refactor, ms_us returns a
LazyPairTable — the flat table is materialized on demand, never persisted.
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
from src.data.pair_table import FLAT_COLUMNS, LazyPairTable

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
N_BUILDINGS = N_CBSA * N_TRACTS_PER_CBSA * N_BLDG_PER_TRACT


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


def _full_flat(table):
    return table.materialize(0, len(table))


# ── load_buildings_index (legacy loader, still used by non-training tools) ──

def test_load_buildings_index_schema(env):
    df = bd.load_buildings_index(index_dir=env["tmp"] / "buildings_index")
    assert {"building_id", "centroid_x", "centroid_y", "GEOID", "state"} <= set(df.columns)
    assert (df["CONSTRUCTION_YEAR"] == 0).all()
    assert (df["DEMOLITION_YEAR"] == 2999).all()
    assert len(df) == len(env["index"])


# ── load_income_dataset (lazy pair table) ────────────────────────────────────

def test_pair_table_schema_and_static_universe(env):
    years = [2012, 2016, 2022]
    table = bd.load_income_dataset(years, tau_meters=100, indicator="W2_r5")
    assert isinstance(table, LazyPairTable)
    # Static universe: every building appears in every requested year.
    assert len(table) == len(env["index"]) * len(years)
    flat = _full_flat(table)
    assert list(flat.columns) == FLAT_COLUMNS
    assert set(flat["year"]) == set(years)
    assert flat["cbsa_code"].nunique() == N_CBSA
    # Slim dtypes on the static frame — the whole point of the refactor.
    assert isinstance(table.buildings["GEOID"].dtype, pd.CategoricalDtype)
    assert isinstance(table.buildings["cbsa_code"].dtype, pd.CategoricalDtype)
    assert table.buildings["centroid_x"].dtype == np.float32


def test_indicator_selects_correct_label(env):
    flat_w2 = _full_flat(bd.load_income_dataset([2016], indicator="W2_r5"))
    flat_inc = _full_flat(bd.load_income_dataset([2016], indicator="inc"))
    panel = env["panel"].set_index("geoid_2023")
    for flat, col, vc in [
        (flat_w2, "Rel_Score_W2_i_r5pct_2016", "valid_change_W2_r5"),
        (flat_inc, "Rel_Score_2016", "valid_change_inc"),
    ]:
        sample = flat.sample(20, random_state=0)
        expected = panel.loc[sample["GEOID"], col].to_numpy()
        assert np.allclose(sample["Rel_Score"].to_numpy(), expected, rtol=1e-6)
        expected_vc = panel.loc[sample["GEOID"], vc].to_numpy()
        assert (sample["Valid_Structural_Change"].to_numpy() == expected_vc).all()


def test_bbox_is_centroid_plus_minus_tau(env):
    flat = _full_flat(bd.load_income_dataset([2016], tau_meters=100, indicator="W2_r5"))
    assert np.allclose(flat["bbox_maxx"] - flat["centroid_x"], 100.0)
    assert np.allclose(flat["centroid_x"] - flat["bbox_minx"], 100.0)
    assert np.allclose(flat["bbox_maxy"] - flat["bbox_miny"], 200.0)


def test_score_bin_matches_rowwise_qcut(env):
    """Weighted tract-level bins must equal the legacy per-row qcut exactly."""
    flat = _full_flat(bd.load_income_dataset([2014, 2016], indicator="W2_r5"))
    expected = (
        flat.groupby("year")["Rel_Score"]
        .transform(lambda x: pd.qcut(x, q=5, labels=False, duplicates="drop"))
    )
    np.testing.assert_array_equal(flat["score_bin"].to_numpy(),
                                  expected.to_numpy().astype("int8"))


def test_artifact_names_carry_source_indicator_crs(env):
    bd.load_income_dataset([2016], indicator="W2_r5")
    bfiles = list(env["tmp"].glob("pair_buildings_*.parquet"))
    lfiles = list(env["tmp"].glob("pair_labels_*.parquet"))
    assert len(bfiles) == 1 and len(lfiles) == 1
    assert "ms_us" in bfiles[0].name and "epsg5070" in bfiles[0].name
    assert "W2_r5" in lfiles[0].name and "ms_us" in lfiles[0].name
    # No 575M-row flat parquet on the lazy path.
    assert not list(env["tmp"].glob("temporal_data_*.parquet"))


def test_artifact_cache_roundtrip(env):
    table1 = bd.load_income_dataset([2016], indicator="W2_r5")
    table2 = bd.load_income_dataset([2016], indicator="W2_r5")  # from parquet cache
    flat1, flat2 = _full_flat(table1), _full_flat(table2)
    pd.testing.assert_frame_equal(flat1, flat2)


def test_missing_indicator_columns_raise(env, monkeypatch):
    panel = env["panel"].drop(columns=["valid_change_W2_r5"])
    monkeypatch.setattr(bd, "process_acs_panel", lambda: panel.copy())
    with pytest.raises(KeyError, match="W2_r5"):
        bd.load_income_dataset([2016], indicator="W2_r5")


def test_dist_to_center_is_within_cbsa_scale(env):
    table = bd.load_income_dataset([2016], indicator="W2_r5")
    # Tracts span ~25 km per CBSA; distance to own-CBSA center must stay local,
    # far below the ~200 km separation between the synthetic CBSAs.
    assert table.buildings["dist_to_center"].max() < 50.0
    assert (table.buildings["dist_to_center"] >= 0).all()


# ── get_closest_acs_year ─────────────────────────────────────────────────────

def test_get_closest_acs_year_clamps_to_panel():
    assert bd.get_closest_acs_year(2010) == 2011
    assert bd.get_closest_acs_year(2016) == 2016
    assert bd.get_closest_acs_year(2024) == 2023


# ── create_train_test_dataframes (whole-city split integrity, #28) ──────────

def _run_split(env, years=(2014, 2016, 2022)):
    table = bd.load_income_dataset(list(years), indicator="W2_r5")
    lazy_train, df_vals, lazy_test, df_dead = bd.create_train_test_dataframes(
        table, "testsave", small_sample=False, indicator="W2_r5",
        naip_coverage_csv=env["tmp"] / "no_such_coverage.csv",  # force deterministic fallback
    )
    assert isinstance(lazy_train, LazyPairTable)
    assert isinstance(lazy_test, LazyPairTable)
    df_train = _full_flat(lazy_train)
    df_test = _full_flat(lazy_test)
    return df_train, df_vals, df_test, df_dead, lazy_train


def test_splits_are_disjoint_rows_and_whole_cities(env):
    df_train_all, df_vals, df_test, df_dead, _ = _run_split(env)
    # Holdout-year train pairs are NaN-masked (they belong to val_temporal);
    # only labeled rows ever train.
    df_train = df_train_all[df_train_all["Rel_Score"].notna()]

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
    df_train_all, df_vals, df_test, _, _ = _run_split(env)
    df_vt = df_vals["val_temporal"]
    df_train = df_train_all[df_train_all["Rel_Score"].notna()]

    for cbsa, grp in df_vt.groupby("cbsa_code"):
        held_years = set(grp["year"])
        assert len(held_years) == 1, f"CBSA {cbsa} holds out {held_years}"
        held = held_years.pop()
        # The holdout year never appears in that city's (labeled) train rows
        assert held not in set(df_train.loc[df_train["cbsa_code"] == cbsa, "year"])
        # ...and the masked train pairs of that city are exactly the holdout year
        masked = df_train_all[(df_train_all["cbsa_code"] == cbsa)
                              & df_train_all["Rel_Score"].isna()]
        assert set(masked["year"]) == {held}
    # Every train city held out exactly one year
    assert set(df_vt["cbsa_code"]) == set(df_train["cbsa_code"])


def test_val_frames_are_small_and_materialized(env):
    _, df_vals, _, _, _ = _run_split(env)
    for name, df_val in df_vals.items():
        assert isinstance(df_val, pd.DataFrame), name
        per_tract = df_val.drop_duplicates("building_id").groupby("GEOID").size()
        assert (per_tract <= 2).all(), name
    # val_cities keeps ALL years of each sampled building
    vc = df_vals["val_cities"]
    years_per_bldg = vc.groupby("building_id")["year"].nunique()
    assert (years_per_bldg == vc["year"].nunique()).all()


def test_cbsa_splits_artifact_consistent(env):
    from src.data import cbsa_brackets as cb
    df_train, df_vals, df_test, _, _ = _run_split(env)

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


def test_train_ordering_groups_twins(env):
    _, _, _, _, lazy_train = _run_split(env)
    flat = _full_flat(lazy_train)
    n_years = lazy_train.n_years
    bids = flat["building_id"].to_numpy()
    for k in range(0, len(flat), n_years):
        assert len(set(bids[k:k + n_years])) == 1


def test_small_sample_materializes_and_splits(env):
    table = bd.load_income_dataset([2014, 2016, 2022], indicator="W2_r5")
    df_train, df_vals, df_test, df_dead = bd.create_train_test_dataframes(
        table, "testsave_small", small_sample=True, indicator="W2_r5",
        naip_coverage_csv=env["tmp"] / "no_such_coverage.csv",
    )
    # small_sample falls back to the legacy row-level path on real DataFrames
    assert isinstance(df_train, pd.DataFrame) and isinstance(df_test, pd.DataFrame)
    assert len(df_train) > 0 and len(df_test) > 0


# ── NAIP-unavailable artifact (coverage-gap masking) ─────────────────────────

def test_load_naip_unavailable_absent_returns_none(env):
    assert bd.load_naip_unavailable() is None


def test_write_then_load_and_merge_naip_unavailable(env):
    p = bd.write_naip_unavailable(
        [{"level": "cbsa", "key": "46520", "year": 2016}], merge=False)
    assert p == env["tmp"] / bd.NAIP_UNAVAILABLE_FILENAME
    back = bd.load_naip_unavailable()
    assert list(back.columns) == ["level", "key", "year"]
    row = back.iloc[0]
    assert row["level"] == "cbsa" and row["key"] == "46520" and int(row["year"]) == 2016
    # merge unions without dup
    bd.write_naip_unavailable([{"level": "cbsa", "key": "46520", "year": 2016},
                               {"level": "tract", "key": "01001000100", "year": 2018}],
                              merge=True)
    merged = bd.load_naip_unavailable()
    assert len(merged) == 2


def test_dead_region_excluded_from_splits(env):
    # A CBSA flagged unavailable for ALL requested years must be dropped from the
    # split universe: absent from cbsa_splits / tract_splits and from every split,
    # with the remaining CBSAs still split into train/val/test.
    years = [2014, 2016, 2022]
    dead_cbsa = CBSAS[0][0]
    bd.write_naip_unavailable(
        [{"level": "cbsa", "key": dead_cbsa, "year": y} for y in years], merge=False)

    table = bd.load_income_dataset(years, indicator="W2_r5")
    df_train_lazy, df_vals, df_test_lazy, _ = bd.create_train_test_dataframes(
        table, "testsave_excl", small_sample=False, indicator="W2_r5",
        naip_coverage_csv=env["tmp"] / "no_such_coverage.csv",
    )

    meta = pd.read_feather(env["tmp"] / "cbsa_splits.feather")
    assert dead_cbsa not in set(meta["cbsa_code"].astype(str)), "dead CBSA still in split"
    assert set(meta["split"]) == {"train", "val", "test"}, "survivors still split 3 ways"

    tract_splits = gpd.read_feather(env["tmp"] / "tract_splits.feather")
    dead_geoids = set(env["panel"].loc[env["panel"]["cbsa_code"] == dead_cbsa, "geoid_2023"])
    assert not (set(tract_splits["GEOID"]) & dead_geoids), "dead tracts still in tract_splits"

    for frame in (df_train_lazy.materialize(0, len(df_train_lazy)),
                  df_test_lazy.materialize(0, len(df_test_lazy)),
                  df_vals["val_cities"], df_vals["val_temporal"]):
        assert dead_cbsa not in set(frame["cbsa_code"].astype(str))


def test_pipeline_auto_masks_unavailable_regions(env):
    # Flag one whole CBSA (all requested years) as unavailable, then confirm the
    # pipeline drops its labels so those pairs are never fetched.
    years = [2014, 2016, 2022]
    dead_cbsa = CBSAS[0][0]
    bd.write_naip_unavailable(
        [{"level": "cbsa", "key": dead_cbsa, "year": y} for y in years], merge=False)

    table = bd.load_income_dataset(years, indicator="W2_r5")
    assert table.unavailable is not None
    flat = table.materialize(0, len(table))
    dead = flat["cbsa_code"] == dead_cbsa
    assert dead.any()
    assert flat.loc[dead, "Rel_Score"].isna().all()          # masked → dropped pre-fetch
    assert flat.loc[~dead, "Rel_Score"].notna().all()         # everyone else untouched
