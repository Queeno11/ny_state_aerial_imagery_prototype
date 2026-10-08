"""Synthetic-data tests for src/csa_predict.py (per-city CSA prediction driver).

Builds a miniature version of the three artifacts the runner reads —
``pair_buildings_ms_us_epsg5070_all.parquet``, ``pair_labels_...parquet`` and
``tract_splits.feather`` — in a tmp_path, so nothing here touches the real 72M-row
tables, a model, a GPU or the network.

The properties worth pinning down:

1. **The sample is byte-identical across years and reruns.** ``csa.estimate``
   assumes a balanced panel; a sample that drifted between years would make a
   tract's outcome move for reasons unrelated to the imagery, which is
   indistinguishable from the treatment effect being measured.
2. **Shards are a disjoint cover.** Parallel processes must not double-predict
   or silently drop tracts.
3. **Every tract survives, only buildings are capped.** A tract missing from the
   panel is neither treated nor control — it vanishes from the event study.
4. **``city_frame`` emits exactly the columns ``predict_year_chunked`` requires**,
   since a missing column surfaces as a KeyError thousands of crops into a run.
5. **The chunk fingerprint separates sensors and sampling designs**, so a dense
   ortho pass can never share a resume cache with the sparse NAIP paper run.
"""

import numpy as np
import pandas as pd
import pytest

from src import csa_event_study as ces
from src import csa_predict as cp

YEARS = [2016, 2018, 2020]
PREFIX = "17031"
N_TRACTS = 12
N_BUILDINGS_PER_TRACT = 40


# ─── fixtures ─────────────────────────────────────────────────────────────────

def _geoid(i: int) -> str:
    return f"{PREFIX}{i:06d}"


@pytest.fixture
def processed(tmp_path):
    """A miniature stand-in for data/processed/."""
    geoids = [_geoid(i) for i in range(N_TRACTS)]

    rows = []
    bid = 10_000
    for g in geoids:
        for _ in range(N_BUILDINGS_PER_TRACT):
            bid += 7
            rows.append({"building_id": bid, "GEOID": g, "cbsa_code": "16980",
                         "centroid_x": 700_000.0 + bid % 997,
                         "centroid_y": 2_100_000.0 + bid % 883,
                         "dist_to_center": float(bid % 50)})
    # GEOID is DICTIONARY-ENCODED in the real national parquet. That matters:
    # pyarrow's `starts_with` has no kernel for dictionary<string> and raises
    # ArrowNotImplementedError instead of falling back, so a plain-string
    # fixture would pass while the real read failed.
    import pyarrow as pa
    import pyarrow.parquet as pq

    frame = pd.DataFrame(rows)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    idx = table.schema.get_field_index("GEOID")
    table = table.set_column(
        idx, pa.field("GEOID", pa.dictionary(pa.int32(), pa.string())),
        table.column("GEOID").dictionary_encode())
    pq.write_table(table, tmp_path / cp.BUILDINGS_PARQUET)

    labels = pd.DataFrame([
        {"GEOID": g, "year": y, "Rel_Score": 0.1 * i}
        for i, g in enumerate(geoids) for y in YEARS
    ])
    labels.to_parquet(tmp_path / cp.LABELS_PARQUET, index=False)

    pd.DataFrame({"GEOID": geoids,
                  "type": ["test"] * (N_TRACTS - 2) + ["train", "val"],
                  }).to_feather(tmp_path / cp.TRACT_SPLITS)
    return tmp_path


@pytest.fixture
def spec():
    return ces.CityCohortSpec(
        key="chicago", label="Chicago", geoid_prefixes=(PREFIX,),
        footprints_filename="buildings_chicago.parquet",
        year_col="year_built", demolition_col=None, id_index="building_id",
        panel_years=tuple(YEARS), baseline_year=2015, sensor="ortho",
        state="Illinois", tract_source="prefix",
    )


def _runner(processed, tmp_path, **kw):
    kw.setdefault("buildings_per_tract", 10)
    return cp.CSAPredictionRunner(
        "run_test", {"image_size": 224, "nbands": 4, "tau_meters": 100},
        processed_dir=processed, results_dir=tmp_path / "results",
        cache_root=tmp_path / "cache", **kw)


# ─── sampling determinism ─────────────────────────────────────────────────────

def test_sample_is_identical_across_reruns(processed, tmp_path, spec):
    a = _runner(processed, tmp_path).sample_buildings(spec)
    b = _runner(processed, tmp_path).sample_buildings(spec)
    pd.testing.assert_frame_equal(a, b)


def test_sample_is_identical_across_years(processed, tmp_path, spec):
    """REGRESSION GUARD: csa.estimate(balanced=True) assumes a balanced panel.
    If the sampled buildings differed between years, the tract mean would move
    year to year for reasons unrelated to the imagery — a confound that looks
    exactly like the effect being estimated."""
    runner = _runner(processed, tmp_path)
    frames = {y: runner.city_frame(spec, y) for y in YEARS}
    ids = [set(f["building_id"]) for f in frames.values()]
    assert ids[0] == ids[1] == ids[2]
    assert len(ids[0]) > 0


def test_sampling_respects_the_per_tract_cap(processed, tmp_path, spec):
    df = _runner(processed, tmp_path, buildings_per_tract=10).sample_buildings(spec)
    assert df.groupby("GEOID").size().max() == 10


def test_every_tract_survives_sampling(processed, tmp_path, spec):
    """Only buildings are capped, never tracts: a tract absent from the panel is
    neither treated nor control, it simply disappears from the event study."""
    df = _runner(processed, tmp_path, buildings_per_tract=3).sample_buildings(spec)
    assert df["GEOID"].nunique() == N_TRACTS


def test_no_cap_keeps_every_building(processed, tmp_path, spec):
    df = _runner(processed, tmp_path, buildings_per_tract=None).sample_buildings(spec)
    assert len(df) == N_TRACTS * N_BUILDINGS_PER_TRACT


def test_cap_above_supply_is_harmless(processed, tmp_path, spec):
    df = _runner(processed, tmp_path, buildings_per_tract=10_000).sample_buildings(spec)
    assert len(df) == N_TRACTS * N_BUILDINGS_PER_TRACT


# ─── sharding ─────────────────────────────────────────────────────────────────

def test_shards_are_a_disjoint_cover(processed, tmp_path, spec):
    """Parallel processes must not double-predict or drop tracts."""
    whole = set(_runner(processed, tmp_path).sample_buildings(spec)["building_id"])
    parts = [set(_runner(processed, tmp_path, shard=(i, 4))
                 .sample_buildings(spec)["building_id"]) for i in range(4)]
    union = set().union(*parts)
    assert union == whole
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (parts[i] & parts[j]), f"shards {i},{j} overlap"


def test_shards_split_by_tract_not_by_building(processed, tmp_path, spec):
    """A tract must live entirely in one shard: its outcome is a mean over its
    sampled buildings, so splitting one tract across processes would have each
    write a partial mean for the same tract-year."""
    for i in range(3):
        df = _runner(processed, tmp_path, shard=(i, 3)).sample_buildings(spec)
        if df.empty:
            continue
        other = _runner(processed, tmp_path, shard=((i + 1) % 3, 3)).sample_buildings(spec)
        assert not (set(df["GEOID"]) & set(other["GEOID"]))


@pytest.mark.parametrize("bad", [(0, 0), (-1, 2), (2, 2), (5, 3)])
def test_invalid_shard_is_rejected(processed, tmp_path, bad):
    with pytest.raises(ValueError, match="shard"):
        _runner(processed, tmp_path, shard=bad)


# ─── city_frame contract ──────────────────────────────────────────────────────

REQUIRED = ["Rel_Score", "GEOID", "building_id", "year", "type",
            "centroid_x", "centroid_y"]


def test_city_frame_has_every_column_predict_year_chunked_needs(processed, tmp_path, spec):
    df = _runner(processed, tmp_path).city_frame(spec, 2018)
    missing = [c for c in REQUIRED if c not in df.columns]
    assert not missing, missing
    assert (df["year"] == 2018).all()


def test_city_frame_drops_unlabeled_rows_before_fetching(processed, tmp_path, spec):
    """An unlabeled tract-year cannot enter the event study, and a crop costs a
    network round trip — so drop before the fetch, not after."""
    labels = pd.read_parquet(processed / cp.LABELS_PARQUET)
    labels.loc[labels["GEOID"] == _geoid(0), "Rel_Score"] = np.nan
    labels.to_parquet(processed / cp.LABELS_PARQUET, index=False)

    df = _runner(processed, tmp_path).city_frame(spec, 2018)
    assert _geoid(0) not in set(df["GEOID"])
    assert df["Rel_Score"].notna().all()


def test_city_frame_carries_the_split_type(processed, tmp_path, spec):
    df = _runner(processed, tmp_path).city_frame(spec, 2018)
    assert set(df["type"]) <= {"test", "train", "val"}


def test_city_frame_is_empty_when_no_tract_matches(processed, tmp_path):
    other = ces.CityCohortSpec(key="x", label="X", geoid_prefixes=("99999",),
                               footprints_filename="x.parquet",
                               panel_years=tuple(YEARS))
    assert _runner(processed, tmp_path).city_frame(other, 2018).empty


# ─── per-city params / fingerprint isolation ──────────────────────────────────

def test_ortho_city_pins_the_sensor_and_city(processed, tmp_path, spec):
    p = _runner(processed, tmp_path)._city_params(spec)
    assert p["predict_sensor"] == "ortho"
    assert p["predict_ortho_city"] == "chicago"


def test_naip_city_does_not_set_an_ortho_city(processed, tmp_path, spec):
    naip = ces.CSA_CITIES["nashville"]
    p = _runner(processed, tmp_path)._city_params(naip)
    assert p["predict_sensor"] == "naip"
    assert "predict_ortho_city" not in p


def test_nyc_zarr_spec_maps_to_naip_for_prediction_params(processed, tmp_path):
    """"zarr" describes where NYC's existing predictions came from, not a fetcher
    this runner can drive; it must not leak into predict_sensor."""
    p = _runner(processed, tmp_path)._city_params(ces.CSA_CITIES["nyc"])
    assert p["predict_sensor"] == "naip"


def test_sampling_knobs_are_cleared_so_caches_cannot_mix(processed, tmp_path, spec):
    """REGRESSION GUARD: these knobs are in prediction_fingerprint. Leaving the
    paper run's 10%/100 values in place would let this dense pass resume from —
    and append to — a sparse run's chunk cache."""
    runner = _runner(processed, tmp_path, buildings_per_tract=200)
    runner.params.update({"predict_tract_sample_frac": 0.10,
                          "predict_tract_sample_min": 100,
                          "predict_buildings_per_tract": 100})
    p = runner._city_params(spec)
    assert p["predict_tract_sample_frac"] is None
    assert p["predict_tract_sample_min"] is None
    assert p["predict_buildings_per_tract"] == 200


def test_fingerprint_separates_ortho_from_naip(processed, tmp_path, spec):
    from src import prediction

    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"x")
    runner = _runner(processed, tmp_path)
    naip_spec = ces.CSA_CITIES["nashville"]
    fp_ortho = prediction.prediction_fingerprint(runner._city_params(spec), model_path)
    fp_naip = prediction.prediction_fingerprint(runner._city_params(naip_spec), model_path)
    assert fp_ortho["predict_sensor"] != fp_naip["predict_sensor"]


def test_legacy_cache_without_predict_sensor_still_resumes(tmp_path):
    """REGRESSION GUARD: adding predict_sensor to the fingerprint must not
    invalidate chunk caches written before it existed. Those runs were all NAIP,
    so the absence of the key is unambiguous — and a spurious mismatch would
    make the user delete completed chunks (93 MB of finished work in the real
    run_20260722/test cache) to recover."""
    import json

    from src import prediction

    model_path = tmp_path / "m.pth"
    model_path.write_bytes(b"x")
    params = {"image_size": 224, "nbands": 4, "predict_sensor": "naip"}
    fp = prediction.prediction_fingerprint(params, model_path)

    root = tmp_path / "chunks"
    root.mkdir()
    legacy = {k: v for k, v in fp.items() if k != "predict_sensor"}
    (root / "manifest.json").write_text(json.dumps(legacy))

    prediction.init_chunk_root(root, fp)          # must not raise
    # ... and the manifest is upgraded so later runs compare exactly.
    assert json.loads((root / "manifest.json").read_text())["predict_sensor"] == "naip"


def test_a_real_sensor_change_still_aborts(tmp_path):
    """The back-fill must not weaken the guard it was added for: naip and ortho
    chunks for the same building-year hold different predictions."""
    import json

    from src import prediction

    model_path = tmp_path / "m.pth"
    model_path.write_bytes(b"x")
    root = tmp_path / "chunks"
    root.mkdir()
    naip_fp = prediction.prediction_fingerprint(
        {"image_size": 224, "nbands": 4, "predict_sensor": "naip"}, model_path)
    (root / "manifest.json").write_text(json.dumps(naip_fp))

    ortho_fp = prediction.prediction_fingerprint(
        {"image_size": 224, "nbands": 4, "predict_sensor": "ortho"}, model_path)
    with pytest.raises(RuntimeError, match="predict_sensor"):
        prediction.init_chunk_root(root, ortho_fp)


def test_output_paths_are_namespaced_per_city(processed, tmp_path, spec):
    runner = _runner(processed, tmp_path)
    assert runner.output_csv(spec, 2018).parts[-3:] == ("csa", "chicago",
                                                        "2018_predictions.csv")
    assert "chicago" in str(runner.chunk_root(spec))


# ─── reporting ────────────────────────────────────────────────────────────────

def test_plan_reports_the_crop_budget_without_fetching(processed, tmp_path, spec,
                                                       monkeypatch):
    monkeypatch.setattr(ces, "CSA_CITIES", {"chicago": spec})
    monkeypatch.setattr(cp.ces, "CSA_CITIES", {"chicago": spec})
    (processed / spec.footprints_filename).write_bytes(b"")
    runner = _runner(processed, tmp_path, buildings_per_tract=10)
    plan = runner.plan()
    assert plan.loc[0, "n_crops"] == plan.loc[0, "n_buildings"] * len(YEARS)
    assert plan.loc[0, "n_years"] == len(YEARS)
    assert plan.loc[0, "sensor"] == "ortho"


def test_status_marks_years_complete_when_the_csv_exists(processed, tmp_path, spec,
                                                         monkeypatch):
    monkeypatch.setattr(cp.ces, "CSA_CITIES", {"chicago": spec})
    (processed / spec.footprints_filename).write_bytes(b"")
    runner = _runner(processed, tmp_path)
    out = runner.output_csv(spec, 2018)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("building_id\n1\n2\n")

    status = runner.status().set_index("year")
    assert bool(status.loc[2018, "done"]) is True
    assert int(status.loc[2018, "n_rows"]) == 2
    assert bool(status.loc[2016, "done"]) is False


def test_unknown_city_key_is_rejected(processed, tmp_path):
    with pytest.raises(KeyError, match="unknown CSA cities"):
        _runner(processed, tmp_path, cities=["atlantis"])


def test_zarr_cities_are_not_re_predicted_by_default(processed, tmp_path,
                                                     monkeypatch):
    """REGRESSION GUARD: NYC's predictions come from the whole-city zarr pass.
    Re-predicting it here is not merely wasteful (2.07M crops) — this runner
    samples 200 buildings/tract from NAIP, and part_d prefers csa/<city>/ over
    the zarr directory, so the sampled wrong-sensor output would silently
    DISPLACE the full-universe NYC event study."""
    zarr_spec = ces.CityCohortSpec(
        key="nyc", label="New York City", geoid_prefixes=(PREFIX,),
        footprints_filename="buildings_nyc.parquet",
        panel_years=tuple(YEARS), sensor="zarr")
    naip_spec = ces.CityCohortSpec(
        key="tampa", label="Tampa", geoid_prefixes=(PREFIX,),
        footprints_filename="buildings_tampa.parquet",
        panel_years=tuple(YEARS), sensor="naip")
    monkeypatch.setattr(cp.ces, "CSA_CITIES",
                        {"nyc": zarr_spec, "tampa": naip_spec})
    (processed / "buildings_nyc.parquet").write_bytes(b"")
    (processed / "buildings_tampa.parquet").write_bytes(b"")

    runner = _runner(processed, tmp_path)
    assert [s.key for s in runner.specs] == ["tampa"]
    assert [s.key for s in runner.skipped_zarr] == ["nyc"]


def test_naming_a_zarr_city_explicitly_still_runs_it(processed, tmp_path,
                                                     monkeypatch):
    """The skip is a default, not a prohibition — asking for it by name works."""
    zarr_spec = ces.CityCohortSpec(
        key="nyc", label="New York City", geoid_prefixes=(PREFIX,),
        footprints_filename="buildings_nyc.parquet",
        panel_years=tuple(YEARS), sensor="zarr")
    monkeypatch.setattr(cp.ces, "CSA_CITIES", {"nyc": zarr_spec})
    (processed / "buildings_nyc.parquet").write_bytes(b"")

    runner = _runner(processed, tmp_path, cities=["nyc"])
    assert [s.key for s in runner.specs] == ["nyc"]
    assert runner.skipped_zarr == []


# ─── baltimore wiring ─────────────────────────────────────────────────────────

def test_baltimore_is_discovered_once_its_footprints_land(processed, tmp_path):
    """The registry is the only wiring: dropping buildings_md_mdp.parquet into
    processed/ must be enough to make the 4th city run, with no code change in
    the runner. Guards the claim that csa_predict is city-generic."""
    (processed / "buildings_md_mdp.parquet").write_bytes(b"")
    runner = _runner(processed, tmp_path, cities=["baltimore"])
    assert [s.key for s in runner.specs] == ["baltimore"]
    assert runner.skipped_zarr == []


def test_baltimore_params_pin_naip_and_the_whole_cbsa(processed, tmp_path):
    """A NAIP city must not acquire an ortho city key (that would route fetches
    at a municipal image server it has no source for), and the full-universe
    prefixes must carry all seven jurisdictions through to prediction."""
    runner = _runner(processed, tmp_path)
    spec = ces.CSA_CITIES["baltimore"]
    p = runner._city_params(spec)

    assert p["predict_sensor"] == "naip"
    assert "predict_ortho_city" not in p
    assert p["predict_split"] == "csa_baltimore"
    assert set(p["predict_full_universe_geoid_prefixes"]) == {
        "24003", "24005", "24013", "24025", "24027", "24035", "24510"}
    # The dense pass must never share a chunk cache with the sparse paper run.
    assert p["predict_tract_sample_frac"] is None
    assert p["predict_tract_sample_min"] is None


def test_san_antonio_is_discovered_once_its_footprints_land(processed, tmp_path):
    """Registry-only wiring, same as Baltimore: dropping the parquet in is enough."""
    (processed / "buildings_bexar.parquet").write_bytes(b"")
    runner = _runner(processed, tmp_path, cities=["san_antonio"])
    assert [s.key for s in runner.specs] == ["san_antonio"]
    assert runner.skipped_zarr == []


def test_san_antonio_params_pin_naip_and_bexar_county(processed, tmp_path):
    runner = _runner(processed, tmp_path)
    spec = ces.CSA_CITIES["san_antonio"]
    p = runner._city_params(spec)

    assert p["predict_sensor"] == "naip"
    assert "predict_ortho_city" not in p
    assert p["predict_split"] == "csa_san_antonio"
    assert tuple(p["predict_full_universe_geoid_prefixes"]) == ("48029",)
    assert p["predict_tract_sample_frac"] is None


def test_san_antonio_panel_years_reach_the_runner(processed, tmp_path):
    """Even-year parity must survive into run_city; Texas has no odd-year NAIP."""
    spec = ces.CSA_CITIES["san_antonio"]
    assert spec.years() == [2012, 2014, 2016, 2018]
    assert spec.baseline() == 2011


def test_baltimore_panel_years_reach_the_runner(processed, tmp_path):
    """spec.years() is what run_city iterates; an empty or off-grid panel here
    would fetch nothing and produce a silently empty event study.

    2018 is deliberately absent even though Maryland NAIP has it: it sits one
    year after 2017 while every other gap is two, which makes the 2018 cohort a
    one-year window and leaves it as the sole identifier of event time k = -3
    (a single ATT(g,t) cell at weight 1.0, from two tracts at the 10% cut). See
    the CSA_CITIES entry and issue #36.
    """
    spec = ces.CSA_CITIES["baltimore"]
    assert spec.years() == [2011, 2013, 2015, 2017]
    assert spec.baseline() == 2010
    assert 2018 not in spec.years()


def test_no_city_panel_ends_in_a_short_stub_period():
    """The final gap may not be shorter than the typical gap.

    Uneven gaps as such are fine and unavoidable — Florida NAIP gives Tampa a
    three-year first gap and two-year gaps after it. What is not fine is a
    final gap SHORTER than the rest, which is what Baltimore's 2018 was (one
    year after 2017, against two everywhere else). A short trailing period
    defines its cohort over less calendar time than every other cohort, so that
    cohort is mechanically small — and because csa's varying base period needs
    t >= 2, the last cohort is the only one observable at the most negative
    event time. The result is a plotted coefficient and a pre-trend test
    resting on a single ATT(g,t) cell: at Baltimore's 10% cut, two tracts at
    weight pge = 1.0. See the CSA_CITIES entry for baltimore and issue #36.
    """
    offenders = {}
    for key, spec in ces.CSA_CITIES.items():
        if not spec.panel_years:
            continue
        years = spec.years()
        if len(years) < 3:
            continue
        gaps = [b - a for a, b in zip(years, years[1:])]
        typical = sorted(gaps[:-1])[len(gaps[:-1]) // 2]
        if gaps[-1] < typical:
            offenders[key] = {"years": years, "gaps": gaps, "typical": typical}
    assert not offenders, f"panel ends in a short stub period: {offenders}"
