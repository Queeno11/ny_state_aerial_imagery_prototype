"""Synthetic-data tests for src/data/run_data_pipeline.py's raw-download checks.

Everything runs against tmp_path fixtures monkeypatched onto the module's path
constants -- no real ACS/buildings data, no network.
"""

import zipfile

import pandas as pd
import pytest

from src.data import run_data_pipeline as pipe

YEARS = [2022, 2023]


def _write_acs_feather(path, columns, n_rows=5):
    """A real (small) Arrow feather -- big enough to clear the shrunk test threshold."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({c: range(n_rows) for c in columns})
    df.to_feather(path)


def _write_junk_file(path, size_bytes):
    """NOT a valid Arrow file -- simulates a truncated/interrupted download."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0" * size_bytes)


def _write_zip(path, size_bytes=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data.geojson", "0" * max(size_bytes, 1))


@pytest.fixture
def acs_root(tmp_path, monkeypatch):
    root = tmp_path / "acs"
    monkeypatch.setattr(pipe, "ACS_ROOT_DIR", root)
    # Real files are MB-scale; shrink so a small synthetic fixture (~6 KB) still clears it.
    monkeypatch.setattr(pipe, "MIN_ACS_FEATHER_BYTES", 3_000)
    return root


# ── check_acs_years ──────────────────────────────────────────────────────────

def test_check_acs_years_all_present_and_valid(acs_root):
    for year in YEARS:
        _write_acs_feather(
            acs_root / str(year) / f"us_tracts_acs5_{year}.feather",
            sorted(pipe.REQUIRED_ACS_COLUMNS),
        )
    c = pipe.Checker()
    pipe.check_acs_years(YEARS, c)
    assert not c.failures


def test_check_acs_years_missing_file(acs_root):
    _write_acs_feather(
        acs_root / "2022" / "us_tracts_acs5_2022.feather",
        sorted(pipe.REQUIRED_ACS_COLUMNS),
    )
    # 2023 never written.
    c = pipe.Checker()
    pipe.check_acs_years(YEARS, c)
    assert any("2023" in f for f in c.failures)
    assert not any("2022" in f for f in c.failures)


def test_check_acs_years_too_small(acs_root):
    for year in YEARS:
        # Truncated/interrupted download: file exists but far below the size floor.
        _write_junk_file(acs_root / str(year) / f"us_tracts_acs5_{year}.feather", 500)
    c = pipe.Checker()
    pipe.check_acs_years(YEARS, c)
    assert len(c.failures) == len(YEARS)


def test_check_acs_years_missing_columns(acs_root):
    for year in YEARS:
        cols = sorted(pipe.REQUIRED_ACS_COLUMNS - {pipe.process_acs.PCI_COL})
        _write_acs_feather(
            acs_root / str(year) / f"us_tracts_acs5_{year}.feather", cols,
        )
    c = pipe.Checker()
    pipe.check_acs_years(YEARS, c)
    assert len(c.failures) == len(YEARS)


# ── check_buildings_states ───────────────────────────────────────────────────

def test_check_buildings_states_partial(tmp_path):
    footprints_dir = tmp_path / "footprints"
    _write_zip(footprints_dir / "Delaware.geojson.zip", size_bytes=20_000)
    # NewYork zip intentionally absent.
    present, missing = pipe.check_buildings_states(["Delaware", "NewYork"], footprints_dir)
    assert present == ["Delaware"]
    assert missing == ["NewYork"]


def test_check_buildings_states_too_small(tmp_path, monkeypatch):
    monkeypatch.setattr(pipe, "MIN_FOOTPRINTS_ZIP_BYTES", 50_000)
    footprints_dir = tmp_path / "footprints"
    _write_zip(footprints_dir / "Delaware.geojson.zip", size_bytes=100)  # below threshold
    present, missing = pipe.check_buildings_states(["Delaware"], footprints_dir)
    assert present == []
    assert missing == ["Delaware"]


# ── check_crosswalk / check_fhfa / check_vre (presence-only, non-fatal ones just warn) ──

def test_check_crosswalk_present_and_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(pipe.process_acs, "CBSA_CROSSWALK_CACHE", tmp_path / "missing.csv")
    monkeypatch.setattr(pipe.process_acs, "CBSA_CROSSWALK_XLSX", tmp_path / "missing.xlsx")
    c = pipe.Checker()
    pipe.check_crosswalk(c)
    assert c.failures

    (tmp_path / "present.csv").write_text("a,b\n1,2\n")
    monkeypatch.setattr(pipe.process_acs, "CBSA_CROSSWALK_CACHE", tmp_path / "present.csv")
    c2 = pipe.Checker()
    pipe.check_crosswalk(c2)
    assert not c2.failures


def test_check_fhfa_absent_does_not_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(pipe.process_acs, "FHFA_COUNTY_FILE", tmp_path / "missing.xlsx")
    monkeypatch.setattr(pipe.process_acs, "FHFA_CBSA_FILE", tmp_path / "missing.xlsx")
    c = pipe.Checker()
    pipe.check_fhfa(c)
    assert not c.failures  # warn-only: must never land in c.failures


def test_check_vre_absent_does_not_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(pipe, "VRE_ROOT", tmp_path / "vre")
    c = pipe.Checker()
    pipe.check_vre(YEARS, c)
    assert not c.failures  # warn-only


# ── clean_stale_caches (pre-#28 val shard caches) ────────────────────────────

def test_clean_stale_caches_removes_only_old_split_dirs(tmp_path):
    stale = ["val_spatial_cache", "val_spatial_temporal_cache", "val_2016_cache"]
    keep = ["train_cache", "val_cities_cache", "val_temporal_cache"]
    for name in stale + keep:
        (tmp_path / name).mkdir()
        (tmp_path / name / "shard_0.pt").write_bytes(b"x")

    removed = pipe.clean_stale_caches(cache_dir=tmp_path)

    assert sorted(removed) == sorted(stale)
    for name in stale:
        assert not (tmp_path / name).exists()
    for name in keep:
        assert (tmp_path / name).exists()


def test_clean_stale_caches_missing_dir_is_noop(tmp_path):
    assert pipe.clean_stale_caches(cache_dir=tmp_path / "nope") == []


# ── run_naip_coverage_stage ──────────────────────────────────────────────────

def _naip_args(tmp_path, states=None, force=False):
    import argparse
    return argparse.Namespace(
        states=states, force=force, naip_years=[2010, 2024],
        naip_coverage_out=str(tmp_path / "naip_coverage.csv"),
        footprints_dir=str(tmp_path / "footprints"),
    )


def test_naip_stage_skips_when_csv_covers_states(tmp_path, monkeypatch):
    from src.data import naip_coverage_audit
    pd.DataFrame({"state": ["Delaware"], "year": [2018], "n_items": [4]}).to_csv(
        tmp_path / "naip_coverage.csv", index=False)

    def boom(argv):  # the audit must NOT run
        raise AssertionError("audit should have been skipped")

    monkeypatch.setattr(naip_coverage_audit, "main", boom)
    args = _naip_args(tmp_path, states=["Delaware"])
    assert pipe.run_naip_coverage_stage(args, tmp_path / "footprints") is True


def test_naip_stage_audits_missing_states(tmp_path, monkeypatch):
    from src.data import naip_coverage_audit
    calls = {}

    def fake_main(argv):
        calls["argv"] = argv
        return 0

    monkeypatch.setattr(naip_coverage_audit, "main", fake_main)
    args = _naip_args(tmp_path, states=["Delaware", "NewYork"])
    assert pipe.run_naip_coverage_stage(args, tmp_path / "footprints") is True
    assert "Delaware" in calls["argv"] and "NewYork" in calls["argv"]
    assert str(tmp_path / "naip_coverage.csv") in calls["argv"]


def test_naip_stage_network_failure_is_nonfatal(tmp_path, monkeypatch):
    from src.data import naip_coverage_audit

    def fake_main(argv):
        raise ConnectionError("no network")

    monkeypatch.setattr(naip_coverage_audit, "main", fake_main)
    args = _naip_args(tmp_path, states=["Delaware"])
    assert pipe.run_naip_coverage_stage(args, tmp_path / "footprints") is True


def test_naip_stage_unknown_states_skip(tmp_path):
    args = _naip_args(tmp_path, states=["Atlantis"])
    assert pipe.run_naip_coverage_stage(args, tmp_path / "footprints") is True
