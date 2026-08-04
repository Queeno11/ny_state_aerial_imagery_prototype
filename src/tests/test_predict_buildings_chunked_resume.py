"""Synthetic-data tests for predict_buildings_chunked's resume behavior.

The zarr-only NYC validation pass (run_nyc_zarr_validation_predictions ->
predict_buildings_chunked) used to write a single .tmp CSV and only promote it
to output_path on a clean finish — a crash partway through a ~4h year lost
all progress. predict_buildings_chunked now writes one parquet per spatial
groupby-chunk under f"{output_path}_chunks/" as each group finishes, skips
groups whose chunk file already exists on a re-run, and only assembles the
final CSV once every group is present.
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch

import src.main as main


class _FakeSlice:
    def __init__(self, arr):
        self._arr = arr

    def to_numpy(self):
        return self._arr


class _FakeZarrArray:
    """Mimics the xarray/zarr slice -> .to_numpy() interface main.py expects."""

    def __init__(self, arr):
        self._arr = arr

    def __getitem__(self, key):
        return _FakeSlice(self._arr[key])


class _DummyModel(torch.nn.Module):
    """Deterministic stand-in: predicted value = mean pixel value of the crop."""

    def eval(self):
        return self

    def forward(self, x, metadata=None):
        return x.mean(dim=(1, 2, 3))


def _identity_transform(x):
    return x.float()


def _build_df(n_groups=3, rows_per_group=4, image_size=64):
    """Buildings spread across `n_groups` disjoint 2500px groupby-chunks."""
    rows = []
    building_id = 0
    for g in range(n_groups):
        # groupby_chunk_size is hardcoded to 2500 in predict_buildings_chunked;
        # offsetting row_start by g * 2500 lands each group's buildings in a
        # distinct groupby_chunk_id.
        base_row = g * 2500
        for i in range(rows_per_group):
            rows.append({
                "row_start": base_row,
                "row_stop": base_row + image_size,
                "col_start": 0,
                "col_stop": image_size,
                "dataset": "ds1",
                "Rel_Score": float(building_id),
                "building_id": building_id,
                "GEOID": f"36{building_id:03d}",
                "year": 2020,
                "type": "train",
                "dist_to_center": 0.0,
            })
            building_id += 1
    return pd.DataFrame(rows)


def _fake_datasets(nbands=4, height=8000, width=64):
    arr = np.random.randint(0, 255, size=(nbands, height, width), dtype=np.uint8)
    return {"ds1": {"value": _FakeZarrArray(arr)}}


def _params(nbands=4):
    return {"batch_size": 2, "nbands": nbands, "subsample_step": 1}


def test_writes_one_chunk_parquet_per_group_and_assembles_full_csv(tmp_path):
    df = _build_df(n_groups=3, rows_per_group=4)
    datasets = _fake_datasets()
    output_path = tmp_path / "2020_predictions.csv"

    main.predict_buildings_chunked(
        model=_DummyModel(), df=df, all_years_datasets=datasets, params=_params(),
        device=torch.device("cpu"), output_path=str(output_path),
        eval_transform=_identity_transform, verbose=False,
    )

    assert output_path.exists()
    result = pd.read_csv(output_path)
    assert len(result) == len(df)
    assert set(result["building_id"]) == set(df["building_id"])

    chunk_dir = tmp_path / "2020_predictions_chunks"
    chunk_files = sorted(p.name for p in chunk_dir.glob("chunk_*.parquet"))
    assert len(chunk_files) == 3   # one per groupby chunk
    assert (chunk_dir / "_meta.json").exists()


def test_resume_skips_already_done_groups(tmp_path, monkeypatch):
    df = _build_df(n_groups=3, rows_per_group=4)
    datasets = _fake_datasets()
    output_path = tmp_path / "2020_predictions.csv"
    chunk_dir = tmp_path / "2020_predictions_chunks"

    # Simulate a crash partway through: pre-seed one group's chunk file (as if
    # a prior run had already produced it) but no final CSV.
    chunk_dir.mkdir(parents=True)
    seeded_group_rows = df[df["row_start"] == 2500]
    seeded = pd.DataFrame({
        "Rel_Score": seeded_group_rows["Rel_Score"].to_numpy(),
        "predicted_value": np.full(len(seeded_group_rows), -999.0),
        "building_id": seeded_group_rows["building_id"].to_numpy(),
        "GEOID": seeded_group_rows["GEOID"].astype(str).to_numpy(),
        "year": seeded_group_rows["year"].to_numpy(),
        "type": seeded_group_rows["type"].astype(str).to_numpy(),
    })
    seeded.to_parquet(chunk_dir / "chunk_000001_000000.parquet")
    meta = {"n_rows": len(df), "image_size": 64, "groupby_chunk_size": 2500}
    (chunk_dir / "_meta.json").write_text(json.dumps(meta))

    extracted_rows = []
    real_extract = main.extract_image_from_chunks

    def _tracking_extract(row, *args, **kwargs):
        extracted_rows.append(row.get("building_id"))
        return real_extract(row, *args, **kwargs)

    monkeypatch.setattr(main, "extract_image_from_chunks", _tracking_extract)

    main.predict_buildings_chunked(
        model=_DummyModel(), df=df, all_years_datasets=datasets, params=_params(),
        device=torch.device("cpu"), output_path=str(output_path),
        eval_transform=_identity_transform, verbose=False,
    )

    # The pre-seeded group's buildings must never have been re-extracted.
    seeded_ids = set(seeded_group_rows["building_id"])
    assert not (seeded_ids & set(extracted_rows))

    result = pd.read_csv(output_path).set_index("building_id")
    # Seeded group keeps its sentinel value; the other two groups were
    # genuinely computed this run.
    for bid in seeded_ids:
        assert result.loc[bid, "predicted_value"] == -999.0
    other_ids = set(df["building_id"]) - seeded_ids
    for bid in other_ids:
        assert result.loc[bid, "predicted_value"] != -999.0
    assert len(result) == len(df)


def test_meta_mismatch_on_resume_raises(tmp_path):
    df = _build_df(n_groups=2, rows_per_group=2)
    datasets = _fake_datasets()
    output_path = tmp_path / "2020_predictions.csv"
    chunk_dir = tmp_path / "2020_predictions_chunks"
    chunk_dir.mkdir(parents=True)
    # Stale meta from a differently-sized run.
    (chunk_dir / "_meta.json").write_text(json.dumps(
        {"n_rows": 999, "image_size": 64, "groupby_chunk_size": 2500}))

    with pytest.raises(RuntimeError, match="Delete"):
        main.predict_buildings_chunked(
            model=_DummyModel(), df=df, all_years_datasets=datasets, params=_params(),
            device=torch.device("cpu"), output_path=str(output_path),
            eval_transform=_identity_transform, verbose=False,
        )


def test_empty_df_produces_empty_csv(tmp_path):
    # An empty df has no row_stop/row_start to infer image_size from via
    # .min(), so use an all-NaN Rel_Score (nothing predictable) instead of an
    # actually-empty frame.
    df = _build_df(n_groups=1, rows_per_group=2)
    df["Rel_Score"] = np.nan
    datasets = _fake_datasets()
    output_path = tmp_path / "2020_predictions.csv"

    main.predict_buildings_chunked(
        model=_DummyModel(), df=df, all_years_datasets=datasets, params=_params(),
        device=torch.device("cpu"), output_path=str(output_path),
        eval_transform=_identity_transform, verbose=False,
    )

    assert output_path.exists()
    result = pd.read_csv(output_path)
    assert len(result) == 0


def test_rerun_after_full_completion_is_a_noop_reassembly(tmp_path):
    df = _build_df(n_groups=2, rows_per_group=3)
    datasets = _fake_datasets()
    output_path = tmp_path / "2020_predictions.csv"

    main.predict_buildings_chunked(
        model=_DummyModel(), df=df, all_years_datasets=datasets, params=_params(),
        device=torch.device("cpu"), output_path=str(output_path),
        eval_transform=_identity_transform, verbose=False,
    )
    first = pd.read_csv(output_path).sort_values("building_id").reset_index(drop=True)

    # Re-run with the identical df: every group's chunk file already exists,
    # so this should just reassemble the same CSV without re-predicting.
    main.predict_buildings_chunked(
        model=_DummyModel(), df=df, all_years_datasets=datasets, params=_params(),
        device=torch.device("cpu"), output_path=str(output_path),
        eval_transform=_identity_transform, verbose=False,
    )
    second = pd.read_csv(output_path).sort_values("building_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(first, second)
