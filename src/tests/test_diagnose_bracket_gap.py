"""Synthetic-data tests for src/diagnose_bracket_gap.py."""

import numpy as np
import pandas as pd
import pytest

import src.diagnose_bracket_gap as dbg
from src.diagnose_city_coding import PRED_COL, LABEL_COL, CITY_COL


def _city_block(rng, cbsa, n, rho_strength=2.0, offset=0.0, noise_on_label=0.0,
                val_set="val_cities", year=2020):
    """One city's rows: pred = strength*z + offset + eps; higher strength -> higher rho."""
    z = rng.standard_normal(n)
    z = (z - z.mean()) / z.std()
    label = z + noise_on_label * rng.standard_normal(n)  # noisy observed label
    pred = rho_strength * z + offset + rng.standard_normal(n)
    return pd.DataFrame({
        PRED_COL: pred, LABEL_COL: label, CITY_COL: cbsa, "year": year, "set": val_set,
    })


# ── H1: pooling artifact ─────────────────────────────────────────────────────

def test_pooling_artifact_reproduces_observed_pattern():
    """Uniform within-city skill + offsets in multi-city brackets must yield
    poor POOLED rho for the multi-city brackets but flat rho_within."""
    rng = np.random.default_rng(825)
    frames = [_city_block(rng, 100, 400)]                      # "mega": 1 city, no offset issue
    offsets = [-3.0, -1.5, 0.0, 1.5, 3.0]
    frames += [_city_block(rng, 200 + i, 150, offset=o)        # "large": 5 cities, spread offsets
               for i, o in enumerate(offsets)]
    df = pd.concat(frames, ignore_index=True)
    bracket_of = {100: "mega", **{200 + i: "large" for i in range(5)}}

    out = dbg.bracket_decomposition(df, bracket_of)
    mega = out[out["bracket"] == "mega"].iloc[0]
    large = out[out["bracket"] == "large"].iloc[0]

    assert large["rho_pooled"] < mega["rho_pooled"] - 0.1     # observed symptom
    assert abs(large["rho_within"] - mega["rho_within"]) < 0.05  # same true skill
    assert large["pooling_gap"] > 0.1
    assert large["rho_demeaned"] > large["rho_pooled"] + 0.1  # demeaning recovers it
    assert mega["pooling_gap"] == pytest.approx(0.0, abs=0.02)
    assert large["n_cities"] == 5 and mega["n_cities"] == 1


def test_no_offsets_no_pooling_gap():
    rng = np.random.default_rng(825)
    df = pd.concat([_city_block(rng, c, 150) for c in (1, 2, 3)], ignore_index=True)
    out = dbg.bracket_decomposition(df, {1: "large", 2: "large", 3: "large"})
    assert out.iloc[0]["pooling_gap"] == pytest.approx(0.0, abs=0.03)


# ── H2/H3: per-city covariates ───────────────────────────────────────────────

def test_covariate_correlations_detect_label_noise():
    """Cities with noisier labels must show lower rho -> negative correlation."""
    rng = np.random.default_rng(825)
    frames, noise = [], {}
    for i in range(12):
        s = i / 11.0                       # label noise 0 .. 1
        frames.append(_city_block(rng, 1000 + i, 200, noise_on_label=s))
        noise[1000 + i] = s ** 2
    df = pd.concat(frames, ignore_index=True)

    meta = pd.DataFrame({"cbsa_code": list(noise), "cbsa_title": "x",
                         "bracket": "large", "population": 1e6, "n_tracts": 200})
    covs = pd.DataFrame({CITY_COL: list(noise), "label_noise_se2": list(noise.values()),
                         "mean_log_pci": 10.3})
    city_df = dbg.build_city_table(df, meta, covariates=covs)
    corr = dbg.covariate_correlations(city_df)
    r = corr.set_index("covariate").loc["label_noise_se2", "spearman_r"]
    assert r < -0.5


def test_build_city_table_merges_meta_and_flags_small_cities():
    rng = np.random.default_rng(825)
    df = pd.concat([_city_block(rng, 1, 100), _city_block(rng, 2, 5)], ignore_index=True)
    meta = pd.DataFrame({"cbsa_code": [1, 2], "cbsa_title": ["A", "B"],
                         "bracket": ["large", "small"], "population": [2e6, 5e5],
                         "n_tracts": [500, 100]})
    out = dbg.build_city_table(df, meta, min_n=20)
    assert out.loc[out[CITY_COL] == 2, "rho"].isna().all()      # too few rows
    assert out.loc[out[CITY_COL] == 1, "cbsa_title"].iloc[0] == "A"
    assert "label_std" in out and "abs_offset" in out


# ── inference plumbing (mocked model, fake shards) ───────────────────────────

def _fake_shard(n, cbsa, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    return {
        "images": torch.randint(0, 255, (n, 3, 8, 8), generator=g, dtype=torch.uint8),
        "scores": torch.randn(n, generator=g),
        "geoids": torch.arange(n, dtype=torch.int64),
        "years": torch.full((n,), 2020, dtype=torch.int64),
        "building_ids": torch.arange(n, dtype=torch.int64),
        "structural_change": torch.zeros(n, dtype=torch.int64),
        "metas": torch.randn(n, 1, generator=g),
        "score_bins": torch.zeros(n, dtype=torch.int64),
        "cbsa_ids": torch.full((n,), cbsa, dtype=torch.int64),
    }


def test_load_val_shards_concatenates_and_drops_cbsa_zero(tmp_path, capsys):
    import torch
    d1 = tmp_path / "val_cities_cache"; d1.mkdir()
    d2 = tmp_path / "val_temporal_cache"; d2.mkdir()
    torch.save(_fake_shard(6, cbsa=100), d1 / "shard_0.pt")
    torch.save(_fake_shard(4, cbsa=0), d1 / "shard_1.pt")     # pre-US rows -> dropped
    torch.save(_fake_shard(5, cbsa=200), d2 / "shard_0.pt")
    df, imgs = dbg.load_val_shards([d1, d2])
    assert len(df) == 11 and len(imgs) == 11
    assert set(df["set"]) == {"val_cities_cache", "val_temporal_cache"}
    assert (df[CITY_COL] != 0).all()
    assert "dropped" in capsys.readouterr().out


def test_predict_scores_order_scaling_and_meta():
    import torch

    class Probe(torch.nn.Module):
        def forward(self, x, metadata=None):
            # mean pixel of the (normalized) image + the meta value
            return x.mean(dim=(1, 2, 3)) + metadata.view(-1)

    n = 10
    images = torch.randint(0, 255, (n, 3, 8, 8), dtype=torch.uint8)
    metas = np.arange(n, dtype=np.float32)
    preds = dbg.predict_scores(Probe(), images, metas, device="cpu", batch_size=3)
    assert preds.shape == (n,)
    tfm = dbg.build_eval_transform(3)
    expected = (tfm(images).mean(dim=(1, 2, 3)) + torch.arange(n, dtype=torch.float32)).numpy()
    assert np.allclose(preds, expected, atol=1e-5)
    # scale=True really rescaled uint8 (normalized means are O(1), not O(100))
    assert abs(tfm(images).mean().item()) < 5


def test_main_analysis_only_end_to_end(tmp_path, capsys):
    rng = np.random.default_rng(825)
    frames = [_city_block(rng, 100, 120), _city_block(rng, 201, 120, offset=1.0),
              _city_block(rng, 202, 120, offset=-1.0)]
    df = pd.concat(frames, ignore_index=True)
    preds_csv = tmp_path / "preds.csv"; df.to_csv(preds_csv, index=False)
    splits = tmp_path / "splits.feather"
    pd.DataFrame({"cbsa_code": [100, 201, 202], "cbsa_title": ["M", "L1", "L2"],
                  "bracket": ["mega", "large", "large"], "population": [9e6, 2e6, 2e6],
                  "n_tracts": [2000, 500, 500], "split": ["val"] * 3,
                  "holdout_year": [np.nan] * 3}).to_feather(splits)

    rc = dbg.main(["--from-preds", str(preds_csv), "--splits-feather", str(splits),
                   "--panel", str(tmp_path / "missing.feather")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "BRACKET GAP DECOMPOSITION" in out
    assert "panel covariates unavailable" in out   # graceful H2 degradation
    assert "pooling_gap" in out


def test_main_requires_inputs():
    with pytest.raises(SystemExit):
        dbg.main([])
