"""Executes the visualize_val_cities notebook cells against synthetic val caches.

Same pattern as test_visualize_debug_batches_notebook: build fake shards with the
real schema, then run the notebook's code cells with the config cell replaced —
so the shard format, the preds-CSV merge, and the renderer can't drift apart.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from src.tests.test_diagnose_bracket_gap import _fake_shard
from src.utils.paths import PROJECT_ROOT

NOTEBOOK = PROJECT_ROOT / "src" / "notebooks" / "visualize_val_cities.ipynb"


@pytest.fixture()
def synthetic_val(tmp_path):
    """Two val cache dirs (2 cities + 1 city), a matching preds CSV, splits feather."""
    d1 = tmp_path / "val_cities_cache"; d1.mkdir()
    d2 = tmp_path / "val_temporal_cache"; d2.mkdir()
    torch.save(_fake_shard(12, cbsa=19430, seed=1), d1 / "shard_0.pt")   # Dayton
    torch.save(_fake_shard(8, cbsa=17820, seed=2), d1 / "shard_1.pt")    # Colo. Springs
    torch.save(_fake_shard(9, cbsa=19100, seed=3), d2 / "shard_0.pt")    # Dallas

    rows = []
    for set_name, cache in (("val_cities_cache", d1), ("val_temporal_cache", d2)):
        for sp in sorted(cache.glob("shard_*.pt")):
            d = torch.load(sp, weights_only=False)
            rows.append(pd.DataFrame({
                "set": set_name,
                "building_id": d["building_ids"].numpy(),
                "year": d["years"].numpy(),
                "Rel_Score": d["scores"].numpy(),
                "cbsa_code": d["cbsa_ids"].numpy(),
                # perfect-rank preds + one huge residual to exercise the red border
                "predicted_value": d["scores"].numpy() * 0.9,
            }))
    preds = pd.concat(rows, ignore_index=True)
    preds.loc[0, "predicted_value"] = preds.loc[0, "Rel_Score"] + 5.0
    preds_csv = tmp_path / "val_bracket_preds.csv"
    preds.to_csv(preds_csv, index=False)

    splits = tmp_path / "cbsa_splits.feather"
    pd.DataFrame({
        "cbsa_code": ["19430", "17820", "19100"],
        "cbsa_title": ["Dayton-Kettering-Beavercreek, OH",
                       "Colorado Springs, CO", "Dallas-Fort Worth-Arlington, TX"],
        "bracket": ["medium", "medium", "mega"],
        "population": [8e5, 7.6e5, 7.8e6], "n_tracts": [221, 175, 1704],
        "split": ["val", "val", "train"], "holdout_year": [np.nan, np.nan, 2016.0],
    }).to_feather(splits)
    return {"cache_dirs": [d1, d2], "preds_csv": preds_csv, "splits": splits}


def test_notebook_cells_run_on_synthetic_val(synthetic_val, tmp_path, monkeypatch):
    nb = json.loads(NOTEBOOK.read_text())
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

    figs = []
    monkeypatch.setattr(plt, "show", lambda: figs.append(
        plt.gcf().savefig(tmp_path / f"fig_{len(figs):02d}.png", dpi=40)) or plt.close("all"))

    ns = {"display": lambda *a, **k: None}
    exec(
        "import os\nfrom pathlib import Path\nimport matplotlib.pyplot as plt\n"
        "import numpy as np\nimport pandas as pd\n"
        "from src.diagnose_bracket_gap import load_val_shards\n", ns,
    )
    # replace the config cell (cell 0 resolves real CACHE_DIR/RESULTS_DIR paths)
    ns.update(CACHE_DIRS=synthetic_val["cache_dirs"], PREDS_CSV=synthetic_val["preds_csv"],
              SPLITS_FEATHER=synthetic_val["splits"],
              CITY="Dayton", SORT_BY="label", PAGE_SIZE=5, PAGES=None)

    # run everything except the final optional-widgets cell first: whether that
    # cell renders an extra page depends on ipywidgets being importable
    for cell in code_cells[1:-1]:
        exec(cell, ns)
    n_static_figs = len(figs)
    exec(code_cells[-1], ns)
    assert len(figs) in (n_static_figs, n_static_figs + 1)

    df = ns["df"]
    assert len(df) == 29 and len(ns["IMAGES"]) == 29
    assert df["predicted_value"].notna().all()          # merge matched every tile
    assert set(df["city"].unique()) == {"Dayton-Kettering-Beavercreek, OH",
                                        "Colorado Springs, CO",
                                        "Dallas-Fort Worth-Arlington, TX"}
    # summary has a rho per (set, city) and Dallas only in the temporal set
    summary = ns["SUMMARY"]
    assert summary["rho"].notna().all()
    assert (summary.loc[summary["city"].str.startswith("Dallas"), "set"]
            == "val_temporal_cache").all()

    # Dayton: 12 tiles at PAGE_SIZE=5 -> 3 pages rendered by the static cell
    assert n_static_figs == 3
    # sorted by label within the rendered city
    pages = ns["city_pages"](19430)
    lab = pd.concat(pages)["Rel_Score"].to_numpy()
    assert (np.diff(lab) >= 0).all()

    # city resolution: substring, exact code, ambiguous/missing
    assert ns["resolve_city"]("colorado") == 17820
    assert ns["resolve_city"](19100) == 19100
    with pytest.raises(ValueError):
        ns["resolve_city"]("D")            # Dayton + Dallas -> ambiguous
    with pytest.raises(ValueError):
        ns["resolve_city"](None)
