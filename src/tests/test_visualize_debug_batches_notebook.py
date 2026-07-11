"""Executes the visualize_debug_batches notebook cells against a synthetic dump.

Builds a dump with the real BatchImageDumper + HybridBatchSampler (the exact
producer path in main.py), then runs the notebook's code cells on it — so the
dump format and the notebook's reader/renderer can never drift apart silently.
"""

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from src.debug_batch_dump import BatchImageDumper
from src.main import HybridBatchSampler
from src.tests.test_debug_batch_dump import _fake_dataset
from src.utils.paths import PROJECT_ROOT

NOTEBOOK = PROJECT_ROOT / "src" / "notebooks" / "visualize_debug_batches.ipynb"


@pytest.fixture()
def synthetic_dump(tmp_path):
    """A 5-batch dump over 2 CBSAs x 3 years with recognizable synthetic tiles."""
    random.seed(0)
    ds = _fake_dataset(n_buildings=24, years=(2014, 2016, 2018), cbsas=(35620, 16980))
    n = len(ds.years)
    imgs = torch.zeros(n, 4, 64, 64, dtype=torch.uint8)
    for i in range(n):
        # brightness encodes year, stripe offset encodes building — visually checkable
        base = 50 + (int(ds.years[i]) - 2014) * 45
        imgs[i, :3] = base
        imgs[i, :3, int(ds.building_ids[i]) % 8 :: 8, :] = 230
    ds.images = imgs
    ds.scores = torch.linspace(-2, 2, n)

    out = tmp_path / "sim_run" / "debug_batches"
    dumper = BatchImageDumper(out, max_batches=5)
    dumper.write_run_info(
        {"indicator": "median_household_income", "years": [2014, 2016, 2018],
         "image_size": 64, "nbands": 4, "max_jitter": 0, "tau_meters": 100.0,
         "subsample_step": 1, "sat_data": "NAIP"},
        "sim_run",
    )
    df_train = pd.DataFrame({
        "building_id": np.arange(24),
        "GEOID": [f"36061{i:06d}" if i % 2 == 0 else f"17031{i:06d}" for i in range(24)],
        "cbsa_code": [35620 if i % 2 == 0 else 16980 for i in range(24)],
    })
    cbsa_meta = pd.DataFrame({
        "cbsa_code": ["35620", "16980"],
        "cbsa_title": ["New York-Newark-Jersey City, NY-NJ-PA", "Chicago-Naperville-Elgin, IL-IN-WI"],
    })
    dumper.write_building_reference(df_train, cbsa_meta=cbsa_meta)

    sampler = HybridBatchSampler(ds, batch_size_cs=8, max_temporal_per_batch=4,
                                 batch_dumper=dumper)
    list(sampler)
    assert len(list(out.glob("batch_*.pt"))) == 5
    return out


def test_notebook_cells_run_on_dump(synthetic_dump, tmp_path, monkeypatch):
    nb = json.loads(NOTEBOOK.read_text())
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]

    figs = []
    monkeypatch.setattr(plt, "show", lambda: figs.append(
        plt.gcf().savefig(tmp_path / f"fig_{len(figs):02d}.png", dpi=50, bbox_inches="tight")
    ))

    ns = {"display": lambda *a, **k: None}
    # Cell 0 only resolves DUMP_DIR from the real RESULTS_DIR — skip it and
    # inject the synthetic dump plus the cell's imports/config instead.
    exec(
        "import json\nfrom pathlib import Path\nimport matplotlib.pyplot as plt\n"
        "import numpy as np\nimport pandas as pd\nimport torch\n", ns,
    )
    ns.update(DUMP_DIR=synthetic_dump, SAVENAME=None,
              BATCHES_TO_SHOW=None, MAX_ROWS_PER_BATCH=None)

    for cell in code_cells[1:]:
        exec(cell, ns)   # includes the notebook's own CS-core/twin sanity asserts

    assert ns["INDICATOR"] == "median_household_income"
    assert len(ns["payloads"]) == 5
    meta = ns["meta"]
    assert set(meta["role"]) <= {"CS", "twin"}
    assert meta["city"].str.startswith(("New York", "Chicago")).all()
    assert meta["GEOID"].str.len().eq(11).all()
    # 5 per-batch figures + 1 cross-batch figure were rendered
    assert len(figs) == 6
    assert all(p.exists() for p in tmp_path.glob("fig_*.png"))
