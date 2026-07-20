"""Manual preview generator for part_main_figure_us — NOT a pytest test file
(doesn't match test_*.py, so pytest won't collect it).

Builds a synthetic 35-city, 8-year US results tree (like the real test/held-
out split shape) and renders the 3-panel main performance figure so you can
eyeball layout/style changes without a real results tree. Run with:

    python src/tests/preview_main_figure.py [output_dir]

Writes <output_dir>/figures/US_main_performance_figure.pdf (default
output_dir: results/_preview under the repo root, gitignored like the rest of
results/).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import src.evaluation as ev


class _FakeCtx:
    """Minimal stand-in for _USContext exposing only what the figure reads."""

    def __init__(self, bld, tract, cbsa_meta, out):
        self.bld, self.tract, self.cbsa_meta, self.out = bld, tract, cbsa_meta, out


def build_synthetic_ctx(out: Path, seed: int = 42) -> _FakeCtx:
    rng = np.random.default_rng(seed)
    brackets = ["mega"] * 3 + ["large"] * 10 + ["medium"] * 12 + ["small"] * 10
    cbsas = {f"{10000 + i}": b for i, b in enumerate(brackets)}
    cbsa_meta = pd.DataFrame({"cbsa_code": list(cbsas.keys()), "bracket": list(cbsas.values())})

    years = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
    bld_rows, tract_rows = [], []
    for cbsa in cbsas:
        city_noise = rng.uniform(0.15, 0.45)
        for yr in years:
            n_tracts = rng.integers(40, 120)
            for i in range(n_tracts):
                label = rng.normal(0, 1)
                pred = label + rng.normal(0, city_noise)
                for b in range(3):
                    bld_rows.append({
                        "cbsa": cbsa, "year": yr, "type": "test",
                        "pred": pred + rng.normal(0, 0.1), "label": label,
                        "building_id": f"{cbsa}_{yr}_{i}_{b}",
                    })
                tract_rows.append({
                    "cbsa": cbsa, "year": yr, "GEOID": f"{cbsa}{i:05d}",
                    "pred": pred, "label": label,
                })

    bld = pd.DataFrame(bld_rows)
    tract = pd.DataFrame(tract_rows)
    return _FakeCtx(bld, tract, cbsa_meta, out)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[2] / "results" / "_preview"
    )
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    ctx = build_synthetic_ctx(out)
    headline = ev.part_main_figure_us(ctx, holdout_year=2016)
    print(headline)
    print("figure ->", out / "figures" / "US_main_performance_figure.pdf")


if __name__ == "__main__":
    main()
