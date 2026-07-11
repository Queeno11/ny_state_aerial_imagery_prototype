"""Real-data memory smoke test for the normalized (lazy) US-scale path.

OPT-IN: skipped unless RUN_REAL_SMOKE=1 and the real buildings_index + ACS panel
are present. Builds the two pair artifacts from the REAL full-US buildings index
(71.8M buildings — the exact input that OOM-killed the legacy flat path),
materializes shard-sized slices, and asserts peak RSS stays well under the box.

Run:
  RUN_REAL_SMOKE=1 IMAGERY_ROOT=... ACS_ROOT_DIR=... \
    python -m pytest src/tests/test_pair_table_real_smoke.py -s -q
"""
import os
import threading
import time

import pytest

import src.build_dataset as bd
from src.data.pair_table import FLAT_COLUMNS, LazyPairTable
from src.utils.paths import PROCESSED_DATA_DIR

CEILING_GB = 20.0   # emergency abort, safely under the 27 GB box
BUDGET_GB = 15.0    # regression guard: observed ~12 GB (legacy flat path blew past 32 GB)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_SMOKE") != "1"
    or not (PROCESSED_DATA_DIR / "buildings_index").exists(),
    reason="real-data smoke test (set RUN_REAL_SMOKE=1 with real data present)",
)


def _rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024 ** 2)
    return 0.0


class _Watchdog:
    def __init__(self, ceiling_gb):
        self.ceiling = ceiling_gb
        self.peak = 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            r = _rss_gb()
            self.peak = max(self.peak, r)
            if r > self.ceiling:
                print(f"\n🚨 RSS {r:.1f} GB exceeded ceiling {self.ceiling} GB — aborting.",
                      flush=True)
                os._exit(2)
            time.sleep(0.25)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.peak = max(self.peak, _rss_gb())


def test_full_us_pair_table_under_budget(tmp_path, monkeypatch):
    # Reads come from the REAL processed dir (index + ACS panel); artifact
    # writes are redirected to tmp (data/processed is read-only in the sandbox).
    real = PROCESSED_DATA_DIR
    (tmp_path / "buildings_index").symlink_to(real / "buildings_index")
    for f in real.glob("us_metros_panel_*.feather"):
        (tmp_path / f.name).symlink_to(f)
    (tmp_path / "figures").mkdir()
    monkeypatch.setattr(bd, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(bd, "FIGURES_DIR", tmp_path / "figures")

    years = list(range(2010, 2025, 2))   # matches main.py __main__
    with _Watchdog(CEILING_GB) as wd:
        t0 = time.time()
        table = bd.load_income_dataset(
            years, tau_meters=100, indicator="W2_r5",
            footprints_source="ms_us", states=None,
        )
        assert isinstance(table, LazyPairTable)
        build_peak = wd.peak

        # Shard generator's access pattern: a 20,480-row cyclic slice.
        sl = table.materialize(0, 20480)
        assert len(sl) == 20480
        assert list(sl.columns) == list(FLAT_COLUMNS)
        assert sl["dataset"].eq("NAIP").all()
        ny = table.n_years
        assert sl["building_id"].iloc[:ny].nunique() == 1          # twins adjacent
        assert sl["Rel_Score"].notna().mean() > 0.5

        wrap = table.materialize(len(table) - 5, len(table) + 5)   # wraparound
        assert len(wrap) == 10

        yr = table.materialize_year(years[0])                      # prediction path
        assert len(yr) == table.n_buildings

    print(f"\n{'='*60}\n✅ REAL SMOKE PASSED in {time.time()-t0:.0f}s"
          f"\n   buildings:        {table.n_buildings:,}"
          f"\n   pairs (implicit): {len(table):,}"
          f"\n   peak RSS @build:  {build_peak:.2f} GB"
          f"\n   peak RSS total:   {wd.peak:.2f} GB  (budget {BUDGET_GB} GB)\n{'='*60}")
    assert wd.peak < BUDGET_GB, f"peak RSS {wd.peak:.2f} GB exceeded budget {BUDGET_GB} GB"

    # ── Coverage-gap masking at real scale ───────────────────────────────────
    # Flag Urban Honolulu, HI (CBSA 46520 — no NAIP anywhere) as unavailable and
    # confirm the pipeline masks every one of its pairs (so the main model skips
    # them before any fetch). Reload is cheap: the pair parquets are cached.
    dead_cbsa = "46520"
    n_honolulu = int((table.buildings["cbsa_code"].astype(str) == dead_cbsa).sum())
    if n_honolulu == 0:
        pytest.skip("CBSA 46520 not present in this build; skipping masking check")
    bd.write_naip_unavailable(
        [{"level": "cbsa", "key": dead_cbsa, "year": y} for y in years], merge=False)
    masked_table = bd.load_income_dataset(
        years, tau_meters=100, indicator="W2_r5",
        footprints_source="ms_us", states=None,
    )
    assert masked_table.unavailable is not None
    y0 = masked_table.materialize_year(years[0])
    dead = y0["cbsa_code"].astype(str) == dead_cbsa
    assert dead.sum() == n_honolulu
    assert y0.loc[dead, "Rel_Score"].isna().all()      # every Honolulu pair dropped
    # Flagged for all 8 years ⇒ recognized as fully dead → excluded from splits.
    dead_cbsas, _ = masked_table.fully_unavailable()
    assert dead_cbsa in dead_cbsas
    print(f"✅ coverage-gap masking: {n_honolulu:,} Honolulu (46520) buildings "
          f"masked for {years[0]} (and every year) — skipped before fetch, "
          f"and excluded from the train/val/test split.")
