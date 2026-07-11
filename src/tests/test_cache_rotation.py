"""Unit tests for CyclicCacheManager background rotation robustness.

Covers the failure modes that silently froze rotation for a full training run:
  * a crashed background thread (exception) must not swap in unwritten shards,
    must retry the failed shard id, and must keep rotating afterwards;
  * a hung background thread must be reported (stall counter), not ignored;
  * normal completion swaps exactly k shards and advances the schedule.

The manager is built via __new__ with synthetic state — no NAIP, no ACS, no
disk-heavy init — so these run anywhere.
"""
import threading
import time
from pathlib import Path

import pytest

from src.main import CyclicCacheManager


def make_manager(tmp_path, num_shards=4, k_written_per_call=None):
    m = CyclicCacheManager.__new__(CyclicCacheManager)
    m.cache_dir = Path(tmp_path)
    m.num_shards = num_shards
    m.shard_size = 8
    m.single_shard_mode = False
    m.progress = 0.0
    m.bg_thread = None
    m._pending_k = 0
    m._bg_error = None
    m._bg_completed = []
    m._bg_start_idx = 0
    m._bg_started_at = None
    m._bg_stall_steps = 0
    m.last_swap_count = 0
    m._is_initialized = True
    # Seed an initial active cache: shards 0..num_shards-1 on disk.
    m.active_shards = []
    for i in range(num_shards):
        p = m.cache_dir / f"shard_{i}.pt"
        p.write_bytes(b"seed")
        m.active_shards.append(p)
    m.next_shard_idx = num_shards
    return m


def patch_worker(m, behavior):
    """behavior(shard_id) -> None; raises to simulate failure; writes the file."""
    def fake_generate(shard_id, show_progress=False):
        behavior(shard_id)
    m._worker_generate = fake_generate


def run_bg_to_completion(m):
    m.bg_thread.join(timeout=10)
    assert not m.bg_thread.is_alive()


def test_normal_rotation_swaps_k_shards(tmp_path):
    m = make_manager(tmp_path)
    patch_worker(m, lambda sid: (m.cache_dir / f"shard_{sid}.pt").write_bytes(b"new"))

    assert m.step(k=2) is False          # first call only launches the thread
    run_bg_to_completion(m)              # shards 4,5 written

    assert m.step(k=2) is True
    assert m.last_swap_count == 2
    names = [p.name for p in m.active_shards]
    assert names == ["shard_2.pt", "shard_3.pt", "shard_4.pt", "shard_5.pt"]
    assert not (m.cache_dir / "shard_0.pt").exists()   # oldest deleted
    assert not (m.cache_dir / "shard_1.pt").exists()
    run_bg_to_completion(m)              # next batch (6,7) relaunched automatically
    assert m.step(k=2) is True           # ... and swapped in on the next epoch
    assert m.next_shard_idx == 8


def test_crashed_thread_swaps_only_completed_and_retries(tmp_path):
    m = make_manager(tmp_path)
    calls = []

    def flaky(sid):
        calls.append(sid)
        if sid == 5 and calls.count(5) == 1:   # transient: fails once
            raise RuntimeError("NAIP failure rate exceeded")
        (m.cache_dir / f"shard_{sid}.pt").write_bytes(b"new")

    patch_worker(m, flaky)
    m.step(k=2)                          # launch shards 4,5; 5 raises
    run_bg_to_completion(m)
    assert m._bg_error is not None

    assert m.step(k=2) is True           # shard 4 swapped in, error surfaced
    assert m.last_swap_count == 1
    assert [p.name for p in m.active_shards] == \
        ["shard_1.pt", "shard_2.pt", "shard_3.pt", "shard_4.pt"]
    # No phantom shard_5 in the active list, shard_0 (one oldest) deleted only.
    assert (m.cache_dir / "shard_1.pt").exists()
    # Failed shard id 5 is retried by the relaunched batch.
    run_bg_to_completion(m)
    assert calls == [4, 5, 5, 6]         # retry of 5, then 6


def test_crash_on_first_shard_swaps_nothing_but_relaunches(tmp_path):
    m = make_manager(tmp_path)
    attempts = []

    def always_fail(sid):
        attempts.append(sid)
        raise RuntimeError("boom")

    patch_worker(m, always_fail)
    m.step(k=2)
    run_bg_to_completion(m)

    assert m.step(k=2) is False          # nothing swapped ...
    assert m.last_swap_count == 0
    assert len(m.active_shards) == 4     # ... and no good shards deleted
    assert all(p.exists() for p in m.active_shards)
    run_bg_to_completion(m)
    assert attempts == [4, 4]            # ... but the batch WAS retried


def test_hung_thread_reports_stall_and_never_swaps(tmp_path):
    m = make_manager(tmp_path)
    release = threading.Event()
    patch_worker(m, lambda sid: release.wait(timeout=30))

    m.step(k=2)                          # launch; worker blocks
    time.sleep(0.05)
    before = list(m.active_shards)
    for i in range(1, 4):                # three epochs of a hung thread
        assert m.step(k=2) is False
        assert m._bg_stall_steps == i
    assert m.active_shards == before     # cache untouched
    assert m.next_shard_idx == 4         # schedule not advanced
    release.set()
    run_bg_to_completion(m)


def test_completed_but_missing_file_is_not_swapped(tmp_path):
    m = make_manager(tmp_path)
    patch_worker(m, lambda sid: None)    # "completes" without writing anything

    m.step(k=2)
    run_bg_to_completion(m)
    assert m.step(k=2) is False
    assert m.last_swap_count == 0
    assert len(m.active_shards) == 4 and all(p.exists() for p in m.active_shards)


def test_single_shard_mode_never_rotates(tmp_path):
    m = make_manager(tmp_path)
    m.single_shard_mode = True
    assert m.step(k=2) is False
    assert m.bg_thread is None
