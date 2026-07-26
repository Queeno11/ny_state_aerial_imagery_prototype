"""Crash-safety of checkpoint writes (no GPU, no network).

Covers the guarantee that a kill mid-save can never leave an unreadable file at
the destination -- the failure that produced

    RuntimeError: PytorchStreamReader failed reading zip archive:
                  failed finding central directory

on a resume from a 670 MB run_*_last.pth whose zip trailer was never written.

Crashes are simulated by raising from inside torch.save / save_pretrained, i.e.
at exactly the point the real kills landed.

Also covers main.select_resume_checkpoint, the reader-side half: a corrupt
checkpoint degrades to the previous epoch / best weights instead of aborting a
run that has already spent ~20 minutes building the dataset.
"""

import os
import zipfile
from pathlib import Path

import pytest
import torch

from src.utils.atomic_io import (
    atomic_save_dir,
    atomic_torch_save,
    atomic_write_text,
    is_valid_torch_checkpoint,
    previous_version_path,
)


def _payload(v: float = 1.0):
    return {"model_state_dict": {"w": torch.full((64,), v)}, "epoch": int(v)}


class _Boom(RuntimeError):
    """Stands in for SIGKILL / OOM / reboot landing inside the save."""


# ── happy path ───────────────────────────────────────────────────────────────

def test_roundtrip(tmp_path):
    p = tmp_path / "run_last.pth"
    atomic_torch_save(_payload(3.0), p)
    assert torch.load(p, weights_only=False)["epoch"] == 3
    assert is_valid_torch_checkpoint(p)


def test_creates_missing_parent(tmp_path):
    p = tmp_path / "models_by_epoch" / "run" / "run_last.pth"
    atomic_torch_save(_payload(), p)
    assert p.is_file()


def test_no_tmp_left_behind(tmp_path):
    p = tmp_path / "run_last.pth"
    atomic_torch_save(_payload(), p)
    assert list(tmp_path.glob("*.tmp")) == []


# ── the actual bug: crash mid-write ──────────────────────────────────────────

def test_crash_midwrite_leaves_previous_file_intact(tmp_path, monkeypatch):
    """The regression test. Old checkpoint must survive an interrupted save."""
    p = tmp_path / "run_last.pth"
    atomic_torch_save(_payload(1.0), p)

    real_save = torch.save

    def exploding_save(obj, fh, *a, **kw):
        real_save(obj, fh)          # write real bytes, as a truncated save would
        fh.flush()
        raise _Boom("killed mid-save")

    monkeypatch.setattr(torch, "save", exploding_save)
    with pytest.raises(_Boom):
        atomic_torch_save(_payload(2.0), p)
    monkeypatch.undo()

    # Destination still holds the *old* complete checkpoint, and is loadable.
    assert is_valid_torch_checkpoint(p)
    assert torch.load(p, weights_only=False)["epoch"] == 1


def test_crash_midwrite_on_first_ever_save_leaves_no_destination(tmp_path, monkeypatch):
    """Nothing at the destination beats a truncated file at the destination."""
    p = tmp_path / "run_last.pth"

    def exploding_save(obj, fh, *a, **kw):
        fh.write(b"PK\x03\x04partial")
        raise _Boom

    monkeypatch.setattr(torch, "save", exploding_save)
    with pytest.raises(_Boom):
        atomic_torch_save(_payload(), p)

    assert not p.exists()


def test_truncated_destination_is_detected(tmp_path):
    """is_valid_torch_checkpoint reproduces the real symptom on a real torch file."""
    p = tmp_path / "run_last.pth"
    atomic_torch_save(_payload(), p)
    assert is_valid_torch_checkpoint(p)

    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])       # lose the central directory

    assert not is_valid_torch_checkpoint(p)
    with pytest.raises(Exception):
        torch.load(p, weights_only=False)


@pytest.mark.parametrize("content", [b"", b"not a checkpoint at all"])
def test_invalid_files_rejected(tmp_path, content):
    p = tmp_path / "junk.pth"
    p.write_bytes(content)
    assert not is_valid_torch_checkpoint(p)


def test_missing_file_rejected(tmp_path):
    assert not is_valid_torch_checkpoint(tmp_path / "nope.pth")


# ── keep_previous rotation ───────────────────────────────────────────────────

def test_keep_previous_rotates_one_deep(tmp_path):
    p = tmp_path / "run_last.pth"
    prev = previous_version_path(p)
    assert prev.name == "run_last.prev.pth"

    atomic_torch_save(_payload(1.0), p, keep_previous=True)
    assert not prev.exists()                     # nothing to rotate yet

    atomic_torch_save(_payload(2.0), p, keep_previous=True)
    assert torch.load(p, weights_only=False)["epoch"] == 2
    assert torch.load(prev, weights_only=False)["epoch"] == 1

    atomic_torch_save(_payload(3.0), p, keep_previous=True)
    assert torch.load(p, weights_only=False)["epoch"] == 3
    assert torch.load(prev, weights_only=False)["epoch"] == 2   # only one deep


def test_keep_previous_survives_destination_corruption(tmp_path):
    """A checkpoint corrupted after the fact still leaves a loadable fallback."""
    p = tmp_path / "run_last.pth"
    atomic_torch_save(_payload(1.0), p, keep_previous=True)
    atomic_torch_save(_payload(2.0), p, keep_previous=True)

    p.write_bytes(p.read_bytes()[:100])

    assert not is_valid_torch_checkpoint(p)
    assert is_valid_torch_checkpoint(previous_version_path(p))
    assert torch.load(previous_version_path(p), weights_only=False)["epoch"] == 1


# ── directory publish (LoRA adapter) ─────────────────────────────────────────

def _write_adapter(d: Path, tag: str):
    (d / "adapter_config.json").write_text('{"tag": "%s"}' % tag)
    (d / "adapter_model.safetensors").write_bytes(tag.encode() * 8)


def test_atomic_save_dir_publishes(tmp_path):
    d = tmp_path / "run_best_lora"
    atomic_save_dir(lambda staging: _write_adapter(staging, "v1"), d)

    assert (d / "adapter_config.json").read_text() == '{"tag": "v1"}'
    assert not (tmp_path / "run_best_lora.tmp").exists()
    assert not (tmp_path / "run_best_lora.old").exists()


def test_atomic_save_dir_replaces_existing(tmp_path):
    d = tmp_path / "run_best_lora"
    atomic_save_dir(lambda s: _write_adapter(s, "v1"), d)
    atomic_save_dir(lambda s: _write_adapter(s, "v2"), d)

    assert (d / "adapter_config.json").read_text() == '{"tag": "v2"}'
    assert not (tmp_path / "run_best_lora.old").exists()


def test_atomic_save_dir_crash_keeps_old_adapter(tmp_path):
    """Interrupted between config and weights -> previous adapter still whole."""
    d = tmp_path / "run_best_lora"
    atomic_save_dir(lambda s: _write_adapter(s, "v1"), d)

    def half_written(staging: Path):
        (staging / "adapter_config.json").write_text('{"tag": "v2"}')
        raise _Boom("killed between config and safetensors")

    with pytest.raises(_Boom):
        atomic_save_dir(half_written, d)

    assert (d / "adapter_config.json").read_text() == '{"tag": "v1"}'
    assert (d / "adapter_model.safetensors").read_bytes() == b"v1" * 8


def test_atomic_save_dir_clears_stale_staging(tmp_path):
    d = tmp_path / "run_best_lora"
    stale = tmp_path / "run_best_lora.tmp"
    stale.mkdir()
    (stale / "garbage.bin").write_bytes(b"x")

    atomic_save_dir(lambda s: _write_adapter(s, "v1"), d)

    assert not (d / "garbage.bin").exists()
    assert not stale.exists()


# ── text ─────────────────────────────────────────────────────────────────────

def test_atomic_write_text_roundtrip(tmp_path):
    p = tmp_path / "params.json"
    atomic_write_text(p, '{"a": 1}')
    assert p.read_text() == '{"a": 1}'
    assert list(tmp_path.glob("*.tmp")) == []


# ── resume selection (main.select_resume_checkpoint) ─────────────────────────

RUN = "run_20260721"


def _ckpt_dir(tmp_path):
    d = tmp_path / "models_by_epoch" / RUN
    d.mkdir(parents=True)
    return d


def _write(d: Path, name: str, epoch: float, *, corrupt=False):
    p = d / name
    atomic_torch_save(_payload(epoch), p)
    if corrupt:
        p.write_bytes(p.read_bytes()[:100])     # truncate: no zip trailer
    return p


def test_select_prefers_last(tmp_path):
    from src.main import select_resume_checkpoint
    d = _ckpt_dir(tmp_path)
    last = _write(d, f"{RUN}_last.pth", 9)
    _write(d, f"{RUN}_last.prev.pth", 8)
    _write(d, f"{RUN}_best.pth", 5)

    assert select_resume_checkpoint(d, RUN) == last


def test_select_falls_back_to_prev_when_last_corrupt(tmp_path, capsys):
    """The exact scenario that crashed the run, now recoverable."""
    from src.main import select_resume_checkpoint
    d = _ckpt_dir(tmp_path)
    _write(d, f"{RUN}_last.pth", 9, corrupt=True)
    prev = _write(d, f"{RUN}_last.prev.pth", 8)
    _write(d, f"{RUN}_best.pth", 5)

    chosen = select_resume_checkpoint(d, RUN)

    assert chosen == prev
    assert torch.load(chosen, weights_only=False)["epoch"] == 8
    assert "corrupt/truncated" in capsys.readouterr().out


def test_select_falls_back_to_best_when_last_and_prev_corrupt(tmp_path):
    from src.main import select_resume_checkpoint
    d = _ckpt_dir(tmp_path)
    _write(d, f"{RUN}_last.pth", 9, corrupt=True)
    _write(d, f"{RUN}_last.prev.pth", 8, corrupt=True)
    best = _write(d, f"{RUN}_best.pth", 5)

    assert select_resume_checkpoint(d, RUN) == best


def test_select_falls_back_to_best_when_no_last(tmp_path):
    """Pre-existing behaviour preserved: best-only dir resumes from best."""
    from src.main import select_resume_checkpoint
    d = _ckpt_dir(tmp_path)
    best = _write(d, f"{RUN}_best.pth", 5)

    assert select_resume_checkpoint(d, RUN) == best


def test_select_returns_none_when_all_corrupt(tmp_path, capsys):
    from src.main import select_resume_checkpoint
    d = _ckpt_dir(tmp_path)
    _write(d, f"{RUN}_last.pth", 9, corrupt=True)
    _write(d, f"{RUN}_best.pth", 5, corrupt=True)

    assert select_resume_checkpoint(d, RUN) is None
    assert "unreadable" in capsys.readouterr().out


def test_select_returns_none_on_retrain(tmp_path):
    from src.main import select_resume_checkpoint
    d = _ckpt_dir(tmp_path)
    _write(d, f"{RUN}_last.pth", 9)

    assert select_resume_checkpoint(d, RUN, retrain=True) is None


def test_select_returns_none_for_missing_dir(tmp_path):
    from src.main import select_resume_checkpoint
    assert select_resume_checkpoint(tmp_path / "nope", RUN) is None


def test_select_ignores_other_runs_checkpoints(tmp_path):
    from src.main import select_resume_checkpoint
    d = _ckpt_dir(tmp_path)
    _write(d, "run_OTHER_last.pth", 9)

    assert select_resume_checkpoint(d, RUN) is None
