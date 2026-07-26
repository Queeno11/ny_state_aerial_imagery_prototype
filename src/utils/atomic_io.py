"""Crash-safe writes for checkpoints and other artifacts.

A ``torch.save`` straight to its final path is not crash-safe: the process is
holding the only copy open, truncated to zero, for as long as the write takes
(~670 MB / epoch for a full training checkpoint, on /mnt/c that is seconds).
A kill in that window -- OOM reaper, Ctrl-C, reboot, full disk -- leaves a
full-size file whose zip central directory was never written, and the next
resume dies with::

    RuntimeError: PytorchStreamReader failed reading zip archive:
                  failed finding central directory

The fix is the same one the NAIP shard cache and the STAC disk cache already
use: write to a sibling ``.tmp``, then ``os.replace`` onto the final name.
``os.replace`` is atomic within a filesystem, so a reader sees either the old
complete file or the new complete file, never a half-written one.

``fsync`` before the rename is what extends that guarantee from "process died"
to "machine died": without it the rename can reach disk before the data does.
It is best-effort here -- drvfs (/mnt/c) does not always honour it -- and
failure to fsync never fails the save, since atomicity does not depend on it.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Callable

import torch

__all__ = [
    "atomic_torch_save",
    "atomic_save_dir",
    "atomic_write_text",
    "is_valid_torch_checkpoint",
    "previous_version_path",
]


# --------------------------------------------------------------------------- #
# Internals                                                                    #
# --------------------------------------------------------------------------- #

def _fsync_file(fh) -> None:
    """Flush a file object all the way to disk. Best-effort (see module docstring)."""
    try:
        fh.flush()
        os.fsync(fh.fileno())
    except (OSError, ValueError):
        pass


def _fsync_dir(path: Path) -> None:
    """Persist a directory entry so a completed rename survives power loss.

    Not available on Windows and a no-op on some mounts; never fatal.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        pass


def previous_version_path(path) -> Path:
    """Sibling path holding the last known-good copy of ``path``.

    ``run_last.pth`` -> ``run_last.prev.pth`` (suffix preserved so the file is
    still recognisable as a checkpoint by tooling and by a human listing a dir).
    """
    path = Path(path)
    return path.with_name(f"{path.stem}.prev{path.suffix}")


# --------------------------------------------------------------------------- #
# Writers                                                                      #
# --------------------------------------------------------------------------- #

def atomic_torch_save(obj, path, *, keep_previous: bool = False) -> Path:
    """``torch.save(obj, path)`` that can never leave a truncated file at ``path``.

    Args:
        obj: anything ``torch.save`` accepts.
        path: final destination.
        keep_previous: before publishing the new file, rotate the current one to
            :func:`previous_version_path`. Costs one extra file's worth of disk
            and buys a one-epoch-old fallback if the newest checkpoint is ever
            unreadable for a reason atomicity does not cover (bad block, a kill
            during the rename window, an interrupted copy off the box).

    Returns:
        The final path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # PID/TID-free name is fine: one writer per checkpoint path by construction
    # (the shard cache, which does have concurrent writers, salts its own tmp).
    tmp = path.with_name(f"{path.name}.tmp")

    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        _fsync_file(fh)

    if keep_previous and path.exists():
        # Rename-only, so the window in which neither `path` nor its replacement
        # exists is microseconds; the .prev copy covers a crash even there.
        os.replace(path, previous_version_path(path))

    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return path


def atomic_save_dir(save_fn: Callable[[Path], None], path) -> Path:
    """Run ``save_fn(staging_dir)`` and publish the result at ``path`` in one step.

    For writers that emit a *directory* rather than a file -- notably
    ``PeftModel.save_pretrained``, which writes ``adapter_config.json`` and
    ``adapter_model.safetensors`` separately and so can leave a config with no
    weights (or weights with no config) if interrupted between them.

    The old directory is moved aside and only removed once the new one is in
    place, so an interrupted publish leaves the previous adapter recoverable at
    ``<name>.old`` instead of leaving nothing at all.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    old = path.with_name(f"{path.name}.old")

    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    save_fn(tmp)

    for child in sorted(tmp.rglob("*")):
        if child.is_file():
            with open(child, "rb") as fh:
                _fsync_file(fh)
    _fsync_dir(tmp)

    if old.exists():
        shutil.rmtree(old)
    if path.exists():
        os.replace(path, old)
    os.replace(tmp, path)
    _fsync_dir(path.parent)

    if old.exists():
        shutil.rmtree(old)
    return path


def atomic_write_text(path, text: str, encoding: str = "utf-8") -> Path:
    """Text-file counterpart of :func:`atomic_torch_save` (params logs, manifests)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with open(tmp, "w", encoding=encoding) as fh:
        fh.write(text)
        _fsync_file(fh)
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return path


# --------------------------------------------------------------------------- #
# Reader-side integrity                                                        #
# --------------------------------------------------------------------------- #

def is_valid_torch_checkpoint(path) -> bool:
    """Cheap structural check: does ``path`` look like a complete torch save?

    Reads only the zip end-of-central-directory record at the tail, so this is
    O(1) rather than O(670 MB) -- and the EOCD is precisely what a truncated
    write is missing, which makes it an exact test for the failure this module
    prevents.

    torch has written the zip format by default since 1.6, so a *legacy*
    (plain-pickle) checkpoint would be reported invalid here. That is only ever
    a demotion: callers fall through to the next candidate and the load-time
    error names the file, so a misjudged legacy file is visible rather than
    silently dropped.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        return zipfile.is_zipfile(path)
    except OSError:
        return False
