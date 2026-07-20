"""Dump the first N training batches to disk for visual QA of the ingested data.

On a FRESH run (no checkpoint being resumed), ``run()`` in ``src/main.py`` creates a
:class:`BatchImageDumper` and hands it to the ``HybridBatchSampler``, which calls
:meth:`BatchImageDumper.dump_batch` for the first ``max_batches`` batches it yields.
Images are saved exactly as they sit in the training shards — uint8 ``(C, H, W)``
padded tiles, BEFORE RandomCrop/flip augmentation and BEFORE float scaling /
ImageNet normalization — i.e. the ingested data itself.

Files written to ``out_dir`` (default ``results/<savename>/debug_batches``):

- ``dump_info.json``          — run params relevant for interpretation (indicator, years, ...).
- ``buildings_ref.parquet``   — ``building_id -> GEOID / cbsa_code / cbsa_title`` lookup.
  Needed because shard ``geoids`` are per-process salted hashes (only usable for
  same-tract grouping inside the loss), so the real tract GEOID is NOT recoverable
  from the shard tensors.
- ``batch_000.pt`` ...        — one payload per dumped batch: raw images + all shard
  metadata tensors, plus the batch structure (``cs_size``, ``anchor_year``,
  ``anchor_cbsa``): positions ``[0, cs_size)`` are the cross-sectional core,
  positions ``[cs_size, ...)`` are the temporal twins.

Visualize with ``src/notebooks/visualize_debug_batches.ipynb``.

Disk cost is modest (~30 batches x ~40 images x 4x244x244 uint8 ~= 250 MB) and the
dump directory is wiped at the start of every fresh run, so it never accumulates.
The dumper is deliberately fail-safe: any exception disables it with a warning
instead of killing the training run.
"""

import json
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

DEFAULT_MAX_BATCHES = 30


def _norm_cbsa(value):
    """CBSA code as a canonical string ('35620'), tolerating int/float/str input."""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


class BatchImageDumper:
    """Persists the first ``max_batches`` hybrid training batches for visual inspection."""

    REF_COLUMNS = ["building_id", "GEOID", "cbsa_code"]

    def __init__(self, out_dir, max_batches=DEFAULT_MAX_BATCHES):
        self.out_dir = Path(out_dir)
        self.max_batches = max_batches
        self._n_dumped = 0
        self._failed = False
        # Wipe stale dumps from a previous fresh run with the same savename so the
        # notebook never mixes batches from two different runs.
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def finished(self):
        """True once the budget is exhausted (or the dumper disabled itself)."""
        return self._failed or self._n_dumped >= self.max_batches

    def write_run_info(self, params, savename):
        """Persist the params needed to interpret the dump (indicator, years, ...)."""
        info = {
            "savename": savename,
            "indicator": params.get("indicator"),
            "years": params.get("years"),
            "image_size": params.get("image_size"),
            "nbands": params.get("nbands"),
            "max_jitter": params.get("max_jitter"),
            "tau_meters": params.get("tau_meters"),
            "subsample_step": params.get("subsample_step"),
            "sat_data": params.get("sat_data"),
            "max_batches": self.max_batches,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.out_dir / "dump_info.json", "w") as f:
            json.dump(info, f, indent=2, default=str)

    def write_building_reference(self, df_train, cbsa_meta=None):
        """Save the ``building_id -> GEOID / cbsa_code [/ cbsa_title]`` lookup table.

        ``cbsa_meta`` is the ``cbsa_brackets.load_city_split()`` frame; when given,
        its ``cbsa_title`` is joined on so the notebook can print city names.
        """
        cols = [c for c in self.REF_COLUMNS if c in df_train.columns]
        if "building_id" not in cols:
            warnings.warn("[BatchImageDumper] df_train has no building_id column — skipping reference table.")
            return
        ref = df_train[cols].drop_duplicates("building_id").copy()
        if "GEOID" in ref.columns:
            ref["GEOID"] = ref["GEOID"].astype(str)
        if (
            cbsa_meta is not None
            and "cbsa_code" in ref.columns
            and {"cbsa_code", "cbsa_title"}.issubset(cbsa_meta.columns)
        ):
            ref["cbsa_code"] = ref["cbsa_code"].map(_norm_cbsa)
            titles = cbsa_meta[["cbsa_code", "cbsa_title"]].copy()
            titles["cbsa_code"] = titles["cbsa_code"].map(_norm_cbsa)
            titles = titles.drop_duplicates("cbsa_code")
            ref = ref.merge(titles, on="cbsa_code", how="left")
        try:
            ref.to_parquet(self.out_dir / "buildings_ref.parquet", index=False)
        except Exception as exc:  # pyarrow missing — degrade to CSV rather than crash
            warnings.warn(f"[BatchImageDumper] parquet write failed ({exc!r}); writing CSV instead.")
            ref.to_csv(self.out_dir / "buildings_ref.csv", index=False)

    def dump_batch(self, dataset, batch_indices, cs_size, anchor_year, anchor_cbsa):
        """Save one hybrid batch (raw shard images + metadata) as ``batch_XXX.pt``.

        ``dataset`` is the live ``InBatchRankingDataset``; ``batch_indices`` are the
        indices the sampler is about to yield (CS core first, then temporal twins).
        Never raises: a failure disables the dumper so training is unaffected.
        """
        if self.finished:
            return
        try:
            idx = torch.as_tensor(batch_indices, dtype=torch.long)
            payload = {
                "batch_index": self._n_dumped,
                "cs_size": int(cs_size),
                "anchor_year": int(anchor_year),
                "anchor_cbsa": int(anchor_cbsa),
                "images": dataset.images[idx].clone(),        # uint8 raw shard tiles (pre-augmentation)
                "scores": dataset.scores[idx].clone(),        # Rel_Score labels (per-CBSA z-scores)
                "geoid_hashes": dataset.geoids[idx].clone(),  # stable_geoid_hash (crc32) of GEOID
                "years": dataset.years[idx].clone(),
                "building_ids": dataset.building_ids[idx].clone(),
                "structural_change": dataset.structural_change[idx].clone(),
                "metas": dataset.metas[idx].clone(),
                "score_bins": dataset.score_bins[idx].clone(),
                "cbsa_ids": dataset.cbsa_ids[idx].clone(),
            }
            torch.save(payload, self.out_dir / f"batch_{self._n_dumped:03d}.pt")
            self._n_dumped += 1
            if self._n_dumped == self.max_batches:
                print(f"📸 [BatchImageDumper] Dumped {self._n_dumped} batches to {self.out_dir} — done.")
        except Exception as exc:
            self._failed = True
            warnings.warn(f"[BatchImageDumper] dump failed ({exc!r}) — disabling batch dumping.")
