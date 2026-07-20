"""Tract-first shard sampling + gradient-based hard-tract mining.

Why tract-first: the legacy shard generator traverses the *building* universe,
so a tract's chance of entering a batch is proportional to its building count.
That over-represents low-density suburbs (many small structures) and starves
dense urban tracts (few large structures) — in the NYC metro the five boroughs
are ~47% of tracts but only ~13% of buildings (Manhattan 0.17%), so the
intra-city ranking pairs the evaluation needs almost never occur in training.
Labels are tract-level, so building multiplicity adds no label information;
sampling tracts uniformly (≈ population-weighted, since tracts are drawn to
hold roughly constant population) removes the bias at zero statistical cost.

Why gradient mining: RankNet's per-pair gradient magnitude sigma(-delta/m)
already vanishes on confidently-correct pairs, so easy pairs stop contributing;
what limits learning is which tracts enter batches at all. We therefore bias
*tract selection* toward tracts the model currently gets wrong, measured by the
per-sample ranking gradient lambda_i = dL_cross/ds_i — the "lambda" of the
learning-to-rank literature (Burges 2010, "From RankNet to LambdaRank to
LambdaMART", MSR-TR-2010-82). Sampling examples proportionally to their
gradient norm is the variance-optimal importance-sampling scheme for SGD
(Katharopoulos & Fleuret 2018, "Not All Samples Are Created Equal", ICML;
antecedents Needell/Ward/Srebro 2014, Zhao & Zhang 2015); hard-example mining
per se goes back to OHEM (Shrivastava, Gupta & Girshick 2016, CVPR). Two
noise guards, motivated by RHO-loss's "learnable, worth learning, not yet
learnt" criterion (Mindermann et al. 2022, ICML) and by semi-hard mining
(Schroff et al. 2015, FaceNet, CVPR): (a) the loss filters statistically
unreliable pairs (ACS MOE rule) *before* lambdas are harvested, so hardness is
only accumulated where the label sign is trustworthy, and (b) sampling weights
mix a uniform component with a capped hardness ratio, so no tract is starved
and irreducibly-hard tracts cannot monopolize the shard budget.

Wiring (all optional — legacy behavior when disabled):
  * ``InBatchPairwiseRankingLoss`` harvests per-tract |lambda| each step and
    feeds ``GradientHardnessRegistry.update`` (keyed by ``stable_geoid_hash``).
  * ``CyclicCacheManager._worker_generate`` (train, lazy ms_us path) calls
    ``LazyPairTable.materialize_tract_sample`` with the registry's
    ``weights_for`` lookup instead of the cyclic building traversal.
"""

from __future__ import annotations

import json
import threading
import zlib
from pathlib import Path

import numpy as np


def stable_geoid_hash(geoid: str) -> int:
    """Deterministic 31-bit hash of a GEOID string.

    Replaces Python's builtin ``hash(str)`` (salted per process via
    PYTHONHASHSEED) so the SAME tract maps to the SAME id in the loss
    (shard tensors), in the registry, and at shard generation — including
    across resumed runs.
    """
    return zlib.crc32(str(geoid).encode("utf-8")) % (2**31)


def hash_geoids(geoids) -> np.ndarray:
    """Vector of :func:`stable_geoid_hash` for an iterable of GEOID strings."""
    return np.fromiter(
        (stable_geoid_hash(g) for g in geoids), dtype=np.int64, count=len(geoids)
    )


def groupby_mean(keys: np.ndarray, values: np.ndarray):
    """(unique_keys, mean value per key) for 1-D arrays. Pure NumPy."""
    order = np.argsort(keys, kind="stable")
    k, v = keys[order], values[order]
    uniq, start = np.unique(k, return_index=True)
    sums = np.add.reduceat(v, start)
    counts = np.diff(np.append(start, len(k)))
    return uniq, sums / counts


class GradientHardnessRegistry:
    """Per-tract EMA of |lambda_i| driving hard-tract importance sampling.

    Parameters
    ----------
    alpha : mixture weight of the hardness component in the sampling weights,
        ``w = (1 - alpha) + alpha * min(ema / mean_ema, cap)``. ``alpha=0``
        reproduces uniform tract sampling; the uniform component guarantees
        every tract keeps a floor probability (coverage; no starved tracts).
    beta : EMA decay, ``ema <- beta * ema + (1 - beta) * batch_mean``. Decay
        lets stale hardness fade once the model improves on a tract.
    cap : upper bound on the hardness ratio, so irreducibly-hard (label-noise /
        invisible-from-above) tracts cannot monopolize shards — the semi-hard
        lesson of Schroff et al. (2015) / Mindermann et al. (2022).

    Thread-safe: ``update`` runs on the training loop thread while
    ``weights_for`` runs on the background shard-generation thread.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.9, cap: float = 10.0):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0.0 < beta < 1.0:
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        if cap < 1.0:
            raise ValueError(f"cap must be >= 1, got {cap}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.cap = float(cap)
        self._ema: dict[int, float] = {}
        self._lock = threading.Lock()
        self.n_updates = 0

    def __len__(self):
        return len(self._ema)

    def update(self, geoid_hashes, hardness) -> None:
        """Fold one batch's per-sample hardness into the per-tract EMAs.

        Accepts array-likes (torch tensors welcome via ``.cpu().numpy()`` on
        the caller side). Samples of the same tract are averaged first so a
        tract's EMA step is independent of how many of its buildings were in
        the batch.
        """
        keys = np.asarray(geoid_hashes, dtype=np.int64)
        vals = np.asarray(hardness, dtype=np.float64)
        if keys.shape != vals.shape:
            raise ValueError("geoid_hashes and hardness must have equal length")
        if keys.size == 0:
            return
        uniq, means = groupby_mean(keys, vals)
        with self._lock:
            for k, m in zip(uniq.tolist(), means.tolist()):
                prev = self._ema.get(k)
                self._ema[k] = (
                    m if prev is None else self.beta * prev + (1.0 - self.beta) * m
                )
            self.n_updates += 1

    def weights_for(self, geoid_hashes) -> np.ndarray:
        """Sampling weight per tract: ``(1-alpha) + alpha*min(ema/mean, cap)``.

        Tracts the registry has never seen get the mean hardness (ratio 1.0),
        i.e. exactly the uniform weight — cold-start and mining-off are the
        same code path.
        """
        keys = np.asarray(geoid_hashes, dtype=np.int64)
        with self._lock:
            if not self._ema or self.alpha == 0.0:
                return np.ones(keys.size, dtype=np.float64)
            mean_ema = float(np.mean(list(self._ema.values())))
            ema = np.array([self._ema.get(int(k), mean_ema) for k in keys])
        if mean_ema <= 0.0:
            return np.ones(keys.size, dtype=np.float64)
        ratio = np.minimum(ema / mean_ema, self.cap)
        return (1.0 - self.alpha) + self.alpha * ratio

    # ── persistence (registry survives run restarts) ─────────────────────────
    def save(self, path) -> None:
        path = Path(path)
        with self._lock:
            state = {
                "alpha": self.alpha, "beta": self.beta, "cap": self.cap,
                "n_updates": self.n_updates,
                "ema": {str(k): v for k, v in self._ema.items()},
            }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(path)

    def load(self, path) -> bool:
        """Restore EMAs from ``path`` (True on success, False if absent).

        alpha/beta/cap keep their constructor values — hyperparameters belong
        to the run, only the learned hardness state is restored.
        """
        path = Path(path)
        if not path.exists():
            return False
        state = json.loads(path.read_text())
        with self._lock:
            self._ema = {int(k): float(v) for k, v in state["ema"].items()}
            self.n_updates = int(state.get("n_updates", 0))
        return True

    def stats(self) -> dict:
        """Small diagnostics dict for W&B logging."""
        with self._lock:
            if not self._ema:
                return {"mining/registry_tracts": 0, "mining/ema_mean": 0.0,
                        "mining/ema_p90": 0.0}
            vals = np.fromiter(self._ema.values(), dtype=np.float64)
        return {
            "mining/registry_tracts": int(vals.size),
            "mining/ema_mean": float(vals.mean()),
            "mining/ema_p90": float(np.quantile(vals, 0.9)),
        }
