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
from dataclasses import dataclass
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
            # Snapshot alpha/cap under the lock together with the EMAs: a
            # concurrent set_alpha/set_cap (the ramp, applied on the training
            # thread) must not land between the early-return test and the
            # weight computation, or a shard would mix two schedule points.
            alpha, cap = self.alpha, self.cap
            if not self._ema or alpha == 0.0:
                return np.ones(keys.size, dtype=np.float64)
            mean_ema = float(np.mean(list(self._ema.values())))
            ema = np.array([self._ema.get(int(k), mean_ema) for k in keys])
        if mean_ema <= 0.0:
            return np.ones(keys.size, dtype=np.float64)
        ratio = np.minimum(ema / mean_ema, cap)
        return (1.0 - alpha) + alpha * ratio

    # ── schedulable hyperparameters (see MiningSchedule) ─────────────────────
    def set_alpha(self, alpha: float) -> None:
        """Set the hardness mixture weight in place.

        Safe to call while shard generation is in flight: ``weights_for``
        snapshots alpha under the same lock, so a shard sees either the old or
        the new value, never a mix. The EMAs are hyperparameter-free, so
        changing alpha mid-run needs no registry reset — only the sampling bias
        moves, not the learned hardness.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        with self._lock:
            self.alpha = float(alpha)

    def set_cap(self, cap: float) -> None:
        """Set the hardness-ratio cap in place (see :meth:`set_alpha`)."""
        if cap < 1.0:
            raise ValueError(f"cap must be >= 1, got {cap}")
        with self._lock:
            self.cap = float(cap)

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


@dataclass(frozen=True)
class MiningSchedule:
    """Warmup-then-mine ramp for the registry's ``alpha`` (and ``cap``).

    Why ramp instead of mining from epoch 1. Two things make early mining both
    useless and mildly harmful here:

    * **The estimate isn't ready.** A tract's weight is its EMA relative to the
      registry mean, and unseen tracts fall back to that mean (weight 1.0
      exactly). Until the registry has covered most of the tract universe, a
      large alpha mostly amplifies which tracts happened to be drawn first.
    * **The ranking loss needs the label range.** Low-|lambda| tracts are the
      ones the model already orders correctly, which in a wealth ranking skews
      toward the distribution's extremes — the anchors that give in-batch pairs
      their spread. Starving them early compresses the within-batch label range
      and degrades exactly the full-distribution within-city Spearman that
      selects checkpoints. This is the ranking-objective analogue of the
      semi-hard mining lesson (Schroff et al. 2015).

    Once coverage is broad, concentrating the shard budget on high-gradient
    tracts is the variance-optimal choice (Katharopoulos & Fleuret 2018), so
    alpha ramps up linearly between ``start_epoch`` and ``end_epoch``.

    ``cap`` ramps DOWN over the same window on purpose. The cap only binds when
    alpha is large enough for the hardness ratio to dominate the weight, so the
    irreducibly-hard / label-noise guard (Mindermann et al. 2022) is inert at
    low alpha and has to tighten as alpha grows or it never fires at all.

    Epochs are **1-indexed**, matching the ``Epoch [n/N]`` training log. The
    schedule is a pure function of the epoch and holds no state, so a resumed
    run lands on the correct ramp point with nothing to persist — which is why
    :meth:`GradientHardnessRegistry.load` deliberately keeps the constructor's
    alpha/beta/cap rather than restoring saved ones.

    ``alpha_final == alpha_start`` (the default) reproduces the old fixed-alpha
    behavior exactly, so the schedule is inert unless configured.
    """

    alpha_start: float = 0.3
    alpha_final: float = 0.3
    cap_start: float = 10.0
    cap_final: float = 10.0
    start_epoch: int = 150
    end_epoch: int = 300

    def __post_init__(self):
        for name in ("alpha_start", "alpha_final"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        for name in ("cap_start", "cap_final"):
            v = getattr(self, name)
            if v < 1.0:
                raise ValueError(f"{name} must be >= 1, got {v}")
        if self.start_epoch < 1:
            raise ValueError(f"start_epoch must be >= 1, got {self.start_epoch}")
        if self.end_epoch < self.start_epoch:
            raise ValueError(
                f"end_epoch ({self.end_epoch}) must be >= start_epoch "
                f"({self.start_epoch})"
            )

    @property
    def is_active(self) -> bool:
        """True when the schedule actually moves something."""
        return (self.alpha_final != self.alpha_start
                or self.cap_final != self.cap_start)

    def progress(self, epoch: int) -> float:
        """Ramp fraction in [0, 1] for a 1-indexed ``epoch``.

        ``end_epoch == start_epoch`` is a clean step at that epoch (the
        ``>= end_epoch`` test is checked first).
        """
        if epoch >= self.end_epoch:
            return 1.0
        if epoch <= self.start_epoch:
            return 0.0
        span = self.end_epoch - self.start_epoch
        return (epoch - self.start_epoch) / span

    def alpha_at(self, epoch: int) -> float:
        t = self.progress(epoch)
        return self.alpha_start + t * (self.alpha_final - self.alpha_start)

    def cap_at(self, epoch: int) -> float:
        t = self.progress(epoch)
        return self.cap_start + t * (self.cap_final - self.cap_start)

    def apply(self, registry, epoch: int) -> dict:
        """Push this epoch's (alpha, cap) onto ``registry``; return W&B diagnostics.

        Call once at the top of each epoch, before any shard for that epoch is
        drawn. Idempotent — re-applying the same epoch is a no-op.
        """
        alpha, cap = self.alpha_at(epoch), self.cap_at(epoch)
        registry.set_alpha(alpha)
        registry.set_cap(cap)
        return {
            "mining/alpha": alpha,
            "mining/cap": cap,
            "mining/ramp_progress": self.progress(epoch),
        }
