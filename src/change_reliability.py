"""How much of the observed tract-level change is signal, on each side?

The change figures compare an ACS change Δy (between two 5-year vintages) with a
predicted change Δp (between flight averages inside the two survey windows).
Both are measured with error, so their observed correlation r understates the
correlation between the true changes. This module measures the two errors
separately instead of assuming one of them (e.g. "stable tracts are pure
noise"):

* ACS sampling noise — Monte Carlo: perturb each tract's label in each vintage
  by its replicate SE, push the perturbed labels through the same
  rank -> GB2 -> clip -> Δlog map as the figure (``to_dlog``), and take the
  per-tract variance across draws. Non-overlapping survey windows make the two
  vintages' errors independent.
* Prediction noise — split halves: the flights of each window are divided into
  two disjoint halves, giving two independent measurements Δp_A, Δp_B of the
  same change. Flight-specific errors are uncorrelated between them, so
  cov(Δp_A, Δp_B) estimates the signal variance of Δp. Averaged over every
  distinct split.

Classical measurement-error model (errors mean-zero, independent of the true
changes and of each other):

    rel_y = 1 - mean(noise_var_y) / var(Δy)        reliability of Δy = R^2 ceiling
    rel_p = cov(Δp_A, Δp_B) / var(Δp)              reliability of Δp
    r*    = r / sqrt(rel_y * rel_p)                correlation of the true changes

Noise is mean-zero, so it lowers correlations but cannot explain a mean shift
(e.g. the construction overshoot); that is bias and is reported elsewhere.

``to_dlog(pre, post) -> Δlog`` is supplied by the caller so the module does not
depend on how the figure maps scores to dollars; it must rank across all the
tracts it is given, which is why every function here works on the full sample
and groups are sliced afterwards.
"""

from __future__ import annotations

from itertools import product
from typing import Callable, Sequence

import numpy as np
import pandas as pd

ToDlog = Callable[[np.ndarray, np.ndarray], np.ndarray]

DEFAULT_N_DRAWS = 200
DEFAULT_N_BOOT = 500


# ─── ACS side ─────────────────────────────────────────────────────────────────

def fill_missing_se(se: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace non-finite or non-positive SEs with the median of the valid ones.
    Returns the filled array and how many entries were filled."""
    se = np.asarray(se, float).copy()
    bad = ~np.isfinite(se) | (se <= 0)
    if bad.all():
        raise ValueError("no valid standard error to fill from")
    se[bad] = np.median(se[~bad])
    return se, int(bad.sum())


def simulate_label_noise(label_pre: np.ndarray, label_post: np.ndarray,
                         se_pre: np.ndarray, se_post: np.ndarray, to_dlog: ToDlog,
                         n_draws: int = DEFAULT_N_DRAWS, seed: int = 0) -> np.ndarray:
    """Per-tract variance of the ACS Δlog induced by sampling error.

    Each draw adds N(0, se^2) independently to every tract's label in each
    vintage and recomputes Δlog through ``to_dlog`` on the whole sample (so the
    re-ranking that one tract's error causes for its neighbours is included).
    Tracts whose label is missing get NaN.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(label_pre, float), np.asarray(label_post, float)
    sa, sb = np.asarray(se_pre, float), np.asarray(se_post, float)
    draws = np.empty((n_draws, len(a)))
    for k in range(n_draws):
        draws[k] = to_dlog(a + sa * rng.standard_normal(len(a)),
                           b + sb * rng.standard_normal(len(b)))
    return np.nanvar(draws, axis=0, ddof=1)


# ─── prediction side ──────────────────────────────────────────────────────────

def _two_way_partitions(flights: Sequence[int]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Every ordered split of ``flights`` into two non-empty halves."""
    flights = tuple(sorted(flights))
    out = []
    for mask in product((0, 1), repeat=len(flights)):
        a = tuple(f for f, m in zip(flights, mask) if m == 0)
        b = tuple(f for f, m in zip(flights, mask) if m == 1)
        if a and b:
            out.append((a, b))
    return out


def split_half_assignments(flights_pre: Sequence[int], flights_post: Sequence[int]
                           ) -> list[tuple[tuple, tuple]]:
    """Distinct splits of the two windows' flights into halves A and B.

    Each split is ((pre_A, post_A), (pre_B, post_B)) with disjoint flights and
    at least one flight per window per half; A/B swaps are dropped (they give
    the same covariance). Needs >= 2 flights in each window.
    """
    if len(flights_pre) < 2 or len(flights_post) < 2:
        raise ValueError("split halves need at least two flights in each window: "
                         f"{list(flights_pre)} / {list(flights_post)}")
    seen, out = set(), []
    for (pa, pb), (qa, qb) in product(_two_way_partitions(flights_pre),
                                      _two_way_partitions(flights_post)):
        key = frozenset({(pa, qa), (pb, qb)})
        if key not in seen:
            seen.add(key)
            out.append(((pa, qa), (pb, qb)))
    return out


def window_mean(preds_wide: pd.DataFrame, flights: Sequence[int]) -> np.ndarray:
    """Per-tract mean over ``flights`` (columns of ``preds_wide``); NaN if the
    tract has none of them."""
    cols = [f for f in flights if f in preds_wide.columns]
    if not cols:
        return np.full(len(preds_wide), np.nan)
    return preds_wide[cols].mean(axis=1, skipna=True).to_numpy(float)


def split_half_deltas(preds_wide: pd.DataFrame, splits, to_dlog: ToDlog
                      ) -> list[tuple[np.ndarray, np.ndarray]]:
    """(Δp_A, Δp_B) per split, each half mapped through ``to_dlog`` on the
    whole sample, rows aligned with ``preds_wide``."""
    out = []
    for (pa, qa), (pb, qb) in splits:
        da = to_dlog(window_mean(preds_wide, pa), window_mean(preds_wide, qa))
        db = to_dlog(window_mean(preds_wide, pb), window_mean(preds_wide, qb))
        out.append((np.asarray(da, float), np.asarray(db, float)))
    return out


# ─── statistics ───────────────────────────────────────────────────────────────

def disattenuated_corr(r: float, rel_y: float, rel_p: float) -> float:
    """r / sqrt(rel_y * rel_p); NaN when a reliability is not positive."""
    if not (np.isfinite(r) and rel_y > 0 and rel_p > 0):
        return np.nan
    return float(r / np.sqrt(rel_y * rel_p))


def _cov(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.cov(a, b, ddof=1)[0, 1])


def reliability_stats(dy: np.ndarray, dp: np.ndarray, noise_var_y: np.ndarray,
                      halves: np.ndarray) -> dict:
    """Measurement-error decomposition for one group.

    ``halves`` is (n_splits, 2, n): Δp_A and Δp_B per split. Rows with any
    non-finite input are dropped first so every quantity uses the same tracts.
    With no split (n_splits = 0, e.g. a window with one flight) the prediction
    side is not identified: rel_p, r* and the Δp signal/noise split are NaN.
    """
    dy, dp, nv = (np.asarray(v, float) for v in (dy, dp, noise_var_y))
    halves = np.asarray(halves, float)
    ok = np.isfinite(dy) & np.isfinite(dp) & np.isfinite(nv) & np.isfinite(halves).all(axis=(0, 1))
    n = int(ok.sum())
    row = {"n": n}
    if n < 5:
        return row
    dy, dp, nv, halves = dy[ok], dp[ok], nv[ok], halves[:, :, ok]
    var_y, var_p = float(np.var(dy, ddof=1)), float(np.var(dp, ddof=1))
    noise_y = float(nv.mean())
    signal_p = float(np.mean([_cov(h[0], h[1]) for h in halves])) if len(halves) else np.nan
    cov_yp = _cov(dy, dp)
    r = cov_yp / np.sqrt(var_y * var_p) if var_y > 0 and var_p > 0 else np.nan
    rel_y = 1.0 - noise_y / var_y if var_y > 0 else np.nan
    rel_p = signal_p / var_p if var_p > 0 and np.isfinite(signal_p) else np.nan
    r_star = disattenuated_corr(r, rel_y, rel_p)
    row.update({
        "mean_dy": float(dy.mean()), "mean_dp": float(dp.mean()),
        "var_dy": var_y, "noise_var_dy": noise_y, "signal_var_dy": var_y - noise_y,
        "var_dp": var_p, "signal_var_dp": signal_p, "noise_var_dp": var_p - signal_p,
        "sd_noise_dy": float(np.sqrt(max(noise_y, 0.0))),
        "sd_noise_dp": (float(np.sqrt(max(var_p - signal_p, 0.0)))
                        if np.isfinite(signal_p) else np.nan),
        "split_half_corr": (float(np.mean([np.corrcoef(h[0], h[1])[0, 1] for h in halves]))
                            if len(halves) else np.nan),
        "rel_y": rel_y, "rel_p": rel_p,
        "r": float(r), "r2_ols": float(r ** 2),
        "r2_ceiling": rel_y,
        "r2_share_of_ceiling": float(r ** 2 / rel_y) if rel_y > 0 else np.nan,
        "r_disattenuated": r_star,
        "slope_ols": cov_yp / var_p if var_p > 0 else np.nan,
        "slope_disattenuated": cov_yp / signal_p if signal_p > 0 else np.nan,
    })
    return row


_CI_KEYS = ("rel_y", "rel_p", "r", "r_disattenuated", "r2_share_of_ceiling",
            "signal_var_dy", "signal_var_dp")


def bootstrap_reliability(dy, dp, noise_var_y, halves, n_boot: int = DEFAULT_N_BOOT,
                          seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile CIs (tract bootstrap) for the main reliability quantities.
    Per-tract inputs are held fixed; only the tract sample is resampled."""
    dy, dp, nv = (np.asarray(v, float) for v in (dy, dp, noise_var_y))
    halves = np.asarray(halves, float)
    ok = np.isfinite(dy) & np.isfinite(dp) & np.isfinite(nv) & np.isfinite(halves).all(axis=(0, 1))
    idx_all = np.flatnonzero(ok)
    out = {}
    if len(idx_all) < 10 or n_boot <= 0:
        return out
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in _CI_KEYS}
    for _ in range(n_boot):
        i = rng.choice(idx_all, size=len(idx_all), replace=True)
        s = reliability_stats(dy[i], dp[i], nv[i], halves[:, :, i])
        for k in _CI_KEYS:
            draws[k].append(s.get(k, np.nan))
    for k, v in draws.items():
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        lo, hi = (np.quantile(v, [alpha / 2, 1 - alpha / 2]) if len(v) else (np.nan, np.nan))
        out[f"{k}_ci_lo"], out[f"{k}_ci_hi"] = float(lo), float(hi)
    return out
