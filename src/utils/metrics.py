# -*- coding: utf-8 -*-
"""Shared validation/evaluation metrics (pure numpy/pandas/scipy).

Canonical home of the paper's headline metrics, used by BOTH the training-time
validation loop (src/main.py) and the post-hoc evaluation script
(src/evaluation.py) so the two can never drift apart.

All functions take the canonical long frame: one row per validated image with
columns ``building_id, year, pred, label, change`` and (optionally) ``cbsa``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def within_city_cells(val_df, min_bld=5):
    """Per-(cbsa, year) Spearman cells — the within-city cross-sectional metric.

    Each cell holds one row per building (a building appears at most once per
    year), so cell Spearmans are free of the repeated-building inflation that
    affects the pooled set-level Spearman. Cells with fewer than ``min_bld``
    buildings, or without label/pred variation, are dropped (they are noise).

    ``n_tracts`` is the number of distinct tracts in the cell — the effective
    sample size, since labels are tract-level: buildings in one tract share a
    label, so extra buildings per tract add rows but no label information.
    Counted from ``GEOID`` when the frame carries it (evaluation.py), else from
    distinct label values (the training-time val frame has no tract column;
    tract z-scores are continuous, so distinct labels ≈ distinct tracts).

    Returns a DataFrame [cbsa, year, n, n_tracts, rho]; empty if no usable cell.
    """
    from scipy.stats import spearmanr

    tract_col = "GEOID" if "GEOID" in val_df.columns else "label"
    rows = []
    for (cbsa, year), g in val_df.groupby(["cbsa", "year"]):
        if len(g) < min_bld or g["pred"].nunique() < 2 or g["label"].nunique() < 2:
            continue
        rho, _ = spearmanr(g["pred"], g["label"])
        if not np.isnan(rho):
            rows.append({"cbsa": cbsa, "year": year, "n": len(g),
                         "n_tracts": int(g[tract_col].nunique()), "rho": float(rho)})
    return pd.DataFrame(rows, columns=["cbsa", "year", "n", "n_tracts", "rho"])


def weighted_within_spearman(cells):
    """Tract-weighted mean of the per-cell Spearmans — the headline number.

    Weighting by ``n_tracts`` (not building count ``n``) keeps dense downtowns
    with many buildings per tract from dominating cells that cover the same
    labeled area with fewer, larger structures.

    Takes the frame returned by :func:`within_city_cells`; returns a dict
    {within_spearman, within_cells, within_n, within_tracts}, empty when no
    usable cell.
    """
    if not len(cells):
        return {}
    return {
        "within_spearman": float(np.average(cells["rho"], weights=cells["n_tracts"])),
        "within_cells": int(len(cells)),
        "within_n": int(cells["n"].sum()),
        "within_tracts": int(cells["n_tracts"].sum()),
    }


DIAGNOSTIC_VAL_SETS = ("val_stability",)


def selection_score(val_set_metrics, mode="within_50_50", temporal_set="val_temporal",
                    diagnostic_sets=DIAGNOSTIC_VAL_SETS):
    """Checkpoint-selection score from the per-set outputs of compute_val_metrics.

    Modes:
      "within_50_50" -- 0.5 * cross-sectional + 0.5 * temporal.
        The cross-sectional side pools every non-temporal set (val_cities plus
        the val_within_* spatial holdouts) by averaging their within_spearman
        weighted by within_tracts. Each set's within_spearman is itself the
        tract-weighted mean of its (cbsa, year) cells, so this equals the
        tract-weighted within-Spearman over the union of their cells -- the
        city holdouts enter with their natural tract weight instead of a fixed
        per-set share. The temporal side is the temporal set's within_spearman.
        A set without within cells falls back to its pooled spearman (weighted
        by its row count n); a side with no usable set drops out and the other
        side carries the score alone.
      "within" -- legacy: unweighted mean of within_spearman across all sets
        (per-set fallback to pooled spearman).
      "pooled" -- legacy: unweighted mean of pooled spearman across all sets.

    ``diagnostic_sets`` names val sets excluded from the score entirely (still
    logged): the temporal-stability pool (val_stability) is a few multi-year
    buildings for MASD / rank-autocorrelation, so its noisy within-Spearman must
    not enter checkpoint selection.

    Returns ``(score, components)``: score is None when no set has a usable
    value; components is a flat {name: value} dict for logging (the per-side
    values under "within_50_50", the per-set values under the legacy modes).
    """
    diag = set(diagnostic_sets or ())
    val_set_metrics = {k: v for k, v in val_set_metrics.items() if k not in diag}
    if mode == "within_50_50":
        cs_vals, cs_wts = [], []
        temporal_val = None
        for name, m in val_set_metrics.items():
            if "within_spearman" in m:
                val, wt = m["within_spearman"], float(m.get("within_tracts", 1))
            elif "spearman" in m:
                val, wt = m["spearman"], float(m.get("n", 1))
            else:
                continue
            if name == temporal_set:
                temporal_val = val
            else:
                cs_vals.append(val)
                cs_wts.append(wt)
        components = {}
        sides = []
        if cs_vals:
            cs_side = float(np.average(cs_vals, weights=cs_wts))
            components["cross_sectional_within"] = cs_side
            sides.append(cs_side)
        if temporal_val is not None:
            components["temporal_within"] = float(temporal_val)
            sides.append(float(temporal_val))
        score = float(np.mean(sides)) if sides else None
        return score, components

    if mode not in ("within", "pooled"):
        raise ValueError(f"Unknown selection mode: {mode!r}")
    key = "within_spearman" if mode == "within" else "spearman"
    components = {}
    for name, m in val_set_metrics.items():
        if key in m:
            components[name] = float(m[key])
        elif "spearman" in m:
            components[name] = float(m["spearman"])
    score = float(np.mean(list(components.values()))) if components else None
    return score, components


def rank_autocorrelation(val_df, min_common=5):
    """Cross-year rank stability of STABLE buildings, per city, n-weighted.

    For each city, picks the year pair with the most stable buildings observed
    in both years (ties -> widest span, mirroring the paper's 2016<->2024 pair)
    and computes Spearman(rank_t, rank_t') for predictions AND for labels over
    those common buildings. Both are comparisons of two within-year rankings —
    never of score levels across years — so per-year z-scoring of the labels
    does not affect them. The label autocorrelation is the genuine-reshuffling
    benchmark: the pred-minus-label gap is model-induced instability.

    Returns {} when no city has >= min_common common stable buildings.
    """
    from scipy.stats import spearmanr

    stable = val_df[val_df["change"] == 0]
    per_city = []
    for cbsa, g in stable.groupby("cbsa"):
        piv_p = g.pivot_table(index="building_id", columns="year", values="pred")
        piv_l = g.pivot_table(index="building_id", columns="year", values="label")
        years = sorted(piv_p.columns)
        best = None  # (n_common, span, t1, t2)
        for i in range(len(years)):
            for j in range(i + 1, len(years)):
                t1, t2 = years[i], years[j]
                n_common = int(piv_p[[t1, t2]].dropna().shape[0])
                key = (n_common, t2 - t1)
                if n_common >= min_common and (best is None or key > best[:2]):
                    best = (n_common, t2 - t1, t1, t2)
        if best is None:
            continue
        _, _, t1, t2 = best
        pp = piv_p[[t1, t2]].dropna()
        ll = piv_l.loc[pp.index, [t1, t2]]
        rho_p, _ = spearmanr(pp[t1], pp[t2])
        rho_l, _ = spearmanr(ll[t1], ll[t2])
        if np.isnan(rho_p) or np.isnan(rho_l):
            continue
        per_city.append({"n": len(pp), "rho_p": float(rho_p), "rho_l": float(rho_l)})
    if not per_city:
        return {}
    ns = np.array([c["n"] for c in per_city], dtype=float)
    return {
        "rank_autocorr_pred": float(np.average([c["rho_p"] for c in per_city], weights=ns)),
        "rank_autocorr_label": float(np.average([c["rho_l"] for c in per_city], weights=ns)),
        "rank_autocorr_n": int(ns.sum()),
    }


def masd_by_change(df):
    """Stable/changed MASD + directional accuracy over all ordered year pairs.

    Returns a dict with (whichever have data) stable_masd, changed_masd,
    changed_da — mean |pred displacement| of stable vs changed buildings and
    the sign-agreement rate of changed-building displacements with the labels.
    """
    out = {}
    counts = df["building_id"].value_counts()
    multi = counts[counts >= 2].index
    if not len(multi):
        return out
    sub = df[df["building_id"].isin(multi)].sort_values(["building_id", "year"])
    stable_disps, changed_disps = [], []
    da_correct, da_total = 0, 0
    for _, grp in sub.groupby("building_id"):
        preds_arr = grp["pred"].values
        labels_arr = grp["label"].values
        is_changed = grp["change"].iloc[0]
        n = len(preds_arr)
        for i in range(n):
            for j in range(i + 1, n):
                pred_d = preds_arr[j] - preds_arr[i]
                label_d = labels_arr[j] - labels_arr[i]
                if is_changed == 0:
                    stable_disps.append(abs(pred_d))
                else:
                    changed_disps.append(abs(pred_d))
                    if label_d != 0:  # skip tied labels
                        da_total += 1
                        if (pred_d > 0) == (label_d > 0):
                            da_correct += 1
    if stable_disps:
        out["stable_masd"] = float(np.mean(stable_disps))
    if changed_disps:
        out["changed_masd"] = float(np.mean(changed_disps))
    if da_total > 0:
        out["changed_da"] = da_correct / da_total
    return out


def compute_val_metrics(val_df, cbsa_meta=None, top_cities=None, min_city_n=50):
    """Validation metrics for one val set (pure pandas/numpy; unit-testable).

    Args:
        val_df: DataFrame with columns building_id, year, pred, label, change,
            and (optionally) cbsa — one row per validated image.
        cbsa_meta: content of cbsa_splits.feather (cbsa_code, bracket,
            population, ...) or None -> base metrics only (pre-US shards).
        top_cities: cbsa codes to report per-city metrics for (#31 top-10),
            wherever they have >= min_city_n predictions in this val set.
        min_city_n: noise guard for the per-city breakdown.

    Returns a flat dict: base keys (mse, spearman, stable_masd, changed_masd,
    changed_da, pred_mean, pred_std), the goal-aligned metrics
    (within_spearman/within_cells/within_n, rank_autocorr_pred/label/n,
    masd_ratio, city_offset_sd — see within_city_cells/rank_autocorrelation),
    plus 'bracket/{name}/{metric}' and 'city/{cbsa}/{metric}' breakdowns.
    Keys with no data are omitted. NOTE: 'spearman' and 'n' pool building x year
    image rows (repeated buildings inflate them); 'within_spearman' is the
    honest per-cell quantity (tract-weighted, see weighted_within_spearman) and
    drives checkpointing when params["selection_metric"] == "within".
    """
    from scipy.stats import spearmanr

    # Under autocast the val loop hands us float16 preds; pandas' unstack
    # (pivot_table in rank_autocorrelation) has no float16 kernel and raises
    # "TypeError: No matching signature found". Normalize once, up front.
    if not val_df.empty:
        val_df = val_df.copy()
        for _c in ("pred", "label"):
            if _c in val_df.columns:
                val_df[_c] = val_df[_c].astype(np.float64)

    def _base(df):
        out = {}
        if df.empty:
            return out
        out["mse"] = float(np.mean((df["pred"] - df["label"]) ** 2))
        if df["pred"].nunique() > 1 and df["label"].nunique() > 1:
            rho, _ = spearmanr(df["pred"], df["label"])
            if not np.isnan(rho):
                out["spearman"] = float(rho)
        # Predicted-score moments: the cross-city latent-drift detector (#31) —
        # per-CBSA z-scored labels mean every city should predict ~N(0, 1).
        out["pred_mean"] = float(df["pred"].mean())
        out["pred_std"] = float(df["pred"].std()) if len(df) > 1 else 0.0
        out.update(masd_by_change(df))
        return out

    metrics = _base(val_df)

    # ── Headline within-city metrics + diagnostics (goal-aligned, additive) ──
    # within_spearman: tract-weighted mean of per-(city, year) cell Spearmans —
    #   one observation per building per cell, so no repeated-building
    #   inflation, and weights reflect label information (tracts), not density.
    # rank_autocorr_pred/label: cross-year rank stability of stable buildings
    #   (label variant = genuine-reshuffling benchmark). Diagnostics only.
    # masd_ratio: changed vs stable displacement (>1 = moves where reality moved).
    # city_offset_sd: dispersion of per-city prediction means (~0 under the
    #   per-CBSA z labels) — the cross-city latent-drift guardrail.
    if {"cbsa", "year"}.issubset(val_df.columns) and not val_df.empty:
        cells = within_city_cells(val_df)
        metrics.update(weighted_within_spearman(cells))
        metrics.update(rank_autocorrelation(val_df))
        if metrics.get("stable_masd", 0) > 0 and "changed_masd" in metrics:
            metrics["masd_ratio"] = metrics["changed_masd"] / metrics["stable_masd"]
        offsets = val_df.groupby("cbsa")["pred"].agg(["mean", "size"])
        offsets = offsets[offsets["size"] >= 5]
        if len(offsets) >= 2:
            metrics["city_offset_sd"] = float(offsets["mean"].std(ddof=0))

    if cbsa_meta is not None and "cbsa" in val_df.columns:
        bracket_of = {}
        for code, bracket in zip(cbsa_meta["cbsa_code"], cbsa_meta["bracket"]):
            try:
                bracket_of[int(code)] = bracket
            except (TypeError, ValueError):
                continue
        df = val_df.copy()
        df["cbsa"] = df["cbsa"].astype(int)
        df["bracket"] = df["cbsa"].map(bracket_of)
        for bracket, grp in df.groupby("bracket"):
            sub = _base(grp)
            sub["n"] = len(grp)
            bcells = within_city_cells(grp)
            if len(bcells):
                sub["within_spearman"] = float(np.average(bcells["rho"], weights=bcells["n_tracts"]))
            metrics.update({f"bracket/{bracket}/{k}": v for k, v in sub.items()})
        if top_cities:
            top_ints = set()
            for c in top_cities:
                try:
                    top_ints.add(int(c))
                except (TypeError, ValueError):
                    continue
            for cbsa, grp in df.groupby("cbsa"):
                if cbsa not in top_ints or len(grp) < min_city_n:
                    continue
                sub = _base(grp)
                sub["n"] = len(grp)
                ccells = within_city_cells(grp)
                if len(ccells):
                    sub["within_spearman"] = float(np.average(ccells["rho"], weights=ccells["n_tracts"]))
                metrics.update({f"city/{cbsa}/{k}": v for k, v in sub.items()})

    return metrics
