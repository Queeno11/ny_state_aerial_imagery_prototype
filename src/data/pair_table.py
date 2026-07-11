"""Lazy building×year pair table for the US-scale (ms_us) training path.

The legacy pipeline materialized one flat row per (building, year) pair. At US
scale that is 71.8M buildings × 8 years ≈ 575M rows (150+ GB) — it OOM-killed a
32 GB box before training ever started. Since the Microsoft footprint universe
is static (every building "exists" in every panel year) and labels only vary at
the (tract, year) level, the cross product never needs to be materialized:

  * ``buildings``  — one slim row per building (category strings, float32).
  * ``labels``     — one row per (tract GEOID, year) with the indicator label.
  * pairs          — implicit: pair k ↔ (building k // n_years, year k % n_years).

``LazyPairTable`` exposes the pieces the training pipeline actually consumes:
``__len__`` plus ``materialize(start, stop)`` for the cyclic shard generator
(which only ever reads sequential ~20k-row slices), per-year materialization
for prediction, and per-tract building sampling for the validation sets.
Traversal is building-major / year-minor, so a building's temporal twins are
adjacent — exactly the layout the legacy categorical sort produced.

Holdout semantics: train tables receive ``holdout_map`` (cbsa → holdout year)
and set ``Rel_Score = NaN`` on pairs falling on their city's holdout year. The
shard generator already drops NaN-labeled rows before fetching imagery, so
those pairs never enter training — equivalent to the legacy row-level
``val_temporal`` exclusion. The fetch-time effective-year guards in
``CyclicCacheManager`` remain the backstop for NAIP flight-year substitution.
"""

import math

import numpy as np
import pandas as pd

import src.geo_utils as geo_utils

# Column order of materialized slices — matches the legacy flat table
# (``relevant_columns`` in build_dataset.create_train_test_dataframes) so every
# downstream consumer sees an identical schema.
FLAT_COLUMNS = [
    "building_id", "GEOID", "cbsa_code", "year", "type",
    "Rel_Score", "Valid_Structural_Change", "score_bin",
    "dataset", "bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy",
    "row_start", "row_stop", "col_start", "col_stop", "dist_to_center",
    "centroid_x", "centroid_y",
]

BUILDING_COLUMNS = [
    "building_id", "GEOID", "cbsa_code",
    "centroid_x", "centroid_y", "dist_to_center",
]

LABEL_COLUMNS = ["GEOID", "year", "Rel_Score", "Valid_Structural_Change", "score_bin"]


def weighted_qcut(values, weights, q=5):
    """Quantile-bin ``values`` with integer ``weights`` — exact equivalent of
    ``pd.qcut(np.repeat(values, weights), q, labels=False, duplicates="drop")``
    evaluated per value, without materializing the expanded array.

    Used for ``score_bin``: the legacy pipeline ran ``pd.qcut`` over
    building-year rows, i.e. quantiles of tract scores weighted by the tract's
    building count. Returns int8 codes; NaN values get -1 (they are dropped by
    the shard generator's NaN-label guard before ever reaching the loss).
    """
    values = np.asarray(values, dtype="float64")
    weights = np.asarray(weights, dtype="int64")
    if (weights <= 0).any():
        raise ValueError("weighted_qcut: weights must be positive integers")
    out = np.full(len(values), -1, dtype="int8")
    mask = ~np.isnan(values)
    v, w = values[mask], weights[mask]
    if len(v) == 0:
        return out

    order = np.argsort(v, kind="mergesort")
    vs, ws = v[order], w[order]
    cum = np.cumsum(ws)          # 1-based end position of each value in the expanded array
    n_expanded = int(cum[-1])

    # np.percentile(expanded, p, linear): index pos = p * (N-1); expanded[i] is
    # the value whose cumulative count first reaches i+1.
    edges = []
    for p in np.linspace(0.0, 1.0, q + 1):
        pos = p * (n_expanded - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        v_lo = vs[np.searchsorted(cum, lo + 1, side="left")]
        v_hi = vs[np.searchsorted(cum, hi + 1, side="left")]
        edges.append(v_lo + (v_hi - v_lo) * (pos - lo))
    edges = np.unique(np.asarray(edges))     # duplicates="drop"

    if len(edges) < 2:
        out[mask] = 0
        return out
    codes = pd.cut(v, bins=edges, labels=False, include_lowest=True)
    out[mask] = codes.astype("int8")
    return out


def _require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing columns: {missing}")


class LazyPairTable:
    """Implicit (building × year) pair table; see module docstring.

    Parameters
    ----------
    buildings : pd.DataFrame with BUILDING_COLUMNS. GEOID / cbsa_code should be
        (string-valued) categoricals for memory; row order defines traversal
        order (call :meth:`shuffle_buildings` for the train ordering).
    labels : pd.DataFrame with LABEL_COLUMNS; GEOID as plain str, one row per
        (tract, year) that has a label. Missing (tract, year) pairs simply
        materialize with NaN ``Rel_Score``.
    years : iterable of imagery years (sorted internally; year-minor traversal).
    tau_meters : crop half-size, used to emit the legacy bbox columns.
    holdout_map : optional {cbsa_code (int or str) -> holdout year}; when given,
        pairs on their city's holdout year materialize with NaN Rel_Score.
    split_type : value for the ``type`` column of materialized rows.
    unavailable : optional DataFrame of (region, year) pairs with no valid NAIP
        imagery — columns ``level`` ("cbsa" | "tract"), ``key`` (cbsa_code or
        GEOID as str), ``year``. Matching pairs materialize with NaN Rel_Score,
        so the shard generator drops them before any fetch (the same mechanism
        as ``holdout_map``). Produced by ``src/tests/test_naip_coverage.py``.
    """

    def __init__(self, buildings, labels, years, tau_meters,
                 holdout_map=None, split_type="train", unavailable=None):
        _require_columns(buildings, BUILDING_COLUMNS, "buildings")
        _require_columns(labels, LABEL_COLUMNS, "labels")
        if len(buildings) == 0:
            raise ValueError("LazyPairTable: empty buildings frame")
        self.buildings = buildings.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.years = sorted(int(y) for y in years)
        if not self.years:
            raise ValueError("LazyPairTable: empty years")
        self.tau_meters = float(tau_meters)
        self._tau_units = geo_utils.meters_to_projected_units(
            self.tau_meters, geo_utils.METRIC_EPSG
        )
        self.holdout_map = {str(k): int(v) for k, v in (holdout_map or {}).items()}
        self.split_type = split_type
        self.unavailable = unavailable
        self._unavail_cbsa_by_year, self._unavail_tract_by_year = \
            self._index_unavailable(unavailable)
        self._years_arr = np.asarray(self.years, dtype="int64")
        # (Rel_Score, VSC, score_bin) lookups per year, keyed by GEOID str —
        # mapping through categorical codes keeps big materializations cheap.
        self._label_maps = {
            int(yr): grp.set_index("GEOID")[["Rel_Score", "Valid_Structural_Change", "score_bin"]]
            for yr, grp in self.labels.groupby("year")
        }

    @staticmethod
    def _index_unavailable(df):
        """Parse the unavailable table into {year -> set(keys)} per level."""
        cbsa_by_year, tract_by_year = {}, {}
        if df is None or len(df) == 0:
            return cbsa_by_year, tract_by_year
        _require_columns(df, ["level", "key", "year"], "unavailable")
        d = df[["level", "key", "year"]].copy()
        d["level"] = d["level"].astype(str)
        d["key"] = d["key"].astype(str)
        d["year"] = d["year"].astype(int)
        for lvl, target in (("cbsa", cbsa_by_year), ("tract", tract_by_year)):
            for yr, grp in d[d["level"] == lvl].groupby("year"):
                target[int(yr)] = set(grp["key"])
        return cbsa_by_year, tract_by_year

    # ------------------------------------------------------------------ #
    # Basic protocol                                                      #
    # ------------------------------------------------------------------ #
    @property
    def n_buildings(self):
        return len(self.buildings)

    @property
    def n_years(self):
        return len(self.years)

    def __len__(self):
        return self.n_buildings * self.n_years

    @property
    def shape(self):
        return (len(self), len(FLAT_COLUMNS))

    @property
    def empty(self):
        return len(self) == 0

    def __repr__(self):
        return (f"LazyPairTable(type={self.split_type!r}, "
                f"{self.n_buildings:,} buildings × {self.n_years} years "
                f"= {len(self):,} pairs)")

    # ------------------------------------------------------------------ #
    # Construction helpers                                                #
    # ------------------------------------------------------------------ #
    def shuffle_buildings(self, seed=825):
        """New table with buildings in a deterministic random order (train
        traversal). Years stay adjacent per building — temporal twins land in
        the same shard slice, replacing the legacy categorical sort."""
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.n_buildings)
        return LazyPairTable(
            self.buildings.take(perm), self.labels, self.years,
            self.tau_meters, holdout_map=self.holdout_map,
            split_type=self.split_type, unavailable=self.unavailable,
        )

    def subset(self, mask_or_idx, split_type=None, holdout_map=None):
        """New table over a subset of buildings (boolean mask or indexer)."""
        sub = (self.buildings[mask_or_idx]
               if getattr(mask_or_idx, "dtype", None) == bool
               else self.buildings.take(mask_or_idx))
        return LazyPairTable(
            sub, self.labels, self.years, self.tau_meters,
            holdout_map=self.holdout_map if holdout_map is None else holdout_map,
            split_type=self.split_type if split_type is None else split_type,
            unavailable=self.unavailable,
        )

    # ------------------------------------------------------------------ #
    # Materialization                                                     #
    # ------------------------------------------------------------------ #
    def _label_lookup(self, geoid_series, year):
        """(Rel_Score, VSC, score_bin) arrays for a GEOID Series at one year.

        ``Series.map`` on a categorical maps the (small) category set and
        expands via codes — no object-string conversion of the full column.
        """
        lm = self._label_maps.get(int(year))
        if lm is None:
            n = len(geoid_series)
            return (np.full(n, np.nan, "float32"),
                    np.zeros(n, "int8"), np.full(n, -1, "int8"))
        rel = geoid_series.map(lm["Rel_Score"]).to_numpy(dtype="float32", na_value=np.nan)
        vsc = geoid_series.map(lm["Valid_Structural_Change"]).to_numpy(dtype="float32", na_value=0)
        sbin = geoid_series.map(lm["score_bin"]).to_numpy(dtype="float32", na_value=-1)
        return rel, vsc.astype("int8"), sbin.astype("int8")

    def _flat_from(self, bldg_rows, year_values, as_str_keys):
        """Assemble a flat legacy-schema frame from building rows + per-row years."""
        out = bldg_rows.reset_index(drop=True).copy()
        out["year"] = np.asarray(year_values, dtype="int64")

        rel = np.empty(len(out), dtype="float32")
        vsc = np.zeros(len(out), dtype="int8")
        sbin = np.full(len(out), -1, dtype="int8")
        for yr in np.unique(out["year"].to_numpy()):
            sel = (out["year"].to_numpy() == yr)
            r, v, s = self._label_lookup(out.loc[sel, "GEOID"], int(yr))
            rel[sel], vsc[sel], sbin[sel] = r, v, s
        out["Rel_Score"] = rel
        out["Valid_Structural_Change"] = vsc
        out["score_bin"] = sbin

        # Train-side holdout exclusion: NaN out the label so the shard
        # generator's existing NaN guard drops the pair before any fetch.
        # (.map on a categorical maps the small category set — no str blowup.)
        if self.holdout_map:
            hold = out["cbsa_code"].map(self.holdout_map)
            on_holdout = hold.to_numpy(dtype="float64", na_value=np.nan) == out["year"].to_numpy()
            if on_holdout.any():
                out.loc[on_holdout, "Rel_Score"] = np.nan

        # Unavailable-imagery exclusion (NAIP coverage gaps / corruption): same
        # NaN-drop path, so these pairs never reach a fetch. Masked per year so
        # a single-year slice or a full-year materialization both stay vectorized.
        if self._unavail_cbsa_by_year or self._unavail_tract_by_year:
            yrs = out["year"].to_numpy()
            unavail = np.zeros(len(out), dtype=bool)
            for yr in np.unique(yrs):
                sel = yrs == int(yr)
                cset = self._unavail_cbsa_by_year.get(int(yr))
                tset = self._unavail_tract_by_year.get(int(yr))
                if cset:
                    unavail[sel] |= out.loc[sel, "cbsa_code"].isin(cset).to_numpy()
                if tset:
                    unavail[sel] |= out.loc[sel, "GEOID"].isin(tset).to_numpy()
            if unavail.any():
                out.loc[unavail, "Rel_Score"] = np.nan

        tau = np.float32(self._tau_units)
        out["bbox_minx"] = out["centroid_x"] - tau
        out["bbox_miny"] = out["centroid_y"] - tau
        out["bbox_maxx"] = out["centroid_x"] + tau
        out["bbox_maxy"] = out["centroid_y"] + tau
        # Constant columns kept for schema compatibility with the legacy flat
        # table — slim dtypes matter for materialize_year (all 71.8M rows at
        # once). row/col are NAIP-mode placeholders (unused; the zarr path needs
        # a real DataFrame); int() on int16 downstream is a no-op.
        out["dataset"] = pd.Categorical(["NAIP"] * len(out))
        for c in ("row_start", "row_stop", "col_start", "col_stop"):
            out[c] = np.int16(0)
        out["type"] = pd.Categorical([self.split_type] * len(out))

        if as_str_keys:
            # Shard-sized slices: plain str keys, byte-identical to the legacy
            # flat table (GEOID feeds hash() and the ACS lookup downstream).
            out["GEOID"] = out["GEOID"].astype(str)
            out["cbsa_code"] = out["cbsa_code"].astype(str)
        return out[FLAT_COLUMNS]

    def materialize(self, start, stop):
        """Flat frame for pair positions [start, stop), cyclic over the table.

        Traversal is building-major / year-minor: pair k is
        (building k // n_years, years[k % n_years]).
        """
        if stop <= start:
            raise ValueError(f"materialize: empty range [{start}, {stop})")
        idx = np.arange(start, stop, dtype="int64") % len(self)
        b_idx = idx // self.n_years
        y_idx = idx % self.n_years
        return self._flat_from(
            self.buildings.take(b_idx),
            self._years_arr[y_idx],
            as_str_keys=True,
        )

    def materialize_year(self, year):
        """Flat frame with ALL buildings at one year (prediction path).

        GEOID / cbsa_code stay categorical here — at 71.8M buildings the str
        conversion alone is ~5 GB; row-wise consumers still see str values.
        """
        year = int(year)
        if year not in self.years:
            return pd.DataFrame(columns=FLAT_COLUMNS)
        return self._flat_from(
            self.buildings,
            np.full(self.n_buildings, year, dtype="int64"),
            as_str_keys=False,
        )

    def sample_pairs(self, n, seed=825):
        """Flat frame of ``n`` random pairs (deterministic) — small_sample path."""
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(self), size=min(int(n), len(self)), replace=False))
        return self._flat_from(
            self.buildings.take(idx // self.n_years),
            self._years_arr[idx % self.n_years],
            as_str_keys=True,
        )

    def sample_buildings_per_tract(self, n_buildings_per_tract=2, seed=825):
        """Small flat frame keeping ALL years of ≤n buildings per tract.

        Replaces main._subsample_val_buildings for lazy validation sets: whole
        buildings are kept (never single year-rows) so stable/changed MASD and
        directional accuracy stay computable.
        """
        rng = np.random.default_rng(seed)
        order = rng.permutation(self.n_buildings)
        shuffled = self.buildings.take(order)
        keep = shuffled.groupby("GEOID", observed=True).cumcount() < n_buildings_per_tract
        picked = shuffled[keep.to_numpy()]
        rep = picked.take(np.repeat(np.arange(len(picked)), self.n_years))
        years = np.tile(self._years_arr, len(picked))
        return self._flat_from(rep, years, as_str_keys=True)

    def materialize_holdout_years(self, holdout_map, n_buildings_per_tract=2, seed=825):
        """val_temporal frame: ≤n buildings per tract, ONLY their city's holdout
        year. Buildings in cities without a holdout year are excluded."""
        hmap = {str(k): int(v) for k, v in holdout_map.items()}
        cbsa_str = self.buildings["cbsa_code"].astype(str)
        hold = cbsa_str.map(hmap)
        eligible = hold.notna().to_numpy()
        if not eligible.any():
            return pd.DataFrame(columns=FLAT_COLUMNS)

        rng = np.random.default_rng(seed)
        order = rng.permutation(self.n_buildings)
        shuffled = self.buildings.take(order)
        shuffled = shuffled[eligible[order]]     # restrict BEFORE the per-tract count
        keep = (shuffled.groupby("GEOID", observed=True).cumcount() < n_buildings_per_tract)
        picked = shuffled[keep.to_numpy()]
        years = picked["cbsa_code"].astype(str).map(hmap).to_numpy(dtype="int64")
        return self._flat_from(picked, years, as_str_keys=True)

    def building_reference(self):
        """building_id → GEOID / cbsa_code lookup for BatchImageDumper."""
        return self.buildings[["building_id", "GEOID", "cbsa_code"]].copy()

    def fully_unavailable(self):
        """Regions with NO valid NAIP in ANY requested year → (cbsas, tracts).

        A region is *fully* unavailable only when it is flagged for **every**
        requested year (the intersection of the per-year gap sets). A region
        flagged for just some years stays available (those years are still
        NaN-masked). Used to drop dead regions from the train/val/test universe.
        Returns two sets of str keys (cbsa_codes, GEOIDs).
        """
        def _all(by_year):
            sets = [by_year.get(y) for y in self.years]
            if any(not s for s in sets):   # a requested year with no gaps ⇒ nothing dead in ALL years
                return set()
            return set.intersection(*(set(s) for s in sets))
        return _all(self._unavail_cbsa_by_year), _all(self._unavail_tract_by_year)
