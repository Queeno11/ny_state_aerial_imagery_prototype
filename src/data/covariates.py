"""Tract-year ACS covariates fed to the late-fusion head (``model_name="scalemae_cov"``).

Diagnostic variant of the imagery model: besides the image features, the head
sees who lives in the tract — the share of adults 25+ with a bachelor's degree
or more (``edu_ba_share``, ACS B15003) and the share of residents aged 25-54
(``age_25_54_share``, ACS B01001). It answers "does wealth change become
predictable once the model is told about resident composition?", NOT the
paper's imagery-only question: at prediction time the covariates come from the
same ACS survey as the wealth label, so changes in them share sampling noise
with the label change. Compare against an ACS-only baseline before reading any
change R^2 as imagery skill.

Scale. Like the wealth label (``Rel_Score``), each covariate is z-scored within
CBSA x year, so city-wide drift (e.g. rising education everywhere) carries no
signal — consistent with the model's invariance to macro drift. Values are
clipped to +-5 SD; a tract-year with no estimate gets 0 (its CBSA-year mean).

Geography. Each ACS vintage uses its own tract vintage (2010 tracts through
2019, 2020 tracts from 2020); ``us_metros_panel_*.feather`` carries the
crosswalk (``geoid_{year}`` for every row keyed by ``geoid_2023``), the same
key the training shards and ``main.build_se_lookup`` use. Years outside the
panel take the nearest panel year, mirroring the label padding (2010 -> 2011,
2024 -> 2023).

Build once (writes ``data/processed/tract_covariates.parquet``):
    python -m src.data.covariates
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COVARIATE_NAMES = ("edu_ba_share", "age_25_54_share")
TABLE_FILENAME = "tract_covariates.parquet"
CLIP_SD = 5.0

# B15003: educational attainment, population 25+. 022-025 = bachelor's,
# master's, professional, doctorate.
_EDU_NUM = [f"B15003_{i:03d}E" for i in (22, 23, 24, 25)]
_EDU_DEN = "B15003_001E"
# B01001: sex by age. Male 011-016 and female 035-040 are 25-29 ... 50-54.
_AGE_NUM = ([f"B01001_{i:03d}E" for i in range(11, 17)]
            + [f"B01001_{i:03d}E" for i in range(35, 41)])
_AGE_DEN = "total_population"
ACS_COLUMNS = ["geoid", *_EDU_NUM, _EDU_DEN, *_AGE_NUM, _AGE_DEN]


def _share(num: pd.Series, den: pd.Series) -> pd.Series:
    den = pd.to_numeric(den, errors="coerce")
    return pd.to_numeric(num, errors="coerce") / den.where(den > 0)


def raw_covariates(acs: pd.DataFrame) -> pd.DataFrame:
    """Shares (0-1) per tract from one ACS vintage -> [geoid, *COVARIATE_NAMES]."""
    out = pd.DataFrame({"geoid": acs["geoid"].astype(str).str.zfill(11)})
    edu_num = acs[_EDU_NUM].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    age_num = acs[_AGE_NUM].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    out["edu_ba_share"] = _share(edu_num, acs[_EDU_DEN]).to_numpy()
    out["age_25_54_share"] = _share(age_num, acs[_AGE_DEN]).to_numpy()
    for c in COVARIATE_NAMES:
        out[c] = out[c].clip(0.0, 1.0)
    return out.drop_duplicates("geoid")


def crosswalk_to_panel(raw: pd.DataFrame, panel: pd.DataFrame, year: int) -> pd.DataFrame:
    """One vintage's raw shares on the panel's rows -> [geoid_2023, cbsa, year, ...].

    ``panel`` needs ``geoid_2023``, ``cbsa_code`` and ``geoid_{year}`` (or, for
    2023, just ``geoid_2023``).
    """
    key = f"geoid_{year}" if f"geoid_{year}" in panel.columns else "geoid_2023"
    src = panel[key].astype(str).str.zfill(11)
    vals = raw.set_index("geoid").reindex(src.to_numpy())
    out = pd.DataFrame({"geoid_2023": panel["geoid_2023"].astype(str).str.zfill(11).to_numpy(),
                        "cbsa": panel["cbsa_code"].astype(str).to_numpy(),
                        "year": int(year)})
    for c in COVARIATE_NAMES:
        out[c] = vals[c].to_numpy()
    return out


def fill_uncovered_years(long: pd.DataFrame, names=COVARIATE_NAMES,
                         min_coverage: float = 0.5) -> pd.DataFrame:
    """Per covariate, a year with less than ``min_coverage`` non-missing values
    takes each tract's value from the nearest well-covered year.

    ACS B15003 (education) starts with the 2012 release, so 2011 has no
    education at all; without this every 2011 tract-year (and the 2010 labels
    padded onto it) would get the CBSA mean. Nearest-year filling is the same
    rule the labels and SEs use for years outside their coverage.
    """
    out = long.copy()
    for c in names:
        cover = out.groupby("year")[c].apply(lambda s: s.notna().mean())
        good = np.array(sorted(cover[cover >= min_coverage].index))
        bad = [y for y in cover.index if cover[y] < min_coverage]
        if not len(good) or not bad:
            continue
        wide = out.pivot_table(index="geoid_2023", columns="year", values=c, aggfunc="first")
        for y in bad:
            src = int(good[np.abs(good - y).argmin()])
            m = out["year"] == y
            out.loc[m, c] = wide[src].reindex(out.loc[m, "geoid_2023"]).to_numpy()
            print(f"  {c}: {y} has {cover[y]:.0%} coverage -> filled from {src}")
    return out


def standardize_within(long: pd.DataFrame, names=COVARIATE_NAMES,
                       by=("cbsa", "year"), clip: float = CLIP_SD) -> pd.DataFrame:
    """z-score each covariate within ``by`` groups; clip; NaN -> 0 (group mean).
    A group with zero spread gets 0 everywhere."""
    out = long.copy()
    g = out.groupby(list(by))
    for c in names:
        mu = g[c].transform("mean")
        sd = g[c].transform("std")
        z = (out[c] - mu) / sd.where(sd > 0)
        out[f"{c}_z"] = z.clip(-clip, clip).fillna(0.0).astype("float32")
    return out


def build_covariate_table(acs_root: Path, panel_path: Path, years) -> pd.DataFrame:
    """Long table [geoid_2023, cbsa, year, raw shares, *_z] over ``years``."""
    panel_cols = ["geoid_2023", "cbsa_code"] + [f"geoid_{y}" for y in years if y != 2023]
    import pyarrow as pa
    present = set(pa.ipc.open_file(panel_path).schema.names)
    panel = pd.read_feather(panel_path, columns=[c for c in panel_cols if c in present])
    panel = panel.drop_duplicates("geoid_2023")
    frames = []
    for y in years:
        acs_path = Path(acs_root) / str(y) / f"us_tracts_acs5_{y}.feather"
        acs = pd.read_feather(acs_path, columns=ACS_COLUMNS)
        frames.append(crosswalk_to_panel(raw_covariates(acs), panel, y))
        print(f"  {y}: {frames[-1][list(COVARIATE_NAMES)].notna().mean().round(3).to_dict()} non-missing")
    return standardize_within(fill_uncovered_years(pd.concat(frames, ignore_index=True)))


class CovariateLookup:
    """(tract, year) -> standardized covariate vector, vectorized.

    ``geoids`` are 11-char GEOID strings (prediction frames) or ints; with
    ``hashed=True`` they are the ``tract_sampling.stable_geoid_hash`` ids the
    training/validation shards store instead of GEOIDs. Years are snapped to
    the nearest year in the table; unknown tracts get zeros (the CBSA-year
    mean).
    """

    def __init__(self, table: pd.DataFrame, names=COVARIATE_NAMES):
        self.names = tuple(names)
        cols = [f"{c}_z" for c in self.names]
        t = table[["geoid_2023", "year", *cols]].copy()
        t["gid"] = pd.to_numeric(t["geoid_2023"], errors="coerce").astype("int64")
        t = t.drop_duplicates(["gid", "year"])
        self.years = np.array(sorted(t["year"].unique()), dtype="int64")
        self._values = t.set_index(["gid", "year"])[cols].astype("float32")
        from src.data.tract_sampling import hash_geoids
        t["hid"] = hash_geoids(t["geoid_2023"].astype(str).str.zfill(11).tolist())
        # CRC32 mod 2^31 over ~73k tracts: a collision or two is possible; the
        # first tract wins (its neighbour in hash space gets its covariates).
        self._hashed = (t.drop_duplicates(["hid", "year"])
                        .set_index(["hid", "year"])[cols].astype("float32"))

    @property
    def dim(self) -> int:
        return len(self.names)

    def snap_years(self, years) -> np.ndarray:
        y = np.asarray(years, dtype="int64").reshape(-1)
        idx = np.abs(y[:, None] - self.years[None, :]).argmin(axis=1)
        return self.years[idx]

    @staticmethod
    def _keys(geoids, hashed: bool) -> np.ndarray:
        arr = np.asarray(geoids).reshape(-1)
        if hashed:
            return arr.astype("int64")
        return (pd.to_numeric(pd.Series(arr).astype(str), errors="coerce")
                .fillna(-1).astype("int64").to_numpy())

    def lookup(self, geoids, years, hashed: bool = False) -> np.ndarray:
        """(n, dim) float32 array aligned with the inputs."""
        table = self._hashed if hashed else self._values
        idx = pd.MultiIndex.from_arrays([self._keys(geoids, hashed), self.snap_years(years)])
        return np.nan_to_num(table.reindex(idx).to_numpy("float32"), nan=0.0)

    def coverage(self, geoids, years, hashed: bool = False) -> float:
        """Share of (tract, year) pairs with a row in the table."""
        table = self._hashed if hashed else self._values
        idx = pd.MultiIndex.from_arrays([self._keys(geoids, hashed), self.snap_years(years)])
        return float(idx.isin(table.index).mean()) if len(idx) else 0.0


def table_path(processed_dir: Path | None = None) -> Path:
    from src.utils.paths import PROCESSED_DATA_DIR
    return Path(processed_dir or PROCESSED_DATA_DIR) / TABLE_FILENAME


def load_lookup(names=COVARIATE_NAMES, path: Path | None = None) -> CovariateLookup:
    """Lookup over the prebuilt table; raises with the build command if absent."""
    names = tuple(names)
    unknown = set(names) - set(COVARIATE_NAMES)
    if unknown:
        raise ValueError(f"unknown covariates {sorted(unknown)}; available: {COVARIATE_NAMES}")
    path = Path(path) if path else table_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — build it first: python -m src.data.covariates")
    return CovariateLookup(pd.read_parquet(path), names)


def main() -> None:
    from src.data.process_acs import PANEL_YEARS
    from src.utils.paths import ACS_ROOT_DIR, PROCESSED_DATA_DIR

    ap = argparse.ArgumentParser(description="Build the tract-year covariate table.")
    ap.add_argument("--acs-root", type=Path, default=Path(ACS_ROOT_DIR))
    ap.add_argument("--panel", type=Path, default=Path(PROCESSED_DATA_DIR) /
                    f"us_metros_panel_{PANEL_YEARS[0]}_{PANEL_YEARS[-1]}.feather")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    table = build_covariate_table(args.acs_root, args.panel, list(PANEL_YEARS))
    out = args.out or table_path()
    table.to_parquet(out, index=False)
    print(f"wrote {out}: {len(table):,} tract-years, {table['geoid_2023'].nunique():,} tracts")


if __name__ == "__main__":
    main()
