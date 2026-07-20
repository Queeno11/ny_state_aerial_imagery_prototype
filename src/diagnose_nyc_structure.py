"""Decompose NYC underperformance into within- vs between-borough failure.

Two competing hypotheses for why the US model collapses on NYC while the
NYC-only model reached rho ~ 0.85:

  H1 (pair exclusion): the permanent ``bin_diff == 0`` curriculum filter drops
     same-quintile pairs; inside one borough most tracts share a quintile, so
     intra-borough ranking gets little/no gradient. Fingerprint: pooled
     WITHIN-borough rho ~ 0, BETWEEN-borough ordering fine.
  H2 (US prior collapse): the model transfers the typical US gradient
     (dense downtown = poor, suburb = rich), which NYC inverts. Fingerprint:
     BETWEEN-borough ordering inverted/compressed (Manhattan predicted poor),
     within-borough rho comparatively OK.

Reads ``results/<run>/predictions_by_tract_<year>.parquet`` (GEOID, Rel_Score,
predicted_value) and prints, per year with NYC coverage: overall NYC rho,
pooled within-borough rho (borough-demeaned), between-borough rho over the
five borough means, and per-borough within rho + mean label/pred.

Part B quantifies H1 directly from the training labels: per borough (and per
test CBSA for reference) the fraction of intra-group tract pairs that share a
``score_bin`` quintile — exactly the pairs the permanent filter removes.

Part C quantifies H2: per CBSA-year, the Spearman of predictions vs log
building density compared with labels vs log density. If predictions lean on
density far more negatively than labels do — everywhere — the model carries a
"dense = poor" prior that NYC uniquely violates.

Read-only diagnostic:
    python src/diagnose_nyc_structure.py [--run run_20260710] [--parts ABC]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# NOTE: not importing src.utils.paths — it hard-fails when .env is unreadable
# (sandboxed sessions); this script only needs the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BOROUGHS = {"36005": "Bronx", "36047": "Brooklyn", "36061": "Manhattan",
            "36081": "Queens", "36085": "StatenIsl"}
BOROUGH_ORDER = ["Manhattan", "Brooklyn", "Queens", "Bronx", "StatenIsl"]
DEFAULT_YEARS = (2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020)
NYC_CBSA = "35620"

PAIR_LABELS = PROJECT_ROOT / "data/processed/pair_labels_ms_us_W2_r5_years2010-2024_all.parquet"
BUILDINGS = PROJECT_ROOT / "data/processed/pair_buildings_ms_us_epsg5070_all.parquet"
TRACT_SPLITS = PROJECT_ROOT / "data/processed/tract_splits.feather"
CBSA_SPLITS = PROJECT_ROOT / "data/processed/cbsa_splits.feather"
CROSSWALK = PROJECT_ROOT / "data/external/cbsa/county_cbsa_crosswalk.csv"


# ── pure pieces (unit-tested on synthetic data) ──────────────────────────────

def add_borough(df: pd.DataFrame) -> pd.DataFrame:
    """Subset to the five boroughs and add a ``boro`` column (GEOID prefix)."""
    out = df.copy()
    out["GEOID"] = out["GEOID"].astype(str)
    county = out["GEOID"].str[:5]
    out = out[county.isin(BOROUGHS)].copy()
    out["boro"] = out["GEOID"].str[:5].map(BOROUGHS)
    return out


def decompose_city(df: pd.DataFrame, group_col: str = "boro",
                   label_col: str = "Rel_Score",
                   pred_col: str = "predicted_value") -> dict:
    """Overall / pooled-within / between-group Spearman decomposition.

    ``pooled_within`` demeans label and prediction by group before the pooled
    Spearman, isolating intra-group ranking skill; ``between`` is the Spearman
    over the group means (ordering of the groups themselves).
    """
    overall = spearmanr(df[label_col], df[pred_col])[0]
    gm = df.groupby(group_col)[[label_col, pred_col]].mean()
    between = (spearmanr(gm[label_col], gm[pred_col])[0]
               if len(gm) >= 3 else np.nan)
    lab_dm = df[label_col] - df.groupby(group_col)[label_col].transform("mean")
    pred_dm = df[pred_col] - df.groupby(group_col)[pred_col].transform("mean")
    pooled_within = spearmanr(lab_dm, pred_dm)[0]
    per_group = {
        g: (spearmanr(sub[label_col], sub[pred_col])[0], len(sub))
        for g, sub in df.groupby(group_col) if len(sub) >= 10
    }
    return {"n": len(df), "overall": overall, "between": between,
            "pooled_within": pooled_within, "group_means": gm,
            "per_group": per_group}


def same_bin_pair_fraction(bins) -> float:
    """Fraction of unordered tract pairs sharing a score_bin (the pairs the
    permanent ``bin_diff == 0`` filter excludes). NaN-safe; NaN if < 2 rows."""
    b = pd.Series(bins).dropna()
    n = len(b)
    if n < 2:
        return np.nan
    counts = b.value_counts().to_numpy(dtype="float64")
    return float((counts * (counts - 1)).sum() / (n * (n - 1)))


def zscore_by(df: pd.DataFrame, col: str, by) -> pd.Series:
    """Within-group z-score of ``col`` (population SD, ddof=0)."""
    g = df.groupby(by)[col]
    return (df[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)


def density_lean(df: pd.DataFrame, label_col="Rel_Score",
                 pred_col="predicted_value", dens_col="log_density") -> dict:
    """Spearman of prediction vs density and label vs density for one city-year.

    ``lean_gap`` = rho(pred, dens) - rho(label, dens): how much more the model
    leans on density than the ground truth warrants (negative = model treats
    dense tracts as poorer than labels do).
    """
    rp = spearmanr(df[pred_col], df[dens_col])[0]
    rl = spearmanr(df[label_col], df[dens_col])[0]
    return {"n": len(df), "rho_pred_dens": rp, "rho_label_dens": rl,
            "lean_gap": rp - rl}


# ── I/O + report ─────────────────────────────────────────────────────────────

def load_year(run_dir: Path, year: int) -> pd.DataFrame | None:
    f = run_dir / f"predictions_by_tract_{year}.parquet"
    if not f.exists():
        return None
    return pd.read_parquet(f)


def report(run: str, years=DEFAULT_YEARS) -> pd.DataFrame:
    run_dir = PROJECT_ROOT / "results" / run
    rows = []
    for year in years:
        df = load_year(run_dir, year)
        if df is None:
            continue
        nyc = add_borough(df)
        if len(nyc) < 30:
            continue
        d = decompose_city(nyc)
        print(f"\n=== {year}  (n={d['n']} borough tracts) ===")
        print(f"overall rho: {d['overall']:+.3f}   "
              f"pooled within-borough: {d['pooled_within']:+.3f}   "
              f"between-borough (5 means): {d['between']:+.3f}")
        gm = d["group_means"]
        for b in BOROUGH_ORDER:
            if b in d["per_group"]:
                r, n = d["per_group"][b]
                print(f"  within {b:10s}: rho={r:+.3f} (n={n:4d})  "
                      f"mean_label={gm.loc[b, 'Rel_Score']:+.2f}  "
                      f"mean_pred={gm.loc[b, 'predicted_value']:+.2f}")
        rows.append({"year": year, "n": d["n"], "overall": d["overall"],
                     "pooled_within": d["pooled_within"],
                     "between": d["between"]})
    out = pd.DataFrame(rows)
    if len(out):
        print("\n=== summary across years ===")
        print(out.round(3).to_string(index=False))
    return out


def _county_to_cbsa() -> pd.Series:
    cw = pd.read_csv(CROSSWALK, dtype=str)
    return cw.set_index("county_fips")["cbsa_code"]


def report_bins() -> None:
    """Part B: same-quintile pair exclusion, boroughs vs test CBSAs."""
    labels = pd.read_parquet(PAIR_LABELS)
    labels["GEOID"] = labels["GEOID"].astype(str).str.zfill(11)
    labels["county"] = labels["GEOID"].str[:5]
    labels["cbsa"] = labels["county"].map(_county_to_cbsa())
    splits = pd.read_feather(CBSA_SPLITS)
    split_of = splits.set_index(splits["cbsa_code"].astype(str))["split"]

    print("\n===== PART B: fraction of intra-group pairs excluded by bin_diff==0 =====")
    print("(mean over training years; higher = less within-group ranking signal)\n")
    nyc = labels[labels["cbsa"] == NYC_CBSA]
    rows = []
    for county, name in BOROUGHS.items():
        sub = nyc[nyc["county"] == county]
        frac = sub.groupby("year")["score_bin"].apply(same_bin_pair_fraction)
        hist = (sub["score_bin"].value_counts(normalize=True)
                .sort_index().round(2).to_dict())
        rows.append({"group": name, "n_tracts": sub["GEOID"].nunique(),
                     "frac_same_bin": frac.mean(), "bin_shares": hist})
    frac_cbsa = nyc.groupby("year")["score_bin"].apply(same_bin_pair_fraction)
    rows.append({"group": "NYC CBSA (all)", "n_tracts": nyc["GEOID"].nunique(),
                 "frac_same_bin": frac_cbsa.mean(),
                 "bin_shares": (nyc["score_bin"].value_counts(normalize=True)
                                .sort_index().round(2).to_dict())})
    for r in rows:
        print(f"  {r['group']:15s} n={r['n_tracts']:5d}  "
              f"frac_same_bin={r['frac_same_bin']:.3f}  bins={r['bin_shares']}")

    # reference: distribution of the same statistic across test CBSAs
    other = labels[labels["cbsa"].map(split_of).eq("test") & (labels["cbsa"] != NYC_CBSA)]
    per_cbsa = (other.groupby(["cbsa", "year"])["score_bin"]
                .apply(same_bin_pair_fraction).groupby("cbsa").mean())
    print(f"\n  test CBSAs (n={len(per_cbsa)}): frac_same_bin "
          f"median={per_cbsa.median():.3f}  "
          f"p10={per_cbsa.quantile(.1):.3f}  p90={per_cbsa.quantile(.9):.3f}")


def report_density(run: str, years=DEFAULT_YEARS) -> None:
    """Part C: does the model lean on building density more than labels do?"""
    import pyarrow.parquet as pq
    import geopandas as gpd

    geoids = pq.read_table(BUILDINGS, columns=["GEOID"])["GEOID"].to_pandas()
    counts = geoids.value_counts()
    counts = counts[counts > 0]
    counts.index = counts.index.astype(str).str.zfill(11)

    tracts = gpd.read_feather(TRACT_SPLITS)
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)
    area_km2 = tracts.set_index("GEOID").geometry.area / 1e6
    dens = (counts / area_km2.reindex(counts.index)).rename("bldg_density")
    log_dens = np.log10(dens.replace(0, np.nan)).rename("log_density")

    c2c = _county_to_cbsa()
    run_dir = PROJECT_ROOT / "results" / run
    nyc_rows, other_rows = [], []
    for year in years:
        df = load_year(run_dir, year)
        if df is None:
            continue
        df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)
        df["cbsa"] = df["GEOID"].str[:5].map(c2c)
        df["log_density"] = df["GEOID"].map(log_dens)
        df = df.dropna(subset=["cbsa", "log_density", "Rel_Score", "predicted_value"])
        for cbsa, sub in df.groupby("cbsa"):
            if len(sub) < 50:
                continue
            d = density_lean(sub)
            d.update({"cbsa": cbsa, "year": year})
            (nyc_rows if cbsa == NYC_CBSA else other_rows).append(d)

    print("\n===== PART C: density lean — rho(pred, log bldg density) vs rho(label, log density) =====")
    print("\n  NYC (35620):")
    for d in nyc_rows:
        print(f"    {d['year']}  n={d['n']:5d}  rho_pred_dens={d['rho_pred_dens']:+.3f}  "
              f"rho_label_dens={d['rho_label_dens']:+.3f}  lean_gap={d['lean_gap']:+.3f}")
    if other_rows:
        o = pd.DataFrame(other_rows)
        agg = o.groupby("cbsa")[["rho_pred_dens", "rho_label_dens", "lean_gap"]].mean()
        print(f"\n  other CBSAs (n={len(agg)} cities, mean over years):")
        print("    " + agg.describe().loc[["mean", "25%", "50%", "75%"]]
              .round(3).to_string().replace("\n", "\n    "))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run_20260710")
    ap.add_argument("--parts", default="A",
                    help="any of A (within/between decomposition), "
                         "B (bin exclusion), C (density lean)")
    args = ap.parse_args()
    if "A" in args.parts.upper():
        report(args.run)
    if "B" in args.parts.upper():
        report_bins()
    if "C" in args.parts.upper():
        report_density(args.run)
