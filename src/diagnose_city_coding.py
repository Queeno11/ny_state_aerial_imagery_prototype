"""City-coding audit: is the model encoding city identity in the score *level*?

Labels are z-scored per CBSA (x year), so between-city label variance is ~0 by
construction. Any between-city variance in the *predictions* is therefore "range
coding" (e.g. LA scored in [0.7, 0.8], NYC in [0.8, 0.9]) and never label signal.
Because HybridBatchSampler builds each cross-sectional core from a single
(year, CBSA), the training loss neither rewards nor penalizes such offsets — this
script measures how much of it the model actually does.

Per year (the z-scores are cross-sectional within CBSA x year) it reports:

  eta2_pred     share of prediction variance explained by city dummies
  eta2_label    the same for labels — the empirical null, ~0 by construction
  rho_pooled    Spearman pooling every city (what val_cities_spearman logs)
  rho_within    n-weighted mean of per-city Spearman
  rho_demeaned  pooled Spearman after subtracting each city's mean prediction

No city coding looks like: eta2_pred ~ eta2_label and rho_demeaned ~ rho_pooled.
Range coding looks like: eta2_pred >> eta2_label and rho_demeaned >> rho_pooled.

Input: prediction CSVs written by predict_buildings_chunked
(``Rel_Score, predicted_value, building_id, GEOID, year, type``). The city is
taken from a ``cbsa_code`` column when present, otherwise derived from
GEOID[:5] (county FIPS) via the county->CBSA crosswalk the ACS pipeline caches.

Usage (GPU box, after generate_predictions):
    python src/diagnose_city_coding.py "results/<savename>/"*_predictions.csv \
        --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CROSSWALK = PROJECT_ROOT / "data/external/cbsa/county_cbsa_crosswalk.csv"

PRED_COL, LABEL_COL, CITY_COL, YEAR_COL = "predicted_value", "Rel_Score", "cbsa_code", "year"

# eta2_pred - eta2_label below this is measurement noise; above HIGH is range coding.
ETA2_EXCESS_MILD = 0.03
ETA2_EXCESS_HIGH = 0.10


# ── core statistics ──────────────────────────────────────────────────────────

def eta_squared(values, groups) -> float:
    """Share of variance in ``values`` explained by ``groups`` (1 - SSwithin/SStotal)."""
    df = pd.DataFrame({"v": np.asarray(values, dtype=float), "g": np.asarray(groups)}).dropna()
    if len(df) < 2:
        return float("nan")
    ss_total = ((df["v"] - df["v"].mean()) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_within = ((df["v"] - df.groupby("g")["v"].transform("mean")) ** 2).sum()
    return float(1.0 - ss_within / ss_total)


def _spearman(pred, label) -> float:
    from scipy.stats import spearmanr

    if len(pred) < 3 or pd.Series(label).nunique() < 2 or pd.Series(pred).nunique() < 2:
        return float("nan")
    rho, _ = spearmanr(pred, label)
    return float(rho)


def per_city_stats(df: pd.DataFrame, min_n: int = 20) -> pd.DataFrame:
    """Per-city n, prediction moments, label mean, and within-city Spearman.

    Cities with fewer than ``min_n`` rows keep their moments but get rho = NaN
    (too few tracts for a stable rank correlation).
    """
    rows = []
    for city, g in df.groupby(CITY_COL):
        rows.append({
            CITY_COL: city,
            "n": len(g),
            "pred_mean": g[PRED_COL].mean(),
            "pred_std": g[PRED_COL].std(),
            "label_mean": g[LABEL_COL].mean(),
            "rho": _spearman(g[PRED_COL], g[LABEL_COL]) if len(g) >= min_n else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("pred_mean").reset_index(drop=True)


def weighted_within_spearman(city_stats: pd.DataFrame) -> float:
    """n-weighted mean of per-city Spearman over cities where it is defined."""
    ok = city_stats.dropna(subset=["rho"])
    if ok.empty:
        return float("nan")
    return float(np.average(ok["rho"], weights=ok["n"]))


def demeaned_pooled_spearman(df: pd.DataFrame) -> float:
    """Pooled Spearman after removing each city's mean prediction (offset-free)."""
    demeaned = df[PRED_COL] - df.groupby(CITY_COL)[PRED_COL].transform("mean")
    return _spearman(demeaned, df[LABEL_COL])


def diagnose_year(df: pd.DataFrame, min_n: int = 20) -> dict:
    """All audit statistics for one cross-section (one year)."""
    stats = per_city_stats(df, min_n=min_n)
    return {
        "n": len(df),
        "n_cities": df[CITY_COL].nunique(),
        "eta2_pred": eta_squared(df[PRED_COL], df[CITY_COL]),
        "eta2_label": eta_squared(df[LABEL_COL], df[CITY_COL]),
        "rho_pooled": _spearman(df[PRED_COL], df[LABEL_COL]),
        "rho_within": weighted_within_spearman(stats),
        "rho_demeaned": demeaned_pooled_spearman(df),
        "city_stats": stats,
    }


def diagnose(df: pd.DataFrame, min_n: int = 20) -> dict:
    """Per-year audits plus n-weighted aggregates and a pooled per-city table."""
    df = df.dropna(subset=[PRED_COL, LABEL_COL, CITY_COL]).copy()
    if df.empty:
        raise ValueError("No rows left after dropping NaN pred/label/city.")

    years = {int(y): diagnose_year(g, min_n=min_n) for y, g in df.groupby(YEAR_COL)}

    ns = np.array([r["n"] for r in years.values()], dtype=float)
    agg = {
        k: float(np.average([r[k] for r in years.values()], weights=ns))
        for k in ("eta2_pred", "eta2_label", "rho_pooled", "rho_within", "rho_demeaned")
    }
    agg["n"] = int(ns.sum())
    agg["n_cities"] = df[CITY_COL].nunique()
    agg["eta2_excess"] = agg["eta2_pred"] - agg["eta2_label"]

    if agg["eta2_excess"] < ETA2_EXCESS_MILD:
        verdict = "NEGLIGIBLE — no meaningful city coding; scores are comparable across cities."
    elif agg["eta2_excess"] < ETA2_EXCESS_HIGH:
        verdict = ("MILD — cities carry small score offsets; within-city use is unaffected, "
                   "cross-city score comparisons are slightly biased.")
    else:
        verdict = ("SUBSTANTIAL — city identity explains a large share of prediction variance; "
                   "raw scores are NOT comparable across cities.")

    return {"years": years, "aggregate": agg, "verdict": verdict,
            "city_stats": per_city_stats(df, min_n=min_n)}


# ── input handling ───────────────────────────────────────────────────────────

def load_crosswalk(path=DEFAULT_CROSSWALK) -> pd.DataFrame:
    xw = pd.read_csv(path, dtype={"county_fips": str, "cbsa_code": str})
    xw["county_fips"] = xw["county_fips"].str.zfill(5)
    return xw[["county_fips", "cbsa_code"]].drop_duplicates("county_fips")


def attach_cbsa(df: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Derive cbsa_code from GEOID[:5] (county FIPS). Rows outside any CBSA are dropped."""
    df = df.copy()
    df["county_fips"] = df["GEOID"].astype(str).str.zfill(11).str[:5]
    df = df.merge(crosswalk, on="county_fips", how="left")
    n_unmapped = int(df[CITY_COL].isna().sum())
    if n_unmapped:
        print(f"⚠️ {n_unmapped}/{len(df)} rows have no CBSA in the crosswalk — dropped.")
    return df.dropna(subset=[CITY_COL]).drop(columns=["county_fips"])


def load_predictions(paths, split: str | None = None,
                     crosswalk_path=DEFAULT_CROSSWALK) -> pd.DataFrame:
    frames = []
    for p in paths:
        p = Path(p)
        frames.append(pd.read_parquet(p) if p.suffix in (".parquet", ".feather")
                      else pd.read_csv(p, dtype={"GEOID": str}))
    df = pd.concat(frames, ignore_index=True)
    if split is not None and "type" in df.columns:
        df = df[df["type"] == split]
        if df.empty:
            raise ValueError(f"No rows with type == '{split}'.")
    if CITY_COL not in df.columns:
        if "GEOID" not in df.columns:
            raise ValueError(f"Need either a '{CITY_COL}' or a 'GEOID' column to identify cities.")
        df = attach_cbsa(df, load_crosswalk(crosswalk_path))
    return df


# ── reporting ────────────────────────────────────────────────────────────────

def format_report(res: dict, top: int = 12) -> str:
    a = res["aggregate"]
    lines = [
        "═" * 78,
        "CITY-CODING AUDIT (labels are per-CBSA z-scores: eta2_label ~ 0 by construction)",
        "═" * 78,
        f"rows: {a['n']}   cities: {a['n_cities']}",
        f"eta2_pred:    {a['eta2_pred']:.4f}   (share of prediction variance from city identity)",
        f"eta2_label:   {a['eta2_label']:.4f}   (empirical null)",
        f"eta2_excess:  {a['eta2_excess']:.4f}",
        f"rho_pooled:   {a['rho_pooled']:.4f}   (cross-city pooled Spearman, as logged in wandb)",
        f"rho_within:   {a['rho_within']:.4f}   (n-weighted per-city Spearman)",
        f"rho_demeaned: {a['rho_demeaned']:.4f}   (pooled after removing per-city offsets)",
        f"offset drag on pooled rho: {a['rho_demeaned'] - a['rho_pooled']:+.4f}",
        f"VERDICT: {res['verdict']}",
        "─" * 78,
        "per-year:",
    ]
    for y, r in sorted(res["years"].items()):
        lines.append(f"  {y}: n={r['n']:>6}  cities={r['n_cities']:>3}  "
                     f"eta2_pred={r['eta2_pred']:.4f}  rho_pooled={r['rho_pooled']:.4f}  "
                     f"rho_within={r['rho_within']:.4f}  rho_demeaned={r['rho_demeaned']:.4f}")
    cs = res["city_stats"]
    lines += ["─" * 78,
              f"most offset cities (all years pooled; |pred_mean| is the offset, labels are ~0):"]
    show = cs.reindex(cs["pred_mean"].abs().sort_values(ascending=False).index).head(top)
    for _, r in show.iterrows():
        rho = f"{r['rho']:.3f}" if pd.notna(r["rho"]) else "  n/a"
        lines.append(f"  cbsa {r[CITY_COL]}: n={int(r['n']):>6}  pred_mean={r['pred_mean']:+.3f}  "
                     f"pred_std={r['pred_std']:.3f}  rho={rho}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("predictions", nargs="+", help="prediction CSV/parquet file(s)")
    ap.add_argument("--split", default=None,
                    help="filter on the 'type' column (e.g. test, val_cities); default: all rows")
    ap.add_argument("--crosswalk", default=str(DEFAULT_CROSSWALK),
                    help="county_fips->cbsa_code crosswalk CSV (used when cbsa_code is absent)")
    ap.add_argument("--min-n", type=int, default=20,
                    help="min tracts per city for the within-city Spearman (default 20)")
    args = ap.parse_args(argv)

    df = load_predictions(args.predictions, split=args.split, crosswalk_path=args.crosswalk)
    res = diagnose(df, min_n=args.min_n)
    print(format_report(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
