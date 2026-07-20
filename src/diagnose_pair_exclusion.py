"""Compare pair-exclusion rules for the in-batch ranking loss.

Question: if the permanent curriculum filter `bin_diff == 0` (same national
label quintile -> pair excluded) were replaced by an ACS-noise rule
`|z_k - z_l| < k * sqrt(SE_k^2 + SE_l^2)` (observed sign not statistically
reliable -> pair excluded), how many pairs would each rule drop, and where?

Pairs are sampled EXACTLY like training sees them (`main.HybridBatchSampler`):
a (year, CBSA) pool is chosen with probability proportional to pool size
(pools below `max(2, batch_size_cs // 4)` rows are dropped), `batch_size_cs`
tracts are drawn uniformly without replacement from the pool (the training
frame holds 1 building per tract, so tract draws == building draws), and every
unordered pair inside the batch is a candidate ranking pair. Only train-split
CBSAs and the training panel years enter the universe.

Statistical assumptions of the SE rule (documented for the record):
  1. z_hat_i = z_i + eps_i, eps_i ~ N(0, SE_i^2): `Rel_SE_<ind>_{acs_year}`
     is the delta-method SE of the within-CBSA z-scored indicator (already on
     the label scale, MOE/1.645 propagated by process_acs).
  2. eps_k independent of eps_l across tracts, so Var(dz) = SE_k^2 + SE_l^2.
     Ignores the shared estimated metro mean/SD inside the z-score (small,
     common component -> the rule is mildly conservative).
  3. Under H0 (z_k == z_l), dz ~ N(0, SE_k^2 + SE_l^2); keeping only
     |dz| >= 2*sqrt(.) trains on pairs whose sign is reliable at ~95.4%.
  4. SE vintage = nearest AVAILABLE SE year. The panel only carries the
     delta-method wealth SEs for 2014 and 2023, so each panel year borrows
     the closer of the two; tract SEs track ACS sample size and move slowly,
     so this is an approximation, not an identity.

Run:  python src/diagnose_pair_exclusion.py  [--batches 20000] [--seed 825]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import indicators

# NOTE: deliberately not importing src.utils.paths — it hard-fails when .env
# is unreadable (sandboxed sessions), and this script only needs the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAIR_LABELS = PROJECT_ROOT / "data/processed/pair_labels_ms_us_W2_r5_years2010-2024_all.parquet"
PANEL = PROJECT_ROOT / "data/processed/us_metros_panel_2011_2023.feather"
SPLITS = PROJECT_ROOT / "data/processed/cbsa_splits.feather"
CROSSWALK = PROJECT_ROOT / "data/external/cbsa/county_cbsa_crosswalk.csv"

ACS_YEARS = list(range(2011, 2024))
NYC_CBSA = "35620"
BOROUGHS = {"36005": "Bronx", "36047": "Brooklyn", "36061": "Manhattan",
            "36081": "Queens", "36085": "StatenIsl"}
K_LIST = (1.0, 1.645, 2.0)


def closest_acs_year(year: int, candidates=None) -> int:
    return min(candidates or ACS_YEARS, key=lambda y: abs(y - year))


def load_universe(indicator: str = "W2_r5") -> pd.DataFrame:
    """Training pair universe: [GEOID, year, cbsa_code, bracket, z, score_bin, se].

    Train-split CBSAs only (the sampler never sees val/test pools). The pair
    label frame already excludes zero-building tracts and holds one row per
    (tract, panel year) == one training building per tract.
    """
    lab = pd.read_parquet(PAIR_LABELS)
    lab["GEOID"] = lab["GEOID"].astype(str).str.zfill(11)

    xw = pd.read_csv(CROSSWALK, dtype={"county_fips": str, "cbsa_code": str})
    xw["county_fips"] = xw["county_fips"].str.zfill(5)
    xw = xw[["county_fips", "cbsa_code"]].drop_duplicates("county_fips")
    lab["county_fips"] = lab["GEOID"].str[:5]
    lab = lab.merge(xw, on="county_fips", how="left").dropna(subset=["cbsa_code"])

    splits = pd.read_feather(SPLITS)
    splits["cbsa_code"] = splits["cbsa_code"].astype(str)
    train = splits.loc[splits["split"] == "train", ["cbsa_code", "bracket", "cbsa_title"]]
    lab = lab.merge(train, on="cbsa_code", how="inner")

    # SE at the closest ACS year, from the wide panel.
    panel = pd.read_feather(PANEL)
    gcol = [c for c in panel.columns if c.startswith("geoid_")][0]
    panel = panel.rename(columns={gcol: "GEOID"})
    panel["GEOID"] = panel["GEOID"].astype(str).str.zfill(11)
    # INDICATORS maps token -> score-column prefix ("Rel_Score_W2_i_r5pct");
    # the matching SE columns swap that prefix for "Rel_SE_". The wealth SEs
    # exist only for a subset of years (2014/2023) — use the nearest vintage.
    se_prefix = indicators.INDICATORS[indicator].replace("Rel_Score_", "Rel_SE_")
    se_years = sorted(int(c.rsplit("_", 1)[1]) for c in panel.columns
                      if c.startswith(se_prefix + "_"))
    if not se_years:
        raise ValueError(f"No '{se_prefix}_*' columns in the panel")

    frames = []
    for year, sub in lab.groupby("year"):
        se_col = f"{se_prefix}_{closest_acs_year(int(year), se_years)}"
        sub = sub.merge(panel[["GEOID", se_col]].rename(columns={se_col: "se"}),
                        on="GEOID", how="left")
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True).rename(columns={"Rel_Score": "z"})
    out = out.dropna(subset=["z", "se", "score_bin"])
    out = out[out["score_bin"] >= 0]  # -1 = unbinned sentinel
    # Counterfactual binning: quintiles WITHIN each (year, CBSA) pool.
    # Unweighted tract qcut is the right analog here: the training frame holds
    # one building per tract, so tract rows == training rows (the shipped
    # national bins are building-count weighted for the same reason).
    out["score_bin_cbsa"] = (
        out.groupby(["year", "cbsa_code"])["z"]
        .transform(lambda x: pd.qcut(x, q=5, labels=False, duplicates="drop"))
    )
    return out[["GEOID", "year", "cbsa_code", "bracket", "county_fips", "z",
                "score_bin", "score_bin_cbsa", "se"]].reset_index(drop=True)


def build_pools(df: pd.DataFrame, batch_size_cs: int):
    """(year, cbsa) pools >= min size, as row-index arrays + sampling weights."""
    min_pool = max(2, batch_size_cs // 4)
    pools = {k: g.index.to_numpy() for k, g in df.groupby(["year", "cbsa_code"])
             if len(g) >= min_pool}
    keys = list(pools)
    weights = np.array([len(pools[k]) for k in keys], dtype=float)
    return pools, keys, weights / weights.sum()


def sample_batch_pairs(df, pools, keys, probs, batch_size_cs, rng):
    """One simulated batch -> DataFrame of its unordered pairs with both rules.

    Columns: cbsa_code, bracket, bin_excl, se_excl_<k>, same_county, county.
    """
    key = keys[rng.choice(len(keys), p=probs)]
    pool = pools[key]
    take = min(batch_size_cs, len(pool))
    rows = df.loc[rng.choice(pool, take, replace=False)]

    z = rows["z"].to_numpy()
    se = rows["se"].to_numpy()
    b = rows["score_bin"].to_numpy()
    bc = rows["score_bin_cbsa"].to_numpy()
    cty = rows["county_fips"].to_numpy()
    iu, il = np.triu_indices(take, 1)

    dz = np.abs(z[iu] - z[il])
    thr = np.sqrt(se[iu] ** 2 + se[il] ** 2)
    out = {
        "cbsa_code": key[1],
        "bracket": rows["bracket"].iloc[0],
        "bin_excl": (b[iu] == b[il]),
        "bin_cbsa_excl": (bc[iu] == bc[il]),
        "same_county": cty[iu] == cty[il],
        "county_k": cty[iu],
        "county_l": cty[il],
    }
    for k in K_LIST:
        out[f"se_excl_{k:g}"] = dz < k * thr
    return pd.DataFrame(out)


def simulate(df, batch_size_cs=32, n_batches=20000, seed=825):
    pools, keys, probs = build_pools(df, batch_size_cs)
    rng = np.random.default_rng(seed)
    parts = [sample_batch_pairs(df, pools, keys, probs, batch_size_cs, rng)
             for _ in range(n_batches)]
    return pd.concat(parts, ignore_index=True)


def bootstrap_ci(x: np.ndarray, n_boot=1000, seed=0):
    """Percentile bootstrap 95% CI for the mean of a 0/1 array."""
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.array([x[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return x.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def report(pairs: pd.DataFrame):
    def line(name, sub):
        if len(sub) == 0:
            return
        parts = []
        for label, col in [("q-natl", "bin_excl"), ("q-cbsa", "bin_cbsa_excl")]:
            m, lo, hi = bootstrap_ci(sub[col].to_numpy())
            parts.append(f"{label}: {m:6.1%} [{lo:.1%},{hi:.1%}]")
        for k in K_LIST:
            m, lo, hi = bootstrap_ci(sub[f"se_excl_{k:g}"].to_numpy())
            parts.append(f"SE k={k:g}: {m:6.1%} [{lo:.1%},{hi:.1%}]")
        print(f"{name:24s} n={len(sub):>9,}  " + "  ".join(parts))

    print("=== P(pair EXCLUDED) under each rule — HybridBatchSampler pair distribution ===")
    line("ALL train pairs", pairs)
    for br in ("mega", "large", "medium", "small"):
        line(f"  bracket={br}", pairs[pairs["bracket"] == br])

    nyc = pairs[pairs["cbsa_code"] == NYC_CBSA]
    boro = nyc[nyc["county_k"].isin(BOROUGHS) & nyc["county_l"].isin(BOROUGHS)]
    line("NYC CBSA (all pairs)", nyc)
    line("  NYC boro-vs-boro", boro)
    for cf, name in BOROUGHS.items():
        line(f"  {name}-internal",
             nyc[(nyc["county_k"] == cf) & (nyc["county_l"] == cf)])

    print("\n=== joint classification vs today's rule (all train pairs) ===")
    q = pairs["bin_excl"].to_numpy()
    for rule, col in [("SE k=2", "se_excl_2"), ("q-cbsa", "bin_cbsa_excl")]:
        s = pairs[col].to_numpy()
        print(f"[{rule}] kept by both {(~q & ~s).mean():6.1%} | excluded by both "
              f"{(q & s).mean():6.1%} | UNLOCKED vs today {(q & ~s).mean():6.1%} | "
              f"newly dropped {(~q & s).mean():6.1%}")
    if len(boro):
        bq = boro["bin_excl"].to_numpy()
        for rule, col in [("SE k=2", "se_excl_2"), ("q-cbsa", "bin_cbsa_excl")]:
            bs = boro[col].to_numpy()
            print(f"[{rule}] NYC boro-boro: UNLOCKED {(bq & ~bs).mean():6.1%} | "
                  f"newly dropped {(~bq & bs).mean():6.1%}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=825)
    args = ap.parse_args(argv)

    df = load_universe()
    n_pools = df.groupby(["year", "cbsa_code"]).ngroups
    print(f"universe: {len(df):,} (tract, year) rows | {df.cbsa_code.nunique()} train CBSAs "
          f"| {n_pools} (year, CBSA) pools | median SE={df.se.median():.3f}")
    pairs = simulate(df, args.batch_size, args.batches, args.seed)
    report(pairs)


if __name__ == "__main__":
    main()
