"""Bootstrap the small-city within-Spearman CI vs (n_tracts, n_years).

Uses the honest offline-eval predictions (~/outputs/offline_small_city_eval.parquet,
116+112 tracts x 3 years, produced by eval_small_city_offline.py) as ground
truth, then resamples TRACTS with replacement to measure how the CI on the
tract-weighted within-Spearman shrinks with tract coverage — and how little a
2nd/3rd flight year adds. Answers: minimum tracts/year budget for a +-0.15 CI.

Crop cost of a config = n_tracts * n_years (1 building/tract). Vectorized:
each city is a tract x year matrix of preds/labels (1 building/tract), so a
bootstrap draw is a row slice + one Spearman per year column.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
B = 1500
SRC = "/home/abbatenicolas/outputs/offline_small_city_eval.parquet"
NAMES = {29540: "Lancaster", 33700: "Modesto"}


def _rank(a):
    """Column-wise ranks (average ties ignored — preds are continuous)."""
    order = a.argsort(axis=0)
    ranks = np.empty_like(order, dtype=float)
    idx = np.arange(a.shape[0])[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(idx, a.shape).astype(float), axis=0)
    return ranks


def spearman_cols(pred_mat, lab_mat):
    """Spearman per column between two (n_tracts, n_years) matrices."""
    rp, rl = _rank(pred_mat), _rank(lab_mat)
    rp -= rp.mean(0); rl -= rl.mean(0)
    num = (rp * rl).sum(0)
    den = np.sqrt((rp**2).sum(0) * (rl**2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def build_matrices(cdf, n_years):
    """(pred, label) tract x year matrices for this city's first ``n_years``
    flight years, over tracts present in all of them (so a row slice is
    well-defined). Each city uses its OWN flight years (no shared grid)."""
    years = sorted(cdf["year"].unique())[:n_years]
    piv_p = cdf.pivot_table(index="GEOID", columns="year", values="pred", aggfunc="mean")
    piv_l = cdf.pivot_table(index="GEOID", columns="year", values="label", aggfunc="mean")
    piv_p, piv_l = piv_p[years].dropna(), piv_l[years].dropna()
    common = piv_p.index.intersection(piv_l.index)
    return piv_p.loc[common].to_numpy(), piv_l.loc[common].to_numpy()


def city_ci(cdf, n_tracts, n_years, b=B):
    pred, lab = build_matrices(cdf, n_years)
    n = pred.shape[0]
    year_w = np.array([np.isfinite(pred[:, j]).sum() for j in range(pred.shape[1])], float)
    point = np.average(spearman_cols(pred, lab), weights=year_w)
    draws = np.empty(b)
    for i in range(b):
        idx = RNG.integers(0, n, size=n_tracts)
        rhos = spearman_cols(pred[idx], lab[idx])
        draws[i] = np.average(rhos, weights=np.full(len(rhos), n_tracts))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, (hi - lo) / 2


def bracket_ci(cities, n_tracts, n_years, b=B):
    mats = {c: build_matrices(g, n_years) for c, g in cities.items()}
    draws = np.empty(b)
    for i in range(b):
        rho_num, rho_den = [], []
        for c, (pred, lab) in mats.items():
            idx = RNG.integers(0, pred.shape[0], size=n_tracts)
            rhos = spearman_cols(pred[idx], lab[idx])
            for r in rhos:
                rho_num.append(r * n_tracts); rho_den.append(n_tracts)
        draws[i] = np.nansum(rho_num) / np.nansum(rho_den)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return (hi - lo) / 2


def main():
    df = pd.read_parquet(SRC)
    print(f"loaded {len(df)} rows; "
          f"flight years/city="
          f"{ {NAMES[c]: sorted(g['year'].unique()) for c, g in df.groupby('cbsa')} }; "
          f"tracts/city={ {NAMES[c]: g['GEOID'].nunique() for c, g in df.groupby('cbsa')} }\n")

    grid = [20, 30, 40, 60, 80, 100]
    print("Per-city 95% CI half-width (±) of within-Spearman:")
    print(f"{'city':10s} {'n_tr':>4s}   1yr      2yr      3yr     cost 1/2/3yr")
    for cbsa, cdf in df.groupby("cbsa"):
        max_tr = cdf["GEOID"].nunique()
        for nt in grid + [max_tr]:
            if nt > max_tr:
                continue
            hws = [city_ci(cdf, nt, ny)[1] for ny in (1, 2, 3)]
            pt = city_ci(cdf, nt, 3)[0]
            cost = "/".join(str(nt * ny) for ny in (1, 2, 3))
            tag = " (all,rho=%.2f)" % pt if nt == max_tr else ""
            print(f"{NAMES[cbsa]:10s} {nt:4d}  " +
                  " ".join(f"{h:6.3f}" for h in hws) + f"   {cost}{tag}")
        print()

    print("Bracket (both cities pooled) 95% CI half-width, per-city n_tracts:")
    cities = {c: g for c, g in df.groupby("cbsa")}
    print(f"{'n_tr/city':>9s}   1yr      2yr      3yr     total crops 1/2/3yr")
    for nt in grid:
        hws = [bracket_ci(cities, nt, ny) for ny in (1, 2, 3)]
        cost = "/".join(str(2 * nt * ny) for ny in (1, 2, 3))
        print(f"{nt:9d}  " + " ".join(f"{h:6.3f}" for h in hws) + f"   {cost}")


if __name__ == "__main__":
    main()
