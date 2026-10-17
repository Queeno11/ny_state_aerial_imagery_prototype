"""Does the model move tracts whose wealth *really and steadily* changed?

The construction scatter asks whether predicted changes track ACS changes
across all tracts, where most observed change is noise or invisible. This
module narrows the question to a high-signal set: NYC tracts whose ACS wealth
score rose (or fell)

* significantly between the 2014 and 2023 releases — the panel's own z-test on
  the relative W2 score with replicate SEs (``process_acs.test_significance``),
* without reversion — monotone across the non-overlapping 5-year releases
  2013 -> 2018 -> 2023 (disjoint survey windows, so no shared respondents).
  A literal year-by-year monotonicity rule over the 13 annual releases keeps no
  tract at all: consecutive 5-year estimates share 80% of their sample yet
  still jitter by ~0.1 SD, so every tract dips somewhere. The alternative rule
  ``rule="tolerance"`` allows dips up to ``tol_se`` single-release SEs,

and, for the main group, built >= 5% new floor area in the window. If the model
does not move *those* tracts, noise and invisibility are not the explanation.

Changes are read from a construction-pairs table
(``evaluation.part_f_construction_scatter`` output): percentile-rank changes
(``*_prob_*``, Hazen ranks within each release) and GB2 Δlog dollars. Selecting
tracts on a large observed rise also selects on positive ACS noise (winner's
curse), so observed changes in the selected groups are biased upward and the
pass-through ratio is a lower bound; within-group correlations suffer range
restriction. The group means and the AUC against same-construction
non-changers are the informative statistics.

Run:  python -m src.consistent_change   (writes paper/latex/Figures/NYC_F_consistent_change.csv)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NYC_PREFIXES = ("36005", "36047", "36061", "36081", "36085")
INDEPENDENT_RELEASES = (2013, 2018, 2023)
TEST_YEARS = (2014, 2023)
DEFAULT_ALPHA = 0.05
DEFAULT_N_BOOT = 1000

# ─── trajectory classification ────────────────────────────────────────────────


def max_drawdown(scores: np.ndarray) -> np.ndarray:
    """Per row, the largest drop below the running maximum (0 if monotone up).
    NaN releases are skipped; an all-NaN row gives NaN."""
    s = np.asarray(scores, float)
    filled = np.where(np.isnan(s), -np.inf, s)
    run = np.maximum.accumulate(filled, axis=1)
    gap = np.where(np.isnan(s), np.nan, run - s)
    gap = np.where(np.isinf(gap), np.nan, gap)
    out = np.full(len(s), np.nan)
    has = ~np.all(np.isnan(gap), axis=1)
    out[has] = np.nanmax(gap[has], axis=1)
    return out


def max_drawup(scores: np.ndarray) -> np.ndarray:
    """Per row, the largest rise above the running minimum (0 if monotone down)."""
    return max_drawdown(-np.asarray(scores, float))


def consistent_direction(scores: pd.DataFrame, diff: pd.Series, pvalue: pd.Series,
                         se: pd.Series | None = None, alpha: float = DEFAULT_ALPHA,
                         rule: str = "independent",
                         releases=INDEPENDENT_RELEASES, tol_se: float = 1.0) -> pd.Series:
    """'up' / 'down' / 'none' per tract (index of ``scores``).

    Significant: ``pvalue < alpha`` with the sign of ``diff``. Consistent:
    ``rule="independent"`` — strictly monotone in that direction across
    ``releases`` (columns of ``scores``); ``rule="tolerance"`` — over every
    column of ``scores`` no move against the direction larger than ``tol_se``
    times the tract's single-release ``se``.
    """
    diff = diff.reindex(scores.index)
    pvalue = pvalue.reindex(scores.index)
    sig = pvalue < alpha
    if rule == "independent":
        sub = scores[list(releases)].to_numpy(float)
        steps = np.diff(sub, axis=1)
        mono_up = np.all(steps > 0, axis=1)
        mono_down = np.all(steps < 0, axis=1)
    elif rule == "tolerance":
        if se is None:
            raise ValueError("rule='tolerance' needs the tract SEs")
        tol = tol_se * se.reindex(scores.index).to_numpy(float)
        mat = scores.to_numpy(float)
        mono_up = max_drawdown(mat) <= tol
        mono_down = max_drawup(mat) <= tol
    else:
        raise ValueError(f"rule must be 'independent' or 'tolerance', got {rule!r}")
    out = pd.Series("none", index=scores.index)
    out[(sig & (diff > 0)).to_numpy() & mono_up] = "up"
    out[(sig & (diff < 0)).to_numpy() & mono_down] = "down"
    return out


# ─── per-group statistics ─────────────────────────────────────────────────────


def add_percentile_changes(pairs: pd.DataFrame) -> pd.DataFrame:
    """Percentile-point changes of the label and prediction ranks (0-100 scale)."""
    out = pairs.copy()
    out["d_pct_label"] = 100 * (out["label_prob_post"] - out["label_prob_pre"])
    out["d_pct_pred"] = 100 * (out["pred_prob_post"] - out["pred_prob_pre"])
    return out


def auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(A > B) + 0.5 P(A = B) — Mann-Whitney AUC of sample ``a`` over ``b``."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([a, b]))
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def _boot_ci(values: tuple, stat, n_boot: int, seed: int, alpha: float = 0.05):
    n = len(values[0])
    if n < 5 or n_boot <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        draws.append(stat(*(v[i] for v in values)))
    draws = np.asarray(draws, float)
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return np.nan, np.nan
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def group_summary(sub: pd.DataFrame, reference: pd.DataFrame | None = None,
                  n_boot: int = DEFAULT_N_BOOT, seed: int = 0) -> dict:
    """Observed vs predicted change for one group of tracts.

    ``reference`` (same columns) is the comparison set for the AUCs — e.g. the
    same-construction tracts whose wealth did not consistently change.
    """
    from scipy.stats import binomtest, spearmanr, wilcoxon

    row = {"n": int(len(sub)),
           "n_train": int((sub.get("split_group") == "train").sum()),
           "n_heldout": int((sub.get("split_group") == "heldout").sum())}
    if len(sub) < 3:
        return row
    dpo, dpp = sub["d_pct_label"].to_numpy(float), sub["d_pct_pred"].to_numpy(float)
    dlo, dlp = sub["dlog_label_usd"].to_numpy(float), sub["dlog_pred_usd"].to_numpy(float)
    up = int((dpp > 0).sum())
    row.update({
        "mean_dpct_obs": float(np.mean(dpo)), "mean_dpct_pred": float(np.mean(dpp)),
        "median_dpct_obs": float(np.median(dpo)), "median_dpct_pred": float(np.median(dpp)),
        "mean_dlog_obs": float(np.nanmean(dlo)), "mean_dlog_pred": float(np.nanmean(dlp)),
        "median_dlog_obs": float(np.nanmedian(dlo)), "median_dlog_pred": float(np.nanmedian(dlp)),
        "pass_through_pct": (float(np.mean(dpp) / np.mean(dpo))
                             if abs(np.mean(dpo)) > 1e-9 else np.nan),
        "share_pred_up": up / len(sub),
        "sign_test_p": float(binomtest(up, len(sub), 0.5).pvalue),
        "wilcoxon_pred_p": (float(wilcoxon(dpp).pvalue) if np.any(dpp != 0) else np.nan),
        "spearman_pct": float(spearmanr(dpo, dpp)[0]),
        "spearman_dlog": float(spearmanr(dlo, dlp, nan_policy="omit")[0]),
    })
    lo, hi = _boot_ci((dpo, dpp), lambda a, b: spearmanr(a, b)[0], n_boot, seed)
    row["spearman_pct_ci_lo"], row["spearman_pct_ci_hi"] = lo, hi
    lo, hi = _boot_ci((dpp,), np.mean, n_boot, seed)
    row["mean_dpct_pred_ci_lo"], row["mean_dpct_pred_ci_hi"] = lo, hi
    if reference is not None and len(reference) >= 3:
        row["auc_pred_vs_ref"] = auc(dpp, reference["d_pct_pred"])
        row["auc_obs_vs_ref"] = auc(dpo, reference["d_pct_label"])
        row["ref_mean_dpct_pred"] = float(reference["d_pct_pred"].mean())
        row["ref_mean_dpct_obs"] = float(reference["d_pct_label"].mean())
    return row


# Groups: (name, direction, construction flag, reference group name)
GROUPS = (
    ("up & built>=5%", "up", "in_change_5pct", "none & built>=5%"),
    ("up & stable", "up", "in_stable", "none & stable"),
    ("down & built>=5%", "down", "in_change_5pct", "none & built>=5%"),
    ("down & stable", "down", "in_stable", "none & stable"),
    ("none & built>=5%", "none", "in_change_5pct", None),
    ("none & stable", "none", "in_stable", None),
)


def consistent_change_table(pairs: pd.DataFrame, direction: pd.Series,
                            n_boot: int = DEFAULT_N_BOOT, seed: int = 0,
                            by_split: bool = True) -> pd.DataFrame:
    """One row per (group, sample) with :func:`group_summary`.

    ``pairs`` needs GEOID, the construction flags, ``split_group`` and the
    percentile / Δlog columns; ``direction`` is indexed by GEOID.
    """
    p = add_percentile_changes(pairs)
    p["direction"] = p["GEOID"].map(direction).fillna("none")
    samples = [("all", np.ones(len(p), bool))]
    if by_split:
        samples += [(s, (p["split_group"] == s).to_numpy()) for s in ("train", "heldout")]
    masks = {name: (p["direction"] == d).to_numpy() & p[flag].astype(bool).to_numpy()
             for name, d, flag, _ in GROUPS}
    rows = []
    for sample, sel in samples:
        for name, _, _, ref in GROUPS:
            sub = p[masks[name] & sel]
            ref_df = p[masks[ref] & sel] if ref else None
            rows.append({"group": name, "sample": sample,
                         **group_summary(sub, ref_df, n_boot=n_boot, seed=seed)})
    return pd.DataFrame(rows)


# ─── data loading ─────────────────────────────────────────────────────────────


def load_nyc_trajectories(processed_dir: Path, indicator: str = "W2_r5",
                          years=range(2011, 2024)) -> pd.DataFrame:
    """Wide NYC panel: relative scores per release, mean single-release SE,
    and the 2014->2023 test's diff and p-value. Index GEOID (11 chars)."""
    import pyarrow as pa
    from src.data import indicators
    from src.data.process_acs import PANEL_YEARS

    var = indicators.token_to_var(indicator)
    path = Path(processed_dir) / f"us_metros_panel_{PANEL_YEARS[0]}_{PANEL_YEARS[-1]}.feather"
    present = set(pa.ipc.open_file(path).schema.names)
    years = [int(y) for y in years if indicators.score_col(indicator, y) in present]
    t0, t1 = TEST_YEARS
    cols = (["geoid_2023"] + [indicators.score_col(indicator, y) for y in years]
            + [f"Rel_SE_{var}_{t0}", f"Rel_SE_{var}_{t1}",
               f"diff_{var}_{t0}_{t1}", f"pvalue_{var}_{t0}_{t1}"])
    missing = [c for c in cols if c not in present]
    if missing:
        raise KeyError(f"panel lacks {missing}")
    pan = pd.read_feather(path, columns=cols)
    pan["GEOID"] = pan["geoid_2023"].astype(str).str.zfill(11)
    pan = pan[pan["GEOID"].str[:5].isin(NYC_PREFIXES)].drop_duplicates("GEOID").set_index("GEOID")
    out = pd.DataFrame({y: pan[indicators.score_col(indicator, y)].astype(float) for y in years},
                       index=pan.index)
    out["se"] = pan[[f"Rel_SE_{var}_{t0}", f"Rel_SE_{var}_{t1}"]].astype(float).mean(axis=1)
    out["diff"] = pan[f"diff_{var}_{t0}_{t1}"].astype(float)
    out["pvalue"] = pan[f"pvalue_{var}_{t0}_{t1}"].astype(float)
    return out


def classify(traj: pd.DataFrame, alpha: float = DEFAULT_ALPHA, rule: str = "independent",
             tol_se: float = 1.0) -> pd.Series:
    """:func:`consistent_direction` on a :func:`load_nyc_trajectories` frame."""
    score_cols = [c for c in traj.columns if isinstance(c, (int, np.integer))]
    return consistent_direction(traj[score_cols], traj["diff"], traj["pvalue"],
                                se=traj["se"], alpha=alpha, rule=rule, tol_se=tol_se)


def with_flags(pairs: pd.DataFrame, flags_from: pd.DataFrame | None) -> pd.DataFrame:
    """Replace the construction flags of ``pairs`` with those of ``flags_from``
    (by GEOID), so tables from different windows share one tract definition."""
    if flags_from is None:
        return pairs
    cols = ["in_stable", "in_change_1pct", "in_change_5pct", "in_change_10pct"]
    f = flags_from.set_index("GEOID")[cols]
    out = pairs.drop(columns=[c for c in cols if c in pairs.columns])
    out = out.merge(f, left_on="GEOID", right_index=True, how="left")
    out[cols] = out[cols].eq(True)
    return out


def main() -> None:
    from src.utils.paths import PROCESSED_DATA_DIR

    fig = Path(__file__).resolve().parents[1] / "paper" / "latex" / "Figures"
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--figures", type=Path, default=fig)
    ap.add_argument("--processed", type=Path, default=PROCESSED_DATA_DIR)
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    read = lambda f: pd.read_csv(args.figures / f, dtype={"GEOID": str})
    ten = read("NYC_F_construction_pairs.csv")
    sources = {
        "zarr 10y (ACS 2013->2023)": ten,
        "NAIP 5y (ACS 2018->2023)": with_flags(read("NYC_F_construction_pairs_naip5y.csv"), ten),
        "zarr 5y (ACS 2018->2023)": with_flags(read("NYC_F_construction_pairs_zarr5y.csv"), ten),
    }
    traj = load_nyc_trajectories(args.processed)
    definitions = {
        "independent p<0.05": classify(traj, 0.05, "independent"),
        "independent p<0.10": classify(traj, 0.10, "independent"),
        "tolerance 1SE p<0.05": classify(traj, 0.05, "tolerance", 1.0),
    }
    frames = []
    for dname, direction in definitions.items():
        print(f"{dname}: up {int((direction == 'up').sum())}, down {int((direction == 'down').sum())}")
        for sname, pairs in sources.items():
            t = consistent_change_table(pairs, direction, n_boot=args.n_boot)
            frames.append(t.assign(definition=dname, source=sname))
    table = pd.concat(frames, ignore_index=True)
    first = ["source", "definition", "group", "sample"]
    table = table[first + [c for c in table.columns if c not in first]]
    out = args.out or args.figures / "NYC_F_consistent_change.csv"
    table.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
