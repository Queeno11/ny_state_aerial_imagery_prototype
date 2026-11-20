#!/usr/bin/env python3
"""Fast exact E1 with an additive baseline: W = a_ct + b_ct exp(0.6 R).

Recomputes only the log + city-year FE arm, by exact least squares on the suite's own
DGP, for a sweep of baseline sizes s in a_ct = s(.6 + .3 tanh(z_c1) + .1 t). The
multiplier b_ct and every draw (features, noise, splits, test pairs) are those of the
finished E1 run, so s=0 must reproduce its log_cell_fe errors exactly (checked).
Ranking errors are reused from that run: every map here preserves all within-cell
orderings, so a ranking fit is literally the same fit under every s.
The correctly specified ML benchmark is fitted from the same training labels
with known transformation/error mechanisms. Its fit is also invariant to s.

    python e1_additive_exact.py [--scales 0 1 5 10 25 100] [--out e1_additive_exact]

Writes replications.csv, summary.csv (suite schema) and paired_replications.csv.
The full trained version of this experiment is configs/e1_additive.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ordinal_mc.config import merge, preset, validate
from ordinal_mc.core import misranking
from ordinal_mc.design import PopulationDGP, make_design
from ordinal_mc.experiments import scenario
from ordinal_mc.likelihood import fit_ml

SUITE_DIR = Path(__file__).resolve().parent
CONFIG = SUITE_DIR / "configs" / "e1_additive.json"
RANKING_RUN = SUITE_DIR / "overnight_measurement"
METRIC = "unseen_within_latent"
BASE_SCENARIO = scenario(psi_stage="multiplier", psi_strength=1., positive_labels=True, composition=0.)


def load_config(path: Path = CONFIG) -> SimpleNamespace:
    """Suite defaults merged with a JSON config, validated like run.py does."""
    d = preset("smoke")
    merge(d, json.loads(Path(path).read_text()))
    validate(d)
    return SimpleNamespace(**d)


def additive_baseline(contexts: np.ndarray, c: np.ndarray, t: np.ndarray, scale: float) -> np.ndarray:
    """a_ct = s(.6 + .3 tanh(z_c1) + .1 t), the suite's bounded location coefficient."""
    return scale * (.6 + .3 * np.tanh(contexts[c, 0]) + .1 * t)


def demean_by_cell(values: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Subtract cell means (rows of a vector or matrix)."""
    out = np.asarray(values, dtype=float).copy()
    for cell in np.unique(cells):
        keep = cells == cell
        out[keep] -= out[keep].mean(0)
    return out


def log_fe_beta(x: np.ndarray, log_w: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Common slope of log W on X with free cell intercepts (within estimator)."""
    return np.linalg.pinv(demean_by_cell(x, cells), rcond=1e-10) @ demean_by_cell(log_w, cells)


def log_fe_risk(design: dict, scale: float) -> float:
    """Within-cell misranking in unseen cities of the log + FE fit under baseline size s."""
    tr = design["train"]
    w = additive_baseline(design["contexts"], tr["c"], tr["t"], scale) + tr["b"] * np.exp(.6 * tr["r"])
    beta = log_fe_beta(tr["x"], np.log(w), tr["cell"])
    test, pairs = design["tests"]["unseen"]
    i, j = pairs["within"]
    score = test["x"] @ beta
    return misranking(score[j] - score[i], test["f"][j] - test["f"][i])


def run(cfg: SimpleNamespace, scales: list[float]) -> pd.DataFrame:
    """One row per (C, replication, s), re-using the finished run's draws."""
    dgp = PopulationDGP(cfg)
    rows = []
    for cities in cfg.city_grid:
        cfg.cities = cities
        for rep in range(1, cfg.reps + 1):
            design = make_design(cfg, dgp, rep, cfg.n[0], cfg.truths[0], BASE_SCENARIO)
            predict,info=fit_ml(design,cfg.truths[0],BASE_SCENARIO,dgp)
            test,pairs=design['tests']['unseen']; i,j=pairs['within']; score=predict(test['x'])
            ml_risk=misranking(score[j]-score[i],test['f'][j]-test['f'][i])
            rows += [dict(cities=cities, rep=rep, scale=s, risk=log_fe_risk(design, s),
                          ml_risk=ml_risk,ml_success=info['ml_success'],ml_message=info['ml_message']) for s in scales]
    return pd.DataFrame(rows)


def ranking_fits(run_dir: Path = RANKING_RUN) -> pd.DataFrame:
    """rank_common and log_cell_fe errors of the finished E1 run (multiplier map, no composition)."""
    fits = pd.read_csv(Path(run_dir) / "selected_fits.csv")
    keep = (fits.psi_stage == "multiplier") & (fits.composition == 0)
    return fits.loc[keep, ["cities", "rep", "method", METRIC]]


def check_baseline(records: pd.DataFrame, fits: pd.DataFrame, tol: float = 1e-10) -> float:
    """s=0 must reproduce the trained log_cell_fe errors; returns the max abs gap."""
    base = records[records.scale == 0].merge(fits[fits.method == "log_cell_fe"], on=["cities", "rep"])
    if len(base) != (records.scale == 0).sum() or base.empty:
        raise ValueError("finished run does not cover every (C, rep) of this diagnostic")
    gap = float(np.max(np.abs(base.risk - base[METRIC])))
    if gap > tol:
        raise ValueError(f"s=0 does not reproduce the trained log + FE errors (max gap {gap:.2e})")
    return gap


def _stage(scale: float) -> tuple[str, float]:
    return ("multiplier", 1.0) if scale == 0 else ("additive", float(scale))


def summarize(records: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """Suite-schema summary: mean, MC SE and mean +/- 1.96 SE per method, map and C.

    Ranking rows are repeated for every s: the fit is identical under every map here.
    """
    rank = fits[fits.method == "rank_common"].rename(columns={METRIC: "risk"})
    parts = []
    for s in sorted(records.scale.unique()):
        stage, strength = _stage(s)
        methods=[("log_cell_fe", records[records.scale == s]), ("rank_common", rank)]
        if 'ml_risk' in records:
            methods.append(('ml_correct',records[records.scale==s].drop(columns='risk').rename(columns={'ml_risk':'risk'})))
        for method, df in methods:
            g = df.groupby("cities").risk.agg(["mean", "std", "count"]).reset_index()
            g = g.rename(columns={"std": "mc_sd", "count": "repetitions"})
            g["mc_se"] = g.mc_sd / np.sqrt(g.repetitions)
            g["ci95_low"], g["ci95_high"] = g["mean"] - 1.96 * g.mc_se, g["mean"] + 1.96 * g.mc_se
            g = g.assign(method=method, psi_stage=stage, psi_strength=strength, composition=0.0, metric=METRIC)
            parts.append(g)
    return pd.concat(parts, ignore_index=True)


def paired(records: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """Per-replication log + FE minus ranking error (pp); positive = ranking better."""
    rank = fits[fits.method == "rank_common"][["cities", "rep", METRIC]]
    out = records.merge(rank, on=["cities", "rep"])
    out["gap_pp"] = 100 * (out.risk - out[METRIC])
    return out.rename(columns={METRIC: "rank_risk"})


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scales", nargs="+", type=float, default=[0, 1, 5, 10, 25, 100])
    ap.add_argument("--out", type=Path, default=SUITE_DIR / "e1_additive_exact")
    ap.add_argument("--ranking-run", type=Path, default=RANKING_RUN)
    args = ap.parse_args(argv)
    if 0 not in args.scales:
        args.scales = [0.0] + args.scales  # the reproduction check needs s=0
    cfg = load_config()
    fits = ranking_fits(args.ranking_run)
    records = run(cfg, args.scales)
    gap = check_baseline(records, fits)
    args.out.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.out / "replications.csv", index=False)
    summary = summarize(records, fits)
    summary.to_csv(args.out / "summary.csv", index=False)
    paired(records, fits).to_csv(args.out / "paired_replications.csv", index=False)
    table = records.groupby(["cities", "scale"]).risk.mean().unstack() * 100
    print(f"s=0 reproduces the trained log + FE errors (max gap {gap:.1e})")
    print("log + FE misranking (%) by baseline size s (s=0 is the multiplier map):")
    print(table.round(3).to_string())
    print(f"written to {args.out}")
    return args.out


if __name__ == "__main__":
    main()
