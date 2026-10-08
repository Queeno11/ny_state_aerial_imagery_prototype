#!/usr/bin/env python3
"""Monte Carlo: does city-specific logistic calibration improve a common NN index?

QUICK START
-----------
Requires Python >= 3.10, numpy, scipy, matplotlib, and PyTorch. Install a CUDA
build of PyTorch appropriate for your computer using https://pytorch.org/get-started/locally/.
Other dependencies:  python -m pip install numpy scipy matplotlib

Default experiment (finite budget; CUDA is selected automatically):
    python rho_city_monte_carlo.py
Shorter first run:
    python rho_city_monte_carlo.py --reps 1 --steps 600 --out rho_quick
Larger Monte Carlo:
    python rho_city_monte_carlo.py --reps 10 --steps 2500 --out rho_large
Check sensitivity to large-city dominance with identical underlying seeds:
    python rho_city_monte_carlo.py --weighting pairs --out rho_pair_weighted
Optional shrinkage of city log-calibrations (zero by default):
    python rho_city_monte_carlo.py --shrinkage 0.01 --out rho_shrunk

DEFAULT DESIGN
--------------
3 replications x 2 noise scenarios x 3 estimators = 18 bounded fits, each with
1,500 optimizer steps. 120 cities; 4 years; total TRAINING observations per city
are 20, 80, 300, 1,000, or 2,000, allocated across years. The balanced default
design has 81,600 training observations. Validation/test populations are freshly
generated and equally large in every city, so small-city performance is measured
precisely without adding information to their training samples. Runtime is
hardware-dependent; increase --steps or --reps only after a first run.

DGP AND IDENTIFICATION
----------------------
* Six numerical amenities with overlapping but city-dependent distributions.
  Every city also has a 25% common-distribution mixture component. City ID is NOT
  a score-network input; it is used only for calibration and sampling.
* One strictly coordinatewise increasing cubic polynomial f(X). Its location
  and scale are fixed on a shared, unlabeled reference sample. The NN uses this
  SAME reference sample for differentiable normalization: no true f values enter
  NN training, normalization, or checkpoint selection.
* R_ict = f(X_ict) + (G_ict - EulerGamma)/rho_c, with independent standard Gumbel
  G. Differences of independent Gumbels are logistic, so exactly
      P(W_jct > W_ict | X_i, X_j, c, t) = sigmoid(rho_c * (f_j - f_i)).
  Giving each observation a logistic error would NOT yield this pairwise model.
* rho_c = clip(exp(log_sd * Z_c), 0.2, 5.0), Z_c iid N(0,1), independent of size
  and city feature centers. log_sd = 0 is the homogeneous control; 0.9 is the
  heterogeneous scenario. Common random numbers couple both scenarios.
* psi_ct(r) = exp(mu_ct + sigma_ct*r), with random mu_ct and positive, lognormally
  distributed sigma_ct. This is the familiar lognormal quantile construction
  Q_LogNormal(mu,sigma)(Phi(r)). W itself is NOT necessarily marginally lognormal,
  because R is not standard normal. Log(W) is stored to preserve ordering without
  overflow; it is exactly the same ranking supervision as W.

ESTIMATORS (same architecture, initial weights, and minibatch stream)
------------------------------------------------------------------
shared: learn ONE positive calibration, constrained equal across all cities.
city:   learn ONE positive calibration PER CITY, shared across years.
oracle: use the true rho_c while still learning f with exactly the same NN.

A learned shared scalar is a fairer homogeneous baseline than fixing rho=1:
with a normalized score, a fixed value would impose an arbitrary noise level.
There is no temporal-stability penalty in this experiment, to isolate the effect
of calibration. Years pool independent training information, not stable twins.
This is a diagnostic experiment, not a proof for the full paper's estimator.

TRAINING AND OUTPUTS
--------------------
Default: equal-city/equal-year loss; draw groups uniformly, take all unique pairs
of distinct observations within each sampled group, average within groups, then
average groups. Small groups use all their observations, without replacement.
--weighting pairs changes group selection probability to n_ct*(n_ct-1), exposing
the consequences of pooling all pairs. No outcome-dependent pair filtering.
All calibrations are constrained to [0.1, 10] for numerical stability. Optional
shrinkage is a common penalty per city; it is NOT advertised as sample-size-
adaptive hierarchical estimation. Exact fixed-reference score normalization
removes the score/calibration scale ambiguity without a variance penalty.

Outputs: metrics.csv, city_metrics.csv, histories.csv, summary.csv,
paired_differences.csv, experiment_summary.png, per-fit diagnostic PNGs and
checkpoints, and reproducible DGP metadata. Key diagnostics:
  - test latent R2 / RMSE: recovery of f, NOT noisy realized R or dollar labels;
  - one globally affine-aligned R2 (alignment fitted on validation truths ONLY
    for evaluation; never used for training, early stopping, or probabilities);
  - within-cell observed NLL, expected excess NLL (KL to true pair probability),
    probability RMSE, latent pair accuracy and cross-city latent pair accuracy;
  - counterfactual temporal-difference RMSE, with changed numerical amenities;
  - calibration recovery and performance broken down by training city size.

Checkpoint selection uses held-out observed ranking NLL only. Look at both
probabilities and index recovery: improved calibration need not improve f much,
especially under substantial overlap and a correctly shared f. With 3 reps the
Monte Carlo SE is descriptive, not strong evidence. Training convergence and
the oracle gap should be inspected before drawing conclusions.

References for implementation / DGP:
https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gumbel_r.html
https://eml.berkeley.edu/books/choice2nd/Ch03_p34-75.pdf

This file was authored and statically checked without running its simulation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F


METHODS = ("shared", "city", "oracle")
SIZE_LEVELS = np.array([20, 80, 300, 1000, 2000])
EULER_GAMMA = 0.5772156649015329
RHO_BOUNDS = (0.1, 10.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("rho_mc_results"))
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--cities", type=int, default=120)
    p.add_argument("--years", type=int, default=4)
    p.add_argument("--log-rho-sds", type=float, nargs="+", default=[0.0, 0.9])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--groups-per-step", type=int, default=16)
    p.add_argument("--units-per-group", type=int, default=32)
    p.add_argument("--reference-size", type=int, default=2048)
    p.add_argument("--val-per-cell", type=int, default=32)
    p.add_argument("--test-per-cell", type=int, default=128)
    p.add_argument("--eval-pairs-per-cell", type=int, default=128)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--rho-lr", type=float, default=0.015)
    p.add_argument("--shrinkage", type=float, default=0.0)
    p.add_argument("--weighting", choices=["city", "pairs"], default="city")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if min(a.reps, a.steps, a.eval_every, a.groups_per_step, a.eval_pairs_per_cell,
           a.threads) < 1:
        p.error("replications, steps, batch/evaluation counts and threads must be positive")
    if a.cities < 5 or not 1 <= a.years <= 10:
        p.error("use at least 5 cities and 1 to 10 years (smallest city needs >=2 per year)")
    if min(a.val_per_cell, a.test_per_cell, a.units_per_group) < 2:
        p.error("at least 2 observations are required for every pair group")
    if a.reference_size < 32 or a.width < 4 or min(a.lr, a.rho_lr) <= 0:
        p.error("invalid reference size, width, or learning rate")
    if a.shrinkage < 0 or any(s < 0 or not math.isfinite(s) for s in a.log_rho_sds):
        p.error("shrinkage and log-rho SDs must be finite and nonnegative")
    if len(set(a.log_rho_sds)) != len(a.log_rho_sds):
        p.error("log-rho SD scenarios must be distinct")
    return a


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def raw_index(x):
    """Strictly increasing in all six coordinates: positive linear/cubic terms.

    Each cubic projection has derivative 3*z**2 times a positive coefficient.
    The positive linear part makes all partial derivatives strictly positive.
    """
    w = np.array([0.45, 0.35, 0.30, 0.25, 0.20, 0.15])
    v = np.array([0.10, 0.20, 0.45, 0.30, 0.20, 0.10])
    z, q = x @ w, x @ v
    return 0.70*z + 0.18*z**3 + 0.12*q**3 + 0.10*x[:, 0]**3 + 0.05*x[:, 1]**3


def features(rng, cities, centers):
    common = rng.random(len(cities)) < 0.25
    latent = rng.normal(size=(len(cities), 6))
    latent += (~common)[:, None] * centers[cities]
    # Smooth bounded transform; every city's distribution has the same support.
    return (2.0 * np.tanh(latent / 2.0)).astype(np.float32)


def make_pairs(data, rng, pairs_per_cell):
    counts, offsets = data["counts"], data["offsets"]
    cells = np.repeat(np.arange(len(counts)), pairs_per_cell)
    n = counts[cells]
    i = rng.integers(0, n)
    # Uniform ordered distinct pair; no diagonal comparisons, no label filtering.
    j = rng.integers(0, n - 1)
    j += j >= i
    return {"i": offsets[cells] + i, "j": offsets[cells] + j,
            "cell": cells, "city": data["city"][offsets[cells] + i]}


def make_design(a, rep, log_sd):
    # Reset per replication, not per scenario: paired scenarios reuse X, G and Z.
    rng = np.random.default_rng(a.seed + 10000 * rep)
    sizes = np.resize(SIZE_LEVELS, a.cities).copy()
    rng.shuffle(sizes)
    centers = rng.normal(0, 0.80, size=(a.cities, 6))
    z_rho = rng.normal(size=a.cities)
    rho = np.exp(np.clip(log_sd*z_rho, np.log(0.2), np.log(5.0)))
    mu = rng.normal(9.0, 0.8, size=(a.cities, a.years))
    sigma = rng.lognormal(-0.65, 0.30, size=(a.cities, a.years))

    ref_cities = np.arange(a.reference_size) % a.cities
    reference = features(rng, ref_cities, centers)
    reference_truth = raw_index(reference)
    f_mean, f_sd = float(reference_truth.mean()), float(reference_truth.std())

    def truth(x):
        return (raw_index(x) - f_mean) / f_sd

    def sample(counts):
        counts = np.asarray(counts, dtype=np.int64).reshape(-1)
        offsets = np.r_[0, np.cumsum(counts)[:-1]]
        cell = np.repeat(np.arange(a.cities*a.years), counts)
        city = cell // a.years
        x = features(rng, city, centers)
        f = truth(x)
        r = f + (rng.gumbel(size=len(city)) - EULER_GAMMA) / rho[city]
        # W = exp(log_w); no need to exponentiate or risk overflow for rankings.
        log_w = mu.reshape(-1)[cell] + sigma.reshape(-1)[cell]*r
        return dict(x=x, f=f, r=r, log_w=log_w, city=city, cell=cell,
                    counts=counts, offsets=offsets)

    train_counts = np.repeat((sizes // a.years)[:, None], a.years, axis=1)
    for c in range(a.cities):
        train_counts[c, :sizes[c] % a.years] += 1
    train = sample(train_counts)
    val = sample(np.full((a.cities, a.years), a.val_per_cell))
    test = sample(np.full((a.cities, a.years), a.test_per_cell))
    val_pairs = make_pairs(val, rng, a.eval_pairs_per_cell)
    test_pairs = make_pairs(test, rng, a.eval_pairs_per_cell)
    # Changed-feature counterfactual endpoints. No outcomes are supplied to training.
    changed_x = np.clip(test["x"] + rng.normal(0, 0.25, test["x"].shape),
                        -1.999, 1.999).astype(np.float32)
    changed_f = truth(changed_x)
    # Cross-city comparisons only of the common latent f, never dollar labels.
    n_cross = a.cities*a.years*a.eval_pairs_per_cell
    c1 = rng.integers(0, a.cities, n_cross)
    c2 = rng.integers(0, a.cities-1, n_cross)
    c2 += c2 >= c1
    n_test_city = a.years*a.test_per_cell
    cross_i = c1*n_test_city + rng.integers(0, n_test_city, n_cross)
    cross_j = c2*n_test_city + rng.integers(0, n_test_city, n_cross)
    return dict(train=train, val=val, test=test, val_pairs=val_pairs,
                test_pairs=test_pairs, reference=reference, sizes=sizes, rho=rho,
                centers=centers, mu=mu, sigma=sigma, f_mean=f_mean, f_sd=f_sd,
                changed_x=changed_x, changed_f=changed_f, cross_i=cross_i,
                cross_j=cross_j)


class ScoreNN(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, 1, bias=False))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Calibrator(nn.Module):
    def __init__(self, method, true_rho, device):
        super().__init__()
        self.method = method
        self.cities = len(true_rho)
        self.register_buffer("true_rho", torch.as_tensor(true_rho, dtype=torch.float32,
                                                       device=device))
        n = 1 if method == "shared" else self.cities
        self.log_rho = nn.Parameter(torch.zeros(n, device=device),
                                    requires_grad=method != "oracle")

    def all_rho(self):
        if self.method == "oracle":
            return self.true_rho
        r = self.log_rho.exp()
        return r.expand(self.cities) if self.method == "shared" else r

    def project(self):
        with torch.no_grad():
            self.log_rho.clamp_(math.log(RHO_BOUNDS[0]), math.log(RHO_BOUNDS[1]))


def normalization(model, reference):
    raw = model(reference)
    return raw.mean(), raw.std(unbiased=False).clamp_min(1e-5)


@torch.no_grad()
def predict(model, x, reference, device):
    model.eval()
    center, scale = normalization(model, reference)
    parts = []
    for begin in range(0, len(x), 16384):
        xb = torch.as_tensor(x[begin:begin+16384], device=device)
        parts.append(((model(xb)-center)/scale).cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def pair_quantities(scores, data, pairs, rho_hat, true_rho):
    i, j, c = pairs["i"], pairs["j"], pairs["city"]
    delta = scores[j] - scores[i]
    df = data["f"][j] - data["f"][i]
    logits = rho_hat[c]*delta
    true_logits = true_rho[c]*df
    p, p_true = expit(logits), expit(true_logits)
    y = (data["log_w"][j] > data["log_w"][i]).astype(np.float64)
    nll = np.logaddexp(0, logits) - y*logits
    expected_nll = np.logaddexp(0, logits) - p_true*logits
    irreducible = np.logaddexp(0, true_logits) - p_true*true_logits
    return dict(nll=nll, kl=np.maximum(expected_nll-irreducible, 0),
                probability_sq_error=(p-p_true)**2,
                observed_accuracy=((logits > 0) == y).astype(float),
                latent_accuracy=((delta > 0) == (df > 0)).astype(float))


def r2(y, prediction):
    return float(1-np.mean((y-prediction)**2)/max(np.var(y), 1e-12))


def correlation(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if min(np.std(x), np.std(y)) > 1e-12 else float("nan")


def fit_one(a, d, method, initial, rep, log_sd, device, folder):
    model = ScoreNN(a.width).to(device)
    model.load_state_dict(copy.deepcopy(initial))
    cal = Calibrator(method, d["rho"], device)
    groups = [{"params": list(model.parameters()), "lr": a.lr, "weight_decay": 1e-5}]
    if method != "oracle":
        groups.append({"params": [cal.log_rho], "lr": a.rho_lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups)
    # Identical generator seeds guarantee identical groups and sampled units by method.
    gen = torch.Generator(device=device).manual_seed(a.seed + 10000*rep + 717)
    reference = torch.as_tensor(d["reference"], device=device)
    train = d["train"]
    x = torch.as_tensor(train["x"], device=device)
    # Keep log-labels in float64 so comparisons remain exactly as generated.
    labels = torch.as_tensor(train["log_w"], device=device)
    counts = torch.as_tensor(train["counts"], device=device)
    offsets = torch.as_tensor(train["offsets"], device=device)
    max_n = int(train["counts"].max())
    k = min(a.units_per_group, max_n)
    row = torch.arange(max_n, device=device)[None, :]
    slots = torch.arange(k, device=device)[None, :]
    pi, pj = torch.triu_indices(k, k, offset=1, device=device)
    group_probs = counts.double()*(counts.double()-1)
    group_probs /= group_probs.sum()
    histories, best_state = [], None
    best_nll, best_step = float("inf"), 0
    start = time.perf_counter()

    def validate(step, train_loss):
        nonlocal best_nll, best_state, best_step
        sv = predict(model, d["val"]["x"], reference, device)
        with torch.no_grad():
            rho_hat = cal.all_rho().cpu().numpy().copy()
        pq = pair_quantities(sv, d["val"], d["val_pairs"], rho_hat, d["rho"])
        nll = float(pq["nll"].mean())  # equal-city held-out selection, all methods
        histories.append(dict(rep=rep, log_rho_sd=log_sd, method=method,
                              weighting=a.weighting, step=step, train_bce=train_loss,
                              val_nll=nll, val_kl=float(pq["kl"].mean()),
                              val_f_r2=r2(d["val"]["f"], sv),
                              mean_rho=float(rho_hat.mean()),
                              elapsed_seconds=time.perf_counter()-start))
        if nll < best_nll:
            best_nll, best_step = nll, step
            best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(cal.state_dict()))
        print(f"  {method:6s} step {step:4d}/{a.steps}: val NLL={nll:.4f}, "
              f"f R2={histories[-1]['val_f_r2']:.3f}, mean rho={rho_hat.mean():.2f}",
              flush=True)

    validate(0, float("nan"))
    for step in range(1, a.steps+1):
        model.train()
        if a.weighting == "city":
            cells = torch.randint(len(counts), (a.groups_per_step,),
                                  generator=gen, device=device)
        else:
            cells = torch.multinomial(group_probs, a.groups_per_step,
                                      replacement=True, generator=gen)
        ns = counts[cells]
        # Uniform sampling without replacement using independent random priorities.
        keys = torch.rand((a.groups_per_step, max_n), generator=gen, device=device)
        keys = keys.masked_fill(row >= ns[:, None], float("inf"))
        local = keys.topk(k, dim=1, largest=False, sorted=True).indices
        valid_slots = slots < ns[:, None]
        safe_local = torch.minimum(local, ns[:, None]-1)
        indices = offsets[cells, None] + safe_local
        flat_x = x[indices.reshape(-1)]
        center, scale = normalization(model, reference)
        scores = ((model(flat_x)-center)/scale).reshape(a.groups_per_step, k)
        yl = labels[indices]
        targets = (yl[:, pj] > yl[:, pi]).float()
        valid_pairs = valid_slots[:, pi] & valid_slots[:, pj]
        rho = cal.all_rho()[cells // a.years]
        logits = rho[:, None]*(scores[:, pj]-scores[:, pi])
        terms = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        group_loss = (terms*valid_pairs).sum(1)/valid_pairs.sum(1).clamp_min(1)
        bce = group_loss.mean()
        loss = bce
        if method == "city" and a.shrinkage > 0:
            loss = loss + a.shrinkage*(cal.log_rho-cal.log_rho.mean()).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        cal.project()
        if step % a.eval_every == 0 or step == a.steps:
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Nonfinite training loss; reduce learning rate.")
            validate(step, float(bce.detach().cpu()))

    model.load_state_dict(best_state[0])
    cal.load_state_dict(best_state[1])
    with torch.no_grad():
        rho_hat = cal.all_rho().cpu().numpy().astype(float)
        center, scale = normalization(model, reference)
    sv = predict(model, d["val"]["x"], reference, device)
    st = predict(model, d["test"]["x"], reference, device)
    schange = predict(model, d["changed_x"], reference, device)
    # Diagnostic only: single global positive affine transformation from validation.
    slope = max(float(np.cov(sv, d["val"]["f"], ddof=0)[0, 1]
                      / max(np.var(sv), 1e-12)), 1e-8)
    intercept = float(d["val"]["f"].mean()-slope*sv.mean())
    aligned = slope*st+intercept
    pq = pair_quantities(st, d["test"], d["test_pairs"], rho_hat, d["rho"])
    truth = d["test"]["f"]
    df_change = d["changed_f"]-truth
    delta_error = (schange-st)-df_change
    cross_i, cross_j = d["cross_i"], d["cross_j"]
    city_rows = []
    # Equal evaluation sample sizes make reshaping safe and city weights explicit.
    n_per_city = a.years*a.test_per_cell
    f_city, s_city = truth.reshape(a.cities, n_per_city), st.reshape(a.cities, n_per_city)
    ppc = a.years*a.eval_pairs_per_cell
    for c in range(a.cities):
        sl = slice(c*n_per_city, (c+1)*n_per_city)
        pl = slice(c*ppc, (c+1)*ppc)
        city_rows.append(dict(rep=rep, log_rho_sd=log_sd, method=method,
                              weighting=a.weighting, city=c, n_train=int(d["sizes"][c]),
                              rho_true=float(d["rho"][c]), rho_hat=float(rho_hat[c]),
                              f_rmse=float(np.sqrt(np.mean((s_city[c]-f_city[c])**2))),
                              f_r2=r2(f_city[c], s_city[c]),
                              f_bias=float((s_city[c]-f_city[c]).mean()),
                              aligned_f_rmse=float(np.sqrt(np.mean((aligned[sl]-truth[sl])**2))),
                              nll=float(pq["nll"][pl].mean()), kl=float(pq["kl"][pl].mean())))
    rmse_c = np.array([r["f_rmse"] for r in city_rows])
    metrics = dict(rep=rep, log_rho_sd=log_sd, method=method, weighting=a.weighting,
                   n_train=len(train["x"]), best_step=best_step,
                   fit_seconds=time.perf_counter()-start,
                   f_r2=r2(truth, st), f_rmse=float(np.sqrt(np.mean((st-truth)**2))),
                   aligned_f_r2=r2(truth, aligned),
                   f_spearman=float(spearmanr(truth, st).statistic),
                   city_weighted_rmse=float(rmse_c.mean()),
                   building_weighted_rmse=float(np.sqrt(np.average(rmse_c**2, weights=d["sizes"]))),
                   test_nll=float(pq["nll"].mean()), test_kl=float(pq["kl"].mean()),
                   probability_rmse=float(np.sqrt(pq["probability_sq_error"].mean())),
                   observed_pair_accuracy=float(pq["observed_accuracy"].mean()),
                   latent_pair_accuracy=float(pq["latent_accuracy"].mean()),
                   cross_city_latent_accuracy=float(np.mean(
                       (st[cross_j] > st[cross_i]) == (truth[cross_j] > truth[cross_i]))),
                   delta_f_rmse=float(np.sqrt(np.mean(delta_error**2))),
                   aligned_delta_f_rmse=float(np.sqrt(np.mean((slope*(schange-st)-df_change)**2))),
                   log_rho_rmse=float(np.sqrt(np.mean((np.log(rho_hat)-np.log(d["rho"]))**2))),
                   log_rho_correlation=correlation(np.log(rho_hat), np.log(d["rho"])),
                   rho_at_bounds_fraction=float(np.mean((rho_hat <= 0.1001) | (rho_hat >= 9.999))),
                   alignment_slope=slope, alignment_intercept=intercept)
    # Save inference-ready normalization constants: no reference sample needed to deploy.
    torch.save(dict(model_state={k: v.cpu() for k, v in model.state_dict().items()},
                    width=a.width, input_dimension=6, raw_score_mean=float(center.cpu()),
                    raw_score_sd=float(scale.cpu()), rho_hat=rho_hat.tolist(),
                    method=method, best_step=best_step,
                    inference="f_hat(x)=(ScoreNN(x)-raw_score_mean)/raw_score_sd"),
               folder/f"{method}_checkpoint.pt")
    np.savez_compressed(folder/f"{method}_predictions.npz", city=d["test"]["city"],
                        f_true=truth, f_hat=st, rho_true=d["rho"], rho_hat=rho_hat,
                        delta_f_true=df_change, delta_f_hat=schange-st)
    diagnostic_plot(folder/f"{method}_diagnostics.png", method, log_sd,
                    truth, st, d["rho"], rho_hat, d["sizes"], city_rows)
    del optimizer, model, cal, x, labels, reference
    return metrics, city_rows, histories


def diagnostic_plot(path, method, log_sd, truth, score, rho, rho_hat, sizes, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    ix = np.linspace(0, len(truth)-1, min(5000, len(truth)), dtype=int)
    axes[0].scatter(truth[ix], score[ix], s=3, alpha=0.2)
    lim = [min(truth[ix].min(), score[ix].min()), max(truth[ix].max(), score[ix].max())]
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set(xlabel="True common index f", ylabel="Estimated index", title="Index recovery")
    points = axes[1].scatter(rho, rho_hat, c=np.log10(sizes), cmap="viridis", s=22)
    axes[1].plot([0.1, 10], [0.1, 10], "k--", lw=1)
    axes[1].set(xscale="log", yscale="log", xlabel="True rho", ylabel="Estimated rho",
                title="Calibration recovery", xlim=(0.09, 11), ylim=(0.09, 11))
    fig.colorbar(points, ax=axes[1], label="log10 training city size")
    axes[2].scatter(sizes, [r["f_rmse"] for r in rows], alpha=0.6, s=18)
    axes[2].set(xscale="log", xlabel="Training observations per city", ylabel="Index RMSE",
                title="Small versus large cities")
    fig.suptitle(f"{method}: log-rho SD = {log_sd:g}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


SUMMARY_METRICS = ("f_r2", "aligned_f_r2", "f_rmse", "test_nll", "test_kl",
                   "probability_rmse", "cross_city_latent_accuracy", "delta_f_rmse",
                   "log_rho_rmse", "city_weighted_rmse", "building_weighted_rmse")


def summarize(a, metrics, city_rows, histories):
    summary, paired = [], []
    for sd in a.log_rho_sds:
        for method in METHODS:
            rows = [r for r in metrics if r["log_rho_sd"] == sd and r["method"] == method]
            if not rows:
                continue
            for metric in SUMMARY_METRICS:
                values = np.array([r[metric] for r in rows])
                summary.append(dict(log_rho_sd=sd, method=method, weighting=a.weighting,
                                    metric=metric, mean=float(values.mean()),
                                    mc_se=float(values.std(ddof=1)/np.sqrt(len(values)))
                                    if len(values) > 1 else float("nan"), reps=len(values)))
        for comparison in ("city", "oracle"):
            for metric in SUMMARY_METRICS:
                diffs = []
                for rep in range(a.reps):
                    matched = {r["method"]: r for r in metrics
                               if r["rep"] == rep and r["log_rho_sd"] == sd}
                    if comparison in matched and "shared" in matched:
                        diffs.append(matched[comparison][metric]-matched["shared"][metric])
                if diffs:
                    vals = np.array(diffs)
                    paired.append(dict(log_rho_sd=sd, comparison=f"{comparison}-shared",
                                       weighting=a.weighting, metric=metric,
                                       mean_difference=float(vals.mean()),
                                       paired_mc_se=float(vals.std(ddof=1)/np.sqrt(len(vals)))
                                       if len(vals) > 1 else float("nan"), reps=len(vals)))
    write_csv(a.out/"metrics.csv", metrics)
    write_csv(a.out/"city_metrics.csv", city_rows)
    write_csv(a.out/"histories.csv", histories)
    write_csv(a.out/"summary.csv", summary)
    write_csv(a.out/"paired_differences.csv", paired)


def summary_plot(a, metrics, city_rows, histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(a.log_rho_sds), 4,
                             figsize=(16, 4*len(a.log_rho_sds)), squeeze=False)
    colors = {"shared": "#64748b", "city": "#2563eb", "oracle": "#059669"}
    for row, sd in enumerate(a.log_rho_sds):
        for method in METHODS:
            ms = [r for r in metrics if r["log_rho_sd"] == sd and r["method"] == method]
            if not ms:
                continue
            for col, metric in enumerate(("f_r2", "test_kl")):
                vs = np.array([r[metric] for r in ms])
                pos = METHODS.index(method)
                axes[row, col].bar(pos, vs.mean(), color=colors[method], alpha=0.8)
                if len(vs) > 1:
                    axes[row, col].errorbar(pos, vs.mean(), yerr=vs.std(ddof=1)/np.sqrt(len(vs)),
                                           fmt="none", color="black", capsize=4)
            cs = [r for r in city_rows if r["log_rho_sd"] == sd and r["method"] == method]
            levels = sorted(set(r["n_train"] for r in cs))
            means = [np.mean([r["f_rmse"] for r in cs if r["n_train"] == n]) for n in levels]
            axes[row, 2].plot(levels, means, "o-", label=method, color=colors[method])
            hs = [r for r in histories if r["log_rho_sd"] == sd and r["method"] == method]
            steps = sorted(set(r["step"] for r in hs))
            vals = [np.mean([r["val_nll"] for r in hs if r["step"] == s]) for s in steps]
            axes[row, 3].plot(steps, vals, color=colors[method], label=method)
        for col, title in enumerate(("Latent index R2 (higher better)", "Probability KL (lower better)")):
            axes[row, col].set(xticks=range(3), xticklabels=METHODS,
                               title=f"SD(log rho)={sd:g}: {title}")
        axes[row, 2].set(xscale="log", xlabel="Training city size", ylabel="Mean city index RMSE")
        axes[row, 3].set(xlabel="Optimizer step", ylabel="Validation observed NLL")
        axes[row, 2].legend()
        axes[row, 3].legend()
    fig.suptitle(f"City calibration Monte Carlo | sampling: {a.weighting} | error bars: Monte Carlo SE")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(a.out/"experiment_summary.png", dpi=160)
    plt.close(fig)


def main():
    a = parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"Output folder {a.out} is not empty. Choose a new --out to preserve previous runs.")
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable in this PyTorch installation.")
    device = torch.device("cuda" if a.device == "cuda" or
                          (a.device == "auto" and torch.cuda.is_available()) else "cpu")
    # Float32 is sufficient and avoids half-precision problems with normalization.
    # Repeated fits share seeds, but bitwise reproducibility across hardware is not promised.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                  device=str(device), gpu=torch.cuda.get_device_name() if device.type == "cuda" else None)
    json_write(a.out/"config.json", config)
    (a.out/"README.txt").write_text(__doc__, encoding="utf-8")
    total_fits = a.reps*len(a.log_rho_sds)*len(METHODS)
    print(f"Device: {device}; {total_fits} fits x {a.steps} steps; output: {a.out.resolve()}", flush=True)
    if device.type == "cpu":
        print("CPU selected: consider --reps 1 --steps 600 for the first run.", flush=True)
    all_metrics, all_cities, all_histories = [], [], []
    total_start = time.perf_counter()
    for rep in range(a.reps):
        torch.manual_seed(a.seed + 10000*rep + 31)
        initial = copy.deepcopy(ScoreNN(a.width).state_dict())
        for log_sd in a.log_rho_sds:
            d = make_design(a, rep, log_sd)
            folder = a.out/f"rep_{rep:02d}_sd_{log_sd:g}"
            folder.mkdir()
            np.savez_compressed(folder/"dgp.npz", sizes=d["sizes"], rho=d["rho"],
                                centers=d["centers"], mu=d["mu"], sigma=d["sigma"],
                                reference=d["reference"], f_mean=d["f_mean"], f_sd=d["f_sd"])
            print(f"\nReplication {rep+1}/{a.reps}, log-rho SD={log_sd:g}, "
                  f"N_train={len(d['train']['x']):,}, rho range="
                  f"[{d['rho'].min():.2f}, {d['rho'].max():.2f}]", flush=True)
            for method in METHODS:
                m, cs, hs = fit_one(a, d, method, initial, rep, log_sd, device, folder)
                all_metrics.append(m)
                all_cities.extend(cs)
                all_histories.extend(hs)
                # Persist after every fit so completed work survives interruption.
                summarize(a, all_metrics, all_cities, all_histories)
                print(f"  BEST {method}: step={m['best_step']}, test f R2={m['f_r2']:.3f}, "
                      f"KL={m['test_kl']:.5f}, rho log-RMSE={m['log_rho_rmse']:.3f}", flush=True)
    summary_plot(a, all_metrics, all_cities, all_histories)
    print(f"\nFinished in {(time.perf_counter()-total_start)/60:.1f} minutes.")
    print("Read experiment_summary.png, summary.csv and paired_differences.csv.")
    print("Negative city-shared differences favor city calibration for errors/NLL; positive for R2/accuracy.")
    print("Probability improvements alone do not establish better recovery of the common index.")


if __name__ == "__main__":
    main()
