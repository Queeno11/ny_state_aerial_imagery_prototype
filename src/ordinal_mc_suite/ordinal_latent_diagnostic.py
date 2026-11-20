"""Fixed-budget, paired ordinal latent recovery diagnostics. See ORDINAL_LATENT_DIAGNOSTIC.md.

The original joint_two_step_pilot.py and its result directories are never written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import joint_two_step_pilot as pilot

ROOT = Path(__file__).resolve().parent
SCENARIOS = (("linear", "simple", 0.), ("quadratic", "simple", 0.),
             ("quadratic", "nonlinear", 0.), ("quadratic", "nonlinear", .9))
MAPS = ("M1", "M2", "M3")
METRICS = ("f_mse", "mu_mse", "delta_f_mse", "delta_mu_mse")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_bytes(path, content):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))


def atomic_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, output.getvalue().encode("utf-8"))


def panel_for_map(p, model, sigma):
    """Reuse exact H, X, eta, contexts, and tract labels across outcome maps."""
    result = dict(p)
    w = np.empty_like(p["f"])
    mu = np.empty_like(w)
    for t in (0, 1):
        a, b, k = pilot.map_parameters(p["c"], np.full(len(p["c"]), t), model, p["contexts"])
        w[:, t] = pilot.psi(p["r"][:, t], a, b, k)
        mu[:, t] = pilot.conditional_mean(p["f"][:, t], a, b, k, sigma)
    result.update(w=w, mu=mu)
    return result


def feature_stats(fit):
    x = fit["x"].reshape(-1, 7)
    return x.mean(0), np.maximum(x.std(0), 1e-6)


def input_array(p, method, shape, center, scale):
    if method == "basis":
        return pilot.basis(p["h"].reshape(-1, 3), shape).reshape(len(p["c"]), 2, -1)
    return (p["x"] - center) / scale


class Net(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, width[0]), nn.ReLU(),
                                 nn.Linear(width[0], width[1]), nn.ReLU(), nn.Linear(width[1], 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Basis(nn.Module):
    def __init__(self, dim, initial):
        super().__init__()
        self.beta = nn.Parameter(initial[:dim].clone())

    def forward(self, x):
        return x @ self.beta


def predict(model, x, device, chunk=8192):
    with torch.no_grad():
        flat = x.reshape(-1, x.shape[-1])
        pieces = [model(torch.as_tensor(a, dtype=torch.float64, device=device)).cpu().numpy()
                  for a in np.array_split(flat, max(1, math.ceil(len(flat) / chunk)))]
    return np.concatenate(pieces).reshape(-1, 2)


def stable_links(p, kind, shape, center, scale, seed, cross, limit, temporal_limit=None):
    """Independent links per split; records comprise paired endpoints, no outcome labels."""
    rng = np.random.default_rng(seed)
    stable = np.flatnonzero(p["category"] < 2)
    temporal = (stable if temporal_limit is None else
                rng.choice(stable, size=min(temporal_limit, len(stable)), replace=False))
    first = input_array(p, kind, shape, center, scale)
    a, b = first[temporal, 0], first[temporal, 1]
    if not cross:
        return (a, b), None
    cities = len(p["contexts"])
    # Reserve one edge per neighboring pair for connected city coverage.
    origin = np.arange(cities - 1)
    selected = np.array([rng.choice(np.flatnonzero(p["c"] == c)) for c in origin])
    destination = origin + 1
    extra_count = max(0, min(limit - len(selected), len(p["c"]) - len(selected)))
    if extra_count:
        remaining = np.setdiff1d(np.arange(len(p["c"])), selected)
        extra = rng.choice(remaining, size=extra_count, replace=False)
        selected = np.r_[selected, extra]
        destination = np.r_[destination, (p["c"][extra] + 1 +
                             rng.integers(0, cities - 1, len(extra))) % cities]
    if kind == "basis":
        other = pilot.basis(p["h"][selected, 0], shape)
    else:
        other = (pilot.features(p["h"][selected, 0], destination,
                                np.zeros(len(selected)), p["contexts"], rng) - center) / scale
    return (a, b), (first[selected, 0], other)


def tensors(links, device):
    return tuple(None if pair is None else tuple(torch.as_tensor(a, dtype=torch.float64, device=device)
                                                     for a in pair) for pair in links)


def pair_objective(s, w, cities, rank, tau):
    # Per-cell comparisons: O(cities * batch_tracts^2), not a global NxN matrix.
    pieces, variances = [], []
    for city in torch.unique(cities):
        ii = torch.nonzero(cities == city).flatten()
        for t in (0, 1):
            z = s[ii, t]
            obs = w[ii, t]
            gap = z[:, None] - z[None, :]
            signs = torch.sign(obs[:, None] - obs[None, :])
            mask = ~torch.eye(len(ii), dtype=torch.bool, device=s.device)
            val = F.softplus(-signs * gap) if rank == "ranknet" else torch.sigmoid(-signs * gap / tau)
            pieces.append(val[mask].mean())
            variances.append(gap.square()[mask].mean() / 2)
    return torch.stack(pieces).mean(), torch.stack(variances).mean()


def objective(model, x, p, indices, links, rank, args, method, device):
    xx = torch.as_tensor(x[indices], dtype=torch.float64, device=device)
    s = model(xx.reshape(-1, xx.shape[-1])).reshape(-1, 2)
    if method == "supervised":
        f = torch.as_tensor(p["f"][indices], dtype=torch.float64, device=device)
        loss = (s - f).square().mean()
        zeros = dict(pair_loss=float("nan"), dispersion=float("nan"),
                     temporal_numerator=float("nan"), cross_numerator=float("nan"),
                     weighted_temporal=float("nan"), weighted_cross=float("nan"))
        return loss, {**zeros, "objective": float(loss.detach())}
    w = torch.as_tensor(p["w"][indices], dtype=torch.float64, device=device)
    cities = torch.as_tensor(p["c"][indices], dtype=torch.long, device=device)
    pair, variance = pair_objective(s, w, cities, rank, args.tau)
    if bool(variance.detach() <= 1e-12):
        raise FloatingPointError("Near-zero within-cell score dispersion")
    temporal = (model(links[0][0]) - model(links[0][1])).square().mean()
    cross = ((model(links[1][0]) - model(links[1][1])).square().mean()
             if links[1] is not None else torch.zeros_like(temporal))
    wt, wc = args.lambda_t * temporal / variance, args.lambda_c * cross / variance
    total = pair + wt + wc
    return total, dict(pair_loss=float(pair.detach()), dispersion=float(variance.detach()),
                       temporal_numerator=float(temporal.detach()), cross_numerator=float(cross.detach()),
                       weighted_temporal=float(wt.detach()), weighted_cross=float(wc.detach()),
                       objective=float(total.detach()))


def monitored_objective(model, x, p, links, rank, args, method, device):
    # Fixed deterministic monitoring sample per split. No gradient or selection.
    rng = np.random.default_rng(pilot.seed(args.seed, "objective_eval", len(p["c"])))
    indices = np.concatenate([rng.choice(np.flatnonzero(p["c"] == c),
                                         min(args.batch_tracts, sum(p["c"] == c)), replace=False)
                              for c in np.unique(p["c"])])
    with torch.no_grad():
        return dict(objective(model, x, p, indices, links, rank, args, method, device)[1],
                    monitored_tract_records=len(indices), panel_tract_records=len(p["c"]),
                    monitoring_design="fixed sampled tracts per city; all selected links")


def oracle_mu(p, latent, model, sigma):
    result = np.empty_like(latent)
    for t in (0, 1):
        a, b, k = pilot.map_parameters(p["c"], np.full(len(p["c"]), t), model, p["contexts"])
        result[:, t] = pilot.conditional_mean(latent[:, t], a, b, k, sigma)
    return result


def metric_rows(p, fhat, muhat, cutoff, metadata):
    rows = []
    for name, mask in (("all", np.ones(len(p["c"]), bool)),
                       ("stable", p["category"] < 2), ("small", p["category"] == 2),
                       ("large", p["category"] == 3)):
        df, dfhat = np.diff(p["f"][mask], axis=1).ravel(), np.diff(fhat[mask], axis=1).ravel()
        dmu, dmuhat = np.diff(p["mu"][mask], axis=1).ravel(), np.diff(muhat[mask], axis=1).ravel()
        row = dict(**metadata, stratum=name, n_tracts=int(mask.sum()),
                   f_mse=float(np.mean((fhat[mask] - p["f"][mask]) ** 2)),
                   mu_mse=float(np.mean((muhat[mask] - p["mu"][mask]) ** 2)),
                   delta_f_mse=float(np.mean((dfhat - df) ** 2)),
                   delta_mu_mse=float(np.mean((dmuhat - dmu) ** 2)),
                   delta_f_bias=float(np.mean(dfhat - df)),
                   delta_mu_bias=float(np.mean(dmuhat - dmu)),
                   delta_f_slope=float(np.cov(df, dfhat, bias=True)[0, 1] / np.var(df))
                   if np.var(df) > 1e-14 else float("nan"),
                   delta_mu_slope=float(np.cov(dmu, dmuhat, bias=True)[0, 1] / np.var(dmu))
                   if np.var(dmu) > 1e-14 else float("nan"),
                   stable_movement_rms=float(np.sqrt(np.mean(dfhat ** 2))) if name == "stable" else float("nan"),
                   detection_rate=float(np.mean(np.abs(dfhat) > cutoff)),
                   cutoff=float(cutoff))
        rows.append(row)
    return rows


def evaluate(model, fit, panels, arrays, shape, maps, method, args, metadata, device):
    raw_fit = predict(model, arrays["fit"], device)
    if method == "supervised":
        offset, slope, alignment = 0., 1., "raw supervised output; no affine alignment"
    else:
        offset, slope, alignment = pilot.fit_alignment(raw_fit, fit)
    rows = []
    for name in ("source_test", "target_test"):
        p = panels[name]
        stable = panels["source_threshold" if name.startswith("source") else "target_threshold"]
        fhat = offset + slope * predict(model, arrays[name], device)
        stable_score = offset + slope * predict(model, arrays["source_threshold" if name.startswith("source")
                                                                 else "target_threshold"], device)
        cutoff = float(np.quantile(np.abs(np.diff(stable_score[stable["category"] < 2], axis=1)),
                                   .95, method="higher"))
        for mapping in maps:
            mapped = panel_for_map(p, mapping, args.sigma)
            muhat = oracle_mu(mapped, fhat, mapping, args.sigma)
            rows.extend(metric_rows(mapped, fhat, muhat, cutoff,
                                    dict(**metadata, panel=name, map=mapping, alignment_offset=offset,
                                         alignment_slope=slope, alignment_status=alignment,
                                         map_info="privileged true cell map and Gaussian marginal; no fitted level head")))
    return rows


def batch_indices(p, args, rng):
    city_count = len(p["contexts"])
    chosen = rng.choice(city_count, size=min(args.batch_cities, city_count), replace=False)
    n = len(p["c"]) // city_count
    return np.concatenate([city * n + rng.choice(n, args.batch_tracts, replace=False)
                           for city in chosen])


def make_panels(args, shape, mechanism, rho, seed):
    class PanelArgs:
        n = args.n
        eval_n = args.eval_n
        sigma = args.sigma
    result = {}
    # City count does not enter random seed: fixed target panels across scaling arms.
    for part, cities in (("fit", args.cities), ("validation", args.validation_cities),
                         ("source_test", args.source_test_cities),
                         ("source_threshold", args.source_test_cities),
                         ("target_test", args.target_cities),
                         ("target_threshold", args.target_cities)):
        result[part] = pilot.panel(PanelArgs, shape, "M1", mechanism, rho,
                                   part, cities, seed)
    # All source splits use actual fitting-city contexts, including the first
    # evaluation cities in the 64-city and 256-city scaling arms. Change only
    # the context columns of the unmixed features, retaining H and nuisances.
    common = np.random.default_rng(pilot.seed(seed, "shared_source_contexts")).normal(
        size=(args.cities, 2))
    for part in ("fit", "validation", "source_test", "source_threshold"):
        p = result[part]
        p["contexts"] = common[:len(p["contexts"])].copy()
        original = p["x"] @ pilot.MIX
        original[:, :, 5:7] = common[p["c"]][:, None, :]
        p["x"] = original @ pilot.MIX.T
        result[part] = panel_for_map(p, "M1", args.sigma)
    return result


def checkpoint(path, step, model, optimizer, rng):
    payload = dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                   numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_bytes(path, buffer.getvalue())


def trajectory(args, out, tag, setting, panels, center, scale, initial_nn, initial_basis, device):
    shape, mechanism, rho = setting
    method, rank, cross = tag
    maps = MAPS
    # Strictly monotone cell maps preserve every within-cell comparison;
    # all ordinal fits use M1 W ranks and are reused for M1/M2/M3 evaluation.
    training_maps = ("M1",) if method != "supervised" else ("truth",)
    for training_map in training_maps:
        stem = f"{args.stage}_{shape}_{mechanism}_rho{rho:g}_{method}_{rank}_cross{int(cross)}_train{training_map}_seed{args.rep}"
        path = out / stem
        path.mkdir(exist_ok=True)
        fit = panel_for_map(panels["fit"], training_map if training_map != "truth" else "M1", args.sigma)
        validation = panel_for_map(panels["validation"], training_map if training_map != "truth" else "M1", args.sigma)
        arrays = {k: input_array(v, method, shape, center, scale) for k, v in panels.items()}
        model = (Basis(arrays["fit"].shape[-1], initial_basis) if method == "basis"
                 else Net(tuple(args.width))).double().to(device)
        if method != "basis":
            model.load_state_dict(initial_nn)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        fit_limit = (args.link_limit if args.link_limit is not None else
                     max(args.cities - 1, math.ceil(args.link_fraction * len(fit["c"]))))
        val_limit = (args.link_limit if args.link_limit is not None else
                     max(args.validation_cities - 1,
                         math.ceil(args.link_fraction * len(validation["c"]))))
        fit_links = tensors(stable_links(fit, method, shape, center, scale,
                                        pilot.seed(args.seed, args.rep, shape, mechanism, "fit_links"),
                                        cross, fit_limit, args.link_limit), device)
        val_links = tensors(stable_links(validation, method, shape, center, scale,
                                        pilot.seed(args.seed, args.rep, shape, mechanism, "val_links"),
                                        cross, val_limit, args.link_limit), device)
        link_info = dict(temporal_count=len(fit_links[0][0]),
                         temporal_coverage=len(fit_links[0][0]) / len(fit["c"]),
                         cross_count=len(fit_links[1][0]) if fit_links[1] is not None else 0,
                         cross_coverage=(len(fit_links[1][0]) / len(fit["c"]))
                         if fit_links[1] is not None else 0.,
                         validation_temporal_count=len(val_links[0][0]),
                         validation_cross_count=len(val_links[1][0]) if val_links[1] is not None else 0)
        rng = np.random.default_rng(pilot.seed(args.seed, args.rep, shape, mechanism, "batches"))
        snapshots = sorted(path.glob("state_*.pt"))
        step = 0
        if snapshots:
            state = torch.load(snapshots[-1], map_location=device, weights_only=False)
            step = int(state["step"])
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            rng.bit_generator.state = state["numpy_rng"]
            torch.set_rng_state(state["torch_rng"].cpu())
            if state["cuda_rng"] is not None and device == "cuda":
                torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        checkpoints = args.checkpoints
        if step and step not in checkpoints:
            raise ValueError(f"Unexpected checkpoint in {path}: {step}")
        # Recreate diagnostics for a saved state when interruption occurred
        # between atomic state and metric writes.
        for update in range(step if step else 1, checkpoints[-1] + 1):
            if update > step:
                indices = batch_indices(fit, args, rng)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = objective(model, arrays["fit"], fit, indices, fit_links,
                                    rank, args, method, device)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite loss at {stem} update {update}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 20.)
                optimizer.step()
            if update not in checkpoints:
                continue
            state_file = path / f"state_{update:08d}.pt"
            objective_file = path / f"objective_{update:08d}.json"
            metric_file = path / f"metrics_{update:08d}.csv"
            if update == step and objective_file.exists() and metric_file.exists():
                continue
            # Save state first; partial result files never mark a job finished.
            if update > step:
                checkpoint(state_file, update, model, optimizer, rng)
            exposure = update * min(args.batch_cities, args.cities) * args.batch_tracts
            meta = dict(stage=args.stage, seed=args.rep, scenario=f"{shape}/{mechanism}/rho{rho:g}",
                        initialization_seed=args.initialization_seed,
                        method=method, rank=rank, cross=int(cross), trained_map=training_map,
                        updates=update, selected_tract_records=exposure,
                        effective_tract_passes=exposure / (args.cities * args.n),
                        source_fit_cities=args.cities, source_fit_n=args.n, **link_info)
            train_diag = monitored_objective(model, arrays["fit"], fit, fit_links, rank, args, method, device)
            val_diag = monitored_objective(model, arrays["validation"], validation, val_links,
                                            rank, args, method, device)
            atomic_json(objective_file, dict(train=train_diag, validation=val_diag, exposure=meta))
            rows = evaluate(model, fit, panels, arrays, shape, maps, method, args, meta, device)
            atomic_csv(metric_file, rows)
            print(f"checkpoint {stem} update {update}/{checkpoints[-1]}", flush=True)
        if not all((path / f"state_{c:08d}.pt").exists() and
                   (path / f"objective_{c:08d}.json").exists() and
                   (path / f"metrics_{c:08d}.csv").exists() for c in checkpoints):
            raise RuntimeError(f"Incomplete checkpoint files in {path}; inspect files before resuming")
        atomic_json(path / "complete.json", dict(last_checkpoint=checkpoints[-1],
                                                   metric_files=[f"metrics_{c:08d}.csv" for c in checkpoints],
                                                   status="complete"))


def plots(rows, objectives, out, stage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    test = [r for r in rows if r["panel"] == "target_test"]
    for scenario in sorted({r["scenario"] for r in test}):
        slug = scenario.replace("/", "_")
        for mapping in MAPS:
            data = [r for r in test if r["scenario"] == scenario and r["map"] == mapping]
            all_rows = [r for r in data if r["stratum"] == "all"]
            keys = sorted({(r["method"], r["rank"], r["cross"]) for r in all_rows})
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for ax, metric in zip(axes.flat, METRICS):
                for key in keys:
                    points = [r for r in all_rows if (r["method"], r["rank"], r["cross"]) == key]
                    updates = sorted({int(r["updates"]) for r in points})
                    values = [np.mean([float(r[metric]) for r in points if int(r["updates"]) == u])
                              for u in updates]
                    ax.plot(updates, values, label="/".join(map(str, key)), marker=".")
                ax.set(xlabel="optimizer updates", ylabel=metric)
                ax.grid(alpha=.2)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", fontsize=8, ncol=3)
            fig.suptitle(f"{scenario} | {mapping} | target test, true-map oracle")
            fig.tight_layout(rect=(0, .12, 1, .95))
            fig.savefig(out / f"{stage}_{slug}_{mapping}_trajectories.png", dpi=140)
            plt.close(fig)
            # Show the stable/small/large tradeoff at the fixed final checkpoint.
            last = max(int(r["updates"]) for r in data)
            fig, ax = plt.subplots(figsize=(11, 5))
            strata = ("stable", "small", "large")
            positions = np.arange(len(strata))
            for i, key in enumerate(keys):
                points = [r for r in data if int(r["updates"]) == last and
                          (r["method"], r["rank"], r["cross"]) == key]
                y = [np.mean([float(r["delta_f_mse"]) for r in points if r["stratum"] == s])
                     for s in strata]
                ax.plot(positions, y, marker="o", label="/".join(map(str, key)))
            ax.set(xticks=positions, xticklabels=strata, ylabel="delta_f_mse",
                   title=f"{scenario} | {mapping} | update {last} | target change strata")
            ax.legend(fontsize=8, ncol=3)
            ax.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(out / f"{stage}_{slug}_{mapping}_strata.png", dpi=140)
            plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, split in zip(axes, ("train", "validation")):
            data = [r for r in objectives if r["scenario"] == scenario and r["split"] == split]
            keys = sorted({(r["method"], r["rank"], r["cross"]) for r in data})
            for key in keys:
                points = [r for r in data if (r["method"], r["rank"], r["cross"]) == key]
                updates = sorted({int(r["updates"]) for r in points})
                values = [np.mean([float(r["objective"]) for r in points if int(r["updates"]) == u])
                          for u in updates]
                ax.plot(updates, values, marker=".", label="/".join(map(str, key)))
            ax.set(xlabel="optimizer updates", ylabel="fixed sampled objective", title=split)
            ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", fontsize=8, ncol=3)
        fig.suptitle(f"{scenario} | monitored train/validation; objective scales differ by method")
        fig.tight_layout(rect=(0, .15, 1, .93))
        fig.savefig(out / f"{stage}_{slug}_objectives.png", dpi=140)
        plt.close(fig)


def collect(out, stage):
    rows = []
    objectives = []
    for folder in sorted(out.glob(stage + "_*")):
        if not folder.is_dir() or not (folder / "complete.json").exists():
            continue
        for file in sorted(folder.glob("metrics_*.csv")):
            with file.open(newline="", encoding="utf-8") as stream:
                rows.extend(csv.DictReader(stream))
        for file in sorted(folder.glob("objective_*.json")):
            data = json.loads(file.read_text(encoding="utf-8"))
            for split in ("train", "validation"):
                objectives.append(dict(**data["exposure"], split=split, **data[split]))
    atomic_csv(out / f"{stage}_metrics.csv", rows)
    atomic_csv(out / f"{stage}_objectives.csv", objectives)
    if rows:
        plots(rows, objectives, out, stage)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    lines = [f"# {stage.title()} ordinal latent diagnostic", "",
             "Implementation run summary; fixed checkpoints and no outcome-based selection.", "",
             f"Completed trajectories: {sum(1 for folder in out.glob(stage + '_*') if folder.is_dir() and (folder / 'complete.json').exists())}.",
             f"Metric rows: {len(rows)}. Objective rows: {len(objectives)}.",
             f"Seed count: {manifest['config']['seeds']}; fitting cities: {manifest['config']['cities']}; "
             f"paired tracts/city: {manifest['config']['n']}.", "",
             "The basis arm observes true H; supervised NN observes fitting f; true maps and Gaussian noise "
             "marginalization are oracle evaluation only.",
             "Objective rows use a deterministic sampled monitoring batch per city and selected links. "
             "They are descriptive and do not select checkpoints.",
             "Static plots are descriptive means across seeds. No local-optimum or likelihood-gap claim follows.",
             "See ORDINAL_LATENT_DIAGNOSTIC.md for methods and limits, and the CSV for paired seed-level results.", ""]
    atomic_bytes(out / "REPORT.md", "\n".join(lines).encode("utf-8"))


def self_check():
    class Args:
        n = 16
        eval_n = 16
        sigma = .8
        cities = 4
        validation_cities = 2
        source_test_cities = 2
        target_cities = 2
    p = pilot.panel(Args, "quadratic", "M1", "simple", .9, "fit", 3, 412)
    m1, m3 = panel_for_map(p, "M1", .8), panel_for_map(p, "M3", .8)
    np.testing.assert_array_equal(m1["x"], m3["x"])
    np.testing.assert_array_equal(m1["r"], m3["r"])
    np.testing.assert_allclose(oracle_mu(m3, p["f"], "M3", .8), m3["mu"], rtol=0, atol=1e-12)
    oracle = metric_rows(m3, p["f"], oracle_mu(m3, p["f"], "M3", .8), 0.,
                         dict(check="oracle"))[0]
    assert all(oracle[key] == 0. for key in METRICS)
    for mapped in (panel_for_map(p, "M2", .8), m3):
        for city in range(3):
            for t in (0, 1):
                baseline = m1["w"][p["c"] == city, t]
                current = mapped["w"][p["c"] == city, t]
                np.testing.assert_array_equal(np.sign(baseline[:, None] - baseline[None, :]),
                                              np.sign(current[:, None] - current[None, :]))
    rng = np.random.default_rng(3)
    simulated = np.mean(np.tanh(p["f"][0, 0] + .8 * rng.normal(size=400000)))
    quadrature = np.tanh(p["f"][0, 0] + .8 * pilot.QX) @ pilot.QW
    assert abs(simulated - quadrature) < .005
    x = pilot.basis(p["h"].reshape(-1, 3), "quadratic").reshape(-1, 2, 9)
    np.testing.assert_allclose(x[p["category"] < 2, 0], x[p["category"] < 2, 1])
    assert not np.array_equal(pilot.panel(Args, "quadratic", "M1", "simple", .9,
                                          "validation", 3, 412)["h"], p["h"])
    panels4 = make_panels(Args, "quadratic", "simple", .9, 412)
    class Larger(Args):
        cities = 8
    panels8 = make_panels(Larger, "quadratic", "simple", .9, 412)
    np.testing.assert_array_equal(panels4["fit"]["contexts"], panels8["fit"]["contexts"][:4])
    for part in ("validation", "source_test", "source_threshold"):
        np.testing.assert_array_equal(panels4[part]["contexts"], panels4["fit"]["contexts"][:2])
        np.testing.assert_array_equal(panels4[part]["h"], panels8[part]["h"])
        np.testing.assert_array_equal(panels4[part]["x"], panels8[part]["x"])
        assert not np.array_equal(panels4[part]["h"], panels4["fit"]["h"][:len(panels4[part]["h"])])
    for part in ("target_test", "target_threshold"):
        np.testing.assert_array_equal(panels4[part]["x"], panels8[part]["x"])
    assert 4800 * 16 * 12 / (64 * 250) == 57.6
    assert 19200 * 16 * 12 / (256 * 250) == 57.6
    print("self-check passed: equal map ranks, oracle four errors, M3 quadrature, "
          "basis invariance, source contexts, independent tracts, scaling panels, exposure")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage", choices=("core", "scaling"), default="core")
    parser.add_argument("--scaling-arm", choices=("wide", "within_city", "city_coverage"), default="wide")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--initialization-seed", type=int, default=None,
                        help="network/basis initialization seed; defaults to --seed; data and batches unaffected")
    parser.add_argument("--cities", type=int, default=64)
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--validation-cities", type=int, default=8)
    parser.add_argument("--source-test-cities", type=int, default=8)
    parser.add_argument("--target-cities", type=int, default=8)
    parser.add_argument("--eval-n", type=int, default=64)
    parser.add_argument("--batch-cities", type=int, default=16)
    parser.add_argument("--batch-tracts", type=int, default=12)
    parser.add_argument("--link-fraction", type=float, default=.2,
                        help="fraction of panel tracts used for cross links; temporal links use all stable tracts")
    parser.add_argument("--link-limit", type=int, default=None,
                        help="optional absolute link cap; overrides fractional coverage")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=(400, 600, 1200, 2400, 4800))
    parser.add_argument("--width", type=int, nargs=2, default=(32, 16))
    parser.add_argument("--rank", choices=("ranknet", "sigmoid", "both"), default="ranknet")
    parser.add_argument("--links", choices=("off", "on", "both"), default="both")
    parser.add_argument("--methods", nargs="+", choices=("nn", "basis", "supervised"),
                        default=None)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--lambda-t", type=float, default=.3)
    parser.add_argument("--lambda-c", type=float, default=.3)
    parser.add_argument("--tau", type=float, default=.3)
    parser.add_argument("--sigma", type=float, default=.8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "ordinal_latent_diagnostic_results")
    args = parser.parse_args()
    if args.initialization_seed is None:
        args.initialization_seed = args.seed
    if args.methods is None:
        args.methods = ("nn",) if args.stage == "scaling" else ("nn", "basis", "supervised")
    if args.stage == "scaling":
        if args.scaling_arm == "wide":
            args.width = (128, 64)
        elif args.scaling_arm == "within_city":
            args.n = 1000
        else:
            args.cities = 256
        # For larger fitting sets, keep selected-tract passes comparable to
        # the 64x250 baseline at default settings. User-supplied budgets win.
        if args.scaling_arm != "wide" and args.checkpoints == (400, 600, 1200, 2400, 4800):
            args.checkpoints = tuple(4 * c for c in args.checkpoints)
    if args.smoke:
        args.cities, args.n, args.validation_cities, args.source_test_cities = 4, 16, 2, 2
        args.target_cities, args.eval_n = 2, 16
        args.batch_cities, args.batch_tracts, args.link_limit = 2, 4, 8
        args.checkpoints = (1, 2)
    args.checkpoints = tuple(sorted(set(args.checkpoints)))
    if (args.n < 4 or args.eval_n < 4 or args.eval_n % 4 or
            min(args.cities, args.validation_cities,
            args.source_test_cities, args.target_cities) < 2 or args.batch_tracts < 2 or
            args.batch_tracts > args.n or args.batch_cities < 1 or args.seeds < 1 or
            args.validation_cities > args.cities or args.source_test_cities > args.cities or
            min(args.width) < 1 or args.checkpoints[0] < 1 or
            args.lr <= 0 or args.sigma <= 0 or args.tau <= 0 or args.link_fraction <= 0 or
            (args.link_limit is not None and args.link_limit < max(args.cities - 1,
                                                                 args.validation_cities - 1)) or
            args.lambda_t < 0 or args.lambda_c < 0):
        parser.error("Invalid sizes, checkpoints, or objective coefficients")
    ranks = ("ranknet", "sigmoid") if args.rank == "both" else (args.rank,)
    links = (False, True) if args.links == "both" else (args.links == "on",)
    tags = [(method, rank, link) for method in args.methods
            for rank in (ranks if method != "supervised" else ("none",))
            for link in (links if method != "supervised" else (False,))]
    count = args.seeds * len(SCENARIOS) * len(tags)
    exposure = args.checkpoints[-1] * min(args.batch_cities, args.cities) * args.batch_tracts
    print(f"PLAN stage={args.stage}, smoke={args.smoke}: {count} optimizer trajectories; "
          f"{count * len(args.checkpoints)} checkpoints; "
          "each ordinal trajectory evaluates M1/M2/M3 on both held-out panels and four strata; "
          f"{exposure} selected tract records / trajectory, "
          f"{exposure / (args.cities * args.n):.3f} effective fitting-tract passes at final checkpoint. "
          "Each selected tract has two endpoint scores. No oracle validation selection.", flush=True)
    if args.plan:
        return
    if args.self_check:
        self_check()
        return
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    # Directory manifest is strict across source and settings; never reuse old pilot outputs.
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                            if k not in ("plan", "self_check")}, device=device,
                    source_sha256=digest(Path(__file__)), pilot_sha256=digest(Path(pilot.__file__)))
    manifest = json.loads(json.dumps(manifest))
    path = args.output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            parser.error("Output manifest/code fingerprint mismatch; choose a new output directory")
    elif any(args.output.iterdir()):
        parser.error("Nonempty output without matching manifest; choose a new output directory")
    else:
        atomic_json(path, manifest)
    for rep in range(1, args.seeds + 1):
        args.rep = rep
        data_seed = pilot.seed(args.seed, rep, "data")
        for shape, mechanism, rho in SCENARIOS:
            panels = make_panels(args, shape, mechanism, rho, data_seed)
            center, scale = feature_stats(panels["fit"])
            torch.manual_seed(pilot.seed(args.initialization_seed, rep, shape, mechanism, "initialization"))
            initial_nn = Net(tuple(args.width)).double().state_dict()
            basis_rng = np.random.default_rng(pilot.seed(args.initialization_seed, rep, shape,
                                                       mechanism, "basis_initialization"))
            initial_basis = torch.as_tensor(basis_rng.normal(0, .15, 9), dtype=torch.float64)
            for tag in tags:
                trajectory(args, args.output, tag, (shape, mechanism, rho), panels,
                           center, scale, initial_nn, initial_basis, device)
    collect(args.output, args.stage)
    print("Complete. No checkpoint selected by held-out truth.", flush=True)


if __name__ == "__main__":
    main()
