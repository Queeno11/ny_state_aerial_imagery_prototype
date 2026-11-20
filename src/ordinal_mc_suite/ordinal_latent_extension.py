"""Continue frozen ordinal trajectories and fit same-panel parametric likelihood.

Run --plan and --self-check before a real run; see ORDINAL_LATENT_EXTENSION.md.
Only load checkpoints from a trusted local baseline directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import ordinal_latent_diagnostic as d
import joint_two_step_pilot as p


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def stem(shape, mechanism, rho, method, cross, rep):
    rank, training = ("none", "truth") if method == "supervised" else ("ranknet", "M1")
    return f"core_{shape}_{mechanism}_rho{rho:g}_{method}_{rank}_cross{int(cross)}_train{training}_seed{rep}"


def canonical(coeff, shape, bb):
    mean, cov = p.REF[shape]
    return ((bb - torch.as_tensor(mean, dtype=torch.float64, device=bb.device)) @ coeff /
            torch.sqrt(coeff @ torch.as_tensor(cov, dtype=torch.float64, device=bb.device) @ coeff))


def paired_nll(w, f, a, b, k, rho, sigma, family):
    """Full paired transformed-Gaussian negative log density, fixed normalizer omitted."""
    y = (w-a)/b
    r = p.inverse_torch(y, k) if family == "M3" else y
    jac = b*(1+k*(1-torch.tanh(r).square()))
    e = r-f
    quad = (e.square().sum(1)-2*rho*e[:, 0]*e[:, 1])/(sigma**2*(1-rho**2))
    return (.5*quad+torch.log(jac).sum(1)).mean()


def ml_fit(fit, shape, family, rho, sigma, budget, device, starts):
    """Full fitting-panel paired likelihood; no latent truth enters fitting."""
    bb = torch.as_tensor(p.basis(fit["h"].reshape(-1, 3), shape).reshape(len(fit["c"]), 2, -1),
                         dtype=torch.float64, device=device)
    w = torch.as_tensor(fit["w"], dtype=torch.float64, device=device)
    cell = torch.as_tensor(2 * fit["c"][:, None] + np.arange(2), dtype=torch.long, device=device)
    ncell = 2 * len(fit["contexts"])
    trials = []
    for start in range(starts):
        try:
            # Ratios to the positive first basis coefficient remove radial redundancy.
            theta = nn.Parameter(torch.full((bb.shape[-1]-1,), 0. if start == 0 else .12,
                                             dtype=torch.float64, device=device))
            intercept = nn.Parameter(torch.zeros(ncell, dtype=torch.float64, device=device))
            raw_b = nn.Parameter(torch.ones(ncell, dtype=torch.float64, device=device)) if family != "M1" else None
            raw_k = nn.Parameter(torch.full((ncell,), -.7, dtype=torch.float64, device=device)) if family == "M3" else None
            variables = [x for x in (theta, intercept, raw_b, raw_k) if x is not None]

            def params():
                coeff = torch.cat((torch.ones(1, dtype=torch.float64, device=device), theta))
                a = intercept[cell]
                b = torch.ones_like(a) if raw_b is None else F.softplus(raw_b[cell]) + 1e-4
                k = torch.zeros_like(a) if raw_k is None else 2 * torch.sigmoid(raw_k[cell])
                return coeff, a, b, k

            def objective():
                coeff, a, b, k = params()
                return paired_nll(w, canonical(coeff, shape, bb), a, b, k, rho, sigma, family)

            first = float(objective().detach())
            optimizer = torch.optim.LBFGS(variables, lr=.8, max_iter=budget,
                                          line_search_fn="strong_wolfe", tolerance_grad=1e-8,
                                          tolerance_change=1e-11, history_size=40)
            calls = 0

            def closure():
                nonlocal calls
                calls += 1
                optimizer.zero_grad(set_to_none=True)
                value = objective()
                if not bool(torch.isfinite(value)):
                    raise FloatingPointError("Nonfinite likelihood")
                value.backward()
                return value

            optimizer.step(closure)
            iterations = int(optimizer.state[variables[0]].get("n_iter", 0))
            optimizer.zero_grad(set_to_none=True)
            final_tensor = objective()
            final_tensor.backward()
            final = float(final_tensor.detach())
            grad = math.sqrt(sum(float(v.grad.square().sum()) for v in variables))
            with torch.no_grad():
                coeff = params()[0].detach().cpu().numpy().tolist()
                pars = dict(coeff=coeff, a=intercept.detach().cpu().numpy().tolist(),
                            b=(np.ones(ncell) if raw_b is None else (F.softplus(raw_b)+1e-4).detach().cpu().numpy()).tolist(),
                            k=(np.zeros(ncell) if raw_k is None else (2*torch.sigmoid(raw_k)).detach().cpu().numpy()).tolist())
            if not all(np.isfinite(np.asarray(v)).all() for v in pars.values()) or not np.isfinite(grad+final):
                raise FloatingPointError("Nonfinite fitted parameters or gradient")
            trials.append(dict(start=start, parameters=pars, initial_nll=first,
                               final_nll=final, objective_improvement=first-final,
                               gradient_norm=grad, iterations=iterations, closure_calls=calls,
                               max_iterations=budget, iteration_cap_reached=iterations >= budget,
                               gradient_check_tolerance=1e-4,
                               convergence="gradient tolerance met" if grad <= 1e-4 else "not certified"))
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            trials.append(dict(start=start, convergence="failed", error=str(exc), max_iterations=budget))
    viable = [trial for trial in trials if trial["convergence"] != "failed"]
    if not viable:
        raise RuntimeError(f"All likelihood starts failed: {trials}")
    chosen = min(viable, key=lambda trial: trial["final_nll"])
    info = {key: value for key, value in chosen.items() if key != "parameters"}
    info.update(starts=trials, known_sigma=sigma, known_rho=rho, device=device,
                first_basis_coefficient_fixed_to_one=True, slope_floor=1e-4,
                tanh_shape_open_bounds=[0., 2.], selection="lowest finite source fitting NLL")
    return chosen["parameters"], info


def ml_predict(panel, shape, coeff):
    return p.latent(panel["h"].reshape(-1, 3), shape, coeff).reshape(-1, 2)


def panels_for(config, setting, rep):
    args = argparse.Namespace(**config)
    args.rep = rep
    return d.make_panels(args, *setting, p.seed(args.seed, rep, "data"))


def fingerprint(panel):
    return {key: hashlib.sha256(np.ascontiguousarray(panel[key]).tobytes()).hexdigest()
            for key in ("h", "x", "c", "category", "contexts", "f", "r", "w")}


def validate_baseline(args):
    manifest_path = args.baseline / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing baseline manifest: {manifest_path}")
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (original["source_sha256"] != sha(Path(d.__file__)) or
            original["pilot_sha256"] != sha(Path(p.__file__))):
        raise ValueError("Frozen diagnostic/pilot hash differs from baseline")
    c = original["config"]
    if (c["stage"] != "core" or c["rank"] != "ranknet" or
            c["checkpoints"][-1] != (2 if args.smoke else 4800)
            or c["sigma"] <= 0 or c["seeds"] < 1):
        raise ValueError("Expected core RankNet baseline with endpoint 4800 (or smoke endpoint 2)")
    if args.smoke != bool(c["smoke"]):
        raise ValueError("--smoke must match the explicitly selected smoke baseline")
    for method in args.methods:
        if method not in c["methods"]:
            raise ValueError(f"Missing baseline method: {method}")
    for method in args.methods:
        for cross in args.cross:
            if method != "supervised" and c["links"] not in ("both", "on" if cross else "off"):
                raise ValueError(f"Baseline lacks cross={cross}")
    endpoint = c["checkpoints"][-1]
    selected = [(setting, rep, method, cross) for rep in range(1, c["seeds"] + 1)
                for setting in d.SCENARIOS for method in args.methods
                for cross in (args.cross if method != "supervised" else (False,))]
    for (shape, mechanism, rho), rep, method, cross in selected:
        folder = args.baseline / stem(shape, mechanism, rho, method, cross, rep)
        expected = [folder / "complete.json"]
        for update in c["checkpoints"]:
            expected.extend((folder / f"state_{update:08d}.pt", folder / f"metrics_{update:08d}.csv",
                             folder / f"objective_{update:08d}.json"))
        for file in expected:
            if not file.is_file():
                raise ValueError(f"Missing baseline artifact: {file}")
        complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
        if complete["last_checkpoint"] != endpoint:
            raise ValueError(f"Incomplete baseline: {folder}")
    return original, selected


def extension_manifest(args, baseline):
    return dict(baseline_manifest_sha256=sha(args.baseline / "manifest.json"),
                baseline_manifest=baseline, extension_source_sha256=sha(Path(__file__)),
                continuation_targets=list(args.targets), ml_iterations=args.ml_iterations,
                ml_starts=args.ml_starts,
                smoke=args.smoke, methods=list(args.methods), cross=list(args.cross),
                device=args.device, baseline_path=str(args.baseline.resolve()))


def ensure_output(out, manifest):
    path = out / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Extension manifest mismatch; use original settings/code or a fresh output")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Nonempty extension output without matching manifest")
    else:
        out.mkdir(parents=True, exist_ok=True)
        d.atomic_json(path, manifest)


def restore(state_file, model, optimizer, rng, device):
    state = torch.load(state_file, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    rng.bit_generator.state = state["numpy_rng"]
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"] is not None and device == "cuda":
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
    return int(state["step"])


def continue_one(args, baseline, out, selected, panels, device):
    setting, rep, method, cross = selected
    shape, mechanism, rho = setting
    config = argparse.Namespace(**baseline["config"])
    config.rep = rep
    tag = stem(shape, mechanism, rho, method, cross, rep)
    source = args.baseline / tag
    dest = out / tag
    dest.mkdir(exist_ok=True)
    center, scale = d.feature_stats(panels["fit"])
    arrays = {key: d.input_array(value, method, shape, center, scale) for key, value in panels.items()}
    model = (d.Basis(arrays["fit"].shape[-1], torch.zeros(arrays["fit"].shape[-1], dtype=torch.float64))
             if method == "basis" else d.Net(tuple(config.width))).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    fit = d.panel_for_map(panels["fit"], "M1", config.sigma)
    validation = d.panel_for_map(panels["validation"], "M1", config.sigma)
    limits = [config.link_limit if config.link_limit is not None else
              max(city_count - 1, math.ceil(config.link_fraction * len(panel["c"])))
              for city_count, panel in ((config.cities, fit), (config.validation_cities, validation))]
    fit_links = d.tensors(d.stable_links(fit, method, shape, center, scale,
                                        p.seed(config.seed, rep, shape, mechanism, "fit_links"),
                                        cross, limits[0], config.link_limit), device)
    val_links = d.tensors(d.stable_links(validation, method, shape, center, scale,
                                        p.seed(config.seed, rep, shape, mechanism, "val_links"),
                                        cross, limits[1], config.link_limit), device)
    rng = np.random.default_rng(p.seed(config.seed, rep, shape, mechanism, "batches"))
    saved = sorted(dest.glob("state_*.pt"))
    state_path = saved[-1] if saved else source / f"state_{baseline['config']['checkpoints'][-1]:08d}.pt"
    if saved and any(int(x.stem.split("_")[-1]) not in args.targets for x in saved):
        raise ValueError(f"Unexpected extension state in {dest}")
    source_sha = sha(source / f"state_{baseline['config']['checkpoints'][-1]:08d}.pt")
    provenance_path = dest / "provenance.json"
    provenance = dict(baseline_state_sha256=source_sha, baseline_manifest_sha256=sha(args.baseline / "manifest.json"),
                      panel_fingerprints={k: fingerprint(v) for k, v in panels.items()},
                      baseline_checkpoint=baseline["config"]["checkpoints"][-1])
    if provenance_path.exists() and json.loads(provenance_path.read_text(encoding="utf-8")) != provenance:
        raise ValueError(f"Parent checkpoint or generated panels changed: {dest}")
    if not provenance_path.exists():
        d.atomic_json(provenance_path, provenance)
    step = restore(state_path, model, optimizer, rng, device)
    start = baseline["config"]["checkpoints"][-1]
    if step not in (start, *args.targets):
        raise ValueError(f"Unexpected restored update {step}")
    for update in range(step, args.targets[-1] + 1):
        if update > step:
            indices = d.batch_indices(fit, config, rng)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = d.objective(model, arrays["fit"], fit, indices, fit_links,
                                  "ranknet", config, method, device)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite objective at {tag}: {update}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.)
            optimizer.step()
        if update not in args.targets:
            continue
        state_file = dest / f"state_{update:08d}.pt"
        objective_file = dest / f"objective_{update:08d}.json"
        metric_file = dest / f"metrics_{update:08d}.csv"
        if update == step and objective_file.exists() and metric_file.exists():
            continue
        if update > step:
            d.checkpoint(state_file, update, model, optimizer, rng)
        exposure = update * min(config.batch_cities, config.cities) * config.batch_tracts
        metadata = dict(stage="extension", origin="continued checkpoint", seed=rep,
                        scenario=f"{shape}/{mechanism}/rho{rho:g}", method=method,
                        rank="ranknet" if method != "supervised" else "none", cross=int(cross),
                        trained_map="M1" if method != "supervised" else "truth", updates=update,
                        incremental_updates=update-start, selected_tract_records=exposure,
                        effective_tract_passes=exposure/(config.cities*config.n),
                        source_fit_cities=config.cities, source_fit_n=config.n,
                        temporal_count=len(fit_links[0][0]),
                        temporal_coverage=len(fit_links[0][0])/len(fit["c"]),
                        cross_count=0 if fit_links[1] is None else len(fit_links[1][0]),
                        cross_coverage=0 if fit_links[1] is None else len(fit_links[1][0])/len(fit["c"]),
                        validation_temporal_count=len(val_links[0][0]),
                        validation_cross_count=0 if val_links[1] is None else len(val_links[1][0]))
        d.atomic_json(objective_file, dict(train=d.monitored_objective(model, arrays["fit"], fit,
                                                                        fit_links, "ranknet", config, method, device),
                                            validation=d.monitored_objective(model, arrays["validation"], validation,
                                                                              val_links, "ranknet", config, method, device),
                                            exposure=metadata))
        d.atomic_csv(metric_file, d.evaluate(model, fit, panels, arrays, shape, d.MAPS,
                                              method, config, metadata, device))
        print(f"extension {tag}: {update}/{args.targets[-1]}", flush=True)
    if not all((dest/f"state_{t:08d}.pt").exists() and (dest/f"objective_{t:08d}.json").exists()
               and (dest/f"metrics_{t:08d}.csv").exists() for t in args.targets):
        raise RuntimeError(f"Incomplete extension trajectory {tag}")
    d.atomic_json(dest/"complete.json", dict(last_checkpoint=args.targets[-1], status="complete"))


def benchmark_one(args, baseline, out, panels, setting, rep):
    shape, mechanism, rho = setting
    fit = panels["fit"]
    fp = fingerprint(fit)
    config = argparse.Namespace(**baseline["config"])
    rows = []
    for family in d.MAPS:
        name = f"ml_{shape}_{mechanism}_rho{rho:g}_{family}_seed{rep}"
        path = out / f"{name}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["fit_fingerprint"] != fp:
                raise ValueError(f"Fit panel fingerprint changed: {path}")
            pars, info = saved["parameters"], saved["optimization"]
        else:
            mapped = d.panel_for_map(fit, family, config.sigma)
            pars, info = ml_fit(mapped, shape, family, rho, config.sigma, args.ml_iterations,
                                args.device, args.ml_starts)
            d.atomic_json(path, dict(parameters=pars, optimization=info, fit_fingerprint=fp,
                                     source_fit_cities=config.cities, source_fit_n=config.n,
                                     scenario=f"{shape}/{mechanism}/rho{rho:g}", map=family, seed=rep))
        raw = {key: ml_predict(panel, shape, pars["coeff"])
               for key, panel in panels.items() if key in ("fit", "source_test", "target_test",
                                                              "source_threshold", "target_threshold")}
        alignment = p.fit_alignment(raw["fit"], fit)
        for aligned in (True, False):
            off, slope, status = alignment if aligned else (0., 1., "raw canonical population normalization")
            for split in ("source_test", "target_test"):
                threshold_name = "source_threshold" if split.startswith("source") else "target_threshold"
                stable = panels[threshold_name]
                threshold_f = off + slope * raw[threshold_name]
                cutoff = float(np.quantile(np.abs(np.diff(threshold_f[stable["category"] < 2], axis=1)),
                                           .95, method="higher"))
                target = d.panel_for_map(panels[split], family, config.sigma)
                fhat = off + slope * raw[split]
                muhat = d.oracle_mu(target, fhat, family, config.sigma)
                metadata = dict(stage="extension", origin="same-panel parametric likelihood",
                                method="ml", scenario=f"{shape}/{mechanism}/rho{rho:g}", seed=rep,
                                map=family, panel=split, updates="", incremental_updates="",
                                source_fit_cities=config.cities, source_fit_n=config.n,
                                alignment_offset=off, alignment_slope=slope,
                                alignment_status=status, evaluation="matched fit-only affine" if aligned else "raw canonical",
                                ml_convergence=info["convergence"], ml_gradient_norm=info["gradient_norm"],
                                ml_iterations=info["iterations"], ml_selected_start=info["start"],
                                map_info="privileged true cell map and Gaussian marginal; no target map fitting")
                rows.extend(d.metric_rows(target, fhat, muhat, cutoff, metadata))
    return rows


def report_and_plots(out, baseline_rows, new_rows, ml_rows, baseline, args):
    combined = baseline_rows + new_rows + ml_rows
    d.atomic_csv(out/"combined_metrics.csv", combined)
    d.atomic_csv(out/"ml_metrics.csv", ml_rows)
    if not combined:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for scenario in sorted(set(r["scenario"] for r in combined)):
        for family in d.MAPS:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            for ax, metric in zip(axes.flat, d.METRICS):
                for method in ("nn", "basis", "supervised"):
                    for cross in ((0,) if method in ("basis", "supervised") else (0, 1)):
                        subset = [r for r in baseline_rows+new_rows if r["scenario"] == scenario and
                                  r["map"] == family and r["panel"] == "target_test" and
                                  r["stratum"] == "all" and r["method"] == method and int(r["cross"]) == cross]
                        if not subset:
                            continue
                        times = sorted(set(int(r["updates"]) for r in subset))
                        means = [np.mean([float(r[metric]) for r in subset if int(r["updates"]) == t]) for t in times]
                        ax.plot(times, means, marker=".", label=f"{method} cross={cross}")
                        for t in times:
                            vals = [float(r[metric]) for r in subset if int(r["updates"]) == t]
                            ax.vlines(t, min(vals), max(vals), alpha=.25)
                ml_subset = [r for r in ml_rows if r["scenario"] == scenario and
                             r["map"] == family and r["panel"] == "target_test" and r["stratum"] == "all" and
                             r["evaluation"] == "matched fit-only affine"]
                matches = [float(r[metric]) for r in ml_subset]
                if matches:
                    uncertified = sum(r["ml_convergence"] != "gradient tolerance met" for r in ml_subset)
                    color = "darkorange" if uncertified else "black"
                    label = (f"ML mean; {uncertified}/{len(matches)} fail gradient check"
                             if uncertified else "ML mean; gradient checks passed")
                    ax.axhline(np.mean(matches), color=color, linestyle="--", label=label)
                    ax.axhspan(min(matches), max(matches), color=color, alpha=.12)
                ax.set_title(metric)
                ax.set_yscale("log")
                ax.set_xlabel("total Adam updates; ML uses full-panel likelihood")
                ax.set_ylabel("MSE, log scale (seed mean; lines show range)")
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
            fig.suptitle(f"{scenario} / {family}: target oracle-map evaluation", y=.99)
            fig.tight_layout(rect=[0, .10, 1, .95])
            fig.savefig(out / f"comparison_{scenario.replace('/', '_')}_{family}.png", dpi=145)
            plt.close(fig)
    fit_rows = {(r["scenario"], r["map"], r["seed"]): r for r in ml_rows}
    failures = sorted((key, value) for key, value in fit_rows.items()
                      if value["ml_convergence"] != "gradient tolerance met")
    lines = ["# Ordinal continuation and matched likelihood", "",
             f"Baseline endpoint: {baseline['config']['checkpoints'][-1]}; new endpoints: {args.targets}.",
             f"Baseline rows: {len(baseline_rows)}; extension rows: {len(new_rows)}; ML rows: {len(ml_rows)}.",
             "Source truth enters only fit-only evaluation alignment; true target maps enter oracle evaluation.",
             "Likelihood fits use source W, known basis family and noise, population latent normalization, "
             "and constrained positive first coefficient; no target map is fitted.",
             "Seed ranges are descriptive and paired by seed; optimizer convergence is reported per ML fit.",
             f"Continuation trajectories completed: {sum((out/stem(*setting, method, cross, rep)/'complete.json').exists() for setting in d.SCENARIOS for rep in range(1, baseline['config']['seeds']+1) for method in args.methods for cross in (args.cross if method != 'supervised' else (0,)))} / {len(d.SCENARIOS)*baseline['config']['seeds']*sum(len(args.cross) if method != 'supervised' else 1 for method in args.methods)}.",
             f"Distinct ML fits completed: {len(fit_rows)} / {len(d.SCENARIOS)*baseline['config']['seeds']*len(d.MAPS)}.",
             f"ML fits without certified gradient tolerance: {len(failures)}.", "",
             *[f"- {key[0]} / {key[1]} / seed {key[2]}: {value['ml_convergence']}; iterations={value['ml_iterations']}; gradient={value['ml_gradient_norm']}; selected start={value['ml_selected_start']}"
               for key, value in failures], "",
             "See combined_metrics.csv for all/stable/small/large strata and source/target splits.", ""]
    d.atomic_bytes(out/"REPORT.md", "\n".join(lines).encode("utf-8"))


def self_check():
    # Small analytic consistency checks only; no baseline/training writes.
    y = torch.tensor([[.31, -.52]], dtype=torch.float64, requires_grad=True)
    k = torch.tensor([[.4, .5]], dtype=torch.float64, requires_grad=True)
    r = p.inverse_torch(y, k)
    assert torch.allclose(r + k * torch.tanh(r), y, atol=1e-10)
    dy = torch.autograd.grad(r.sum(), y, retain_graph=True)[0]
    dk = torch.autograd.grad(r.sum(), k)[0]
    expected = 1 + k.detach() * (1 - torch.tanh(r.detach()).square())
    torch.testing.assert_close(dy, 1/expected)
    torch.testing.assert_close(dk, -torch.tanh(r.detach())/expected)
    fd_eps = 1e-5
    plus = p.inverse_numpy(y.detach().numpy(), k.detach().numpy()+fd_eps)
    minus = p.inverse_numpy(y.detach().numpy(), k.detach().numpy()-fd_eps)
    np.testing.assert_allclose(dk.detach().numpy(), (plus-minus)/(2*fd_eps), atol=1e-8)
    sigma, rho = .8, .9
    e = np.array([.3, -.7])
    cov = sigma**2 * np.array([[1., rho], [rho, 1.]])
    numeric = .5 * e @ np.linalg.inv(cov) @ e
    closed = .5 * (e@e - 2*rho*e[0]*e[1])/(sigma**2*(1-rho*rho))
    np.testing.assert_allclose(closed, numeric)
    # Known-parameter W density equals a multivariate Gaussian transformed
    # through both cell maps, including each inverse-Jacobian contribution.
    f0 = np.array([.2, -.3]); a0 = np.array([.4, -.2])
    b0 = np.array([2.1, 1.8]); k0 = np.array([.5, .3])
    r0 = f0 + e
    w0 = p.psi(r0, a0, b0, k0)
    inverse = p.inverse_numpy((w0-a0)/b0, k0)
    np.testing.assert_allclose(inverse, r0, atol=1e-10)
    jac = b0*(1+k0*(1-np.tanh(inverse)**2))
    eps = 1e-5
    numeric_jac = (p.psi(r0+eps,a0,b0,k0)-p.psi(r0-eps,a0,b0,k0))/(2*eps)
    np.testing.assert_allclose(jac, numeric_jac, atol=1e-9)
    normalizer = np.log(2*np.pi)+.5*np.log(np.linalg.det(cov))
    np.testing.assert_allclose(closed+np.log(jac).sum()+normalizer,
                               .5*e@np.linalg.inv(cov)@e + normalizer + np.log(numeric_jac).sum())
    objective = paired_nll(torch.tensor(w0[None,:], dtype=torch.float64),
                           torch.tensor(f0[None,:], dtype=torch.float64),
                           torch.tensor(a0[None,:], dtype=torch.float64),
                           torch.tensor(b0[None,:], dtype=torch.float64),
                           torch.tensor(k0[None,:], dtype=torch.float64), rho, sigma, "M3")
    np.testing.assert_allclose(float(objective.detach())+normalizer,
                               closed+np.log(jac).sum()+normalizer)
    x = np.array([-.5, .2, .8]); a, b, kk = np.ones(3)*.4, np.ones(3)*2., np.ones(3)*.5
    rng = np.random.default_rng(42)
    mc = a+b*(x+kk*np.tanh(x+.8*rng.normal(size=(300000, 3))).mean(axis=0))
    np.testing.assert_allclose(p.conditional_mean(x,a,b,kk,.8), mc, atol=.009)
    class Tiny:
        n=16;eval_n=16;sigma=.8;cities=4;validation_cities=2;source_test_cities=2;target_cities=2
    panels = d.make_panels(Tiny,"quadratic","nonlinear",.9,42)
    for family in d.MAPS:
        target = d.panel_for_map(panels["target_test"],family,.8)
        oracle = d.metric_rows(target, target["f"], d.oracle_mu(target,target["f"],family,.8),0.,
                               dict(test="known truth"))
        assert all(row[key] == 0 for row in oracle for key in d.METRICS)
    # Round-trip the serialized optimizer/RNG and compare one uninterrupted step.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        torch.manual_seed(3)
        model = d.Net((4, 3)).double()
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        rng = np.random.default_rng(3)
        def update(net, optimizer):
            optimizer.zero_grad()
            net(torch.ones((3, 7), dtype=torch.float64)).square().sum().backward()
            optimizer.step()
        update(model, opt)
        update(model, opt)
        assert all("exp_avg" in opt.state[v] for v in model.parameters())
        path = Path(tmp)/"state.pt"
        d.checkpoint(path, 2, model, opt, rng)
        torch_expected = torch.rand(4)
        replica = d.Net((4, 3)).double()
        restored = torch.optim.Adam(replica.parameters(), lr=.001)
        other = np.random.default_rng(4)
        assert restore(path, replica, restored, other, "cpu") == 2
        np.testing.assert_array_equal(rng.integers(0, 100, 10), other.integers(0, 100, 10))
        torch.testing.assert_close(torch_expected, torch.rand(4))
        for left, right in zip(model.parameters(), replica.parameters()):
            torch.testing.assert_close(opt.state[left]["exp_avg"], restored.state[right]["exp_avg"])
            torch.testing.assert_close(opt.state[left]["exp_avg_sq"], restored.state[right]["exp_avg_sq"])
        for net, optimizer in ((model, opt), (replica, restored)):
            update(net, optimizer)
        for left, right in zip(model.parameters(), replica.parameters()):
            torch.testing.assert_close(left, right)
    print("Extension self-check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=d.ROOT/"ordinal_latent_core")
    parser.add_argument("--output", type=Path, default=d.ROOT/"ordinal_latent_extension_results")
    parser.add_argument("--mode", choices=("all", "continue", "benchmark"), default="all")
    parser.add_argument("--methods", nargs="+", choices=("nn", "basis", "supervised"), default=["nn"])
    parser.add_argument("--cross", nargs="+", type=int, choices=(0, 1), default=[1])
    parser.add_argument("--targets", nargs="+", type=int, default=None)
    parser.add_argument("--ml-iterations", type=int, default=None)
    parser.add_argument("--ml-starts", type=int, choices=(1, 2), default=2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.targets = sorted(set(args.targets or ([3, 4] if args.smoke else [9600, 19200])))
    args.ml_iterations = args.ml_iterations if args.ml_iterations is not None else (20 if args.smoke else 1000)
    args.cross = sorted(set(args.cross))
    args.methods = list(dict.fromkeys(args.methods))
    if args.ml_iterations < 1:
        parser.error("--ml-iterations must be positive")
    baseline, selected = validate_baseline(args)
    endpoint = baseline["config"]["checkpoints"][-1]
    if not args.targets or args.targets[0] <= endpoint or (args.smoke and args.targets != [3, 4]):
        parser.error("Extension targets must increase above the baseline; smoke targets are 3 and 4")
    if args.device == "cuda" and not args.plan and not args.self_check and not torch.cuda.is_available():
        parser.error("CUDA unavailable; run in WSL CUDA environment or select --device cpu")
    fits = baseline["config"]["seeds"]*len(d.SCENARIOS)*len(d.MAPS)
    effective = min(baseline['config']['batch_cities'],baseline['config']['cities'])*baseline['config']['batch_tracts']/(baseline['config']['cities']*baseline['config']['n'])
    print(f"PLAN mode={args.mode}: {len(selected) if args.mode != 'benchmark' else 0} continuation trajectories; "
          f"{len(selected)*len(args.targets) if args.mode != 'benchmark' else 0} new checkpoints; "
          f"{fits if args.mode != 'continue' else 0} distinct full-panel ML fits × {args.ml_starts} fixed starts "
          f"(at most {args.ml_iterations} LBFGS iterations/start); "
          f"each continuation adds {args.targets[-1]-endpoint} updates; "
          f"{effective*args.targets[-1]:g} total and {effective*(args.targets[-1]-endpoint):g} incremental "
          "selected-tract passes. One fit per scenario/map/seed serves both cross arms.", flush=True)
    if args.plan:
        return
    if args.self_check:
        self_check()
        return
    manifest = extension_manifest(args, baseline)
    ensure_output(args.output, manifest)
    new_rows, ml_rows, baseline_rows = [], [], []
    for setting in d.SCENARIOS:
        shape, mechanism, rho = setting
        for rep in range(1, baseline["config"]["seeds"]+1):
            panels = panels_for(baseline["config"], setting, rep)
            for selected_job in selected:
                if selected_job[0] != setting or selected_job[1] != rep:
                    continue
                tag = stem(shape,mechanism,rho,selected_job[2],selected_job[3],rep)
                if args.mode in ("all", "continue"):
                    continue_one(args, baseline, args.output, selected_job, panels, args.device)
                folder = args.output/tag
                for t in args.targets:
                    metric_file = folder/f"metrics_{t:08d}.csv"
                    if metric_file.exists():
                        new_rows.extend(read_csv(metric_file))
            # Context rows are always available, even when benchmark runs first.
            # Basis cross-on coincides in H with basis cross-off; show one curve.
            methods = [m for m in ("nn", "basis", "supervised") if m in baseline["config"]["methods"]]
            for method in methods:
                for cross in ((0, 1) if method == "nn" and baseline["config"]["links"] == "both"
                              else ((1,) if method == "nn" and baseline["config"]["links"] == "on" else (0,))):
                    if method == "basis" and cross != 0:
                        continue
                    tag = stem(shape,mechanism,rho,method,cross,rep)
                    baseline_folder = args.baseline/tag
                    if not baseline_folder.is_dir():
                        continue
                    for t in baseline["config"]["checkpoints"]:
                        metric_file = baseline_folder/f"metrics_{t:08d}.csv"
                        if metric_file.is_file():
                            for row in read_csv(metric_file):
                                row["origin"] = "frozen baseline"
                                row["incremental_updates"] = 0
                                baseline_rows.append(row)
            ml_cache = args.output/f"ml_rows_{shape}_{mechanism}_rho{rho:g}_seed{rep}.csv"
            if args.mode in ("all", "benchmark"):
                d.atomic_csv(ml_cache, benchmark_one(args,baseline,args.output,panels,setting,rep))
            if ml_cache.exists():
                ml_rows.extend(read_csv(ml_cache))
    report_and_plots(args.output,baseline_rows,new_rows,ml_rows,baseline,args)
    print("Extension complete; see REPORT.md and per-fit optimization status", flush=True)


if __name__ == "__main__":
    main()
