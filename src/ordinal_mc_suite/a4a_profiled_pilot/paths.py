"""Own-head selection and matched ordinal, Adam, and profiled trajectories."""
from dataclasses import dataclass
import time

import numpy as np
import torch

from prior import ORIGINAL
from fitting import loss_parts, normalize, scores
from heads import MonotoneHead
from profile_objective import cardinal_objective, fitted_module, objective, solve
from training import capture, check_updates, composite, measure_rank, mse, refit_record


@dataclass
class Problem:
    model: object
    anchor: dict
    fit: object
    val: object
    fit_aux: np.ndarray
    panel: object
    valid: object
    cfg: dict
    protocol: dict
    spec: object
    denoms: dict
    options: dict
    initial: dict
    common_head: np.ndarray


def simpler(record, protocol):
    head = record["head_name"]
    return protocol["head_complexity"][head], protocol["head_tie_order"].index(head)


def choose(records, protocol, family=False):
    best = min(x["selector"] for x in records)
    eligible = [x for x in records if x["selector"] <= best + protocol["tie_tolerance"]]
    if family:
        return min(eligible, key=lambda x: (simpler(x, protocol)[0], x["update"], simpler(x, protocol)[1]))
    return min(eligible, key=lambda x: (x["update"], *simpler(x, protocol)))


def own_checkpoint(p, name, update, head, coefs, z, zv, head_info, own_trace):
    ranks = measure_rank(zv, p.val, p.cfg)
    zv_np = zv[:p.val.n].cpu().double().numpy()
    val_w, val_d = mse(p.spec.direct(coefs, zv_np, p.valid.cells), p.valid, p.denoms)
    ranks.update(level=val_w, change=val_d)
    value = composite(ranks, val_w, val_d, p.cfg)
    record = capture(p.model, p.fit, p.val, p.cfg, update, value, ranks, native=coefs)
    record.update(head_name=head, path=name, selection_protocol="own")
    own_trace.append(dict(path=name, head_name=head, update=update, selector=value, **ranks,
                          state_hash=record["state_hash"], raw_v=record["score_sd"]**2,
                          profiled_head_projected_grad=head_info["projected_gradient"]))
    return record


def frozen_options(p, z, starts=None):
    fits = {}
    for head, weights in p.options.items():
        start = None if starts is None else starts[head][0]
        fits[head] = solve(z, p.panel, p.spec, p.denoms, p.cfg, weights, start=start)
    return fits


def checks_for_records(records, p):
    total = p.cfg["joint_updates"]
    return dict(selected=choose(records, p.protocol),
                selected_active=choose([x for x in records if x["update"] > 0], p.protocol),
                fixed=choose([x for x in records if x["update"] == total], p.protocol))


def ordinal_path(p, own_trace, common_trace):
    p.model.load_state_dict(p.anchor["model"])
    adam = torch.optim.Adam(p.model.parameters(), lr=p.cfg["joint_lr"])
    total = p.cfg["joint_updates"]
    checks = check_updates(total, p.cfg["check_every"])
    fitted = p.initial
    previous_common = p.common_head
    records, common_records = [], []
    for update in range(total + 1):
        if update in checks:
            with torch.no_grad():
                z, zv, _ = scores(p.model, p.fit, p.val, p.cfg)
                fitted = p.initial if update == 0 else frozen_options(p, z, fitted)
                for head, (coefs, info) in fitted.items():
                    records.append(own_checkpoint(p, "O", update, head, coefs, z, zv, info, own_trace))
                historical = refit_record(p.model, p.fit, p.val, p.panel, p.valid, p.spec, p.denoms,
                                          p.cfg, update, previous_common, None, "O", common_trace)
                previous_common = historical["calibrated"]
                common_records.append(historical)
        if update == total:
            break
        adam.zero_grad(set_to_none=True)
        z, _ = normalize(p.model(p.fit.inputs), p.fit, p.cfg["min_score_variance"])
        parts = loss_parts(z, p.fit, p.cfg["taus"][-1], p.denoms)
        (parts["rank"] + p.cfg["lambda_stability"] * parts["stability"]).backward()
        adam.step()
    endpoints = checks_for_records(records, p)
    historical = min(common_records, key=lambda x: (x["selector"], x["update"]))
    return endpoints, historical, choose([x for x in records if x["update"] == 0], p.protocol)


def cardinal_path(p, name, mode, head_name, own_trace, common_trace, fit_trace):
    p.model.load_state_dict(p.anchor["model"])
    total = p.cfg["joint_updates"]
    checks = check_updates(total, p.cfg["check_every"])
    weights = p.options[head_name]
    optimal, info = p.initial[head_name]
    previous_common = p.common_head
    groups = [dict(params=p.model.parameters(), lr=p.cfg["joint_lr"])]
    profiled = mode == "profile"
    if not profiled:
        init = p.common_head if mode == "legacy" else optimal
        native = MonotoneHead(p.spec, init, p.fit.inputs.dtype, p.fit.inputs.device)
        groups.append(dict(params=native.parameters(), lr=p.cfg["head_lr"]))
    adam = torch.optim.Adam(groups)
    records, history = [], []
    for update in range(total + 1):
        started = time.perf_counter()
        adam.zero_grad(set_to_none=True)
        z, _ = normalize(p.model(p.fit.inputs), p.fit, p.cfg["min_score_variance"])
        if update and (profiled or update in checks):
            optimal, info = solve(z, p.panel, p.spec, p.denoms, p.cfg, weights, start=optimal)
        head = fitted_module(p.spec, optimal, p.fit) if profiled else native
        value, parts = objective(z, p.fit, head, p.denoms, p.cfg, weights)
        if profiled or update in checks:
            vals = {key: float(v.detach()) for key, v in parts.items()}
            weighted = sum(weights[i] * vals[key] for i, key in enumerate(("level", "change", "curvature")))
            exact = info["objective"] * weights[0]
            if profiled and abs(weighted - exact) > 5e-7:
                raise ArithmeticError("Profile head and scorer cardinal objectives differ")
            if update in checks and not profiled and weighted < exact - 5e-7:
                raise ArithmeticError("Native head appears to beat the fitted conditional optimum")
            fit_trace.append(dict(path=name, head_name=head_name, update=update, fit_objective=float(value.detach()),
                                  cardinal_loss=weighted, conditional_cardinal_optimum=exact,
                                  head_gap=weighted - exact, projected_head_gradient=info["projected_gradient"],
                                  head_iterations=info["iterations"], **vals))
        if update in checks:
            # At the cap this is the head solved on the just-computed final scorer, not the previous step.
            z_eval, zv, _ = scores(p.model, p.fit, p.val, p.cfg)
            coefs = optimal if profiled else native.coefficients()
            record = own_checkpoint(p, name, update, head_name, coefs, z_eval, zv, info, own_trace)
            records.append(record)
            if mode == "legacy":
                old = refit_record(p.model, p.fit, p.val, p.panel, p.valid, p.spec, p.denoms,
                                   p.cfg, update, previous_common, coefs, name, common_trace)
                previous_common = old["calibrated"]
                history.append((old, record))
        if update == total:
            fit_trace[-1]["step_seconds"] = time.perf_counter() - started
            break
        value.backward()
        adam.step()
        if not profiled:
            native.project()
        if profiled or update in checks:
            fit_trace[-1]["step_seconds"] = time.perf_counter() - started
    endpoints = checks_for_records(records, p)
    if history:
        historical, own = min(history, key=lambda x: (x[0]["selector"], x[0]["update"]))
        endpoints["selected_common"] = {**own, "historical_selector": historical["selector"],
                                        "selection_protocol": "historical_common"}
    return endpoints, records


def cold_check(p, label, endpoint, record):
    p.model.load_state_dict(record["model"])
    z, zv, (mu, sd) = scores(p.model, p.fit, p.val, p.cfg)
    assert mu == record["score_mu"] and sd == record["score_sd"]
    weights = p.options[record["head_name"]]
    cold, info = solve(z, p.panel, p.spec, p.denoms, p.cfg, weights)
    saved = record["native"]
    cold_obj = cardinal_objective(z, p.panel, p.spec, p.denoms, weights, cold)
    saved_obj = cardinal_objective(z, p.panel, p.spec, p.denoms, weights, saved)
    zv_np = zv[:p.val.n].cpu().double().numpy()
    pred_gap = float(np.max(np.abs(p.spec.direct(cold, zv_np, p.valid.cells)
                                  - p.spec.direct(saved, zv_np, p.valid.cells))))
    if label.startswith("Profile_") and abs(cold_obj - saved_obj) > 5e-8:
        raise ArithmeticError("Saved profile head does not match the cold conditional objective")
    return dict(arm=label, endpoint=endpoint, update=record["update"], saved_objective=saved_obj,
                cold_objective=cold_obj, saved_minus_cold=saved_obj-cold_obj,
                cold_projected_gradient=info["projected_gradient"], validation_prediction_max_diff=pred_gap)
