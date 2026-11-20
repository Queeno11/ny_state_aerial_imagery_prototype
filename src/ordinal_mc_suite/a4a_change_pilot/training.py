"""Matched ordinal, cardinal, and privileged training trajectories."""
import hashlib
import itertools
import time

import numpy as np
import torch

from fitting import cardinal, known_prediction, loss_parts, normalize, pair_stats, scores, selected_tau, stability, state
from heads import HeadSpec, MonotoneHead, fit_heads


def tensor_hash(params):
    h = hashlib.sha256()
    for name, value in params.items():
        h.update(name.encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def capture(model, fit, val, cfg, update, selector, metrics, native=None, calibrated=None):
    _, _, (mu, sd) = scores(model, fit, val, cfg)
    snapshot = state(model)
    return dict(update=update, selector=selector, model=snapshot, state_hash=tensor_hash(snapshot),
                score_mu=mu, score_sd=sd, metrics=metrics, native=native, calibrated=calibrated)


def measure_rank(z, data, cfg):
    actual = float(pair_stats(z, data, cfg["taus"][-1], actual=True))
    return dict(rank=actual, smooth=float(pair_stats(z, data, cfg["taus"][-1])),
                stability=float(stability(z, data)))


def criterion(rank, cfg):
    return rank["rank"] + cfg["lambda_stability"] * rank["stability"]


def composite(rank, level, change, cfg):
    return criterion(rank, cfg) + level + change


def mse(pred, panel, denoms):
    err = pred - panel.w
    return (float(np.mean(err**2) / denoms["W"]),
            float(np.mean((err[panel.m:] - err[:panel.m])**2) / denoms["delta_W"]))


def calibrate(z, panel, spec, denoms, cfg, start=None, level=1., change=1.):
    return fit_heads(z, panel, spec, denoms, cfg["head_max_iter"], level=level,
                     change=change, start=start)


def select_best(best, candidate):
    if best is None or candidate["selector"] < best["selector"]:
        return candidate
    return best


def save_trace(metrics, row, prefix):
    row.update({f"{prefix}_{key}": value for key, value in metrics.items()})


def check_updates(total, period):
    return {0, total, *range(period, total, period)}


def baseline(model, fit, val, cfg, denoms, trace):
    adam = torch.optim.Adam(model.parameters(), lr=cfg["baseline_lr"])
    best = None
    total = cfg["baseline_updates"]
    checkpoints = check_updates(total, cfg["check_every"])
    started = time.perf_counter()
    for update in range(total + 1):
        tau = selected_tau(update, total, cfg)
        if update in checkpoints:
            z, zv, (_, scale) = scores(model, fit, val, cfg)
            valid = measure_rank(zv, val, cfg)
            selector = criterion(valid, cfg)
            row = dict(path="anchor", update=update, tau=tau, selector=selector,
                       raw_v=scale**2, elapsed_seconds=time.perf_counter() - started)
            save_trace(valid, row, "val")
            save_trace(measure_rank(z, fit, cfg), row, "fit")
            trace.append(row)
            fixed = capture(model, fit, val, cfg, update, selector, valid)
            best = select_best(best, fixed)
        if update == total:
            break
        adam.zero_grad(set_to_none=True)
        z, _ = normalize(model(fit.inputs), fit, cfg["min_score_variance"])
        parts = loss_parts(z, fit, tau, denoms)
        (parts["rank"] + cfg["lambda_stability"] * parts["stability"]).backward()
        adam.step()
    model.load_state_dict(best["model"])
    return best, fixed


def gradient_audit(model, fit, denoms, cfg, head, weights, path, checkpoint):
    z, (_, sd) = normalize(model(fit.inputs), fit, cfg["min_score_variance"])
    parts = loss_parts(z, fit, cfg["taus"][-1], denoms, head)
    weighted = dict(rank=parts["rank"], stability=cfg["lambda_stability"] * parts["stability"],
                    level=weights[0] * parts["level"], change=weights[1] * parts["change"],
                    curvature=weights[2] * parts["curvature"])
    gradients, rows = {}, []
    params = tuple(model.parameters())
    for name, loss in {**parts, **{f"weighted_{k}": v for k, v in weighted.items()}}.items():
        grad = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        g = torch.cat([(p.new_zeros(p.shape) if x is None else x).ravel() for x, p in zip(grad, params)])
        gradients[name] = g
        rows.append(dict(path=path, checkpoint=checkpoint, component=name, comparison="",
                         loss=float(loss.detach()), grad_norm=float(g.norm()), cosine=np.nan,
                         raw_v=float(sd.detach()**2)))
    for first, second in itertools.combinations(parts, 2):
        a, b = gradients[first], gradients[second]
        denom = a.norm() * b.norm()
        rows.append(dict(path=path, checkpoint=checkpoint, component=first, comparison=second,
                         cosine=float(a.dot(b) / denom) if float(denom) > 0 else np.nan))
    total = sum(weighted.values())
    observed = torch.cat([x.ravel() for x in torch.autograd.grad(total, params)])
    expected = sum(gradients["weighted_" + name] for name in weighted)
    torch.testing.assert_close(observed, expected, rtol=2e-4, atol=2e-6)
    rows.append(dict(path=path, checkpoint=checkpoint, component="total", comparison="",
                     loss=float(total.detach()), grad_norm=float(observed.norm())))
    return rows


def refit_record(model, fit, val, train_panel, val_panel, spec, denoms, cfg,
                 update, previous_head, native, path, trace):
    z, zv, (_, sd) = scores(model, fit, val, cfg)
    train_z, val_z = z[:fit.n].cpu().double().numpy(), zv[:val.n].cpu().double().numpy()
    coefficients, info = calibrate(train_z, train_panel, spec, denoms, cfg, start=previous_head)
    valid = measure_rank(zv, val, cfg)
    level, change = mse(spec.direct(coefficients, val_z, val_panel.cells), val_panel, denoms)
    fitted = measure_rank(z, fit, cfg)
    train_level, train_change = mse(spec.direct(coefficients, train_z, train_panel.cells), train_panel, denoms)
    valid.update(level=level, change=change)
    selector = composite(valid, level, change, cfg)
    row = dict(path=path, update=update, tau=cfg["taus"][-1], selector=selector, raw_v=sd**2,
               fit_refit_level=train_level, fit_refit_change=train_change,
               head_iterations=info["iterations"], head_projected_grad=info["projected_gradient"],
               head_status=info["solver_status"], calibrated_slope_min=float(coefficients[:, 1:].min()),
               calibrated_slope_max=float(coefficients[:, 1:].max()))
    save_trace(valid, row, "val")
    save_trace(fitted, row, "fit")
    if native is not None:
        n_fit_level, n_fit_change = mse(spec.direct(native, train_z, train_panel.cells), train_panel, denoms)
        n_val_level, n_val_change = mse(spec.direct(native, val_z, val_panel.cells), val_panel, denoms)
        row.update(fit_native_level=n_fit_level, fit_native_change=n_fit_change,
                   val_native_level=n_val_level, val_native_change=n_val_change,
                   native_slope_min=float(native[:, 1:].min()), native_slope_max=float(native[:, 1:].max()))
    trace.append(row)
    return capture(model, fit, val, cfg, update, selector, valid, native, coefficients)


def branch(path, weights, model, anchor, fit, val, train_panel, val_panel, spec, denoms, cfg,
           common_head, trace, gradient_rows):
    model.load_state_dict(anchor["model"])
    active = any(weights)
    head = MonotoneHead(spec, common_head, fit.inputs.dtype, fit.inputs.device)
    groups = [dict(params=model.parameters(), lr=cfg["joint_lr"])]
    if active:
        groups.append(dict(params=head.parameters(), lr=cfg["head_lr"]))
    adam = torch.optim.Adam(groups)
    best = positive_best = rank_only = None
    total = cfg["joint_updates"]
    checkpoints = check_updates(total, cfg["check_every"])
    previous_head = common_head
    started = time.perf_counter()
    gradient_rows.extend(gradient_audit(model, fit, denoms, cfg, head, weights, path, "start"))
    for update in range(total + 1):
        if update in checkpoints:
            fixed = refit_record(model, fit, val, train_panel, val_panel, spec, denoms, cfg, update,
                                 previous_head, head.coefficients() if active else None, path, trace)
            trace[-1]["elapsed_seconds"] = time.perf_counter() - started
            previous_head = fixed["calibrated"]
            best = select_best(best, fixed)
            if update:
                positive_best = select_best(positive_best, fixed)
            if rank_only is None or criterion(fixed["metrics"], cfg) < criterion(rank_only["metrics"], cfg):
                rank_only = fixed
        if update == total:
            break
        adam.zero_grad(set_to_none=True)
        z, _ = normalize(model(fit.inputs), fit, cfg["min_score_variance"])
        terms = loss_parts(z, fit, cfg["taus"][-1], denoms, head if active else None)
        loss = terms["rank"] + cfg["lambda_stability"] * terms["stability"]
        if active:
            loss = loss + weights[0] * terms["level"] + weights[1] * terms["change"] + weights[2] * terms["curvature"]
        loss.backward()
        adam.step()
        if active:
            head.project()
    model.load_state_dict(best["model"])
    chosen_head = MonotoneHead(spec, best["native"] if active else best["calibrated"],
                              fit.inputs.dtype, fit.inputs.device)
    gradient_rows.extend(gradient_audit(model, fit, denoms, cfg, chosen_head, weights, path, "selected"))
    return dict(selected=best, selected_active=positive_best, fixed=fixed, selected_rank=rank_only)


def finalize(record, model, fit, val, panel, val_panel, spec, denoms, cfg, trace, label):
    model.load_state_dict(record["model"])
    z, zv, _ = scores(model, fit, val, cfg)
    train_z = z[:fit.n].cpu().double().numpy()
    val_z = zv[:val.n].cpu().double().numpy()
    cold, info = calibrate(train_z, panel, spec, denoms, cfg)
    warm = record["calibrated"]
    differences = np.max(np.abs(spec.direct(warm, val_z, val_panel.cells)
                                - spec.direct(cold, val_z, val_panel.cells)))
    record = {**record, "calibrated": cold}
    trace.append(dict(path=label, update=record["update"], fresh_cold_warm_prediction_max=float(differences),
                      iterations=info["iterations"], projected_gradient=info["projected_gradient"],
                      solver_status=info["solver_status"]))
    return record


def privileged(model, fit, val, denoms, cfg, path, trace):
    initial = tensor_hash(state(model))
    total = cfg["baseline_updates"]
    checkpoints = check_updates(total, cfg["check_every"])
    adam = torch.optim.Adam(model.parameters(), lr=cfg["baseline_lr"])
    direct = path == "direct_f"
    def direct_loss(z, data):
        return (z[:data.n] - (data.target_f - fit.mean_f) / fit.sd_f).square().mean()
    best = None
    started = time.perf_counter()
    for update in range(total + 1):
        tau = selected_tau(update, total, cfg)
        if update in checkpoints:
            z, zv, (_, sd) = scores(model, fit, val, cfg)
            valid = measure_rank(zv, val, cfg)
            lw, ld = cardinal(known_prediction(zv, val, fit), val, denoms)
            selector = float(direct_loss(zv, val)) if direct else composite(valid, float(lw), float(ld), cfg)
            valid.update(level=float(lw), change=float(ld), direct_f=float(direct_loss(zv, val)))
            row = dict(path=path, update=update, tau=tau, selector=selector, raw_v=sd**2,
                       elapsed_seconds=time.perf_counter() - started)
            save_trace(valid, row, "val")
            save_trace(measure_rank(z, fit, cfg), row, "fit")
            row["fit_direct_f"] = float(direct_loss(z, fit))
            fl, fd = cardinal(known_prediction(z, fit, fit), fit, denoms)
            row.update(fit_level=float(fl), fit_change=float(fd))
            trace.append(row)
            fixed = capture(model, fit, val, cfg, update, selector, valid)
            best = select_best(best, fixed)
        if update == total:
            break
        adam.zero_grad(set_to_none=True)
        z, _ = normalize(model(fit.inputs), fit, cfg["min_score_variance"])
        if direct:
            loss = direct_loss(z, fit)
        else:
            parts = loss_parts(z, fit, tau, denoms, known_fit=fit)
            loss = parts["rank"] + cfg["lambda_stability"] * parts["stability"]
            if path == "known_full":
                loss = loss + cfg["joint_weights"][0] * (parts["level"] + parts["change"])
        loss.backward()
        adam.step()
    return dict(selected=best, fixed=fixed, initial_hash=initial)
