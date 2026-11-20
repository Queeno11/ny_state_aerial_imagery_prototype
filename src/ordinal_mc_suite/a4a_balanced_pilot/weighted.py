"""Weighted continuations with the original fit-only head refits and validation selector."""
import itertools
import time

import numpy as np
import torch

from prior import ORIGINAL
from fitting import cardinal, known_prediction, loss_parts, normalize, scores
from heads import MonotoneHead
from training import (capture, check_updates, composite, criterion, measure_rank, refit_record,
                      save_trace, select_best, tensor_hash)
from fitting import selected_tau, state


def vector(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p, g in zip(parameters, gradients)])


def moments(optimizer, model):
    group = optimizer.param_groups[0]
    beta = group["betas"][1]
    pieces = []
    for p in model.parameters():
        state_p = optimizer.state[p]
        if not state_p:
            return None
        corrected = state_p["exp_avg_sq"] / (1 - beta**float(state_p["step"]))
        pieces.append((corrected.sqrt() + group["eps"]).flatten())
    return torch.cat(pieces)


def gradient_table(model, fit, denoms, cfg, head, weights, *, path, update, optimizer=None, known=False):
    z, (_, sd) = normalize(model(fit.inputs), fit, cfg["min_score_variance"])
    parts = loss_parts(z, fit, cfg["taus"][-1], denoms, head, known_fit=fit if known else None)
    params = tuple(model.parameters())
    vectors = {name: vector(loss, params) for name, loss in parts.items() if loss.requires_grad}
    vectors["weighted_stability"] = cfg["lambda_stability"] * vectors["stability"]
    vectors["ordinal_net"] = vectors["rank"] + vectors["weighted_stability"]
    vectors["weighted_level"] = weights[0] * vectors["level"]
    vectors["weighted_change"] = weights[1] * vectors["change"]
    vectors["total"] = vectors["ordinal_net"] + vectors["weighted_level"] + vectors["weighted_change"]
    def record(family, gradients):
        table = []
        for label, g in gradients.items():
            table.append(dict(path=path, update=update, family=family, component=label, comparison="",
                              norm=float(g.norm()), raw_v=float(sd.detach()**2)))
        selected = ["rank", "weighted_stability", "level", "change", "ordinal_net"]
        if family == "head":
            selected = ["level", "change", "curvature"]
        for a, b in itertools.combinations(selected, 2):
            x, y = gradients[a], gradients[b]
            product = x.norm() * y.norm()
            table.append(dict(path=path, update=update, family=family, component=a, comparison=b,
                              cosine=float(x.dot(y) / product) if float(product) > 0 else np.nan))
        return table
    rows = record("scorer", vectors)
    if head is not None:
        hp = tuple(head.parameters())
        head_vectors = {name: vector(parts[name], hp) for name in ("level", "change", "curvature")}
        for name, w in zip(("level", "change", "curvature"), weights):
            head_vectors["weighted_" + name] = w * head_vectors[name]
        rows.extend(record("head", head_vectors))
    if optimizer is not None and update:
        denominator = moments(optimizer, model)
        if denominator is not None:
            rows.extend(record("scorer_shared_adam_v", {key: g / denominator for key, g in vectors.items()}))
    norm = lambda name: float(vectors[name].norm())
    ref = float(np.hypot(norm("rank"), norm("weighted_stability")))
    return rows, dict(rank=norm("rank"), weighted_stability=norm("weighted_stability"), ref=ref,
                      ordinal_net=norm("ordinal_net"), level=norm("level"), change=norm("change"))


def continuation(path, weights, model, anchor, fit, val, panel, valid, spec, denoms, cfg, head0,
                 trace, gradient_rows=None, adam_checks=()):
    model.load_state_dict(anchor["model"])
    active = any(weights)
    head = MonotoneHead(spec, head0, fit.inputs.dtype, fit.inputs.device)
    groups = [dict(params=model.parameters(), lr=cfg["joint_lr"])]
    if active:
        groups.append(dict(params=head.parameters(), lr=cfg["head_lr"]))
    adam = torch.optim.Adam(groups)
    best = best_active = rank_only = None
    checkpoints = check_updates(cfg["joint_updates"], cfg["check_every"])
    previous_head = head0
    started = time.perf_counter()
    for update in range(cfg["joint_updates"] + 1):
        if update in checkpoints:
            fixed = refit_record(model, fit, val, panel, valid, spec, denoms, cfg, update, previous_head,
                                 head.coefficients() if active else None, path, trace)
            trace[-1]["elapsed_seconds"] = time.perf_counter() - started
            previous_head = fixed["calibrated"]
            best = select_best(best, fixed)
            if update:
                best_active = select_best(best_active, fixed)
            if rank_only is None or criterion(fixed["metrics"], cfg) < criterion(rank_only["metrics"], cfg):
                rank_only = fixed
            if gradient_rows is not None and update in adam_checks:
                rows, _ = gradient_table(model, fit, denoms, cfg, head, weights, path=path,
                                         update=update, optimizer=adam)
                gradient_rows.extend(rows)
        if update == cfg["joint_updates"]:
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
    if gradient_rows is not None:
        model.load_state_dict(best["model"])
        chosen = MonotoneHead(spec, best["native"] if active else best["calibrated"],
                              fit.inputs.dtype, fit.inputs.device)
        rows, _ = gradient_table(model, fit, denoms, cfg, chosen, weights, path=path + "_selected",
                                 update=best["update"])
        gradient_rows.extend(rows)
    return dict(selected=best, selected_active=best_active, fixed=fixed, selected_rank=rank_only)


def known_control(model, fit, val, denoms, cfg, weights, path, trace, gradients):
    total = cfg["baseline_updates"]
    checkpoints = check_updates(total, cfg["check_every"])
    adam = torch.optim.Adam(model.parameters(), lr=cfg["baseline_lr"])
    initial_hash = tensor_hash(state(model))
    direct = path == "direct_f"
    use_full = path == "known_full"
    best = None
    started = time.perf_counter()
    def direct_loss(z, data):
        return (z[:data.n] - (data.target_f - fit.mean_f) / fit.sd_f).square().mean()
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
            if use_full:
                loss = loss + weights[0] * parts["level"] + weights[1] * parts["change"]
        loss.backward()
        adam.step()
    model.load_state_dict(best["model"])
    if not direct:
        applied = weights if use_full else (0., 0., 0.)
        rows, _ = gradient_table(model, fit, denoms, cfg, None, applied,
                                 path=path + "_selected", update=best["update"], known=True)
        gradients.extend(rows)
    return dict(selected=best, fixed=fixed, initial_hash=initial_hash)
