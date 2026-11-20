"""Paired A4(a) outcomes and independent panels, using the suite's structural DGP."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.special import ndtr

from ordinal_mc.core import seed_for
from ordinal_mc.design import PopulationDGP, make_inputs, matched_panel, structural_gradient
from ordinal_mc.e6.data import cell_parameters, measurement

TRUTH = "quadratic"


def structural_dgp(cfg):
    runtime = SimpleNamespace(**cfg, noise_sd=0.0, bayes_target=None, rff=16)
    dgp = PopulationDGP(runtime)
    dgp.configure(TRUTH, cfg["composition"])
    return runtime, dgp


def fit_moments(x, n_cells):
    cell_values = np.asarray(x).reshape(n_cells, -1)
    return float(cell_values.mean()), float(cell_values.var(axis=1, ddof=1).mean())


def draw_contexts(seed, n):
    return np.random.default_rng(seed).normal(size=(n, 2))


@dataclass
class Panel:
    seed: int
    x: np.ndarray
    h: np.ndarray
    f: np.ndarray
    mean_w: np.ndarray
    w: np.ndarray
    errors: np.ndarray
    a: np.ndarray
    b: np.ndarray
    k: np.ndarray
    city: np.ndarray
    ids: np.ndarray
    changed: np.ndarray

    @property
    def n_cells(self):
        return 2 * len(np.unique(self.city))

    @property
    def m(self):
        return len(self.ids)

    @property
    def cells(self):
        return np.r_[2 * self.city, 2 * self.city + 1]

    def take(self, take):
        endpoint = np.r_[take, take + self.m]
        arrays = (self.x, self.h, self.f, self.mean_w, self.w, self.errors, self.a, self.b, self.k)
        return Panel(self.seed, *(x[endpoint] for x in arrays),
                     self.city[take], self.ids[take], self.changed[take])


def paired_panel(seed, contexts, count, cfg, runtime, dgp, step, changed_share, noise=None):
    runtime.test_per_city = count
    p = matched_panel(seed, contexts, runtime, dgp, TRUTH, step, changed_share)
    x = np.vstack((p["x_before"], p["x_after"]))
    h = (x @ dgp.mix.T)[:, :3]
    stable = ~p["assigned_changed"]
    h[len(stable):][stable] = h[:len(stable)][stable]
    f = dgp.score(h, TRUTH)
    c = np.repeat(np.arange(len(contexts)), count)
    time = np.repeat([0, 1], len(c))
    a, b, k = cell_parameters(contexts, np.tile(c, 2), time, cfg)
    means = measurement(f, a, b, k, "nonlinear")
    noise = cfg["noise_ratio"] if noise is None else noise
    errors = np.concatenate([np.random.default_rng(seed_for(seed, "error", t)).normal(size=len(c))
                             for t in (0, 1)]) * noise * b
    ids = (np.uint64(seed) << np.uint64(32)) + np.arange(len(c), dtype=np.uint64)
    np.testing.assert_allclose(f[len(c):] - f[:len(c)], p["delta_f"], atol=2e-13)
    return Panel(seed, x, h, f, means, means + errors, errors, a, b, k, c, ids,
                 p["assigned_changed"])


def train_validation(panel, seed, val_fraction):
    rng = np.random.default_rng(seed)
    fit, val = [], []
    for city in np.unique(panel.city):
        indices = rng.permutation(np.flatnonzero(panel.city == city))
        n_val = round(len(indices) * val_fraction)
        fit.extend(np.sort(indices[n_val:]))
        val.extend(np.sort(indices[:n_val]))
    fit, val = panel.take(np.array(fit)), panel.take(np.array(val))
    assert not np.intersect1d(fit.ids, val.ids).size
    assert fit.m + val.m == panel.m
    return fit, val


def aux_views(panel, contexts, seed, cfg, runtime, dgp):
    source = np.repeat(np.arange(len(panel.x)), cfg["views"])
    cities = np.tile(panel.city, 2)[source]
    dates = np.repeat([0, 1], panel.m)[source]
    aux = make_inputs(panel.h[source], cities, 1 - dates, contexts,
                      np.random.default_rng(seed), runtime, dgp)
    np.testing.assert_allclose((aux @ dgp.mix.T)[:, :3], panel.h[source], atol=2e-13)
    return aux


def replication(cfg, rep, noise=None):
    runtime, dgp = structural_dgp(cfg)
    seed = seed_for(cfg["seed"], "rep", rep)
    keys = ("contexts", "paired", "split", "fit_aux", "val_aux", "cutoff_contexts",
            "cutoff", "seen", "unseen_contexts", "unseen", "seen_regimes", "unseen_regimes")
    seeds = {key: seed_for(seed, key) for key in keys}
    assert len(set(seeds.values())) == len(seeds)
    contexts = draw_contexts(seeds["contexts"], cfg["cities"])
    paired = paired_panel(seeds["paired"], contexts, cfg["tracts_per_city"], cfg, runtime,
                          dgp, cfg["training_step"], cfg["changed_share"], noise)
    fit, val = train_validation(paired, seeds["split"], cfg["validation_fraction"])
    fit_aux = aux_views(fit, contexts, seeds["fit_aux"], cfg, runtime, dgp)
    val_aux = aux_views(val, contexts, seeds["val_aux"], cfg, runtime, dgp)
    return runtime, dgp, seeds, contexts, fit, val, fit_aux, val_aux


def test_panels(cfg, runtime, dgp, seeds, contexts, noise=None):
    count = cfg["test_tracts_per_city"]
    n_unseen = max(4, len(contexts) // 2)
    unseen = draw_contexts(seeds["unseen_contexts"], n_unseen)
    cutoff_cx = draw_contexts(seeds["cutoff_contexts"], len(contexts))
    cutoff = paired_panel(seeds["cutoff"], cutoff_cx, count, cfg, runtime, dgp, 0, 0, noise)
    result = {}
    for where, cx in (("seen", contexts), ("unseen", unseen)):
        support = paired_panel(seeds[where], cx, count, cfg, runtime, dgp,
                               cfg["training_step"], cfg["changed_share"], noise)
        fixture = paired_panel(seeds[where + "_regimes"], cx, count, cfg, runtime, dgp, 0, 0, noise)
        result[where] = (support, fixture)
        assert not np.intersect1d(support.ids, cutoff.ids).size
    return cutoff, result


def counterfactual(fixture, step, dgp):
    """Same starting tract and acquisition views at every signed dose."""
    h = fixture.h[:fixture.m]
    g = structural_gradient(h, TRUTH, dgp)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    h1 = np.clip(h + np.asarray(step).reshape(-1, 1) * g, dgp.hlow, dgp.hhigh)
    x1 = fixture.x[fixture.m:] + (h1 - h) @ dgp.mix[:3]
    return x1, dgp.score(h1, TRUTH)


def population_check(cfg, dgp):
    rng = np.random.default_rng(cfg["population_seed"])
    n = cfg["population_pairs"]
    contexts = rng.normal(size=(n, 2))
    date = rng.integers(2, size=n)
    k = cfg["kappa_base"] + cfg["kappa_amplitude"] * np.tanh(
        contexts[:, 0] + cfg["kappa_time_shift"] * date)
    f1, f2 = (dgp.score(dgp.draw_h(rng, contexts[:, 0]), TRUTH) for _ in (0, 1))
    q1, q2 = f1 + k * np.tanh(f1), f2 + k * np.tanh(f2)
    gap = np.abs(q2 - q1)
    def risk(nu):
        return float(np.mean(ndtr(-gap / (np.sqrt(2) * nu))))
    nu = cfg["noise_ratio"]
    lo, hi = 1e-6, 10.
    for _ in range(55):
        mid = (lo + hi) / 2
        if risk(mid) < 0.2:
            lo = mid
        else:
            hi = mid
    normal = rng.normal(size=(2, n))
    simulated = np.mean(np.sign(q2 - q1) != np.sign(q2 - q1 + nu * (normal[1] - normal[0])))
    conditional = ndtr(-gap / (np.sqrt(2) * nu))
    return dict(bayes_pair_error=risk(nu), simulated_pair_error=float(simulated),
                bayes_numerical_mcse=float(conditional.std() / np.sqrt(n)),
                population_pair_dispersion=float(np.mean((f2 - f1)**2) / 2),
                noise_to_signal_sd=float(nu / np.sqrt(np.mean(gap**2) / 2)),
                nu_for_bayes_20=(lo + hi) / 2, paired_noise_mean=float((normal[1] - normal[0]).mean()),
                paired_noise_median=float(np.median(normal[1] - normal[0])))
