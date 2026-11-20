"""Matched linear-index panels with separate fit, validation, and test randomness."""
from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np


TRUE_BETA = np.array([1., .5, 0.]) / np.sqrt(1.25)
CASES = ("A", "B", "C")


def seed_for(seed, *tags):
    payload = json.dumps([int(seed), *tags], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


@dataclass
class Cities:
    a: np.ndarray
    b: np.ndarray
    k: np.ndarray
    sigma: np.ndarray


@dataclass
class Panel:
    x: np.ndarray
    change: np.ndarray
    epsilon: np.ndarray
    cities: Cities
    tract_id: np.ndarray

    def take(self, indices):
        ci = np.arange(self.x.shape[0])[:, None]
        return Panel(self.x[ci, indices], self.change[ci, indices], self.epsilon[ci, indices],
                     self.cities, self.tract_id[ci, indices])

    def dose(self, amount):
        x = self.x.copy()
        x[:, :, 1, 0] = x[:, :, 0, 0] + amount * np.sqrt(1.25)
        return replace(self, x=x, change=np.full(self.change.shape, amount))

    def outcomes(self, case):
        r = self.x @ TRUE_BETA
        a, b, k, sigma = (getattr(self.cities, key)[:, None, None] for key in ("a", "b", "k", "sigma"))
        if case == "A":
            signal = r
        elif case == "B":
            signal = b * r
        elif case == "C":
            signal = b * (r + k * r**3) / np.sqrt(1 + 6 * k + 15 * k**2)
        else:
            raise ValueError(case)
        mean = a + signal
        return mean, mean + sigma * self.epsilon


def draw_cities(n, seed):
    u = np.random.default_rng(seed).random((4, n))
    return Cities(2 * u[0] - 1, .5 + u[1], .05 + .15 * u[2], .2 + .4 * u[3])


def make_panel(cities, n, seed, stable=False):
    nc = len(cities.a)
    rng = np.random.default_rng(seed_for(seed, "images"))
    h = rng.normal(size=(nc, n, 2))
    acq = rng.normal(size=(nc, n, 2))
    base_change = np.zeros(n)
    if not stable:
        if n % 8:
            raise ValueError("The balanced change design requires multiples of eight")
        base_change[n // 2:] = np.repeat([-.4, -.15, .15, .4], n // 8)
    d_rng = np.random.default_rng(seed_for(seed, "changes"))
    change = np.stack([d_rng.permutation(base_change) for _ in range(nc)])
    x = np.empty((nc, n, 2, 3))
    x[..., :2] = h[:, :, None, :]
    x[:, :, 1, 0] += change * np.sqrt(1.25)
    x[..., 2] = acq
    errors = np.random.default_rng(seed_for(seed, "errors")).normal(size=(nc, n, 2))
    ids = np.arange(nc * n, dtype=np.int64).reshape(nc, n)
    np.testing.assert_allclose(np.diff(x @ TRUE_BETA, axis=2)[..., 0], change, atol=1e-14)
    return Panel(x, change, errors, cities, ids)


def training(cfg, rep):
    root = seed_for(cfg["seed"], "simple", rep)
    keys = ("cities", "paired", "split", "fit_links", "validation_links",
            "cutoff_cities", "cutoff", "seen", "unseen_cities", "unseen", "seen_dose", "unseen_dose")
    seeds = {name: seed_for(root, name) for name in keys}
    assert len(set(seeds.values())) == len(seeds)
    nc = cfg["cities"]
    cities = draw_cities(nc, seeds["cities"])
    panel = make_panel(cities, cfg["paired_tracts"], seeds["paired"])
    rng = np.random.default_rng(seeds["split"])
    order = np.stack([rng.permutation(cfg["paired_tracts"]) for _ in range(nc)])
    fit = panel.take(order[:, :cfg["fit_tracts"]])
    valid = panel.take(order[:, cfg["fit_tracts"]:])
    assert np.intersect1d(fit.tract_id, valid.tract_id).size == 0
    return seeds, fit, valid


def external_views(panel, seed, views):
    """The same H and city in both views; A alone is independently redrawn."""
    rng = np.random.default_rng(seed)
    draws = rng.normal(size=(*panel.x.shape[:-1], views))
    return draws - panel.x[..., 2, None]


def test_panels(cfg, seeds, seen_cities):
    unseen = draw_cities(cfg["cities"], seeds["unseen_cities"])
    cutoff_cities = draw_cities(cfg["cities"], seeds["cutoff_cities"])
    n = cfg["test_tracts"]
    cutoff = make_panel(cutoff_cities, n, seeds["cutoff"], stable=True)
    cases = {where: (make_panel(cities, n, seeds[where]),
                     make_panel(cities, n, seeds[f"{where}_dose"], stable=True))
             for where, cities in (("seen", seen_cities), ("unseen", unseen))}
    return cutoff, cases
