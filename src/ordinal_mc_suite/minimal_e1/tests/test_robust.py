"""Synthetic-data unit tests for round 2 (robust scorers and noise scenarios).

From src/ordinal_mc_suite:  python -m unittest minimal_e1.tests.test_robust -v
"""
import unittest

import numpy as np

from minimal_e1.config import make_config
from minimal_e1 import dgp
from minimal_e1.losses import PairData, newton_ranknet
from minimal_e1 import robust_dgp as rd
from minimal_e1 import robust_estimators as re_
from minimal_e1.estimators import fixed_effects, unit
from minimal_e1.robust_experiment import run_dataset, arms
from minimal_e1.run_robust import _moments

CFG = make_config(reference_size=40000, test_n=200)


def fd_grad(fun, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = h
        g[i] = (fun(x + e) - fun(x - e)) / (2 * h)
    return g


class TestNoise(unittest.TestCase):
    def test_noise_laws(self):
        g = dgp.rng(1, 1)
        mix = rd.skewed_mixture(g, 400000)
        self.assertAlmostEqual(mix.mean(), 0, delta=0.01)
        self.assertAlmostEqual(mix.var(), 1, delta=0.01)
        self.assertLess(np.median(mix), -0.1)                      # skewed
        ln = rd.lognormal_noise(g, 400000)
        self.assertAlmostEqual(np.median(ln), 0, delta=0.01)
        self.assertAlmostEqual(ln.var(), 1, delta=0.03)

    def test_scenarios_build_and_ranks_preserved(self):
        base = rd.noisy_panel(CFG, 'A', 6, 20, 1, 2)
        for sc in rd.SCENARIOS:
            p = rd.noisy_panel(CFG, sc, 6, 20, 1, 2)
            self.assertEqual(p.W.shape, (6, 20, 2))
            self.assertTrue(np.all(np.isfinite(p.W)))
            np.testing.assert_array_equal(p.z, base.z)               # same covariates
        f = rd.noisy_panel(CFG, 'F', 32, 40, 1, 2)
        self.assertAlmostEqual(f.contaminated, 0.05, delta=0.02)

    def test_scenario_e_noise_scale(self):
        p = rd.noisy_panel(CFG, 'E', 32, 200, 1, 2)
        resid = p.W - p.f - p.W.mean(axis=1, keepdims=True) + p.f.mean(axis=1, keepdims=True)
        self.assertAlmostEqual(resid.var(), 1.0, delta=0.35)

    def test_evaluation_frame(self):
        e1, e2, q, a1 = rd.evaluation_frame(CFG)
        p = rd.noisy_panel(CFG, 'A', 6, 20, 1, 2)
        basis, rank, _ = dgp.link_null_space(p.z, p.stable)
        # The link null space is the H-plane, identical in every dataset.
        np.testing.assert_allclose(basis @ basis.T, q @ q.T, atol=1e-10)
        self.assertAlmostEqual(e1 @ e2, 0, places=12)
        self.assertLess(np.linalg.norm(q.T @ a1), 1e-12)


class TestScorers(unittest.TestCase):
    def setUp(self):
        p = rd.noisy_panel(CFG, 'B', 8, 25, 1, 2)
        self.p = p
        self.pairs = PairData.build(p.z, p.W)

    def test_cell_ids(self):
        counts = np.bincount(self.pairs.cell, minlength=self.pairs.n_cells)
        self.assertEqual(len(counts), 16)
        self.assertTrue(np.all(counts == 25 * 24 // 2))

    def test_ranknet_cell_reduces_to_global_with_strong_penalty(self):
        th_g = unit(newton_ranknet(self.pairs)[0])
        th_c, ex, info = re_.ranknet_cell(self.pairs, n_per_cell=25, prior_sd=1e-4)
        self.assertGreater(th_c @ th_g, 1 - 1e-6)
        th_free, ex, info = re_.ranknet_cell(self.pairs, n_per_cell=25, prior_sd=0.5)
        self.assertTrue(info['success'])
        self.assertGreater(ex['tau_cell_sd_log'], 0.05)          # scales differ by city in B

    def test_ranknet_cell_gradient(self):
        K = self.pairs.n_cells
        pen = np.full(K, 1.0 / K / (2 * 0.25 * 25))
        x = np.random.default_rng(1).normal(size=6 + K) * 0.3
        g = re_.cell_value_grad(self.pairs, x, pen)[1]
        num = fd_grad(lambda y: re_.cell_value_grad(self.pairs, y, pen)[0], x)
        np.testing.assert_allclose(g, num, rtol=1e-5, atol=1e-9)

    def test_surrogate_gradient(self):
        theta = unit(np.random.default_rng(0).normal(size=6))
        _, g = re_.surrogate_value_grad(self.pairs, theta, 0.3)
        num = fd_grad(lambda t: re_.surrogate_value_grad(self.pairs, t, 0.3)[0], theta)
        np.testing.assert_allclose(g, num, rtol=1e-5, atol=1e-9)

    def test_mrc_sweep_matches_brute_force(self):
        basis, _, _ = dgp.link_null_space(self.p.z, self.p.stable)
        p2 = self.pairs.project(basis)
        theta, _, info = re_.mrc_exact_2d(p2)
        grid = np.linspace(-np.pi, np.pi, 200001)
        vals = np.array([p2.weight @ (p2.v @ np.array([np.cos(a), np.sin(a)]) > 0)
                         for a in grid[::50]])
        self.assertAlmostEqual(info['objective'], p2.weight @ (p2.v @ theta > 0), places=12)
        self.assertGreaterEqual(info['objective'] + 1e-12, vals.max())

    def test_surrogate_2d_global(self):
        basis, _, _ = dgp.link_null_space(self.p.z, self.p.stable)
        p2 = self.pairs.project(basis)
        out = re_.surrogate_2d(p2, (0.3,))
        theta, info = out[0.3]
        grid = np.linspace(-np.pi, np.pi, 3601)
        vals = re_._angle_values(p2.v, p2.weight, grid, 0.3)
        self.assertLessEqual(info['objective'], vals.min() + 1e-12)

    def test_normal_scores_invariant_to_monotone_map(self):
        th1 = re_.normal_scores_fe(self.p.z, self.p.W)[0]
        th2 = re_.normal_scores_fe(self.p.z, np.exp(self.p.W / 3) + 7)[0]
        np.testing.assert_allclose(th1, th2, atol=1e-12)


class TestPipeline(unittest.TestCase):
    def test_end_to_end(self):
        moments = _moments(CFG)
        for sc in ('A', 'E'):
            for s_only in (False, True):
                rows, _ = run_dataset(CFG, moments, dict(scenario=sc, block=2, cities=6, n=12,
                                                         rep=1, s_only=s_only))
                self.assertEqual(len(rows), len(arms(s_only)))
                self.assertEqual([r for r in rows if r['status'] == 'error'], [])
                for r in rows:
                    if r['arm'].endswith('_S'):
                        self.assertLess(r['off_plane'], 1e-9)
                        self.assertAlmostEqual(r['cos_true'], np.cos(r['signed_angle']), places=9)


if __name__ == '__main__':
    unittest.main()
