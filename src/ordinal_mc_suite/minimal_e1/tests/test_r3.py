"""Synthetic-data unit tests for round 3 (SURN/SURH scorers, A8, step-2 heads).

From src/ordinal_mc_suite:  python -m unittest minimal_e1.tests.test_r3 -v
"""
import itertools
import unittest

import numpy as np
from scipy.stats import kendalltau

from minimal_e1 import dgp
from minimal_e1 import r3_estimators as r3e
from minimal_e1 import r3_heads as r3h
from minimal_e1 import robust_dgp as rd
from minimal_e1.config import make_config
from minimal_e1.estimators import unit
from minimal_e1.losses import PairData
from minimal_e1.r3_experiment import run_dataset, arms
from minimal_e1.robust_estimators import ranknet_cell
from minimal_e1.run_robust import _moments
from minimal_e1.summarize_r3 import paired_ratio
import pandas as pd

CFG = make_config(reference_size=40000, test_n=200)


def fd_grad(fun, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = h
        g[i] = (fun(x + e) - fun(x - e)) / (2 * h)
    return g


class TestSurObjective(unittest.TestCase):
    def setUp(self):
        self.p = rd.noisy_panel(CFG, 'B', 8, 25, 1, 4)
        self.pairs = PairData.build(self.p.z, self.p.W)
        self.theta = unit(np.random.default_rng(0).normal(size=6))

    def test_abrevaya_equivalence(self):
        """SUR = const - sum w y K(Delta'theta/tau) with the logistic kernel K (Eq. 9)."""
        z, W = self.p.z, self.p.W
        C, n, _, _ = z.shape
        ii, jj = np.triu_indices(n, 1)
        y = np.sign(W[:, jj, :] - W[:, ii, :])                 # (C, m, 2)
        delta = z[:, jj] - z[:, ii]                            # (C, m, 2, p)
        w = 1.0 / (2 * C * y.shape[1])                         # no ties: equal weights
        K = lambda t: 1.0 / (1.0 + np.exp(-t))
        for tau in (0.7, 0.1):
            smoothed_obj1 = np.sum(w * y * K(delta @ self.theta / tau))
            const = np.sum(w * (y > 0))
            sur = r3e.SurObjective(self.pairs, tau).value(self.theta)
            self.assertAlmostEqual(sur, const - smoothed_obj1, places=12)

    def test_per_cell_tau_gradient(self):
        tau = np.random.default_rng(1).uniform(0.1, 0.8, size=self.pairs.n_cells)
        obj = r3e.SurObjective(self.pairs, tau)
        _, g = obj.value_grad(self.theta)
        num = fd_grad(lambda t: obj.value(t), self.theta)
        np.testing.assert_allclose(g, num, rtol=1e-5, atol=1e-9)

    def test_per_cell_tau_matches_scalar_when_equal(self):
        a = r3e.SurObjective(self.pairs, 0.3).value_grad(self.theta)
        b = r3e.SurObjective(self.pairs, np.full(self.pairs.n_cells, 0.3)).value_grad(self.theta)
        self.assertAlmostEqual(a[0], b[0], places=14)
        np.testing.assert_allclose(a[1], b[1], atol=1e-14)

    def test_surh_gradient_and_formula(self):
        tau = np.random.default_rng(2).uniform(0.1, 0.8, size=self.pairs.n_cells)
        obj = r3e.SurObjective(self.pairs, tau, eps=1 / 25, tau0=0.4)
        val, g = obj.value_grad(self.theta)
        num = fd_grad(lambda t: obj.value(t), self.theta)
        np.testing.assert_allclose(g, num, rtol=1e-5, atol=1e-9)
        expected = (r3e.SurObjective(self.pairs, tau).value(self.theta)
                    + (1 / 25) * self.pairs.value(self.theta / 0.4))
        self.assertAlmostEqual(val, expected, places=12)
        with self.assertRaises(ValueError):
            r3e.SurObjective(self.pairs, 0.3, eps=0.1)

    def test_angle_values_match_pointwise(self):
        basis, _, _ = dgp.link_null_space(self.p.z, self.p.stable)
        p2 = self.pairs.project(basis)
        tau = np.random.default_rng(3).uniform(0.1, 0.8, size=p2.n_cells)
        obj = r3e.SurObjective(p2, tau, eps=0.05, tau0=0.5)
        angles = np.array([-2.0, -0.3, 1.1])
        direct = [obj.value(np.array([np.cos(a), np.sin(a)])) for a in angles]
        np.testing.assert_allclose(obj.angle_values(angles), direct, atol=1e-13)

    def test_subsample_is_unbiased_scale(self):
        obj = r3e.SurObjective(self.pairs, 0.3)
        sub = obj.subsample(2000, seed=1)
        self.assertEqual(len(sub.w), 2000)
        self.assertAlmostEqual(sub.w.sum(), obj.w.sum(), places=10)
        self.assertAlmostEqual(sub.value(self.theta), obj.value(self.theta), delta=0.03)


class TestSchedule(unittest.TestCase):
    def test_bandwidth_values(self):
        np.testing.assert_allclose(r3e.bandwidth([10, 40, 250]), [0.2512, 0.1093, 0.0364], atol=5e-5)

    def test_anneal_schedule(self):
        s = lambda t: [np.unique(v).tolist() for v in r3e.anneal_schedule(t)]
        np.testing.assert_allclose(np.concatenate(s(0.036)), [1.0, 0.3, 0.036])
        np.testing.assert_allclose(np.concatenate(s(0.25)), [1.0, 0.3, 0.25])
        np.testing.assert_allclose(np.concatenate(s(0.4)), [1.0, 0.4])        # 0.3 level skipped
        np.testing.assert_allclose(np.concatenate(s(1.2)), [1.2])
        mixed = r3e.anneal_schedule(np.array([0.1, 0.5]))
        np.testing.assert_allclose(mixed[0], [1.0, 1.0])
        np.testing.assert_allclose(mixed[1], [0.3, 0.5])
        np.testing.assert_allclose(mixed[-1], [0.1, 0.5])

    def test_tau_per_cell(self):
        p = PairData.build(rd.noisy_panel(CFG, 'A', 3, 10, 1, 4).z, rd.noisy_panel(CFG, 'A', 3, 10, 1, 4).W)
        np.testing.assert_allclose(r3e.tau_per_cell(p, 10), np.full(6, 10 ** -0.6))
        np.testing.assert_allclose(r3e.tau_per_cell(p, np.arange(1, 7) * 10.0), (np.arange(1, 7) * 10.0) ** -0.6)


class TestOptimisers(unittest.TestCase):
    def setUp(self):
        self.p = rd.noisy_panel(CFG, 'A', 8, 25, 1, 4)
        self.pairs = PairData.build(self.p.z, self.p.W)
        self.basis, _, _ = dgp.link_null_space(self.p.z, self.p.stable)

    def test_2d_global_optimum_below_fine_grid(self):
        p2 = self.pairs.project(self.basis)
        for eps, tau0 in ((0.0, None), (1 / 25, 0.5)):
            theta, ex, info = r3e.sur_2d(p2, r3e.tau_per_cell(p2, 25), eps, tau0)
            obj = r3e.SurObjective(p2, r3e.tau_per_cell(p2, 25), eps, tau0)
            grid = np.linspace(-np.pi, np.pi, 3601)
            self.assertLessEqual(info['objective'], obj.angle_values(grid).min() + 1e-9)
            self.assertAlmostEqual(info['objective'], obj.value(theta), places=9)
            self.assertAlmostEqual(np.linalg.norm(theta), 1.0, places=12)

    def test_2d_with_subsample_close(self):
        p2 = self.pairs.project(self.basis)
        tau = r3e.tau_per_cell(p2, 25)
        full = r3e.sur_2d(p2, tau)[0]
        sub = r3e.sur_2d(p2, tau, grid_pairs=1500)[0]
        self.assertGreater(full @ sub, 0.99)

    def test_sphere_recovers_direction_and_beats_start(self):
        rng = np.random.default_rng(5)
        z = rng.normal(size=(6, 80, 2, 4))
        th = unit(np.array([1.0, 0.5, -0.5, 0.2]))
        W = z @ th + 0.5 * rng.normal(size=(6, 80, 2))
        pairs = PairData.build(z, W)
        start = unit(rng.normal(size=4))
        for name, fun in (('SURN', lambda: r3e.surn(pairs, 80, start, seed=1)),
                          ('SURH', lambda: r3e.surh(pairs, 80, start, 0.5, seed=1))):
            theta, ex, info = fun()
            self.assertGreater(theta @ th, 0.97, name)
            obj_start = r3e.SurObjective(pairs, r3e.tau_per_cell(pairs, 80),
                                         0 if name == 'SURN' else 1 / 80, None if name == 'SURN' else 0.5)
            self.assertLessEqual(info['objective'], obj_start.value(start) + 1e-12)
            self.assertEqual(info['n_starts'], 4)
            self.assertAlmostEqual(info['objective'], obj_start.value(theta), places=9)

    def test_multistart_is_deterministic(self):
        a = r3e.surn(self.pairs, 25, unit(np.ones(6)), seed=7)[0]
        b = r3e.surn(self.pairs, 25, unit(np.ones(6)), seed=7)[0]
        np.testing.assert_array_equal(a, b)

    def test_surh_dispatch_eps_is_one_over_n(self):
        p2 = self.pairs.project(self.basis)
        _, _, info = r3e.surh(p2, 25, None, 0.5)
        obj = r3e.SurObjective(p2, r3e.tau_per_cell(p2, 25), 1 / 25, 0.5)
        theta = r3e.surh(p2, 25, None, 0.5)[0]
        self.assertAlmostEqual(info['objective'], obj.value(theta), places=9)


class TestHeads(unittest.TestCase):
    def brute_lad(self, s, w, family, kappa):
        B = r3h.basis_columns(s, family, kappa)
        k = B.shape[1]
        sign = [1, -1][:k]
        best = np.inf
        n = len(s)
        for active in range(k + 1):
            for cols in itertools.combinations(range(k), active):
                D = np.hstack([np.ones((n, 1)), B[:, list(cols)]])
                for rows in itertools.combinations(range(n), D.shape[1]):
                    try:
                        beta = np.linalg.solve(D[list(rows)], w[list(rows)])
                    except np.linalg.LinAlgError:
                        continue
                    if any(beta[1 + i] * sign[c] < -1e-12 for i, c in enumerate(cols)):
                        continue
                    best = min(best, np.abs(w - D @ beta).sum())
        return best

    def test_l1_matches_brute_force(self):
        rng = np.random.default_rng(0)
        for family in ('b', 'c'):
            for trial in range(6):
                s = rng.normal(size=6)
                w = (-1.0 if trial % 2 else 1.0) * (np.exp(0.5 * s) if family == 'c' else s) \
                    + rng.standard_t(3, size=6)
                a, c, obj = r3h.lad_fit(s, w, family, 0.5)
                B = r3h.basis_columns(s, family, 0.5)
                self.assertAlmostEqual(obj, np.abs(w - a - B @ c).sum(), places=8)
                self.assertAlmostEqual(obj, self.brute_lad(s, w, family, 0.5), places=7)

    def test_l1_signs_and_monotone(self):
        rng = np.random.default_rng(1)
        s = rng.normal(size=(3, 30, 2))
        W = -2.0 * s + rng.normal(size=s.shape)                  # wrong-signed slope
        for family in ('b', 'c'):
            h = r3h.fit_l1_head(s, W, family, 0.5)
            if family == 'b':
                self.assertTrue(np.all(h.coef >= -1e-9))
            else:
                self.assertTrue(np.all(h.coef[..., 0] >= -1e-9))
                self.assertTrue(np.all(h.coef[..., 1] <= 1e-9))
            grid = np.sort(rng.normal(size=(3, 50, 2)), axis=1)
            self.assertTrue(np.all(np.diff(h.predict(grid), axis=1) >= -1e-9))

    def test_l1_robust_to_outlier_and_recovers_line(self):
        s = np.linspace(-1, 1, 21)[None, :, None] * np.ones((1, 21, 2))
        W = 0.5 + 2.0 * s
        W[0, 3, :] += 50.0
        h = r3h.fit_l1_head(s, W, 'b', 0.5)
        np.testing.assert_allclose(h.alpha, 0.5, atol=1e-7)
        np.testing.assert_allclose(h.coef, 2.0, atol=1e-7)
        np.testing.assert_allclose(h.predict(s)[0, :, 0][[0, 10, 20]], [-1.5, 0.5, 2.5], atol=1e-7)

    def test_l2_head_is_heads_fit_head(self):
        rng = np.random.default_rng(2)
        s = rng.normal(size=(3, 30, 2))
        W = np.exp(0.5 * s) + rng.normal(size=s.shape) * 0.1
        h = r3h.fit_l2_head(s, W, 'c', 0.5)
        self.assertEqual(h.family, 'c')
        self.assertEqual(h.predict(s).shape, s.shape)

    def test_kendall_matches_scipy(self):
        rng = np.random.default_rng(3)
        s = rng.normal(size=(1, 40, 2))
        W = s + rng.normal(size=s.shape)
        expected = np.mean([kendalltau(s[0, :, t], W[0, :, t])[0] for t in range(2)])
        self.assertAlmostEqual(r3h.mean_kendall(s, W), expected, places=14)
        self.assertAlmostEqual(r3h.mean_kendall(s[:, :, :1].repeat(2, 2), W[:, :, :1].repeat(2, 2)),
                               kendalltau(s[0, :, 0], W[0, :, 0])[0], places=14)

    def test_errors(self):
        p = np.array([[[1.0, 2.0]]]); w = np.array([[[0.0, 4.0]]])
        self.assertEqual(r3h.errors(p, w), (1.5, 2.5))

    def test_target_panels(self):
        (zc, Wc), (zt, Wt) = r3h.target_panels(CFG, 'A', 1)
        self.assertEqual(zc.shape, (8, 20, 2, 6))
        self.assertEqual(zt.shape, (8, 500, 2, 6))
        self.assertEqual(Wt.shape, (8, 500, 2))
        (zc2, Wc2), (zt2, Wt2) = r3h.target_panels(CFG, 'A', 2)
        self.assertIs(zt2, zt)                                   # fixed across reps
        self.assertFalse(np.allclose(zc, zc2))                   # fresh calibration per rep
        # Calibration shares the test cities: city-level shifts of the features agree.
        self.assertLess(np.abs(zc.mean(axis=(1, 2)) - zt.mean(axis=(1, 2))).max(), 1.5)

    def test_step2_true_beats_noise_score(self):
        cal, test = r3h.target_panels(CFG, 'A', 1)
        ref = dgp.reference(CFG)
        good = r3h.step2_metrics(ref.theta_star, cal, test, 'c', 0.5)
        bad = r3h.step2_metrics(unit(np.random.default_rng(0).normal(size=6)), cal, test, 'c', 0.5)
        self.assertGreater(good['kendall'], bad['kendall'])
        self.assertLess(good['l2_mse'], bad['l2_mse'])
        for k in ('l1_mae', 'l1_mse', 'l2_mae', 'l2_mse'):
            self.assertTrue(np.isfinite(good[k]))
        # map a family
        cal, test = r3h.target_panels(CFG, 'E', 1)
        self.assertTrue(np.isfinite(r3h.step2_metrics(ref.theta_star, cal, test, 'a', 0.5)['l1_mae']))


class TestScenarioA8(unittest.TestCase):
    def test_shares_x_and_persistence(self):
        a = rd.noisy_panel(CFG, 'A', 8, 250, 1, 4)
        a8 = rd.noisy_panel(CFG, 'A8', 8, 250, 1, 4)
        np.testing.assert_array_equal(a.z, a8.z)
        np.testing.assert_array_equal(a.f, a8.f)
        self.assertEqual(a8.map_name, 'c')
        seed = CFG['seed']
        etas = {}
        for name, panel in (('A', a), ('A8', a8)):
            e = []
            for c in range(8):
                ctx = dgp.city_context(dgp.rng(seed, dgp.STREAM['src_city'], 4, 1, c), CFG)
                params = dict(alpha=ctx['alpha'][None], beta=ctx['beta'][None], lam=np.array([ctx['lam']]))
                r = dgp.psi_inverse(panel.W[c][None], 'c', params, CFG['kappa'])[0]
                e.append(r - panel.f[c])
            etas[name] = np.concatenate(e)
        ra = np.corrcoef(etas['A'].T)[0, 1]
        r8 = np.corrcoef(etas['A8'].T)[0, 1]
        self.assertAlmostEqual(ra, 0.0, delta=0.07)
        self.assertAlmostEqual(r8, 0.8, delta=0.04)
        self.assertAlmostEqual(etas['A8'].var(), 1.0, delta=0.08)
        # same innovations: corr(eta_A, eta_A8) = sqrt(1 - rho) = 0.447 within a date
        self.assertAlmostEqual(np.corrcoef(etas['A'][:, 0], etas['A8'][:, 0])[0, 1], np.sqrt(0.2), delta=0.06)

    def test_existing_scenarios_untouched_by_city_key(self):
        for sc in rd.SCENARIOS:
            a = rd.noisy_panel(CFG, sc, 4, 20, 2, 3)
            b = rd.noisy_panel(CFG, sc, 4, 20, 2, 3, city_key=(3, 2))
            np.testing.assert_array_equal(a.W, b.W)
            np.testing.assert_array_equal(a.z, b.z)
        self.assertEqual(rd.SCENARIOS, ['A', 'B', 'C', 'D', 'E', 'F'])
        self.assertEqual(rd.SCENARIOS_R3[:2], ['A', 'A8'])

    def test_city_key_changes_only_city_level(self):
        a = rd.noisy_panel(CFG, 'B', 4, 20, 2, 3)
        b = rd.noisy_panel(CFG, 'B', 4, 20, 3, 3, city_key=(3, 2))
        self.assertFalse(np.allclose(a.z, b.z))                 # tract draws differ (rep)


class TestSummaryHelpers(unittest.TestCase):
    def test_paired_ratio(self):
        a = pd.Series([2.0, 4.0, 6.0], index=[1, 2, 3])
        b = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3])
        r, se = paired_ratio(a, b)
        self.assertAlmostEqual(r, 2.0)
        self.assertAlmostEqual(se, 0.0, places=12)
        r, se = paired_ratio(a, pd.Series([1.0, 3.0, 2.0], index=[1, 2, 3]))
        self.assertGreater(se, 0)


class TestPipeline(unittest.TestCase):
    def test_end_to_end(self):
        moments = _moments(CFG)
        for sc in ('A', 'E'):
            for s_only in (False, True):
                rows, _ = run_dataset(CFG, moments, dict(scenario=sc, block=4, cities=6, n=12, rep=1,
                                                         s_only=s_only, cal_n=20, test_n=60))
                self.assertEqual(len(rows), len(arms(s_only)) + 1)
                self.assertEqual([r for r in rows if r['status'] == 'error'], [])
                self.assertEqual([r['error'] for r in rows if r.get('error')], [])
                for r in rows:
                    for k in ('kendall', 'l1_mae', 'l2_mae'):
                        self.assertTrue(np.isfinite(r[k]), (r['arm'], k))
                    if r['arm'].endswith('_S'):
                        self.assertLess(r['off_plane'], 1e-9)
                true = [r for r in rows if r['arm'] == 'TRUE'][0]
                self.assertAlmostEqual(true['f_mse'], 0.0, places=12)


if __name__ == '__main__':
    unittest.main()
