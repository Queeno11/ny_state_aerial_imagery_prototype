"""Synthetic-data unit tests for minimal_e1 (seconds; no Monte Carlo).

From src/ordinal_mc_suite:  python -m unittest minimal_e1.tests.test_minimal_e1 -v
"""
import unittest

import numpy as np

from minimal_e1.config import make_config, arms_for
from minimal_e1 import dgp
from minimal_e1.heads import (fit_head, criterion, profile, v_star, weight_matrix, null_value)
from minimal_e1.losses import PairData, newton_ranknet
from minimal_e1 import estimators as est
from minimal_e1.evaluate import evaluate
from minimal_e1.experiment import run_dataset

CFG = make_config(reference_size=40000, test_n=200)


def fd_grad(fun, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = h
        g[i] = (fun(x + e) - fun(x - e)) / (2 * h)
    return g


def panel(rho=0.8, C=6, n=30, rep=1):
    return dgp.source_panel(CFG, rho, C, n, rep)


class TestDGP(unittest.TestCase):
    def test_truth_in_basis_and_normalised(self):
        ref = dgp.reference(CFG)
        self.assertAlmostEqual(np.linalg.norm(ref.theta_star), 1.0, places=12)
        p = panel()
        np.testing.assert_allclose(p.z @ ref.theta_star, p.f, atol=1e-10)

    def test_residual_moments(self):
        eta = dgp.residuals(dgp.rng(1, 2), 400000, 0.8)
        self.assertAlmostEqual(eta.var(axis=0).mean(), 1.0, delta=0.01)
        self.assertAlmostEqual(np.corrcoef(eta.T)[0, 1], 0.8, delta=0.01)

    def test_common_random_numbers_and_prefix(self):
        a, b = panel(0.0, 6, 30), panel(0.8, 8, 40)
        np.testing.assert_array_equal(a.z, b.z[:6, :30])
        np.testing.assert_array_equal(a.stable, b.stable[:30])

    def test_maps_monotone_inverse_and_mean(self):
        kappa = CFG['kappa']
        r = np.linspace(-6, 6, 2001)
        for lam in (-0.95, -0.5, 0.0, 0.6, 0.95):
            self.assertTrue(np.all(np.diff(dgp.sinh_map(r, lam, kappa)) > 0))
            y = dgp.sinh_map(r, lam, kappa)
            np.testing.assert_allclose(dgp.sinh_inverse(y, lam, kappa), r, atol=1e-10)
        params = dict(alpha=np.array([[0.3, -0.2]]), beta=np.array([[1.4, 0.7]]), lam=np.array([0.7]))
        s = np.array([-1.0, 0.0, 1.5])[None, :, None] * np.ones((1, 3, 2))
        eta = dgp.rng(3, 4).normal(size=(400000, 1, 1))
        many = dict(alpha=np.repeat(params['alpha'], len(eta), 0),
                    beta=np.repeat(params['beta'], len(eta), 0),
                    lam=np.repeat(params['lam'], len(eta)))
        for fam in 'abc':
            big = dgp.psi((s + eta).reshape(-1, 3, 2), fam, many, kappa)
            mc = big.mean(0)[None]
            np.testing.assert_allclose(dgp.conditional_mean(s, fam, params, kappa), mc, atol=0.01)

    def test_links_identify_nuisance(self):
        p = panel()
        basis, rank, _ = dgp.link_null_space(p.z, p.stable)
        self.assertEqual(rank, CFG['p_a'])
        np.testing.assert_allclose(basis.T @ basis, np.eye(CFG['p_h']), atol=1e-12)
        theta_star = dgp.reference(CFG).theta_star
        self.assertAlmostEqual(np.linalg.norm(basis @ (basis.T @ theta_star)), 1.0, places=10)
        gaps = p.z[:, p.stable, 1] - p.z[:, p.stable, 0]
        self.assertLess(np.max(np.abs(gaps @ theta_star)), 1e-10)
        self.assertGreater(np.max(np.abs(gaps @ np.ones(6))), 0.1)

    def test_ranks_shared_across_maps(self):
        p = panel()
        ii, jj = np.triu_indices(p.z.shape[1], 1)
        signs = {m: np.sign(p.W[m][:, jj] - p.W[m][:, ii]) for m in 'abc'}
        np.testing.assert_array_equal(signs['a'], signs['b'])
        np.testing.assert_array_equal(signs['a'], signs['c'])


class TestHeads(unittest.TestCase):
    def test_q_half_is_half_endpoint_mse(self):
        p = panel()
        W = p.W['b']
        s = p.z @ dgp.reference(CFG).theta_star
        vs = v_star(W)
        head = fit_head(s, W, 'b', 0.5, CFG['kappa'])
        val, _ = criterion(s, W, head, 0.5, vs)
        self.assertAlmostEqual(val, 0.5 * np.mean((head.predict(s) - W) ** 2) / vs, places=12)

    def test_envelope_gradient(self):
        p = panel(n=20)
        theta = np.random.default_rng(0).normal(size=6); theta /= np.linalg.norm(theta)
        for fam in 'abc':
            W = p.W[fam]
            vs = v_star(W)
            fun = est.cardinal_objective(p.z, W, fam, 0.8, CFG['kappa'], vs)
            g = fun(theta)[1]
            num = fd_grad(lambda t: fun(t)[0], theta)
            np.testing.assert_allclose(g, num, rtol=1e-4, atol=1e-7)

    def test_constrained_heads_are_monotone(self):
        p = panel(n=10)
        s = p.z @ np.random.default_rng(1).normal(size=6)
        for fam in 'bc':
            head = fit_head(s, p.W[fam], fam, 0.5, CFG['kappa'])
            self.assertTrue(np.all(head.derivative(s) >= 0))

    def test_null_value(self):
        p = panel()
        W = p.W['a']
        vs = v_star(W)
        head = fit_head(np.zeros_like(W), W, 'b', 0.5, CFG['kappa'])
        self.assertAlmostEqual(null_value(W, 0.5, vs),
                               criterion(np.zeros_like(W), W, head, 0.5, vs)[0], places=10)


class TestEstimators(unittest.TestCase):
    def test_fe_equals_cardinal_a_and_sphere(self):
        p = panel()
        W = p.W['a']
        vs = v_star(W)
        fe, fe_x, _ = est.fixed_effects(p.z, W)
        card, card_x, _ = est.cardinal(p.z, W, 'a', 0.5, CFG['kappa'], vs, [])
        np.testing.assert_allclose(fe, card, atol=1e-12)
        self.assertAlmostEqual(fe_x['b'], card_x['b'], places=12)
        fun = est._sphere_wrap(est.cardinal_objective(p.z, W, 'a', 0.5, CFG['kappa'], vs))
        res = est._lbfgs(fun, np.ones(6) / np.sqrt(6), 1e-11, 5000)
        np.testing.assert_allclose(est.unit(res.x), fe, atol=1e-5)

    def test_qstar_is_gls(self):
        p = panel(rho=0.8)
        rho = 0.6
        q = (1 + rho) / 2
        d1, _ = est.weighted_within(p.z, p.W['a'], weight_matrix(q))
        d2, _ = est.weighted_within(p.z, p.W['a'], np.array([[1, -rho], [-rho, 1]]))
        np.testing.assert_allclose(d1, d2, rtol=1e-10)

    def test_fgls_at_rho0_is_fe(self):
        p = panel(rho=0.0)
        th, _, _ = est.fgls(p.z, p.W['a'], rho_bounds=(0.0, 1e-12))
        np.testing.assert_allclose(th, est.fixed_effects(p.z, p.W['a'])[0], atol=1e-8)

    def test_ranknet_derivatives_and_newton(self):
        p = panel(n=12)
        pairs = PairData.build(p.z, p.W['a'])
        self.assertAlmostEqual(pairs.value(np.zeros(6)), np.log(2), places=12)
        self.assertAlmostEqual(pairs.weight.sum(), 1.0, places=12)
        u = np.random.default_rng(2).normal(size=6)
        _, g, H = pairs.value_grad_hess(u)
        np.testing.assert_allclose(g, fd_grad(pairs.value, u), rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(H, np.stack([fd_grad(lambda x: pairs.value_grad(x)[1][i], u)
                                                for i in range(6)]), rtol=1e-4, atol=1e-7)
        uhat, info = newton_ranknet(pairs)
        self.assertTrue(info['success'])
        self.assertLess(info['grad_norm'], 1e-8)

    def test_joint_w0_is_cardinal_and_gradient(self):
        p = panel(n=15)
        W = p.W['b']
        vs = v_star(W)
        pairs = PairData.build(p.z, W)
        card, _, _ = est.cardinal(p.z, W, 'b', 0.5, CFG['kappa'], vs,
                                  [est.fixed_effects(p.z, W)[0]], gtol=1e-11)
        j0, _, _ = est.joint(p.z, W, pairs, 'b', 0.5, CFG['kappa'], vs, 0.0, [card * 3.0])
        self.assertGreater(abs(j0 @ card), 1 - 1e-8)
        c0 = null_value(W, 0.5, vs)
        card_fun = est.cardinal_objective(p.z, W, 'b', 0.5, CFG['kappa'], vs)

        def J(u):
            return 0.5 * card_fun(u / np.linalg.norm(u))[0] / c0 + 0.5 * pairs.value(u) / np.log(2)
        u = np.random.default_rng(3).normal(size=6)
        theta = u / np.linalg.norm(u)
        g = 0.5 / c0 * est._tangent(theta, card_fun(theta)[1]) / np.linalg.norm(u) \
            + 0.5 / np.log(2) * pairs.value_grad(u)[1]
        np.testing.assert_allclose(g, fd_grad(J, u), rtol=1e-4, atol=1e-7)

    def test_ml_gradients(self):
        p = panel(n=12)
        rng = np.random.default_rng(4)
        for fam in 'bc':
            model = est.MLModel(p.z, p.W[fam], fam, CFG['kappa'])
            x = model.start(dgp.reference(CFG).theta_star) + 0.05 * rng.normal(size=sum(model.sizes()))
            g = model.value_grad(x)[1]
            np.testing.assert_allclose(g, fd_grad(lambda v: model.value_grad(v)[0], x, 1e-6),
                                       rtol=1e-4, atol=1e-7)

    def test_ml_recovers_direction_map_c(self):
        p = dgp.source_panel(CFG, 0.8, 16, 120, 2)
        theta_star = dgp.reference(CFG).theta_star
        th, ex, info = est.maximum_likelihood(p.z, p.W['c'], 'c', CFG['kappa'],
                                              [est.fixed_effects(p.z, p.W['c'])[0]])
        self.assertGreater(th @ theta_star, 0.995)
        self.assertAlmostEqual(ex['rho_hat'], 0.8, delta=0.08)
        self.assertAlmostEqual(ex['sigma_hat'], 1.0, delta=0.1)

    def test_mixed_model_woodbury_and_fit(self):
        p = panel(n=8, C=3)
        W = p.W['b']
        model = est.MixedModel(p.z, W, 'unstructured')
        rng = np.random.default_rng(5)
        x = model.start(0.5, 0.2) + 0.1 * rng.normal(size=model.n_params())
        # Dense reference likelihood for the same parameters.
        se2, su2 = np.exp(2 * x[0]), np.exp(2 * x[1])
        L = model.chol(x); Om = L @ L.T
        total_logdet, Xs, Vis, ys = 0.0, [], [], []
        for c in range(3):
            zt = p.z[c].transpose(1, 0, 2).reshape(-1, 6)            # rows (t, i)
            n = p.z.shape[1]
            Z = np.zeros((2 * n, 12)); Z[:n, :6] = zt[:n]; Z[n:, 6:] = zt[n:]
            K = np.kron(np.ones((2, 2)), np.eye(n))
            V = se2 * np.eye(2 * n) + su2 * K + Z @ np.kron(np.eye(2), Om) @ Z.T
            D = np.zeros((2 * n, 2 * 3)); D[:n, 2 * c] = 1; D[n:, 2 * c + 1] = 1
            Xs.append(np.hstack([D, zt])); Vis.append(np.linalg.inv(V))
            ys.append(W[c].T.ravel()); total_logdet += np.linalg.slogdet(V)[1]
        A = sum(X.T @ Vi @ X for X, Vi in zip(Xs, Vis))
        bvec = sum(X.T @ Vi @ y for X, Vi, y in zip(Xs, Vis, ys))
        beta = np.linalg.solve(A, bvec)
        quad = sum((y - X @ beta) @ Vi @ (y - X @ beta) for X, Vi, y in zip(Xs, Vis, ys))
        dense = 0.5 * (total_logdet + quad) / (3 * p.z.shape[1])
        self.assertAlmostEqual(model.profiled(x), dense, places=9)
        big = dgp.source_panel(CFG, 0.8, 16, 40, 3)
        th, ex, info = est.mixed_model(big.z, big.W['b'])
        self.assertGreater(th @ dgp.reference(CFG).theta_star, 0.95)


class TestPipeline(unittest.TestCase):
    def test_true_arm_is_exact(self):
        target = dgp.target_environment(CFG)
        cal_z, cal_W, _ = dgp.calibration_sample(CFG, target, 0.8, 1)
        p = panel()
        ts = dgp.reference(CFG).theta_star
        for fam in 'abc':
            out = evaluate(ts, 'TRUE', fam, {}, p.z, p.W[fam], target, cal_z, cal_W[fam], CFG, ts)
            self.assertLess(out['f_mse'], 1e-20)
            self.assertLess(out['mu_oracle_mse'], 1e-20)
            self.assertGreater(out['mu_cal_mse'], 0.0)

    def test_end_to_end_dataset(self):
        target = dgp.target_environment(CFG)
        for rho in (0.0, 0.8):
            rows, _ = run_dataset(CFG, target, dict(block=0, rho=rho, cities=4, n=10, rep=1,
                                                    maps=['a', 'b', 'c']))
            expected = sum(len(arms_for(m, rho)) for m in 'abc')
            self.assertEqual(len(rows), expected)
            errors = [r for r in rows if r['status'] == 'error']
            self.assertEqual(errors, [])
            self.assertTrue(all(r['ranks_shared'] for r in rows))
            byarm = {(r['map'], r['arm']): r for r in rows}
            for fam in 'abc':
                self.assertAlmostEqual(byarm[(fam, 'ORD')]['f_mse'], byarm[('a', 'ORD')]['f_mse'])
            self.assertAlmostEqual(byarm[('a', 'FE')]['f_mse'], byarm[('a', 'L2')]['f_mse'], places=12)


if __name__ == '__main__':
    unittest.main()
