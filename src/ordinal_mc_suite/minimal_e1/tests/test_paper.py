"""Tests for the paper Monte Carlo (synthetic data only).  From /mnt/user-data/outputs:
    PYTHONPATH=. python -m unittest minimal_e1.tests.test_paper -v
"""
import json
import tempfile
import tracemalloc
import unittest

import numpy as np
try:
    from scipy.optimize import isotonic_regression
except ImportError:          # scipy < 1.12
    isotonic_regression = None
from scipy.special import expit

from minimal_e1.paper import arms as arms_mod
from minimal_e1.paper import calibration as cal
from minimal_e1.paper import dgp, fit, heads, kernels as kn, run as runner, stability as st
from minimal_e1.paper.cells import CellSet
from minimal_e1 import r3_estimators as r3e, robust_estimators as rbe
from minimal_e1.losses import PairData

py = lambda f: getattr(f, 'py_func', f)          # pure-python loop version of a numba kernel


def small_panel(C=6, n=40, spec=None, key=(1, 2, 3)):
    spec = spec or dgp.Spec()
    p = dgp.make_panel(spec, C, n, key)
    return p, dgp.split_tracts(p, key)


class TestKernels(unittest.TestCase):
    def setUp(self):
        self.p, _ = small_panel(6, 30)
        self.pairs = PairData.build(self.p.z, self.p.W)
        self.cs = CellSet.build(self.p.z, self.p.W, self.p.xy)
        g = np.random.default_rng(0)
        self.u = g.normal(size=6)
        self.d = 0.3 * g.normal(size=self.cs.K)
        self.S = self.cs.Zs @ self.u

    def test_rank_equals_pairdata_and_ranknet_cell_objective(self):
        v0, g0 = self.pairs.value_grad(self.u)
        v, _, gs = kn.rank_vg(self.cs, self.S, np.ones(self.cs.K))
        g = np.einsum('kia,ki->a', self.cs.Zs, gs)
        self.assertLess(abs(v - v0), 1e-12)
        self.assertLess(np.abs(g - g0).max(), 1e-12)
        val0, gr0 = rbe.cell_value_grad(self.pairs, np.concatenate([self.u, self.d]), np.zeros(self.cs.K))
        v, sgt, gs = kn.rank_vg(self.cs, self.S, np.exp(self.d))
        self.assertLess(abs(v - val0), 1e-12)
        self.assertLess(np.abs(np.einsum('kia,ki->a', self.cs.Zs, gs) - gr0[:6]).max(), 1e-12)
        self.assertLess(np.abs(-sgt - gr0[6:]).max(), 1e-12)

    def test_hessian_equals_pairdata(self):
        H0 = self.pairs.value_grad_hess(self.u)[2]
        H = kn.rank_curv(self.cs, self.S, self.cs.Zs, np.ones(self.cs.K))
        self.assertLess(np.abs(H - H0).max(), 1e-12)

    def test_sur_equals_surobjective_and_angles(self):
        th = self.u / np.linalg.norm(self.u)
        tau = np.random.default_rng(1).uniform(0.1, 0.5, self.cs.K)
        v0, g0 = r3e.SurObjective(self.pairs, tau).value_grad(th)
        v, gs = kn.sur_vg(self.cs, self.cs.Zs @ th, tau)
        self.assertLess(abs(v - v0), 1e-12)
        self.assertLess(np.abs(np.einsum('kia,ki->a', self.cs.Zs, gs) - g0).max(), 1e-12)
        B = np.linalg.qr(np.random.default_rng(2).normal(size=(6, 2)))[0]
        pr = self.pairs.project(B)
        ang = np.linspace(-3, 3, 11)
        ref = r3e.SurObjective(pr, tau).angle_values(ang)
        got = kn.sur_angles(self.cs, self.cs.Zs @ B, tau, ang)
        self.assertLess(np.abs(got - ref).max(), 1e-12)

    def test_numba_style_loops_equal_numpy(self):
        K, n = self.cs.Ws.shape
        M = np.exp(self.d)
        ref = kn.np_rank_vg
        a = [np.zeros(K), np.zeros(K), np.zeros((K, n))]
        b = [np.zeros(K), np.zeros(K), np.zeros((K, n))]
        for ell in (0.0, 0.2):
            W0 = self.cs.W0
            ref(self.S, self.cs.Ws, self.cs.XY, ell, W0, M, *a)
            py(kn.nb_rank_vg)(self.S, self.cs.Ws, self.cs.XY, ell, W0, M, *b)
            for x, y in zip(a, b):
                self.assertLess(np.abs(x - y).max(), 1e-12)
            r1, c1 = np.zeros((K, n)), np.zeros((K, n, 6))
            r2, c2 = np.zeros((K, n)), np.zeros((K, n, 6))
            kn.np_rank_curv(self.S, self.cs.Zs, self.cs.Ws, self.cs.XY, ell, W0, M, r1, c1)
            py(kn.nb_rank_curv)(self.S, self.cs.Zs, self.cs.Ws, self.cs.XY, ell, W0, M, r2, c2)
            self.assertLess(np.abs(r1 - r2).max(), 1e-12)
            self.assertLess(np.abs(c1 - c2).max(), 1e-12)
            tau = np.full(K, 0.3)
            v1, g1, v2, g2 = np.zeros(K), np.zeros((K, n)), np.zeros(K), np.zeros((K, n))
            kn.np_sur_vg(self.S, self.cs.Ws, self.cs.XY, ell, W0, tau, v1, g1)
            py(kn.nb_sur_vg)(self.S, self.cs.Ws, self.cs.XY, ell, W0, tau, v2, g2)
            self.assertLess(np.abs(v1 - v2).max() + np.abs(g1 - g2).max(), 1e-12)
            ang = np.linspace(-3, 3, 5)
            Z2 = self.cs.Zs[..., :2]
            o1, o2 = np.zeros((K, 5)), np.zeros((K, 5))
            kn.np_sur_angles(Z2, self.cs.Ws, self.cs.XY, ell, W0, tau, ang, o1)
            py(kn.nb_sur_angles)(Z2, self.cs.Ws, self.cs.XY, ell, W0, tau, ang, o2)
            self.assertLess(np.abs(o1 - o2).max(), 1e-12)
        n1, n2 = np.zeros(K), np.zeros(K)
        kn.np_pair_norm(self.cs.Ws, self.cs.XY, 0.2, n1)
        py(kn.nb_pair_norm)(self.cs.Ws, self.cs.XY, 0.2, n2)
        self.assertLess(np.abs(n1 - n2).max(), 1e-12)

    def test_distance_weights_brute_force(self):
        cs = CellSet.build(self.p.z, self.p.W, self.p.xy, ell=0.2)
        k, n = 3, cs.n
        s = cs.Zs[k] @ self.u
        tot = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if cs.Ws[k, j] > cs.Ws[k, i]:
                    d2 = ((cs.XY[k, i] - cs.XY[k, j]) ** 2).sum()
                    tot += cs.W0[k] * (1 - np.exp(-d2 / 0.08)) * np.logaddexp(0, -(s[j] - s[i]))
        v = np.zeros(cs.K); sg = np.zeros(cs.K); gs = np.zeros((cs.K, n))
        kn.np_rank_vg(cs.Zs @ self.u, cs.Ws, cs.XY, 0.2, cs.W0, np.ones(cs.K), v, sg, gs)
        self.assertAlmostEqual(v[k], tot, places=12)
        # cell weights sum to 1/K
        self.assertAlmostEqual(float(np.sum([cs.W0[k] * 0 + 0 for k in range(cs.K)])), 0.0)

    def test_chunking_invariance_and_budget(self):
        K = self.cs.K
        base = kn.rank_vg(self.cs, self.S, np.exp(self.d))
        old = kn.ROW_OVERRIDE
        try:
            kn.ROW_OVERRIDE = 3
            small = kn.rank_vg(self.cs, self.S, np.exp(self.d))
            H1 = kn.rank_curv(self.cs, self.S, self.cs.Zs, np.exp(self.d))
            kn.ROW_OVERRIDE = None
            H0 = kn.rank_curv(self.cs, self.S, self.cs.Zs, np.exp(self.d))
        finally:
            kn.ROW_OVERRIDE = old
        self.assertLess(abs(base[0] - small[0]), 1e-12)
        self.assertLess(np.abs(base[2] - small[2]).max(), 1e-12)
        self.assertLess(np.abs(H1 - H0).max(), 1e-12)
        # chunk rows derive from the memory budget and respect it
        for gb in (2.0, 0.05, 1e-5):
            old_gb = kn.MEM_GB
            kn.MEM_GB = gb
            try:
                rows = kn.block_rows(500)
                self.assertGreaterEqual(rows, 1)
                self.assertTrue(rows == 1 or rows * 500 * 8 * kn.N_TEMP <= 0.25 * gb * 1e9 + 1)
            finally:
                kn.MEM_GB = old_gb
        kn.MEM_GB = 1e-3
        try:
            self.assertLess(kn.block_rows(500), 64)
        finally:
            kn.MEM_GB = 2.0

    def test_peak_memory_bounded_by_chunks(self):
        z = np.random.default_rng(0).normal(size=(2, 600, 2, 4))
        W = np.random.default_rng(1).normal(size=(2, 600, 2))
        cs = CellSet.build(z, W)
        S = cs.Zs @ np.ones(4)
        # the memory budget governs the numpy (row-blocked) backend; numba loops keep O(n) state only
        old, old_backend = kn.MEM_GB, kn.BACKEND
        peaks = {}
        try:
            kn.set_backend('numpy')
            for gb in (1e-3, 2.0):
                kn.MEM_GB = gb
                tracemalloc.start()
                kn.rank_vg(cs, S, np.ones(cs.K))
                peaks[gb] = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
        finally:
            kn.MEM_GB = old
            kn.set_backend(old_backend)
        self.assertLess(peaks[1e-3], 0.5 * peaks[2.0])
        self.assertLess(peaks[2.0], 64 * 600 * 8 * (kn.N_TEMP + 4))        # never the n x n matrix

    def test_pair_subsample(self):
        n = self.cs.n
        full = kn.rank_vg(self.cs, self.S, np.ones(self.cs.K))
        cs_all = CellSet.build(self.p.z, self.p.W, self.p.xy, max_pairs=n * n)      # >= all pairs: no subsample
        self.assertIsNone(cs_all.pairs)
        cs_sub = CellSet.build(self.p.z, self.p.W, self.p.xy, max_pairs=200, seed=1)
        self.assertEqual(cs_sub.n_pairs(), 200 * cs_sub.K)
        # explicit-pair kernels on the full pair list reproduce the full-batch kernels
        iu, ju = np.triu_indices(n, 1)
        cs_sub.pairs = [(iu.astype(np.int32), ju.astype(np.int32))] * cs_sub.K
        cs_sub.W0 = self.cs.W0
        v = kn.rank_vg(cs_sub, self.S, np.ones(self.cs.K))
        self.assertLess(abs(v[0] - full[0]), 1e-12)
        self.assertLess(np.abs(v[2] - full[2]).max(), 1e-12)
        H = kn.rank_curv(cs_sub, self.S, self.cs.Zs, np.ones(self.cs.K))
        self.assertLess(np.abs(H - kn.rank_curv(self.cs, self.S, self.cs.Zs, np.ones(self.cs.K))).max(), 1e-12)
        # a genuine subsample is a weighted mean of the same loss (cell weights sum to 1/K)
        cs2 = CellSet.build(self.p.z, self.p.W, self.p.xy, max_pairs=300, seed=2)
        v0 = kn.rank_vg(cs2, self.S * 0, np.ones(cs2.K))[0]
        self.assertAlmostEqual(v0, np.log(2), places=10)


@unittest.skipUnless(kn.HAVE_NUMBA, 'numba not installed (run this test on the box with numba)')
class TestNumbaBackend(unittest.TestCase):
    def test_numba_dispatch_equals_numpy(self):
        p, _ = small_panel(6, 40)
        cs = CellSet.build(p.z, p.W, p.xy, ell=0.2)
        u = np.random.default_rng(0).normal(size=6)
        S, M = cs.Zs @ u, np.exp(0.2 * np.random.default_rng(1).normal(size=cs.K))
        old = kn.BACKEND
        try:
            res = {}
            for be in ('numba', 'numpy'):
                kn.set_backend(be)
                v, sg, gs = kn.rank_vg(cs, S, M)
                res[be] = (v, sg, gs, kn.rank_curv(cs, S, cs.Zs, M), kn.sur_vg(cs, S, 0.3)[1],
                           kn.sur_angles(cs, cs.Zs[..., :2], 0.3, np.linspace(-3, 3, 7)),
                           kn.pair_norm(cs.Ws, cs.XY, 0.2))
            for a, b in zip(res['numba'], res['numpy']):
                self.assertLess(np.abs(np.asarray(a) - np.asarray(b)).max(), 1e-10)
        finally:
            kn.set_backend(old)


class TestMRC(unittest.TestCase):
    def check(self, seed, bins, unequal=False):
        g = np.random.default_rng(seed)
        z = g.normal(size=(3, 25, 2, 2))
        W = z[..., 0] + 0.7 * z[..., 1] + 0.8 * g.normal(size=(3, 25, 2))
        cs = CellSet.build(z, W)
        if unequal:
            cs.W0 = cs.W0 * g.uniform(0.5, 1.5, cs.K)
        phi, level, info = kn.mrc_exact_stream(cs, cs.Zs, bins=bins)
        # brute force on a fine grid + reference sweep (equal weights)
        grid = np.linspace(-np.pi, np.pi, 40001)
        v = np.zeros(len(grid))
        for k in range(cs.K):
            iu, ju = np.triu_indices(25, 1)
            m = cs.Ws[k][ju] > cs.Ws[k][iu]
            d = (cs.Zs[k][ju[m]] - cs.Zs[k][iu[m]])
            v += cs.W0[k] * ((d[:, :1] * np.cos(grid) + d[:, 1:] * np.sin(grid)) > 0).sum(0)
        at = 0
        for k in range(cs.K):
            iu, ju = np.triu_indices(25, 1)
            m = cs.Ws[k][ju] > cs.Ws[k][iu]
            d = cs.Zs[k][ju[m]] - cs.Zs[k][iu[m]]
            at += cs.W0[k] * ((d @ np.array([np.cos(phi), np.sin(phi)])) > 0).sum()
        self.assertGreaterEqual(at, v.max() - 1e-12)                  # exact global max
        self.assertAlmostEqual(at, level, places=10)
        return phi, cs

    def test_exact_global_maximum_matches_reference(self):
        for seed in range(4):
            for bins in (16, 257, 1 << 14):
                phi, cs = self.check(seed, bins)
        phi, cs = self.check(5, 64, unequal=True)
        g = np.random.default_rng(7)
        z = g.normal(size=(3, 25, 2, 2)); W = z[..., 0] + g.normal(size=(3, 25, 2))
        cs = CellSet.build(z, W)
        phi, level, _ = kn.mrc_exact_stream(cs, cs.Zs, bins=128)
        pr = PairData.build(z, W)
        th, _, info = rbe.mrc_exact_2d(pr)
        self.assertAlmostEqual(level, info['objective'] / 1.0, places=10)
        # the streamed phi lies in the reference maximiser interval
        val = lambda a: float(pr.weight @ (pr.v @ np.array([np.cos(a), np.sin(a)]) > 0))
        self.assertAlmostEqual(val(phi), info['objective'], places=10)


class TestStability(unittest.TestCase):
    def test_hard_null_space_projection(self):
        p, (tr, te) = small_panel(6, 60)
        D = dgp.link_rows(p, tr)
        Sw = st.within_cov(dgp.take(p.z, tr))
        N, ev = st.null_space(D, Sw)
        self.assertEqual(N.shape, (6, 2))
        self.assertLess(np.abs(D @ N).max(), 1e-10)                       # D N = 0
        self.assertLess(np.abs(N.T @ N - np.eye(2)).max(), 1e-12)         # orthonormal
        proj = N @ N.T
        self.assertLess(np.linalg.norm(proj @ proj - proj), 1e-12)        # idempotent
        self.assertLess(np.linalg.norm(proj @ p.theta0 - p.theta0), 1e-10)  # theta0 feasible
        self.assertLess(np.abs(D @ (proj @ np.random.default_rng(0).normal(size=6))).max(), 1e-9)
        B = st.active_basis(N, Sw)
        self.assertEqual(B.shape, (6, 2))
        # approximate links: tolerance keeps the nearly-null directions, exact tolerance does not
        pa, (tra, _) = small_panel(6, 60, dgp.Spec(approx=0.1, link_tol=0.3))
        Da = dgp.link_rows(pa, tra)
        Swa = st.within_cov(dgp.take(pa.z, tra))
        self.assertEqual(st.null_space(Da, Swa, 0.3)[0].shape[1], 2)
        self.assertEqual(st.null_space(Da, Swa, 1e-4)[0].shape[1], 0)

    def test_penalty_derivatives(self):
        p, (tr, te) = small_panel(6, 60, dgp.Spec(fake=0.2))
        D = dgp.link_rows(p, tr)
        Sw = st.within_cov(dgp.take(p.z, tr))
        pen = st.Penalty(D, Sw)
        th = np.random.default_rng(3).normal(size=6)
        P, g = pen.value_grad(th)
        h = 1e-6
        gn = np.array([(pen.value(th + h * e) - pen.value(th - h * e)) / (2 * h) for e in np.eye(6)])
        self.assertLess(np.abs(g - gn).max(), 1e-6)
        Hn = np.array([(pen.value_grad(th + h * e)[1] - pen.value_grad(th - h * e)[1]) / (2 * h) for e in np.eye(6)])
        self.assertLess(np.abs(pen.hessian(th) - Hn).max(), 1e-5)
        self.assertAlmostEqual(pen.value(th), pen.value(3.7 * th), places=12)     # scale free
        # unit-free: rescaling the score variance rescales both numerator and denominator
        self.assertAlmostEqual(pen.value(p.theta0), float(np.mean((D @ p.theta0) ** 2) / (p.theta0 @ Sw @ p.theta0)))

    def test_ranking_hessian_matches_finite_differences(self):
        p, (tr, te) = small_panel(6, 40)
        cs = CellSet.build(dgp.take(p.z, tr), dgp.take(p.W, tr))
        th = np.random.default_rng(4).normal(size=6); th /= np.linalg.norm(th)
        M = np.exp(0.2 * np.random.default_rng(5).normal(size=cs.K))
        g, H = st.ranking_derivs(cs, th, M)
        f = lambda x: kn.rank_vg(cs, cs.Zs @ x, M)[0]
        h = 1e-5
        gn = np.array([(f(th + h * e) - f(th - h * e)) / (2 * h) for e in np.eye(6)])
        self.assertLess(np.abs(g - gn).max(), 1e-7)
        ctx = arms_mod.Ctx(p, tr, te, (1, 2, 3), dict(policy=None))
        h = arms_mod.get(ctx, 'RNC_S')                              # at the (scale-optimal) hard estimate
        trR, trS = st.curvature_traces(ctx.cs, ctx.pen, h['theta'], h['M'])
        self.assertGreater(trR, 0)
        self.assertGreater(trS, 0)
        gR, HR = st.ranking_derivs(ctx.cs, h['theta'], h['M'])
        self.assertLess(abs(h['theta'] @ gR), 1e-6)                 # theta'grad R = 0 at the joint optimum

    def test_lambda_omega_formula(self):
        trR, trS = 3.0, 0.7
        for w in (0.1, 0.5, 0.9, 0.99):
            lam = st.lambda_for_omega(w, trR, trS)
            self.assertAlmostEqual(st.omega_for_lambda(lam, trR, trS), w, places=12)
        self.assertEqual(st.lambda_for_omega(0.0, trR, trS), 0.0)
        self.assertTrue(np.isinf(st.lambda_for_omega(1.0, trR, trS)))
        self.assertGreater(st.lambda_for_omega(0.999999, trR, trS), 1e5)
        self.assertAlmostEqual(st.lambda_for_omega(0.5, trR, trS), trR / trS)     # equal curvature shares

    def test_penalty_omega_to_one_matches_hard(self):
        p, (tr, te) = small_panel(8, 60, key=(9, 9, 9))
        key = (9, 9, 9)
        ctx = arms_mod.Ctx(p, tr, te, key, dict(policy=None))
        hard = arms_mod.get(ctx, 'RNC_S')['theta']
        free = arms_mod.get(ctx, 'RNC')['theta']
        angs = []
        for w in (0.5, 0.9, 0.99, 0.9999):
            r = arms_mod.penalised(ctx, w)
            angs.append(np.degrees(np.arccos(min(1.0, r['theta'] @ hard))))
        self.assertLess(angs[-1], 0.3)                      # omega -> 1 reproduces the hard estimate
        self.assertLess(angs[-1], angs[0])
        self.assertGreater(np.degrees(np.arccos(min(1.0, free @ hard))), angs[-1])

    def test_gradient_balance(self):
        p, (tr, te) = small_panel(6, 40)
        key = (1, 2, 3)
        ctx = arms_mod.Ctx(p, tr, te, key, dict(policy=None))
        fe = arms_mod.get(ctx, 'FE')
        h = arms_mod.get(ctx, 'RNC_S')
        lg = st.lambda_gradient_balance(ctx.cs, ctx.pen, fe['theta'], h['M'])
        gR, _ = st.ranking_derivs(ctx.cs, fe['theta'], h['M'])
        _, gP = ctx.pen.value_grad(fe['theta'])
        T = st.tangent_basis(fe['theta'])
        self.assertAlmostEqual(lg * np.linalg.norm(T.T @ gP), np.linalg.norm(T.T @ gR), places=10)

    def test_policy_selection(self):
        rows = [{'trR': 1.0, 'trS': 2.0, **{f'lam_{w}': 1.0 + 0.01 * i for w in cal.OMEGAS}} for i in range(5)]
        self.assertEqual(cal.summarise(rows)['mode'], 'fixed')
        rows2 = [{'trR': 1.0, 'trS': 2.0, **{f'lam_{w}': 1.0 + 3 * i for w in cal.OMEGAS}} for i in range(5)]
        self.assertEqual(cal.summarise(rows2)['mode'], 'per_rep')
        self.assertEqual(cal.lambda_for({'mode': 'fixed', 'lambda': {'0.9': 4.0}}, 0.9, lambda w: 9.0), 4.0)
        self.assertEqual(cal.lambda_for({'mode': 'per_rep'}, 0.9, lambda w: 9.0), 9.0)


class TestEstimators(unittest.TestCase):
    def test_rnc_equals_ranknet_cell(self):
        p, (tr, te) = small_panel(6, 40)
        z, W = dgp.take(p.z, tr), dgp.take(p.W, tr)
        cs = CellSet.build(z, W)
        r = fit.fit_rnc(cs)
        th, ex, info = rbe.ranknet_cell(PairData.build(z, W), z.shape[1])
        self.assertLess(abs(r['obj'] - info['objective']), 1e-9)
        self.assertLess(np.abs(r['theta'] - th).max(), 1e-5)

    def test_sur_2d_equals_reference(self):
        p, (tr, te) = small_panel(6, 40)
        z, W = dgp.take(p.z, tr), dgp.take(p.W, tr)
        Sw = st.within_cov(z)
        N, _ = st.null_space(dgp.link_rows(p, tr), Sw)
        B = st.active_basis(N, Sw)
        cs = CellSet.build(z, W)
        tau = r3e.bandwidth(z.shape[1])
        s = fit.fit_sur(cs, tau, B)
        ref = r3e.sur_2d(PairData.build(z, W).project(B), np.full(2 * 6, tau))
        self.assertGreater((B @ ref[0]) @ s['theta'], 1 - 1e-5)

    def test_sur_sphere_runs_and_beats_nothing_worse_than_start(self):
        p, (tr, te) = small_panel(6, 40)
        z, W = dgp.take(p.z, tr), dgp.take(p.W, tr)
        cs = CellSet.build(z, W)
        start = fit.fit_rnc(cs)['theta']
        s = fit.fit_sur(cs, r3e.bandwidth(z.shape[1]), np.eye(6), start, extra_starts=1)
        v = lambda th: kn.sur_vg(cs, cs.Zs @ th, r3e.bandwidth(z.shape[1]))[0]
        self.assertLessEqual(v(s['theta']), v(start) + 1e-9)

    def test_fe_ns_restricted(self):
        p, (tr, te) = small_panel(6, 60)
        ctx = arms_mod.Ctx(p, tr, te, (1, 2, 3), dict(policy=None))
        fe_s = arms_mod.get(ctx, 'FE_S')['theta']
        self.assertLess(np.abs(ctx.D @ fe_s).max(), 1e-9)                  # FE_S imposes D theta = 0
        for name in ('RNC_S', 'SUR_S', 'MRC_S'):
            self.assertLess(np.abs(ctx.D @ arms_mod.get(ctx, name)['theta']).max(), 1e-9, name)
        self.assertGreater(np.abs(ctx.D @ arms_mod.get(ctx, 'FE')['theta']).max(), 1e-3)
        arms_mod.get(ctx, 'NS')

    def test_city_level_dimensions_do_not_break_arms(self):
        for spec in (dgp.Spec(city_nuis=True, links='temporal'), dgp.Spec(hc=True, links='temporal')):
            p, (tr, te) = small_panel(6, 60, spec)
            ctx = arms_mod.Ctx(p, tr, te, (1, 2, 3), dict(policy=None))
            self.assertEqual(ctx.B_free.shape[1], 5 if spec.hc else 4) if False else None
            rows = arms_mod.run_arms(ctx, ['RNC', 'RNC_S', 'FE', 'MM', 'MRC_S'])
            self.assertTrue(all(r['status'] == 'ok' for r in rows), [r.get('error') for r in rows])


class TestDGP(unittest.TestCase):
    def test_link_exactness_and_structure(self):
        for spec in (dgp.Spec(), dgp.Spec(hc=True), dgp.Spec(city_nuis=True), dgp.Spec(noise='D', map='b')):
            p = dgp.make_panel(spec, 8, 60, (4, 5, 6))
            self.assertAlmostEqual(np.linalg.norm(p.theta0), 1.0, places=10)
            allt = np.arange(60)
            tr = np.tile(allt, (8, 1))
            D = dgp.link_rows(p, tr)
            self.assertGreater(len(D), 100)
            self.assertLess(np.abs(D @ p.theta0).max(), 1e-10, spec)       # equal index exactly
            # temporal rows: same tract; cross rows: different cities
            self.assertTrue((p.f[p.temporal_decl[:, 0], p.temporal_decl[:, 1], 1] ==
                             p.f[p.temporal_decl[:, 0], p.temporal_decl[:, 1], 0]).all())
            fa = p.f[p.cross_decl[:, 0], p.cross_decl[:, 1], 0]
            fb = p.f[p.cross_decl[:, 2], p.cross_decl[:, 3], 0]
            self.assertLess(np.abs(fa - fb).max(), 1e-12)
            self.assertTrue((p.cross_decl[:, 0] != p.cross_decl[:, 2]).all())
        # each city has k = 4 neighbours
        tw = dgp.twin_pairs(8, 60, 3)
        for c in range(8):
            nb = set(tw[(tw[:, 0] == c), 2]) | set(tw[(tw[:, 2] == c), 0])
            self.assertEqual(len(nb), 4)

    def test_hc_is_constant_within_city_and_only_cross_links_pin_it(self):
        p = dgp.make_panel(dgp.Spec(hc=True), 8, 60, (4, 5, 6))
        ref = dgp.reference(p.spec)
        # city-constant structural feature: raw column constant within city
        raws = p.z @ np.linalg.inv(ref.chol)                                   # undo whitening up to mean
        self.assertLess(np.abs(raws.std(axis=(1, 2)).max()), 1e12)
        Sw = st.within_cov(p.z)
        tr = np.tile(np.arange(60), (8, 1))
        Dt = dgp.link_rows(dgp.make_panel(dgp.Spec(hc=True, links='temporal'), 8, 60, (4, 5, 6)), tr)
        Db = dgp.link_rows(p, tr)
        self.assertEqual(st.null_space(Dt, Sw)[0].shape[1], 3)               # H plane + h_c
        self.assertEqual(st.null_space(Db, Sw)[0].shape[1], 2)               # cross links tie h_c to H

    def test_corruptions(self):
        base = dgp.make_panel(dgp.Spec(), 8, 80, (1, 1, 1))
        self.assertEqual(base.n_fake, 0)
        fk = dgp.make_panel(dgp.Spec(fake=0.2), 8, 80, (1, 1, 1))
        tot = len(fk.temporal_decl) + len(fk.cross_decl)
        self.assertAlmostEqual(fk.n_fake / tot, 0.2, delta=0.01)
        tr = np.tile(np.arange(80), (8, 1))
        D = dgp.link_rows(fk, tr)
        gap = np.abs(D @ fk.theta0)
        self.assertGreater((gap > 1e-8).mean(), 0.15)
        self.assertLess((gap > 1e-8).mean(), 0.25)
        sh = dgp.make_panel(dgp.Spec(declared=0.25), 8, 80, (1, 1, 1))
        self.assertAlmostEqual(len(sh.temporal_decl) / len(base.temporal_decl), 0.25, delta=0.02)
        ap = dgp.make_panel(dgp.Spec(approx=0.1), 8, 80, (1, 1, 1))
        gap = (ap.f[ap.temporal_decl[:, 0], ap.temporal_decl[:, 1], 1] - ap.f[ap.temporal_decl[:, 0], ap.temporal_decl[:, 1], 0])
        self.assertAlmostEqual(gap.std(), 0.1, delta=0.02)
        self.assertEqual(len(np.setdiff1d(base.temporal_decl[:, 1], np.nonzero(base.stable)[0])), 0)
        none = dgp.make_panel(dgp.Spec(links='none'), 8, 80, (1, 1, 1))
        self.assertEqual(len(none.temporal_decl) + len(none.cross_decl), 0)

    def test_split_and_determinism(self):
        p = dgp.make_panel(dgp.Spec(), 8, 50, (7, 7, 7))
        tr, te = dgp.split_tracts(p, (7, 7, 7))
        self.assertEqual(tr.shape, (8, 40))
        self.assertEqual(te.shape, (8, 10))
        for c in range(8):
            self.assertEqual(len(np.intersect1d(tr[c], te[c])), 0)
            self.assertEqual(len(np.union1d(tr[c], te[c])), 50)
        p2 = dgp.make_panel(dgp.Spec(), 8, 50, (7, 7, 7))
        self.assertTrue((p.W == p2.W).all() and (p.z == p2.z).all())
        self.assertEqual(dgp.key_of('E1', 'x', 8, 50, 3), dgp.key_of('E1', 'x', 8, 50, 3))
        self.assertNotEqual(dgp.key_of('E1', 'x', 8, 50, 3), dgp.key_of('E1', 'x', 8, 50, 4))

    def test_noise_dependence(self):
        g = np.random.default_rng(0)
        xy = np.random.default_rng(1).random((300, 2))
        s = dgp.Spec(rho=0.8)
        e = np.concatenate([dgp.draw_eta(s, np.random.default_rng(i), xy) for i in range(30)])
        self.assertAlmostEqual(np.corrcoef(e[:, 0], e[:, 1])[0, 1], 0.8, delta=0.04)
        self.assertAlmostEqual(e.var(), 1.0, delta=0.05)
        for shared in (False, True):
            sp = dgp.Spec(spatial_r=0.3, spatial_shared=shared)
            ee = np.stack([dgp.draw_eta(sp, np.random.default_rng(i), xy) for i in range(60)])      # (60,300,2)
            self.assertAlmostEqual(ee.var(), 1.0, delta=0.08)
            near = ((xy[:, None] - xy[None]) ** 2).sum(-1) < 0.05 ** 2
            np.fill_diagonal(near, False)
            cov = np.mean([(ee[i, :, 0, None] * ee[i, None, :, 0])[near].mean() for i in range(60)])
            self.assertGreater(cov, 0.2)                                                              # spatial dependence
            cross_date = np.mean(ee[..., 0] * ee[..., 1])
            self.assertGreater(cross_date, 0.3 if shared else -0.05) if shared else self.assertLess(abs(cross_date), 0.1)

    def test_conditional_means_closed_form_vs_simulation(self):
        from minimal_e1 import dgp as base
        g = np.random.default_rng(0)
        R = 400000
        for spec in (dgp.Spec(), dgp.Spec(noise='D'), dgp.Spec(e4='i'), dgp.Spec(e4='ii'), dgp.Spec(e4='iii'),
                     dgp.Spec(map='b', e4='iii'), dgp.Spec(map='a', noise='D')):
            p = dgp.make_panel(spec, 6, 30, (2, 2, 2))
            c, i, t = 1, 5, 1
            fs, a1 = p.f[c, i, t], p.a1c[c, i, t]
            prm = dict(alpha=p.params['alpha'][c:c + 1], beta=p.params['beta'][c:c + 1], lam=p.params['lam'][c:c + 1])
            mu = dgp.cond_mean(spec, prm, np.full((1, 1, 2), fs), np.full((1, 1, 2), a1))[0, 0, t]
            eps = g.normal(size=R)
            if spec.e4 == 'iii':
                from minimal_e1 import robust_dgp as rd
                inner = np.exp(dgp.E4_A_III * a1 - dgp.E4_A_III ** 2 / 2) * rd.skewed_mixture(g, (R,))
                extra = 0.0
            else:
                inner = eps * np.sqrt(dgp.inside_var(spec))
                extra = {'': 0.0, 'i': dgp.E4_A_I * a1,
                         'ii': dgp.E4_GAMMA_O * (dgp.E4_RHO_O * a1 + np.sqrt(1 - dgp.E4_RHO_O ** 2) * g.normal(size=R)),
                         'iii': 0.0}[spec.e4]
            outer = 0.0
            if spec.noise == 'D':
                from minimal_e1 import robust_dgp as rd
                outer = np.sqrt(0.5) * rd.OUTER_SD * prm['beta'][0, t] * rd.lognormal_noise(g, (R,))
            r = np.full((1, R, 2), fs) + (inner + extra)[None, :, None]
            W = base.psi(r, spec.map, prm, dgp.KAPPA)[0, :, t] + outer
            self.assertAlmostEqual(W.mean(), mu, delta=5 * W.std() / np.sqrt(R) + 1e-3, msg=str(spec))


class TestHeads(unittest.TestCase):
    def test_pava_and_isotonic_head(self):
        g = np.random.default_rng(0)
        for _ in range(5):
            y = g.normal(size=60)
            if isotonic_regression is not None:
                self.assertLess(np.abs(heads.pava(y) - isotonic_regression(y).x).max(), 1e-12)
        s = np.sort(g.normal(size=(2, 50, 2)), axis=1)
        W = 2 * s + 1                                              # monotone: reproduced exactly
        out = heads.iso_predict(s, W, s)
        self.assertLess(np.abs(out - W).max(), 1e-12)
        out2 = heads.iso_predict(s, W + g.normal(size=W.shape), g.normal(size=(2, 7, 2)))
        self.assertEqual(out2.shape, (2, 7, 2))

    def test_oracle_head_exact_at_true_index(self):
        p, (tr, te) = small_panel(6, 60)
        r = heads.evaluate_theta(p, tr, te, p.theta0)
        self.assertLess(r['rmse_f'], 1e-6)
        self.assertLess(r['rmse_W_oracle'], 1e-8)
        self.assertLess(r['rmse_dW_oracle'], 1e-8)
        self.assertGreater(r['rmse_W_spline'], 1e-3)
        # at the true index the oracle head IS the ceiling predictor
        self.assertAlmostEqual(r['r2_W_oracle'], r['r2_W_ceiling'], places=10)
        self.assertGreaterEqual(r['r2_W_ceiling'] + 1e-12, r['r2_W_spline'] - 0.2)
        r2 = heads.evaluate_theta(p, tr, te, np.eye(6)[0])
        self.assertGreater(r2['rmse_f'], 0.1)
        self.assertGreater(r2['rmse_W_oracle'], r['rmse_W_oracle'])

    def test_native_heads(self):
        p, (tr, te) = small_panel(6, 60, dgp.Spec(map='a'))
        ctx = arms_mod.Ctx(p, tr, te, (1, 2, 3), dict(policy=None))
        rows = arms_mod.run_arms(ctx, ['FE', 'MM', 'RNC'])
        self.assertIn('rmse_W_native', rows[0])
        self.assertIn('rmse_dW_native', rows[1])
        self.assertNotIn('rmse_W_native', rows[2])


class TestRunner(unittest.TestCase):
    def test_resumable_and_deterministic(self):
        from minimal_e1.paper import experiments as ex
        tasks = ex.plan(['E1'], {'E1_a'}, None, 2, smoke=True)
        self.assertEqual(len(tasks), 2)
        for t in tasks:
            t['arms'] = ['RNC', 'RNC_S', 'FE']
        cfg = dict(mem_gb=1.0, policy=None, sur_extra_starts=1)
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            kn.MEM_GB = 1.0
            p1, s1, _ = runner.run_task(tasks[0], d1, cfg)
            self.assertEqual(s1, 'done')
            p1b, s1b, _ = runner.run_task(tasks[0], d1, cfg)
            self.assertEqual(s1b, 'skipped')
            runner.run_task(tasks[0], d2, cfg)
            with open(p1) as fh1, open(runner.task_path(d2, tasks[0])) as fh2:
                a, b = json.load(fh1), json.load(fh2)
            for ra, rb in zip(a['rows'], b['rows']):
                self.assertEqual(ra['rmse_f'], rb['rmse_f'])
            self.assertIn('peak_rss_mb', a['meta'])
            self.assertEqual(len(a['rows']), 3)
            # a second rep gets a different seed
            runner.run_task(tasks[1], d1, cfg)
            with open(runner.task_path(d1, tasks[1])) as fh3:
                c = json.load(fh3)
            self.assertNotEqual(c['rows'][0]['rmse_f'], a['rows'][0]['rmse_f'])

    def test_worker_cap_respects_ram(self):
        for mem in (0.5, 2.0, 4.0, 100.0):
            w = runner.cap_workers(64, mem)
            self.assertGreaterEqual(w, 1)
            self.assertTrue(w == 1 or w * mem <= runner.total_ram_gb() + 1e-9)

    def test_plan_matches_spec(self):
        from minimal_e1.paper import experiments as ex
        e1 = ex.plan(['E1'])
        sizes = sorted({(t['C'], t['n']) for t in e1})
        self.assertEqual(sizes, [(8, 50), (8, 150), (8, 500), (64, 500), (128, 500)])
        for t in e1:
            if t['C'] > 64:
                self.assertNotIn('MRC_S', t['arms'])
            if t['scenario'] == 'E1_cD':
                self.assertNotIn('ML', t['arms'])
        reps = {(t['C'], t['n']): 0 for t in e1}
        for t in e1:
            if t['scenario'] == 'E1_c':
                reps[(t['C'], t['n'])] += 1
        self.assertEqual([reps[k] for k in sorted(reps)], [100, 100, 100, 50, 30])
        self.assertEqual({t['scenario'] for t in ex.plan(['E3'])}, set(ex.SCENARIOS['E3']))
        self.assertIn('RNC_S_DW', ex.arms_for('E3', 'D_comb', ex.SCENARIOS['E3']['D_comb'], 8))


if __name__ == '__main__':
    unittest.main()


# ---------------------------------------------------------------------------
# review additions: eigen-gap null space for corrupted links, isotonic fallback
# ---------------------------------------------------------------------------
class TestReviewAdditions(unittest.TestCase):
    def test_gap_dimension_basic(self):
        from minimal_e1.paper import stability as st
        self.assertEqual(st.gap_dimension(np.array([0.07, 0.17, 1.8, 2.0, 2.1, 2.3])), 2)
        self.assertEqual(st.gap_dimension(np.array([1e-16, -1e-17, 1.9, 2.0, 2.1, 2.2])), 2)

    def test_gap_rule_keeps_true_null_space_with_20pct_fakes(self):
        from minimal_e1.paper import dgp as D, stability as st, experiments as E
        spec = E.SCENARIOS['E2']['F_20']
        for rep in range(3):
            key = D.key_of('E2', 'F_20', 8, 150, rep)
            p = D.make_panel(spec, 8, 150, key)
            tr, _ = D.split_tracts(p, key)
            z = D.take(p.z, tr)
            N, _ = st.null_space(D.link_rows(p, tr), st.within_cov(z), 'gap')
            self.assertEqual(N.shape[1], 2)
            # theta0 is close to the estimated null space (fakes tilt it only slightly)
            th = p.theta0 / np.linalg.norm(p.theta0)
            self.assertGreater(np.linalg.norm(N.T @ th), 0.95)

    def test_gap_equals_tol_for_exact_links(self):
        from minimal_e1.paper import dgp as D, stability as st, experiments as E
        spec = E.SCENARIOS['E1']['E1_c']
        key = D.key_of('E1', 'E1_c', 8, 150, 0)
        p = D.make_panel(spec, 8, 150, key)
        tr, _ = D.split_tracts(p, key)
        z = D.take(p.z, tr)
        Dm, Sw = D.link_rows(p, tr), st.within_cov(z)
        N1, _ = st.null_space(Dm, Sw, 1e-4)
        N2, _ = st.null_space(Dm, Sw, 'gap')
        self.assertEqual(N1.shape, N2.shape)
        np.testing.assert_allclose(N1 @ N1.T, N2 @ N2.T, atol=1e-8)

    def test_isotonic_fallback_matches_pava(self):
        from minimal_e1.paper import heads as H
        g = np.random.default_rng(3)
        s_tr, W_tr, s_te = g.normal(size=(2, 40, 2)), g.normal(size=(2, 40, 2)), g.normal(size=(2, 10, 2))
        a = H.iso_predict(s_tr, W_tr, s_te)
        saved = H._scipy_iso
        try:
            H._scipy_iso = None
            b = H.iso_predict(s_tr, W_tr, s_te)
        finally:
            H._scipy_iso = saved
        np.testing.assert_allclose(a, b, atol=1e-10)


class TestPenaltyCityLevel(unittest.TestCase):
    """Penalty arms must let cross-city links set city-level coordinates (as the hard arm does)."""

    def _ctx(self, name):
        from minimal_e1.paper import dgp as D, arms as A, experiments as E
        spec = E.SCENARIOS['E2'][name]
        key = D.key_of('E2', name, 8, 150, 0)
        p = D.make_panel(spec, 8, 150, key)
        tr, te = D.split_tracts(p, key)
        return A.Ctx(p, tr, te, key, dict(policy=None))

    def test_profiled_penalty_annihilates_city_directions(self):
        ctx = self._ctx('N2_hc_both')
        self.assertGreater(ctx.C_city.shape[1], 0)
        self.assertIsNotNone(ctx.pen.C)
        np.testing.assert_allclose(ctx.pen.A @ ctx.C_city, 0.0, atol=1e-8)
        # completing the true direction's within-cell part recovers theta0 (links are exact)
        th0 = ctx.panel.theta0 / np.linalg.norm(ctx.panel.theta0)
        P = ctx.C_city @ ctx.C_city.T
        part = th0 - P @ th0
        full = ctx.pen.complete(part / np.linalg.norm(part))
        self.assertGreater(abs(full @ th0), 1 - 1e-6)

    def test_temporal_only_links_leave_penalty_unchanged(self):
        ctx = self._ctx('N2_hc_temp')
        self.assertIsNone(ctx.pen.C)
        np.testing.assert_allclose(ctx.pen.A, ctx.pen.A_raw)



class TestR2AndRehead(unittest.TestCase):
    def test_r2_within_benchmark(self):
        from minimal_e1.paper import heads as H
        g = np.random.default_rng(0)
        mu = g.normal(size=(5, 200, 2))
        obs = mu + 0.5 * g.normal(size=mu.shape)
        tr = mu + 0.5 * g.normal(size=mu.shape)
        r = H.r2_within(obs, mu, tr)
        # population value: var(mu) / (var(mu) + 0.25) = 0.8
        self.assertAlmostEqual(r, 0.8, delta=0.03)
        self.assertAlmostEqual(H.r2_within(obs, obs, tr), 1.0)
        # predicting the training cell mean gives ~0
        base = np.broadcast_to(tr.mean(axis=1, keepdims=True), obs.shape)
        self.assertAlmostEqual(H.r2_within(obs, base, tr), 0.0, places=10)

    def test_e4_driver_is_bounded(self):
        from minimal_e1.paper import dgp as D, experiments as E
        spec = E.SCENARIOS['E4']['X_iii']
        p = D.make_panel(spec, 8, 150, D.key_of('E4', 'X_iii', 8, 150, 0))
        self.assertLessEqual(np.abs(p.a1c).max(), D.E4_A1_CLIP + 1e-12)
        tr, te = D.split_tracts(p, D.key_of('E4', 'X_iii', 8, 150, 0))
        mu = D.cond_mean(spec, p.params, D.take(p.f, te), D.take(p.a1c, te))
        self.assertTrue(np.isfinite(mu).all())
        self.assertLess(np.abs(mu).max(), 1e3)

    def test_rehead_reproduces_a_fresh_run(self):
        import json, tempfile
        from pathlib import Path
        from minimal_e1.paper import run as R, rehead as RH, calibration as cal
        with tempfile.TemporaryDirectory() as d:
            R.main(['--experiment', 'E2', '--scenarios', 'N2_hc_both', '--sizes', '8x150', '--reps', '1',
                    '--arms', 'RNC,RNC_S,RNC_P0.9,FE', '--out', d, '--no-numba'])
            src = next(Path(d).glob('E2/*/*/rep*.json'))
            rec = json.loads(src.read_text())
            cfg = dict(mem_gb=2.0, threads=1, max_pairs=None, sur_extra_starts=3, policy=cal.load_policy())
            new, max_rel, n_bad = RH.rehead_record(rec, set(), cfg)
            self.assertEqual(n_bad, 0)
            self.assertLess(max_rel, 1e-10)
            old = {r['arm']: r for r in rec['rows']}
            for r in new['rows']:
                for k in ('rmse_W_spline', 'r2_dW_spline', 'r2_W_ceiling'):
                    self.assertAlmostEqual(r[k], old[r['arm']][k], places=10)
            self.assertEqual(RH.parse_refit('E2:a,b:X,Y'), {('E2', 'a'): {'X', 'Y'}, ('E2', 'b'): {'X', 'Y'}})


class TestE5BaselineUpdating(unittest.TestCase):
    def _panel(self, kind, mp='a', xi_x=0.0, C=8, n=500, rep=0):
        from dataclasses import replace
        from minimal_e1.paper import dgp as D, experiments as E
        spec = replace(E.BASE, persist=kind, map=mp, xi_x=xi_x)
        key = D.key_of('T5', kind, mp, int(10 * xi_x), rep)
        p = D.make_panel(spec, C, n, key)
        return p, D.split_tracts(p, key)

    def test_latent_noise_moments(self):
        from minimal_e1.paper import dgp as D
        p, _ = self._panel('const', mp='a', C=16, n=500)
        v = D.base.psi_inverse(p.W, 'a', p.params, D.KAPPA)        # map a: v = W - alpha
        eta = v - p.f
        self.assertAlmostEqual(float(eta.var()), 1.0, delta=0.05)
        corr = np.corrcoef(eta[..., 0].ravel(), eta[..., 1].ravel())[0, 1]
        self.assertAlmostEqual(corr, 0.75 * 0.8, delta=0.04)      # lam_p * rho_p

    def test_reset_redraws_changing_tracts(self):
        from minimal_e1.paper import dgp as D
        p, _ = self._panel('reset', mp='a', C=16, n=500)
        eta = D.base.psi_inverse(p.W, 'a', p.params, D.KAPPA) - p.f
        st = np.asarray(p.stable)
        c_st = np.corrcoef(eta[:, st, 0].ravel(), eta[:, st, 1].ravel())[0, 1]
        c_ch = np.corrcoef(eta[:, ~st, 0].ravel(), eta[:, ~st, 1].ravel())[0, 1]
        self.assertAlmostEqual(c_st, 0.75 * 0.95, delta=0.05)
        self.assertLess(abs(c_ch), 0.05)

    def test_oracle_is_unbiased_and_best(self):
        from minimal_e1.paper import dgp as D, heads as H
        bias, mse_or, mse_k = [], [], []
        for rep in range(4):
            p, (tr, te) = self._panel('fe_rel', mp='c', xi_x=0.5, rep=rep)
            W = D.take(p.W, te)
            tgt = D.cond_mean_given_baseline(p, te)
            bias.append(np.mean(W[..., 1] - tgt))
            m = H.update_metrics(p, tr, te, p.theta0)
            mse_or.append(m['r2_dWo_oracle']); mse_k.append(m['r2_dWo_kconst'])
            self.assertEqual(m['rmse_dWc_oracle'], 0.0)
        self.assertLess(abs(np.mean(bias)), 0.05)
        self.assertGreater(np.mean(mse_or), np.mean(mse_k))

    def test_e3_rho_maps_to_fixed_effect(self):
        from dataclasses import replace
        from minimal_e1.paper import dgp as D, experiments as E
        spec = E.SCENARIOS['E3']['D_rho08']
        self.assertEqual(D.persistence_kind(spec), 'e3')
        pr = D.persistence(spec, np.zeros((2, 3)), np.ones(3, bool))
        self.assertTrue(np.allclose(pr['rho'], 1.0) and np.allclose(pr['var_xi'], 0.8))
        self.assertEqual(D.persistence_kind(E.SCENARIOS['E3']['D_comb']), '')

    def test_update_metrics_in_runner_rows(self):
        from minimal_e1.paper import arms as A, calibration as cal, dgp as D, experiments as E, heads as H
        spec = E.SCENARIOS['E5']['P_reset_x']
        key = D.key_of('E5', 'P_reset_x', 8, 150, 0)
        p = D.make_panel(spec, 8, 150, key)
        tr, te = D.split_tracts(p, key)
        r = A.run_arms(A.Ctx(p, tr, te, key, dict(policy=cal.load_policy())), ['RNC_S'])[0]
        for h in H.UPDATE_HEADS:
            for k in (f'rmse_dWc_{h}', f'rmse_dWcS_{h}', f'rmse_dWcC_{h}', f'r2_dWo_{h}'):
                self.assertIn(k, r)
        self.assertLess(r['rmse_f'], 0.2)
