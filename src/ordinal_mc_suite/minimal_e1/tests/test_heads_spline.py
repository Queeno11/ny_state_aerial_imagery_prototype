"""Synthetic tests for paper/heads_spline.py.  From /mnt/user-data/outputs:
    PYTHONPATH=. python -m pytest -q minimal_e1/tests/test_heads_spline.py
"""
import unittest

import numpy as np

from minimal_e1.paper import heads_spline as hs


def make(truth, C=3, n=300, noise=0.3, seed=1):
    r = np.random.default_rng(seed)
    s = r.normal(size=(C, n, 2))
    return s, truth(s) + noise * r.normal(size=s.shape)


def grid_s(C=3, m=200, lo=-4, hi=4):
    return np.broadcast_to(np.linspace(lo, hi, m)[None, :, None], (C, m, 2)).copy()


class TestSpline(unittest.TestCase):
    def test_linear_truth_recovered(self):
        lin = lambda s: 1.0 + 2.0 * s
        s, W = make(lin, n=600, noise=0.2)
        g = grid_s(lo=-1.5, hi=1.5)
        for df in (2, 4):
            err = np.sqrt(np.mean((hs.bspline_predict(s, W, g, df) - lin(g)) ** 2))
            self.assertLess(err, 0.08, df)

    def test_spline_beats_linear_on_convex_monotone(self):
        tr = lambda s: np.exp(1.2 * s)
        s, W = make(tr, n=400, noise=0.3)
        g = grid_s(lo=-1.5, hi=1.5)
        e = lambda p: np.sqrt(np.mean((p - tr(g)) ** 2))
        self.assertLess(e(hs.bspline_predict(s, W, g, 6)), 0.5 * e(hs.bspline_predict(s, W, g, 2)))

    def test_monotone_output_nondecreasing(self):
        s, W = make(lambda s: s + 0.0, n=150, noise=1.5)      # noisy: unconstrained fit wiggles
        g = grid_s(m=600, lo=-6, hi=6)
        un = hs.bspline_predict(s, W, g, 8, False)
        self.assertTrue((np.diff(un, axis=1) < -1e-9).any())   # the test has teeth
        for df in (2, 5, 8):
            mo = hs.bspline_predict(s, W, g, df, True)
            self.assertTrue((np.diff(mo, axis=1) >= -1e-9).all(), df)

    def test_linear_extrapolation(self):
        s, W = make(lambda s: np.tanh(s), n=300)
        for mono in (False, True):
            xs = np.stack([np.linspace(8, 12, 50), np.linspace(-12, -8, 50)])   # far outside both ends
            for df in (4, 6):
                m = hs.fit_cell(s[0, :, 0], W[0, :, 0], df, mono)
                for x in xs:
                    p = hs.predict_cell(m, x)
                    self.assertLess(np.abs(np.diff(p, 2)).max(), 1e-9)
        # continuity at the boundary
        m = hs.fit_cell(s[0, :, 0], W[0, :, 0], 6)
        hi = s[0, :, 0].max()
        a, b = hs.predict_cell(m, np.array([hi - 1e-7, hi + 1e-7]))
        self.assertLess(abs(a - b), 1e-5)

    def test_cv_picks_small_then_large_df(self):
        s, W = make(lambda s: 1 + 2 * s, n=400, noise=0.5)
        d_lin, tab = hs.cv_select_df(s, W, seed=0)
        self.assertIn(d_lin, (2, 4))
        s, W = make(lambda s: 2 * np.sin(1.8 * s), n=400, noise=0.3)
        d_curv, _ = hs.cv_select_df(s, W, seed=0)
        self.assertGreaterEqual(d_curv, 6)
        self.assertEqual(set(tab), {2, 4, 5, 6, 8})

    def test_cv_predict_shapes_and_monotone_df2(self):
        s, W = make(lambda s: s, n=100)
        p, df = hs.cv_spline_predict(s, W, s[:, :20], monotone=True)
        self.assertEqual(p.shape, (3, 20, 2))
        self.assertIn(df, (2, 4, 5, 6, 8))
        m = hs.fit_cell(s[0, :, 0], -W[0, :, 0], 2, True)     # decreasing truth -> slope clamped at 0
        self.assertAlmostEqual(m['coef'][1], 0.0)


class TestPSpline(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(5)
        self.s = r.normal(size=400)
        self.w = np.sin(2 * self.s) + 0.1 * r.normal(size=400)

    def test_huge_penalty_gives_straight_line(self):
        m = hs.fit_cell_ps(self.s, self.w, 1e8)
        g = np.linspace(-2, 2, 50)
        p = hs.predict_cell(m, g)
        b = np.polyfit(self.s, self.w, 1)
        np.testing.assert_allclose(p, np.polyval(b, g), atol=1e-3)

    def test_small_penalty_follows_curvature(self):
        g = np.linspace(-1.5, 1.5, 50)
        err = np.mean((hs.predict_cell(hs.fit_cell_ps(self.s, self.w, 1e-3), g) - np.sin(2 * g)) ** 2)
        self.assertLess(err, 0.01)

    def test_monotone_respected(self):
        m = hs.fit_cell_ps(self.s, self.w, 1e-2, monotone=True)
        p = hs.predict_cell(m, np.linspace(-3, 3, 200))
        self.assertTrue(np.all(np.diff(p) >= -1e-9))

    def test_cv_penalty_larger_for_linear_truth(self):
        s, W = make(lambda x: 2 * x, C=2, n=300, noise=0.5)
        lam_lin, _ = hs.cv_select_df(s, W, kind='ps')
        s, W = make(lambda x: np.sin(3 * x), C=2, n=300, noise=0.2)
        lam_wig, _ = hs.cv_select_df(s, W, kind='ps')
        self.assertGreater(lam_lin, lam_wig)

    def test_generic_predict_and_joint_cv_accept_ps(self):
        s, W, stable = make_panel_pair(C=2, n=200)
        lam, tab = hs.cv_select_df_joint(s, W, dw_mask=~stable, kind='ps')
        self.assertIn(lam, tab)
        p = hs.bspline_predict(s, W, s, lam, True, 'ps')
        self.assertEqual(p.shape, W.shape)


class TestLassoAndMedian(unittest.TestCase):
    def test_l1_huge_penalty_is_a_line_and_small_follows_curve(self):
        r = np.random.default_rng(2)
        s = r.normal(size=400)
        w = np.sin(2 * s) + 0.1 * r.normal(size=400)
        g = np.linspace(-1.5, 1.5, 40)
        p = hs.predict_cell(hs.fit_cell_l1(s, w, 1e6), g)
        self.assertLess(np.max(np.abs(np.diff(p, 2))), 1e-3)            # straight line
        p = hs.predict_cell(hs.fit_cell_l1(s, w, 1e-3), g)
        self.assertLess(np.mean((p - np.sin(2 * g)) ** 2), 0.01)

    def test_l1_finds_a_single_kink(self):
        r = np.random.default_rng(3)
        s = np.sort(r.uniform(-2, 2, 600))
        w = np.maximum(s, 0) * 2 + 0.05 * r.normal(size=600)            # piecewise linear, one kink at 0
        m = hs.fit_cell_l1(s, w, 0.05, monotone=True)
        g = np.linspace(-1.8, 1.8, 37)
        self.assertLess(np.max(np.abs(hs.predict_cell(m, g) - 2 * np.maximum(g, 0))), 0.15)

    def test_median_ignores_outliers_mean_does_not(self):
        r = np.random.default_rng(4)
        s = r.normal(size=400)
        w = s + 0.1 * r.normal(size=400)
        w[np.argsort(s)[-3:]] += 100.0                                  # three extreme tracts in the top tail
        g = np.linspace(1.0, 2.0, 11)
        med = hs.predict_cell(hs.fit_cell_median(s, w, 5, monotone=True), g)
        mean = hs.predict_cell(hs.fit_cell(s, w, 5, monotone=True), g)
        self.assertLess(np.max(np.abs(med - g)), 0.2)
        self.assertGreater(np.max(np.abs(mean - g)), 1.0)

    def test_median_targets_median_under_skew(self):
        r = np.random.default_rng(6)
        s = r.normal(size=3000)
        w = np.exp(s + r.normal(size=3000))                             # median exp(s), mean exp(s + 1/2)
        g = np.linspace(-1, 1, 9)
        med = hs.predict_cell(hs.fit_cell_median(s, w, 5, monotone=True), g)
        np.testing.assert_allclose(med, np.exp(g), rtol=0.15)

    def test_cv_accepts_new_kinds(self):
        s, W = make(lambda x: np.tanh(x), C=2, n=200)
        for kind in ('l1', 'med'):
            g, tab = hs.cv_select_df(s, W, kind=kind)
            self.assertIn(g, tab)


def make_panel_pair(C=4, n=300, curve=np.tanh, kappa=0.6, noise=0.2, seed=3):
    """Two-date synthetic panel: half the tracts move their index, a persistent residual carries over."""
    r = np.random.default_rng(seed)
    s0 = r.normal(size=(C, n))
    move = np.arange(n) % 2 == 1
    s1 = s0 + np.where(move, 0.8 * r.normal(size=(C, n)), 0.0)
    u0 = r.normal(size=(C, n))
    u1 = kappa * u0 + np.sqrt(1 - kappa ** 2) * r.normal(size=(C, n))
    W = np.stack([2 * curve(s0) + 0.5 * u0, 2 * curve(s1) + 0.5 + 0.5 * u1], -1) + noise * r.normal(size=(C, n, 2))
    return np.stack([s0, s1], -1), W, ~move


class TestJointDf(unittest.TestCase):
    def test_fit_carry_recovers_weights_and_intercepts(self):
        r = np.random.default_rng(0)
        city = np.repeat(np.arange(3), 2000)
        r0, R0 = r.normal(size=6000), r.normal(size=6000)
        r1 = np.array([0.1, -0.2, 0.3])[city] + (0.5 + 0.2 * R0) * r0 + 0.01 * r.normal(size=6000)
        pred = hs.fit_carry(r0, r1, R0, city, by_r=True)
        np.testing.assert_allclose(pred(r0, R0, city), r1, atol=0.05)
        p0 = hs.fit_carry(r0, r1, R0, city, by_r=False)
        self.assertLess(np.mean((p0(r0, R0, city) - r1) ** 2), np.var(r1))

    def test_r2_pooled_within(self):
        obs = np.array([1., 2., 3., 10., 11., 12.])
        g = np.array([0, 0, 0, 1, 1, 1])
        self.assertAlmostEqual(hs._r2_pooled_within(obs, obs, g), 1.0)
        self.assertAlmostEqual(hs._r2_pooled_within(obs, np.array([2., 2, 2, 11, 11, 11]), g), 0.0)

    def test_weight_zero_is_levels_only(self):
        s, W, stable = make_panel_pair()
        _, tab = hs.cv_select_df_joint(s, W, grid=(2, 4, 6), weight=0.0)
        for loss, r2w, _ in tab.values():
            self.assertAlmostEqual(loss, 1 - r2w)

    def test_returns_grid_value_and_sane_r2(self):
        s, W, stable = make_panel_pair()
        df, tab = hs.cv_select_df_joint(s, W, grid=(2, 4, 6, 8), dw_mask=~stable, weight=0.5, monotone=True)
        self.assertIn(df, (2, 4, 6, 8))
        r2w, r2d = tab[df][1], tab[df][2]
        self.assertGreater(r2w, 0.5)
        self.assertGreater(r2d, 0.2)

    def test_curved_truth_beats_linear_on_changes(self):
        s, W, stable = make_panel_pair(curve=lambda x: np.tanh(2 * x), noise=0.1)
        df, tab = hs.cv_select_df_joint(s, W, grid=(2, 4, 6, 8), dw_mask=~stable, weight=1.0)
        self.assertGreater(df, 2)
        self.assertGreater(tab[df][2], tab[2][2])

    def test_single_df_grid_and_per_tract_mask(self):
        """A one-value grid and a (n,) mask broadcast over cities both work."""
        s, W, stable = make_panel_pair(n=200)
        _, t1 = hs.cv_select_df_joint(s, W, grid=(4,), weight=1.0, dw_mask=~stable)
        self.assertTrue(np.isfinite(t1[4][2]))


if __name__ == '__main__':
    unittest.main()
