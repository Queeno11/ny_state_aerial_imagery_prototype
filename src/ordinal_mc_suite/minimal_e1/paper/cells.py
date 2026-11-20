"""CellSet: training data arranged as K = 2C city-date cells, each sorted by outcome."""
import numpy as np

from . import kernels as kn


class CellSet:
    """Zs (K,n,p) features, Ws (K,n) outcomes ascending, XY (K,n,2) tract coordinates,
    W0 (K,) per-cell pair weight (includes the 1/K cell average), ell (distance weighting,
    0 = off), pairs (None = all pairs, else fixed pair subsample)."""

    def __init__(self, Zs, Ws, XY, W0, ell=0.0, pairs=None):
        self.Zs, self.Ws, self.XY, self.W0, self.ell, self.pairs = Zs, Ws, XY, W0, float(ell), pairs
        self.K, self.n, self.p = Zs.shape

    @classmethod
    def build(cls, z, W, xy=None, ell=0.0, max_pairs=None, seed=0):
        """z (C,n,2,p), W (C,n,2), xy (C,n,2) -> cells indexed k = 2c + t (as PairData)."""
        C, n, _, p = z.shape
        Z = np.ascontiguousarray(z.transpose(0, 2, 1, 3).reshape(2 * C, n, p))
        Wc = np.ascontiguousarray(W.transpose(0, 2, 1).reshape(2 * C, n))
        if xy is None:
            xy = np.zeros((C, n, 2))
        XY = np.ascontiguousarray(np.repeat(xy, 2, axis=0).reshape(C, 2, n, 2).reshape(2 * C, n, 2))
        order = np.argsort(Wc, axis=1, kind='stable')
        Z = np.take_along_axis(Z, order[..., None], 1)
        Wc = np.take_along_axis(Wc, order, 1)
        XY = np.take_along_axis(XY, order[..., None], 1)
        cs = cls(Z, Wc, XY, np.ones(2 * C), ell)
        if max_pairs is not None and max_pairs < n * (n - 1) // 2:
            if ell > 0:
                raise ValueError('pair subsample is not combined with distance weighting')
            cs.pairs = kn.draw_pairs(Wc, int(max_pairs), np.random.default_rng(seed))
            norm = np.array([float((Wc[k][jj] > Wc[k][ii]).sum()) for k, (ii, jj) in enumerate(cs.pairs)])
        else:
            norm = kn.pair_norm(Wc, XY, ell)
        if np.any(norm <= 0):
            raise ValueError('a city-date cell has only tied outcomes')
        cs.W0 = 1.0 / (cs.K * norm)
        return cs

    def project(self, B):
        """Same cells, features restricted to coordinates u -> z'(B u)."""
        out = CellSet(np.ascontiguousarray(self.Zs @ B), self.Ws, self.XY, self.W0, self.ell, self.pairs)
        return out

    def scores(self, theta):
        return self.Zs @ theta

    def n_pairs(self):
        if self.pairs is not None:
            return int(sum(len(p[0]) for p in self.pairs))
        return int(self.K * self.n * (self.n - 1) // 2)
