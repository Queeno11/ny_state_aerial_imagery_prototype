"""RankNet pairwise loss on a linear score and its exact Newton solver.

P(u) = sum_{c,t} 1/(2C) * mean_{i<j} log(1 + exp(-y_ij u'(z_j - z_i))),
y_ij = sign(W_j - W_i) within each city-date cell. With linear f and a free
temperature, u = theta / tau and P is convex (a logistic regression on pair
differences). P(0) = log 2.
"""
import numpy as np
from scipy.special import expit


class PairData:
    def __init__(self, v, weight, ties=0, cell=None, n_cells=None):
        self.v = v                  # (N, p) sign-oriented pair differences
        self.weight = weight        # (N,) sums to one
        self.ties = ties
        self.cell = cell            # (N,) city-date cell index c*2 + t
        self.n_cells = n_cells

    @classmethod
    def build(cls, z, W):
        C, n, _, p = z.shape
        ii, jj = np.triu_indices(n, 1)
        y = np.sign(W[:, jj, :] - W[:, ii, :])                     # (C, m, 2)
        diff = z[:, jj] - z[:, ii]                                   # (C, m, 2, p)
        v = (y[..., None] * diff).transpose(0, 2, 1, 3)             # (C, 2, m, p)
        y = y.transpose(0, 2, 1)
        counts = np.count_nonzero(y, axis=2)                         # (C, 2)
        if np.any(counts == 0):
            raise ValueError('A city-date cell has only tied outcomes')
        weight = np.broadcast_to((1.0 / (2 * C * counts))[..., None], y.shape)
        keep = (y != 0).ravel()
        cell = np.broadcast_to(np.arange(2 * C).reshape(C, 2)[..., None], y.shape).ravel()
        return cls(np.ascontiguousarray(v.reshape(-1, p)[keep]),
                   np.ascontiguousarray(weight.ravel()[keep]),
                   int(y.size - keep.sum()), cell[keep].astype(np.int32), 2 * C)

    def project(self, basis):
        return PairData(self.v @ basis, self.weight, self.ties, self.cell, self.n_cells)

    def value(self, u):
        m = self.v @ u
        return float(self.weight @ np.logaddexp(0.0, -m))

    def value_grad(self, u):
        m = self.v @ u
        val = float(self.weight @ np.logaddexp(0.0, -m))
        grad = -(self.weight * expit(-m)) @ self.v
        return val, grad

    def value_grad_hess(self, u):
        m = self.v @ u
        val = float(self.weight @ np.logaddexp(0.0, -m))
        sig = expit(-m)
        grad = -(self.weight * sig) @ self.v
        curvature = self.weight * sig * (1.0 - sig)
        hess = (self.v * curvature[:, None]).T @ self.v
        return val, grad, hess


def newton_ranknet(pairs, tol=1e-11, max_iter=200, u0=None):
    """Damped Newton from u0 (default 0). Returns u and convergence diagnostics."""
    p = pairs.v.shape[1]
    u = np.zeros(p) if u0 is None else np.array(u0, float)
    val, grad, hess = pairs.value_grad_hess(u)
    it = 0
    for it in range(1, max_iter + 1):
        step = -np.linalg.solve(hess + 1e-14 * np.eye(p), grad)
        decrement = -grad @ step
        # Stop on the gradient (quadratic convergence makes this cheap); the
        # decrement test alone can stop early when the Hessian is large.
        if np.max(np.abs(grad)) < 1e-10 or decrement / 2 < tol * 1e-8:
            break
        t = 1.0
        while True:
            new = u + t * step
            new_val = pairs.value(new)
            if new_val <= val - 1e-4 * t * decrement or t < 1e-12:
                break
            t *= 0.5
        if t < 1e-12 or new_val >= val:
            break                    # no further decrease at floating-point precision
        u = new
        val, grad, hess = pairs.value_grad_hess(u)
    norm = np.linalg.norm(u)
    info = dict(objective=val, grad_norm=float(np.max(np.abs(grad))), iterations=it,
                success=bool(np.max(np.abs(grad)) < 1e-6 and norm < 1e6),
                separation=bool(norm >= 1e6))
    return u, info
