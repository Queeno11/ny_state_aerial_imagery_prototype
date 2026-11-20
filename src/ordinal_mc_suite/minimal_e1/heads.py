"""Correct-family outcome heads profiled for a frozen score, and the C_q loss.

C_q(theta) = (1/(C n v*)) sum_{c,i} e_ci' P e_ci,  P = 1/4 [[1, 1-2q], [1-2q, 1]],
which equals (1/(C n v*)) sum [(1-q) a^2 + (q/4) d^2] with a = mean error and
d = date-1 minus date-0 error. At q = 0.5 it is half the endpoint MSE / v*.

Heads (intercepts alpha_ct always free and profiled by within-cell centering,
which is exact for any positive-definite P):
  'a'  alpha_ct + b s, common b > 0 (map a: additive / FE)
  'b'  alpha_ct + beta_ct s, beta_ct > 0 (map b; also the practitioner affine head)
  'c'  alpha_ct + c1_ct e^{kappa s} + c2_ct e^{-kappa s}, c1 >= 0 >= c2
       (the conditional mean of map c is exactly in this family; the city-level
       sharing of lambda across dates is relaxed, documented in DEVIATIONS.md)
"""
from dataclasses import dataclass
import itertools

import numpy as np

FLOOR = 1e-8


def weight_matrix(q):
    r = 1.0 - 2.0 * q
    return 0.25 * np.array([[1.0, r], [r, 1.0]])


def center(a):
    return a - a.mean(axis=1, keepdims=True)


def v_star(W):
    """Average within-cell-date sample variance of the source outcomes."""
    return float(np.mean(np.var(W, axis=1, ddof=1)))


@dataclass
class Head:
    family: str
    alpha: np.ndarray            # (C, 2)
    coef: np.ndarray             # 'a': scalar b; 'b': (C,2); 'c': (C,2,2) [c1, -c2] >= 0
    kappa: float
    boundary: bool = False

    def _parts(self, s):
        if self.family == 'a':
            return self.coef * s, np.full_like(s, self.coef)
        if self.family == 'b':
            beta = self.coef[:, None, :]
            return beta * s, np.broadcast_to(beta, s.shape).copy()
        if self.family == 'c':
            x1 = self.coef[:, None, :, 0]
            x2 = self.coef[:, None, :, 1]
            ep, em = np.exp(self.kappa * s), np.exp(-self.kappa * s)
            return x1 * ep - x2 * em, self.kappa * (x1 * ep + x2 * em)
        raise ValueError(self.family)

    def predict(self, s):
        part, _ = self._parts(s)
        return self.alpha[:, None, :] + part

    def derivative(self, s):
        return self._parts(s)[1]

    @property
    def common_slope(self):
        return float(self.coef) if self.family == 'a' else None


def _solve_nonneg(G, rhs):
    """min x'Gx - 2 x'rhs s.t. x >= 0 for a batch of tiny convex QPs, by
    enumerating free sets (the optimum is the unconstrained minimiser of its
    own active face, which is feasible)."""
    C, k = rhs.shape
    best = np.zeros((C, k))
    best_val = np.zeros(C)                    # x = 0 is always feasible
    for size in range(1, k + 1):
        for free in itertools.combinations(range(k), size):
            idx = list(free)
            Gf = G[:, idx][:, :, idx]
            rf = rhs[:, idx]
            try:
                xf = np.linalg.solve(Gf, rf[..., None])[..., 0]
            except np.linalg.LinAlgError:
                xf = np.stack([np.linalg.lstsq(g, r, rcond=None)[0] for g, r in zip(Gf, rf)])
            feasible = np.all(xf >= 0, axis=1)
            val = np.einsum('ci,cij,cj->c', xf, Gf, xf) - 2 * np.einsum('ci,ci->c', xf, rf)
            take = feasible & (val < best_val - 1e-15)
            if np.any(take):
                cand = np.zeros((C, k))
                cand[:, idx] = xf
                best[take] = cand[take]
                best_val[take] = val[take]
    return best


def fit_head(s, W, family, q, kappa, fixed_b=None):
    """Least-squares head under C_q for a frozen score s (C, n, 2)."""
    P = weight_matrix(q)
    sm, Wm = s.mean(axis=1), W.mean(axis=1)
    sc, Wc = s - sm[:, None, :], W - Wm[:, None, :]
    if family == 'a':
        if fixed_b is None:
            num = np.einsum('cnt,tu,cnu->', sc, P, Wc)
            den = np.einsum('cnt,tu,cnu->', sc, P, sc)
            b = num / den if den > 0 else 0.0
            boundary = b <= FLOOR
            b = max(b, FLOOR)
        else:
            b, boundary = float(fixed_b), False
        return Head('a', Wm - b * sm, np.float64(b), kappa, bool(boundary))
    if family == 'b':
        G = P[None] * np.einsum('cnt,cnu->ctu', sc, sc)
        rhs = np.einsum('tu,cnt,cnu->ct', P, sc, Wc)
        # beta >= FLOOR  <=>  x = beta - FLOOR >= 0
        x = _solve_nonneg(G, rhs - np.einsum('ctu,u->ct', G, np.full(2, FLOOR)))
        beta = x + FLOOR
        return Head('b', Wm - beta * sm, beta, kappa, bool(np.any(x <= 0)))
    if family == 'c':
        ep, em = np.exp(kappa * s), np.exp(-kappa * s)
        basis = np.stack([center(ep), -center(em)], axis=-1)        # (C,n,2,2) [date, k]
        C = s.shape[0]
        # Variables ordered (t, k); x >= 0 encodes c1 >= 0 and c2 = -x2 <= 0.
        G = np.einsum('tu,cntk,cnul->ctkul', P, basis, basis).reshape(C, 4, 4)
        rhs = np.einsum('tu,cntk,cnu->ctk', P, basis, Wc).reshape(C, 4)
        x = _solve_nonneg(G, rhs).reshape(C, 2, 2)
        head = Head('c', np.zeros((C, 2)), x, kappa, bool(np.any(x <= 0)))
        part, _ = head._parts(s)
        head.alpha = Wm - part.mean(axis=1)
        return head
    raise ValueError(family)


def criterion(s, W, head, q, vstar):
    """C_q value and its envelope derivative with respect to the score."""
    P = weight_matrix(q)
    e = head.predict(s) - W
    Pe = np.einsum('tu,cnu->cnt', P, e)
    scale = s.shape[0] * s.shape[1] * vstar
    value = float(np.sum(e * Pe) / scale)
    grad_s = 2.0 * Pe * head.derivative(s) / scale
    return value, grad_s


def profile(s, W, family, q, kappa, vstar):
    head = fit_head(s, W, family, q, kappa)
    value, grad_s = criterion(s, W, head, q, vstar)
    return value, grad_s, head


def null_value(W, q, vstar):
    """C_q at a constant score: only the cell intercepts are fitted."""
    P = weight_matrix(q)
    Wc = center(W)
    return float(np.einsum('cnt,tu,cnu->', Wc, P, Wc) / (W.shape[0] * W.shape[1] * vstar))
