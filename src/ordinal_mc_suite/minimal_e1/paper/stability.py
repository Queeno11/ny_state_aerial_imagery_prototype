"""Stability restrictions: hard null space, curvature-share calibrated penalty, lambda(omega).

Links give rows d_l = x_a - x_b (equal index: d'theta0 = 0), stacked in D (L, p) in whitened
coordinates.  HARD: theta = N u, N an orthonormal basis of {theta : mean_l (d_l'theta)^2 /
Var_w(x'theta) <= tol^2} (generalised eigenproblem A v = rho S_w v, A = D'D/L; tol = 1e-4 for exact
links = the exact null space; a larger declared tolerance gives an approximate null space).
PENALTY: P(theta) = theta'A theta / theta'S_w theta (unit-free: mean squared link gap over the
pooled within-cell variance of the score; 0-homogeneous), added to the MEAN ranking loss as
lambda * P.  lambda is calibrated by the curvature share
      omega(lambda) = lambda tr(H_S) / (tr(H_R) + lambda tr(H_S)),
tr over the tangent space of the unit sphere at the hard (or unrestricted) estimate on TRAINING
data:  lambda(omega) = omega tr(H_R) / ((1 - omega) tr(H_S)), omega -> 1 = hard.
Gradient balance: lambda_g = ||grad R|| / ||grad P|| at the starting value.
"""
import numpy as np
from scipy.linalg import eigh

from . import kernels as kn


def within_cov(z):
    """Pooled within-cell covariance of z (C,n,2,p): mean over cells of the cell covariance."""
    zc = z - z.mean(axis=1, keepdims=True)
    return np.einsum('cntp,cntq->pq', zc, zc) / (z.shape[0] * z.shape[1] * 2)


def gap_dimension(ev, floor=1e-12):
    """Null-space dimension by the largest log-gap of the sorted generalised eigenvalues
    (data-driven, needs no tolerance; used when declared links may be fake or approximate)."""
    e = np.maximum(np.sort(ev), floor)
    if len(e) < 2:
        return 0
    gaps = np.diff(np.log(e))
    return int(np.argmax(gaps)) + 1


def null_space(D, Sw, tol=1e-4):
    """Orthonormal basis (p x k) of the (approximate) link null space; also returns the
    generalised eigenvalues rho_j = rms link gap^2 / within-cell variance along each eigenvector.
    tol: float -> keep rho_j <= tol^2;  'gap' -> keep the k smallest, k by the largest log-gap."""
    p = Sw.shape[0]
    if D is None or len(D) == 0:
        return np.eye(p), np.zeros(p)
    A = D.T @ D / len(D)
    ridge = 1e-4 * np.trace(Sw) / p          # directions without within-cell variance: rho = gap^2 / ridge
    ev, V = eigh(A, Sw + ridge * np.eye(p))
    if isinstance(tol, str):
        if tol != 'gap':
            raise ValueError(tol)
        keep = np.zeros(p, bool)
        keep[np.argsort(ev)[:gap_dimension(ev)]] = True
    else:
        keep = ev <= tol ** 2
    if not keep.any():
        return np.zeros((p, 0)), ev
    Q, _ = np.linalg.qr(V[:, keep])
    return Q, ev


def active_basis(N, Sw, rel=1e-8):
    """Columns of N (p x k) whose scores vary within cells: B = N Q, Q = eigvecs of N'S_wN with
    eigenvalue > rel * tr(S_w)/p. Directions without within-cell variation (city-level coordinates)
    do not enter any within-cell criterion; min-norm convention drops them."""
    if N.shape[1] == 0:
        return N
    M = N.T @ Sw @ N
    ev, Q = np.linalg.eigh(M)
    keep = ev > rel * np.trace(Sw) / Sw.shape[0]
    return N @ Q[:, keep][:, ::-1]


def city_basis(Sw, rel=1e-8):
    """Orthonormal basis (p x m) of the directions WITHOUT within-cell variation (city-level coordinates);
    the complement of active_basis(I, Sw). Within-cell criteria cannot see them; links can."""
    ev, Q = np.linalg.eigh(Sw)
    return Q[:, ev <= rel * np.trace(Sw) / Sw.shape[0]]


class Penalty:
    """P(theta) = theta'A theta / theta'S theta with A = D'D/L (declared training links).

    City-level directions C (no within-cell variation) are invisible to the ranking loss, so a
    penalised fit over the within-cell basis could never set them. They are profiled out:
    for each theta, v*(theta) = argmin_v (theta + Cv)'A(theta + Cv) = -(C'AC)^+ C'A theta, which gives
    A_eff = A - AC (C'AC)^+ C'A in the criterion and `complete(theta)` = theta + C v*(theta) for the
    final estimate. Directions in C that no link spans keep v = 0 (min-norm), as in the hard arm.
    """

    def __init__(self, D, Sw, C=None):
        A = D.T @ D / len(D)
        self.A_raw = A
        self.S = Sw
        self.p = Sw.shape[0]
        self.C, self.V = None, None
        if C is not None and C.shape[1] > 0:
            M = C.T @ A @ C
            ev, Q = np.linalg.eigh(M)
            keep = ev > 1e-10 * max(np.trace(A), 1e-300) / self.p      # absolute: temporal-only links give M = 0
            if keep.any():
                Minv = (Q[:, keep] / ev[keep]) @ Q[:, keep].T
                self.C = C
                self.V = -Minv @ C.T @ A                                  # v*(theta) = V theta
                A = A - A @ C @ Minv @ C.T @ A
                A = 0.5 * (A + A.T)
        self.A = A

    def complete(self, th):
        """theta + C v*(theta), renormalised to unit length (the scale convention of every arm)."""
        if self.C is None:
            return th
        full = th + self.C @ (self.V @ th)
        return full / np.linalg.norm(full)

    def value(self, th):
        return float(th @ self.A @ th / (th @ self.S @ th))

    def value_grad(self, th):
        a, s = self.A @ th, self.S @ th
        q = float(th @ s)
        P = float(th @ a) / q
        return P, 2.0 * (a - P * s) / q

    def hessian(self, th):
        a, s = self.A @ th, self.S @ th
        q = float(th @ s)
        P = float(th @ a) / q
        g = 2.0 * (a - P * s) / q
        return (2.0 / q) * (self.A - P * self.S - np.outer(s, g) - np.outer(g, s))


def tangent_basis(th):
    th = th / np.linalg.norm(th)
    Q, _ = np.linalg.qr(np.column_stack([th, np.eye(len(th))[:, :len(th) - 1]]))
    if Q[:, 0] @ th < 0:
        Q = -Q
    return Q[:, 1:]


def riemannian(th, g, H):
    """Tangent gradient and Riemannian Hessian on the unit sphere of a function with Euclidean
    gradient g and Hessian H at the unit vector th."""
    T = tangent_basis(th)
    return T.T @ g, T.T @ (H - (th @ g) * np.eye(len(th))) @ T


def ranking_derivs(cs, th, M):
    """Euclidean gradient/Hessian of R(th) = sum w log(1+exp(-M_k th'dz)) at unit th."""
    S = cs.Zs @ th
    _, _, gs = kn.rank_vg(cs, S, M)
    g = np.einsum('kia,ki->a', cs.Zs, gs)
    H = kn.rank_curv(cs, S, cs.Zs, M)
    return g, H


def lambda_for_omega(omega, trR, trS):
    """Closed form from the traces; omega = 1 returns inf (hard), omega = 0 returns 0."""
    omega = np.asarray(omega, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = np.where(omega >= 1.0, np.inf, omega * trR / ((1.0 - omega) * trS))
    return lam if lam.ndim else float(lam)


def omega_for_lambda(lam, trR, trS):
    return float(lam * trS / (trR + lam * trS))


def lambda_gradient_balance(cs, pen, th, M):
    """lambda_g = ||grad_T R|| / ||grad_T P|| at th (tangent gradients)."""
    gR, _ = ranking_derivs(cs, th, M)
    _, gP = pen.value_grad(th)
    T = tangent_basis(th)
    nP = np.linalg.norm(T.T @ gP)
    return float(np.linalg.norm(T.T @ gR) / nP) if nP > 0 else np.inf


def curvature_traces(cs, pen, th, M):
    """(tr H_R, tr H_S) on the tangent space at th."""
    gR, HR = ranking_derivs(cs, th, M)
    _, HTR = riemannian(th, gR, HR)
    _, gP = pen.value_grad(th)
    _, HTS = riemannian(th, gP, pen.hessian(th))
    return float(np.trace(HTR)), float(np.trace(HTS))
