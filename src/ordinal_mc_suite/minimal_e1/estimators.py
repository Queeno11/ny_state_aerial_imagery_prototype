"""Estimators. Inputs are only observed features z (possibly projected on the
link null space) and outcomes W. Every function returns (theta, extras, info)
with theta a unit vector in the coordinates it was given."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .dgp import sinh_inverse, sinh_deriv
from .heads import fit_head, profile, null_value, weight_matrix, center
from .losses import newton_ranknet

LOG2 = np.log(2.0)


def unit(v):
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm < 1e-14:
        raise FloatingPointError('Degenerate score direction')
    return v / norm


def _tangent(theta, g):
    return g - theta * (theta @ g)


def _sphere_wrap(fun_theta):
    """Objective on the unit sphere written in unconstrained u, theta=u/|u|."""
    def wrapped(u):
        norm = np.linalg.norm(u)
        theta = u / norm
        val, g = fun_theta(theta)
        return val, _tangent(theta, g) / norm
    return wrapped


def _lbfgs(fun, x0, gtol, max_iter):
    return minimize(fun, x0, jac=True, method='L-BFGS-B',
                    options=dict(maxiter=max_iter, gtol=gtol, ftol=1e-16, maxcor=30))


GRAD_OK = 1e-6


def _converged(res, grad_norm):
    """L-BFGS-B often ends with a line-search warning at machine precision;
    the projected gradient is the convergence criterion that counts."""
    return bool(res.success or (np.isfinite(grad_norm) and grad_norm <= GRAD_OK))


def _best_of(results):
    """Keep the best converged objective; report spread across starts."""
    finite = [r for r in results if np.isfinite(r[0])]
    if not finite:
        raise FloatingPointError('No finite start')
    finite.sort(key=lambda r: r[0])
    spread = finite[-1][0] - finite[0][0]
    return finite[0], float(spread)


# ---------------------------------------------------------------------------
# Closed-form within estimators (FE, cardinal map-a head, GLS)
# ---------------------------------------------------------------------------
def weighted_within(z, W, P):
    """argmin_delta sum_ci (zc_i delta - Wc_i)' P (zc_i delta - Wc_i), cell
    intercepts profiled. P = I/4 is FE OLS; P = R(rho)^{-1} is GLS."""
    zc, Wc = center(z), center(W)
    gram = np.einsum('cntp,tu,cnuq->pq', zc, P, zc)
    rhs = np.einsum('cntp,tu,cnu->p', zc, P, Wc)
    delta = np.linalg.solve(gram, rhs)
    return delta, gram


def fixed_effects(z, W):
    delta, gram = weighted_within(z, W, weight_matrix(0.5))
    b = float(np.linalg.norm(delta))
    return unit(delta), dict(b=b), dict(success=True, design_rank=int(np.linalg.matrix_rank(gram)),
                                         n_starts=1, spread=0.0)


def fgls(z, W, rho_bounds=(0.0, 0.98)):
    """Gaussian ML with cell fixed intercepts, tract random intercept and free
    observed variance = feasible GLS (exact ML for map a under the DGP)."""
    def fit(rho):
        Rinv = np.array([[1.0, -rho], [-rho, 1.0]]) / (1 - rho * rho)
        delta, _ = weighted_within(z, W, Rinv)
        e = center(W) - center(z) @ delta
        quad = np.einsum('cnt,tu,cnu->cn', e, Rinv, e)
        omega2 = float(quad.mean() / 2)
        return np.log(omega2) + 0.5 * np.log(1 - rho * rho) + 1.0, delta, omega2

    opt = minimize_scalar(lambda r: fit(r)[0], bounds=rho_bounds, method='bounded',
                          options=dict(xatol=1e-10))
    candidates = sorted([(fit(r)[0], r) for r in (rho_bounds[0], opt.x, rho_bounds[1])])
    value, rho = candidates[0]
    _, delta, omega2 = fit(rho)
    return unit(delta), dict(b=float(np.linalg.norm(delta)), rho_hat=float(rho),
                             omega=float(np.sqrt(omega2))), \
        dict(success=True, objective=float(value), n_starts=1, spread=0.0,
             boundary=bool(rho in rho_bounds))


# ---------------------------------------------------------------------------
# Cardinal C_q with a profiled correct-family head
# ---------------------------------------------------------------------------
def cardinal_objective(z, W, family, q, kappa, vstar):
    def fun(theta):
        s = z @ theta
        val, grad_s, _ = profile(s, W, family, q, kappa, vstar)
        return val, np.einsum('cntp,cnt->p', z, grad_s)
    return fun


def cardinal(z, W, family, q, kappa, vstar, starts, gtol=1e-9, max_iter=3000):
    if family == 'a':
        delta, gram = weighted_within(z, W, weight_matrix(q))
        theta = unit(delta)
        val, _ = cardinal_objective(z, W, 'a', q, kappa, vstar)(theta)
        return theta, dict(b=float(np.linalg.norm(delta))), \
            dict(success=True, objective=val, n_starts=1, spread=0.0, closed_form=True)
    fun = _sphere_wrap(cardinal_objective(z, W, family, q, kappa, vstar))
    results = []
    for start in starts:
        res = _lbfgs(fun, unit(start), gtol, max_iter)
        theta = unit(res.x)
        results.append((float(res.fun), theta, res))
    (val, theta, res), spread = _best_of(results)
    head = fit_head(z @ theta, W, family, q, kappa)
    g = cardinal_objective(z, W, family, q, kappa, vstar)(theta)[1]
    gn = float(np.max(np.abs(_tangent(theta, g))))
    return theta, dict(head_boundary=head.boundary), \
        dict(success=_converged(res, gn), objective=val, n_starts=len(starts), spread=spread,
             grad_norm=gn, iterations=int(res.nit), message=str(res.message))


# ---------------------------------------------------------------------------
# Ordinal and joint
# ---------------------------------------------------------------------------
def ordinal(pairs, tol=1e-11):
    u, info = newton_ranknet(pairs, tol=tol)
    norm = np.linalg.norm(u)
    info = dict(info, n_starts=1, spread=0.0)
    return unit(u), dict(tau=float(1 / norm) if norm > 0 else np.inf, u=u), info


def joint(z, W, pairs, family, q, kappa, vstar, w, starts_u, gtol=1e-10, max_iter=3000):
    """(1-w) C_q(theta)/C_q(0) + w P_RN(u)/log 2, theta = u/|u|, tau = 1/|u|."""
    c0 = null_value(W, q, vstar)
    card = cardinal_objective(z, W, family, q, kappa, vstar)

    def fun(u):
        norm = np.linalg.norm(u)
        theta = u / norm
        cv, cg = card(theta)
        pv, pg = pairs.value_grad(u)
        val = (1 - w) * cv / c0 + w * pv / LOG2
        grad = (1 - w) / c0 * _tangent(theta, cg) / norm + w / LOG2 * pg
        return val, grad

    results = []
    for u0 in starts_u:
        res = _lbfgs(fun, u0, gtol, max_iter)
        results.append((float(res.fun), res.x, res))
    (val, u, res), spread = _best_of(results)
    gn = float(np.max(np.abs(fun(u)[1])))
    return unit(u), dict(tau=float(1 / np.linalg.norm(u))), \
        dict(success=_converged(res, gn), objective=val, n_starts=len(starts_u), spread=spread,
             grad_norm=gn, iterations=int(res.nit), message=str(res.message))


# ---------------------------------------------------------------------------
# Maximum likelihood for maps b and c (map a is FGLS above)
# ---------------------------------------------------------------------------
class MLModel:
    """Exact paired Gaussian likelihood of W including the map Jacobian.

    Parameters: u (score direction, theta = u/|u|), alpha_ct, log beta_ct,
    [map c: xi_c with lambda_c = bound * tanh(xi_c)], log sigma, zeta
    (rho = tanh zeta). kappa is known. Values are averaged over tracts."""

    def __init__(self, z, W, family, kappa, lam_bound=0.95):
        self.z, self.W, self.family = z, W, family
        self.kappa, self.lam_bound = kappa, lam_bound
        self.C, self.n, _, self.k = z.shape

    def sizes(self):
        sizes = [self.k, 2 * self.C, 2 * self.C]
        if self.family == 'c':
            sizes.append(self.C)
        return sizes + [1, 1]

    def unpack(self, vec):
        parts = np.split(vec, np.cumsum(self.sizes())[:-1])
        out = dict(u=parts[0], alpha=parts[1].reshape(self.C, 2),
                   logbeta=parts[2].reshape(self.C, 2))
        if self.family == 'c':
            out['xi'] = parts[3]
        out['logsigma'], out['zeta'] = float(parts[-2][0]), float(parts[-1][0])
        return out

    def pack(self, u, alpha, logbeta, logsigma, zeta, xi=None):
        parts = [u, alpha.ravel(), logbeta.ravel()]
        if self.family == 'c':
            parts.append(xi)
        return np.concatenate(parts + [[logsigma], [zeta]])

    def latent(self, prm):
        alpha = prm['alpha'][:, None, :]
        beta = np.exp(prm['logbeta'])[:, None, :]
        y = (self.W - alpha) / beta
        if self.family == 'b':
            return y, y, None, beta
        lam = self.lam_bound * np.tanh(prm['xi'])[:, None, None]
        r = sinh_inverse(y, lam, self.kappa)
        return r, y, lam, beta

    def value_grad(self, vec):
        prm = self.unpack(vec)
        u = prm['u']
        norm = np.linalg.norm(u)
        theta = u / norm
        s = self.z @ theta
        r, y, lam, beta = self.latent(prm)
        sigma, rho = np.exp(prm['logsigma']), np.tanh(prm['zeta'])
        one = 1 - rho * rho
        K = 1 / (sigma * sigma * one)
        e = r - s
        e0, e1 = e[..., 0], e[..., 1]
        Q = e0 * e0 + e1 * e1 - 2 * rho * e0 * e1
        N = self.C * self.n
        kap = self.kappa
        if self.family == 'b':
            logjac = np.broadcast_to(prm['logbeta'][:, None, :], e.shape)
            dlj_dr = 0.0
        else:
            sp = sinh_deriv(r, lam, kap)
            logjac = prm['logbeta'][:, None, :] + np.log(sp)
            dlj_dr = kap * kap * y / sp
        nll = (np.log(2 * np.pi) + 2 * prm['logsigma'] + 0.5 * np.log(one)
               + 0.5 * K * Q + logjac.sum(-1))
        value = float(nll.mean())
        de = K * np.stack([e0 - rho * e1, e1 - rho * e0], axis=-1)
        gr = de + dlj_dr
        g_theta = -np.einsum('cntp,cnt->p', self.z, de) / N
        g_u = _tangent(theta, g_theta) / norm
        if self.family == 'b':
            g_alpha = (gr * (-1 / beta)).sum(1) / N
            g_logbeta = (gr * (-r) + 1).sum(1) / N
            g_xi = None
        else:
            g_alpha = (gr * (-1 / (beta * sp))).sum(1) / N
            g_logbeta = (gr * (-y / sp) + 1).sum(1) / N
            g_lam = (gr * (-np.cosh(kap * r) / (kap * sp)) + np.sinh(kap * r) / sp).sum((1, 2)) / N
            g_xi = g_lam * self.lam_bound * (1 - np.tanh(prm['xi']) ** 2)
        g_logsigma = float(np.mean(2 - K * Q))
        g_rho = np.mean(-rho / one + (-e0 * e1 * one + rho * Q) / (sigma * sigma * one * one))
        g_zeta = float(g_rho * one)
        grad = self.pack(g_u, g_alpha, g_logbeta, g_logsigma, g_zeta, g_xi)
        return value, grad

    def start(self, theta0):
        s0 = self.z @ theta0
        sc, Wc = center(s0), center(self.W)
        slope = np.clip((sc * Wc).sum(1) / (sc * sc).sum(1), 0.05, None)
        alpha = self.W.mean(1) - slope * s0.mean(1)
        xi = np.zeros(self.C) if self.family == 'c' else None
        prm = dict(alpha=alpha, logbeta=np.log(slope), xi=xi)
        r, *_ = self.latent(prm)
        e = r - s0
        sigma = float(np.std(e))
        rho = float(np.clip(np.corrcoef(e[..., 0].ravel(), e[..., 1].ravel())[0, 1], -0.9, 0.9))
        return self.pack(theta0, alpha, np.log(slope), np.log(sigma), np.arctanh(rho), xi)


def maximum_likelihood(z, W, family, kappa, starts, lam_bound=0.95, gtol=1e-8, max_iter=3000):
    model = MLModel(z, W, family, kappa, lam_bound)
    results = []
    for theta0 in starts:
        x0 = model.start(unit(theta0))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            res = _lbfgs(model.value_grad, x0, gtol, max_iter)
        results.append((float(res.fun) if np.isfinite(res.fun) else np.inf, res.x, res))
    (val, x, res), spread = _best_of(results)
    prm = model.unpack(x)
    _, grad = model.value_grad(x)
    extras = dict(rho_hat=float(np.tanh(prm['zeta'])), sigma_hat=float(np.exp(prm['logsigma'])),
                  beta_hat=np.exp(prm['logbeta']).tolist())
    if family == 'c':
        lam = lam_bound * np.tanh(prm['xi'])
        extras['lambda_hat'] = lam.tolist()
        extras['lambda_at_bound'] = int(np.sum(np.abs(lam) > 0.999 * lam_bound))
    gn = float(np.max(np.abs(grad)))
    return unit(prm['u']), extras, dict(success=_converged(res, gn), objective=val,
                                        n_starts=len(starts), spread=spread, grad_norm=gn,
                                        iterations=int(res.nit), message=str(res.message))


# ---------------------------------------------------------------------------
# Practitioner random-slope linear mixed model (map b)
# ---------------------------------------------------------------------------
class MixedModel:
    """W_cit = alpha_ct + z_cit'(delta + g_ct) + u_ci + e_cit, Gaussian ML.

    Fixed: city-date intercepts (kept fixed because they correlate with X) and
    a common slope vector delta. Random: slope vector g_ct ~ N(0, Omega) per
    city-date, tract intercept u_ci ~ N(0, s_u^2) shared by both dates, and
    e ~ N(0, s_e^2). Omega = L L' with L lower-triangular ('unstructured') or
    diagonal. Fixed effects are profiled by GLS; ML (not REML)."""

    def __init__(self, z, W, spec='unstructured'):
        self.z, self.W, self.spec = z, W, spec
        self.C, self.n, _, self.p = z.shape
        C, n, p = self.C, self.n, self.p
        zt = z.transpose(0, 2, 1, 3)                                  # (C,2,n,p)
        dummies = np.zeros((C, 2, n, 2))
        dummies[:, 0, :, 0] = 1.0
        dummies[:, 1, :, 1] = 1.0
        self.Q = np.concatenate([dummies, zt, W.transpose(0, 2, 1)[..., None]], axis=-1)
        self.zt = zt
        self.tril = np.tril_indices(p)

    def n_params(self):
        return 2 + (len(self.tril[0]) if self.spec == 'unstructured' else self.p)

    def chol(self, theta):
        L = np.zeros((self.p, self.p))
        if self.spec == 'unstructured':
            L[self.tril] = theta[2:]
        else:
            L[np.diag_indices(self.p)] = theta[2:]
        return L

    def _ainv(self, M, a, b):
        """Apply A^{-1}, A = per-tract [[a, b], [b, a]] across dates; M (C,2,n,k)."""
        return (a * M - b * M[:, ::-1]) / (a * a - b * b)

    def profiled(self, theta, return_delta=False):
        se2, su2 = np.exp(2 * theta[0]), np.exp(2 * theta[1])
        a, b = se2 + su2, su2
        L = self.chol(theta)
        C, n, p = self.C, self.n, self.p
        ZL = np.zeros((C, 2, n, 2 * p))
        ZL[:, 0, :, :p] = self.zt[:, 0] @ L
        ZL[:, 1, :, p:] = self.zt[:, 1] @ L
        AinvZL = self._ainv(ZL, a, b)
        AinvQ = self._ainv(self.Q, a, b)
        M = np.eye(2 * p) + np.einsum('ctnk,ctnl->ckl', ZL, AinvZL)
        cross = np.einsum('ctnk,ctnl->ckl', ZL, AinvQ)               # (C,2p,q)
        gram = (np.einsum('ctnk,ctnl->ckl', self.Q, AinvQ)
                - np.einsum('ckq,ckl->cql', cross, np.linalg.solve(M, cross)))
        sign, logdet_m = np.linalg.slogdet(M)
        logdet = n * np.log(a * a - b * b) + logdet_m
        d = slice(0, 2)
        rest = slice(2, None)
        schur = gram[:, rest, rest] - np.einsum(
            'cid,cde->cie', gram[:, rest, d], np.linalg.solve(gram[:, d, d], gram[:, d, rest]))
        tot = schur.sum(0)
        delta = np.linalg.solve(tot[:p, :p], tot[:p, p])
        quad = tot[p, p] - tot[:p, p] @ delta
        value = 0.5 * (logdet.sum() + quad) / (C * n)
        if return_delta:
            return value, delta
        return value

    def start(self, omega2, slope_sd):
        theta = np.zeros(self.n_params())
        theta[0] = theta[1] = 0.5 * np.log(omega2 / 2)
        L = np.eye(self.p) * slope_sd
        theta[2:] = L[self.tril] if self.spec == 'unstructured' else np.diag(L)
        return theta


def mixed_model(z, W, spec='unstructured', max_iter=2000):
    delta_fe, _ = weighted_within(z, W, weight_matrix(0.5))
    e = center(W) - center(z) @ delta_fe
    omega2 = float(np.mean(e * e))
    tried = []
    for current in ([spec, 'diagonal'] if spec == 'unstructured' else [spec]):
        model = MixedModel(z, W, current)
        results = []
        for slope_sd in (0.05, 0.3):
            x0 = model.start(omega2, slope_sd * np.linalg.norm(delta_fe))
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                res = minimize(lambda t: model.profiled(t), x0, method='L-BFGS-B',
                               options=dict(maxiter=max_iter, ftol=1e-14, gtol=1e-8))
            val = float(res.fun) if np.isfinite(res.fun) else np.inf
            results.append((val, res.x, res))
        (val, x, res), spread = _best_of(results)
        tried.append(current)
        ok = bool(res.success) and np.isfinite(val)
        if ok or current == 'diagonal':
            _, delta = model.profiled(x, return_delta=True)
            L = model.chol(x)
            omega = L @ L.T
            return unit(delta), dict(b=float(np.linalg.norm(delta)), spec=current,
                                     sigma_e=float(np.exp(x[0])), sigma_u=float(np.exp(x[1])),
                                     omega_eigen=np.linalg.eigvalsh(omega).tolist()), \
                dict(success=ok, objective=val, n_starts=len(results), spread=spread,
                     iterations=int(res.nit), specs_tried=tried)
