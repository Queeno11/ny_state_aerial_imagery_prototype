"""Exact objectives and analytic score derivatives; no outcome truth inputs."""
from dataclasses import dataclass
import itertools
import numpy as np
from scipy.optimize import nnls, minimize_scalar
from scipy.special import expit
from .design import h_mean, inverse_map, conditional_mean, paired_logpdf


SLOPE_FLOOR = 1e-8
DISPERSION_FLOOR = 1e-6


def normalized(raw):
    norm = np.linalg.norm(raw)
    if norm < 1e-14:
        raise FloatingPointError('Degenerate score direction')
    return raw/norm, norm


def within_cov(z):
    center = z-z.mean(axis=1, keepdims=True)
    c, n, t, p = center.shape
    return np.einsum('cntp,cntq->pq', center, center)/(c*t*(n-1))


def fe(z, w, rho=0.):
    """Within GLS, including usual unrestricted observed residual variance."""
    xc, yc = z-z.mean(1, keepdims=True), w-w.mean(1, keepdims=True)
    precision = np.array([[1., -rho], [-rho, 1.]])/(1-rho*rho)
    gram = np.einsum('cntp,tu,cnuq->pq', xc, precision, xc)
    rhs = np.einsum('cntp,tu,cnu->p', xc, precision, yc)
    delta = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    scale = np.linalg.norm(delta)
    if scale <= SLOPE_FLOOR:
        raise FloatingPointError('Degenerate FE score')
    alpha = (w-np.einsum('cntp,p->cnt', z, delta)).mean(1)
    return delta/scale, dict(alpha=alpha, beta=np.full_like(alpha, scale), k=np.zeros_like(alpha)), gram


def working_gls(z, w):
    """Gaussian paired random-intercept ML, city-date intercepts fixed.

    Profiles regression coefficients and unrestricted observed variance. The
    nonnegative random-intercept component restricts rho to [0,.98].
    """
    def evaluate(rho, full=False):
        theta, heads, gram = fe(z, w, rho)
        pred = heads['alpha'][:, None, :]+heads['beta'][:, None, :]*(z@theta)
        e = w-pred
        quadratic = np.mean((e[..., 0]**2+e[..., 1]**2-2*rho*e[..., 0]*e[..., 1])/(1-rho*rho))
        observed_variance = quadratic/2
        value = np.log(observed_variance)+.5*np.log(1-rho*rho)+1
        return (theta, heads, gram, observed_variance) if full else value
    opt = minimize_scalar(evaluate, bounds=(0., .98), method='bounded', options={'xatol': 1e-10})
    candidates = [(evaluate(r), r) for r in (0., opt.x, .98)]
    _, rho = min(candidates)
    theta, heads, gram, variance = evaluate(rho, True)
    return theta, heads, dict(rho=rho, observed_variance=variance, objective=evaluate(rho),
                              boundary=bool(rho in (0., .98)), design_rank=int(np.linalg.matrix_rank(gram)))


class PairCriterion:
    """All unordered pairs, equally weighted within each city-date cell.

    Pair blocks bound memory without sampling or changing the objective.
    Only indices and observed signs are cached, not an O(C*n^2*p) design.
    """
    def __init__(self, z, w, block=65536):
        self.z, self.w, self.block = z, w, block
        n = z.shape[1]
        self.ii, self.jj = np.triu_indices(n, 1)
        self.signs = np.sign(w[:, self.jj, :]-w[:, self.ii, :])
        self.counts = np.count_nonzero(self.signs, axis=1)
        self.ties = int(np.size(self.signs)-np.count_nonzero(self.signs))
        if np.any(self.counts == 0):
            raise ValueError('An entire cell has tied outcomes')

    def value_grad(self, theta, kind, tau):
        s = self.z@theta
        score_grad = np.zeros_like(s)
        value, logtau_grad = 0., 0.
        cnum = len(s)
        for c in range(cnum):
            for t in range(2):
                weight = 1/(2*cnum*self.counts[c, t])
                for start in range(0, len(self.ii), self.block):
                    ii, jj = self.ii[start:start+self.block], self.jj[start:start+self.block]
                    y = self.signs[c, start:start+self.block, t]
                    margin = y*(s[c, jj, t]-s[c, ii, t])/tau
                    if kind == 'RN':
                        loss = np.logaddexp(0., -margin)
                        dm = -expit(-margin)
                    elif kind == 'BS':
                        loss = expit(-margin)
                        dm = -loss*(1-loss)
                    else:
                        raise ValueError(kind)
                    value += weight*np.sum(loss[y != 0])
                    logtau_grad -= weight*np.sum(dm*margin)
                    ds = weight*dm*y/tau
                    np.add.at(score_grad[c, :, t], jj, ds)
                    np.add.at(score_grad[c, :, t], ii, -ds)
        return float(value), np.einsum('cntp,cnt->p', self.z, score_grad), float(logtau_grad)


def stability(theta, qt, qc, vw, lt, lc, ratio=True):
    numerator = lt*qt+lc*qc
    v = float(theta@vw@theta)
    if v < DISPERSION_FLOOR:
        raise FloatingPointError('Within-cell dispersion below feasibility floor')
    s = float(theta@numerator@theta)
    if ratio:
        return s/v, 2*(numerator@theta-(s/v)*vw@theta)/v
    return s, 2*numerator@theta


def cardinal_error(pred, w, q, vstar):
    e = pred-w
    a = e.mean(-1)
    d = e[..., 1]-e[..., 0]
    value = np.mean((1-q)*a*a+(q/4)*d*d)/vstar
    derivative = np.stack([(1-q)*a-q*d/2, (1-q)*a+q*d/2], -1)/(e.shape[0]*e.shape[1]*vstar)
    return float(value), derivative


def profile_heads(s, w, family, q, law, rho, sigma=1., order=48, fixed_scale=None):
    """Exact constrained small least-squares heads, paired dates coupled by q.

    Intercepts profile out at each date's residual mean. For M3 use beta=eps+u+v,
    D=2v with u,v>=0; this enforces beta>=eps and 0<=k<=2, including boundaries.
    Returns the envelope derivative with respect to the frozen score.
    """
    cnum, n, _ = s.shape
    xc, yc = s-s.mean(1, keepdims=True), w-w.mean(1, keepdims=True)
    r = 1-2*q
    precision = np.array([[1., r], [r, 1.]])
    h, hp = (h_mean(s, law, rho, sigma, order) if family == 'M3'
             else (np.zeros_like(s), np.zeros_like(s)))
    if family == 'M1':
        beta = (fixed_scale if fixed_scale is not None else
                max(SLOPE_FLOOR, np.einsum('cnt,tu,cnu->', xc, precision, yc)/
                    np.einsum('cnt,tu,cnu->', xc, precision, xc)))
        beta = np.full((cnum, 2), beta)
        kval = np.zeros_like(beta)
    else:
        beta, kval = np.empty((cnum, 2)), np.zeros((cnum, 2))
        hc = h-h.mean(1, keepdims=True)
        for c in range(cnum):
            # Row order is (tract,date); block diagonal cell regressors.
            columns = 2 if family == 'M2' else 4
            design = np.zeros((n, 2, columns))
            for t in range(2):
                if family == 'M2':
                    design[:, t, t] = xc[c, :, t]
                else:
                    design[:, t, 2*t] = xc[c, :, t]
                    design[:, t, 2*t+1] = xc[c, :, t]+2*hc[c, :, t]
            # Cholesky precision=L L'; left multiply by L' along date axis.
            whiten = np.linalg.cholesky(precision).T
            matrix = np.einsum('tu,nup->ntp', whiten, design).reshape(-1, columns)
            response = ((yc[c]-SLOPE_FLOOR*xc[c])@whiten.T).ravel()
            coef = np.linalg.lstsq(matrix, response, rcond=None)[0]
            if np.any(coef < 0):
                coef, _ = nnls(matrix, response, maxiter=1000, atol=1e-12)
            if family == 'M2':
                beta[c] = SLOPE_FLOOR+coef
            else:
                uv = coef.reshape(2, 2)
                beta[c] = SLOPE_FLOOR+uv.sum(-1)
                kval[c] = 2*uv[:, 1]/beta[c]
    alpha = (w-beta[:, None, :]*(s+kval[:, None, :]*h)).mean(1)
    pred = alpha[:, None, :]+beta[:, None, :]*(s+kval[:, None, :]*h)
    derivative = beta[:, None, :]*(1+kval[:, None, :]*hp)
    heads = dict(alpha=alpha, beta=beta, k=kval)
    return heads, pred, derivative


def mean_prediction(s, heads, law, rho, sigma=1., order=48):
    return conditional_mean(s, heads['alpha'][:, None, :], heads['beta'][:, None, :],
                            heads['k'][:, None, :], law, rho, sigma, order)


def likelihood_value(s, w, heads, law, rho, sigma=1., gradient=True):
    """Averaged full paired NLL INCLUDING all transformation Jacobians."""
    alpha = heads['alpha'][:, None, :]
    beta = heads['beta'][:, None, :]
    k = heads['k'][:, None, :]
    r = inverse_map(w, alpha, beta, k)
    th = np.tanh(r)
    sech2 = 1-th*th
    derivative = 1+k*sech2
    lp, ge = paired_logpdf(r-s, law, rho, sigma)
    jacobian_log = np.log(beta*derivative)
    value = float(np.mean(-lp+jacobian_log.sum(-1)))
    if not gradient:
        return value
    size = s.shape[0]*s.shape[1]
    # d/d r[-log p + log Psi'].
    gr = -ge-2*k*sech2*th/derivative
    gs = ge/size
    ga = (-gr/(beta*derivative)).sum(1)/size
    # Gradients use log beta, k (not logit k).
    gb = (-gr*(r+k*th)/derivative+1).sum(1)/size
    gk = (-gr*th/derivative+sech2/derivative).sum(1)/size
    return value, gs, ga, gb, gk
