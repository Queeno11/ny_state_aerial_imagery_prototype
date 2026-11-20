"""Correctly specified conditional ML with known simulation nuisance mechanisms.

Coefficients (including an intercept) are estimated from fitting labels only.
The structural basis, input rotation, inverse measurement map and error law/scale
are oracle information. This is a full observation likelihood, not pairwise BCE.
Parameter-independent measurement Jacobians can be omitted from optimization.
"""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.special import logsumexp
from scipy import stats


def structural_basis(h, truth, rotation):
    u = h @ rotation
    if truth == 'linear':
        z = u
    elif truth == 'quadratic':
        z = np.column_stack((u[:, 0], u[:, 1]*u[:, 2], u[:, 2]**2))
    elif truth == 'smooth':
        z = np.column_stack((np.sin(1.2*u[:, 0]), np.tanh(u[:, 1]), u[:, 1]*u[:, 2]))
    elif truth == 'regime':
        z = np.column_stack((u[:, 0], np.maximum(u[:, 1], 0), np.maximum(-u[:, 2], 0)))
    else:
        raise ValueError(truth)
    return np.column_stack((np.ones(len(h)), z))


def error_nll(residual, scale, family, severity):
    """Per-observation negative log density, matching design.unit_errors exactly."""
    z = residual / scale
    if family in ('gumbel', 'cell_scale', 'median_violation'):
        k = np.sqrt(6)/np.pi
        logp = stats.gumbel_r.logpdf(z/k + np.euler_gamma) - np.log(k)
    elif family in ('normal', 'feature_scale'):
        logp = stats.norm.logpdf(z)
    elif family == 'student':
        k = np.sqrt((severity-2)/severity)
        logp = stats.t.logpdf(z/k, severity) - np.log(k)
    elif family == 'mixture':
        k = np.sqrt(1+severity**2)
        logp = logsumexp(np.vstack((stats.norm.logpdf(k*z-severity),
                                   stats.norm.logpdf(k*z+severity))), axis=0) + np.log(k/2)
    elif family == 'chi2':
        k = np.sqrt(2*severity)
        logp = stats.chi2.logpdf(k*z+severity, severity) + np.log(k)
    else:
        raise ValueError(family)
    return -logp + np.log(scale)


def fit_coefficients(x, response, scale, family, severity, maxiter=1000):
    """Optimize unpenalized likelihood; retain and report numerical failures."""
    beta = np.linalg.lstsq(x, response, rcond=None)[0]
    if np.all(np.asarray(scale) == 0):
        return beta, dict(ml_success=True, ml_message='Noiseless least-squares limit',
                          ml_iterations=0, ml_train_nll=None, ml_gradient_inf=None)
    if np.any(np.asarray(scale) <= 0):
        raise ValueError('ML noise scales must be positive or all zero')
    if family == 'chi2' and severity < 2:
        # The density diverges at its moving boundary: no finite regular MLE.
        raise ValueError('Correct-specification chi2 ML requires df>=2; df<2 has unbounded likelihood')
    def objective(b):
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return float(np.mean(error_nll(response-x@b, scale, family, severity)))
    options = dict(maxiter=maxiter, ftol=1e-12, gtol=1e-7)
    constraints = ()
    method = 'L-BFGS-B'
    if family == 'chi2':
        # Strictly feasible start and explicit moving-support constraint.
        upper = response + np.asarray(scale)*np.sqrt(severity/2)
        margin = 1e-9
        beta[0] -= max(0., float(np.max(x@beta-upper))+1.)
        constraints = (LinearConstraint(x, -np.inf, upper-margin),)
        method = 'SLSQP'
        options = dict(maxiter=maxiter, ftol=1e-12)
    # Mixture likelihood can be multimodal. Deterministic starts use labels only.
    starts = [beta]
    if family == 'mixture' and severity:
        for sign in (-1, 1):
            other = beta.copy(); other[0] += sign*float(np.mean(scale))*severity/np.sqrt(1+severity**2)
            starts.append(other)
    fits = [minimize(objective, b, method=method, constraints=constraints, options=options) for b in starts]
    finite = [r for r in fits if np.isfinite(r.fun)]
    if not finite:
        raise FloatingPointError('ML produced no finite likelihood')
    fit = min(finite, key=lambda r: r.fun)
    return fit.x, dict(ml_success=bool(fit.success), ml_message=str(fit.message),
                      ml_iterations=int(fit.nit), ml_train_nll=float(fit.fun),
                      ml_gradient_inf=float(np.max(np.abs(fit.jac))))


def fit_ml(design, truth, scenario, dgp):
    from .design import inverse_measurement, BOUND
    tr = design['train']; start = time.perf_counter()
    # Invert observed W explicitly: never read the stored latent f or r targets.
    if scenario['positive_labels']:
        z = np.log((tr['w']-tr['a'])/tr['b'])/.6
        response = inverse_measurement(z, 0., 1., tr['kappa'], scenario['psi_delta'], scenario['psi_shape'])
    else:
        response = inverse_measurement(tr['w'], tr['a'], tr['b'], tr['kappa'], scenario['psi_delta'], scenario['psi_shape'])
    scale = np.full(len(response), design['noise_sd_effective'])
    family, severity = scenario['error'], scenario['error_severity']
    if family == 'cell_scale':
        scale *= np.exp(severity*design['cell_z'][tr['c'], tr['t']]-severity**2)
    if family in ('feature_scale', 'median_violation'):
        a = 2*severity*BOUND
        rms = np.sqrt(np.sinh(a)/a) if a else 1.
        scale *= np.exp(severity*tr['h'][:, 0])/rms
    x = structural_basis(tr['h'], truth, dgp.rotation)
    beta, stats_out = fit_coefficients(x, response, scale, family, severity)
    # X = [H, acquisition, group] @ mix. Known inverse map is oracle information.
    def predict(x):
        h = (x @ dgp.mix.T)[:, :3]
        return structural_basis(h, truth, dgp.rotation) @ beta
    stats_out.update(ml_coefficients=beta.tolist(), ml_seconds=time.perf_counter()-start,
                     ml_information='known structural basis, input map, measurement map, error law and scale; fitted coefficients',
                     ml_parameters=len(beta))
    return predict, stats_out
