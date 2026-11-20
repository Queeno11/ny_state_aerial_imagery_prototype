"""Parametric/semiparametric second-step heads: per-cell least-squares cubic B-splines of W on the index.

Arrays follow the package convention: (C cities, n tracts, 2 dates); a cell = (city, date).
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear

DEGREE = 3


def _knots(s, df):
    """Clamped cubic knot vector with df-4 interior knots at quantiles of s (duplicates dropped)."""
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        hi = lo + 1.0
    k = max(df - DEGREE - 1, 0)
    inner = np.quantile(s, np.linspace(0, 1, k + 2)[1:-1]) if k > 0 else np.empty(0)
    inner = np.unique(inner[(inner > lo) & (inner < hi)])
    return np.r_[[lo] * (DEGREE + 1), inner, [hi] * (DEGREE + 1)]


def _design(x, t):
    """Cubic B-spline design matrix at x (clipped into the knot range)."""
    xc = np.clip(x, t[DEGREE], t[-DEGREE - 1])
    return BSpline.design_matrix(xc, t, DEGREE).toarray()


def fit_cell(s, w, df, monotone=False):
    """Fit one cell. Returns a model dict (internally s is standardised)."""
    mu, sd = float(s.mean()), float(s.std())
    sd = sd if sd > 0 else 1.0
    z = (s - mu) / sd
    if df <= 2:
        if monotone:
            zc = z - z.mean()
            b = max(float(zc @ (w - w.mean())) / max(float(zc @ zc), 1e-300), 0.0)
            coef = np.array([w.mean() - b * z.mean(), b])
        else:
            coef = np.linalg.lstsq(np.c_[np.ones_like(z), z], w, rcond=None)[0]
        return dict(kind='lin', coef=coef, mu=mu, sd=sd)
    t = _knots(z, df)
    B = _design(z, t)
    if monotone:
        m = B.shape[1]
        L = np.tril(np.ones((m, m)))                     # coefficients = L @ (c0, increments)
        lb = np.r_[-np.inf, np.zeros(m - 1)]
        u = lsq_linear(B @ L, w, bounds=(lb, np.full(m, np.inf)), method='bvls').x
        coef = L @ u
    else:
        coef = np.linalg.lstsq(B, w, rcond=None)[0]
    return dict(kind='bs', coef=coef, t=t, mu=mu, sd=sd, spl=BSpline(t, coef, DEGREE, extrapolate=False))


PS_NSEG = 20                        # P-spline: equally spaced segments over the cell's index range
PS_ORDER = 2                        # difference penalty order (2: shrinks towards a straight line)
PS_LAMBDAS = tuple(10.0 ** np.arange(-3.0, 4.01, 0.5))   # penalty grid (scaled by n / PS_NSEG, see fit_cell_ps)


def _knots_equal(z, nseg):
    """Uniform cubic knot vector (Eilers-Marx): nseg equal segments over [min z, max z], extended by DEGREE
    segments on each side, so the basis is translation-invariant and a 2nd-difference penalty's null space is
    exactly the straight lines. t[DEGREE] = min z, t[-DEGREE-1] = max z (the fitted range)."""
    lo, hi = float(z.min()), float(z.max())
    if hi <= lo:
        hi = lo + 1.0
    dx = (hi - lo) / nseg
    return lo + dx * np.arange(-DEGREE, nseg + DEGREE + 1)


def fit_cell_ps(s, w, lam, nseg=PS_NSEG, monotone=False, order=PS_ORDER):
    """Eilers-Marx P-spline for one cell: cubic B-splines on nseg equal segments, penalty
    lam * (n / nseg) * ||D_order coef||^2 (the n / nseg factor keeps lam roughly sample-size free, since each
    basis function covers ~n / nseg tracts). Monotone: coefficient increments >= 0 (BVLS on the augmented
    system). Returns a model dict usable by predict_cell."""
    mu, sd = float(s.mean()), float(s.std())
    sd = sd if sd > 0 else 1.0
    z = (s - mu) / sd
    t = _knots_equal(z, nseg)
    B = _design(z, t)
    m = B.shape[1]
    Dm = np.diff(np.eye(m), order, axis=0)
    A = np.vstack([B, np.sqrt(lam * len(z) / nseg) * Dm])
    y = np.r_[w, np.zeros(Dm.shape[0])]
    if monotone:
        L = np.tril(np.ones((m, m)))
        lb = np.r_[-np.inf, np.zeros(m - 1)]
        coef = L @ lsq_linear(A @ L, y, bounds=(lb, np.full(m, np.inf)), method='bvls').x
    else:
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return dict(kind='bs', coef=coef, t=t, mu=mu, sd=sd, spl=BSpline(t, coef, DEGREE, extrapolate=False))


L1_LAMBDAS = tuple(10.0 ** np.arange(-3.0, 1.51, 0.5))   # lasso P-spline penalty grid (scaled, see fit_cell_l1)
L1_ITERS = 30
L1_EPS = 1e-6


def fit_cell_l1(s, w, lam, nseg=PS_NSEG, monotone=False, order=PS_ORDER, iters=L1_ITERS):
    """Lasso P-spline (trend-filtering flavour) for one cell: cubic B-splines on nseg equal segments, penalty
    lam * (n / nseg) * sd(w) * sum |D_order coef| (the sd(w) factor makes lam unit-free). Most differences are
    driven to ~0, so the curve bends only where the data ask. Solved by iteratively reweighted ridge
    (|d| ~ d^2 / |d_prev|), keeping the monotone bounds; starts from the ridge fit with the same lam."""
    mu, sd = float(s.mean()), float(s.std())
    sd = sd if sd > 0 else 1.0
    z = (s - mu) / sd
    t = _knots_equal(z, nseg)
    B = _design(z, t)
    m = B.shape[1]
    Dm = np.diff(np.eye(m), order, axis=0)
    scale = lam * len(z) / nseg * max(float(np.std(w)), 1e-12)
    L = np.tril(np.ones((m, m))) if monotone else np.eye(m)
    lb = np.r_[-np.inf, np.zeros(m - 1)] if monotone else np.full(m, -np.inf)
    y = np.r_[w, np.zeros(Dm.shape[0])]
    wts = np.full(Dm.shape[0], 1.0 / max(float(np.std(w)), 1e-12))   # first pass ~ ridge
    coef = None
    for _ in range(iters):
        A = np.vstack([B, np.sqrt(scale * wts)[:, None] * Dm])
        if monotone:
            u = lsq_linear(A @ L, y, bounds=(lb, np.full(m, np.inf)), method='bvls').x
            new = L @ u
        else:
            new = np.linalg.lstsq(A, y, rcond=None)[0]
        if coef is not None and np.max(np.abs(new - coef)) < 1e-7 * (1 + np.max(np.abs(coef))):
            coef = new
            break
        coef = new
        wts = 1.0 / np.maximum(np.abs(Dm @ coef), L1_EPS * max(float(np.std(w)), 1e-12))
    return dict(kind='bs', coef=coef, t=t, mu=mu, sd=sd, spl=BSpline(t, coef, DEGREE, extrapolate=False))


def fit_cell_median(s, w, df, monotone=False):
    """Median (least absolute deviation) cubic regression spline for one cell, same basis as fit_cell
    (quantile knots, df). Linear programme: min sum |w - B c|, monotone via non-negative coefficient
    increments. NOTE: estimates the conditional MEDIAN of W, not the mean."""
    from scipy.optimize import linprog
    mu, sd = float(s.mean()), float(s.std())
    sd = sd if sd > 0 else 1.0
    z = (s - mu) / sd
    n = len(z)
    if df <= 2:
        X = np.c_[np.ones(n), z]
        t = None
    else:
        t = _knots(z, df)
        X = _design(z, t)
    m = X.shape[1]
    L = np.tril(np.ones((m, m))) if (monotone and t is not None) else np.eye(m)
    XL = X @ L
    # variables: u (m), e_plus (n), e_minus (n);  XL u + e+ - e- = w
    cobj = np.r_[np.zeros(m), np.ones(2 * n)]
    A_eq = np.hstack([XL, np.eye(n), -np.eye(n)])
    if monotone:
        lb_u = [None] + [0.0] * (m - 1)
    else:
        lb_u = [None] * m
    bounds = [(b, None) for b in lb_u] + [(0.0, None)] * (2 * n)
    res = linprog(cobj, A_eq=A_eq, b_eq=w, bounds=bounds, method='highs')
    coef = L @ res.x[:m]
    if t is None:
        return dict(kind='lin', coef=coef, mu=mu, sd=sd)
    return dict(kind='bs', coef=coef, t=t, mu=mu, sd=sd, spl=BSpline(t, coef, DEGREE, extrapolate=False))


def make_fitter(kind='bs', monotone=False):
    """(s, w, g) -> model for a tuning value g: df of the regression spline (kind 'bs'), the penalty of the
    ridge P-spline ('ps') or of the lasso P-spline ('l1'), or df of the median spline ('med')."""
    if kind == 'bs':
        return lambda s, w, g: fit_cell(s, w, int(g), monotone)
    if kind == 'ps':
        return lambda s, w, g: fit_cell_ps(s, w, float(g), monotone=monotone)
    if kind == 'l1':
        return lambda s, w, g: fit_cell_l1(s, w, float(g), monotone=monotone)
    if kind == 'med':
        return lambda s, w, g: fit_cell_median(s, w, int(g), monotone)
    raise ValueError(kind)


DF_KINDS = ('bs', 'med')            # tuning value = df (ties -> fewer df); otherwise a penalty (ties -> larger)


def default_grid(kind, grid=None):
    if grid is not None:
        return tuple(grid)
    return {'bs': (2, 4, 5, 6, 8), 'med': (2, 4, 5, 6, 8), 'ps': PS_LAMBDAS, 'l1': L1_LAMBDAS}[kind]


def predict_cell(model, s_new):
    z = (np.asarray(s_new, float) - model['mu']) / model['sd']
    if model['kind'] == 'lin':
        return model['coef'][0] + model['coef'][1] * z
    t, spl = model['t'], model['spl']
    lo, hi = t[DEGREE], t[-DEGREE - 1]
    zc = np.clip(z, lo, hi)
    out = spl(zc)
    d = spl.derivative()
    out = out + np.where(z < lo, d(lo), 0.0) * (z - lo) + np.where(z > hi, d(hi), 0.0) * (z - hi)
    return out


def bspline_predict(s_tr, W_tr, s_te, df, monotone=False, kind='bs'):
    """Per-cell cubic B-spline head (kind 'bs': least squares with df; 'ps': P-spline with penalty df),
    linear extrapolation outside the training range."""
    C, _, T = s_tr.shape
    out = np.empty(s_te.shape)
    fitter = make_fitter(kind, monotone)
    for c in range(C):
        for t in range(T):
            m = fitter(s_tr[c, :, t], W_tr[c, :, t], df)
            out[c, :, t] = predict_cell(m, s_te[c, :, t])
    return out


def cv_select_df(s_tr, W_tr, grid=None, K=5, monotone=False, seed=0, kind='bs'):
    """K-fold CV inside every cell (same folds for all df); total CV squared error over ALL cells.
    Returns (best_df, table) with table = {df: total CV SSE}."""
    C, n, T = s_tr.shape
    grid = default_grid(kind, grid)
    fitter = make_fitter(kind, monotone)
    cast = int if kind in DF_KINDS else float
    sse = {cast(g): 0.0 for g in grid}
    for c in range(C):
        for t in range(T):
            rg = np.random.default_rng([int(seed), c, t])
            fold = rg.permutation(n) % K
            s, w = s_tr[c, :, t], W_tr[c, :, t]
            for k in range(K):
                te = fold == k
                for g in sse:
                    m = fitter(s[~te], w[~te], g)
                    sse[g] += float(np.sum((w[te] - predict_cell(m, s[te])) ** 2))
    best = min(sse, key=lambda g: (sse[g], g if kind in DF_KINDS else -g))
    return best, sse


def fit_carry(r0, r1, R0, city, by_r=True):
    """Carry-over of the head residual between dates, pooled over cities with city intercepts:
    r1 = a_c + (k0 + k1 R0) r0 (k1 = 0 if not by_r). Flat inputs. Returns predict(r0, R0, city)."""
    r0, r1, R0, city = (np.asarray(a).ravel() for a in (r0, r1, R0, city))
    ids = np.unique(city)

    def demean(v):
        out = v.astype(float).copy()
        for c in ids:
            m = city == c
            out[m] -= v[m].mean()
        return out
    X = np.c_[demean(r0), demean(r0 * R0)] if by_r else demean(r0)[:, None]
    k = np.linalg.lstsq(X, demean(r1), rcond=None)[0]
    k0, k1 = float(k[0]), (float(k[1]) if by_r else 0.0)
    a = {c: float(r1[city == c].mean() - k0 * r0[city == c].mean() - k1 * (r0 * R0)[city == c].mean()) for c in ids}

    def predict(r0n, R0n, cityn):
        r0n, R0n, cityn = (np.asarray(v).ravel() for v in (r0n, R0n, cityn))
        return np.array([a.get(c, 0.0) for c in cityn]) + (k0 + k1 * R0n) * r0n
    return predict


def _r2_pooled_within(obs, pred, groups):
    """1 - SSE/SST with SST around each group's own mean (groups: integer labels, e.g. cell or city)."""
    obs, pred, groups = (np.asarray(v).ravel() for v in (obs, pred, groups))
    sst = 0.0
    for g in np.unique(groups):
        m = groups == g
        sst += float(np.sum((obs[m] - obs[m].mean()) ** 2))
    return 1.0 - float(np.sum((obs - pred) ** 2)) / sst if sst > 0 else float('nan')


def cv_select_df_joint(s_tr, W_tr, grid=None, dw_mask=None, weight=0.5, by_r=True, K=5,
                       monotone=False, seed=0, kind='bs'):
    """Choose ONE df (shared by all cells) for the per-date level heads, scoring both uses of the head:
        loss(df) = (1 - weight) * (1 - R2_W) + weight * (1 - R2_dW)
    R2_W  = pooled out-of-fold R2 of W at both dates (SST within city-date);
    R2_dW = pooled out-of-fold R2 of observed dW = W1 - W0 over the tracts in dw_mask (SST within city), where
            the dW prediction is the updating predictor actually used: h1(R1) + carry-over(r0) - W0, with the
            carry-over (kappa linear in R0 if by_r) refitted on the training folds.
    Folds are drawn per city at the TRACT level, so a tract's two dates are always held out together.
    s_tr, W_tr: (C, n, 2); dw_mask: (C, n) bool (default all). weight = 0 scores levels only.
    kind 'ps': the tuning value is the P-spline penalty instead of df (ties -> the larger penalty).
    Returns (best_df, table) with table = {df: (loss, R2_W, R2_dW)}."""
    C, n, T = s_tr.shape
    grid = default_grid(kind, grid)
    fitter = make_fitter(kind, monotone)
    cast = int if kind in DF_KINDS else float
    assert T == 2
    dw_mask = np.ones((C, n), bool) if dw_mask is None else np.broadcast_to(np.asarray(dw_mask, bool), (C, n))
    folds = np.stack([np.random.default_rng([int(seed), c]).permutation(n) % K for c in range(C)])
    city = np.broadcast_to(np.arange(C)[:, None], (C, n))
    cell = np.broadcast_to((np.arange(C)[:, None, None] * 2 + np.arange(2)), (C, n, 2))
    table = {}
    for g in (cast(v) for v in grid):
        oof_W = np.empty((C, n, 2))
        oof_W1u = np.empty((C, n))                       # updated date-1 prediction h1(R1) + carry
        for k in range(K):
            te = folds == k                              # (C, n)
            h = np.empty((C, n, 2))
            for c in range(C):
                for t in range(2):
                    m = fitter(s_tr[c, ~te[c], t], W_tr[c, ~te[c], t], g)
                    h[c, :, t] = predict_cell(m, s_tr[c, :, t])
            r = W_tr - h
            carry = fit_carry(r[..., 0][~te], r[..., 1][~te], s_tr[..., 0][~te], city[~te], by_r)
            oof_W[te] = h[te]
            oof_W1u[te] = h[..., 1][te] + carry(r[..., 0][te], s_tr[..., 0][te], city[te])
        r2w = _r2_pooled_within(W_tr, oof_W, cell)
        dw = W_tr[..., 1] - W_tr[..., 0]
        r2d = _r2_pooled_within(dw[dw_mask], (oof_W1u - W_tr[..., 0])[dw_mask], city[dw_mask])
        table[g] = ((1 - weight) * (1 - r2w) + weight * (1 - r2d), r2w, r2d)
    sign = 1 if kind in DF_KINDS else -1                 # ties: fewer df / more penalty (the smoother fit)
    best = min(table, key=lambda g: (table[g][0], sign * g))
    return best, table


def cv_spline_predict(s_tr, W_tr, s_te, grid=None, K=5, monotone=False, seed=0, kind='bs'):
    """Select df (or the P-spline penalty) by CV, refit on all training tracts, predict s_te. Returns (pred, df)."""
    df, _ = cv_select_df(s_tr, W_tr, grid, K, monotone, seed, kind)
    return bspline_predict(s_tr, W_tr, s_te, df, monotone, kind), df
