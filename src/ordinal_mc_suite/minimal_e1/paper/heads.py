"""Second-step heads on a frozen index and the evaluation metrics (held-out tracts)."""
import numpy as np
try:                                    # scipy >= 1.12
    from scipy.optimize import isotonic_regression as _scipy_iso
except ImportError:                     # older scipy: fall back to the local PAVA
    _scipy_iso = None

from . import dgp as D
from . import heads_spline as HS
from .stability import within_cov

# Feasible head (decided Nov 2026): per-city-date cubic B-spline, df chosen by 5-fold CV with ONE df shared
# by all cells (df = 2 is plain linear), monotone (non-decreasing) fit. Isotonic is kept only as a reference.
SPLINE_GRID = (2, 3, 4, 5, 6, 8)
SPLINE_MONOTONE = True


def pava(y, w=None):
    """Pool-adjacent-violators: non-decreasing least-squares fit."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wts[-2] += wts[-1]; cnt[-2] += cnt[-1]; vals[-2] = v
            vals.pop(); wts.pop(); cnt.pop()
    return np.repeat(vals, cnt)


def iso_predict(s_tr, W_tr, s_te):
    """Per-cell isotonic head: PAVA of W on the training index, linear interpolation (clamped)."""
    C, _, T = s_tr.shape
    out = np.empty(s_te.shape)
    for c in range(C):
        for t in range(T):
            o = np.argsort(s_tr[c, :, t], kind='stable')
            y = W_tr[c, o, t]
            fit = _scipy_iso(y).x if _scipy_iso is not None else pava(y)
            out[c, :, t] = np.interp(s_te[c, :, t], s_tr[c, o, t], fit)
    return out


def oracle_predict(spec, params, s_te):
    """True conditional-mean map (structural, nominal noise) evaluated at the estimated index."""
    return D.cond_mean(spec, params, s_te, nominal=True)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def index_metrics(z_te, theta, theta0):
    """RMSE of the standardised index (within-cell pooled; and pooled across cities)."""
    out = {}
    for name, S in (('f', within_cov(z_te)), ('fpool', np.cov(z_te.reshape(-1, z_te.shape[-1]).T, bias=True))):
        den = np.sqrt((theta @ S @ theta) * (theta0 @ S @ theta0))
        cos = float(theta @ S @ theta0 / den) if den > 0 else 0.0
        out[name] = float(np.sqrt(max(2.0 * (1.0 - np.clip(cos, -1, 1)), 0.0)))
    return out


def r2_within(obs, pred, base=None):
    """Out-of-sample R^2 against OBSERVED values, within cell, pooled over cells:
    1 - sum (obs - pred)^2 / sum (obs - base_cell)^2. obs/pred: (C, m, T) for W, (C, m) for dW.
    base: per-cell benchmark prediction, by default the cell mean of the TRAINING values (pass the training
    array, same leading/trailing layout); the held-out mean is used only if base is None (biased downward when
    held-out cells are small, because the benchmark then sees the test data)."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    if base is None:
        ref = obs.mean(axis=1, keepdims=True)
    else:
        ref = np.asarray(base, float).mean(axis=1, keepdims=True)
    sst = np.sum((obs - ref) ** 2)
    return float(1.0 - np.sum((obs - pred) ** 2) / sst) if sst > 0 else float('nan')


def _d(a):
    return a[..., 1] - a[..., 0]


def head_metrics(W_hat, mu, W_obs=None, W_base=None):
    """RMSE vs the true conditional mean (W, dW) and, if W_obs is given, within-cell out-of-sample R^2 vs
    observed W, dW (benchmark: cell means of W_base, the training values)."""
    out = dict(W=rmse(W_hat, mu), dW=rmse(_d(W_hat), _d(mu)))
    if W_obs is not None:
        out['r2W'] = r2_within(W_obs, W_hat, W_base)
        out['r2dW'] = r2_within(_d(W_obs), _d(W_hat), None if W_base is None else _d(W_base))
    return out


def spline_predict(s_tr, W_tr, s_te):
    pred, df = HS.cv_spline_predict(s_tr, W_tr, s_te, grid=SPLINE_GRID, monotone=SPLINE_MONOTONE)
    return pred, df


def _put(row, met, head):
    for k, v in met.items():
        if k in ('W', 'dW'):
            row[f'rmse_{k}_{head}'] = v
        else:                                   # r2W / r2dW
            row[f'r2_{k[2:]}_{head}'] = v


def evaluate_theta(panel, tr, te, theta, native=None, with_iso=False, with_spline=True):
    """Metrics for one direction. native: optional callable(z_tr, W_tr, z_te) -> W_hat.
    Heads: oracle (true map at the estimated index), spline (CV B-spline; the feasible head), native (FE/MM),
    iso (reference only, with_iso=True). Every head gets RMSE vs E[W|x] and within-cell out-of-sample R^2 vs
    observed held-out W and dW, benchmarked against each cell's training mean.
    The ceiling (the true E[W|x] as the predictor) is stored as r2_W_ceiling / r2_dW_ceiling."""
    z_tr, z_te = D.take(panel.z, tr), D.take(panel.z, te)
    W_tr, W_te = D.take(panel.W, tr), D.take(panel.W, te)
    f_te, a1c_te = D.take(panel.f, te), D.take(panel.a1c, te)
    mu = D.cond_mean(panel.spec, panel.params, f_te, a1c_te)
    row = {f'rmse_{k}': v for k, v in index_metrics(z_te, theta, panel.theta0).items()}
    row['r2_W_ceiling'] = r2_within(W_te, mu, W_tr)
    row['r2_dW_ceiling'] = r2_within(_d(W_te), _d(mu), _d(W_tr))
    s_tr, s_te = z_tr @ theta, z_te @ theta
    _put(row, head_metrics(oracle_predict(panel.spec, panel.params, s_te), mu, W_te, W_tr), 'oracle')
    if with_spline:
        pred, df = spline_predict(s_tr, W_tr, s_te)
        _put(row, head_metrics(pred, mu, W_te, W_tr), 'spline')
        row['spline_df'] = int(df)
    if with_iso:
        _put(row, head_metrics(iso_predict(s_tr, W_tr, s_te), mu, W_te, W_tr), 'iso')
    if native is not None:
        _put(row, head_metrics(native(z_tr, W_tr, z_te), mu, W_te, W_tr), 'native')
    if with_spline and D.persistence_kind(panel.spec):
        row.update(update_metrics(panel, tr, te, theta))
    return row


# ---------------------------------------------------------------------------
# E5: baseline updating (does the tract's own baseline label improve date-1 W and dW?)
# ---------------------------------------------------------------------------
UPDATE_HEADS = ('survey', 'image', 'imgchg', 'kconst', 'kR', 'oracle')
# df of the level heads used for updating: chosen by a joint CV loss (1-w)(1-R2_W) + w(1-R2_dW), see
# heads_spline.cv_select_df_joint. group = which training tracts score the dW term:
#   'all' | 'changing' (known structural changers; in the MC the non-stable tracts) |
#   'bigchange' (observable proxy: tracts whose image index moved more than the city median, |R1 - R0|).
# weight = 0 is the levels-only rule used for the W heads. Task: compare with Cobb-Douglas etc. later.
UPDATE_DF_RULE = dict(weight=0.5, group='changing', kind='bs')   # kind: 'bs' (df grid) | 'ps' (P-spline)


def dw_group_mask(rule_group, stable_tr, s_tr):
    """(C, m_tr) bool mask of training tracts entering the dW term of the joint df criterion."""
    C, m = s_tr.shape[:2]
    if rule_group == 'all':
        return np.ones((C, m), bool)
    if rule_group == 'changing':
        return ~np.broadcast_to(stable_tr, (C, m))
    if rule_group == 'bigchange':
        d = np.abs(s_tr[..., 1] - s_tr[..., 0])
        return d > np.median(d, axis=1, keepdims=True)
    raise ValueError(rule_group)


def update_df(s_tr, W_tr, stable_tr, rule=None):
    """df for the updating heads under rule (default UPDATE_DF_RULE). Returns (df, table)."""
    rule = dict(UPDATE_DF_RULE, **(rule or {}))
    grid = SPLINE_GRID if rule['kind'] in HS.DF_KINDS else None
    if rule['weight'] == 0:
        return HS.cv_select_df(s_tr, W_tr, grid, monotone=SPLINE_MONOTONE, kind=rule['kind'])
    mask = dw_group_mask(rule['group'], stable_tr, s_tr)
    return HS.cv_select_df_joint(s_tr, W_tr, grid, dw_mask=mask, weight=rule['weight'],
                                 monotone=SPLINE_MONOTONE, kind=rule['kind'])


def _survey_forecast(W_tr, W0_te):
    """Survey-only: keep each tract's baseline rank within its city (training tracts' date-0 distribution)
    and read the same quantile of the city's date-1 training distribution. (C, m)."""
    out = np.empty_like(W0_te)
    for c in range(W_tr.shape[0]):
        ref0 = np.sort(W_tr[c, :, 0])
        q = np.searchsorted(ref0, W0_te[c], side='right') / (len(ref0) + 1.0)
        q = np.clip(q, 0.5 / len(ref0), 1 - 0.5 / len(ref0))
        out[c] = np.quantile(W_tr[c, :, 1], q)
    return out


def _carry_over(r0_tr, r1_tr, R0_tr, by_r):
    """OLS of r1 on r0 (and r0*R0) with city intercepts, on training tracts. Returns (a_c, k0, k1)."""
    C, m = r0_tr.shape
    a = np.zeros(C)
    x0 = r0_tr - r0_tr.mean(axis=1, keepdims=True)
    y = r1_tr - r1_tr.mean(axis=1, keepdims=True)
    if by_r:
        x1 = r0_tr * R0_tr
        x1 = x1 - x1.mean(axis=1, keepdims=True)
        X = np.stack([x0.ravel(), x1.ravel()], axis=1)
        k = np.linalg.lstsq(X, y.ravel(), rcond=None)[0]
        k0, k1 = float(k[0]), float(k[1])
        a = r1_tr.mean(axis=1) - k0 * r0_tr.mean(axis=1) - k1 * (r0_tr * R0_tr).mean(axis=1)
    else:
        k0 = float((x0 * y).sum() / max((x0 * x0).sum(), 1e-300))
        k1 = 0.0
        a = r1_tr.mean(axis=1) - k0 * r0_tr.mean(axis=1)
    return a, k0, k1


def _r2_group(obs, pred, base_obs, mask_te, mask_tr):
    """Within-city R^2 restricted to a group of tracts; benchmark = the city's mean of the same group's
    training values. obs/pred: (C, m); base_obs: (C, m_tr); masks: same shapes."""
    sse = sst = 0.0
    for c in range(obs.shape[0]):
        mt, mb = mask_te[c], mask_tr[c]
        if not mt.any():
            continue
        ref = base_obs[c][mb].mean() if mb.any() else obs[c][mt].mean()
        sse += float(np.sum((obs[c][mt] - pred[c][mt]) ** 2))
        sst += float(np.sum((obs[c][mt] - ref) ** 2))
    return 1.0 - sse / sst if sst > 0 else float('nan')


def _r2_target(true, pred, mask):
    """R^2 of a noise-free target within city over the tracts in mask (benchmark: its city mean)."""
    sse = sst = 0.0
    for c in range(true.shape[0]):
        m = mask[c]
        if m.sum() < 2:
            continue
        sse += float(np.sum((true[c][m] - pred[c][m]) ** 2))
        sst += float(np.sum((true[c][m] - true[c][m].mean()) ** 2))
    return 1.0 - sse / sst if sst > 0 else float('nan')


def update_metrics(panel, tr, te, theta, df_rule=None):
    """Predict held-out tracts' date-1 W (hence dW = W_1 - W_0, with W_0 observed) with:
      survey  - baseline rank carried forward, no imagery;
      image   - CV-spline head on the date-1 index, ignores the baseline (implies the whole baseline
                residual disappears: kappa = 0);
      imgchg  - baseline plus the image-predicted change, W_0 + h_1(R_1) - h_0(R_0) (the usual
                levels-plus-changes practice; implies the whole residual persists: kappa = 1);
      kconst  - image head + constant carry-over weight times the baseline residual;
      kR      - same with the weight linear in the baseline index, kappa(R) = k0 + k1 R;
      oracle  - true E[W_1 | x, W_0] (infeasible ceiling).
    Residuals r_t = W_t - h_t(R_t) use the per-date spline heads fitted on training tracts, with ONE df chosen
    by the joint W/dW criterion (update_df, rule df_rule or UPDATE_DF_RULE); kappa is estimated on training
    tracts (both dates labelled) by OLS with city intercepts. Also: spline_df_upd, r2_W_upd (levels R^2 of
    these heads, to see what the joint rule costs in levels).
    Metrics (S = stable tracts, C = changing tracts):
      rmse_dWc[S|C]_<p>  RMSE of predicted dW vs the target E[dW | x, W_0];
      r2_dWo[S|C]_<p>    within-city R^2 of OBSERVED dW, overall and within each group (benchmark: the city's
                         training mean of the same group);
      r2_dWsC_<p>, rmse_dWsC_<p>  recovery of the STRUCTURAL part of the change on changing tracts: true =
                         oracle 'struct' component (change caused by the index change); predicted = the
                         predictor's change attributable to its index change, h_1(R_1) - h_1(R_0)
                         (0 for the survey-only forecast).
      vdec[C]_<part>     Shapley decomposition of the within-city R^2 of observed dW (all / changing tracts)
                         among the oracle components map (the city map moving at the tract's structural
                         position), struct (structural change) and resid (residual dynamics); noise =
                         1 - oracle R^2. Shares sum to 1."""
    z_tr, z_te = D.take(panel.z, tr), D.take(panel.z, te)
    W_tr, W_te = D.take(panel.W, tr), D.take(panel.W, te)
    s_tr, s_te = z_tr @ theta, z_te @ theta
    m_tr = s_tr.shape[1]
    s_all = np.concatenate([s_tr, s_te], axis=1)
    stable_tr = np.broadcast_to(np.asarray(panel.stable)[np.asarray(tr)], W_tr.shape[:2])
    df, _ = update_df(s_tr, W_tr, stable_tr, df_rule)
    kind = dict(UPDATE_DF_RULE, **(df_rule or {}))['kind']
    pred_all = HS.bspline_predict(s_tr, W_tr, s_all, df, SPLINE_MONOTONE, kind)
    h_tr, h_te = pred_all[:, :m_tr], pred_all[:, m_tr:]
    s_frozen = np.stack([s_te[..., 0], s_te[..., 0]], axis=-1)
    h1_at_R0 = HS.bspline_predict(s_tr, W_tr, s_frozen, df, SPLINE_MONOTONE, kind)[..., 1]
    struct_hat = h_te[..., 1] - h1_at_R0                 # change attributable to the index change
    r_tr = W_tr - h_tr                               # (C, m_tr, 2)
    r0_te = W_te[..., 0] - h_te[..., 0]
    W0, W1 = W_te[..., 0], W_te[..., 1]
    comp = D.change_components(panel, te)
    target = W0 + comp['total']                      # E[W_1 | x, W_0]
    preds = {'survey': _survey_forecast(W_tr, W0), 'image': h_te[..., 1],
             'imgchg': W0 + h_te[..., 1] - h_te[..., 0], 'oracle': target}
    a, k0, _ = _carry_over(r_tr[..., 0], r_tr[..., 1], s_tr[..., 0], by_r=False)
    preds['kconst'] = h_te[..., 1] + a[:, None] + k0 * r0_te
    a2, j0, j1 = _carry_over(r_tr[..., 0], r_tr[..., 1], s_tr[..., 0], by_r=True)
    preds['kR'] = h_te[..., 1] + a2[:, None] + (j0 + j1 * s_te[..., 0]) * r0_te
    stable = np.broadcast_to(np.asarray(panel.stable)[np.asarray(te)], W0.shape)
    dW_obs = W1 - W0
    dW_tr = W_tr[..., 1] - W_tr[..., 0]
    out = dict(kappa_const=k0, kappa_R0=j0, kappa_R1=j1, spline_df_upd=float(df),
               r2_W_upd=r2_within(W_te, h_te, W_tr))
    struct_preds = {'survey': np.zeros_like(W0), 'oracle': comp['struct']}
    for name, p1 in preds.items():
        err = p1 - target                            # = predicted dW - E[dW | x, W_0]
        out[f'rmse_dWc_{name}'] = float(np.sqrt(np.mean(err ** 2)))
        out[f'rmse_dWcS_{name}'] = float(np.sqrt(np.mean(err[stable] ** 2))) if stable.any() else float('nan')
        out[f'rmse_dWcC_{name}'] = float(np.sqrt(np.mean(err[~stable] ** 2))) if (~stable).any() else float('nan')
        out[f'r2_dWo_{name}'] = r2_within(dW_obs, p1 - W0, dW_tr)
        out[f'r2_dWoS_{name}'] = _r2_group(dW_obs, p1 - W0, dW_tr, stable, stable_tr)
        out[f'r2_dWoC_{name}'] = _r2_group(dW_obs, p1 - W0, dW_tr, ~stable, ~stable_tr)
        sp = struct_preds.get(name, struct_hat)
        out[f'rmse_dWsC_{name}'] = float(np.sqrt(np.mean((sp - comp['struct'])[~stable] ** 2)))
        out[f'r2_dWsC_{name}'] = _r2_target(comp['struct'], sp, ~stable)
    # Reference for the structural panel: the best a predictor built on the marginal curve E[W_1 | index] can do
    # (true map, true index, no estimation error). Its gap to 1 is the part of the structural change that depends
    # on the tract's baseline residual (nonlinear map: the effect of an index change depends on where the tract
    # sits in latent space); the gap between it and the feasible predictors is first-step + head estimation error.
    f_te = D.take(panel.f, te)
    f_frozen = np.stack([f_te[..., 0], f_te[..., 0]], axis=-1)
    marg = (D.cond_mean(panel.spec, panel.params, f_te)[..., 1]
            - D.cond_mean(panel.spec, panel.params, f_frozen)[..., 1])
    out['r2_dWsC_marginal'] = _r2_target(comp['struct'], marg, ~stable)
    # Shapley decomposition of the within-city R^2 of observed dW among the three oracle components
    # (map moving, structural change, residual dynamics); noise = 1 - oracle R^2. Order-free; sums to 1.
    def _wc(a, m):
        return np.concatenate([a[c][m[c]] - a[c][m[c]].mean() for c in range(a.shape[0]) if m[c].sum() > 1])

    import itertools
    parts = ('map', 'struct', 'resid')
    for tag, mask in (('', np.ones_like(stable)), ('C', ~stable)):
        o = _wc(dW_obs, mask)
        sst = float(np.sum(o ** 2))
        cache = {}

        def r2_of(subset):
            key = tuple(sorted(subset))
            if key not in cache:
                pred = sum((_wc(comp[q], mask) for q in key), np.zeros_like(o))
                cache[key] = 1.0 - float(np.sum((o - pred) ** 2)) / sst
            return cache[key]
        shap = dict.fromkeys(parts, 0.0)
        perms = list(itertools.permutations(parts))
        for perm in perms:
            done = []
            for q in perm:
                shap[q] += (r2_of(done + [q]) - r2_of(done)) / len(perms)
                done.append(q)
        for q in parts:
            out[f'vdec{tag}_{q}'] = shap[q]
        out[f'vdec{tag}_noise'] = 1.0 - r2_of(list(parts))
    return out
