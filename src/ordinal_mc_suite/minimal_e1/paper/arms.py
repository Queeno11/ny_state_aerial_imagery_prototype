"""Arms: fit on training tracts, evaluate on held-out tracts of the same cities."""
import time
import traceback

import numpy as np

from . import calibration as cal
from . import dgp as D
from . import fit
from . import heads
from . import kernels as kn
from . import stability as st
from .cells import CellSet
from .. import r3_estimators as r3e

SEL_GRID = (0.0, 0.5, 0.9, 0.99, 1.0)
ARM_ORDER = ['RNC', 'RNC_S', 'RNC_P0.5', 'RNC_P0.9', 'RNC_P0.99', 'RNC_SEL', 'RNC_S_DW', 'SUR', 'SUR_S',
             'MRC_S', 'FE', 'FE_S', 'NS', 'MM', 'ML']
STAB_ARMS = {'RNC_S', 'RNC_P0.5', 'RNC_P0.9', 'RNC_P0.99', 'RNC_SEL', 'RNC_S_DW', 'SUR_S', 'MRC_S', 'FE_S'}


class Ctx:
    """Training-side objects shared by the arms of one replication."""

    def __init__(self, panel, tr, te, key, cfg):
        self.panel, self.tr, self.te, self.key, self.cfg = panel, tr, te, key, cfg
        self.spec = panel.spec
        self.z = D.take(panel.z, tr)
        self.W = D.take(panel.W, tr)
        self.xy = D.take(panel.xy, tr)
        self.n_tr = self.z.shape[1]
        self.p = self.z.shape[-1]
        self.Sw = st.within_cov(self.z)
        self.D = D.link_rows(panel, tr)
        self.has_links = len(self.D) > 0
        self.N, self.rho_eig = st.null_space(self.D, self.Sw, panel.spec.link_tol)
        self.B_hard = st.active_basis(self.N, self.Sw)
        self.B_free = st.active_basis(np.eye(self.p), self.Sw)
        self.cs = CellSet.build(self.z, self.W, self.xy, 0.0, cfg.get('max_pairs'), seed=key[-1])
        self.C_city = st.city_basis(self.Sw)
        self.pen = st.Penalty(self.D, self.Sw, self.C_city) if self.has_links else None
        self.cache = {}
        self.policy = cfg.get('policy')

    def cells_dw(self):
        if 'cs_dw' not in self.cache:
            self.cache['cs_dw'] = CellSet.build(self.z, self.W, self.xy, self.cfg.get('dw_ell', 0.1),
                                                self.cfg.get('max_pairs'), seed=self.key[-1])
        return self.cache['cs_dw']

    @property
    def n_c(self):
        return self.n_tr


def _timed(fn):
    t = time.perf_counter()
    out = fn()
    out['secs'] = time.perf_counter() - t
    return out


def _fallback(ctx, out, info):
    """Restriction annihilated every active direction: report the unrestricted estimate, flagged."""
    r = dict(out)
    r['fallback'] = 1
    r.update(info)
    return r


# ---------------------------------------------------------------------------
def arm_RNC(ctx):
    return fit.fit_rnc(ctx.cs, ctx.B_free, max_iter=ctx.cfg.get('rnc_iter', 1000))


def arm_RNC_S(ctx):
    if ctx.B_hard.shape[1] == 0:
        return dict(get(ctx, 'RNC'), fallback=1)
    r = fit.fit_rnc(ctx.cs, ctx.B_hard, max_iter=ctx.cfg.get('rnc_iter', 1000))
    r['q'] = int(ctx.B_hard.shape[1])
    return r


def arm_RNC_S_DW(ctx):
    if ctx.B_hard.shape[1] == 0:
        return dict(get(ctx, 'RNC'), fallback=1)
    return fit.fit_rnc(ctx.cells_dw(), ctx.B_hard, max_iter=ctx.cfg.get('rnc_iter', 1000))


def lam_compute(ctx):
    """Per-rep lambda(omega) from training-data curvature traces at the hard estimate."""
    def compute(omega):
        h = get(ctx, 'RNC_S')
        trR, trS = st.curvature_traces(ctx.cs, ctx.pen, h['theta'], h['M'])
        return st.lambda_for_omega(omega, trR, trS)
    return compute


def get_lambda(ctx, omega):
    return cal.lambda_for(ctx.policy, omega, lam_compute(ctx))


def penalised(ctx, omega, cs=None, B=None):
    lam = get_lambda(ctx, omega)
    r = fit.fit_rnc(cs or ctx.cs, B if B is not None else ctx.B_free, ctx.pen, lam,
                    max_iter=ctx.cfg.get('rnc_iter', 1000))
    r['theta'] = ctx.pen.complete(r['theta'])        # city-level coordinates set by the links (profiled)
    r['lambda'], r['omega'] = float(lam), omega
    return r


def make_arm_RNC_P(omega):
    def arm(ctx):
        if not ctx.has_links:
            raise LookupError('no links')
        return penalised(ctx, omega)
    return arm


def arm_RNC_SEL(ctx):
    """omega chosen by held-out within-cell Kendall tau on a validation split of the TRAINING tracts."""
    if not ctx.has_links:
        raise LookupError('no links')
    n = ctx.n_tr
    n_val = max(10, n // 4)
    gs = [np.random.default_rng(np.random.SeedSequence([*ctx.key, 31, c])) for c in range(ctx.z.shape[0])]
    perm = np.stack([g.permutation(n) for g in gs])
    sub, val = np.sort(perm[:, n_val:], 1), np.sort(perm[:, :n_val], 1)
    z_s, W_s, xy_s = D.take(ctx.z, sub), D.take(ctx.W, sub), D.take(ctx.xy, sub)
    z_v, W_v = D.take(ctx.z, val), D.take(ctx.W, val)
    cs_s = CellSet.build(z_s, W_s, xy_s, 0.0, ctx.cfg.get('max_pairs'), seed=ctx.key[-1])
    scores = {}
    for w in SEL_GRID:
        if w >= 1.0:
            B = ctx.B_hard if ctx.B_hard.shape[1] else ctx.B_free
            r = fit.fit_rnc(cs_s, B, max_iter=ctx.cfg.get('rnc_iter', 1000))
        elif w <= 0.0:
            r = fit.fit_rnc(cs_s, ctx.B_free, max_iter=ctx.cfg.get('rnc_iter', 1000))
        else:
            r = penalised(ctx, w, cs_s)
        scores[w] = fit.mean_kendall(z_v @ r['theta'], W_v)
    best = max(SEL_GRID, key=lambda w: scores[w])
    name = {0.0: 'RNC', 1.0: 'RNC_S'}.get(best, f'RNC_P{best}')
    out = dict(get(ctx, name))
    out['omega_sel'] = float(best)
    out['kendall_val'] = {str(k): float(v) for k, v in scores.items()}
    return out


def arm_SUR(ctx):
    start = get(ctx, 'RNC')['theta']
    return fit.fit_sur(ctx.cs, r3e.bandwidth(ctx.n_c), ctx.B_free, start, ctx.cfg.get('sur_extra_starts', 3),
                       seed=ctx.key[-1])


def arm_SUR_S(ctx):
    if ctx.B_hard.shape[1] == 0:
        return dict(get(ctx, 'SUR'), fallback=1)
    return fit.fit_sur(ctx.cs, r3e.bandwidth(ctx.n_c), ctx.B_hard, get(ctx, 'RNC_S')['theta'],
                       ctx.cfg.get('sur_extra_starts', 3), seed=ctx.key[-1])


def arm_MRC_S(ctx):
    if ctx.B_hard.shape[1] != 2:
        raise LookupError(f'restricted active space is {ctx.B_hard.shape[1]}-d, exact MRC needs 2-d')
    return fit.fit_mrc(ctx.cs, ctx.B_hard)


def arm_FE(ctx):
    return fit.fit_fe(ctx.z, ctx.W, ctx.B_free)


def arm_FE_S(ctx):
    if ctx.B_hard.shape[1] == 0:
        return dict(get(ctx, 'FE'), fallback=1)
    return fit.fit_fe(ctx.z, ctx.W, ctx.B_hard)


def arm_NS(ctx):
    return fit.fit_fe(ctx.z, ctx.W, ctx.B_free, 'normal')


def arm_MM(ctx):
    return fit.fit_mm(ctx.z, ctx.W, ctx.B_free)


def arm_ML(ctx):
    if not ctx.spec.ml_correct():
        raise LookupError('ML only where the model is correctly specified')
    return fit.fit_ml(ctx.z, ctx.W, ctx.B_free, ctx.spec.map, D.KAPPA, get(ctx, 'FE')['theta'])


ARM_FUNS = {'RNC': arm_RNC, 'RNC_S': arm_RNC_S, 'RNC_S_DW': arm_RNC_S_DW, 'RNC_SEL': arm_RNC_SEL,
            'RNC_P0.5': make_arm_RNC_P(0.5), 'RNC_P0.9': make_arm_RNC_P(0.9), 'RNC_P0.99': make_arm_RNC_P(0.99),
            'SUR': arm_SUR, 'SUR_S': arm_SUR_S, 'MRC_S': arm_MRC_S, 'FE': arm_FE, 'FE_S': arm_FE_S,
            'NS': arm_NS, 'MM': arm_MM, 'ML': arm_ML}
DEPS = {'RNC_SEL': ['RNC', 'RNC_S', 'RNC_P0.5', 'RNC_P0.9', 'RNC_P0.99'], 'SUR': ['RNC'], 'SUR_S': ['RNC_S'],
        'ML': ['FE'], 'FE_S': ['FE'], 'RNC_S': ['RNC'], 'RNC_P0.5': ['RNC_S'], 'RNC_P0.9': ['RNC_S'],
        'RNC_P0.99': ['RNC_S'], 'RNC_S_DW': ['RNC']}
# dependencies only needed in some modes: computed lazily by get()


def get(ctx, name):
    """Fit an arm once per replication (cached; includes time)."""
    if name not in ctx.cache:
        ctx.cache[name] = _timed(lambda: ARM_FUNS[name](ctx))
    return ctx.cache[name]


def native_for(ctx, name, res):
    if name in ('FE', 'FE_S') and 'delta' in res:
        d = res['delta']
        return lambda z_tr, W_tr, z_te: fit.native_fe(z_tr, W_tr, d, z_te)
    if name == 'MM' and 'delta' in res:
        d = res['delta']
        return lambda z_tr, W_tr, z_te: fit.native_mm(z_tr, W_tr, d, z_te)
    return None


def run_arms(ctx, names):
    """Fit and evaluate. Returns a list of row dicts (one per arm)."""
    rows = []
    for name in names:
        row = dict(arm=name, status='ok')
        try:
            if name in STAB_ARMS and not ctx.has_links:
                raise LookupError('no links declared')
            res = get(ctx, name)
            met = heads.evaluate_theta(ctx.panel, ctx.tr, ctx.te, res['theta'], native_for(ctx, name, res))
            row.update(met)
            row['secs'] = res['secs']
            row['ok'] = bool(res.get('ok', True))
            row['theta'] = [float(x) for x in res['theta']]
            for k in ('lambda', 'omega', 'omega_sel', 'fallback', 'q', 'tau_median', 'nit'):
                if k in res:
                    row[k] = res[k]
            if 'kendall_val' in res:
                row['kendall_val'] = res['kendall_val']
        except LookupError as e:
            row.update(status='na', error=str(e))
        except Exception as e:                                   # keep the replication alive
            row.update(status='error', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-600:])
        rows.append(row)
    return rows
