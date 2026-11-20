"""Paper DGP. Cells = city x date (t in {0,1}); x = M [H, A, (city nuisance), (h_c)] mixed and
whitened with a fixed population reference; f(x) = x'theta0, ||theta0|| = 1 (whitened coordinates).

Raw coordinates: H (2, within-city structural, skewed, city shift), A (p_a tract nuisance redrawn
every date, loaded on H), optionally N (2, city-level nuisance, NOT in f), optionally h_c (1, city
constant, in f with gamma = 0.5). p = 6 in every scenario (p_a = 4 - 2*city_nuis - hc).

Links (exactly equal index):
  temporal : tract i with Delta H_i = 0 (stable tracts; A is redrawn so the gap spans the nuisance)
  cross    : twin tracts a (city c) and b (neighbour city c', ring graph with offsets +-1, +-2 = 4
             neighbours): H_b is set so that g(b) = g(a) EXACTLY (g = coef'H + gamma h_c; if h_c is
             present H_b is shifted along coef), the nuisance A_b is an independent draw.
Corruptions: fake (declared links that are not equal-index; fake temporal tracts drift in direction
e_f, fake cross pairs are random neighbour tracts), declared share, approximate (index change of
sd delta on the stable tracts / twins).
Outcome: W = G_ct(f + eta) (+ outside term in D) with maps a/b/c from dgp.py; eta has unit marginal
variance. Persistence rho, spatial GRF (exponential covariance, range r, share 0.5), E4 cases.
All conditional means E[W | x] are closed form (no Monte Carlo), see `cond_mean`.
"""
from dataclasses import dataclass, field, replace
from functools import lru_cache
import zlib

import numpy as np

from .. import dgp as base
from .. import robust_dgp as rdgp

SEED = 20261117
STREAMS = dict(ref_city=101, ref_tract=102, ref_extra=103, city=111, tract=112, extra=113, resid=114,
               spatial=115, coords=116, links=117, e4=118, split=119, drift=120)
CHANGE_SD = 0.5
FAKE_DRIFT = 0.8
E_F = np.array([1.0, 0.3]) / np.hypot(1.0, 0.3)
E4_A_I, E4_GAMMA_O, E4_RHO_O, E4_A_III = 0.6, 0.7, 0.6, 0.5
E4_A1_CLIP = 2.5
KAPPA = 0.5
COEF = np.array([1.0, 0.7])
_SM = dict(p=(0.2, 0.8), m=(1.0 / np.sqrt(0.65), -0.25 / np.sqrt(0.65)), s=(1.0 / np.sqrt(0.65), 0.5 / np.sqrt(0.65)))


@dataclass(frozen=True)
class Spec:
    map: str = 'c'
    noise: str = 'gauss'            # 'gauss' | 'D' (scenario-D law of robust_dgp)
    city_nuis: bool = False
    hc: bool = False
    gamma: float = 0.5
    links: str = 'both'             # both | temporal | cross | none
    declared: float = 1.0
    fake: float = 0.0
    approx: float = 0.0
    rho: float = 0.0
    spatial_r: float = 0.0
    spatial_shared: bool = False
    spatial_share: float = 0.5
    e4: str = ''                    # '' | i | ii | iii
    # E5 baseline updating: latent noise = persistent tract part xi + transitory e (unit total variance at
    # each date); xi_x > 0 moves a share of xi into the index: xi = c*f + xi_tilde, so the target index is
    # f* = (1+c) f (same direction). persist: const | rdep | reset | fe_rel (see persistence()).
    persist: str = ''
    lam_p: float = 0.75             # persistent share of the latent noise at baseline
    rho_p: float = 0.8              # carry-over of xi_tilde between dates ('const')
    xi_x: float = 0.0               # share of xi's variance explained by x (0 or 0.5)
    link_tol: object = 1e-4         # hard null space: float tolerance (rms gap / within-cell sd) or 'gap'
    n_twin: int = 3

    @property
    def p_a(self):
        return 4 - 2 * int(self.city_nuis) - int(self.hc)

    @property
    def p(self):
        return 2 + self.p_a + 2 * int(self.city_nuis) + int(self.hc)

    def ml_correct(self):
        """Gaussian-inside noise, independence, no E4: the exact likelihood is correct."""
        return (self.noise == 'gauss' and self.rho == 0 and self.spatial_r == 0 and not self.e4
                and not self.hc and not self.city_nuis and not self.persist)


def rng(*keys):
    return base.rng(SEED, *keys)


def key_of(*parts):
    """Deterministic integer key from strings/ints (no Python hash)."""
    return tuple(zlib.crc32(str(x).encode()) % (2 ** 31) if isinstance(x, str) else int(x) for x in parts)


# --------------------------------------------------------------------------
# raw features
# --------------------------------------------------------------------------
def _ctx_cfg(spec):
    return dict(p_h=2, p_a=spec.p_a, city_sd=0.5, change_sd=CHANGE_SD, logbeta_sd=0.3, lambda_range=0.9)


def city_context(spec, *keys):
    ctx = base.city_context(rng(STREAMS['city'], *keys), _ctx_cfg(spec))
    g = rng(STREAMS['extra'], *keys)
    ctx['nuis'] = 0.7 * g.normal(size=2) if spec.city_nuis else None
    ctx['hc'] = float(g.normal()) if spec.hc else None
    return ctx


def draw_H(spec, ctx, n, g, with_drift=False, approx_scale=0.0):
    """Structural features (n,2,2), stable flags, and the A-noise (n,2,p_a)."""
    v = g.exponential(size=(n, 2)) - 1.0
    dz = g.normal(size=(n, 2))
    approx_dz = g.normal(size=(n, 2))
    noise = g.normal(size=(n, 2, spec.p_a))
    stable = np.arange(n) % 2 == 0
    h0 = ctx['dc'] + v
    dh_ch = CHANGE_SD * dz + (FAKE_DRIFT * E_F if with_drift else 0.0)
    dh_st = approx_scale * approx_dz
    dh = np.where(stable[:, None], dh_st, dh_ch)
    return np.stack([h0, h0 + dh], axis=1), stable, noise


def approx_scale(spec):
    """Isotropic dH sd that gives an index change of sd `approx` on 'stable' tracts:
    sd(coef'dH)/sigma_g = approx * |coef| s / (|coef| ...) -> s = approx * sigma_g / |coef|."""
    if spec.approx <= 0:
        return 0.0
    return spec.approx * reference(spec).sigma_g / np.linalg.norm(COEF)


def assemble_raw(spec, ctx, H, noise):
    n = H.shape[0]
    A = H @ base.B_NUISANCE[:spec.p_a].T + ctx['da'] + noise
    parts = [H, A]
    if spec.city_nuis:
        parts.append(np.broadcast_to(ctx['nuis'], (n, 2, 2)))
    if spec.hc:
        parts.append(np.full((n, 2, 1), ctx['hc']))
    return np.concatenate(parts, axis=-1)


def coef_full(spec):
    c = np.zeros(spec.p)
    c[:2] = COEF
    if spec.hc:
        c[-1] = spec.gamma
    return c


# --------------------------------------------------------------------------
# population reference (whitening, index standardisation, A1 moments)
# --------------------------------------------------------------------------
@dataclass
class Reference:
    spec: Spec
    mix: np.ndarray
    mean: np.ndarray
    chol: np.ndarray
    gm: float
    sigma_g: float
    theta0: np.ndarray
    a1_mean: float
    a1_sd: float

    def whiten(self, x):
        flat = x.reshape(-1, x.shape[-1]) - self.mean
        from scipy.linalg import solve_triangular
        return solve_triangular(self.chol, flat.T, lower=True).T.reshape(x.shape)


@lru_cache(maxsize=32)
def _reference(spec_key):
    spec = Spec(**dict(spec_key))
    mix = base.mixing(SEED, spec.p)
    raws = []
    for c in range(300):
        ctx = city_context(spec, 900000 + c)
        H, _, noise = draw_H(spec, ctx, 200, rng(STREAMS['ref_tract'], c))
        raws.append(assemble_raw(spec, ctx, H, noise).reshape(-1, spec.p))
    raw = np.concatenate(raws)
    x = raw @ mix.T
    mean = x.mean(0)
    cov = (x - mean).T @ (x - mean) / len(x)
    chol = np.linalg.cholesky(cov)
    cf = coef_full(spec)
    g = raw @ cf
    gm, sg = float(g.mean()), float(g.std())
    theta0 = chol.T @ (mix @ cf) / sg
    a1 = raw[:, 2]
    return Reference(spec, mix, mean, chol, gm, sg, theta0, float(a1.mean()), float(a1.std()))


def reference(spec):
    return _reference((('city_nuis', spec.city_nuis), ('hc', spec.hc), ('gamma', spec.gamma)))


# --------------------------------------------------------------------------
# noise
# --------------------------------------------------------------------------
def _spatial_chol(xy, r):
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    return np.linalg.cholesky(np.exp(-d / r) + 1e-8 * np.eye(len(xy)))


def draw_eta(spec, g, xy):
    """(n,2) inside noise with unit marginal variance; persistence rho and spatial field."""
    n = len(xy)
    u = g.normal(size=(n, 1))
    eps = g.normal(size=(n, 2))
    nug = np.sqrt(spec.rho) * u + np.sqrt(1 - spec.rho) * eps
    if spec.spatial_r <= 0:
        return nug
    L = _spatial_chol(xy, spec.spatial_r)
    if spec.spatial_shared:
        field = np.repeat(L @ g.normal(size=(n, 1)), 2, axis=1)
    else:
        field = L @ g.normal(size=(n, 2))
    s = spec.spatial_share
    return np.sqrt(s) * field + np.sqrt(1 - s) * nug


# --------------------------------------------------------------------------
# E5: persistent tract component with carry-over (baseline updating)
# --------------------------------------------------------------------------
RHO_STABLE_RESET = 0.95             # 'reset': carry-over of stable tracts; changing tracts redraw (rho = 0)


def persistence_kind(spec):
    """'' (none), 'e3' (E3 rho: literal fixed effect u + transitory eps) or spec.persist."""
    if spec.persist:
        return spec.persist
    if spec.rho > 0 and spec.spatial_r <= 0 and not spec.e4 and spec.noise == 'gauss':
        return 'e3'
    return ''


def persistence(spec, f0, stable):
    """Per-tract carry-over and variances, all (C, m) arrays (f0 = true baseline index, (C, m)).
    Returns dict(c, var_xi, var_e, rho) with  v_t = (1+c) f_t + xi_t + e_t,
    xi_1 = rho * xi_0 + sqrt(1 - rho^2) * xi_new,  Var(xi_t) = var_xi,  Var(e_t) = var_e (both dates)."""
    kind = persistence_kind(spec)
    shape = np.shape(f0)
    one = np.ones(shape)
    if kind == 'e3':                                    # eta_t = sqrt(rho) u + sqrt(1-rho) eps_t
        return dict(c=0.0, var_xi=spec.rho * one, var_e=(1 - spec.rho) * one, rho=one)
    lam = spec.lam_p
    var_xi = (1 - spec.xi_x) * lam * one
    c = float(np.sqrt(spec.xi_x * lam))
    var_e = (1 - lam) * one
    if kind == 'const':
        rho = spec.rho_p * one
    elif kind == 'rdep':                                # richer tracts more persistent: 0.5 .. 0.95
        rho = 0.5 + 0.45 * _ndtr(f0)
    elif kind == 'reset':                               # changing tracts re-sort: xi redrawn
        st = np.broadcast_to(np.asarray(stable, bool), shape)
        rho = np.where(st, RHO_STABLE_RESET, 0.0)
    elif kind == 'fe_rel':                              # literal fixed effect, noisier baselines for poorer tracts
        rho = one
        var_e = (1 - lam) * (0.4 + 1.2 * _ndtr(-f0))    # mean (1 - lam) over a symmetric f0
    else:
        raise ValueError(kind)
    return dict(c=c, var_xi=var_xi, var_e=var_e, rho=rho)


def _ndtr(x):
    from scipy.special import ndtr
    return ndtr(x)


def draw_persistent(spec, g, f0, stable):
    """(n, 2) latent noise xi_t + e_t for one city (spec.persist set)."""
    pr = persistence(spec, f0, stable)
    n = len(f0)
    xi0 = np.sqrt(pr['var_xi']) * g.normal(size=n)
    xi_new = np.sqrt(pr['var_xi']) * g.normal(size=n)
    xi1 = pr['rho'] * xi0 + np.sqrt(1 - pr['rho'] ** 2) * xi_new
    e = np.sqrt(pr['var_e'])[:, None] * g.normal(size=(n, 2))
    return np.stack([xi0, xi1], axis=1) + e


def _given_baseline(panel, idx, freeze_index=False):
    """E[W_1 | x, W_0] for tracts idx; freeze_index=True replaces the date-1 index by the date-0 one
    (the counterfactual 'no structural change'). Also returns v0 (true baseline latent value)."""
    spec = panel.spec
    f = take(panel.f, idx)
    W = take(panel.W, idx)
    pr = persistence(spec, f[..., 0], np.asarray(panel.stable)[np.asarray(idx)])
    c = pr['c']
    v0 = base.psi_inverse(W, spec.map, panel.params, KAPPA)[..., 0]
    u0 = v0 - (1 + c) * f[..., 0]
    k = pr['var_xi'] / (pr['var_xi'] + pr['var_e'])
    mean1 = pr['rho'] * k * u0
    var1 = pr['rho'] ** 2 * pr['var_xi'] * (1 - k) + (1 - pr['rho'] ** 2) * pr['var_xi'] + pr['var_e']
    ff = np.stack([f[..., 0], f[..., 0]], axis=-1) if freeze_index else f
    m = cm_normal((1 + c) * ff, panel.params, spec.map, KAPPA, mean1[..., None], var1[..., None])
    return m[..., 1], v0


def cond_mean_given_baseline(panel, idx):
    """Oracle E[W_1 | x, W_0] for tracts idx ((C, m) indices): invert the true date-0 map to get the
    baseline latent residual u0 = v0 - (1+c) f0 = xi0 + e0, then v1 | info is normal with
    mean (1+c) f1 + rho * k * u0 and variance rho^2 var_xi (1-k) + (1-rho^2) var_xi + var_e,
    k = var_xi / (var_xi + var_e). Returns (C, m)."""
    return _given_baseline(panel, idx)[0]


def change_components(panel, idx):
    """Oracle decomposition of the predictable change E[dW | x, W_0] = map + struct + resid, (C, m) each:
      map    = psi_1(vbar0) - psi_0(vbar0), vbar0 = (1+c) f0 the tract's structural position at baseline:
               the change if only the city-date map moved (global: the same function of the index for every
               tract in the city);
      struct = E[W_1 | x, W_0] - E[W_1 | x, W_0, index frozen at date 0]: change caused by the structural
               index change (exactly 0 for stable tracts);
      resid  = the rest: expected movement of the tract's residual (mean reversion, persistence, the map
               acting on the residual, Jensen)."""
    total, _ = _given_baseline(panel, idx)
    frozen, _ = _given_baseline(panel, idx, freeze_index=True)
    W0 = take(panel.W, idx)[..., 0]
    f0 = take(panel.f, idx)[..., 0]
    c = persistence(panel.spec, f0, np.asarray(panel.stable)[np.asarray(idx)])['c'] if panel.spec.persist else 0.0
    vbar = (1 + c) * f0                                   # the tract's structural position at baseline
    psi_v = base.psi(np.stack([vbar, vbar], axis=-1), panel.spec.map, panel.params, KAPPA)
    mapc = psi_v[..., 1] - psi_v[..., 0]
    struct = total - frozen
    return dict(total=total - W0, map=mapc, struct=struct, resid=total - W0 - mapc - struct)


# --------------------------------------------------------------------------
# conditional means (closed form)
# --------------------------------------------------------------------------
def mgf_mixture(a):
    (p1, p2), (m1, m2), (s1, s2) = _SM['p'], _SM['m'], _SM['s']
    return p1 * np.exp(a * m1 + a * a * s1 * s1 / 2) + p2 * np.exp(a * m2 + a * a * s2 * s2 / 2)


def _alpha_beta_lam(params, ndim_like):
    return base._cell(params, ndim_like)


def cm_normal(s, params, family, kappa, mean=0.0, var=1.0):
    """E[psi_ct(s + eta)], eta ~ N(mean, var). s: (C, m, 2); mean/var broadcastable."""
    alpha, beta, lam = _alpha_beta_lam(params, s.ndim)
    s = s + mean
    if family == 'a':
        return alpha + s
    if family == 'b':
        return alpha + beta * s
    return alpha + beta * np.exp(kappa ** 2 * var / 2) * base.sinh_map(s, lam, kappa)


def cm_skew_scale(s, scale, params, family, kappa):
    """E[psi_ct(s + scale * SM)], SM the skewed unit-variance mean-zero mixture."""
    alpha, beta, lam = _alpha_beta_lam(params, s.ndim)
    if family == 'a':
        return alpha + s
    if family == 'b':
        return alpha + beta * s
    a = kappa * scale
    return alpha + beta * ((1 + lam) * np.exp(kappa * s) * mgf_mixture(a)
                           - (1 - lam) * np.exp(-kappa * s) * mgf_mixture(-a)) / (2 * kappa)


def outside_mean(spec, params):
    """E of the outside term of scenario D, per (city, date): (C,1,2)."""
    if spec.noise != 'D':
        return 0.0
    s = rdgp.LOGNORMAL_S
    sd = np.sqrt((np.exp(s * s) - 1) * np.exp(s * s))
    enu = (np.exp(s * s / 2) - 1.0) / sd
    return np.sqrt(0.5) * rdgp.OUTER_SD * params['beta'][:, None, :] * enu


def inside_var(spec):
    return 0.5 if spec.noise == 'D' else 1.0


def cond_mean(spec, params, s, a1c=None, nominal=False):
    """E[W | x] at index s (C,m,2) (a1c: standardised raw A1, needed for E4 unless nominal).
    nominal=True: the structural map's conditional mean given the index only (the ORACLE head:
    the x-dependent E4 error terms are not part of G)."""
    fam, kap = spec.map, KAPPA
    var = inside_var(spec)
    if spec.persist:                                   # E5: index (1+c) f, latent noise var_xi + var_e
        pr = persistence(spec, s[..., 0], np.ones(s.shape[1], bool))
        return cm_normal((1 + pr['c']) * s, params, fam, kap, 0.0, (pr['var_xi'] + pr['var_e'])[..., None])
    if nominal or not spec.e4:
        return cm_normal(s, params, fam, kap, 0.0, var) + outside_mean(spec, params)
    if spec.e4 == 'i':
        return cm_normal(s, params, fam, kap, E4_A_I * a1c, var)
    if spec.e4 == 'ii':
        return cm_normal(s, params, fam, kap, E4_GAMMA_O * E4_RHO_O * a1c,
                         var + E4_GAMMA_O ** 2 * (1 - E4_RHO_O ** 2))
    if spec.e4 == 'iii':
        return cm_skew_scale(s, np.exp(E4_A_III * a1c - E4_A_III ** 2 / 2), params, fam, kap)
    raise ValueError(spec.e4)


# --------------------------------------------------------------------------
# panel
# --------------------------------------------------------------------------
@dataclass
class Panel:
    spec: Spec
    C: int
    n: int
    z: np.ndarray                  # (C,n,2,p) whitened features
    W: np.ndarray                  # (C,n,2)
    xy: np.ndarray                 # (C,n,2)
    f: np.ndarray                  # (C,n,2) true standardised index (evaluation only)
    a1c: np.ndarray                # (C,n,2)
    stable: np.ndarray             # (n,) truly stable tracts
    params: dict
    theta0: np.ndarray
    twins: np.ndarray              # (T,4) ca, ia, cb, ib (true cross-city equal-index pairs)
    temporal_decl: np.ndarray      # (L,2) declared temporal links (c, i)
    cross_decl: np.ndarray         # (Lx,4) declared cross links
    n_fake: int = 0


def ring_neighbors(C):
    """Undirected edges (c, c+1), (c, c+2) of the ring graph (4 neighbours per city for C >= 5).
    Returns [(c, c', slot_c, slot_c')]: slot indices 0..3 in each city's twin block."""
    if C < 5:
        raise ValueError('cross-city graph needs C >= 5')
    return [(c, (c + 1) % C, 0, 1) for c in range(C)] + [(c, (c + 2) % C, 2, 3) for c in range(C)]


def twin_pairs(C, n, n_twin):
    need = 2 * 4 * n_twin
    if need > n:
        n_twin = max(1, n // 8)
    out = []
    for c, c2, sa, sb in ring_neighbors(C):
        for q in range(n_twin):
            out.append((c, 2 * (sa * n_twin + q), c2, 2 * (sb * n_twin + q)))
    return np.array(out, dtype=np.int64)


def make_panel(spec, C, n, key):
    """key: tuple of ints (deterministic per experiment/scenario/size/rep)."""
    ref = reference(spec)
    ctxs = [city_context(spec, *key, c) for c in range(C)]
    Hs, noises, stable = [], [], None
    for c in range(C):
        H, stable, noise = draw_H(spec, ctxs[c], n, rng(STREAMS['tract'], *key, c), with_drift=spec.fake > 0,
                                    approx_scale=approx_scale(spec))
        Hs.append(H); noises.append(noise)
    twins = twin_pairs(C, n, spec.n_twin)
    g_links = rng(STREAMS['links'], *key)
    coef2 = float(COEF @ COEF)
    for ca, ia, cb, ib in twins:
        Hb = Hs[ca][ia, 0].copy()
        if spec.hc:
            Hb = Hb + spec.gamma * (ctxs[ca]['hc'] - ctxs[cb]['hc']) / coef2 * COEF
        if spec.approx > 0:
            Hb = Hb + approx_scale(spec) * g_links.normal(size=2)
        Hs[cb][ib] = np.stack([Hb, Hb])
    raws = np.stack([assemble_raw(spec, ctxs[c], Hs[c], noises[c]) for c in range(C)])   # (C,n,2,p)
    cf = coef_full(spec)
    f = (raws @ cf - ref.gm) / ref.sigma_g
    # A1 (standardised raw nuisance coordinate) drives the E4 error terms only. It is heavy-tailed, and inside
    # the sinh map a few extreme tracts made E[W|x] explode (pseudo-true W RMSE 1e1-1e78): clip at +-2.5 SD.
    a1c = np.clip((raws[..., 2] - ref.a1_mean) / ref.a1_sd, -E4_A1_CLIP, E4_A1_CLIP)
    z = ref.whiten(raws @ ref.mix.T)
    xy = np.stack([rng(STREAMS['coords'], *key, c).random((n, 2)) for c in range(C)])
    if spec.persist:
        if spec.noise != 'gauss' or spec.e4 or spec.rho or spec.spatial_r:
            raise ValueError('persist is only defined with Gaussian inside noise and no E3/E4 terms')
        eta = np.stack([draw_persistent(spec, rng(STREAMS['resid'], *key, c), f[c, :, 0], stable)
                        for c in range(C)])
    else:
        eta = np.stack([draw_eta(spec, rng(STREAMS['resid'], *key, c), xy[c]) for c in range(C)])
    params = dict(alpha=np.stack([c['alpha'] for c in ctxs]), beta=np.stack([c['beta'] for c in ctxs]),
                  lam=np.array([c['lam'] for c in ctxs]))
    gE = rng(STREAMS['e4'], *key)
    inner = eta * np.sqrt(inside_var(spec))
    extra = np.zeros_like(eta)
    if spec.e4 == 'i':
        extra = E4_A_I * a1c
    elif spec.e4 == 'ii':
        o = E4_RHO_O * a1c + np.sqrt(1 - E4_RHO_O ** 2) * gE.normal(size=a1c.shape)
        extra = E4_GAMMA_O * o
    elif spec.e4 == 'iii':
        sm = rdgp.skewed_mixture(gE, eta.shape)
        inner = np.exp(E4_A_III * a1c - E4_A_III ** 2 / 2) * sm
    outer = 0.0
    if spec.noise == 'D':
        nu = rdgp.lognormal_noise(rng(STREAMS['extra'], *key, 7777), eta.shape)
        outer = np.sqrt(0.5) * rdgp.OUTER_SD * params['beta'][:, None, :] * nu
    c_x = persistence(spec, f[..., 0], stable)['c'] if spec.persist else 0.0
    W = base.psi((1 + c_x) * f + inner + extra, spec.map, params, KAPPA) + outer
    temporal, cross, n_fake = declare_links(spec, C, n, stable, twins, g_links)
    return Panel(spec, C, n, z, W, xy, f, a1c, stable, params, ref.theta0, twins, temporal, cross, n_fake)


def declare_links(spec, C, n, stable, twins, g):
    """Declared link sets (indices only; rows are built from training tracts later)."""
    temporal = np.array([(c, i) for c in range(C) for i in np.nonzero(stable)[0]], dtype=np.int64)
    cross = twins.copy()
    use_t, use_x = spec.links in ('both', 'temporal'), spec.links in ('both', 'cross')

    def thin(arr):
        if spec.declared >= 1.0 or len(arr) == 0:
            return arr
        k = max(1, int(round(spec.declared * len(arr))))
        return arr[np.sort(g.choice(len(arr), size=k, replace=False))]

    temporal, cross = thin(temporal) if use_t else temporal[:0], thin(cross) if use_x else cross[:0]
    n_fake = 0
    if spec.fake > 0:
        ratio = spec.fake / (1 - spec.fake)
        odd = np.nonzero(~stable)[0]
        kt = int(round(ratio * len(temporal)))
        if kt:
            fake_t = np.stack([g.integers(0, C, size=kt), g.choice(odd, size=kt)], axis=1)
            temporal = np.concatenate([temporal, fake_t])
        kx = int(round(ratio * len(cross)))
        if kx:
            ca = g.integers(0, C, size=kx)
            cb = (ca + g.choice([1, 2, C - 1, C - 2], size=kx)) % C
            fake_x = np.stack([ca, g.choice(odd, size=kx), cb, g.choice(odd, size=kx)], axis=1)
            cross = np.concatenate([cross, fake_x])
        n_fake = kt + kx
    return temporal, cross, n_fake


def split_tracts(panel, key, frac=0.8):
    """Per-city random tract split shared by both dates. Returns train (C,n_tr), test (C,n_te) indices."""
    C, n = panel.C, panel.n
    n_tr = int(round(frac * n))
    tr, te = [], []
    for c in range(C):
        perm = rng(STREAMS['split'], *key, c).permutation(n)
        tr.append(np.sort(perm[:n_tr])); te.append(np.sort(perm[n_tr:]))
    return np.stack(tr), np.stack(te)


def take(arr, idx):
    """arr (C,n,...) -> (C,len,...) rows idx (C,m)."""
    return np.take_along_axis(arr, idx.reshape(idx.shape + (1,) * (arr.ndim - 2)), 1)


def link_rows(panel, tr_idx):
    """Gap rows (L,p) of declared links with all tracts in the training set."""
    C, n = panel.C, panel.n
    in_tr = np.zeros((C, n), bool)
    np.put_along_axis(in_tr, tr_idx, True, 1)
    rows = []
    t = panel.temporal_decl
    if len(t):
        ok = in_tr[t[:, 0], t[:, 1]]
        t = t[ok]
        rows.append(panel.z[t[:, 0], t[:, 1], 1] - panel.z[t[:, 0], t[:, 1], 0])
    x = panel.cross_decl
    if len(x):
        ok = in_tr[x[:, 0], x[:, 1]] & in_tr[x[:, 2], x[:, 3]]
        x = x[ok]
        for d in (0, 1):
            rows.append(panel.z[x[:, 0], x[:, 1], d] - panel.z[x[:, 2], x[:, 3], d])
    return np.concatenate(rows) if rows else np.zeros((0, panel.z.shape[-1]))
