"""Streaming pair kernels (full batch, O(n p) memory per cell).

Every kernel works on cells whose observations are SORTED by outcome (ascending), so
every pair i < j with W_j > W_i is concordant (y = +1) and the pair difference is
v = z_j - z_i; tied pairs are dropped. The pair weight is  W0[k] * omega_ij  with
omega_ij = 1 (default) or 1 - exp(-d_ij^2 / (2 ell^2)) (distance weighting, ell > 0),
W0[k] = 1 / (K * sum of omega over the cell's concordant pairs): the ranking loss is a MEAN
over pairs within a cell and a mean over cells (same convention as losses.PairData).

Two implementations of each kernel with identical signatures:
  nb_*  plain loops over numpy arrays (numba @njit, prange over cells; if numba is not
        installed `njit` is the identity and these run as slow pure Python: only used in tests)
  np_*  vectorised numpy, row-blocked so that no more than `block_rows(n) * n` pair
        entries are ever alive (memory budget MEM_GB), used automatically when numba is absent.
The public dispatchers (rank_vg, rank_curv, sur_vg, sur_angles, pair_norm) pick the backend.
Optional fixed pair subsample ("incomplete U-statistic", off by default) is handled by the
explicit-pair numpy kernels (sub_*).
"""
import math
import os

import numpy as np
from scipy.special import expit

try:                                                    # numba is optional
    if os.environ.get('MC_NO_NUMBA'):
        raise ImportError
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:                                     # pragma: no cover (depends on box)
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f
    prange = range

BACKEND = 'numba' if HAVE_NUMBA else 'numpy'
MEM_GB = 2.0                  # per-worker budget for pair temporaries (set by the runner)
ROW_CAP = 64                  # rows per block upper cap (limits the wasted triangle)
ROW_OVERRIDE = None           # tests: force tiny blocks
N_TEMP = 10                   # live (rows x n) float64 temporaries in the numpy kernels


def set_backend(name):
    global BACKEND
    if name == 'numba' and not HAVE_NUMBA:
        raise RuntimeError('numba not installed')
    BACKEND = name


def set_threads(n):
    if HAVE_NUMBA:
        import numba
        numba.set_num_threads(max(1, min(int(n), numba.config.NUMBA_NUM_THREADS)))


def block_rows(n):
    """Rows per block so that rows * n * 8B * N_TEMP <= 25% of the memory budget."""
    if ROW_OVERRIDE is not None:
        return int(ROW_OVERRIDE)
    by_mem = int(0.25 * MEM_GB * 1e9 / (8.0 * N_TEMP * max(n, 1)))
    return int(max(1, min(ROW_CAP, by_mem)))


# ---------------------------------------------------------------------------
# scalar helpers (numba friendly)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _lse(x):
    if x > 0.0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


@njit(cache=True)
def _sig(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@njit(cache=True)
def _omega(XY, k, i, j, ell):
    if ell > 0.0:
        dx = XY[k, i, 0] - XY[k, j, 0]
        dy = XY[k, i, 1] - XY[k, j, 1]
        return 1.0 - math.exp(-(dx * dx + dy * dy) / (2.0 * ell * ell))
    return 1.0


# ---------------------------------------------------------------------------
# numba-style kernels (loops)
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def nb_pair_norm(Ws, XY, ell, out):
    K, n = Ws.shape
    for k in prange(K):
        tot = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if Ws[k, j] > Ws[k, i]:
                    tot += _omega(XY, k, i, j, ell)
        out[k] = tot


@njit(parallel=True, cache=True)
def nb_rank_vg(S, Ws, XY, ell, W0, M, val, sgt, gs):
    """Per cell: val = sum w log(1+exp(-t)), sgt = sum w sigma(-t) t, gs = d val / d S,
    t = M_k (S_j - S_i)."""
    K, n = S.shape
    for k in prange(K):
        v = 0.0
        st = 0.0
        for i in range(n):
            gs[k, i] = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if Ws[k, j] > Ws[k, i]:
                    w = W0[k] * _omega(XY, k, i, j, ell)
                    t = M[k] * (S[k, j] - S[k, i])
                    v += w * _lse(-t)
                    sg = _sig(-t)
                    st += w * sg * t
                    c = w * sg * M[k]
                    gs[k, j] -= c
                    gs[k, i] += c
        val[k] = v
        sgt[k] = st


@njit(parallel=True, cache=True)
def nb_rank_curv(S, Z, Ws, XY, ell, W0, M, r, CZ):
    """Laplacian pieces of the Hessian: sum_pairs c_ij dz dz' = Z'(diag r - C) Z,
    c_ij = w M^2 sigma(1-sigma). r_i = sum_j c_ij ; CZ_i = sum_j c_ij z_j."""
    K, n = S.shape
    p = Z.shape[2]
    for k in prange(K):
        for i in range(n):
            r[k, i] = 0.0
            for a in range(p):
                CZ[k, i, a] = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if Ws[k, j] > Ws[k, i]:
                    w = W0[k] * _omega(XY, k, i, j, ell)
                    t = M[k] * (S[k, j] - S[k, i])
                    sg = _sig(-t)
                    c = w * M[k] * M[k] * sg * (1.0 - sg)
                    r[k, i] += c
                    r[k, j] += c
                    for a in range(p):
                        CZ[k, i, a] += c * Z[k, j, a]
                        CZ[k, j, a] += c * Z[k, i, a]


@njit(parallel=True, cache=True)
def nb_sur_vg(S, Ws, XY, ell, W0, TAU, val, gs):
    """SUR = sum w sigmoid(-(S_j - S_i)/tau_k); gs = d val / d S."""
    K, n = S.shape
    for k in prange(K):
        v = 0.0
        for i in range(n):
            gs[k, i] = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if Ws[k, j] > Ws[k, i]:
                    w = W0[k] * _omega(XY, k, i, j, ell)
                    sg = _sig(-(S[k, j] - S[k, i]) / TAU[k])
                    v += w * sg
                    c = w * sg * (1.0 - sg) / TAU[k]
                    gs[k, j] -= c
                    gs[k, i] += c
        val[k] = v


@njit(parallel=True, cache=True)
def nb_sur_angles(Z2, Ws, XY, ell, W0, TAU, angles, out):
    """SUR value along theta = (cos a, sin a) for every angle; out[k, g]."""
    K, n = Ws.shape
    G = angles.shape[0]
    for k in prange(K):
        sc = np.empty(n)
        for g in range(G):
            ca = math.cos(angles[g])
            sa = math.sin(angles[g])
            for i in range(n):
                sc[i] = Z2[k, i, 0] * ca + Z2[k, i, 1] * sa
            v = 0.0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if Ws[k, j] > Ws[k, i]:
                        w = W0[k] * _omega(XY, k, i, j, ell)
                        v += w * _sig(-(sc[j] - sc[i]) / TAU[k])
            out[k, g] = v


# ---------------------------------------------------------------------------
# numpy kernels (row-blocked)
# ---------------------------------------------------------------------------
def _blocks(n):
    b = block_rows(n)
    for i0 in range(0, n - 1, b):
        yield i0, min(n - 1, i0 + b)


def _wmat(Ws_k, XY_k, ell, i0, i1, W0k):
    """(rows, n-i0-1) pair weights for rows i0..i1-1 against columns i0+1..n-1."""
    mask = Ws_k[None, i0 + 1:] > Ws_k[i0:i1, None]
    w = mask * W0k
    if ell > 0.0:
        dxy = XY_k[None, i0 + 1:, :] - XY_k[i0:i1, None, :]
        w = w * (1.0 - np.exp(-(dxy[..., 0] ** 2 + dxy[..., 1] ** 2) / (2.0 * ell * ell)))
    return w


def np_pair_norm(Ws, XY, ell, out):
    for k in range(Ws.shape[0]):
        tot = 0.0
        for i0, i1 in _blocks(Ws.shape[1]):
            tot += _wmat(Ws[k], XY[k], ell, i0, i1, 1.0).sum()
        out[k] = tot


def np_rank_vg(S, Ws, XY, ell, W0, M, val, sgt, gs):
    K, n = S.shape
    for k in range(K):
        v = st = 0.0
        g = np.zeros(n)
        for i0, i1 in _blocks(n):
            w = _wmat(Ws[k], XY[k], ell, i0, i1, W0[k])
            t = M[k] * (S[k, None, i0 + 1:] - S[k, i0:i1, None])
            sg = expit(-t)
            v += float((w * np.logaddexp(0.0, -t)).sum())
            ws = w * sg
            st += float((ws * t).sum())
            c = ws * M[k]
            g[i0 + 1:] -= c.sum(0)
            g[i0:i1] += c.sum(1)
        val[k], sgt[k], gs[k] = v, st, g


def np_rank_curv(S, Z, Ws, XY, ell, W0, M, r, CZ):
    K, n = S.shape
    for k in range(K):
        rk = np.zeros(n)
        cz = np.zeros((n, Z.shape[2]))
        for i0, i1 in _blocks(n):
            w = _wmat(Ws[k], XY[k], ell, i0, i1, W0[k])
            t = M[k] * (S[k, None, i0 + 1:] - S[k, i0:i1, None])
            sg = expit(-t)
            c = w * (M[k] ** 2) * sg * (1.0 - sg)
            rk[i0:i1] += c.sum(1)
            rk[i0 + 1:] += c.sum(0)
            cz[i0:i1] += c @ Z[k, i0 + 1:]
            cz[i0 + 1:] += c.T @ Z[k, i0:i1]
        r[k], CZ[k] = rk, cz


def np_sur_vg(S, Ws, XY, ell, W0, TAU, val, gs):
    K, n = S.shape
    for k in range(K):
        v = 0.0
        g = np.zeros(n)
        for i0, i1 in _blocks(n):
            w = _wmat(Ws[k], XY[k], ell, i0, i1, W0[k])
            sg = expit(-(S[k, None, i0 + 1:] - S[k, i0:i1, None]) / TAU[k])
            v += float((w * sg).sum())
            c = w * sg * (1.0 - sg) / TAU[k]
            g[i0 + 1:] -= c.sum(0)
            g[i0:i1] += c.sum(1)
        val[k], gs[k] = v, g


def np_sur_angles(Z2, Ws, XY, ell, W0, TAU, angles, out):
    """Angles are processed in groups so that rows*n*group stays within the budget."""
    K, n = Ws.shape
    G = len(angles)
    cs = np.stack([np.cos(angles), np.sin(angles)])            # (2, G)
    for k in range(K):
        sc = Z2[k] @ cs                                          # (n, G)
        acc = np.zeros(G)
        for i0, i1 in _blocks(n):
            w = _wmat(Ws[k], XY[k], ell, i0, i1, W0[k])          # (b, m)
            if not w.any():
                continue
            rr, mm = w.shape
            grp = max(1, int(0.25 * MEM_GB * 1e9 / (8.0 * 3 * max(rr * mm, 1))))
            grp = min(grp, G) if ROW_OVERRIDE is None else max(1, min(G, 7))
            for g0 in range(0, G, grp):
                d = sc[None, i0 + 1:, g0:g0 + grp] - sc[i0:i1, None, g0:g0 + grp]
                acc[g0:g0 + grp] += np.einsum('rm,rmg->g', w, expit(-d / TAU[k]))
        out[k] = acc


# ---------------------------------------------------------------------------
# optional fixed pair subsample (explicit pair lists; numpy only)
# ---------------------------------------------------------------------------
def draw_pairs(Ws, max_pairs, rng):
    """Fixed uniform subsample of index pairs i<j per cell (cells with fewer pairs keep all).
    Returns list of (ii, jj) int32 arrays."""
    K, n = Ws.shape
    iu, ju = np.triu_indices(n, 1)
    out = []
    for k in range(K):
        if len(iu) <= max_pairs:
            sel = np.arange(len(iu))
        else:
            sel = np.sort(rng.choice(len(iu), size=max_pairs, replace=False))
        out.append((iu[sel].astype(np.int32), ju[sel].astype(np.int32)))
    return out


def _sub_w(Ws_k, ii, jj, W0k):
    return (Ws_k[jj] > Ws_k[ii]) * W0k


def sub_rank_vg(S, Ws, PAIRS, W0, M, val, sgt, gs):
    for k, (ii, jj) in enumerate(PAIRS):
        w = _sub_w(Ws[k], ii, jj, W0[k])
        t = M[k] * (S[k, jj] - S[k, ii])
        sg = expit(-t)
        val[k] = float((w * np.logaddexp(0.0, -t)).sum())
        sgt[k] = float((w * sg * t).sum())
        c = w * sg * M[k]
        gs[k] = np.bincount(jj, -c, S.shape[1]) + np.bincount(ii, c, S.shape[1])


def sub_rank_curv(S, Z, Ws, PAIRS, W0, M, r, CZ):
    for k, (ii, jj) in enumerate(PAIRS):
        w = _sub_w(Ws[k], ii, jj, W0[k])
        t = M[k] * (S[k, jj] - S[k, ii])
        sg = expit(-t)
        c = w * M[k] ** 2 * sg * (1 - sg)
        n = S.shape[1]
        r[k] = np.bincount(ii, c, n) + np.bincount(jj, c, n)
        cz = np.zeros((n, Z.shape[2]))
        np.add.at(cz, ii, c[:, None] * Z[k, jj])
        np.add.at(cz, jj, c[:, None] * Z[k, ii])
        CZ[k] = cz


def sub_sur_vg(S, Ws, PAIRS, W0, TAU, val, gs):
    for k, (ii, jj) in enumerate(PAIRS):
        w = _sub_w(Ws[k], ii, jj, W0[k])
        sg = expit(-(S[k, jj] - S[k, ii]) / TAU[k])
        val[k] = float((w * sg).sum())
        c = w * sg * (1 - sg) / TAU[k]
        gs[k] = np.bincount(jj, -c, S.shape[1]) + np.bincount(ii, c, S.shape[1])


def sub_sur_angles(Z2, Ws, PAIRS, W0, TAU, angles, out):
    cs = np.stack([np.cos(angles), np.sin(angles)])
    for k, (ii, jj) in enumerate(PAIRS):
        w = _sub_w(Ws[k], ii, jj, W0[k])
        d = (Z2[k, jj] - Z2[k, ii]) @ cs
        out[k] = w @ expit(-d / TAU[k])


# ---------------------------------------------------------------------------
# dispatchers on a CellSet (see cells.py)
# ---------------------------------------------------------------------------
def _impl(name):
    return globals()[('nb_' if BACKEND == 'numba' else 'np_') + name]


def pair_norm(Ws, XY, ell):
    out = np.zeros(Ws.shape[0])
    _impl('pair_norm')(np.ascontiguousarray(Ws), XY, float(ell), out)
    return out


def rank_vg(cs, S, M):
    """-> (value, per-cell sum w sigma(-t) t, gs (K,n))."""
    K, n = S.shape
    val, sgt, gs = np.zeros(K), np.zeros(K), np.zeros((K, n))
    M = np.ascontiguousarray(M, float)
    if cs.pairs is not None:
        sub_rank_vg(S, cs.Ws, cs.pairs, cs.W0, M, val, sgt, gs)
    else:
        _impl('rank_vg')(np.ascontiguousarray(S), cs.Ws, cs.XY, cs.ell, cs.W0, M, val, sgt, gs)
    return float(val.sum()), sgt, gs


def rank_curv(cs, S, Z, M):
    """Euclidean Hessian (p x p) of the cell-temperature RankNet loss wrt a score vector
    u with scores S = Z u:  sum_k Z_k'(diag r_k - C_k) Z_k."""
    K, n = S.shape
    p = Z.shape[2]
    r, CZ = np.zeros((K, n)), np.zeros((K, n, p))
    M = np.ascontiguousarray(M, float)
    if cs.pairs is not None:
        sub_rank_curv(S, Z, cs.Ws, cs.pairs, cs.W0, M, r, CZ)
    else:
        _impl('rank_curv')(np.ascontiguousarray(S), np.ascontiguousarray(Z), cs.Ws, cs.XY,
                           cs.ell, cs.W0, M, r, CZ)
    return np.einsum('kia,ki,kib->ab', Z, r, Z) - np.einsum('kia,kib->ab', Z, CZ)


def sur_vg(cs, S, tau):
    K, n = S.shape
    val, gs = np.zeros(K), np.zeros((K, n))
    tau = np.ascontiguousarray(np.broadcast_to(np.asarray(tau, float), (K,)))
    if cs.pairs is not None:
        sub_sur_vg(S, cs.Ws, cs.pairs, cs.W0, tau, val, gs)
    else:
        _impl('sur_vg')(np.ascontiguousarray(S), cs.Ws, cs.XY, cs.ell, cs.W0, tau, val, gs)
    return float(val.sum()), gs


def sur_angles(cs, Z2, tau, angles):
    """Total SUR value at each angle (Z2: (K,n,2) projected scores)."""
    K = Z2.shape[0]
    tau = np.ascontiguousarray(np.broadcast_to(np.asarray(tau, float), (K,)))
    angles = np.ascontiguousarray(angles, float)
    out = np.zeros((K, len(angles)))
    if cs.pairs is not None:
        sub_sur_angles(Z2, cs.Ws, cs.pairs, cs.W0, tau, angles, out)
    else:
        _impl('sur_angles')(np.ascontiguousarray(Z2), cs.Ws, cs.XY, cs.ell, cs.W0, tau, angles, out)
    return out.sum(0)


# ---------------------------------------------------------------------------
# Exact maximum rank correlation in 2-d, streamed (two-pass histogram + exact sweep)
# ---------------------------------------------------------------------------
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _events_from_v(v):
    om = np.arctan2(v[:, 1], v[:, 0])
    return _wrap(om - np.pi / 2), _wrap(om + np.pi / 2), v[:, 0] < 0


def _iter_event_chunks(cs, Z2):
    """Yield (k, start, end, at_minus_pi) chunks covering every concordant pair once. A pair
    with difference v is concordant on the open half circle centred at atan2(v)."""
    K, n = cs.Ws.shape
    if cs.pairs is not None:
        for k, (ii, jj) in enumerate(cs.pairs):
            m = cs.Ws[k][jj] > cs.Ws[k][ii]
            s, e, f = _events_from_v(Z2[k][jj[m]] - Z2[k][ii[m]])
            yield k, s, e, f
        return
    for k in range(K):
        for i0, i1 in _blocks(n):
            ii, jj = np.nonzero(cs.Ws[k][None, i0 + 1:] > cs.Ws[k][i0:i1, None])
            s, e, f = _events_from_v(Z2[k][jj + i0 + 1] - Z2[k][ii + i0])
            yield k, s, e, f


def mrc_exact_stream(cs, Z2, bins=1 << 20, max_events=4_000_000):
    """argmax_phi sum_k W0_k #{concordant pairs of cell k at theta=(cos phi, sin phi)} for the
    2-d coordinates Z2 (K, n, 2): exact global maximum with O(bins + candidate events) memory.

    Pass 1 histograms arc starts/ends into `bins` angular bins. The level at every bin
    boundary is then exact, and inside a bin the level cannot exceed boundary level + (sum
    of starts in the bin). Pass 2 re-streams the pairs, keeps only events of bins that can
    still contain the global maximum and sweeps them exactly. phi = midpoint of the first
    (in angular order) maximal piece (pieces are clipped at bin edges, which only matters
    for which point of a flat maximiser interval is returned).
    Returns (phi, level, info)."""
    wk = np.asarray(cs.W0, float)
    unit = bool(np.allclose(wk, wk[0], rtol=1e-12, atol=0.0))
    scale = float(wk[0]) if unit else 1.0
    wt = (lambda k: 1.0) if unit else (lambda k: float(wk[k]))
    width = 2 * np.pi / bins
    binof = lambda ev: np.minimum(((ev + np.pi) / width).astype(np.int64), bins - 1)
    pos, neg = np.zeros(bins), np.zeros(bins)
    initial = 0.0
    for k, s, e, f in _iter_event_chunks(cs, Z2):
        w = wt(k)
        initial += w * float(f.sum())
        pos += w * np.bincount(binof(s), minlength=bins)
        neg += w * np.bincount(binof(e), minlength=bins)
    start_level = initial + np.concatenate([[0.0], np.cumsum(pos - neg)[:-1]])
    end_level = start_level + pos - neg
    lb = max(float(end_level.max()), float(initial))
    tol = 0.0 if unit else 1e-12 * max(1.0, abs(lb))
    cand = (start_level + pos) >= lb - tol
    ev_pos, ev_delta, count = [], [], 0
    for k, s, e, f in _iter_event_chunks(cs, Z2):
        w = wt(k)
        for ev, sign in ((s, 1.0), (e, -1.0)):
            keep = cand[binof(ev)]
            if keep.any():
                ev_pos.append(ev[keep])
                ev_delta.append(np.full(int(keep.sum()), sign * w))
                count += int(keep.sum())
        if count > max_events:
            raise MemoryError('MRC candidate events exceed budget: raise `bins`')
    cb = np.nonzero(cand)[0]
    left_b = cb * width - np.pi
    right_b = np.minimum((cb + 1) * width - np.pi, np.pi)
    # one 'start' piece per candidate bin: [bin start, first event in bin or bin end)
    p_left, p_right, p_level = [left_b], [right_b.copy()], [start_level[cb]]
    if ev_pos:
        ep = np.concatenate(ev_pos)
        ed = np.concatenate(ev_delta)
        order = np.argsort(ep, kind='mergesort')
        ep, ed = ep[order], ed[order]
        eb = binof(ep)
        csd = np.cumsum(ed)
        first = np.r_[True, eb[1:] != eb[:-1]]
        grp0 = np.maximum.accumulate(np.where(first, np.arange(len(eb)), 0))
        level_after = start_level[eb] + (csd - (csd - ed)[grp0])
        same_next = np.r_[eb[1:] == eb[:-1], False]
        nxt = np.r_[ep[1:], 0.0]
        right_e = np.where(same_next, nxt, np.minimum((eb + 1) * width - np.pi, np.pi))
        # shorten start pieces of bins that contain events
        idx = np.searchsorted(cb, eb[first])
        p_right[0][idx] = ep[first]
        p_left.append(ep); p_right.append(right_e); p_level.append(level_after)
    left, right, level = np.concatenate(p_left), np.concatenate(p_right), np.concatenate(p_level)
    level = np.where(right - left > 1e-15, level, -np.inf)
    top = level.max()
    ties = np.nonzero(level >= top - tol)[0]
    j = ties[np.argmin(left[ties])]
    phi = float(_wrap(0.5 * (left[j] + right[j])))
    return phi, float(level[j]) * scale, dict(bins=bins, candidate_bins=int(cand.sum()), events=count)
