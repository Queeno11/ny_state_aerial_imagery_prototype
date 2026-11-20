"""Mixed observed features, paired residuals, maps and external restrictions."""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.special import expit, logsumexp, roots_hermitenorm, roots_laguerre


def rng(seed, stream, *indices):
    return np.random.default_rng(np.random.SeedSequence([seed, stream, *indices]))


def mixing(seed=91073):
    q, r = np.linalg.qr(rng(seed, 1).normal(size=(5, 5)))
    return q * np.sign(np.diag(r))


def raw_basis(x, truth):
    if truth == 'linear':
        return x
    return np.concatenate([x, np.stack([x[..., j]*x[..., k]
                           for j in range(5) for k in range(j, 5)], axis=-1)], axis=-1)


def raw_signal(h, truth):
    g = h[..., 0] + .7*h[..., 1] - .5*h[..., 2]
    if truth == 'quadratic':
        g = g + .35*h[..., 0]**2 + .25*h[..., 0]*h[..., 1] - .2*h[..., 1]*h[..., 2]
    return g


def city_context(seed, stream, c):
    g = rng(seed, stream, c)
    dc = g.normal(size=3)*.5
    da = .5*dc[:2] + g.normal(size=2)*.5
    alpha = .5*dc[0] + .25*dc[1] + .5*g.normal() + .2*np.arange(2) + .1*g.normal(size=2)
    beta = np.exp(.3*g.normal() + .1*g.normal(size=2) + .15*np.arange(2))
    k = .5 + .2*np.tanh(g.normal() + .25*np.arange(2))
    return dict(dc=dc, da=da, alpha=alpha, beta=beta, k=k)


def nuisance(h, da, g, paired=True):
    bh = np.stack([.8*h[..., 0]+.4*h[..., 1], -.3*h[..., 1]+.7*h[..., 2]], axis=-1)
    if paired:
        return (bh + da + .7*g.normal(size=(len(h), 1, 2))
                + .7*g.normal(size=h.shape[:-1]+(2,))
                + .3*np.arange(2)[None, :, None]*np.array([1., -1.]))
    return bh + da + np.sqrt(.98)*g.normal(size=h.shape[:-1]+(2,))


def features(context, n, g, mix):
    # Fixed counts: floor(.5n), floor(.25n), remainder large; random assignment.
    groups = np.repeat(np.arange(3), [n//2, n//4, n-n//2-n//4])
    g.shuffle(groups)
    h0 = context['dc'] + g.normal(size=(n, 3))
    h1 = h0 + np.array([0., .1, .5])[groups, None]*g.normal(size=(n, 3))
    h = np.stack([h0, h1], axis=1)
    a = nuisance(h, context['da'], g)
    return np.concatenate([h, a], axis=-1) @ mix.T, h, groups


@dataclass
class Reference:
    truth: str
    mean: np.ndarray
    chol: np.ndarray
    signal_mean: float
    signal_sd: float
    true_coef: np.ndarray  # Evaluation only: never passed to a fitting routine.
    mix: np.ndarray

    def transform(self, x):
        b = raw_basis(x, self.truth)-self.mean
        return np.linalg.solve(self.chol, b.reshape(-1, b.shape[-1]).T).T.reshape(b.shape)

    def signal(self, h):
        return (raw_signal(h, self.truth)-self.signal_mean)/self.signal_sd


@lru_cache(maxsize=8)
def reference(truth='linear', size=200000, seed=91073):
    mix = mixing(seed)
    # Equal endpoints, independent reference cities, no labels. Same environment
    # law as the estimation panel; 500 trajectories/reference city.
    xs, hs = [], []
    remaining = max(2, size//2)
    c = 0
    while remaining:
        n = min(500, remaining)
        x, h, _ = features(city_context(seed, 2, c), n, rng(seed, 3, c), mix)
        xs.append(x.reshape(-1, 5)); hs.append(h.reshape(-1, 3))
        remaining -= n; c += 1
    x, h = np.concatenate(xs), np.concatenate(hs)
    b = raw_basis(x, truth)
    mean = b.mean(0)
    centered = b-mean
    chol = np.linalg.cholesky(centered.T@centered/len(b))
    g = raw_signal(h, truth)
    gm, gs = float(g.mean()), float(g.std())
    # Recover the coefficient mapping for evaluation, using only this reference.
    z = np.linalg.solve(chol, centered.T).T
    coef = np.linalg.lstsq(z, (g-gm)/gs, rcond=None)[0]
    if np.max(np.abs(z@coef-(g-gm)/gs)) > 1e-9:
        raise ArithmeticError('Truth not in centered fitted basis')
    return Reference(truth, mean, chol, gm, gs, coef, mix)


def residuals(g, shape, law, rho, sigma=1.):
    if law == 'gumbel' and rho != 0:
        raise ValueError('The Gumbel diagnostic requires rho=0')
    u = g.normal(size=shape[:-1]+(1,))
    if law == 'mixture':
        label = g.random(shape) >= .8
        eps = (np.where(label, 1., -.25)+np.where(label, 1., .5)*g.normal(size=shape))/np.sqrt(.65)
    elif law == 'gaussian':
        eps = g.normal(size=shape)
    elif law == 'gumbel':
        eps = (g.gumbel(size=shape)-np.euler_gamma)*np.sqrt(6)/np.pi
    else:
        raise ValueError(law)
    return sigma*(np.sqrt(rho)*u + np.sqrt(1-rho)*eps)


def map_values(s, alpha, beta, k):
    return alpha + beta*(s+k*np.tanh(s))


def inverse_map(w, alpha, beta, k):
    y = (w-alpha)/beta
    # Monotone bracket r in [y-k,y+k]. Safeguarded Newton, differentiable
    # derivatives used explicitly by the likelihood rather than through iterations.
    lo, hi = y-k, y+k
    r = y.copy()
    for _ in range(45):
        th = np.tanh(r)
        err = r+k*th-y
        lo = np.where(err < 0, r, lo); hi = np.where(err > 0, r, hi)
        new = r-err/(1+k*(1-th*th))
        r = np.where((new >= lo) & (new <= hi), new, (lo+hi)/2)
        if np.max(np.abs(err)) < 2e-13:
            break
    return r


@lru_cache(maxsize=64)
def quadrature(law, rho, sigma=1., order=48):
    if law == 'gumbel':
        z, w = roots_laguerre(order)
        return sigma*np.sqrt(6)/np.pi*(-np.log(z)-np.euler_gamma), w
    z, w = roots_hermitenorm(order)
    w = w/np.sqrt(2*np.pi)
    if law == 'gaussian':
        return sigma*z, w
    m = np.array([-.25, 1.])/np.sqrt(.65)
    v = np.array([.25, 1.])/.65
    nodes = sigma*(np.sqrt(1-rho)*m[:, None]+np.sqrt(rho+(1-rho)*v[:, None])*z)
    return nodes.ravel(), (np.array([.8, .2])[:, None]*w).ravel()


def h_mean(s, law, rho, sigma=1., order=48):
    nodes, weights = quadrature(law, float(rho), float(sigma), order)
    flat = np.asarray(s).ravel()
    value, derivative = np.empty_like(flat), np.empty_like(flat)
    for start in range(0, len(flat), 4096):
        th = np.tanh(flat[start:start+4096, None]+nodes)
        value[start:start+4096] = th@weights
        derivative[start:start+4096] = (1-th*th)@weights
    return value.reshape(s.shape), derivative.reshape(s.shape)


def conditional_mean(s, alpha, beta, k, law, rho, sigma=1., order=48):
    if not np.any(k):
        return alpha+beta*s
    h, _ = h_mean(s, law, rho, sigma, order)
    return alpha+beta*(s+k*h)


def paired_logpdf(e, law, rho, sigma=1.):
    """Exact paired log density and derivative with respect to both residuals."""
    if law == 'gumbel':
        scale = sigma*np.sqrt(6)/np.pi
        z = e/scale + np.euler_gamma
        ez = np.exp(np.minimum(-z, 700))
        return (-z-ez-np.log(scale)).sum(-1), (ez-1)/scale
    if law == 'gaussian':
        means, variances, weights = [0.], [1.], [1.]
    else:
        means = np.array([-.25, 1.])/np.sqrt(.65)
        variances = np.array([.25, 1.])/.65
        weights = [.8, .2]
    logs, grads = [], []
    for a in range(len(weights)):
        for b in range(len(weights)):
            mean = sigma*np.sqrt(1-rho)*np.array([means[a], means[b]])
            cov = sigma**2*np.array([[rho+(1-rho)*variances[a], rho],
                                      [rho, rho+(1-rho)*variances[b]]])
            precision = np.linalg.inv(cov)
            dif = e-mean
            grad = -dif@precision
            log = (np.log(weights[a]*weights[b])-np.log(2*np.pi)
                   -.5*np.linalg.slogdet(cov)[1]+.5*np.sum(dif*grad, axis=-1))
            logs.append(log); grads.append(grad)
    logs = np.stack(logs, -1)
    lp = logsumexp(logs, axis=-1)
    post = np.exp(logs-lp[..., None])
    return lp, np.sum(post[..., None]*np.stack(grads, -2), axis=-2)


def panel(ref, cfg, rep=0, kind='source', n=None):
    """Evaluation truth kept in this object; fitting consumes only Z,W."""
    stream = {'source': 10, 'holdout': 10, 'target_test': 30, 'target_cal': 30}[kind]
    count = cfg['cities'] if kind in ('source', 'holdout') else cfg['target_cities']
    n = n or (cfg['n'] if kind == 'source' else cfg['source_test_n'] if kind == 'holdout'
              else cfg['test_n'] if kind == 'target_test' else cfg['calibration_n'])
    context_rep = rep if stream == 10 else 0
    draw_stream = {'source': 11, 'holdout': 14, 'target_test': 31, 'target_cal': 34}[kind]
    # Calibration size variants are literal prefixes of a common pool. Source n
    # variants use disjoint streams per size (declared in deviations).
    draw_n = max(cfg.get('calibration_sizes', [n])) if kind == 'target_cal' else n
    xs, hs, gs, es, contexts = [], [], [], [], []
    for c in range(count):
        context = city_context(cfg['seed'], stream, context_rep*100000+c)
        x, h, group = features(context, draw_n, rng(cfg['seed'], draw_stream, rep, c), ref.mix)
        eta = residuals(rng(cfg['seed'], draw_stream+1, rep, c), (draw_n, 2), cfg['law'], cfg['rho'])
        xs.append(x[:n]); hs.append(h[:n]); gs.append(group[:n]); es.append(eta[:n]); contexts.append(context)
    x, h, group, eta = map(np.stack, (xs, hs, gs, es))
    alpha = np.stack([v['alpha'] for v in contexts])[:, None, :]
    beta = np.stack([v['beta'] for v in contexts])[:, None, :] if cfg['map'] != 'M1' else np.ones((count, 1, 2))
    k = np.stack([v['k'] for v in contexts])[:, None, :] if cfg['map'] == 'M3' else np.zeros((count, 1, 2))
    f = ref.signal(h)
    w = map_values(f+eta, alpha, beta, k)
    mu = conditional_mean(f, alpha, beta, k, cfg['law'], cfg['rho'], order=cfg['quadrature'])
    return dict(x=x, h=h, group=group, eta=eta, contexts=contexts, z=ref.transform(x),
                f=f, r=f+eta, w=w, mu=mu, alpha=alpha, beta=beta, k=k)


def links(ref, cfg, data, rep):
    """External restrictions. Corruption never changes a labeled endpoint."""
    cnum, n = data['group'].shape
    temporal, cross = [], []
    gaps = {'temporal': [], 'cross': []}
    corrupt = cfg.get('corrupt', 'both')
    delta, bad = cfg.get('delta', 0.), cfg.get('bad', 0.)
    def right_view(h, context, g, active):
        if active:
            scale = np.ones(3) if cfg.get('isotropic', False) else np.array([1., .5, .25])
            h = h + delta*scale*g.normal(size=h.shape)
            wrong = g.random(len(h)) < bad
            h = np.where(wrong[:, None], context['dc']+g.normal(size=h.shape), h)
        a = nuisance(h, context['da'], g, paired=False)
        return np.concatenate([h, a], -1)@ref.mix.T, h
    for c, context in enumerate(data['contexts']):
        stable = data['group'][c] == 0
        left = data['x'][c, stable, 0]
        hleft = data['h'][c, stable, 0]
        if (delta or bad) and corrupt in ('both', 'temporal'):
            right, hr = right_view(hleft, context, rng(cfg['seed'], 21, rep, c), True)
        else:
            right, hr = data['x'][c, stable, 1], hleft
        temporal.append(ref.transform(right)-ref.transform(left))
        gaps['temporal'].extend((ref.signal(hr)-ref.signal(hleft)).tolist())
    # Randomly permuted cycle with +/-1,+/-2 neighbors: connected four-regular.
    # Random relabeling is explicit; this is not a uniform random regular graph.
    perm = rng(cfg['seed'], 22, rep).permutation(cnum)
    edges = sorted({tuple(sorted((int(perm[c]), int(perm[(c+d)%cnum]))))
                    for c in range(cnum) for d in (1, 2) if c != (c+d)%cnum})
    adjacency = np.zeros((cnum, cnum))
    for edge_index, (c, d) in enumerate(edges):
        adjacency[c, d] = adjacency[d, c] = 1
        g = rng(cfg['seed'], 23, rep, edge_index)
        m = max(1, int(np.floor(.05*n+.5)))
        cc, cd = data['contexts'][c], data['contexts'][d]
        h = (cc['dc']+cd['dc'])/2+g.normal(size=(m, 3))
        xl, hl = right_view(h, cc, g, False)
        xr, hr = right_view(h, cd, g, corrupt in ('both', 'cross'))
        cross.append(ref.transform(xr)-ref.transform(xl))
        gaps['cross'].extend((ref.signal(hr)-ref.signal(hl)).tolist())
    t = np.concatenate(temporal)
    p = t.shape[-1]
    qt = t.T@t/len(t)
    qc = sum(v.T@v/len(v) for v in cross)/len(cross) if cross else np.zeros((p, p))
    diag = dict(temporal_count=len(t), cross_count=sum(map(len, cross)), edges=edges,
                degree=adjacency.sum(0).tolist(),
                spectral_gap=float(np.linalg.eigvalsh(np.diag(adjacency.sum(0))-adjacency)[1]),
                temporal_rank=int(np.linalg.matrix_rank(t)), cross_rank=int(np.linalg.matrix_rank(qc)))
    for key, gap in gaps.items():
        gap = np.asarray(gap)
        diag[key+'_true_rms'] = float(np.sqrt(np.mean(gap**2))) if len(gap) else None
        diag[key+'_gap_quantiles'] = np.quantile(np.abs(gap), [0, .5, .9, .99, 1]).tolist() if len(gap) else []
        diag[key+'_severe_fraction'] = float(np.mean(np.abs(gap) > .5)) if len(gap) else None
    return qt, qc, diag
