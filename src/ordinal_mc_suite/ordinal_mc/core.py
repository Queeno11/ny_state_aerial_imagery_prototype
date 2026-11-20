"""Shared scorers, losses and full-batch training, adapted from experiment 3 v2."""

from __future__ import annotations

import copy, hashlib, json, os, time

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from torch import nn

from torch.nn import functional as F

TRUTHS = ("linear", "quadratic", "smooth", "regime")

class InadmissibleScoreError(FloatingPointError):
    """A constant score lies outside the V>0 domain of the ranking criterion."""

def seed_for(seed, *tags):
    text = json.dumps([int(seed), *tags], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")

class StructuralDGP:
    """Fixed across MC repetitions; change --function-seed for a new structural draw."""
    def __init__(self, cfg):
        rng = np.random.default_rng(cfg.function_seed)
        self.rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        self.mix, _ = np.linalg.qr(rng.normal(size=(7, 7)))
        h = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(100_000, 3))
        self.moments = {}
        for shape in TRUTHS:
            y = self.raw(h, shape)
            self.moments[shape] = (float(y.mean()), float(y.std()))

    def raw(self, h, shape):
        u = h @ self.rotation
        if shape == "linear":
            return u[:, 0] + 0.6*u[:, 1] - 0.4*u[:, 2]
        if shape == "quadratic":
            return u[:, 0] + 0.7*u[:, 1]*u[:, 2] + 0.4*(u[:, 2]**2-1)
        if shape == "smooth":
            return np.sin(1.2*u[:, 0]) + 0.5*np.tanh(u[:, 1]) + 0.4*u[:, 1]*u[:, 2]
        if shape == "regime":
            return u[:, 0] + 0.8*np.maximum(u[:, 1], 0) - 0.6*np.maximum(-u[:, 2], 0)
        raise ValueError(shape)

    def score(self, h, shape):
        mean, sd = self.moments[shape]
        return (self.raw(h, shape)-mean)/sd

def subset(data, index):
    return {k:v[index] for k,v in data.items()}

def split_labels(data, fraction, rng):
    train, val = [], []
    for cell in np.unique(data["cell"]):
        loc = rng.permutation(np.flatnonzero(data["cell"] == cell))
        nv = max(2, int(round(fraction*len(loc))))
        nv = min(nv, len(loc)-2)
        val.extend(loc[:nv]); train.extend(loc[nv:])
    return subset(data, np.array(train)), subset(data, np.array(val))

def pair_indices(data, rng, count, kind="within"):
    c, t = data["c"], data["t"]
    if kind == "within":
        groups = [np.flatnonzero(data["cell"] == g) for g in np.unique(data["cell"])]
        if min(map(len, groups)) < 2:
            raise ValueError("Need two observations in every comparison cell.")
        group_id = rng.choice(len(groups), size=count, p=np.array([len(g) for g in groups])/sum(map(len,groups)))
        i, j = np.empty(count, np.int64), np.empty(count, np.int64)
        for k, group in enumerate(groups):
            loc = np.flatnonzero(group_id == k)
            a = rng.integers(len(group), size=len(loc))
            b = (a+rng.integers(1, len(group), size=len(loc))) % len(group)
            i[loc], j[loc] = group[a], group[b]
        return i, j
    i = rng.integers(len(c), size=count)
    j = rng.integers(len(c), size=count)
    for _ in range(2000):
        bad = ((c[i] == c[j]) if kind == "cross_city" else
               ((c[i] != c[j]) | (t[i] == t[j])))
        if not bad.any():
            return i, j
        j[bad] = rng.integers(len(c), size=bad.sum())
    raise RuntimeError("Invalid evaluation pair support.")

class Features:
    def __init__(self, x, model, cfg):
        self.model = model
        self.mu = x.mean(0)
        self.sd = np.maximum(x.std(0), 1e-6)
        self.ij = np.triu_indices(x.shape[1])
        rng = np.random.default_rng(seed_for(cfg.function_seed, "basis"))
        self.omega = rng.normal(size=(x.shape[1], cfg.rff))/np.sqrt(x.shape[1])
        self.phase = rng.uniform(0, 2*np.pi, cfg.rff)
        raw = self.raw(x)
        self.bmu, self.bsd = raw.mean(0), np.maximum(raw.std(0), 1e-6)

    def raw(self, x):
        z = (x-self.mu)/self.sd
        if self.model != "basis":
            return z
        return np.column_stack((z, z[:, self.ij[0]]*z[:, self.ij[1]],
                                np.cos(z@self.omega+self.phase)))

    def __call__(self, x):
        return np.ascontiguousarray((self.raw(x)-self.bmu)/self.bsd)

class Scorer(nn.Module):
    def __init__(self, dimension, model, hidden, weight_bound=10.):
        super().__init__()
        self.weight_bound=weight_bound
        sizes = [dimension] + (list(hidden) if model == "nn" else []) + [1]
        self.layers = nn.ModuleList([nn.Linear(a, b, bias=k < len(sizes)-2)
                                    for k,(a,b) in enumerate(zip(sizes[:-1], sizes[1:]))])

    def effective(self, value):
        return self.weight_bound*torch.tanh(value/self.weight_bound)

    def forward(self, x):
        for k,layer in enumerate(self.layers):
            bias=self.effective(layer.bias) if layer.bias is not None else None
            x=F.linear(x,self.effective(layer.weight),bias)
            if k<len(self.layers)-1: x=F.relu(x)
        return x.squeeze(-1)

    def squared_path_norm(self):
        # Linear/basis: exactly squared coefficient norm. ReLU MLP: squared
        # input/bias-to-output path norm, invariant to positive node rescalings.
        layer = self.layers[0]
        value = torch.ones(layer.in_features, dtype=layer.weight.dtype, device=layer.weight.device)
        for layer in self.layers:
            value = self.effective(layer.weight).square() @ value
            if layer.bias is not None:
                value = value + self.effective(layer.bias).square()
        return value.sum()

class Estimator(nn.Module):
    def __init__(self, dimension, model, loss, cfg):
        super().__init__()
        self.scorer = Scorer(dimension, model, cfg.hidden, cfg.weight_bound)
        self.loss = loss
        nscale = 2*cfg.cities if loss == "heterogeneous_logit" else 0
        self.raw_rho = nn.Parameter(torch.full((nscale,), 0.541324854612918), requires_grad=nscale > 0)
        self.level_bias = nn.Parameter(torch.zeros(()), requires_grad=loss == "squared")
        self.cell_effects = nn.Parameter(torch.zeros(2*cfg.cities), requires_grad=bool(getattr(cfg,"fit_cell_effects",False)))

    def rho(self):
        return F.softplus(self.raw_rho)

def tensor(x, cfg, integer=False):
    return torch.as_tensor(x, device=cfg.device,
                           dtype=torch.long if integer else cfg.torch_dtype)

def prepare(design, features, cfg):
    i,j = design["pairs"]
    tr = design["train"]
    return dict(x=tensor(features(design["x"]), cfg), i=tensor(i,cfg,True),
                j=tensor(j,cfg,True), y=tensor(np.sign(tr["w"][j]-tr["w"][i]),cfg),
                cells=tensor(tr["cell"][i],cfg,True), obs_cells=tensor(tr["cell"],cfg,True), weights=tensor(design["pair_weights"],cfg))

def score_objective(model, data, edges, link_weight, reg, tau):
    """Paper criterion: raw common-logit margin, V-normalized heterogeneous score.

    edges is a dict with separate temporal/cross_city arrays; link_weight=(lambda_t,lambda_c).
    Optional alpha*complexity is explicitly additional regularization, zero in paper presets.
    """
    s = model.scorer(data["x"]) + model.level_bias
    gaps = s[data["j"]]-s[data["i"]]
    variance = 0.5*(data["weights"]*gaps.square()).sum()
    if model.loss == "squared":
        fit=s[:len(data["target"])]+model.cell_effects[data["obs_cells"]]
        cross=(fit-data["target"]).square().mean()
        complexity=model.scorer.squared_path_norm()
        return cross+reg*complexity,(cross,s.new_tensor(0.),complexity,variance)
    if not bool(torch.isfinite(variance.detach())):
        raise FloatingPointError("Nonfinite within-cell score dispersion.")
    if not bool(variance.detach()>0):
        raise InadmissibleScoreError("Zero within-cell score dispersion: outside the paper's admissible domain.")
    delta=gaps/torch.sqrt(variance)
    # Tied W pairs are left out of q, but remain in V with the original tract weights.
    active=data["y"]!=0
    if model.loss == "common_logit":
        terms=F.softplus(-data["y"]*gaps)
        complexity=model.scorer.squared_path_norm()
    elif model.loss == "heterogeneous_logit":
        rho=model.rho()
        terms=F.softplus(-data["y"]*rho[data["cells"]]*delta)
        complexity=rho.square().mean()*model.scorer.squared_path_norm()/variance
    else:
        terms=torch.sigmoid(-data["y"]*delta/tau)
        complexity=model.scorer.squared_path_norm()/variance
    cross=(data["weights"]*terms*active).sum()
    stable=s.new_tensor(0.)
    for arm,weight in zip(("temporal","cross_city"),link_weight):
        e=edges[arm]
        if weight and len(e): stable=stable+weight*(s[e[:,0]]-s[e[:,1]]).square().mean()/variance
    return cross+stable+reg*complexity,(cross,stable,complexity,variance)

def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def train_candidate(data, edges, kind, loss, alpha, lam, seed, cfg):
    start = time.perf_counter()
    # A dead ReLU initialization can already have V=0. Retry only inadmissible
    # initializations, deterministically, without changing any valid start.
    for initialization_retries in range(10):
        initial_seed=seed if initialization_retries==0 else seed_for(seed,'admissible_init',initialization_retries)
        torch.manual_seed(initial_seed)
        if cfg.device == "cuda":
            torch.cuda.manual_seed_all(initial_seed)
        model = Estimator(data["x"].shape[1],kind,loss,cfg).to(device=cfg.device,dtype=cfg.torch_dtype)
        try:
            initial,_=score_objective(model,data,edges,lam,alpha,cfg.tau)
        except InadmissibleScoreError:
            continue
        if not torch.isfinite(initial):
            raise FloatingPointError("Nonfinite objective at initialization.")
        break
    else:
        raise InadmissibleScoreError('Ten deterministic initializations had zero score dispersion')
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr)
    fallback_state=cpu_state(model)
    adam_recoveries=0
    lbfgs_recoveries=0
    best_value, best_state = float("inf"), None
    # Warm start only: preserve the lowest objective in the FINAL tau phase.
    for epoch in range(cfg.epochs):
        tau = cfg.tau
        if loss == "ordinal_smooth" and epoch < cfg.epochs//2:
            tau = max(cfg.tau, 0.5*(1-epoch/max(1,cfg.epochs//2)))
        optimizer.zero_grad(set_to_none=True)
        try:
            value,_ = score_objective(model,data,edges,lam,alpha,tau)
        except InadmissibleScoreError:
            model.load_state_dict(best_state if best_state is not None else fallback_state)
            adam_recoveries+=1
            break
        if not torch.isfinite(value):
            raise FloatingPointError("Nonfinite objective; use float64 or reduce learning rate.")
        if tau == cfg.tau and value.item() < best_value:
            best_value, best_state = value.item(), cpu_state(model)
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 20.)
        optimizer.step()
    try:
        value,_=score_objective(model,data,edges,lam,alpha,cfg.tau)
    except InadmissibleScoreError:
        model.load_state_dict(best_state if best_state is not None else fallback_state)
        adam_recoveries+=1
        value,_=score_objective(model,data,edges,lam,alpha,cfg.tau)
    if value.item() < best_value:
        best_value,best_state=value.item(),cpu_state(model)
    if best_state is None:
        raise RuntimeError("Training did not produce a finite checkpoint.")
    model.load_state_dict(best_state)

    def refine(steps):
        nonlocal lbfgs_recoveries
        if steps == 0:
            return 0
        before = cpu_state(model)
        before_value = float(score_objective(model,data,edges,lam,alpha,cfg.tau)[0].detach())
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=steps,
                                  tolerance_grad=cfg.grad_tol,
                                  tolerance_change=1e-10 if cfg.dtype=="float64" else 1e-7,
                                  history_size=20,line_search_fn="strong_wolfe")
        calls=0
        def closure():
            nonlocal calls
            calls+=1
            lbfgs.zero_grad(set_to_none=True)
            q,_=score_objective(model,data,edges,lam,alpha,cfg.tau)
            if not torch.isfinite(q):
                raise FloatingPointError("Nonfinite objective during L-BFGS.")
            q.backward()
            return q
        try:
            lbfgs.step(closure)
            after=float(score_objective(model,data,edges,lam,alpha,cfg.tau)[0].detach())
        except InadmissibleScoreError:
            model.load_state_dict(before)
            lbfgs_recoveries+=1
            return calls
        if not np.isfinite(after) or after>before_value:
            model.load_state_dict(before)
        return calls

    evaluations=refine(cfg.lbfgs_steps)
    before_audit=float(score_objective(model,data,edges,lam,alpha,cfg.tau)[0].detach())
    before_audit_recoveries=lbfgs_recoveries
    evaluations+=refine(cfg.audit_steps)
    audit_recoveries=lbfgs_recoveries-before_audit_recoveries
    model.zero_grad(set_to_none=True)
    value,parts=score_objective(model,data,edges,lam,alpha,cfg.tau)
    value.backward()
    grad_inf=max(float(p.grad.abs().max()) for p in model.parameters() if p.grad is not None and p.numel())
    final=float(value.detach())
    audit_drop=max(0.,before_audit-final)
    stats=dict(train_objective=final,train_cross=float(parts[0].detach()),
               train_stable=float(parts[1].detach()),train_complexity=float(parts[2].detach()),
               score_variance=float(parts[3].detach()),gradient_inf=grad_inf,
               first_order_small=grad_inf<=cfg.grad_tol,
               audit_objective_drop=audit_drop,
               audit_stable=(audit_drop<=cfg.audit_tol and audit_recoveries==0) if cfg.audit_steps else None,
               initialization_retries=initialization_retries,
               adam_dispersion_recoveries=adam_recoveries,
               lbfgs_dispersion_recoveries=lbfgs_recoveries,
               audit_dispersion_recoveries=audit_recoveries,
               optimizer_status='recovered' if initialization_retries or adam_recoveries or lbfgs_recoveries else 'ok',
               closure_evaluations=evaluations,seconds=time.perf_counter()-start,
               parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model,stats

def misranking(predicted, target):
    # Exact target ties are excluded; predicted ties receive half error.
    keep=target!=0
    if not keep.any():
        return float("nan")
    product=predicted[keep]*target[keep]
    return float(np.mean((product<0).astype(float)+0.5*(predicted[keep]==0)))

@torch.no_grad()
def predictions(model, features, x, cfg):
    chunks=[]
    for start in range(0,len(x),8192):
        chunks.append(model.scorer(tensor(features(x[start:start+8192]),cfg)).cpu().numpy())
    return np.concatenate(chunks)

def validation_error(model, features, design, cfg):
    # No latent f, cross-cell W, or test data enters hyperparameter selection.
    val=design["val"]; i,j=design["val_pairs"]
    s=predictions(model,features,val["x"],cfg)
    return misranking(s[j]-s[i],val["w"][j]-val["w"][i])

def test_metrics(model,features,design,cfg):
    return evaluate_predictions(lambda x: predictions(model,features,x,cfg),design,cfg,
                                normalized_mse=model.loss in ('common_logit','heterogeneous_logit'))


def evaluate_predictions(predict,design,cfg,normalized_mse=True,absolute_scale=False):
    from .change import evaluate_changes
    result={}
    train_score=predict(design["train"]["x"])
    ii,jj=design["pairs"]
    train_sd=np.sqrt(.5*np.sum(design['pair_weights']*(train_score[jj]-train_score[ii])**2))
    for name,(data,pp) in design["tests"].items():
        s=predict(data["x"])
        for kind,(i,j) in pp.items():
            result[name+"_"+kind+"_latent"]=misranking(s[j]-s[i],data["f"][j]-data["f"][i])
            result[name+"_"+kind+"_accuracy"]=1-result[name+"_"+kind+"_latent"]
            if kind=="within":
                result[name+"_within_observed"]=misranking(s[j]-s[i],data["w"][j]-data["w"][i])
                result[name+"_oracle_observed"]=misranking(data["f"][j]-data["f"][i],data["w"][j]-data["w"][i])
                if normalized_mse and train_sd>0:
                    result[name+"_within_difference_mse"]=float(np.mean(((s[j]-s[i])/train_sd-(data["f"][j]-data["f"][i]))**2))
    result['evaluation_zero_dispersion']=bool(train_sd==0)
    result.update(evaluate_changes(predict,design,cfg,1. if absolute_scale or train_sd==0 else train_sd))
    return result
