"""Matched ordinal versus cardinal pilot; see JOINT_TWO_STEP_PILOT.md."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, LinearConstraint, lsq_linear
from numpy.polynomial.hermite import hermgauss

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise SystemExit("PyTorch is required; install requirements.txt into the selected Python environment") from exc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SQ3 = math.sqrt(3.)
QX, QW = hermgauss(32)
QX, QW = QX*math.sqrt(2.), QW/math.sqrt(math.pi)
SETTINGS = [("linear","M1","simple",0.),("linear","M2","simple",0.),
            ("linear","M3","simple",0.),("quadratic","M1","simple",0.),
            ("quadratic","M2","simple",0.),("quadratic","M3","simple",0.),
            ("quadratic","M3","nonlinear",0.),("quadratic","M3","nonlinear",.9)]
MIX,_ = np.linalg.qr(np.random.default_rng(984).normal(size=(7,7)))


def seed(base,*tags):
    return int.from_bytes(hashlib.sha256(json.dumps((base,*tags)).encode()).digest()[:4],"little")


def basis(h,shape):
    u=h # structural coordinates; the observed seven-vector is separately mixed
    if shape=="linear": return u
    return np.column_stack((u,u[:,0]**2,u[:,1]**2,u[:,2]**2,
                            u[:,0]*u[:,1],u[:,0]*u[:,2],u[:,1]*u[:,2]))


def reference(shape):
    b=basis(np.random.default_rng(983).uniform(-SQ3,SQ3,(100000,3)),shape)
    return b.mean(0),np.cov(b,rowvar=False)


REF={s:reference(s) for s in ("linear","quadratic")}


def true_coeff(shape):
    return np.array([1.,.6,-.4]) if shape=="linear" else np.array([1.,0.,0.,0.,0.,.4,0.,0.,.7])


def latent(h,shape,coeff=None):
    coeff=true_coeff(shape) if coeff is None else np.asarray(coeff)
    mean,cov=REF[shape]
    return (basis(h,shape)-mean)@coeff/math.sqrt(coeff@cov@coeff)


def map_parameters(c,t,model,contexts):
    z=contexts[c]; a=.45*z[:,0]+.35*t
    b=np.ones(len(c)) if model=="M1" else 1.3+np.exp(.18*z[:,1]+.08*t)
    k=np.zeros(len(c)) if model!="M3" else .45+.12*np.tanh(z[:,0]+.5*t)
    return a,b,k


def psi(r,a,b,k): return a+b*(r+k*np.tanh(r))


def inverse_torch(y,k):
    """Bracketed numeric root with implicit first derivative for LBFGS."""
    with torch.no_grad():
        yy,kk=y.detach(),k.detach()
        lo,hi=yy-kk,yy+kk
        root=yy.clone()
        tolerance=1e-11*(1+yy.abs())
        for _ in range(64):
            residual=root+kk*torch.tanh(root)-yy
            converged=residual.abs()<=tolerance
            if bool(torch.all(converged)): break
            lo=torch.where((residual<0)&~converged,root,lo)
            hi=torch.where((residual>0)&~converged,root,hi)
            proposal=root-residual/(1+kk*(1-torch.tanh(root).square()))
            next_root=torch.where((proposal>lo)&(proposal<hi),proposal,(lo+hi)/2)
            root=torch.where(converged,root,next_root)
        if bool(torch.any((root+kk*torch.tanh(root)-yy).abs()>tolerance)):
            raise FloatingPointError("M3 inverse residual exceeded tolerance")
    # The detached denominator yields dr/dy=1/J, dr/dk=-tanh(root)/J.
    denominator=(1+k*(1-torch.tanh(root).square())).detach()
    return root+(y-root-k*torch.tanh(root))/denominator


def inverse_numpy(y,k):
    """Same safeguarded root geometry for held-out target likelihood."""
    lo,hi=y-k,y+k
    root=y.copy();tolerance=1e-11*(1+np.abs(y))
    for _ in range(64):
        residual=root+k*np.tanh(root)-y
        converged=np.abs(residual)<=tolerance
        if np.all(converged): break
        lo=np.where((residual<0)&~converged,root,lo)
        hi=np.where((residual>0)&~converged,root,hi)
        proposal=root-residual/(1+k*(1-np.tanh(root)**2))
        next_root=np.where((proposal>lo)&(proposal<hi),proposal,(lo+hi)/2)
        root=np.where(converged,root,next_root)
    if np.any(np.abs(root+k*np.tanh(root)-y)>tolerance):
        raise FloatingPointError("Target M3 inverse residual exceeded tolerance")
    return root


def conditional_mean(f,a,b,k,sigma):
    if not np.any(k): return a+b*f
    return a+b*(f+k*(np.tanh(f[:,None]+sigma*QX[None,:])@QW))


def features(h,c,t,contexts,rng):
    nuisance=rng.uniform(-1,1,(len(h),2))+np.asarray(t)[:,None]*np.array([.12,-.12])
    return np.column_stack((h,nuisance,contexts[c]))@MIX.T


def panel(args,shape,model,mechanism,rho,part,cities,base_seed):
    rng=np.random.default_rng(seed(base_seed,part,mechanism,cities))
    group="target" if part.startswith("target") else "source"
    contexts=np.random.default_rng(seed(base_seed,"contexts",group,cities)).normal(size=(cities,2))
    n=args.n if part=="fit" else args.eval_n
    count=cities*n;c=np.repeat(np.arange(cities),n)
    h0=rng.uniform(-SQ3,SQ3,(count,3));h1=h0.copy()
    category=np.tile(np.arange(n)%4,cities) # stable 50%, small 25%, large 25%
    attempts=accepted=0
    for cat,lo,hi in ((2,.12,.38),(3,.48,1.10)):
        todo=np.flatnonzero(category==cat)
        for trial in range(2000):
            if not len(todo): break
            attempts+=len(todo)
            # Redraw the baseline too: a fixed baseline can make a requested
            # signed bin unattainable inside bounded support.
            baseline=rng.uniform(-SQ3,SQ3,(len(todo),3))
            candidate=baseline.copy()
            step=rng.uniform(-1.2,1.2,len(todo))
            if mechanism=="simple": candidate[:,0]+=step
            else:
                candidate[:,1]+=step
                candidate[:,2]+=rng.uniform(-1,1,len(todo)) # H0 fixed; nonlinear terms remain active
            delta=latent(candidate,shape)-latent(baseline,shape)
            sign=np.where((todo//n+(todo%n)//4)%2==0,1,-1)
            good=(np.max(np.abs(candidate),axis=1)<=SQ3)&(np.abs(delta)>=lo)&(np.abs(delta)<hi)&(np.sign(delta)==sign)
            h0[todo[good]]=baseline[good]
            h1[todo[good]]=candidate[good];accepted+=int(good.sum());todo=todo[~good]
        if len(todo): raise RuntimeError(f"Change rejection exhausted: {shape}/{mechanism}/{cat}: {len(todo)}")
    f=np.stack((latent(h0,shape),latent(h1,shape)),1)
    view_rng=np.random.default_rng(seed(base_seed,part,"views",cities))
    x=np.stack((features(h0,c,np.zeros(count),contexts,view_rng),features(h1,c,np.ones(count),contexts,view_rng)),1)
    u,ep0,ep1=np.random.default_rng(seed(base_seed,part,"noise",cities)).normal(size=(3,count))
    eta=np.stack((args.sigma*(math.sqrt(rho)*u+math.sqrt(1-rho)*ep0),
                  args.sigma*(math.sqrt(rho)*u+math.sqrt(1-rho)*ep1)),1)
    r=f+eta; w=np.empty_like(r);mu=np.empty_like(r)
    for t in (0,1):
        a,b,k=map_parameters(c,np.full(count,t),model,contexts)
        w[:,t]=psi(r[:,t],a,b,k);mu[:,t]=conditional_mean(f[:,t],a,b,k,args.sigma)
    return dict(h=np.stack((h0,h1),1),x=x,f=f,r=r,w=w,mu=mu,c=c,
                category=category,contexts=contexts,rejection_rate=1-accepted/max(attempts,1),
                attempts=attempts,endpoint_min=np.minimum(h0.min(0),h1.min(0)).tolist(),
                endpoint_max=np.maximum(h0.max(0),h1.max(0)).tolist(),
                empirical_residual_corr=float(np.corrcoef(eta.T)[0,1]))


class Scorer(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(7,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x): return self.net(x).squeeze(-1)


class Head(nn.Module):
    def __init__(self,model,cities,knots,center,scale):
        super().__init__();self.model=model
        self.register_buffer("knots",torch.as_tensor(knots,dtype=torch.float64))
        self.register_buffer("center",torch.tensor(center,dtype=torch.float64))
        self.register_buffer("scale",torch.tensor(scale,dtype=torch.float64))
        self.intercept=nn.Parameter(torch.zeros(2*cities,dtype=torch.float64))
        self.raw=nn.Parameter(torch.full((1 if model=="M1" else 2*cities,
                                         len(knots)+1 if model=="M3" else 1),.541324854612918,dtype=torch.float64))
    def forward(self,s,cell):
        z=(s-self.center)/self.scale
        slopes=F.softplus(self.raw)[torch.zeros_like(cell) if self.model=="M1" else cell]
        result=self.intercept[cell]+slopes[:,0]*z
        if self.model=="M3": result=result+((slopes[:,1:]-slopes[:,:-1])*F.relu(z[:,None]-self.knots)).sum(1)
        return result


def tensor_panel(p,device,center,scale):
    out={k:torch.as_tensor(p[k],dtype=torch.float64,device=device) for k in ("x","w")}
    out["x"]=(out["x"]-torch.as_tensor(center,device=device))/torch.as_tensor(scale,device=device)
    out["cell"]=torch.as_tensor(2*p["c"][:,None]+np.array([0,1]),dtype=torch.long,device=device)
    return out


def score(model,d): return model(d["x"].reshape(-1,7)).reshape(-1,2)


def pair_loss(s,w,cell,kind,tau):
    gap=s[:,None]-s[None,:]; sign=torch.sign(w[:,None]-w[None,:])
    eligible=(cell[:,None]==cell[None,:])&~torch.eye(len(s),device=s.device,dtype=torch.bool)
    count=torch.bincount(cell,minlength=int(cell.max())+1)
    weights=eligible.to(s.dtype)/(torch.unique(cell).numel()*count[cell][:,None]*(count[cell][:,None]-1))
    v=.5*(weights*gap.square()).sum()
    if v<=1e-12: raise FloatingPointError("Degenerate scorer variance")
    z=-sign*gap
    return (weights*(F.softplus(z) if kind=="ranknet" else torch.sigmoid(z/tau))).sum(),v


def cardinal(pred,observed,vstar,kappa,q):
    e=pred-observed;a=e.mean(1);d=e[:,1]-e[:,0]
    return kappa/vstar*((1-q)*a.square().mean()+q*d.square().mean()/4)


def batches(d,n,base_seed,step):
    rng=np.random.default_rng(seed(base_seed,"batch",step)); cities=len(d["x"])//n
    idx=np.concatenate([c*n+rng.choice(n,min(n,12),replace=False) for c in range(cities)])
    return {k:v[idx] for k,v in d.items()}


def links(fit,device,center,scale,on,base_seed):
    rng=np.random.default_rng(seed(base_seed,"links"));n=len(fit["c"])
    x0=fit["x"][:,0];h=fit["h"][:,0];c=fit["c"]
    stable=fit["category"]<2
    # Only observed before/after endpoints of truly stable fitting tracts.
    pairs=[(x0[stable],fit["x"][stable,1])]
    stats=dict(temporal_pairs=int(stable.sum()),cross_pairs=0,
               cross_origin_cities=0,cross_destination_cities=0,
               cross_coverage=0.,cross_graph_components=int(c.max())+1,
               cross_graph_connected=False)
    if on:
        cities=int(c.max())+1;budget=round(.2*n)
        if budget<cities-1:
            raise ValueError(f"20% cross-link budget {budget} cannot connect {cities} cities; need at least {cities-1}")
        # Chain edges guarantee a connected city graph; remaining origins are distinct.
        reserved=np.array([rng.choice(np.flatnonzero(c==city)) for city in range(cities-1)],dtype=int)
        available=np.setdiff1d(np.arange(n),reserved)
        extra=rng.choice(available,budget-len(reserved),replace=False)
        ids=np.r_[reserved,extra]
        dest=np.r_[np.arange(1,cities),
                   (c[extra]+1+rng.integers(0,cities-1,len(extra)))%cities]
        pairs.append((x0[ids],features(h[ids],dest,np.zeros(len(ids)),fit["contexts"],rng)))
        parent=list(range(cities))
        def find(node):
            while parent[node]!=node:node=parent[node]
            return node
        for origin,destination in zip(c[ids],dest):
            parent[find(int(origin))]=find(int(destination))
        components=len({find(i) for i in range(cities)})
        stats.update(cross_pairs=len(ids),cross_origin_cities=int(np.unique(c[ids]).size),
                     cross_destination_cities=int(np.unique(dest).size),cross_coverage=len(ids)/n,
                     cross_graph_components=components,cross_graph_connected=(components==1))
    tensors=[(torch.as_tensor((a-center)/scale,dtype=torch.float64,device=device),
              torch.as_tensor((b-center)/scale,dtype=torch.float64,device=device)) for a,b in pairs]
    return tensors,stats


def new_head(family,scorer,d):
    with torch.no_grad(): s=score(scorer,d).flatten().cpu().numpy()
    center=float(s.mean());scale=max(float(s.std()),1e-5)
    knots=np.quantile((s-center)/scale,[.2,.5,.8]) if family=="M3" else []
    return Head(family,int(d["cell"].max().item()//2+1),knots,center,scale).to(d["x"].device)


def head_design(z,cell,family,ncell,knots):
    eye=np.eye(ncell)[cell]
    if family=="M1": return np.column_stack((eye,z))
    if family=="M2": return np.column_stack((eye,eye*z[:,None]))
    hinge=np.column_stack((z,*[np.maximum(z-k,0) for k in knots]))
    return np.column_stack((eye,np.einsum("ni,nj->nij",eye,hinge).reshape(len(z),-1)))


def bounded_head_fallback(x,target,ncell,family,width):
    """Same convex problem in segment slopes, solved by bounded least squares.

    Hinge coefficients are differences of adjacent slopes. Reparameterizing
    them makes monotonicity simple lower bounds, with unrestricted intercepts.
    Column scaling changes numerical units only, not the objective or bounds.
    """
    transform=np.eye(x.shape[1])
    if family=="M3":
        difference=np.eye(width)-np.eye(width,k=-1)
        for cell in range(ncell):
            start=ncell+cell*width
            transform[start:start+width,start:start+width]=difference
    design=x@transform
    scales=np.maximum(np.linalg.norm(design,axis=0),1e-12)
    normalized=design/scales
    lower=np.r_[np.full(ncell,-np.inf),np.full(x.shape[1]-ncell,1e-5)]
    res=lsq_linear(normalized,target,bounds=(lower*scales,np.inf),
                   method="bvls",tol=1e-12,max_iter=10000)
    slopes_and_intercepts=res.x/scales
    residual=normalized@res.x-target
    gradient=normalized.T@residual
    projected=gradient.copy()
    at_lower=np.isfinite(lower)&(res.x<=lower*scales+1e-9*np.maximum(1,np.abs(res.x)))
    projected[at_lower]=np.minimum(projected[at_lower],0.)
    kkt=float(np.max(np.abs(projected)))
    tolerance=1e-7*max(1.,float(np.linalg.norm(target)))
    if (not res.success or not np.all(np.isfinite(res.x)) or
            np.min(slopes_and_intercepts[ncell:])<1e-5-1e-10 or kkt>tolerance):
        raise RuntimeError(f"Bounded head fit failed: {res.message}; projected gradient {kkt:.4g}")
    coefficients=transform@slopes_and_intercepts
    return coefficients,dict(solver="BVLS segment-slope fallback",iterations=int(res.nit),
        projected_gradient=kkt,projected_gradient_tolerance=tolerance,
        minimum_slope=float(np.min(slopes_and_intercepts[ncell:])),
        objective=float(np.mean((x@coefficients-target)**2)))


def check_resume_manifest(path,manifest):
    """Permit only this solver-fallback patch over the exact reviewed version."""
    if not path.exists(): return
    previous=json.loads(path.read_text(encoding="utf-8"))
    if previous==manifest: return
    legacy="0f403b3db4188f300fe87fcd8bcf3d0ffa547115c464116af46d4fe571de57ca"
    old_settings={k:v for k,v in previous.items() if k!="source_sha256"}
    new_settings={k:v for k,v in manifest.items() if k!="source_sha256"}
    if previous.get("source_sha256")!=legacy or old_settings!=new_settings:
        raise ValueError("Output manifest differs; use a fresh output directory for changed configuration")
    # Keep the old manifest and explicit provenance for reused successful jobs.
    backup=path.with_name("manifest_before_solver_fallback.json")
    if not backup.exists(): backup.write_text(json.dumps(previous,indent=2),encoding="utf-8")
    completed=sorted(p.name for p in path.parent.glob("seed*.json") if p.with_suffix(".csv").exists())
    history=dict(previous_source_sha256=legacy,current_source_sha256=manifest["source_sha256"],
        reused_configurations=completed,
        reason="Same objective/constraints; BVLS fallback only when original SLSQP fails")
    path.with_name("solver_patch_resume.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    print(f"Solver patch: reusing {len(completed)} completed configurations; provenance saved.",flush=True)


def fit_frozen(head,s,d,q,vstar,kappa,fixed_m1_slope=None):
    """Constrained QP with date-coupled change terms; exact same cardinal objective."""
    z=((s-head.center)/head.scale).detach().cpu().numpy().reshape(-1)
    cell=d["cell"].cpu().numpy().reshape(-1);y=d["w"].cpu().numpy()
    knots=head.knots.cpu().numpy();ncell=len(head.intercept)
    if fixed_m1_slope is not None:
        if head.model!="M1": raise ValueError("Fixed common scale applies only to M1")
        design=np.eye(ncell)[cell].reshape(-1,2,ncell)
        y=y-fixed_m1_slope*z.reshape(-1,2)
    else:
        design=head_design(z,cell,head.model,ncell,knots).reshape(-1,2,ncell+head.raw.numel())
    x=np.vstack((math.sqrt(1-q)*design.mean(1),math.sqrt(q)*(design[:,1]-design[:,0])/2))
    target=np.r_[math.sqrt(1-q)*y.mean(1),math.sqrt(q)*(y[:,1]-y[:,0])/2]
    cons=[]
    if fixed_m1_slope is not None: pass
    elif head.model=="M3":
        # Hinge coefficients are slope differences. Each cumulative segment slope is constrained.
        width=len(knots)+1
        for c in range(ncell):
            for seg in range(width):
                row=np.zeros(design.shape[-1]);row[ncell+c*width]=1
                for k in range(seg): row[ncell+c*width+k+1]=1
                cons.append(row)
    else:
        for j in range(head.raw.numel()):
            row=np.zeros(design.shape[-1]);row[ncell+j]=1;cons.append(row)
    a=np.asarray(cons)
    def objective(v):
        e=x@v-target
        return float(e@e/len(e)),2*x.T@e/len(e)
    init=np.linalg.lstsq(x,target,rcond=None)[0]
    if fixed_m1_slope is not None:
        with torch.no_grad(): head.intercept.copy_(torch.as_tensor(init,device=head.center.device))
        return dict(success=True,iterations=0,objective=float(objective(init)[0]*kappa/vstar),
                    common_scale_fixed=True)
    res=minimize(objective,init,jac=True,method="SLSQP",
                 constraints=[LinearConstraint(a,1e-5,np.inf)],options={"maxiter":300,"ftol":1e-10})
    diagnostic=dict(solver="SLSQP",iterations=int(res.nit),objective=float(res.fun))
    if not res.success or np.min(a@res.x)<9e-6:
        failure=str(res.message)
        print(f"Frozen head SLSQP: {failure}; retrying equivalent bounded least squares.",flush=True)
        coefficients,diagnostic=bounded_head_fallback(x,target,ncell,head.model,len(knots)+1)
        diagnostic["slsqp_message"]=failure
        diagnostic["slsqp_iterations"]=int(res.nit)
    else: coefficients=res.x
    coeff=coefficients[ncell:].reshape(head.raw.shape)
    slopes=np.cumsum(coeff,axis=1) if head.model=="M3" else coeff
    # Stable inverse softplus, including slopes too large for expm1.
    inverse=np.where(slopes>30,slopes+np.log1p(-np.exp(-slopes)),np.log(np.expm1(np.minimum(slopes,30))))
    with torch.no_grad():
        head.intercept.copy_(torch.as_tensor(coefficients[:ncell],device=head.center.device))
        head.raw.copy_(torch.as_tensor(inverse,device=head.center.device))
    return dict(success=True,**{**diagnostic,"objective":float(diagnostic["objective"]*kappa/vstar)})


def train(initial,head0,fit,val,link_pairs,args,rank,q,vstar,stage,base_seed):
    scorer=copy.deepcopy(initial);head=copy.deepcopy(head0) if head0 is not None else None
    if stage=="ordinal": head=None
    steps=args.ord_steps if stage=="ordinal" else args.joint_steps
    groups=[{"params":scorer.parameters(),"lr":args.lr}]
    if head is not None: groups.append({"params":head.parameters(),"lr":args.head_lr})
    opt=torch.optim.Adam(groups);validation_trace=[];gradients=[]
    for step in range(steps+1):
        if step:
            opt.zero_grad(set_to_none=True)
            batch=batches(fit,args.n,base_seed,step)
            s=score(scorer,batch)
            rank_value,v=pair_loss(s.flatten(),batch["w"].flatten(),batch["cell"].flatten(),rank,args.tau)
            st=(scorer(link_pairs[0][0])-scorer(link_pairs[0][1])).square().mean()/v
            cr=(scorer(link_pairs[1][0])-scorer(link_pairs[1][1])).square().mean()/v if len(link_pairs)>1 else 0.
            temporal=args.lambda_t*st;cross=args.lambda_c*cr
            ordinal=rank_value+temporal+cross
            components=[ordinal]
            if head is not None:
                pred=head(s.flatten(),batch["cell"].flatten()).reshape(-1,2)
                components.append(cardinal(pred,batch["w"],vstar,args.kappa,q))
            if step in {1,max(1,steps//2),steps}:
                vectors=[]
                for comp in (rank_value,temporal,cross,ordinal,*components[1:]):
                    if not isinstance(comp,torch.Tensor): comp=torch.as_tensor(comp,dtype=torch.float64,device=s.device)
                    if not comp.requires_grad:
                        vectors.append(torch.zeros(sum(p.numel() for p in scorer.parameters()),device=s.device));continue
                    g=torch.autograd.grad(comp,list(scorer.parameters()),retain_graph=True,allow_unused=True)
                    vectors.append(torch.cat([(v if v is not None else torch.zeros_like(p)).flatten()
                                              for v,p in zip(g,scorer.parameters())]).detach())
                gradients.append(dict(step=step,rank_norm=float(vectors[0].norm()),
                    weighted_temporal_norm=float(vectors[1].norm()),weighted_cross_norm=float(vectors[2].norm()),
                    ordinal_norm=float(vectors[3].norm()),
                    cardinal_norm=float(vectors[4].norm()) if len(vectors)>4 else 0.,
                    ordinal_cardinal_cosine=float(F.cosine_similarity(vectors[3],vectors[4],dim=0)) if len(vectors)>4 else None,
                    temporal_cardinal_cosine=float(F.cosine_similarity(vectors[1],vectors[4],dim=0)) if len(vectors)>4 else None,
                    cross_cardinal_cosine=float(F.cosine_similarity(vectors[2],vectors[4],dim=0)) if len(vectors)>4 and
                        bool(vectors[2].norm()>0) else None,
                    rank_temporal_cosine=float(F.cosine_similarity(vectors[0],vectors[1],dim=0))))
            loss=sum(components)
            if not bool(torch.isfinite(loss)): raise FloatingPointError("Nonfinite training loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(scorer.parameters(),20.);opt.step()
        if step==0 or step%args.check_every==0 or step==steps:
            with torch.no_grad():
                sv=score(scorer,val)
                rv,_=pair_loss(sv.flatten(),val["w"].flatten(),val["cell"].flatten(),rank,args.tau)
                selection=float(rv)
                if head is not None:
                    pv=head(sv.flatten(),val["cell"].flatten()).reshape(-1,2)
                    selection+=float(cardinal(pv,val["w"],vstar,args.kappa,q))
                validation_trace.append(dict(step=step,value=selection))
    return scorer,head,dict(selected_step=steps,validation=validation_trace[-1]["value"],
                           validation_trace=validation_trace,updates=steps,gradients=gradients,
                           convergence="short optimization budget; not assessed" if args.smoke else "fixed-budget endpoint")


def benchmark(args,fit,shape,family,rho,iterations):
    """Correct family paired Gaussian likelihood; sigma/rho are known design inputs.

    f is centered with population variance one, fixing global location/scale;
    positive orientation is fixed by the first basis coefficient. Cell map parameters
    are estimated from W, with exact M1 slope one, M2 positive slopes, M3 exact tanh.
    """
    device="cpu"
    h=torch.as_tensor(fit["h"],dtype=torch.float64,device=device)
    w=torch.as_tensor(fit["w"],dtype=torch.float64,device=device)
    c=torch.as_tensor(fit["c"],dtype=torch.long,device=device)
    bb=torch.as_tensor(basis(h.numpy().reshape(-1,3),shape).reshape(-1,2,len(true_coeff(shape))),dtype=torch.float64)
    mean,cov=REF[shape];bm=torch.as_tensor(mean);bcov=torch.as_tensor(cov)
    ncell=2*(int(c.max())+1);cell=2*c[:,None]+torch.arange(2)[None,:]
    start=np.full(len(true_coeff(shape)),.1);start[0]=1. # generic initialization, no true coefficient insertion
    theta=nn.Parameter(torch.as_tensor(start,dtype=torch.float64))
    intercept=nn.Parameter(torch.zeros(ncell,dtype=torch.float64))
    raw_b=nn.Parameter(torch.full((ncell,),1.,dtype=torch.float64)) if family!="M1" else None
    raw_k=nn.Parameter(torch.full((ncell,),-.7,dtype=torch.float64)) if family=="M3" else None
    variables=[theta,intercept]+([raw_b] if raw_b is not None else [])+([raw_k] if raw_k is not None else [])
    def current():
        t=torch.cat((F.softplus(theta[:1])+1e-5,theta[1:]))
        f=((bb-bm)@t)/torch.sqrt(t@bcov@t)
        a=intercept[cell]
        b=torch.ones_like(a) if raw_b is None else F.softplus(raw_b[cell])+1e-4
        k=torch.zeros_like(a) if raw_k is None else 2*torch.sigmoid(raw_k[cell])
        return f,a,b,k
    def objective():
        f,a,b,k=current()
        y=(w-a)/b
        if family=="M3":
            r=inverse_torch(y,k)
            jac=b*(1+k*(1-torch.tanh(r).square()))
        else: r=y;jac=b
        e=r-f
        quadratic=(e.square().sum(1)-2*rho*e[:,0]*e[:,1])/(args.sigma**2*(1-rho*rho))
        return (.5*quadratic+torch.log(jac).sum(1)).mean()
    opt=torch.optim.LBFGS(variables,lr=.8,max_iter=iterations,line_search_fn="strong_wolfe",
                           tolerance_grad=1e-8,tolerance_change=1e-11)
    calls=0
    def closure():
        nonlocal calls
        calls+=1;opt.zero_grad();loss=objective()
        if not torch.isfinite(loss): raise FloatingPointError("Parametric likelihood nonfinite")
        loss.backward();return loss
    initial=float(objective().detach());opt.step(closure)
    for p in variables:p.grad=None
    final_loss=objective();final_loss.backward();final=float(final_loss.detach())
    grad=float(math.sqrt(sum(float(p.grad.square().sum()) for p in variables)))
    with torch.no_grad():
        t=torch.cat((F.softplus(theta[:1])+1e-5,theta[1:])).numpy()
        pars=dict(a=intercept.numpy().copy(),b=(np.ones(ncell) if raw_b is None else (F.softplus(raw_b)+1e-4).numpy().copy()),
                  k=(np.zeros(ncell) if raw_k is None else (2*torch.sigmoid(raw_k)).numpy().copy()))
    return dict(coeff=t,**pars),dict(initial_nll=initial,final_nll=final,gradient_norm=grad,calls=calls,
                                    convergence="not certified" if args.smoke or grad>1e-4 else "gradient tolerance met",
                                    known_noise=True)


def predict_benchmark(p,shape,pars,args,calibrate=False):
    f=latent(p["h"].reshape(-1,3),shape,pars["coeff"]).reshape(-1,2)
    a=pars["a"][2*p["c"][:,None]+np.arange(2)]
    b=pars["b"][2*p["c"][:,None]+np.arange(2)]
    k=pars["k"][2*p["c"][:,None]+np.arange(2)]
    mu=np.column_stack([conditional_mean(f[:,t],a[:,t],b[:,t],k[:,t],args.sigma) for t in (0,1)])
    return f,mu


def target_parametric(args,p,shape,family,source_pars):
    # Freeze estimated f; fit only fresh target-cell map parameters by correct likelihood.
    f=latent(p["h"].reshape(-1,3),shape,source_pars["coeff"]).reshape(-1,2)
    w=p["w"];rho=args.rho_current;ncell=2*(p["c"].max()+1)
    def unpack(x):
        a=x[:ncell];b=np.ones(ncell) if family=="M1" else np.exp(x[ncell:2*ncell])
        k=np.zeros(ncell) if family!="M3" else 2/(1+np.exp(-x[2*ncell:]))
        return a,b,k
    cell=2*p["c"][:,None]+np.arange(2)
    def nll(x):
        a,b,k=unpack(x);a=a[cell];b=b[cell];k=k[cell]
        y=(w-a)/b
        if family=="M3":
            try:r=inverse_numpy(y,k)
            except FloatingPointError:return 1e20
            jac=b*(1+k*(1-np.tanh(r)**2))
        else: r=y;jac=b
        e=r-f
        return float(np.mean(.5*(np.sum(e*e,axis=1)-2*rho*e[:,0]*e[:,1])/(args.sigma**2*(1-rho*rho))+
                             np.log(jac).sum(1)))
    x0=np.r_[np.zeros(ncell),np.full(0 if family=="M1" else ncell,.7),
             np.full(ncell,-.8) if family=="M3" else []]
    bounds=([(-10,10)]*ncell+([] if family=="M1" else [(-3,3)]*ncell)+
            ([] if family!="M3" else [(-8,8)]*ncell))
    res=minimize(nll,x0,method="L-BFGS-B",bounds=bounds,options={"maxiter":args.benchmark_steps})
    a,b,k=unpack(res.x)
    return dict(coeff=source_pars["coeff"],a=a,b=b,k=k),dict(success=bool(res.success),
        message=str(res.message),nll=float(res.fun),gradient_norm=float(np.linalg.norm(res.jac)),iterations=int(res.nit))


def score_numpy(scorer,p,center,scale,device):
    d=tensor_panel(p,device,center,scale)
    with torch.no_grad(): return score(scorer,d).cpu().numpy()


def head_numpy(head,s,p,device):
    cell=torch.as_tensor(2*p["c"][:,None]+np.arange(2),dtype=torch.long,device=device)
    with torch.no_grad():
        return head(torch.as_tensor(s.reshape(-1),dtype=torch.float64,device=device),cell.reshape(-1)).reshape(-1,2).cpu().numpy()


def fit_alignment(s,fit):
    # Evaluation only. Exactly one fitting-data affine map reused for all cells/dates/panels.
    x=s.reshape(-1);y=fit["f"].reshape(-1)
    b=float(np.dot(x-x.mean(),y-y.mean())/np.dot(x-x.mean(),x-x.mean()))
    if not np.isfinite(b): raise FloatingPointError("Nonfinite fitting alignment")
    if b<=0:
        # No oracle sign flip: this scorer failed to recover orientation.
        fallback=1/max(float(np.std(x)),1e-8)
        return float(y.mean()-fallback*x.mean()),fallback,"nonpositive fitting correlation; positive SD fallback"
    return float(y.mean()-b*x.mean()),b,"positive fitting alignment"


def metrics(p,fp,wp,cutoff,arm,setting,rank,link,seed_value,where,split,fit_info):
    rows=[]
    for name,mask in (("all",np.ones(len(p["c"]),bool)),("stable",p["category"]<2),
                      ("small",p["category"]==2),("large",p["category"]==3)):
        if not mask.any(): continue
        row=dict(setting=setting,rank=rank,links=link,seed=seed_value,arm=arm,where=where,split=split,
                 stratum=name,count=int(mask.sum()),cutoff=float(cutoff),**fit_info)
        for target,pred in (("f",fp),("r",fp),("mu",wp),("w",wp)):
            true=p[target][mask];estimated=pred[mask]
            row[target+"_mse"]=float(np.mean((estimated-true)**2))
            row["delta_"+target+"_mse"]=float(np.mean((np.diff(estimated,axis=1)-np.diff(true,axis=1))**2))
        df=np.diff(p["f"][mask],axis=1).reshape(-1);dp=np.diff(fp[mask],axis=1).reshape(-1)
        row["change_bias"]=float(np.mean(dp-df))
        row["change_slope"]=float(np.dot(df-df.mean(),dp-dp.mean())/max(np.dot(df-df.mean(),df-df.mean()),1e-12))
        row["sign_accuracy"]=float(np.mean(np.sign(dp)==np.sign(df))) if name!="stable" else None
        row["stable_movement_rms"]=float(np.sqrt(np.mean(dp**2))) if name=="stable" else None
        row["detection_rate"]=float(np.mean(np.abs(dp)>cutoff))
        rows.append(row)
    return rows


def eval_arm(args,scorer,head,source_fit,source_test,target_cal,target_test,
             source_threshold,target_threshold,setting,rank,link,seed_value,arm,info,
             center,scale,device,q,vstar):
    fit_score=score_numpy(scorer,source_fit,center,scale,device)
    offset,slope,alignment_status=fit_alignment(fit_score,source_fit)
    results=[]
    for where,p,stable,cal in (("source",source_test,source_threshold,None),
                               ("target",target_test,target_threshold,target_cal)):
        used_head=head
        if cal is not None:
            # Fresh held-out-city heads: scorer remains frozen, no final test labels.
            score_cal=score_numpy(scorer,cal,center,scale,device)
            c0=float(head.center) if head.model=="M1" else float(score_cal.mean())
            cs=float(head.scale) if head.model=="M1" else max(float(score_cal.std()),1e-5)
            knots=np.quantile((score_cal-c0)/cs,[.2,.5,.8]) if head.model=="M3" else []
            used_head=Head(head.model,int(cal["c"].max())+1,knots,c0,cs).to(device)
            if head.model=="M1":
                with torch.no_grad():used_head.raw.copy_(head.raw)
            d=tensor_panel(cal,device,center,scale)
            common=float(F.softplus(head.raw)[0,0]) if head.model=="M1" else None
            fit_frozen(used_head,torch.as_tensor(score_cal,dtype=torch.float64,device=device),d,q,vstar,args.kappa,
                       fixed_m1_slope=common)
        stable_score=score_numpy(scorer,stable,center,scale,device)
        threshold=float(np.quantile(np.abs(np.diff(offset+slope*stable_score,axis=1)),.95,method="higher"))
        for split,part in (("train",source_fit if where=="source" else target_cal),("test",p)):
            s=score_numpy(scorer,part,center,scale,device)
            fp=offset+slope*s;wp=head_numpy(used_head,s,part,device)
            results.extend(metrics(part,fp,wp,threshold,arm,setting,rank,link,seed_value,where,split,
                                   {**info,"alignment_status":alignment_status}))
    return results


def eval_reference(args,source_fit,source_test,target_cal,target_test,source_threshold,target_threshold,
                   setting,seed_value,shape,family,pars,benchmark_info):
    results=[]
    args.rho_current=SETTINGS[setting-1][3]
    target_pars,target_info=target_parametric(args,target_cal,shape,family,pars)
    for arm in ("estimated_parametric","true_parameter_oracle"):
        for where,test,cal,stable in (("source",source_test,source_fit,source_threshold),
                                       ("target",target_test,target_cal,target_threshold)):
            pp=pars if where=="source" else target_pars
            if arm=="true_parameter_oracle":
                sf=stable["f"]
            else: sf,_=predict_benchmark(stable,shape,pp,args)
            threshold=float(np.quantile(np.abs(np.diff(sf,axis=1)),.95,method="higher"))
            for split,p in (("train",cal),("test",test)):
                if arm=="true_parameter_oracle": fp,wp=p["f"],p["mu"]
                else: fp,wp=predict_benchmark(p,shape,pp,args)
                info={"benchmark_convergence":benchmark_info["convergence"],
                      "target_benchmark_convergence":target_info["message"]} if arm=="estimated_parametric" else {}
                results.extend(metrics(p,fp,wp,threshold,arm,setting,"shared","shared",seed_value,
                                       where,split,info))
    return results,target_info


def plot(rows,out,smoke):
    title="IMPLEMENTATION CHECK / NOT SCIENTIFIC EVIDENCE" if smoke else "Paired Monte Carlo pilot"
    test=[r for r in rows if r["split"]=="test" and r["where"]=="target" and r["stratum"]=="all"]
    arms=["A","B","C","D","O","scratch_C","scratch_D","estimated_parametric","true_parameter_oracle"]
    colors={a:plt.cm.tab10(i%10) for i,a in enumerate(arms)}
    fig,axs=plt.subplots(2,2,figsize=(15,9),sharey=True)
    for ax,(rank,link) in zip(axs.flat,((r,l) for r in ("ranknet","sigmoid") for l in (0,1))):
        group=[r for r in test if r["rank"]==rank and r["links"]==link]
        for i,arm in enumerate(arms):
            points=[r for r in group if r["arm"]==arm]
            if arm in ("estimated_parametric","true_parameter_oracle"):
                points=[r for r in test if r["arm"]==arm]
            vals=[np.mean([math.sqrt(r["delta_f_mse"]) for r in points if r["setting"]==s])
                  if any(r["setting"]==s for r in points) else np.nan for s in range(1,9)]
            ax.plot(np.arange(8)+1,vals,marker="o",markersize=3,label=arm,color=colors[arm])
        ax.set(title=f"{rank}, cross-city {'on' if link else 'off'}",xlabel="Ladder setting",ylabel="Target test delta f RMSE")
    handles,labels=axs.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=5)
    fig.suptitle(title);fig.tight_layout(rect=(0,.09,1,.94));fig.savefig(out/"delta_f_ladder.png",dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(14,10),sharex=True,sharey=True)
    for ax,(rank,link) in zip(axs.flat,((r,l) for r in ("ranknet","sigmoid") for l in (0,1))):
        for arm in arms:
            points=[r for r in test if r["arm"]==arm and r["rank"]==rank and r["links"]==link]
            if arm in ("estimated_parametric","true_parameter_oracle"):
                points=[r for r in test if r["arm"]==arm]
            ax.scatter([math.sqrt(r["delta_f_mse"]) for r in points],
                       [math.sqrt(r["delta_mu_mse"]) for r in points],label=arm,s=22,alpha=.7,color=colors[arm])
        ax.set(title=f"{rank}, cross-city {'on' if link else 'off'}",
               xlabel="Target test delta f RMSE",ylabel="Target test delta mu RMSE")
    handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=5)
    fig.suptitle(title);fig.tight_layout(rect=(0,.08,1,.94));fig.savefig(out/"latent_outcome_tradeoff.png",dpi=150);plt.close(fig)
    fig,axs=plt.subplots(4,3,figsize=(16,16),sharex=True,sharey=True)
    for row,(rank,link) in enumerate((r,l) for r in ("ranknet","sigmoid") for l in (0,1)):
        for col,stratum in enumerate(("stable","small","large")):
            ax=axs[row,col]
            for arm in arms:
                points=[r for r in rows if r["split"]=="test" and r["where"]=="target" and
                        r["stratum"]==stratum and r["arm"]==arm and r["rank"]==rank and r["links"]==link]
                if arm in ("estimated_parametric","true_parameter_oracle"):
                    points=[r for r in rows if r["split"]=="test" and r["where"]=="target" and
                            r["stratum"]==stratum and r["arm"]==arm]
                vals=[np.mean([r["detection_rate"] for r in points if r["setting"]==s])
                      if any(r["setting"]==s for r in points) else np.nan for s in range(1,9)]
                ax.plot(np.arange(8)+1,vals,marker="o",markersize=2,label=arm,color=colors[arm])
            if stratum=="stable": ax.axhline(.05,color="black",linestyle="--",linewidth=.8)
            ax.set(title=f"{rank}, links {link}: {stratum}",xlabel="Ladder setting",
                   ylabel="Independent test detection rate",ylim=(0,1))
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=5)
    fig.suptitle(title+" - thresholds from separate stable calibration");fig.tight_layout(rect=(0,.06,1,.97))
    fig.savefig(out/"detection.png",dpi=150);plt.close(fig)


def write_csv(path,rows):
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def paired_contrasts(rows):
    keyed={(r["seed"],r["setting"],r["rank"],r["links"],r["arm"]):r for r in rows
           if r["split"]=="test" and r["where"]=="target" and r["stratum"]=="all" and
           r["arm"] not in ("estimated_parametric","true_parameter_oracle")}
    contrasts=[]
    for (rep,setting,rank,link,arm),row in keyed.items():
        for left,right in (("C","A"),("D","B"),("D","C")):
            if arm!=left: continue
            other=keyed.get((rep,setting,rank,link,right))
            if other is None: continue
            contrasts.append(dict(seed=rep,setting=setting,rank=rank,links=link,contrast=f"{left}-{right}",
                delta_f_rmse_difference=math.sqrt(row["delta_f_mse"])-math.sqrt(other["delta_f_mse"]),
                delta_mu_rmse_difference=math.sqrt(row["delta_mu_mse"])-math.sqrt(other["delta_mu_mse"])))
    return contrasts


def run_setting(args,setting,rep,out,device):
    shape,family,mechanism,rho=SETTINGS[setting]
    base_seed=seed(args.seed,rep)
    fit=panel(args,shape,family,mechanism,rho,"fit",args.cities,base_seed)
    val=panel(args,shape,family,mechanism,rho,"validation",args.cities,base_seed)
    source_test=panel(args,shape,family,mechanism,rho,"source_test",args.cities,base_seed)
    source_threshold=panel(args,shape,family,mechanism,rho,"source_stable_threshold",args.cities,base_seed)
    target_cal=panel(args,shape,family,mechanism,rho,"target_calibration",args.target_cities,base_seed)
    target_test=panel(args,shape,family,mechanism,rho,"target_test",args.target_cities,base_seed)
    target_threshold=panel(args,shape,family,mechanism,rho,"target_stable_threshold",args.target_cities,base_seed)
    # Threshold observations must all be structurally stable; select only assigned stable tracts.
    for name,p in (("source_threshold",source_threshold),("target_threshold",target_threshold)):
        mask=p["category"]<2
        for key in ("h","x","f","r","w","mu","c","category"):
            p[key]=p[key][mask]
    xcenter=fit["x"].reshape(-1,7).mean(0)
    xscale=np.maximum(fit["x"].reshape(-1,7).std(0),1e-6)
    td=tensor_panel(fit,device,xcenter,xscale);vd=tensor_panel(val,device,xcenter,xscale)
    # A deterministic reference at rho=0 fixes the normalizer across persistence variants.
    marginal=panel(args,shape,family,mechanism,0.,"fit",args.cities,base_seed)
    cell=2*marginal["c"][:,None]+np.arange(2)
    vstar=float(np.mean([np.var(marginal["w"].reshape(-1)[cell.reshape(-1)==i]) for i in range(2*args.cities)]))
    if vstar<=0: raise RuntimeError("Zero fitting within-cell outcome variance")
    provenance=dict(setting=setting+1,shape=shape,map=family,change_mechanism=mechanism,rho=rho,
                    rejection_rate=fit["rejection_rate"],rejection_attempts=fit["attempts"],
                    endpoint_min=fit["endpoint_min"],endpoint_max=fit["endpoint_max"],
                    empirical_residual_corr=fit["empirical_residual_corr"],vstar=vstar,
                    effective_coefficients={"levels":dict(average=args.kappa*(1-args.q_levels)/vstar,
                                                           difference=args.kappa*args.q_levels/(4*vstar)),
                                            "changes":dict(average=args.kappa*(1-args.q_changes)/vstar,
                                                           difference=args.kappa*args.q_changes/(4*vstar))})
    rows=[]; audit=[]
    bfile=out/f"benchmark_seed{rep}_setting{setting+1}.json"
    rfile=out/f"benchmark_seed{rep}_setting{setting+1}.csv"
    if bfile.exists() and rfile.exists():
        saved=json.loads(bfile.read_text(encoding="utf-8"))
        pars={k:np.asarray(v) for k,v in saved["parameters"].items()}
        benchmark_info=saved["fit"]
        with rfile.open(newline="",encoding="utf-8") as f: rows.extend(csv.DictReader(f))
    else:
        pars,benchmark_info=benchmark(args,fit,shape,family,rho,args.benchmark_steps)
        reference_rows,target_info=eval_reference(args,fit,source_test,target_cal,target_test,
            source_threshold,target_threshold,setting+1,rep,shape,family,pars,benchmark_info)
        rows.extend(reference_rows)
        bfile.write_text(json.dumps(dict(parameters={k:v.tolist() for k,v in pars.items()},fit=benchmark_info,
                                         target_fit=target_info),indent=2),encoding="utf-8")
        write_csv(rfile,reference_rows)
    for rank in ("ranknet","sigmoid"):
        for link in (0,1):
            stem=f"seed{rep}_setting{setting+1}_{rank}_links{link}"
            output=out/(stem+".csv");diagnostics=out/(stem+".json")
            if output.exists() and diagnostics.exists():
                with output.open(newline="",encoding="utf-8") as f: rows.extend(csv.DictReader(f))
                audit.append(json.loads(diagnostics.read_text(encoding="utf-8")))
                continue
            torch.manual_seed(seed(base_seed,"initial",shape,mechanism))
            initial=Scorer().double().to(device)
            link_pairs,link_audit=links(fit,device,xcenter,xscale,link,base_seed)
            started=time.perf_counter()
            a,_,a_info=train(initial,None,td,vd,link_pairs,args,rank,args.q_levels,vstar,"ordinal",base_seed)
            s=score(a,td).detach()
            frozen_state=copy.deepcopy(a.state_dict())
            ah=new_head(family,a,td);a_head=fit_frozen(ah,s,td,args.q_levels,vstar,args.kappa)
            bh=new_head(family,a,td);b_head=fit_frozen(bh,s,td,args.q_changes,vstar,args.kappa)
            # Head QP must never modify the common scorer.
            for key,value in frozen_state.items():
                torch.testing.assert_close(value,a.state_dict()[key],rtol=0,atol=0)
            c,ch,c_info=train(a,ah,td,vd,link_pairs,args,rank,args.q_levels,vstar,"joint",base_seed)
            d,dh,d_info=train(a,bh,td,vd,link_pairs,args,rank,args.q_changes,vstar,"joint",base_seed)
            arms=[("A",a,ah,a_info,args.q_levels),("B",a,bh,b_head,args.q_changes),
                  ("C",c,ch,c_info,args.q_levels),("D",d,dh,d_info,args.q_changes)]
            if args.preset in ("controlled128","full144"):
                o,_,o_info=train(a,None,td,vd,link_pairs,args,rank,args.q_levels,vstar,"continuation",base_seed)
                oh=new_head(family,o,td)
                fit_frozen(oh,score(o,td).detach(),td,args.q_levels,vstar,args.kappa)
                arms.append(("O",o,oh,o_info,args.q_levels))
            if args.preset=="full144" and setting in (0,7):
                scratch_head=new_head(family,initial,td)
                sc,sch,sc_info=train(initial,scratch_head,td,vd,link_pairs,args,rank,args.q_levels,vstar,"joint",base_seed)
                sd,sdh,sd_info=train(initial,scratch_head,td,vd,link_pairs,args,rank,args.q_changes,vstar,"joint",base_seed)
                arms.extend((("scratch_C",sc,sch,sc_info,args.q_levels),
                             ("scratch_D",sd,sdh,sd_info,args.q_changes)))
            result=[]
            for arm,model,head,information,q in arms:
                result.extend(eval_arm(args,model,head,fit,source_test,target_cal,target_test,
                    source_threshold,target_threshold,setting+1,rank,link,rep,arm,
                    {"best_step":information.get("best_step"),"fit_status":information.get("convergence","QP success")},
                    xcenter,xscale,device,q,vstar))
            write_csv(output,result)
            entry=dict(stem=stem,elapsed_seconds=time.perf_counter()-started,provenance=provenance,
                       arms={name:info for name,_,_,info,_ in arms},links=link_audit,
                       frozen_heads={"A":a_head,"B":b_head})
            diagnostics.write_text(json.dumps(entry,indent=2),encoding="utf-8")
            rows.extend(result);audit.append(entry)
            print(f"completed {stem}: {entry['elapsed_seconds']:.1f}s",flush=True)
    return rows,audit


def numeric_rows(rows):
    for r in rows:
        for key,value in list(r.items()):
            if key in ("setting","seed","links","count") and value not in ("shared",None,""):
                r[key]=int(value)
            elif key.endswith(("_mse","_rate","_bias","_slope","_accuracy","_rms")) or key=="cutoff":
                if value not in (None,""):r[key]=float(value)
    return rows


def self_check():
    """Fast deterministic numerical assertions; does not train or run the Monte Carlo."""
    class Args: pass
    a=Args();a.n=16;a.eval_n=16;a.sigma=.8
    for shape,model,mechanism,rho in SETTINGS:
        p=panel(a,shape,model,mechanism,rho,"fit",4,7)
        delta=np.diff(p["f"],axis=1).ravel()
        assert np.max(np.abs(delta[p["category"]<2]))<1e-12
        assert np.all((np.abs(delta[p["category"]==2])>=.12)&(np.abs(delta[p["category"]==2])<.38))
        assert np.all((np.abs(delta[p["category"]==3])>=.48)&(np.abs(delta[p["category"]==3])<1.10))
        if mechanism=="nonlinear": np.testing.assert_array_equal(p["h"][:,0,0],p["h"][:,1,0])
    e=torch.tensor([[2.,4.],[3.,5.]],dtype=torch.float64)
    got=cardinal(e,torch.zeros_like(e),2.,1.,.5)
    torch.testing.assert_close(got,e.square().mean()/4) # endpoint MSE times kappa/(2*vstar)
    a0=np.array([.2]);b0=np.array([2.]);k0=np.array([.6]);r=np.array([1.3])
    w=psi(r,a0,b0,k0);y=(w-a0)/b0;inv=y.copy()
    for _ in range(20): inv-=(inv+k0*np.tanh(inv)-y)/(1+k0*(1-np.tanh(inv)**2))
    np.testing.assert_allclose(inv,r,atol=1e-12)
    jac=b0*(1+k0*(1-np.tanh(r)**2))
    numerical=(psi(r+1e-6,a0,b0,k0)-psi(r-1e-6,a0,b0,k0))/2e-6
    np.testing.assert_allclose(jac,numerical,rtol=1e-9)
    print("self-check passed: strata/stability/geometry, cardinal algebra, nonlinear inverse/Jacobian")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset",choices=("core96","controlled128","full144"),default="full144")
    parser.add_argument("--plan",action="store_true")
    parser.add_argument("--self-check",action="store_true")
    parser.add_argument("--smoke",action="store_true",help="tiny implementation check; all grid/arm paths remain")
    parser.add_argument("--seeds",type=int,default=1)
    parser.add_argument("--seed",type=int,default=271828)
    parser.add_argument("--cities",type=int,default=16)
    parser.add_argument("--target-cities",type=int,default=4)
    parser.add_argument("--n",type=int,default=128)
    parser.add_argument("--eval-n",type=int,default=64)
    parser.add_argument("--ord-steps",type=int,default=400)
    parser.add_argument("--joint-steps",type=int,default=200)
    parser.add_argument("--benchmark-steps",type=int,default=160)
    parser.add_argument("--check-every",type=int,default=25)
    parser.add_argument("--lr",type=float,default=.001)
    parser.add_argument("--head-lr",type=float,default=.003)
    parser.add_argument("--lambda-t",type=float,default=.3)
    parser.add_argument("--lambda-c",type=float,default=.3)
    parser.add_argument("--kappa",type=float,default=1.)
    parser.add_argument("--q-levels",type=float,default=.5)
    parser.add_argument("--q-changes",type=float,default=.8)
    parser.add_argument("--tau",type=float,default=.3)
    parser.add_argument("--sigma",type=float,default=.8)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--output",type=Path,default=ROOT/"joint_two_step_results")
    args=parser.parse_args()
    if args.smoke:
        # Explicit small-data/short-optimization override, not reduced algorithmic coverage.
        args.cities,args.target_cities,args.n,args.eval_n=4,2,16,16
        args.ord_steps,args.joint_steps,args.benchmark_steps,args.check_every=5,3,12,1
    if args.n%4 or args.eval_n%4 or args.cities<2 or args.target_cities<2:
        parser.error("n/eval-n must be multiples of four; at least two source/target cities")
    if not (args.seeds>0 and args.n>0 and args.eval_n>0 and args.ord_steps>0 and
            args.joint_steps>0 and args.benchmark_steps>0 and args.check_every>0 and
            args.tau>0 and args.lr>0 and args.head_lr>0 and args.kappa>0 and
            args.lambda_t>=0 and args.lambda_c>=0 and
            0<args.q_levels<1 and 0<args.q_changes<1 and args.sigma>0):
        parser.error("seeds, sample sizes, steps, rates, tau, kappa and sigma must be positive; q in (0,1)")
    per={"core96":96,"controlled128":128,"full144":144}[args.preset]
    print(f"plan: {8*2*2} configurations/seed; {per} scorer stages/seed; "
          f"64 frozen-head QPs/seed plus target refits; 8 source and 8 target known-noise parametric likelihood jobs/seed; "
          f"{args.seeds} seed(s). Preset={args.preset}; smoke={args.smoke}")
    if args.plan: return
    if args.self_check: self_check();return
    device="cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)
    if device=="cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    manifest={k:v for k,v in vars(args).items() if k!="plan"};manifest["output"]=str(args.output)
    manifest["settings"]=[list(s) for s in SETTINGS];manifest["device_used"]=device
    manifest["source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    file=args.output/"manifest.json"
    try: check_resume_manifest(file,manifest)
    except ValueError as exc: parser.error(str(exc))
    file.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    started=time.perf_counter();rows=[];audits=[]
    for rep in range(1,args.seeds+1):
        for setting in range(8):
            rr,aa=run_setting(args,setting,rep,args.output,device)
            rows.extend(rr);audits.extend(aa)
    rows=numeric_rows(rows)
    write_csv(args.output/"metrics.csv",rows)
    contrasts=paired_contrasts(rows)
    write_csv(args.output/"paired_contrasts.csv",contrasts)
    plot(rows,args.output,args.smoke)
    elapsed=time.perf_counter()-started
    report=["# Joint versus two-step ordinal pilot", "",
            "**IMPLEMENTATION CHECK / NOT SCIENTIFIC EVIDENCE**" if args.smoke else "Fixed-hyperparameter Monte Carlo pilot", "",
            f"Preset {args.preset}; {args.seeds} seed(s); {per*args.seeds} scorer stages planned; "
            f"{len(audits)} configurations completed; elapsed {elapsed:.1f}s on {device}.","",
            "Known-noise parametric benchmark estimates structural coefficients and exact cell maps by paired Gaussian likelihood; "
            "the true-parameter oracle predicts f and E[W|H,cell]. Both are distinct from realized R and W.","",
            "NN latent metrics use a single fitting-only evaluation affine alignment per scorer. Target heads use separate local calibration. "
            "Thresholds use independent stable calibration panels; no final test labels select parameters.","",
            "See metrics.csv, paired_contrasts.csv, per-configuration JSON diagnostics and the three PNG figures. "
            "Short smoke optimizer budgets do not certify convergence. Parametric convergence fields must be checked before performance interpretation.","",
            "M3 generating map: Psi_ct(r)=a_ct+b_ct[r+k_ct tanh(r)], b_ct>0, k_ct>0; "
            "its conditional mean integrates Gaussian error with 32-point Gauss-Hermite quadrature.","",
            "Persistent eta has fixed marginal sigma. Under changing slopes or nonlinear maps it need not cancel in delta W."]
    report.extend(["","## Target test summaries", "",
                   "These are descriptive means over available grid cells; matched contrasts are in the raw CSV. "
                   "A single seed supplies no uncertainty interval.","",
                   "| Arm | delta f RMSE | delta mu RMSE | Stable FPR | Small sensitivity | Large sensitivity |",
                   "|---|---:|---:|---:|---:|---:|"])
    for arm in ("A","B","C","D","O","scratch_C","scratch_D","estimated_parametric","true_parameter_oracle"):
        group=[r for r in rows if r["arm"]==arm and r["split"]=="test" and r["where"]=="target"]
        if not group: continue
        def average(stratum,key,transform=lambda x:x):
            values=[transform(r[key]) for r in group if r["stratum"]==stratum]
            return f"{np.mean(values):.4f}" if values else "n/a"
        report.append(f"| {arm} | {average('all','delta_f_mse',math.sqrt)} | "
                      f"{average('all','delta_mu_mse',math.sqrt)} | {average('stable','detection_rate')} | "
                      f"{average('small','detection_rate')} | {average('large','detection_rate')} |")
    report.extend(["","Scratch arms cover only settings 1 and 8; their averages are not comparable "
                   "to eight-setting averages without matching the settings.","",
                   "Paired C-A, D-B and D-C contrasts are keyed by seed, setting, rank loss and link regime. "
                   "With fewer than three seeds, no uncertainty interval is reported."])
    (args.output/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(f"complete: {args.output}; elapsed {elapsed:.1f}s",flush=True)


if __name__=="__main__": main()
