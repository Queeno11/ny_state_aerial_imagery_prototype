"""Conditional Gaussian likelihood with oracle pair covariance shape and unknown SD."""
import numpy as np
from scipy.optimize import minimize

from prior import SIMPLE
from classical import Model, gaussian
from losses import SLOPE_FLOOR, fit_head, minimize_sphere, stable, unit


def precision(value, rho):
    before,after=value[...,0],value[...,1]
    den=1-rho**2
    return np.stack(((before-rho*after)/den,(after-rho*before)/den),axis=2)


def weighted_mean(value,rho):
    # An intercept's GLS weights differ between stable and changed pairs.
    return np.sum((value[...,0]+value[...,1])/(1+rho),axis=1)/np.sum(2/(1+rho),axis=1)


def inner(x,y,rho):
    return np.mean(x*precision(y,rho),axis=(1,2))


def heads(beta,data,rho):
    r=data.x@beta
    w=data.w
    mr,mw=weighted_mean(r,rho),weighted_mean(w,rho)
    rc,wc=r-mr[:,None,None],w-mw[:,None,None]
    a11,t1=inner(rc,rc,rho),inner(rc,wc,rho)
    if data.case=="A":
        raise ValueError("A uses unrestricted raw beta and an intercept")
    if data.case=="B":
        u,v=np.maximum(t1/a11,SLOPE_FLOOR),np.zeros_like(t1)
        m3=np.zeros_like(mr)
    else:
        r3=r**3
        m3=weighted_mean(r3,rho)
        r3c=r3-m3[:,None,None]
        a12,a22,t2=inner(rc,r3c,rho),inner(r3c,r3c,rho),inner(r3c,wc,rho)
        determinant=a11*a22-a12**2
        if np.min(determinant)<=0:
            raise ArithmeticError("Singular correlated cubic head")
        uf,vf=(t1*a22-t2*a12)/determinant,(t2*a11-t1*a12)/determinant
        candidates=np.stack((np.column_stack((uf,vf)),
                             np.column_stack((np.maximum(t1/a11,SLOPE_FLOOR),np.zeros_like(t1))),
                             np.column_stack((np.full_like(t1,SLOPE_FLOOR),
                                              np.maximum((t2-a12*SLOPE_FLOOR)/a22,0.)))),axis=1)
        uu,vv=candidates[...,0],candidates[...,1]
        objective=a11[:,None]*uu**2+2*a12[:,None]*uu*vv+a22[:,None]*vv**2-2*t1[:,None]*uu-2*t2[:,None]*vv
        objective[(uu<SLOPE_FLOOR)|(vv<0)]=np.inf
        u,v=candidates[np.arange(len(w)),np.argmin(objective,axis=1)].T
    a=mw-u*mr-v*m3
    return np.column_stack((a,u,v))


def a_heads(beta,data,rho):
    a=weighted_mean(data.w-data.x@beta,rho)
    return np.column_stack((a,np.ones_like(a),np.zeros_like(a)))


def likelihood(beta,data,rho,head,stability_weight,raw_a=False):
    r=data.x@beta
    a,u,v=(head[:,i,None,None] for i in range(3))
    residual=a+u*r+v*r**3-data.w
    q=precision(residual,rho)
    sigma2=np.mean(residual*q,axis=(1,2))
    if np.min(sigma2)<1e-10:
        raise ArithmeticError("Degenerate correlated residual variance")
    value=.5*np.mean(np.log(sigma2)+1+np.log(2*np.pi))+.25*np.mean(np.log(1-rho**2))
    derivative=(u+3*v*r**2)[...,None]*data.x
    gradient=np.mean(q[...,None]*derivative/sigma2[:,None,None,None],axis=(0,1,2))
    if stability_weight:
        z=unit(beta) if raw_a else beta
        penalty,g=stable(z,data)
        if raw_a:
            g=(g-z*np.dot(g,z))/np.linalg.norm(beta)
        value+=stability_weight*penalty
        gradient+=stability_weight*g
    return float(value),gradient,np.sqrt(sigma2)


def fit(data,changed,cfg,start,grid,stability_weight=0.):
    rho=np.where(changed,0.,.5)
    if data.case=="A":
        def objective(beta):
            value,gradient,_=likelihood(beta,data,rho,a_heads(beta,data,rho),stability_weight,raw_a=True)
            return value,gradient
        starts=[start,.5*start,1.5*start,-start]
        results=[minimize(objective,x,jac=True,method="BFGS",
                          options=dict(maxiter=cfg["optimizer_maxiter"],gtol=3e-8)) for x in starts]
        best=min(results,key=lambda x:x.fun)
        beta=best.x
        value,gradient=objective(beta)
        if np.linalg.norm(gradient)>1e-5:
            raise ArithmeticError("Correlated A solution failed stationarity")
        head=a_heads(beta,data,rho)
        info=dict(objective=value,min_tangent_grad=float(np.linalg.norm(gradient)),starts=4,
                  evaluations=sum(x.nfev for x in results),successful_starts=sum(x.success for x in results),
                  start_best_spread=float(np.ptp([x.fun for x in results])))
    else:
        def objective(beta):
            value,gradient,_=likelihood(beta,data,rho,heads(beta,data,rho),stability_weight)
            return value,gradient
        values=[objective(beta)[0] for beta in grid]
        beta,info=minimize_sphere(objective,[start,-start],grid,values,cfg["optimizer_maxiter"])
        head=heads(beta,data,rho)
    _,_,sigma=likelihood(beta,data,rho,head,stability_weight,raw_a=data.case=="A")
    return Model(beta,head,sigma=sigma),info


def check(data,changed,beta):
    from losses import angles,sphere
    rho=np.where(changed,0.,.5)
    if data.case=="A":
        point=beta
        direction=np.array([.3,.4,.5])
        direction/=np.linalg.norm(direction)
        def objective(x):
            return likelihood(x,data,rho,a_heads(x,data,rho),0.,raw_a=True)[:2]
        analytic=float(objective(point)[1]@direction)
    else:
        point=angles(beta)
        direction=np.array([.6,.8])
        def objective(x):
            b,jac=sphere(x)
            val,grad,_=likelihood(b,data,rho,heads(b,data,rho),0.)
            return val,jac@grad
        analytic=float(objective(point)[1]@direction)
    h=1e-5
    finite=(objective(point+h*direction)[0]-objective(point-h*direction)[0])/(2*h)
    assert abs(analytic-finite)<2e-7
    zero=np.zeros_like(rho)
    head0=a_heads(beta,data,zero) if data.case=="A" else heads(beta,data,zero)
    if data.case!="A":
        np.testing.assert_allclose(head0,fit_head(beta,data),rtol=1e-12,atol=1e-12)
    value,g,sd=likelihood(beta,data,zero,head0,0.,raw_a=data.case=="A")
    old_value,old_g,old_sd=gaussian(beta,data,head0,0.,raw_a=data.case=="A")
    np.testing.assert_allclose([value],[old_value],atol=1e-13)
    np.testing.assert_allclose(g,old_g,atol=1e-12)
    np.testing.assert_allclose(sd,old_sd,atol=1e-13)
    return dict(directional_error=abs(analytic-finite),zero_correlation_matches_independent=True,
                variance_profile="mean residual transpose R_inverse residual over city endpoints")
