"""Optional real-PyTorch regression checks for collapsed-score recovery.

Run: python verification/check_dispersion_recovery.py
Injects deterministic invalid optimizer steps; does not touch result folders.
"""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from ordinal_mc import core

def main():
    torch.set_num_threads(1);torch.manual_seed(8675309)
    x=torch.randn(40,3,dtype=torch.float64)
    x[20:]=x[:20];x[20:,2]+=1.
    i=torch.arange(20);j=torch.roll(i,1)
    data=dict(x=x,i=i,j=j,y=torch.sign(x[j,0]-x[i,0]),
        weights=torch.full((20,),1/20,dtype=torch.float64),cells=torch.zeros(20,dtype=torch.long))
    edges=dict(temporal=torch.stack((i,i+20),dim=1),cross_city=torch.empty((0,2),dtype=torch.long))
    cfg=SimpleNamespace(device='cpu',torch_dtype=torch.float64,hidden=[8],weight_bound=10.,cities=4,
        lr=.01,tau=.1,epochs=2,lbfgs_steps=3,audit_steps=3,grad_tol=1e-5,dtype='float64',audit_tol=1e-5)
    def train():
        model,stats=core.train_candidate(data,edges,'nn','common_logit',.01,(.3,0.),123,cfg)
        assert stats['score_variance']>0
        assert torch.isfinite(torch.tensor(stats['train_objective']))
        return stats
    adam=torch.optim.Adam;lbfgs=torch.optim.LBFGS;estimator=core.Estimator
    class CollapseAdam(adam):
        def step(self,*args,**kwargs):
            result=super().step(*args,**kwargs)
            with torch.no_grad():
                for group in self.param_groups:
                    for parameter in group['params']:parameter.zero_()
            return result
    try:
        torch.optim.Adam=CollapseAdam
        stats=train();assert stats['adam_dispersion_recoveries']>=1
    finally:torch.optim.Adam=adam
    class CollapseProbe(lbfgs):
        def step(self,closure):
            closure()
            with torch.no_grad():
                for group in self.param_groups:
                    for parameter in group['params']:parameter.zero_()
            return closure()
    try:
        torch.optim.LBFGS=CollapseProbe
        stats=train()
        assert stats['lbfgs_dispersion_recoveries']==2
        assert stats['audit_stable'] is False
    finally:torch.optim.LBFGS=lbfgs
    calls=[0]
    def dead_first(*args,**kwargs):
        model=estimator(*args,**kwargs);calls[0]+=1
        if calls[0]==1:
            with torch.no_grad():
                for p in model.parameters():p.zero_()
        return model
    try:
        core.Estimator=dead_first
        stats=train();assert stats['initialization_retries']>=1
    finally:core.Estimator=estimator
    print('PASS: real-PyTorch dead initialization, Adam collapse and L-BFGS probe recovery; final scores stay admissible.')

if __name__=='__main__':main()
