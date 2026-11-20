"""G_ct: pooled monotone conditional-mean calibration, scalar score + cell only."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CalibrationHead(nn.Module):
    def __init__(self,scores,cells,outcomes,kind,settings):
        super().__init__()
        scores=np.asarray(scores,float);cells=np.asarray(cells,int);outcomes=np.asarray(outcomes,float)
        if kind not in ('affine','spline'): raise ValueError('Unknown calibration head')
        if scores.ndim!=1 or cells.shape!=scores.shape or outcomes.shape!=scores.shape: raise ValueError('Head takes one scalar score and one cell id per observation')
        self.kind=kind
        center=float(scores.mean());scale=float(scores.std());yscale=float(outcomes.std())
        if min(scale,yscale)<=0: raise ValueError('Degenerate fitting score/outcome dispersion')
        z=(scores-center)/scale
        knots=np.unique(np.quantile(z,settings['knots'])) if kind=='spline' else np.empty(0)
        ncell=int(cells.max())+1;counts=np.bincount(cells,minlength=ncell)
        if np.any(counts==0): raise ValueError('Missing fitting cell')
        for name,value in dict(score_center=center,score_scale=scale,y_center=float(outcomes.mean()),
                               y_scale=yscale,knots=knots,pool_weights=16./counts).items():
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float64))
        self.global_intercept=nn.Parameter(torch.zeros((),dtype=torch.float64))
        self.cell_intercepts=nn.Parameter(torch.zeros(ncell,dtype=torch.float64))
        self.global_raw_slopes=nn.Parameter(torch.full((len(knots)+1,),.541324854612918,dtype=torch.float64))
        self.cell_raw_slopes=nn.Parameter(torch.zeros(ncell,len(knots)+1,dtype=torch.float64))
        self.pool_relative=settings['pool_relative']
        normalized=(outcomes-outcomes.mean())/yscale
        with torch.no_grad():
            self.cell_intercepts.copy_(torch.tensor([np.mean(normalized[cells==c]-z[cells==c]) for c in range(ncell)]))

    def slopes(self):
        return F.softplus(self.global_raw_slopes[None,:]+self.cell_raw_slopes)

    def forward(self,score,cell):
        if score.ndim!=1 or cell.shape!=score.shape: raise ValueError('G receives scalar scores and cell ids, never hidden features')
        z=(score-self.score_center)/self.score_scale;m=self.slopes()[cell]
        value=self.global_intercept+self.cell_intercepts[cell]+m[:,0]*z
        if self.knots.numel():
            value=value+((m[:,1:]-m[:,:-1])*F.relu(z[:,None]-self.knots)).sum(1)
        return self.y_center+self.y_scale*value

    def penalty(self):
        m=self.slopes()
        curvature=(m[:,1:]-m[:,:-1]).square().mean() if m.shape[1]>1 else m.new_tensor(0.)
        pooling=(self.pool_weights*(self.cell_intercepts.square()+self.cell_raw_slopes.square().mean(1))).mean()
        return curvature+self.pool_relative*pooling

    @torch.no_grad()
    def audit_monotonicity(self):
        z=torch.linspace(-8,8,161,dtype=self.y_center.dtype,device=self.y_center.device)
        for cell in range(len(self.cell_intercepts)):
            s=self.score_center+self.score_scale*z;c=torch.full((len(s),),cell,dtype=torch.long,device=s.device)
            differences=torch.diff(self(s,c))
            if not bool(torch.all(differences>0)): raise AssertionError('Non-monotone or nonfinite calibration head')
        return float(self.slopes().min())
