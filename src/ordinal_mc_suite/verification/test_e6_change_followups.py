"""Small numerical checks, not simulation replications."""
import unittest
import numpy as np
import torch
from e6_change_followups import project,residual_stats,moments


class NumericalChecks(unittest.TestCase):
    def test_two_active_constraints(self):
        J=torch.tensor([[1.,0.],[1.,1.]],dtype=torch.float64)
        d=project(torch.tensor([2.,1.],dtype=torch.float64),J,torch.tensor([.1,.2],dtype=torch.float64))
        torch.testing.assert_close(d,torch.tensor([.1,.1],dtype=torch.float64),rtol=0,atol=1e-13)

    def test_free_direction_is_unchanged(self):
        d=torch.tensor([-.1,-.2],dtype=torch.float64)
        torch.testing.assert_close(project(d,torch.eye(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64)),d,rtol=0,atol=0)

    def test_affine_noise_moments(self):
        f=np.array([-2.,0.,1.]);a=np.array([1.,-1.,3.]);b=np.array([2.,3.,4.]);k=np.zeros(3);sd=.8028980597616118
        mu,var=moments(f,a,b,k,sd,1e-11,[-12,45])
        np.testing.assert_array_equal(mu,a+b*f);np.testing.assert_array_equal(var,b*b*sd*sd)

    def test_paired_weights_identity(self):
        e0=np.array([1.,2.,4.]);e1=np.array([-2.,3.,5.]);s=residual_stats(e0,e1)
        self.assertAlmostEqual(s['change_MSE'],np.mean((e1-e0)**2),places=13)
        self.assertAlmostEqual(s['change_bias_squared']+s['change_variance'],s['change_MSE'],places=13)


if __name__=='__main__': unittest.main()
