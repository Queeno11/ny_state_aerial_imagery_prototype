"""Small implementation tests only; no 8/32/64-city Monte Carlo experiments."""
import copy
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ordinal_mc import core
from ordinal_mc.design import PopulationDGP, inverse_measurement
from ordinal_mc.e6.config import DEFAULT, e5_runtime, validate
from ordinal_mc.e6.data import build, conditional_mean, measurement, coefficient_ranges
from ordinal_mc.e6.heads import CalibrationHead
from ordinal_mc.e6 import training
from ordinal_mc.e6.experiment import compute_job
from ordinal_mc.e6.report import interval, paired_differences, export
from ordinal_mc.e6.runner import ledger, jobs
from ordinal_mc.execution import iter_results
from ordinal_mc.e6.evaluation import evaluate


def tiny_config():
    d=copy.deepcopy(DEFAULT)
    d.update(replications=1,cities=[2],cases=[['linear','linear']],labels_per_city=8,
        test_per_city=8,test_pairs=32,restarts=1,workers=1,quadrature_nodes=32,level_weights=[.1,1.],change_strengths=[0.,.4])
    d['baseline'].update(adam_steps=2,check_every=1,lbfgs_steps=2,lbfgs_chunk=1)
    d['head_training'].update(steps=2,check_every=1)
    d['joint'].update(steps=3,check_every=1)
    d['oracle_R'].update(steps=2)
    return d


class E6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=tiny_config();cls.cfg=e5_runtime(cls.d,2);cls.dgp=PopulationDGP(cls.cfg)
        cls.base,cls.variants=build(cls.cfg,cls.dgp,1,'linear',cls.d)
        cls.features=core.Features(cls.base['train']['x'],'linear',cls.cfg)
        cls.data=core.prepare(cls.base,cls.features,cls.cfg)
        cls.edges={k:core.tensor(cls.base['links'][k],cls.cfg,True) for k in ('temporal','cross_city')}

    def model(self):
        torch.manual_seed(123)
        return core.Estimator(7,'linear','common_logit',self.cfg).double()

    def head(self,model,kind='spline'):
        tr=self.variants['nonlinear']['train']
        return CalibrationHead(core.predictions(model,self.features,tr['x'],self.cfg),tr['cell'],tr['w'],kind,self.d['head_training'])

    def test_config_rejects_bad_values(self):
        validate(self.d)
        for key,value in [('level_weights',[0.]),('kappa_base',.1),('false_positive_target',0.)]:
            d=copy.deepcopy(self.d);d[key]=value
            with self.assertRaises(ValueError): validate(d)

    def test_matched_measurements_and_splits(self):
        a=self.variants['affine'];n=self.variants['nonlinear']
        for group in ('train','val'):
            for key in ('x','r','a','b','h'): np.testing.assert_array_equal(a[group][key],n[group][key])
            r=inverse_measurement(n[group]['w'],n[group]['a'],n[group]['b'],n[group]['kappa'],1.,'tanh')
            np.testing.assert_allclose(r,n[group]['r'],atol=1e-12)
        self.assertFalse(set(a['train']['tract_id'])&set(a['val']['tract_id']))
        self.assertFalse(set(a['auxiliary_tract_ids'])&set(a['val']['tract_id']))
        self.assertEqual(a['link_stats']['temporal']['link_gap_rms'],0.)
        for where in ('seen','unseen'):
            p=a['e6_panels'][.4][where];q=n['e6_panels'][.4][where]
            for key in ('x_before','x_after','delta_f','r_before','r_after'): np.testing.assert_array_equal(p[key],q[key])
            # Matching error draws across step sizes too.
            p0=a['e6_panels'][0.][where]
            np.testing.assert_allclose(p['r_after']-p['f_after'],p0['r_after']-p0['f_after'],atol=1e-14)

    def test_affine_oracle_mean_and_nonlinear_quadrature(self):
        f=np.linspace(-3,3,41);a=np.full(41,2.);b=np.full(41,3.);k=np.full(41,.6)
        np.testing.assert_array_equal(conditional_mean(f,a,b,k,'affine',1.2),a+b*f)
        m128=conditional_mean(f,a,b,k,'nonlinear',1.2,128)
        m256=conditional_mean(f,a,b,k,'nonlinear',1.2,256)
        np.testing.assert_allclose(m128,m256,atol=2e-5,rtol=0)
        self.assertGreater(float(np.max(abs(m256-measurement(f,a,b,k,'nonlinear')))),.01)

    def test_head_monotonicity_and_changing_curvature(self):
        m=self.model();h=self.head(m)
        with torch.no_grad(): h.global_raw_slopes.copy_(torch.tensor([-.5,1.,-.2,.8]))
        self.assertGreater(h.audit_monotonicity(),0.)
        slopes=h.slopes()[0].detach().numpy()
        self.assertTrue(np.any(np.diff(slopes)>0) and np.any(np.diff(slopes)<0))
        affine=self.head(m,'affine');self.assertEqual(affine.knots.numel(),0);affine.audit_monotonicity()
        with self.assertRaises(ValueError): h(torch.zeros(4,2),torch.zeros(4,dtype=torch.long))

    def test_knots_and_scaling_use_fitting_scores(self):
        m=self.model();h=self.head(m);scores=core.predictions(m,self.features,self.base['train']['x'],self.cfg)
        expected=np.unique(np.quantile((scores-scores.mean())/scores.std(),self.d['head_training']['knots']))
        np.testing.assert_allclose(h.knots.numpy(),expected)

    def test_level_gradient_and_zero_weight_preserve_e5_gradient(self):
        m=self.model();h=self.head(m);levels=training.level_data(self.variants['nonlinear']['train'],self.cfg)
        training.level_loss(m,h,self.data['x'],levels).backward()
        self.assertGreater(sum(float(p.grad.square().sum()) for p in m.scorer.parameters()),0.)
        m.zero_grad(set_to_none=True)
        training.joint_loss(m,h,self.data,self.edges,levels,self.cfg,self.d,0.).backward()
        actual=[p.grad.clone() for p in m.scorer.parameters()]
        m.zero_grad(set_to_none=True);training.e5_loss(m,self.data,self.edges,self.cfg,self.d).backward()
        for a,p in zip(actual,m.scorer.parameters()): torch.testing.assert_close(a,p.grad,rtol=0,atol=0)

    def test_stability_ratio_differentiates_dispersion(self):
        m=self.model();s=m.scorer(self.data['x']);i=self.data['i'];j=self.data['j'];e=self.edges['temporal']
        v=.5*(self.data['weights']*(s[j]-s[i]).square()).sum()
        ratio=(s[e[:,0]]-s[e[:,1]]).square().mean()/v
        _,parts=core.score_objective(m,self.data,self.edges,(self.d['lambda_t'],0.),self.d['alpha'],self.cfg.tau)
        torch.testing.assert_close(parts[1],self.d['lambda_t']*ratio)
        g1=torch.autograd.grad(parts[1],list(m.scorer.parameters()),retain_graph=True)
        g2=torch.autograd.grad(self.d['lambda_t']*ratio,list(m.scorer.parameters()),retain_graph=True)
        g_detached=torch.autograd.grad(self.d['lambda_t']*ratio*v/v.detach(),list(m.scorer.parameters()))
        for a,b in zip(g1,g2): torch.testing.assert_close(a,b)
        self.assertGreater(sum(float((a-b).square().sum()) for a,b in zip(g1,g_detached)),0.)

    def test_frozen_scorer_and_fixed_validation_rule(self):
        m=self.model();h=self.head(m);before=core.cpu_state(m)
        v=training.validation_data(self.variants['nonlinear']['val'],self.features,self.cfg)
        levels=training.level_data(self.variants['nonlinear']['train'],self.cfg)
        training.fit_head(m,h,self.data,levels,v,self.cfg,self.d,1.)
        for key,value in before.items(): torch.testing.assert_close(value,m.state_dict()[key],rtol=0,atol=0)
        rule=training.observable_rule(m,h,v,levels['variance'],self.d)
        d=copy.deepcopy(self.d);d['level_weights']=[100.]
        self.assertEqual(rule,training.observable_rule(m,h,v,levels['variance'],d))

    def test_budget_and_best_checkpoint(self):
        m=self.model();h=self.head(m);levels=training.level_data(self.variants['nonlinear']['train'],self.cfg)
        v=training.validation_data(self.variants['nonlinear']['val'],self.features,self.cfg)
        info,common=training.continue_ordinal(m,h,self.data,self.edges,levels,v,self.cfg,self.d,3)
        self.assertEqual(info['updates'],3)
        self.assertAlmostEqual(float(training.rank_loss(m,v).detach()),min(x['validation'] for x in info['trace']))
        self.assertAlmostEqual(training.observable_rule(common,h,v,levels['variance'],self.d),info['common_rule_validation'])

    def test_predeclared_coefficient_ranges(self):
        r=coefficient_ranges(self.d)
        self.assertGreater(r['realized']['b']['min'],1.5)
        self.assertGreater(r['realized']['kappa']['min'],.4)
        self.assertLess(r['realized']['kappa']['max'],.8)

    def test_primary_change_estimator_matches_e5(self):
        m=self.model();design=self.variants['affine']
        predict=lambda x:core.predictions(m,self.features,x,self.cfg)
        reference=core.evaluate_predictions(predict,design,self.cfg)
        rows,_,_=evaluate(predict,None,design,self.d,self.cfg,'affine')
        for where in ('seen','unseen'):
            row=next(r for r in rows if r['strength']==.4 and r['where']==where)
            for new,old in [('change_rmse','changed_rmse'),('sign_accuracy','change_direction_accuracy'),('cutoff','change_threshold')]:
                self.assertAlmostEqual(row[new],reference[where+'_'+old],places=12)

    def test_resume_guards(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            p=Path(temp);con=ledger(p,self.d,False);con.close()
            with self.assertRaises(ValueError): ledger(p,self.d,False)
            d=copy.deepcopy(self.d);d['workers']=2
            con=ledger(p,d,True);con.close()
            d['alpha']=.5
            with self.assertRaises(ValueError): ledger(p,d,True)

    def test_replication_intervals(self):
        x=interval([1.,2.,3.]);self.assertEqual(x['n'],3);self.assertEqual(x['mean'],2.)
        self.assertIsNone(interval([1.])['ci_low'])
        rows=[]
        for rep in (1,2,3):
            for arm,value in [('B',2+rep),('C',1+rep),('D',3+rep),('D_common',4+rep)]:
                row=dict(rep=rep,cities=2,truth='linear',model='linear',map='affine',head='affine',strength=.4,where='seen',arm=arm)
                for metric in __import__('ordinal_mc.e6.report',fromlist=['METRICS']).METRICS: row[metric]=value
                rows.append(row)
        paired=paired_differences(pd.DataFrame(rows))
        r=paired[(paired.comparison=='C-B')&(paired.metric=='change_rmse')].iloc[0]
        self.assertEqual(r['mean'],-1.);self.assertEqual(r['n'],3)

    def test_tiny_integration_exports(self):
        # Only two cities, 2--3 optimizer updates: exercises wiring, not MC performance.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            result=compute_job((self.d,jobs(self.d)[0],temp))
            self.assertTrue(all(v for k,v in result['checks'].items() if k!='test_used_for_selection'))
            self.assertFalse(result['checks']['test_used_for_selection'])
            for r in result['metrics']:
                if r['where']=='unseen': self.assertIsNone(r['outcome_change_rmse'])
            c=[r for r in result['metrics'] if r['arm']=='C'];dd=[r for r in result['metrics'] if r['arm']=='D']
            self.assertEqual([r['scorer_updates'] for r in c],[r['scorer_updates'] for r in dd])
            export([result],Path(temp),self.d,1)
            self.assertTrue((Path(temp)/'latent_change.png').is_file())
            self.assertTrue((Path(temp)/'paired_differences.csv').is_file())
            self.assertTrue((Path(temp)/'report.md').is_file())

    def test_tiny_relu_spawn_worker(self):
        d=copy.deepcopy(self.d);d['cases']=[['quadratic','nn']];d['heads']=['spline'];d['level_weights']=[1.]
        iterator=iter_results(compute_job,[(d,jobs(d)[0],None)],workers=2,threads=1)
        try: results=list(iterator)
        finally: iterator.close()
        self.assertEqual(len(results),1)
        self.assertTrue(results[0][1]['checks']['level_gradient'])
        serial=compute_job((d,jobs(d)[0],None))
        for a,b in zip(results[0][1]['metrics'],serial['metrics']):
            for key in ('weight','change_rmse','sign_accuracy','cutoff','stable_rms','outcome_change_rmse'):
                self.assertEqual(a[key],b[key],key)


if __name__=='__main__': unittest.main()
