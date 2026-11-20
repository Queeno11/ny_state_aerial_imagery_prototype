"""Validation-checkpoint correctness and leakage tests; standard unittest runner."""
import copy
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]/'ordinal_mc_suite'
sys.path.insert(0,str(ROOT/'verification'))
import check_validation_stopping as check


class ValidationStoppingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ordinal_mc.config import preset
        d=preset('smoke'); d.update(cities=2,views=1,validation_fraction=.2,
            test_pairs=32,test_per_city=8,epochs=5,lbfgs_steps=5,audit_steps=2)
        cls.cfg=check.base.runtime(d); cls.cfg.fit_cell_effects=False
        cls.design=check.base.make_design(cls.cfg,check.base.PopulationDGP(cls.cfg),1,16,'linear',check.base.scenario())
        cls.features=check.core.Features(cls.design['train']['x'],'nn',cls.cfg)
        cls.data=check.core.prepare(cls.design,cls.features,cls.cfg)
        cls.edges={a:check.core.tensor(cls.design['links'][a],cls.cfg,True) for a in ('temporal','cross_city')}
        cls.val=check.prepare_validation(cls.design,cls.features,cls.cfg)

    def test_tracker_restores_best_validation_instead_of_last_training_step(self):
        model=torch.nn.Linear(1,1,bias=False).double()
        tracker=check.ValidationTracker(patience=2)
        stopped=[]
        for step,loss in enumerate([.4,.3,.32,.35],1):
            with torch.no_grad(): model.weight.fill_(step)
            stopped.append(tracker.observe(model,loss,('adam',step)))
        self.assertEqual(stopped,[False,False,False,True])
        tracker.restore(model)
        self.assertEqual(float(model.weight.detach()),2.)
        self.assertEqual(tracker.best_location,('adam',2))

    def test_min_delta_patience_does_not_discard_best_small_improvement(self):
        model=torch.nn.Linear(1,1,bias=False).double()
        tracker=check.ValidationTracker(patience=2,min_delta=.01)
        for step,loss in enumerate([.5,.495,.494],1):
            with torch.no_grad(): model.weight.fill_(step)
            stop=tracker.observe(model,loss,step)
        self.assertTrue(stop); tracker.restore(model)
        self.assertEqual(float(model.weight.detach()),3.)
        self.assertEqual(tracker.best_loss,.494)

    def test_validation_compares_disjoint_tracts_within_cells(self):
        train,val=self.design['train'],self.design['val']
        self.assertFalse(set(map(tuple,train['h'])) & set(map(tuple,val['h'])))
        i=self.val['i'].numpy(); j=self.val['j'].numpy()
        self.assertTrue(np.all(i!=j))
        self.assertTrue(np.all(val['cell'][i]==val['cell'][j]))
        self.assertLess(max(i.max(),j.max()),len(val['x']))
        np.testing.assert_allclose(self.val['weights'].sum(),1.)
        np.testing.assert_allclose(self.features.mu,train['x'].mean(0))

    def test_validation_targets_do_not_use_latent_truth_or_test_data(self):
        altered=copy.deepcopy(self.design)
        altered['val']['f'][:]=1e9; altered['val']['r'][:]=-1e9
        for test,_ in altered['tests'].values(): test['f'][:]=-1e9; test['w'][:]=1e9
        actual=check.prepare_validation(altered,self.features,self.cfg)
        for key in self.val: self.assertTrue(torch.equal(actual[key],self.val[key]))

    def test_early_stopping_returns_minimum_recorded_validation_loss(self):
        torch.manual_seed(101)
        model=check.core.Estimator(self.data['x'].shape[1],'nn','common_logit',self.cfg).double()
        model,info,trace=check.early_stop(model,self.data,self.edges,(.3,.3),self.val,self.cfg,
            max_adam=30,check_every=5,patience=2,max_lbfgs=20,lbfgs_chunk=5,lbfgs_patience=2)
        loss,_=check.validation(model,self.val)
        self.assertAlmostEqual(loss,min(r['validation_loss'] for r in trace),places=12)
        self.assertAlmostEqual(info['validation_loss'],loss,places=12)
        best=min(trace,key=lambda r:r['validation_loss'])
        self.assertEqual((info['best_stage'],info['best_step']),(best['stage'],best['step']))

    def test_restart_selection_uses_validation_not_training_objective(self):
        low_training=dict(restart=0,model='overfit',info=dict(validation_loss=.8,train_objective=.001))
        low_validation=dict(restart=1,model='generalizes',info=dict(validation_loss=.4,train_objective=.3))
        self.assertEqual(check.choose_validation([low_training,low_validation])['model'],'generalizes')

    def test_exponential_additive_map_preserves_the_ranking_training_problem(self):
        cfg=self.cfg
        exp=check.base.make_design(cfg,check.base.PopulationDGP(cfg),1,16,'linear',
            check.case_scenario('exponential_additive',10.))
        np.testing.assert_array_equal(exp['train']['x'],self.design['train']['x'])
        observed=check.core.prepare(exp,self.features,cfg)
        np.testing.assert_array_equal(observed['y'].numpy(),self.data['y'].numpy())
        transformed_val=check.prepare_validation(exp,self.features,cfg)
        for key in self.val: self.assertTrue(torch.equal(transformed_val[key],self.val[key]))
        self.assertTrue(np.all(exp['train']['w']>0))
        self.assertFalse(np.allclose(exp['train']['w'],self.design['train']['w']))

    def test_additive_baseline_cannot_be_removed_by_log_cell_intercepts(self):
        # With the same b and R gap, changing a changes the within-cell log gap.
        r=np.array([0.,1.])
        gap_without_a=np.diff(np.log(np.exp(.6*r)))[0]
        gap_with_a=np.diff(np.log(10.+np.exp(.6*r)))[0]
        self.assertGreater(abs(gap_without_a-gap_with_a),.1)

    def test_log_fe_reference_uses_fitting_observed_labels_only(self):
        exp=check.base.make_design(self.cfg,check.base.PopulationDGP(self.cfg),1,16,'linear',
            check.case_scenario('exponential_additive',10.))
        original=check.log_fe_prediction(exp)
        altered=copy.deepcopy(exp);altered['train']['f'][:]=1e9;altered['train']['r'][:]=-1e9
        altered['val']['w'][:]=1e9
        for test,_ in altered['tests'].values():test['f'][:]=1e9;test['w'][:]=-1e9
        other=check.log_fe_prediction(altered)
        np.testing.assert_array_equal(original(exp['val']['x']),other(exp['val']['x']))

    def test_quadratic_run_changes_the_latent_signal(self):
        quad=check.base.make_design(self.cfg,check.base.PopulationDGP(self.cfg),1,16,'quadratic',check.case_scenario())
        self.assertFalse(np.allclose(quad['train']['f'],self.design['train']['f']))
        np.testing.assert_array_equal(quad['train']['x'],self.design['train']['x'])


if __name__=='__main__': unittest.main()
