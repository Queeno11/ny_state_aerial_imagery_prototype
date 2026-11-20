"""Numerical recovery, no test-truth leakage, matched panels and reporting."""
import copy
import json
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ordinal_mc_suite'))
from ordinal_mc.likelihood import error_nll, fit_coefficients, structural_basis
from ordinal_mc.change import change_metrics


@pytest.mark.parametrize('family,severity', [('gumbel',0),('normal',0),('student',5),('chi2',4),('mixture',1.2)])
def test_likelihood_matches_scipy(family,severity):
    z=np.array([-.2,.1,.7]); scale=np.array([.8,1.,1.3])
    if family=='gumbel':
        k=np.sqrt(6)/np.pi; ref=stats.gumbel_r.logpdf(z,loc=-np.euler_gamma*k,scale=k)
    elif family=='normal': ref=stats.norm.logpdf(z)
    elif family=='student': ref=stats.t.logpdf(z,severity,scale=np.sqrt((severity-2)/severity))
    elif family=='chi2': ref=stats.chi2.logpdf(z,severity,loc=-severity/np.sqrt(2*severity),scale=1/np.sqrt(2*severity))
    else:
        k=np.sqrt(1+severity**2)
        ref=np.log(.5*stats.norm.pdf(z,loc=-severity/k,scale=1/k)+.5*stats.norm.pdf(z,loc=severity/k,scale=1/k))
    np.testing.assert_allclose(error_nll(z*scale,scale,family,severity),-ref+np.log(scale))


def test_gaussian_ml_is_least_squares_and_gumbel_recovers_coefficients():
    rng=np.random.default_rng(14)
    x=np.column_stack((np.ones(12000),rng.normal(size=(12000,3))))
    true=np.array([.2,.7,-.4,1.1]); y=x@true+rng.normal(size=len(x))*.5
    b,info=fit_coefficients(x,y,np.full(len(x),.5),'normal',0)
    np.testing.assert_allclose(b,np.linalg.lstsq(x,y,rcond=None)[0],atol=1e-6)
    y=x@true+.5*(rng.gumbel(size=len(x))-np.euler_gamma)*np.sqrt(6)/np.pi
    b,info=fit_coefficients(x,y,np.full(len(x),.5),'gumbel',0)
    assert info['ml_success']
    np.testing.assert_allclose(b,true,atol=.025)


def test_noiseless_limit_and_unbounded_chi_square():
    x=np.column_stack((np.ones(8),np.arange(8)))
    beta,info=fit_coefficients(x,x@np.array([.5,2.]),np.zeros(8),'gumbel',0.)
    np.testing.assert_allclose(beta,[.5,2.],atol=1e-12)
    assert info['ml_success'] and info['ml_train_nll'] is None
    with pytest.raises(ValueError,match='unbounded likelihood'):
        fit_coefficients(x,np.arange(8),np.ones(8),'chi2',1.)


def test_change_metrics_perfect_and_collapsed():
    actual=np.array([0.,0.,.4,-.4])
    perfect=change_metrics(actual,actual,.1,.05)
    assert perfect['change_rmse']==0 and perfect['change_auc']==1
    assert perfect['change_recall']==perfect['change_direction_accuracy']==1
    assert perfect['change_false_positive_rate']==0
    flat=change_metrics(np.zeros(4),actual,.1,.05)
    assert flat['change_recall']==flat['change_slope']==0
    assert flat['change_direction_accuracy']==flat['change_auc']==.5
    assert flat['change_precision'] is None
    assert change_metrics(np.zeros(4),np.zeros(4),0,.05)['change_auc'] is None


@pytest.fixture
def setup_design():
    pytest.importorskip('torch')
    from ordinal_mc.config import preset,runtime
    from ordinal_mc.design import PopulationDGP,make_design
    from ordinal_mc.experiments import scenario
    d=preset('smoke'); d.update(cities=4,city_grid=[4],n=[16],models=['linear'],truths=['linear'],
        test_per_city=24,test_pairs=128,epochs=8,lbfgs_steps=10,audit_steps=2)
    cfg=runtime(d); dgp=PopulationDGP(cfg)
    s=scenario(positive_labels=True,psi_stage='shape',psi_delta=1.)
    return d,cfg,dgp,s,make_design(cfg,dgp,1,16,'linear',s)


def test_matched_panels_and_ml_never_fit_latent_or_test_labels(setup_design):
    from ordinal_mc.likelihood import fit_ml
    d,cfg,dgp,s,design=setup_design
    for panel in design['change_tests'].values():
        stable=~panel['assigned_changed']
        assert stable.any() and (~stable).any()
        np.testing.assert_array_equal(panel['delta_f'][stable],0)
        assert np.any(panel['delta_f']>0) and np.any(panel['delta_f']<0)
    np.testing.assert_array_equal(design['change_calibration']['delta_f'],0)
    predict,info=fit_ml(design,'linear',s,dgp)
    changed=copy.deepcopy(design)
    for key in ('f','r'): changed['train'][key][:]=1e9
    changed['val']['f'][:]=1e8
    for data,_ in changed['tests'].values(): data['f'][:]=1e7
    other,_=fit_ml(changed,'linear',s,dgp)
    np.testing.assert_array_equal(predict(design['val']['x']),other(design['val']['x']))


def test_change_cutoff_does_not_depend_on_test_data(setup_design):
    from ordinal_mc.change import evaluate_changes
    _,cfg,_,_,design=setup_design
    predict=lambda x:x[:,0]
    before=evaluate_changes(predict,design,cfg,1.)
    altered=copy.deepcopy(design)
    for panel in altered['change_tests'].values():
        panel['x_after'][:]=1e6; panel['delta_f'][:]=-1e6
    after=evaluate_changes(predict,altered,cfg,1.)
    for where in ('seen','unseen'):
        assert before[where+'_change_threshold']==after[where+'_change_threshold']


def test_all_experiments_include_unique_ml_jobs(setup_design):
    from ordinal_mc.experiments import jobs
    from ordinal_mc.config import validate
    d,*_=setup_design; validate(d)
    jj=list(jobs(d,[1,2,3,4,5]))
    for exp in range(1,6): assert any(j['method']=='ml_correct' and j['experiment']==exp for j in jj)
    assert len({json.dumps(j,sort_keys=True) for j in jj})==len(jj)


def test_runner_ml_and_change_reporting(setup_design,tmp_path):
    from ordinal_mc.runner import fit_job,canonical
    from ordinal_mc.report import export
    d,cfg,dgp,s,design=setup_design
    job=dict(experiment=5,rep=1,n=16,cities=4,truth='linear',model='linear',scenario=s,
        method='ml_correct',loss='common_logit',arm='none',weights=[(0.,0.)],baseline='none')
    result,candidates=fit_job(job,design,cfg,dgp)
    assert not candidates and result['ml_parameters']==4
    assert 'unseen_change_rmse' in result and 'unseen_within_accuracy' in result
    conn=sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE jobs (result TEXT)'); conn.execute('CREATE TABLE candidates (data TEXT)')
    baseline={**result,'method':'none','unseen_change_recall':.1}
    result['unseen_change_recall']=.8
    for r in (baseline,result): conn.execute('INSERT INTO jobs VALUES (?)',(canonical(r),))
    export(conn,tmp_path,plots=True)
    import csv
    paired=list(csv.DictReader((tmp_path/'paired_replications.csv').open()))
    r=next(r for r in paired if r['metric']=='unseen_change_recall')
    assert float(r['gain'])==pytest.approx(.7)
    assert list((tmp_path/'figures').glob('exp5_*.png'))


def test_changed_link_contamination_does_not_change_labeled_data(setup_design):
    from ordinal_mc.design import make_design
    from ordinal_mc.experiments import scenario
    d,cfg,dgp,_,_=setup_design
    a=make_design(cfg,dgp,1,16,'linear',scenario(link_case='changed',link_severity=0.))
    b=make_design(cfg,dgp,1,16,'linear',scenario(link_case='changed',link_severity=1.))
    np.testing.assert_array_equal(a['train']['x'],b['train']['x'])
    np.testing.assert_array_equal(a['train']['w'],b['train']['w'])
    assert a['link_stats']['temporal']['link_gap_rms']==0
    assert b['link_stats']['temporal']['link_gap_rms']>0
    for name in a['change_tests']:
        np.testing.assert_array_equal(a['change_tests'][name]['x_after'],b['change_tests'][name]['x_after'])


@pytest.mark.parametrize('truth',['linear','quadratic','smooth','regime'])
def test_ml_basis_contains_the_true_structural_function(setup_design,truth):
    _,_,dgp,_,_=setup_design
    rng=np.random.default_rng(21); h=rng.normal(size=(300,3))
    x=structural_basis(h,truth,dgp.rotation); y=dgp.score(h,truth)
    b=np.linalg.lstsq(x,y,rcond=None)[0]
    np.testing.assert_allclose(x@b,y,atol=1e-12)


def test_rho_simulation_full_ml_benchmark(tmp_path):
    pytest.importorskip('torch')
    import importlib.util
    from types import SimpleNamespace
    path=Path(__file__).resolve().parents[1]/'rho_city_monte_carlo.py'
    spec=importlib.util.spec_from_file_location('rho_mc',path)
    rho=importlib.util.module_from_spec(spec); spec.loader.exec_module(rho)
    import torch
    a=SimpleNamespace(seed=12,cities=5,years=1,reference_size=128,val_per_cell=16,test_per_cell=32,
        eval_pairs_per_cell=32,width=4,lr=.001,rho_lr=.01,weighting='city',steps=1,units_per_group=4,
        groups_per_step=2,shrinkage=0.,eval_every=1)
    design=rho.make_design(a,0,.4)
    metrics,cities,histories=rho.fit_one(a,design,'ml_correct',None,0,.4,torch.device('cpu'),tmp_path)
    assert metrics['ml_success'] and metrics['f_r2']>.95
    assert len(cities)==5 and len(histories)==1
    assert (tmp_path/'ml_correct_checkpoint.pt').exists()


def test_additive_exact_includes_ml_summary():
    pytest.importorskip('torch')
    import e1_additive_exact as ex
    import pandas as pd
    records=pd.DataFrame([dict(cities=4,rep=r,scale=s,risk=.2,ml_risk=.03)
                          for r in (1,2) for s in (0.,10.)])
    fits=pd.DataFrame([dict(cities=4,rep=r,method='rank_common',unseen_within_latent=.1) for r in (1,2)])
    summary=ex.summarize(records,fits)
    ml=summary[summary.method=='ml_correct']
    assert len(ml)==2 and np.all(ml['mean']==.03)
