"""Numerical tests for the single-sample E5 optimizer investigation."""
import copy
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]/'ordinal_mc_suite'
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('e5_optimization_diagnostic',ROOT/'verification/diagnose_e5_optimization.py')
diag=importlib.util.module_from_spec(spec); spec.loader.exec_module(diag)
from ordinal_mc import core
from ordinal_mc.config import preset,runtime
from ordinal_mc.design import PopulationDGP,make_design
from ordinal_mc.experiments import scenario


@pytest.fixture(scope='module')
def example():
    d=preset('smoke'); d.update(cities=2,views=1,epochs=8,lbfgs_steps=8,audit_steps=3,
        test_pairs=64,test_per_city=12,hidden=[32,16])
    cfg=runtime(d); cfg.fit_cell_effects=False
    design=make_design(cfg,PopulationDGP(cfg),1,16,'linear',scenario())
    features=core.Features(design['train']['x'],'nn',cfg); data=core.prepare(design,features,cfg)
    edges={arm:core.tensor(design['links'][arm],cfg,True) for arm in ('temporal','cross_city')}
    torch.manual_seed(1729)
    model=core.Estimator(data['x'].shape[1],'nn','common_logit',cfg).double()
    return cfg,design,features,data,edges,model


def test_nn_loss_and_parameter_gradients_against_independent_calculations(example):
    cfg,_,_,data,edges,model=example
    report=diag.gradient_checks(copy.deepcopy(model),data,edges,(.3,.3))
    assert len(report['directional_checks'])==5
    assert max(r['scaled_error'] for r in report['directional_checks'])<2e-7
    assert report['penalty_gradient_max_error']<1e-12


def test_denominator_derivative_is_material(example):
    _,_,_,data,edges,model=example
    s=model.scorer(data['x']).detach().clone().requires_grad_(True)
    gap=s[data['j']]-s[data['i']]; variance=.5*(data['weights']*gap.square()).sum()
    e=edges['temporal']; raw=(s[e[:,0]]-s[e[:,1]]).square().mean()
    correct=torch.autograd.grad(raw/variance,s,retain_graph=True)[0]
    wrong=torch.autograd.grad(raw/variance.detach(),s)[0]
    expected=diag.explicit_penalty_gradient(s.detach().numpy(),data,edges,(1.,0.))
    np.testing.assert_allclose(correct.numpy(),expected,rtol=1e-11,atol=1e-12)
    assert np.max(abs(wrong.numpy()-expected))>1e-5


def test_linear_function_and_criterion_embed_exactly_in_nn(example):
    cfg,design,features,data,edges,_=example
    torch.manual_seed(42)
    linear=core.Estimator(data['x'].shape[1],'linear','common_logit',cfg).double()
    nn=diag.embed_linear(linear,cfg)
    for x in (design['x'],design['val']['x'],design['tests']['unseen'][0]['x']):
        np.testing.assert_allclose(core.predictions(linear,features,x,cfg),core.predictions(nn,features,x,cfg),atol=2e-12,rtol=1e-12)
    for lam in ((0.,0.),(.3,0.),(.3,.3)):
        a=core.score_objective(linear,data,edges,lam,0.,.1)[0]
        b=core.score_objective(nn,data,edges,lam,0.,.1)[0]
        np.testing.assert_allclose(float(a.detach()),float(b.detach()),atol=1e-12)
    assert all(torch.all(nn.scorer.effective(layer.weight).abs()<cfg.weight_bound) for layer in nn.scorer.layers)


def test_penalty_scale_invariance_for_neural_scorer(example):
    cfg,_,_,data,edges,model=example
    original=copy.deepcopy(model); scaled=copy.deepcopy(model)
    w=2*scaled.scorer.effective(scaled.scorer.layers[-1].weight).detach()
    diag.set_effective(scaled.scorer.layers[-1],w,None,cfg.weight_bound)
    _,a=core.score_objective(original,data,edges,(.3,.3),0.,.1)
    _,b=core.score_objective(scaled,data,edges,(.3,.3),0.,.1)
    np.testing.assert_allclose(float(a[1].detach()),float(b[1].detach()),rtol=1e-12)
    np.testing.assert_allclose(4*float(a[3].detach()),float(b[3].detach()),rtol=1e-12)
    assert abs(float(a[0].detach())-float(b[0].detach()))>1e-5


def test_seeded_optimizer_is_deterministic(example):
    cfg,_,_,data,edges,_=example
    first,a=core.train_candidate(data,edges,'nn','common_logit',0.,(.3,.3),771,cfg)
    second,b=core.train_candidate(data,edges,'nn','common_logit',0.,(.3,.3),771,cfg)
    for key,value in first.state_dict().items(): assert torch.equal(value,second.state_dict()[key])
    assert a['train_objective']==b['train_objective']


def test_continuation_preserves_admissibility_and_never_accepts_worse_endpoint(example):
    cfg,_,_,data,edges,model=example
    model=copy.deepcopy(model)
    initial=float(core.score_objective(model,data,edges,(.3,.3),0.,.1)[0].detach()); values=[]
    def record(model,chunk,info):
        q,parts=core.score_objective(model,data,edges,(.3,.3),0.,.1)
        assert torch.isfinite(q) and parts[3]>0
        values.append(float(q.detach()))
    opt=diag.continue_lbfgs(model,data,edges,(.3,.3),cfg,3,10,record)
    assert len(values)>=1
    assert np.all(np.diff([initial]+values)<=1e-12)
    assert any(state.get('n_iter',0)>0 for state in opt.state.values())


def test_detection_failure_is_not_caused_by_arbitrary_global_score_units(example):
    from ordinal_mc.change import evaluate_changes
    cfg,design,features,_,_,model=example
    predict=lambda x:core.predictions(model,features,x,cfg)
    before=evaluate_changes(predict,design,cfg,1.)
    after=evaluate_changes(lambda x:100*predict(x)+50,design,cfg,1.)
    for where in ('seen','unseen'):
        for metric in ('change_recall','change_direction_accuracy','change_false_positive_rate','change_auc'):
            assert before[where+'_'+metric]==after[where+'_'+metric]
