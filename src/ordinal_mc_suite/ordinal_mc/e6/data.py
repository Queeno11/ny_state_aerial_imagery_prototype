"""E5 samples with paired affine/nonlinear measurements and outcome test panels."""
import copy
import numpy as np
from scipy.special import roots_legendre
from ..core import seed_for, split_labels
from ..design import make_design, matched_panel, coefficients
from ..experiments import scenario


def cell_parameters(contexts, c, t, d):
    a,b,_=coefficients(contexts,c,t)
    k=d['kappa_base']+d['kappa_amplitude']*np.tanh(contexts[c,0]+d['kappa_time_shift']*t)
    return a,b,k


def measurement(r, a, b, k, kind):
    if kind not in ('affine', 'nonlinear'): raise ValueError('Unknown measurement map')
    return a+b*(r+(k*np.tanh(r) if kind=='nonlinear' else 0.))


def conditional_mean(f, a, b, k, kind, noise_sd, nodes=128):
    """E[W|H,c,t], integrating noise; this is generally not Psi(f)."""
    f=np.asarray(f)
    if kind=='affine': return a+b*f  # exact because errors are centered
    if kind!='nonlinear': raise ValueError('Unknown map')
    q,w=roots_legendre(nodes); u=(q+1)/2; w=w/2
    errors=noise_sd*np.sqrt(6)/np.pi*(-np.log(-np.log(u))-np.euler_gamma)
    expectation=np.empty_like(f,dtype=float)
    for start in range(0,len(f),2048):
        expectation[start:start+2048]=np.tanh(f[start:start+2048,None]+errors)@w
    return a+b*(f+k*expectation)


def contexts_for(design, cfg, where):
    if where=='seen': return design['contexts']
    return np.random.default_rng(seed_for(design['seed'],'unseen')).normal(size=(max(4,cfg.cities//2),2))


def build(cfg,dgp,rep,truth,d):
    s=scenario(link_case='exact',change_strength=.4,
               acquisition_strength=cfg.acquisition_strength,group_strength=cfg.group_strength)
    base=make_design(cfg,dgp,rep,d['labels_per_city'],truth,s)
    # E5 constructs a cross-city pool too; E6 offers only the unchanged exact temporal pool.
    base['x']=base['x'][:base['n_fit']+base['budget']]
    base['links']['cross_city']=base['links']['none'].copy()
    base['links']['both']=base['links']['temporal'].copy()
    base['link_stats']['cross_city']=base['link_stats']['none'].copy()
    base['link_stats']['both']=base['link_stats']['temporal'].copy()
    # Recover original labeled-tract IDs using exactly E5's split. Each labeled
    # tract has one observed date; every additional date/view is unlabeled and
    # generated only from a fitting tract. IDs exist before the split.
    n=d['labels_per_city'];original_cells=2*np.repeat(np.arange(cfg.cities),n)+np.tile(np.arange(n)%2,cfg.cities)
    split=split_labels(dict(cell=original_cells,tract_id=np.arange(len(original_cells))),cfg.validation_fraction,
                       np.random.default_rng(seed_for(base['seed'],'split')))
    for name,part in zip(('train','val'),split):
        np.testing.assert_array_equal(base[name]['cell'],part['cell'])
        base[name]['tract_id']=part['tract_id']
    base['auxiliary_tract_ids']=np.repeat(base['train']['tract_id'],cfg.views)
    assert not set(base['auxiliary_tract_ids']) & set(base['val']['tract_id'])
    assert not set(map(tuple,base['train']['h'])) & set(map(tuple,base['val']['h']))
    assert base['link_stats']['temporal']['link_gap_rms'] < 1e-12
    panels={}
    for strength in d['change_strengths']:
        panels[strength]={}
        for where in ('seen','unseen'):
            contexts=contexts_for(base,cfg,where)
            p=matched_panel(seed_for(base['seed'],'matched',where),contexts,cfg,dgp,truth,strength,d['changed_share'])
            h0=(p['x_before']@dgp.mix.T)[:,:3]; h1=(p['x_after']@dgp.mix.T)[:,:3]
            f0=dgp.score(h0,truth); f1=dgp.score(h1,truth)
            np.testing.assert_allclose(f1-f0,p['delta_f'],atol=2e-13,rtol=1e-12)
            c=np.repeat(np.arange(len(contexts)),cfg.test_per_city); t0=np.zeros(len(c),int);t1=np.ones(len(c),int)
            noise=[]
            for endpoint in ('before','after'):
                rng=np.random.default_rng(seed_for(base['seed'],'e6_outcome',where,endpoint))
                noise.append(base['noise_sd_effective']*(rng.gumbel(size=len(c))-np.euler_gamma)*np.sqrt(6)/np.pi)
            p.update(c=c,cell_before=2*c,cell_after=2*c+1,f_before=f0,f_after=f1,
                     r_before=f0+noise[0],r_after=f1+noise[1],contexts=contexts,t_before=t0,t_after=t1)
            panels[strength][where]=p
    base['e6_panels']=panels
    variants={}
    for kind in d['maps']:
        v=copy.copy(base)
        def remap(data,contexts):
            out=data.copy();a,b,k=cell_parameters(contexts,data['c'],data['t'],d)
            out.update(a=a,b=b,kappa=k,w=measurement(data['r'],a,b,k,kind))
            return out
        v['train']=remap(base['train'],base['contexts']);v['val']=remap(base['val'],base['contexts'])
        v['tests']={where:(remap(data,contexts_for(base,cfg,where)),pairs) for where,(data,pairs) in base['tests'].items()}
        v['e6_panels']={}
        for strength,groups in panels.items():
            v['e6_panels'][strength]={}
            for where,p in groups.items():
                out=p.copy()
                for endpoint in ('before','after'):
                    a,b,k=cell_parameters(p['contexts'],p['c'],p['t_'+endpoint],d)
                    out['w_'+endpoint]=measurement(p['r_'+endpoint],a,b,k,kind)
                    out['mu_'+endpoint]=conditional_mean(p['f_'+endpoint],a,b,k,kind,base['noise_sd_effective'],d['quadrature_nodes'])
                v['e6_panels'][strength][where]=out
        v['measurement_scenario']={**s,'psi_delta':1. if kind=='nonlinear' else 0.,'psi_shape':'tanh'}
        variants[kind]=v
    if len(variants)>1:
        i,j=base['pairs']; reference=None
        for v in variants.values():
            y=np.sign(v['train']['w'][j]-v['train']['w'][i])
            if reference is not None: np.testing.assert_array_equal(y,reference)
            reference=y
    return base,variants


def coefficient_ranges(d):
    values={name:[] for name in ('a','b','kappa')}
    unseen={name:[] for name in values}
    for cities in d['cities']:
        for rep in range(1,d['replications']+1):
            seed=seed_for(d['seed'],'data',rep,d['labels_per_city'],cities)
            contexts=np.random.default_rng(seed).normal(size=(cities,2))
            c=np.repeat(np.arange(cities),2);t=np.tile([0,1],cities)
            for name,x in zip(values,cell_parameters(contexts,c,t,d)): values[name].extend(x.tolist())
            cx=np.random.default_rng(seed_for(seed,'unseen')).normal(size=(max(4,cities//2),2))
            cc=np.repeat(np.arange(len(cx)),2);tt=np.tile([0,1],len(cx))
            for name,x in zip(unseen,cell_parameters(cx,cc,tt,d)): unseen[name].extend(x.tolist())
    return dict(formulas={'a':'3*Z[c,0] + t','b':'1.5 + exp(0.2*Z[c,1])',
        'kappa':'kappa_base + kappa_amplitude*tanh(Z[c,0] + kappa_time_shift*t)'},
        theoretical={'a':'unbounded; finite Gaussian moments','b':'(1.5, infinity); finite lognormal moments',
        'kappa':[d['kappa_base']-d['kappa_amplitude'],d['kappa_base']+d['kappa_amplitude']]},
        realized={name:{'min':float(np.min(x)),'max':float(np.max(x))} for name,x in values.items()},
        realized_unseen={name:{'min':float(np.min(x)),'max':float(np.max(x))} for name,x in unseen.items()},
        monotonicity='Psi_prime=b*(1+kappa*sech(R)^2)>0; affine derivative=b>0',
        finite_second_moments='Gumbel R has finite second moment; tanh is bounded; Gaussian/lognormal cell coefficients have finite moments')
