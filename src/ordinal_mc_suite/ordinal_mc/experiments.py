"""Paper experiment grids with distinct temporal and cross-group weights."""
import itertools

def scenario(**kw):
    s=dict(error='gumbel',error_severity=0.,psi_shape='tanh',psi_delta=0.,psi_stage='shape',psi_strength=1.,
      positive_labels=False,composition=0.,link_case='exact',link_severity=0.,acquisition_strength=1.,group_strength=1.,change_strength=.4)
    s.update(kw); return s

def weights(d,arm,tuned=False):
    ts=d['lambda_grid'] if tuned else [d['lambda_t']]
    cs=d['lambda_grid'] if tuned else [d['lambda_c']]
    return list(itertools.product(ts if arm in ('temporal','both') else [0.],cs if arm in ('cross_city','both') else [0.]))

def estimator_specifications(d,exp):
    e=d[f'experiment{exp}']
    base=dict(acquisition_strength=d['acquisition_strength'],group_strength=d['group_strength'])
    if exp==1:
        for shape,(stage,v),composition in itertools.product(e['psi_shapes'],e['cases'],e['composition_grid']):
            s=scenario(**base,psi_shape=shape,psi_stage=stage,psi_strength=v,psi_delta=v if stage=='shape' else 0.,positive_labels=True,composition=composition)
            for method in e['methods']:
                loss={'rank_common':'common_logit','rank_heterogeneous':'heterogeneous_logit','rank_ordinal':'ordinal_smooth'}.get(method,'squared')
                if loss!='squared' and loss not in e['losses']: continue
                yield s,method,loss,'none',[(0.,0.)],'rank_common'
    elif exp==2:
        for (family,v),arm,loss in itertools.product(e['cases'],e['arms'],e['losses']):
            yield scenario(**base,error=family,error_severity=v),loss+'__'+arm,loss,arm,weights(d,arm),'common_logit__'+arm
    elif exp==3:
        for (a,c),loss,arm in itertools.product(e['nuisance_cases'],e['losses'],e['arms']):
            yield scenario(acquisition_strength=a,group_strength=c),arm,loss,arm,weights(d,arm),'none'
    elif exp==4:
        for (kind,v),loss in itertools.product(e['cases'],e['losses']):
            s=scenario(**base,link_case=kind,link_severity=v)
            yield s,'none',loss,'none',[(0.,0.)],'none'
            for arm in e['arms']:
                yield s,arm+'_fixed',loss,arm,weights(d,arm),'none'
                yield s,arm+'_tuned',loss,arm,weights(d,arm,True),'none'
    elif exp==5:
        for strength,contamination,loss,arm in itertools.product(e['change_strength_grid'],e['contamination_grid'],e['losses'],e['arms']):
            s=scenario(**base,change_strength=strength,link_case='changed',link_severity=contamination)
            yield s,arm,loss,arm,weights(d,arm),'none'


def specifications(d,exp):
    """Add a parametric ML reference to every scenario and comparison-loss panel."""
    benchmarks={}
    for spec in estimator_specifications(d,exp):
        yield spec
        s,method,loss,arm,ww,baseline=spec
        ref_loss=loss if exp in (3,4,5) else 'common_logit'
        key=(tuple(sorted(s.items())),ref_loss)
        benchmarks.setdefault(key,(s,'ml_correct',ref_loss,'none',[(0.,0.)],baseline))
    yield from benchmarks.values()

def jobs(d,experiments):
    for rep,n,truth,model,exp in itertools.product(range(1,d['reps']+1),d['n'],d['truths'],d['models'],experiments):
        # Q1/Q3: vary labeled groups. Q2/Q4: keep group count fixed at cities.
        city_grid=d['city_grid'] if exp in (1,3) else [d['cities']]
        for cities in city_grid:
            for s,method,loss,arm,ww,baseline in specifications(d,exp):
                yield dict(experiment=exp,rep=rep,n=n,cities=cities,truth=truth,model=model,scenario=s,method=method,
                    loss=loss,arm=arm,weights=sorted(set(ww)),baseline=baseline)
