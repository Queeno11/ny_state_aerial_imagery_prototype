"""Predeclared E6 design; separate from existing E1--E5 configurations."""
import copy
import json
import math
from pathlib import Path
from ..config import preset, runtime

DEFAULT = {
    'seed': 20261103, 'function_seed': 983, 'replications': 30,
    'cities': [32, 64], 'cases': [['linear', 'linear'], ['quadratic', 'nn']],
    'labels_per_city': 32, 'validation_fraction': .2, 'views': 2,
    'noise_sd': .7, 'bayes_target': .2, 'hidden': [32, 16], 'weight_bound': 10.,
    'acquisition_strength': 1., 'group_strength': 1., 'alpha': 0., 'lambda_t': .3,
    'test_per_city': 128, 'test_pairs': 3000, 'change_strengths': [0., .15, .4, .8],
    'changed_share': .5, 'truth_threshold': .05, 'false_positive_target': .05,
    'actual_size_edges': [.05, .15, .3, .6], 'quadrature_nodes': 128,
    'maps': ['affine', 'nonlinear'], 'heads': ['affine', 'spline'],
    'level_weights': [.1, 1., 10.], 'validation_rank_weight': 1., 'validation_level_weight': 1.,
    'kappa_base': .6, 'kappa_amplitude': .2, 'kappa_time_shift': .5,
    'restarts': 3, 'threads': 1, 'workers': 4,
    'baseline': {'adam_steps': 3000, 'lr': .01, 'check_every': 10, 'patience': 20,
                 'lbfgs_steps': 400, 'lbfgs_chunk': 10, 'lbfgs_patience': 10},
    'head_training': {'steps': 600, 'lr': .02, 'check_every': 10, 'patience': 20,
                      'knots': [.2, .5, .8], 'kappa_G': .01, 'pool_relative': 10.},
    'joint': {'steps': 600, 'scorer_lr': .001, 'head_lr': .01, 'check_every': 10, 'patience': 20},
    'oracle_R': {'enabled': True, 'steps': 600, 'lr': .001},
    'min_delta': 1e-5,
}


def load(path):
    d = copy.deepcopy(DEFAULT)
    def merge(base, extra):
        for k, v in extra.items():
            if k not in base: raise ValueError('Unknown E6 configuration key: '+k)
            if isinstance(base[k], dict): merge(base[k], v)
            else: base[k] = v
    if path: merge(d, json.loads(Path(path).read_text(encoding='utf-8')))
    validate(d)
    return d


def validate(d):
    def finite(v):
        if isinstance(v, dict):
            for x in v.values(): finite(x)
        elif isinstance(v, list):
            for x in v: finite(x)
        elif isinstance(v, float) and not math.isfinite(v): raise ValueError('Nonfinite configuration value')
    finite(d)
    for k in ('replications', 'labels_per_city', 'views', 'restarts', 'threads', 'workers', 'test_per_city', 'test_pairs', 'quadrature_nodes'):
        if not isinstance(d[k], int) or d[k] < 1: raise ValueError(k+' must be a positive integer')
    if d['labels_per_city'] < 8 or d['test_per_city'] < 4: raise ValueError('Insufficient tract counts')
    for key in ('cities', 'cases', 'maps', 'heads', 'level_weights', 'change_strengths'):
        values=d[key]
        if not values or len({json.dumps(x) for x in values}) != len(values): raise ValueError('Empty/duplicate '+key)
    if any(not isinstance(c, int) or c < 2 for c in d['cities']): raise ValueError('cities must be >=2')
    if any(pair not in [['linear', 'linear'], ['quadratic', 'nn']] for pair in d['cases']): raise ValueError('Unsupported truth/scorer case')
    if not set(d['maps']) <= {'affine', 'nonlinear'} or not set(d['heads']) <= {'affine', 'spline'}: raise ValueError('Invalid map/head')
    if min(d['level_weights']) <= 0 or min(d['change_strengths']) < 0: raise ValueError('Invalid loss/change weights')
    if not 0 < d['validation_fraction'] < 1 or not 0 < d['changed_share'] < 1: raise ValueError('Invalid fractions')
    if not 0 < d['false_positive_target'] < 1 or d['truth_threshold'] < 0: raise ValueError('Invalid detection settings')
    if not d['kappa_base'] > d['kappa_amplitude'] > 0: raise ValueError('Kappa must have bounded positive support')
    if sorted(set(d['actual_size_edges'])) != d['actual_size_edges'] or d['actual_size_edges'][0] != d['truth_threshold']: raise ValueError('Size edges must start at truth_threshold and increase')
    if sorted(set(d['head_training']['knots'])) != d['head_training']['knots'] or any(not 0 < p < 1 for p in d['head_training']['knots']): raise ValueError('Invalid knot quantiles')
    for part, counts in [('baseline', ('adam_steps', 'check_every', 'patience', 'lbfgs_chunk', 'lbfgs_patience')), ('head_training', ('steps', 'check_every', 'patience')), ('joint', ('steps', 'check_every', 'patience'))]:
        for key in counts:
            if not isinstance(d[part][key], int) or d[part][key] < 1: raise ValueError(part+'.'+key+' must be positive')
    if not isinstance(d['baseline']['lbfgs_steps'],int) or d['baseline']['lbfgs_steps'] < 0: raise ValueError('Invalid L-BFGS budget')
    for v in (d['alpha'], d['lambda_t'], d['head_training']['kappa_G'], d['head_training']['pool_relative']):
        if v < 0: raise ValueError('Negative penalty')
    if min(d['validation_rank_weight'], d['validation_level_weight']) <= 0: raise ValueError('Fixed validation weights must be positive')
    for part,key in [('baseline','lr'),('head_training','lr'),('joint','scorer_lr'),('joint','head_lr'),('oracle_R','lr')]:
        if d[part][key]<=0: raise ValueError('Nonpositive learning rate')
    if not isinstance(d['oracle_R']['steps'],int) or d['oracle_R']['steps']<1: raise ValueError('Invalid oracle budget')
    if not isinstance(d['oracle_R']['enabled'],bool): raise ValueError('oracle_R.enabled must be boolean')
    if d['min_delta']<0 or d['noise_sd']<0: raise ValueError('Invalid stopping/noise settings')
    if d['bayes_target'] is not None and not 0<d['bayes_target']<.5: raise ValueError('Invalid Bayes target')
    if not d['hidden'] or any(not isinstance(x,int) or x<1 for x in d['hidden']): raise ValueError('Invalid hidden architecture')
    if d['weight_bound']<=0 or min(d['acquisition_strength'],d['group_strength'])<0: raise ValueError('Invalid feature settings')


def e5_runtime(d, cities):
    old=preset('smoke')
    for key in ('seed', 'function_seed', 'validation_fraction', 'views', 'noise_sd', 'bayes_target',
                'hidden', 'weight_bound', 'acquisition_strength', 'group_strength', 'test_per_city', 'test_pairs', 'threads'):
        old[key]=d[key]
    old.update(cities=cities, city_grid=[cities], n=[d['labels_per_city']], device='cpu', dtype='float64',
               pair_mode='all', workers=1)
    old['experiment5'].update(changed_share=d['changed_share'], truth_threshold=d['truth_threshold'],
                              false_positive_target=d['false_positive_target'])
    cfg=runtime(old); cfg.fit_cell_effects=False
    return cfg
