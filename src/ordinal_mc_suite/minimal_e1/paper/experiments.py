"""Experiment registry E1-E5: scenarios (DGP Spec), sizes, repetitions, arms."""
from dataclasses import replace

from .dgp import Spec

E1_SIZES = [(8, 50, 100), (8, 150, 100), (8, 500, 100), (64, 500, 50), (128, 500, 30)]
E_SIZES = [(8, 150, 100), (8, 500, 100), (64, 500, 30)]
MRC_MAX_C = 64
CURVE_MAX_C = 8             # omega-curve / selector arms only up to 8 cities (compute)

E1_ARMS = ['RNC', 'RNC_S', 'RNC_P0.9', 'SUR', 'SUR_S', 'MRC_S', 'FE', 'FE_S', 'NS', 'MM', 'ML']
E2_ARMS = ['RNC', 'RNC_S', 'RNC_P0.9', 'SUR_S', 'MRC_S', 'FE', 'FE_S']
E2_CURVE = ['RNC_P0.5', 'RNC_P0.99', 'RNC_SEL']
E3_ARMS = [a for a in E1_ARMS if a != 'ML'] + ['RNC_S_DW']
E4_ARMS = [a for a in E1_ARMS if a != 'ML']
E5_ARMS = ['RNC_S', 'RNC', 'SUR_S', 'FE']

BASE = Spec()
CORRUPT_TOL = 'gap'         # hard null space for fake/approximate-link cells: largest eigen-gap rule
                            # (a fixed tolerance 0.3 dropped true null directions at 20% fakes)

SCENARIOS = {
    'E1': {
        'E1_a': Spec(map='a'), 'E1_b': Spec(map='b'), 'E1_c': Spec(map='c'),
        'E1_cD': Spec(map='c', noise='D'),
    },
    'E2': {
        'L_none': replace(BASE, links='none'), 'L_temp': replace(BASE, links='temporal'),
        'L_cross': replace(BASE, links='cross'), 'L_both': BASE,
        'N1_nuis_temp': replace(BASE, city_nuis=True, links='temporal'),
        'N1_nuis_both': replace(BASE, city_nuis=True, links='both'),
        'N2_hc_temp': replace(BASE, hc=True, links='temporal'),
        'N2_hc_both': replace(BASE, hc=True, links='both'),
        'F_5': replace(BASE, fake=0.05, link_tol=CORRUPT_TOL), 'F_20': replace(BASE, fake=0.20, link_tol=CORRUPT_TOL),
        'S_25': replace(BASE, declared=0.25), 'S_5': replace(BASE, declared=0.05),
        'A_02': replace(BASE, approx=0.02, link_tol=CORRUPT_TOL), 'A_10': replace(BASE, approx=0.10, link_tol=CORRUPT_TOL),
    },
    'E3': {
        'D_rho08': replace(BASE, rho=0.8),
        'D_sp005_ind': replace(BASE, spatial_r=0.05), 'D_sp03_ind': replace(BASE, spatial_r=0.3),
        'D_sp005_sh': replace(BASE, spatial_r=0.05, spatial_shared=True),
        'D_sp03_sh': replace(BASE, spatial_r=0.3, spatial_shared=True),
        'D_comb': replace(BASE, rho=0.8, spatial_r=0.3, spatial_shared=True),
    },
    'E4': {
        'X_i': replace(BASE, e4='i'), 'X_ii': replace(BASE, e4='ii'), 'X_iii': replace(BASE, e4='iii'),
    },
    # E5: baseline updating. Latent noise = persistent tract part (75%) + transitory part; '_x' versions move
    # half of the persistent part into the index (xi = c f + xi_tilde).
    'E5': {
        'P_const': replace(BASE, persist='const'), 'P_rdep': replace(BASE, persist='rdep'),
        'P_reset': replace(BASE, persist='reset'), 'P_ferel': replace(BASE, persist='fe_rel'),
        'P_const_x': replace(BASE, persist='const', xi_x=0.5), 'P_rdep_x': replace(BASE, persist='rdep', xi_x=0.5),
        'P_reset_x': replace(BASE, persist='reset', xi_x=0.5), 'P_ferel_x': replace(BASE, persist='fe_rel', xi_x=0.5),
    },
}
CURVE_SCENARIOS = {'F_5', 'F_20', 'A_02', 'A_10', 'S_25', 'S_5'}


def arms_for(exp, scenario, spec, C):
    if exp == 'E1':
        arms = list(E1_ARMS)
    elif exp == 'E2':
        arms = list(E2_ARMS)
        if scenario in CURVE_SCENARIOS and C <= CURVE_MAX_C:
            arms += E2_CURVE
    elif exp == 'E3':
        arms = list(E3_ARMS)
    elif exp == 'E5':
        arms = list(E5_ARMS)
    else:
        arms = list(E4_ARMS)
    if C > MRC_MAX_C:
        arms = [a for a in arms if a != 'MRC_S']
    if not spec.ml_correct():
        arms = [a for a in arms if a != 'ML']
    return arms


def sizes_for(exp):
    return E1_SIZES if exp == 'E1' else E_SIZES


def plan(exps=('E1', 'E2', 'E3', 'E4', 'E5'), scenarios=None, sizes=None, reps=None, smoke=False):
    """List of tasks (exp, scenario, spec, C, n, rep, arms)."""
    tasks = []
    for exp in exps:
        for name, spec in SCENARIOS[exp].items():
            if scenarios and name not in scenarios:
                continue
            ss = sizes_for(exp)
            if smoke:
                ss = [(8, 50, 2)] if exp == 'E1' else [(8, 150, 2)]
            for C, n, R in ss:
                if sizes and (C, n) not in sizes:
                    continue
                R = reps if reps is not None else R
                if smoke:
                    R = min(R, 2)
                for rep in range(R):
                    tasks.append(dict(exp=exp, scenario=name, C=C, n=n, rep=rep, spec=spec,
                                      arms=arms_for(exp, name, spec, C)))
    return tasks


def find_spec(scenario):
    for exp, d in SCENARIOS.items():
        if scenario in d:
            return exp, d[scenario]
    raise KeyError(scenario)
