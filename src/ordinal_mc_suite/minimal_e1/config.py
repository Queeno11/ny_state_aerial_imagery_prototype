"""Default configuration for the minimal loss-ablation experiment."""
import copy

DEFAULT = dict(
    seed=20261116,
    # Covariates: H (structural, 2-d, skewed) and A (nuisance, 4-d), mixed by a
    # fixed orthogonal matrix. Estimators see only the whitened mix z.
    p_h=2,
    p_a=4,
    signal=[1.0, 0.7],            # raw f = H1 + 0.7 H2 (then standardized)
    city_sd=0.5,                  # d_c ~ N(0, 0.25 I)
    change_sd=0.5,                # Delta H = 0.5 z for changing tracts
    n_draw=250,                   # every panel is a prefix of a 250-tract draw
    # Residual eta = sqrt(rho) u_i + sqrt(1-rho) eps_it, Gaussian, sigma = 1.
    rhos=[0.0, 0.8],
    # Maps: (a) alpha_ct + r; (b) alpha_ct + beta_ct r;
    # (c) alpha_ct + beta_ct S_lambda_c(r), asymmetric sinh with known kappa.
    maps=['a', 'b', 'c'],
    logbeta_sd=0.3,
    kappa=0.5,
    lambda_range=0.9,             # lambda_c ~ U(-0.9, 0.9) by city
    ml_lambda_bound=0.95,         # |lambda| <= 0.95 inside ML
    # Sample axis.
    small_sizes=[[8, 10], [8, 40], [32, 10], [32, 40]],
    anchor_size=[64, 250],
    reps=5,
    anchor_reps=5,
    # Fixed target environment (rep-independent) and per-rep calibration draws.
    target_cities=8,
    test_n=1000,
    cal_n=20,
    reference_size=200000,
    # Joint objective: (1-w) C_q/C_q(0) + w P_RN/log 2.
    joint_w=0.5,
    # Optimisation.
    sphere_gtol=1e-9,
    joint_gtol=1e-10,
    ml_gtol=1e-8,
    max_iter=3000,
    newton_tol=1e-11,
    mm_spec='unstructured',       # random-slope covariance; falls back to diagonal
)

STAGES = {
    '1': dict(sizes='small', maps=['a']),
    '2': dict(sizes='small', maps=['b']),
    '3': dict(sizes='small', maps=['c']),
    '4': dict(sizes='anchor', maps=['a', 'b', 'c']),
}


def make_config(**overrides):
    cfg = copy.deepcopy(DEFAULT)
    for key, value in overrides.items():
        if key not in cfg:
            raise KeyError(f'Unknown config key {key}')
        cfg[key] = value
    return cfg


def arms_for(map_name, rho):
    """Estimator arms for one map. Change-weighted arms only exist when rho>0,
    because q* = (1+rho)/2 = 0.5 makes them identical to the level arms at rho=0."""
    arms = ['TRUE', 'FE', 'L2', 'L2_S', 'ORD', 'ORD_S', 'ORD_L2', 'ORD_L2_S', 'ML', 'ML_S']
    if rho > 0:
        arms += ['L2D', 'L2D_S', 'ORD_L2D', 'ORD_L2D_S']
    if map_name == 'b':
        arms.append('MM')
    if map_name == 'c':
        arms.append('L2aff')
    return arms
