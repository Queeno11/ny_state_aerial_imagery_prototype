"""New paired residual law on exactly the original feature and city draws."""
from dataclasses import dataclass, replace

import numpy as np

from prior import SIMPLE
from data import Cities, Panel, draw_cities, make_panel, seed_for, training as old_training
from calibrate_10pct import CALIBRATION_TRACTS, PANEL_NAMESPACE, PANEL_TRACTS, array_sha, city_split


NOISE_NAMESPACE = "simple-index-tract-persistent-halfvariance-v1"


@dataclass
class PersistentPanel(Panel):
    v_standard: np.ndarray
    replacement_standard: np.ndarray
    noise_seeds: dict

    def outcomes(self, case):
        mean = super().outcomes(case)[0]
        change = self.change != 0
        persistent = np.stack((self.v_standard,
                               np.where(change, self.replacement_standard, self.v_standard)), axis=2)
        residual = self.cities.sigma[:, None, None] / np.sqrt(2.) * (persistent + self.epsilon)
        return mean, mean + residual

    def take(self, indices):
        city = np.arange(len(self.x))[:, None]
        return replace(self, x=self.x[city, indices], change=self.change[city, indices],
                       epsilon=self.epsilon[city, indices], tract_id=self.tract_id[city, indices],
                       v_standard=self.v_standard[city, indices],
                       replacement_standard=self.replacement_standard[city, indices])


def wrap(panel, cfg, rep, parent_panel_seed, original_tracts):
    """Assign independent component innovations by original full-panel tract ID."""
    tags = ("transitory", "persistent", "replacement")
    seeds = {tag: seed_for(cfg["seed"], NOISE_NAMESPACE, rep, parent_panel_seed, tag) for tag in tags}
    ncity = len(panel.cities.a)
    n = ncity * original_tracts
    transitory = np.random.default_rng(seeds["transitory"]).normal(size=(n, 2))[panel.tract_id]
    persistent = np.random.default_rng(seeds["persistent"]).normal(size=n)[panel.tract_id]
    replacement = np.random.default_rng(seeds["replacement"]).normal(size=n)[panel.tract_id]
    assert len(set(seeds.values())) == 3
    return PersistentPanel(panel.x, panel.change, transitory, panel.cities, panel.tract_id,
                           persistent, replacement, seeds)


def training(cfg, rep):
    seeds, fit, valid = old_training(cfg, rep)
    fit = wrap(fit, cfg, rep, seeds["paired"], cfg["paired_tracts"])
    valid = wrap(valid, cfg, rep, seeds["paired"], cfg["paired_tracts"])
    assert not np.intersect1d(fit.tract_id, valid.tract_id).size
    return seeds, fit, valid


def calibration_panels(cfg, rep, seen_cities):
    root = seed_for(cfg["seed"], PANEL_NAMESPACE, rep)
    seeds = {tag: seed_for(root, tag) for tag in
             ("unseen_cities", "seen_panel", "unseen_panel", "seen_split", "unseen_split")}
    unseen = draw_cities(cfg["cities"], seeds["unseen_cities"])
    result, design = {}, {}
    for tag, cities in (("seen", seen_cities), ("unseen", unseen)):
        panel_seed = seeds[f"{tag}_panel"]
        base = make_panel(cities, PANEL_TRACTS, panel_seed)
        full = wrap(base, cfg, rep, panel_seed, PANEL_TRACTS)
        calibration, evaluation = city_split(full, panel_seed, seeds[f"{tag}_split"])
        assert isinstance(calibration, PersistentPanel)
        assert isinstance(evaluation, PersistentPanel)
        result[f"heldout_{tag}"] = calibration, evaluation
        design[tag] = dict(
            panel_seed=panel_seed, split_seed=seeds[f"{tag}_split"], noise_seeds=full.noise_seeds,
            city_effect_seed=seeds["unseen_cities"] if tag == "unseen" else None,
            calibration_ids_sha256=array_sha(calibration.tract_id),
            evaluation_ids_sha256=array_sha(evaluation.tract_id),
            calibration_features_sha256=array_sha(calibration.x),
            evaluation_features_sha256=array_sha(evaluation.x),
            calibration_tracts_per_city=CALIBRATION_TRACTS,
            evaluation_tracts_per_city=PANEL_TRACTS-CALIBRATION_TRACTS,
            city_count=cfg["cities"], calibration_dates_per_tract=2)
    return result, design


def detection_panels(cfg, rep, seeds, seen_cities):
    n = cfg["test_tracts"]
    cutoff_cities = draw_cities(cfg["cities"], seeds["cutoff_cities"])
    cutoff = wrap(make_panel(cutoff_cities, n, seeds["cutoff"], stable=True),cfg,rep,seeds["cutoff"],n)
    unseen = draw_cities(cfg["cities"], seeds["unseen_cities"])
    panels = {tag: wrap(make_panel(cities,n,seeds[f"{tag}_dose"],stable=True),
                         cfg,rep,seeds[f"{tag}_dose"],n)
              for tag,cities in (("seen",seen_cities),("unseen",unseen))}
    for panel in panels.values():
        assert isinstance(panel.dose(.15), PersistentPanel)
        assert np.all(panel.dose(0.).change == 0) and np.all(panel.dose(.15).change != 0)
    return cutoff,panels


def noise_audit(cfg):
    seeds={tag:seed_for(cfg["seed"],NOISE_NAMESPACE,"audit",tag)
           for tag in ("persistent","replacement","u0","u1","features")}
    n=500000
    values={tag:np.random.default_rng(seed).normal(size=n) for tag,seed in seeds.items()}
    v,vp,u0,u1=(values[tag]/np.sqrt(2.) for tag in ("persistent","replacement","u0","u1"))
    e0=v+u0
    rows={}
    for name,e1,covariance,delta_var in (("stable",v+u1,.5,1.),("changed",vp+u1,0.,2.)):
        moments=dict(var0=float(np.var(e0)),var1=float(np.var(e1)),
                     covariance=float(np.cov(e0,e1,ddof=0)[0,1]),
                     correlation=float(np.corrcoef(e0,e1)[0,1]),delta_variance=float(np.var(e1-e0)),
                     correlation_with_feature=float(np.corrcoef(e1,values["features"])[0,1]))
        assert abs(moments["var0"]-1)<.015 and abs(moments["var1"]-1)<.015
        assert abs(moments["covariance"]-covariance)<.015
        assert abs(moments["delta_variance"]-delta_var)<.025
        rows[name]=moments
    return dict(component_seeds=seeds,independent_diagnostic_pairs=n,normalized_sigma=1.,
                analytic_stable=dict(endpoint_variance=1.,covariance=.5,correlation=.5,delta_variance=1.),
                analytic_changed=dict(endpoint_variance=1.,covariance=0.,correlation=0.,delta_variance=2.),
                empirical=rows)
