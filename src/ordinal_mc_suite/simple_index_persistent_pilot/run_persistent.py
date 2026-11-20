"""Five matched replications with half persistent, half transitory residual variance."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from prior import HERE, SIMPLE
from persistent_data import (NOISE_NAMESPACE, calibration_panels, detection_panels,
                             noise_audit, training)
from correlated_ml import check as check_correlated, fit as correlated_fit
from calibrate_10pct import array_sha, calibrate, outcome_rows, read, sha, structural_rows, write
from changed_only_delta import fit_changed, fit_masked_head, head_kkt, numerical_checks
from classical import fixed_effects, Model
from data import CASES, TRUE_BETA, external_views, training as original_training
from estimators import estimators, library, optimize, tune
from losses import (fit_head, fit_index, head_prediction, prepare, smooth_rank, spherical_grid, unit,
                    validation)
from run_simple import json_numpy


WORKING = {"Gaussian_ML":"Gaussian_ML_working_independence",
           "Gaussian_ML_S":"Gaussian_ML_S_working_independence"}
ALL = "Ordinal_L2_D_all_matched"
CHANGED = "Ordinal_L2_D_changed_matched"
MASS = "Ordinal_L2_D_all_massmatched"
DETECTION_DOSES = [-.4,-.15,-.05,-.02,-.01,-.005,-.002,0.,.002,.005,.01,.02,.05,.15,.4]


def dataset(cfg,rep):
    seeds,fit,valid=training(cfg,rep)
    links=external_views(fit,seeds["fit_links"],cfg["stability_views"])
    vlinks=external_views(valid,seeds["validation_links"],cfg["stability_views"])
    return seeds,fit,valid,links,vlinks


def data_for(case,fit,valid,links,vlinks):
    a=prepare(fit,case,links)
    b=prepare(valid,case,vlinks,(a.denom_w,a.denom_d))
    return a,b


def check_panel(panel):
    c=panel.change!=0
    v0=panel.v_standard
    v1=np.where(c,panel.replacement_standard,v0)
    assert np.array_equal(v0[~c],v1[~c])
    for case in CASES:
        mean,w=panel.outcomes(case)
        e=w-mean
        expected=panel.cities.sigma[:,None]/np.sqrt(2.)*(
            panel.epsilon[...,1]-panel.epsilon[...,0]+v1-v0)
        np.testing.assert_allclose(e[...,1]-e[...,0],expected,atol=2e-14,rtol=2e-13)
        transient=panel.cities.sigma[:,None]/np.sqrt(2.)*(panel.epsilon[...,1]-panel.epsilon[...,0])
        np.testing.assert_allclose((e[...,1]-e[...,0])[~c],transient[~c],atol=2e-14,rtol=2e-13)
    return dict(stable_v_exact=True,residual_delta_matches_components=True,
                stable_delta_has_only_transitory_noise=True,changed_fraction=float(c.mean()))


def preflight(cfg):
    report=dict(noise=noise_audit(cfg),conditional_likelihood={},panels={})
    seeds,fit,valid,links,vlinks=dataset(cfg,101)
    _,old_fit,old_valid=original_training(cfg,101)
    np.testing.assert_array_equal(fit.x,old_fit.x)
    np.testing.assert_array_equal(valid.x,old_valid.x)
    report["panels"]["fit"]=check_panel(fit)
    report["panels"]["validation"]=check_panel(valid)
    for case in CASES:
        np.testing.assert_array_equal(fit.outcomes(case)[0],old_fit.outcomes(case)[0])
        assert not np.array_equal(fit.outcomes(case)[1],old_fit.outcomes(case)[1])
        data,_=data_for(case,fit,valid,links,vlinks)
        beta=fixed_effects(data).beta
        if case!="A":
            beta=unit(beta)
        report["conditional_likelihood"][case]=check_correlated(data,fit.change!=0,beta)
    panels,_=calibration_panels(cfg,1,old_fit.cities)
    for name,(cal,test) in panels.items():
        report["panels"][name+"_calibration"]=check_panel(cal)
        report["panels"][name+"_evaluation"]=check_panel(test)
        assert not np.intersect1d(cal.tract_id,test.tract_id).size
        for dose in (0.,-.15,.15):
            report["panels"][name+f"_dose_{dose:g}"]=check_panel(test.dose(dose))
    return report


def calibrate_weights(cfg,out):
    seeds,fit,valid,links,vlinks=dataset(cfg,101)
    weights,rows={},[]
    for case in CASES:
        data,val=data_for(case,fit,valid,links,vlinks)
        weights[case],candidate=tune(data,val,cfg)
        rows.extend(dict(case=case,**r) for r in candidate)
        print(f"New-law calibration {case}: {weights[case]['grid_winner']}",flush=True)
    assert len(rows)==39
    pd.DataFrame(rows).to_csv(out/"calibration_candidates.csv",index=False)
    write(out/"frozen_selection.json",dict(rep=101,seeds=seeds,noise_seeds=fit.noise_seeds,weights=weights,
                                          tests_used_for_selection=False))
    return weights


def fit_battery(cfg,case,data,val,panel,newweights,oldweights):
    models,_,values,audits=estimators(data,val,cfg,newweights)
    models={WORKING.get(name,name):value for name,value in models.items()}
    audits={WORKING.get(name,name):value for name,value in audits.items()}
    values={WORKING.get(name,name):value for name,value in values.items()}
    start_raw=fixed_effects(data).beta
    start=unit(start_raw)
    grid=spherical_grid(cfg["sphere_grid"])
    cache=[smooth_rank(data.pairs,b,cfg["tau"])[0] for b in grid]
    changed=panel.change!=0
    w,d=newweights["level"],newweights["full"][1]
    weights=(1.,cfg["lambda_S"],w,d,0.)
    check=numerical_checks(data,cfg,weights,start,changed)
    for name in (ALL,CHANGED,MASS):
        active_weights=weights if name!=MASS else (1.,cfg["lambda_S"],w,d*changed.mean(),0.)
        if name==CHANGED:
            beta,info=fit_changed(data,cfg,active_weights,start,grid,cache,changed)
            native=fit_masked_head(beta,data,w,d,changed)
            mask=changed
        else:
            beta,info=fit_index(data,active_weights,cfg,start,grid,cache)
            native=fit_head(beta,data,w,active_weights[3],0.)
            mask=np.ones_like(changed)
        models[name]=Model(beta,native)
        info.update(training_weights=list(active_weights),oracle_change_mask=name==CHANGED,
                    fitting_changed_fraction=float(changed.mean()),
                    head_kkt_gradient=head_kkt(beta,native,data,active_weights,mask))
        audits[name]=info
        values[name]=dict(own=validation(beta,models[name].predict(val.x),val,cfg))
    for name,s in (("Gaussian_ML_correlated_oracle",0.),
                   ("Gaussian_ML_S_correlated_oracle",cfg["lambda_S"])):
        model,info=correlated_fit(data,changed,cfg,start_raw,grid,s)
        info.update(oracle_change_status_and_covariance_shape=True,estimated_city_sigma=True)
        models[name],audits[name]=model,info
        values[name]=dict(own=validation(model.beta,model.predict(val.x),val,cfg))
    aliases={}
    options=library(data,cfg,newweights)
    specifications=(
        ("Ordinal_L2_S_oldweights","Ordinal_L2_S",(1.,cfg["lambda_S"],oldweights["level"],0.,0.),
         (newweights["level"],0.)),
        ("Ordinal_L2_D_S_oldweights","Ordinal_L2_D_S",
         (1.,cfg["lambda_S"],*oldweights["full"],0.),tuple(newweights["full"])),
    )
    for name,primary,weights,new_pair in specifications:
        if tuple(weights[2:4])==tuple(new_pair):
            models[name]=models[primary]
            values[name]=values[primary]
            aliases[name]=primary
        else:
            model,info,v=optimize(data,val,cfg,weights,start,grid,cache,options)
            models[name],audits[name]=model,info
            info["training_weights"]=list(weights)
            values[name]=dict(own=v)
    return models,audits,values,check,aliases


def detect(rep,case,models,cutoff,test_panels,cfg):
    rows=[]
    for method,model in models.items():
        beta=unit(model.beta)
        movement=(cutoff.x[...,1,:]-cutoff.x[...,0,:])@beta
        threshold=float(np.quantile(abs(movement),.95,method="higher"))
        for where,panel in test_panels.items():
            for dose in DETECTION_DOSES:
                p=panel.dose(dose)
                true=p.change.ravel()
                estimated=np.diff(p.x@beta,axis=2)[...,0].ravel()
                rows.append(dict(rep=rep,case=case,method=method,where=where,dose=dose,count=len(true),
                                 cutoff=threshold,flag_rate=float(np.mean(abs(estimated)>threshold)),
                                 sign_accuracy=float(np.mean(np.sign(estimated)==np.sign(true))) if dose else None,
                                 true_mean=float(true.mean()),prediction_mean=float(estimated.mean()),
                                 mse=float(np.mean((estimated-true)**2))))
    return rows


def summarize(city,out):
    frame=pd.DataFrame(city)
    frame.to_csv(out/"city_metrics.csv",index=False)
    keys=["rep","case","method","panel","target"]
    reps=frame.groupby(keys).mse.agg(city_count="size",city_mean_mse="mean",city_sd_mse="std",
                                    city_median_mse="median",city_max_mse="max").reset_index()
    assert reps.city_count.eq(20).all()
    reps.to_csv(out/"per_rep_city_summary.csv",index=False)
    summary=reps.groupby(["case","method","panel","target"]).agg(
        replication_count=("rep","size"),mean_city_mse=("city_mean_mse","mean"),
        mean_within_rep_city_sd=("city_sd_mse","mean"),
        mean_within_rep_city_median=("city_median_mse","mean"),
        mean_within_rep_city_max=("city_max_mse","mean"),
        min_rep_mean_mse=("city_mean_mse","min"),max_rep_mean_mse=("city_mean_mse","max")).reset_index()
    assert summary.replication_count.eq(5).all()
    summary.to_csv(out/"five_rep_summary.csv",index=False)
    contrasts=(
        ("Ordinal_L2_S","Ordinal_S"),("Ordinal_L2_D_S","Ordinal_L2_S"),(CHANGED,ALL),(CHANGED,MASS),
        (MASS,ALL),(ALL,"Ordinal_L2_S"),(CHANGED,"Ordinal_L2_S"),
        ("L2_D_S","L2_S"),("Gaussian_ML_S_correlated_oracle","Gaussian_ML_correlated_oracle"),
        ("Gaussian_ML_S_correlated_oracle","Gaussian_ML_S_working_independence"),
        ("Ordinal_S","Gaussian_ML_S_correlated_oracle"),
        ("Ordinal_L2_S_oldweights","Ordinal_S"),("Ordinal_L2_D_S_oldweights","Ordinal_L2_S_oldweights"))
    pairs=[]
    for a,b in contrasts:
        index=["rep","case","panel","target"]
        x,y=(reps[reps.method.eq(name)].set_index(index).city_mean_mse for name in (a,b))
        pairs.append((x-y).dropna().rename("difference").reset_index().assign(method=a,reference=b))
    pair=pd.concat(pairs,ignore_index=True)
    pair.to_csv(out/"paired_by_rep.csv",index=False)
    pair.groupby(["case","method","reference","panel","target"]).difference.agg(
        replication_count="size",mean_difference="mean",minimum="min",maximum="max",
        lower=lambda x:(x<0).sum(),higher=lambda x:(x>0).sum(),ties=lambda x:(x==0).sum()).reset_index().to_csv(
            out/"paired_summary.csv",index=False)


def run(cfg,out,old_run,old_cal):
    started=time.perf_counter()
    write(out/"preflight.json",preflight(cfg))
    newweights=calibrate_weights(cfg,out)
    oldweights=read(old_run/"frozen_selection.json")["weights"]
    old_cal_manifest=read(old_cal/"manifest.json")
    original_hashes={str(path):sha(path.read_bytes()) for path in old_run.glob("rep_*/case_*_models.json")}
    primary,heads,diagnostics,validations,native_seen,detected,mask_checks=[],[],[],[],[],[],[]
    count=0
    aliases_all={}
    feature_checks=[]
    noise_seeds=set()
    for rep in cfg["replications"]:
        t0=time.perf_counter()
        dest=out/f"rep_{rep}"
        dest.mkdir()
        seeds,fit,valid,links,vlinks=dataset(cfg,rep)
        _,oldfit,oldvalid=original_training(cfg,rep)
        np.testing.assert_array_equal(fit.x,oldfit.x)
        np.testing.assert_array_equal(valid.x,oldvalid.x)
        np.testing.assert_array_equal(fit.change,oldfit.change)
        assert not set(fit.noise_seeds.values())&noise_seeds
        noise_seeds.update(fit.noise_seeds.values())
        check_panel(fit);check_panel(valid)
        fits={}
        aliases_all[str(rep)]={}
        for case in CASES:
            data,val=data_for(case,fit,valid,links,vlinks)
            models,audits,values,check,aliases=fit_battery(cfg,case,data,val,fit,newweights[case],oldweights[case])
            fits[case]=models
            count+=len(models)-len(aliases)
            aliases_all[str(rep)][case]=aliases
            mask_checks.append(dict(rep=rep,case=case,**check))
            diagnostics.extend(dict(rep=rep,case=case,method=name,**info) for name,info in audits.items())
            validations.extend(dict(rep=rep,case=case,method=name,**value["own"]) for name,value in values.items())
            for method,model in models.items():
                structural_rows(primary,dict(rep=rep,case=case,method=method,panel="original_score_fit"),
                                fit,fit.x@unit(model.beta))
            print(f"Persistent battery fitted {rep}/{case}: {len(models)} named methods",flush=True)
        role_panels,design=calibration_panels(cfg,rep,fit.cities)
        for tag in ("seen","unseen"):
            for field in ("panel_seed","split_seed","calibration_ids_sha256","evaluation_ids_sha256",
                          "calibration_features_sha256","evaluation_features_sha256"):
                assert design[tag][field]==old_cal_manifest["panels"][str(rep)][tag][field]
            noise=design[tag]["noise_seeds"]
            assert not noise_seeds&set(noise.values())
            noise_seeds.update(noise.values())
        for role,(cal,test) in role_panels.items():
            check_panel(cal);check_panel(test)
            for case,models in fits.items():
                observed_cal=cal.outcomes(case)[1]
                row_heads={}
                for method,model in models.items():
                    beta=unit(model.beta)
                    score=test.x@beta
                    head,kkt=calibrate(beta,cal.x,observed_cal,case)
                    np.testing.assert_array_equal(score,test.x@beta)
                    row_heads[method]=head
                    heads.append(dict(rep=rep,case=case,method=method,panel=role,beta=beta.tolist(),head=head.tolist(),
                                      calibration_gradient=kkt,features_sha256=array_sha(cal.x),
                                      outcomes_sha256=array_sha(observed_cal)))
                mean,observed=test.outcomes(case)
                for method,model in models.items():
                    beta=unit(model.beta)
                    prefix=dict(rep=rep,case=case,method=method,panel=role)
                    structural_rows(primary,prefix,test,test.x@beta)
                    outcome_rows(primary,prefix,head_prediction(beta,row_heads[method],test.x),mean,observed)
                    if role=="heldout_seen":
                        outcome_rows(native_seen,prefix,model.predict(test.x),mean,observed)
        cutoff,dose_panels=detection_panels(cfg,rep,seeds,fit.cities)
        for case,models in fits.items():
            detected.extend(detect(rep,case,models,cutoff,dose_panels,cfg))
            saved={name:asdict(model) for name,model in models.items()}
            write(dest/f"case_{case}_models.json",json.loads(json.dumps(saved,default=json_numpy)))
        write(dest/"design.json",dict(seeds=seeds,fit_noise_seeds=fit.noise_seeds,
                                      calibration_panels=design,fit_features_sha256=array_sha(fit.x),
                                      validation_features_sha256=array_sha(valid.x)))
        feature_checks.append(rep)
        print(f"Persistent replication {rep} complete in {time.perf_counter()-t0:.1f}s",flush=True)
    for path,digest in original_hashes.items():
        assert sha(Path(path).read_bytes())==digest
    summarize(primary,out)
    write(out/"calibrated_heads.json",heads)
    pd.DataFrame(native_seen).to_csv(out/"native_seen_city_outcomes.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(out/"optimizer.csv",index=False)
    pd.DataFrame(validations).to_csv(out/"validation.csv",index=False)
    pd.DataFrame(detected).to_csv(out/"detection.csv",index=False)
    pd.DataFrame(mask_checks).to_csv(out/"mask_checks.csv",index=False)
    assert count<=280 and len(feature_checks)==5
    assert len(heads)==560
    h=pd.DataFrame(heads)
    assert h.groupby(["rep","case","panel"]).outcomes_sha256.nunique().eq(1).all()
    write(out/"audit.json",dict(
        passed=True,scorer_fits=count,maximum_scorer_fits=280,tuning_fits=39,replications=feature_checks,
        same_structural_inputs_city_parameters_changes_and_splits=True,same_calibration_test_feature_hashes=True,
        original_model_files_unchanged=True,conditional_head_objectives_mask_checked=True,
        shared_calibration_labels=True,calibration_head_sets=len(heads),score_freeze_checks=len(heads),
        max_calibration_gradient=max(x["calibration_gradient"] for x in heads),
        max_mask_finite_difference_error=max(x["directional_finite_difference_error"] for x in mask_checks),
        independently_namespaced_noise_seeds=len(noise_seeds),aliases=aliases_all,
        elapsed_seconds=time.perf_counter()-started))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    cfg=read(SIMPLE/"config.json")
    old_run,old_cal=SIMPLE/"run_5rep",SIMPLE/"calibration_10pct"
    args.out.mkdir(parents=True,exist_ok=False)
    source_files=list(HERE.glob("*.py"))+[SIMPLE/name for name in (
        "data.py","losses.py","estimators.py","classical.py","calibrate_10pct.py","changed_only_delta.py")]
    write(args.out/"manifest.json",dict(
        config=cfg,noise_namespace=NOISE_NAMESPACE,total_endpoint_noise_variance="sigma_city squared",
        persistent_variance_share=.5,transitory_variance_share=.5,
        stable_pair_correlation=.5,changed_pair_correlation=0.,
        source_sha256={str(p):sha(p.read_bytes()) for p in source_files},
        original_run_manifest_sha256=sha((old_run/"manifest.json").read_bytes()),
        original_calibration_manifest_sha256=sha((old_cal/"manifest.json").read_bytes()),
        primary_target="conditional mean psi_city(f), integrating persistent and transitory noise",
        detection_doses=DETECTION_DOSES,
        working_likelihood="Gaussian_ML*_working_independence ignores pair correlation",
        oracle_likelihood="Gaussian_ML*_correlated_oracle knows true change-status covariance shape; city sigma estimated",
        fixed_oldweights="level and full ordinal coefficients from old independent-noise tuning; no retuning",
        platform=platform.platform()))
    try:
        with threadpool_limits(limits=1):
            run(cfg,args.out,old_run,old_cal)
    except Exception:
        (args.out/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    main()
