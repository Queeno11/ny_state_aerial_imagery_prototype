"""All-pair difference loss with the same total mass as the oracle changed mask."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from calibrate_10pct import (array_sha, calibrate, outcome_rows, panels, read, sha, structural_rows, write)
from changed_only_delta import (ALL, ANCHOR, CHANGED, head_kkt, numerical_checks, summarize)
from classical import Model, fixed_effects
from data import CASES
from losses import fit_head, fit_index, head_prediction, smooth_rank, spherical_grid, unit, validation
from run_simple import case_data, create_data, json_numpy

ROOT = Path(__file__).resolve().parent
MASS = "Ordinal_L2_D_all_massmatched"


def additional_contrasts(out):
    reps = pd.read_csv(out/"per_rep_city_summary.csv")
    pairs = pd.read_csv(out/"paired_by_rep.csv")
    new = []
    for method, reference in ((CHANGED,MASS),(MASS,ALL),(MASS,ANCHOR)):
        index=["rep","case","panel","target"]
        a,b=(reps[reps.method.eq(name)].set_index(index).city_mean_mse for name in (method,reference))
        new.append((a-b).dropna().rename("difference").reset_index().assign(method=method,reference=reference))
    pairs=pd.concat([pairs,*new],ignore_index=True)
    pairs.to_csv(out/"paired_by_rep.csv",index=False)
    stats=pairs.groupby(["case","method","reference","panel","target"]).difference.agg(
        replication_count="size",mean_difference="mean",minimum="min",maximum="max",
        lower=lambda x:(x<0).sum(),higher=lambda x:(x>0).sum(),ties=lambda x:(x==0).sum()).reset_index()
    assert stats.replication_count.eq(5).all()
    stats.to_csv(out/"paired_summary.csv",index=False)


def run(source,calibration,changed,out):
    cfg=read(source/"manifest.json")["config"]
    selection=read(source/"frozen_selection.json")["weights"]
    old_manifest=read(calibration/"manifest.json")
    assert read(changed/"audit.json")["passed"]
    fingerprints={str(path):sha(path.read_bytes()) for path in
                  [source/"manifest.json",changed/"manifest.json",changed/"audit.json",calibration/"manifest.json"]}
    write(out/"manifest.json",dict(
        config=cfg,script_sha256=sha(Path(__file__).read_bytes()),source_sha256=fingerprints,new_scorer_fits=15,
        coefficient_rule="old delta coefficient times actual fraction of fitting pairs with true change != 0",
        training_level_weight="same frozen Ordinal_L2_S level weight",curvature=0.,
        difference_loss="all-pair mean; original all-pair variance denominator; no selective mask",
        role="supplementary total difference-loss mass control"))
    old_heads=read(calibration/"calibrated_heads.json")
    used=set(read(source/"frozen_selection.json")["fitting_and_validation_seeds"].values())
    for rep in cfg["replications"]:
        used.update(read(source/f"rep_{rep}/design.json")["seeds"].values())
    rows,heads,optimizer,numerical,values=[],[],[],[],[]
    started=time.perf_counter()
    for rep in cfg["replications"]:
        dest=out/f"rep_{rep}"
        dest.mkdir()
        seeds,panel,valid,links,val_links=create_data(cfg,rep)
        role_panels,design=panels(cfg,rep,panel,seeds["paired"],used)
        for tag in ("seen","unseen"):
            for key in ("panel_seed","split_seed","calibration_features_sha256","evaluation_features_sha256",
                        "calibration_ids_sha256","evaluation_ids_sha256"):
                assert design[tag][key]==old_manifest["panels"][str(rep)][tag][key]
        fraction=float(np.mean(panel.change!=0))
        fitted={}
        for case in CASES:
            fit,val=case_data(case,panel,valid,links,val_links)
            level=selection[case]["level"]
            original_delta=selection[case]["full"][1]
            weights=(1.,cfg["lambda_S"],level,original_delta*fraction,0.)
            start=unit(fixed_effects(fit).beta)
            grid=spherical_grid(cfg["sphere_grid"])
            cache=[smooth_rank(fit.pairs,b,cfg["tau"])[0] for b in grid]
            ones=np.ones_like(panel.change,dtype=bool)
            check=numerical_checks(fit,cfg,weights,start,ones)
            numerical.append(dict(rep=rep,case=case,**check))
            beta,audit=fit_index(fit,weights,cfg,start,grid,cache)
            native=fit_head(beta,fit,level,weights[3],0.)
            kkt=head_kkt(beta,native,fit,weights,ones)
            model=Model(beta,native)
            fitted[case]=(model,unit(beta))
            optimizer.append(dict(rep=rep,case=case,method=MASS,fitting_changed_fraction=fraction,
                                  level_weight=level,original_delta_weight=original_delta,
                                  scaled_delta_weight=weights[3],delta_denominator=fit.denom_d,
                                  head_kkt_gradient=kkt,**audit))
            values.append(dict(rep=rep,case=case,method=MASS,
                               **validation(beta,model.predict(valid.x),val,cfg)))
            structural_rows(rows,dict(rep=rep,case=case,method=MASS,panel="original_score_fit"),
                            panel,panel.x@unit(beta))
            print(f"Fitted mass control {rep}/{case} at fraction {fraction:.6f}",flush=True)
        for case in CASES:
            _,beta=fitted[case]
            for role,(cal,test) in role_panels.items():
                observed_cal=cal.outcomes(case)[1]
                old=next(x for x in old_heads if x["rep"]==rep and x["case"]==case and x["panel"]==role)
                assert array_sha(cal.x)==old["calibration_features_sha256"]
                assert array_sha(observed_cal)==old["calibration_outcome_sha256"]
                score=test.x@beta
                head,kkt=calibrate(beta,cal.x,observed_cal,case)
                np.testing.assert_array_equal(score,test.x@beta)
                heads.append(dict(rep=rep,case=case,method=MASS,panel=role,beta=beta.tolist(),head=head.tolist(),
                                  calibration_gradient=kkt,calibration_outcome_sha256=array_sha(observed_cal)))
                mean,observed=test.outcomes(case)
                prefix=dict(rep=rep,case=case,method=MASS,panel=role)
                structural_rows(rows,prefix,test,score)
                outcome_rows(rows,prefix,head_prediction(beta,head,test.x),mean,observed)
        saved={case:asdict(model) for case,(model,_) in fitted.items()}
        write(dest/"native_models.json",json.loads(json.dumps(saved,default=json_numpy)))
    for path,digest in fingerprints.items():
        assert sha(Path(path).read_bytes())==digest
    pd.DataFrame(optimizer).to_csv(out/"optimizer.csv",index=False)
    pd.DataFrame(numerical).to_csv(out/"numerical_checks.csv",index=False)
    pd.DataFrame(values).to_csv(out/"validation.csv",index=False)
    pd.DataFrame(rows).to_csv(out/"new_city_metrics.csv",index=False)
    write(out/"calibrated_heads.json",heads)
    summarize(rows,changed,out)
    additional_contrasts(out)
    assert len(optimizer)==15 and len(heads)==30
    write(out/"audit.json",dict(
        passed=True,new_scorer_fits=15,total_new_scorer_fits_including_mask_experiment=45,
        score_freeze_checks=30,calibration_head_sets=30,
        all_previous_stage_files_unchanged=True,
        shared_calibration_and_test_hashes_match=True,original_difference_denominator_unchanged=True,
        min_fitting_changed_fraction=min(x["fitting_changed_fraction"] for x in optimizer),
        max_fitting_changed_fraction=max(x["fitting_changed_fraction"] for x in optimizer),
        max_tangent_gradient=max(x["min_tangent_grad"] for x in optimizer),
        max_conditional_head_gradient=max(x["head_kkt_gradient"] for x in optimizer),
        elapsed_seconds=time.perf_counter()-started))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--calibration",type=Path,required=True)
    parser.add_argument("--changed",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    with threadpool_limits(limits=1):
        run(args.run,args.calibration,args.changed,args.out)


if __name__=="__main__":
    main()
