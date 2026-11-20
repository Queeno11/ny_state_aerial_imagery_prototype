"""Add the invariant ML reference to the finished exact additive E1 charts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import e1_additive_exact as ex

ROOT=Path(__file__).resolve().parent


def main():
    folder=ROOT/'e1_additive_exact'
    records=pd.read_csv(folder/'replications.csv')
    fits=pd.read_csv(ROOT/'overnight_measurement/selected_fits.csv')
    original_summary=pd.read_csv(folder/'summary.csv')
    ex.check_baseline(records,fits[(fits.psi_stage=='multiplier') & (fits.composition==0)])
    ml=pd.read_csv(ROOT/'overnight_measurement/selected_fits_with_ml.csv')
    ml=ml[(ml.method=='ml_correct') & (ml.psi_stage=='multiplier') & (ml.composition==0)]
    if ml.duplicated(['cities','rep']).any(): raise ValueError('Ambiguous ML reference sample')
    ml=ml[['cities','rep',ex.METRIC,'ml_success']].rename(columns={ex.METRIC:'ml_risk'})
    augmented=records.merge(ml,on=['cities','rep'],validate='many_to_one')
    if len(augmented)!=len(records): raise ValueError('Missing additive ML samples')
    summary=ex.summarize(augmented,fits[(fits.psi_stage=='multiplier') & (fits.composition==0)])
    keys=['method','cities','psi_stage','psi_strength','metric']
    comparison=original_summary.merge(summary,on=keys,validate='one_to_one',suffixes=('_old','_new'))
    if len(comparison)!=len(original_summary): raise ValueError('Original additive summary missing')
    np.testing.assert_allclose(comparison.mean_old,comparison.mean_new,rtol=0,atol=1e-14)
    augmented.to_csv(folder/'replications_with_ml.csv',index=False)
    summary.to_csv(folder/'summary_with_ml.csv',index=False)
    manifest=dict(original_replications_sha256=hashlib.sha256((folder/'replications.csv').read_bytes()).hexdigest(),
                  existing_summary_verified=True,original_estimators_rerun=0,
                  matched_ml_fits=len(ml),reason='Same latent data/noise and known inverse transformations; ML fit invariant to additive map')
    (folder/'ml_append_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(manifest,flush=True)


if __name__=='__main__': main()
