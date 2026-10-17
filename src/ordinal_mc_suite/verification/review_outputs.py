"""Reconstruct the selection rule and MC summaries independently of the runner."""
import csv,json,sqlite3
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def rows(p):
    with p.open() as f:return list(csv.DictReader(f))

def main():
    fits=rows(ROOT/'all_experiments'/'selected_fits.csv'); candidates=rows(ROOT/'all_experiments'/'candidates.csv')
    assert len(fits)==414 and len(candidates)==534
    assert sum(r['selected']=='True' for r in candidates)==414
    metrics=[k for k in fits[0] if k.endswith('_latent') or k.endswith('_observed') or k.endswith('_mse')]
    for r in fits:
        for k in metrics:
            if r[k]: assert np.isfinite(float(r[k])) and float(r[k])>=0
        if r['loss']=='ordinal_smooth':assert not r['unseen_within_difference_mse']
        if r['loss'] in ('common_logit','heterogeneous_logit'):assert r['unseen_within_difference_mse']
        if r['method']=='both_fixed':assert float(r['lambda_t'])==.3 and float(r['lambda_c'])==.3
    for model in ('linear','nn'):
        for composition in ('0.0','0.7'):
            rr=[r for r in fits if r['experiment']=='1' and r['model']==model and r['composition']==composition and r['method']=='rank_common']
            assert len(rr)==4
            for k in ('unseen_within_latent','train_cross','validation_misranking'):assert len({r[k] for r in rr})==1
        for loss in ('common_logit','heterogeneous_logit','ordinal_smooth'):
            rr=[r for r in fits if r['experiment']=='4' and r['model']==model and r['loss']==loss and r['method']=='none']
            assert len(rr)==4
            assert len({r['unseen_within_latent'] for r in rr})==1
    for folder in ('all_experiments','group_curves'):
        con=sqlite3.connect(ROOT/folder/'checkpoint.sqlite')
        for jid,text in con.execute('SELECT id,result FROM jobs'):
            result=json.loads(text); cc=[json.loads(x[0]) for x in con.execute('SELECT data FROM candidates WHERE job=?',(jid,))]; best={}
            for c in cc:
                key=(c['reg'],c['lambda_t'],c['lambda_c'])
                if key not in best or c['train_objective']<best[key]['train_objective']:best[key]=c
            winner=min(best.values(),key=lambda r:(r['validation_misranking'],-r['reg'],r['lambda_t']+r['lambda_c'],(r['lambda_t'],r['lambda_c'])))
            for k in ('reg','lambda_t','lambda_c','restart'):assert winner[k]==result[k]
        con.close()
    raw=rows(ROOT/'group_curves'/'paired_replications.csv'); summary=rows(ROOT/'group_curves'/'paired_summary.csv')
    for r in summary:
        rr=[x for x in raw if all(x[k]==r[k] for k in ('cities','n','model','truth','loss','method','metric','acquisition_strength','group_strength'))]
        assert len(rr)==2
        gains=np.array([float(x['gain']) for x in rr])
        np.testing.assert_allclose(float(r['mean']),gains.mean(),atol=1e-14)
        np.testing.assert_allclose(float(r['mc_se']),gains.std(ddof=1)/np.sqrt(2),atol=1e-14)
    print('Output audit passed: 414 jobs/534 candidates; finite metrics; logistic spacing MSE; separate weights; rank invariance; unchanged unlinked baselines; candidate selection; independent-repetition MC SEs.')
if __name__=='__main__':main()
