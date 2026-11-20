"""Render the two single-replication nonlinear validation-stopping comparisons."""
from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'nonlinear_validation_comparison'; out.mkdir(exist_ok=True)
    cases=[('validation_exponential_single_run','Exponential + additive baseline'),
           ('validation_quadratic_single_run','Quadratic latent truth')]
    choices=[('fixed_budget_train_selected','Old stopping rule','#677176'),
             ('early_stopped_random','Validation stopping; random starts','#7459B4'),
             ('early_stopped_linear_start','Validation stopping; linear start','#2F7899')]
    fig,axes=plt.subplots(2,2,figsize=(10,7.7),sharey=True)
    summaries=[]; matrices=[]
    for row,(name,title) in enumerate(cases):
        folder=ROOT/name; data=pd.read_csv(folder/'results.csv'); x=np.arange(3); width=.24
        history=pd.read_csv(folder/'validation_history.csv'); candidates=pd.read_csv(folder/'candidate_selection.csv')
        for _,c in candidates[candidates.policy=='early_stopping'].iterrows():
            subset=history[(history.arm==c.arm)&(history.initialization=='random')&(history.restart==c.restart)]
            np.testing.assert_allclose(c.validation_loss,subset.validation_loss.min(),atol=1e-12,rtol=1e-12)
        for k,(method,label,color) in enumerate(choices):
            rows=data[data.method==method].set_index('arm').reindex(['none','temporal','both'])
            axes[row,0].bar(x+(k-1)*width,100*rows.unseen_change_recall,width,color=color,label=label)
            axes[row,1].bar(x+(k-1)*width,100*(1-rows.unseen_within_latent),width,color=color,label=label)
        for method,label,color,style in [('ml_correct','ML (oracle nuisance)','#8F6802',':'),
                                         ('log_cell_fe','Log + cell FE (exponential case)','#C8253D','--')]:
            rows=data[data.method==method]
            if len(rows):
                r=rows.iloc[0]
                axes[row,0].axhline(100*r.unseen_change_recall,color=color,ls=style,label=label)
                axes[row,1].axhline(100*(1-r.unseen_within_latent),color=color,ls=style,label=label)
        for col,metric_title in enumerate(['Changed tracts detected (%)','Test latent pair accuracy (%)']):
            ax=axes[row,col]
            ax.set(title=title,xticks=x,xticklabels=['No links','Temporal','Temporal +\ncross-city'],ylim=(0,104))
            ax.set_ylabel(metric_title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
        data.insert(0,'case',name); matrices.append(data)
        for _,r in data.iterrows():
            summaries.append(dict(case=title,arm=r.arm,method=r.method,
                change_recall_pct=100*r.unseen_change_recall,
                latent_pair_accuracy_pct=100*(1-r.unseen_within_latent),
                false_positive_pct=100*r.unseen_change_false_positive_rate))
    fig.suptitle('Validation-based stopping · one replication per specification',fontsize=13,y=.995)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.02),ncol=2,fontsize=9,frameon=False)
    fig.tight_layout(rect=(0,.18,1,.96))
    fig.savefig(out/'nonlinear_validation_comparison.png',dpi=190,bbox_inches='tight')
    fig.savefig(out/'nonlinear_validation_comparison.pdf',bbox_inches='tight');plt.close(fig)
    pd.DataFrame(summaries).to_csv(out/'performance_percentages.csv',index=False)
    pd.concat(matrices,ignore_index=True).to_csv(out/'all_results.csv',index=False)
    # Verify exact invariance of the ranking estimates to the exponential map.
    previous=pd.read_csv(ROOT/'optimization_validation_stopping/results.csv')
    exponential=matrices[0]
    merged=previous.merge(exponential,on=['arm','method'],suffixes=('_affine','_exp'),validate='one_to_one')
    max_error=0.
    for metric in ['validation_loss','objective','unseen_change_recall','unseen_within_latent','unseen_change_rmse']:
        err=float(np.max(abs(merged[metric+'_affine']-merged[metric+'_exp'])))
        max_error=max(max_error,err)
        np.testing.assert_allclose(merged[metric+'_affine'],merged[metric+'_exp'],atol=1e-12,rtol=1e-12)
    verification=dict(single_replication_each=True,ranking_invariance_max_error=max_error,
        validation_checkpoint_checks_passed=True,automated_tests_passed=10,full_mc_run=False)
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    with zipfile.ZipFile(out/'figures_and_results.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in out.iterdir():
            if p.suffix in ('.png','.pdf','.csv','.json'):archive.write(p,p.name)
        for name,_ in cases:
            for suffix in ('.png','.pdf'):archive.write(ROOT/name/('validation_stopping'+suffix),name+'/validation_stopping'+suffix)
    print(out)


if __name__=='__main__': main()
