"""Replication-level uncertainty and figures. Never used for model selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

KEYS=['cities','truth','model','map','head','strength','where']
METRICS=['change_rmse','sign_accuracy','stable_rms','cutoff','false_positive_rate','sensitivity',
         'level_rmse','level_mean_rmse','outcome_change_rmse','mean_change_rmse']


def interval(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)];n=len(a)
    if not n: return dict(n=0,mean=None,ci_low=None,ci_high=None)
    m=float(a.mean());half=float(t.ppf(.975,n-1)*a.std(ddof=1)/np.sqrt(n)) if n>1 else None
    return dict(n=n,mean=m,ci_low=m-half if half is not None else None,ci_high=m+half if half is not None else None)


def paired_differences(frame):
    result=[]
    for control in ('B','D','D_common'):
        c=frame[frame.arm=='C'];b=frame[frame.arm==control]
        matched=c.merge(b,on=KEYS+['rep'],suffixes=('_C','_control'),validate='one_to_one')
        for key,g in matched.groupby(KEYS,dropna=False):
            context=dict(zip(KEYS,key))
            for metric in METRICS:
                delta=pd.to_numeric(g[metric+'_C'],errors='coerce')-pd.to_numeric(g[metric+'_control'],errors='coerce')
                result.append(dict(**context,comparison='C-'+control,metric=metric,**interval(delta)))
    return pd.DataFrame(result)


def summary(frame,keys,metrics):
    rows=[]
    for key,g in frame.groupby(keys,dropna=False):
        for metric in metrics:
            rows.append(dict(zip(keys,key if isinstance(key,tuple) else (key,)),metric=metric,
                             **interval(pd.to_numeric(g[metric],errors='coerce'))))
    return pd.DataFrame(rows)


def lead_figure(strata,out,d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Nominal E5 step sizes; means are computed within actual direction, not absolute values.
    s=strata[(strata['cities']==max(d['cities']))&(strata['where']=='unseen')&(strata['size'].astype(str)=='all')]
    if s.empty: raise ValueError('No unseen-city, direction-specific change means for the lead figure')
    truths=[c[0] for c in d['cases']];maps=d['maps']
    fig,axes=plt.subplots(len(truths),len(maps),figsize=(6*len(maps),4.5*len(truths)),squeeze=False)
    styles=[('A',d['heads'][0],'Ordinal A','#68747b','o')]
    styles += [('C',h,'Joint C: '+h,color,mark) for h,color,mark in zip(d['heads'],['#147d92','#be4b7a'],['s','D'])]
    styles += [('oracle_ML','oracle','Privileged ML','#b68b25','^'),('oracle_noisy_R','oracle','Oracle noisy R','#5d51a6','v')]
    series_count=0;point_count=0
    for row,truth in enumerate(truths):
        for col,kind in enumerate(maps):
            ax=axes[row,col];g=s[(s.truth==truth)&(s['map']==kind)];extent=1.
            for arm,head,label,color,marker in styles:
                part=g[(g.arm==arm)&(g['head']==head)]
                xs=[];ys=[];los=[];his=[]
                for _,grp in part.groupby(['strength','direction']):
                    x=interval(grp.true_mean);y=interval(grp.predicted_mean)
                    if y['mean'] is None: continue
                    xs.append(x['mean']);ys.append(y['mean']);los.append(y['mean']-(y['ci_low'] if y['ci_low'] is not None else y['mean']))
                    his.append((y['ci_high'] if y['ci_high'] is not None else y['mean'])-y['mean'])
                if xs:
                    series_count+=1;point_count+=len(xs)
                    order=np.argsort(xs);xs=np.array(xs)[order];ys=np.array(ys)[order]
                    ax.errorbar(xs,ys,yerr=np.array([los,his])[:,order],label=label,color=color,marker=marker,lw=1.2,capsize=2)
                    extent=max(extent,float(max(abs(xs).max(),abs(ys).max())))
            ax.plot([-extent*1.1,extent*1.1],[-extent*1.1,extent*1.1],'k--',lw=1,label='45°' if row==col==0 else None)
            ax.axhline(0,color='k',lw=.4);ax.axvline(0,color='k',lw=.4);ax.grid(alpha=.15)
            ax.set(title=truth.capitalize()+' latent truth · '+kind+' measurement',xlabel='Mean true latent change',ylabel='Mean predicted latent change')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.suptitle(f"E6 · {max(d['cities'])} fitting cities · unseen-city change panels",y=.995)
    fig.tight_layout(rect=(0,.09,1,.97))
    fig.savefig(out/'latent_change.png',dpi=180,bbox_inches='tight');fig.savefig(out/'latent_change.pdf',bbox_inches='tight');plt.close(fig)
    return dict(series=series_count,points=point_count)


def format_estimate(row):
    if pd.isna(row['mean']): return 'unavailable'
    if pd.isna(row['ci_low']): return f"{row['mean']:.4f} (n={int(row['n'])}; CI unavailable)"
    return f"{row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}] (n={int(row['n'])})"


def export(results,out,d,expected):
    out=Path(out);frame=pd.DataFrame([r for job in results for r in job['metrics']])
    strata=pd.DataFrame([r for job in results for r in job['strata']])
    frame.to_csv(out/'per_replication.csv',index=False);strata.to_csv(out/'change_strata.csv',index=False)
    pd.DataFrame([dict(**j['job'],**r) for j in results for r in j['selection']]).to_csv(out/'selection.csv',index=False)
    pd.DataFrame([dict(**j['job'],**r) for j in results for r in j['history']]).to_csv(out/'validation_history.csv',index=False)
    pd.DataFrame([dict(**j['job'],**j['checks'],**j['design']) for j in results]).to_json(out/'checks.json',orient='records',indent=2)
    paired=paired_differences(frame);paired.to_csv(out/'paired_differences.csv',index=False)
    summary(frame,KEYS+['arm'],METRICS+['zero_outcome_change_rmse','zero_mean_change_rmse']).to_csv(out/'summary.csv',index=False)
    summary(strata,KEYS+['arm','direction','size'],['true_mean','predicted_mean','rmse','sensitivity']).to_csv(out/'strata_summary.csv',index=False)
    figure_audit=lead_figure(strata,out,d)
    (out/'figure_checks.json').write_text(json.dumps(figure_audit,indent=2),encoding='utf-8')
    complete=len(results)==expected
    has_intervals=d['replications']>1
    text=['![True versus predicted latent change](latent_change.png)','',
        '# E6: outcome levels and within-tract changes','',
        f"Completed {len(results)}/{expected} paired jobs. "+('Full configured design.' if complete else 'PARTIAL RUN: do not interpret this as the complete replicated experiment.'),'',
        'The figure shows direction-specific mean changes at each nonzero E5 step size. '+('Error bars are 95% t intervals across independent replications. ' if has_intervals else 'This is a single-replication smoke run; no across-replication intervals are available. ')+
        'The dashed line is exact mean recovery. Individual-change RMSE is reported separately; good group means do not imply precise individual magnitudes. All latent scorers, including both privileged references, use the E5 fitting-pair dispersion normalization. `normalization_only_rmse` records its finite-sample effect even for the true structural function.','',
        'Selection uses fitting/validation observations only. A uses validation rank logistic loss; B and C use the fixed rank-plus-standardized-level validation rule, independent of training lambda. D receives exactly C’s executed additional scorer updates and restores its best rank-validation checkpoint; D_common instead restores its best checkpoint under C’s common rule using the frozen B head. D outcome predictions use separate post-hoc calibration without scorer updates.','',
        'The following direct paired comparisons use the largest city count, the quadratic/ReLU case when present, and step size 0.4 when present. Negative differences improve RMSE; positive differences improve sensitivity. Detection also requires inspecting its cutoff and fresh-stable false-positive rate in `summary.csv`. All other conditions and D_common comparisons are in `paired_differences.csv`.','',
        '| Measurement | Head | Target | Contrast | Difference [95% CI] |','|---|---|---|---|---|']
    truth='quadratic' if any(c[0]=='quadratic' for c in d['cases']) else d['cases'][0][0]
    strength=.4 if .4 in d['change_strengths'] else max(d['change_strengths'])
    selected=paired[(paired.cities==max(d['cities']))&(paired.truth==truth)&(paired.strength==strength)&paired.comparison.isin(['C-B','C-D'])]
    targets=[('Latent magnitudes','change_rmse','unseen'),('Detection sensitivity','sensitivity','unseen'),
             ('Reported outcome changes','outcome_change_rmse','seen'),('Conditional-mean changes','mean_change_rmse','seen')]
    for title,metric,where in targets:
        subset=selected[(selected['metric']==metric)&(selected['where']==where)]
        for _,r in subset.iterrows(): text.append(f"| {r['map']} | {r['head']} | {title} | {r['comparison']} | {format_estimate(r)} |")
    selected_c=frame[frame['arm']=='C'].drop_duplicates(['rep','cities','truth','model','map','head'])
    if 'best_step' in selected_c:
        n_initial=int((selected_c['best_step']==0).sum())
        text += ['',f"C retained its initial scorer/head checkpoint in {n_initial}/{len(selected_c)} selected fits. Executed training updates and the retained checkpoint step are different quantities; inspect `validation_history.csv` for the optimization trajectory."]
    text += ['', '## Separate conclusions', '']
    for title,metric,where in targets[:3]:
        subset=selected[(selected['metric']==metric)&(selected['where']==where)]
        text.append('**'+title+'.**')
        for (kind,head),g in subset.groupby(['map','head']):
            enough=len(g)==2 and all(g.n==d['replications']) and d['replications']>1 and complete
            better=bool((g.ci_high<0).all()) if metric.endswith('rmse') else bool((g.ci_low>0).all())
            if not has_intervals: verdict='Single-replication smoke result; no replicated conclusion or across-replication interval is available.'
            elif not enough: verdict='Configured replications are incomplete; no replicated conclusion yet.'
            elif better: verdict='The paired intervals favor joint training against both frozen calibration and matched ordinal continuation in this condition.'
            else: verdict='The paired intervals do not establish an improvement against both controls in this condition.'
            text.append(f'- {kind}, {head}: {verdict}')
        text.append('')
    text += ['Better reported outcome changes alone are successful outcome calibration. They do not establish improved latent cardinal recovery. Stable-tract RMS, cutoff and sensitivity by actual size distinguish stronger true-change response from lower nuisance movement. Cell-specific heads have no zero-shot unseen-city outcome predictions. Privileged ML knows the structural basis, mixing, measurement map and noise law; the noisy-R oracle instead uses the same scorer architecture and noisy inverse labels.', '',
        'Intervals treat replications as sampling units, retain all finite completed seeds, and use no seed selection. They are pointwise, with no multiplicity adjustment. This report describes this finite design and budget; it is not an identification claim.']
    (out/'report.md').write_text('\n'.join(text),encoding='utf-8')
