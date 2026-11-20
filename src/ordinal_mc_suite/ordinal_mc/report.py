"""CSV summaries and paired Monte Carlo comparisons; repetitions are the units."""
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from .change import CHANGE_METRICS, HIGHER_BETTER

METRICS=[f'{where}_{kind}_latent' for where in ('seen','unseen') for kind in ('within','cross_city','across_time')]+['seen_within_observed','unseen_within_observed','seen_oracle_observed','unseen_oracle_observed','seen_within_difference_mse','unseen_within_difference_mse']
METRICS += [f'{where}_{kind}_accuracy' for where in ('seen','unseen') for kind in ('within','cross_city','across_time')]
METRICS += [f'{where}_{metric}' for where in ('seen','unseen') for metric in CHANGE_METRICS]
SCENARIO=['error','error_severity','psi_shape','psi_delta','psi_stage','psi_strength','positive_labels','composition','link_case','link_severity','acquisition_strength','group_strength','change_strength']
FACTORS=['experiment','n','cities','truth','model','loss','method','arm']+SCENARIO

def write_csv(path,rows):
    if not rows:
        path.write_text(''); return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)

def summarize(values):
    v=np.asarray(values,float); n=len(v); mean=float(v.mean())
    sd=float(v.std(ddof=1)) if n>1 else None
    se=sd/math.sqrt(n) if sd is not None else None
    return dict(repetitions=n,mean=mean,mc_sd=sd,mc_se=se,
       ci95_low=mean-1.96*se if se is not None else None,ci95_high=mean+1.96*se if se is not None else None)

def comparison_key(r,method=None):
    keys=['experiment','rep','n','cities','truth','model']+SCENARIO
    # Exp1 compares level losses with ranking; exp2 compares ranking losses.
    if r['experiment'] in (3,4,5): keys+=['loss']
    return tuple(r[k] for k in keys)+(method or r['method'],)

def export(conn,out,plots=True):
    rows=[json.loads(x[0]) for x in conn.execute('SELECT result FROM jobs ORDER BY rowid')]
    candidates=[json.loads(x[0]) for x in conn.execute('SELECT data FROM candidates ORDER BY rowid')]
    write_csv(out/'selected_fits.csv',rows); write_csv(out/'candidates.csv',candidates)
    groups=defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in FACTORS)].append(r)
    summaries=[]
    for key,rr in groups.items():
        for metric in METRICS:
            values=[r[metric] for r in rr if r.get(metric) is not None and np.isfinite(r[metric])]
            if not values: continue
            summaries.append(dict(zip(FACTORS,key),metric=metric,missing_repetitions=len(rr)-len(values),**summarize(values)))
    write_csv(out/'summary.csv',summaries)
    lookup={comparison_key(r):r for r in rows}; paired=[]
    for r in rows:
        if r['method']==r['baseline']: continue
        b=lookup.get(comparison_key(r,r['baseline']))
        if b is None: continue
        for metric in METRICS:
            if r.get(metric) is None or b.get(metric) is None: continue
            suffix=metric.split('_',1)[1]
            if suffix in ('change_slope','change_bias','change_threshold','change_true_share','change_true_rms','change_increase_share','change_decrease_share'): continue
            sign=-1 if metric.endswith('_accuracy') or suffix in HIGHER_BETTER else 1
            gain=sign*(b[metric]-r[metric])
            paired.append({**{k:r[k] for k in FACTORS},'rep':r['rep'],'baseline':r['baseline'],'metric':metric,
              'method_error':r[metric],'baseline_error':b[metric],'gain':gain,
              'higher_is_better':sign==-1,'relative_gain':gain/b[metric] if b[metric] else None})
    write_csv(out/'paired_replications.csv',paired)
    pg=defaultdict(list)
    for r in paired: pg[tuple(r[k] for k in FACTORS)+ (r['baseline'],r['metric'])].append(r)
    ps=[]
    for key,rr in pg.items():
        stats=summarize([r['gain'] for r in rr])
        stats['positive_gain_share']=float(np.mean([r['gain']>0 for r in rr]))
        b=np.mean([r['baseline_error'] for r in rr]); m=np.mean([r['method_error'] for r in rr])
        stats['relative_reduction_of_mean']=float(np.mean([r['gain'] for r in rr])/b) if b else None
        ps.append(dict(zip(FACTORS+['baseline','metric'],key),**stats))
    write_csv(out/'paired_summary.csv',ps)
    diag=dict(selected_fits=len(rows),candidate_fits=len(candidates),
      audit_unstable=sum(r.get('audit_stable') is False for r in rows),
      first_order_not_small=sum(r.get('first_order_small') is False for r in rows),
      recovered_selected_fits=sum(r.get('optimizer_status')=='recovered' for r in rows),
      recovered_candidates=sum(r.get('optimizer_status')=='recovered' for r in candidates),
      inadmissible_candidates=sum(r.get('optimizer_status')=='inadmissible' for r in candidates),
      ml_not_converged=sum(r.get('ml_success') is False for r in rows),
      note='Diagnostics are not global optimality certificates. CI uses independent repetitions, not pairs; fewer than 20 repetitions is exploratory.')
    (out/'diagnostics.json').write_text(json.dumps(diag,indent=2))
    if plots and summaries: make_plots(summaries,out/'figures')
    return diag

def make_plots(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    panels=defaultdict(list)
    for r in rows:
        exp=r['experiment']
        plot_metrics=('unseen_change_rmse','unseen_change_direction_accuracy','unseen_change_recall','unseen_change_false_positive_rate','unseen_change_slope','unseen_change_auc') if exp==5 else ('unseen_within_latent','unseen_within_difference_mse')
        if r['metric'] not in plot_metrics: continue
        axis='cities' if exp in (1,3) else 'error_severity' if exp==2 else 'change_strength' if exp==5 else 'link_severity'
        factors=['experiment','truth','model','metric','n']+[k for k in SCENARIO if k!=axis]
        if exp in (2,4,5): factors+=['cities']
        if exp in (3,4,5): factors+=['loss']
        panels[(axis,tuple((k,r[k]) for k in factors))].append(r)
    for index,((axis,key),rr) in enumerate(panels.items()):
        info=dict(key); exp=info['experiment']; methods=defaultdict(list)
        for r in rr: methods[r['method']].append(r)
        fig,ax=plt.subplots(figsize=(6.7,4.5))
        for method,series in methods.items():
            series.sort(key=lambda r:r[axis])
            ax.errorbar([r[axis] for r in series],[r['mean'] for r in series],yerr=[r['mc_se'] or 0. for r in series],marker='o',capsize=3,label=method)
        label={'cities':'Number of labeled training groups','error_severity':'Error departure dial','link_severity':'Link departure dial','change_strength':'Structural feature change (step length)'}[axis]
        ylabel='Within-cell misranking in unseen groups' if info['metric']=='unseen_within_latent' else 'Normalized within-cell difference MSE, unseen groups'
        if exp==5: ylabel=info['metric'].removeprefix('unseen_').replace('_',' ')
        details=f"Ψ={info['psi_stage']} ({info['psi_strength']:g}), composition={info['composition']:g}" if exp==1 else f"errors={info['error']}" if exp==2 else f"acquisition={info['acquisition_strength']:g}, group cue={info['group_strength']:g}" if exp==3 else info['link_case']
        if exp==5: details=f"Changed links treated as stable: {info['link_severity']:.0%}"
        ax.set(xlabel=label,ylabel=ylabel,title=f"E{exp} · {info['truth']} / {info['model']} · {info.get('loss','loss comparison')}\n{details}")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=6.5,loc='best'); fig.text(.02,.01,'Bars: ±1 Monte Carlo SE; smoke runs check implementation only.',fontsize=7)
        fig.tight_layout(rect=(0,.035,1,1)); fig.savefig(out/f'exp{exp}_{index:04d}.png',dpi=140); plt.close(fig)
