"""Post-selection targets: latent change, detection, and observed outcome scale."""
import numpy as np
import torch
from .. import core
from .data import cell_parameters, conditional_mean


def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2))) if len(x) else None


def mean(x):
    return float(np.mean(x)) if len(x) else None


def dispersion(score,design):
    i,j=design['pairs']
    value=float(.5*np.sum(design['pair_weights']*(score[j]-score[i])**2))
    if not np.isfinite(value) or value<=0: raise ValueError('Invalid E5 fitting dispersion')
    return value


def head_predict(head,s,cell):
    with torch.no_grad():
        return head(torch.as_tensor(s,dtype=torch.float64),torch.as_tensor(cell,dtype=torch.long)).cpu().numpy()


def evaluate(predict,head,design,d,cfg,kind,*,oracle_head=False):
    """All true quantities enter only here, after every checkpoint is finalized."""
    sd=np.sqrt(dispersion(predict(design['train']['x']),design))
    true_sd=np.sqrt(dispersion(design['train']['f'],design))
    stable=design['change_calibration']
    calibration=(predict(stable['x_after'])-predict(stable['x_before']))/sd
    cutoff=float(np.quantile(abs(calibration),1-d['false_positive_target'],method='higher'))
    rows=[];strata=[];arrays={}
    for strength,panels in design['e6_panels'].items():
        for where,p in panels.items():
            s0=predict(p['x_before']);s1=predict(p['x_after']);delta=(s1-s0)/sd;truth=p['delta_f']
            changed=abs(truth)>d['truth_threshold']
            stable_mask=~p['assigned_changed'] if strength>0 else np.ones(len(truth),bool)
            detected=abs(delta)>cutoff
            test,pairs=design['tests'][where];test_score=predict(test['x']);i,j=pairs['within']
            row=dict(strength=float(strength),where=where,n_changed=int(changed.sum()),
                change_rmse=rmse((delta-truth)[changed]),
                sign_accuracy=mean(((delta[changed]*truth[changed])>0).astype(float)+.5*(delta[changed]==0)),
                predicted_change_mean=mean(delta[changed]),true_change_mean=mean(truth[changed]),
                stable_rms=rmse(delta[stable_mask]),cutoff=cutoff,
                false_positive_rate=mean(detected[stable_mask]),sensitivity=mean(detected[changed]),
                score_sd_fit=sd,true_f_sd_fit=true_sd,
                normalization_only_rmse=rmse((truth/true_sd-truth)[changed]),
                within_latent_accuracy=1-core.misranking(test_score[j]-test_score[i],test['f'][j]-test['f'][i]),
                level_rmse=None,level_mean_rmse=None,outcome_change_rmse=None,mean_change_rmse=None,
                zero_outcome_change_rmse=None,zero_mean_change_rmse=None,
                outcome_calibration='unavailable_zero_shot' if where=='unseen' else 'no_head')
            if where=='seen':
                observed=p['w_after']-p['w_before'];mu=p['mu_after']-p['mu_before']
                row.update(zero_outcome_change_rmse=rmse(observed),zero_mean_change_rmse=rmse(mu))
                if head is not None or oracle_head:
                    fitted=[]
                    for endpoint,s in [('before',s0),('after',s1)]:
                        if oracle_head:
                            a,b,k=cell_parameters(p['contexts'],p['c'],p['t_'+endpoint],d)
                            value=conditional_mean(s,a,b,k,kind,design['noise_sd_effective'],d['quadrature_nodes'])
                        else: value=head_predict(head,s,p['cell_'+endpoint])
                        fitted.append(value)
                    row.update(level_rmse=rmse(np.r_[fitted[0]-p['w_before'],fitted[1]-p['w_after']]),
                        level_mean_rmse=rmse(np.r_[fitted[0]-p['mu_before'],fitted[1]-p['mu_after']]),
                        outcome_change_rmse=rmse(fitted[1]-fitted[0]-observed),
                        mean_change_rmse=rmse(fitted[1]-fitted[0]-mu),
                        outcome_calibration='oracle_measurement_mean' if oracle_head else 'fitted_cell_head')
            rows.append(row)
            edges=d['actual_size_edges'];bin_index=np.searchsorted(edges,abs(truth),side='right')-1
            for direction,sgn in [('decline',-1),('increase',1)]:
                groups=[('all',changed&(np.sign(truth)==sgn))]
                groups += [(str(b),changed&(np.sign(truth)==sgn)&(bin_index==b)) for b in range(len(edges))]
                for size,mask in groups:
                    strata.append(dict(strength=float(strength),where=where,direction=direction,size=size,
                        n=int(mask.sum()),true_mean=mean(truth[mask]),predicted_mean=mean(delta[mask]),
                        rmse=rmse((delta-truth)[mask]),sensitivity=mean(detected[mask])))
            tag=f'{strength:g}_{where}'
            arrays[tag+'_predicted']=delta;arrays[tag+'_true']=truth;arrays[tag+'_assigned_changed']=p['assigned_changed']
    return rows,strata,arrays
