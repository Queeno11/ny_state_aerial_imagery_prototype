"""Matched-tract change diagnostics; calibration never uses changed test tracts."""
import numpy as np
from scipy.stats import rankdata

CHANGE_METRICS = ['change_rmse', 'change_bias', 'changed_rmse', 'stable_rmse',
                  'change_slope', 'change_direction_accuracy', 'change_recall',
                  'change_precision', 'change_false_positive_rate', 'change_auc',
                  'change_threshold', 'change_true_share', 'change_true_rms',
                  'change_increase_share', 'change_decrease_share',
                  'increase_recall', 'decrease_recall', 'increase_direction_accuracy', 'decrease_direction_accuracy']
HIGHER_BETTER = {'change_direction_accuracy', 'change_recall', 'change_precision', 'change_auc',
                 'increase_recall', 'decrease_recall', 'increase_direction_accuracy', 'decrease_direction_accuracy'}


def change_metrics(predicted, actual, threshold, truth_threshold):
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    changed = np.abs(actual) > truth_threshold
    stable = ~changed
    detected = np.abs(predicted) > threshold
    err = predicted-actual
    def mean(x):
        return float(np.mean(x)) if len(x) else None
    def rmse(mask):
        return float(np.sqrt(np.mean(err[mask]**2))) if mask.any() else None
    def auc():
        n1, n0 = int(changed.sum()), int(stable.sum())
        if not n1 or not n0: return None
        ranks = rankdata(np.abs(predicted))
        return float((ranks[changed].sum()-n1*(n1+1)/2)/(n1*n0))
    cov = np.mean((actual-actual.mean())*(predicted-predicted.mean()))
    var = np.var(actual)
    return dict(change_rmse=rmse(np.ones(len(actual), bool)), change_bias=mean(err),
                changed_rmse=rmse(changed), stable_rmse=rmse(stable),
                change_slope=float(cov/var) if var else None,
                change_direction_accuracy=mean((predicted[changed]*actual[changed]>0).astype(float)+.5*(predicted[changed]==0)),
                change_recall=mean(detected[changed]),
                change_precision=mean(changed[detected]),
                change_false_positive_rate=mean(detected[stable]), change_auc=auc(),
                change_threshold=float(threshold), change_true_share=mean(changed),
                change_true_rms=float(np.sqrt(np.mean(actual**2))),
                change_increase_share=mean(actual>truth_threshold),
                change_decrease_share=mean(actual < -truth_threshold),
                increase_recall=mean(detected[actual>truth_threshold]),
                decrease_recall=mean(detected[actual < -truth_threshold]),
                increase_direction_accuracy=mean((predicted[actual>truth_threshold]>0).astype(float)+.5*(predicted[actual>truth_threshold]==0)),
                decrease_direction_accuracy=mean((predicted[actual < -truth_threshold]<0).astype(float)+.5*(predicted[actual < -truth_threshold]==0)))


def evaluate_changes(predict, design, cfg, scale):
    # A separate stable calibration panel supplies a fixed false-positive target.
    # No test truths, test predictions or changed calibration labels set the cutoff.
    cal = design['change_calibration']
    delta = (predict(cal['x_after'])-predict(cal['x_before']))/scale
    threshold = np.quantile(np.abs(delta), 1-cfg.experiment5['false_positive_target'], method='higher')
    result = {}
    for name, panel in design['change_tests'].items():
        predicted = (predict(panel['x_after'])-predict(panel['x_before']))/scale
        metrics = change_metrics(predicted, panel['delta_f'], threshold, cfg.experiment5['truth_threshold'])
        result.update({name+'_'+k: v for k, v in metrics.items()})
        result[name+'_change_n'] = len(predicted)
    return result
