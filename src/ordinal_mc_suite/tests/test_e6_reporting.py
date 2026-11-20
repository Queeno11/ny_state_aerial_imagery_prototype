"""Reporting regressions; no PyTorch dependency or model fitting."""
import json
from pathlib import Path
import sys
import sqlite3
import tempfile
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ordinal_mc.e6.report import export, lead_figure, METRICS
from render_e6 import render_saved


def fixture():
    job=dict(rep=1,cities=8,truth='quadratic',model='nn')
    metrics=[];strata=[]
    for arm,offset in [('A',2.),('B',2.),('C',1.),('D',3.),('D_common',3.)]:
        for where in ('seen','unseen'):
            common=dict(**job,map='affine',head='affine',arm=arm,where=where,strength=.4)
            metrics.append(dict(**common,**{k:offset for k in METRICS},best_step=0,
                                zero_outcome_change_rmse=4.,zero_mean_change_rmse=3.))
            for direction,sign in [('increase',1),('decline',-1)]:
                strata.append(dict(**common,direction=direction,size='all',true_mean=sign*.4,
                                   predicted_mean=sign*.35,rmse=.05,sensitivity=.9))
    result=dict(job=job,metrics=metrics,strata=strata,selection=[],history=[],checks={},design={})
    d=dict(cities=[8],cases=[['quadratic','nn']],maps=['affine'],heads=['affine'],replications=1,change_strengths=[.4])
    return result,d


class ReportingTests(unittest.TestCase):
    def test_curves_and_comparison_rows_are_populated(self):
        result,d=fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            out=Path(folder);export([result],out,d,1)
            audit=json.loads((out/'figure_checks.json').read_text())
            self.assertEqual(audit,dict(series=2,points=4))
            report=(out/'report.md').read_text(encoding='utf-8')
            self.assertIn('| affine | affine | Latent magnitudes | C-B | -1.0000',report)
            self.assertIn('| affine | affine | Reported outcome changes | C-D | -2.0000',report)
            self.assertIn('single-replication smoke run',report)
            self.assertIn('C retained its initial scorer/head checkpoint in 1/1',report)
            self.assertNotIn('Configured replications are incomplete',report)

    def test_missing_plot_data_cannot_silently_create_blank_figure(self):
        result,d=fixture();s=pd.DataFrame(result['strata']);s=s[s['where']=='seen']
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            with self.assertRaisesRegex(ValueError,'No unseen-city'):
                lead_figure(s,Path(folder),d)

    def test_saved_result_renderer_keeps_original_ledger(self):
        result,d=fixture()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            run=Path(folder)/'original';run.mkdir();out=Path(folder)/'review'
            (run/'manifest.json').write_text(json.dumps(dict(experiment='E6',config=d,signature='fixture')),encoding='utf-8')
            con=sqlite3.connect(run/'results.sqlite')
            con.execute('CREATE TABLE results (job TEXT PRIMARY KEY,payload TEXT NOT NULL)')
            con.execute('INSERT INTO results VALUES (?,?)',('fixture',json.dumps(result)));con.commit();con.close()
            audit=render_saved(run,out)
            self.assertFalse(audit['original_fitting_results_modified'])
            self.assertEqual(audit['completed_jobs'],1)
            self.assertFalse((run/'report.md').exists())
            self.assertTrue((out/'report.md').exists())
            with self.assertRaises(ValueError): render_saved(run,out)


if __name__=='__main__': unittest.main()
