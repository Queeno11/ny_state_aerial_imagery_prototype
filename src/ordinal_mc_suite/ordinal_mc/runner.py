"""Full-batch fits, honest validation, and transactional job-level resume."""
import hashlib, json, platform, sqlite3, time, copy, multiprocessing, signal
from pathlib import Path
import numpy as np
import torch
from .config import parse_args, runtime
from .core import StructuralDGP, Features, prepare, tensor, train_candidate, validation_error, test_metrics, seed_for, InadmissibleScoreError
from .design import make_design, level_target, PopulationDGP
from .experiments import jobs
from .report import export
from .execution import iter_results

# The uploaded serial release has identical statistical/training code. Only
# this exact source version is eligible for automatic execution-only migration.
LEGACY_SOURCE_SHA256 = {'25190dbdd3174ebf4f890cbfd4690ea056f78bf2b225b528e888aa6a97806f2e'}
LEGACY_SOURCE_SHA256.add('3f54aac44913e1dd9fe1a0130c62679a2840c151a9b37168d4251be65cd25c4a')

def scientific_config(d):
    return {k:v for k,v in d.items() if k!='workers'}

def compatible_signature(old, new):
    previous=copy.deepcopy(old)
    if 'config' in previous:  # real run signatures; --check uses bare toy signatures
        previous['config']=scientific_config(previous['config'])
    if previous.get('source_sha256') in LEGACY_SOURCE_SHA256:
        previous['source_sha256']=new['source_sha256']
    return previous==new

def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()

def source_hash():
    h=hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()

def open_database(out,signature,resume):
    out.mkdir(parents=True,exist_ok=True); path=out/'checkpoint.sqlite'
    exists=path.exists()
    if exists and not resume: raise ValueError(f'{path} already exists. Use --resume with the same configuration, or choose a new --out.')
    if resume and not exists: raise ValueError('--resume requested but no checkpoint exists')
    conn=sqlite3.connect(path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY,result TEXT NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS candidates (job TEXT NOT NULL, ordinal INTEGER NOT NULL,data TEXT NOT NULL,PRIMARY KEY(job,ordinal))')
    old=conn.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
    if old and old[0]!=canonical(signature):
        if not compatible_signature(json.loads(old[0]),signature):
            conn.close(); raise ValueError('Resume refused: config, selected experiments, source code, or numerical runtime changed. Use a new output directory.')
        with conn:
            conn.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',('prior_signature_'+digest(json.loads(old[0])),old[0]))
            conn.execute("UPDATE metadata SET value=? WHERE key='signature'",(canonical(signature),))
        print('Checkpoint verified: resuming with the multicore runner; scientific settings match.',flush=True)
    if not old:
        conn.execute('INSERT INTO metadata VALUES (?,?)',('signature',canonical(signature))); conn.commit()
    return conn

def fit_job(job,design,cfg):
    # Preprocessing uses fitting labels only, never auxiliary views or validation.
    features=Features(design['train']['x'],job['model'],cfg)
    data=prepare(design,features,cfg); extra={}
    if job['loss']=='squared':
        target,extra=level_target(job['method'],design,cfg,job['scenario']); data['target']=tensor(target,cfg)
    # Exclude unused auxiliary X when lambda=0: exact matched unlinked computations.
    all_candidates=[]; selected=None; selected_key=None
    base={k:v for k,v in job.items() if k not in ('scenario','weights')}; base.update(job['scenario'])
    for alpha in sorted(set(cfg.ridge_grid)):
        for lam in job['weights']:
            edges={arm:tensor(design['links'][arm] if weight else design['links']['none'],cfg,True) for arm,weight in zip(('temporal','cross_city'),lam)}
            working=data if any(lam) else {**data,'x':data['x'][:design['n_fit']]}
            best=None; records=[]
            for restart in range(cfg.restarts):
                seed=seed_for(cfg.seed,'fit',job['rep'],job['n'],job['cities'],job['truth'],job['model'],restart)
                candidate_start=time.perf_counter()
                try:
                    model,stats=train_candidate(working,edges,job['model'],job['loss'],alpha,lam,seed,cfg)
                except InadmissibleScoreError as error:
                    records.append({**base,'reg':alpha,'lambda_t':lam[0],'lambda_c':lam[1],
                        'restart':restart,'validation_misranking':None,'train_objective':None,
                        'seconds':time.perf_counter()-candidate_start,'optimizer_status':'inadmissible',
                        'failure_reason':str(error),'best_restart_for_hyperparameters':False,'selected':False})
                    continue
                val=validation_error(model,features,design,cfg)
                record={**base,'reg':alpha,'lambda_t':lam[0],'lambda_c':lam[1],'restart':restart,'validation_misranking':val,**stats,
                        'best_restart_for_hyperparameters':False,'selected':False}
                records.append(record)
                if best is None or stats['train_objective']<best[1]['train_objective']:
                    best=(model,stats,val,restart)
            if best is None:
                all_candidates.extend(records)
                continue
            model,stats,val,restart=best
            records[restart]['best_restart_for_hyperparameters']=True
            # Restart selected on training objective; hyperparameters on validation.
            key=(val,-alpha,sum(lam),lam)
            if selected_key is None or key<selected_key:
                selected_key=key; selected=(model,stats,val,alpha,lam,restart)
            all_candidates.extend(records)
    if selected is None:
        raise RuntimeError('No admissible restart in any hyperparameter candidate; job was not saved. Inspect model initialization or feature support.')
    model,stats,val,alpha,lam,restart=selected
    for r in all_candidates:
        r['selected']=(r['reg']==alpha and (r['lambda_t'],r['lambda_c'])==lam and r['restart']==restart)
    linked='both' if all(lam) else 'temporal' if lam[0] else 'cross_city' if lam[1] else 'none'
    rho=model.rho().detach().cpu().numpy()
    result={**base,**stats,**test_metrics(model,features,design,cfg),**extra,
      'reg':alpha,'lambda_t':lam[0],'lambda_c':lam[1],'restart':restart,'validation_misranking':val,
      'reg_at_upper_grid':alpha==max(cfg.ridge_grid),'lambda_t_at_upper_grid':lam[0]==max(w[0] for w in job['weights']),
      'lambda_c_at_upper_grid':lam[1]==max(w[1] for w in job['weights']),
      'rho_min':float(rho.min()) if len(rho) else None,'rho_max':float(rho.max()) if len(rho) else None,
      'noise_sd_effective':design['noise_sd_effective'],'calibrated_baseline_oracle_risk':design['baseline_oracle_risk'],
      'n_fit':design['n_fit'],'n_validation':design['n_val'],'n_training_cities':cfg.cities,
      'n_rank_pairs':len(design['pairs'][0]),'auxiliary_pool_size':len(design['x'])-design['n_fit'],
      'inadmissible_candidates':sum(r.get('optimizer_status')=='inadmissible' for r in all_candidates),
      'recovered_candidates':sum(r.get('optimizer_status')=='recovered' for r in all_candidates),
      **design['link_stats'][linked],
      **{'offered_'+k:v for k,v in design['link_stats'][job['arm']].items()},
      'total_candidate_seconds':sum(r['seconds'] for r in all_candidates)}
    return result,all_candidates

_worker_state=None

def initialize_worker(d):
    global _worker_state
    if multiprocessing.current_process().name!='MainProcess':
        signal.signal(signal.SIGINT,signal.SIG_IGN)
    cfg=runtime(d)
    _worker_state=dict(cfg=cfg,dgp=PopulationDGP(cfg),cache_key=None,design=None)

def compute_job(job):
    state=_worker_state
    cfg=state['cfg']; jobcfg=copy.copy(cfg)
    jobcfg.cities=job['cities']; jobcfg.fit_cell_effects=job['method']=='log_cell_fe'
    jobcfg.acquisition_strength=job['scenario']['acquisition_strength']; jobcfg.group_strength=job['scenario']['group_strength']
    key=(job['rep'],job['n'],job['cities'],job['truth'],canonical(job['scenario']))
    if key!=state['cache_key']:
        state['design']=make_design(jobcfg,state['dgp'],job['rep'],job['n'],job['truth'],job['scenario'])
        state['cache_key']=key
    return fit_job(job,state['design'],jobcfg)

def main():
    args,d,experiments=parse_args()
    if args.check:
        from .checks import run_checks
        run_checks(); return
    joblist=list(jobs(d,experiments)); count=sum(len(j['weights'])*len(set(d['ridge_grid']))*d['restarts'] for j in joblist)
    print(f'Planning {count:,} candidate fits; {len(joblist):,} selected fits. Device={d["device"]}, dtype={d["dtype"]}, workers={d["workers"]}, threads/worker={d["threads"]}.',flush=True)
    if args.plan:
        for e in experiments:
            jj=[j for j in joblist if j['experiment']==e]
            print(f'  Experiment {e}: {len(jj):,} jobs / {sum(len(j["weights"])*len(set(d["ridge_grid"]))*d["restarts"] for j in jj):,} candidates')
        return
    if d['reps']<20: print('Exploratory run: too few repetitions for reliable Monte Carlo uncertainty.',flush=True)
    runtime(d)
    signature=dict(config=scientific_config(d),experiments=experiments,source_sha256=source_hash(),python=platform.python_version(),numpy=np.__version__,torch=torch.__version__)
    conn=open_database(args.out,signature,args.resume)
    (args.out/'resolved_config.json').write_text(json.dumps(d,indent=2))
    (args.out/'run_metadata.json').write_text(json.dumps({**signature,'execution':dict(workers=d['workers'],threads_per_worker=d['threads'])},indent=2))
    done={x[0] for x in conn.execute('SELECT id FROM jobs')}; start=time.perf_counter()
    pending=[job for job in joblist if digest(job) not in done]
    completed=len(joblist)-len(pending)
    print(f'{completed:,} jobs already saved; {len(pending):,} remaining.',flush=True)
    results=None
    try:
        if pending:
            results=iter_results(compute_job,pending,workers=min(d['workers'],len(pending)),
                threads=d['threads'],initializer=initialize_worker,initargs=(d,))
        for job,(result,candidates) in results or ():
            jid=digest(job)
            with conn:
                conn.execute('INSERT INTO jobs VALUES (?,?)',(jid,canonical(result)))
                conn.executemany('INSERT INTO candidates VALUES (?,?,?)',[(jid,i,canonical(r)) for i,r in enumerate(candidates)])
            completed+=1
            print(f'[{completed}/{len(joblist)}] E{job["experiment"]} rep={job["rep"]} C={job["cities"]} n={job["n"]} {job["truth"]}/{job["model"]} {job["loss"]} {job["method"]} unseen={result["unseen_within_latent"]:.3f} reg={result["reg"]:g} lambda_t={result["lambda_t"]:g} lambda_c={result["lambda_c"]:g} audit_drop={result["audit_objective_drop"]:.2g}',flush=True)
    except BaseException:
        if results is not None: results.close()
        export(conn,args.out,plots=False); conn.close()
        print('Completed jobs saved. Resume with --resume; unfinished jobs are rerun. Worker count may change.',flush=True)
        raise
    diag=export(conn,args.out); conn.close(); diag['this_invocation_seconds']=time.perf_counter()-start
    print(json.dumps(diag,indent=2))

if __name__=='__main__': main()
