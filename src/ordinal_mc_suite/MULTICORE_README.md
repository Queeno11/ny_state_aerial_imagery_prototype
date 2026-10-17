# Multicore update

Use `--workers 4 --threads 1` to run four independent CPU fits concurrently.
The default remains one worker. `--threads` is the number of numerical threads
per worker, not the number of independent fits. The parallel scheduler uses
spawned processes on Linux, WSL and Windows. CUDA requires `--workers 1`.

For the complete presentation battery, activate `torch_geo_env`, then run:

```bash
bash run_overnight_multicore.sh 4
```

Start with four workers. You can also use `bash run_overnight_multicore.sh 6`
or set `WORKERS=6`. More workers require more RAM and do not guarantee linear
speedups. Each worker constructs its own small DGP and keeps one design cache.
NumPy/BLAS and PyTorch thread counts are limited per worker to avoid
oversubscribing the machine. Candidate grids and restarts inside one fit stay
serial; independent selected-fit jobs run in parallel. The three experiments
run sequentially so they share the chosen CPU budget.

Stop any currently running serial batch with Ctrl+C before updating the code
or launching this runner against the same output directories. Extract this
archive into the existing suite folder, overwriting source files and the
three included overnight configs. It does not contain old result folders.
The new launcher automatically adds --resume when a matching checkpoint
exists. It retains completed jobs and reruns only jobs not yet saved. The
parent process alone writes SQLite and exports result CSVs. Progress lines
can arrive in a different job order; the estimator, data seeds, losses and
selection rules are unchanged.

Worker count is an execution setting and may change on resume. Other config
changes, including threads, dtype, device, training budget and seeds, remain
guarded. Checkpoints from the exact serial release uploaded in this conversation
(source SHA256 25190dbdd3174ebf4f890cbfd4690ea056f78bf2b225b528e888aa6a97806f2e)
are accepted after verifying all scientific/runtime settings. Their original
signature is retained in checkpoint metadata. Unknown source versions or
incompatible settings are refused rather than mixed.

For an individual run, append `--workers 4 --threads 1` to its existing command.
To resume it, also append `--resume`. Do not use a second terminal to write to
the same output folder while another runner is active.

The bundled configs retain the previously supplied overnight battery:
measurement (E1, linear), flexible temporal comparison (E3, small NN), and
imperfect links (E4, linear), each with 30 replications and fresh simulation
seed 20261015. Existing completed experiments are not otherwise rerun.

## Verification

Executed here: real spawned-process concurrency; deterministic seeded task
outputs; bounded scheduling; NumPy/BLAS environment limits; worker exceptions
and abrupt crashes; cleanup on cancellation; and real runner/SQLite/CSV
integration using a deterministic fitting fixture. The latter checks serial
and parallel records, partial-run recovery, no duplicate jobs/candidates,
worker-count changes, guarded serial-checkpoint migration and rejection of
incompatible resumes. All passed. Python and shell syntax checks also passed.

This environment has no PyTorch, so actual numerical fits could not be tested
here. An optional real-PyTorch check is included:

```bash
python verification/check_multicore.py
```

It runs tiny linear/NN fits serially and with two workers, compares fit and
candidate records excluding elapsed times, and verifies a worker-count change
on resume. It uses a temporary directory and never touches existing results.

The existing serial README and verification reports below describe the
original numerical implementation; they do not certify this process update.

