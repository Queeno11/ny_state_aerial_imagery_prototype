"""Core budget for the paper Monte Carlo: workers (processes) x threads (numba prange + BLAS) <= cores.

The thread environment must be set before numpy/numba are imported, so the entry points call
`thread_env_from_argv(sys.argv)` at the very top. Children of a spawn pool inherit os.environ;
`apply_threads(n)` additionally sets numba's thread count inside each worker.
Note (hybrid CPUs, WSL): processes cannot be pinned to P-cores from inside WSL (virtual CPUs float
over the host's cores); Windows schedules the busiest threads on P-cores first, so a budget equal to
the number of P-cores is the closest approximation to "P-cores only".
"""
import os

THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def _arg(argv, name):
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + '='):
            return a.split('=', 1)[1]
    return None


def plan(workers, threads=None, cores=None):
    """(workers, threads, cores): threads defaults to cores // workers; workers x threads <= cores."""
    total = os.cpu_count() or 1
    cores = int(cores) if cores else total
    cores = max(1, min(cores, total))
    workers = max(1, min(int(workers), cores))
    if threads is None:
        threads = max(1, cores // workers)
    threads = max(1, int(threads))
    if workers * threads > cores:
        threads = max(1, cores // workers)
    return workers, threads, cores


def thread_env_from_argv(argv):
    """Set BLAS/OpenMP/numba thread env vars from --workers/--threads/--cores before numpy is imported."""
    w = _arg(argv, '--workers')
    t = _arg(argv, '--threads')
    c = _arg(argv, '--cores')
    _, threads, _ = plan(int(w) if w else 1, int(t) if t else None, int(c) if c else None)
    for v in THREAD_VARS:
        os.environ.setdefault(v, str(threads))
    os.environ.setdefault('NUMBA_NUM_THREADS', str(max(threads, 1)))
    return threads


def apply_threads(n):
    """Inside a worker: numba thread count (<= NUMBA_NUM_THREADS)."""
    try:
        import numba
        numba.set_num_threads(max(1, min(int(n), numba.config.NUMBA_NUM_THREADS)))
    except ImportError:
        pass
