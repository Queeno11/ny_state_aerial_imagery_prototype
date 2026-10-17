"""Bounded, spawn-based process execution; no numerical-library dependencies."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import multiprocessing
import os


@contextmanager
def numerical_thread_limits(threads):
    """Spawned interpreters read these before importing NumPy/PyTorch."""
    names = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = str(threads)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def abort_pool(pool):
    """Stop active fits promptly on interruption or failure, also on Python 3.12."""
    terminate = getattr(pool, 'terminate_workers', None)
    if terminate is not None:
        terminate()
        return
    # Before Python 3.14 the executor has no public terminate API. Take a copy
    # before shutdown releases the handles; this is cleanup, not scheduling.
    processes = list((getattr(pool, '_processes', None) or {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    pool.shutdown(wait=True, cancel_futures=True)


def iter_results(function, tasks, *, workers=1, threads=1,
                 initializer=None, initargs=()):
    """Yield (task, result) as jobs finish, with at most 2*workers in flight.

    Only the caller receives results and writes checkpoints. Closing this
    generator terminates unfinished jobs; its caller must close it on errors.
    The worker function must seed numerical randomness from the job identity.
    """
    if workers < 1 or threads < 1:
        raise ValueError('workers and threads must be positive')
    tasks = iter(tasks)
    if workers == 1:
        if initializer is not None:
            initializer(*initargs)
        for task in tasks:
            yield task, function(task)
        return

    with numerical_thread_limits(threads):
        pool = ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=initializer, initargs=initargs)
        pending = {}
        complete = False
        try:
            def refill():
                while len(pending) < 2*workers:
                    try:
                        task = next(tasks)
                    except StopIteration:
                        break
                    pending[pool.submit(function, task)] = task

            refill()
            while pending:
                ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    task = pending.pop(future)
                    yield task, future.result()
                refill()
            complete = True
        finally:
            if complete:
                pool.shutdown(wait=True)
            else:
                for future in pending:
                    future.cancel()
                abort_pool(pool)
