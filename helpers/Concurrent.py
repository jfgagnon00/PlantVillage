from itertools import islice
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait

def parallel_for(iterables,
                 task_fn,
                 *task_args,
                 task_completed=None,
                 max_workers=None,
                 executor=None,
                 chunk_size=None):
    """
    Parallelise l'appel de task_fn sur chaque element de iterables.

    iterables:
        elements sur lesquels task_fn va etre appele

    *task_args:
        arguments passes a task_fn en plus de l'iterable courant

    task_completed:
        callback appele apres que chaque appel a task_fn ait termine

    max_workers:
        configure le nombre de thread a utiliser

    executor:
        configuration custom pour les threads/process

    chunk_size:
        au lieu de faire l'appel a task_fn pour chaque iterable, decoupe
        iterables en chunk_size elements. task_fn sera reponsable de faire
        la 'sous iteration'
    """
    exec = ThreadPoolExecutor(max_workers=max_workers) if executor is None else executor

    def chunkify_iterables():
        if chunk_size is None:
            yield from iterables
        else:
            it = iter(iterables)
            yield from iter(lambda: list(islice(it, chunk_size)), [])

    def generate_futures():
        if task_args is None or len(task_args) == 0:
            for i in chunkify_iterables():
                yield exec.submit(task_fn, i)
        else:
            for i in chunkify_iterables():
                yield exec.submit(task_fn, *task_args, i)

    futures = [f for f in generate_futures()]

    if task_completed is None:
        wait(futures)
    else:
        for f in as_completed(futures):
            task_completed(f.result())

    if executor is None:
        exec.shutdown()

def create_thread_pool_executor(max_workers=None):
    return ThreadPoolExecutor(max_workers=max_workers)

def create_process_pool_executor(max_workers=None):
    return ProcessPoolExecutor(max_workers=max_workers)
