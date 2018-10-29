import sys

import GPUtil

import importlib
import logging
import multiprocessing as mp
import os
import traceback
import click
from toml import TomlDecodeError
from tqdm import tqdm
from multiprocessing import Queue, Event, Process
from multiprocessing.queues import Empty as QueueEmpty
from exp.params import ParamSpace, ParamDecodeError


def load_module(runnable_path):
    """ Loads a python file with the module to be evaluated.

    The module is a python file with a run function member. Each worker then
    passes each configuration it receives from a Queue to run as keyword arguments

    Args:
        runnable_path:

    Raises:
        TypeError: if the loaded module doesn't have a run function

    Returns:
        a reference to the newly loaded module so

    """
    runnable_path = os.path.abspath(runnable_path)
    spec = importlib.util.spec_from_file_location("runnable", location=runnable_path)
    runnable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runnable)
    try:
        getattr(runnable, "run")
    except AttributeError:
        raise TypeError("module in {} does not contain a \"run\" method".format(runnable_path))

    return runnable


def worker(pid: int,
           module_path: str,
           config_queue: Queue,
           result_queue: Queue,
           error_queue: Queue,
           terminated: QueueEmpty,
           cancel):
    """ Worker to be executed in its own process

    Args:
        cancel: if true terminates when an error is encountered, otherwise keeps running
        error_queue: used to pass formatted stack traces to the main process
        module_path: path to model runnable that is imported. It's method run is called on a given configuration
        terminated: each worker should have its own flag
        pid: (int) with worker id
        config_queue: configuration queue used to receive the parameters for this worker, each configuration is a task
        result_queue: queue where the worker deposits the results

    Returns:
        each time a new result is returned from calling the run function on the module, the worker puts this in to its
        result multiprocessing Queue in the form (worker_id, configuration_id, result)

        If an exception occurs during run(...) the worker puts that exception as the result into the queue instead

    """

    os.environ["CUDA_VISIBLE_DEVICES"] = str(pid)
    module = load_module(module_path)

    while not terminated.is_set():
        try:
            kwargs = config_queue.get(timeout=0.5)
            cfg_id = kwargs["id"]
            kwargs["pid"] = pid
            result = module.run(**kwargs)
            result_queue.put((pid, cfg_id, result))
        except QueueEmpty:
            pass
        except Exception as e:
            if cancel:
                terminated.set()
            error_queue.put((pid, cfg_id, traceback.format_exc()))
            result_queue.put((pid, cfg_id, e))


@click.command(help='runs all the configurations in a defined space')
@click.option('-p', '--params', required=True, type=click.Path(exists=True), help='path to parameter space file')
@click.option('-m', '--module', required=True, type=click.Path(exists=True), help='path to python module file')
@click.option('-r', '--runs', default=1, type=int, help='number of configuration runs')
@click.option('--name', default="exp", type=str, help='experiment name: used as prefix for some output files')
@click.option('-w', '--workers', default=1, type=int, help="number of workers: limited to CPU core count or GPU "
                                                           "count, cannot be <=0.")
@click.option('-g', '--gpu', is_flag=True,
              help="bounds the number of workers to the number of available GPUs (not under load)."
                   "Each process only sees a single GPU.")
@click.option('-c', '--config-ids', type=int, multiple=True)
@click.option('--cancel', is_flag=True, help="cancel all tasks if one fails")
def main(params, module, runs, name, workers, gpu, config_ids, cancel):
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler('{name}.log'.format(name=name), delay=True)
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    try:
        if gpu:
            # detecting available gpus with load < 0.1
            worker_ids = [g.id for g in GPUtil.getGPUs() if g.load < 0.2]
            num_workers = min(workers, len(worker_ids))

            if num_workers <= 0:
                logger.log(logging.ERROR, "no gpus available")
                sys.exit(1)
        else:
            num_workers = min(workers, mp.cpu_count())
            if num_workers <= 0:
                logger.log(logging.ERROR, "--workers cannot be 0")
                sys.exit(1)

        ps = ParamSpace(filename=params)
        ps.write_configs('{}_params.csv'.format(name))

        param_grid = ps.param_grid(runs=runs)
        n_tasks = ps.size * runs

        if len(config_ids) > 0:
            n_tasks = len(config_ids) * runs
            param_grid = [p for p in param_grid if p["id"] in config_ids]
            param_grid = iter(param_grid)

        num_workers = min(n_tasks, num_workers)

        print("----------Parameter Space Runner------------")
        print(":: tasks: {}".format(n_tasks))
        print(":: workers: {}".format(num_workers))
        print("--------------------------------------------")

        config_queue = Queue()
        result_queue = Queue()
        error_queue = Queue()
        progress_bar = tqdm(total=n_tasks, leave=True)

        terminate_flags = [Event() for _ in range(num_workers)]
        processes = [
            Process(target=worker,
                    args=(i, module, config_queue, result_queue, error_queue, terminate_flags[i], cancel))
            for i in range(num_workers)]

        scores = {}
        configs = {}

        # submit num worker jobs
        for _ in range(num_workers):
            next_cfg = next(param_grid)
            configs[next_cfg["id"]] = next_cfg
            config_queue.put(next_cfg)

        for p in processes:
            p.daemon = True
            p.start()

        num_completed = 0
        pending = num_workers
        done = False
        successful = set()

        while num_completed < n_tasks and not done:
            try:
                res = result_queue.get(timeout=1)
                pid, cfg_id, result = res
                if not isinstance(result, Exception):
                    successful.add(cfg_id)
                    # cfg = configs[cfg_id]
                    scores[cfg_id] = result
                    num_completed += 1
                    pending -= 1

                    if (num_completed + pending) != n_tasks:
                        next_cfg = next(param_grid)
                        configs[next_cfg["id"]] = next_cfg
                        config_queue.put(next_cfg)

                        pending += 1
                    else:
                        # signal the current worker for termination no more work to be done
                        terminate_flags[pid].set()

                    progress_bar.update()
                else:
                    # retrieve one error from queue, might not be exactly the one that failed
                    # since other worker can write to the queue, but we will have at least one error to retrieve
                    _, cfg_id_err, err = error_queue.get()
                    logger.error("configuration {} failed".format(cfg_id_err))
                    logger.error(err)

                    if cancel:
                        done = True
                    else:
                        num_completed += 1
                        pending -= 1

                        if (num_completed + pending) != n_tasks:
                            next_cfg = next(param_grid)
                            configs[next_cfg["id"]] = next_cfg
                            config_queue.put(next_cfg)
                            pending += 1
                        else:
                            # signal the current worker for termination no more work to be done
                            terminate_flags[pid].set()
                        progress_bar.update()

            except QueueEmpty:
                pass
            # try to wait for process termination
        for process in processes:
            process.join(timeout=0.5)

            if process.is_alive():
                process.terminate()

        if len(config_ids) > 0:
            all_ids = set(config_ids)
        else:
            all_ids = set(range(ps.size))
        failed_tasks = all_ids.difference(successful)
        if len(failed_tasks) > 0:
            ids = " ".join(map(str, failed_tasks))
            fail_runs = "failed runs: {}".format(ids)
            print(fail_runs, file=sys.stderr)
            logger.warn(fail_runs)

        progress_bar.close()

    except TomlDecodeError as e:
        logger.error(traceback.format_exc())
        print("\n\n[Invalid parameter file] TOML decode error:\n {}".format(e), file=sys.stderr)
    except ParamDecodeError as e:
        logger.error(traceback.format_exc())
        print("\n\n[Invalid parameter file]\n {}".format(e), file=sys.stderr)


if __name__ == '__main__':
    main()
