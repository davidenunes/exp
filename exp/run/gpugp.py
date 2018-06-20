""" Hyper parameter optimisation with multiple gpus and gaussian processes

"""
import sys

import os
from exp.params import ParamSpace
import argparse
import multiprocessing as mp
import GPUtil as gpu
from tqdm import tqdm
import importlib
import logging
import traceback
from bayes_opt import BayesianOptimization

from functools import partial


def init_gpu_worker(queue):
    global gpu_id
    gpu_id = queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def gpu_worker(runnable_path, kwargs):
    global gpu_id

    try:
        runnable_path = os.path.abspath(runnable_path)

        spec = importlib.util.spec_from_file_location("worker", location=runnable_path)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)

        return worker.run(**kwargs)
    except Exception as e:
        return e


def gpu_run(runnable, params, name="exp", ngpus=1, patience=3):
    """

    :param runnable: path to python module with run function
    :param params: path to parameter space file
    :param name: name for the experiment
    :param ngpus: number of gpus to be used in the experiment
        (defaults to minimum between ngpus and number of gpus not being used)
    :param cancel: if True cancels the entire experiment if one worker encounters an error
    :param repeat_cfg: a list of configuration ids to be run from the parameter space (nruns is applied)
    :param repeat_run: a list of run ids to be run from the parameter space
    :param nruns: number of runs to be executed for each unique configuration in the parameter space
    """

    # setup logger
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler('{name}.log'.format(name=name))
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # detecting available gpus with load < 0.1
    gpu_ids = [g.id for g in gpu.getGPUs() if g.load < 0.2]
    ngpus = min(ngpus, len(gpu_ids))

    if ngpus <= 0:
        print("No gpus available", file=sys.stderr)
        sys.exit(1)
    else:
        if not os.path.exists(params) and not os.path.isfile(params):
            print("params file not found: {}".format(params), file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(runnable) or not os.path.isfile(runnable):
            print("runnable file not found: {}".format(runnable), file=sys.stderr)
            sys.exit(1)

        ps = ParamSpace(filename=params)
        ps.write_grid_summary(name + '_params.csv')

        param_grid = ps.param_grid(include_id=True, id_param="id", nruns=True, runs=nruns)

        n_tasks = ps.grid_size * nruns

        if repeat_cfg is not None:
            param_grid = [p for p in param_grid if p["id"] in repeat_cfg]
            n_tasks = len(repeat_cfg) * nruns
        elif repeat_run is not None:
            param_grid = [p for p in param_grid if p["run_id"] in repeat_run]
            n_tasks = len(repeat_run)

        print("----------GPU DISPATCHER--------------------")
        print("::  GPUs: {}/{}".format(ngpus, len(gpu_ids)))
        print(":: tasks: {}".format(n_tasks))
        print("--------------------------------------------")

        manager = mp.Manager()
        psqueue = manager.Queue()
        # done = mp.Event()
        pbar = tqdm(total=n_tasks, leave=True)

        # load gpu ids to the queue to be read by each worker

        for i in range(ngpus):
            psqueue.put(gpu_ids[i])

        # pool = mp.Pool(processes=ngpus, initializer=init_gpu_worker, initargs=(psqueue,))

        successful = set()

        def worker_done(res, worker_id):
            """Executed on main process when worker result becomes available
            """
            if isinstance(res, Exception):
                try:
                    raise res
                except:
                    traceback.print_exc()
                    logger.error('Worker {} terminated with errors: '.format(worker_id), exc_info=True)

        current_patience = 0
        next_params = []
        while current_patience < patience:
            # 1. select param dicts to test
            processes = []
            for param_dict in next_params:
                p = mp.Process(target=gpu_worker(), args=(runnable, param_dict,))
                p.start()
                processes.append(p)

            for process in processes:
                process.join()

        # for param_dict in param_grid:
        #    run_id = param_dict["run_id"]
        #    id_callback = partial(worker_done, worker_id=run_id)
        #    pool.apply_async(gpu_worker, args=(runnable, param_dict,), callback=id_callback)

        # pool.close()
        # pool.join()

        pbar.write("DONE")
        if repeat_cfg is not None:
            all_ids = repeat_cfg
        else:
            all_ids = set(range(ps.grid_size))
        failed_tasks = all_ids.difference(successful)
        if len(failed_tasks) > 0:
            ids = " ".join(map(str, failed_tasks))
            fail_runs = "failed runs: {}".format(ids)
            pbar.write(fail_runs)
            logger.error(fail_runs)

        pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPU Runner')
    parser.add_argument('--runnable', metavar='runnable', type=str,
                        help='path for python module with run function')
    parser.add_argument('-p', '--params', metavar='params', type=str,
                        help='path for param space file')
    parser.add_argument('--name', metavar='name', type=str, default="exp", help='name for experiment')
    parser.add_argument('-g', '--gpus', metavar='gpus', type=int, default=1,
                        help='number of gpus to be used')

    parser.add_argument('-n', '--nruns', metavar='nruns', type=int, default=1, required=False,
                        help='number of gpus to be used')

    parser.add_argument('-c', '--cancel', nargs='?', metavar='cancel', const=True, required=False)

    parser.add_argument('--repeat-cfg', metavar='repeat_cfg', nargs='+', type=int, required=False)
    parser.add_argument('--repeat-run', metavar='repeat_run', nargs='+', type=int, required=False)

    args = parser.parse_args()
    repeat_cfg = set(args.repeat_cfg) if args.repeat_cfg is not None else None
    repeat_run = set(args.repeat_run) if args.repeat_run is not None else None
    n_runs = args.nruns if args.nruns is not None else 1
    cancel = args.cancel if args.cancel is not None else False
    gpu_run(runnable=args.runnable, params=args.params, name=args.name, ngpus=args.gpus, cancel=cancel,
            nruns=n_runs, repeat_cfg=repeat_cfg, repeat_run=repeat_run)
