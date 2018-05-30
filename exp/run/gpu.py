import sys

import os
from exp.params import ParamSpace
import argparse
import multiprocessing as mp
import GPUtil as gpu
from tqdm import tqdm
import importlib
import time

import traceback

from functools import partial


def init_gpu_worker(queue):
    global gpu_id
    gpu_id = queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def gpu_worker(runnable_path, kargs):
    global gpu_id

    args = dict(kargs)

    print("runnable: ", runnable_path)
    print("params: ", args)

    runnable_path = os.path.abspath(runnable_path)
    spec = importlib.util.spec_from_file_location("worker", location=runnable_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    # sys.modules["worker"] = module
    # worker = __import__(name="worker")
    try:
        worker.run(**args)
    except Exception as e:
        return e
    time.sleep(0.6)


def gpu_run(runnable, params, name="exp", ngpus=1, cancel=True, repeat=None):
    # detecting gpus available
    gpu_ids = [g.id for g in gpu.getGPUs()]
    ngpus = min(ngpus, len(gpu_ids))
    print("using {}/{} GPUs".format(ngpus, len(gpu_ids)))

    if ngpus <= 0:
        print("No gpus available", file=sys.stderr)
        sys.exit(1)
    else:
        if not os.path.exists(params) and not os.path.isfile(params):
            print("params file not found: {}".format(params), file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(runnable) and not os.path.isfile(runnable):
            print("runnable file not found: {}".format(runnable), file=sys.stderr)
            sys.exit(1)

        ps = ParamSpace(filename=params)
        ps.add_list("test", [1, 2, 3])
        ps.write_grid_summary(name + '_params.csv')
        param_grid = ps.param_grid(include_id=True, id_param="id")
        n_tasks = ps.grid_size

        if repeat is not None:
            param_grid = [p for p in param_grid if p["id"] in repeat]
            n_tasks = len(repeat)

        print("number of runs: ", n_tasks)

        manager = mp.Manager()
        psqueue = manager.Queue()
        pbar = tqdm(total=n_tasks)

        # load gpu ids to the queue to be read by each worker
        for i in range(ngpus):
            print("insering ", gpu_ids[i])
            psqueue.put(gpu_ids[i])

        pool = mp.Pool(processes=ngpus, initializer=init_gpu_worker, initargs=(psqueue,))

        successful = set()

        def worker_done(res, worker_id):
            """Executed on main process when worker result becomes available
            """
            if isinstance(res, Exception):
                print("worker terminated with errors: \n")
                try:
                    raise res
                except:
                    traceback.print_exc()
                if cancel:
                    pool.terminate()
            else:
                pbar.update()
                successful.add(worker_id)

        for param_dict in param_grid:
            id_callback = partial(worker_done, worker_id=param_dict["id"])
            pool.apply_async(gpu_worker, args=(runnable, param_dict,), callback=id_callback)
            # res[r] = arg_dict["id"]

        # print(mp.active_children())

        # print("main process:", mp.current_process())

        pool.close()
        pool.join()

        pbar.write("DONE")
        if repeat is not None:
            all_ids = repeat
        else:
            all_ids = set(range(ps.grid_size))
        failled_ids = all_ids.difference(successful)
        if len(failled_ids) > 0:
            pbar.write("failed runs: {}".format(failled_ids))

        pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPU Runner')
    parser.add_argument('-r', '--runnable', metavar='runnable', type=str,
                        help='path for python module with run function')
    parser.add_argument('-p', '--params', metavar='params', type=str,
                        help='path for param space file')
    parser.add_argument('--name', metavar='name', type=str, default="exp", help='name for experiment')
    parser.add_argument('-g', '--gpus', metavar='gpus', type=int, default=1,
                        help='number of gpus to be used')

    parser.add_argument('-c', '--cancel', nargs='?', metavar='cancel', const=True, required=False)

    parser.add_argument('--repeat', metavar='repeat', nargs='+', type=int, required=False)

    args = parser.parse_args()
    print(args.repeat)
    repeat = set(args.repeat) if args.repeat is not None else None
    cancel = args.cancel if args.cancel is not None else False
    gpu_run(runnable=args.runnable, params=args.params, name=args.name, ngpus=args.gpus, cancel=cancel,
            repeat=repeat)
