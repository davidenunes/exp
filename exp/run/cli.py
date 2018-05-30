import sys
import plac
from plac import Annotation
import os
from exp.params import ParamSpace
import multiprocessing as mp
import GPUtil as gpu
from tqdm import tqdm
import importlib
import time
import logging
import traceback
from functools import partial
from multiprocessing.pool import ApplyResult


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
        print("worker terminated with errors: \n")
        traceback.print_exc()
        print()
        return e
    time.sleep(0.5)


class RunCli:
    commands = ['gpus']

    @plac.annotations(
        runnable=Annotation("path to python script with a run method that receives params"),
        params=Annotation("path to param space file"),
        name=Annotation("name for this run", "option"),
        ngpus=Annotation("number of gpus to be used", "option", type=int),
        cancel=Annotation("cancel all the jobs if error is found", "flag", "c")
    )
    def gpus(self, runnable, params, name="exp", ngpus=1, cancel=False):
        print("name ", name)
        print("params ", params)
        print("runnable ", runnable)

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

            print("param grid size: ", ps.grid_size)

            manager = mp.Manager()
            psqueue = manager.Queue()
            pbar = tqdm(total=ps.grid_size)

            # load gpu ids to the queue to be read by each worker
            for i in range(ngpus):
                print("insering ", gpu_ids[i])
                psqueue.put(gpu_ids[i])

            pool = mp.Pool(processes=ngpus, initializer=init_gpu_worker, initargs=(psqueue,))

            successful = set()

            def worker_done(res, id):
                """Executed on main process when worker result becomes available
                """
                pbar.update()
                print(type(res))
                if isinstance(res, Exception):
                    if cancel:
                        for worker in mp.active_children():
                            worker.terminate()
                else:
                    successful.add()

            for param_dict in param_grid:
                # arg_dict = manager.dict(param_dict)
                # print(arg_dict)
                arg_dict = param_dict
                id_callback = partial(worker_done, id=arg_dict["id"])
                pool.apply_async(gpu_worker, args=(runnable, arg_dict,), callback=id_callback)
                # res[r] = arg_dict["id"]

            # print(mp.active_children())

            print("main process:", mp.current_process())

            pool.close()
            pool.join()
            pbar.close()
            print("done")

    def __missing__(self, name):
        print("command not found {cmd}".format(cmd=name))


if __name__ == '__main__':
    plac.Interpreter.call(RunCli)
