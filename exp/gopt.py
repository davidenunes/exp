""" CLI for Hyper parameter optimisation

prompt_toolkit run sequential model algorithmic optimisation
based on gaussian processes, random forests etc.
"""
import sys

import GPUtil
import click
import csv
import importlib
import logging
import multiprocessing as mp
import os
from matplotlib import pyplot as plt
from multiprocessing import Event
from multiprocessing import Queue, Process
from multiprocessing.queues import Empty
from skopt import Optimizer, plots
from skopt.space import Integer, Real, Categorical
from tqdm import tqdm
from tqdm._utils import _term_move_up
import traceback
from toml import TomlDecodeError
from exp.params import ParamSpace, DTypes, ParamDecodeError


def params_to_skopt(param_space: ParamSpace):
    """ Converts a parameter space to a list of Dimention objects that can be used with
    a skopt Optimizer.

    A skopt Optimizer only receives 3 types of Dimensions: Categorical, Real, or Integer
    we convert parameters from our parameter space into one of those 3 types. Note that we only
    convert parameters that have either bounds or with a categorical domain with more than 1 value.
    If we have constant values in our parameter space, these don't need to be optimized anyway.

    Another function is provided to convert skopt output values back into a dictionary with
    a full configuration according to the parameter space (@see values_to_params).

    Args:
        param_space: a ParameterSpace where we can get the domain of each parameter

    Returns:
        a list of Dimension that can be passed to a skopt Optimizer

    """
    dimensions = []
    for param_name in param_space.param_names():
        domain_param = param_space.domain(param_name)
        domain = domain_param["domain"]
        dtype = DTypes.from_type(domain_param["dtype"])
        if len(domain) > 1:
            if dtype == DTypes.INT:
                low = min(domain)
                high = max(domain)
                dimensions.append(Integer(low, high, name=param_name))
            elif dtype == DTypes.FLOAT:
                low = min(domain)
                high = max(domain)
                prior = domain_param.get("prior", None)
                dimensions.append(Real(low, high, prior=prior, name=param_name))
            elif dtype == DTypes.CATEGORICAL:
                prior = domain_param.get("prior", None)
                dimensions.append(Categorical(domain, prior, transform="onehot", name=param_name))
    return dimensions


def values_to_params(param_values, param_space):
    """
    We don't need to optimize parameters with a single value in their domain so we filter
    them out with params to skopt and convert back to configuration using this method

    Args:
        param_values: dict {param_name: value}
        param_space: rest of the parameter space used to complete this configuration

    """
    cfg = {}
    for param_name in param_space.param_names():
        if param_name in param_values:
            cfg[param_name] = param_values[param_name]
        else:
            # complete with value from parameter space
            value = param_space.get_param(param_name)

            if isinstance(value, (tuple, list)) and len(value) > 1:
                raise ValueError("don't know how to complete a configuration that might have multiple values")
            else:
                if isinstance(value, (tuple, list)):
                    value = value[0]
                cfg[param_name] = value

    return cfg


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


def submit(n, optimizer: Optimizer, opt_param_names, current_configs, param_space: ParamSpace, queue: Queue):
    """ Generate and submit n new configurations to a queue.

    Asks the optimizer for n new values to explore, creates configurations for those points and puts them
    in the given queue.

    Args:
        n: the number of configurations to be generated
        optimizer: the optimiser object from skopt with the model used for the suggested points to explore
        opt_param_names: the names for the parameters using the same order of the dimensions in the optimizer
        current_configs: current list of configurations (updated with the newly generated ones)
        param_space: parameter space which we can use to convert optimizer points to fully specified configurations
        queue: que multiprocessing queue in which we put the new configurations
    """
    dims = opt_param_names
    xs = optimizer.ask(n_points=n)
    cfgs = [values_to_params(dict(zip(dims, x)), param_space) for x in xs]
    for i, c in enumerate(cfgs):
        c["id"] = i + len(current_configs)
        queue.put(c)
    current_configs += cfgs


def worker(id: int,
           module_path: str,
           config_queue: Queue,
           result_queue: Queue,
           error_queue: Queue,
           terminated: Event):
    """ Worker to be executed in its own process

    Args:
        module_path: path to model runnable that must be imported and runned on a given configuration
        terminated: each worker should have its own flag
        id: (int) with worker id
        config_queue: configuration queue used to receive the parameters for this worker, each configuration is a task
        result_queue: queue where the worker deposits the results

    Returns:
        each time a new result is returned from calling the run function on the module, the worker puts this in to its
        result multiprocessing Queue in the form (worker_id, configuration_id, result)

        If an exception occurs during run(...) the worker puts that exception as the result into the queue instead

    """
    # ids should be 0, 1, 2, n where n is the maximum number of gpus available
    os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
    module = load_module(module_path)

    while not terminated.is_set():
        try:
            kwargs = config_queue.get(timeout=0.5)
            cfg_id = kwargs["id"]
            # e.g. model score
            result = module.run(**kwargs)
            result_queue.put((id, cfg_id, result))
        except Empty:
            # nothing to do, check if it's time to terminate
            pass
        except Exception as e:
            terminated.set()
            error_queue.put((id, cfg_id, traceback.format_exc()))
            result_queue.put((id, cfg_id, e))


def update_progress_kuma(progress):
    tqdm.write(_term_move_up() * 10)  # move cursor up
    offset = " " * int(progress.n / progress.total * (progress.ncols - 40))

    tqdm.write(offset + '    _______________')
    tqdm.write(offset + '   |               |')
    tqdm.write(offset + '   |  KUMA-SAN IS  |')
    tqdm.write(offset + '   |  OPTIMIZING!  |')
    tqdm.write(offset + '   |   {:>3}/{:<3}     |'.format(progress.n, progress.total))
    tqdm.write(offset + '   |＿＿＿_＿＿＿＿|')
    tqdm.write(offset + ' ( )  ( )||')
    tqdm.write(offset + ' ( •(ｴ)•)|| ')
    tqdm.write(offset + ' / 　    づ')


@click.command(help='optimizes the hyperparameters for a given function',
               context_settings=dict(help_option_names=['-h', '--help'])
               )
@click.option('-p', '--params', required=True, type=click.Path(exists=True), help='path to parameter space file')
@click.option('-m', '--module', required=True, type=click.Path(exists=True), help='path to python module file')
@click.option('-w', '--workers', default=1, type=int, help="number of workers: limited to CPU core count or GPU "
                                                           "count, cannot be <=0.")
@click.option('-g', '--gpu', is_flag=True,
              help="bounds the number of workers to the number of available GPUs (not under load)."
                   "Each process only sees a single GPU.")
@click.option('-n', '--n', default=1, type=int, help='number of configuration runs')
@click.option('--name', default="opt", type=str,
              help='optimization experiment name: used as prefix for some output files')
@click.option('-s', '--surrogate', default="GP",
              type=click.Choice(["GP", "RF", "ET"], case_sensitive=False),
              help='surrogate model for global optimisation: '
                   '(GP) gaussian process, '
                   '(RF) random forest, or'
                   '(ET) extra trees.')
@click.option('-a', '--acquisition', default="EI",
              type=click.Choice(["LCB", "EI", "PI"], case_sensitive=False),
              help='acquisition function: '
                   '(LCB) Lower Confidence Bound, '
                   '(EI) Expected Improvement, or '
                   '(PI) Probability of Improvement.')
@click.option('--plot', is_flag=True, help='shows a convergence plot during the optimization process and saves it at'
                                           'the current working dir')
@click.option('--logfile', is_flag=True, help="if set outputs a log file with errors that might occur in the working "
                                              "processes.")
@click.option('-o', '--out', type=click.Path(), help="output directory for the results file. If plotting "
                                                     "convergence, the plot is also saved in the first directory in the"
                                                     "path.")
@click.option('--sync/--async', default=False, help="--async (default) submission, means that we submit a new "
                                                    "configuration to a worker for each new result the optimizer gets."
                                                    "--sync mode makes the optimizer wait for all workers before "
                                                    "submitting new configurations to all of them")
@click.option('--kuma', is_flag=True, help='kuma-san will display the progress on your global optimization procedure')
def run(params, module, workers, gpu, n, surrogate, acquisition, name, plot, logfile, out, sync, kuma):
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler('{name}.log'.format(name=name), delay=True)
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    opt_results = None
    out_file = None
    try:
        if gpu:
            # detecting available gpus with load < 0.1
            gpu_ids = [g.id for g in GPUtil.getGPUs() if g.load < 0.2]
            num_workers = min(workers, len(gpu_ids))

            if num_workers <= 0:
                sys.exit(1)
        else:
            num_workers = min(workers, mp.cpu_count())

            logger.log(logging.DEBUG, "Spawning {} workers".format(num_workers))
            if num_workers <= 0:
                logger.log(logging.ERROR, "--workers cannot be 0")
                sys.exit(1)

        # prepare output file
        out_file_name = '{}_configurations.csv'.format(name)
        out = out_file_name if out is None else out
        if out is not None and os.path.isdir(out):
            out_file_path = os.path.join(out, out_file_name)
        else:
            out_file_path = out

        out_dir = os.path.abspath(os.path.join(out_file_path, os.pardir))
        out_file_path = os.path.join(out_dir, out_file_name)

        param_space = ParamSpace(params)

        dimensions = params_to_skopt(param_space)
        optimizer_dims = [d.name for d in dimensions]
        optimizer = Optimizer(dimensions=dimensions,
                              base_estimator=surrogate,
                              acq_func=acquisition)

        out_file = open(out_file_path, 'w')
        out_writer = csv.DictWriter(out_file, fieldnames=param_space.param_names() + ["id", "evaluation"])
        out_writer.writeheader()

        # setup process pool and queues
        # manager = mp.Manager()
        config_queue = Queue()
        result_queue = Queue()
        error_queue = Queue()

        terminate_flags = [Event() for _ in range(num_workers)]
        processes = [
            Process(target=worker, args=(i, module, config_queue, result_queue, error_queue, terminate_flags[i]))
            for i in range(num_workers)]

        configs = []
        scores = {}
        # get initial points at random and submit one job per worker
        submit(num_workers, optimizer, optimizer_dims, configs, param_space, config_queue)
        # cfg_if: score

        num_completed = 0
        pending = len(configs)
        cancel = False

        for p in processes:
            p.daemon = True
            p.start()

        if plot:
            fig = plt.gcf()
            fig.show()
            fig.canvas.draw()

        progress_bar = tqdm(total=n, leave=True)

        if kuma:
            update_progress_kuma(progress_bar)

        while num_completed < n and not cancel:
            try:
                res = result_queue.get(timeout=1)
                pid, cfg_id, result = res
                if not isinstance(result, Exception):
                    cfg = configs[cfg_id]
                    # convert dictionary to x vector that optimizer takes
                    x = [cfg[param] for param in optimizer_dims]
                    # store scores for each config
                    scores[cfg_id] = result

                    out_row = dict(cfg)
                    out_row["evaluation"] = result
                    out_writer.writerow(out_row)
                    opt_results = optimizer.tell(x, result)

                    num_completed += 1
                    pending -= 1

                    if plot:
                        plots.plot_convergence(opt_results)
                        fig.canvas.draw()

                    # sync submission of jobs means we wait for all workers to finish
                    if sync and pending == 0:
                        if num_completed != n:
                            num_submit = min(num_workers, n - num_completed)
                            submit(num_submit, optimizer, optimizer_dims, configs, param_space, config_queue)
                            pending = num_submit
                        else:
                            terminate_flags[pid].set()

                    # async submission of jobs: as soon as we receive one result we submit the next
                    if not sync:
                        if (num_completed + pending) != n:
                            submit(1, optimizer, optimizer_dims, configs, param_space, config_queue)
                            pending += 1
                        else:
                            # signal the current worker for termination
                            terminate_flags[pid].set()

                    progress_bar.update()
                    progress_bar.set_postfix({"best solution ": opt_results["fun"]})

                    if kuma:
                        update_progress_kuma(progress_bar)

                else:
                    _, cfg_id_err, err = error_queue.get()
                    logger.error("configuration {} failed".format(cfg_id_err))
                    logger.error(err)

                    cancel = True
            except Empty:
                pass

        # try to wait for process termination
        for process in processes:
            process.join(timeout=0.5)

            if process.is_alive():
                process.terminate()

        progress_bar.close()

    except TomlDecodeError as e:
        logger.error(traceback.format_exc())
        print("\n\n[Invalid parameter file] TOML decode error:\n {}".format(e), file=sys.stderr)
    except ParamDecodeError as e:
        logger.error(traceback.format_exc())
        print("\n\n[Invalid parameter file]\n {}".format(e), file=sys.stderr)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise e
    except KeyboardInterrupt:
        pass
    finally:
        # debugging
        if opt_results is not None and plot:
            plt_file = '{}_convergence.pdf'.format(name)
            out_path = os.path.join(out_dir, plt_file)
            plt.savefig(out_path, bbox_inches='tight')

        if out_file is not None:
            out_file.close()


if __name__ == '__main__':
    run()
