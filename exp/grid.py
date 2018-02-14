from enum import Enum
from exp import qsub
from subprocess import run
from exp.params import ParamSpace
from collections import namedtuple
from itertools import zip_longest


def _groupe_it(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


GridConf = namedtuple('GridConf', ['venv_name',
                                   'venv_root',
                                   'venv',
                                   'pythonpath',
                                   'parallel_env',
                                   'num_cores',
                                   'queue_name',
                                   'sge_params',
                                   'resource_dict',
                                   'module_names'])


class VirtualEnvs(Enum):
    Conda = 0
    VirtualEnv = 1


def grid_conf(venv_name,
              venv_root=None,
              venv=VirtualEnvs.Conda,
              pythonpath=None,
              parallel_env="smp",
              num_cores=8,
              queue_name=None,
              sge_params=None,
              resource_dict=None,
              module_names=[]):
    if venv_root is None and venv is VirtualEnvs.VirtualEnv:
        raise ValueError("if venv is a virtualenv, venv_root path should be supplied")

    return GridConf(venv_name,
                    venv_root,
                    venv,
                    pythonpath,
                    parallel_env,
                    num_cores,
                    queue_name,
                    sge_params,
                    resource_dict,
                    module_names)


class GridRunner:
    """ GridRunner.

    Can submit jobs to the grid based on a :class:`GridConf`.

    Submissions can be made based on a single parameter dictionary or
    a :class:`ParamSpace` instance.
    """

    def __init__(self, grid_conf: GridConf):
        self.grid_conf = grid_conf

    def write_job_file(self, script_path, job_filename, param_dict, job_name="job", run_in_cwd=False, working_dir=None):
        """

        Args:
            job_name: sge param that specifies the name for the job (as appears in qstat)
            param_dict: a dictionary with the parameters to be passed to the python script
            job_filename: filename for the filename.sh file to be generated
            script_path: the path to the python script to be executed
            working_dir (str): sets the current working dir to the provided value
            run_in_cwd (bool): if set to True, adds the cwd param to the grid job script which specifies
            that the job is the be executed in the current working directory: uses sge path alias facilities
        """
        cfg = self.grid_conf

        if cfg.venv is VirtualEnvs.Conda:
            env_activate = qsub.conda_activate(cfg.venv_name)
            env_deactivate = qsub.conda_deactivate
        elif cfg.venv is VirtualEnvs.VirtualEnv:
            env_activate = qsub.venv_activate(cfg.venv_root, cfg.venv_name)
            env_deactivate = qsub.venv_deactivate
        else:
            raise ValueError("invalid venv set in grid configuration")

        qsub.write_qsub_file(job_filename=job_filename,
                             script_path=script_path,
                             script_params=param_dict,
                             env_activate=env_activate,
                             env_deactivate=env_deactivate,
                             pythonpath=cfg.pythonpath,
                             job_name=job_name,
                             queue_name=cfg.queue_name,
                             parallel_env=cfg.parallel_env,
                             num_cores=cfg.num_cores,
                             sge_params=cfg.sge_params,
                             resource_dict=cfg.resource_dict,
                             module_names=cfg.module_names,
                             run_in_cwd=run_in_cwd,
                             working_dir=working_dir
                             )

    def submit_one(self, script_path, job_filename, param_dict, job_name="job", run_in_cwd=False, working_dir=None,
                   call_qsub=True):
        """ Submit a single job to the grid

        Args:
            call_qsub: if False, generates the job file only
            job_name: sge param that specifies the name for the job (as appears in qstat)
            param_dict: a dictionary with the parameters to be passed to the python script
            job_filename: filename for the filename.sh file to be generated
            script_path: the path to the python script to be executed
            working_dir (str): sets the current working dir to the provided value
            run_in_cwd (bool): if set to True, adds the cwd param to the grid job script which specifies
            that the job is the be executed in the current working directory: uses sge path alias facilities

        """
        self.write_job_file(script_path, job_filename, param_dict, job_name, run_in_cwd, working_dir)

        if call_qsub:
            cmd = ["qsub", job_filename]
            return run(cmd)
        return None

    def submit_all(self, python_script, param_space: ParamSpace, job_filename="job", run_in_cwd=False,
                   working_dir=None, call_qsub=True, write_params_file=True, group=1):
        """ Submit all jobs in a  ``ParamSpace``

        creates a submission script for each each parameter configuration in the given ``ParamSpace``
        the submission file activates the relevant environment (based on the supplied grid configuration)
        and runs the given script with each of the parameter combinations

        Args:
            write_params_file: if True writes a csv with all the parameters in the parameter space parameter grid
            call_qsub: if False, does not submit the jobs instead, it generates the submission files only
            job_filename: base job submission filename, submit all will add an id to the name, example:
            ``job`` -> ``job_0.sh``.
            param_space: ``ParamSpace`` with all the possible parameter configurations the python script will be
            run with.
            python_script: path to the python script to be executed
            working_dir (str): sets the current working dir to the provided value
            run_in_cwd (bool): if set to True, adds the cwd param to the grid job script which specifies
            that the job is the be executed in the current working directory: uses sge path alias facilities
            group: each job calls n number of scripts with their respective params (masks number of real jobs in the
            grid but has to perform each run in the job sequentially
        """
        param_space.write_grid_summary(job_filename + '_params.csv')
        param_grid = param_space.param_grid(include_id=True, id_param="id")

        grouped_params = _groupe_it(param_grid, group)

        for params_ls in grouped_params:
            params_ls = [param for param in params_ls if param is not None]

            ids = [params["id"] for params in params_ls]
            if len(ids) > 1:
                jobname = "{job}_{id0}_{id1}".format(job=job_filename, id0=ids[0], id1=ids[-1])
            else:
                jobname = "{job}_{id}".format(job=job_filename, id=ids[0])

            self.submit_one(python_script, jobname, params_ls, jobname, run_in_cwd, working_dir, call_qsub)
