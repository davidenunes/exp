from enum import Enum
from exp import qsub
from subprocess import run
from exp.params import ParamSpace


class GridConf:
    """ GridConf grid configuration object

    Note: I could use just a dict but this way I get the parameters to be documented
    """

    class VENVS(Enum):
        VIRTUALENV: 0
        CONDA: 1

    def __init__(self,
                 venv_name,
                 venv_root=None,
                 venv=VENVS.CONDA,
                 pythonpath=None,
                 parallel_env="smp",
                 num_cores=8,
                 queue_name=None,
                 sge_params=None,
                 resource_dict=None,
                 module_names=[]
                 ):
        if venv_root is None and venv is GridConf.VENVS.VIRTUALENV:
            raise ValueError("if venv is a virtualenv, venv_root path should be supplied")

        self.venv_root = venv_root
        self.venv = venv
        self.venv_name = venv_name
        self.pythonpath = pythonpath
        self.parallel_env = parallel_env
        self.num_cores = num_cores
        self.queue_name = queue_name
        self.sge_params = sge_params
        self.resource_dict = resource_dict
        self.module_names = module_names


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

        if cfg.venv == GridConf.VENVS.CONDA:
            env_activate = qsub.conda_activate(cfg.venv_name)
            env_deactivate = qsub.conda_deactivate
        elif cfg is GridConf.VENVS.VIRTUALENV:
            env_activate = qsub.venv_activate(cfg.venv_root, cfg.venv_name)
            env_deactivate = qsub.venv_deactivate

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

    def submit_one(self, script_path, job_filename, param_dict, job_name="job", run_in_cwd=False, working_dir=None):
        """ Submit a single job to the grid

        Args:
            job_name: sge param that specifies the name for the job (as appears in qstat)
            param_dict: a dictionary with the parameters to be passed to the python script
            job_filename: filename for the filename.sh file to be generated
            script_path: the path to the python script to be executed
            working_dir (str): sets the current working dir to the provided value
            run_in_cwd (bool): if set to True, adds the cwd param to the grid job script which specifies
            that the job is the be executed in the current working directory: uses sge path alias facilities

        """
        self.write_job_file(script_path, job_filename, param_dict, job_name, run_in_cwd, working_dir)

        cmd = ["qsub", job_filename]
        return run(cmd)

    def submit_all(self, python_script, param_space: ParamSpace, job_filename="job", run_in_cwd=False,
                   working_dir=None):
        """ Submit all jobs in a  ``ParamSpace``

        creates a submission script for each each parameter configuration in the given ``ParamSpace``
        the submission file activates the relevant environment (based on the supplied grid configuration)
        and runs the given script with each of the parameter combinations

        Args:
            job_filename: base job submission filename, submit all will add an id to the name, example:
            ``job_0.sh``.
            param_space: ``ParamSpace`` with all the possible parameter configurations the python script will be
            run with.
            python_script: path to the python script to be executed
            working_dir (str): sets the current working dir to the provided value
            run_in_cwd (bool): if set to True, adds the cwd param to the grid job script which specifies
            that the job is the be executed in the current working directory: uses sge path alias facilities
        """
        param_space.write_grid_summary()
        param_grid = param_space.param_grid(include_id=True, id_param="id")

        for params in param_grid:
            i = params["id"]
            jobname = "{job}_{i}".format(job=job_filename, i=i)
            self.submit_one(python_script, jobname, params, jobname, run_in_cwd, working_dir)
