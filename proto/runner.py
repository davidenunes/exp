from exp.params import ParamSpace
from subprocess import run
from exp import qsub
from enum import Enum

""" Prototype for Experiment Runner
- run method params:
    python script: the script to be run by the experiment runner
    parameter space: file containing the parameter space specification
    runner: name for the runner used for the experiment
    runner params: possibly the runner will need parameters, not necessary at this stage though
"""


class GridConf:
    class VENVS(Enum):
        VIRTUALENV: 0
        CONDA: 1

    def __init__(self, venv=VENVS.CONDA, venv_name="exp"):
        self.venv = venv
        self.venv_Name = venv_name


class GridRunner:
    """ GridRunner.

    Can submit jobs to the grid based on a :class:`GridConf`.

    Submissions can be made based on a single parameter dictionary or
    a :class:`ParamSpace` instance.
    """

    def __init__(self, grid_conf):
        self.grid_conf = grid_conf

    def write_job_file(self, script_path, job_filename, param_dict):
        cfg = self.grid_conf
        if cfg.venv == GridConf.VENVS.CONDA:
            qsub.qsub_file_conda(job_filename=job_filename,
                                 script_path=script_path,
                                 env_name=cfg.venv_name,
                                 script_params=param_dict)
        elif cfg is GridConf.VENVS.VIRTUALENV:
            qsub.qsub_file_venv(job_filename=job_filename,
                                script_path=script_path,
                                env_name=cfg.venv_name,
                                script_params=param_dict)

    def submit_one(self, script_path, job_filename, param_dict):
        """ Submit a single job to the grid
        """
        self.write_job_file(script_path, job_filename, param_dict)

        cmd = ["qsub", job_filename]
        return run(cmd)

    def submit_all(self, script_path: str, job_filename: str, param_space: ParamSpace):
        self.param_space.write_grid_summary()

        param_grid = param_space.param_grid(include_id=True)

        for params in param_grid:
            self.submit_one(script_path, job_filename, params)
