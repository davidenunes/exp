import sys
import plac
from plac import Annotation
import os
from exp.grid import qsub
from itertools import zip_longest
from exp.params import ParamSpace
from exp.grid.conf import parse_gridconf
import glob
import subprocess

DEFAULT_CONFIGS = {
    "mas": {"venv": "conda",
            "venv_name": "deepsign",
            "parallel_env": "smp",
            "num_cores": 8
            },

    "ingrid": {"venv": "virtualenv",
               "venv_name": "deepsign-gpu",
               "venv_root": "envs",
               "parallel_env": "mp",
               "num_cores": 8,
               "resource_dict": {"release": "el7"},
               "queue_name": "hpcgrid",
               "modules": ["hdf5-1.8.16", "python-3.5.1"]
               },

    "ingrid-gpu": {"venv": "virtualenv",
                   "venv_root": "envs",
                   "venv_name": "mp",
                   "num_cores": 8,
                   "resource_dict": {"release": "el7", "gpu": "1"},
                   "queue_name": "hpcgrid",
                   "modules": ["hdf5-1.8.16", "python-3.5.1", "cuda-7.5"]
                   }
}


def _groupe_it(iterable, n, fillvalue=None):
    """ Collect data into fixed-length chunks or blocks
    Args:
        iterable: the it from which we will be grouping
        n: size of each group
        fillvalue: value to be filled when there are not enough elems to fill all groups

    Examples::
        _groupe_it('ABCDEFG', 3, 'x') --> ABC DEF Gxx

    Returns:
        a new it over groups of n instead of 1
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def write_job_file(grid_conf,
                   script_path,
                   job_filename,
                   script_params,
                   job_name="job",
                   run_in_cwd=False,
                   working_dir=None):
    """

    Args:
        grid_conf (dict): a dictionary with the configurations for the grid: pythonpath, virtualenv, modules, etc
        job_name: sge param that specifies the name for the job (as appears in qstat)
        script_params: a dictionary with the parameters to be passed to the python script
        job_filename: filename for the filename.sh file to be generated
        script_path: the path to the python script to be executed
        working_dir (str): sets the current working dir to the provided value
        run_in_cwd (bool): if set to True, adds the cwd param to the grid job script which specifies
        that the job is the be executed in the current working directory: uses sge path alias facilities
    """
    grid_conf = parse_gridconf(grid_conf)

    jobfile = qsub.write_qsub_file(job_filename=job_filename,
                                   script_path=script_path,
                                   script_params=script_params,
                                   venv=grid_conf["venv"],
                                   venv_name=grid_conf["venv_name"],
                                   venv_root=grid_conf["venv_root"],
                                   pythonpath=grid_conf["pythonpath"],
                                   job_name=job_name,
                                   queue_name=grid_conf["queue_name"],
                                   parallel_env=grid_conf["parallel_env"],
                                   num_cores=grid_conf["num_cores"],
                                   sge_params=grid_conf["sge_params"],
                                   resource_dict=grid_conf["resource_dict"],
                                   modules=grid_conf["modules"],
                                   run_in_cwd=run_in_cwd,
                                   working_dir=working_dir
                                   )

    return jobfile


def write_job_files(grid_conf,
                    script_path,
                    job_filename,
                    param_grid,
                    group=1,
                    job_name="job",
                    cwd=False,
                    wd=None):
    grouped_param_grid = _groupe_it(param_grid, group)

    job_files = []
    for params_ls in grouped_param_grid:
        params_ls = [param for param in params_ls if param is not None]

        ids = [params["id"] for params in params_ls]
        if len(ids) > 1:
            group_jobname = "{job}{id0}{id1}".format(job=job_name, id0=ids[0], id1=ids[-1])
            group_job_filename = "{job}{id0}{id1}".format(job=job_filename, id0=ids[0], id1=ids[-1])
        else:
            group_jobname = "{job}{id}".format(job=job_name, id=ids[0])
            group_job_filename = "{job}{id}".format(job=job_filename, id=ids[0])

        job_file = write_job_file(grid_conf,
                                  script_path,
                                  group_job_filename,
                                  params_ls,
                                  group_jobname,
                                  cwd,
                                  wd)
        job_files.append(job_file)
    return job_files


class GridCli:
    commands = ['run', 'runs', 'job', 'jobs', 'qsub']

    @plac.annotations(
        script=Annotation("path to python script to be run by the job"),
        params=Annotation("path to param space file", 'option'),
        group=Annotation("number of script calls per job", 'option', type=int),
        jobname=Annotation("name for the job to be submitted", 'option'),
        grid=Annotation("grid configuration", kind='option'),
        job_cwd=Annotation("run job in current working dir?", "flag"),
        job_wd=Annotation("working dir for job", 'option'),
    )
    def runs(self, script, params=None, group=1, grid="mas", jobname="job", job_cwd=False, job_wd=None):
        if not os.path.exists(params) and not os.path.isfile(script):
            print("Parameter space file not found: {path}".format(path=params), file=sys.stderr)
            sys.exit(1)

        ps = ParamSpace(filename=params)
        ps.write_grid_summary(jobname + '_params.csv')

        grid_cfg = DEFAULT_CONFIGS[grid]
        param_grid = ps.param_grid(include_id=True, id_param="id")

        job_files = write_job_files(grid_cfg, script, jobname, param_grid, group, jobname, job_cwd, job_wd)

        for job_file in job_files:
            try:
                print("submitting ", job_file)
                subprocess.call(args=["qsub", job_file])
            except FileNotFoundError:
                print("qsub command not found", file=sys.stderr)

    @plac.annotations(
        script=Annotation("path to python script to be run by the job"),
        jobname=Annotation("name for the job to be submitted", "option"),
        grid=Annotation("grid configuration", "option"),
        job_cwd=Annotation("run job in current working dir?", "flag"),
        job_wd=Annotation("working dir for job", "option"),
        params=Annotation("python script parameters")
    )
    def run(self, script, grid="mas", jobname="job", job_cwd=False, job_wd=None, **params):
        """ creates job scripts and submits them using qsub
        """

        grid_cfg = DEFAULT_CONFIGS[grid]
        job_file = write_job_file(grid_cfg, script, jobname, params, jobname, job_cwd, job_wd)

        try:
            print("submitting ", job_file)
            subprocess.call(args=["qsub", job_file])
        except FileNotFoundError:
            print("qsub command not found", file=sys.stderr)

    @plac.annotations(
        script=Annotation("path to python script to be run by the job"),
        jobname=Annotation("name for the job to be submitted", "option"),
        grid=Annotation("grid configuration", "option"),
        job_cwd=Annotation("run job in current working dir?", "flag"),
        job_wd=Annotation("working dir for job", "option"),
        params=Annotation("python script parameters")
    )
    def job(self, script, grid="mas", jobname="job", job_cwd=False, job_wd=None, **params):
        """ creates script job for sge

        """
        grid_cfg = DEFAULT_CONFIGS[grid]
        write_job_file(grid_cfg, script, jobname, params, jobname, job_cwd, job_wd)

    @plac.annotations(
        script=Annotation("path to python script to be run by the job"),
        params=Annotation("path to param space file", 'option'),
        group=Annotation("number of script calls per job", 'option', type=int),
        jobname=Annotation("name for the job to be submitted", 'option'),
        grid=Annotation("grid configuration", kind='option'),
        job_cwd=Annotation("run job in current working dir?", "flag"),
        job_wd=Annotation("working dir for job", 'option'),
    )
    def jobs(self, script, params=None, group=1, grid="mas", jobname="job", job_cwd=False, job_wd=None):
        """ creates script jobs for sge from a parameter space

        """
        if not os.path.exists(params) and not os.path.isfile(script):
            print("Parameter space file not found: {path}".format(path=params), file=sys.stderr)
            sys.exit(1)

        ps = ParamSpace(filename=params)
        ps.write_grid_summary(jobname + '_params.csv')

        grid_cfg = DEFAULT_CONFIGS[grid]
        param_grid = ps.param_grid(include_id=True, id_param="id")
        write_job_files(grid_cfg, script, jobname, param_grid, group, jobname, job_cwd, job_wd)

    # noinspection PyMethodMayBeStatic
    def qsub(self, regex_fname):
        """ calls qsub on all files that match the given expression for their filename

        """
        files = glob.glob(regex_fname)
        if len(files) == 0:
            print("no files found matching {}".format(regex_fname), file=sys.stderr)
        else:
            try:
                for f in files:
                    if os.path.isfile(f):
                        print("submitting ", f)
                        subprocess.call(args=["qsub", f])
            except FileNotFoundError:
                print("qsub command not found", file=sys.stderr)

    def __missing__(self, name):
        print("command not found {cmd}".format(cmd=name))


if __name__ == '__main__':
    plac.Interpreter.call(GridCli)
