import os

bash_header = "#!/bin/bash\n"


def sge(job_name=None,
        queue_name=None,
        parallel_env="smp",
        num_cores=4,
        max_memory=None,
        resource_dict=None,
        std_out_file=None,
        std_err_file=None,
        run_in_cwd=False,
        working_dir=None):
    """

    Args:
        max_memory: a string with the maximum amount of memory required (memory per processor slot)
        resource_dict: you can request multiple resources with a dict {resource_name:value} will result in a call
            "#$ -l key1=value1,key2=value2
            and so on.
        working_dir: if provided, runs the job in the given dir path
        run_in_cwd: if True, runs the job in the current dir
        queue_name: name for the queue e.g. hpcgrid
        num_cores: number of processors / cores / nodes, depending on the parallel environment specified
        job_name: name for the job
        parallel_env: can be "smp" in some grids, "mp" in others or even "mpi" if mpi is used
        std_out_file: path for the std out log file
        std_err_file: path for the error log file
        run_in_cwd: run job in the current directory?
    """
    job_params = []
    if queue_name is not None:
        job_params.append("#$ -q {queue}\n".format(queue=queue_name))

    if job_name is not None:
        job_params.append("#$ -N {name}\n".format(name=job_name))
    job_params.append("#$ -pe {pe_name} {cores}\n".format(pe_name=parallel_env, cores=num_cores))

    if max_memory is not None:
        job_params.append("#$ h_vmem={max_mem}\n".format(max_mem=max_memory))

    if std_out_file is not None:
        job_params.append("#$ -o {std_out}\n".format(std_out=std_out_file))

    if std_err_file is not None:
        job_params.append("#$ -e {std_err}\n".format(std_err=std_err_file))

    if run_in_cwd:
        job_params.append("#$ -cwd\n")

    if working_dir is not None:
        job_params.append("#$ -wd {dir_path}\n".format(working_dir))

    if resource_dict is not None:
        resource_dict = dict()
        res = ["{r}={v}".format(r=r, v=resource_dict[r]) for r in resource_dict.keys()]
        res = ",".join(res)
        job_params.append("#$ -l {resources}\n".format(resources=res))

    return job_params


def module_load(modules=[]):
    env_load = []
    for module in modules:
        env_load.append("module load {m}\n".format(m=module))
    return env_load


def conda_activate(env_name):
    return "source activate {env}\n".format(env=env_name)


conda_deactivate = "source deactivate\n"


def venv_activate(venv_root, env_name):
    """ Str for virtualenv activation

    Args:
        venv_root: base dir for the virtual environments
        env_name: name for the virtual environment

    Returns:
        a string with a directive to activate the given virtual environment
    """
    activate_path = "bin/activate"
    env_path = os.path.join(venv_root, env_name)
    activate_path = os.path.join(env_path, activate_path)
    return "source {path}\n".format(path=activate_path)


venv_deactivate = "deactivate\n"


def write_qsub_file(out_filename,
                    script_path,
                    env_activate,
                    env_deactivate,
                    script_params=None,
                    job_name=None,
                    queue_name=None,
                    parallel_env="smp",
                    num_cores=4,
                    max_memory=None,
                    resource_dict=None,
                    std_out_file=None,
                    std_err_file=None,
                    run_in_cwd=False,
                    working_dir=None,
                    module_names=[]
                    ):
    """ Writes a file that can be submitted using the ``qsub`` command.

    The file created is intended as a means to write a python script under some virtual environment
    (either virtualenv or conda). See :func:`qsub.qsub_file_conda` or :func:`qsub.qsub_file_venv` methods
    instead of calling this directly.

    Args:

        out_filename: qsub script file name (e.g. something.sh)
        script_path: path to the python script to be executed using python script_path
        env_activate: line that activates the environment to be used
        env_deactivate: line that deactivates the environment to be used
        job_name: name for the grid job to be submitted
        queue_name: name of the queue where the job is to be submitted
        parallel_env: parallel env name (mp, smp, mpi)
        num_cores: number of cores requested in the given environment
        max_memory: maximum memory requested by the job
        resource_dict: additional resources requested as a dict (e.g. {"gpu":"True"}
        std_out_file: file where the std out from the job will be written (path is relative to working dir)
        std_err_file: file where the std err from the job will be written (path is relative to working dir)
        run_in_cwd: if True, runs the script in the current working directory
        working_dir: if a path is supplied, sets the working directory for the job
        module_names: if the system has a ``module load`` system, you can provide a list of module names and this
        creates the necessary lines to load them
        script_params: dictionary parameter_name:parameter_value
    """
    qsub_file = [bash_header]

    qsub_file.extend(sge(job_name,
                         queue_name,
                         parallel_env,
                         num_cores, max_memory,
                         resource_dict,
                         std_out_file,
                         std_err_file,
                         run_in_cwd,
                         working_dir))

    qsub_file.extend(module_load(module_names))
    qsub_file.append(env_activate)

    if script_params is not None and len(script_params) > 0:
        script_params_str = " ".join(["-{p}={v}".format(p=param, v=script_params[param]) for param in script_params])
    else:
        script_params_str = ""

    script_run = "python {path} {params}\n".format(path=script_path, params=script_params_str)
    qsub_file.append(script_run)
    qsub_file.append(env_deactivate)

    with open(out_filename, "w") as sh_file:
        sh_file.writelines(qsub_file)


def qsub_file_venv(out_filename,
                   script_path,
                   venv_root,
                   venv_name,
                   **kwargs):
    """ Creates a file that can be submitted to a sge queue system using qsub
    and activating a virtualenv environment

    Args:
        out_filename: name for the resulting file (e.g. "job0.sh")
        script_path: path to the script to be executed using python
        venv_root: path to the folder where all the virtual envs are installed
        venv_name: name of the virtual env to be activated
        **kwargs: args for qsub file (see :func:`~qsub.write_qsub_file`
    """
    write_qsub_file(out_filename=out_filename,
                    script_path=script_path,
                    env_activate=venv_activate(venv_root, venv_name),
                    env_deactivate=venv_deactivate,
                    **kwargs)


def qsub_file_conda(job_filename,
                    script_path,
                    env_name,
                    **kwargs):
    """ Creates a file that can be submitted to a sge queue system using qsub
        and activating a conda environment

        Args:

            job_filename: name for the resulting file (e.g. "job0.sh")
            script_path: path to the script to be executed using python
            env_name: conda environment name
            **kwargs: args for qsub file (see :func:`~qsub.write_qsub_file`
        """
    write_qsub_file(out_filename=job_filename,
                    script_path=script_path,
                    env_activate=conda_activate(env_name),
                    env_deactivate=conda_deactivate,
                    **kwargs)
