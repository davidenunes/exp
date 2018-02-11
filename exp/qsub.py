import os

bash_header = "#!/bin/bash\n"


def _as_list(elems):
    """ returns a list from the given element(s)

    Args:
        elems: one or more objects

    Returns:
        a list with the elements in elems
    """
    if elems is None:
        elems = []
    elif isinstance(elems, (list, tuple)):
        elems = list(elems)
    else:
        elems = [elems]
    return elems


def sge(job_name=None,
        queue_name=None,
        sge_params=None,
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
        sge_params: list of standalone parameters to be added (one per line)
            example::

                for sge params= "V", generates the line
                #$ -V

                for sge params= ["V","cwd"], generates the lines:
                #$ -V
                #$ -cwd
        it prints the parameters in a case-sensitive manner

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
    if sge_params is not None:
        sge_params = _as_list(sge_params)
        for param in sge_params:
            job_params.append("#$ -{param}\n".format(param=param))

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


def pythonpath_add(path, validate=False):
    """ line that adds a path to PYTHONPATH env var

    Args:
        path: path to be added to PYTHONPATH
        validate: if True checks if path exists and is a dir

    Returns:
        str: a string with a line in bash used to update the PYTHONPATH with a given path

    """
    if validate:
        if not os.path.exists(path):
            raise ValueError("trying to add invalid path to PYTHON: {p} does not exist".format(p=path))

        if not os.path.isdir(path):
            raise ValueError("trying to add invalid path to PYTHONPATH: {p} should be a dir".format(p=path))
    return "export PYTHONPATH=\"${PYTHONPATH}:" + path + '\"\n'


def conda_activate(env_name):
    """ conda environment activate

    Args:
        env_name: name of the environment

    Returns:
        str: a string with the command to activate a given conda environment
    """
    return "source activate {env}\n".format(env=env_name)


conda_deactivate = "source deactivate\n"


def venv_activate(venv_root, env_name):
    """ virtualenv activation

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
                    pythonpath=None,
                    job_name=None,
                    queue_name=None,
                    parallel_env="smp",
                    num_cores=4,
                    max_memory=None,
                    sge_params=None,
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

        sge_params: list or single str case-sensitive params without arguments
        pythonpath: a list with the paths to add to python path before executing the script
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
                         sge_params,
                         parallel_env,
                         num_cores, max_memory,
                         resource_dict,
                         std_out_file,
                         std_err_file,
                         run_in_cwd,
                         working_dir))

    qsub_file.extend(module_load(module_names))

    if pythonpath is not None:
        pythonpath = _as_list(pythonpath)
        exports = [pythonpath_add(path) for path in pythonpath]
        qsub_file.extend(exports)

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
