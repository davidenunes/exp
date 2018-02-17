import configobj


def dummy_cfg():
    cfg = configobj.ConfigObj()
    cfg["venv"] = "conda"
    cfg["venv_name"] = "deepsign"
    cfg["venv_root"] = None
    cfg["pythonpath"] = []
    cfg["parallel_env"] = "smp"
    cfg["num_cores"] = 8
    cfg["queue_name"] = None
    cfg["sge_params"] = []
    cfg["resource_dict"] = {}
    cfg["modules"] = []


def parse_gridconf(grid_cfg):
    """ validates grid configuration,
        updates the given grid cfg with default values
        parses the values to the correct type (e.g. None)

    Args:
        grid_cfg: a configobj.ConfigObj instance
    """
    if "venv" not in grid_cfg:
        grid_cfg["venv"] = None
    venv = grid_cfg["venv"]
    venv = None if venv == "None" else venv

    if venv is not None and venv != "virtualenv" and venv != "conda":
        raise ValueError("Invalid venv: expected conda or virtualenv got {} instead".format(venv))

    if venv is not None and venv == "virtualenv" and "venv_root" not in grid_cfg:
        raise ValueError("venv is virtualenv so venv_root cannot be None")

    if venv is not None and "venv_name" not in grid_cfg:
        raise ValueError("venv is supplied so venv_name cannot be None")

    if venv is None:
        grid_cfg["venv_name"] = None
        grid_cfg["venv_root"] = None

    if venv == "conda":
        grid_cfg["venv_root"] = None

    if "pythonpath" not in grid_cfg:
        grid_cfg["pythonpath"] = []

    if "parallel_env" not in grid_cfg:
        grid_cfg["parallel_env"] = "smp"

    if "num_cores" not in grid_cfg:
        grid_cfg["num_cores"] = 1

    if "queue_name" not in grid_cfg:
        grid_cfg["queue_name"] = None

    if "sge_params" not in grid_cfg:
        grid_cfg["sge_params"] = []

    if "resource_dict" not in grid_cfg:
        grid_cfg["resource_dict"] = {}

    if "modules" not in grid_cfg:
        grid_cfg["modules"] = []

    return grid_cfg
