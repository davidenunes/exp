import math
import configobj as cfg
import itertools
import numpy as np
import csv
import os


def _repeat_it(iterable, n):
    return itertools.chain.from_iterable(itertools.repeat(x, n) for x in iterable)


class ParamSpace:
    class Types:
        RANGE = "range"
        LINSPACE = "linspace"
        LIST = "list"
        VALUE = "value"
        RANDOM = "random"
        RANDINT = "randint"
        LOGSPACE = "logspace"

    @staticmethod
    def from_file(path):
        """ New ParamSpace from file
                Note:
                    this file is expected to be written using :meth:`write`. If the file doesn't exist,
                    returns a new empty :class:`ParamSpace`.
                Args:
                    path: path from which the param space is to be loaded
                Returns:
                    :class:`ParamSpace` : a new parameter space initialized from the given path
        """
        return ParamSpace(path)

    def __init__(self, filename=None):
        self.config = cfg.ConfigObj(filename)
        # if we build it from a file we must get the size

        self.grid_size = self._compute_size()

    def _compute_size(self):
        params = self.config.sections
        if len(params) > 0:
            size = 1
        else:
            return 0

        for param in params:
            param_type = self.config[param]["type"]
            param_value = self.get_param(param, param_type)
            if param_type == ParamSpace.Types.VALUE:
                param_value = [param_value]
            size *= len(param_value)

        return size

    def get_params(self):
        return self.config.sections

    def _update_grid_size(self, n):
        if self.grid_size == 0:
            self.grid_size = n
        else:
            self.grid_size *= n

    def _new_param(self, name):
        self.config[name] = {}
        return self.config[name]

    def add_linspace(self, name, start, stop, num):
        sec = self._new_param(name)
        sec["type"] = ParamSpace.Types.LINSPACE
        sec["start"] = start
        sec["stop"] = stop
        sec["num"] = num

        self._update_grid_size(num)

    def add_logspace(self, name, start, stop, num):
        sec = self._new_param(name)
        sec["type"] = ParamSpace.Types.LOGSPACE
        sec["start"] = start
        sec["stop"] = stop
        sec["num"] = num

        self._update_grid_size(num)

    def add_random(self, name, low, high, num, persist=False):
        """ Adds a random uniform float parameter definition to the parameter space.
        Args:
            name: parameter name
            low: low limit for random uniform distribution
            high: high limit for random uniform distribution
            num: number of samples
            persist: if True, generates the parameters and stores them as ParamSpace.Types.LIST
            instead of ParamSpace.Types.RANDOM which generates a new grid each time param_grid is
            called.
        """
        sec = self._new_param(name)
        if persist:
            values = list(np.random.uniform(low, high, num))
            self.add_list(name, values)
        else:
            sec["type"] = ParamSpace.Types.RANDOM
            sec["low"] = low
            sec["high"] = high
            sec["num"] = num
            self._update_grid_size(num)

    def add_randint(self, name, low, high, num, persist=False):
        sec = self._new_param(name)

        if persist:
            values = list(np.random.randint(low, high, num))
            self.add_list(name, values)
        else:
            sec["type"] = ParamSpace.Types.RANDINT
            sec["low"] = low
            sec["high"] = high
            sec["num"] = num
            self._update_grid_size(num)

    def add_range(self, name, stop, start=0, step=1):
        sec = self._new_param(name)
        sec["type"] = ParamSpace.Types.RANGE
        sec["start"] = start
        sec["stop"] = stop
        sec["step"] = step

        self._update_grid_size((stop - start - 1) // step + 1)

    def add_value(self, name, value):
        sec = self._new_param(name)
        sec["type"] = ParamSpace.Types.VALUE
        sec["value"] = value

        self._update_grid_size(1)

    def add_list(self, name, values):
        values = list(values)

        sec = self._new_param(name)
        sec["type"] = ParamSpace.Types.LIST
        sec["values"] = values

        self._update_grid_size(len(values))

    def _get_param(self, name, param_type):
        """
        Returns a parameter of a given type
        Args:
            name: parameter name
            param_type: type of parameter (ParamSpace.Types)
        Returns:
            reference to a dictionary with the parameter values
        """
        if name not in self.config:
            raise KeyError("Couldn't find a parameter named {}".format(name))

        param = self.config[name]
        if param["type"] != param_type:
            raise TypeError("{param} is not a sequence but a {type}".format(param=name, type=param["type"]))

        return param

    def get_random(self, name):
        param = self._get_param(name, ParamSpace.Types.RANDOM)
        low = float(param["low"])
        high = float(param["high"])
        num = int(param["num"])

        return np.random.uniform(low, high, num)

    def get_randint(self, name):
        param = self._get_param(name, ParamSpace.Types.RANDINT)
        low = int(param["low"])
        high = int(param["high"])
        num = int(param["num"])

        return np.random.randint(low, high, num)

    def get_range(self, name):
        param = self._get_param(name, ParamSpace.Types.RANGE)

        start = float(param["start"])
        stop = float(param["stop"])
        step = float(param["step"])

        return np.arange(start, stop, step)

    def get_logspace(self, name):
        param = self._get_param(name, ParamSpace.Types.LOGSPACE)

        start = float(param["start"])
        stop = float(param["stop"])
        num = int(param["num"])

        return np.logspace(start, stop, num)

    def get_linspace(self, name):
        param = self._get_param(name, ParamSpace.Types.LINSPACE)

        start = float(param["start"])
        stop = float(param["stop"])
        num = int(param["num"])

        return np.linspace(start, stop, num)

    def get_value(self, name):
        param = self._get_param(name, ParamSpace.Types.VALUE)
        return param["value"]

    def get_list(self, name):
        param = self._get_param(name, ParamSpace.Types.LIST)
        return param["values"]

    def get_param(self, param, param_type):
        if param_type == ParamSpace.Types.RANGE:
            return self.get_range(param)
        elif param_type == ParamSpace.Types.VALUE:
            return self.get_value(param)
        elif param_type == ParamSpace.Types.LIST:
            return self.get_list(param)
        elif param_type == ParamSpace.Types.LINSPACE:
            return self.get_linspace(param)
        elif param_type == ParamSpace.Types.LOGSPACE:
            return self.get_logspace(param)
        elif param_type == ParamSpace.Types.RANDOM:
            return self.get_random(param)
        elif param_type == ParamSpace.Types.RANDINT:
            return self.get_randint(param)
        else:
            raise TypeError("Unknown Parameter Type")

    def param_grid(self, include_id=False, id_param="id", run_id_param="run_id", nruns=False, runs=1):
        """ Returns a generator of dictionaries with all the possible parameter combinations
        the keys are the parameter names the values are the current value for each parameter
        Note:
            The parameters are combined using the same order by which they were added to the
            parameter space.
        Returns: an generator over all possible parameter configurations.
        Args:
            include_id (bool): if True ads a parameter id for each parameter dictionary it outputs
            id_param (str): if :arg:`include_id` is True, this is used as the default name for th id parameter
            runs (int): number of repeats for each unique configuration
        """
        params = self.config.sections
        param_values = []

        for param in params:
            param_type = self.config[param]["type"]
            param_value = self.get_param(param, param_type)
            if param_type == ParamSpace.Types.VALUE:
                param_value = [param_value]
            param_values.append(param_value)

        if include_id and nruns:
            if runs < 1:
                raise ValueError("runs must be >0: runs set to {}".format(runs))
            # add run to parameter names and run number to parameters
            params.append("run")
            run_ids = np.linspace(1, runs, runs, dtype=np.int32)
            param_values.append(run_ids)

        param_product = itertools.product(*param_values)
        param_product = (dict(zip(params, values)) for values in param_product)

        if include_id:
            # create ids for each unique configuration
            ids = np.linspace(0, self.grid_size - 1, self.grid_size, dtype=np.int32)
            ids = list(_repeat_it(ids, runs))
            id_param_names = [id_param] * (self.grid_size * runs)

            id_params = [{k: v} for k, v in zip(id_param_names, ids)]

            param_product = ({**p, **i} for i, p in zip(id_params, param_product))

        if include_id:
            param_product = (dict(param, **{run_id_param: i}) for i, param in enumerate(param_product))
        return param_product

    def write(self, filename):
        """ Write the parameter space to a given file
        Args:
            filename: the path to the file to which the configuration file is to be written to.
        """
        self.config.filename = filename
        self.config.write()

    def write_grid_summary(self, output_path="params.csv", conf_id_header="id"):
        """ Writes a csv file with each line containing a configuration value with a unique
        id for each configuration
        Args:
            conf_id_header : the header the be displayed on the summary file with the configuration id
            output_path: the output path for the summary file
        """
        summary_header = [conf_id_header]
        summary_header += self.get_params()

        with open(output_path, mode="w", newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=summary_header)
            writer.writeheader()

            param_grid = self.param_grid()
            conf_id = 0
            for param_row in param_grid:
                # add id to the current row
                param_row[conf_id_header] = conf_id
                conf_id += 1
                writer.writerow(param_row)

    def write_config_files(self, output_path="params", file_prefix="params", conf_id_header="id"):
        """ Writes one :mod:`configobj` file for each parameter configuration in the parameter space
        to a given path
        Args:
            output_path: the path where the configuration files are to be created
            file_prefix : a prefix for the params created to be followed by their id (enumeration)
            (e.g. if default is kept, the prefix is params and the files are written as "params_0.conf")
        """

        param_grid = self.param_grid(include_id=True, id_param=conf_id_header)
        conf_id = 0
        for current_config in param_grid:
            conf_file = "{prefix}_{id}.conf".format(prefix=file_prefix, id=conf_id)
            conf_file = os.path.join(output_path, conf_file)
            conf_obj = cfg.ConfigObj()
            conf_obj.filename = conf_file
            conf_obj.update(current_config)
            conf_obj.write()
            conf_id += 1
