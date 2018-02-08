import math
import configobj as cfg
import itertools
import numpy as np


class ParamSpace:
    class Types:
        RANGE = "range"
        LINSPACE = "linspace"
        LIST = "list"
        VALUE = "value"
        RANDOM = "random"
        RANDINT = "randint"

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
        self.grid_size = 0

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

    def add_range(self, name, start, stop, step):
        sec = self._new_param(name)
        sec["type"] = ParamSpace.Types.RANGE
        sec["start"] = start
        sec["stop"] = stop
        sec["step"] = step

        self._update_grid_size(math.ceil(stop / step))

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
        elif param_type == ParamSpace.Types.RANDOM:
            return self.get_random(param)
        elif param_type == ParamSpace.Types.RANDINT:
            return self.get_randint(param)
        else:
            raise TypeError("Unknown Parameter Type")

    def param_grid(self):
        """ Returns a generator of dictionaries with all the possible parameter combinations
        the keys are the parameter names the values are the current value for each parameter

        Note:
            The parameters are combined using the same order by which they were added to the
            parameter space.

        Returns: an generator over all possible parameter configurations.

        """
        params = self.config.sections
        param_values = []

        for param in params:
            param_type = self.config[param]["type"]
            param_value = self.get_param(param, param_type)
            if param_type == ParamSpace.Types.VALUE:
                param_value = [param_value]
            param_values.append(param_value)

        param_product = itertools.product(*param_values)
        param_product = (dict(zip(params, values)) for values in param_product)
        return param_product

    def write(self, filename):
        """ Write the parameter space to a given file
        Args:
            filename: the path to the file to which the configuration file is to be written to.
        """
        self.config.filename = filename
        self.config.write()
