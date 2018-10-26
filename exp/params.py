""" Parameter Space definition writing and loading

    :obj:`ParamSpace` uses TOML as the underlying format to write and load
    configurations to and from
"""
import math
import toml
import itertools
import numpy as np
import csv
import os
from enum import Enum


def _repeat_it(iterable, n):
    return itertools.chain.from_iterable(itertools.repeat(x, n) for x in iterable)


class Types(Enum):
    """ Enum with valid parameter types supported by :obj:`ParamSpace`

    """
    VALUE = "value"
    LIST = "list"
    RANGE = "range"
    RANDOM = "random"

    @staticmethod
    def from_str(value, case_sensitive=False):
        if not case_sensitive:
            value.lower()
        if value == Types.VALUE.value:
            return Types.VALUE
        elif value == Types.LIST.value:
            return Types.LIST
        elif value == Types.RANGE.value:
            return Types.RANGE
        elif value == Types.RANDOM.value:
            return Types.RANDOM
        else:
            raise ValueError("invalid parameter type: {}\n"
                             "supported values: {}".format(value,
                                                           ",".join([t.name for t in Types])))


class DTypes(Enum):
    """ Enum with valid parameter dtypes supported by :obj:`ParamSpace`

        Useful to convert parameter spaces into scikit-optimization Dimensions
    """
    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"

    @staticmethod
    def from_type(dtype, case_sensitive=False):
        if not case_sensitive and isinstance(dtype, str):
            dtype.lower()

        if dtype in (DTypes.FLOAT.value, float):
            return DTypes.FLOAT
        elif dtype in (DTypes.INT.value, int):
            return DTypes.INT
        elif dtype == DTypes.CATEGORICAL.value:
            return DTypes.CATEGORICAL
        else:
            raise ValueError("invalid parameter dtype: {}\n"
                             "supported values: {}".format(dtype,
                                                           ",".join([t.name for t in DTypes])))


class ParamSpace:
    """ ParamSpace

    Create parameter spaces from configuration files.

    ParamSpace creates and read from parameter configuration files
    to create parameter spaces used for hyperparameter grid search
    (:py:mod:`exp.run`) or Global optimization procedures (:py:mod:`exp.gopt`)

    Args:
        filename (str): [optional] path to a configuration file. If not specified, creates an empty ParamSpace.
    """

    def __init__(self, filename=None):
        if filename is not None:
            self.params = toml.load(filename)
        else:
            self.params = {}
        self.size = self._compute_size()

    def _compute_size(self):
        """ Returns the size of the parameter space in terms of number of
        unique configurations

        Returns:
            (int) number of unique configurations

        """
        params = self.params.keys()
        if len(params) > 0:
            size = 1
        else:
            return 0

        for param in params:
            param_type = Types.from_str(self.params[param]["type"])
            param_value = self.get_param(param, param_type)

            if param_type == Types.VALUE:
                param_value = [param_value]
            size *= len(param_value)
        return size

    def param_names(self):
        """ param_names.

        Returns:
            (list) list of strings with the names of the parameters in this space

        """
        return list(self.params.keys())

    def _update_space_size(self, n):
        """ Updates the static grid size each instance maintains so it's cheap to return it

        Args:
            n: number of values a new parameter being added has
        """
        if self.size == 0:
            self.size = n
        else:
            self.size *= n

    def _get_param(self, name, param_type):
        """ get parameter

        gets a parameter and checks if a parameter is of a given type

        Args:
            name: name of the parameter to be returned
            param_type: (Type) parameter type

        Raises:
            LookupError: if parameter is not found in parameter space
            TypeError: if parameter found is not of the type specified

        Returns:
            (dict) dictionary with the parameter value and configurations

        """
        if name not in self.params:
            raise LookupError("Parameter not found: {}".format(name))

        param = self.params[name]
        actual_type = Types.from_str(param['type'])
        if actual_type != param_type:
            raise TypeError("expected {param} to be a {expected} but got {actual}".format(param=name,
                                                                                          expected=param_type,
                                                                                          actual=actual_type))

        return param

    def add_random(self, name, low=0., high=1., prior="uniform", dtype=float, n=None, persist=False):
        """ Specify random params within bounds and a given prior distribution

        Args:
            name: name for the parameter
            low: lower bound for the distribution (optional)
            high: higher bound for the distribution (optional)
            prior: distribution to be used for the random sampling, one of the following values:
                -uniform
                -log-uniform
            n: number of random values to be sampled if persist
            persist:
        Returns:

        """
        if dtype not in (float, int):
            raise TypeError(
                """\n Unknown dtype "{}", valid dtypes are:\n \t-float, \n \t-int """.format(dtype))

        param = self.params[name] = {}
        param["type"] = Types.RANDOM.value
        param["dtype"] = dtype.__name__
        if n:
            param["n"] = n
            self._update_space_size(n)
        else:
            self._update_space_size(1)

        param["bounds"] = [low, high]

        if prior not in ("uniform", "log-uniform"):
            raise TypeError(
                """\n Unknown prior "{}", valid priors are:\n \t-"uniform", \n \t-"log-uniform" """.format(prior))
        param["prior"] = prior

        if persist:
            value = self.get_random(name)
            del self.params[name]
            self.add_list(name, value)

    def get_random(self, name):
        """ get random parameter

        Args:
            name: parameter name

        Returns:
            array with one or more random parameter values according to the prior distribution
            and the given bounds, if no bounds are found defaults to [0,1), if no priors
            are found uniform is used

        """
        param: dict = self._get_param(name, Types.RANDOM)
        n = param.get("n", 1)
        bounds = param.get("bounds", [0, 1])
        prior = param.get("prior", "uniform")
        dtype = param.get("dtype", "float")

        if prior == "uniform":
            if dtype == "float":
                rvs = np.random.uniform(low=bounds[0], high=bounds[1], size=n)
            else:
                rvs = np.random.randint(low=bounds[0], high=bounds[1], size=n)
        if prior == "log-uniform":
            if dtype == "float":
                if bounds[0] == 0:
                    raise ValueError("lower bound on a log space cannot be 0")
                low = np.log10(bounds[0])
                high = np.log10(bounds[1])
                rvs = np.power(10, np.random.uniform(low, high, size=n))
            else:
                raise NotImplementedError("log uniform not implemented for random integer numbers")

        return rvs

    def add_value(self, name, value):
        """ Create a single value parameter

        Args:
            name: parameter name
            value: value to be attributed to param
        """
        param = self.params[name] = {}
        param["type"] = Types.VALUE.value
        param["value"] = value
        self._update_space_size(1)

    def get_value(self, name):
        """ get single value parameter

        Args:
            name: parameter name

        Returns:
            obj some value for the single value parameter

        """
        param = self._get_param(name, Types.VALUE)
        return param["value"]

    def add_list(self, name, values):
        """ Create list parameter

        Args:
            name: parameter name
            values: list of values
        """
        values = list(values)
        param = self.params[name] = {}
        param["type"] = Types.LIST.value
        param["value"] = values
        self._update_space_size(len(values))

    def add_range(self, name, low=0, high=1, step=1, dtype=float):
        """ Creates range parameter

        Args:
            name: name for the parameter
            low: where the range starts
            high: where the range stops (not included in the values)
            step: distance between each point in the range
            dtype: float or int if int the points in the range are rounded
        """
        param = self.params[name] = {}
        param["type"] = Types.RANGE.value
        param["bounds"] = [low, high]
        param["step"] = step
        param["dtype"] = dtype.__name__
        self._update_space_size(math.ceil((high - low) / step))

    def get_range(self, name):
        """ get range value for given range parameter

        works like numpy.arange

        Args:
            name: parameter name associated with the range

        Returns:
            an array with n numbers according to the range specification
        """
        param = self._get_param(name, Types.RANGE)
        if "bounds" not in param:
            raise LookupError(""" "bounds" not found for parameter {}""".format(name))
        low = float(param["bounds"][0])
        # high = float(param["bounds"]["high"])
        high = float(param["bounds"][1])

        step = float(param.get("step", 1.0))
        if "dtype" not in param:
            dtype = float
        else:
            dtype = param["dtype"]
            if dtype not in ("int", "float"):
                raise TypeError("invalid dtype for {}: expected int or float, got {}".format(name, dtype))
            dtype = int if dtype == "int" else float

        return np.arange(low, high, step, dtype=dtype)

    def get_list(self, name, unique=False):
        """ get list parameter

        Args:
            name: parameter name

        Returns:
            a list with the parameter values

        """
        param = self._get_param(name, Types.LIST)

        # return list of unique items
        value = param["value"]
        if not unique:
            return value
        else:
            _, idx = np.unique(value, return_index=True)
            return np.array(value)[np.sort(idx)].tolist()

    def get_param(self, param, type=None):
        if param not in self.params:
            raise KeyError("Parameter {} not found".format(param))

        if "type" not in self.params[param]:
            raise ValueError("Parameter found but not specified properly: missing \"type\" property")
        actual_type = Types.from_str(self.params[param]["type"])

        if type is not None and actual_type != type:
            raise ValueError("Parameter {p} has type {ta}, you requested {t}".format(p=param, ta=actual_type, t=type))

        if actual_type == Types.RANGE:
            return self.get_range(param)
        elif actual_type == Types.VALUE:
            return self.get_value(param)
        elif actual_type == Types.LIST:
            return self.get_list(param)
        elif actual_type == Types.RANDOM:
            return self.get_random(param)
        else:
            raise TypeError("Unknown Parameter Type")

    def domain(self, param_name):

        param = self.params[param_name]
        param_type = Types.from_str(param["type"])
        prior = param.get("prior", None)

        if param_type == Types.LIST:
            return {"domain": self.get_list(param_name, unique=True),
                    "dtype": DTypes.CATEGORICAL.value}

        elif param_type == Types.RANDOM:
            bounds = param.get("bounds", [0., 1.])
            dtype = param.get("dtype", DTypes.FLOAT.value)

            if prior is None:
                prior = "uniform"
            return {"domain": bounds, "dtype": dtype, "prior": prior}

        elif param_type == Types.RANGE:
            dtype = param.get("dtype", DTypes.FLOAT.value)
            bounds = param.get("bounds", [0., 1.])
            if prior is None:
                prior = "uniform"
            return {"domain": bounds, "dtype": dtype, "prior": prior}

        else:
            dtype = param.get("dtype", DTypes.CATEGORICAL.value)
            bounds = [self.get_param(param_name)]
            return {"domain": bounds, "dtype": dtype}

    def sample_param(self, name):
        """ draws a sample from a single parameter

        If this is a value it returns the value itself
        for a list or range draws one element uniformly at random
        for a random parameter, respects the distribution specified

        Args:
            name: parameter name to be sampled

        Returns:
            returns the sampled value for the given parameter name

        """
        if name in self.params:
            param = self.params[name]
            param_type = Types.from_str(param["type"])
            value = self.get_param(name, param_type)
            if param_type == Types.VALUE:
                return value
            elif param_type in (Types.LIST, Types.RANGE):
                randi = np.random.randint(0, len(value))
                return value[randi]
            elif param_type == Types.RANDOM:
                return value[0]
        else:
            raise KeyError("{} not in parameter space".format(name))

    def sample_space(self):
        """ Samples a configuration from the parameter space

        Returns:
            a dictionary with the sampled configuration
        """
        return {param: self.sample_param(param) for param in self.params.keys()}

    def param_grid(self, runs=1):
        """ Returns a generator of dictionaries with all the possible parameter combinations
            the keys are the parameter names the values are the current value for each parameter.

        Warnings:
            you shouldn't use id and run as parameter names, param grid automatically uses those names to identify each
            parameter combination.

        Args:
            runs(int): number of repeats for each unique configuration
        """
        if runs < 1:
            raise ValueError("runs must be >0: runs set to {}".format(runs))

        param_values = []
        params = list(self.params.keys())

        for param in self.params.keys():
            param_type = self.params[param]["type"]
            param_type = Types.from_str(param_type)
            param_value = self.get_param(param, param_type)
            if param_type == Types.VALUE:
                param_value = [param_value]
            param_values.append(param_value)

        if runs < 1:
            raise ValueError("runs must be >0: runs set to {}".format(runs))
        # add run to parameter names and run number to parameters
        params.append("run")
        run_ids = np.linspace(1, runs, runs, dtype=np.int32)
        param_values.append(run_ids)

        param_product = itertools.product(*param_values)
        param_product = (dict(zip(params, values)) for values in param_product)

        # create ids for each unique configuration
        ids = np.linspace(0, self.size - 1, self.size, dtype=np.int32)
        ids = list(_repeat_it(ids, runs))
        id_param_names = ["id"] * (self.size * runs)

        id_params = [{k: v} for k, v in zip(id_param_names, ids)]

        param_product = ({**p, **i} for i, p in zip(id_params, param_product))
        return param_product

    def write(self, filename):
        with open(filename, "w") as f:
            toml.dump(self.params, f)

    def write_configs(self, output_path="params.csv"):
        """ Writes a csv file with each line containing a configuration value with a unique id for each configuration

            Args:
                output_path: the output path for the summary file
        """
        summary_header = ["id", "run"]
        summary_header += self.param_names()

        with open(output_path, mode="w", newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=summary_header)
            writer.writeheader()

            param_grid = self.param_grid()
            for param_row in param_grid:
                writer.writerow(param_row)

    def write_config_files(self, output_path="params", file_prefix="params"):
        """ Writes one configuration file per unique configuration in the grid space
        Args:
            output_path:
            file_prefix:
        """
        param_grid = self.param_grid()
        conf_id = 0
        for current_config in param_grid:
            conf_file = "{prefix}_{id}.conf".format(prefix=file_prefix, id=conf_id)
            conf_file = os.path.join(output_path, conf_file)

            with open(conf_file, "w") as f:
                toml.dump(current_config, f)

            conf_id += 1
