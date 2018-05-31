# Provides a very simple dictionary with default values and type conversion
# did not include validation of options
from collections import namedtuple


class Param(namedtuple('Param', ["typefn", "value", "options"])):
    def __new__(cls, typefn, value=None, options=None):
        if options is not None:
            options = set(options)
        typefn = convert_type(typefn)

        if value is not None:
            value = typefn(value)

        if options is not None and value is not None:
            if value not in options:
                raise ValueError("Invalid Param Value: {} not in options {}".format(value, options))

        return super().__new__(cls, typefn, value, options)


class defargs(dict):
    def __init__(self, defaults):
        # key -> (typefn,value) e.g. "a": (int,2)
        self.defaults = dict()

        self.add_params(defaults)

        for arg in self.defaults:
            param = self.defaults[arg]
            dict.__setitem__(self, arg, param.value)

        self.__dict__ = self

    def __setitem__(self, key, val):
        if key in self.defaults:
            default = self.defaults[key]
            if val is None:
                val = default.value
            else:
                val = default.typefn(val)
                if default.options is not None:
                    if val not in default.options:
                        raise ValueError("Invalid Param Value: {} not in options {}".format(val, default.options))

        dict.__setitem__(self, key, val)

    def add_params(self, param_dict):
        for arg in param_dict:
            param = Param(*param_dict[arg])
            self.defaults[arg] = param

    def from_dict(self, args):
        for arg in args:
            self.__setitem__(arg, args[arg])


def as_bool(v):
    if v is None:
        return False
    elif isinstance(v, bool):
        return v
    elif isinstance(v, str):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise TypeError('Boolean value expected.')
    elif isinstance(v, int) or isinstance(v, float):
        if v <= 0:
            return False
        else:
            return True


def as_int(v):
    return int(float(v))


def convert_type(typeclass):
    if typeclass == bool:
        return as_bool
    elif typeclass == int:
        return as_int
    else:
        return typeclass
