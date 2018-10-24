""" Simple parameter Dictionary with default values,
    option constraints, and type conversion

"""
from collections import namedtuple


class Param(namedtuple('Param', ["typefn", "value", "options"])):
    """ Simple class representing parameters

    with values, validated/restricted by options
    and be converted from other values using a type function
    used to read parameter tuples that serve as entries to :obj:`ParamDict`

    """

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


class Namespace(object):
    def __init__(self, dict_attr):
        if not isinstance(dict_attr, dict):
            raise TypeError("Namespace requires a dict, {} found".format(dict_attr))

        for k, v in dict_attr.items():
            self.__setattr__(k, v)

    def __str__(self):
        attr = ["{}={}".format(k, v) for k, v in self.__dict__.items()]

        return "Namespace({s})".format(s=','.join(attr))

    def __setattr__(self, key, value):
        super().__setattr__(key, value)


class ParamDict(dict):
    """ Dictionary of parameters with default values

    Attributes:
        defaults = a dictionary :py:`str` -> :obj:`Param`

    """

    def __init__(self, defaults):
        self.defaults = dict()
        self.add_params(defaults)

        for arg in self.defaults:
            param = self.defaults[arg]
            dict.__setitem__(self, arg, param.value)

        # self.__dict__ .update(self)

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
        """ Adds a set of parameter values from a given dictionary to the current values
        overwrites the default values for the parameters that already exist in the defaults

        Args:
            param_dict: a dictionary with param_name -> (type,vale,options) :obj:`Param`
        """
        for arg in param_dict:
            param = Param(*param_dict[arg])
            self.defaults[arg] = param

    def from_dict(self, args):
        for arg in args:
            self.__setitem__(arg, args[arg])

    def to_namespace(self):
        """ Converts the ParamDict to a :obj:`Namespace` object
        which allows you to access ``namespace.param1``

        Returns:
            a :obj:`Namespace` object with the current values of this parameter dictionary
        """
        return Namespace(self)


def as_bool(v):
    """ Converts a given value to a boolean

    Args:
        v (int,str,bool): and integer, string, or boolean to be converted to boolean value.
    Returns:
        (bool): if the value is an int any value <= 0 returns False, else True
                if the value is a boolean simply forwards this value
                if the value is a string, ignores case and converts any (yes,true,t,y,1) to True
                and ('no', 'false', 'f', 'n', '0') to False
    """
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
    """ Converts a value to int by converting it first to float

    because calling ``int("2.3")`` raises a ValueError since 2.3 is
    not an integer. ``as_int("2.3")`` returns the same as ``int(2.3)``

    Args:
        v: a value convertible to numerical

    Returns:
        (int) the integer value of the given value

    """
    return int(float(v))


def convert_type(type_class):
    """ Maps classes to convert functions present in this module
    Args:
        type_class: some class that can also be called to convert values into its types


    Returns:
        a type conversion function for the given class capable of converting more than literal values, for instance,
        requesting a type conversion class for boolean, returns a function capable of converting strings, or integers
        to a boolean value (see :obj:`as_bool`)
    """
    if type_class == bool:
        return as_bool
    elif type_class == int:
        return as_int
    else:
        return type_class
