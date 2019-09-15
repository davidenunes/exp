<p align="center">
  <a href="https://github.com/davidenunes/exp" target="_blank">
    <img width="200"src="https://raw.githubusercontent.com/davidenunes/exp/master/extras/exp.png">
  </a>
</p>
<p align="center">Experiment <strong>design</strong>, <strong>deployment</strong>, and <strong>optimization</strong></p>
	
	
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)


EXP is a python experiment management toolset created to simplify two simple use cases: design and deploy experiments in the form of python modules/files.

An experiment is a series of runs of a given configurable module for a specified set of parameters. This tool covers one of the most prevalent experiment deployment scenarios: testing a set of parameters in parallel in a local machine or homogeneous cluster. EXP also supports [global optimization](https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf) using **gaussian processes** or other surrogate models such as **random forests**. This can be used for instance as a tool for **hyperoparameter tuning** for machine learning models.

## Features
* **parameter space design** based on configuration files ([TOML](https://github.com/toml-lang/toml) format);
* **parallel experiment deployment** using ``multiprocessing`` processes;
* **CUDA gpu workers** one parallel process per available GPUs: uses the variable [CUDA_VISIBLE_DEVICES](https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices);
* **global optimization** from parameter spaces (e.g. for hyperparameter tunning) using [scikit-optimize](https://scikit-optimize.github.io/).

## Found this useful?
<a href='https://ko-fi.com/Y8Y0RZO6' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://az743702.vo.msecnd.net/cdn/kofi4.png?v=2' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

## Installation
``pip install exp`` 

``pipenv install exp`` with [pipenv](https://pipenv.readthedocs.io/en/latest/install/#pragmatic-installation-of-pipenv)

## Available CLI tools
EXP provides two CLI modules:
* exp.run:  ``python -m exp.run -p basic.conf -m runnable.py --workers 10``
* exp.gopt:``python -m exp.gopt -p basic.conf -m runnable.py --workers 4 -n 100 --plot``

for more information check each commands help:

``python -m exp.run -h``

## Getting Started: Optimization

### 1. Runnable Module
The first step is to create a module to use in our experiments. A basic configurable module ``runnable.py`` looks like this:

```python
def run(x=1, **kwargs):
    return x ** 2
```

This module computes the square of a parameter ``x``. Note that ``kwargs`` is included to capture other parameters that the experiment runner might use (even if they are not used by your module). Since run receives a dictionary, you could also define it as follows.

```python
def run(**kwargs):
    x = kwargs.get('x',1)
    return x ** 2
```

### 2. Parameter Space Definition
Next, we need a configuration file ``basic.conf`` were the parameters are specified:
```markdown
[x]
type = "range"
bounds = [-10,10]
```
This defines a parameter space with a single parameter ``x`` with values in the range ``[-10,10]``. For how to specify parameter spaces, see the [Parameter Space Specification](#parameter-space-specification).

### 3. Module Optimization
Our simple module returns the ``x**2``, the optimizer tries to find the minimum value of this function based on the parameter space given by the configuration file. In this case, the optimizer will look at values of ``x`` between ``[-10,10]`` and try to find the minimum value.

```bash
python -m exp.gopt --params basic.conf --module runnable.py --n 20 --workers 4
```

<p align="center">
    <img src="https://raw.githubusercontent.com/davidenunes/exp/master/extras/getting_started.gif">
</p>

finds a solution very close to ``0``. By default, the optimizer assumes a range defines the boundaries of a real-valued variable. If you wish to optimize discrete integers use the following specification:

```markdown
[x]
type = "range"
bounds = [-10,10]
dtype = "int"
``` 
The optimizer will explore discrete values between -10 and 10 inclusively. Also, using the ``--plot`` flag displays a real-time **convergence plot** for the optimization process.

<p align="center">
    <img src="https://raw.githubusercontent.com/davidenunes/exp/master/extras/convergence.png">
</p>

which in this case converges immediately because the function to be optimized is quite simple, but the goal is to optimize complex models and choosing from a large set of parameters without having to run an exhaustive search through all the possible parameter combinations.

## Parameter Space Specification
Parameter space files use [TOML](https://github.com/toml-lang/toml) format, I recommend taking a look at the specification and getting familiar with how to define values, arrays, etc. ParamSpaces in EXP has **4 types of parametes**, namely:
* **value**: single value parameter;
* **range**: a range of numbers between bounds;
* **random**: a random *real/int* value between bounds;
* **list**: a list of values (used for example to specify categorical parameters);

Bellow, I supply an example for each type of parameter:

### Value
Single value parameter.
```python
# this a single valued parameter with a boolean value
[some_param]
type = "value"
value = true
```
### Range
A parameter with a set of values within a range.
```python
# TOML files can handle comments which is useful to document experiment configurations
[some_range_param]
type = "range"
bounds = [-10,10]
step = 1 	 	# this is optional and assumed to be 1
dtype = "float"   # also optional and assumed to be float
```
The commands ``run`` and ``gopt`` will treat this parameter definition differently. The optimizer will explore values within the bounds including the end-points. The runner will take values between ``bounds[0]`` and ``bounds[1]`` excluding the last end-point (much like a python range or numpy arange).

The ``dtype`` also influences how the optimizer looks for values in the range, if set to ``"int"``, it explores discrete integer values within the bounds; if set to ``"float"``, it assumes the parameter takes a continuous value between the specified bounds.

### Random
A parameter with ``n`` random values sampled from "uniform" or "log-uniform" between the given bounds. If used with ``run``, a parameter space will be populated with a list of random values according to the specification. If used with ``gopt``, ``n`` is ignored and bounds are used instead, along with the prior.

For optimization purposes, this works like range, except that you can specify the prior which can be "uniform" or "log-uniform", range assumes that the values are generated from "uniform" prior, when the parameter is used for optimization.

The other difference between parameter grids and optimization is that the bounds do not include the end-points when generating parameter values for grid search. The optimizer will explore random values within the bounds specified, including the high end-point.

```python
[random_param]
type="random"
bounds=[0,3]    # optional, default range is [0,1]
prior="uniform" # optional, default value is "uniform"
dtype="float"   # optional, default value is "float"
n=1             # optional, default value is 1 (number of random parameters to be sampled)
```
### List
A list is just an homogeneous series of values a parameter can take.
```python
[another_param]
type="list"
value = [1,2,3]
```
The array in ``"value"`` must be homogenous, something like ``value=[1,2,"A"]`` would throw a *Not a homogeneous array* error. List parameters are treated by ``gopt`` command as a **categorical** parameter. This is encoded using a *one-hot-encoding* for optimization.

Also, for optimization purposes, a list is treated like a set, if you provide duplicate values it will only explore the unique values. For example if you want to specify a boolean parameter, use a list:

```python
[some_boolean_decision]
type="list"
value = [true,false]
```

# Library Modules
EXP also provides different tools to specify param spaces programmatically
## ParamSpace
The ``exp.params.ParamSpace`` class provides a way to create parameter spaces and iterate over all the possible 
combinations of parameters as follows: 
```python
>>>from exp.params import ParamSpace
>>>ps = ParamSpace()
>>>ps.add_value("p1",1)
>>>ps.add_list("p2",[True,False])
>>>ps.add_range("p3",low=0,high=10,dtype=int)
>>>ps.size
20
```

```python
grid = ps.param_grid(runs=2)
```
``grid`` has ``2*ps.size`` configurations because we repeat each configuration ``2`` times (number of runs). Each configuration dictionary includes 2 additional parameters ``"id"`` and ``"run"`` which are the unique configuration id and run id respectively.

```python
for config in grid:
	# config is a dictionary with the params of a unique configuration in the parameter space
	do_something(config)
```

## ParamDict & Namespace
``ParamDict`` from ``exp.args`` module is a very simple dictionary where you can specify default values for different parameters. ``exp.args.Param`` is a named tuple: ``(typefn,default,options)`` where ``typefn`` is a type function like ``int`` or ``float`` that transforms strings into values of the given type if necessary, ``default`` is a default value, ``options`` is a list of possible values for the parameter.

This is just a very simple alternative to using argparse with a lot of of parameters. Example of usage:

```python
from exp.args import ParamDict,Namespace

# these are interpreted by a ParamDict as a exp.args.Param named tuple
param_spec = {
    'x': (float, None),
    'id': (int, 0),
    'run': (int, 1),
    'cat': (str, "A", ["A","B"])
}

def run(**kargs):
    args = ParamDict(param_spec) # creates a param dict from default values and options
    args.from_dict(kargs)        # updates the dictionary with new values where the parameter name overlaps
    ns = args.to_namespace()     # creates a namespace object so you can access ns.x, ns.run etc
    ...
```

Another nice thing is that there is basic type conversions from string to boolean, int, float, etc. Depending
on the arguments received in ``kwargs``, ``ParamDict`` converts the values automatically according to the parameter
specifications.

## Created by
**[Davide Nunes](https://github.com/davidenunes)**

## Licence

[Apache License 2.0](LICENSE)
