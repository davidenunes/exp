import sys
import plac
from plac import Annotation
import os
from exp.params import ParamSpace
import multiprocessing as mp

class RunCli:
    commands = ['gpus']

    @plac.annotations(
        runnable=Annotation("path to python script with a run method that receives params"),
        params=Annotation("path to param space file", 'option'),
        name=Annotation("name for this run")
    )
    def gpus(self, runnable, params, name):
        if not os.path.exists(params) and not os.path.isfile(params):
            print("Parameter space file not found: {path}".format(path=params), file=sys.stderr)
            sys.exit(1)

        ps = ParamSpace(filename=params)
        ps.write_grid_summary(name + '_params.csv')

        param_grid = ps.param_grid(include_id=True, id_param="id")

        # distributed work fir processes
        # wait for one to finish


    def __missing__(self, name):
        print("command not found {cmd}".format(cmd=name))
if __name__ == '__main__':
    plac.Interpreter.call(RunCli)

