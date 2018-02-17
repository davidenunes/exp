import plac
from exp.grid import cli

if __name__ == '__main__':
    plac.Interpreter.call(cli.GridCli)
