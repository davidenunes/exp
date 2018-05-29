import plac
from exp.run import cli

if __name__ == '__main__':
    plac.Interpreter.call(cli.RunCli)
