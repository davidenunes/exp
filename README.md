# EXP: A python experiment management toolset

A simple set of utilities to automate experiment definition, submission and monitoring.
Support for Sun Grid Engine (SGE) jobs

## qsub file generation
It contains utilities to generate files that can be submitted using ``qsub``. These utils help
to create submission scripts that configure SGE parameters (like resource requesting), load modules
if ``module load`` is used, activate virtual environments using either ``conda`` or ``virtualenv``.

## job monitoring
Besides creating ``qsub`` jobs, this library can be imported by the scripts that will be run on the grid (provided it is
installed on the respective virtual environment). It allows to log
job progress in a ``sqlite`` file which can then be used in a client machine to list and measure
the progress on running jobs.