#!/usr/bin/env python

from distutils.core import setup

setup(name='exp',
      version='0.2',
      description='Python tool do design and run experiments with global optimisation and grid search',
      author='Davide Nunes',
      author_email='davide@davidenunes.com',
      packages=['exp'],
      install_requires=[
          'tabulate',
          'toml',
          'plac',
          'click',
          'numpy',
          'matplotlib']
      )
