#!/usr/bin/env python

from distutils.core import setup

setup(name='exp',
      version='0.3',
      description='Python tool do design and run experiments with global optimisation and grid search',
      licence='apache-2.0',
      author='Davide Nunes',
      author_email='mail@davidenunes.com',
      url = 'https://github.com/davidenunes/exp',
      download_url = 'https://github.com/davidenunes/exp/archive/v0.3.tar.gz',
      packages=['exp'],
      install_requires=[
          'toml',
          'click',
          'numpy',
          'matplotlib']
      )
