#!/usr/bin/env python

from distutils.core import setup

setup(name='exp',
      version='0.1',
      description='Python experiment running utils with support for SGE grids',
      author='Davide Nunes',
      author_email='davidelnunes@gmail.com',
      packages=['exp'],
      install_requires=[
          'tabulate',
          'pygments',
          'prompt-toolkit'
      ]
)
