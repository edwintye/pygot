#!/bin/env python
'''
Created on 18th of Feb, 2015

@author: Edwin Tye (Edwin.Tye@gmail.com)
'''
from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='pygotools',
      version='0.1.0',
      description='Global Optimization Toolbox',
      long_description=readme(),
      author="Edwin Tye",
      author_email="Edwin.Tye@phe.gov.uk",
      packages=[
                'pygotools',
                'pygotools.direct',
                'pygotools.gradient',
                'pygotools.convex',
                'pygotools.optutils'
                ],
      license='LICENCE.txt',
      install_requires=[
                        'scipy',
                        'numpy',
                        'cvxopt',
                        'enum34'],
      test_suite='nose.collector',
      tests_require=[
                     'nose',
                     'numpy',
                     'scipy'
                     ],
      scripts=[]
      )