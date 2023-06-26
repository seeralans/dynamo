from setuptools import setup, find_packages

setup(name='dynamo',
      version='0.0.1',
      packages=find_packages(include=["dynamo", 'dynamo.*']),
      description='A package for dynamical modelling of protein structures',')
