#!/usr/bin/env python
'''setup.py'''

from glob import glob
from os.path import basename
from os.path import splitext
from setuptools import setup, find_packages
setup(
  author='Blas',
  name='gpu-image-corr',
  version='0.2.4',
  packages=find_packages('src'),
  package_dir={'': 'src'},
  py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
  scripts=['src/main.py'],
  entry_points={
    'console_scripts': [
      'gpu-image-corr=main:main',
    ],
  },
)
