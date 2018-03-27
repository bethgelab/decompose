#!/usr/bin/env python
import sys
from setuptools import setup, find_packages
from os import path
from decompose.version import __version__

if sys.version_info < (3, 6, 0):
    sys.stderr.write("ERROR: You need Python 3.6 or later to use decompose.\n")
    exit(1)

description = 'Blind source separation based on the probabilistic tensor ' \
              'factorisation framework'
long_description = '''
Decompose -- Blind source separation framework
==============================================

DECOMPOSE is a probabilistic blind source separation framework that
can be flexibly adjusted to the data, is extensible and easy to use,
adapts to individual sources and handles large-scale data through
algorithmic efficiency. DECOMPOSE encompasses and generalises many
traditional BSS algorithms such as PCA, ICA and NMF.
'''.lstrip()

here = path.abspath(path.dirname(__file__))

setup(
    name='decompose',
    version=__version__,
    description=description,
    long_description=long_description,
    url='https://github.com/bethgelab/decompose',
    author='Alexander Boettcher',
    author_email='alexander.boettcher@bethgelab.org',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='data-analysis machine-learning blind-source-separation',
    packages=find_packages(),
    install_requires=['tensorflow>=1.6',
                      'mypy'],
    extras_require={
        'test': ['pytest'],
    },

    project_urls={
        'Bug Reports': 'https://github.com/bethgelab/decompose/issues',
        'Source': 'https://github.com/bethgelab/decompose/',
    },
)
