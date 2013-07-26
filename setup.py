#!/usr/bin/env python

from setuptools import setup, Extension

import sys

# what I *should* do is either make our use of pthreads work on Mac OS X or
# at least do the OS configuation in some structured way that I'm sure setuptools
# allows. But instead, I'm just hackishly excluding the extension if the OS seems to 
# be a Mac. 

extension_modules = []
setup(
    name="shiver",
    description="A multi-threaded work queue for functions compiled with llvmpy",
    long_description='''
Shiver 
=========

A multi-threaded work queue for functions compiled with llvmpy. Give Shiver a function whose last argument is an index (or multiple indices) and an iteration space (i.e. a number of iterations, a tuple of integers, or even slice objects with start/stop/step fields), and shiver does all the messy plumbing of running your code in parallel.
''',
    classifiers=['Development Status :: 3 - Alpha',
                 'Topic :: Software Development :: Libraries',
                 'License :: OSI Approved :: BSD License',
                 'Intended Audience :: Developers',
                 'Programming Language :: Python :: 2.7',
                 ],
    author="Alex Rubinsteyn",
    author_email="alexr@cs.nyu.edu",
    license="BSD",
    version="0.12",
    url="http://github.com/iskandr/shiver",
    packages=[ 'shiver' ],
    package_dir={ '' : '.' },
    requires=[
      'llvmpy', 
      'numpy', 
      'scipy',
    ],
    ext_modules = extension_modules)
