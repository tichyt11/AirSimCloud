from setuptools import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    name='app',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize("*.pyx", annotate=True), include_dirs=[numpy.get_include(), '.']
)
