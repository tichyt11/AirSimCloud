import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

ext_modules = [
    Extension("theta_star",  ["theta_star.pyx"]),
    Extension("fast_heap",  ["fast_heap.pyx"]),
]

setup(
    name='My Program',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules, include_dirs=[numpy.get_include()]
)