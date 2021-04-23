from setuptools import setup
from Cython.Build import cythonize
import sys

# compile and link flags
if sys.platform.startswith('win'):
    # to compile this package on windows you need MSVC
    compile_args = ['/openmp', '/fp:fast']
    link_args = ['/openmp']
elif sys.platform.startswith('darwin'):
    # use real gcc (like gcc-9) and not clang on mac
    compile_args = ['-fopenmp', '-ffast-math']
    link_args = ['-lomp']
else:
    # linux
    compile_args = ['-fopenmp', '-ffast-math']
    link_args = ['-fopenmp']

setup(
    name='shrink',
    version='0.1',
    ext_modules=cythonize(
        ['shrink/shrink.pyx'],
        aliases={'XCOMPARGS': compile_args, 'XLINKARGS': link_args}
    ))
