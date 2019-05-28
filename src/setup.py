# compile with
# python setup.py build_ext --inplace

import setuptools
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(["binarizer.pyx", "smear_test.pyx", "line_segmenter.pyx"], build_dir="build", annotate=True),
	                               script_args=['build'], 
	                               include_dirs=[numpy.get_include()],
	                               options={'build':{'build_lib':'.'}
	                               }
)