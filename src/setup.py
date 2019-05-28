# compile with
# python setup.py build_ext --inplace

import setuptools
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["binarizer.pyx", "smear_test.pyx"], build_dir="build", annotate=True),
	                               script_args=['build'], 
	                               options={'build':{'build_lib':'.'}
	                               }
)