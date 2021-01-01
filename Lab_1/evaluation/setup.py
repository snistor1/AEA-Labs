import numpy

from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
        Extension('evaluation_cy',
                  sources=["evaluation_cy.pyx"],
                  include_dirs=[numpy.get_include()]
                  )
        ])
)
