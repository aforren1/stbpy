import setuptools
import sys
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np
from sys import platform

ext = [Extension('stb.image',
                 sources=['stb/image.pyx'],
                 include_dirs=[np.get_include()],
                 define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])]

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="stbpy",
    version="0.0.2",
    install_requires=['numpy'],
    author="Alex Forrence",
    author_email="alex.forrence@gmail.com",
    description="Cython wrapper for nothings/stb",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aforren1/stbpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    ext_modules=cythonize(ext, compiler_directives={'language_level': 3}, annotate=True)
)
