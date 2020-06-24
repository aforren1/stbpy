import setuptools
import sys
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np
from sys import platform

defs = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
inc_path = np.get_include()

ext = [Extension('stb.image',
                 sources=['stb/image.pyx'],
                 include_dirs=[inc_path],
                 define_macros=defs)]

with open("README.md", "r") as f:
    long_description = f.read()

# with open('requirements.txt') as f:
#     requirements = f.read().splitlines()

setuptools.setup(
    name="stb",
    version="0.0.1",
    #install_requires=requirements,
    author="Alex Forrence",
    author_email="alex.forrence@gmail.com",
    description="",
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
