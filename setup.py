import setuptools
import sys
from Cython.Build import cythonize
from setuptools.extension import Extension
from sys import platform

extra_compile_args = []
if sys.platform == 'linux':
    extra_compile_args.append('-g0')

ext = [Extension('stb.image',
                 sources=['stb/image.pyx'],
                 extra_compile_args=extra_compile_args)]

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="stbpy",
    version="0.0.5",
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
    ext_modules=cythonize(ext,
                          compiler_directives={'language_level': 3},
                          annotate=True)
)
