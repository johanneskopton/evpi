import os
from setuptools import setup, Extension


def read(fname):
    # Utility function to read the README file
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


evpi_module = Extension("evpi_cffi",
                        sources=["evpi_cffi/evpi.c"],
                        include_dirs=['evpi_cffi'])

setup(
    name="evpi_cffi",
    version="0.1.0",
    author="Johannes Kopton",
    author_email="johannes@kopton.org",
    description=("Python wrapper for EVPI implementation in C."),
    keywords="bayesian information economics value decision analysis",
    url="https://github.com/johanneskopton/evpi",
    packages=["evpi_cffi"],
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3"
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy >= 1.19",
        "cffi >= 1.15"
    ],
    setup_requires=[
        "cffi >= 1.15"
    ],
    extras_require={
        "dev": [
            "autopep8",
            "flake8",
            "pytest"
        ],
    },
    # ext_modules=[evpi_module],
    cffi_modules=["evpi_cffi/build_evpi_cffi.py:ffibuilder"],
)
