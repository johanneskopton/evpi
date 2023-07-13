import os
from setuptools import setup


def read(fname):
    # Utility function to read the README file
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="py_evpi",
    version="1.0.0",
    author="Johannes Kopton",
    author_email="johannes@kopton.org",
    description=("EVPI implementation in Python using Numpy."),
    keywords="bayesian information economics value decision analysis",
    url="https://github.com/johanneskopton/evpi",
    packages=["py_evpi"],
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3"
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy >= 1.19",
        "scipy >= 1.10"
    ],
    extras_require={
        "dev": [
            "autopep8",
            "flake8",
            "matplotlib>3.6",
            "pandas",
            "pygam >= 0.8",
            "numpy < 1.24",  # for pygam
            "pytest"
        ],
    },
)
