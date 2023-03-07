# Python wrapper around C implementation of EVPI

This is a Python wrapper (using CFFI) around a C implementation of the expected value of perfect information (EVPI).

## Installation

Make sure, your working directory is the package root (where the `setup.py`) is in. You will need a C compiler (like `gcc`) to build the package from source.

Plain setuptools:

```sh
python setup.py install
```

Or via pip:

```sh
pip install .
```

## Tests

```
pytest
```
