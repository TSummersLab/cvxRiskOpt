# CVXPY Risk Optimization

A package for risk-based optimization using CVXPY and CVXPYgen.

## Installation

### Installing from PyPI

The package can be installed using pip:

```bash
pip install cvxRiskOpt
```

Notes:

- The installation will also include cvxpy and cvxpygen.
- Please refer to cvxpy's documentation for [installing additional solvers](https://www.cvxpy.org/install/).
- Compiling code with Clarabel requires `Rust`, `Eigen`, and `cbindgen`. (e.g. These can be installed with `homebrew` on MacOS)

### Installing from source

- Clone/Download the package
- Create and activate conda env

```bash
conda create --name cvxRiskOpt python=3.10 pip -y
conda activate cvxRiskOpt
```

- Install dependencies (see `setup.py`)
- Install the package

```bash
python3 -m pip install -e .
```

## Tests

To run tests, execute the following from the root of the package.

```bash
pytest
```

## Examples

There are several examples in `examples` demonstrating the usage of the package.
