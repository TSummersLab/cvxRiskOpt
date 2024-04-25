# CVXPY Risk Optimization

A package for risk-based optimization using CVXPY and CVXPYgen.

## Installation

- create and activate conda env 
```
conda create --name cvxRiskOpt python=3.10 pip -y
conda activate cvxRiskOpt
```
- install dependencies
```
pip install cvxpy>=1.4.1
pip install cvxpygen>=0.3.4
pip install scipy>=1.11.4
pip install numpy>=1.26.2
pip install matplotlib>=3.8.0
```
For development, also install:
```
pip install pytest==7.4.0
pip install polytope==0.2.5
pip install Sphinx==7.2.6
```
- install the package
```
python3 -m pip install -e .
```

## Tests
To run tests,
```
pytest
```

## Examples
There are several examples in `examples` demonstrating the usage of the package's functionality.
