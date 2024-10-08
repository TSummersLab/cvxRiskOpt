from setuptools import setup

setup(
    name="cvxRiskOpt",
    version="0.2.0",
    description="Risk-Based Optimization tool using CVXPY and CVXPYgen",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Sleiman Safaoui, Tyler Summers",
    author_email="snsafaoui@gmail.com, tyler.summers@utdallas.edu",
    maintainer="Sleiman Safaoui",
    maintainer_email="snsafaoui@gmail.com",
    url="https://github.com/TSummersLab/cvxRiskOpt",
    license="Apache-2.0",
    keywords=["dro", "risk", "optimization", "robust", "code generation", "mpc"],
    python_requires=">=3.10",
    install_requires=[
        "scipy>=1.11.4",
        "numpy>=1.26.2",
        "matplotlib>=3.8.0",
        "cvxpygen>=0.3.5",
        "cvxpy>=1.4.3",
    ],
    extras_require={
        "dev": ["pytest==7.4.0", "polytope==0.2.5", "Sphinx==7.2.6"],
    },
)
