.. _install:

Install
=======

cxvRiskOpt supports Python 3.10 and higher.
It has been tested on: macOS.
You can use pip for installation or install from source.
You may want to isolate your installation in a `virtualenv <https://virtualenv.pypa.io/en/stable/>`_,
or a `conda environment <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`_.

Installation with pip
---------------------

    Install cvxRiskOpt using `pip <https://pip.pypa.io/>`_:
      ::

          pip install cvxRiskOpt

Installation from source
------------------------
        We recommend using a fresh virtual environment (virtualenv or conda) when installing cvxRiskOpt from source.

        cvxRiskOpt has the following dependencies:

         * Python >= 3.10
         * scipy>=1.11.4
         * numpy>=1.26.2
         * matplotlib>=3.8.0
         * cvxpygen>=0.3.4
         * cvxpy>=1.4.1

        Perform the following steps to install CVXPY from source:

         1. Clone the official `cvxRiskOpt git repository <https://github.com/TSummersLab/cvxRiskOpt>`_.
         2. Navigate to the top-level of the cloned directory.
         3. If you want to use cvxRiskOpt with editable source code, run
            ::

                pip install -e .

            otherwise, run
            ::

                pip install .

