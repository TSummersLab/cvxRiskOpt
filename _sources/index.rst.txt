.. cvxRiskOpt documentation master file, created by
   sphinx-quickstart on Fri Apr 26 20:31:19 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to cvxRiskOpt's documentation!
======================================
cvxRiskOpt is a Python package for risk-based optimization using CVXPY and CVXPYgen.

Risk-based optimization problems can be hard to code and they may require looking up reformulations.
To encourage risk-based optimization problems and help speed up the development cycle, we created cvxRiskOpt.

cvxRiskOpt provides users with tools to automatically generate risk-based optimization problems and risk-based constraints. cvxRiskOpt integrates with CVXPY directly resulting in CVXPY Problems and constraints. These also enables using CVXPYgen directly allowing users to automatically generate C-code which can be used on embedded systems or utilized with the python-wrapper to speed up solving the optimization problems.


.. toctree::
   :hidden:

   install/index


.. toctree::
   :maxdepth: 4

   modules.rst


.. toctree::
   :hidden:

   examples/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
