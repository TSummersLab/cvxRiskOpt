
Production Optimization
=======================

Consider a simple production optimization problem where the optimal production amount :math:`x` is to be determined subject to uncertainty in the demand :math:`\mathbf{d}` which is represented by a random variable.

The optimization problem is given by:

.. math::

    \begin{align*}
    \min_x \quad & c \cdot x \\
    \text{subject to} \quad & \mathbb{P}(x \geq \mathbf{d}) \geq 1 - \epsilon
    \end{align*}

Where :math:`c` is the unit cost of the product, :math:`\mathbb{P}` measures the probability of an event, and :math:`\epsilon \in (0, 0.5]` is the risk bound.

Assume that :math:`\mathbf{d}` follows a Gaussian distribution with mean :math:`\overline{d}` and variance :math:`\sigma^2`: :math:`\mathbf{d} \sim \mathcal{N}(\overline{d}, \sigma^2)`.

The chance constraint can be reformulated into a deterministic constraint as follows:

.. math::

    \begin{align*}
    \mathbb{P}(x \geq \mathbf{d}) \geq 1 - \epsilon & \iff \mathbb{P}(-x + \mathbf{d} \leq 0) \geq 1 - \epsilon \\
    & \iff -x + \overline{d} + \sigma \Phi^{-1}(1-\epsilon) \leq 0 \\
    & \iff x \geq \overline{d} + \sigma \Phi^{-1}(1-\epsilon)
    \end{align*}

where :math:`\Phi^{-1}` is the inverse CDF of the standard normal Gaussian distribution.

Example
-------

In the following code, we solve the production optimization problem with CVXPY and cvxRiskOpt and with CVXPY only.
The main difference between using cvxRiskOpt and not doing so is in the inclusion of the chance constraint.

With cvxRiskOpt we only need to rearrange the chance constraint:

.. math::

    \begin{align*}
    & \mathbb{P}(-x + \mathbf{d} \leq 0) \geq 1 - \epsilon \iff \mathbb{P}(a \mathbf{\xi_1} + b + \mathbf{\xi_2} \leq 0) \geq 1 - \epsilon\\
    \rightarrow & a = 1, \ \mathbf{\xi_1} = \mathbf{d}, \ b=-x, \ \mathbf{\xi_2} = 0
    \end{align*}

However, using CVXPY only, we need to reformulate it into the deterministic constraint as show earlier.

.. code-block:: python

    import cvxpy as cp
    import numpy as np
    c = 10  # cost
    x = cp.Variable(name='x')  # decision variable
    d_mean = 700  # demain mean
    d_var = 30  # demand variance
    eps = 0.1  # risk bound

    # cvxpy problems
    objective = cp.Minimize(c * x)

    # cvxpy + cvxRiskOpt
    from cvxRiskOpt.cclp_risk_opt import cclp_gauss
    cc_contr = cclp_gauss(eps, a=1, b=-x,
                          xi1_hat=d_mean,
                          gam11=d_var)
    constraints_with_cro = [x >= 0, cc_contr]
    prob_with_cro = cp.Problem(objective, constraints_with_cro)
    prob_with_cro.solve(solver=cp.CLARABEL)
    print("Production amount (CVXPY + cvxRiskOpt): ", x.value)

    # cvxpy only
    from scipy.stats import norm
    d_std_div = np.sqrt(d_var)
    constraints = [x >= 0, x >= d_mean + d_std_div * norm.ppf(1-eps)]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)
    print("Production amount (CVXPY only): ", x.value)


.. parsed-literal::

    Production amount (CVXPY + cvxRiskOpt):  707.0193468163586
    Production amount (CVXPY only):  707.0193468163586

Another benefit or using cvxRiskOpt is that if the Guassian assumption about the noise were to change, updating the chance constraint can easily be done by changing the `cclp_gauss` call to call another `cclp_risk_opt` function.
Below is an example using a DR-VaR risk metric where the probability of meeting the demand must be realized under the work case distribution in a moment-based ambiguity set using the mean and covariance of the uncertainty

.. code-block:: python

    from cvxRiskOpt.cclp_risk_opt import cclp_dro_mean_cov
    dr_cc_contr = cclp_dro_mean_cov(eps, a=1, b=-x,
                                    xi1_hat=d_mean,
                                    gam11=d_var)
    dr_constraints_with_cro = [x >= 0, dr_cc_contr]
    dr_prob_with_cro = cp.Problem(objective, dr_constraints_with_cro)
    dr_prob_with_cro.solve(solver=cp.CLARABEL)
    print("DR Production amount (CVXPY + cvxRiskOpt): ", x.value)

.. parsed-literal::

    DR Production amount (CVXPY + cvxRiskOpt):  716.4316765283791