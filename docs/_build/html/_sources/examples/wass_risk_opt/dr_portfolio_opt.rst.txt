
Distributionally Robust Mean-CVaR Portfolio Optimization using Wasserstein-Based Ambiguity Sets
===============================================================================================

Consider the distributionally robust portfolio optimization problem described in Section 7.1 of [Esfahani2018]_:

.. math::

    \begin{align*}
        \sup_{\mathbb{Q} \in \mathbb{B}_\epsilon(\hat{\mathbb{P}}_N)} \inf_{x \in \mathbb{X}} \left\{\mathbb{E}^\mathbb{Q}[-\langle x, \xi \rangle] + \rho \text{CVaR}_\alpha^\mathbb{Q}(-\langle x, \xi \rangle) \right\}
    \end{align*}

where :math:`\langle \cdot, \cdot \rangle` is the inner product, :math:`\hat{\mathbb{P}}` is the empirical distribution based on data, :math:`\xi = [\xi_1, \dots \xi_m]^T` are the market values of :math:`m` assets and :math:`x` is the portfolio with :math:`\mathbb{x} = \{x \in \mathbb{R}_+^m, \sum_i x_i = 1\}`.

Using the convex optimization-based definition of :math:`\text{CVaR}`, this portfolio optimization problem can be rewritten as a worst-case expectation involving the max of affine terms. This worst-case optimization problem can then be formulated into the following convex optimization problem:

.. math::

    \begin{align*}
        \inf_{x, \tau, \lambda, s_i, \gamma_{i,k}} \quad & \lambda \epsilon + \frac{1}{N} \sum_{i=1}^{N} s_i \\
        \text{subject to} \quad & x \in \mathbb{X} \\
        & a_k \langle x, \hat{\xi_i} \rangle + b_k \tau + \langle \gamma_{i,k}, d - C \hat{\xi_i} \rangle \leq s_i\\
        & \|C^\top \gamma_{i,k} - a_k x\|_\star \leq \lambda \\
        & \gamma_{i,k} \geq 0
    \end{align*}

where :math:`N` is the number of samples, :math:`\|\cdot\|_*` is the dual norm, and :math:`k \in \{1,2\}`
with :math:`a_1 = −1, \ a_2 = −1 − \rho/\alpha, \ b_1 = \rho, b_2 = \rho(1 − 1/\alpha)`.
The support set is given by :math:`\Xi = \{\xi | C \xi \leq d\}`.


Using cvxRiskOpt, the original problem can be encoded using high-level function without having to reformulate the problem.


Example
-------

The distributionally robust mean-cvar optimization problem, as described in [Esfahani2018]_ with :math:`\Xi = \mathbb{R}^m` and using the 1-norm so that the dual norm is the :math:`\infty` norm, can be encoded using cvxRiskOpt as follows:

.. code-block:: python

    import numpy as np
    import cvxpy as cp
    from numpy.random import normal as gauss
    from cvxRiskOpt.wass_risk_opt_pb import WassDRExpectation, WassDRCVaR

    def generate_esfahani_portfolio_prob_dataset(num_samples, num_assets, num_simulations):
        # parameters
        phi_mu, phi_sigma = 0, 2 / 100  # systematic risk factor: Gaussian(mu, var)
        zeta_mu, zeta_sigma = lambda i: i * 3 / 100, lambda i: i * 2.5 / 100  # unsystematic risk factor: Gaussian(mu, var)

        # dataset
        xi_dataset = np.zeros([num_samples, num_assets, num_simulations])
        for sim in range(num_simulations):
            for i in range(num_assets):
                # this implementation matches the paper results
                phi = gauss(phi_mu, phi_sigma, num_samples)
                zeta = gauss(zeta_mu(i + 1), zeta_sigma(i + 1), num_samples)
                xi_dataset[:, i, sim] = phi + zeta

        return xi_dataset

    eps = 0.01  # Wasserstein radius
    num_samples = 25

    # problem settings
    num_assets = 10
    alpha, rho = 0.2, 10
    num_sim = 1

    # generate the dataset
    xi_dataset = generate_esfahani_portfolio_prob_dataset(num_samples, num_assets, num_sim)

    # create the optimization problem
    # set up the problem we're testing
    x = cp.Variable(num_assets, name='x')
    # expectation part
    a_e = -1 * x
    expect_part = WassDRExpectation(num_samples, a_e, used_norm=1)
    # CVaR part
    a_c = -1 * x
    cvar_part = WassDRCVaR(num_samples, num_assets, a_c, alpha=alpha, used_norm=1)
    # portfolio constraints
    portfolio_constr = cp.Problem(cp.Minimize(0), [cp.sum(x) == 1, x >= 0])
    # put together the DR portfolio problem
    test_prob = expect_part + rho * cvar_part + portfolio_constr

    # run the simulations
    x_test, t_test = [], []
    for sim in range(num_sim):
        xi = xi_dataset[:, :, sim]
        # solve the problem we are testing
        for par in test_prob.param_dict.keys():
            if 'eps' in par:
                test_prob.param_dict[par].value = eps
            if 'samples' in par:
                test_prob.param_dict[par].value = xi
        test_prob.solve(solver=cp.CLARABEL)
        test_result = x.value
        x_test.append(test_result)

        print("Portfolio distribution: ", x_test)


.. parsed-literal::

    Portfolio distribution:  [array([5.30878878e-11, 5.75609346e-11, 4.01569505e-10, 2.26943160e-01,
           5.28422920e-11, 2.03573625e-01, 2.26943187e-01, 2.26943184e-01,
           1.15596836e-01, 8.42485446e-09])]

Note that compared to the code above which uses cvxRiskOpt, the CVXPY-only implementation of the problem, after its manual reformulation, is given by:

.. code-block:: python

    # Optimization variables
    self.x = cp.Variable(self.n, name='x')  # decision variable
    self.tau = cp.Variable(name='tau')  # variable for CVaR computation
    self.lam = cp.Variable(name='lam')  # lambda variable from dual problem
    self.s = cp.Variable(self.N, name='s')  # epigraph auxiliary variables

    # l(xi) = max_k a_k * xi + b_k loss function
    self.a = [-self.expectation_rho * self.x, (-self.expectation_rho - self.rho / self.alpha) * self.x]
    self.b = [self.rho * self.tau, self.rho * (1 - 1 / self.alpha) * self.tau]
    self.K = len(self.a)  # number of affine functions
    # one more optimization variable
    if self.use_gamma:
        self.gamma = [cp.Variable((self.K, len(self.d)), name='gamma_' + str(i)) for i in range(self.N)]

    # Parameters
    self.eps = cp.Parameter(name='eps')
    self.xi = cp.Parameter((self.N, self.m), name='xi')

    # Optimization objective
    objective = self.lam * self.eps + 1 / self.N * cp.sum(self.s)
    self.objective = cp.Minimize(objective)

    # constraints
    constraints = []
    # DR (Expectation + CVaR) constraints
    for i in range(self.N):
        for k in range(self.K):
            if self.use_gamma:
                constraints += [self.b[k] + self.a[k] @ self.xi[i, :] + self.gamma[i][k, :] @ (self.d - self.C @ self.xi[i, :]) <= self.s[i]]
                constraints += [cp.norm(self.C.T @ self.gamma[i][k, :] - self.a[k], np.inf) <= self.lam]
            else:
                constraints += [self.b[k] + self.a[k] @ self.xi[i, :] <= self.s[i]]
                constraints += [cp.norm(- self.a[k], np.inf) <= self.lam]
        if self.use_gamma:
            constraints += [self.gamma[i] >= 0]
    # additional constraints on x
    constraints += [cp.sum(self.x) == 1]
    constraints += [self.x >= 0]
    self.constraints = constraints
    self.problem = cp.Problem(self.objective, self.constraints)


i.e. instead of the following using cvxRiskOpt:

.. code-block:: python

    x = cp.Variable(num_assets, name='x')
    # expectation part
    a_e = -1 * x
    expect_part = WassDRExpectation(num_samples, a_e, used_norm=1)
    # CVaR part
    a_c = -1 * x
    cvar_part = WassDRCVaR(num_samples, num_assets, a_c, alpha=alpha, used_norm=1)
    # portfolio constraints
    portfolio_constr = cp.Problem(cp.Minimize(0), [cp.sum(x) == 1, x >= 0])
    # put together the DR portfolio problem
    test_prob = expect_part + rho * cvar_part + portfolio_constr


.. [Esfahani2018]
    P. Mohajerin Esfahani and D. Kuhn, "Data-driven distributionally robust optimization using the wasserstein metric: Performance guarantees and tractable reformulations," Mathematical Programming, vol. 171, no. 1, pp. 115–166, 2018.
