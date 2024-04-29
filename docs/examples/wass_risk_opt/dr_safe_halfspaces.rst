
Distributionally Robust Safe Halfspaces
=======================================

Consider the problem of computing the location of a halfspace subject to a risk constraint, i.e.:

.. math::
    \begin{align*}
        \min \quad & g \\
        \text{subject to} \quad & \mathcal{R}(\ell(\mathbf{p})) \leq \delta
    \end{align*}

where the halfspace is given by :math:`\mathcal{H} = \{p \mid h \cdot p + g \leq 0\}` with a known halfspace normal :math:`h` and :math:`h \cdot p` being the inner product between the two vectors. :math:`\mathcal{R}` is a risk metric and :math:`\delta` is a risk-bound. The function :math:`\ell(\cdot)` is a loss function that returns the value whose risk needs to be bounded.

Let :math:`\ell(p) = -(h \cdot p + g - r)` be the loss function where :math:`r` is some constant that represents an existing tightening of the halfpsace (e.g. robot radius in a collision avoidance problem). Using the :math:`\text{DR-CVaR}` risk metric for :math:`\mathcal{R}` we end up with the following optimization problem:

.. math::
    \begin{align*}
        \min \quad & g \\
        \text{subject to} \quad & \sup_{\mathbb{P} \in \mathcal{P}} \text{CVaR}_\alpha^{\mathbb{P}} (-(h \cdot \mathbf{p} + g - r)) \leq \delta
    \end{align*}

This problem can be encoded with cvxRiskOpt as follows:

.. code-block:: python

    import numpy as np
    import cvxpy as cp
    from numpy.random import normal as gauss
    from cvxRiskOpt.wass_risk_opt_pb import WassWCEMaxAffine, WassDRCVaR

    from scipy.stats import norm
    from scipy.stats import expon
    from scipy.stats import laplace
    from scipy.stats import bernoulli

    def generate_noise_samples(shape, loc, scale, dist='norm'):
        if dist == "norm":
            xi = norm.rvs(loc=loc, scale=scale, size=shape)
        elif dist == 'expo':
            xi = expon.rvs(loc=loc, scale=scale, size=shape)
        elif dist == 'lap':
            xi = laplace.rvs(loc=loc, scale=scale, size=shape)
        elif dist == 'bern':
            p = 0.5
            xi = (bernoulli.rvs(p, loc=0, size=shape) - p) * scale + loc
        else:
            raise NotImplementedError('Chosen distribution not implemented')
        return xi

    def generate_safaoui_halfspace_prob_dataset(num_samples):
        np.random.seed(1)
        ob = np.array([0.5, 0])
        noise_std_dev = np.array([0.1, 0.1])
        xi_dataset = np.zeros((2, num_samples))
        xi_dataset[0, :] = generate_noise_samples(num_samples, ob[0], np.sqrt(noise_std_dev[0]), dist='norm')
        xi_dataset[1, :] = generate_noise_samples(num_samples, ob[1], np.sqrt(noise_std_dev[1]), dist='norm')
        return xi_dataset

    # problem settings
    alpha = 0.1
    eps = 0.01
    delta = -1
    h = np.array([1., 1])
    h = h / np.linalg.norm(h)
    r = [1]
    solver = cp.CLARABEL
    num_samples = 30

    # generate the dataset
    xi = generate_safaoui_halfspace_prob_dataset(num_samples)

    # encode and solve the problem using cvxRiskOpt's DR-CVaR class
    g = cp.Variable(1, name='g')
    risk_prob = WassDRCVaR(num_samples=num_samples, xi_length=2, a=-h, b=-g+r[0], alpha=alpha, used_norm=2)
    risk_constraints = [risk_prob.objective.expr <= delta] + risk_prob.constraints
    halfspace_prob = cp.Problem(cp.Minimize(g), risk_constraints)
    for par in halfspace_prob.param_dict.keys():
        if 'eps' in par:
            halfspace_prob.param_dict[par].value = eps
        if 'samples' in par:
            halfspace_prob.param_dict[par].value = xi.T
    halfspace_prob.solve(solver=solver)
    halfspace_prob_result = g.value

    print("Halfspace location with cvxRiskOpt's WassDRCVaR: g = ", halfspace_prob_result)

    # encode and solve the problem with cvxRiskOpt's general max affine class (this requires some reformulation of the CVaR constraint)
    m = xi.shape[0]
    h_xi = h @ xi  # alternative formulation where h@xi are the samples
    tau = cp.Variable(1, name='tau')
    a_k_list = [- 1 / alpha, 0]
    b_k_list = [-1 / alpha * g + 1 / alpha * r[0] + (1 - 1 / alpha) * tau, tau]
    wce = WassWCEMaxAffine(num_samples, a_k_list, b_k_list, used_norm=2, vp_suffix='')
    # for the DR-CVaR synthesis problem, wce is a constraint
    dr_cvar_bound = [wce.objective.expr <= delta] + wce.constraints
    halfspace_prob2 = cp.Problem(cp.Minimize(g), dr_cvar_bound)
    # solve the problem we are testing
    halfspace_prob2.param_dict['eps'].value = eps
    halfspace_prob2.param_dict['samples'].value = h_xi
    halfspace_prob2.solve(solver=solver)
    test_result = g.value

    print("Halfspace location with cvxRiskOpt's WassWCEMaxAffine: g = ", test_result)


.. parsed-literal::
    Halfspace location with cvxRiskOpt's WassDRCVaR: g =  [2.28162319]
    Halfspace location with cvxRiskOpt's WassWCEMaxAffine: g =  [2.28162319]




