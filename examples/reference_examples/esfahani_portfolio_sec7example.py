import numpy as np
from numpy.random import normal as gauss
import matplotlib.pyplot as plt
import cvxpy as cp


def generate_esfahani_portfolio_prob_dataset(num_samples, num_assets, num_simulations):
    """
    Generates the dataset for the Mean-risk portforlio optimization problem
    presented in eq (27) and desribed in Section 7 of
    "Data-driven distributionally robust optimization using
        the Wasserstein metric: performance guarantees
        and tractable reformulations"
    by Peyman Mohajerin Esfahani, Daniel Kuhn

    The generated data corresponds xi_i = psi + zeta_i
    with
    systematic risk factor psi ~ N(0,2%)
    and
    unsystematic risk factor zeta_i ~ N(i x 3%, i * 2.5%)
    where N denotes a Gaussian distribution

    Note: Python Gauss function takes the mean and standard deviations as inputs (mu, sigma)
    The usual N(.,.) notation used in the paper refers to the mean and variance --> (mu, sigma^2)
    However, here we use the notation N(mu, sigma) so that the results match the results reported in the paper

    :param num_samples: number of samples
    :param num_assets: number of assets
    :param num_simulations:
    :return:
    """
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


class EsfahaniPortfolioProb:
    def __init__(self, num_samples, num_assets=10, alpha=0.2, rho=10, C=None, d=None, expectation_rho=1):
        """
        Implements the Mean-risk portforlio optimization problem
        presented in eq (27) and desribed in Section 7 of
        "Data-driven distributionally robust optimization using
            the Wasserstein metric: performance guarantees
            and tractable reformulations"
        by Peyman Mohajerin Esfahani, Daniel Kuhn

        :param num_samples: number of samples
        :param num_assets: number of assets (10 used in paper)
        :param alpha: CVaR risk bound (0.2 used in paper)
        :param rho: risk aversion variable (10 used in paper)
        :param C: uncertainty set C matrix (C xi <= d) (None used in paper) (set = real numbers)
        :param d: uncertainty set d vector (C xi <= d) (None used in paper) (set = real numbers)
        :param expectation_rho: scaling term for expectation term (not in paper, i.e. 1 is used)
        """
        # inputs
        self.N = num_samples  # number of samples
        self.alpha, self.rho = alpha, rho
        self.expectation_rho = expectation_rho
        self.C, self.d = C, d
        self.use_gamma = True if self.C is not None else False

        # problem dimension
        self.m = num_assets  # dimension of sample (number of assets)
        self.n = self.m  # dimension of decision variable x

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

    def set_params(self, eps, xi):
        self.eps.value = eps
        self.xi.value = xi

    def solve(self, *args, **kwargs):
        self.problem.solve(*args, **kwargs)

    @property
    def status(self):
        return self.problem.status

    def get_result(self):
        return self.x.value


def generate_paper_results(N=30, use_support=False):
    """
    Generates the paper results
    :param N: number of samples (30, 300, 3000 used in paper)
    :return:
    """
    # dimensions and data
    m = 10  # dimension of sample (number of assets)
    num_sim = 100  # number of independent simulations

    x_vals = []
    num_steps = 50
    start = 1e-3
    stop = 1
    eps_range = np.logspace(np.log10(start), np.log10(stop), num=num_steps)

    xi_dataset = generate_esfahani_portfolio_prob_dataset(N, m, num_sim)
    min_vect = xi_dataset.min(axis=2).min(axis=0)
    max_vect = xi_dataset.max(axis=2).max(axis=0)

    if use_support:
        C = np.vstack([np.eye(m), -1 * np.eye(m)])
        d = np.hstack([max_vect, -min_vect])
    else:
        C = d = None
    prob = EsfahaniPortfolioProb(N, m, C=C, d=d)

    for eps_val in eps_range:
        print("eps: ", eps_val)
        x_vals_tmp = []
        for sim in range(num_sim):
            prob.set_params(eps_val, xi_dataset[:, :, sim])
            prob.solve(solver=cp.CLARABEL)
            x_vals_tmp.append(prob.get_result())
        x_vals_tmp = np.array(x_vals_tmp)
        x_vals_avrg = np.mean(x_vals_tmp, 0)
        x_vals.append(x_vals_avrg)

    x_vals = np.array(x_vals)
    sum = np.zeros([num_steps, ])
    fig = plt.figure()
    for i in range(m):
        sum += x_vals[:, i]
        plt.plot(eps_range, sum)
    plt.xscale('log')

    plt.show()


def formulation_check_with_expectation_rho():
    eps = 0.1  # Wasserstein radius
    solver = cp.CLARABEL

    # problem settings
    num_samples = 30
    num_assets = 10
    num_sim = 1
    # generate the dataset
    xi_dataset = generate_esfahani_portfolio_prob_dataset(num_samples, num_assets, num_sim)
    xi = xi_dataset[:, :, 0]

    # reference optimization problem
    ref_prob = EsfahaniPortfolioProb(num_samples, num_assets, rho=10, expectation_rho=1)
    ref_prob.set_params(eps, xi)
    ref_prob.solve(solver=solver)
    reference_result = ref_prob.get_result()
    reference_obj_result = ref_prob.problem.value

    # test optimization problem
    scale = 3
    test_prob = EsfahaniPortfolioProb(num_samples, num_assets, rho=10*scale, expectation_rho=scale)
    test_prob.set_params(eps, xi)
    test_prob.solve(solver=solver)
    test_result = test_prob.get_result()
    test_obj_result = test_prob.problem.value

    # check
    diff = np.linalg.norm(test_result-reference_result)
    obj_diff = np.abs(scale * reference_obj_result - test_obj_result)
    print("Opt x Diff: ", diff)
    print("Objective Diff: ", obj_diff)
    assert diff < 1e-6 and obj_diff < 1e-6


if __name__ == "__main__":
    # formulation_check_with_expectation_rho()

    N = 30
    generate_paper_results(N)

    N = 30
    generate_paper_results(N, use_support=True)
