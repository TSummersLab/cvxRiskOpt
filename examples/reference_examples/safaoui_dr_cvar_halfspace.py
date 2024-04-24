import numpy as np
from numpy.random import normal as gauss
import matplotlib.pyplot as plt
import cvxpy as cp
from copy import copy
import polytope


from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import laplace
from scipy.stats import bernoulli


def generate_noise_samples(shape, loc, scale, dist='norm'):
    """
    :param shape: shape of random variables
    :param loc: mean value
    :param scale: standard deviation
    :param dist: distribution type
    :return:
    """
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
    """
    Generates a dataset for the DR Safe halfspace problem
    presented in eq (6) (and a verification version of it) of
    "Distributionally Robust CVaR-Based Safety Filtering for Motion Planning in Uncertain Environments"
    by Sleiman Safaoui, Tyler Summers

    The generated data corresponds randomly generated positions in a 2D environment

    :param num_samples: number of samples
    :return: xi_dataset (2 x num_samples ndarray)
    """
    np.random.seed(1)
    ob = np.array([0.5, 0])
    noise_std_dev = np.array([0.1, 0.1])
    xi_dataset = np.zeros((2, num_samples))
    xi_dataset[0, :] = generate_noise_samples(num_samples, ob[0], np.sqrt(noise_std_dev[0]), dist='norm')
    xi_dataset[1, :] = generate_noise_samples(num_samples, ob[1], np.sqrt(noise_std_dev[1]), dist='norm')

    return xi_dataset


class Halfspace:
    """
    Class representing a 2D halfspace given by {x | h x + g <= 0}
    """
    def __init__(self, A=None, b=None):
        self.h = None
        self.g = None
        self._halfspace_bounds(A, b)
        self.poly = None

    def _halfspace_bounds(self, A, b):
        """
        A * X <= b linear constraint to place bounds on X (2 x 1 vector)
        :param A: c x 2 matrix
        :param b: c x 1 vector
        :return:
        """
        if A is None:
            self.A = [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]
        else:
            self.A = A
        if b is None:
            self.b = [100, 100, 100, 100]
        else:
            self.b = b

    @property
    def get_poly(self):
        """
        Create a Polytope from the polytope library given by the halfspace bound to the specified environment bounds
        :return: polytope.Polytope
        """
        A, b = copy(self.A), copy(self.b)
        A.append(self.h)
        b.append(-self.g)
        A, b = np.array(A), np.array(b)
        poly = polytope.Polytope(A, b)  # polytope defined as Ax <= b
        self.poly = poly
        return poly


    @property
    def get_Ax_leq_b_A(self):
        """ returns the A component of the halfspace formatted as A * x <= b. Excludes the environment bounds """
        return self.h

    @property
    def get_Ax_leq_b_b(self):
        """ returns the b component of the halfspace formatted as A * x <= b. Excludes the environment bounds """
        return -self.g

    def plot_poly(self, poly=None, ax=None, color='g', alpha=0.1, show=False, linewidth=1):
        """
        Plots the polytope on the provides axis or a new one.
        :param poly: poyltope.Polytope object
        :param ax: matplotlib axis
        :param color: polytope color
        :param alpha: opacity value [0, 1] (1 opaque, 0 transparent)
        :param show: True --> show the plot, False --> don't show the plot
        :return: if plot not shown, returns the axis
        """
        if self.h is None or self.g is None:
            raise NotImplementedError('Plotting error: Halfspace not defined yet!')
        if poly is None:
            poly = self.get_poly
        if ax is None:
            fig, ax = plt.subplots()
            ax.axis('equal')
        poly.plot(ax=ax, color=color, alpha=alpha, linestyle='solid', linewidth=linewidth)

        if show:
            plt.show()
        else:
            return ax


class drcvar_halfspace_synthesis(Halfspace):
    def __init__(self, alpha, eps, bound, num_samp, A=None, b=None, support_C=None, support_d=None):
        """
        Computes a DR-CVaR halfspace based on equation (6) from
        "Distributionally Robust CVaR-Based Safety Filtering for Motion Planning in Uncertain Environments"
        by Sleiman Safaoui, Tyler Summers

        :param alpha: alpha-worst cases considered for the CVaR computation
        :param eps: wasserstein ball radius for DR part
        :param bound: DR-CVaR bound
        :param num_samp: number of samples for empirically estimating the expected value term in the CVaR definition
        :param A: A matrix of environment bounds on the halfspace Ax <= b (c x 2 matrix)
        :param b: b vector of environment bounds on the halfspace Ax <= b (c x 1 vector)
        :param solver: cvxpy solver to use
        """
        super().__init__(A, b)

        self.use_gamma = False
        self.C, self.d = support_C, support_d
        if support_d is not None:
            self.use_gamma = True

        self.alpha = alpha
        self.eps = eps
        self.delta = bound
        self.n = num_samp
        self.xi = None
        self.K = 2

        self._def_opt_pb_vars()
        self._def_opt_pb()

    def _def_opt_pb_vars(self):
        """ defines the optimization variables and parameters"""
        self._g = cp.Variable(1, name='g')
        self._tau = cp.Variable(1, name='tau')
        self._lam = cp.Variable(1, name='lam')
        self._eta = cp.Variable(self.n, name='eta')
        if self.use_gamma:
            self.gamma = [cp.Variable((self.K, len(self.d)), name='gamma_' + str(i)) for i in range(self.n)]

        # instead of defining params for h and xi then multiplying them, this is the product h @ xi
        self._h_xi_prod = cp.Parameter((self.n,), name='h_xi_prod')
        self._r = cp.Parameter((1,), name='r')
        self.param_names = ['h_xi_prod', 'r']
        if self.use_gamma:
            self.samples = cp.Parameter((self.n, 2), name='samples')
            self.h_param = cp.Parameter(2, name='h')

    def _def_opt_pb(self):
        """ defines the optimization problem """
        a_k = [-1 / self.alpha, 0]
        b_k = [-1 / self.alpha, 0]
        c_k = [1 - 1 / self.alpha, 1]

        constraint = [self._lam * self.eps + 1 / self.n * cp.sum(self._eta) <= self.delta]  # cvar objective
        if self.use_gamma:
            for i in range(self.n):
                for k in range(len(a_k)):
                    constraint += [a_k[k] * self._h_xi_prod[i] + b_k[k] * (self._g - self._r) + c_k[k] * self._tau + self.gamma[i][k, :] @ (self.d - self.C @ self.samples[i, :]) <= self._eta[i]]
                    constraint += [cp.norm(self.C.T @ self.gamma[i][k, :] - a_k[k] * self.h_param) <= self._lam]
                constraint += [self.gamma[i] >= 0]
        else:
            for i in range(self.n):
                for k in range(len(a_k)):
                    constraint += [a_k[k] * self._h_xi_prod[i] + b_k[k] * (self._g - self._r) + c_k[k] * self._tau <= self._eta[i]]
                constraint += [1 / self.alpha <= self._lam]
        self.problem = cp.Problem(cp.Minimize(self._g), constraint)

    def set_opt_pb_params(self, h, xi, r):
        """
        Set the optimization problem parameters
        :param h: halfspace normal
        :param xi: samples
        :param r: "radius" or padding to be added to the halfspace
        :return:
        """
        self.h = h
        self.xi = xi
        self._h_xi_prod.value = h @ xi
        self._r.value = r
        if self.use_gamma:
            self.samples.value = xi.T
            self.h_param.value = h

    def solve_opt_pb(self, *args, **kwargs):
        """
        Solve the optimization problem and extract the result
        :return: status (solved or not) and problem info (setup time, solve time, ...)
        """

        self.problem.solve(*args, **kwargs)

        if self.problem.status not in ["infeasible", "unbounded"]:
            self.g = self._g.value[0]
            return True
        return False

    def get_result(self):
        return self.g


def example(N=30, use_support=False):
    """
    Generates the paper results
    :param N: number of samples (10, 100, ...)
    :return:
    """
    # problem settings
    alpha = 0.1
    eps = 0.01
    delta = -1
    h = np.array([1., 1])
    h = h / np.linalg.norm(h)
    r = [1]

    # dimensions and data
    xi_dataset = generate_safaoui_halfspace_prob_dataset(N)
    m = xi_dataset.shape[0]
    min_vect = xi_dataset.min(axis=1)
    max_vect = xi_dataset.max(axis=1)

    if use_support:
        C = np.vstack([np.eye(m), -1 * np.eye(m)])
        d = np.hstack([max_vect, -min_vect])
    else:
        C = d = None
    prob = drcvar_halfspace_synthesis(alpha, eps, delta, N, support_C=C, support_d=d)

    prob.set_opt_pb_params(h, xi_dataset, r)
    prob.solve_opt_pb(solver=cp.CLARABEL)
    print("g = ", prob.get_result())

    prob.plot_poly(color='g', alpha=0.1, show=False)
    plt.scatter(xi_dataset[0, :], xi_dataset[1, :], color='k')
    plt.show()


if __name__ == "__main__":
    example(N=30, use_support=False)

    example(N=30, use_support=True)