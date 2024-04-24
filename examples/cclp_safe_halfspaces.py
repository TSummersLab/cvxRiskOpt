import numpy as np
from scipy.stats import norm as gauss
import matplotlib.pyplot as plt
from reference_examples.safaoui_dr_cvar_halfspace import Halfspace


def generate_gaussian_samples(mu, std_div, num_samples):
    return np.random.normal(mu, std_div, num_samples)


def moment_based_safe_halfspace():
    """
    Similar to the safe halfspace in
    "Safe multi-agent motion planning under uncertainty for drones using filtered reinforcement learning"
    by
    Sleiman Safaoui, Abraham P. Vinod, Ankush Chakrabarty, Rien Quirynen, Nobuyuki Yoshikawa, and Stefano Di Cairano

    Prob(a^T * xi + b <= 0) >= 1-eps = bound (eps \in (0, 0.5] )
    :return:
    """

    a = np.array([1, 2])
    b = 3
    hs = Halfspace()
    hs.h = a
    hs.g = b
    fig, ax = plt.subplots()
    ax.axis('equal')
    plt.xlim([-5, 1])
    plt.ylim([-5, 1])
    hs.plot_poly(ax=ax)

    # Gaussian:
    eps = 0.1  # 90% success
    d1_hat = np.array([0, 0])
    d1_cov = np.diag([0.01, 0.01])
    d2_hat = 0
    d2_cov = [[0]]
    d1d2_cross_cov = np.array([[0, 0]])
    kappa_e = gauss.ppf(1-eps)
    x_til = np.hstack([a, 1])
    gamma_top = np.hstack([d1_cov, d1d2_cross_cov.T])
    gamma_bot = np.hstack([d1d2_cross_cov, d2_cov])
    gamma = np.vstack([gamma_top, gamma_bot])
    sigma_x = np.sqrt(x_til.T @ gamma @ x_til)
    a_gauss = a
    b_gauss = kappa_e * sigma_x + b + d2_hat
    hs_gauss = Halfspace()
    hs_gauss.h = a_gauss
    hs_gauss.g = b_gauss
    hs_gauss.plot_poly(ax=ax, color='tab:blue')

    # DRO
    kappa_e = np.sqrt((1-eps) / eps)
    a_dro = a
    b_dro = kappa_e * sigma_x + b + d2_hat
    hs_dro = Halfspace()
    hs_dro.h = a_dro
    hs_dro.g = b_dro
    hs_dro.plot_poly(ax=ax, color='tab:red')

    plt.show()


if __name__ == "__main__":
    moment_based_safe_halfspace()
