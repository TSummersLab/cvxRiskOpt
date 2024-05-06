import cvxpy as cp
import numpy as np
from scipy.stats import norm as gauss


def generate_gaussian_samples(mu, std_div, num_samples):
    return np.random.normal(mu, std_div, num_samples)


def production_optimization(verbose=True):
    """
    Solves:
    min c * x
    s.t. Prob(x > d) >= 1-eps
    where
    c is the cost
    x is the decision variable for the production amount
    d is the random variable representing demand (continuous r.v. e.g. electricity demand)
    eps in (0, 0.5] is a risk bound
    :return:
    """
    c = 10  # cost
    x = cp.Variable(name='x')
    d_mean = 700
    d_var = 30
    eps = 0.1
    num_samples = 1000000

    d_std_div = np.sqrt(d_var)
    d = generate_gaussian_samples(d_mean, d_std_div, num_samples)

    prob = cp.Problem(cp.Minimize(c * x), [x >= 0, x >= d_mean + d_std_div * gauss.ppf(1-eps)])
    prob.solve(solver=cp.CLARABEL)
    x_val = x.value
    if verbose:
        print("Should produce: ", x_val)

    # testing:
    good = x_val >= d
    prob_satisfy = np.sum(good) / num_samples
    if verbose:
        print("Desired probability: ", 1-eps)
        print("Empirical probability: ", prob_satisfy)

    # now with cclp fxn
    from cvxRiskOpt.cclp_risk_opt import cclp_gauss
    cc_contr = cclp_gauss(eps, a=1, b=-x,
                          xi1_hat=d_mean,
                          gam11=d_var,
                          assume_sym=False, assume_psd=False)
    prob2 = cp.Problem(cp.Minimize(c * x), [x >= 0, cc_contr])
    prob2.solve(solver=cp.CLARABEL)
    x2_val = x.value
    if verbose:
        print("Should produce: ", x2_val)

    # testing:
    good2 = x2_val >= d
    prob_satisfy2 = np.sum(good2) / num_samples
    if verbose:
        print("Empirical probability with fxn: ", prob_satisfy2)

    return x_val, x2_val


if __name__ == "__main__":
    production_optimization()
