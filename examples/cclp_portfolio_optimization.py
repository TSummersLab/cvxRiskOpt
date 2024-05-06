import csv
import time
from cvxpygen import cpg
import cvxpy as cp
import numpy as np
import scipy.linalg
from scipy.stats import norm as gauss


def generate_gaussian_samples(mu, std_div, num_samples):
    return np.random.normal(mu, std_div, num_samples)


def portfolio_optimization(solver=cp.CLARABEL):
    """
    Solves:
    min sum(x_i)
    s.t. x >=0, Prob(R >= R0) >= 1-eps
    where
    x is the decision variable for the investment amount in each asset
    r is the random variable representing the return of each asset (continuous r.vector)
    R is the return: R = sum(x_i * r_i)
    R0 is the desired return to exceed
    eps in (0, 0.5] is a risk bound
    :return:
    """
    num_assets = 10
    x = cp.Variable(num_assets, name='x')
    r_mean = [5 / 100 * (i + 1) for i in range(num_assets)]
    r_sigma = [2 / 100 * (i + 1) for i in range(num_assets)]
    r0 = 100
    eps = 0.3
    num_samples = 1000000

    dataset = np.zeros([num_samples, num_assets])
    for i in range(num_assets):
        dataset[:, i] = generate_gaussian_samples(r_mean[i], r_sigma[i], num_samples)

    d_std_div = cp.pnorm(scipy.linalg.sqrtm(np.diag(r_sigma)) @ x)
    constr = 0 >= r0 -x @ r_mean + d_std_div * gauss.ppf(1-eps)
    prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0, constr])
    prob.solve(solver=cp.CLARABEL)
    x_val = x.value
    print("Reference investment: ", x_val)
    ret = 0
    for i in range(num_samples):
        ret += x_val @ dataset[i, :]
    ret /= num_samples
    print("Return: ", ret)
    print("Minimum desired: ", r0)

    from cvxRiskOpt.cclp_risk_opt import cclp_gauss
    cc_contsr = cclp_gauss(eps, a=-x, b=r0, xi1_hat=r_mean, gam11=np.diag(r_sigma))
    prob2 = cp.Problem(cp.Minimize(cp.sum(x)),
                      [x >= 0, cc_contsr])
    prob2.solve(solver=solver)
    x_val2 = x.value
    print("Invest based on problem: ", x_val2)
    ret = 0
    for i in range(num_samples):
        ret += x_val2 @ dataset[i, :]
    ret /= num_samples
    print("Return: ", ret)
    print("Minimum desired: ", r0)

    # DRO Cent Sym
    from cvxRiskOpt.cclp_risk_opt import cclp_dro_mean_cov
    cc_contsr = cclp_dro_mean_cov(eps, a=-x, b=r0, xi1_hat=r_mean, gam11=np.diag(r_sigma), centrally_symmetric=True)
    prob3 = cp.Problem(cp.Minimize(cp.sum(x)),
                       [x >= 0, cc_contsr])
    prob3.solve(solver=solver)
    x_val3 = x.value
    print("Invest based on DRO problem: ", x_val3)
    ret = 0
    for i in range(num_samples):
        ret += x_val3 @ dataset[i, :]
    ret /= num_samples
    print("Return: ", ret)
    print("Minimum desired: ", r0)

    # DRO
    cc_contsr = cclp_dro_mean_cov(eps, a=-x, b=r0, xi1_hat=r_mean, gam11=np.diag(r_sigma), centrally_symmetric=False)
    prob4 = cp.Problem(cp.Minimize(cp.sum(x)),
                       [x >= 0, cc_contsr])
    prob4.solve(solver=solver)
    x_val4 = x.value
    print("Invest based on DRO problem: ", x_val4)
    ret = 0
    for i in range(num_samples):
        ret += x_val4 @ dataset[i, :]
    ret /= num_samples
    print("Return: ", ret)
    print("Minimum desired: ", r0)


def moment_portfolio_optimization(num_sim=200, use_cpg=True, gen_code=True, keep_init_run=False, solver=cp.CLARABEL):
    """
    Solves:
    min sum(x_i)
    s.t. x >=0, Prob(R >= R0) >= 1-eps
    where
    x is the decision variable for the investment amount in each asset
    r is the random variable representing the return of each asset (continuous r.vector)
    R is the return: R = sum(x_i * r_i)
    R0 is the desired return to exceed
    eps in (0, 0.5] is a risk bound
    :return:
    """
    num_assets = 10
    x = cp.Variable(num_assets, name='x')
    r0 = cp.Parameter(name='r0')
    r_mean = [5 / 100 * (i + 1) for i in range(num_assets)]
    r_sigma = [2 / 100 * (i + 1) for i in range(num_assets)]
    r0_val = 100
    eps = 0.3
    additional_run = 0 if keep_init_run else 1

    # DRO
    from cvxRiskOpt.cclp_risk_opt import cclp_dro_mean_cov
    cc_contsr = cclp_dro_mean_cov(eps, a=-x, b=r0, xi1_hat=r_mean, gam11=np.diag(r_sigma), centrally_symmetric=False)
    prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0, cc_contsr])

    x_test, t_test = [], []
    x_test_codegen, t_test_codegen = [], []
    for sim in range(num_sim + additional_run):
        r0.value = r0_val
        ts = time.time()
        prob.solve(solver=solver)
        te = time.time()
        if sim == 0 and not keep_init_run:
            continue
        t_test.append(te - ts)
        x_test.append(x.value)

    if use_cpg:
        if gen_code:
            cpg.generate_code(prob, code_dir='portfolio_opt', solver=solver)
        from portfolio_opt.cpg_solver import cpg_solve
        prob.register_solve('cpg', cpg_solve)
        for sim in range(num_sim + additional_run):
            ts = time.time()
            prob.solve(method='cpg', updated_params=["r0"])
            te = time.time()
            if sim == 0 and not keep_init_run:
                continue
            t_test_codegen.append(te - ts)
            x_test_codegen.append(x.value)

        for t, (xt, xtc) in enumerate(zip(x_test, x_test_codegen)):
            if not np.allclose(xt, xtc, rtol=1e-5, atol=1e-5):
                print("Sim {} results not close enough (xt, xtc)!".format(t))

    return t_test, t_test_codegen


if __name__ == "__main__":
    solver = cp.ECOS
    portfolio_optimization(solver)
    t_test, t_test_codegen = (
        moment_portfolio_optimization(num_sim=200,
                                      use_cpg=True,
                                      gen_code=True,
                                      keep_init_run=False,
                                      solver=solver))
    # # Uncomment below to save results
    # print(t_test)
    # print(t_test_codegen)
    # with open('portfolio_optimization_results_{}.csv'.format(solver), 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     for t in range(len(t_test_codegen)):
    #         writer.writerow([t_test[t], t_test_codegen[t], solver])
