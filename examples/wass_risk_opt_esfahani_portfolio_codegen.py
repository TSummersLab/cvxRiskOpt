import time
import csv
import numpy as np
import cvxpy as cp
from cvxpygen import cpg
from cvxRiskOpt.wass_risk_opt_pb import WassDRExpectation, WassDRCVaR
VERBOSE = True


def esfahani_portfolio_codegen(num_samples=10, num_sim=1, gen_code=True, keep_init_run=False, solver=cp.CLARABEL):
    from examples.reference_examples.esfahani_portfolio_sec7example import EsfahaniPortfolioProb, \
        generate_esfahani_portfolio_prob_dataset
    from cvxRiskOpt.wass_risk_opt_pb import update_wass_wce_params

    # problem settings
    eps = 0.01  # Wasserstein radius
    num_assets = 10

    # for timing
    additional_run = 0 if keep_init_run else 1

    # generate the dataset
    xi_dataset = generate_esfahani_portfolio_prob_dataset(num_samples, num_assets,
                                                          num_sim + additional_run)

    # ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #
    #  set up the reference optimization problem that only uses cvxpy   #
    # ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #
    ref_prob = EsfahaniPortfolioProb(num_samples, num_assets, expectation_rho=1, rho=1)

    # get some parameters from the reference problem
    alpha, rho = ref_prob.alpha, ref_prob.rho

    # ##### ##### ##### ##### ##### ##### ##### ##### ##### #
    #  set up the problem we're testing (using cvxRiskOpt)  #
    # ##### ##### ##### ##### ##### ##### ##### ##### ##### #
    x = cp.Variable(num_assets, name='x')
    # expectation part
    a_e = -1 * x
    expect_part = WassDRExpectation(num_samples, a_e, used_norm=1)
    # CVaR part
    a_c = -1 * x
    cvar_part = WassDRCVaR(num_samples, num_assets, a_c, alpha=alpha, used_norm=1)
    # portfolio constraints
    portfolio_constr = cp.Problem(cp.Minimize(0), [cp.sum(x) == 1, x >= 0])
    test_prob = expect_part + rho * cvar_part + portfolio_constr

    # ##### ##### ##### #
    #  generate C code  #
    # ##### ##### ##### #
    if gen_code:
        cpg.generate_code(test_prob, code_dir='esfahani_portfolio', solver=solver)

    # ##### ##### ##### ##### #
    #   run the simulations   #
    # ##### ##### ##### ##### #
    x_ref, t_ref = [], []
    x_test, t_test = [], []
    x_test_codegen, t_test_codegen = [], []

    # solve the reference problem
    for sim in range(num_sim + additional_run):
        xi = xi_dataset[:, :, sim]
        ref_prob.set_params(eps, xi)
        ts = time.time()
        ref_prob.solve(solver=solver)
        te = time.time()
        if sim == 0 and not keep_init_run:
            continue
        reference_result = ref_prob.get_result()
        t_ref.append(te - ts)
        x_ref.append(reference_result)

    # solve the test problem
    for sim in range(num_sim + additional_run):
        xi = xi_dataset[:, :, sim]
        update_wass_wce_params(test_prob, eps, xi)
        ts = time.time()
        test_prob.solve(solver=solver)
        te = time.time()
        if sim == 0 and not keep_init_run:
            continue
        test_result = x.value
        t_test.append(te - ts)
        x_test.append(test_result)

    # solve the test problem with C code via python wrapper
    from esfahani_portfolio.cpg_solver import cpg_solve
    test_prob.register_solve('cpg', cpg_solve)
    update_params = []  # get the list of parameters that should be updated
    for par in test_prob.param_dict.keys():
        if 'eps' in par or 'samples' in par:
            update_params.append(par)
    for sim in range(num_sim + additional_run):
        if sim == 0 and not keep_init_run:
            continue  # c code does not usually take significantly longer on first solve, so we don't account for that
        xi = xi_dataset[:, :, sim]
        update_wass_wce_params(test_prob, eps, xi)
        if solver in [cp.ECOS]:  # because with cvxpygen `verbose=False` fails with ECOS (once fixed, this can be removed)
            ts = time.time()
            test_prob.solve(method='cpg', updated_params=update_params)
            te = time.time()
        else:
            ts = time.time()
            test_prob.solve(method='cpg', updated_params=update_params, verbose=False)
            te = time.time()
        test_result_cpg = x.value
        t_test_codegen.append(te - ts)
        x_test_codegen.append(test_result_cpg)

    for t, (xr, xt, xtc) in enumerate(zip(x_ref, x_test, x_test_codegen)):
        if not np.allclose(xr, xt, rtol=1e-5, atol=1e-5):
            print("Sim {} results not close enough (xr, xt)!".format(t))
            print("Distance = ", np.linalg.norm(xr-xt))
        if not np.allclose(xr, xtc, rtol=1e-5, atol=1e-5):
            print("Sim {} results not close enough (xr, xtc)!".format(t))
            print("Distance = ", np.linalg.norm(xr - xtc))

    return t_ref, t_test, t_test_codegen


if __name__ == "__main__":
    solver = cp.CLARABEL
    t_ref, t_test, t_test_codegen = (
        esfahani_portfolio_codegen(num_samples=10,
                                   num_sim=100,
                                   gen_code=True,
                                   keep_init_run=False,
                                   solver=solver))
    print(t_ref)
    print(t_test)
    print(t_test_codegen)
    with open('esfahani_portfolio_results_{}.csv'.format(solver), 'a', newline='') as file:
        writer = csv.writer(file)
        for t in range(len(t_test_codegen)):
            writer.writerow([t_ref[t], t_test[t], t_test_codegen[t], solver])
