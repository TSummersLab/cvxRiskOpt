import numpy as np
import cvxpy as cp
from cvxpygen import cpg
from cvxRiskOpt.wass_risk_opt_pb import WassWCEMaxAffine

VERBOSE = True


def safaoui_dr_cvar_halfspaces_codegen():
    from examples.reference_examples.safaoui_dr_cvar_halfspace import drcvar_halfspace_synthesis, \
        generate_safaoui_halfspace_prob_dataset

    # problem settings
    alpha = 0.1
    eps = 0.01
    delta = -1
    h = np.array([1., 1])
    h = h / np.linalg.norm(h)
    r = [1]
    solver = cp.ECOS

    num_samples = 30

    # generate the dataset
    xi = generate_safaoui_halfspace_prob_dataset(num_samples)
    m = xi.shape[0]
    h_xi = h @ xi  # alternative formulation where h@xi are the samples

    # set up and solve the reference optimization problem
    ref_prob = drcvar_halfspace_synthesis(alpha, eps, delta, num_samples)
    ref_prob.set_opt_pb_params(h, xi, r)
    ref_prob.solve_opt_pb(solver=solver)
    reference_result = ref_prob.get_result()

    # set up and solve the problem we're testing
    g = cp.Variable(1, name='g')
    tau = cp.Variable(1, name='tau')
    a_k_list = [- 1 / alpha, 0]
    b_k_list = [-1 / alpha * g + 1 / alpha * r[0] + (1 - 1 / alpha) * tau, tau]
    wce = WassWCEMaxAffine(num_samples, a_k_list, b_k_list, used_norm=2, vp_suffix='')
    # for the DR-CVaR synthesis problem, wce is a constraint
    dr_cvar_bound = [wce.objective.expr <= delta]
    dr_cvar_bound.extend(wce.constraints)
    test_prob = cp.Problem(cp.Minimize(g), dr_cvar_bound)
    # solve the problem we are testing
    test_prob.param_dict['eps'].value = eps
    test_prob.param_dict['samples'].value = h_xi
    test_prob.solve(solver=solver)
    test_result = g.value
    test_obj = test_prob.value

    cpg.generate_code(test_prob, code_dir='safaoui_dr_cvar_hs', solver='ECOS')

    from safaoui_dr_cvar_hs.cpg_solver import cpg_solve
    test_prob.register_solve('cpg', cpg_solve)
    test_prob.solve(method='cpg', updated_params=['eps', 'samples'])

    test_result_cpg = g.value
    test_obj_cpg = test_prob.value

    close = np.isclose(test_result_cpg, test_result, rtol=1e-5, atol=1e-5)
    all_close = np.all(close)
    print("g: ", test_result)
    print("g with codegen: ", test_result_cpg)
    print("Decision var close: ", all_close)
    close = np.isclose(test_obj_cpg, test_obj, rtol=1e-5, atol=1e-5)
    print("Objective close: ", close)


if __name__ == "__main__":
    safaoui_dr_cvar_halfspaces_codegen()
