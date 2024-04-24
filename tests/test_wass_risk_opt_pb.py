import unittest
import numpy as np
import cvxpy as cp
from cvxRiskOpt.wass_risk_opt_pb import WassWCEMaxAffine
from cvxRiskOpt.wass_risk_opt_pb import WassDRExpectation, WassDRCVaR
VERBOSE = True  # set to True to print additional data


def print_test_results(ref_res, test_res, test_status):
    print("Result: ", test_res)
    print("Status: ", test_status)
    print("Reference: ", ref_res)
    if test_status not in ["infeasible", "unbounded"]:
        print("Difference: ", np.linalg.norm(test_res - ref_res))
    else:
        print("Cannot compute difference")


class TestWassWCEMaxAffine(unittest.TestCase):
    def test_a_scaled_mD_b_scaled_1D_esfanahi_7_1(self):
        """
        Tests the case where:

        a_k = a * x with
            a \in \mathbb{R},
            x \in \mathbb{R}^m, decision variable

        b_k = b * tau with
            b \in \mathbb{R},
            tau \in \mathbb{R}, decision variable
        """

        from examples.reference_examples.esfahani_portfolio_sec7example import EsfahaniPortfolioProb, generate_esfahani_portfolio_prob_dataset

        eps = 0.1  # Wasserstein radius
        solver = cp.ECOS

        # problem settings
        num_samples = 30
        num_assets = 10
        num_sim = 1

        # generate the dataset
        xi_dataset = generate_esfahani_portfolio_prob_dataset(num_samples, num_assets, num_sim)
        xi = xi_dataset[:, :, 0]

        # set up and solve the reference optimization problem
        ref_prob = EsfahaniPortfolioProb(num_samples, num_assets)
        ref_prob.set_params(eps, xi)
        ref_prob.solve(solver=solver)
        reference_result = ref_prob.get_result()

        # get some parameters from the reference problem
        alpha, rho = ref_prob.alpha, ref_prob.rho

        # set up and solve the problem we're testing
        x = cp.Variable(num_assets, name='x')
        tau = cp.Variable(name='tau')
        a_k_list = [-x, (-1 - rho / alpha) * x]
        b_k_list = [rho * tau, rho * (1 - 1 / alpha) * tau]
        wce = WassWCEMaxAffine(num_samples, a_k_list, b_k_list, used_norm=1, vp_suffix='')
        portfolio_constr = cp.Problem(cp.Minimize(0),[cp.sum(x) == 1, x >= 0])
        test_prob = wce + portfolio_constr
        # solve the problem we are testing
        test_prob.param_dict['eps'].value = eps
        test_prob.param_dict['samples'].value = xi
        test_prob.solve(solver=solver)
        test_result = x.value

        # Compare results
        if VERBOSE:
            print_test_results(reference_result, test_result, test_prob.status)

        close = np.isclose(test_result, reference_result, rtol=1e-5, atol=1e-5)
        all_close = np.all(close)
        self.assertTrue(all_close)

    def test_a_scaled_mD_b_scaled_1D_esfanahi_7_1_with_gamma(self):
        """
        Same as test_a_scaled_mD_b_scaled_1D_esfanahi_7_1 but with an uncertainty set Xi = {xi : C * xi <= d}
        """

        from examples.reference_examples.esfahani_portfolio_sec7example import EsfahaniPortfolioProb, generate_esfahani_portfolio_prob_dataset

        eps = 0.1  # Wasserstein radius
        solver = cp.ECOS

        # problem settings
        num_samples = 30
        num_assets = 10
        num_sim = 1

        # generate the dataset
        xi_dataset = generate_esfahani_portfolio_prob_dataset(num_samples, num_assets, num_sim)
        xi = xi_dataset[:, :, 0]
        min_vect = xi_dataset.min(axis=2).min(axis=0)
        max_vect = xi_dataset.max(axis=2).max(axis=0)

        # set up and solve the reference optimization problem
        C = np.vstack([np.eye(num_assets), -1 * np.eye(num_assets)])
        d = np.hstack([max_vect, -min_vect])
        ref_prob = EsfahaniPortfolioProb(num_samples, num_assets, C=C, d=d)
        ref_prob.set_params(eps, xi)
        ref_prob.solve(solver=solver)
        reference_result = ref_prob.get_result()

        # get some parameters from the reference problem
        alpha, rho = ref_prob.alpha, ref_prob.rho

        # set up and solve the problem we're testing
        x = cp.Variable(num_assets, name='x')
        tau = cp.Variable(name='tau')
        a_k_list = [-x, (-1 - rho / alpha) * x]
        b_k_list = [rho * tau, rho * (1 - 1 / alpha) * tau]
        wce = WassWCEMaxAffine(num_samples, a_k_list, b_k_list, support_C=C, support_d=d, used_norm=1, vp_suffix='')
        portfolio_constr = cp.Problem(cp.Minimize(0),[cp.sum(x) == 1, x >= 0])
        test_prob = wce + portfolio_constr
        # solve the problem we are testing
        test_prob.param_dict['eps'].value = eps
        test_prob.param_dict['samples'].value = xi
        test_prob.solve(solver=solver)
        test_result = x.value

        # Compare results
        if VERBOSE:
            print_test_results(reference_result, test_result, test_prob.status)
        close = np.isclose(test_result, reference_result, rtol=1e-5, atol=1e-5)
        all_close = np.all(close)
        self.assertTrue(all_close)

    def test_a_cst_mD_b_affine_1D_safaoui_drcvar_synth(self):
        """
        Tests the case where:

        a_k = a with
            a \in \mathbb{R},

        b_k = b * dv + c with
            b \in \mathbb{R}^2,
            dv = [g tau] \in \mathbb{R}^2, decision variable
            c \in \mathbb{R}

        Also tests making WCE expression a constraint.
        """

        from examples.reference_examples.safaoui_dr_cvar_halfspace import drcvar_halfspace_synthesis, generate_safaoui_halfspace_prob_dataset

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
        xiT = xi.T  # WCE needs data as num_samp x m. xi is m x num_samp

        # set up and solve the reference optimization problem
        ref_prob = drcvar_halfspace_synthesis(alpha, eps, delta, num_samples)
        ref_prob.set_opt_pb_params(h, xi, r)
        ref_prob.solve_opt_pb(solver=solver)
        reference_result = ref_prob.get_result()

        # set up and solve the problem we're testing
        g = cp.Variable(1, name='g')
        tau = cp.Variable(1, name='tau')
        a_k_list = [- h / alpha, 0 * h]
        b_k_list = [-1 / alpha * g + 1 / alpha * r[0] + (1-1/alpha) * tau, tau]
        wce = WassWCEMaxAffine(num_samples, a_k_list, b_k_list, used_norm=2, vp_suffix='')
        # for the DR-CVaR synthesis problem, wce is a constraint
        dr_cvar_bound = [wce.objective.expr <= delta]
        dr_cvar_bound.extend(wce.constraints)
        test_prob = cp.Problem(cp.Minimize(g), dr_cvar_bound)
        # solve the problem we are testing
        test_prob.param_dict['eps'].value = eps
        test_prob.param_dict['samples'].value = xiT
        test_prob.solve(solver=solver)
        test_result = g.value

        # Compare results
        if VERBOSE:
            print_test_results(reference_result, test_result, test_prob.status)
        close = np.isclose(test_result, reference_result, rtol=1e-5, atol=1e-5)
        all_close = np.all(close)
        self.assertTrue(all_close)

    def test_a_cst_mD_b_affine_1D_safaoui_drcvar_synth_with_gamma(self):
        """
        Same as test_a_cst_mD_b_affine_1D_safaoui_drcvar_synth but with an uncertainty set Xi = {xi : C * xi <= d}
        """

        from examples.reference_examples.safaoui_dr_cvar_halfspace import drcvar_halfspace_synthesis, generate_safaoui_halfspace_prob_dataset

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
        xiT = xi.T  # WCE needs data as num_samp x m. xi is m x num_samp
        min_vect = xi.min(axis=1)
        max_vect = xi.max(axis=1)
        C = np.vstack([np.eye(m), -1 * np.eye(m)])
        d = np.hstack([max_vect, -min_vect])

        # set up and solve the reference optimization problem
        ref_prob = drcvar_halfspace_synthesis(alpha, eps, delta, num_samples, support_C=C, support_d=d)
        ref_prob.set_opt_pb_params(h, xi, r)
        ref_prob.solve_opt_pb(solver=solver)
        reference_result = ref_prob.get_result()

        # set up and solve the problem we're testing
        g = cp.Variable(1, name='g')
        tau = cp.Variable(1, name='tau')
        a_k_list = [- h / alpha, 0 * h]
        b_k_list = [-1 / alpha * g + 1 / alpha * r[0] + (1-1/alpha) * tau, tau]
        wce = WassWCEMaxAffine(num_samples, a_k_list, b_k_list, support_C=C, support_d=d, used_norm=2, vp_suffix='')
        # for the DR-CVaR synthesis problem, wce is a constraint
        dr_cvar_bound = [wce.objective.expr <= delta]
        dr_cvar_bound.extend(wce.constraints)
        test_prob = cp.Problem(cp.Minimize(g), dr_cvar_bound)
        # solve the problem we are testing
        test_prob.param_dict['eps'].value = eps
        test_prob.param_dict['samples'].value = xiT
        test_prob.solve(solver=solver)
        test_result = g.value

        # Compare results
        if VERBOSE:
            print_test_results(reference_result, test_result, test_prob.status)
        close = np.isclose(test_result, reference_result, rtol=1e-5, atol=1e-5)
        all_close = np.all(close)
        self.assertTrue(all_close)

    def test_a_cst_1D_b_affine_1D_safaoui_drcvar_synth(self):
        """
        Same as test_a_cst_mD_b_affine_1D_safaoui_drcvar_synth but with

        a_k = a with
            a \in \mathbb{R},

        and 1D samples given by h * xi where xi are the 2D samples

        Also tests making WCE expression a constraint.
        """

        from examples.reference_examples.safaoui_dr_cvar_halfspace import drcvar_halfspace_synthesis, generate_safaoui_halfspace_prob_dataset

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
        b_k_list = [-1 / alpha * g + 1 / alpha * r[0] + (1-1/alpha) * tau, tau]
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

        # Compare results
        if VERBOSE:
            print_test_results(reference_result, test_result, test_prob.status)
        close = np.isclose(test_result, reference_result, rtol=1e-5, atol=1e-5)
        all_close = np.all(close)
        self.assertTrue(all_close)


class TestWassDRExpectation(unittest.TestCase):
    def test_adding_dr_expectations(self):
        from examples.reference_examples.esfahani_portfolio_sec7example import generate_esfahani_portfolio_prob_dataset, EsfahaniPortfolioProb

        eps = 0.1  # Wasserstein radius
        solver = cp.ECOS

        # problem settings
        num_samples = 30
        num_assets = 10
        num_sim = 1
        # generate the dataset
        xi_dataset = generate_esfahani_portfolio_prob_dataset(num_samples, num_assets, num_sim)
        xi = xi_dataset[:, :, 0]

        # set up and solve the reference optimization problem
        ref_prob = EsfahaniPortfolioProb(num_samples, num_assets, rho=0, expectation_rho=2)
        ref_prob.set_params(eps, xi)
        ref_prob.solve(solver=solver)
        reference_result = ref_prob.get_result()
        reference_obj = ref_prob.problem.value

        # set up and solve the problem we're testing
        x = cp.Variable(num_assets, name='x')
        portfolio_constr = cp.Problem(cp.Minimize(0), [cp.sum(x) == 1, x >= 0])
        a_e1 = -1 * x
        a_e2 = -1 * x
        expect1 = WassDRExpectation(num_samples, a_e1, used_norm=1)
        expect2 = WassDRExpectation(num_samples, a_e2, used_norm=1)
        test_prob = expect1 + expect2
        test_prob += portfolio_constr
        # solve the problem we are testing
        for par in test_prob.param_dict.keys():
            if 'eps' in par:
                test_prob.param_dict[par].value = eps
            if 'samples' in par:
                test_prob.param_dict[par].value = xi
        test_prob.solve(solver=solver)
        test_result = x.value
        test_obj = test_prob.value

        # Compare results
        if VERBOSE:
            print_test_results(reference_result, test_result, test_prob.status)
            print_test_results(reference_obj, test_obj, test_prob.status)
        close = np.isclose(test_result, reference_result, rtol=1e-5, atol=1e-5)
        all_close = np.all(close)
        obj_close = np.isclose(test_obj, reference_obj, rtol=1e-5, atol=1e-5)
        self.assertTrue(all_close)
        self.assertTrue(obj_close)

    def test_dr_expectation_multiplication(self):
        from examples.reference_examples.esfahani_portfolio_sec7example import generate_esfahani_portfolio_prob_dataset, EsfahaniPortfolioProb

        eps = 0.1  # Wasserstein radius
        solver = cp.ECOS

        # problem settings
        num_samples = 30
        num_assets = 10
        num_sim = 1
        # generate the dataset
        xi_dataset = generate_esfahani_portfolio_prob_dataset(num_samples, num_assets, num_sim)
        xi = xi_dataset[:, :, 0]

        scaling = 3

        # set up and solve the reference optimization problem
        ref_prob = EsfahaniPortfolioProb(num_samples, num_assets, rho=0, expectation_rho=scaling)
        ref_prob.set_params(eps, xi)
        ref_prob.solve(solver=solver)
        reference_result = ref_prob.get_result()
        reference_obj = ref_prob.problem.value

        # set up and solve the problem we're testing
        x = cp.Variable(num_assets, name='x')
        portfolio_constr = cp.Problem(cp.Minimize(0), [cp.sum(x) == 1, x >= 0])
        a_e = -1 * x
        expect = WassDRExpectation(num_samples, a_e, used_norm=1)
        test_prob = scaling * expect
        test_prob += portfolio_constr
        # solve the problem we are testing
        for par in test_prob.param_dict.keys():
            if 'eps' in par:
                test_prob.param_dict[par].value = eps
            if 'samples' in par:
                test_prob.param_dict[par].value = xi
        test_prob.solve(solver=solver)
        test_result = x.value
        test_obj = test_prob.value

        # Compare results
        if VERBOSE:
            print_test_results(reference_result, test_result, test_prob.status)
            print_test_results(reference_obj, test_obj, test_prob.status)
        close = np.isclose(test_result, reference_result, rtol=1e-5, atol=1e-5)
        all_close = np.all(close)
        obj_close = np.isclose(test_obj, reference_obj, rtol=1e-5, atol=1e-5)
        self.assertTrue(all_close)
        self.assertTrue(obj_close)

        # try right multiplication as well
        a_e = -1 * x
        expect = WassDRExpectation(num_samples, a_e, used_norm=1)
        test_prob = expect * scaling
        test_prob += portfolio_constr
        # solve the problem we are testing
        for par in test_prob.param_dict.keys():
            if 'eps' in par:
                test_prob.param_dict[par].value = eps
            if 'samples' in par:
                test_prob.param_dict[par].value = xi
        test_prob.solve(solver=solver)
        test_result = x.value
        test_obj = test_prob.value

        # Compare results
        if VERBOSE:
            print_test_results(reference_result, test_result, test_prob.status)
            print_test_results(reference_obj, test_obj, test_prob.status)
        close = np.isclose(test_result, reference_result, rtol=1e-5, atol=1e-5)
        all_close = np.all(close)
        obj_close = np.isclose(test_obj, reference_obj, rtol=1e-5, atol=1e-5)
        self.assertTrue(all_close)
        self.assertTrue(obj_close)

    def test_dr_expectation_multiplication_zero(self):
        from examples.reference_examples.esfahani_portfolio_sec7example import generate_esfahani_portfolio_prob_dataset

        eps = 0.1  # Wasserstein radius
        solver = cp.ECOS

        # problem settings
        num_samples = 30
        num_assets = 10
        num_sim = 1
        # generate the dataset
        xi_dataset = generate_esfahani_portfolio_prob_dataset(num_samples, num_assets, num_sim)
        xi = xi_dataset[:, :, 0]

        scaling = 0

        # set up and solve the reference optimization problem
        x = cp.Variable(num_assets, name='x')
        a_e = -scaling * x
        expect = WassDRExpectation(num_samples, a_e, used_norm=1)
        portfolio_constr = cp.Problem(cp.Minimize(0), [cp.sum(x) == 1, x >= 0])
        ref_prob = expect + portfolio_constr
        for par in ref_prob.param_dict.keys():
            if 'eps' in par:
                ref_prob.param_dict[par].value = eps
            if 'samples' in par:
                ref_prob.param_dict[par].value = xi
        ref_prob.solve(solver=solver)
        reference_result = x.value

        # set up and solve the problem we're testing
        a_e = -1 * x
        expect = WassDRExpectation(num_samples, a_e, used_norm=1)
        test_prob = scaling * expect
        test_prob += portfolio_constr
        # solve the problem we are testing
        for par in test_prob.param_dict.keys():
            if 'eps' in par:
                test_prob.param_dict[par].value = eps
            if 'samples' in par:
                test_prob.param_dict[par].value = xi
        test_prob.solve(solver=solver)
        test_result = x.value

        # Compare results
        if VERBOSE:
            print_test_results(reference_result, test_result, test_prob.status)
        close = np.isclose(test_result, reference_result, rtol=1e-5, atol=1e-5)
        all_close = np.all(close)
        self.assertTrue(all_close)

        # try right multiplication as well
        a_e = -1 * x
        expect = WassDRExpectation(num_samples, a_e, used_norm=1)
        test_prob = expect * scaling
        test_prob += portfolio_constr
        # solve the problem we are testing
        for par in test_prob.param_dict.keys():
            if 'eps' in par:
                test_prob.param_dict[par].value = eps
            if 'samples' in par:
                test_prob.param_dict[par].value = xi
        test_prob.solve(solver=solver)
        test_result = x.value

        # Compare results
        if VERBOSE:
            print_test_results(reference_result, test_result, test_prob.status)
        close = np.isclose(test_result, reference_result, rtol=1e-5, atol=1e-5)
        all_close = np.all(close)
        self.assertTrue(all_close)

    def test_dr_expectation_multiplication_neg(self):
        from examples.reference_examples.esfahani_portfolio_sec7example import generate_esfahani_portfolio_prob_dataset

        eps = 0.1  # Wasserstein radius
        solver = cp.ECOS

        # problem settings
        num_samples = 30
        num_assets = 10
        num_sim = 1
        # generate the dataset
        xi_dataset = generate_esfahani_portfolio_prob_dataset(num_samples, num_assets, num_sim)
        xi = xi_dataset[:, :, 0]

        scaling = -4

        # set up and solve the reference optimization problem
        x = cp.Variable(num_assets, name='x')
        a_e = -scaling * x
        expect = WassDRExpectation(num_samples, a_e, used_norm=1)
        portfolio_constr = cp.Problem(cp.Minimize(0), [cp.sum(x) == 1, x >= 0])
        ref_prob = expect + portfolio_constr
        for par in ref_prob.param_dict.keys():
            if 'eps' in par:
                ref_prob.param_dict[par].value = eps
            if 'samples' in par:
                ref_prob.param_dict[par].value = xi
        ref_prob.solve(solver=solver)
        reference_result = x.value

        # set up and solve the problem we're testing
        a_e = -1 * x
        expect = WassDRExpectation(num_samples, a_e, used_norm=1)
        test_prob = scaling * expect
        test_prob += portfolio_constr
        # solve the problem we are testing
        for par in test_prob.param_dict.keys():
            if 'eps' in par:
                test_prob.param_dict[par].value = eps
            if 'samples' in par:
                test_prob.param_dict[par].value = xi
        test_prob.solve(solver=solver)
        test_result = x.value

        # Compare results
        if VERBOSE:
            print_test_results(reference_result, test_result, test_prob.status)
        close = np.isclose(test_result, reference_result, rtol=1e-5, atol=1e-5)
        all_close = np.all(close)
        self.assertTrue(all_close)

        # try right multiplication as well
        a_e = -1 * x
        expect = WassDRExpectation(num_samples, a_e, used_norm=1)
        test_prob = expect * scaling
        test_prob += portfolio_constr
        # solve the problem we are testing
        for par in test_prob.param_dict.keys():
            if 'eps' in par:
                test_prob.param_dict[par].value = eps
            if 'samples' in par:
                test_prob.param_dict[par].value = xi
        test_prob.solve(solver=solver)
        test_result = x.value

        # Compare results
        if VERBOSE:
            print_test_results(reference_result, test_result, test_prob.status)
        close = np.isclose(test_result, reference_result, rtol=1e-5, atol=1e-5)
        all_close = np.all(close)
        self.assertTrue(all_close)

    def test_dr_expectation_division(self):
        from examples.reference_examples.esfahani_portfolio_sec7example import generate_esfahani_portfolio_prob_dataset

        eps = 0.1  # Wasserstein radius
        solver = cp.ECOS

        # problem settings
        num_samples = 30
        num_assets = 10
        num_sim = 1
        # generate the dataset
        xi_dataset = generate_esfahani_portfolio_prob_dataset(num_samples, num_assets, num_sim)
        xi = xi_dataset[:, :, 0]

        div_scaling = 3

        # set up and solve the reference optimization problem
        x = cp.Variable(num_assets, name='x')
        a_e = - x / div_scaling
        expect = WassDRExpectation(num_samples, a_e, used_norm=1)
        portfolio_constr = cp.Problem(cp.Minimize(0), [cp.sum(x) == 1, x >= 0])
        ref_prob = expect + portfolio_constr
        for par in ref_prob.param_dict.keys():
            if 'eps' in par:
                ref_prob.param_dict[par].value = eps
            if 'samples' in par:
                ref_prob.param_dict[par].value = xi
        ref_prob.solve(solver=solver)
        reference_result = x.value

        # set up and solve the problem we're testing
        a_e = -1 * x
        expect = WassDRExpectation(num_samples, a_e, used_norm=1)
        test_prob = expect / div_scaling
        test_prob += portfolio_constr
        # solve the problem we are testing
        for par in test_prob.param_dict.keys():
            if 'eps' in par:
                test_prob.param_dict[par].value = eps
            if 'samples' in par:
                test_prob.param_dict[par].value = xi
        test_prob.solve(solver=solver)
        test_result = x.value

        # Compare results
        if VERBOSE:
            print_test_results(reference_result, test_result, test_prob.status)
        close = np.isclose(test_result, reference_result, rtol=1e-5, atol=1e-5)
        all_close = np.all(close)
        self.assertTrue(all_close)


class TestWassWCEMaxAffineFunctions(unittest.TestCase):
    def test_dr_expectation_and_dr_cvar(self):
        from examples.reference_examples.esfahani_portfolio_sec7example import EsfahaniPortfolioProb, \
            generate_esfahani_portfolio_prob_dataset

        eps = 0.01  # Wasserstein radius
        solver = cp.ECOS

        # problem settings
        num_samples = 1
        num_assets = 2
        num_sim = 1

        # generate the dataset
        xi_dataset = generate_esfahani_portfolio_prob_dataset(num_samples, num_assets, num_sim)
        xi = xi_dataset[:, :, 0]

        # set up and solve the reference optimization problem
        ref_prob = EsfahaniPortfolioProb(num_samples, num_assets, expectation_rho=1, rho=1)
        ref_prob.set_params(eps, xi)
        ref_prob.solve(solver=solver)
        reference_result = ref_prob.get_result()
        reference_obj = ref_prob.problem.value

        # get some parameters from the reference problem
        alpha, rho = ref_prob.alpha, ref_prob.rho

        # set up and solve the problem we're testing
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
        # solve the problem we are testing
        for par in test_prob.param_dict.keys():
            if 'eps' in par:
                test_prob.param_dict[par].value = eps
            if 'samples' in par:
                test_prob.param_dict[par].value = xi
        test_prob.solve(solver=solver)
        test_result = x.value
        test_obj = test_prob.value

        # Compare results
        if VERBOSE:
            print("~~~~~ Solution: ~~~~~")
            print_test_results(reference_result, test_result, test_prob.status)
            print("~~~~ Objective: ~~~~")
            print_test_results(reference_obj, test_obj, test_prob.status)
        close = np.isclose(test_result, reference_result, rtol=1e-5, atol=1e-5)
        all_close = np.all(close)
        obj_close = np.isclose(test_obj, reference_obj, rtol=1e-5, atol=1e-5)
        self.assertTrue(all_close)
        self.assertTrue(obj_close)


if __name__ == '__main__':
    unittest.main()