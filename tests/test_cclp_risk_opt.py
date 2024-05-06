import unittest
import numpy as np
import cvxpy as cp

VERBOSE = False  # set to True to print additional data


class TestCCLPRiskOptHelperFunctions(unittest.TestCase):
    def test_check_b_term(self):
        from cvxRiskOpt.cclp_risk_opt import _check_b_term

        # Test case: b is not None and is an integer
        b = 1
        b_present = False
        self.assertTrue(_check_b_term(b, b_present))

        # Test case: b is not None and is a float
        b = 1.0
        b_present = False
        self.assertTrue(_check_b_term(b, b_present))

        # Test case: b is not None and is a cvxpy Variable
        b = cp.Variable()
        b_present = False
        self.assertTrue(_check_b_term(b, b_present))

        # Test case: b is not None and is a cvxpy Expression
        b = cp.Variable() + 1
        b_present = False
        self.assertTrue(_check_b_term(b, b_present))

        # Test case: b is None
        b = None
        b_present = False
        self.assertFalse(_check_b_term(b, b_present))

    def test_check_xi2_term(self):
        from cvxRiskOpt.cclp_risk_opt import _check_xi2_term

        # Test case: xi2_hat and gam22 are not None
        xi2_hat = 1.0
        gam22 = 2.0
        xi2_present = False
        expected_output = (np.array([[2.0]]), True)
        self.assertEqual(_check_xi2_term(xi2_hat, gam22, xi2_present), expected_output)

        # Test case: xi2_hat and gam22 are None
        xi2_hat = None
        gam22 = None
        xi2_present = False
        expected_output = (None, False)
        self.assertEqual(_check_xi2_term(xi2_hat, gam22, xi2_present), expected_output)

        # Test case: only one of xi2_hat and gam22 is None
        xi2_hat = 1.0
        gam22 = None
        xi2_present = False
        expected_output = (None, False)
        self.assertEqual(_check_xi2_term(xi2_hat, gam22, xi2_present), expected_output)
        xi2_hat = None
        gam22 = 3.0
        xi2_present = False
        expected_output = (np.array([3.0]), False)
        self.assertEqual(_check_xi2_term(xi2_hat, gam22, xi2_present), expected_output)

        # Test case: xi2_hat is not a scalar
        xi2_hat = [1, 2, 3]
        gam22 = 2.0
        xi2_present = False
        with self.assertRaises(ValueError):
            _check_xi2_term(xi2_hat, gam22, xi2_present)

        # Test case: gam22 is not a scalar
        xi2_hat = 1.0
        gam22 = [1, 2, 3]
        xi2_present = False
        with self.assertRaises(ValueError):
            _check_xi2_term(xi2_hat, gam22, xi2_present)

    def test_check_a_xi1_term(self):
        from cvxRiskOpt.cclp_risk_opt import _check_a_xi1_term


        # Test case: a, xi1_hat, and gam11 are not None
        a = np.array([1.0, 2.0, 3.0])
        xi1_hat = np.array([1.0, 2.0, 3.0])
        gam11 = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        a_xi1_present = False
        expected_output = (a, xi1_hat, gam11, 3, True)
        self.assertEqual(_check_a_xi1_term(a, xi1_hat, gam11, a_xi1_present), expected_output)

        # Test case: a, xi1_hat, and gam11 are None
        a = None
        xi1_hat = None
        gam11 = None
        a_xi1_present = False
        expected_output = (None, None, None, -1, False)
        self.assertEqual(_check_a_xi1_term(a, xi1_hat, gam11, a_xi1_present), expected_output)

        # Test case: a is not a scalar, cp Var or expression, a 1-D list, or a 1-D array
        a = [[1, 2, 3], [4, 5, 6]]
        xi1_hat = np.array([1.0, 2.0, 3.0])
        gam11 = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        a_xi1_present = False
        with self.assertRaises(ValueError):
            _check_a_xi1_term(a, xi1_hat, gam11, a_xi1_present)

        # Test case: xi1_hat is not a scalar, 1-D list or a 1-D array
        a = np.array([1.0, 2.0, 3.0])
        xi1_hat = [[1, 2, 3], [4, 5, 6]]
        gam11 = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        a_xi1_present = False
        with self.assertRaises(ValueError):
            _check_a_xi1_term(a, xi1_hat, gam11, a_xi1_present)

        # Test case: gam11 is not a scalar, 2-D list/array
        a = np.array([1.0, 2.0, 3.0])
        xi1_hat = np.array([1.0, 2.0, 3.0])
        gam11 = [1.0, 2.0, 3.0]
        a_xi1_present = False
        with self.assertRaises(ValueError):
            _check_a_xi1_term(a, xi1_hat, gam11, a_xi1_present)

    def test_check_gam12_term(self):
        from cvxRiskOpt.cclp_risk_opt import _check_gam12_term

        # Test case: gam12 is None
        gam12 = None
        a_len = 3
        a_xi1_present = True
        xi2_present = True
        expected_output = np.zeros((a_len, 1))
        np.testing.assert_array_equal(_check_gam12_term(gam12, a_len, a_xi1_present, xi2_present), expected_output)

        # Test case: gam12 is a scalar and a_len is 1
        gam12 = 2.0
        a_len = 1
        a_xi1_present = True
        xi2_present = True
        expected_output = np.array([[2.0]])
        np.testing.assert_array_equal(_check_gam12_term(gam12, a_len, a_xi1_present, xi2_present), expected_output)

        # Test case: gam12 is a scalar and a_len is greater than 1
        gam12 = 2.0
        a_len = 3
        a_xi1_present = True
        xi2_present = True
        with self.assertRaises(ValueError):
            _check_gam12_term(gam12, a_len, a_xi1_present, xi2_present)

        # Test case: gam12 is a 1D array
        gam12 = np.array([1.0, 2.0, 3.0])
        a_len = 3
        a_xi1_present = True
        xi2_present = True
        expected_output = np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_array_equal(_check_gam12_term(gam12, a_len, a_xi1_present, xi2_present), expected_output)

        # Test case: gam12 is a 2D array
        gam12 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        a_len = 3
        a_xi1_present = True
        xi2_present = True
        with self.assertRaises(ValueError):
            _check_gam12_term(gam12, a_len, a_xi1_present, xi2_present)

        # Test case: a_xi1_present and xi2_present are False but gam12 is not None
        gam12 = np.array([1.0, 2.0, 3.0])
        a_len = 3
        a_xi1_present = False
        xi2_present = False
        with self.assertRaises(ValueError):
            _check_gam12_term(gam12, a_len, a_xi1_present, xi2_present)

    def test_format_inputs(self):
        from cvxRiskOpt.cclp_risk_opt import _format_inputs

        # Test case: all inputs are not None
        eps = 0.1
        a = np.array([1.0, 2.0, 3.0])
        b = 1.0
        xi1_hat = np.array([1.0, 2.0, 3.0])
        xi2_hat = 1.0
        gam11 = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        gam12 = np.array([1.0, 2.0, 3.0])
        gam22 = 2.0
        output = _format_inputs(eps, a, b, xi1_hat, xi2_hat, gam11, gam12, gam22)
        expected_output = (a, b, xi1_hat, xi2_hat, gam11, np.expand_dims(gam12,  axis=1), gam22, True, True, True)
        for out, exp_out in zip(output, expected_output):
            if isinstance(out, np.ndarray):
                np.testing.assert_array_equal(out, exp_out)
            else:
                self.assertEqual(out, exp_out)

        # Test case: eps is not in (0, 0.5]
        eps = 0.6
        a = np.array([1.0, 2.0, 3.0])
        b = 1.0
        xi1_hat = np.array([1.0, 2.0, 3.0])
        xi2_hat = 1.0
        gam11 = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        gam12 = np.array([1.0, 2.0, 3.0])
        gam22 = 2.0
        with self.assertRaises(ValueError):
            _format_inputs(eps, a, b, xi1_hat, xi2_hat, gam11, gam12, gam22)

        # Test case: a, xi1_hat, and gam11 are None but gam12 is not None
        eps = 0.1
        a = None
        b = 1.0
        xi1_hat = None
        xi2_hat = 1.0
        gam11 = None
        gam12 = np.array([1.0, 2.0, 3.0])
        gam22 = 2.0
        with self.assertRaises(ValueError):
            _format_inputs(eps, a, b, xi1_hat, xi2_hat, gam11, gam12, gam22)

    def test_det_cclp(self):
        from cvxRiskOpt.cclp_risk_opt import _det_cclp, _deconstruct_gamma

        # Test case: all inputs are not None
        kappa_e = 0.1
        a = np.array([1.0, 2.0, 3.0])
        b = 1.0
        xi1_hat = np.array([1.0, 2.0, 3.0])
        xi2_hat = 1.0
        gam = np.array([[1.62200806, 0.5141724,  0.64933035, 1.02301933],
                         [0.5141724,  1.47296812, 0.51560885, 0.80548296],
                         [0.64933035, 0.51560885, 1.91069752, 1.42846559],
                         [1.02301933, 0.80548296, 1.42846559, 3.55243959]])
        gam11, gam12, gam22 = _deconstruct_gamma(gam)
        a_xi1_present = True
        xi2_present = True
        b_present = True
        assume_sym = False
        assume_psd = False
        with self.assertRaises(ValueError):
            _det_cclp(kappa_e, a, b, xi1_hat, xi2_hat, gam11, gam12, gam22, a_xi1_present, xi2_present, b_present,
                      assume_sym, assume_psd)
        b = cp.Variable()
        output = _det_cclp(kappa_e, a, b, xi1_hat, xi2_hat, gam11, gam12, gam22, a_xi1_present, xi2_present, b_present,
                           assume_sym, assume_psd)
        self.assertTrue(output.is_dcp())
        b = 1
        a = cp.Variable(3)
        output = _det_cclp(kappa_e, a, b, xi1_hat, xi2_hat, gam11, gam12, gam22, a_xi1_present, xi2_present, b_present,
                           assume_sym, assume_psd)
        self.assertTrue(output.is_dcp())
        a = np.array([1.0, 2.0, 3.0])
        xi1_hat = cp.Variable(3)
        output = _det_cclp(kappa_e, a, b, xi1_hat, xi2_hat, gam11, gam12, gam22, a_xi1_present, xi2_present, b_present,
                           assume_sym, assume_psd)
        self.assertTrue(output.is_dcp())

        # Test case: only b term present
        kappa_e = 0.1
        a = None
        b = 1.0
        xi1_hat = None
        xi2_hat = None
        gam11 = None
        gam12 = None
        gam22 = None
        a_xi1_present = False
        xi2_present = False
        b_present = True
        assume_sym = False
        assume_psd = False
        with self.assertRaises(ValueError):
            _det_cclp(kappa_e, a, b, xi1_hat, xi2_hat, gam11, gam12, gam22, a_xi1_present, xi2_present, b_present,
                               assume_sym, assume_psd)
        b = cp.Variable()
        output = _det_cclp(kappa_e, a, b, xi1_hat, xi2_hat, gam11, gam12, gam22, a_xi1_present, xi2_present, b_present,
                           assume_sym, assume_psd)
        self.assertTrue(output.is_dcp())

        # Test case: no expression provided
        kappa_e = 0.1
        a = None
        b = None
        xi1_hat = None
        xi2_hat = None
        gam11 = None
        gam12 = None
        gam22 = None
        a_xi1_present = False
        xi2_present = False
        b_present = False
        assume_sym = False
        assume_psd = False
        with self.assertRaises(ValueError):
            _det_cclp(kappa_e, a, b, xi1_hat, xi2_hat, gam11, gam12, gam22, a_xi1_present, xi2_present, b_present,
                      assume_sym, assume_psd)

        # TODO: ADD TEST CASES FOR THE SPECIAL CASES


class TestCCLPRiskOptFunctions(unittest.TestCase):
    def test_cclp_gauss(self):
        import cvxpy as cp
        from cvxRiskOpt.cclp_risk_opt import cclp_gauss
        from scipy.stats import norm as gauss
        from scipy.linalg import sqrtm
        # simple test for Prob(x * xi + b <= 0) >= 1-eps
        # where
        # x is a decision variable
        # xi is a random variable
        # b is a constant
        eps = 0.1

        if VERBOSE:
            print("~~~~~~~~~")
        xi_mean = 0
        xi_var = 0.01
        x = cp.Variable(name='x')
        b = 1
        constr = cclp_gauss(eps, a=x, b=b, xi1_hat=xi_mean, gam11=xi_var)
        known_reform = gauss.ppf(1-eps) * cp.norm(np.sqrt(xi_var) * x) + x * xi_mean + b <= 0
        if VERBOSE:
            print(constr.expr)
            print(known_reform)

            print("~~~~~~~~~")
        xi_mean = np.array([0, 0])
        xi_var = np.diag([0.01, 0.1])
        x = cp.Variable(2, name='x')
        b = 1
        constr = cclp_gauss(eps, a=x, b=b, xi1_hat=xi_mean, gam11=xi_var)
        known_reform = gauss.ppf(1 - eps) * cp.norm(sqrtm(xi_var) @ x) + (x @ xi_mean + b) <= 0
        if VERBOSE:
            print(constr.expr)
            print(known_reform.expr)
            # print(str(constr.expr) == str(known_reform.expr))

            print("~~~~~~~~~")
        a = np.array([1, 0])
        x = cp.Variable(2, name='x')
        xi_mean = np.array([0, 0]) + x
        xi_var = np.diag([0.01, 0.1])
        b = 0
        constr = cclp_gauss(eps, a=a, b=b, xi1_hat=xi_mean, gam11=xi_var)
        known_reform = gauss.ppf(1 - eps) * cp.norm(sqrtm(xi_var) @ a) + (a @ xi_mean + b) <= 0
        if VERBOSE:
            print(constr.expr)
            print(known_reform.expr)

    def test_simple_1D_mpc(self):
        from examples.cclp_mpc import simple_1d_mpc
        x1, u1 = simple_1d_mpc(use_cpg=False, gen_cpg=False, with_cclp=False, seed=1)
        x2, u2 = simple_1d_mpc(use_cpg=False, gen_cpg=False, with_cclp=True, seed=1)
        if len(x1) == len(x2):
            self.assertFalse(np.allclose(x1, x2, rtol=1e-4, atol=1e-4))
        if len(u1) == len(u2):
            self.assertFalse(np.allclose(u1, u2, rtol=1e-4, atol=1e-4))

    def test_simple_2D_mpc(self):
        from examples.cclp_mpc import simple_2d_mpc
        simple_2d_mpc(use_cpg=False, gen_cpg=False, plot_res=False)

    def test_hvac_mpc(self):
        from examples.cclp_mpc import hvac_mpc_time_varying_constraints
        hvac_mpc_time_varying_constraints(plot_res=False)

    def test_hvac_mpc_reg(self):
        from examples.cclp_mpc import temp_mpc_regulator_time_varying_constraints
        temp_mpc_regulator_time_varying_constraints(plot_res=False)

    def test_portfolio_opt(self):
        from examples.cclp_portfolio_optimization import portfolio_optimization, moment_portfolio_optimization
        solver = cp.CLARABEL
        portfolio_optimization(solver=solver, verbose=VERBOSE)
        moment_portfolio_optimization(num_sim=1, use_cpg=False, gen_code=False, solver=solver)

    def test_production_opt(self):
        from examples.cclp_safe_production_optimization import production_optimization
        cvxpy_x, cro_x = production_optimization(verbose=VERBOSE)
        self.assertTrue(np.allclose(cvxpy_x, cro_x, rtol=1e-4, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
