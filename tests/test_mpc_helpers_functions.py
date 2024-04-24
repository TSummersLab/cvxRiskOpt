import unittest
import numpy as np
import cvxpy as cp
VERBOSE = True  # set to True to print additional data


class TestMPCHelpersFunctions(unittest.TestCase):
    def test_cp_var_mat_to_list(self):
        from cvxRiskOpt.mpc_helpers import cp_var_mat_to_list

        # Test case: (3, 4) matrix where horizon is 4
        mat0 = cp.Variable((3, 4))
        result0 = cp_var_mat_to_list(mat0, 4)
        self.assertEqual([var.shape for var in result0], [(3,), (3,), (3,), (3,)])

        # Test case: (4, 3) matrix where horizon is 4
        mat1 = cp.Variable((4, 3))
        result1 = cp_var_mat_to_list(mat1, 4)
        self.assertEqual([var.shape for var in result1], [(3,), (3,), (3,), (3,)])

        # Test case: (3,) of horizon 1
        mat2 = cp.Variable(3)
        result2 = cp_var_mat_to_list(mat2, 1)
        self.assertEqual([var.shape for var in result2], [(3,)])

        # Test case: (4,) where 4 is horizon
        mat3 = cp.Variable(4)
        result3 = cp_var_mat_to_list(mat3, 4)
        self.assertEqual([var.shape for var in result3], [(), (), (), ()])

        # Test case: Unsupported shape
        mat4 = cp.Variable((2, 3))
        with self.assertRaises(ValueError):
            cp_var_mat_to_list(mat4, 4)

    def test_expect_cov_x_t(self):
        from cvxRiskOpt.mpc_helpers import expect_cov_x_t, cp_var_mat_to_list
        # Test case 1: Simple case with horizon 1
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[0.5], [0.5]])
        x0_mean = np.array([0, 0])
        x0_cov = np.array([[1, 0], [0, 1]])
        w_mean = [np.array([0, 0])]
        w_cov = [np.array([[0.1, 0], [0, 0.1]])]

        horizon = 4
        u = cp.Variable(horizon)
        u_list = cp_var_mat_to_list(u, horizon)
        xtp1_mean, xtp1_cov = expect_cov_x_t(horizon-1, A, B, x0_mean, x0_cov, u_list, w_mean, w_cov)
        self.assertEqual(xtp1_mean.shape, (2, 1))
        self.assertEqual(xtp1_cov.shape, (2, 2))

        # Test case 2: Test ValueError when lengths of u and w_mean are not equal
        u = [np.array([1]), np.array([2])]
        with self.assertRaises(ValueError):
            expect_cov_x_t(1, A, B, x0_mean, x0_cov, u, w_mean, w_cov)

        # Test case 3: Test ValueError when lengths of u and w_cov are not equal
        u = [np.array([1])]
        w_cov = [np.array([[0.1, 0], [0, 0.1]]), np.array([[0.2, 0], [0, 0.2]])]
        with self.assertRaises(ValueError):
            expect_cov_x_t(1, A, B, x0_mean, x0_cov, u, w_mean, w_cov)

        #
        xt_mean, xt_cov = expect_cov_x_t(0, A, B, x0_mean, x0_cov, u, w_mean, w_cov)
        np.testing.assert_array_equal(xt_mean, x0_mean)
        np.testing.assert_array_equal(xt_cov, x0_cov)


if __name__ == '__main__':
    unittest.main()
