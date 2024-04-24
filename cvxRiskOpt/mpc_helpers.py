"""
Copyright 2024 Sleiman Safaoui
Licensed under the GNU GENERAL PUBLIC LICENSE Version 3.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.gnu.org/licenses/gpl-3.0.en.html
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""
Some helper functions for designing and solving MPC problems
"""
import numpy as np
import cvxpy as cp


def cp_var_mat_to_list(mat: cp.Variable | cp.Parameter, horizon: int):
    """
    Converts a 1D or 2D cp.Variable matrix to a list of cp.Variable vectors.
    E.g.
    (3, 4) matrix where horizon is 4 turns into [(3,), (3,), (3,), (3,)].
    (3,) of horizon 1 turns into [(3,)]
    (4,) where 4 is horizon turns into [(), (), (), ()]
    :param mat: (m, horizon), (horizon, m), (m,), or (horizon,) cp.Variable
    :param horizon: horizon (e.g. MPC horizon).
    :return:
    """
    # Get the shape of u
    shape = mat.shape

    # Check the shape of u and generate the list accordingly
    if horizon == 1:
        # If t = 1, return [u]
        return [mat]
    elif len(shape) == 1 and shape[0] == horizon:
        # If t > 1 and shape = (t,), return [u[0], ..., u[t-1]]
        return [mat[i] for i in range(horizon)]
    elif len(shape) == 2 and shape[1] == horizon:
        # If t > 1 and shape = (m, t), return [u[:,0], ..., u[:, t-1]]
        return [mat[:, i] for i in range(horizon)]
    elif len(shape) == 2 and shape[0] == horizon:
        # If t > 1 and shape = (t, m), return [u[0, :], ..., u[t-1, :]]
        return [mat[i, :] for i in range(horizon)]
    else:
        raise ValueError("Input variable u has an unsupported shape.")


def expect_cov_x_t(t, A, B,
                   x0_mean: cp.Variable | cp.Parameter | cp.Expression,
                   x0_cov: np.ndarray,
                   u: list,
                   w_mean: list, w_cov: list):
    """
    Computes the expected next state E(x_{t}) and its covariance Cov(x_{t})
    for a linear system x_{t+1} = Ax_t + Bu_t + w_t.
    E(x_{t}) = A^{t} x0_mean + sum_{i=0}^{t-1} (A^i B u_{t-1-i} + A^i w_{t-1-i}_mean)
    Cov(x_{t}) = A^{t} x0_cov A^{t}^T + sum_{i=0}^{t-1} (A^i w_{t-1-i}_cov A^i^T)
    :param t: current time step for x_t (t >= 1)
    :param A: dynamics matrix
    :param B: input matrix
    :param x0_mean: mean state at t=0
    :param x0_cov: covariance of state at t=0
    :param u: list of control decision variables
    :param w_mean: list of noise mean value
    :param w_cov: list of noise covariance values
    :return:
    """
    if isinstance(A, (int, float)):
        n = 1
    elif isinstance(A, np.ndarray):
        n = A.shape[0]
    else:
        raise ValueError("A can either be a square matrix or a scalar")
    if isinstance(B, (int, float)):
        m = 1
    elif isinstance(B, np.ndarray):
        if B.ndim < 2:
            m = 1
        elif B.ndim == 2:
            m = B.shape[1]
        else:
            raise ValueError("B can only be up to 2-dimensional")
    else:
        raise ValueError("B can either be a matrix or a scalar")

    if t < 0:
        raise ValueError("t must be > 0")
    if t == 0:
        print("Expectation and covariance of current state are simply x0_mean and x0_cov")
        return x0_mean, x0_cov

    if len(u) != len(w_mean):
        if len(w_mean) == 1:
            w_mean *= len(u)
        else:
            raise ValueError("u and w_mean should either have the same length or w_mean should only have one value")
    if len(u) != len(w_cov):
        if len(w_cov) == 1:
            w_cov *= len(u)
        else:
            raise ValueError("u and w_cov should either have the same length or w_cov should only have one value")

    if x0_mean.ndim == 1 and u[0].ndim == 2:
        x0_mean = cp.reshape(x0_mean, (x0_mean.shape[0], 1))
    if n == 1:
        At = A ** t
        xt_mean = At * x0_mean
        xt_cov = At ** 2 * x0_cov
    else:
        At = np.linalg.matrix_power(A, t)
        xt_mean = At @ x0_mean
        xt_cov = At @ x0_cov @ At.T

    for i in range(t):
        Ai = A ** i if n == 1 else np.linalg.matrix_power(A, i)
        Bu = B * u[t-1-i] if m == 1 else B @ u[t-1-i]
        AiBu = Ai * Bu if n == 1 else Ai @ Bu
        Aiw_mean = Ai * w_mean[t-1-i] if n == 1 else Ai @ w_mean[t-1-i]
        xt_mean = xt_mean + AiBu + Aiw_mean

        AcovA = w_cov[t-1-i] * Ai ** 2 if n == 1 else Ai @ w_cov[t-1-i] @ Ai.T
        xt_cov = xt_cov + AcovA

        # if u[0].shape == ():
        #     xt_mean = xt_mean + Ai @ B * u[t-1-i]
        # else:
        #     xt_mean = xt_mean + Ai @ B @ u[t-1-i]
        # if w_mean[0].shape == ():
        #     xt_mean = xt_mean + Ai * w_mean[t-1-i]
        # else:
        #     xt_mean = xt_mean + Ai @ w_mean[t-1-i]
        # if w_cov[0].shape == ():
        #     xt_cov = xt_cov + w_cov[t-1-i] * Ai @ Ai.T
        # else:
        #     xt_cov = xt_cov + Ai @ w_cov[t-1-i] @ Ai.T

    return xt_mean, xt_cov


def lin_mpc_expect_xQx(t: int, horizon: int,
                       A: int | float | np.ndarray, B: int | float | np.ndarray,
                       u: list | cp.Variable,
                       Q: int | float | np.ndarray,
                       x0_mean: cp.Parameter | cp.Variable | cp.Expression,
                       x0_cov: np.ndarray = None,
                       w_mean: list | np.ndarray = None,
                       w_cov: list | np.ndarray = None):
    """
    Finds the E(x_t^T Q x_t) term
    where
    x_t is a random variable at timestep t in the MPC horizon
    and
    the dynamics are:
    x_{t+1} = Ax_t + Bu_t + w_t.
    :param t: current control time step (starting from 0)
    :param horizon: MPC horizon length
    :param A: dynamics matrix
    :param B: input matrix
    :param u: list of control decision variables (or cp Variable)
    :param Q: state cost matrix x^T Q x
    :param x0_mean: mean state at t=0
    :param x0_cov: covariance of state at t=0
    :param w_mean: list of noise mean value (or single noise mean)
    :param w_cov: list of noise covariance values (or single noise covar)
    :return:
    """
    # handle optional arguments
    n = x0_mean.shape[0] if x0_mean.ndim > 0 else 1
    if x0_cov is None:
        x0_cov = np.zeros((n, n))
    if w_mean is None:
        w_mean = np.zeros(n)
    if w_cov is None:
        w_cov = np.zeros((n, n))

    # format inputs
    if isinstance(w_cov, np.ndarray):
        w_cov = [w_cov]
    if isinstance(w_mean, np.ndarray):
        w_mean = [w_mean]
    if isinstance(u, cp.Variable):
        u = cp_var_mat_to_list(u, horizon)

    # get expectation and covariance of the state x_{t+`}
    xtp1_mean, xtp1_cov = expect_cov_x_t(t, A, B, x0_mean, x0_cov, u, w_mean, w_cov)
    # return the expression for E(x^T Q x)
    traceQxcov = Q * xtp1_cov if n == 1 else cp.trace(Q @ xtp1_cov)
    QxmeanQ = cp.square(xtp1_mean) * Q if n == 1 else cp.QuadForm(xtp1_mean, Q)
    return traceQxcov + QxmeanQ, {"x_mean": xtp1_mean, "x_cov": xtp1_cov}
