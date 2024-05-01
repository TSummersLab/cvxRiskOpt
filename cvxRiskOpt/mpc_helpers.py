""""""
"""
cvxRiskOpt: Risk-Based Optimization tool using CVXPY and CVXPYgen
Copyright (C) 2024  Sleiman Safaoui

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

GitHub:
@The-SS
Email:
snsafaoui@gmail.com
sleiman.safaoui@utdallas.edu



Some helper functions for designing and solving MPC problems
"""
import numpy as np
import cvxpy as cp


def cp_var_mat_to_list(mat: cp.Variable | cp.Parameter, horizon: int):
    """
    Converts a 1D or 2D cp.Variable/Parameter matrix to a list of cp.Variable/Parameter vectors.

    Some functions, such as lin_mpc_expect_xQx, require lists or arrays of variables/parameters over time as inputs.
    This function splits a variable/parameter into a list of variables/parameters.
    e.g.
        - (3, 4) matrix where the horizon is 4 --> turns into [(3,), (3,), (3,), (3,)].
        - (3,) of horizon 1 --> turns into [(3,)]
        - (4,) where 4 is horizon --> turns into [(), (), (), ()]

    Arguments:
    ----------
        mat: cp.Variable | cp.Parameter:
            (m, horizon), (horizon, m), (m,), or (horizon,) cp.Variable
        horizon: int:
            mat horizon, e.g. MPC horizon (to identify the "horizon" dimension in mat)

    Returns:
    --------
        list:
            List of cp.Variables or cp.Parameters
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


def expect_cov_x_t(t: int, A: int | float | np.ndarray, B: int | float | np.ndarray,
                   x0_mean: cp.Variable | cp.Parameter | cp.Expression,
                   x0_cov: np.ndarray,
                   u: list,
                   w_mean: list, w_cov: list):
    """
    Computes the expressions for the expected next state :math:`\\mathbb{E}(x_{t})` and its covariance :math:`\\text{Cov}(x_t)`
    for a linear system :math:`x_{t+1} = Ax_t + Bu_t + w_t`.

    The expected value and covariance are found by recursively applying the dynamics stating from an initial state :math:`x_0`
    whose mean :math:`\\overline{x}_0` and covariance :math:`\\Sigma_{x_0}` are known.
    The mean :math:`\\overline{w}` and covariance :math:`\\Sigma_{w}` of the noise must also be known.
    The terms are computed as follows:

    .. math ::
        \\begin{align*}
        \\mathbb{E}(x_{t}) &= A^{t} \\overline{x}_0 + \\sum_{i=0}^{t-1} (A^i B u_{t-1-i} + A^i \\overline{w}_{t-1-i}) \\\\
        \\text{Cov}(x_{t}) &= A^{t} \\Sigma_{x_0} {A^{t}}^T + \\sum_{i=0}^{t-1} (A^i \\Sigma_{w_{t-1-i}} {A^i}^T)
        \\end{align*}

    Arguments:
    ----------
        t: int:
            Current time step for x_t (t >= 1)
        A: int | float | np.ndarray:
            Dynamics matrix
        B: int | float | np.ndarray:
            Input matrix
        x0_mean: cp.Variable | cp.Parameter | cp.Expression:
            Mean state at t=0
        x0_cov: np.ndarray:
            Covariance of state at t=0
        u: list:
            List of control decision variables
        w_mean: list:
            List of noise mean value
        w_cov: list:
            List of noise covariance values

    Returns:
    --------
        cp.Expression:
            Expected value expression of the state at time t (xt_mean)
        cp.Expression:
            Covariance expression of the state at time t (xt_cov)
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
    Finds the expression for :math:`\\mathbb{E}(x_t^T Q x_t)`, the weighted quadratic state cost at time :math:`t`.

    Finds the expression for :math:`\\mathbb{E}(x_t^T Q x_t)`
    where
    :math:`x_t` is a random variable (due to the noise) representing the state at timestep :math:`t` in the MPC horizon
    and
    the dynamics are:
    :math:`x_{t+1} = Ax_t + Bu_t + w_t`.

    Arguments:
    ----------
        t: int:
            Control time step (starting from 0)
        horizon: int:
            MPC horizon length
        A: int | float | np.ndarray:
            Dynamics matrix
        B: int | float | np.ndarray:
            Input matrix
        u: list | cp.Variable
            List of control decision variables (or cp Variable)
        Q: int | float | np.ndarray:
            State cost matrix
        x0_mean: cp.Parameter | cp.Variable | cp.Expression:
            Mean state at :math:`t=0`
        x0_cov: np.ndarray, optional:
            Covariance of state at :math:`t=0`. If not passed, assumed to be zero.
        w_mean: list | np.ndarray, optional:
            List of noise mean value (or single noise mean). If not passed, assumed to be zero.
        w_cov: list | np.ndarray, optional:
            List of noise covariance values (or single noise covar). If not passed, assumed to be zero.

    Returns:
    --------
        cp.Expression:
            Expression for :math:`\\mathbb{E}(x_t^T Q x_t)`
        dict:
            dictionary containing the expressions for the mean and covariance of the state at time :math:`t`
            ("x_mean", "x_cov")
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
