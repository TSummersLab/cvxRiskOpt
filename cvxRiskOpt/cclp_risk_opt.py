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
CVXPY-based Chance Constraints Linear Programs

We implement some results from:
"On Distributionally Robust Chance-Constrained Linear Programs"
"""
import cvxpy as cp
import numpy as np
import scipy
from scipy.stats import norm as gauss


def _construct_gamma(gam11: np.ndarray, gam12: np.ndarray, gam22: np.ndarray):
    """
    Constructs the gamma covariance matrix.

    Combines the gamma terms to form an overall gamma matrix.
    Assumes that all three terms are 2D np.ndarray's.
    :param gam11: (m,m) numpy array
    :param gam12: (m,1) numpy array
    :param gam22: (1,1) numpy array
    :return:
    """
    gamma_top = np.hstack([gam11, gam12])
    gamma_bot = np.hstack([gam12.T, gam22])
    gamma = np.vstack([gamma_top, gamma_bot])
    return gamma


def _deconstruct_gamma(gamma: np.ndarray):
    """
    Deconstructs the gamma matrix into its subcomponents terms.

    [gam11       gam12
     gam12^T    gam22]
    where gam22 is a 1x1 scalar
    :param gamma: (n,n) numpy array
    :return:
    """
    gam11 = gamma[:-1, :-1]
    gam12 = np.expand_dims(gamma[-1, :-1], axis=1)
    gam22 = np.expand_dims(gamma[-1, -1], axis=(0, 1))
    return gam11, gam12, gam22


def _cp_sqrt_quad_form(v: np.ndarray | cp.Variable | cp.Expression,
                       mat: np.ndarray,
                       assume_sym=False, assume_psd=False):
    """
    Computes the sqrt of the quadratic for v^T * mat * v using cvxpy functions.

    Uses cp functions to find sqrt(v^T mat v) where v is a vector and mat is a matrix.
    :param v: 1D numpy array
    :param mat: 2D numpy array
    :param assume_sym: assumes matrix is symmetric. Else, it is checked. Set to True if you are confident
    :param assume_psd: assumes PSD matrix mat. Else, it is checked.
    :return:
    """
    if not assume_sym:
        scipy.linalg.issymmetric(mat)
    if not assume_psd:
        psd = np.all(np.linalg.eigvals(mat) >= 0)
        if not psd:
            raise ValueError("Input matrix mat is not PSD")
    matsqrt = scipy.linalg.sqrtm(mat)
    return cp.norm(matsqrt @ v)


def _check_b_term(b: int | float | cp.Variable | cp.Expression | None,
                  b_present: bool):
    """
    Checks the b term in the CCLP.

    :param b: b term passed by the used
    :param b_present: flag indicating the term's presence
    :return:
    """
    if (b is not None and
            isinstance(b, (int, float, cp.Variable, cp.Expression))):
        b_present = True
    return b_present


def _check_xi2_term(xi2_hat: int | float | None,
                    gam22: int | float | None,
                    xi2_present: bool):
    """
    Check the xi2 term and its presence

    :param xi2_hat:
    :param gam22:
    :param xi2_present:
    :return:
    """
    if xi2_hat is not None and gam22 is not None:
        xi2_present = True
        if not isinstance(xi2_hat, (int, float)):
            raise ValueError("xi2 mean must be a scalar (int or float)")
        if not isinstance(gam22, (int, float)):
            raise ValueError("xi2 variance must be a scalar (int or float)")
        # make it a 2D array
        gam22 = np.array([[gam22]])
    return gam22, xi2_present


def _check_a_xi1_term(a: int | float | list | np.ndarray | cp.Variable | cp.Expression | None,
                      xi1_hat: int | float | list | np.ndarray | cp.Variable | cp.Expression | None,
                      gam11: int | float | list | np.ndarray | None,
                      a_xi1_present: bool):
    if isinstance(a, (cp.Variable, cp.Expression)) and isinstance(xi1_hat, (cp.Variable, cp.Expression)):
        raise NotImplementedError("Currently, only a or xi1 mean may be a cxvpy variable/expression. "
                                  "Note: Product of variables is not convex. "
                                  "Product of parameters is not DPP. ")
    a_len = -1
    if a is not None and xi1_hat is not None and gam11 is not None:
        a_xi1_present = True
        # check a
        if isinstance(a, (int, float)):
            a_len = 1
            a = np.array([a])
        elif isinstance(a, (cp.Variable, cp.Expression)):
            a_len = a.shape[0]
        elif isinstance(a, list):
            a = np.array(a)
            if a.ndim > 1:
                raise ValueError("a can only be (m,)-dimensional")
            a_len = a.shape[0]
        elif isinstance(a, np.ndarray):
            if a.ndim > 1:
                raise ValueError("a can only be (m,)-dimensional")
            a_len = a.shape[0]
        else:
            raise NotImplementedError("a can only be a scalar, cp Var or expression, a 1-D list, or a 1-D array")
        # check xi1_hat
        if isinstance(xi1_hat, (int, float)):
            if a_len > 1:
                raise ValueError("a and xi1 mean must of of same size")
            else:
                xi1_hat = np.array([xi1_hat])
        elif isinstance(xi1_hat, (cp.Variable, cp.Expression)):
            if xi1_hat.ndim == 0:
                xi1_hat = cp.reshape(xi1_hat, (1, ))
            xi1_hat_len = xi1_hat.shape[0]
            if xi1_hat_len != a_len:
                raise ValueError("a and xi1 mean must of of same size")
        elif isinstance(xi1_hat, list):
            xi1_hat = np.array(xi1_hat)
            if xi1_hat.ndim > 1:
                raise ValueError("xi1 mean can only be (m,)-dimensional")
            if a_len != xi1_hat.shape[0]:
                raise ValueError("a and xi1 mean must of of same size")
        elif isinstance(xi1_hat, np.ndarray):
            if xi1_hat.ndim > 1:
                raise ValueError("xi1 mean can only be (m,)-dimensional")
            if a_len != xi1_hat.shape[0]:
                raise ValueError("a and xi1 mean must of of same size")
        else:
            raise NotImplementedError("xi1_hat can only be a scalar, 1-D list or a 1-D array")
        # check gam11
        if isinstance(gam11, list):
            gam11 = np.array(gam11)
        if isinstance(gam11, (int, float)):
            if a_len > 1:
                raise ValueError("gam11 and xi1 mean must of of same size")
            gam11 = np.array([[gam11]])
        elif isinstance(gam11, np.ndarray):
            if gam11.ndim != 2:
                raise ValueError("gam11 must be a scalar or 2D list/array")
            if gam11.shape[0] != gam11.shape[1] != a_len:
                raise ValueError("gam11 must be square and of same size as xi1_hat")
        else:
            raise NotImplementedError("gam11 can only be a scalar, 2-D list/array")
    elif not (a is None and xi1_hat is None and gam11 is None):
        raise NotImplementedError("If any of a, xi1 mean, or xi1 cov provided, then all three terms must be provided")
    return a, xi1_hat, gam11, a_len, a_xi1_present


def _check_gam12_term(gam12: int | float | list | np.ndarray | None,
                      a_len: int, a_xi1_present: bool, xi2_present: bool):
    if a_xi1_present and xi2_present:
        if isinstance(gam12, list):
            gam12 = np.array([gam12]).T
        if gam12 is None:
            gam12 = np.zeros((a_len, 1))
        elif isinstance(gam12, (int, float)):
            if a_len > 1:
                raise ValueError("gam12 length must match a length")
            gam12 = np.array([[gam12]])
        elif isinstance(gam12, np.ndarray):
            if gam12.ndim == 0:
                gam12 = np.expand_dims(gam12, axis=(0, 1))
            elif gam12.ndim == 1:
                gam12 = np.expand_dims(gam12, axis=1)
            elif gam12.ndim > 2:
                raise ValueError("gam12 cannot have more than 2 dimensions")
            if gam12.shape[1] > gam12.shape[0]:
                gam12 = gam12.T
            if gam12.shape[0] != a_len or gam12.shape[1] != 1:
                raise ValueError("gam12 should be a vector of the same size as a.")
    else:
        if gam12 is not None:
            raise ValueError("one of the either xi1 or xi2 is not provided, but gam12 provided. It should be None")
    return gam12


def _format_inputs(eps: int | float,
                   a: int | float | list | np.ndarray | cp.Variable | cp.Expression = None,
                   b: int | float | cp.Variable | cp.Expression = None,
                   xi1_hat: int | float | list | np.ndarray | cp.Variable | cp.Expression = None,
                   xi2_hat: int | float = None,
                   gam11: int | float | list | np.ndarray = None,
                   gam12=None,
                   gam22: int | float = None):
    # check eps
    if not isinstance(eps, (int, float)) or eps <= 0 or eps > 0.5:
        raise ValueError("eps must be a number in (0, 0.5].")

    # checks to figure out what terms are included in the sum
    a_xi1_present, xi2_present, b_present = False, False, False

    # check b term
    b_present = _check_b_term(b, b_present)

    # check xi2 term
    gam22, xi2_present = _check_xi2_term(xi2_hat, gam22, xi2_present)

    # check a and xi1 terms
    a, xi1_hat, gam11, a_len, a_xi1_present = _check_a_xi1_term(a, xi1_hat, gam11, a_xi1_present)

    # check gam12
    gam12 = _check_gam12_term(gam12, a_len, a_xi1_present, xi2_present)

    return (a, b,
            xi1_hat, xi2_hat,
            gam11, gam12, gam22,
            a_xi1_present, xi2_present, b_present)


def _det_cclp(kappa_e: int | float,
              a: np.ndarray | cp.Variable | cp.Expression,
              b: int | float | cp.Variable | cp.Expression,
              xi1_hat: np.ndarray | cp.Variable | cp.Expression,
              xi2_hat: int | float,
              gam11: np.ndarray, gam12: np.ndarray, gam22: np.ndarray,
              a_xi1_present: bool, xi2_present: bool, b_present: bool,
              assume_sym=False, assume_psd=False):
    """
    returns the deterministic constraint from reformulating the CCLP.
    Form: kappa_e * sigma_x + psi_hat <= 0

    :param kappa_e: tightening term based on special cases
    :param a: a term
    :param b: b term
    :param xi1_hat: xi1 term mean
    :param xi2_hat: xi2 term mean
    :param gam11: xi1 term covar
    :param gam12: xi2 term var
    :param gam22: xi1 and xi2 cross covar
    :param a_xi1_present: indicates a and xi1 present
    :param xi2_present: indicates xi2 present
    :param b_present: indicates b present
    :param assume_sym: assume covar symmetric (else check)
    :param assume_psd: assume covar PSD (else check)
    :return:
    """

    # There are 5 cases:
    # 1. No expression (user error?)
    if not a_xi1_present and not xi2_present and not b_present:
        raise ValueError("No expression provided")

    # 2. only b term present -> automatically a deterministic constraint (if b is an expression or variable
    if b_present and not a_xi1_present and not xi2_present:
        if isinstance(b, (cp.Variable, cp.Expression)):
            return b <= 0
        else:
            raise ValueError("No constraint! Input is just a scalar with no decision variables.")

    if (not isinstance(a, (cp.Variable, cp.Expression)) and
            not isinstance(b, (cp.Variable, cp.Expression)) and
            not isinstance(xi1_hat, (cp.Variable, cp.Expression))):
        raise ValueError("Provided inputs do not include any decision variables. It does not result in an expression.")

    # 3. no xi1 term present (but xi2 present) -> special case
    if not a_xi1_present:
        psi_hat = xi2_hat + b
        sigma_x = np.sqrt(gam22[0, 0])
    # 4. no xi2 term present (but xi1 present) -> special case
    elif not xi2_present:
        psi_hat = a @ xi1_hat + b
        sigma_x = _cp_sqrt_quad_form(a, gam11,
                                     assume_sym=assume_sym,
                                     assume_psd=assume_psd)
    # 5. both xi1 and xi2 terms present -> general case
    else:
        a_til = cp.hstack([a, 1])
        xi_hat = cp.hstack([xi1_hat, xi2_hat])
        psi_hat = a_til @ xi_hat + b

        gamma = _construct_gamma(gam11, gam12, gam22)
        sigma_x = _cp_sqrt_quad_form(a_til, gamma, assume_sym=assume_sym, assume_psd=assume_psd)
    return kappa_e * sigma_x + psi_hat <= 0


def cclp_gauss(eps, a=None, b=None,
               xi1_hat=None, xi2_hat=None,
               gam11=None, gam12=None, gam22=None,
               assume_sym=False, assume_psd=False):
    """
    Reformulates a CC of the type Prob(a^T xi1 + b + xi2 <= 0) >= 1-eps.

    This function reformulates a chance constraint (CC) where the distribution is known to be a Gaussian distribution with:
    xi = [xi1^T xi2]^T
    Expect(xi) = [xi1_hat^T xi2_hat]^T
    Cov(xi) = [gam11       gam12
               gam12^T    gam22]

    :param eps: The epsilon value in the chance constraint.
    :type eps: float
    :param a: The coefficient vector a in the chance constraint.
    :type a: array_like, optional
    :param b: The constant term b in the chance constraint.
    :type b: float, optional
    :param xi1_hat: The expected value of xi1.
    :type xi1_hat: array_like, optional
    :param xi2_hat: The expected value of xi2.
    :type xi2_hat: array_like, optional
    :param gam11: The covariance matrix of xi1.
    :type gam11: array_like, optional
    :param gam12: The cross-covariance matrix between xi1 and xi2.
    :type gam12: array_like, optional
    :param gam22: The covariance matrix of xi2.
    :type gam22: array_like, optional
    :param assume_sym: Whether to assume the covariance matrix is symmetric.
    :type assume_sym: bool, optional
    :param assume_psd: Whether to assume the covariance matrix is positive semi-definite.
    :type assume_psd: bool, optional
    :return: The reformulated chance constraint.
    :rtype: cvxpy.Constraint or cp.Expression
    """
    # check inputs
    inputs = _format_inputs(eps, a, b,
                            xi1_hat, xi2_hat,
                            gam11, gam12, gam22)

    kappa_e = gauss.ppf(1 - eps)

    constr = _det_cclp(kappa_e, *inputs, assume_sym, assume_psd)

    return constr


def cclp_dro_mean_cov(eps, a=None, b=None,
                      xi1_hat=None, xi2_hat=None,
                      gam11=None, gam12=None, gam22=None,
                      assume_sym=False, assume_psd=False,
                      centrally_symmetric=False):
    """
    Reformulates a DRO CC of the type inf_{moment based set} Prob(a^T exp + b <= 0) >= bound.

    This function reformulates a distributionally robust optimization (DRO) chance constraint (CC) where the distribution is unknown but belongs to a moment based ambiguity set with known mean and covariance:
    d = [a^T b]^T
    Expect(d) = [a_hat^T b_hat]^T
    Cov(d) = [a_cov             ab_cross_cov
              ab_cross_cov^T    b_cross     ]

    :param eps: The epsilon value in the chance constraint.
    :type eps: float
    :param a: The coefficient vector a in the chance constraint.
    :type a: array_like, optional
    :param b: The constant term b in the chance constraint.
    :type b: float, optional
    :param xi1_hat: The expected value of xi1.
    :type xi1_hat: array_like, optional
    :param xi2_hat: The expected value of xi2.
    :type xi2_hat: array_like, optional
    :param gam11: The covariance matrix of xi1.
    :type gam11: array_like, optional
    :param gam12: The cross-covariance matrix between xi1 and xi2.
    :type gam12: array_like, optional
    :param gam22: The covariance matrix of xi2.
    :type gam22: array_like, optional
    :param assume_sym: Whether to assume the covariance matrix is symmetric.
    :type assume_sym: bool, optional
    :param assume_psd: Whether to assume the covariance matrix is positive semi-definite.
    :type assume_psd: bool, optional
    :param centrally_symmetric: Flag if distribution is centrally symmetric (Def 3.1).
    :type centrally_symmetric: bool, optional
    :return: The reformulated chance constraint.
    :rtype: cvxpy.Constraint or cvxpy.Expression
    """
    # check inputs
    inputs = _format_inputs(eps, a, b,
                            xi1_hat, xi2_hat,
                            gam11, gam12, gam22)

    if centrally_symmetric:
        kappa_e = np.sqrt(1 / (2 * eps))
    else:
        kappa_e = np.sqrt((1 - eps) / eps)

    constr = _det_cclp(kappa_e, *inputs, assume_sym, assume_psd)
    return constr
