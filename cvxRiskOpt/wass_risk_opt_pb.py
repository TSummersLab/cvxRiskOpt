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



CVXPY-based data-driven optimization problems that compute risk metrics.

We implement some results from:
"Data-driven distributionally robust optimization using the Wasserstein metric: 
    Performance guarantees and tractable reformulations"
"""
import cvxpy as cp
import numpy as np
from numbers import Number
from cvxpy.problems.problem import Problem as CVXPYProb
from cvxpy.utilities.deterministic import unique_list
from typing import Union


class WassWCEMaxAffine(CVXPYProb):
    """
    Wasserstein Worst Case Expectation Problem with Max of Affine Terms.

    This class implements a generic solution for Cor 5.1 (i), eq (15a) with affine loss functions:

    .. math::

        \\ell = \\max_k \\{ \\langle a_k,  \\xi \\rangle + b_k \\}

    where
        - xi (m x 1): the random variable for which we have samples
        - a_k (m x 1): a vector that is either a constant vector or a scalar times a decision variable (cvxpy affine expression)
        - b_k (1 x 1): a scalar or a 1-dimensional cvxpy affine expression
        - :math:`\\langle \\cdot, \\cdot \\rangle` is the inner product

    .. note::
        Currently, this is only implemented for the 1-, 2-, and inf-norm cases.

    Arguments:
    ----------
        num_samples: int:
            The number of samples.
        a_k_list: Number | array_like
            A list of vectors that are either constant vectors or scalars times a decision variable.
        b_k_list: array_like
            A list of scalars or 1-dimensional cvxpy affine expressions.
        support_C: np.ndarray, optional
            The support matrix C. If None passed, the support is considered to be the real space.
        support_d: np.ndarray, optional
            The support vector d. If None passed, the support is considered to be the real space.
        used_norm: int | np.inf, optional
            The norm to be used. Either 1, 2, or infinity (np.inf). Default is 2.
        vp_suffix: str, optional
            The suffix for the variable/parameter names.
    """
    instance_counter = 0

    def __init__(self, num_samples: int,
                 a_k_list: list, b_k_list: list = None,
                 support_C=None, support_d=None, used_norm=2,
                 vp_suffix: str = None) -> None:
        # choose the cp variable and parameter suffix
        WassWCEMaxAffine.instance_counter += 1
        if vp_suffix is None:
            vp_suffix = "_wwcema" + str(WassWCEMaxAffine.instance_counter)

        # number of samples
        self._num_samples = num_samples

        # # loss function terms
        # if a_k_list is None or a_k_list == []:
        #     raise ValueError('Problem statement error: a_k_list must be non-empty')
        self._a_k_list = a_k_list
        self._K = len(self._a_k_list)
        if isinstance(self._a_k_list[0], Number):  # a_k is just a number
            self._m = 0
        else:
            self._m = self._a_k_list[0].shape[0] if self._a_k_list[0].shape else 0
        # TODO: consider adding some logic to check if all values of a_k are similar in type/dimension
        #     raise ValueError('a_k_list error: all value must be of the same dimension')

        self._b_k_list = b_k_list if b_k_list is not None else []
        # if self._b_k_list:
        #     if isinstance(self._b_k_list[0], Number):  # b_k is just a number
        #         b_dim = 0
        #     else:
        #         b_dim = self._b_k_list[0].shape[0] if self._b_k_list[0].shape else 0
        # TODO: consider adding some logic to check if all values of a_k are similar in type/dimension
        #     raise ValueError('b_k_list error: all value must be of the same dimension')
        if self._b_k_list and len(self._b_k_list) != self._K:
            raise ValueError('Dimension missmatch: b_k_list must either be empty or of the same dimension as a_k_list.')
        # # TODO: add some type checking to make sure that a_k and b_k terms are cst or cvxpy affine expressions
        # #  and of the right dimensions

        # Random variable support: Xi = {xi | C xi <= d}
        self._C, self._d = support_C, support_d
        if support_C is not None and support_d is not None:
            self._use_gamma = True
            self._d_size = len(self._d)
        elif support_C is None and support_d is None:
            self._use_gamma = False
        else:
            raise ValueError('Random variable support error: Both C and d must be set.')

        # norm selection
        self._used_norm = used_norm
        if self._used_norm == 2:
            self._dual_norm = 2
        elif self._used_norm == 1:
            self._dual_norm = np.inf
        elif self._used_norm == np.inf:
            self._dual_norm = 1
        else:
            raise NotImplementedError('Chosen norm not supported.')

        # Optimization variables due to problem reformulation
        self._lam = cp.Variable(name='lam' + vp_suffix)  # dual problem lambda variable
        self._s = cp.Variable(self._num_samples, name='s' + vp_suffix)  # epigraph auxiliary variables
        if self._use_gamma:  # gamma terms s.t. gamma @ (d - C @ sample)
            self._gamma = [cp.Variable((self._K, self._d_size), name='gamma_' + str(i) + vp_suffix) for i in
                           range(self._num_samples)]

        # Parameters
        self._eps = cp.Parameter(name='eps' + vp_suffix)  # Wasserstain ball radius
        # Random variable samples
        if self._m > 0:
            self._samples = cp.Parameter([self._num_samples, self._m], name='samples' + vp_suffix)
        elif self._m == 0:
            self._samples = cp.Parameter([self._num_samples, ], name='samples' + vp_suffix)
        else:
            raise ValueError('m should be >= 0, but that is not the case.')

        self._obj = self._def_opt_pb_obj()
        self._constr = self._def_opt_pb_contr()

        super().__init__(self._obj, self._constr)

    def _def_opt_pb_obj(self):
        """
        defines the objective function
        :return:
        """
        # Optimization Objective
        return cp.Minimize(self._lam * self._eps + 1 / self._num_samples * cp.sum(self._s))

    def _def_opt_pb_contr(self):
        """
        defines the constraints
        :return:
        """
        constraints = []
        for i in range(self._num_samples):
            for k in range(self._K):
                # b_k + a_k @ xi_i + gamma_ik @ (d - C xi_i) <= s_i
                lhs1 = 0
                if self._b_k_list:
                    lhs1 += self._b_k_list[k]
                a_k = self._a_k_list[k]
                samp = self._samples[i]
                if self._m > 1:
                    lhs1 += a_k @ samp
                else:
                    lhs1 += a_k * samp
                if self._use_gamma:
                    if self._m > 1:
                        lhs1 += self._gamma[i][k, :] @ (self._d - self._C @ samp)
                    else:
                        lhs1 += self._gamma[i][k, :] @ (self._d - np.squeeze(self._C) * samp)
                constraints += [lhs1 <= self._s[i]]

                # dual_norm{C @ gamma_ik - a_k} <= lambda
                if self._use_gamma:
                    lhs2 = self._C.T @ self._gamma[i][k, :] - a_k
                else:
                    lhs2 = -a_k
                constraints += [cp.norm(lhs2, self._dual_norm) <= self._lam]

            # gamma_ik >= 0
            if self._use_gamma:
                constraints += [self._gamma[i] >= 0]
        return constraints

    # TODO: consider implementing some operators.


def update_wass_wce_params(prob, eps, samples) -> None:
    """
    Update the parameters associated with the WassWCEMaxAffine problem.

    Arguments:
    ----------
        prob: WassWCEMaxAffine:
            WassWCEMaxAffine problem
        eps: int:
            Wasserstein ball radius
        samples: np.ndarray:
            Data samples
    """
    for par in prob.param_dict.keys():
        if 'eps' in par:
            prob.param_dict[par].value = eps
        if 'samples' in par:
            prob.param_dict[par].value = samples
    return


class WassDRExpectation(WassWCEMaxAffine):
    """
    Provides a high-level implementation of the DR Expectation with a Wasserstein-based ambiguity set.

    Implements

    .. math::

        \\sup_{\\mathbb{P} \\in \\mathcal{P}} \\quad \\mathbb{E}^\\mathbb{P} \\left[ \\langle a, \\xi \\rangle + b \\right]

    where :math:`a, b` may contain decision variables and :math:`\\langle \\cdot, \\cdot \\rangle` is the inner product.


    Arguments:
    ----------
        num_samples: int:
            The number of samples.
        a: int | float | cp.Variable | cp.Parameter | cp.Expression:
            The "a" term in :math:`\\langle a, xi \\rangle + b`.
        b: int | float | cp.Variable | cp.Parameter | cp.Expression, optional"
            The "b" term in :math:`\\langle a, xi \\rangle + b`. If not assigned, 0 is used.
        support_C: np.ndarray, optional
            The support matrix C. If None passed, the support is considered to be the real space.
        support_d: np.ndarray, optional
            The support vector d. If None passed, the support is considered to be the real space.
        used_norm: int | np.inf, optional
            The norm to be used. Either 1, 2, or infinity (np.inf). Default is 2.
        vp_suffix: str, optional
            The suffix for the variable/parameter names.
    """
    instance_counter = 0

    def __init__(self,
                 num_samples: int,
                 a: int | float | cp.Variable | cp.Parameter | cp.Expression,
                 b: int | float | cp.Variable | cp.Parameter | cp.Expression = 0,
                 support_C=None, support_d=None, used_norm=2, vp_suffix=None):
        WassDRExpectation.instance_counter += 1
        if vp_suffix is None:
            vp_suffix = "_wdre" + str(WassDRExpectation.instance_counter)
        if b == 0:
            b = []
        a = [a]
        super().__init__(num_samples, a, b, support_C, support_d, used_norm, vp_suffix=vp_suffix)

    def _problems_match(self, other) -> bool:
        if self._used_norm != other._used_norm:
            raise NotImplementedError("Norms are different")
        elif self._num_samples != other._num_samples:
            raise NotImplementedError("Number of samples is different")
        elif self._C != other._C or self._d != other._d:
            raise NotImplementedError("Supports are different")
        return True

    # TODO: might be a good idea to check that the params used by both problems (xi and eps) are the same

    def __add__(self, other) -> Union["WassDRExpectation", "CVXPYProb", "WassDRCVaR"]:
        if other == 0:
            return self
        elif isinstance(other, WassDRExpectation):  # DR-E + DR-E --> just add a and b terms if problems match
            if self._problems_match(other):
                a = self._a_k_list[0] + other._a_k_list[0]
                b1 = self._b_k_list[0] if self._b_k_list else 0
                b2 = other._b_k_list[0] if other._b_k_list else 0
                b = b1 + b2
                return WassDRExpectation(self._num_samples, a, b, self._C, self._d, self._used_norm)
            else:
                raise NotImplementedError()
        elif isinstance(other, WassWCEMaxAffine):
            if self._problems_match(other):
                a_expect = self._a_k_list[0]
                b_expect = self._b_k_list[0] if self._b_k_list else 0

                a_list_new = [a_expect + a_k_other for a_k_other in other._a_k_list]
                if b_expect == 0 and (not other._b_k_list):  # no b terms in either problem
                    b_list_new = None
                elif b_expect == 0:  # no b in DR-E. b term in other only
                    b_list_new = other._b_k_list
                elif not other._b_k_list:  # b in DR-E only. no b term in other
                    b_list_new = [b_expect for _ in other._a_k_list]
                else:  # b terms in both
                    b_list_new = [b_expect + b_other for b_other in other._b_k_list]
                if isinstance(other, WassDRCVaR):
                    return WassDRCVaR(other._num_samples, other._m, a=None,
                                      a_k_list=a_list_new, b_k_list=b_list_new, alpha=other._alpha,
                                      support_C=other._C, support_d=other._d, used_norm=other._used_norm)
                else:
                    return WassWCEMaxAffine(num_samples=other._num_samples, a_k_list=a_list_new, b_k_list=b_list_new,
                                            support_C=other._C, support_d=other._d, used_norm=other._used_norm)
            else:
                raise NotImplementedError()
        elif not isinstance(other, CVXPYProb):
            raise NotImplementedError()
        else:
            return CVXPYProb(self.objective + other.objective, unique_list(self.constraints + other.constraints))

    def __sub__(self, other) -> Union["WassDRExpectation", "CVXPYProb"]:
        if isinstance(other, WassDRExpectation):
            if self._problems_match(other):
                a = self._a_k_list[0] - other._a_k_list[0]
                b1 = self._b_k_list[0] if self._b_k_list else 0
                b2 = other._b_k_list[0] if other._b_k_list else 0
                b = b1 - b2
                return WassDRExpectation(self._num_samples, a, b, self._C, self._d, self._used_norm)
            else:
                raise NotImplementedError()
        elif not isinstance(other, CVXPYProb):
            raise NotImplementedError()
        else:
            CVXPYProb(self.objective - other.objective, unique_list(self.constraints + other.constraints))

    def __mul__(self, other) -> "WassDRExpectation":
        if not isinstance(other, (int, float)):
            raise NotImplementedError()
        a = self._a_k_list[0] * other
        b = self._b_k_list[0] * other if self._b_k_list else 0
        return WassDRExpectation(self._num_samples, a, b, self._C, self._d, self._used_norm)

    __rmul__ = __mul__

    def __div__(self, other) -> "WassDRExpectation":
        if not isinstance(other, (int, float)):
            raise NotImplementedError()
        if np.abs(other) < 1e-8:  # basically division by zero
            raise ZeroDivisionError()
        a = self._a_k_list[0] * (1.0 / other)
        b = self._b_k_list[0] * (1.0 / other) if self._b_k_list else 0
        return WassDRExpectation(self._num_samples, a, b, self._C, self._d, self._used_norm)

    __truediv__ = __div__

    # TODO: consider implementing the following
    #   __neg__, __rsub__, __radd__


class WassDRCVaR(WassWCEMaxAffine):
    """
    Provides a high-level implementation of the DR-CVaR with a Wasserstein-based ambiguity set.

    Implements

    .. math ::

        \\sup_{\\mathbb{P} \\in \\mathcal{P}} \\quad  \\text{CVaR}_{\\alpha}^\\mathbb{P}\\left[ \\langle a, \\xi \\rangle + b \\right]

    where :math:`a, b` may contain decision variables and :math:`\\langle \\cdot, \\cdot \\rangle` is the inner product.
    :math:`\\alpha` is the CVaR level (average in alpha * 100% of the worst/highest cases)

    Alternatively, this can also implement the CVaR reformulated form as a max of two affine terms:

     .. math ::

        \\sup_{\\mathbb{P} \\in \\mathcal{P}} \\quad  \\mathbb{E}^{\\mathbb{P}} \\max_{k \\in \\{1, 2\\}} \\left[ \\langle a_k, \\xi \\rangle + b_k \\right].


    .. Note ::

        Either "a" and "b" must be passed or "a_k_list" and "b_k_list", but not both nor neither.


    Arguments:
    ----------
        num_samples: int:
            The number of samples.
        xi_length: int:
            Size of a sample. If :math:`\\xi` is m x 1 --> xi_length = m
        a: int | float | cp.Variable | cp.Parameter | cp.Expression, optional:
            The "a" term in :math:`\\langle a, xi \\rangle + b`.
            If this is passed, the "b" term is expected.
            This should not be passed if "a_k_list" and "b_k_list" are passed.
        b: int | float | cp.Variable | cp.Parameter | cp.Expression, optional, optional:
            The "b" term in :math:`\\langle a, xi \\rangle + b`. If not assigned, 0 is used.
            This should not be passed if "a_k_list" and "b_k_list" are passed.
        a_k_list: Number | array_like, optional:
            A list of vectors that are either constant vectors or scalars times a decision variable.
            If this is passed, the "b_k_list" term is expected.
            This should not be passed if "a" and "b" are passed.
        b_k_list: array_like, optional:
            A list of scalars or 1-dimensional cvxpy affine expressions.
            This should not be passed if "a" and "b" are passed.
        alpha: float, optional:
            CVaR level (average in alpha * 100% of the worst/highest cases)
        support_C: np.ndarray, optional
            The support matrix C. If None passed, the support is considered to be the real space.
        support_d: np.ndarray, optional
            The support vector d. If None passed, the support is considered to be the real space.
        used_norm: int | np.inf, optional
            The norm to be used. Either 1, 2, or infinity (np.inf). Default is 2.
        vp_suffix: str, optional
            The suffix for the variable/parameter names.

    """
    instance_counter = 0

    def __init__(self,
                 num_samples: int, xi_length: int,
                 a: int | float | cp.Variable | cp.Parameter | cp.Expression = None,
                 b: int | float | cp.Variable | cp.Parameter | cp.Expression = 0,
                 a_k_list=None,
                 b_k_list=None,
                 alpha=0.1,
                 support_C=None, support_d=None,
                 used_norm=2, vp_suffix=None):

        WassDRCVaR.instance_counter += 1
        if vp_suffix is None:
            vp_suffix = "_wdrcvar" + str(WassDRCVaR.instance_counter)

        self._alpha = alpha

        if a is None and (a_k_list is None or b_k_list is None):
            raise ValueError(
                'Option 1: provide a (optionally b). Option 2: provided a_k_list and b_k_list. Both failed.')
        if a is None:
            super().__init__(num_samples, a_k_list, b_k_list, support_C, support_d, used_norm, vp_suffix=vp_suffix)
        else:
            tau = cp.Variable(1, name='tau' + vp_suffix)
            a1 = np.zeros(xi_length)
            a2 = a / alpha
            b1 = tau
            b2 = b / alpha + (1 - 1 / alpha) * tau
            a_k_list = [a1, a2]
            b_k_list = [b1, b2]
            super().__init__(num_samples, a_k_list, b_k_list, support_C, support_d, used_norm, vp_suffix=vp_suffix)

    def _problems_match(self, other) -> bool:
        if self._used_norm != other._used_norm:
            raise NotImplementedError("Norms are different")
        elif self._num_samples != other._num_samples:
            raise NotImplementedError("Number of samples is different")
        elif self._C != other._C or self._d != other._d:
            raise NotImplementedError("Supports are different")
        elif isinstance(other, WassDRCVaR) and self._C != other._alpha or self._d != other._alpha:
            raise NotImplementedError("CVaR alpha levels are different")
        return True
        # TODO: might be a good idea to check that the params used by both problems (xi and eps) are the same

    def __add__(self, other) -> Union["WassDRExpectation", "CVXPYProb", "WassDRCVaR"]:
        if other == 0:
            return self
        elif isinstance(other, WassDRExpectation):  # DR-CVaR + DR-E special case --> call __add__ in DR-E
            return other + self
        elif not isinstance(other, CVXPYProb):
            raise NotImplementedError()
        else:
            return CVXPYProb(self.objective + other.objective, unique_list(self.constraints + other.constraints))

    def __mul__(self, other) -> "WassDRCVaR":
        if not isinstance(other, (int, float)):
            raise NotImplementedError()
        if other <= 0:
            raise NotImplementedError('Multiplication is only supported with positive scalars')
        a_list_new = [a_k * other for a_k in self._a_k_list]
        b_list_new = [b_k * other for b_k in self._b_k_list]
        return WassDRCVaR(self._num_samples, self._m, a=None,
                          a_k_list=a_list_new, b_k_list=b_list_new,
                          alpha=self._alpha, support_C=self._C,
                          support_d=self._d, used_norm=self._used_norm)

    __rmul__ = __mul__

    def __div__(self, other) -> "WassDRCVaR":
        if not isinstance(other, (int, float)):
            raise NotImplementedError()
        if np.abs(other) < 1e-8:  # basically division by zero
            raise ZeroDivisionError()
        return (1.0 / other) * self

    __truediv__ = __div__

    # TODO: consider implementing the rest of the operators.
