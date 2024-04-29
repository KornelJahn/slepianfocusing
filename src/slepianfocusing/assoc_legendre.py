# assoc_legendre.py

"""Implementation of the normalized associated Legendre functions (NALFs)
P_{lm}"""

__author__ = "Kornel JAHN (kornel.jahn@gmail.com)"
__copyright__ = "Copyright (c) 2012-2024 Kornel Jahn"
__license__ = "MIT"

import numpy as np

try:
    from numba import jit
except ImportError:
    from .numba_dummy import jit

from .math_ext import minus_one_to_pow


SQRT3_OVER_2 = 0.5 * np.sqrt(3.0)
EPS = np.finfo(np.float64).eps


@jit("i8(i8)", nopython=True, cache=True)
def nalf_lowest_degree(m):
    """Return the lowest allowed degree for a given order m."""
    return abs(m)


@jit("i8(i8, i8)", nopython=True, cache=True)
def nalf_function_count(lmax, m):
    """
    Return the number of associated Legendre functions for a given maximal
    degree lmax and an order m.
    """
    lmin = nalf_lowest_degree(m)
    return lmax - lmin + 1


@jit(["f8(i8, i8)", "f8[:](i8[:], i8)"], nopython=True, cache=True)
def nalf_recurr_factor_nu(l, m):
    """
    Calculate the factor appearing in the three-term recurrence relation of the
    normalized associated Legendre functions P_{lm}:

        x P_{lm}(x) = nu_{lm} P_{l-1,m}(x) + nu_{l+1, m} P_{l+1,m}(x).
    """
    l2, m2 = l * l, m * m
    return np.sqrt((l2 - m2) / (4 * l2 - 1.0))


@jit(["f8(i8, i8, i8)", "f8[:](i8[:], i8, i8)"], nopython=True, cache=True)
def nalf_recurr_factor_gamma(l, m, sign):
    """
    Calculate the factor appearing in the following recurrence relation of the
    normalized associated Legendre functions P_{lm}:

        m P_{lm}(x) / sqrt(1 - x^2) = gamma^+_{lm} P_{l+1,m+1}(x) +
                                      gamma^-_{lm} P_{l+1,m-1}(x)
    """
    return 0.5 * np.sqrt(
        (l + sign * m + 1) * (l + sign * m + 2) * (2 * l + 1) / (2.0 * l + 3.0)
    )


SCALE_FACTOR = 1.0e280
INV_SCALE_FACTOR = 1.0 / SCALE_FACTOR


@jit("f8(i8, f8, f8)", nopython=True, cache=True)
def _nalf_P_first_scaled(m, cos_theta, sin_theta):
    """
    Calculate the upscaled value of the normalized associated Legendre
    function P_{lm}(cos(theta)) for l=lmin where lmin is the lowest allowed
    degree for the order m.

    NOTE #1: For large m and small theta, P_{lmin,m}(cos_theta) may underflow
    because of the factor [sin(theta)]^|m|, which then affects
    all subsequent values for higher degrees obtained using a recurrence
    relation! This underflow is delayed by using an internal scaling [1].

    NOTE #2: working simulatneously with precomputed values of cos(theta) and
    sin(theta) helps us to avoid cancellation associated with calculating
    sin(theta) from cos(theta) as sin(theta) = sqrt(1 - cos(theta)^2).

    References:

    [1] Holmes, S. A., & Featherstone, W. E. (2002). A unified approach to the
        Clenshaw summation and the recursive computation of very high degree
        and order normalised associated Legendre functions. Journal of Geodesy,
        76(5), 279-299. http://dx.doi.org/10.1007/s00190-002-0216-2
    """

    abs_m = abs(m)
    result = SCALE_FACTOR
    product = 1.0
    for j in range(1, abs_m + 1):
        # Calculate [sin(theta)]^|m| and (2|m| - 1)!! simultaneously in one
        # fused loop
        two_j = 2 * j
        product *= (two_j - 1.0) / two_j
        result *= sin_theta
    result *= np.sqrt(0.5 * (2 * abs_m + 1) * product)
    if m > 0:
        result *= minus_one_to_pow(abs_m)
    return result


@jit("f8[:](i8, i8, f8, f8)", nopython=True, cache=True)
def nalf_P(lmax, m, cos_theta, sin_theta):
    """
    Return values of P_{l,m}(cos(theta)) from l=lmin up to l=lmax.
    """
    assert lmax >= 0
    assert abs(m) <= lmax

    lmin = nalf_lowest_degree(m)
    result = np.zeros((lmax + 1,), dtype=np.float64)

    P_lmin_scaled = _nalf_P_first_scaled(m, cos_theta, sin_theta)
    P_l = INV_SCALE_FACTOR
    result[lmin] = P_l * P_lmin_scaled
    P_lm1 = 0.0
    nu_l = 0.0

    # Calculate values corresponding to a higher degree l using the 3-term
    # recurrence relation involving P_{l+1,m}(cos(theta)), P_{lm}(cos(theta)),
    # and P_{l-1,m}(cos(theta)).
    for l in range(lmin, lmax):
        nu_lp1 = nalf_recurr_factor_nu(l + 1, m)
        P_lp1 = cos_theta * P_l - nu_l * P_lm1
        P_lp1 /= nu_lp1
        result[l + 1] = P_lp1 * P_lmin_scaled
        P_lm1 = P_l
        P_l = P_lp1
        nu_l = nu_lp1
    return result


@jit(nopython=True, cache=True)
def nalf_P_vectorized(lmax, m, cos_theta, sin_theta):
    """
    Vectorized version of nalf_P that operates on multiple values of cos(theta)
    and sin(theta).
    """
    assert cos_theta.shape == sin_theta.shape
    lmin = nalf_lowest_degree(m)
    result = np.zeros((lmax + 1, cos_theta.size), dtype=np.float64)
    for i in range(cos_theta.size):
        result[:, i] = nalf_P(lmax, m, cos_theta.flat[i], sin_theta.flat[i])
    return result.reshape((lmax + 1,) + cos_theta.shape)
