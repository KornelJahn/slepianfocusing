# sheppard_torok.py

"""Implementation of the Sheppard--Torok functions F_{lm}"""

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
def stf_lowest_degree(m):
    """Return the lowest allowed degree for a given order m."""
    return max(1, abs(m))


@jit("i8(i8, i8)", nopython=True, cache=True)
def stf_function_count(lmax, m):
    """Return the number of Sheppard--Torok functions for a given maximal
    degree lmax and an order m.
    """
    lmin = stf_lowest_degree(m)
    return lmax - lmin + 1


@jit(["f8(i8, i8)", "f8[:](i8[:], i8)"], nopython=True, cache=True)
def stf_recurr_factor_zeta(l, m):
    """Calculate the factor appearing in the three-term recurrence relation of
    the Sheppard--Torok functions F_{lm}.

    Expressing the recurrence relation as

        {x - m / [l(l + 1)]} F_{lm}(x) = zeta_{lm} F_{l-1,m}(x) +
                                         zeta_{l+1, m} F_{l+1,m}(x),

    the factor to be calculated is zeta_{lm}.
    """
    l2, m2 = l * l, m * m
    return np.sqrt((l2 - m2) * (l2 - 1) / (4 * l2 - 1.0)) / l


SCALE_FACTOR = 1.0e280
INV_SCALE_FACTOR = 1.0 / SCALE_FACTOR


@jit("f8(i8, f8, f8)", nopython=True, cache=True)
def _stf_F_first_scaled(m, cos_theta, sin_theta):
    """Calculate the upscaled value of the Sheppard--Torok function
    F_{lm}(cos(theta)) for l=lmin where lmin is the lowest allowed degree for
    the order m.

    NOTE #1: For large m and small theta, F_{lmin,m}(cos_theta) may underflow
    because of the factor [sin(theta)]^(|m| - 1), which then affects
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
    result = SCALE_FACTOR
    if m == 0:
        result *= SQRT3_OVER_2 * sin_theta
        return result

    # Calculate first the portion of F_{lmin,m}(cos(theta)) which depends on |m|
    # only
    abs_m = abs(m)
    product = 0.5
    for j in range(2, abs_m + 1):
        # Calculate [sin(theta)]^(|m| - 1) and (2|m| - 1)!! simultaneously in
        # one fused loop
        two_j = 2 * j
        product *= (two_j - 1.0) / two_j
        result *= sin_theta

    two_abs_m_p_1 = 2 * abs_m + 1
    result *= np.sqrt(two_abs_m_p_1 * abs_m / (two_abs_m_p_1 + 1.0) * product)

    # Multiply the result with the factor that depends on the sign of m, too.
    # We replace 1 - |cos(theta)| by sin^2(theta) / ( 1.0 + |cos(theta)| ) which
    # does not exhibit cancellation (see commented lines below).
    one_p_abs_cos_theta = 1.0 + np.abs(cos_theta)
    one_m_abs_cos_theta = sin_theta**2 / one_p_abs_cos_theta
    if cos_theta * m >= 0.0:
        result *= one_p_abs_cos_theta
    else:
        result *= one_m_abs_cos_theta
    if m > 0:
        result *= minus_one_to_pow(abs_m + 1)
    return result


@jit("f8[:](i8, i8, f8, f8)", nopython=True, cache=True)
def stf_F(lmax, m, cos_theta, sin_theta):
    """Fill the array result with values of F_{l,m}(cos(theta)) from l=lmin up
    to l=lmax.
    """
    assert lmax >= 1
    assert abs(m) <= lmax

    lmin = stf_lowest_degree(m)
    result = np.zeros((lmax + 1,), dtype=np.float64)
    F_lmin_scaled = _stf_F_first_scaled(m, cos_theta, sin_theta)
    F_l = INV_SCALE_FACTOR
    result[lmin] = F_l * F_lmin_scaled
    F_lm1 = 0.0
    zeta_l = 0.0

    # Calculate values corresponding to a higher degree l using the 3-term
    # recurrence relation involving F_{l+1,m}(cos(theta)), F_{l,m}(cos(theta)),
    # and F_{l-1,m}(cos(theta)).
    for l in range(lmin, lmax):
        zeta_lp1 = stf_recurr_factor_zeta(l + 1, m)
        F_lp1 = (cos_theta - m / (l * (l + 1.0))) * F_l - zeta_l * F_lm1
        F_lp1 /= zeta_lp1
        result[l + 1] = F_lp1 * F_lmin_scaled
        F_lm1 = F_l
        F_l = F_lp1
        zeta_l = zeta_lp1
    return result


@jit(nopython=True, cache=True)
def stf_F_vectorized(lmax, m, cos_theta, sin_theta):
    """Vectorized version of stf_F that operates on multiple values of
    cos(theta) and sin(theta).
    """
    assert cos_theta.shape == sin_theta.shape
    lmin = stf_lowest_degree(m)
    result = np.empty((lmax + 1, cos_theta.size), dtype=np.float64)
    for i in range(cos_theta.size):
        result[:, i] = stf_F(lmax, m, cos_theta.flat[i], sin_theta.flat[i])
    return result.reshape((lmax + 1,) + cos_theta.shape)


# @jit(nopython=True, cache=True)
# def sum_stf_F(coeffs, m, cos_theta, sin_theta):
#    lmin = stf_lowest_degree(m)
#    lmax = lmin + len(coeffs) - 1
#
#    # Clenshaw's summation formula, downward iteration
#    y_lp2 = 0.0
#    y_lp1 = 0.0
#    zeta_lp2 = 1.0
#    zeta_lp1 = stf_recurr_factor_zeta(lmax + 1, m)
#    for i in range(len(coeffs) - 1, 0, -1):
#        l = lmin + i
#        yl = ((cos_theta - m / (l * (l + 1))) / zeta_lp1 * y_lp1 -
#              zeta_lp1 / zeta_lp2 * y_lp2 + coeffs[i])
#        zeta_lp2 = zeta_lp1
#        zeta_lp1 = stf_recurr_factor_zeta(l, m)
#        y_lp2 = y_lp1
#        y_lp1 = yl#
#
#    zeta_lminp2 = zeta_lp2
#    zeta_lminp1 = stf_recurr_factor_zeta(lmin + 1, m)
#    F_1st = _stf_F_first_scaled(m, cos_theta, sin_theta) * INV_SCALE_FACTOR
#    F_2nd = (cos_theta - m / (lmin * (lmin + 1.0))) * F_1st / zeta_lminp1
#    result = ((-zeta_lminp1 / zeta_lminp2 * y_lp2 + coeffs[0]) * F_1st +
#              y_lp1 * F_2nd)
#    # TODO: make function robust for cancellation: detect whether the previous
#    # line involves the addition of terms which have nearly equal magnitude but
#    # opposite sign. In such a case, restart with an upward recursion.
#    # See:
#    #    Press et al. Numerical Recipes (3rd edition), p. 223
#    return result
