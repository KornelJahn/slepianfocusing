# vector_multipole_fields.py

"""Implementation of vector multipole fields M_{lm} and N_{lm} as well as mixed
vector multiple fields T^{+/-}_{lm}."""

__author__ = "Kornel JAHN (kornel.jahn@gmail.com)"
__copyright__ = "Copyright (c) 2012-2024 Kornel Jahn"
__license__ = "MIT"

import numpy as np
import math

from .vector_spherical_harmonics import vsh_series
from .sph_bessel import sph_bessel_j
from .math_ext import minus_one_to_pow, sign

try:
    from numba import jit
except ImportError:
    from .numba_dummy import jit

ONE_OVER_SQRT2 = 1.0 / math.sqrt(2.0)


@jit("c16[:](i8, i8, c16[:], f8, f8, f8, f8, f8)", nopython=True, cache=True)
def vmf_series(
    pol, m, coeffs, rlambda, cos_theta, sin_theta, cos_phi, sin_phi
):
    """Evaluate a finite series of mixed vector multipole fields T^{+/-}_{lm}
    for a fixed value of m.
    """
    pol = sign(pol)
    L = len(coeffs) - 1
    kr = 2.0 * np.pi * rlambda
    minus_one_to_m = minus_one_to_pow(m)
    minus_one_to_mp1 = -minus_one_to_m

    # Precompute values of spherical Bessel functions of the first kind
    j = sph_bessel_j(L, kr)
    # Compute jinc(l, kr) = j(l, kr) / kr
    if kr != 0.0:
        jinc = j / kr
    else:
        jinc = np.zeros((L + 1,), dtype=np.float64)
        jinc[0] = 1.0
        jinc[1] = 1.0 / 3.0

    l = np.arange(0, L + 1)
    T_rad = 1j * np.sqrt(2 * l * (l + 1)) * jinc
    T_plus = np.zeros((L + 1,), dtype=np.complex128)
    T_plus[1:] = j[1:] - 1j * (j[:-1] - l[1:] * jinc[1:])
    T_minus = np.conj(T_plus)
    i_to_pow_l = 1j**l

    if pol == 1:
        coeffs_rad = i_to_pow_l * coeffs * T_rad
        coeffs_plus = i_to_pow_l * coeffs * T_plus
        coeffs_minus = i_to_pow_l * minus_one_to_mp1 * coeffs * T_minus
    elif pol == -1:
        coeffs_rad = i_to_pow_l * minus_one_to_m * coeffs * T_rad
        coeffs_plus = i_to_pow_l * minus_one_to_mp1 * coeffs * T_minus
        coeffs_minus = i_to_pow_l * coeffs * T_plus
    else:
        raise ValueError("pol = 0 is not allowed for vector multipole fields.")

    result = vsh_series(
        0, m, coeffs_rad, cos_theta, sin_theta, cos_phi, sin_phi
    )
    result += vsh_series(
        1, m, coeffs_plus, cos_theta, sin_theta, cos_phi, sin_phi
    )
    result += vsh_series(
        -1, m, coeffs_minus, cos_theta, sin_theta, cos_phi, sin_phi
    )
    result *= 2.0 * np.pi
    return result


@jit(nopython=True, cache=True)
def vmf_series_vectorized(
    pol, m, coeffs, rlambda, cos_theta, sin_theta, cos_phi, sin_phi
):
    """Vectorized version of `vmf_series` that operates on multiple values of
    r/lambda, cos(theta), and sin(theta).
    """
    assert rlambda.shape == cos_theta.shape
    assert rlambda.shape == sin_theta.shape
    assert rlambda.shape == cos_phi.shape
    assert rlambda.shape == sin_phi.shape
    result = np.zeros((3, rlambda.size), dtype=np.complex128)
    for i in range(rlambda.size):
        result[:, i] = vmf_series(
            pol,
            m,
            coeffs,
            rlambda.flat[i],
            cos_theta.flat[i],
            sin_theta.flat[i],
            cos_phi.flat[i],
            sin_phi.flat[i],
        )
    return result.reshape((3,) + rlambda.shape)


def vmf(pol, l, m, *coords):
    coeffs = np.zeros((l + 1,), dtype=np.complex128)
    coeffs[-1] = 1.0
    return vmf_series(pol, m, coeffs, *coords)


def vmf_vectorized(pol, l, m, *coords):
    coeffs = np.zeros((l + 1,), dtype=np.complex128)
    coeffs[-1] = 1.0
    return vsh_series_vectorized(pol, m, coeffs, *coords)


def vmf_M(l, m, *coords):
    Tp = vmf(1, l, m, *coords)
    Tm = vmf(-1, l, m, *coords)
    return ONE_OVER_SQRT2 * (Tp + minus_one_to_pow(m + 1) * Tm)


def vmf_M_vectorized(l, m, *coords):
    Tp = vmf_vectorized(1, l, m, *coords)
    Tm = vmf_vectorized(-1, l, m, *coords)
    return ONE_OVER_SQRT2 * (Tp + minus_one_to_pow(m + 1) * Tm)


def vmf_N(l, m, *coords):
    Tp = vmf(1, l, m, *coords)
    Tm = vmf(-1, l, m, *coords)
    return -1j * ONE_OVER_SQRT2 * (Tp - minus_one_to_pow(m + 1) * Tm)


def vmf_N_vectorized(l, m, *coords):
    Tp = vmf_vectorized(1, l, m, *coords)
    Tm = vmf_vectorized(-1, l, m, *coords)
    return -1j * ONE_OVER_SQRT2 * (Tp - minus_one_to_pow(m + 1) * Tm)
