# vector_spherical_harmonics.py

"""Implementation of vector spherical harmonics Y_{lm} and Z_{lm} as well as
the mixed vector spherical harmonics Q^{+/-}_{lm}.
"""

__author__ = "Kornel JAHN (kornel.jahn@gmail.com)"
__copyright__ = "Copyright (c) 2012-2024 Kornel Jahn"
__license__ = "MIT"

import numpy as np
import math

from .assoc_legendre import (
    nalf_P,
    nalf_P_vectorized,
    nalf_function_count,
    nalf_lowest_degree,
)
from .sheppard_torok import (
    stf_F,
    stf_F_vectorized,
    stf_function_count,
    stf_lowest_degree,
)
from .unit_vectors import (
    radial_uv,
    e_plus_uv,
    e_plus_uv_vectorized,
    e_minus_uv,
    e_minus_uv_vectorized,
)
from .quadrature import GaussLegendreRule
from .math_ext import (
    minus_one_to_pow,
    next_power_of_2,
    log2_next_power_of_2,
    sign,
)

try:
    from numba import jit
except ImportError:
    from .numba_dummy import jit

ONE_OVER_SQRT2 = 1.0 / math.sqrt(2.0)
ONE_OVER_SQRT_2PI = 1.0 / math.sqrt(2.0 * np.pi)
EPS = np.finfo(np.float64).eps


@jit("c16[:](i8, i8, c16[:], f8, f8, f8, f8)", nopython=True, cache=False)
def vsh_series(pol, m, coeffs, cos_theta, sin_theta, cos_phi, sin_phi):
    """Evaluate a finite series of Mixed Vector Spherical Harmonics
    Q^{+/-}_{lm} for a fixed value of m.
    """
    pol = sign(pol)
    L = len(coeffs) - 1
    if pol == 0:
        basis_funcs = nalf_P(L, m, cos_theta, sin_theta)
    else:
        basis_funcs = stf_F(L, pol * m, cos_theta, sin_theta)
    scalar_part = np.dot(
        np.ascontiguousarray(coeffs),
        basis_funcs.astype(np.complex128)
    )
    if m != 0:
        scalar_part *= (cos_phi + 1j * sign(m) * sin_phi) ** abs(
            m
        )  # exp(i m phi)
    if pol == -1:
        polarization = e_minus_uv(cos_theta, sin_theta, cos_phi, sin_phi)
    elif pol == 0:
        polarization = radial_uv(cos_theta, sin_theta, cos_phi, sin_phi)
    else:
        polarization = e_plus_uv(cos_theta, sin_theta, cos_phi, sin_phi)
    return ONE_OVER_SQRT_2PI * scalar_part * polarization


@jit(nopython=True, cache=False)
def vsh_series_vectorized(
    pol, m, coeffs, cos_theta, sin_theta, cos_phi, sin_phi
):
    """Vectorized version of `vsh_series` that operates on multiple values of
    cos(theta) and sin(theta).
    """
    assert cos_theta.shape == sin_theta.shape
    assert cos_theta.shape == cos_phi.shape
    assert cos_theta.shape == sin_phi.shape
    result = np.zeros((3, cos_theta.size), dtype=np.complex128)
    for i in range(cos_theta.size):
        result[:, i] = vsh_series(
            pol,
            m,
            coeffs,
            cos_theta.flat[i],
            sin_theta.flat[i],
            cos_phi.flat[i],
            sin_phi.flat[i],
        )
    return result.reshape((3,) + cos_theta.shape)


def vsh(pol, l, m, *angular_coords):
    coeffs = np.zeros((l + 1,), dtype=np.complex128)
    coeffs[-1] = 1.0
    return vsh_series(pol, m, coeffs, *angular_coords)


def vsh_vectorized(pol, l, m, *angular_coords):
    coeffs = np.zeros((l + 1,), dtype=np.complex128)
    coeffs[-1] = 1.0
    return vsh_series_vectorized(pol, m, coeffs, *angular_coords)


def vsh_Y(l, m, *angular_coords):
    Qp = vsh(1, l, m, *angular_coords)
    Qm = vsh(-1, l, m, *angular_coords)
    return ONE_OVER_SQRT2 * (Qp + minus_one_to_pow(m + 1) * Qm)


def vsh_Y_vectorized(l, m, *angular_coords):
    Qp = vsh_vectorized(1, l, m, *angular_coords)
    Qm = vsh_vectorized(-1, l, m, *angular_coords)
    return ONE_OVER_SQRT2 * (Qp + minus_one_to_pow(m + 1) * Qm)


def vsh_Z(l, m, *angular_coords):
    Qp = vsh(1, l, m, *angular_coords)
    Qm = vsh(-1, l, m, *angular_coords)
    return -1j * ONE_OVER_SQRT2 * (Qp - minus_one_to_pow(m + 1) * Qm)


def vsh_Z_vectorized(l, m, *angular_coords):
    Qp = vsh_vectorized(1, l, m, *angular_coords)
    Qm = vsh_vectorized(-1, l, m, *angular_coords)
    return -1j * ONE_OVER_SQRT2 * (Qp - minus_one_to_pow(m + 1) * Qm)


def vsh_coeffs_Qpm_to_YZ(m, coeffs_Qp, coeffs_Qm):
    assert len(coeffs_Qp) == len(coeffs_Qm)
    minus_one_to_pow_mp1 = minus_one_to_pow(m + 1)
    coeffs_Y = coeffs_Qp + minus_one_to_pow_mp1 * coeffs_Qm
    coeffs_Z = 1j * (coeffs_Qp - minus_one_to_pow_mp1 * coeffs_Qm)
    coeffs_Y *= ONE_OVER_SQRT2
    coeffs_Z *= ONE_OVER_SQRT2
    return coeffs_Y, coeffs_Z


def vsh_coeffs_YZ_to_Qpm(m, coeffs_Y, coeffs_Z):
    assert len(coeffs_Y) == len(coeffs_Z)
    coeffs_Qp = coeffs_Y - 1j * coeffs_Z
    coeffs_Qm = minus_one_to_pow(m + 1) * (coeffs_Y + 1j * coeffs_Z)
    coeffs_Qp *= ONE_OVER_SQRT2
    coeffs_Qm *= ONE_OVER_SQRT2
    return coeffs_Qp, coeffs_Qm


def vsh_Q_coeffs_cap(
    vector_func,
    Theta,
    pol,
    L,
    M,
    rtol=1.0e2 * EPS,
    atol=1.0e2 * EPS,
    purify=True,
):
    assert Theta > 0.0 and Theta < np.pi / 2
    assert L >= 1
    assert M >= 0 and M <= L

    pol = sign(pol)
    # Choose Kphi so that it is some power of 2 where Kphi is the number of
    # azimuthal sampling points
    Kphi = next_power_of_2(4 * M) if M > 0 else 1

    # Choose theta_level, which influences the number of internal quadrature
    # nodes in the theta direction
    theta_level = max(5, log2_next_power_of_2(2 * L))
    theta_level = min(theta_level, 9)

    # Construct sampling grid
    rule = GaussLegendreRule(theta_level).get_nodes_weights(0.0, Theta)
    theta, wtheta, wtheta_lower_prec = rule
    Ktheta = len(theta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    phi = 2.0 * np.pi / Kphi * np.arange(0, Kphi)

    # Construct 2d arrays of shape (Ktheta, Kphi)
    cos_theta_2d = np.tile(cos_theta[:, np.newaxis], (1, Kphi))
    sin_theta_2d = np.tile(sin_theta[:, np.newaxis], (1, Kphi))
    cos_phi_2d = np.tile(np.cos(phi), (Ktheta, 1))
    sin_phi_2d = np.tile(np.sin(phi), (Ktheta, 1))

    # Sample the function
    if pol == 1:
        e_uv = e_plus_uv_vectorized(
            cos_theta_2d, sin_theta_2d, cos_phi_2d, sin_phi_2d
        )
    elif pol == -1:
        e_uv = e_minus_uv_vectorized(
            cos_theta_2d, sin_theta_2d, cos_phi_2d, sin_phi_2d
        )
    else:
        raise ValueError

    values = vector_func(cos_theta_2d, sin_theta_2d, cos_phi_2d, sin_phi_2d)
    values *= sin_theta_2d  # Jacobian

    # Calculate the projections onto the tau_minus and tau_plus unit vectors
    values = (e_uv.conj() * values).sum(0)
    assert values.shape == (Ktheta, Kphi)

    m_range = np.arange(-M, M + 1)
    if M > 0:
        # Perform integration in the phi direction (last dimension) using FFT
        # and select the original range of m (i.e. m = -M .. M)
        values = 2 * np.pi * np.fft.fft(values)[:, m_range] / Kphi
    else:
        values *= 2 * np.pi
    assert values.shape == (Ktheta, 2 * M + 1)

    values = {m: values[:, idx] for idx, m in enumerate(m_range)}
    coeffs = {m: None for m in m_range}
    for m in m_range:
        # Sample the Sheppard--Torok functions (basis functions)
        Flm = stf_F_vectorized(L, pol * m, cos_theta, sin_theta)
        assert Flm.shape == (L + 1, Ktheta)

        values_m = values[m]
        coeffs_m = np.dot(Flm, wtheta * values_m)
        assert coeffs_m.shape == (L + 1,)
        coeffs_m *= ONE_OVER_SQRT_2PI
        coeffs_m_lower_prec = np.dot(Flm, wtheta_lower_prec * values_m)
        coeffs_m_lower_prec *= ONE_OVER_SQRT_2PI
        error_m = abs(coeffs_m - coeffs_m_lower_prec)
        is_accurate_enough = np.allclose(error_m, 0.0, atol=atol, rtol=rtol)
        if not is_accurate_enough:
            raise ValueError("the requested tolerance could not be met.")

        if purify:
            coeffs_m.real[abs(coeffs_m.real) < 4.0 * EPS] = 0.0
            coeffs_m.imag[abs(coeffs_m.imag) < 4.0 * EPS] = 0.0
        coeffs[m] = coeffs_m
    return coeffs
