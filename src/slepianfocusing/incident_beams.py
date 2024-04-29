# incident_beams.py

"""Helper functions for incident beam profile construction"""

__author__ = "Kornel JAHN (kornel.jahn@gmail.com)"
__copyright__ = "Copyright (c) 2012-2024 Kornel Jahn"
__license__ = "MIT"

import numpy as np
import math
from scipy.special import hermite, assoc_laguerre


SQRT2 = math.sqrt(2.0)
ONE_OVER_SQRT2 = 1.0 / SQRT2


def gaussian_amplitude(w0, rho, *args):
    if not (np.isscalar(w0) and w0 > 0):
        raise ValueError("waist radius must be a positive scalar.")
    A = np.exp(-((rho / w0) ** 2)) / w0
    return A


def hermite_gaussian_amplitude(nx, ny, w0, rho, cos_phi, sin_phi):
    are_indices_ok = (
        isinstance(nx, int) and nx >= 0 and isinstance(ny, int) and ny >= 0
    )
    if not are_indices_ok:
        raise ValueError("indices nx and ny must be positive.")
    if not (np.isscalar(w0) and w0 > 0):
        raise ValueError("waist radius w0 must be a positive scalar.")

    xi = SQRT2 * rho / w0
    A = (
        hermite(nx)(xi * cos_phi)
        * hermite(ny)(xi * sin_phi)
        * np.exp(-0.5 * xi**2)
        / w0
    )
    return A


def laguerre_gaussian_amplitude(n, m, w0, rho, cos_phi, sin_phi):
    are_indices_ok = isinstance(n, int) and n >= 0 and isinstance(m, int)
    if not are_indices_ok:
        raise ValueError("indices n and m must be positive.")
    if not (np.isscalar(w0) and w0 > 0):
        raise ValueError("waist radius w0 must be a positive scalar.")
    abs_m = abs(m)
    xi = SQRT2 * rho / w0
    xi2 = xi**2
    exp_mi_phi = cos_phi - 1j * sin_phi
    A = (
        xi**abs_m
        * assoc_laguerre(xi2, n, abs_m)
        * np.exp(-0.5 * xi2)
        * exp_mi_phi**m
        / w0
    )
    return A


def linear_polarization(cos_phi0, sin_phi0, rho, *args):
    is_phi0_ok = (
        np.isscalar(cos_phi0)
        and np.isscalar(sin_phi0)
        and np.allclose(cos_phi0**2 + sin_phi0**2, 1.0, atol=1e-14, rtol=1e-14)
    )
    if not is_phi0_ok:
        raise ValueError(
            "cos_phi0 and sin_phi0 must be scalar values and "
            "cos_phi0**2 + sin_phi0**2 ~ 1 must hold."
        )
    elp = np.ones((2,) + rho.shape, dtype=np.float64)
    elp[0] *= cos_phi0
    elp[1] *= sin_phi0
    return elp


def linear_x_polarization(rho, *args):
    return linear_polarization(1, 0, rho, *args)


def linear_y_polarization(rho, *args):
    return linear_polarization(0, 1, rho, *args)


def circular_polarization(direction, rho, *args):
    if abs(direction) != 1:
        raise ValueError("direction must be +/-1.")
    ecp = ONE_OVER_SQRT2 * np.ones((2,) + rho.shape, dtype=np.complex128)
    ecp[1] *= direction * 1j
    return ecp


def left_circular_polarization(rho, *args):
    return circular_polarization(1, rho, *args)


def right_circular_polarization(rho, *args):
    return circular_polarization(-1, rho, *args)


def hg_radially_polarized_beam(w0, *args):
    if not (np.isscalar(w0) and w0 > 0):
        raise ValueError("waist radius must be a positive scalar.")
    HG10 = hermite_gaussian_amplitude(1, 0, w0, *args)
    HG01 = hermite_gaussian_amplitude(0, 1, w0, *args)
    ex = linear_x_polarization(*args)
    ey = linear_y_polarization(*args)
    return HG10 * ex + HG01 * ey


def hg_azimuthally_polarized_beam(w0, *args):
    if not (np.isscalar(w0) and w0 > 0):
        raise ValueError("waist radius must be a positive scalar.")
    HG10 = hermite_gaussian_amplitude(1, 0, w0, *args)
    HG01 = hermite_gaussian_amplitude(0, 1, w0, *args)
    ex = linear_x_polarization(*args)
    ey = linear_y_polarization(*args)
    return -HG01 * ex + HG10 * ey


def helical_phase_factor(m, rho, cos_phi, sin_phi):
    sign = 1 if m >= 0 else -1
    return (cos_phi + sign * 1j * sin_phi) ** abs(m)
