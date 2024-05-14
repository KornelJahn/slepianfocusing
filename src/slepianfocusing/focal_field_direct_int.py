# focal_field_direct_int.py

"""Focal field calculation by direct Debye--Wolf integration"""

__author__ = "Kornel JAHN (kornel.jahn@gmail.com)"
__copyright__ = "Copyright (c) 2012-2024 Kornel Jahn"
__license__ = "MIT"

import math
import numpy as np

try:
    from numba import jit
except ImportError:
    from .numba_dummy import jit

from .quadrature import GaussLegendreRule
from .coord_transforms import spherical2d_to_radialunitvec3d

EPS = np.finfo(np.float64).eps


def calculate_focal_field_direct_int(
    pwa_func,
    Theta,
    theta_level,
    phi_level,
    x,
    y,
    z,
    *,
    atol=1e3*EPS,
    rtol=1e3*EPS,
    theta_rule=GaussLegendreRule,
):
    """Calculate the focal field at `(x, y, z)` using 2D Debye--Wolf
    integration in the plane-wave amplitudes domain.

    Notes
    -----
    If entrance pupil field function is available, you may convert it with help
    of the `lens` model to a plane-wave amplitudes function as

    ```
    pwa_func=lens.transform_epf_func_to_pwa_func(epf_func)
    ```
    with
    ```
    Theta=lens.Theta
    ```
    """
    assert atol > 0
    ff_coords = np.broadcast(x, y, z).shape
    are_coords_scalar = not bool(shape)
    x, y, z = np.atleast_1d(x, y, z)

    # Construct the product rule
    rule = theta_rule(theta_level).get_nodes_weights(0.0, Theta)
    theta1, wtheta1, wtheta_error1 = rule

    Kphi = 2**phi_level
    phi1 = np.linspace(0, 2 * np.pi, Kphi + 1)[:-1]
    wphi1 = 2 * np.pi / Kphi * np.ones((Kphi,), dtype=np.float64)
    wphi_error1 = np.zeros_like(wphi1)
    wphi_error1[::2] = 2 * np.pi / (Kphi // 2)

    theta = theta1[:, np.newaxis] * np.ones_like(phi1)
    phi = np.ones_like(theta1)[:, np.newaxis] * phi1

    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    w = wtheta1[:, np.newaxis] * wphi1
    w_error = wtheta_error1[:, np.newaxis] * wphi_error1

    pwa = pwa_func(cos_theta, sin_theta, cos_phi, sin_phi) * sin_theta
    sx, sy, sz = spherical2d_to_radialunitvec3d(
        cos_theta, sin_theta, cos_phi, sin_phi
    )
    ff, errors = _integrate(pwa, sx, sy, sz, x, y, z, w, w_error)

    tol = np.maximum(atol, rtol * abs(ff))
    if not np.all(errors < tol):
        max_error_indices = np.unravel_index(
            (errors / tol).argmax(), errors.shape
        )
        print(f"Max. error: {errors[max_error_indices]:.2e}")
        print(f"Tolerance: {tol[max_error_indices]:.2e}")
        raise RuntimeError(
            "integral could not be evaluated everywhere with an absolute "
            f"precision of {atol:.2e} and a relative precision of {rtol:.2e}."
        )

    if are_coords_scalar:
        ff = np.squeeze(ff)
    return ff


@jit(nopython=True, cache=True)
def _integrate(pwa, sx, sy, sz, x, y, z, w, w_error):
    ff_coords = np.broadcast(x, y, z)
    pwa_w = w * pwa
    pwa_w_error = w_error * pwa

    # Pre-allocating array for integral values
    integral = np.empty((3,), dtype=np.complex128)
    integral_crude = np.empty_like(integral)

    ff = np.empty((3, ff_coords.size), dtype=np.complex128)
    errors = np.empty((3, ff_coords.size), dtype=np.float64)
    for i, (xi, yi, zi) in enumerate(ff_coords):
        if i % 100 == 0:
            print(f"Progress: {i + 1} / {ff_coords.size}")
        phase = 2.0 * np.pi * (sx * xi + sy * yi + sz * zi)
        exp_factor = np.exp(1j * phase)
        for j in range(3):
            integral[j] = np.sum(pwa_w[j] * exp_factor)
            integral_crude[j] = np.sum(pwa_w_error[j] * exp_factor)
        ff[:, i] = integral
        errors[:, i] = np.abs(integral - integral_crude)
    ff = ff.reshape((3,) + ff_coords.shape)
    errors = errors.reshape((3,) + ff_coords.shape)
    return ff, errors
