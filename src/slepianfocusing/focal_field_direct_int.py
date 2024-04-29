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


@jit(nopython=True, cache=True)
def _integrate(pwa, sx, sy, sz, x, y, z, w, w_error):
    pwa_w = w * pwa
    pwa_w_error = w_error * pwa
    exp_factor = np.empty_like(sx, dtype=np.complex128)
    integral = np.empty((3,), dtype=np.complex128)
    integral_crude = np.empty_like(integral)
    ff = np.empty((3, x.size), dtype=np.complex128)
    errors = np.empty((3, x.size), dtype=np.float64)
    xflat = x.flatten()
    yflat = y.flatten()
    zflat = z.flatten()
    for i in range(xflat.size):
        if i % 100 == 0:
            print(i + 1, "/", xflat.size)
        xi, yi, zi = xflat[i], yflat[i], zflat[i]
        phase = 2.0 * np.pi * (sx * xi + sy * yi + sz * zi)
        exp_factor = np.cos(phase) + 1j * np.sin(phase)
        for j in range(3):
            integral[j] = np.sum(pwa_w[j] * exp_factor)
            integral_crude[j] = np.sum(pwa_w_error[j] * exp_factor)
        ff[:, i] = integral
        errors[:, i] = np.abs(integral - integral_crude)
    ff = ff.reshape((3,) + x.shape)
    errors = errors.reshape((3,) + x.shape)
    return ff, errors


def calculate_focal_field_direct_int(
    func, param, theta_level, phi_level, *ff_coords, tol=1e3
):
    shape = np.broadcast(*ff_coords).shape
    x, y, z = ff_coords
    assert x.shape == shape and y.shape == shape and z.shape == shape
    are_coords_scalar = not bool(shape)
    ff_coords = np.atleast_1d(*ff_coords)

    if np.isscalar(param):
        pwa_func = func
        Theta = param
    else:
        epf_func = func
        lens = param
        Theta = lens.Theta
        pwa_func = lens.transform_epf_func_to_pwa_func(epf_func)

    # Construct the product rule
    rule = GaussLegendreRule(theta_level).get_nodes_weights(0.0, Theta)
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

    atol, rtol = tol * EPS, tol * EPS
    if not np.all(errors < np.maximum(atol, rtol * abs(ff))):
        print("Max error:", errors.max())
        print("Tolerance:", np.maximum(atol, rtol * abs(ff)).max())
        raise RuntimeError(
            "integral could not be evaluated everywhere with an absolute "
            "precision of %.2e and a relative precision of %.2e."
            % (atol, rtol)
        )

    if are_coords_scalar:
        ff = np.squeeze(ff)
    return ff
