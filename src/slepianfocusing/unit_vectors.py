# unit_vectors.py

"""Helper functions for unit vector construction"""

__author__ = "Kornel JAHN (kornel.jahn@gmail.com)"
__copyright__ = "Copyright (c) 2012-2024 Kornel Jahn"
__license__ = "MIT"

import math
import numpy as np

try:
    from numba import jit
except ImportError:
    from .numba_dummy import jit

ONE_OVER_SQRT2 = 1.0 / math.sqrt(2.0)


@jit("c16[:](f8, f8, f8, f8)", nopython=True, cache=True)
def radial_uv(cos_theta, sin_theta, cos_phi, sin_phi):
    return np.array(
        [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta],
        dtype=np.complex128,
    )


@jit(nopython=True, cache=True)
def radial_uv_vectorized(cos_theta, sin_theta, cos_phi, sin_phi):
    assert cos_theta.shape == sin_theta.shape
    assert cos_theta.shape == cos_phi.shape
    assert cos_theta.shape == sin_phi.shape
    result = np.zeros((3, cos_theta.size), dtype=np.complex128)
    for i in range(cos_theta.size):
        result[:, i] = radial_uv(
            cos_theta.flat[i],
            sin_theta.flat[i],
            cos_phi.flat[i],
            sin_phi.flat[i],
        )
    return result.reshape((3,) + cos_theta.shape)


@jit("c16[:](f8, f8, f8, f8)", nopython=True, cache=True)
def polar_uv(cos_theta, sin_theta, cos_phi, sin_phi):
    return np.array(
        [cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta],
        dtype=np.complex128,
    )


@jit(nopython=True, cache=True)
def polar_uv_vectorized(cos_theta, sin_theta, cos_phi, sin_phi):
    assert cos_theta.shape == sin_theta.shape
    assert cos_theta.shape == cos_phi.shape
    assert cos_theta.shape == sin_phi.shape
    result = np.zeros((3, cos_theta.size), dtype=np.complex128)
    for i in range(cos_theta.size):
        result[:, i] = polar_uv(
            cos_theta.flat[i],
            sin_theta.flat[i],
            cos_phi.flat[i],
            sin_phi.flat[i],
        )
    return result.reshape((3,) + cos_theta.shape)


@jit("c16[:](f8, f8, f8, f8)", nopython=True, cache=True)
def azimuthal_uv(cos_theta, sin_theta, cos_phi, sin_phi):
    return np.array([-sin_phi, cos_phi, 0.0], dtype=np.complex128)


@jit(nopython=True, cache=True)
def azimuthal_uv_vectorized(cos_theta, sin_theta, cos_phi, sin_phi):
    assert cos_theta.shape == sin_theta.shape
    assert cos_theta.shape == cos_phi.shape
    assert cos_theta.shape == sin_phi.shape
    result = np.zeros((3, cos_theta.size), dtype=np.complex128)
    for i in range(cos_theta.size):
        result[:, i] = azimuthal_uv(
            cos_theta.flat[i],
            sin_theta.flat[i],
            cos_phi.flat[i],
            sin_phi.flat[i],
        )
    return result.reshape((3,) + cos_theta.shape)


@jit("c16[:](f8, f8, f8, f8)", nopython=True, cache=True)
def e_plus_uv(cos_theta, sin_theta, cos_phi, sin_phi):
    theta_uv = polar_uv(cos_theta, sin_theta, cos_phi, sin_phi)
    phi_uv = azimuthal_uv(cos_theta, sin_theta, cos_phi, sin_phi)
    return (theta_uv + 1j * phi_uv) * ONE_OVER_SQRT2


@jit(nopython=True, cache=True)
def e_plus_uv_vectorized(cos_theta, sin_theta, cos_phi, sin_phi):
    assert cos_theta.shape == sin_theta.shape
    assert cos_theta.shape == cos_phi.shape
    assert cos_theta.shape == sin_phi.shape
    result = np.zeros((3, cos_theta.size), dtype=np.complex128)
    for i in range(cos_theta.size):
        result[:, i] = e_plus_uv(
            cos_theta.flat[i],
            sin_theta.flat[i],
            cos_phi.flat[i],
            sin_phi.flat[i],
        )
    return result.reshape((3,) + cos_theta.shape)


@jit("c16[:](f8, f8, f8, f8)", nopython=True, cache=True)
def e_minus_uv(cos_theta, sin_theta, cos_phi, sin_phi):
    theta_uv = polar_uv(cos_theta, sin_theta, cos_phi, sin_phi)
    phi_uv = azimuthal_uv(cos_theta, sin_theta, cos_phi, sin_phi)
    return (theta_uv - 1j * phi_uv) * ONE_OVER_SQRT2


@jit(nopython=True, cache=True)
def e_minus_uv_vectorized(cos_theta, sin_theta, cos_phi, sin_phi):
    assert cos_theta.shape == sin_theta.shape
    assert cos_theta.shape == cos_phi.shape
    assert cos_theta.shape == sin_phi.shape
    result = np.zeros((3, cos_theta.size), dtype=np.complex128)
    for i in range(cos_theta.size):
        result[:, i] = e_minus_uv(
            cos_theta.flat[i],
            sin_theta.flat[i],
            cos_phi.flat[i],
            sin_phi.flat[i],
        )
    return result.reshape((3,) + cos_theta.shape)
