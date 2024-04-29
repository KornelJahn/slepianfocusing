# focal_field_czt.py

"""Focal field calculation using the Chirp-Z Transform"""

__author__ = "Kornel JAHN (kornel.jahn@gmail.com)"
__copyright__ = "Copyright (c) 2012-2024 Kornel Jahn"
__license__ = "MIT"

import math
import numpy as np
from .math_ext import next_power_of_2
from .coord_transforms import radialunitvec3d_to_spherical2d

EPS = np.finfo(np.float64).eps


def _czt_parameters(ds, dx, M, N):
    """Precompute Chirp-Z Transform constants."""
    # Calculate CZT constants A and W
    A = np.exp(1j * np.pi * ds * dx * (N - 1.0))
    W = np.exp(2j * np.pi * ds * dx)

    # Calculate prefactor
    C = (
        ds
        * np.exp(0.5j * np.pi * ds * dx * (M - 1.0) * (N - 1.0))
        * np.exp(-1j * np.pi * ds * dx * (M - 1.0) * np.arange(0, N))
    )

    return A, W, C


def _czt_1d_on_each_row(x, A, W, N):
    """Compute the 1d Chirp Z-Transform."""
    fft = np.fft.fft
    ifft = np.fft.ifft
    A = complex(A)
    W = complex(W)
    M = x.shape[1]
    L = next_power_of_2(M + N - 1)

    range_ = np.arange(L)
    y = np.power(A, -range_[:M]) * np.power(W, 0.5 * range_[:M] ** 2) * x
    fy = fft(y, L)

    v = np.zeros(L, dtype=np.complex128)
    v[:N] = np.power(W, -0.5 * range_[:N] ** 2)
    v[(L - M + 1) :] = np.power(W, -0.5 * range_[(M - 1) : 0 : -1] ** 2)
    fv = fft(v)

    result = ifft(fv * fy)[:, :N]
    n = np.arange(N)
    result *= np.power(W, 0.5 * n**2)

    return result


def calculate_focal_field_czt(epf_func, lens, M, Nx, Lx, z):
    z = np.atleast_1d(z)
    if z.ndim != 1:
        raise ValueError("z must be a scalar or a 1d vector.")
    if Lx <= 0.0:
        raise ValueError("the lateral side-length Lx must be positive.")
    if not (M > 0 and Nx > 0):
        raise ValueError("the grid sizes M and Nx must be positive.")

    # TODO: check sampling condition
    # TODO: suggest optimal CZT combination of M and Nx

    sin_Theta = lens.sin_Theta

    pwa_func = lens.transform_epf_func_to_pwa_func(epf_func)
    ds = 2 * sin_Theta / M
    dx = Lx / Nx
    s_range = -sin_Theta + (np.arange(0, M) + 0.5) * ds
    sx, sy = np.meshgrid(s_range, s_range)
    mask = (sx**2 + sy**2) > sin_Theta**2
    sz = np.sqrt(np.where(mask, 0.0, 1.0 - sx**2 - sy**2))
    pwdir_angles = radialunitvec3d_to_spherical2d(sx, sy, sz)
    pwa_focal_plane = pwa_func(*pwdir_angles) / np.where(sz != 0.0, sz, 1.0)
    pwa_focal_plane[:, mask] = 0.0
    A, W, C = _czt_parameters(ds, dx, M, Nx)

    Nz = len(z)
    ff = np.empty((3, Nz, Nx, Nx), dtype=np.complex128)
    for iz in range(Nz):
        if iz % 10:
            print(iz + 1, "/", Nz)
        defocus_term = np.exp(2j * np.pi * sz * z[iz])
        pwa = pwa_focal_plane * defocus_term
        for ipol in range(3):
            temp = pwa[ipol]
            temp = _czt_1d_on_each_row(temp, A, W, Nx)
            temp *= C
            temp = temp.T
            temp = _czt_1d_on_each_row(temp, A, W, Nx)
            temp *= C
            ff[ipol, iz, ...] = temp.T
    return np.squeeze(ff)
