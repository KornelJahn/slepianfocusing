# sph_bessel.py

"""Implementation of the spherical Bessel functions j_{l}"""

__author__ = "Kornel JAHN (kornel.jahn@gmail.com)"
__copyright__ = "Copyright (c) 2012-2024 Kornel Jahn"
__license__ = "MIT"

import numpy as np

try:
    from numba import jit
except ImportError:
    from .numba_dummy import jit

EPS = np.finfo(np.float64).eps


@jit("f8[:](i8, f8)", nopython=True, cache=True)
def sph_bessel_j(lmax, x):
    """Compute the spherical Bessel function values j_l(x) for fixed x and
    0 <= l <= L. Uses the Steed--Barnett algorithm.

    It is based on the Fortran routine SBESJ of [1] and its implementation
    in the GNU Scientific Library.

    References:

    [1] Barnett, A. R. (1981). An algorithm for regular and irregular Coulomb
        and Bessel functions of real order to machine accuracy. Computer Physics
        Communications, 21(3), 297-314.
    """
    assert lmax < 1000

    result = np.empty((lmax + 1,), dtype=np.float64)

    if x < 0.0 or lmax < 0:
        raise ValueError("negative arguments are not supported.")
    elif x == 0.0:
        for l in range(lmax + 1):
            result[l] = 0.0
        result[0] = 1.0
        return result
    elif x < 1.0e-7:
        # First two terms of the Taylor series
        result[0] = 1.0
        c = 1.0
        x_sq = x * x
        for l in range(1, lmax + 1):
            c *= x / (2 * l + 1)
            result[l] = c * (1.0 - x_sq / (2 * (2 * l + 3)))
        return result

    x_inv = 1.0 / x
    W = 2.0 * x_inv
    F = 1.0
    FP = (lmax + 1) * x_inv
    B = 2.0 * FP + x_inv
    end = B + 20000.0 * W
    D = 1.0 / B
    delta = -D
    limit = 20000

    FP += delta

    # Continued fraction
    for l in range(1, limit):
        B += W
        D = 1.0 / (B - D)
        delta *= B * D - 1.0
        FP += delta
        if D < 0.0:
            F = -F
        if B > end:
            raise ValueError("the algorithm failed to converge.")
        if abs(delta) < abs(FP) * EPS:
            break
    if l == limit:
        raise ValueError(
            "calculation of spherical Bessel function values " "failed"
        )

    FP *= F

    if lmax > 0:
        # Downward recursion
        XP2 = FP
        PL = lmax * x_inv
        result[lmax] = F
        for l in range(lmax, 0, -1):
            result[l - 1] = PL * result[l] + XP2
            FP = PL * result[l - 1] - result[l]
            XP2 = FP
            PL -= x_inv
        F = result[0]

    # Normalization
    W = x_inv / np.hypot(FP, F)
    result[0] = W * F
    if lmax > 0:
        for l in range(1, lmax + 1):
            result[l] *= W

    return result


@jit(nopython=True, cache=True)
def sph_bessel_j_vectorized(lmax, x):
    """Vectorized version of sph_bessel_j that operates on multiple values of
    x.
    """
    result = np.empty((lmax + 1, x.size), dtype=np.float64)
    for i in range(x.size):
        result[:, i] = sph_bessel_j(lmax, x.flat[i])
    return result.reshape((lmax + 1,) + x.shape)
