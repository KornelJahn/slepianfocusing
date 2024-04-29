# assoc_legendre_test.py

import numpy as np
from slepianfocusing.assoc_legendre import (
    INV_SCALE_FACTOR,
    _nalf_P_first_scaled,
    nalf_P,
    nalf_P_vectorized,
)

EPS = np.finfo(np.float64).eps


def test_nalf_P_first_scaled():
    tol = dict(atol=0, rtol=10 * EPS)
    assert np.allclose(
        _nalf_P_first_scaled(-2, 0.6, 0.8) * INV_SCALE_FACTOR,
        0.61967733539318670,
        **tol,
    )
    assert np.allclose(
        _nalf_P_first_scaled(-1, 0.6, 0.8) * INV_SCALE_FACTOR,
        0.69282032302755092,
        **tol,
    )
    assert np.allclose(
        _nalf_P_first_scaled(0, 0.6, 0.8) * INV_SCALE_FACTOR,
        0.70710678118654752,
        **tol,
    )
    assert np.allclose(
        _nalf_P_first_scaled(1, 0.6, 0.8) * INV_SCALE_FACTOR,
        -0.69282032302755092,
        **tol,
    )
    assert np.allclose(
        _nalf_P_first_scaled(2, 0.6, 0.8) * INV_SCALE_FACTOR,
        0.61967733539318670,
        **tol,
    )


def test_nalf_P():
    tol = dict(atol=0, rtol=1e3 * EPS)
    P_array = nalf_P(
        0,
        0,
        1.0,
        0.0,
    )
    assert np.allclose(P_array[0], 0.70710678118654752, **tol)
    P_array = nalf_P(1, 0, 1.0, 0.0)
    assert np.allclose(P_array[1], 1.2247448713915890, **tol)
    P_array = nalf_P(2, 1, 1.0, 0.0)
    assert np.allclose(P_array[2], 0.0, **tol)
    P_array = nalf_P(3, 1, 0.6, 0.8)
    assert np.allclose(P_array[3], -0.51845925587262882, **tol)
    P_array = nalf_P(3, -1, 0.6, 0.8)
    assert np.allclose(P_array[3], 0.51845925587262882, **tol)
    P_array = nalf_P(3, 3, 0.6, 0.8)
    assert np.allclose(P_array[3], -0.53546241698180835, **tol)
    P_array = nalf_P(200, 50, 0.6, 0.8)
    assert np.allclose(P_array[200], -0.14950358680644932, **tol)
    P_array = nalf_P(1000, 50, 0.6, 0.8)
    assert np.allclose(P_array[1000], 0.37185592207991392, **tol)
    P_array = nalf_P(1000, 100, 0.6, 0.8)
    assert np.allclose(P_array[1000], 0.61015996871967542, **tol)
    P_array = nalf_P(1000, 1000, 0.6, 0.8)
    assert np.allclose(P_array[1000], 5.1973375872595806e-97, **tol)
    P_array = nalf_P(1000, 400, np.cos(0.1), np.sin(0.1))
    assert np.allclose(P_array[1000], 8.2324452716734146e-196, **tol)


def test_nalf_P_vectorized():
    tol = dict(atol=0, rtol=10 * EPS)
    theta = np.array([[0.0, np.pi / 2], [np.pi, 3 * np.pi / 2]])
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    L = 10

    P_array_obtained = nalf_P_vectorized(L, 0, cos_theta, sin_theta)

    P_array_expected = np.zeros((L + 1, 2, 2), dtype=np.float64)
    P_array_expected[:, 0, 0] = nalf_P(L, 0, cos_theta[0, 0], sin_theta[0, 0])
    P_array_expected[:, 0, 1] = nalf_P(L, 0, cos_theta[0, 1], sin_theta[0, 1])
    P_array_expected[:, 1, 0] = nalf_P(L, 0, cos_theta[1, 0], sin_theta[1, 0])
    P_array_expected[:, 1, 1] = nalf_P(L, 0, cos_theta[1, 1], sin_theta[1, 1])

    assert np.allclose(P_array_obtained, P_array_expected, **tol)
