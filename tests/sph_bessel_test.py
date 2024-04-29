# sph_bessel_test.py

import numpy as np
from slepianfocusing.sph_bessel import sph_bessel_j, sph_bessel_j_vectorized

EPS = np.finfo(np.float64).eps


def test_sph_bessel_j():
    tol = dict(atol=0, rtol=10 * EPS)
    expected = np.array(
        [0.95885107720840600, 0.16253703063606657, 0.016371106607993413]
    )
    jl_array = sph_bessel_j(2, 0.5)
    assert np.allclose(jl_array, expected, **tol)
    jl_array = sph_bessel_j(200, 16.0)
    assert np.allclose(jl_array[-1], 2.3948463961666591e-196, **tol)


def test_sph_bessel_j_vectorized():
    tol = dict(atol=0, rtol=10 * EPS)
    expected = np.array(
        [
            [1.0, 0.95885107720840600],
            [0.0, 0.16253703063606657],
            [0.0, 0.016371106607993413],
        ]
    )
    x = np.array([0.0, 0.5])
    jl_array = sph_bessel_j_vectorized(2, x)
    assert np.allclose(jl_array, expected, **tol)
    x = np.array([0.0, 16.0])
    jl_array = sph_bessel_j_vectorized(200, x)
    assert np.allclose(jl_array[-1], [0.0, 2.3948463961666591e-196], **tol)
