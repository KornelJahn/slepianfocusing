# sheppard_torok_test.py

import numpy as np
from slepianfocusing.sheppard_torok import (
    INV_SCALE_FACTOR,
    _stf_F_first_scaled,
    stf_F,
    stf_F_vectorized,
)

EPS = np.finfo(np.float64).eps


def test_stf_F_first_scaled():
    tol = dict(atol=0, rtol=10 * EPS)
    assert np.allclose(
        _stf_F_first_scaled(-2, 0.6, 0.8) * INV_SCALE_FACTOR,
        0.25298221281347035,
        **tol,
    )
    assert np.allclose(
        _stf_F_first_scaled(-1, 0.6, 0.8) * INV_SCALE_FACTOR,
        0.24494897427831781,
        **tol,
    )
    assert np.allclose(
        _stf_F_first_scaled(0, 0.6, 0.8) * INV_SCALE_FACTOR,
        0.69282032302755092,
        **tol,
    )
    assert np.allclose(
        _stf_F_first_scaled(1, 0.6, 0.8) * INV_SCALE_FACTOR,
        0.97979589711327124,
        **tol,
    )
    assert np.allclose(
        _stf_F_first_scaled(2, 0.6, 0.8) * INV_SCALE_FACTOR,
        -1.0119288512538814,
        **tol,
    )


def test_stf_F():
    tol = dict(atol=0, rtol=1e3 * EPS)
    F_array = stf_F(1, 0, 1.0, 0.0)
    assert np.allclose(F_array[1], 0.0, **tol)
    F_array = stf_F(2, 1, 1.0, 0.0)
    assert np.allclose(F_array[2], 1.5811388300841897, **tol)
    F_array = stf_F(3, 1, 0.6, 0.8)
    assert np.allclose(F_array[3], -0.59866518188383062, **tol)
    F_array = stf_F(3, -1, 0.6, 0.8)
    assert np.allclose(F_array[3], 0.97283092056122476, **tol)
    F_array = stf_F(3, 3, 0.6, 0.8)
    assert np.allclose(F_array[3], 0.92744811175612407, **tol)
    F_array = stf_F(200, 50, 0.6, 0.8)
    assert np.allclose(F_array[200], -0.81158425393639232, **tol)
    F_array = stf_F(1000, 50, 0.6, 0.8)
    assert np.allclose(F_array[1000], 0.78714726209707386, **tol)
    F_array = stf_F(1000, 100, 0.6, 0.8)
    assert np.allclose(F_array[1000], 0.57442778397145688, **tol)
    F_array = stf_F(1000, 1000, 0.6, 0.8)
    assert np.allclose(F_array[1000], -1.0389481731689596e-96, **tol)
    F_array = stf_F(1000, 400, np.cos(0.1), np.sin(0.1))
    assert np.allclose(F_array[1000], -6.4894817948888661e-195, **tol)


def test_stf_F_vectorized():
    tol = dict(atol=0, rtol=10 * EPS)
    theta = np.array([[0.0, np.pi / 2], [np.pi, 3 * np.pi / 2]])
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    L = 10

    F_array_obtained = stf_F_vectorized(L, 0, cos_theta, sin_theta)

    F_array_expected = np.zeros((L + 1, 2, 2), dtype=np.float64)
    F_array_expected[:, 0, 0] = stf_F(L, 0, cos_theta[0, 0], sin_theta[0, 0])
    F_array_expected[:, 0, 1] = stf_F(L, 0, cos_theta[0, 1], sin_theta[0, 1])
    F_array_expected[:, 1, 0] = stf_F(L, 0, cos_theta[1, 0], sin_theta[1, 0])
    F_array_expected[:, 1, 1] = stf_F(L, 0, cos_theta[1, 1], sin_theta[1, 1])

    assert np.allclose(F_array_obtained, F_array_expected, **tol)


# def test_sum_stf_F():
#    tol = dict(atol=0, rtol=10 * EPS)
#
#    c = np.array([1, 2])
#    assert(np.allclose(sum_stf_F(c, 0, 0.6, 0.8),
#                       2.5518523292071110, **tol))
#    assert(np.allclose(sum_stf_F(c, -1, 0.6, 0.8),
#                       1.6363511447524047, **tol))
#
#    c = np.array([1, 2, 3])
#    assert(np.allclose(sum_stf_F(c, 0, 0.6, 0.8),
#                       4.1072300968249975, **tol))
#
#
#    c = np.array([1, 2, 3, 4])
#    assert(np.allclose(sum_stf_F(c, -1, 0.6, 0.8),
#                       7.5280864999692740, **tol))
#    assert(np.allclose(sum_stf_F(c, 0, 0.6, 0.8),
#                       3.0143469374708056, **tol))
#    assert(np.allclose(sum_stf_F(c, 1, 0.6, 0.8),
#                       -3.8944180753896520, **tol))
