# lenses_test.py

import numpy as np

from slepianfocusing.lenses import AplanaticLens, HerschelLens

EPS = np.finfo(np.float64).eps


def test_aplanatic_lens():
    _test_lens(AplanaticLens())


def test_herschel_lens():
    _test_lens(HerschelLens())


def _test_lens(lens):
    tol = dict(atol=10 * EPS, rtol=10 * EPS)
    rho_ep = np.zeros((9,))
    phi_ep = np.zeros_like(rho_ep)
    rho_ep[1:] = np.repeat([0.5, 0.95], 4)
    phi_ep[1:] = np.tile([0, 90, 180, 270], 2) * np.pi / 360.0
    ep_coords = rho_ep, np.cos(phi_ep), np.sin(phi_ep)
    pw_angles = lens.transform_ep_coords_to_pw_angles(*ep_coords)
    ep_coords_re = lens.transform_pw_angles_to_ep_coords(*pw_angles)
    assert np.allclose(ep_coords, ep_coords_re, **tol)

    epf = np.zeros((2, 9))
    epf[0] = 1.0
    pwa = lens.transform_epf_to_pwa(epf, *pw_angles)

    # Check vector orthogonality of the plane wave amplitudes
    cos_theta, sin_theta, cos_phi, sin_phi = pw_angles
    # Unit vector in the radial direction
    ruv = np.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])
    ruv_dot_pwa = np.sum(pwa * ruv, 0)
    assert np.allclose(0.0, ruv_dot_pwa, **tol)

    epf_re = lens.transform_pwa_to_epf(pwa, *pw_angles)
    assert np.allclose(epf, epf_re, **tol)

    def epf_func(rho_ep, cos_theta_ep, sin_theta_ep):
        Ex = rho_ep * (cos_theta_ep - sin_theta_ep)
        Ey = rho_ep * (sin_theta_ep + cos_theta_ep)
        return np.array([Ex, Ey])

    pwa_func = lens.transform_epf_func_to_pwa_func(epf_func)
    epf_func_re = lens.transform_pwa_func_to_epf_func(pwa_func)

    epf = epf_func(*ep_coords)
    epf_re = epf_func_re(*ep_coords)
    assert np.allclose(epf, epf_re, **tol)
