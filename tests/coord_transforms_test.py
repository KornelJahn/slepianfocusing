# coord_transforms_test.py

import numpy as np

from slepianfocusing.coord_transforms import (
    cartesian2d_to_polar2d,
    polar2d_to_cartesian2d,
    cartesian3d_to_spherical3d,
    spherical3d_to_cartesian3d,
    radialunitvec3d_to_spherical2d,
    spherical2d_to_radialunitvec3d,
)

EPS = np.finfo(np.float64).eps


def test_cartesian2d_polar2d():
    tol = dict(atol=10 * EPS, rtol=10 * EPS)
    cart2_coords = np.meshgrid([-1, 0, 1], [-1, 0, 1])
    pol2_coords = cartesian2d_to_polar2d(*cart2_coords)
    cart2_coords_re = polar2d_to_cartesian2d(*pol2_coords)
    assert np.allclose(cart2_coords, cart2_coords_re, **tol)


def test_cartesian3d_spherical3d():
    tol = dict(atol=10 * EPS, rtol=10 * EPS)
    cart3_coords = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])
    sph3_coords = cartesian3d_to_spherical3d(*cart3_coords)
    cart3_coords_re = spherical3d_to_cartesian3d(*sph3_coords)
    assert np.allclose(cart3_coords, cart3_coords_re, **tol)


def test_radialunitvec3d_spherical2d():
    tol = dict(atol=10 * EPS, rtol=10 * EPS)
    scart = np.meshgrid([-1, 1], [-1, 1], [-1, 1])
    scart /= np.sqrt(3.0)
    ssph = radialunitvec3d_to_spherical2d(*scart)
    scart_re = spherical2d_to_radialunitvec3d(*ssph)
    assert np.allclose(scart, scart_re, **tol)
