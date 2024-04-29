# coord_transforms.py

"""Helper functions for coordinate transforms"""

__author__ = "Kornel JAHN (kornel.jahn@gmail.com)"
__copyright__ = "Copyright (c) 2012-2024 Kornel Jahn"
__license__ = "MIT"

import numpy as np
import math


def cartesian2d_to_polar2d(x, y):
    rho = np.sqrt(x**2 + y**2)
    mask = rho != 0.0
    cos_phi = np.where(mask, x, 1.0) / np.where(mask, rho, 1.0)
    sin_phi = y / np.where(mask, rho, 1.0)
    return rho, cos_phi, sin_phi


def polar2d_to_cartesian2d(rho, cos_phi, sin_phi):
    x = rho * cos_phi
    y = rho * sin_phi
    return x, y


def cartesian3d_to_spherical3d(x, y, z):
    rho2 = x**2 + y**2
    r = np.sqrt(rho2 + z**2)
    rho = np.sqrt(rho2)
    mask = r != 0.0
    cos_theta = np.where(mask, z, 1.0) / np.where(mask, r, 1.0)
    sin_theta = rho / np.where(mask, r, 1.0)
    mask = rho != 0.0
    cos_phi = np.where(mask, x, 1.0) / np.where(mask, rho, 1.0)
    sin_phi = y / np.where(mask, rho, 1.0)
    return r, cos_theta, sin_theta, cos_phi, sin_phi


def spherical3d_to_cartesian3d(r, cos_theta, sin_theta, cos_phi, sin_phi):
    rho = r * sin_theta
    x = rho * cos_phi
    y = rho * sin_phi
    z = r * cos_theta
    return x, y, z


def cartesian3d_to_cylindrical3d(x, y, z):
    rho, cos_phi, sin_phi = cartesian2d_to_polar2d(x, y)
    return rho, cos_phi, sin_phi, z


def cylindrical3d_to_cartesian3d(rho, cos_phi, sin_phi, z):
    x, y = polar2d_to_cartesian2d(rho, cos_phi, sin_phi)
    return x, y, z


def cylindrical3d_to_spherical3d(rho, cos_phi, sin_phi, z):
    r = np.sqrt(rho**2 + z**2)
    mask = r != 0.0
    cos_theta = np.where(mask, z, 1.0) / np.where(mask, r, 1.0)
    sin_theta = rho / np.where(mask, r, 1.0)
    return r, cos_theta, cos_phi, sin_theta, sin_phi


def spherical3d_to_cylindrical3d(r, cos_theta, sin_theta, cos_phi, sin_phi):
    rho = r * sin_theta
    z = r * cos_theta
    return rho, cos_phi, sin_phi, z


def radialunitvec3d_to_spherical2d(sx, sy, sz):
    cos_theta = sz
    sin_theta = np.sqrt(sx**2 + sy**2)
    mask = sin_theta != 0.0
    cos_phi = np.where(mask, sx, 1.0) / np.where(mask, sin_theta, 1.0)
    sin_phi = sy / np.where(mask, sin_theta, 1.0)
    return cos_theta, sin_theta, cos_phi, sin_phi


def spherical2d_to_radialunitvec3d(cos_theta, sin_theta, cos_phi, sin_phi):
    sx = sin_theta * cos_phi
    sy = sin_theta * sin_phi
    sz = cos_theta
    return sx, sy, sz
