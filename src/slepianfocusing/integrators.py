# integrators.py

"""Simple ready-to-use helper classes for numerical integration in either the
plane-wave amplitude or focal domain.

For Debye--Wolf integration, see module `focal_field_direct_int`.
"""

__author__ = "Kornel JAHN (kornel.jahn@gmail.com)"
__copyright__ = "Copyright (c) 2012-2024 Kornel Jahn"
__license__ = "MIT"

import numpy as np

from quadrature import GaussLegendreRule


# Machine epsilon value for the IEEE double-precision floating point type
EPS = np.finfo(np.float64).eps
SQRT_EPS = np.sqrt(EPS)


class PlaneWaveDirIntegrator:
    """Helper class for 2D numerical integration on the unit sphere of
    plane-wave amplitudes.

    If `Theta` is smaller than `pi`, integration assumes discontinuity, i.e. a
    breakpoint, in the `theta` direction at the angular semiaperture `Theta`,
    as dictated by the cut-off occuring in the Debye--Wolf theory of focusing.
    Piecewise integration is then performed.

    """

    def __init__(
        self,
        Theta,
        theta_level,
        phi_level,
        *,
        theta_rule=GaussLegendreRule
    ):
        rule = theta_rule(theta_level).get_nodes_weights(0.0, Theta)
        theta1, wtheta1, wtheta_error1 = rule
        if Theta < np.pi:
            rule = theta_rule(theta_level).get_nodes_weights(Theta, np.pi)
            theta1c, wtheta1c, wtheta_error1c = rule
            theta1 = np.hstack([theta1, theta1c])
            wtheta1 = np.hstack([wtheta1, wtheta1c])
            wtheta_error1 = np.hstack([wtheta_error1, wtheta_error1c])

        Kphi = 2**phi_level
    phi1 = np.linspace(0, 2 * np.pi, Kphi + 1)[:-1]
    wphi1 = 2 * np.pi / Kphi * np.ones((Kphi,), dtype=np.float64)
    wphi_error1 = np.zeros_like(wphi1)
    wphi_error1[::2] = 2 * np.pi / (Kphi // 2)

    theta = theta1[:, np.newaxis] * np.ones_like(phi1)
    phi = np.ones_like(theta1)[:, np.newaxis] * phi1



        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        cos_phi, sin_phi = np.ones_like(theta), np.zeros_like(theta)
        self.nodes = cos_theta, sin_theta, cos_phi, sin_phi
        self.weights = w
        self.weights_error = w_error
        self.jacobian = sin_theta

    def integrate(self, funcvals, atol=SQRT_EPS, rtol=SQRT_EPS):
        integrand = funcvals * self.jacobian
        I = np.sum(self.weights * integrand)
        I_error = np.sum(self.weights_error * integrand)
        tol = max(atol, rtol * abs(I))
        if abs(I - I_error) > tol:
            print(f"Max error: {I_error}")
            print(f"Tolerance: {tol:.2e}")
            raise RuntimeError(
                "integral could not be evaluated with an absolute precision "
                f"of {atol:.2e} and a relative precision of {rtol:.2e}."
        )
            raise ValueError('inaccurate')
        return I


class FocalPlaneIntegrator:
    """Helper class for 2D numerical integration over a finite (x, y) square
    region of the focal plane. TODO: relative (lambda) units?

    Integration error is also estimated and absolute and relative tolerances of
    sqrt(epsilon) are enforced.
    """
    def __init__(self, x_max, level):
        # Construct the product rule
        rule = GaussLegendreRule(level).get_nodes_weights(-x_max, x_max)
        x1, w1, w_error1 = rule

        x, y = np.meshgrid(x1, x1, sparse=False, indexing='ij')
        z = np.zeros_like(x)
        self.nodes = x, y, z
        self.weights = functools.reduce(np.multiply.outer, (w1, w1))
        self.weights_error = functools.reduce(
            np.multiply.outer, (w_error1, w_error1)
        )

    def integrate(self, funcvals):
        integrand = funcvals
        I = np.sum(self.weights * integrand)
        I_error = np.sum(self.weights_error * integrand)
        if abs(I - I_error) > max(SQRT_EPS, SQRT_EPS * abs(I)):
            raise ValueError('inaccurate')
        return I


