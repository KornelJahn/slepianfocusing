# lenses.py

"""Classes to represent simple lens models"""

__author__ = "Kornel JAHN (kornel.jahn@gmail.com)"
__copyright__ = "Copyright (c) 2012-2024 Kornel Jahn"
__license__ = "MIT"

import math
import numpy as np

EPS = np.finfo(np.float64).eps


## Common functionality


class Lens:
    """Abstract base class for simple lens models."""

    def __init__(self, n, sin_Theta, Theta):
        """Construct a representation of the lens with image-space refractive
        index `n` and angular semi-aperture `Theta`. Either `sin_Theta` or
        `Theta` has to be specified (the latter overrides the former).
        """
        if not (np.isscalar(n) and n >= 1.0):
            raise ValueError(
                "refractive index n must be scalar and not less " "than 1."
            )
        if not (sin_Theta > 0.0 and sin_Theta <= 1.0):
            raise ValueError("invalid value of sin_Theta.")

        self.n = n
        self.sqrt_n = math.sqrt(n)
        if Theta:
            self.Theta = Theta
            self.cos_Theta = math.cos(Theta)
            self.sin_Theta = math.sin(Theta)
        else:
            self.Theta = math.asin(sin_Theta)
            self.sin_Theta = sin_Theta
            self.cos_Theta = math.sqrt(1.0 - sin_Theta**2)

    def effective_fresnel_transmission_coeffs(
        self, cos_theta, sin_theta, cos_phi, sin_phi
    ):
        """Return local effective Fresnel transmission coefficients for the s
        and p polarization components.
        """
        Ts = 1.0
        Tp = 1.0
        return Ts, Tp

    def apodization_func(self, cos_theta, sin_theta):
        """Evaluate the apodization function of the lens."""
        return self.sqrt_n * self._a(cos_theta, sin_theta)

    def principal_surface_func(self, cos_theta, sin_theta):
        """
        Calculate the normalized pupil radius rho_ep (0 <= rho_ep <= 1) from
        cos(theta) and sin(theta).
        """
        return self._g(cos_theta, sin_theta) / self._g(
            self.cos_Theta, self.sin_Theta
        )

    def inv_principal_surface_func(self, rho_ep):
        """
        Calculate cos(theta) and sin(theta) for a given normalized pupil radius
        rho_ep (0 <= rho_ep <= 1).
        """
        return self._g_inv(rho_ep * self._g(self.cos_Theta, self.sin_Theta))

    @staticmethod
    def _a(cos_theta, sin_theta):
        raise NotImplementedError

    @staticmethod
    def _g(cos_theta, sin_theta):
        raise NotImplementedError

    @staticmethod
    def _g_inv(rho):
        raise NotImplementedError

    def transform_epf_to_pwa(
        self, epf, cos_theta, sin_theta, cos_phi, sin_phi
    ):
        """Compute the plane wave amplitudes from the entrance pupil field by
        vectorial ray tracing.
        """
        assert epf.shape[0] == 2

        # Short-hand notation
        ct, st, cp, sp = cos_theta, sin_theta, cos_phi, sin_phi
        cp2, sp2 = cp**2, sp**2

        Ts, Tp = self.effective_fresnel_transmission_coeffs(ct, st, cp, sp)

        M_xx = Tp * ct * cp2 + Ts * sp2
        M_xy = (Tp * ct - Ts) * sp * cp
        M_yx = M_xy
        M_yy = Tp * ct * sp2 + Ts * cp2
        M_zx = -Tp * st * cp
        M_zy = -Tp * st * sp

        M = np.array([[M_xx, M_xy], [M_yx, M_yy], [M_zx, M_zy]])
        M *= self.apodization_func(ct, st)
        pwa = -1j * np.einsum("ij...,j...->i...", M, epf, optimize=True)
        assert pwa.shape == ((3,) + cos_theta.shape)
        return pwa

    def transform_pwa_to_epf(
        self, pwa, cos_theta, sin_theta, cos_phi, sin_phi
    ):
        """Compute the entrance pupil field from the plane wave amplitudes by
        (reversed) vectorial ray tracing.
        """
        assert pwa.shape[0] == 3

        # Short-hand notation
        ct, st, cp, sp = cos_theta, sin_theta, cos_phi, sin_phi
        cp2, sp2 = cp**2, sp**2

        Ts, Tp = self.effective_fresnel_transmission_coeffs(ct, st, cp, sp)
        iTs, iTp = 1.0 / Ts, 1.0 / Tp

        M_xx = iTp * ct * cp2 + iTs * sp2
        M_xy = (iTp * ct - iTs) * sp * cp
        M_xz = -iTp * st * cp
        M_yx = M_xy
        M_yy = iTp * ct * sp2 + iTs * cp2
        M_yz = -iTp * st * sp

        M = np.array([[M_xx, M_xy, M_xz], [M_yx, M_yy, M_yz]])
        M /= self.apodization_func(ct, st)
        epf = 1j * np.einsum("ij...,j...->i...", M, pwa, optimize=True)
        assert epf.shape == ((2,) + cos_theta.shape)
        return epf

    def transform_ep_coords_to_pw_angles(self, rho_ep, cos_phi_ep, sin_phi_ep):
        """Transform normalized polar coordinates measured at the entrance
        pupil to corresponding plane wave direction angles (polar and
        azimuthal).
        """
        cos_theta, sin_theta = self.inv_principal_surface_func(rho_ep)
        cos_phi, sin_phi = -cos_phi_ep, -sin_phi_ep

        mask = abs(cos_theta - 1.0) < 4.0 * EPS
        cos_phi[mask] = 1.0
        sin_phi[mask] = 0.0

        return cos_theta, sin_theta, cos_phi, sin_phi

    def transform_pw_angles_to_ep_coords(
        self, cos_theta, sin_theta, cos_phi, sin_phi
    ):
        """Transform plane wave direction angles (polar and azimuthal) to
        corresponding normalized polar coordinates measured at the entrance
        pupil.
        """
        rho_ep = self.principal_surface_func(cos_theta, sin_theta)
        cos_phi_ep, sin_phi_ep = -cos_phi, -sin_phi

        mask = rho_ep < 4.0 * EPS
        cos_phi_ep[mask] = 1.0
        sin_phi_ep[mask] = 0.0

        return rho_ep, cos_phi_ep, sin_phi_ep

    def transform_epf_func_to_pwa_func(self, epf_func):
        """Construct the plane wave amplitude function from the entrance pupil
        field.
        """

        def pwa_func(*pw_angles):
            cos_theta = pw_angles[0]
            mask = cos_theta >= self.cos_Theta
            cap_pw_angles = [a[mask] for a in pw_angles]
            ep_coords = self.transform_pw_angles_to_ep_coords(*cap_pw_angles)
            epf = epf_func(*ep_coords)
            cap_pwa = self.transform_epf_to_pwa(epf, *cap_pw_angles)
            pwa = np.zeros((3,) + cos_theta.shape, dtype=np.complex128)
            pwa[:, mask] = cap_pwa
            return pwa

        return pwa_func

    def transform_pwa_func_to_epf_func(self, pwa_func):
        """Construct the entrance pupil field from the plane wave amplitude
        function.
        """

        def epf_func(*ep_coords):
            pw_angles = self.transform_ep_coords_to_pw_angles(*ep_coords)
            pwa = pwa_func(*pw_angles)
            epf = self.transform_pwa_to_epf(pwa, *pw_angles)
            return epf

        return epf_func


## Lenses


class AplanaticLens(Lens):
    """Model of an aplanatic lens satisfying Abbe's sine condition."""

    def __init__(self, n=1.0, sin_Theta=0.95, Theta=None):
        if sin_Theta == 1.0:
            # Integral substitution to deal with the diverging derivative of
            # sqrt(cos(theta)) @ cos(theta) = 0 is not implemented yet.
            raise NotImplementedError
        super().__init__(n, sin_Theta, Theta)

    @staticmethod
    def _a(cos_theta, sin_theta):
        return np.sqrt(cos_theta)

    @staticmethod
    def _g(cos_theta, sin_theta):
        return sin_theta

    @staticmethod
    def _g_inv(rho):
        sin_theta = rho
        assert (abs(sin_theta) <= 1.0).all()
        cos_theta = np.sqrt(1.0 - sin_theta**2)
        return cos_theta, sin_theta


class HerschelLens(Lens):
    """Model of a lens satisfying the Herschel condition."""

    def __init__(self, n=1.0, sin_Theta=0.95, Theta=None):
        super().__init__(n, sin_Theta, Theta)

    @staticmethod
    def _a(cos_theta, sin_theta):
        return np.ones_like(cos_theta)

    @staticmethod
    def _g(cos_theta, sin_theta):
        return 2.0 * np.sqrt(0.5 * (1.0 - cos_theta))  # = 2 * sin(theta/2)

    @staticmethod
    def _g_inv(rho):
        cos_theta = 1.0 - 0.5 * rho**2
        assert (abs(cos_theta) <= 1.0).all()
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        return cos_theta, sin_theta
