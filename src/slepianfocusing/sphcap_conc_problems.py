# sphcap_conc_problems.py

"""Representations of concentration problems within a spherical cap."""

__author__ = "Kornel JAHN (kornel.jahn@gmail.com)"
__copyright__ = "Copyright (c) 2012-2024 Kornel Jahn"
__license__ = "MIT"

import numpy as np
import math
from scipy.linalg import eig_banded, eigh

from .math_ext import log2_next_power_of_2
from .quadrature import GaussLegendreRule
from .assoc_legendre import (
    nalf_lowest_degree,
    nalf_function_count,
    nalf_recurr_factor_nu,
    nalf_P_vectorized,
)
from .sheppard_torok import (
    stf_lowest_degree,
    stf_function_count,
    stf_recurr_factor_zeta,
    stf_F_vectorized,
)

EPS = np.finfo(np.float64).eps


def _stable_sum_along_rows(A):
    sum_ = np.asarray([math.fsum(row) for row in A])
    return sum_


class _ConcentrationProblem:
    """Abstract base class for all scalar and vector concentration problems
    within an axisymmetrical spherical cap.
    """

    def __init__(self, maximal_degree, orders, half_apex_angle):
        self.maximal_degree = L = maximal_degree
        assert L >= 1
        if isinstance(orders, str) and orders == "all":
            self.orders = np.arange(-L, L + 1)
        else:
            # Convert orders to set to ensure each order occurs only once
            self.orders = np.array(sorted(set(np.atleast_1d(orders))))
        assert self.orders[0] >= -L and self.orders[-1] <= L
        self.half_apex_angle = half_apex_angle
        self.cos_half_apex_angle = cos_Theta = math.cos(half_apex_angle)
        assert cos_Theta > -1.0 and cos_Theta < 1.0
        self._calculate_shannon_number()

        self.partial_shannon_numbers = {}
        self.eigenvectors = {}
        self.energy_conc_ratios = {}
        self.energy_leakage_ratios = {}
        self.cdo_matrix_eigenvalues = {}
        for m in self.orders:
            self._solve_variational_problem(m)
            self._calculate_conc_properties(m)

    def _get_quadrature_nodes_weights(self, a, b):
        L = self.maximal_degree
        rule = GaussLegendreRule(9)
        # rule = GaussLegendreRule(log2_next_power_of_2(max(64, 4 * L)))
        x, w, wn = rule.get_nodes_weights(a, b)
        return x, w, wn

    @staticmethod
    def _calculate_partial_shannon_number(eta_arr):
        # Reverse array for greater accuracy in summation
        NS_m = int(round(eta_arr[::-1].sum()))
        return NS_m

    def get_all_energy_conc_ratios(self, with_orders=False):
        eta = []
        orders = []
        for m in self.energy_conc_ratios.keys():
            eta.append(self.energy_conc_ratios[m])
            orders.append(np.ones_like(eta[-1]) * m)
        eta = np.hstack(eta)
        orders = np.hstack(orders)
        idx_array = np.argsort(eta)[::-1]
        eta = eta[idx_array]
        orders = orders[idx_array]
        if with_orders:
            return eta, orders
        else:
            return eta

    def _solve_variational_problem(self, m):
        L = self.maximal_degree
        lmin = self._basis_lowest_degree_func(m)
        G_m = self._construct_cdo_matrix(m)
        # print(G_m)
        chi, g = eig_banded(G_m)
        assert np.allclose(np.dot(g, g.T), np.eye(g.shape[0]), rtol=1e-12)
        self.cdo_matrix_eigenvalues[m] = chi
        self.eigenvectors[m] = np.zeros((g.shape[1], L + 1), dtype=np.float64)
        self.eigenvectors[m][:, lmin:] = g.T
        # Fix signs
        row_count = self.eigenvectors[m].shape[0]
        for i in range(row_count):
            s = 1.0 if self.eigenvectors[m][i, lmin] >= 0.0 else -1.0
            self.eigenvectors[m][i, :] *= s

    def _calculate_conc_properties(self, m):
        L, Theta = self.maximal_degree, self.half_apex_angle
        lmin = self._basis_lowest_degree_func(m)
        g = self.eigenvectors[m]

        theta_in, w_in, wn_in = self._get_quadrature_nodes_weights(0.0, Theta)
        cos_theta_in, sin_theta_in = np.cos(theta_in), np.sin(theta_in)
        func_basis_in = self._basis_functions_vectorized(
            L, m, cos_theta_in, sin_theta_in
        )

        theta_out, w_out, wn_out = self._get_quadrature_nodes_weights(
            Theta, np.pi
        )
        cos_theta_out, sin_theta_out = np.cos(theta_out), np.sin(theta_out)
        func_basis_out = self._basis_functions_vectorized(
            L, m, cos_theta_out, sin_theta_out
        )

        Gmj_in = np.dot(g, func_basis_in)
        Gmj_out = np.dot(g, func_basis_out)

        eta = _stable_sum_along_rows(w_in * Gmj_in**2 * sin_theta_in)
        eta_check = _stable_sum_along_rows(wn_in * Gmj_in**2 * sin_theta_in)
        if not np.allclose(eta, eta_check, rtol=0, atol=1e5 * EPS):
            print(eta)
            print(eta_check)
            raise RuntimeError("quadrature level is too low.")
        del eta_check
        one_minus_eta = _stable_sum_along_rows(
            w_out * Gmj_out**2 * sin_theta_out
        )
        one_minus_eta_check = _stable_sum_along_rows(
            wn_out * Gmj_out**2 * sin_theta_out
        )
        if not np.allclose(
            one_minus_eta, one_minus_eta_check, rtol=0, atol=1e5 * EPS
        ):
            print(one_minus_eta)
            print(one_minus_eta_check)
            raise RuntimeError("quadrature level is too low.")
        del one_minus_eta_check

        # Numerical correction
        # Estimate the partial Shannon number
        NS_m = self._calculate_partial_shannon_number(eta)
        # Correct the energy concentration/leakage ratios
        eta[:NS_m] = 1.0 - one_minus_eta[:NS_m]
        one_minus_eta[NS_m:] = 1.0 - eta[NS_m:]

        self.energy_conc_ratios[m] = eta
        self.energy_leakage_ratios[m] = one_minus_eta
        # Refine the estimate of the partial Shannon number
        NS_m = self._calculate_partial_shannon_number(eta)
        self.partial_shannon_numbers[m] = NS_m

    def get_all_cdo_matrix_eigenvalues(self):
        chi = np.hstack(self.cdo_matrix_eigenvalues.values())
        chi = np.sort(chi)
        return chi


class ScalarConcProblem(_ConcentrationProblem):
    def __init__(self, maximal_degree, orders, half_apex_angle):
        self._basis_functions_vectorized = nalf_P_vectorized
        self._basis_lowest_degree_func = nalf_lowest_degree
        super().__init__(maximal_degree, orders, half_apex_angle)

    def _calculate_shannon_number(self):
        L, cos_Theta = self.maximal_degree, self.cos_half_apex_angle
        NS = int(round((L + 1) ** 2 * 0.5 * (1.0 - cos_Theta)))
        self.shannon_number = NS

    def _construct_cdo_matrix(self, m, dense=False):
        """Construct the symmetric tridiagonal matrix $G$ corresponding to
        $P_{lm}$.

        One may obtain the matrix $G$ in the dense form
            [ G_{11} G_{12}      0      0 ]
            [ G_{21} G_{22} G_{23}      0 ]
            [      0 G_{32} G_{33} G_{34} ]
            [      0      0 G_{43} G_{44} ]
        or in the compact "upper" form
            [      0 G_{12} G_{23} G_{34} ]
            [ G_{11} G_{22} G_{33} G_{44} ]
        storing the bands only.
        """
        L, cos_Theta = self.maximal_degree, self.cos_half_apex_angle
        lmin = nalf_lowest_degree(m)
        l = np.arange(lmin, L + 1)
        nu_lp1_m = nalf_recurr_factor_nu(l + 1, m)
        diag = -l * (l + 1) * cos_Theta
        off_diag = (l * (l + 2) - L * (L + 2)) * nu_lp1_m
        if dense:
            G = np.diag(diag)
            G += np.diag(off_diag[:-1], 1) + np.diag(off_diag[:-1], -1)
        else:
            G = np.vstack((np.hstack((0.0, off_diag[:-1])), diag))
        return G


class VectorRelatedScalarConcProblem(_ConcentrationProblem):
    def __init__(self, maximal_degree, orders, half_apex_angle):
        self._basis_functions_vectorized = stf_F_vectorized
        self._basis_lowest_degree_func = stf_lowest_degree
        super().__init__(maximal_degree, orders, half_apex_angle)

    def _calculate_shannon_number(self):
        L, cos_Theta = self.maximal_degree, self.cos_half_apex_angle
        NS = int(round(L * (L + 2) * 0.5 * (1.0 - cos_Theta)))
        self.shannon_number = NS

    def _construct_cdo_matrix(self, m, dense=False):
        """Construct the symmetric tridiagonal matrix $J$ corresponding to
        $F_{lm}$.

        One may obtain the matrix $J$ in the dense form
            [ J_{11} J_{12}      0      0 ]
            [ J_{21} J_{22} J_{23}      0 ]
            [      0 J_{32} J_{33} J_{34} ]
            [      0      0 J_{43} J_{44} ]
        or in the compact "upper" form
            [      0 J_{12} J_{23} J_{34} ]
            [ J_{11} J_{22} J_{33} J_{44} ]
        storing the bands only.
        """
        L, cos_Theta = self.maximal_degree, self.cos_half_apex_angle
        lmin = stf_lowest_degree(m)
        l = np.arange(lmin, L + 1)
        zeta_lp1_m = stf_recurr_factor_zeta(l + 1, m)
        diag = -l * (l + 1) * cos_Theta + m * (
            1.0 - (L * (L + 2) + 1.0) / (l * (l + 1))
        )
        off_diag = (l * (l + 2) - L * (L + 2)) * zeta_lp1_m
        if dense:
            J = np.diag(diag)
            J += np.diag(off_diag[:-1], 1) + np.diag(off_diag[:-1], -1)
        else:
            J = np.vstack((np.hstack((0.0, off_diag[:-1])), diag))
        return J


class TangentialVectorConcProblem(_ConcentrationProblem):
    def __init__(self, maximal_degree, orders, half_apex_angle):
        if isinstance(orders, str) and orders == "all":
            scalar_orders = orders
        else:
            orders = np.atleast_1d(orders)
            orders = np.hstack([orders, -orders])
            scalar_orders = np.array(sorted(set(orders)))

        self._scalar_problem = VectorRelatedScalarConcProblem(
            maximal_degree, scalar_orders, half_apex_angle
        )

        super().__init__(maximal_degree, orders, half_apex_angle)

    def _calculate_shannon_number(self):
        L, cos_Theta = self.maximal_degree, self.cos_half_apex_angle
        self.shannon_number = 2 * self._scalar_problem.shannon_number

    def _solve_variational_problem(self, m):
        L = self.maximal_degree
        # Duplicate eigenvalues
        for m in self.orders:
            chi_m = self._scalar_problem.cdo_matrix_eigenvalues[m]
            chi_minus_m = self._scalar_problem.cdo_matrix_eigenvalues[-m]
            chi_vector = np.hstack([chi_m, chi_minus_m])

            eta_m = self._scalar_problem.energy_conc_ratios[m]
            eta_minus_m = self._scalar_problem.energy_conc_ratios[-m]
            eta_m_vector = np.hstack([eta_m, eta_minus_m])

            one_minus_eta_m = self._scalar_problem.energy_leakage_ratios[m]
            one_minus_eta_minus_m = self._scalar_problem.energy_leakage_ratios[
                -m
            ]
            one_minus_eta_m_vector = np.hstack(
                [one_minus_eta_m, one_minus_eta_minus_m]
            )

            g_plus = self._scalar_problem.eigenvectors[m]
            g_plus = np.hstack([g_plus, np.zeros_like(g_plus)])
            g_minus = self._scalar_problem.eigenvectors[-m]
            g_minus = np.hstack([np.zeros_like(g_minus), g_minus])
            g_vector = np.vstack([g_plus, g_minus])

            idx_array = np.argsort(chi_vector)
            self.cdo_matrix_eigenvalues[m] = chi_vector[idx_array]
            self.eigenvectors[m] = g_vector[idx_array]
            self.scalar_eigenvalues = self._scalar_problem.energy_conc_ratios
            self.scalar_eigenvectors = self._scalar_problem.eigenvectors
            self.energy_conc_ratios[m] = eta_m_vector[idx_array]
            self.energy_leakage_ratios[m] = one_minus_eta_m_vector[idx_array]

    def _calculate_conc_properties(self, m):
        eta = self.energy_conc_ratios[m]
        one_minus_eta = self.energy_leakage_ratios[m]

        # Numerical correction
        # Estimate the partial Shannon number
        NS_m = self._calculate_partial_shannon_number(eta)
        # Correct the energy concentration/leakage ratios
        eta[:NS_m] = 1.0 - one_minus_eta[:NS_m]
        one_minus_eta[NS_m:] = 1.0 - eta[NS_m:]

        self.energy_conc_ratios[m] = eta
        self.energy_leakage_ratios[m] = one_minus_eta
        # Refine the estimate of the partial Shannon number
        NS_m = self._calculate_partial_shannon_number(eta)
        self.partial_shannon_numbers[m] = NS_m
