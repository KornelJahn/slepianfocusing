# forward.py

"""Example of forward focal field calculation.

Plane-wave (complex) amplitudes of the incident beam are calculated and then
expanded using vector Slepian harmonics. The focal field in the focal plane is
calculated both using the corresponding vector Slepian multipole field
expansion and Debye--Wolf integration.

Convergence of the expansion is examined by increasing the number of terms
involved. The optical intensity profiles of both the plane-wave amplitude
expansion and the focal field expansion is visualized.

The calculations are based on the theory described in Sections 4.1 to 4.3 of
the thesis.
"""

__author__ = "Kornel JAHN (kornel.jahn@gmail.com)"
__copyright__ = "Copyright (c) 2012-2024 Kornel Jahn"
__license__ = "MIT"

import functools
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as tck
import pathlib

from slepianfocusing.quadrature import GaussLegendreRule
from slepianfocusing.lenses import AplanaticLens
from slepianfocusing.sphcap_conc_problems import VectorRelatedScalarConcProblem
from slepianfocusing.coord_transforms import (
    cartesian2d_to_polar2d,
    cartesian3d_to_spherical3d,
)
from slepianfocusing.vector_spherical_harmonics import (
    vsh_series_vectorized,
    vsh_Q_coeffs_cap,
)
from slepianfocusing.vector_multipole_fields import vmf_series_vectorized
from slepianfocusing.incident_beams import (
    gaussian_amplitude,
    linear_y_polarization,
    left_circular_polarization,
    hg_radially_polarized_beam,
)
from slepianfocusing.focal_field_direct_int import (
    calculate_focal_field_direct_int,
)

# Machine epsilon value for the IEEE double-precision floating point type
EPS = np.finfo(np.float64).eps
SQRT_EPS = np.sqrt(EPS)

FILENAME = pathlib.Path(__file__).stem


def main():
    """Main (top-level) function of the example."""

    ### Input parameter definitions

    # Selection of entrance pupil incident beam profiles (field distributions),
    # as described in Subsection 4.3.2 of the thesis.
    descriptions = {
        'lcp': 'Homogeneous beam with left circular polarization',
        'lpy': 'Gaussian beam with linear (y) polarization',
        'rad': 'Radially polarized combination of Hermite--Gaussian beams',
    }
    # Incident beam entrance pupil fields, as defined in Eqs. (4.15); numerical
    # values for the Gaussian waist radii of beams 'lpy' and 'rad' have been
    # chosen in a way that approximately 90% of the total power of the incident
    # beams fall over the entrance pupil.
    incident_beams = {
        'lcp':  left_circular_polarization,
        'lpy':  lambda *args: (
            gaussian_amplitude(0.93198120356931215, *args)
            * linear_y_polarization(*args)
        ),
        'rad':  lambda *args: (
            hg_radially_polarized_beam(0.71706053099546160, *args)
        ),
    }
    # Pre-determined orders `m` of non-vanishing terms in the vector Slepian
    # harmonics expansion of the plane-wave amplitudes. See Section 4.1 on how
    # to determine them and Eqs. (4.3) for a "cheat sheet". The values are
    # pairs of `(p, m)` where `p` is the polarization index (+/-1) and `m` the
    # order.
    orders = {
        'lcp': [(1, 1)],
        'lpy': [(1, 1), (-1, -1)],
        'rad': [(1, 0), (-1, 0)],
    }
    # Maximal degree L of the expansion
    L = 20
    # Angular semiaperture
    Theta = np.pi / 3  # 60 degrees (NA = 0.866 in air)

    ### Initialization

    configure_matplotlib()
    computation_tasks = {
        'conv': compute_convergence_data,
        'pwa': compute_plane_wave_amplitudes,
        'ff': compute_focal_field,
    }
    data = {}

    ### Computation (results are cached in .npz files)

    for case in descriptions.keys():
        epf_func = incident_beams[case]  # Entrance pupil field function
        m_values = orders[case]
        for task_key, task_func in computation_tasks.items():
            label = case + str(L) + task_key
            npz_filename = FILENAME + '_' + label + '.npz'
            print(f'Searching for {npz_filename}...')
            try:
                data[label] = np.load(npz_filename)
            except FileNotFoundError:
                # If .npz file is not found, compute the results and store them
                lens = AplanaticLens(Theta=Theta)
                # FIXME: naming
                TODO = make_funcs(L, Theta, m_values, epf_func)
                data[label] = task_func(L, Theta, m_values, TODO)
                np.savez(npz_filename, **data[label])

    ### Visualization

    plot(data)


def configure_matplotlib():
    """Set up common plot settings."""
    # plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif='cm', size=2 * 10)
    plt.rc('pdf', use14corefonts=True) # Use Type 1 fonts
    plt.rc('legend', fontsize='small')
    plt.rc('axes', labelsize='medium')
    plt.rc('xtick', direction='out', labelsize='medium')
    plt.rc('xtick.major', pad=5, size=3)
    plt.rc('ytick', direction='out', labelsize='medium')
    plt.rc('ytick.major', pad=5, size=3)
    plt.rc('grid', linestyle=':')


def compute_convergence_data(Theta, m_values, epf_func):
    """Compute data to study the convergence of vector Slepian harmonics
    expansions.
    """
    Ns = result['Ns']
    eta_values = result['eta_values']
    Nm = result['Nm']
    pwa_func = result['pwa_func']
    pwa_slepian_series_func = result['pwa_slepian_series_func']
    ff_slepian_series_func = result['ff_slepian_series_func']

    print('Computation of the reference plane-wave amplitudes...')
    pw_int = PolarPlaneWaveDirIntegrator(Theta, 8)
    pw_angles = pw_int.nodes
    pwa_ref = pwa_func(*pw_angles)
    norm2_pwa_ref = pw_int.integrate(intensity(pwa_ref))

    print('Computation of the reference focal field (Debye--Wolf)...')
    fp_int = FocalPlaneIntegrator(10.0, 8)
    ff_coords_cart = fp_int.nodes
    ff_coords_sph = cartesian3d_to_spherical3d(*ff_coords_cart)
    ff_ref = calculate_focal_field_direct_int(epf_func, lens, 7, 8,
                                              *ff_coords_cart)
    norm2_ff_ref = fp_int.integrate(intensity(ff_ref))

    print('Computation of the approximation errors...')
    pwa_errors_slepian = []
    ff_errors_slepian = []
    for num in range(1, Nm + 1):
        print('Evaluation of the error for %d terms...' % num)
        pwa_slepian_inner = pwa_slepian_series_func(*pw_angles, num=num)
        pwa_slepian_error = pw_int.integrate(intensity(pwa_slepian_inner -
                                                       pwa_ref))
        pwa_errors_slepian.append(pwa_slepian_error / norm2_pwa_ref)

        ff_slepian_inner = ff_slepian_series_func(*ff_coords_sph, num=num)
        ff_slepian_error = fp_int.integrate(intensity(ff_slepian_inner -
                                                      ff_ref))
        ff_errors_slepian.append(ff_slepian_error / norm2_ff_ref)
    pwa_errors_slepian = np.array(pwa_errors_slepian, dtype=np.float64)
    ff_errors_slepian = np.array(ff_errors_slepian, dtype=np.float64)

    data = {}
    data['Ns'] = Ns
    data['Nm'] = Nm
    data['eta_values'] = eta_values
    data['pwa_errors_slepian'] = pwa_errors_slepian
    data['ff_errors_slepian'] = ff_errors_slepian
    return data




def intensity(A):
    """Calculate the optical intensity of an array-like `A` of vectorial
    complex amplitude values.
    """
    return np.sum(A * A.conj(), 0).real


class PolarPlaneWaveDirIntegrator:
    """Helper class for numerical integration in the `theta` direction of the
    unit sphere (plane-wave amplitude domain).

    Furthermore, note that this integration assumes discontinuity, i.e. a
    breakpoint, in the `theta` direction at the angular semiaperture `Theta`,
    as dictated by the cut-off occuring in the Debye--Wolf theory of focusing.
    Piecewise integration is therefore performed.

    Gauss--Legendre rule of given level is used and the integration error is
    also estimated. Absolute and relative tolerances of sqrt(epsilon) are
    enforced.
    """

    def __init__(self, Theta, theta_level):
        rule1 = GaussLegendreRule(theta_level).get_nodes_weights(0.0, Theta)
        theta1, w1, w_error1 = rule1
        rule2 = GaussLegendreRule(theta_level).get_nodes_weights(Theta, np.pi)
        theta2, w2, w_error2 = rule2
        theta = np.hstack([theta1, theta2])
        w = np.hstack([w1, w2])
        w_error = np.hstack([w_error1, w_error2])
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        cos_phi, sin_phi = np.ones_like(theta), np.zeros_like(theta)
        self.nodes = cos_theta, sin_theta, cos_phi, sin_phi
        self.weights = w
        self.weights_error = w_error
        self.jacobian = sin_theta

    def integrate(self, funcvals):
        integrand = funcvals * self.jacobian
        I = np.sum(self.weights * integrand)
        I_error = np.sum(self.weights_error * integrand)
        if abs(I - I_error) > max(SQRT_EPS, SQRT_EPS * abs(I)):
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


def make_funcs(L, Theta, orders, epf_func):
    """TODO
    """
    # Assume we only need positive `m` values for the given input functions
    assert min(p * m for p, m in orders) == 0
    M = max(p * m for p, m in orders)
    lens = AplanaticLens(Theta=Theta)
    # Transform the function describing the field at the entrance pupil to
    # another function yielding vectorial plane-wave amplitudes
    pwa_func = lens.transform_epf_func_to_pwa_func(epf_func)

    vrscp = VectorRelatedScalarConcProblem(L, np.arange(0, M + 1), Theta)
    Ns = vrscp.partial_shannon_numbers[M]
    eta_values = vrscp.energy_conc_ratios[M]
    Nm = len(eta_values)

    coeffs = {}
    for p in (1, -1):
        temp = vsh_Q_coeffs_cap(pwa_func, lens.Theta, p, L, M)
        coeffs.update({(p, m): c for m, c in temp.items()})
    coeffs = {k: coeffs[k] for k in orders}

    def pwa_slepian_series_func(*pwa_angles, num=Nm):
        result = 0
        for (p, m), c in coeffs.items():
            T = vrscp.eigenvectors[p * m]
            slepian_coeffs = np.dot(T, c)
            slepian_coeffs[num:] = 0.0
            new_c = np.dot(T.T, slepian_coeffs)
            result += vsh_series_vectorized(p, m, new_c, *pwa_angles)
        return result

    def ff_slepian_series_func(*ff_coords, num=Nm):
        result = 0
        for (p, m), c in coeffs.items():
            T = vrscp.eigenvectors[p * m]
            slepian_coeffs = np.dot(T, c)
            slepian_coeffs[num:] = 0.0
            new_c = np.dot(T.T, slepian_coeffs)
            result += vmf_series_vectorized(p, m, new_c, *ff_coords)
        return result

    return dict(
        # TODO
        Ns=Ns,
        # TODO
        Nm=Nm,
        # TODO
        eta_values=eta_values,
        # TODO
        pwa_func=pwa_func,
        # TODO
        pwa_slepian_series_func=pwa_slepian_series_func,
        # TODO
        ff_slepian_series_func=ff_slepian_series_func,
    )




def compute_plane_wave_amplitudes(L, Theta, m_values, epf_func):
    lens = AplanaticLens(Theta=Theta)
    result = make_funcs(L, Theta, m_values, epf_func)
    Ns = result['Ns']
    pwa_func = result['pwa_func']
    pwa_slepian_series_func = result['pwa_slepian_series_func']

    num = Ns + 2

    theta = np.linspace(0, np.pi/2, 256)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.ones_like(theta)
    sin_phi = np.zeros_like(theta)
    pwa_exact = pwa_func(cos_theta, sin_theta, cos_phi, sin_phi)
    pwa_exact[:, theta > Theta] = 0.0
    pwa_slepian = pwa_slepian_series_func(cos_theta, sin_theta, cos_phi,
                                          sin_phi, num=num)

    data = {}
    data['Theta'] = Theta
    data['theta'] = theta
    data['pwa_exact'] = pwa_exact
    data['pwa_slepian'] = pwa_slepian
    return data


def compute_focal_field(L, Theta, m_values, epf_func):
    lens = AplanaticLens(Theta=Theta)
    result = make_funcs(L, Theta, m_values, epf_func)
    Ns = result['Ns']
    ff_slepian_series_func = result['ff_slepian_series_func']

    num = Ns + 2

    zmax = 20.0
    ymax = 10.0

    y2, z2 = np.meshgrid(np.linspace(-ymax, ymax, 127),
                         np.linspace(-zmax, zmax, 255))
    x2 = np.zeros_like(y2)
    ff_coords = cartesian3d_to_spherical3d(x2, y2, z2)
    ff_merid_exact = calculate_focal_field_direct_int(epf_func, lens, 9, 10,
                                                      x2, y2, z2, tol=1e5)
    ff_merid_slepian = ff_slepian_series_func(*ff_coords, num=num)

    data = {}
    data['ff_merid_extent'] = np.array([-zmax, zmax, -ymax, ymax])
    data['ff_merid_exact'] = np.array(ff_merid_exact, dtype=np.complex128)
    data['ff_merid_slepian'] = np.array(ff_merid_slepian, dtype=np.complex128)
    return data


#==============================================================================

def plot_conv(data):
    def custom_log10_formatter(x, pos):
        return '$10^{%d}$' % int(math.log10(x)) if x != 1 else '$1$'

    cases = ['pwa', 'ff']
    log10_ymin_values = [-3, -6]


    figsize = (0.9 * TEXTWIDTH, 0.5 * TEXTWIDTH)
    for case, log10_ymin in zip(cases, log10_ymin_values):
        fig = plt.figure(figsize=figsize)

        row_count = len(names)
        col_count = len(max_degrees) + 1

        wr = [3] * col_count
        wr[0] = 1

        gs = gridspec.GridSpec(row_count, col_count,
                               width_ratios=wr,
                               left=0.02, right=0.99, bottom=0.105, top=0.92,
                               wspace=0.15, hspace=0.35)

        for row_idx, name in enumerate(names):
            ax = fig.add_subplot(gs[row_idx, 0], frameon=False)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='both', which='both', bottom='off',
                           left='off', top='off', right='off',
                           labelbottom='off', labelright='off')
            ax.text(-1.1, 0, "\\textbf{(%s)}" % 'abc'[row_idx],
                    ha='center', va='center')
            for col_idx, L in enumerate(max_degrees):
                label = name + str(L) + 'conv'
                data_ = data[label]
                Ns = data_['Ns']
                Nm = data_['Nm']
                eta_values = data_['eta_values']
                errors = data_[case + '_errors_slepian']
                #print(label, errors[-1])

                ax = fig.add_subplot(gs[row_idx, col_idx + 1])
                ax.axhline(errors[-1], lw=1.5, ls='-', c='0.7')
                ax.axvline(Ns, lw=1, ls='--', c='k')
                ax.semilogy(np.arange(1, len(errors) + 1), errors, 'ko',
                            markersize=3)

                ax.tick_params(axis='both', which='both', bottom='on',
                               left='on', top='on', right='on')

                ax.set_ylim(10**log10_ymin, 1.0)
                if case == 'pwa':
                    ax.set_yticks([10.0**k for k in range(log10_ymin, 1)])
                else:
                    ax.set_yticks([10.0**k for k in range(log10_ymin, 1, 2)])

                # Hack to make minor ticks work properly
                minorticks = []
                for e in range(log10_ymin, 0):
                    minorticks = np.hstack([minorticks,
                                            np.arange(2, 10) * 10.0**e])
                ax.yaxis.set_ticks(minorticks, minor=True)

                ax.yaxis.set_major_formatter(tck.FuncFormatter(
                    custom_log10_formatter))

                ax.xaxis.set_major_locator(tck.FixedLocator([1, Ns, Nm]))
                #axt = ax.twiny()
                #axt.set_xlim(1, Nm)
                #axt.xaxis.set_major_locator(tck.FixedLocator([1, Ns, Nm]))
                #axt.set_xticklabels([
                #    '',
                #    "$N'_{\\mathrm{S},%d}$" % (0 if name == 'rad' else 1),
                #    ''
                #])

                if row_idx == row_count - 1:
                    ax.set_xlabel(r'$j_\mathrm{max}$')

                if col_idx == 0:
                    if case == 'pwa':
                        ax.set_ylabel(r'$\tilde{\epsilon}(j_\mathrm{max})$')
                    else:
                        ax.set_ylabel(r'$\epsilon(j_\mathrm{max})$')
                else:
                    ax.set_yticklabels([])

                if row_idx == 0:
                    ax.set_title('$L = %d$' % L, y=1.1)

                #print(eta_values[Ns-1], eta_values[Ns])
                #np.set_printoptions(precision=17)
                #print(errors)

        plt.savefig(FILENAME + '_conv_' + case + '.pdf')
        plt.close()


def plot_pwa(data):
    figsize = (TEXTWIDTH, 0.7 * TEXTWIDTH)
    fig = plt.figure(figsize=figsize)

    row_count = len(names)
    col_count = len(max_degrees) + 1

    wr = [3] * col_count
    wr[0] = 1

    gs = gridspec.GridSpec(row_count, col_count,
                            width_ratios=wr,
                            left=0.02, right=0.98, bottom=0.08, top=0.95,
                            wspace=0.15, hspace=0.15)

    ypad = 0.05

    for row_idx, name in enumerate(names):
        ax = fig.add_subplot(gs[row_idx, 0], frameon=False)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', bottom='off',
                        left='off', top='off', right='off',
                        labelbottom='off', labelright='off')
        ax.text(-1.1, 0, "\\textbf{(%s)}" % 'abc'[row_idx],
                ha='center', va='center')

        ymax = 0.0
        for col_idx, L in enumerate(max_degrees):
            label = name + str(L) + 'pwa'
            data_ = data[label]
            I_pwa_e = intensity(data_['pwa_exact'])
            norm = I_pwa_e.max()
            I_pwa_e /= norm
            I_pwa_s = intensity(data_['pwa_slepian']) / norm
            ymax = max(ymax, I_pwa_e.max(), I_pwa_s.max())

        for col_idx, L in enumerate(max_degrees):
            label = name + str(L) + 'pwa'
            data_ = data[label]

            Theta_deg = np.degrees(data_['Theta'])
            theta_deg = np.degrees(data_['theta'])
            I_pwa_e = intensity(data_['pwa_exact'])
            norm = I_pwa_e.max()
            I_pwa_e /= norm
            I_pwa_s = intensity(data_['pwa_slepian']) / norm

            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            ax.axvline(Theta_deg, c='0.7', ls='--')
            l1, = ax.plot(theta_deg, I_pwa_e, 'b--', label='exact')
            l2, = ax.plot(theta_deg, I_pwa_s, 'k-', label='approx.')
            ax.legend(loc=0)

            ax.set_xlim(0.0, 90.0)
            ax.set_ylim(0.0 - ypad * ymax, ymax + ypad * ymax)

            ax.tick_params(axis='both', which='both', bottom='on',
                            left='on', top='on', right='on')

            ax.xaxis.set_major_locator(tck.FixedLocator([0, 30, 60, 90]))
            #axt = ax.twiny()
            #axt.set_xlim(0.0, 90.0)
            #axt.set_ylim(0.0 - ypad * ymax, ymax + ypad * ymax)
            #axt.xaxis.set_major_locator(tck.FixedLocator([0, 30, 60, 90]))

            #if row_idx == 0:
            #    axt.set_xticklabels(['', '', '$\Theta$', ''])
            #else:
            #    axt.set_xticklabels([])

            if row_idx == row_count - 1:
                ax.set_xlabel(r'$\theta_\mathrm{s}\ (^\circ)$')
            else:
                ax.set_xticklabels([])

            if col_idx == 0:
                sup = {'lcp': 'LCP', 'lpy': 'LPy', 'rad': 'rad'}
                ax.set_ylabel(r'$\lvert \tilde{\mathbf{E}}^\mathrm{' +
                              sup[name] + r'}'
                              r'(\theta_\mathrm{s}, 0)'
                              r'\rvert^2\ \text{(a.u.)}$',
                              labelpad=12)
            else:
                ax.set_yticklabels([])

            if row_idx == 0:
                ax.set_title('$L = %d$' % L, y=1.05)

            #print(eta_values[Ns-1], eta_values[Ns])
            #np.set_printoptions(precision=17)
            #print(errors)

    plt.savefig(FILENAME + '_pwa.pdf')
    plt.close()


def plot_ff(data):
    figsize = (TEXTWIDTH, 0.7 * TEXTWIDTH)
    fig = plt.figure(figsize=figsize)

    row_count = len(names) + 1
    col_count = len(max_degrees) + 1

    wr = [3] * col_count
    wr[0] = 1
    #hr = [10] * row_count
    #hr[-2] = 1
    #hr[-1] = 4
    hr = [3] * row_count
    hr[-1] = 1

    gs = gridspec.GridSpec(row_count, col_count,
                            width_ratios=wr, height_ratios=hr,
                            left=0.02, right=0.97, bottom=0.11, top=0.94,
                            wspace=0.17, hspace=0.15)

    ypad = 0.05
    vmin, vmax = -9, -2
    for row_idx, name in enumerate(names):
        ax = fig.add_subplot(gs[row_idx, 0], frameon=False)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', bottom='off',
                        left='off', top='off', right='off',
                        labelbottom='off', labelright='off')
        ax.text(-0.8, 0, "\\textbf{(%s)}" % 'abc'[row_idx],
                ha='center', va='center')

        for col_idx, L in enumerate(max_degrees):
            label = name + str(L) + 'ff'
            data_ = data[label]

            extent = data_['ff_merid_extent']
            ff_e = data_['ff_merid_exact']
            ff_s = data_['ff_merid_slepian']

            log_I_error = np.log10(intensity(ff_s - ff_e) /
                                   intensity(ff_e).max())

            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            print(log_I_error.max())
            cset = ax.imshow(log_I_error, vmin=vmin, vmax=vmax, cmap='inferno',
                             interpolation='bilinear', extent=extent)

            ax.tick_params(axis='both', which='both', bottom='on',
                            left='on', top='on', right='on')

            if row_idx == row_count - 2:
                ax.set_xlabel(r'$z / \lambda$')
            else:
                ax.set_xticklabels([])

            if col_idx == 0:
                ax.set_ylabel(r'$y / \lambda$')
            else:
                ax.set_yticklabels([])

            if row_idx == 0:
                ax.set_title('$L = %d$' % L, y=1.05)

    cax = fig.add_subplot(gs[-1, 1:])
    pos1 = cax.get_position() # get the original position
    pos2 = [pos1.x0, pos1.y0, pos1.width, 0.2 * pos1.height]
    cax.set_position(pos2) # set a new position
    clabel = (r'$\lvert \mathbf{E}^\mathrm{Slep}(\mathbf{r}) - \mathbf{E}(\mathbf{r}) \rvert^2 / '
              r'\max \lvert \mathbf{E}(\mathbf{r}) \rvert^2$')
    cbar = fig.colorbar(cset, cax=cax, orientation='horizontal', extend='min')
    cbar.set_label(clabel, labelpad=12)
    cbar.set_ticks(range(vmin, vmax + 1))
    cax.set_xticklabels(['$10^{%d}$'% e for e in range(vmin, vmax + 1)])

    plt.savefig(FILENAME + '_ff.pdf')
    plt.close()




def plot(data):
    plot_conv(data)
    plot_pwa(data)
    plot_ff(data)



if __name__ == '__main__':
    main()

# Note that normally, to calculate inner products over the  2D integration
# would be needed over the whole `0 <= theta < pi`, `0 <= phi < 2*pi` domain of
# the unit sphere, as prescribed in Eq. (3.30). However, we use this integrator
# to calculate inner products of normalized fields which are harmonic in the
# azimuthal (phi) direction. Therefore, azimuthal integration of such inner
# products can either yield 0 or 1. We already only calculate non-vanishing
# inner products of specific mixed vector spherical/Slepian harmonics orders
# `m`. Therefore inner products occuring in our calculations reduce to
# one-dimensional integrals in the `theta` direction.

