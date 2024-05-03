# inverse.py

import math
import numpy as np
import os, sys
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as tck
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from scipy.special import lambertw

sys.path.append(os.path.join('.', 'slepianfocusing'))

from slepianfocusing.quadrature import GaussLegendreRule
from slepianfocusing.lenses import AplanaticLens
from slepianfocusing.sphcap_conc_problems import VectorRelatedScalarConcProblem
from slepianfocusing.coord_transforms import (
    cartesian2d_to_polar2d,
    cartesian3d_to_spherical3d,
)
from slepianfocusing.vector_spherical_harmonics import (
    vsh_series_vectorized,
    vsh_coeffs_YZ_to_Qpm,
)
from slepianfocusing.vector_multipole_fields import vmf_series_vectorized
from slepianfocusing.focal_field_direct_int import (
    calculate_focal_field_direct_int,
)
#from slepianfocusing.focal_field_czt import (
#    calculate_focal_field_czt,
#)

mm = 0.0393701
TEXTWIDTH = 2 * 155.0 * mm

base = os.path.basename(__file__)
FILENAME = os.path.splitext(base)[0]

EPS = np.finfo(np.float64).eps
SQRT_EPS = np.sqrt(EPS)

L = 30
Theta = math.asin(0.95)

coeffs = {
    'needle': {
        'rad': np.array([
                -0.0019417978306726536,
                0.009719312236961932,
                -0.03505300506668949,
                0.07040000421369326,
                -0.09615979126119716,
                0.11181934960058307,
                -0.19934402684141564,
                0.30089088102285205,
                -0.4397254436384754,
                -0.5234881624557329,
                -0.9378505070315907
        ]),
        'azi': np.zeros((11,), dtype=np.float64),
    },
    'tube': {
        'rad': np.zeros((11,), dtype=np.float64),
        'azi': np.array([
                0.01330449174633379,
                0.19479199041856893,
                1.185294492116033,
                1.1423206757824849,
                0.2668508498469438,
                -0.8917496457312647,
                0.14562036958440755,
                0.24101747807559648,
                0.08446772481564867,
                -0.30743552841186017,
                0.14527354395631226
        ]),
    },
    'bubble': {
        'azi': np.array([
                -0.10509112439835276,
                -0.3007957343431141,
                -0.308830369007004,
                -0.47728457824740955,
                -0.39900821929028907,
                -0.266429229249927,
                -0.2943602385520328,
                -0.1952097443095488,
                -0.10432271565213452,
                -0.041517605569856134,
                -0.04198674814954057,
        ]),
        'rad': np.array([
                0.17778647726184352,
                0.4751142994761531,
                0.3494964954115787,
                0.5079720186513149,
                -0.06100592054628026,
                0.07742281303813048,
                -0.44597891053923844,
                -0.033879747237165685,                    
                -0.4561996087728384,
                0.05067208831673662,
                -0.3286884391738718
        ]),
    },
}

FWHM_y = 0.4
FWHM_z = 10.0
y0 = FWHM_y / 2 / math.log(2.0)**0.5
z0 = FWHM_z / 2 / math.log(2.0)**0.1
y1 = FWHM_y / 2 / math.sqrt(-lambertw(-0.5/np.e).real)
z1 = 3 * y1

def needle(x, y, z):
    # FWHM_y = 2 rho0 log(2)**(1/2)
    # FWHM_z = 2 z0 log(2)**(1/10)
    return np.exp(-(y / y0)**2 - (z / z0)**10)

def tube(x, y, z):
    # FWHM_y = 2 y0 sqrt(-lambertW(-0.5/np.e))
    # FWHM_z = see needle
    t = (y / y1)**2
    return t * np.exp(-t - (z / z0)**10 + 1.0)

def bubble(x, y, z):
    # FWHM_y = 2 y0 log(2)**(1/2)
    # FWHM_z = 2 z0 log(2)**(1/10)
    t = (y / y1)**2 + (z / z1)**2
    return t * np.exp(-t + 1)

prescriptions = dict(needle=needle, tube=tube, bubble=bubble)


def disable_ticks_and_ticklabels(ax, axis='xy', no_ticks=True,
        no_ticklabels=True):
    axis_list = []
    if 'x' in axis:
        axis_list.append(ax.get_xaxis())
    if 'y' in axis:
        axis_list.append(ax.get_yaxis())
    for a in axis_list:
        if no_ticklabels:
            for t in a.get_ticklabels():
                t.set_visible(False)
            a.get_offset_text().set_visible(False)
        if no_ticks:
            for t in a.get_ticklines():
                t.set_visible(False)
                

def intensity(A):
    return np.sum(A * A.conj(), 0).real


def make_funcs(name):
    lens = AplanaticLens(Theta=Theta)
    vrscp = VectorRelatedScalarConcProblem(L, 0, Theta)
    num = 11
        
    coeffs_slepian_rad = np.hstack([coeffs[name]['rad'],
                                    np.zeros((19,), dtype=np.complex128)])
    coeffs_slepian_azi = np.hstack([coeffs[name]['azi'],
                                    np.zeros((19,), dtype=np.complex128)])
   
    T = vrscp.eigenvectors[0]
    coeffs_Y = np.dot(T.T, coeffs_slepian_azi)
    coeffs_Z = np.dot(T.T, coeffs_slepian_rad)
    coeffs_Qp, coeffs_Qm = vsh_coeffs_YZ_to_Qpm(0, coeffs_Y, coeffs_Z)
    
    def pwa_slepian_series_func(*pwa_angles):
        result = (vsh_series_vectorized(1, 0, coeffs_Qp, *pwa_angles) + 
                  vsh_series_vectorized(-1, 0, coeffs_Qm, *pwa_angles))
        return result
        
    def ff_slepian_series_func(*ff_coords):
        result = (vmf_series_vectorized(1, 0, coeffs_Qp, *ff_coords) + 
                  vmf_series_vectorized(-1, 0, coeffs_Qm, *ff_coords))
        return result

    return pwa_slepian_series_func, ff_slepian_series_func


def compute_pwa(name):
    from operator import xor
    
    lens = AplanaticLens(Theta=Theta)
    pwa_func, ff_func = make_funcs(name)
    
    M = 1000
    ran = np.linspace(-1, 1, M)
    x_ep, y_ep = np.meshgrid(ran, ran)
    mask = x_ep**2 + y_ep**2 <= 1.0
    x_ep = np.where(mask, x_ep, 0.0)
    y_ep = np.where(mask, y_ep, 0.0)
    ep_coords = cartesian2d_to_polar2d(x_ep, y_ep)
    pw_angles = lens.transform_ep_coords_to_pw_angles(*ep_coords)
    E_pwa = pwa_func(*pw_angles)
    E_ep = lens.transform_pwa_to_epf(E_pwa, *pw_angles)
    I_ep_2d = intensity(E_ep)
    I_ep_2d[~mask] = np.nan
    
    assert(xor(np.allclose(E_ep.real, 0.0), np.allclose(E_ep.imag, 0.0)))
    #if np.allclose(E_ep.real, 0.0):
    #    E_ep = (-1j * E_ep).real
    #else:
    #    E_ep = E_ep.real
    
    rho, cos_phi, sin_phi = ep_coords
    rad_uv = np.array([cos_phi, sin_phi])
    abs_E_ep = np.sqrt(intensity(E_ep))
    assert(np.allclose(np.sum(rad_uv * E_ep, 0).imag, 0.0))
    cos_alpha = np.sum(rad_uv * E_ep, 0).real / abs_E_ep
    cos_alpha = np.where(cos_alpha >= -1.0, cos_alpha, -1.0)
    cos_alpha = np.where(cos_alpha <= 1.0, cos_alpha, 1.0)
    #sign = np.where((rad_uv[0] * E_ep[1] - rad_uv[1] * E_ep[0]) >= 0.0, 1, -1)
    #alpha_deg_2d = sign * np.degrees(np.arccos(cos_alpha))
    assert(np.allclose((rad_uv[0] * E_ep[1] - rad_uv[1] * E_ep[0]).imag, 0.0))
    sin_alpha = (rad_uv[0] * E_ep[1] - rad_uv[1] * E_ep[0]).real / abs_E_ep
    sin_alpha = np.where(sin_alpha >= -1.0, sin_alpha, -1.0)
    sin_alpha = np.where(sin_alpha <= 1.0, sin_alpha, 1.0)
    alpha_deg_2d = np.degrees(np.arctan2(sin_alpha, cos_alpha))
    #if name == 'needle':
    #    np.set_printoptions(precision=2, linewidth=160)
    #    print(alpha_deg_2d[400:430, 400:430].astype(int))
    alpha_deg_2d[~mask] = np.nan
    
    
    M = 512
    y_ep = np.linspace(0.0, 1.0, M)[1:-1]
    x_ep = np.zeros_like(y_ep)
    ep_coords = cartesian2d_to_polar2d(x_ep, y_ep)
    pw_angles = lens.transform_ep_coords_to_pw_angles(*ep_coords)
    E_pwa = pwa_func(*pw_angles)
    E_ep = lens.transform_pwa_to_epf(E_pwa, *pw_angles)
    I_ep_1d = intensity(E_ep)
    
    assert(np.allclose(E_ep.real, 0.0) or np.allclose(E_ep.imag, 0.0))
    #if np.allclose(E_ep.real, 0.0):
    #    E_ep = (-1j * E_ep).real
    #else:
    #    E_ep = E_ep.real
    
    rho, cos_phi, sin_phi = ep_coords
    rad_uv = np.array([cos_phi, sin_phi])
    abs_E_ep = np.sqrt(intensity(E_ep))
    cos_alpha = np.sum(rad_uv * E_ep, 0).real / abs_E_ep
    cos_alpha = np.where(cos_alpha >= -1.0, cos_alpha, -1.0)
    cos_alpha = np.where(cos_alpha <= 1.0, cos_alpha, 1.0)
    #sign = np.where((rad_uv[0] * E_ep[1] - rad_uv[1] * E_ep[0]) >= 0.0, 1, -1)
    #alpha_deg_1d = sign * np.degrees(np.arccos(cos_alpha))
    sin_alpha = (rad_uv[0] * E_ep[1] - rad_uv[1] * E_ep[0]).real / abs_E_ep
    sin_alpha = np.where(sin_alpha >= -1.0, sin_alpha, -1.0)
    sin_alpha = np.where(sin_alpha <= 1.0, sin_alpha, 1.0)
    alpha_deg_1d = np.degrees(np.arctan2(sin_alpha, cos_alpha))

    data = dict(
        y=y_ep,
        I_2d=I_ep_2d,
        alpha_deg_2d=alpha_deg_2d,
        I_1d=I_ep_1d,
        alpha_deg_1d=alpha_deg_1d,
    )
    return data
    
    
def compute_ff(name):
    y0_values = {'needle': 0.0, 'tube': 0.48, 'bubble': 0.0}
    pwa_func, ff_func = make_funcs(name)
    
    ymax = 4.0
    zmax = 7.0
    
    y1 = np.linspace(-ymax, ymax, 201)
    z1 = np.linspace(-zmax, zmax, 351)
    
    z_mp, y_mp = np.meshgrid(z1, y1)
    x_mp = np.zeros_like(y_mp)
    mp_coords_cart = x_mp, y_mp, z_mp
    mp_coords_sph = cartesian3d_to_spherical3d(*mp_coords_cart)
    E_mp = ff_func(*mp_coords_sph)
    
    x_fp, y_fp = np.meshgrid(y1, y1)
    z_fp = np.zeros_like(y_fp)
    fp_coords_cart = x_fp, y_fp, z_fp
    fp_coords_sph = cartesian3d_to_spherical3d(*fp_coords_cart)
    E_fp = ff_func(*fp_coords_sph)
    
    y1 = np.linspace(0.0, ymax, 200)
    x1 = np.zeros_like(y1)
    z1 = np.zeros_like(y1)
    ycs_coords_cart = x1, y1, z1
    ycs_coords_sph = cartesian3d_to_spherical3d(*ycs_coords_cart)
    I_ycs_pr = prescriptions[name](*ycs_coords_cart)
    E_ycs_sl = ff_func(*ycs_coords_sph)
    E_ycs_dw = calculate_focal_field_direct_int(pwa_func, Theta, 7, 8,
                                                *ycs_coords_cart)
    ycs = y1
    
    z1 = np.linspace(0.0, zmax, 350)
    y1 = np.ones_like(z1) * y0_values[name]
    x1 = np.zeros_like(y1)
    zcs_coords_cart = x1, y1, z1
    zcs_coords_sph = cartesian3d_to_spherical3d(*zcs_coords_cart)
    I_zcs_pr = prescriptions[name](*zcs_coords_cart)
    E_zcs_sl = ff_func(*zcs_coords_sph)
    E_zcs_dw = calculate_focal_field_direct_int(pwa_func, Theta, 7, 8,
                                                *zcs_coords_cart)
    zcs = z1
            
    data = dict(
        fp_extent=(-ymax, ymax, -ymax, ymax),
        mp_extent=(-zmax, zmax, -ymax, ymax),
        E_mp=E_mp,
        E_fp=E_fp,
        ycs=ycs,
        I_ycs_pr=I_ycs_pr,
        E_ycs_sl=E_ycs_sl,
        E_ycs_dw=E_ycs_dw,
        zcs=zcs,
        I_zcs_pr=I_zcs_pr,
        E_zcs_sl=E_zcs_sl,
        E_zcs_dw=E_zcs_dw,
    )
    return data


def _calculate_inset_placement(ax, p_abs_inch):
    fig = ax.figure
    figw, figh = fig.get_figwidth(), fig.get_figheight()
    offset = ax.get_position().get_points()
    p_abs_inch = np.asarray(p_abs_inch)
    p_rel = p_abs_inch / np.array((figw, figh, figw, figh))
    p_rel[0:2] += offset[0]
    return p_rel.tolist()


def plot_ff(data):
    figsize = (TEXTWIDTH, 0.9 * TEXTWIDTH)        
    fig = plt.figure(figsize=figsize)
    
    row_count = 4
    col_count = 3
    
    wr = [1, 5, 7]
    hr = [4] * row_count
    hr[-1] = 1
    
    gs = gridspec.GridSpec(row_count, col_count,
                           width_ratios=wr,
                           height_ratios=hr,
                           left=0.02, right=0.97, bottom=0.08, top=0.97,
                           wspace=0.17, hspace=0.15)
    
    inset_size_inch = 1.4
    
    for row_idx, name in enumerate(['needle', 'tube', 'bubble']):
        data_ = data[name]

        ax = fig.add_subplot(gs[row_idx, 0], frameon=False)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', bottom='off',
                        left='off', top='off', right='off', 
                        labelbottom='off', labelright='off')
        ax.text(-1.0, 0, "\\textbf{(%s)}" % 'abc'[row_idx], 
                ha='center', va='center')
        
        I_mp = intensity(data_['E_mp'])
        I_fp = intensity(data_['E_fp'])
        #I_fp /= I_mp.max()
        #I_mp /= I_mp.max()
        
        extent = data_['fp_extent']
        ycs = data_['ycs']
        I_cs_pr = data_['I_ycs_pr']
        I_cs_sl = intensity(data_['E_ycs_sl'])
        I_cs_dw = intensity(data_['E_ycs_dw'])
               
        ax = fig.add_subplot(gs[row_idx, 1])
        
        ax.plot(ycs, I_cs_pr, 'r:')
        ax.plot(ycs, I_cs_dw, 'b--')
        ax.plot(ycs, I_cs_sl, 'k-')
        sup = {'needle': 'N', 'tube': 'T', 'bubble': 'B'}[name]
        ax.set_ylabel(r'$\lvert \mathbf{E}^\mathrm{' + sup +
                      r'} \rvert^2\ \text{(a.u.)}$',
                      labelpad=12)
        ax.set_xlim(ycs.min(), ycs.max())
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticks(np.arange(0.0, ycs.max() + 1))
        
        if row_idx == row_count - 2:
            ax.set_xlabel(r'$\rho / \lambda$')
        else:
            ax.set_xticklabels([])
        
        iax = inset_axes(ax, width=inset_size_inch, height=inset_size_inch,
                         loc=1, borderpad=0)
        disable_ticks_and_ticklabels(iax)
        iax.imshow(I_fp, vmin=0.0, vmax=1.0, cmap='inferno', 
                   interpolation='bilinear', extent=extent)
                      
        extent = data_['mp_extent']
        zcs = data_['zcs']
        I_cs_pr = data_['I_zcs_pr']
        I_cs_sl = intensity(data_['E_zcs_sl'])
        I_cs_dw = intensity(data_['E_zcs_dw'])
               
        ax = fig.add_subplot(gs[row_idx, 2])
        
        Y = I_cs_dw / I_cs_dw.max()
        Ydiff = np.abs(Y - 0.5)
        D = np.vstack([zcs, Ydiff]).T
        np.savetxt('/home/korn/' + name + '.txt', D)
        
        ax.plot(zcs, I_cs_pr, 'r:')
        ax.plot(zcs, I_cs_dw, 'b--')
        ax.plot(zcs, I_cs_sl, 'k-')
        ax.set_xlim(zcs.min(), zcs.max())
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticks(np.arange(0.0, zcs.max() + 1))
        ax.set_yticklabels([])
        
        if row_idx == row_count - 2:
            ax.set_xlabel(r'$z / \lambda$')
        else:
            ax.set_xticklabels([])
        
        loc = 1 if name == 'bubble' else 3
        iax = inset_axes(ax, width=7 * inset_size_inch / 4,
                         height=inset_size_inch, loc=loc, borderpad=0)
        disable_ticks_and_ticklabels(iax)
        cset = iax.imshow(I_mp, vmin=0.0, vmax=1.0, cmap='inferno', 
                          interpolation='bilinear', extent=extent)
    
    cax = fig.add_subplot(gs[-1, 1:])
    pos1 = cax.get_position() # get the original position 
    pos2 = [pos1.x0, pos1.y0, pos1.width, 0.2 * pos1.height] 
    cax.set_position(pos2) # set a new position
    clabel = r'$\lvert \mathbf{E} \rvert^2\ \text{(a.u.)}$'
    cbar = fig.colorbar(cset, cax=cax, orientation='horizontal')
    cbar.set_label(clabel, labelpad=12)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        
    #plt.savefig(FILENAME + '_ff.pdf')
    #plt.close()


def import_phase_cmap():
    rgb = np.loadtxt('phase-rgb.txt')
    cmap = LinearSegmentedColormap.from_list('phase', rgb, N=256)
    return cmap
   

def plot_pwa(data):
    phase_cmap = import_phase_cmap()
    
    figsize = (TEXTWIDTH, TEXTWIDTH)        
    fig = plt.figure(figsize=figsize)
    
    row_count = 4
    col_count = 3
    
    wr = [1, 8, 8]
    hr = [5] * row_count
    hr[-1] = 1
    
    gs = gridspec.GridSpec(row_count, col_count,
                           width_ratios=wr, height_ratios=hr,
                           left=0.02, right=0.97, bottom=0.07, top=0.98,
                           wspace=0.5, hspace=0.15)
    
    inset1_size_inch = 1.4
    inset2_size_inch = 0.9
    
    for row_idx, name in enumerate(['needle', 'tube', 'bubble']):
        data_ = data[name]

        ax = fig.add_subplot(gs[row_idx, 0], frameon=False)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', bottom='off',
                        left='off', top='off', right='off', 
                        labelbottom='off', labelright='off')
        ax.text(-1.0, 0, "\\textbf{(%s)}" % 'abc'[row_idx], 
                ha='center', va='center')
        
        y = data_['y']
        
        I_2d = data_['I_2d']
        I_2d /= np.nanmax(I_2d)
        I_1d = data_['I_1d']
        I_1d /= np.nanmax(I_1d)
                               
        ax = fig.add_subplot(gs[row_idx, 1])
        
        ax.plot(y, I_1d, 'k-')
        sup = {'needle': 'N', 'tube': 'T', 'bubble': 'B'}[name]
        ax.set_ylabel(r'$\lvert \mathbf{\mathcal{E}}_\mathrm{EP}^\mathrm{' +
                      sup + r'} \rvert^2\ '
                      r'\text{(a.u.)}$',
                      labelpad=12)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(True)
        
        if row_idx == row_count - 2:
            ax.set_xlabel(r'$\rho_\mathrm{e} / R_\mathrm{e}$')
        else:
            ax.set_xticklabels([])
        
        loc = {'needle': 9, 'tube': 2, 'bubble': 1}
        iax = inset_axes(ax, width=inset1_size_inch, height=inset1_size_inch,
                         loc=loc[name], borderpad=0.2)
        iax.axis('off')
        disable_ticks_and_ticklabels(iax)
        iax.add_artist(plt.Circle((0, 0), 1.01, color='k', clip_on=False))
        cset1 = iax.imshow(I_2d, vmin=0.0, vmax=1.0, cmap='inferno', 
                           interpolation='bilinear', extent=(-1, 1, -1, 1),
                           zorder=10)

        alpha_deg_2d = data_['alpha_deg_2d']
        alpha_deg_1d = data_['alpha_deg_1d']
               
        ax = fig.add_subplot(gs[row_idx, 2])
        
        ax.plot(y, alpha_deg_1d, 'k-')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-270, 270)
        ax.set_yticks([-180, -90, 0, 90, 180])
        
        ax.grid(True)
        ax.set_ylabel(r'$\alpha^\mathrm{' + sup + r'}\ (^\circ)$')
        if row_idx == row_count - 2:
            ax.set_xlabel(r'$\rho_\mathrm{e} / R_\mathrm{e}$')
        else:
            ax.set_xticklabels([])
        
        loc = {'needle': 1, 'tube': 3, 'bubble': 1}
        iax = inset_axes(ax, width=inset2_size_inch, height=inset2_size_inch,
                         loc=loc[name], borderpad=0.2)
        iax.axis('off')
        disable_ticks_and_ticklabels(iax)
        #iax.add_artist(plt.Circle((0, 0), 1.01, color='k', clip_on=False))
        cset2 = iax.imshow(alpha_deg_2d, vmin=-180, vmax=180, cmap=phase_cmap, 
                           interpolation='nearest', extent=(-1, 1, -1, 1),
                           zorder=10)
    
    cax = fig.add_subplot(gs[-1, 1])
    pos1 = cax.get_position() # get the original position 
    pos2 = [pos1.x0, pos1.y0, pos1.width, 0.2 * pos1.height] 
    cax.set_position(pos2) # set a new position
    clabel = (r'$\lvert \mathbf{\mathcal{E}}_\mathrm{EP} \rvert^2\ '
              r'\text{(a.u.)}$')
    cbar = fig.colorbar(cset1, cax=cax, orientation='horizontal')
    cbar.set_label(clabel, labelpad=12)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    
    cax = fig.add_subplot(gs[-1, 2])
    pos1 = cax.get_position() # get the original position 
    pos2 = [pos1.x0, pos1.y0, pos1.width, 0.2 * pos1.height] 
    cax.set_position(pos2) # set a new position
    clabel = r'$\alpha\ (^\circ)$'
    cbar = fig.colorbar(cset2, cax=cax, orientation='horizontal')
    cbar.set_label(clabel, labelpad=12)
    cbar.set_ticks([-180, -90, 0, 90, 180])
    
    plt.savefig(FILENAME + '_pwa.pdf', dpi=600)
    plt.close()


def plot_coeffs():
    figsize = (TEXTWIDTH, 0.3 * TEXTWIDTH)        
    fig = plt.figure(figsize=figsize)
    
    idx = np.arange(1, 12)
    
    ax = fig.add_subplot(1, 4, 1)
    c = coeffs['needle']['rad']**2
    ax.bar(idx, c, width=0.6, color='k', edgecolor='k', align='center')
    ax.set_ylabel(r'$(c_j^\mathrm{N})^2$')
    ax.set_xlabel(r'$j$')
    ax.set_title(r'\textbf{(a)}', y=1.1)
    ax.set_ylim(0, 1.0)
    
    ax = fig.add_subplot(1, 4, 2)
    c = coeffs['tube']['azi']**2
    ax.bar(idx, c, width=0.6, color='k', edgecolor='k', align='center')
    ax.set_ylabel(r'$(d_j^\mathrm{T})^2$')
    ax.set_xlabel(r'$j$')
    ax.set_title(r'\textbf{(b)}', y=1.1)
    ax.set_ylim(0, 1.5)
    
    ax = fig.add_subplot(1, 4, 3)
    c = coeffs['bubble']['rad']**2
    ax.bar(idx, c, width=0.6, color='k', edgecolor='k', align='center')
    ax.set_ylabel(r'$(c_j^\mathrm{B})^2$')
    ax.set_xlabel(r'$j$')
    ax.set_title(r'\textbf{(c)}', y=1.1)
    ax.set_ylim(0, 0.3)
    
    ax = fig.add_subplot(1, 4, 4)
    c = coeffs['bubble']['azi']**2
    ax.bar(idx, c, width=0.6, color='k', edgecolor='k', align='center')
    ax.set_ylabel(r'$(d_j^\mathrm{B})^2$')
    ax.set_xlabel(r'$j$')
    ax.set_title(r'\textbf{(d)}', y=1.1)
    ax.set_ylim(0, 0.3)
    
    plt.tight_layout()
    plt.savefig(FILENAME + '_coeffs.pdf')
    plt.close()
    

if __name__ == '__main__':
    _latex_preamble = r'\usepackage{amsmath}'
    
    # Configure Matplotlib
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=_latex_preamble)
    plt.rc('font', family='serif', serif='cm', size=2 * 10)
    plt.rc('pdf', use14corefonts=True) # Use Type 1 fonts
    plt.rc('legend', fontsize='small')
    plt.rc('axes', labelsize='medium')
    plt.rc('xtick', direction='out', labelsize='medium')
    plt.rc('xtick.major', pad=5, size=3)
    plt.rc('ytick', direction='out', labelsize='medium')
    plt.rc('ytick.major', pad=5, size=3)
    plt.rc('grid', linestyle=':')

    data = {'needle': {}, 'tube': {}, 'bubble': {}}
    for name in ['needle', 'tube', 'bubble']:
        npz_filename = FILENAME + '_' + name + '.npz'
        print('Searching for %s...' % npz_filename)
        try:
            data[name].update(np.load(npz_filename))
        except:
            data[name].update(compute_ff(name))
            data[name].update(compute_pwa(name))
            np.savez(npz_filename, **data[name])
    #plot_pwa(data)
    plot_ff(data)
    #plot_coeffs()
    
   
