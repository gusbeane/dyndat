import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import warnings
from copy import copy
from pyia import GaiaData
import astropy.coordinates as coord
import astropy.units as u
import gala.dynamics as gd

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

with warnings.catch_warnings(record=True):
    warnings.simplefilter("ignore")
    APOKASCfile = 'GAIA-APOKASC2_actions.fits'
    APOKASCtable = fits.open(APOKASCfile)
    APOKASCtable = Table(APOKASCtable[1].data)
    STfile = 'GAIA-ST_actions.fits'
    STtable = fits.open(STfile)
    STtable = Table(STtable[1].data)

MgFe = APOKASCtable['afe']
Fe = APOKASCtable['feh']
Jr = APOKASCtable['Jr']
Lz = np.abs(APOKASCtable['Lz'])
Jz = APOKASCtable['Jz']
age = 10.**APOKASCtable['log_age']/1000.0
uvel = APOKASCtable['uvel']
vvel = APOKASCtable['vvel']
wvel = APOKASCtable['wvel']

STJr = STtable['Jr']
STLz = np.abs(STtable['Lz'])
STJz = STtable['Jz']
STage = STtable['age_mean']
STuvel = STtable['uvel']
STvvel = STtable['vvel']
STwvel = STtable['wvel']
STFe = STtable['feh']
STMgFe = STtable['[MgI]'] - STtable['feh']

STgd = GaiaData(STtable)

gaiadata = GaiaData(APOKASCtable)

rsun = 8 * u.kpc
zsun = 0.025 * u.kpc
vsun = [11.1, 232.24, 7.25] * u.km/u.s
gc_frame = coord.Galactocentric(galcen_distance=rsun, galcen_v_sun=coord.CartesianDifferential(*vsun), z_sun=zsun)

sc = gaiadata.skycoord
STsc = STgd.skycoord
APOKASCdyn = gd.PhaseSpacePosition(sc.transform_to(gc_frame).cartesian)
STdyn = gd.PhaseSpacePosition(STsc.transform_to(gc_frame).cartesian)

APOKASCrad = APOKASCdyn.represent_as('cylindrical').rho.to(u.kpc).value
STrad = STdyn.represent_as('cylindrical').rho.to(u.kpc).value

APOKASCz = APOKASCdyn.represent_as('cylindrical').z.to(u.kpc).value
STz = STdyn.represent_as('cylindrical').z.to(u.kpc).value

#process
Jr = np.sqrt(Jr)
Lz = Lz/(8. * 220.)
vvel = vvel - 220.

STJr = np.sqrt(STJr)
STLz = STLz/(8. * 220.)
STvvel = STvvel - 220.

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=9)

"""
Jr vs. age
"""
fig = plt.figure(figsize=(7.1,6))
axarr = fig.subplots(2,2)

cm = copy(plt.cm.winter_r)
cm.set_bad(color='white', alpha=0.5)

myrange = [[-150,150],[-125,75]]

heatmap, xedges, yedges = np.histogram2d(uvel, vvel, bins=35, range=myrange)
heatmap.T[heatmap.T == 0] = np.nan
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]
axarr[0][0].imshow(heatmap.T, extent=extent, origin='lower', cmap=cm, vmin=0, vmax=80)

axarr[0][0].scatter(STuvel, STvvel,s=6, c='none', lw=0.8, edgecolor='black')

axarr[0][0].set(xlabel=r'$U_{\text{LSR}}\,[\,\text{km}/\text{s}\,]$')
axarr[0][0].set(ylabel=r'$V_{\text{LSR}}\,[\,\text{km}/\text{s}\,]$')
axarr[0][0].set(xlim=myrange[0])
axarr[0][0].set(ylim=myrange[1])
#axarr[0][0].xaxis.set_ticks(range(15), minor=True)

"""
Lz vs. age
"""
myrange = [[0.5,1.3],[0,16]]

heatmap, xedges, yedges = np.histogram2d(Lz, Jr, bins=35, range=myrange)
heatmap.T[heatmap.T == 0] = np.nan
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]
im = axarr[0][1].imshow(heatmap.T, extent=extent, origin='lower', cmap=cm, vmin=0, vmax=80)

axarr[0][1].scatter(STLz, STJr,s=6, c='none', lw=0.8, edgecolor='black')

axarr[0][1].set(xlabel=r'$L_z\,[\,8\,\text{kpc}\,220\,\text{km}/\text{s}\,]$')
axarr[0][1].set(ylabel=r'$\sqrt{J_r}\,[\,\text{kpc}\,\text{km}/\text{s}\,]^{1/2}$')
axarr[0][1].set(xlim=myrange[0])
axarr[0][1].set(ylim=myrange[1])


myrange = [[-1.,0.5],[-0.1,0.3]]

heatmap, xedges, yedges = np.histogram2d(Fe, MgFe, bins=40, range=myrange)
heatmap.T[heatmap.T == 0] = np.nan
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]
im =  axarr[1][1].imshow(heatmap.T, extent=extent, origin='lower', cmap=cm, vmin=0, vmax=80)

agekeys = np.where(np.logical_and(age >7, age < 8))[0]
scatterx = Fe[agekeys]
scattery = MgFe[agekeys]
#axarr[1][1].scatter(scatterx, scattery, s=6, c='cyan')
axarr[1][1].scatter(STFe, STMgFe, s=6, c='none', lw=0.8, edgecolor='black')

axarr[1][1].set(xlabel=r'$[\text{Fe}/\text{H}]$')
axarr[1][1].set(ylabel=r'$[\alpha/\text{Fe}]$')
axarr[1][1].set(xlim=myrange[0])
axarr[1][1].set(ylim=myrange[1])
#ax[0].set_aspect(1.25)

myrange = [[7.5,8.12],[-0.12,1.2]]

heatmap, xedges, yedges = np.histogram2d(APOKASCrad, APOKASCz, bins=40, range=myrange)
heatmap.T[heatmap.T == 0] = np.nan
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]
im =  axarr[1][0].imshow(heatmap.T, extent=extent, origin='lower', cmap=cm, vmin=0, vmax=80)

axarr[1][0].scatter(STrad, STz, s=6, c='none', lw=0.8, edgecolor='black')

axarr[1][0].set(xlabel=r'$R\,[\,\text{kpc}\,]$')
axarr[1][0].set(ylabel=r'$z\,[\,\text{kpc}\,]$')
axarr[1][0].set(xlim=myrange[0])
axarr[1][0].set(ylim=myrange[1])



for ax in axarr:
    ax[0].axis('auto')
    ax[1].axis('auto')
plt.tight_layout()
#cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7], frameon=False)
cb = fig.colorbar(im, ax=axarr.ravel().tolist(), orientation='horizontal', fraction=0.02, pad=0.1)
cb.set_label('number')
plt.savefig('uv_JrLz_MgFe.pdf')
