import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import warnings
from copy import copy
from scipy import stats as st

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

STJr = STtable['Jr']
STLz = np.abs(STtable['Lz'])
STJz = STtable['Jz']
STFe = STtable['feh']
STMgFe = STtable['[MgI]'] - STtable['feh']
STage = STtable['age_mean']

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=9)

Jr = np.sqrt(Jr)
STJr = np.sqrt(STJr)
Jz = np.sqrt(Jz)
STJz = np.sqrt(STJz)

Jrbool = np.logical_not(np.isnan(Jr))
Jzbool = np.logical_not(np.isnan(Jz))
nankeys = np.where(np.logical_and(Jrbool, Jzbool))[0]
age = age[nankeys]
Jr = Jr[nankeys]
Lz = Lz[nankeys]
Jz = Jz[nankeys]
MgFe = MgFe[nankeys]
Fe = Fe[nankeys]

scut = 20
Jrlcut = st.sigmaclip(Jr, low=scut, high=scut)[1]
Jrhcut = st.sigmaclip(Jr, low=scut, high=scut)[2]
Lzlcut = st.sigmaclip(Lz, low=scut, high=scut)[1]
Lzhcut = st.sigmaclip(Lz, low=scut, high=scut)[2]
Jzlcut = st.sigmaclip(Jz, low=scut, high=scut)[1]
Jzhcut = st.sigmaclip(Jz, low=scut, high=scut)[2]

Jrcutkeys = np.logical_and(Jr > Jrlcut, Jr < Jrhcut)
Lzcutkeys = np.logical_and(Lz > Lzlcut, Lz < Lzhcut)
Jzcutkeys = np.logical_and(Jz > Jzlcut, Jz < Jzhcut)

Jzcutkeys = np.logical_and(Jzcutkeys, Jz > 0)

sigcutkeys = np.logical_and(Jrcutkeys, Lzcutkeys)
sigcutkeys = np.logical_and(sigcutkeys, Jzcutkeys)
sigcutkeys = np.where(sigcutkeys)[0]

age = age[sigcutkeys]
Jr = Jr[sigcutkeys]
Lz = Lz[sigcutkeys]
Jz = Jz[sigcutkeys]
MgFe = MgFe[sigcutkeys]
Fe = Fe[sigcutkeys]

"""
Jr vs. age
"""
fig = plt.figure(figsize=(7.1,4))
axarr = fig.subplots(2,3)

cm = copy(plt.cm.winter_r)
cm.set_bad(color='white', alpha=0.5)

myrange = [[-0.8,0.8],[0,20]]
heatmap, xedges, yedges = np.histogram2d(Fe[np.where(np.logical_not(np.isnan(Jr)))], Jr[np.where(np.logical_not(np.isnan(Jr)))], bins=30, range=myrange)
heatmap.T[heatmap.T == 0] = np.nan
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]
axarr[0][0].imshow(heatmap.T, extent=extent, origin='lower', cmap=cm, vmin=0, vmax=80)

axarr[0][0].scatter(STFe, STJr, s=6, c='none', lw=0.5, edgecolor='black')

axarr[0][0].set(xlabel=r'$[\text{Fe}/\text{H}]$')
axarr[0][0].set(ylabel=r'$\sqrt{J_r} [\,\text{kpc}\,\text{km}/\text{s}\,]^{1/2}$')
axarr[0][0].set(xlim=myrange[0])
axarr[0][0].set(ylim=myrange[1])
#axarr[0][0].set_aspect(0.5)
#axarr[0][0].xaxis.set_ticks(range(15), minor=True)

"""
Lz vs. age
"""
myrange=[[-0.8,0.8],[800,2000]]
heatmap, xedges, yedges = np.histogram2d(Fe[np.where(np.logical_not(np.isnan(Lz)))], Lz[np.where(np.logical_not(np.isnan(Lz)))], bins=30, range=myrange)
heatmap.T[heatmap.T == 0] = np.nan
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]

axarr[0][1].imshow(heatmap.T, extent=extent, origin='lower', aspect=0.01, cmap=cm, vmin=0, vmax=80)
axarr[0][1].scatter(STFe, STLz, s=6, c='none', lw=0.5, edgecolor='black')

axarr[0][1].set(xlabel=r'$[\text{Fe}/\text{H}]$')
axarr[0][1].set(ylabel=r'$L_z [\,\text{kpc}\,\text{km}/\text{s}\,]$')
axarr[0][1].set(xlim=myrange[0])
axarr[0][1].set(ylim=myrange[1])
#axarr[0][1].xaxis.set_ticks(range(15), minor=True)

"""
Jz vs. age
"""
myrange=[[-0.8,0.8],[0,20]]
heatmap, xedges, yedges = np.histogram2d(Fe[np.where(np.logical_not(np.isnan(Jz)))], Jz[np.where(np.logical_not(np.isnan(Jz)))], bins=30, range=myrange)
heatmap.T[heatmap.T == 0] = np.nan
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]

im = axarr[0][2].imshow(heatmap.T, extent=extent, origin='lower', cmap=cm, vmin=0, vmax=80)
axarr[0][2].scatter(STFe, STJz, s=6, c='none', lw=0.5, edgecolor='black')

axarr[0][2].set(xlabel=r'$[\text{Fe}/\text{H}]$')
axarr[0][2].set(ylabel=r'$\sqrt{J_z} [\,\text{kpc}\,\text{km}/\text{s}\,]^{1/2}$')
axarr[0][2].set(xlim=myrange[0])
axarr[0][2].set(ylim=myrange[1])
axarr[0][2].set_yticks([0,5,10,15,20])

myrange = [[-0.1,0.3],[0,20]]
heatmap, xedges, yedges = np.histogram2d(MgFe[np.where(np.logical_not(np.isnan(Jr)))], Jr[np.where(np.logical_not(np.isnan(Jr)))], bins=30, range=myrange)
heatmap.T[heatmap.T == 0] = np.nan
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]
axarr[1][0].imshow(heatmap.T, extent=extent, origin='lower', cmap=cm, vmin=0, vmax=80)

axarr[1][0].scatter(STMgFe, STJr, s=6, c='none', lw=0.5, edgecolor='black')

axarr[1][0].set(xlabel=r'$[\alpha/\text{Fe}]$')
axarr[1][0].set(ylabel=r'$\sqrt{J_r} [\,\text{kpc}\,\text{km}/\text{s}\,]^{1/2}$')
axarr[1][0].set(xlim=myrange[0])
axarr[1][0].set(ylim=myrange[1])
#axarr[0].set_aspect(0.5)
#axarr[0].xaxis.set_ticks(range(15), minor=True)

"""
Lz vs. age
"""
myrange=[[-0.1,0.3],[800,2000]]
heatmap, xedges, yedges = np.histogram2d(MgFe[np.where(np.logical_not(np.isnan(Lz)))], Lz[np.where(np.logical_not(np.isnan(Lz)))], bins=30, range=myrange)
heatmap.T[heatmap.T == 0] = np.nan
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]

axarr[1][1].imshow(heatmap.T, extent=extent, origin='lower', cmap=cm, vmin=0, vmax=80)
axarr[1][1].scatter(STMgFe, STLz, s=6, c='none', lw=0.5, edgecolor='black')

axarr[1][1].set(xlabel=r'$[\alpha/\text{Fe}]$')
axarr[1][1].set(ylabel=r'$L_z [\,\text{kpc}\,\text{km}/\text{s}\,]$')
axarr[1][1].set(xlim=myrange[0])
axarr[1][1].set(ylim=myrange[1])

"""
Jz vs. age
"""
myrange=[[-0.1,0.3],[0,20]]
heatmap, xedges, yedges = np.histogram2d(MgFe[np.where(np.logical_not(np.isnan(Jz)))], Jz[np.where(np.logical_not(np.isnan(Jz)))], bins=30, range=myrange)
heatmap.T[heatmap.T == 0] = np.nan
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]

im = axarr[1][2].imshow(heatmap.T, extent=extent, origin='lower', cmap=cm, vmin=0, vmax=80)
#axarr[1][2].scatter(MgFe[np.where(np.logical_not(np.isnan(Jz)))], Jz[np.where(np.logical_not(np.isnan(Jz)))], s=2, c='blue')
axarr[1][2].scatter(STMgFe, STJz, s=6, c='none', lw=0.5, edgecolor='black')

axarr[1][2].set(xlabel=r'$[\alpha/\text{Fe}]$')
axarr[1][2].set(ylabel=r'$\sqrt{J_z} [\,\text{kpc}\,\text{km}/\text{s}\,]^{1/2}$')
axarr[1][2].set(xlim=myrange[0])
axarr[1][2].set(ylim=myrange[1])
axarr[1][2].set_yticks([0,5,10,15,20])

for ax in np.concatenate(axarr):
    ax.axis('auto')


fig.subplots_adjust(bottom=0.05)
plt.tight_layout()
#cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7], frameon=False)
cb = fig.colorbar(im, ax=axarr.ravel().tolist(), orientation='horizontal', fraction=0.026, pad=0.17)
cb.set_label('number')
plt.savefig('actions_vs_metallicity.pdf')
