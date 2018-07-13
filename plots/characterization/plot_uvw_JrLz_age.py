import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import warnings
from copy import copy

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

#MgFe = APOKASCtable['afe']
#Fe = APOKASCtable['feh']
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

#process
Jr = np.sqrt(Jr)
Lz = Lz/(8. * 220.)
vvel = vvel + 220.

STJr = np.sqrt(STJr)
STLz = STLz/(8. * 220.)
STvvel = STvvel + 220.

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)

"""
Jr vs. age
"""
fig = plt.figure(figsize=(4,6))
axarr = fig.subplots(2,1)



myrange = [[-150,150],[-75,100]]

heatmap, xedges, yedges = np.histogram2d(uvel, vvel, bins=25, range=myrange)
heatmapage, xedges, yedges = np.histogram2d(uvel, vvel, weights=age, bins=25, range=myrange)
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]
axarr[0].imshow(heatmapage.T/heatmap.T, extent=extent, origin='lower', cmap='hot', vmin=0, vmax=12)

axarr[0].scatter(STuvel, STvvel, s=16, c=STage, edgecolor='black',linewidths=1, vmin=0, vmax=12, cmap='hot')

axarr[0].set(xlabel=r'$U_{\text{LSR}}\,[\,\text{km}/\text{s}\,]$')
axarr[0].set(ylabel=r'$V_{\text{LSR}}\,[\,\text{km}/\text{s}\,]$')
axarr[0].set(xlim=myrange[0])
axarr[0].set(ylim=myrange[1])
#axarr[0].xaxis.set_ticks(range(15), minor=True)

"""
Lz vs. age
"""
myrange = [[0.5,1.3],[0,16]]

heatmap, xedges, yedges = np.histogram2d(Lz, Jr, bins=25, range=myrange)
heatmapage, xedges, yedges = np.histogram2d(Lz, Jr, weights=age, bins=25, range=myrange)
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]
im = axarr[1].imshow(heatmapage.T/heatmap.T, extent=extent, origin='lower', cmap='hot', vmin=0, vmax=12)

axarr[1].scatter(STLz, STJr, s=16, c=STage, edgecolor='black',linewidths=1, vmin=0, vmax=12, cmap='hot')

axarr[1].set(xlabel=r'$L_z\,[\,8\,\text{kpc}\,220\,\text{km}/\text{s}\,]$')
axarr[1].set(ylabel=r'$\sqrt{J_r}\,[\,\text{kpc}\,\text{km}/\text{s}\,]^{1/2}$')
axarr[1].set(xlim=myrange[0])
axarr[1].set(ylim=myrange[1])

for ax in axarr:
    ax[0].axis('auto')
    ax[1].axis('auto')

plt.tight_layout()
#cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7], frameon=False)
cb = fig.colorbar(im, ax=axarr.ravel().tolist(), orientation='horizontal', fraction=0.036, pad=0.11)
cb.set_label(r'$\text{Age}\,[\,\text{Gyr}\,]$')
plt.savefig('uv_JrLz_age.pdf')
