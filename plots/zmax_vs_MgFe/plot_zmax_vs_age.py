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

# get APOKASC2 uncertainty (kind of a pain)

age = APOKASCtable['log_age']
ageplus = APOKASCtable['s_logagep']
ageminus = APOKASCtable['s_logagem']
agep = age+ageplus
agem = age-ageminus

age = (10.**age)/1000.0
agep = (10.**agep)/1000.0
agem = (10.**agem)/1000.0
ageunc = np.maximum(agep-age, age-agem)

STage = STtable['age_mean']
STageunc = STtable['age_std']

#age = age[np.where(ageunc < 100)]
#ageunc = ageunc[np.where(ageunc < 100)]

zmax = APOKASCtable['zmax']
STzmax = STtable['zmax']

MgFe = APOKASCtable['afe']
STMgFe = STtable['[MgI]'] - STtable['feh']

#process

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=9)

"""
Jr vs. age
"""
fig = plt.figure(figsize=(3.3938,6))
axarr = fig.subplots(2,1)
#fig = plt.figure()
#axarr = fig.subplots(2,1)

cm = copy(plt.cm.winter_r)
cm.set_bad(color='white', alpha=0.5)

myrange = [[0,14],[-0.05,4]]

heatmap, xedges, yedges = np.histogram2d(age, zmax/1000.0, bins=50, range=myrange)
heatmap.T[heatmap.T == 0] = np.nan
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]
im =  axarr[0].imshow(heatmap.T, extent=extent, origin='lower', cmap=cm, vmin=0, vmax=60)

axarr[0].scatter(STage, STzmax/1000.0, s=6, c='none', lw=0.8, edgecolor='black')

axarr[0].set(xlabel=r'$\text{age}\,[\,\text{Gyr}\,]$')
axarr[0].set(ylabel=r'$z_{\text{max}}\,[\,\text{kpc}\,]$')
axarr[0].set(xlim=myrange[0])
axarr[0].set(ylim=myrange[1])
axarr[0].set_yticks([0,1,2,3,4])
#ax[0].set_aspect(1.25)

myrange = [[-0.1,0.3],[-0.05,4]]

heatmap, xedges, yedges = np.histogram2d(MgFe, zmax/1000.0, bins=50, range=myrange)
heatmap.T[heatmap.T == 0] = np.nan
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]
im =  axarr[1].imshow(heatmap.T, extent=extent, origin='lower', cmap=cm, vmin=0, vmax=60)

axarr[1].scatter(STMgFe, STzmax/1000.0, s=6, c='none', lw=0.8, edgecolor='black')

axarr[1].set(xlabel=r'$[\,\text{Mg}/\text{Fe}\,]$')
axarr[1].set(ylabel=r'$z_{\text{max}}\,[\,\text{kpc}\,]$')
axarr[1].set(xlim=myrange[0])
axarr[1].set(ylim=myrange[1])
axarr[1].set_yticks([0,1,2,3,4])
axarr[0].axis('auto')
axarr[1].axis('auto')

#cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7], frameon=False)
plt.tight_layout()
cb = fig.colorbar(im, ax=axarr.ravel().tolist(), orientation='horizontal', fraction=0.036, pad=0.1)
cb.set_label('number')
plt.savefig('zmax_age_MgFe.pdf')
