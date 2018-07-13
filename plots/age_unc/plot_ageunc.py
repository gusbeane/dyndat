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

age = age[np.where(ageunc < 100)]
ageunc = ageunc[np.where(ageunc < 100)]

#process

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=9)

"""
Jr vs. age
"""
fig, ax = plt.subplots(figsize=(3.3938,3))

cm = copy(plt.cm.winter_r)
cm.set_bad(color='white', alpha=0.5)

myrange = [[0,14],[0,4]]

heatmap, xedges, yedges = np.histogram2d(age, ageunc, bins=40, range=myrange)
heatmap.T[heatmap.T == 0] = np.nan
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]
im =  ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=cm, vmin=0, vmax=80)

ax.scatter(STage, STageunc/STage, s=6, c='none', lw=0.8, edgecolor='black')

ax.set(xlabel=r'$\text{age}\,[\,\text{Gyr}\,]$')
ax.set(ylabel=r'$\sigma(\text{age})\,[\,\text{Gyr}\,]$')
ax.set(xlim=myrange[0])
ax.set(ylim=myrange[1])
#ax[0].set_aspect(1.25)
ax.axis('auto')

plt.tight_layout()
#cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7], frameon=False)
cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.036, pad=0.22)
cb.set_label('number')
plt.savefig('age_unc.pdf')
