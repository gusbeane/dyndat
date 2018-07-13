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

Jr = APOKASCtable['Jr']
Lz = np.abs(APOKASCtable['Lz'])
Jz = APOKASCtable['Jz']
age = 10.**APOKASCtable['log_age']/1000.0

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)

#clean out Jr

fig = plt.figure(figsize=(3,6))
axarr = fig.subplots(2,1)

nanarr = np.logical_not(np.isnan(Jr))
valarr = np.logical_and(Jr >= 0, Jr<1000)

cleankeys = np.where(np.logical_and(nanarr,valarr))[0]
Jr = Jr[cleankeys]
Lz = Lz[cleankeys]
Jz = Jz[cleankeys]
age = age[cleankeys]

Jrkeys = np.where(Jr < 4.4*4.4)
axarr[0].hist(age, bins=60, label='APOKASC2 full sample', alpha=0.5, density=True)
axarr[0].hist(age[Jrkeys], bins=60, label=r'$\sqrt{J_r} < 4.4\,\sqrt{\text{kpc}\,\text{km}/\text{s}}$', alpha=0.5, density=True)
axarr[0].set(xlabel=r'$\text{age}\,[\,\text{Gyr}\,]$')
axarr[0].set(xlim=[0,20])
axarr[0].set(ylim=[0,0.25])
axarr[0].legend(frameon=False)

Jrkeys = np.where(Jr > 8*8)
axarr[1].hist(age, bins=60, label='APOKASC2 full sample', alpha=0.5, density=True)
axarr[1].hist(age[Jrkeys], bins=60, label=r'$\sqrt{J_r} > 8\,\sqrt{\text{kpc}\,\text{km}/\text{s}}$', alpha=0.5, density=True)
axarr[1].set(xlabel=r'$\text{age}\,[\,\text{Gyr}\,]$')
axarr[1].set(xlim=[0,20])
axarr[1].set(ylim=[0,0.25])
axarr[1].legend(frameon=False)

plt.tight_layout()
plt.savefig('agehist_actioncuts.pdf')
plt.close()

