import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import warnings
from pyia import GaiaData
import gala.dynamics as gd
import astropy.coordinates as coord
import astropy.units as u

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
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

APage = 10.**APOKASCtable['log_age']/1000.0
STage = STtable['age_mean'] 

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=9)

bins = np.arange(0,14,0.5)

fig = plt.figure(figsize=(3.3938,5))
axarr = fig.subplots(2,1)
axarr[0].hist(STage, bins=bins, edgecolor='black', fc='none', histtype='stepfilled')
axarr[0].set(xlabel=r'$\text{age}\,[\,\text{Gyr}\,]$')
axarr[0].set(ylabel=r'number')
axarr[0].set(xlim=[0,14])
axarr[0].set(title='Solar Twins')
axarr[0].set_xticks(range(15), minor=True)

axarr[1].hist(APage, bins=bins, edgecolor='black', fc='none', histtype='stepfilled')
axarr[1].set(xlabel=r'$\text{age}\,[\,\text{Gyr}\,]$')
axarr[1].set(xlim=[0,14])
axarr[1].set(ylabel=r'number')
axarr[1].set(title='APOKASC2')
axarr[1].set_xticks(range(15), minor=True)
#axarr[1].set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('age_hist.pdf')
