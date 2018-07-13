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
plt.rc('font', family='serif', size=16)

#clean out Jr

nanarr = np.logical_not(np.isnan(Jr))
valarr = np.logical_and(Jr >= 0, Jr<1000)

cleankeys = np.where(np.logical_and(nanarr,valarr))[0]
Jr = Jr[cleankeys]
Lz = Lz[cleankeys]
Jz = Jz[cleankeys]
age = age[cleankeys]


Jrhcutlist = np.linspace(30,100,10)
Lzlcutlist = np.linspace(900,1800,10)
Jzhcutlist = np.linspace(30,80,10)
output=[]

for Jrcut in Jrhcutlist:
    for Lzcut in Lzlcutlist:
        for Jzcut in Jzhcutlist:
            agekeys = np.logical_and(age > 0, age < 6)
            Jzkeys = Jz < Jzcut
            Lzkeys = Lz > Lzcut
            Jrkeys = Jr < Jrcut
            LzandJr = np.logical_and(Lzkeys, Jrkeys)
            actionkeys = np.logical_and(LzandJr, Jzkeys)
            actandagekeys = np.logical_and(agekeys, actionkeys)
            totage = len(np.where(agekeys)[0])
            totact = len(np.where(actionkeys)[0])
            totageandact = len(np.where(actandagekeys)[0])

            try:
                contamrate = 1.0 - (totageandact/totact)
            except:
                contamrate = np.nan

            try:
                capturerate = totageandact/totact
            except:
                capturerate = np.nan
            output.append([Jrcut, Lzcut, Jzcut, contamrate, capturerate])
            quit()
