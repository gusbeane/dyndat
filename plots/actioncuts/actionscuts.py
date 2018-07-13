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
plt.rc('font', family='serif', size=9)

#clean out Jr

nanarr = np.logical_not(np.isnan(Jr))
valarr = np.logical_and(Jr >= 0, Jr<1000)

cleankeys = np.where(np.logical_and(nanarr,valarr))[0]
Jr = Jr[cleankeys]
Lz = Lz[cleankeys]
Jz = Jz[cleankeys]
age = age[cleankeys]


fig = plt.figure(figsize=(7.1,4.8))
axarr = fig.subplots(2,3)

contamlist = []
capturelist = []
Jrlcutlist = np.linspace(0,100,5000)
for Jrcut in Jrlcutlist:
    agekeys = np.logical_and(age > 0, age < 4)
    Jrkeys = Jr < Jrcut
    ageandJrkeys = np.logical_and(Jrkeys, agekeys)

    totage = len(np.where(agekeys)[0])
    totJr = len(np.where(Jrkeys)[0])
    totageandJr = len(np.where(ageandJrkeys)[0])
    
    try:
        contamrate = 1.0 - (totageandJr/totJr)
    except:
        contamrate = np.nan
    contamlist.append(contamrate)

    try:
        capturerate = totageandJr/totage
    except:
        capturerate = np.nan
    capturelist.append(capturerate)

axarr[0][0].plot(np.sqrt(Jrlcutlist), contamlist, label='contamination rate', color='black')
axarr[0][0].plot(np.sqrt(Jrlcutlist), capturelist, label='capture rate', color='black', linestyle='dashed')
axarr[0][0].set(xlim=[0,10])
axarr[0][0].set(ylim=[0,1])
axarr[0][0].set(xlabel=r'$\sqrt{J_{r,\text{cut}}}\,[\,\text{kpc}\,\text{km}/\text{s}\,]^{1/2}$')
axarr[0][0].text(2,0.8,r'$J_r < J_{r,\text{cut}}$')

contamlist = np.array(contamlist)
capturelist = np.array(capturelist)
hfkey = np.argmin(np.abs(capturelist - 0.5))
print(Jrlcutlist[hfkey])
print(contamlist[hfkey])


contamlist = []
capturelist = []
Lzlcutlist = np.linspace(900,2000,5000)
for Lzcut in Lzlcutlist:
    agekeys = np.logical_and(age > 0, age < 4)
    Lzkeys = Lz > Lzcut
    ageandLzkeys = np.logical_and(Lzkeys, agekeys)

    totage = len(np.where(agekeys)[0])
    totLz = len(np.where(Lzkeys)[0])
    totageandLz = len(np.where(ageandLzkeys)[0])
    
    try:
        contamrate = 1.0 - (totageandLz/totLz)
    except:
        contamrate = np.nan
    contamlist.append(contamrate)

    try:
        capturerate = totageandLz/totage
    except:
        capturerate = np.nan
    capturelist.append(capturerate)

axarr[0][1].plot(Lzlcutlist, contamlist, label='contam', color='black')
axarr[0][1].plot(Lzlcutlist, capturelist, label='capture', color='black', linestyle='dashed')
axarr[0][1].set(xlim=[1000,2000])
axarr[0][1].set(ylim=[0,1])
axarr[0][1].set(xlabel=r'$L_{z,\text{cut}}\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')
axarr[0][1].set(title=r'$\text{age} < 4\,\text{Gyr}$')
axarr[0][1].text(1100,0.8,r'$L_z > L_{z,\text{cut}}$')

contamlist = np.array(contamlist)
capturelist = np.array(capturelist)
hfkey = np.argmin(np.abs(capturelist - 0.5))
print(Lzlcutlist[hfkey])
print(contamlist[hfkey])

contamlist = []
capturelist = []
Jzlcutlist = np.linspace(0,100,2000)
for Jzcut in Jzlcutlist:
    agekeys = np.logical_and(age > 0, age < 4)
    Jzkeys = Jz < Jzcut
    ageandJzkeys = np.logical_and(Jzkeys, agekeys)

    totage = len(np.where(agekeys)[0])
    totJz = len(np.where(Jzkeys)[0])
    totageandJz = len(np.where(ageandJzkeys)[0])
    
    try:
        contamrate = 1.0 - (totageandJz/totJz)
    except:
        contamrate = np.nan
    contamlist.append(contamrate)

    try:
        capturerate = totageandJz/totage
    except:
        capturerate = np.nan
    capturelist.append(capturerate)

axarr[0][2].plot(Jzlcutlist, contamlist, label='contamination rate', color='black')
axarr[0][2].plot(Jzlcutlist, capturelist, label='capture rate', color='black', linestyle='dashed')
axarr[0][2].set(xlim=[0,20])
axarr[0][2].set(ylim=[0,1])
axarr[0][2].set(xlabel=r'$J_{z,\text{cut}} \,[\,\text{kpc}\,\text{km}/\text{s}\,]$')
axarr[0][2].legend(frameon=False)
axarr[0][2].text(1.9,0.8,r'$J_z < J_{z,\text{cut}}$')

contamlist = np.array(contamlist)
capturelist = np.array(capturelist)
hfkey = np.argmin(np.abs(capturelist - 0.5))
print(Jzlcutlist[hfkey])
print(contamlist[hfkey])

contamlist = []
capturelist = []
Jrlcutlist = np.linspace(0,1000,5000)
for Jrcut in Jrlcutlist:
    agekeys = age > 7
    Jrkeys = Jr > Jrcut
    ageandJrkeys = np.logical_and(Jrkeys, agekeys)

    totage = len(np.where(agekeys)[0])
    totJr = len(np.where(Jrkeys)[0])
    totageandJr = len(np.where(ageandJrkeys)[0])
    
    try:
        contamrate = 1.0 - (totageandJr/totJr)
    except:
        contamrate = np.nan
    contamlist.append(contamrate)

    try:
        capturerate = totageandJr/totage
    except:
        capturerate = np.nan
    capturelist.append(capturerate)

axarr[1][0].plot(np.sqrt(Jrlcutlist), contamlist, label='contamination rate', color='black')
axarr[1][0].plot(np.sqrt(Jrlcutlist), capturelist, label='capture rate', color='black', linestyle='dashed')
axarr[1][0].set(xlim=[0,20])
axarr[1][0].set(ylim=[0,1])
axarr[1][0].set(xlabel=r'$\sqrt{J_{r,\text{cut}}}\,[\,\text{kpc}\,\text{km}/\text{s}\,]^{1/2}$')
axarr[1][0].text(12,0.8,r'$J_r > J_{r,\text{cut}}$')

contamlist = np.array(contamlist)
capturelist = np.array(capturelist)
hfkey = np.argmin(np.abs(capturelist - 0.5))
print(Jrlcutlist[hfkey])
print(contamlist[hfkey])

contamlist = []
capturelist = []
Lzlcutlist = np.linspace(900,2500,5000)
for Lzcut in Lzlcutlist:
    agekeys = age > 7
    Lzkeys = Lz < Lzcut
    ageandLzkeys = np.logical_and(Lzkeys, agekeys)

    totage = len(np.where(agekeys)[0])
    totLz = len(np.where(Lzkeys)[0])
    totageandLz = len(np.where(ageandLzkeys)[0])
    
    try:
        contamrate = 1.0 - (totageandLz/totLz)
    except:
        contamrate = np.nan
    contamlist.append(contamrate)

    try:
        capturerate = totageandLz/totage
    except:
        capturerate = np.nan
    capturelist.append(capturerate)

axarr[1][1].plot(Lzlcutlist, contamlist, label='contam', color='black')
axarr[1][1].plot(Lzlcutlist, capturelist, label='capture', color='black', linestyle='dashed')
axarr[1][1].set(xlim=[1200,2200])
axarr[1][1].set(ylim=[0,1])
axarr[1][1].set(xlabel=r'$L_{z,\text{cut}}\,[\,\text{kpc}\,\text{km}/\text{s}\,]$')
axarr[1][1].set(title=r'$\text{age} > 7\,\text{Gyr}$')
axarr[1][1].text(1300,0.8,r'$L_z < L_{z,\text{cut}}$')

contamlist = np.array(contamlist)
capturelist = np.array(capturelist)
hfkey = np.argmin(np.abs(capturelist - 0.5))
print(Lzlcutlist[hfkey])
print(contamlist[hfkey])

contamlist = []
capturelist = []
Jzlcutlist = np.linspace(0,40,2000)
for Jzcut in Jzlcutlist:
    agekeys = age > 7
    Jzkeys = Jz > Jzcut
    ageandJzkeys = np.logical_and(Jzkeys, agekeys)

    totage = len(np.where(agekeys)[0])
    totJz = len(np.where(Jzkeys)[0])
    totageandJz = len(np.where(ageandJzkeys)[0])
    
    try:
        contamrate = 1.0 - (totageandJz/totJz)
    except:
        contamrate = np.nan
    contamlist.append(contamrate)

    try:
        capturerate = totageandJz/totage
    except:
        capturerate = np.nan
    capturelist.append(capturerate)

contamlist = np.array(contamlist)
capturelist = np.array(capturelist)
hfkey = np.argmin(np.abs(capturelist - 0.5))
print(Jzlcutlist[hfkey])
print(contamlist[hfkey])

axarr[1][2].plot(Jzlcutlist, contamlist, label='contamination rate', color='black')
axarr[1][2].plot(Jzlcutlist, capturelist, label='capture rate', color='black', linestyle='dashed')
axarr[1][2].set(xlim=[0,40])
axarr[1][2].set(ylim=[0,1])
axarr[1][2].set(xlabel=r'$J_{z,\text{cut}} \,[\,\text{kpc}\,\text{km}/\text{s}\,]$')
axarr[1][2].text(20,0.8,r'$J_z > J_{z,\text{cut}}$')
#axarr[1][2].legend(frameon=False)
plt.tight_layout()
plt.savefig('actioncuts_4Gyr.pdf')
