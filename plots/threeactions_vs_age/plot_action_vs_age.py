import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import warnings
from copy import copy
#from astropy.stats import SigmaClip
from scipy import stats as st

from tqdm import tqdm
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

STJr = STtable['Jr']
STLz = np.abs(STtable['Lz'])
STJz = STtable['Jz']
STage = STtable['age_mean']

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=9)

Jr = np.sqrt(Jr)
STJr = np.sqrt(STJr)
Jz = np.sqrt(Jz)
STJz = np.sqrt(STJz)

nanbool = np.logical_or(np.isnan(Jr), np.isnan(Jz))
nankeys = np.where(np.logical_not(nanbool))[0]
age = age[nankeys]
Jr = Jr[nankeys]
Lz = Lz[nankeys]
Jz = Jz[nankeys]

N = 200
ageline = []
agelinerr = []
Jrline = []
Jrlinerr = []
Jrlinsterr = []
Lzline = []
Lzlinerr = []
Lzlinsterr = []
Jzline = []
Jzlinerr = []
Jzlinsterr = []
ival = np.arange(0, len(age)-N)

print("before clip:", len(age))

scut = 6
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

print("after clip:", len(age))

for bin in tqdm(ival):
    keys = np.argsort(age)[bin:bin+N]
    thisage = age[keys]
    thisJr = Jr[keys]
    thisLz = Lz[keys]
    thisJz = Jz[keys]
    
    ageline.append(np.nanmedian(thisage))
    agelinerr.append(np.nanstd(thisage, ddof=1)/np.sqrt(N-1))
    
    Jrline.append(np.nanmedian(thisJr))
    Lzline.append(np.nanmedian(thisLz))
    Jzline.append(np.nanmedian(thisJz))

    Jrlinerr.append(np.nanstd(thisJr, ddof=1))
    Lzlinerr.append(np.nanstd(thisLz, ddof=1))
    Jzlinerr.append(np.nanstd(thisJr, ddof=1))
    
    Jrlinsterr.append(np.nanstd(thisJr, ddof=1)/np.sqrt(N-1))
    Lzlinsterr.append(np.nanstd(thisLz, ddof=1)/np.sqrt(N-1))
    Jzlinsterr.append(np.nanstd(thisJz, ddof=1)/np.sqrt(N-1))

ageline = np.array(ageline)
agelinerr = np.array(agelinerr)
Jrline = np.array(Jrline)
Jrlinerr = np.array(Jrlinerr)
Jrlinsterr = np.array(Jrlinsterr)
Lzline = np.array(Lzline)
Lzlinerr = np.array(Lzlinerr)
Lzlinsterr = np.array(Lzlinsterr)
Jzline = np.array(Jzline)
Jzlinerr = np.array(Jzlinerr)
Jzlinsterr = np.array(Jzlinsterr)


"""
Jr vs. age
"""
fig = plt.figure(figsize=(7.1,4))
axarr = fig.subplots(2,3)


heatmap, xedges, yedges = np.histogram2d(age[np.where(np.logical_not(np.isnan(Jr)))], Jr[np.where(np.logical_not(np.isnan(Jr)))], bins=20, range=[[0,14],[0,20]])
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]

cm = copy(plt.cm.winter_r)
cm.set_bad(color='white', alpha=0.5)
heatmap.T[heatmap.T == 0] = np.nan

axarr[0][0].imshow(heatmap.T, extent=extent, origin='lower', cmap=cm, vmin=0, vmax=80)

axarr[0][0].scatter(STage, STJr, s=6, c='none', lw=0.8, edgecolor='black')

axarr[0][0].set(xlabel=r'$\text{age}\,[\,\text{Gyr}\,]$')
axarr[0][0].set(ylabel=r'$\sqrt{J_r} [\,\text{kpc}\,\text{km}/\text{s}\,]^{1/2}$')
axarr[0][0].set(xlim=[0, 14])
axarr[0][0].set(ylim=[0, 20])
axarr[0][0].xaxis.set_ticks(range(15), minor=True)
axarr[0][0].yaxis.set_ticks([0,5,10,15,20])
axarr[0][0].yaxis.set_ticks(range(20), minor=True)

"""
Lz vs. age
"""
heatmap, xedges, yedges = np.histogram2d(age[np.where(np.logical_not(np.isnan(Lz)))], Lz[np.where(np.logical_not(np.isnan(Lz)))], bins=20, range=[[0,14],[800,2000]])
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]

heatmap.T[heatmap.T == 0] = np.nan

axarr[0][1].imshow(heatmap.T, extent=extent, origin='lower', cmap=cm, vmin=0, vmax=80)
axarr[0][1].scatter(STage, STLz, s=6, c='none', lw=0.8, edgecolor='black')

axarr[0][1].set(xlabel=r'$\text{age}\,[\,\text{Gyr}\,]$')
axarr[0][1].set(ylabel=r'$L_z [\,\text{kpc}\,\text{km}/\text{s}\,]$')
axarr[0][1].set(xlim=[0, 14])
axarr[0][1].set(ylim=[800,2000])
axarr[0][1].xaxis.set_ticks(range(15), minor=True)
axarr[0][1].yaxis.set_ticks([1000,1500,2000])
axarr[0][1].yaxis.set_ticks(np.arange(800,2000,100), minor=True)

"""
Jz vs. age
"""
#Jz = Jz[np.where(Jz >= 0.0)]
#STJz = STJz[np.where(STJz >= 0.0)]
#age = age[np.where(Jz >= 0.0)]
#STage = STage[np.where(STJz >= 0.0)]

myrange=[[0,14],[0,10]]
#mybins = [30,np.logspace(np.log10(myrange[1][0]),np.log10(myrange[1][1]),30)]
mybins = 30
heatmap, xedges, yedges = np.histogram2d(age[np.where(np.logical_not(np.isnan(Jz)))], Jz[np.where(np.logical_not(np.isnan(Jz)))], bins=mybins, range=myrange)
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]

heatmap.T[heatmap.T == 0] = np.nan

im = axarr[0][2].imshow(heatmap.T, extent=extent, origin='lower', cmap=cm, vmin=0, vmax=80)
#axarr[0][2].autoscale(False)
axarr[0][2].scatter(STage, STJz, s=6, c='none', lw=0.8, edgecolor='black')

#axarr[2].set_yscale('log')
axarr[0][2].set(xlabel=r'$\text{age}\,[\,\text{Gyr}\,]$')
axarr[0][2].set(ylabel=r'$\sqrt{J_z} [\,\text{kpc}\,\text{km}/\text{s}\,]^{1/2}$')
axarr[0][2].set_xlim(myrange[0])
axarr[0][2].set_ylim(myrange[1])
axarr[0][2].xaxis.set_ticks(range(15), minor=True)
axarr[0][2].yaxis.set_ticks([0,5,10])
axarr[0][2].yaxis.set_ticks(range(10), minor=True)
axarr[0][2].axis('auto')

axarr[1][0].set(xlabel=r'$\text{age}\,[\,\text{Gyr}\,]$')
axarr[1][0].set(ylabel=r'$\sqrt{J_r} [\,\text{kpc}\,\text{km}/\text{s}\,]^{1/2}$')
axarr[1][0].xaxis.set_ticks(range(15), minor=True)
axarr[1][0].yaxis.set_ticks([0,5,10,15,20])
axarr[1][0].yaxis.set_ticks(range(20), minor=True)
axarr[1][0].plot(ageline, Jrline, lw=1, color='black', zorder=5)
axarr[1][0].plot(ageline, Jrline+Jrlinsterr, lw=1, linestyle='dashed', color='gray', zorder=4)
axarr[1][0].plot(ageline, Jrline-Jrlinsterr, lw=1, linestyle='dashed', color='gray', zorder=3)
axarr[1][0].plot(ageline, Jrline+Jrlinerr, lw=1, linestyle='dotted', color='gray', zorder=2)
axarr[1][0].plot(ageline, Jrline-Jrlinerr, lw=1, linestyle='dotted', color='gray', zorder=1)
#axarr[1][0].axis('auto')

axarr[1][1].set(xlabel=r'$\text{age}\,[\,\text{Gyr}\,]$')
axarr[1][1].set(ylabel=r'$L_z [\,\text{kpc}\,\text{km}/\text{s}\,]$')
axarr[1][1].xaxis.set_ticks(range(15), minor=True)
axarr[1][1].yaxis.set_ticks([1000,1500,2000])
axarr[1][1].yaxis.set_ticks(np.arange(800,2000,100), minor=True)
axarr[1][1].plot(ageline, Lzline, lw=1, color='black', zorder=5)
axarr[1][1].plot(ageline, Lzline+Lzlinsterr, lw=1, linestyle='dashed', color='gray', zorder=4)
axarr[1][1].plot(ageline, Lzline-Lzlinsterr, lw=1, linestyle='dashed', color='gray', zorder=3)
axarr[1][1].plot(ageline, Lzline+Lzlinerr, lw=1, linestyle='dotted', color='gray', zorder=2)
axarr[1][1].plot(ageline, Lzline-Lzlinerr, lw=1, linestyle='dotted', color='gray', zorder=1)
#axarr[1][1].axis('auto')

myrange=[[0,14],[0,10]]
axarr[1][2].set(xlabel=r'$\text{age}\,[\,\text{Gyr}\,]$')
axarr[1][2].set(ylabel=r'$\sqrt{J_z} [\,\text{kpc}\,\text{km}/\text{s}\,]^{1/2}$')
axarr[1][2].xaxis.set_ticks(range(15), minor=True)
axarr[1][2].yaxis.set_ticks([0,5,10])
axarr[1][2].yaxis.set_ticks(range(10), minor=True)
axarr[1][2].plot(ageline, Jzline, lw=1, color='black', zorder=5)
axarr[1][2].plot(ageline, Jzline+Jzlinsterr, lw=1, linestyle='dashed', color='gray', zorder=4)
axarr[1][2].plot(ageline, Jzline-Jzlinsterr, lw=1, linestyle='dashed', color='gray', zorder=3)
axarr[1][2].plot(ageline, Jzline+Jzlinerr, lw=1, linestyle='dotted', color='gray', zorder=2)
axarr[1][2].set_adjustable("box")
#axarr[1][2].plot(ageline, Jzline-Jzlinerr, lw=1, linestyle='dotted', color='gray', zorder=1)
#axarr[1][2].axis('auto')

for ax in axarr:
    ax[0].axis('auto')
    ax[1].axis('auto')
    ax[2].axis('auto')

axarr[1][0].set(ylim=[0, 20])
axarr[1][1].set(ylim=[800,2000])
axarr[1][2].set(ylim=myrange[1])
axarr[1][0].set_xlim(myrange[0][0], myrange[0][1])
axarr[1][1].set_xlim(myrange[0][0], myrange[0][1])
axarr[1][2].set_xlim(myrange[0][0], myrange[0][1])


plt.tight_layout()
fig.subplots_adjust(bottom=0.1)
#cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7], frameon=False)
cb = fig.colorbar(im, ax=axarr.ravel().tolist(), orientation='horizontal', fraction=0.02, pad=0.15)
cb.set_label('number')
plt.savefig('actions_vs_age.pdf')

#plt.plot(ageline, Jrlinerr/Jrline)
