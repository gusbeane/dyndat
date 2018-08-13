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

rangedict={'Jr': [[0,14],[0,20]],
            'Lz': [[0,14],[800,2200]],
            'Jz': [[0,14],[0,10]]}

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

N = int(sys.argv[1])
ageline = []
agelinerr = []
Jrline = []
Jrline_upspread = []
Jrline_lowspread = []
Jrline_uperr = []
Jrline_lowerr = []
Lzline = []
Lzline_upspread = []
Lzline_lowspread = []
Lzline_uperr = []
Lzline_lowerr = []
Jzline = []
Jzline_upspread = []
Jzline_lowspread = []
Jzline_uperr = []
Jzline_lowerr = []
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

def bootstrap_error(actionlist, nsample):
    central = np.nanmedian(actionlist)
    distribution = []
    for i in range(nsample):
        newlist = np.random.choice(actionlist, len(actionlist))
        distribution.append(np.nanmedian(newlist) - central)
    up = np.nanpercentile(distribution,95)
    low = np.nanpercentile(distribution,5)
    return central - up, central - low, low

"""
testlow = []
testup = []
testbin = ival[2000]
testkeys = np.argsort(age)[testbin:testbin+N]
thisJr = Jr[testkeys]
nsample_list = np.arange(10,5000,10)
for n in tqdm(nsample_list):
    low, up = bootstrap_error(thisJr, n)
    testlow.append(low)
    testup.append(up)

testlow = np.array(testlow)
testup = np.array(testup)

plt.plot(nsample_list, np.abs(testlow-testlow[-1])/testlow[-1])
plt.plot(nsample_list, np.abs(testup-testup[-1])/testup[-1])
plt.set_yscale('log')
plt.show()
plt.close()
quit()0
"""

bootstrap_n = 2000

diagnostic = []
age_argsort = np.argsort(age)
for bin in tqdm(ival):
    keys = age_argsort[bin:bin+N]
    thisage = age[keys]
    thisJr = Jr[keys]
    thisLz = Lz[keys]
    thisJz = Jz[keys]
    
    ageline.append(np.nanmedian(thisage))
    agelinerr.append(np.nanstd(thisage, ddof=1)/np.sqrt(N-1))
    
    Jrline.append(np.nanmedian(thisJr))
    Lzline.append(np.nanmedian(thisLz))
    Jzline.append(np.nanmedian(thisJz))

    Jrline_lowspread.append(np.nanpercentile(thisJr,10))
    Lzline_lowspread.append(np.nanpercentile(thisLz,10))
    Jzline_lowspread.append(np.nanpercentile(thisJz,10))

    Jrline_upspread.append(np.nanpercentile(thisJr,90))
    Lzline_upspread.append(np.nanpercentile(thisLz,90))
    Jzline_upspread.append(np.nanpercentile(thisJz,90))

    Jrlow, Jrhigh, low = bootstrap_error(thisJr, bootstrap_n)
    Lzlow, Lzhigh, lowboo = bootstrap_error(thisLz, bootstrap_n)
    Jzlow, Jzhigh, lowboo = bootstrap_error(thisJz, bootstrap_n)
    diagnostic.append(low)

    Jrline_lowerr.append(Jrlow)
    Jrline_uperr.append(Jrhigh)

    Lzline_lowerr.append(Lzlow)
    Lzline_uperr.append(Lzhigh)

    Jzline_lowerr.append(Jzlow)
    Jzline_uperr.append(Jzhigh)

print(np.median(diagnostic))

ageline = np.array(ageline)
agelinerr = np.array(agelinerr)
Jrline = np.array(Jrline)
Jrline_upspread = np.array(Jrline_upspread)
Jrline_lowspread = np.array(Jrline_lowspread)
Jrline_uperr = np.array(Jrline_uperr)
Jrline_lowerr = np.array(Jrline_lowerr)
Lzline = np.array(Lzline)
Lzline_upspread = np.array(Lzline_upspread)
Lzline_lowspread = np.array(Lzline_lowspread)
Lzline_uperr = np.array(Lzline_uperr)
Lzline_lowerr = np.array(Lzline_lowerr)
Jzline = np.array(Jzline)
Jzline_upspread = np.array(Jzline_upspread)
Jzline_lowspread = np.array(Jzline_lowspread)
Jzline_uperr = np.array(Jzline_uperr)
Jzline_lowerr = np.array(Jzline_lowerr)

"""
Jr vs. age
"""
fig = plt.figure(figsize=(7.1,4))
axarr = fig.subplots(2,3)

myrange = rangedict['Jr']
heatmap, xedges, yedges = np.histogram2d(age[np.where(np.logical_not(np.isnan(Jr)))], Jr[np.where(np.logical_not(np.isnan(Jr)))], bins=20, range=myrange)
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]

cm = copy(plt.cm.winter_r)
cm.set_bad(color='white', alpha=0.5)
heatmap.T[heatmap.T == 0] = np.nan

axarr[0][0].imshow(heatmap.T, extent=extent, origin='lower', cmap=cm, vmin=0, vmax=80)

axarr[0][0].scatter(STage, STJr, s=6, c='none', lw=0.8, edgecolor='black')

axarr[0][0].set(xlabel=r'$\text{age}\,[\,\text{Gyr}\,]$')
axarr[0][0].set(ylabel=r'$\sqrt{J_r} [\,\text{kpc}\,\text{km}/\text{s}\,]^{1/2}$')
axarr[0][0].set(xlim=myrange[0])
axarr[0][0].set(ylim=myrange[1])
axarr[0][0].yaxis.set_ticks([0,5,10,15,20])
axarr[0][0].yaxis.set_ticks(range(20), minor=True)

"""
Lz vs. age
"""
myrange = rangedict['Lz']
heatmap, xedges, yedges = np.histogram2d(age[np.where(np.logical_not(np.isnan(Lz)))], Lz[np.where(np.logical_not(np.isnan(Lz)))], bins=20, range=myrange)
extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]

heatmap.T[heatmap.T == 0] = np.nan

axarr[0][1].imshow(heatmap.T, extent=extent, origin='lower', cmap=cm, vmin=0, vmax=80)
axarr[0][1].scatter(STage, STLz, s=6, c='none', lw=0.8, edgecolor='black')

axarr[0][1].set(xlabel=r'$\text{age}\,[\,\text{Gyr}\,]$')
axarr[0][1].set(ylabel=r'$L_z [\,\text{kpc}\,\text{km}/\text{s}\,]$')
axarr[0][1].set(xlim=myrange[0])
axarr[0][1].set(ylim=myrange[1])
axarr[0][1].xaxis.set_ticks(range(15), minor=True)
axarr[0][1].yaxis.set_ticks([1000,1500,2000])
axarr[0][1].yaxis.set_ticks(np.arange(800,2200,100), minor=True)

"""
Jz vs. age
"""
#Jz = Jz[np.where(Jz >= 0.0)]
#STJz = STJz[np.where(STJz >= 0.0)]
#age = age[np.where(Jz >= 0.0)]
#STage = STage[np.where(STJz >= 0.0)]

myrange=rangedict['Jz']
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
axarr[1][0].set(xlim=rangedict['Jr'][0])
axarr[1][0].xaxis.set_ticks(range(15), minor=True)
axarr[1][0].yaxis.set_ticks([0,5,10,15,20])
axarr[1][0].yaxis.set_ticks(range(20), minor=True)
axarr[1][0].plot(ageline, Jrline, lw=0.4, color='black', zorder=5)
#axarr[1][0].plot(ageline, Jrline_uperr, lw=1, linestyle='dashed', color='gray', zorder=4)
#axarr[1][0].plot(ageline, Jrline_lowerr, lw=1, linestyle='dashed', color='gray', zorder=3)
axarr[1][0].fill_between(ageline, Jrline_lowerr, Jrline_uperr, lw=1, linestyle='dotted', color='gray', zorder=2)
axarr[1][0].plot(ageline, Jrline_upspread, lw=1, linestyle='dotted', color='gray', zorder=1)
axarr[1][0].plot(ageline, Jrline_lowspread, lw=1, linestyle='dotted', color='gray', zorder=1)
#axarr[1][0].axis('auto')

axarr[1][1].set(xlabel=r'$\text{age}\,[\,\text{Gyr}\,]$')
axarr[1][1].set(ylabel=r'$L_z [\,\text{kpc}\,\text{km}/\text{s}\,]$')
axarr[1][1].set(xlim=[0, 14])
axarr[1][1].xaxis.set_ticks(range(15), minor=True)
axarr[1][1].yaxis.set_ticks([1000,1500,2000])
axarr[1][1].yaxis.set_ticks(np.arange(800,2300,100), minor=True)
axarr[1][1].plot(ageline, Lzline, lw=0.4, color='black', zorder=5)
#axarr[1][1].plot(ageline, Lzline_uperr, lw=1, linestyle='dashed', color='gray', zorder=4)
#axarr[1][1].plot(ageline, Lzline_lowerr, lw=1, linestyle='dashed', color='gray', zorder=3)
axarr[1][1].fill_between(ageline, Lzline_lowerr, Lzline_uperr, lw=1, linestyle='dotted', color='gray', zorder=2)
axarr[1][1].plot(ageline, Lzline_upspread, lw=1, linestyle='dotted', color='gray', zorder=1)
axarr[1][1].plot(ageline, Lzline_lowspread, lw=1, linestyle='dotted', color='gray', zorder=1)
#axarr[1][1].axis('auto')

myrange=rangedict['Jz']
axarr[1][2].set(xlabel=r'$\text{age}\,[\,\text{Gyr}\,]$')
axarr[1][2].set(ylabel=r'$\sqrt{J_z} [\,\text{kpc}\,\text{km}/\text{s}\,]^{1/2}$')
axarr[1][2].set(xlim=myrange[0])
axarr[1][2].xaxis.set_ticks(range(15), minor=True)
axarr[1][2].yaxis.set_ticks([0,5,10])
axarr[1][2].plot(ageline, Jzline, lw=0.4, color='black', zorder=5)
#axarr[1][2].plot(ageline, Jzline_uperr, lw=1, linestyle='dashed', color='gray', zorder=4)
#axarr[1][2].plot(ageline, Jzline_lowerr, lw=1, linestyle='dashed', color='gray', zorder=3)
axarr[1][2].fill_between(ageline, Jzline_lowerr, Jzline_uperr, lw=1, linestyle='dotted', color='gray', zorder=2)
axarr[1][2].plot(ageline, Jzline_upspread, lw=1, linestyle='dotted', color='gray', zorder=1)
axarr[1][2].plot(ageline, Jzline_lowspread, lw=1, linestyle='dotted', color='gray', zorder=1)
#axarr[1][2].axis('auto')

for ax in axarr:
    ax[0].axis('auto')
    ax[1].axis('auto')
    ax[2].axis('auto')

axarr[1][0].set(ylim=rangedict['Jr'][1])
axarr[1][1].set(ylim=rangedict['Lz'][1])
axarr[1][2].set(ylim=rangedict['Jz'][1])


plt.tight_layout()
fig.subplots_adjust(bottom=0.1)
#cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7], frameon=False)
cb = fig.colorbar(im, ax=axarr.ravel().tolist(), orientation='horizontal', fraction=0.02, pad=0.15)
cb.set_label('number')
plt.savefig('actions_vs_age.pdf')

#plt.plot(ageline, Jrlinerr/Jrline)
