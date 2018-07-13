import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import warnings
from tqdm import tqdm
from copy import copy
from scipy.stats import gaussian_kde

from tqdm import tqdm

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

MgFe = APOKASCtable['afe']
Fe = APOKASCtable['feh']
Jr = APOKASCtable['Jr']
Lz = np.abs(APOKASCtable['Lz'])
Jz = APOKASCtable['Jz']
uvel = APOKASCtable['uvel']
vvel = APOKASCtable['vvel']
wvel = APOKASCtable['wvel']
age = 10.**APOKASCtable['log_age']/1000.0

STJr = STtable['Jr']
STLz = np.abs(STtable['Lz'])
STJz = STtable['Jz']
STuvel = STtable['uvel']
STvvel = STtable['vvel']
STwvel = STtable['wvel']
STage = STtable['age_mean']

#Mgerr = APOKASCtable['err_MgI']
#Feerr = APOKASCtable['err_feh']
#Jrerr = APOKASCtable['Jr_err']
#Lzerr = APOKASCtable['Lz_err']
#Jzerr = APOKASCtable['Jz_err']
#ageerr = APOKASCtable['age_std']

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=9)

age = np.array(age.tolist())
Jr = np.array(Jr.tolist())
Lz = np.array(Lz.tolist())
Jz = np.array(Jz.tolist()) 


#agelist = [4, 7, 10]
agelist = np.linspace(2,14,150)

titlelist =[ r'$2\,\text{Gyr} < \text{age} < 4\,\text{Gyr}$',r'$5\,\text{Gyr} < \text{age} < 7\,\text{Gyr}$', r'$8\,\text{Gyr} < \text{age} < 10\,\text{Gyr}$']

cm = copy(plt.cm.winter_r)
cm.set_bad(color='white', alpha=0.5)

for i in tqdm(range(len(agelist))):
    thisage = agelist[i]
    fig = plt.figure(figsize=(3,5))
    axarr = fig.subplots(2,1)
    ax1 = axarr[0]
    ylist = vvel - 220.
    xlist = uvel

    STylist = STvvel - 220.
    STxlist = STuvel
    
    xlistles = xlist[np.where(np.logical_and(age > thisage-2, age < thisage))]
    ylistles = ylist[np.where(np.logical_and(age > thisage-2, age < thisage))]
    ageles = age[np.where(np.logical_and(age > thisage-2, age < thisage))]

    STxlistles = STxlist[np.where(np.logical_and(STage > thisage-2, STage < thisage))]
    STylistles = STylist[np.where(np.logical_and(STage > thisage-2, STage < thisage))]
    STageles = STage[np.where(np.logical_and(STage > thisage-2, STage < thisage))]
    
    myrange=[[-150,150],[-100,75]]

    heatmap, xedges, yedges = np.histogram2d(xlistles, ylistles, bins=20, range=myrange)
    heatmap.T[heatmap.T == 0] = np.nan
    extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]
    #ax1.clf()
    im = ax1.imshow(heatmap.T, extent=extent, origin='lower', aspect=0.05, cmap=cm, vmin=0, vmax=60, zorder=1)

    ax1.scatter(STxlistles, STylistles, s=6, c='none', lw=0.8, edgecolor='black', zorder=2)

    ax1.set(xlabel=r'$U_{\text{LSR}}[\,\text{km}/\text{s}\,]$')
    ax1.set(ylabel=r'$V_{\text{LSR}}\,[\,\text{km}/\text{s}\,]$')

    ax2 = ax1.twinx()
    barwidth = 25. 
    ax2.bar(150-barwidth/2.0,2,barwidth,thisage-2)
    ax2.set(ylim=[0,14])
    ax2.yaxis.set_ticks([0,2,4,6,8,10,12,14])
    ax2.set(ylabel=r'$\text{age}\,[\,\text{Gyr}\,]$')
    ax1.axis('auto')
    
    # # # contours # # #
    # see https://stackoverflow.com/questions/50812810/contour-plot-from-xy-data-in-python

    x=xlistles
    y=ylistles
    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi = (zi-zi.min())/(zi.max() - zi.min())
    zi =zi.reshape(xi.shape)
    levels = [0.25, 0.5, 0.75]
    origin = 'lower'
    
    CS = ax1.contour(xi, yi, zi,levels = levels, colors=('black',), linewidths=(1.2,),origin=origin, zorder=3)
    #ax1.clabel(CS, fmt='%.3f', colors='black', fontsize=6)

    ax1.set(xlim = myrange[0], ylim=myrange[1])

    # #
    # #  Jr Lz plot
    # #
    ax1 = axarr[1]
    ylist = np.sqrt(Jr[np.where(np.logical_not(np.isnan(Jr)))])
    xlist = Lz[np.where(np.logical_not(np.isnan(Jr)))]/ ( 8. * 220.)
    agehere= age[np.where(np.logical_not(np.isnan(Jr)))]

    STylist = np.sqrt(STJr[np.where(np.logical_not(np.isnan(STJr)))])
    STxlist = STLz[np.where(np.logical_not(np.isnan(STJr)))]/ ( 8. * 220.)
    STagehere = STage[np.where(np.logical_not(np.isnan(STJr)))]
    
    xlistles = xlist[np.where(np.logical_and(agehere > thisage-2, agehere < thisage))]
    ylistles = ylist[np.where(np.logical_and(agehere > thisage-2, agehere < thisage))]
    ageles = agehere[np.where(np.logical_and(agehere > thisage-2, agehere < thisage))]

    STxlistles = STxlist[np.where(np.logical_and(STagehere > thisage-2, STagehere < thisage))]
    STylistles = STylist[np.where(np.logical_and(STagehere > thisage-2, STagehere < thisage))]
    STageles = STagehere[np.where(np.logical_and(STagehere > thisage-2, STagehere < thisage))]
    
    myrange=[[0.5,1.3],[0,20]]

    heatmap, xedges, yedges = np.histogram2d(xlistles, ylistles, bins=20, range=myrange)
    heatmap.T[heatmap.T == 0] = np.nan
    extent = [xedges[0], xedges[len(xedges)-1], yedges[0], yedges[len(yedges)-1]]
    #ax1.clf()
    im = ax1.imshow(heatmap.T, extent=extent, origin='lower', aspect=0.05, cmap=cm, vmin=0, vmax=60)

    ax1.scatter(STxlistles, STylistles, s=6, c='none', lw=0.8, edgecolor='black')

    ax1.set(xlabel=r'$L_z\,[8\,\text{kpc}\,220\,\text{km}/\text{s}]$')
    ax1.set(ylabel=r'$\sqrt{J_r}\,[\text{kpc}\,\text{km}/\text{s}]^{1/2}$')
    
    ax1.set(xlim = myrange[0], ylim=myrange[1])

    ax2 = ax1.twinx()
    barwidth *= (1.3-0.5)/300
    ax2.bar(1.3-barwidth/2.0,2,barwidth,thisage-2)
    ax2.set(ylim=[0,14])
    ax2.yaxis.set_ticks([0,2,4,6,8,10,12, 14])
    ax2.set(ylabel=r'$\text{age}\,[\,\text{Gyr}\,]$')
    ax1.axis('auto')
    


    plt.tight_layout()
    #cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7], frameon=False)
    cb = fig.colorbar(im, ax=axarr.ravel().tolist(), orientation='horizontal', fraction=0.022, pad=0.13)
    cb.set_label('number')
    plt.savefig('movie/JrLz_uv_agecut_'+"{0:0=3d}".format(i)+'.png', dpi=1000)
