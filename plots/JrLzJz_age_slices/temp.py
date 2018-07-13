import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import warnings
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

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
age = 10.**APOKASCtable['log_age']/1000.0

STJr = STtable['Jr']
STLz = np.abs(STtable['Lz'])
STJz = STtable['Jz']
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

ylist = np.sqrt(Jr[np.where(np.logical_not(np.isnan(Jr)))])
xlist = Lz[np.where(np.logical_not(np.isnan(Jr)))]/ ( 8. * 220.)
zlist = Jz[np.where(np.logical_not(np.isnan(Jr)))]
age = age[np.where(np.logical_not(np.isnan(Jr)))]

STylist = np.sqrt(STJr[np.where(np.logical_not(np.isnan(STJr)))])
STxlist = STLz[np.where(np.logical_not(np.isnan(STJr)))]/ ( 8. * 220.)
STzlist = STJz[np.where(np.logical_not(np.isnan(STJr)))]
STage = STage[np.where(np.logical_not(np.isnan(STJr)))]

agelist = [4, 7, 10]


def plot_angle(theta, i):
    fig = plt.figure(figsize=(7.1,2.8))
    axarr = fig.subplots(1,3, subplot_kw={'projection': '3d'})

    for i in range(len(axarr)):
        thisage = agelist[i]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            xlistles = xlist[np.where(np.logical_and(age > thisage-2, age < thisage))]
            ylistles = ylist[np.where(np.logical_and(age > thisage-2, age < thisage))]
            zlistles = zlist[np.where(np.logical_and(age > thisage-2, age < thisage))]
            ageles = age[np.where(np.logical_and(age > thisage-2, age < thisage))]
    
            STxlistles = STxlist[np.where(np.logical_and(STage > thisage-2, STage < thisage))]
            STylistles = STylist[np.where(np.logical_and(STage > thisage-2, STage < thisage))]
            STzlistles = STzlist[np.where(np.logical_and(STage > thisage-2, STage < thisage))]
            STageles = STage[np.where(np.logical_and(STage > thisage-2, STage < thisage))]
            axarr[i].axis('auto')
        
            myrange=[[0.5,1.3],[0,20],[0,60]]

            axarr[i].scatter(xlistles, ylistles, zlistles, c='black', s=4, linewidth=0)

            axarr[i].scatter(STxlistles, STylistles, STzlistles, s=4, c='red', linewidth=0)

            if(i==0):
                axarr[i].set(xlabel=r'$L_z\,[8\,\text{kpc}\,220\,\text{km}/\text{s}]$')
            if(i==len(axarr)-1):
                axarr[i].set(ylabel=r'$\sqrt{J_r}\,[\text{kpc}\,\text{km}/\text{s}]^{1/2}$')
                axarr[i].set(zlabel=r'$J_z\,[\text{kpc}\,\text{km}/\text{s}]$')
            axarr[i].set(xlim = myrange[0], ylim=myrange[1], zlim=myrange[2])
            mytitle = '$' + str(thisage-2) + r'\,\text{Gyr}\leq \text{age} \leq' + str(thisage) +r'\,\text{Gyr}'+ '$'
            #axarr[i].set(title=r'${} \leq \text\{age\} \leq {}$'.format(thisage-2, thisage))
            axarr[i].set(title=mytitle)
            axarr[i].dist=12
            axarr[i].view_init(30, theta)
            

    fig.subplots_adjust(bottom=0.1)
    plt.savefig('movie/JrLzJz_agecut_'+"{0:0=3d}".format(i)+'.png', dpi=1200)
