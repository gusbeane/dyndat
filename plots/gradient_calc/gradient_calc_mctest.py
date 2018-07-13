import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import warnings
from copy import copy

from tqdm import tqdm
from scipy import stats

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

Jrerr = APOKASCtable['Jr_err']
Lzerr = APOKASCtable['Lz_err']
Jzerr = APOKASCtable['Jz_err']

STJr = STtable['Jr']
STLz = np.abs(STtable['Lz'])
STJz = STtable['Jz']
STage = STtable['age_mean']

age = APOKASCtable['log_age']
ageplus = APOKASCtable['s_logagep']
ageminus = APOKASCtable['s_logagem']
agep = age+ageplus
agem = age-ageminus

age = (10.**age)/1000.0
agep = (10.**agep)/1000.0
agem = (10.**agem)/1000.0
ageunc = np.maximum(agep-age, age-agem)


plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

#Jr = np.sqrt(Jr)
#STJr = np.sqrt(STJr)
#Jz = np.sqrt(Jz)
#STJz = np.sqrt(STJz)

Jrkey = np.logical_and(Jr > 0, Jr < 1000)

age = age[np.where(Jrkey)[0]]
ageunc = ageunc[np.where(Jrkey)[0]]
Jzerr = Jzerr[np.where(Jrkey)[0]]
Jz = Jz[np.where(Jrkey)[0]]
Lzerr = Lzerr[np.where(Jrkey)[0]]
Lz = Lz[np.where(Jrkey)[0]]
Jrerr = Jrerr[np.where(Jrkey)[0]]
Jr = Jr[np.where(Jrkey)[0]]



print(len(age))

lowagekeys = np.logical_and(age < 2, age > 0)
highagekeys = np.logical_and(age < 11, age > 9)

nankeys = np.logical_not(np.isnan(Jr))

lowagekeys = np.where(np.logical_and(nankeys, lowagekeys))[0]
highagekeys = np.where(np.logical_and(nankeys, highagekeys))[0]

meanlowage = np.nanmean(age[lowagekeys])
meanhighage = np.nanmean(age[highagekeys])

meanJrlow = np.nanmean(Jr[lowagekeys])
meanJrhigh = np.nanmean(Jr[highagekeys])

meanLzlow = np.nanmean(Lz[lowagekeys])
meanLzhigh = np.nanmean(Lz[highagekeys])

meanJzlow = np.nanmean(Jz[lowagekeys])
meanJzhigh = np.nanmean(Jz[highagekeys])

dage = meanhighage - meanlowage
print("mean low age:", meanlowage)
print("mean high age:", meanhighage)

dJr = meanJrhigh - meanJrlow
dLz = meanLzhigh - meanLzlow
dJz = meanJzhigh - meanJzlow

print("dJr: ", dJr, "dLz: ", dLz, "dJz: ", dJz)

print("Jr grad: ", dJr/dage, "Lz grad: ", dLz/dage, "Jz grad: ", dJz/dage)

#now error bars

Jrgrad = [dJr/dage]
Lzgrad = [dLz/dage]
Jzgrad = [dJz/dage]

np.random.seed(1776)

Jruncert = []
Lzuncert = []
Jzuncert = []
for i in tqdm(range(10000)):
    newages = np.random.normal(loc=age, scale=ageunc)
    newJr = np.random.normal(loc=Jr, scale=Jrerr)
    newLz = np.random.normal(loc=Lz, scale=Lzerr)
    newJz = np.random.normal(loc=Jz, scale=Jzerr)
    
    lowagekeys = np.logical_and(newages < 2, newages > 0)
    highagekeys = np.logical_and(newages < 11, newages > 9)

    nankeys = np.logical_not(np.isnan(Jr))

    lowagekeys = np.where(np.logical_and(nankeys, lowagekeys))[0]
    highagekeys = np.where(np.logical_and(nankeys, highagekeys))[0]
    
    meanlowage = np.nanmean(age[lowagekeys])
    meanhighage = np.nanmean(age[highagekeys])
    
    meanJrlow = np.nanmean(Jr[lowagekeys])
    meanJrhigh = np.nanmean(Jr[highagekeys])
    
    meanLzlow = np.nanmean(Lz[lowagekeys])
    meanLzhigh = np.nanmean(Lz[highagekeys])
    
    meanJzlow = np.nanmean(Jz[lowagekeys])
    meanJzhigh = np.nanmean(Jz[highagekeys])
    
    dage = meanhighage - meanlowage
    
    dJr = meanJrhigh - meanJrlow
    dLz = meanLzhigh - meanLzlow
    dJz = meanJzhigh - meanJzlow
    
    Jrgrad.append(dJr/dage)
    Lzgrad.append(dLz/dage)
    Jzgrad.append(dJz/dage)

    Jruncert.append(np.nanstd(Jrgrad, ddof=1))
    Lzuncert.append(np.nanstd(Lzgrad, ddof=1))
    Jzuncert.append(np.nanstd(Jrgrad, ddof=1))

plt.plot(range(10000), np.abs(Jruncert - Jruncert[len(Jruncert)-1])/Jruncert[len(Jruncert)-1])
plt.plot(range(10000), np.abs((Lzuncert - Lzuncert[len(Lzuncert)-1])/Lzuncert[len(Lzuncert)-1]))
plt.plot(range(10000), np.abs(Jzuncert - Jzuncert[len(Jzuncert)-1])/Jzuncert[len(Jzuncert)-1])
plt.ylim([1e-5,10])
plt.yscale('log')
plt.show()
