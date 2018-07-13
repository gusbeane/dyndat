import numpy as np
from astropy.io import fits
from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u
import argparse

import pandas as pd
import gala.dynamics as gd
from pyia import GaiaData

import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

parser = argparse.ArgumentParser(description='Read in processors, data file.')
parser.add_argument('--fg', help='input foreground stars with neighbors, pickle', required=True)
parser.add_argument('--bg', help='input background stars with actions, fits', required=True)
parser.add_argument('-o', help='output fits file, one column for each star', required=True)
parser.add_argument('--cuts', help='text file containing rcuts, if two columns will do annuli instead', required=True)
parser.add_argument('-p', default=1, type=int, help='number of processors, default=1')
parser.add_argument('--norm', action='store_true', help='compute the normalized peculiar actions, default=no')
args = parser.parse_args()

filefg = args.fg
filebg = args.bg
fileout = args.o
cutfile = args.cuts
nproc = args.p
norm = args.norm

cuts = np.genfromtxt(cutfile)
if(len(np.concatenate(cuts))==2):
    print("WARNING: only two cuts detected in the input file")
    print("this will NOT be treated as annuli, even if horizontal")
    print("add a second annuli line if you want an annulus")
if(len(cuts[0]) == 2):
    annuli = True
else:
    annuli = False

# define some stuff about the peculiar actions, will make user definable later


#define reference frame
rsun = 8 * u.kpc
zsun = 0.025 * u.kpc
vsun = [11.1, 232.24, 7.25] * u.km/u.s
gc_frame = coord.Galactocentric(galcen_distance=rsun, galcen_v_sun=coord.CartesianDifferential(*vsun), z_sun=zsun)

with warnings.catch_warnings(record=True):
    warnings.simplefilter("ignore")
    # open file
    fgpckl = pd.read_pickle(filefg)
    bghdul = fits.open(filebg)

    # extract table
    fgtable = Table.from_pandas(fgpckl)
    #fgtable = fgtable[0:3]
    bgtable = Table(bghdul[1].data)
    

# create GaiaData object
fggaia = GaiaData(fgtable)
bggaia = GaiaData(bgtable)

# convert to sky coordinates
fgsc = fggaia.skycoord
bgsc = bggaia.skycoord

# convert to gc coordinates
fggc = gd.PhaseSpacePosition(fgsc.transform_to(gc_frame).cartesian)
bggc = gd.PhaseSpacePosition(bgsc.transform_to(gc_frame).cartesian)

# now loop through dcuts
def nbactions(fgkey):
    nbkeys = fgtable['nb_keys'][fgkey]
    nbrad = fgtable['nb_rad'][fgkey]
    fgJr = fgtable['Jr'][fgkey]
    fgLz = fgtable['Lz'][fgkey]
    fgJz = fgtable['Jz'][fgkey]
    output = []
    for d in cuts:
        if(annuli):
            internalkeys = np.where(np.logical_and(fgtable['nb_rad'][fgkey] >= d[0], fgtable['nb_rad'][fgkey] < d[1]))
        else:
            internalkeys = np.where(nbrad <= d)[0]
        
        radkeys = nbkeys[internalkeys]
        radmean = np.nanmean(nbrad[internalkeys])
        radstd = np.nanstd(nbrad[internalkeys], ddof=1)
        
        nbJrarray = bgtable['Jr'][radkeys]
        nbLzarray = bgtable['Lz'][radkeys]
        nbJzarray = bgtable['Jz'][radkeys]
        
        nbJr = np.nanmean(nbJrarray)
        nbJrstd = np.nanstd(nbJrarray, ddof=1)
        nbLz = np.nanmean(nbLzarray)
        nbLzstd = np.nanstd(nbLzarray, ddof=1)
        nbJz = np.nanmean(nbJzarray)
        nbJzstd = np.nanstd(nbJzarray, ddof=1)
        
        if(norm):
            output.append([radmean, radstd, (fgJr - nbJr)/nbJr, nbJrstd, (fgLz - nbLz)/nbLz, nbJzstd, (fgJz - nbJz)/nbJz, nbJrstd, len(radkeys) ])
        else:
            output.append([radmean, radstd, (fgJr - nbJr), nbJrstd, (fgLz - nbLz), nbJzstd, (fgJz - nbJz), nbJrstd, len(radkeys) ])
    
    return output

if(nproc > 1):
    result = Parallel(n_jobs=nproc) (delayed(nbactions)(i) for i in tqdm(range(len(fggc.x))))
else:
    result = []
    for i in tqdm(range(len(fggc.x))):
        result.append(nbactions(i))

#result = list(map(list, zip(*result)))
output = [cuts] + result
if(annuli):
    colnames = ['annuli'] + fgtable['source_id'].tolist()
else:
    colnames = ['dcut'] + fgtable['source_id'].tolist()
output_table = Table(output, names=colnames)
output_table.write(fileout, format='fits', overwrite=True)
