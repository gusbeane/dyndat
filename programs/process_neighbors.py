import numpy as np
from astropy.io import fits
from astropy import table
from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u
import argparse

import gala.dynamics as gd
from pyia import GaiaData

import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

parser = argparse.ArgumentParser(description='Read in processors, data file.')
parser.add_argument('--fg', help='input foreground stars, fits', required=True)
parser.add_argument('--bg', help='input background stars, fits', required=True)
parser.add_argument('-o', help='output nb file, pickle', required=True)
parser.add_argument('--rcut', default=10., type=float, help='max neighbor radius (pc), default=10')
parser.add_argument('-p', default=1, type=int, help='number of processors, default=1')
args = parser.parse_args()

filefg = args.fg
filebg = args.bg
fileout = args.o
rcut = args.rcut
nproc = args.p

if(rcut < 5):
    print("WARNING: rcut is smaller than 5pc, some stars may have no neighbors...")

# define some stuff about the peculiar actions, will make user definable later

#define reference frame
rsun = 8 * u.kpc
zsun = 0.025 * u.kpc
vsun = [11.1, 232.24, 7.25] * u.km/u.s
gc_frame = coord.Galactocentric(galcen_distance=rsun, galcen_v_sun=coord.CartesianDifferential(*vsun), z_sun=zsun)

with warnings.catch_warnings(record=True):
    warnings.simplefilter("ignore")
    # open file
    fghdul = fits.open(filefg)
    bghdul = fits.open(filebg)

    # extract table
    fgtable = Table(fghdul[1].data)
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

fgx = fggc.x.to(u.pc).value
fgy = fggc.y.to(u.pc).value
fgz = fggc.z.to(u.pc).value
bgx = bggc.x.to(u.pc).value
bgy = bggc.y.to(u.pc).value
bgz = bggc.z.to(u.pc).value

# loop through foreground stars,
# first define each step of the loop
# each step will compute the dist between the fg stars and bg stars,
# then filter out by rcut
def rcut_keys(fgkey):
    dx = bgx - fgx[fgkey]
    dy = bgy - fgy[fgkey]
    dz = bgz - fgz[fgkey]
    dist = np.sqrt(dx*dx + dy*dy + dz*dz)
    outkeys = np.where(dist <= rcut)[0]
    return (outkeys, dist[outkeys])

if(nproc > 1):
    result = Parallel(n_jobs=nproc) (delayed(rcut_keys)(i) for i in tqdm(range(len(fgx))))
else:
    result = []
    for i in tqdm(range(len(fgx))):
        result.append(rcut_keys(i))

#transpose result table
result = list(map(list, zip(*result)))

keys_table = Table([fgtable['source_id'], result[0], result[1]], names=('source_id', 'nb_keys', 'nb_rad'))
keys_table['nb_rad'].unit = u.pc

output_table = table.join(fgtable, keys_table)
output_table = output_table.to_pandas()
output_table.to_pickle(fileout, compression=None)
