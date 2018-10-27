# here we test different time steps to see
# what is necessary to get converging actions
#
# based on https://github.com/adrn/dr2-zero-day-happy-fun-time/blob/master/jay-z/Vertical-action-demo.ipynb
#
# Angus Beane - 5.30.18

import warnings
import sys

from astropy import table
from astropy.table import Table
from astropy.io import fits
import astropy.coordinates as coord
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
from pyia import GaiaData

import argparse
from joblib import Parallel, delayed
import multiprocessing

parser = argparse.ArgumentParser(description='Read in processors, data file.')
parser.add_argument('-i', help='input filename, fits', required=True)
parser.add_argument('-o', help='output filename, fits', required=True)
parser.add_argument('-p', default=1, type=int, help='number of processors, default=1')
parser.add_argument('-n', default=300, type=int, help='number of MC loops, default=300')
parser.add_argument('--dt', default=1, type=float, help='timestep in Myr, default=1 Myr')
parser.add_argument('--no-errors', action='store_true', help='do not compute action errors')
parser.add_argument('--node', type=int, help='node number')
parser.add_argument('--tot', type=int, help='total number of nodes', default=1)
args = parser.parse_args()


filein = args.i
fileout = args.o
nproc = args.p
nloop = args.n
dt = args.dt * u.Myr
noerr = args.no_errors
node = args.node
totnodes = args.tot

if(nproc > 1):
    print('multithreading activated, using ',nproc,' processors')

# defines reference frame
rsun = 8 * u.kpc
zsun = 0.025 * u.kpc
vsun = [11.1, 232.24, 7.25] * u.km/u.s
gc_frame = coord.Galactocentric(galcen_distance=rsun, galcen_v_sun=coord.CartesianDifferential(*vsun), z_sun=zsun)

# load in data, load MW potential
with warnings.catch_warnings(record=True):
    warnings.simplefilter("ignore")
    gaiadata = GaiaData(filein)
    hdul = fits.open(filein)
    sttable = Table(hdul[1].data)
    if(totnodes > 1):
        thiskeys = np.array_split(range(len(sttable['source_id'])),totnodes)[node]
        gaiadata = gaiadata[thiskeys]
        sttable = sttable[thiskeys]
    mw = gp.MilkyWayPotential()

print(sttable['source_id'][0])

# for the montecarlo loop later
cov = gaiadata.get_cov()
meanvals = [gaiadata.ra.value, gaiadata.dec.value, gaiadata.parallax.value, 
            gaiadata.pmra.value, gaiadata.pmdec.value, gaiadata.radial_velocity.value]
meanvals = list(map(list, zip(*meanvals))) # transpose

nentries = len(sttable)

# convert to galactocentric coordinates
sc = gaiadata.skycoord
scdyn = gd.PhaseSpacePosition(sc.transform_to(gc_frame).cartesian)
""" 
compute u, v, w
"""

distgalcenter = np.sqrt(scdyn.pos.x*scdyn.pos.x + scdyn.pos.y*scdyn.pos.y)
phi = np.arctan2(scdyn.pos.y, scdyn.pos.x)
z = scdyn.pos.z

cylndvel = scdyn.vel.represent_as(coord.CylindricalDifferential, base=scdyn.pos)
#rvel = cylndvel.d_rho.to(u.km / u.s)
uvel = -cylndvel.d_rho.to(u.km / u.s).value
vvel = -(distgalcenter*cylndvel.d_phi).to(u.km / u.s, equivalencies=u.dimensionless_angles()).value
#zvel = cylndvel.d_z.to(u.km/u.s)
wvel = cylndvel.d_z.to(u.km/u.s).value

cyl_pos = np.transpose([distgalcenter, phi, z])
cyl_vel = np.transpose([uvel, vvel, wvel])

"""
# now compute actions for different timesteps
"""
all_actions=[]

#calculate the actions!

def actions(star):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        orbit = mw.integrate_orbit(star, dt=dt, t1=0*u.Gyr, t2=5*u.Gyr, Integrator=gi.DOPRI853Integrator)
        res = gd.actionangle.find_actions(orbit, N_max=8)
        act = res['actions'].to(u.kpc * u.km / u.s).value
        angles = res['angles'].to(u.rad).value
        freqs = res['freqs'].to(1/u.Myr).value
        zmax = orbit.zmax(approximate=True).to(u.kpc).value
        return act, angles, freqs, zmax

def actionloop(star):
    try:
        thisact, thisangle, thisfreqs, zmax = actions(star)
        mask=True
    except:
        thisact = np.full(3, np.nan)
        thisangle = np.full(3, np.nan)
        thisfreqs = np.full(3, np.nan)
        zmax = np.nan
        mask=False
        pass
    return thisact, thisangle, thisfreqs, zmax, mask

print('computing the actions given best values of vel, pos...')
result = Parallel(n_jobs=nproc) (delayed(actionloop)(scdyn[i]) for i in tqdm(range(nentries)))
actions, angles, freqs, zmax, mask = np.transpose(result)

if(not noerr):
    def genmcgaiadata(i):
        #np.random.seed(seed)
        mcdata = []
        for j in range(nentries):
            mcdata.append(np.random.multivariate_normal(meanvals[j], cov[j]))
        mcdata = list(map(list, zip(*mcdata)))
        mctable = Table(mcdata, names=('ra','dec','parallax','pmra','pmdec','radial_velocity'))
        return GaiaData(mctable)
    
    print('now starting mc loop, this takes a while...')
    np.random.seed(1605)
    mcdata = []
    for i in tqdm(range(nloop)):
        mcgaiadata = genmcgaiadata(i)
        mcsc = mcgaiadata.skycoord
        mcdyn = gd.PhaseSpacePosition(mcsc.transform_to(gc_frame).cartesian)
        mcaction = Parallel(n_jobs=nproc) (delayed(actionloop)(mcdyn[k]) for k in range(nentries))
        mcdata.append(mcaction)

    #compute standard deviations
    Jr_err = []
    Lz_err = []
    Jz_err = []
    zmax_err = []
    for j in range(nentries):
        this_Jr_err = np.nanstd([mcdata[k][j][0] for k in range(len(mcdata))], ddof=1)
        Jr_err.append(this_Jr_err)
        this_Lz_err = np.nanstd([mcdata[k][j][1] for k in range(len(mcdata))], ddof=1)
        Lz_err.append(this_Lz_err)
        this_Jz_err = np.nanstd([mcdata[k][j][2] for k in range(len(mcdata))], ddof=1)
        Jz_err.append(this_Jz_err)
        this_zmax_err = np.nanstd([mcdata[k][j][3] for k in range(len(mcdata))], ddof=1)
        zmax_err.append(this_zmax_err)
        

#output
if(noerr):
    action_table = Table([sttable['source_id'], cyl_pos, cyl_vel, actions, angles, freqs, zmax], 
                         names=('source_id','cyl_pos', 'cyl_vel', 'actions', 'angles', 'freqs', 'zmax'),
                         mask=mask)
else:
    action_table = Table([sttable['source_id'],Jr,Jr_err,Lz,Lz_err,Jz,Jz_err,zmax,zmax_err,uvel,vvel,wvel], 
                         names=('source_id','Jr','Jr_err','Lz','Lz_err','Jz','Jz_err','zmax','zmax_err','uvel','vvel','wvel'),
                         mask=mask)
    action_table['Jr_err'].unit = u.kpc*u.km/u.s
    action_table['Lz_err'].unit = u.kpc*u.km/u.s
    action_table['Jz_err'].unit = u.kpc*u.km/u.s

action_table['cyl_pos'].unit = u.kpc
action_table['cyl_vel'].unit = u.km/u.s
action_table['actions'].unit = u.kpc*u.km/u.s
action_table['angles'].unit = u.rad
action_table['freqs'].unit = 1/u.Myr
action_table['zmax'].unit = u.kpc


output_table = table.join(sttable, action_table)

with warnings.catch_warnings(record=True):
    warnings.simplefilter("ignore")
    if(totnodes > 1):
        output_table.write("{0}-{2}.{1}".format(*fileout.rsplit('.', 1) + [node]), format='fits', overwrite=True)
    else:
        output_table.write(fileout, format='fits', overwrite=True)
