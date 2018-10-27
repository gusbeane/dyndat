# here we test different time steps to see
# what is necessary to get converging actions
#
# based on https://github.com/adrn/dr2-zero-day-happy-fun-time/blob/master/jay-z/Vertical-action-demo.ipynb
#
# Angus Beane - 5.30.18

import warnings
import sys

from astropy import table
from astropy.table import Table, Column, MaskedColumn
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
        mask=False
    except:
        thisact = np.full(3, np.nan)
        thisangle = np.full(3, np.nan)
        thisfreqs = np.full(3, np.nan)
        zmax = np.nan
        mask=True
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
    mcactions = []
    mcangles = []
    mcfreqs = []
    mczmax = []
    for i in tqdm(range(nloop)):
        mcgaiadata = genmcgaiadata(i)
        mcsc = mcgaiadata.skycoord
        mcdyn = gd.PhaseSpacePosition(mcsc.transform_to(gc_frame).cartesian)
        result = Parallel(n_jobs=nproc) (delayed(actionloop)(mcdyn[k]) for k in range(nentries))
        actions, angles, freqs, zmax, mask = np.transpose(result)
        mcactions.append(actions)
        mcangles.append(angles)
        mcfreqs.append(freqs)
        mczmax.append(zmax)

    # compute covariances

    # actions

    JrLz = np.array([np.cov(mcactions[:,i,0], mcactions[:,i,1]) for i in range(nentries) ])
    LzJz = np.array([np.cov(mcactions[:,i,1], mcactions[:,i,2]) for i in range(nentries) ])
    JzJr = np.array([np.cov(mcactions[:,i,2], mcactions[:,i,0]) for i in range(nentries) ])

    Jrvar = np.array([JrLz[i][0][0] for i in range(nentries)])
    JrLzcov = np.array([JrLz[i][0][1] for i in range(nentries)])
    Lzvar = np.array([LzJz[i][0][0] for i in range(nentries)])
    LzJzcov = np.array([LzJz[i][0][1] for i in range(nentries)])
    Jzvar = np.array([JzJr[i][0][0] for i in range(nentries)])
    JzJrcov = np.array([JzJr[i][0][1] for i in range(nentries)])
    mask_act = Jrvar == np.nan

    # angles
    rphi = np.array([np.cov(mcangles[:,i,0], mcangles[:,i,1]) for i in range(nentries) ])
    phiz = np.array([np.cov(mcangles[:,i,1], mcangles[:,i,2]) for i in range(nentries) ])
    zr = np.array([np.cov(mcangles[:,i,2], mcangles[:,i,0]) for i in range(nentries) ])

    angle_rvar = np.array([rphi[i][0][0] for i in range(nentries)])
    angle_rzcov = np.array([rphi[i][0][1] for i in range(nentries)])
    angle_phivar = np.array([phiz[i][0][0] for i in range(nentries)])
    angle_phizcov = np.array([phiz[i][0][1] for i in range(nentries)])
    angle_zvar = np.array([zr[i][0][0] for i in range(nentries)])
    angle_zrcovar= np.array([zrp[i][0][1] for i in range(nentries)])
    mask_angle = angle_rvar == np.nan

    # freqs
    rphi = np.array([np.cov(mcfreqs[:,i,0], mcfreqs[:,i,1]) for i in range(nentries) ])
    phiz = np.array([np.cov(mcfreqs[:,i,1], mcfreqs[:,i,2]) for i in range(nentries) ])
    zr = np.array([np.cov(mcfreqs[:,i,2], mcfreqs[:,i,0]) for i in range(nentries) ])

    freq_rvar = np.array([rphi[i][0][0] for i in range(nentries)])
    freq_rzcov = np.array([rphi[i][0][1] for i in range(nentries)])
    freq_phivar = np.array([phiz[i][0][0] for i in range(nentries)])
    freq_phizcov = np.array([phiz[i][0][1] for i in range(nentries)])
    freq_zvar = np.array([zr[i][0][0] for i in range(nentries)])
    freq_zrcovar= np.array([zrp[i][0][1] for i in range(nentries)])
    mask_freq = freq_rvar == np.nan

# i dont know why this is necessary
actions = np.array([actions[i].tolist() for i in range(len(actions))])
angles = np.array([angles[i].tolist() for i in range(len(actions))])
freqs = np.array([freqs[i].tolist() for i in range(len(actions))])

#output
sid = Column(sttable['source_id'], name='source_id', dtype='i8')
r_c = MaskedColumn(cyl_pos[:,0], name='cyl_pos.r', mask=mask, dtype='f8')
phi_c = MaskedColumn(cyl_pos[:,1], name='cyl_pos.phi', mask=mask, dtype='f8')
z_c = MaskedColumn(cyl_pos[:,2], name='cyl_pos.z', mask=mask, dtype='f8')
vr_c = MaskedColumn(cyl_vel[:,0], name='cyl_vel.r', mask=mask, dtype='f8')
vphi_c = MaskedColumn(cyl_vel[:,1], name='cyl_vel.phi', mask=mask, dtype='f8')
vz_c = MaskedColumn(cyl_vel[:,2], name='cyl_vel.z', mask=mask, dtype='f8')
Jr_c = MaskedColumn(actions[:,0], name='action.Jr', mask=mask, dtype='f8')
Lz_c = MaskedColumn(actions[:,1], name='action.Lz', mask=mask, dtype='f8')
Jz_c = MaskedColumn(actions[:,2], name='action.Jz', mask=mask, dtype='f8')
angle_r_c = MaskedColumn(angles[:,0], name='angle.r', mask=mask, dtype='f8')
angle_phi_c = MaskedColumn(angles[:,1], name='angle.phi', mask=mask, dtype='f8')
angle_z_c = MaskedColumn(angles[:,2], name='angle.z', mask=mask, dtype='f8')
freqs_r_c = MaskedColumn(freqs[:,0], name='freq.r', mask=mask, dtype='f8')
freqs_phi_c = MaskedColumn(freqs[:,1], name='freq.phi', mask=mask, dtype='f8')
freqs_z_c = MaskedColumn(freqs[:,2], name='freq.z', mask=mask, dtype='f8')
zmax_c = MaskedColumn(zmax, name='zmax', mask=mask, dtype='f8')
if noerr: 
    # action_table = Table([sttable['source_id'], cyl_pos, cyl_vel, actions, angles, freqs, zmax], 
    #                      names=('source_id','cyl_pos', 'cyl_vel', 'actions', 'angles', 'freqs', 'zmax'),
    #                      mask=mask)
    action_table = Table([sid, r_c, phi_c, z_c, vr_c, vphi_c, vz_c, Jr_c, Lz_c, Jz_c,
                          angle_r_c, angle_phi_c, angle_z_c, freqs_r_c, freqs_phi_c, freqs_z_c, zmax_c])
else:
    mask = np.logical_or(np.logical_or(mask_angle, mask_act), mask_freq)
    Jrvar_c = MaskedColumn(Jrvar, name='action.Jr.var', mask=mask, dtype='f8')
    JrLzcov_c = MaskedColumn(JrLzcov, name='action.JrLz.cov', mask=mask, dtype='f8')
    Lzvar_c = MaskedColumn(Lzvar, name='action.Lz.var', mask=mask, dtype='f8')
    LzJzcov_c = MaskedColumn(LzJzcov, name='action.LzJz.cov', mask=mask, dtype='f8')
    Jzvar_c = MaskedColumn(Jzvar, name='action.Jz.var', mask=mask, dtype='f8')
    JzJrcov_c = MaskedColumn(JzJrcov, name='action.JzJr.cov', mask=mask, dtype='f8')
    angle_rvar_c = MaskedColumn(angle_rvar, name='angle.r.var', mask=mask, dtype='f8')
    angle_rzcov_c = MaskedColumn(angle_rzcov, name='angle.rphi.cov', mask=mask, dtype='f8')
    angle_phivar_c = MaskedColumn(angle_phivar, name='angle.phi.var', mask=mask, dtype='f8')
    angle_phizcov_c = MaskedColumn(angle_phizcov, name='angle.phiz.cov', mask=mask, dtype='f8')
    angle_zvar_c = MaskedColumn(angle_zvar, name='angle.z.var', mask=mask, dtype='f8')
    angle_zrcovar_c = MaskedColumn(angle_zrcovar, name='angle.zr.cov', mask=mask, dtype='f8')
    freq_rvar_c = MaskedColumn(freq_rvar, name='freq.r.var', mask=mask, dtype='f8')
    freq_rzcov_c = MaskedColumn(freq_rzcov, name='freq.rphi.cov', mask=mask, dtype='f8')
    freq_phivar_c = MaskedColumn(freq_phivar, name='freq.phi.var', mask=mask, dtype='f8')
    freq_phizcov_c = MaskedColumn(freq_phizcov, name='freq.phiz.cov', mask=mask, dtype='f8')
    freq_zvar_c = MaskedColumn(freq_zvar, name='freq.z.var', mask=mask, dtype='f8')
    freq_zrcovar_c = MaskedColumn(freq_zrcovar, name='freq.zr.covar', mask=mask, dtype='f8')

    action_table = Table([sttable['source_id'],Jr,Jr_err,Lz,Lz_err,Jz,Jz_err,zmax,zmax_err,uvel,vvel,wvel], 
                         names=('source_id','Jr','Jr_err','Lz','Lz_err','Jz','Jz_err','zmax','zmax_err','uvel','vvel','wvel'),
                         mask=mask)
    action_table['action.Jr.var'].unit = (u.kpc * u.km/u.s)**2
    action_table['action.JrLz.cov'].unit = (u.kpc * u.km/u.s)**2
    action_table['action.Lz.var'].unit = (u.kpc * u.km/u.s)**2
    action_table['action.LzJz.cov'].unit = (u.kpc * u.km/u.s)**2
    action_table['action.Jz.var'].unit = (u.kpc * u.km/u.s)**2
    action_table['action.JzJr.cov'].unit = (u.kpc * u.km/u.s)**2
    action_table['angle.r.var'].unit = u.kpc**2
    action_table['angle.rphi.cov'].unit = u.kpc*u.rad
    action_table['angle.phi.var'].unit = u.rad**2
    action_table['angle.phiz.cov'].unit = u.rad*u.kpc
    action_table['angle.z.var'].unit = u.kpc**2
    action_table['angle.zr.cov'].unit = u.kpc**2
    action_table['freq.r.var'].unit = (1/u.Myr)**2
    action_table['freq.rphi.cov'].unit = (1/u.Myr)**2
    action_table['freq.phi.var'].unit = (1/u.Myr)**2
    action_table['freq.phiz.cov'].unit = (1/u.Myr)**2
    action_table['freq.z.var'].unit = (1/u.Myr)**2
    action_table['freq.zr.covar'].unit = (1/u.Myr)**2

action_table['cyl_pos.r'].unit = u.kpc
action_table['cyl_pos.phi'].unit = u.rad
action_table['cyl_pos.z'].unit = u.kpc
action_table['cyl_vel.r'].unit = u.km/u.s
action_table['cyl_vel.phi'].unit = u.km/u.s
action_table['cyl_vel.z'].unit = u.km/u.s
action_table['action.Jr'].unit = u.kpc*u.km/u.s
action_table['action.Lz'].unit = u.kpc*u.km/u.s
action_table['action.Jz'].unit = u.kpc*u.km/u.s
action_table['angle.r'].unit = u.rad
action_table['angle.phi'].unit = u.rad
action_table['angle.z'].unit = u.rad
action_table['freq.r'].unit = 1/u.Myr
action_table['freq.phi'].unit = 1/u.Myr
action_table['freq.z'].unit = 1/u.Myr
action_table['zmax'].unit = u.kpc


output_table = table.join(sttable, action_table)

with warnings.catch_warnings(record=True):
    warnings.simplefilter("ignore")
    if(totnodes > 1):
        output_table.write("{0}-{2}.{1}".format(*fileout.rsplit('.', 1) + [node]), format='fits', overwrite=True)
    else:
        output_table.write(fileout, format='fits', overwrite=True)
