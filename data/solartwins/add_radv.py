from astropy.io import fits
from astropy.io import ascii
from astropy import table as at
import numpy as np

STdata = ascii.read('GAIA-ST_fromMegan_noradv.csv')
radvdata = ascii.read('STradv.csv')

print(len(STdata))

#radvdata.rename_column('\ufeffHIP','HIP')
radvdata['HIP'] = radvdata['HIP'].astype(str)


hiparray = []
for i in range(len(radvdata['HIP'])):
    hiparray.append('HIP'+str(radvdata['HIP'][i]))
radvdata.remove_column('HIP')
radvdata['HIP'] = np.array(hiparray)

jdata = at.join(STdata, radvdata, keys='HIP')
print(len(jdata))

for i in range(len(jdata['radial_velocity'])):
    if(jdata['radial_velocity'][i][0] == '_'):
        jdata['radial_velocity'][i] = jdata['radial_velocity'][i].replace('_','-')

radvs = jdata['radial_velocity'].astype(float)
jdata.remove_column('radial_velocity')
jdata['radial_velocity'] = radvs
jdata['radial_velocity_error'] = jdata['radial_velocity_error'].astype(float)

jdata.write('GAIA-ST.fits', format='fits')
