from astropy.io import fits
from astropy.table import Table

hdul = fits.open('GAIA-ST_fromMegan.fits')
mytable = Table(hdul[1].data)

mytable.remove_columns(['radial_velocity','radial_velocity_error'])
mytable.write('GAIA-ST_fromMegan_noradv.csv', format='csv')
