from astropy.io import fits
from astropy.table import Table

hdul = fits.open('GAIA-APOKASC2_wrongnames.fits')
mytable = Table(hdul[1].data)
mytable.rename_column('vhelio_avg', 'radial_velocity')
mytable.rename_column('verr', 'radial_velocity_error')

mytable.write('GAIA-APOKASC2.fits',format='fits')
