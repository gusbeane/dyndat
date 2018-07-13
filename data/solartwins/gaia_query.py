import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity
from astroquery.gaia import Gaia
from astropy.io import fits

job = Gaia.launch_job_async("SELECT \
* \
FROM gaiadr2.gaia_source \
WHERE ( ((1000.0/parallax)<=152) \
AND parallax > 0.0 \
AND radial_velocity != 'nan'\
AND parallax_error/parallax<0.05 \
AND astrometric_excess_noise_sig < 2 \
AND SQRT( POWER(radial_velocity_error*radial_velocity/SQRT(radial_velocity*radial_velocity + POWER(4.7623/parallax,2) * (pmra*pmra + pmdec*pmdec)),2) + \
POWER(parallax_error * ((pmra*pmra + pmdec*pmdec)*POWER(4.7623/parallax,2)/parallax)/SQRT(radial_velocity*radial_velocity + POWER(4.7623/parallax,2) * (pmra*pmra + pmdec*pmdec)) ,2) + \
POWER(pmra_error * pmra * POWER(4.7623/parallax,2)/SQRT(radial_velocity*radial_velocity + POWER(4.7623/parallax,2) * (pmra*pmra + pmdec*pmdec)) ,2) + \
POWER(pmdec_error * pmdec * POWER(4.7632/parallax,2)/SQRT(radial_velocity*radial_velocity + POWER(4.7623/parallax,2) * (pmra*pmra + pmdec*pmdec)) ,2) ) <= 8 \
)",\
dump_to_file=True, output_format='fits', output_file='ST_bg_152pc.fits')

job = Gaia.launch_job_async("SELECT \
* \
FROM gaiadr2.gaia_source \
WHERE ( ((1000.0/parallax)<=210) \
AND parallax > 0.0 \
AND radial_velocity != 'nan'\
AND parallax_error/parallax<0.05 \
AND astrometric_excess_noise_sig < 2 \
AND SQRT( POWER(radial_velocity_error*radial_velocity/SQRT(radial_velocity*radial_velocity + POWER(4.7623/parallax,2) * (pmra*pmra + pmdec*pmdec)),2) + \
POWER(parallax_error * ((pmra*pmra + pmdec*pmdec)*POWER(4.7623/parallax,2)/parallax)/SQRT(radial_velocity*radial_velocity + POWER(4.7623/parallax,2) * (pmra*pmra + pmdec*pmdec)) ,2) + \
POWER(pmra_error * pmra * POWER(4.7623/parallax,2)/SQRT(radial_velocity*radial_velocity + POWER(4.7623/parallax,2) * (pmra*pmra + pmdec*pmdec)) ,2) + \
POWER(pmdec_error * pmdec * POWER(4.7632/parallax,2)/SQRT(radial_velocity*radial_velocity + POWER(4.7623/parallax,2) * (pmra*pmra + pmdec*pmdec)) ,2) ) <= 8 \
)",\
dump_to_file=True, output_format='fits', output_file='ST_bg_210pc.fits')
