APOKASC2_Table5.txt is downloaded from https://www.dropbox.com/s/k33td8ukefwy5tv/APOKASC2_Table5.txt?dl=0
https://arxiv.org/abs/1804.09983

APOKASC2_Table5_filtered.txt is the above file, but all rejects are filtered via this command:
cat APOKASC2_Table5.txt | grep -v REJECT | grep -v Bad | grep -v 'No Fdnu' | grep -v 'No Seis' > APOKASC2_Table5_filtered.txt

Also manually changed the colnames and removed l. 331 by hand (bad data for some reason)

Then uploaded to SDSS to get apogee RV's from DR14 ( deleted the Note column...)
When I downloaded, there was 74 duplicate entries in DR14, so I deleted these rows (since there's so few)

then the following queries were made on the GAIA database:

upload the .vots file as apokasctmassid (6676 rows)
first, perform the 2mass cross match

SELECT * FROM gaiadr2.tmass_best_neighbour AS g
JOIN user_abeane.apokasctmassid ON g.original_ext_source_id=user_abeane.apokasctmassid.col1

then, save the resulting table as apokasctgaia (6447 rows)
then, do some quality cuts to get accurate actions

SELECT * FROM gaiadr2.gaia_source AS g
JOIN user_abeane.apokasctgaia AS a ON a.source_id=g.source_id
WHERE (g.radial_velocity != 'nan'
AND parallax > 0.0
AND parallax_error/parallax<0.05
AND astrometric_excess_noise_sig < 2
AND SQRT( POWER(radial_velocity_error*radial_velocity/SQRT(radial_velocity*radial_velocity + POWER(4.7623/parallax,2) * (pmra*pmra + pmdec*pmdec)),2) + POWER(parallax_error * ((pmra*pmra + pmdec*pmdec)*POWER(4.7623/parallax,2)/parallax)/SQRT(radial_velocity*radial_velocity + POWER(4.7623/parallax,2) * (pmra*pmra + pmdec*pmdec)) ,2) + POWER(pmra_error * pmra * POWER(4.7623/parallax,2)/SQRT(radial_velocity*radial_velocity + POWER(4.7623/parallax,2) * (pmra*pmra + pmdec*pmdec)) ,2) + POWER(pmdec_error * pmdec * POWER(4.7632/parallax,2)/SQRT(radial_velocity*radial_velocity + POWER(4.7623/parallax,2) * (pmra*pmra + pmdec*pmdec)) ,2) ) <= 8 )

download this table as GAIA-APOKASC2.fits (4376 rows)
