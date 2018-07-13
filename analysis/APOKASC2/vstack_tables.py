from astropy.io import fits
from astropy.table import vstack
from astropy.table import Table
import sys

outfile = sys.argv[1]
infiles = sys.argv[2:]

stackme = []
for filename in infiles:
    hdul = fits.open(filename)
    stackme.append(Table(hdul[1].data))
vstack(stackme).write(outfile, format='fits', overwrite=True)
