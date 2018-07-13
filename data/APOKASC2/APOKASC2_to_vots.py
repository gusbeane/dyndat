from astropy.table import Table
t = Table.read('APOKASC2_Table5_filtered.txt', format='ascii')
t.write('APOKASC2_Table5_filtered.vots', format='votable')
