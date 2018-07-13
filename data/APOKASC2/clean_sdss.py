import numpy as np
from astropy.table import Table

mytable = np.genfromtxt('MyTable_0_abeane_2.csv', dtype=None, delimiter=',', names=True)

tmassids = []
for i in range(len(mytable)):
    tmassids.append(mytable[i][0])
    mytable[i][6] = mytable[i][6][2:]

ct = 0
delme = []
for i in range(len(tmassids)):
    for j in range(i):
        if(tmassids[i] == tmassids[j] and i != j):
            delme.append(i)
            delme.append(j)

outarray = np.delete(mytable, delme)
writeme = Table(outarray)
writeme.write('apokasc2_radvel.vots', format='votable')
