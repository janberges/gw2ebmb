#!/usr/bin/env python3

import numpy as np

filename = 'dat.JvsE.%03d-%03d'
pdata = np.loadtxt('a2fp.txt')

we = np.loadtxt(filename % (1, 1), skiprows=2, usecols=0).T
wp = pdata[:, 0]
wep = np.concatenate((wp, we[we > wp[-1]]))

norb = int(np.sqrt(pdata.shape[1] - 1))

a2Fe = np.empty((len(wep), norb, norb))
a2Fp = pdata[:, 1:].reshape((len(wp), norb, norb))

muC = np.empty((norb, norb))

for a in range(norb):
    for b in range(a, norb):
        w, re, im = np.loadtxt(filename % (a + 1, b + 1),
            skiprows=2, usecols=(0, 2, 3)).T

        a2Fe[:, a, b] = -np.interp(wep, w, im) / np.pi
        a2Fe[:, b, a] = a2Fe[:, a, b]

        muC[a, b] = re[-1]
        muC[b, a] = muC[a, b]

a2Fep = a2Fe.copy()
a2Fep[:len(wp)] += a2Fp

for name, a2F in ('a2fe.txt', a2Fe), ('a2fep.txt', a2Fep):
    with open(name, 'w') as data:
        for i in range(len(wep)):
            if np.any(a2F[max(0, i - 1):i + 2] > 5e-7):
                data.write(('%10.6f' + norb * norb * ' %8.6f' + '\n')
                    % (wep[i], *a2F[i].ravel()))

np.savetxt('muc.txt', muC, fmt='%9.6f')
