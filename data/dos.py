#!/usr/bin/env python3

import elphmod
import numpy as np

comm = elphmod.MPI.comm

nk = 150
dw = 5e-3
cutoff = 7.0

kT = 300 * elphmod.misc.kB
f = elphmod.occupations.fermi_dirac

k = np.array(sorted(elphmod.bravais.irreducibles_ibrav(nk, nk, nk, ibrav=7)))

weights = np.array([len(elphmod.bravais.images_ibrav(k1, k2, k3, nk, nk, nk,
    ibrav=7)) for k1, k2, k3 in k])

k = 2 * np.pi * k / nk

el = elphmod.el.Model('sro')

e, U = elphmod.dispersion.dispersion(el.H, k, vectors=True, shared_memory=True)

e = e.ravel()

U2 = abs(U[..., 0::2, :]) ** 2 + abs(U[..., 1::2, :]) ** 2
U2 *= weights[..., np.newaxis, np.newaxis]

weight = 0.5 * np.moveaxis(U2, -2, 0).reshape((3, -1))

buf = cutoff * kT

w = np.arange(np.ceil((e.min() - buf) / dw) * dw, e.max() + buf, dw)

sizes, bounds = elphmod.MPI.distribute(len(w), bounds=True)

my_DOS = np.empty((sizes[comm.rank], 3))

status = elphmod.misc.StatusBar(sizes[comm.rank], title='calculate DOS')

for my_iw, iw in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
    delta = f.delta((w[iw] - e) / kT) / kT

    my_DOS[my_iw] = np.sum(delta * weight, axis=1)

    status.update()

my_DOS /= nk ** 3

DOS = np.empty((len(w), 3))

comm.Gatherv(my_DOS, (DOS, comm.allgather(my_DOS.size)))

if comm.rank == 0:
    DOSsym = []

    for S in elphmod.bravais.symmetries_ibrav(7, cartesian=True):
        St2g = np.zeros((3, 3))
        St2g[:2, :2] = S[:2, :2] * S[2, 2]
        St2g[2, 2] = S[0, 0] * S[1, 1] + S[0, 1] * S[1, 0]

        DOSsym.append(St2g.T @ (DOS[:, :, np.newaxis] * np.eye(3)) @ St2g)

    DOS = np.average(DOSsym, axis=0)
    DOS = np.diagonal(DOS, axis1=1, axis2=2)

    with open('dos.txt', 'w') as data:
        for iw in range(len(w)):
            data.write(('%6.3f' + 3 * ' %8.6f' + '\n')
                % (w[iw], *DOS[iw]))
