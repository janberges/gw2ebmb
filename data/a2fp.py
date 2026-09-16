#!/usr/bin/env python3

import ebmb
import elphmod
import numpy as np

comm = elphmod.MPI.comm

nw = 224

if comm.rank == 0:
    omega = ebmb.get(upper=0.1033, points=nw, logscale=10, tell=False)['omega']
else:
    omega = np.empty(nw)

comm.Bcast(omega)

nq = 10
nk = 10

kT = 0.0005
f = elphmod.occupations.fermi_dirac

path = '../../../Sr2RuO4/epw_mv_t2g_101010'

el = elphmod.el.Model('sro_nosoc')
ph = elphmod.ph.Model('dyn', apply_asr_simple=True)
elph = elphmod.elph.Model('%s/work/Sr2RuO4.epmatwp' % path,
    '%s/wigner.fmt' % path, el, ph)

q = elphmod.bravais.mesh(nq, nq, nq, flat=True)

w2, u = elphmod.dispersion.dispersion(ph.D, q, vectors=True)

g2 = elph.sample(q, nk=(nk, nk, nk), u=u, squared=True, broadcast=False)

if comm.rank == 0:
    g2 = np.average(g2, axis=(2, 3, 4))
else:
    g2 = np.empty((len(q), ph.size, el.size, el.size))

comm.Bcast(g2)

w = elphmod.ph.sgnsqrt(w2)

w[0, :3] = w.max()
g2[0, :3] = 0.0

w = w.reshape((-1, 1, 1))
g2 = g2.reshape((-1, el.size, el.size))

g2 /= 2 * w

w *= elphmod.misc.Ry
g2 *= elphmod.misc.Ry ** 2

w = np.concatenate((w, -w))
g2 = np.concatenate((g2, -g2))

sizes, bounds = elphmod.MPI.distribute(len(omega), bounds=True)

my_a2F = np.empty((sizes[comm.rank], el.size, el.size))

status = elphmod.misc.StatusBar(sizes[comm.rank], title='calculate a2F')

for my_iw, iw in enumerate(range(*bounds[comm.rank:comm.rank + 2])):
    my_a2F[my_iw] = np.sum(f.delta((omega[iw] - w) / kT) * g2, axis=0) / kT

    status.update()

my_a2F /= len(q)

a2F = np.empty((len(omega), el.size, el.size))

comm.Gatherv(my_a2F, (a2F, comm.allgather(my_a2F.size)))

if comm.rank == 0:
    with open('a2fp.txt', 'w') as data:
        for iw in range(len(omega)):
            data.write(('%10.6f' + el.size ** 2 * ' %8.6f' + '\n')
                % (omega[iw], *a2F[iw].ravel()))
