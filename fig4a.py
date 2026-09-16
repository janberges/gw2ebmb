#!/usr/bin/env python3

import ebmb
import elphmod
import numpy as np

comm = elphmod.MPI.comm

a2F = ['e', 'p', 'ep']
T = np.linspace(50, 900, 35)
sigma = np.empty((T.size, len(a2F) + 1, 3, 3))

el = elphmod.el.Model('data/sro', rydberg=True)
ph = elphmod.ph.Model('data/dyn', apply_asr_simple=True)

nk = 30

k = np.array(sorted(elphmod.bravais.irreducibles_ibrav(nk, nk, nk, ibrav=7)))

weights = np.array([len(elphmod.bravais.images_ibrav(k1, k2, k3, nk, nk, nk,
    ibrav=7)) for k1, k2, k3 in k])

k = 2 * np.pi * k / nk

H = elphmod.dispersion.sample(el.H, k)
e, U = elphmod.dispersion.dispersion(el.H, k, vectors=True)

v = elphmod.dispersion.sample(el.v, k)
v = np.einsum('kan,kiab,kbn->kni', U.conj(), v, U).real

status = elphmod.misc.StatusBar(T.size * len(a2F),
    title='calculate resistivity')

for iT in range(T.size):
    for ia2F in range(len(a2F)):
        if comm.rank == 0:
            results = ebmb.get(
                file='fig4a.tmp.dat',
                tell=False,

                realgw=True,
                normal=True,
                chiC=True,
                steps=1,

                dos='data/dos.txt',
                a2F='data/a2f%s.txt' % a2F[ia2F],
                muC=np.loadtxt('data/muc.txt') if 'e' in a2F[ia2F]
                    else np.zeros((3, 3)),
                divdos=False,

                n=4.0,

                lower=-100.0,
                upper=+100.0,
                points=1001,
                logscale=10.0,

                eta=0.01,
                cutoff=0.0,
                T=T[iT],
            )
        else:
            results = None

        elphmod.MPI.idle_wait()

        results = comm.bcast(results)

        omega = results['omega'] / elphmod.misc.Ry
        mu = results['mu'] / elphmod.misc.Ry
        kT = elphmod.misc.kB * T[iT] / elphmod.misc.Ry

        for i, fact in enumerate([1] + [4] * (ia2F == 2)):
            Sigma = np.repeat(results['Re[Sigma]']
                + fact * 1j * results['Im[Sigma]'], 2, 0) / elphmod.misc.Ry

            if comm.rank == 0:
                G = np.linalg.inv((omega + mu - Sigma).T[:, None, :, None]
                    * np.eye(6) - H[None, :, :, :]) # omega, k, alpha, beta

                G = U.swapaxes(-2, -1).conj() @ G @ U # omega, k, m, n

                A = -1 / np.pi * np.diagonal(G.imag, axis1=2, axis2=3)
                # omega, k, n

                A = A.transpose(1, 2, 0).copy() # k, n, omega
            else:
                A = np.empty((len(H), el.size, len(omega)))

            comm.Bcast(A)

            sigma[iT, ia2F + i] = elphmod.diagrams.green_kubo_conductivity(v, A,
                omega, kT, a=ph.a, weights=weights, dc_only=True)

        status.update()

sigma /= 2 # bands are not spin-degenerate

sigma = np.average([S.T @ sigma @ S
    for S in elphmod.bravais.symmetries_ibrav(ibrav=7)], axis=0)

sigma = ph.a.T @ sigma @ ph.a

rho = np.linalg.inv(sigma)
rho *= 1e6 / elphmod.misc.ohmmRy

rho = rho[..., 0, 0] # in-plane component (x)

if comm.rank == 0:
    with open('fig4a.txt', 'w') as data:
        for iT in range(len(T)):
            data.write(('%3.0f' + rho[iT].size * ' %5.3f' + '\n')
                % (T[iT], *rho[iT].ravel()))
