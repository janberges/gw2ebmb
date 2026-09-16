#!/usr/bin/env python3

import ebmb
import elphmod
import numpy as np

comm = elphmod.MPI.comm

el = elphmod.el.Model('data/sro', rydberg=True)
ph = elphmod.ph.Model('data/dyn', apply_asr_simple=True)

nk = 30
eta = 1e-3
T = 290.0

k = np.array(sorted(elphmod.bravais.irreducibles_ibrav(nk, nk, nk, ibrav=7)))

weights = np.array([len(elphmod.bravais.images_ibrav(k1, k2, k3, nk, nk, nk,
    ibrav=7)) for k1, k2, k3 in k])

k = 2 * np.pi * k / nk

H = elphmod.dispersion.sample(el.H, k)
e, U = elphmod.dispersion.dispersion(el.H, k, vectors=True)

v = elphmod.dispersion.sample(el.v, k)
v = np.einsum('kan,kiab,kbn->kni', U.conj(), v, U).real

if comm.rank == 0:

    results = ebmb.get(
        file='fig4b.tmp.dat',

        realgw=True,
        normal=True,
        chiC=True,
        steps=1,

        dos='data/dos.txt',
        a2F='data/a2fep.txt',
        muC=np.loadtxt('data/muc.txt'),
        divdos=False,

        n=4.0,

        lower=-100.0,
        upper=+100.0,
        points=1001,
        logscale=10.0,

        eta=0.01,
        cutoff=0.0,
        T=T,
    )
else:
    results = None

elphmod.MPI.idle_wait()

results = comm.bcast(results)

mu = results['mu'] / elphmod.misc.Ry
kT = elphmod.misc.kB * T / elphmod.misc.Ry

omega_orig = results['omega'] / elphmod.misc.Ry
Sigma_orig = results['Re[Sigma]'] + 1j * results['Im[Sigma]']
Sigma_orig /= elphmod.misc.Ry

omega0 = 2 * np.pi * elphmod.misc.kB * T
omega_max = 10 * omega0 # decayed according Fig. 4 by Stricker et al.

omega_max *= 1e3
omega_max = int(round(omega_max))
points = 2 * omega_max + 1
omega_max /= 1e3

omega = np.linspace(-omega_max, omega_max, points) / elphmod.misc.Ry

sigma = np.empty((2, len(omega), 3, 3), dtype=complex)

for i, fact in enumerate([1, 4]):
    Sigma = np.empty((len(Sigma_orig), len(omega)), dtype=complex)

    for a in range(len(Sigma)):
        Sigma[a] = np.interp(omega, omega_orig, Sigma_orig[a])

    Sigma = np.repeat(Sigma, 2, 0)

    Sigma.imag *= fact

    A = elphmod.MPI.SharedArray((len(k), el.size, len(omega)))

    if comm.rank == 0:
        G = np.linalg.inv((omega + mu - Sigma).T[:, None, :, None] * np.eye(6)
            - H[None, :, :, :]) # omega, k, alpha, beta

        G = U.swapaxes(-2, -1).conj() @ G @ U # omega, k, m, n

        A[...] = -1 / np.pi * np.diagonal(G.imag,
            axis1=2, axis2=3).transpose(1, 2, 0)

    A.Bcast()

    sigma[i] = elphmod.diagrams.green_kubo_conductivity(v, A, omega, kT,
        eta=eta, a=ph.a, weights=weights)

sigma /= 2 # bands are not spin-degenerate

sigma = np.average([S.T @ sigma @ S
    for S in elphmod.bravais.symmetries_ibrav(ibrav=7)], axis=0)

sigma = ph.a.T @ sigma @ ph.a

omega *= elphmod.misc.Ry
sigma *= 1e-6 * elphmod.misc.ohmmRy

if comm.rank == 0:
    with open('fig4b.txt', 'w') as data:
        for iw in range(len(omega) // 2, len(omega)):
            data.write('%5.3f %5.3f %5.3f %5.3f %5.3f\n'
                % (omega[iw], sigma[0, iw, 0, 0].real, sigma[0, iw, 0, 0].imag,
                    sigma[1, iw, 0, 0].real, sigma[1, iw, 0, 0].imag))
