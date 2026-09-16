#!/usr/bin/env python3

import ebmb
import elphmod
import numpy as np

if elphmod.MPI.comm.rank != 0:
    raise SystemExit

dos = 'fig6def.dos.txt'
a2f = 'fig6def.a2f.txt'

kT = 0.1
t = 1.0
lamda = np.linspace(0.2, 2.0, 10)

wlog = ebmb.get(
    file='fig6def.tmp.dat',
    lower=0.1,
    upper=4.0,
    points=100,
    tell=False,
)['omega']

k = np.linspace(0, 2 * np.pi, 200, endpoint=False)

e = -2 * t * np.cos(k)
v = 2 * t * np.sin(k)

ebmb.chain_dos(dos, de=5e-3, t=t)

status = elphmod.misc.StatusBar(len(lamda) * len(wlog),
    title='calculate data for Fig. 6')

with open('fig6def.txt', 'w') as data:
    data.write(' ' * 8)

    for l in range(len(lamda)):
        data.write('%36.6f' % lamda[l])

    data.write('\n')

    for w in range(len(wlog)):
        data.write('%8.6f' % wlog[w])

        for l in range(len(lamda)):
            ebmb.chain_a2F(a2f, dw=5e-3, l=lamda[l], wlog=wlog[w])

            results = ebmb.get(
                file='fig6def.tmp.dat',
                tell=False,

                realgw=True,
                normal=True,
                steps=1,

                dos=dos,
                a2F=a2f,

                mu=0.0,
                conserve=False,

                lower=-15.0,
                upper=+15.0,
                resolution=1001,
                logscale=10.0,

                eta=0.03,
                T=kT / elphmod.misc.kB,
            )

            omega = results['omega']
            Sigma = results['Re[Sigma]'] + 1j * results['Im[Sigma]']

            zero = np.argmin(abs(omega))

            Z = Sigma.real[zero + 1] - Sigma.real[zero - 1]
            Z /= omega[zero + 1] - omega[zero - 1]
            Z = 1 - Z

            data.write(' %8.6f' % Z)
            data.write(' %8.6f' % -Sigma.imag[zero])

            G = 1 / (omega[:, None] - e - Sigma[:, None])
            A = -G.imag / np.pi

            domega = elphmod.misc.differential(omega)
            delta = elphmod.occupations.fermi_dirac.delta(omega / kT) / kT

            sigma = 2 * np.pi * np.sum(domega * delta
                * np.average((v * A) ** 2, axis=1))

            data.write(' %8.6f' % (1 / sigma))

            E = e / Z
            deltaE = elphmod.occupations.fermi_dirac.delta(E / kT) / kT
            ImSigma = -Sigma.imag[np.argmin(abs(omega[:, None] - E), axis=0)]

            sigma = np.average(deltaE * v ** 2 / ImSigma) / Z

            data.write(' %8.6f' % (1 / sigma))

            status.update()

        data.write('\n')
