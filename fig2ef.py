#!/usr/bin/env python3

import ebmb
import elphmod
import numpy as np

if elphmod.MPI.comm.rank != 0:
    raise SystemExit

T = np.linspace(50, 1000, 39)

status = elphmod.misc.StatusBar(2 * T.size,
    title='calculate data for Fig. 2')

with open('fig2ef.txt', 'w') as data:
    for i in range(len(T)):
        data.write('%4.0f' % T[i])

        for typ in 'ep', 'e':
            out = ebmb.get(
                file='fig2ef.tmp.dat',
                tell=False,

                realgw=True,
                normal=True,
                chiC=True,
                steps=1,

                dos='data/dos.txt',
                a2F='data/a2f%s.txt' % typ,
                muC=np.loadtxt('data/muc.txt'),
                divdos=False,

                n=4.0,
                conserve=False,

                lower=-100.0,
                upper=+100.0,
                points=1001,
                logscale=10.0,

                eta=0.01,
                cutoff=0.0,
                T=T[i],
            )

            zero = np.argmin(abs(out['omega']))

            Z = out['Re[Sigma]'][1:, zero + 1] - out['Re[Sigma]'][1:, zero - 1]
            Z /= out['omega'][zero + 1] - out['omega'][zero - 1]
            Z = 1 - Z

            width = -out['Im[Sigma]'][1:, zero]

            data.write((4 * ' %6.4f') % (*Z, *width))

            status.update()

        data.write('\n')
