#!/usr/bin/env python3

import copy
import ebmb
import elphmod
import numpy as np
import storylines

if elphmod.MPI.comm.rank != 0:
    raise SystemExit

line = dict(thick=True, cut=True)

default = dict(color=storylines.color['orange'])

for labels, style, this, that in [
    ('ab', dict(color=storylines.color['magenta']),
        r'$i = 1$ ($G_0 \widetilde W_0$)',
        r'$i = \infty$ ($G \widetilde W_0$)'),
    ('cd', dict(dashed=True), 'direct', r"Pad\'e"),
]:
    plot1 = storylines.Plot(
        packages=['mathtools'],

        style='APS',
        font='Utopia',

        height=-4.65,
        left=1.0,
        right=0.2,
        bottom=1.0,
        top=0.5,

        xmin=-4.0,
        xmax=+4.0,
        xstep=2.0,
        xminorstep=1.0,

        ymin=-2.0,
        ymax=+2.0,
        ystep=1.0,

        lpos='rt',
        lopt='below left=2mm',
        lbox=True,

        xlabel=r'Electron energy $\varepsilon$ (eV)',
        ylabel=r'Electron self-energy $\varSigma_{i \alpha}(\varepsilon)$ (eV)',

        grid=True,
    )

    plot1.width = (plot1.left + 7 * plot1.right - plot1.double) / 4

    plot2 = copy.deepcopy(plot1)

    if labels == 'cd':
        plot1.ylabels = False
        plot1.left = plot1.right

    plot2.ylabels = False
    plot2.left = plot2.right

    plot1.title = 'Re'
    plot2.title = '$-$Im'

    plot1.label = labels[0]
    plot2.label = labels[1]

    x = []
    y = []

    for variation in False, True:
        realgw = labels == 'ab' or not variation

        results = ebmb.get(
            file='fig5.tmp.dat',

            realgw=realgw,
            normal=True,
            chiC=True,
            steps=100 if labels == 'ab' and variation else 1,
            align0=labels == 'ab',

            dos='data/dos.txt',
            a2F='data/a2fep.txt',
            muC=np.loadtxt('data/muc.txt'),
            divdos=False,

            n=4.0,

            lower=-100.0,
            upper=+100.0,
            points=1001,
            logscale=10.0,

            eta=0.01 if realgw else 0.0,
            cutoff=0.0 if realgw else 5.0,
            cutoffP=1.0,
            T=300.0,
        )

        for a in range(1, 3):
            plot1.line(results['omega'], results['Re[Sigma]'][a],
                **(style if variation else default), **line)

            plot2.line(results['omega'], -results['Im[Sigma]'][a],
                **(style if variation else default), **line)

            l = np.argmin(abs(results['omega'] + 4))
            u = np.argmin(abs(results['omega'] + 2))

            argmin = l + np.argmin(results['Re[Sigma]'][a, l:u])
            x.append(results['omega'][argmin])
            y.append(results['Re[Sigma]'][a, argmin])

    if labels == 'ab':
        box = dict(draw='gray', fill='white', rounded_corners='5pt')

        X = (x[0] + x[2]) / 2
        Y = (y[0] + y[2]) / 2 - 0.4
        plot1.line([x[0], X, x[2]], [y[0], Y, y[2]], **{'<->': True})
        plot1.node(X, Y, '$d_{x z, y z}$', below=True, **box)

        X = (x[1] + x[3]) / 2
        Y = (y[1] + y[3]) / 2 + 0.4
        plot1.line([x[1], X, x[3]], [y[1], Y, y[3]], **{'<->': True})
        plot1.node(X, Y, '$d_{x y}$', above=True, **box)

    plot1.line(label=this, **default, **line)
    plot1.line(label=that, **style, **line)

    inset = copy.deepcopy(plot2)

    inset.xlabel = None
    inset.ylabels = True
    inset.title = None
    inset.label = None

    inset.width = 2.7
    inset.height = 1.8

    inset.top = inset.right = 0.3
    inset.left = inset.bottom = 0.4

    inset.xmin = -0.5
    inset.xmax = +0.5
    inset.xstep = 0.5

    inset.ymin = 0.0
    inset.ymax = 0.05
    inset.ystep = 0.05

    inset.save('fig5%s.inset.pdf' % labels[1])

    plot2.node(0, -1, r'\includegraphics{fig5%s.inset.pdf}' % labels[1],
        draw='gray', fill='white', rounded_corners='1pt')

    plot1.save('fig5%s.pdf' % labels[0])
    plot2.save('fig5%s.pdf' % labels[1])

storylines.combine('fig5.pdf', ['fig5a', 'fig5b', 'fig5c', 'fig5d'])
