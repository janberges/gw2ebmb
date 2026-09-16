#!/usr/bin/env python3

import elphmod
import numpy as np
import storylines

if elphmod.MPI.comm.rank != 0:
    raise SystemExit

orange = storylines.color['orange']
magenta = storylines.color['magenta']
purple = storylines.color['purple']

line = dict(thick=True, cut=True)
mark = dict(mark='*', only_marks=True, cut=True)
bar = dict(line_width='1mm', line_cap='butt')

settings = dict(
    packages=['mathtools'],

    style='APS',
    font='Utopia',

    height=6.7,

    margin=0.2,
    left=1.0,
    bottom=1.0,

    lpos='rt',
    lopt='below left=2mm',
    lbox=True,

    grid=True,
)

plot = storylines.Plot(
    label='a',

    xmin=0.0,
    ymin=0.0,
    ymax=1.2,

    xstep=200.0,
    ystep=0.2,

    xlabel=r'Temperature $T$ (K)',
    ylabel=r'In-plane resistivity $\rho_{1 1 1}$ (\textmu$\Omega$m)',

    lcol=3,
    lrmo=True,
    lwid=3.5,

    **settings,
)

plot.width = plot.double / 2

data = np.loadtxt('fig4a.txt').T

plot.line(label='el.', color=orange, **bar)
plot.line(label='ph.', color=magenta, **bar)
plot.line(label='both', **bar)

plot.line(data[0], data[3], label='This work', **line)
plot.line(data[0], data[1], label='*none*', color=orange, **line)
plot.line(data[0], data[2], label='*none*', color=magenta, **line)
plot.line(data[0], data[4], dashed=True,
    label=r'This work, $\mathrm{Im}\,\varSigma '
        r'\rightarrow 4\,\mathrm{Im}\,\varSigma$', **line)

data = np.loadtxt('ref/abramovitch.txt').T

plot.line(data[0], data[2], label='*none*', color=orange, **mark)
plot.line(data[0], data[1], label='*none*', color=magenta, **mark)

plot.line(data[0], data[3], label='Theoretical reference', **mark)

for _ in range(2):
    plot.nolabel()

plot.line(*np.loadtxt('ref/tyler.txt').T, dotted=True, label='Experiment',
    **line)

plot.save('fig4a.pdf')

plot = storylines.Plot(
    label='b',

    xmin=0.0,
    xmax=0.2,
    ymin=0.0,
    ymax=1.2,

    xstep=0.05,
    ystep=0.2,

    xlabel=r'Photon energy $\omega$ (eV)',
    ylabel=r'Optical conductivity $\sigma_{1 1 1}(\omega) / \sigma_{1 1 1}(0)$',

    lcol=2,
    lrmo=True,
    lwid=3.5,

    **settings,
)

plot.width = plot.double / 2

plot.line(label='Re', **bar)
plot.line(label='Im', color=purple, **bar)

data = np.loadtxt('fig4b.txt').T

plot.line(data[0], data[1] / data[1][0], label='This work', **line)
plot.line(data[0], data[2] / data[1][0], label='*none*', color=purple, **line)
plot.line(data[0], data[3] / data[3][0], dashed=True,
    label=r'This work, $\mathrm{Im}\,\varSigma '
        r'\rightarrow 4\,\mathrm{Im}\,\varSigma$', **line)
plot.line(data[0], data[4] / data[3][0], dashed=True,
    label='*none*', color=purple, **line)

re = np.loadtxt('ref/stricker_real.txt').T
im = np.loadtxt('ref/stricker_imag.txt').T

T = 290.0
wunit = 2 * np.pi * elphmod.misc.kB * T

plot.line(wunit * re[0], 0.1 * re[1], dotted=True,
    label=r'Experiment (S/\textmu m)', **line)
plot.line(wunit * im[0], 0.1 * im[1], dotted=True, color=purple, **line)

plot.save('fig4b.pdf')

storylines.combine('fig4.pdf', ['fig4a', 'fig4b'])
