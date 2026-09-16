#!/usr/bin/env python3

import copy
import ebmb
import elphmod
import numpy as np
import storylines

if elphmod.MPI.comm.rank != 0:
    raise SystemExit

margin = 0.2
left = 1.0
top = 0.5

colors = [storylines.color['orange'], storylines.color['magenta']]

settings = dict(
    packages=['mathtools'],

    style='APS',
    font='Utopia',

    height=(2 * left + margin + top - 12) / 2,
    margin=margin,
    bottom=left,
    left=left,

    lbox=True,
    grid=True,
)

line = dict(cut=True, thick=True)
area = dict(cut=True, draw='none', yref=0.0)

data = np.loadtxt('data/a2fep.txt')

w = data[:, 0]
a2F = data[:, 1:]

plot = storylines.Plot(
    label='a',

    xmin=0.002,
    xmax=200.0,
    ymin=0.001,
    ymax=10.0,

    xformat=lambda x: '%g' % x + r'\,' * (x == 100),

    xlabel=r'Boson energy $\omega$ (eV)',
    ylabel=r'Spectral function $\widetilde B_{\alpha \beta}(\omega)$ (1/eV)',

    lcol=3,
    lopt='below right=2mm',
    lpos='lt',
    lwid=4.5,

    xlog=True,
    ylog=True,

    **settings,
)

plot.width = plot.double / 2

indices = [0, 8, 1, 2]

labels = [
    '$d_{x z, y z}$',
    '$d_{x y}$',
    '$d_{x z}$--$d_{y z}$',
    '$d_{x z, y z}$--$d_{x y}$',
]

styles = [
    dict(color=colors[0]),
    dict(color=colors[1], dashed=True),
    dict(color='black'),
    dict(color='gray', dash_pattern='on 1pt off 1pt'),
]

for n, i in enumerate(indices):
    ok = np.minimum(w, a2F[:, i]) >= 1e-10

    plot.line(w[ok], a2F[ok, i], label=labels[n], **line, **styles[n])

plot.save('fig2a.pdf')

results = dict()

for typ in 'ep', 'e', 'p':
    results[typ] = ebmb.get(
        file='fig2.tmp.dat',

        realgw=True,
        normal=True,
        chiC=True,
        steps=1,

        dos='data/dos.txt',
        a2F='data/a2f%s.txt' % typ,
        muC=np.loadtxt('data/muc.txt'),
        divdos=False,

        n=4.0,

        lower=-100.0,
        upper=+100.0,
        points=1001,
        logscale=10.0,

        eta=0.01,
        cutoff=0.0,
        T=300.0,
    )

data = np.loadtxt('data/dos.txt')

e = data[:, 0] - results['ep']['mu0']
DOS = data[:, 1:]

for typ in results:
    print('Typ: %s' % typ)

    print('Effective coupling: %g, %g, %g' % tuple(DOS[np.argmin(abs(e))]
        * np.diag(results[typ]['lambda'])))

    print('Effective energy: %g eV' % results[typ]['omegaLog'])

plot = storylines.Plot(
    label='b',

    xlabel=r'Electron energy $\varepsilon$ from Fermi level (eV)',
    ylabel=r'Spectral function $A_{1 \alpha}(\varepsilon)$ (1/eV)',

    xmin=-3.0,
    xmax=1.0,
    xstep=1.0,
    ymin=0.0,
    ymax=3.0,
    ystep=1.0,

    lopt='below right=2mm',
    lpos='lt',

    line_cap='butt',

    **settings,
)

plot.width = plot.double / 2

plot.grids()

def pale(color=storylines.color['gray'], r=0.3):
    return r * color + (1 - r) * storylines.color['white']

plot.line(e, DOS.sum(axis=1), fill=pale(), **area)

for a in range(1, 3):
    plot.line(e, DOS[:, a], fill=pale(colors[a - 1]), **area)

plot.line(e, DOS.min(axis=1), fill=pale((colors[0] + colors[1]) / 2), **area)

labels = [
    r'$\alpha = d_{x z, y z}$',
    r'$\alpha = d_{x y}$',
]

omega = results['ep']['omega']
A = results['ep']['DOS']

for a in range(1, 3):
    plot.line(omega, A[a], color=colors[a - 1], label=labels[a - 1], **line)

plot.line(omega, A.sum(axis=0), draw='gray', label=r'$\sum_\alpha$', **line)

plot.line(line_width='2mm', color=pale(), label='bare')

plot.axes()

plot.save('fig2b.pdf')

settings.update(
    width=(left + 3 * margin - plot.double / 2) / 2,
    top=top,
)

plot1 = storylines.Plot(
    label='c',
    title='Re',

    xmin=-4.0,
    xmax=+4.0,
    xstep=2.0,
    xminorstep=1.0,

    ymin=-2.0,
    ymax=+2.0,
    ystep=1.0,

    xlabel=r'Electron energy $\varepsilon$ (eV)',
    ylabel=r'Electron self-energy $\varSigma_{1 \alpha}(\varepsilon)$ (eV)',

    lopt='below left=2mm',
    lpos='rt',

    **settings,
)

plot2 = copy.deepcopy(plot1)

plot2.label = 'd'
plot2.title = '$-$Im'
plot2.left = margin
plot2.ylabels = False

for typ in 'ep', 'e':
    p = 'p' in typ

    for a in range(1, 3):
        color = colors[a - 1] if p else 'black'

        plot1.line(results[typ]['omega'], results[typ]['Re[Sigma]'][a],
            color=color, dashed=not p, label=labels[a - 1] if p else 'w/o ph.',
            **line)

        plot2.line(results[typ]['omega'], -results[typ]['Im[Sigma]'][a],
            color=color, dashed=not p, **line)

plot1.save('fig2c.pdf')

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

inset.save('fig2d.inset.pdf')

plot2.node(0, -1, r'\includegraphics{fig2d.inset.pdf}',
    draw='gray', fill='white', rounded_corners='1pt')

plot2.save('fig2d.pdf')

mark = r'\,\tikz [baseline=-2.5pt] \draw [mark=%s] plot coordinates {(0, 0)};\,'
ball = mark % '*'
square = mark % 'square*'

plot1 = storylines.Plot(
    label='e',
    title=r'$%s = (%s - 1) / 3 + 1$' % (ball, square),

    xmin=1.0,
    xmax=5.0,
    xstep=1.0,

    ymin=0.0,
    ymax=1000.0,
    ystep=200.0,

    xlabel=r'Renormalization $Z_{1 \alpha}(0)$',
    ylabel='Temperature $T$ (K)',

    lopt='below left=2mm',
    lpos='rt',

    **settings,
)

plot2 = copy.deepcopy(plot1)

plot2.label = 'f'
plot2.title = r'$%s = %s / 4$' % (ball, square)

plot2.left = margin

plot2.xmin = 0.0
plot2.xmax = 0.13
plot2.xstep = 0.05

plot2.xlabel = r'Broadening $-\mathrm{Im}~$\Sigma_{1 \alpha}(0)$ (eV)'
plot2.ylabels = False

data = np.loadtxt('ref/hunter.txt').T

for mark, div in ('square*', 1), ('*', 3):
    for a in range(1, 3):
        plot1.line((data[a] - 1) / div + 1, data[0], color=colors[a - 1],
            mark=mark, only_marks=True)

plot1.line(mark='square*', only_marks=True, label='DMFT')

for mark, div in ('square*', 1), ('*', 4):
    for a in range(1, 3):
        plot2.line(data[a + 2] / div, data[0], color=colors[a - 1],
            mark=mark, only_marks=True)

data = np.loadtxt('fig2ef.txt').T

plot1.line(data[1], data[0], color=colors[0], **line)
plot1.line(data[2], data[0], color=colors[1], **line)
plot2.line(data[3], data[0], color=colors[0], **line)
plot2.line(data[4], data[0], color=colors[1], **line)

plot1.line(data[5], data[0], dashed=True, **line)
plot1.line(data[6], data[0], dashed=True, **line)
plot2.line(data[7], data[0], dashed=True, **line)
plot2.line(data[8], data[0], dashed=True, **line)

plot1.save('fig2e.pdf')
plot2.save('fig2f.pdf')

storylines.combine('fig2ab.pdf', ['fig2a', 'fig2b'])
storylines.combine('fig2cdef.pdf', ['fig2c', 'fig2d', 'fig2e', 'fig2f'])
storylines.combine('fig2.pdf', ['fig2ab', 'fig2cdef'], columns=1)
