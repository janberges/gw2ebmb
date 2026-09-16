#!/usr/bin/env python3

import ebmb
import elphmod
import numpy as np
import storylines

if elphmod.MPI.comm.rank != 0:
    raise SystemExit

dos = 'fig6.dos.txt'
a2f = 'fig6.a2f.txt'

t = 1.0
kT = 0.1
lamda = 1.0
wlog = 2.0

e, DOS = ebmb.chain_dos(dos, de=5e-3, t=t)
w, a2F = ebmb.chain_a2F(a2f, dw=1e-2, l=lamda, wlog=wlog)

results = ebmb.get(
    file='fig6.tmp.dat',

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

settings = dict(
    packages=['mathtools'],

    style='APS',
    font='Utopia',

    margin=0.2,
    bottom=1.0,
    left=1.0,

    grid=True,

    upper=storylines.color['orange'],
    lower=storylines.color['purple'],
)

line = dict(cut=True, thick=True)
area = dict(cut=True, draw='none', fill='lightgray', yref=0.0)

right = 1.2

settings['width'] = (3 * settings['left'] + 2 * settings['margin'] + right
    - storylines.Plot(style='APS').double) / 3

settings['height'] = settings['width']

plot = storylines.Plot(
    label='a',

    xmin=-2.1,
    xmax=+2.1,
    xstep=1.0,

    ymin=-2.0,
    ymax=+2.0,
    ystep=1.0,

    xlabel=r'Boson energy $\omega$ ($\langle \omega \rangle$)',
    ylabel=r'Spectral function $\alpha^2 F(\omega)$ ($\lambda$)',

    **settings,
)

for sgn in -1, +1:
    plot.line(sgn * w / wlog, sgn * a2F / lamda, **area)

plot.axes()

plot.save('fig6a.pdf')

plot = storylines.Plot(
    label='b',

    xmin=-8.0,
    xmax=+8.0,
    xstep=4.0,

    ymin=0.0,
    ymax=0.4,
    ystep=0.1,

    xlabel=r'Electron energy $\varepsilon$ ($t$)',
    ylabel=r'Spectral function $A_1(\varepsilon)$ ($1 / t$)',

    **settings,
)

plot.line(e / t, DOS * t, **area)

plot.axes()

plot.line(results['omega'] / t, results['DOS'] * t, **line)

plot.save('fig6b.pdf')

plot = storylines.Plot(
    label='c',

    right=right,

    xmin=-8.0,
    xmax=+8.0,
    xstep=4.0,

    ymin=-6.0,
    ymax=+6.0,
    ystep=3.0,

    xlabel=r'Electron energy $\varepsilon$ ($t$)',
    ylabel=r'Electron self-energy $\varSigma_1(\varepsilon)$ ($t$)',

    lpos='rm',
    lopt='right=%gcm' % plot.gap,

    **settings,
)

plot.line(results['omega'] / t, results['Re[Sigma]'] / t, label='Re', **line)
plot.line(results['omega'] / t, results['Im[Sigma]'] / t, label='Im',
    dashed=True, **line)

plot.save('fig6c.pdf')

lamda = np.loadtxt('fig6def.txt', max_rows=1)
data = np.loadtxt('fig6def.txt', skiprows=1).T

wlog = data[0]
Z = data[1::4]
width = data[2::4]
rho1 = data[3::4]
rho2 = data[4::4]

plot = storylines.Plot(
    label='d',

    xmin=0.0,
    xstep=1.0,

    ymin=1.0,
    ymax=lamda.max() + 1.0,
    ystep=0.5,

    zmin=0.0,

    xlabel=r'Effective boson energy $\langle \omega \rangle$ ($t$)',
    ylabel=r'Renormalization $Z_1(0)$',

    colorbar=False,

    **settings,
)

for i in range(lamda.size):
    plot.line(wlog / t, Z[i], lamda[i], **line)

plot.save('fig6d.pdf')

scale = width[-1, -1] / t / rho2[-1, -1]

plot = storylines.Plot(
    label='e',

    xmin=0.0,
    xstep=1.0,

    ymin=0.0,
    ymax=scale,
    ystep=0.1,

    zmin=0.0,

    xlabel=r'Effective boson energy $\langle \omega \rangle$ ($t$)',
    ylabel=r'Broadening $-\mathrm{Im}~\Sigma_1(0)$ ($t$)',

    colorbar=False,

    **settings,
)

for i in range(lamda.size):
    plot.line(wlog / t, width[i] / t, lamda[i], **line)

plot.save('fig6e.pdf')

plot = storylines.Plot(
    label='f',

    right=right,

    xmin=0.0,
    xstep=1.0,

    ymin=0.0,
    ymax=1.0,
    ystep=0.2,

    zmin=0.0,
    zstep=1.0,

    xlabel=r'Effective boson energy $\langle \omega \rangle$ ($t$)',
    ylabel=r'Resistivity $\rho_1$ (arb.~u.)',
    zlabel=r'Effective coupling strength $\lambda$',

    lbox=True,
    lopt='below left=2mm',
    lpos='rt',

    **settings,
)

for i in range(lamda.size):
    plot.line(wlog[::-1], rho1[i, ::-1], lamda[i], **line)

for i in range(lamda.size):
    plot.line(wlog[::-1], rho2[i, ::-1], dashed=True, label='QPA', **line)

for i in range(lamda.size):
    plot.line(wlog[::-1], scale * rho2[i, ::-1] / width[i, ::-1], thick=True,
        dash_pattern='on 1pt off 1pt',
        label=r'$\mathrm{QPA}/\mathrm{Im}~\Sigma_1(0)$ (scaled)')

plot.save('fig6f.pdf')

storylines.combine('fig6.pdf', ['fig6%s' % s for s in 'abcdef'], columns=3)
