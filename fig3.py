#!/usr/bin/env python3

import ebmb
import elphmod
import matplotlib.pyplot as plt
import numpy as np
import storylines

comm = elphmod.MPI.comm

margin = 0.2
left = 1.0
right = 1.2
bottom = 0.5

settings = dict(
    packages=['mathtools', 'bm', r'\let\vec\bm'],

    style='APS',
    font='Utopia',

    margin=margin,
    bottom=bottom,
)

pw = elphmod.bravais.read_pwi('data/scf.in')
el = elphmod.el.Model('data/sro')
el_nosoc = elphmod.el.Model('data/sro_nosoc')

a = pw['a']
c = pw['c']

a6 = elphmod.bravais.primitives(ibrav=6, a=a, c=c)
b6 = np.array(elphmod.bravais.reciprocals(*a6))

a7 = elphmod.bravais.primitives(**pw)

def f627(*k6):
    return np.dot(np.dot(k6, b6), a7.T)

d = 0.5 * (a / c) ** 2

k_main, x_main, corners_main = elphmod.bravais.path([
    f627(0.5, 0.5, 0.0),
    'G',
    f627(0.5, 0.0, 0.0),
    f627(0.5 + d, 0.0, 0.0),
], N=3000, **pw)

x_main -= x_main[corners_main[1]]

k_inset, x_inset, corners_inset = elphmod.bravais.path([
    f627(0.375, 0.375, 0.0),
    f627(0.250, 0.250, 0.0),
    'G',
], N=6000, **pw)

x_inset -= x_inset[corners_inset[2]]

k_inset = k_inset[:corners_inset[1] + 1]
x_inset = x_inset[:corners_inset[1] + 1]
corners_inset = corners_inset[:2]

if comm.rank == 0:
    results = ebmb.get(
        file='fig3.tmp.dat',

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
        points=5001,
        logscale=10.0,

        eta=0.01,
        cutoff=0.0,
        T=5.0,
    )

    levels = ebmb.get(
        file='fig3.tmp.dat',
        lower=0.0,
        upper=50.0,
        points=256,
        logscale=0.2,
    )['omega']
else:
    results = levels = None

elphmod.MPI.idle_wait()

results = comm.bcast(results)
levels = comm.bcast(levels)

omega_orig = results['omega']
Sigma_orig = results['Re[Sigma]'] + 1j * results['Im[Sigma]']

maximum = 50.0

cmap = storylines.colormap(
    (0.00, storylines.color['white']),
    (0.25, storylines.color['orange']),
    (0.50, storylines.color['magenta']),
    (0.75, storylines.color['purple']),
    (1.00, storylines.color['black']),
)

omega_inset = np.linspace(-0.1, 0.1, 401)
omega_main = np.linspace(-1, 1, 1001)

for inset, k, x, corners, omega in [
    (True, k_inset, x_inset, corners_inset, omega_inset),
    (False, k_main, x_main, corners_main, omega_main),
]:
    H = elphmod.dispersion.sample(el.H, k)
    e = elphmod.dispersion.dispersion(el.H, k)

    Sigma = np.empty((len(Sigma_orig), len(omega)), dtype=complex)

    for a in range(len(Sigma)):
        Sigma[a] = np.interp(omega, omega_orig, Sigma_orig[a])

    Sigma = np.repeat(Sigma, 2, 0)

    if comm.rank == 0:
        G = np.linalg.inv(np.eye(6)
            * (omega + results['mu'] - Sigma).T[:, None, :, None]
            - H[None, :, :, :]) # indices: omega, k, alpha, beta

        A = -1 / np.pi * np.trace(G.imag, axis1=2, axis2=3).copy()

        A = levels[np.argmin(abs(A[:, :, None] - levels), axis=2)]
    else:
        A = np.empty((len(omega), len(k)))

    comm.Bcast(A)

    image = elphmod.plot.color(A[::-1], cmap=cmap,
        minimum=0, maximum=maximum)

    if comm.rank != 0:
        continue

    plot = storylines.Plot(
        label=None if inset else 'a',

        width=3.5,
        left=bottom if inset else left,
        right=margin if inset else right,

        cmap=cmap,
        colorbar=not inset,

        xmin=x[0],
        xmax=x[-1],
        xticks=list(zip(x[corners], [r'3\rlap{/4\,X}', r'\llap{1/2\,}X']
            if inset else ['X', r'$\Gamma$', r'M\,\,', r'\,\,S'])),

        ymin=omega[0],
        ymax=omega[-1],
        ystep=0.1 if inset else 0.5,

        zmin=0.0,
        zmax=maximum,
        zticks=[0, 10, 20, 30, 40, (50, r'$\mathllap\geq 50$')],

        ylabel=None if inset
            else r'Electron energy $\varepsilon$ from Fermi level (eV)',
        zlabel=r'Spectral function $A_{1 \vec k}(\varepsilon)$ (1/eV)',

        background='fig3a.inset.png' if inset else 'fig3a.png',

        lpos='bl',
        lopt='above right',
        lput=not inset,

        **settings,
    )

    if not inset:
        plot.width = plot.double / 2

        plot.code(r'\draw[gray] (<x=%g>, <y=%g>) rectangle (<x=%g>, <y=%g>);'
            % (x_inset[0], omega_inset[0], x_inset[-1], omega_inset[-1]))
        plot.node(0.0, 0.5, r'\includegraphics{fig3a.inset.pdf}')

    plot.height = plot.width

    storylines.save(plot.background, image)

    plot.line(y=0, color='gray')

    plot.line(mark='*', only_marks=True, label='ARPES')

    for n in range(0, el.size, 2):
        plot.line(x, e[:, n] - results['mu0'], cut=True, label='DFT')

    scaleX = 2 * np.pi
    scaleY = 1e3

    marks = dict(mark='*', only_marks=True, mark_size='0.1pt', cut=True)

    X, Y = np.loadtxt('ref/tamai_00deg.txt').T

    plot.line(X / scaleX, Y / scaleY, **marks)

    X, Y = np.loadtxt('ref/tamai_45deg.txt').T

    plot.line(-X / scaleX, Y / scaleY, **marks)

    plot.save(plot.background.replace('.png', '.pdf'))

nk = 500

kxy, dk = np.linspace(0.0, np.pi / pw['a'], nk, endpoint=False, retstep=True)
kxy += dk / 2

k = np.array([[[kx, ky, 0.0] for ky in kxy] for kx in kxy])
k = np.dot(k, a7.T)

isolines = []

for H in el_nosoc.H, el.H:
    e = elphmod.dispersion.dispersion(H, k) - results['mu0']

    isolines.append([isoline / (nk - 1) for m in range(e.shape[2])
        for isoline in plt.contour(e[:, :, m], levels=[0.0]).allsegs[0]])

H = elphmod.dispersion.sample(el.H, k)

Sigma0 = Sigma[:, np.argmin(abs(omega))]

if comm.rank == 0:
    G = np.linalg.inv((results['mu'] - Sigma0)[None, :, None] * np.eye(6)
        - H) # indices: k, alpha, beta

    A = -1 / np.pi * np.trace(G.imag, axis1=2, axis2=3)

    A = levels[np.argmin(abs(1e4 * A[:, :, None] - levels), axis=2)]
else:
    A = np.empty((nk, nk))

comm.Bcast(A)

A = elphmod.plot.color(A, cmap, minimum=0, maximum=maximum)

if comm.rank != 0:
    raise SystemExit

storylines.save('fig3b.png', A[:, ::-1])

plot = storylines.Plot(
    label='b',

    height=0,
    left=bottom,

    xmin=-1.0,
    xmax=+1.0,
    ymin=-1.0,
    ymax=+1.0,
    xstep=1.0,
    ystep=1.0,

    xyaxes=False,
    grid=True,
    frame=True,

    line_cap='butt',

    **settings,
)

plot.width = plot.double / 2

plot.image('fig3b.png', -1, -1, 0, 0)
plot.image('ref/tamai_fs.png', 0, -1, 1, 0)

d = 0.3

plot.node(-d, +d, r'DFT w/o SOC')
plot.node(+d, +d, r'DFT')
plot.node(-d, -d, r'$10^4 \cdot A_{1 \vec k}(0)$')
plot.node(+d, -d, r'ARPES')

for isoline in isolines[0]:
    plot.line(-isoline[:, 0], isoline[:, 1], thick=True)

for isoline in isolines[1]:
    plot.line(isoline[:, 0], isoline[:, 1], thick=True)

plot.axes()

plot.line([0, 0, -1], [0, -1, -1], only_marks=True, mark='*')

plot.node(0, 0, r'$\Gamma$', below_left=True)
plot.node(0, -1, 'M', below='%scm' % plot.tick)
plot.node(-1, -1, 'X', below='%scm' % plot.tick)

plot.node(0.55, -0.55, r'$\beta$')
plot.node(0.64, -0.64, r'$\gamma$')
plot.node(0.73, -0.73, r'$\alpha$')

plot.save('fig3b.pdf')

storylines.combine('fig3.pdf', ['fig3a', 'fig3b'])
