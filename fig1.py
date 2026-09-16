#!/usr/bin/env python3

import ebmb
import elphmod
import numpy as np
import storylines

margin = 0.2
bottom = 0.5
left = 1.0

orange = storylines.color['orange']
magenta = storylines.color['magenta']
purple = storylines.color['purple']

scale = 0.2

rSr = scale * 2.0
rRu = scale * 1.3
rO = scale * 0.6

pw = elphmod.bravais.read_pwi('data/scf.in')

a = pw['a']
c = pw['c']

a6 = elphmod.bravais.primitives(ibrav=6, a=a, c=c)
a7 = elphmod.bravais.primitives(**pw)

b6 = np.array(elphmod.bravais.reciprocals(*a6))

def f627(*k6):
    return np.dot(np.dot(k6, b6), a7.T)

k0, x0, e0 = elphmod.el.read_bands('data/bands.dat')

# X^2 + |R - X|^2 = (X - |R - X|)^2 + H_BZ^2
# 0 = -2 X |R - X| + H_BZ^2
# |R - X| / X = (H_BZ / X)^2 / 2 = [(2 / c) / (sqrt(2) / a)]^2 / 2 = (a / c)^2

d = 0.5 * (a / c) ** 2

k, x, corners = elphmod.bravais.path([
    'G',
    f627(0.5, 0.0, 0.0), # M
    f627(0.5 + d, 0.0, 0.0), # S
    f627(1.0, 0.0, 0.0), # Z
    f627(0.5 + d, 0.5 - d, 0.0), # R
    f627(0.5, 0.5, 0.0), # X
    'G',
], N=768, **pw)

x0 *= x[-1] / x0[-1]

el = elphmod.el.Model('data/sro')

e, U = elphmod.dispersion.dispersion(el.H, k, vectors=True)

q = k

ph = elphmod.ph.Model('data/dyn', apply_asr_simple=True)

w2, u, order = elphmod.dispersion.dispersion(ph.D, q, order=True, vectors=True)

pol = elphmod.ph.polarization(u, q)

w = elphmod.ph.sgnsqrt(w2) * elphmod.misc.Ry * 1e3

if elphmod.MPI.comm.rank != 0:
    raise SystemExit

mu = ebmb.get(
    file='fig1.tmp.dat',
    dos='data/dos.txt',
    bands=3,
    n=4.0,
    T=300.0,
)['mu0']

e0 -= mu
e -= mu

X = np.array(pw['at'])

Sr = pw['r'][X == 'Sr']
Ru = pw['r'][X == 'Ru']
O = pw['r'][X == 'O']

SC = range(-2, 3)

D = np.empty(3, dtype=int)

Sr = np.array([(r + D) @ a7
    for r in Sr for D[0] in SC for D[1] in SC for D[2] in SC])

Ru = np.array([(r + D) @ a7
    for r in Ru for D[0] in SC for D[1] in SC for D[2] in SC])

O = np.array([(r + D) @ a7
    for r in O for D[0] in SC for D[1] in SC for D[2] in SC])

eps = 0.01

Sr = np.array([(x, y, z) for x, y, z in Sr
    if -eps < x < a + eps and -eps < y < a + eps and -eps < z < c + eps])

Ru = np.array([(x, y, z) for x, y, z in Ru
    if -eps < x < a + eps and -eps < y < a + eps and -eps < z < c + eps])

O = np.array([(x, y, z) for x, y, z in O
    if -eps < x < a + eps and -eps < y < a + eps and -eps < z < c + eps])

bonds = storylines.bonds(R1=O, R2=O, d1=rO, d2=rO, dmin=0.1, dmax=3.0)

objects = []

atom = dict(mark='ball', only_marks=True, omit=False)

objects.extend([([r], dict(mark_size=rSr, ball_color=purple, **atom))
    for r in Sr])

objects.extend([([r], dict(mark_size=rRu, ball_color=magenta, **atom))
    for r in Ru])

objects.extend([([r], dict(mark_size=rO, ball_color=orange, **atom))
    for r in O])

objects.extend([(R, dict(line_width=0.03, color=orange)) for R in bonds])

r = rRu / np.linalg.norm(a7[1])

arrow = {'very thick': True, '->': True}

objects.append(([a7[1] + a7[0] * r, a7[1] + a7[0] * (1 - r)], arrow))
objects.append(([a7[1] + a7[1] * r, a7[1] + a7[1] * (1 - r)], arrow))
objects.append(([a7[1] + a7[2] * r, a7[1] + a7[2] * (1 - r)], arrow))

arrow = {'very thick': True, 'color': 'gray'}
tip = {'->': True, **arrow}

objects.append(([[0, rRu, 0], [0, a / 2 - rO, 0]], arrow))
objects.append(([[0, a / 2 + rO, 0], [0, a - rRu, 0]], tip))

objects.append(([[rRu, 0, 0], [a / 2 - rO, 0, 0]], arrow))
objects.append(([[a / 2 + rO, 0, 0], [a - rRu, 0, 0]], tip))

zO = sorted([O[group[0], 2] for group in elphmod.misc.group(O[:, 2])])[1]
zSr = sorted([Sr[group[0], 2] for group in elphmod.misc.group(Sr[:, 2])])[1]

objects.append(([[0, 0, rRu], [0, 0, zO - rO]], arrow))
objects.append(([[0, 0, zO + rO], [0, 0, zSr - rSr]], arrow))
objects.append(([[0, 0, zSr + rSr], [0, 0, c - zSr - rSr]], arrow))
objects.append(([[0, 0, c - zSr + rSr], [0, 0, c - zO + rO]], arrow))
objects.append(([[0, 0, c - zO - rO], [0, 0, c - rRu]], tip))

objects.append(([[a, a, zSr]],
    dict(content='Sr', color=purple, above='%smm' % (rSr / scale))))

objects.append(([[a, a, 0]],
    dict(content='Ru', color=magenta, above='%gmm' % (rRu / scale))))

objects.append(([[a, a, zO]],
    dict(content='O', color=orange, above='%gmm' % (rO / scale))))

objects.append(([a7[1] + a7[0] / 2], dict(content=r'$\vec a_1$', left=True)))
objects.append(([a7[1] + a7[1] / 2], dict(content=r'$\vec a_2$', left=True)))
objects.append(([a7[1] + a7[2] / 2], dict(content=r'$\vec a_3$', left=True)))

objects.append(([[a / 2, 0, 0]],
    dict(content=r"$\vec a_1'$", color='gray', above='1mm')))

objects.append(([[0, a / 2, 0]],
    dict(content=r"$\vec a_2'$", color='gray', below=True)))

objects.append(([[0, 0, c / 2]],
    dict(content=r"$\vec a_3'$", color='gray', right=True)))

objects = storylines.project(objects,
    R=np.array([-3.5 * a, 4.5 * a, 0.8 * c]),
    T=np.array([0.5 * a, 0.5 * a, 0.5 * c]))

plot = storylines.Plot(
    label='a',

    packages=['bm', r'\let\vec\bm'],

    style='APS',
    font='Utopia',

    xyaxes=False,
    height=0,
    margin=bottom,
)

plot.width = plot.double / 4

for R, style in objects:
    if 'content' in style:
        plot.node(*R[0][:2], style.pop('content'), **style)
    else:
        plot.line(*list(zip(*R))[:2], **style)

plot.save('fig1a.pdf')

G0 = [0.0, 0.0]

M0 = [0.0, 0.5]
M1 = [0.5, 0.0]

S0 = [+0.0, +0.5 + d]
S1 = [-0.5 - d, +0.0]
S2 = [+0.0, -0.5 - d]
S3 = [+0.5 + d, +0.0]

Z0 = [0.0, 1.0]

X0 = [+0.5, +0.5]
X1 = [-0.5, +0.5]
X2 = [-0.5, -0.5]
X3 = [+0.5, -0.5]

R0 = [+0.5 - d, +0.5 + d]
R1 = [-0.5 + d, +0.5 + d]
R2 = [-0.5 - d, +0.5 - d]
R3 = [-0.5 - d, -0.5 + d]
R4 = [-0.5 + d, -0.5 - d]
R5 = [+0.5 - d, -0.5 - d]
R6 = [+0.5 + d, -0.5 + d]
R7 = [+0.5 + d, +0.5 - d]

plot = storylines.Plot(
    label='b',

    style='APS',
    font='Utopia',

    xyaxes=False,
    height=0,
    margin=bottom,

    xmin=-0.6,
    xmax=+0.6,

    ymin=-1.0,
    ymax=+2.0,

    thick=True,
)

plot.width = plot.double / 4

def shift(*points):
    return list(zip(*np.add(points, D)))

marks = dict(mark='*', only_marks=True, cut=True)

D = [0, 0]

for D[0] in SC:
    for D[1] in SC:
        plot.line(*shift(X0, X1, X2, X3, X0), cut=True)

        plot.line(*shift(M0, M1), **marks)

for line in plot.lines:
    if line['code'] is None:
        line['options']['color'] = 'lightgray'
    else:
        line['code'] = line['code'].replace(']', ', color=lightgray]')

for D[0] in SC:
    for D[1] in SC:
        if sum(D) % 2:
            continue

        plot.line(*shift(R0, R1, R2, R3, R4, R5, R6, R7, R0), cut=True)

        plot.line(*shift(G0), **marks)
        plot.line(*shift(S0, S1, S2, S3), **marks)
        plot.line(*shift(Z0), **marks)
        plot.line(*shift(X0, X1), **marks)
        plot.line(*shift(R0, R2, R4, R6), **marks)

plot.node(*G0, r'$\Gamma$', below=True)
plot.node(*M0, 'M', below_left=True)
plot.node(*S0, 'S', above_left=True)
plot.node(*Z0, 'Z', above=True)
plot.node(*R0, 'R', below_left=True)
plot.node(*X0, 'X', above_right=True)

plot.line(*list(zip(G0, M0, S0, Z0, R0, X0, G0)), densely_dashed=True)

plot.save('fig1b.pdf')

height = plot.height

settings = dict(
    packages=['bm', r'\let\vec\bm'],

    style='APS',
    font='Utopia',

    width=plot.double / 2,
    height=(3 * margin + bottom - height) / 2,

    margin=margin,
    left=left,

    xticks=list(zip(x[corners], [r'$\Gamma$', r'M\,\,', r'\,\,S', 'Z', r'R\,',
        r'\,X', r'$\Gamma$'])),

    lbox=True,
)

plot = storylines.Plot(
    label='c',

    ymin=-3.0,
    ymax=+3.0,
    ystep=1.0,

    xmarks=False,

    ylabel=r'Electron energy $\varepsilon_{\vec k n} - \mu_0$ (eV)',

    lpos='lt',
    lopt='below right=2mm',

    **settings,
)

plot.grids()

fatband = dict(thickness=0.04, protrusion=10.0, cut=True)

colors = ['black', orange, magenta]
labels = [
    '$e_g$ ($d_{z^2}, d_{x^2 - y^2}$)',
    '$t_{2 g}$ ($d_{x z}, d_{y z}$)',
    '$t_{2 g}$ ($d_{x y}$)',
]

m = np.argmin(abs(e0[:, 0]))

for n in range(0, len(e0), 2):
    if n in {m, m + 2}:
        plot.fatband(x0, e0[n], color=colors[0], label=labels[0], **fatband)
    else:
        plot.line(x0, e0[n], color='gray', cut=True)

U2 = np.empty((len(x), el.size, 2))

U2[:, :, 0] = np.sum(abs(U[:, :4, :]) ** 2, axis=1)
U2[:, :, 1] = np.sum(abs(U[:, 4:, :]) ** 2, axis=1)

for n in range(0, 6, 2):
    plot.compline(x, e[:, n], U2[:, n], colors=colors[1:], labels=labels[1:],
        **fatband)

plot.axes()

plot.save('fig1c.pdf')

plot = storylines.Plot(
    label='d',

    bottom=bottom,

    ymin=0.0,
    ymax=97.0,
    ystep=20.0,

    ylabel=r'Phonon energy $\omega_{\vec q \nu}$ (meV)',

    lcol=3,
    lpos='ccrmt',
    lrmo=True,
    lwid=2.3,

    **settings,
)

plot.grids()

colors = [purple, orange, 'black']

for nu in range(ph.size):
    plot.compline(x, w[:, nu], pol[:, nu], colors, 'LTZ', **fatband)

plot.axes()

X, Y = np.loadtxt('ref/braden_x00.txt').T

plot.line(X * x[corners[3]], 1e3 * Y * elphmod.misc.THz, label='Experiment',
    **marks)

X, Y = np.loadtxt('ref/braden_xx0.txt').T

plot.line(x[corners[-1]] - X * 2 * (x[corners[-1]] - x[corners[-2]]),
    1e3 * Y * elphmod.misc.THz, **marks)

plot.save('fig1d.pdf')

storylines.combine('fig1cd.pdf', ['fig1c', 'fig1d'], columns=1)
storylines.combine('fig1.pdf', ['fig1a', 'fig1b', 'fig1cd'], align=1.0)
