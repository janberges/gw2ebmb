#!/usr/bin/env python3

import elphmod
import numpy as np
import storylines

comm = elphmod.MPI.comm

# work _wsvec.dat into _hr.dat files:

for seedname in 'sro', 'sro_nosoc':
    el = elphmod.el.Model(seedname + '_orig')
    el.standardize()
    el.to_hrdat(seedname)

    # once more to remove vanishing elements and minus signs before zeros:

    el = elphmod.el.Model(seedname)
    el.standardize()
    el.to_hrdat(seedname)

# save ARPES data as grayscale image:

if comm.rank == 0:
    img = np.loadtxt('../ref/tamai_fs.txt', skiprows=1)
else:
    img = None

img = comm.bcast(img)

img = img[img.shape[0] // 2 + 1:, img.shape[0] // 2 - 1:]

img = elphmod.plot.color(img, storylines.colormap(
    (0, storylines.Color(255, 255, 255)),
    (1, storylines.Color(0, 0, 0))), minimum=0, maximum=img.max())

if comm.rank == 0:
    storylines.save('../ref/tamai_fs.png', img[:, :, :1])

# sro{,_nosoc}_orig_{hr,wsvec}.dat and tamai_fs.txt can now be removed
