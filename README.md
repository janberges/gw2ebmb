# GW/Eliashberg calculations for Sr₂RuO₄

This directory contains all scripts, input files, and reference data needed to
reproduce the figures shown in the paper:

> *Unified treatment of local dynamical interactions in correlated metals using
  Eliashberg theory* by Jan Berges, Samuel Poncé, Mario Caserta, Nicola Marzari,
  and Tommaso Chiarotti (2026).

## Installation

Install a Fortran compiler, LAPACK/BLAS, and LaTeX, if needed:

    sudo apt install gfortran liblapack-dev texlive-full

Optionally, create a virtual Python environment:

    python3 -m venv venv
    source venv/bin/activate

Install the required Python packages (other versions may also work):

    python3 -m pip install numpy==2.5.3 matplotlib==3.11.2
    python3 -m pip install elphmod==0.36 storylines==0.18

Optionally, install MPI to run the scripts in parallel:

    sudo apt install libopenmpi-dev
    python3 -m pip install mpi4py==4.1.2 --no-binary=mpi4py

Download and compile ebmb (version 3.0.0 was released for this work):

    git clone https://github.com/janberges/ebmb.git
    cd ebmb
    git checkout v3.0.0
    make FC=gfortran FFLAGS='-O3 -fopenmp'
    cd ..

Install ebmb (`PATH` may be set in your `.bashrc`):

    export PATH=$(pwd)/ebmb/bin:$PATH

Install the Python wrapper:

    python3 -m pip install ./ebmb

## Figure scripts

Make all figures:

    make

Note: If other programs are running on your computer, setting `OMP_NUM_THREADS`
to a value below the number of available processors may speed up the task:

    OMP_NUM_THREADS=$(($(nproc)-1)) make

You can also run the scripts in parallel using MPI (the second argument is only
needed to correctly display status bars since version 5.0.0 of OpenMPI):

    make CMD='mpirun -n 4 --stream-buffering 0 python3'

Some results have been precomputed and stored in text files. To recompute them:

    mpirun -n 4 python3 fig2ef.py
    mpirun -n 4 python3 fig4a.py
    mpirun -n 4 python3 fig4b.py
    mpirun -n 4 python3 fig6def.py

## Data files

The subdirectory `data` contains all relevant input and output files of the
first-principles calculations.

### Input data

- `scf.in`, `nscf.in`: Input files for PWscf with SOC
- `scf_nosoc.in`, `nscf_nosoc.in`: Input files for PWscf without SOC
- `ph.in`: Input file for PHonon
- `epw.in`: Input file for EPW
- `respack.in`: Input file for RESPACK (patched with `respack.patch`)

### Output data

- `bands.dat`: Kohn-Sham band structure from PWscf
- `sro_hr.dat`: Hamiltonian from Wannier90 with SOC (via `data.py`)
- `sro_nosoc_hr.dat`: Hamiltonian from Wannier90 without SOC (via `data.py`)
- `dyn*`: Dynamical matrices from PHonon
- `dat.JvsE.00?-00?`: Local, dynamical exchange interaction from RESPACK

### Processed data

- `dos.txt`: Orbital-resolved noninteracting density of states (via `dos.py`)
- `a2fp.txt`: Phonon-mediated interaction (via `a2fp.py`, EPW data not included)
- `a2f{e,ep}.txt`, `muc.txt`: Direct and effective interaction (via `a2f.py`)

## Reference data

The subdirectory `ref` contains reference data extracted from the following
published figures:

- `abramovitch.txt`: Fig. 5a by Abramovitch *et al.*, PRM **7**, 093801 (2023)
- `braden_*.txt`: Fig. 2 by Braden *et al.*, PRB **76**, 014505 (2007)
- `hunter.txt`: Fig. 3a,c by Hunter *et al.*, PRL **131**, 236502 (2023)
- `stricker_*.txt`: Fig. 4 by Stricker *et al.*, PRL **113**, 087404 (2014)
- `tamai_*deg.txt`: Fig. 5 by Tamai *et al.*, PRX **9**, 021048 (2019)
- `tamai_fs.png`: Fig. 1a from the same paper (via `data.py`)
- `tyler.txt`: Fig. 1 by Tyler *et al.*, PRB **58**, R10107 (1998)
