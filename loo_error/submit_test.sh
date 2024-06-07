#!/bin/bash
#SBATCH --job-name=loo_spec
#SBATCH --time=0-00:10:00
#SBATCH --mem=25G
#SBATCH --nodes=1
# SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=epyc

module unload miniconda3/py39_4.12.0
# module unload openmpi

hostname
date
echo '--job-name=test'
# mpirun python -u loo_spectra_test.py 1 2
mpirun python test.py
date
