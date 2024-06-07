#!/bin/bash
#SBATCH --job-name=loo_spec
#SBATCH --time=3-00:00:00
#SBATCH --mem=900G
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=12
#SBATCH --partition=epyc

module unload miniconda3/py39_4.12.0

hostname
date
echo '--job-name=loo_spec_3d'
# mpirun python -u loo_spectra_test.py 1 2
mpirun python -u loo_spectra_upper.py 1
date
