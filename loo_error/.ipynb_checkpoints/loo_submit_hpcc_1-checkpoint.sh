#!/bin/bash
#SBATCH --job-name=loo_spec
#SBATCH --time=4-00:00:00
#SBATCH --mem=250G
#SBATCH --nodes=1
# SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=highmem

module unload miniconda3/py39_4.12.0

hostname
date
echo '--job-name=loo_spec_3d'
# mpirun python -u loo_spectra_test.py 1 2
python -u loo_spectra_test_1.py 1
date
