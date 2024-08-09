#!/bin/bash
#SBATCH --job-name=loo_spec
#SBATCH --time=4-00:00:00
# SBATCH --mem=900G
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=14
#SBATCH --partition=epyc

module unload miniconda3/py39_4.12.0
source ~/.bashrc
conda activate gpy-env

hostname
date
echo '--job-name=loo_spec'
# mpirun python -u loo_spectra_test.py 1 2
mpirun python -u loo_spectra_upper.py 1
date
