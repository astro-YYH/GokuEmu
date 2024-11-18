#!/bin/bash
#SBATCH --job-name=loo_up
#SBATCH --time=0-0:10:00
# SBATCH --mem=128G
#SBATCH --nodes=2
# SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=development
#SBATCH -A AST21005

# module unload miniconda3/py39_4.12.0
# module unload python3
source ~/.bashrc
conda activate gpy-env

hostname
date
echo '--job-name=loo_up'
# ibrun python -u loo_spectra_upper.py 5
ibrun python -u loo_spectra_narrow_upper.py 5
date
