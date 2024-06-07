#!/bin/bash
#SBATCH --job-name=loo_3
#SBATCH --time=0-26:00:00
# SBATCH --mem=128G
#SBATCH --nodes=2
# SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=small
#SBATCH -A AST21005

# module unload miniconda3/py39_4.12.0
module unload python3

hostname
date
echo '--job-name=loo_spec np3'
ibrun python -u loo_spectra.py 3
date
