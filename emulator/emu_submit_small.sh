#!/bin/bash
#SBATCH --job-name=narrow-pre
#SBATCH --time=0-12:00:00
# SBATCH --mem=128G
#SBATCH --nodes=2   # no more than num redshifts
#SBATCH --ntasks-per-node=3
#SBATCH --partition=small
#SBATCH -A AST21005

source ~/.bashrc
conda activate gpy-env

hostname
date
echo '--job-name=narrow'
ibrun python -u matter_pow_emu_narrow_pre.py --outdir narrow_pre_test
date

