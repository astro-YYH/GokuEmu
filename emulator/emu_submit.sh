#!/bin/bash
#SBATCH --job-name=emu-single
#SBATCH --time=0-8:00:00
# SBATCH --mem=128G
#SBATCH --nodes=2   # no more than num redshifts
#SBATCH --ntasks-per-node=1
#SBATCH --partition=small
#SBATCH -A AST21005

source ~/.bashrc
conda activate gpy-env
which python

hostname
date
echo '--job-name=emu-single'
ibrun python -u matter_pow_emu.py --outdir Goku-W_single_param_dependence
date

