#!/bin/bash
#SBATCH --job-name=emu-EE2
#SBATCH --time=0-8:00:00
# SBATCH --mem=128G
#SBATCH --nodes=6   # no more than num redshifts
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH -A AST21005

hostname
date
echo '--job-name=emu-EE2'
ibrun python -u matter_pow_emu.py --outdir EE2_compare
date

