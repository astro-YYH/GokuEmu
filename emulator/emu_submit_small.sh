#!/bin/bash
#SBATCH --job-name=pre-combined
#SBATCH --time=0-24:00:00
# SBATCH --mem=128G
#SBATCH --nodes=1   # no more than num redshifts
#SBATCH --ntasks-per-node=1
#SBATCH --partition=small
#SBATCH -A AST21005

source ~/.bashrc
conda activate gpy-env

hostname
date
echo '--job-name=pre-combined'
# ibrun python -u matter_pow_emu_narrow_pre.py --outdir narrow_pre_test
# matter_pow_emu_pre_combined-W-N.py
ibrun python -u matter_pow_emu_pre_combined-W-N.py --outdir pre_combined_test
date

