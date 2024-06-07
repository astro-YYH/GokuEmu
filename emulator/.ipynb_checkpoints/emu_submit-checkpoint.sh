#!/bin/bash
#SBATCH --job-name=emu
#SBATCH --time=0-24:00:00
# SBATCH --mem=128G
#SBATCH --nodes=6   # no more than num redshifts
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH -A AST21005

# module unload miniconda3/py39_4.12.0
module unload python3

hostname
date
echo '--job-name=emu'
ibrun python -u matter_pow_emu.py
date

