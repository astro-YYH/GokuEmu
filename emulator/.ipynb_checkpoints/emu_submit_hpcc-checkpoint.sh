#!/bin/bash
#SBATCH --job-name=emu
#SBATCH --time=8-0:00:00
#SBATCH --mem=512G
#SBATCH --nodes=1   # ranks no more than num redshifts
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=2
#SBATCH --partition=epyc

module unload miniconda3/py39_4.12.0
# module unload python3

hostname
date
echo '--job-name=emu'
mpirun python -u matter_pow_emu.py
date
