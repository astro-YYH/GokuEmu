#!/bin/bash
#SBATCH --job-name=emu
#SBATCH --time=0-00:05:00
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10 # not working
#SBATCH --ntasks-per-node=2
#SBATCH --partition=epyc
#SBATCH --exclusive
# SBATCH -A AST21005

# module unload miniconda3/py39_4.12.0

hostname
date
echo '--job-name=emu'
python -u emu_test.py
date
