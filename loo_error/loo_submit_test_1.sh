#!/bin/bash
#SBATCH --job-name=loo_spec
#SBATCH --time=0-02:00:00
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --partition=epyc

module unload miniconda3/py39_4.12.0

hostname
date
echo '--job-name=loo_spec_1'
python -u loo_spectra_single_lz_test.py
date
