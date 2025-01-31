#!/bin/bash
#SBATCH --partition=epyc
#SBATCH --job-name=beam
#SBATCH --time=10-0:0:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G 
# SBATCH --mail-type=end
# SBATCH --mail-user=yyang440@ucr.edu

module unload miniconda3/py39_4.12.0

hostname
date

python -u beam_search.py --data_dir=../data/cosmo_10p_Box25_Part75_data --len_slice=3 --n_select_slc=7 --beams=16 --n_optimization_restarts=20 --print_all=1 --output_file=best_slices.txt # only L2
# wait
date
