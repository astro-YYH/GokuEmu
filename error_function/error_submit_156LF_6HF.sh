#!/bin/bash
#SBATCH --job-name=err156L6H
#SBATCH --time=2-00:00:00
#SBATCH --mem=7G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --partition=intel

hostname
date
echo job-name=err156L6H
python -u dgmgp_error.py --data_dir=/rhome/yyang440/bigdata/tentative_sims/data_for_emu --L1HF_base=matter_power_270_Box100_Part75_18_Box100_Part300 --L2HF_base=matter_power_270_Box25_Part75_18_Box100_Part300 --num_LF=156 --num_HF=6 --n_optimization_restarts=25 --parallel=0 --output_file=error_function.txt 
date
