#!/bin/bash
#SBATCH --job-name=err
#SBATCH --time=0-02:00:00
# SBATCH --mem=7G
#SBATCH --nodes=1
# SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --partition=development
#SBATCH -A AST21005

hostname
date
echo job-name=err_test
# python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=12 --num_HF=3 --n_optimization_restarts=15 --output_file='test.txt'
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=297 --num_HF=27 --n_optimization_restarts=15 --output_file='test.txt'
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=102 --num_HF=12 --n_optimization_restarts=15 --output_file='test.txt'
date
