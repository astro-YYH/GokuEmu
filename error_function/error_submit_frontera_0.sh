#!/bin/bash
#SBATCH --job-name=err_0
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
# SBATCH --ntasks-per-node=56
#SBATCH -A AST21005
#SBATCH --partition=small

hostname
date
echo job-name=err_0
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=3 --num_HF=3 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=3 --num_HF=6 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=3 --num_HF=9 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=3 --num_HF=12 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=3 --num_HF=15 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=3 --num_HF=18 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=3 --num_HF=21 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=3 --num_HF=24 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=3 --num_HF=27 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=6 --num_HF=3 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=6 --num_HF=6 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=6 --num_HF=9 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=6 --num_HF=12 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=6 --num_HF=15 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=6 --num_HF=18 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=6 --num_HF=21 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=6 --num_HF=24 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=6 --num_HF=27 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=9 --num_HF=3 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=9 --num_HF=6 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=9 --num_HF=9 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=9 --num_HF=12 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=9 --num_HF=15 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=9 --num_HF=18 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=9 --num_HF=21 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=9 --num_HF=24 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=9 --num_HF=27 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=12 --num_HF=3 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=12 --num_HF=6 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=12 --num_HF=9 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=12 --num_HF=12 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=12 --num_HF=15 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=12 --num_HF=18 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=12 --num_HF=21 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=12 --num_HF=24 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=12 --num_HF=27 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=15 --num_HF=3 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=15 --num_HF=6 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=15 --num_HF=9 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=15 --num_HF=12 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=15 --num_HF=15 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=15 --num_HF=18 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=15 --num_HF=21 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=15 --num_HF=24 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=15 --num_HF=27 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=18 --num_HF=3 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=18 --num_HF=6 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=18 --num_HF=9 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=18 --num_HF=12 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
python -u dgmgp_error_singlebin_z0z3.py --data_dir=../data --L1HF_base=matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_297_Box25_Part75_27_Box100_Part300 --num_LF=18 --num_HF=15 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_pre_frontera.txt &
wait
date
