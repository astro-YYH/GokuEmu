#!/bin/bash
#SBATCH --job-name=err_12
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
# SBATCH --ntasks-per-node=56
#SBATCH -A AST21005
#SBATCH --partition=small

hostname
date
echo job-name=err_12
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=201 --num_HF=21 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=201 --num_HF=24 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=201 --num_HF=27 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=204 --num_HF=3 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=204 --num_HF=6 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=204 --num_HF=9 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=204 --num_HF=12 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=204 --num_HF=15 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=204 --num_HF=18 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=204 --num_HF=21 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=204 --num_HF=24 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=204 --num_HF=27 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=207 --num_HF=3 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=207 --num_HF=6 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=207 --num_HF=9 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=207 --num_HF=12 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=207 --num_HF=15 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=207 --num_HF=18 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=207 --num_HF=21 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=207 --num_HF=24 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=207 --num_HF=27 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=210 --num_HF=3 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=210 --num_HF=6 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=210 --num_HF=9 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=210 --num_HF=12 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=210 --num_HF=15 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=210 --num_HF=18 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=210 --num_HF=21 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=210 --num_HF=24 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=210 --num_HF=27 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=213 --num_HF=3 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=213 --num_HF=6 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=213 --num_HF=9 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=213 --num_HF=12 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=213 --num_HF=15 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=213 --num_HF=18 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=213 --num_HF=21 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=213 --num_HF=24 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=213 --num_HF=27 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=216 --num_HF=3 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=216 --num_HF=6 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=216 --num_HF=9 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=216 --num_HF=12 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=216 --num_HF=15 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=216 --num_HF=18 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=216 --num_HF=21 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=216 --num_HF=24 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=216 --num_HF=27 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=219 --num_HF=3 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
python -u dgmgp_error.py --data_dir=../data/narrow --L1HF_base=matter_power_378_Box100_Part75_27_Box100_Part300 --L2HF_base=matter_power_378_Box25_Part75_27_Box100_Part300 --num_LF=219 --num_HF=6 --n_optimization_restarts=20 --parallel=0 --output_file=error_function_goku_narrow_pre_frontera.txt &
wait
date
