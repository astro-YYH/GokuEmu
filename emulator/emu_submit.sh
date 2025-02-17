#!/bin/bash
#SBATCH --job-name=emu-train
#SBATCH --time=0-4:00:00
# SBATCH --mem=128G
#SBATCH --nodes=6   # no more than num redshifts
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gg
#SBATCH -A AST21005

source ~/.bashrc
conda activate gpy-env
which python

hostname
date
echo '--job-name=emu-train'
# ibrun python -u matter_pow_emu.py --outdir Goku-W_single_param_dependence
# ibrun python -u matter_pow_emu_train.py --outdir pre-trained/goku  # GokuEmu
# ibrun python -u matter_pow_emu_train.py --L1HF_base ../data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000 --L2HF_base ../data/narrow/matter_power_564_Box250_Part750_15_Box1000_Part3000 --outdir pre-trained/goku-n  # GokuEmu-n
ibrun python -u matter_pow_emu_train.py --L1HF_base ../data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300 --L2HF_base ../data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300 --outdir pre-trained/goku-pre-n  # GokuEmu-pre-n
# ibrun python -u matter_pow_emu_train.py --L1HF_base ../data/matter_power_564_Box1000_Part750_21_Box1000_Part3000 --L2HF_base ../data/matter_power_564_Box250_Part750_21_Box1000_Part3000 --outdir pre-trained/goku-w  # GokuEmu-w
date

