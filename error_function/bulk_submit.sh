#!/bin/bash

source ~/.bashrc
conda activate gpy-env

# Loop 
# for submit_file in error_submit_frontera*.sh; do
for submit_file in error_submit_wide_frontera*.sh; do
    echo "Submitting $submit_file"
    sbatch $submit_file
done
