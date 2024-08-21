#!/bin/bash

# Loop 
# for submit_file in error_submit_frontera*.sh; do
for submit_file in error_submit_narrow_frontera*.sh; do
    echo "Submitting $submit_file"
    sbatch $submit_file
done
