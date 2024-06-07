#!/bin/bash

# Loop 
for submit_file in second_submit*.sh; do
    echo "Submitting $submit_file"
    sbatch $submit_file
done
