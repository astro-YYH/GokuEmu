# print all the choices of slices (not only the best one)

import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# command: python beam_search.py --beams=1 --n_optimization_restarts=3

import argparse
import numpy as np
# from matplotlib import pyplot as plt

import datetime
import time

from matter_multi_fidelity_emu.data_loader import PowerSpecsMultiRedshift

# from matter_multi_fidelity_emu.gpemulator_singlebin import _map_params_to_unit_cube as input_normalize

from trainset_optimize.optimize import select_slices_redshifts

def write_output(file_path, output):
    with open(file_path, 'a+') as file:
        file.seek(0)
        content = file.read()
        if len(content) == 0:
            file.write("# 1. len_slice  2. n_select_slc  3. beams  4. n_optimization_restarts  5. best slices  6. best points  7. loss (variance)  8. time spent (min)\n")
        file.write(output + "\n")

def slc_ind(points, len_slc):
    ind_slc = []
    for slc in points:
        ind = int(slc[0] / len_slc)
        ind_slc.append(ind)
    return ind_slc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--X_file", type=str,
        default="train_input.txt")
    parser.add_argument("--Y_base", type=str,
        default="train_output")
    parser.add_argument("--data_dir", type=str, default="../data/cosmo_11p_Box100_Part75_interp50")
    
    parser.add_argument("--len_slice", type=int, default=3)
    parser.add_argument("--n_select_slc", type=int, default=3)
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--n_optimization_restarts", type=int, default=5)
    parser.add_argument("--output_file", type=str, default="best_slices.txt")
    parser.add_argument("--print_all", type=int, default=0)
    parser.add_argument("--parallel_redshift", type=int, default=0)

    args = parser.parse_args()


print("Date and time:", datetime.datetime.now())
start_time = time.time()
# set a random number seed to reproducibility
np.random.seed(0)

# from itertools import combinations

# from trainset_optimize.optmize import TrainSetOptimize
# from trainset_optimize.optmize import select_slices


X_file = args.X_file
Y_base = args.Y_base
data_folder = args.data_dir

len_slice = args.len_slice
n_select_slc = args.n_select_slc
beams = args.beams
n_optimization_restarts = args.n_optimization_restarts
output_file = args.output_file
print_all = args.print_all
parallel_redshift = args.parallel_redshift

print("len_slice  n_select_slc  beams  n_optimization_restarts  output_file")
print(len_slice, n_select_slc, beams, n_optimization_restarts, output_file)

data = PowerSpecsMultiRedshift(folder=data_folder, X_file=X_file, Y_base=Y_base)

X = data.X_norm
Y = data.Y

scale_factors = data.scale_factors
redshifts = 1 / scale_factors - 1
redshifts = np.round(redshifts, decimals=1)
print("Redshifts: ", redshifts)

ind_selected, loss = select_slices_redshifts(X, Y, len_slice=len_slice, n_select_slc=n_select_slc, beams=beams, n_optimization_restarts=n_optimization_restarts, print_all=print_all, parallel_redshift=parallel_redshift)
# ind_selected, loss = select_slices(X, Y, len_slice=3, n_select_slc=3, beams=1, n_optimization_restarts=3)
# print("Selected indices:", ind_selected)
# print("Loss:", loss)

end_time = time.time()
elapsed_time = (end_time - start_time) / 60



if not print_all:
    ind_slc = slc_ind(ind_selected, len_slice)
    ind_selected = [element for row in ind_selected for element in row]

    formatted_info = "%d  %d  %d  %d  %s  %s  %.6e %.2f" % (len_slice, n_select_slc, beams, n_optimization_restarts, str(ind_slc), str(ind_selected), loss, elapsed_time)
else:
    for i in range(len(loss)):
        ind_slc_el = slc_ind(ind_selected[i], len_slice)
        ind_selected_el = [element for row in ind_selected[i] for element in row]
        if i==0:
            formatted_info = "%d  %d  %d  %d  %s  %s  %.6e %.2f" % (len_slice, n_select_slc, beams, n_optimization_restarts, str(ind_slc_el), str(ind_selected_el), loss[i], elapsed_time)
            continue
        formatted_info = formatted_info + "\n%d  %d  %d  %d  %s  %s  %.6e %.2f" % (0, 0, 0, 0, str(ind_slc_el), str(ind_selected_el), loss[i], 0.)

write_output(output_file, formatted_info)

print("Date and time:", datetime.datetime.now())