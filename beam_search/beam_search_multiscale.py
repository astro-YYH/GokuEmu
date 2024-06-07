import sys
sys.path.append("../")

# command: python beam_search.py --beams=1 --n_optimization_restarts=3

import argparse
import numpy as np
# from matplotlib import pyplot as plt

import datetime
import time

from matter_multi_fidelity_emu.data_loader import PowerSpecsMultiRedshift

# from matter_multi_fidelity_emu.gpemulator_singlebin import _map_params_to_unit_cube as input_normalize

from trainset_optimize.optimize import select_slices_scales_redshifts

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
    parser.add_argument("--data_dir1", type=str, default="../data/cosmo_11p_Box100_Part75_data")
    parser.add_argument("--data_dir2", type=str, default="../data/cosmo_11p_Box25_Part75_data")
    
    parser.add_argument("--len_slice", type=int, default=3)
    parser.add_argument("--n_select_slc", type=int, default=3)
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--n_optimization_restarts", type=int, default=5)
    parser.add_argument("--output_file", type=str, default="best_slices.txt")

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
data_folder1 = args.data_dir1
data_folder2 = args.data_dir2


len_slice = args.len_slice
n_select_slc = args.n_select_slc
beams = args.beams
n_optimization_restarts = args.n_optimization_restarts
output_file = args.output_file

data1 = PowerSpecsMultiRedshift(folder=data_folder1, X_file=X_file, Y_base=Y_base)

X1 = data1.X_norm
Y1 = data1.Y

data2 = PowerSpecsMultiRedshift(folder=data_folder2, X_file=X_file, Y_base=Y_base)

X2 = data2.X_norm
Y2 = data2.Y

scale_factors = data1.scale_factors
redshifts = 1 / scale_factors - 1
redshifts = np.round(redshifts, decimals=1)
print("Redshifts: ", redshifts)

Xs = [X1, X2]
Ys = [Y1, Y2]

ind_selected, loss = select_slices_scales_redshifts(Xs, Ys, len_slice=len_slice, n_select_slc=n_select_slc, beams=beams, n_optimization_restarts=n_optimization_restarts)
# ind_selected, loss = select_slices(X, Y, len_slice=3, n_select_slc=3, beams=1, n_optimization_restarts=3)
# print("Selected indices:", ind_selected)
# print("Loss:", loss)

end_time = time.time()
elapsed_time = (end_time - start_time) / 60

ind_slc = slc_ind(ind_selected, len_slice)
ind_selected = [element for row in ind_selected for element in row]

formatted_info = "%d  %d  %d  %d  %s  %s  %.6e %.2f" % (len_slice, n_select_slc, beams, n_optimization_restarts, str(ind_slc), str(ind_selected), loss, elapsed_time)

write_output(output_file, formatted_info)

print("Date and time:", datetime.datetime.now())