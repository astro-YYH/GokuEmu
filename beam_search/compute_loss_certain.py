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
from trainset_optimize.optimize import TrainSetOptimize, loss_redshifts

n_optimization_restarts = 20
parallel_redshift = True

data = PowerSpecsMultiRedshift(folder='../data/cosmo_10p_Box250_Part750_data', X_file='train_input.txt', Y_base='train_output')

X = data.X_norm
Y = data.Y

train_opt_zs = []
for y in Y:
    train_opt_zs.append(TrainSetOptimize(X=X, Y=y))

# slices
ind_slc = np.array([136,  36,  56, 186,  68,  15,  64])
selected_ind = np.concatenate([ind_slc * 3, ind_slc * 3 + 1, ind_slc * 3 + 2]) 

num_samples = X.shape[0]
ind = np.zeros(num_samples, dtype=bool)
ind[selected_ind] = True

print('Computing loss for slice combination', ind_slc)
print('points', selected_ind)

loss = loss_redshifts(train_opt_zs, ind, n_optimization_restarts=n_optimization_restarts, parallel=parallel_redshift)

print('loss =',loss)
