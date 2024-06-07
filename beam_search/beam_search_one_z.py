import sys
sys.path.append("/rhome/yyang440/bigdata/matter_emu_dgmgp/")

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import os

from matter_multi_fidelity_emu.gpemulator_singlebin import (
    SingleBinGP,
    SingleBinLinearGP,
    SingleBinNonLinearGP,
)
from matter_multi_fidelity_emu.data_loader import PowerSpecsSingle

from matter_multi_fidelity_emu.gpemulator_singlebin import _map_params_to_unit_cube as input_normalize


# set a random number seed to reproducibility
np.random.seed(0)

# from itertools import combinations

# from trainset_optimize.optmize import TrainSetOptimize
from trainset_optimize.optmize import select_slices


# load a single-fidelity data set

X_file = "train_input_fidelity_0.txt"
Y_file = "train_output_fidelity_0.txt"
data_folder = "/rhome/yyang440/bigdata/matter_emu_dgmgp/data/50_LR_3_HR/"

data = PowerSpecsSingle(folder=data_folder, X_file=X_file, Y_file=Y_file)

X = data.X_norm
Y = data.Y

ind_selected, loss = select_slices(X, Y, len_slice=2, n_select_slc=3, beam=1, n_optimization_restarts=3)

# print("Selected indices:", ind_selected)
# print("Loss:", loss)

