import sys
sys.path.append("../../matter_emu_dgmgp-MF-Box-yanhui/")
import numpy as np
# import multiprocessing
import itertools
from error_function.dgmgp_error import generate_data
# import contextlib
# import io
from matter_multi_fidelity_emu.gpemulator_singlebin import SingleBindGMGP 

def spec_leave(L1HF_dir, L2HF_dir, l, n_optimization_restarts=20):
    data_1, data_2 = generate_data(folder_1=L1HF_dir, folder_2=L2HF_dir)

    # highres: log10_ks; lowres: log10_k
    # we are emulating data_1's highre
    HF_inds = np.arange(data_1.X_train_norm[1].shape[0])
    HF_inds = np.delete(HF_inds, l)
    # with contextlib.redirect_stdout(io.StringIO()):  # Redirect stdout to a null stream
    dgmgp = SingleBindGMGP(
            X_train=[data_1.X_train_norm[0], data_2.X_train_norm[0], data_1.X_train_norm[1][HF_inds]],  # L1, L2, HF
            Y_train=[data_1.Y_train_norm[0], data_2.Y_train_norm[0], data_1.Y_train_norm[1][HF_inds]],
            n_fidelities=2,
            n_samples=400,
            optimization_restarts=n_optimization_restarts,
            ARD_last_fidelity=False,
            parallel=False
            )
    x_test = data_1.X_test_norm[0][l]
    x_test = x_test[None, :]
    lg_k_mean, lg_P_var = dgmgp.predict(x_test)
    return lg_k_mean[0], lg_P_var[0]

def loo_spec(L1HF_base, L2HF_base, lz):
    l = lz[0] # e.g., 0, 1, ...
    z = lz[1] # 0, 0.2
    L1HF_dir = L1HF_base + '_z%s' % z
    L2HF_dir = L2HF_base + '_z%s' % z
    print('training the emulator based on the data of z =', z, 'excluding HF', l)
    lg_k, lg_P = spec_leave(L1HF_dir, L2HF_dir, l)
    print('predicted the matter power spectrum at z =', z, 'in cosmology HF', l)
    # Combine arrays vertically to create a 2D array where each array is a column
    header_str = 'mean predicted from Gaussian processes: lg_k, lg_P'
    combined_array = np.column_stack((lg_k, lg_P))
    # Save the combined array to a text file, with each array as a column
    np.savetxt('matter_pow_z%s_l%d.txt' % (z, l), combined_array, fmt='%f', header=header_str)

L1HF_base = '/rhome/yyang440/bigdata/cosmo_11p_sims/data_for_emu/matter_power_1120_Box1000_Part750_8_Box1000_Part3000' 
L2HF_base = '/rhome/yyang440/bigdata/cosmo_11p_sims/data_for_emu/matter_power_1120_Box250_Part750_8_Box1000_Part3000' 


zs = ['0', '0.2', '0.5', '1', '2', '3']
leaves = np.arange(8)
lz_combs = list(itertools.product(leaves, zs))

lz = lz_combs[0]
  # List to keep track of the processes
# processes = []


loo_spec(L1HF_base, L2HF_base, lz)