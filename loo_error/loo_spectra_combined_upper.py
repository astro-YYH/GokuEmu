import sys
import os
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from mpi4py import MPI
import itertools
from error_function.dgmgp_error import generate_data
import math
# import contextlib
# import io
from matter_multi_fidelity_emu.gpemulator_singlebin import SingleBindGMGP 

def spec_leave(L1HF_dir, L2HF_dir, l, n_optimization_restarts=20, num_processes=5):
    data_1, data_2 = generate_data(folder_1=L1HF_dir, folder_2=L2HF_dir)
    lg_k = np.loadtxt(os.path.join(L1HF_dir, 'kf.txt'), usecols=(0))
    HF_wide_inds_orig = [54, 55, 56, 240, 241, 242, 522, 523, 524, 207, 208, 209, 300, 301, 302, 24, 25, 26, 72, 73, 74]
    HF_narrow_inds_orig = [144, 145, 146, 168, 169, 170, 195, 196, 197, 204, 205, 206, 336, 337, 338]
    # HF_narrow_inds_orig + 564 every element
    HF_narrow_inds_orig = [i + 564 for i in HF_narrow_inds_orig]
    # combine the two lists
    HF_inds_orig = HF_wide_inds_orig + HF_narrow_inds_orig
    HF_inds_orig.sort()   # must sort
    # highres: log10_ks; lowres: log10_k
    # we are emulating data_1's highre
    HF_inds = np.arange(data_1.X_train_norm[1].shape[0])
    HF_inds = np.delete(HF_inds, l)

    LF_inds = np.arange(data_1.X_train_norm[0].shape[0])
    LF_inds = np.delete(LF_inds, HF_inds_orig[l])

    print("training with L1: %d, L2: %d, HF: %d" % (len(LF_inds), len(LF_inds), len(HF_inds)))
    # with contextlib.redirect_stdout(io.StringIO()):  # Redirect stdout to a null stream
    dgmgp = SingleBindGMGP(
            X_train=[data_1.X_train_norm[0][LF_inds], data_2.X_train_norm[0][LF_inds], data_1.X_train_norm[1][HF_inds]],  # L1, L2, HF
            Y_train=[data_1.Y_train_norm[0][LF_inds], data_2.Y_train_norm[0][LF_inds], data_1.Y_train_norm[1][HF_inds]],
            n_fidelities=2,
            n_samples=400,
            optimization_restarts=n_optimization_restarts,
            ARD_last_fidelity=False,
            parallel=True,
            num_processes=num_processes
            )
    x_test = data_1.X_test_norm[0][l]
    x_test = x_test[None, :]
    lg_P_mean, lg_P_var = dgmgp.predict(x_test)
    return lg_k, lg_P_mean[0], lg_P_var[0]

def loo_spec(L1HF_base, L2HF_base, lz, outdir, num_processes=5):
    l = lz[0] # e.g., 0, 1, ...
    z = lz[1] # 0, 0.2
    L1HF_dir = L1HF_base + '_z%s' % z
    L2HF_dir = L2HF_base + '_z%s' % z
    print('training the emulator based on the data of z =', z, 'excluding HF', l,'\n')
    lg_k, lg_P_mean, lg_P_var = spec_leave(L1HF_dir, L2HF_dir, l, num_processes=num_processes)
    P_mode = 10**lg_P_mean * np.exp(-lg_P_var * (np.log(10)) **2)
    print('predicted the matter power spectrum at z =', z, 'in cosmology HF', l, '\n')
    # Combine arrays vertically to create a 2D array where each array is a column
    header_str = 'lg_k, mean(lg_P), var(lg_P), mode(P)'
    combined_array = np.column_stack((lg_k, lg_P_mean, lg_P_var, P_mode))
    # Save the combined array to a text file, with each array as a column
    savepath = os.path.join(outdir,'matter_pow_z%s_l%d.txt' % (z, l))
    np.savetxt(savepath, combined_array, fmt='%f', header=header_str)

num_processes=int(sys.argv[1])

L1HF_base = '../data/combined/matter_power_1128_Box1000_Part750_36_Box1000_Part3000' 
L2HF_base = '../data/combined/matter_power_1128_Box250_Part750_36_Box1000_Part3000' 

outdir='loo_combined_upper'

zs = ['0', '0.2', '0.5', '1', '2', '3']
# zs = ['0', '0.2', '0.5']
# zs = ['1', '2', '3']
leaves = np.arange(36)
lz_combs = list(itertools.product(leaves, zs))

# only a half left
# mid = 24
# lz_combs = lz_combs[mid:]

if __name__ == "__main__":
    n_tasks = len(lz_combs)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size() # this does not work on hpcc, weird, always size == 1
    # size = mpi_processes
    if rank == 0:
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        print('mpi size', size)
    # Calculate tasks per rank
    tasks_per_rank = math.ceil(n_tasks / size)
    print('tasks_per_rank', tasks_per_rank)
    # Determine the start and end index of tasks for the current rank
    start_index = rank * tasks_per_rank
    end_index = min(start_index + tasks_per_rank, n_tasks)

    # Distribute the tasks evenly across ranks
    tasks_for_this_rank = [i for i in range(start_index, end_index)]

    # Each rank executes its assigned tasks
    for task_id in tasks_for_this_rank:
        print(task_id)
        loo_spec(L1HF_base,L2HF_base,lz_combs[task_id], outdir, num_processes=num_processes)

    # Ensure all ranks finish their tasks before ending
