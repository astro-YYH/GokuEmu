import sys
import os
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from mpi4py import MPI
import itertools
from error_function.dgmgp_error import generate_data
# import contextlib
# import io
from matter_multi_fidelity_emu.gpemulator_singlebin import SingleBindGMGP 
from matter_multi_fidelity_emu.gpemulator_singlebin import _map_params_to_unit_cube as input_normalize
from matter_multi_fidelity_emu.data_loader_dgmgp import interpolate
import argparse
import shutil

def predict(L1HF_dir, L2HF_dir, X_target, n_optimization_restarts=20, num_processes=10):
    data_1, data_2 = generate_data(folder_1=L1HF_dir, folder_2=L2HF_dir)

    # highres: log10_ks; lowres: log10_k
    # we are emulating data_1's highres
    log10_k_target = data_1.kf
    log10_k_train = data_2.kf
    ind_min = (log10_k_target >= log10_k_train.min()) & (
        log10_k_target <= log10_k_train.max()
    )
# interpolate: interp(log10_k, Y_lf)(log10_k[ind_min])
    data_2.Y_train_norm[0] = interpolate(
    data_2.kf, data_2.Y_train_norm[0], data_1.kf[ind_min]
    )
    data_2.Y_train[0] = interpolate(data_2.kf, data_2.Y_train[0], data_1.kf[ind_min])
    assert data_2.Y_train_norm[0].shape[1] == data_1.kf[ind_min].shape[0]
# HF powerspecs trim to same size as LF
    data_1.Y_train_norm[0] = data_1.Y_train_norm[0][:, ind_min]
    data_1.Y_train[0] = data_1.Y_train[0][:, ind_min]
    data_1.Y_train_norm[1] = data_1.Y_train_norm[1][:, ind_min]
    data_1.Y_train[1] = data_1.Y_train[1][:, ind_min]

    data_1.Y_test[0] = data_1.Y_test[0][:, ind_min]

    kf = data_1.kf[ind_min]
    data_1.kf = kf
    data_2.kf = kf

    LF_inds = [60, 61, 62, 168, 169, 170, 282, 283, 284, 6, 7, 8, 192, 193, 194, 189, 190, 191, 90, 91, 92, 207, 208, 209, 24, 25, 26, 201, 202, 203, 156, 157, 158, 45, 46, 47, 48, 49, 50, 285, 286, 287, 105, 106, 107, 9, 10, 11, 222, 223, 224, 225, 226, 227, 270, 271, 272, 126, 127, 128, 81, 82, 83, 18, 19, 20, 276, 277, 278, 33, 34, 35, 60, 61, 62, 234, 235, 236, 261, 262, 263, 153, 154, 155, 150, 151, 152, 195, 196, 197, 54, 55, 56, 69, 70, 71, 21, 22, 23, 63, 64, 65, 183, 184, 185, 111, 112, 113, 192, 193, 194, 135, 136, 137, 180, 181, 182, 249, 250, 251, 198, 199, 200, 87, 88, 89, 12, 13, 14, 117, 118, 119, 171, 172, 173, 96, 97, 98, 102, 103, 104, 282, 283, 284, 246, 247, 248, 294, 295, 296, 228, 229, 230, 0, 1, 2, 123, 124, 125, 213, 214, 215, 120, 121, 122, 189, 190, 191, 237, 238, 239, 75, 76, 77, 291, 292, 293, 108, 109, 110, 231, 232, 233, 93, 94, 95, 36, 37, 38, 243, 244, 245, 15, 16, 17, 99, 100, 101, 138, 139, 140, 216, 217, 218, 3, 4, 5, 39, 40, 41, 51, 52, 53, 162, 163, 164, 30, 31, 32, 240, 241, 242, 273, 274, 275, 264, 265, 266, 78, 79, 80, 288, 289, 290, 66, 67, 68, 165, 166, 167, 174, 175, 176, 141, 142, 143, 129, 130, 131, 204, 205, 206, 42, 43, 44, 147, 148, 149, 27, 28, 29, 258, 259, 260, 279, 280, 281, 144, 145, 146, 72, 73, 74, 255, 256, 257, 168, 169, 170, 267, 268, 269, 6, 7, 8, 159, 160, 161, 210, 211, 212, 84, 85, 86]
    HF_inds = [6, 7, 8, 15, 16, 17, 24, 25, 26, 0, 1, 2, 21, 22, 23, 18, 19, 20]
    # with contextlib.redirect_stdout(io.StringIO()):  # Redirect stdout to a null stream
    dgmgp = SingleBindGMGP(
            X_train=[data_1.X_train_norm[0][LF_inds], data_2.X_train_norm[0][LF_inds], data_1.X_train_norm[1][HF_inds]],  # L1, L2, HF
            Y_train=[data_1.Y_train_norm[0][LF_inds], data_2.Y_train_norm[0][LF_inds], data_1.Y_train_norm[1][HF_inds]],
            n_fidelities=2,
            n_samples=400,
            optimization_restarts=n_optimization_restarts,
            ARD_last_fidelity=False,
            parallel=False,
            num_processes=num_processes
            )
    X_target_norm = input_normalize(X_target, data_1.parameter_limits)
    lg_P_mean, lg_P_var = dgmgp.predict(X_target_norm)
    return lg_P_mean, lg_P_var

def predict_z(L1HF_base,L2HF_base,z, X_target, n_optimization_restarts,num_processes, outdir):
    L1HF_dir = L1HF_base + '_z%s' % z
    L2HF_dir = L2HF_base + '_z%s' % z
    print('training the emulator based on the data of z =', z)
    lg_P_mean, lg_P_var = predict(L1HF_dir, L2HF_dir, X_target, n_optimization_restarts = n_optimization_restarts,num_processes = num_processes)
    print('predicted the matter power spectrum at z =', z)
        # Combine arrays vertically to create a 2D array where each array is a column
    header_str_mean = 'mean(lg_P) also median/mode'
        # Save the combined array to a text file, with each array as a column
    file_mean = os.path.join(outdir, 'matter_pow_lg_mean_z%s.txt' % (z))
    np.savetxt(file_mean, lg_P_mean, fmt='%f', header=header_str_mean)

    header_str_var = 'var(lg_P)'
        # Save the combined array to a text file, with each array as a column
    file_var = os.path.join(outdir, 'matter_pow_lg_var_z%s.txt' % (z))
    np.savetxt(file_var, lg_P_var, fmt='%f', header=header_str_var)

    header_str_mode = 'mode(P)'
    file_mode = os.path.join(outdir, 'matter_pow_mode_z%s.txt' % (z))
    P_mode = 10**lg_P_mean * np.exp(-lg_P_var * (np.log(10)) **2)
        # Save the combined array to a text file, with each array as a column
    np.savetxt(file_mode, P_mode, fmt='%f', header=header_str_mode)

L1HF_base = '../data/narrow/matter_power_297_Box100_Part75_27_Box100_Part300' 
L2HF_base = '../data/narrow/matter_power_297_Box25_Part75_27_Box100_Part300' 
num_processes = 1 # multiprocessing cores per MPI rank
n_optimization_restarts = 20 # larger safer possibly

zs = ['0', '0.2', '0.5', '1', '2', '3']
# zs = ['0', '0.2', '0.5', '1', '3']
data_in = np.loadtxt('input-narrow-pre.txt')

if len(data_in.shape)==1:
    data_in = data_in[None,:]

if __name__ == "__main__":
    n_tasks = len(zs)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #-------------- Cmd line Args ------------------------------
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--outdir',required=False,default='predictions',type=str,help='output directory')

    args = parser.parse_args()
    
    
    #--------------------------
    outdir = args.outdir

    

    if rank == 0:
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        # save k
        file_k = L1HF_base + '_z0/kf.txt'
        shutil.copy(file_k, os.path.join(outdir, "lgk.txt"))

    # Distribute the tasks evenly across ranks
    tasks_for_this_rank = [i for i in range(rank, n_tasks, size)]

    # Each rank executes its assigned tasks
    for task_id in tasks_for_this_rank:
        predict_z(L1HF_base,L2HF_base,zs[task_id], data_in,n_optimization_restarts,num_processes,outdir)

    # Ensure all ranks finish their tasks before ending
    comm.Barrier()
