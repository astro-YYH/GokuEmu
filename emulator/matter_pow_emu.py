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
import argparse
import shutil

def predict(L1HF_dir, L2HF_dir, X_target, n_optimization_restarts=20, num_processes=10):
    data_1, data_2 = generate_data(folder_1=L1HF_dir, folder_2=L2HF_dir)

    # highres: log10_ks; lowres: log10_k
    # with contextlib.redirect_stdout(io.StringIO()):  # Redirect stdout to a null stream
    dgmgp = SingleBindGMGP(
            X_train=[data_1.X_train_norm[0], data_2.X_train_norm[0], data_1.X_train_norm[1]],  # L1, L2, HF
            Y_train=[data_1.Y_train_norm[0], data_2.Y_train_norm[0], data_1.Y_train_norm[1]],
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
    np.savetxt(file_mode, P_mode, fmt='%e', header=header_str_mode)

L1HF_base = '../data/matter_power_564_Box1000_Part750_21_Box1000_Part3000' 
L2HF_base = '../data/matter_power_564_Box250_Part750_21_Box1000_Part3000' 
num_processes = 1 # multiprocessing cores per MPI rank
n_optimization_restarts = 25 # larger safer possibly

zs = ['0', '0.2', '0.5', '1', '2', '3']
# zs = ['0', '0.2', '0.5', '1', '3']
# data_in = np.loadtxt('input_reference_sensitivity.txt')
data_in = np.loadtxt('input_HF_narrow.txt')  # test against Goku-N HF simulations

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
