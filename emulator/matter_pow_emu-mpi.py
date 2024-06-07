import sys
sys.path.append("../../matter_emu_dgmgp-MF-Box-yanhui/")
import numpy as np
from mpi4py import MPI
import itertools
from error_function.dgmgp_error import generate_data
# import contextlib
# import io
from matter_multi_fidelity_emu.gpemulator_singlebin import SingleBindGMGP 
from matter_multi_fidelity_emu.gpemulator_singlebin import _map_params_to_unit_cube as input_normalize

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
            parallel=True,
            num_processes=num_processes
            )
    X_target_norm = input_normalize(X_target, data_1.parameter_limits)
    lg_P_mean, lg_P_var = dgmgp.predict(X_target_norm)
    return lg_P_mean, lg_P_var

from mpi4py import MPI
import numpy as np

def predict_mpi(L1HF_dir, L2HF_dir, X_target, n_optimization_restarts=20, num_processes=10):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        data_1, data_2 = generate_data(folder_1=L1HF_dir, folder_2=L2HF_dir)
    else:
        data_1 = data_2 = None

    # Broadcast data to all processes
    data_1 = comm.bcast(data_1, root=0)
    data_2 = comm.bcast(data_2, root=0)

    # Split Y_train_norm across processes
    Y_train_1_split_0 = np.array_split(data_1.Y_train_norm[0], size, axis=1)[rank]
    Y_train_2_split_0 = np.array_split(data_2.Y_train_norm[0], size, axis=1)[rank]
    Y_train_1_split_1 = np.array_split(data_1.Y_train_norm[1], size, axis=1)[rank]
    # Y_train_2_split_1 = np.array_split(data_2.Y_train_norm[1], size, axis=1)[rank]
    print('mpi rank: {rank}')
    # Initialize dgmgp with distributed data
    dgmgp = SingleBindGMGP(
        X_train=[data_1.X_train_norm[0], data_2.X_train_norm[0], data_1.X_train_norm[1]],  # L1, L2, HF
        Y_train=[Y_train_1_split_0, Y_train_2_split_0, Y_train_1_split_1],
        n_fidelities=2,
        n_samples=400,
        optimization_restarts=n_optimization_restarts,
        ARD_last_fidelity=False,
        parallel=True,
        num_processes=num_processes
    )

    # Normalize X_target in root and broadcast
    if rank == 0:
        X_target_norm = input_normalize(X_target, data_1.parameter_limits)
    else:
        X_target_norm = None
    X_target_norm = comm.bcast(X_target_norm, root=0)

    # Each process makes a prediction
    lg_P_mean, lg_P_var = dgmgp.predict(X_target_norm)

    # Gather predictions from all processes to the root process
    lg_P_mean_all = comm.gather(lg_P_mean, root=0)
    lg_P_var_all = comm.gather(lg_P_var, root=0)

    if rank == 0:
        # Process or return the gathered predictions as needed
        # lg_P_mean_all and lg_P_var_all are lists of results from each process
        return lg_P_mean_all, lg_P_var_all

    return None, None  # Non-root processes return None

def predict_z(L1HF_base,L2HF_base,z, X_target, n_optimization_restarts,num_processes):
    L1HF_dir = L1HF_base + '_z%s' % z
    L2HF_dir = L2HF_base + '_z%s' % z
    print('training the emulator based on the data of z =', z)
    lg_P_mean, lg_P_var = predict_mpi(L1HF_dir, L2HF_dir, X_target, n_optimization_restarts = n_optimization_restarts,num_processes = num_processes)
    print('predicted the matter power spectrum at z =', z)
        # Combine arrays vertically to create a 2D array where each array is a column
    header_str_mean = 'mean(lg_P) also median/mode'
        # Save the combined array to a text file, with each array as a column
    np.savetxt('matter_pow_lg_mean_z%s.txt' % (z), lg_P_mean, fmt='%f', header=header_str_mean)
    header_str_var = 'var(lg_P)'
        # Save the combined array to a text file, with each array as a column
    np.savetxt('matter_pow_lg_var_z%s.txt' % (z), lg_P_var, fmt='%f', header=header_str_var)
    header_str_mode = 'mode(P)'
    P_mode = 10**lg_P_mean * np.exp(-lg_P_var * (np.log(10)) **2)
        # Save the combined array to a text file, with each array as a column
    np.savetxt('matter_pow_mode_z%s.txt' % (z), P_mode, fmt='%f', header=header_str_mode)

L1HF_base = '/work2/01317/yyang440/frontera/cosmo_11p_sims/data_for_emu/matter_power_1120_Box1000_Part750_8_Box1000_Part3000' 
L2HF_base = '/work2/01317/yyang440/frontera/cosmo_11p_sims/data_for_emu/matter_power_1120_Box250_Part750_8_Box1000_Part3000' 
num_processes = 1 # multiprocessing cores per MPI rank
n_optimization_restarts = 20

zs = ['0', '0.2', '0.5', '1', '2', '3']
data_in = np.loadtxt('input.txt')
if len(data_in.shape)==1:
    data_in = data_in[None,:]

if __name__ == "__main__":

    predict_z(L1HF_base,L2HF_base, z[0], data_in, n_optimization_restarts = n_optimization_restarts,num_processes = num_processes)
