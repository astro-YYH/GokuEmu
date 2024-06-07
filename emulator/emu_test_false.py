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

def predict(L1HF_dir, L2HF_dir, X_target, n_optimization_restarts=20):
    data_1, data_2 = generate_data(folder_1=L1HF_dir, folder_2=L2HF_dir)

    # highres: log10_ks; lowres: log10_k
    # with contextlib.redirect_stdout(io.StringIO()):  # Redirect stdout to a null stream
    dgmgp = SingleBindGMGP(
            X_train=[data_1.X_train_norm[0][:100], data_2.X_train_norm[0][:100], data_1.X_train_norm[1]],  # L1, L2, HF
            Y_train=[data_1.Y_train_norm[0][:100], data_2.Y_train_norm[0][:100], data_1.Y_train_norm[1]],
            n_fidelities=2,
            n_samples=400,
            optimization_restarts=n_optimization_restarts,
            ARD_last_fidelity=False,
            parallel=False,
            # num_processes=None
            )
    X_target_norm = input_normalize(X_target, data_1.parameter_limits)
    lg_P_mean, lg_P_var = dgmgp.predict(X_target_norm)
    return lg_P_mean, lg_P_var

def predict_z(L1HF_base,L2HF_base,z, X_target):
    L1HF_dir = L1HF_base + '_z%s' % z
    L2HF_dir = L2HF_base + '_z%s' % z
    print('training the emulator based on the data of z =', z)
    lg_P_mean, lg_P_var = predict(L1HF_dir, L2HF_dir, X_target)
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

L1HF_base = '../data/matter_power_270_Box100_Part75_18_Box100_Part300' 
L2HF_base = '../data/matter_power_270_Box25_Part75_18_Box100_Part300' 


zs = ['0', '0.2', '0.5', '1', '2', '3']
data_in = np.loadtxt('input.txt')
if len(data_in.shape)==1:
    data_in = data_in[None,:]

if __name__ == "__main__":
    n_tasks = len(zs)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Calculate the number of tasks per rank and the remaining tasks
    # tasks_per_rank = n_tasks // size
    # remaining_tasks = n_tasks % size

    # Distribute the tasks evenly across ranks
    tasks_for_this_rank = [i for i in range(rank, n_tasks, size)]

    # Each rank executes its assigned tasks
    for task_id in tasks_for_this_rank:
        predict_z(L1HF_base,L2HF_base,zs[task_id], data_in)

    # Ensure all ranks finish their tasks before ending
    comm.Barrier()