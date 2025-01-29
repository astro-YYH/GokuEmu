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
import argparse
import shutil

def save_gp(L1HF_dir, L2HF_dir, save_path, n_optimization_restarts=20, num_processes=10):
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
    dgmgp.save(save_path)

def save_z(L1HF_base,L2HF_base,z, save_path, n_optimization_restarts,num_processes):
    L1HF_dir = L1HF_base + '_z%s' % z
    L2HF_dir = L2HF_base + '_z%s' % z
    print('training the emulator based on the data of z =', z)
    save_gp(L1HF_dir, L2HF_dir, save_path, n_optimization_restarts = n_optimization_restarts,num_processes = num_processes)

L1HF_base = '../data/combined/matter_power_1128_Box1000_Part750_36_Box1000_Part3000' 
L2HF_base = '../data/combined/matter_power_1128_Box250_Part750_36_Box1000_Part3000' 
num_processes = 10 # multiprocessing cores per MPI rank
n_optimization_restarts = 25 # larger safer possibly

zs = np.array([0, 0.2, 0.5, 1, 2, 3])
# float zs and scale factors
zs_str = ['0', '0.2', '0.5', '1', '2', '3']
a_times = 1/(1+zs)

# mpi does not work on my macbook

if __name__ == "__main__":
    n_tasks = len(zs)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #-------------- Cmd line Args ------------------------------
    parser = argparse.ArgumentParser(description=' ')
    # L1HF_base
    parser.add_argument('--L1HF_base',required=False,default=L1HF_base,type=str,help='directory for L1HF_base')
    # L2HF_base
    parser.add_argument('--L2HF_base',required=False,default=L2HF_base,type=str,help='directory for L2HF_base')
    parser.add_argument('--outdir',required=False,default='pre-trained',type=str,help='directory for saving the models')

    args = parser.parse_args()
    
    #--------------------------
    outdir = args.outdir

    if rank == 0:
        if not os.path.exists(outdir):
            os.mkdir(outdir)
    
    comm.Barrier()

    # Distribute the tasks evenly across ranks
    tasks_for_this_rank = [i for i in range(rank, n_tasks, size)]

    # Each rank executes its assigned tasks
    for task_id in tasks_for_this_rank:
        # model directory for each redshift: outdir/a_times[task_id]
        model_dir = os.path.join(outdir, f'a{a_times[task_id]:.4f}')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        save_z(L1HF_base,L2HF_base,zs_str[task_id],model_dir, n_optimization_restarts,num_processes)  # save models for each redshift

    # Ensure all ranks finish their tasks before ending
    comm.Barrier()

