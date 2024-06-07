import sys
sys.path.append("../../matter_emu_dgmgp-MF-Box-yanhui/")

import os
import numpy as np
from matter_multi_fidelity_emu.gpemulator_singlebin import SingleBindGMGP 
from matter_multi_fidelity_emu.data_loader import PowerSpecs
from matter_multi_fidelity_emu.data_loader_dgmgp import interpolate

import glob
import random
import contextlib
import io
import argparse
import time

# set a random number seed to reproducibility
np.random.seed(0)

# This is only redshift zero
def generate_data(
    folder_1: str = "../data/processed/Matterpower_60_res128box256_3_res512box256_z1_ind-0-1-2/",
    folder_2: str = "../data/processed/Matterpower_60_res128box100_3_res512box256_z1_ind-0-1-2/",
    n_fidelities: int = 2,
):
    data_1 = PowerSpecs(
        n_fidelities=n_fidelities,
    )
    data_1.read_from_txt(folder=folder_1)

    data_2 = PowerSpecs(
        n_fidelities=n_fidelities,
    )
    data_2.read_from_txt(folder=folder_2)

    return data_1, data_2


def validate_dgmgp(data: PowerSpecs, model: SingleBindGMGP):
    """
    Validate the trained MFEmulators
    """
    x_test, y_test = data.X_test_norm[0], data.Y_test[0]

    mean, var = model.predict(x_test)

    # use mode (maximumlikelihood) as the prediction, restore the spectrum from log10(P(k)) which distributes normally
    mode_P = 10**mean * np.exp(-var * (np.log(10)) ** 2)
    var_P = (
        10 ** (2 * mean)
        * np.exp(var * (np.log(10)) ** 2)
        * (np.exp(var * (np.log(10)) ** 2) - 1)
    )

    # If you'd like to use mean instead of mode, the variance is here:
    # ---
    # # dx = x dlog(x) = log(10) x dlog10(x)
    # # dlog10(x) = d(log(x) / log(10))
    # vars = (10**mean * np.log(10) * np.sqrt(var)) ** 2

    # predicted/exact
    pred_exacts = mode_P / 10**y_test

    return mode_P, var_P, pred_exacts


def get_error(L1HF_dir, L2HF_dir, LF_selected_ind, HF_selected_ind, n_optimization_restarts: int = 5, n_fidelities: int = 2, parallel: bool = True):
    data_1, data_2 = generate_data(folder_1=L1HF_dir, folder_2=L2HF_dir)

    # interpolation
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

    print("Low-fidelity points:", LF_selected_ind)
    print("High-fidelity points:", HF_selected_ind, "\n")
    print("n_optimization_restarts:", n_optimization_restarts, "\n")
    with contextlib.redirect_stdout(io.StringIO()):  # Redirect stdout to a null stream
        dgmgp = SingleBindGMGP(
            X_train=[data_1.X_train_norm[0][LF_selected_ind], data_2.X_train_norm[0][LF_selected_ind], data_1.X_train_norm[1][HF_selected_ind]],
            Y_train=[data_1.Y_train_norm[0][LF_selected_ind], data_2.Y_train_norm[0][LF_selected_ind], data_1.Y_train_norm[1][HF_selected_ind]],
            n_fidelities=n_fidelities,
            n_samples=400,
            optimization_restarts=n_optimization_restarts,
            ARD_last_fidelity=False,
            parallel=parallel,
            )

        _, _, pred_exacts_dgmgp = validate_dgmgp(data_1, model=dgmgp)
        print("pred_exacts_dgmgp:", pred_exacts_dgmgp, "\n")
        error_selected = np.mean(np.absolute(pred_exacts_dgmgp-1), axis=1).mean()
        print(LF_selected_ind, HF_selected_ind, "error:", error_selected, "\n")

    return error_selected
    
def search_LF(data, num_LF, HF_selected_ind, len_slice):

    X_0 = data.X_train[0] # LF cosmologies
    X_1 = data.X_train[1] # HF cosmologies

    n_LF_slc = num_LF//len_slice
    n_HF_slc = len(HF_selected_ind)//len_slice

    LF_selected_ind = []

    for i in range(n_HF_slc):
        if i == n_LF_slc:
            break
        first_ind = HF_selected_ind[i*len_slice]
        ind_LF_HF = np.where(X_1[first_ind] == X_0)[0][0]
        for j in range(len_slice):
            LF_selected_ind.append(ind_LF_HF + j)
    return LF_selected_ind

def select_ind(data, num_LF, num_HF, len_slice=3):

    n_sample_LF = data.X_train[0].shape[0]
    n_sample_HF = data.X_train[1].shape[0]

    assert num_LF <= n_sample_LF
    assert num_HF <= n_sample_HF

    assert n_sample_LF%len_slice == 0
    assert n_sample_HF%len_slice == 0
    assert num_LF%len_slice == 0
    assert num_HF%len_slice == 0

    ind_slc_list = random.sample(list(range(int(n_sample_HF/len_slice))), k=int(num_HF/len_slice))
        # print("n_sample_HF/len_slice:", int(n_sample_HF/len_slice))
        # print("num_HF/len_slice:", int(num_HF/len_slice))
    HF_selected_ind = []
    for ind_slc in ind_slc_list:
        for i in range(len_slice):
            HF_ind = len_slice*ind_slc + i
            HF_selected_ind.append(HF_ind)

    LF_selected_ind = search_LF(data, num_LF, HF_selected_ind, len_slice)
    print("LF_HF_common_ind:", LF_selected_ind, "\n")

    if num_LF > num_HF:
        ind_slc_list = random.sample(list(range(int(n_sample_LF/len_slice))), k=int((num_LF - num_HF)/len_slice))
        for ind_slc in ind_slc_list:
            for i in range(len_slice):
                LF_ind = len_slice*ind_slc + i
                LF_selected_ind.append(LF_ind)

    return LF_selected_ind, HF_selected_ind

def error_select(L1HF_base, L2HF_base, num_LF: int = 6, num_HF: int = 3, n_optimization_restarts: int = 5, len_slice: int = 3, parallel: bool = True) -> float:
    errors_z = []

    pattern = L1HF_base + "*"
    L1HF_dirs = [folder for folder in glob.glob(pattern) if os.path.isdir(folder)]
    L1HF_dirs = sorted(L1HF_dirs)

    print("L1HF directories:", L1HF_dirs, "\n")

    data_0 = PowerSpecs(
        n_fidelities=2,
    )
    data_0.read_from_txt(folder=L1HF_dirs[0])
    LF_selected_ind, HF_selected_ind = select_ind(data_0, num_LF, num_HF, len_slice=len_slice)

    for L1HF_dir in L1HF_dirs:
        suffix = L1HF_dir[len(L1HF_base):]
        L2HF_dir = L2HF_base + suffix
        print("Computing errors for", suffix[1:], "\n")
        error_z = get_error(L1HF_dir, L2HF_dir, LF_selected_ind, HF_selected_ind, n_optimization_restarts=n_optimization_restarts, parallel=parallel)  # error_z = [#, #, #]
        print("Mean error for", suffix[1:], ":", error_z, "\n")
        errors_z.append(error_z)

    errors_z = np.array(errors_z)
    error = np.mean(errors_z, axis=0)
    return error 

def write_error(file_path, num_LF, num_HF, error, n_optimization_restarts, elapsed_time):
    with open(file_path, 'a+') as file:
        file.seek(0)
        content = file.read()
        if len(content) == 0:
            file.write("# 1. n_LF  2. n_HF  3. error  4. n_optimization_restarts 5. time spent (min)\n")

        formatted_info = "%d  %d  %e  %d  %.2f\n" % (num_LF, num_HF, error, n_optimization_restarts, elapsed_time)
        file.write(formatted_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str,
        default="./")
    parser.add_argument("--L1HF_base", type=str,
        default="Matterpower_60_res128box256_3_res512box256")
    parser.add_argument("--L2HF_base", type=str, default="Matterpower_60_res128box100_3_res512box256")
    
    parser.add_argument("--num_LF", type=int, default=6)
    parser.add_argument("--num_HF", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--n_optimization_restarts", type=int, default=5)
    parser.add_argument("--output_file", type=str, default="error_function.txt")

    args = parser.parse_args()

    start_time = time.time()

    print("parallel:", args.parallel, type(args.parallel))

    data_dir = args.data_dir
    L1HF_base = os.path.join(data_dir,args.L1HF_base)  
    L2HF_base = os.path.join(data_dir,args.L2HF_base) 
    error = error_select(L1HF_base, L2HF_base, num_LF=args.num_LF, num_HF=args.num_HF, n_optimization_restarts=args.n_optimization_restarts, parallel=args.parallel)

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60

    print("error = ", error)

    write_error(args.output_file, args.num_LF, args.num_HF, error, args.n_optimization_restarts, elapsed_time)

# python dgmgp_error.py --data_dir=/work2/01317/yyang440/frontera/matter_emu_dgmgp/data/processed --L1HF_base=Matterpower_60_res128box256_3_res512box256 --L2HF_base=Matterpower_60_res128box100_3_res512box256 --num_LF=9 --num_HF=3 --n_optimization_restarts=20