"""
Methods to prepare training data for the data
of the graphical multi-fidelity graphical GP.
"""

import os

import numpy as np
from scipy.interpolate import interp1d


def interpolate(log10_k: np.ndarray, Y: np.ndarray, log10_ks: np.ndarray) -> np.ndarray:
    """
    interpolate the log P(k) based on a given log10 ks

    Parameters:
    ---
    log10_k: k bins of the input function.
    Y: output of the input function.
    log10_ks: the interpolant we want to interpolate the input function on.
    """

    # select interpolatable bins at LR
    ind = (log10_ks <= log10_k.max()) * (log10_ks >= log10_k.min())

    # currently the truncation is done at outer loop,
    # the ind here is just for checking length for interpolant.
    # log10_ks = log10_ks[ind]

    new_kbins = np.sum(ind)
    assert new_kbins > 0
    print(new_kbins, log10_k.shape[0], Y.shape[-1])
    assert new_kbins == log10_ks.shape[0]

    # original LR powerspectrum shape
    n_parms, kbins = Y.shape

    # initialize new Y array for interpolated LR powerspecs
    Y_new = np.full((n_parms, new_kbins), np.nan)

    # loop over each power spec: 1) each LH set; 2) each redshift bin
    for i in range(n_parms):
        f = interp1d(log10_k, Y[i, :])

        Y_new[i, :] = f(log10_ks)

        # remove the interpolant
        del f

    print(
        "[Info] rebin powerspecs from {} k bins to {} k bins.".format(kbins, new_kbins)
    )

    return Y_new


def interp_lf_to_hf_bins(
    folder_lf: str,
    folder_hf: str,
    folder_test_hf: str,
    output_folder: str = "50_dmonly64_3_fullphysics512_mpgadget",
):
    """
    Prepare training data, for power spectra with different k bins between high- and low-fidelity

    we need to
    1) maximum k of LF <- maximum k of HF
    2) minimum k of HF <- minimum k of LF

    The training k range would be confined by low-fidelity's (kmin, kmax).
    """
    # interpolant should be from highres
    log10_ks = np.loadtxt(os.path.join(folder_hf, "kf.txt"))
    Y_hf = np.loadtxt(os.path.join(folder_hf, "output.txt"))
    input_hf = np.loadtxt(os.path.join(folder_hf, "input.txt"))

    log10_ks_test = np.loadtxt(os.path.join(folder_test_hf, "kf.txt"))
    Y_hf_test = np.loadtxt(os.path.join(folder_test_hf, "output.txt"))
    input_hf_test = np.loadtxt(os.path.join(folder_test_hf, "input.txt"))

    assert np.all(log10_ks == log10_ks_test)

    # low-fidelity is in a set of k bins, but we want them
    # to be the same as the high-fidelity k bins
    log10_k = np.loadtxt(os.path.join(folder_lf, "kf.txt"))
    Y_lf = np.loadtxt(os.path.join(folder_lf, "output.txt"))
    input_lf = np.loadtxt(os.path.join(folder_lf, "input.txt"))
    # we want to trim the high-fidelity k to have the same
    # minimum k.
    # highres: log10_ks; lowres: log10_k
    ind_min = (log10_ks >= log10_k.min()) & (log10_ks <= log10_k.max())

    # interpolate: interp(log10_k, Y_lf)(log10_k[ind_min])
    Y_lf_new = interpolate(log10_k, Y_lf, log10_ks[ind_min])

    assert Y_lf_new.shape[1] == log10_ks[ind_min].shape[0]

    # create a folder containing ready-to-use emulator train files
    base_dir = os.path.join("data", output_folder)
    os.makedirs(base_dir, exist_ok=True)

    np.savetxt(os.path.join(base_dir, "train_output_fidelity_0.txt"), Y_lf_new)
    np.savetxt(os.path.join(base_dir, "train_output_fidelity_1.txt"), Y_hf[:, ind_min])
    np.savetxt(os.path.join(base_dir, "test_output.txt"), Y_hf_test[:, ind_min])

    np.savetxt(os.path.join(base_dir, "train_input_fidelity_0.txt"), input_lf)
    np.savetxt(os.path.join(base_dir, "train_input_fidelity_1.txt"), input_hf)
    np.savetxt(os.path.join(base_dir, "test_input.txt"), input_hf_test)

    np.savetxt(os.path.join(base_dir, "kf.txt"), log10_ks[ind_min])

    input_limits = np.loadtxt(os.path.join(folder_lf, "input_limits.txt"))

    np.savetxt(os.path.join(base_dir, "input_limits.txt"), input_limits)
