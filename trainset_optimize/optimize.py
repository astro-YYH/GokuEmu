"""
A class to handle optimizing the choice of high-res simulations
using low-res only emulators.
"""

import numpy as np
import GPy
import contextlib
import io
import multiprocessing
from functools import partial
import datetime

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, y_pred_variance=None) -> float:
    """
    Mean squared error

        MSE = 1/N \Sum (y_true - y_pred)^2
    """
    # print(y_true - y_pred)
    return np.mean((y_true - y_pred)**2)


class TrainSetOptimize:
    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = X
        self.Y = Y

    def loss(self, ind: np.ndarray, loss_fn=None, n_optimization_restarts:int = 5):
        """
        Train a GP conditioned on (X[ind], Y[ind]),
        return the loss average over (X[~ind], Y[~ind])

        Parameters:
        ----
        ind: boolean array, indicator function for the training data.
        loss_fn: the loss function we used. If not specified, mean squared errors.
        """
        assert ind.dtype == bool

        # train a GP across all k bins
        _, nparams = self.X.shape

        # here I hard coded the kernel
        kernel = GPy.kern.RBF(nparams, ARD=True)
        gp = GPy.models.GPRegression(self.X[ind], self.Y[ind], kernel)

        # gp.optimize_restarts(n_optimization_restarts, parallel=False)

        try:
            gp.optimize_restarts(n_optimization_restarts, parallel=False)
        except np.linalg.LinAlgError as e:
            print(f"[Warning] GP optimization failed: {e}")
            return np.inf  # or some high penalty value

        # predicting on the rest of X
        mean, variance = gp.predict(self.X[~ind])
        # print("mean.shape: ", mean.shape)

        if not loss_fn:
            loss_fn = mean_squared_error
        print("calculating loss")
        loss = loss_fn(self.Y[~ind], mean, variance)

        return loss

    def optimize(self, prev_ind: np.ndarray, loss_fn=None, n_optimization_restarts:int = 5) -> None:
        """
        Find the optimal index in the X space for the next run,
        via optimizing the acquisition function (or loss function).

        Parameters:
        ----
        selected_ind: the index

        Return:
        ----
        (optimal index, loss values)
        """
        assert prev_ind.dtype == bool

        all_loss = []

        n_samples, _ = self.X.shape

        rest_index = np.arange(n_samples)[~prev_ind]

        for i in rest_index:
            ind = np.zeros(n_samples, dtype=bool)

            # set previous index and the additional index
            ind[prev_ind] = True
            ind[i] = True

            assert np.sum(ind) == (np.sum(prev_ind) + 1)

            loss = self.loss(ind, loss_fn=loss_fn, n_optimization_restarts=n_optimization_restarts)

            all_loss.append(loss)

        # get the index for minimizing loss
        I = np.argmin(all_loss)

        return (rest_index[I], all_loss)
    
def select_slices(X: np.ndarray, Y: np.ndarray, len_slice: int = 3, n_select_slc: int = 1, beams: int = 1, n_optimization_restarts: int = 5):
    """Select the slices of the training set for the low-fidelity simulation.
    
    Args:
        X: the input of the training set
        Y: the output (power spectra) of the training set
        len_slice: the length of the slice (3 if 3 pints in a single slice)
        n_select_slc: the number of slices to select"""
    # The number of slices
    num_samples = X.shape[0]
    num_slices = int(num_samples / len_slice)
    assert num_slices * len_slice == num_samples

    all_slices = np.arange(len_slice * num_slices).reshape(num_slices, len_slice)



    train_opt = TrainSetOptimize(X=X, Y=Y)
    

    beam_loss = []
    beam_ind = []

    all_slice_loss = []
    for i, selected_ind in enumerate(all_slices):
        # need to convert to boolean array
        ind = np.zeros(num_samples, dtype=bool)
        ind[np.array(selected_ind)] = True

        loss = train_opt.loss(ind, n_optimization_restarts=n_optimization_restarts)
    
        print("iteration:", i)

        all_slice_loss.append(loss)

    if beams == -1:
        beams = num_slices - n_select_slc + 1
    # find the set of indices best minimize the loss
    ind_small_loss = np.argsort(all_slice_loss)[:beams]

    for ind_slc in ind_small_loss:  # e.g., 4 in [4,11,9]
        selected_ind = np.array(all_slices[ind_slc])  # e.g., [4,5,6]
        ind_selected_slc = np.array([ind_slc])  # e.g., [4]
        if n_select_slc == 1:
            beam_ind.append(selected_ind)
            beam_loss.append(all_slice_loss[ind_slc])
        else:
            for i in range(n_select_slc - 1):
                combine_loss = []
                ind_loss = []
                for j in range(num_slices):
                    if j in ind_selected_slc:
                        continue
                    combine_ind = np.append(selected_ind, [all_slices[j]])
                    print("combine_ind:", combine_ind)
                    ind = np.zeros(num_samples, dtype=bool)
                    ind[np.array(combine_ind)] = True
                    # print("loss uses ind:", ind)
                    loss = train_opt.loss(ind, n_optimization_restarts=n_optimization_restarts)
    
                    print("Combine slice {} with {}, loss = {}".format(j, ind_selected_slc, loss))
                    combine_loss.append(loss)
                    ind_loss.append(j)
                i_min_loss = np.argmin(combine_loss)
                ind_selected_slc = np.append(ind_selected_slc,np.array([ind_loss[i_min_loss]]))
                selected_ind = np.array(all_slices[ind_selected_slc])
            beam_ind.append(selected_ind)
            beam_loss.append(np.min(combine_loss))
            assert np.min(combine_loss) == combine_loss[i_min_loss]
    print("Beam slices of points:", beam_ind)
    print("All beam losses:", beam_loss)
    min_loss = np.min(beam_loss)
    optimal_ind = beam_ind[np.argmin(beam_loss)]

    return optimal_ind, min_loss

def loss_redshifts(train_opt_zs, ind: np.ndarray, n_optimization_restarts: int = 5, parallel: bool = True) -> float:
    if parallel:
        return loss_redshifts_parallel(train_opt_zs, ind, n_optimization_restarts)
    else:
        return loss_redshifts_sequential(train_opt_zs, ind, n_optimization_restarts)

def loss_redshifts_sequential(train_opt_zs, ind: np.ndarray, n_optimization_restarts: int = 5) -> float:
    losses = []
    for train_opt in train_opt_zs:
        # print(ind, n_optimization_restarts)
        with contextlib.redirect_stdout(io.StringIO()):
            loss = train_opt.loss(ind, n_optimization_restarts=n_optimization_restarts)
            losses.append(loss)
    loss = np.mean(losses)
    return loss

def loss_redshift(train_opt, ind, n_optimization_restarts):
    """Function to compute loss for a single train_opt instance."""
    # print(ind, n_optimization_restarts)
    with contextlib.redirect_stdout(io.StringIO()):
        return train_opt.loss(ind, n_optimization_restarts=n_optimization_restarts)

def loss_redshifts_parallel(train_opt_zs, ind: np.ndarray, n_optimization_restarts: int = 5) -> float:
    num_processes = len(train_opt_zs)  # Number of processes to use

    # Create a partial function with fixed arguments
    partial_loss_redshift = partial(loss_redshift, ind=ind, n_optimization_restarts=n_optimization_restarts)

    with multiprocessing.Pool(processes=num_processes) as pool:
        losses = pool.map(partial_loss_redshift, train_opt_zs)
    loss = np.mean(losses)
    return loss

def loss_scales_redshifts(train_opt_scales_zs, ind: np.ndarray, n_optimization_restarts:int = 5) -> float:
    losses = []
    for train_opt_zs in train_opt_scales_zs:
        for train_opt in train_opt_zs:
            with contextlib.redirect_stdout(io.StringIO()):
                loss = train_opt.loss(ind, n_optimization_restarts=n_optimization_restarts)
                losses.append(loss)
    loss = np.mean(losses)
    return loss


def select_slices_redshifts(X: np.ndarray, Y: np.ndarray, len_slice: int = 3, n_select_slc: int = 1, beams: int = 1, n_optimization_restarts: int = 5, print_all: bool=False, parallel_redshift: bool=True, selected_slices: list = None):
    """Select the slices of the training set for the low-fidelity simulation.
    
    Args:
        X: the input of the training set
        Y: the output (power spectra) of the training set
        len_slice: the length of the slice (3 if 3 pints in a single slice)
        n_select_slc: the number of slices to select"""
    # The number of slices
    num_samples = X.shape[0]
    num_slices = int(num_samples / len_slice)
    assert num_slices * len_slice == num_samples

    all_slices = np.arange(len_slice * num_slices).reshape(num_slices, len_slice)

    train_opt_zs = []
    for y in Y:
        train_opt_zs.append(TrainSetOptimize(X=X, Y=y))

    beam_loss = []
    beam_ind = []

    all_slice_loss = []

    if selected_slices is None:
        selected_slices = []

    # beam search layer 1
    for i, selected_ind in enumerate(all_slices):

        # if selected_slices is not None:
        if len(selected_slices) > 0:
            if i in selected_slices:
                print("Skipping slice", i, "as it is already selected.")
                all_slice_loss.append(np.inf)
                continue

        test_slices = selected_slices + [i]  # e.g., [4,5,6] points

        # combine selected slices with the current slice
        selected_ind = all_slices[test_slices].flatten()  # e.g., [4,5,6] point indices

        print("Testing slices:", test_slices, "i.e., points", selected_ind)
        # need to convert to boolean array
        ind = np.zeros(num_samples, dtype=bool)
        ind[np.array(selected_ind)] = True

        print("Computing loss function for slice", i)
        loss = loss_redshifts(train_opt_zs, ind, n_optimization_restarts=n_optimization_restarts, parallel=parallel_redshift)
        # print time now
        print("Time now:", datetime.datetime.now())
        print("Loss function for slice", i, "(combined with", selected_slices, ") =", loss)
        all_slice_loss.append(loss)
        
    if beams == -1:
        beams = num_slices - n_select_slc + 1  # maximum
    # find the set of indices best minimize the loss
    ind_small_loss = np.argsort(all_slice_loss)[:beams]

    for k, ind_slc in enumerate(ind_small_loss):  # e.g., 4 in [4,11,9]
        print("\nBeam search: chain %d/%d" % (k+1, beams), "\n")

        
        ind_selected_slc = np.array(selected_slices+[ind_slc])  # e.g., [4]
        selected_ind = all_slices[ind_selected_slc].flatten()  # e.g., [4,5,6] point indices

        if n_select_slc == 1:
            beam_ind.append(selected_ind)
            beam_loss.append(all_slice_loss[ind_slc])
        else:
            for i in range(n_select_slc - 1):
                combine_loss = []
                ind_loss = []
                for j in range(num_slices):
                    if j in ind_selected_slc:
                        continue
                    if i == 0:    # avoid repeating combinations
                        if j in ind_small_loss[:k]:
                            continue
                    combine_ind = np.append(selected_ind, [all_slices[j]])
                    print("combine_ind:", combine_ind)
                    ind = np.zeros(num_samples, dtype=bool)
                    ind[np.array(combine_ind)] = True
                    # print("loss uses ind:", ind)
                    loss = loss_redshifts(train_opt_zs, ind, n_optimization_restarts=n_optimization_restarts, parallel=parallel_redshift)

                    print("Combine slice {} with {}, loss = {}".format(j, ind_selected_slc, loss))
                    combine_loss.append(loss)
                    ind_loss.append(j)
                i_min_loss = np.argmin(combine_loss)
                ind_selected_slc = np.append(ind_selected_slc,np.array([ind_loss[i_min_loss]]))
                selected_ind = all_slices[ind_selected_slc]
            beam_ind.append(selected_ind)
            beam_loss.append(np.min(combine_loss))
            assert np.min(combine_loss) == combine_loss[i_min_loss]
    beam_loss = np.array(beam_loss)
    beam_ind = np.array(beam_ind)
    
    print("Beam slices of points:", beam_ind)
    print("All beam losses:", beam_loss)
    min_loss = np.min(beam_loss)
    optimal_ind = beam_ind[np.argmin(beam_loss)]

    if print_all:
        indices = np.argsort(beam_loss)
        beam_loss_sorted = beam_loss[indices]
        beam_ind_sorted = beam_ind[indices]
        return beam_ind_sorted, beam_loss_sorted
        
    return optimal_ind, min_loss

def select_slices_scales_redshifts(Xs: list, Ys: list, len_slice: int = 3, n_select_slc: int = 1, beams: int = 1, n_optimization_restarts: int = 5):
    """Select the slices of the training set for the given low-fidelity simulations L1 and L2.
    
    Args:
        X: the input of the training set
        Y: the output (power spectra) of the training set
        len_slice: the length of the slice (3 if 3 pints in a single slice)
        n_select_slc: the number of slices to select"""
    # The number of slices
    num_samples = Xs[0].shape[0]
    num_slices = int(num_samples / len_slice)
    assert num_slices * len_slice == num_samples

    all_slices = np.arange(len_slice * num_slices).reshape(num_slices, len_slice)

    train_opt_scales_zs = []
    for i, X in enumerate(Xs):
        train_opt_scales_zs.append([])
        for Y in Ys[i]:
            train_opt_scales_zs[i].append(TrainSetOptimize(X=X, Y=Y))

    beam_loss = []
    beam_ind = []

    all_slice_loss = []

    # beam search layer 1
    for i, selected_ind in enumerate(all_slices):
        # need to convert to boolean array
        ind = np.zeros(num_samples, dtype=bool)
        ind[np.array(selected_ind)] = True

        print("Computing loss function for slice", i)
        loss = loss_scales_redshifts(train_opt_scales_zs, ind, n_optimization_restarts=n_optimization_restarts)
        print("Loss function for slice", i, "=", loss)
        all_slice_loss.append(loss)
        
    if beams == -1:
        beams = num_slices - n_select_slc + 1  # maximum
    # find the set of indices best minimize the loss
    ind_small_loss = np.argsort(all_slice_loss)[:beams]

    for k, ind_slc in enumerate(ind_small_loss):  # e.g., 4 in [4,11,9]
        print("Beam search: chain %d/%d" % (k+1, beams), "\n")
        selected_ind = np.array(all_slices[ind_slc])  # e.g., [4,5,6] point indices
        ind_selected_slc = np.array([ind_slc])  # e.g., [4]
        if n_select_slc == 1:
            beam_ind.append(selected_ind)
            beam_loss.append(all_slice_loss[ind_slc])
        else:
            for i in range(n_select_slc - 1):
                combine_loss = []
                ind_loss = []
                for j in range(num_slices):
                    if j in ind_selected_slc:
                        continue
                    combine_ind = np.append(selected_ind, [all_slices[j]])
                    print("combine_ind:", combine_ind)
                    ind = np.zeros(num_samples, dtype=bool)
                    ind[np.array(combine_ind)] = True
                    # print("loss uses ind:", ind)
                    loss = loss_scales_redshifts(train_opt_scales_zs, ind, n_optimization_restarts=n_optimization_restarts)

                    print("Combine slice {} with {}, loss = {}".format(j, ind_selected_slc, loss))
                    combine_loss.append(loss)
                    ind_loss.append(j)
                i_min_loss = np.argmin(combine_loss)
                ind_selected_slc = np.append(ind_selected_slc,np.array([ind_loss[i_min_loss]]))
                selected_ind = np.array(all_slices[ind_selected_slc])
            beam_ind.append(selected_ind)
            beam_loss.append(np.min(combine_loss))
            assert np.min(combine_loss) == combine_loss[i_min_loss]
    print("Beam slices of points:", beam_ind)
    print("All beam losses:", beam_loss)
    min_loss = np.min(beam_loss)
    optimal_ind = beam_ind[np.argmin(beam_loss)]

    return optimal_ind, min_loss
            
