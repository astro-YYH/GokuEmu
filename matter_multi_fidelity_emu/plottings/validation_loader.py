"""
A Plotting class to plot the MF matter power class
"""

from typing import List, Tuple, Optional, Union

import os
import numpy as np
from matplotlib import pyplot as plt

class ValidationLoader:
    """
    A custom loader to load the outputs from examples.make_validations.validate
    This will be useful when we are not re-training the MF emulators for batch runs.

    :param outdirs: a list of dirs we want to load. In principle it should look like
        this: "val_ns_{}_ind_{}".format(
            "-".join(map(str, num_samples)), "-".join(map(str, selected_ind)))
    """

    def __init__(
        self, outdirs: List[str], num_lowres_list: List[int], num_highres: int, dGMGP: bool = False,
    ):
        self.outdirs = outdirs

        # placeholders for some property parameters for later on testings
        self.num_lowres_list = num_lowres_list
        self.num_highres = num_highres
        assert len(self.num_lowres_list) == len(self.outdirs)

        # load all arrays files in the outdirs into attributes
        self.load_data(dGMGP=dGMGP)
        self.compute_errors()

    def load_data(self, dGMGP: bool = False) -> None:
        """
        Load the data arrays in the validation folder into
        """
        # load GP predictions
        self._all_gp_mean = self.load_array(self.outdirs, "all_gp_mean")
        self._all_gp_std = self.load_array(
            self.outdirs, "all_gp_var"
        )  # this is a typo I forgot to fix in make_validations
        assert self._all_gp_mean.shape[0] == len(self.num_lowres_list)

        # load predict/exact
        self._pred_exacts =  self.load_array(self.outdirs, "pred_exacts")
        self._pred_exacts_hf =  self.load_array(self.outdirs, "pred_exacts_hf")

        # load highRes GP predictions
        self._all_hf_gp_mean = self.load_array(self.outdirs, "all_hf_gp_mean")
        self._all_hf_gp_std = np.sqrt(self.load_array(
            self.outdirs, "all_hf_gp_var"
        ))
        if len(self._all_hf_gp_std.shape) != len(self._all_hf_gp_mean.shape):
            print("[Warning] only have 1 std for HF per spectrum.")
            self._all_hf_gp_std = np.repeat(self._all_hf_gp_std[:, :, None], self._all_hf_gp_mean.shape[2], axis=2)

        if not dGMGP:
            # load lowRes GP predictions
            self._all_lf_gp_mean = self.load_array(self.outdirs, "all_lf_gp_mean")
            self._all_lf_gp_std = np.sqrt(self.load_array(
                self.outdirs, "all_lf_gp_var"
            ))
            if len(self._all_lf_gp_std.shape) != len(self._all_lf_gp_mean.shape):
                print("[Warning] only have 1 std for LF per spectrum.")
                self._all_lf_gp_std = np.repeat(self._all_lf_gp_std[:, :, None], self._all_lf_gp_mean.shape[2], axis=2)


            # load predict/exact
            self._pred_exacts_lf =  self.load_array(self.outdirs, "pred_exacts_lf")

        else:
            # For dGMGP files, there are two LF predictions
            # L1 node
            self._all_lf_gp_mean = self.load_array(self.outdirs, "all_lf_gp_mean_1")
            self._all_lf_gp_std = np.sqrt(self.load_array(
                self.outdirs, "all_lf_gp_var_1"
            ))
            if len(self._all_lf_gp_std.shape) != len(self._all_lf_gp_mean.shape):
                print("[Warning] only have 1 std for LF per spectrum.")
                self._all_lf_gp_std = np.repeat(self._all_lf_gp_std[:, :, None], self._all_lf_gp_mean.shape[2], axis=2)

            # load predict/exact
            self._pred_exacts_lf =  self.load_array(self.outdirs, "pred_exacts_lf_1")


            # L2 node
            self._all_lf_gp_mean_2 = self.load_array(self.outdirs, "all_lf_gp_mean_2")
            self._all_lf_gp_std_2 = np.sqrt(self.load_array(
                self.outdirs, "all_lf_gp_var_2"
            ))
            if len(self._all_lf_gp_std_2.shape) != len(self._all_lf_gp_mean_2.shape):
                print("[Warning] only have 1 std for LF per spectrum.")
                self._all_lf_gp_std_2 = np.repeat(self._all_lf_gp_std_2[:, :, None], self._all_lf_gp_mean_2.shape[2], axis=2)

            # load predict/exact
            self._pred_exacts_lf_2 =  self.load_array(self.outdirs, "pred_exacts_lf_2")


        # load true simulations
        self._kf = self.load_array(self.outdirs, "kf")
        self._all_true = self.load_array(self.outdirs, "all_true")
        assert np.all(
            np.array(self._all_gp_mean.shape) == np.array(self._all_gp_std.shape)
        )
        assert self._kf.shape[0] == self._all_gp_mean.shape[0]
        assert self._kf.shape[1] == self._all_gp_mean.shape[2]

    def compute_errors(self):
        """
        Pre-calculate different kinds of errors:
        1) fractional errors: abs[ (pred_mean - true) / true ]
        2) standardized errors: (pred_mean - true) / pred_std

        Note: Calcaulate all values in real scales not in the log-scale.
        """
        # 1) Fractional errors
        # [MF errors] fractional errors
        self._all_res = np.abs(
            self.log2real(self._all_gp_mean) - self.log2real(self._all_true)
        ) / self.log2real(self._all_true)
        # [HF errors] fractional errors
        self._all_res_hf = np.abs(
            self.log2real(self._all_hf_gp_mean) - self.log2real(self._all_true)
        ) / self.log2real(self._all_true)

        # 2) Standardized errors
        # [MF errors]
        self._all_norm_res = (
            self.log2real(self._all_gp_mean) - self.log2real(self._all_true)
        ) / self.log2real(self._all_gp_std)
        self._all_norm_res_hf = (
            self.log2real(self._all_hf_gp_mean) - self.log2real(self._all_true)
        ) / self.log2real(self._all_hf_gp_std)

    @staticmethod
    def load_array(outdirs: List[str], filename: str) -> np.ndarray:
        """
        Load required variables for each run into an array
        """
        loaded_array = []
        for outdir in outdirs:
            loaded_array.append(np.loadtxt(os.path.join(outdir, filename)))
        return np.array(loaded_array)

    @staticmethod
    def log2real(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 10.0 ** x

    def plot_predictions(self, memu: int, nspec: int, plot_hf_residuals: bool = False):
        """
        Plot the nth spec from mth multi-fidelity emulator, with
        1) MF pred; 2) HF pred; 3) true simulation; 4) fractional errors.
        """
        # 1) MF pred, with uncertainty
        plt.loglog(
            self.log2real(self._kf[memu, :]),
            self.log2real(self._all_gp_mean[memu, nspec, :]),
            color="C0",
        )
        plt.fill_between(
            self.log2real(self._kf[memu, :]),
            self.log2real(
                (self._all_gp_mean[memu, nspec, :])
                - 1.96 * self._all_gp_std[memu, nspec, :]
            ),
            self.log2real(
                (self._all_gp_mean[memu, nspec, :])
                + 1.96 * self._all_gp_std[memu, nspec, :]
            ),
            alpha=0.3,
            color="C0",
            label="Multi-fidelity emulator",
        )

        # 2) HF pred, with uncertainty
        plt.loglog(
            self.log2real(self._kf[memu, :]),
            self.log2real(self._all_hf_gp_mean[memu, nspec, :]),
            color="C1",
        )
        plt.fill_between(
            self.log2real(self._kf[memu, :]),
            self.log2real(
                (self._all_hf_gp_mean[memu, nspec, :])
                - 1.96 * self._all_hf_gp_std[memu, nspec, :]
            ),
            self.log2real(
                (self._all_hf_gp_mean[memu, nspec, :])
                + 1.96 * self._all_hf_gp_std[memu, nspec, :]
            ),
            alpha=0.3,
            color="C1",
            label="High fidelity only emulator",
        )

        # 3) True simulation power spectrum
        plt.loglog(
            self.log2real(self._kf[memu, :]),
            self.log2real(self._all_true[memu, nspec, :]),
            label="True simulation",
            color="k",
        )

        # 4) Fractional errors
        plt.loglog(
            self.log2real(self._kf[memu, :]),
            self._all_res[memu, nspec, :],
            label=r"$Mean(pred - true)/true = {:.2g}$".format(
                self._all_res[memu, nspec, :].mean()
            ),
            color="C3",
        )
        if plot_hf_residuals:
            plt.loglog(
                self.log2real(self._kf[memu, :]),
                self._all_res_hf[memu, nspec, :],
                label=r"HF: $Mean(pred - true)/true = {:.2g}$".format(
                    self._all_res_hf[memu, nspec, :].mean()
                ),
                color="C4",
            )

        # plotting settings
        plt.legend()
        plt.xlabel("k")
        plt.ylabel("P(k)")

    @property
    def pred_div_exact(self):
        return self._pred_exacts

    @property
    def relative_errors(self):
        """
        Shape (number of emulators, number of test simulations, number of k bins)
        """
        return np.abs(self.pred_div_exact - 1)

    @property
    def all_test_errors(self):
        """
        Relative test errors averaged over k bins,
        shape (number of emulators, number of test simulations)
        """
        return self.relative_errors.mean(axis=2)

    @property
    def pred_div_exact_hf(self):
        return self._pred_exacts_hf

    @property
    def pred_div_exact_lf(self):
        return self._pred_exacts_lf

    @property
    def relative_errors_hf(self):
        """
        Shape (number of emulators, number of test simulations, number of k bins)
        """
        return np.abs(self.pred_div_exact_hf - 1)

    @property
    def relative_errors_lf(self):
        """
        Shape (number of emulators, number of test simulations, number of k bins)
        """
        return np.abs(self.pred_div_exact_lf - 1)

    @property
    def all_test_hf_errors(self):
        """
        Relative test errors averaged over k bins,
        shape (number of emulators, number of test simulations)
        """
        return self.relative_errors_lf.mean(axis=2)

    @property
    def all_test_lf_errors(self):
        """
        Relative test errors averaged over k bins,
        shape (number of emulators, number of test simulations)
        """
        return self.relative_errors_lf.mean(axis=2)


    def plot_pred_exact(
        self, memu: int, nspec: int, average: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Plot the pred/exact per k modes, similar to Figure 4 in sbird's paper.

        :param memu: mth MF emulator
        :param nspec: nth power spectrum in the highRes simulations
        :param average: if True, average the pred/exact over all available test spectra

        :return (lower, upper): 1sigma region of the emulator errors
        """
        if average:
            # you only want to take into account test set
            pred_div_exact_mean = np.mean(
                self.log2real(self._all_gp_mean[memu, :, :])
                / self.log2real(self._all_true[memu, :, :]),
                axis=0,
            )
            plt.semilogx(
                self.log2real(self._kf[memu, :]),
                pred_div_exact_mean,
                label="MF: {}-{} average".format(
                    self.num_lowres_list[memu], self.num_highres
                ),
            )
            plt.xlabel(r"$k (h/\mathrm{Mpc})$")
            plt.ylabel(r"$\mathrm{Predicted/Exact}$")

            # get the errors
            pred_div_exact_upper = np.mean(
                self.log2real(
                    self._all_gp_mean[memu,  :, :]
                    + self._all_gp_std[memu, :, :]
                )
                / self.log2real(self._all_true[memu,  :, :]),
                axis=0,
            )
            pred_div_exact_lower = np.mean(
                self.log2real(
                    self._all_gp_mean[memu, :, :]
                    - self._all_gp_std[memu, :, :]
                )
                / self.log2real(self._all_true[memu, :, :]),
                axis=0,
            )
            return (pred_div_exact_lower, pred_div_exact_upper)

        plt.semilogx(
            self.log2real(self._kf[memu, :]),
            self.log2real(self._all_gp_mean[memu, nspec, :])
            / self.log2real(self._all_true[memu, nspec, :]),
            label="MF: {}-{}; test {}".format(
                self.num_lowres_list[memu], self.num_highres, nspec
            ),
        )
        plt.xlabel(r"$k (h/\mathrm{Mpc})$")
        plt.ylabel(r"$\mathrm{Predicted/Exact}$")

        # return 1-sigma errors in real scale
        upper = self.log2real(
            self._all_gp_mean[memu, nspec, :] + self._all_gp_std[memu, nspec, :]
        ) / self.log2real(self._all_true[memu, nspec, :])
        lower = self.log2real(
            self._all_gp_mean[memu, nspec, :] - self._all_gp_std[memu, nspec, :]
        ) / self.log2real(self._all_true[memu, nspec, :])
        return (lower, upper)

    def plot_average_test_errors(self):
        """
        Plot the average test errors as a function of number of lowRes.
        """
        # as a function of lowRes
        # average over test simulations, in axis 1
        test_mean = self.all_test_errors.mean(axis=1)
        test_max = self.all_test_errors.max(axis=1)
        test_min = self.all_test_errors.min(axis=1)

        # mean HF single-fidelity test error
        test_hf_error = self.all_test_hf_errors.mean(axis=1)[0] # all emulators have the same HF test errors

        plt.plot(self.num_lowres_list, test_mean, label="Mean[(pred-true)/true]")
        plt.fill_between(
            self.num_lowres_list,
            test_min,
            test_max,
            label="(Max, Min) of the test errors",
            alpha=0.3,
        )
        plt.hlines(
            test_hf_error,
            min(self.num_lowres_list),
            max(self.num_lowres_list),
            ls="--",
            color="C1",
            label="High-fidelity only emulator with {} HR: Mean test error: {:.2g}".format(
                self.num_highres, test_hf_error
            ),
        )
        plt.xlabel("Number of LowRes Simulations")
        plt.ylabel("Mean test errors")
        plt.ylim(min(test_min), max(test_max))
        plt.legend()

    @property
    def all_gp_mean(self) -> np.ndarray:
        """
        All predicted GP mean from multi-fidelity emulator

        shape: (len(num_lowres_list), number of total highRes, k bins)
        """
        return self._all_gp_mean

    @property
    def all_gp_std(self) -> np.ndarray:
        """
        All predicted GP std from multi-fidelity emulator

        shape: (len(num_lowres_list), number of total highRes, k bins)
        """
        return self._all_gp_std

    @property
    def all_hf_gp_mean(self) -> np.ndarray:
        """
        All predicted GP mean from high-fidelity emulator

        shape: (len(num_lowres_list), number of total highRes, k bins)
        """
        return self._all_hf_gp_mean

    @property
    def all_hf_gp_std(self) -> np.ndarray:
        """
        All predicted GP std from high-fidelity emulator

        shape: (len(num_lowres_list), number of total highRes, k bins)
        """
        return self._all_hf_gp_std

    @property
    def kf(self) -> np.ndarray:
        """
        All k-mode bins. Taken from rebinned MP-Gadget matter power spectrum.

        shape: (len(num_lowres_list), k bins)
        """
        return self._kf

    @property
    def all_true(self) -> np.ndarray:
        """
        All power spectra from true highRes simulations

        shape: (len(num_lowres_list), number of total highRes, k bins)
        """
        return self._all_true


    @property
    def all_res(self) -> np.ndarray:
        """
        All fractional errors from multi-fidelity emulator,

            abs (pred - true / true)
        
        shape: (len(num_lowres_list), number of total highRes, k bins)        
        """
        return self._all_res

    @property
    def all_res_hf(self) -> np.ndarray:
        """
        All fractional errors from high-fidelity emulator,

            abs (pred - true / true)
        
        shape: (len(num_lowres_list), number of total highRes, k bins)        
        """
        return self._all_res_hf

    @property
    def all_norm_res(self) -> np.ndarray:
        """
        All standardized errors from multi-fidelity emulator,

            (pred_mean - true / pred_std)

        Note: calculations done in real scales
        shape: (len(num_lowres_list), number of total highRes, k bins)        
        """
        return self._all_norm_res

    @property
    def all_norm_res_hf(self) -> np.ndarray:
        """
        All standardized errors from high-fidelity emulator,

            (pred_mean - true / pred_std)

        Note: calculations done in real scales
        shape: (len(num_lowres_list), number of total highRes, k bins)        
        """
        return self._all_norm_res_hf
