"""
Loading the total mass power spectrum from default
MP-Gadget simulations.
"""
from .multi_sims import MultiPowerSpec, PowerSpec
from typing import Tuple, List, Optional, Generator

import os
import re
from glob import glob

import numpy as np
from scipy.interpolate import interp1d
import h5py

from .rebin_powerspectrum import modecount_rebin

powerspec_fn = lambda scale_factor: "powerspectrum-{:.4f}.txt".format(scale_factor)


def load_total_mass(
    submission_dir: str, scale_factor: int, minmodes: int = 200, ndesired: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the total mass power spectrum from MP-Gadget powerspectrum files

    Parameters:
    ----
    submission_dir: your MP-Gadget simulation submission folder.
    scale_factor: the scale factor ( 1 / (1 + z0)) of the power spectrum.

    Returns:
    ----
    kk: k bins of the power spectrum (after rebinning)
    pk: power spectrum
    modes: number of modes at each bin
    """

    load_fn = lambda f: os.path.join(submission_dir, "output", f)

    # check if the simulation run until z=0
    assert os.path.exists(load_fn(powerspec_fn(1)))

    # store all available scale factors
    powerspec_files = glob(
        os.path.join(submission_dir, "output", "powerspectrum-*.txt")
    )
    # store the length first
    length = len(powerspec_files)
    # these files should be in-order, so must be super careful
    # here search a list of scale factor first by a regex
    regex = load_fn("powerspectrum-(.*).txt")
    # those powerspec files are named by the scale factors from 0.1 to 1
    scale_factors = [get_number(regex, f) for f in powerspec_files]
    # make it in order so that we can read in in-order
    scale_factors = sorted(scale_factors)
    assert scale_factors[-1] == 1.0  # make sure you reach the z = 0
    assert length == len(scale_factors)

    # make sure the input scale factor is available
    diff = np.abs(np.array(scale_factors) - scale_factor)
    if min(diff) > 0.0001:
        raise Exception(
            "input scale factor {} is not in the MP-Gadget output files.".format(
                scale_factor
            )
        )
    filename = "powerspectrum-{:.4f}.txt".format(scale_factors[np.argmin(diff)])
    print("[Info] Loading powerspetrum file ...", filename)

    # load the file
    powerspec = np.loadtxt(load_fn(filename))
    kk, pk, modes, _ = powerspec.T
    print("[Info] number of k bins before rebinning", kk.shape)

    # rebin powerspectrum
    kk, pk, modes = modecount_rebin(kk, pk, modes, minmodes=minmodes, ndesired=ndesired)

    return kk, pk, modes


def get_number(regex: str, filename: str) -> float:
    """
    Get the number out of the filename using regex
    """
    r = re.compile(regex)

    out = r.findall(filename)

    assert len(out) == 1
    del r

    return float(out[0])


class MPGadgetPowerSpec(PowerSpec):

    """
    Loading MP-Gadget powerspec.

    Loading all snapshots for MP-Gadget outputs.

    Note: the training set will be directly outputed from MultiMPGadgetPowerSpec
    """

    def __init__(
        self, submission_dir: str = "test/", scale_factor: float = 1.0,
    ) -> None:
        super(PowerSpec, self).__init__(submission_dir)

        # read into arrays
        # Matter power specs from simulations
        kk, pk, modes = self.read_powerspec(
            submission_dir=submission_dir, scale_factor=scale_factor
        )

        self._scale_factor = scale_factor

        self._k0 = kk
        self._powerspecs = pk
        self._modes = modes  # TODO: implement uncertainty

        # Matter power specs from CAMB linear theory code
        redshifts, out = self.read_camblinear(self.camb_files)

        self._camb_redshifts = redshifts
        self._camb_matters = out

    @property
    def powerspecs(self) -> np.ndarray:
        """
        P(k) from output folder. The same length as k0.
        """
        return self._powerspecs

    @property
    def modes(self) -> np.ndarray:
        """
        Number of modes counted in a given k bin. The same length as k0.
        """
        return self._modes

    @property
    def poisson_noise(self) -> np.ndarray:
        """
        Poisson noise,
        Var = number counts
        here return the square root of the variance
        """
        return np.sqrt(self._modes)

    @property
    def k0(self) -> np.ndarray:
        """
        k from output folder. The same length as powerspecs
        """
        return self._k0

    def read_powerspec(
        self, submission_dir: str, scale_factor: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read power spectrum from a output folder

        Parameters:
        ----
        z0 : the redshift of the power spectrum you want to load.
        """

        # the maximum k is controlled by Ng
        kk, pk, modes = load_total_mass(submission_dir, scale_factor=scale_factor)

        # filter out NaN values
        ind = ~np.isnan(kk)
        assert np.all(ind == ~np.isnan(kk))

        return kk, pk, modes


class MultiMPGadgetPowerSpec(MultiPowerSpec):

    """
    Output a HDF5 file

    This class tries to convert the powerspec data (across redshifts) output from MP-Gadget to a single h5 file.
    This meant to be similar to the routines in sbird.lya_emulator_full.

    Parameters:
    ----
    all_submission_dirs : a List of paths to the MP-Gadget simulation submission folders.
    Latin_json : path to the Latin hypercube json file.
    selected_ind : select a subset of LHD to generate the h5 file.
    """

    def __init__(
        self,
        all_submission_dirs: List[str],
        Latin_json: str,
        selected_ind: Optional[np.ndarray],
        scale_factors: List[float] = [1.0000, 0.8333, 0.6667, 0.5000, 0.3333, 0.2500],
    ) -> None:
        super().__init__(
            all_submission_dirs, Latin_json=Latin_json, selected_ind=selected_ind
        )

        # assign attrs for loading MPGadget power specs
        self.scale_factors = scale_factors

        self.selected_ind = selected_ind

    def create_hdf5(self, hdf5_name: str = "MultiMPGadgetPowerSpec.hdf5") -> None:

        """
        - Create a HDF5 file for powerspecs from multiple simulations.
        - Parameters from Latin HyperCube sampling stored in first layer,
        the order of the sampling is the same as the order of simulations.
        - Powerspecs stored as a large array in the first layer.
        - Stores redshift and scale factor arrays
        - Store k bins.
        TODO: add a method to append new simulations to a created hdf5.
        <KeysViewHDF5 ['flux_vectors', 'kfkms', 'kfmpc', 'params', 'zout']>

        """
        # open a hdf5 file to store simulations
        with h5py.File(hdf5_name, "w") as f:
            # Save the selected ind for comparing with LF
            if self.selected_ind is not None:
                f.create_dataset("selected_ind", data=np.array(self.selected_ind))

            # store the sampling from Latin Hyper cube dict into datasets:
            # since the sampling size should be arbitrary, we should use
            # datasets instead of attrs to stores these sampling arrays
            for key, val in self.Latin_dict.items():
                f.create_dataset(key, data=val)

            # Make input parameter column
            parameter_names = self.Latin_dict["parameter_names"]
            # (num of sims, num of parameters)
            params = np.full(
                (len(self.Latin_dict[parameter_names[0]]), len(parameter_names),),
                fill_value=np.nan,
            )
            for ith_param, pname in enumerate(parameter_names):
                params[:, ith_param] = self.Latin_dict[pname]
            assert np.sum(np.isnan(params)) < 1
            f.create_dataset("params", data=params)

            f.create_dataset("scale_factors", data=np.array(self.scale_factors))
            f.create_dataset("zout", data=1 / np.array(self.scale_factors) - 1)

            # Get the shape first
            # TODO: might be a better way append arrays
            ps_test = MPGadgetPowerSpec(
                self.all_submission_dirs[0], self.scale_factors[0]
            )  # Note: there are default rebinning parameters

            kk = ps_test.k0  # Note: assumption is all k bins are the same
            ps_size = ps_test.powerspecs.shape[0]
            mode_size = ps_test.modes.shape[0]
            assert kk.shape[0] == ps_size
            assert mode_size == ps_size

            # (num of simulations, num of redshifts, num of k bins)
            sim_powerspecs = np.full(
                (len(self.all_submission_dirs), len(self.scale_factors), ps_size,),
                fill_value=np.nan,
            )
            sim_modes = np.full(
                (len(self.all_submission_dirs), len(self.scale_factors), ps_size,),
                fill_value=np.nan,
            )

            f.create_dataset("kfmpc", data=np.array(kk))

            for ith_sim, submission_dir in enumerate(self.all_submission_dirs):
                for jth_red, scale_factor in enumerate(self.scale_factors):
                    ps = MPGadgetPowerSpec(submission_dir, scale_factor)

                    sim_powerspecs[ith_sim, jth_red, :] = ps.powerspecs
                    sim_modes[ith_sim, jth_red, :] = ps.modes

            assert np.sum(np.isnan(sim_powerspecs)) < 1
            f.create_dataset("powerspecs", data=sim_powerspecs)
            f.create_dataset("modes", data=sim_modes)

    @staticmethod
    def load_PowerSpecs(
        all_submission_dirs: List[str], scale_factor: float
    ) -> Generator:
        """
        Iteratively load the PowerSpec class
        """
        for submission_dir in all_submission_dirs:
            yield MPGadgetPowerSpec(submission_dir, scale_factor)
