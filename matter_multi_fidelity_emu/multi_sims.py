"""
Create a HDF5 product from multiple simulations from different
input parameters.

Class:
----
:GadgetLoad: handle a single Gadget output filenames
:PowerSpec: handle all powerspec files to a big array
:MultiPowerSpec: a class to generate a single hdf5 catalogue
    for powerspecs in the folder
"""
from typing import Generator, List, Union, Tuple, Optional
import re
import os
import json
from glob import glob
import numpy as np
import h5py

# the function I used to generate dm-only tests outdirs
# outdir auto generated, since we will have many folders
fn_outdir = lambda i, res, box: "test-{}-{}-dmonly_{}".format(res, box, str(i).zfill(4))

# get powerspec filename using scale factor
powerspec_fn = lambda scale_factor: "powerspectrum-{:.4f}.txt".format(scale_factor)


class GadgetLoad(object):
    """
    handle the output filenames generated by MP-Gadget

    Parameters:
    ----
    submission_dir (str) : path to the folder of MP-Gadget submission

    Files:
    ----
    slurm log
    SimulationICs.json -> dict
    camb_linear/*
    output/cpu.txt*
    output/sfr.txt*
    output/powerspectrum-*.txt
    """

    def __init__(
        self,
        submission_dir: str = "test/",
        slurm_prefix: str = "slurm",
        powerspec_prefix: str = "powerspectrum-",
        cpu_prefix: str = "cpu.txt",
        sfr_prefix: str = "sfr.txt",
        snapshots: str = "Snapshots.txt",
    ) -> None:
        self.submission_dir = submission_dir

        # assume the files are generated the same as the files generated
        # by the simulationics
        # theses filenames are saved for reference
        self._files = glob(os.path.join(submission_dir, "*"))
        self._outputfiles = glob(os.path.join(submission_dir, "output", "*"))

        assert os.path.join(submission_dir, "mpgadget.param") in self._files

        # make sure you run until a = 1
        assert (
            os.path.join(submission_dir, "output", "powerspectrum-1.0000.txt")
            in self._outputfiles
        )

        # these read into np.arrays
        self.powerspec_files = glob(
            os.path.join(submission_dir, "output", "powerspectrum-*.txt")
        )
        self.camb_files = glob(
            os.path.join(submission_dir, "camb_linear", "ics_matterpow_*.dat")
        )

        assert len(self.powerspec_files) > 0
        assert len(self.camb_files) > 0

        # these read into strings
        self.slurm_files = glob(
            os.path.join(submission_dir, "{}*".format(slurm_prefix))
        )
        self.cpu_files = glob(
            os.path.join(submission_dir, "output", "{}*".format(cpu_prefix))
        )
        self.sfr_files = glob(
            os.path.join(submission_dir, "output", "{}*".format(sfr_prefix))
        )

        # read in params
        param_filename = os.path.join(submission_dir, "SimulationICs.json")
        self._param_dict = self.read_simulationics(param_filename)

        # read the snapshot list
        # | No. of snapshot | scale factor |
        self._snapshots = np.loadtxt(os.path.join(submission_dir, "output", snapshots))

        assert np.all(self._snapshots[:, 1] <= 1)
        assert np.all(self._snapshots[:, 1] >= 0)

    @property
    def snapshots(self) -> np.ndarray:
        """
        A table records the index of snapshots and
        the scale factor of the snapshots.
        """
        return self._snapshots

    @property
    def param_dict(self) -> dict:
        """
        A json file generated by simulationics, which includes all params
        to reproduce a MP-Gadget simulation.
        """
        return self._param_dict

    @property
    def files(self) -> List:
        """files in the submission folder"""
        return self._files

    @property
    def outputfiles(self) -> List:
        """files in the output folder"""
        return self._outputfiles

    @staticmethod
    def read_simulationics(filename: str) -> dict:
        """
        read in the simulationics as a dict
        """
        with open(filename, "r") as f:
            out = json.load(f)

        return out

    @staticmethod
    def get_number(regex: str, filename: str) -> float:
        """
        Get the number out of the filename using regex
        """
        r = re.compile(regex)

        out = r.findall(filename)

        assert len(out) == 1
        del r

        return float(out[0])

    @staticmethod
    def read_array(filename: str) -> np.ndarray:
        out = np.loadtxt(filename)
        return out

    @staticmethod
    def read_strings(filename: str) -> str:
        with open(filename, "r") as f:
            out = f.read()

        return out


class PowerSpec(GadgetLoad):
    """
    A class to generate a matrix of powerspec(k, z),
    with the corresponding SimultionICs.json file.

    Attrs:
    ----
    :camb_matter:
    :camb_redshits:
    :scale_factors:
    :powerspecs:
    """

    def __init__(self, submission_dir: str = "test/") -> None:
        super(PowerSpec, self).__init__(submission_dir)

        # read into arrays
        # Matter power specs from simulations
        scale_factors, out = self.read_powerspec(self.powerspec_files)

        self._scale_factors = scale_factors
        self._powerspecs = out

        # Matter power specs from CAMB linear theory code
        redshifts, out = self.read_camblinear(self.camb_files)

        self._camb_redshifts = redshifts
        self._camb_matters = out

    @property
    def camb_matters(self) -> np.ndarray:
        """
        The matter powerspecs output by the camb linear theory code
        """
        return self._camb_matters

    @property
    def camb_redshifts(self) -> List:
        """
        The redshifts for the camb linear powerspecs
        """
        return self._camb_redshifts

    @property
    def powerspecs(self) -> np.ndarray:
        """
        Power specs generated by MP-Gadgets, from multiple redshifts
        P(k, z) = (num of redshifts, number k modes, 4 cols)
        """
        return self._powerspecs

    @property
    def scale_factors(self) -> List:
        """The scale factors of powerspecs in output/"""
        return self._scale_factors

    def read_camblinear(self, camb_files: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a list of camb files into a single linear powerspec matrix,
        in shape: (n files, m k-modes, 2 cols)
        """
        # some loading funcions to make the writing clearer
        load_fn = lambda f: os.path.join(self.submission_dir, "camb_linear", f)
        matterpower_fn = lambda z: "ics_matterpow_{:.2g}.dat".format(z)
        assert os.path.exists(load_fn(matterpower_fn(0)))

        # store the length first
        length = len(camb_files)

        # careful about the ordering
        regex = load_fn("ics_matterpow_(.*).dat")

        # those powerspec files are named by the redshifts from 99 to 0
        redshifts = [self.get_number(regex, f) for f in camb_files]

        redshifts = sorted(redshifts, reverse=True)
        assert redshifts[-1] == 0.0
        assert length == len(redshifts)

        # make sure the size
        (rows, cols) = self.read_array(load_fn(matterpower_fn(0))).shape

        # alloc the array
        out = np.zeros((length, rows, cols), dtype=np.float)

        for i, this_z in enumerate(redshifts):
            this_matter = self.read_array(load_fn(matterpower_fn(this_z)))

            out[i, :, :] = this_matter

        return redshifts, out

    def read_powerspec(
        self, powerspec_files: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a list of powerspec files into a single powerspec matrix,
        in shape: (n files, m k-modes, 4 cols)
        """
        # some loading funcions to make the writing clearer
        load_fn = lambda f: os.path.join(self.submission_dir, "output", f)
        assert os.path.exists(load_fn(powerspec_fn(1)))

        # store the length first
        length = len(powerspec_files)

        # these files should be in-order, so must be super careful
        # here search a list of scale factor first by a regex
        regex = load_fn("powerspectrum-(.*).txt")

        # those powerspec files are named by the scale factors from 0.1 to 1
        scale_factors = [self.get_number(regex, f) for f in powerspec_files]

        # make it in order so that we can read in in-order
        scale_factors = sorted(scale_factors)
        assert scale_factors[-1] == 1.0  # make sure you reach the z = 0
        assert length == len(scale_factors)

        # make sure the size
        (rows, cols) = self.read_array(load_fn(powerspec_fn(1))).shape

        # alloc the array
        out = np.zeros((length, rows, cols), dtype=np.float)

        for i, this_scale_factor in enumerate(scale_factors):
            this_powerspec = self.read_array(load_fn(powerspec_fn(this_scale_factor)))

            out[i, :, :] = this_powerspec

        return scale_factors, out


class MultiPowerSpec(object):
    """
    A class to generate a single HDF5 file from powerspecs
    of multiple simulations.

    Parameters:
    ----
    all_submission_dirs (list) : a list of paths to the simulation folders
    Latin_file (str)           : a path to the Latin hypercube json file

    Output File:
    ----
    **LatinDict
    simulation_1
        - powerspecs
        - scale_factors
        - camb_matter
        - camb_redshifts
        - **param_dict
    simulation_2
    ...    
    """

    def __init__(
        self,
        all_submission_dirs: List[str],
        Latin_json: str = "Latin.json",
        selected_ind: Optional[np.ndarray] = None,
    ) -> None:
        # all the paths you want to load PowerSpecs
        # note these paths will be discared after loading
        # will not store in the hdf5 file.
        self.all_submission_dirs = all_submission_dirs
        self.Latin_json = Latin_json

        # load Latin HyperCube sampling into memory
        self.Latin_dict = self.load_Latin(self.Latin_json)

        # [selected_ind] if you only run partially the Latin Hyper cube
        if selected_ind is not None:
            for name in self.Latin_dict["parameter_names"]:
                self.Latin_dict[name] = self.Latin_dict[name][selected_ind]

                assert len(self.Latin_dict[name]) == len(selected_ind)

    @staticmethod
    def load_Latin(Latin_json: str) -> dict:
        with open(Latin_json, "r") as f:
            out = json.load(f)

        # make each list to be ndarrays
        for key, val in out.items():
            out[key] = np.array(val)

        # you need special treatment to handle list of strs in hdf5
        out["parameter_names"] = np.array(
            out["parameter_names"], dtype=h5py.string_dtype(encoding="utf-8")
        )

        return out

    def create_hdf5(self, hdf5_name: str = "MutliPowerSpecs.hdf5") -> None:
        """
        - Create a HDF5 file for powerspecs from multiple simulations.
        - Each simulation stored in subgroup, includeing powerspecs and
        camb linear power specs.
        - Each subgroup has their own simulation parameters extracted from
        SimulationICs.json to reproduce this simulation.
        - Parameters from Latin HyperCube sampling stored in upper group level,
        the order of the sampling is the same as the order of simulations.

        TODO: add a method to append new simulations to a created hdf5.
        """
        # open a hdf5 file to store simulations
        with h5py.File(hdf5_name, "w") as f:
            # store the sampling from Latin Hyper cube dict into datasets:
            # since the sampling size should be arbitrary, we should use
            # datasets instead of attrs to stores these sampling arrays
            for key, val in self.Latin_dict.items():
                f.create_dataset(key, data=val)

            # using generator to iterate through simulations,
            # PowerSpec stores big arrays so we don't want to load
            # everything to memory
            for i, ps in enumerate(self.load_PowerSpecs(self.all_submission_dirs)):
                sim = f.create_group("simulation_{}".format(i))

                # store arrays to sim subgroup
                sim.create_dataset("scale_factors", data=np.array(ps.scale_factors))
                sim.create_dataset("powerspecs", data=ps.powerspecs)

                sim.create_dataset("camb_redshifts", data=np.array(ps.camb_redshifts))
                sim.create_dataset("camb_matters", data=ps.camb_matters)

                # stores param json to metadata attrs
                for key, val in ps.param_dict.items():
                    sim.attrs[key] = val

    @staticmethod
    def load_PowerSpecs(all_submission_dirs: List[str]) -> Generator:
        """
        Iteratively load the PowerSpec class
        """
        for submission_dir in all_submission_dirs:
            yield PowerSpec(submission_dir)


def take_params_dict(Latin_dict: dict) -> Generator:
    """
    take the next param dict with a single
    sample for each param
    """
    parameter_names = Latin_dict["parameter_names"]
    length = len(Latin_dict[parameter_names[0]])

    assert length == len(Latin_dict[parameter_names[-1]])

    for i in range(length):
        param_dict = {}

        for key in parameter_names:
            param_dict[key] = Latin_dict[key][i]

        yield param_dict
