"""
Compute halo mass functions and save into a HDF5 file
"""
from typing import Generator, List

import os
import numpy as np
import re

from .multi_sims import GadgetLoad
from .hmffromfof import HMFFromFOF

class MPISubmit(object):

    """
    Read the mpi_submit, the sbatch submission for MP-Gadget
    into a class.
    """

    def __init__(self, filename: str, gadget_dir: str = "~/codes/MP-Gadget/"):
        self.filename = filename
        self.gadget_dir = gadget_dir

        with open(self.filename, "r") as f:
            self.txt = f.readlines()


    def get_basic_setups(self):
        """
        Return the lines without `mpirun` - this (ideally) includes
        - SBATCH parameters
        - module loads
        - exports
        """
        # only check the 1st position in a line;
        # this is because sbatch job's name could have mpirun
        exclusion_bag = {"mpirun"}

        basic_txt = []

        for line in self.txt:
            words = line.split()

            if np.any([words[0] == exclusion_word for exclusion_word in exclusion_bag]):
                continue
    
            basic_txt.append(line)

        return basic_txt

    def make_simulation_foftable(self, snapshots: List[int], mpi_submit_file: str = "mpi_submit_foftables", mpgadget_param_file: str = "mpgadget.param") -> None:
        """
        Generate a submission file for making the fof table (PART -> PIG)
        """
        basic_txt = self.get_basic_setups()

        # snapshots could be more than one
        for snapshot in snapshots:
            basic_txt.append(self.mpirun_foftable(snapshot, mpgadget_param_file))

        with open(mpi_submit_file, "w") as f:
            f.write("".join(basic_txt))


    def mpirun_foftable(self, snapshot: int, mpgadget_param_file: str = "mpgadget.param") -> str:
        """
        The line to make foftable from PART snapshot       
        """
        return "mpirun --map-by core {} {} 3 {}\n".format(self.gadget_dir, mpgadget_param_file, snapshot)

class HaloMassFunction(GadgetLoad):

    """
    Compute the halo mass functions from foftable using MP-Gadget's
    tool, HMFFromFOF (TODO: binning scheme might need to adjust,
    https://jakevdp.github.io/blog/2012/09/12/dynamic-programming-in-python/)

    There was a bug before that FOF tables were not generated from
    the PART during the run time. Generate a script to get FOF tables
    out of PART.

    Example:
    ./MP-Gadget paramfile.gadget 3 $snapnum
    """

    def __init__(self, submission_dir: str = "test/", gadget_dir: str = "~/codes/MP-Gadget/") -> None:
        super(HaloMassFunction, self).__init__(submission_dir)

        self.gadget_dir = gadget_dir

        # acquire the current PART files
        self._parts = sorted(self.get_PART_array())
        # acquire the current PIG files
        self._pigs = sorted(self.get_PIG_array())

        # read mpi_submit into a class
        # keep all SBATCH parameters the same;
        # keep all module load the same
        # make the mpirun free parameter
        self.mpi_submit = MPISubmit(
            os.path.join(self.submission_dir, "mpi_submit"), self.gadget_dir
        )

        # check # of FOF tables == # of snapshots
        self._pigs_to_run = set(self._parts) - set(self._pigs)
        if len(self._pigs_to_run) > 0:
            print("[Warning] Some snapshots lack of FOF tables, recommend do self.make_foftables.")

    def read_hmf(self):
        """
        Compute halo mass functions from PIG/ and read them into memory
        """
        # checking if you need to re-generate FOF tables
        assert len(self._parts) == len(self._pigs)
        # checking if all snaphots in the table are available
        if len(self.snapshots[:, 0]) > len(self._parts):
            print("[Warning] missing snaphots ...")
            print("... recommend rerun {}".format( set(self.snapshots[:, 1]) - set(self._parts) ))

        # snapshot table records all PIG need to read
        # snapshot table: | No. of snapshot | scale factor |
        
        # Note: here we only load all available PIGs, instead of all PIGs on the table
        self.scale_factors = self.snapshots[:, 1][self._pigs]

        foftable = lambda snapshot : "PIG_{}".format(snapshot)

        for i in self._pigs:
            HMFFromFOF(foftable=foftable(i), h0=False, bins="auto")


    def make_foftables(self, mpgadget_param_file: str = "mpgadget.param"):
        """
        Make FOF tables from PART.

        After generating the sh files, go to the submission folder and submit the .sh file.
        """
        self.mpi_submit.make_simulation_foftable(
            self._pigs_to_run, os.path.join(self.submission_dir, "mpi_submit_foftables", mpgadget_param_file)
        )



    def get_PART_array(self) -> Generator:
        """
        Get indexes for PART snapshot folders
        """
        for f in self._outputfiles:
            out = re.findall(r"PART_([0-9]*)", f)

            # if matched
            if len(out) == 1:
                yield int(out[0])
    
    def get_PIG_array(self) -> Generator:
        """
        Get indexes for PIG snapshot folders
        """
        for f in self._outputfiles:
            out = re.findall(r"PIG_([0-9]*)", f)

            # if matched
            if len(out) == 1:
                yield int(out[0])
    




