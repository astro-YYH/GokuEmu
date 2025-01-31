import sys
sys.path.append("../")
import numpy as np
# import contextlib
# import io
import os
from matter_multi_fidelity_emu.gpemulator_singlebin import SingleBindGMGP 
from typing import Tuple, List, Optional, Dict

def input_norm(cosmo_params: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Normalize cosmological parameters to the range [0,1] based on given parameter bounds.

    Parameters:
    ----
    :param cosmo_params: (np.ndarray) Array of cosmological parameters to normalize.
    :param bounds: (np.ndarray) Array of shape (n_params, 2) containing min/max values for each parameter.

    Returns:
    ----
    :return: (np.ndarray) Normalized cosmological parameters in the range [0,1].
    """
    return (cosmo_params - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

class MatterPowerEmulator:
    """
    An emulator for predicting the nonlinear matter power spectrum using a pre-trained Gaussian Process model.

    This class loads pre-trained models for different redshifts and provides predictions based on input cosmological parameters.

    Parameters:
    ----
    :param model: Name of the emulator model to load. Options:
        - "GokuEmu"  : Standard emulator trained on the whole Goku suite
        - "GokuEmu-W": Wide-range parameter space emulator trained solely on Goku-W
        - "GokuEmu-N": Narrow-range parameter space emulator trained solely on Goku-N
    """

    # Define the allowed redshifts for the emulator
    PRESET_ZS = np.array([0, 0.2, 0.5, 1, 2, 3], dtype=float)

    def __init__(self, model: str="GokuEmu", redshifts: Optional[List[float]] = None):
        """
        Initialize the Matter Power Emulator by loading pre-trained models for selected redshifts.

        Parameters:
        ----
        :param model: (str) The name of the emulator model to use. Choices:
            - "GokuEmu"  : Default model
            - "GokuEmu-W": Wide-range emulator
            - "GokuEmu-N": Narrow-range emulator

        Raises:
        ----
        :raises ValueError: If an invalid model name is provided.
        """

        self.goku_model = model

        model_paths = {
            "GokuEmu": "../emulator/pre-trained/goku",
            "GokuEmu-W": "../emulator/pre-trained/goku-w",
            "GokuEmu-N": "../emulator/pre-trained/goku-n"
        }

        model_path = model_paths.get(model)
        if model_path is None:
            raise ValueError(f"Invalid model '{model}'. Choose from {list(model_paths.keys())}.")

        # Validate and set redshifts
        if redshifts is None:
            self.zs = self.PRESET_ZS  # Use all pre-set redshifts by default
        else:
            redshifts = np.array(redshifts, dtype=float)
            if np.any((redshifts < self.PRESET_ZS.min()) | (redshifts > self.PRESET_ZS.max())):
                raise ValueError(f"Redshifts must be within {self.PRESET_ZS}. Given: {redshifts}")

            # Keep only valid redshifts from PRESET_ZS
            self.zs = np.array([z for z in self.PRESET_ZS if z in redshifts])

            if self.zs.size == 0:
                raise ValueError(f"No valid redshifts selected. Choose from {self.PRESET_ZS}.")
            
        models_zs = []
        # Load models for each redshift
        for z in self.zs:
            model_dir = os.path.join(model_path, f'a{1/(1+z):.4f}')
            models_zs.append(SingleBindGMGP(load_path=model_dir))

        self.models_zs = models_zs
        # load k values
        lgk = np.loadtxt("../data/combined/matter_power_1128_Box1000_Part750_36_Box1000_Part3000_z0/kf.txt")
        self.k = 10**lgk

    def predict(
        self,
        cosmo_params: Optional[np.ndarray] = None,
        Om: float = 0.3,
        Ob: float = 0.05,
        hubble: float = 0.7,
        As: float = 2.1e-9,
        ns: float = 0.96,
        w0: float = -1.0,
        wa: float = 0.0,
        mnu: float = 0.06,
        Neff: float = 3.044,
        alphas: float = 0.0,
        redshift: float = 0.0,
        ) -> Dict[str, np.ndarray]:
        """
        Predict the nonlinear matter power spectrum for a given set of cosmological parameters.

        If `cosmo_params` is provided, it should be a NumPy array of shape `(n_samples, n_params)`. Otherwise, individual parameter values will be used.

        Parameters:
        ----
        :param cosmo_params: (Optional[np.ndarray]) An array containing one/multiple set(s) of cosmological parameters.
        :param Om: (float) Total matter density parameter (Omega_m).
        :param Ob: (float) Baryon density parameter (Omega_b).
        :param hubble: (float) Reduced Hubble parameter (h).
        :param As: (float) Scalar amplitude of primordial fluctuations.
        :param ns: (float) Spectral index of the primordial power spectrum.
        :param w0: (float) Dark energy equation of state parameter at present.
        :param wa: (float) Evolution of dark energy equation of state.
        :param mnu: (float) Sum of neutrino masses (eV).
        :param Neff: (float) Effective number of relativistic neutrino species.
        :param alphas: (float) Running of the spectral index.
        :param redshift: (float) Redshift at which to compute the power spectrum.

        Returns:
        ----
        :return: A tuple containing:
            - k (np.ndarray): Wavenumber values in h/Mpc.
            - mode_P (np.ndarray): Predicted power spectrum values.
            - variance_P (np.ndarray): Variance of the predictions.

        Raises:
        ----
        :raises ValueError: If the given redshift is not supported.
        """
        # Get the model for the given redshift, raise error if redshift is not supported
        idx = np.where(self.zs == redshift)[0]
        if idx.size == 0:
            raise ValueError(f"Redshift {redshift} is not supported. Choose from {self.zs}")

        model_z = self.models_zs[idx[0]]
        
        # If cosmo_params is not provided, use individual parameters
        if cosmo_params is None:
            cosmo_params = np.array([[Om, Ob, hubble, As, ns, w0, wa, mnu, Neff, alphas]])
        else:
            cosmo_params = np.atleast_2d(cosmo_params)

        # Load parameter bounds
        if self.goku_model == "GokuEmu-N":
            bounds = np.loadtxt("../data/narrow/matter_power_564_Box1000_Part750_15_Box1000_Part3000_z0/input_limits.txt")
        else:
            bounds = np.loadtxt("../data/combined/matter_power_1128_Box1000_Part750_36_Box1000_Part3000_z0/input_limits.txt")

        # Normalize input
        cosmo_params_norm = input_norm(cosmo_params, bounds)  

        # Get predictions
        mean, var = model_z.predict(cosmo_params_norm)  # mean(log10(P)), var(log10(P))

        # Compute mode and variance in linear scale
        log10_sq = (np.log(10)) ** 2
        mode_P = 10**mean * np.exp(-var * log10_sq)
        var_P = 10**(2 * mean) * np.exp(var * log10_sq) * (np.exp(var * log10_sq) - 1)

        # Return results: k, mode_P, and var_P
        return self.k, mode_P, var_P