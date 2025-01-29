import sys
sys.path.append("../")
import numpy as np
from error_function.dgmgp_error import generate_data
# import contextlib
# import io
import os
from matter_multi_fidelity_emu.gpemulator_singlebin import SingleBindGMGP 
from matter_multi_fidelity_emu.gpemulator_singlebin import _map_params_to_unit_cube as input_normalize
from typing import Tuple, List, Optional, Dict

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
            parallel=True,
            num_processes=10
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


zs = ['0', '0.2', '0.5', '1', '2', '3']


def input_norm(cosmo_params: np.ndarray, bounds):
    cosmo_params_norm = np.zeros_like(cosmo_params)
    for i in range(len(cosmo_params)):
        for j in range(len(bounds)):
            cosmo_params_norm[i][j] = (cosmo_params[i][j] - bounds[j][0]) / (bounds[j][1] - bounds[j][0])
    return cosmo_params_norm

class MatterPowerEmulator:
    """
    API for predicting the matter power spectrum using a pre-trained emulator.

    Parameters:
    ----
    :param model_path: Path to the pretrained models.
    """
    def __init__(self, model_path: str="../test/pre-trained"):

        zs_str = ['0', '0.2', '0.5', '1', '2', '3']  # redshifts supported at the moment
        self.zs = [float(z) for z in zs_str]
        models_zs = []
        # Load models for each redshift
        for z in zs_str:
            model_dir = os.path.join(model_path, f'a{1/(1+float(z)):.4f}')
            models_zs.append(SingleBindGMGP(load_path=model_dir))

        self.models_zs = models_zs
        # load k values
        lgk = np.loadtxt("../data/dev/dev_297_Box25_Part75_27_Box100_Part300_z0/kf.txt")
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
        Predict the matter power spectrum for given cosmological parameters.
        
        If `cosmo_params` is not provided, individual cosmological parameters will be used.

        Parameters:
        ----
        :param cosmo_params: A NumPy array of shape (n_samples, n_params) containing all cosmological parameters.
        :param Omega_m: Matter density parameter.
        :param Omega_b: Baryon density parameter.
        :param hubble: Hubble constant (h).
        :param scalar_amp: Scalar amplitude.
        :param ns: Spectral index.
        :param w0: Dark energy equation of state parameter.
        :param wa: Evolution of dark energy equation of state.
        :param mnu: Sum of neutrino masses (eV).
        :param Neff: Effective number of neutrino species.
        :param alphas: Running of the spectral index.
        
        Returns:
        ----
        :return: A dictionary containing "k", "mode_P", and "variance_P".
        """
        # Get the model for the given redshift, raise error if redshift is not supported
        if redshift not in self.zs:
            raise ValueError(f"Redshift {redshift} is not supported. Choose from {self.zs}")
        model_z = self.models_zs[self.zs.index(redshift)]
        
        # If cosmo_params is not provided, use individual parameters
        if cosmo_params is None:
            cosmo_params = np.array([Om, Ob, hubble, As, ns, w0, wa, mnu, Neff, alphas])
            cosmo_params = cosmo_params.reshape(1, -1)  # Ensure it has correct shape
        elif len(cosmo_params.shape) == 1:
            cosmo_params = cosmo_params.reshape(1, -1)

        # Load parameter bounds
        bounds = np.loadtxt("../data/dev/dev_297_Box25_Part75_27_Box100_Part300_z0/input_limits.txt")

        # Normalize input
        cosmo_params_norm = input_norm(cosmo_params, bounds)  

        # Get predictions
        mean, var = model_z.predict(cosmo_params_norm)  # mean(log10(P)), var(log10(P))

        # Compute mode and variance in linear scale
        mode_P = 10**mean * np.exp(-var * (np.log(10)) ** 2)
        var_P = (
            10 ** (2 * mean)
            * np.exp(var * (np.log(10)) ** 2)
            * (np.exp(var * (np.log(10)) ** 2) - 1)
        )
        # Return results: k, mode_P, and var_P
        return self.k, mode_P, var_P