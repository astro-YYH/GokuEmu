"""
Building multi-Fidelity emulator using many single-output GP.

1. SingleBinGP: the single-fidelity emulator in the paper.
2. SingleBinLinearGP: the linear multi-fidelity emulator (AR1).
3. SingleBinNonLinearGP: the non-linear multi-fidelity emulator (NARGP).
4. SingleBinDeepGP: the deep GP for multi-fidelity (MF-DGP). This one is not
    mentioned in the paper due to we haven't found a way to fine-tune the
    hyperparameters.

Most of the model constructions are similar to Emukit's examples, with some
modifications on the choice of hyperparameters and modelling each output as
an independent GP (many single-output GP).
"""
from typing import Tuple, List, Optional, Dict

import logging
import numpy as np
import os
import GPy
from emukit.model_wrappers import GPyModelWrapper, GPyMultiOutputWrapper

from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
from emukit.multi_fidelity.convert_lists_to_array import (
    convert_y_list_to_array,
    convert_xy_lists_to_arrays,
    convert_x_list_to_array,
)
import multiprocessing
import psutil
import threading


# we made modifications on not using the ARD for high-fidelity
from .non_linear_multi_fidelity_models.non_linear_multi_fidelity_model import NonLinearMultiFidelityModel, make_non_linear_kernels
# from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import NonLinearMultiFidelityModel, make_non_linear_kernels

from .latin_hypercube import map_to_unit_cube_list

_log = logging.getLogger(__name__)

class SingleBinGP:
    """
    A GPRegression models GP on each k bin of powerspecs
    
    :param X_hf: (n_points, n_dims) input parameters
    :param Y_hf: (n_points, k modes) power spectrum
    """
    def __init__(self, X_hf: np.ndarray, Y_hf: np.ndarray):
        # a list of GP emulators
        gpy_models: List = []

        # make a GP on each P(k) bin
        for i in range(Y_hf.shape[1]):
            # Standard squared-exponential kernel with a different length scale for
            # each parameter, as they may have very different physical properties.
            nparams = np.shape(X_hf)[1]

            kernel = GPy.kern.RBF(nparams, ARD=True)            

            gp = GPy.models.GPRegression(X_hf, Y_hf[:, [i]], kernel)
            gpy_models.append(gp)

        self.gpy_models = gpy_models

        self.name = "single_fidelity"

    def optimize_restarts(self, n_optimization_restarts: int, parallel: bool = False) -> None:
        """
        Optimize GP on each bin of the power spectrum.
        """
        models = []

        _log.info("\n --- Optimization: ---\n".format(self.name))
        for i,gp in enumerate(self.gpy_models):
            gp.optimize_restarts(n_optimization_restarts, parallel=parallel)
            models.append(gp)

        self.models = models

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and variance for fidelity specified by last column of X.
        Note that we predict from gp from each k bin.

        :param X: point(s) at which to predict
        :return: predicted P(all k bins) (mean, variance) at X
        """
        means = np.full((X.shape[0], len(self.models)), fill_value=np.nan)
        variances = np.full((X.shape[0], len(self.models)), fill_value=np.nan)

        for i,model in enumerate(self.models):
            mean, variance = model.predict(X)

            means[:, i] = mean[:, 0]
            variances[:, i] = variance[:, 0]

        return means, variances

    def to_dict(self) -> Dict:
        """
        Save hyperparameters into a dict
        """
        param_dict = {}

        for i,model in enumerate(self.models):        

            param_dict["bin_{}".format(i)] = model.to_dict()

        return param_dict

class SingleBinLinearGP:
    """
    A thin wrapper around GPy.core.GP that does some input checking and provides
    a default likelihood. Also model each k bin as an independent GP.

    :param X_train:  (n_fidelities, n_points, n_dims) list of parameter vectors.
    :param Y_train:  (n_fidelities, n_points, k modes) list of matter power spectra.
    :param n_fidelities: number of fidelities stored in the list.
    :param ARD_last_fidelity: whether to apply ARD for the last (highest) fidelity.
        Default, False.
    """

    def __init__(
        self,
        X_train: List[np.ndarray],
        Y_train: List[np.ndarray],
        kernel_list: Optional[List],
        n_fidelities: int,
        likelihood: GPy.likelihoods.Likelihood = None,
        ARD_last_fidelity: bool = False,
    ):
        # a list of GP emulators
        gpy_models: List = []

        self.n_fidelities = len(X_train)

        # convert into X,Y for MultiOutputGP
        # not normalize due to no improvements
        X, Y = convert_xy_lists_to_arrays(X_train, Y_train)

        # linear multi-fidelity setup
        if X.ndim != 2:
            raise ValueError("X should be 2d")

        if Y.ndim != 2:
            raise ValueError("Y should be 2d")

        if np.any(X[:, -1] >= n_fidelities):
            raise ValueError(
                "One or more points has a higher fidelity index than number of fidelities"
            )

        # make a GP on each P(k) bin
        for i in range(Y.shape[1]):
            y_metadata = {"output_index": X[:, -1].astype(int)}

            # Make default likelihood as different noise for each fidelity
            likelihood = GPy.likelihoods.mixed_noise.MixedNoise(
                [GPy.likelihoods.Gaussian(variance=1.0) for _ in range(n_fidelities)]
            )

            # Standard squared-exponential kernel with a different length scale for
            # each parameter, as they may have very different physical properties.
            kernel_list = []
            for j in range(n_fidelities):
                nparams = np.shape(X_train[j])[1]

                # kernel = GPy.kern.Linear(nparams, ARD=True)
                # kernel = GPy.kern.RatQuad(nparams, ARD=True)
                kernel = GPy.kern.RBF(nparams, ARD=True)
                
                # final fidelity not ARD due to lack of training data
                if j == n_fidelities - 1:
                    kernel = GPy.kern.RBF(nparams, ARD=ARD_last_fidelity)

                kernel_list.append(kernel)

            # make multi-fidelity kernels
            kernel = LinearMultiFidelityKernel(kernel_list)

            gp = GPy.core.GP(X, Y[:, [i]], kernel, likelihood, Y_metadata=y_metadata)
            gpy_models.append(gp)

        self.gpy_models = gpy_models

        self.name = "ar1"

    def optimize(self, n_optimization_restarts: int, parallel: bool = False) -> None:
        """
        Optimize GP on each bin of the power spectrum.
        """
        models = []

        _log.info("\n--- Optimization ---\n".format(self.name))

        for i,gp in enumerate(self.gpy_models):
            # fix noise and optimize
            getattr(gp.mixed_noise, "Gaussian_noise").fix(1e-6)
            for j in range(1, self.n_fidelities):
                getattr(
                    gp.mixed_noise, "Gaussian_noise_{}".format(j)
                ).fix(1e-6)

            model = GPyMultiOutputWrapper(gp, n_outputs=self.n_fidelities, n_optimization_restarts=n_optimization_restarts)

            _log.info("\n[Info] Optimizing {} bin ...\n".format(i))

            # first step optimization with fixed noise
            model.gpy_model.optimize_restarts(
                n_optimization_restarts,
                verbose=model.verbose_optimization,
                robust=True,
                parallel=parallel,
            )

            # unfix noise and re-optimize
            getattr(model.gpy_model.mixed_noise, "Gaussian_noise").unfix()
            for j in range(1, self.n_fidelities):
                getattr(
                    model.gpy_model.mixed_noise, "Gaussian_noise_{}".format(j)
                ).unfix()

            # first step optimization with fixed noise
            model.gpy_model.optimize_restarts(
                n_optimization_restarts,
                verbose=model.verbose_optimization,
                robust=True,
                parallel=parallel,
            )

            models.append(model)

        self.models = models

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and variance for fidelity specified by last column of X.
        Note that we predict from gp from each k bin.

        :param X: point(s) at which to predict
        :return: predicted P(all k bins) (mean, variance) at X
        """
        means = np.full((X.shape[0], len(self.models)), fill_value=np.nan)
        variances = np.full((X.shape[0], len(self.models)), fill_value=np.nan)

        for i,model in enumerate(self.models):
            mean, variance = model.predict(X)

            means[:, i] = mean[:, 0]
            variances[:, i] = variance[:, 0]

        return means, variances

    def to_dict(self) -> Dict:
        """
        Save hyperparameters into a dict
        """
        param_dict = {}

        for i,model in enumerate(self.models):

            this_param_dict = {}
        
            # a constant scaling value
            this_param_dict["scale"] = model.gpy_model.multifidelity.scale.values.tolist()
            # append dict from each key
            for j, kern in enumerate(model.gpy_model.multifidelity.kernels):
                this_param_dict["kern_{}".format(j)] = kern.to_dict()

            param_dict["bin_{}".format(i)] = this_param_dict

        return param_dict


class SingleBindGMGP:
    """
    Deep graphical Gaussian process for multi-fidelity, the core part of the code
    is provided by Dr Simon Mak and Irene Ji.

    Ref: A graphical multi-fidelity Gaussian process model, with application to
         emulation of expensive computer simulations
    https://arxiv.org/abs/2108.00306  

    NOTE: Current version only support two low-fidelity nodes.
    TODO: Customize the code to run for multiple low-fidelity nodes.

    Parameters:
    ----
    :param X_train: a List of training input parameters. Assume 2 fidelity. The final
        element is high-fidelity. (low-fidelity node 1, low-fidelity node 2, high-fidelity node).
    :param Y_train: a List of training input parameters. In the same order as the X_train.
    :param n_samples: Number of samples to use to do quasi-Monte-Carlo integration at each fidelity.
    :param optimization_restarts: number of optimization restarts you want in GPy.
    :param ARD_last_fidelity: whether to apply ARD for the last (highest) fidelity.
        Default, False.
    """

    def __init__(
        self,
        X_train: Optional[List[np.ndarray]] = None,  # Optional for loading pretrained
        Y_train: Optional[List[np.ndarray]] = None,  # Optional for loading pretrained
        n_fidelities: Optional[int] = None,          # Optional for pretrained models
        n_samples: int = 500,                       
        optimization_restarts: int = 30,             # Only used for training
        ARD_last_fidelity: bool = False,             # Only used for training
        parallel: bool = False,                      # Only used for training
        num_processes: Optional[int] = None,         # Only used for training
        load_path: Optional[str] = None              # Path for pretrained models
        ) -> None:
        """
        Initialize the multi-fidelity Gaussian Process model.

        Parameters:
        ----
        :param X_train: List of training input parameters for multiple fidelities.
        :param Y_train: List of training output parameters for multiple fidelities.
        :param n_fidelities: Number of fidelities (required for training only).
        :param n_samples: Number of samples for quasi-Monte-Carlo integration at each fidelity (training only).
        :param optimization_restarts: Number of optimization restarts in GPy (training only).
        :param ARD_last_fidelity: Whether to apply ARD for the last (highest) fidelity (training only).
        :param parallel: Whether to optimize in parallel (training only).
        :param num_processes: Number of parallel processes (training only).
        :param load_path: Path to pretrained models. If provided, models will be loaded from this path.
        """
        self.models: List = []  # Models will be stored here
        self.n_samples = n_samples
        
        if load_path:
            # Load pretrained models from the specified path
            self._load_models(load_path)
        else:
            # Ensure training parameters are provided
            if X_train is None or Y_train is None or n_fidelities is None:
                raise ValueError("X_train, Y_train, and n_fidelities must be provided for training.")
            
            # Store training-only parameters
            self.optimization_restarts = optimization_restarts

            # Train new models
            self._train_models(X_train, Y_train, n_fidelities, ARD_last_fidelity, parallel, num_processes)
    
    def _load_models(self, path: str) -> None:
        """
        Load pre-trained models from the specified path.
        """
        models = []
        i = 0
        while True:
            try:
                # Load models for each bin
                m1 = GPy.models.GPRegression.load_model(os.path.join(path,f"bin{i:03d}_m1.zip"))
                m2 = GPy.models.GPRegression.load_model(os.path.join(path,f"bin{i:03d}_m2.zip"))
                m3 = GPy.models.GPRegression.load_model(os.path.join(path,f"bin{i:03d}_m3.zip"))
                models.append([m1, m2, m3])
                i += 1
            except FileNotFoundError:
                # Stop when no more models are found
                break

        if not models:
            raise ValueError(f"No models found in the path: {path}")
        
        self.models = models
        
    def _train_models(
        self,
        X_train: List[np.ndarray],
        Y_train: List[np.ndarray],
        n_fidelities: int,
        ARD_last_fidelity: bool,
        parallel: bool,
        num_processes: Optional[int],
    ) -> None:

        # a list of GP emulators
        models: List = []

        D1 = X_train[0]
        D2 = X_train[1]
        D3 = X_train[2]

        z1 = Y_train[0]
        z2 = Y_train[1]
        z3 = Y_train[2]

        # loop through k bins
        for i in range(z3.shape[1]):
            this_z1 = z1[:, [i]]
            this_z2 = z2[:, [i]]
            this_z3 = z3[:, [i]]

            dim = D3.shape[1]
            Nts = D3.shape[0]

            active_dimensions = np.arange(0,dim)

            ''' M1 : Train LF model 1 '''
            k1 = GPy.kern.RBF(dim, ARD = True)
            m1 = GPy.models.GPRegression(X=D1, Y=this_z1, kernel=k1)

            m1[".*Gaussian_noise"] = m1.Y.var()*0.01
            m1[".*Gaussian_noise"].fix()
            m1.optimize(max_iters = 500)
            m1[".*Gaussian_noise"].unfix()
            m1[".*Gaussian_noise"].constrain_positive()
            m1.optimize_restarts(self.optimization_restarts, optimizer = "bfgs",  max_iters = 100, parallel=parallel,num_processes=num_processes)

            mu1, v1 = m1.predict(D3)

            ''' M2 : Train LF model 2 '''
            k2 = GPy.kern.RBF(dim, ARD = True)
            m2 = GPy.models.GPRegression(X=D2, Y=this_z2, kernel=k2)

            m2[".*Gaussian_noise"] = m2.Y.var()*0.01
            m2[".*Gaussian_noise"].fix()
            m2.optimize(max_iters = 500)
            m2[".*Gaussian_noise"].unfix()
            m2[".*Gaussian_noise"].constrain_positive()
            m2.optimize_restarts(self.optimization_restarts, optimizer = "bfgs",  max_iters = 100, parallel=parallel, num_processes=num_processes)

            mu2, v2 = m2.predict(D3)

            ''' M3 : Train HF model 3 '''
            XX = np.hstack((D3, mu1, mu2))

            # [jibancat] This part has re-written by me since for many cases the optimization
            #            failed and the loop stuck.
            def train_m3(tries: int = 0):

                while True:
                    try:
                        k3 = (
                                GPy.kern.Linear(2,active_dims=[dim,dim+1])+
                                GPy.kern.RBF(1, active_dims = [dim])*
                                GPy.kern.RBF(1, active_dims = [dim+1])
                            )*  GPy.kern.RBF(dim, active_dims = active_dimensions, ARD = ARD_last_fidelity) \
                            +   GPy.kern.RBF(dim, active_dims = active_dimensions, ARD = ARD_last_fidelity)

                        m3 = GPy.models.GPRegression(X=XX, Y=this_z3, kernel=k3)

                        m3[".*Gaussian_noise"] = m3.Y.var()*0.01
                        m3[".*Gaussian_noise"].fix()
                        m3.optimize(max_iters = 500)
                        m3[".*Gaussian_noise"].unfix()
                        m3[".*Gaussian_noise"].constrain_positive()
                        m3.optimize_restarts(self.optimization_restarts, optimizer = "bfgs",  max_iters = 100, parallel=parallel, num_processes=num_processes)
                        
                        return m3

                    except np.linalg.LinAlgError as e:
                        if tries <= 10:

                            print(e, "re-try")
                            return train_m3(tries + 1)
                        else:
                            print("Hit the limit!")

            m3 = train_m3(0)

            models.append([m1, m2, m3])

        self.models = models

    def save(self, save_dir: str) -> None: 
        """
        Save the models to individual files.
        """
        for i, model in enumerate(self.models):  # loop through k bins
            m1, m2, m3 = model
            m1.save_model(os.path.join(save_dir,f"bin{i:03d}_m1"))
            m2.save_model(os.path.join(save_dir,f"bin{i:03d}_m2"))
            m3.save_model(os.path.join(save_dir,f"bin{i:03d}_m3"))

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and variance for fidelity specified by last column of X.
        Note that we predict from gp from each k bin.

        :param X: point(s) at which to predict
        :return: predicted P(all k bins) (mean, variance) at X
        """
        means = np.full((X.shape[0], len(self.models)), fill_value=np.nan)
        variances = np.full((X.shape[0], len(self.models)), fill_value=np.nan)

        for i,model in enumerate(self.models):
            print(f"Number of CPU cores: {multiprocessing.cpu_count()}")
            process = psutil.Process()
            print(f"Active child processes: {len(process.children())}")
            print(f"NumPy max threads: {np.__config__.show()}")  # Show backend info
            print(f"Active threads: {threading.active_count()}")  # Count active threads
            m1, m2, m3 = model

            ''' Compute posterior mean and variance for level 3 evaluated at the test points '''
            nsamples = self.n_samples
            ntest = X.shape[0]

            # sample M1 at Xtest
            mu0, C0 = m1.predict(X, full_cov=True)
            Z1 = np.random.multivariate_normal(mu0.flatten(),C0,nsamples)

            # sample M2 at Xtest
            mu0, C0 = m2.predict(X, full_cov=True)
            Z2 = np.random.multivariate_normal(mu0.flatten(),C0,nsamples)

            # push samples through M3
            tmp_m = np.zeros((nsamples,ntest))
            tmp_v = np.zeros((nsamples,ntest))
            for j in range(0,nsamples):
                mu, v = m3.predict(np.hstack((X, Z1[j,:][:,None], Z2[j,:][:,None])))
                tmp_m[j,:] = mu.flatten()
                tmp_v[j,:] = v.flatten()

            # get M3 posterior mean and variance at Xtest
            mu_final = np.mean(tmp_m, axis = 0)
            v_final = np.mean(tmp_v, axis = 0) + np.var(tmp_m, axis = 0)
            y_pred = mu_final[:,None]
            var_pred = np.abs(v_final[:,None])

            means[:, i] = y_pred[:, 0]
            variances[:, i] = var_pred[:, 0]

        return means, variances

    def to_dict(self) -> Dict:
        """
        Save hyperparameters into a dict.
        """
        param_dict = {}

        for i,model in enumerate(self.models):
            this_param_dict = {}

            # append low-fidelity nodes
            for j, m in enumerate(model[:-1]):
                this_param_dict["low_fidelity_{}".format(j)] = m.kern.to_dict()
            # append high-fidelity node
            m = model[-1]
            this_param_dict["high_fidelity"] = m.kern.to_dict()

            param_dict["bin_{}".format(i)] = this_param_dict

        return param_dict


class SingleBinNonLinearGP:
    """
    A thin wrapper around NonLinearMultiFidelityModel. It models each k input as
    an independent GP.

    :param X_train:  (n_fidelities, n_points, n_dims) list of parameter vectors.
    :param Y_train:  (n_fidelities, n_points, k modes) list of matter power spectra.
    :param n_fidelities: number of fidelities stored in the list.
    :param n_samples: Number of samples to use to do quasi-Monte-Carlo integration at each fidelity.
    :param optimization_restarts: number of optimization restarts you want in GPy.
    :param ARD_last_fidelity: whether to apply ARD for the last (highest) fidelity.
        Default, False.
    """

    def __init__(
        self,
        X_train: List[np.ndarray],
        Y_train: List[np.ndarray],
        n_fidelities: int,
        n_samples: int = 500,
        optimization_restarts: int = 30,
        turn_off_bias: bool = False,
        ARD_last_fidelity: bool = False
    ):
        # a list of GP emulators
        models: List = []

        self.n_fidelities = len(X_train)

        # convert into X,Y for MultiOutputGP
        # not normalize due to no improvements
        X, Y = convert_xy_lists_to_arrays(X_train, Y_train)

        # linear multi-fidelity setup
        if X.ndim != 2:
            raise ValueError("X should be 2d")

        if Y.ndim != 2:
            raise ValueError("Y should be 2d")

        if np.any(X[:, -1] >= n_fidelities):
            raise ValueError(
                "One or more points has a higher fidelity index than number of fidelities"
            )

        # make a GP on each P(k) bin
        for i in range(Y.shape[1]):
            # make GP non linear kernel
            base_kernel_1 = GPy.kern.RBF
            kernels = make_non_linear_kernels(
                base_kernel_1, n_fidelities, X.shape[1] - 1, ARD=True, n_output_dim=1,
                turn_off_bias=turn_off_bias, ARD_last_fidelity=ARD_last_fidelity,
            )  # -1 for the multi-fidelity labels

            model = NonLinearMultiFidelityModel(
                X,
                Y[:, [i]],
                n_fidelities,
                kernels=kernels,
                verbose=True,
                n_samples=n_samples,
                optimization_restarts=optimization_restarts,
            )

            models.append(model)

        self.models = models

        self.name = "nargp"

    def optimize(self, parallel: bool = False) -> None:
        """
        Optimize GP on each bin of the power spectrum.
        """

        _log.info("\n--- Optimization: ---\n".format(self.name))

        for i,gp in enumerate(self.models):
            _log.info("\n [Info] Optimizing {} bin ... \n".format(i))

            for m in gp.models:
                m.Gaussian_noise.variance.fix(1e-6)
            
            gp.optimize(parallel=parallel)

            for m in gp.models:
                m.Gaussian_noise.variance.unfix()
            
            gp.optimize(parallel=parallel)


    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and variance for fidelity specified by last column of X.
        Note that we predict from gp from each k bin.

        :param X: point(s) at which to predict
        :return: predicted P(all k bins) (mean, variance) at X
        """
        means = np.full((X.shape[0], len(self.models)), fill_value=np.nan)
        variances = np.full((X.shape[0], len(self.models)), fill_value=np.nan)

        for i,model in enumerate(self.models):
            mean, variance = model.predict(X)

            means[:, i] = mean[:, 0]
            variances[:, i] = variance[:, 0]

        return means, variances

    def to_dict(self) -> Dict:
        """
        Save hyperparameters into a dict
        """
        param_dict = {}

        for i,model in enumerate(self.models):
            this_param_dict = {}
            # append a list of kernel paramaters
            for j, m in enumerate(model.models):
                this_param_dict["fidelity_{}".format(j)] = m.kern.to_dict()

            param_dict["bin_{}".format(i)] = this_param_dict

        return param_dict

class SingleBinDeepGP:
    """
    A thin wrapper around MultiFidelityDeepGP. Help to handle inputs.
    
    To run this model, you need additional packages:
    - tensorflow==1.8
    - gpflow==1.3 (Note: it said 1.1.1 on the website, but the code actually only works
        for 1.3 version)
    - pip install git+https://github.com/ICL-SML/Doubly-Stochastic-DGP.git

    Warning: this deepGP code hasn't fully tested on the matter power spectrum we have
        here. Be aware you might need more HR samples for train it.

    :param X_train:  (n_fidelities, n_points, n_dims) list of parameter vectors.
    :param Y_train:  (n_fidelities, n_points, k modes) list of matter power spectra.
    :param n_fidelities: number of fidelities stored in the list.
    """

    def __init__(
        self,
        X_train: List[np.ndarray],
        Y_train: List[np.ndarray],
        n_fidelities: int,
    ):
        # DGP model
        from emukit.examples.multi_fidelity_dgp.multi_fidelity_deep_gp import MultiFidelityDeepGP

        # a list of GP emulators
        models: List = []

        self.n_fidelities = len(X_train)

        # make a GP on each P(k) bin
        for i in range(Y_train[0].shape[1]):

            model = MultiFidelityDeepGP(X_train, [power[:, [i]] for power in Y_train])

            models.append(model)

        self.models = models

        self.name = "dgp"

    def optimize(self) -> None:
        """
        Optimize GP on each bin of the power spectrum.
        """

        _log.info("\n--- Optimization ---\n".format(self.name))

        for i,gp in enumerate(self.models):
            _log.info("\n [Info] Optimizing {} bin ... \n".format(i))
            gp.optimize()

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and variance for fidelity specified by last column of X.
        Note that we predict from gp from each k bin.

        :param X: point(s) at which to predict
        :return: predicted P(all k bins) (mean, variance) at X
        """
        means = np.full((X.shape[0], len(self.models)), fill_value=np.nan)
        variances = np.full((X.shape[0], len(self.models)), fill_value=np.nan)

        for i,model in enumerate(self.models):
            mean, variance = model.predict(X)

            means[:, i] = mean[:, 0]
            variances[:, i] = variance[:, 0]

        return means, variances

    def to_dict(self) -> Dict:
        """
        Save hyperparameters into a dict
        """
        NotImplementedError

def _map_params_to_unit_cube(
    params: np.ndarray, param_limits: np.ndarray
) -> np.ndarray:
    """
    Map the parameters onto a unit cube so that all the variations are
    similar in magnitude.
    
    :param params: (n_points, n_dims) parameter vectors
    :param param_limits: (n_dim, 2) param_limits is a list 
        of parameter limits.
    :return: params_cube, (n_points, n_dims) parameter vectors 
        in a unit cube.
    """
    nparams = np.shape(params)[1]
    params_cube = map_to_unit_cube_list(params, param_limits)
    assert params_cube.shape[1] == nparams

    return params_cube
