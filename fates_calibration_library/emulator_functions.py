"""Methods to assist with  model emulation"""

import os
import json
import gpflow
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import root_mean_squared_error, r2_score
from SALib.sample import fast_sampler, saltelli
from SALib.analyze import fast, sobol
import fates_calibration_library.plotting_functions as plotting


def load_lhc_metadata(ensemble_config: dict[str, str]) -> tuple[list[str], pd.DataFrame]:
    """Loads information about Latin Hypercube and parameters for emulation

    Args:
        ensemble_config (dict[str, str]): config dictionary with information about LH key, etc.

    Returns:
        tuple[list[str], pd.DataFrame]: list of parameter names, pandas DataFrame with 
            the normalized default values of the parameters
    """

    # load Latin Hypercube key to get parameter names
    lhc_key = pd.read_csv(ensemble_config["lhc_key_file"], index_col=[0])
    lhc_key = lhc_key.drop(columns=["ensemble"])
    param_names = list(lhc_key.columns)
    
    # get normalized default parameter values
    default_norm = pd.read_csv(ensemble_config["default_norm"], index_col=[0])

    return param_names, default_norm


def build_kernel_dict(num_params: int) -> dict[int, gpflow.kernels.Kernel]:
    """Builds a dictionary of composite Gaussian Process kernel

    Args:
        num_params (int): number of parameters in ensemble

    Returns:
        dict[int, gpflow.kernels.Kernel]: dictionary of GP kernels
    """

    # adds white nose to GP - helps with numerical stability
    kernel_noise = gpflow.kernels.White(variance=1e-3)

    # Matern kernel with v=3/2 (moderate smoothness)
    # lengthscales=10 - smoother across inputs
    # variance=10 - allows broad function output range
    kernel_matern32 = gpflow.kernels.Matern32(
        active_dims=range(num_params), variance=10, lengthscales=np.tile(10, num_params)
    )

    # Matern kernel with v=5/2 (smoother than Matern32)
    kernel_matern52 = gpflow.kernels.Matern52(
        active_dims=range(num_params), variance=1, lengthscales=np.tile(1, num_params)
    )

    # Constant offset (bias term) - output is invariant to input
    kernel_bias = gpflow.kernels.Bias(active_dims=range(num_params))

    # linear relationship between inputs and outputs
    # each input has its own variance
    # encourages linear trends in data
    kernel_linear = gpflow.kernels.Linear(active_dims=range(num_params), variance=1.0)

    # polynomial relationships
    # adds higher-order interactions, requires specification of degree (defaults 2)
    kernel_poly = gpflow.kernels.Polynomial(
        active_dims=range(num_params), variance=[1.0] * num_params
    )

    # Radial Bias Function kernel (squared exponential)
    # Very smooth - assumes function is infinitely differentiable
    kernel_RBF = gpflow.kernels.RBF(
        active_dims=range(num_params), lengthscales=np.tile(1, num_params)
    )

    kernel_matern32_lowvar = gpflow.kernels.Matern32(
        active_dims=range(num_params), variance=1, lengthscales=np.tile(1, num_params)
    )
    kernel_dict = {
        0: kernel_RBF + kernel_linear + kernel_noise,
        1: kernel_RBF + kernel_linear + kernel_noise + kernel_bias,
        2: kernel_poly + kernel_linear + kernel_noise,
        3: kernel_RBF + kernel_linear + kernel_noise + kernel_bias + kernel_poly,
        4: kernel_matern32,
        5: kernel_matern32 * kernel_linear + kernel_noise,
        6: kernel_linear * kernel_RBF + kernel_matern32 + kernel_noise,
        7: kernel_linear + kernel_matern32_lowvar,
        8: kernel_matern52,
    }

    return kernel_dict


def split_dataset(
    var: xr.DataArray, param_key: pd.DataFrame, n_test: int, default_ind: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits dataset and parameter key into training and testing

    Args:
        var (xr.DataArray): input variable
        param_key (pd.DataFrame): parameter key
        n_test (int): number of testing values
        default_ind (int, optional): Index of default simulation. Defaults to 0.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: testing and training datasets
    """

    # target variable (excluding default [0])
    if default_ind is not None:
        Y_var = np.delete(var.values, default_ind)
    else:
        Y_var = var.values

    # test and training parameters
    X_test, X_train = param_key.iloc[:n_test].copy(), param_key.iloc[n_test:].copy()

    # test and training output
    y_test, y_train = Y_var[:n_test], Y_var[n_test:]

    return X_test, X_train, y_test, y_train


def train_emulator(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: gpflow.kernels.Kernel,
    maxiter: int = 1000,
):
    """Train a GP emulator on provided data

    Args:
        X_train (np.ndarray): Training input of shape (n_samples, n_features)
        y_train (np.ndarray): Training target of shape (n_samples,)
        kernel (gpflow.kernels.Kernel): GPflow kernel to use
        maxiter (int, optional): maximum iterations to optimize. Defaults to 1000.
    """

    # ensure shapes are correct
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)

    # train GP model
    model = gpflow.models.GPR(
        data=(np.asarray(X_train, dtype=np.float64), y_train),
        kernel=kernel,
        mean_function=None,
    )

    # optimize model
    optimizer = gpflow.optimizers.Scipy()
    opt_logs = optimizer.minimize(
        model.training_loss, model.trainable_variables, options=dict(maxiter=maxiter)
    )

    return opt_logs, model


def test_emulator(model, X_test, y_test, variable_name):

    # predict test points with emulator
    y_pred, y_pred_var = model.predict_y(np.asarray(X_test, dtype=np.float64))
    y_pred_sd = np.sqrt(y_pred_var.numpy().flatten())

    # ensure shapes are the same
    y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)

    df = {
        f"{variable_name}_test": y_test,
        f"{variable_name}_pred": y_pred.numpy().flatten(),
        f"{variable_name}_sd": y_pred_sd,
    }

    return pd.DataFrame(df)


def extract_kernel_hyperparams(kernel: gpflow.kernels.Kernel) -> dict:
    """Extracts hyperparameters from a GP kernel

    Args:
        kernel (gpflow.kernels.Kernel): input kernel

    Returns:
        dict: dictionary of hyperparameter names and values
    """
    params = {}
    for param in kernel.trainable_parameters:
        params[param.name] = param.numpy()
    return params


def train_val_save(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    kernel: gpflow.kernels.Kernel,
    variable: str,
    maxiter: int = 1000,
    out_file: str = None,
    save_dir: str = None,
) -> tuple[float, float, float]:
    """Train a GP emulator on provided data, evaluate R², optionally save model and plot predictions.


    Args:
        X_train (np.ndarray): Training input of shape (n_samples, n_features)
        X_test (np.ndarray): Testing input of shape (n_samples, n_features)
        y_train (np.ndarray): Training target of shape (n_samples,)
        y_test (np.ndarray): Testing target of shape (n_samples,)
        kernel (gpflow.kernels.Kernel): GPflow kernel to use
        variable (str): Variable name we are emulating.
        maxiter (int, optional): Max optimization iterations. Defaults to 1000.
        out_file (str, optional): Path to save validation plot (optional). Defaults to None.
        save_dir (str, optional): Directory to save the trained model. Defaults to None.


    Returns:
        tuple[float, float, float]:
        r2: float
            R² score on test set
        rmse: float
            RMSE of test set
        emulator_std: float
            Average standard deviation of emulator predictions
    """

    # train emulator
    _, model = train_emulator(X_train, y_train, kernel, maxiter)

    # test emulator
    test_df = test_emulator(model, X_test, y_test, variable)

    # compute r2, rmse, and emulator sd
    r2 = r2_score(test_df[f"{variable}_test"], test_df[f"{variable}_pred"])
    rmse = root_mean_squared_error(
        test_df[f"{variable}_test"], test_df[f"{variable}_pred"]
    )
    em_sd = np.mean(test_df[f"{variable}_sd"].values)

    if save_dir:

        # save hyperparameters
        hyperparams = extract_kernel_hyperparams(kernel)

        input_dim = np.shape(X_train)[1]

        @tf.function(
            input_signature=[tf.TensorSpec(shape=[None, input_dim], dtype=tf.float64)]
        )
        def serving_fn(X):
            y_mean, y_var = model.predict_y(X)
            return {"mean": y_mean, "variance": y_var}

        print(f"Saving model to: {save_dir}...")
        os.makedirs(save_dir, exist_ok=True)
        tf.saved_model.save(model, save_dir, signatures={"serving_default": serving_fn})

        with open(os.path.join(save_dir, "kernel_hyperparameters.json"), "w") as f:
            json.dump(
                {
                    k: v.tolist() if isinstance(v, np.ndarray) else float(v)
                    for k, v in hyperparams.items()
                },
                f,
                indent=2,
            )

    if out_file:
        print(f"Saving validation plot to: {out_file}")
        plotting.plot_emulator_validation(test_df, variable, r2, rmse)
        plt.savefig(out_file)

    return r2, rmse, em_sd


def select_kernel(
    kernel_dict: dict[int, gpflow.kernels.Kernel],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    variable: str,
    maxiter: int = 1000,
    alpha: float = 0.8,
    verbose: bool = False,
) -> gpflow.kernels.Kernel:
    """Evaluate multiple kernels and select the best one based on R² and uncertainty.

    Args:
        kernel_dict (dict[int, gpflow.kernels.Kernel]): Dictionary of candidate kernels
        X_train (np.ndarray): Training input of shape (n_samples, n_features)
        X_test (np.ndarray): Testing input of shape (n_samples, n_features)
        y_train (np.ndarray): Training target of shape (n_samples,)
        y_test (np.ndarray): Testing target of shape (n_samples,)
        variable (str): Variable name
        maxiter (int, optional): maximum iterations for optimization. Defaults to 1000.0
        alpha (float, optional): Weight given to RMSE (vs. 1 - uncertainty); between 0 and 1. Defaults to 0.8.
        verbose (bool, optional): If True, print scores for each kernel. Defaults to False.

    Returns:
        gpflow.kernels.Kernel: The kernel that achieves the best weighted score
    """

    results = []
    for k, kernel in kernel_dict.items():
        r2, rmse, sd = train_val_save(
            X_train, X_test, y_train, y_test, kernel, variable, maxiter
        )
        results.append((k, r2, rmse, sd))
        if verbose:
            print(f"Kernel {k}: R² = {r2:.4f}, RMSE = {rmse:.4f}, mean std = {sd:.4f}")

    # unpack
    kernel_ids, r2_scores, rmses, sds = zip(*results)
    r2_scores = np.array(r2_scores)
    rmses = np.array(rmses)
    sds = np.array(sds)

    # normalize to [0, 1]
    r2_norm = (r2 - np.min(r2_scores)) / (np.max(r2_scores) - np.min(r2_scores))
    sd_norm = 1 - (sds - np.min(sds)) / (np.ptp(sds) + 1e-8)  # lower sd is better
    rmse_norm = 1 - (rmses - np.min(rmses)) / (
        np.ptp(rmses) + 1e-8
    )  # lower RMSE is better

    # combine into weighted score
    w_rmse = alpha
    w_other = (1.0 - alpha) / 2
    score = w_rmse * rmse_norm + w_other * r2_norm + w_other * sd_norm

    # choose best score
    best_idx = np.argmax(score)
    best_kernel_id = kernel_ids[best_idx]
    best_kernel = kernel_dict[best_kernel_id]

    if verbose:
        print(
            f"\nSelected kernel {best_kernel_id} with combined score = {score[best_idx]:.4f}"
        )

    return best_kernel


def sensitivity_analysis(emulator, param_names, update_pars=None, sample_type="Sobol"):

    # create a fast sample for fourier sensitivity
    problem = {
        "names": param_names,
        "num_vars": len(param_names),
        "bounds": [[0, 1]] * len(param_names),
    }

    sample = create_sample(param_names, problem, update_pars, sample_type)

    # fourier amplitude sensitivity test w/emulator
    Y_emulated, _ = emulator(sample)

    if sample_type == "Sobol":
        results = sobol.analyze(
            problem, Y_emulated.numpy().flatten(), calc_second_order=False
        )
    else:
        results = fast.analyze(
            problem,
            Y_emulated.numpy().flatten(),
            M=10,
            num_resamples=100,
            conf_level=0.95,
        )

    df = pd.DataFrame(
        {
            "parameter": problem["names"],
            "S1": results["S1"],
            "S1_conf": results["S1_conf"],
            "ST": results["ST"],
            "ST_conf": results["ST_conf"],
        }
    )

    return df.sort_values("ST", ascending=False)


def create_sample(
    param_names: list[str], problem, update_pars=None, sample_type="Sobol"
) -> np.ndarray:
    """Create a sample for conducting sensitivity analyses with an emulator

    Args:
        param_names (list[str]): _description_
        problem (_type_): _description_
        update_pars (_type_, optional): _description_. Defaults to None.
        sample_type (str, optional): _description_. Defaults to "Sobol".

    Raises:
        ValueError: _description_

    Returns:
        np.ndarray: _description_
    """

    if sample_type == "Sobol":
        sample = saltelli.sample(problem, 4000, calc_second_order=False)

    elif sample_type == "FAST":
        sample = fast_sampler.sample(problem, 1000, M=10, seed=None)
    else:
        raise ValueError(
            f"sample_type must be either 'Sobol' or 'FAST', not {sample_type}"
        )

    if update_pars is not None:
        sample = update_sample(sample, update_pars, param_names)

    return sample


def update_sample(sample, update_pars, param_names):

    sample_update = sample.copy()

    cols_to_update = update_pars.columns
    indices = [np.where(np.array(param_names) == par)[0][0] for par in cols_to_update]

    new_values = update_pars.values[0]  # assuming update_pars has 1 row

    sample_update[:, indices] = new_values

    return sample_update


def chunked_emulation(emulator, large_sample, chunk_size=100000):
    results_pred = []
    results_var = []
    # split the sample into smaller blocks
    for i in range(0, len(large_sample), chunk_size):
        chunk = large_sample[i : i + chunk_size]
        # perform emulation on just this block
        y_pred, y_var = emulator(chunk)
        results_pred.append(y_pred.numpy().flatten())
        results_var.append(y_var.numpy().flatten())

    return np.concatenate(results_pred), np.concatenate(results_var)


def get_full_array(X0, fixed_indices, X_default_all):
    X0_full = np.zeros(len(X_default_all))
    j = 0
    for i in range(len(X_default_all)):
        if i in fixed_indices:
            X0_full[i] = X_default_all[i]
        else:
            X0_full[i] = X0[j]
            j = j + 1

    return X0_full
