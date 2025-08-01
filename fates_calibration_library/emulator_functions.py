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
from SALib.sample import fast_sampler
from SALib.analyze import fast
from SALib.sample import saltelli
from SALib.analyze import sobol
from matplotlib.patches import Rectangle
from scipy.optimize import minimize
from fates_calibration_library.TFClass import TFEmulator

def create_sample(param_names, problem, update_vars=None, sample_type='Sobol'):
    
    
    if sample_type == 'Sobol':
        sample = saltelli.sample(problem, 4000, calc_second_order=False)
    
    elif sample_type == 'FAST':
        sample = fast_sampler.sample(problem, 1000, M=10, seed=None)
    else:
        raise ValueError(f"sample_type must be either 'Sobol' or 'FAST', not {sample_type}")
        
    if update_vars is not None:
        sample = update_sample(sample, update_vars, param_names)
        
    return sample

def sensitivity_analysis(emulator, param_names, update_vars=None, sample_type='Sobol'):
    
    # create a fast sample for fourier sensitivity
    problem = {
        'names': param_names,
        'num_vars': len(param_names),
        'bounds': [[0, 1]] * len(param_names),
    }

    sample = create_sample(param_names, problem, update_vars, sample_type)
    
    # fourier amplitude sensitivity test w/emulator
    Y_emulated, _ = emulator(sample)
    
    if sample_type == 'Sobol':
        results = sobol.analyze(problem, Y_emulated.numpy().flatten(), 
                                      calc_second_order=False)
    else:
        results = fast.analyze(problem, Y_emulated.numpy().flatten(), 
                               M=10, num_resamples=100, conf_level=0.95)
        
    df = pd.DataFrame({
        'parameter': problem['names'],
        'S1': results['S1'],
        'S1_conf': results['S1_conf'],
        'ST': results['ST'],
        'ST_conf': results['ST_conf'],
    })
    
    return df.sort_values("ST", ascending=False)

def plot_global_sensitivity(sens_df, variable, type='Sobol'):

    plt.figure(num=None, figsize=(12, 6), dpi=100, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': 12})

    ax = plt.subplot(1, 1, 1)
    ax.bar(sens_df.parameter, sens_df['ST'], color='lightgrey', label='interactions')
    ax.bar(sens_df.parameter, sens_df['S1'], color='darkolivegreen', label='main effects')
    ax.tick_params(axis='x', labelrotation=90)
    plt.legend(loc='upper right')
    plt.ylabel('Proportion of total \n emulated variance')
    plt.title(f'Global Sensitivity Analysis ({type}) - {variable}')
    plt.tight_layout()
    
def plot_oaat_sens(param_names, emulator, default_vals):

    num_params = len(param_names)

    # hold all parameters at median value 
    n = 50
    unif = pd.concat([pd.DataFrame(np.tile(0.5, n))]*num_params, axis=1)
    unif.columns = param_names
    
    s = np.linspace(0, 1, n)
    param = np.array([])
    oaats = np.array([])
    vars = np.array([])
    samps = np.array([])
    
    sample = unif
    
    plt.figure(figsize=[18, 16])
    for i, p in enumerate(param_names):
        
        # save old value and update to be 0-1
        save = sample[p]
        sample[p] = s

        default_val = default_vals[p].values
        
        # oaat prediction
        oaat, v = emulator(sample.values)
        oaat = oaat.numpy().flatten()
        v = v.numpy().flatten()
        ax = plt.subplot(7, 5, i + 1)
        ax.fill_between(s, oaat - 2.0*v**0.5, oaat + 2.0*v**0.5, color='peru',
                        alpha=0.4)  # shade two standard deviations
        ax.plot(s, oaat, c='k')
        ax.axvline(x=default_val, color='r', linestyle='--')
        ax.set_xlabel(p)

        oaats = np.append(oaats, oaat)
        vars = np.append(vars, v)
        param = np.append(param, np.repeat(p, n))
        samps = np.append(samps, s)
        
        # set column back to before
        sample[p] = save 

    plt.tight_layout()
    df = pd.DataFrame({'sample': samps,
      'predict': oaats,
      'variance': vars,
      'parameter': param})
    dataf = pd.DataFrame(df)
    return dataf

def plot_emulated_sample(pred_sampled, obs_mean, obs_sd, pft_id, var, units,
                         ensemble_vals=None, label_1='Emulated', label_2='FATES'):
    
    plt.figure(figsize=(7, 5))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xticks(fontsize=11)

    my_hist, _ = np.histogram(pred_sampled, bins=40)
    
    plt.hist(pred_sampled, fc="darkgray", bins=40, label=label_1, alpha=0.75)
    if ensemble_vals is not None:
        plt.hist(ensemble_vals, fc="dodgerblue", bins=40, label=label_2, alpha=0.5)
        
    maxv = max(my_hist.max(), np.histogram(ensemble_vals, bins=40)[0].max() if ensemble_vals is not None else 0)
    
    patch = Rectangle((obs_mean - obs_sd, 0), 2*obs_sd, maxv,
                        facecolor='red', alpha=0.4, label="Observed ± SD")
    ax.add_patch(patch)
    ax.axvline(x=obs_mean, ymin=0.0, ymax=maxv, color='r', label="Observed Mean")
    
    plt.xlabel(f"Emulated {pft_id} Annual {var} ({units})", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    
    if ensemble_vals is not None:
        plt.legend()
        
    plt.tight_layout()
    
def get_proportion_implausible(implaus, tol=3.0):
    return len(implaus[implaus <= tol])/len(implaus)*100.0

def plot_implausibility_histogram(implaus, tol=3.0):
    plt.figure(figsize=(7, 5))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xticks(fontsize=11)

    counts, bins = np.histogram(implaus, bins=40)
    for i in range(len(bins) - 1):
        bin_left = bins[i]
        bin_right = bins[i + 1]
        bin_center = (bin_left + bin_right) / 2
        color = "darkgray" if bin_center > tol else "dodgerblue"
        ax.bar(bin_left, counts[i], width=bin_right - bin_left, color=color, align='edge')

    custom_legend = [
        plt.Line2D([0], [0], color='dodgerblue', lw=6, label='plausible'),
        plt.Line2D([0], [0], color='darkgray', lw=6, label='implausible')
    ]
    ax.legend(handles=custom_legend)
    plt.xlabel("Implausibility Score", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()

def update_sample(sample, update_vars, param_names):
    
    sample_update = sample.copy()
    pars_to_update = update_vars.columns
    for i in range(len(sample_update)):
        for par in pars_to_update:
            j = np.where(param_names == par)
            val = update_vars[par].values[0]
            sample_update[i][j] = val
    
    return sample_update

def test_emulator(model, X_test, y_test, variable_name):

    # predict test points with emulator
    y_pred, y_pred_var = model.predict_y(np.asarray(X_test, dtype=np.float64))
    y_pred_sd = np.sqrt(y_pred_var.numpy().flatten())
    
    # ensure shapes are the same
    y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)
    
    df = {f'{variable_name}_test': y_test,
          f'{variable_name}_pred': y_pred.numpy().flatten(),
          f'{variable_name}_sd': y_pred_sd}

    return pd.DataFrame(df)

def split_dataset(var: xr.DataArray, param_key: pd.DataFrame, n_test: int, 
                  default_ind: int=0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

def build_kernel_dict(num_params: int) -> dict:
    """Builds a dictionary of composite Gaussian Process kernel

    Args:
        num_params (int): number of parameters in ensemble

    Returns:
        dict: dictionary of GP kernels
    """
    
    # adds white nose to GP - helps with numerical stability
    kernel_noise = gpflow.kernels.White(variance=1e-3)
    
    # Matern kernel with v=3/2 (moderate smoothness)
    # lengthscales=10 - smoother across inputs
    # variance=10 - allows broad function output range
    kernel_matern32 = gpflow.kernels.Matern32(active_dims=range(num_params), variance=10, 
                                              lengthscales = np.tile(10, num_params))
    
    # Matern kernel with v=5/2 (smoother than Matern32)
    kernel_matern52 = gpflow.kernels.Matern52(active_dims=range(num_params), variance=1, 
                                              lengthscales=np.tile(1,num_params))
    
    # Constant offset (bias term) - output is invariant to input
    kernel_bias = gpflow.kernels.Bias(active_dims = range(num_params))
    
    # linear relationship between inputs and outputs
    # each input has its own variance
    # encourages linear trends in data
    kernel_linear = gpflow.kernels.Linear(active_dims=range(num_params), 
                                          variance=1.0)
    
    # polynomial relationships
    # adds higher-order interactions, requires specification of degree (defaults 2)
    kernel_poly = gpflow.kernels.Polynomial(active_dims = range(num_params), 
                                            variance=[1.]*num_params)
    
    # Radial Bias Function kernel (squared exponential)
    # Very smooth - assumes function is infinitely differentiable
    kernel_RBF = gpflow.kernels.RBF(active_dims=range(num_params), 
                                    lengthscales=np.tile(1, num_params))
    
    kernel_matern32_lowvar = gpflow.kernels.Matern32(active_dims=range(num_params),
                                            variance=1,
                                            lengthscales=np.tile(1, num_params))
    kernel_dict = {
        0: kernel_RBF + kernel_linear + kernel_noise,
        1: kernel_RBF + kernel_linear + kernel_noise + kernel_bias,
        2: kernel_poly + kernel_linear + kernel_noise,
        3: kernel_RBF + kernel_linear + kernel_noise + kernel_bias + kernel_poly,
        4: kernel_matern32,
        5: kernel_matern32*kernel_linear + kernel_noise,
        6: kernel_linear*kernel_RBF + kernel_matern32 + kernel_noise,
        7: kernel_linear + kernel_matern32_lowvar
    }
    
    return kernel_dict

def select_kernel(kernel_dict: dict[int, gpflow.kernels.Kernel], X_train: np.ndarray, 
                  X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                  variable: str, maxiter: int=1000, alpha: float=0.8, 
                  verbose: bool=False) -> gpflow.kernels.Kernel:
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
        r2, rmse, sd = train_val_save(X_train, X_test, y_train, y_test, kernel, variable, maxiter)
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
    sd_norm = 1 - (sds - np.min(sds)) / (np.ptp(sds) + 1e-8) # lower sd is better
    rmse_norm = 1 - (rmses - np.min(rmses)) / (np.ptp(rmses) + 1e-8) # lower RMSE is better

    # combine into weighted score
    w_rmse = alpha
    w_other = (1.0 - alpha) / 2
    score = w_rmse * rmse_norm + w_other * r2_norm + w_other * sd_norm
    
    # choose best score
    best_idx = np.argmax(score)
    best_kernel_id = kernel_ids[best_idx]
    best_kernel = kernel_dict[best_kernel_id]
    
    if verbose:
        print(f"\nSelected kernel {best_kernel_id} with combined score = {score[best_idx]:.4f}")
    
    return best_kernel

def train_emulator(X_train: np.ndarray, y_train: np.ndarray,
                   kernel: gpflow.kernels.Kernel, maxiter: int=1000):
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
    model = gpflow.models.GPR(data=(np.asarray(X_train, dtype=np.float64), y_train), kernel=kernel,
                            mean_function=None)
    
    # optimize model
    optimizer = gpflow.optimizers.Scipy()
    opt_logs = optimizer.minimize(model.training_loss, 
                                model.trainable_variables,
                                options=dict(maxiter=maxiter))
    
    return opt_logs, model
    
def train_val_save(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, 
                   y_test: np.ndarray, kernel: gpflow.kernels.Kernel, variable: str,
                   maxiter: int=1000, out_file: str=None, save_dir: str=None) -> tuple[float, float, float]:
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
    rmse = root_mean_squared_error(test_df[f"{variable}_test"], test_df[f"{variable}_pred"])
    em_sd = np.mean(test_df[f"{variable}_sd"].values)
    
    if save_dir:
    
        # save hyperparameters
        hyperparams = extract_kernel_hyperparams(kernel)
        
        input_dim = np.shape(X_train)[1]
        
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, input_dim], dtype=tf.float64)])
        def serving_fn(X):
            y_mean, y_var = model.predict_y(X)
            return {'mean': y_mean, 'variance': y_var}
        
        print(f'Saving model to: {save_dir}...')
        os.makedirs(save_dir, exist_ok=True)
        tf.saved_model.save(model, save_dir, signatures={"serving_default": serving_fn})
        
        with open(os.path.join(save_dir, "kernel_hyperparameters.json"), "w") as f:
            json.dump({k: v.tolist() if isinstance(v, np.ndarray) else float(v) for k, v in hyperparams.items()}, f, indent=2)

    if out_file:
        print(f"Saving validation plot to: {out_file}")
        plot_emulator_validation(test_df, variable, r2, rmse)
        plt.savefig(out_file)

    return r2, rmse, em_sd

def extract_kernel_hyperparams(kernel: gpflow.kernels.Kernel) -> dict:
    params = {}
    for param in kernel.trainable_parameters:
        params[param.name] = param.numpy()
    return params

def plot_emulator_validation(test_df, variable, r2, rmse):
    """Plots emulator testing set (actual vs. pred) with error bars

    Args:
        y_test (np.ndarray): Testing target of shape (n_samples,)
        y_pred (np.ndarray): Predicted target of shape (n_samples,)
        y_pred_sd (np.ndarray): Standard deviation of predictions
        r2 (float): R² score on test set
        RMSE (float): RMSE of test set
    """
    
    em_sd = np.mean(test_df[f"{variable}_sd"].values)
    
    plt.figure(figsize=(5, 5))
    plt.scatter(test_df[f"{variable}_test"], test_df[f"{variable}_pred"])
    plt.errorbar(test_df[f"{variable}_test"], test_df[f"{variable}_pred"], 
                 yerr=2*test_df[f"{variable}_sd"], fmt="o")
    plt.plot([min(test_df[f"{variable}_test"]),
              max(test_df[f"{variable}_test"])],
              [min(test_df[f"{variable}_test"]), 
               max(test_df[f"{variable}_test"])], linestyle='--', c='k')
    plt.text(0.02, 0.98, f'$R^2$ = {r2:.2f}', fontsize=10, transform=plt.gca().transAxes, va='top', ha='left')
    plt.text(0.02, 0.93, f'Emulator std ≈ {em_sd:.2f}', 
             fontsize=10, transform=plt.gca().transAxes, va='top', ha='left')
    plt.text(0.02, 0.88, f'RSME = {rmse:.3f}', 
             fontsize=10, transform=plt.gca().transAxes, va='top', ha='left')
    plt.xlabel(f'FATES mean annual mean {variable}')
    plt.ylabel(f'Emulated mean annual mean {variable}')
    plt.tight_layout()


def get_full_array(X0, fixed_indices, X_default_all):
    X0_full = np.zeros(len(X_default_all))
    j = 0
    for i in range(len(X_default_all)):
        if i in fixed_indices:
            X0_full[i] = X_default_all[i]
        else:
            X0_full[i] = X0[j]
            j = j+1

    return X0_full

def get_params_to_optimize(sens_df, param_names, num_params, sobol_threshold=0.01):
    
    sobol_indices = np.array([sens_df[sens_df.parameter == p]['ST'].sum() for p in param_names])
    optimize_mask = sobol_indices > sobol_threshold
    fixed_indices = np.where(np.logical_not(optimize_mask))[0]
    optimize_indices = np.arange(num_params)[optimize_mask]
    num_optimize = num_params - len(fixed_indices)

    return fixed_indices, optimize_indices, num_optimize

def misfit(X, emulator_array, targets, stdevs, fixed_indices, X_default_all, config):
    
    # loop over each emulator/target/stddev
    total_error = 0.0
    for emulator, target, stddev in zip(emulator_array, targets, stdevs):
        
        X_full = get_full_array(X, fixed_indices, X_default_all)
        y_pred, y_var = emulator(X_full.reshape(1,-1))
        
        # calculate loss
        z = config['loss_fn'](y_pred, target, stddev, y_var)
        total_error += z
        
    return total_error

def run_optimization(emulators, targets, sds, fixed_indices, optimize_pars, 
                     X_default_all, num_optimize, config, param_update_config):
    
    x0 = np.random.rand(num_optimize)
    bounds = [(0, 1)] * len(x0)
    
    for parameter, par_bounds in param_update_config.items():
        if parameter in optimize_pars:
            idx = np.argwhere(optimize_pars == parameter)[0][0]
            bounds[idx] = (par_bounds['min_val'], par_bounds['max_val'])

    result = minimize(
        misfit,
        x0,
        args=(emulators, targets, sds, fixed_indices, X_default_all, config),
        bounds=bounds,
        method='L-BFGS-B',
        options={'ftol': config['tol'], 'maxiter': config['maxiter']}
    )

    return result
    
def squared_z_loss(y_pred, target, stdev, y_var=None):
    z = np.abs((y_pred - target)/(stdev + 1e-8))
    return z**2

def implausibility_loss(y_pred, target, stdev, y_var):
    total_variance = y_var + stdev**2 + 1e-8 
    z = np.abs(y_pred - target) / np.sqrt(total_variance)
    return z

def default_penalty_l1(X, X_default):
    return np.sum(np.abs(X - X_default))

def barrier_penalty(X):
    return np.mean(1.0 / (X + 1e-6) + 1.0 / (1.0 - X + 1e-6))
    
def get_obs_mean_and_sd(df, variable):
    weighted_var = df['land_area']*df[variable]
    weighted_sd = df['land_area']*np.sqrt(df[f'{variable}_var'])
    total_land = df.land_area.sum()

    weighted_var_mean = weighted_var.sum()/total_land
    weighted_sd_mean = weighted_sd.sum()/total_land

    return weighted_var_mean, weighted_sd_mean

    
def implausibility_metric(pred, obs, pred_var, obs_var):

    top = np.abs(pred - obs)
    bottom = np.sqrt(pred_var + obs_var)
    return top/bottom

def calculate_implaus_sum(df, col_list):
    
    implausiblity_sum = df[col_list].sum(axis=1)
    
    return implausiblity_sum

def subset_sample(sample_dat, col_list, implaus_tol):
    
    for col in col_list:
        sample_dat = sample_dat.where(sample_dat[col] < implaus_tol)
    
    sample_sub = sample_dat.dropna()
    
    return sample_sub


def plot_parameter_hists(sample, min_max_pars):
    
    pars = [par for par in sample.columns if par != 'wave']
    
    plt.figure(figsize=[18, 16])
    for i, par in enumerate(pars):
        
        par_min_max = min_max_pars[min_max_pars.parameter == par]
        minpar = float(par_min_max[par_min_max.type == 'min']['val'].values[0])
        maxpar = float(par_min_max[par_min_max.type == 'max']['val'].values[0])
        defaultpar = float(par_min_max['default'].values[0])
        p = (defaultpar - minpar)/(maxpar - minpar)
        
        ax = plt.subplot(7, 5, i + 1)
        ax.hist(sample[par])
        ax.axvline(x=p, color='r', linestyle='-')
        ax.set_xlabel(par)
        ax.set_xlim(0, 1)
    plt.tight_layout()
    
def find_sensitive_parameters(sens_df, vars, sens_tol):
    
    sub_dat = sens_df[sens_df['var'].isin(vars)]
    sub_dat = sub_dat.where(sub_dat.S1 > sens_tol)
    sub_dat = sub_dat.dropna()
    
    sensitive_pars = sorted(np.unique(sub_dat.names))

    return sensitive_pars

def get_param_samples(sample_dir):
    
    files = sorted([os.path.join(sample_dir, file) for file in os.listdir(sample_dir) if file.endswith(".csv")])
    sample_dfs = []
    for file in files:
        df = pd.read_csv(file, index_col=[0])
        sample_dfs.append(df)
    sample_df = pd.concat(sample_dfs)
    sample_df = sample_df.drop(columns=['wave'])

    return sample_df

def find_best_parameter_sets(sample):
    best_sample_index = sample[['implaus_sum']].idxmin()
    best_sample = sample.loc[best_sample_index, :]
    return best_sample


def run_batch_optimization(emulators, targets, sds, fixed_indices, X_default_all, num_optimize, 
                           param_names, optimize_indices, config, 
                           param_update_config, num_batch=100):

    optimize_pars = param_names[optimize_indices]
    
    all_results = []
    for i in range(num_batch):
        result = run_optimization(emulators, targets, sds, fixed_indices, optimize_pars,
                                  X_default_all, num_optimize, config, param_update_config)
        df = pd.DataFrame({'parameter': param_names[optimize_indices],
                          'values': result['x']})
        df['batch'] = i
        all_results.append(df)
    all_out = pd.concat(all_results)
    return all_out.pivot(index='batch', columns='parameter', values='values')

def prep_calibration_data(obs, calibration_vars, obs_config, emulator_dir, pft_name, pft_id):
    
    # get observations for this pft
    obs_pft = obs[obs.pft == pft_name]
    
    targets = []
    sds = []
    emulators = []
    for variable in calibration_vars:
        
        # observations for this pft and variable
        obs_mean, obs_sd = get_obs_mean_and_sd(obs_pft, obs_config[variable]['var'])
        
        # convert to tf objects
        targets.append(obs_mean)
        sds.append(obs_sd)

        # load the emulator
        emulators.append(TFEmulator(emulator_dir, pft=pft_id, variable=variable))
        
    return emulators, targets, sds

def get_default_pft_values(default_norm, pft):
    default_pft = default_norm[default_norm.pft == pft]
    default_pft = default_pft.drop(columns=['pft'])
    return default_pft.to_numpy().flatten()

def get_calibrated_distribution(param_dir):
    param_files = [os.path.join(param_dir, f) for f in os.listdir(param_dir) if f.endswith('.csv')]
    dat_list = []
    for file in param_files:
        dat_list.append(pd.read_csv(file))
    dat = pd.concat(dat_list)
    dat = dat.drop(columns=['batch'])
    return dat