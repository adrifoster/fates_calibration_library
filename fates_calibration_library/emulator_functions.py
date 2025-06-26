"""Methods to assist with  model emulation"""

import os
import json
import gpflow
import pickle
from typing import List, Dict, Callable, Tuple
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import root_mean_squared_error, r2_score
from SALib.sample import fast_sampler
from SALib.analyze import fast
from SALib.sample import saltelli
from SALib.analyze import sobol
from matplotlib.patches import Rectangle

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
    Y_emulated, _ = emulator.predict_y(sample)
    
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
        oaat, v = emulator.predict_y(sample.values)
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

def plot_emulated_sample(pred_sampled, obs_mean, obs_sd, pft_id, var, units):
    
    plt.figure(figsize=(7, 5))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xticks(fontsize=11)

    my_hist, _ = np.histogram(pred_sampled, bins=40)
    maxv = my_hist.max()
    plt.xlabel(f"Emulated {pft_id} Annual {var} ({units})", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.hist(pred_sampled, fc="darkgray", bins=40)
    ax.add_patch(Rectangle((obs_mean - obs_sd, 0), 2*obs_sd, maxv,
                        facecolor='red', alpha=0.4))
    ax.axvline(x=obs_mean, ymin=0.0, ymax=maxv, color='r')

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
    y_pred, y_pred_var = model.predict_y(np.asarray(X_test))
    y_pred_sd = np.sqrt(y_pred_var.numpy().flatten())
    
    # ensure shapes are the same
    y_test = np.asarray(y_test).reshape(-1)
    
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
    y_train = np.asarray(y_train).reshape(-1, 1)
    
    # train GP model
    model = gpflow.models.GPR(data=(np.asarray(X_train), y_train), kernel=kernel,
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
    
@tf.function
def optimization_step_batch(X, emulator_array, targets, stdevs, X_default, optimizer,
                            config):
    
    # grab and check config values
    lambda_penalty = config.get('lambda_penalty', None)
    if lambda_penalty is not None:
        if lambda_penalty <= 0.0:
            raise ValueError("lambda_penalty must be > 0.")
    
    barrier_strength = config.get('barrier_strenth', 0.0)
    if barrier_strength < 0.0:
        raise ValueError("barrier_strength must be >= 0.")
    
    earlystop_pct = config.get('earlystop_pct', 75.0)
    if earlystop_pct <= 0.0:
        raise ValueError("earlystop_pct must be > 0.")
    
    # batch size
    batch_size = tf.shape(X)[0]  
    
    # tile default params to match batch size
    X_default_tiled = tf.tile(X_default, [batch_size, 1])

    with tf.GradientTape() as tape:
        
        per_model_losses = []
        
        # loop over each emulator/target/stddev
        for emulator, target, stddev in zip(emulator_array, targets, stdevs):
            
            # expand target and stdev to batch shape
            target_tiled = tf.tile(tf.reshape(target, (1, -1)), [batch_size, 1])
            stdev_tiled = tf.tile(tf.reshape(stddev, (1, -1)), [batch_size, 1])
            
            y_pred, y_var = emulator(X)
            
            # calculate loss
            z = config['loss_fn'](y_pred, target_tiled, stdev_tiled, y_var)
            loss_i = tf.reshape(z, [-1])
            per_model_losses.append(loss_i)
            
        combined_loss = tf.add_n(per_model_losses)
        
        if lambda_penalty is not None:
            penalty_per_sample = config['default_penalty_fn'](X, X_default_tiled)  # shape: [batch]
            default_penalty = tf.maximum(penalty_per_sample / lambda_penalty, 1.0)
        else:
            default_penalty = tf.ones_like(combined_loss)
            
        if barrier_strength > 0.0:
            # penalty for moving too close to bounds [0, 1]
            barrier = config['barrier_penalty_fn'](X)
            barrier_penalty = (1.0 + barrier_strength * barrier)
        else:
            barrier_penalty = 1.0

        penalized_loss = combined_loss * default_penalty * barrier_penalty
        
        total_loss = tf.reduce_mean(penalized_loss)
        max_z = tfp.stats.percentile(combined_loss, earlystop_pct, 
                                     interpolation='midpoint')
         
    grads = tape.gradient(total_loss, [X])
    optimizer.apply_gradients(zip(grads, [X]))
    X.assign(tf.clip_by_value(X, 0.0, 1.0))

    return total_loss, max_z, penalized_loss

def run_optimization(X: tf.Variable, emulator_array: List[Callable], targets: List[tf.Tensor], 
                     stdevs: List[tf.Tensor], x_default: tf.Tensor, config: Dict) -> Tuple[tf.Tensor, Dict[str, List[float]]]:
    """Run optmization loop with configurable parameters

    Args:
        X (tf.Variable): Optimizable parameter tensor
        emulator_array (List[Callable]): List of emulators returning (mean, variance)
        targets (List[tf.Tensor]): List of observation targets (1D tensors)
        stdevs (List[tf.Tensor]): List of observational standard deviations (1D tensors)
        x_default (tf.Tensor): Default parameter tensor
        config (Dict): Dictionary containing all config options.

    Returns:
        Tuple[tf.Tensor, Dict[str, List[float]]]:
            - x_opt: final optimized parameters (numpy array)
            - logs: Dictionary of loss histories
    """
    
    if 'checkpoint_dir' not in config:
        raise ValueError("Missing required config key: 'checkpoint_dir'")
    
    checkpoint_dir = config['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # learning rate schedule
    learning_rate = config.get('learning_rate', 1e-3)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=config.get('lr_decay_steps', 300),
        decay_rate=0.5,
        staircase=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # ensure optimizer tracks 'x'
    _ = optimizer.iterations # touch optimizer to ensure it is initialized
    optimizer.apply_gradients([(tf.zeros_like(X), X)]) 
        
    # history trackers
    logs = {
        'total_loss': [],
        'max_z': [],
        'losses': [],
    }
    
    for step in range(config.get('maxiter', 3000)):
        
        total_loss, max_z, losses = optimization_step_batch(X, emulator_array, targets, stdevs, x_default, optimizer, config)
        
        # log history
        logs['total_loss'].append(total_loss.numpy())
        logs['max_z'].append(max_z.numpy())
        logs['losses'].append(losses.numpy())
        
        # periodic printout
        if step % 10 == 0:
            tf.print(f"Step {step:03d}: total={total_loss:.6f} max_z={max_z:.6f}")
            
        # save checkpoints
        if step % config.get('checkpoint_n', 10) == 0:
            checkpoint = {
                'step': step,
                'params': X.numpy(),
                'loss': total_loss.numpy()
            }
            path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pkl')
            try:
                with open(path, 'wb') as f:
                    pickle.dump(checkpoint, f)
            except Exception as e:
                print(f"WARNING: Failed to save checkpoint at step {step}: {e}")
                
        # early stopping based on max implausibility
        if tf.reduce_max(max_z) <= config.get('epsilon', 0.5):
            print(f"Converged at step {step}")
            tf.print(f"Step {step:03d}: total={total_loss:.6f} max_z={max_z:.6f}")
            break

    return X.numpy(), logs
    
def squared_z_loss(y_pred, target, stdev, y_var=None):
    z = tf.abs((y_pred - target)/(stdev + 1e-8))
    return tf.reduce_sum(z**2, axis=1)

def implausibility_loss(y_pred, target, stdev, y_var):
    total_variance = y_var + stdev**2 + 1e-8 
    z = tf.abs(y_pred - target) / tf.sqrt(total_variance)
    return tf.reduce_sum(z**2, axis=1)

def default_penalty_l1(X, X_default):
    return tf.reduce_sum(tf.abs(X - X_default), axis=1)

def barrier_penalty(X):
    return tf.reduce_mean(1.0 / (X + 1e-6) + 1.0 / (1.0 - X + 1e-6))
    
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

    imp = top/bottom

    return imp

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
